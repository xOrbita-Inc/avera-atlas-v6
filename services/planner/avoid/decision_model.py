# avoid/decision_model.py
"""
APS Planner – Avoidance Decision Model (v2.4)

OpenAPI endpoints implemented by this module:
  - POST /v1/evaluate        -> evaluate_conjunction(req)
  - POST /v1/evaluate/batch  -> evaluate_batch(req)

This file is intended to be "drop-in" for a service container wrapper (e.g., FastAPI),
but also supports a CLI for local testing.

Conventions (per OpenAPI spec):
- Frame: ECI
- Positions: km
- Velocities: km/s
- Δv magnitudes: m/s
- Δv vector: km/s
- Times: UTC, ISO-8601
- Covariance: km² (3×3 symmetric, row-major flat array length 9)

Core math: Clohessy–Wiltshire Phi_rv + Mahalanobis confidence-gain utility.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aps_math import conventions, frames
from aps_math.pc_utils import compute_pc


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_utc(s: str) -> datetime:
    """
    Parse ISO-8601 timestamp, accepting 'Z' suffix.
    """
    if not isinstance(s, str):
        raise ValueError("timestamp must be a string")
    s2 = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s2)


def _dt_seconds(t_burn_utc: str, t_ca_utc: str) -> float:
    tb = _parse_iso_utc(t_burn_utc)
    tc = _parse_iso_utc(t_ca_utc)
    return (tc - tb).total_seconds()


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _as_vec3(x: Any, name: str) -> np.ndarray:
    if not isinstance(x, list) or len(x) != 3:
        raise ValueError(f"{name} must be a 3-element array")
    v = np.asarray(x, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"{name} must be a 3-element array")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must contain finite numbers")
    return v


def _as_cov9(x: Any, name: str) -> np.ndarray:
    if not isinstance(x, list) or len(x) != 9:
        raise ValueError(f"{name} must be a 9-element row-major flat array")
    c = np.asarray(x, dtype=float)
    if c.shape != (9,):
        raise ValueError(f"{name} must be a 9-element row-major flat array")
    if not np.all(np.isfinite(c)):
        raise ValueError(f"{name} must contain finite numbers")
    return c.reshape((3, 3))


def _require(obj: Dict[str, Any], key: str, where: str) -> Any:
    if key not in obj:
        raise ValueError(f"Missing required field '{where}.{key}'" if where else f"Missing required field '{key}'")
    return obj[key]


def _unit(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= 0.0 or not math.isfinite(n):
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vec / n


def _interval_overlaps(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
    """
    True if intervals [a0,a1] and [b0,b1] overlap (inclusive).
    """
    return not (a1 < b0 or b1 < a0)


def _validate_constraints(policy_raw: Dict[str, Any], t_burn_utc: str) -> None:
    """
    Validate burn-time constraints from OperatorPolicy.
    Enforces:
      - burn_window (earliest/latest) if present
      - hard_constraints.no_burn_windows if present
    """
    tb = _parse_iso_utc(t_burn_utc)

    burn_window = policy_raw.get("burn_window")
    if isinstance(burn_window, dict):
        earliest = burn_window.get("earliest_utc")
        latest = burn_window.get("latest_utc")
        if earliest is not None:
            te = _parse_iso_utc(earliest)
            if tb < te:
                raise ValueError("t_burn_utc violates burn_window.earliest_utc")
        if latest is not None:
            tl = _parse_iso_utc(latest)
            if tb > tl:
                raise ValueError("t_burn_utc violates burn_window.latest_utc")

    hard = policy_raw.get("hard_constraints")
    if isinstance(hard, dict):
        no_burn_windows = hard.get("no_burn_windows")
        if isinstance(no_burn_windows, list):
            # treat burn as an instant; forbid if tb is inside any window
            for i, w in enumerate(no_burn_windows):
                if not isinstance(w, dict):
                    continue
                s = w.get("start_utc")
                e = w.get("end_utc")
                if s is None or e is None:
                    continue
                ts = _parse_iso_utc(s)
                te = _parse_iso_utc(e)
                if ts <= tb <= te:
                    raise ValueError(f"t_burn_utc falls inside hard_constraints.no_burn_windows[{i}]")


# -----------------------------------------------------------------------------
# CW + Mahalanobis
# -----------------------------------------------------------------------------

def cw_phi_rv(a_km: float, dt_s: float) -> np.ndarray:
    """
    Clohessy–Wiltshire Phi_rv block for a circular reference orbit.
    Maps impulsive Δv (km/s) at burn time to Δr (km) at time dt later.

    SCRUM-386, SCRUM-378. Thin wrapper around aps_math.frames.cw_phi_full's
    rv block. One closed-form implementation now, not two: see that
    function's docstring for the full derivation, the other three blocks
    (Phi_rr, Phi_vr, Phi_vv) the SCRUM-378 observability Gramian needs and
    this function alone does not provide, and why the consolidation
    happened here rather than duplicating Phi_rv a second time.

    This previously implemented Phi_rv directly. SCRUM-386 fixed a defect
    where it returned Phi_rr, the position-to-position block, instead,
    which is dimensionless, so applying it to a Δv in km/s produced a
    number in km/s that was then read as km. The error was large, not
    marginal: a 2 m/s along-track burn 4 hours before TCA at a = 6928 km
    came out as 2 metres of separation change where the correct block
    gives 86.8 km. services/planner/tests/test_cw_phi_rv.py still guards
    this block's behaviour directly against an independent closed form,
    unchanged by this refactor apart from where it imports MU_EARTH from.

    Reference: Clohessy & Wiltshire (1960); Vallado, Fundamentals of
    Astrodynamics and Applications, §6.7.
    """
    return frames.cw_phi_full(a_km, dt_s)[0:3, 3:6]


def mahalanobis_sq(r_km: np.ndarray, cov_km2: np.ndarray) -> float:
    """
    m^2 = r^T P^{-1} r for 3D relative position.
    """
    try:
        inv_cov = np.linalg.inv(cov_km2)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.inv(cov_km2 + 1e-12 * np.eye(3))
    return float(r_km.T @ inv_cov @ r_km)


def pc_from_geometry(
    r_sat_km: np.ndarray,
    v_sat_km_s: np.ndarray,
    r_rel_km: np.ndarray,
    v_rel_km_s: Optional[np.ndarray],
    p_rel_km2: np.ndarray,
    hbr_m: float,
) -> Optional[float]:
    """Probability of collision from conjunction geometry, or None.

    SCRUM-387. The same computation maneuver_scorer performs, written here
    because maneuver_scorer imports from this module and the reverse would be a
    cycle. Both call the same compute_pc in aps_math, so the two paths cannot
    disagree about the physics, only about which of them is asked.

    The covariance is SPLIT in half between the two objects rather than put
    entirely on the primary. compute_pc routes to frisbee_max_pc, an upper
    bound, whenever either covariance is all-zero, and the bound falls off as
    1/k rather than exp(-k^2/2). Halves sum to the same combined matrix and take
    the ordinary path. Same trap SCRUM-389 documented, same fix.
    """
    if v_rel_km_s is None:
        return None
    try:
        r1 = np.asarray(r_sat_km, dtype=float) * 1000.0
        v1 = np.asarray(v_sat_km_s, dtype=float) * 1000.0
        r2 = r1 + np.asarray(r_rel_km, dtype=float) * 1000.0
        v2 = v1 + np.asarray(v_rel_km_s, dtype=float) * 1000.0
        half_m2 = np.asarray(p_rel_km2, dtype=float) * 1.0e6 * 0.5
        pc = float(compute_pc(r1, v1, half_m2, r2, v2, half_m2, hbr_m).Pc)
    except Exception:
        return None
    return pc if math.isfinite(pc) else None


def risk_exchange_rate(
    lambda_dv: float, max_dv_m_s: float, pc_maneuver_threshold: float
) -> float:
    """Value of one unit of collision probability, in the cost terms' units.

    SCRUM-387. Lives here rather than in maneuver_scorer because both scoring
    paths need it and maneuver_scorer already imports from this module, so this
    is the direction that does not create a cycle. The v2.5 wrapper reads the
    three numbers off an OperatorPolicy and calls this.

    At exactly the maneuver threshold, spending the whole per-event delta-v
    budget to eliminate the risk breaks even. Every input is a number the
    operator already set for another reason, so nothing here is fitted to the
    demo fixtures, which is what AC3 requires.
    """
    return lambda_dv * max_dv_m_s / pc_maneuver_threshold


def compute_q_exec_km2(
    direction_hat: np.ndarray,
    dv_mag_km_s: float,
    thrust_misalignment_deg: Optional[float],
    dv_magnitude_sigma: Optional[float],
    phi_rv: np.ndarray,
) -> np.ndarray:
    """
    SCRUM-365: execution-error position covariance contribution (Q_exec),
    from thrust misalignment and burn-magnitude uncertainty.

    Physics
    -------
    Two independent error sources on the actual (vs. commanded) delta-v:
      - Magnitude error: along the commanded direction, 1-sigma =
        dv_magnitude_sigma (a dimensionless fraction) * dv_mag_km_s.
      - Pointing error: perpendicular to the commanded direction.
        thrust_misalignment_deg is treated as a per-axis 1-sigma, not a
        total cone-angle sigma -- each of the two perpendicular axes
        independently gets 1-sigma = dv_mag_km_s * sin(thrust_misalignment_deg)
        (isotropic in the plane perpendicular to the burn, absent a
        preferred clocking angle for the misalignment -- a standard,
        defensible assumption with no more specific information). Total
        perpendicular variance is therefore 2 * (dv_mag_km_s *
        sin(thrust_misalignment_deg))^2 across both axes combined, not
        split/halved between them. This matches the P_burn formula in
        gnc_interface.yaml's ExecutionError block.

    This gives a 3x3 velocity-error covariance in a local frame aligned
    with the commanded burn direction, which is then propagated to a
    position-error covariance using the SAME linear map (phi_rv) already
    used elsewhere in this module to convert a commanded delta-v into a
    post-maneuver position change (Cov(A x) = A Cov(x) A^T).

    Frame contract (SCRUM-397)
    --------------------------
    direction_hat and phi_rv must be in the SAME frame. This function does
    not rotate anything. It builds the velocity covariance in whatever
    frame direction_hat is given in and propagates it through whatever map
    phi_rv is, so the caller owns the frame and gets what it asks for.

    Both callers pass an ECI direction and an ECI-expressed map, built by
    conjugating cw_phi_rv's RTN block through aps_math.frames.rotate_cw_block.
    That is correct because M_eci Q_eci M_eci^T expands to
    rot (M_rtn Q_rtn M_rtn^T) rot^T, which is the RTN answer rotated into ECI.

    What was here before, and why it is worth reading
    ------------------------------------------------
    This docstring used to carry a note saying phi_rv is RTN-ordered, that the
    caller applies it directly to an ECI delta-v, that this is "only exact when
    the satellite's local R/T/N axes happen to align with the ECI axes", and
    that fully resolving it was out of scope, to be flagged "for a future ticket
    if the team wants it fixed".

    The defect was correctly identified and the ticket was never raised. It sat
    for a sprint, and it is not a small approximation: on a 53 degree inclined
    orbit the post-burn displacement was wrong by 54 km for a prograde burn and
    by a factor of 543 for a cross-track one. The reasoning for deferring was
    sound in itself, that fixing the covariance while leaving the position wrong
    would be worse than leaving both wrong, and SCRUM-397 fixes both together as
    that note asked.

    The lesson worth keeping is about where a finding lives. A defect recorded
    only in a docstring is a defect nobody is accountable for. If you find one
    while doing something else, raise it.

    Parameters
    ----------
    direction_hat : np.ndarray
        Unit vector of the commanded burn direction, same frame as the
        caller's dv_vec_km_s (currently ECI, per the note above).
    dv_mag_km_s : float
        Commanded delta-v magnitude [km/s].
    thrust_misalignment_deg : float or None
        1-sigma pointing error [deg]. None treated as 0 (no contribution).
    dv_magnitude_sigma : float or None
        1-sigma fractional magnitude uncertainty (dimensionless). None
        treated as 0 (no contribution).
    phi_rv : np.ndarray
        3x3 CW Phi_rv block (from cw_phi_rv), same one used for the
        nominal delta_r_km calculation for this candidate.

    Returns
    -------
    np.ndarray
        3x3 position-error covariance [km^2], same frame as delta_r_km.
        All-zero matrix if both parameters are None (preserves existing
        perfect-burn behavior exactly -- AC1).
    """
    if thrust_misalignment_deg is None and dv_magnitude_sigma is None:
        return np.zeros((3, 3))

    misalignment_rad = (
        math.radians(thrust_misalignment_deg) if thrust_misalignment_deg is not None else 0.0
    )
    sigma_mag_frac = dv_magnitude_sigma if dv_magnitude_sigma is not None else 0.0

    u = direction_hat / np.linalg.norm(direction_hat)
    # Any vector not parallel to u, to build a perpendicular basis.
    seed = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p1 = np.cross(u, seed)
    p1 = p1 / np.linalg.norm(p1)
    p2 = np.cross(u, p1)

    sigma_along_km_s = sigma_mag_frac * dv_mag_km_s
    sigma_perp_km_s = dv_mag_km_s * math.sin(misalignment_rad)

    # Velocity-error covariance in the {u, p1, p2} local burn frame.
    q_v_local = np.diag([sigma_along_km_s ** 2, sigma_perp_km_s ** 2, sigma_perp_km_s ** 2])
    basis = np.column_stack([u, p1, p2])
    q_v = basis @ q_v_local @ basis.T

    # Propagate velocity-error covariance to position-error covariance,
    # using the same phi_rv already applied to the nominal delta-v.
    q_exec_km2 = phi_rv @ q_v @ phi_rv.T
    return q_exec_km2


# -----------------------------------------------------------------------------
# Policy + core evaluation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatorPolicy:
    lambda_v: float
    lambda_L: float
    dv_mag_limit_m_s: float
    a_ref_km: float = 7000.0  # optional override (not required by spec)

    # SCRUM-387. Needed to derive the risk exchange rate on this path the same
    # way the v2.5 path derives it. Defaults to the same 1e-4 the v2.5 operator
    # policy defaults to, so a v2.4 caller that says nothing gets the same trade
    # a v2.5 caller would.
    pc_maneuver_threshold: float = 1.0e-4


def _candidate_directions(
    r_sat_km: np.ndarray,
    v_sat_km_s: np.ndarray,
    attitude_restricted: bool,
) -> List[Tuple[str, np.ndarray]]:
    """
    OpenAPI direction enum: [prograde, radial, cross-track]
    """
    prograde = ("prograde", _unit(v_sat_km_s))
    if attitude_restricted:
        return [prograde]
    radial = ("radial", _unit(r_sat_km))
    cross = ("cross-track", _unit(np.cross(r_sat_km, v_sat_km_s)))
    return [prograde, radial, cross]


def _validate_request(req: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any], OperatorPolicy, Dict[str, Any]]:
    """
    Validate EvaluateRequest, returning:
      (conjunction_id, satellite, conjunction, policy_obj, policy_raw)
    """
    if not isinstance(req, dict):
        raise ValueError("request must be a JSON object")

    conjunction_id = _require(req, "conjunction_id", "")
    satellite = _require(req, "satellite", "")
    conjunction = _require(req, "conjunction", "")
    policy_raw = _require(req, "policy", "")

    if not isinstance(conjunction_id, str) or not conjunction_id:
        raise ValueError("conjunction_id must be a non-empty string")
    if not isinstance(satellite, dict):
        raise ValueError("satellite must be an object")
    if not isinstance(conjunction, dict):
        raise ValueError("conjunction must be an object")
    if not isinstance(policy_raw, dict):
        raise ValueError("policy must be an object")

    # SatelliteState required
    sat_id = _require(satellite, "sat_id", "satellite")
    r_sat_km = _require(satellite, "r_sat_km", "satellite")
    v_sat_km_s = _require(satellite, "v_sat_km_s", "satellite")
    t_burn_utc = _require(satellite, "t_burn_utc", "satellite")
    v_remaining_m_s = _require(satellite, "v_remaining_m_s", "satellite")

    if not isinstance(sat_id, str) or not sat_id:
        raise ValueError("satellite.sat_id must be a non-empty string")

    _ = _as_vec3(r_sat_km, "satellite.r_sat_km")
    _ = _as_vec3(v_sat_km_s, "satellite.v_sat_km_s")
    _parse_iso_utc(t_burn_utc)

    if not isinstance(v_remaining_m_s, (int, float)) or not math.isfinite(float(v_remaining_m_s)) or float(v_remaining_m_s) <= 0:
        raise ValueError("satellite.v_remaining_m_s must be a positive number")

    # ConjunctionState required
    obj_id = _require(conjunction, "obj_id", "conjunction")
    t_ca_utc = _require(conjunction, "t_ca_utc", "conjunction")
    r_rel_km = _require(conjunction, "r_rel_km", "conjunction")
    p_rel_km2 = _require(conjunction, "p_rel_km2", "conjunction")

    if not isinstance(obj_id, str) or not obj_id:
        raise ValueError("conjunction.obj_id must be a non-empty string")

    _parse_iso_utc(t_ca_utc)
    _ = _as_vec3(r_rel_km, "conjunction.r_rel_km")
    _ = _as_cov9(p_rel_km2, "conjunction.p_rel_km2")

    pc_pre = conjunction.get("pc_precomputed", None)
    if pc_pre is not None:
        if not isinstance(pc_pre, (int, float)) or not math.isfinite(float(pc_pre)) or float(pc_pre) < 0:
            raise ValueError("conjunction.pc_precomputed must be null or a non-negative number")

    # OperatorPolicy required
    lambda_v = _require(policy_raw, "lambda_v", "policy")
    lambda_L = _require(policy_raw, "lambda_L", "policy")
    dv_mag_limit = _require(policy_raw, "dv_mag_limit_m_s", "policy")

    lambda_v_f = float(lambda_v)
    lambda_L_f = float(lambda_L)
    dv_mag_limit_f = float(dv_mag_limit)

    if not math.isfinite(lambda_v_f) or lambda_v_f < 0:
        raise ValueError("policy.lambda_v must be a non-negative number")
    if not math.isfinite(lambda_L_f) or lambda_L_f < 0:
        raise ValueError("policy.lambda_L must be a non-negative number")
    if not math.isfinite(dv_mag_limit_f) or dv_mag_limit_f <= 0:
        raise ValueError("policy.dv_mag_limit_m_s must be a positive number")

    a_ref_km = float(policy_raw.get("a_ref_km", 7000.0))
    if not math.isfinite(a_ref_km) or a_ref_km <= 0:
        raise ValueError("policy.a_ref_km must be a positive number if provided")

    policy_obj = OperatorPolicy(
        lambda_v=lambda_v_f,
        lambda_L=lambda_L_f,
        dv_mag_limit_m_s=dv_mag_limit_f,
        a_ref_km=a_ref_km,
        # SCRUM-387
        pc_maneuver_threshold=float(
            policy_raw.get("pc_maneuver_threshold", 1.0e-4)
        ),
    )

    # Enforce constraint checks on t_burn_utc if present
    _validate_constraints(policy_raw, t_burn_utc)

    return conjunction_id, satellite, conjunction, policy_obj, policy_raw



# -----------------------------------------------------------------------------
# Thrust model for power_constrained path
# -----------------------------------------------------------------------------

_G0_M_S2: float = 9.80665  # standard gravity [m/s^2]


def _power_constrained_dv_m_s(
    propulsion_raw: dict,
    mass_kg: float,
    dv_limit_m_s: float,
    dt_to_ca_s: float,
) -> float:
    """
    Compute achievable delta-v for a power-constrained satellite.

    Reads propulsion parameters from the satellite propulsion sub-dict
    (satellite.propulsion in the OpenAPI request), consistent with
    PropulsionProfile in satellite_capability.py.

    Physics:
        F   = 2 * eta * P_W / (Isp * g0)   [N]
        dv  = F * t_burn / mass_kg           [m/s]

    Result is capped at dv_limit_m_s. Falls back to 0.5 * dv_limit_m_s
    when any required field is absent or invalid, preserving
    backward-compatible behavior for callers that have not supplied
    power propulsion parameters.

    Parameters
    ----------
    propulsion_raw : dict
        satellite.propulsion sub-dict. Expected fields:
        power_available_w, thruster_efficiency, isp_s, burn_window_s.
    mass_kg : float
        Satellite wet mass [kg] from satellite.lifetime.mass_kg.
    dv_limit_m_s : float
        Policy delta-v ceiling [m/s].
    dt_to_ca_s : float
        Time to closest approach [s]; used as burn window when
        propulsion_raw does not supply burn_window_s.
    """
    P_W    = propulsion_raw.get("power_available_w")
    eta    = propulsion_raw.get("thruster_efficiency")
    isp    = float(propulsion_raw.get("isp_s", 220.0))
    t_burn = propulsion_raw.get("burn_window_s", dt_to_ca_s)

    if P_W is None or eta is None:
        return dv_limit_m_s * 0.5

    P_W    = float(P_W)
    eta    = float(eta)
    t_burn = float(t_burn)

    if P_W <= 0 or eta <= 0 or eta > 1.0 or isp <= 0 or mass_kg <= 0 or t_burn <= 0:
        return dv_limit_m_s * 0.5

    thrust_n      = 2.0 * eta * P_W / (isp * _G0_M_S2)
    dv_achievable = thrust_n * t_burn / mass_kg
    return float(min(dv_achievable, dv_limit_m_s))


def evaluate_conjunction(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single conjunction and return OpenAPI EvaluateResponse.
    """
    conjunction_id, sat, conj, policy, policy_raw = _validate_request(req)

    r_sat_km = _as_vec3(sat["r_sat_km"], "satellite.r_sat_km")
    v_sat_km_s = _as_vec3(sat["v_sat_km_s"], "satellite.v_sat_km_s")
    r_rel_km = _as_vec3(conj["r_rel_km"], "conjunction.r_rel_km")
    P_rel = _as_cov9(conj["p_rel_km2"], "conjunction.p_rel_km2")

    # SCRUM-387 / SCRUM-398: optional relative velocity, same field and same
    # meaning as on the v2.5 path. Present means a Pc can be computed and the
    # utility can trade in probability. Absent means it cannot, and this path
    # falls back to the Mahalanobis gain exactly as it always did.
    _v_rel_raw = conj.get("v_rel_km_s")
    v_rel_km_s = (
        _as_vec3(_v_rel_raw, "conjunction.v_rel_km_s")
        if _v_rel_raw is not None
        else None
    )

    t_burn_utc = sat["t_burn_utc"]
    t_ca_utc = conj["t_ca_utc"]
    dt_to_ca_s = _dt_seconds(t_burn_utc, t_ca_utc)
    if dt_to_ca_s <= 0:
        raise ValueError("t_burn_utc must be earlier than t_ca_utc (dt_to_ca_s > 0 required)")

    hard = policy_raw.get("hard_constraints", {}) if isinstance(policy_raw.get("hard_constraints"), dict) else {}
    attitude_restricted = bool(hard.get("attitude_restricted", False))
    power_constrained = bool(hard.get("power_constrained", False))

    dv_mag_m_s = float(policy.dv_mag_limit_m_s)
    if power_constrained:
        _prop_raw = sat.get("propulsion", {}) if isinstance(sat.get("propulsion"), dict) else {}
        _lt_raw   = sat.get("lifetime", {}) if isinstance(sat.get("lifetime"), dict) else {}
        _mass_kg  = float(_lt_raw.get("mass_kg", sat.get("mass_kg", 100.0)))
        dv_mag_m_s = _power_constrained_dv_m_s(
            propulsion_raw=_prop_raw,
            mass_kg=_mass_kg,
            dv_limit_m_s=dv_mag_m_s,
            dt_to_ca_s=dt_to_ca_s,
        )

    dv_mag_km_s = dv_mag_m_s / 1000.0

    # Candidate directions
    directions = _candidate_directions(r_sat_km, v_sat_km_s, attitude_restricted)

    # CW mapping
    #
    # SCRUM-397. Same defect and same fix as maneuver_scorer.py. cw_phi_rv is
    # RTN-ordered, the candidate directions and r_rel_km are ECI, and nothing
    # rotated between them. Conjugating the block by the RTN-to-ECI rotation
    # gives a map that takes ECI in and returns ECI out.
    rtn_to_eci = frames.rtn_to_eci_rotation(r_sat_km, v_sat_km_s)
    phi_rv = frames.rotate_cw_block(
        cw_phi_rv(policy.a_ref_km, dt_to_ca_s), rtn_to_eci
    )

    # Mahalanobis baseline
    m2_pre = mahalanobis_sq(r_rel_km, P_rel)

    v_remaining = float(sat["v_remaining_m_s"])
    lifetime_penalty = dv_mag_m_s / max(1e-6, v_remaining)

    all_candidates: List[Dict[str, Any]] = []

    # No-burn baseline: delta_C=0, dv=0, U=0.
    # Wins whenever all burn directions produce negative utility,
    # making "no burn optimal" cases scientifically defensible
    # without requiring policy weight tuning per event.
    no_burn = {
        "direction": "no-burn",
        "dv_eci_km_s": [0.0, 0.0, 0.0],
        "delta_C": 0.0,
        "utility": 0.0,
    }
    all_candidates.append(no_burn)
    best: Optional[Dict[str, Any]] = {
        "direction": "no-burn",
        "dv_eci_km_s": [0.0, 0.0, 0.0],
        "dv_magnitude_m_s": 0.0,
        "t_burn_utc": t_burn_utc,
        "utility": 0.0,
        "_m2_post": float(m2_pre),
        "_delta_C": 0.0,
    }

    # SCRUM-387: this path's own Pc, so the utility can trade in probability
    # rather than in log-risk. AC8 asks for the two scoring paths to be
    # consistent or for the difference to be recorded; this is the consistent
    # option. Same aps_math.compute_pc, same hard-body radius convention, same
    # exchange rate formula, same fallback when no relative velocity is supplied.
    #
    # This path still applies no risk screen of its own. That difference is
    # deliberate and predates this ticket: /v1/evaluate/batch is a backward
    # compatibility contract with its own callers, and SCRUM-389 AC5 recorded
    # the decision not to add one. Scoring consistently and screening
    # differently is the intended state, not an oversight.
    hbr_m, _hbr_source = conventions.combined_hbr_m(
        conventions.DEFAULT_PRIMARY_RADIUS_M
    )
    pc_pre_geometric = pc_from_geometry(
        r_sat_km, v_sat_km_s, r_rel_km, v_rel_km_s, P_rel, hbr_m
    )
    # pc_precomputed is read again further down for the response; read it here
    # too rather than moving that line, so the response assembly is untouched.
    _pc_supplied = conj.get("pc_precomputed", None)
    pc_pre_level = (
        float(_pc_supplied) if _pc_supplied is not None else pc_pre_geometric
    )
    lambda_risk = risk_exchange_rate(
        policy.lambda_v, policy.dv_mag_limit_m_s, policy.pc_maneuver_threshold
    )

    for name, d_hat in directions:
        dv_vec_km_s = d_hat * dv_mag_km_s
        delta_r_km = phi_rv @ dv_vec_km_s
        r_post_km = r_rel_km - delta_r_km

        m2_post = mahalanobis_sq(r_post_km, P_rel)

        pc_post = pc_from_geometry(
            r_sat_km, v_sat_km_s, r_post_km, v_rel_km_s, P_rel, hbr_m
        )

        # SCRUM-386. Two quantities, deliberately kept apart.
        #
        # delta_C is the v2.4 OUTPUT convention (m2_pre - m2_post) and is
        # reported unchanged, so the response schema is untouched.
        #
        # The utility must use the research-doc convention, APS_2_5_Research
        # section 5.4, delta_C = m2_post - m2_pre. Mahalanobis distance is
        # separation measured in sigma, so larger post-maneuver is safer and
        # is what a burn should be rewarded for. This line previously fed the
        # v2.4 quantity straight into U, which paid the planner for reducing
        # separation, that is for maneuvering toward the secondary.
        #
        # It was invisible until now because cw_phi_rv returned the Phi_rr
        # block, so a 1 m/s burn moved the relative position by about a
        # millimetre and m2_post was numerically indistinguishable from
        # m2_pre. With the corrected Phi_rv the displacement is kilometres,
        # the sign dominates, and every burn direction scored negative,
        # leaving no-burn optimal for every conjunction on this path.
        #
        # maneuver_scorer.py has always used the research convention. This
        # brings the v2.4 path, which still serves POST /v1/evaluate/batch,
        # into agreement with it.
        delta_C_v24 = m2_pre - m2_post          # reported (unchanged)
        confidence_gain = m2_post - m2_pre      # scored (section 5.4)

        # SCRUM-387. Trade in Pc when one exists, fall back to the Mahalanobis
        # gain when it does not. The benefit is the FRACTION of risk the burn
        # removes in our model applied to the authoritative level, so a supplied
        # Pc that disagrees with our geometry does not get subtracted from a
        # number computed on ours. When the Pc was computed rather than supplied
        # the two are the same and this collapses to the difference.
        if (
            pc_pre_level is not None
            and pc_post is not None
            and pc_pre_geometric is not None
            and pc_pre_geometric > 0.0
        ):
            reduction_fraction = max(0.0, 1.0 - pc_post / pc_pre_geometric)
            risk_benefit = lambda_risk * pc_pre_level * reduction_fraction
        else:
            risk_benefit = confidence_gain

        U = risk_benefit - policy.lambda_v * dv_mag_m_s - policy.lambda_L * lifetime_penalty

        all_candidates.append(
            {
                "direction": name,
                "dv_eci_km_s": dv_vec_km_s.tolist(),
                "delta_C": float(delta_C_v24),
                "utility": float(U),
            }
        )

        if U > best["utility"]:
            best = {
                "direction": name,
                "dv_eci_km_s": dv_vec_km_s.tolist(),
                "dv_magnitude_m_s": float(dv_mag_m_s),
                "t_burn_utc": t_burn_utc,
                "utility": float(U),
                "_m2_post": float(m2_post),
                "_delta_C": float(delta_C_v24),
            }

    assert best is not None

    # Post-maneuver risk surrogate: Pc if available else 1/m2_post
    pc_pre = conj.get("pc_precomputed", None)
    if pc_pre is not None:
        risk_surrogate_post = float(pc_pre)
    else:
        m2_post_best = float(best["_m2_post"])
        risk_surrogate_post = float(1.0 / max(1e-12, m2_post_best))

    resp = {
        "conjunction_id": conjunction_id,
        "recommendation": {
            "direction": best["direction"],
            "dv_eci_km_s": best["dv_eci_km_s"],
            "dv_magnitude_m_s": best["dv_magnitude_m_s"],
            "t_burn_utc": best["t_burn_utc"],
            "utility": best["utility"],
        },
        "metrics": {
            "delta_C": float(best["_delta_C"]),
            "m2_pre": float(m2_pre),
            "m2_post": float(best["_m2_post"]),
            "fuel_cost_m_s": float(dv_mag_m_s),
            "lifetime_penalty": float(lifetime_penalty),
            "risk_surrogate_post": float(risk_surrogate_post),
            "all_candidates": all_candidates,
        },
        "evaluated_at": _now_iso(),
    }
    return resp


def evaluate_batch(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate multiple conjunctions.
    Input:  { "conjunctions": [EvaluateRequest, ...] }
    Output: { "results": [EvaluateResponse, ...], "evaluated_at": "..." }
    """
    if not isinstance(req, dict):
        raise ValueError("request must be a JSON object")
    if "conjunctions" not in req:
        raise ValueError("Missing required field 'conjunctions'")
    items = req["conjunctions"]
    if not isinstance(items, list) or len(items) < 1:
        raise ValueError("'conjunctions' must be a non-empty array")

    results = [evaluate_conjunction(item) for item in items]
    results.sort(key=lambda r: float(r["recommendation"]["utility"]), reverse=True)

    return {"results": results, "evaluated_at": _now_iso()}


# -----------------------------------------------------------------------------
# ErrorResponse helper
# -----------------------------------------------------------------------------

def error_response(msg: str, detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"error": msg, "detail": detail or {}}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _read_json(path: str) -> Dict[str, Any]:
    if path == "-" or path.strip() == "":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    if path == "-" or path.strip() == "":
        sys.stdout.write(json.dumps(obj, indent=2) + "\n")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="APS v2.4 decision model (OpenAPI-compatible)")
    ap.add_argument("--mode", choices=["evaluate", "batch"], default="evaluate")
    ap.add_argument("--in", dest="in_path", default="-", help="Input JSON path, or '-' for stdin")
    ap.add_argument("--out", dest="out_path", default="-", help="Output JSON path, or '-' for stdout")

    args = ap.parse_args()

    try:
        req = _read_json(args.in_path)
        if args.mode == "evaluate":
            resp = evaluate_conjunction(req)
        else:
            resp = evaluate_batch(req)
        _write_json(args.out_path, resp)
    except Exception as e:
        # For CLI usage, emit ErrorResponse-shaped JSON and exit nonzero.
        _write_json(args.out_path, error_response(str(e)))
        raise SystemExit(1)


if __name__ == "__main__":
    main()