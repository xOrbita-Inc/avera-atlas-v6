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

MU_EARTH = 398600.4418  # km^3/s^2


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
    Clohessy–Wiltshire Phi_rv block for circular reference orbit.
    Maps impulsive Δv (km/s) at burn time to Δr (km) at time dt later.
    """
    if a_km <= 0:
        raise ValueError("a_ref_km must be > 0")
    omega = math.sqrt(MU_EARTH / (a_km ** 3))
    c = math.cos(omega * dt_s)
    s = math.sin(omega * dt_s)
    return np.array(
        [
            [4 - 3 * c, 0.0, 0.0],
            [6 * (s - omega * dt_s), 1.0, 0.0],
            [0.0, 0.0, c],
        ],
        dtype=float,
    )


def mahalanobis_sq(r_km: np.ndarray, cov_km2: np.ndarray) -> float:
    """
    m^2 = r^T P^{-1} r for 3D relative position.
    """
    try:
        inv_cov = np.linalg.inv(cov_km2)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.inv(cov_km2 + 1e-12 * np.eye(3))
    return float(r_km.T @ inv_cov @ r_km)


# -----------------------------------------------------------------------------
# Policy + core evaluation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatorPolicy:
    lambda_v: float
    lambda_L: float
    dv_mag_limit_m_s: float
    a_ref_km: float = 7000.0  # optional override (not required by spec)


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
    )

    # Enforce constraint checks on t_burn_utc if present
    _validate_constraints(policy_raw, t_burn_utc)

    return conjunction_id, satellite, conjunction, policy_obj, policy_raw


def evaluate_conjunction(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single conjunction and return OpenAPI EvaluateResponse.
    """
    conjunction_id, sat, conj, policy, policy_raw = _validate_request(req)

    r_sat_km = _as_vec3(sat["r_sat_km"], "satellite.r_sat_km")
    v_sat_km_s = _as_vec3(sat["v_sat_km_s"], "satellite.v_sat_km_s")
    r_rel_km = _as_vec3(conj["r_rel_km"], "conjunction.r_rel_km")
    P_rel = _as_cov9(conj["p_rel_km2"], "conjunction.p_rel_km2")

    t_burn_utc = sat["t_burn_utc"]
    t_ca_utc = conj["t_ca_utc"]
    dt_to_ca_s = _dt_seconds(t_burn_utc, t_ca_utc)
    if dt_to_ca_s <= 0:
        raise ValueError("t_burn_utc must be earlier than t_ca_utc (dt_to_ca_s > 0 required)")

    hard = policy_raw.get("hard_constraints", {}) if isinstance(policy_raw.get("hard_constraints"), dict) else {}
    attitude_restricted = bool(hard.get("attitude_restricted", False))
    power_constrained = bool(hard.get("power_constrained", False))

    # If power_constrained, you can optionally downscale dv magnitude.
    # Keep deterministic + explicit:
    dv_mag_m_s = float(policy.dv_mag_limit_m_s)
    if power_constrained:
        dv_mag_m_s *= 0.5  # conservative; replace with proper thrust model later

    dv_mag_km_s = dv_mag_m_s / 1000.0

    # Candidate directions
    directions = _candidate_directions(r_sat_km, v_sat_km_s, attitude_restricted)

    # CW mapping
    phi_rv = cw_phi_rv(policy.a_ref_km, dt_to_ca_s)

    # Mahalanobis baseline
    m2_pre = mahalanobis_sq(r_rel_km, P_rel)

    v_remaining = float(sat["v_remaining_m_s"])
    lifetime_penalty = dv_mag_m_s / max(1e-6, v_remaining)

    all_candidates: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for name, d_hat in directions:
        dv_vec_km_s = d_hat * dv_mag_km_s
        delta_r_km = phi_rv @ dv_vec_km_s
        r_post_km = r_rel_km - delta_r_km

        m2_post = mahalanobis_sq(r_post_km, P_rel)
        delta_C = m2_pre - m2_post

        U = delta_C - policy.lambda_v * dv_mag_m_s - policy.lambda_L * lifetime_penalty

        all_candidates.append(
            {
                "direction": name,
                "dv_eci_km_s": dv_vec_km_s.tolist(),
                "delta_C": float(delta_C),
                "utility": float(U),
            }
        )

        if best is None or U > best["utility"]:
            best = {
                "direction": name,
                "dv_eci_km_s": dv_vec_km_s.tolist(),
                "dv_magnitude_m_s": float(dv_mag_m_s),
                "t_burn_utc": t_burn_utc,
                "utility": float(U),
                "_m2_post": float(m2_post),
                "_delta_C": float(delta_C),
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