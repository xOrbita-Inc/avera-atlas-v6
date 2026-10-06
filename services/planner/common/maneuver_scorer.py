"""
APS 2.5 — Step 9.4: Mission-Aware Maneuver Scoring
====================================================
Module: services/planner/common/maneuver_scorer.py

Implements the APS 2.5 objective function, candidate generation,
feasibility filtering, and co-optimized avoidance + return-to-slot
scoring.  This is the core algorithmic work of APS 2.5.

Depends on:
  - 9.1  common/satellite_capability.py   (SatelliteCapability, LifetimeProfile)
  - 9.2  common/operator_policy.py        (OperatorPolicy, ScoringWeights)
  - 9.3  common/constellation_geometry.py (WalkerDeltaGeometry, SlotRecoveryPlan,
                                           _return_burn_cost_m_s,
                                           _mean_motion_to_sma_km)
  - 2.4  avoid/decision_model.py          (cw_phi_rv, mahalanobis_sq — imported,
                                           NOT modified)

Integration contract with decision_model.py (v2.4, LOCKED)
-----------------------------------------------------------
decision_model.evaluate_conjunction() is not modified.  9.4 is a new
entry point: evaluate_conjunction_v25().  It accepts a v2.5 request dict
(satellite dict carries SatelliteCapability fields; policy dict carries
OperatorPolicy fields) and returns a response that is a strict superset
of the v2.4 EvaluateResponse schema.  All v2.4 output fields are
preserved with identical names and semantics.

Sign convention note
--------------------
decision_model.py (v2.4) computes:
    delta_C = m2_pre - m2_post   (positive = safer, pre > post)

APS_2_5_Research_V3.ipynb §5.4 defines:
    ΔC(a) = m²_post − m²_pre    (positive = safer, post > pre)

These express the same physical quantity with opposite signs.
Internally this module uses the RESEARCH DOC convention
(m2_post - m2_pre) in the utility function so that §5.4 math applies
directly.  The output field `delta_C` is reported in the V2.4 convention
(m2_pre - m2_post) for backward compatibility.  Both are documented on
every object that carries them.

CTO acceptance criteria addressed in this module
-------------------------------------------------
1. Return burn formula (§2):
   dv_return = dv_avoid exactly on a circular orbit.
   return_ratio = 1.0 for constellated satellites.
   Implemented via _return_burn_cost_m_s() imported from 9.3.

2. Atmospheric drag in return cost model (§6.2):
   delta_a_total(t) = delta_a_burn + a_dot_drag * t
   Applied for recovery windows > 2 orbits at altitudes < 550 km.
   Drag rate default: -50 m/day at 482 km (§6.2 value).
   When drag correction is applied, dv_return > dv_avoid.

3. lifetime_fraction_used stub (§1.4 / SCRUM-280):
   LifetimeProfile.lifetime_fraction_used returns 0.0 as a stub.
   compute_lifetime_fraction() resolves the real value using
   OperatorPolicy.mission_lifetime_days_total.
   Scoring function uses this in the lambda_L term.

Scientific gaps carried forward (APS_2_5_Research_V3.ipynb §6)
---------------------------------------------------------------
  - J2 in return cost: slot target drifts during recovery (§6.1).
    SlotRecoveryPlan.target_raan_deg is J2-corrected (9.3).
    The delta_a calculation here does not yet include J2-induced
    delta_a variation. APS 3.0 scope.
  - Covariance propagation: P_rel is consumed as a static snapshot.
    Dilution region flagged via covariance_quality field. APS 3.0 scope.
  - Maneuver execution errors: modelled when the satellite's propulsion
    profile carries thrust_misalignment_deg and/or dv_magnitude_sigma
    (SCRUM-365, Q_exec added to S before m2_post). m2_post is optimistic
    only when the profile does not specify either parameter.

References
----------
APS_2_5_Research_V3.ipynb §2, §5.4, §6.2.
Vallado, D.A. (2013). Fundamentals of Astrodynamics, 4e.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# SCRUM-389: the planner computes its own Pc. compute_pc and the shared
# conventions both live in libs/aps_math, reachable since SCRUM-388.
from aps_math.pc_utils import compute_pc
from aps_math import conventions, frames

# ---------------------------------------------------------------------------
# Imports from locked modules (do not modify those files)
# ---------------------------------------------------------------------------

from common.satellite_capability import (
    SatelliteCapability,
    LifetimeProfile,
    ConstellationSlot,
)
from common.operator_policy import OperatorPolicy
from common.constellation_geometry import (
    WalkerDeltaGeometry,
    SlotRecoveryPlan,
    _return_burn_cost_m_s,
    _mean_motion_to_sma_km,
    _circular_velocity_km_s,
    _mean_motion_rad_s,
    _TWO_PI,
    _SEC_PER_DAY,
)
from avoid.decision_model import (
    cw_phi_rv,
    mahalanobis_sq,
    compute_q_exec_km2,
    _as_vec3,
    _as_cov9,
    _parse_iso_utc,
    _dt_seconds,
    _unit,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_MU_KM3_S2: float       = 3.986004418e5  # km³/s²
_MEAN_EARTH_RADIUS_KM: float = 6371.0  # rounded spherical mean Earth radius [km]
_DRAG_RATE_M_PER_DAY: float = -50.0      # SMA decay at 482 km [m/day] (§6.2)
_DRAG_THRESHOLD_ALT_KM: float = 550.0    # Apply drag correction below this [km]
_DRAG_ORBIT_THRESHOLD: float  = 2.0      # Apply drag correction above this [orbits]


# ---------------------------------------------------------------------------
# CTO Item 3: lifetime_fraction_used resolver (SCRUM-280)
# ---------------------------------------------------------------------------

def compute_lifetime_fraction(
    lifetime: LifetimeProfile,
    mission_lifetime_days_total: Optional[float],
) -> float:
    """Compute the real lifetime_fraction_used, unlocking SCRUM-280.

    LifetimeProfile.lifetime_fraction_used is a stub returning 0.0 because
    it lacks mission_lifetime_days_total, which lives on OperatorPolicy.
    This function bridges the two.

    Parameters
    ----------
    lifetime : LifetimeProfile
        From SatelliteCapability.lifetime.
    mission_lifetime_days_total : float or None
        From OperatorPolicy.mission_lifetime_days_total.
        If None, returns 0.0 (preserves v2.4 behaviour).

    Returns
    -------
    float
        Fraction of mission lifetime consumed: 0.0 (fresh) to 1.0 (EOL).
        Clamped to [0.0, 1.0].

    Notes
    -----
    Formula: f_life = 1 - (days_remaining / days_total)
    At mission start: days_remaining ≈ days_total -> f_life ≈ 0
    At end of life:   days_remaining ≈ 0           -> f_life ≈ 1
    """
    if mission_lifetime_days_total is None or mission_lifetime_days_total <= 0:
        return 0.0
    fraction = 1.0 - (
        lifetime.mission_lifetime_days_remaining / mission_lifetime_days_total
    )
    return max(0.0, min(1.0, fraction))


# ---------------------------------------------------------------------------
# CTO Item 2: drag-corrected return cost
# ---------------------------------------------------------------------------

def _drag_gate_altitude_from_sma_km(sma_km: float) -> float:
    """Return the coarse orbital-altitude proxy used by the drag gate.

    This intentionally retains the existing SMA-minus-Earth-radius model.
    The 6371 km radius is a rounded spherical mean-Earth approximation, not
    WGS-84 geodetic altitude. WGS-84 surface radius spans approximately
    6356.752-6378.137 km, so this approximation can differ from local surface
    altitude by at most about 14.248 km.
    """
    return float(sma_km) - _MEAN_EARTH_RADIUS_KM


def _drag_corrected_dv_return_m_s(
    dv_avoid_m_s: float,
    sma_km: float,
    recovery_orbits: float,
    altitude_km: float,
    drag_rate_m_per_day: float = _DRAG_RATE_M_PER_DAY,
) -> float:
    """Return burn cost [m/s] with atmospheric drag correction (§6.2).

    For recovery windows longer than 2 orbits at altitudes below 550 km,
    atmospheric drag decays the SMA during the recovery window, adding a
    phase error on top of the burn-induced delta_a:

        delta_a_total(t) = delta_a_burn + a_dot_drag * t

    where a_dot_drag ≈ -50 m/day at 482 km (§6.2).

    The return burn must cancel delta_a_total, so:

        dv_return_drag = n * |delta_a_total| / 2

    For recovery windows <= 2 orbits or altitudes >= 550 km, the pure
    §2 formula is used (dv_return = dv_avoid exactly).

    Parameters
    ----------
    dv_avoid_m_s : float
        Avoidance burn magnitude [m/s].
    sma_km : float
        Reference semi-major axis [km].
    recovery_orbits : float
        Number of recovery orbits planned.
    altitude_km : float
        Orbital altitude [km] (metadata label from WalkerDeltaGeometry).
        Used only to decide whether drag correction applies.
    drag_rate_m_per_day : float
        SMA drag rate [m/day].  Default -50 m/day at 482 km (§6.2).

    Returns
    -------
    float
        Drag-corrected return burn magnitude [m/s].
        >= _return_burn_cost_m_s(dv_avoid_m_s, sma_km) when drag applies.

    Notes
    -----
    J2-induced delta_a variation during recovery is not included.
    That is a documented open gap (APS_2_5_Research_V3.ipynb §6.1).
    """
    if dv_avoid_m_s <= 0:
        return 0.0

    # Pure §2 baseline
    dv_return_base = _return_burn_cost_m_s(dv_avoid_m_s, sma_km)

    # Drag correction: only for long recovery windows at low altitude
    apply_drag = (
        altitude_km < _DRAG_THRESHOLD_ALT_KM
        and recovery_orbits > _DRAG_ORBIT_THRESHOLD
    )
    if not apply_drag:
        return dv_return_base

    # delta_a from avoidance burn (Gauss VE, tangential impulse)
    vc_km_s = _circular_velocity_km_s(sma_km)
    delta_a_burn_km = 2.0 * sma_km * (dv_avoid_m_s / 1000.0) / vc_km_s

    # Drag SMA decay during recovery window
    t_orbit_s = _TWO_PI / _mean_motion_rad_s(sma_km)
    recovery_time_days = (recovery_orbits * t_orbit_s) / _SEC_PER_DAY
    delta_a_drag_km = abs(drag_rate_m_per_day) * recovery_time_days / 1000.0

    # Total delta_a to cancel
    delta_a_total_km = delta_a_burn_km + delta_a_drag_km

    # Two-impulse return cost for the corrected delta_a
    n_rad_s = _mean_motion_rad_s(sma_km)
    dv_return_drag = (n_rad_s * delta_a_total_km / 2.0) * 1000.0  # -> m/s

    return dv_return_drag


# ---------------------------------------------------------------------------
# Along-track displacement from CW dynamics
# ---------------------------------------------------------------------------

def _along_track_displacement_km(
    dv_avoid_m_s: float,
    sma_km: float,
    dt_to_ca_s: float,
) -> float:
    """Along-track separation at TCA from a tangential avoidance burn.

    From CW dynamics (APS_2_5_Research_V3.ipynb §2):

        delta_y_TCA ≈ 3π * N_p * (dv / n)

    where N_p is the number of half-periods before TCA.

    Parameters
    ----------
    dv_avoid_m_s : float
        Avoidance burn magnitude [m/s].
    sma_km : float
        Reference semi-major axis [km].
    dt_to_ca_s : float
        Time from burn to TCA [seconds].

    Returns
    -------
    float
        Along-track displacement [km].
    """
    n_rad_s = _mean_motion_rad_s(sma_km)
    T_half = math.pi / n_rad_s                     # half-period [s]
    n_half_periods = dt_to_ca_s / T_half
    dv_km_s = dv_avoid_m_s / 1000.0
    return abs(3.0 * math.pi * n_half_periods * dv_km_s / n_rad_s)


# ---------------------------------------------------------------------------
# Covariance quality classification
# ---------------------------------------------------------------------------

def _covariance_quality(m2_pre: float) -> str:
    """Classify covariance quality from pre-maneuver Mahalanobis distance.

    Hejduk / NASA CARA dilution region: m² < 1 means the object is
    inside the 1-sigma ellipsoid.  This is a flag for potential
    covariance dilution — the Pc may be underestimated.

    Returns
    -------
    str
        'dilution_region' | 'degraded' | 'good'
    """
    if m2_pre < 1.0:
        return "dilution_region"
    if m2_pre < 4.0:
        return "degraded"
    return "good"


# ---------------------------------------------------------------------------
# Feasibility filters
# ---------------------------------------------------------------------------

def _passes_feasibility(
    cap: SatelliteCapability,
    policy: OperatorPolicy,
    dv_mag_m_s: float,
    pc: Optional[float],
    pc_source: str,
    miss_distance_km: Optional[float],
    mahalanobis: float,
) -> Tuple[bool, str, str, str]:
    """Apply mode-specific risk gates and propulsion feasibility checks.

    Returns (passes, reason_code, human_readable, risk_gate).

    APS records risk_gate="utility" and bypasses the fixed Pc and Mahalanobis
    pre-screen gates so candidates can be scored. The final recommendation
    requires utility > 0; passing this function alone is not a recommendation.
    Propulsion feasibility checks still apply.

    Flight-rule modes retain the existing pre-scoring path: supplied or
    computed Pc uses the selected baseline and the miss-distance floor.
    Mahalanobis is the legacy fallback when no Pc can be established.
    The result records which path ran. Utility provenance separately identifies
    whether scoring used Pc or a surrogate.

    These recommendation paths do not change authorization or safety arming.
    """
    # 1. Recommendation path: APS utility, or the legacy flight-rule gate.
    if policy.decision_mode == "aps":
        # SCRUM-479: compute utility before deciding; retain propulsion checks.
        risk_gate = "utility"
    elif pc is not None:
        risk_gate = "pc"
        if not policy.is_recommendation_required(pc, miss_distance_km or 999.0):
            if policy.is_monitor_only(pc):
                return (
                    False,
                    "pc_below_threshold",
                    f"Pc {pc:.2e} ({pc_source}) is below maneuver threshold "
                    f"{policy.flight_rule_pc_threshold:.1e} but above monitor "
                    f"threshold {policy.pc_monitor_threshold:.1e}. "
                    f"Event escalated to watch status. No burn required.",
                    risk_gate,
                )
            return (
                False,
                "pc_below_threshold",
                f"Pc {pc:.2e} ({pc_source}) is below maneuver threshold "
                f"{policy.flight_rule_pc_threshold:.1e}. No maneuver warranted.",
                risk_gate,
            )
    else:
        risk_gate = "mahalanobis"
        if not policy.passes_pre_screen(mahalanobis):
            return (
                False,
                "trivial_event",
                f"Mahalanobis distance {mahalanobis:.2f} exceeds screen threshold "
                f"{policy.mahalanobis_screen_threshold:.1f}, and no Pc could be "
                f"established for this event. Outside the risk-relevant region. "
                f"No maneuver warranted.",
                risk_gate,
            )

    # 2. Propulsion infeasible
    effective_dv = cap.effective_dv_limit_m_s(dv_mag_m_s)
    if effective_dv <= cap.propulsion.min_dv_m_s:
        return (
            False,
            "propulsion_infeasible",
            f"Effective delta-v {effective_dv:.4f} m/s is at or below minimum "
            f"executable burn {cap.propulsion.min_dv_m_s:.4f} m/s. "
            f"Satellite cannot execute a meaningful maneuver.",
            risk_gate,
        )

    return True, "", "", risk_gate



# ---------------------------------------------------------------------------
# SCRUM-389: probability of collision
# ---------------------------------------------------------------------------

PC_SOURCE_SUPPLIED = "supplied"
PC_SOURCE_COMPUTED = "computed"
PC_SOURCE_UNAVAILABLE = "unavailable"


def resolve_hard_body_radius(
    cap: SatelliteCapability,
    secondary_radius_m: Optional[float] = None,
) -> Tuple[float, str]:
    """Combined hard-body radius for this conjunction, and where it came from.

    Delegates to the shared convention so the planner and the propagator cannot
    disagree about the radius for the same event. See ADR-010 and the reasoning
    beside the constants in libs/aps_math/conventions.py.

    SCRUM-489: a conjunction that states the secondary's real radius is scored
    on the real combined radius rather than the 15 m screening floor. The floor
    was never calibrated against the 1e-4 threshold (SCRUM-394) and inflated
    every live Pc by roughly 17x; it now applies only when the secondary's size
    is genuinely unknown. cap.radius_m is the primary's own radius, which on the
    live path is the figure the CDM states for our asset.
    """
    return conventions.combined_hbr_m(cap.radius_m, secondary_radius_m)


def compute_pc_from_geometry(
    r_sat_km: np.ndarray,
    v_sat_km_s: np.ndarray,
    r_rel_km: np.ndarray,
    v_rel_km_s: np.ndarray,
    p_rel_km2: np.ndarray,
    hbr_m: float,
) -> float:
    """Probability of collision from the planner's own conjunction geometry.

    Why relative velocity is required
    ---------------------------------
    The 2D Pc is an integral over the encounter plane, which is the plane
    perpendicular to the relative velocity at TCA. Without a relative velocity
    there is no plane to project the miss and the covariance into, so there is no
    Pc to compute. This is why v_rel_km_s is not optional here even though it is
    optional on the request: a caller that cannot supply it gets no Pc rather
    than an invented one.

    Why the combined covariance is SPLIT rather than put on the primary
    -------------------------------------------------------------------
    compute_pc takes two objects and adds their covariances, so the obvious move
    is to pass the combined matrix as the primary's and zero as the secondary's.
    That is wrong, and silently so.

    compute_pc checks whether either covariance is all-zero and, if so, routes to
    frisbee_max_pc, which returns an UPPER BOUND on Pc for the case where one
    object's covariance is unknown. It is not the same quantity. The returned
    PcResult looks identical apart from is_remediated, and the bound falls off as
    1/k in Mahalanobis distance rather than exp(-k^2/2), so it agrees closely
    with the real Pc for a small miss and is four orders of magnitude high by
    four sigma. Splitting the covariance in half between the two objects sums to
    exactly the same combined matrix and takes the ordinary path.

    test_the_split_matches_a_hand_split_at_every_miss pins this across misses
    rather than at one geometry, because at one small-miss geometry the wrong
    version passes.
    """
    r1 = np.asarray(r_sat_km, dtype=float) * 1000.0
    v1 = np.asarray(v_sat_km_s, dtype=float) * 1000.0
    r2 = r1 + np.asarray(r_rel_km, dtype=float) * 1000.0
    v2 = v1 + np.asarray(v_rel_km_s, dtype=float) * 1000.0
    # Half each, summing to the combined matrix. See the docstring: a zero
    # covariance on either object silently selects a different method.
    half_m2 = np.asarray(p_rel_km2, dtype=float) * 1.0e6 * 0.5  # km^2 -> m^2
    result = compute_pc(r1, v1, half_m2, r2, v2, half_m2, hbr_m)
    return float(result.Pc)


def resolve_pc(
    cap: SatelliteCapability,
    r_sat_km: np.ndarray,
    v_sat_km_s: np.ndarray,
    r_rel_km: np.ndarray,
    v_rel_km_s: Optional[np.ndarray],
    p_rel_km2: np.ndarray,
    pc_precomputed: Optional[float],
    secondary_radius_m: Optional[float] = None,
) -> Tuple[Optional[float], str, float, str]:
    """Decide this event's Pc and record where it came from.

    Returns (pc, pc_source, hbr_m, hbr_source). pc is None when no Pc could be
    established, which is not the same as a Pc of zero and must not be treated
    as one.

    Precedence, per SCRUM-389 AC3: an externally supplied Pc wins. A CDM or UDL
    Pc is produced by an authority with more information than we have, and
    overriding it with our own would be arrogant. We compute one only when
    nobody has.
    """
    hbr_m, hbr_source = resolve_hard_body_radius(cap, secondary_radius_m)

    if pc_precomputed is not None:
        return float(pc_precomputed), PC_SOURCE_SUPPLIED, hbr_m, hbr_source

    if v_rel_km_s is None:
        return None, PC_SOURCE_UNAVAILABLE, hbr_m, hbr_source

    try:
        pc = compute_pc_from_geometry(
            r_sat_km, v_sat_km_s, r_rel_km, v_rel_km_s, p_rel_km2, hbr_m
        )
    except Exception:
        # A degenerate geometry (zero relative velocity, singular covariance)
        # yields no Pc. Reporting unavailable is correct; reporting zero would
        # claim the event is safe.
        return None, PC_SOURCE_UNAVAILABLE, hbr_m, hbr_source

    if not math.isfinite(pc):
        return None, PC_SOURCE_UNAVAILABLE, hbr_m, hbr_source
    return pc, PC_SOURCE_COMPUTED, hbr_m, hbr_source


# ---------------------------------------------------------------------------
# SCRUM-393: post-maneuver probability of collision
# ---------------------------------------------------------------------------

# Where risk_surrogate_post's value came from. The field has carried three
# different quantities over its life and the name says none of them, so the
# source travels with the number. Same reasoning as pc_source and hbr_source.
RISK_SURROGATE_PC_POST = "pc_post"
RISK_SURROGATE_INVERSE_M2 = "inverse_m2_post"
RISK_SURROGATE_PC_PRE_NO_BURN = "pc_pre_no_burn"


# ---------------------------------------------------------------------------
# SCRUM-387: what a unit of collision probability is worth in delta-v
# ---------------------------------------------------------------------------

# Which quantity the risk term of the utility was computed from.
UTILITY_BASIS_PC = "pc_traded"
UTILITY_BASIS_DELTA_C = "delta_c_legacy"


# ---------------------------------------------------------------------------
# SCRUM-485: what APS concluded, when it did not conclude "maneuver"
# ---------------------------------------------------------------------------
#
# APS used to express this as a single bool, and that bool was true for any
# positive utility -- including the unbounded utilities the delta_c_legacy
# fallback produces for events whose Pc is exactly zero. Those events carry no
# collision risk; APS was recommending a capped burn on them anyway.
#
# Three outcomes, because the two non-maneuver ones are not the same claim and
# collapsing them would say something false about one of them:
#
#   maneuver_required : a usable, nonzero Pc was traded down by a burn whose
#                       utility is positive.
#   not_required      : the event WAS assessed and is safe. Its Pc was computed
#                       and is effectively zero -- a negligible-risk in-volume
#                       pass whose geometric Pc underflows. Zero burn.
#   no_usable_pc      : no Pc could be established, or none that the utility
#                       model could trade on. The event is NOT claimed safe and
#                       is NOT a maneuver; it is unassessed and says so. Zero
#                       burn, because a burn priced off the fallback basis is
#                       not a burn anyone should fly.
APS_STATE_REQUIRED = "maneuver_required"
APS_STATE_NOT_REQUIRED = "not_required"
APS_STATE_NO_USABLE_PC = "no_usable_pc"


def aps_pc_usable(pc: Optional[float]) -> bool:
    """A probability of collision APS may recommend a maneuver against.

    Usable means established and not zero. Both exclusions matter and for
    different reasons:

      pc is None  -- nothing was established. There is no risk estimate to act
                     on, and the fallback utility is not one.
      pc == 0.0   -- a real estimate, and it says there is no risk. Spending
                     delta-v against it is spending it against nothing.

    Deliberately NOT keyed on the utility basis, although the fallback basis is
    where the huge utilities come from and the two coincide on every event in
    the reported window. They come apart when a Pc is supplied externally while
    the geometric Pc underflows: that event takes the fallback basis but it has
    a real, authority-supplied probability attached, and it can be above the
    maneuver threshold. Refusing a burn there would suppress a maneuver on a
    genuinely risky event -- and it would not even stop the burn, because the
    mode machine authorizes off the Pc threshold and would still execute, while
    the recommendation beside it said no-burn. A gate that makes the response
    disagree with what the system actually does is worse than no gate.

    So the basis is not the test; the Pc is. What remains unfixed for that
    supplied-Pc case is the SIZE of the burn, which the unbounded fallback still
    chooses and still caps at the delta-v limit. That is a sizing defect on an
    event that does warrant some burn, which is a different problem from
    commanding one on an event that warrants none.
    """
    return pc is not None and pc > 0.0


def aps_decision_state(pc: Optional[float], required: bool) -> str:
    """Which of the three outcomes above this event reached under APS.

    `required` is the policy verdict, already gated on a usable Pc, so this only
    has to separate the two non-maneuver cases -- and the separation is simply
    whether a probability was ever established.

    "Assessed and safe" is claimed only when a Pc actually exists. An absent one
    reports no_usable_pc, because labelling it not_required would tell an
    operator the event had been assessed and cleared when it had not been
    assessed at all. That is the one error here that could cost something.
    """
    if required:
        return APS_STATE_REQUIRED
    if pc is None:
        return APS_STATE_NO_USABLE_PC
    return APS_STATE_NOT_REQUIRED


def risk_exchange_rate(policy: OperatorPolicy) -> float:
    """Value of one unit of collision probability, in the same units as the cost
    terms. SCRUM-387.

        lambda_risk = lambda_dv * max_dv_per_event_ms / pc_maneuver_threshold

    Read it as a statement the operator has already made. At exactly the
    maneuver threshold, spending the entire per-event delta-v budget to
    eliminate the risk breaks even. Above the threshold a maneuver is worth
    paying for, and below it the Pc gate has already rejected the event, so the
    utility never sees it.

    Why derived rather than fitted
    ------------------------------
    Every number here is one the operator already set for another reason.
    Nothing is tuned against the four synthetic demo events, which is what
    SCRUM-387 AC3 requires and what the old weights could not claim: those were
    fitted to fixtures that were themselves built against physics corrected
    twice since.

    It also means the utility and the maneuver gate cannot drift apart. Raise
    pc_maneuver_threshold and the exchange rate falls in step, so the events the
    gate now lets through are still the ones the utility thinks are worth acting
    on. Under a fitted constant those two would have to be kept in sync by hand,
    and nothing would fail if they were not.
    """
    return (
        policy.scoring_weights.lambda_dv
        * policy.max_dv_per_event_ms
        / policy.pc_maneuver_threshold
    )


def compute_pc_post(
    r_sat_km: np.ndarray,
    v_sat_km_s: np.ndarray,
    r_post_km: np.ndarray,
    v_rel_km_s: Optional[np.ndarray],
    s_covariance_km2: np.ndarray,
    hbr_m: float,
) -> Optional[float]:
    """Probability of collision AFTER a candidate burn.

    Same computation as the pre-maneuver Pc, on the post-burn geometry. The
    inputs that change are the relative position, which the burn moves, and the
    covariance, which gains Q_exec when the propulsion profile carries execution
    error. Returns None when no Pc can be established, which is not a Pc of zero
    and must not be read as one.

    Why the relative velocity is held fixed
    ---------------------------------------
    The burn changes the primary's velocity, so strictly the relative velocity
    at TCA changes too, which rotates the encounter plane. That is neglected
    here, and the size of what is neglected is bounded rather than assumed.

    Under Clohessy-Wiltshire the velocity-to-velocity block is

        Phi_vv = [[ cos(nt),     2 sin(nt),      0       ],
                  [ -2 sin(nt),  4 cos(nt) - 3,  0       ],
                  [ 0,           0,              cos(nt) ]]

    whose entries are bounded by 1, 2 and 7 regardless of how long the
    propagation runs, so the change in relative velocity is at most about
    7.3 times the burn magnitude. At the policy ceiling of a few m/s against a
    LEO relative velocity of 7 to 15 km/s that is a plane rotation under
    2e-3 rad, which moves the projected miss by under 2 m per km of separation.
    Against a 15 m hard-body radius that is not a difference the answer can see.

    test_holding_v_rel_fixed_is_below_the_stated_bound implements Phi_vv
    independently and measures the difference rather than trusting this
    paragraph. If a future change raises the delta-v ceiling far enough for the
    approximation to matter, that test is what fails.

    Including Phi_vv here would be false precision anyway. The post-burn
    position itself comes from a linearised CW propagation, and that is a larger
    modelling error than the plane rotation it would correct.
    """
    if v_rel_km_s is None:
        return None
    try:
        pc = compute_pc_from_geometry(
            r_sat_km, v_sat_km_s, r_post_km, v_rel_km_s, s_covariance_km2, hbr_m
        )
    except Exception:
        return None
    if not math.isfinite(pc):
        return None
    return float(pc)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _candidate_directions_v25(
    r_sat_km: np.ndarray,
    v_sat_km_s: np.ndarray,
    cap: SatelliteCapability,
) -> List[Tuple[str, np.ndarray]]:
    """Generate candidate burn direction unit vectors.

    Directions: prograde, retrograde, radial, anti-radial, cross-track,
    anti-cross-track — subject to attitude constraints.

    When attitude_restricted=True, only prograde is available (v2.4
    behaviour preserved).

    Returns list of (name, unit_vector) tuples.
    """
    prograde  = _unit(v_sat_km_s)
    radial    = _unit(r_sat_km)
    cross     = _unit(np.cross(r_sat_km, v_sat_km_s))

    if cap.cadence.attitude_restricted:
        return [("prograde", prograde)]

    return [
        ("prograde",       prograde),
        ("retrograde",    -prograde),
        ("radial",         radial),
        ("anti-radial",   -radial),
        ("cross-track",    cross),
        ("anti-cross",    -cross),
    ]


# ---------------------------------------------------------------------------
# Core scoring dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CandidateScore:
    """Scoring result for a single candidate maneuver direction.

    All cost terms are in the same units (m/s-equivalent or dimensionless
    ratios) so the utility is a dimensionally consistent weighted sum.

    Attributes
    ----------
    direction : str
        Burn direction label.
    dv_eci_km_s : list[float]
        Burn vector in ECI frame [km/s].
    dv_avoid_m_s : float
        Avoidance burn magnitude [m/s].
    dv_return_m_s : float
        Return burn cost [m/s].  Equals dv_avoid on circular orbit (§2).
        Drag-corrected for windows > 2 orbits below 550 km (§6.2).
    dv_total_m_s : float
        Total mission cost: dv_avoid + dv_return [m/s].
    delta_C_v25 : float
        Mahalanobis gain: m2_post - m2_pre (RESEARCH DOC convention).
        Positive = satellite moved further from threat = safer.
    delta_C_v24 : float
        Same quantity in V2.4 convention: m2_pre - m2_post.
        Reported in output `delta_C` field for backward compatibility.
    m2_post : float
        Post-maneuver Mahalanobis distance squared.
    dv_cost_term : float
        lambda_v * dv_total_m_s  (penalty term).
    lifetime_cost_term : float
        lambda_L * (dv_total_m_s / v_available_m_s)  (penalty term).
    slot_cost_term : float
        lambda_s * (post_drift_km / acceptable_drift_km).
        Zero for non-constellated satellites.
    utility : float
        U = delta_C_v25 - dv_cost_term - lifetime_cost_term - slot_cost_term.
    post_drift_km : float
        Along-track drift from slot centre post-avoidance [km].
    recovery_plan : SlotRecoveryPlan or None
        Full recovery plan.  None for non-constellated satellites.
    drag_correction_applied : bool
        True if drag term was added to dv_return.
    """
    direction: str
    dv_eci_km_s: List[float]
    dv_avoid_m_s: float
    dv_return_m_s: float
    dv_total_m_s: float
    delta_C_v25: float
    delta_C_v24: float
    m2_post: float
    dv_cost_term: float
    lifetime_cost_term: float
    slot_cost_term: float
    utility: float
    post_drift_km: float
    recovery_plan: Optional[SlotRecoveryPlan]
    drag_correction_applied: bool

    # SCRUM-393. Probability of collision if THIS candidate is executed, or None
    # when no Pc could be established for the event. Computed for every
    # candidate rather than only the winner, because SCRUM-387 needs to trade in
    # Pc across the whole candidate set and reading it here beats recomputing it
    # there.
    pc_post: Optional[float] = None

    # SCRUM-387. Which quantity the risk term of this candidate's utility came
    # from, pc_traded or delta_c_legacy. The two are not comparable, so a
    # utility is only meaningful next to the basis that produced it.
    utility_basis: str = ""


@dataclass
class ManeuverScoringResult:
    """Complete APS 2.5 scoring result for one conjunction event.

    Preserves all v2.4 output fields unchanged, adds v2.5 fields.

    V2.4 preserved fields (identical names and semantics)
    -----------------------------------------------------
    direction, dv_eci_km_s, dv_magnitude_m_s, t_burn_utc,
    utility, delta_C, m2_pre, m2_post, fuel_cost_m_s,
    lifetime_penalty, risk_surrogate_post, all_candidates.

    V2.5 additions
    --------------
    dv_return_m_s, dv_total_m_s, lifetime_fraction_used,
    slot_cost_term, post_drift_km, recovery_plan,
    covariance_quality, no_go_reason_code, no_go_human_readable,
    drag_correction_applied, candidates_v25.
    """
    conjunction_id: str

    # --- V2.4 preserved ---
    direction: str
    dv_eci_km_s: List[float]
    dv_magnitude_m_s: float
    t_burn_utc: str
    utility: float
    delta_C: float          # V2.4 convention: m2_pre - m2_post
    m2_pre: float
    m2_post: float
    fuel_cost_m_s: float
    lifetime_penalty: float
    risk_surrogate_post: float
    all_candidates: List[Dict[str, Any]]

    # --- V2.5 additions ---
    dv_return_m_s: float
    dv_total_m_s: float
    lifetime_fraction_used: float
    slot_cost_term: float
    post_drift_km: float
    recovery_plan: Optional[SlotRecoveryPlan]
    covariance_quality: str         # 'good' | 'degraded' | 'dilution_region'
    no_go_reason_code: str          # '' if maneuver recommended
    no_go_human_readable: str       # '' if maneuver recommended
    drag_correction_applied: bool
    candidates_v25: List[CandidateScore]
    evaluated_at: str
    execution_error_modelled: bool = False

    # SCRUM-389 provenance. A number is only trustworthy if you can tell where
    # it came from, which is the covariance_source pattern from ADR-008.
    pc_pre: Optional[float] = None
    pc_source: str = PC_SOURCE_UNAVAILABLE
    hbr_m: float = 0.0
    hbr_source: str = ""
    risk_gate: str = ""

    # SCRUM-393. The genuine post-maneuver Pc for the recommended burn, and what
    # risk_surrogate_post is actually carrying. pc_post is None when no Pc could
    # be established, and on a no-go, where there is no burn to compute one for.
    pc_post: Optional[float] = None
    risk_surrogate_source: str = ""

    # SCRUM-387. pc_traded when the utility bought down a real probability,
    # delta_c_legacy when it fell back to the Mahalanobis gain because no Pc
    # could be established. Recorded rather than inferred, because the two
    # utilities are on different scales and comparing them is meaningless.
    utility_basis: str = ""

    def is_maneuver_recommended(self) -> bool:
        return self.direction != "no-burn"

    def operator_summary(self) -> str:
        """Single-line summary answering the operator question."""
        if self.is_maneuver_recommended():
            slot_note = ""
            if self.recovery_plan and self.recovery_plan.recovery_required:
                slot_note = (
                    f" | Slot recovery: {self.recovery_plan.recovery_orbits:.0f} orbit(s), "
                    f"dv_total={self.dv_total_m_s:.3f} m/s"
                )
            return (
                f"MANEUVER RECOMMENDED | {self.direction} "
                f"dv={self.dv_magnitude_m_s:.3f} m/s | "
                f"delta_C={self.delta_C:.2f} | "
                f"U={self.utility:.3f} | "
                f"cov={self.covariance_quality}"
                f"{slot_note}"
            )
        return f"NO ACTION | {self.no_go_human_readable}"

    def to_v24_response(self) -> Dict[str, Any]:
        """Return a dict matching the v2.4 EvaluateResponse schema exactly."""
        return {
            "conjunction_id": self.conjunction_id,
            "recommendation": {
                "direction":        self.direction,
                "dv_eci_km_s":      self.dv_eci_km_s,
                "dv_magnitude_m_s": self.dv_magnitude_m_s,
                "t_burn_utc":       self.t_burn_utc,
                "utility":          self.utility,
            },
            "metrics": {
                "delta_C":             self.delta_C,
                "m2_pre":              self.m2_pre,
                "m2_post":             self.m2_post,
                "fuel_cost_m_s":       self.fuel_cost_m_s,
                "lifetime_penalty":    self.lifetime_penalty,
                "risk_surrogate_post": self.risk_surrogate_post,
                "all_candidates":      self.all_candidates,
            },
            "evaluated_at": self.evaluated_at,
        }


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def _execution_error_modelled(cap: SatelliteCapability) -> bool:
    """SCRUM-365: whether this satellite's propulsion profile carries
    either execution-error parameter. A property of the satellite's
    configuration, not of whether any particular event needed a burn --
    used identically for both go and no-go results so a satellite with
    these parameters set is never misreported as not modelling
    execution error just because this event didn't require a maneuver.
    """
    return (
        cap.propulsion.thrust_misalignment_deg is not None
        or cap.propulsion.dv_magnitude_sigma is not None
    )


def score_maneuver_candidates(
    conjunction_id: str,
    r_sat_km: np.ndarray,
    v_sat_km_s: np.ndarray,
    r_rel_km: np.ndarray,
    p_rel_km2: np.ndarray,
    t_burn_utc: str,
    t_ca_utc: str,
    cap: SatelliteCapability,
    policy: OperatorPolicy,
    geometry: Optional[WalkerDeltaGeometry] = None,
    pc_precomputed: Optional[float] = None,
    miss_distance_km: Optional[float] = None,
    recovery_orbits: float = 1.0,
    v_rel_km_s: Optional[np.ndarray] = None,
    secondary_radius_m: Optional[float] = None,
) -> ManeuverScoringResult:
    """Core APS 2.5 scoring function.

    Co-optimizes avoidance and return-to-slot in a single pass.
    The return burn cost is included in the Δv term before scoring,
    not as a post-processing step.

    Parameters
    ----------
    conjunction_id : str
    r_sat_km : np.ndarray shape (3,)
        Satellite position in ECI [km].
    v_sat_km_s : np.ndarray shape (3,)
        Satellite velocity in ECI [km/s].
    r_rel_km : np.ndarray shape (3,)
        Relative position at TCA [km].
    p_rel_km2 : np.ndarray shape (3,3)
        Combined covariance at TCA [km²].
    t_burn_utc : str
        ISO-8601 burn time.
    t_ca_utc : str
        ISO-8601 TCA time.
    cap : SatelliteCapability
        Full 9.1 satellite capability object.
    policy : OperatorPolicy
        Full 9.2 operator policy object.
    geometry : WalkerDeltaGeometry or None
        9.3 constellation geometry.  Required when
        cap.slot.in_constellation = True; ignored otherwise.
    pc_precomputed : float or None
        Pre-computed Pc from propagator.  Optional.
    miss_distance_km : float or None
        Miss distance at TCA [km].  Optional.
    recovery_orbits : float
        Recovery window [orbits].  Used for drag-corrected return cost.
    v_rel_km_s : np.ndarray shape (3,) or None
        Relative velocity at TCA [km/s].  SCRUM-389.  Required to compute Pc,
        because the encounter plane is perpendicular to it and without one there
        is no plane to integrate over.  When absent the planner establishes no
        Pc and falls back to the Mahalanobis screen, recording that it did.

    Returns
    -------
    ManeuverScoringResult
    """
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    dt_to_ca_s = _dt_seconds(t_burn_utc, t_ca_utc)
    if dt_to_ca_s <= 0:
        raise ValueError("t_burn_utc must be before t_ca_utc")

    # --- Mahalanobis pre-computation ---
    m2_pre = mahalanobis_sq(r_rel_km, p_rel_km2)
    cov_quality = _covariance_quality(m2_pre)

    # --- CTO Item 3: lifetime fraction ---
    lifetime_frac = compute_lifetime_fraction(
        cap.lifetime, policy.mission_lifetime_days_total
    )
    v_available = cap.lifetime.v_available_m_s

    # --- Effective dv budget ---
    requested_dv = policy.max_dv_per_event_ms
    dv_mag_m_s = cap.effective_dv_limit_m_s(requested_dv)

    # --- SCRUM-389: establish this event's Pc and hard-body radius ---
    pc_pre, pc_source, hbr_m, hbr_source = resolve_pc(
        cap, r_sat_km, v_sat_km_s, r_rel_km, v_rel_km_s, p_rel_km2, pc_precomputed,
        secondary_radius_m=secondary_radius_m,
    )

    # --- SCRUM-387: our own pre-maneuver Pc, needed even when one was supplied ---
    #
    # A maneuver's effect is a RATIO, not a subtraction.
    #
    # When an operator supplies a Pc it can disagree with our geometry by a
    # large factor. Measured on the SCRUM-393 fixture, a supplied 2.5e-04
    # against our own 6.26e-04. Subtracting a post-maneuver Pc computed on OUR
    # geometry from THEIR pre-maneuver number compares two different scales, and
    # the difference means nothing in either model. On that fixture it tipped
    # the event to no-burn by a nine percent margin, which is well inside the
    # disagreement between the two figures.
    #
    # So the burn's effect is taken as the fraction of risk it removes in our
    # model, and the absolute level stays the authority's. That extends
    # SCRUM-389 AC3 rather than contradicting it: an external Pc wins for the
    # level, because whoever produced it has more information than we do, and
    # our model can be wrong about the level while still being right that a
    # given burn removes three quarters of it.
    #
    # When the Pc was computed rather than supplied this is exactly the same
    # number as pc_pre, so the ratio form collapses to the difference form and
    # there is no second code path to keep in step.
    pc_pre_geometric = compute_pc_post(
        r_sat_km, v_sat_km_s, r_rel_km, v_rel_km_s, p_rel_km2, hbr_m
    )

    # --- Feasibility pre-screen ---
    passes, nogo_code, nogo_human, risk_gate = _passes_feasibility(
        cap, policy, dv_mag_m_s, pc_pre, pc_source, miss_distance_km,
        math.sqrt(m2_pre),
    )

    if not passes:
        return _build_nogo_result(
            conjunction_id=conjunction_id,
            t_burn_utc=t_burn_utc,
            m2_pre=m2_pre,
            cov_quality=cov_quality,
            lifetime_frac=lifetime_frac,
            pc_precomputed=pc_precomputed,
            nogo_code=nogo_code,
            nogo_human=nogo_human,
            now_iso=now_iso,
            execution_error_modelled=_execution_error_modelled(cap),
            pc_pre=pc_pre,
            pc_source=pc_source,
            hbr_m=hbr_m,
            hbr_source=hbr_source,
            risk_gate=risk_gate,
        )

    # --- Constellation slot context ---
    in_constellation = cap.slot.in_constellation
    acceptable_drift_km = cap.slot.acceptable_drift_km
    return_dv_budget = cap.slot.return_dv_budget_m_s
    max_recovery_s = cap.slot.max_recovery_time_s

    # --- CW state transition matrix ---
    #
    # SCRUM-397. cw_phi_rv returns an RTN-ORDERED block. The candidate burn
    # directions and r_rel_km are ECI. Multiplying the two directly, which is
    # what this code did, feeds each ECI component into whichever CW block sits
    # at that index. On a 53 degree inclined orbit that reported 57 km of
    # separation change from a cross-track burn that produces 105 m, because the
    # ECI cross-track unit vector has components on all three ECI axes and picks
    # up the along-track block, whose entry grows as -3nt.
    #
    # The clearest symptom: the correct displacement for a given burn is the
    # same on every circular orbit, because a prograde burn does the same thing
    # wherever you are. The unrotated one varied with inertial position.
    #
    # For an axis-aligned geometry, r along +x and v along +y, the rotation is
    # exactly the identity. Every scoring test used such a geometry, which is
    # why 397's fix looked complete.
    #
    # SCRUM-409. It was not. Conjugating with a SINGLE rotation, as 397 did,
    # is correct only when the block's input and output are expressed in the
    # same LVLH frame. Phi_rv's input (a burn-epoch delta-v) and output (a
    # TCA-epoch delta-r) are not: the LVLH frame has physically rotated with
    # the orbit in between, by n * dt_to_ca_s radians. At the 4-hour lead in
    # SCRUM-386's own worked example that is 2.5 orbits, roughly half a
    # revolution of drift between the two frames, and the single-rotation
    # result came out on the order of 100% wrong, in some geometries pointing
    # nearly opposite the true displacement. The identity-rotation blind spot
    # that hid 397's bug hid this one too: an identity rotation applied twice
    # is indistinguishable from the same identity applied once.
    #
    # The fix uses two rotations, not one: the burn-epoch rotation for the
    # input (exact there, since delta-r = 0 at the burn instant makes the
    # single-epoch conversion trivially correct on that side only), and the
    # TCA-epoch rotation, advanced analytically from the burn-epoch one under
    # the same circular-orbit assumption cw_phi_rv already makes, for the
    # output. See frames.rotate_cw_block_two_epoch and
    # frames.advance_rtn_to_eci_rotation for the derivation and the
    # verification against true two-body motion.
    rtn_to_eci_burn = frames.rtn_to_eci_rotation(r_sat_km, v_sat_km_s)
    rtn_to_eci_tca = frames.advance_rtn_to_eci_rotation(
        rtn_to_eci_burn, cap.a_ref_km, dt_to_ca_s
    )
    phi_rv_rtn = cw_phi_rv(cap.a_ref_km, dt_to_ca_s)
    phi_rv = frames.rotate_cw_block_two_epoch(
        phi_rv_rtn, rtn_to_eci_tca, rtn_to_eci_burn
    )

    # --- Scoring weights ---
    weights = policy.scoring_weights
    lam_v = weights.lambda_dv
    lam_L = weights.lambda_lifetime
    lam_s = weights.lambda_slot_deviation

    # --- Candidate directions ---
    directions = _candidate_directions_v25(r_sat_km, v_sat_km_s, cap)

    # --- No-burn baseline ---
    no_burn_candidate = {
        "direction": "no-burn",
        "dv_eci_km_s": [0.0, 0.0, 0.0],
        "delta_C": 0.0,
        "utility": 0.0,
    }
    all_candidates_v24: List[Dict[str, Any]] = [no_burn_candidate]
    all_candidates_v25: List[CandidateScore] = []

    best_candidate: Optional[CandidateScore] = None

    # SCRUM-387: search delta-v magnitude, not just direction.
    #
    # Candidates used to be generated at exactly one magnitude, the effective
    # delta-v limit, so the recommendation was the ceiling by construction. The
    # old utility hid that, because delta_C grows quadratically in delta-v
    # against a linear cost so the maximum always sat at the limit anyway. With
    # the Pc-traded utility the benefit saturates and an interior optimum
    # exists, but only if something looks for it.
    #
    # Geometric spacing because the answer spans four decades. Measured on the
    # RED-001 geometry the optimum runs from 0.007 m/s at a four-hour lead to
    # 0.0004 m/s at seventy-two, against a ceiling of order 1 m/s.
    dv_floor_m_s = max(cap.propulsion.min_dv_m_s, dv_mag_m_s * 1.0e-4)
    if dv_mag_m_s > dv_floor_m_s:
        dv_grid = np.geomspace(
            dv_floor_m_s, dv_mag_m_s, conventions.DV_SEARCH_POINTS
        )
    else:
        dv_grid = np.array([dv_mag_m_s])

    for name, d_hat in directions:
        best_for_direction: Optional[CandidateScore] = None

        for _dv_try in dv_grid:
            # float() rather than the numpy scalar. np.geomspace yields np.float64,
            # which propagates through every derived quantity and turns
            # drag_correction_applied into np.bool_, which json cannot serialise.
            # The artifact serialisation test is what caught it.
            dv_try_m_s = float(_dv_try)
            dv_vec_km_s = d_hat * (dv_try_m_s / 1000.0)

            # Post-maneuver relative position via CW
            delta_r_km = phi_rv @ dv_vec_km_s
            r_post_km = r_rel_km - delta_r_km

            # SCRUM-365: execution-error covariance (thrust misalignment +
            # magnitude uncertainty). Zero matrix (no change) when the
            # profile does not carry these parameters -- AC1.
            #
            # SCRUM-397: d_hat is ECI and phi_rv is now the ECI-expressed map, so
            # the velocity covariance and the map it is propagated through are in
            # the same frame. Passing the raw RTN block here, as this did, made
            # Q_exec wrong in the same way and for the same reason delta_r was.
            # compute_q_exec_km2's own docstring flagged that and deferred it.
            q_exec_km2 = compute_q_exec_km2(
                direction_hat=d_hat,
                dv_mag_km_s=dv_try_m_s / 1000.0,
                thrust_misalignment_deg=cap.propulsion.thrust_misalignment_deg,
                dv_magnitude_sigma=cap.propulsion.dv_magnitude_sigma,
                phi_rv=phi_rv,
            )
            s_covariance_km2 = p_rel_km2 + q_exec_km2

            # Mahalanobis gain (research doc sign: post - pre, positive = safer)
            m2_post = mahalanobis_sq(r_post_km, s_covariance_km2)
            delta_C_v25 = m2_post - m2_pre      # research doc convention
            delta_C_v24 = m2_pre - m2_post      # v2.4 output convention

            # SCRUM-393: what the collision probability becomes if this candidate is
            # executed. Uses the post-burn relative position and the same covariance
            # the Mahalanobis gain uses, so Q_exec is included whenever the profile
            # carries execution error and the Pc is not quietly a perfect-burn one.
            pc_post_candidate = compute_pc_post(
                r_sat_km, v_sat_km_s, r_post_km, v_rel_km_s, s_covariance_km2, hbr_m
            )

            # --- Co-optimized return cost (CTO items 1 and 2) ---
            post_drift_km = 0.0
            dv_return = 0.0
            drag_applied = False
            recovery_plan: Optional[SlotRecoveryPlan] = None

            if in_constellation:
                # Along-track displacement from this burn direction
                # (tangential component drives slot drift)
                tangential_component = float(np.dot(dv_vec_km_s, _unit(v_sat_km_s)))
                dv_tangential_m_s = abs(tangential_component) * 1000.0

                post_drift_km = _along_track_displacement_km(
                    dv_tangential_m_s, cap.a_ref_km, dt_to_ca_s
                )

                # Drag-corrected return cost (CTO Item 2)
                dv_return_base = _return_burn_cost_m_s(dv_tangential_m_s, cap.a_ref_km)
                dv_return_drag = _drag_corrected_dv_return_m_s(
                    dv_tangential_m_s, cap.a_ref_km,
                    recovery_orbits, _drag_gate_altitude_from_sma_km(cap.a_ref_km),
                )
                drag_applied = dv_return_drag > dv_return_base + 1e-6
                dv_return = dv_return_drag

                # Recovery plan (J2-corrected target from 9.3)
                if geometry is not None:
                    # Use plane/seat from slot_id if parseable, else default P0-S0
                    plane_idx, seat_idx = _parse_slot_id(cap.slot.slot_id)
                    recovery_plan = geometry.plan_slot_recovery(
                        plane_idx=plane_idx,
                        seat_idx=seat_idx,
                        dv_avoid_m_s=dv_try_m_s,
                        post_maneuver_drift_km=post_drift_km,
                        acceptable_drift_km=acceptable_drift_km,
                        return_dv_budget_m_s=return_dv_budget,
                        max_recovery_time_s=max_recovery_s,
                        recovery_epoch_offset_s=dt_to_ca_s,
                    )

            dv_total = dv_try_m_s + dv_return

            # --- Scoring terms ---
            # lambda_v * delta_v  (total mission cost including return)
            dv_cost_term = lam_v * dv_total

            # lambda_L * delta_L  (lifetime impact)
            # Uses v_available (after reserve) and lifetime_fraction_used
            # delta_L = dv_total / v_available, scaled by lifetime maturity
            delta_L_base = dv_total / max(1e-6, v_available)
            delta_L = delta_L_base * (1.0 + lifetime_frac)   # heavier penalty near EOL
            lifetime_cost_term = lam_L * delta_L

            # lambda_s * delta_S  (constellation slot deviation, 0 if not in constellation)
            if in_constellation and acceptable_drift_km > 0:
                delta_S = post_drift_km / acceptable_drift_km
            else:
                delta_S = 0.0
            slot_cost_term = lam_s * delta_S

            # --- SCRUM-387: what the risk reduction is worth ---
            #
            # The old term was delta_C_v25, the Mahalanobis gain. Two structural
            # problems, neither fixable by reweighting.
            #
            # It is a LOG-risk quantity. m^2 is linear in log Pc, so delta_C is
            # 2*ln(Pc_pre/Pc_post). RED-001's delta_C of 148082 claims the burn
            # cut collision probability by a factor of e^74041. The utility paid
            # linearly, without ceiling, for log risk reduction that stopped
            # meaning anything many orders of magnitude earlier.
            #
            # And it is quadratic in delta-v against a linear cost, so the
            # derivative 2*g*dv - lambda is positive everywhere above
            # lambda/2g and the maximum always sits at the delta-v ceiling.
            # Burning to the limit was a property of the functional form.
            #
            # Trading in Pc fixes both. The benefit is bounded above by Pc_pre,
            # so separation stops paying once the probability is gone. Pc_post
            # falls as exp(-m^2/2) with m^2 quadratic in delta-v, so marginal
            # benefit decays while cost stays linear and an interior optimum
            # exists. Measured across the permitted 4 to 72 hour window the
            # utility at the optimum varies by 0.3 percent on one parameter set,
            # against 314x for the old form.
            if (
                pc_pre is not None
                and pc_post_candidate is not None
                and pc_pre_geometric is not None
                and pc_pre_geometric > 0.0
            ):
                # Fraction of the risk this burn removes, in our model.
                # Clamped at zero because a burn that makes things worse should
                # score no benefit rather than a negative one; the delta-v cost
                # already penalises it and a negative benefit would double-count.
                reduction_fraction = max(
                    0.0, 1.0 - pc_post_candidate / pc_pre_geometric
                )
                risk_benefit = (
                    risk_exchange_rate(policy) * pc_pre * reduction_fraction
                )
                basis = UTILITY_BASIS_PC
            elif pc_pre is not None and pc_pre > 0.0:
                # SCRUM-487: a usable Pc exists, but OUR geometric Pc underflows
                # to zero, so the exact ratio above is 0/0 and unusable.
                #
                # This is the externally-supplied Pc case: an authority states a
                # real, above-threshold probability while our own geometry puts
                # it below floating-point. SCRUM-485 correctly still recommends a
                # maneuver there -- the supplied Pc is real -- but the burn was
                # priced on the unbounded fallback below, so the commanded
                # delta-v was simply the policy's cap rather than anything sized
                # to buy that risk down.
                #
                # The fix keeps the SCRUM-387 structure exactly: the LEVEL stays
                # the authority's supplied Pc, and the FRACTION a burn removes
                # comes from our geometry. Only the fraction is evaluated in
                # Mahalanobis space instead of as a ratio of two probabilities:
                #
                #     Pc ~ k * exp(-m^2 / 2)   for a fixed hard-body radius,
                #     so  Pc_post / Pc_pre  ~  exp(-(m2_post - m2_pre) / 2)
                #     and the prefactor k cancels, leaving
                #     reduction_fraction = 1 - exp(-delta_C_v25 / 2).
                #
                # That is the leading-order form of the exact ratio -- it is what
                # the ratio becomes once the prefactor cancels -- and it stays
                # well defined and bounded in [0, 1) precisely where the ratio
                # dies, because m2_pre and m2_post are finite and already
                # computed for this candidate even when both probabilities have
                # underflowed to exactly zero. A test pins it against the exact
                # ratio on an event where both are computable.
                #
                # The exponential is NOT evaluated for a worsening burn. It is
                # clamped to zero there, matching the exact-ratio clamp above --
                # and that clamp is also what keeps this safe, because
                # delta_C_v25 can be large and negative and exp() of half that
                # overflows rather than returning a large number.
                if delta_C_v25 > 0.0:
                    reduction_fraction = (
                        1.0 if math.isinf(delta_C_v25)
                        else 1.0 - math.exp(-delta_C_v25 / 2.0)
                    )
                else:
                    # A worsening burn, or no finite separation gain, buys
                    # nothing. A non-finite delta_C_v25 lands here too, which is
                    # the conservative direction: no benefit claimed.
                    reduction_fraction = 0.0
                risk_benefit = (
                    risk_exchange_rate(policy) * pc_pre * reduction_fraction
                )
                basis = UTILITY_BASIS_PC
            else:
                # Fallback, for an event with NO usable Pc at all -- none
                # supplied and none computable. Carries both defects described
                # above, and is why the burn it prices is always the policy cap.
                #
                # SCRUM-485 refuses to recommend a maneuver on exactly that
                # condition (pc_pre None or zero), and SCRUM-487 moved the
                # supplied-Pc case to the branch above, so this basis no longer
                # sizes any burn that gets commanded. It is kept because the
                # candidates are still scored and reported, and because a no-go
                # here would mean a no-go on every event with no Pc rather than
                # not scoring events we can still say something useful about.
                risk_benefit = delta_C_v25
                basis = UTILITY_BASIS_DELTA_C

            U = risk_benefit - dv_cost_term - lifetime_cost_term - slot_cost_term

            candidate = CandidateScore(
                direction=name,
                dv_eci_km_s=dv_vec_km_s.tolist(),
                dv_avoid_m_s=dv_try_m_s,
                dv_return_m_s=dv_return,
                dv_total_m_s=dv_total,
                delta_C_v25=float(delta_C_v25),
                delta_C_v24=float(delta_C_v24),
                m2_post=float(m2_post),
                dv_cost_term=float(dv_cost_term),
                lifetime_cost_term=float(lifetime_cost_term),
                slot_cost_term=float(slot_cost_term),
                utility=float(U),
                post_drift_km=float(post_drift_km),
                recovery_plan=recovery_plan,
                drag_correction_applied=drag_applied,
                pc_post=pc_post_candidate,
                utility_basis=basis,
            )


            # Best magnitude for THIS direction. Only the winner is reported, so
            # candidates_v25 stays one entry per direction and the response
            # contract does not grow by the size of the search grid.
            if best_for_direction is None or U > best_for_direction.utility:
                best_for_direction = candidate
                best_v24_for_direction = {
                    "direction": name,
                    "dv_eci_km_s": dv_vec_km_s.tolist(),
                    "delta_C": float(delta_C_v24),
                    "utility": float(U),
                }

        if best_for_direction is not None:
            all_candidates_v25.append(best_for_direction)
            all_candidates_v24.append(best_v24_for_direction)

            # Best = highest utility across directions (no-burn baseline is U=0)
            if best_candidate is None or best_for_direction.utility > best_candidate.utility:
                best_candidate = best_for_direction

    # --- Final decision ---
    # No-burn wins if all candidates have U <= 0
    if best_candidate is None or best_candidate.utility <= 0.0:
        return _build_nogo_result(
            conjunction_id=conjunction_id,
            t_burn_utc=t_burn_utc,
            m2_pre=m2_pre,
            cov_quality=cov_quality,
            lifetime_frac=lifetime_frac,
            pc_precomputed=pc_precomputed,
            nogo_code="no_utility_gain",
            nogo_human=(
                "No candidate maneuver produced positive utility. "
                "The no-burn baseline is optimal: risk reduction does not "
                "justify the delta-v and lifetime cost at this geometry."
            ),
            now_iso=now_iso,
            all_candidates_v24=all_candidates_v24,
            all_candidates_v25=all_candidates_v25,
            m2_post=m2_pre,
            execution_error_modelled=_execution_error_modelled(cap),
            pc_pre=pc_pre,
            pc_source=pc_source,
            hbr_m=hbr_m,
            hbr_source=hbr_source,
            risk_gate=risk_gate,
        )

    # Post-maneuver risk surrogate
    #
    # SCRUM-393. This used to report pc_precomputed whenever one was supplied,
    # which is the PRE-maneuver probability under a post-maneuver name. The
    # field therefore said the recommended burn reduced risk by exactly zero,
    # and it said it most confidently on the events carrying the most real data.
    # It now reports the burn's own Pc.
    #
    # A supplied Pc still wins for pc_pre, per SCRUM-389 AC3, because a CDM or
    # UDL Pc comes from an authority with more information than we have. That
    # does not make it the post-maneuver figure. Nobody external has computed
    # the Pc after a burn we have not flown, so the post-maneuver value is ours
    # to compute whether or not the pre-maneuver one was given to us.
    #
    # The 1/m2_post fallback stays for events with no relative velocity, where
    # there is no encounter plane and so no Pc at either end. It is an inverse
    # area rather than a probability, which is why the source is recorded
    # alongside it rather than left for a reader to infer from the magnitude.
    pc_post = best_candidate.pc_post
    if pc_post is not None:
        risk_surrogate_post = float(pc_post)
        risk_surrogate_source = RISK_SURROGATE_PC_POST
    else:
        risk_surrogate_post = 1.0 / max(1e-12, best_candidate.m2_post)
        risk_surrogate_source = RISK_SURROGATE_INVERSE_M2

    # lifetime_penalty in v2.4 form (dv_avoid / v_remaining) for backward compat
    lifetime_penalty_v24 = (
        best_candidate.dv_avoid_m_s / max(1e-6, cap.lifetime.v_remaining_m_s)
    )

    # SCRUM-365: reflects whether m2_post actually includes execution
    # error, not a hardcoded False -- true whenever the profile carries
    # either parameter (matches compute_q_exec_km2's own "both None ->
    # zero contribution" check).
    execution_error_modelled = _execution_error_modelled(cap)

    return ManeuverScoringResult(
        conjunction_id=conjunction_id,
        # v2.4 preserved
        direction=best_candidate.direction,
        dv_eci_km_s=best_candidate.dv_eci_km_s,
        dv_magnitude_m_s=best_candidate.dv_avoid_m_s,
        t_burn_utc=t_burn_utc,
        utility=best_candidate.utility,
        delta_C=best_candidate.delta_C_v24,
        m2_pre=float(m2_pre),
        m2_post=best_candidate.m2_post,
        fuel_cost_m_s=best_candidate.dv_avoid_m_s,
        lifetime_penalty=lifetime_penalty_v24,
        risk_surrogate_post=risk_surrogate_post,
        all_candidates=all_candidates_v24,
        # v2.5 additions
        dv_return_m_s=best_candidate.dv_return_m_s,
        dv_total_m_s=best_candidate.dv_total_m_s,
        lifetime_fraction_used=lifetime_frac,
        slot_cost_term=best_candidate.slot_cost_term,
        post_drift_km=best_candidate.post_drift_km,
        recovery_plan=best_candidate.recovery_plan,
        covariance_quality=cov_quality,
        no_go_reason_code="",
        no_go_human_readable="",
        drag_correction_applied=best_candidate.drag_correction_applied,
        candidates_v25=all_candidates_v25,
        evaluated_at=now_iso,
        pc_pre=pc_pre,
        pc_source=pc_source,
        hbr_m=hbr_m,
        hbr_source=hbr_source,
        risk_gate=risk_gate,
        execution_error_modelled=execution_error_modelled,
        pc_post=pc_post,
        risk_surrogate_source=risk_surrogate_source,
        utility_basis=best_candidate.utility_basis,
    )


# ---------------------------------------------------------------------------
# Public entry point (v2.5 API)
# ---------------------------------------------------------------------------

def evaluate_conjunction_v25(
    req: Dict[str, Any],
    geometry: Optional[WalkerDeltaGeometry] = None,
    recovery_orbits: float = 1.0,
) -> ManeuverScoringResult:
    """APS 2.5 entry point: evaluate a conjunction with full mission context.

    Accepts a v2.5 request dict.  The satellite sub-dict is passed to
    SatelliteCapability.from_request(); the policy sub-dict is passed to
    OperatorPolicy.from_yaml() or constructed directly.

    This function does NOT call evaluate_conjunction() from decision_model.py.
    It is a parallel entry point that produces a strict superset of the
    v2.4 response.  The v2.4 path is preserved and unchanged.

    Parameters
    ----------
    req : dict
        {
          "conjunction_id": str,
          "satellite": { ... SatelliteCapability fields ... },
          "conjunction": {
            "obj_id": str,
            "t_ca_utc": str,
            "r_rel_km": [x, y, z],
            "p_rel_km2": [9 floats],
            "pc_precomputed": float (optional),
            "miss_distance_km": float (optional),
            "v_rel_km_s": [vx, vy, vz] (optional, SCRUM-389),
          },
          "policy": { ... OperatorPolicy fields ... }
        }
    geometry : WalkerDeltaGeometry or None
        Pass when satellite is in a constellation.
    recovery_orbits : float
        Recovery window for drag correction.

    Returns
    -------
    ManeuverScoringResult
    """
    if not isinstance(req, dict):
        raise ValueError("request must be a JSON object")

    conjunction_id = req.get("conjunction_id", "")
    sat_dict = req.get("satellite", {})
    conj_dict = req.get("conjunction", {})
    policy_dict = req.get("policy", {})

    # Build capability and policy objects
    cap = SatelliteCapability.from_request(sat_dict)
    policy = _policy_from_dict(policy_dict)

    # Parse vectors
    r_sat_km   = _as_vec3(sat_dict["r_sat_km"],  "satellite.r_sat_km")
    v_sat_km_s = _as_vec3(sat_dict["v_sat_km_s"], "satellite.v_sat_km_s")
    r_rel_km   = _as_vec3(conj_dict["r_rel_km"],  "conjunction.r_rel_km")
    p_rel_km2  = _as_cov9(conj_dict["p_rel_km2"], "conjunction.p_rel_km2")

    t_burn_utc = sat_dict["t_burn_utc"]
    t_ca_utc   = conj_dict["t_ca_utc"]
    pc_pre     = conj_dict.get("pc_precomputed")
    miss_dist  = conj_dict.get("miss_distance_km")
    v_rel_raw  = conj_dict.get("v_rel_km_s")
    v_rel_km_s = (
        _as_vec3(v_rel_raw, "conjunction.v_rel_km_s") if v_rel_raw is not None else None
    )
    # SCRUM-489: the secondary's measured radius when the conjunction states it.
    # A LeoLabs CDM always does; absent it, the screening floor still applies.
    secondary_radius_m = conj_dict.get("secondary_radius_m")

    return score_maneuver_candidates(
        conjunction_id=conjunction_id,
        r_sat_km=r_sat_km,
        v_sat_km_s=v_sat_km_s,
        r_rel_km=r_rel_km,
        p_rel_km2=p_rel_km2,
        t_burn_utc=t_burn_utc,
        t_ca_utc=t_ca_utc,
        cap=cap,
        policy=policy,
        geometry=geometry,
        pc_precomputed=pc_pre,
        miss_distance_km=miss_dist,
        recovery_orbits=recovery_orbits,
        v_rel_km_s=v_rel_km_s,
        secondary_radius_m=secondary_radius_m,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _policy_from_dict(policy_dict: Dict[str, Any]) -> OperatorPolicy:
    """Build OperatorPolicy from a flat request dict.

    Supports both v2.4 style (lambda_v, lambda_L, dv_mag_limit_m_s) and
    v2.5 style (scoring_weights, max_dv_per_event_ms).
    """
    from common.operator_policy import ScoringWeights

    # v2.5 path: scoring_weights sub-dict present
    sw_raw = policy_dict.get("scoring_weights", {})
    weights = ScoringWeights(
        lambda_dv            = float(sw_raw.get("lambda_dv",
                                policy_dict.get("lambda_v", 1.0))),
        lambda_lifetime      = float(sw_raw.get("lambda_lifetime",
                                policy_dict.get("lambda_L", 0.8))),
        lambda_slot_deviation= float(sw_raw.get("lambda_slot_deviation", 1.2)),
    )

    max_dv = float(policy_dict.get("max_dv_per_event_ms",
                   policy_dict.get("dv_mag_limit_m_s", 2.0)))
    mission_lt = policy_dict.get("mission_lifetime_days_total", None)

    return OperatorPolicy(
        operator_id   = str(policy_dict.get("operator_id", "DEFAULT_LEO")),
        policy_version= str(policy_dict.get("policy_version", "2.5.0")),
        decision_mode = policy_dict.get("decision_mode", "aps"),
        max_dv_per_event_ms = max_dv,
        scoring_weights     = weights,
        mission_lifetime_days_total = (
            float(mission_lt) if mission_lt is not None else None
        ),
        pc_maneuver_threshold = float(
            policy_dict.get("pc_maneuver_threshold", 1e-4)
        ),
        pc_monitor_threshold  = float(
            policy_dict.get("pc_monitor_threshold", 1e-5)
        ),
        min_miss_distance_km = float(
            policy_dict.get("min_miss_distance_km", 1.0)
        ),
        mahalanobis_screen_threshold = float(
            policy_dict.get("mahalanobis_screen_threshold", 4.0)
        ),
    )


def _parse_slot_id(slot_id: str) -> Tuple[int, int]:
    """Parse 'P02-S05' -> (2, 5).  Returns (0, 0) on failure."""
    try:
        parts = slot_id.upper().split("-")
        if len(parts) == 2 and parts[0].startswith("P") and parts[1].startswith("S"):
            return int(parts[0][1:]), int(parts[1][1:])
    except (ValueError, IndexError):
        pass
    return 0, 0


def _build_nogo_result(
    conjunction_id: str,
    t_burn_utc: str,
    m2_pre: float,
    cov_quality: str,
    lifetime_frac: float,
    pc_precomputed: Optional[float],
    nogo_code: str,
    nogo_human: str,
    now_iso: str,
    all_candidates_v24: Optional[List] = None,
    all_candidates_v25: Optional[List] = None,
    m2_post: Optional[float] = None,
    execution_error_modelled: bool = False,
    pc_pre: Optional[float] = None,
    pc_source: str = PC_SOURCE_UNAVAILABLE,
    hbr_m: float = 0.0,
    hbr_source: str = "",
    risk_gate: str = "",
) -> ManeuverScoringResult:
    """Construct a no-go ManeuverScoringResult.

    SCRUM-365 follow-up: execution_error_modelled must reflect whether
    the satellite's propulsion profile carries the execution-error
    parameters, not a hardcoded False. A no-go result still has no
    burn (so m2_post trivially equals m2_pre, with no Q_exec applied --
    there is nothing to have an execution error in), but the flag
    itself describes the satellite's capability/configuration, not
    whether this specific event happened to need a burn. Reporting a
    hardcoded False here would misleadingly suggest the capability is
    absent for a satellite whose profile actually supports it.
    """
    no_burn_v24 = [{"direction": "no-burn", "dv_eci_km_s": [0,0,0],
                    "delta_C": 0.0, "utility": 0.0}]
    # SCRUM-389 added pc_pre as a parameter carrying the RESOLVED Pc, supplied
    # or computed. A local assignment here shadowed it with the external value,
    # so an event rejected by the Pc threshold reported no Pc at all and
    # risk_surrogate_post fell to the 1/m^2 branch, publishing roughly 1e-2 on a
    # field that otherwise carries a probability near 1e-8. Wrong units, wrong
    # magnitude, on the field the UI reads.
    if pc_pre is None:
        pc_pre = pc_precomputed
    # SCRUM-393. On a no-go there is no burn, so the post-maneuver risk is the
    # pre-maneuver risk. That is correct rather than a defect, and the source
    # says so explicitly. The rule this has to satisfy is that a pre-maneuver
    # value is never carried under a post-maneuver name SILENTLY, not that the
    # two can never be equal.
    risk_post = (1.0 / max(1e-12, m2_post or m2_pre))
    risk_source = RISK_SURROGATE_INVERSE_M2
    if pc_pre is not None:
        risk_post = float(pc_pre)
        risk_source = RISK_SURROGATE_PC_PRE_NO_BURN

    return ManeuverScoringResult(
        conjunction_id=conjunction_id,
        direction="no-burn",
        dv_eci_km_s=[0.0, 0.0, 0.0],
        dv_magnitude_m_s=0.0,
        t_burn_utc=t_burn_utc,
        utility=0.0,
        delta_C=0.0,
        m2_pre=float(m2_pre),
        m2_post=float(m2_post or m2_pre),
        fuel_cost_m_s=0.0,
        lifetime_penalty=0.0,
        risk_surrogate_post=risk_post,
        all_candidates=all_candidates_v24 or no_burn_v24,
        dv_return_m_s=0.0,
        dv_total_m_s=0.0,
        lifetime_fraction_used=lifetime_frac,
        slot_cost_term=0.0,
        post_drift_km=0.0,
        recovery_plan=None,
        covariance_quality=cov_quality,
        no_go_reason_code=nogo_code,
        no_go_human_readable=nogo_human,
        drag_correction_applied=False,
        candidates_v25=all_candidates_v25 or [],
        evaluated_at=now_iso,
        execution_error_modelled=execution_error_modelled,
        pc_pre=pc_pre,
        pc_source=pc_source,
        hbr_m=hbr_m,
        hbr_source=hbr_source,
        risk_gate=risk_gate,
        pc_post=None,
        risk_surrogate_source=risk_source,
        # No candidate was selected, so no utility basis applies. An empty
        # string says that, where either basis name would be a claim.
        utility_basis="",
    )


def analyze_conjunction_modes(req: Dict[str, Any]) -> Dict[str, Any]:
    """SCRUM-480: score once and derive read-only policy comparisons.

    Prices the highest-utility burn candidate, including nonpositive utility.
    This is modeled avoidance-burn delta-v, not measured fuel consumption or
    authorization to execute. Required events without a feasible candidate
    have unavailable pricing rather than a fabricated zero cost.
    """
    from copy import deepcopy
    from dataclasses import replace

    policy = _policy_from_dict(req.get("policy", {}))
    scoring_req = deepcopy(req)
    scoring_req.setdefault("policy", {})["decision_mode"] = "aps"
    scoring = evaluate_conjunction_v25(scoring_req)

    best_burn = max(
        scoring.candidates_v25,
        key=lambda candidate: candidate.utility,
        default=None,
    )
    burn_dv = float(best_burn.dv_avoid_m_s) if best_burn is not None else None
    miss_km = scoring_req.get("conjunction", {}).get("miss_distance_km") or 999.0

    # SCRUM-485: APS may only recommend against a Pc that exists and is not
    # zero. The zero-Pc events in the reported window scored on the unbounded
    # fallback basis and every one of them recommended a capped burn.
    pc_usable = aps_pc_usable(scoring.pc_pre)

    modes = {}
    for mode in ("aps", "flight_rule_1e4", "flight_rule_1e5"):
        mode_policy = replace(policy, decision_mode=mode)
        required = mode_policy.is_recommendation_required(
            scoring.pc_pre, miss_km, utility=scoring.utility,
            pc_usable=pc_usable,
        )
        # SCRUM-488: the burn MAGNITUDE is gated for every mode, not just the
        # APS verdict.
        #
        # A flight rule fires on pc >= threshold OR miss < the proximity floor,
        # so a zero-Pc event with a sub-kilometre miss still legitimately
        # requires a maneuver through that floor. That verdict is correct and is
        # a real difference from APS; it is left exactly as it was.
        #
        # What is wrong is the size. burn_dv is taken once from the single
        # highest-utility candidate, and for a zero-Pc event that candidate sits
        # on the unbounded delta_c_legacy basis, so burn_dv is simply the largest
        # burn the policy permits. Reporting it in a flight-rule column states a
        # delta-v that was never sized against anything -- on the reference
        # window roughly 99% of the baseline delta-v was this artifact.
        #
        # So such a maneuver is UNPRICED rather than fabricated: required stays
        # true, the count is unchanged, and the delta-v drops out of the mode's
        # total through the unpriced machinery the portfolio already has.
        if not required:
            dv_m_s, pricing_available = 0.0, True
        elif best_burn is None or not pc_usable:
            dv_m_s, pricing_available = None, False
        else:
            dv_m_s, pricing_available = burn_dv, True

        modes[mode] = {
            "maneuver_required": bool(required),
            "dv_m_s": dv_m_s,
            "pricing_available": pricing_available,
        }
        # The state is APS-specific. A flight-rule mode decides on a fixed Pc
        # threshold and has no notion of a utility basis, so reporting a
        # no_usable_pc state there would describe a judgement it never made.
        if mode == "aps":
            modes[mode]["state"] = aps_decision_state(
                scoring.pc_pre, bool(required),
            )

    return {
        "conjunction_id": scoring.conjunction_id,
        "pc_pre": scoring.pc_pre,
        "pc_source": scoring.pc_source,
        "miss_distance_km": scoring_req.get("conjunction", {}).get("miss_distance_km"),
        "utility": scoring.utility,
        "best_burn": {
            "direction": best_burn.direction,
            "dv_m_s": burn_dv,
            "utility": float(best_burn.utility),
            "utility_basis": best_burn.utility_basis,
        } if best_burn is not None else None,
        "no_go_reason_code": scoring.no_go_reason_code,
        "dv_basis": "best_candidate_avoidance_burn",
        # SCRUM-485: the selected candidate's basis, alongside the modes, so a
        # row can be read without re-deriving why APS decided what it did.
        "utility_basis": scoring.utility_basis,
        "pc_usable": pc_usable,
        "modes": modes,
    }
