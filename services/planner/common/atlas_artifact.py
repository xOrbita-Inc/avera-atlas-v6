"""
APS 2.5 -- Step 9.5: Verification Outputs and ATLAS Artifacts
=============================================================
Module: services/planner/common/atlas_artifact.py

Defines post-burn verification outputs, the ATLAS artifact bundle for
dashboard display, mission impact summary, and the decision log for
audit and retrieval.

Depends on:
  - 9.1  common/satellite_capability.py   (SatelliteCapability)
  - 9.2  common/operator_policy.py        (OperatorPolicy)
  - 9.3  common/constellation_geometry.py (SlotRecoveryPlan)
  - 9.4  common/maneuver_scorer.py        (ManeuverScoringResult,
                                           CandidateScore)

Design: APS 2.5 is a recommendation system. The operator retains final
decision authority. The artifacts in this module exist to give the
operator a complete, auditable explanation of every recommendation so
they can make an informed decision and the decision can be reviewed.

Five required artifacts (APS_2_5_Research_V3.ipynb §5.3):
  A1  RiskSummary           -- Why is action needed?
  A2  ManeuverRecommendation -- What should be done?
  A3  DecisionRationale     -- Why this maneuver?
  A4  PostManeuverProjection -- What happens if we execute?
  A5  NoGoReasoning         -- Why are we not acting? (when applicable)

New in 9.5:
  VerificationResult        -- Post-burn pass/fail assessment
  SecondaryConflictCheck    -- Did the burn introduce new conjunctions?
  MissionImpactSummary      -- Lifetime, slot, constellation impact
  ATLASManeuverArtifact     -- Top-level bundle for ATLAS display
  DecisionLog               -- JSON-serialisable audit record
  build_atlas_artifact()    -- Assembly function from ManeuverScoringResult

Scientific gaps carried forward
--------------------------------
  - Secondary conflict screening requires an object catalog at decision time.
    Missing catalog data or an unrunnable horizon screen fails closed with
    secondary_check_performed = False and secondary_conjunction_clear = False,
    which requires M4 safe hold.
  - Post-burn verification uses the pre-computed m2_post from CW dynamics
    as the risk proxy, not an updated Pc from a propagator. The
    verification pass/fail is therefore a planning-time estimate, not a
    real-time observation. APS 3.0 scope.

References
----------
APS_2_5_Research_V3.ipynb §5.3 (five required artifacts).
NASA CARA operational framework: risk summary, rationale, post-maneuver
  projection required before maneuver authorization.
Hejduk / Snow (2019): dilution region classification.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from common.satellite_capability import SatelliteCapability
from common.operator_policy import OperatorPolicy
from common.constellation_geometry import SlotRecoveryPlan
from common.maneuver_scorer import (
    ManeuverScoringResult,
    CandidateScore,
)
from common.secondary_horizon import screen_secondary_catalog


# ---------------------------------------------------------------------------
# A1: RiskSummary
# ---------------------------------------------------------------------------

@dataclass
class RiskSummary:
    """A1: Why is action needed?

    Captures the pre-maneuver risk state for operator review and ATLAS display.

    Attributes
    ----------
    pc_pre : float or None
        Probability of collision before maneuver. None if not supplied.
    miss_distance_km : float or None
        Miss distance at TCA [km]. None if not supplied.
    mahalanobis_pre : float
        Mahalanobis distance at TCA pre-maneuver (sqrt of m2_pre).
    tca_utc : str
        Time of closest approach (ISO-8601 UTC).
    covariance_quality : str
        'good' | 'degraded' | 'dilution_region'
        Derived from m2_pre (< 1.0 = dilution_region, < 4.0 = degraded).
    maneuver_required : bool
        True if Pc >= pc_maneuver_threshold OR miss_distance < floor.
    monitor_only : bool
        True if Pc is between monitor and maneuver thresholds.
    """
    pc_pre: Optional[float]
    miss_distance_km: Optional[float]
    mahalanobis_pre: float
    tca_utc: str
    covariance_quality: str
    maneuver_required: bool
    monitor_only: bool


# ---------------------------------------------------------------------------
# A2: ManeuverRecommendation
# ---------------------------------------------------------------------------

@dataclass
class ManeuverRecommendation:
    """A2: What should be done?

    Attributes
    ----------
    direction : str
        Burn direction: prograde | retrograde | radial | anti-radial |
        cross-track | anti-cross.
    dv_eci_km_s : list[float]
        Burn vector in ECI frame [km/s].
    dv_magnitude_m_s : float
        Avoidance burn magnitude [m/s].
    dv_return_m_s : float
        Return burn cost [m/s]. 0.0 for non-constellated satellites.
    dv_total_m_s : float
        Total mission delta-v cost including return [m/s].
    burn_time_utc : str
        Recommended burn execution time (ISO-8601 UTC).
    drag_correction_applied : bool
        True if drag term was added to dv_return (recovery > 2 orbits
        below 550 km).
    """
    direction: str
    dv_eci_km_s: List[float]
    dv_magnitude_m_s: float
    dv_return_m_s: float
    dv_total_m_s: float
    burn_time_utc: str
    drag_correction_applied: bool


# ---------------------------------------------------------------------------
# A3: DecisionRationale
# ---------------------------------------------------------------------------

@dataclass
class DecisionRationale:
    """A3: Why this maneuver?

    All scoring terms exposed individually so the operator can see exactly
    what drove the recommendation. Answers: why this direction over others,
    and what policy constraints were checked.

    Attributes
    ----------
    utility_score : float
        Final utility U = delta_C - dv_cost - lifetime_cost - slot_cost.
    delta_C : float
        Mahalanobis gain in v2.4 convention (m2_pre - m2_post).
        Negative = post-maneuver position is further from threat = safer.
    dv_cost_term : float
        lambda_v * dv_total_m_s.
    lifetime_cost_term : float
        lambda_L * (dv_total / v_available) * (1 + lifetime_fraction_used).
    slot_cost_term : float
        lambda_s * (post_drift_km / acceptable_drift_km). 0.0 if not in
        constellation.
    lifetime_fraction_used : float
        Mission maturity: 0.0 (fresh) to 1.0 (EOL).
    policy_constraints_applied : list[str]
        Ordered list of checks that ran before scoring.
    candidate_rank : int
        Rank of the recommended direction (1 = best).
    n_candidates_evaluated : int
        Total number of directions scored (including no-burn baseline).
    all_candidates : list[dict]
        Full candidate list with direction, delta_C, utility for each.
    scoring_weights : dict
        Lambda weights used: lambda_dv, lambda_lifetime, lambda_slot_deviation.
    """
    utility_score: float
    delta_C: float
    dv_cost_term: float
    lifetime_cost_term: float
    slot_cost_term: float
    lifetime_fraction_used: float
    policy_constraints_applied: List[str]
    candidate_rank: int
    n_candidates_evaluated: int
    all_candidates: List[Dict[str, Any]]
    scoring_weights: Dict[str, float]

    def human_readable_rationale(self) -> str:
        """Single-paragraph explanation suitable for operator review."""
        lines = [
            f"Direction '{self.all_candidates[self.candidate_rank]['direction'] if self.candidate_rank < len(self.all_candidates) else 'selected'}' "
            f"ranked {self.candidate_rank} of {self.n_candidates_evaluated} candidates "
            f"with utility score {self.utility_score:.4f}.",
            f"Risk reduction (delta_C): {self.delta_C:.4f}. "
            f"Delta-v cost term: {self.dv_cost_term:.4f}. "
            f"Lifetime cost term: {self.lifetime_cost_term:.4f}. "
            f"Slot deviation cost: {self.slot_cost_term:.4f}.",
            f"Satellite is at {self.lifetime_fraction_used * 100:.1f}% of mission lifetime.",
            f"Policy checks applied: {', '.join(self.policy_constraints_applied)}.",
        ]
        return " ".join(lines)


# ---------------------------------------------------------------------------
# Secondary conflict check
# ---------------------------------------------------------------------------

@dataclass
class SecondaryConflictCheck:
    """Result of checking whether the maneuver introduces new conjunctions.

    The secondary conflict check propagates the post-maneuver asset state and
    normalized J2000 catalog states across the operator-policy horizon, evaluates
    Pc at each object's refined TCA, and fails closed when safety cannot be
    established.

    Attributes
    ----------
    secondary_check_performed : bool
        True only when the required catalog/state inputs were available and the
        policy-horizon screen completed successfully.
    secondary_conjunction_clear : bool
        True if no new conjunctions were detected. Meaningful only when
        secondary_check_performed is True.
    flagged_objects : list[str]
        Object IDs of any new conjunctions detected.
    operator_note : str
        Human-readable explanation of the check result.
    closest_approach_km : float or None
        Smallest propagated separation found across screened catalog objects over
        the policy horizon [km]. None when check was not performed.
    closest_object_id : str or None
        Object ID associated with closest_approach_km. None when the check was
        not performed or the catalog was unavailable.
    screening_epoch_utc : str or None
        ISO-8601 UTC burn epoch that anchors the policy-horizon propagation.
        None when no epoch was supplied.
    """
    secondary_check_performed: bool
    secondary_conjunction_clear: bool
    flagged_objects: List[str]
    operator_note: str
    closest_approach_km: Optional[float] = None
    closest_object_id: Optional[str] = None
    screening_epoch_utc: Optional[str] = None


def _run_secondary_conflict_check(
    r_post_km: Optional[List[float]],
    known_objects: Optional[List[Dict[str, Any]]],
    screening_epoch_utc: Optional[str] = None,
    v_post_km_s: Optional[List[float]] = None,
    policy: Optional[OperatorPolicy] = None,
    cap: Optional[SatelliteCapability] = None,
) -> SecondaryConflictCheck:
    """Run the SCRUM-381 secondary-conjunction horizon Pc screen.

    The gate is fail closed. A result is CLEAR only when the full required
    catalog and post-burn state are available, the horizon screen completes,
    and every secondary Pc is strictly below the operator action threshold.
    Missing inputs or any screening failure require NOT CLEAR and M4 safe hold.
    """
    missing = []
    if not known_objects:
        missing.append("object catalog")
    if r_post_km is None:
        missing.append("post-burn position")
    if v_post_km_s is None:
        missing.append("post-burn velocity")
    if not screening_epoch_utc:
        missing.append("burn epoch")
    if policy is None:
        missing.append("operator policy")
    if cap is None:
        missing.append("satellite capability")

    if missing:
        return SecondaryConflictCheck(
            secondary_check_performed=False,
            secondary_conjunction_clear=False,
            flagged_objects=[],
            operator_note=(
                "Secondary horizon screen not performed; missing required input(s): "
                + ", ".join(missing)
                + ". Safety could not be established. MAF requires NOT CLEAR and "
                "M4 safe hold."
            ),
            closest_approach_km=None,
            closest_object_id=None,
            screening_epoch_utc=screening_epoch_utc,
        )

    try:
        result = screen_secondary_catalog(
            r_post_km=r_post_km,
            v_post_km_s=v_post_km_s,
            known_objects=known_objects,
            burn_time_utc=screening_epoch_utc,
            horizon_hours=policy.max_hours_before_tca,
            pc_action=policy.pc_maneuver_threshold,
            primary_radius_m=cap.radius_m,
        )
        return SecondaryConflictCheck(**result)
    except Exception as exc:
        return SecondaryConflictCheck(
            secondary_check_performed=False,
            secondary_conjunction_clear=False,
            flagged_objects=[],
            operator_note=(
                "Secondary horizon screen could not establish safety; "
                f"M4 safe hold is required. Reason: {exc}"
            ),
            closest_approach_km=None,
            closest_object_id=None,
            screening_epoch_utc=screening_epoch_utc,
        )


# ---------------------------------------------------------------------------
# A4: PostManeuverProjection
# ---------------------------------------------------------------------------

@dataclass
class PostManeuverProjection:
    """A4: What happens if we execute?

    Attributes
    ----------
    m2_post : float
        Post-maneuver Mahalanobis distance squared.
    mahalanobis_post : float
        sqrt(m2_post). Higher = further from threat = safer.
    risk_surrogate_post : float
        Post-maneuver risk proxy: pc_precomputed if available,
        else 1 / m2_post.
    slot_recovery_orbits : float
        Minimum recovery orbits to return to slot. 0.0 if not in
        constellation or no recovery required.
    slot_recovery_time_s : float
        Estimated slot recovery time [s].
    slot_recovery_feasible : bool
        True if recovery is within budget and max_recovery_time_s.
    secondary_conflict : SecondaryConflictCheck
        Result of secondary conjunction check.
    recovery_plan : SlotRecoveryPlan or None
        Full slot recovery plan from 9.3/9.4. None for non-constellated.
    execution_error_modelled : bool
        SCRUM-365: True when the satellite's PropulsionProfile carries
        thrust_misalignment_deg and/or dv_magnitude_sigma, in which case
        m2_post includes the resulting execution-error covariance
        (Q_exec). False when the profile does not specify either
        parameter (m2_post is then the pre-SCRUM-365, perfect-burn
        estimate -- optimistic, as before).
    operator_note : str
        Human-readable explanation of whether execution error is
        reflected in this estimate, and why not if it isn't.
    """
    m2_post: float
    mahalanobis_post: float
    risk_surrogate_post: float
    slot_recovery_orbits: float
    slot_recovery_time_s: float
    slot_recovery_feasible: bool
    secondary_conflict: SecondaryConflictCheck
    recovery_plan: Optional[SlotRecoveryPlan]
    execution_error_modelled: bool = False
    operator_note: str = (
        "m2_post excludes execution error covariance. Thrust misalignment and "
        "magnitude uncertainty are not modelled. Post-maneuver separation may be "
        "optimistic. APS 3.0 scope."
    )


# ---------------------------------------------------------------------------
# A5: NoGoReasoning
# ---------------------------------------------------------------------------

@dataclass
class NoGoReasoning:
    """A5: Why are we not acting?

    Attributes
    ----------
    reason_code : str
        Machine-readable code:
        'pc_below_threshold' | 'propulsion_infeasible' |
        'no_utility_gain' | 'trivial_event' | 'blackout_window'
    human_readable : str
        Full explanation suitable for operator review and audit log.
    pc_at_decision : float or None
        Pc value at the time of the no-go decision.
    mahalanobis_at_decision : float
        Mahalanobis distance at the time of the no-go decision.
    """
    reason_code: str
    human_readable: str
    pc_at_decision: Optional[float]
    mahalanobis_at_decision: float


# ---------------------------------------------------------------------------
# VerificationResult -- post-burn pass/fail assessment (new in 9.5)
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Post-burn verification: did the maneuver actually reduce risk?

    This is a planning-time estimate based on CW dynamics. It is not a
    real-time post-burn observation -- that requires updated TLE/covariance
    data after burn execution. APS 3.0 scope for real-time verification.

    Attributes
    ----------
    passed : bool
        True if verification passes all checks below.
    risk_reduced : bool
        True if m2_post > m2_pre (maneuver moved satellite further from
        threat in Mahalanobis space).
    utility_positive : bool
        True if utility > 0 (burn has net positive mission value).
    secondary_clear : bool
        True if secondary conflict check is clear.
    budget_within_limits : bool
        True if dv_total <= max_dv_per_event_ms from OperatorPolicy.
    recovery_within_limits : bool
        True if slot recovery is feasible (or not required).
    failure_reasons : list[str]
        Human-readable explanation of any failed checks.
    m2_pre : float
        Pre-maneuver Mahalanobis distance squared.
    m2_post : float
        Post-maneuver Mahalanobis distance squared.
    delta_m2 : float
        m2_post - m2_pre. Positive = maneuver increased separation = good.
    verification_note : str
        One-line operator-readable verdict.
    """
    passed: bool
    risk_reduced: bool
    utility_positive: bool
    secondary_clear: bool
    budget_within_limits: bool
    recovery_within_limits: bool
    failure_reasons: List[str]
    m2_pre: float
    m2_post: float
    delta_m2: float
    verification_note: str


def _build_verification_result(
    scoring: ManeuverScoringResult,
    policy: OperatorPolicy,
    secondary: SecondaryConflictCheck,
) -> VerificationResult:
    """Build VerificationResult from a ManeuverScoringResult.

    Parameters
    ----------
    scoring : ManeuverScoringResult
        Output from 9.4 score_maneuver_candidates().
    policy : OperatorPolicy
        Operator policy used for budget limits.
    secondary : SecondaryConflictCheck
        Result of secondary conflict check.

    Returns
    -------
    VerificationResult
    """
    failures = []

    risk_reduced = scoring.m2_post > scoring.m2_pre
    if not risk_reduced:
        failures.append(
            f"Maneuver did not increase Mahalanobis separation: "
            f"m2_pre={scoring.m2_pre:.3f}, m2_post={scoring.m2_post:.3f}."
        )

    utility_positive = scoring.utility > 0.0
    if not utility_positive:
        failures.append(
            f"Utility is non-positive ({scoring.utility:.4f}). "
            "No-burn baseline would have been equally or more optimal."
        )

    secondary_clear = (
        secondary.secondary_check_performed
        and secondary.secondary_conjunction_clear
    )
    if not secondary_clear:
        if not secondary.secondary_check_performed:
            failures.append(
                "Secondary conjunction screen was not performed; safety could not "
                "be established and M4 safe hold is required."
            )
        else:
            failures.append(
                f"Secondary conjunction detected with: "
                f"{', '.join(secondary.flagged_objects)}."
            )

    budget_ok = scoring.dv_total_m_s <= policy.max_dv_per_event_ms
    if not budget_ok:
        failures.append(
            f"Total delta-v {scoring.dv_total_m_s:.3f} m/s exceeds "
            f"policy limit {policy.max_dv_per_event_ms:.3f} m/s."
        )

    recovery_ok = True
    if scoring.recovery_plan is not None:
        if scoring.recovery_plan.recovery_required:
            recovery_ok = (
                scoring.recovery_plan.budget_feasible
                and scoring.recovery_plan.within_max_recovery_time
            )
            if not recovery_ok:
                failures.append(
                    "Slot recovery is not feasible within budget or time limit."
                )

    passed = len(failures) == 0
    delta_m2 = scoring.m2_post - scoring.m2_pre

    if passed:
        note = (
            f"VERIFICATION PASSED. Mahalanobis separation increased by "
            f"{delta_m2:.3f} (m2: {scoring.m2_pre:.3f} -> {scoring.m2_post:.3f}). "
            f"Utility: {scoring.utility:.4f}. Budget and recovery within limits."
        )
    else:
        note = (
            f"VERIFICATION FAILED ({len(failures)} check(s)): "
            + " | ".join(failures)
        )

    if scoring.execution_error_modelled:
        note += (
            " Pass/fail includes execution error covariance (thrust "
            "misalignment, magnitude uncertainty modelled from the "
            "satellite's propulsion profile)."
        )
    else:
        note += (
            " Pass/fail based on nominal burn assumption. Execution error "
            "covariance (thrust misalignment, magnitude uncertainty) not "
            "modelled for this satellite (profile does not specify them). "
            "Verification result may be optimistic."
        )

    return VerificationResult(
        passed=passed,
        risk_reduced=risk_reduced,
        utility_positive=utility_positive,
        secondary_clear=secondary_clear,
        budget_within_limits=budget_ok,
        recovery_within_limits=recovery_ok,
        failure_reasons=failures,
        m2_pre=scoring.m2_pre,
        m2_post=scoring.m2_post,
        delta_m2=delta_m2,
        verification_note=note,
    )


# ---------------------------------------------------------------------------
# MissionImpactSummary
# ---------------------------------------------------------------------------

@dataclass
class MissionImpactSummary:
    """Quantifies how the maneuver affected lifetime, slot, and constellation.

    Attributes
    ----------
    dv_consumed_m_s : float
        Total delta-v consumed by this event [m/s].
    v_remaining_after_m_s : float
        Estimated remaining delta-v budget after this event [m/s].
    lifetime_fraction_before : float
        Mission maturity before this event (0.0 to 1.0).
    lifetime_fraction_after : float
        Estimated mission maturity after this event.
        Approximated as: 1 - (v_remaining_after / v_total), using
        dv as a fuel proxy.
    slot_drift_km : float
        Along-track drift from slot centre induced by avoidance burn [km].
        0.0 for non-constellated satellites.
    slot_tolerance_km : float
        Acceptable drift threshold [km].
    slot_recovery_required : bool
        True if slot_drift_km > slot_tolerance_km.
    slot_recovery_orbits : float
        Minimum recovery orbits. 0.0 if not required.
    constellation_impact_note : str
        Human-readable summary of constellation impact.
    """
    dv_consumed_m_s: float
    v_remaining_after_m_s: float
    lifetime_fraction_before: float
    lifetime_fraction_after: float
    slot_drift_km: float
    slot_tolerance_km: float
    slot_recovery_required: bool
    slot_recovery_orbits: float
    constellation_impact_note: str


def _build_mission_impact(
    scoring: ManeuverScoringResult,
    cap: SatelliteCapability,
) -> MissionImpactSummary:
    """Build MissionImpactSummary from scoring result and satellite capability."""
    dv_consumed = scoring.dv_total_m_s
    v_remaining_after = max(0.0, cap.lifetime.v_remaining_m_s - dv_consumed)
    lf_before = scoring.lifetime_fraction_used

    # Approximate post-event lifetime fraction using dv as fuel proxy
    # If v_remaining drops, remaining lifetime proxy decreases proportionally
    if cap.lifetime.v_remaining_m_s > 0:
        lf_after = max(lf_before, 1.0 - (v_remaining_after / cap.lifetime.v_remaining_m_s) * (1.0 - lf_before))
        lf_after = min(1.0, lf_after)
    else:
        lf_after = 1.0

    recovery_required = False
    recovery_orbits = 0.0
    slot_drift = scoring.post_drift_km
    slot_tol = cap.slot.acceptable_drift_km if cap.slot.in_constellation else 0.0

    if scoring.recovery_plan is not None:
        recovery_required = scoring.recovery_plan.recovery_required
        recovery_orbits = scoring.recovery_plan.recovery_orbits

    if not cap.slot.in_constellation:
        const_note = "Non-constellated satellite. No slot constraints apply."
    elif recovery_required:
        const_note = (
            f"Slot recovery required. Drift: {slot_drift:.3f} km "
            f"(tolerance: {slot_tol:.3f} km). "
            f"Recovery: {recovery_orbits:.0f} orbit(s), "
            f"dv_return={scoring.dv_return_m_s:.3f} m/s."
        )
    else:
        const_note = (
            f"Slot drift {slot_drift:.3f} km within tolerance "
            f"({slot_tol:.3f} km). No slot recovery required."
        )

    return MissionImpactSummary(
        dv_consumed_m_s=dv_consumed,
        v_remaining_after_m_s=v_remaining_after,
        lifetime_fraction_before=lf_before,
        lifetime_fraction_after=lf_after,
        slot_drift_km=slot_drift,
        slot_tolerance_km=slot_tol,
        slot_recovery_required=recovery_required,
        slot_recovery_orbits=recovery_orbits,
        constellation_impact_note=const_note,
    )


# ---------------------------------------------------------------------------
# ATLASManeuverArtifact -- top-level bundle
# ---------------------------------------------------------------------------

@dataclass
class ATLASManeuverArtifact:
    """Complete artifact bundle for ATLAS display, logging, and operator review.

    Assembles all five required sub-artifacts (§5.3) plus 9.5 additions:
    verification result, mission impact summary, and decision log entry.

    All v2.4 fields are preserved on this object for backward compatibility:
    direction, dv_eci_km_s, delta_C, m2_pre, m2_post, utility,
    all_candidates.

    Attributes
    ----------
    conjunction_id : str
    sat_id : str
    evaluated_at : str

    -- Five required artifacts (§5.3) --
    risk_summary : RiskSummary                    (A1)
    recommendation : ManeuverRecommendation       (A2, None if no-go)
    rationale : DecisionRationale                 (A3)
    post_maneuver : PostManeuverProjection         (A4, None if no-go)
    no_go : NoGoReasoning                         (A5, None if maneuver)

    -- 9.5 additions --
    verification : VerificationResult
    mission_impact : MissionImpactSummary

    -- v2.4 preserved fields --
    direction, dv_eci_km_s, delta_C, m2_pre, m2_post, utility,
    all_candidates.
    """
    conjunction_id: str
    sat_id: str
    evaluated_at: str

    # Five required artifacts
    risk_summary: RiskSummary
    recommendation: Optional[ManeuverRecommendation]
    rationale: DecisionRationale
    post_maneuver: Optional[PostManeuverProjection]
    no_go: Optional[NoGoReasoning]

    # 9.5 additions
    verification: VerificationResult
    mission_impact: MissionImpactSummary

    # v2.4 preserved
    direction: str
    dv_eci_km_s: List[float]
    delta_C: float
    m2_pre: float
    m2_post: float
    utility: float
    all_candidates: List[Dict[str, Any]]

    def is_maneuver_recommended(self) -> bool:
        return self.recommendation is not None

    def operator_summary(self) -> str:
        """Single-line verdict suitable for ATLAS dashboard header."""
        if self.is_maneuver_recommended():
            r = self.recommendation
            v = self.verification
            slot_note = ""
            if self.mission_impact.slot_recovery_required:
                slot_note = (
                    f" | Slot recovery: "
                    f"{self.mission_impact.slot_recovery_orbits:.0f} orbit(s)"
                )
            verified = "VERIFIED" if v.passed else "VERIFICATION FAILED"
            return (
                f"MANEUVER RECOMMENDED [{verified}] | "
                f"{r.direction} {r.dv_magnitude_m_s:.3f} m/s | "
                f"delta_C={self.delta_C:.2f} | U={self.utility:.3f}"
                f"{slot_note}"
            )
        return f"NO ACTION | {self.no_go.human_readable}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable dict for ATLAS API and decision log."""

        def _safe(obj):
            """Recursively convert dataclasses and numpy types."""
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _safe(v) for k, v in vars(obj).items()}
            if isinstance(obj, list):
                return [_safe(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _safe(v) for k, v in obj.items()}
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj

        return _safe(self)


# ---------------------------------------------------------------------------
# DecisionLog
# ---------------------------------------------------------------------------

@dataclass
class DecisionLog:
    """JSON-serialisable audit record of a maneuver decision.

    Satisfies the AC: "All verification outputs are logged and retrievable."

    Attributes
    ----------
    log_id : str
        Unique log entry ID: '{conjunction_id}_{sat_id}_{timestamp}'.
    conjunction_id : str
    sat_id : str
    operator_id : str
    policy_version : str
    decision : str
        'maneuver_recommended' | 'no_go'
    reason_code : str
        Empty string for maneuver recommendations.
    direction : str
    dv_total_m_s : float
    utility : float
    verification_passed : bool
    secondary_check_performed : bool
    secondary_conjunction_clear : bool
    covariance_quality : str
    lifetime_fraction_used : float
    slot_recovery_required : bool
    logged_at : str
        ISO-8601 UTC timestamp when the log entry was created.
    artifact_summary : str
        operator_summary() string for quick log review.
    """
    log_id: str
    conjunction_id: str
    sat_id: str
    operator_id: str
    policy_version: str
    decision: str
    reason_code: str
    direction: str
    dv_total_m_s: float
    utility: float
    verification_passed: bool
    secondary_check_performed: bool
    secondary_conjunction_clear: bool
    covariance_quality: str
    lifetime_fraction_used: float
    slot_recovery_required: bool
    logged_at: str
    artifact_summary: str

    def to_json(self) -> str:
        """Serialise to JSON string for storage / retrieval."""
        return json.dumps(vars(self), indent=2)

    @classmethod
    def from_artifact(
        cls,
        artifact: ATLASManeuverArtifact,
        operator_id: str,
        policy_version: str,
    ) -> "DecisionLog":
        """Build a DecisionLog from a completed ATLASManeuverArtifact."""
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        log_id = f"{artifact.conjunction_id}_{artifact.sat_id}_{now}"

        secondary = (
            artifact.post_maneuver.secondary_conflict
            if artifact.post_maneuver is not None
            else SecondaryConflictCheck(
                secondary_check_performed=False,
                secondary_conjunction_clear=True,
                flagged_objects=[],
                operator_note="",
            )
        )

        return cls(
            log_id=log_id,
            conjunction_id=artifact.conjunction_id,
            sat_id=artifact.sat_id,
            operator_id=operator_id,
            policy_version=policy_version,
            decision=(
                "maneuver_recommended"
                if artifact.is_maneuver_recommended()
                else "no_go"
            ),
            reason_code=(
                artifact.no_go.reason_code
                if artifact.no_go is not None
                else ""
            ),
            direction=artifact.direction,
            dv_total_m_s=(
                artifact.recommendation.dv_total_m_s
                if artifact.recommendation is not None
                else 0.0
            ),
            utility=artifact.utility,
            verification_passed=artifact.verification.passed,
            secondary_check_performed=secondary.secondary_check_performed,
            secondary_conjunction_clear=secondary.secondary_conjunction_clear,
            covariance_quality=artifact.risk_summary.covariance_quality,
            lifetime_fraction_used=artifact.rationale.lifetime_fraction_used,
            slot_recovery_required=artifact.mission_impact.slot_recovery_required,
            logged_at=now,
            artifact_summary=artifact.operator_summary(),
        )


# ---------------------------------------------------------------------------
# Assembly function
# ---------------------------------------------------------------------------

def build_atlas_artifact(
    scoring: ManeuverScoringResult,
    cap: SatelliteCapability,
    policy: OperatorPolicy,
    tca_utc: str,
    pc_precomputed: Optional[float] = None,
    miss_distance_km: Optional[float] = None,
    known_objects: Optional[List[Dict[str, Any]]] = None,
    r_post_km: Optional[List[float]] = None,
    v_post_km_s: Optional[List[float]] = None,
) -> ATLASManeuverArtifact:
    """Assemble a complete ATLASManeuverArtifact from a ManeuverScoringResult.

    This is the primary 9.5 entry point. Takes the output of
    score_maneuver_candidates() and assembles all five required artifacts
    plus verification result, mission impact, and the decision log.

    Parameters
    ----------
    scoring : ManeuverScoringResult
        Output from 9.4 score_maneuver_candidates().
    cap : SatelliteCapability
        Satellite capability used for the scoring run.
    policy : OperatorPolicy
        Operator policy used for the scoring run.
    tca_utc : str
        Time of closest approach (ISO-8601 UTC).
    pc_precomputed : float or None
        Pre-computed Pc. None if not available.
    miss_distance_km : float or None
        Miss distance [km]. None if not available.
    known_objects : list[dict] or None
        Known conjunction objects for secondary conflict check.
    r_post_km : list[float] or None
        Post-maneuver satellite position at burn time [km], ECI J2000.
    v_post_km_s : list[float] or None
        Post-maneuver satellite velocity at burn time [km/s], ECI J2000.

    Returns
    -------
    ATLASManeuverArtifact
    """
    # --- A1: RiskSummary ---
    maneuver_required = False
    monitor_only = False
    if pc_precomputed is not None:
        maneuver_required = policy.is_maneuver_required(
            pc_precomputed, miss_distance_km or 999.0
        )
        monitor_only = policy.is_monitor_only(pc_precomputed)

    risk_summary = RiskSummary(
        pc_pre=pc_precomputed,
        miss_distance_km=miss_distance_km,
        mahalanobis_pre=math.sqrt(max(0.0, scoring.m2_pre)),
        tca_utc=tca_utc,
        covariance_quality=scoring.covariance_quality,
        maneuver_required=maneuver_required,
        monitor_only=monitor_only,
    )

    # --- Secondary conflict check ---
    secondary = _run_secondary_conflict_check(
        r_post_km,
        known_objects,
        scoring.t_burn_utc,
        v_post_km_s=v_post_km_s,
        policy=policy,
        cap=cap,
    )

    # --- A4: PostManeuverProjection ---
    post_maneuver: Optional[PostManeuverProjection] = None
    if scoring.is_maneuver_recommended():
        post_maneuver = PostManeuverProjection(
            m2_post=scoring.m2_post,
            mahalanobis_post=math.sqrt(max(0.0, scoring.m2_post)),
            risk_surrogate_post=scoring.risk_surrogate_post,
            slot_recovery_orbits=(
                scoring.recovery_plan.recovery_orbits
                if scoring.recovery_plan is not None else 0.0
            ),
            slot_recovery_time_s=(
                scoring.recovery_plan.recovery_time_s
                if scoring.recovery_plan is not None else 0.0
            ),
            slot_recovery_feasible=(
                scoring.recovery_plan.budget_feasible
                and scoring.recovery_plan.within_max_recovery_time
                if scoring.recovery_plan is not None else True
            ),
            secondary_conflict=secondary,
            recovery_plan=scoring.recovery_plan,
            execution_error_modelled=scoring.execution_error_modelled,
            operator_note=(
                "m2_post includes execution error covariance (thrust misalignment "
                "and magnitude uncertainty modelled from the satellite's propulsion "
                "profile)."
                if scoring.execution_error_modelled else
                "m2_post excludes execution error covariance. Thrust misalignment and "
                "magnitude uncertainty are not modelled for this satellite (profile "
                "does not specify them). Post-maneuver separation may be optimistic."
            ),
        )

    # --- A2: ManeuverRecommendation ---
    recommendation: Optional[ManeuverRecommendation] = None
    if scoring.is_maneuver_recommended():
        recommendation = ManeuverRecommendation(
            direction=scoring.direction,
            dv_eci_km_s=scoring.dv_eci_km_s,
            dv_magnitude_m_s=scoring.dv_magnitude_m_s,
            dv_return_m_s=scoring.dv_return_m_s,
            dv_total_m_s=scoring.dv_total_m_s,
            burn_time_utc=scoring.t_burn_utc,
            drag_correction_applied=scoring.drag_correction_applied,
        )

    # --- A3: DecisionRationale ---
    constraints_applied = []
    if pc_precomputed is not None:
        if maneuver_required:
            constraints_applied.append("pc_threshold_exceeded")
        else:
            constraints_applied.append("pc_below_threshold")
    constraints_applied.append("blackout_windows_checked")
    constraints_applied.append(
        "dv_within_budget"
        if scoring.dv_magnitude_m_s <= policy.max_dv_per_event_ms
        else "dv_exceeds_budget"
    )
    if scoring.covariance_quality == "dilution_region":
        constraints_applied.append("covariance_dilution_region_flagged")
    # SCRUM-365: this flag must reflect what actually happened for this
    # candidate, not a hardcoded claim that execution error is always
    # excluded (which stopped being true once Q_exec was modelled).
    constraints_applied.append(
        "execution_error_modelled"
        if scoring.execution_error_modelled else
        "execution_error_not_modelled"
    )

    # Determine candidate rank of recommended direction
    best_dir = scoring.direction
    rank = 1
    for i, c in enumerate(scoring.all_candidates):
        if c.get("direction") == best_dir:
            rank = i  # 0-indexed; no-burn is index 0
            break

    rationale = DecisionRationale(
        utility_score=scoring.utility,
        delta_C=scoring.delta_C,
        dv_cost_term=(
            scoring.candidates_v25[0].dv_cost_term
            if scoring.candidates_v25 else 0.0
        ),
        lifetime_cost_term=(
            scoring.candidates_v25[0].lifetime_cost_term
            if scoring.candidates_v25 else 0.0
        ),
        slot_cost_term=scoring.slot_cost_term,
        lifetime_fraction_used=scoring.lifetime_fraction_used,
        policy_constraints_applied=constraints_applied,
        candidate_rank=rank,
        n_candidates_evaluated=len(scoring.all_candidates),
        all_candidates=scoring.all_candidates,
        scoring_weights={
            "lambda_dv":            policy.scoring_weights.lambda_dv,
            "lambda_lifetime":      policy.scoring_weights.lambda_lifetime,
            "lambda_slot_deviation": policy.scoring_weights.lambda_slot_deviation,
        },
    )

    # --- A5: NoGoReasoning ---
    no_go: Optional[NoGoReasoning] = None
    if not scoring.is_maneuver_recommended():
        no_go = NoGoReasoning(
            reason_code=scoring.no_go_reason_code,
            human_readable=scoring.no_go_human_readable,
            pc_at_decision=pc_precomputed,
            mahalanobis_at_decision=math.sqrt(max(0.0, scoring.m2_pre)),
        )

    # --- VerificationResult ---
    verification = _build_verification_result(scoring, policy, secondary)

    # --- MissionImpactSummary ---
    mission_impact = _build_mission_impact(scoring, cap)

    return ATLASManeuverArtifact(
        conjunction_id=scoring.conjunction_id,
        sat_id=cap.sat_id,
        evaluated_at=scoring.evaluated_at,
        risk_summary=risk_summary,
        recommendation=recommendation,
        rationale=rationale,
        post_maneuver=post_maneuver,
        no_go=no_go,
        verification=verification,
        mission_impact=mission_impact,
        # v2.4 preserved
        direction=scoring.direction,
        dv_eci_km_s=scoring.dv_eci_km_s,
        delta_C=scoring.delta_C,
        m2_pre=scoring.m2_pre,
        m2_post=scoring.m2_post,
        utility=scoring.utility,
        all_candidates=scoring.all_candidates,
    )
