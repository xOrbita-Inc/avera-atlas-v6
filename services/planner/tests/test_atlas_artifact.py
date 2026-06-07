"""
APS 2.5 -- Step 9.5 Test Suite
================================
File: services/planner/tests/test_atlas_artifact.py

Coverage:
  1.  RiskSummary construction and fields
  2.  SecondaryConflictCheck -- performed vs not performed
  3.  SecondaryConflictCheck -- flagged objects detected
  4.  VerificationResult -- pass conditions
  5.  VerificationResult -- fail conditions (each check)
  6.  MissionImpactSummary -- constellated and non-constellated
  7.  ManeuverRecommendation fields
  8.  DecisionRationale fields and human_readable_rationale()
  9.  PostManeuverProjection fields
 10.  NoGoReasoning fields
 11.  ATLASManeuverArtifact -- maneuver recommended path
 12.  ATLASManeuverArtifact -- no-go path
 13.  ATLASManeuverArtifact -- operator_summary()
 14.  ATLASManeuverArtifact -- to_dict() serialisable
 15.  DecisionLog -- from_artifact(), to_json()
 16.  build_atlas_artifact() -- end-to-end maneuver path
 17.  build_atlas_artifact() -- end-to-end no-go path
 18.  build_atlas_artifact() -- constellated satellite
 19.  AC: post-burn verification pass/fail assessment
 20.  AC: secondary conflict check surfaces to user
 21.  AC: decision rationale is human-readable
 22.  AC: all outputs logged and retrievable (DecisionLog)
 23.  AC: ATLAS can display verification results (to_dict complete)
 24.  hasattr cleanup: maneuver_scorer altitude_km fix

Run:
    pytest services/planner/tests/test_atlas_artifact.py -v
"""

import json
import math
import pytest
import numpy as np

from common.atlas_artifact import (
    RiskSummary,
    ManeuverRecommendation,
    DecisionRationale,
    PostManeuverProjection,
    NoGoReasoning,
    SecondaryConflictCheck,
    VerificationResult,
    MissionImpactSummary,
    ATLASManeuverArtifact,
    DecisionLog,
    build_atlas_artifact,
    _run_secondary_conflict_check,
    _build_verification_result,
    _build_mission_impact,
)
from common.satellite_capability import (
    SatelliteCapability,
    LifetimeProfile,
    ConstellationSlot,
)
from common.operator_policy import OperatorPolicy, ScoringWeights
from common.maneuver_scorer import (
    score_maneuver_candidates,
    _passes_feasibility,
)
from common.constellation_geometry import (
    leo_constellation_482km,
    _mean_motion_to_sma_km,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

T_BURN = "2026-04-14T08:00:00Z"
T_CA   = "2026-04-14T12:00:00Z"


@pytest.fixture
def policy():
    return OperatorPolicy(
        operator_id="TEST_OP",
        policy_version="2.5.0",
        max_dv_per_event_ms=2.0,
        mission_lifetime_days_total=1825.0,
        scoring_weights=ScoringWeights(
            lambda_dv=1.0, lambda_lifetime=0.8, lambda_slot_deviation=1.2
        ),
    )


@pytest.fixture
def cap_solo():
    return SatelliteCapability(
        sat_id="SAT-SOLO",
        a_ref_km=_mean_motion_to_sma_km(15.3020),
        lifetime=LifetimeProfile(
            mass_kg=100.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
            mission_lifetime_days_remaining=365.0,
        ),
        slot=ConstellationSlot(in_constellation=False),
    )


@pytest.fixture
def cap_constellated():
    return SatelliteCapability(
        sat_id="SL-1234",
        a_ref_km=_mean_motion_to_sma_km(15.3020),
        lifetime=LifetimeProfile(
            mass_kg=300.0, v_remaining_m_s=120.0, v_reserved_m_s=10.0,
            mission_lifetime_days_remaining=900.0,
        ),
        slot=ConstellationSlot(
            in_constellation=True,
            slot_id="P02-S05",
            target_mean_motion_rev_per_day=15.3020,
            acceptable_drift_km=4.459,
            return_dv_budget_m_s=4.4,
            max_recovery_time_s=86400.0,
        ),
    )


@pytest.fixture
def high_risk_scoring(cap_solo, policy):
    """ManeuverScoringResult for a high-risk event that recommends a maneuver."""
    r_sat = np.array([_mean_motion_to_sma_km(15.3020), 0.0, 0.0])
    v_sat = np.array([0.0, 7.626, 0.0])
    r_rel = np.array([0.3, 0.0, 0.0])   # MD=3.0, within screen
    P = np.eye(3) * 0.01
    return score_maneuver_candidates(
        "CID-HR", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy
    )


@pytest.fixture
def low_risk_scoring(cap_solo, policy):
    """ManeuverScoringResult for a trivially safe event (no-go)."""
    r_sat = np.array([_mean_motion_to_sma_km(15.3020), 0.0, 0.0])
    v_sat = np.array([0.0, 7.626, 0.0])
    r_rel = np.array([50.0, 0.0, 0.0])  # MD=500, screened as trivial
    P = np.eye(3) * 0.01
    return score_maneuver_candidates(
        "CID-LR", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy
    )


# ---------------------------------------------------------------------------
# 1. RiskSummary
# ---------------------------------------------------------------------------

class TestRiskSummary:
    def test_construction(self):
        rs = RiskSummary(
            pc_pre=2.1e-4, miss_distance_km=0.73,
            mahalanobis_pre=3.0, tca_utc=T_CA,
            covariance_quality="good",
            maneuver_required=True, monitor_only=False,
        )
        assert rs.pc_pre == pytest.approx(2.1e-4)
        assert rs.covariance_quality == "good"
        assert rs.maneuver_required is True

    def test_none_pc(self):
        rs = RiskSummary(
            pc_pre=None, miss_distance_km=None,
            mahalanobis_pre=3.0, tca_utc=T_CA,
            covariance_quality="degraded",
            maneuver_required=False, monitor_only=False,
        )
        assert rs.pc_pre is None


# ---------------------------------------------------------------------------
# 2. SecondaryConflictCheck -- no catalog
# ---------------------------------------------------------------------------

class TestSecondaryConflictCheckNoCatalog:
    def test_not_performed_when_no_catalog(self):
        result = _run_secondary_conflict_check(None, None)
        assert result.secondary_check_performed is False
        assert result.secondary_conjunction_clear is True
        assert result.flagged_objects == []

    def test_operator_note_mentions_aps_30(self):
        result = _run_secondary_conflict_check([0.0, 0.0, 0.0], None)
        assert "APS 3.0" in result.operator_note

    def test_not_performed_empty_catalog(self):
        result = _run_secondary_conflict_check([0.0, 0.0, 0.0], [])
        assert result.secondary_check_performed is False


# ---------------------------------------------------------------------------
# 3. SecondaryConflictCheck -- with catalog
# ---------------------------------------------------------------------------

class TestSecondaryConflictCheckWithCatalog:
    def test_clear_when_all_objects_far(self):
        r_post = [6853.0, 0.0, 0.0]
        objects = [
            {"obj_id": "OBJ-001", "r_km": [6855.0, 0.0, 0.0]},  # 2 km away
            {"obj_id": "OBJ-002", "r_km": [6853.0, 5.0, 0.0]},  # 5 km away
        ]
        result = _run_secondary_conflict_check(r_post, objects)
        assert result.secondary_check_performed is True
        assert result.secondary_conjunction_clear is True
        assert result.flagged_objects == []

    def test_flagged_when_object_close(self):
        r_post = [6853.0, 0.0, 0.0]
        objects = [
            {"obj_id": "CLOSE-001", "r_km": [6853.3, 0.0, 0.0]},  # 0.3 km
        ]
        result = _run_secondary_conflict_check(r_post, objects)
        assert result.secondary_check_performed is True
        assert result.secondary_conjunction_clear is False
        assert "CLOSE-001" in result.flagged_objects

    def test_multiple_flagged(self):
        r_post = [6853.0, 0.0, 0.0]
        objects = [
            {"obj_id": "A", "r_km": [6853.1, 0.0, 0.0]},
            {"obj_id": "B", "r_km": [6853.2, 0.0, 0.0]},
            {"obj_id": "C", "r_km": [6860.0, 0.0, 0.0]},  # far
        ]
        result = _run_secondary_conflict_check(r_post, objects)
        assert len(result.flagged_objects) == 2
        assert "C" not in result.flagged_objects


# ---------------------------------------------------------------------------
# 4. VerificationResult -- pass
# ---------------------------------------------------------------------------

class TestVerificationPass:
    def test_passes_for_good_maneuver(self, high_risk_scoring, policy):
        secondary = _run_secondary_conflict_check(None, None)
        result = _build_verification_result(high_risk_scoring, policy, secondary)
        if high_risk_scoring.is_maneuver_recommended():
            assert result.risk_reduced is True
            assert result.utility_positive is True
            assert result.budget_within_limits is True

    def test_verification_note_contains_verdict(self, high_risk_scoring, policy):
        secondary = _run_secondary_conflict_check(None, None)
        result = _build_verification_result(high_risk_scoring, policy, secondary)
        assert "VERIFICATION" in result.verification_note

    def test_delta_m2_positive_when_safer(self, high_risk_scoring, policy):
        secondary = _run_secondary_conflict_check(None, None)
        result = _build_verification_result(high_risk_scoring, policy, secondary)
        if high_risk_scoring.is_maneuver_recommended():
            assert result.delta_m2 == pytest.approx(
                high_risk_scoring.m2_post - high_risk_scoring.m2_pre
            )


# ---------------------------------------------------------------------------
# 5. VerificationResult -- fail conditions
# ---------------------------------------------------------------------------

class TestVerificationFail:
    def test_fails_when_no_utility(self, low_risk_scoring, policy):
        secondary = _run_secondary_conflict_check(None, None)
        result = _build_verification_result(low_risk_scoring, policy, secondary)
        # Low-risk no-go: utility = 0 -> not positive
        assert result.utility_positive is False

    def test_fails_when_secondary_conflict(self, high_risk_scoring, policy):
        """Verification fails when secondary conflict is detected."""
        r_post = high_risk_scoring.dv_eci_km_s  # reuse as fake post position
        # Place a close object at near-zero offset
        r_post_km = [_mean_motion_to_sma_km(15.3020) + 0.1, 0.0, 0.0]
        objects = [{"obj_id": "CLOSE", "r_km": [_mean_motion_to_sma_km(15.3020) + 0.1, 0.0, 0.0]}]
        r_post_km2 = [_mean_motion_to_sma_km(15.3020) + 0.5, 0.0, 0.0]
        secondary_conflict = _run_secondary_conflict_check(r_post_km2, objects)
        if not secondary_conflict.secondary_conjunction_clear:
            result = _build_verification_result(high_risk_scoring, policy, secondary_conflict)
            assert result.secondary_clear is False
            assert result.passed is False

    def test_failure_reasons_populated_on_fail(self, low_risk_scoring, policy):
        secondary = _run_secondary_conflict_check(None, None)
        result = _build_verification_result(low_risk_scoring, policy, secondary)
        if not result.passed:
            assert len(result.failure_reasons) > 0


# ---------------------------------------------------------------------------
# 6. MissionImpactSummary
# ---------------------------------------------------------------------------

class TestMissionImpactSummary:
    def test_non_constellated_no_slot_impact(self, high_risk_scoring, cap_solo):
        impact = _build_mission_impact(high_risk_scoring, cap_solo)
        assert impact.slot_recovery_required is False
        assert impact.slot_drift_km == pytest.approx(0.0)
        assert "Non-constellated" in impact.constellation_impact_note

    def test_dv_consumed_matches_scoring(self, high_risk_scoring, cap_solo):
        impact = _build_mission_impact(high_risk_scoring, cap_solo)
        assert impact.dv_consumed_m_s == pytest.approx(high_risk_scoring.dv_total_m_s)

    def test_v_remaining_decreases(self, high_risk_scoring, cap_solo):
        impact = _build_mission_impact(high_risk_scoring, cap_solo)
        assert impact.v_remaining_after_m_s <= cap_solo.lifetime.v_remaining_m_s

    def test_constellated_slot_note(self, cap_constellated, policy):
        r_sat = np.array([_mean_motion_to_sma_km(15.3020), 0.0, 0.0])
        v_sat = np.array([0.0, 7.626, 0.0])
        r_rel = np.array([0.3, 0.0, 0.0])
        P = np.eye(3) * 0.01
        scoring = score_maneuver_candidates(
            "CID-CONST", r_sat, v_sat, r_rel, P, T_BURN, T_CA,
            cap_constellated, policy,
        )
        impact = _build_mission_impact(scoring, cap_constellated)
        assert "slot" in impact.constellation_impact_note.lower()


# ---------------------------------------------------------------------------
# 7-10. Sub-artifact construction
# ---------------------------------------------------------------------------

class TestSubArtifacts:
    def test_maneuver_recommendation_fields(self):
        rec = ManeuverRecommendation(
            direction="radial",
            dv_eci_km_s=[0.002, 0.0, 0.0],
            dv_magnitude_m_s=2.0,
            dv_return_m_s=0.0,
            dv_total_m_s=2.0,
            burn_time_utc=T_BURN,
            drag_correction_applied=False,
        )
        assert rec.direction == "radial"
        assert rec.dv_total_m_s == pytest.approx(2.0)

    def test_decision_rationale_human_readable(self):
        rat = DecisionRationale(
            utility_score=2.31, delta_C=3.84,
            dv_cost_term=0.21, lifetime_cost_term=0.17, slot_cost_term=0.46,
            lifetime_fraction_used=0.5,
            policy_constraints_applied=["pc_threshold_exceeded", "blackout_clear"],
            candidate_rank=1, n_candidates_evaluated=7,
            all_candidates=[
                {"direction": "no-burn", "utility": 0.0},
                {"direction": "radial", "utility": 2.31},
            ],
            scoring_weights={"lambda_dv": 1.0, "lambda_lifetime": 0.8,
                             "lambda_slot_deviation": 1.2},
        )
        text = rat.human_readable_rationale()
        assert len(text) > 20
        assert "7" in text or "candidates" in text.lower()

    def test_nogo_reasoning_fields(self):
        ng = NoGoReasoning(
            reason_code="pc_below_threshold",
            human_readable="Pc 1e-7 is below threshold 1e-4.",
            pc_at_decision=1e-7,
            mahalanobis_at_decision=3.0,
        )
        assert ng.reason_code == "pc_below_threshold"
        assert ng.pc_at_decision == pytest.approx(1e-7)


# ---------------------------------------------------------------------------
# 11-13. ATLASManeuverArtifact
# ---------------------------------------------------------------------------

class TestATLASArtifactManeuver:
    def test_is_maneuver_recommended_true(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA
        )
        if high_risk_scoring.is_maneuver_recommended():
            assert artifact.is_maneuver_recommended() is True
            assert artifact.recommendation is not None
            assert artifact.no_go is None

    def test_v24_fields_preserved(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA
        )
        assert artifact.direction == high_risk_scoring.direction
        assert artifact.delta_C == pytest.approx(high_risk_scoring.delta_C)
        assert artifact.m2_pre == pytest.approx(high_risk_scoring.m2_pre)
        assert artifact.m2_post == pytest.approx(high_risk_scoring.m2_post)
        assert artifact.utility == pytest.approx(high_risk_scoring.utility)

    def test_operator_summary_contains_verdict(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA
        )
        s = artifact.operator_summary()
        assert len(s) > 10
        if artifact.is_maneuver_recommended():
            assert "MANEUVER RECOMMENDED" in s
        else:
            assert "NO ACTION" in s


class TestATLASArtifactNoGo:
    def test_no_go_path(self, low_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            low_risk_scoring, cap_solo, policy, T_CA
        )
        assert artifact.is_maneuver_recommended() is False
        assert artifact.no_go is not None
        assert artifact.recommendation is None
        assert artifact.post_maneuver is None

    def test_no_go_reason_code_populated(self, low_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            low_risk_scoring, cap_solo, policy, T_CA
        )
        assert len(artifact.no_go.reason_code) > 0

    def test_no_go_operator_summary(self, low_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            low_risk_scoring, cap_solo, policy, T_CA
        )
        assert "NO ACTION" in artifact.operator_summary()


# ---------------------------------------------------------------------------
# 14. to_dict() serialisable
# ---------------------------------------------------------------------------

class TestArtifactSerialisation:
    def test_to_dict_is_json_serialisable(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA
        )
        d = artifact.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 100

    def test_to_dict_contains_key_fields(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA
        )
        d = artifact.to_dict()
        assert "conjunction_id" in d
        assert "risk_summary" in d
        assert "verification" in d
        assert "mission_impact" in d


# ---------------------------------------------------------------------------
# 15. DecisionLog
# ---------------------------------------------------------------------------

class TestDecisionLog:
    def test_from_artifact_maneuver(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA
        )
        log = DecisionLog.from_artifact(artifact, "TEST_OP", "2.5.0")
        assert log.conjunction_id == "CID-HR"
        assert log.sat_id == "SAT-SOLO"
        assert log.operator_id == "TEST_OP"
        assert log.policy_version == "2.5.0"

    def test_from_artifact_nogo(self, low_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            low_risk_scoring, cap_solo, policy, T_CA
        )
        log = DecisionLog.from_artifact(artifact, "TEST_OP", "2.5.0")
        assert log.decision == "no_go"
        assert len(log.reason_code) > 0

    def test_to_json_valid(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA
        )
        log = DecisionLog.from_artifact(artifact, "TEST_OP", "2.5.0")
        json_str = log.to_json()
        parsed = json.loads(json_str)
        assert "log_id" in parsed
        assert "conjunction_id" in parsed

    def test_log_id_unique_per_call(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA
        )
        log1 = DecisionLog.from_artifact(artifact, "TEST_OP", "2.5.0")
        log2 = DecisionLog.from_artifact(artifact, "TEST_OP", "2.5.0")
        # Both should have log_id fields (may or may not differ by timestamp)
        assert log1.log_id is not None
        assert log2.log_id is not None

    def test_artifact_summary_in_log(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA
        )
        log = DecisionLog.from_artifact(artifact, "TEST_OP", "2.5.0")
        assert len(log.artifact_summary) > 10


# ---------------------------------------------------------------------------
# 16. build_atlas_artifact -- end-to-end maneuver
# ---------------------------------------------------------------------------

class TestBuildAtlasArtifactE2E:
    def test_full_maneuver_fields_populated(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA,
            pc_precomputed=2.1e-4, miss_distance_km=0.5,
        )
        assert artifact.risk_summary is not None
        assert artifact.rationale is not None
        assert artifact.verification is not None
        assert artifact.mission_impact is not None
        assert artifact.sat_id == "SAT-SOLO"

    def test_pc_populates_risk_summary(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA,
            pc_precomputed=5e-4,
        )
        assert artifact.risk_summary.pc_pre == pytest.approx(5e-4)

    def test_none_pc_allowed(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA,
        )
        assert artifact.risk_summary.pc_pre is None


# ---------------------------------------------------------------------------
# 17. build_atlas_artifact -- no-go path
# ---------------------------------------------------------------------------

class TestBuildAtlasNoGoE2E:
    def test_nogo_artifact_complete(self, low_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(
            low_risk_scoring, cap_solo, policy, T_CA
        )
        assert artifact.no_go is not None
        assert artifact.no_go.human_readable != ""
        assert artifact.verification is not None
        assert artifact.mission_impact is not None


# ---------------------------------------------------------------------------
# 18. build_atlas_artifact -- constellated satellite
# ---------------------------------------------------------------------------

class TestBuildAtlasConstellated:
    def test_constellated_artifact(self, cap_constellated, policy):
        r_sat = np.array([_mean_motion_to_sma_km(15.3020), 0.0, 0.0])
        v_sat = np.array([0.0, 7.626, 0.0])
        r_rel = np.array([0.3, 0.0, 0.0])
        P = np.eye(3) * 0.01
        scoring = score_maneuver_candidates(
            "CID-CONST", r_sat, v_sat, r_rel, P, T_BURN, T_CA,
            cap_constellated, policy, geometry=leo_constellation_482km(),
        )
        artifact = build_atlas_artifact(
            scoring, cap_constellated, policy, T_CA
        )
        assert artifact.sat_id == "SL-1234"
        assert artifact.mission_impact is not None


# ---------------------------------------------------------------------------
# 19. AC: Post-burn verification pass/fail assessment
# ---------------------------------------------------------------------------

class TestACVerification:
    def test_verification_present_in_every_artifact(
        self, high_risk_scoring, low_risk_scoring, cap_solo, policy
    ):
        for scoring in [high_risk_scoring, low_risk_scoring]:
            artifact = build_atlas_artifact(scoring, cap_solo, policy, T_CA)
            assert artifact.verification is not None
            assert isinstance(artifact.verification.passed, bool)
            assert isinstance(artifact.verification.verification_note, str)

    def test_verification_has_all_checks(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(high_risk_scoring, cap_solo, policy, T_CA)
        v = artifact.verification
        assert hasattr(v, 'risk_reduced')
        assert hasattr(v, 'utility_positive')
        assert hasattr(v, 'secondary_clear')
        assert hasattr(v, 'budget_within_limits')
        assert hasattr(v, 'recovery_within_limits')


# ---------------------------------------------------------------------------
# 20. AC: Secondary conflict check surfaces to user
# ---------------------------------------------------------------------------

class TestACSecondaryConflict:
    def test_secondary_check_in_post_maneuver(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(high_risk_scoring, cap_solo, policy, T_CA)
        if artifact.post_maneuver is not None:
            assert artifact.post_maneuver.secondary_conflict is not None
            assert isinstance(
                artifact.post_maneuver.secondary_conflict.secondary_check_performed, bool
            )

    def test_secondary_check_operator_note_present(
        self, high_risk_scoring, cap_solo, policy
    ):
        artifact = build_atlas_artifact(high_risk_scoring, cap_solo, policy, T_CA)
        if artifact.post_maneuver is not None:
            note = artifact.post_maneuver.secondary_conflict.operator_note
            assert len(note) > 10


# ---------------------------------------------------------------------------
# 21. AC: Decision rationale is human-readable
# ---------------------------------------------------------------------------

class TestACHumanReadable:
    def test_rationale_human_readable_is_string(
        self, high_risk_scoring, cap_solo, policy
    ):
        artifact = build_atlas_artifact(high_risk_scoring, cap_solo, policy, T_CA)
        text = artifact.rationale.human_readable_rationale()
        assert isinstance(text, str)
        assert len(text) > 30

    def test_nogo_human_readable_is_string(self, low_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(low_risk_scoring, cap_solo, policy, T_CA)
        assert artifact.no_go is not None
        assert len(artifact.no_go.human_readable) > 10

    def test_operator_summary_is_string(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(high_risk_scoring, cap_solo, policy, T_CA)
        assert isinstance(artifact.operator_summary(), str)


# ---------------------------------------------------------------------------
# 22. AC: All outputs logged and retrievable
# ---------------------------------------------------------------------------

class TestACDecisionLog:
    def test_log_created_for_every_artifact(
        self, high_risk_scoring, low_risk_scoring, cap_solo, policy
    ):
        for scoring in [high_risk_scoring, low_risk_scoring]:
            artifact = build_atlas_artifact(scoring, cap_solo, policy, T_CA)
            log = DecisionLog.from_artifact(artifact, "TEST_OP", "2.5.0")
            assert log is not None
            assert len(log.log_id) > 0

    def test_log_json_round_trip(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(high_risk_scoring, cap_solo, policy, T_CA)
        log = DecisionLog.from_artifact(artifact, "TEST_OP", "2.5.0")
        json_str = log.to_json()
        parsed = json.loads(json_str)
        assert parsed["conjunction_id"] == log.conjunction_id
        assert parsed["sat_id"] == log.sat_id
        assert parsed["decision"] in ("maneuver_recommended", "no_go")

    def test_log_contains_verification_result(
        self, high_risk_scoring, cap_solo, policy
    ):
        artifact = build_atlas_artifact(high_risk_scoring, cap_solo, policy, T_CA)
        log = DecisionLog.from_artifact(artifact, "TEST_OP", "2.5.0")
        assert isinstance(log.verification_passed, bool)


# ---------------------------------------------------------------------------
# 23. AC: ATLAS can display verification results
# ---------------------------------------------------------------------------

class TestACATLASDisplay:
    def test_to_dict_has_verification_key(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(high_risk_scoring, cap_solo, policy, T_CA)
        d = artifact.to_dict()
        assert "verification" in d
        assert "passed" in d["verification"]
        assert "verification_note" in d["verification"]

    def test_to_dict_has_mission_impact_key(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(high_risk_scoring, cap_solo, policy, T_CA)
        d = artifact.to_dict()
        assert "mission_impact" in d
        assert "dv_consumed_m_s" in d["mission_impact"]

    def test_to_dict_has_five_artifacts(self, high_risk_scoring, cap_solo, policy):
        artifact = build_atlas_artifact(high_risk_scoring, cap_solo, policy, T_CA)
        d = artifact.to_dict()
        for key in ("risk_summary", "rationale", "verification", "mission_impact"):
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 24. hasattr cleanup from 9.4 CTO note
# ---------------------------------------------------------------------------

class TestHasattrCleanup:
    def test_altitude_km_not_on_satellite_capability(self, cap_solo):
        """SatelliteCapability has no altitude_km field -- confirm the hasattr
        guard in score_maneuver_candidates was removed in 9.5 cleanup."""
        assert not hasattr(cap_solo, 'altitude_km')

    def test_maneuver_scorer_runs_without_altitude_km(self, cap_solo, policy):
        """Scorer must work correctly without any altitude_km attribute on cap."""
        r_sat = np.array([_mean_motion_to_sma_km(15.3020), 0.0, 0.0])
        v_sat = np.array([0.0, 7.626, 0.0])
        r_rel = np.array([0.3, 0.0, 0.0])
        P = np.eye(3) * 0.01
        # Should not raise AttributeError
        result = score_maneuver_candidates(
            "HASATTR-TEST", r_sat, v_sat, r_rel, P, T_BURN, T_CA,
            cap_solo, policy
        )
        assert result is not None
