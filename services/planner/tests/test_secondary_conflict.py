"""
tests/test_secondary_conflict.py

SCRUM-381: Unit tests for fail-closed secondary horizon Pc screening.

Tests cover:
  - strict Pc_action CLEAR / NOT CLEAR semantics
  - policy-owned screening horizon
  - continuously refined TCA between 60-second brackets
  - fail-closed missing-input, propagation, and Pc-computation paths
  - VerificationResult.secondary_clear propagation

Uses synthetic TLE pairs and known post-burn states so tests are
deterministic and require no network access.

Run with (from repo root):
    python -m pytest services/planner/tests/test_secondary_conflict.py -v
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from common.atlas_artifact import (
    SecondaryConflictCheck,
    VerificationResult,
    _run_secondary_conflict_check,
    _build_verification_result,
)
from common.secondary_horizon import _find_object_tca, screen_secondary_catalog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scoring_stub(
    m2_pre: float = 8.0,
    m2_post: float = 15.0,
    utility: float = 3.5,
    dv_total_m_s: float = 1.0,
    recovery_plan=None,
):
    """Minimal ManeuverScoringResult-like stub for VerificationResult tests."""
    stub = MagicMock()
    stub.m2_pre = m2_pre
    stub.m2_post = m2_post
    stub.utility = utility
    stub.dv_total_m_s = dv_total_m_s
    stub.recovery_plan = recovery_plan
    stub.direction = "prograde"
    return stub


def _make_policy_stub(max_dv: float = 2.0):
    stub = MagicMock()
    stub.max_dv_per_event_ms = max_dv
    return stub


# ---------------------------------------------------------------------------
# Real ISS TLE (2024 epoch -- deterministic, publicly known)
# Used for propagation tests. Not fetched from network.
# ---------------------------------------------------------------------------

_ISS_TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00010000  00000-0  17814-3 0  9990"
_ISS_TLE_LINE2 = "2 25544  51.6400 337.6640 0001234  84.4096 275.7258 15.50000000440102"
_ISS_NORAD = "25544"

# A synthetic nearby object TLE in almost the same orbit as ISS
# Placed ~2 km ahead in the along-track direction
_NEARBY_TLE_LINE1 = "1 99001U 98067Z   24001.50000000  .00010000  00000-0  17814-3 0  9991"
_NEARBY_TLE_LINE2 = "2 99001  51.6400 337.6640 0001234  84.4096 275.7300 15.50000000440103"

# A synthetic far object TLE in a completely different orbit
_FAR_TLE_LINE1 = "1 99002U 99025A   24001.50000000  .00000000  00000-0  00000-0 0  9992"
_FAR_TLE_LINE2 = "2 99002  98.0000  20.0000 0010000  90.0000 270.0000 14.20000000000001"


# ---------------------------------------------------------------------------
# SCRUM-381: strict horizon Pc contract
# ---------------------------------------------------------------------------

class TestSecondaryHorizonPcContract:
    @pytest.mark.parametrize(
        ("pc_value", "expected_clear"),
        [
            (9.0e-5, True),
            (1.0e-4, False),
            (1.1e-4, False),
        ],
    )
    def test_pc_action_boundary_is_strict(self, pc_value, expected_clear):
        # CLEAR requires every Pc to be strictly below Pc_action.
        state = (
            37.5,
            0.25,
            np.array([7000.0, 0.0, 0.0]),
            np.array([0.0, 7.5, 0.0]),
            np.array([7000.0, 0.25, 0.0]),
            np.array([0.0, 7.5, 0.0]),
        )
        with patch(
            "common.secondary_horizon._find_object_tca",
            return_value=state,
        ), patch("common.secondary_horizon.compute_pc") as pc_mock:
            pc_mock.return_value = MagicMock(Pc=pc_value)
            result = screen_secondary_catalog(
                r_post_km=[7000.0, 0.0, 0.0],
                v_post_km_s=[0.0, 7.5, 0.0],
                known_objects=[{"obj_id": "OBJ-1", "r_km": [7000.0, 1.0, 0.0], "v_km_s": [0.0, 7.5, 0.0]}],
                burn_time_utc="2026-08-24T12:00:00Z",
                horizon_hours=12.0,
                pc_action=1.0e-4,
                primary_radius_m=0.6,
            )

        assert result["secondary_check_performed"] is True
        assert result["secondary_conjunction_clear"] is expected_clear
        assert ("OBJ-1" in result["flagged_objects"]) is (not expected_clear)

    def test_closest_approach_is_from_each_objects_horizon_tca(self):
        # Closest-object reporting uses refined horizon TCA results.
        states = [
            (
                15.0,
                2.0,
                np.array([7000.0, 0.0, 0.0]),
                np.array([0.0, 7.5, 0.0]),
                np.array([7002.0, 0.0, 0.0]),
                np.array([0.0, 7.5, 0.0]),
            ),
            (
                75.0,
                0.4,
                np.array([7000.0, 0.0, 0.0]),
                np.array([0.0, 7.5, 0.0]),
                np.array([7000.4, 0.0, 0.0]),
                np.array([0.0, 7.5, 0.0]),
            ),
        ]
        with patch(
            "common.secondary_horizon._find_object_tca",
            side_effect=states,
        ), patch("common.secondary_horizon.compute_pc") as pc_mock:
            pc_mock.return_value = MagicMock(Pc=1.0e-6)
            result = screen_secondary_catalog(
                r_post_km=[7000.0, 0.0, 0.0],
                v_post_km_s=[0.0, 7.5, 0.0],
                known_objects=[{"obj_id": "FAR", "r_km": [7002.0, 0.0, 0.0], "v_km_s": [0.0, 7.5, 0.0]}, {"obj_id": "NEAR", "r_km": [7000.4, 0.0, 0.0], "v_km_s": [0.0, 7.5, 0.0]}],
                burn_time_utc="2026-08-24T12:00:00Z",
                horizon_hours=6.0,
                pc_action=1.0e-4,
                primary_radius_m=0.6,
            )

        assert result["closest_object_id"] == "NEAR"
        assert result["closest_approach_km"] == pytest.approx(0.4)
        assert result["screening_epoch_utc"] == "2026-08-24T12:00:00Z"


class TestHorizonTcaRefinement:
    def test_refines_tca_between_sixty_second_brackets(self):
        # A TCA at T+37.5 s must not collapse to the 0 s or 60 s grid.
        def fake_relative_state(_r0, _v0, _obj, _epoch, dt_s):
            r_primary = np.zeros(3)
            v_primary = np.zeros(3)
            r_secondary = np.array([float(dt_s) - 37.5, 0.2, 0.0])
            v_secondary = np.array([1.0, 0.0, 0.0])
            return r_primary, v_primary, r_secondary, v_secondary

        with patch(
            "common.secondary_horizon._relative_state",
            side_effect=fake_relative_state,
        ):
            dt_s, miss_km, *_ = _find_object_tca(
                np.array([7000.0, 0.0, 0.0]),
                np.array([0.0, 7.5, 0.0]),
                {"obj_id": "BETWEEN-GRID"},
                datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc),
                120.0,
            )

        assert dt_s == pytest.approx(37.5, abs=1e-5)
        assert miss_km == pytest.approx(0.2, abs=1e-8)


# ---------------------------------------------------------------------------
# AC4: Catalog unavailability fallback
# ---------------------------------------------------------------------------

class TestCatalogUnavailabilityFallback:
    """AC4: No catalog available falls back to not_performed, no crash."""

    def test_not_performed_when_known_objects_none(self):
        """None known_objects must return not_performed."""
        result = _run_secondary_conflict_check([6778.0, 0.0, 0.0], None)
        assert result.secondary_check_performed is False
        assert result.secondary_conjunction_clear is False

    def test_not_performed_when_r_post_none(self):
        """None r_post_km must return not_performed."""
        result = _run_secondary_conflict_check(None, [{"obj_id": "X", "r_km": [0, 0, 0]}])
        assert result.secondary_check_performed is False

    def test_not_performed_when_both_none(self):
        """Both None must return not_performed without raising."""
        result = _run_secondary_conflict_check(None, None)
        assert result.secondary_check_performed is False

    def test_not_performed_operator_note_present(self):
        """not_performed result must have a non-empty operator_note."""
        result = _run_secondary_conflict_check(None, None)
        assert len(result.operator_note) > 0


# ---------------------------------------------------------------------------
# SCRUM-381: full-input gate and fail-closed routing
# ---------------------------------------------------------------------------

class TestSecondaryHorizonGate:
    @staticmethod
    def _policy(horizon_hours=18.5, pc_action=1.0e-4):
        policy = MagicMock()
        policy.max_hours_before_tca = horizon_hours
        policy.pc_maneuver_threshold = pc_action
        return policy

    @staticmethod
    def _cap(radius_m=0.6):
        cap = MagicMock()
        cap.radius_m = radius_m
        return cap

    @staticmethod
    def _success_result():
        return {
            "secondary_check_performed": True,
            "secondary_conjunction_clear": True,
            "flagged_objects": [],
            "operator_note": "CLEAR",
            "closest_approach_km": 2.5,
            "closest_object_id": "OBJ-1",
            "screening_epoch_utc": "2026-08-24T12:00:00Z",
        }

    def test_policy_max_hours_before_tca_owns_horizon(self):
        # The screen consumes policy horizon and Pc threshold, not literals.
        policy = self._policy(horizon_hours=18.5, pc_action=7.5e-5)
        cap = self._cap(radius_m=0.75)
        with patch(
            "common.atlas_artifact.screen_secondary_catalog",
            return_value=self._success_result(),
        ) as screen:
            result = _run_secondary_conflict_check(
                [7000.0, 0.0, 0.0],
                [{"obj_id": "OBJ-1", "r_km": [7000.0, 1.0, 0.0], "v_km_s": [0.0, 7.5, 0.0]}],
                "2026-08-24T12:00:00Z",
                [0.0, 7.5, 0.0],
                policy,
                cap,
            )

        assert result.secondary_check_performed is True
        kwargs = screen.call_args.kwargs
        assert kwargs["horizon_hours"] == pytest.approx(18.5)
        assert kwargs["pc_action"] == pytest.approx(7.5e-5)
        assert kwargs["primary_radius_m"] == pytest.approx(0.75)

    def test_missing_velocity_does_not_fall_back_to_proximity_proxy(self):
        result = _run_secondary_conflict_check(
            [7000.0, 0.0, 0.0],
            [{"obj_id": "OBJ-1", "r_km": [7000.1, 0.0, 0.0]}],
            "2026-08-24T12:00:00Z",
            None,
            self._policy(),
            self._cap(),
        )
        assert result.secondary_check_performed is False
        assert result.secondary_conjunction_clear is False
        assert "post-burn velocity" in result.operator_note
        assert "M4" in result.operator_note

    def test_propagation_failure_fails_closed_to_m4(self):
        with patch(
            "common.atlas_artifact.screen_secondary_catalog",
            side_effect=RuntimeError("propagation failed"),
        ):
            result = _run_secondary_conflict_check(
                [7000.0, 0.0, 0.0],
                [{"obj_id": "OBJ-1", "r_km": [7000.0, 1.0, 0.0], "v_km_s": [0.0, 7.5, 0.0]}],
                "2026-08-24T12:00:00Z",
                [0.0, 7.5, 0.0],
                self._policy(),
                self._cap(),
            )

        assert result.secondary_check_performed is False
        assert result.secondary_conjunction_clear is False
        assert "propagation failed" in result.operator_note
        assert "M4" in result.operator_note

    def test_pc_computation_failure_fails_closed_to_m4(self):
        state = (
            30.0,
            0.5,
            np.array([7000.0, 0.0, 0.0]),
            np.array([0.0, 7.5, 0.0]),
            np.array([7000.5, 0.0, 0.0]),
            np.array([0.0, 7.5, 0.0]),
        )
        with patch(
            "common.secondary_horizon._find_object_tca",
            return_value=state,
        ), patch(
            "common.secondary_horizon.compute_pc",
            side_effect=RuntimeError("Pc failed"),
        ):
            result = _run_secondary_conflict_check(
                [7000.0, 0.0, 0.0],
                [{"obj_id": "OBJ-1", "r_km": [7000.0, 1.0, 0.0], "v_km_s": [0.0, 7.5, 0.0]}],
                "2026-08-24T12:00:00Z",
                [0.0, 7.5, 0.0],
                self._policy(),
                self._cap(),
            )

        assert result.secondary_check_performed is False
        assert result.secondary_conjunction_clear is False
        assert "Pc failed" in result.operator_note
        assert "M4" in result.operator_note


# ---------------------------------------------------------------------------
# AC6: VerificationResult.secondary_clear reflects actual check
# ---------------------------------------------------------------------------

class TestVerificationResultSecondaryField:
    """AC6: VerificationResult.secondary_clear must reflect the actual
    SecondaryConflictCheck result, not the stub default."""

    def test_secondary_clear_true_when_check_clear(self):
        """secondary_clear must be True when check performed and clear."""
        secondary = SecondaryConflictCheck(
            secondary_check_performed=True,
            secondary_conjunction_clear=True,
            flagged_objects=[],
            operator_note="clear",
        )
        scoring = _make_scoring_stub()
        policy = _make_policy_stub()
        result = _build_verification_result(scoring, policy, secondary)
        assert result.secondary_clear is True

    def test_secondary_clear_false_when_conflict_detected(self):
        """secondary_clear must be False when a conflict is detected."""
        secondary = SecondaryConflictCheck(
            secondary_check_performed=True,
            secondary_conjunction_clear=False,
            flagged_objects=["35929"],
            operator_note="conflict detected",
        )
        scoring = _make_scoring_stub()
        policy = _make_policy_stub()
        result = _build_verification_result(scoring, policy, secondary)
        assert result.secondary_clear is False

    def test_secondary_clear_false_when_not_performed(self):
        """When check is not performed, verification must fail closed."""
        secondary = SecondaryConflictCheck(
            secondary_check_performed=False,
            secondary_conjunction_clear=False,
            flagged_objects=[],
            operator_note="not performed",
        )
        scoring = _make_scoring_stub()
        policy = _make_policy_stub()
        result = _build_verification_result(scoring, policy, secondary)
        assert result.secondary_clear is False
        assert result.passed is False
        assert any("M4" in reason for reason in result.failure_reasons)

    def test_verification_fails_when_secondary_conflict(self):
        """VerificationResult.passed must be False when secondary conflict detected."""
        secondary = SecondaryConflictCheck(
            secondary_check_performed=True,
            secondary_conjunction_clear=False,
            flagged_objects=["OBJ-1"],
            operator_note="conflict",
        )
        scoring = _make_scoring_stub()
        policy = _make_policy_stub()
        result = _build_verification_result(scoring, policy, secondary)
        assert result.passed is False
        assert any("OBJ-1" in r for r in result.failure_reasons)


# ---------------------------------------------------------------------------
# SCRUM-330 follow-on: closest_approach_km and closest_object_id fields
# ---------------------------------------------------------------------------


class TestSecondaryCovarianceConvention:
    @pytest.mark.parametrize(
        ("extra", "expected_uncertainty_m"),
        [
            ({"position_sigma_m": 250.0, "confidence": 0.2}, 250.0),
            ({"confidence": 0.5}, 4000.0),
            ({}, 2500.0),
        ],
    )
    def test_reuses_scrum_391_debris_uncertainty_convention(
        self, extra, expected_uncertainty_m
    ):
        obj = {
            "obj_id": "OBJ-1",
            "r_km": [7000.0, 1.0, 0.0],
            "v_km_s": [0.0, 7.5, 0.0],
            **extra,
        }
        state = (
            30.0,
            1.0,
            np.array([7000.0, 0.0, 0.0]),
            np.array([0.0, 7.5, 0.0]),
            np.array([7000.0, 1.0, 0.0]),
            np.array([0.0, 7.5, 0.0]),
        )

        with patch(
            "common.secondary_horizon._find_object_tca",
            return_value=state,
        ), patch(
            "common.secondary_horizon.compute_pc",
            return_value=MagicMock(Pc=1.0e-6),
        ), patch(
            "common.secondary_horizon.default_covariance_from_uncertainty",
            return_value=np.eye(3),
        ) as covariance_mock:
            screen_secondary_catalog(
                r_post_km=[7000.0, 0.0, 0.0],
                v_post_km_s=[0.0, 7.5, 0.0],
                known_objects=[obj],
                burn_time_utc="2026-08-24T12:00:00Z",
                horizon_hours=12.0,
                pc_action=1.0e-4,
                primary_radius_m=0.6,
            )

        secondary_call = covariance_mock.call_args_list[1]
        assert secondary_call.args[0] == pytest.approx(expected_uncertainty_m)
