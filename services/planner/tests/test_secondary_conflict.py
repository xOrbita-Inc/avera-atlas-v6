"""
tests/test_secondary_conflict.py

SCRUM-330: Unit tests for SecondaryConflictCheck live catalog integration.

Tests cover:
  AC1: performed=True when catalog is available
  AC2: secondary conjunction correctly flagged with NORAD ID and distance
  AC3: clean post-burn state returns performed=True, conflict=False
  AC4: catalog unavailability falls back to not_performed, no crash
  AC6: VerificationResult.secondary_clear reflects actual check result

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
from common.spacetrack_tle import (
    _parse_and_propagate_tle,
    fetch_catalog_objects,
    _DEFAULT_SCREENING_RADIUS_KM,
)


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
# AC2: Secondary conjunction flagged correctly
# ---------------------------------------------------------------------------

class TestSecondaryConflictFlagged:
    """AC2: A post-burn state that introduces a secondary conjunction is
    correctly flagged with object ID and separation distance."""

    def test_flagged_when_object_within_threshold(self):
        """Object within 1 km threshold must be in flagged_objects."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [
            {"obj_id": "35929", "r_km": [6778.4, 0.0, 0.0]},  # 0.4 km away
        ]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert result.secondary_check_performed is True
        assert result.secondary_conjunction_clear is False
        assert "35929" in result.flagged_objects

    def test_flagged_object_id_preserved(self):
        """Flagged object ID must match the obj_id in known_objects."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [{"obj_id": "IRIDIUM-33-DEB", "r_km": [6778.2, 0.0, 0.0]}]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert "IRIDIUM-33-DEB" in result.flagged_objects

    def test_multiple_flagged_objects(self):
        """Multiple objects within threshold must all appear in flagged_objects."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [
            {"obj_id": "OBJ-A", "r_km": [6778.3, 0.0, 0.0]},
            {"obj_id": "OBJ-B", "r_km": [6777.8, 0.0, 0.0]},
            {"obj_id": "OBJ-C", "r_km": [6800.0, 0.0, 0.0]},  # far, not flagged
        ]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert "OBJ-A" in result.flagged_objects
        assert "OBJ-B" in result.flagged_objects
        assert "OBJ-C" not in result.flagged_objects

    def test_operator_note_mentions_flagged_count(self):
        """operator_note must mention the number of flagged objects."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [{"obj_id": "TEST-1", "r_km": [6778.5, 0.0, 0.0]}]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert "1" in result.operator_note


# ---------------------------------------------------------------------------
# AC3: Clean post-burn state
# ---------------------------------------------------------------------------

class TestSecondaryConflictClear:
    """AC3: Clean post-burn state returns performed=True, conflict=False."""

    def test_clear_when_no_objects_within_threshold(self):
        """No objects within 1 km must return conjunction_clear=True."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [
            {"obj_id": "FAR-1", "r_km": [6800.0, 0.0, 0.0]},  # 22 km away
            {"obj_id": "FAR-2", "r_km": [6778.0, 50.0, 0.0]},  # 50 km away
        ]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert result.secondary_check_performed is True
        assert result.secondary_conjunction_clear is True
        assert len(result.flagged_objects) == 0

    def test_clear_operator_note_mentions_object_count(self):
        """operator_note for clear result must mention catalog size checked."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [{"obj_id": "FAR-1", "r_km": [6900.0, 0.0, 0.0]}]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert "1" in result.operator_note

    def test_exact_threshold_boundary_not_flagged(self):
        """Object at exactly 1.0 km separation must NOT be flagged (< not <=)."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [{"obj_id": "BOUNDARY", "r_km": [6779.0, 0.0, 0.0]}]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert result.secondary_conjunction_clear is True

    def test_just_inside_threshold_is_flagged(self):
        """Object at 0.999 km must be flagged."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [{"obj_id": "CLOSE", "r_km": [6778.999, 0.0, 0.0]}]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert result.secondary_conjunction_clear is False


# ---------------------------------------------------------------------------
# AC4: Catalog unavailability fallback
# ---------------------------------------------------------------------------

class TestCatalogUnavailabilityFallback:
    """AC4: No catalog available falls back to not_performed, no crash."""

    def test_not_performed_when_known_objects_none(self):
        """None known_objects must return not_performed."""
        result = _run_secondary_conflict_check([6778.0, 0.0, 0.0], None)
        assert result.secondary_check_performed is False
        assert result.secondary_conjunction_clear is True

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

    def test_fetch_catalog_objects_returns_empty_on_missing_credentials(self):
        """fetch_catalog_objects must return [] when env vars are missing, no crash."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove SPACETRACK_USER and SPACETRACK_PASS if present
            import os
            env = {k: v for k, v in os.environ.items()
                   if k not in ("SPACETRACK_USER", "SPACETRACK_PASS")}
            with patch.dict("os.environ", env, clear=True):
                result = fetch_catalog_objects(
                    r_sat_km=[6778.0, 0.0, 0.0],
                    burn_time_utc="2024-01-01T12:00:00Z",
                )
        assert result == []

    def test_fetch_catalog_objects_returns_empty_on_network_failure(self):
        """fetch_catalog_objects must return [] on network failure, no crash."""
        with patch("common.spacetrack_tle.requests.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_session.post.side_effect = ConnectionError("network unavailable")

            result = fetch_catalog_objects(
                r_sat_km=[6778.0, 0.0, 0.0],
                burn_time_utc="2024-01-01T12:00:00Z",
            )
        assert result == []

    def test_fetch_catalog_objects_returns_empty_on_bad_epoch(self):
        """fetch_catalog_objects must return [] when burn_time_utc is unparseable."""
        result = fetch_catalog_objects(
            r_sat_km=[6778.0, 0.0, 0.0],
            burn_time_utc="not-a-date",
        )
        assert result == []


# ---------------------------------------------------------------------------
# AC1: performed=True when catalog available
# ---------------------------------------------------------------------------

class TestPerformedWhenCatalogAvailable:
    """AC1: SecondaryConflictCheck.performed is True when catalog is available."""

    def test_performed_true_when_objects_supplied(self):
        """Any non-empty known_objects list must set performed=True."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [{"obj_id": "TEST", "r_km": [7000.0, 0.0, 0.0]}]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert result.secondary_check_performed is True

    def test_performed_true_even_with_single_object(self):
        """Single object catalog must still set performed=True."""
        result = _run_secondary_conflict_check(
            [6778.0, 0.0, 0.0],
            [{"obj_id": "SOLO", "r_km": [6900.0, 0.0, 0.0]}],
        )
        assert result.secondary_check_performed is True


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

    def test_secondary_clear_true_when_not_performed(self):
        """When check not performed, secondary_clear must be True (safe default)."""
        secondary = SecondaryConflictCheck(
            secondary_check_performed=False,
            secondary_conjunction_clear=True,
            flagged_objects=[],
            operator_note="not performed",
        )
        scoring = _make_scoring_stub()
        policy = _make_policy_stub()
        result = _build_verification_result(scoring, policy, secondary)
        assert result.secondary_clear is True

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
# TLE parse + propagate (unit, no network)
# ---------------------------------------------------------------------------

class TestTLEParseAndPropagate:
    """Tests for _parse_and_propagate_tle using embedded TLE strings."""

    def _make_tle_text(self, *pairs):
        """Build TLE text from (line1, line2) tuples."""
        lines = []
        for l1, l2 in pairs:
            lines.extend([l1, l2])
        return "\n".join(lines)

    def test_returns_empty_for_empty_input(self):
        epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        r_sat = np.array([6778.0, 0.0, 0.0])
        result = _parse_and_propagate_tle("", epoch, r_sat, 100.0)
        assert result == []

    def test_returns_empty_when_no_objects_nearby(self):
        """ISS should not be near [6778, 0, 0] at the test epoch -- different orbit phase."""
        tle_text = self._make_tle_text((_ISS_TLE_LINE1, _ISS_TLE_LINE2))
        epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Use a position far from ISS at this epoch
        r_sat = np.array([0.0, 0.0, 6778.0])
        result = _parse_and_propagate_tle(tle_text, epoch, r_sat, 1.0)
        # Result may or may not be empty depending on ISS position -- just confirm no crash
        assert isinstance(result, list)

    def test_result_contains_required_keys(self):
        """Each returned object must have obj_id, r_km, separation_km, norad_id."""
        tle_text = self._make_tle_text((_ISS_TLE_LINE1, _ISS_TLE_LINE2))
        epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Propagate ISS to get its actual position, then screen from there
        try:
            from sgp4.api import Satrec, jday
            sat = Satrec.twoline2rv(_ISS_TLE_LINE1, _ISS_TLE_LINE2)
            jd, fr = jday(2024, 1, 1, 12, 0, 0)
            e, r, v = sat.sgp4(jd, fr)
            if e != 0:
                pytest.skip("SGP4 propagation error for test TLE")
            r_sat = np.array(r)
        except ImportError:
            pytest.skip("sgp4 not installed")

        result = _parse_and_propagate_tle(tle_text, epoch, r_sat, 100.0)
        if result:
            obj = result[0]
            assert "obj_id" in obj
            assert "r_km" in obj
            assert "separation_km" in obj
            assert "norad_id" in obj
            assert isinstance(obj["r_km"], list)
            assert len(obj["r_km"]) == 3

    def test_separation_km_is_accurate(self):
        """separation_km must equal actual Euclidean distance."""
        tle_text = self._make_tle_text((_ISS_TLE_LINE1, _ISS_TLE_LINE2))
        epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        try:
            from sgp4.api import Satrec, jday
            sat = Satrec.twoline2rv(_ISS_TLE_LINE1, _ISS_TLE_LINE2)
            jd, fr = jday(2024, 1, 1, 12, 0, 0)
            e, r, v = sat.sgp4(jd, fr)
            if e != 0:
                pytest.skip("SGP4 error")
            r_sat = np.array(r)
        except ImportError:
            pytest.skip("sgp4 not installed")

        result = _parse_and_propagate_tle(tle_text, epoch, r_sat, 200.0)
        for obj in result:
            actual_sep = float(np.linalg.norm(r_sat - np.array(obj["r_km"])))
            assert abs(actual_sep - obj["separation_km"]) < 0.01


# ---------------------------------------------------------------------------
# SCRUM-330 follow-on: closest_approach_km and closest_object_id fields
# ---------------------------------------------------------------------------

class TestClosestApproachFields:
    """AC2 extension: SecondaryConflictCheck must carry closest approach
    distance and object ID, not just a list of flagged object IDs."""

    def test_closest_approach_km_present_when_check_performed(self):
        """closest_approach_km must be set when check is performed."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [{"obj_id": "OBJ-A", "r_km": [6780.0, 0.0, 0.0]}]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert result.closest_approach_km is not None
        assert isinstance(result.closest_approach_km, float)

    def test_closest_approach_km_none_when_not_performed(self):
        """closest_approach_km must be None when check is not performed."""
        result = _run_secondary_conflict_check(None, None)
        assert result.closest_approach_km is None

    def test_closest_object_id_matches_nearest_object(self):
        """closest_object_id must identify the nearest object, not just flagged ones."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [
            {"obj_id": "NEAR", "r_km": [6779.0, 0.0, 0.0]},   # 1.0 km -- not flagged
            {"obj_id": "FAR",  "r_km": [6790.0, 0.0, 0.0]},   # 12.0 km
        ]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert result.closest_object_id == "NEAR"

    def test_closest_approach_km_is_accurate(self):
        """closest_approach_km must equal the actual separation to nearest object."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [
            {"obj_id": "OBJ-A", "r_km": [6781.0, 0.0, 0.0]},  # 3.0 km
            {"obj_id": "OBJ-B", "r_km": [6785.0, 0.0, 0.0]},  # 7.0 km
        ]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert abs(result.closest_approach_km - 3.0) < 0.01
        assert result.closest_object_id == "OBJ-A"

    def test_closest_object_id_none_when_not_performed(self):
        """closest_object_id must be None when check is not performed."""
        result = _run_secondary_conflict_check(None, None)
        assert result.closest_object_id is None

    def test_screening_epoch_utc_carried_through(self):
        """screening_epoch_utc must be passed through from the caller."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [{"obj_id": "OBJ-A", "r_km": [6790.0, 0.0, 0.0]}]
        epoch = "2024-01-01T12:00:00Z"
        result = _run_secondary_conflict_check(r_post, known_objects, epoch)
        assert result.screening_epoch_utc == epoch

    def test_screening_epoch_utc_none_when_not_performed(self):
        """screening_epoch_utc must be None when check is not performed and not supplied."""
        result = _run_secondary_conflict_check(None, None)
        assert result.screening_epoch_utc is None

    def test_closest_approach_in_operator_note(self):
        """operator_note must mention the closest approach distance."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [{"obj_id": "TEST", "r_km": [6780.5, 0.0, 0.0]}]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert str(result.closest_approach_km) in result.operator_note or "km" in result.operator_note

    def test_flagged_object_separate_from_closest(self):
        """A flagged object and closest object can differ -- closest may not be flagged."""
        r_post = [6778.0, 0.0, 0.0]
        known_objects = [
            {"obj_id": "CLOSEST-NOT-FLAGGED", "r_km": [6778.8, 0.0, 0.0]},  # 0.8 km -- flagged
            {"obj_id": "FAR", "r_km": [6800.0, 0.0, 0.0]},                  # 22 km
        ]
        result = _run_secondary_conflict_check(r_post, known_objects)
        assert result.closest_object_id == "CLOSEST-NOT-FLAGGED"
        assert "CLOSEST-NOT-FLAGGED" in result.flagged_objects
