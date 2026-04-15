"""
Tests for APS 2.5 OperatorPolicy layer (Section 9.2).

Acceptance criteria covered:
  - Operator policy schema is defined and documented
  - At least one default policy configuration exists for LEO operations
  - Hard constraint overrides are implemented and tested
  - Policy is loadable at runtime without code changes
  - merge_blackout_windows() correctly merges operator-level and
    satellite-level blackout windows, preserving timing data from both
  - Unit test covers the merge with overlapping and non-overlapping
    windows from both sources
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.operator_policy import BlackoutWindow, OperatorPolicy, ScoringWeights

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "operator_policy_leo.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def default_policy():
    return OperatorPolicy(operator_id="TEST_LEO", policy_version="2.5.0")


@pytest.fixture(scope="module")
def yaml_policy():
    return OperatorPolicy.from_yaml(str(CONFIG_PATH))


@pytest.fixture(scope="module")
def policy_with_windows():
    return OperatorPolicy(
        operator_id="TEST_LEO",
        policy_version="2.5.0",
        blackout_windows=[
            BlackoutWindow(
                window_type="payload_operation",
                start_utc="2026-04-14T08:00:00Z",
                end_utc="2026-04-14T09:00:00Z",
                buffer_hours_before=1.0,
                buffer_hours_after=1.0,
                source="operator",
            )
        ],
    )


# ---------------------------------------------------------------------------
# Schema and defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_operator_id(self, default_policy):
        assert default_policy.operator_id == "TEST_LEO"

    def test_policy_version(self, default_policy):
        assert default_policy.policy_version == "2.5.0"

    def test_pc_maneuver_threshold(self, default_policy):
        assert default_policy.pc_maneuver_threshold == 1.0e-4

    def test_pc_monitor_threshold(self, default_policy):
        assert default_policy.pc_monitor_threshold == 1.0e-5

    def test_min_miss_distance(self, default_policy):
        assert default_policy.min_miss_distance_km == 1.0

    def test_mahalanobis_screen_threshold(self, default_policy):
        assert default_policy.mahalanobis_screen_threshold == 4.0

    def test_max_dv_per_event(self, default_policy):
        assert default_policy.max_dv_per_event_ms == 2.0

    def test_max_maneuvers_per_week(self, default_policy):
        assert default_policy.max_maneuvers_per_week == 3

    def test_operational_philosophy(self, default_policy):
        assert default_policy.operational_philosophy == "balanced"

    def test_fleet_priority(self, default_policy):
        assert default_policy.fleet_priority_method == "highest_pc_first"

    def test_scoring_weights_defaults(self, default_policy):
        assert default_policy.scoring_weights.lambda_dv == 1.0
        assert default_policy.scoring_weights.lambda_lifetime == 0.8
        assert default_policy.scoring_weights.lambda_slot_deviation == 1.2


# ---------------------------------------------------------------------------
# YAML loader -- default LEO policy
# ---------------------------------------------------------------------------

class TestYamlLoader:
    def test_loads_without_error(self, yaml_policy):
        assert yaml_policy is not None

    def test_operator_id(self, yaml_policy):
        assert yaml_policy.operator_id == "DEFAULT_LEO"

    def test_policy_version(self, yaml_policy):
        assert yaml_policy.policy_version == "2.5.0"

    def test_pc_maneuver_threshold(self, yaml_policy):
        assert yaml_policy.pc_maneuver_threshold == 1.0e-4

    def test_scoring_weights(self, yaml_policy):
        assert yaml_policy.scoring_weights.lambda_dv == 1.0
        assert yaml_policy.scoring_weights.lambda_lifetime == 0.8
        assert yaml_policy.scoring_weights.lambda_slot_deviation == 1.2

    def test_blackout_windows_loaded(self, yaml_policy):
        assert len(yaml_policy.blackout_windows) == 2

    def test_blackout_window_types(self, yaml_policy):
        types = {w.window_type for w in yaml_policy.blackout_windows}
        assert "payload_operation" in types
        assert "ground_contact" in types

    def test_blackout_window_source(self, yaml_policy):
        for w in yaml_policy.blackout_windows:
            assert w.source == "operator"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            OperatorPolicy.from_yaml("/nonexistent/path/policy.yaml")


# ---------------------------------------------------------------------------
# Hard constraint gates (Tier 2 embedded)
# ---------------------------------------------------------------------------

class TestHardConstraints:
    def test_maneuver_required_high_pc(self, default_policy):
        assert default_policy.is_maneuver_required(pc=2.0e-4, miss_distance_km=2.0) is True

    def test_maneuver_required_close_miss(self, default_policy):
        assert default_policy.is_maneuver_required(pc=1.0e-6, miss_distance_km=0.5) is True

    def test_maneuver_not_required_low_risk(self, default_policy):
        assert default_policy.is_maneuver_required(pc=1.0e-6, miss_distance_km=3.0) is False

    def test_pre_screen_pass(self, default_policy):
        assert default_policy.passes_pre_screen(mahalanobis_distance=3.2) is True

    def test_pre_screen_fail(self, default_policy):
        assert default_policy.passes_pre_screen(mahalanobis_distance=6.1) is False

    def test_pre_screen_boundary(self, default_policy):
        assert default_policy.passes_pre_screen(
            mahalanobis_distance=default_policy.mahalanobis_screen_threshold
        ) is True

    def test_monitor_only_in_range(self, default_policy):
        assert default_policy.is_monitor_only(pc=5.0e-5) is True

    def test_monitor_only_below_range(self, default_policy):
        assert default_policy.is_monitor_only(pc=1.0e-6) is False

    def test_monitor_only_above_range(self, default_policy):
        assert default_policy.is_monitor_only(pc=2.0e-4) is False


# ---------------------------------------------------------------------------
# merge_blackout_windows -- non-overlapping
# ---------------------------------------------------------------------------

class TestMergeNonOverlapping:
    @pytest.fixture
    def cadence_windows(self):
        return [
            {
                "type": "ground_contact",
                "start_utc": "2026-04-14T12:00:00Z",
                "end_utc": "2026-04-14T12:15:00Z",
                "buffer_hours_before": 0.5,
                "buffer_hours_after": 0.0,
            }
        ]

    def test_merged_count(self, policy_with_windows, cadence_windows):
        merged = policy_with_windows.merge_blackout_windows(cadence_windows)
        assert len(merged) == 2

    def test_operator_window_preserved(self, policy_with_windows, cadence_windows):
        merged = policy_with_windows.merge_blackout_windows(cadence_windows)
        assert merged[0].source == "operator"
        assert merged[0].window_type == "payload_operation"

    def test_cadence_window_source(self, policy_with_windows, cadence_windows):
        merged = policy_with_windows.merge_blackout_windows(cadence_windows)
        assert merged[1].source == "satellite_cadence"

    def test_cadence_start_utc_preserved(self, policy_with_windows, cadence_windows):
        merged = policy_with_windows.merge_blackout_windows(cadence_windows)
        assert merged[1].start_utc == "2026-04-14T12:00:00Z"

    def test_cadence_end_utc_preserved(self, policy_with_windows, cadence_windows):
        merged = policy_with_windows.merge_blackout_windows(cadence_windows)
        assert merged[1].end_utc == "2026-04-14T12:15:00Z"

    def test_cadence_buffer_preserved(self, policy_with_windows, cadence_windows):
        merged = policy_with_windows.merge_blackout_windows(cadence_windows)
        assert merged[1].buffer_hours_before == 0.5
        assert merged[1].buffer_hours_after == 0.0


# ---------------------------------------------------------------------------
# merge_blackout_windows -- overlapping
# ---------------------------------------------------------------------------

class TestMergeOverlapping:
    @pytest.fixture
    def overlapping_cadence_windows(self):
        return [
            {
                "type": "ground_contact",
                "start_utc": "2026-04-14T08:30:00Z",  # overlaps operator window
                "end_utc": "2026-04-14T08:45:00Z",
                "buffer_hours_before": 0.0,
                "buffer_hours_after": 0.0,
            }
        ]

    def test_merged_count(self, policy_with_windows, overlapping_cadence_windows):
        merged = policy_with_windows.merge_blackout_windows(overlapping_cadence_windows)
        assert len(merged) == 2  # both retained -- overlap handled at constraint check time

    def test_overlapping_cadence_start_preserved(
        self, policy_with_windows, overlapping_cadence_windows
    ):
        merged = policy_with_windows.merge_blackout_windows(overlapping_cadence_windows)
        assert merged[1].start_utc == "2026-04-14T08:30:00Z"

    def test_overlapping_cadence_source(
        self, policy_with_windows, overlapping_cadence_windows
    ):
        merged = policy_with_windows.merge_blackout_windows(overlapping_cadence_windows)
        assert merged[1].source == "satellite_cadence"

    def test_operator_window_unchanged(
        self, policy_with_windows, overlapping_cadence_windows
    ):
        merged = policy_with_windows.merge_blackout_windows(overlapping_cadence_windows)
        assert merged[0].start_utc == "2026-04-14T08:00:00Z"
        assert merged[0].source == "operator"


# ---------------------------------------------------------------------------
# merge_blackout_windows -- edge cases
# ---------------------------------------------------------------------------

class TestMergeEdgeCases:
    def test_empty_cadence_windows(self, policy_with_windows):
        merged = policy_with_windows.merge_blackout_windows([])
        assert len(merged) == 1
        assert merged[0].source == "operator"

    def test_non_dict_cadence_entry_skipped(self, policy_with_windows):
        merged = policy_with_windows.merge_blackout_windows(["not_a_dict", None])
        assert len(merged) == 1

    def test_no_operator_windows(self):
        policy_no_windows = OperatorPolicy(
            operator_id="NO_WINDOWS", policy_version="2.5.0"
        )
        cadence = [
            {
                "type": "ground_contact",
                "start_utc": "2026-04-14T12:00:00Z",
                "end_utc": "2026-04-14T12:15:00Z",
            }
        ]
        merged = policy_no_windows.merge_blackout_windows(cadence)
        assert len(merged) == 1
        assert merged[0].source == "satellite_cadence"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_operator_id_raises(self):
        with pytest.raises(ValueError):
            OperatorPolicy(operator_id="", policy_version="2.5.0")

    def test_monitor_threshold_above_maneuver_raises(self):
        with pytest.raises(ValueError):
            OperatorPolicy(
                operator_id="X", policy_version="2.5.0",
                pc_maneuver_threshold=1.0e-5,
                pc_monitor_threshold=1.0e-4,  # monitor > maneuver
            )

    def test_invalid_philosophy_raises(self):
        with pytest.raises(ValueError):
            OperatorPolicy(
                operator_id="X", policy_version="2.5.0",
                operational_philosophy="aggressive",
            )

    def test_invalid_fleet_priority_raises(self):
        with pytest.raises(ValueError):
            OperatorPolicy(
                operator_id="X", policy_version="2.5.0",
                fleet_priority_method="random",
            )

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError):
            ScoringWeights(lambda_dv=-1.0)

    def test_max_hours_before_min_raises(self):
        with pytest.raises(ValueError):
            OperatorPolicy(
                operator_id="X", policy_version="2.5.0",
                min_hours_before_tca=10.0,
                max_hours_before_tca=5.0,
            )


# ---------------------------------------------------------------------------
# to_dict round-trip
# ---------------------------------------------------------------------------

class TestMissionLifetime:
    def test_default_is_none(self, default_policy):
        assert default_policy.mission_lifetime_days_total is None

    def test_yaml_loads_lifetime(self, yaml_policy):
        assert yaml_policy.mission_lifetime_days_total == 1825.0

    def test_explicit_value_accepted(self):
        p = OperatorPolicy(operator_id="X", policy_version="2.5.0",
                           mission_lifetime_days_total=730.0)
        assert p.mission_lifetime_days_total == 730.0

    def test_zero_lifetime_raises(self):
        with pytest.raises(ValueError):
            OperatorPolicy(operator_id="X", policy_version="2.5.0",
                           mission_lifetime_days_total=0.0)

    def test_in_to_dict(self, default_policy):
        d = default_policy.to_dict()
        assert "mission_lifetime_days_total" in d
        assert d["mission_lifetime_days_total"] is None

    def test_in_to_dict_with_value(self):
        p = OperatorPolicy(operator_id="X", policy_version="2.5.0",
                           mission_lifetime_days_total=1825.0)
        assert p.to_dict()["mission_lifetime_days_total"] == 1825.0


class TestSerialization:
    def test_to_dict_contains_required_keys(self, default_policy):
        d = default_policy.to_dict()
        for key in ["operator_id", "policy_version", "pc_maneuver_threshold",
                    "scoring_weights", "blackout_windows", "fleet_priority_method"]:
            assert key in d

    def test_scoring_weights_in_dict(self, default_policy):
        d = default_policy.to_dict()
        assert d["scoring_weights"]["lambda_dv"] == 1.0

    def test_blackout_windows_serialized(self, policy_with_windows):
        d = policy_with_windows.to_dict()
        assert len(d["blackout_windows"]) == 1
        assert d["blackout_windows"][0]["source"] == "operator"
