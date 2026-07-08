from __future__ import annotations

import json

from services.sandbox.canonical_scenarios import (
    LOST_CUSTODY_THRESHOLD_KM,
    UNCERTAINTY_REFERENCE_KM,
    run_all_canonical_scenarios,
    run_tier1_coorbital_single_pass_scenario,
    run_tier2_tasked_reobservation_scenario,
    run_tier3_custody_maintenance_scenario,
)


def test_tier1_coorbital_single_pass_scenario_passes() -> None:
    report = run_tier1_coorbital_single_pass_scenario(seed=42)

    assert report.scenario_id == "A"
    assert report.passed
    assert report.metrics["acceptance_validated"] is True
    assert report.metrics["validation_level"] == "measured_truth_comparison"
    assert report.metrics["host_count"] == 1
    assert report.metrics["debris_count"] == 1
    assert report.metrics["observation_count"] >= 3
    assert 4.0 <= report.metrics["transit_time_seconds"] <= 40.0
    assert report.metrics["iod_solver_success"] is True
    assert report.metrics["solution_epoch_seconds"] is not None
    assert report.metrics["along_track_error_km"] is not None
    assert report.metrics["along_track_error_km"] < 10.0
    assert (
        report.metrics["along_track_error_source"]
        == "measured_against_scenario_truth_at_solution_epoch"
    )


def test_tier2_tasked_reobservation_scenario_is_validated() -> None:
    report = run_tier2_tasked_reobservation_scenario(seed=42)

    assert report.scenario_id == "B"
    assert report.passed
    assert report.metrics["acceptance_validated"] is True
    assert report.metrics["validation_level"] == "tasking_driven_reobservation"
    assert report.metrics["partial_tracklet_source"] == "sensor_model_detection"
    assert report.metrics["tasking_interface_used"] is True
    assert (
        report.metrics["tasking_target_source"]
        == "predicted_position_from_partial_tracklet"
    )
    assert report.metrics["truth_used_for_tasking_target"] is False
    assert report.metrics["truth_used_for_reobserver_placement"] is True
    assert (
        report.metrics["reobserver_placement_source"]
        == "constructed_from_true_future_debris_position"
    )
    assert report.metrics["reobservation_source"] == "execute_tasking_command"
    assert report.metrics["tasking_command_type"] == "predicted_point"
    assert report.metrics["tasking_result_status"] == "accepted"
    assert report.metrics["tasking_result_reason"] == "executed"

    assert report.metrics["host_count"] == 3
    assert report.metrics["debris_count"] == 1
    assert report.metrics["simulation_snapshot_count"] > 1
    assert report.metrics["partial_observation_count"] == 2
    assert report.metrics["partial_iod_solver_success"] is False
    assert report.metrics["tasked_observation_count"] > 0
    assert report.metrics["combined_observation_count"] >= 3
    assert report.metrics["final_iod_solver_success"] is True
    assert report.metrics["closure_time_seconds"] <= 24.0 * 3600.0
    assert report.metrics["solution_epoch_seconds"] is not None
    assert report.metrics["final_along_track_error_km"] is not None
    assert report.metrics["final_along_track_error_km"] < 10.0

    assert (
        report.metrics["along_track_error_source"]
        == "measured_against_scenario_truth_at_solution_epoch"
    )


def test_tier3_custody_reference_curve_is_deferred() -> None:
    report = run_tier3_custody_maintenance_scenario(seed=42)

    assert report.scenario_id == "C"
    assert report.passed
    assert report.metrics["acceptance_validated"] is False
    assert report.metrics["validation_level"] == "deferred_to_SCRUM-343"
    assert report.metrics["emergent_uncertainty_computed"] is False

    metrics = report.metrics
    assert metrics["host_count"] == 6
    assert metrics["debris_count"] == 50
    assert metrics["duration_days"] == 7
    assert metrics["lost_custody_threshold_km"] == LOST_CUSTODY_THRESHOLD_KM
    assert metrics["reference_uncertainty_km"] == UNCERTAINTY_REFERENCE_KM


def test_regression_report_contains_all_three_canonical_scenarios() -> None:
    report = run_all_canonical_scenarios(seed=42)

    scenario_ids = [
        scenario.scenario_id
        for scenario in report.scenarios
    ]

    assert scenario_ids == ["A", "B", "C"]
    assert report.passed
    assert report.acceptance_validated is False


def test_regression_report_is_json_serializable() -> None:
    report = run_all_canonical_scenarios(seed=42)

    payload = report.to_dict()
    encoded = report.to_json()
    decoded = json.loads(encoded)

    assert payload["suite"] == "sandbox_canonical_regression"
    assert payload["seed"] == 42
    assert payload["passed"] is True
    assert payload["acceptance_validated"] is False
    assert decoded == payload


def test_regression_report_is_deterministic_except_elapsed_time() -> None:
    report_a = run_all_canonical_scenarios(seed=42).to_dict()
    report_b = run_all_canonical_scenarios(seed=42).to_dict()

    report_a["elapsed_sec"] = 0.0
    report_b["elapsed_sec"] = 0.0

    assert report_a == report_b