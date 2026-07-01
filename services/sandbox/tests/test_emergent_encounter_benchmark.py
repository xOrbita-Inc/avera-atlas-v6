from __future__ import annotations

import pytest

from services.sandbox.emergent_encounter_benchmark import (
    run_emergent_encounter_benchmark,
    run_emergent_encounter_case,
)


def test_emergent_encounter_case_uses_propagated_swarm() -> None:
    result = run_emergent_encounter_case(
        host_count=1,
        host_mode="single",
        seed=42,
        debris_count=20,
        duration_seconds=600.0,
        dt_seconds=10.0,
        save_every_n_steps=1,
    )

    assert result.case_id == "one_sensor"
    assert result.host_count == 1
    assert result.debris_count == 20
    assert result.saved_snapshot_count > 1
    assert result.eligible_debris_count > 0
    assert result.total_sensor_samples > 0
    assert result.target_min_per_day == 3.0
    assert result.target_max_per_day == 6.0
    assert result.accepted_min_per_day == pytest.approx(2.1)
    assert result.accepted_max_per_day == pytest.approx(7.8)
    assert result.detections_per_day >= 0.0
    assert result.rejection_reasons


def test_emergent_benchmark_reports_required_cases() -> None:
    report = run_emergent_encounter_benchmark(
        seed=42,
        debris_count=20,
        duration_seconds=600.0,
        dt_seconds=10.0,
        save_every_n_steps=1,
    )

    case_ids = {
        result.case_id
        for result in report.results
    }

    assert report.benchmark == "emergent_encounter_rate"
    assert report.derived_from_propagated_swarm
    assert report.debris_count == 20
    assert case_ids == {
        "one_sensor",
        "6_sensor_same_host",
        "6_sensor_distributed",
    }


def test_six_sensor_cases_exercise_scaling_modes() -> None:
    report = run_emergent_encounter_benchmark(
        seed=42,
        debris_count=20,
        duration_seconds=600.0,
        dt_seconds=10.0,
        save_every_n_steps=1,
    )

    results_by_case = {
        result.case_id: result
        for result in report.results
    }

    same_host = results_by_case[
        "6_sensor_same_host"
    ]
    distributed = results_by_case[
        "6_sensor_distributed"
    ]

    assert same_host.host_count == 6
    assert distributed.host_count == 6
    assert same_host.host_mode == "same_host"
    assert distributed.host_mode == "distributed"


def test_acceptance_cases_are_in_band_or_explained() -> None:
    report = run_emergent_encounter_benchmark(
        seed=42,
        debris_count=20,
        duration_seconds=600.0,
        dt_seconds=10.0,
        save_every_n_steps=1,
    )

    acceptance_cases = [
        result
        for result in report.results
        if result.within_acceptance_band is not None
    ]

    assert acceptance_cases

    for result in acceptance_cases:
        assert (
            result.within_acceptance_band
            or result.deviation_explanation is not None
        )


def test_benchmark_is_reproducible_for_same_seed() -> None:
    first = run_emergent_encounter_benchmark(
        seed=42,
        debris_count=20,
        duration_seconds=600.0,
        dt_seconds=10.0,
        save_every_n_steps=1,
    )
    second = run_emergent_encounter_benchmark(
        seed=42,
        debris_count=20,
        duration_seconds=600.0,
        dt_seconds=10.0,
        save_every_n_steps=1,
    )

    first_counts = [
        (
            result.case_id,
            result.total_sensor_samples,
            result.candidate_in_fov_samples,
            result.in_fov_outside_range_samples,
            result.detected_samples,
            result.unique_targets_detected,
            result.detections_per_day,
            result.rejection_reasons,
            result.in_fov_rejection_reasons,
            result.closest_in_fov_range_km,
            result.closest_in_fov_range_cutoff_km,
            result.closest_in_fov_range_margin_km,
        )
        for result in first.results
    ]
    second_counts = [
        (
            result.case_id,
            result.total_sensor_samples,
            result.candidate_in_fov_samples,
            result.in_fov_outside_range_samples,
            result.detected_samples,
            result.unique_targets_detected,
            result.detections_per_day,
            result.rejection_reasons,
            result.in_fov_rejection_reasons,
            result.closest_in_fov_range_km,
            result.closest_in_fov_range_cutoff_km,
            result.closest_in_fov_range_margin_km,
        )
        for result in second.results
    ]

    assert first_counts == second_counts


def test_deviation_explanation_uses_in_fov_gate_diagnostics() -> None:
    result = run_emergent_encounter_case(
        host_count=1,
        host_mode="single",
        seed=42,
        debris_count=50,
        duration_seconds=3600.0,
        dt_seconds=5.0,
        save_every_n_steps=1,
    )

    if result.within_acceptance_band:
        assert result.deviation_explanation is None
        return

    assert result.deviation_explanation is not None

    if result.candidate_in_fov_samples == 0:
        assert "no in-FOV cone-transit" in (
            result.deviation_explanation
        )
    elif (
        result.in_fov_outside_range_samples
        == result.candidate_in_fov_samples
    ):
        assert "failed the hard detection range gate" in (
            result.deviation_explanation
        )