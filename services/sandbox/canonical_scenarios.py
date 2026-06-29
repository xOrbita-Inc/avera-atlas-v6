from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import numpy as np

from services.tracker.iod import IODObservation, IODSolution, IODSolver


@dataclass(frozen=True)
class ScenarioReport:
    scenario_id: str
    name: str
    passed: bool
    summary: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class RegressionReport:
    suite: str
    seed: int
    passed: bool
    acceptance_validated: bool
    elapsed_sec: float
    scenarios: list[ScenarioReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "seed": self.seed,
            "passed": self.passed,
            "acceptance_validated": self.acceptance_validated,
            "elapsed_sec": self.elapsed_sec,
            "scenarios": [
                asdict(scenario)
                for scenario in self.scenarios
            ],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            indent=2,
            sort_keys=True,
        )


EPOCH_UTC = datetime(2026, 1, 1, tzinfo=timezone.utc)

UNCERTAINTY_REFERENCE_KM = {
    "0_min": 5.0,
    "5_min": 5.1,
    "30_min": 5.8,
    "2_hr": 9.2,
    "6_hr": 22.0,
    "1_day": 85.0,
}

LOST_CUSTODY_THRESHOLD_KM = 78.0


def within_relative_tolerance(
    actual: float,
    expected: float,
    tolerance_fraction: float,
) -> bool:
    lower = expected * (1.0 - tolerance_fraction)
    upper = expected * (1.0 + tolerance_fraction)
    return lower <= actual <= upper


def _los_to_ra_dec(los_eci: np.ndarray) -> tuple[float, float]:
    los = np.asarray(los_eci, dtype=np.float64)
    los_norm = float(np.linalg.norm(los))

    if los_norm < 1e-12:
        raise ValueError("Cannot convert near-zero line of sight.")

    los = los / los_norm

    x, y, z = los
    ra = float(np.arctan2(y, x) % (2.0 * np.pi))
    dec = float(np.arcsin(np.clip(z, -1.0, 1.0)))

    return ra, dec


def _make_iod_observation(
    *,
    t_seconds: float,
    observer_position_km: np.ndarray,
    observer_velocity_km_s: np.ndarray,
    target_position_km: np.ndarray,
    include_range: bool = False,
) -> IODObservation:
    los = target_position_km - observer_position_km
    range_km = float(np.linalg.norm(los))
    ra, dec = _los_to_ra_dec(los)

    sigma_rad = float(np.deg2rad(2.9 / 3600.0))

    return IODObservation(
        timestamp=EPOCH_UTC + timedelta(seconds=t_seconds),
        ra=ra,
        dec=dec,
        ra_sigma=sigma_rad,
        dec_sigma=sigma_rad,
        observer_position_km=np.asarray(
            observer_position_km,
            dtype=np.float64,
        ),
        observer_velocity_km_s=np.asarray(
            observer_velocity_km_s,
            dtype=np.float64,
        ),
        range_km=range_km if include_range else None,
        range_sigma_km=0.05 if include_range else None,
    )


def _run_tracker_iod(
    observations: list[IODObservation],
) -> IODSolution:
    return IODSolver().solve(
        observations=observations,
        track_id=uuid4(),
    )


def _circular_leo_state(
    *,
    radius_km: float = 6978.137,
    phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    speed_km_s = float(np.sqrt(3.986004418e5 / radius_km))

    position = radius_km * np.array(
        [
            np.cos(phase_rad),
            np.sin(phase_rad),
            0.0,
        ],
        dtype=np.float64,
    )
    velocity = speed_km_s * np.array(
        [
            -np.sin(phase_rad),
            np.cos(phase_rad),
            0.0,
        ],
        dtype=np.float64,
    )

    return position, velocity


def _propagate_linear(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    t_seconds: float,
) -> np.ndarray:
    return position_km + velocity_km_s * t_seconds


def _seconds_since_harness_epoch(timestamp: datetime) -> float:
    if timestamp.tzinfo is None:
        raise ValueError("solution epoch must be timezone-aware.")

    return float((timestamp - EPOCH_UTC).total_seconds())


def _along_track_error_km(
    *,
    estimated_r_km: np.ndarray,
    truth_r_km: np.ndarray,
    truth_v_km_s: np.ndarray,
) -> float:
    truth_v = np.asarray(
        truth_v_km_s,
        dtype=np.float64,
    )
    truth_speed = float(np.linalg.norm(truth_v))

    if truth_speed < 1e-12:
        raise ValueError(
            "Cannot compute along-track error with near-zero truth velocity."
        )

    along_track_hat = truth_v / truth_speed

    return abs(
        float(
            np.dot(
                np.asarray(
                    estimated_r_km,
                    dtype=np.float64,
                )
                - np.asarray(
                    truth_r_km,
                    dtype=np.float64,
                ),
                along_track_hat,
            )
        )
    )


def _measured_solution_error(
    *,
    solution: IODSolution,
    truth_r0_km: np.ndarray,
    truth_v_km_s: np.ndarray,
) -> tuple[float | None, float | None]:
    if not solution.success:
        return None, None

    if solution.position_km is None:
        return None, None

    solution_t_seconds = _seconds_since_harness_epoch(solution.epoch)
    truth_r_at_solution_epoch = _propagate_linear(
        truth_r0_km,
        truth_v_km_s,
        solution_t_seconds,
    )

    error_km = _along_track_error_km(
        estimated_r_km=solution.position_km,
        truth_r_km=truth_r_at_solution_epoch,
        truth_v_km_s=truth_v_km_s,
    )

    return solution_t_seconds, error_km


def run_tier1_coorbital_single_pass_scenario(
    *,
    seed: int = 42,
) -> ScenarioReport:
    """
    Scenario A: Tier 1 closure from a co-orbital single-pass tracklet.

    This scenario validates the tracker IOD closure and measures along-track
    error against scenario truth at the solver-selected epoch.
    """
    _ = seed

    host_r0, host_v = _circular_leo_state(radius_km=6978.137)

    relative_position_km = np.array(
        [0.0, 18.0, 1.0],
        dtype=np.float64,
    )
    relative_velocity_km_s = np.array(
        [0.0, 0.30, -0.08],
        dtype=np.float64,
    )

    debris_r0 = host_r0 + relative_position_km
    debris_v = host_v + relative_velocity_km_s

    sample_times = [
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
    ]

    observations = [
        _make_iod_observation(
            t_seconds=t_seconds,
            observer_position_km=_propagate_linear(
                host_r0,
                host_v,
                t_seconds,
            ),
            observer_velocity_km_s=host_v,
            target_position_km=_propagate_linear(
                debris_r0,
                debris_v,
                t_seconds,
            ),
            include_range=True,
        )
        for t_seconds in sample_times
    ]

    solution = _run_tracker_iod(observations)

    solution_t_seconds, along_track_error_km = _measured_solution_error(
        solution=solution,
        truth_r0_km=debris_r0,
        truth_v_km_s=debris_v,
    )

    transit_time_seconds = 40.0

    passed = (
        bool(solution.success)
        and along_track_error_km is not None
        and len(observations) >= 3
        and 4.0 <= transit_time_seconds <= 40.0
        and along_track_error_km < 10.0
    )

    return ScenarioReport(
        scenario_id="A",
        name="tier1_coorbital_single_pass",
        passed=passed,
        summary=(
            "Tier 1 canonical co-orbital tracklet closes without Tier 2 "
            "tasking and measures along-track error against scenario truth."
        ),
        metrics={
            "acceptance_validated": passed,
            "validation_level": "measured_truth_comparison",
            "host_count": 1,
            "debris_count": 1,
            "relative_speed_km_s": float(
                np.linalg.norm(relative_velocity_km_s)
            ),
            "transit_time_seconds": transit_time_seconds,
            "observation_count": len(observations),
            "iod_solver_success": bool(solution.success),
            "iod_solver_observations_used": solution.observations_used,
            "iod_solver_method": solution.method_used,
            "iod_solver_error_message": solution.error_message,
            "solution_epoch_seconds": solution_t_seconds,
            "closure_time_seconds": max(sample_times),
            "along_track_error_km": along_track_error_km,
            "along_track_error_source": (
                "measured_against_scenario_truth_at_solution_epoch"
            ),
            "pass_threshold_along_track_error_km": 10.0,
        },
    )


def run_tier2_tasked_reobservation_scenario(
    *,
    seed: int = 42,
) -> ScenarioReport:
    """
    Scenario B: Tier 2 closure scaffold from partial tracklet plus
    re-observation samples.

    This verifies the partial-tracklet failure and post-reobservation IOD
    closure shape. It does not yet validate the SCRUM-293 tasking execution
    path, because these observations are still hand-built rather than produced
    by execute_tasking_command(...).
    """
    _ = seed

    host1_r0, host1_v = _circular_leo_state(
        radius_km=6978.137,
        phase_rad=0.0,
    )
    host2_r0, host2_v = _circular_leo_state(
        radius_km=6978.137,
        phase_rad=2.1,
    )
    host3_r0, host3_v = _circular_leo_state(
        radius_km=6978.137,
        phase_rad=4.2,
    )

    _ = host3_r0
    _ = host3_v

    relative_position_km = np.array(
        [0.0, 20.0, 0.0],
        dtype=np.float64,
    )
    high_crossing_velocity_km_s = np.array(
        [0.0, 0.0, 8.0],
        dtype=np.float64,
    )

    debris_r0 = host1_r0 + relative_position_km
    debris_v = host1_v + high_crossing_velocity_km_s

    partial_times = [
        0.0,
        0.25,
    ]
    tasked_times = [
        3600.0,
        3630.0,
        3660.0,
    ]

    partial_observations = [
        _make_iod_observation(
            t_seconds=t_seconds,
            observer_position_km=_propagate_linear(
                host1_r0,
                host1_v,
                t_seconds,
            ),
            observer_velocity_km_s=host1_v,
            target_position_km=_propagate_linear(
                debris_r0,
                debris_v,
                t_seconds,
            ),
        )
        for t_seconds in partial_times
    ]

    partial_solution = _run_tracker_iod(partial_observations)

    tasked_observations = [
        _make_iod_observation(
            t_seconds=t_seconds,
            observer_position_km=_propagate_linear(
                host2_r0,
                host2_v,
                t_seconds,
            ),
            observer_velocity_km_s=host2_v,
            target_position_km=_propagate_linear(
                debris_r0,
                debris_v,
                t_seconds,
            ),
        )
        for t_seconds in tasked_times
    ]

    combined_observations = partial_observations + tasked_observations
    final_solution = _run_tracker_iod(combined_observations)

    solution_t_seconds, final_along_track_error_km = _measured_solution_error(
        solution=final_solution,
        truth_r0_km=debris_r0,
        truth_v_km_s=debris_v,
    )

    closure_time_seconds = max(tasked_times) - min(partial_times)

    scaffold_passed = (
        len(partial_observations) < 3
        and partial_solution.success is False
        and len(combined_observations) >= 3
        and bool(final_solution.success)
        and closure_time_seconds <= 24.0 * 3600.0
    )

    return ScenarioReport(
        scenario_id="B",
        name="tier2_high_crossing_tasked_reobservation_scaffold",
        passed=scaffold_passed,
        summary=(
            "Tier 2 closure scaffold verifies partial-tracklet failure and "
            "post-reobservation IOD closure shape, but does not yet route "
            "through the SCRUM-293 tasking interface."
        ),
        metrics={
            "acceptance_validated": False,
            "validation_level": "closure_scaffold_only",
            "tasking_interface_used": False,
            "future_story": (
                "Route Scenario B through execute_tasking_command(...) with "
                "real task-window snapshots and predicted_point tasking."
            ),
            "host_count": 3,
            "debris_count": 1,
            "relative_speed_km_s": float(
                np.linalg.norm(high_crossing_velocity_km_s)
            ),
            "transit_time_seconds": 0.25,
            "partial_observation_count": len(partial_observations),
            "partial_iod_solver_success": bool(partial_solution.success),
            "partial_iod_error_message": partial_solution.error_message,
            "tasked_sensor_id": "host_002",
            "tasked_observation_count": len(tasked_observations),
            "combined_observation_count": len(combined_observations),
            "final_iod_solver_success": bool(final_solution.success),
            "final_iod_solver_observations_used": (
                final_solution.observations_used
            ),
            "final_iod_solver_method": final_solution.method_used,
            "final_iod_solver_error_message": final_solution.error_message,
            "solution_epoch_seconds": solution_t_seconds,
            "closure_time_seconds": closure_time_seconds,
            "final_along_track_error_km": final_along_track_error_km,
            "along_track_error_source": (
                "measured_against_scenario_truth_at_solution_epoch"
            ),
            "pass_threshold_along_track_error_km": 10.0,
        },
    )


def run_tier3_custody_maintenance_scenario(
    *,
    seed: int = 42,
) -> ScenarioReport:
    """
    Scenario C: Tier 3 custody-maintenance report scaffold.

    The real emergent uncertainty validation is deferred to SCRUM-343. This
    scenario keeps the canonical report shape and reference values visible, but
    it does not compare the reference curve to itself or claim production
    custody-uncertainty validation.
    """
    _ = seed

    return ScenarioReport(
        scenario_id="C",
        name="tier3_custody_maintenance_deferred",
        passed=True,
        summary=(
            "Tier 3 custody report scaffold only. Emergent uncertainty "
            "validation is deferred to SCRUM-343."
        ),
        metrics={
            "acceptance_validated": False,
            "validation_level": "deferred_to_SCRUM-343",
            "emergent_uncertainty_computed": False,
            "host_count": 6,
            "debris_count": 50,
            "duration_days": 7,
            "reobservation_interval_hours": [
                2,
                6,
            ],
            "reference_uncertainty_km": dict(UNCERTAINTY_REFERENCE_KM),
            "tolerance_fraction": 0.30,
            "lost_custody_threshold_km": LOST_CUSTODY_THRESHOLD_KM,
            "future_story": (
                "SCRUM-343 owns full-scale swarm runs and emergent "
                "custody-uncertainty validation against the 5 / 22 / 85 km "
                "timeline."
            ),
        },
    )


def run_all_canonical_scenarios(
    *,
    seed: int = 42,
) -> RegressionReport:
    started = time.perf_counter()

    scenarios = [
        run_tier1_coorbital_single_pass_scenario(seed=seed),
        run_tier2_tasked_reobservation_scenario(seed=seed),
        run_tier3_custody_maintenance_scenario(seed=seed),
    ]

    elapsed_sec = time.perf_counter() - started

    return RegressionReport(
        suite="sandbox_canonical_regression",
        seed=seed,
        passed=all(scenario.passed for scenario in scenarios),
        acceptance_validated=all(
            bool(
                scenario.metrics.get(
                    "acceptance_validated",
                    False,
                )
            )
            for scenario in scenarios
        ),
        elapsed_sec=elapsed_sec,
        scenarios=scenarios,
    )