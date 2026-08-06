from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from correlate import CorrelatedObservation, CorrelationConfig, CorrelationEngine
from iod import IODObservation, IODSolver


EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
MU_EARTH_KM3_S2 = 398600.4418

# These observations are noiseless and generated from exact two-body circular
# truth. The tolerances are intentionally much wider than floating-point and
# finite-difference error, but still require a useful initial orbit rather than
# accepting a solver result that is hundreds or thousands of kilometres wrong.
POSITION_ERROR_TOLERANCE_KM = 10.0
VELOCITY_ERROR_TOLERANCE_KM_S = 0.01


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("zero vector")
    return vector / norm


def _circular_state(
    *,
    radius_km: float = 6978.137,
    phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    speed_km_s = float(math.sqrt(MU_EARTH_KM3_S2 / radius_km))

    position = radius_km * np.array(
        [
            math.cos(phase_rad),
            math.sin(phase_rad),
            0.0,
        ],
        dtype=np.float64,
    )
    velocity = speed_km_s * np.array(
        [
            -math.sin(phase_rad),
            math.cos(phase_rad),
            0.0,
        ],
        dtype=np.float64,
    )

    return position, velocity


def _circular_state_at_elapsed_time(
    *,
    initial_position_km: np.ndarray,
    elapsed_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    initial_position_km = np.asarray(initial_position_km, dtype=np.float64)
    radius_km = float(np.linalg.norm(initial_position_km))
    phase_rad = float(math.atan2(initial_position_km[1], initial_position_km[0]))
    mean_motion_rad_s = float(
        math.sqrt(MU_EARTH_KM3_S2 / radius_km**3)
    )

    return _circular_state(
        radius_km=radius_km,
        phase_rad=phase_rad + mean_motion_rad_s * elapsed_seconds,
    )


def _rotate_about_x(
    vector: np.ndarray,
    inclination_rad: float,
) -> np.ndarray:
    cosine = math.cos(inclination_rad)
    sine = math.sin(inclination_rad)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=np.float64,
    )
    return rotation @ np.asarray(vector, dtype=np.float64)


def _assert_solution_matches_truth(
    *,
    solution,
    truth_position_km: np.ndarray,
    truth_velocity_km_s: np.ndarray,
) -> None:
    assert solution.success is True, (
        f"IOD did not return a successful state: {solution.error_message}; "
        f"attempted_methods={solution.attempted_methods}"
    )
    assert solution.position_km is not None
    assert solution.velocity_km_s is not None

    position_error_km = float(
        np.linalg.norm(
            np.asarray(solution.position_km, dtype=np.float64)
            - np.asarray(truth_position_km, dtype=np.float64)
        )
    )
    velocity_error_km_s = float(
        np.linalg.norm(
            np.asarray(solution.velocity_km_s, dtype=np.float64)
            - np.asarray(truth_velocity_km_s, dtype=np.float64)
        )
    )

    assert position_error_km <= POSITION_ERROR_TOLERANCE_KM, (
        f"IOD position error {position_error_km:.3f} km exceeds "
        f"{POSITION_ERROR_TOLERANCE_KM:.3f} km; "
        f"velocity error={velocity_error_km_s:.6f} km/s; "
        f"method={solution.method_used}"
    )
    assert velocity_error_km_s <= VELOCITY_ERROR_TOLERANCE_KM_S, (
        f"IOD velocity error {velocity_error_km_s:.6f} km/s exceeds "
        f"{VELOCITY_ERROR_TOLERANCE_KM_S:.6f} km/s; "
        f"position error={position_error_km:.3f} km; "
        f"method={solution.method_used}"
    )


def _ra_dec_from_los(los_km: np.ndarray) -> tuple[float, float]:
    los_hat = _unit(los_km)
    ra = float(math.atan2(los_hat[1], los_hat[0]) % (2.0 * math.pi))
    dec = float(math.asin(np.clip(los_hat[2], -1.0, 1.0)))
    return ra, dec


def _make_correlated_observation(
    *,
    sensor_id: str,
    timestamp: datetime,
    observer_position_km: np.ndarray,
    target_position_km: np.ndarray,
    object_class: str = "Debris",
) -> CorrelatedObservation:
    ra, dec = _ra_dec_from_los(target_position_km - observer_position_km)

    sigma_rad = math.radians(2.9 / 3600.0)

    return CorrelatedObservation(
        obs_id=uuid4(),
        detection_id=uuid4(),
        sensor_id=sensor_id,
        timestamp=timestamp,
        ra=ra,
        dec=dec,
        ra_sigma=sigma_rad,
        dec_sigma=sigma_rad,
        observer_position_eci=(
            np.asarray(observer_position_km, dtype=np.float64) * 1000.0
        ),
        observer_velocity_eci=np.zeros(3, dtype=np.float64),
        object_class=object_class,
        confidence=0.95,
    )


def _make_iod_observation(
    *,
    timestamp: datetime,
    observer_position_km: np.ndarray,
    observer_velocity_km_s: np.ndarray,
    target_position_km: np.ndarray,
    include_range: bool,
) -> IODObservation:
    los_km = target_position_km - observer_position_km
    range_km = float(np.linalg.norm(los_km))
    ra, dec = _ra_dec_from_los(los_km)
    sigma_rad = math.radians(2.9 / 3600.0)

    return IODObservation(
        timestamp=timestamp,
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


def test_cross_sensor_observations_for_same_target_share_one_uct() -> None:
    engine = CorrelationEngine(CorrelationConfig(min_arc_length_deg=0.0))

    host1_r, _ = _circular_state(phase_rad=0.0)
    host2_r, _ = _circular_state(phase_rad=0.00005)
    debris_r = host1_r + np.array(
        [0.0, 20.0, 0.0],
        dtype=np.float64,
    )

    observations = [
        _make_correlated_observation(
            sensor_id="sensor-a",
            timestamp=EPOCH,
            observer_position_km=host1_r,
            target_position_km=debris_r,
        ),
        _make_correlated_observation(
            sensor_id="sensor-b",
            timestamp=EPOCH + timedelta(seconds=10.0),
            observer_position_km=host2_r,
            target_position_km=debris_r,
        ),
        _make_correlated_observation(
            sensor_id="sensor-a",
            timestamp=EPOCH + timedelta(seconds=20.0),
            observer_position_km=host1_r,
            target_position_km=debris_r,
        ),
    ]

    correlation_results = [
        engine.correlate(observation)
        for observation in observations
    ]

    assert correlation_results[0][1] is True
    assert correlation_results[1][1] is False
    assert correlation_results[2][1] is False

    assert len(engine.ucts) == 1

    uct = next(iter(engine.ucts.values()))
    assert uct.observation_count == 3
    assert set(uct.sensor_ids) == {"sensor-a", "sensor-b"}
    assert uct.is_ready_for_iod(engine.config)


def test_cross_sensor_correlation_keeps_separated_targets_apart() -> None:
    engine = CorrelationEngine(CorrelationConfig())

    host_r, _ = _circular_state(phase_rad=0.0)
    debris_a_r = host_r + np.array(
        [0.0, 20.0, 0.0],
        dtype=np.float64,
    )
    debris_b_r = host_r + np.array(
        [20.0, 0.0, 0.0],
        dtype=np.float64,
    )

    first_uct, first_is_new = engine.correlate(
        _make_correlated_observation(
            sensor_id="sensor-a",
            timestamp=EPOCH,
            observer_position_km=host_r,
            target_position_km=debris_a_r,
        )
    )
    second_uct, second_is_new = engine.correlate(
        _make_correlated_observation(
            sensor_id="sensor-b",
            timestamp=EPOCH + timedelta(seconds=10.0),
            observer_position_km=host_r,
            target_position_km=debris_b_r,
        )
    )

    assert first_is_new is True
    assert second_is_new is True
    assert first_uct.uct_id != second_uct.uct_id
    assert len(engine.ucts) == 2


def test_iod_solver_selects_range_angles_when_all_observations_are_ranged() -> None:
    host_r0, _ = _circular_state(phase_rad=0.0)
    debris_r0 = host_r0 + np.array(
        [0.0, 20.0, 0.0],
        dtype=np.float64,
    )

    observations = []
    for t_seconds in [0.0, 10.0, 20.0]:
        host_r, host_v = _circular_state_at_elapsed_time(
            initial_position_km=host_r0,
            elapsed_seconds=t_seconds,
        )
        debris_r, _ = _circular_state_at_elapsed_time(
            initial_position_km=debris_r0,
            elapsed_seconds=t_seconds,
        )
        observations.append(
            _make_iod_observation(
                timestamp=EPOCH + timedelta(seconds=t_seconds),
                observer_position_km=host_r,
                observer_velocity_km_s=host_v,
                target_position_km=debris_r,
                include_range=True,
            )
        )

    solution = IODSolver().solve(
        observations=observations,
        track_id=uuid4(),
    )

    assert solution.method_used == "range+angles"
    assert solution.attempted_methods is not None
    assert solution.attempted_methods[0]["method"] == "range+angles"

    truth_elapsed_seconds = (solution.epoch - EPOCH).total_seconds()
    truth_r, truth_v = _circular_state_at_elapsed_time(
        initial_position_km=debris_r0,
        elapsed_seconds=truth_elapsed_seconds,
    )
    _assert_solution_matches_truth(
        solution=solution,
        truth_position_km=truth_r,
        truth_velocity_km_s=truth_v,
    )

def test_iod_solver_uses_angles_only_dispatch_when_ranges_are_absent() -> None:
    host_r0, _ = _circular_state(phase_rad=0.0)
    debris_r0 = host_r0 + np.array(
        [0.0, 900.0, 0.0],
        dtype=np.float64,
    )

    observations = []
    for t_seconds in [0.0, 300.0, 600.0]:
        host_r, host_v = _circular_state_at_elapsed_time(
            initial_position_km=host_r0,
            elapsed_seconds=t_seconds,
        )
        debris_r, _ = _circular_state_at_elapsed_time(
            initial_position_km=debris_r0,
            elapsed_seconds=t_seconds,
        )
        observations.append(
            _make_iod_observation(
                timestamp=EPOCH + timedelta(seconds=t_seconds),
                observer_position_km=host_r,
                observer_velocity_km_s=host_v,
                target_position_km=debris_r,
                include_range=False,
            )
        )

    solution = IODSolver().solve(
        observations=observations,
        track_id=uuid4(),
    )

    attempted_methods = [
        attempt["method"]
        for attempt in (solution.attempted_methods or [])
    ]

    assert "range+angles" not in attempted_methods
    assert "range-search" in attempted_methods
    assert solution.method_used != "range+angles"

    truth_elapsed_seconds = (solution.epoch - EPOCH).total_seconds()
    truth_r, truth_v = _circular_state_at_elapsed_time(
        initial_position_km=debris_r0,
        elapsed_seconds=truth_elapsed_seconds,
    )

    # SCRUM-401 permits either a truth-accurate state or an honest rejection.
    # This fixture is constructed with all three LOS vectors in the equatorial
    # plane, and therefore has rank-deficient angles-only geometry.
    if solution.success:
        _assert_solution_matches_truth(
            solution=solution,
            truth_position_km=truth_r,
            truth_velocity_km_s=truth_v,
        )
    else:
        assert solution.error_message is not None
        assert "rank-deficient line-of-sight geometry" in solution.error_message
        assert "estimate_orbit_from_directions" not in attempted_methods

def test_angles_only_range_search_refines_observable_three_range_solution() -> None:
    host_r0, _ = _circular_state(phase_rad=0.0)
    debris_r0 = host_r0 + np.array(
        [0.0, 900.0, 0.0],
        dtype=np.float64,
    )
    inclination_rad = math.radians(1.0)

    observations = []
    truth_states = []
    for t_seconds in [0.0, 300.0, 600.0]:
        host_r, host_v = _circular_state_at_elapsed_time(
            initial_position_km=host_r0,
            elapsed_seconds=t_seconds,
        )
        debris_r_equatorial, debris_v_equatorial = (
            _circular_state_at_elapsed_time(
                initial_position_km=debris_r0,
                elapsed_seconds=t_seconds,
            )
        )
        debris_r = _rotate_about_x(
            debris_r_equatorial,
            inclination_rad,
        )
        debris_v = _rotate_about_x(
            debris_v_equatorial,
            inclination_rad,
        )
        truth_states.append((debris_r, debris_v))
        observations.append(
            _make_iod_observation(
                timestamp=EPOCH + timedelta(seconds=t_seconds),
                observer_position_km=host_r,
                observer_velocity_km_s=host_v,
                target_position_km=debris_r,
                include_range=False,
            )
        )

    solution = IODSolver().solve(
        observations=observations,
        track_id=uuid4(),
    )

    attempted_methods = [
        attempt["method"]
        for attempt in (solution.attempted_methods or [])
    ]
    assert attempted_methods[0] == "range-search"
    assert solution.method_used == "range-search"

    truth_r, truth_v = truth_states[1]
    _assert_solution_matches_truth(
        solution=solution,
        truth_position_km=truth_r,
        truth_velocity_km_s=truth_v,
    )



def test_angles_only_range_search_rejects_near_coplanar_weak_observability() -> None:
    host_r0, _ = _circular_state(phase_rad=0.0)
    debris_r0 = host_r0 + np.array(
        [0.0, 900.0, 0.0],
        dtype=np.float64,
    )

    observations = []
    for t_seconds in [0.0, 300.0, 600.0]:
        host_r, host_v = _circular_state_at_elapsed_time(
            initial_position_km=host_r0,
            elapsed_seconds=t_seconds,
        )
        debris_r, _ = _circular_state_at_elapsed_time(
            initial_position_km=debris_r0,
            elapsed_seconds=t_seconds,
        )
        observations.append(
            _make_iod_observation(
                timestamp=EPOCH + timedelta(seconds=t_seconds),
                observer_position_km=host_r,
                observer_velocity_km_s=host_v,
                target_position_km=debris_r,
                include_range=False,
            )
        )

    # Construct a technically full-rank but still near-coplanar case by
    # perturbing the middle declination by one stated measurement sigma.
    observations[1].dec += observations[1].dec_sigma

    solution = IODSolver().solve(
        observations=observations,
        track_id=uuid4(),
    )

    attempted_methods = [
        attempt["method"]
        for attempt in (solution.attempted_methods or [])
    ]
    assert solution.success is False
    assert solution.error_message is not None
    assert "weakly observable range geometry" in solution.error_message
    assert "range-search" in attempted_methods
    assert "estimate_orbit_from_directions" not in attempted_methods