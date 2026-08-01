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
    host_r, host_v = _circular_state(phase_rad=0.0)
    debris_r0 = host_r + np.array(
        [0.0, 20.0, 0.0],
        dtype=np.float64,
    )
    debris_v = host_v + np.array(
        [0.0, 0.05, 0.0],
        dtype=np.float64,
    )

    observations = [
        _make_iod_observation(
            timestamp=EPOCH + timedelta(seconds=t_seconds),
            observer_position_km=host_r + host_v * t_seconds,
            observer_velocity_km_s=host_v,
            target_position_km=debris_r0 + debris_v * t_seconds,
            include_range=True,
        )
        for t_seconds in [0.0, 10.0, 20.0]
    ]

    solution = IODSolver().solve(
        observations=observations,
        track_id=uuid4(),
    )

    assert solution.success is True
    assert solution.method_used == "range+angles"
    assert solution.attempted_methods is not None
    assert solution.attempted_methods[0]["method"] == "range+angles"


def test_iod_solver_uses_angles_only_dispatch_when_ranges_are_absent() -> None:
    host_r, host_v = _circular_state(phase_rad=0.0)
    debris_r0 = host_r + np.array(
        [0.0, 900.0, 0.0],
        dtype=np.float64,
    )
    debris_v = host_v + np.array(
        [0.0, 0.05, 0.0],
        dtype=np.float64,
    )

    observations = [
        _make_iod_observation(
            timestamp=EPOCH + timedelta(seconds=t_seconds),
            observer_position_km=host_r + host_v * t_seconds,
            observer_velocity_km_s=host_v,
            target_position_km=debris_r0 + debris_v * t_seconds,
            include_range=False,
        )
        for t_seconds in [0.0, 300.0, 600.0]
    ]

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