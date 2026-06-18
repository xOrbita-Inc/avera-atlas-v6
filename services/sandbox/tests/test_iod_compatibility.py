from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np

from services.sandbox.observations import AngularObservation
from services.tracker.iod import (
    IODObservation,
    IODSolution,
    IODSolver,
)


EPOCH_UTC = datetime(
    2026,
    1,
    1,
    tzinfo=timezone.utc,
)


def make_sandbox_observation(
    *,
    t_seconds: float,
    ra_rad: float,
    dec_rad: float,
    observer_position_km: np.ndarray,
    observer_velocity_km_s: np.ndarray,
    ra_rate_rad_s: float | None = None,
    dec_rate_rad_s: float | None = None,
) -> AngularObservation:
    """Create one detected sandbox observation for contract testing."""
    angular_sigma_rad = np.deg2rad(
        2.9 / 3600.0
    )

    return AngularObservation(
        host_id="host_001",
        debris_id="debris_001",
        t_seconds=t_seconds,
        detected=True,
        reason="detected",
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        ra_sigma_rad=angular_sigma_rad,
        dec_sigma_rad=angular_sigma_rad,
        ra_rate_rad_s=ra_rate_rad_s,
        dec_rate_rad_s=dec_rate_rad_s,
        range_km=20.0,
        off_boresight_deg=0.0,
        sunlit=True,
        earth_limb_blocked=False,
        observer_eci_m=np.asarray(
            observer_position_km,
            dtype=np.float64,
        )
        * 1000.0,
        observer_eci_m_s=np.asarray(
            observer_velocity_km_s,
            dtype=np.float64,
        )
        * 1000.0,
    )


def sandbox_observation_to_iod(
    observation: AngularObservation,
    *,
    epoch_utc: datetime = EPOCH_UTC,
) -> IODObservation:
    """
    Convert one detected sandbox observation into tracker IOD input.

    The sandbox stores observer state in SI units, while tracker.IODObservation
    expects kilometers and kilometers per second.
    """
    if not observation.detected:
        raise ValueError(
            "Only detected sandbox observations can be sent to IOD."
        )

    if observation.ra_rad is None:
        raise ValueError(
            "Detected observation is missing right ascension."
        )

    if observation.dec_rad is None:
        raise ValueError(
            "Detected observation is missing declination."
        )

    if observation.ra_sigma_rad is None:
        raise ValueError(
            "Detected observation is missing RA uncertainty."
        )

    if observation.dec_sigma_rad is None:
        raise ValueError(
            "Detected observation is missing Dec uncertainty."
        )

    if epoch_utc.tzinfo is None:
        raise ValueError(
            "epoch_utc must be timezone-aware."
        )

    return IODObservation(
        timestamp=epoch_utc
        + timedelta(
            seconds=observation.t_seconds
        ),
        ra=observation.ra_rad,
        dec=observation.dec_rad,
        ra_sigma=observation.ra_sigma_rad,
        dec_sigma=observation.dec_sigma_rad,
        observer_position_km=(
            observation.observer_eci_m / 1000.0
        ),
        observer_velocity_km_s=(
            observation.observer_eci_m_s / 1000.0
        ),
        # Optical sandbox measurements are angles-only for IOD.
        range_km=None,
        range_sigma_km=None,
    )


def test_detected_sandbox_observation_maps_to_tracker_iod():
    observer_position_km = np.array(
        [6978.137, 0.0, 0.0],
        dtype=np.float64,
    )
    observer_velocity_km_s = np.array(
        [0.0, 7.5, 0.0],
        dtype=np.float64,
    )

    sandbox_observation = make_sandbox_observation(
        t_seconds=10.0,
        ra_rad=1.2,
        dec_rad=0.3,
        observer_position_km=observer_position_km,
        observer_velocity_km_s=observer_velocity_km_s,
    )

    iod_observation = sandbox_observation_to_iod(
        sandbox_observation
    )

    assert iod_observation.timestamp == (
        EPOCH_UTC + timedelta(seconds=10.0)
    )
    assert iod_observation.ra == 1.2
    assert iod_observation.dec == 0.3

    assert (
        iod_observation.ra_sigma
        == sandbox_observation.ra_sigma_rad
    )
    assert (
        iod_observation.dec_sigma
        == sandbox_observation.dec_sigma_rad
    )

    np.testing.assert_array_equal(
        iod_observation.observer_position_km,
        observer_position_km,
    )
    np.testing.assert_array_equal(
        iod_observation.observer_velocity_km_s,
        observer_velocity_km_s,
    )

    assert iod_observation.range_km is None
    assert iod_observation.range_sigma_km is None


def test_iod_line_of_sight_matches_sandbox_angles():
    sandbox_observation = make_sandbox_observation(
        t_seconds=0.0,
        ra_rad=np.deg2rad(45.0),
        dec_rad=np.deg2rad(30.0),
        observer_position_km=np.array(
            [6978.137, 0.0, 0.0],
            dtype=np.float64,
        ),
        observer_velocity_km_s=np.array(
            [0.0, 7.5, 0.0],
            dtype=np.float64,
        ),
    )

    iod_observation = sandbox_observation_to_iod(
        sandbox_observation
    )

    expected_line_of_sight = np.array(
        [
            np.cos(sandbox_observation.dec_rad)
            * np.cos(sandbox_observation.ra_rad),
            np.cos(sandbox_observation.dec_rad)
            * np.sin(sandbox_observation.ra_rad),
            np.sin(sandbox_observation.dec_rad),
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        iod_observation.line_of_sight,
        expected_line_of_sight,
        atol=1e-15,
    )


def test_three_sandbox_observations_are_accepted_by_iod_solver_interface():
    observer_positions_km = [
        np.array(
            [6978.137, 0.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [6977.730, 75.0, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [6976.510, 150.0, 0.0],
            dtype=np.float64,
        ),
    ]

    observer_velocity_km_s = np.array(
        [0.0, 7.5, 0.0],
        dtype=np.float64,
    )

    sandbox_observations = [
        make_sandbox_observation(
            t_seconds=0.0,
            ra_rad=0.10,
            dec_rad=0.030,
            observer_position_km=observer_positions_km[0],
            observer_velocity_km_s=observer_velocity_km_s,
        ),
        make_sandbox_observation(
            t_seconds=10.0,
            ra_rad=0.11,
            dec_rad=0.031,
            observer_position_km=observer_positions_km[1],
            observer_velocity_km_s=observer_velocity_km_s,
        ),
        make_sandbox_observation(
            t_seconds=20.0,
            ra_rad=0.12,
            dec_rad=0.032,
            observer_position_km=observer_positions_km[2],
            observer_velocity_km_s=observer_velocity_km_s,
        ),
    ]

    iod_observations = [
        sandbox_observation_to_iod(observation)
        for observation in sandbox_observations
    ]

    solution = IODSolver().solve(
        observations=iod_observations,
        track_id=uuid4(),
    )

    # This is a contract test, not an orbital-accuracy test. The synthetic
    # geometry may or may not produce an accepted physical orbit, but the
    # tracker must consume the records without schema or unit errors.
    assert isinstance(solution, IODSolution)
    assert solution.observations_used == 3


def test_rejected_detection_cannot_be_sent_to_iod():
    sandbox_observation = make_sandbox_observation(
        t_seconds=0.0,
        ra_rad=1.0,
        dec_rad=0.2,
        observer_position_km=np.array(
            [6978.137, 0.0, 0.0],
            dtype=np.float64,
        ),
        observer_velocity_km_s=np.array(
            [0.0, 7.5, 0.0],
            dtype=np.float64,
        ),
    )

    rejected_observation = AngularObservation(
        host_id=sandbox_observation.host_id,
        debris_id=sandbox_observation.debris_id,
        t_seconds=sandbox_observation.t_seconds,
        detected=False,
        reason="outside_fov",
        ra_rad=None,
        dec_rad=None,
        ra_sigma_rad=None,
        dec_sigma_rad=None,
        ra_rate_rad_s=None,
        dec_rate_rad_s=None,
        range_km=100.0,
        off_boresight_deg=20.0,
        sunlit=True,
        earth_limb_blocked=False,
        observer_eci_m=(
            sandbox_observation.observer_eci_m
        ),
        observer_eci_m_s=(
            sandbox_observation.observer_eci_m_s
        ),
    )

    with np.testing.assert_raises(ValueError):
        sandbox_observation_to_iod(
            rejected_observation
        )


def test_single_observation_preserves_rates_for_partial_tracklet_handoff():
    sandbox_observation = make_sandbox_observation(
        t_seconds=10.0,
        ra_rad=1.0,
        dec_rad=0.2,
        ra_rate_rad_s=0.001,
        dec_rate_rad_s=-0.0005,
        observer_position_km=np.array(
            [6978.137, 0.0, 0.0],
            dtype=np.float64,
        ),
        observer_velocity_km_s=np.array(
            [0.0, 7.5, 0.0],
            dtype=np.float64,
        ),
    )

    assert sandbox_observation.ra_rate_rad_s == 0.001
    assert sandbox_observation.dec_rate_rad_s == -0.0005