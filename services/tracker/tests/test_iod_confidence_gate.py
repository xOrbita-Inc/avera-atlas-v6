from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pytest

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

import iod as iod_module
from iod import (
    IODConfidenceVerdict,
    IODObservation,
    IODSolver,
    classify_iod_confidence,
)


EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ranged_observations() -> list[IODObservation]:
    observations: list[IODObservation] = []
    for elapsed_seconds in (0.0, 300.0, 600.0):
        observations.append(
            IODObservation(
                timestamp=EPOCH + timedelta(seconds=elapsed_seconds),
                ra=0.0,
                dec=0.0,
                ra_sigma=np.deg2rad(2.9 / 3600.0),
                dec_sigma=np.deg2rad(2.9 / 3600.0),
                observer_position_km=np.array(
                    [6978.137, 0.0, 0.0],
                    dtype=np.float64,
                ),
                observer_velocity_km_s=np.array(
                    [0.0, 7.5, 0.0],
                    dtype=np.float64,
                ),
                range_km=900.0,
                range_sigma_km=0.1,
            )
        )
    return observations


def _install_ranged_solution(
    monkeypatch: pytest.MonkeyPatch,
    rms_arcsec: float,
) -> None:
    position_km = np.array([7000.0, 0.0, 0.0], dtype=np.float64)
    velocity_km_s = np.array([0.0, 7.5, 0.0], dtype=np.float64)

    monkeypatch.setattr(
        iod_module,
        "range_angles_iod",
        lambda observations, mu: (
            position_km.copy(),
            velocity_km_s.copy(),
            observations[len(observations) // 2].timestamp,
            "range+angles test solution",
        ),
    )
    monkeypatch.setattr(
        iod_module,
        "compute_residuals",
        lambda observations, position, velocity, epoch, mu: (
            rms_arcsec,
            [rms_arcsec for _ in observations],
        ),
    )
    monkeypatch.setattr(
        iod_module,
        "state_to_elements",
        lambda position, velocity, mu: {
            "semi_major_axis_km": 7000.0,
            "eccentricity": 0.01,
            "inclination_deg": 0.0,
            "raan_deg": 0.0,
            "arg_perigee_deg": 0.0,
            "true_anomaly_deg": 0.0,
            "perigee_km": 551.863,
            "apogee_km": 691.863,
        },
    )


@pytest.mark.parametrize(
    ("rms_arcsec", "expected"),
    [
        (0.0, IODConfidenceVerdict.CONFIDENT),
        (300.0, IODConfidenceVerdict.CONFIDENT),
        (300.000001, IODConfidenceVerdict.DEGRADED),
        (900.0, IODConfidenceVerdict.DEGRADED),
        (900.000001, IODConfidenceVerdict.REJECTED),
    ],
)
def test_confidence_classifier_uses_locked_boundaries(
    rms_arcsec: float,
    expected: IODConfidenceVerdict,
) -> None:
    assert classify_iod_confidence(rms_arcsec) == expected


@pytest.mark.parametrize(
    "invalid_rms",
    [None, -1.0, float("nan"), float("inf")],
)
def test_confidence_classifier_rejects_invalid_rms(
    invalid_rms: float | None,
) -> None:
    assert (
        classify_iod_confidence(invalid_rms)
        == IODConfidenceVerdict.REJECTED
    )


@pytest.mark.parametrize(
    (
        "rms_arcsec",
        "expected_verdict",
        "expected_success",
        "expected_proceeds_to_validity",
        "expected_blocks_action",
    ),
    [
        (
            300.0,
            IODConfidenceVerdict.CONFIDENT,
            True,
            True,
            False,
        ),
        (
            600.0,
            IODConfidenceVerdict.DEGRADED,
            True,
            False,
            True,
        ),
        (
            900.000001,
            IODConfidenceVerdict.REJECTED,
            False,
            False,
            True,
        ),
    ],
)
def test_range_angles_path_applies_confidence_gate(
    monkeypatch: pytest.MonkeyPatch,
    rms_arcsec: float,
    expected_verdict: IODConfidenceVerdict,
    expected_success: bool,
    expected_proceeds_to_validity: bool,
    expected_blocks_action: bool,
) -> None:
    _install_ranged_solution(monkeypatch, rms_arcsec)

    solution = IODSolver().solve(
        observations=_ranged_observations(),
        track_id=uuid4(),
    )

    assert solution.success is expected_success
    assert solution.rms_residual_arcsec == pytest.approx(rms_arcsec)
    assert solution.confidence_verdict == expected_verdict
    assert (
        solution.proceeds_to_validity
        is expected_proceeds_to_validity
    )
    assert (
        solution.confidence_gate_blocks_autonomous_action
        is expected_blocks_action
    )

    payload = solution.to_dict()
    assert payload["confidence_verdict"] == expected_verdict.value
    assert (
        payload["proceeds_to_validity"]
        is expected_proceeds_to_validity
    )
    assert (
        payload["confidence_gate_blocks_autonomous_action"]
        is expected_blocks_action
    )
    assert payload["rms_residual_arcsec"] == pytest.approx(rms_arcsec)

    attempted = solution.attempted_methods
    assert attempted is not None
    assert attempted[0]["confidence_verdict"] == expected_verdict.value

    if expected_verdict == IODConfidenceVerdict.DEGRADED:
        assert "degraded" in attempted[0]["status"]
    elif expected_verdict == IODConfidenceVerdict.REJECTED:
        assert solution.error_message is not None
        assert "RMS residual too high" in solution.error_message


def _circular_ranged_observation(
    elapsed_seconds: float,
    *,
    radius_km: float = 7000.0,
) -> tuple[IODObservation, np.ndarray, np.ndarray]:
    mean_motion_rad_s = math.sqrt(
        iod_module.MU_EARTH_KM / radius_km**3
    )
    angle_rad = mean_motion_rad_s * elapsed_seconds

    target_position_km = np.array(
        [
            radius_km * math.cos(angle_rad),
            radius_km * math.sin(angle_rad),
            0.0,
        ],
        dtype=np.float64,
    )
    target_velocity_km_s = np.array(
        [
            -radius_km * mean_motion_rad_s * math.sin(angle_rad),
            radius_km * mean_motion_rad_s * math.cos(angle_rad),
            0.0,
        ],
        dtype=np.float64,
    )

    line_of_sight = target_position_km / np.linalg.norm(
        target_position_km
    )

    observation = IODObservation(
        timestamp=EPOCH + timedelta(seconds=elapsed_seconds),
        ra=math.atan2(line_of_sight[1], line_of_sight[0]),
        dec=math.asin(line_of_sight[2]),
        ra_sigma=np.deg2rad(2.9 / 3600.0),
        dec_sigma=np.deg2rad(2.9 / 3600.0),
        observer_position_km=np.zeros(3, dtype=np.float64),
        observer_velocity_km_s=np.zeros(3, dtype=np.float64),
        range_km=float(np.linalg.norm(target_position_km)),
        range_sigma_km=0.05,
    )
    return observation, target_position_km, target_velocity_km_s


def test_range_angles_uses_middle_epoch_for_uneven_observation_spacing(
) -> None:
    constructed = [
        _circular_ranged_observation(elapsed_seconds)
        for elapsed_seconds in (0.0, 0.1, 10.0)
    ]
    observations = [item[0] for item in constructed]

    position_km, velocity_km_s, epoch, status = (
        iod_module.range_angles_iod(
            observations,
            iod_module.MU_EARTH_KM,
        )
    )

    assert position_km is not None
    assert velocity_km_s is not None
    assert epoch == EPOCH + timedelta(seconds=0.1)
    assert status == "Range+angles IOD success"

    expected_position_km = constructed[1][1]
    expected_velocity_km_s = constructed[1][2]

    assert np.linalg.norm(
        position_km - expected_position_km
    ) < 1.0e-9
    assert np.linalg.norm(
        velocity_km_s - expected_velocity_km_s
    ) < 1.0e-6

    solution = IODSolver().solve(
        observations=observations,
        track_id=uuid4(),
    )

    assert solution.success is True
    assert (
        solution.confidence_verdict
        == IODConfidenceVerdict.CONFIDENT
    )
    assert solution.proceeds_to_validity is True
    assert solution.rms_residual_arcsec is not None
    assert solution.rms_residual_arcsec <= 300.0

