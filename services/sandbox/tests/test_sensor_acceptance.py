from __future__ import annotations

import math

import numpy as np
import pytest

from services.sandbox.sensor_model import (
    SensorConfig,
    is_within_fov,
)


# Section 5.2 of the feasibility analysis:
#
# Co-orbital:       0.1–1 km/s  -> 4–40 s
# Moderate crossing: 3–5 km/s   -> 0.5–1.5 s
# High crossing:     7–15 km/s  -> 0.15–0.5 s
TRANSIT_EXPECTATIONS = {
    "co_orbital": {
        "relative_speed_km_s": 0.5,
        "minimum_seconds": 4.0,
        "maximum_seconds": 40.0,
    },
    "moderate_crossing": {
        "relative_speed_km_s": 4.0,
        "minimum_seconds": 0.5,
        "maximum_seconds": 1.5,
    },
    "high_crossing": {
        "relative_speed_km_s": 10.0,
        "minimum_seconds": 0.15,
        "maximum_seconds": 0.5,
    },
}


def analytic_transit_duration_seconds(
    *,
    closest_approach_range_km: float,
    relative_crossing_speed_km_s: float,
    fov_full_angle_deg: float,
) -> float:
    """
    Compute transit duration through a conical FOV for a straight-line pass.

    The target crosses perpendicular to the boresight at constant relative
    speed. The closest-approach point lies on the boresight.
    """
    if closest_approach_range_km <= 0.0:
        raise ValueError(
            "closest_approach_range_km must be positive."
        )

    if relative_crossing_speed_km_s <= 0.0:
        raise ValueError(
            "relative_crossing_speed_km_s must be positive."
        )

    if not 0.0 < fov_full_angle_deg < 180.0:
        raise ValueError(
            "fov_full_angle_deg must be between 0 and 180 degrees."
        )

    half_angle_rad = math.radians(
        0.5 * fov_full_angle_deg
    )

    half_width_km = (
        closest_approach_range_km
        * math.tan(half_angle_rad)
    )

    return (
        2.0
        * half_width_km
        / relative_crossing_speed_km_s
    )


def sampled_transit_duration_seconds(
    *,
    closest_approach_range_km: float,
    relative_crossing_speed_km_s: float,
    sensor_config: SensorConfig,
    sample_step_seconds: float = 0.001,
) -> float:
    """
    Measure a straight-line FOV transit using the production FOV gate.

    The boresight is the positive y-axis. The target passes across the z-axis
    while maintaining a fixed y-coordinate at closest approach.
    """
    if sample_step_seconds <= 0.0:
        raise ValueError(
            "sample_step_seconds must be positive."
        )

    expected_duration_seconds = (
        analytic_transit_duration_seconds(
            closest_approach_range_km=(
                closest_approach_range_km
            ),
            relative_crossing_speed_km_s=(
                relative_crossing_speed_km_s
            ),
            fov_full_angle_deg=(
                sensor_config.fov_full_angle_deg
            ),
        )
    )

    # Sample beyond both edges of the expected transit.
    half_window_seconds = (
        expected_duration_seconds
    )

    sample_times = np.arange(
        -half_window_seconds,
        half_window_seconds
        + sample_step_seconds,
        sample_step_seconds,
        dtype=np.float64,
    )

    boresight_eci = np.array(
        [0.0, 1.0, 0.0],
        dtype=np.float64,
    )

    detected_times: list[float] = []

    for time_seconds in sample_times:
        cross_track_offset_km = (
            relative_crossing_speed_km_s
            * time_seconds
        )

        line_of_sight_eci = np.array(
            [
                0.0,
                closest_approach_range_km,
                cross_track_offset_km,
            ],
            dtype=np.float64,
        )

        inside_fov, _ = is_within_fov(
            boresight_eci=boresight_eci,
            line_of_sight_eci=line_of_sight_eci,
            fov_full_angle_deg=(
                sensor_config.fov_full_angle_deg
            ),
        )

        if inside_fov:
            detected_times.append(
                float(time_seconds)
            )

    assert detected_times

    # Include one sample interval because both boundary samples represent
    # covered intervals in the discrete measurement.
    return (
        detected_times[-1]
        - detected_times[0]
        + sample_step_seconds
    )


@pytest.mark.parametrize(
    "transit_case",
    [
        "co_orbital",
        "moderate_crossing",
        "high_crossing",
    ],
)
def test_analytic_transit_duration_matches_section_5_2(
    transit_case: str,
):
    expectation = TRANSIT_EXPECTATIONS[
        transit_case
    ]

    duration_seconds = (
        analytic_transit_duration_seconds(
            # The 59 km range is the operational R_max for 5 cm debris.
            closest_approach_range_km=59.0,
            relative_crossing_speed_km_s=(
                expectation[
                    "relative_speed_km_s"
                ]
            ),
            fov_full_angle_deg=3.7,
        )
    )

    assert (
        expectation["minimum_seconds"]
        <= duration_seconds
        <= expectation["maximum_seconds"]
    )


@pytest.mark.parametrize(
    "transit_case",
    [
        "co_orbital",
        "moderate_crossing",
        "high_crossing",
    ],
)
def test_production_fov_gate_reproduces_transit_duration(
    transit_case: str,
):
    expectation = TRANSIT_EXPECTATIONS[
        transit_case
    ]
    sensor_config = SensorConfig()

    analytic_duration_seconds = (
        analytic_transit_duration_seconds(
            closest_approach_range_km=59.0,
            relative_crossing_speed_km_s=(
                expectation[
                    "relative_speed_km_s"
                ]
            ),
            fov_full_angle_deg=(
                sensor_config.fov_full_angle_deg
            ),
        )
    )

    measured_duration_seconds = (
        sampled_transit_duration_seconds(
            closest_approach_range_km=59.0,
            relative_crossing_speed_km_s=(
                expectation[
                    "relative_speed_km_s"
                ]
            ),
            sensor_config=sensor_config,
        )
    )

    assert math.isclose(
        measured_duration_seconds,
        analytic_duration_seconds,
        abs_tol=0.005,
    )

    assert (
        expectation["minimum_seconds"]
        <= measured_duration_seconds
        <= expectation["maximum_seconds"]
    )


def test_transit_durations_decrease_with_crossing_speed():
    durations = {
        transit_case: (
            analytic_transit_duration_seconds(
                closest_approach_range_km=59.0,
                relative_crossing_speed_km_s=(
                    expectation[
                        "relative_speed_km_s"
                    ]
                ),
                fov_full_angle_deg=3.7,
            )
        )
        for transit_case, expectation
        in TRANSIT_EXPECTATIONS.items()
    }

    assert (
        durations["co_orbital"]
        > durations["moderate_crossing"]
        > durations["high_crossing"]
    )


def test_neuromorphic_mode_has_no_frame_duty_penalty():
    sensor_config = SensorConfig()

    assert (
        sensor_config.sensor_mode
        == "neuromorphic_event"
    )
    assert sensor_config.duty_cycle == 1.0


@pytest.mark.parametrize(
    (
        "closest_approach_range_km",
        "relative_crossing_speed_km_s",
        "fov_full_angle_deg",
    ),
    [
        (0.0, 1.0, 3.7),
        (59.0, 0.0, 3.7),
        (59.0, 1.0, 0.0),
    ],
)
def test_transit_duration_rejects_invalid_inputs(
    closest_approach_range_km: float,
    relative_crossing_speed_km_s: float,
    fov_full_angle_deg: float,
):
    with pytest.raises(ValueError):
        analytic_transit_duration_seconds(
            closest_approach_range_km=(
                closest_approach_range_km
            ),
            relative_crossing_speed_km_s=(
                relative_crossing_speed_km_s
            ),
            fov_full_angle_deg=(
                fov_full_angle_deg
            ),
        )