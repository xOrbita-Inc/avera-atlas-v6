from __future__ import annotations

import math

import numpy as np
import pytest

from services.sandbox.scenario import (
    RE_EARTH_KM,
    circular_speed_km_s,
    create_forced_detection_scenario,
    create_range_gate_scenario,
    create_sensor_transit_scenario,
)
from services.sandbox.sensor_model import (
    SensorConfig,
    detect_object,
)


def test_forced_detection_scenario_has_expected_objects():
    objects = create_forced_detection_scenario()

    assert set(objects) == {
        "host_001",
        "debris_001",
    }

    host = objects["host_001"]
    debris = objects["debris_001"]

    assert host.kind == "host"
    assert debris.kind == "debris"
    assert debris.physical.size_bin == "5cm"


def test_forced_detection_scenario_uses_requested_altitude_and_range():
    objects = create_forced_detection_scenario(
        altitude_km=600.0,
        range_km=20.0,
        size_bin="5cm",
    )

    host = objects["host_001"]
    debris = objects["debris_001"]

    host_altitude_km = (
        np.linalg.norm(host.r_eci_km)
        - RE_EARTH_KM
    )
    separation_km = np.linalg.norm(
        debris.r_eci_km - host.r_eci_km
    )

    assert math.isclose(
        host_altitude_km,
        600.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        separation_km,
        20.0,
        abs_tol=1e-12,
    )


def test_forced_detection_scenario_places_target_on_velocity_boresight():
    objects = create_forced_detection_scenario()

    host = objects["host_001"]
    debris = objects["debris_001"]

    line_of_sight = (
        debris.r_eci_km - host.r_eci_km
    )
    line_of_sight /= np.linalg.norm(
        line_of_sight
    )

    velocity_direction = (
        host.v_eci_km_s
        / np.linalg.norm(host.v_eci_km_s)
    )

    np.testing.assert_allclose(
        line_of_sight,
        velocity_direction,
        atol=1e-12,
    )


def test_forced_detection_scenario_passes_sensor_gates():
    objects = create_forced_detection_scenario()

    result = detect_object(
        host=objects["host_001"],
        debris=objects["debris_001"],
        sensor_cfg=SensorConfig(
            pointing_mode="velocity_aligned"
        ),
        sun_dir_eci=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
    )

    assert result.detected
    assert result.reason == "detected"
    assert result.range_cutoff_km == 59.0


@pytest.mark.parametrize(
    ("size_bin", "expected_cutoff_km"),
    [
        ("1cm", 12.0),
        ("5cm", 59.0),
        ("10cm", 117.0),
    ],
)
def test_range_gate_scenario_places_targets_at_and_beyond_cutoff(
    size_bin: str,
    expected_cutoff_km: float,
):
    objects = create_range_gate_scenario(
        size_bin=size_bin,
        outside_by_km=0.001,
    )

    host = objects["host_001"]
    at_cutoff = objects["debris_at_cutoff"]
    outside_cutoff = objects[
        "debris_outside_cutoff"
    ]

    at_cutoff_range_km = np.linalg.norm(
        at_cutoff.r_eci_km
        - host.r_eci_km
    )
    outside_range_km = np.linalg.norm(
        outside_cutoff.r_eci_km
        - host.r_eci_km
    )

    assert math.isclose(
        at_cutoff_range_km,
        expected_cutoff_km,
        abs_tol=1e-12,
    )
    assert math.isclose(
        outside_range_km,
        expected_cutoff_km + 0.001,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    "size_bin",
    [
        "1cm",
        "5cm",
        "10cm",
    ],
)
def test_range_gate_scenario_produces_expected_detection_results(
    size_bin: str,
):
    objects = create_range_gate_scenario(
        size_bin=size_bin,
        outside_by_km=0.001,
    )

    sensor_config = SensorConfig(
        pointing_mode="velocity_aligned"
    )
    sun_direction = np.array(
        [0.0, 1.0, 0.0],
        dtype=np.float64,
    )

    at_cutoff_result = detect_object(
        host=objects["host_001"],
        debris=objects["debris_at_cutoff"],
        sensor_cfg=sensor_config,
        sun_dir_eci=sun_direction,
    )
    outside_result = detect_object(
        host=objects["host_001"],
        debris=objects[
            "debris_outside_cutoff"
        ],
        sensor_cfg=sensor_config,
        sun_dir_eci=sun_direction,
    )

    assert at_cutoff_result.detected
    assert at_cutoff_result.reason == "detected"

    assert not outside_result.detected
    assert outside_result.reason == "outside_range"


@pytest.mark.parametrize(
    ("transit_case", "expected_speed_km_s"),
    [
        ("co_orbital", 0.5),
        ("moderate_crossing", 4.0),
        ("high_crossing", 10.0),
    ],
)
def test_transit_scenario_uses_expected_relative_cross_track_speed(
    transit_case: str,
    expected_speed_km_s: float,
):
    objects = create_sensor_transit_scenario(
        transit_case
    )

    host = objects["host_001"]
    debris = objects["debris_001"]

    relative_velocity_km_s = (
        debris.v_eci_km_s
        - host.v_eci_km_s
    )

    assert math.isclose(
        relative_velocity_km_s[0],
        0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        relative_velocity_km_s[1],
        0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        relative_velocity_km_s[2],
        expected_speed_km_s,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    "transit_case",
    [
        "co_orbital",
        "moderate_crossing",
        "high_crossing",
    ],
)
def test_transit_scenario_starts_inside_fov_and_range(
    transit_case: str,
):
    objects = create_sensor_transit_scenario(
        transit_case,
        range_km=20.0,
        size_bin="5cm",
    )

    result = detect_object(
        host=objects["host_001"],
        debris=objects["debris_001"],
        sensor_cfg=SensorConfig(
            pointing_mode="velocity_aligned"
        ),
        sun_dir_eci=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
    )

    assert result.detected
    assert result.reason == "detected"
    assert math.isclose(
        result.range_km or 0.0,
        20.0,
        abs_tol=1e-12,
    )


def test_transit_scenarios_have_increasing_relative_speed():
    co_orbital = create_sensor_transit_scenario(
        "co_orbital"
    )
    moderate = create_sensor_transit_scenario(
        "moderate_crossing"
    )
    high = create_sensor_transit_scenario(
        "high_crossing"
    )

    def relative_speed(
        objects,
    ) -> float:
        return float(
            np.linalg.norm(
                objects["debris_001"].v_eci_km_s
                - objects["host_001"].v_eci_km_s
            )
        )

    assert (
        relative_speed(co_orbital)
        < relative_speed(moderate)
        < relative_speed(high)
    )


def test_host_velocity_matches_circular_speed_at_requested_altitude():
    altitude_km = 600.0

    objects = create_forced_detection_scenario(
        altitude_km=altitude_km
    )

    host = objects["host_001"]

    expected_speed_km_s = circular_speed_km_s(
        RE_EARTH_KM + altitude_km
    )
    measured_speed_km_s = float(
        np.linalg.norm(host.v_eci_km_s)
    )

    assert math.isclose(
        measured_speed_km_s,
        expected_speed_km_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    ("function_name", "kwargs"),
    [
        (
            "forced_detection",
            {"range_km": 0.0},
        ),
        (
            "range_gate",
            {
                "size_bin": "5cm",
                "outside_by_km": 0.0,
            },
        ),
        (
            "range_gate",
            {
                "size_bin": "2cm",
            },
        ),
        (
            "transit",
            {
                "transit_case": "unsupported",
            },
        ),
    ],
)
def test_sensor_scenarios_reject_invalid_inputs(
    function_name: str,
    kwargs: dict,
):
    with pytest.raises(ValueError):
        if function_name == "forced_detection":
            create_forced_detection_scenario(
                **kwargs
            )
        elif function_name == "range_gate":
            create_range_gate_scenario(
                **kwargs
            )
        else:
            create_sensor_transit_scenario(
                **kwargs
            )

