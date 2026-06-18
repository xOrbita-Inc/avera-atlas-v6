import math

import numpy as np
import pytest

from services.sandbox.models import PhysicalProperties, SimObject
from services.sandbox.sensor_model import (
    RE_EARTH_KM,
    DetectionResult,
    SensorConfig,
    angle_between_deg,
    boresight_from_velocity,
    detect_object,
    is_sunlit,
    is_within_fov,
    passes_earth_limb_gate,
    passes_range_gate,
    range_cutoff_km_for_size_bin,
)


def make_host(
    *,
    position_km: np.ndarray | None = None,
    velocity_km_s: np.ndarray | None = None,
) -> SimObject:
    if position_km is None:
        position_km = np.array(
            [6978.137, 0.0, 0.0],
            dtype=np.float64,
        )

    if velocity_km_s is None:
        velocity_km_s = np.array(
            [0.0, 7.5, 0.0],
            dtype=np.float64,
        )

    return SimObject(
        object_id="host_001",
        kind="host",
        r_eci_km=np.asarray(
            position_km,
            dtype=np.float64,
        ),
        v_eci_km_s=np.asarray(
            velocity_km_s,
            dtype=np.float64,
        ),
        physical=PhysicalProperties(
            size_bin="10cm",
            characteristic_size_m=1.0,
            mass_kg=12.0,
            area_m2=0.1,
            cd=2.2,
        ),
    )


def make_debris(
    position_km: np.ndarray,
    *,
    size_bin: str = "5cm",
    object_id: str = "debris_001",
) -> SimObject:
    size_lookup = {
        "1cm": (0.01, 0.001, 0.0001),
        "5cm": (0.05, 0.05, 0.002),
        "10cm": (0.10, 0.4, 0.008),
    }

    size_m, mass_kg, area_m2 = size_lookup[size_bin]

    return SimObject(
        object_id=object_id,
        kind="debris",
        r_eci_km=np.asarray(
            position_km,
            dtype=np.float64,
        ),
        v_eci_km_s=np.array(
            [0.0, 7.4, 0.0],
            dtype=np.float64,
        ),
        physical=PhysicalProperties(
            size_bin=size_bin,
            characteristic_size_m=size_m,
            mass_kg=mass_kg,
            area_m2=area_m2,
            cd=2.2,
        ),
    )


def test_default_sensor_config_matches_tt240_40_ticket():
    config = SensorConfig()

    assert config.fov_full_angle_deg == 3.7
    assert config.aperture_mm == 40.0
    assert config.focal_length_mm == 240.0
    assert config.angular_resolution_arcsec == 2.9
    assert config.sensor_mode == "neuromorphic_event"
    assert config.duty_cycle == 1.0
    assert config.pointing_mode == "nadir_minus_30"
    assert config.nadir_offset_deg == 30.0


@pytest.mark.parametrize(
    ("size_bin", "expected_cutoff_km"),
    [
        ("1cm", 12.0),
        ("5cm", 59.0),
        ("10cm", 117.0),
    ],
)
def test_range_cutoffs_match_ticket(
    size_bin: str,
    expected_cutoff_km: float,
):
    assert (
        range_cutoff_km_for_size_bin(size_bin)
        == expected_cutoff_km
    )


def test_range_cutoff_rejects_unknown_size_bin():
    with pytest.raises(ValueError):
        range_cutoff_km_for_size_bin("2cm")


def test_velocity_aligned_boresight_points_with_velocity():
    host = make_host()

    boresight = boresight_from_velocity(
        host_r_eci_km=host.r_eci_km,
        host_v_eci_km_s=host.v_eci_km_s,
        pointing_mode="velocity_aligned",
    )

    expected = np.array(
        [0.0, 1.0, 0.0],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        boresight,
        expected,
        atol=1e-12,
    )


def test_nadir_offset_boresight_is_30_degrees_from_nadir():
    host = make_host()

    boresight = boresight_from_velocity(
        host_r_eci_km=host.r_eci_km,
        host_v_eci_km_s=host.v_eci_km_s,
        pointing_mode="nadir_minus_30",
        nadir_offset_deg=30.0,
    )

    nadir = -host.r_eci_km

    offset_deg = angle_between_deg(
        boresight,
        nadir,
    )

    assert math.isclose(
        offset_deg,
        30.0,
        abs_tol=1e-10,
    )


def test_fov_accepts_line_of_sight_on_boresight():
    boresight = np.array(
        [0.0, 1.0, 0.0],
        dtype=np.float64,
    )

    inside_fov, off_boresight_deg = is_within_fov(
        boresight_eci=boresight,
        line_of_sight_eci=boresight,
        fov_full_angle_deg=3.7,
    )

    assert inside_fov
    assert math.isclose(
        off_boresight_deg,
        0.0,
        abs_tol=1e-12,
    )


def test_fov_accepts_exact_half_angle_boundary():
    half_angle_rad = math.radians(1.85)

    boresight = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float64,
    )
    line_of_sight = np.array(
        [
            math.cos(half_angle_rad),
            math.sin(half_angle_rad),
            0.0,
        ],
        dtype=np.float64,
    )

    inside_fov, off_boresight_deg = is_within_fov(
        boresight_eci=boresight,
        line_of_sight_eci=line_of_sight,
        fov_full_angle_deg=3.7,
    )

    assert inside_fov
    assert math.isclose(
        off_boresight_deg,
        1.85,
        abs_tol=1e-10,
    )


def test_fov_rejects_just_outside_half_angle():
    angle_rad = math.radians(1.851)

    boresight = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float64,
    )
    line_of_sight = np.array(
        [
            math.cos(angle_rad),
            math.sin(angle_rad),
            0.0,
        ],
        dtype=np.float64,
    )

    inside_fov, off_boresight_deg = is_within_fov(
        boresight_eci=boresight,
        line_of_sight_eci=line_of_sight,
        fov_full_angle_deg=3.7,
    )

    assert not inside_fov
    assert off_boresight_deg > 1.85


@pytest.mark.parametrize(
    ("size_bin", "cutoff_km"),
    [
        ("1cm", 12.0),
        ("5cm", 59.0),
        ("10cm", 117.0),
    ],
)
def test_range_gate_accepts_exact_cutoff(
    size_bin: str,
    cutoff_km: float,
):
    host_position = np.array(
        [7000.0, 0.0, 0.0],
        dtype=np.float64,
    )
    debris_position = host_position + np.array(
        [0.0, cutoff_km, 0.0],
        dtype=np.float64,
    )

    passes, measured_range_km, measured_cutoff_km = (
        passes_range_gate(
            host_r_eci_km=host_position,
            debris_r_eci_km=debris_position,
            size_bin=size_bin,
        )
    )

    assert passes
    assert math.isclose(
        measured_range_km,
        cutoff_km,
        abs_tol=1e-12,
    )
    assert measured_cutoff_km == cutoff_km


@pytest.mark.parametrize(
    ("size_bin", "cutoff_km"),
    [
        ("1cm", 12.0),
        ("5cm", 59.0),
        ("10cm", 117.0),
    ],
)
def test_range_gate_rejects_above_cutoff(
    size_bin: str,
    cutoff_km: float,
):
    host_position = np.array(
        [7000.0, 0.0, 0.0],
        dtype=np.float64,
    )
    debris_position = host_position + np.array(
        [0.0, cutoff_km + 0.001, 0.0],
        dtype=np.float64,
    )

    passes, measured_range_km, measured_cutoff_km = (
        passes_range_gate(
            host_r_eci_km=host_position,
            debris_r_eci_km=debris_position,
            size_bin=size_bin,
        )
    )

    assert not passes
    assert measured_range_km > cutoff_km
    assert measured_cutoff_km == cutoff_km


def test_sunlit_object_on_sunward_side_is_visible():
    object_position = np.array(
        [7000.0, 0.0, 0.0],
        dtype=np.float64,
    )

    assert is_sunlit(
        object_r_eci_km=object_position,
        sun_dir_eci=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    )


def test_object_inside_earth_shadow_is_not_sunlit():
    object_position = np.array(
        [-7000.0, 0.0, 0.0],
        dtype=np.float64,
    )

    assert not is_sunlit(
        object_r_eci_km=object_position,
        sun_dir_eci=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    )


def test_object_behind_earth_but_outside_shadow_cylinder_is_sunlit():
    object_position = np.array(
        [-7000.0, RE_EARTH_KM + 10.0, 0.0],
        dtype=np.float64,
    )

    assert is_sunlit(
        object_r_eci_km=object_position,
        sun_dir_eci=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    )


def test_earth_limb_gate_rejects_nadir_pointing():
    host = make_host()

    limb_clear = passes_earth_limb_gate(
        host_r_eci_km=host.r_eci_km,
        boresight_eci=-host.r_eci_km,
        fov_full_angle_deg=3.7,
    )

    assert not limb_clear


def test_earth_limb_gate_accepts_zenith_pointing():
    host = make_host()

    limb_clear = passes_earth_limb_gate(
        host_r_eci_km=host.r_eci_km,
        boresight_eci=host.r_eci_km,
        fov_full_angle_deg=3.7,
    )

    assert limb_clear


def test_detection_rejects_target_outside_fov():
    host = make_host()
    debris = make_debris(
        host.r_eci_km
        + np.array(
            [20.0, 0.0, 0.0],
            dtype=np.float64,
        )
    )

    result = detect_object(
        host=host,
        debris=debris,
        sensor_cfg=SensorConfig(
            pointing_mode="velocity_aligned"
        ),
        sun_dir_eci=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
    )

    assert not result.detected
    assert result.reason == "outside_fov"
    assert result.off_boresight_deg is not None
    assert result.off_boresight_deg > 1.85


def test_detection_rejects_target_outside_range():
    host = make_host()
    debris = make_debris(
        host.r_eci_km
        + np.array(
            [0.0, 60.0, 0.0],
            dtype=np.float64,
        ),
        size_bin="5cm",
    )

    result = detect_object(
        host=host,
        debris=debris,
        sensor_cfg=SensorConfig(
            pointing_mode="velocity_aligned"
        ),
        sun_dir_eci=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
    )

    assert not result.detected
    assert result.reason == "outside_range"
    assert result.range_km == 60.0
    assert result.range_cutoff_km == 59.0


def test_detection_rejects_eclipsed_target():
    host = make_host(
        position_km=np.array(
            [-6978.137, 0.0, 0.0],
            dtype=np.float64,
        ),
        velocity_km_s=np.array(
            [0.0, 7.5, 0.0],
            dtype=np.float64,
        ),
    )
    debris = make_debris(
        host.r_eci_km
        + np.array(
            [0.0, 20.0, 0.0],
            dtype=np.float64,
        )
    )

    result = detect_object(
        host=host,
        debris=debris,
        sensor_cfg=SensorConfig(
            pointing_mode="velocity_aligned"
        ),
        sun_dir_eci=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    )

    assert not result.detected
    assert result.reason == "not_sunlit"
    assert not result.sunlit


def test_detection_rejects_earth_limb_view():
    host = make_host()
    sensor_config = SensorConfig(
        pointing_mode="nadir_minus_30",
        nadir_offset_deg=0.0,
    )

    boresight = boresight_from_velocity(
        host_r_eci_km=host.r_eci_km,
        host_v_eci_km_s=host.v_eci_km_s,
        pointing_mode=sensor_config.pointing_mode,
        nadir_offset_deg=sensor_config.nadir_offset_deg,
    )

    debris = make_debris(
        host.r_eci_km + 20.0 * boresight,
        size_bin="5cm",
    )

    result = detect_object(
        host=host,
        debris=debris,
        sensor_cfg=sensor_config,
        sun_dir_eci=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
    )

    assert not result.detected
    assert result.reason == "earth_limb_blocked"
    assert result.sunlit
    assert result.earth_limb_blocked


def test_detection_accepts_valid_geometry():
    host = make_host()
    debris = make_debris(
        host.r_eci_km
        + np.array(
            [0.0, 20.0, 0.0],
            dtype=np.float64,
        ),
        size_bin="5cm",
    )

    result = detect_object(
        host=host,
        debris=debris,
        sensor_cfg=SensorConfig(
            pointing_mode="velocity_aligned"
        ),
        sun_dir_eci=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
    )

    assert isinstance(result, DetectionResult)
    assert result.detected
    assert result.reason == "detected"
    assert result.range_km == 20.0
    assert result.range_cutoff_km == 59.0
    assert math.isclose(
        result.off_boresight_deg or 0.0,
        0.0,
        abs_tol=1e-12,
    )
    assert result.sunlit
    assert not result.earth_limb_blocked


def test_detection_rejects_co_located_objects():
    host = make_host()
    debris = make_debris(
        host.r_eci_km.copy()
    )

    result = detect_object(
        host=host,
        debris=debris,
        sensor_cfg=SensorConfig(),
    )

    assert not result.detected
    assert result.reason == "co_located"
    assert result.range_km == 0.0
    assert result.range_cutoff_km is None


@pytest.mark.parametrize(
    "invalid_config",
    [
        {"fov_full_angle_deg": 0.0},
        {"aperture_mm": 0.0},
        {"focal_length_mm": -1.0},
        {"angular_resolution_arcsec": -1.0},
        {"duty_cycle": 0.0},
        {"duty_cycle": 1.1},
        {"nadir_offset_deg": -1.0},
        {"nadir_offset_deg": 181.0},
    ],
)
def test_sensor_config_rejects_invalid_values(
    invalid_config: dict[str, float],
):
    with pytest.raises(ValueError):
        SensorConfig(**invalid_config)