import math
import random

import numpy as np

from services.sandbox.models import PhysicalProperties, SimObject
from services.sandbox.observations import (
    ARCSEC_TO_RAD,
    AngularObservation,
    estimate_angular_rates,
    generate_angular_observation,
    los_to_ra_dec,
    make_noisy_ra_dec,
    wrapped_ra_difference_rad,
)
from services.sandbox.sensor_model import SensorConfig


def make_host() -> SimObject:
    return SimObject(
        object_id="host_001",
        kind="host",
        r_eci_km=np.array(
            [6978.137, 0.0, 0.0],
            dtype=np.float64,
        ),
        v_eci_km_s=np.array(
            [0.0, 7.5, 0.0],
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
    size_bin: str = "5cm",
) -> SimObject:
    size_lookup = {
        "1cm": (0.01, 0.001, 0.0001),
        "5cm": (0.05, 0.05, 0.002),
        "10cm": (0.10, 0.4, 0.008),
    }

    size_m, mass_kg, area_m2 = size_lookup[size_bin]

    return SimObject(
        object_id="debris_001",
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


def make_detected_observation(
    *,
    t_seconds: float,
    ra_rad: float,
    dec_rad: float,
) -> AngularObservation:
    return AngularObservation(
        host_id="host_001",
        debris_id="debris_001",
        t_seconds=t_seconds,
        detected=True,
        reason="detected",
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        ra_sigma_rad=2.9 * ARCSEC_TO_RAD,
        dec_sigma_rad=2.9 * ARCSEC_TO_RAD,
        ra_rate_rad_s=None,
        dec_rate_rad_s=None,
        range_km=20.0,
        off_boresight_deg=0.0,
        sunlit=True,
        earth_limb_blocked=False,
        observer_eci_m=np.array(
            [6978137.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        observer_eci_m_s=np.array(
            [0.0, 7500.0, 0.0],
            dtype=np.float64,
        ),
    )


def test_los_to_ra_dec_basic_case():
    line_of_sight = np.array(
        [0.0, 1.0, 0.0],
        dtype=np.float64,
    )

    ra_rad, dec_rad = los_to_ra_dec(line_of_sight)

    assert math.isclose(
        ra_rad,
        math.pi / 2.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        dec_rad,
        0.0,
        abs_tol=1e-12,
    )


def test_los_to_ra_dec_normalizes_input_vector():
    line_of_sight = np.array(
        [0.0, 10.0, 0.0],
        dtype=np.float64,
    )

    ra_rad, dec_rad = los_to_ra_dec(line_of_sight)

    assert math.isclose(
        ra_rad,
        math.pi / 2.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        dec_rad,
        0.0,
        abs_tol=1e-12,
    )


def test_los_to_ra_dec_rejects_zero_vector():
    with np.testing.assert_raises(ValueError):
        los_to_ra_dec(
            np.zeros(3, dtype=np.float64)
        )


def test_noisy_angles_are_deterministic_for_same_seed():
    line_of_sight = np.array(
        [0.0, 1.0, 0.0],
        dtype=np.float64,
    )

    result_a = make_noisy_ra_dec(
        los_eci=line_of_sight,
        angular_sigma_arcsec=2.9,
        rng=random.Random(42),
    )
    result_b = make_noisy_ra_dec(
        los_eci=line_of_sight,
        angular_sigma_arcsec=2.9,
        rng=random.Random(42),
    )

    assert result_a == result_b


def test_noisy_angles_report_ticket_sigma():
    line_of_sight = np.array(
        [0.0, 1.0, 0.0],
        dtype=np.float64,
    )

    _, _, ra_sigma_rad, dec_sigma_rad = make_noisy_ra_dec(
        los_eci=line_of_sight,
        angular_sigma_arcsec=2.9,
        rng=random.Random(42),
    )

    expected_sigma_rad = 2.9 * ARCSEC_TO_RAD

    assert math.isclose(
        ra_sigma_rad,
        expected_sigma_rad,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        dec_sigma_rad,
        expected_sigma_rad,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_wrapped_ra_difference_handles_zero_crossing():
    previous_ra_rad = math.radians(359.0)
    current_ra_rad = math.radians(1.0)

    difference_rad = wrapped_ra_difference_rad(
        current_ra_rad=current_ra_rad,
        previous_ra_rad=previous_ra_rad,
    )

    assert math.isclose(
        difference_rad,
        math.radians(2.0),
        abs_tol=1e-12,
    )


def test_angular_rate_estimation_handles_ra_wrap():
    previous_observation = make_detected_observation(
        t_seconds=0.0,
        ra_rad=math.radians(359.0),
        dec_rad=math.radians(10.0),
    )
    current_observation = make_detected_observation(
        t_seconds=2.0,
        ra_rad=math.radians(1.0),
        dec_rad=math.radians(12.0),
    )

    ra_rate_rad_s, dec_rate_rad_s = estimate_angular_rates(
        previous_observation=previous_observation,
        current_observation=current_observation,
    )

    assert ra_rate_rad_s is not None
    assert dec_rate_rad_s is not None

    assert math.isclose(
        ra_rate_rad_s,
        math.radians(1.0),
        abs_tol=1e-12,
    )
    assert math.isclose(
        dec_rate_rad_s,
        math.radians(1.0),
        abs_tol=1e-12,
    )


def test_generate_detected_observation_has_iod_fields():
    host = make_host()
    debris = make_debris(
        host.r_eci_km
        + np.array(
            [0.0, 20.0, 0.0],
            dtype=np.float64,
        ),
        size_bin="5cm",
    )

    observation = generate_angular_observation(
        host=host,
        debris=debris,
        t_seconds=0.0,
        sensor_cfg=SensorConfig(
            pointing_mode="velocity_aligned"
        ),
        rng=random.Random(42),
        sun_dir_eci=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
    )

    assert observation.detected
    assert observation.reason == "detected"

    assert observation.ra_rad is not None
    assert observation.dec_rad is not None
    assert observation.ra_sigma_rad is not None
    assert observation.dec_sigma_rad is not None

    assert observation.ra_rate_rad_s is None
    assert observation.dec_rate_rad_s is None

    assert observation.range_km == 20.0
    assert observation.off_boresight_deg is not None
    assert observation.sunlit
    assert not observation.earth_limb_blocked

    np.testing.assert_array_equal(
        observation.observer_eci_m,
        host.r_eci_km * 1000.0,
    )
    np.testing.assert_array_equal(
        observation.observer_eci_m_s,
        host.v_eci_km_s * 1000.0,
    )


def test_generate_consecutive_observation_adds_rates():
    host = make_host()
    sensor_config = SensorConfig(
        pointing_mode="velocity_aligned"
    )
    random_generator = random.Random(42)
    sun_direction = np.array(
        [0.0, 1.0, 0.0],
        dtype=np.float64,
    )

    first_debris = make_debris(
        host.r_eci_km
        + np.array(
            [0.0, 20.0, 0.0],
            dtype=np.float64,
        )
    )
    second_debris = make_debris(
        host.r_eci_km
        + np.array(
            [0.1, 20.0, 0.1],
            dtype=np.float64,
        )
    )

    first_observation = generate_angular_observation(
        host=host,
        debris=first_debris,
        t_seconds=0.0,
        sensor_cfg=sensor_config,
        rng=random_generator,
        sun_dir_eci=sun_direction,
    )

    second_observation = generate_angular_observation(
        host=host,
        debris=second_debris,
        t_seconds=1.0,
        sensor_cfg=sensor_config,
        rng=random_generator,
        previous_observation=first_observation,
        sun_dir_eci=sun_direction,
    )

    assert first_observation.detected
    assert second_observation.detected
    assert second_observation.ra_rate_rad_s is not None
    assert second_observation.dec_rate_rad_s is not None