import numpy as np

from services.sandbox.config import (
    DebrisConfig,
    HostConfig,
    IntegratorConfig,
    SimConfig,
)
from services.sandbox.observation_bundle import (
    generate_observation_bundle,
)
from services.sandbox.scenario import (
    create_seeded_swarm_and_hosts,
)
from services.sandbox.sensor_model import SensorConfig
from services.sandbox.simulation import run_simulation


def make_small_simulation_result():
    config = SimConfig(
        seed=42,
        debris=DebrisConfig(count=5),
        hosts=HostConfig(
            count=2,
            mode="distributed",
        ),
        integrator=IntegratorConfig(
            dt_seconds=10.0,
            duration_seconds=600.0,
            save_every_n_steps=10,
        ),
    )

    objects = create_seeded_swarm_and_hosts(config)
    simulation_result = run_simulation(
        objects,
        config,
    )

    return config, simulation_result


def test_observation_bundle_returns_one_record_per_pair_per_snapshot():
    config, simulation_result = make_small_simulation_result()

    bundle = generate_observation_bundle(
        sim_result=simulation_result,
        sensor_cfg=SensorConfig(
            pointing_mode="velocity_aligned"
        ),
        seed=42,
    )

    expected_record_count = (
        len(simulation_result.snapshots)
        * config.hosts.count
        * config.debris.count
    )

    assert len(bundle.observations) == expected_record_count


def test_observation_bundle_records_have_expected_identity_fields():
    _, simulation_result = make_small_simulation_result()

    bundle = generate_observation_bundle(
        sim_result=simulation_result,
        sensor_cfg=SensorConfig(
            pointing_mode="velocity_aligned"
        ),
        seed=42,
    )

    sample = bundle.observations[0]

    assert sample.host_id.startswith("host_")
    assert sample.debris_id.startswith("debris_")
    assert sample.reason
    assert sample.t_seconds >= 0.0
    assert sample.observer_eci_m.shape == (3,)
    assert sample.observer_eci_m_s.shape == (3,)


def test_bundle_detected_and_rejected_properties_partition_records():
    _, simulation_result = make_small_simulation_result()

    bundle = generate_observation_bundle(
        sim_result=simulation_result,
        sensor_cfg=SensorConfig(
            pointing_mode="velocity_aligned"
        ),
        seed=42,
    )

    assert (
        len(bundle.detected_observations)
        + len(bundle.rejected_observations)
        == len(bundle.observations)
    )

    assert all(
        observation.detected
        for observation in bundle.detected_observations
    )
    assert all(
        not observation.detected
        for observation in bundle.rejected_observations
    )


def test_reason_counts_cover_every_observation():
    _, simulation_result = make_small_simulation_result()

    bundle = generate_observation_bundle(
        sim_result=simulation_result,
        sensor_cfg=SensorConfig(
            pointing_mode="velocity_aligned"
        ),
        seed=42,
    )

    reason_counts = bundle.reason_counts()

    assert sum(reason_counts.values()) == len(
        bundle.observations
    )
    assert all(
        isinstance(reason, str)
        for reason in reason_counts
    )


def test_observation_bundle_is_bit_deterministic_for_same_seed():
    _, simulation_result = make_small_simulation_result()

    sensor_config = SensorConfig(
        pointing_mode="velocity_aligned"
    )

    bundle_a = generate_observation_bundle(
        sim_result=simulation_result,
        sensor_cfg=sensor_config,
        seed=99,
    )
    bundle_b = generate_observation_bundle(
        sim_result=simulation_result,
        sensor_cfg=sensor_config,
        seed=99,
    )

    assert len(bundle_a.observations) == len(
        bundle_b.observations
    )

    for observation_a, observation_b in zip(
        bundle_a.observations,
        bundle_b.observations,
    ):
        assert observation_a.host_id == observation_b.host_id
        assert observation_a.debris_id == observation_b.debris_id
        assert observation_a.t_seconds == observation_b.t_seconds
        assert observation_a.detected == observation_b.detected
        assert observation_a.reason == observation_b.reason

        assert observation_a.ra_rad == observation_b.ra_rad
        assert observation_a.dec_rad == observation_b.dec_rad
        assert (
            observation_a.ra_sigma_rad
            == observation_b.ra_sigma_rad
        )
        assert (
            observation_a.dec_sigma_rad
            == observation_b.dec_sigma_rad
        )
        assert (
            observation_a.ra_rate_rad_s
            == observation_b.ra_rate_rad_s
        )
        assert (
            observation_a.dec_rate_rad_s
            == observation_b.dec_rate_rad_s
        )

        assert observation_a.range_km == observation_b.range_km
        assert (
            observation_a.off_boresight_deg
            == observation_b.off_boresight_deg
        )
        assert observation_a.sunlit == observation_b.sunlit
        assert (
            observation_a.earth_limb_blocked
            == observation_b.earth_limb_blocked
        )

        np.testing.assert_array_equal(
            observation_a.observer_eci_m,
            observation_b.observer_eci_m,
        )
        np.testing.assert_array_equal(
            observation_a.observer_eci_m_s,
            observation_b.observer_eci_m_s,
        )


def test_observation_bundle_rejects_invalid_sun_direction():
    _, simulation_result = make_small_simulation_result()

    with np.testing.assert_raises(ValueError):
        generate_observation_bundle(
            sim_result=simulation_result,
            sensor_cfg=SensorConfig(),
            seed=42,
            sun_dir_eci=np.zeros(
                3,
                dtype=np.float64,
            ),
        )