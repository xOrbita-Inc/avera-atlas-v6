from pathlib import Path
import sys

import numpy as np

from services.sandbox.models import ObjectState, SimulationResult
from services.sandbox.scenario import create_forced_detection_scenario
from services.sandbox.sensor_model import SensorConfig
from services.sandbox.snapshot import make_snapshot
from services.sandbox.observation_bundle import generate_observation_bundle
from services.sandbox.schema_emission import write_observations_multi_npz

TRACKER_DIR = Path(__file__).resolve().parents[2] / "tracker"
if str(TRACKER_DIR) not in sys.path:
    sys.path.insert(0, str(TRACKER_DIR))

from observation_loader import load_observations_multi  # noqa: E402


def make_forced_detection_sim_result() -> SimulationResult:
    objects = create_forced_detection_scenario()

    states = {
        object_id: ObjectState(
            object_id=object_id,
            t_seconds=0.0,
            r_eci_km=sim_object.r_eci_km.copy(),
            v_eci_km_s=sim_object.v_eci_km_s.copy(),
        )
        for object_id, sim_object in objects.items()
    }

    return SimulationResult(
        objects=objects,
        snapshots=[make_snapshot(t_seconds=0.0, states=states)],
    )


def emit_loaded_observation_stream(tmp_path, seed: int):
    bundle = generate_observation_bundle(
        sim_result=make_forced_detection_sim_result(),
        sensor_cfg=SensorConfig(pointing_mode="velocity_aligned"),
        seed=seed,
    )

    output_path = tmp_path / f"observations_multi_{seed}.npz"
    write_observations_multi_npz(bundle, output_path)

    return load_observations_multi(output_path)


def test_same_seed_replays_identical_observation_stream(tmp_path):
    first = emit_loaded_observation_stream(tmp_path, seed=123)
    second = emit_loaded_observation_stream(tmp_path, seed=123)

    assert np.array_equal(first.times_utc, second.times_utc)
    assert np.array_equal(first.observer_ids, second.observer_ids)
    assert np.array_equal(first.target_ids, second.target_ids)
    assert first.meta == second.meta
    assert len(first.observations) == len(second.observations)

    for obs_a, obs_b in zip(first.observations, second.observations):
        assert obs_a.observer_id == obs_b.observer_id
        assert obs_a.target_id == obs_b.target_id
        assert obs_a.timestamp_utc == obs_b.timestamp_utc
        assert obs_a.obs_type == obs_b.obs_type
        assert np.array_equal(obs_a.observer_eci_m, obs_b.observer_eci_m)
        assert np.array_equal(obs_a.observer_eci_m_s, obs_b.observer_eci_m_s)
        assert obs_a.obs_ra_rad == obs_b.obs_ra_rad
        assert obs_a.obs_dec_rad == obs_b.obs_dec_rad
        assert obs_a.obs_sigma_ra_arcsec == obs_b.obs_sigma_ra_arcsec
        assert obs_a.obs_sigma_dec_arcsec == obs_b.obs_sigma_dec_arcsec
        assert obs_a.obs_quality == obs_b.obs_quality
        assert obs_a.obs_range_m == obs_b.obs_range_m
        assert obs_a.obs_range_rate_m_s == obs_b.obs_range_rate_m_s


def test_different_seed_changes_noisy_angles_but_not_schema_identity(tmp_path):
    first = emit_loaded_observation_stream(tmp_path, seed=123)
    second = emit_loaded_observation_stream(tmp_path, seed=456)

    assert len(first.observations) == len(second.observations)

    first_obs = first.observations[0]
    second_obs = second.observations[0]

    assert first_obs.observer_id == second_obs.observer_id
    assert first_obs.target_id == second_obs.target_id
    assert first_obs.timestamp_utc == second_obs.timestamp_utc

    angle_changed = (
        first_obs.obs_ra_rad != second_obs.obs_ra_rad
        or first_obs.obs_dec_rad != second_obs.obs_dec_rad
    )
    assert angle_changed