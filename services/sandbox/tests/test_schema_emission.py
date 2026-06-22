from pathlib import Path
import sys
from uuid import uuid4

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

from observation_loader import load_observations_multi, to_iod_observations  # noqa: E402
from iod import IODSolver  # noqa: E402

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


def make_forced_detection_bundle():
    return generate_observation_bundle(
        sim_result=make_forced_detection_sim_result(),
        sensor_cfg=SensorConfig(pointing_mode="velocity_aligned"),
        seed=123,
    )


def test_emitted_observations_multi_loads_with_tracker_loader(tmp_path):
    bundle = make_forced_detection_bundle()

    output_path = tmp_path / "observations_multi.npz"
    artifact = write_observations_multi_npz(bundle, output_path)

    assert artifact.path == output_path
    assert artifact.observation_count >= 1

    loaded = load_observations_multi(output_path)

    assert len(loaded.observations) == artifact.observation_count
    assert loaded.meta["schema_name"] == "observations_multi"
    assert loaded.meta["schema_version"] == 1
    assert loaded.meta["frame"] == "ECI"
    assert loaded.meta["time_scale"] == "UTC"
    assert loaded.meta["god_view_isolation"] is True


def test_emitted_observations_convert_to_iod_without_sandbox_adapter(tmp_path):
    bundle = make_forced_detection_bundle()

    output_path = tmp_path / "observations_multi.npz"
    write_observations_multi_npz(bundle, output_path)

    loaded = load_observations_multi(output_path)
    iod_observations = to_iod_observations(loaded)

    assert len(iod_observations) == len(loaded.observations)

    first = iod_observations[0]
    assert np.isfinite(first.ra)
    assert np.isfinite(first.dec)
    assert np.all(np.isfinite(first.observer_position_km))
    assert np.all(np.isfinite(first.observer_velocity_km_s))
    assert first.range_km is None
    assert first.range_sigma_km is None

def test_emitted_observations_are_consumed_by_tracker_iod_solver(tmp_path):
    objects = create_forced_detection_scenario()

    snapshots = []
    for t_seconds in [0.0, 30.0, 60.0]:
        states = {}
        for object_id, sim_object in objects.items():
            states[object_id] = ObjectState(
                object_id=object_id,
                t_seconds=t_seconds,
                r_eci_km=sim_object.r_eci_km
                + sim_object.v_eci_km_s * t_seconds,
                v_eci_km_s=sim_object.v_eci_km_s.copy(),
            )
        snapshots.append(make_snapshot(t_seconds=t_seconds, states=states))

    sim_result = SimulationResult(objects=objects, snapshots=snapshots)

    bundle = generate_observation_bundle(
        sim_result=sim_result,
        sensor_cfg=SensorConfig(pointing_mode="velocity_aligned"),
        seed=123,
    )

    output_path = tmp_path / "observations_multi.npz"
    write_observations_multi_npz(bundle, output_path)

    loaded = load_observations_multi(output_path)
    iod_observations = to_iod_observations(loaded)

    assert len(iod_observations) >= 3

    solution = IODSolver().solve(iod_observations, uuid4())

    assert solution.observations_used >= 3
    assert solution.error_message != "Insufficient observations: 0 < 3"