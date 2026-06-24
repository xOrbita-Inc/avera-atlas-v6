from __future__ import annotations

import numpy as np
import pytest

from services.sandbox.models import ObjectState, SimulationResult
from services.sandbox.scenario import create_forced_detection_scenario
from services.sandbox.schema_emission import write_observations_multi_npz
from services.sandbox.sensor_model import SensorConfig
from services.sandbox.observation_bundle import generate_observation_bundle
from services.sandbox.snapshot import make_snapshot
from services.sandbox.observation_bundle import _state_to_sim_object
from services.sandbox.sensor_model import detect_object
from services.sandbox.tasking import (
    TaskingCommand,
    TaskingTarget,
    TaskingConstraints,
    execute_tasking_command,
    predicted_point_target,
    tasking_result_to_bundle,
)


def make_forced_detection_sim_result(
    *,
    t_seconds: float = 0.0,
) -> SimulationResult:
    objects = create_forced_detection_scenario()

    states = {
        object_id: ObjectState(
            object_id=object_id,
            t_seconds=t_seconds,
            r_eci_km=sim_object.r_eci_km.copy(),
            v_eci_km_s=sim_object.v_eci_km_s.copy(),
        )
        for object_id, sim_object in objects.items()
    }

    return SimulationResult(
        objects=objects,
        snapshots=[
            make_snapshot(
                t_seconds=t_seconds,
                states=states,
            )
        ],
    )


def make_two_snapshot_forced_detection_sim_result() -> SimulationResult:
    objects = create_forced_detection_scenario()
    snapshots = []

    for t_seconds in [0.0, 30.0]:
        states = {
            object_id: ObjectState(
                object_id=object_id,
                t_seconds=t_seconds,
                r_eci_km=sim_object.r_eci_km
                + sim_object.v_eci_km_s * t_seconds,
                v_eci_km_s=sim_object.v_eci_km_s.copy(),
            )
            for object_id, sim_object in objects.items()
        }

        snapshots.append(
            make_snapshot(
                t_seconds=t_seconds,
                states=states,
            )
        )

    return SimulationResult(
        objects=objects,
        snapshots=snapshots,
    )


def test_predicted_point_task_repoints_and_detects_target() -> None:
    sim_result = make_forced_detection_sim_result()
    target_position = sim_result.objects["debris_001"].r_eci_km

    snapshot = sim_result.snapshots[0]

    host_state = snapshot.states["host_001"]
    debris_state = snapshot.states["debris_001"]

    host_object = _state_to_sim_object(
        base_object=sim_result.objects["host_001"],
        r_eci_km=host_state.r_eci_km,
        v_eci_km_s=host_state.v_eci_km_s,
    )
    debris_object = _state_to_sim_object(
        base_object=sim_result.objects["debris_001"],
        r_eci_km=debris_state.r_eci_km,
        v_eci_km_s=debris_state.v_eci_km_s,
    )

    survey_detection = detect_object(
        host=host_object,
        debris=debris_object,
        sensor_cfg=SensorConfig(pointing_mode="nadir_minus_30"),
    )

    assert not survey_detection.detected
    assert survey_detection.reason == "outside_fov"

    command = TaskingCommand(
        task_id="task_001",
        sensor_id="host_001",
        start_time=0.0,
        end_time=0.0,
        target=predicted_point_target(target_position),
        priority=10,
    )

    result = execute_tasking_command(
        sim_result=sim_result,
        command=command,
        sensor_cfg=SensorConfig(pointing_mode="nadir_minus_30"),
        seed=123,
    )

    assert result.status == "accepted"
    assert result.reason == "executed"
    assert len(result.observations) == 1
    assert len(result.detected_observations) == 1

    observation = result.detected_observations[0]

    assert observation.detected
    assert observation.reason == "tasked_detected"
    assert observation.host_id == "host_001"
    assert observation.debris_id == "debris_001"
    assert observation.ra_rad is not None
    assert observation.dec_rad is not None
    assert observation.ra_sigma_rad is not None
    assert observation.dec_sigma_rad is not None
    assert observation.range_km is not None
    assert observation.off_boresight_deg is not None
    assert observation.off_boresight_deg <= 1.85


def test_tasked_observations_emit_through_schema_path(tmp_path) -> None:
    sim_result = make_forced_detection_sim_result()
    target_position = sim_result.objects["debris_001"].r_eci_km

    command = TaskingCommand(
        task_id="task_001",
        sensor_id="host_001",
        start_time=0.0,
        end_time=0.0,
        target=predicted_point_target(target_position),
        priority=10,
    )

    result = execute_tasking_command(
        sim_result=sim_result,
        command=command,
        sensor_cfg=SensorConfig(pointing_mode="nadir_minus_30"),
        seed=123,
    )

    bundle = tasking_result_to_bundle(result)

    artifact = write_observations_multi_npz(
        bundle,
        tmp_path / "observations_multi.npz",
    )

    assert artifact.observation_count == 1
    assert artifact.observer_count == 1
    assert artifact.target_count == 1


def test_tasking_rejects_unknown_sensor() -> None:
    sim_result = make_forced_detection_sim_result()
    target_position = sim_result.objects["debris_001"].r_eci_km

    command = TaskingCommand(
        task_id="task_001",
        sensor_id="missing_sensor",
        start_time=0.0,
        end_time=0.0,
        target=predicted_point_target(target_position),
        priority=10,
    )

    result = execute_tasking_command(
        sim_result=sim_result,
        command=command,
        sensor_cfg=SensorConfig(),
    )

    assert result.status == "rejected"
    assert result.reason == "unknown_sensor"
    assert result.observations == []


def test_tasking_rejects_non_host_sensor() -> None:
    sim_result = make_forced_detection_sim_result()
    target_position = sim_result.objects["debris_001"].r_eci_km

    command = TaskingCommand(
        task_id="task_001",
        sensor_id="debris_001",
        start_time=0.0,
        end_time=0.0,
        target=predicted_point_target(target_position),
        priority=10,
    )

    result = execute_tasking_command(
        sim_result=sim_result,
        command=command,
        sensor_cfg=SensorConfig(),
    )

    assert result.status == "rejected"
    assert result.reason == "sensor_is_not_host"
    assert result.observations == []


def test_tasking_rejects_no_snapshot_in_window() -> None:
    sim_result = make_forced_detection_sim_result()
    target_position = sim_result.objects["debris_001"].r_eci_km

    command = TaskingCommand(
        task_id="task_001",
        sensor_id="host_001",
        start_time=100.0,
        end_time=200.0,
        target=predicted_point_target(target_position),
        priority=10,
    )

    result = execute_tasking_command(
        sim_result=sim_result,
        command=command,
        sensor_cfg=SensorConfig(),
    )

    assert result.status == "rejected"
    assert result.reason == "no_snapshots_in_task_window"
    assert result.observations == []


def test_tasking_rejects_admissible_region_mvp() -> None:
    sim_result = make_forced_detection_sim_result()

    command = TaskingCommand(
        task_id="task_001",
        sensor_id="host_001",
        start_time=0.0,
        end_time=0.0,
        target=TaskingTarget(
            type="admissible_region",
            params={
                "ra_min_rad": 0.0,
                "ra_max_rad": 1.0,
                "dec_min_rad": 0.0,
                "dec_max_rad": 1.0,
            },
        ),
        priority=10,
    )

    result = execute_tasking_command(
        sim_result=sim_result,
        command=command,
        sensor_cfg=SensorConfig(),
    )

    assert result.status == "rejected"
    assert result.reason == "admissible_region_not_implemented"
    assert result.observations == []


def test_tasking_rejects_invalid_time_window() -> None:
    sim_result = make_forced_detection_sim_result()
    target_position = sim_result.objects["debris_001"].r_eci_km

    command = TaskingCommand(
        task_id="task_001",
        sensor_id="host_001",
        start_time=30.0,
        end_time=0.0,
        target=predicted_point_target(target_position),
        priority=10,
    )

    with pytest.raises(ValueError, match="end_time"):
        execute_tasking_command(
            sim_result=sim_result,
            command=command,
            sensor_cfg=SensorConfig(),
        )


def test_tasking_replay_is_deterministic_for_same_seed() -> None:
    sim_result = make_two_snapshot_forced_detection_sim_result()
    target_position = sim_result.objects["debris_001"].r_eci_km

    command = TaskingCommand(
        task_id="task_001",
        sensor_id="host_001",
        start_time=0.0,
        end_time=30.0,
        target=predicted_point_target(target_position),
        priority=10,
    )

    result_a = execute_tasking_command(
        sim_result=sim_result,
        command=command,
        sensor_cfg=SensorConfig(pointing_mode="nadir_minus_30"),
        seed=123,
    )
    result_b = execute_tasking_command(
        sim_result=sim_result,
        command=command,
        sensor_cfg=SensorConfig(pointing_mode="nadir_minus_30"),
        seed=123,
    )

    assert result_a.status == result_b.status
    assert len(result_a.observations) == len(result_b.observations)

    for observation_a, observation_b in zip(
        result_a.observations,
        result_b.observations,
    ):
        assert observation_a.detected == observation_b.detected
        assert observation_a.reason == observation_b.reason
        assert observation_a.host_id == observation_b.host_id
        assert observation_a.debris_id == observation_b.debris_id
        assert observation_a.t_seconds == observation_b.t_seconds
        assert observation_a.ra_rad == observation_b.ra_rad
        assert observation_a.dec_rad == observation_b.dec_rad
        assert observation_a.ra_rate_rad_s == observation_b.ra_rate_rad_s
        assert observation_a.dec_rate_rad_s == observation_b.dec_rate_rad_s
        np.testing.assert_array_equal(
            observation_a.observer_eci_m,
            observation_b.observer_eci_m,
        )
        np.testing.assert_array_equal(
            observation_a.observer_eci_m_s,
            observation_b.observer_eci_m_s,
        )

def test_tasking_detects_target_that_survey_mode_misses() -> None:
    sim_result = make_forced_detection_sim_result()
    target_position = sim_result.objects["debris_001"].r_eci_km

    survey_bundle = generate_observation_bundle(
        sim_result=sim_result,
        sensor_cfg=SensorConfig(pointing_mode="nadir_minus_30"),
        seed=123,
    )

    assert len(survey_bundle.detected_observations) == 0
    assert survey_bundle.observations[0].reason == "outside_fov"

    command = TaskingCommand(
        task_id="task_001",
        sensor_id="host_001",
        start_time=0.0,
        end_time=0.0,
        target=predicted_point_target(target_position),
        priority=10,
    )

    tasked_result = execute_tasking_command(
        sim_result=sim_result,
        command=command,
        sensor_cfg=SensorConfig(pointing_mode="nadir_minus_30"),
        seed=123,
    )

    assert tasked_result.status == "accepted"
    assert len(tasked_result.detected_observations) == 1
    assert tasked_result.detected_observations[0].reason == "tasked_detected"

def test_tasking_rejects_blackout_window_constraint() -> None:
    sim_result = make_forced_detection_sim_result()
    target_position = sim_result.objects["debris_001"].r_eci_km

    command = TaskingCommand(
        task_id="task_001",
        sensor_id="host_001",
        start_time=10.0,
        end_time=20.0,
        target=predicted_point_target(target_position),
        priority=10,
    )

    result = execute_tasking_command(
        sim_result=sim_result,
        command=command,
        sensor_cfg=SensorConfig(),
        constraints=TaskingConstraints(
            blackout_windows=((15.0, 30.0),),
        ),
    )

    assert result.status == "rejected"
    assert result.reason == "blackout_window"
    assert result.observations == []