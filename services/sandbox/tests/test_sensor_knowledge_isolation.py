import pytest

from services.sandbox.models import ObjectState, SimulationResult
from services.sandbox.scenario import create_forced_detection_scenario
from services.sandbox.sensor_model import SensorConfig
from services.sandbox.snapshot import make_snapshot
from services.sandbox.observation_bundle import generate_observation_bundle
from services.sandbox.sensor_knowledge import (
    GodViewAccessError,
    SensorKnowledgeStore,
)


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


def test_sensor_knowledge_exposes_only_detected_observations():
    bundle = make_forced_detection_bundle()
    store = SensorKnowledgeStore.from_observation_bundle(bundle)

    assert store.sensor_ids == ("host_001",)

    stream = store.stream_for_sensor("host_001")

    assert len(stream) == len(bundle.detected_observations)
    assert len(stream) >= 1

    first = next(iter(stream))
    assert first.detected is True
    assert first.host_id == "host_001"


def test_sensor_knowledge_blocks_god_view_access():
    bundle = make_forced_detection_bundle()
    store = SensorKnowledgeStore.from_observation_bundle(bundle)

    with pytest.raises(GodViewAccessError):
        _ = store.objects

    with pytest.raises(GodViewAccessError):
        _ = store.snapshots

    with pytest.raises(GodViewAccessError):
        _ = store.truth_states


def test_sensor_stream_blocks_god_view_access():
    bundle = make_forced_detection_bundle()
    store = SensorKnowledgeStore.from_observation_bundle(bundle)
    stream = store.stream_for_sensor("host_001")

    with pytest.raises(GodViewAccessError):
        _ = stream.objects

    with pytest.raises(GodViewAccessError):
        _ = stream.snapshots

    with pytest.raises(GodViewAccessError):
        _ = stream.truth_states


def test_unknown_sensor_id_is_rejected():
    bundle = make_forced_detection_bundle()
    store = SensorKnowledgeStore.from_observation_bundle(bundle)

    with pytest.raises(KeyError):
        store.stream_for_sensor("missing_sensor")