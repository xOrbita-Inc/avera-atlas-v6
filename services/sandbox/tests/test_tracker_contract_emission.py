from __future__ import annotations

import json
import random

import numpy as np

from services.sandbox.observation_bundle import ObservationBundle
from services.sandbox.observations import generate_angular_observation
from services.sandbox.schema_emission import (
    bundle_to_tracker_ingest_request,
    write_tracker_observations_json,
)
from services.sandbox.sensor_model import SensorConfig
from services.sandbox.tests.test_observations import make_debris, make_host


FORBIDDEN_GOD_VIEW_FIELDS = {
    "truth_state",
    "truth_states",
    "objects",
    "snapshots",
    "r_eci_km",
    "v_eci_km_s",
    "true_position_km",
    "true_velocity_km_s",
    "future_debris_position",
}


def make_detected_bundle() -> ObservationBundle:
    host = make_host()
    debris = make_debris(
        host.r_eci_km + np.array([0.0, 20.0, 0.0], dtype=np.float64),
        size_bin="5cm",
    )

    observation = generate_angular_observation(
        host=host,
        debris=debris,
        t_seconds=0.0,
        sensor_cfg=SensorConfig(pointing_mode="velocity_aligned"),
        rng=random.Random(42),
        sun_dir_eci=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )

    assert observation.detected
    return ObservationBundle(observations=[observation])


def test_tracker_contract_emits_inline_json_observation_records() -> None:
    bundle = make_detected_bundle()

    request = bundle_to_tracker_ingest_request(bundle)

    assert sorted(request.keys()) == ["observations"]
    assert len(request["observations"]) == 1

    record = request["observations"][0]

    assert record["observation_id"].startswith("obs-host_001-debris_001-")
    assert record["sensor_id"] == "host_001"
    assert record["target_id"] == "debris_001"
    assert record["detected"] is True
    assert record["source"] == "sandbox"

    assert record["timestamp_utc"] is not None
    assert record["timestamp_utc"].endswith("Z")
    assert record["t_seconds"] == 0.0

    assert record["ra_rad"] is not None
    assert record["dec_rad"] is not None
    assert record["ra_sigma_rad"] is not None
    assert record["dec_sigma_rad"] is not None
    assert record["range_m"] is not None
    assert record["range_m"] == 20000.0
    assert record["range_sigma_m"] == 50.0
    assert "range_km" not in record
    assert "range_sigma_km" not in record
    assert "range_rate_km_s" not in record

    assert isinstance(record["observer_eci_m"], list)
    assert len(record["observer_eci_m"]) == 3
    assert isinstance(record["observer_eci_m_s"], list)
    assert len(record["observer_eci_m_s"]) == 3


def test_tracker_contract_excludes_god_view_fields() -> None:
    bundle = make_detected_bundle()

    request = bundle_to_tracker_ingest_request(bundle)
    encoded = json.dumps(request)

    for forbidden_field in FORBIDDEN_GOD_VIEW_FIELDS:
        assert forbidden_field not in encoded


def test_tracker_contract_writer_outputs_json_body(tmp_path) -> None:
    bundle = make_detected_bundle()
    output_path = tmp_path / "tracker_observations.json"

    artifact = write_tracker_observations_json(bundle, output_path)

    assert artifact.path == output_path
    assert artifact.observation_count == 1
    assert artifact.schema_name == "tracker_observation_ingest"
    assert artifact.schema_version == "0.1.0"

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert "observations" in payload
    assert "payload_ref" not in payload
    assert payload["observations"][0]["sensor_id"] == "host_001"
    assert payload["observations"][0]["timestamp_utc"].endswith("Z")


def test_tracker_contract_rejects_empty_detection_stream() -> None:
    bundle = ObservationBundle(observations=[])

    try:
        bundle_to_tracker_ingest_request(bundle)
    except ValueError as exc:
        assert "zero detections" in str(exc)
    else:
        raise AssertionError("Expected zero-detection tracker contract emission to fail")