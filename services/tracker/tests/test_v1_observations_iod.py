"""
Tests for SCRUM-373: /v1/observations and /v1/iod endpoints, and the
/detections deprecation.

Run via:
    python3 -m pytest services/tracker/tests/test_v1_observations_iod.py -v
or as part of the full tracker suite:
    python3 -m pytest services/tracker/tests -v
"""

import pytest
from fastapi.testclient import TestClient

import main as main_module
from main import app, state


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    TestClient used as a context manager triggers the app's startup
    lifespan, which calls os.makedirs(DATA_DIR, exist_ok=True). The
    production default (/data/planner_artifacts) is a container mount
    path (see docker-compose.yaml) and is not writable on a local dev
    machine, so point it at a pytest tmp_path for the duration of each
    test. Without the `with` block on TestClient, lifespan never runs at
    all and /detections silently drops everything -- that was the
    original bug John flagged, so this must stay a real context manager.
    """
    monkeypatch.setattr(main_module, "DATA_DIR", str(tmp_path))
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_state():
    """Clear in-memory tracker state before every test."""
    state.observations.clear()
    state.tracks.clear()
    state.correlation_engine.ucts.clear()
    yield


def make_observation(i, ra=1.0, dec=0.5, detected=True, include_measurement=True):
    """Build a valid-shaped ObservationRecord dict for tests."""
    obs = {
        "observation_id": f"obs-test-{i:03d}",
        "sensor_id": "test-sensor-01",
        "target_id": "debris-test-001",
        "timestamp_utc": f"2026-01-01T00:00:{i:02d}Z",
        "detected": detected,
    }
    if include_measurement:
        obs.update({
            "ra_rad": ra + i * 0.001,
            "dec_rad": dec + i * 0.0005,
            "ra_sigma_rad": 0.0001,
            "dec_sigma_rad": 0.0001,
            "observer_eci_m": [6778000.0, 0.0, 0.0],
            "observer_eci_m_s": [0.0, 7668.0, 0.0],
        })
    return obs


def make_detection():
    """Build a valid-shaped legacy DetectionInput dict for /detections tests."""
    return {
        "detection_id": "det-001",
        "sensor_id": "AVERA-SAT-01-SWIR",
        "timestamp": "2026-01-01T00:00:00Z",
        "pixel_u": 512.5,
        "pixel_v": 384.2,
        "bbox_x": 500.0,
        "bbox_y": 370.0,
        "bbox_w": 25.0,
        "bbox_h": 28.0,
        "confidence": 0.87,
        "object_class": "Debris",
    }


class TestObservationsIngestFullPipeline:
    """Full pipeline processing per John/Minh's decision: contract
    observations feed the same correlation/track-building path as
    /detections, not just storage."""

    def test_valid_inline_record_accepted(self, client):
        resp = client.post("/v1/observations", json={"observations": [make_observation(1)]})
        assert resp.status_code in (200, 202)
        data = resp.json()
        assert data["accepted"] is True
        assert data["observation_count"] == 1
        assert data["observation_ids"] == ["obs-test-001"]

    def test_observation_persisted(self, client):
        client.post("/v1/observations", json={"observations": [make_observation(1)]})
        assert "obs-test-001" in state.observations

    def test_valid_record_feeds_correlation(self, client):
        client.post("/v1/observations", json={"observations": [make_observation(1)]})
        assert len(state.correlation_engine.ucts) >= 1

    def test_correlated_observation_tagged_v1_observations(self, client):
        client.post("/v1/observations", json={"observations": [make_observation(1)]})
        uct = list(state.correlation_engine.ucts.values())[0]
        assert uct.observations[0].ingest_path == "v1_observations"

    def test_object_class_defaults_to_unknown(self, client):
        """John's review: object_class defaults on the CorrelatedObservation
        dataclass itself, so no call site can omit it and reintroduce a
        None flowing into add_observation()'s majority vote."""
        client.post("/v1/observations", json={"observations": [make_observation(1)]})
        uct = list(state.correlation_engine.ucts.values())[0]
        assert uct.observations[0].object_class == "Unknown"

    def test_detection_id_and_confidence_remain_unset(self, client):
        """No honest default exists for these (SCRUM-395 lesson): unlike
        object_class, they must not be invented."""
        client.post("/v1/observations", json={"observations": [make_observation(1)]})
        uct = list(state.correlation_engine.ucts.values())[0]
        assert uct.observations[0].detection_id is None
        assert uct.observations[0].confidence is None


class TestObservationsIngestValidation:
    """Contract enforcement: additionalProperties: false and the anyOf
    requirement on observations/payload_ref."""

    def test_additional_properties_rejected(self, client):
        bad_obs = make_observation(1)
        bad_obs["not_a_real_field"] = "should be rejected"
        resp = client.post("/v1/observations", json={"observations": [bad_obs]})
        assert resp.status_code == 422

    def test_empty_request_rejected(self, client):
        resp = client.post("/v1/observations", json={})
        assert resp.status_code == 422


class TestObservationsIngestMissingFields:
    """Missing ra/dec/sigma/observer fields must never be defaulted: a
    defaulted sigma of 0.0 would silently tell the correlator/IOD the
    measurement is perfectly certain (SCRUM-395 failure mode)."""

    def test_detected_true_missing_measurement_stored_not_correlated(self, client):
        incomplete_obs = make_observation(1, include_measurement=False)
        incomplete_obs["detected"] = True
        resp = client.post("/v1/observations", json={"observations": [incomplete_obs]})
        assert resp.status_code in (200, 202)
        assert "obs-test-001" in state.observations
        assert len(state.correlation_engine.ucts) == 0

    def test_detected_false_skips_correlation(self, client):
        undetected_obs = make_observation(1, detected=False)
        resp = client.post("/v1/observations", json={"observations": [undetected_obs]})
        assert resp.status_code in (200, 202)
        assert "obs-test-001" in state.observations
        assert len(state.correlation_engine.ucts) == 0


class TestIodTrigger:
    def test_trigger_via_observation_ids_succeeds(self, client):
        obs_list = [make_observation(i, ra=1.0, dec=0.5) for i in range(5)]
        client.post("/v1/observations", json={"observations": obs_list})
        resp = client.post(
            "/v1/iod",
            json={"observation_ids": [o["observation_id"] for o in obs_list]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "iod_job_id" in data
        # solve() runs synchronously within the request; claiming "queued"
        # would misrepresent an already-completed job.
        assert data["status"] != "queued"

    def test_unknown_observation_id_rejected(self, client):
        resp = client.post("/v1/iod", json={"observation_ids": ["does-not-exist"]})
        assert resp.status_code == 400

    def test_solver_hint_accepted_but_not_honored(self, client, caplog):
        """IODSolver.solve() has no method-override parameter. A solver
        hint must not silently do nothing -- confirm the warning is
        actually logged, not just that the request succeeds."""
        obs_list = [make_observation(i) for i in range(5)]
        client.post("/v1/observations", json={"observations": obs_list})
        with caplog.at_level("WARNING"):
            resp = client.post(
                "/v1/iod",
                json={
                    "observation_ids": [o["observation_id"] for o in obs_list],
                    "solver": "angles-only",
                },
            )
        assert resp.status_code == 200
        assert any("not currently honored" in rec.message for rec in caplog.records)


class TestDetectionsDeprecation:
    """/detections is explicitly retired, not folded (John's decision):
    kept fully working while marked deprecated, so the detector service
    is unaffected and deprecated-path usage is measurable."""

    def test_detections_still_functionally_works(self, client):
        resp = client.post("/detections", json={"detections": [make_detection()]})
        assert resp.status_code == 200
        assert len(state.correlation_engine.ucts) >= 1

    def test_detections_still_tags_ingest_path(self, client):
        client.post("/detections", json={"detections": [make_detection()]})
        uct = list(state.correlation_engine.ucts.values())[0]
        assert uct.observations[0].ingest_path == "detections"

    def test_detections_marked_deprecated_in_openapi_schema(self, client):
        schema = app.openapi()
        assert schema["paths"]["/detections"]["post"].get("deprecated") is True
        assert schema["paths"]["/detections/single"]["post"].get("deprecated") is True

    def test_v1_observations_not_marked_deprecated(self, client):
        schema = app.openapi()
        assert schema["paths"]["/v1/observations"]["post"].get("deprecated", False) is False
