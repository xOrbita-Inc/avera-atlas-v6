"""
Manual smoke test for SCRUM-373 Step 1: /v1/observations and /v1/iod.

Run from services/tracker/ after copying in the updated main.py,
schemas.py, and correlate.py:

    python3 test_scrum_373_endpoints.py

Requires fastapi, pydantic, numpy already installed (same requirements.txt
as the tracker service). Uses FastAPI's TestClient, no server needs to be
running separately.
"""

from fastapi.testclient import TestClient
from main import app, state

client = TestClient(app)

# TestClient(app) without a `with` block does not trigger the app's startup
# lifespan, so register_default_sensors() (which registers the AVERA-SAT-01
# platform used by /detections) never runs. Call it directly here -- it's
# documented as idempotent, so this is safe even if lifespan did run.
state.register_default_sensors()


def reset_state():
    """Clear in-memory state between test cases."""
    state.observations.clear()
    state.tracks.clear()
    state.correlation_engine.ucts.clear()


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


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


print("=" * 60)
print("TEST 1: /v1/observations - valid inline record, full pipeline")
print("=" * 60)
reset_state()
resp = client.post("/v1/observations", json={"observations": [make_observation(1)]})
check("status 202/200", resp.status_code in (200, 202))
data = resp.json()
check("accepted == True", data.get("accepted") is True)
check("observation_count == 1", data.get("observation_count") == 1)
check("observation_ids echoes input", data.get("observation_ids") == ["obs-test-001"])
check("observation persisted in state.observations", "obs-test-001" in state.observations)
check("observation fed into correlation (a UCT exists)", len(state.correlation_engine.ucts) >= 1)
if state.correlation_engine.ucts:
    uct = list(state.correlation_engine.ucts.values())[0]
    check("correlated observation tagged ingest_path=v1_observations",
          uct.observations[0].ingest_path == "v1_observations")
    check("object_class defaults to 'Unknown' for contract observations (John's call)",
          uct.observations[0].object_class == "Unknown")
    check("detection_id and confidence remain unset (no honest default exists)",
          uct.observations[0].detection_id is None and uct.observations[0].confidence is None)


print("\n" + "=" * 60)
print("TEST 2: /v1/observations - additionalProperties: false enforced")
print("=" * 60)
reset_state()
bad_obs = make_observation(1)
bad_obs["not_a_real_field"] = "should be rejected"
resp = client.post("/v1/observations", json={"observations": [bad_obs]})
check("extra field rejected (422)", resp.status_code == 422)


print("\n" + "=" * 60)
print("TEST 3: /v1/observations - anyOf validation (neither field given)")
print("=" * 60)
resp = client.post("/v1/observations", json={})
check("empty request rejected (422)", resp.status_code == 422)


print("\n" + "=" * 60)
print("TEST 4: /v1/observations - detected=True but missing ra/dec")
print("=" * 60)
reset_state()
incomplete_obs = make_observation(1, include_measurement=False)
incomplete_obs["detected"] = True
resp = client.post("/v1/observations", json={"observations": [incomplete_obs]})
check("still 202/200 (stored, not rejected)", resp.status_code in (200, 202))
check("observation persisted despite missing fields", "obs-test-001" in state.observations)
check("NOT fed into correlation (no UCT created)", len(state.correlation_engine.ucts) == 0)


print("\n" + "=" * 60)
print("TEST 5: /v1/observations - detected=False skips correlation")
print("=" * 60)
reset_state()
undetected_obs = make_observation(1, detected=False)
resp = client.post("/v1/observations", json={"observations": [undetected_obs]})
check("stored", "obs-test-001" in state.observations)
check("not correlated", len(state.correlation_engine.ucts) == 0)


print("\n" + "=" * 60)
print("TEST 6: /v1/iod - trigger via observation_ids from prior ingest")
print("=" * 60)
reset_state()
# Ingest 3 observations with a bit of angular spread so IOD has a chance to succeed.
obs_list = [make_observation(i, ra=1.0, dec=0.5) for i in range(5)]
client.post("/v1/observations", json={"observations": obs_list})
resp = client.post("/v1/iod", json={"observation_ids": [o["observation_id"] for o in obs_list]})
check("request accepted (200)", resp.status_code == 200)
iod_data = resp.json()
print(f"    iod response: {iod_data}")
check("iod_job_id present", "iod_job_id" in iod_data)
check("status reflects real outcome (not 'queued')", iod_data.get("status") != "queued")


print("\n" + "=" * 60)
print("TEST 7: /v1/iod - unknown observation_id rejected with 400")
print("=" * 60)
resp = client.post("/v1/iod", json={"observation_ids": ["does-not-exist"]})
check("unknown id rejected (400)", resp.status_code == 400)


print("\n" + "=" * 60)
print("TEST 8: /v1/iod - solver hint accepted but logged as not honored")
print("=" * 60)
reset_state()
obs_list = [make_observation(i) for i in range(5)]
client.post("/v1/observations", json={"observations": obs_list})
resp = client.post("/v1/iod", json={
    "observation_ids": [o["observation_id"] for o in obs_list],
    "solver": "angles-only",
})
check("request still accepted despite solver hint (200)", resp.status_code == 200)
print("    (check server logs for the 'solver hint not honored' warning)")


print("\n" + "=" * 60)
print("TEST 9: existing /detections still works, tags ingest_path=detections")
print("=" * 60)
reset_state()
resp = client.post("/detections", json={
    "detections": [{
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
    }]
})
check("detections endpoint still works", resp.status_code == 200)
check("a UCT was actually created (platform state resolved)", len(state.correlation_engine.ucts) >= 1)
if state.correlation_engine.ucts:
    uct = list(state.correlation_engine.ucts.values())[0]
    check("tagged ingest_path=detections", uct.observations[0].ingest_path == "detections")

print("\nDone. Review any FAIL lines above before proceeding.")


print("\n" + "=" * 60)
print("TEST 10: /detections and /detections/single marked deprecated in OpenAPI schema")
print("=" * 60)
openapi_schema = app.openapi()
detections_deprecated = openapi_schema["paths"]["/detections"]["post"].get("deprecated", False)
detections_single_deprecated = openapi_schema["paths"]["/detections/single"]["post"].get("deprecated", False)
check("/detections marked deprecated in OpenAPI schema", detections_deprecated is True)
check("/detections/single marked deprecated in OpenAPI schema", detections_single_deprecated is True)
v1_obs_not_deprecated = openapi_schema["paths"]["/v1/observations"]["post"].get("deprecated", False)
check("/v1/observations is NOT marked deprecated", v1_obs_not_deprecated is False)
