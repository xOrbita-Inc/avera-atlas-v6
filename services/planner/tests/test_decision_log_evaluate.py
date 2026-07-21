"""SCRUM-351: /v1/evaluate emits a DecisionLog id and posts the full record."""
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import server

_BODY = {
    "conjunction_id": "test-conj-001",
    "satellite": {"norad_id": 25544, "r_sat_km": [6778.0, 0.0, 0.0],
                  "v_sat_km_s": [0.0, 7.66, 0.0], "t_burn_utc": "2026-06-22T08:00:00Z",
                  "v_remaining_m_s": 50.0, "mass_kg": 100.0},
    "policy": {"operator_id": "TEST", "policy_version": "2.5.0", "lambda_v": 0.01,
               "lambda_L": 0.001, "dv_mag_limit_m_s": 0.5, "a_ref_km": 6778.0},
    "conjunction": {"primary_norad": 25544, "secondary_norad": 99001,
                    "t_ca_utc": "2026-06-22T10:00:00Z", "r_rel_km": [0.1, 0.2, 0.3],
                    "p_rel_km2": [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4]},
}


def test_evaluate_returns_decision_log_id():
    with patch.object(server, "UDL_ENABLED", False), \
         patch.object(server.http_requests, "post", return_value=MagicMock(status_code=201)):
        r = TestClient(server.svc).post("/v1/evaluate", json=_BODY)
    assert r.status_code == 200
    j = r.json()
    assert j.get("decision_log_id")
    assert "decision_log" in j
    assert j["decision_log"]["decision"] in ("maneuver_recommended", "no_go")


def test_evaluate_posts_full_decision_log_to_ingest():
    with patch.object(server, "UDL_ENABLED", False), \
         patch.object(server.http_requests, "post", return_value=MagicMock(status_code=201)) as mp:
        TestClient(server.svc).post("/v1/evaluate", json=_BODY)
    assert any("/decision_log" in str(call) for call in mp.call_args_list)
