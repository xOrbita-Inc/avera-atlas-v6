"""SCRUM-351: /v1/evaluate emits a DecisionLog id and posts the full record."""
from copy import deepcopy
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import server

_BODY = {
    "conjunction_id": "test-conj-001",
    "satellite": {"norad_id": 25544, "r_sat_km": [6778.0, 0.0, 0.0],
                  "v_sat_km_s": [0.0, 7.66, 0.0], "t_burn_utc": "2026-06-22T08:00:00Z",
                  "v_remaining_m_s": 50.0, "mass_kg": 100.0},
    "policy": {"decision_mode": "flight_rule_1e4", "operator_id": "TEST", "policy_version": "2.5.0", "lambda_v": 0.01,
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


def test_evaluate_exposes_resolved_computed_pc_and_provenance():
    body = deepcopy(_BODY)
    body["conjunction"]["v_rel_km_s"] = [0.0, -15.32, 0.0]

    with patch.object(server, "UDL_ENABLED", False), \
         patch.object(server.http_requests, "post", return_value=MagicMock(status_code=201)):
        r = TestClient(server.svc).post("/v1/evaluate", json=body)

    assert r.status_code == 200
    j = r.json()
    metrics = j["metrics"]

    assert metrics["pc_pre"] is not None
    assert metrics["pc_source"] == "computed"
    assert metrics["hbr_m"] > 0.0
    assert metrics["risk_gate"] == "pc"

    risk = j["atlas_artifact"]["risk_summary"]
    assert risk["pc_pre"] == metrics["pc_pre"]
    assert risk["pc_source"] == "computed"

    decision_log = j["decision_log"]
    assert decision_log["pc_at_transition"] == metrics["pc_pre"]
    assert decision_log["pc_source"] == "computed"


def test_scrum479_evaluate_mode_drives_required():
    # Supplied Pc is a controlled input, not a measured collision probability.
    for mode, expected in (
        (None, True),
        ("aps", True),
        ("flight_rule_1e4", False),
        ("flight_rule_1e5", True),
    ):
        body = deepcopy(_BODY)
        body["conjunction_id"] = "479-EVALUATE-" + str(mode)
        body["conjunction"].update({
            "r_rel_km": [2.0, 0.0, 0.0],
            "miss_distance_km": 2.0,
            "pc_precomputed": 5.0e-5,
            "v_rel_km_s": [0.0, -15.32, 0.0],
        })
        if mode is None:
            body["policy"].pop("decision_mode")
        else:
            body["policy"]["decision_mode"] = mode

        with patch.object(server, "UDL_ENABLED", False), patch.object(
            server.http_requests, "post",
            return_value=MagicMock(status_code=201),
        ):
            response = TestClient(server.svc).post("/v1/evaluate", json=body)

        assert response.status_code == 200, response.text
        result = response.json()
        risk = result["atlas_artifact"]["risk_summary"]
        assert risk["pc_pre"] == 5.0e-5
        assert risk["maneuver_required"] is expected

        utility = result["recommendation"]["utility"]
        if mode in (None, "aps"):
            assert utility > 0.0
            assert risk["maneuver_required"] == (utility > 0.0)
            assert result["metrics"]["risk_gate"] == "utility"
        elif mode == "flight_rule_1e4":
            assert result["recommendation"]["direction"] == "no-burn"


def test_scrum479_invalid_mode_returns_422():
    for invalid in ("unknown", "", None):
        body = deepcopy(_BODY)
        body["policy"]["decision_mode"] = invalid
        with patch.object(server, "UDL_ENABLED", False), patch.object(
            server.http_requests, "post",
            return_value=MagicMock(status_code=201),
        ):
            response = TestClient(server.svc).post("/v1/evaluate", json=body)
        assert response.status_code == 422, response.text
        assert "decision_mode" in response.text
