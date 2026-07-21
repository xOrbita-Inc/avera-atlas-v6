"""SCRUM-351: DecisionLog persistence and retrieval by decision ID."""
from fastapi.testclient import TestClient
import main

_DL = {
    "log_id": "conj-1_25544_2026-07-20T00:00:00.000000Z",
    "conjunction_id": "conj-1",
    "sat_id": "25544",
    "decision": "no_go",
    "decision_log": {
        "log_id": "conj-1_25544_2026-07-20T00:00:00.000000Z",
        "decision": "no_go",
        "utility": 0.0,
        "verification_passed": True,
    },
}


def test_persist_and_retrieve_by_id():
    with TestClient(main.app) as c:
        r = c.post("/decision_log", json=_DL)
        assert r.status_code == 201 and r.json()["status"] == "saved"
        g = c.get("/store/decision_log/" + _DL["log_id"])
        assert g.status_code == 200
        body = g.json()
        assert body["decision"] == "no_go"
        assert body["verification_passed"] is True


def test_unknown_id_returns_404_with_message():
    with TestClient(main.app) as c:
        g = c.get("/store/decision_log/no-such-id")
        assert g.status_code == 404
        assert "No decision log found" in g.json()["error"]


def test_missing_log_id_returns_400():
    with TestClient(main.app) as c:
        r = c.post("/decision_log", json={"conjunction_id": "x"})
        assert r.status_code == 400


def test_upsert_is_idempotent():
    with TestClient(main.app) as c:
        c.post("/decision_log", json=_DL)
        r = c.post("/decision_log", json=_DL)
        assert r.status_code == 201 and r.json()["status"] == "saved"
