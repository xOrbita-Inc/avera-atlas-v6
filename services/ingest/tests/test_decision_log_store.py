"""SCRUM-351: DecisionLog persistence and retrieval by decision ID."""
import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# The ingest service module is named 'main', which collides with the other
# services' main.py when the full repo suite runs (sys.modules['main'] gets
# whichever service imported first). Load the ingest app by explicit path under
# a unique module name so this test is import-order independent (SCRUM-351).
_INGEST_ROOT = Path(__file__).resolve().parents[1]
if str(_INGEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_INGEST_ROOT))
_spec = importlib.util.spec_from_file_location("ingest_main", _INGEST_ROOT / "main.py")
ingest_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingest_main)
app = ingest_main.app

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
    with TestClient(app) as c:
        r = c.post("/decision_log", json=_DL)
        assert r.status_code == 201 and r.json()["status"] == "saved"
        g = c.get("/store/decision_log/" + _DL["log_id"])
        assert g.status_code == 200
        body = g.json()
        assert body["decision"] == "no_go"
        assert body["verification_passed"] is True


def test_unknown_id_returns_404_with_message():
    with TestClient(app) as c:
        g = c.get("/store/decision_log/no-such-id")
        assert g.status_code == 404
        assert "No decision log found" in g.json()["error"]


def test_missing_log_id_returns_400():
    with TestClient(app) as c:
        r = c.post("/decision_log", json={"conjunction_id": "x"})
        assert r.status_code == 400


def test_upsert_is_idempotent():
    with TestClient(app) as c:
        c.post("/decision_log", json=_DL)
        r = c.post("/decision_log", json=_DL)
        assert r.status_code == 201 and r.json()["status"] == "saved"
