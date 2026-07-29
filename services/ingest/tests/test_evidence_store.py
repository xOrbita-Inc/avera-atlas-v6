"""SCRUM-377: append-only evidence chain store, and the decision_log overwrite fix.

The two things this guards:
  - the store refuses to overwrite an audit record, and refuses an append that
    would create a gap
  - SCRUM-351 retrieval by decision ID keeps working unchanged
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Same import-by-path dance as test_decision_log_store.py: the ingest module is
# named 'main' and collides with other services when the full suite runs.
_INGEST_ROOT = Path(__file__).resolve().parents[1]
if str(_INGEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_INGEST_ROOT))
_spec = importlib.util.spec_from_file_location("ingest_main_evidence", _INGEST_ROOT / "main.py")
ingest_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingest_main)
app = ingest_main.app

GENESIS = "0" * 64


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _make(chain_id: str, seq: int, prev_hash: str, value: str = "C1") -> dict:
    """Build a record body the way the planner does, hash included."""
    hashable = {
        "record_id": f"{chain_id}#{seq}",
        "chain_id": chain_id,
        "seq": seq,
        "record_type": "decision",
        "recorded_at": f"2026-07-28T00:00:{seq:02d}Z",
        "fields": {
            "conjunction_id": {"state": "present", "value": value, "producer": "main"},
        },
        "prev_hash": prev_hash,
    }
    canonical = _canonical(hashable)
    content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    record = dict(hashable)
    record["content_hash"] = content_hash
    return {"record": record, "canonical_payload": canonical}


def _append(c, body) -> "object":
    return c.post("/evidence_record", json=body)


def _seed(c, chain_id: str, n: int):
    """Append n records and return them."""
    made = []
    prev = GENESIS
    for i in range(n):
        body = _make(chain_id, i, prev)
        r = _append(c, body)
        assert r.status_code == 201, r.text
        made.append(body)
        prev = body["record"]["content_hash"]
    return made


class TestAppendOnly:
    def test_append_and_read_back_the_chain(self):
        with TestClient(app) as c:
            _seed(c, "SAT-APPEND", 3)
            r = c.get("/store/evidence/SAT-APPEND")
            assert r.status_code == 200
            assert r.json()["count"] == 3
            assert [rec["seq"] for rec in r.json()["records"]] == [0, 1, 2]

    def test_identical_repost_is_idempotent(self):
        """Retries and at-least-once delivery must not fail."""
        with TestClient(app) as c:
            body = _seed(c, "SAT-IDEM", 1)[0]
            r = _append(c, body)
            assert r.status_code == 201
            assert r.json()["changed"] is False

    def test_overwrite_with_different_content_is_refused(self):
        """The core append-only requirement."""
        with TestClient(app) as c:
            _seed(c, "SAT-OVER", 1)
            tampered = _make("SAT-OVER", 0, GENESIS, value="C-DIFFERENT")
            r = _append(c, tampered)
            assert r.status_code == 409
            assert "append-only" in r.json()["error"]

    def test_out_of_order_append_is_refused(self):
        """A gap cannot be created by a well-behaved writer.

        That is what makes a gap found later actual evidence rather than noise.
        """
        with TestClient(app) as c:
            _seed(c, "SAT-GAP", 1)
            r = _append(c, _make("SAT-GAP", 5, "x" * 64))
            assert r.status_code == 409
            assert r.json()["expected_seq"] == 1

    def test_append_with_wrong_prev_hash_is_refused(self):
        with TestClient(app) as c:
            _seed(c, "SAT-LINK", 1)
            r = _append(c, _make("SAT-LINK", 1, "f" * 64))
            assert r.status_code == 409
            assert "prev_hash" in r.json()["error"]

    def test_hash_that_does_not_cover_the_payload_is_refused(self):
        with TestClient(app) as c:
            body = _make("SAT-BADHASH", 0, GENESIS)
            body["record"]["content_hash"] = "0" * 64
            r = _append(c, body)
            assert r.status_code == 400
            assert "does not match" in r.json()["error"]


class TestChainHeadAndVerify:
    def test_head_of_an_empty_chain_is_genesis(self):
        with TestClient(app) as c:
            r = c.get("/store/evidence/SAT-EMPTY/head")
            assert r.json() == {
                "chain_id": "SAT-EMPTY", "seq": -1, "next_seq": 0,
                "content_hash": GENESIS, "exists": False,
            }

    def test_head_tracks_the_last_append(self):
        with TestClient(app) as c:
            made = _seed(c, "SAT-HEAD", 3)
            r = c.get("/store/evidence/SAT-HEAD/head").json()
            assert r["seq"] == 2 and r["next_seq"] == 3
            assert r["content_hash"] == made[-1]["record"]["content_hash"]

    def test_clean_chain_verifies(self):
        with TestClient(app) as c:
            _seed(c, "SAT-VERIFY", 4)
            r = c.get("/store/evidence/SAT-VERIFY/verify").json()
            assert r["ok"] is True
            assert r["checked"] == 4

    def test_verify_detects_a_modified_record_in_the_store(self):
        """Edit the row directly, the way a tamper would."""
        with TestClient(app) as c:
            _seed(c, "SAT-TAMPER", 3)
            with ingest_main.get_session() as s:
                row = (
                    s.query(ingest_main.EvidenceRecordRow)
                    .filter_by(record_id="SAT-TAMPER#1").first()
                )
                row.canonical_payload = row.canonical_payload.replace("C1", "C9")
            r = c.get("/store/evidence/SAT-TAMPER/verify").json()
            assert r["ok"] is False
            assert r["break_seq"] == 1
            assert "modified" in r["reason"]

    def test_verify_detects_a_deleted_record(self):
        with TestClient(app) as c:
            _seed(c, "SAT-DELETE", 4)
            with ingest_main.get_session() as s:
                s.query(ingest_main.EvidenceRecordRow).filter_by(
                    record_id="SAT-DELETE#2"
                ).delete()
            r = c.get("/store/evidence/SAT-DELETE/verify").json()
            assert r["ok"] is False
            assert r["break_seq"] == 2
            assert "missing" in r["reason"]

    def test_unknown_chain_returns_404(self):
        with TestClient(app) as c:
            assert c.get("/store/evidence/NOPE").status_code == 404
            assert c.get("/store/evidence/NOPE/verify").status_code == 404


class TestDecisionLogAppendOnlyFix:
    _DL = {
        "log_id": "conj-377_25544_2026-07-28T00:00:00Z",
        "conjunction_id": "conj-377",
        "sat_id": "25544",
        "decision": "no_go",
        "decision_log": {"log_id": "conj-377_25544_2026-07-28T00:00:00Z", "utility": 0.0},
    }

    def test_identical_repost_still_succeeds(self):
        """SCRUM-351's idempotency contract is preserved."""
        with TestClient(app) as c:
            c.post("/decision_log", json=self._DL)
            r = c.post("/decision_log", json=self._DL)
            assert r.status_code == 201
            assert r.json()["changed"] is False

    def test_overwrite_with_different_content_is_refused(self):
        """This used to silently overwrite the stored decision."""
        with TestClient(app) as c:
            c.post("/decision_log", json=self._DL)
            changed = dict(self._DL)
            changed["decision_log"] = {"log_id": self._DL["log_id"], "utility": 99.0}
            r = c.post("/decision_log", json=changed)
            assert r.status_code == 409
            assert "append-only" in r.json()["error"]

    def test_original_content_survives_a_refused_overwrite(self):
        with TestClient(app) as c:
            c.post("/decision_log", json=self._DL)
            changed = dict(self._DL)
            changed["decision_log"] = {"log_id": self._DL["log_id"], "utility": 99.0}
            c.post("/decision_log", json=changed)
            stored = c.get(f"/store/decision_log/{self._DL['log_id']}").json()
            assert stored["utility"] == 0.0
