"""SCRUM-453 item 1: the on-demand screening result persists and reads back.

The planner runs the secondary screen as part of a decision and POSTs the result
here. Before this endpoint existed the POST 404'd, the planner logged an audit
failure, and the decision came back with screening_record_id null -- correct
fail-safe behaviour, but the record of what a CLEAR or NOT CLEAR was judged on
existed only in the service logs.

So these tests are about what ends up in the store and what comes back out of it:
that the record is tied to the decision log id, that the conjunctions and their
covariance survive the round trip intact, that a retry does not accumulate rows,
and that an unknown id 404s rather than inventing an answer.

The record shape here is copied from what the planner's _persist_screening_result
actually builds -- that code is unchanged by this ticket, so the endpoint has to
accept it as-is.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Same explicit-path import as test_decision_log_store.py: the ingest module is
# named 'main' and collides with the other services' main.py in a full-repo run.
_INGEST_ROOT = Path(__file__).resolve().parents[1]
if str(_INGEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_INGEST_ROOT))
_spec = importlib.util.spec_from_file_location("ingest_main", _INGEST_ROOT / "main.py")
ingest_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingest_main)
app = ingest_main.app

from db import ScreeningRecord, get_session, init_db  # noqa: E402

_DECISION_LOG_ID = "conj-7_36508_2026-09-24T12:00:00.000000Z"


def a_record(**overrides) -> dict:
    """The record the planner posts, as _persist_screening_result assembles it."""
    base = {
        "decision_log_id": _DECISION_LOG_ID,
        "screening_id": "9f2c41aa7e2b4f0f8d2a",
        "source": "leolabs_on_demand",
        "cdm_count": 12,
        "conjunction_count": 2,
        # A LIST of unparseable CDMs with reasons, not a count or a flag.
        "skipped": [],
        "clear": True,
        "breaches": [],
        "conjunctions": [
            {
                "cdm_id": "76592783517",
                "event_id": "3538687541",
                "secondary_norad": 270302,
                "secondary_designator": "270302",
                "secondary_name": "STARLINK-37550",
                "tca_utc": "2026-09-25T04:12:09.482000Z",
                "miss_distance_m": 20115.063,
                "pc": 4.3858e-13,
                "cov_eci_pos_m2": [
                    [1427261.0, -11074.1, 52.0],
                    [-11074.1, 133839500.0, 952.6],
                    [52.0, 952.6, 25724680.0],
                ],
            },
            {
                "cdm_id": "76592783999",
                "event_id": "3538687999",
                "secondary_norad": 44713,
                "secondary_designator": "44713",
                "secondary_name": "STARLINK-1007",
                "tca_utc": "2026-09-25T06:40:00.000000Z",
                "miss_distance_m": 812.5,
                "pc": 1.2e-5,
                "cov_eci_pos_m2": [
                    [900.0, 1.0, 2.0],
                    [1.0, 1600.0, 3.0],
                    [2.0, 3.0, 2500.0],
                ],
            },
        ],
        "covariance_model": "execution_error_seed_grown_by_stm",
        "p_post_source": "cdm_primary_own",
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _clean_store():
    init_db()
    with get_session() as session:
        session.query(ScreeningRecord).delete()
    yield
    with get_session() as session:
        session.query(ScreeningRecord).delete()


class TestItStoresAndReadsBack:
    def test_persist_returns_ok_so_the_planner_keeps_its_screening_id(self):
        """The planner only checks the status code; a non-2xx is what used to
        make screening_record_id come back null."""
        with TestClient(app) as c:
            r = c.post("/screening/persist", json=a_record())
        assert r.status_code in (200, 201), r.text
        body = r.json()
        assert body["stored"] is True
        assert body["decision_log_id"] == _DECISION_LOG_ID

    def test_the_record_is_retrievable_by_decision_log_id(self):
        with TestClient(app) as c:
            c.post("/screening/persist", json=a_record())
            g = c.get(f"/store/screening/{_DECISION_LOG_ID}")
        assert g.status_code == 200, g.text
        got = g.json()
        assert got["decision_log_id"] == _DECISION_LOG_ID
        assert got["screening_id"] == "9f2c41aa7e2b4f0f8d2a"
        assert got["clear"] is True
        assert got["cdm_count"] == 12
        assert got["conjunction_count"] == 2
        assert got["p_post_source"] == "cdm_primary_own"
        assert got["covariance_model"] == "execution_error_seed_grown_by_stm"

    def test_the_conjunctions_survive_the_round_trip_intact(self):
        """The point of the record: the verdict is auditable against the exact
        events behind it, covariance included."""
        with TestClient(app) as c:
            c.post("/screening/persist", json=a_record())
            got = c.get(f"/store/screening/{_DECISION_LOG_ID}").json()
        assert got["conjunctions"] == a_record()["conjunctions"]

    def test_a_not_clear_verdict_keeps_its_breaches(self):
        rec = a_record(clear=False, breaches=["pc_threshold", "miss_distance"])
        with TestClient(app) as c:
            c.post("/screening/persist", json=rec)
            got = c.get(f"/store/screening/{_DECISION_LOG_ID}").json()
        assert got["clear"] is False
        assert got["breaches"] == ["pc_threshold", "miss_distance"]


class TestSkippedIsKeptWhole:
    """A non-empty skipped means the set the verdict was formed over was
    incomplete. The reasons are the audit value, so they are not reduced away."""

    def test_the_skipped_list_and_its_reasons_are_stored(self):
        skipped = [
            {"cdm_id": "1", "reason": "covariance_not_psd"},
            {"cdm_id": "2", "reason": "missing_tca"},
        ]
        with TestClient(app) as c:
            c.post("/screening/persist", json=a_record(skipped=skipped))
            got = c.get(f"/store/screening/{_DECISION_LOG_ID}").json()
        assert got["skipped"] == skipped
        assert got["skipped_count"] == 2

    def test_an_empty_skipped_reads_back_as_empty_not_null(self):
        with TestClient(app) as c:
            c.post("/screening/persist", json=a_record())
            got = c.get(f"/store/screening/{_DECISION_LOG_ID}").json()
        assert got["skipped"] == []
        assert got["skipped_count"] == 0


class TestRetriesDoNotAccumulate:
    """Persist-on-evaluate offers the same screening on every evaluate."""

    def test_a_duplicate_post_upserts_rather_than_erroring(self):
        with TestClient(app) as c:
            first = c.post("/screening/persist", json=a_record())
            second = c.post("/screening/persist", json=a_record())
        assert first.status_code in (200, 201)
        assert second.status_code in (200, 201), second.text
        assert first.json()["created"] is True
        assert second.json()["created"] is False

    def test_a_duplicate_post_leaves_one_row(self):
        with TestClient(app) as c:
            for _ in range(3):
                c.post("/screening/persist", json=a_record())
        with get_session() as session:
            rows = (
                session.query(ScreeningRecord)
                .filter(ScreeningRecord.decision_log_id == _DECISION_LOG_ID)
                .all()
            )
        assert len(rows) == 1

    def test_a_re_screen_updates_the_stored_verdict(self):
        """A second screen for the same decision replaces, rather than leaving
        the first verdict as the answer to a query for that decision."""
        with TestClient(app) as c:
            c.post("/screening/persist", json=a_record(clear=True))
            c.post("/screening/persist", json=a_record(clear=False,
                                                       breaches=["pc_threshold"]))
            got = c.get(f"/store/screening/{_DECISION_LOG_ID}").json()
        assert got["clear"] is False
        assert got["breaches"] == ["pc_threshold"]

    def test_different_decisions_each_get_their_own_row(self):
        other = "conj-8_36508_2026-09-24T13:00:00.000000Z"
        with TestClient(app) as c:
            c.post("/screening/persist", json=a_record())
            c.post("/screening/persist", json=a_record(
                decision_log_id=other, screening_id="other-screening-id"))
            a = c.get(f"/store/screening/{_DECISION_LOG_ID}").json()
            b = c.get(f"/store/screening/{other}").json()
        assert a["screening_id"] == "9f2c41aa7e2b4f0f8d2a"
        assert b["screening_id"] == "other-screening-id"


class TestItFailsCleanly:
    def test_an_unknown_decision_log_id_404s(self):
        with TestClient(app) as c:
            g = c.get("/store/screening/no-such-decision")
        assert g.status_code == 404
        assert "error" in g.json()

    def test_a_record_with_neither_id_is_rejected(self):
        with TestClient(app) as c:
            r = c.post("/screening/persist",
                       json=a_record(decision_log_id=None, screening_id=None))
        assert r.status_code == 400

    def test_a_non_object_body_is_rejected_not_500(self):
        with TestClient(app) as c:
            r = c.post("/screening/persist", json=["not", "a", "record"])
        assert r.status_code == 422

    def test_a_missing_clear_stores_as_null(self):
        """verdict is None when the screen could not form one; that is not False."""
        with TestClient(app) as c:
            c.post("/screening/persist", json=a_record(clear=None))
            got = c.get(f"/store/screening/{_DECISION_LOG_ID}").json()
        assert got["clear"] is None
