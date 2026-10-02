"""SCRUM-453 item 1: the planner's screening record reaches the ingest store.

The planner has been building and POSTing this record since SCRUM-442, against an
endpoint that did not exist. The POST 404'd, the planner logged an audit failure
and returned None, and every decision came back with screening_record_id null.
SCRUM-453 adds the endpoint; the planner side is unchanged.

Because the two halves were written apart, the thing most worth testing is not
either half on its own but that they FIT: the record _persist_screening_result
assembles has to be the record POST /screening/persist accepts. So the contract
test below runs the real planner function against the real ingest app -- no
hand-written payload in between, which is where a mismatch would hide -- and then
reads the record back out of the store.

The fail-safe is pinned alongside it. A store that is down, erroring, or slow
must cost the decision its audit link and nothing else: the planner returns the
decision it already made. A decision that was made must still be returned even if
we could not write down what it was made from.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import server

# Load the ingest app by explicit path under a unique name: both services call
# their entrypoint main.py, so a plain import would collide in a full-repo run.
_INGEST_ROOT = Path(__file__).resolve().parents[3] / "services" / "ingest"
_spec = importlib.util.spec_from_file_location(
    "ingest_main_for_contract", _INGEST_ROOT / "main.py")


@pytest.fixture(scope="module")
def ingest_client():
    """A TestClient over the real ingest app, with a throwaway store."""
    import os
    os.environ.setdefault("AVERA_DB_URL", "sqlite:////tmp/avera_contract_pytest.db")
    if str(_INGEST_ROOT) not in sys.path:
        sys.path.insert(0, str(_INGEST_ROOT))
    module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(module)
    from fastapi.testclient import TestClient
    with TestClient(module.app) as client:
        yield client


_DECISION_LOG_ID = "conj-453_36508_2026-10-02T12:00:00.000000Z"


def a_screening_result(n_conjunctions: int = 2, skipped=None):
    """Stands in for LeoLabsScreeningResult with only what the mapper reads."""
    conjunctions = []
    for i in range(n_conjunctions):
        conjunctions.append(SimpleNamespace(
            provenance={"cdm_id": f"cdm-{i}", "event_id": f"evt-{i}"},
            t_ca_utc=f"2026-10-03T0{i}:00:00.000000Z",
            miss_distance_m=500.0 + i,
            cdm_collision_probability=1.0e-5 * (i + 1),
            secondary=SimpleNamespace(
                norad_id=44713 + i,
                designator=str(44713 + i),
                object_name=f"STARLINK-{1000 + i}",
                cov_eci_pos_m2=[[900.0, 1.0, 2.0],
                                [1.0, 1600.0, 3.0],
                                [2.0, 3.0, 2500.0]],
            ),
        ))
    return SimpleNamespace(
        screening_id="screen-453-abc",
        cdm_count=7,
        skipped=skipped if skipped is not None else [],
        conjunctions=conjunctions,
    )


def a_verdict(clear=True, breaches=None):
    return SimpleNamespace(clear=clear, breaches=breaches or [])


class TestThePlannerRecordFitsTheIngestEndpoint:
    """The real function against the real app. No payload written by hand."""

    def _persist_through(self, client, result, verdict, decision_log_id):
        """Route the planner's POST into the ingest TestClient."""
        def _post(url, json=None, timeout=None, **kwargs):
            assert url.endswith("/screening/persist"), url
            return client.post("/screening/persist", json=json)

        with patch.object(server.http_requests, "post", side_effect=_post):
            return server._persist_screening_result(
                result, verdict, decision_log_id, p_post_source="cdm_primary_own")

    def test_the_post_is_accepted_and_an_id_comes_back(self, ingest_client):
        """This is what used to 404. A non-null return is the acceptance."""
        got = self._persist_through(
            ingest_client, a_screening_result(), a_verdict(), _DECISION_LOG_ID)
        assert got is not None
        assert got == "screen-453-abc"

    def test_the_record_is_then_retrievable_by_decision_log_id(self, ingest_client):
        self._persist_through(
            ingest_client, a_screening_result(), a_verdict(clear=False,
                                                           breaches=["pc_threshold"]),
            _DECISION_LOG_ID + "-retrieve")
        got = ingest_client.get(f"/store/screening/{_DECISION_LOG_ID}-retrieve")
        assert got.status_code == 200, got.text
        body = got.json()
        assert body["screening_id"] == "screen-453-abc"
        assert body["clear"] is False
        assert body["breaches"] == ["pc_threshold"]
        assert body["p_post_source"] == "cdm_primary_own"

    def test_the_conjunctions_and_their_covariance_survive(self, ingest_client):
        """The record exists so a verdict can be audited against its events."""
        self._persist_through(
            ingest_client, a_screening_result(n_conjunctions=3), a_verdict(),
            _DECISION_LOG_ID + "-conj")
        body = ingest_client.get(f"/store/screening/{_DECISION_LOG_ID}-conj").json()
        assert body["conjunction_count"] == 3
        assert len(body["conjunctions"]) == 3
        first = body["conjunctions"][0]
        assert first["secondary_name"] == "STARLINK-1000"
        assert first["cov_eci_pos_m2"] == [[900.0, 1.0, 2.0],
                                           [1.0, 1600.0, 3.0],
                                           [2.0, 3.0, 2500.0]]

    def test_a_skipped_cdm_list_reaches_the_store_with_its_reasons(self, ingest_client):
        """skipped is a LIST of unparseable CDMs, not a count. A non-empty one
        means the set the verdict was formed over was incomplete."""
        skipped = [{"cdm_id": "bad-1", "reason": "covariance_not_psd"}]
        self._persist_through(
            ingest_client, a_screening_result(skipped=skipped), a_verdict(),
            _DECISION_LOG_ID + "-skipped")
        body = ingest_client.get(f"/store/screening/{_DECISION_LOG_ID}-skipped").json()
        assert body["skipped"] == skipped
        assert body["skipped_count"] == 1

    def test_re_persisting_the_same_decision_does_not_accumulate(self, ingest_client):
        """Persist-on-evaluate offers the same screening on every evaluate."""
        key = _DECISION_LOG_ID + "-repeat"
        for _ in range(3):
            got = self._persist_through(
                ingest_client, a_screening_result(), a_verdict(), key)
            assert got is not None
        body = ingest_client.get(f"/store/screening/{key}").json()
        assert body["decision_log_id"] == key


class TestTheStoreNeverReachesBackIntoTheDecision:
    """Unchanged fail-safe behaviour, pinned so the new endpoint cannot make a
    store problem fatal."""

    def test_a_404_still_returns_none_rather_than_raising(self):
        """What happened before the endpoint existed."""
        resp = MagicMock(status_code=404)
        with patch.object(server.http_requests, "post", return_value=resp):
            got = server._persist_screening_result(
                a_screening_result(), a_verdict(), _DECISION_LOG_ID)
        assert got is None

    def test_a_500_still_returns_none(self):
        resp = MagicMock(status_code=500)
        with patch.object(server.http_requests, "post", return_value=resp):
            got = server._persist_screening_result(
                a_screening_result(), a_verdict(), _DECISION_LOG_ID)
        assert got is None

    def test_an_unreachable_store_still_returns_none(self):
        with patch.object(server.http_requests, "post",
                          side_effect=OSError("connection refused")):
            got = server._persist_screening_result(
                a_screening_result(), a_verdict(), _DECISION_LOG_ID)
        assert got is None

    def test_a_none_result_is_not_posted_at_all(self):
        posts = MagicMock()
        with patch.object(server.http_requests, "post", posts):
            got = server._persist_screening_result(None, a_verdict(), _DECISION_LOG_ID)
        assert got is None
        posts.assert_not_called()

    def test_an_unmappable_result_is_not_posted(self):
        """A mapping failure must be caught before the POST, not raise."""
        posts = MagicMock()
        broken = SimpleNamespace(screening_id="x", cdm_count=0, skipped=[],
                                 conjunctions=[SimpleNamespace()])  # missing fields
        with patch.object(server.http_requests, "post", posts):
            got = server._persist_screening_result(broken, a_verdict(), _DECISION_LOG_ID)
        assert got is None
        posts.assert_not_called()
