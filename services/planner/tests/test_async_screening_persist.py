"""tests/test_async_screening_persist.py

SCRUM-484: the ASYNC screen's record reaches the store.

SCRUM-453 added the ingest endpoint and the mapper, and they worked -- but only
the inline screen ever called them. The screen runs async by default (SCRUM-456,
because a 30 s to 2 min screen cannot run inside a 60 s evaluate), and on that
path nothing persisted: the evaluate returns while the screen is still running,
so the single inline persist was handed result=None and returned before posting.
A live decision came back with screening_record_id null and no record was stored,
which left a CLEAR or NOT CLEAR auditable only through the service logs.

The fix has two halves and both are pinned here: the decision log id is stamped
onto the screen's store entry during the evaluate, and the poll endpoint persists
once the screen is terminal. The poll is the only moment both halves exist -- the
decision log id does not exist when the worker starts, and the result does not
exist when the evaluate ends.

The safety direction is what matters most, because this adds a side effect to a
read-only endpoint that a gate is polled through. Persistence must never be able
to change the verdict. So the fail-safe tests here are not an afterthought: a
store that is down, erroring or unreachable has to leave the poll's clear /
not-clear answer byte for byte what it would have been, and cost only the record.

Run from repo root:
    python -m pytest services/planner/tests/test_async_screening_persist.py -v
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import server
from common import atlas_artifact as aa
from common import leolabs_screening as ls
from common import secondary_screen_async as sa
from common.leolabs_cdm_parser import parse_leolabs_cdm
from common.leolabs_screening import ScreeningResult
from common.operator_policy import OperatorPolicy

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"
_EPOCH = "2026-09-24T12:00:00Z"
_R_POST = [6838.0, 0.0, 0.0]
_V_POST = [0.0, 0.3419, 7.5754]
_P_POST = [[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.16]]

_DECISION_LOG_ID = "conj-484_36508_2026-10-02T18:00:00.000000Z"

_INGEST_ROOT = Path(__file__).resolve().parents[3] / "services" / "ingest"


@pytest.fixture(autouse=True)
def _clean_store():
    sa.reset_store()
    yield
    sa.reset_store()


@pytest.fixture(scope="module")
def ingest_client():
    """A TestClient over the real ingest app, with a throwaway store."""
    import os
    os.environ.setdefault("AVERA_DB_URL", "sqlite:////tmp/avera_484_pytest.db")
    if str(_INGEST_ROOT) not in sys.path:
        sys.path.insert(0, str(_INGEST_ROOT))
    spec = importlib.util.spec_from_file_location(
        "ingest_main_for_484", _INGEST_ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with TestClient(module.app) as client:
        yield client


@pytest.fixture
def policy() -> OperatorPolicy:
    return OperatorPolicy(operator_id="test", policy_version="v1",
                          decision_mode="flight_rule_1e4")


@pytest.fixture
def cap():
    return MagicMock(radius_m=1.0)


@pytest.fixture
def cdm() -> dict:
    return json.loads(_FIXTURE.read_text())


def _parsed(cdm_dict, **overrides):
    out = copy.deepcopy(cdm_dict)
    out.update(overrides)
    return parse_leolabs_cdm(out, "L2669", validate_miss_distance=False,
                             strict_psd=False)


def _stub_screening(monkeypatch, *, conjunctions=(), screening_id="scr-484"):
    def _run(*a, **k):
        return ScreeningResult(screening_id=screening_id,
                               conjunctions=list(conjunctions),
                               cdm_count=len(conjunctions))
    monkeypatch.setattr(ls, "run_screening", _run)


def _start_resolved(policy, cap, monkeypatch, cdm_dict, *,
                    seed_source="cdm_primary_own", screening_id="scr-484"):
    """Run an async screen to completion on the calling thread."""
    _stub_screening(monkeypatch, conjunctions=[_parsed(cdm_dict)],
                    screening_id=screening_id)
    check = aa._start_on_demand_secondary_check(
        r_post_km=_R_POST, v_post_km_s=_V_POST, p_post_eci_km2=_P_POST,
        screening_epoch_utc=_EPOCH, policy=policy, cap=cap,
        primary_catalog_number="L2669", submit=lambda job: job(),
        p_post_source=seed_source,
    )
    entry = sa.get_entry(check.screen_job_id)
    assert entry is not None and entry.is_terminal, "screen did not resolve"
    return check, entry


def _poll(job_id: str, ingest_client=None):
    """GET the poll endpoint, optionally routing the persist into real ingest."""
    client = TestClient(server.svc)
    if ingest_client is None:
        return client.get(f"/v1/secondary-screen/{job_id}")

    def _post(url, json=None, timeout=None, **kwargs):
        assert url.endswith("/screening/persist"), url
        return ingest_client.post("/screening/persist", json=json)

    with patch.object(server.http_requests, "post", side_effect=_post):
        return client.get(f"/v1/secondary-screen/{job_id}")


# ---------------------------------------------------------------------------
# The decision log id gets onto the entry
# ---------------------------------------------------------------------------

class TestTheDecisionIsStampedOntoTheScreen:
    def test_a_fresh_entry_has_no_decision_log_id(self, policy, cap, monkeypatch, cdm):
        _, entry = _start_resolved(policy, cap, monkeypatch, cdm)
        assert entry.decision_log_id is None

    def test_the_setter_stamps_it(self, policy, cap, monkeypatch, cdm):
        check, entry = _start_resolved(policy, cap, monkeypatch, cdm)
        assert sa.set_decision_log_id(check.screen_job_id, _DECISION_LOG_ID) is True
        assert sa.get_entry(check.screen_job_id).decision_log_id == _DECISION_LOG_ID

    def test_stamping_an_unknown_job_reports_false_rather_than_raising(self):
        """Evicted, or reset. Not an error; the screen just cannot be tied back."""
        assert sa.set_decision_log_id("never-existed", _DECISION_LOG_ID) is False

    def test_the_evaluate_helper_stamps_through_the_artifact(
            self, policy, cap, monkeypatch, cdm):
        """The seam the evaluate actually uses, not the setter directly."""
        check, _ = _start_resolved(policy, cap, monkeypatch, cdm)
        artifact = MagicMock()
        artifact.post_maneuver.secondary_conflict = check
        server._stamp_async_screen_decision(artifact, _DECISION_LOG_ID)
        assert sa.get_entry(check.screen_job_id).decision_log_id == _DECISION_LOG_ID

    def test_an_artifact_with_no_screen_is_a_no_op(self):
        """Deferred, inline, disabled, or no maneuver recommended."""
        artifact = MagicMock()
        artifact.post_maneuver.secondary_conflict.screen_job_id = None
        server._stamp_async_screen_decision(artifact, _DECISION_LOG_ID)   # no raise

    def test_an_artifact_without_a_post_maneuver_block_is_a_no_op(self):
        broken = object()
        server._stamp_async_screen_decision(broken, _DECISION_LOG_ID)     # no raise


# ---------------------------------------------------------------------------
# The record reaches the real ingest store
# ---------------------------------------------------------------------------

class TestTheResolvedScreenPersists:
    def test_polling_a_terminal_screen_writes_the_record(
            self, policy, cap, monkeypatch, cdm, ingest_client):
        """The acceptance: what used to leave screening_record_id null."""
        check, _ = _start_resolved(policy, cap, monkeypatch, cdm)
        key = _DECISION_LOG_ID + "-write"
        sa.set_decision_log_id(check.screen_job_id, key)

        resp = _poll(check.screen_job_id, ingest_client)
        assert resp.status_code == 200, resp.text
        assert resp.json()["screening_record_id"] == "scr-484"

        stored = ingest_client.get(f"/store/screening/{key}")
        assert stored.status_code == 200, stored.text
        body = stored.json()
        assert body["decision_log_id"] == key
        assert body["screening_id"] == "scr-484"

    def test_the_poll_response_reports_the_decision_it_belongs_to(
            self, policy, cap, monkeypatch, cdm, ingest_client):
        check, _ = _start_resolved(policy, cap, monkeypatch, cdm)
        key = _DECISION_LOG_ID + "-reports"
        sa.set_decision_log_id(check.screen_job_id, key)
        body = _poll(check.screen_job_id, ingest_client).json()
        assert body["decision_log_id"] == key

    def test_the_seed_source_is_carried_through_to_the_store(
            self, policy, cap, monkeypatch, cdm, ingest_client):
        """Ties to SCRUM-453 item 2: a stored-path decision seeds cdm_primary_own
        and the record has to say so, or the audit cannot tell which quantity
        the screen was grown from."""
        check, _ = _start_resolved(policy, cap, monkeypatch, cdm,
                                   seed_source="cdm_primary_own")
        key = _DECISION_LOG_ID + "-seed"
        sa.set_decision_log_id(check.screen_job_id, key)
        _poll(check.screen_job_id, ingest_client)
        assert ingest_client.get(
            f"/store/screening/{key}").json()["p_post_source"] == "cdm_primary_own"

    def test_the_conjunctions_reach_the_store(
            self, policy, cap, monkeypatch, cdm, ingest_client):
        """The record exists so a verdict can be audited against its events."""
        check, _ = _start_resolved(policy, cap, monkeypatch, cdm)
        key = _DECISION_LOG_ID + "-conj"
        sa.set_decision_log_id(check.screen_job_id, key)
        _poll(check.screen_job_id, ingest_client)
        body = ingest_client.get(f"/store/screening/{key}").json()
        assert body["conjunction_count"] == 1
        assert len(body["conjunctions"]) == 1
        assert body["conjunctions"][0]["cov_eci_pos_m2"]


class TestARepeatPollDoesNotRePost:
    def test_the_second_poll_reports_the_same_id(
            self, policy, cap, monkeypatch, cdm, ingest_client):
        check, _ = _start_resolved(policy, cap, monkeypatch, cdm)
        key = _DECISION_LOG_ID + "-repeat"
        sa.set_decision_log_id(check.screen_job_id, key)
        first = _poll(check.screen_job_id, ingest_client).json()
        second = _poll(check.screen_job_id, ingest_client).json()
        assert first["screening_record_id"] == second["screening_record_id"] == "scr-484"

    def test_the_second_poll_does_not_post_again(
            self, policy, cap, monkeypatch, cdm, ingest_client):
        check, _ = _start_resolved(policy, cap, monkeypatch, cdm)
        key = _DECISION_LOG_ID + "-nopost"
        sa.set_decision_log_id(check.screen_job_id, key)
        _poll(check.screen_job_id, ingest_client)          # writes it

        posts = MagicMock(return_value=MagicMock(status_code=200))
        with patch.object(server.http_requests, "post", posts):
            again = TestClient(server.svc).get(
                f"/v1/secondary-screen/{check.screen_job_id}")
        assert again.status_code == 200
        posts.assert_not_called()

    def test_the_stored_row_is_unchanged_by_the_repeat(
            self, policy, cap, monkeypatch, cdm, ingest_client):
        check, _ = _start_resolved(policy, cap, monkeypatch, cdm)
        key = _DECISION_LOG_ID + "-unchanged"
        sa.set_decision_log_id(check.screen_job_id, key)
        _poll(check.screen_job_id, ingest_client)
        before = ingest_client.get(f"/store/screening/{key}").json()
        _poll(check.screen_job_id, ingest_client)
        after = ingest_client.get(f"/store/screening/{key}").json()
        assert before == after


class TestNothingElsePersists:
    def test_a_pending_screen_does_not_persist(self, policy, cap, monkeypatch, cdm):
        """Nothing is submitted, so the screen cannot have resolved."""
        _stub_screening(monkeypatch, conjunctions=[_parsed(cdm)])
        check = aa._start_on_demand_secondary_check(
            r_post_km=_R_POST, v_post_km_s=_V_POST, p_post_eci_km2=_P_POST,
            screening_epoch_utc=_EPOCH, policy=policy, cap=cap,
            primary_catalog_number="L2669", submit=lambda job: None,
        )
        sa.set_decision_log_id(check.screen_job_id, _DECISION_LOG_ID + "-pending")

        posts = MagicMock()
        with patch.object(server.http_requests, "post", posts):
            body = _poll(check.screen_job_id).json()
        posts.assert_not_called()
        assert body["pending"] is True
        assert body["clear"] is False
        assert body["screening_record_id"] is None

    def test_a_terminal_screen_without_a_decision_log_id_does_not_persist(
            self, policy, cap, monkeypatch, cdm):
        """The record is keyed to the decision; without one there is nothing to
        key it to, and inventing a key would be worse than not writing."""
        check, _ = _start_resolved(policy, cap, monkeypatch, cdm)
        posts = MagicMock()
        with patch.object(server.http_requests, "post", posts):
            body = _poll(check.screen_job_id).json()
        posts.assert_not_called()
        assert body["screening_record_id"] is None

    def test_an_errored_screen_with_no_result_does_not_persist_and_does_not_raise(
            self, policy, cap, monkeypatch):
        def _boom(*a, **k):
            raise ls.LeoLabsScreeningUnavailable("no access")
        monkeypatch.setattr(ls, "run_screening", _boom)
        check = aa._start_on_demand_secondary_check(
            r_post_km=_R_POST, v_post_km_s=_V_POST, p_post_eci_km2=_P_POST,
            screening_epoch_utc=_EPOCH, policy=policy, cap=cap,
            primary_catalog_number="L2669", submit=lambda job: job(),
        )
        sa.set_decision_log_id(check.screen_job_id, _DECISION_LOG_ID + "-error")

        posts = MagicMock()
        with patch.object(server.http_requests, "post", posts):
            resp = _poll(check.screen_job_id)
        assert resp.status_code == 200
        posts.assert_not_called()
        body = resp.json()
        assert body["status"] == sa.STATUS_ERROR
        assert body["clear"] is False            # an error is still NOT clear
        assert body["screening_record_id"] is None


# ---------------------------------------------------------------------------
# The verdict never depends on whether the record was written
# ---------------------------------------------------------------------------

class TestTheGuardsAreIndividuallyLoadBearing:
    """Each early return in _persist_resolved_screen, pinned on its own.

    Without these, a guard can be deleted and the suite still passes because a
    later guard happens to catch the same case.
    """

    def test_a_pending_entry_is_refused_even_if_it_somehow_carries_a_result(
            self, policy, cap, monkeypatch, cdm):
        """Pins the is_terminal check specifically rather than relying on the
        result being absent, which is what normally stops a pending entry."""
        check, entry = _start_resolved(policy, cap, monkeypatch, cdm)
        sa.set_decision_log_id(check.screen_job_id, _DECISION_LOG_ID + "-forced")
        # Terminal with a result, then forced back to pending.
        live = sa.get_entry(check.screen_job_id)
        live.status = sa.STATUS_PENDING
        assert live.result is not None and live.is_terminal is False

        posts = MagicMock()
        with patch.object(server.http_requests, "post", posts):
            resp = _poll(check.screen_job_id)
        assert resp.status_code == 200
        posts.assert_not_called()

    def test_an_unexpected_error_in_the_persist_is_swallowed(
            self, policy, cap, monkeypatch, cdm):
        """_persist_screening_result swallows its own failures, so this pins the
        outer guard: anything else -- an evicted entry, a threadpool refusal --
        must still not reach the caller, because the poll reports a safety
        verdict and must answer."""
        check, _ = _start_resolved(policy, cap, monkeypatch, cdm)
        sa.set_decision_log_id(check.screen_job_id, _DECISION_LOG_ID + "-boom")
        with patch.object(server, "_persist_screening_result",
                          side_effect=RuntimeError("unexpected")):
            resp = _poll(check.screen_job_id)
        assert resp.status_code == 200
        body = resp.json()
        assert body["clear"] is True
        assert body["screening_record_id"] is None


class TestPersistenceCannotChangeTheVerdict:
    def _verdict_fields(self, body):
        return {k: body[k] for k in
                ("status", "clear", "pending", "verdict",
                 "secondary_check_performed", "secondary_conjunction_clear")}

    def test_an_unreachable_store_leaves_the_verdict_intact(
            self, policy, cap, monkeypatch, cdm):
        check, _ = _start_resolved(policy, cap, monkeypatch, cdm)
        sa.set_decision_log_id(check.screen_job_id, _DECISION_LOG_ID + "-down")
        with patch.object(server.http_requests, "post",
                          side_effect=OSError("connection refused")):
            resp = _poll(check.screen_job_id)
        assert resp.status_code == 200
        body = resp.json()
        assert body["clear"] is True
        assert body["screening_record_id"] is None

    def test_a_500_from_the_store_leaves_the_verdict_intact(
            self, policy, cap, monkeypatch, cdm):
        check, _ = _start_resolved(policy, cap, monkeypatch, cdm)
        sa.set_decision_log_id(check.screen_job_id, _DECISION_LOG_ID + "-500")
        with patch.object(server.http_requests, "post",
                          return_value=MagicMock(status_code=500)):
            body = _poll(check.screen_job_id).json()
        assert body["clear"] is True
        assert body["screening_record_id"] is None

    def test_the_poll_payload_is_identical_with_and_without_the_store(
            self, policy, cap, monkeypatch, cdm, ingest_client):
        """The sharpest form of the fail-safe: every verdict-bearing field is the
        same whether or not the record could be written."""
        check_a, _ = _start_resolved(policy, cap, monkeypatch, cdm)
        sa.set_decision_log_id(check_a.screen_job_id, _DECISION_LOG_ID + "-a")
        with patch.object(server.http_requests, "post",
                          side_effect=OSError("down")):
            without = _poll(check_a.screen_job_id).json()

        check_b, _ = _start_resolved(policy, cap, monkeypatch, cdm)
        sa.set_decision_log_id(check_b.screen_job_id, _DECISION_LOG_ID + "-b")
        with_store = _poll(check_b.screen_job_id, ingest_client).json()

        assert self._verdict_fields(without) == self._verdict_fields(with_store)
        # And the only difference that matters is the record id itself.
        assert without["screening_record_id"] is None
        assert with_store["screening_record_id"] == "scr-484"

    def test_a_not_clear_screen_still_reads_not_clear_when_the_store_is_down(
            self, policy, cap, monkeypatch, cdm):
        """The direction that actually matters for safety."""
        breaching = _parsed(cdm, MISS_DISTANCE=500.0)
        _stub_screening(monkeypatch, conjunctions=[breaching])
        check = aa._start_on_demand_secondary_check(
            r_post_km=_R_POST, v_post_km_s=_V_POST, p_post_eci_km2=_P_POST,
            screening_epoch_utc=_EPOCH, policy=policy, cap=cap,
            primary_catalog_number="L2669", submit=lambda job: job(),
        )
        sa.set_decision_log_id(check.screen_job_id, _DECISION_LOG_ID + "-notclear")
        with patch.object(server.http_requests, "post",
                          side_effect=OSError("down")):
            body = _poll(check.screen_job_id).json()
        assert body["status"] == sa.STATUS_NOT_CLEAR
        assert body["clear"] is False


class TestTheEvaluateStampsTheScreenItStarted:
    """The integration the whole fix hangs on.

    Everything above stamps the entry by calling the setter directly. That leaves
    the actual wiring uncovered -- the evaluate reaching into the artifact for the
    job id and stamping it once the decision log exists -- and that step is
    precisely what did not exist before this ticket. Without a test here, deleting
    the call from the evaluate leaves the whole suite green while no live decision
    ever persists again.

    Asserted with a spy on the stamp rather than by driving a screen to start,
    because whether a screen starts depends on the scoring recommending a
    maneuver, and tuning a synthetic conjunction until it does would make this
    test about the physics instead of about the wiring. The helper's own
    behaviour is covered by TestTheDecisionIsStampedOntoTheScreen.
    """

    def test_the_evaluate_stamps_with_the_decision_log_id_it_just_created(
            self, monkeypatch, cdm):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        iso = lambda d: d.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

        def _ingest_get(url, *a, **k):
            resp = MagicMock(status_code=200, ok=True)
            resp.json.return_value = {"next_seq": 0, "content_hash": "0" * 64}
            return resp

        body = {
            "conjunction_id": "SCRUM-484-STAMP",
            "satellite": {
                "sat_id": "36508",
                "r_sat_km": [-1484.865223, -5293.446853, -4495.437378],
                "v_sat_km_s": [6.464033802, 0.661818202, -3.00266646],
                "t_burn_utc": iso(now - timedelta(hours=1)),
                "v_remaining_m_s": 45.0,
            },
            "conjunction": {
                "obj_id": "DEB",
                "t_ca_utc": iso(now + timedelta(hours=3)),
                "r_rel_km": [0.0005, 0.0004, 0.0003],
                "primary_norad": "36508",
                "secondary_norad": "270302",
            },
            "policy": {"lambda_v": 1.0, "lambda_L": 0.8,
                       "dv_mag_limit_m_s": 2.0, "a_ref_km": 7000.0},
        }

        seen = []
        real_stamp = server._stamp_async_screen_decision

        def _spy(artifact, decision_log_id):
            seen.append(decision_log_id)
            return real_stamp(artifact, decision_log_id)

        with patch.object(server, "UDL_ENABLED", False), \
             patch.object(server, "LEOLABS_ENABLED", False), \
             patch.object(server, "SECONDARY_SCREEN_ENABLED", True), \
             patch.object(server, "_stamp_async_screen_decision", _spy), \
             patch.object(server.http_requests, "post",
                          MagicMock(return_value=MagicMock(status_code=201))), \
             patch.object(server.http_requests, "get", side_effect=_ingest_get):
            resp = TestClient(server.svc).post("/v1/evaluate", json=body)

        assert resp.status_code == 200, resp.text
        decision_log_id = resp.json().get("decision_log_id")
        assert decision_log_id, "the evaluate produced no decision log id"
        assert seen == [decision_log_id], (
            "the evaluate did not tie its async screen to the decision it just "
            "logged, so a resolved screen could never be persisted"
        )

    def test_the_stamp_happens_after_the_decision_log_exists(self):
        """Ordering, read out of the source: stamping before the log is created
        would stamp None and silently disable the persist."""
        source = Path("services/planner/server.py").read_text()
        created = source.index("result[\"decision_log_id\"] = decision_log.log_id")
        stamped = source.index("_stamp_async_screen_decision(artifact, decision_log.log_id)")
        assert created < stamped


class TestTheInlinePathIsUntouched:
    def test_the_inline_persist_call_site_still_exists(self):
        """SCRUM-484 adds a second trigger; it does not move the first."""
        source = Path("services/planner/server.py").read_text()
        assert "_screening_capture.get(\"result\")" in source
        assert source.count("_persist_screening_result(") >= 2

    def test_the_mapper_itself_is_not_reimplemented(self):
        """The async path reuses SCRUM-453's mapper rather than building a
        second record shape that could drift from it."""
        source = Path("services/planner/server.py").read_text()
        start = source.index("async def _persist_resolved_screen")
        end = source.index("def _async_screen_job_id")
        body = source[start:end]
        # Passed by reference into the threadpool, so it appears without a call
        # paren -- the point is that the SCRUM-453 mapper is what runs.
        assert "_persist_screening_result" in body
        assert "/screening/persist" not in body      # no second ingest client
        assert "await run_in_threadpool(" in body    # and never on the event loop
