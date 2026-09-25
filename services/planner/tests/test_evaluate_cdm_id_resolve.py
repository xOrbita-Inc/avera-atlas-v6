"""tests/test_evaluate_cdm_id_resolve.py

SCRUM-460: evaluate a clicked conjunction without pulling the whole window, and
never on the event loop.

SCRUM-459 bounded and offloaded the LIST fetch. Evaluate was out of scope there and
still resolved the operator's clicked row by pulling the entire in-volume window and
filtering it -- synchronously, on a single uvicorn worker. For SWARM B that is 75,257
CDMs over 77 pages and 773 s, so clicking a SWARM B row wedged the planner exactly
as the list used to.

Two things had to be true, and they are what this file tests:

1. A clicked cdm_id resolves with NO window search, and to the SAME conjunction the
   window search would have found -- because this must change how the conjunction is
   found and nothing about how it is judged.
2. No evaluate, on any path, blocks /health. The cdm_id path is two requests and the
   fallback is still the whole window; both run off the loop.

What the API actually allows, confirmed against it (the plan's premise was wrong):
`/catalog/conjunctions/cdms/{id}` does NOT return a CCSDS CDM. It returns a summary
-- tca, sat1, sat2, missDistance -- with no states and no covariance, so nothing can
be scored from it. It is used for the event's coordinates, and a narrow search on
that object pair inside a TCA bracket returns the real CDMs in one page.

Run from repo root:
    python -m pytest services/planner/tests/test_evaluate_cdm_id_resolve.py -v
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import server
from common import leolabs_runtime
from common.leolabs_asset_map import AssetRegistry
from common.leolabs_cdm_parser import parse_leolabs_cdm
from common.leolabs_runtime import fetch_leolabs_conjunction_by_cdm_id

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"
_OBJECTS = [{"catalogNumber": "L2669", "noradCatalogNumber": 36508,
             "name": "CRYOSAT 2"}]
_CDM_ID = "79867238980"
_TCA = "2026-09-26T18:05:15.953564Z"


@pytest.fixture
def registry() -> AssetRegistry:
    return AssetRegistry.from_objects(_OBJECTS)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(leolabs_runtime, "LEOLABS_ENABLED", True)
    monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
    leolabs_runtime.reset_caches()
    server._list_inflight = 0
    yield
    leolabs_runtime.reset_caches()
    server._list_inflight = 0


def _cdm(cdm_id: str, tca: str = _TCA, secondary: str = "L186018") -> dict:
    raw = json.loads(_FIXTURE.read_text())
    raw["COMMENT_ID"] = cdm_id
    raw["TCA_ISO"] = tca
    raw["SAT2_OBJECT_DESIGNATOR"] = secondary
    return raw


class _FakeClient:
    """Counts the two call kinds separately, which is the whole point.

    `searches` records every whole-window-shaped search; `narrow_searches` records
    the ones bounded to an object pair and a TCA bracket. A cdm_id resolve must make
    one of the second kind and none of the first.
    """

    def __init__(self, cdms, summaries=None):
        self._cdms = cdms
        self._summaries = summaries
        self.summary_calls = []
        self.searches = []
        self.narrow_searches = []

    def get_cdm_summaries(self, cdm_ids):
        self.summary_calls.append(cdm_ids)
        if self._summaries is not None:
            return self._summaries
        return [{"id": int(cdm_ids), "conjunction": 3590491017, "tca": _TCA,
                 "sat1": "L2669", "sat2": "L186018",
                 "missDistance": 11226.6, "collisionProbability": 7.03e-05}]

    def search_conjunction_cdms(self, **kw):
        if kw.get("object2") and kw.get("min_tca") and kw.get("max_tca"):
            self.narrow_searches.append(kw)
        else:
            self.searches.append(kw)
        return list(self._cdms)


# ---------------------------------------------------------------------------
# 1. A cdm_id resolves with no window pull
# ---------------------------------------------------------------------------

class TestCdmIdResolve:
    def test_it_looks_the_id_up_and_narrow_searches_its_tca_bracket(self, registry):
        client = _FakeClient([_cdm(_CDM_ID)])
        parsed = fetch_leolabs_conjunction_by_cdm_id(
            _CDM_ID, 36508, client=client, registry=registry)

        assert parsed is not None
        assert client.summary_calls == [_CDM_ID]
        assert len(client.narrow_searches) == 1
        assert client.searches == [], "a whole-window search was made"

    def test_the_bracket_is_centred_on_the_events_own_tca(self, registry):
        client = _FakeClient([_cdm(_CDM_ID)])
        fetch_leolabs_conjunction_by_cdm_id(
            _CDM_ID, 36508, client=client, registry=registry)

        kw = client.narrow_searches[0]
        lo = datetime.fromisoformat(kw["min_tca"].replace("Z", "+00:00"))
        hi = datetime.fromisoformat(kw["max_tca"].replace("Z", "+00:00"))
        tca = datetime.fromisoformat(_TCA.replace("Z", "+00:00"))
        bracket = timedelta(minutes=leolabs_runtime._CDM_ID_TCA_BRACKET_MINUTES)
        assert lo == tca - bracket
        assert hi == tca + bracket
        # One event's reissues, not a catalog sweep, so it stays one page.
        assert (hi - lo) <= timedelta(minutes=10)

    def test_the_search_is_pinned_to_the_object_pair(self, registry):
        client = _FakeClient([_cdm(_CDM_ID)])
        fetch_leolabs_conjunction_by_cdm_id(
            _CDM_ID, 36508, client=client, registry=registry)
        kw = client.narrow_searches[0]
        assert kw["object1"] == "L2669"
        assert kw["object2"] == "L186018"

    def test_it_returns_the_cdm_whose_id_was_clicked_not_the_first_in_the_bracket(
        self, registry
    ):
        """A bracket holds every reissue of the event; only one is the click."""
        others = [_cdm("11111111"), _cdm("22222222"), _cdm(_CDM_ID),
                  _cdm("33333333")]
        client = _FakeClient(others)
        parsed = fetch_leolabs_conjunction_by_cdm_id(
            _CDM_ID, 36508, client=client, registry=registry)
        assert str(parsed.provenance.get("cdm_id")) == _CDM_ID

    def test_our_asset_is_always_object1_however_the_summary_orders_them(
        self, registry
    ):
        """LeoLabs 403s the whole request if object1 is not a subscribed object.

        The summary does not put our asset in sat1 reliably. On a live SWARM B row it
        came back sat1=L150849 (a Starlink) and sat2=L5429 (the asset), and passing
        the Starlink as object1 was refused with a 403 that reads like a credentials
        problem. Both orderings must produce the same, subscribed, query.
        """
        for sat1, sat2 in (("L2669", "L186018"), ("L186018", "L2669")):
            client = _FakeClient([_cdm(_CDM_ID)], summaries=[
                {"id": 1, "tca": _TCA, "sat1": sat1, "sat2": sat2}])
            parsed = fetch_leolabs_conjunction_by_cdm_id(
                _CDM_ID, 36508, client=client, registry=registry)
            assert parsed is not None, f"failed for sat1={sat1} sat2={sat2}"
            kw = client.narrow_searches[0]
            assert kw["object1"] == "L2669", (
                f"object1 must be our subscribed asset, got {kw['object1']!r}")
            assert kw["object2"] == "L186018"

    def test_an_unresolvable_id_is_none_not_the_worst_conjunction(self, registry):
        """Falling back would score another object under the clicked label."""
        client = _FakeClient([_cdm("99999999")])     # the clicked id is absent
        assert fetch_leolabs_conjunction_by_cdm_id(
            _CDM_ID, 36508, client=client, registry=registry) is None

    def test_an_id_with_no_summary_is_none(self, registry):
        client = _FakeClient([_cdm(_CDM_ID)], summaries=[])
        assert fetch_leolabs_conjunction_by_cdm_id(
            _CDM_ID, 36508, client=client, registry=registry) is None
        assert client.narrow_searches == []          # and it did not search blindly

    def test_an_id_belonging_to_another_asset_is_refused(self, registry):
        """The id is real, but not this spacecraft's encounter."""
        client = _FakeClient([_cdm(_CDM_ID)], summaries=[
            {"id": 1, "tca": _TCA, "sat1": "L9999", "sat2": "L8888"}])
        assert fetch_leolabs_conjunction_by_cdm_id(
            _CDM_ID, 36508, client=client, registry=registry) is None
        assert client.narrow_searches == []

    def test_an_unparseable_tca_is_none_not_a_crash(self, registry):
        client = _FakeClient([_cdm(_CDM_ID)], summaries=[
            {"id": 1, "tca": "not-a-timestamp", "sat1": "L2669", "sat2": "L186018"}])
        assert fetch_leolabs_conjunction_by_cdm_id(
            _CDM_ID, 36508, client=client, registry=registry) is None

    def test_an_unsubscribed_asset_raises(self, registry):
        client = _FakeClient([_cdm(_CDM_ID)])
        with pytest.raises(leolabs_runtime.LeoLabsRuntimeError):
            fetch_leolabs_conjunction_by_cdm_id(
                44444, 44444, client=client, registry=registry)

    def test_a_cdm_that_fails_its_guards_is_none(self, registry):
        """Same treatment as on the window path: not scorable, not scored."""
        bad = _cdm(_CDM_ID)
        bad["SAT1_COVARIANCE_METHOD"] = "DEFAULT"     # guard rejects non-CALCULATED
        client = _FakeClient([bad])
        assert fetch_leolabs_conjunction_by_cdm_id(
            _CDM_ID, 36508, client=client, registry=registry) is None


# ---------------------------------------------------------------------------
# 2. The decision is unchanged
# ---------------------------------------------------------------------------

class TestDecisionUnchanged:
    def test_the_parsed_conjunction_matches_the_window_path_exactly(self, registry):
        """The property the whole ticket rests on.

        Verified against the live API too, on SWARM C: bit-identical relative
        position, velocity and combined covariance, and an identical
        to_conjunction_state(), which is the whole of what the scorer reads.
        """
        raw = _cdm(_CDM_ID)
        from_window = parse_leolabs_cdm(copy.deepcopy(raw), "L2669")

        client = _FakeClient([copy.deepcopy(raw)])
        direct = fetch_leolabs_conjunction_by_cdm_id(
            _CDM_ID, 36508, client=client, registry=registry)

        assert direct.to_conjunction_state() == from_window.to_conjunction_state()
        assert direct.t_ca_utc == from_window.t_ca_utc
        assert direct.miss_distance_m == from_window.miss_distance_m
        assert direct.combined_hbr_m == from_window.combined_hbr_m
        assert (direct.cdm_collision_probability
                == from_window.cdm_collision_probability)
        assert direct.secondary.designator == from_window.secondary.designator
        assert direct.provenance.get("cdm_id") == from_window.provenance.get("cdm_id")

    def test_the_evaluate_decision_is_the_same_through_either_seam(
        self, monkeypatch, registry
    ):
        """End to end: the recommendation for one row must not depend on the seam."""
        raw = _cdm(_CDM_ID)
        parsed = parse_leolabs_cdm(copy.deepcopy(raw), "L2669")
        body = {
            "conjunction_id": "seam-test",
            "satellite": {"sat_id": "36508", "v_remaining_m_s": 25.0},
            "conjunction": {"primary_norad": "36508", "cdm_id": _CDM_ID},
        }
        http = TestClient(server.svc)

        # via the window search (the pre-SCRUM-460 seam)
        monkeypatch.setattr(server, "fetch_leolabs_conjunctions",
                            lambda n, **kw: [parsed])
        monkeypatch.setattr(server, "fetch_leolabs_conjunction_by_cdm_id",
                            lambda *a, **kw: None)
        window_body = dict(body)
        window_body["conjunction"] = {"primary_norad": "36508",
                                      "secondary_norad": parsed.secondary.norad_id}
        via_window = http.post("/v1/evaluate", json=window_body).json()

        # via the cdm_id resolve (the new seam)
        monkeypatch.setattr(server, "fetch_leolabs_conjunction_by_cdm_id",
                            lambda *a, **kw: parsed)
        via_cdm_id = http.post("/v1/evaluate", json=body).json()

        assert via_window["recommendation"]["direction"] == \
               via_cdm_id["recommendation"]["direction"]
        assert via_window["metrics"]["pc_pre"] == via_cdm_id["metrics"]["pc_pre"]
        assert via_window["metrics"]["m2_pre"] == via_cdm_id["metrics"]["m2_pre"]
        assert via_window["conjunction"]["secondary"] == \
               via_cdm_id["conjunction"]["secondary"]


# ---------------------------------------------------------------------------
# 3. Routing: which seam each selector takes
# ---------------------------------------------------------------------------

class TestRouting:
    def _http(self, monkeypatch, registry, parsed):
        calls = {"window": 0, "by_id": 0, "worst": 0}

        def _window(n, **kw):
            calls["window"] += 1
            return [parsed]

        def _by_id(cdm_id, primary_norad, **kw):
            calls["by_id"] += 1
            return parsed

        def _worst(n, **kw):
            calls["worst"] += 1
            return parsed

        monkeypatch.setattr(server, "fetch_leolabs_conjunctions", _window)
        monkeypatch.setattr(server, "fetch_leolabs_conjunction_by_cdm_id", _by_id)
        monkeypatch.setattr(server, "fetch_leolabs_conjunction", _worst)
        return TestClient(server.svc), calls

    @staticmethod
    def _body(**selector):
        return {
            "conjunction_id": "routing",
            "satellite": {"sat_id": "36508", "v_remaining_m_s": 25.0},
            "conjunction": {"primary_norad": "36508", **selector},
        }

    @pytest.fixture
    def parsed(self):
        return parse_leolabs_cdm(_cdm(_CDM_ID), "L2669")

    def test_a_cdm_id_selector_takes_the_direct_seam_only(
        self, monkeypatch, registry, parsed
    ):
        http, calls = self._http(monkeypatch, registry, parsed)
        assert http.post("/v1/evaluate", json=self._body(
            cdm_id=_CDM_ID)).status_code == 200
        assert calls == {"window": 0, "by_id": 1, "worst": 0}

    def test_an_event_id_selector_still_uses_the_window(
        self, monkeypatch, registry, parsed
    ):
        http, calls = self._http(monkeypatch, registry, parsed)
        http.post("/v1/evaluate", json=self._body(
            event_id=parsed.provenance.get("event_id")))
        assert calls["window"] == 1
        assert calls["by_id"] == 0

    def test_a_secondary_norad_selector_still_uses_the_window(
        self, monkeypatch, registry, parsed
    ):
        http, calls = self._http(monkeypatch, registry, parsed)
        http.post("/v1/evaluate", json=self._body(
            secondary_norad=parsed.secondary.norad_id))
        assert calls["window"] == 1
        assert calls["by_id"] == 0

    def test_no_selector_still_evaluates_the_worst_conjunction(
        self, monkeypatch, registry, parsed
    ):
        http, calls = self._http(monkeypatch, registry, parsed)
        http.post("/v1/evaluate", json=self._body())
        assert calls["worst"] == 1
        assert calls["by_id"] == 0
        assert calls["window"] == 0

    def test_a_cdm_id_that_resolves_to_nothing_is_404(
        self, monkeypatch, registry, parsed
    ):
        monkeypatch.setattr(server, "fetch_leolabs_conjunction_by_cdm_id",
                            lambda *a, **kw: None)
        resp = TestClient(server.svc).post(
            "/v1/evaluate", json=self._body(cdm_id=_CDM_ID))
        assert resp.status_code == 404
        assert "re-read" in json.dumps(resp.json())


# ---------------------------------------------------------------------------
# 4. No evaluate may block /health
# ---------------------------------------------------------------------------

class TestEvaluateOffTheEventLoop:
    """Driven through the async transport, and timed from before the evaluate.

    Both details are load-bearing, and SCRUM-459 learned them the hard way. A sync
    test client cannot show a blocked event loop at all, and a clock started after
    the blocking call is under way measures nothing -- that is the same mistake that
    made the request log report /health at 0.2 ms while a client waited two minutes.
    """

    @staticmethod
    def _body(norad=36508, **selector):
        return {
            "conjunction_id": "concurrency",
            "satellite": {"sat_id": str(norad), "v_remaining_m_s": 25.0},
            "conjunction": {"primary_norad": str(norad), **selector},
        }

    @pytest.fixture
    def parsed(self):
        return parse_leolabs_cdm(_cdm(_CDM_ID), "L2669")

    @pytest.mark.anyio
    async def test_health_answers_while_a_cdm_id_evaluate_fetch_runs(
        self, monkeypatch, parsed
    ):
        import anyio
        import httpx

        fetch_s = 1.5

        def _slow(cdm_id, primary_norad, **kw):
            time.sleep(fetch_s)
            return parsed

        monkeypatch.setattr(server, "fetch_leolabs_conjunction_by_cdm_id", _slow)
        transport = httpx.ASGITransport(app=server.svc)
        at = {}
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://planner") as http:
            t0 = time.monotonic()
            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    lambda: http.post("/v1/evaluate", json=self._body(
                        cdm_id=_CDM_ID)))
                await anyio.sleep(0.1)
                resp = await http.get("/health")
                at["t"] = time.monotonic() - t0

        assert resp.status_code == 200
        assert at["t"] < fetch_s * 0.6, (
            f"/health answered {at['t']:.2f}s after the evaluate started "
            f"(fetch takes {fetch_s}s) -- the event loop was blocked"
        )

    @pytest.mark.anyio
    async def test_health_answers_while_the_worst_selection_window_pull_runs(
        self, monkeypatch, parsed
    ):
        """The fallback is still the whole window, and is still not allowed to wedge."""
        import anyio
        import httpx

        fetch_s = 1.5

        def _slow(n, **kw):
            time.sleep(fetch_s)
            return parsed

        monkeypatch.setattr(server, "fetch_leolabs_conjunction", _slow)
        transport = httpx.ASGITransport(app=server.svc)
        at = {}
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://planner") as http:
            t0 = time.monotonic()
            async with anyio.create_task_group() as tg:
                tg.start_soon(lambda: http.post("/v1/evaluate", json=self._body()))
                await anyio.sleep(0.1)
                resp = await http.get("/health")
                at["t"] = time.monotonic() - t0

        assert resp.status_code == 200
        assert at["t"] < fetch_s * 0.6, (
            f"/health answered {at['t']:.2f}s into a worst-selection pull"
        )

    @pytest.mark.anyio
    async def test_a_list_still_serves_while_an_evaluate_fetch_runs(
        self, monkeypatch, parsed
    ):
        """One asset's evaluate must not take the live-asset view down."""
        import anyio
        import httpx

        def _slow_eval(cdm_id, primary_norad, **kw):
            time.sleep(0.8)
            return parsed

        monkeypatch.setattr(
            server, "fetch_leolabs_conjunction_by_cdm_id", _slow_eval)
        monkeypatch.setattr(
            server, "fetch_leolabs_conjunction_page",
            lambda *a, **kw: leolabs_runtime.ConjunctionPage(
                rows=[], next_cursor=None, total=0, cdm_total=0, offset=0,
                page_size=100, volume_filters={}),
        )
        transport = httpx.ASGITransport(app=server.svc)
        got = {}
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://planner") as http:
            t0 = time.monotonic()
            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    lambda: http.post("/v1/evaluate", json=self._body(
                        cdm_id=_CDM_ID)))
                await anyio.sleep(0.1)
                r = await http.get(
                    "/v1/leolabs/conjunctions?primary_norad=39453")
                got["status"] = r.status_code
                got["t"] = time.monotonic() - t0

        assert got["status"] == 200
        assert got["t"] < 0.6

    @pytest.mark.anyio
    async def test_past_the_cap_evaluate_sheds_503(self, monkeypatch, parsed):
        import anyio
        import httpx

        def _slow(cdm_id, primary_norad, **kw):
            time.sleep(0.6)
            return parsed

        monkeypatch.setattr(server, "fetch_leolabs_conjunction_by_cdm_id", _slow)
        monkeypatch.setattr(server, "_LIST_MAX_INFLIGHT", 1)
        transport = httpx.ASGITransport(app=server.svc)
        shed = []
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://planner") as http:
            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    lambda: http.post("/v1/evaluate", json=self._body(
                        cdm_id=_CDM_ID)))
                await anyio.sleep(0.05)
                r = await http.post("/v1/evaluate", json=self._body(
                    norad=39451, cdm_id="11111111"))
                shed.append(r)

        resp = shed[0]
        assert resp.status_code == 503
        body = json.dumps(resp.json()).lower()
        assert "busy" in body and "retry" in body
        # A shed evaluate must not look like a decision.
        assert "recommendation" not in resp.json()

    @pytest.mark.anyio
    async def test_the_in_flight_counter_returns_to_zero_after_an_evaluate(
        self, monkeypatch, parsed
    ):
        import httpx

        monkeypatch.setattr(server, "fetch_leolabs_conjunction_by_cdm_id",
                            lambda *a, **kw: parsed)
        transport = httpx.ASGITransport(app=server.svc)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://planner") as http:
            await http.post("/v1/evaluate", json=self._body(cdm_id=_CDM_ID))
        assert server._list_inflight == 0

    @pytest.mark.anyio
    async def test_the_counter_returns_to_zero_when_the_resolve_raises(
        self, monkeypatch
    ):
        import httpx

        def _boom(*a, **kw):
            raise RuntimeError("upstream exploded")

        monkeypatch.setattr(server, "fetch_leolabs_conjunction_by_cdm_id", _boom)
        transport = httpx.ASGITransport(app=server.svc)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://planner") as http:
            resp = await http.post("/v1/evaluate", json=self._body(cdm_id=_CDM_ID))
        assert resp.status_code == 503
        assert server._list_inflight == 0
