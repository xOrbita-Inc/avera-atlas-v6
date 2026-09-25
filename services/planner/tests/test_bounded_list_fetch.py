"""tests/test_bounded_list_fetch.py

SCRUM-459: a dense asset must not take the planner down through the live list.

Two independent failures were behind the incident, and the tests are split the
same way:

1. The cold window pull is unbounded. SWARM B (39451 / L5429) flies in formation
   with SWARM A and C and sees 43x their conjunctions: measured against the live
   API, 75,257 CDMs over 77 pages in 773 s, against a ui-to-planner read timeout of
   60 s. It was never going to finish. It is now bounded by a deadline and a cap,
   and a bounded pull that stopped early says so.

2. The fetch ran ON the event loop. Both list routes are `async def` and called the
   blocking fetch directly, and the planner runs one uvicorn worker, so a single
   fetch blocked /health and every other asset for its whole duration. Measured
   from a client: /health did not answer at all during one SWARM B fetch. The
   request log said 0.2 ms, because the middleware times the handler and not the
   wait for a blocked loop.

The safety direction is the same one as SCRUM-458: an incomplete list must never
read as the full, clear picture. CDMs arrive in LeoLabs order, not Pc order, so a
truncated prefix can be missing the worst conjunction. A short list that looks
complete reads as a quiet sky.

All offline. The fake client never touches the network.

Run from repo root:
    python -m pytest services/planner/tests/test_bounded_list_fetch.py -v
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import server
from common import leolabs_runtime
from common.leolabs_asset_map import AssetRegistry
from common.leolabs_client import CdmPullReport, LeoLabsClient

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"
_OBJECTS = [{"catalogNumber": "L2669", "noradCatalogNumber": 36508,
             "name": "CRYOSAT 2"}]
_NOW = datetime(2026, 8, 25, tzinfo=timezone.utc)


@pytest.fixture
def cdm() -> dict:
    return json.loads(_FIXTURE.read_text())


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


# ---------------------------------------------------------------------------
# A fake LeoLabs HTTP layer, so the real paginator runs
# ---------------------------------------------------------------------------

def _paged_client(cdm: dict, *, pages: int, per_page: int,
                  page_delay_s: float = 0.0, total: int = None) -> LeoLabsClient:
    """A real LeoLabsClient with only _request faked.

    The real _paginate, the real nextToken handling and the real bounding all run;
    only the HTTP call is replaced. A fake that stubbed search_conjunction_cdms
    instead would test nothing about the thing this ticket changed.
    """
    client = LeoLabsClient()
    made = {"requests": 0}
    grand_total = total if total is not None else pages * per_page

    def _fake_request(method, path, params=None, **kw):
        made["requests"] += 1
        if page_delay_s:
            time.sleep(page_delay_s)
        page_index = int(params.get("token", 0) or 0) if params else 0
        rows = []
        for i in range(per_page):
            row = copy.deepcopy(cdm)
            n = page_index * per_page + i
            row["COMMENT_EVENT_ID"] = 1_000_000 + n
            row["COMMENT_ID"] = f"cdm-{n}"
            # descending Pc, so the worst conjunction is on the FIRST page and a
            # later test can show a truncated pull keeping or losing it
            row["COLLISION_PROBABILITY"] = 1.0e-4 / (n + 1)
            rows.append(row)
        body = {"cdms": rows, "total": grand_total}
        if page_index + 1 < pages:
            body["nextToken"] = str(page_index + 1)
        return body

    client._request = _fake_request                     # type: ignore[assignment]
    client._fake_requests = made                        # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# 1. The deadline
# ---------------------------------------------------------------------------

class TestDeadline:
    def test_a_slow_window_returns_within_the_budget_not_at_the_timeout(
        self, cdm, registry
    ):
        """The SWARM B shape: pages so slow the window cannot be finished."""
        client = _paged_client(cdm, pages=50, per_page=5, page_delay_s=0.05)
        started = time.monotonic()
        page = leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
            deadline_s=0.2, max_cdms=100000,
        )
        elapsed = time.monotonic() - started

        assert page.complete is False
        assert page.truncation_reason == "deadline"
        # Bounded by the deadline plus the one page already under way, not by the
        # 50 pages it would have taken to finish.
        assert elapsed < 0.2 + 0.05 * 3

    def test_the_deadline_is_checked_before_issuing_the_next_page(
        self, cdm, registry
    ):
        """Otherwise the overshoot is a whole request nobody had budget for."""
        client = _paged_client(cdm, pages=50, per_page=5, page_delay_s=0.05)
        leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
            deadline_s=0.12, max_cdms=100000,
        )
        # ~0.12 s of budget at 0.05 s a page is three or four requests, not fifty.
        assert client._fake_requests["requests"] <= 5

    def test_a_window_that_fits_the_budget_is_complete(self, cdm, registry):
        client = _paged_client(cdm, pages=2, per_page=5)
        page = leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
        )
        assert page.complete is True
        assert page.truncation_reason is None
        assert page.total == 10


# ---------------------------------------------------------------------------
# 2. The cap
# ---------------------------------------------------------------------------

class TestCap:
    def test_an_oversized_window_is_capped_and_incomplete(self, cdm, registry):
        client = _paged_client(cdm, pages=20, per_page=10)
        page = leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
            deadline_s=600.0, max_cdms=25, page_size=100,
        )
        assert page.complete is False
        assert page.truncation_reason == "cap"
        assert page.pulled_cdms == 25
        assert page.total == 25            # events from the prefix, not the window

    def test_the_cap_does_not_fire_on_a_window_that_fits(self, cdm, registry):
        client = _paged_client(cdm, pages=2, per_page=5)
        page = leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
            deadline_s=600.0, max_cdms=25,
        )
        assert page.complete is True
        assert page.pulled_cdms == 10

    def test_the_window_total_is_reported_so_partial_has_a_denominator(
        self, cdm, registry
    ):
        """"3,000 of 75,000" is actionable; "partial" alone is not."""
        client = _paged_client(cdm, pages=20, per_page=10, total=200)
        page = leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
            deadline_s=600.0, max_cdms=25,
        )
        assert page.window_cdm_total == 200
        assert page.pulled_cdms == 25


# ---------------------------------------------------------------------------
# 3. Incomplete is never presented as complete
# ---------------------------------------------------------------------------

class TestHonesty:
    def _client_for_route(self, monkeypatch, client, registry):
        monkeypatch.setattr(leolabs_runtime, "get_client", lambda: client)
        monkeypatch.setattr(leolabs_runtime, "get_registry", lambda c=None: registry)
        return TestClient(server.svc)

    def test_the_incomplete_flag_rides_through_to_the_json_body(
        self, monkeypatch, cdm, registry
    ):
        client = _paged_client(cdm, pages=20, per_page=10, total=200)
        monkeypatch.setattr(leolabs_runtime, "_LIST_FETCH_MAX_CDMS", 25)
        http = self._client_for_route(monkeypatch, client, registry)
        body = http.get("/v1/leolabs/conjunctions?primary_norad=36508").json()

        assert body["complete"] is False
        assert body["partial"] is True
        assert body["truncation"]["truncated_by"] == "cap"
        assert body["truncation"]["cdms_pulled"] == 25
        assert body["truncation"]["cdms_in_window"] == 200

    def test_a_complete_listing_says_so_and_carries_no_truncation_block(
        self, monkeypatch, cdm, registry
    ):
        client = _paged_client(cdm, pages=2, per_page=5)
        http = self._client_for_route(monkeypatch, client, registry)
        body = http.get("/v1/leolabs/conjunctions?primary_norad=36508").json()

        assert body["complete"] is True
        assert body["partial"] is False
        assert body["truncation"] is None

    def test_an_incomplete_listing_still_returns_200_with_the_rows_it_has(
        self, monkeypatch, cdm, registry
    ):
        """A partial view is useful; it just must not claim to be whole."""
        client = _paged_client(cdm, pages=20, per_page=10)
        monkeypatch.setattr(leolabs_runtime, "_LIST_FETCH_MAX_CDMS", 25)
        http = self._client_for_route(monkeypatch, client, registry)
        resp = http.get("/v1/leolabs/conjunctions?primary_norad=36508")

        assert resp.status_code == 200
        assert resp.json()["count"] >= 1
        assert resp.json()["complete"] is False

    def test_the_globe_response_reports_partial_too(
        self, monkeypatch, cdm, registry
    ):
        """An uncluttered globe reads as a quiet sky, so it must say partial."""
        client = _paged_client(cdm, pages=20, per_page=10, total=200)
        monkeypatch.setattr(leolabs_runtime, "_LIST_FETCH_MAX_CDMS", 25)
        monkeypatch.setattr(
            server, "latest_state_km",
            lambda *a, **k: ([6878.0, 0.0, 0.0], [0.0, 7.6, 0.0], "2026-08-25T00:00:00Z"),
        )
        http = self._client_for_route(monkeypatch, client, registry)
        body = http.get("/v1/leolabs/orbits?primary_norad=36508").json()

        assert body["complete"] is False
        assert body["partial"] is True
        assert body["truncation"]["truncated_by"] == "cap"


# ---------------------------------------------------------------------------
# 4. The completed case is unchanged
# ---------------------------------------------------------------------------

class TestCompletedCaseUnchanged:
    def test_worst_pc_first_survives(self, cdm, registry):
        """The fixture client emits descending Pc, so page one must lead with it."""
        client = _paged_client(cdm, pages=3, per_page=4)
        page = leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW, page_size=12,
        )
        pcs = [p.cdm_collision_probability for p in page.rows]
        assert pcs == sorted(pcs, reverse=True)
        assert page.complete is True

    def test_the_event_total_is_exact_on_a_complete_pull(self, cdm, registry):
        client = _paged_client(cdm, pages=3, per_page=4)
        page = leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
        )
        assert page.total == 12
        assert page.cdm_total == 12

    def test_reissues_of_one_event_still_collapse(self, cdm, registry):
        """One event per reissue, unchanged by the bounding."""
        client = LeoLabsClient()
        a, b = copy.deepcopy(cdm), copy.deepcopy(cdm)
        a["COMMENT_EVENT_ID"] = b["COMMENT_EVENT_ID"] = 7777
        a["COMMENT_ID"], b["COMMENT_ID"] = "cdm-a", "cdm-b"
        client._request = lambda *args, **kw: {"cdms": [a, b], "total": 2}
        page = leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
        )
        assert page.total == 1
        assert page.cdm_total == 2
        assert page.complete is True

    def test_an_unbounded_pull_reports_complete_and_asks_for_no_bounds(
        self, cdm, registry
    ):
        """The evaluate path must keep its whole-window pull."""
        client = MagicMock()
        client.search_conjunction_cdms.return_value = [cdm]
        leolabs_runtime.fetch_leolabs_conjunctions(
            36508, client=client, registry=registry, now=_NOW)
        kwargs = client.search_conjunction_cdms.call_args.kwargs
        assert kwargs.get("deadline_s") is None
        assert kwargs.get("max_cdms") is None


# ---------------------------------------------------------------------------
# 5. The client's own bounding contract
# ---------------------------------------------------------------------------

class TestClientBounding:
    def test_an_unbounded_search_reports_complete(self, cdm):
        client = _paged_client(cdm, pages=2, per_page=3)
        report = CdmPullReport()
        items = client.search_conjunction_cdms(object1="L2669", report=report)
        assert len(items) == 6
        assert report.complete is True
        assert report.reason is None

    def test_a_capped_search_reports_incomplete(self, cdm):
        client = _paged_client(cdm, pages=5, per_page=3, total=15)
        report = CdmPullReport()
        items = client.search_conjunction_cdms(
            object1="L2669", max_cdms=4, report=report)
        assert len(items) == 4
        assert report.complete is False
        assert report.reason == "cap"
        assert report.window_total == 15

    def test_a_report_is_optional(self, cdm):
        """Bounding without a report must not raise; it just cannot be observed."""
        client = _paged_client(cdm, pages=5, per_page=3)
        assert len(client.search_conjunction_cdms(object1="L2669", max_cdms=4)) == 4

    def test_a_default_report_means_whole_window(self):
        """So a test double that returns its fixture is never read as truncated."""
        assert CdmPullReport().complete is True
        assert CdmPullReport().reason is None


# ---------------------------------------------------------------------------
# 6. The cache carries completeness
# ---------------------------------------------------------------------------

class TestCacheInteraction:
    def test_a_repeat_listing_is_still_a_cache_hit(self, cdm, registry):
        """SCRUM-450 still works; this ticket did not add a second cache."""
        client = _paged_client(cdm, pages=2, per_page=5)
        for _ in range(3):
            leolabs_runtime.fetch_leolabs_conjunction_page(
                36508, client=client, registry=registry, now=_NOW)
        # 2 pages for the one cold pull, and nothing after it.
        assert client._fake_requests["requests"] == 2

    def test_an_incomplete_result_is_cached_as_incomplete(self, cdm, registry):
        """A hit must not launder a truncated window into a complete one."""
        client = _paged_client(cdm, pages=20, per_page=10)
        first = leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
            deadline_s=600.0, max_cdms=25)
        second = leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
            deadline_s=600.0, max_cdms=25)
        assert first.complete is False
        assert second.complete is False

    def test_reset_caches_clears_the_window_entries(self, cdm, registry):
        client = _paged_client(cdm, pages=1, per_page=5)
        leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW)
        before = client._fake_requests["requests"]
        leolabs_runtime.reset_caches()
        leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW)
        assert client._fake_requests["requests"] == before + 1

    def test_a_cached_complete_window_is_served_instead_of_a_fresh_truncated_pull(
        self, cdm, registry
    ):
        """A complete window in hand beats re-pulling and truncating.

        Keyed with the module's own default window, so this fails if the key
        composition changes rather than silently becoming a miss that still passes.
        """
        key = leolabs_runtime._cdm_cache_key(
            "L2669",
            leolabs_runtime._DEFAULT_LOOKBACK_DAYS,
            leolabs_runtime._DEFAULT_LOOKAHEAD_DAYS,
            {},
            True,                                    # the bounded (listing) entry
        )
        whole = leolabs_runtime.WindowPull(cdms=[cdm], complete=True)
        leolabs_runtime._cdm_cache[key] = (time.monotonic(), whole)

        client = _paged_client(cdm, pages=20, per_page=10)
        page = leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
            volume_filters={}, deadline_s=600.0, max_cdms=25)
        assert page.complete is True
        assert client._fake_requests["requests"] == 0

    def test_the_store_guard_keeps_a_fresh_complete_entry(self, cdm):
        """The race guard, exercised where it lives rather than through a race.

        Two callers can both miss a cold key, because the fetch deliberately happens
        outside the lock. If one returns a whole window and the other a truncated
        one, the outcome must not depend on which finishes last. Unreachable from a
        single-threaded call -- the second caller would hit the cache -- so the store
        path is driven directly.
        """
        key = leolabs_runtime._cdm_cache_key("L2669", 0, 7, {}, True)
        whole = leolabs_runtime.WindowPull(cdms=[cdm, cdm], complete=True)
        leolabs_runtime._cdm_cache[key] = (time.monotonic(), whole)

        client = _paged_client(cdm, pages=20, per_page=10)
        # A truncated pull for the same key, stored as if it had raced the above.
        truncated = leolabs_runtime.WindowPull(
            cdms=[cdm], complete=False, reason="cap")
        leolabs_runtime._store_window(key, truncated)

        stored = leolabs_runtime._cdm_cache[key][1]
        assert stored.complete is True
        assert stored.pulled == 2

        # And the other direction: a complete result does replace an incomplete one.
        leolabs_runtime._cdm_cache[key] = (time.monotonic(), truncated)
        leolabs_runtime._store_window(key, whole)
        assert leolabs_runtime._cdm_cache[key][1].complete is True


# ---------------------------------------------------------------------------
# 7. Concurrency: a heavy fetch must not wedge the planner
# ---------------------------------------------------------------------------

class TestConcurrency:
    """The half of the incident the deadline alone does not fix.

    Both list routes are `async def` and called the blocking fetch directly. On one
    uvicorn worker that is one event loop, so a single fetch blocked /health and
    every other asset for its whole duration -- measured live, /health did not answer
    at all during one SWARM B fetch, while the request log said 0.2 ms because the
    middleware times the handler and not the wait.

    These drive the ASGI app through httpx's async transport so the event loop is
    real and a blocking call would actually block it, which a sync TestClient cannot
    show.
    """

    @staticmethod
    def _slow_page(delay_s: float):
        def _fetch(*args, **kwargs):
            time.sleep(delay_s)
            return leolabs_runtime.ConjunctionPage(
                rows=[], next_cursor=None, total=0, cdm_total=0,
                offset=0, page_size=100, volume_filters={},
            )
        return _fetch

    @pytest.mark.anyio
    async def test_health_answers_while_a_heavy_fetch_runs(self, monkeypatch):
        import anyio
        import httpx

        fetch_s = 1.5
        monkeypatch.setattr(
            server, "fetch_leolabs_conjunction_page", self._slow_page(fetch_s))
        transport = httpx.ASGITransport(app=server.svc)

        health_at = {}
        # Timed from BEFORE the fetch starts, deliberately. Timing from after it is
        # under way is the very mistake that made this bug look benign: the request
        # log showed /health at 0.2 ms during a SWARM B fetch because the middleware
        # starts its clock when the handler runs, which on a blocked loop is after
        # the block has already cleared. A clock started inside the blocked window
        # measures nothing. An earlier draft of this test made the same error and
        # passed against the blocking code.
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://planner") as http:
            t0 = time.monotonic()
            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    http.get, "/v1/leolabs/conjunctions?primary_norad=36508")
                await anyio.sleep(0.1)          # let the fetch get under way
                resp = await http.get("/health")
                health_at["t"] = time.monotonic() - t0

        assert resp.status_code == 200
        # /health must come back while the fetch is still running. On a blocked loop
        # it cannot answer until the fetch finishes, so it would land at or after
        # fetch_s.
        assert health_at["t"] < fetch_s * 0.6, (
            f"/health answered {health_at['t']:.2f}s after the fetch started "
            f"(fetch takes {fetch_s}s) -- the event loop was blocked"
        )

    @pytest.mark.anyio
    async def test_another_asset_still_lists_while_one_is_fetching(
        self, monkeypatch
    ):
        """One dense asset must not take the whole live-asset view down."""
        import anyio
        import httpx

        calls = {"n": 0}

        def _fetch(*args, **kwargs):
            calls["n"] += 1
            time.sleep(0.5)
            return leolabs_runtime.ConjunctionPage(
                rows=[], next_cursor=None, total=0, cdm_total=0,
                offset=0, page_size=100, volume_filters={},
            )

        monkeypatch.setattr(server, "fetch_leolabs_conjunction_page", _fetch)
        transport = httpx.ASGITransport(app=server.svc)
        results = []
        fetch_s = 0.5
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://planner") as http:
            async def _list(norad):
                r = await http.get(
                    f"/v1/leolabs/conjunctions?primary_norad={norad}")
                results.append((norad, r.status_code))

            t0 = time.monotonic()
            async with anyio.create_task_group() as tg:
                tg.start_soon(_list, 36508)
                tg.start_soon(_list, 39451)
            elapsed = time.monotonic() - t0

        assert sorted(results) == [(36508, 200), (39451, 200)]
        assert calls["n"] == 2
        # Both returning 200 is not the property -- they would do that serialised
        # too, just slowly. The property is that they OVERLAP: two fetches of
        # fetch_s finishing in appreciably less than 2 * fetch_s.
        assert elapsed < fetch_s * 1.8, (
            f"two {fetch_s}s fetches took {elapsed:.2f}s -- they serialised"
        )

    @pytest.mark.anyio
    async def test_past_the_cap_the_endpoint_sheds_503_rather_than_queueing(
        self, monkeypatch
    ):
        """Queueing behind two 25 s pulls spends a budget the request has not got."""
        import anyio
        import httpx

        monkeypatch.setattr(
            server, "fetch_leolabs_conjunction_page", self._slow_page(0.6))
        monkeypatch.setattr(server, "_LIST_MAX_INFLIGHT", 2)
        transport = httpx.ASGITransport(app=server.svc)
        codes = []
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://planner") as http:
            async def _list(norad):
                r = await http.get(
                    f"/v1/leolabs/conjunctions?primary_norad={norad}")
                codes.append(r.status_code)

            async with anyio.create_task_group() as tg:
                for norad in (36508, 39451, 39452, 39453):
                    tg.start_soon(_list, norad)
                    await anyio.sleep(0.02)

        assert codes.count(503) >= 1, f"nothing shed: {codes}"
        assert codes.count(200) == 2, f"the cap should admit exactly two: {codes}"

    @pytest.mark.anyio
    async def test_a_shed_request_says_busy_not_empty(self, monkeypatch):
        """An empty sky and a refused fetch must never look the same."""
        import anyio
        import httpx

        monkeypatch.setattr(
            server, "fetch_leolabs_conjunction_page", self._slow_page(0.6))
        monkeypatch.setattr(server, "_LIST_MAX_INFLIGHT", 1)
        transport = httpx.ASGITransport(app=server.svc)
        shed = []
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://planner") as http:
            async def _first():
                await http.get("/v1/leolabs/conjunctions?primary_norad=36508")

            async with anyio.create_task_group() as tg:
                tg.start_soon(_first)
                await anyio.sleep(0.05)
                r = await http.get(
                    "/v1/leolabs/conjunctions?primary_norad=39451")
                shed.append(r)

        resp = shed[0]
        assert resp.status_code == 503
        body = json.dumps(resp.json()).lower()
        assert "busy" in body
        assert "retry" in body
        # and it carries no conjunction list that could be read as a quiet sky
        assert "conjunctions" not in resp.json()

    @pytest.mark.anyio
    async def test_the_in_flight_counter_returns_to_zero(self, monkeypatch):
        """A leak would wedge the endpoint permanently at the cap."""
        import httpx

        monkeypatch.setattr(
            server, "fetch_leolabs_conjunction_page", self._slow_page(0.01))
        transport = httpx.ASGITransport(app=server.svc)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://planner") as http:
            await http.get("/v1/leolabs/conjunctions?primary_norad=36508")
        assert server._list_inflight == 0

    @pytest.mark.anyio
    async def test_the_counter_returns_to_zero_when_the_fetch_raises(
        self, monkeypatch
    ):
        import httpx

        def _boom(*args, **kwargs):
            raise RuntimeError("upstream exploded")

        monkeypatch.setattr(server, "fetch_leolabs_conjunction_page", _boom)
        transport = httpx.ASGITransport(app=server.svc)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://planner") as http:
            resp = await http.get(
                "/v1/leolabs/conjunctions?primary_norad=36508")
        assert resp.status_code == 503
        assert server._list_inflight == 0


# ---------------------------------------------------------------------------
# 8. The window total, which LeoLabs sends as a string
# ---------------------------------------------------------------------------

class TestWindowTotalParsing:
    """SCRUM-459 found this by needing the number for the partial-view denominator.

    LeoLabs returns `total` on a paginated search as a STRING -- "22008", confirmed
    against the live API. The client checked isinstance(value, (int, float)), which a
    string never satisfies, so `total` stayed None on every live response and
    SCRUM-438's retrieved-versus-total cross-check -- added because silent truncation
    was the failure it feared -- never ran at all. Its silence proved nothing.
    """

    def test_a_string_total_is_parsed(self, cdm):
        client = LeoLabsClient()
        client._request = lambda *a, **kw: {"cdms": [cdm], "total": "4321"}
        report = CdmPullReport()
        client.search_conjunction_cdms(
            object1="L2669", max_cdms=1, report=report)
        assert report.window_total == 4321

    def test_an_integer_total_still_works(self, cdm):
        client = LeoLabsClient()
        client._request = lambda *a, **kw: {"cdms": [cdm], "total": 4321}
        report = CdmPullReport()
        client.search_conjunction_cdms(
            object1="L2669", max_cdms=1, report=report)
        assert report.window_total == 4321

    @pytest.mark.parametrize("value", [None, "", "not-a-number", "12.x", {}, []])
    def test_an_unparseable_total_is_none_not_a_crash(self, cdm, value):
        client = LeoLabsClient()
        client._request = lambda *a, **kw: {"cdms": [cdm], "total": value}
        report = CdmPullReport()
        client.search_conjunction_cdms(
            object1="L2669", max_cdms=1, report=report)
        assert report.window_total is None

    def test_a_boolean_total_is_rejected(self, cdm):
        """True is an int subclass and would otherwise read as a total of 1."""
        from common.leolabs_client import _as_int
        assert _as_int(True) is None
        assert _as_int(False) is None

    def test_the_truncation_denominator_reaches_the_response(
        self, monkeypatch, cdm, registry
    ):
        """The point of fixing it: "3,000 of 75,257" instead of "3,000 of null"."""
        client = _paged_client(cdm, pages=30, per_page=10, total=300)
        # the fake sends an int; assert the string path end to end too
        raw = client._request

        def _string_total(*a, **kw):
            body = raw(*a, **kw)
            body["total"] = str(body["total"])
            return body

        client._request = _string_total
        monkeypatch.setattr(leolabs_runtime, "get_client", lambda: client)
        monkeypatch.setattr(leolabs_runtime, "get_registry", lambda c=None: registry)
        monkeypatch.setattr(leolabs_runtime, "_LIST_FETCH_MAX_CDMS", 25)
        body = TestClient(server.svc).get(
            "/v1/leolabs/conjunctions?primary_norad=36508").json()
        assert body["truncation"]["cdms_in_window"] == 300
        assert body["truncation"]["cdms_pulled"] == 25
