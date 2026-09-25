"""tests/test_leolabs_conjunction_list.py

SCRUM-422: listing an asset's LeoLabs conjunctions, and scoring a chosen one.

Covers the list fetch's ordering and empty handling, the row shape the SCRUM-421
table renders, the GET /v1/leolabs/conjunctions endpoint, and the evaluate
selector that makes a row individually clickable.

All offline: the client is mocked and the feature flag is patched, same as
test_leolabs_runtime.py. CI needs no credential.

Run from repo root:
    python -m pytest services/planner/tests/test_leolabs_conjunction_list.py -v
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import server
from common import leolabs_runtime
from common.leolabs_asset_map import AssetRegistry

from common.leolabs_conjunction_list import (
    PC_AMBER_THRESHOLD,
    PC_RED_THRESHOLD,
    SELECTOR_FIELDS,
    conjunction_row,
    dedupe_by_event,
    dedupe_raw_by_event,
    event_key,
    pc_to_risk_level,
    raw_event_key,
    select_conjunction,
    selector_from_conjunction_block,
)
from common.leolabs_runtime import (
    LeoLabsCursorError,
    LeoLabsRuntimeError,
    fetch_leolabs_conjunction,
    fetch_leolabs_conjunction_page,
    fetch_leolabs_conjunctions,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"
_OBJECTS = [{"catalogNumber": "L2669", "noradCatalogNumber": 36508, "name": "CRYOSAT 2"}]
_NOW = datetime(2026, 8, 25, tzinfo=timezone.utc)


@pytest.fixture
def cdm() -> dict:
    return json.loads(_FIXTURE.read_text())


@pytest.fixture
def registry() -> AssetRegistry:
    return AssetRegistry.from_objects(_OBJECTS)


@pytest.fixture(autouse=True)
def _clean_caches():
    leolabs_runtime.reset_caches()
    yield
    leolabs_runtime.reset_caches()


def _variant(base: dict, *, pc, tca, norad, designator, cdm_id, event_id, name=None):
    """One CDM for a distinct secondary, varying only what a row keys on."""
    out = copy.deepcopy(base)
    out["COLLISION_PROBABILITY"] = pc
    out["TCA_ISO"] = tca
    out["TCA"] = tca.replace("T", " ").replace("Z", "")
    out["SAT2_COMMENT_NORAD_ID"] = norad
    out["SAT2_OBJECT_DESIGNATOR"] = designator
    out["SAT2_OBJECT_NAME"] = name or f"DEBRIS {norad}"
    out["COMMENT_ID"] = cdm_id
    out["COMMENT_EVENT_ID"] = event_id
    out["SAT2_COMMENT_OBJECT_URL"] = f"https://platform.leolabs.space/catalog/{designator}"
    out["SAT2_COMMENT_STATE_URL"] = (
        f"https://platform.leolabs.space/catalog/{designator}/states/1"
    )
    return out


@pytest.fixture
def three_cdms(cdm) -> list:
    """Three distinct events, deliberately NOT in risk order in the raw list."""
    return [
        _variant(cdm, pc=1.0e-6, tca="2026-08-27T00:00:00Z", norad=100001,
                 designator="L100001", cdm_id=1001, event_id=9001),
        _variant(cdm, pc=5.0e-4, tca="2026-08-28T00:00:00Z", norad=100002,
                 designator="L100002", cdm_id=1002, event_id=9002),
        _variant(cdm, pc=2.0e-5, tca="2026-08-26T00:00:00Z", norad=100003,
                 designator="L100003", cdm_id=1003, event_id=9003),
    ]


def _client_for(cdms):
    client = MagicMock()
    client.search_conjunction_cdms.return_value = cdms
    return client


# ---------------------------------------------------------------------------
# fetch_leolabs_conjunctions: ordering, emptiness, and the refactor invariant
# ---------------------------------------------------------------------------


class TestListFetch:
    def test_returns_all_scorable_cdms(self, three_cdms, registry):
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for(three_cdms), registry=registry, now=_NOW
        )

        assert len(parsed) == 3

    def test_ordered_highest_pc_first(self, three_cdms, registry):
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for(three_cdms), registry=registry, now=_NOW
        )

        pcs = [p.cdm_collision_probability for p in parsed]
        assert pcs == sorted(pcs, reverse=True)
        assert pcs == [5.0e-4, 2.0e-5, 1.0e-6]

    def test_earliest_tca_breaks_a_pc_tie(self, cdm, registry):
        """Mirrors the UDL selection order: min by (-pc, tca)."""
        later = _variant(cdm, pc=1.0e-5, tca="2026-08-29T00:00:00Z", norad=200001,
                         designator="L200001", cdm_id=2001, event_id=8001)
        earlier = _variant(cdm, pc=1.0e-5, tca="2026-08-26T00:00:00Z", norad=200002,
                           designator="L200002", cdm_id=2002, event_id=8002)
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for([later, earlier]), registry=registry, now=_NOW
        )

        assert [p.secondary.norad_id for p in parsed] == [200002, 200001]

    def test_empty_window_returns_an_empty_list_not_an_error(self, registry):
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for([]), registry=registry, now=_NOW
        )

        assert parsed == []

    def test_guard_failing_cdms_are_skipped_not_fatal(self, three_cdms, registry):
        bad = copy.deepcopy(three_cdms[1])          # the highest-Pc one
        bad["SAT1_COVARIANCE_METHOD"] = "DEFAULT"   # fails the CALCULATED guard
        cdms = [three_cdms[0], bad, three_cdms[2]]
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for(cdms), registry=registry, now=_NOW
        )

        assert [p.cdm_collision_probability for p in parsed] == [2.0e-5, 1.0e-6]

    def test_unsubscribed_norad_raises(self, three_cdms, registry):
        with pytest.raises(LeoLabsRuntimeError):
            fetch_leolabs_conjunctions(
                99999, client=_client_for(three_cdms), registry=registry, now=_NOW
            )

    def test_singular_fetch_is_the_first_of_the_list(self, three_cdms, registry):
        """The refactor invariant: one code path, and the top has not moved."""
        top = fetch_leolabs_conjunction(
            36508, client=_client_for(three_cdms), registry=registry, now=_NOW
        )
        listed = fetch_leolabs_conjunctions(
            36508, client=_client_for(three_cdms), registry=registry, now=_NOW
        )

        assert top.provenance["cdm_id"] == listed[0].provenance["cdm_id"]
        assert top.cdm_collision_probability == 5.0e-4

    def test_singular_fetch_still_stops_at_the_first_scorable_cdm(
        self, three_cdms, registry
    ):
        """Laziness is preserved: the singular path does not parse the tail.

        Asserted through the registry, which the parse step calls once per CDM
        it actually attempts.
        """
        spy = MagicMock(wraps=registry)
        fetch_leolabs_conjunction(
            36508, client=_client_for(three_cdms), registry=spy, now=_NOW
        )

        assert spy.resolve_our_catalog_id.call_count == 1


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


class TestRowShape:
    def test_a_row_carries_every_field_the_table_needs(self, three_cdms, registry):
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for(three_cdms), registry=registry, now=_NOW
        )
        row = conjunction_row(parsed[0], now=_NOW)

        assert set(row) == {
            "cdm_id", "event_id", "secondary_norad",
            "secondary_designator", "secondary_name",
            "primary_norad", "primary_designator", "primary_name",
            "tca_utc", "time_to_tca_s", "time_to_tca_min",
            "miss_distance_m", "miss_distance_km",
            "pc", "pc_display", "risk_level",
            # SCRUM-445: the RTN relative state the encounter viz draws from, so
            # selecting a row needs no round-trip.
            "relative_position_rtn_m", "relative_velocity_rtn_m_s",
            "relative_state_source",
            "pc_source", "pc_method", "covariance_source", "source",
        }

    def test_a_row_carries_a_usable_selector(self, three_cdms, registry):
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for(three_cdms), registry=registry, now=_NOW
        )
        row = conjunction_row(parsed[0], now=_NOW)

        for field in SELECTOR_FIELDS:
            assert row[field] is not None
        assert row["secondary_norad"] == 100002

    def test_the_row_pc_is_labelled_as_leolabs_not_ours(self, three_cdms, registry):
        """The parser sets pc_precomputed=None on purpose: the CDM Pc is a
        cross-check, not the planner's Pc. A row must say so."""
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for(three_cdms), registry=registry, now=_NOW
        )
        row = conjunction_row(parsed[0], now=_NOW)

        assert row["pc"] == 5.0e-4
        assert row["pc_source"] == "leolabs_cdm"

    def test_time_to_tca_is_measured_from_the_supplied_clock(
        self, three_cdms, registry
    ):
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for(three_cdms), registry=registry, now=_NOW
        )
        # The top row's TCA is 2026-08-28, three days after _NOW.
        row = conjunction_row(parsed[0], now=_NOW)

        assert row["time_to_tca_s"] == pytest.approx(3 * 86400.0)
        assert row["time_to_tca_min"] == pytest.approx(3 * 1440.0)

    def test_every_row_in_one_listing_shares_one_clock(self, three_cdms, registry):
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for(three_cdms), registry=registry, now=_NOW
        )
        rows = [conjunction_row(p, now=_NOW) for p in parsed]

        for row in rows:
            tca = datetime.fromisoformat(row["tca_utc"].replace("Z", "+00:00"))
            assert row["time_to_tca_s"] == pytest.approx(
                (tca - _NOW).total_seconds(), abs=1e-3
            )


class TestRiskLevel:
    def test_the_bands_are_the_locked_action_and_watch_lines(self):
        # RED at Pc_action (1e-4), AMBER at pc_monitor_threshold (1e-5).
        assert PC_RED_THRESHOLD == 1.0e-4
        assert PC_AMBER_THRESHOLD == 1.0e-5

    @pytest.mark.parametrize(
        "pc, level",
        [
            (1.0e-3, "RED"),
            (1.0e-4, "RED"),       # at the action line is RED, not AMBER
            (9.9e-5, "AMBER"),
            (1.0e-5, "AMBER"),     # at the watch line is AMBER, not GREEN
            (9.9e-6, "GREEN"),
            (1.0e-7, "GREEN"),
            (9.9e-8, "NOMINAL"),
            (0.0, "NOMINAL"),
        ],
    )
    def test_bands(self, pc, level):
        assert pc_to_risk_level(pc) == level

    def test_a_cdm_with_no_pc_is_reported_as_absent_not_as_zero(self, cdm, registry):
        no_pc = copy.deepcopy(cdm)
        no_pc["COLLISION_PROBABILITY"] = None
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for([no_pc]), registry=registry, now=_NOW
        )
        row = conjunction_row(parsed[0], now=_NOW)

        assert row["pc"] is None
        assert row["pc_display"] == "n/a"
        assert row["risk_level"] == "NOMINAL"


class TestDedupeByEvent:
    def test_reissued_cdms_for_one_event_collapse_to_one_row(self, cdm, registry):
        """LeoLabs reissues a CDM per event as the solution refines. A table
        showing the same secondary five times is unusable."""
        first = _variant(cdm, pc=3.0e-4, tca="2026-08-28T00:00:00Z", norad=300001,
                         designator="L300001", cdm_id=3001, event_id=7001)
        reissue = _variant(cdm, pc=1.0e-4, tca="2026-08-28T00:00:10Z", norad=300001,
                           designator="L300001", cdm_id=3002, event_id=7001)
        other = _variant(cdm, pc=2.0e-4, tca="2026-08-29T00:00:00Z", norad=300002,
                         designator="L300002", cdm_id=3003, event_id=7002)
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for([first, reissue, other]),
            registry=registry, now=_NOW,
        )

        assert len(parsed) == 3          # the fetch is faithful: one per CDM
        events = dedupe_by_event(parsed)
        assert len(events) == 2          # the table is one per event

    def test_dedupe_keeps_the_highest_risk_cdm_for_an_event(self, cdm, registry):
        low = _variant(cdm, pc=1.0e-6, tca="2026-08-28T00:00:00Z", norad=300001,
                       designator="L300001", cdm_id=3001, event_id=7001)
        high = _variant(cdm, pc=9.0e-4, tca="2026-08-28T00:00:10Z", norad=300001,
                        designator="L300001", cdm_id=3002, event_id=7001)
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for([low, high]), registry=registry, now=_NOW
        )
        events = dedupe_by_event(parsed)

        assert len(events) == 1
        assert events[0].provenance["cdm_id"] == 3002

    def test_dedupe_preserves_risk_order(self, three_cdms, registry):
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for(three_cdms), registry=registry, now=_NOW
        )
        events = dedupe_by_event(parsed)

        assert [p.cdm_collision_probability for p in events] == [5.0e-4, 2.0e-5, 1.0e-6]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class TestRawEventKey:
    """SCRUM-445: the raw-CDM dedupe key must not drift from the parsed one."""

    def test_raw_and_parsed_event_keys_agree(self, three_cdms, registry):
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for(three_cdms), registry=registry, now=_NOW
        )
        # Same CDMs, same order: fetch preserves risk order and these are distinct
        # events, so the two key functions are compared pairwise on one CDM each.
        raw_sorted = sorted(three_cdms, key=lambda c: -c["COLLISION_PROBABILITY"])
        assert [raw_event_key(c) for c in raw_sorted] == [
            event_key(p) for p in parsed
        ]

    def test_the_key_falls_back_the_same_way_when_ids_are_missing(self, cdm):
        no_event = {**cdm, "COMMENT_EVENT_ID": None}
        assert raw_event_key(no_event) == f"cdm_id:{cdm['COMMENT_ID']}"
        bare = {**cdm, "COMMENT_EVENT_ID": None, "COMMENT_ID": None}
        assert raw_event_key(bare) == (
            f"secondary:{cdm['SAT2_OBJECT_DESIGNATOR']}"
        )

    def test_raw_dedupe_collapses_reissues_and_keeps_order(self, cdm):
        pair = [
            _variant(cdm, pc=3.0e-4, tca="2026-08-28T00:00:00Z", norad=300001,
                     designator="L300001", cdm_id=3001, event_id=7001),
            _variant(cdm, pc=1.0e-4, tca="2026-08-28T00:00:10Z", norad=300001,
                     designator="L300001", cdm_id=3002, event_id=7001),
            _variant(cdm, pc=2.0e-4, tca="2026-08-29T00:00:00Z", norad=300002,
                     designator="L300002", cdm_id=3003, event_id=7002),
        ]
        kept = dedupe_raw_by_event(pair)
        assert [c["COMMENT_ID"] for c in kept] == [3001, 3003]


class TestUnpagedFetchIsUntouched:
    """The evaluate/selector path must keep seeing the whole undeduped window.

    SCRUM-445 bounds the *listing*. A row selector has to resolve no matter which
    page the operator clicked it on, and the secondary screen needs the complete
    in-volume set, so fetch_leolabs_conjunctions keeps its old behaviour: every
    scorable CDM, reissues included, and no volume filter.
    """

    def test_it_still_returns_every_cdm_including_reissues(self, cdm, registry):
        pair = [
            _variant(cdm, pc=3.0e-4, tca="2026-08-28T00:00:00Z", norad=300001,
                     designator="L300001", cdm_id=3001, event_id=7001),
            _variant(cdm, pc=1.0e-4, tca="2026-08-28T00:00:10Z", norad=300001,
                     designator="L300001", cdm_id=3002, event_id=7001),
        ]
        parsed = fetch_leolabs_conjunctions(
            36508, client=_client_for(pair), registry=registry, now=_NOW
        )
        assert len(parsed) == 2

    def test_it_sends_no_volume_filter(self, three_cdms, registry):
        client = _client_for(three_cdms)
        fetch_leolabs_conjunctions(
            36508, client=client, registry=registry, now=_NOW
        )
        sent = client.search_conjunction_cdms.call_args.kwargs
        for param in leolabs_runtime.VOLUME_FILTER_PARAMS:
            assert param not in sent


class TestPageFetch:
    """SCRUM-445: the paged fetch, below the endpoint."""

    def test_the_cursor_round_trips_through_the_page(self, cdm, registry):
        cdms = _many_cdms(cdm, 30)
        page1 = fetch_leolabs_conjunction_page(
            36508, client=_client_for(cdms), registry=registry, now=_NOW,
            page_size=10,
        )
        page2 = fetch_leolabs_conjunction_page(
            36508, client=_client_for(cdms), registry=registry, now=_NOW,
            page_size=10, cursor=page1.next_cursor,
        )
        assert page1.total == 30 and page2.total == 30
        assert page1.offset == 0 and page2.offset == 10
        assert len(page1.rows) == 10 and len(page2.rows) == 10

    def test_a_cursor_for_another_query_raises(self, cdm, registry):
        cdms = _many_cdms(cdm, 30)
        page = fetch_leolabs_conjunction_page(
            36508, client=_client_for(cdms), registry=registry, now=_NOW,
            page_size=10,
        )
        with pytest.raises(LeoLabsCursorError):
            fetch_leolabs_conjunction_page(
                36508, client=_client_for(cdms), registry=registry, now=_NOW,
                page_size=5, cursor=page.next_cursor,
            )

    def test_volume_filters_are_passed_to_the_client(self, cdm, registry):
        client = _client_for(_many_cdms(cdm, 3))
        fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
            volume_filters={"maxRelativePositionR": 1234.0},
        )
        assert client.search_conjunction_cdms.call_args.kwargs[
            "maxRelativePositionR"
        ] == 1234.0

    def test_in_volume_reflects_whether_a_bound_was_applied(self, cdm, registry):
        cdms = _many_cdms(cdm, 3)
        bounded = fetch_leolabs_conjunction_page(
            36508, client=_client_for(cdms), registry=registry, now=_NOW,
            volume_filters=leolabs_runtime.DEFAULT_REPORTING_VOLUME_M,
        )
        wide = fetch_leolabs_conjunction_page(
            36508, client=_client_for(cdms), registry=registry, now=_NOW,
            volume_filters={},
        )
        assert bounded.in_volume is True
        assert wide.in_volume is False

    def test_an_offset_past_the_end_is_an_empty_page_not_an_error(
        self, cdm, registry
    ):
        cdms = _many_cdms(cdm, 12)
        page = fetch_leolabs_conjunction_page(
            36508, client=_client_for(cdms), registry=registry, now=_NOW,
            page_size=10,
        )
        last = fetch_leolabs_conjunction_page(
            36508, client=_client_for(cdms), registry=registry, now=_NOW,
            page_size=10, cursor=page.next_cursor,
        )
        assert len(last.rows) == 2
        assert last.next_cursor is None

    def test_an_unsubscribed_asset_raises(self, cdm, registry):
        with pytest.raises(LeoLabsRuntimeError):
            fetch_leolabs_conjunction_page(
                99999, client=_client_for(_many_cdms(cdm, 3)),
                registry=registry, now=_NOW,
            )

    def test_resolve_volume_filters_defaults_to_the_reporting_volume(self):
        assert leolabs_runtime.resolve_volume_filters() == (
            leolabs_runtime.DEFAULT_REPORTING_VOLUME_M
        )
        assert leolabs_runtime.resolve_volume_filters(in_volume=False) == {}
        narrowed = leolabs_runtime.resolve_volume_filters(500.0)
        assert narrowed["maxRelativePositionR"] == 500.0
        assert narrowed["maxRelativePositionI"] == (
            leolabs_runtime.DEFAULT_REPORTING_VOLUME_M["maxRelativePositionI"]
        )


class TestSelectConjunction:
    @pytest.fixture
    def parsed(self, three_cdms, registry):
        return fetch_leolabs_conjunctions(
            36508, client=_client_for(three_cdms), registry=registry, now=_NOW
        )

    def test_select_by_secondary_norad(self, parsed):
        chosen = select_conjunction(parsed, {"secondary_norad": 100003})

        assert chosen.secondary.norad_id == 100003

    def test_a_norad_supplied_as_a_string_still_matches(self, parsed):
        """A JSON request body carries it as a string; the parser holds an int."""
        chosen = select_conjunction(parsed, {"secondary_norad": "100003"})

        assert chosen.secondary.norad_id == 100003

    def test_select_by_cdm_id(self, parsed):
        chosen = select_conjunction(parsed, {"cdm_id": 1001})

        assert chosen.secondary.norad_id == 100001

    def test_select_by_event_id(self, parsed):
        chosen = select_conjunction(parsed, {"event_id": 9003})

        assert chosen.secondary.norad_id == 100003

    def test_cdm_id_wins_over_a_less_specific_field(self, parsed):
        chosen = select_conjunction(
            parsed, {"cdm_id": 1001, "secondary_norad": 100002}
        )

        assert chosen.secondary.norad_id == 100001

    def test_an_unmatched_selector_returns_none_not_the_top_row(self, parsed):
        """Falling back would label another object's numbers as the clicked one."""
        assert select_conjunction(parsed, {"secondary_norad": 999999}) is None
        assert select_conjunction(parsed, {"cdm_id": 999999}) is None

    def test_an_empty_selector_selects_nothing(self, parsed):
        assert select_conjunction(parsed, {}) is None

    def test_selector_extraction_ignores_absent_and_blank_fields(self):
        assert selector_from_conjunction_block({"primary_norad": "36508"}) == {}
        assert selector_from_conjunction_block({"secondary_norad": ""}) == {}
        assert selector_from_conjunction_block(
            {"primary_norad": "36508", "secondary_norad": 42}
        ) == {"secondary_norad": 42}


# ---------------------------------------------------------------------------
# GET /v1/leolabs/conjunctions
# ---------------------------------------------------------------------------


def _parsed_from(cdms, registry):
    """What the real fetch would hand the endpoint, risk-ordered.

    Deliberately goes through fetch_leolabs_conjunctions rather than mapping
    parse_leolabs_cdm over the raw list: the endpoint's contract is that rows
    arrive already ordered, so a stub that skipped the sort would let an
    ordering bug pass.
    """
    return fetch_leolabs_conjunctions(
        36508, client=_client_for(cdms), registry=registry, now=_NOW
    )


def _serve(cdms, registry, monkeypatch):
    """Wire the endpoint to a mocked LeoLabs client and return the TestClient.

    SCRUM-445: the endpoint's paging is the thing under test, so these tests stub
    the *client* and let the real route -> fetch_leolabs_conjunction_page ->
    dedupe/sort/slice/cursor path run. Stubbing the fetch function instead (what
    the SCRUM-422 tests did, when there was no paging to get wrong) would let a
    cursor or ordering bug straight through.

    Returns the mock client too, so a test can assert what reached LeoLabs.
    """
    client = _client_for(cdms)
    monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
    monkeypatch.setattr(leolabs_runtime, "get_client", lambda: client)
    monkeypatch.setattr(leolabs_runtime, "get_registry", lambda c=None: registry)
    return TestClient(server.svc), client


def _get(tc, **params):
    params.setdefault("primary_norad", 36508)
    return tc.get("/v1/leolabs/conjunctions", params=params)


def _many_cdms(base, n, *, start_norad=200000):
    """n CDMs for n distinct events, Pc descending so risk order is known."""
    return [
        _variant(
            base,
            pc=1.0e-4 * (n - i),
            tca=f"2026-08-27T{i // 60:02d}:{i % 60:02d}:00Z",
            norad=start_norad + i,
            designator=f"L{start_norad + i}",
            cdm_id=500000 + i,
            event_id=600000 + i,
        )
        for i in range(n)
    ]


class TestListEndpoint:
    def test_lists_an_assets_conjunctions(self, three_cdms, registry, monkeypatch):
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        resp = _get(tc)

        assert resp.status_code == 200
        data = resp.json()
        assert data["primary_norad"] == 36508
        assert data["source"] == "leolabs"
        assert data["count"] == 3
        assert len(data["conjunctions"]) == 3

    def test_rows_come_back_highest_risk_first(self, three_cdms, registry, monkeypatch):
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        rows = _get(tc).json()["conjunctions"]

        assert [r["pc"] for r in rows] == [5.0e-4, 2.0e-5, 1.0e-6]
        assert [r["risk_level"] for r in rows] == ["RED", "AMBER", "GREEN"]

    def test_an_empty_window_is_an_empty_list_with_200(self, registry, monkeypatch):
        tc, _ = _serve([], registry, monkeypatch)
        resp = _get(tc)

        assert resp.status_code == 200
        assert resp.json()["conjunctions"] == []
        assert resp.json()["count"] == 0
        # An empty window has no next page and nothing to page to.
        assert resp.json()["total"] == 0
        assert resp.json()["next_cursor"] is None

    def test_the_response_states_the_window_it_searched(self, registry, monkeypatch):
        """An empty table means something different over an hour than a week."""
        tc, _ = _serve([], registry, monkeypatch)
        window = _get(tc, lookahead_days=3).json()["window"]

        assert window["lookahead_days"] == 3
        assert window["min_tca_utc"] < window["max_tca_utc"]

    def test_reissued_cdms_are_collapsed_and_the_count_says_so(
        self, cdm, registry, monkeypatch
    ):
        pair = [
            _variant(cdm, pc=3.0e-4, tca="2026-08-28T00:00:00Z", norad=300001,
                     designator="L300001", cdm_id=3001, event_id=7001),
            _variant(cdm, pc=1.0e-4, tca="2026-08-28T00:00:10Z", norad=300001,
                     designator="L300001", cdm_id=3002, event_id=7001),
        ]
        tc, _ = _serve(pair, registry, monkeypatch)
        data = _get(tc).json()

        assert data["count"] == 1
        assert data["cdm_count"] == 2
        # The event total counts events, not reissues, so paging cannot advertise
        # a second page that is really the same conjunction again.
        assert data["total"] == 1
        assert data["next_cursor"] is None

    # -- SCRUM-445: paging, the volume filters, and the row geometry ------

    def test_a_page_reports_next_cursor_and_the_event_total(
        self, cdm, registry, monkeypatch
    ):
        tc, _ = _serve(_many_cdms(cdm, 250), registry, monkeypatch)
        data = _get(tc, page_size=100).json()

        assert data["count"] == 100
        assert len(data["conjunctions"]) == 100
        # total is the whole in-volume set, not the page, so the dashboard can
        # render "1-100 of 250" from one response.
        assert data["total"] == 250
        assert data["next_cursor"]
        assert data["page"] == {
            "size": 100, "offset": 0, "returned": 100, "has_more": True
        }

    def test_the_cursor_returns_the_next_page_and_the_pages_do_not_overlap(
        self, cdm, registry, monkeypatch
    ):
        tc, _ = _serve(_many_cdms(cdm, 250), registry, monkeypatch)

        page1 = _get(tc, page_size=100).json()
        page2 = _get(tc, page_size=100, cursor=page1["next_cursor"]).json()
        page3 = _get(tc, page_size=100, cursor=page2["next_cursor"]).json()

        assert [p["count"] for p in (page1, page2, page3)] == [100, 100, 50]
        assert page2["page"]["offset"] == 100
        assert page3["page"]["offset"] == 200
        # The last page ends the walk rather than looping.
        assert page3["next_cursor"] is None
        assert page3["page"]["has_more"] is False

        ids = [r["cdm_id"] for p in (page1, page2, page3) for r in p["conjunctions"]]
        assert len(ids) == 250
        assert len(set(ids)) == 250, "a conjunction appeared on two pages"

    def test_worst_pc_first_holds_across_pages_not_just_within_one(
        self, cdm, registry, monkeypatch
    ):
        """The point of the ordering: page one really is the worst conjunctions.

        Paging on LeoLabs' own cursor would return LeoLabs' order, so the worst
        conjunction in a window could land on page seven while an operator read
        page one top-down. The whole in-volume set is ordered before it is sliced,
        so that cannot happen.
        """
        tc, _ = _serve(_many_cdms(cdm, 120), registry, monkeypatch)

        page1 = _get(tc, page_size=50).json()
        page2 = _get(tc, page_size=50, cursor=page1["next_cursor"]).json()
        pcs = [r["pc"] for r in page1["conjunctions"] + page2["conjunctions"]]

        assert pcs == sorted(pcs, reverse=True)
        assert page1["conjunctions"][0]["pc"] == max(pcs)

    def test_the_volume_filters_reach_the_client(self, cdm, registry, monkeypatch):
        tc, client = _serve(_many_cdms(cdm, 5), registry, monkeypatch)
        _get(tc, max_relative_position_r_m=1500, max_relative_position_i_m=20000)

        sent = client.search_conjunction_cdms.call_args.kwargs
        assert sent["maxRelativePositionR"] == 1500.0
        assert sent["maxRelativePositionI"] == 20000.0
        # The unset axis falls back to LeoLabs' reporting volume rather than being
        # dropped, so one narrowed axis cannot silently widen another.
        assert sent["maxRelativePositionC"] == (
            leolabs_runtime.DEFAULT_REPORTING_VOLUME_M["maxRelativePositionC"]
        )

    def test_the_reporting_volume_is_the_default_and_is_reported(
        self, cdm, registry, monkeypatch
    ):
        tc, client = _serve(_many_cdms(cdm, 5), registry, monkeypatch)
        data = _get(tc).json()

        sent = client.search_conjunction_cdms.call_args.kwargs
        for param, expected in leolabs_runtime.DEFAULT_REPORTING_VOLUME_M.items():
            assert sent[param] == expected
        # And the response says so: a short list because the volume cropped it
        # must not read like a quiet sky.
        assert data["volume_filter"]["in_volume"] is True
        assert data["volume_filter"]["max_relative_position_r_m"] == 2000.0

    def test_in_volume_false_widens_by_dropping_the_filters(
        self, cdm, registry, monkeypatch
    ):
        tc, client = _serve(_many_cdms(cdm, 5), registry, monkeypatch)
        data = _get(tc, in_volume="false").json()

        sent = client.search_conjunction_cdms.call_args.kwargs
        for param in leolabs_runtime.VOLUME_FILTER_PARAMS:
            assert param not in sent
        assert data["volume_filter"]["in_volume"] is False
        assert data["volume_filter"]["max_relative_position_r_m"] is None

    def test_a_cursor_from_a_different_query_is_422_not_a_wrong_page(
        self, cdm, registry, monkeypatch
    ):
        """A replayed cursor must not walk one query's offsets through another's."""
        tc, _ = _serve(_many_cdms(cdm, 250), registry, monkeypatch)
        cursor = _get(tc, page_size=100).json()["next_cursor"]

        # Same cursor, different page size: the offset no longer means what it did.
        assert _get(tc, page_size=50, cursor=cursor).status_code == 422
        # Same cursor, different window.
        assert _get(tc, page_size=100, cursor=cursor,
                    lookahead_days=3).status_code == 422
        # Same cursor, different volume filter.
        assert _get(tc, page_size=100, cursor=cursor,
                    in_volume="false").status_code == 422
        # And unchanged, it still works -- the guard is not simply refusing all.
        assert _get(tc, page_size=100, cursor=cursor).status_code == 200

    def test_a_garbage_cursor_is_422(self, cdm, registry, monkeypatch):
        tc, _ = _serve(_many_cdms(cdm, 10), registry, monkeypatch)
        assert _get(tc, cursor="not-a-cursor").status_code == 422

    def test_page_size_is_capped_rather_than_refused(
        self, cdm, registry, monkeypatch
    ):
        tc, _ = _serve(_many_cdms(cdm, 600), registry, monkeypatch)
        data = _get(tc, page_size=100000).json()

        assert data["page"]["size"] == leolabs_runtime.MAX_PAGE_SIZE
        assert data["count"] == leolabs_runtime.MAX_PAGE_SIZE

    def test_a_page_size_below_one_is_422(self, cdm, registry, monkeypatch):
        tc, _ = _serve(_many_cdms(cdm, 10), registry, monkeypatch)
        assert _get(tc, page_size=0).status_code == 422

    def test_only_the_returned_page_is_parsed(self, cdm, registry, monkeypatch):
        """The bounded parse is the fix; an unbounded one is the timeout.

        Parsing is where the time goes, so a page must not parse the whole set on
        its way to slicing ten rows out of it.
        """
        import common.leolabs_runtime as rt

        calls = {"n": 0}
        real = rt.parse_leolabs_cdm

        def _counting(cdm_dict, our_id):
            calls["n"] += 1
            return real(cdm_dict, our_id)

        monkeypatch.setattr(rt, "parse_leolabs_cdm", _counting)
        tc, _ = _serve(_many_cdms(cdm, 400), registry, monkeypatch)
        data = _get(tc, page_size=10).json()

        assert data["total"] == 400
        assert calls["n"] == 10

    def test_a_row_carries_the_rtn_geometry_the_viz_draws(
        self, three_cdms, registry, monkeypatch
    ):
        """Row-to-geometry is select-and-highlight, so the row carries the state."""
        tc, _ = _serve(three_cdms, registry, monkeypatch)
        row = _get(tc).json()["conjunctions"][0]

        assert len(row["relative_position_rtn_m"]) == 3
        assert len(row["relative_velocity_rtn_m_s"]) == 3
        # Provenance-labelled: these are the CDM's own numbers, not a propagation.
        assert row["relative_state_source"] == "leolabs_cdm_rtn"

    def test_disabled_is_503_not_an_empty_list(self, monkeypatch):
        """'The feed is off' and 'the sky is clear' must not look alike."""
        monkeypatch.setattr(server, "LEOLABS_ENABLED", False)
        resp = TestClient(server.svc).get(
            "/v1/leolabs/conjunctions", params={"primary_norad": 36508}
        )

        assert resp.status_code == 503

    def test_an_unsubscribed_asset_is_404(self, monkeypatch):
        monkeypatch.setattr(server, "LEOLABS_ENABLED", True)

        def _raise(n, **kw):
            raise LeoLabsRuntimeError("NORAD 99999 is not in the registry")

        monkeypatch.setattr(server, "fetch_leolabs_conjunction_page", _raise)
        resp = TestClient(server.svc).get(
            "/v1/leolabs/conjunctions", params={"primary_norad": 99999}
        )

        assert resp.status_code == 404

    def test_a_fetch_failure_is_503(self, monkeypatch):
        monkeypatch.setattr(server, "LEOLABS_ENABLED", True)

        def _boom(n, **kw):
            raise RuntimeError("upstream 500")

        monkeypatch.setattr(server, "fetch_leolabs_conjunction_page", _boom)
        resp = TestClient(server.svc).get(
            "/v1/leolabs/conjunctions", params={"primary_norad": 36508}
        )

        assert resp.status_code == 503

    def test_a_window_wider_than_the_leolabs_cap_is_422(self, monkeypatch):
        monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
        resp = TestClient(server.svc).get(
            "/v1/leolabs/conjunctions",
            params={"primary_norad": 36508, "lookahead_days": 31},
        )

        assert resp.status_code == 422

    def test_a_missing_primary_norad_is_rejected_by_validation(self, monkeypatch):
        monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
        resp = TestClient(server.svc).get("/v1/leolabs/conjunctions")

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /v1/evaluate with a row selector
# ---------------------------------------------------------------------------


def _evaluate_body(primary_norad=36508, **conjunction_extra):
    conjunction = {"primary_norad": primary_norad, "obj_id": "unknown"}
    conjunction.update(conjunction_extra)
    return {
        "conjunction_id": "run-422",
        "satellite": {
            "sat_id": "XORB-001",
            "r_sat_km": [6778.0, 0.0, 0.0],
            "v_sat_km_s": [0.0, 7.66, 0.0],
            "v_remaining_m_s": 25.0,
        },
        "policy": {
            "operator_id": "O", "policy_version": "2.5.7",
            "lambda_v": 0.01, "lambda_L": 0.001, "dv_mag_limit_m_s": 0.5,
        },
        "conjunction": conjunction,
    }


class TestEvaluateBySelector:
    @pytest.fixture(autouse=True)
    def _enabled(self, monkeypatch, three_cdms, registry):
        monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
        monkeypatch.setattr(
            server, "fetch_leolabs_conjunctions",
            lambda n, **kw: _parsed_from(three_cdms, registry),
        )

        # SCRUM-460: a cdm_id selector no longer resolves through the whole-window
        # search -- it fetches that one CDM directly, because the window pull is
        # what took the planner down on a dense asset. Stubbed from the same fixture
        # set so the property under test is unchanged: the selector scores the row it
        # names. Which mechanism found it is asserted in
        # test_evaluate_cdm_id_resolve.py.
        def _by_cdm_id(cdm_id, primary_norad, **kw):
            wanted = str(cdm_id)
            for parsed in _parsed_from(three_cdms, registry):
                if str(parsed.provenance.get("cdm_id")) == wanted:
                    return parsed
            return None

        monkeypatch.setattr(
            server, "fetch_leolabs_conjunction_by_cdm_id", _by_cdm_id)

    def test_no_selector_still_scores_the_highest_risk_conjunction(
        self, monkeypatch, three_cdms, registry
    ):
        """SCRUM-412 behaviour, unchanged: the singular fetch is still the call."""
        singular = MagicMock(return_value=_parsed_from(three_cdms, registry)[0])
        monkeypatch.setattr(server, "fetch_leolabs_conjunction", singular)
        resp = TestClient(server.svc).post("/v1/evaluate", json=_evaluate_body())

        assert resp.status_code == 200
        assert resp.json()["conjunction"]["secondary"]["norad_id"] == 100002
        singular.assert_called_once()

    def test_a_selector_scores_the_chosen_conjunction_not_the_top_one(self):
        body = _evaluate_body(secondary_norad=100003)
        resp = TestClient(server.svc).post("/v1/evaluate", json=body)

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "leolabs"
        # 100002 is the highest-risk conjunction; we asked for 100003.
        assert data["conjunction"]["secondary"]["norad_id"] == 100003
        assert data["recommendation"]["direction"]

    def test_a_selector_by_cdm_id_scores_that_cdm(self):
        resp = TestClient(server.svc).post(
            "/v1/evaluate", json=_evaluate_body(cdm_id=1001)
        )

        assert resp.status_code == 200
        assert resp.json()["conjunction"]["secondary"]["norad_id"] == 100001

    def test_a_selector_by_event_id_scores_that_event(self):
        resp = TestClient(server.svc).post(
            "/v1/evaluate", json=_evaluate_body(event_id=9003)
        )

        assert resp.status_code == 200
        assert resp.json()["conjunction"]["secondary"]["norad_id"] == 100003

    def test_the_scored_row_carries_real_cdm_covariance(self):
        resp = TestClient(server.svc).post(
            "/v1/evaluate", json=_evaluate_body(secondary_norad=100003)
        )

        assert resp.json()["covariance_source"] == "real_cdm"

    def test_a_stale_selector_is_404_rather_than_scoring_something_else(self):
        """The row aged out of the window. Returning the top conjunction would
        label another object's numbers as the one the operator clicked."""
        resp = TestClient(server.svc).post(
            "/v1/evaluate", json=_evaluate_body(secondary_norad=999999)
        )

        assert resp.status_code == 404

    def test_a_selector_without_primary_norad_stays_on_the_surrogate_path(
        self, monkeypatch
    ):
        """SCRUM-420 still holds: no primary_norad means evaluate what I gave you,
        and a stray selector does not drag the request onto the live path."""
        called = {"hit": False}

        def _spy(n, **kw):
            called["hit"] = True
            return []

        monkeypatch.setattr(server, "fetch_leolabs_conjunctions", _spy)
        body = _evaluate_body(secondary_norad=100003)
        del body["conjunction"]["primary_norad"]
        body["satellite"]["t_burn_utc"] = "2026-06-22T08:00:00Z"
        body["conjunction"].update({
            "t_ca_utc": "2026-06-22T10:00:00Z",
            "r_rel_km": [0.1, 0.2, 0.3],
            "v_rel_km_s": [0.0, -0.5, 0.1],
            "p_rel_km2": [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4],
        })
        resp = TestClient(server.svc).post("/v1/evaluate", json=body)

        assert resp.status_code == 200
        assert resp.json()["source"] == "surrogate"
        assert called["hit"] is False
