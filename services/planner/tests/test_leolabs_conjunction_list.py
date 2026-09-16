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
    pc_to_risk_level,
    select_conjunction,
    selector_from_conjunction_block,
)
from common.leolabs_runtime import (
    LeoLabsRuntimeError,
    fetch_leolabs_conjunction,
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


class TestListEndpoint:
    def test_lists_an_assets_conjunctions(self, three_cdms, registry, monkeypatch):
        monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
        monkeypatch.setattr(
            server, "fetch_leolabs_conjunctions",
            lambda n, **kw: _parsed_from(three_cdms, registry),
        )
        resp = TestClient(server.svc).get(
            "/v1/leolabs/conjunctions", params={"primary_norad": 36508}
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["primary_norad"] == 36508
        assert data["source"] == "leolabs"
        assert data["count"] == 3
        assert len(data["conjunctions"]) == 3

    def test_rows_come_back_highest_risk_first(self, three_cdms, registry, monkeypatch):
        monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
        monkeypatch.setattr(
            server, "fetch_leolabs_conjunctions",
            lambda n, **kw: _parsed_from(three_cdms, registry),
        )
        rows = TestClient(server.svc).get(
            "/v1/leolabs/conjunctions", params={"primary_norad": 36508}
        ).json()["conjunctions"]

        assert [r["pc"] for r in rows] == [5.0e-4, 2.0e-5, 1.0e-6]
        assert [r["risk_level"] for r in rows] == ["RED", "AMBER", "GREEN"]

    def test_an_empty_window_is_an_empty_list_with_200(self, monkeypatch):
        monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
        monkeypatch.setattr(server, "fetch_leolabs_conjunctions", lambda n, **kw: [])
        resp = TestClient(server.svc).get(
            "/v1/leolabs/conjunctions", params={"primary_norad": 36508}
        )

        assert resp.status_code == 200
        assert resp.json()["conjunctions"] == []
        assert resp.json()["count"] == 0

    def test_the_response_states_the_window_it_searched(self, monkeypatch):
        """An empty table means something different over an hour than a week."""
        monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
        monkeypatch.setattr(server, "fetch_leolabs_conjunctions", lambda n, **kw: [])
        window = TestClient(server.svc).get(
            "/v1/leolabs/conjunctions",
            params={"primary_norad": 36508, "lookahead_days": 3},
        ).json()["window"]

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
        monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
        monkeypatch.setattr(
            server, "fetch_leolabs_conjunctions",
            lambda n, **kw: _parsed_from(pair, registry),
        )
        data = TestClient(server.svc).get(
            "/v1/leolabs/conjunctions", params={"primary_norad": 36508}
        ).json()

        assert data["count"] == 1
        assert data["cdm_count"] == 2

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

        monkeypatch.setattr(server, "fetch_leolabs_conjunctions", _raise)
        resp = TestClient(server.svc).get(
            "/v1/leolabs/conjunctions", params={"primary_norad": 99999}
        )

        assert resp.status_code == 404

    def test_a_fetch_failure_is_503(self, monkeypatch):
        monkeypatch.setattr(server, "LEOLABS_ENABLED", True)

        def _boom(n, **kw):
            raise RuntimeError("upstream 500")

        monkeypatch.setattr(server, "fetch_leolabs_conjunctions", _boom)
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
