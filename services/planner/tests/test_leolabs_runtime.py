"""tests/test_leolabs_runtime.py

SCRUM-412: the LeoLabs live runtime -- fetch/select/parse, the /v1/evaluate
wiring behind LEOLABS_ENABLED, precedence over UDL, and /leolabs-status.

All offline: the client is mocked and the feature flag is patched. CI needs no
credential.

Run from repo root:
    python -m pytest services/planner/tests/test_leolabs_runtime.py -v
"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import server
from common import leolabs_runtime
from common.leolabs_asset_map import AssetRegistry
from common.leolabs_cdm_parser import parse_leolabs_cdm
from common.leolabs_runtime import LeoLabsRuntimeError, fetch_leolabs_conjunction

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


# ---------------------------------------------------------------------------
# SCRUM-450: the short-TTL conjunction-window cache
# ---------------------------------------------------------------------------

class TestCdmWindowCache:
    """The fetch is ~13 to 20 s, so the question is only how often it is made.

    Every test here counts client calls. The autouse _clean_caches fixture above
    resets the cache around each one, so nothing leaks between them.
    """

    @staticmethod
    def _client(cdms):
        client = MagicMock()
        client.search_conjunction_cdms.return_value = cdms
        return client

    def _fetch(self, client, registry, **kw):
        return leolabs_runtime.fetch_leolabs_conjunctions(
            36508, client=client, registry=registry, now=_NOW, **kw
        )

    def test_a_second_fetch_in_the_ttl_does_not_call_the_client_again(
        self, cdm, registry
    ):
        client = self._client([cdm])
        first = self._fetch(client, registry)
        second = self._fetch(client, registry)

        assert client.search_conjunction_cdms.call_count == 1
        # And the caller is not shortchanged for the hit.
        assert len(second) == len(first) == 1
        assert second[0].primary.designator == first[0].primary.designator

    def test_the_cached_result_matches_the_uncached_one_exactly(
        self, cdm, registry
    ):
        """Nothing downstream may notice whether a fetch was served from cache."""
        client = self._client([cdm])
        uncached = self._fetch(client, registry)
        leolabs_runtime.reset_caches()
        client2 = self._client([cdm])
        fresh = self._fetch(client2, registry)
        cached = self._fetch(client2, registry)   # a hit

        assert client2.search_conjunction_cdms.call_count == 1
        for a, b in ((uncached, fresh), (fresh, cached)):
            assert [p.provenance.get("cdm_id") for p in a] == \
                   [p.provenance.get("cdm_id") for p in b]
            assert [p.t_ca_utc for p in a] == [p.t_ca_utc for p in b]

    def test_a_different_asset_is_a_miss(self, cdm, registry):
        two_assets = AssetRegistry.from_objects([
            {"catalogNumber": "L2669", "noradCatalogNumber": 36508, "name": "A"},
            {"catalogNumber": "L9999", "noradCatalogNumber": 44444, "name": "B"},
        ])
        client = self._client([cdm])
        leolabs_runtime.fetch_leolabs_conjunctions(
            36508, client=client, registry=two_assets, now=_NOW)
        leolabs_runtime.fetch_leolabs_conjunctions(
            44444, client=client, registry=two_assets, now=_NOW)

        assert client.search_conjunction_cdms.call_count == 2

    @pytest.mark.parametrize("kw", [
        {"lookahead_days": 3},
        {"lookback_days": 2},
    ])
    def test_a_different_window_in_days_is_a_miss(self, cdm, registry, kw):
        client = self._client([cdm])
        self._fetch(client, registry)
        self._fetch(client, registry, **kw)

        assert client.search_conjunction_cdms.call_count == 2

    def test_a_different_volume_filter_is_a_miss(self, cdm, registry):
        """The evaluate path fetches unfiltered and the listing path does not.

        They are genuinely different result sets and must never be served to each
        other, so the filters are part of the key.
        """
        client = self._client([cdm])
        leolabs_runtime.fetch_leolabs_conjunctions(
            36508, client=client, registry=registry, now=_NOW)          # no filter
        leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
            volume_filters=dict(leolabs_runtime.DEFAULT_REPORTING_VOLUME_M),
        )                                                                # filtered
        assert client.search_conjunction_cdms.call_count == 2

        # A narrower volume than the default is a third distinct entry.
        leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
            volume_filters={"maxRelativePositionR": 500.0},
        )
        assert client.search_conjunction_cdms.call_count == 3

    def test_the_two_paths_share_an_entry_when_their_filters_match(
        self, cdm, registry
    ):
        """Same asset, window and filters is one window, fetched once."""
        client = self._client([cdm])
        leolabs_runtime.fetch_leolabs_conjunctions(
            36508, client=client, registry=registry, now=_NOW)
        leolabs_runtime.fetch_leolabs_conjunction_page(
            36508, client=client, registry=registry, now=_NOW,
            volume_filters={},                      # also unfiltered
        )
        assert client.search_conjunction_cdms.call_count == 1

    def test_the_entry_expires_after_the_ttl(self, cdm, registry, monkeypatch):
        """Driven by moving the clock, not by sleeping through the TTL."""
        client = self._client([cdm])
        fake_now = {"t": 1000.0}
        monkeypatch.setattr(
            leolabs_runtime.time, "monotonic", lambda: fake_now["t"])

        self._fetch(client, registry)
        assert client.search_conjunction_cdms.call_count == 1

        # Just inside the TTL: still a hit.
        fake_now["t"] += leolabs_runtime._CDM_CACHE_TTL_SECONDS - 0.01
        self._fetch(client, registry)
        assert client.search_conjunction_cdms.call_count == 1

        # Past it: fetched again.
        fake_now["t"] += 0.02
        self._fetch(client, registry)
        assert client.search_conjunction_cdms.call_count == 2

    def test_reset_caches_empties_it(self, cdm, registry):
        client = self._client([cdm])
        self._fetch(client, registry)
        assert leolabs_runtime._cdm_cache

        leolabs_runtime.reset_caches()
        assert leolabs_runtime._cdm_cache == {}

        self._fetch(client, registry)
        assert client.search_conjunction_cdms.call_count == 2

    def test_a_cache_hit_still_sends_the_right_request_when_it_does_fetch(
        self, cdm, registry
    ):
        """The cached fetch is the same request it always was."""
        client = self._client([cdm])
        self._fetch(client, registry)

        kwargs = client.search_conjunction_cdms.call_args.kwargs
        assert kwargs["object1"] == "L2669"
        assert kwargs["cdm_source"] == "LeoLabs"
        assert kwargs["min_tca"] < kwargs["max_tca"]

    def test_an_empty_window_is_cached_too(self, registry):
        """Otherwise a quiet asset pays the full fetch on every selection."""
        client = self._client([])
        assert self._fetch(client, registry) == []
        assert self._fetch(client, registry) == []
        assert client.search_conjunction_cdms.call_count == 1


# ---------------------------------------------------------------------------
# fetch_leolabs_conjunction
# ---------------------------------------------------------------------------

def test_fetch_returns_parsed_conjunction(cdm, registry):
    client = MagicMock()
    client.search_conjunction_cdms.return_value = [cdm]
    parsed = fetch_leolabs_conjunction(36508, client=client, registry=registry, now=_NOW)
    assert parsed is not None
    assert parsed.primary.designator == "L2669"
    # cdm_source filter and object1 mapping were applied.
    kwargs = client.search_conjunction_cdms.call_args.kwargs
    assert kwargs["object1"] == "L2669"
    assert kwargs["cdm_source"] == "LeoLabs"


def test_fetch_returns_none_when_no_cdms(registry):
    client = MagicMock()
    client.search_conjunction_cdms.return_value = []
    assert fetch_leolabs_conjunction(36508, client=client, registry=registry, now=_NOW) is None


def test_fetch_skips_guard_failing_cdm_and_picks_next(cdm, registry):
    """A higher-Pc CDM that fails a guard is skipped for the next scorable one."""
    bad = copy.deepcopy(cdm)
    bad["COLLISION_PROBABILITY"] = 1.0e-3          # sorts first (higher risk)
    bad["SAT1_COVARIANCE_METHOD"] = "DEFAULT"       # but fails the CALCULATED guard
    good = cdm                                       # lower Pc, scorable
    client = MagicMock()
    client.search_conjunction_cdms.return_value = [good, bad]
    parsed = fetch_leolabs_conjunction(36508, client=client, registry=registry, now=_NOW)
    assert parsed is not None
    assert parsed.cdm_collision_probability == pytest.approx(cdm["COLLISION_PROBABILITY"])


def test_fetch_unsubscribed_norad_raises(registry):
    client = MagicMock()
    with pytest.raises(LeoLabsRuntimeError):
        fetch_leolabs_conjunction(99999, client=client, registry=registry, now=_NOW)


def test_registry_is_built_once(cdm):
    client = MagicMock()
    client.list_subscribed_objects.return_value = _OBJECTS
    r1 = leolabs_runtime.get_registry(client)
    r2 = leolabs_runtime.get_registry(client)
    assert r1 is r2
    client.list_subscribed_objects.assert_called_once()


# ---------------------------------------------------------------------------
# /v1/evaluate wiring
# ---------------------------------------------------------------------------

def _body(primary_norad=36508):
    return {
        "conjunction_id": "run-1",
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
        "conjunction": {"primary_norad": primary_norad, "obj_id": "unknown"},
    }


def test_evaluate_uses_leolabs_when_enabled(cdm, monkeypatch):
    parsed = parse_leolabs_cdm(cdm, "L2669")
    monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
    monkeypatch.setattr(server, "fetch_leolabs_conjunction", lambda n: parsed)

    resp = TestClient(server.svc).post("/v1/evaluate", json=_body())
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "leolabs"
    assert data["covariance_source"] == "real_cdm"
    assert data["recommendation"]["direction"]


def test_evaluate_leolabs_no_records_returns_no_maneuver(monkeypatch):
    monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
    monkeypatch.setattr(server, "fetch_leolabs_conjunction", lambda n: None)

    resp = TestClient(server.svc).post("/v1/evaluate", json=_body())
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "leolabs"
    assert data["recommendation"]["direction"] == "no_maneuver_needed"


def test_evaluate_leolabs_without_primary_norad_uses_surrogate(monkeypatch):
    """SCRUM-420: LEOLABS_ENABLED=true with no primary_norad must NOT 422.

    A request that omits primary_norad is a self-contained (surrogate) evaluate --
    e.g. a dashboard scenario preset carrying its own covariance. It must evaluate
    the supplied conjunction block directly, never attempt a live LeoLabs fetch,
    and report source=surrogate. Enabling LeoLabs must not break the scenario path.
    """
    monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
    fetch_called = {"hit": False}

    def _spy(n):
        fetch_called["hit"] = True
        return None

    monkeypatch.setattr(server, "fetch_leolabs_conjunction", _spy)
    body = _body()
    del body["conjunction"]["primary_norad"]
    body["satellite"]["t_burn_utc"] = "2026-06-22T08:00:00Z"
    # Self-contained surrogate conjunction block (no live fetch needed).
    body["conjunction"].update({
        "t_ca_utc": "2026-06-22T10:00:00Z",
        "r_rel_km": [0.1, 0.2, 0.3],
        "v_rel_km_s": [0.0, -0.5, 0.1],
        "p_rel_km2": [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4],
    })
    resp = TestClient(server.svc).post("/v1/evaluate", json=body)
    assert resp.status_code == 200
    assert resp.json()["source"] == "surrogate"
    assert fetch_called["hit"] is False


def test_evaluate_leolabs_fetch_failure_returns_503(monkeypatch):
    def _boom(n):
        raise RuntimeError("network down")
    monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
    monkeypatch.setattr(server, "fetch_leolabs_conjunction", _boom)
    resp = TestClient(server.svc).post("/v1/evaluate", json=_body())
    assert resp.status_code == 503


def test_leolabs_takes_precedence_over_udl(cdm, monkeypatch):
    """With both flags on, LeoLabs drives and UDL is never consulted."""
    parsed = parse_leolabs_cdm(cdm, "L2669")
    monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
    monkeypatch.setattr(server, "UDL_ENABLED", True)
    monkeypatch.setattr(server, "fetch_leolabs_conjunction", lambda n: parsed)

    udl_called = {"hit": False}
    def _udl_spy(*a, **k):
        udl_called["hit"] = True
        return []
    monkeypatch.setattr(server, "get_conjunctions", _udl_spy)

    resp = TestClient(server.svc).post("/v1/evaluate", json=_body())
    assert resp.status_code == 200
    assert resp.json()["source"] == "leolabs"
    assert udl_called["hit"] is False


def test_evaluate_unchanged_when_leolabs_disabled(monkeypatch):
    """LEOLABS_ENABLED=false -> the LeoLabs path is inert; source is not leolabs."""
    monkeypatch.setattr(server, "LEOLABS_ENABLED", False)
    monkeypatch.setattr(server, "UDL_ENABLED", False)
    body = _body()
    body["satellite"]["t_burn_utc"] = "2026-08-30T09:00:00Z"  # standard path needs it
    body["conjunction"] = {
        "obj_id": "x", "t_ca_utc": "2026-08-30T10:00:00Z",
        "r_rel_km": [0.1, 0.2, 0.3], "v_rel_km_s": [0.0, -0.5, 0.1],
        "p_rel_km2": [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4],
    }
    resp = TestClient(server.svc).post("/v1/evaluate", json=body)
    assert resp.status_code == 200
    assert resp.json()["source"] != "leolabs"


# ---------------------------------------------------------------------------
# /leolabs-status
# ---------------------------------------------------------------------------

def test_status_disabled(monkeypatch):
    monkeypatch.setattr(leolabs_runtime, "LEOLABS_ENABLED", False)
    resp = TestClient(server.svc).get("/leolabs-status")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "disabled"


def test_status_misconfigured_when_enabled_without_creds(monkeypatch):
    monkeypatch.setattr(leolabs_runtime, "LEOLABS_ENABLED", True)
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LEOLABS_ACCESS_KEY", None)
        os.environ.pop("LEOLABS_SECRET_KEY", None)
        resp = TestClient(server.svc).get("/leolabs-status")
    assert resp.json()["mode"] == "misconfigured"


def test_status_live_when_enabled_valid(monkeypatch):
    monkeypatch.setattr(leolabs_runtime, "LEOLABS_ENABLED", True)
    fake_client = MagicMock()
    fake_client.ping.return_value = True
    monkeypatch.setattr(leolabs_runtime, "get_client", lambda: fake_client)
    with patch.dict(os.environ, {"LEOLABS_ACCESS_KEY": "ak", "LEOLABS_SECRET_KEY": "sk"}):
        resp = TestClient(server.svc).get("/leolabs-status")
    data = resp.json()
    assert data["mode"] == "live"
    assert data["label"] == "LEOLABS LIVE"
