"""
SCRUM-432 -- the TIROS 4 E2E smoke test evaluates against the real stored CDM.

The smoke test lives in the dashboard and runs in a browser, so nothing in the
suite guarded its wiring. It injected the real TIROS 4 (226) / IRIDIUM 33 DEB
(35929) CDM, then evaluated a conjunction block that named the secondary as
'IRIDIUM 33 DEB' -- a name, not a NORAD -- and carried a hand-rolled p_rel_km2.
The adapter resolved GET /cdm//IRIDIUM 33 DEB, found nothing, and fell back to
the surrogate, which then overwrote the supplied matrix. Thirteen steps passed
and not one of them touched the real covariance.

These tests pin the wiring the fix depends on: the NORAD pair is what routes an
evaluate to the stored CDM. The ingest read is mocked, so what is under test is
the routing, not whether the store happens to be up. The control -- the same
request without the pair -- is what makes that non-vacuous: it reproduces the
bug on demand, so a regression that stops sending the pair fails here.

Run with (from repo root -- conftest.py handles PYTHONPATH):
    python -m pytest services/planner/tests/test_tiros_e2e_real_covariance.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server

# The injected pair. These two strings are the whole ticket.
PRIMARY_NORAD = "226"
SECONDARY_NORAD = "35929"

# The row the store returns for that pair, with the id the audit write links to.
STORED_CDM_ID = 7

_NOW = datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# The TIROS 4 and IRIDIUM 33 DEB state vectors the smoke test sends, straight
# from the CDM it injects in Step 1.
_R_TIROS = [-1484.865223, -5293.446853, -4495.437378]
_V_TIROS = [6.464033802, 0.661818202, -3.00266646]
_R_DEBRIS = [-1484.285818, -5292.263622, -4494.23968]
_R_REL = [a - b for a, b in zip(_R_TIROS, _R_DEBRIS)]


def _smoke_body(with_pair: bool = True, use_stored_cdm: bool = True) -> dict:
    """The smoke test's /v1/evaluate request.

    with_pair False is the pre-432 shape: the secondary named only by obj_id,
    which is what sent the adapter looking for a NORAD it never had.
    """
    conjunction = {
        "obj_id": "IRIDIUM 33 DEB",
        "t_ca_utc": _iso(_NOW + timedelta(hours=3)),
        "r_rel_km": _R_REL,
    }
    if with_pair:
        conjunction["primary_norad"] = PRIMARY_NORAD
        conjunction["secondary_norad"] = SECONDARY_NORAD
        if use_stored_cdm:
            conjunction["use_stored_cdm"] = True

    return {
        "conjunction_id": "TIROS4-SMOKE-E2E",
        "satellite": {
            "sat_id": "TIROS 4",
            "r_sat_km": _R_TIROS,
            "v_sat_km_s": _V_TIROS,
            "t_burn_utc": _iso(_NOW - timedelta(hours=1)),
            "v_remaining_m_s": 45.0,
        },
        "conjunction": conjunction,
        "policy": {
            "lambda_v": 1.0,
            "lambda_L": 0.8,
            "dv_mag_limit_m_s": 2.0,
            "a_ref_km": 7000.0,
        },
    }


def _ingest_get(url, *args, **kwargs):
    """Stand in for the ingest reads.

    Only the injected pair resolves. Any other pair 404s exactly as the real
    store does, so a request that fails to send the pair cannot accidentally
    pass by hitting a permissive mock.
    """
    if "/cdm/" in url:
        if url.endswith(f"/cdm/{PRIMARY_NORAD}/{SECONDARY_NORAD}"):
            response = MagicMock(status_code=200, ok=True)
            response.json.return_value = {
                "id": STORED_CDM_ID,
                # What the ingest _SOURCE_MAP returns for an injected
                # source='real_cdm' row.
                "covariance_source": "real_cdm",
                "covariance_combined_rtn": np.diag([1e-4, 1e-4, 1e-4]).tolist(),
            }
            return response
        return MagicMock(status_code=404, ok=False)

    response = MagicMock(status_code=200, ok=True)
    response.json.return_value = {"next_seq": 0, "content_hash": "0" * 64}
    return response


def _evaluate(body: dict):
    """Drive the real endpoint. Returns (response, captured posts)."""
    posts = MagicMock(return_value=MagicMock(status_code=201))
    with patch.object(server, "UDL_ENABLED", False), \
         patch.object(server, "LEOLABS_ENABLED", False), \
         patch.object(server.http_requests, "post", posts), \
         patch.object(server.http_requests, "get", side_effect=_ingest_get):
        response = TestClient(server.svc).post("/v1/evaluate", json=body)
    return response, posts


def _audit_payload(posts: MagicMock) -> dict | None:
    """The planner_output audit write, if one was made."""
    for call in posts.call_args_list:
        url = call.args[0] if call.args else call.kwargs.get("url", "")
        if url.endswith("/planner_output"):
            return call.kwargs.get("json")
    return None


# ---------------------------------------------------------------------------
# With the pair: the stored CDM drives the evaluate
# ---------------------------------------------------------------------------


class TestPairResolvesTheStoredCdm:
    def test_the_evaluate_succeeds(self):
        response, _ = _evaluate(_smoke_body())

        assert response.status_code == 200
        assert "recommendation" in response.json()

    def test_it_reports_real_cdm(self):
        """The assertion the smoke test now makes in the browser."""
        response, _ = _evaluate(_smoke_body())

        assert response.json()["covariance_source"] == "real_cdm"

    def test_it_asks_the_store_for_the_injected_pair(self):
        """Routing, not luck: the adapter must request exactly 226/35929."""
        requested = []

        def _record(url, *args, **kwargs):
            requested.append(url)
            return _ingest_get(url, *args, **kwargs)

        with patch.object(server, "UDL_ENABLED", False), \
             patch.object(server, "LEOLABS_ENABLED", False), \
             patch.object(server.http_requests, "post",
                          MagicMock(return_value=MagicMock(status_code=201))), \
             patch.object(server.http_requests, "get", side_effect=_record):
            TestClient(server.svc).post("/v1/evaluate", json=_smoke_body())

        assert any(
            u.endswith(f"/cdm/{PRIMARY_NORAD}/{SECONDARY_NORAD}") for u in requested
        ), f"never requested the injected pair; asked for {requested}"

    def test_the_audit_write_links_the_stored_cdm(self):
        """cdm_record_id non-None is what makes the decision traceable back to
        the CDM it was made from. The audit write is guarded on it, so a None
        here means no record at all, not merely an unlinked one."""
        _, posts = _evaluate(_smoke_body())

        payload = _audit_payload(posts)
        assert payload is not None, "no planner_output audit write was made"
        assert payload["cdm_record_id"] is not None
        assert payload["cdm_record_id"] == STORED_CDM_ID

    def test_the_audit_write_records_the_real_source(self):
        _, posts = _evaluate(_smoke_body())

        assert _audit_payload(posts)["covariance_source"] == "real_cdm"

    def test_the_scored_covariance_is_not_the_surrogate(self):
        """The adapter replaces whatever covariance the request carried. With
        the pair it must be the store's, so it must differ from what the
        surrogate would have produced for this same orbit."""
        surrogate, _, _ = server._surrogate_covariance(_R_TIROS, _V_TIROS)
        response, _ = _evaluate(_smoke_body())

        assert response.json()["covariance_source"] == "real_cdm"
        # The store row is a 1e-4 km2 diagonal; the surrogate is kilometre-scale.
        assert not np.allclose(
            np.array(surrogate).reshape(3, 3),
            np.diag([1e-4, 1e-4, 1e-4]),
        )

    def test_pc_is_computed_rather_than_pinned(self):
        """The old request pinned pc_precomputed 0.0. Dropping it means the Pc
        reported is the one the real covariance produced."""
        response, _ = _evaluate(_smoke_body())

        metrics = response.json()["metrics"]
        assert metrics["m2_pre"] > 0


# ---------------------------------------------------------------------------
# The control: the pre-432 request shape, which is the bug
# ---------------------------------------------------------------------------


class TestNoPairFallsBackToSurrogate:
    def test_a_name_only_secondary_reports_a_surrogate(self):
        """Reproduces the bug this ticket fixes. Without the pair the adapter
        has no NORAD to resolve and falls back, so the evaluate never touches
        the injected CDM -- while still returning 200 and looking healthy."""
        response, _ = _evaluate(_smoke_body(with_pair=False))

        assert response.status_code == 200
        assert response.json()["covariance_source"] != "real_cdm"
        assert "surrogate" in response.json()["covariance_source"]

    def test_the_control_writes_no_linked_audit_record(self):
        """A surrogate evaluate has no cdm_record_id, and the audit write is
        guarded on it, so the decision is recorded against no CDM at all."""
        _, posts = _evaluate(_smoke_body(with_pair=False))

        assert _audit_payload(posts) is None

    @pytest.mark.parametrize("with_pair,expected", [(True, "real_cdm"), (False, "surrogate_elliptical")])
    def test_the_pair_is_the_only_difference(self, with_pair, expected):
        """Same body, same store, same mock. The NORAD pair alone decides
        whether the real covariance is used."""
        response, _ = _evaluate(_smoke_body(with_pair=with_pair))

        assert response.json()["covariance_source"] == expected


# ---------------------------------------------------------------------------
# use_stored_cdm, and what it must not do to the live path
# ---------------------------------------------------------------------------


def _evaluate_with_leolabs_live(body: dict, fetch):
    """Drive the endpoint with LeoLabs enabled, so the SCRUM-420 switch is live."""
    posts = MagicMock(return_value=MagicMock(status_code=201))
    with patch.object(server, "UDL_ENABLED", False), \
         patch.object(server, "LEOLABS_ENABLED", True), \
         patch.object(server, "fetch_leolabs_conjunction", fetch), \
         patch.object(server.http_requests, "post", posts), \
         patch.object(server.http_requests, "get", side_effect=_ingest_get):
        return TestClient(server.svc).post("/v1/evaluate", json=body)


class TestStoredCdmOptOut:
    """SCRUM-432: naming a NORAD pair has meant 'go ask LeoLabs' since 420, and
    TIROS 4 is not a subscribed asset, so the pair alone cannot reach the store
    while LeoLabs is on. The opt-out has to buy that without loosening the live
    path -- an unsubscribed NORAD must still fail loudly rather than quietly
    scoring on something else."""

    def test_without_the_opt_out_an_unsubscribed_primary_still_fails(self):
        """The 503 this ticket must not trade away. Nothing that omits the flag
        may reach the store by accident."""
        def _unsubscribed(_norad):
            raise ValueError(
                "NORAD 226 is not in the LeoLabs subscribed-objects registry"
            )

        response = _evaluate_with_leolabs_live(
            _smoke_body(use_stored_cdm=False), _unsubscribed
        )

        assert response.status_code == 503

    def test_the_opt_out_reaches_the_store_instead(self):
        """Same request, same disabled asset, flag set: it scores the stored CDM."""
        def _must_not_run(_norad):
            raise AssertionError(
                "use_stored_cdm must keep the request off the LeoLabs fetch"
            )

        response = _evaluate_with_leolabs_live(_smoke_body(), _must_not_run)

        assert response.status_code == 200
        assert response.json()["covariance_source"] == "real_cdm"

    def test_it_does_not_disturb_a_live_leolabs_request(self):
        """A request that does not set the flag is routed exactly as before.

        Which of the two live entry points it takes depends on whether the
        block carries a selector, so this asserts that the LeoLabs branch is
        entered at all rather than pinning one of them.
        """
        reached = []

        def _record(*args, **kwargs):
            reached.append(args)
            raise ValueError("live fetch reached")

        body = _smoke_body(use_stored_cdm=False)
        body["conjunction"]["primary_norad"] = "36508"
        with patch.object(server, "fetch_leolabs_conjunctions", _record), \
             patch.object(server, "select_conjunction", _record):
            _evaluate_with_leolabs_live(body, _record)

        assert reached, "the live LeoLabs branch must still be reached"

    def test_the_flag_is_inert_when_leolabs_is_off(self):
        """It only suppresses a fetch that would otherwise happen; with LeoLabs
        off there is nothing to suppress and the result is unchanged."""
        with_flag, _ = _evaluate(_smoke_body())
        without_flag, _ = _evaluate(_smoke_body(use_stored_cdm=False))

        assert with_flag.json()["covariance_source"] == "real_cdm"
        assert without_flag.json()["covariance_source"] == "real_cdm"
