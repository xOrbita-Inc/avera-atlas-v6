"""tests/test_leolabs_evaluate_e2e.py

SCRUM-411 AC7: drive /v1/evaluate on a real LeoLabs CDM (the saved fixture) and
produce a recommendation on real per-object covariance -- offline.

The live-API version of this run lives in test_leolabs_live_smoke.py (opt-in).
Here the CDM is the saved fixture, so this exercises the full glue -- parse,
build the request, evaluate -- with no network, and asserts the real covariance
actually reaches the scorer rather than being overwritten by the covariance
adapter.

Run from repo root:
    python -m pytest services/planner/tests/test_leolabs_evaluate_e2e.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from common.leolabs_cdm_parser import parse_leolabs_cdm
from common.leolabs_evaluate import build_evaluate_request

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"
_OUR_ID = "L2669"


@pytest.fixture
def parsed():
    cdm = json.loads(_FIXTURE.read_text())
    return parse_leolabs_cdm(cdm, _OUR_ID)


# ---------------------------------------------------------------------------
# Request assembly
# ---------------------------------------------------------------------------

def test_build_request_uses_primary_state_and_leolabs_source(parsed):
    req = build_evaluate_request(parsed, sat_id="XORB-001", v_remaining_m_s=25.0)

    # Satellite block is the primary (our asset) at TCA.
    np.testing.assert_allclose(
        req["satellite"]["r_sat_km"], parsed.primary.r_km, rtol=1e-12
    )
    np.testing.assert_allclose(
        req["satellite"]["v_sat_km_s"], parsed.primary.v_km_s, rtol=1e-12
    )
    assert req["satellite"]["radius_m"] == pytest.approx(parsed.primary.radius_m)
    # t_burn defaults to a lead before TCA (the scorer requires burn < TCA).
    from datetime import datetime
    t_burn = datetime.fromisoformat(req["satellite"]["t_burn_utc"].replace("Z", "+00:00"))
    t_ca = datetime.fromisoformat(parsed.t_ca_utc.replace("Z", "+00:00"))
    assert t_burn < t_ca

    conj = req["conjunction"]
    assert conj["source"] == "leolabs"
    assert conj["covariance_source"] == "real_cdm"
    assert len(conj["p_rel_km2"]) == 9
    assert conj["pc_precomputed"] is None
    assert conj["primary_norad"] == "36508"
    assert conj["secondary_norad"] == "270302"
    assert req["conjunction_id"].startswith("leolabs-")


def test_build_request_respects_operator_overrides(parsed):
    policy = {
        "operator_id": "OPS", "policy_version": "2.5.6",
        "lambda_v": 0.02, "lambda_L": 0.002, "dv_mag_limit_m_s": 1.0,
    }
    req = build_evaluate_request(
        parsed, sat_id="S", v_remaining_m_s=40.0,
        t_burn_utc="2026-08-30T09:00:00Z", a_ref_km=7080.0, policy=policy,
        conjunction_id="run-1",
    )
    assert req["satellite"]["t_burn_utc"] == "2026-08-30T09:00:00Z"
    assert req["satellite"]["a_ref_km"] == 7080.0
    assert req["policy"] == policy
    assert req["conjunction_id"] == "run-1"


# ---------------------------------------------------------------------------
# End-to-end through /v1/evaluate
# ---------------------------------------------------------------------------

def test_evaluate_on_real_cdm_produces_recommendation(parsed, monkeypatch):
    monkeypatch.setattr(server, "UDL_ENABLED", False)
    req = build_evaluate_request(parsed, sat_id="XORB-001", v_remaining_m_s=25.0)

    resp = TestClient(server.svc).post("/v1/evaluate", json=req)
    assert resp.status_code == 200
    data = resp.json()

    assert data["source"] == "leolabs"
    assert data["covariance_source"] == "real_cdm"
    # A recommendation is produced (AC7). With a ~20 km miss and Pc ~4e-13 the
    # correct call is no maneuver, but the field must be present and populated.
    assert data["recommendation"]["direction"]


def test_real_covariance_reaches_scorer_not_overwritten(parsed, monkeypatch):
    """The AC7 bypass must skip the ingest covariance adapter entirely."""
    monkeypatch.setattr(server, "UDL_ENABLED", False)

    called = {"fetch": False}

    def _boom(*a, **k):
        called["fetch"] = True
        raise AssertionError("_fetch_cdm_covariance must not run for a leolabs source")

    monkeypatch.setattr(server, "_fetch_cdm_covariance", _boom)

    req = build_evaluate_request(parsed, sat_id="XORB-001", v_remaining_m_s=25.0)
    sent_cov = list(req["conjunction"]["p_rel_km2"])

    resp = TestClient(server.svc).post("/v1/evaluate", json=req)
    assert resp.status_code == 200
    assert called["fetch"] is False
    # The covariance we sent is the parser's real rotated matrix, unchanged.
    np.testing.assert_allclose(
        sent_cov, np.array(parsed.p_rel_eci_km2()).flatten(), rtol=1e-12
    )


def test_leolabs_source_without_covariance_falls_through(monkeypatch):
    """Guard: source=leolabs but no p_rel_km2 must NOT bypass; adapter runs."""
    monkeypatch.setattr(server, "UDL_ENABLED", False)
    seen = {"fetch": False}
    real = server._fetch_cdm_covariance

    def _spy(*a, **k):
        seen["fetch"] = True
        return real(*a, **k)

    monkeypatch.setattr(server, "_fetch_cdm_covariance", _spy)

    body = {
        "conjunction_id": "x",
        "satellite": {
            "sat_id": "S", "r_sat_km": [6778.0, 0, 0], "v_sat_km_s": [0, 7.66, 0],
            "t_burn_utc": "2026-08-30T09:00:00Z", "v_remaining_m_s": 25.0,
        },
        "conjunction": {
            "obj_id": "L143957", "source": "leolabs",
            "t_ca_utc": "2026-08-30T10:24:48Z",
            "r_rel_km": [0.1, 0.2, 0.3], "v_rel_km_s": [0, -0.5, 0.1],
            # no p_rel_km2 -> bypass condition not met
        },
        "policy": {
            "operator_id": "O", "policy_version": "2.5.6",
            "lambda_v": 0.01, "lambda_L": 0.001, "dv_mag_limit_m_s": 0.5,
        },
    }
    resp = TestClient(server.svc).post("/v1/evaluate", json=body)
    assert resp.status_code == 200
    assert seen["fetch"] is True  # adapter ran because covariance was absent


# SCRUM-480: controlled inputs and offline CDM fixtures, never live measurements.

@pytest.mark.parametrize("candidate_utility", [-2.0, 0.0, 2.0])
def test_480_scores_once_and_prices_forced_burn(monkeypatch, candidate_utility):
    from copy import deepcopy
    from types import SimpleNamespace
    from unittest.mock import Mock
    from common import maneuver_scorer as scorer

    req = {
        "conjunction_id": "480-controlled",
        "policy": {"decision_mode": "flight_rule_1e4"},
        "conjunction": {"miss_distance_km": 2.0},
    }
    original = deepcopy(req)
    candidate = SimpleNamespace(
        direction="prograde", dv_avoid_m_s=0.4,
        utility=candidate_utility, utility_basis="pc_traded",
    )
    scoring = SimpleNamespace(
        conjunction_id="480-controlled", pc_pre=2e-4, pc_source="supplied",
        utility=max(0.0, candidate_utility), candidates_v25=[candidate],
        no_go_reason_code="no_utility_gain" if candidate_utility <= 0 else "",
    )
    spy = Mock(return_value=scoring)
    monkeypatch.setattr(scorer, "evaluate_conjunction_v25", spy)

    result = scorer.analyze_conjunction_modes(req)
    spy.assert_called_once()
    assert spy.call_args.args[0]["policy"]["decision_mode"] == "aps"
    assert req == original
    assert result["best_burn"]["utility"] == candidate_utility
    assert result["modes"]["aps"]["maneuver_required"] == (candidate_utility > 0)
    assert result["modes"]["aps"]["dv_m_s"] == (0.4 if candidate_utility > 0 else 0.0)
    for mode in ("flight_rule_1e4", "flight_rule_1e5"):
        assert result["modes"][mode]["maneuver_required"] is True
        assert result["modes"][mode]["dv_m_s"] == 0.4


def test_480_missing_candidate_is_unpriced(monkeypatch):
    from types import SimpleNamespace
    from common import maneuver_scorer as scorer

    scoring = SimpleNamespace(
        conjunction_id="480-unpriced", pc_pre=2e-4, pc_source="supplied",
        utility=0.0, candidates_v25=[], no_go_reason_code="insufficient_budget",
    )
    monkeypatch.setattr(scorer, "evaluate_conjunction_v25", lambda req: scoring)
    result = scorer.analyze_conjunction_modes({
        "policy": {}, "conjunction": {"miss_distance_km": 2.0},
    })
    assert result["modes"]["aps"]["dv_m_s"] == 0.0
    for mode in ("flight_rule_1e4", "flight_rule_1e5"):
        assert result["modes"][mode]["dv_m_s"] is None
        assert result["modes"][mode]["pricing_available"] is False


@pytest.mark.parametrize("pc,miss", [(5e-5, 2.0), (2e-4, 2.0), (5e-6, 0.5)])
def test_480_real_scorer_verdict_parity(parsed, pc, miss):
    from copy import deepcopy
    from common.maneuver_scorer import (
        analyze_conjunction_modes, evaluate_conjunction_v25, _policy_from_dict,
    )

    req = build_evaluate_request(parsed, sat_id="480", v_remaining_m_s=25.0)
    # Deliberately supplied comparison inputs; these are not measured Pc values.
    req["conjunction"]["pc_precomputed"] = pc
    req["conjunction"]["miss_distance_km"] = miss
    comparison = analyze_conjunction_modes(req)
    for mode, verdict in comparison["modes"].items():
        mode_req = deepcopy(req)
        mode_req["policy"]["decision_mode"] = mode
        scoring = evaluate_conjunction_v25(mode_req)
        policy = _policy_from_dict(mode_req["policy"])
        expected = policy.is_recommendation_required(
            scoring.pc_pre, miss, utility=scoring.utility,
        )
        assert verdict["maneuver_required"] == expected
        if expected and scoring.utility > 0:
            assert verdict["dv_m_s"] == pytest.approx(scoring.dv_magnitude_m_s)


@pytest.mark.parametrize("case", ["complete", "empty", "fetch_partial", "deadline", "parse_failure"])
def test_480_window_totals_and_coverage(parsed, monkeypatch, case):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from common import leolabs_runtime as runtime
    from common import leolabs_cdm_parser as parser
    from common.leolabs_evaluate import analyze_leolabs_portfolio

    cdms = [] if case == "empty" else [{"CDM_ID": "controlled-1"}, {"CDM_ID": "controlled-2"}]
    pull = SimpleNamespace(
        complete=case != "fetch_partial",
        reason="cap" if case == "fetch_partial" else None,
        window_total=10 if case == "fetch_partial" else len(cdms),
    )
    fetch = Mock(return_value=("L2669", cdms, len(cdms), pull))
    monkeypatch.setattr(runtime, "_raw_cdms_in_risk_order", fetch)
    if case == "parse_failure":
        def fail_parse(*args):
            raise parser.LeoLabsParseError("controlled malformed CDM")
        monkeypatch.setattr(parser, "parse_leolabs_cdm", fail_parse)
    else:
        monkeypatch.setattr(parser, "parse_leolabs_cdm", lambda *args: parsed)
    registry = SimpleNamespace(resolve_our_catalog_id=lambda cdm: _OUR_ID)
    result = analyze_leolabs_portfolio(
        36508, client=object(), registry=registry,
        satellite={"sat_id": "480", "v_remaining_m_s": 25.0},
        processing_budget_s=0.0 if case == "deadline" else 25.0,
    )
    assert fetch.call_args.kwargs["deadline_s"] == runtime._LIST_FETCH_DEADLINE_S
    assert fetch.call_args.kwargs["max_cdms"] == runtime._LIST_FETCH_MAX_CDMS
    assert result["complete"] == (case in ("complete", "empty"))
    assert result["partial"] == (not result["complete"])
    assert result["count"] + len(result["failures"]) + result["events_not_attempted"] == len(cdms)
    assert result["cost"]["scoring_calls"] == result["count"]
    assert result["cost"]["total_ms"] >= 0
    for mode, summary in result["modes"].items():
        verdicts = [row["modes"][mode] for row in result["conjunctions"]]
        assert summary["maneuver_count"] == sum(v["maneuver_required"] for v in verdicts)
        known = sum(v["dv_m_s"] for v in verdicts if v["dv_m_s"] is not None)
        assert summary["known_dv_m_s"] == pytest.approx(known)
        missing = sum(not v["pricing_available"] for v in verdicts)
        assert summary["unpriced_maneuver_count"] == missing
        assert summary["total_dv_m_s"] == (None if missing else known)
    if case == "fetch_partial":
        assert "fetch_cap" in result["partial_reasons"]
    if case == "deadline":
        assert result["events_not_attempted"] == len(cdms)
        assert "processing_deadline" in result["partial_reasons"]
    if case == "parse_failure":
        assert len(result["failures"]) == len(cdms)
        assert "event_failures" in result["partial_reasons"]


def test_480_http_matches_evaluate_without_emission(parsed, monkeypatch):
    import threading
    from types import SimpleNamespace
    from unittest.mock import Mock
    from common import leolabs_runtime as runtime
    from common import leolabs_cdm_parser as parser
    from common import maneuver_scorer as scorer
    from common.leolabs_evaluate import analyze_leolabs_portfolio

    monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
    monkeypatch.setattr(server, "UDL_ENABLED", False)
    monkeypatch.setattr(server, "_list_inflight", 0)
    monkeypatch.setattr(runtime, "get_client", lambda: object())
    registry = SimpleNamespace(resolve_our_catalog_id=lambda cdm: _OUR_ID)
    monkeypatch.setattr(runtime, "get_registry", lambda client: registry)
    pull = SimpleNamespace(complete=True, reason=None, window_total=1)
    monkeypatch.setattr(
        runtime, "_raw_cdms_in_risk_order",
        lambda *args, **kwargs: ("L2669", [{"CDM_ID": "saved-fixture"}], 1, pull),
    )
    monkeypatch.setattr(parser, "parse_leolabs_cdm", lambda *args: parsed)
    score_spy = Mock(wraps=scorer.evaluate_conjunction_v25)
    monkeypatch.setattr(scorer, "evaluate_conjunction_v25", score_spy)

    worker_threads = []
    async def dispatch(fn, *args, **kwargs):
        event_loop_thread = threading.get_ident()
        result = await original_dispatch(fn, *args, **kwargs)
        assert worker_threads[-1] != event_loop_thread
        return result
    original_dispatch = server._run_list_fetch
    def analysis(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        return analyze_leolabs_portfolio(*args, **kwargs)
    monkeypatch.setattr(server, "_run_list_fetch", dispatch)
    monkeypatch.setattr(server, "analyze_leolabs_portfolio", analysis)

    def forbidden(*args, **kwargs):
        raise AssertionError("portfolio must not enter evaluate or GNC emission")
    with monkeypatch.context() as read_only:
        read_only.setattr(server, "evaluate_conjunction_v25", forbidden)
        read_only.setattr(server, "_emit_to_gnc", forbidden)
        response = TestClient(server.svc).post(
            "/v1/leolabs/portfolio?primary_norad=36508",
            json={"satellite": {"sat_id": "480", "v_remaining_m_s": 25.0}},
        )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["analysis_only"] is True
    assert result["complete"] is True
    assert result["count"] == 1
    assert result["cost"]["scoring_calls"] == 1
    assert score_spy.call_count == 1
    assert server._list_inflight == 0
    verdicts = result["conjunctions"][0]["modes"]

    # Offline live-source path: same saved CDM, fetched by its selector.
    monkeypatch.setattr(
        server, "fetch_leolabs_conjunction_by_cdm_id", lambda *args: parsed,
    )
    monkeypatch.setattr(server, "_run_list_fetch", original_dispatch)
    monkeypatch.setattr(server, "_emit_to_gnc", lambda *args: None)
    client = TestClient(server.svc)
    for mode, verdict in verdicts.items():
        response = client.post("/v1/evaluate", json={
            "conjunction": {
                "primary_norad": "36508", "cdm_id": "saved-fixture",
            },
            "satellite": {"sat_id": "480", "v_remaining_m_s": 25.0},
            "policy": {
                "operator_id": "LEOLABS-AC7", "policy_version": "leolabs",
                "lambda_v": 0.01, "lambda_L": 0.001,
                "dv_mag_limit_m_s": 0.5, "decision_mode": mode,
            },
        })
        assert response.status_code == 200, response.text
        evaluated = response.json()
        required = evaluated["atlas_artifact"]["risk_summary"]["maneuver_required"]
        assert verdict["maneuver_required"] == required
        if required and evaluated["recommendation"]["utility"] > 0:
            assert verdict["dv_m_s"] == pytest.approx(
                evaluated["recommendation"]["dv_magnitude_m_s"],
            )


@pytest.mark.parametrize("case,status", [
    ("disabled", 503), ("busy", 503), ("fetch_failure", 503),
    ("unsubscribed", 404), ("bad_window", 422),
    ("bad_policy", 422), ("bad_body", 422), ("bad_budget", 422),
])
def test_480_http_errors_are_not_empty_windows(monkeypatch, case, status):
    from unittest.mock import Mock
    from common.leolabs_runtime import LeoLabsRuntimeError

    monkeypatch.setattr(server, "LEOLABS_ENABLED", case != "disabled")
    monkeypatch.setattr(server, "_list_inflight", 2 if case == "busy" else 0)
    analysis = Mock(side_effect=RuntimeError("controlled fetch failure"))
    if case == "unsubscribed":
        analysis.side_effect = LeoLabsRuntimeError("controlled unsubscribed asset")
    monkeypatch.setattr(server, "analyze_leolabs_portfolio", analysis)
    body = {}
    query = "?primary_norad=36508"
    if case == "bad_window":
        query += "&lookahead_days=31"
    elif case == "bad_policy":
        body = {"policy": {"decision_mode": "invalid"}}
    elif case == "bad_body":
        body = []
    elif case == "bad_budget":
        body = {"satellite": {"v_remaining_m_s": -1}}
    response = TestClient(server.svc).post(
        "/v1/leolabs/portfolio" + query, json=body,
    )
    assert response.status_code == status, response.text
    assert "conjunctions" not in response.json()
    if case not in ("fetch_failure", "unsubscribed"):
        analysis.assert_not_called()


def test_480_http_rejects_invalid_json(monkeypatch):
    monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
    response = TestClient(server.svc).post(
        "/v1/leolabs/portfolio?primary_norad=36508",
        content="{broken",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_480_real_negative_utility_burn_is_priced(parsed, monkeypatch):
    from copy import deepcopy
    from unittest.mock import Mock
    from common import maneuver_scorer as scorer

    # Controlled policy and supplied Pc exercise forced-burn pricing.
    # They are not a measured portfolio or an operational policy recommendation.
    req = build_evaluate_request(
        parsed, sat_id="480-negative", v_remaining_m_s=25.0,
        policy={
            "operator_id": "480-TEST", "policy_version": "test",
            "lambda_v": 0.01, "lambda_L": 1e12,
            "dv_mag_limit_m_s": 0.5,
            "mission_lifetime_days_total": 1825.0,
        },
    )
    req["conjunction"]["pc_precomputed"] = 2e-4
    req["conjunction"]["miss_distance_km"] = 2.0

    real_score = scorer.evaluate_conjunction_v25
    spy = Mock(wraps=real_score)
    with monkeypatch.context() as once:
        once.setattr(scorer, "evaluate_conjunction_v25", spy)
        result = scorer.analyze_conjunction_modes(req)
    spy.assert_called_once()
    assert result["best_burn"] is not None
    assert result["best_burn"]["utility"] < 0.0
    assert result["best_burn"]["dv_m_s"] > 0.0
    assert result["modes"]["aps"]["maneuver_required"] is False
    assert result["modes"]["aps"]["dv_m_s"] == 0.0

    monkeypatch.setattr(server, "LEOLABS_ENABLED", False)
    monkeypatch.setattr(server, "UDL_ENABLED", False)
    monkeypatch.setattr(server, "_emit_to_gnc", lambda *args: None)
    client = TestClient(server.svc)
    for mode, verdict in result["modes"].items():
        body = deepcopy(req)
        body["policy"]["decision_mode"] = mode
        response = client.post("/v1/evaluate", json=body)
        assert response.status_code == 200, response.text
        evaluated = response.json()
        assert verdict["maneuver_required"] == (
            evaluated["atlas_artifact"]["risk_summary"]["maneuver_required"]
        )
        assert evaluated["recommendation"]["direction"] == "no-burn"
        if mode != "aps":
            assert verdict["maneuver_required"] is True
            assert verdict["dv_m_s"] == result["best_burn"]["dv_m_s"]
            assert verdict["pricing_available"] is True
