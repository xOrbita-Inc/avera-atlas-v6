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
