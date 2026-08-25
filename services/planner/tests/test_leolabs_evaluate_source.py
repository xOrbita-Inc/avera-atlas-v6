"""tests/test_leolabs_evaluate_source.py

SCRUM-411 AC6: the per-evaluate `source` field on /v1/evaluate.

`source` is the conjunction data source that drove the evaluate, distinct from
`covariance_source` (the provenance of the covariance matrix). It lights the
dashboard LIVE badge per evaluate. A request assembled from a parsed LeoLabs CDM
carries source="leolabs"; absent an explicit source the planner derives one from
covariance_source.

Run from repo root:
    python -m pytest services/planner/tests/test_leolabs_evaluate_source.py -v
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import server

_BASE_SATELLITE = {
    "norad_id": 25544,
    "r_sat_km": [6778.0, 0.0, 0.0],
    "v_sat_km_s": [0.0, 7.66, 0.0],
    "t_burn_utc": "2026-06-22T08:00:00Z",
    "v_remaining_m_s": 50.0,
    "mass_kg": 100.0,
}

_BASE_POLICY = {
    "operator_id": "TEST",
    "policy_version": "2.5.6",
    "lambda_v": 0.01,
    "lambda_L": 0.001,
    "dv_mag_limit_m_s": 0.5,
    "a_ref_km": 6778.0,
}


def _body(**conj_extra):
    conj = {
        "obj_id": "L143957",
        "t_ca_utc": "2026-06-22T10:00:00Z",
        "r_rel_km": [0.1, 0.2, 0.3],
        "v_rel_km_s": [0.0, -0.5, 0.1],
        "p_rel_km2": [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4],
    }
    conj.update(conj_extra)
    return {
        "conjunction_id": "test-conj-001",
        "satellite": _BASE_SATELLITE,
        "policy": _BASE_POLICY,
        "conjunction": conj,
    }


# ---------------------------------------------------------------------------
# End-to-end through /v1/evaluate
# ---------------------------------------------------------------------------

def test_response_always_carries_source(monkeypatch):
    """Every evaluate response includes a source field."""
    monkeypatch.setattr(server, "UDL_ENABLED", False)
    resp = TestClient(server.svc).post("/v1/evaluate", json=_body())
    assert resp.status_code == 200
    assert "source" in resp.json()


def test_no_norad_defaults_source_to_surrogate(monkeypatch):
    """No primary_norad -> surrogate covariance -> source surrogate (not live)."""
    monkeypatch.setattr(server, "UDL_ENABLED", False)
    resp = TestClient(server.svc).post("/v1/evaluate", json=_body())
    assert resp.status_code == 200
    assert resp.json()["source"] == "surrogate"


def test_explicit_leolabs_source_is_echoed(monkeypatch):
    """A request assembled from a parsed LeoLabs CDM reports source=leolabs."""
    monkeypatch.setattr(server, "UDL_ENABLED", False)
    resp = TestClient(server.svc).post(
        "/v1/evaluate", json=_body(source="leolabs")
    )
    assert resp.status_code == 200
    assert resp.json()["source"] == "leolabs"


def test_explicit_source_wins_over_covariance_derivation(monkeypatch):
    """Even with a real CDM covariance, an explicit leolabs source wins."""
    monkeypatch.setattr(server, "UDL_ENABLED", False)
    monkeypatch.setattr(
        server,
        "_fetch_cdm_covariance",
        lambda *a, **k: ([1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4], "real_cdm", 42),
    )
    resp = TestClient(server.svc).post(
        "/v1/evaluate",
        json=_body(source="LEOLABS", primary_norad="36508", secondary_norad="270302"),
    )
    assert resp.status_code == 200
    assert resp.json()["source"] == "leolabs"


def test_udl_path_reports_source_udl(monkeypatch):
    record = {
        "obj_id": "X", "t_ca_utc": "2026-06-22T10:00:00Z",
        "r_rel_km": [0.1, 0.2, 0.3], "v_rel_km_s": [0.0, -0.5, 0.1],
        "p_rel_km2": [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4],
        "pc_precomputed": 0.003, "miss_distance_km": 0.5,
        "r_sat_km": [6778.0, 0.0, 0.0], "v_sat_km_s": [0.0, 7.66, 0.0],
        "satNo1": 25544, "satNo2": 99001,
    }
    monkeypatch.setattr(server, "UDL_ENABLED", True)
    monkeypatch.setattr(server, "get_conjunctions", lambda sat_no: [record])
    body = _body()
    body["conjunction"]["primary_norad"] = 25544
    resp = TestClient(server.svc).post("/v1/evaluate", json=body)
    assert resp.status_code == 200
    assert resp.json()["source"] == "udl"


def test_udl_no_records_early_return_has_source(monkeypatch):
    monkeypatch.setattr(server, "UDL_ENABLED", True)
    monkeypatch.setattr(server, "get_conjunctions", lambda sat_no: [])
    body = _body()
    body["conjunction"]["primary_norad"] = 25544
    resp = TestClient(server.svc).post("/v1/evaluate", json=body)
    assert resp.status_code == 200
    assert resp.json()["source"] == "udl"


# ---------------------------------------------------------------------------
# _resolve_evaluate_source unit coverage
# ---------------------------------------------------------------------------

def test_resolver_maps_real_cdm_to_spacetrack():
    assert server._resolve_evaluate_source({}, "real_cdm", False) == "spacetrack"


def test_resolver_maps_surrogate_variants():
    assert server._resolve_evaluate_source({}, "surrogate_elliptical", False) == "surrogate"
    assert server._resolve_evaluate_source({}, "surrogate_identity", False) == "surrogate"


def test_resolver_udl_used_wins_when_no_explicit():
    assert server._resolve_evaluate_source({}, "real_cdm", True) == "udl"


def test_resolver_explicit_normalizes_spacetrack_spelling():
    body = {"conjunction": {"source": "Space-Track"}}
    assert server._resolve_evaluate_source(body, "surrogate_elliptical", False) == "spacetrack"


def test_resolver_live_sources_set_is_accurate():
    for s in ("leolabs", "udl", "spacetrack"):
        assert s in server._LIVE_SOURCES
    assert "surrogate" not in server._LIVE_SOURCES
