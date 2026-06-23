"""
tests/test_udl_evaluate.py

SCRUM-331 AC5: Tests for the UDL conjunction wiring in /v1/evaluate.

Covers:
  - Highest Pc record selected
  - Tie-break on earliest TCA when Pc is equal
  - 422 returned when primary_norad missing and UDL_ENABLED=true
  - 200 with no-maneuver-needed when UDL returns no records
  - covariance_source set to UDL in the response

Run with (from repo root -- conftest.py handles PYTHONPATH):
    python -m pytest services/planner/tests/test_udl_evaluate.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import server


# ---------------------------------------------------------------------------
# Minimal valid request body for /v1/evaluate
# Satellite state and policy are required by evaluate_conjunction_v25().
# ---------------------------------------------------------------------------

_BASE_SATELLITE = {
    "norad_id":        25544,
    "r_sat_km":        [6778.0, 0.0, 0.0],
    "v_sat_km_s":      [0.0, 7.66, 0.0],
    "t_burn_utc":      "2026-06-22T08:00:00Z",
    "v_remaining_m_s": 50.0,
    "mass_kg":         100.0,
}

_BASE_POLICY = {
    "operator_id":      "TEST",
    "policy_version":   "2.5.0",
    "lambda_v":         0.01,
    "lambda_L":         0.001,
    "dv_mag_limit_m_s": 0.5,
    "a_ref_km":         6778.0,
}

# Minimal UDL conjunction record as returned by _parse_conjunction()
def _udl_record(obj_id: str, pc: float, tca: str):
    return {
        "obj_id":           obj_id,
        "t_ca_utc":         tca,
        "r_rel_km":         [0.1, 0.2, 0.3],
        "p_rel_km2":        [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4],
        "pc_precomputed":   pc,
        "miss_distance_km": 0.5,
        "r_sat_km":         [6778.0, 0.0, 0.0],
        "v_sat_km_s":       [0.0, 7.66, 0.0],
        "satNo1":           25544,
        "satNo2":           99001,
        "udl_id":           obj_id,
        "primary_norad":    25544,
        "secondary_norad":  99001,
        "covariance_source": "UDL",
    }


def _body(primary_norad=25544):
    body = {
        "conjunction_id": "test-conj-001",
        "satellite":      _BASE_SATELLITE,
        "policy":         _BASE_POLICY,
        "conjunction": {
            "primary_norad":   primary_norad,
            "secondary_norad": 99001,
            "t_ca_utc":        "2026-06-22T10:00:00Z",
            "r_rel_km":        [0.1, 0.2, 0.3],
            "p_rel_km2":       [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4],
        },
    }
    return body


# ---------------------------------------------------------------------------
# AC5 evaluate-path tests
# ---------------------------------------------------------------------------

class TestUDLEvaluatePath:

    def test_highest_pc_record_selected(self, monkeypatch):
        """When multiple records returned, highest Pc wins."""
        records = [
            _udl_record("A", pc=0.001, tca="2026-06-22T10:00:00Z"),
            _udl_record("B", pc=0.001, tca="2026-06-22T09:00:00Z"),
            _udl_record("C", pc=0.005, tca="2026-06-22T11:00:00Z"),
        ]
        monkeypatch.setattr(server, "UDL_ENABLED", True)
        monkeypatch.setattr(server, "get_conjunctions", lambda sat_no: records)

        client = TestClient(server.svc)
        resp = client.post("/v1/evaluate", json=_body())
        assert resp.status_code == 200
        assert resp.json()["udl_record_id"] == "C"

    def test_tiebreak_on_earliest_tca(self, monkeypatch):
        """When Pc is equal, earliest TCA wins."""
        records = [
            _udl_record("A", pc=0.001, tca="2026-06-22T10:00:00Z"),
            _udl_record("B", pc=0.001, tca="2026-06-22T09:00:00Z"),
        ]
        monkeypatch.setattr(server, "UDL_ENABLED", True)
        monkeypatch.setattr(server, "get_conjunctions", lambda sat_no: records)

        client = TestClient(server.svc)
        resp = client.post("/v1/evaluate", json=_body())
        assert resp.status_code == 200
        assert resp.json()["udl_record_id"] == "B"

    def test_missing_primary_norad_returns_422(self, monkeypatch):
        """UDL_ENABLED=true with missing primary_norad must return 422."""
        monkeypatch.setattr(server, "UDL_ENABLED", True)
        monkeypatch.setattr(server, "get_conjunctions", lambda sat_no: [])

        body = _body()
        del body["conjunction"]["primary_norad"]

        client = TestClient(server.svc)
        resp = client.post("/v1/evaluate", json=body)
        assert resp.status_code == 422

    def test_empty_records_returns_no_maneuver(self, monkeypatch):
        """UDL returning no records above threshold must return 200 with no-maneuver."""
        monkeypatch.setattr(server, "UDL_ENABLED", True)
        monkeypatch.setattr(server, "get_conjunctions", lambda sat_no: [])

        client = TestClient(server.svc)
        resp = client.post("/v1/evaluate", json=_body())
        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendation"]["direction"] == "no_maneuver_needed"

    def test_covariance_source_set_to_udl(self, monkeypatch):
        """Response covariance_source must be UDL in live mode."""
        records = [_udl_record("X", pc=0.003, tca="2026-06-22T10:00:00Z")]
        monkeypatch.setattr(server, "UDL_ENABLED", True)
        monkeypatch.setattr(server, "get_conjunctions", lambda sat_no: records)

        client = TestClient(server.svc)
        resp = client.post("/v1/evaluate", json=_body())
        assert resp.status_code == 200
        assert resp.json()["covariance_source"] == "UDL"
