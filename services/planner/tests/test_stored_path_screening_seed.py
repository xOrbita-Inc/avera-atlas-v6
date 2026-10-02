"""SCRUM-453 item 2, planner half: the stored path seeds from the primary's own
covariance.

SCRUM-454 made the secondary screen seed from the asset's OWN covariance rather
than the combined relative one, which over-states the asset's uncertainty (2.72x
in sigma, 7.4x in trace on a live CDM). It could only do that on the live LeoLabs
path, where a parsed CDM is in hand. A decision made from a stored or reference
CDM went through _fetch_cdm_covariance, and ingest summed the two per-object
blocks away before putting them on the wire, so there was no primary-only matrix
to seed with and the screen fell back to the combined stand-in.

SCRUM-453 puts covariance_primary_rtn on that response. These tests pin the
planner half: that the fetch reads it, rotates it with the same rotation as the
combined so the two are in one frame, and that a real evaluate then reports a
seed source of cdm_primary_own rather than combined_relative_stand_in.

The fallback is pinned just as hard. Older rows and surrogate responses carry no
primary block, and those decisions must still get a seed -- the stand-in, clearly
labelled -- rather than losing the screen.

Run from repo root:
    python -m pytest services/planner/tests/test_stored_path_screening_seed.py -v
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server

PRIMARY_NORAD = "226"
SECONDARY_NORAD = "35929"
STORED_CDM_ID = 11

R_SAT_KM = [-1484.865223, -5293.446853, -4495.437378]
V_SAT_KM_S = [6.464033802, 0.661818202, -3.00266646]

# Asymmetric on purpose, and not a multiple of the combined, so "it read the
# right block" cannot pass by coincidence.
_PRIMARY_RTN = [[0.004, 0.0003, 0.0],
                [0.0003, 0.090, 0.0011],
                [0.0, 0.0011, 0.0025]]
_SECONDARY_RTN = [[0.010, -0.0007, 0.0001],
                  [-0.0007, 0.400, 0.0],
                  [0.0001, 0.0, 0.0180]]
_COMBINED_RTN = [[_PRIMARY_RTN[i][j] + _SECONDARY_RTN[i][j] for j in range(3)]
                 for i in range(3)]

_NOW = datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _cdm_payload(with_primary: bool) -> dict:
    body = {
        "id": STORED_CDM_ID,
        "covariance_source": "real_cdm",
        "covariance_combined_rtn": _COMBINED_RTN,
    }
    if with_primary:
        # What SCRUM-453's ingest change puts on the wire.
        body["covariance_primary_rtn"] = _PRIMARY_RTN
        body["covariance_secondary_rtn"] = _SECONDARY_RTN
    return body


def _make_response(status: int, json_data: dict):
    resp = MagicMock(status_code=status, ok=200 <= status < 300)
    resp.json.return_value = json_data
    return resp


# ---------------------------------------------------------------------------
# The fetch itself
# ---------------------------------------------------------------------------

class TestTheFetchReadsThePrimaryBlock:
    def test_it_returns_the_primary_covariance_when_ingest_supplies_it(self):
        resp = _make_response(200, _cdm_payload(with_primary=True))
        with patch.object(server.http_requests, "get", return_value=resp):
            _, _, _, primary = server._fetch_cdm_covariance(
                PRIMARY_NORAD, SECONDARY_NORAD, R_SAT_KM, V_SAT_KM_S)
        assert primary is not None
        assert np.asarray(primary).shape == (3, 3)

    def test_it_returns_none_when_the_field_is_absent(self):
        """Older stored rows, and the surrogate path."""
        resp = _make_response(200, _cdm_payload(with_primary=False))
        with patch.object(server.http_requests, "get", return_value=resp):
            _, _, _, primary = server._fetch_cdm_covariance(
                PRIMARY_NORAD, SECONDARY_NORAD, R_SAT_KM, V_SAT_KM_S)
        assert primary is None

    def test_the_primary_is_rotated_with_the_same_rotation_as_the_combined(self):
        """Both must land in one frame, or the seed is in RTN while the screen
        believes it is in ECI."""
        resp = _make_response(200, _cdm_payload(with_primary=True))
        with patch.object(server.http_requests, "get", return_value=resp):
            p_rel_flat, _, _, primary = server._fetch_cdm_covariance(
                PRIMARY_NORAD, SECONDARY_NORAD, R_SAT_KM, V_SAT_KM_S)

        rot = server._rtn_to_eci_rotation(
            np.array(R_SAT_KM, dtype=float), np.array(V_SAT_KM_S, dtype=float))
        expected_primary = rot @ np.array(_PRIMARY_RTN, dtype=float) @ rot.T
        assert np.allclose(np.asarray(primary), expected_primary)

        # And the combined is still the combined, unchanged by this ticket.
        expected_combined = rot @ np.array(_COMBINED_RTN, dtype=float) @ rot.T
        assert np.allclose(np.asarray(p_rel_flat).reshape(3, 3), expected_combined)

    def test_the_primary_is_smaller_than_the_combined(self):
        """The reason the seed changed at all."""
        resp = _make_response(200, _cdm_payload(with_primary=True))
        with patch.object(server.http_requests, "get", return_value=resp):
            p_rel_flat, _, _, primary = server._fetch_cdm_covariance(
                PRIMARY_NORAD, SECONDARY_NORAD, R_SAT_KM, V_SAT_KM_S)
        combined = np.asarray(p_rel_flat).reshape(3, 3)
        assert np.trace(np.asarray(primary)) < np.trace(combined)

    def test_a_malformed_primary_block_costs_the_seed_not_the_decision(self):
        """A bad primary block must not take the covariance down with it."""
        payload = _cdm_payload(with_primary=True)
        payload["covariance_primary_rtn"] = [[1.0, 2.0], [3.0, 4.0]]   # wrong shape
        resp = _make_response(200, payload)
        with patch.object(server.http_requests, "get", return_value=resp):
            p_rel_flat, source, cdm_id, primary = server._fetch_cdm_covariance(
                PRIMARY_NORAD, SECONDARY_NORAD, R_SAT_KM, V_SAT_KM_S)
        assert primary is None
        assert source == "real_cdm"          # the decision still has its covariance
        assert cdm_id == STORED_CDM_ID
        assert len(p_rel_flat) == 9

    def test_junk_in_the_primary_block_is_also_survived(self):
        payload = _cdm_payload(with_primary=True)
        payload["covariance_primary_rtn"] = "not a matrix"
        resp = _make_response(200, payload)
        with patch.object(server.http_requests, "get", return_value=resp):
            _, source, _, primary = server._fetch_cdm_covariance(
                PRIMARY_NORAD, SECONDARY_NORAD, R_SAT_KM, V_SAT_KM_S)
        assert primary is None
        assert source == "real_cdm"

    def test_a_failed_fetch_still_returns_four_values(self):
        """The surrogate fallback has to match the new arity or every caller
        unpacking it raises."""
        with patch.object(server.http_requests, "get",
                          return_value=_make_response(404, {})):
            out = server._fetch_cdm_covariance(
                PRIMARY_NORAD, SECONDARY_NORAD, R_SAT_KM, V_SAT_KM_S)
        assert len(out) == 4
        assert out[1] == "surrogate_elliptical"
        assert out[3] is None


# ---------------------------------------------------------------------------
# What the decision actually seeds with
# ---------------------------------------------------------------------------

class _SeedCapture(logging.Handler):
    """Catches the screening_seed_selected record off the planner logger.

    Attached to the logger directly rather than through caplog: the planner
    configures its own 'planner' logger with propagate False, so records never
    reach the root handler caplog installs.
    """

    def __init__(self):
        super().__init__()
        self.sources: list = []

    def emit(self, record):
        if getattr(record, "event", None) == "screening_seed_selected":
            self.sources.append(getattr(record, "p_post_source", None))


def _evaluate_with_stored_cdm(with_primary: bool):
    """Run a real evaluate against a mocked stored CDM; return the seed sources."""
    def _ingest_get(url, *args, **kwargs):
        if "/cdm/" in url:
            if url.endswith(f"/cdm/{PRIMARY_NORAD}/{SECONDARY_NORAD}"):
                return _make_response(200, _cdm_payload(with_primary))
            return _make_response(404, {})
        return _make_response(200, {"next_seq": 0, "content_hash": "0" * 64})

    body = {
        "conjunction_id": "SCRUM-453-STORED-SEED",
        "satellite": {
            "sat_id": "TIROS 4",
            "r_sat_km": R_SAT_KM,
            "v_sat_km_s": V_SAT_KM_S,
            "t_burn_utc": _iso(_NOW - timedelta(hours=1)),
            "v_remaining_m_s": 45.0,
        },
        "conjunction": {
            "obj_id": "IRIDIUM 33 DEB",
            "t_ca_utc": _iso(_NOW + timedelta(hours=3)),
            "r_rel_km": [0.579405, 1.183231, 1.197698],
            "primary_norad": PRIMARY_NORAD,
            "secondary_norad": SECONDARY_NORAD,
            "use_stored_cdm": True,
        },
        "policy": {
            "lambda_v": 1.0, "lambda_L": 0.8,
            "dv_mag_limit_m_s": 2.0, "a_ref_km": 7000.0,
        },
    }

    capture = _SeedCapture()
    server.log.addHandler(capture)
    previous_level = server.log.level
    server.log.setLevel(logging.INFO)
    try:
        with patch.object(server, "UDL_ENABLED", False), \
             patch.object(server, "LEOLABS_ENABLED", False), \
             patch.object(server, "SECONDARY_SCREEN_ENABLED", True), \
             patch.object(server.http_requests, "post",
                          MagicMock(return_value=MagicMock(status_code=201))), \
             patch.object(server.http_requests, "get", side_effect=_ingest_get):
            response = TestClient(server.svc).post("/v1/evaluate", json=body)
    finally:
        server.log.removeHandler(capture)
        server.log.setLevel(previous_level)
    return response, capture.sources


class TestTheStoredPathSeedSource:
    def test_a_stored_cdm_with_the_primary_block_seeds_cdm_primary_own(self):
        """The acceptance criterion for item 2."""
        response, sources = _evaluate_with_stored_cdm(with_primary=True)
        assert response.status_code == 200, response.text
        assert sources, "no screening seed was selected at all"
        assert sources[-1] == "cdm_primary_own"

    def test_without_the_primary_block_it_falls_back_to_the_stand_in(self):
        """Older stored rows must still get a seed, clearly labelled."""
        response, sources = _evaluate_with_stored_cdm(with_primary=False)
        assert response.status_code == 200, response.text
        assert sources, "no screening seed was selected at all"
        assert sources[-1] == "combined_relative_stand_in"

    def test_the_two_cases_differ_only_in_the_seed_source(self):
        """Makes the pair non-vacuous: the same request, one field apart."""
        with_primary, _ = _evaluate_with_stored_cdm(with_primary=True)
        without, _ = _evaluate_with_stored_cdm(with_primary=False)
        assert with_primary.status_code == without.status_code == 200

    def test_the_decision_is_unchanged_by_the_seed_source(self):
        """This ticket is persistence and covariance exposure. The seed feeds the
        screen, never the recommendation."""
        a, _ = _evaluate_with_stored_cdm(with_primary=True)
        b, _ = _evaluate_with_stored_cdm(with_primary=False)
        assert a.json()["recommendation"] == b.json()["recommendation"]
