"""
services/planner/tests/test_covariance_adapter.py

SCRUM-369 item 5: tests for the covariance adapter fix.

Covers:
  - A synthetic event (no NORAD IDs) uses the documented elliptical
    surrogate, not the old flat 100 m identity value.
  - The source-priority order holds: real CDM covariance when NORADs
    are present and the fetch succeeds, else the documented surrogate
    on any failure (missing NORADs, network error, 404, 503, malformed
    response).
  - A real kilometer-scale miss now passes the Mahalanobis feasibility
    screen instead of being wrongly discarded as trivial -- demonstrated
    as a direct before/after comparison against the old 100 m value, so
    the fix itself is what's under test, not just the new code in
    isolation.

Run with (from repo root -- conftest.py handles PYTHONPATH):
    python -m pytest services/planner/tests/test_covariance_adapter.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from common.operator_policy import CovarianceSurrogate, OperatorPolicy
from avoid.decision_model import mahalanobis_sq

# ---------------------------------------------------------------------------
# Shared test orbit: real circular LEO, 550 km altitude.
# Same geometry used in the SCRUM-369 item 6 delta_C sanity check, so
# results here are directly comparable to that verification.
# ---------------------------------------------------------------------------
MU_EARTH = 398600.4418
_R_EARTH = 6378.137
_ALT_KM = 550.0
_A_KM = _R_EARTH + _ALT_KM
_V_CIRC = math.sqrt(MU_EARTH / _A_KM)

R_SAT_KM = [_A_KM, 0.0, 0.0]
V_SAT_KM_S = [0.0, _V_CIRC, 0.0]

# The old, buggy hardcoded value this fix replaces. Kept here only as a
# regression fixture -- to prove the new code no longer produces this.
OLD_HARDCODED_COV = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]


def _make_response(status_code, json_data=None, ok=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = ok if ok is not None else (200 <= status_code < 300)
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


# ---------------------------------------------------------------------------
# Surrogate covariance itself
# ---------------------------------------------------------------------------

class TestSurrogateCovariance:
    """The documented elliptical surrogate, not the old flat 100 m identity."""

    def test_not_the_old_100m_identity(self):
        """Regression guard: the new surrogate must not match the old value."""
        cov, source, cdm_id = server._surrogate_covariance(R_SAT_KM, V_SAT_KM_S)
        assert cov != OLD_HARDCODED_COV

    def test_source_label_is_surrogate_elliptical(self):
        _, source, _ = server._surrogate_covariance(R_SAT_KM, V_SAT_KM_S)
        assert source == "surrogate_elliptical"

    def test_cdm_record_id_is_none(self):
        _, _, cdm_id = server._surrogate_covariance(R_SAT_KM, V_SAT_KM_S)
        assert cdm_id is None

    def test_uses_operator_policy_values_when_loaded(self):
        """If a custom policy is loaded, its sigma values must be used,
        not the CovarianceSurrogate defaults."""
        custom_policy = OperatorPolicy(
            operator_id="TEST", policy_version="2.5.0",
            covariance_surrogate=CovarianceSurrogate(
                radial_sigma_km=1.0, along_track_sigma_km=1.0, cross_track_sigma_km=1.0,
            ),
        )
        with patch.object(server, "_operator_policy", custom_policy):
            cov, _, _ = server._surrogate_covariance(R_SAT_KM, V_SAT_KM_S)
        cov_matrix = np.array(cov).reshape(3, 3)
        # A 1.0/1.0/1.0 km sigma surrogate is spherical -- trace should be 3.0,
        # not the default surrogate's 6.59.
        assert math.isclose(np.trace(cov_matrix), 3.0, rel_tol=1e-6)

    def test_falls_back_to_defaults_when_policy_not_loaded(self):
        """If _operator_policy failed to load at startup (None), fall back
        to CovarianceSurrogate() defaults rather than crashing."""
        with patch.object(server, "_operator_policy", None):
            cov, source, _ = server._surrogate_covariance(R_SAT_KM, V_SAT_KM_S)
        cov_matrix = np.array(cov).reshape(3, 3)
        defaults = CovarianceSurrogate()
        expected_trace = (
            defaults.radial_sigma_km ** 2
            + defaults.along_track_sigma_km ** 2
            + defaults.cross_track_sigma_km ** 2
        )
        assert math.isclose(np.trace(cov_matrix), expected_trace, rel_tol=1e-6)
        assert source == "surrogate_elliptical"

    def test_matrix_is_symmetric(self):
        cov, _, _ = server._surrogate_covariance(R_SAT_KM, V_SAT_KM_S)
        m = np.array(cov).reshape(3, 3)
        assert np.allclose(m, m.T)

    def test_eigenvalues_match_configured_sigmas(self):
        """Trace/eigenvalues are rotation-invariant -- confirms the ellipse
        shape survives the RTN->ECI rotation intact, for any orbit."""
        cov, _, _ = server._surrogate_covariance(R_SAT_KM, V_SAT_KM_S)
        m = np.array(cov).reshape(3, 3)
        eigvals = sorted(np.linalg.eigvalsh(m))
        defaults = CovarianceSurrogate()
        expected = sorted([
            defaults.radial_sigma_km ** 2,
            defaults.along_track_sigma_km ** 2,
            defaults.cross_track_sigma_km ** 2,
        ])
        for got, want in zip(eigvals, expected):
            assert math.isclose(got, want, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Source-priority order: real CDM > surrogate, on every failure path
# ---------------------------------------------------------------------------

class TestFetchCdmCovariancePriority:
    """_fetch_cdm_covariance must always return a usable covariance, and
    must prefer the real CDM store whenever it succeeds."""

    def test_missing_both_norads_uses_surrogate_without_network_call(self):
        with patch.object(server.http_requests, "get") as mock_get:
            p_rel_km2, source, cdm_id = server._fetch_cdm_covariance(
                "", "", R_SAT_KM, V_SAT_KM_S
            )
        mock_get.assert_not_called()
        assert source == "surrogate_elliptical"
        assert p_rel_km2 != OLD_HARDCODED_COV

    def test_missing_primary_norad_uses_surrogate(self):
        with patch.object(server.http_requests, "get") as mock_get:
            _, source, _ = server._fetch_cdm_covariance(
                "", "35929", R_SAT_KM, V_SAT_KM_S
            )
        mock_get.assert_not_called()
        assert source == "surrogate_elliptical"

    def test_real_cdm_success_returns_real_source(self):
        fake_cov_rtn = [[0.05, 0, 0], [0, 0.3, 0], [0, 0, 0.02]]
        resp = _make_response(200, json_data={
            "covariance_combined_rtn": fake_cov_rtn,
            "covariance_source": "cdm_store",
            "id": 42,
        })
        with patch.object(server.http_requests, "get", return_value=resp):
            p_rel_km2, source, cdm_id = server._fetch_cdm_covariance(
                "226", "35929", R_SAT_KM, V_SAT_KM_S
            )
        assert source == "cdm_store"
        assert cdm_id == 42
        assert p_rel_km2 != OLD_HARDCODED_COV

    def test_network_failure_falls_back_to_surrogate(self):
        with patch.object(server.http_requests, "get", side_effect=ConnectionError("down")):
            _, source, _ = server._fetch_cdm_covariance(
                "226", "35929", R_SAT_KM, V_SAT_KM_S
            )
        assert source == "surrogate_elliptical"

    def test_404_falls_back_to_surrogate(self):
        resp = _make_response(404)
        with patch.object(server.http_requests, "get", return_value=resp):
            _, source, _ = server._fetch_cdm_covariance(
                "226", "35929", R_SAT_KM, V_SAT_KM_S
            )
        assert source == "surrogate_elliptical"

    def test_503_falls_back_to_surrogate(self):
        resp = _make_response(503)
        with patch.object(server.http_requests, "get", return_value=resp):
            _, source, _ = server._fetch_cdm_covariance(
                "226", "35929", R_SAT_KM, V_SAT_KM_S
            )
        assert source == "surrogate_elliptical"

    def test_other_bad_status_falls_back_to_surrogate(self):
        resp = _make_response(500, ok=False)
        with patch.object(server.http_requests, "get", return_value=resp):
            _, source, _ = server._fetch_cdm_covariance(
                "226", "35929", R_SAT_KM, V_SAT_KM_S
            )
        assert source == "surrogate_elliptical"

    def test_malformed_response_falls_back_to_surrogate(self):
        """Missing covariance_combined_rtn key must not raise -- falls
        back to the surrogate instead."""
        resp = _make_response(200, json_data={"unexpected": "shape"})
        with patch.object(server.http_requests, "get", return_value=resp):
            _, source, _ = server._fetch_cdm_covariance(
                "226", "35929", R_SAT_KM, V_SAT_KM_S
            )
        assert source == "surrogate_elliptical"


# ---------------------------------------------------------------------------
# The actual bug: a real km-scale miss now passes the screen
# ---------------------------------------------------------------------------

class TestKilometerMissScreening:
    """SCRUM-369 root cause: at the old 100 m covariance, a real km-scale
    miss was wrongly screened out as trivial. At the new surrogate, the
    same miss correctly passes. Tested as a direct before/after
    comparison so the fix itself is what's verified, not just the new
    code's behavior in isolation.
    """

    # A realistic along-track-dominant miss, consistent with real orbit
    # determination uncertainty shape (matches the geometry used in the
    # SCRUM-369 item 6 delta_C sanity check).
    MISS_VECTOR_RTN_KM = np.array([0.10, 1.20, 0.05])

    def _mahalanobis_distance(self, cov_flat_eci, r_rel_rtn):
        """Rotate the RTN miss vector into ECI (matching how the real
        system builds r_rel_km) and compute the Mahalanobis distance
        against the given ECI covariance."""
        r = np.array(R_SAT_KM, dtype=float)
        v = np.array(V_SAT_KM_S, dtype=float)
        rot = server._rtn_to_eci_rotation(r, v)
        r_rel_eci = rot @ r_rel_rtn
        cov = np.array(cov_flat_eci).reshape(3, 3)
        return math.sqrt(mahalanobis_sq(r_rel_eci, cov))

    def test_old_100m_covariance_wrongly_screens_out_real_miss(self):
        """Documents the bug: at the old value, this real event was
        wrongly treated as trivial."""
        md = self._mahalanobis_distance(OLD_HARDCODED_COV, self.MISS_VECTOR_RTN_KM)
        policy = OperatorPolicy(operator_id="TEST", policy_version="2.5.0")
        assert policy.passes_pre_screen(mahalanobis_distance=md) is False

    def test_new_surrogate_correctly_passes_the_same_miss(self):
        """The fix: the identical miss vector now correctly passes the
        screen, using the documented elliptical surrogate."""
        cov, _, _ = server._surrogate_covariance(R_SAT_KM, V_SAT_KM_S)
        md = self._mahalanobis_distance(cov, self.MISS_VECTOR_RTN_KM)
        policy = OperatorPolicy(operator_id="TEST", policy_version="2.5.0")
        assert policy.passes_pre_screen(mahalanobis_distance=md) is True
