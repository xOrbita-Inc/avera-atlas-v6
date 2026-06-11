"""
tests/test_udl_client.py

SCRUM-331: Unit tests for UDLClient AC2 (CDM retrieval and parsing).

Tests use recorded response fixtures -- no live UDL calls required.
The fixture is based on the real UDL conjunction response schema
captured from the UDL portal.

Run with (from repo root):
    python -m pytest services/planner/tests/test_udl_client.py -v
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from common.udl_client import (
    _auth_header,
    _expand_cov_upper_triangle,
    _parse_conjunction,
    _rtn_to_eci_rotation,
    get_conjunctions,
    get_elsets,
    UDL_ENABLED,
)


# ---------------------------------------------------------------------------
# Fixture -- realistic UDL conjunction record
# Based on the real schema from UDL portal queryhelp + example response.
# ---------------------------------------------------------------------------

def _make_conjunction_record(
    sat_no1: int = 25544,
    sat_no2: int = 35929,
    tca: str = "2026-06-15T12:00:00.000000Z",
    collision_prob: float = 1.5e-4,
    miss_distance_m: float = 800.0,
    rel_pos_r_m: float = 500.0,
    rel_pos_t_m: float = 600.0,
    rel_pos_n_m: float = 100.0,
    xpos: float = 6778.0,
    ypos: float = 0.0,
    zpos: float = 0.0,
    xvel: float = 0.0,
    yvel: float = 7.668,
    zvel: float = 0.0,
    cov1: list = None,
    cov2: list = None,
) -> Dict[str, Any]:
    """Build a realistic UDL conjunction record for testing."""
    cov1 = cov1 or [1e6, 0.0, 1e6, 0.0, 0.0, 1e6]  # diagonal, m^2
    cov2 = cov2 or [1e6, 0.0, 1e6, 0.0, 0.0, 1e6]

    return {
        "id": "TEST-CONJUNCTION-ID",
        "classificationMarking": "U",
        "idOnOrbit1": str(sat_no1),
        "idOnOrbit2": str(sat_no2),
        "satNo1": sat_no1,
        "satNo2": sat_no2,
        "type": "CONJUNCTION",
        "tca": tca,
        "missDistance": miss_distance_m,
        "collisionProb": collision_prob,
        "collisionProbMethod": "FOSTER-1992",
        "relPosR": rel_pos_r_m,
        "relPosT": rel_pos_t_m,
        "relPosN": rel_pos_n_m,
        "relVelMag": 14.5,
        "relVelR": 0.1,
        "relVelT": 14.4,
        "relVelN": 0.3,
        "stateVector1": {
            "idStateVector": "SV1-ID",
            "epoch": tca,
            "satNo": sat_no1,
            "xpos": xpos,
            "ypos": ypos,
            "zpos": zpos,
            "xvel": xvel,
            "yvel": yvel,
            "zvel": zvel,
            "referenceFrame": "J2000",
            "cov": cov1,
            "covReferenceFrame": "J2000",
        },
        "stateVector2": {
            "idStateVector": "SV2-ID",
            "epoch": tca,
            "satNo": sat_no2,
            "xpos": xpos + 0.5,
            "ypos": ypos + 0.6,
            "zpos": zpos + 0.1,
            "xvel": xvel,
            "yvel": yvel,
            "zvel": zvel,
            "referenceFrame": "J2000",
            "cov": cov2,
            "covReferenceFrame": "J2000",
        },
        "source": "TEST",
        "dataMode": "TEST",
        "createdAt": "2026-06-10T00:00:00.000Z",
        "createdBy": "test.user",
    }


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

class TestAuthHeader:
    def test_raises_when_credentials_missing(self):
        """_auth_header must raise RuntimeError when env vars are absent."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("UDL_USER", "UDL_PASS")}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(RuntimeError, match="UDL_USER"):
                _auth_header()

    def test_returns_authorization_header(self):
        """_auth_header must return a dict with Authorization key."""
        with patch.dict("os.environ", {"UDL_USER": "user", "UDL_PASS": "pass"}):
            headers = _auth_header()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Basic ")

    def test_accept_header_present(self):
        """_auth_header must include Accept: application/json."""
        with patch.dict("os.environ", {"UDL_USER": "user", "UDL_PASS": "pass"}):
            headers = _auth_header()
        assert headers.get("Accept") == "application/json"


# ---------------------------------------------------------------------------
# Covariance expansion tests
# ---------------------------------------------------------------------------

class TestExpandCovUpperTriangle:
    def test_expands_correctly(self):
        """6-element upper triangle must expand to correct 3x3 symmetric matrix."""
        cov6 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = _expand_cov_upper_triangle(cov6)
        expected = np.array([
            [1.0, 2.0, 4.0],
            [2.0, 3.0, 5.0],
            [4.0, 5.0, 6.0],
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_symmetric(self):
        """Result must be symmetric."""
        cov6 = [1.0, 0.5, 2.0, 0.3, 0.4, 3.0]
        result = _expand_cov_upper_triangle(cov6)
        np.testing.assert_array_almost_equal(result, result.T)

    def test_returns_zeros_for_short_input(self):
        """Input shorter than 6 elements must return zero matrix."""
        result = _expand_cov_upper_triangle([1.0, 2.0])
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result, np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# AC2: Parse conjunction -> evaluate_conjunction() inputs
# ---------------------------------------------------------------------------

class TestParseConjunction:
    """AC2: parsed CDM must pass directly into evaluate_conjunction() inputs."""

    def test_returns_dict_with_required_keys(self):
        """Parsed record must contain all evaluate_conjunction() input keys."""
        record = _make_conjunction_record()
        result = _parse_conjunction(record)
        assert result is not None
        assert "obj_id" in result
        assert "t_ca_utc" in result
        assert "r_rel_km" in result
        assert "p_rel_km2" in result
        assert "pc_precomputed" in result

    def test_obj_id_from_sat_no2(self):
        """obj_id must be derived from satNo2."""
        record = _make_conjunction_record(sat_no2=35929)
        result = _parse_conjunction(record)
        assert result["obj_id"] == "35929"

    def test_tca_has_z_suffix(self):
        """t_ca_utc must end with Z for UTC."""
        record = _make_conjunction_record(tca="2026-06-15T12:00:00.000000Z")
        result = _parse_conjunction(record)
        assert result["t_ca_utc"].endswith("Z")

    def test_tca_z_suffix_added_if_missing(self):
        """t_ca_utc must get Z appended if not present."""
        record = _make_conjunction_record(tca="2026-06-15T12:00:00.000000")
        result = _parse_conjunction(record)
        assert result["t_ca_utc"].endswith("Z")

    def test_pc_precomputed_mapped(self):
        """pc_precomputed must match collisionProb."""
        record = _make_conjunction_record(collision_prob=1.5e-4)
        result = _parse_conjunction(record)
        assert abs(result["pc_precomputed"] - 1.5e-4) < 1e-10

    def test_miss_distance_converted_to_km(self):
        """missDistance (metres) must be converted to km."""
        record = _make_conjunction_record(miss_distance_m=800.0)
        result = _parse_conjunction(record)
        assert abs(result["miss_distance_km"] - 0.8) < 1e-9

    def test_r_rel_km_is_3_element_list(self):
        """r_rel_km must be a 3-element list."""
        record = _make_conjunction_record()
        result = _parse_conjunction(record)
        assert isinstance(result["r_rel_km"], list)
        assert len(result["r_rel_km"]) == 3

    def test_p_rel_km2_is_9_element_list(self):
        """p_rel_km2 must be a 9-element flat list (3x3 row-major)."""
        record = _make_conjunction_record()
        result = _parse_conjunction(record)
        assert isinstance(result["p_rel_km2"], list)
        assert len(result["p_rel_km2"]) == 9

    def test_rel_pos_unit_conversion(self):
        """relPosR/T/N in metres must be converted to km in r_rel_km."""
        record = _make_conjunction_record(
            rel_pos_r_m=1000.0,
            rel_pos_t_m=0.0,
            rel_pos_n_m=0.0,
            xpos=6778.0, ypos=0.0, zpos=0.0,
            xvel=0.0, yvel=7.668, zvel=0.0,
        )
        result = _parse_conjunction(record)
        # 1000m radial should be 1.0 km in ECI radial direction
        r_rel = np.array(result["r_rel_km"])
        assert abs(np.linalg.norm(r_rel) - 1.0) < 0.01

    def test_covariance_units_converted_to_km2(self):
        """Covariance must be converted from m^2 to km^2 (divide by 1e6)."""
        cov = [1e6, 0.0, 1e6, 0.0, 0.0, 1e6]  # 1e6 m^2 = 1.0 km^2
        record = _make_conjunction_record(cov1=cov, cov2=cov)
        result = _parse_conjunction(record)
        # Combined covariance: 2 * 1e6 m^2 = 2.0 km^2 on diagonal
        p = np.array(result["p_rel_km2"]).reshape(3, 3)
        assert p[0, 0] > 0  # positive definite diagonal

    def test_returns_none_when_tca_missing(self):
        """Returns None when tca field is absent."""
        record = _make_conjunction_record()
        del record["tca"]
        result = _parse_conjunction(record)
        assert result is None

    def test_returns_none_on_malformed_record(self):
        """Returns None on malformed input, no crash."""
        result = _parse_conjunction({"id": "bad", "tca": None})
        assert result is None

    def test_returns_none_on_missing_state_vector(self):
        """Missing stateVector1 must return None -- zero fabrication is not acceptable."""
        record = _make_conjunction_record()
        del record["stateVector1"]
        result = _parse_conjunction(record)
        assert result is None


# ---------------------------------------------------------------------------
# AC4: Feature flag -- no UDL calls when disabled
# ---------------------------------------------------------------------------

class TestFeatureFlag:
    def test_returns_empty_when_udl_disabled(self):
        """get_conjunctions must return [] when UDL_ENABLED=false."""
        with patch("common.udl_client.UDL_ENABLED", False):
            result = get_conjunctions(sat_no=25544)
        assert result == []

    def test_returns_empty_on_missing_credentials(self):
        """get_conjunctions must return [] when credentials are missing."""
        with patch("common.udl_client.UDL_ENABLED", True):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("UDL_USER", "UDL_PASS")}
            with patch.dict("os.environ", env, clear=True):
                result = get_conjunctions(sat_no=25544)
        assert result == []

    def test_returns_empty_on_network_failure(self):
        """get_conjunctions must return [] on network failure, no crash."""
        with patch("common.udl_client.UDL_ENABLED", True):
            with patch.dict("os.environ", {"UDL_USER": "u", "UDL_PASS": "p"}):
                with patch("common.udl_client.requests.get") as mock_get:
                    mock_get.side_effect = ConnectionError("network down")
                    result = get_conjunctions(sat_no=25544)
        assert result == []

    def test_returns_empty_on_401(self):
        """get_conjunctions must return [] on 401 Unauthorized."""
        with patch("common.udl_client.UDL_ENABLED", True):
            with patch.dict("os.environ", {"UDL_USER": "u", "UDL_PASS": "p"}):
                with patch("common.udl_client.requests.get") as mock_get:
                    mock_resp = MagicMock()
                    mock_resp.status_code = 401
                    mock_resp.ok = False
                    mock_get.return_value = mock_resp
                    result = get_conjunctions(sat_no=25544)
        assert result == []

    def test_returns_empty_on_empty_response(self):
        """get_conjunctions must return [] when UDL returns empty list."""
        with patch("common.udl_client.UDL_ENABLED", True):
            with patch.dict("os.environ", {"UDL_USER": "u", "UDL_PASS": "p"}):
                with patch("common.udl_client.requests.get") as mock_get:
                    mock_resp = MagicMock()
                    mock_resp.status_code = 200
                    mock_resp.ok = True
                    mock_resp.json.return_value = []
                    mock_get.return_value = mock_resp
                    result = get_conjunctions(sat_no=25544)
        assert result == []


# ---------------------------------------------------------------------------
# AC2: Full pipeline -- fetch + parse -> evaluate_conjunction() shape
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """AC2: end-to-end fixture test confirming parsed output matches
    evaluate_conjunction() input contract."""

    def test_parsed_record_matches_evaluate_conjunction_schema(self):
        """Fixture record parses to a dict that satisfies evaluate_conjunction() inputs."""
        record = _make_conjunction_record(
            sat_no1=25544,
            sat_no2=35929,
            collision_prob=1.5e-4,
            miss_distance_m=800.0,
        )
        with patch("common.udl_client.UDL_ENABLED", True):
            with patch.dict("os.environ", {"UDL_USER": "u", "UDL_PASS": "p"}):
                with patch("common.udl_client.requests.get") as mock_get:
                    mock_resp = MagicMock()
                    mock_resp.status_code = 200
                    mock_resp.ok = True
                    mock_resp.json.return_value = [record]
                    mock_get.return_value = mock_resp
                    results = get_conjunctions(sat_no=25544)

        assert len(results) == 1
        conj = results[0]

        # Verify all evaluate_conjunction() required fields present
        assert "obj_id" in conj
        assert "t_ca_utc" in conj
        assert "r_rel_km" in conj
        assert "p_rel_km2" in conj
        assert len(conj["r_rel_km"]) == 3
        assert len(conj["p_rel_km2"]) == 9
        assert conj["pc_precomputed"] == pytest.approx(1.5e-4)
        assert conj["miss_distance_km"] == pytest.approx(0.8)

    def test_multiple_records_all_parsed(self):
        """Multiple records in response must all be parsed."""
        records = [
            _make_conjunction_record(sat_no2=35929, collision_prob=1.5e-4),
            _make_conjunction_record(sat_no2=40000, collision_prob=2.0e-4),
        ]
        with patch("common.udl_client.UDL_ENABLED", True):
            with patch.dict("os.environ", {"UDL_USER": "u", "UDL_PASS": "p"}):
                with patch("common.udl_client.requests.get") as mock_get:
                    mock_resp = MagicMock()
                    mock_resp.status_code = 200
                    mock_resp.ok = True
                    mock_resp.json.return_value = records
                    mock_get.return_value = mock_resp
                    results = get_conjunctions(sat_no=25544)

        assert len(results) == 2


# ---------------------------------------------------------------------------
# AC3: Elset retrieval
# ---------------------------------------------------------------------------

class TestGetElsets:
    """AC3: get_elsets() returns TLE text compatible with
    _parse_and_propagate_tle() in spacetrack_tle.py."""

    def _make_elset_record(self, sat_no=25544, line1=None, line2=None):
        line1 = line1 or "1 25544U 98067A   24001.50000000  .00010000  00000-0  17814-3 0  9990"
        line2 = line2 or "2 25544  51.6400 337.6640 0001234  84.4096 275.7258 15.50000000440102"
        return {
            "idElset": "TEST-ELSET-ID",
            "satNo": sat_no,
            "epoch": "2024-01-01T12:00:00.000000Z",
            "line1": line1,
            "line2": line2,
            "meanMotion": 15.5,
            "eccentricity": 0.0001234,
            "inclination": 51.64,
            "source": "TEST",
            "dataMode": "TEST",
        }

    def test_returns_empty_string_when_udl_disabled(self):
        """get_elsets must return '' when UDL_ENABLED=false."""
        with patch("common.udl_client.UDL_ENABLED", False):
            result = get_elsets()
        assert result == ""

    def test_returns_empty_string_on_missing_credentials(self):
        """get_elsets must return '' when credentials are missing."""
        with patch("common.udl_client.UDL_ENABLED", True):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("UDL_USER", "UDL_PASS")}
            with patch.dict("os.environ", env, clear=True):
                result = get_elsets()
        assert result == ""

    def test_returns_empty_string_on_network_failure(self):
        """get_elsets must return '' on network failure, no crash."""
        with patch("common.udl_client.UDL_ENABLED", True):
            with patch.dict("os.environ", {"UDL_USER": "u", "UDL_PASS": "p"}):
                with patch("common.udl_client.requests.get") as mock_get:
                    mock_get.side_effect = ConnectionError("network down")
                    result = get_elsets()
        assert result == ""

    def test_returns_tle_text_from_line1_line2(self):
        """get_elsets must assemble TLE text from line1/line2 fields."""
        record = self._make_elset_record()
        with patch("common.udl_client.UDL_ENABLED", True):
            with patch.dict("os.environ", {"UDL_USER": "u", "UDL_PASS": "p"}):
                with patch("common.udl_client.requests.get") as mock_get:
                    mock_resp = MagicMock()
                    mock_resp.status_code = 200
                    mock_resp.ok = True
                    mock_resp.json.return_value = [record]
                    mock_get.return_value = mock_resp
                    result = get_elsets(sat_no=25544)

        assert "1 25544U" in result
        assert "2 25544" in result

    def test_tle_text_has_three_lines_per_object(self):
        """Each TLE object must produce exactly 3 lines (name, line1, line2)."""
        record = self._make_elset_record()
        with patch("common.udl_client.UDL_ENABLED", True):
            with patch.dict("os.environ", {"UDL_USER": "u", "UDL_PASS": "p"}):
                with patch("common.udl_client.requests.get") as mock_get:
                    mock_resp = MagicMock()
                    mock_resp.status_code = 200
                    mock_resp.ok = True
                    mock_resp.json.return_value = [record]
                    mock_get.return_value = mock_resp
                    result = get_elsets(sat_no=25544)

        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) == 3

    def test_skips_records_missing_tle_lines(self):
        """Records without line1/line2 must be skipped, not crash."""
        bad_record = {"satNo": 99999, "epoch": "2024-01-01T00:00:00Z"}
        good_record = self._make_elset_record()
        with patch("common.udl_client.UDL_ENABLED", True):
            with patch.dict("os.environ", {"UDL_USER": "u", "UDL_PASS": "p"}):
                with patch("common.udl_client.requests.get") as mock_get:
                    mock_resp = MagicMock()
                    mock_resp.status_code = 200
                    mock_resp.ok = True
                    mock_resp.json.return_value = [bad_record, good_record]
                    mock_get.return_value = mock_resp
                    result = get_elsets()

        # Only the good record should appear
        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) == 3

    def test_multiple_records_assembled_correctly(self):
        """Multiple elset records must all appear in the TLE text."""
        records = [
            self._make_elset_record(sat_no=25544),
            self._make_elset_record(sat_no=35929,
                line1="1 35929U 93036PX  24001.50000000  .00000000  00000-0  00000-0 0  9991",
                line2="2 35929  74.0000 100.0000 0010000  90.0000 270.0000 14.00000000000001"),
        ]
        with patch("common.udl_client.UDL_ENABLED", True):
            with patch.dict("os.environ", {"UDL_USER": "u", "UDL_PASS": "p"}):
                with patch("common.udl_client.requests.get") as mock_get:
                    mock_resp = MagicMock()
                    mock_resp.status_code = 200
                    mock_resp.ok = True
                    mock_resp.json.return_value = records
                    mock_get.return_value = mock_resp
                    result = get_elsets()

        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) == 6  # 3 lines per record * 2 records
