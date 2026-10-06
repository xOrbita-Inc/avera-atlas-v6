"""tests/test_leolabs_cdm_parser.py

SCRUM-411: unit tests for the LeoLabs CCSDS CDM parser.

The two golden checks from the design (sections 6.2 and 6.5) are the heart of
this suite and both must pass:

  1. After rotating each object's RTN covariance to ECI, the position-covariance
     diagonal matches the CDM's own EME2000 diagonal comments
     (SATn_COMMENT_CX_X/CY_Y/CZ_Z) to better than 0.01 percent.
  2. Pc computed from aps_math.pc_utils on the parsed ECI states and rotated
     covariances matches the CDM's COLLISION_PROBABILITY within an Alfano-2005
     tolerance.

Everything else here guards the guards: object resolution, the CALCULATED and
EME2000 gates, the miss-distance sanity check, and the output contract.

Run from repo root:
    python -m pytest services/planner/tests/test_leolabs_cdm_parser.py -v
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from common.leolabs_cdm_parser import (
    LeoLabsGuardError,
    LeoLabsParseError,
    build_rtn_covariance_6x6,
    cross_check_pc,
    parse_leolabs_cdm,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"

# Our subscribed asset on the sample CDM is CRYOSAT 2 = L2669 (SAT1). We resolve
# by catalog id, never by assuming SAT1 is ours.
_OUR_ID = "L2669"


@pytest.fixture
def cdm() -> dict:
    return json.loads(_FIXTURE.read_text())


# ---------------------------------------------------------------------------
# Golden check 1: rotated ECI diagonal vs the CDM's EME2000 comments
# ---------------------------------------------------------------------------

def test_golden_rotated_diagonal_matches_eme2000_comments(cdm):
    """Design 6.2: diag(C_eci_pos) matches SATn_COMMENT_CX_X/CY_Y/CZ_Z."""
    parsed = parse_leolabs_cdm(cdm, _OUR_ID)

    for obj in (parsed.primary, parsed.secondary):
        eci = obj.cov_eci_m2
        expected = np.array([
            cdm[f"{obj.sat_key}_COMMENT_CX_X"],
            cdm[f"{obj.sat_key}_COMMENT_CY_Y"],
            cdm[f"{obj.sat_key}_COMMENT_CZ_Z"],
        ], dtype=float)
        got = np.array([eci[0, 0], eci[1, 1], eci[2, 2]], dtype=float)
        rel_err = np.abs(got - expected) / np.abs(expected)
        # Design says "better than 0.01 percent" (1e-4). Observed ~1e-6.
        assert np.max(rel_err) < 1.0e-4, (
            f"{obj.sat_key} position diagonal rel err {np.max(rel_err):.3e}"
        )


def test_rotated_velocity_diagonal_matches_comments(cdm):
    """The velocity diagonals rotate correctly too, a stronger regression net."""
    parsed = parse_leolabs_cdm(cdm, _OUR_ID)
    for obj in (parsed.primary, parsed.secondary):
        eci = obj.cov_eci_m2
        expected = np.array([
            cdm[f"{obj.sat_key}_COMMENT_CXDOT_XDOT"],
            cdm[f"{obj.sat_key}_COMMENT_CYDOT_YDOT"],
            cdm[f"{obj.sat_key}_COMMENT_CZDOT_ZDOT"],
        ], dtype=float)
        got = np.array([eci[3, 3], eci[4, 4], eci[5, 5]], dtype=float)
        rel_err = np.abs(got - expected) / np.abs(expected)
        assert np.max(rel_err) < 1.0e-4


# ---------------------------------------------------------------------------
# Golden check 2: our Pc vs the CDM's COLLISION_PROBABILITY
# ---------------------------------------------------------------------------

def test_golden_pc_cross_check(cdm):
    """Design 6.5: aps_math Pc matches CDM COLLISION_PROBABILITY (ALFANO-2005)."""
    parsed = parse_leolabs_cdm(cdm, _OUR_ID)
    ours = cross_check_pc(parsed)
    theirs = parsed.cdm_collision_probability

    assert cdm["COLLISION_PROBABILITY_METHOD"] == "ALFANO-2005"
    assert theirs is not None and theirs > 0
    rel_err = abs(ours - theirs) / theirs
    # Observed ~1.6e-4; 5e-3 leaves ample margin while still meaningful.
    assert rel_err < 5.0e-3, (
        f"our Pc {ours:.6e} vs CDM {theirs:.6e}, rel err {rel_err:.3e}"
    )


# ---------------------------------------------------------------------------
# Object resolution (design 6.3): do not assume SAT1 is our asset
# ---------------------------------------------------------------------------

def test_primary_resolves_by_catalog_id_sat1(cdm):
    parsed = parse_leolabs_cdm(cdm, "L2669")
    assert parsed.primary.designator == "L2669"
    assert parsed.primary.sat_key == "SAT1"
    assert parsed.secondary.designator == "L143957"


def test_primary_resolves_by_catalog_id_sat2(cdm):
    """If our sat is SAT2, roles swap; SAT1 must not be assumed primary."""
    parsed = parse_leolabs_cdm(cdm, "L143957")
    assert parsed.primary.designator == "L143957"
    assert parsed.primary.sat_key == "SAT2"
    assert parsed.secondary.designator == "L2669"


def test_role_swap_flips_relative_sign(cdm):
    """r_rel is secondary-minus-primary, so swapping roles negates it."""
    a = parse_leolabs_cdm(cdm, "L2669")
    b = parse_leolabs_cdm(cdm, "L143957")
    np.testing.assert_allclose(a.r_rel_km(), -np.array(b.r_rel_km()), rtol=1e-12)


def test_unknown_catalog_id_raises(cdm):
    with pytest.raises(LeoLabsParseError):
        parse_leolabs_cdm(cdm, "L999999")


# ---------------------------------------------------------------------------
# Guards (design 6.4)
# ---------------------------------------------------------------------------

def test_guard_rejects_non_calculated_covariance(cdm):
    bad = copy.deepcopy(cdm)
    bad["SAT2_COVARIANCE_METHOD"] = "DEFAULT"
    with pytest.raises(LeoLabsGuardError, match="COVARIANCE_METHOD"):
        parse_leolabs_cdm(bad, _OUR_ID)


def test_guard_rejects_non_eme2000_frame(cdm):
    bad = copy.deepcopy(cdm)
    bad["SAT1_REF_FRAME"] = "ITRF"
    with pytest.raises(LeoLabsGuardError, match="REF_FRAME"):
        parse_leolabs_cdm(bad, _OUR_ID)


def test_guard_diagonal_mismatch_trips(cdm):
    """Corrupting an RTN covariance term breaks the ECI-diagonal agreement."""
    bad = copy.deepcopy(cdm)
    bad["SAT1_CR_R"] = float(bad["SAT1_CR_R"]) * 10.0
    with pytest.raises(LeoLabsGuardError, match="diagonal"):
        parse_leolabs_cdm(bad, _OUR_ID)


def test_guard_miss_distance_mismatch_trips(cdm):
    """A perturbed secondary position breaks the miss-distance sanity check.

    Diagonal validation is disabled here so the miss-distance guard is exercised
    in isolation (moving the state also changes that object's rotation).
    """
    bad = copy.deepcopy(cdm)
    bad["SAT2_X"] = float(bad["SAT2_X"]) + 5.0  # +5 km
    with pytest.raises(LeoLabsGuardError, match="miss distance"):
        parse_leolabs_cdm(bad, _OUR_ID, validate_diagonals=False)


def test_guard_missing_diagonal_element_raises(cdm):
    bad = copy.deepcopy(cdm)
    del bad["SAT1_CR_R"]
    with pytest.raises(LeoLabsParseError, match="CR_R"):
        parse_leolabs_cdm(bad, _OUR_ID)


# ---------------------------------------------------------------------------
# Covariance construction and properties
# ---------------------------------------------------------------------------

def test_rtn_covariance_is_symmetric_6x6(cdm):
    cov = build_rtn_covariance_6x6(cdm, "SAT1")
    assert cov.shape == (6, 6)
    np.testing.assert_allclose(cov, cov.T, rtol=0, atol=0)
    # Spot-check a known off-diagonal maps to the right cell (CT_R -> row T,col R).
    assert cov[1, 0] == pytest.approx(cdm["SAT1_CT_R"])
    assert cov[0, 0] == pytest.approx(cdm["SAT1_CR_R"])


def test_combined_covariance_symmetric_and_psd(cdm):
    parsed = parse_leolabs_cdm(cdm, _OUR_ID)
    p = np.array(parsed.p_rel_eci_km2())
    np.testing.assert_allclose(p, p.T, rtol=1e-12, atol=0)
    assert np.min(np.linalg.eigvalsh(p)) > 0


def test_combined_hard_body_radius_is_sum(cdm):
    parsed = parse_leolabs_cdm(cdm, _OUR_ID)
    assert parsed.combined_hbr_m == pytest.approx(3.1 + 0.5)


# ---------------------------------------------------------------------------
# Output contract (design 6.1): matches the existing ConjunctionState shape
# ---------------------------------------------------------------------------

def test_to_conjunction_state_contract(cdm):
    parsed = parse_leolabs_cdm(cdm, _OUR_ID)
    state = parsed.to_conjunction_state()

    assert set(state) == {
        "obj_id", "t_ca_utc", "r_rel_km", "v_rel_km_s",
        "v_rel_source", "p_rel_km2", "pc_precomputed",
        # SCRUM-489: the secondary's measured radius, so the scorer computes Pc
        # on the real combined hard-body radius instead of the screening floor.
        "secondary_radius_m",
    }
    assert state["obj_id"] == "L143957"
    assert state["t_ca_utc"] == "2026-08-30T10:24:48.130573Z"
    assert len(state["r_rel_km"]) == 3
    assert len(state["v_rel_km_s"]) == 3
    assert len(state["p_rel_km2"]) == 9
    assert state["v_rel_source"] == "state_vector_difference"
    # The CDM Pc is a cross-check, not our Pc: the planner computes it from
    # geometry. SCRUM-489 changed the hard-body radius that goes into that
    # computation, not where the Pc comes from.
    assert state["pc_precomputed"] is None
    assert state["secondary_radius_m"] == parsed.secondary.radius_m


def test_units_relative_position_km(cdm):
    """r_rel is in km: its magnitude equals MISS_DISTANCE (m) / 1000."""
    parsed = parse_leolabs_cdm(cdm, _OUR_ID)
    mag_km = float(np.linalg.norm(parsed.r_rel_km()))
    assert mag_km == pytest.approx(cdm["MISS_DISTANCE"] / 1000.0, rel=1e-3)


def test_provenance_records_source_and_ids(cdm):
    parsed = parse_leolabs_cdm(cdm, _OUR_ID)
    prov = parsed.provenance
    assert prov["cdm_source"] == "LEOLABS"
    assert prov["primary_designator"] == "L2669"
    assert prov["secondary_designator"] == "L143957"
    assert prov["primary_norad_id"] == 36508
    assert prov["ref_frame"] == "EME2000"
    assert prov["cdm_collision_probability"] == pytest.approx(cdm["COLLISION_PROBABILITY"])


# --- SCRUM-417: response conjunction block ---------------------------------

def test_to_response_conjunction_block(cdm):
    parsed = parse_leolabs_cdm(cdm, _OUR_ID)
    block = parsed.to_response_conjunction()
    assert block["primary"] == {
        "designator": "L2669", "norad_id": 36508, "object_name": "CRYOSAT 2"}
    assert block["secondary"] == {
        "designator": "L143957", "norad_id": 270302,
        "object_name": "TBA - TO BE ASSIGNED"}
    assert block["tca_utc"] == parsed.t_ca_utc
    assert block["miss_distance_km"] == pytest.approx(20.115063)
    assert block["relative_position_rtn_m"] == pytest.approx(
        [-307.847, 19903.728, 2891.813])
    assert block["relative_velocity_rtn_m_s"] == pytest.approx(
        [-29.797, -311.922, 2143.705])


def test_to_response_conjunction_rtn_none_when_missing(cdm):
    bad = copy.deepcopy(cdm)
    del bad["RELATIVE_POSITION_T"]
    parsed = parse_leolabs_cdm(bad, _OUR_ID)
    block = parsed.to_response_conjunction()
    assert block["relative_position_rtn_m"] is None
    # velocity still fully present, so it is returned
    assert block["relative_velocity_rtn_m_s"] == pytest.approx(
        [-29.797, -311.922, 2143.705])
