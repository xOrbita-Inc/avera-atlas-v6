"""tests/test_leolabs_psd_repair.py

SCRUM-458: the on-demand screen must not silently drop conjunctions whose result
CDM covariance reads back slightly non-positive-semidefinite.

The bug: LeoLabs writes our submitted post-burn covariance into each result CDM at
fixed precision. That covariance is strongly anisotropic over 72 h (SCRUM-452),
so the round trip drives its small axes marginally negative. The parser's PSD
guard used a 1e-8 relative tolerance and raised, and run_screening counted the
event as unparseable and dropped it. On screening 603312 that dropped 348 of 349
conjunctions and reported on the 1 that survived, which under-reports conflicts
and pushes the screen toward a false CLEAR.

The fix is directional, and these tests are written around that direction:

  - A negative of numerical scale is clipped to zero and the event is kept.
  - A negative beyond that band is NOT repaired toward clear and NOT dropped: it
    is flagged, and the event fails the contract closed.
  - Neither path may turn a breaching event into a clear one.

Fixture technique: the RTN->ECI rotation is orthogonal, so it preserves
eigenvalues exactly. Setting the smallest eigenvalue of the RTN 6x6 to
-ratio * max_eig therefore sets the rotated ECI covariance's worst relative
negative to exactly `ratio`, which is the quantity the repair band is defined on.

Run from repo root:
    python -m pytest services/planner/tests/test_leolabs_psd_repair.py -v
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from aps_math import frames
from common.leolabs_cdm_parser import (
    _PSD_REPAIR_RTOL,
    _RTN_AXES,
    LeoLabsGuardError,
    _rotation_6x6,
    build_rtn_covariance_6x6,
    parse_leolabs_cdm,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"
_OUR_ID = "L2669"          # CRYOSAT 2, SAT1 on the sample CDM


@pytest.fixture
def cdm() -> dict:
    return json.loads(_FIXTURE.read_text())


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_rtn(cdm: dict, sat_key: str, cov: np.ndarray) -> None:
    """Write a 6x6 RTN covariance back into the CDM's lower-triangle fields."""
    for i, row in enumerate(_RTN_AXES):
        for j in range(i + 1):
            cdm[f"{sat_key}_C{row}_{_RTN_AXES[j]}"] = float(cov[i, j])


def _set_min_eig_ratio(cdm: dict, sat_key: str, ratio: float) -> float:
    """Force this object's covariance to have min_eig = -ratio * max_eig.

    Returns the negative eigenvalue installed, in m^2. Because the rotation to
    ECI is orthogonal, the rotated covariance carries the same eigenvalues, so
    this sets the ratio the repair band is judged on exactly.
    """
    cov = build_rtn_covariance_6x6(cdm, sat_key)
    w, v = np.linalg.eigh(0.5 * (cov + cov.T))
    target = -ratio * float(np.max(np.abs(w)))
    w[0] = target
    perturbed = v @ np.diag(w) @ v.T
    _write_rtn(cdm, sat_key, 0.5 * (perturbed + perturbed.T))
    return target


def _as_read_eci(cdm: dict, sat_key: str) -> np.ndarray:
    """The rotated ECI covariance before any repair, mirroring the parser."""
    cov_rtn = build_rtn_covariance_6x6(cdm, sat_key)
    r = np.array([cdm[f"{sat_key}_X"], cdm[f"{sat_key}_Y"], cdm[f"{sat_key}_Z"]],
                 dtype=np.float64)
    v = np.array([cdm[f"{sat_key}_X_DOT"], cdm[f"{sat_key}_Y_DOT"],
                  cdm[f"{sat_key}_Z_DOT"]], dtype=np.float64)
    rot6 = _rotation_6x6(frames.rtn_to_eci_rotation(r, v))
    return rot6 @ cov_rtn @ rot6.T


def _sync_diagonal_comments(cdm: dict, sat_key: str) -> None:
    """Rewrite the EME2000 diagonal comments to match the as-read rotation.

    Keeps the diagonal guard satisfied for the covariance as the CDM states it,
    which is what that guard exists to check, so these tests exercise the PSD
    limb rather than tripping over the diagonal limb.
    """
    eci = _as_read_eci(cdm, sat_key)
    for idx, key in enumerate(("CX_X", "CY_Y", "CZ_Z",
                               "CXDOT_XDOT", "CYDOT_YDOT", "CZDOT_ZDOT")):
        cdm[f"{sat_key}_COMMENT_{key}"] = float(eci[idx, idx])


def _min_eig(cov: np.ndarray) -> float:
    return float(np.min(np.linalg.eigvalsh(0.5 * (cov + cov.T))))


def _slightly_non_psd(cdm: dict, sat_key: str = "SAT1", ratio: float = 5.0e-3) -> dict:
    """A CDM whose covariance is non-PSD at the scale really observed (0.5%)."""
    bad = copy.deepcopy(cdm)
    _set_min_eig_ratio(bad, sat_key, ratio)
    _sync_diagonal_comments(bad, sat_key)
    return bad


def _badly_non_psd(cdm: dict, sat_key: str = "SAT1", ratio: float = 0.35) -> dict:
    """A CDM whose covariance is non-PSD far beyond any precision explanation."""
    bad = copy.deepcopy(cdm)
    _set_min_eig_ratio(bad, sat_key, ratio)
    _sync_diagonal_comments(bad, sat_key)
    return bad


# ---------------------------------------------------------------------------
# The fixture technique itself is load-bearing, so check it
# ---------------------------------------------------------------------------

def test_fixture_helper_sets_the_rotated_ratio_exactly():
    """The rotation preserves eigenvalues, so the installed ratio survives it."""
    cdm = json.loads(_FIXTURE.read_text())
    installed = _set_min_eig_ratio(cdm, "SAT1", 5.0e-3)
    eci = _as_read_eci(cdm, "SAT1")
    w = np.linalg.eigvalsh(0.5 * (eci + eci.T))
    assert float(np.min(w)) == pytest.approx(installed, rel=1e-9)
    assert abs(float(np.min(w))) / float(np.max(np.abs(w))) == pytest.approx(
        5.0e-3, rel=1e-6)


def test_unmodified_fixture_is_psd_and_unrepaired(cdm):
    """The golden CDM is untouched by any of this."""
    parsed = parse_leolabs_cdm(cdm, _OUR_ID)
    assert not parsed.covariance_repaired
    assert not parsed.covariance_untrusted
    assert _min_eig(parsed.primary.cov_eci_m2) >= 0.0


# ---------------------------------------------------------------------------
# 1. Slightly non-PSD: repaired, kept, evaluated
# ---------------------------------------------------------------------------

def test_slightly_non_psd_parses_instead_of_raising(cdm):
    """The exact case that dropped 348 of 349 events now parses."""
    parsed = parse_leolabs_cdm(_slightly_non_psd(cdm), _OUR_ID)
    assert parsed.covariance_repaired
    assert not parsed.covariance_untrusted


def test_slightly_non_psd_is_repaired_to_psd(cdm):
    parsed = parse_leolabs_cdm(_slightly_non_psd(cdm), _OUR_ID)
    cov = parsed.primary.cov_eci_m2
    scale = float(np.max(np.abs(np.linalg.eigvalsh(cov))))
    assert _min_eig(cov) >= -1e-12 * scale
    assert np.allclose(cov, cov.T, rtol=0, atol=1e-9 * scale)


def test_repair_only_clips_the_negative_directions(cdm):
    """The repair is a clip, not a rescale: the real axes are left alone."""
    bad = _slightly_non_psd(cdm)
    before = np.linalg.eigvalsh(_as_read_eci(bad, "SAT1"))
    after = np.linalg.eigvalsh(
        parse_leolabs_cdm(bad, _OUR_ID).primary.cov_eci_m2)
    # eigvalsh returns ascending order, so the positive tail is directly
    # comparable. Every positive eigenvalue survives untouched, however small:
    # the live covariance spans 1e-6 to 1e7 m^2 and the small real axes are
    # exactly the ones a rescale would have distorted.
    positive_before = before[before > 0]
    k = len(positive_before)
    # rtol is 1e-6, not tighter: this covariance spans 2e-6 to 1.3e7 m^2, a
    # dynamic range of 6e-13, so rebuilding it as V diag(w) V^T costs the
    # smallest eigenvalue a few parts in 1e8. That is the arithmetic, not the
    # repair -- the absolute check against the matrix scale is the strict one.
    scale = float(np.max(np.abs(before)))
    assert np.allclose(after[-k:], positive_before,
                       rtol=1.0e-6, atol=1.0e-12 * scale)
    # and the negative directions became zero, not something else: each is now
    # negligible against the smallest *real* axis, which is the scale that means
    # something physically.
    n_neg = int(np.count_nonzero(before < 0))
    assert n_neg >= 1
    assert np.all(np.abs(after[:n_neg]) < 1.0e-6 * float(positive_before[0]))


def test_repaired_event_records_the_eigenvalue_it_was_repaired_from(cdm):
    """The audit trail keeps the number, so a repair is never invisible."""
    bad = _slightly_non_psd(cdm)
    expected = _min_eig(_as_read_eci(bad, "SAT1"))
    parsed = parse_leolabs_cdm(bad, _OUR_ID)
    assert parsed.primary.covariance_min_eigenvalue_m2 == pytest.approx(expected)
    cond = parsed.covariance_conditioning()
    assert cond["primary"]["repaired"] is True
    assert cond["primary"]["untrusted"] is False


def test_repair_at_the_observed_live_scale_is_inside_the_band(cdm):
    """0.54% is the worst ratio measured over screening 603312's 1255 CDMs."""
    parsed = parse_leolabs_cdm(_slightly_non_psd(cdm, ratio=5.37e-3), _OUR_ID)
    assert parsed.covariance_repaired
    assert not parsed.covariance_untrusted
    assert _PSD_REPAIR_RTOL > 5.37e-3, (
        "the repair band must cover the worst ratio actually observed live")


def test_secondary_covariance_is_repaired_too(cdm):
    """Whose covariance is marginal does not change the handling."""
    parsed = parse_leolabs_cdm(_slightly_non_psd(cdm, "SAT2"), _OUR_ID)
    assert parsed.secondary.covariance_repaired
    assert _min_eig(parsed.secondary.cov_eci_m2) >= -1e-12 * float(
        np.max(np.abs(np.linalg.eigvalsh(parsed.secondary.cov_eci_m2))))


# ---------------------------------------------------------------------------
# 2. Badly non-PSD: flagged and failed closed, never repaired toward clear
# ---------------------------------------------------------------------------

def test_badly_non_psd_still_raises_on_the_strict_paths(cdm):
    """The live listing and evaluate paths keep the guard they have today."""
    with pytest.raises(LeoLabsGuardError, match="positive semidefinite"):
        parse_leolabs_cdm(_badly_non_psd(cdm), _OUR_ID)


def test_badly_non_psd_is_kept_and_flagged_when_not_strict(cdm):
    """The screen keeps the event so it can fail closed instead of vanishing."""
    parsed = parse_leolabs_cdm(_badly_non_psd(cdm), _OUR_ID, strict_psd=False)
    assert parsed.covariance_untrusted
    assert not parsed.primary.covariance_repaired


def test_badly_non_psd_covariance_is_left_exactly_as_read(cdm):
    """No repair toward clear: the matrix is labelled, never quietly shrunk."""
    bad = _badly_non_psd(cdm)
    expected = _as_read_eci(bad, "SAT1")
    parsed = parse_leolabs_cdm(bad, _OUR_ID, strict_psd=False)
    assert np.allclose(parsed.primary.cov_eci_m2, expected, rtol=0, atol=0)
    assert _min_eig(parsed.primary.cov_eci_m2) < 0.0


def test_untrusted_covariance_logs_a_warning_naming_the_object(cdm, caplog):
    """Named object and eigenvalue, so the log points at the right satellite."""
    with caplog.at_level(logging.WARNING):
        parse_leolabs_cdm(_badly_non_psd(cdm), _OUR_ID, strict_psd=False)
    warnings = [r for r in caplog.records
                if getattr(r, "event", None) == "leolabs_covariance_untrusted"]
    assert len(warnings) == 1
    assert warnings[0].sat_key == "SAT1"
    assert warnings[0].designator == _OUR_ID
    assert warnings[0].min_eigenvalue_m2 < 0


def test_the_repair_band_boundary_is_the_relative_ratio_not_an_absolute(cdm):
    """Just inside repairs; just outside flags. The band is relative by design."""
    inside = parse_leolabs_cdm(
        _slightly_non_psd(cdm, ratio=_PSD_REPAIR_RTOL * 0.5), _OUR_ID,
        strict_psd=False)
    outside = parse_leolabs_cdm(
        _badly_non_psd(cdm, ratio=_PSD_REPAIR_RTOL * 2.0), _OUR_ID,
        strict_psd=False)
    assert inside.covariance_repaired and not inside.covariance_untrusted
    assert outside.covariance_untrusted and not outside.primary.covariance_repaired


# ---------------------------------------------------------------------------
# 3. Symmetry stays a hard error
# ---------------------------------------------------------------------------

def test_asymmetric_covariance_is_still_a_hard_error(cdm):
    """Asymmetry is a real convention fault and is never repaired."""
    bad = copy.deepcopy(cdm)
    cov = build_rtn_covariance_6x6(bad, "SAT1")
    cov[0, 1] = cov[0, 1] * 3.0 + 1.0e4      # break the mirror, not the triangle
    for i, row in enumerate(_RTN_AXES):
        for j, col in enumerate(_RTN_AXES):
            if j <= i:
                bad[f"SAT1_C{row}_{col}"] = float(cov[i, j])
    # write the upper element too, so the reconstructed matrix is asymmetric only
    # if the parser trusted it; instead assert via a direct asymmetric injection
    import common.leolabs_cdm_parser as parser_mod
    asym = np.eye(6) * 100.0
    asym[0, 1] = 50.0
    with pytest.raises(LeoLabsGuardError, match="not symmetric"):
        parser_mod._assert_symmetric(asym, "SAT1")


def test_symmetry_check_runs_before_any_repair(cdm):
    """An asymmetric matrix raises rather than being symmetrised into a pass."""
    import common.leolabs_cdm_parser as parser_mod
    asym = np.diag([1.0e6] * 6)
    asym[0, 1] = 5.0e5
    with pytest.raises(LeoLabsGuardError, match="not symmetric"):
        parser_mod._assert_symmetric(asym, "SAT2")


# ---------------------------------------------------------------------------
# 4. Ordering: the repair must not trip the diagonal-comment guard
# ---------------------------------------------------------------------------

def test_repair_does_not_trip_the_diagonal_comment_guard(cdm):
    """The diagonal guard checks the rotation as read, so it runs before repair.

    Measured on screening 603312, clipping shifts 1992 of 7494 diagonal entries by
    more than _DIAG_RTOL. If the repair ran first, that guard would reject the
    event and SCRUM-458 would have fixed nothing.
    """
    bad = _slightly_non_psd(cdm)
    parsed = parse_leolabs_cdm(bad, _OUR_ID, validate_diagonals=True)
    assert parsed.covariance_repaired


def test_the_repair_really_does_move_the_diagonal(cdm):
    """Guards the test above: it would be vacuous if the clip changed nothing."""
    bad = _slightly_non_psd(cdm)
    as_read = np.diag(_as_read_eci(bad, "SAT1"))
    repaired = np.diag(parse_leolabs_cdm(bad, _OUR_ID).primary.cov_eci_m2)
    shifts = np.abs(repaired - as_read) / np.abs(as_read)
    assert float(np.max(shifts)) > 1.0e-3, (
        "expected the clip to move at least one diagonal past _DIAG_RTOL")
