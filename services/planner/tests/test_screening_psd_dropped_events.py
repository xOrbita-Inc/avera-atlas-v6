"""tests/test_screening_psd_dropped_events.py

SCRUM-458, the screen-level half: a conjunction the screen could not fully assess
must never be silently absent from a result that then reads CLEAR.

Before this ticket, a result CDM whose covariance read back marginally non-PSD
raised in the parser, and run_screening counted it as unparseable and dropped it.
On screening 603312 that dropped 348 of 349 conjunctions. The screen then judged
the single survivor and could report CLEAR.

Three properties are tested here, in the order they matter:

  1. Those events come back. The skip count goes to zero for numerical non-PSD.
  2. An event that is genuinely unassessable is carried and breaches, rather than
     being omitted -- it fails the screen closed.
  3. Neither the repair nor the flag can turn a breaching event into a clear one.

The parser-level behaviour (the repair band, the clip, the ordering against the
diagonal guard) is in test_leolabs_psd_repair.py.

Run from repo root:
    python -m pytest services/planner/tests/test_screening_psd_dropped_events.py -v
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from common import atlas_artifact as aa
from common import leolabs_runtime, leolabs_screening
from common import leolabs_screening as ls
from common.leolabs_asset_map import AssetRegistry
from common.leolabs_cdm_parser import (
    _PSD_REPAIR_RTOL,
    _RTN_AXES,
    build_rtn_covariance_6x6,
    parse_leolabs_cdm,
)
from common.leolabs_screening import (
    ScreeningResult,
    ScreeningThresholds,
    evaluate_clear_contract,
    run_screening,
)
from common.operator_policy import OperatorPolicy

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"
_OBJECTS = [{"catalogNumber": "L2669", "noradCatalogNumber": 36508,
             "name": "CRYOSAT 2"}]
_THRESHOLDS = ScreeningThresholds(
    min_probability_of_collision=1.0e-4,
    max_miss_distance_km=1.0,
    max_mahalanobis=4.0,
    combined_hbr_m=15.0,
    screening_volume_km=50.0,
)
_EPHEMERIS = {
    "frame": "EME2000",
    "covarianceFrame": "EME2000",
    "states": [{
        "timestamp": "2026-09-24T12:00:00Z",
        "position": [6792000.0, 0.0, 0.0],
        "velocity": [0.0, 5000.0, 5800.0],
        "covariance": [[40000.0] + [0.0] * 5] + [[0.0] * 6] * 5,
    }],
}

_EPOCH = "2026-09-24T12:00:00Z"
_R_POST = [6838.0, 0.0, 0.0]
_V_POST = [0.0, 0.3419, 7.5754]
_P_POST = [[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.16]]


@pytest.fixture
def cdm() -> dict:
    return json.loads(_FIXTURE.read_text())


@pytest.fixture
def registry() -> AssetRegistry:
    return AssetRegistry.from_objects(_OBJECTS)


@pytest.fixture
def policy() -> OperatorPolicy:
    return OperatorPolicy(operator_id="test", policy_version="v1")


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(leolabs_screening, "LEOLABS_ENABLED", True)
    leolabs_runtime.reset_caches()
    yield
    leolabs_runtime.reset_caches()


# ---------------------------------------------------------------------------
# Fixture helpers: install an exact relative negative eigenvalue
# ---------------------------------------------------------------------------

def _with_min_eig_ratio(cdm: dict, ratio: float, sat_key: str = "SAT1") -> dict:
    """A copy of this CDM whose covariance has min_eig = -ratio * max_eig.

    The RTN->ECI rotation is orthogonal, so installing the eigenvalue in RTN sets
    the rotated covariance's relative negative to exactly `ratio`. The EME2000
    diagonal comments are re-synced so the diagonal guard still describes the
    covariance the CDM states, leaving the PSD limb as the thing under test.
    """
    out = copy.deepcopy(cdm)
    cov = build_rtn_covariance_6x6(out, sat_key)
    w, v = np.linalg.eigh(0.5 * (cov + cov.T))
    w[0] = -ratio * float(np.max(np.abs(w)))
    perturbed = v @ np.diag(w) @ v.T
    perturbed = 0.5 * (perturbed + perturbed.T)
    for i, row in enumerate(_RTN_AXES):
        for j in range(i + 1):
            out[f"{sat_key}_C{row}_{_RTN_AXES[j]}"] = float(perturbed[i, j])

    from aps_math import frames
    from common.leolabs_cdm_parser import _rotation_6x6
    r = np.array([out[f"{sat_key}_X"], out[f"{sat_key}_Y"], out[f"{sat_key}_Z"]])
    vv = np.array([out[f"{sat_key}_X_DOT"], out[f"{sat_key}_Y_DOT"],
                   out[f"{sat_key}_Z_DOT"]])
    rot6 = _rotation_6x6(frames.rtn_to_eci_rotation(r, vv))
    eci = rot6 @ build_rtn_covariance_6x6(out, sat_key) @ rot6.T
    for idx, key in enumerate(("CX_X", "CY_Y", "CZ_Z",
                               "CXDOT_XDOT", "CYDOT_YDOT", "CZDOT_ZDOT")):
        out[f"{sat_key}_COMMENT_{key}"] = float(eci[idx, idx])
    return out


def _event(cdm: dict, event_id: str, minute: int = 0, **overrides) -> dict:
    """A distinct conjunction event, in the shape a screening result really has.

    Screening result CDMs all carry COMMENT_EVENT_ID = "0" and COMMENT_ID = the
    screening id, so those fields are deliberately left constant here: what makes
    two screened conjunctions distinct is the secondary and the TCA (SCRUM-458).
    """
    out = copy.deepcopy(cdm)
    out["COMMENT_EVENT_ID"] = "0"
    out["COMMENT_ID"] = "603312"
    out["SAT2_OBJECT_DESIGNATOR"] = f"L{90000 + abs(hash(event_id)) % 9000}"
    out["TCA_ISO"] = f"2026-08-30T{10 + minute // 60:02d}:{minute % 60:02d}:48.130573Z"
    out.update(overrides)
    return out


def _client(cdms, *, created=None, status=None):
    client = MagicMock()
    client.create_screening.return_value = created or {"id": "scr-458"}
    client.wait_for_screening.return_value = status or {"status": "completed"}
    client.get_screening_cdms.return_value = cdms
    return client


def _run(client, registry, **kw):
    return run_screening(_EPHEMERIS, _THRESHOLDS, "L2669",
                         client=client, registry=registry, **kw)


def _parsed(cdm_dict, **overrides):
    """Parse for contract cases, with the miss-distance cross-check off.

    The overrides deliberately set a miss distance that does not match the
    fixture's state vectors; the contract evaluation is what is under test here,
    not the parser guards, which have their own suite.
    """
    out = copy.deepcopy(cdm_dict)
    out.update(overrides)
    return parse_leolabs_cdm(out, "L2669", validate_miss_distance=False,
                             strict_psd=False)


# ---------------------------------------------------------------------------
# 1. The dropped events come back
# ---------------------------------------------------------------------------

class TestEventsAreNoLongerDropped:
    def test_a_marginally_non_psd_result_cdm_is_no_longer_skipped(
        self, cdm, registry
    ):
        """The exact live failure: one event, previously skipped, now returned."""
        bad = _with_min_eig_ratio(cdm, 5.0e-3)
        result = _run(_client([bad]), registry)
        assert result.skipped == []
        assert len(result.conjunctions) == 1
        assert result.repaired_count == 1
        assert result.untrusted_count == 0

    def test_the_603312_shape_returns_the_real_set_not_one_event(
        self, cdm, registry
    ):
        """349 events, 348 marginally non-PSD: all 349 come back, none skipped.

        This is the regression in the shape it actually occurred. Ratios are
        spread across the observed live range (median 2.0e-9 to worst 5.37e-3).
        """
        ratios = np.logspace(np.log10(2.0e-9), np.log10(5.37e-3), 348)
        cdms = [_event(cdm, "event-clean", minute=0)]
        cdms += [
            _event(_with_min_eig_ratio(cdm, float(r)), f"event-{i}", minute=i + 1)
            for i, r in enumerate(ratios)
        ]
        result = _run(_client(cdms), registry)

        assert len(result.conjunctions) == 349, (
            "every event in the screened volume must be in the judged set")
        assert result.skipped == []
        assert result.untrusted_count == 0
        # Most were repaired; the ones already PSD to the tight tolerance are not.
        assert result.repaired_count >= 300

    def test_the_result_reports_that_the_set_is_complete(self, cdm, registry):
        result = _run(_client([_with_min_eig_ratio(cdm, 5.0e-3)]), registry)
        assert result.is_complete is True

    def test_a_genuinely_unparseable_cdm_is_still_skipped_and_reported(
        self, cdm, registry
    ):
        """SCRUM-458 does not make the parser credulous, only correct about PSD."""
        broken = copy.deepcopy(cdm)
        del broken["SAT1_CR_R"]
        result = _run(_client([broken]), registry)
        assert len(result.skipped) == 1
        assert result.conjunctions == []
        assert result.is_complete is False


# ---------------------------------------------------------------------------
# 1b. The dedupe identity: screening results carry no usable event id
# ---------------------------------------------------------------------------

class TestScreeningEventIdentity:
    """SCRUM-458, second finding.

    Screening result CDMs carry COMMENT_EVENT_ID = "0" and COMMENT_ID = the
    screening id, identically on every CDM. The live feed's event_key therefore
    returns one identity for the whole screening, and dedupe collapsed the entire
    screened set into a single conjunction. It was invisible while the PSD guard
    was already dropping nearly every CDM before dedupe ran.
    """

    def test_the_live_event_key_collapses_a_whole_screening(self, cdm):
        """The bug, stated directly, so a regression is unmistakable."""
        from common.leolabs_conjunction_list import event_key

        a = _parsed(_event(cdm, "a", minute=1))
        b = _parsed(_event(cdm, "b", minute=2))
        assert event_key(a) == event_key(b) == "event_id:0"

    def test_the_screening_key_separates_them(self, cdm):
        from common.leolabs_screening import screening_event_key

        a = _parsed(_event(cdm, "a", minute=1))
        b = _parsed(_event(cdm, "b", minute=2))
        assert screening_event_key(a) != screening_event_key(b)

    def test_distinct_secondaries_are_distinct_conjunctions(self, cdm, registry):
        cdms = [_event(cdm, f"e{i}", minute=i) for i in range(20)]
        result = _run(_client(cdms), registry)
        assert len(result.conjunctions) == 20

    def test_the_same_secondary_at_different_tcas_stays_two_conjunctions(
        self, cdm, registry
    ):
        """185 secondaries on 603312 recur at different TCAs; one has 29.

        Deduping on the secondary alone would have discarded 303 real conjunctions.
        """
        first = _event(cdm, "same", minute=1)
        second = _event(cdm, "same", minute=2)
        assert (first["SAT2_OBJECT_DESIGNATOR"]
                == second["SAT2_OBJECT_DESIGNATOR"])
        result = _run(_client([first, second]), registry)
        assert len(result.conjunctions) == 2

    def test_a_reissued_cdm_for_one_encounter_still_collapses(self, cdm, registry):
        """217 (secondary, TCA) pairs on 603312 appear more than once."""
        one = _event(cdm, "same", minute=5)
        result = _run(_client([one, copy.deepcopy(one)]), registry)
        assert len(result.conjunctions) == 1

    def test_the_live_listing_dedupe_is_untouched(self, cdm):
        """The live feed keeps event_key: its COMMENT_EVENT_ID is a real id."""
        from common.leolabs_conjunction_list import dedupe_by_event

        a = _parsed(_event(cdm, "a", minute=1, COMMENT_EVENT_ID="ev-1"))
        b = _parsed(_event(cdm, "b", minute=2, COMMENT_EVENT_ID="ev-2"))
        c = _parsed(_event(cdm, "c", minute=3, COMMENT_EVENT_ID="ev-1"))
        assert len(dedupe_by_event([a, b, c])) == 2


# ---------------------------------------------------------------------------
# 2. An unassessable event fails closed rather than disappearing
# ---------------------------------------------------------------------------

class TestUnassessableFailsClosed:
    def test_a_badly_non_psd_event_is_carried_not_dropped(self, cdm, registry):
        bad = _with_min_eig_ratio(cdm, 0.4)
        result = _run(_client([bad]), registry)
        assert result.skipped == []
        assert len(result.conjunctions) == 1
        assert result.untrusted_count == 1

    def test_an_unassessable_event_can_never_be_clear(self, cdm, policy):
        """The whole point: not assessable means not clear."""
        parsed = _parsed(_with_min_eig_ratio(cdm, 0.4),
                         COLLISION_PROBABILITY=1.0e-14, MISS_DISTANCE=400_000.0)
        verdict = evaluate_clear_contract([parsed], policy)
        assert verdict.clear is False
        assert "covariance_untrusted" in verdict.breaches[0]["limbs"]

    def test_the_mahalanobis_limb_is_not_evaluated_on_an_untrusted_covariance(
        self, cdm, policy
    ):
        """Inverting a non-PSD matrix can read as arbitrarily many sigma.

        That number would argue for CLEAR, so it is not computed at all rather
        than computed and then hopefully ignored.
        """
        parsed = _parsed(_with_min_eig_ratio(cdm, 0.4),
                         COLLISION_PROBABILITY=1.0e-14, MISS_DISTANCE=400_000.0)
        breach = evaluate_clear_contract([parsed], policy).breaches[0]
        assert breach["mahalanobis"] is None
        assert breach["covariance_untrusted"] is True
        assert breach["covariance_min_eigenvalue_m2"] < 0

    def test_an_unassessable_event_still_reports_its_geometric_miss(
        self, cdm, policy
    ):
        """The miss distance is covariance-free, so it survives the flag."""
        parsed = _parsed(_with_min_eig_ratio(cdm, 0.4), MISS_DISTANCE=8_000.0)
        verdict = evaluate_clear_contract([parsed], policy)
        assert verdict.closest_miss_km == pytest.approx(8.0)

    def test_an_unassessable_event_also_breaching_on_miss_reports_both_limbs(
        self, cdm, policy
    ):
        parsed = _parsed(_with_min_eig_ratio(cdm, 0.4), MISS_DISTANCE=500.0)
        limbs = evaluate_clear_contract([parsed], policy).breaches[0]["limbs"]
        assert "miss_distance" in limbs
        assert "covariance_untrusted" in limbs

    def test_one_unassessable_event_among_clear_ones_still_fails_the_screen(
        self, cdm, policy
    ):
        """It cannot be outvoted: the screen is a conjunction of every event."""
        clear = _parsed(_event(cdm, "e1"), COLLISION_PROBABILITY=1.0e-14,
                        MISS_DISTANCE=400_000.0)
        assert evaluate_clear_contract([clear], policy).clear is True
        untrusted = _parsed(_event(_with_min_eig_ratio(cdm, 0.4), "e2"),
                            COLLISION_PROBABILITY=1.0e-14,
                            MISS_DISTANCE=400_000.0)
        verdict = evaluate_clear_contract([clear, untrusted], policy)
        assert verdict.clear is False
        assert verdict.evaluated == 2


# ---------------------------------------------------------------------------
# 3. The repair never manufactures a clear
# ---------------------------------------------------------------------------

class TestRepairNeverClears:
    def test_a_repaired_event_breaching_on_miss_still_breaches(self, cdm, policy):
        """The safety direction, stated as a test: repair cannot rescue a breach."""
        parsed = _parsed(_with_min_eig_ratio(cdm, 5.0e-3), MISS_DISTANCE=500.0)
        assert parsed.covariance_repaired
        verdict = evaluate_clear_contract([parsed], policy)
        assert verdict.clear is False
        assert "miss_distance" in verdict.breaches[0]["limbs"]

    def test_a_repaired_event_breaching_on_pc_still_breaches(self, cdm, policy):
        parsed = _parsed(_with_min_eig_ratio(cdm, 5.0e-3),
                         COLLISION_PROBABILITY=policy.pc_maneuver_threshold * 10)
        assert parsed.covariance_repaired
        assert "pc" in evaluate_clear_contract([parsed], policy).breaches[0]["limbs"]

    def test_the_repair_does_not_change_the_verdict_on_an_already_clear_event(
        self, cdm, policy
    ):
        """Clipping a numerical negative is not supposed to move the decision."""
        clean = _parsed(cdm, COLLISION_PROBABILITY=1.0e-14,
                        MISS_DISTANCE=400_000.0)
        repaired = _parsed(_with_min_eig_ratio(cdm, 5.0e-3),
                           COLLISION_PROBABILITY=1.0e-14,
                           MISS_DISTANCE=400_000.0)
        assert repaired.covariance_repaired
        assert (evaluate_clear_contract([clean], policy).clear
                == evaluate_clear_contract([repaired], policy).clear)

    def test_repair_keeps_the_mahalanobis_on_the_same_side_of_the_threshold(
        self, cdm, policy
    ):
        """Measured live: over screening 603312's 1249 non-PSD CDMs, clipping to
        zero versus the conservative |lambda| reflection never moved an event
        across the Mahalanobis threshold. This asserts the property offline.
        """
        from common.leolabs_screening import _mahalanobis_distance

        parsed = _parsed(_with_min_eig_ratio(cdm, 5.0e-3))
        m_clip = _mahalanobis_distance(parsed)
        # the conservative alternative: reflect the negatives instead of clipping
        reflected = _parsed(_with_min_eig_ratio(cdm, 5.0e-3))
        cov = reflected.primary.cov_eci_m2
        w, v = np.linalg.eigh(0.5 * (cov + cov.T))
        reflected.primary.cov_eci_m2 = v @ np.diag(np.abs(w)) @ v.T
        m_reflect = _mahalanobis_distance(reflected)
        th = policy.mahalanobis_screen_threshold
        assert (m_clip <= th) == (m_reflect <= th)


# ---------------------------------------------------------------------------
# 4. The decision path will not certify CLEAR over an incomplete set
# ---------------------------------------------------------------------------

class TestIncompleteSetFailsClosed:
    def _check(self, monkeypatch, result):
        monkeypatch.setattr(ls, "run_screening", lambda *a, **k: result)
        return aa._run_on_demand_secondary_check(
            r_post_km=_R_POST, v_post_km_s=_V_POST, p_post_eci_km2=_P_POST,
            screening_epoch_utc=_EPOCH,
            policy=OperatorPolicy(operator_id="t", policy_version="v1"),
            cap=MagicMock(radius_m=1.0), primary_catalog_number="L2669",
        )

    def test_an_empty_screen_is_still_a_clear_sky(self, monkeypatch):
        check = self._check(monkeypatch, ScreeningResult(
            screening_id="scr-458", conjunctions=[], cdm_count=0))
        assert check.secondary_check_performed is True
        assert check.secondary_conjunction_clear is True

    def test_a_skipped_cdm_prevents_a_clear_verdict(self, monkeypatch):
        """A skipped CDM is an event that was never assessed.

        Certifying CLEAR over it is the same under-report SCRUM-458 exists to
        remove, just reached by a different skip reason, so it fails closed.
        """
        check = self._check(monkeypatch, ScreeningResult(
            screening_id="scr-458", conjunctions=[], cdm_count=1,
            skipped=[{"cdm_id": "c1", "event_id": "e1",
                      "reason": "CDM missing diagonal covariance element"}]))
        assert check.secondary_conjunction_clear is False
        assert "incomplete" in check.operator_note

    def test_the_skip_reason_is_named_in_the_operator_note(self, monkeypatch):
        check = self._check(monkeypatch, ScreeningResult(
            screening_id="scr-458", conjunctions=[], cdm_count=1,
            skipped=[{"cdm_id": "c1", "event_id": "e1", "reason": "unresolvable"}]))
        assert "unresolvable" in check.operator_note

    def test_a_clear_screen_over_repaired_covariance_says_so(
        self, monkeypatch, cdm
    ):
        """A clear verdict resting on repaired covariance is visible, not implied."""
        parsed = _parsed(_with_min_eig_ratio(cdm, 5.0e-3),
                         COLLISION_PROBABILITY=1.0e-14, MISS_DISTANCE=400_000.0)
        check = self._check(monkeypatch, ScreeningResult(
            screening_id="scr-458", conjunctions=[parsed], cdm_count=1))
        assert check.secondary_conjunction_clear is True
        assert "repaired" in check.operator_note

    def test_a_not_clear_screen_names_the_unassessable_count(
        self, monkeypatch, cdm
    ):
        parsed = _parsed(_with_min_eig_ratio(cdm, 0.4),
                         COLLISION_PROBABILITY=1.0e-14, MISS_DISTANCE=400_000.0)
        check = self._check(monkeypatch, ScreeningResult(
            screening_id="scr-458", conjunctions=[parsed], cdm_count=1))
        assert check.secondary_conjunction_clear is False
        assert "could not be fully assessed" in check.operator_note
        assert "M4 safe hold" in check.operator_note


# ---------------------------------------------------------------------------
# 5. The band is where the live data says it is
# ---------------------------------------------------------------------------

def test_the_repair_band_covers_the_worst_live_ratio():
    """5.37e-3 over screening 603312's 1255 result CDMs; the band is 1e-2.

    1.86x of margin. Enough that the observed spread does not sit on the boundary,
    small enough that the band is still two orders below the anisotropy itself.
    """
    assert _PSD_REPAIR_RTOL >= 1.8 * 5.37e-3
    assert _PSD_REPAIR_RTOL <= 5.0e-2, (
        "a band this wide would start repairing real covariance faults")
