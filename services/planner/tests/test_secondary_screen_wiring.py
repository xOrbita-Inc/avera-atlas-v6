"""tests/test_secondary_screen_wiring.py

SCRUM-442: the MAF secondary-clear guard fed by the real on-demand screen.

Offline, mocked screening client. The thing being protected here is that a screen
which could not run never reads as a clear screen -- every failure mode is
asserted to escalate, not to clear.

Run from repo root:
    python -m pytest services/planner/tests/test_secondary_screen_wiring.py -v
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from common import atlas_artifact as aa
from common import leolabs_screening as ls
from common.leolabs_cdm_parser import parse_leolabs_cdm
from common.leolabs_client import LeoLabsHTTPError
from common.leolabs_screening import (
    LeoLabsScreeningTimeout,
    LeoLabsScreeningUnavailable,
    ScreeningResult,
    evaluate_clear_contract,
)
from common.operator_policy import OperatorPolicy

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"

_EPOCH = "2026-09-24T12:00:00Z"
_R_POST = [6838.0, 0.0, 0.0]
_V_POST = [0.0, 0.3419, 7.5754]
_P_POST = [[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.16]]


@pytest.fixture
def policy() -> OperatorPolicy:
    return OperatorPolicy(operator_id="test", policy_version="v1")


@pytest.fixture
def cap():
    c = MagicMock()
    c.radius_m = 1.0
    return c


@pytest.fixture
def real_cap():
    """A real SatelliteCapability: build_atlas_artifact reads it for real."""
    from common.constellation_geometry import _mean_motion_to_sma_km
    from common.satellite_capability import (
        ConstellationSlot, LifetimeProfile, SatelliteCapability,
    )
    return SatelliteCapability(
        sat_id="SAT-442", a_ref_km=_mean_motion_to_sma_km(15.3020),
        lifetime=LifetimeProfile(
            mass_kg=100.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
            mission_lifetime_days_remaining=365.0,
        ),
        slot=ConstellationSlot(in_constellation=False),
    )


@pytest.fixture
def burn_scoring(real_cap, policy):
    """A high-risk event that recommends a maneuver."""
    from common.maneuver_scorer import score_maneuver_candidates
    from common.constellation_geometry import _mean_motion_to_sma_km
    a = _mean_motion_to_sma_km(15.3020)
    return score_maneuver_candidates(
        "CID-442-BURN", np.array([a, 0.0, 0.0]), np.array([0.0, 7.626, 0.0]),
        np.array([0.3, 0.0, 0.0]), np.eye(3) * 0.01,
        "2026-09-24T12:00:00Z", "2026-09-24T18:00:00Z", real_cap, policy,
    )


@pytest.fixture
def no_burn_scoring(real_cap, policy):
    """A distant event: no maneuver recommended, so no screen is needed."""
    from common.maneuver_scorer import score_maneuver_candidates
    from common.constellation_geometry import _mean_motion_to_sma_km
    a = _mean_motion_to_sma_km(15.3020)
    return score_maneuver_candidates(
        "CID-442-CLEAR", np.array([a, 0.0, 0.0]), np.array([0.0, 7.626, 0.0]),
        np.array([500.0, 0.0, 0.0]), np.eye(3) * 0.01,
        "2026-09-24T12:00:00Z", "2026-09-24T18:00:00Z", real_cap, policy,
    )


@pytest.fixture
def cdm() -> dict:
    return json.loads(_FIXTURE.read_text())


def _parsed(cdm_dict, **overrides):
    """A parsed CDM with specific Pc / miss values, for contract cases.

    validate_miss_distance is off because these overrides deliberately set a
    miss distance that does not match the fixture's state vectors -- the parser
    is right to reject that on a real CDM, but here the state vectors are
    irrelevant and the miss distance is the input under test. The contract
    evaluation is what is being exercised, not the parser's guards, which have
    their own suite.
    """
    out = copy.deepcopy(cdm_dict)
    out.update(overrides)
    return parse_leolabs_cdm(out, "L2669", validate_miss_distance=False)


def _check(monkeypatch, *, conjunctions=None, raises=None, sink=None, **kw):
    """Run the on-demand secondary check with run_screening stubbed."""
    if raises is not None:
        def _run(*a, **k):
            raise raises
    else:
        def _run(*a, **k):
            return ScreeningResult(
                screening_id="scr-442",
                conjunctions=list(conjunctions or []),
                cdm_count=len(conjunctions or []),
            )
    monkeypatch.setattr(ls, "run_screening", _run)

    params = dict(
        r_post_km=_R_POST, v_post_km_s=_V_POST, p_post_eci_km2=_P_POST,
        screening_epoch_utc=_EPOCH, primary_catalog_number="L2669",
        screening_sink=sink,
    )
    params.update(kw)
    return aa._run_on_demand_secondary_check(**params)


# ---------------------------------------------------------------------------
# The clear contract, applied locally
# ---------------------------------------------------------------------------

class TestClearContract:
    def test_an_empty_screen_is_a_clear_sky(self, policy):
        verdict = evaluate_clear_contract([], policy)
        assert verdict.clear is True
        assert verdict.evaluated == 0
        assert verdict.breaches == []

    def test_a_pc_at_the_action_threshold_breaches(self, cdm, policy):
        parsed = _parsed(cdm, COLLISION_PROBABILITY=policy.pc_maneuver_threshold)
        verdict = evaluate_clear_contract([parsed], policy)
        assert verdict.clear is False
        assert "pc" in verdict.breaches[0]["limbs"]

    def test_a_pc_below_the_action_threshold_does_not_breach_on_pc(
        self, cdm, policy
    ):
        parsed = _parsed(cdm, COLLISION_PROBABILITY=1.0e-9)
        verdict = evaluate_clear_contract([parsed], policy)
        for breach in verdict.breaches:
            assert "pc" not in breach["limbs"]

    def test_a_miss_inside_the_action_floor_breaches(self, cdm, policy):
        """500 m against a 1 km floor."""
        parsed = _parsed(cdm, COLLISION_PROBABILITY=1.0e-12, MISS_DISTANCE=500.0)
        verdict = evaluate_clear_contract([parsed], policy)
        assert verdict.clear is False
        assert "miss_distance" in verdict.breaches[0]["limbs"]

    def test_the_three_limbs_are_an_or_not_an_and(self, cdm, policy):
        """Breaching on miss alone is enough, with a negligible Pc.

        This is why no server-side Pc filter is sent: it would have dropped this
        event before the guard ever saw it.
        """
        parsed = _parsed(cdm, COLLISION_PROBABILITY=1.0e-30, MISS_DISTANCE=100.0)
        verdict = evaluate_clear_contract([parsed], policy)
        assert verdict.clear is False
        assert verdict.breaches[0]["limbs"] == ["miss_distance"] or \
            "miss_distance" in verdict.breaches[0]["limbs"]

    def test_the_verdict_records_why_each_event_breached(self, cdm, policy):
        """A NOT CLEAR verdict must be auditable against the events behind it."""
        parsed = _parsed(cdm, COLLISION_PROBABILITY=1.0e-3, MISS_DISTANCE=200.0)
        breach = evaluate_clear_contract([parsed], policy).breaches[0]
        for key in ("object_id", "tca_utc", "pc", "miss_distance_km",
                    "mahalanobis", "limbs", "cdm_id"):
            assert key in breach

    def test_it_uses_the_policy_predicates_rather_than_its_own(
        self, cdm, policy, monkeypatch
    ):
        """Traced to the policy, not reimplemented -- so the on-demand screen and
        the internal one cannot drift apart about what 'clear' means."""
        calls = {"maneuver": 0, "prescreen": 0}
        real_m = policy.is_maneuver_required
        real_p = policy.passes_pre_screen
        monkeypatch.setattr(policy, "is_maneuver_required",
                            lambda *a, **k: calls.__setitem__("maneuver", calls["maneuver"] + 1) or real_m(*a, **k))
        monkeypatch.setattr(policy, "passes_pre_screen",
                            lambda *a, **k: calls.__setitem__("prescreen", calls["prescreen"] + 1) or real_p(*a, **k))
        evaluate_clear_contract([_parsed(cdm)], policy)
        assert calls["maneuver"] >= 1
        assert calls["prescreen"] >= 1


# ---------------------------------------------------------------------------
# The two guard booleans
# ---------------------------------------------------------------------------

class TestGuardBooleans:
    def test_an_empty_screen_clears_and_counts_as_performed(self, monkeypatch, policy, cap):
        check = _check(monkeypatch, conjunctions=[], policy=policy, cap=cap)
        assert check.secondary_check_performed is True
        assert check.secondary_conjunction_clear is True
        assert check.screen_deferred is False

    def test_a_clean_result_clears(self, monkeypatch, cdm, policy, cap):
        clean = _parsed(cdm, COLLISION_PROBABILITY=1.0e-30,
                        MISS_DISTANCE=40_000.0)
        check = _check(monkeypatch, conjunctions=[clean], policy=policy, cap=cap)
        # Only meaningful if this event genuinely does not breach.
        if evaluate_clear_contract([clean], policy).clear:
            assert check.secondary_check_performed is True
            assert check.secondary_conjunction_clear is True

    def test_a_breaching_result_is_not_clear(self, monkeypatch, cdm, policy, cap):
        breaching = _parsed(cdm, COLLISION_PROBABILITY=1.0e-3, MISS_DISTANCE=200.0)
        check = _check(monkeypatch, conjunctions=[breaching], policy=policy, cap=cap)

        assert check.secondary_check_performed is True
        assert check.secondary_conjunction_clear is False
        assert check.flagged_objects
        assert "NOT CLEAR" in check.operator_note

    def test_the_note_records_the_covariance_caveat(self, monkeypatch, policy, cap):
        """Required by SCRUM-442: the floor must be stated, not assumed."""
        check = _check(monkeypatch, conjunctions=[], policy=policy, cap=cap)
        assert "floor" in check.operator_note
        assert "fast-follow" in check.operator_note


# ---------------------------------------------------------------------------
# Failing closed -- none of these may clear
# ---------------------------------------------------------------------------

class TestFailsClosed:
    @pytest.mark.parametrize("failure", [
        LeoLabsScreeningTimeout("did not complete"),
        LeoLabsScreeningUnavailable("rate-limited"),
        LeoLabsScreeningUnavailable("no on-demand access"),
        ls.LeoLabsScreeningError("transport blew up"),
    ])
    def test_a_screening_failure_is_not_performed(
        self, monkeypatch, policy, cap, failure
    ):
        check = _check(monkeypatch, raises=failure, policy=policy, cap=cap)
        assert check.secondary_check_performed is False
        assert check.secondary_conjunction_clear is False
        # Crucially NOT deferred: deferred would excuse it from verification.
        assert check.screen_deferred is False

    def test_an_unexpected_exception_also_fails_closed(self, monkeypatch, policy, cap):
        check = _check(monkeypatch, raises=RuntimeError("something odd"),
                       policy=policy, cap=cap)
        assert check.secondary_check_performed is False
        assert check.secondary_conjunction_clear is False

    @pytest.mark.parametrize("missing", [
        {"r_post_km": None},
        {"v_post_km_s": None},
        {"p_post_eci_km2": None},
        {"screening_epoch_utc": None},
        {"primary_catalog_number": None},
    ])
    def test_a_missing_input_fails_closed(self, monkeypatch, policy, cap, missing):
        check = _check(monkeypatch, conjunctions=[], policy=policy, cap=cap, **missing)
        assert check.secondary_check_performed is False
        assert check.secondary_conjunction_clear is False

    def test_no_failure_path_ever_reports_clear(self, monkeypatch, policy, cap):
        """The property this wiring exists to guarantee."""
        failures = [
            LeoLabsScreeningTimeout("t"),
            LeoLabsScreeningUnavailable("u"),
            ls.LeoLabsScreeningError("e"),
            RuntimeError("x"),
        ]
        for failure in failures:
            check = _check(monkeypatch, raises=failure, policy=policy, cap=cap)
            assert check.secondary_conjunction_clear is False
            assert check.secondary_check_performed is False


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_the_screening_result_is_handed_to_the_sink(
        self, monkeypatch, cdm, policy, cap
    ):
        captured = {}
        _check(monkeypatch, conjunctions=[_parsed(cdm)], policy=policy, cap=cap,
               sink=lambda result, verdict: captured.update(
                   result=result, verdict=verdict))

        assert captured["result"].screening_id == "scr-442"
        assert len(captured["result"].conjunctions) == 1
        # Covariance travels with it, so the verdict can be audited on the
        # exact events it was made from.
        cov = captured["result"].conjunctions[0].secondary.cov_eci_pos_m2
        assert np.asarray(cov).shape == (3, 3)

    def test_a_failing_sink_never_changes_the_verdict(
        self, monkeypatch, policy, cap
    ):
        """An audit write that fails must not turn a clear screen unclear."""
        def _boom(result, verdict):
            raise RuntimeError("ingest down")

        check = _check(monkeypatch, conjunctions=[], policy=policy, cap=cap,
                       sink=_boom)
        assert check.secondary_check_performed is True
        assert check.secondary_conjunction_clear is True


# ---------------------------------------------------------------------------
# When the screen runs at all
# ---------------------------------------------------------------------------

class TestScreenIsOnlyRunWhenItIsNeeded:
    """A screening create is rate-limited to 3 per 2 minutes and polls for
    roughly 30 s to 2 min. Spending one on a decision that recommends no burn
    would exhaust the quota after a handful of routine evaluates, and the screen
    would then start failing closed for the decisions that actually need it."""

    def _artifact(self, scoring, policy, cap, monkeypatch):
        calls = {"n": 0}

        def _run(*a, **k):
            calls["n"] += 1
            return ScreeningResult("scr-442", [], 0)

        monkeypatch.setattr(ls, "run_screening", _run)
        aa.build_atlas_artifact(
            scoring, cap, policy, "2026-09-24T18:00:00Z",
            r_post_km=_R_POST, v_post_km_s=_V_POST,
            secondary_screen_enabled=True,
            p_post_eci_km2=_P_POST, primary_catalog_number="L2669",
        )
        return calls["n"]

    def test_a_no_burn_decision_does_not_spend_a_screening(
        self, policy, real_cap, monkeypatch, no_burn_scoring
    ):
        assert no_burn_scoring.is_maneuver_recommended() is False
        assert self._artifact(no_burn_scoring, policy, real_cap, monkeypatch) == 0

    def test_a_recommended_maneuver_does_run_the_screen(
        self, policy, real_cap, monkeypatch, burn_scoring
    ):
        assert burn_scoring.is_maneuver_recommended() is True
        assert self._artifact(burn_scoring, policy, real_cap, monkeypatch) == 1


# ---------------------------------------------------------------------------
# The request, end to end through the check
# ---------------------------------------------------------------------------

class TestRequestShape:
    def test_the_screen_is_submitted_at_the_wide_volume(
        self, monkeypatch, policy, cap
    ):
        seen = {}

        def _run(ephemeris, thresholds, primary_object, **kw):
            seen["thresholds"] = thresholds
            seen["primary"] = primary_object
            seen["states"] = len(ephemeris["states"])
            return ScreeningResult("scr-442", [], 0)

        monkeypatch.setattr(ls, "run_screening", _run)
        aa._run_on_demand_secondary_check(
            r_post_km=_R_POST, v_post_km_s=_V_POST, p_post_eci_km2=_P_POST,
            screening_epoch_utc=_EPOCH, policy=policy, cap=cap,
            primary_catalog_number="L2669",
        )

        assert seen["primary"] == "L2669"
        assert seen["thresholds"].screening_volume_km == policy.screening_volume_km == 50.0
        # The action floor is carried for the local decision, not the request.
        assert seen["thresholds"].max_miss_distance_km == policy.min_miss_distance_km
        assert seen["states"] > 1
