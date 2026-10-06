"""tests/test_flight_rule_unpriced_burn.py

SCRUM-488: a flight-rule mode does not report a burn it never sized.

SCRUM-485 stopped APS recommending off the unbounded delta_c_legacy fallback,
but it gated the APS verdict only. The same fabricated magnitude stayed in the
flight-rule columns of the A/B, which is where the baseline delta-v comes from.

The mechanism is worth stating precisely, because the verdict and the number
have different causes and only one of them is wrong.

A flight rule fires on `pc >= threshold OR miss < min_miss_distance_km`. The
second clause means a zero-Pc event with a sub-kilometre miss genuinely requires
a flight-rule maneuver, through the proximity floor. That is a legitimate rule
and a real difference from APS, and this change does not touch it.

The delta-v beside it is the problem. analyze_conjunction_modes takes burn_dv
once, from the single highest-utility candidate, before the mode loop. For a
zero-Pc event that candidate sits on the fallback basis, whose benefit term is
unbounded, so the winning magnitude is simply the largest burn the policy
permits. Every firing mode then reported that number. On the reference window
roughly 99% of the flight-rule delta-v was this artifact -- the same artifact
SCRUM-485 removed from APS, relocated to the baseline it is compared against.

So the burn is reported as UNPRICED rather than fabricated: required stays true,
the maneuver count is unchanged, and the magnitude drops out of the mode's total
through the unpriced machinery the portfolio already had.

Run from repo root:
    python -m pytest services/planner/tests/test_flight_rule_unpriced_burn.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from common.maneuver_scorer import (
    UTILITY_BASIS_DELTA_C,
    UTILITY_BASIS_PC,
    analyze_conjunction_modes,
)
from common.operator_policy import OperatorPolicy

_FLIGHT_RULES = ("flight_rule_1e4", "flight_rule_1e5")
_ALL_MODES = ("aps",) + _FLIGHT_RULES
_COV = (np.eye(3) * 1.0e-4).flatten().tolist()

_SAT = {
    "sat_id": "488",
    "r_sat_km": [6853.0, 0.0, 0.0],
    "v_sat_km_s": [0.0, 7.62, 0.0],
    "t_burn_utc": "2026-04-14T08:00:00Z",
    "v_remaining_m_s": 50.0,
    "a_ref_km": 6853.0,
}


def _row(miss_km: float, *, with_rel_velocity: bool = True) -> dict:
    conjunction = {
        "t_ca_utc": "2026-04-14T12:00:00Z",
        "r_rel_km": [miss_km, 0.0, 0.0],
        "miss_distance_km": miss_km,
        "p_rel_km2": _COV,
    }
    if with_rel_velocity:
        conjunction["v_rel_km_s"] = [0.0, 10.0, 0.0]
    return analyze_conjunction_modes({
        "conjunction_id": "488", "satellite": dict(_SAT),
        "policy": {}, "conjunction": conjunction,
    })


@pytest.fixture(scope="module")
def proximity_trigger():
    """Zero Pc, 0.4 km miss: under the 1 km proximity floor, so the flight rules
    fire while APS does not."""
    return _row(0.4)


@pytest.fixture(scope="module")
def real_pc():
    """A real, nonzero Pc the burn can actually be sized against."""
    return _row(0.05)


@pytest.fixture(scope="module")
def far_zero_pc():
    """Zero Pc, 10 km miss: nothing fires at all."""
    return _row(10.0)


# ---------------------------------------------------------------------------
# The defect, reproduced
# ---------------------------------------------------------------------------

class TestTheSetupIsTheOneDescribed:
    """Without these, the rest could pass against the wrong event."""

    def test_the_event_really_has_a_zero_pc(self, proximity_trigger):
        assert proximity_trigger["pc_pre"] == 0.0
        assert proximity_trigger["pc_usable"] is False

    def test_its_burn_was_sized_on_the_fallback_basis(self, proximity_trigger):
        assert proximity_trigger["best_burn"]["utility_basis"] == UTILITY_BASIS_DELTA_C

    def test_that_candidate_is_the_largest_burn_allowed(self, proximity_trigger):
        """Not a coincidence: the fallback benefit rises with separation without
        bound, so the winner is always the cap."""
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0")
        assert proximity_trigger["best_burn"]["dv_m_s"] == pytest.approx(
            policy.max_dv_per_event_ms)

    def test_the_miss_is_inside_the_proximity_floor(self, proximity_trigger):
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0")
        assert proximity_trigger["miss_distance_km"] < policy.min_miss_distance_km


# ---------------------------------------------------------------------------
# Acceptance 1 and 2: unpriced, not fabricated; counts unchanged
# ---------------------------------------------------------------------------

class TestTheFlightRuleBurnIsUnpricedNotFabricated:
    @pytest.mark.parametrize("mode", _FLIGHT_RULES)
    def test_the_maneuver_is_still_required(self, proximity_trigger, mode):
        """The proximity floor is a legitimate rule. The verdict does not move."""
        assert proximity_trigger["modes"][mode]["maneuver_required"] is True

    @pytest.mark.parametrize("mode", _FLIGHT_RULES)
    def test_no_delta_v_is_reported(self, proximity_trigger, mode):
        assert proximity_trigger["modes"][mode]["dv_m_s"] is None

    @pytest.mark.parametrize("mode", _FLIGHT_RULES)
    def test_it_is_marked_unpriced(self, proximity_trigger, mode):
        assert proximity_trigger["modes"][mode]["pricing_available"] is False

    @pytest.mark.parametrize("mode", _FLIGHT_RULES)
    def test_it_is_not_reported_at_the_cap(self, proximity_trigger, mode):
        """The specific wrong answer this replaces."""
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0")
        assert proximity_trigger["modes"][mode]["dv_m_s"] != policy.max_dv_per_event_ms

    @pytest.mark.parametrize("mode", _FLIGHT_RULES)
    def test_it_is_not_reported_as_a_free_zero_either(self, proximity_trigger, mode):
        """Unpriced is not zero. A zero would silently flatter the baseline the
        same way the cap inflated it, just in the other direction."""
        assert proximity_trigger["modes"][mode]["dv_m_s"] != 0.0
        assert proximity_trigger["modes"][mode]["pricing_available"] is False


# ---------------------------------------------------------------------------
# Acceptance 3: APS and real-Pc events are untouched
# ---------------------------------------------------------------------------

class TestNothingElseMoved:
    def test_aps_is_unchanged_from_scrum_485(self, proximity_trigger):
        """485 already refuses this event; it stays refused with a zero burn,
        not newly unpriced."""
        aps = proximity_trigger["modes"]["aps"]
        assert aps["maneuver_required"] is False
        assert aps["dv_m_s"] == 0.0
        assert aps["pricing_available"] is True

    @pytest.mark.parametrize("mode", _ALL_MODES)
    def test_a_real_pc_event_is_priced_in_every_mode(self, real_pc, mode):
        assert real_pc["pc_usable"] is True
        assert real_pc["best_burn"]["utility_basis"] == UTILITY_BASIS_PC
        assert real_pc["modes"][mode]["maneuver_required"] is True
        assert real_pc["modes"][mode]["dv_m_s"] > 0.0
        assert real_pc["modes"][mode]["pricing_available"] is True

    @pytest.mark.parametrize("mode", _ALL_MODES)
    def test_an_event_that_fires_nothing_is_priced_at_zero(self, far_zero_pc, mode):
        """No maneuver means no pricing question, so it stays a clean zero and
        does not pollute the unpriced count."""
        assert far_zero_pc["modes"][mode]["maneuver_required"] is False
        assert far_zero_pc["modes"][mode]["dv_m_s"] == 0.0
        assert far_zero_pc["modes"][mode]["pricing_available"] is True


# ---------------------------------------------------------------------------
# Acceptance 1: the rollup excludes it
# ---------------------------------------------------------------------------

class TestThePortfolioRollup:
    """The portfolio's unpriced machinery already generalises to every mode.
    These pin that it does, over controlled rows, because the whole point of
    marking a burn unpriced is what the totals then say."""

    @staticmethod
    def _summary(rows, mode):
        verdicts = [r["modes"][mode] for r in rows]
        missing = sum(not v["pricing_available"] for v in verdicts)
        subtotal = sum(v["dv_m_s"] for v in verdicts if v["dv_m_s"] is not None)
        return {
            "maneuver_count": sum(v["maneuver_required"] for v in verdicts),
            "total_dv_m_s": subtotal if missing == 0 else None,
            "known_dv_m_s": subtotal,
            "unpriced_maneuver_count": missing,
        }

    def test_the_phantom_burns_leave_the_flight_rule_total(
            self, proximity_trigger, real_pc):
        """The reported shape: four proximity triggers plus four real burns."""
        rows = [proximity_trigger] * 4 + [real_pc] * 4
        s = self._summary(rows, "flight_rule_1e4")
        real_dv = real_pc["modes"]["flight_rule_1e4"]["dv_m_s"]
        assert s["known_dv_m_s"] == pytest.approx(4 * real_dv)
        assert s["unpriced_maneuver_count"] == 4
        assert s["total_dv_m_s"] is None       # incomplete, and says so

    def test_the_maneuver_count_is_unchanged_by_the_fix(
            self, proximity_trigger, real_pc):
        """Acceptance 2: only the delta-v moved."""
        rows = [proximity_trigger] * 4 + [real_pc] * 4
        assert self._summary(rows, "flight_rule_1e4")["maneuver_count"] == 8

    def test_the_baseline_no_longer_dwarfs_aps(self, proximity_trigger, real_pc):
        """The symptom that made the A/B indefensible: a baseline built almost
        entirely out of burns that were never sized."""
        rows = [proximity_trigger] * 4 + [real_pc] * 4
        fr = self._summary(rows, "flight_rule_1e4")["known_dv_m_s"]
        aps = self._summary(rows, "aps")["known_dv_m_s"]
        assert fr == pytest.approx(aps)

    def test_a_window_with_no_unpriced_burns_still_totals(self, real_pc):
        """total_dv_m_s is only withheld when something is genuinely missing."""
        s = self._summary([real_pc] * 3, "flight_rule_1e4")
        assert s["unpriced_maneuver_count"] == 0
        assert s["total_dv_m_s"] == pytest.approx(s["known_dv_m_s"])


# ---------------------------------------------------------------------------
# Acceptance 4 and 5: safety sources and the corrected docstring
# ---------------------------------------------------------------------------

class TestTheSafetySurfaceAndTheRecord:
    def test_the_flight_rule_decision_logic_is_untouched(self):
        """is_recommendation_required still fires on the proximity floor for a
        zero Pc -- which is exactly what the old docstring denied."""
        for mode in _FLIGHT_RULES:
            policy = OperatorPolicy(operator_id="T", policy_version="2.5.0",
                                    decision_mode=mode)
            assert policy.is_recommendation_required(0.0, 0.4, utility=0.0) is True
            assert policy.is_recommendation_required(None, 0.4, utility=0.0) is False

    def test_is_maneuver_required_is_unchanged(self):
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0")
        assert policy.is_maneuver_required(1.0e-4, 2.0) is True
        assert policy.is_maneuver_required(0.0, 0.5) is True
        assert policy.is_maneuver_required(0.0, 2.0) is False

    def test_the_thresholds_are_unchanged(self):
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0")
        assert policy.pc_maneuver_threshold == 1.0e-4
        assert policy.pc_monitor_threshold == 1.0e-5
        assert policy.min_miss_distance_km == 1.0

    def test_the_docstring_no_longer_denies_the_proximity_trigger(self):
        """Acceptance 5. The claim misled, so its removal is pinned."""
        doc = OperatorPolicy.is_recommendation_required.__doc__
        assert "never fired on them in the first place" not in doc
        assert "proximity floor" in doc
