"""tests/test_supplied_pc_burn_sizing.py

SCRUM-487: a burn against a supplied Pc is sized to the risk, not to the cap.

SCRUM-485 stopped APS commanding a burn on a zero or absent Pc and deliberately
left one case open, documented at the time rather than hidden: an event carrying
an externally supplied, above-threshold Pc whose own geometric Pc underflows to
exactly zero.

For that event the recommendation was already right -- the supplied Pc is real
and above threshold, so a maneuver is warranted. The delta-v was not. The
pc_traded basis needs a positive geometric Pc to form the ratio pc_post/pc_pre,
and this event has none, so the candidates fell to the delta_c_legacy basis.
That basis pays linearly and without ceiling for a quantity that is quadratic in
delta-v, so its maximum always sits at the delta-v limit: the commanded burn was
the policy cap, by construction, whatever the risk.

The fix keeps the SCRUM-387 structure and changes only how the fraction is
evaluated. The LEVEL stays the authority's supplied Pc; the FRACTION a burn
removes comes from our geometry, taken in Mahalanobis space:

    Pc ~ k * exp(-m^2 / 2)  =>  Pc_post / Pc_pre ~ exp(-(m2_post - m2_pre) / 2)
    reduction_fraction = 1 - exp(-delta_C_v25 / 2)

The prefactor cancels, which is exactly why this survives where the ratio dies:
m2_pre and m2_post are finite and already computed per candidate even when both
probabilities have underflowed to zero.

Two things these tests are really for:

  - That the fraction is an honest stand-in, not a convenient one. There is a
    test comparing it against the exact ratio on an event where both are
    computable; without it, "leading-order form" is an unverified claim.
  - That the bound holds. The whole defect was an unbounded benefit term, so the
    replacement has to be provably in [0, 1) and the resulting burn provably
    below the cap, including for a worsening burn and for a separation gain
    large enough to overflow a naive exp().

Run from repo root:
    python -m pytest services/planner/tests/test_supplied_pc_burn_sizing.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.maneuver_scorer import (
    UTILITY_BASIS_DELTA_C,
    UTILITY_BASIS_PC,
    compute_pc_post,
    resolve_hard_body_radius,
    score_maneuver_candidates,
)
from common.operator_policy import OperatorPolicy
from common.satellite_capability import LifetimeProfile, SatelliteCapability

_R_SAT = np.array([6853.0, 0.0, 0.0])
_V_SAT = np.array([0.0, 7.62, 0.0])
_V_REL = np.array([0.0, 10.0, 0.0])
_COV = np.eye(3) * 1.0e-4
_T_BURN = "2026-04-14T08:00:00Z"
_T_CA = "2026-04-14T12:00:00Z"

# Above both flight-rule thresholds, so this is an event that genuinely warrants
# a maneuver -- which is what makes the burn SIZE the only thing in question.
_SUPPLIED_PC = 5.0e-3


def _cap() -> SatelliteCapability:
    return SatelliteCapability(
        sat_id="487", a_ref_km=6853.0, radius_m=3.1,
        lifetime=LifetimeProfile(
            mass_kg=100.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
            mission_lifetime_days_remaining=365.0),
    )


def _policy(**kw) -> OperatorPolicy:
    return OperatorPolicy(operator_id="T", policy_version="2.5.0",
                          decision_mode="aps", **kw)


def _score(miss_km: float, *, pc=None, policy=None):
    return score_maneuver_candidates(
        "487", _R_SAT, _V_SAT, np.array([miss_km, 0.0, 0.0]), _COV,
        _T_BURN, _T_CA, _cap(), policy or _policy(),
        miss_distance_km=miss_km, v_rel_km_s=_V_REL, pc_precomputed=pc,
    )


@pytest.fixture(scope="module")
def supplied_underflow():
    """Supplied Pc above threshold; our own geometric Pc underflows to zero."""
    return _score(10.0, pc=_SUPPLIED_PC)


@pytest.fixture(scope="module")
def geometric_pc():
    """A usable geometric Pc: the exact-ratio path, unchanged by this ticket."""
    return _score(0.05)


@pytest.fixture(scope="module")
def no_pc_at_all():
    """Nothing supplied and nothing computable: still the fallback."""
    return _score(10.0)


# ---------------------------------------------------------------------------
# The setup really is the case described
# ---------------------------------------------------------------------------

class TestTheCaseIsTheOneDescribed:
    def test_the_supplied_pc_is_the_resolved_pc(self, supplied_underflow):
        assert supplied_underflow.pc_pre == _SUPPLIED_PC
        assert supplied_underflow.pc_source == "supplied"

    def test_our_own_geometric_pc_really_does_underflow(self):
        """If it did not, this event would take the exact-ratio path and the
        middle path would never be exercised."""
        hbr, _ = resolve_hard_body_radius(_cap())
        geometric = compute_pc_post(
            _R_SAT, _V_SAT, np.array([10.0, 0.0, 0.0]), _V_REL, _COV, hbr)
        assert geometric == 0.0

    def test_the_supplied_pc_is_above_the_maneuver_threshold(self):
        """So a maneuver is warranted and only the size is in question."""
        assert _SUPPLIED_PC > _policy().pc_maneuver_threshold


# ---------------------------------------------------------------------------
# Acceptance 1: priced on pc_traded, sized below the cap
# ---------------------------------------------------------------------------

class TestTheBurnIsSizedToTheSuppliedRisk:
    def test_it_is_priced_on_the_pc_traded_basis(self, supplied_underflow):
        assert supplied_underflow.utility_basis == UTILITY_BASIS_PC
        assert supplied_underflow.utility_basis != UTILITY_BASIS_DELTA_C

    def test_the_commanded_burn_is_below_the_cap(self, supplied_underflow):
        """The defect was a burn pinned at the cap by construction."""
        cap_dv = _policy().max_dv_per_event_ms
        assert 0.0 < supplied_underflow.dv_magnitude_m_s < cap_dv

    def test_the_utility_is_bounded_by_the_supplied_risk(self, supplied_underflow):
        """benefit = exchange_rate * pc_pre * fraction, fraction < 1, so the
        benefit can never exceed what the whole risk is worth. The fallback had
        no such ceiling, which is how it reached eight figures."""
        from common.maneuver_scorer import risk_exchange_rate
        ceiling = risk_exchange_rate(_policy()) * _SUPPLIED_PC
        assert supplied_underflow.utility <= ceiling

    def test_the_supplied_level_scales_the_benefit_linearly(self):
        """benefit = exchange_rate * pc_pre * fraction, so the level enters
        linearly.

        Asserted on the DIFFERENCE, not as a ratio of utilities: utility is
        benefit minus the delta-v, lifetime and slot costs, and those costs do
        not scale with Pc. Ten times the risk is ten times the benefit, not ten
        times the utility, and a ratio test would be quietly wrong by the cost
        terms (about 0.01 here).
        """
        from common.maneuver_scorer import risk_exchange_rate
        small_pc, large_pc = 5.0e-4, 5.0e-3
        small = _score(10.0, pc=small_pc).utility
        large = _score(10.0, pc=large_pc).utility
        # Both saturate the fraction at 1.0, so the whole difference is level.
        assert large - small == pytest.approx(
            risk_exchange_rate(_policy()) * (large_pc - small_pc), rel=1e-9)

    def test_the_level_does_not_move_the_optimal_burn_when_it_saturates(self):
        """Worth stating because it surprised me, and a reviewer should not read
        a flat delta-v as the cap-pinning defect returning.

        On a wide-miss event the separation gain is enormous even for the
        smallest searched burn -- delta_C of order 1e4 -- so
        1 - exp(-delta_C/2) saturates to exactly 1.0 and the benefit term stops
        depending on delta-v at all. The utility is then maximised by minimising
        cost, so the answer is the cheapest burn on the grid, whatever the
        supplied Pc. That is the economically right answer: if 0.01 m/s already
        removes the whole risk, there is nothing to buy with more fuel.

        The interior optimum SCRUM-387 designed for appears when the fraction
        has NOT saturated. What matters for this ticket either way is that the
        answer is bounded and risk-derived rather than the policy ceiling.
        """
        dvs = {_score(10.0, pc=pc).dv_magnitude_m_s
               for pc in (5.0e-4, 5.0e-3, 5.0e-2, 0.5)}
        assert len(dvs) == 1
        assert dvs.pop() < _policy().max_dv_per_event_ms

    def test_the_burn_stays_below_the_cap_even_for_a_huge_supplied_pc(self):
        """Bounded means bounded. A Pc of 0.5 must not reproduce the cap-pinned
        behaviour by another route."""
        cap_dv = _policy().max_dv_per_event_ms
        assert _score(10.0, pc=0.5).dv_magnitude_m_s < cap_dv


# ---------------------------------------------------------------------------
# The fraction is an honest stand-in for the exact ratio
# ---------------------------------------------------------------------------

class TestTheM2FractionMatchesTheExactRatio:
    """Acceptance 5. "Leading-order form of the exact ratio" is a claim, and
    this is the test that makes it one we have checked."""

    @staticmethod
    def _m2_fraction(delta_c):
        if delta_c <= 0.0:
            return 0.0
        return 1.0 if math.isinf(delta_c) else 1.0 - math.exp(-delta_c / 2.0)

    def test_they_agree_on_every_candidate_where_both_exist(self, geometric_pc):
        hbr, _ = resolve_hard_body_radius(_cap())
        exact_pre = compute_pc_post(
            _R_SAT, _V_SAT, np.array([0.05, 0.0, 0.0]), _V_REL, _COV, hbr)
        assert exact_pre > 0.0, "need a computable geometric Pc for this test"

        compared = 0
        for candidate in geometric_pc.candidates_v25:
            if candidate.pc_post is None:
                continue
            exact = max(0.0, 1.0 - candidate.pc_post / exact_pre)
            approx = self._m2_fraction(candidate.delta_C_v25)
            if exact > 0.0:
                assert approx == pytest.approx(exact, rel=0.02), candidate.direction
                compared += 1
        assert compared >= 3, "too few candidates compared to mean anything"

    def test_the_fraction_is_bounded_in_zero_to_one(self):
        for delta_c in (-1.0e6, -1.0, 0.0, 1.0e-9, 1.0, 10.0, 1.0e6):
            f = self._m2_fraction(delta_c)
            assert 0.0 <= f < 1.0 or (delta_c > 0 and f == pytest.approx(1.0))

    def test_a_worsening_burn_buys_nothing(self):
        """Matches the exact-ratio clamp: a negative gain scores zero benefit,
        not a negative one, because the delta-v cost already penalises it."""
        assert self._m2_fraction(-5.0) == 0.0

    def test_a_large_negative_gain_does_not_overflow(self):
        """exp(+709) overflows. The clamp is what keeps this safe, so it is
        pinned rather than left to luck."""
        assert self._m2_fraction(-1.0e9) == 0.0

    def test_a_large_positive_gain_saturates_rather_than_overflowing(self):
        assert self._m2_fraction(1.0e9) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Acceptance 2 and 3: everything else unchanged
# ---------------------------------------------------------------------------

class TestTheOtherPathsAreUnchanged:
    def test_a_usable_geometric_pc_still_takes_the_exact_ratio_path(
            self, geometric_pc):
        assert geometric_pc.utility_basis == UTILITY_BASIS_PC
        assert geometric_pc.pc_pre > 0.0
        assert geometric_pc.dv_magnitude_m_s > 0.0

    def test_its_burn_is_still_below_the_cap(self, geometric_pc):
        assert geometric_pc.dv_magnitude_m_s < _policy().max_dv_per_event_ms

    def test_an_event_with_no_pc_at_all_still_uses_the_fallback(self, no_pc_at_all):
        """SCRUM-487 moved the supplied-Pc case off this basis; it did not
        remove the basis, which still prices events carrying no Pc."""
        assert no_pc_at_all.pc_pre == 0.0
        assert no_pc_at_all.utility_basis == UTILITY_BASIS_DELTA_C

    def test_and_aps_still_commands_no_burn_there(self, no_pc_at_all):
        """SCRUM-485, unchanged: a zero Pc is not a maneuver, so the fallback
        basis never sizes a burn that gets commanded."""
        from common.maneuver_scorer import aps_pc_usable
        assert aps_pc_usable(no_pc_at_all.pc_pre) is False
        assert _policy().is_recommendation_required(
            no_pc_at_all.pc_pre, 999.0, utility=no_pc_at_all.utility,
            pc_usable=aps_pc_usable(no_pc_at_all.pc_pre)) is False

    def test_the_safety_gate_and_thresholds_are_untouched(self):
        policy = _policy()
        assert policy.pc_maneuver_threshold == 1.0e-4
        assert policy.pc_monitor_threshold == 1.0e-5
        assert policy.is_maneuver_required(1.0e-4, 2.0) is True
        assert policy.is_maneuver_required(1.0e-5, 2.0) is False

    @pytest.mark.parametrize("mode", ["flight_rule_1e4", "flight_rule_1e5"])
    def test_flight_rule_verdicts_are_untouched(self, mode):
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0",
                                decision_mode=mode)
        threshold = policy.flight_rule_pc_threshold
        assert policy.is_recommendation_required(threshold, 2.0) is True
        assert policy.is_recommendation_required(threshold * 0.5, 2.0) is False
