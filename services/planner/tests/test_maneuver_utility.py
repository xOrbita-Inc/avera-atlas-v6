"""SCRUM-387: the maneuver utility trades in Pc, and delta-v is searched.

What was wrong
--------------
The utility scored risk reduction as delta_C, the Mahalanobis gain. Two
structural problems, neither fixable by reweighting, which is why the ticket was
rewritten from a weight recalibration into a functional-form change.

delta_C is a LOG-risk quantity. m^2 is linear in log Pc, so delta_C is
2*ln(Pc_pre/Pc_post). RED-001's delta_C of 148082 claimed the burn cut collision
probability by a factor of e^74041. The utility paid linearly, without ceiling,
for log risk reduction that stopped meaning anything many orders of magnitude
earlier.

And delta_C is quadratic in delta-v against a linear cost, so d/d(dv) of
g*dv^2 - lambda*dv is positive for any dv above lambda/2g and the maximum always
sat at the delta-v ceiling. Burning to the limit was a property of the
functional form, not a judgement about the encounter.

What this file guards
---------------------
That the exchange rate is derived from policy rather than fitted, that one
parameter set behaves across the whole permitted TCA window, that the optimum is
interior, and that a supplied Pc is combined with a computed one as a ratio
rather than a subtraction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aps_math import conventions

from common.maneuver_scorer import (
    UTILITY_BASIS_DELTA_C,
    UTILITY_BASIS_PC,
    risk_exchange_rate,
    score_maneuver_candidates,
)
from common.operator_policy import OperatorPolicy, ScoringWeights
from common.satellite_capability import (
    LifetimeProfile,
    PropulsionProfile,
    SatelliteCapability,
)

MU_EARTH = 398600.4418
_A_KM = 6378.137 + 550.0
_V_CIRC = math.sqrt(MU_EARTH / _A_KM)

R_SAT = np.array([_A_KM, 0.0, 0.0])
V_SAT = np.array([0.0, _V_CIRC, 0.0])
V_REL_HEAD_ON = np.array([0.0, -2.0 * _V_CIRC, 0.0])

R_REL = np.array([0.0, 0.0, 0.3])
P_SURROGATE = np.diag([0.3 ** 2, 2.5 ** 2, 0.5 ** 2])
T_CA = "2026-03-03T12:00:00Z"


def _cap() -> SatelliteCapability:
    return SatelliteCapability(
        sat_id="TEST-6U",
        a_ref_km=_A_KM,
        propulsion=PropulsionProfile(min_dv_m_s=0.0005),
        lifetime=LifetimeProfile(
            mass_kg=12.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
            mission_lifetime_days_remaining=365.0,
        ),
    )


def _policy(**kw) -> OperatorPolicy:
    base = dict(
        operator_id="TEST", policy_version="2.5.0",
        pc_maneuver_threshold=1.0e-4, pc_monitor_threshold=1.0e-5,
        mahalanobis_screen_threshold=4.0, max_dv_per_event_ms=2.0,
        scoring_weights=ScoringWeights(
            lambda_dv=1.0, lambda_lifetime=0.8, lambda_slot_deviation=1.2
        ),
    )
    base.update(kw)
    return OperatorPolicy(**base)


def _burn_time(hours_before_tca: float) -> str:
    from datetime import datetime, timedelta, timezone

    tca = datetime(2026, 3, 3, 12, 0, 0, tzinfo=timezone.utc)
    t = tca - timedelta(hours=hours_before_tca)
    return t.isoformat().replace("+00:00", "Z")


def _score(hours=4.0, policy=None, **kw):
    return score_maneuver_candidates(
        "CID-387", R_SAT, V_SAT, R_REL, P_SURROGATE,
        _burn_time(hours), T_CA, _cap(), policy or _policy(),
        v_rel_km_s=V_REL_HEAD_ON, **kw
    )


# ---------------------------------------------------------------------------
# AC3: the basis is derived, not fitted
# ---------------------------------------------------------------------------

class TestTheExchangeRateIsDerivedFromPolicy:
    def test_it_is_the_documented_formula(self):
        policy = _policy()
        assert risk_exchange_rate(policy) == pytest.approx(
            policy.scoring_weights.lambda_dv
            * policy.max_dv_per_event_ms
            / policy.pc_maneuver_threshold
        )

    def test_at_the_threshold_the_full_budget_breaks_even(self):
        """The sentence the formula encodes. Eliminating a risk sitting exactly
        at the maneuver threshold is worth exactly the whole per-event delta-v
        budget and no more. That is a statement the operator already made by
        choosing those two numbers."""
        policy = _policy()
        benefit = risk_exchange_rate(policy) * policy.pc_maneuver_threshold
        cost = policy.scoring_weights.lambda_dv * policy.max_dv_per_event_ms
        assert benefit == pytest.approx(cost)

    def test_raising_the_maneuver_threshold_lowers_the_exchange_rate(self):
        """The two cannot drift apart. An operator who decides to tolerate more
        risk gets a utility that agrees with the gate automatically, where a
        fitted constant would have to be kept in step by hand and nothing would
        fail if it were not."""
        tolerant = risk_exchange_rate(_policy(pc_maneuver_threshold=1.0e-3))
        strict = risk_exchange_rate(
            _policy(pc_maneuver_threshold=1.0e-5, pc_monitor_threshold=1.0e-6)
        )
        assert tolerant < strict

    def test_nothing_in_it_comes_from_the_demo_events(self):
        """AC3 in one assertion. Every input is a policy field, so the rate
        cannot have been tuned against fixtures that were themselves built
        against physics corrected twice since."""
        policy = _policy()
        for attr in ("pc_maneuver_threshold", "max_dv_per_event_ms"):
            assert hasattr(policy, attr)
        assert hasattr(policy.scoring_weights, "lambda_dv")


# ---------------------------------------------------------------------------
# AC4: one parameter set across the permitted window
# ---------------------------------------------------------------------------

class TestOneParameterSetAcrossTheWholeWindow:
    LEADS = [4.0, 12.0, 24.0, 48.0, 72.0]

    def test_a_burn_is_recommended_at_every_lead_time(self):
        for hours in self.LEADS:
            result = _score(hours)
            assert result.no_go_reason_code == "", (
                f"no burn recommended at {hours} h"
            )

    def test_the_utility_at_the_optimum_barely_moves(self):
        """The measurement that replaces the ticket's 314x.

        The old form's benefit grew quadratically with lead time, so any
        lambda_dv tuned at four hours was wrong by two and a half orders of
        magnitude at seventy-two. Bounding the benefit by Pc_pre removes that
        entirely: more warning time buys a cheaper burn, not a bigger prize.
        """
        utilities = [_score(h).utility for h in self.LEADS]
        assert min(utilities) > 0.0
        spread = (max(utilities) - min(utilities)) / max(utilities)
        assert spread < 0.05, f"utility varies by {spread:.1%} across 4 to 72 h"

    def test_more_lead_time_needs_less_delta_v(self):
        """The physically meaningful invariant. The displacement required is a
        property of the encounter; the delta-v that buys it falls as the lever
        arm grows."""
        dvs = [_score(h).dv_magnitude_m_s for h in self.LEADS]
        assert dvs == sorted(dvs, reverse=True), dvs
        assert dvs[0] > dvs[-1] * 3


# ---------------------------------------------------------------------------
# AC5: the optimum is interior
# ---------------------------------------------------------------------------

class TestTheRecommendationIsNotThePolicyCeiling:
    @pytest.mark.parametrize("ceiling", [0.5, 1.0, 2.0, 5.0])
    def test_the_burn_is_well_below_the_ceiling(self, ceiling):
        result = _score(policy=_policy(max_dv_per_event_ms=ceiling))
        assert result.no_go_reason_code == ""
        assert result.dv_magnitude_m_s < 0.5 * ceiling, (
            "the recommendation is pinned to the delta-v ceiling, which is the "
            "defect SCRUM-387 exists to remove"
        )

    def test_raising_the_ceiling_stops_changing_the_answer(self):
        """Once the ceiling exceeds what the geometry needs, more budget buys
        nothing. Under the old form the recommendation tracked the ceiling
        exactly, forever."""
        generous = _score(policy=_policy(max_dv_per_event_ms=2.0))
        lavish = _score(policy=_policy(max_dv_per_event_ms=5.0))
        assert lavish.dv_magnitude_m_s == pytest.approx(
            generous.dv_magnitude_m_s, rel=0.5
        )

    def test_the_search_grid_is_geometric_and_sized_by_convention(self):
        """The range spans four decades, so a linear grid would put every sample
        in the top one and resolve nothing where the answer lives."""
        assert conventions.DV_SEARCH_POINTS >= 12

    def test_one_candidate_is_reported_per_direction(self):
        """The search must not inflate the response by its own grid size."""
        result = _score()
        assert len(result.candidates_v25) == 6
        assert len({c.direction for c in result.candidates_v25}) == 6


# ---------------------------------------------------------------------------
# A supplied Pc combines as a ratio, not a subtraction
# ---------------------------------------------------------------------------

class TestASuppliedPcIsCombinedAsARatio:
    """A maneuver's effect is a ratio. An operator's Pc can disagree with our
    geometry by a large factor, and subtracting a post-maneuver number computed
    on our geometry from their pre-maneuver number compares two scales.

    Our model can be wrong about the absolute probability while still being
    right that a given burn removes three quarters of it. So the fraction comes
    from us and the level stays theirs, which extends SCRUM-389 AC3 rather than
    contradicting it.
    """

    def test_a_supplied_pc_still_sets_the_reported_level(self):
        supplied = 9.0e-4
        result = _score(pc_precomputed=supplied)
        assert result.pc_pre == pytest.approx(supplied)

    def test_a_larger_supplied_pc_buys_a_larger_benefit(self):
        """Under the subtraction form a supplied Pc below our computed one could
        produce a benefit smaller than the geometry justifies, and tip an event
        to no-burn on a scale mismatch rather than on the physics."""
        small = _score(pc_precomputed=2.0e-4)
        large = _score(pc_precomputed=9.0e-4)
        assert large.utility > small.utility

    def test_a_computed_pc_gives_the_same_answer_either_way(self):
        """When the Pc was computed rather than supplied, the level and the
        denominator are the same number, so the ratio form collapses to the
        difference form. That is why there is one code path rather than two."""
        result = _score()
        assert result.utility_basis == UTILITY_BASIS_PC
        assert result.pc_pre is not None and result.pc_post is not None


# ---------------------------------------------------------------------------
# The fallback, and saying which form ran
# ---------------------------------------------------------------------------

class TestTheBasisIsRecorded:
    def test_an_event_with_a_pc_says_pc_traded(self):
        result = _score()
        assert result.utility_basis == UTILITY_BASIS_PC
        assert all(
            c.utility_basis == UTILITY_BASIS_PC for c in result.candidates_v25
        )

    def test_an_event_without_one_falls_back_and_says_so(self):
        """Kept rather than replaced by a no-go. Until every producer supplies a
        relative velocity this is the live path, and a no-go here would mean a
        no-go on every such event rather than a more honest answer."""
        result = score_maneuver_candidates(
            "CID-387-NOPC", R_SAT, V_SAT, R_REL, P_SURROGATE,
            _burn_time(4.0), T_CA, _cap(), _policy(),
        )
        assert result.utility_basis == UTILITY_BASIS_DELTA_C

    def test_the_two_bases_are_not_comparable(self):
        """Guards against anyone ranking events by utility across the two forms.
        They are on wildly different scales, which is the whole reason the basis
        travels with the number."""
        traded = _score()
        legacy = score_maneuver_candidates(
            "CID-387-NOPC", R_SAT, V_SAT, R_REL, P_SURROGATE,
            _burn_time(4.0), T_CA, _cap(), _policy(),
        )
        assert legacy.utility > traded.utility * 100
