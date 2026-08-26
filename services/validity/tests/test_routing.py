"""Tests for services/validity/routing.py, SCRUM-378."""
from __future__ import annotations

import pytest

from aps_math.observability import ValidityStatus, ValidityVerdict
from routing import RoutingDecision, route_validity_verdict


def _verdict(status, epsilon, epsilon_threshold=0.20, phenomenologies=None):
    return ValidityVerdict(
        status=status,
        epsilon=epsilon,
        epsilon_threshold=epsilon_threshold,
        weak_directions=[],
        phenomenologies_used=phenomenologies if phenomenologies is not None else ["TLE"],
    )


class TestNotEarnedIsUnconditional:
    """NOT_EARNED must block regardless of phenomenology or epsilon value."""

    def test_not_earned_blocks_even_with_high_epsilon(self):
        # Epsilon far above the review band -- must still block, since
        # NOT_EARNED short-circuits before the band question is asked.
        v = _verdict(ValidityStatus.NOT_EARNED, epsilon=0.99)
        assert route_validity_verdict(v) == RoutingDecision.BLOCKED

    def test_not_earned_blocks_with_non_tle_phenomenology(self):
        v = _verdict(ValidityStatus.NOT_EARNED, epsilon=0.05, phenomenologies=["optical"])
        assert route_validity_verdict(v) == RoutingDecision.BLOCKED


class TestReviewBandBoundaries:
    """The boundary-condition fix: both ends of [threshold, 0.25] are closed."""

    def test_epsilon_at_floor_routes_to_review(self):
        v = _verdict(ValidityStatus.EARNED, epsilon=0.20, epsilon_threshold=0.20)
        assert route_validity_verdict(v) == RoutingDecision.OPERATOR_REVIEW

    def test_epsilon_at_upper_band_edge_routes_to_review(self):
        # The specific off-by-boundary case flagged before writing this:
        # epsilon == 0.25 is NOT "above 0.25", so it must be REVIEW, not
        # AUTONOMOUS.
        v = _verdict(ValidityStatus.EARNED, epsilon=0.25)
        assert route_validity_verdict(v) == RoutingDecision.OPERATOR_REVIEW

    def test_epsilon_just_above_band_is_autonomous(self):
        v = _verdict(ValidityStatus.EARNED, epsilon=0.250001)
        assert route_validity_verdict(v) == RoutingDecision.AUTONOMOUS

    def test_epsilon_mid_band_routes_to_review(self):
        v = _verdict(ValidityStatus.EARNED, epsilon=0.225)
        assert route_validity_verdict(v) == RoutingDecision.OPERATOR_REVIEW

    def test_epsilon_well_above_band_is_autonomous(self):
        v = _verdict(ValidityStatus.EARNED, epsilon=0.73)
        assert route_validity_verdict(v) == RoutingDecision.AUTONOMOUS


class TestPhenomenologyGate:
    """The review band only applies to TLE-only arcs."""

    def test_non_tle_only_in_band_range_is_autonomous(self):
        # Same epsilon that would trigger review if TLE-only, but this
        # arc has other phenomenologies -- band doesn't apply, no
        # invented answer for the mixed-sensor case.
        v = _verdict(ValidityStatus.EARNED, epsilon=0.225, phenomenologies=["optical", "TLE"])
        assert route_validity_verdict(v) == RoutingDecision.AUTONOMOUS

    def test_optical_only_in_band_range_is_autonomous(self):
        v = _verdict(ValidityStatus.EARNED, epsilon=0.225, phenomenologies=["optical"])
        assert route_validity_verdict(v) == RoutingDecision.AUTONOMOUS

    @pytest.mark.parametrize("epsilon", [0.20, 0.225, 0.25])
    def test_tle_only_across_whole_band_routes_to_review(self, epsilon):
        v = _verdict(ValidityStatus.EARNED, epsilon=epsilon)
        assert route_validity_verdict(v) == RoutingDecision.OPERATOR_REVIEW


class TestAutonomousOutsideBand:
    def test_earned_high_epsilon_tle_only_is_autonomous(self):
        v = _verdict(ValidityStatus.EARNED, epsilon=0.90, phenomenologies=["TLE"])
        assert route_validity_verdict(v) == RoutingDecision.AUTONOMOUS
