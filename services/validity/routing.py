"""
services/validity/routing.py -- operator-review routing decision, SCRUM-378.

Per John (Slack, SCRUM-378 Comment 1): "it is not a third verdict status,
the interface stays EARNED and NOT_EARNED. Implement the TLE-only
routing as a review decision on top of an EARNED verdict at the
consumer, not inside the ValidityVerdict."

This module is that consumer-side decision. It takes an already-built
ValidityVerdict and returns a routing outcome; it does not compute or
modify the verdict itself (build_validity_verdict in aps_math is
untouched, see wrapper.py).

The 0.20 EARNED/NOT_EARNED floor is locked (ValidityVerdict already
enforces it via classify_validity). REVIEW_BAND_UPPER_EPSILON below is
a SEPARATE, provisional threshold -- John's exact words: "Use a fixed
band above the floor: TLE-only and epsilon in 0.20 to 0.25 routes to
operator review, autonomous above 0.25... since like the floor it is
provisional until we run on real geometries."

Per John, Slack, 2026-08-2x -- to be formalized as ADR-013. ADR-013 did
not exist as a file as of this session; check whether it has been
created before treating this citation as final.
"""
from __future__ import annotations

from enum import Enum

from aps_math.observability import ValidityStatus, ValidityVerdict

# Provisional, per John (Slack, pending ADR-013). Named and isolated here,
# not in aps_math or conventions.py, for the same reason phenomenologies_used's
# default lives at this boundary: it's a deployment-stage policy call, not a
# shared numerical convention two services must independently agree on, and
# it is explicitly expected to be retuned once real geometries are available.
REVIEW_BAND_UPPER_EPSILON: float = 0.25

# The one phenomenology combination the review band applies to. Per John:
# "TLE-only and epsilon in 0.20 to 0.25 routes to operator review." Any other
# phenomenology mix (i.e. onboard sensing contributing) is out of scope for
# this band -- SCRUM-333 doesn't say what should happen there, and nothing
# here should invent an answer.
_TLE_ONLY = ["TLE"]


class RoutingDecision(str, Enum):
    """Outcome of the operator-review routing check.

    Not part of the published ValidityVerdict interface -- this is a
    separate, planner-internal decision, per John's "not inside the
    ValidityVerdict" instruction.
    """

    BLOCKED = "BLOCKED"  # NOT_EARNED: no autonomous action, unconditionally.
    OPERATOR_REVIEW = "OPERATOR_REVIEW"  # EARNED, TLE-only, epsilon in the band.
    AUTONOMOUS = "AUTONOMOUS"  # EARNED, and out of the review band.


def route_validity_verdict(verdict: ValidityVerdict) -> RoutingDecision:
    """
    Decide what an EARNED/NOT_EARNED verdict means for autonomous action.

    Ordered three-way branch, deliberately not two independent boolean
    checks -- NOT_EARNED must short-circuit before the review-band
    question is ever asked, regardless of phenomenology or epsilon's
    exact value. Two independent checks risk being evaluated out of
    order or combined incorrectly; this makes the precedence explicit
    and untestable-wrong.

    Boundary is closed at both ends: epsilon == 0.20 (the EARNED floor
    itself) and epsilon == REVIEW_BAND_UPPER_EPSILON both route to
    review, not autonomous. Per John's literal wording, "autonomous
    above 0.25" -- "above" is strict, so epsilon == 0.25 has not cleared
    it and belongs to the review band, not autonomous.

    Args:
        verdict: an already-built ValidityVerdict (from
            build_validity_verdict, unmodified). This function does not
            build one itself.

    Returns:
        RoutingDecision.BLOCKED if verdict.status is NOT_EARNED.
        RoutingDecision.OPERATOR_REVIEW if EARNED, phenomenologies_used
            is TLE-only, and epsilon_threshold <= epsilon <=
            REVIEW_BAND_UPPER_EPSILON.
        RoutingDecision.AUTONOMOUS otherwise (EARNED, and either not
            TLE-only, or epsilon > REVIEW_BAND_UPPER_EPSILON).
    """
    if verdict.status == ValidityStatus.NOT_EARNED:
        return RoutingDecision.BLOCKED

    is_tle_only = list(verdict.phenomenologies_used) == _TLE_ONLY
    in_review_band = (
        verdict.epsilon_threshold <= verdict.epsilon <= REVIEW_BAND_UPPER_EPSILON
    )

    if is_tle_only and in_review_band:
        return RoutingDecision.OPERATOR_REVIEW

    return RoutingDecision.AUTONOMOUS
