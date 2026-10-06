"""tests/test_real_hbr_from_cdm.py

SCRUM-489: live Pc is computed against the CDM's real object radii.

Every live Pc was computed with the 15 m combined hard-body radius screening
floor. The floor itself is not wrong -- it is deliberately inflated to absorb
attitude uncertainty and an unknown secondary -- but SCRUM-394 established that
the 1e-4 maneuver threshold is applied in practice against realistic
sum-of-radii radii. The two were never calibrated together, so pairing them
inflated every Pc by roughly 17x to 56x and produced screening-conservative
maneuver flags rather than calibrated risk.

A LeoLabs CDM states both objects' measured radii. On the sample fixture that is
3.1 m and 0.5 m, so 3.6 m combined -- against a 15 m floor that would imply a
secondary about 14.8 m across, which is larger than anything in the catalogue.

So the floor's SCOPE changes, not its value: a conjunction that supplies the
secondary's real radius is scored on the real combined radius, and the floor
applies only when the secondary's size is genuinely unknown.

Two things these tests guard above all:

  - The zero-radius trap. A CDM that omits a radius reports it as 0.0, and a
    zero hard-body radius drives Pc to zero and suppresses maneuvers. Absence of
    a measurement must fall back to the floor, never be read as a point-sized
    object. That is the one direction an error here must not go.
  - Where the Pc comes from is unchanged. We still compute our own Pc from the
    real geometry; only the radius input moved. The CDM's published Pc stays a
    cross-check, and the agreement test below is what makes that cross-check
    worth having.

Run from repo root:
    python -m pytest services/planner/tests/test_real_hbr_from_cdm.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aps_math import conventions
from common.leolabs_cdm_parser import parse_leolabs_cdm
from common.leolabs_evaluate import build_evaluate_request
from common.maneuver_scorer import (
    evaluate_conjunction_v25,
    resolve_hard_body_radius,
)
from common.satellite_capability import SatelliteCapability

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"
_OUR_ID = "L2669"

_FLOOR = conventions.DEFAULT_COMBINED_HBR_M


@pytest.fixture(scope="module")
def cdm() -> dict:
    return json.loads(_FIXTURE.read_text())


@pytest.fixture(scope="module")
def parsed(cdm):
    return parse_leolabs_cdm(cdm, _OUR_ID)


@pytest.fixture(scope="module")
def scored(parsed):
    """The live path, end to end, as the A/B runs it."""
    return evaluate_conjunction_v25(
        build_evaluate_request(parsed, sat_id="SWARM B", v_remaining_m_s=25.0))


@pytest.fixture(scope="module")
def scored_without_radius(parsed):
    """The same event with the secondary radius withheld -- the old behaviour,
    and what a source that does not state object sizes still gets."""
    request = build_evaluate_request(parsed, sat_id="SWARM B", v_remaining_m_s=25.0)
    request["conjunction"].pop("secondary_radius_m")
    return evaluate_conjunction_v25(request)


def _cap(radius_m: float) -> SatelliteCapability:
    return SatelliteCapability(sat_id="HBR-TEST", a_ref_km=6853.0, radius_m=radius_m)


# ---------------------------------------------------------------------------
# Acceptance 1: a known secondary is scored on the real combined radius
# ---------------------------------------------------------------------------

class TestAKnownSecondaryUsesItsRealRadius:
    def test_the_fixture_really_does_state_both_radii(self, parsed):
        """Otherwise everything below is testing a default."""
        assert parsed.primary.radius_m == 3.1
        assert parsed.secondary.radius_m == 0.5
        assert parsed.combined_hbr_m == pytest.approx(3.6)

    def test_the_secondary_radius_reaches_the_scorer(self, parsed):
        state = parsed.to_conjunction_state()
        assert state["secondary_radius_m"] == 0.5

    def test_pc_is_computed_on_the_real_combined_radius(self, scored):
        assert scored.hbr_m == pytest.approx(3.6)

    def test_the_source_says_where_the_radius_came_from(self, scored):
        assert scored.hbr_source == "cdm_object_radii"
        assert scored.hbr_source != "screening_convention"

    def test_the_real_radius_is_well_under_the_floor(self, scored):
        """The floor is not a lower bound on a known pair. If it were applied
        here it would be the 17x inflation this ticket removes."""
        assert scored.hbr_m < _FLOOR

    def test_we_still_compute_our_own_pc(self, scored):
        """Only the radius input moved. The CDM's Pc is still not our Pc."""
        assert scored.pc_source == "computed"


# ---------------------------------------------------------------------------
# Acceptance 2: an unknown secondary still gets the floor, and says so
# ---------------------------------------------------------------------------

class TestAnUnknownSecondaryFallsBackToTheFloor:
    def test_a_conjunction_that_states_no_radius_uses_the_floor(
            self, scored_without_radius):
        assert scored_without_radius.hbr_m == pytest.approx(_FLOOR)
        assert scored_without_radius.hbr_source == "screening_convention"

    def test_the_convention_falls_back_when_the_secondary_is_absent(self):
        assert conventions.combined_hbr_m(0.6, None) == (_FLOOR, "screening_convention")

    def test_a_zero_secondary_radius_is_absence_not_a_point_object(self):
        """The trap. A CDM that omits a radius reports 0.0, and taking that
        literally would give a 0.6 m hard-body radius -- or 0.0 if the primary
        is missing too -- driving Pc down and suppressing maneuvers."""
        assert conventions.combined_hbr_m(0.6, 0.0) == (_FLOOR, "screening_convention")

    def test_a_zero_primary_radius_also_falls_back(self):
        """Both radii have to be real for the sum to be real."""
        assert conventions.combined_hbr_m(0.0, 0.5) == (_FLOOR, "screening_convention")

    def test_a_negative_radius_falls_back_rather_than_shrinking_the_pair(self):
        assert conventions.combined_hbr_m(0.6, -1.0) == (_FLOOR, "screening_convention")

    def test_resolve_hard_body_radius_without_a_secondary_is_unchanged(self):
        """The signature gained an optional argument; the old call is identical."""
        assert resolve_hard_body_radius(_cap(0.60)) == (_FLOOR, "screening_convention")

    def test_a_physically_large_primary_still_beats_the_floor(self):
        """Pre-existing behaviour, preserved: a 14 m spacecraft plus an
        upper-stage-scale secondary is larger than the convention."""
        hbr, source = conventions.combined_hbr_m(14.0)
        assert hbr == pytest.approx(16.0)
        assert source == "physical_sum"


# ---------------------------------------------------------------------------
# Acceptance 3: agreement with LeoLabs' own Pc
# ---------------------------------------------------------------------------

class TestAgreementWithTheCdmCrossCheck:
    """The validation this change buys, and the reason the cross-check exists.

    LeoLabs compute their published Pc against their own object radii. Scoring
    the same geometry against the same radii should land on the same number; the
    15 m floor is what used to put us an order of magnitude away from it.
    """

    def test_our_pc_now_agrees_with_the_published_pc(self, scored, parsed):
        published = parsed.cdm_collision_probability
        assert published is not None
        assert scored.pc_pre == pytest.approx(published, rel=0.05)

    def test_the_floor_put_us_far_outside_that_band(
            self, scored_without_radius, parsed):
        """Non-vacuous: the agreement above is a real result, not a loose
        tolerance that anything would pass."""
        published = parsed.cdm_collision_probability
        assert scored_without_radius.pc_pre == pytest.approx(published, rel=100.0)
        assert scored_without_radius.pc_pre > published * 10.0

    def test_the_inflation_is_the_size_the_finding_described(
            self, scored, scored_without_radius):
        """SCRUM-394 put it at roughly 17x to 56x."""
        ratio = scored_without_radius.pc_pre / scored.pc_pre
        assert 17.0 <= ratio <= 56.0


# ---------------------------------------------------------------------------
# Acceptance 5: a previously-RED event falls below the threshold
# ---------------------------------------------------------------------------

class TestAPreviouslyRedEventIsNowBelowThreshold:
    """The operational consequence: the inflated flags deflate.

    Built as a controlled geometry rather than from the fixture, because the
    fixture's event is far below threshold either way and would not show the
    crossing. The 1e-4 threshold itself is untouched; only the radius moved.
    """

    @staticmethod
    def _score(secondary_radius_m):
        from common.operator_policy import OperatorPolicy
        import numpy as np
        from common.maneuver_scorer import score_maneuver_candidates
        from common.satellite_capability import LifetimeProfile

        cap = SatelliteCapability(
            sat_id="RED", a_ref_km=6853.0, radius_m=3.1,
            lifetime=LifetimeProfile(
                mass_kg=100.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
                mission_lifetime_days_remaining=365.0),
        )
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0",
                                decision_mode="flight_rule_1e4")
        return score_maneuver_candidates(
            "RED-489",
            np.array([6853.0, 0.0, 0.0]), np.array([0.0, 7.62, 0.0]),
            # 140 m miss against a 50 m 1-sigma: chosen so the two radii land
            # on opposite sides of 1e-4, which is the crossing under test.
            np.array([0.14, 0.0, 0.0]), np.eye(3) * 0.0025,
            "2026-04-14T08:00:00Z", "2026-04-14T12:00:00Z",
            cap, policy, miss_distance_km=0.14,
            v_rel_km_s=np.array([0.0, 10.0, 0.0]),
            secondary_radius_m=secondary_radius_m,
        )

    def test_the_floor_put_this_event_above_the_threshold(self):
        scored = self._score(None)
        assert scored.hbr_source == "screening_convention"
        assert scored.pc_pre > 1.0e-4

    def test_the_real_radius_puts_it_below(self):
        scored = self._score(0.5)
        assert scored.hbr_source == "cdm_object_radii"
        assert scored.pc_pre < 1.0e-4

    def test_the_crossing_is_the_radius_and_nothing_else(self):
        """Same geometry, same covariance, same threshold -- one input moved."""
        inflated = self._score(None)
        real = self._score(0.5)
        assert inflated.hbr_m == pytest.approx(_FLOOR)
        assert real.hbr_m == pytest.approx(3.6)
        assert inflated.pc_pre / real.pc_pre == pytest.approx(
            (inflated.hbr_m / real.hbr_m) ** 2, rel=0.15)

    def test_the_threshold_itself_did_not_move(self):
        """The crossing above is the radius changing, not the gate."""
        from common.operator_policy import OperatorPolicy
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0")
        assert policy.pc_maneuver_threshold == 1.0e-4
        assert policy.pc_monitor_threshold == 1.0e-5

    def test_the_maneuver_gate_still_fires_on_the_inflated_pc(self):
        """is_maneuver_required is untouched: handed the old Pc it still says
        yes. What changed is the Pc it is handed."""
        from common.operator_policy import OperatorPolicy
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0")
        inflated = self._score(None).pc_pre
        real = self._score(0.5).pc_pre
        assert policy.is_maneuver_required(inflated, 2.0) is True
        assert policy.is_maneuver_required(real, 2.0) is False


# ---------------------------------------------------------------------------
# Acceptance 4: nothing else moved
# ---------------------------------------------------------------------------

class TestNothingElseMoved:
    def test_the_screening_floor_value_is_unchanged(self):
        """The value was never the problem; its scope was."""
        assert conventions.DEFAULT_COMBINED_HBR_M == 15.0

    def test_the_other_conventions_are_unchanged(self):
        assert conventions.DEFAULT_PRIMARY_RADIUS_M == 0.60
        assert conventions.DEFAULT_SECONDARY_RADIUS_M == 2.0

    def test_sources_that_state_no_radius_keep_the_floor(self):
        """Ingest and UDL build their own conjunction states and carry no
        secondary radius, so they are unaffected by this change."""
        from common.udl_client import _parse_conjunction  # noqa: F401
        assert conventions.combined_hbr_m(0.60)[1] == "screening_convention"
