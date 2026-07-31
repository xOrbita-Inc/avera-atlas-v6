"""
Tests for APS Planner decision model v2.4.

Four synthetic conjunction events covering:
  RED-001        High risk — strong maneuver recommended
  AMBER-001      Moderate risk — maneuver with fuel/risk tradeoff
  GREEN-001      Low risk — no-burn baseline wins (negative utility for all burns)
  EFFICIENCY-001 Favorable geometry — small burn, large safety gain
"""

import json
import sys
from pathlib import Path

import pytest

# Allow imports from the planner package root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avoid.decision_model import evaluate_conjunction

EVENTS_PATH = Path(__file__).resolve().parents[1] / "tests" / "synthetic_conjunction_demo_events.json"


@pytest.fixture(scope="module")
def events():
    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {ev["conjunction_id"]: ev for ev in data}


def _eval(events, cid):
    return evaluate_conjunction(events[cid])


# ---------------------------------------------------------------------------
# RED-001: high risk, strong maneuver
# ---------------------------------------------------------------------------

class TestRed001:
    def test_direction(self, events):
        """SCRUM-386: prograde, not radial.

        This expected radial because cw_phi_rv returned the Phi_rr block,
        whose largest entry sits in the radial row (4 - 3cos(nt), about 7 at
        this geometry), so radial produced the largest displacement. The true
        Phi_rv is dominated by the along-track secular term, (4sin(nt) -
        3nt)/n, which grows with time and is why operators fly along-track
        avoidance burns. Corrected, prograde beats radial here by roughly
        three orders of magnitude in confidence gain.
        """
        result = _eval(events, "RED-001")
        assert result["recommendation"]["direction"] == "prograde"

    def test_dv_magnitude(self, events):
        result = _eval(events, "RED-001")
        assert abs(result["recommendation"]["dv_magnitude_m_s"] - 1.0) < 0.001

    def test_utility_positive(self, events):
        result = _eval(events, "RED-001")
        assert result["recommendation"]["utility"] > 0

    def test_response_structure(self, events):
        result = _eval(events, "RED-001")
        assert "recommendation" in result
        assert "metrics" in result


# ---------------------------------------------------------------------------
# AMBER-001: moderate risk, balanced tradeoff
# ---------------------------------------------------------------------------

class TestAmber001:
    """SCRUM-387 inverted these three.

    AMBER-001's scenario string has always said monitor, and its Pc of 3e-05
    sits between pc_monitor_threshold and pc_maneuver_threshold, which is the
    definition of watch-and-do-nothing. The tests asserted a full 1.0 m/s
    prograde burn anyway.

    That was not a mistake anyone made in isolation. Under the old utility every
    event maneuvered to the delta-v ceiling regardless of risk, because the
    benefit term grew quadratically in delta-v against a linear cost, so the
    assertions were written to match what the code did. The ticket's own table
    recorded the disagreement between the events' stated risk categories and the
    Pc policy before any of this was fixed.

    Now that the utility trades in probability, an event below the maneuver
    threshold cannot justify spending the budget, and AMBER-001 does what its
    own scenario string says.
    """

    def test_no_burn_because_the_pc_is_below_the_maneuver_threshold(self, events):
        result = _eval(events, "AMBER-001")
        assert result["recommendation"]["direction"] == "no-burn"

    def test_dv_magnitude_is_zero(self, events):
        result = _eval(events, "AMBER-001")
        assert result["recommendation"]["dv_magnitude_m_s"] == 0.0

    def test_utility_is_zero(self, events):
        """The no-burn baseline. A positive utility here would mean some burn
        was worth flying for an event the policy says only needs watching."""
        result = _eval(events, "AMBER-001")
        assert result["recommendation"]["utility"] == 0.0

    def test_the_pc_sits_between_the_two_thresholds(self, events):
        """Guards the premise. If the regenerated geometry ever drifts out of
        the monitor band, the three assertions above stop describing AMBER and
        start describing whatever it became."""
        assert 1.0e-5 <= events["AMBER-001"]["Pc"] < 1.0e-4

    def test_response_structure(self, events):
        result = _eval(events, "AMBER-001")
        assert "recommendation" in result
        assert "metrics" in result


# ---------------------------------------------------------------------------
# GREEN-001: low risk, no-burn wins
# ---------------------------------------------------------------------------

# SCRUM-387 removed the GREEN-001 xfail marks.
#
# They were added by SCRUM-386, which corrected cw_phi_rv and in doing so
# revealed that the scoring weights had been calibrated against a delta_r scale
# that was wrong by orders of magnitude. With the correct scale the confidence
# gain ran into the hundreds, the cost terms stopped biting, and no-burn never
# won for any event. GREEN-001 is the designed no-burn case, so it was the
# natural acceptance test for the recalibration, and the marks were left in
# place deliberately so the gap showed on every run rather than disappearing.
#
# The recalibration turned out not to be a recalibration. No weights could fix
# a benefit term that was unbounded and quadratic in delta-v against a linear
# cost. SCRUM-387 replaced the functional form instead, and GREEN-001 now
# recommends no-burn because a Pc of 4e-06 cannot justify spending fuel.
        assert result["recommendation"]["direction"] == "no-burn"

    def test_dv_magnitude(self, events):
        result = _eval(events, "GREEN-001")
        assert result["recommendation"]["dv_magnitude_m_s"] == 0.0

    def test_utility_zero(self, events):
        result = _eval(events, "GREEN-001")
        assert result["recommendation"]["utility"] == 0.0

    def test_response_structure(self, events):
        result = _eval(events, "GREEN-001")
        assert "recommendation" in result
        assert "metrics" in result


# ---------------------------------------------------------------------------
# EFFICIENCY-001: favorable geometry, small burn, large gain
# ---------------------------------------------------------------------------

class TestEfficiency001:
    def test_direction(self, events):
        result = _eval(events, "EFFICIENCY-001")
        assert result["recommendation"]["direction"] == "prograde"

    def test_dv_magnitude(self, events):
        result = _eval(events, "EFFICIENCY-001")
        assert abs(result["recommendation"]["dv_magnitude_m_s"] - 0.6) < 0.001

    def test_utility_positive(self, events):
        result = _eval(events, "EFFICIENCY-001")
        assert result["recommendation"]["utility"] > 0

    def test_response_structure(self, events):
        result = _eval(events, "EFFICIENCY-001")
        assert "recommendation" in result
        assert "metrics" in result
