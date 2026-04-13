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
        result = _eval(events, "RED-001")
        assert result["recommendation"]["direction"] == "radial"

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
    def test_direction(self, events):
        result = _eval(events, "AMBER-001")
        assert result["recommendation"]["direction"] == "radial"

    def test_dv_magnitude(self, events):
        result = _eval(events, "AMBER-001")
        assert abs(result["recommendation"]["dv_magnitude_m_s"] - 1.0) < 0.001

    def test_utility_positive(self, events):
        result = _eval(events, "AMBER-001")
        assert result["recommendation"]["utility"] > 0

    def test_response_structure(self, events):
        result = _eval(events, "AMBER-001")
        assert "recommendation" in result
        assert "metrics" in result


# ---------------------------------------------------------------------------
# GREEN-001: low risk, no-burn wins
# ---------------------------------------------------------------------------

class TestGreen001:
    def test_direction(self, events):
        result = _eval(events, "GREEN-001")
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
