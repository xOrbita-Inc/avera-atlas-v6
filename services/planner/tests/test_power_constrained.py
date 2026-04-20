"""
Tests for the power_constrained thrust model in decision_model.py.

Covers:
  1. _power_constrained_dv_m_s -- unit tests for the thrust model function
  2. evaluate_conjunction -- integration tests via a synthetic power-constrained event
  3. Fallback behaviour -- legacy callers without propulsion fields
  4. Validation -- invalid parameter handling

Physics reference:
    F     = 2 * eta * P_W / (Isp * g0)   [N]
    dv    = F * t_burn / mass_kg           [m/s]
    g0    = 9.80665 m/s^2

Run:
    pytest services/planner/tests/test_power_constrained.py -v
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avoid.decision_model import evaluate_conjunction, _power_constrained_dv_m_s

_G0 = 9.80665  # m/s^2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_dv(P_W, isp_s, eta, mass_kg, t_burn_s, limit_m_s):
    """Expected dv from thrust model, capped at limit."""
    F = 2.0 * eta * P_W / (isp_s * _G0)
    return min((F * t_burn_s) / mass_kg, limit_m_s)


def _make_event(power_constrained, propulsion_params=None, dv_limit=1.0):
    """Build a synthetic conjunction event with power_constrained flag.

    Propulsion parameters (power_available_w, thruster_efficiency,
    isp_s, burn_window_s) are placed in satellite.propulsion, consistent
    with PropulsionProfile in satellite_capability.py.
    mass_kg is placed in satellite.lifetime.
    """
    prop = {}
    mass_kg = 100.0
    if propulsion_params:
        mass_kg = propulsion_params.pop("satellite_mass_kg", 100.0)
        prop.update(propulsion_params)

    return {
        "conjunction_id": "PWR-TEST",
        "satellite": {
            "sat_id": "SAT-PWR",
            "r_sat_km": [7000.0, 0.0, 0.0],
            "v_sat_km_s": [0.0, 7.546, 0.0],
            "t_burn_utc": "2026-04-14T08:00:00Z",
            "v_remaining_m_s": 50.0,
            "propulsion": prop,
            "lifetime": {"mass_kg": mass_kg, "v_remaining_m_s": 50.0},
        },
        "conjunction": {
            "obj_id": "OBJ-001",
            "t_ca_utc": "2026-04-14T12:00:00Z",    # 4 hours to TCA (14400s)
            "r_rel_km": [0.3, 0.0, 0.0],
            "p_rel_km2": [0.01,0,0, 0,0.01,0, 0,0,0.01],
        },
        "policy": {
            "lambda_v": 1.0,
            "lambda_L": 0.8,
            "dv_mag_limit_m_s": dv_limit,
            "a_ref_km": 7000.0,
            "hard_constraints": {
                "power_constrained": power_constrained,
                "attitude_restricted": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# 1. _power_constrained_dv_m_s -- unit tests
# ---------------------------------------------------------------------------

class TestThrustModelFunction:
    def test_physics_limited(self):
        """When achievable dv < limit, physics sets the effective dv."""
        P, Isp, eta, m, t = 200.0, 1600.0, 0.55, 300.0, 600.0
        expected = _expected_dv(P, Isp, eta, m, t, 2.0)
        result = _power_constrained_dv_m_s(
            {"power_available_w": P, "isp_s": Isp, "thruster_efficiency": eta,
             "burn_window_s": t},
            mass_kg=m, dv_limit_m_s=2.0, dt_to_ca_s=t,
        )
        assert result == pytest.approx(expected, rel=1e-9)
        assert result < 2.0  # physics-limited, not policy-limited

    def test_policy_limited(self):
        """When achievable dv > limit, the policy ceiling is applied."""
        P, Isp, eta, m, t = 5000.0, 300.0, 0.90, 100.0, 600.0
        limit = 0.5
        result = _power_constrained_dv_m_s(
            {"power_available_w": P, "isp_s": Isp, "thruster_efficiency": eta,
             "burn_window_s": t},
            mass_kg=m, dv_limit_m_s=limit, dt_to_ca_s=t,
        )
        assert result == pytest.approx(limit, rel=1e-9)

    def test_result_never_exceeds_limit(self):
        """Effective dv must always be <= dv_limit regardless of propulsion."""
        for P in [100.0, 1000.0, 10000.0]:
            result = _power_constrained_dv_m_s(
                {"power_available_w": P, "isp_s": 220.0,
                 "thruster_efficiency": 0.8, "burn_window_s": 600.0},
                mass_kg=100.0, dv_limit_m_s=1.0, dt_to_ca_s=600.0,
            )
            assert result <= 1.0 + 1e-9, f"dv {result} exceeded limit for P={P}"

    def test_result_positive(self):
        """Effective dv must be positive."""
        result = _power_constrained_dv_m_s(
            {"power_available_w": 200.0, "isp_s": 1600.0,
             "thruster_efficiency": 0.55},
            mass_kg=300.0, dv_limit_m_s=2.0, dt_to_ca_s=600.0,
        )
        assert result > 0.0

    def test_burn_window_from_dt_to_ca_when_not_supplied(self):
        """burn_window_s defaults to dt_to_ca_s when absent."""
        P, Isp, eta, m = 200.0, 1600.0, 0.55, 300.0
        dt = 600.0
        result_explicit = _power_constrained_dv_m_s(
            {"power_available_w": P, "isp_s": Isp, "thruster_efficiency": eta,
             "burn_window_s": dt},
            mass_kg=m, dv_limit_m_s=2.0, dt_to_ca_s=dt,
        )
        result_default = _power_constrained_dv_m_s(
            {"power_available_w": P, "isp_s": Isp, "thruster_efficiency": eta},
            mass_kg=m, dv_limit_m_s=2.0, dt_to_ca_s=dt,
        )
        assert result_explicit == pytest.approx(result_default, rel=1e-9)

    def test_longer_burn_window_gives_more_dv(self):
        """More burn time = more achievable dv (up to the limit)."""
        params = {"power_available_w": 200.0, "isp_s": 1600.0,
                  "thruster_efficiency": 0.55}
        dv_short = _power_constrained_dv_m_s(params, mass_kg=300.0, dv_limit_m_s=2.0, dt_to_ca_s=300.0)
        dv_long  = _power_constrained_dv_m_s(params, mass_kg=300.0, dv_limit_m_s=2.0, dt_to_ca_s=1200.0)
        assert dv_long > dv_short

    def test_higher_power_gives_more_dv(self):
        """Higher available power = higher thrust = more achievable dv."""
        base = {"isp_s": 1600.0, "thruster_efficiency": 0.55, "burn_window_s": 600.0}
        dv_low  = _power_constrained_dv_m_s({**base, "power_available_w": 100.0}, mass_kg=300.0, dv_limit_m_s=2.0, dt_to_ca_s=600.0)
        dv_high = _power_constrained_dv_m_s({**base, "power_available_w": 500.0}, mass_kg=300.0, dv_limit_m_s=2.0, dt_to_ca_s=600.0)
        assert dv_high > dv_low

    def test_physics_formula_exact(self):
        """Verify exact thrust model formula: F = 2*eta*P/(Isp*g0), dv = F*t/m."""
        P, Isp, eta, m, t = 300.0, 220.0, 0.85, 100.0, 600.0
        F_expected = 2.0 * eta * P / (Isp * _G0)
        dv_expected = F_expected * t / m
        result = _power_constrained_dv_m_s(
            {"power_available_w": P, "isp_s": Isp, "thruster_efficiency": eta,
             "burn_window_s": t},
            mass_kg=m, dv_limit_m_s=100.0, dt_to_ca_s=t,
        )
        assert result == pytest.approx(dv_expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. Fallback behaviour -- legacy callers
# ---------------------------------------------------------------------------

class TestFallbackBehaviour:
    def test_fallback_when_no_propulsion_params(self):
        """Missing propulsion params -> 0.5 * dv_limit (legacy behavior)."""
        result = _power_constrained_dv_m_s(
            {}, mass_kg=100.0, dv_limit_m_s=2.0, dt_to_ca_s=600.0
        )
        assert result == pytest.approx(1.0, rel=1e-9)

    def test_fallback_when_partial_params(self):
        """Partial propulsion params (missing efficiency) -> fallback."""
        result = _power_constrained_dv_m_s(
            {"power_available_w": 200.0, "isp_s": 1600.0},
            mass_kg=300.0, dv_limit_m_s=2.0, dt_to_ca_s=600.0,
        )
        assert result == pytest.approx(1.0, rel=1e-9)

    def test_fallback_scales_with_limit(self):
        """Fallback is always 0.5 * dv_limit, regardless of limit value."""
        for limit in [0.5, 1.0, 2.0, 5.0]:
            result = _power_constrained_dv_m_s({}, mass_kg=100.0, dv_limit_m_s=limit, dt_to_ca_s=600.0)
            assert result == pytest.approx(0.5 * limit, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Validation -- invalid parameter handling
# ---------------------------------------------------------------------------

class TestInvalidParameters:
    def _params(self, **overrides):
        base = {"power_available_w": 200.0, "isp_s": 1600.0,
                "thruster_efficiency": 0.55, "burn_window_s": 600.0}
        base.update(overrides)
        return base

    def test_negative_power_falls_back(self):
        result = _power_constrained_dv_m_s(
            self._params(power_available_w=-10.0), mass_kg=300.0, dv_limit_m_s=2.0, dt_to_ca_s=600.0
        )
        assert result == pytest.approx(1.0)

    def test_zero_isp_falls_back(self):
        result = _power_constrained_dv_m_s(
            self._params(isp_s=0.0), mass_kg=300.0, dv_limit_m_s=2.0, dt_to_ca_s=600.0
        )
        assert result == pytest.approx(1.0)

    def test_efficiency_above_one_falls_back(self):
        result = _power_constrained_dv_m_s(
            self._params(thruster_efficiency=1.5), mass_kg=300.0, dv_limit_m_s=2.0, dt_to_ca_s=600.0
        )
        assert result == pytest.approx(1.0)

    def test_zero_mass_falls_back(self):
        result = _power_constrained_dv_m_s(
            self._params(), mass_kg=0.0, dv_limit_m_s=2.0, dt_to_ca_s=600.0
        )
        assert result == pytest.approx(1.0)

    def test_zero_burn_window_falls_back(self):
        result = _power_constrained_dv_m_s(
            self._params(burn_window_s=0.0), mass_kg=300.0, dv_limit_m_s=2.0, dt_to_ca_s=600.0
        )
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. evaluate_conjunction -- integration tests
# ---------------------------------------------------------------------------

class TestPowerConstrainedIntegration:
    def test_power_constrained_uses_thrust_model_not_hardcode(self):
        """Effective dv must match the thrust model, not 0.5 * limit."""
        P, Isp, eta, m, t_burn = 200.0, 1600.0, 0.55, 300.0, 14400.0
        propulsion = {
            "power_available_w": P, "isp_s": Isp,
            "thruster_efficiency": eta, "satellite_mass_kg": m,
            "burn_window_s": t_burn,
        }
        event = _make_event(power_constrained=True,
                            propulsion_params=propulsion, dv_limit=2.0)
        result = evaluate_conjunction(event)

        expected_dv = _expected_dv(P, Isp, eta, m, t_burn, 2.0)
        actual_dv = result["metrics"]["fuel_cost_m_s"]

        # Must use physics model, not 0.5 * 2.0 = 1.0
        assert actual_dv == pytest.approx(expected_dv, rel=1e-6)
        assert abs(actual_dv - 1.0) > 1e-4, (
            "dv should not equal 0.5 * limit; thrust model should govern"
        )

    def test_power_constrained_fallback_matches_legacy(self):
        """Without propulsion params, behaviour matches old 0.5 * limit."""
        event = _make_event(power_constrained=True,
                            propulsion_params=None, dv_limit=2.0)
        result = evaluate_conjunction(event)
        assert result["metrics"]["fuel_cost_m_s"] == pytest.approx(1.0, rel=1e-6)

    def test_power_constrained_false_unaffected(self):
        """power_constrained=False must use the full dv_limit unchanged."""
        event = _make_event(power_constrained=False, dv_limit=2.0)
        result = evaluate_conjunction(event)
        assert result["metrics"]["fuel_cost_m_s"] == pytest.approx(2.0, rel=1e-6)

    def test_response_structure_preserved(self):
        """evaluate_conjunction response schema must be unchanged."""
        event = _make_event(power_constrained=True, dv_limit=2.0)
        result = evaluate_conjunction(event)
        assert "recommendation" in result
        assert "metrics" in result
        assert "direction" in result["recommendation"]
        assert "dv_magnitude_m_s" in result["recommendation"]
        assert "utility" in result["recommendation"]

    def test_dv_never_exceeds_limit_when_power_constrained(self):
        """Thrust model must never allow dv above the policy limit."""
        for P in [100.0, 500.0, 5000.0]:
            props = {"power_available_w": P, "isp_s": 220.0,
                     "thruster_efficiency": 0.85, "satellite_mass_kg": 100.0,
                     "burn_window_s": 14400.0}
            event = _make_event(True, props, dv_limit=1.0)
            result = evaluate_conjunction(event)
            assert result["metrics"]["fuel_cost_m_s"] <= 1.0 + 1e-9

    def test_policy_limited_scenario(self):
        """High-power thruster capped at policy limit."""
        props = {"power_available_w": 5000.0, "isp_s": 300.0,
                 "thruster_efficiency": 0.90, "satellite_mass_kg": 100.0,
                 "burn_window_s": 14400.0}
        event = _make_event(True, props, dv_limit=0.5)
        result = evaluate_conjunction(event)
        assert result["metrics"]["fuel_cost_m_s"] == pytest.approx(0.5, rel=1e-6)