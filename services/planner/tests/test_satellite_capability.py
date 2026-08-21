"""Regression coverage for SatelliteCapability request adaptation."""

import math

import pytest

from common.maneuver_scorer import _drag_gate_altitude_from_sma_km
from common.satellite_capability import SatelliteCapability


MU_EARTH_KM3_S2 = 398600.4418
DEMO_A_KM = 6871.0
DEMO_V_KM_S = math.sqrt(MU_EARTH_KM3_S2 / DEMO_A_KM)


def _demo_state_request():
    return {
        "sat_id": "DEMO",
        "r_sat_km": [DEMO_A_KM, 0.0, 0.0],
        "v_sat_km_s": [0.0, DEMO_V_KM_S, 0.0],
    }


def test_from_request_derives_a_ref_from_state_when_omitted():
    cap = SatelliteCapability.from_request(_demo_state_request())

    assert cap.a_ref_km == pytest.approx(DEMO_A_KM, abs=1e-9)
    assert cap.a_ref_km != pytest.approx(7000.0)


def test_from_request_preserves_explicit_a_ref():
    request = _demo_state_request()
    request["a_ref_km"] = 6900.0

    cap = SatelliteCapability.from_request(request)

    assert cap.a_ref_km == 6900.0


def test_drag_gate_altitude_uses_documented_spherical_radius_approximation():
    assert _drag_gate_altitude_from_sma_km(6871.0) == pytest.approx(500.0)
