"""SCRUM-399 regression tests for two-body Kepler propagation.

The numerical reference below is deliberately independent of the production
universal-variable solver. It advances classical elliptic elements by solving
M = E - e sin(E) and reconstructs the Cartesian state in the orbital plane.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_PROP_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("propagator_main_scrum399", _PROP_ROOT / "main.py")
propagator_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(propagator_main)

MU = propagator_main.MU_EARTH
kepler_propagate = propagator_main.kepler_propagate


def _solve_elliptic_kepler(mean_anomaly: float, eccentricity: float) -> float:
    """Solve M = E - e sin(E) independently of the production propagator."""
    E = mean_anomaly
    for _ in range(50):
        residual = E - eccentricity * np.sin(E) - mean_anomaly
        derivative = 1.0 - eccentricity * np.cos(E)
        step = residual / derivative
        E -= step
        if abs(step) < 1e-14:
            return float(E)
    raise RuntimeError("reference elliptic Kepler solve did not converge")


def _reference_state(a_km: float, eccentricity: float, nu0_rad: float, dt_s: float):
    """Propagate a planar elliptic orbit using classical Kepler elements."""
    e = eccentricity
    root = np.sqrt((1.0 - e) / (1.0 + e))
    E0 = 2.0 * np.arctan2(
        root * np.sin(nu0_rad / 2.0),
        np.cos(nu0_rad / 2.0),
    )
    M0 = E0 - e * np.sin(E0)
    mean_motion = np.sqrt(MU / a_km**3)
    E = _solve_elliptic_kepler(M0 + mean_motion * dt_s, e)

    cos_E = np.cos(E)
    sin_E = np.sin(E)
    beta = np.sqrt(1.0 - e * e)
    radius_km = a_km * (1.0 - e * cos_E)

    r = np.array([
        a_km * (cos_E - e),
        a_km * beta * sin_E,
        0.0,
    ])
    v = np.sqrt(MU * a_km) / radius_km * np.array([
        -sin_E,
        beta * cos_E,
        0.0,
    ])
    return r, v


@pytest.mark.parametrize("eccentricity", [0.0, 0.001, 0.01, 0.05, 0.2, 0.3])
def test_eccentricity_sweep_matches_independent_reference(eccentricity):
    rp_km = 7000.0
    a_km = rp_km / (1.0 - eccentricity)
    dt_s = 600.0

    r0, v0 = _reference_state(a_km, eccentricity, 0.0, 0.0)
    expected_r, expected_v = _reference_state(a_km, eccentricity, 0.0, dt_s)
    actual_r, actual_v = kepler_propagate(r0, v0, dt_s)

    assert actual_r == pytest.approx(expected_r, rel=1e-11, abs=1e-8)
    assert actual_v == pytest.approx(expected_v, rel=1e-11, abs=1e-10)

    # Regression guard for SCRUM-399: this must never silently become
    # straight-line propagation again.
    linear_r = r0 + v0 * dt_s
    assert np.linalg.norm(actual_r - linear_r) > 1.0


def test_non_apsis_state_matches_independent_reference():
    """Exercise nonzero radial velocity, which the old equations mishandled."""
    a_km = 9000.0
    eccentricity = 0.2
    nu0_rad = np.deg2rad(57.0)
    dt_s = 900.0

    r0, v0 = _reference_state(a_km, eccentricity, nu0_rad, 0.0)
    assert abs(np.dot(r0, v0)) > 1.0  # definitely not at an apsis

    expected_r, expected_v = _reference_state(a_km, eccentricity, nu0_rad, dt_s)
    actual_r, actual_v = kepler_propagate(r0, v0, dt_s)

    assert actual_r == pytest.approx(expected_r, rel=1e-11, abs=1e-8)
    assert actual_v == pytest.approx(expected_v, rel=1e-11, abs=1e-10)


def test_circular_case_preserves_pre_scrum399_result():
    r0 = np.array([7000.0, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(MU / 7000.0), 0.0])

    r, v = kepler_propagate(r0, v0, 600.0)

    expected_r = np.array([5586.094941801408, 4218.476419417409, 0.0])
    expected_v = np.array([-4.547549694855, 6.021852873491, 0.0])
    assert r == pytest.approx(expected_r, abs=1e-9)
    assert v == pytest.approx(expected_v, abs=1e-9)


def test_near_parabolic_state_is_explicitly_unsupported():
    r0 = np.array([7000.0, 0.0, 0.0])
    escape_speed = np.sqrt(2.0 * MU / 7000.0)
    v0 = np.array([0.0, escape_speed, 0.0])

    with pytest.raises(ValueError, match="near-parabolic"):
        kepler_propagate(r0, v0, 600.0)


def test_hyperbolic_state_is_explicitly_unsupported():
    r0 = np.array([7000.0, 0.0, 0.0])
    escape_speed = np.sqrt(2.0 * MU / 7000.0)
    v0 = np.array([0.0, 1.05 * escape_speed, 0.0])

    with pytest.raises(ValueError, match="hyperbolic"):
        kepler_propagate(r0, v0, 600.0)


def test_backward_propagation_recovers_initial_state():
    a_km = 9000.0
    eccentricity = 0.2
    nu0_rad = np.deg2rad(57.0)
    dt_s = 900.0

    r0, v0 = _reference_state(a_km, eccentricity, nu0_rad, 0.0)
    r1, v1 = kepler_propagate(r0, v0, dt_s)
    r_back, v_back = kepler_propagate(r1, v1, -dt_s)

    assert r_back == pytest.approx(r0, rel=1e-11, abs=1e-8)
    assert v_back == pytest.approx(v0, rel=1e-11, abs=1e-10)
