"""
services/planner/tests/_true_two_body.py

SCRUM-409: independent two-body reference for tests whose expected values
used to be computed with the single-rotation CW formula (the same formula
the production bug used), which meant the tests were grading the code
against an oracle carrying the same defect.

Per review: do not rebaseline against the new two-epoch CW formula either,
since that just rebuilds a different self-referential oracle that would
not have caught a bug in the new formula. Anchor to independent truth
instead, a short numerical propagation under plain two-body gravity, nothing
CW, nothing RTN, nothing from aps_math.frames or decision_model.py.

This is deliberately NOT imported from services/tracker/iod.py's
kepler_propagate or any other shipped propagator, for the same reason: a
reference has to be independent of the thing it is checking, or agreement
proves nothing.

Not a pytest test module itself (no test_ prefix), a plain helper other
test files import.
"""

from __future__ import annotations

import numpy as np

MU_EARTH_KM3_S2 = 398600.4418


def true_two_body_propagate(
    r0_km: np.ndarray,
    v0_km_s: np.ndarray,
    dt_s: float,
    mu: float = MU_EARTH_KM3_S2,
    n_steps: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed-step RK4 propagation under plain two-body gravity. No CW, no
    linearization, no reference orbit assumption -- the equation of motion
    is exactly r'' = -mu r / |r|^3, integrated directly.

    n_steps=4000 over the lead times these tests use (up to tens of hours)
    keeps RK4's own truncation error many orders of magnitude below the
    CW linearization error this helper exists to measure against, so a
    disagreement with a CW-based prediction can be attributed to CW, not
    to this propagator. See test_true_two_body_matches_a_closed_form_orbit
    for a check that this claim holds.
    """
    def deriv(state: np.ndarray) -> np.ndarray:
        r = state[:3]
        v = state[3:]
        r_norm = np.linalg.norm(r)
        a = -mu * r / r_norm ** 3
        return np.concatenate([v, a])

    state = np.concatenate([np.asarray(r0_km, dtype=float), np.asarray(v0_km_s, dtype=float)])
    h = dt_s / n_steps
    for _ in range(n_steps):
        k1 = deriv(state)
        k2 = deriv(state + h / 2 * k1)
        k3 = deriv(state + h / 2 * k2)
        k4 = deriv(state + h * k3)
        state = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return state[:3], state[3:]


def true_burn_displacement_km(
    r_sat_km: np.ndarray,
    v_sat_km_s: np.ndarray,
    dv_eci_km_s: np.ndarray,
    dt_to_ca_s: float,
    mu: float = MU_EARTH_KM3_S2,
) -> np.ndarray:
    """
    The true ECI displacement, at dt_to_ca_s after the burn, that an
    impulsive delta-v dv_eci_km_s applied at (r_sat_km, v_sat_km_s)
    actually produces, by propagating the burned and unburned states and
    differencing. This is what CW's Phi_rv approximates; comparing a
    CW-based prediction against this is comparing it against the thing
    it is an approximation OF, not against another approximation.
    """
    r_ref, v_ref = true_two_body_propagate(r_sat_km, v_sat_km_s, dt_to_ca_s, mu=mu)
    r_pert, _v_pert = true_two_body_propagate(
        r_sat_km, np.asarray(v_sat_km_s, dtype=float) + np.asarray(dv_eci_km_s, dtype=float),
        dt_to_ca_s, mu=mu,
    )
    return r_pert - r_ref
