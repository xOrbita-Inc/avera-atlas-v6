"""
libs/aps_math/tests/test_observability.py

SCRUM-378: verify observation_jacobian against an independent finite-
difference reference, the same pattern test_cw_phi_rv.py uses for the CW
state transition matrix. The function's own docstring derives H
analytically; this file does not trust that derivation, it measures it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aps_math import frames
from aps_math.observability import measurement_noise_covariance, observation_jacobian


def _forward_model(r_target: np.ndarray, r_observer: np.ndarray, include_range: bool) -> np.ndarray:
    """RA/Dec[/range] from target and observer position, written out
    independently of observation_jacobian. Matches
    services/tracker/transform.py::eci_direction_to_ra_dec's convention
    (ra = atan2(y,x), dec = asin(z)) but is not imported from it, so a
    change to one does not silently validate against itself."""
    los = r_target - r_observer
    rng = np.linalg.norm(los)
    u = los / rng
    dec = math.asin(np.clip(u[2], -1.0, 1.0))
    ra = math.atan2(u[1], u[0])
    if include_range:
        return np.array([ra, dec, rng])
    return np.array([ra, dec])


def _finite_difference_jacobian(
    r_target: np.ndarray, r_observer: np.ndarray, include_range: bool, eps: float = 1e-4
) -> np.ndarray:
    """Central finite difference w.r.t. the three position components.
    Velocity columns are not measured this way: they are zero by
    construction (RA/Dec/range do not depend on velocity), asserted
    separately below rather than finite-differenced, since finite-
    differencing a genuinely-zero quantity mostly measures numerical
    noise, not the thing itself."""
    n_meas = 3 if include_range else 2
    H = np.zeros((n_meas, 6))
    for i in range(3):
        dr = np.zeros(3)
        dr[i] = eps
        plus = _forward_model(r_target + dr, r_observer, include_range)
        minus = _forward_model(r_target - dr, r_observer, include_range)
        diff = plus - minus
        # RA wraps at 0/2pi; unwrap so a perturbation near the wrap point
        # doesn't look like a ~2pi jump.
        if diff[0] > math.pi:
            diff[0] -= 2 * math.pi
        elif diff[0] < -math.pi:
            diff[0] += 2 * math.pi
        H[:, i] = diff / (2 * eps)
    return H


# Geometries spanning ordinary LEO conjunction cases plus a near-polar
# declination, which is the case that makes finite-difference verification
# itself numerically delicate (see module docstring in observability.py) --
# included deliberately, not avoided.
GEOMETRIES = [
    pytest.param(
        np.array([6928.0, 100.0, 200.0]), np.array([6900.0, 0.0, 0.0]), id="ordinary-1"
    ),
    pytest.param(
        np.array([7000.0, -300.0, 500.0]), np.array([6928.0, 50.0, -50.0]), id="ordinary-2"
    ),
    pytest.param(
        np.array([6800.0, 0.0, 6800.0]), np.array([6778.0, 20.0, 20.0]), id="near-polar-89.75deg"
    ),
]


class TestMatchesFiniteDifference:
    @pytest.mark.parametrize("r_target,r_observer", GEOMETRIES)
    @pytest.mark.parametrize("include_range", [False, True])
    def test_matches_across_geometries(self, r_target, r_observer, include_range):
        H_analytical = observation_jacobian(r_target, r_observer, include_range)
        H_fd = _finite_difference_jacobian(r_target, r_observer, include_range)
        assert np.allclose(H_analytical, H_fd, rtol=1e-5, atol=1e-6)


class TestVelocityColumnsAreZero:
    """Not approximately zero: exactly zero, by construction. RA, Dec, and
    range at one epoch are functions of position only."""

    @pytest.mark.parametrize("r_target,r_observer", GEOMETRIES)
    @pytest.mark.parametrize("include_range", [False, True])
    def test_velocity_columns_exactly_zero(self, r_target, r_observer, include_range):
        H = observation_jacobian(r_target, r_observer, include_range)
        assert np.array_equal(H[:, 3:6], np.zeros((H.shape[0], 3)))


class TestShape:
    def test_angles_only_is_2x6(self):
        H = observation_jacobian(
            np.array([6928.0, 0.0, 0.0]), np.array([6900.0, 0.0, 0.0]), include_range=False
        )
        assert H.shape == (2, 6)

    def test_angles_and_range_is_3x6(self):
        H = observation_jacobian(
            np.array([6928.0, 0.0, 0.0]), np.array([6900.0, 0.0, 0.0]), include_range=True
        )
        assert H.shape == (3, 6)


class TestGuards:
    def test_coincident_positions_raise(self):
        r = np.array([6928.0, 100.0, 0.0])
        with pytest.raises(ValueError, match="coincide"):
            observation_jacobian(r, r.copy(), include_range=False)

    def test_exact_pole_raises(self):
        r_target = np.array([0.0, 0.0, 1000.0])
        r_observer = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="pole"):
            observation_jacobian(r_target, r_observer, include_range=False)

    def test_near_polar_does_not_raise(self):
        """89.75 degrees declination is not a singularity, just a
        challenging geometry. It must compute, not be rejected."""
        H = observation_jacobian(
            np.array([6800.0, 0.0, 6800.0]), np.array([6778.0, 20.0, 20.0]), include_range=False
        )
        assert np.all(np.isfinite(H))


class TestMeasurementNoiseCovariance:
    """R must stay unit- and shape-consistent with observation_jacobian's
    output, since the Gramian combines them directly as H^T R^-1 H."""

    def test_angles_only_is_2x2(self):
        R = measurement_noise_covariance(1e-5, 1e-5)
        assert R.shape == (2, 2)

    def test_angles_and_range_is_3x3(self):
        R = measurement_noise_covariance(1e-5, 1e-5, range_sigma_km=0.05)
        assert R.shape == (3, 3)

    def test_shape_matches_observation_jacobian_rows(self):
        r_target = np.array([6928.0, 100.0, 200.0])
        r_observer = np.array([6900.0, 0.0, 0.0])

        H_angles = observation_jacobian(r_target, r_observer, include_range=False)
        R_angles = measurement_noise_covariance(1e-5, 1e-5)
        assert H_angles.shape[0] == R_angles.shape[0]

        H_range = observation_jacobian(r_target, r_observer, include_range=True)
        R_range = measurement_noise_covariance(1e-5, 1e-5, range_sigma_km=0.05)
        assert H_range.shape[0] == R_range.shape[0]

    def test_is_diagonal_with_variances(self):
        R = measurement_noise_covariance(2e-5, 3e-5, range_sigma_km=0.1)
        expected = np.diag([4e-10, 9e-10, 0.01])
        assert np.allclose(R, expected)
        # And genuinely diagonal, not just diag-equal by coincidence.
        off_diag = R - np.diag(np.diag(R))
        assert np.array_equal(off_diag, np.zeros((3, 3)))

    def test_invertible_and_feeds_the_gramian_term(self):
        """R must actually be usable as R^-1 inside H^T R^-1 H, the real
        quantity the Gramian sums over a tracking arc. This is not a
        shape check alone: it exercises the inversion and the resulting
        product's symmetry, which a singular or malformed R would break."""
        r_target = np.array([6928.0, 100.0, 200.0])
        r_observer = np.array([6900.0, 0.0, 0.0])
        H = observation_jacobian(r_target, r_observer, include_range=False)
        R = measurement_noise_covariance(1e-5, 1e-5)

        R_inv = np.linalg.inv(R)
        term = H.T @ R_inv @ H

        assert term.shape == (6, 6)
        assert np.allclose(term, term.T)


class TestMeasurementNoiseCovarianceGuards:
    def test_zero_ra_sigma_raises(self):
        with pytest.raises(ValueError, match="ra_sigma_rad"):
            measurement_noise_covariance(0.0, 1e-5)

    def test_negative_dec_sigma_raises(self):
        with pytest.raises(ValueError, match="dec_sigma_rad"):
            measurement_noise_covariance(1e-5, -1e-5)

    def test_nan_sigma_raises(self):
        with pytest.raises(ValueError):
            measurement_noise_covariance(float("nan"), 1e-5)

    def test_zero_range_sigma_raises(self):
        with pytest.raises(ValueError, match="range_sigma_km"):
            measurement_noise_covariance(1e-5, 1e-5, range_sigma_km=0.0)

    def test_range_sigma_none_is_valid_and_angles_only(self):
        """None is the documented way to request angles-only; must not
        raise."""
        R = measurement_noise_covariance(1e-5, 1e-5, range_sigma_km=None)
        assert R.shape == (2, 2)


class TestStateTransforms:
    """SCRUM-378: rtn_to_eci_state_transform / eci_to_rtn_state_transform,
    the full 6-state RTN<->ECI conversion the Gramian needs to combine
    cw_phi_full's RTN-frame output with observation_jacobian's ECI-frame
    input. Verified against an independent two-body propagator, not
    against either function under test, matching the convention already
    used for cw_phi_full and observation_jacobian."""

    @staticmethod
    def _true_two_body_propagate(r0, v0, dt_s, mu, n_steps=6000):
        """Independent RK4 two-body propagator, written out here rather
        than imported from anywhere in this repo, so agreement with it
        means something."""
        def deriv(state):
            r = state[:3]
            v = state[3:]
            r_norm = np.linalg.norm(r)
            a = -mu * r / r_norm ** 3
            return np.concatenate([v, a])

        state = np.concatenate([np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)])
        h = dt_s / n_steps
        for _ in range(n_steps):
            k1 = deriv(state)
            k2 = deriv(state + h / 2 * k1)
            k3 = deriv(state + h / 2 * k2)
            k4 = deriv(state + h * k3)
            state = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return state[:3], state[3:]

    @pytest.mark.parametrize(
        "a_km,inc_deg,dt_s",
        [
            (6928.0, 0.0, 1800.0),
            (6928.0, 53.0, 3600.0),
            (7000.0, 97.8, 900.0),
        ],
    )
    def test_full_round_trip_matches_true_two_body_motion(self, a_km, inc_deg, dt_s):
        """Build a state deviation in RTN at t0, propagate it through
        cw_phi_full, convert to ECI at TCA, and compare against a real
        perturbed-vs-unperturbed two-body propagation. This is the
        actual combination the Gramian needs, exercised end to end."""
        mu = frames.MU_EARTH
        inc = math.radians(inc_deg)
        r0 = np.array([a_km, 0.0, 0.0])
        n = math.sqrt(mu / a_km ** 3)
        v0 = n * a_km * np.array([0.0, math.cos(inc), math.sin(inc)])

        dr0_eci = np.array([0.05, 0.1, 0.02])
        dv0_eci = np.array([0.00005, -0.00003, 0.00002])

        r_ref_tau, v_ref_tau = self._true_two_body_propagate(r0, v0, dt_s, mu)
        r_pert_tau, v_pert_tau = self._true_two_body_propagate(
            r0 + dr0_eci, v0 + dv0_eci, dt_s, mu
        )
        dr_true = r_pert_tau - r_ref_tau
        dv_true = v_pert_tau - v_ref_tau

        rot0 = frames.rtn_to_eci_rotation(r0, v0)
        rot_tau = frames.rtn_to_eci_rotation(r_ref_tau, v_ref_tau)

        m0_inv = frames.eci_to_rtn_state_transform(rot0, a_km)
        m_tau = frames.rtn_to_eci_state_transform(rot_tau, a_km)

        dx0_eci = np.concatenate([dr0_eci, dv0_eci])
        dx0_rtn = m0_inv @ dx0_eci

        phi_rtn = frames.cw_phi_full(a_km, dt_s)
        dx_tau_rtn = phi_rtn @ dx0_rtn
        dx_tau_eci = m_tau @ dx_tau_rtn

        dr_pred, dv_pred = dx_tau_eci[:3], dx_tau_eci[3:]
        rel_err_r = np.linalg.norm(dr_pred - dr_true) / np.linalg.norm(dr_true)
        rel_err_v = np.linalg.norm(dv_pred - dv_true) / np.linalg.norm(dv_true)

        # Linearization error, not a defect: bounded well under 1e-3 at
        # these lead times, an order of magnitude tighter than the bound
        # used elsewhere in this repo for CW-vs-truth comparisons.
        assert rel_err_r < 1e-3, f"position rel error {rel_err_r:.2e}"
        assert rel_err_v < 1e-3, f"velocity rel error {rel_err_v:.2e}"

    @pytest.mark.parametrize(
        "a_km,inc_deg",
        [
            (6928.0, 0.0),
            (6928.0, 53.0),
            (7000.0, 90.0),
        ],
    )
    def test_eci_to_rtn_is_the_exact_analytic_inverse(self, a_km, inc_deg):
        """eci_to_rtn_state_transform must match numpy's numerical inverse
        of rtn_to_eci_state_transform, not just be A valid inverse -- the
        analytic form is claimed to be exact and cheaper than inverting."""
        inc = math.radians(inc_deg)
        r = np.array([a_km, 0.0, 0.0])
        n = math.sqrt(frames.MU_EARTH / a_km ** 3)
        v = n * a_km * np.array([0.0, math.cos(inc), math.sin(inc)])
        rot = frames.rtn_to_eci_rotation(r, v)

        m = frames.rtn_to_eci_state_transform(rot, a_km)
        m_inv_analytic = frames.eci_to_rtn_state_transform(rot, a_km)
        m_inv_numeric = np.linalg.inv(m)

        assert np.allclose(m_inv_analytic, m_inv_numeric, atol=1e-10)
        assert np.allclose(m_inv_analytic @ m, np.eye(6), atol=1e-10)

    def test_position_block_is_a_plain_rotation_no_correction(self):
        """The position half needs no transport-theorem term: r_eci =
        rot @ r_rtn exactly, so the top-left 3x3 block of the state
        transform must equal rot itself, unmodified."""
        r = np.array([6928.0, 0.0, 0.0])
        v = np.array([0.0, 7.6, 1.0])
        rot = frames.rtn_to_eci_rotation(r, v)
        m = frames.rtn_to_eci_state_transform(rot, 6928.0)
        assert np.allclose(m[0:3, 0:3], rot)
        assert np.allclose(m[0:3, 3:6], np.zeros((3, 3)))

    def test_velocity_block_reduces_to_plain_rotation_at_zero_mean_motion(self):
        """If the frame were not rotating (n=0, an unphysical but useful
        limiting case), the transport-theorem correction term vanishes
        and velocity should transform by the same plain rotation as
        position. Checked by passing a huge a_km, which drives n toward
        zero."""
        r = np.array([6928.0, 0.0, 0.0])
        v = np.array([0.0, 7.6, 1.0])
        rot = frames.rtn_to_eci_rotation(r, v)
        huge_a_km = 1e9
        m = frames.rtn_to_eci_state_transform(rot, huge_a_km)
        assert np.allclose(m[3:6, 3:6], rot)
        assert np.allclose(m[3:6, 0:3], np.zeros((3, 3)), atol=1e-9)

    def test_non_positive_a_km_raises(self):
        rot = np.eye(3)
        with pytest.raises(ValueError):
            frames.rtn_to_eci_state_transform(rot, 0.0)
        with pytest.raises(ValueError):
            frames.eci_to_rtn_state_transform(rot, -100.0)
