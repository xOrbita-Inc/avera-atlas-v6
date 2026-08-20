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
from aps_math.observability import (
    ValidityStatus,
    classify_validity,
    conjunction_plane_basis,
    conjunction_plane_epsilon,
    marginalize_information,
    measurement_noise_covariance,
    observability_gramian,
    observability_gramian_epoch_term,
    observation_jacobian,
    weak_direction_rtn_label,
)


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


class TestMarginalizeInformation:
    """SCRUM-378: the reduction the conjunction-plane projection needs,
    applied twice (6D->3D dropping velocity, 3D->2D dropping the
    relative-velocity direction). Per review: Schur complement is the
    method for the gate, not the naive sub-block -- W is an information
    matrix, and the naive sub-block is only correct for a covariance."""

    def test_zero_coupling_naive_and_schur_agree_exactly(self):
        """When the kept and dropped subspaces are uncorrelated, there is
        nothing for the Schur complement to correct: both methods must
        give the identical result."""
        W = np.diag([2.0, 1.5, 1.0])
        naive = marginalize_information(W, [0, 2], [1], method="naive")
        schur = marginalize_information(W, [0, 2], [1], method="schur")
        assert np.allclose(naive, schur)

    def test_with_coupling_schur_differs_from_naive(self):
        """The whole point of the distinction: with real coupling, the
        two methods must NOT agree, or the fix does nothing."""
        W = np.array([
            [2.0, 0.5, 0.3],
            [0.5, 1.5, 0.4],
            [0.3, 0.4, 1.0],
        ])
        naive = marginalize_information(W, [0, 2], [1], method="naive")
        schur = marginalize_information(W, [0, 2], [1], method="schur")
        assert not np.allclose(naive, schur)

    def test_schur_result_stays_symmetric(self):
        W = np.array([
            [2.0, 0.5, 0.3],
            [0.5, 1.5, 0.4],
            [0.3, 0.4, 1.0],
        ])
        schur = marginalize_information(W, [0, 2], [1], method="schur")
        assert np.allclose(schur, schur.T)

    def test_schur_result_stays_positive_semidefinite(self):
        """A genuine information matrix property: marginalizing must not
        produce negative "information", or the result cannot be a valid
        Fisher information matrix for anything."""
        W = np.array([
            [2.0, 0.5, 0.3],
            [0.5, 1.5, 0.4],
            [0.3, 0.4, 1.0],
        ])
        schur = marginalize_information(W, [0, 2], [1], method="schur")
        eigvals = np.linalg.eigvalsh(schur)
        assert np.all(eigvals >= -1e-10)

    def test_schur_correction_is_conservative_not_optimistic(self):
        """The Schur-complement correction term is positive semi-definite
        by construction (X^T inv(W_bb) X with W_bb positive definite), so
        marginalizing can only reduce apparent information relative to
        the naive sub-block, never increase it. A validity gate that
        moved the wrong way here would be dangerous in the permissive
        direction."""
        W = np.array([
            [2.0, 0.5, 0.3],
            [0.5, 1.5, 0.4],
            [0.3, 0.4, 1.0],
        ])
        naive = marginalize_information(W, [0, 2], [1], method="naive")
        schur = marginalize_information(W, [0, 2], [1], method="schur")
        # naive - schur must itself be PSD (naive is an upper bound on schur).
        diff_eigvals = np.linalg.eigvalsh(naive - schur)
        assert np.all(diff_eigvals >= -1e-10)

    def test_works_at_6d_to_3d_scale(self):
        """The first of the Gramian's two reductions: dropping velocity
        from a full 6-state information matrix. Exercised at the actual
        scale it will be used at, not just 3x3 toy matrices."""
        rng = np.random.default_rng(42)
        A = rng.normal(size=(6, 6))
        W6 = A @ A.T + 6 * np.eye(6)  # guaranteed SPD
        position_idx = [0, 1, 2]
        velocity_idx = [3, 4, 5]
        W3 = marginalize_information(W6, position_idx, velocity_idx, method="schur")
        assert W3.shape == (3, 3)
        assert np.allclose(W3, W3.T)
        assert np.all(np.linalg.eigvalsh(W3) >= -1e-8)

    def test_naive_matches_direct_subblock_indexing(self):
        """method='naive' must be exactly equivalent to indexing the
        sub-block directly -- the 'one-line switch' this function
        provides, not a disguised different operation."""
        W = np.array([
            [2.0, 0.5, 0.3],
            [0.5, 1.5, 0.4],
            [0.3, 0.4, 1.0],
        ])
        naive = marginalize_information(W, [0, 2], [1], method="naive")
        direct = W[np.ix_([0, 2], [0, 2])]
        assert np.array_equal(naive, direct)


class TestMarginalizeInformationGuards:
    def test_singular_dropped_block_raises(self):
        """A singular W_bb means the dropped direction carries no
        information at all -- a real finding about the arc, not a
        numerical nuisance to paper over with a pseudo-inverse."""
        W = np.array([
            [2.0, 0.5, 0.3],
            [0.5, 0.0, 0.4],
            [0.3, 0.4, 1.0],
        ])
        with pytest.raises(ValueError, match="singular"):
            marginalize_information(W, [0, 2], [1], method="schur")

    def test_invalid_method_raises(self):
        W = np.eye(3)
        with pytest.raises(ValueError, match="method"):
            marginalize_information(W, [0, 2], [1], method="bogus")

    def test_overlapping_indices_raise(self):
        W = np.eye(3)
        with pytest.raises(ValueError, match="partition"):
            marginalize_information(W, [0, 1], [1, 2], method="naive")

    def test_incomplete_partition_raises(self):
        """keep_idx + drop_idx must cover every index -- silently
        dropping an index the caller forgot about is exactly the kind of
        defect this whole ticket has been about."""
        W = np.eye(3)
        with pytest.raises(ValueError, match="partition"):
            marginalize_information(W, [0], [1], method="naive")  # index 2 missing

    def test_non_square_raises(self):
        W = np.zeros((3, 4))
        with pytest.raises(ValueError, match="square"):
            marginalize_information(W, [0], [1, 2], method="naive")


class TestObservabilityGramianEpochTerm:
    """SCRUM-378: the correctly frame-combined single-epoch Gramian term,
    verified against true two-body motion, not against Phi or H
    individually -- this is the integration point where a frame mismatch
    (the SCRUM-409 failure mode, for a general state deviation) would
    show up."""

    @staticmethod
    def _true_two_body_propagate(r0, v0, dt_s, mu, n_steps=6000):
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

    @staticmethod
    def _ra_dec(r_target, r_observer):
        los = r_target - r_observer
        rng = np.linalg.norm(los)
        u = los / rng
        dec = math.asin(np.clip(u[2], -1.0, 1.0))
        ra = math.atan2(u[1], u[0])
        return np.array([ra, dec])

    def test_linear_prediction_matches_true_two_body_measurement_change(self):
        """The actual thing that matters: does H_eff correctly predict
        how the measurement changes for a real perturbation, propagated
        with real (nonlinear) two-body motion? This is the test that
        would catch a frame mismatch in combining Phi and H."""
        mu = frames.MU_EARTH
        a_km = 6928.0
        r0 = np.array([a_km, 0.0, 0.0])
        n = math.sqrt(mu / a_km ** 3)
        v0 = np.array([0.0, n * a_km, 0.0])
        r_observer = np.array([6900.0, 0.0, 0.0])
        dt_s = 300.0

        r_tau, v_tau = self._true_two_body_propagate(r0, v0, dt_s, mu)

        dr0 = np.array([0.01, -0.02, 0.005])
        dv0 = np.array([0.00001, 0.000005, -0.000003])
        dx0 = np.concatenate([dr0, dv0])

        r_pert_tau, _ = self._true_two_body_propagate(r0 + dr0, v0 + dv0, dt_s, mu)

        meas_ref = self._ra_dec(r_tau, r_observer)
        meas_pert = self._ra_dec(r_pert_tau, r_observer)
        dy_true = meas_pert - meas_ref

        rot_t0 = frames.rtn_to_eci_rotation(r0, v0)
        rot_tau = frames.rtn_to_eci_rotation(r_tau, v_tau)
        phi_rtn = frames.cw_phi_full(a_km, dt_s)
        m_tau = frames.rtn_to_eci_state_transform(rot_tau, a_km)
        m0_inv = frames.eci_to_rtn_state_transform(rot_t0, a_km)
        h_tau = observation_jacobian(r_tau, r_observer, include_range=False)
        h_eff = h_tau @ m_tau @ phi_rtn @ m0_inv

        dy_pred = h_eff @ dx0
        rel_err = np.linalg.norm(dy_pred - dy_true) / np.linalg.norm(dy_true)
        assert rel_err < 1e-3, f"rel error {rel_err:.2e}"

    def test_returns_symmetric_psd_6x6(self):
        a_km = 6928.0
        r0 = np.array([a_km, 0.0, 0.0])
        n = math.sqrt(frames.MU_EARTH / a_km ** 3)
        v0 = np.array([0.0, n * a_km, 0.0])
        r_observer = np.array([6900.0, 50.0, 20.0])

        term = observability_gramian_epoch_term(
            r0, v0, r0, v0, r_observer, a_km, 0.0,
            ra_sigma_rad=1e-5, dec_sigma_rad=1e-5,
        )
        assert term.shape == (6, 6)
        assert np.allclose(term, term.T)
        eigvals = np.linalg.eigvalsh(term)
        assert np.all(eigvals >= -1e-6)

    def test_single_epoch_term_is_rank_deficient(self):
        """A single observation cannot fully determine a 6-state: at most
        2 (angles-only) or 3 (angles+range) of the 6 eigenvalues should
        be meaningfully nonzero."""
        a_km = 6928.0
        r0 = np.array([a_km, 0.0, 0.0])
        n = math.sqrt(frames.MU_EARTH / a_km ** 3)
        v0 = np.array([0.0, n * a_km, 0.0])
        r_observer = np.array([6900.0, 50.0, 20.0])

        term = observability_gramian_epoch_term(
            r0, v0, r0, v0, r_observer, a_km, 300.0,
            ra_sigma_rad=1e-5, dec_sigma_rad=1e-5,
        )
        eigvals = np.linalg.eigvalsh(term)
        significant = np.sum(eigvals > 1e-6 * eigvals.max())
        assert significant <= 2, f"expected rank <= 2 for angles-only, got {significant}"

    def test_angles_and_range_gives_higher_rank_than_angles_only(self):
        """A sanity check on the whole chain: adding a range measurement
        should never reduce the observable rank relative to angles-only
        at the same epoch."""
        a_km = 6928.0
        r0 = np.array([a_km, 0.0, 0.0])
        n = math.sqrt(frames.MU_EARTH / a_km ** 3)
        v0 = np.array([0.0, n * a_km, 0.0])
        r_observer = np.array([6900.0, 50.0, 20.0])

        term_angles = observability_gramian_epoch_term(
            r0, v0, r0, v0, r_observer, a_km, 300.0,
            ra_sigma_rad=1e-5, dec_sigma_rad=1e-5,
        )
        term_range = observability_gramian_epoch_term(
            r0, v0, r0, v0, r_observer, a_km, 300.0,
            ra_sigma_rad=1e-5, dec_sigma_rad=1e-5, range_sigma_km=0.05,
        )
        rank_angles = np.sum(np.linalg.eigvalsh(term_angles) > 1e-6 * np.linalg.eigvalsh(term_angles).max())
        rank_range = np.sum(np.linalg.eigvalsh(term_range) > 1e-6 * np.linalg.eigvalsh(term_range).max())
        assert rank_range >= rank_angles


class TestObservabilityGramian:
    """SCRUM-378: summing single-epoch terms into the full-arc Gramian."""

    def _build_arc_terms(self, n_obs=5, dt_step_s=60.0):
        a_km = 6928.0
        r0 = np.array([a_km, 0.0, 0.0])
        n = math.sqrt(frames.MU_EARTH / a_km ** 3)
        v0 = np.array([0.0, n * a_km, 0.0])
        r_observer = np.array([6900.0, 50.0, 20.0])

        def kepler_propagate(r, v, dt, mu, n_steps=2000):
            def deriv(state):
                rr = state[:3]
                vv = state[3:]
                rn = np.linalg.norm(rr)
                aa = -mu * rr / rn ** 3
                return np.concatenate([vv, aa])
            state = np.concatenate([r, v])
            h = dt / n_steps if dt != 0 else 1.0
            steps = n_steps if dt != 0 else 0
            for _ in range(steps):
                k1 = deriv(state)
                k2 = deriv(state + h / 2 * k1)
                k3 = deriv(state + h / 2 * k2)
                k4 = deriv(state + h * k3)
                state = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return state[:3], state[3:]

        terms = []
        for i in range(n_obs):
            dt_s = i * dt_step_s
            r_tau, v_tau = kepler_propagate(r0, v0, dt_s, frames.MU_EARTH)
            term = observability_gramian_epoch_term(
                r0, v0, r_tau, v_tau, r_observer, a_km, dt_s,
                ra_sigma_rad=1e-5, dec_sigma_rad=1e-5,
            )
            terms.append(term)
        return terms

    def test_arc_sum_reaches_full_rank(self):
        """The core physical claim: individual epochs are rank-deficient,
        but a short arc with real geometric diversity builds up full
        6D rank. If this stopped being true, epsilon would be
        meaningless for every real geometry, not just degenerate ones."""
        terms = self._build_arc_terms(n_obs=5, dt_step_s=60.0)
        W = observability_gramian(terms)
        eigvals = np.linalg.eigvalsh(W)
        assert np.all(eigvals > 1e-6 * eigvals.max()), (
            f"expected full rank (6 significant eigenvalues), got "
            f"eigenvalues {eigvals}"
        )

    def test_is_symmetric(self):
        terms = self._build_arc_terms(n_obs=3)
        W = observability_gramian(terms)
        assert np.allclose(W, W.T)

    def test_is_positive_semidefinite(self):
        terms = self._build_arc_terms(n_obs=3)
        W = observability_gramian(terms)
        assert np.all(np.linalg.eigvalsh(W) >= -1e-6)

    def test_equals_sum_of_individual_terms(self):
        """Not testing anything clever -- confirms the function actually
        sums and doesn't silently drop, average, or otherwise mangle the
        inputs."""
        terms = self._build_arc_terms(n_obs=4)
        W = observability_gramian(terms)
        manual_sum = np.zeros((6, 6))
        for t in terms:
            manual_sum += t
        assert np.array_equal(W, manual_sum)

    def test_more_observations_do_not_reduce_information(self):
        """Adding an observation can only add PSD information, never
        remove it: W_more - W_fewer must itself be PSD."""
        terms_3 = self._build_arc_terms(n_obs=3, dt_step_s=60.0)
        terms_5 = self._build_arc_terms(n_obs=5, dt_step_s=60.0)
        W_3 = observability_gramian(terms_3)
        W_5 = observability_gramian(terms_5)
        diff_eigvals = np.linalg.eigvalsh(W_5 - W_3)
        assert np.all(diff_eigvals >= -1e-6)


class TestObservabilityGramianGuards:
    def test_empty_arc_raises(self):
        with pytest.raises(ValueError, match="empty"):
            observability_gramian([])

    def test_wrong_shape_term_raises(self):
        with pytest.raises(ValueError, match="shape"):
            observability_gramian([np.eye(3)])

    def test_one_correct_one_wrong_shape_raises(self):
        """The whole call must be rejected, not just the bad element
        silently skipped."""
        good = np.eye(6)
        bad = np.eye(3)
        with pytest.raises(ValueError, match="shape"):
            observability_gramian([good, bad])


class TestConjunctionPlaneBasis:
    """SCRUM-378: matches the encounter-frame convention already used by
    libs/aps_math/pc_utils.py's Pc calculation (y = relative-velocity
    direction, z = relative-motion orbit normal, x = y cross z)."""

    def test_columns_are_orthonormal(self):
        r_rel = np.array([1.0, 0.5, 0.2])
        v_rel = np.array([0.1, -0.05, 0.02])
        q = conjunction_plane_basis(r_rel, v_rel)
        assert np.allclose(q.T @ q, np.eye(3), atol=1e-12)
        assert math.isclose(np.linalg.det(q), 1.0, rel_tol=1e-9)

    def test_y_is_the_relative_velocity_direction(self):
        r_rel = np.array([1.0, 0.5, 0.2])
        v_rel = np.array([0.1, -0.05, 0.02])
        q = conjunction_plane_basis(r_rel, v_rel)
        expected_y = v_rel / np.linalg.norm(v_rel)
        assert np.allclose(q[:, 1], expected_y)

    def test_z_is_the_relative_orbit_normal(self):
        r_rel = np.array([1.0, 0.5, 0.2])
        v_rel = np.array([0.1, -0.05, 0.02])
        q = conjunction_plane_basis(r_rel, v_rel)
        h = np.cross(r_rel, v_rel)
        expected_z = h / np.linalg.norm(h)
        assert np.allclose(q[:, 2], expected_z)

    def test_zero_relative_velocity_raises(self):
        with pytest.raises(ValueError, match="relative velocity"):
            conjunction_plane_basis(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))

    def test_parallel_position_and_velocity_raises(self):
        with pytest.raises(ValueError, match="degenerate"):
            conjunction_plane_basis(np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))


class TestConjunctionPlaneEpsilon:
    """SCRUM-378, MAF v2.0 Sec 7."""

    @staticmethod
    def _kepler_propagate(r0, v0, dt, mu, n_steps=2000):
        def deriv(state):
            r = state[:3]
            v = state[3:]
            r_norm = np.linalg.norm(r)
            a = -mu * r / r_norm ** 3
            return np.concatenate([v, a])
        state = np.concatenate([r0, v0])
        steps = n_steps if dt != 0 else 0
        h = dt / n_steps if dt != 0 else 1.0
        for _ in range(steps):
            k1 = deriv(state)
            k2 = deriv(state + h / 2 * k1)
            k3 = deriv(state + h / 2 * k2)
            k4 = deriv(state + h * k3)
            state = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return state[:3], state[3:]

    def _build_realistic_arc_position_info(self):
        """A physically realistic short angles-only tracking arc, ground
        station to LEO primary, all observations before TCA (t0=TCA per
        this ticket's own resolved epoch convention -- see
        conjunction_plane_epsilon's docstring)."""
        mu = frames.MU_EARTH
        a_km = 6928.0  # ~550 km altitude
        r_tca = np.array([a_km, 0.0, 0.0])
        n = math.sqrt(mu / a_km ** 3)
        v_tca = np.array([0.0, n * a_km, 0.0])

        r_earth = 6378.0
        r_observer = np.array(
            [r_earth * math.cos(math.radians(5)), 0.0, r_earth * math.sin(math.radians(5))]
        )

        terms = []
        for i in range(5):
            dt_s = -(300.0 - i * 60.0)
            r_tau, v_tau = self._kepler_propagate(r_tca, v_tca, dt_s, mu)
            term = observability_gramian_epoch_term(
                r_tca, v_tca, r_tau, v_tau, r_observer, a_km, dt_s,
                ra_sigma_rad=1e-5, dec_sigma_rad=1e-5,
            )
            terms.append(term)

        w = observability_gramian(terms)
        w_pos = marginalize_information(w, keep_idx=[0, 1, 2], drop_idx=[3, 4, 5], method="schur")
        return w_pos, r_tca, v_tca

    def test_epsilon_stays_in_zero_one(self):
        """A ratio of the smallest to largest eigenvalue of a positive
        definite matrix is always in [0, 1] -- checked across several
        random PSD position-information matrices and geometries, not
        just one convenient case."""
        rng = np.random.default_rng(1)
        checked = 0
        for _ in range(20):
            a = rng.normal(size=(3, 3))
            w_pos = a @ a.T + 3 * np.eye(3)
            r_rel = rng.normal(size=3)
            v_rel = rng.normal(size=3) * 0.01
            if np.linalg.norm(np.cross(r_rel, v_rel)) < 1e-6:
                continue
            eps, _, _ = conjunction_plane_epsilon(w_pos, r_rel, v_rel)
            assert 0.0 <= eps <= 1.0
            checked += 1
        assert checked >= 15, "too many degenerate geometries were skipped to be a meaningful check"

    def test_weak_direction_is_a_unit_vector(self):
        w_pos = np.diag([2.0, 1.5, 1.0])
        r_rel = np.array([1.0, 0.0, 0.0])
        v_rel = np.array([0.0, 1.0, 0.0])
        _, _, weak_dir = conjunction_plane_epsilon(w_pos, r_rel, v_rel)
        assert math.isclose(np.linalg.norm(weak_dir), 1.0, rel_tol=1e-9)

    def test_weak_direction_has_zero_relative_velocity_component(self):
        """The weak direction lives IN the conjunction plane by
        construction, so it must be orthogonal to the relative-velocity
        direction (the y-axis that got marginalized out)."""
        w_pos = np.diag([2.0, 1.5, 1.0])
        r_rel = np.array([1.0, 0.3, 0.1])
        v_rel = np.array([0.05, 1.0, -0.02])
        _, _, weak_dir = conjunction_plane_epsilon(w_pos, r_rel, v_rel)
        y_hat = v_rel / np.linalg.norm(v_rel)
        assert math.isclose(float(np.dot(weak_dir, y_hat)), 0.0, abs_tol=1e-9)

    def test_matches_the_docs_stated_physical_expectation(self):
        """SCRUM-333's own validity doc: 'Typical failure mode for
        CDM-based LEO conjunctions: poor radial observability from short
        tracking arcs. Cross-track is typically the best-resolved
        direction.' Checked here against the FULL 3D position
        information (before any conjunction-plane projection, which
        depends on a specific secondary's geometry): the weakest and
        strongest full-3D directions should be radial and cross-track
        respectively, in RTN terms. This is not just internal
        self-consistency -- it is the pipeline reproducing a real,
        independently-stated physical prediction."""
        w_pos_eci, r_tca, v_tca = self._build_realistic_arc_position_info()
        rot_tca = frames.rtn_to_eci_rotation(r_tca, v_tca)
        w_pos_rtn = rot_tca.T @ w_pos_eci @ rot_tca
        eigvals, eigvecs = np.linalg.eigh(w_pos_rtn)

        labels = ["R", "T", "N"]
        weakest_label = labels[np.argmax(np.abs(eigvecs[:, 0]))]
        strongest_label = labels[np.argmax(np.abs(eigvecs[:, -1]))]

        assert weakest_label == "R", (
            f"expected radial to be the weakest-observed direction for a "
            f"short single-station angles-only arc, got {weakest_label}"
        )
        assert strongest_label == "N", (
            f"expected cross-track to be the best-resolved direction, "
            f"got {strongest_label}"
        )

    def test_wrong_shape_w_position_raises(self):
        with pytest.raises(ValueError, match="3x3"):
            conjunction_plane_epsilon(np.eye(6), np.array([1.0, 0, 0]), np.array([0.0, 1.0, 0]))

    def test_singular_dropped_block_raises(self):
        """Propagates marginalize_information's own guard: zero
        information in the direction being dropped is a real finding
        about the geometry, not a numerical nuisance."""
        w_singular = np.diag([1.0, 0.0, 1.0])
        r_rel = np.array([1.0, 0.0, 0.0])
        v_rel = np.array([0.0, 1.0, 0.0])
        with pytest.raises(ValueError):
            conjunction_plane_epsilon(w_singular, r_rel, v_rel)


class TestClassifyValidity:
    """SCRUM-378, MAF v2.0 Sec 7: EARNED if epsilon >= threshold, else
    NOT_EARNED. Matches services/tracker/iod.py's classify_iod_confidence
    pattern: a classify function plus a str Enum verdict."""

    def test_above_threshold_is_earned(self):
        assert classify_validity(0.25, 0.20) == ValidityStatus.EARNED

    def test_below_threshold_is_not_earned(self):
        assert classify_validity(0.15, 0.20) == ValidityStatus.NOT_EARNED

    def test_exactly_at_threshold_is_earned(self):
        """The doc says 'at or above' -- the boundary is inclusive."""
        assert classify_validity(0.20, 0.20) == ValidityStatus.EARNED

    def test_locked_leo_threshold_from_the_ticket(self):
        """Pins the actual SCRUM-333/SCRUM-378 locked value, 0.20, so the
        ticket and the code cannot silently drift apart."""
        assert classify_validity(0.1999, 0.20) == ValidityStatus.NOT_EARNED
        assert classify_validity(0.2000, 0.20) == ValidityStatus.EARNED
        assert classify_validity(0.2001, 0.20) == ValidityStatus.EARNED

    def test_nan_epsilon_is_not_earned(self):
        """The conservative fallback on invalid input, matching
        classify_iod_confidence's own pattern."""
        assert classify_validity(float("nan"), 0.20) == ValidityStatus.NOT_EARNED

    def test_negative_epsilon_is_not_earned(self):
        assert classify_validity(-0.1, 0.20) == ValidityStatus.NOT_EARNED

    def test_returns_a_str_enum_member(self):
        """Matches IODConfidenceVerdict's str, Enum pattern: usable
        directly as a string (e.g. for JSON serialization) without an
        explicit .value access."""
        result = classify_validity(0.25, 0.20)
        assert result == "EARNED"
        assert isinstance(result, str)


class TestWeakDirectionRtnLabel:
    """SCRUM-378, MAF v2.0 Sec 7: maps the conjunction-plane weak
    direction back to an RTN label for ValidityVerdict's weak_directions
    field."""

    def _rtn_rotation(self):
        a_km = 6928.0
        r = np.array([a_km, 0.0, 0.0])
        n = math.sqrt(frames.MU_EARTH / a_km ** 3)
        v = np.array([0.0, n * a_km, 0.0])
        return frames.rtn_to_eci_rotation(r, v)

    def test_pure_radial_direction_labels_radial(self):
        rot = self._rtn_rotation()
        weak_dir = rot[:, 0]  # R column
        assert weak_direction_rtn_label(weak_dir, rot) == ["radial"]

    def test_pure_transverse_direction_labels_transverse(self):
        """gnc_interface.yaml's weak_directions enum is [radial,
        transverse, normal], not along-track/cross-track."""
        rot = self._rtn_rotation()
        weak_dir = rot[:, 1]  # T column
        assert weak_direction_rtn_label(weak_dir, rot) == ["transverse"]

    def test_pure_normal_direction_labels_normal(self):
        rot = self._rtn_rotation()
        weak_dir = rot[:, 2]  # N column
        assert weak_direction_rtn_label(weak_dir, rot) == ["normal"]

    def test_returns_a_single_element_list(self):
        """Matches the interface field's documented shape, e.g.
        ["radial"], not a bare string."""
        rot = self._rtn_rotation()
        result = weak_direction_rtn_label(rot[:, 0], rot)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_dominant_component_wins_for_a_mixed_direction(self):
        """A direction mostly radial with a small along-track component
        should still label as radial -- the dominant axis, not a
        compound label, per this function's documented simplification."""
        rot = self._rtn_rotation()
        mixed_rtn = np.array([0.9, 0.1, 0.0])
        mixed_rtn = mixed_rtn / np.linalg.norm(mixed_rtn)
        mixed_eci = rot @ mixed_rtn
        assert weak_direction_rtn_label(mixed_eci, rot) == ["radial"]

    def test_output_always_validates_against_the_published_contract_enum(self):
        """Direct regression guard for the bug this exact test file
        originally had: weak_direction_rtn_label used to emit
        along-track/cross-track, which openapi/gnc_interface.yaml's
        weak_directions enum ([radial, transverse, normal]) does not
        accept. Checked against all three RTN axes, not just one."""
        contract_enum = {"radial", "transverse", "normal"}
        rot = self._rtn_rotation()
        for axis_idx in range(3):
            result = weak_direction_rtn_label(rot[:, axis_idx], rot)
            assert len(result) == 1
            assert result[0] in contract_enum, (
                f"'{result[0]}' is not in the published contract's "
                f"weak_directions enum {contract_enum}"
            )
