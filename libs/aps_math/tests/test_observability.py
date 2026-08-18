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
