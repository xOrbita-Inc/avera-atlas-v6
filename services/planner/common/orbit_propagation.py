from __future__ import annotations

from typing import Tuple

import numpy as np

MU_EARTH = 398600.4418  # km^3/s^2


def kepler_propagate(r0: np.ndarray, v0: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate a bound elliptic state with universal-variable two-body dynamics.

    Hyperbolic and near-parabolic trajectories are deliberately unsupported
    here rather than silently replaced with straight-line motion.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    if r0.shape != (3,) or v0.shape != (3,):
        raise ValueError("r0 and v0 must each be 3-element vectors")
    if not np.all(np.isfinite(r0)) or not np.all(np.isfinite(v0)):
        raise ValueError("r0 and v0 must contain only finite values")
    if not np.isfinite(dt):
        raise ValueError("dt must be finite")

    if abs(dt) < 1e-10:
        return r0.copy(), v0.copy()

    mu = MU_EARTH
    sqrt_mu = np.sqrt(mu)
    r0_mag = float(np.linalg.norm(r0))
    if r0_mag < 100.0:
        raise ValueError("initial position magnitude is too small for Earth-orbit propagation")

    v0_sq = float(np.dot(v0, v0))
    alpha = 2.0 / r0_mag - v0_sq / mu  # reciprocal semi-major axis [1/km]

    # Preserve the old near-parabolic boundary (|a| > 1e8 km), but fail
    # explicitly instead of substituting a linear trajectory.
    if abs(alpha) < 1e-8:
        raise ValueError("near-parabolic trajectories are unsupported by kepler_propagate")
    if alpha < 0.0:
        raise ValueError("hyperbolic trajectories are unsupported by kepler_propagate")

    vr0 = float(np.dot(r0, v0) / r0_mag)
    radial_coeff = r0_mag * vr0 / sqrt_mu

    def stumpff_c2_c3(z: float) -> Tuple[float, float]:
        if z > 1e-6:
            sqrt_z = np.sqrt(z)
            c2 = (1.0 - np.cos(sqrt_z)) / z
            c3 = (sqrt_z - np.sin(sqrt_z)) / (sqrt_z * z)
        elif z < -1e-6:
            sqrt_neg_z = np.sqrt(-z)
            c2 = (np.cosh(sqrt_neg_z) - 1.0) / (-z)
            c3 = (np.sinh(sqrt_neg_z) - sqrt_neg_z) / ((-z) ** 1.5)
        else:
            z2 = z * z
            c2 = 0.5 - z / 24.0 + z2 / 720.0
            c3 = 1.0 / 6.0 - z / 120.0 + z2 / 5040.0
        return float(c2), float(c3)

    # Elliptic universal-anomaly initial guess. The sign follows dt so
    # backward propagation is handled by the same equations.
    chi = sqrt_mu * dt * alpha
    converged = False
    max_iter = 50
    tol = 1e-10

    for _ in range(max_iter):
        z = alpha * chi * chi
        c2, c3 = stumpff_c2_c3(z)

        # Universal Kepler residual F(chi) = 0 and its derivative.
        # The r0_mag factor on the radial-velocity term is required for
        # general states away from an apsis.
        F = (
            radial_coeff * chi * chi * c2
            + (1.0 - alpha * r0_mag) * chi**3 * c3
            + r0_mag * chi
            - sqrt_mu * dt
        )
        dF = (
            radial_coeff * chi * (1.0 - z * c3)
            + (1.0 - alpha * r0_mag) * chi * chi * c2
            + r0_mag
        )

        if not np.isfinite(F) or not np.isfinite(dF) or abs(dF) < 1e-12:
            raise RuntimeError("universal-variable iteration produced an invalid Newton step")

        delta_chi = F / dF
        chi -= delta_chi
        if abs(delta_chi) < tol:
            converged = True
            break

    if not converged or not np.isfinite(chi):
        raise RuntimeError("universal-variable Kepler propagation did not converge")

    z = alpha * chi * chi
    c2, c3 = stumpff_c2_c3(z)
    chi2 = chi * chi

    f = 1.0 - chi2 / r0_mag * c2
    g = dt - chi**3 / sqrt_mu * c3
    r_new = f * r0 + g * v0
    r_new_mag = float(np.linalg.norm(r_new))

    if r_new_mag < 100.0 or not np.all(np.isfinite(r_new)):
        raise RuntimeError("universal-variable propagation produced an invalid position")

    fdot = sqrt_mu / (r_new_mag * r0_mag) * chi * (z * c3 - 1.0)
    gdot = 1.0 - chi2 / r_new_mag * c2
    v_new = fdot * r0 + gdot * v0

    if not np.all(np.isfinite(v_new)):
        raise RuntimeError("universal-variable propagation produced an invalid velocity")

    return r_new, v_new
