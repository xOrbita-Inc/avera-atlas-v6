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


# ---------------------------------------------------------------------------
# Two-body plus J2 (SCRUM-451)
# ---------------------------------------------------------------------------
#
# kepler_propagate above is untouched and stays the planner's internal
# propagator. This is added beside it for one consumer: the ephemeris we hand
# LeoLabs for on-demand screening.
#
# Why it exists. The SCRUM-441 screen came back falsely clear -- 0 conjunctions
# at a 25 km miss-distance box where the catalog has 64 events over the same
# window -- because the submitted trajectory was propagated two-body. A 25 km box
# is a geometric filter, so covariance cannot explain a miss at that size; the
# trajectory itself was in the wrong place. Measured two-body vs J2 divergence for
# a Swarm-C-like orbit: ~26 km at 1 h, ~462 km at 24 h, ~1400 km at 72 h. The
# submitted path left even a 25 km box inside the first hour.
#
# Scope. J2 only. Drag, solar radiation pressure and higher zonal harmonics are
# deliberately not modelled and are later refinement -- J2 removes the bulk of the
# error above, and the residual over 72 h in LEO is dominated by drag.
#
# Frame. The J2 bulge is aligned with the Earth equator *of date*, while our states
# are EME2000. Integrating in EME2000 therefore mis-aligns the bulge by the
# J2000-to-date pole offset, which is small and whose effect over 72 h is well
# inside any screening volume we use. Capturing the secular J2 effect is what
# matters here, and it dwarfs the pole-frame subtlety. Recorded rather than
# silently assumed.

J2 = 1.08262668e-3
RE_EARTH_KM = 6378.137

# Fallback fixed-step size, used only when scipy is unavailable. Small enough that
# RK4 tracks the reference integrator to well inside a screening volume over 72 h.
_J2_RK4_STEP_S = 10.0


def j2_acceleration(
    r: np.ndarray, mu: float = MU_EARTH, j2: float = J2, re: float = RE_EARTH_KM
) -> np.ndarray:
    """The J2 perturbing acceleration at position r (km), in km/s^2.

    The oblateness term only: the two-body term is added by the caller. Signs are
    worth checking against physical intuition rather than trusting the algebra --
    at the equator (z = 0) this is inward, because the bulge puts extra mass in
    the equatorial plane, and over a pole (x = y = 0) it is outward, because there
    is less mass below.
    """
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    r_mag = float(np.sqrt(x * x + y * y + z * z))
    zr2 = (z * z) / (r_mag * r_mag)
    k = 1.5 * j2 * mu * re * re / (r_mag ** 5)
    return np.array([
        k * x * (5.0 * zr2 - 1.0),
        k * y * (5.0 * zr2 - 1.0),
        k * z * (5.0 * zr2 - 3.0),
    ], dtype=float)


def _j2_derivative(state: np.ndarray, mu: float, j2: float, re: float) -> np.ndarray:
    r = state[:3]
    v = state[3:]
    r_mag = float(np.linalg.norm(r))
    a = -mu * r / (r_mag ** 3) + j2_acceleration(r, mu=mu, j2=j2, re=re)
    return np.concatenate((v, a))


def _validate_state(r0, v0):
    r = np.asarray(r0, dtype=float)
    v = np.asarray(v0, dtype=float)
    if r.shape != (3,) or v.shape != (3,):
        raise ValueError("r0 and v0 must each be 3-element vectors")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
        raise ValueError("r0 and v0 must contain only finite values")
    if float(np.linalg.norm(r)) < 100.0:
        raise ValueError(
            "initial position magnitude is too small for Earth-orbit propagation"
        )
    return r, v


def propagate_j2_series(
    r0,
    v0,
    times_s,
    mu: float = MU_EARTH,
    j2: float = J2,
    re: float = RE_EARTH_KM,
    max_step_s: float = _J2_RK4_STEP_S,
):
    """Propagate one state to many times under two-body plus J2.

    Returns (positions, velocities) as (N, 3) arrays in km and km/s, one row per
    entry in times_s. times_s must be non-negative and non-decreasing, measured
    from the epoch of (r0, v0); t = 0 returns the input state unchanged.

    This is the form the screening-ephemeris builder wants: one integration
    sampled at the output cadence, rather than N independent propagations of
    growing length, which for an 865-state file is the difference between one
    pass and 865.

    scipy's DOP853 is used when available, with tight tolerances. Without scipy it
    falls back to fixed-step RK4 at max_step_s, sub-stepping between requested
    times so the internal step stays small regardless of the output cadence. Both
    are deterministic; a test pins the two against each other.
    """
    r, v = _validate_state(r0, v0)
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times_s must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(times)):
        raise ValueError("times_s must be finite")
    if float(times[0]) < 0.0 or np.any(np.diff(times) < 0.0):
        raise ValueError("times_s must be non-negative and non-decreasing")

    state0 = np.concatenate((r, v))

    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        return _propagate_j2_rk4(state0, times, mu, j2, re, max_step_s)

    t_end = float(times[-1])
    if t_end == 0.0:
        return (np.repeat(r[None, :], times.size, axis=0),
                np.repeat(v[None, :], times.size, axis=0))

    solution = solve_ivp(
        lambda _t, y: _j2_derivative(y, mu, j2, re),
        (0.0, t_end),
        state0,
        method="DOP853",
        t_eval=times,
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"J2 propagation failed: {solution.message}")
    out = solution.y.T
    return out[:, :3].copy(), out[:, 3:].copy()


def _propagate_j2_rk4(state0, times, mu, j2, re, max_step_s):
    """Fixed-step RK4 fallback, sub-stepping between requested output times."""
    positions = np.empty((times.size, 3), dtype=float)
    velocities = np.empty((times.size, 3), dtype=float)
    state = np.asarray(state0, dtype=float).copy()
    t_now = 0.0

    def deriv(y):
        return _j2_derivative(y, mu, j2, re)

    for i, t_target in enumerate(times):
        span = float(t_target) - t_now
        if span > 0.0:
            n = max(1, int(np.ceil(span / max_step_s)))
            h = span / n
            for _ in range(n):
                k1 = deriv(state)
                k2 = deriv(state + 0.5 * h * k1)
                k3 = deriv(state + 0.5 * h * k2)
                k4 = deriv(state + h * k3)
                state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t_now = float(t_target)
        positions[i] = state[:3]
        velocities[i] = state[3:]
    return positions, velocities


def propagate_j2(
    r0,
    v0,
    dt: float,
    mu: float = MU_EARTH,
    j2: float = J2,
    re: float = RE_EARTH_KM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate one state by dt seconds under two-body plus J2.

    The singular form, mirroring kepler_propagate's signature so the two read the
    same at a call site. Delegates to propagate_j2_series; a caller wanting many
    epochs should use that directly rather than calling this in a loop.
    """
    if not np.isfinite(dt):
        raise ValueError("dt must be finite")
    if dt < 0.0:
        raise ValueError("propagate_j2 does not support backward propagation")
    positions, velocities = propagate_j2_series(
        r0, v0, np.array([float(dt)]), mu=mu, j2=j2, re=re
    )
    return positions[0], velocities[0]
