"""services/planner/common/leolabs_ephemeris.py

SCRUM-440: build the post-burn trajectory ephemeris LeoLabs screens on demand.

This module stops at the JSON. Submitting it, polling the screening and reading
the resulting CDMs is SCRUM-441; the client already has that lifecycle
(create_screening, get_screening_status, wait_for_screening, get_screening_cdms)
and nothing here calls it.

The emitted shape, confirmed on the wire
----------------------------------------
Checked against a live GET /catalog/objects/<catalog>/states on 2026-09-24 rather
than taken from the format note. `frames.EME2000` carries:

    position            3 floats, metres
    velocity            3 floats, metres/second
    covariance          6x6 nested lists, m^2 and (m/s)^2, [r, v] ordering:
                        position block top-left, velocity block bottom-right
    covarianceExtended  8x8, of which `covariance` is exactly the top-left 6x6

So a 6x6 under the key `covariance` is LeoLabs' own representation, and what we
emit here is the same object they hand back. The extra two rows of the extended
form are not ours to fill and are not emitted.

Units are the thing most likely to be wrong here, so they are stated once and
converted in exactly one place each: km -> m (x1e3) on position, km/s -> m/s
(x1e3) on velocity, km^2 -> m^2 (x1e6) on covariance.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common.orbit_propagation import propagate_j2_series

log = logging.getLogger("planner")

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

# The SCRUM-381 secondary screen looks across OperatorPolicy.max_hours_before_tca,
# which authorization_envelope locks at 72 h (LOCKED_MAX_TCA_HOURS). The ephemeris
# has to cover exactly what the screen covers, so that is the default here. A
# caller holding the live policy should pass its own value rather than rely on
# this constant drifting in step.
DEFAULT_HORIZON_HOURS = 72.0

# Sampling cadence. Deliberately NOT secondary_horizon._TCA_BRACKET_STEP_S: that
# 60 s is a root-finding bracket for locating stationary points of relative
# distance, not a statement about how densely a trajectory should be described.
#
# 300 s over 72 h is 865 states, roughly 18 per LEO revolution, which keeps the
# file small while still describing the arc. It is a judgement, not a measurement:
# the right cadence depends on how LeoLabs interpolates between submitted states,
# which is not documented in what we have and is a question for the first live
# submit in SCRUM-441. Pass a finer step if that answer demands one.
DEFAULT_STEP_S = 300.0

FRAME = "EME2000"

_KM_TO_M = 1.0e3
_KM2_TO_M2 = 1.0e6

# A covariance out of the parser has been through a rotation and a sum, so exact
# symmetry is not guaranteed in floating point; this is relative to the matrix
# scale, matching leolabs_runtime's own tolerance for the same reason.
_SYMMETRY_RTOL = 1e-6
_PSD_RTOL = 1e-8


class LeoLabsEphemerisError(ValueError):
    """The inputs cannot produce an honest ephemeris."""


# ---------------------------------------------------------------------------
# The one physics decision: a 3x3 position covariance into a 6x6
# ---------------------------------------------------------------------------

# SCRUM-452 removed covariance_6x6_from_position_3x3. It placed a position
# covariance into a 6x6 with a zero velocity block and emitted it unchanged on
# every state, and its own docstring named the replacement as follow-on work:
# propagate the seed along the trajectory so the covariance grows. That is what
# the helpers below now do, so the old one was dead code carrying a caveat that
# is no longer true.


def execution_error_velocity_covariance_km2_s2(
    dv_eci_km_s: Sequence[float],
    magnitude_fraction: float,
    pointing_sigma_rad: float,
) -> np.ndarray:
    """Burn execution error as a 3x3 velocity covariance in EME2000, km^2/s^2.

    The Gates model. In the burn frame, with dv_hat the commanded burn
    direction:

        magnitude 1-sigma   f * |dv|           along dv_hat
        pointing 1-sigma    theta * |dv|       in each direction perpendicular
                                               to dv_hat

    so the burn-frame covariance is diag((f|dv|)^2, (theta|dv|)^2, (theta|dv|)^2),
    rotated into EME2000 by the orthonormal frame whose first axis is dv_hat.
    The small-angle form is used for the pointing term: a pointing error theta
    tips the burn by theta, putting |dv| * sin(theta) ~ |dv| * theta of velocity
    into the perpendicular plane.

    This is the load-bearing half of the post-burn covariance. A velocity
    1-sigma integrates into an along-track position 1-sigma of order sigma_v * t,
    so across a 72 h screening horizon it dominates the position seed by orders
    of magnitude -- which is exactly why the position-only seed this replaces
    barely fanned out at all over the same window.

    A zero delta-v gives a zero velocity covariance, which is correct rather than
    degenerate: no burn, no execution error. The caller decides whether a
    zero-velocity seed is meaningful.
    """
    dv = np.asarray(dv_eci_km_s, dtype=float)
    if dv.shape != (3,):
        raise LeoLabsEphemerisError("delta-v must be a 3-component vector, km/s")
    if not np.all(np.isfinite(dv)):
        raise LeoLabsEphemerisError("delta-v must be finite")
    f = float(magnitude_fraction)
    theta = float(pointing_sigma_rad)
    if f < 0 or theta < 0:
        raise LeoLabsEphemerisError(
            "execution-error parameters must be non-negative"
        )

    dv_mag = float(np.linalg.norm(dv))
    if dv_mag == 0.0:
        return np.zeros((3, 3), dtype=float)

    dv_hat = dv / dv_mag
    # Any vector not parallel to dv_hat completes the frame; the ellipsoid is
    # rotationally symmetric about dv_hat, so which perpendicular pair is chosen
    # cannot change the result.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, dv_hat))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(dv_hat, seed)
    u /= np.linalg.norm(u)
    w = np.cross(dv_hat, u)

    rotation = np.column_stack((dv_hat, u, w))   # burn frame -> ECI
    burn_frame = np.diag([
        (f * dv_mag) ** 2,
        (theta * dv_mag) ** 2,
        (theta * dv_mag) ** 2,
    ])
    cov = rotation @ burn_frame @ rotation.T
    return (cov + cov.T) / 2.0


def seed_post_burn_covariance_km2(
    p_post_pos_km2: Any,
    dv_eci_km_s: Sequence[float],
    magnitude_fraction: float,
    pointing_sigma_rad: float,
) -> np.ndarray:
    """The post-burn 6x6 at the burn epoch, EME2000, km-based units.

    Position block: the primary's own post-burn position covariance. Velocity
    block: the execution error above. Cross terms zero, because position and
    velocity are uncorrelated at the instant of the burn -- the correlations are
    not assumed here, they develop under propagation and the growth step
    produces them.

    Units are km^2, km^2/s and km^2/s^2 by block, matching the propagator the
    growth uses. The emitting layer converts to metres once.
    """
    position = _validated_position_covariance(p_post_pos_km2)
    velocity = execution_error_velocity_covariance_km2_s2(
        dv_eci_km_s, magnitude_fraction, pointing_sigma_rad
    )
    p0 = np.zeros((6, 6), dtype=float)
    p0[:3, :3] = position
    p3 = velocity
    p0[3:, 3:] = p3
    return p0


def state_transition_matrices(
    r0_km: Sequence[float],
    v0_km_s: Sequence[float],
    times_s: Sequence[float],
    step_fraction: float = 1.0e-6,
) -> np.ndarray:
    """Phi(t) for each requested time, by finite-differencing the J2 propagator.

    Returns an (N, 6, 6) array. Column j of Phi(t) is the response of the state
    at t to a unit perturbation of initial component j, estimated by propagating
    the perturbed initial state and differencing against the nominal.

    Seven passes total -- one nominal and six perturbed -- so an 865-state file
    costs seven integrations rather than one per state. The SCRUM-451 J2
    propagator is used unchanged; this differentiates it rather than replacing
    it, which means the covariance grows under exactly the dynamics the emitted
    trajectory was generated with.

    The perturbation is scaled to each component's own magnitude, with a floor,
    so a near-zero component still gets a usable step and a large one is not
    perturbed below numerical resolution.
    """
    r0 = np.asarray(r0_km, dtype=float)
    v0 = np.asarray(v0_km_s, dtype=float)
    times = np.asarray(times_s, dtype=float)
    state0 = np.concatenate((r0, v0))

    nominal_r, nominal_v = propagate_j2_series(r0, v0, times)
    nominal = np.hstack((nominal_r, nominal_v))          # (N, 6)

    phis = np.empty((times.size, 6, 6), dtype=float)
    for j in range(6):
        # Position components are ~1e3 km and velocity ~1e0 km/s, so a relative
        # step keeps both well-conditioned; the floor covers a component at zero.
        scale = max(abs(float(state0[j])), 1.0)
        eps = step_fraction * scale
        perturbed_state = state0.copy()
        perturbed_state[j] += eps
        pert_r, pert_v = propagate_j2_series(
            perturbed_state[:3], perturbed_state[3:], times
        )
        perturbed = np.hstack((pert_r, pert_v))
        phis[:, :, j] = (perturbed - nominal) / eps
    return phis


def propagate_covariance(p0: np.ndarray, phis: np.ndarray) -> np.ndarray:
    """P(t) = Phi(t) P0 Phi(t)^T for each Phi, symmetrised.

    Symmetrising at each step is not cosmetic: Phi P Phi^T accumulates asymmetry
    in floating point, and a covariance that is not symmetric is not a
    covariance -- downstream consumers eigendecompose it.
    """
    p0 = np.asarray(p0, dtype=float)
    out = np.empty_like(phis)
    for i in range(phis.shape[0]):
        phi = phis[i]
        p = phi @ p0 @ phi.T
        out[i] = (p + p.T) / 2.0
    return out


def _validated_position_covariance(p_pos_km2: Any) -> np.ndarray:
    """The 3x3 as a usable covariance, or raise.

    post_burn_covariance_km2's own rule is that None means unknown and must not
    be read as zero. Honouring that here means refusing to build rather than
    emitting a zero covariance: a screening run against a trajectory declared
    perfectly known is not the screening the caller asked for, and it would look
    like a successful submission.
    """
    if p_pos_km2 is None:
        raise LeoLabsEphemerisError(
            "post-burn covariance is None, which means unknown, not zero; refusing "
            "to emit an ephemeris that would declare the trajectory exactly known"
        )
    try:
        p = np.asarray(p_pos_km2, dtype=float)
    except (TypeError, ValueError) as exc:
        raise LeoLabsEphemerisError(f"covariance is not numeric: {exc}") from exc
    if p.shape != (3, 3):
        raise LeoLabsEphemerisError(f"covariance is {p.shape}, expected (3, 3)")
    if not np.all(np.isfinite(p)):
        raise LeoLabsEphemerisError("covariance has non-finite entries")

    scale = float(np.max(np.abs(p)))
    if scale == 0.0:
        raise LeoLabsEphemerisError(
            "covariance is all zero, which asserts a perfectly known trajectory"
        )
    if float(np.max(np.abs(p - p.T))) > _SYMMETRY_RTOL * scale:
        raise LeoLabsEphemerisError("covariance is not symmetric")
    eigenvalues = np.linalg.eigvalsh(p)
    if float(np.min(eigenvalues)) < -_PSD_RTOL * scale:
        raise LeoLabsEphemerisError(
            f"covariance is not positive semi-definite (min eigenvalue "
            f"{float(np.min(eigenvalues)):.6g})"
        )
    return p


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _parse_epoch(epoch_utc: str) -> datetime:
    if not epoch_utc:
        raise LeoLabsEphemerisError("epoch_utc is required")
    try:
        parsed = datetime.fromisoformat(str(epoch_utc).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise LeoLabsEphemerisError(f"epoch_utc is not ISO-8601: {epoch_utc!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_z(moment: datetime) -> str:
    """ISO-8601 with a literal Z, matching what LeoLabs emits and this repo uses."""
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _state_vector(r_km: Any, v_km_s: Any) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r_km, dtype=float)
    v = np.asarray(v_km_s, dtype=float)
    if r.shape != (3,) or v.shape != (3,):
        raise LeoLabsEphemerisError(
            "post-burn state must be 3 position and 3 velocity components"
        )
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
        raise LeoLabsEphemerisError("post-burn state must be finite")
    return r, v


def build_screening_ephemeris(
    epoch_utc: str,
    r_sat_km: Sequence[float],
    v_sat_km_s: Sequence[float],
    p_post_eci_km2: Optional[Sequence[Sequence[float]]],
    horizon_hours: float = DEFAULT_HORIZON_HOURS,
    step_s: float = DEFAULT_STEP_S,
    *,
    dv_eci_km_s: Optional[Sequence[float]] = None,
    execution_error_magnitude_fraction: float = 0.0,
    execution_error_pointing_sigma_rad: float = 0.0,
) -> Dict[str, Any]:
    """The post-burn trajectory as a LeoLabs screening ephemeris.

    Propagates the post-burn state under two-body plus J2 (SCRUM-451) from
    epoch_utc across horizon_hours at step_s, and emits:

        {"frame": "EME2000", "covarianceFrame": "EME2000",
         "states": [{"timestamp": ..., "position": [m x3],
                     "velocity": [m/s x3], "covariance": [6x6]}, ...]}

    The first state is the post-burn state itself at epoch_utc, unpropagated.

    Why J2 and not kepler_propagate (SCRUM-451)
    -------------------------------------------
    SCRUM-440 propagated this file two-body, and the SCRUM-441 screen built on it
    came back falsely clear: 0 conjunctions at a 25 km miss-distance box where the
    catalog holds 64 events over the same window. A miss-distance box is a
    geometric filter, so covariance could not explain a miss at that size -- the
    submitted trajectory was simply in the wrong place. Two-body versus J2 for a
    Swarm-C-like orbit diverges by ~26 km within the first hour and ~1400 km by
    72 h, so the submitted path left even a 25 km box almost immediately.

    This is the fidelity of the file we hand LeoLabs, not the planner's internal
    dynamics: kepler_propagate is untouched and secondary_horizon, maneuver_scorer,
    the globe and the propagator service all keep using it. Drag, SRP and higher
    zonals remain unmodelled and are later refinement.

    Each state carries its OWN covariance, grown along the trajectory (SCRUM-452).
    The seed is the primary's post-burn position covariance plus the burn
    execution error as a velocity block, and it is propagated as
    P(t) = Phi(t) P0 Phi(t)^T with Phi finite-differenced from the same J2
    propagator that generated the states. So the uncertainty fans out the way real
    uncertainty does, most of it along-track, instead of being repeated unchanged
    across 72 h.

    dv_eci_km_s and the two execution-error parameters seed the velocity block.
    Left at their defaults the velocity block is zero and the covariance grows
    only by propagating the position seed, which is a much weaker fan-out; that
    case is logged, because a caller that forgot to pass the burn would otherwise
    get a quietly under-grown covariance.

    Raises LeoLabsEphemerisError on inputs that cannot produce an honest file --
    an absent or degenerate covariance included, per this path's own rule that
    None means unknown rather than zero.
    """
    epoch = _parse_epoch(epoch_utc)
    r0, v0 = _state_vector(r_sat_km, v_sat_km_s)

    horizon_s = float(horizon_hours) * 3600.0
    step = float(step_s)
    if not np.isfinite(horizon_s) or horizon_s <= 0:
        raise LeoLabsEphemerisError("horizon_hours must be positive and finite")
    if not np.isfinite(step) or step <= 0:
        raise LeoLabsEphemerisError("step_s must be positive and finite")
    if step > horizon_s:
        raise LeoLabsEphemerisError(
            f"step_s ({step:g}s) exceeds the horizon ({horizon_s:g}s); the "
            f"ephemeris would be a single state"
        )

    # Seeded once, then grown per state. Built before any propagation so an
    # unusable covariance is refused before work is done.
    p0_km = seed_post_burn_covariance_km2(
        p_post_eci_km2,
        dv_eci_km_s if dv_eci_km_s is not None else [0.0, 0.0, 0.0],
        execution_error_magnitude_fraction,
        execution_error_pointing_sigma_rad,
    )
    velocity_seeded = bool(np.any(p0_km[3:, 3:]))
    if not velocity_seeded:
        log.info(
            "screening ephemeris has no execution-error velocity seed; the "
            "covariance will grow only from the position block",
            extra={"event": "leolabs_ephemeris_no_velocity_seed"},
        )

    # Inclusive of the horizon endpoint when the step divides it, so the file
    # really does span the window the screen looks across.
    n_steps = int(np.floor(horizon_s / step + 1e-9))
    offsets = [i * step for i in range(n_steps + 1)]

    # One integration sampled at the output cadence, rather than one propagation
    # per state: for an 865-state file that is one pass instead of 865.
    offsets_arr = np.asarray(offsets, dtype=float)
    positions, velocities = propagate_j2_series(r0, v0, offsets_arr)

    # Seven more passes for the state transition matrices, then one matrix
    # triple-product per state. Phi(0) is the identity, so the first state's
    # covariance is exactly the seed.
    phis = state_transition_matrices(r0, v0, offsets_arr)
    covariances_km = propagate_covariance(p0_km, phis)

    states: List[Dict[str, Any]] = []
    for index, dt_s in enumerate(offsets):
        # t = 0 is the post-burn state itself, returned unpropagated.
        r = r0 if dt_s == 0.0 else positions[index]
        v = v0 if dt_s == 0.0 else velocities[index]
        # One conversion for the whole matrix: every block scales by 1e6, since
        # position is km->m and velocity km/s->m/s are both x1e3.
        cov_m = covariances_km[index] * _KM2_TO_M2
        states.append({
            "timestamp": _iso_z(epoch + timedelta(seconds=dt_s)),
            "position": [float(c) * _KM_TO_M for c in r],
            "velocity": [float(c) * _KM_TO_M for c in v],
            "covariance": [[float(c) for c in row] for row in cov_m],
        })

    log.info(
        "built LeoLabs screening ephemeris",
        extra={"event": "leolabs_ephemeris_built", "states": len(states),
               "dynamics": "two_body_plus_j2",
               "covariance": "execution_error_seed_grown_by_stm",
               "velocity_seeded": velocity_seeded,
               "horizon_hours": float(horizon_hours), "step_s": step,
               "epoch_utc": _iso_z(epoch)},
    )
    return {
        "frame": FRAME,
        "covarianceFrame": FRAME,
        "states": states,
    }
