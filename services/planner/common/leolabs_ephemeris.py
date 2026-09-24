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

def covariance_6x6_from_position_3x3(
    p_pos_km2: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Place a 3x3 ECI position covariance into a 6x6 position-velocity block.

    Position block (top-left 3x3) is P_post converted km^2 -> m^2. The velocity
    block and the position-velocity cross terms are zero.

    This is the flagged decision in SCRUM-440, and it is a floor rather than a
    model. We do not have a post-burn velocity covariance anywhere in this
    codebase -- post_burn_covariance_km2 returns a 3x3 position block -- so the
    choice is between emitting the uncertainty we actually have and inventing one
    we do not. Zeros say "we are not telling you about velocity uncertainty",
    which LeoLabs reads the same way it reads an absent one. That is honest; a
    plausible-looking fabricated velocity block would not be, and would quietly
    change the screening result.

    Two better options exist and are deliberately NOT built here:

      1. A real 6x6 from GNC, if the flight computer's OD ever supplies one.
         PostBurnState.p_post_km2 is read as a 3x3 today; a 6x6 there would flow
         straight through and this helper would become a fallback.
      2. Propagate P_post along the trajectory with the state transition matrix
         (aps_math.cw_phi_full) so the emitted covariance grows per state instead
         of being repeated. That is the physically right answer for a 72 h
         horizon, where position uncertainty genuinely does not stay constant.

    Both are follow-on work. This function exists as one small seam precisely so
    either can replace it without touching the builder, and so this decision is
    reviewable in one place. Raised for John or Sreejit to confirm before
    SCRUM-441 relies on it.
    """
    p = _validated_position_covariance(p_pos_km2)
    cov = np.zeros((6, 6), dtype=float)
    cov[:3, :3] = p * _KM2_TO_M2
    return [[float(c) for c in row] for row in cov]


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

    Every state carries the same covariance, which is the limitation documented on
    covariance_6x6_from_position_3x3: position uncertainty genuinely grows over a
    72 h horizon and this does not model that. It is the uncertainty we have,
    stated once per state because the format wants it per state.

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

    # Built once and shared: the covariance is constant across states by the
    # documented decision, so this also validates before any propagation happens.
    covariance = covariance_6x6_from_position_3x3(p_post_eci_km2)

    # Inclusive of the horizon endpoint when the step divides it, so the file
    # really does span the window the screen looks across.
    n_steps = int(np.floor(horizon_s / step + 1e-9))
    offsets = [i * step for i in range(n_steps + 1)]

    # One integration sampled at the output cadence, rather than one propagation
    # per state: for an 865-state file that is one pass instead of 865.
    positions, velocities = propagate_j2_series(r0, v0, np.asarray(offsets, dtype=float))

    states: List[Dict[str, Any]] = []
    for index, dt_s in enumerate(offsets):
        # t = 0 is the post-burn state itself, returned unpropagated.
        r = r0 if dt_s == 0.0 else positions[index]
        v = v0 if dt_s == 0.0 else velocities[index]
        states.append({
            "timestamp": _iso_z(epoch + timedelta(seconds=dt_s)),
            "position": [float(c) * _KM_TO_M for c in r],
            "velocity": [float(c) * _KM_TO_M for c in v],
            "covariance": covariance,
        })

    log.info(
        "built LeoLabs screening ephemeris",
        extra={"event": "leolabs_ephemeris_built", "states": len(states),
               "dynamics": "two_body_plus_j2",
               "horizon_hours": float(horizon_hours), "step_s": step,
               "epoch_utc": _iso_z(epoch)},
    )
    return {
        "frame": FRAME,
        "covarianceFrame": FRAME,
        "states": states,
    }
