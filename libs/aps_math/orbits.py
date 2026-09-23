"""libs/aps_math/orbits.py

Two-body orbit propagation for visualization tracks (SCRUM-447).

This is the propagator the 3D globe draws its orbit rings with. It was written
for services/ui/app/main.py's /api/orbits and lived inside that request handler;
SCRUM-447 needed the same propagator on the planner side for the live-data globe,
so it moved here rather than being copied. Both services import this one copy:
the planner and the UI must not be able to draw the same orbit differently.

Scope, deliberately narrow
--------------------------
Two-body (Keplerian) only. No J2, no drag, no SRP. That is adequate for what it
is used for -- drawing one closed revolution as a visual ring -- and is not
adequate for anything that has to be right about where an object *is*. The
planner's real propagation and the conjunction geometry do not come from here;
they come from the CDM and the scorer. If you are tempted to use this for a
decision, use the propagator service instead.

Units are km and km/s throughout, matching the globe's scene units. Callers
holding metres (LeoLabs get_states returns metres) convert before calling.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

# Earth gravitational parameter, km^3/s^2.
MU_EARTH_KM3_S2 = 398600.4418

DEFAULT_STEPS = 90


def orbital_period_s(r0: Sequence[float], mu: float = MU_EARTH_KM3_S2) -> float:
    """Period of a circular orbit at radius |r0|, seconds.

    Uses |r0| as the semi-major axis, which is exact for a circular orbit and
    close enough for the near-circular LEO orbits the globe draws. Taking the
    period this way is what makes the track close on itself instead of leaving a
    visible gap or overlapping its own start.
    """
    a = float(np.linalg.norm(np.asarray(r0, dtype=float)))
    return 2.0 * np.pi * float(np.sqrt(a ** 3 / mu))


def propagate_two_body(
    r0: Sequence[float],
    v0: Sequence[float],
    n_steps: int = DEFAULT_STEPS,
    dt_s: Optional[float] = None,
    mu: float = MU_EARTH_KM3_S2,
) -> List[List[float]]:
    """Propagate one revolution from a state vector. Returns [[x,y,z], ...] km.

    RK4 over the two-body acceleration. With dt_s left as None the step is one
    computed orbital period divided by n_steps, so the returned track is exactly
    one closed revolution -- the property the globe relies on when it appends the
    first point to close the loop.

    Behaviour is unchanged from the implementation this replaced in
    services/ui/app/main.py; it was moved, not rewritten.
    """
    r = np.asarray(r0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    if r.shape != (3,) or v.shape != (3,):
        raise ValueError("r0 and v0 must each be three components, km and km/s")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
        raise ValueError("r0 and v0 must be finite")
    if float(np.linalg.norm(r)) <= 0.0:
        raise ValueError("r0 must have non-zero magnitude")

    if dt_s is None:
        dt_s = orbital_period_s(r, mu) / float(n_steps)

    def accel(rv: np.ndarray) -> np.ndarray:
        rmag = float(np.linalg.norm(rv))
        return -mu / rmag ** 3 * rv

    points: List[List[float]] = [r.tolist()]
    for _ in range(int(n_steps)):
        k1v = accel(r)
        k1r = v
        k2v = accel(r + 0.5 * dt_s * k1r)
        k2r = v + 0.5 * dt_s * k1v
        k3v = accel(r + 0.5 * dt_s * k2r)
        k3r = v + 0.5 * dt_s * k2v
        k4v = accel(r + dt_s * k3r)
        k4r = v + dt_s * k3v
        r = r + (dt_s / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + (dt_s / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        points.append(r.tolist())

    return points
