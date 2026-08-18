"""Observability-gate numerics shared across AVERA-ATLAS services.

SCRUM-378.

Why this module exists
-----------------------
The Validity Assessor's Fisher-information observability check (MAF v2.0
Sec 7, SCRUM-333) needs the measurement Jacobian H and, once assembled,
projects the observability Gramian onto the conjunction plane. This is
safety-relevant numerics in the same sense frames.py and pc_utils.py are:
a value or a formula two or more services must agree on belongs in
exactly one place, not duplicated and guarded by a test. See
libs/aps_math/__init__.py for that principle stated in full, and
frames.py's cw_phi_full for the same consolidation already done for the
CW state transition matrix this module's Gramian will use alongside H.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["measurement_noise_covariance", "observation_jacobian"]


def observation_jacobian(
    r_target_km: np.ndarray,
    r_observer_km: np.ndarray,
    include_range: bool,
) -> np.ndarray:
    """
    Measurement Jacobian H = d(measurement)/d(state) for an angle
    [+ range] observation. State is [r_target(3), v_target(3)] in ECI,
    km and km/s.

    Matches the forward observation model in
    services/tracker/transform.py::eci_direction_to_ra_dec exactly:
        ra  = atan2(u_y, u_x)
        dec = asin(u_z)
    where u is the observer-to-target line-of-sight unit vector. This
    Jacobian is consistent with what the tracker actually measures, not a
    generic textbook angle model that happens to look similar.

    Velocity columns are exactly zero, not approximated to zero. RA, Dec,
    and range at a single epoch are functions of the target's position
    only; none of them involve velocity. This is why one observation
    cannot see velocity at all, and why the observability Gramian needs
    the state transition matrix (frames.cw_phi_full) to propagate
    position sensitivity across a tracking arc before velocity becomes
    observable -- a single H alone never provides it.

    Derivation: chain rule through the unit line-of-sight vector u.

        d(u)/d(r_target)   = (I - u u^T) / range        [3x3]
        d(ra, dec)/d(u)     = [ -u_y/cos(dec)^2,  u_x/cos(dec)^2,  0 ]
                              [  0,                0,               1/cos(dec) ]
        d(range)/d(r_target) = u^T                       [1x3, if include_range]

    Verified against central finite differences across multiple
    geometries, including near-polar declinations (~90 deg) where the
    analytical formula stays exact but naive finite differences become
    numerically unreliable at small step sizes -- see
    tests/test_observability.py.

    Args:
        r_target_km: target ECI position at the observation epoch (km).
            This is the reference trajectory position, e.g. from
            services/tracker/iod.py's kepler_propagate applied to an IOD
            solution, not a raw state estimate.
        r_observer_km: observer ECI position at the same epoch (km).
        include_range: whether to include the range row. Most real
            observations from this system's own sensor (a passive
            optical camera; see services/tracker/README.md) will not
            have range, per ObservationRecord's optional range_m field
            (openapi/tracker.yaml).

    Returns:
        2x6 matrix (angles only) or 3x6 matrix (angles and range),
        columns ordered [r_x, r_y, r_z, v_x, v_y, v_z].

    Raises:
        ValueError if r_target_km and r_observer_km coincide (zero
        line-of-sight range) or the line of sight is within machine
        precision of a celestial pole, where d(RA)/d(u) is genuinely
        singular. A near-pole geometry that is merely poorly observed,
        not singular, is exactly what this Jacobian and the resulting
        Gramian should report as poor conditioning, not silently avoid.
    """
    r_target = np.asarray(r_target_km, dtype=float)
    r_observer = np.asarray(r_observer_km, dtype=float)

    los = r_target - r_observer
    rng = float(np.linalg.norm(los))
    if rng < 1e-9:
        raise ValueError(
            "observation_jacobian: target and observer positions "
            "coincide (zero line-of-sight range); RA/Dec/range are "
            "undefined there."
        )
    u = los / rng

    cos_dec_sq = max(1.0 - u[2] ** 2, 0.0)
    cos_dec = math.sqrt(cos_dec_sq)
    if cos_dec < 1e-9:
        raise ValueError(
            "observation_jacobian: line of sight is within machine "
            "precision of a celestial pole (dec ~ +/-90 deg), where "
            "d(RA)/d(u) is genuinely singular, not merely poorly "
            "conditioned. The observation itself is still well-defined; "
            "this Jacobian is not."
        )

    d_u_d_r = (np.eye(3) - np.outer(u, u)) / rng  # 3x3

    d_radec_d_u = np.array(
        [
            [-u[1] / cos_dec_sq, u[0] / cos_dec_sq, 0.0],
            [0.0, 0.0, 1.0 / cos_dec],
        ]
    )  # 2x3

    h_pos = d_radec_d_u @ d_u_d_r  # 2x3

    if include_range:
        d_range_d_r = u.reshape(1, 3)  # 1x3
        h_pos = np.vstack([h_pos, d_range_d_r])  # 3x3

    n_rows = h_pos.shape[0]
    return np.hstack([h_pos, np.zeros((n_rows, 3))])


def measurement_noise_covariance(
    ra_sigma_rad: float,
    dec_sigma_rad: float,
    range_sigma_km: float | None = None,
) -> np.ndarray:
    """
    Measurement noise covariance R for a single angle [+ range]
    observation, matching observation_jacobian's row order and units
    exactly: [ra (rad), dec (rad), (range (km))].

    Diagonal, since the only uncertainty this system's observations carry
    per-axis (ObservationRecord's ra_sigma_rad, dec_sigma_rad,
    range_sigma_m) has no recorded cross-axis correlation to build an
    off-diagonal term from. A diagonal R is the assumption the available
    data actually supports, not a simplification chosen over a richer
    one that was on the table.

    range_sigma_km, not range_sigma_m: observation_jacobian's range row
    is d(range_km)/d(r_km), a km-based state, so R must be expressed in
    the same km units to be dimensionally consistent when combined with
    H in the Gramian. Convert range_sigma_m from the published contract
    by /1000.0 before calling this, the same conversion already used at
    services/tracker/main.py's /v1/iod endpoint
    (range_sigma_km=rec.range_sigma_m / 1000.0) and carried on
    IODObservation.range_sigma_km.

    Returns a 2x2 matrix (angles only) or 3x3 matrix (angles and range),
    matching whichever observation_jacobian call this feeds.

    Raises ValueError if any sigma is not strictly positive: zero implies
    infinite certainty, which makes R singular and R^-1 in the Gramian
    formula undefined, and negative is not physically meaningful.
    """
    if ra_sigma_rad <= 0.0 or not math.isfinite(ra_sigma_rad):
        raise ValueError(
            f"measurement_noise_covariance: ra_sigma_rad must be finite "
            f"and > 0, got {ra_sigma_rad}."
        )
    if dec_sigma_rad <= 0.0 or not math.isfinite(dec_sigma_rad):
        raise ValueError(
            f"measurement_noise_covariance: dec_sigma_rad must be finite "
            f"and > 0, got {dec_sigma_rad}."
        )

    diag = [ra_sigma_rad ** 2, dec_sigma_rad ** 2]

    if range_sigma_km is not None:
        if range_sigma_km <= 0.0 or not math.isfinite(range_sigma_km):
            raise ValueError(
                f"measurement_noise_covariance: range_sigma_km must be "
                f"finite and > 0 when provided, got {range_sigma_km}."
            )
        diag.append(range_sigma_km ** 2)

    return np.diag(diag)
