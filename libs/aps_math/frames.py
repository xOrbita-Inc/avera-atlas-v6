"""Reference-frame transforms shared across AVERA-ATLAS services.

SCRUM-397.

Why this module exists
----------------------
Three copies of the RTN-to-ECI rotation existed in this repo, in
services/planner/common/udl_client.py, services/planner/server.py and
services/ingest/cdm_to_conjunction.py. All three were correct and all three
were used correctly, for covariance transforms.

The place that needed one most called none of them. maneuver_scorer.py applied
an RTN-ordered Clohessy-Wiltshire block directly to an ECI burn direction, with
nothing rotating between the two. On a 53 degree inclined orbit that reported
57 km of separation change from a cross-track burn that produces 105 m, because
the ECI cross-track unit vector has components on all three ECI axes and picks
up the along-track CW block, whose entry grows as -3nt.

Every scoring test placed the satellite at [a, 0, 0] moving along [0, v, 0],
which makes the rotation exactly the identity, so the suite was green.

A frame convention is a value two services must agree on, which is the third
bucket in ADR-010. So is the function that implements it.

RTN
---
R is radial, r_hat.
N is cross-track (normal), (r x v) / |r x v|.
T is along-track (tangential), N x R.

Also called RSW, or RIC with the axes reordered. The ordering here is R, T, N,
matching cw_phi_rv and the CDM covariance layout.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "MU_EARTH",
    "cw_phi_full",
    "rtn_to_eci_rotation",
    "rotate_cw_block",
]

# WGS84 Earth gravitational parameter, km^3/s^2.
#
# SCRUM-378: consolidated here from services/planner/avoid/decision_model.py,
# which defined its own copy used only inside cw_phi_rv. decision_model.py
# now imports this one rather than carrying a second definition; see
# cw_phi_full's docstring for the consolidation this is part of.
MU_EARTH: float = 398600.4418


def cw_phi_full(a_km: float, dt_s: float) -> np.ndarray:
    """
    Full 6x6 Clohessy-Wiltshire state transition matrix Phi(tau, t0) for a
    circular reference orbit. RTN ordering, state = [r_R, r_T, r_N, v_R,
    v_T, v_N].

    SCRUM-378. Built for the observability Gramian, which needs the state
    transition of the full 6-state, not just the delta-v-to-delta-r mapping
    a single block provides.

    services/planner/avoid/decision_model.py::cw_phi_rv is now a thin
    wrapper returning this function's rv block: one closed-form
    implementation, not two copies of safety-relevant numerics. Same
    consolidation as the frames unification in SCRUM-397 and pc_utils in
    SCRUM-388 -- an independent copy guarded by a test is still the
    two-copies failure this package exists to prevent.
    services/planner/tests/test_cw_phi_rv.py is unchanged apart from
    importing MU_EARTH from here instead of decision_model.py, and
    continues to guard the rv block's behaviour directly against an
    independent closed form.

    Closed form (Clohessy-Wiltshire, RTN ordering, n = orbital mean motion,
    omega = sqrt(mu / a^3)). Reference: Clohessy & Wiltshire (1960);
    Vallado, Fundamentals of Astrodynamics and Applications, Sec 6.7.

        Phi_rr = [[ 4 - 3cos(nt),      0,  0        ],
                  [ 6(sin(nt) - nt),   1,  0        ],
                  [ 0,                 0,  cos(nt)  ]]

        Phi_rv = [[  sin(nt)/n,        2(1 - cos(nt))/n,      0          ],
                  [ -2(1 - cos(nt))/n, (4 sin(nt) - 3nt)/n,   0          ],
                  [  0,                0,                     sin(nt)/n  ]]

        Phi_vr = [[  3n sin(nt),         0,  0            ],
                  [ -6n(1 - cos(nt)),    0,  0            ],
                  [  0,                  0,  -n sin(nt)   ]]

        Phi_vv = [[ cos(nt),     2 sin(nt),      0       ],
                  [ -2 sin(nt),  4 cos(nt) - 3,  0       ],
                  [ 0,           0,              cos(nt) ]]

    Returns the 6x6 matrix in RTN. As with any CW block, rotate to ECI once
    via rotate_cw_block rather than rotating pieces separately and
    recombining -- see that function's own docstring for why.
    """
    if a_km <= 0:
        raise ValueError("a_km must be > 0")
    omega = math.sqrt(MU_EARTH / (a_km ** 3))
    c = math.cos(omega * dt_s)
    s = math.sin(omega * dt_s)

    phi_rr = np.array(
        [
            [4.0 - 3.0 * c, 0.0, 0.0],
            [6.0 * (s - omega * dt_s), 1.0, 0.0],
            [0.0, 0.0, c],
        ],
        dtype=float,
    )

    phi_rv = np.array(
        [
            [s / omega, 2.0 * (1.0 - c) / omega, 0.0],
            [-2.0 * (1.0 - c) / omega, (4.0 * s - 3.0 * omega * dt_s) / omega, 0.0],
            [0.0, 0.0, s / omega],
        ],
        dtype=float,
    )

    phi_vr = np.array(
        [
            [3.0 * omega * s, 0.0, 0.0],
            [-6.0 * omega * (1.0 - c), 0.0, 0.0],
            [0.0, 0.0, -omega * s],
        ],
        dtype=float,
    )

    phi_vv = np.array(
        [
            [c, 2.0 * s, 0.0],
            [-2.0 * s, 4.0 * c - 3.0, 0.0],
            [0.0, 0.0, c],
        ],
        dtype=float,
    )

    phi = np.zeros((6, 6), dtype=float)
    phi[0:3, 0:3] = phi_rr
    phi[0:3, 3:6] = phi_rv
    phi[3:6, 0:3] = phi_vr
    phi[3:6, 3:6] = phi_vv
    return phi


def rtn_to_eci_rotation(r_km: np.ndarray, v_km_s: np.ndarray) -> np.ndarray:
    """3x3 rotation whose columns are the RTN unit vectors expressed in ECI.

    Use it as `eci = rot @ rtn` and `rtn = rot.T @ eci`. The matrix is
    orthonormal, so the transpose is the inverse.

    Returns the identity for a degenerate state vector, meaning a zero position
    or a velocity parallel to the position, where no orbital frame exists. The
    planner reaches that case through its outermost exception handler, which
    substitutes a zero state rather than failing the request, and the identity
    keeps the arithmetic defined. It is not a correct rotation and a caller that
    can tell the difference should say so. server.py logs when it happens.
    """
    r = np.asarray(r_km, dtype=float)
    v = np.asarray(v_km_s, dtype=float)

    r_norm = float(np.linalg.norm(r))
    if r_norm < 1e-9:
        return np.eye(3)

    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    if h_norm < 1e-9:
        return np.eye(3)

    r_hat = r / r_norm
    n_hat = h / h_norm
    t_hat = np.cross(n_hat, r_hat)
    return np.column_stack([r_hat, t_hat, n_hat])


def is_degenerate_state(r_km: np.ndarray, v_km_s: np.ndarray) -> bool:
    """Whether rtn_to_eci_rotation would fall back to the identity.

    Exists so a caller can log or reject rather than silently accepting an
    unrotated result. rtn_to_eci_rotation itself stays quiet because it lives in
    a library and has no business choosing a service's logging behaviour.
    """
    r = np.asarray(r_km, dtype=float)
    if float(np.linalg.norm(r)) < 1e-9:
        return True
    return float(np.linalg.norm(np.cross(r, np.asarray(v_km_s, dtype=float)))) < 1e-9


def rotate_cw_block(block_rtn: np.ndarray, rtn_to_eci: np.ndarray) -> np.ndarray:
    """Express an RTN-ordered linear map in ECI coordinates.

    A Clohessy-Wiltshire block maps an RTN vector to an RTN vector. To apply it
    to an ECI vector and get an ECI vector back, conjugate it by the rotation:

        M_eci = rot @ M_rtn @ rot.T

    so that `M_eci @ dv_eci` is exactly `rot @ (M_rtn @ (rot.T @ dv_eci))`.

    Doing it once per candidate set rather than rotating in and out around every
    multiplication is both cheaper and harder to half-apply, which is the
    failure this module exists to prevent. It also makes the covariance case
    fall out for free, since `M_eci @ Q_eci @ M_eci.T` is then correct with no
    further rotations.
    """
    rot = np.asarray(rtn_to_eci, dtype=float)
    return rot @ np.asarray(block_rtn, dtype=float) @ rot.T
