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

import numpy as np

__all__ = ["rtn_to_eci_rotation", "rotate_cw_block"]


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
