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

from . import frames

__all__ = [
    "conjunction_plane_basis",
    "conjunction_plane_epsilon",
    "marginalize_information",
    "measurement_noise_covariance",
    "observability_gramian",
    "observability_gramian_epoch_term",
    "observation_jacobian",
]


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


def marginalize_information(
    W: np.ndarray,
    keep_idx: list[int],
    drop_idx: list[int],
    method: str = "schur",
) -> np.ndarray:
    """
    Marginal Fisher information for the subspace indexed by keep_idx,
    after eliminating the subspace indexed by drop_idx from a symmetric
    information matrix W.

    SCRUM-378. The observability Gramian's conjunction-plane projection
    needs to drop a subspace twice (velocity from the full 6-state, then
    the relative-velocity direction from the resulting 3D position
    information within the conjunction plane), and both are the same
    operation: this function, not two different ad hoc reductions.

    method="schur" (default, and the one to use for W): the correct
    reduction for an information matrix. Partitioning W into blocks over
    keep_idx ("a") and drop_idx ("b"),

        W_marginal = W_aa - W_ab @ inv(W_bb) @ W_ba

    This is standard: the Schur complement of the eliminated block in a
    partitioned precision matrix gives the marginal precision. Reduces
    to W_aa exactly when W_ab is zero (the kept and dropped subspaces
    are uncorrelated), and is otherwise strictly more conservative (the
    subtracted term is positive semi-definite, since it has the form
    X^T inv(W_bb) X with W_bb positive definite), reflecting that
    information partially explained by the correlation with the dropped
    subspace does not count as information about the kept subspace alone.

    method="naive": the raw W_aa sub-block, with no correction. This is
    the CORRECT reduction for a COVARIANCE (marginal covariance is
    always just the relevant sub-block, no Schur complement needed,
    e.g. pc_utils.py's own conjunction-plane projection, which operates
    on a covariance). It is generally WRONG for an information matrix
    unless the off-diagonal coupling happens to be exactly zero. Kept
    here as an explicit, one-line-switch alternative for comparison
    only, per review -- not the default, and not to be used for W
    without a specific reason to.

    Args:
        W: symmetric information matrix, shape (n, n).
        keep_idx: indices of the subspace to retain.
        drop_idx: indices of the subspace to eliminate. Must partition
            {0, ..., n-1} together with keep_idx: no overlap, nothing
            left out.
        method: "schur" or "naive".

    Returns:
        The marginal information matrix, shape (len(keep_idx), len(keep_idx)).

    Raises:
        ValueError if keep_idx/drop_idx don't partition W's indices
        exactly, if method is not "schur" or "naive", or if W_bb (the
        dropped-subspace block) is singular under method="schur" -- a
        singular W_bb means the dropped directions carry no information
        at all, which makes "the information explained by correlation
        with them" undefined, not zero.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    if W.shape != (n, n):
        raise ValueError(f"marginalize_information: W must be square, got shape {W.shape}")

    keep_idx = list(keep_idx)
    drop_idx = list(drop_idx)
    all_idx = sorted(keep_idx + drop_idx)
    if all_idx != list(range(n)):
        raise ValueError(
            f"marginalize_information: keep_idx and drop_idx must together "
            f"partition range({n}) exactly with no overlap and nothing left "
            f"out, got keep_idx={keep_idx}, drop_idx={drop_idx}"
        )

    W_aa = W[np.ix_(keep_idx, keep_idx)]

    if method == "naive":
        return W_aa
    elif method == "schur":
        W_ab = W[np.ix_(keep_idx, drop_idx)]
        W_ba = W[np.ix_(drop_idx, keep_idx)]
        W_bb = W[np.ix_(drop_idx, drop_idx)]
        try:
            W_bb_inv = np.linalg.inv(W_bb)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "marginalize_information: the dropped-subspace block W_bb "
                "is singular, so it carries no information and the "
                "correlation correction is undefined. This means the arc "
                "does not observe the dropped direction(s) at all -- a "
                "genuine finding, not a numerical edge case to paper over."
            ) from e
        return W_aa - W_ab @ W_bb_inv @ W_ba
    else:
        raise ValueError(
            f"marginalize_information: method must be 'schur' or 'naive', "
            f"got {method!r}"
        )



def observability_gramian_epoch_term(
    r_target_t0_km: np.ndarray,
    v_target_t0_km_s: np.ndarray,
    r_target_tau_km: np.ndarray,
    v_target_tau_km_s: np.ndarray,
    r_observer_tau_km: np.ndarray,
    a_km: float,
    dt_s: float,
    ra_sigma_rad: float,
    dec_sigma_rad: float,
    range_sigma_km: float | None = None,
) -> np.ndarray:
    """
    Single-epoch contribution to the observability Gramian: the 6x6 term
    H_eff(tau)^T R^-1 H_eff(tau), where H_eff maps an ECI state deviation
    at t0 directly to a measurement deviation at tau, correctly combining
    Phi (RTN-frame propagation) and H (ECI-frame measurement Jacobian).

    SCRUM-378. cw_phi_full propagates a state deviation within the
    rotating RTN frame from t0 to tau; observation_jacobian differentiates
    the ECI-frame observation model at tau. Naively multiplying H(tau) by
    Phi(tau,t0) mixes frames -- the same category of error SCRUM-409
    found in maneuver_scorer.py, just for a general state deviation
    rather than a pure burn. The correct combination converts the ECI
    deviation at t0 into RTN, propagates it, converts the result back to
    ECI at tau, then applies H:

        H_eff(tau) = H(tau) @ rtn_to_eci_state_transform(rot_tau, a) @
                     Phi(tau, t0) @ eci_to_rtn_state_transform(rot_t0, a)

    This is algebraically the same quantity the SCRUM-333 formula's
    Phi(tau,t0)^T H(tau)^T R^-1 H(tau) Phi(tau,t0) describes, correctly
    accounting for the fact that Phi and H are not naturally expressed in
    the same frame -- the formula's own notation implicitly assumes a
    single consistent frame throughout, which does not hold here.

    Verified against true two-body motion: H_eff's linear prediction of
    the measurement change from a real perturbed-vs-unperturbed
    propagation matches to ~1e-5 relative error at a 300s arc segment,
    consistent with the linearization error already present in Phi and
    the state transform individually. See tests/test_observability.py.

    The returned 6x6 term is individually rank-deficient (rank at most 2
    or 3, matching H's row count), since a single epoch cannot fully
    determine a 6-state -- this is expected, not a defect, and shows up
    as several eigenvalues near zero (floating-point noise around the
    true zero, positive or negative). Summing this term across multiple
    epochs in a tracking arc is what observability_gramian accumulates
    toward full rank.

    Args:
        r_target_t0_km, v_target_t0_km_s: reference target state at the
            arc's start epoch t0 (e.g. from an IOD solution).
        r_target_tau_km, v_target_tau_km_s: reference target state at
            this observation's epoch tau (e.g. from propagating the t0
            state forward, such as services/tracker/iod.py's
            kepler_propagate).
        r_observer_tau_km: observer ECI position at epoch tau.
        a_km: reference semi-major axis for the circular-orbit CW/state-
            transform assumption, consistent across the whole arc.
        dt_s: tau - t0, seconds. May be negative.
        ra_sigma_rad, dec_sigma_rad: this observation's angular
            measurement uncertainty.
        range_sigma_km: this observation's range uncertainty, or None
            for an angles-only observation (most observations from this
            system's own sensor; see observation_jacobian's docstring).

    Returns:
        6x6 symmetric positive semi-definite matrix, in ECI-state-
        deviation-at-t0 coordinates.
    """
    rot_t0 = frames.rtn_to_eci_rotation(r_target_t0_km, v_target_t0_km_s)
    rot_tau = frames.rtn_to_eci_rotation(r_target_tau_km, v_target_tau_km_s)

    phi_rtn = frames.cw_phi_full(a_km, dt_s)
    m_tau = frames.rtn_to_eci_state_transform(rot_tau, a_km)
    m0_inv = frames.eci_to_rtn_state_transform(rot_t0, a_km)

    h_tau = observation_jacobian(
        r_target_tau_km, r_observer_tau_km, include_range=range_sigma_km is not None
    )
    r_cov = measurement_noise_covariance(ra_sigma_rad, dec_sigma_rad, range_sigma_km)

    h_eff = h_tau @ m_tau @ phi_rtn @ m0_inv

    r_inv = np.linalg.inv(r_cov)
    return h_eff.T @ r_inv @ h_eff


def observability_gramian(epoch_terms: list) -> np.ndarray:
    """
    Sum single-epoch Gramian terms into the full-arc observability
    Gramian W.

    SCRUM-378. W = sum_i observability_gramian_epoch_term(...) over every
    observation in the tracking arc. A single epoch's term is individually
    rank-deficient (see observability_gramian_epoch_term); the sum across
    an arc with enough geometric diversity is what builds up full rank
    and makes the Gramian invertible/well-conditioned. That diversity, or
    the lack of it, is exactly what epsilon (the ratio of smallest to
    largest eigenvalue, after projecting onto the conjunction plane via
    marginalize_information) is meant to detect.

    Deliberately takes a plain list of already-computed 6x6 terms rather
    than the raw per-observation inputs: this function's only job is the
    summation, not looping over observations, fetching reference states,
    or deciding which observations belong in the arc. Keeping it this
    narrow means it needs no changes regardless of how observations get
    selected or how their reference states get propagated.

    Args:
        epoch_terms: list of 6x6 matrices, each from
            observability_gramian_epoch_term, all expressed relative to
            the SAME t0 (they must be, since each term already maps an
            ECI deviation at that shared t0 to a measurement deviation).

    Returns:
        6x6 symmetric positive semi-definite matrix, W.

    Raises:
        ValueError if epoch_terms is empty (a Gramian over zero
        observations is not zero information, it is undefined -- the
        arc has not been characterized at all) or if any term is not
        6x6.
    """
    if not epoch_terms:
        raise ValueError(
            "observability_gramian: epoch_terms is empty. A Gramian over "
            "zero observations is undefined, not zero -- there is no arc "
            "to characterize."
        )
    w = np.zeros((6, 6), dtype=float)
    for i, term in enumerate(epoch_terms):
        term = np.asarray(term, dtype=float)
        if term.shape != (6, 6):
            raise ValueError(
                f"observability_gramian: epoch_terms[{i}] has shape "
                f"{term.shape}, expected (6, 6)"
            )
        w += term
    return w


def conjunction_plane_basis(r_rel_km: np.ndarray, v_rel_km_s: np.ndarray) -> np.ndarray:
    """
    3x3 rotation whose columns are the conjunction-plane unit vectors
    [x_hat, y_hat, z_hat] expressed in ECI.

    SCRUM-378. Matches the encounter-frame convention already used by
    libs/aps_math/pc_utils.py's Pc calculation: y is the relative-
    velocity direction, z is the relative-motion orbit normal, x
    completes the frame. The conjunction plane itself is the x-z plane;
    y (relative velocity) is dropped when projecting onto it, since
    motion along the relative-velocity direction does not change
    whether the two objects collide, only when.

        y_hat = v_rel / |v_rel|
        z_hat = (r_rel x v_rel) / |r_rel x v_rel|
        x_hat = y_hat x z_hat

    Use it the same way rtn_to_eci_rotation is used: `eci = rot @ cp`,
    `cp = rot.T @ eci`.

    Args:
        r_rel_km: relative position (primary minus secondary), ECI, km.
        v_rel_km_s: relative velocity (primary minus secondary), ECI,
            km/s, at the same epoch as r_rel_km (TCA, for the validity
            gate's use).

    Returns:
        3x3 orthonormal matrix, columns [x_hat, y_hat, z_hat] in ECI.

    Raises:
        ValueError if the relative velocity is zero (no encounter frame
        is defined without relative motion) or if r_rel and v_rel are
        parallel (a degenerate geometry with no defined orbit normal --
        physically, a purely radial approach with no cross-track
        component at all).
    """
    r_rel = np.asarray(r_rel_km, dtype=float)
    v_rel = np.asarray(v_rel_km_s, dtype=float)

    v_norm = float(np.linalg.norm(v_rel))
    if v_norm < 1e-12:
        raise ValueError(
            "conjunction_plane_basis: relative velocity is zero, no "
            "encounter plane is defined without relative motion."
        )
    y_hat = v_rel / v_norm

    h = np.cross(r_rel, v_rel)
    h_norm = float(np.linalg.norm(h))
    if h_norm < 1e-12:
        raise ValueError(
            "conjunction_plane_basis: relative position and velocity "
            "are parallel, the encounter geometry is degenerate (no "
            "defined orbit normal)."
        )
    z_hat = h / h_norm

    x_hat = np.cross(y_hat, z_hat)

    return np.column_stack([x_hat, y_hat, z_hat])


def conjunction_plane_epsilon(
    w_position_eci: np.ndarray,
    r_rel_km: np.ndarray,
    v_rel_km_s: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Project a 3x3 ECI position information matrix onto the conjunction
    plane and compute epsilon = lambda_min(W_CP) / lambda_max(W_CP).

    SCRUM-378, MAF v2.0 Sec 7. Two steps, both using marginalize_
    information's Schur-complement default (per review: W is an
    information matrix, and the naive sub-block, correct for a
    covariance, is not correct here):

    1. Rotate w_position_eci into the conjunction-plane-aligned [x,y,z]
       basis (a similarity transform, valid for an information matrix
       exactly as for a covariance -- only the marginalization step
       needs the Schur complement, not this rotation).
    2. Marginalize out y (the relative-velocity direction), keeping
       [x, z]: the 2D conjunction plane.

    w_position_eci must already be information about the state AT THE
    SAME EPOCH r_rel_km/v_rel_km_s are given at -- TCA, for the validity
    gate. Obtain it by building the observability Gramian with t0=TCA
    (each observation's dt_s to observability_gramian_epoch_term will
    then be negative, since observations happen before TCA), then
    marginalizing out velocity via marginalize_information(W, keep_idx=
    [0,1,2], drop_idx=[3,4,5], method="schur"). Passing position
    information referenced to a different epoch (e.g. the arc's start)
    would silently mix two different epochs' geometry with no error --
    the shapes all still work, the answer is just wrong.

    Args:
        w_position_eci: 3x3 symmetric position information matrix, ECI,
            at the same epoch as r_rel_km/v_rel_km_s.
        r_rel_km, v_rel_km_s: relative position/velocity (primary minus
            secondary), ECI, at that same epoch.

    Returns:
        (epsilon, W_CP, weak_direction_eci): epsilon is
        lambda_min/lambda_max of the 2x2 conjunction-plane information
        matrix W_CP. weak_direction_eci is the 3D ECI unit vector
        (embedded back from the 2D x-z eigenvector, with y=0) for the
        eigenvector corresponding to the SMALLEST eigenvalue -- the
        poorly-observed direction, for weak_directions reporting.
    """
    w_pos = np.asarray(w_position_eci, dtype=float)
    if w_pos.shape != (3, 3):
        raise ValueError(
            f"conjunction_plane_epsilon: w_position_eci must be 3x3, "
            f"got shape {w_pos.shape}"
        )

    q = conjunction_plane_basis(r_rel_km, v_rel_km_s)
    w_xyz = q.T @ w_pos @ q

    w_cp = marginalize_information(w_xyz, keep_idx=[0, 2], drop_idx=[1], method="schur")

    eigvals, eigvecs = np.linalg.eigh(w_cp)  # ascending order
    if eigvals[0] <= 0:
        raise ValueError(
            f"conjunction_plane_epsilon: W_CP is not positive definite "
            f"(smallest eigenvalue {eigvals[0]:.3e}), epsilon is "
            f"undefined. This means the tracking arc provides no "
            f"information at all in some direction within the "
            f"conjunction plane."
        )
    epsilon = float(eigvals[0] / eigvals[-1])

    weak_xz = eigvecs[:, 0]  # 2D, [x-component, z-component]
    weak_xyz = np.array([weak_xz[0], 0.0, weak_xz[1]])  # embed into full xyz, y=0
    weak_direction_eci = q @ weak_xyz

    return epsilon, w_cp, weak_direction_eci
