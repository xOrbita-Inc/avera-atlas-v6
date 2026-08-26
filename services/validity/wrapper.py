"""
services/validity/wrapper.py -- thin wrapper service, SCRUM-378.

REWRITE per John (Slack, SCRUM-378 coupling decision):
"Go with option 2. I do not want services/validity propagating anything.
Have the caller hand it the already-propagated reference states...
The assessor's whole job is then to assemble the Gramian and call
build_validity_verdict, and it never imports from tracker or touches a
propagator... Pulling those into one shared aps_math propagator is
worth doing, but it is its own ticket and not part of 378."

This replaces the earlier draft, which imported kepler_propagate from
services/tracker/iod.py directly -- exactly the coupling John rejected.
This version does no propagation at all. Every position/velocity it
touches is supplied by the caller, already at the epoch it's needed at.

Also per John: GNC-command assembly is SCRUM-382's job, not this
ticket's -- this module produces a ValidityVerdict and nothing more.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aps_math import frames
from aps_math.observability import (
    ValidityVerdict,
    build_validity_verdict,
    marginalize_information,
    observability_gramian,
    observability_gramian_epoch_term,
)

# SCRUM-333 Sec 1.7: "phenomenologies_used reflects current data sources.
# For APS 3.0 on Space-Track / UDL data this will typically be ['TLE'].
# Expands to ['optical', 'RF', 'LIDAR'] as onboard sensing matures."
# Deployment-scope fact, not a tunable numeric convention (ADR-010's
# conventions.py bucket) -- lives here at the service boundary, not in
# aps_math or conventions.py. Update the day onboard sensing is
# integrated -- see Sec 1.7.
DEFAULT_PHENOMENOLOGIES_USED: list[str] = ["TLE"]


@dataclass(frozen=True)
class ObservationEpochState:
    """
    One observation's contribution to the Gramian, entirely pre-computed
    by the caller. No timestamps, no IODObservation, no propagation --
    per John, this module never touches a propagator.

    Attributes:
        r_target_km: target position at THIS observation's own epoch,
            ECI, already propagated by the caller.
        v_target_km_s: target velocity at this observation's own epoch,
            ECI, already propagated by the caller.
        dt_s: elapsed time from TCA to this observation's epoch
            (negative for every real observation, since they precede
            TCA -- expected, per observability_gramian_epoch_term's own
            docstring, not an error). Plain arithmetic on two known
            timestamps, not propagation -- the caller computes this
            however it likes; this module just consumes the number.
        r_observer_km: observer (sensor) position at this epoch, ECI.
        ra_sigma_rad, dec_sigma_rad: measurement noise, radians.
        range_sigma_km: measurement noise on range, or None if this
            observation type has no range measurement (angles-only).
    """

    r_target_km: np.ndarray
    v_target_km_s: np.ndarray
    dt_s: float
    r_observer_km: np.ndarray
    ra_sigma_rad: float
    dec_sigma_rad: float
    range_sigma_km: float | None = None


def build_validity_verdict_for_arc(
    r_target_tca_km: np.ndarray,
    v_target_tca_km_s: np.ndarray,
    a_km: float,
    observation_epochs: list[ObservationEpochState],
    r_rel_km_at_tca: np.ndarray,
    v_rel_km_s_at_tca: np.ndarray,
    epsilon_threshold: float,
    phenomenologies_used: list[str] | None = None,
) -> ValidityVerdict:
    """
    Assemble a ValidityVerdict for one conjunction from already-propagated
    arc states and the conjunction geometry at TCA.

    This function does no propagation and imports nothing from
    services/tracker. Every state it needs is a parameter, already at
    the epoch it's needed at. If the caller's states were derived by
    propagating from some other epoch, that happened before this
    function was called, and is the caller's responsibility.

    Args:
        r_target_tca_km, v_target_tca_km_s: the tracked object's (the
            IOD-solved secondary's) state AT TCA, ECI. This is t0 for
            the Gramian -- conjunction_plane_epsilon requires
            w_position_eci to be at the SAME epoch as r_rel_km/
            v_rel_km_s, so this must genuinely be the state at TCA, not
            at the IOD solution's own epoch.
        a_km: reference semi-major axis for the CW/state-transform
            assumption (frames.cw_phi_full and friends). Caller's
            responsibility to supply the right reference value.
        observation_epochs: one ObservationEpochState per observation in
            the arc, each already propagated to its own epoch by the
            caller.
        r_rel_km_at_tca, v_rel_km_s_at_tca: relative state (primary
            minus secondary) at TCA, ECI. Supplied by the caller.
        epsilon_threshold: per-regime EARNED floor (0.20 for LEO,
            locked per SCRUM-333). Passed through unchanged, not
            defaulted here -- same reasoning as build_validity_verdict
            itself.
        phenomenologies_used: sensor types for this arc. Defaults to
            DEFAULT_PHENOMENOLOGIES_USED (["TLE"]) if omitted.

    Returns:
        ValidityVerdict, unchanged from build_validity_verdict's own
        return value.

    Raises:
        ValueError: propagated from build_validity_verdict if
            phenomenologies_used contains an invalid entry.
    """
    if phenomenologies_used is None:
        phenomenologies_used = DEFAULT_PHENOMENOLOGIES_USED

    epoch_terms = []
    for obs in observation_epochs:
        term = observability_gramian_epoch_term(
            r_target_t0_km=r_target_tca_km,
            v_target_t0_km_s=v_target_tca_km_s,
            r_target_tau_km=obs.r_target_km,
            v_target_tau_km_s=obs.v_target_km_s,
            r_observer_tau_km=obs.r_observer_km,
            a_km=a_km,
            dt_s=obs.dt_s,
            ra_sigma_rad=obs.ra_sigma_rad,
            dec_sigma_rad=obs.dec_sigma_rad,
            range_sigma_km=obs.range_sigma_km,
        )
        epoch_terms.append(term)

    w_full_6x6 = observability_gramian(epoch_terms)

    # Marginalize out velocity (Schur complement -- W is an information
    # matrix, a naive sub-block would be wrong here; see
    # marginalize_information's own docstring).
    w_position_eci = marginalize_information(
        w_full_6x6, keep_idx=[0, 1, 2], drop_idx=[3, 4, 5], method="schur"
    )

    rot_rtn_to_eci_at_tca = frames.rtn_to_eci_rotation(
        r_target_tca_km, v_target_tca_km_s
    )

    return build_validity_verdict(
        w_position_eci=w_position_eci,
        r_rel_km=r_rel_km_at_tca,
        v_rel_km_s=v_rel_km_s_at_tca,
        rot_rtn_to_eci=rot_rtn_to_eci_at_tca,
        epsilon_threshold=epsilon_threshold,
        phenomenologies_used=phenomenologies_used,
    )


def validity_evidence_values(verdict: ValidityVerdict) -> dict:
    """
    Map a ValidityVerdict onto the MAF section 10 evidence-record
    catalogue's field names, ready to hand to server.py's
    _evidence_values() (or wherever a decision/transition record is
    eventually assembled).

    Per John (Slack, SCRUM-378): "Wire them in where the verdict is
    produced... the verdict, the routing logic, and its own
    evidence-record fields, all on the one PR." This function is that:
    it lives alongside build_validity_verdict_for_arc, in the same
    module that produces the verdict, rather than leaving the caller to
    reconstruct evidence-record field names from ValidityVerdict.to_dict()
    itself.

    Deliberately NOT the same shape as ValidityVerdict.to_dict(): the
    evidence catalogue's field names (services/planner/common/
    evidence_record.py) use "validity_status"/"validity_epsilon" with
    the "validity_" prefix, matching MINIMUM_TRANSITION_FIELDS' naming,
    whereas ValidityVerdict.to_dict() uses "status"/"epsilon" per the
    published GNC interface field names instead. The other three field
    names (weak_directions, epsilon_threshold, phenomenologies_used)
    are identical between the two, but the two fields that differ are
    exactly the kind of silent-mismatch risk this function exists to
    remove -- a caller passing verdict.to_dict() straight into
    EvidenceRecord.build()'s values dict would raise EvidenceFieldError
    (unknown field: "status", "epsilon"), not silently do the wrong
    thing, but it would fail at the wrong place rather than being
    correct by construction here.

    Returns a dict with exactly these five keys, suitable to merge into
    an existing `values` dict before calling build_decision_record() /
    build_transition_record() -- e.g.
        values.update(validity_evidence_values(verdict))
    """
    return {
        "validity_status": verdict.status.value,
        "validity_epsilon": verdict.epsilon,
        "weak_directions": list(verdict.weak_directions),
        "epsilon_threshold": verdict.epsilon_threshold,
        "phenomenologies_used": list(verdict.phenomenologies_used),
    }
