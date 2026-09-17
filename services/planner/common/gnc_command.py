"""
SCRUM-382 -- assemble a GNCCommand from an AuthorizedExecution.

MAF v2.0 section 9. SCRUM-379 decides and authorises; this module carries that
authorisation out to GNC as a command conforming to openapi/gnc_interface.yaml.

Two things it deliberately does not do. It does not re-plan the maneuver: the
delta-v is the scorer's, rotated but never recomputed. And it does not decide
anything: an AuthorizedExecution only exists because the state machine already
cleared every guard and floor, so there is no second opinion to form here.

The frame conversion is the one piece of real work. AuthorizedExecution carries
dv in ECI km/s; ManeuverSpec.dv_rtn_m_s is an RTN burn in m/s. The rotation
uses libs/aps_math.frames.rtn_to_eci_rotation -- the existing convention, which
is also what the contract's frame section states -- transposed, because that
matrix takes RTN to ECI and this direction is the inverse. The contract makes
GNC responsible for rotating back with the same r_sat_km / v_sat_km_s, so both
sides must be using the one definition; a second local implementation here
would be the way that quietly stops being true.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from aps_math import frames
from common.decision_state_machine import (
    APPROVAL_BASIS_L1_OPERATOR,
    APPROVAL_BASIS_L2_VETO_EXPIRED,
    T_MARGIN_S,
    T_SLEW_SETTLE_S,
    AuthorizedExecution,
)
from common.gnc_contract import (
    contract_covariance_source,
    contract_direction,
)

# The two bases section 3 defines for an M2 to M3 transition, and nothing else.
_KNOWN_APPROVAL_BASES = (APPROVAL_BASIS_L1_OPERATOR, APPROVAL_BASIS_L2_VETO_EXPIRED)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_command_id() -> str:
    """A fresh command_id.

    Random rather than derived from the conjunction: GNC acks, approvals, vetoes
    and the post-burn report all key on command_id, so two commands for the same
    conjunction -- a re-staged burn after a veto, say -- must not collide.
    """
    return f"gnc-cmd-{uuid.uuid4().hex[:16]}"


def eci_to_rtn_m_s(
    dv_eci_km_s: Sequence[float],
    r_sat_km: Sequence[float],
    v_sat_km_s: Sequence[float],
) -> list:
    """Rotate an ECI delta-v in km/s to an RTN delta-v in m/s.

    r_sat_km / v_sat_km_s are the state at BURN time, which is the state the
    contract tells GNC to use for the inverse rotation. Using any other epoch
    here would give GNC a vector it would then rotate back with a different
    basis, and the burn would come out in the wrong direction while every
    magnitude still looked right.
    """
    rot_rtn_to_eci = frames.rtn_to_eci_rotation(
        np.asarray(r_sat_km, dtype=float), np.asarray(v_sat_km_s, dtype=float)
    )
    dv_eci = np.asarray(dv_eci_km_s, dtype=float)
    # Orthonormal, so the transpose is the inverse. km/s to m/s at the end,
    # once, so the unit change is visible in one place.
    dv_rtn_km_s = rot_rtn_to_eci.T @ dv_eci
    return [float(component * 1000.0) for component in dv_rtn_km_s]


def may_emit_without_further_approval(authorized: AuthorizedExecution) -> bool:
    """Whether this authorisation can go to GNC as it stands.

    Both of section 3's M2 to M3 rows are already-approved states by the time an
    AuthorizedExecution exists: an L2 one because its veto window expired
    un-vetoed, an L1 one because an operator approved it. The gate is that an
    L1 command cannot exist without that approval, not that it needs a second.

    An unrecognised basis returns False. Emitting a burn whose lawfulness we
    cannot name would defeat the point of recording the basis at all.
    """
    return authorized.approval_basis in _KNOWN_APPROVAL_BASES


@dataclass(frozen=True)
class GNCCommandContext:
    """Everything the contract requires that AuthorizedExecution does not carry.

    SCRUM-379's payload is deliberately small and flat -- twelve fields, the
    burn and the authority context. The contract asks for more: the risk inputs
    behind the decision, the timing budget, the safe-action fallback, and the
    burn-time state needed for the frame rotation. Those are assembled by the
    caller, which has the scoring result and the guard inputs to hand.

    Required rather than defaulted, for the fields the contract requires. A
    defaulted m2_pre or data_age_s would put a plausible number in an audit
    record that no one measured.
    """

    approving_identity: str
    r_sat_km: Tuple[float, float, float]
    v_sat_km_s: Tuple[float, float, float]
    direction: str
    burn_duration_s: float
    epsilon_threshold: float
    pc_computed: float
    m2_pre: float
    data_age_s: float
    t_ca_utc: str
    latest_burn_utc: str
    veto_window_open_utc: str
    veto_window_close_utc: str
    safe_action_id: str
    safe_action_type: str

    command_id: str = ""
    command_issued_utc: str = ""
    # Optional contract fields, omitted from the command when not supplied
    # rather than sent as a zero.
    weak_directions: Tuple[str, ...] = ()
    phenomenologies_used: Tuple[str, ...] = ()
    pc_space_track: Optional[float] = None
    dv_return_m_s: Optional[float] = None
    safe_action_dv_m_s: Optional[float] = None
    execution_timeout_s: Optional[float] = None


def build_gnc_command(
    authorized: AuthorizedExecution, context: GNCCommandContext
) -> Dict[str, Any]:
    """Build a contract-conforming GNCCommand. Pure: no I/O, no clock beyond
    stamping command_issued_utc when the caller did not.

    Returns a plain dict rather than a model, because the contract is the schema
    and a second class definition here would be a copy of it that could drift.
    Tests validate the result against openapi/gnc_interface.yaml directly.
    """
    issued_at = context.command_issued_utc or _utc_now_iso()

    # Section 5 gives L1 no veto window, but TimingBudget requires both
    # instants. A zero-length window at the issue instant is the honest
    # encoding of "there is no window"; fabricating one L1 does not have would
    # tell GNC it may auto-execute at a time no one authorised.
    veto_open = context.veto_window_open_utc or issued_at
    veto_close = context.veto_window_close_utc or issued_at

    validity: Dict[str, Any] = {
        "status": authorized.validity_status,
        "epsilon": authorized.validity_epsilon,
        "epsilon_threshold": context.epsilon_threshold,
    }
    if context.weak_directions:
        validity["weak_directions"] = list(context.weak_directions)
    if context.phenomenologies_used:
        validity["phenomenologies_used"] = list(context.phenomenologies_used)

    risk: Dict[str, Any] = {
        "pc_computed": context.pc_computed,
        "m2_pre": context.m2_pre,
        "covariance_source": contract_covariance_source(authorized.covariance_source),
        "data_age_s": context.data_age_s,
    }
    if context.pc_space_track is not None:
        risk["pc_space_track"] = context.pc_space_track

    maneuver: Dict[str, Any] = {
        "direction": contract_direction(context.direction),
        "dv_rtn_m_s": eci_to_rtn_m_s(
            authorized.dv_eci_km_s, context.r_sat_km, context.v_sat_km_s
        ),
        # The scorer's magnitude, not recomputed from the rotated vector. They
        # agree to floating point, and quoting the scorer's number keeps the
        # command traceable to the decision that authorised it.
        "dv_magnitude_m_s": authorized.dv_magnitude_m_s,
        "burn_time_utc": authorized.t_burn_utc,
        "burn_duration_s": context.burn_duration_s,
    }
    if context.dv_return_m_s is not None:
        maneuver["dv_return_m_s"] = context.dv_return_m_s

    safe_action: Dict[str, Any] = {
        "safe_action_id": context.safe_action_id,
        "safe_action_type": context.safe_action_type,
    }
    if context.safe_action_dv_m_s is not None:
        safe_action["safe_action_dv_m_s"] = context.safe_action_dv_m_s

    timing: Dict[str, Any] = {
        "command_issued_utc": issued_at,
        "veto_window_open_utc": veto_open,
        "veto_window_close_utc": veto_close,
        "latest_burn_utc": context.latest_burn_utc,
        "t_ca_utc": context.t_ca_utc,
        # Reported, not decided. SCRUM-379 locks these; a change escalates to
        # Minh, and GNC needs to see the values the decision was made under.
        "t_slew_s": T_SLEW_SETTLE_S,
        "t_margin_s": T_MARGIN_S,
    }
    if context.execution_timeout_s is not None:
        timing["execution_timeout_s"] = context.execution_timeout_s

    return {
        "command_id": context.command_id or new_command_id(),
        "conjunction_id": authorized.conjunction_id,
        "authority_level": authorized.authority_level,
        "envelope_version": authorized.envelope_version,
        "approving_identity": context.approving_identity,
        "validity": validity,
        "risk": risk,
        "maneuver": maneuver,
        "safe_action": safe_action,
        "timing": timing,
    }
