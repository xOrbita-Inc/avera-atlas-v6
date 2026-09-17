"""
SCRUM-382 -- operator approve and veto, translated into state-machine inputs.

MAF v2.0 section 9. SCRUM-379 already models the operator's decision as guard
inputs: approval_command_received / approval_accepted for L1, and
veto_command_received / veto_accepted for L2. This module does not re-model
either. It takes a contract-shaped GNCApprovalCommand or GNCVetoCommand,
records it against its conjunction so the next evaluation's GuardInputs carry
it, and builds the contract's ack.

A note on where these paths live. openapi/gnc_interface.yaml declares
/v1/gnc/approve and /v1/gnc/veto on the GNC server, called by ARBITER. The
planner serves them too, as the APS-side receiver, because an operator decision
has to reach the state machine that acts on it -- ARBITER telling GNC and APS
never hearing about it would leave the two disagreeing about whether a burn was
approved. The request and response shapes are the contract's, unchanged.

The store is process-local and deliberately small, the same posture as
mode_persistence's default: an explicit flag on the evaluate request still
wins, so nothing here becomes a hidden source of authority.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from common.decision_state_machine import FlightMode

APPROVE_MODE_HONOURED = "M3"
APPROVE_MODE_REFUSED = "M4"
VETO_MODE_RESTAGED = "M2"
VETO_MODE_SAFEHOLD = "M4"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class OperatorDecision:
    """One recorded approve or veto, keyed by conjunction."""

    kind: str                  # "approve" | "veto"
    command_id: str
    conjunction_id: str
    issued_by: str
    issued_at_utc: str
    accepted: bool
    reason: str = ""
    recorded_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "command_id": self.command_id,
            "conjunction_id": self.conjunction_id,
            "issued_by": self.issued_by,
            "issued_at_utc": self.issued_at_utc,
            "accepted": self.accepted,
            "reason": self.reason,
            "recorded_at_utc": self.recorded_at_utc,
        }


class OperatorCommandStore:
    """Process-local record of the latest operator decision per conjunction.

    Latest wins: an operator who approves and then vetoes has vetoed. Keeping a
    history here would invite a caller to reason about ordering that the state
    machine, which sees one evaluation at a time, cannot honour anyway.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._decisions: Dict[str, OperatorDecision] = {}

    def record(self, decision: OperatorDecision) -> None:
        with self._lock:
            self._decisions[decision.conjunction_id] = decision

    def latest(self, conjunction_id: str) -> Optional[OperatorDecision]:
        with self._lock:
            return self._decisions.get(conjunction_id)

    def clear(self, conjunction_id: str = "") -> None:
        with self._lock:
            if conjunction_id:
                self._decisions.pop(conjunction_id, None)
            else:
                self._decisions.clear()

    def guard_inputs(self, conjunction_id: str) -> Dict[str, Any]:
        """The SCRUM-379 guard-input fields this decision implies.

        Empty when nothing was recorded, so a caller can merge it without
        having to branch, and an absent decision never reads as a refusal.
        """
        decision = self.latest(conjunction_id)
        if decision is None:
            return {}
        if decision.kind == "approve":
            return {
                "approval_command_received": True,
                "approval_accepted": decision.accepted,
            }
        return {
            "veto_command_received": True,
            "veto_accepted": decision.accepted,
        }


def approval_ack(
    command: Dict[str, Any],
    *,
    current_mode: Optional[FlightMode],
    has_staged_command: bool,
) -> Dict[str, Any]:
    """Build a GNCApprovalAck for a received GNCApprovalCommand.

    The contract's enum admits only M3 or M4 for flight_mode_after, and its own
    description pairs a refused approval with M4 ("approval_accepted is false if
    the TCA window has closed or slew is no longer feasible -- GNC transitions
    to M4 in that case"). So an approval that cannot be honoured is reported as
    M4, which is also the fail-closed answer for an approval against a
    conjunction APS has no staged burn for.

    M3 here is the mode the approval drives, not a claim that the transition has
    already happened: the state machine decides that on the next evaluation,
    where every M2 guard is re-checked. An approval is permission, not a burn.
    """
    honoured = current_mode is FlightMode.M2_STAGED and has_staged_command
    ack: Dict[str, Any] = {
        "command_id": str(command.get("command_id", "")),
        "conjunction_id": str(command.get("conjunction_id", "")),
        "approval_accepted": honoured,
        "flight_mode_after": (
            APPROVE_MODE_HONOURED if honoured else APPROVE_MODE_REFUSED
        ),
        "ack_time_utc": _utc_now_iso(),
    }
    if not honoured:
        if current_mode is None:
            ack["rejection_note"] = (
                "no flight mode on record for this conjunction; nothing is staged "
                "to approve"
            )
        elif current_mode is not FlightMode.M2_STAGED:
            ack["rejection_note"] = (
                f"conjunction is in {current_mode.value}, not M2; there is no "
                "staged burn to approve"
            )
        else:
            ack["rejection_note"] = "no staged maneuver command on record"
    return ack


def veto_ack(
    command: Dict[str, Any],
    *,
    current_mode: Optional[FlightMode],
    window_closed: bool,
) -> Dict[str, Any]:
    """Build a GNCVetoAck for a received GNCVetoCommand.

    The contract: a veto after veto_window_close_utc has veto_accepted false and
    the burn proceeds, with the reason in late_veto_note.

    flight_mode_after is awkward here and worth flagging rather than papering
    over. The enum admits only M2 or M4, but a late veto means the burn is
    proceeding, which is M3 -- a mode the enum does not offer for this ack. M2 is
    returned as the mode at the instant of receipt, with late_veto_note carrying
    what actually happens. The contract is final so this conforms to it rather
    than working around it.
    """
    accepted = not window_closed and current_mode is FlightMode.M2_STAGED
    ack: Dict[str, Any] = {
        "command_id": str(command.get("command_id", "")),
        "conjunction_id": str(command.get("conjunction_id", "")),
        "veto_accepted": accepted,
        "flight_mode_after": (
            VETO_MODE_RESTAGED if accepted else _refused_veto_mode(current_mode)
        ),
        "ack_time_utc": _utc_now_iso(),
    }
    if not accepted:
        if window_closed:
            ack["late_veto_note"] = (
                "veto arrived after veto_window_close_utc; the burn proceeds"
            )
        elif current_mode is None:
            ack["late_veto_note"] = (
                "no flight mode on record for this conjunction; nothing is staged "
                "to veto"
            )
        else:
            ack["late_veto_note"] = (
                f"conjunction is in {current_mode.value}, not M2; there is no "
                "staged burn to veto"
            )
    return ack


def _refused_veto_mode(current_mode: Optional[FlightMode]) -> str:
    """M2 for a late veto (the burn proceeds), M4 when nothing is staged.

    Split out because the two refusals mean opposite things: a late veto is a
    burn going ahead, and a veto against nothing is an event that should not be
    left in an unknown state.
    """
    if current_mode is FlightMode.M2_STAGED:
        return VETO_MODE_RESTAGED
    return VETO_MODE_SAFEHOLD


def record_approval(
    store: OperatorCommandStore, command: Dict[str, Any], accepted: bool
) -> OperatorDecision:
    decision = OperatorDecision(
        kind="approve",
        command_id=str(command.get("command_id", "")),
        conjunction_id=str(command.get("conjunction_id", "")),
        issued_by=str(command.get("issued_by", "")),
        issued_at_utc=str(command.get("issued_at_utc", "")),
        accepted=accepted,
        recorded_at_utc=_utc_now_iso(),
    )
    store.record(decision)
    return decision


def record_veto(
    store: OperatorCommandStore, command: Dict[str, Any], accepted: bool
) -> OperatorDecision:
    decision = OperatorDecision(
        kind="veto",
        command_id=str(command.get("command_id", "")),
        conjunction_id=str(command.get("conjunction_id", "")),
        issued_by=str(command.get("issued_by", "")),
        issued_at_utc=str(command.get("issued_at_utc", "")),
        accepted=accepted,
        reason=str(command.get("reason", "")),
        recorded_at_utc=_utc_now_iso(),
    )
    store.record(decision)
    return decision
