"""
SCRUM-379 -- the MAF v2.0 section 8 safety monitor.

One pure function, evaluate_safety_monitor, over the fully populated
GuardInputs bundle. It returns the mode the spacecraft should be in and a
transition record explaining why. No I/O, no clock reads, no service calls: the
caller assembles the bundle (server.py does this in /v1/evaluate), the monitor
decides. That is what makes every guard in docs/scrum-333/state_machine_guards.md
testable without standing up a service.

Two rules run through the whole module.

Fail closed. A guard whose input is missing fails; it never passes on a default.
The one place absence is treated differently from a negative verdict is the
M1 to M4 row -- see _m1_escalation_guards for why, and it is the interpretive
call in this module most worth reviewing.

Escalate by default. Every mode change goes through resolve_transition, so a
pair section 3 does not define lands in M4 rather than wherever it was headed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.decision_state_machine import (
    APPROVAL_BASIS_L1_OPERATOR,
    APPROVAL_BASIS_L2_VETO_EXPIRED,
    AUTONOMOUS_AUTHORITY_LEVELS,
    COMMS_GAP_THRESHOLD_S,
    CONSECUTIVE_BELOW_MONITOR_TO_CLEAR,
    DATA_FRESHNESS_BOUND_S,
    REAL_CDM_COVARIANCE_SOURCE,
    AuthorizedExecution,
    FlightMode,
    GuardInputs,
    GuardResult,
    TransitionDecision,
    ValidityRouting,
    as_mode,
    first_failure,
    hold,
    resolve_transition,
    slew_lead_time_s,
)
from common.safety_floors import (
    ABORT_TRIGGER,
    BASELINE_TRIGGER,
    L3_TRIGGER,
    RECORD_QUALITY_TRIGGER,
    l3_refusal_reason,
)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _all_passed(results: Sequence[GuardResult]) -> bool:
    return all(r.passed for r in results)


# ---------------------------------------------------------------------------
# Individual guards. One function per row or clause in sections 3 and 4.
# ---------------------------------------------------------------------------


def guard_pc_above_monitor(inputs: GuardInputs) -> GuardResult:
    """Section 3, M0 to M1: Pc >= pc_monitor_threshold.

    Pc is the resolved value from ManeuverScoringResult.pc_pre, and its
    pc_source travels with it into the record.
    """
    name = "pc_above_monitor_threshold"
    values = {
        "pc": inputs.pc,
        "pc_source": inputs.pc_source,
        "pc_monitor_threshold": inputs.pc_monitor_threshold,
    }
    if inputs.pc is None or inputs.pc_monitor_threshold is None:
        return GuardResult(
            name, False, "no resolved Pc or no monitor threshold", values
        )
    if inputs.pc >= inputs.pc_monitor_threshold:
        return GuardResult(
            name, True,
            f"Pc {inputs.pc:.6g} >= monitor line {inputs.pc_monitor_threshold:.6g}",
            values,
        )
    return GuardResult(
        name, False,
        f"Pc {inputs.pc:.6g} < monitor line {inputs.pc_monitor_threshold:.6g}",
        values,
    )


def guard_pc_below_monitor_sustained(inputs: GuardInputs) -> GuardResult:
    """Section 3, M1 to M0: Pc below the monitor line for 2 consecutive
    evaluations. One quiet evaluation is not enough."""
    name = "pc_below_monitor_two_evaluations"
    values = {
        "pc": inputs.pc,
        "pc_monitor_threshold": inputs.pc_monitor_threshold,
        "consecutive_below_monitor": inputs.consecutive_below_monitor,
        "consecutive_required": CONSECUTIVE_BELOW_MONITOR_TO_CLEAR,
    }
    if inputs.pc is None or inputs.pc_monitor_threshold is None:
        return GuardResult(
            name, False, "no resolved Pc or no monitor threshold", values
        )
    if inputs.pc >= inputs.pc_monitor_threshold:
        return GuardResult(name, False, "Pc is still at or above the monitor line", values)
    if inputs.consecutive_below_monitor < CONSECUTIVE_BELOW_MONITOR_TO_CLEAR:
        return GuardResult(
            name, False,
            f"only {inputs.consecutive_below_monitor} consecutive evaluation(s) "
            f"below the monitor line, {CONSECUTIVE_BELOW_MONITOR_TO_CLEAR} required",
            values,
        )
    return GuardResult(
        name, True,
        f"Pc below the monitor line for {inputs.consecutive_below_monitor} "
        "consecutive evaluations",
        values,
    )


def guard_pc_above_action(inputs: GuardInputs) -> GuardResult:
    """Section 3, M1 to M2: Pc >= pc_maneuver_threshold."""
    name = "pc_above_maneuver_threshold"
    values = {
        "pc": inputs.pc,
        "pc_source": inputs.pc_source,
        "pc_maneuver_threshold": inputs.pc_maneuver_threshold,
    }
    if inputs.pc is None or inputs.pc_maneuver_threshold is None:
        return GuardResult(
            name, False, "no resolved Pc or no action threshold", values
        )
    if inputs.pc >= inputs.pc_maneuver_threshold:
        return GuardResult(
            name, True,
            f"Pc {inputs.pc:.6g} >= action line {inputs.pc_maneuver_threshold:.6g}",
            values,
        )
    return GuardResult(
        name, False,
        f"Pc {inputs.pc:.6g} < action line {inputs.pc_maneuver_threshold:.6g}",
        values,
    )


def guard_iod_confident(inputs: GuardInputs) -> GuardResult:
    """Section 3: IOD confidence CONFIDENT.

    The single boolean is IODSolution.proceeds_to_validity, which is already
    'success and verdict == CONFIDENT'. Reading the verdict enum separately
    here would let a failed solve with a stale CONFIDENT verdict through.
    """
    name = "iod_confident"
    values = {
        "iod_proceeds_to_validity": inputs.iod_proceeds_to_validity,
        "iod_confidence_verdict": inputs.iod_confidence_verdict,
    }
    if inputs.iod_proceeds_to_validity is None:
        return GuardResult(name, False, "IOD confidence gate was not evaluated", values)
    if inputs.iod_proceeds_to_validity:
        return GuardResult(name, True, "IOD gate proceeds to validity", values)
    return GuardResult(
        name, False,
        f"IOD confidence gate blocks autonomous action "
        f"({inputs.iod_confidence_verdict or 'verdict not recorded'})",
        values,
    )


def guard_validity_earned(inputs: GuardInputs) -> GuardResult:
    """Section 3 and 4.4: validity EARNED, and cleared for autonomous action.

    AUTONOMOUS clears the guard. OPERATOR_REVIEW and BLOCKED do not: BLOCKED is
    NOT_EARNED, and OPERATOR_REVIEW is EARNED but explicitly not autonomous, so
    neither may stage a burn the machine would then execute on its own.

    Section 6.3: an unavailable validity assessor is NOT_EARNED, not unknown.
    That check comes first, so an outage cannot be masked by a stale routing
    value left in the bundle.
    """
    name = "validity_earned"
    values = {
        "validity_routing": (
            inputs.validity_routing.value if inputs.validity_routing else None
        ),
        "validity_service_available": inputs.validity_service_available,
        "validity_status": inputs.validity_evidence.get("validity_status"),
        "validity_epsilon": inputs.validity_evidence.get("validity_epsilon"),
        "epsilon_threshold": inputs.validity_evidence.get("epsilon_threshold"),
    }
    if not inputs.validity_service_available:
        return GuardResult(
            name, False,
            "validity assessor unavailable; section 6.3 treats this as NOT_EARNED",
            values,
        )
    if inputs.validity_routing is None:
        return GuardResult(name, False, "validity gate was not evaluated", values)
    if inputs.validity_routing is ValidityRouting.AUTONOMOUS:
        return GuardResult(name, True, "validity EARNED and cleared as AUTONOMOUS", values)
    if inputs.validity_routing is ValidityRouting.OPERATOR_REVIEW:
        return GuardResult(
            name, False,
            "validity EARNED but routed to OPERATOR_REVIEW; not cleared for "
            "autonomous action",
            values,
        )
    return GuardResult(name, False, "validity NOT_EARNED (routing BLOCKED)", values)


def guard_secondary_clear(inputs: GuardInputs) -> GuardResult:
    """Section 4.2: SecondaryConflictCheck.secondary_conjunction_clear is true.

    'If the check was not performed, treat as NOT CLEAR and escalate.' Absence
    fails here, exactly as the SCRUM-381 gate already fails closed.
    """
    name = "secondary_conflict_clear"
    values = {
        "secondary_check_performed": inputs.secondary_check_performed,
        "secondary_conjunction_clear": inputs.secondary_conjunction_clear,
    }
    if not inputs.secondary_check_performed:
        return GuardResult(
            name, False,
            "secondary conflict check was not performed; section 4.2 treats "
            "this as NOT CLEAR",
            values,
        )
    if inputs.secondary_conjunction_clear:
        return GuardResult(name, True, "secondary conjunction screen is clear", values)
    return GuardResult(name, False, "secondary conjunction conflict is not clear", values)


def guard_slew_feasible(inputs: GuardInputs) -> GuardResult:
    """Section 4.3: t_now <= t_burn - (t_slew + t_settle + t_margin),
    and t_burn <= latest_burn_utc.

    Checked at M1 to M2 and continuously in M2.
    """
    name = "slew_feasible"
    lead_s = slew_lead_time_s()
    command = inputs.command
    values: Dict[str, Any] = {
        "t_now_utc": _iso(inputs.t_now_utc),
        "t_burn_utc": _iso(command.t_burn_utc) if command else None,
        "latest_burn_utc": (
            _iso(inputs.latest_burn_utc) if inputs.latest_burn_utc else None
        ),
        "slew_lead_time_s": lead_s,
    }
    if command is None:
        return GuardResult(name, False, "no burn to check slew feasibility for", values)

    slack_s = (command.t_burn_utc - inputs.t_now_utc).total_seconds() - lead_s
    values["slack_s"] = slack_s

    if inputs.latest_burn_utc is not None and command.t_burn_utc > inputs.latest_burn_utc:
        return GuardResult(
            name, False,
            "t_burn is after latest_burn_utc; the burn window has closed",
            values,
        )
    if slack_s < 0.0:
        return GuardResult(
            name, False,
            f"t_now is {-slack_s:.1f} s past the latest slew start "
            f"(lead time {lead_s:.0f} s)",
            values,
        )
    return GuardResult(
        name, True,
        f"{slack_s:.1f} s of slack beyond the {lead_s:.0f} s slew lead time",
        values,
    )


def guard_authority_baseline_validated(inputs: GuardInputs) -> GuardResult:
    """SCRUM-380, MAF v2.0 section 6: the running configuration is the one
    ground validated.

    SCRUM-379's envelope guard checks the envelope is cryptographically valid
    and unexpired. It does not notice that the envelope, or the monitor software
    itself, has CHANGED since ground last signed off. This is that floor. A
    drift clamps authority to L0 and holds there until an explicit ground
    re-validate, which is what unblocks SCRUM-384's tamper case.
    """
    name = "authority_baseline_validated"
    reason = inputs.authority_demotion()
    baseline = inputs.ground_validated_baseline
    values = {
        "baseline_enforced": baseline is not None,
        "baseline_envelope_version": baseline.envelope_version if baseline else None,
        "running_envelope_version": inputs.envelope_version,
        "baseline_validated_by": baseline.validated_by if baseline else None,
        "authority_granted": inputs.authority() or None,
        "authority_effective": inputs.effective_authority() or None,
    }
    if not reason:
        return GuardResult(
            name, True,
            "no ground-validated baseline on record; floor not armed"
            if baseline is None
            else "running configuration matches the ground-validated baseline",
            values,
        )
    return GuardResult(
        name, False,
        f"{BASELINE_TRIGGER}: {reason}; authority clamped to "
        f"{inputs.effective_authority()} until ground re-validates",
        values,
    )


def guard_authority_l1_or_l2(inputs: GuardInputs) -> GuardResult:
    """Section 3: authority L1 or L2. L0 is advisory only (section 5).

    Reads the EFFECTIVE authority, so a SCRUM-380 clamp to L0 blocks staging
    here without this guard needing to know why it was clamped.
    """
    name = "authority_l1_or_l2"
    level = inputs.effective_authority()
    values = {
        "authority_level": level or None,
        "authority_granted": inputs.authority() or None,
    }
    if level in AUTONOMOUS_AUTHORITY_LEVELS:
        return GuardResult(name, True, f"authority {level} may stage a maneuver", values)
    granted = inputs.authority()
    if granted and granted != level:
        return GuardResult(
            name, False,
            f"authority {granted} was granted but is clamped to {level}",
            values,
        )
    return GuardResult(
        name, False,
        f"authority {level or 'none granted'} is not L1 or L2",
        values,
    )


def guard_envelope_satisfied(inputs: GuardInputs) -> GuardResult:
    """Section 4.1: the compiled envelope permits this burn.

    Checks, in the order section 4.1 lists them and stopping at the first
    failure: the envelope compiled and activated at all, it is unexpired, the
    dv is within the cap for the granted level, data_age_s is within the
    freshness bound, and v_remaining exceeds dv plus the reserve.

    Two clauses in section 4.1 are enforced elsewhere on purpose.
    'Authority level matches the operator grant' is guard_authority_l1_or_l2,
    so a failure names the authority rather than the envelope. covariance_source
    is guard_covariance_provenance, because section 4.1 scopes its consequence
    to an L2 auto-execute -- a surrogate covariance does not stop an operator
    approving the same burn by hand at L1.
    """
    name = "envelope_satisfied"
    command = inputs.command
    values: Dict[str, Any] = {
        "authority_level": inputs.authority() or None,
        "envelope_version": inputs.envelope_version,
        "dv_magnitude_m_s": command.dv_magnitude_m_s if command else None,
        "dv_cap_m_s": inputs.dv_cap_m_s,
        "data_age_s": inputs.data_age_s,
        "data_freshness_bound_s": inputs.data_freshness_bound_s,
        "v_remaining_m_s": inputs.v_remaining_m_s,
        "v_reserved_m_s": inputs.v_reserved_m_s,
    }

    if inputs.envelope_error:
        return GuardResult(
            name, False, f"envelope not usable: {inputs.envelope_error}", values
        )
    if not inputs.envelope_version:
        return GuardResult(name, False, "no active authorization envelope", values)
    if inputs.envelope_expired:
        return GuardResult(name, False, "authorization envelope has expired", values)
    if command is None:
        return GuardResult(name, False, "no burn to check against the envelope", values)
    if inputs.dv_cap_m_s is None:
        return GuardResult(
            name, False, "no dv cap for the granted authority level", values
        )
    if command.dv_magnitude_m_s > inputs.dv_cap_m_s:
        return GuardResult(
            name, False,
            f"dv {command.dv_magnitude_m_s:.4g} m/s exceeds the "
            f"{inputs.dv_cap_m_s:.4g} m/s cap",
            values,
        )
    if inputs.data_age_s is None:
        return GuardResult(name, False, "data_age_s is unknown", values)
    if inputs.data_age_s > inputs.data_freshness_bound_s:
        return GuardResult(
            name, False,
            f"data_age_s {inputs.data_age_s:.0f} s exceeds the freshness bound "
            f"{inputs.data_freshness_bound_s:.0f} s",
            values,
        )
    if inputs.v_remaining_m_s is None or inputs.v_reserved_m_s is None:
        return GuardResult(name, False, "propellant budget is unknown", values)
    required = command.dv_magnitude_m_s + inputs.v_reserved_m_s
    values["v_required_m_s"] = required
    if inputs.v_remaining_m_s <= required:
        return GuardResult(
            name, False,
            f"v_remaining {inputs.v_remaining_m_s:.4g} m/s does not exceed dv plus "
            f"reserve {required:.4g} m/s",
            values,
        )
    return GuardResult(name, True, "authorization envelope satisfied", values)


def guard_covariance_provenance(inputs: GuardInputs) -> GuardResult:
    """Section 4.1: covariance_source is real_cdm for an L2 auto-execute.

    'surrogate_identity triggers a warning and requires explicit operator
    acknowledgement before L2 auto-execute.' So a surrogate covariance is not
    fatal, but the acknowledgement has to exist; a warning alone is not one.

    This is the guard that makes a live LeoLabs event eligible for L2
    auto-execute and a surrogate one not: the LeoLabs path sets
    covariance_source to real_cdm.
    """
    name = "covariance_source_real_cdm"
    values = {
        "covariance_source": inputs.covariance_source or None,
        "operator_ack_surrogate_covariance": inputs.operator_ack_surrogate_covariance,
    }
    if inputs.covariance_source == REAL_CDM_COVARIANCE_SOURCE:
        return GuardResult(name, True, "covariance_source is real_cdm", values)
    if inputs.operator_ack_surrogate_covariance:
        return GuardResult(
            name, True,
            f"covariance_source {inputs.covariance_source or 'unknown'} was "
            "explicitly acknowledged by the operator",
            values,
        )
    return GuardResult(
        name, False,
        f"covariance_source {inputs.covariance_source or 'unknown'} is not "
        "real_cdm and carries no operator acknowledgement",
        values,
    )


def guard_tca_above_floor(inputs: GuardInputs) -> GuardResult:
    """Section 3, the M1 to M4 row: TCA < min_hours_before_tca (4.0 hrs) with
    Pc still above the monitor threshold.

    Written the positive way -- it passes while there is still time -- so it
    reads the same as every other guard. M1 to M4 fires when it fails.
    """
    name = "tca_above_min_hours"
    hours = inputs.hours_to_tca()
    values = {
        "hours_to_tca": hours,
        "min_hours_before_tca": inputs.min_hours_before_tca,
        "pc": inputs.pc,
        "pc_monitor_threshold": inputs.pc_monitor_threshold,
    }
    if hours is None:
        return GuardResult(name, True, "no TCA supplied; the floor cannot bite", values)
    pc_above_monitor = (
        inputs.pc is not None
        and inputs.pc_monitor_threshold is not None
        and inputs.pc >= inputs.pc_monitor_threshold
    )
    if hours < inputs.min_hours_before_tca and pc_above_monitor:
        return GuardResult(
            name, False,
            f"TCA is {hours:.2f} h away, inside the {inputs.min_hours_before_tca:.1f} h "
            "floor, with Pc still above the monitor line",
            values,
        )
    return GuardResult(name, True, f"TCA is {hours:.2f} h away", values)


def guard_last_cdm_still_fresh(inputs: GuardInputs) -> GuardResult:
    """Section 6.3, the ingest row: 'Ingest service unavailable (no fresh CDM) |
    M1 | Continue monitoring with last CDM if data_age_s is within freshness
    bound. If stale, go to M4.'

    Only meaningful while ingest is down. With ingest up, a data_age_s beyond
    the bound is the envelope guard's business (section 4.1), which blocks
    staging rather than forcing a safehold.
    """
    name = "last_cdm_still_fresh"
    values = {
        "ingest_available": inputs.ingest_available,
        "data_age_s": inputs.data_age_s,
        "data_freshness_bound_s": inputs.data_freshness_bound_s,
    }
    if inputs.ingest_available:
        return GuardResult(name, True, "ingest is available", values)
    if inputs.data_age_s is None:
        return GuardResult(
            name, False, "ingest is unavailable and the CDM's age is unknown", values
        )
    if inputs.data_age_s > inputs.data_freshness_bound_s:
        return GuardResult(
            name, False,
            f"ingest is unavailable and the last CDM is {inputs.data_age_s:.0f} s "
            f"old, beyond the {inputs.data_freshness_bound_s:.0f} s bound",
            values,
        )
    return GuardResult(
        name, True, "ingest is unavailable but the last CDM is still fresh", values
    )


def guard_cdm_record_usable(inputs: GuardInputs) -> GuardResult:
    """SCRUM-380, MAF v2.0 section 6: degraded or zero-filled records are
    rejected, not parsed.

    The freshness guard asks how old the record is. This asks whether there is a
    record at all. A zero-filled covariance does not mean "no uncertainty", it
    means the record carries no covariance, and scoring it yields a
    confident-looking Pc computed from nothing -- worse than having no record,
    because it looks like an answer.

    Deliberately not the artifact's covariance_quality, which reads 'degraded'
    or 'dilution_region' for a real covariance in an awkward geometry. Those are
    legitimate, scorable conjunctions.
    """
    name = "cdm_record_usable"
    reason = inputs.cdm_record_rejected_reason
    values = {"cdm_record_rejected_reason": reason or None}
    if not reason:
        return GuardResult(name, True, "CDM record is structurally usable", values)
    return GuardResult(name, False, f"{RECORD_QUALITY_TRIGGER}: {reason}", values)


def guard_l3_pre_verified_action(inputs: GuardInputs) -> GuardResult:
    """SCRUM-380: an L3 authorization is refused without a pre-verified safe
    action.

    Guard doc section 5 gates L3 to a later phase. The only part of L3 that
    exists today is this refusal; nothing here executes an L3 maneuver, and a
    passing check is not permission to -- it only means the floor did not fire.
    """
    name = "l3_pre_verified_safe_action"
    reason = l3_refusal_reason(inputs.authority(), inputs.pre_verified_safe_action)
    values = {
        "requested_authority": inputs.authority() or None,
        "pre_verified_safe_action": inputs.pre_verified_safe_action or None,
    }
    if not reason:
        return GuardResult(
            name, True, "L3 refusal floor does not apply to this request", values
        )
    return GuardResult(name, False, f"{L3_TRIGGER}: {reason}", values)


def guard_no_ground_abort(inputs: GuardInputs) -> GuardResult:
    """SCRUM-380, MAF v2.0 section 6: no ground-commanded abort is standing.

    Written the positive way like every other guard -- it passes while no abort
    is in force -- but it is not consulted like the others. evaluate_safety_monitor
    checks it before the transition table so an abort can interrupt a staged or
    executing sequence mid-pass, which is the whole point of the command.
    """
    name = "no_ground_abort"
    abort = inputs.ground_abort
    if abort is None:
        return GuardResult(name, True, "no ground abort standing", {})
    return GuardResult(
        name, False,
        f"ground abort from {abort.operator_id}: {abort.abort_reason}",
        {
            "abort_reason": abort.abort_reason,
            "operator_id": abort.operator_id,
            "issued_at_utc": abort.issued_at_utc,
            "command_id": abort.command_id,
        },
    )


def guard_subsystem_healthy(inputs: GuardInputs) -> GuardResult:
    """Section 6.3: any partial subsystem failure goes to M4.

    'The system does not degrade gracefully into an uncertified burn.'
    """
    name = "no_subsystem_failure"
    values = {"subsystem_failure": inputs.subsystem_failure or None}
    if inputs.subsystem_failure:
        return GuardResult(
            name, False,
            f"subsystem failure reported: {inputs.subsystem_failure}",
            values,
        )
    return GuardResult(name, True, "no subsystem failure reported", values)


def guard_watchdog_alive(inputs: GuardInputs) -> GuardResult:
    """Section 3, M2 to M4: watchdog expired."""
    name = "watchdog_alive"
    values = {"watchdog_expired": inputs.watchdog_expired}
    if inputs.watchdog_expired:
        return GuardResult(name, False, "staging watchdog expired", values)
    return GuardResult(name, True, "staging watchdog is alive", values)


def guard_l1_approval_accepted(inputs: GuardInputs) -> GuardResult:
    """Section 3, M2 to M3 (L1): GNCApprovalCommand received and
    approval_accepted = true."""
    name = "l1_approval_accepted"
    values = {
        "approval_command_received": inputs.approval_command_received,
        "approval_accepted": inputs.approval_accepted,
    }
    if not inputs.approval_command_received:
        return GuardResult(name, False, "no GNCApprovalCommand received", values)
    if inputs.approval_accepted is not True:
        return GuardResult(name, False, "approval_accepted is not true", values)
    return GuardResult(name, True, "operator approval accepted", values)


def guard_l2_veto_window_expired(inputs: GuardInputs) -> GuardResult:
    """Section 3, M2 to M3 (L2): veto_window_close_utc passed with no
    GNCVetoCommand."""
    name = "l2_veto_window_expired"
    close = inputs.veto_window_close_utc
    values = {
        "veto_window_close_utc": _iso(close) if close else None,
        "t_now_utc": _iso(inputs.t_now_utc),
        "veto_command_received": inputs.veto_command_received,
    }
    if close is None:
        return GuardResult(name, False, "no veto window has been opened", values)
    if inputs.veto_command_received:
        return GuardResult(name, False, "a GNCVetoCommand was received", values)
    if inputs.t_now_utc < close:
        remaining = (close - inputs.t_now_utc).total_seconds()
        return GuardResult(
            name, False, f"veto window still open for {remaining:.0f} s", values
        )
    return GuardResult(name, True, "veto window closed with no veto", values)


def guard_reveto_may_restage(inputs: GuardInputs) -> GuardResult:
    """Section 3, the M2 to M2 row, and section 8's re-veto recommendation:
    hold staged and re-enter the veto window only if a fresh CDM is available
    within the freshness bound and validity is still EARNED. Otherwise M4."""
    name = "reveto_may_restage"
    values = {
        "veto_accepted": inputs.veto_accepted,
        "pc": inputs.pc,
        "pc_maneuver_threshold": inputs.pc_maneuver_threshold,
        "fresh_cdm_available": inputs.fresh_cdm_available,
        "data_age_s": inputs.data_age_s,
        "data_freshness_bound_s": inputs.data_freshness_bound_s,
        "validity_routing": (
            inputs.validity_routing.value if inputs.validity_routing else None
        ),
    }
    if inputs.veto_accepted is not True:
        return GuardResult(name, False, "veto_accepted is not true", values)
    if not guard_pc_above_action(inputs).passed:
        return GuardResult(
            name, False, "Pc is no longer at or above the action line", values
        )
    if not inputs.fresh_cdm_available:
        return GuardResult(name, False, "no fresh CDM is available", values)
    if (
        inputs.data_age_s is None
        or inputs.data_age_s > inputs.data_freshness_bound_s
    ):
        return GuardResult(
            name, False, "the new CDM is outside the freshness bound", values
        )
    if not guard_validity_earned(inputs).passed:
        return GuardResult(name, False, "validity is not EARNED on the new CDM", values)
    return GuardResult(name, True, "fresh CDM still EARNED; re-enter the veto window", values)


def guard_execution_nominal(inputs: GuardInputs) -> GuardResult:
    """Section 3, M3 rows: GNCReport received and execution_status = NOMINAL."""
    name = "execution_nominal"
    values = {
        "gnc_report_received": inputs.gnc_report_received,
        "execution_status": inputs.execution_status or None,
    }
    if not inputs.gnc_report_received:
        return GuardResult(name, False, "no GNCReport received", values)
    if inputs.execution_status != "NOMINAL":
        return GuardResult(
            name, False,
            f"execution_status is {inputs.execution_status or 'unset'}, not NOMINAL",
            values,
        )
    return GuardResult(name, True, "burn executed nominally", values)


def guard_m2_post_safe(inputs: GuardInputs) -> GuardResult:
    """Section 4.5: m2_post > m2_safe (25, 5-sigma in the conjunction plane).

    Strictly greater, as the doc writes it.
    """
    name = "m2_post_above_safe_threshold"
    values = {
        "m2_post": inputs.m2_post,
        "m2_safe_threshold": inputs.m2_safe_threshold,
    }
    if inputs.m2_post is None:
        return GuardResult(name, False, "no post-burn Mahalanobis distance", values)
    if inputs.m2_post > inputs.m2_safe_threshold:
        return GuardResult(
            name, True,
            f"m2_post {inputs.m2_post:.4g} > m2_safe {inputs.m2_safe_threshold:.4g}",
            values,
        )
    return GuardResult(
        name, False,
        f"m2_post {inputs.m2_post:.4g} does not exceed m2_safe "
        f"{inputs.m2_safe_threshold:.4g}",
        values,
    )


# ---------------------------------------------------------------------------
# Guard sets, one per transition row
# ---------------------------------------------------------------------------


def m1_to_m2_guards(inputs: GuardInputs) -> Tuple[GuardResult, ...]:
    """Section 3, the M1 to M2 row, in the order the doc lists it.

    'Pc >= pc_maneuver_threshold AND IOD confidence CONFIDENT AND validity
    EARNED AND envelope satisfied AND secondary conflict clear AND slew
    feasible AND authority L1 or L2.'

    Every guard is evaluated, not short-circuited, so the transition record can
    show all seven verdicts rather than only the first failure.
    """
    return (
        guard_pc_above_action(inputs),
        guard_iod_confident(inputs),
        guard_validity_earned(inputs),
        guard_envelope_satisfied(inputs),
        guard_secondary_clear(inputs),
        guard_slew_feasible(inputs),
        guard_authority_l1_or_l2(inputs),
        # SCRUM-380: floors layered on top of the section 3 row, not among its
        # seven, so they sit after them. A hollow record blocks staging here
        # (holding M1); a staged event with one escalates via the M2 set.
        guard_authority_baseline_validated(inputs),
        guard_cdm_record_usable(inputs),
        guard_l3_pre_verified_action(inputs),
    )


def _m1_escalation_guards(inputs: GuardInputs) -> Tuple[GuardResult, ...]:
    """Section 3, the M1 to M4 row.

    'IOD confidence DEGRADED or REJECTED OR validity NOT_EARNED OR secondary
    conflict not clear OR TCA < min_hours_before_tca with Pc still above
    monitor threshold.'

    Interpretive call, and the one most worth a reviewer's attention: these
    clauses fire on a negative verdict, not on an absent one. In M1 the IOD and
    validity gates may legitimately not have run yet -- they are M1 to M2
    preconditions -- so escalating on absence would safehold every watch event
    the moment it was flagged. Absence still blocks M1 to M2, because
    m1_to_m2_guards fails closed on None. What it does not do is force a
    safehold.

    Section 4.2's 'if the check was not performed, treat as NOT CLEAR' is read
    the same way: it makes the secondary guard fail, which blocks staging. Only
    a performed check that came back not clear escalates.
    """
    results: List[GuardResult] = []

    if inputs.iod_proceeds_to_validity is False:
        results.append(guard_iod_confident(inputs))
    if inputs.validity_routing is ValidityRouting.BLOCKED or (
        not inputs.validity_service_available
    ):
        results.append(guard_validity_earned(inputs))
    if inputs.secondary_check_performed and not inputs.secondary_conjunction_clear:
        results.append(guard_secondary_clear(inputs))

    tca_guard = guard_tca_above_floor(inputs)
    if not tca_guard.passed:
        results.append(tca_guard)

    # Section 6.3, the ingest row. Not in the section 3 table, but the same
    # destination for the same reason: a stale CDM cannot certify anything.
    freshness = guard_last_cdm_still_fresh(inputs)
    if not freshness.passed:
        results.append(freshness)

    return tuple(results)


def m2_continuous_guards(inputs: GuardInputs) -> Tuple[GuardResult, ...]:
    """The three guards section 4 says are monitored continuously while staged.

    Slew feasibility (4.3: 'checked at the M1 to M2 transition and continuously
    monitored in M2'), validity (4.4: re-evaluated on every new CDM, and a drop
    to NOT_EARNED goes to M4 immediately regardless of veto window status), and
    the envelope (3, M2 to M4: 'envelope violated').

    These are the guards a burn must still satisfy at the moment of execution,
    not merely when it was staged, so they are also the ones recorded on the
    M2 to M3 transition.
    """
    return (
        guard_slew_feasible(inputs),
        guard_validity_earned(inputs),
        guard_envelope_satisfied(inputs),
    )


def _m2_escalation_guards(inputs: GuardInputs) -> Tuple[GuardResult, ...]:
    """Section 3, the M2 to M4 row: 'Slew infeasible at veto_window_close_utc
    OR validity gate fails mid-staging OR envelope violated OR L1
    approval_accepted = false OR watchdog expired.'"""
    results: List[GuardResult] = [
        result for result in m2_continuous_guards(inputs) if not result.passed
    ]

    # SCRUM-380. Staged is past the point where blocking is enough: a burn is
    # loaded against a record that turns out to be hollow, so it escalates.
    for guard in (guard_cdm_record_usable, guard_authority_baseline_validated):
        result = guard(inputs)
        if not result.passed:
            results.append(result)

    if inputs.approval_command_received and inputs.approval_accepted is False:
        results.append(guard_l1_approval_accepted(inputs))

    watchdog = guard_watchdog_alive(inputs)
    if not watchdog.passed:
        results.append(watchdog)

    return tuple(results)


# ---------------------------------------------------------------------------
# Section 6.1: comms gaps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommsGapStatus:
    """What a comms gap means for the mode the spacecraft is already in.

    Section 6.1 changes no mode by itself. What it does is say which behaviours
    remain available during a gap, and the one that matters is the M2 L2 row:
    'Continue veto countdown with onboard clock. Auto-execute at
    veto_window_close_utc if no veto received. This is the intended L2
    behaviour; autonomy was pre-granted for exactly this case.'

    So this is a report, not a decision. The monitor's guards already produce
    the right modes during a gap -- an L1 cannot execute because no approval can
    arrive, an L2 can because it needs nothing from the ground -- and this makes
    that explicit and auditable instead of merely emergent.
    """

    in_gap: bool
    gap_s: float
    mode: FlightMode
    behaviour: str
    may_auto_execute: bool
    data_stale: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "in_comms_gap": self.in_gap,
            "comms_gap_s": self.gap_s,
            "comms_gap_threshold_s": COMMS_GAP_THRESHOLD_S,
            "behaviour": self.behaviour,
            "may_auto_execute": self.may_auto_execute,
            "data_stale": self.data_stale,
        }


_GAP_BEHAVIOUR = {
    FlightMode.M0_NOMINAL: "continue monitoring; no action required",
    FlightMode.M1_WATCH: "continue monitoring with the last known CDM",
    FlightMode.M3_EXECUTING: "continue burn execution; report on restore",
    FlightMode.M4_SAFE_HOLD: "hold safehold; the operator clears via ARBITER",
}


def comms_gap_status(
    mode: Any,
    *,
    comms_gap_s: float,
    authority_level: str = "",
    data_age_s: Optional[float] = None,
    data_freshness_bound_s: float = DATA_FRESHNESS_BOUND_S,
) -> CommsGapStatus:
    """Section 6.1's table, as a pure lookup."""
    current = as_mode(mode)
    in_gap = comms_gap_s > COMMS_GAP_THRESHOLD_S
    stale = data_age_s is not None and data_age_s > data_freshness_bound_s

    if current is FlightMode.M2_STAGED:
        if authority_level == "L2":
            behaviour = (
                "continue the veto countdown on the onboard clock; auto-execute "
                "at veto_window_close_utc if no veto arrives"
            )
            may_auto_execute = True
        else:
            behaviour = (
                "hold staged; an L1 burn cannot execute without operator approval"
            )
            may_auto_execute = False
    else:
        behaviour = _GAP_BEHAVIOUR[current]
        may_auto_execute = False

    if not in_gap:
        behaviour = "no comms gap"

    return CommsGapStatus(
        in_gap=in_gap,
        gap_s=float(comms_gap_s),
        mode=current,
        behaviour=behaviour,
        may_auto_execute=may_auto_execute and in_gap,
        data_stale=stale,
    )


# ---------------------------------------------------------------------------
# The monitor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitorDecision:
    """The mode the spacecraft should be in, and the record of how it got there.

    inputs is the bundle the decision was made from, carried so an audit record
    can be assembled from the decision alone. It is deliberately left out of
    to_dict(): the guard results already carry the values each guard actually
    read, and dumping the whole bundle would put unread inputs into the record
    as though they had been part of the decision.
    """

    mode: FlightMode
    transition: TransitionDecision
    guards: Tuple[GuardResult, ...] = ()
    authorized_execution: Optional[AuthorizedExecution] = None
    conjunction_id: str = ""
    evaluated_at_utc: str = ""
    inputs: Optional[GuardInputs] = None
    comms_gap: Optional["CommsGapStatus"] = None  # section 6.1
    # Set only when this evaluation started from a persisted mode, i.e. the
    # section 6.2 reboot path. None on a normal evaluation.
    reboot_recovery: Optional[Any] = None

    # -- SCRUM-380 ----------------------------------------------------------

    @property
    def aborted(self) -> bool:
        """Whether a ground abort drove this decision."""
        return bool(self.inputs is not None and self.inputs.ground_abort is not None)

    def abort_audit_entry(self) -> Optional[Dict[str, Any]]:
        """The section 7 abort entry for the tamper-evident chain, or None."""
        if not self.aborted:
            return None
        return self.inputs.ground_abort.audit_entry(
            from_mode=self.transition.from_mode.value,
            conjunction_id=self.conjunction_id,
        )

    @property
    def post_burn_feasible(self) -> Optional[bool]:
        """The section 4.5 post-burn feasibility flag, or None if not evaluated.

        Surfaced as its own field so a post-burn residual failure shows up in the
        evidence package as a feasibility result, not only as a mode change that
        a reader has to interpret.
        """
        for guard in self.guards:
            if guard.name == "m2_post_above_safe_threshold":
                return guard.passed
        return None

    @property
    def escalated(self) -> bool:
        return self.transition.escalated

    @property
    def changed_mode(self) -> bool:
        return self.transition.is_transition

    def failing_guard(self) -> Optional[GuardResult]:
        return first_failure(self.guards)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "from_mode": self.transition.from_mode.value,
            "to_mode": self.transition.to_mode.value,
            "requested_mode": self.transition.requested_mode.value,
            "trigger": self.transition.trigger,
            "escalated": self.transition.escalated,
            "changed_mode": self.changed_mode,
            "conjunction_id": self.conjunction_id,
            "evaluated_at_utc": self.evaluated_at_utc,
            "guards": [g.to_dict() for g in self.guards],
            "authorized_execution": (
                self.authorized_execution.to_dict()
                if self.authorized_execution
                else None
            ),
            "comms_gap": self.comms_gap.to_dict() if self.comms_gap else None,
            "reboot_recovery": (
                self.reboot_recovery.to_dict() if self.reboot_recovery else None
            ),
            # SCRUM-380
            "aborted": self.aborted,
            "abort": (
                self.inputs.ground_abort.to_dict()
                if self.aborted else None
            ),
            "post_burn_feasible": self.post_burn_feasible,
            "authority_granted": (
                self.inputs.authority() if self.inputs else ""
            ),
            "authority_effective": (
                self.inputs.effective_authority() if self.inputs else ""
            ),
        }


def _decide(
    inputs: GuardInputs,
    requested_mode: Optional[FlightMode],
    trigger: str,
    guards: Sequence[GuardResult],
    authorized_execution: Optional[AuthorizedExecution] = None,
) -> MonitorDecision:
    if requested_mode is None:
        transition = hold(inputs.current_mode, trigger)
    else:
        transition = resolve_transition(inputs.current_mode, requested_mode, trigger)
    return MonitorDecision(
        mode=transition.to_mode,
        transition=transition,
        guards=tuple(guards),
        authorized_execution=authorized_execution,
        conjunction_id=inputs.conjunction_id,
        evaluated_at_utc=_iso(inputs.t_now_utc),
        inputs=inputs,
    )


def _authorize(inputs: GuardInputs, approval_basis: str) -> AuthorizedExecution:
    """Build the payload SCRUM-382 consumes. Only called on entry to M3."""
    command = inputs.command
    assert command is not None, "M3 is unreachable without a command"
    return AuthorizedExecution(
        conjunction_id=inputs.conjunction_id,
        dv_eci_km_s=command.dv_eci_km_s,
        dv_magnitude_m_s=command.dv_magnitude_m_s,
        t_burn_utc=_iso(command.t_burn_utc),
        mode=FlightMode.M3_EXECUTING.value,
        authority_level=inputs.effective_authority(),
        envelope_version=inputs.envelope_version or "",
        approval_basis=approval_basis,
        validity_status=str(inputs.validity_evidence.get("validity_status", "")),
        validity_epsilon=float(inputs.validity_evidence.get("validity_epsilon", 0.0)),
        covariance_source=inputs.covariance_source,
        authorized_at_utc=_iso(inputs.t_now_utc),
    )


def _evaluate_m0(inputs: GuardInputs) -> MonitorDecision:
    watch = guard_pc_above_monitor(inputs)
    if watch.passed:
        return _decide(inputs, FlightMode.M1_WATCH, watch.name, (watch,))
    return _decide(inputs, None, "pc_below_monitor_threshold", (watch,))


def _evaluate_m1(inputs: GuardInputs) -> MonitorDecision:
    escalations = _m1_escalation_guards(inputs)
    if escalations:
        return _decide(inputs, FlightMode.M4_SAFE_HOLD, escalations[0].name, escalations)

    staging = m1_to_m2_guards(inputs)
    if _all_passed(staging):
        return _decide(inputs, FlightMode.M2_STAGED, "m1_to_m2_guards_all_satisfied", staging)

    resolved = guard_pc_below_monitor_sustained(inputs)
    if resolved.passed:
        return _decide(inputs, FlightMode.M0_NOMINAL, resolved.name, (resolved,))

    failing = first_failure(staging)
    return _decide(
        inputs, None,
        f"holding M1: {failing.name} not satisfied" if failing else "holding M1",
        staging,
    )


def _evaluate_m2(inputs: GuardInputs) -> MonitorDecision:
    escalations = _m2_escalation_guards(inputs)
    if escalations:
        return _decide(inputs, FlightMode.M4_SAFE_HOLD, escalations[0].name, escalations)

    # Section 3, M2 to M0: the event resolved before the burn. Checked before
    # any execute path, so a conjunction that has already cleared never gets a
    # burn it no longer needs.
    resolved = guard_pc_below_monitor_sustained(inputs)
    if resolved.passed:
        return _decide(inputs, FlightMode.M0_NOMINAL, resolved.name, (resolved,))

    # Section 3, M2 to M2: a veto arrived. Evaluated before the execute paths,
    # so an accepted veto can never race an auto-execute.
    if inputs.veto_command_received:
        restage = guard_reveto_may_restage(inputs)
        if restage.passed:
            return _decide(inputs, FlightMode.M2_STAGED, restage.name, (restage,))
        return _decide(inputs, FlightMode.M4_SAFE_HOLD, restage.name, (restage,))

    authority = inputs.effective_authority()

    # Reached only when no escalation fired, so these three have passed. They
    # are recorded on the transition because they are what makes the burn
    # lawful at the moment of execution, not merely when it was staged.
    continuous = m2_continuous_guards(inputs)

    if authority == "L1":
        approval = guard_l1_approval_accepted(inputs)
        if approval.passed:
            return _decide(
                inputs, FlightMode.M3_EXECUTING, approval.name,
                (approval,) + continuous,
                _authorize(inputs, APPROVAL_BASIS_L1_OPERATOR),
            )
        return _decide(inputs, None, "holding M2: awaiting operator approval", (approval,))

    if authority == "L2":
        window = guard_l2_veto_window_expired(inputs)
        provenance = guard_covariance_provenance(inputs)
        if window.passed and provenance.passed:
            return _decide(
                inputs, FlightMode.M3_EXECUTING, window.name,
                (window, provenance) + continuous,
                _authorize(inputs, APPROVAL_BASIS_L2_VETO_EXPIRED),
            )
        if window.passed and not provenance.passed:
            # The window closed but the burn is not eligible to auto-execute.
            # Section 4.1 wants an operator, not a silent execution.
            return _decide(
                inputs, FlightMode.M4_SAFE_HOLD, provenance.name, (window, provenance)
            )
        return _decide(
            inputs, None, "holding M2: veto window still open", (window, provenance)
        )

    # Staged with no authority to act is ambiguous, and section 1 says
    # ambiguous escalates.
    authority_guard = guard_authority_l1_or_l2(inputs)
    return _decide(
        inputs, FlightMode.M4_SAFE_HOLD, authority_guard.name, (authority_guard,)
    )


def _evaluate_m3(inputs: GuardInputs) -> MonitorDecision:
    nominal = guard_execution_nominal(inputs)

    if not nominal.passed:
        if not inputs.gnc_report_received:
            return _decide(inputs, None, "holding M3: burn in progress", (nominal,))
        # Section 3, M3 to M4: ABORTED or PARTIAL with residual Pc still
        # elevated. A non-nominal execution with residual risk NOT elevated has
        # no row at all in section 3, and section 6.3 sends both the thruster
        # fault and the attitude failure to M4 unconditionally, so the
        # unspecified case escalates rather than inventing an M3 to M0 path for
        # a burn that did not complete. residual_pc_elevated is recorded either
        # way, so the operator sees which of the two it was.
        residual = GuardResult(
            "residual_risk_resolved",
            False,
            f"execution_status {inputs.execution_status or 'unset'} with "
            f"residual_pc_elevated={inputs.residual_pc_elevated}",
            {
                "execution_status": inputs.execution_status or None,
                "residual_pc_elevated": inputs.residual_pc_elevated,
            },
        )
        return _decide(inputs, FlightMode.M4_SAFE_HOLD, nominal.name, (nominal, residual))

    # Section 3, M3 to M1: nominal but Pc still above the monitor line, so a
    # post-burn replan is required. Checked before M3 to M0, since a replan is
    # required whenever it applies.
    still_watching = guard_pc_above_monitor(inputs)
    if still_watching.passed:
        return _decide(
            inputs, FlightMode.M1_WATCH, "post_burn_replan_required",
            (nominal, still_watching),
        )

    safe = guard_m2_post_safe(inputs)
    if safe.passed:
        return _decide(inputs, FlightMode.M0_NOMINAL, safe.name, (nominal, safe))

    return _decide(inputs, FlightMode.M4_SAFE_HOLD, safe.name, (nominal, safe))


def _evaluate_m4(inputs: GuardInputs) -> MonitorDecision:
    """Section 6.1: 'Hold safehold. Do not exit M4 autonomously.'"""
    if inputs.operator_clearance_received:
        return _decide(inputs, FlightMode.M0_NOMINAL, "operator_clearance_via_arbiter", ())
    return _decide(inputs, None, "holding M4: awaiting operator clearance", ())


_MODE_EVALUATORS = {
    FlightMode.M0_NOMINAL: _evaluate_m0,
    FlightMode.M1_WATCH: _evaluate_m1,
    FlightMode.M2_STAGED: _evaluate_m2,
    FlightMode.M3_EXECUTING: _evaluate_m3,
    FlightMode.M4_SAFE_HOLD: _evaluate_m4,
}


def evaluate_safety_monitor(inputs: GuardInputs) -> MonitorDecision:
    """Decide the mode for one evaluation. Pure: no I/O, no clock, no defaults.

    Precedence, in this order, and the reason for it:

    1. Subsystem failure (section 6.3) beats everything, from any mode. M4 is
       reached through resolve_transition, so an origin section 3 gives no
       M4 row -- M0 -- lands in M4 as an escalation with the failure named in
       the trigger, rather than being silently dropped.
    2. M4 never exits autonomously (section 6.1), so a failure while already
       safeheld holds rather than re-escalating.
    3. Within a mode, the M4 row is evaluated before any forward transition.
       An ambiguous evaluation escalates rather than staging or executing.
    """
    mode = inputs.current_mode

    # SCRUM-380, MAF v2.0 section 6. The ground abort is resolved before the
    # transition table, before the subsystem check, before anything. That
    # ordering is the feature: an abort that waited its turn behind the normal
    # guards could not interrupt an L2 whose veto window closes on this very
    # evaluation, which is exactly the case a ground operator issues it for.
    abort = guard_no_ground_abort(inputs)
    if not abort.passed:
        if mode is FlightMode.M4_SAFE_HOLD:
            # Already safeheld. Hold, and do not let a clearance arriving in the
            # same evaluation lift a still-standing abort: section 1 escalates
            # the ambiguous case rather than resuming flight on it.
            decision = _decide(
                inputs, None,
                f"holding M4: {ABORT_TRIGGER} still standing "
                f"({inputs.ground_abort.abort_reason})",
                (abort,),
            )
        else:
            decision = _decide(
                inputs, FlightMode.M4_SAFE_HOLD,
                f"{ABORT_TRIGGER}: {inputs.ground_abort.abort_reason}",
                (abort,),
            )
        return replace(
            decision,
            comms_gap=comms_gap_status(
                mode,
                comms_gap_s=inputs.comms_gap_s,
                authority_level=inputs.effective_authority(),
                data_age_s=inputs.data_age_s,
                data_freshness_bound_s=inputs.data_freshness_bound_s,
            ),
        )

    failure = guard_subsystem_healthy(inputs)
    if not failure.passed:
        if mode is FlightMode.M4_SAFE_HOLD:
            decision = _decide(
                inputs, None, "holding M4: awaiting operator clearance", (failure,)
            )
        else:
            decision = _decide(
                inputs, FlightMode.M4_SAFE_HOLD, failure.detail, (failure,)
            )
    else:
        decision = _MODE_EVALUATORS[mode](inputs)

    # Section 6.1 changes no mode on its own; it reports which behaviours are
    # available during a gap. Attached to the decision so the audit record shows
    # that an L2 auto-execute during a gap was the pre-granted behaviour and not
    # a monitor that lost track of the ground.
    return replace(
        decision,
        comms_gap=comms_gap_status(
            mode,
            comms_gap_s=inputs.comms_gap_s,
            authority_level=inputs.effective_authority(),
            data_age_s=inputs.data_age_s,
            data_freshness_bound_s=inputs.data_freshness_bound_s,
        ),
    )
