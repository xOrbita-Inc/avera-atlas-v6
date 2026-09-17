"""
SCRUM-379 -- assemble GuardInputs from what /v1/evaluate already has.

The safety monitor is pure and reads nothing but its inputs bundle. This module
is the one place that knows where each of those inputs lives in the planner's
world: the scorer's resolved Pc, the ATLAS artifact's secondary-conflict check,
the compiled authorization envelope, the operator policy's thresholds, and the
optional request blocks that carry the IOD verdict and the observation arc.

Nothing is invented. An input the request did not supply arrives as None, and
the guard that needs it fails closed. That is deliberate and is why a plain
/v1/evaluate call, which carries no IOD verdict and no arc, cannot stage a
maneuver: it has not shown that it may.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

from common.authorization_envelope import (
    AuthorityLevel,
    EnvelopeCompileError,
    ManeuverCapacityInput,
    compile_from_operator_policy,
)
from common.decision_state_machine import (
    FlightMode,
    GuardInputs,
    ManeuverCommand,
    as_mode,
)
from common.safety_floors import (
    AuthorityBaseline,
    GroundAbortCommand,
    cdm_record_rejection_reason,
    monitor_logic_hash,
)
from common.mode_persistence import (
    ModeStore,
    PersistedMode,
    RebootRecovery,
    recover_after_reboot,
)
from common.safety_monitor import MonitorDecision, evaluate_safety_monitor
from common.validity_seam import ArcObservation, assess_validity_at_tca

# Optional request blocks. All additive: a request that omits them evaluates
# exactly as it did before 379, and simply cannot stage.
AUTHORIZATION_KEY = "authorization"
IOD_KEY = "iod"
VALIDITY_KEY = "validity"
MONITOR_KEY = "monitor"


def parse_utc(text: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 UTC timestamp, returning None for anything unusable.

    A malformed timestamp is not an error to raise here: it is a missing input,
    and a missing input fails its guard closed.
    """
    if not text:
        return None
    try:
        value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _maneuver_command(scoring: Any) -> Optional[ManeuverCommand]:
    """The scorer's burn, or None when it recommended no burn at all."""
    t_burn = parse_utc(getattr(scoring, "t_burn_utc", ""))
    dv = getattr(scoring, "dv_eci_km_s", None)
    if t_burn is None or not dv or len(dv) != 3:
        return None
    if getattr(scoring, "no_go_reason_code", ""):
        return None
    return ManeuverCommand(
        dv_eci_km_s=tuple(float(c) for c in dv),
        dv_magnitude_m_s=float(getattr(scoring, "dv_magnitude_m_s", 0.0)),
        t_burn_utc=t_burn,
        direction=str(getattr(scoring, "direction", "")),
    )


def _secondary_check(artifact: Any) -> tuple[bool, bool]:
    """Section 4.2's two booleans, straight off the SCRUM-381 gate.

    A no-go artifact has no post_maneuver block and therefore no screen, which
    is 'not performed' and so NOT CLEAR.
    """
    post = getattr(artifact, "post_maneuver", None)
    check = getattr(post, "secondary_conflict", None) if post is not None else None
    if check is None:
        return False, False
    return (
        bool(check.secondary_check_performed),
        bool(check.secondary_conjunction_clear),
    )


def _envelope_fields(
    policy: Any, cap: Any, authorization: Dict[str, Any]
) -> Dict[str, Any]:
    """Compile the authorization envelope for this request.

    Returns the envelope-shaped GuardInputs fields. A compile failure is not an
    exception here: EnvelopeCompileError is precisely the 'envelope not
    satisfied' signal section 4.1 describes, so it is recorded as envelope_error
    and the guard reports it.
    """
    requested = str(authorization.get("authority_level", AuthorityLevel.L0.value))
    lifetime = getattr(cap, "lifetime", None)
    propulsion = getattr(cap, "propulsion", None)

    fields: Dict[str, Any] = {
        "authority_level": requested,
        "envelope_version": None,
        "envelope_approving_identity": authorization.get("approving_identity"),
        "envelope_error": "",
        "envelope_expired": bool(authorization.get("envelope_expired", False)),
        "dv_cap_m_s": None,
        "v_remaining_m_s": getattr(lifetime, "v_remaining_m_s", None),
        "v_reserved_m_s": getattr(lifetime, "v_reserved_m_s", None),
    }

    if requested == AuthorityLevel.L0.value:
        # L0 is advisory only (section 5). Nothing to compile, and the
        # authority guard will decline it anyway.
        return fields

    try:
        envelope = compile_from_operator_policy(
            policy,
            profile_id=str(authorization.get("profile_id", f"{policy.operator_id}-profile")),
            mission_class=str(authorization.get("mission_class", "LEO")),
            risk_budget_id=str(authorization.get("risk_budget_id", f"{policy.operator_id}-risk")),
            maneuver_capacity=ManeuverCapacityInput(
                max_dv_per_burn_m_s=float(getattr(propulsion, "max_dv_per_burn_m_s", 0.0)),
                v_remaining_m_s=float(getattr(lifetime, "v_remaining_m_s", 0.0)),
                v_reserved_m_s=float(getattr(lifetime, "v_reserved_m_s", 0.0)),
                max_maneuvers_per_week=int(policy.max_maneuvers_per_week),
            ),
            authority_level=AuthorityLevel(requested),
            l2_raise_to_1_m_s=bool(authorization.get("l2_raise_to_1_m_s", False)),
        )
    except (EnvelopeCompileError, ValueError, TypeError) as exc:
        fields["envelope_error"] = str(exc)
        return fields

    fields["envelope_version"] = envelope.version
    fields["dv_cap_m_s"] = (
        envelope.effective_l1_dv_cap_m_s
        if envelope.authority_level is AuthorityLevel.L1
        else envelope.effective_l2_dv_cap_m_s
    )
    fields["v_reserved_m_s"] = envelope.effective_reserved_dv_m_s
    return fields


def _validity_fields(
    validity_block: Dict[str, Any], tca_utc: Optional[datetime]
) -> Dict[str, Any]:
    """Run the SCRUM-378 seam when the request carries an observation arc.

    Two paths, and no third. With an arc, the verdict is computed here, through
    the seam that propagates to TCA. Without one, validity is simply unknown --
    None, not EARNED, not NOT_EARNED -- and the guard fails closed.

    A caller may not hand in a routing decision directly. Accepting one would
    let an unverified 'AUTONOMOUS' in the request body authorise a burn, which
    is the single worst thing this seam could allow.
    """
    fields: Dict[str, Any] = {
        "validity_routing": None,
        "validity_evidence": {},
        "validity_service_available": bool(
            validity_block.get("service_available", True)
        ),
    }

    arc = validity_block.get("observations") or []
    target = validity_block.get("target_state") or {}
    if not arc or not target or tca_utc is None:
        return fields

    observations = []
    for entry in arc:
        epoch = parse_utc(entry.get("epoch_utc"))
        if epoch is None:
            return fields
        observations.append(
            ArcObservation(
                epoch_utc=epoch,
                r_observer_km=np.asarray(entry["r_observer_km"], dtype=float),
                ra_sigma_rad=float(entry["ra_sigma_rad"]),
                dec_sigma_rad=float(entry["dec_sigma_rad"]),
                range_sigma_km=(
                    float(entry["range_sigma_km"])
                    if entry.get("range_sigma_km") is not None
                    else None
                ),
            )
        )

    state_epoch = parse_utc(target.get("epoch_utc"))
    if state_epoch is None:
        return fields

    assessment = assess_validity_at_tca(
        r_target_km=np.asarray(target["r_km"], dtype=float),
        v_target_km_s=np.asarray(target["v_km_s"], dtype=float),
        state_epoch_utc=state_epoch,
        tca_utc=tca_utc,
        observations=observations,
        r_rel_km_at_tca=np.asarray(validity_block["r_rel_km_at_tca"], dtype=float),
        v_rel_km_s_at_tca=np.asarray(validity_block["v_rel_km_s_at_tca"], dtype=float),
        phenomenologies_used=validity_block.get("phenomenologies_used"),
    )
    fields["validity_routing"] = assessment.routing
    fields["validity_evidence"] = dict(assessment.evidence)
    fields["validity_assessment"] = assessment
    return fields


def build_guard_inputs(
    *,
    body: Dict[str, Any],
    scoring: Any,
    artifact: Any,
    policy: Any,
    cap: Any,
    covariance_source: str,
    current_mode: FlightMode,
    t_now_utc: datetime,
    standing_abort: Optional[GroundAbortCommand] = None,
) -> GuardInputs:
    """Map one /v1/evaluate call onto the section 3 and 4 guard inputs.

    standing_abort is an abort already in force from a previous evaluation, read
    back out of the mode store. A ground abort holds until ground clearance, so
    it has to outlive the single request that issued it.
    """
    conjunction = body.get("conjunction") or {}
    authorization = body.get(AUTHORIZATION_KEY) or {}
    iod = body.get(IOD_KEY) or {}
    validity_block = body.get(VALIDITY_KEY) or {}
    monitor = body.get(MONITOR_KEY) or {}

    tca_utc = parse_utc(conjunction.get("t_ca_utc"))
    performed, clear = _secondary_check(artifact)

    # SCRUM-380: an abort on this request, otherwise one already standing. A
    # ground clearance is the only thing that lifts it, and that is handled by
    # the M4 to M0 row rather than by the abort simply going absent.
    ground_abort = (
        GroundAbortCommand.from_dict(monitor.get("ground_abort")) or standing_abort
    )
    envelope_fields = _envelope_fields(policy, cap, authorization)
    validity_fields = _validity_fields(validity_block, tca_utc)
    validity_fields.pop("validity_assessment", None)

    proceeds = iod.get("proceeds_to_validity")

    return GuardInputs(
        conjunction_id=scoring.conjunction_id,
        current_mode=current_mode,
        t_now_utc=t_now_utc,
        # The resolved Pc off the scoring result, which is where SCRUM-389 and
        # SCRUM-396 resolve it, rather than off the artifact's summary of it.
        pc=getattr(scoring, "pc_pre", None),
        pc_source=str(getattr(scoring, "pc_source", "")),
        pc_monitor_threshold=policy.pc_monitor_threshold,
        pc_maneuver_threshold=policy.pc_maneuver_threshold,
        consecutive_below_monitor=int(monitor.get("consecutive_below_monitor", 0)),
        command=_maneuver_command(scoring),
        latest_burn_utc=parse_utc(conjunction.get("latest_burn_utc")),
        t_ca_utc=tca_utc,
        min_hours_before_tca=policy.min_hours_before_tca,
        iod_proceeds_to_validity=(None if proceeds is None else bool(proceeds)),
        iod_confidence_verdict=str(iod.get("confidence_verdict", "")),
        secondary_check_performed=performed,
        secondary_conjunction_clear=clear,
        covariance_source=covariance_source,
        operator_ack_surrogate_covariance=bool(
            authorization.get("operator_ack_surrogate_covariance", False)
        ),
        data_age_s=(
            float(conjunction["data_age_s"])
            if conjunction.get("data_age_s") is not None
            else None
        ),
        approval_command_received=bool(monitor.get("approval_command_received", False)),
        approval_accepted=monitor.get("approval_accepted"),
        veto_window_close_utc=parse_utc(monitor.get("veto_window_close_utc")),
        veto_command_received=bool(monitor.get("veto_command_received", False)),
        veto_accepted=monitor.get("veto_accepted"),
        fresh_cdm_available=bool(monitor.get("fresh_cdm_available", False)),
        watchdog_expired=bool(monitor.get("watchdog_expired", False)),
        gnc_report_received=bool(monitor.get("gnc_report_received", False)),
        execution_status=str(monitor.get("execution_status", "")),
        m2_post=getattr(scoring, "m2_post", None),
        residual_pc_elevated=monitor.get("residual_pc_elevated"),
        comms_gap_s=float(monitor.get("comms_gap_s", 0.0)),
        subsystem_failure=str(monitor.get("subsystem_failure", "")),
        # SCRUM-380. A ground abort may arrive on this request, or may already
        # be standing from an earlier one -- a persisted abort stays in force
        # until ground clearance, so a later evaluate that carries no abort
        # block must not read as the abort having been lifted.
        ground_abort=ground_abort,
        ground_validated_baseline=AuthorityBaseline.from_dict(
            monitor.get("ground_validated_baseline")
        ),
        monitor_logic_hash=monitor_logic_hash(),
        cdm_record_rejected_reason=cdm_record_rejection_reason(
            conjunction.get("p_rel_km2"),
            conjunction.get("r_rel_km"),
            declared_degraded=bool(conjunction.get("degraded", False)),
        ),
        pre_verified_safe_action=str(
            authorization.get("pre_verified_safe_action", "")
        ),
        operator_clearance_received=bool(
            monitor.get("operator_clearance_received", False)
        ),
        **envelope_fields,
        **validity_fields,
    )


def resolve_current_mode(
    body: Dict[str, Any],
    store: Optional[ModeStore] = None,
    sat_id: str = "",
    t_now_utc: Optional[datetime] = None,
) -> tuple[FlightMode, Optional[RebootRecovery]]:
    """The mode this evaluation starts from, and any reboot recovery behind it.

    Precedence, and the reason for it:

    1. An explicit monitor.current_mode in the request. Ground stating where the
       spacecraft actually is beats anything this service remembers, because
       this service is not the spacecraft. A mode nobody recognises starts at
       M4: section 6.2 says the same about a reboot whose persisted state cannot
       be read, and for the same reason.
    2. A persisted mode, run through the section 6.2 recovery ladder. Reading a
       persisted mode is exactly the reboot case -- this process did not make
       that transition, so it cannot assume the burn window survived.
    3. M0. The honest reading of a stateless call with nothing persisted: no
       conjunction is being tracked yet.
    """
    monitor = body.get(MONITOR_KEY) or {}
    raw = monitor.get("current_mode")
    if raw is not None:
        try:
            return as_mode(raw), None
        except ValueError:
            return FlightMode.M4_SAFE_HOLD, None

    if store is not None and sat_id:
        persisted = store.read(sat_id)
        if persisted is not None:
            recovery = recover_after_reboot(
                persisted, t_now_utc or datetime.now(timezone.utc)
            )
            return recovery.mode, recovery

    return FlightMode.M0_NOMINAL, None


def evaluate_request(
    *,
    body: Dict[str, Any],
    scoring: Any,
    artifact: Any,
    policy: Any,
    cap: Any,
    covariance_source: str,
    current_mode: Optional[FlightMode] = None,
    t_now_utc: Optional[datetime] = None,
    store: Optional[ModeStore] = None,
    software_version: str = "",
) -> MonitorDecision:
    """Run the safety monitor for one /v1/evaluate call, and persist the result.

    The clock read and the store live here rather than in the monitor, which
    stays a pure function of its inputs.

    Section 6.2 asks for the mode to be persisted 'before any transition'. On a
    stateless HTTP service the closest honest reading is: persist the mode the
    monitor decided on, as part of the same call that decided it, before the
    response goes out. There is no window in which the caller has been told M3
    and the store still says M2.
    """
    now = t_now_utc or datetime.now(timezone.utc)
    sat_id = str(getattr(artifact, "sat_id", "") or "")
    recovery = None
    if current_mode is None:
        current_mode, recovery = resolve_current_mode(body, store, sat_id, now)

    # SCRUM-380: an abort persisted by an earlier evaluation is still in force.
    standing_abort = None
    if store is not None and sat_id:
        persisted = store.read(sat_id)
        if persisted is not None:
            standing_abort = persisted.ground_abort

    inputs = build_guard_inputs(
        body=body,
        scoring=scoring,
        artifact=artifact,
        policy=policy,
        cap=cap,
        covariance_source=covariance_source,
        current_mode=current_mode,
        t_now_utc=now,
        standing_abort=standing_abort,
    )
    decision = evaluate_safety_monitor(inputs)
    decision = replace(decision, reboot_recovery=recovery)

    if store is not None and sat_id:
        store.write(
            PersistedMode(
                sat_id=sat_id,
                mode=decision.mode,
                conjunction_id=decision.conjunction_id,
                persisted_at_utc=decision.evaluated_at_utc,
                command=inputs.command,
                latest_burn_utc=inputs.latest_burn_utc,
                software_version=software_version,
                # SCRUM-380: the abort persists until ground clears it, which
                # is precisely the M4 to M0 transition. Clearing it on any other
                # exit would let a mode change lift a ground stop.
                ground_abort=(
                    None
                    if decision.mode is FlightMode.M0_NOMINAL
                    else inputs.ground_abort
                ),
            )
        )
    return decision
