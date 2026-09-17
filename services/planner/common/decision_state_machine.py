"""
SCRUM-379 -- MAF v2.0 section 8 decision state machine: modes, transition
table, and the guard-inputs bundle.

Authoritative values live in docs/scrum-333/state_machine_guards.md. They are
locked. Nothing in this module retunes a threshold; a floor change escalates to
Minh. Where a locked value already has a home in the tree
(authorization_envelope's LOCKED_MIN_TCA_HOURS, the compiled envelope's
pc_watch / pc_action, operator policy's thresholds) this module consumes it
rather than restating it. Only the timings that had no home before -- slew,
settle, margin, review, comms gap -- are declared here, each cited to its
section.

This module is deliberately data-only: modes, which transitions exist, and the
inputs a guard may read. The guard logic itself is in safety_monitor.py, so the
two can be reasoned about separately and the table can be checked against the
spec without executing a single guard.

The hard rule from section 1 -- "Any undefined or ambiguous transition defaults
to ESCALATE to M4" -- is implemented once, in resolve_transition, as the
default arm of the table lookup. There is no fall-through to a lower mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from common.authorization_envelope import LOCKED_MIN_TCA_HOURS
from common.evidence_record import DEFINED_TRANSITIONS
from common.evidence_record import FlightMode as EvidenceFlightMode
from common.safety_floors import (
    DEMOTED_AUTHORITY,
    AuthorityBaseline,
    GroundAbortCommand,
    authority_demotion_reason,
)

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


class FlightMode(str, Enum):
    """MAF v2.0 section 8 / guard doc section 2. Flight modes M0 to M4.

    A str-valued enum whose members are exactly the string constants
    evidence_record.FlightMode already publishes, so a mode can be handed
    straight to build_transition_record without translation. The equality of
    the two sets is asserted below rather than assumed.
    """

    M0_NOMINAL = "M0"
    M1_WATCH = "M1"
    M2_STAGED = "M2"
    M3_EXECUTING = "M3"
    M4_SAFE_HOLD = "M4"


assert {m.value for m in FlightMode} == set(EvidenceFlightMode.ALL), (
    "decision_state_machine.FlightMode and evidence_record.FlightMode have "
    "drifted apart; the evidence chain would record a mode the state machine "
    "cannot produce, or vice versa"
)


def as_mode(value: Any) -> FlightMode:
    """Coerce a mode string or enum member to FlightMode.

    Raises ValueError for anything that is not one of M0..M4, rather than
    inventing a mode. An unrecognised mode is a programming error, not a
    condition to escalate on.
    """
    if isinstance(value, FlightMode):
        return value
    return FlightMode(value)


# ---------------------------------------------------------------------------
# Locked timings and floors (guard doc sections 3 to 6)
# ---------------------------------------------------------------------------

# Section 4.3 and section 5. OPEN ASSUMPTION in the doc: bus spec pending
# (Blackwing). Modelled as one number because the doc quotes it as one number.
T_SLEW_SETTLE_S: float = 120.0

# Section 4.3 and section 5. Safety buffer.
T_MARGIN_S: float = 30.0

# Section 5. Operator minimum review time; drives the L2 minimum veto window.
T_REVIEW_S: float = 300.0

# Section 5. t_veto = max(t_review, t_slew + t_settle + t_margin), which at the
# locked values is max(300, 150) = 300 s. Declared as the doc's own
# "Minimum veto window" row rather than recomputed at every call site.
MIN_VETO_WINDOW_S: float = 300.0

# Section 6.1. No uplink for longer than this is a comms gap.
COMMS_GAP_THRESHOLD_S: float = 600.0

# Section 4.5. m2_safe for the M3 to M0 post-burn transition, 5-sigma
# separation in the conjunction plane.
M2_SAFE_MAHALANOBIS: float = 25.0

# Section 4.1 and section 8. Freshness bound for a Space-Track CDM.
DATA_FRESHNESS_BOUND_S: float = 86400.0

# Section 3, the M1 to M0 row: Pc below the monitor line for 2 consecutive
# evaluations, not one.
CONSECUTIVE_BELOW_MONITOR_TO_CLEAR: int = 2

# Section 3, the M1 to M4 row. Same 4.0 hour floor the envelope compiler
# already locks, imported rather than restated so the two cannot drift.
MIN_HOURS_BEFORE_TCA: float = LOCKED_MIN_TCA_HOURS

# Section 4.1. The covariance provenance an L2 auto-execute requires. A
# surrogate covariance is not disqualifying on its own, but it needs an
# explicit operator acknowledgement first.
REAL_CDM_COVARIANCE_SOURCE: str = "real_cdm"

# The authority levels that may stage a maneuver at all (section 3, M1 to M2).
AUTONOMOUS_AUTHORITY_LEVELS: Tuple[str, ...] = ("L1", "L2")


def slew_lead_time_s() -> float:
    """t_slew + t_settle + t_margin, the lead time the burn needs. Section 4.3."""
    return T_SLEW_SETTLE_S + T_MARGIN_S


def l2_veto_window_s() -> float:
    """Section 5: t_veto = max(t_review, t_slew + t_settle + t_margin)."""
    return max(T_REVIEW_S, slew_lead_time_s())


# ---------------------------------------------------------------------------
# Validity routing, mirrored at the module boundary
# ---------------------------------------------------------------------------


class ValidityRouting(str, Enum):
    """The three outcomes of services/validity's route_validity_verdict.

    Mirrored here as strings rather than imported so this module stays pure and
    the planner does not take a hard import dependency on the validity service.
    RoutingDecision is itself a str enum with these exact values, so a caller
    may pass either a RoutingDecision member or a plain string. The seam module
    (validity_seam.py) asserts the two enums have not drifted.
    """

    BLOCKED = "BLOCKED"
    OPERATOR_REVIEW = "OPERATOR_REVIEW"
    AUTONOMOUS = "AUTONOMOUS"


# ---------------------------------------------------------------------------
# The transition table
# ---------------------------------------------------------------------------

# Guard doc section 3, in full. The set is owned by evidence_record so the
# audit chain and the state machine cannot disagree about which transitions
# exist; imported here rather than restated.
TRANSITION_TABLE: frozenset = DEFINED_TRANSITIONS

ESCALATE_TRIGGER_PREFIX = "undefined_transition_escalate"


@dataclass(frozen=True)
class TransitionDecision:
    """One resolved transition: where the machine actually goes, and why.

    Attributes
    ----------
    from_mode, to_mode : FlightMode
        to_mode is the mode the machine actually enters. On an undefined
        transition that is M4, not the requested destination.
    requested_mode : FlightMode
        What the caller asked for. Preserved even when it was refused, so the
        audit record can state what was attempted.
    trigger : str
        The guard that fired, or the escalation reason.
    escalated : bool
        True when the requested transition was undefined and section 8's
        default arm forced M4.
    """

    from_mode: FlightMode
    to_mode: FlightMode
    requested_mode: FlightMode
    trigger: str
    escalated: bool

    @property
    def is_transition(self) -> bool:
        """False for a self-loop that leaves the mode unchanged."""
        return self.from_mode != self.to_mode


def is_defined_transition(from_mode: Any, to_mode: Any) -> bool:
    """Whether section 3 defines this (from, to) pair."""
    return (as_mode(from_mode).value, as_mode(to_mode).value) in TRANSITION_TABLE


def resolve_transition(
    from_mode: Any,
    requested_mode: Any,
    trigger: str = "",
) -> TransitionDecision:
    """Resolve a requested transition against the section 3 table.

    This is the default arm the whole state machine rests on. A pair that
    section 3 does not define resolves to M4 with an escalation trigger; it
    never falls through to the requested mode, and never falls back to a lower
    mode such as M0 or M1.

    A self-loop (from_mode == requested_mode) is honoured only where section 3
    defines one -- M2 to M2, the L2 re-veto row. Holding the current mode
    because no guard fired is not a transition at all and does not come through
    here; see safety_monitor.evaluate_safety_monitor.
    """
    origin = as_mode(from_mode)
    target = as_mode(requested_mode)

    if (origin.value, target.value) in TRANSITION_TABLE:
        return TransitionDecision(
            from_mode=origin,
            to_mode=target,
            requested_mode=target,
            trigger=trigger or f"{origin.value}->{target.value}",
            escalated=False,
        )

    return TransitionDecision(
        from_mode=origin,
        to_mode=FlightMode.M4_SAFE_HOLD,
        requested_mode=target,
        trigger=(
            f"{ESCALATE_TRIGGER_PREFIX}: {origin.value}->{target.value} is not a "
            f"defined transition (attempted trigger: {trigger or 'unspecified'}); "
            "MAF v2.0 section 8 escalates to M4"
        ),
        escalated=True,
    )


def hold(mode: Any, trigger: str) -> TransitionDecision:
    """Stay in the current mode because no transition guard fired.

    Not a transition, so it does not go through resolve_transition and is not
    an undefined-transition escalation. Section 1's escalate rule is about a
    transition that was attempted and is not in the table; declining to move is
    the state machine working normally, and is the common case in M0.
    """
    origin = as_mode(mode)
    return TransitionDecision(
        from_mode=origin,
        to_mode=origin,
        requested_mode=origin,
        trigger=trigger,
        escalated=False,
    )


# ---------------------------------------------------------------------------
# The maneuver under consideration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManeuverCommand:
    """The scorer's recommended burn, as the state machine sees it.

    Straight off ManeuverScoringResult -- direction, dv_eci_km_s,
    dv_magnitude_m_s, t_burn_utc -- and nothing else. The state machine never
    recomputes a maneuver; it only decides whether this one may be executed.

    A no-go evaluation has no command at all, which is why several guards
    (the dv cap, slew feasibility) fail closed when command is None: there is
    no burn to authorise.
    """

    dv_eci_km_s: Tuple[float, float, float]
    dv_magnitude_m_s: float
    t_burn_utc: datetime
    direction: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "dv_eci_km_s", tuple(float(c) for c in self.dv_eci_km_s))
        if len(self.dv_eci_km_s) != 3:
            raise ValueError("dv_eci_km_s must be a 3-vector")
        if self.t_burn_utc.tzinfo is None:
            raise ValueError("t_burn_utc must be timezone-aware")
        object.__setattr__(
            self, "t_burn_utc", self.t_burn_utc.astimezone(timezone.utc)
        )


# ---------------------------------------------------------------------------
# Guard inputs
# ---------------------------------------------------------------------------


def _utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("every timestamp in GuardInputs must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class GuardInputs:
    """Everything any guard in section 3 or 4 is allowed to read.

    One fully populated bundle, assembled by the caller, so the monitor is a
    pure function of it and every guard is testable without standing up the
    service. Nothing here is fetched, propagated, or defaulted from ambient
    state at guard time: if a value is unknown it arrives as None and the guard
    that needs it fails closed.

    Field groups follow the guard doc:
      identity/clock          -- who and when
      risk (section 3)        -- resolved Pc and the two thresholds
      timing (section 4.3)    -- burn window and slew feasibility
      gates (section 3)       -- IOD, validity, secondary
      envelope (section 4.1)  -- authority, dv cap, freshness, provenance
      M2 dynamics (sections 3, 4.4, 5)
      M3 outcome (sections 3, 4.5)
      recovery (section 6)
    """

    # -- identity and clock -------------------------------------------------
    conjunction_id: str
    current_mode: FlightMode
    t_now_utc: datetime

    # -- risk, section 3 ----------------------------------------------------
    # pc is the resolved value from ManeuverScoringResult.pc_pre, with its
    # pc_source travelling alongside it. Read from the scorer rather than from
    # RiskSummary.pc_pre because the scorer is where the number is resolved and
    # the artifact is a view of it. Since SCRUM-396 the two agree and either
    # would work; taking the origin means a future change to how the artifact
    # summarises risk cannot silently move the guard.
    pc: Optional[float] = None
    pc_source: str = ""
    pc_monitor_threshold: Optional[float] = None
    pc_maneuver_threshold: Optional[float] = None
    # Section 3, M1 to M0: consecutive evaluations already seen below the
    # monitor line, counting this one.
    consecutive_below_monitor: int = 0

    # -- the burn under consideration, and timing, section 4.3 --------------
    command: Optional[ManeuverCommand] = None
    latest_burn_utc: Optional[datetime] = None
    t_ca_utc: Optional[datetime] = None
    min_hours_before_tca: float = MIN_HOURS_BEFORE_TCA

    # -- gates, section 3 ---------------------------------------------------
    # The single boolean from services/tracker/iod.py's
    # IODSolution.proceeds_to_validity. None means the IOD gate was never run,
    # which fails closed exactly like False but is reported differently.
    iod_proceeds_to_validity: Optional[bool] = None
    iod_confidence_verdict: str = ""
    # RoutingDecision from services/validity, as a ValidityRouting value.
    validity_routing: Optional[ValidityRouting] = None
    # validity_evidence_values(verdict) from SCRUM-378, carried verbatim into
    # the transition record.
    validity_evidence: Mapping[str, Any] = field(default_factory=dict)
    # Section 6.3: an unavailable validity assessor is NOT_EARNED, not unknown.
    validity_service_available: bool = True
    # Section 6.3, the ingest row: with ingest down, M1 may continue on the last
    # CDM only while data_age_s is within the freshness bound.
    ingest_available: bool = True
    # Section 4.2: the SCRUM-381 horizon gate, both halves. A check that was
    # never performed is NOT CLEAR.
    secondary_check_performed: bool = False
    secondary_conjunction_clear: bool = False

    # -- envelope, section 4.1 ----------------------------------------------
    authority_level: Optional[str] = None
    envelope_version: Optional[str] = None
    envelope_approving_identity: Optional[str] = None
    # Non-empty when the envelope compiler / manager raised
    # EnvelopeApprovalError or EnvelopeActivationError. Any text here is a
    # not-satisfied signal.
    envelope_error: str = ""
    envelope_expired: bool = False
    # The effective cap for the granted authority level, straight off the
    # compiled envelope (effective_l1_dv_cap_m_s or effective_l2_dv_cap_m_s).
    dv_cap_m_s: Optional[float] = None
    v_remaining_m_s: Optional[float] = None
    v_reserved_m_s: Optional[float] = None
    covariance_source: str = ""
    operator_ack_surrogate_covariance: bool = False
    data_age_s: Optional[float] = None
    data_freshness_bound_s: float = DATA_FRESHNESS_BOUND_S

    # -- M2 dynamics, sections 3, 4.4 and 5 ---------------------------------
    approval_command_received: bool = False
    approval_accepted: Optional[bool] = None
    veto_window_close_utc: Optional[datetime] = None
    veto_command_received: bool = False
    veto_accepted: Optional[bool] = None
    fresh_cdm_available: bool = False
    watchdog_expired: bool = False

    # -- M3 outcome, sections 3 and 4.5 -------------------------------------
    gnc_report_received: bool = False
    execution_status: str = ""
    m2_post: Optional[float] = None
    m2_safe_threshold: float = M2_SAFE_MAHALANOBIS
    residual_pc_elevated: Optional[bool] = None

    # -- recovery, section 6 ------------------------------------------------
    comms_gap_s: float = 0.0
    subsystem_failure: str = ""
    operator_clearance_received: bool = False

    # -- SCRUM-380 safety floors, MAF v2.0 section 6 ------------------------
    # A ground abort. The highest-priority input here: evaluated before any
    # guard so it can interrupt a staged or executing sequence during a pass.
    ground_abort: Optional[GroundAbortCommand] = None
    # The configuration ground last validated, and what is running now. A
    # mismatch clamps authority to L0 until an explicit ground re-validate.
    ground_validated_baseline: Optional[AuthorityBaseline] = None
    monitor_logic_hash: str = ""
    # Non-empty when the CDM record was refused outright as degraded or
    # zero-filled. Reject-not-parse: a hollow record never reaches the guards.
    cdm_record_rejected_reason: str = ""
    # L3 is gated to a later phase; the only part that exists is the refusal.
    pre_verified_safe_action: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_mode", as_mode(self.current_mode))
        if self.validity_routing is not None:
            object.__setattr__(
                self, "validity_routing", ValidityRouting(self.validity_routing)
            )
        for name in (
            "t_now_utc",
            "latest_burn_utc",
            "t_ca_utc",
            "veto_window_close_utc",
        ):
            object.__setattr__(self, name, _utc(getattr(self, name)))
        if self.t_now_utc is None:
            raise ValueError("t_now_utc is required")

        # SCRUM-380, MAF v2.0 section 6: the locked floors are not negotiable.
        #
        # SCRUM-379 made these fields with locked defaults, which stops an
        # accident but not an argument: a caller passing min_hours_before_tca=1.0
        # or m2_safe_threshold=1.0 would have been honoured, and the floor would
        # have been loosened by the request it was meant to constrain.
        #
        # Clamped in one direction only. A caller may be MORE cautious than the
        # floor -- a longer TCA lead, a stricter separation, a shorter freshness
        # window -- and that value is kept. Anything laxer is discarded and the
        # locked value stands. A genuine threshold change is an escalation to
        # Minh, not a field on a request.
        object.__setattr__(
            self, "min_hours_before_tca",
            max(float(self.min_hours_before_tca), MIN_HOURS_BEFORE_TCA),
        )
        object.__setattr__(
            self, "m2_safe_threshold",
            max(float(self.m2_safe_threshold), M2_SAFE_MAHALANOBIS),
        )
        object.__setattr__(
            self, "data_freshness_bound_s",
            min(float(self.data_freshness_bound_s), DATA_FRESHNESS_BOUND_S),
        )

    # -- derived quantities, all pure --------------------------------------

    def hours_to_tca(self) -> Optional[float]:
        if self.t_ca_utc is None:
            return None
        return (self.t_ca_utc - self.t_now_utc).total_seconds() / 3600.0

    def authority(self) -> str:
        """The authority the envelope states, before any SCRUM-380 clamp.

        Recorded alongside effective_authority() so an audit entry can show that
        the envelope said L2 and the machine acted at L0, rather than leaving a
        reader to wonder why an L2 asset declined to execute.
        """
        level = self.authority_level
        if level is None:
            return ""
        return getattr(level, "value", str(level))

    def authority_demotion(self) -> str:
        """Why authority is clamped to L0, or '' when it is not. SCRUM-380."""
        return authority_demotion_reason(
            self.ground_validated_baseline,
            self.envelope_version,
            self.monitor_logic_hash,
        )

    def effective_authority(self) -> str:
        """The authority the machine may actually act on. SCRUM-380.

        Clamped to L0 whenever the envelope version or the monitor logic differs
        from the last ground-validated baseline. L0 is advisory only (section 5),
        so one clamp here blocks every autonomous execution path without each
        guard having to know about baselines.

        A floor, not a preference: no input raises the clamp. Only a ground
        re-validate, which changes the baseline itself.
        """
        if self.authority_demotion():
            return DEMOTED_AUTHORITY
        return self.authority()


# ---------------------------------------------------------------------------
# Guard results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardResult:
    """One guard's verdict, with the values it actually read.

    values is what the guard looked at, not a restatement of the outcome, so an
    audit record can show why a guard failed without re-running it.
    """

    name: str
    passed: bool
    detail: str
    values: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "guard": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "values": dict(self.values),
        }


def first_failure(results: Sequence[GuardResult]) -> Optional[GuardResult]:
    """The first guard that failed, or None when every guard passed."""
    for result in results:
        if not result.passed:
            return result
    return None


# ---------------------------------------------------------------------------
# The authorized-to-execute payload (SCRUM-379 to SCRUM-382)
# ---------------------------------------------------------------------------

# What "approval_basis" may say. Section 3's two M2 to M3 rows, and nothing
# else: a burn is either operator-approved or its L2 veto window expired.
APPROVAL_BASIS_L1_OPERATOR = "l1_operator_approval"
APPROVAL_BASIS_L2_VETO_EXPIRED = "l2_veto_window_expired"


@dataclass(frozen=True)
class AuthorizedExecution:
    """The signal and payload 379 emits on reaching M3, for SCRUM-382.

    Deliberately small and flat. It carries the burn itself (the scorer's dv,
    unchanged -- 379 never modifies a maneuver), the mode and authority context
    that made it lawful, and the two provenance facts an operator would ask
    about first: which covariance the decision rested on, and what the validity
    gate said. 382 assembles the GNC command from this; it should not need to
    reach back into the planner for anything else, and it should not have to
    infer why the burn was authorised.

    Nothing optional and nothing derived: a field that is not known at
    authorisation time does not belong here, because a consumer cannot tell a
    missing value from a real one.
    """

    conjunction_id: str
    dv_eci_km_s: Tuple[float, float, float]
    dv_magnitude_m_s: float
    t_burn_utc: str
    mode: str
    authority_level: str
    envelope_version: str
    approval_basis: str
    validity_status: str
    validity_epsilon: float
    covariance_source: str
    authorized_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conjunction_id": self.conjunction_id,
            "dv_eci_km_s": list(self.dv_eci_km_s),
            "dv_magnitude_m_s": self.dv_magnitude_m_s,
            "t_burn_utc": self.t_burn_utc,
            "mode": self.mode,
            "authority_level": self.authority_level,
            "envelope_version": self.envelope_version,
            "approval_basis": self.approval_basis,
            "validity_status": self.validity_status,
            "validity_epsilon": self.validity_epsilon,
            "covariance_source": self.covariance_source,
            "authorized_at_utc": self.authorized_at_utc,
        }
