"""
SCRUM-375 — MAF authorization-envelope compiler and authority-ladder enforcement.

Locked values are specification values from SCRUM-375/SCRUM-333. They are not
calibrated or measured by this implementation. Pc_watch is derived only by the
locked formula max(Pc_action / 10, Pc_watch_min). Capability inputs may only
make the effective envelope more restrictive.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

LOCKED_PC_ACTION_CEILING = 1.0e-4
LOCKED_MIN_MISS_DISTANCE_KM = 1.0
LOCKED_MIN_TCA_HOURS = 4.0
LOCKED_MAX_TCA_HOURS = 72.0
LOCKED_L1_DV_CAP_M_S = 2.0
LOCKED_L2_DV_CAP_M_S = 0.5
LOCKED_L2_RAISED_DV_CAP_M_S = 1.0
LOCKED_MAX_MANEUVERS_PER_WEEK = 3
LOCKED_RESERVED_DV_M_S = 5.0
HMAC_SHA256_ALG = "HMAC-SHA256"
VERSION_PREFIX = "env-v1-sha256"


class EnvelopeCompileError(ValueError):
    pass


class EnvelopeApprovalError(ValueError):
    pass


class EnvelopeActivationError(ValueError):
    pass


class AuthorityLevel(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"


_AUTHORITY_RANK = {
    AuthorityLevel.L0: 0,
    AuthorityLevel.L1: 1,
    AuthorityLevel.L2: 2,
}


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RiskBudgetInput:
    risk_budget_id: str
    pc_watch_min: float

    def __post_init__(self) -> None:
        if not self.risk_budget_id.strip():
            raise EnvelopeCompileError("risk_budget_id must be non-empty")
        if not math.isfinite(self.pc_watch_min) or self.pc_watch_min <= 0.0:
            raise EnvelopeCompileError("pc_watch_min must be finite and > 0")


@dataclass(frozen=True)
class ManeuverCapacityInput:
    max_dv_per_burn_m_s: float
    v_remaining_m_s: float
    v_reserved_m_s: float = LOCKED_RESERVED_DV_M_S
    max_maneuvers_per_week: int = LOCKED_MAX_MANEUVERS_PER_WEEK

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_dv_per_burn_m_s) or self.max_dv_per_burn_m_s <= 0.0:
            raise EnvelopeCompileError("max_dv_per_burn_m_s must be finite and > 0")
        if not math.isfinite(self.v_remaining_m_s) or self.v_remaining_m_s < 0.0:
            raise EnvelopeCompileError("v_remaining_m_s must be finite and >= 0")
        if not math.isfinite(self.v_reserved_m_s) or self.v_reserved_m_s < 0.0:
            raise EnvelopeCompileError("v_reserved_m_s must be finite and >= 0")
        if self.v_reserved_m_s > self.v_remaining_m_s:
            raise EnvelopeCompileError("v_reserved_m_s must be <= v_remaining_m_s")
        if self.max_maneuvers_per_week < 1:
            raise EnvelopeCompileError("max_maneuvers_per_week must be >= 1")


@dataclass(frozen=True)
class OperatorAuthorizationProfile:
    profile_id: str
    operator_id: str
    mission_class: str
    risk_budget: RiskBudgetInput
    maneuver_capacity: ManeuverCapacityInput
    pc_action: float
    authority_level: AuthorityLevel
    l2_raise_to_1_m_s: bool = False

    def __post_init__(self) -> None:
        for field_name, value in (
            ("profile_id", self.profile_id),
            ("operator_id", self.operator_id),
            ("mission_class", self.mission_class),
        ):
            if not value.strip():
                raise EnvelopeCompileError(f"{field_name} must be non-empty")
        if not isinstance(self.authority_level, AuthorityLevel):
            try:
                object.__setattr__(self, "authority_level", AuthorityLevel(self.authority_level))
            except (TypeError, ValueError) as exc:
                raise EnvelopeCompileError("authority_level must be one of L0, L1, or L2") from exc
        if not math.isfinite(self.pc_action) or self.pc_action <= 0.0:
            raise EnvelopeCompileError("pc_action must be finite and > 0")
        if self.pc_action > LOCKED_PC_ACTION_CEILING:
            raise EnvelopeCompileError("pc_action exceeds the locked 1e-4 first-flight ceiling")


@dataclass(frozen=True)
class CompiledAuthorizationEnvelope:
    profile_id: str
    operator_id: str
    mission_class: str
    risk_budget_id: str
    authority_level: AuthorityLevel
    pc_action: float
    pc_watch_min: float
    pc_watch: float
    min_miss_distance_km: float
    min_tca_hours: float
    max_tca_hours: float
    l1_dv_cap_m_s: float
    l2_dv_cap_m_s: float
    max_maneuvers_per_week: int
    reserved_dv_m_s: float
    capability_max_dv_per_burn_m_s: float
    capability_v_remaining_m_s: float
    capability_v_reserved_m_s: float
    capability_max_maneuvers_per_week: int
    effective_l1_dv_cap_m_s: float
    effective_l2_dv_cap_m_s: float
    effective_max_maneuvers_per_week: int
    effective_reserved_dv_m_s: float
    available_dv_after_reserve_m_s: float
    content_hash: str
    version: str

    def canonical_payload(self) -> dict:
        payload = asdict(self)
        payload["authority_level"] = self.authority_level.value
        payload.pop("content_hash", None)
        payload.pop("version", None)
        return payload

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["authority_level"] = self.authority_level.value
        return payload

    def evidence_fields(self, approving_identity: Optional[str] = None) -> dict:
        return {
            "authority_level": self.authority_level.value,
            "envelope_version": self.version,
            "envelope_approving_identity": approving_identity,
        }


def compile_authorization_envelope(profile: OperatorAuthorizationProfile) -> CompiledAuthorizationEnvelope:
    pc_watch = max(profile.pc_action / 10.0, profile.risk_budget.pc_watch_min)
    if pc_watch >= profile.pc_action:
        raise EnvelopeCompileError("compiled Pc_watch must be strictly below Pc_action")

    locked_l2_cap = (
        LOCKED_L2_RAISED_DV_CAP_M_S
        if profile.l2_raise_to_1_m_s
        else LOCKED_L2_DV_CAP_M_S
    )
    capacity = profile.maneuver_capacity
    effective_reserved = max(LOCKED_RESERVED_DV_M_S, capacity.v_reserved_m_s)
    available_after_reserve = max(0.0, capacity.v_remaining_m_s - effective_reserved)

    unsigned = {
        "profile_id": profile.profile_id,
        "operator_id": profile.operator_id,
        "mission_class": profile.mission_class,
        "risk_budget_id": profile.risk_budget.risk_budget_id,
        "authority_level": profile.authority_level.value,
        "pc_action": profile.pc_action,
        "pc_watch_min": profile.risk_budget.pc_watch_min,
        "pc_watch": pc_watch,
        "min_miss_distance_km": LOCKED_MIN_MISS_DISTANCE_KM,
        "min_tca_hours": LOCKED_MIN_TCA_HOURS,
        "max_tca_hours": LOCKED_MAX_TCA_HOURS,
        "l1_dv_cap_m_s": LOCKED_L1_DV_CAP_M_S,
        "l2_dv_cap_m_s": locked_l2_cap,
        "max_maneuvers_per_week": LOCKED_MAX_MANEUVERS_PER_WEEK,
        "reserved_dv_m_s": LOCKED_RESERVED_DV_M_S,
        "capability_max_dv_per_burn_m_s": capacity.max_dv_per_burn_m_s,
        "capability_v_remaining_m_s": capacity.v_remaining_m_s,
        "capability_v_reserved_m_s": capacity.v_reserved_m_s,
        "capability_max_maneuvers_per_week": capacity.max_maneuvers_per_week,
        "effective_l1_dv_cap_m_s": min(LOCKED_L1_DV_CAP_M_S, capacity.max_dv_per_burn_m_s, available_after_reserve),
        "effective_l2_dv_cap_m_s": min(locked_l2_cap, capacity.max_dv_per_burn_m_s, available_after_reserve),
        "effective_max_maneuvers_per_week": min(LOCKED_MAX_MANEUVERS_PER_WEEK, capacity.max_maneuvers_per_week),
        "effective_reserved_dv_m_s": effective_reserved,
        "available_dv_after_reserve_m_s": available_after_reserve,
    }
    content_hash = _sha256_hex(_canonical_json(unsigned))
    version = f"{VERSION_PREFIX}-{content_hash[:12]}"

    compiled_values = dict(unsigned)
    compiled_values["authority_level"] = AuthorityLevel(
        unsigned["authority_level"]
    )
    return CompiledAuthorizationEnvelope(
        **compiled_values,
        content_hash=content_hash,
        version=version,
    )


def validate_section6_safety_floors(policy: Any) -> None:
    """Reject operator-policy values that are laxer than locked first-flight limits."""
    checks = (
        (
            policy.pc_maneuver_threshold <= LOCKED_PC_ACTION_CEILING,
            "pc_maneuver_threshold exceeds the locked 1e-4 ceiling",
        ),
        (
            policy.min_miss_distance_km >= LOCKED_MIN_MISS_DISTANCE_KM,
            "min_miss_distance_km is below the locked 1.0 km floor",
        ),
        (
            policy.max_dv_per_event_ms <= LOCKED_L1_DV_CAP_M_S,
            "max_dv_per_event_ms exceeds the locked 2.0 m/s L1 cap",
        ),
        (
            policy.max_maneuvers_per_week <= LOCKED_MAX_MANEUVERS_PER_WEEK,
            "max_maneuvers_per_week exceeds the locked 3/week limit",
        ),
        (
            policy.min_hours_before_tca >= LOCKED_MIN_TCA_HOURS,
            "min_hours_before_tca is below the locked 4 hour floor",
        ),
        (
            policy.max_hours_before_tca <= LOCKED_MAX_TCA_HOURS,
            "max_hours_before_tca exceeds the locked 72 hour horizon",
        ),
    )
    for passed, message in checks:
        if not passed:
            raise EnvelopeCompileError(message)


def compile_from_operator_policy(
    policy: Any,
    *,
    profile_id: str,
    mission_class: str,
    risk_budget_id: str,
    maneuver_capacity: ManeuverCapacityInput,
    authority_level: AuthorityLevel,
    l2_raise_to_1_m_s: bool = False,
) -> CompiledAuthorizationEnvelope:
    """Bridge the existing OperatorPolicy model into the locked compiler."""
    validate_section6_safety_floors(policy)
    profile = OperatorAuthorizationProfile(
        profile_id=profile_id,
        operator_id=policy.operator_id,
        mission_class=mission_class,
        risk_budget=RiskBudgetInput(
            risk_budget_id=risk_budget_id,
            pc_watch_min=policy.pc_monitor_threshold,
        ),
        maneuver_capacity=maneuver_capacity,
        pc_action=policy.pc_maneuver_threshold,
        authority_level=authority_level,
        l2_raise_to_1_m_s=l2_raise_to_1_m_s,
    )
    return compile_authorization_envelope(profile)


@dataclass(frozen=True)
class ApprovedAuthorizationEnvelope:
    """
    Explicitly approved envelope plus its authorization authenticator.

    First flight uses a keyed HMAC-SHA256 MAC. The serialized proof fields are
    algorithm-neutral (alg, key_id, auth_value) so a later asymmetric verifier
    such as Ed25519 can use the same envelope format without renaming fields.
    """

    envelope: CompiledAuthorizationEnvelope
    approving_identity: str
    approved_at_utc: str
    human_review_confirmed: bool
    alg: str
    key_id: str
    auth_value: str

    def mac_payload(self) -> dict:
        """
        Canonical material authenticated by the first-flight keyed MAC.

        The MAC binds the canonical compiled envelope, its deterministic
        version, approving identity, approval metadata, algorithm, and key id.
        """
        return {
            "canonical_envelope": self.envelope.canonical_payload(),
            "envelope_version": self.envelope.version,
            "approving_identity": self.approving_identity,
            "approved_at_utc": self.approved_at_utc,
            "human_review_confirmed": self.human_review_confirmed,
            "alg": self.alg,
            "key_id": self.key_id,
        }

    def to_dict(self) -> dict:
        return {
            "envelope": self.envelope.to_dict(),
            "approving_identity": self.approving_identity,
            "approved_at_utc": self.approved_at_utc,
            "human_review_confirmed": self.human_review_confirmed,
            "alg": self.alg,
            "key_id": self.key_id,
            "auth_value": self.auth_value,
        }


def approve_authorization_envelope(
    envelope: CompiledAuthorizationEnvelope,
    *,
    approving_identity: str,
    mac_key: bytes,
    key_id: str,
    human_review_confirmed: bool,
    approved_at: datetime,
) -> ApprovedAuthorizationEnvelope:
    """Record explicit human approval and authenticate it with a keyed MAC."""
    if not approving_identity.strip():
        raise EnvelopeApprovalError("approving_identity must be non-empty")
    if not human_review_confirmed:
        raise EnvelopeApprovalError("activation requires confirmed human review")
    if not mac_key:
        raise EnvelopeApprovalError("mac_key must be non-empty")
    if not key_id.strip():
        raise EnvelopeApprovalError("key_id must be non-empty")

    approved_at_utc = _utc_iso(approved_at)
    provisional = ApprovedAuthorizationEnvelope(
        envelope=envelope,
        approving_identity=approving_identity,
        approved_at_utc=approved_at_utc,
        human_review_confirmed=True,
        alg=HMAC_SHA256_ALG,
        key_id=key_id,
        auth_value="",
    )
    auth_value = hmac.new(
        mac_key,
        _canonical_json(provisional.mac_payload()).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return ApprovedAuthorizationEnvelope(
        envelope=envelope,
        approving_identity=approving_identity,
        approved_at_utc=approved_at_utc,
        human_review_confirmed=True,
        alg=HMAC_SHA256_ALG,
        key_id=key_id,
        auth_value=auth_value,
    )


def verify_approved_envelope(
    approved: ApprovedAuthorizationEnvelope,
    *,
    mac_key: bytes,
    expected_key_id: Optional[str] = None,
) -> bool:
    """Verify the first-flight HMAC-SHA256 keyed MAC."""
    if (
        approved.alg != HMAC_SHA256_ALG
        or not approved.human_review_confirmed
        or not approved.key_id
        or not mac_key
    ):
        return False
    if expected_key_id is not None and approved.key_id != expected_key_id:
        return False

    expected = hmac.new(
        mac_key,
        _canonical_json(approved.mac_payload()).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, approved.auth_value)


@dataclass(frozen=True)
class ActiveAuthorizationEnvelope:
    approved: ApprovedAuthorizationEnvelope
    activation_time_utc: str

    def to_dict(self) -> dict:
        return {
            "approved": self.approved.to_dict(),
            "activation_time_utc": self.activation_time_utc,
        }


class AuthorizationEnvelopeManager:
    def __init__(self) -> None:
        self._pending: Optional[CompiledAuthorizationEnvelope] = None
        self._approved_pending: Optional[ApprovedAuthorizationEnvelope] = None
        self._active: Optional[ActiveAuthorizationEnvelope] = None
        self._history: list[CompiledAuthorizationEnvelope] = []

    @property
    def pending(self) -> Optional[CompiledAuthorizationEnvelope]:
        return self._pending

    @property
    def active(self) -> Optional[ActiveAuthorizationEnvelope]:
        return self._active

    def stage(self, envelope: CompiledAuthorizationEnvelope) -> None:
        same_as_active = (
            self._active is not None
            and self._active.approved.envelope.version == envelope.version
            and self._active.approved.envelope.content_hash == envelope.content_hash
        )
        if same_as_active:
            self._pending = envelope
            return
        if self._active is not None:
            self._history.append(self._active.approved.envelope)
        self._pending = envelope
        self._approved_pending = None
        self._active = None

    def record_approval(self, approved: ApprovedAuthorizationEnvelope, *, mac_key: bytes) -> None:
        if self._pending is None:
            raise EnvelopeActivationError("no compiled envelope is staged")
        if approved.envelope.version != self._pending.version or approved.envelope.content_hash != self._pending.content_hash:
            raise EnvelopeActivationError("approval does not match the staged envelope")
        if not verify_approved_envelope(approved, mac_key=mac_key):
            raise EnvelopeApprovalError("authorization-envelope keyed MAC verification failed")
        self._approved_pending = approved

    def activate(self, *, activation_time: datetime) -> ActiveAuthorizationEnvelope:
        if self._approved_pending is None:
            raise EnvelopeActivationError("explicit approved envelope is required before activation")
        active = ActiveAuthorizationEnvelope(
            approved=self._approved_pending,
            activation_time_utc=_utc_iso(activation_time),
        )
        self._active = active
        self._pending = None
        self._approved_pending = None
        return active

    def effective_authority(self, requested: AuthorityLevel, *, at_time: datetime) -> AuthorityLevel:
        if not isinstance(requested, AuthorityLevel):
            try:
                requested = AuthorityLevel(requested)
            except (TypeError, ValueError) as exc:
                raise EnvelopeActivationError("requested authority must be L0, L1, or L2") from exc
        if self._active is None:
            return AuthorityLevel.L0
        if at_time.tzinfo is None:
            raise EnvelopeActivationError("at_time must be timezone-aware")
        activation_time = datetime.fromisoformat(self._active.activation_time_utc.replace("Z", "+00:00"))
        if at_time.astimezone(timezone.utc) < activation_time:
            return AuthorityLevel.L0
        envelope_level = self._active.approved.envelope.authority_level
        permitted_rank = min(_AUTHORITY_RANK[requested], _AUTHORITY_RANK[envelope_level])
        for level, rank in _AUTHORITY_RANK.items():
            if rank == permitted_rank:
                return level
        return AuthorityLevel.L0

    def rollback(self) -> CompiledAuthorizationEnvelope:
        if not self._history:
            raise EnvelopeActivationError("no previous authorization envelope is available")
        previous = self._history.pop()
        self._active = None
        self._approved_pending = None
        self._pending = previous
        return previous