"""
SCRUM-380 -- MAF v2.0 section 6 safety floors, refusal gates, and the
ground-commanded emergency abort.

SCRUM-379 built the guard floors and locked the envelope values; this module is
the delta on top. It holds the data types and the pure logic for the four things
379 does not do:

  1. A ground-commanded emergency abort that preempts the transition table.
  2. Baseline tracking, so an envelope or monitor-logic change demotes authority
     to L0 until ground re-validates.
  3. A record-quality floor that rejects a degraded or zero-filled CDM outright
     rather than parsing it and scoring a hollow record.
  4. The L3 refusal gate: no L3 authorization without a pre-verified safe action.

Nothing here is tunable. Every floor SCRUM-379 locked stays locked; a threshold
change is an escalation to Minh, not a config knob, and there is deliberately no
way to pass a looser value through this module.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

# ---------------------------------------------------------------------------
# Ground-commanded emergency abort
# ---------------------------------------------------------------------------

ABORT_TRIGGER = "ground_commanded_abort"
ABORT_EVENT = "GROUND_ABORT"


@dataclass(frozen=True)
class GroundAbortCommand:
    """A ground abort. The highest-priority input the state machine takes.

    Distinct from execution_status ABORTED, which is a GNC *report* that a burn
    did not complete. This is a command from the ground saying stop, and it is
    evaluated before any guard so it can interrupt a staged or executing
    sequence mid-pass.

    operator_id and abort_reason are required rather than optional: an abort
    that cannot say who sent it or why is not auditable, and section 7 wants
    both in the log entry.
    """

    abort_reason: str
    operator_id: str
    issued_at_utc: str = ""
    command_id: str = ""

    def __post_init__(self) -> None:
        if not str(self.abort_reason).strip():
            raise ValueError("a ground abort must carry an abort_reason")
        if not str(self.operator_id).strip():
            raise ValueError("a ground abort must carry an operator_id")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "abort_reason": self.abort_reason,
            "operator_id": self.operator_id,
            "issued_at_utc": self.issued_at_utc,
            "command_id": self.command_id,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> Optional["GroundAbortCommand"]:
        """Build from a request or persisted block, or None if there isn't one.

        Returns None for a missing block, but raises for a block that is present
        and malformed. A half-formed abort is a bug worth surfacing, not an
        abort to quietly drop -- dropping it would be the one failure mode this
        whole feature exists to prevent.
        """
        if not payload:
            return None
        return cls(
            abort_reason=str(payload.get("abort_reason", "")),
            operator_id=str(payload.get("operator_id", "")),
            issued_at_utc=str(payload.get("issued_at_utc", "")),
            command_id=str(payload.get("command_id", "")),
        )

    def audit_entry(self, from_mode: str, conjunction_id: str = "") -> Dict[str, Any]:
        """The section 7 shape for an abort, for the tamper-evident chain."""
        return {
            "event": ABORT_EVENT,
            "from_mode": from_mode,
            "to_mode": "M4",
            "trigger": ABORT_TRIGGER,
            "abort_reason": self.abort_reason,
            "operator_id": self.operator_id,
            "issued_at_utc": self.issued_at_utc,
            "command_id": self.command_id,
            "conjunction_id": conjunction_id,
        }


# ---------------------------------------------------------------------------
# Authority baseline: envelope / monitor-logic change demotes to L0
# ---------------------------------------------------------------------------

# The level an unvalidated baseline is clamped to. L0 is advisory only (guard
# doc section 5), so this blocks every autonomous execution path without
# needing to touch the individual guards.
DEMOTED_AUTHORITY = "L0"

BASELINE_TRIGGER = "authority_baseline_unvalidated"

# The modules whose content defines "the monitor logic". A change to any of them
# changes the hash and demotes to L0 until ground re-validates.
_MONITOR_LOGIC_MODULES = (
    "decision_state_machine.py",
    "safety_monitor.py",
    "safety_floors.py",
)

_UNREADABLE_LOGIC_HASH = "unreadable"


@dataclass(frozen=True)
class AuthorityBaseline:
    """What ground last validated: an envelope version and a monitor-logic hash.

    Both, not either. An operator can swap the envelope without touching the
    software, and software can be changed without touching the envelope; either
    on its own is a new configuration ground has not signed off.
    """

    envelope_version: str
    monitor_logic_hash: str
    validated_by: str = ""
    validated_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "envelope_version": self.envelope_version,
            "monitor_logic_hash": self.monitor_logic_hash,
            "validated_by": self.validated_by,
            "validated_at_utc": self.validated_at_utc,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> Optional["AuthorityBaseline"]:
        if not payload:
            return None
        return cls(
            envelope_version=str(payload.get("envelope_version", "")),
            monitor_logic_hash=str(payload.get("monitor_logic_hash", "")),
            validated_by=str(payload.get("validated_by", "")),
            validated_at_utc=str(payload.get("validated_at_utc", "")),
        )


def monitor_logic_hash(module_dir: Optional[Path] = None) -> str:
    """A content hash over the modules that decide whether a burn may happen.

    Source content, not a declared version string: a version number is something
    an editor can forget to bump, and this floor exists precisely for the case
    where the logic changed without anyone saying so. Any edit to a monitor
    module therefore changes this hash and demotes authority to L0 until ground
    re-validates -- including an edit that only touches a comment. That is the
    intended cost of a tamper floor, and re-validating is one command.

    An unreadable module hashes to a sentinel that matches no baseline, so a
    monitor whose own source cannot be read runs at L0 rather than assuming it
    is the same software ground signed off.
    """
    directory = module_dir or Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _MONITOR_LOGIC_MODULES:
        try:
            digest.update(name.encode("utf-8"))
            digest.update((directory / name).read_bytes())
        except OSError:
            return _UNREADABLE_LOGIC_HASH
    return digest.hexdigest()


def baseline_enforced(baseline: Optional[AuthorityBaseline]) -> bool:
    """Whether baseline control is engaged for this evaluation.

    The floor is CHANGE detection, so it needs a baseline to have changed from.
    With no baseline on record the floor does not fire and authority is whatever
    the envelope granted.

    This is the one deliberately non-fail-closed decision in SCRUM-380, and it is
    worth stating plainly. Demoting on a missing baseline would be the stricter
    reading, but it would also clamp every caller that has not yet adopted
    baselines to L0 -- which is not a safety win, it is an outage that operators
    would route around. Ground establishing a baseline is what arms the floor;
    from that moment any drift in the envelope or the monitor logic demotes to
    L0 until ground re-validates.
    """
    return baseline is not None


def baseline_matches(
    baseline: Optional[AuthorityBaseline],
    envelope_version: Optional[str],
    logic_hash: Optional[str],
) -> bool:
    """Whether the running configuration is the one ground validated.

    Strict: once a baseline exists, anything that is not demonstrably the same
    configuration is a mismatch. A missing envelope version or an unreadable
    monitor hash cannot show the configuration is unchanged, so neither passes.
    """
    if baseline is None:
        return False
    if not envelope_version or not logic_hash:
        return False
    if logic_hash == _UNREADABLE_LOGIC_HASH:
        return False
    return (
        baseline.envelope_version == envelope_version
        and baseline.monitor_logic_hash == logic_hash
    )


def authority_demotion_reason(
    baseline: Optional[AuthorityBaseline],
    envelope_version: Optional[str],
    logic_hash: Optional[str],
) -> str:
    """Why authority must be clamped to L0, or '' when it need not be.

    Empty for an unarmed floor (no baseline) and for a configuration that
    matches the baseline. Non-empty names what drifted, for the audit entry.
    """
    if not baseline_enforced(baseline):
        return ""
    if baseline_matches(baseline, envelope_version, logic_hash):
        return ""
    return baseline_change_reason(baseline, envelope_version, logic_hash)


def baseline_change_reason(
    baseline: Optional[AuthorityBaseline],
    envelope_version: Optional[str],
    logic_hash: Optional[str],
) -> str:
    """Why the baseline does not match, for the audit entry. '' when it does."""
    if baseline is None:
        return "no ground-validated baseline on record"
    if not envelope_version:
        return "no active envelope version to compare against the baseline"
    if not logic_hash or logic_hash == _UNREADABLE_LOGIC_HASH:
        return "monitor logic hash unavailable"
    changed = []
    if baseline.envelope_version != envelope_version:
        changed.append(
            f"envelope version changed from {baseline.envelope_version} to {envelope_version}"
        )
    if baseline.monitor_logic_hash != logic_hash:
        changed.append(
            f"monitor logic hash changed from {baseline.monitor_logic_hash[:12]} "
            f"to {logic_hash[:12]}"
        )
    return "; ".join(changed)


def revalidated_baseline(
    envelope_version: str,
    logic_hash: str,
    validated_by: str,
    validated_at_utc: str,
) -> AuthorityBaseline:
    """The new baseline a ground re-validate command establishes.

    Re-baselining is deliberately an explicit ground action and takes an
    identity: the whole point of the floor is that the spacecraft cannot decide
    on its own that a new configuration is acceptable.
    """
    if not str(validated_by).strip():
        raise ValueError("a ground re-validate must carry the validating identity")
    return AuthorityBaseline(
        envelope_version=envelope_version,
        monitor_logic_hash=logic_hash,
        validated_by=validated_by,
        validated_at_utc=validated_at_utc,
    )


# ---------------------------------------------------------------------------
# Degraded / zero-filled record rejection
# ---------------------------------------------------------------------------

RECORD_QUALITY_TRIGGER = "cdm_record_rejected"


def _finite_numbers(values: Optional[Iterable[Any]]) -> Optional[list]:
    if values is None:
        return None
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if not out or any(not math.isfinite(v) for v in out):
        return None
    return out


def cdm_record_rejection_reason(
    p_rel_km2: Optional[Sequence[Any]] = None,
    r_rel_km: Optional[Sequence[Any]] = None,
    *,
    declared_degraded: bool = False,
) -> str:
    """Why this record must be refused outright, or '' if it is usable.

    The floor is reject-not-parse: a hollow record must not reach the guards at
    all. A zero-filled covariance does not mean "no uncertainty", it means the
    record carries no covariance, and scoring it produces a confident-looking Pc
    computed from nothing. That is far more dangerous than having no record.

    Deliberately NOT the same thing as the artifact's covariance_quality, which
    reads 'degraded' or 'dilution_region' for a real covariance in an awkward
    geometry. Those are legitimate, scorable conjunctions. This function is
    about a record that is structurally empty or corrupt.
    """
    if declared_degraded:
        return "record flagged degraded by its producer"

    if p_rel_km2 is not None:
        values = _finite_numbers(p_rel_km2)
        if values is None:
            return "relative covariance is missing or non-finite"
        if all(v == 0.0 for v in values):
            return "relative covariance is zero-filled"
        size = int(round(math.sqrt(len(values))))
        if size * size == len(values):
            diagonal = [values[i * size + i] for i in range(size)]
            if any(v <= 0.0 for v in diagonal):
                return "relative covariance has a non-positive variance on its diagonal"

    if r_rel_km is not None:
        values = _finite_numbers(r_rel_km)
        if values is None:
            return "relative position is missing or non-finite"
        if all(v == 0.0 for v in values):
            return "relative position is zero-filled"

    return ""


# ---------------------------------------------------------------------------
# L3 refusal gate (stub floor -- the L3 execution path is a later phase)
# ---------------------------------------------------------------------------

L3_AUTHORITY = "L3"
L3_TRIGGER = "l3_requires_pre_verified_safe_action"


def l3_refusal_reason(
    requested_authority: str, pre_verified_safe_action: Optional[str]
) -> str:
    """Why an L3 request is refused, or '' when the floor does not apply.

    Guard doc section 5 gates L3 to a later phase ("no veto window, pre-authorized,
    single-shot"). The only part of L3 that exists today is this refusal: an L3
    authorization without a pre-verified safe action attached is refused. Nothing
    here executes an L3 maneuver, and a passing check is not permission to -- it
    only means the floor did not fire.
    """
    if str(requested_authority or "").upper() != L3_AUTHORITY:
        return ""
    if not str(pre_verified_safe_action or "").strip():
        return (
            "L3 authorization refused: no pre-verified safe action attached "
            "(L3 execution is gated to a later phase)"
        )
    return ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
