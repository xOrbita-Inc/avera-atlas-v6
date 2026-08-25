"""
services/planner/common/evidence_record.py

SCRUM-377: the Evidence Package. Implements MAF v2.0 section 10.

Every autonomous decision and every state transition produces a tamper-evident,
append-only record. This module owns the record schema, the field catalogue, the
hash chain that makes the record tamper-evident, and the transition-recording
entry point that SCRUM-379 will consume.

Why the field catalogue exists
------------------------------
MAF section 10 specifies a field set that spans work landing across several
sprints. Most of those producers do not exist yet. The tempting shortcut is to
emit only the fields we can fill and add the rest later, but then a record
written today is indistinguishable from a complete one, and an operator (or us,
in six months) reading an audit trail full of zeros has no way to tell "the
monitor said nothing was wrong" from "no monitor existed."

So the catalogue carries every section 10 field from the start, and every field
in a record carries an explicit state:

    present                   -- a real value, produced by a real producer
    not_applicable            -- meaningfully absent for this record type
    producer_not_implemented  -- the producing component has not shipped yet

A field is never silently defaulted to 0, "", None, or False. Building a record
with an unknown field name, or omitting a catalogue field without a state, is an
error rather than a quiet gap.

This is the SCRUM-364 lesson applied. The NASA daily worklist looked like a
complete conjunction record right up until someone checked what was in it.

Tamper evidence
---------------
Records are chained. Each record stores the SHA-256 of its own canonical
serialization (content_hash) and the content_hash of the previous record in the
same chain (prev_hash). Chains are keyed per satellite.

verify_chain() recomputes the chain and reports the first break. This detects
modification (content_hash no longer matches the record) and deletion (a seq gap,
and the prev_hash linkage breaks). It deliberately does NOT use cryptographic
signing: MAF section 5 requires signing for the authorization envelope, which is
SCRUM-375's job, whereas section 10 asks only that the package be tamper-evident.
A hash chain gets detection with no key management, and a dropped record shows up
as a chain break rather than as silence.

Scope note
----------
This story delivers the substrate. Population of the validity, monitor, command
and post-burn fields is carried by SCRUM-378 through 382. See FIELD_PRODUCERS.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Field states
# ---------------------------------------------------------------------------

class FieldState:
    """How to read a field's value. Never inferred, always explicit."""

    PRESENT = "present"
    NOT_APPLICABLE = "not_applicable"
    PRODUCER_NOT_IMPLEMENTED = "producer_not_implemented"

    ALL = (PRESENT, NOT_APPLICABLE, PRODUCER_NOT_IMPLEMENTED)


# ---------------------------------------------------------------------------
# Record types and flight modes (MAF v2.0 section 4)
# ---------------------------------------------------------------------------

class RecordType:
    DECISION = "decision"
    TRANSITION = "transition"

    ALL = (DECISION, TRANSITION)


class FlightMode:
    """MAF v2.0 section 4. Operating / flight modes M0 to M4."""

    M0_NOMINAL = "M0"
    M1_WATCH = "M1"
    M2_STAGED = "M2"
    M3_EXECUTING = "M3"
    M4_SAFE_HOLD = "M4"

    ALL = (M0_NOMINAL, M1_WATCH, M2_STAGED, M3_EXECUTING, M4_SAFE_HOLD)


# MAF v2.0 section 8. The transitions the state machine defines.
# Anything not in this set is undefined and escalates to M4 with a reason.
DEFINED_TRANSITIONS: frozenset = frozenset({
    (FlightMode.M0_NOMINAL, FlightMode.M1_WATCH),
    (FlightMode.M1_WATCH, FlightMode.M2_STAGED),
    (FlightMode.M1_WATCH, FlightMode.M4_SAFE_HOLD),
    (FlightMode.M2_STAGED, FlightMode.M3_EXECUTING),
    (FlightMode.M2_STAGED, FlightMode.M4_SAFE_HOLD),
    (FlightMode.M3_EXECUTING, FlightMode.M0_NOMINAL),
    (FlightMode.M3_EXECUTING, FlightMode.M4_SAFE_HOLD),
    (FlightMode.M4_SAFE_HOLD, FlightMode.M0_NOMINAL),
})

ESCALATE_EVENT = "ESCALATE"


# ---------------------------------------------------------------------------
# The field catalogue (MAF v2.0 section 10)
# ---------------------------------------------------------------------------

# Minimum fields per transition, quoted from section 10.
MINIMUM_TRANSITION_FIELDS: Tuple[str, ...] = (
    "event",
    "from_mode",
    "to_mode",
    "trigger",
    "conjunction_id",
    "timestamp",
    "pc_at_transition",
    "validity_status",
    "validity_epsilon",
    "epsilon_threshold",
    "phenomenologies_used",
    "authority_level",
    "envelope_version",
    "software_version",
)

# The remainder of the full package, also from section 10.
FULL_PACKAGE_FIELDS: Tuple[str, ...] = (
    "inputs_and_provenance",
    "weak_directions",
    "time_sync_quality",
    "orbit_state",
    "covariance_state",
    "model_version",
    "policy_version",
    "envelope_approving_identity",
    "candidate_maneuvers",
    "monitor_results",
    "commands_and_acknowledgments",
    "actual_vs_predicted",
    "post_maneuver_od",
    "residual_risk",
)

CATALOGUE: Tuple[str, ...] = MINIMUM_TRANSITION_FIELDS + FULL_PACKAGE_FIELDS

# Which ticket populates each field. "main" means a producer already exists on
# main today. This is the map that keeps the gaps visible instead of implied,
# and it is the thing to update as 378 through 382 land.
#
# time_sync_quality has no owning ticket. It is in MAF section 10 but was
# dropped from the SCRUM-377 transcription and is not covered by 378 to 382
# either. Recorded here as unassigned rather than quietly omitted.
FIELD_PRODUCERS: Dict[str, str] = {
    "event": "SCRUM-379",
    "from_mode": "SCRUM-379",
    "to_mode": "SCRUM-379",
    "trigger": "SCRUM-379",
    "conjunction_id": "main",
    "timestamp": "main",
    "pc_at_transition": "main",
    "validity_status": "SCRUM-378",
    "validity_epsilon": "SCRUM-378",
    "epsilon_threshold": "SCRUM-378",
    "phenomenologies_used": "SCRUM-378",
    "authority_level": "SCRUM-375",
    "envelope_version": "SCRUM-375",
    "software_version": "main",
    "inputs_and_provenance": "main",
    "weak_directions": "SCRUM-378",
    "time_sync_quality": "unassigned",
    "orbit_state": "main",
    "covariance_state": "main",
    "model_version": "main",
    "policy_version": "main",
    "envelope_approving_identity": "SCRUM-375",
    "candidate_maneuvers": "main",
    "monitor_results": "SCRUM-379",
    "commands_and_acknowledgments": "SCRUM-382",
    "actual_vs_predicted": "SCRUM-382",
    "post_maneuver_od": "SCRUM-382",
    "residual_risk": "SCRUM-382",
}

# Every catalogue field must have a declared producer. Guards against a field
# being added to the catalogue without anyone deciding who fills it.
assert set(FIELD_PRODUCERS) == set(CATALOGUE), (
    "FIELD_PRODUCERS and CATALOGUE are out of sync: "
    f"{set(FIELD_PRODUCERS) ^ set(CATALOGUE)}"
)


class EvidenceFieldError(ValueError):
    """Raised when a record would be built with an unknown or invalid field."""


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------

def canonical_json(payload: Any) -> str:
    """Deterministic JSON for hashing.

    Sorted keys and no incidental whitespace, so two logically identical
    records hash identically regardless of construction order.

    Note on floats: values are serialised with repr semantics via json, which
    is stable within a Python version for the same value. Records therefore
    hash reproducibly, but a float re-derived through a different computation
    path may not reproduce byte-identically. Producers should record the value
    they actually used, not recompute it at verification time.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


GENESIS_HASH = "0" * 64


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# EvidenceRecord
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRecord:
    """One append-only, tamper-evident audit record.

    Attributes
    ----------
    record_id : str
        Unique record ID, '{chain_id}#{seq}'.
    chain_id : str
        Chain this record belongs to. Keyed per satellite.
    seq : int
        Position in the chain, starting at 0. Contiguous by construction, so a
        gap is evidence of deletion.
    record_type : str
        RecordType.DECISION or RecordType.TRANSITION.
    recorded_at : str
        ISO-8601 UTC timestamp the record was created.
    fields : dict
        Every CATALOGUE field, each as
        {"state": <FieldState>, "value": <any>, "producer": <ticket>}.
    prev_hash : str
        content_hash of the previous record in this chain, or GENESIS_HASH.
    content_hash : str
        SHA-256 over the canonical serialization of everything above.
    """

    record_id: str
    chain_id: str
    seq: int
    record_type: str
    recorded_at: str
    fields: Dict[str, Dict[str, Any]]
    prev_hash: str
    content_hash: str = ""

    # -- construction -------------------------------------------------------

    @classmethod
    def build(
        cls,
        chain_id: str,
        seq: int,
        record_type: str,
        prev_hash: str,
        values: Optional[Dict[str, Any]] = None,
        not_applicable: Iterable[str] = (),
        recorded_at: Optional[str] = None,
    ) -> "EvidenceRecord":
        """Build a record, filling the catalogue explicitly.

        Parameters
        ----------
        values : dict
            Field name to real value, for fields with a producer that exists.
            Every key must be in CATALOGUE.
        not_applicable : iterable of str
            Catalogue fields that are meaningfully absent for this record,
            as distinct from fields whose producer has not shipped.

        Any catalogue field not named in `values` or `not_applicable` is marked
        PRODUCER_NOT_IMPLEMENTED with its producer from FIELD_PRODUCERS. There
        is no path by which a field silently becomes 0, "", None, or False.
        """
        if record_type not in RecordType.ALL:
            raise EvidenceFieldError(f"unknown record_type: {record_type!r}")

        values = dict(values or {})
        na = set(not_applicable)

        unknown = (set(values) | na) - set(CATALOGUE)
        if unknown:
            raise EvidenceFieldError(
                f"field(s) not in the MAF section 10 catalogue: {sorted(unknown)}"
            )
        overlap = set(values) & na
        if overlap:
            raise EvidenceFieldError(
                f"field(s) both valued and not_applicable: {sorted(overlap)}"
            )

        built: Dict[str, Dict[str, Any]] = {}
        for name in CATALOGUE:
            producer = FIELD_PRODUCERS[name]
            if name in values:
                built[name] = {
                    "state": FieldState.PRESENT,
                    "value": values[name],
                    "producer": producer,
                }
            elif name in na:
                built[name] = {
                    "state": FieldState.NOT_APPLICABLE,
                    "value": None,
                    "producer": producer,
                }
            else:
                built[name] = {
                    "state": FieldState.PRODUCER_NOT_IMPLEMENTED,
                    "value": None,
                    "producer": producer,
                }

        rec = cls(
            record_id=f"{chain_id}#{seq}",
            chain_id=chain_id,
            seq=seq,
            record_type=record_type,
            recorded_at=recorded_at or _utc_now_iso(),
            fields=built,
            prev_hash=prev_hash,
        )
        rec.content_hash = rec.compute_hash()
        return rec

    # -- hashing ------------------------------------------------------------

    def _hashable_payload(self) -> Dict[str, Any]:
        """Everything the content_hash covers. Excludes content_hash itself."""
        return {
            "record_id": self.record_id,
            "chain_id": self.chain_id,
            "seq": self.seq,
            "record_type": self.record_type,
            "recorded_at": self.recorded_at,
            "fields": self.fields,
            "prev_hash": self.prev_hash,
        }

    def compute_hash(self) -> str:
        return _sha256(canonical_json(self._hashable_payload()))

    def hash_matches(self) -> bool:
        """True when the stored content_hash still matches the content."""
        return self.content_hash == self.compute_hash()

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        payload = self._hashable_payload()
        payload["content_hash"] = self.content_hash
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceRecord":
        return cls(
            record_id=d["record_id"],
            chain_id=d["chain_id"],
            seq=int(d["seq"]),
            record_type=d["record_type"],
            recorded_at=d["recorded_at"],
            fields=d["fields"],
            prev_hash=d["prev_hash"],
            content_hash=d.get("content_hash", ""),
        )

    # -- convenience --------------------------------------------------------

    def value(self, name: str) -> Any:
        """Return a field's value, or raise if it is not PRESENT.

        Deliberately strict. Callers must not read a value out of a field whose
        producer has not shipped, because the answer would be None and would
        read as a real measurement.
        """
        if name not in self.fields:
            raise EvidenceFieldError(f"unknown field: {name!r}")
        entry = self.fields[name]
        if entry["state"] != FieldState.PRESENT:
            raise EvidenceFieldError(
                f"field {name!r} is {entry['state']} (producer {entry['producer']}), "
                "not a value you can read"
            )
        return entry["value"]

    def completeness(self) -> Dict[str, int]:
        """Count of catalogue fields by state. For operator display."""
        counts = {s: 0 for s in FieldState.ALL}
        for entry in self.fields.values():
            counts[entry["state"]] += 1
        return counts

    def pending_producers(self) -> List[str]:
        """Distinct tickets still owed for this record, sorted."""
        return sorted({
            entry["producer"]
            for entry in self.fields.values()
            if entry["state"] == FieldState.PRODUCER_NOT_IMPLEMENTED
        })


# ---------------------------------------------------------------------------
# Chain verification
# ---------------------------------------------------------------------------

@dataclass
class ChainVerification:
    """Result of walking a chain.

    ok : bool
        True when every record hashes correctly and links to its predecessor.
    break_seq : int or None
        Sequence number of the first record that failed, if any.
    reason : str
        Human-readable description of the first break.
    checked : int
        Number of records examined.
    """

    ok: bool
    break_seq: Optional[int] = None
    reason: str = ""
    checked: int = 0


def verify_chain(records: Sequence[EvidenceRecord]) -> ChainVerification:
    """Verify an ordered run of records from one chain.

    Detects, in order of the first failure found:
      - modification: a record whose content no longer matches its content_hash
      - deletion: a gap in the seq run, or a prev_hash that does not match the
        preceding record's content_hash
      - a chain that does not start at the genesis hash

    Records are expected in ascending seq order. An empty run verifies as ok.
    """
    if not records:
        return ChainVerification(ok=True, checked=0)

    ordered = sorted(records, key=lambda r: r.seq)
    expected_prev = GENESIS_HASH
    expected_seq = ordered[0].seq

    if expected_seq == 0 and ordered[0].prev_hash != GENESIS_HASH:
        return ChainVerification(
            ok=False,
            break_seq=0,
            reason="first record does not carry the genesis prev_hash",
            checked=1,
        )
    if expected_seq != 0:
        # Partial run. Trust the first record's prev_hash as the anchor and
        # verify everything after it.
        expected_prev = ordered[0].prev_hash

    for i, rec in enumerate(ordered):
        if rec.seq != expected_seq:
            return ChainVerification(
                ok=False,
                break_seq=expected_seq,
                reason=(
                    f"sequence gap: expected seq {expected_seq}, found {rec.seq}. "
                    "A record is missing from this chain."
                ),
                checked=i,
            )
        if not rec.hash_matches():
            return ChainVerification(
                ok=False,
                break_seq=rec.seq,
                reason=(
                    f"record {rec.record_id} has been modified: stored content_hash "
                    "does not match its content"
                ),
                checked=i + 1,
            )
        if rec.prev_hash != expected_prev:
            return ChainVerification(
                ok=False,
                break_seq=rec.seq,
                reason=(
                    f"record {rec.record_id} does not link to its predecessor: "
                    "prev_hash mismatch. A record was removed or reordered."
                ),
                checked=i + 1,
            )
        expected_prev = rec.content_hash
        expected_seq += 1

    return ChainVerification(ok=True, checked=len(ordered))


# ---------------------------------------------------------------------------
# Transition recording (the interface SCRUM-379 consumes)
# ---------------------------------------------------------------------------

def is_defined_transition(from_mode: str, to_mode: str) -> bool:
    return (from_mode, to_mode) in DEFINED_TRANSITIONS


def build_transition_record(
    chain_id: str,
    seq: int,
    prev_hash: str,
    from_mode: str,
    to_mode: str,
    trigger: str,
    conjunction_id: str,
    software_version: str,
    event: str = "",
    pc_at_transition: Optional[float] = None,
    extra_values: Optional[Dict[str, Any]] = None,
    recorded_at: Optional[str] = None,
) -> EvidenceRecord:
    """Build the record for one state transition.

    MAF v2.0 section 8: any undefined or ambiguous transition defaults to
    ESCALATE (M4). That rule lives here rather than in the state machine, so
    SCRUM-379 inherits it instead of reimplementing it, and so an undefined
    transition can never be recorded as though it were a normal one.

    An undefined transition is recorded with event ESCALATE, to_mode forced to
    M4, and a trigger that states what was attempted and why it escalated. The
    attempted destination is preserved in the trigger text rather than being
    thrown away.
    """
    for mode, label in ((from_mode, "from_mode"), (to_mode, "to_mode")):
        if mode not in FlightMode.ALL:
            raise EvidenceFieldError(f"{label} is not a known flight mode: {mode!r}")

    values: Dict[str, Any] = dict(extra_values or {})

    if is_defined_transition(from_mode, to_mode):
        values["event"] = event or f"{from_mode}->{to_mode}"
        values["from_mode"] = from_mode
        values["to_mode"] = to_mode
        values["trigger"] = trigger
    else:
        values["event"] = ESCALATE_EVENT
        values["from_mode"] = from_mode
        values["to_mode"] = FlightMode.M4_SAFE_HOLD
        values["trigger"] = (
            f"undefined transition {from_mode}->{to_mode} attempted "
            f"(trigger: {trigger or 'unspecified'}); "
            "MAF v2.0 section 8 escalates to M4"
        )

    values["conjunction_id"] = conjunction_id
    values["software_version"] = software_version
    if pc_at_transition is not None:
        values["pc_at_transition"] = pc_at_transition
    values.setdefault("timestamp", recorded_at or _utc_now_iso())

    return EvidenceRecord.build(
        chain_id=chain_id,
        seq=seq,
        record_type=RecordType.TRANSITION,
        prev_hash=prev_hash,
        values=values,
        recorded_at=recorded_at,
    )


def build_decision_record(
    chain_id: str,
    seq: int,
    prev_hash: str,
    values: Dict[str, Any],
    not_applicable: Iterable[str] = (),
    recorded_at: Optional[str] = None,
) -> EvidenceRecord:
    """Build the record for one autonomous decision.

    A decision record is not a mode change, so the transition-shaped fields are
    marked NOT_APPLICABLE rather than left as producer_not_implemented. That
    distinction matters: from_mode being absent on a decision record is correct,
    whereas from_mode being absent on a transition record means SCRUM-379 has
    not shipped.
    """
    na = set(not_applicable) | {"from_mode", "to_mode"}
    na -= set(values)
    return EvidenceRecord.build(
        chain_id=chain_id,
        seq=seq,
        record_type=RecordType.DECISION,
        prev_hash=prev_hash,
        values=values,
        not_applicable=na,
        recorded_at=recorded_at,
    )
