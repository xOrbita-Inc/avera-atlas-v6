"""
services/planner/tests/test_evidence_record.py

SCRUM-377: the Evidence Package (MAF v2.0 section 10).

Covers the four things the story actually turns on:
  - a field with no producer is never defaulted, it is marked
  - the chain detects modification
  - the chain detects deletion
  - an undefined transition is recorded as an ESCALATE with a reason

Run with (from repo root -- conftest.py handles PYTHONPATH):
    python -m pytest services/planner/tests/test_evidence_record.py -v
"""

from __future__ import annotations

import json

import pytest

from common.evidence_record import (
    CATALOGUE,
    DEFINED_TRANSITIONS,
    ESCALATE_EVENT,
    FIELD_PRODUCERS,
    FULL_PACKAGE_FIELDS,
    GENESIS_HASH,
    MINIMUM_TRANSITION_FIELDS,
    EvidenceFieldError,
    EvidenceRecord,
    FieldState,
    FlightMode,
    RecordType,
    build_decision_record,
    build_transition_record,
    canonical_json,
    is_defined_transition,
    verify_chain,
)


SW = "2.5.4"


def _chain(n: int = 3):
    """Build a small valid chain: one decision then n-1 transitions."""
    recs = [
        build_decision_record(
            "SAT-A", 0, GENESIS_HASH,
            {"conjunction_id": "C1", "software_version": SW,
             "timestamp": "2026-07-28T00:00:00Z"},
        )
    ]
    hops = [
        (FlightMode.M0_NOMINAL, FlightMode.M1_WATCH, "Pc >= watch line"),
        (FlightMode.M1_WATCH, FlightMode.M2_STAGED, "Pc >= action line"),
        (FlightMode.M2_STAGED, FlightMode.M3_EXECUTING, "veto window closed"),
    ]
    for i in range(1, n):
        frm, to, trig = hops[(i - 1) % len(hops)]
        recs.append(build_transition_record(
            "SAT-A", i, recs[-1].content_hash, frm, to, trig, "C1", SW,
        ))
    return recs


# ---------------------------------------------------------------------------
# The field catalogue
# ---------------------------------------------------------------------------

class TestFieldCatalogue:
    def test_catalogue_covers_maf_section_10_minimum(self):
        """Every minimum-per-transition field from section 10 is present."""
        for name in (
            "event", "from_mode", "to_mode", "trigger", "conjunction_id",
            "timestamp", "pc_at_transition", "validity_status",
            "validity_epsilon", "authority_level", "envelope_version",
            "software_version",
        ):
            assert name in MINIMUM_TRANSITION_FIELDS

    def test_time_sync_quality_is_carried(self):
        """SCRUM-377's original text dropped it from the section 10 list.

        It is in the MAF and is not covered by 378 to 382 either, so it is
        carried here with an explicitly unassigned producer rather than being
        quietly omitted a second time.
        """
        assert "time_sync_quality" in FULL_PACKAGE_FIELDS
        assert FIELD_PRODUCERS["time_sync_quality"] == "unassigned"

    def test_every_catalogue_field_has_a_declared_producer(self):
        assert set(FIELD_PRODUCERS) == set(CATALOGUE)

    def test_unknown_field_is_rejected(self):
        with pytest.raises(EvidenceFieldError):
            EvidenceRecord.build(
                "SAT-A", 0, RecordType.DECISION, GENESIS_HASH,
                values={"not_a_maf_field": 1},
            )

    def test_field_cannot_be_both_valued_and_not_applicable(self):
        with pytest.raises(EvidenceFieldError):
            EvidenceRecord.build(
                "SAT-A", 0, RecordType.DECISION, GENESIS_HASH,
                values={"conjunction_id": "C1"},
                not_applicable=["conjunction_id"],
            )


# ---------------------------------------------------------------------------
# Absent is marked, never defaulted (AC1)
# ---------------------------------------------------------------------------

class TestAbsentFieldsAreMarkedNotDefaulted:
    def test_unproduced_field_is_marked_not_zeroed(self):
        rec = _chain(1)[0]
        entry = rec.fields["validity_epsilon"]
        assert entry["state"] == FieldState.PRODUCER_NOT_IMPLEMENTED
        assert entry["producer"] == "SCRUM-378"
        assert entry["value"] is None

    def test_no_unproduced_field_serializes_as_a_usable_default(self):
        """The failure this guards: a record full of zeros reading as complete.

        Every non-present field must be flagged by state. A consumer that
        checks state can never mistake 0.0, "", or False for a measurement.
        """
        rec = _chain(1)[0]
        blob = json.loads(rec.to_json())
        for name, entry in blob["fields"].items():
            if entry["state"] != FieldState.PRESENT:
                assert entry["value"] is None, (
                    f"{name} is {entry['state']} but carries a value that could "
                    "be mistaken for real data"
                )

    def test_reading_an_unproduced_field_raises_rather_than_returning_none(self):
        rec = _chain(1)[0]
        with pytest.raises(EvidenceFieldError) as e:
            rec.value("monitor_results")
        assert "SCRUM-379" in str(e.value)

    def test_decision_record_marks_mode_fields_not_applicable(self):
        """A decision is not a mode change.

        from_mode absent on a decision record is correct; from_mode absent on a
        transition record means SCRUM-379 has not shipped. The two must not
        look the same.
        """
        dec = _chain(1)[0]
        assert dec.fields["from_mode"]["state"] == FieldState.NOT_APPLICABLE
        trans = build_transition_record(
            "SAT-B", 0, GENESIS_HASH, FlightMode.M0_NOMINAL,
            FlightMode.M1_WATCH, "Pc >= watch line", "C1", SW,
        )
        assert trans.fields["from_mode"]["state"] == FieldState.PRESENT

    def test_pending_producers_lists_outstanding_tickets(self):
        rec = _chain(1)[0]
        pending = rec.pending_producers()
        for ticket in ("SCRUM-375", "SCRUM-378", "SCRUM-379", "SCRUM-382"):
            assert ticket in pending

    def test_completeness_counts_every_catalogue_field(self):
        rec = _chain(1)[0]
        assert sum(rec.completeness().values()) == len(CATALOGUE)


# ---------------------------------------------------------------------------
# Tamper evidence (AC3)
# ---------------------------------------------------------------------------

class TestChainDetectsTampering:
    def test_clean_chain_verifies(self):
        assert verify_chain(_chain(4)).ok is True

    def test_modification_is_detected_and_located(self):
        recs = _chain(4)
        recs[2].fields["trigger"]["value"] = "something else"
        result = verify_chain(recs)
        assert result.ok is False
        assert result.break_seq == 2
        assert "modified" in result.reason

    def test_modification_of_a_value_changes_the_hash(self):
        rec = _chain(1)[0]
        assert rec.hash_matches()
        rec.fields["conjunction_id"]["value"] = "C-OTHER"
        assert not rec.hash_matches()

    def test_deletion_of_a_middle_record_is_detected(self):
        recs = _chain(4)
        without_middle = [recs[0], recs[1], recs[3]]
        result = verify_chain(without_middle)
        assert result.ok is False
        assert result.break_seq == 2
        assert "missing" in result.reason

    def test_deletion_of_the_last_record_leaves_a_valid_prefix(self):
        """Truncation is not detectable from the chain alone.

        Dropping the tail leaves a shorter but internally consistent chain.
        Detecting that needs an anchor outside the store, which is out of scope
        for this story. Asserted here so the limit is recorded rather than
        assumed away.
        """
        recs = _chain(4)
        assert verify_chain(recs[:-1]).ok is True

    def test_reordering_is_detected(self):
        recs = _chain(4)
        recs[1].seq, recs[2].seq = recs[2].seq, recs[1].seq
        assert verify_chain(recs).ok is False

    def test_first_record_must_carry_the_genesis_hash(self):
        bad = build_decision_record(
            "SAT-C", 0, "f" * 64, {"conjunction_id": "C1", "software_version": SW},
        )
        result = verify_chain([bad])
        assert result.ok is False
        assert "genesis" in result.reason

    def test_empty_chain_verifies(self):
        assert verify_chain([]).ok is True

    def test_canonical_json_is_order_independent(self):
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


# ---------------------------------------------------------------------------
# Undefined transitions escalate (AC5)
# ---------------------------------------------------------------------------

class TestUndefinedTransitionEscalates:
    def test_defined_transition_records_normally(self):
        rec = build_transition_record(
            "SAT-D", 0, GENESIS_HASH, FlightMode.M1_WATCH,
            FlightMode.M2_STAGED, "Pc >= action line", "C1", SW,
        )
        assert rec.value("event") == "M1->M2"
        assert rec.value("to_mode") == FlightMode.M2_STAGED

    def test_undefined_transition_is_recorded_as_escalate_to_m4(self):
        rec = build_transition_record(
            "SAT-D", 0, GENESIS_HASH, FlightMode.M0_NOMINAL,
            FlightMode.M3_EXECUTING, "bad guard", "C1", SW,
        )
        assert rec.value("event") == ESCALATE_EVENT
        assert rec.value("to_mode") == FlightMode.M4_SAFE_HOLD

    def test_escalate_record_states_the_reason_and_keeps_the_attempt(self):
        rec = build_transition_record(
            "SAT-D", 0, GENESIS_HASH, FlightMode.M0_NOMINAL,
            FlightMode.M3_EXECUTING, "bad guard", "C1", SW,
        )
        trigger = rec.value("trigger")
        assert "M0->M3" in trigger
        assert "bad guard" in trigger
        assert "section 8" in trigger

    def test_m4_never_self_exits_except_to_m0(self):
        """MAF section 8: M4 exits only on ground clearance to M0."""
        assert is_defined_transition(FlightMode.M4_SAFE_HOLD, FlightMode.M0_NOMINAL)
        for target in (FlightMode.M1_WATCH, FlightMode.M2_STAGED, FlightMode.M3_EXECUTING):
            assert not is_defined_transition(FlightMode.M4_SAFE_HOLD, target)

    def test_every_defined_transition_is_from_the_maf_section_8_table(self):
        assert DEFINED_TRANSITIONS == frozenset({
            ("M0", "M1"), ("M1", "M2"), ("M1", "M4"), ("M2", "M3"),
            ("M2", "M4"), ("M3", "M0"), ("M3", "M4"), ("M4", "M0"),
        })

    def test_unknown_flight_mode_is_rejected(self):
        with pytest.raises(EvidenceFieldError):
            build_transition_record(
                "SAT-D", 0, GENESIS_HASH, "M9", FlightMode.M1_WATCH,
                "t", "C1", SW,
            )


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_round_trip_preserves_the_hash(self):
        rec = _chain(1)[0]
        again = EvidenceRecord.from_dict(json.loads(rec.to_json()))
        assert again.content_hash == rec.content_hash
        assert again.hash_matches()

    def test_record_id_is_chain_and_seq(self):
        rec = _chain(2)[1]
        assert rec.record_id == "SAT-A#1"
