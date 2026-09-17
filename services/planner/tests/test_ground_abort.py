"""
SCRUM-380 -- the ground-commanded emergency abort.

MAF v2.0 section 6: a ground abort must be able to interrupt an in-progress
sequence during a pass. It is therefore the highest-priority input the state
machine takes, evaluated before any guard, and it never auto-clears -- guard doc
section 6.1 already says M4 does not exit autonomously, and an abort is no
exception.

Distinct from execution_status ABORTED, which is a GNC report that a burn did
not complete (guard doc section 3, the M3 to M4 row). That is an outcome; this
is a command.

Anchored to the guard doc and MAF section 6, not to a restatement of the
implementation.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from common.decision_state_machine import FlightMode
from common.mode_persistence import (
    InMemoryModeStore,
    PersistedMode,
    recover_after_reboot,
)
from common.safety_floors import ABORT_EVENT, ABORT_TRIGGER, GroundAbortCommand
from common.safety_monitor import evaluate_safety_monitor

from test_safety_monitor import T_NOW, staged_inputs

ABORT = GroundAbortCommand(
    abort_reason="conjunction superseded by a closer event",
    operator_id="ops-jhavera",
    issued_at_utc="2026-04-01T13:44:00Z",
    command_id="abort-0001",
)


def aborting(**overrides):
    return staged_inputs(ground_abort=ABORT, **overrides)


# ---------------------------------------------------------------------------
# The command itself
# ---------------------------------------------------------------------------


class TestAbortCommand:
    def test_an_abort_must_say_why(self):
        with pytest.raises(ValueError):
            GroundAbortCommand(abort_reason="", operator_id="ops-jhavera")

    def test_an_abort_must_say_who(self):
        with pytest.raises(ValueError):
            GroundAbortCommand(abort_reason="stop", operator_id="")

    def test_a_missing_abort_block_is_simply_no_abort(self):
        assert GroundAbortCommand.from_dict(None) is None
        assert GroundAbortCommand.from_dict({}) is None

    def test_a_malformed_abort_block_raises_rather_than_being_dropped(self):
        """Silently dropping a half-formed abort is the one failure this
        feature exists to prevent."""
        with pytest.raises(ValueError):
            GroundAbortCommand.from_dict({"operator_id": "ops-jhavera"})

    def test_the_audit_entry_carries_the_section_7_fields(self):
        entry = ABORT.audit_entry(from_mode="M3", conjunction_id="conj-2026-0302-001")

        assert entry["event"] == ABORT_EVENT
        assert entry["from_mode"] == "M3"
        assert entry["to_mode"] == "M4"
        assert entry["trigger"] == ABORT_TRIGGER
        assert entry["abort_reason"] == ABORT.abort_reason
        assert entry["operator_id"] == "ops-jhavera"
        assert entry["conjunction_id"] == "conj-2026-0302-001"


# ---------------------------------------------------------------------------
# Preemption: an abort beats the transition table
# ---------------------------------------------------------------------------


class TestAbortPreemptsEverything:
    @pytest.mark.parametrize("mode", ["M1", "M2", "M3"])
    def test_an_abort_forces_m4_from_every_active_mode(self, mode):
        """MAF section 6: the abort must reach M4 from a watch, a staged burn,
        and critically from an execution already under way."""
        decision = evaluate_safety_monitor(aborting(current_mode=mode))

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.aborted is True

    def test_an_abort_interrupts_an_executing_burn(self):
        """The in-progress case the ticket is about: M3 with no GNC report yet
        would otherwise hold M3 while the burn runs."""
        holding = evaluate_safety_monitor(staged_inputs(current_mode="M3"))
        assert holding.mode is FlightMode.M3_EXECUTING  # no abort: burn continues

        decision = evaluate_safety_monitor(aborting(current_mode="M3"))
        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_an_abort_beats_an_otherwise_executable_burn(self):
        """A staged L2 whose veto window has closed would auto-execute on this
        very evaluation. The abort lands first."""
        executable = staged_inputs(
            current_mode="M2",
            veto_window_close_utc=T_NOW - timedelta(seconds=1),
        )
        assert evaluate_safety_monitor(executable).mode is FlightMode.M3_EXECUTING

        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M2",
                veto_window_close_utc=T_NOW - timedelta(seconds=1),
                ground_abort=ABORT,
            )
        )
        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.authorized_execution is None

    def test_an_abort_beats_an_operator_approval_arriving_together(self):
        decision = evaluate_safety_monitor(
            aborting(
                current_mode="M2",
                authority_level="L1",
                approval_command_received=True,
                approval_accepted=True,
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.authorized_execution is None

    def test_an_abort_beats_a_resolved_event(self):
        """Even a conjunction that has cleared does not outrank a ground stop."""
        decision = evaluate_safety_monitor(
            aborting(current_mode="M2", pc=1e-12, consecutive_below_monitor=5)
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_an_abort_from_m0_is_still_honoured(self):
        """Section 3 defines no M0 to M4 row, so this lands as an escalation --
        the right destination, recorded as undefined. A ground stop is never
        ignored for want of a table row."""
        decision = evaluate_safety_monitor(aborting(current_mode="M0"))

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.aborted is True

    def test_the_trigger_names_the_abort_and_its_reason(self):
        decision = evaluate_safety_monitor(aborting(current_mode="M2"))

        assert ABORT_TRIGGER in decision.transition.trigger
        assert "superseded" in decision.transition.trigger

    def test_the_abort_guard_records_who_sent_it(self):
        decision = evaluate_safety_monitor(aborting(current_mode="M2"))
        guard = [g for g in decision.guards if g.name == "no_ground_abort"][0]

        assert guard.passed is False
        assert guard.values["operator_id"] == "ops-jhavera"
        assert guard.values["abort_reason"] == ABORT.abort_reason


# ---------------------------------------------------------------------------
# It never auto-clears
# ---------------------------------------------------------------------------


class TestAbortNeverAutoClears:
    def test_an_abort_while_safeheld_holds_m4(self):
        decision = evaluate_safety_monitor(aborting(current_mode="M4"))

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.changed_mode is False

    def test_a_clean_event_does_not_lift_an_abort(self):
        """Nothing about the conjunction improving clears a ground stop."""
        decision = evaluate_safety_monitor(
            aborting(current_mode="M4", pc=1e-12, consecutive_below_monitor=5)
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_only_ground_clearance_exits_the_abort(self):
        """Guard doc section 3, last row: ground command only, via ARBITER."""
        decision = evaluate_safety_monitor(
            staged_inputs(current_mode="M4", operator_clearance_received=True)
        )

        assert decision.mode is FlightMode.M0_NOMINAL

    def test_a_still_standing_abort_outranks_a_clearance_in_the_same_evaluation(self):
        """An abort and a clearance arriving together is ambiguous, and section 1
        escalates the ambiguous case rather than resolving it in favour of
        resuming flight."""
        decision = evaluate_safety_monitor(
            aborting(current_mode="M4", operator_clearance_received=True)
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD


# ---------------------------------------------------------------------------
# It survives a reboot (guard doc section 6.2)
# ---------------------------------------------------------------------------


class TestAbortPersistence:
    def test_an_abort_is_persisted_with_the_mode(self):
        store = InMemoryModeStore()
        store.write(
            PersistedMode(sat_id="SAT-380", mode="M4", ground_abort=ABORT)
        )

        restored = store.read("SAT-380")
        assert restored.ground_abort is not None
        assert restored.ground_abort.operator_id == "ops-jhavera"

    def test_a_persisted_abort_survives_a_file_round_trip(self, tmp_path):
        from common.mode_persistence import FileModeStore

        store = FileModeStore(tmp_path)
        store.write(PersistedMode(sat_id="SAT-380", mode="M4", ground_abort=ABORT))

        restored = store.read("SAT-380")
        assert restored.ground_abort.abort_reason == ABORT.abort_reason
        assert restored.ground_abort.command_id == "abort-0001"

    def test_reboot_recovery_holds_m4_on_a_persisted_abort(self):
        """Section 6.2 would otherwise re-enter M2 for a still-open burn window.
        An abort outranks that: the burn must not resume because the spacecraft
        rebooted."""
        recovery = recover_after_reboot(
            PersistedMode(
                sat_id="SAT-380",
                mode="M2",
                latest_burn_utc=T_NOW + timedelta(hours=1),
                ground_abort=ABORT,
            ),
            T_NOW,
        )

        assert recovery.mode is FlightMode.M4_SAFE_HOLD
        assert "abort" in recovery.reason.lower()

    def test_reboot_without_an_abort_still_follows_the_section_6_2_ladder(self):
        """The abort branch must not swallow the normal recovery path."""
        recovery = recover_after_reboot(
            PersistedMode(
                sat_id="SAT-380",
                mode="M2",
                latest_burn_utc=T_NOW + timedelta(hours=1),
            ),
            T_NOW,
        )

        assert recovery.mode is FlightMode.M2_STAGED

    def test_a_persisted_abort_holds_m4_even_from_m3(self):
        recovery = recover_after_reboot(
            PersistedMode(
                sat_id="SAT-380",
                mode="M3",
                latest_burn_utc=T_NOW + timedelta(hours=1),
                ground_abort=ABORT,
            ),
            T_NOW,
        )

        assert recovery.mode is FlightMode.M4_SAFE_HOLD


# ---------------------------------------------------------------------------
# It is audit-logged (guard doc section 7)
# ---------------------------------------------------------------------------


class TestAbortAudit:
    def test_the_decision_carries_the_abort_audit_entry(self):
        decision = evaluate_safety_monitor(aborting(current_mode="M3"))

        entry = decision.abort_audit_entry()
        assert entry["event"] == ABORT_EVENT
        assert entry["from_mode"] == "M3"
        assert entry["to_mode"] == "M4"
        assert entry["operator_id"] == "ops-jhavera"
        assert entry["conjunction_id"] == "conj-2026-0302-001"

    def test_no_abort_means_no_abort_entry(self):
        decision = evaluate_safety_monitor(staged_inputs(current_mode="M2"))

        assert decision.aborted is False
        assert decision.abort_audit_entry() is None

    def test_the_abort_appears_in_the_serialised_decision(self):
        payload = evaluate_safety_monitor(aborting(current_mode="M2")).to_dict()

        assert payload["aborted"] is True
        assert payload["abort"]["operator_id"] == "ops-jhavera"
        assert payload["to_mode"] == "M4"
