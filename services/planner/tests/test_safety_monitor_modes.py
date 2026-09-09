"""
SCRUM-379 -- mode-level behaviour of the safety monitor.

test_safety_monitor.py checks the individual guards. This file checks the
transition rows of docs/scrum-333/state_machine_guards.md section 3 end to end,
plus the two scenarios the story calls out by name: a live-style event
(covariance_source real_cdm, IOD CONFIDENT, validity EARNED, envelope L2)
driven all the way to an execute emission, and its mirror that forces M4 on a
NOT_EARNED re-check.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from common.decision_state_machine import (
    APPROVAL_BASIS_L1_OPERATOR,
    APPROVAL_BASIS_L2_VETO_EXPIRED,
    FlightMode,
    ValidityRouting,
)
from common.safety_monitor import evaluate_safety_monitor

from test_safety_monitor import (  # reuse the one baseline event
    EPSILON,
    PC_AT_TRANSITION,
    T_NOW,
    a_command,
    staged_inputs,
)


def staged_l2(**overrides):
    """The baseline event already staged in M2 with an open L2 veto window."""
    base = dict(
        current_mode=FlightMode.M2_STAGED,
        authority_level="L2",
        veto_window_close_utc=T_NOW + timedelta(seconds=60),
    )
    base.update(overrides)
    return staged_inputs(**base)


# ---------------------------------------------------------------------------
# M0
# ---------------------------------------------------------------------------


class TestM0:
    def test_a_quiet_sky_holds_m0(self):
        decision = evaluate_safety_monitor(staged_inputs(current_mode="M0", pc=1e-9))

        assert decision.mode is FlightMode.M0_NOMINAL
        assert decision.changed_mode is False
        assert decision.escalated is False

    def test_pc_at_the_monitor_line_flags_a_watch(self):
        decision = evaluate_safety_monitor(staged_inputs(current_mode="M0", pc=1.0e-5))

        assert decision.mode is FlightMode.M1_WATCH
        assert decision.transition.trigger == "pc_above_monitor_threshold"

    def test_m0_does_not_skip_straight_to_staging(self):
        """Section 3 has no M0 to M2 row even when every staging guard holds."""
        decision = evaluate_safety_monitor(staged_inputs(current_mode="M0"))

        assert decision.mode is FlightMode.M1_WATCH


# ---------------------------------------------------------------------------
# M1
# ---------------------------------------------------------------------------


class TestM1:
    def test_the_full_and_stages_the_burn(self):
        decision = evaluate_safety_monitor(staged_inputs())

        assert decision.mode is FlightMode.M2_STAGED
        assert decision.transition.trigger == "m1_to_m2_guards_all_satisfied"
        assert decision.authorized_execution is None  # staging is not authorising

    @pytest.mark.parametrize(
        "override",
        [
            {"iod_proceeds_to_validity": False, "iod_confidence_verdict": "DEGRADED"},
            {"iod_proceeds_to_validity": False, "iod_confidence_verdict": "REJECTED"},
            {"validity_routing": ValidityRouting.BLOCKED},
            {"validity_service_available": False},
            {"secondary_conjunction_clear": False},
            {"t_ca_utc": T_NOW + timedelta(hours=3)},
        ],
    )
    def test_each_m1_to_m4_clause_escalates(self, override):
        """Section 3, M1 to M4: IOD DEGRADED or REJECTED OR validity NOT_EARNED
        OR secondary conflict not clear OR TCA inside the 4 h floor with Pc
        still above the monitor line."""
        decision = evaluate_safety_monitor(staged_inputs(**override))

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.escalated is False  # M1 to M4 is a defined transition

    def test_an_unevaluated_gate_blocks_staging_without_safeholding(self):
        """The interpretive call in _m1_escalation_guards, stated as a test.

        A watch event whose validity gate has not run yet must not stage, and
        must not safehold either: absence is not a negative verdict.
        """
        decision = evaluate_safety_monitor(
            staged_inputs(validity_routing=None, validity_evidence={})
        )

        assert decision.mode is FlightMode.M1_WATCH
        assert decision.changed_mode is False
        assert "validity_earned" in [g.name for g in decision.guards if not g.passed]

    def test_two_quiet_evaluations_return_to_m0(self):
        decision = evaluate_safety_monitor(
            staged_inputs(pc=1e-9, consecutive_below_monitor=2)
        )

        assert decision.mode is FlightMode.M0_NOMINAL

    def test_one_quiet_evaluation_holds_m1(self):
        decision = evaluate_safety_monitor(
            staged_inputs(pc=1e-9, consecutive_below_monitor=1)
        )

        assert decision.mode is FlightMode.M1_WATCH


# ---------------------------------------------------------------------------
# M2
# ---------------------------------------------------------------------------


class TestM2Escalations:
    def test_validity_lost_mid_staging_escalates_immediately(self):
        """Section 4.4: 'If a new CDM arrives and validity drops to NOT_EARNED,
        transition to M4 immediately regardless of veto window status.'"""
        decision = evaluate_safety_monitor(
            staged_l2(
                validity_routing=ValidityRouting.BLOCKED,
                validity_evidence={"validity_status": "NOT_EARNED", "validity_epsilon": 0.11},
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.authorized_execution is None

    def test_validity_lost_escalates_even_after_the_window_closed(self):
        """No grace: the window having closed does not buy an execution."""
        decision = evaluate_safety_monitor(
            staged_l2(
                veto_window_close_utc=T_NOW - timedelta(seconds=1),
                validity_routing=ValidityRouting.BLOCKED,
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_slew_infeasible_while_staged_escalates(self):
        decision = evaluate_safety_monitor(
            staged_l2(command=a_command(t_burn=T_NOW + timedelta(seconds=100)))
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_an_envelope_violation_while_staged_escalates(self):
        decision = evaluate_safety_monitor(staged_l2(dv_cap_m_s=0.1))

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_a_rejected_l1_approval_escalates(self):
        decision = evaluate_safety_monitor(
            staged_l2(
                authority_level="L1",
                veto_window_close_utc=None,
                approval_command_received=True,
                approval_accepted=False,
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_an_expired_watchdog_escalates(self):
        decision = evaluate_safety_monitor(staged_l2(watchdog_expired=True))

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_staged_with_no_authority_is_ambiguous_and_escalates(self):
        decision = evaluate_safety_monitor(staged_l2(authority_level="L0"))

        assert decision.mode is FlightMode.M4_SAFE_HOLD


class TestM2Dynamics:
    def test_an_event_that_resolves_while_staged_returns_to_m0(self):
        decision = evaluate_safety_monitor(
            staged_l2(pc=1e-9, consecutive_below_monitor=2)
        )

        assert decision.mode is FlightMode.M0_NOMINAL
        assert decision.authorized_execution is None

    def test_an_accepted_veto_with_a_fresh_earned_cdm_restages(self):
        decision = evaluate_safety_monitor(
            staged_l2(
                veto_command_received=True,
                veto_accepted=True,
                fresh_cdm_available=True,
                data_age_s=600.0,
            )
        )

        assert decision.mode is FlightMode.M2_STAGED
        assert decision.escalated is False
        assert decision.authorized_execution is None

    def test_a_veto_with_no_fresh_cdm_escalates(self):
        """Section 8: 'Otherwise go to M4.'"""
        decision = evaluate_safety_monitor(
            staged_l2(veto_command_received=True, veto_accepted=True,
                      fresh_cdm_available=False)
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_a_veto_beats_an_expired_window(self):
        """An accepted veto that arrives as the window closes must not race the
        auto-execute."""
        decision = evaluate_safety_monitor(
            staged_l2(
                veto_window_close_utc=T_NOW - timedelta(seconds=1),
                veto_command_received=True,
                veto_accepted=True,
                fresh_cdm_available=True,
                data_age_s=600.0,
            )
        )

        assert decision.mode is FlightMode.M2_STAGED
        assert decision.authorized_execution is None

    def test_l2_holds_while_the_veto_window_is_still_open(self):
        decision = evaluate_safety_monitor(staged_l2())

        assert decision.mode is FlightMode.M2_STAGED
        assert decision.changed_mode is False
        assert decision.authorized_execution is None

    def test_l1_holds_until_the_operator_approves(self):
        decision = evaluate_safety_monitor(
            staged_l2(authority_level="L1", veto_window_close_utc=None)
        )

        assert decision.mode is FlightMode.M2_STAGED
        assert decision.changed_mode is False


# ---------------------------------------------------------------------------
# M2 to M3: the execute emission
# ---------------------------------------------------------------------------


class TestExecuteEmission:
    def test_a_live_l2_event_auto_executes_when_the_window_closes(self):
        """The story's headline case: covariance_source real_cdm, IOD
        CONFIDENT, validity EARNED, envelope L2, veto window expired."""
        decision = evaluate_safety_monitor(
            staged_l2(veto_window_close_utc=T_NOW - timedelta(seconds=1))
        )

        assert decision.mode is FlightMode.M3_EXECUTING
        assert decision.escalated is False
        assert decision.authorized_execution is not None

    def test_the_payload_carries_the_scorers_dv_unchanged(self):
        command = a_command(dv_m_s=0.42)
        decision = evaluate_safety_monitor(
            staged_l2(command=command, veto_window_close_utc=T_NOW - timedelta(seconds=1))
        )

        payload = decision.authorized_execution
        assert payload.dv_eci_km_s == command.dv_eci_km_s
        assert payload.dv_magnitude_m_s == 0.42
        assert payload.t_burn_utc == "2026-04-01T15:45:00Z"

    def test_the_payload_carries_the_mode_and_authority_context(self):
        decision = evaluate_safety_monitor(
            staged_l2(veto_window_close_utc=T_NOW - timedelta(seconds=1))
        )

        payload = decision.authorized_execution
        assert payload.mode == "M3"
        assert payload.authority_level == "L2"
        assert payload.approval_basis == APPROVAL_BASIS_L2_VETO_EXPIRED
        assert payload.envelope_version == "env-v1-sha256-abc123456789"
        assert payload.covariance_source == "real_cdm"
        assert payload.validity_status == "EARNED"
        assert payload.validity_epsilon == EPSILON
        assert payload.conjunction_id == "conj-2026-0302-001"
        assert payload.authorized_at_utc == "2026-04-01T13:45:00Z"

    def test_the_payload_serialises_flat_for_382(self):
        decision = evaluate_safety_monitor(
            staged_l2(veto_window_close_utc=T_NOW - timedelta(seconds=1))
        )

        as_dict = decision.authorized_execution.to_dict()
        assert set(as_dict) == {
            "conjunction_id", "dv_eci_km_s", "dv_magnitude_m_s", "t_burn_utc",
            "mode", "authority_level", "envelope_version", "approval_basis",
            "validity_status", "validity_epsilon", "covariance_source",
            "authorized_at_utc",
        }
        assert as_dict["dv_eci_km_s"] == list(decision.authorized_execution.dv_eci_km_s)

    def test_an_l1_burn_executes_on_operator_approval(self):
        decision = evaluate_safety_monitor(
            staged_l2(
                authority_level="L1",
                dv_cap_m_s=2.0,
                veto_window_close_utc=None,
                approval_command_received=True,
                approval_accepted=True,
            )
        )

        assert decision.mode is FlightMode.M3_EXECUTING
        assert decision.authorized_execution.approval_basis == APPROVAL_BASIS_L1_OPERATOR
        assert decision.authorized_execution.authority_level == "L1"

    def test_a_surrogate_covariance_never_auto_executes_at_l2(self):
        """The mirror of the live case: identical event, surrogate covariance.

        Section 4.1 wants an operator, so the closed window escalates rather
        than executing.
        """
        decision = evaluate_safety_monitor(
            staged_l2(
                covariance_source="surrogate_identity",
                veto_window_close_utc=T_NOW - timedelta(seconds=1),
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.authorized_execution is None

    def test_a_not_earned_recheck_forces_m4_instead_of_executing(self):
        """The story's mirror case: the same live event, but the re-check on a
        new CDM came back NOT_EARNED."""
        decision = evaluate_safety_monitor(
            staged_l2(
                veto_window_close_utc=T_NOW - timedelta(seconds=1),
                validity_routing=ValidityRouting.BLOCKED,
                validity_evidence={
                    "validity_status": "NOT_EARNED",
                    "validity_epsilon": 0.11,
                    "epsilon_threshold": 0.20,
                },
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.authorized_execution is None
        assert "validity_earned" in [g.name for g in decision.guards if not g.passed]

    def test_the_execute_transition_records_the_continuously_monitored_guards(self):
        decision = evaluate_safety_monitor(
            staged_l2(veto_window_close_utc=T_NOW - timedelta(seconds=1))
        )

        names = [g.name for g in decision.guards]
        assert "slew_feasible" in names
        assert "validity_earned" in names
        assert "envelope_satisfied" in names
        assert all(g.passed for g in decision.guards)


# ---------------------------------------------------------------------------
# M3
# ---------------------------------------------------------------------------


class TestM3:
    def test_a_burn_in_progress_holds_m3(self):
        decision = evaluate_safety_monitor(staged_inputs(current_mode="M3"))

        assert decision.mode is FlightMode.M3_EXECUTING
        assert decision.changed_mode is False

    def test_a_nominal_burn_that_clears_the_event_returns_to_m0(self):
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M3", gnc_report_received=True,
                execution_status="NOMINAL", pc=1e-9, m2_post=30.0,
            )
        )

        assert decision.mode is FlightMode.M0_NOMINAL

    def test_a_nominal_burn_with_pc_still_elevated_requires_a_replan(self):
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M3", gnc_report_received=True,
                execution_status="NOMINAL", pc=PC_AT_TRANSITION, m2_post=30.0,
            )
        )

        assert decision.mode is FlightMode.M1_WATCH
        assert decision.transition.trigger == "post_burn_replan_required"

    def test_a_nominal_burn_that_did_not_reach_m2_safe_escalates(self):
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M3", gnc_report_received=True,
                execution_status="NOMINAL", pc=1e-9, m2_post=25.0,
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    @pytest.mark.parametrize("status", ["ABORTED", "PARTIAL"])
    def test_a_non_nominal_burn_escalates(self, status):
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M3", gnc_report_received=True,
                execution_status=status, residual_pc_elevated=True,
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_a_non_nominal_burn_escalates_even_with_no_residual_risk_recorded(self):
        """Section 3 gives no M3 to M0 row for a burn that did not complete, and
        section 1 escalates the unspecified case rather than inventing one."""
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M3", gnc_report_received=True,
                execution_status="ABORTED", residual_pc_elevated=False,
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD


# ---------------------------------------------------------------------------
# M4
# ---------------------------------------------------------------------------


class TestM4:
    def test_m4_does_not_exit_autonomously(self):
        """Section 6.1: 'Hold safehold. Do not exit M4 autonomously.'"""
        decision = evaluate_safety_monitor(staged_inputs(current_mode="M4"))

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.changed_mode is False

    def test_a_perfectly_clean_event_still_does_not_lift_a_safehold(self):
        decision = evaluate_safety_monitor(
            staged_inputs(current_mode="M4", pc=1e-9, consecutive_below_monitor=5)
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_ground_clearance_returns_to_m0(self):
        decision = evaluate_safety_monitor(
            staged_inputs(current_mode="M4", operator_clearance_received=True)
        )

        assert decision.mode is FlightMode.M0_NOMINAL
        assert decision.transition.trigger == "operator_clearance_via_arbiter"


# ---------------------------------------------------------------------------
# Section 6.3, from any mode
# ---------------------------------------------------------------------------


class TestSubsystemFailureFromAnyMode:
    @pytest.mark.parametrize("mode", ["M0", "M1", "M2", "M3"])
    def test_a_subsystem_failure_goes_to_m4_from_any_mode(self, mode):
        """Section 6.3: 'Navigation filter degraded (EKF divergence) | Any |
        Go to M4. Cannot certify state estimate.'"""
        decision = evaluate_safety_monitor(
            staged_inputs(current_mode=mode, subsystem_failure="navigation_filter_degraded")
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.authorized_execution is None

    def test_m0_to_m4_is_recorded_as_an_escalation(self):
        """Section 3 has no M0 to M4 row, so the destination is right and the
        record says the transition itself was undefined. The failure reason is
        preserved in the trigger rather than being thrown away."""
        decision = evaluate_safety_monitor(
            staged_inputs(current_mode="M0", subsystem_failure="navigation_filter_degraded")
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.escalated is True
        assert "navigation_filter_degraded" in decision.transition.trigger

    def test_m1_to_m4_on_a_failure_is_a_defined_transition(self):
        decision = evaluate_safety_monitor(
            staged_inputs(current_mode="M1", subsystem_failure="thruster_fault")
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.escalated is False

    def test_a_failure_while_already_safeheld_holds_rather_than_re_escalating(self):
        decision = evaluate_safety_monitor(
            staged_inputs(current_mode="M4", subsystem_failure="thruster_fault")
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.escalated is False
        assert decision.changed_mode is False

    def test_a_failure_beats_an_otherwise_executable_burn(self):
        decision = evaluate_safety_monitor(
            staged_l2(
                veto_window_close_utc=T_NOW - timedelta(seconds=1),
                subsystem_failure="ATTITUDE_ERROR_EXCEEDED",
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.authorized_execution is None
