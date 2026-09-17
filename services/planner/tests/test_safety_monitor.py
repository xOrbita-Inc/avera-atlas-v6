"""
SCRUM-379 -- guard-by-guard tests for the MAF v2.0 section 8 safety monitor.

Every expectation is anchored to docs/scrum-333/state_machine_guards.md. The
numbers are written out literally (1e-5, 1e-4, 120 s, 30 s, 300 s, 4.0 h, 25,
86400 s, 2 evaluations) rather than imported from the module under test, so a
retuned constant fails a test instead of silently redefining the spec. The
locked values themselves are separately asserted in
test_decision_state_machine.py.

The section 3 example values are the doc's own audit-log sample:
pc_at_transition 0.000312, epsilon 0.73, authority L2.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from common.decision_state_machine import (
    FlightMode,
    GuardInputs,
    ManeuverCommand,
    ValidityRouting,
)
from common.safety_monitor import (
    evaluate_safety_monitor,
    guard_authority_l1_or_l2,
    guard_covariance_provenance,
    guard_envelope_satisfied,
    guard_execution_nominal,
    guard_iod_confident,
    guard_l1_approval_accepted,
    guard_l2_veto_window_expired,
    guard_m2_post_safe,
    guard_pc_above_action,
    guard_pc_above_monitor,
    guard_pc_below_monitor_sustained,
    guard_reveto_may_restage,
    guard_secondary_clear,
    guard_slew_feasible,
    guard_subsystem_healthy,
    guard_tca_above_floor,
    guard_validity_earned,
    m1_to_m2_guards,
)

# Doc section 7, the audit-log example.
T_NOW = datetime(2026, 4, 1, 13, 45, 0, tzinfo=timezone.utc)
PC_AT_TRANSITION = 0.000312
EPSILON = 0.73

PC_MONITOR = 1.0e-5      # section 3, pc_monitor_threshold
PC_MANEUVER = 1.0e-4     # section 3, pc_maneuver_threshold

VALIDITY_EARNED_EVIDENCE = {
    "validity_status": "EARNED",
    "validity_epsilon": EPSILON,
    "epsilon_threshold": 0.20,
    "weak_directions": [],
    "phenomenologies_used": ["TLE"],
}


def a_command(t_burn: datetime | None = None, dv_m_s: float = 0.35) -> ManeuverCommand:
    return ManeuverCommand(
        dv_eci_km_s=(0.0, dv_m_s / 1000.0, 0.0),
        dv_magnitude_m_s=dv_m_s,
        t_burn_utc=t_burn or (T_NOW + timedelta(hours=2)),
        direction="along_track",
    )


def staged_inputs(**overrides) -> GuardInputs:
    """An L2 live-style event with every M1 to M2 guard satisfied.

    covariance_source real_cdm, IOD CONFIDENT, validity EARNED and AUTONOMOUS,
    envelope L2 with dv inside the 0.5 m/s cap. Individual guard tests break
    exactly one field at a time from this baseline.
    """
    base = dict(
        conjunction_id="conj-2026-0302-001",
        current_mode=FlightMode.M1_WATCH,
        t_now_utc=T_NOW,
        pc=PC_AT_TRANSITION,
        pc_source="cdm",
        pc_monitor_threshold=PC_MONITOR,
        pc_maneuver_threshold=PC_MANEUVER,
        command=a_command(),
        latest_burn_utc=T_NOW + timedelta(hours=3),
        t_ca_utc=T_NOW + timedelta(hours=6),
        iod_proceeds_to_validity=True,
        iod_confidence_verdict="CONFIDENT",
        validity_routing=ValidityRouting.AUTONOMOUS,
        validity_evidence=VALIDITY_EARNED_EVIDENCE,
        secondary_check_performed=True,
        secondary_conjunction_clear=True,
        authority_level="L2",
        envelope_version="env-v1-sha256-abc123456789",
        dv_cap_m_s=0.5,
        v_remaining_m_s=20.0,
        v_reserved_m_s=5.0,
        covariance_source="real_cdm",
        data_age_s=3600.0,
    )
    base.update(overrides)
    return GuardInputs(**base)


# ---------------------------------------------------------------------------
# Section 3: Pc guards
# ---------------------------------------------------------------------------


class TestPcGuards:
    def test_pc_at_the_monitor_line_exactly_passes(self):
        # Section 3 writes 'Pc >= pc_monitor_threshold (1e-5)'. Inclusive.
        inputs = staged_inputs(current_mode="M0", pc=1.0e-5)
        assert guard_pc_above_monitor(inputs).passed is True

    def test_pc_just_below_the_monitor_line_fails(self):
        inputs = staged_inputs(current_mode="M0", pc=0.99e-5)
        assert guard_pc_above_monitor(inputs).passed is False

    def test_pc_at_the_action_line_exactly_passes(self):
        # Section 3: 'Pc >= pc_maneuver_threshold (1e-4)'.
        assert guard_pc_above_action(staged_inputs(pc=1.0e-4)).passed is True

    def test_pc_just_below_the_action_line_fails(self):
        assert guard_pc_above_action(staged_inputs(pc=0.99e-4)).passed is False

    def test_an_unresolved_pc_fails_both_pc_guards(self):
        inputs = staged_inputs(pc=None)
        assert guard_pc_above_monitor(inputs).passed is False
        assert guard_pc_above_action(inputs).passed is False

    def test_the_guard_records_pc_and_its_source(self):
        result = guard_pc_above_action(staged_inputs())
        assert result.values["pc"] == PC_AT_TRANSITION
        assert result.values["pc_source"] == "cdm"

    def test_one_quiet_evaluation_does_not_clear_the_event(self):
        # Section 3, M1 to M0: 'Pc < pc_monitor_threshold for 2 consecutive
        # evaluations.'
        inputs = staged_inputs(pc=1e-9, consecutive_below_monitor=1)
        assert guard_pc_below_monitor_sustained(inputs).passed is False

    def test_two_quiet_evaluations_clear_the_event(self):
        inputs = staged_inputs(pc=1e-9, consecutive_below_monitor=2)
        assert guard_pc_below_monitor_sustained(inputs).passed is True

    def test_two_evaluations_do_not_clear_an_event_still_above_the_line(self):
        inputs = staged_inputs(pc=PC_AT_TRANSITION, consecutive_below_monitor=2)
        assert guard_pc_below_monitor_sustained(inputs).passed is False


# ---------------------------------------------------------------------------
# Section 3: IOD confidence
# ---------------------------------------------------------------------------


class TestIodGuard:
    def test_confident_iod_passes(self):
        assert guard_iod_confident(staged_inputs()).passed is True

    def test_a_degraded_or_rejected_iod_fails(self):
        inputs = staged_inputs(
            iod_proceeds_to_validity=False, iod_confidence_verdict="DEGRADED"
        )
        result = guard_iod_confident(inputs)
        assert result.passed is False
        assert "DEGRADED" in result.detail

    def test_an_unevaluated_iod_gate_fails_closed(self):
        assert guard_iod_confident(staged_inputs(iod_proceeds_to_validity=None)).passed is False


# ---------------------------------------------------------------------------
# Section 3 and 4.4: validity
# ---------------------------------------------------------------------------


class TestValidityGuard:
    def test_autonomous_clears_the_guard(self):
        assert guard_validity_earned(staged_inputs()).passed is True

    def test_blocked_not_earned_fails(self):
        inputs = staged_inputs(
            validity_routing=ValidityRouting.BLOCKED,
            validity_evidence={**VALIDITY_EARNED_EVIDENCE, "validity_status": "NOT_EARNED"},
        )
        assert guard_validity_earned(inputs).passed is False

    def test_operator_review_does_not_clear_autonomous_action(self):
        # EARNED, but the TLE-only review band. Not autonomous.
        inputs = staged_inputs(validity_routing=ValidityRouting.OPERATOR_REVIEW)
        result = guard_validity_earned(inputs)
        assert result.passed is False
        assert "OPERATOR_REVIEW" in result.detail

    def test_an_unavailable_validity_assessor_is_not_earned(self):
        # Section 6.3: 'Validity assessor service unavailable | M1 or M2 |
        # Treat as NOT_EARNED. Go to M4.'
        inputs = staged_inputs(validity_service_available=False)
        assert guard_validity_earned(inputs).passed is False

    def test_an_outage_beats_a_stale_autonomous_routing_value(self):
        inputs = staged_inputs(
            validity_routing=ValidityRouting.AUTONOMOUS,
            validity_service_available=False,
        )
        assert guard_validity_earned(inputs).passed is False

    def test_the_guard_records_the_378_evidence_values(self):
        values = guard_validity_earned(staged_inputs()).values
        assert values["validity_status"] == "EARNED"
        assert values["validity_epsilon"] == EPSILON
        assert values["epsilon_threshold"] == 0.20


# ---------------------------------------------------------------------------
# Section 4.2: secondary conflict
# ---------------------------------------------------------------------------


class TestSecondaryGuard:
    def test_a_clear_performed_check_passes(self):
        assert guard_secondary_clear(staged_inputs()).passed is True

    def test_a_conflicting_secondary_fails(self):
        inputs = staged_inputs(secondary_conjunction_clear=False)
        assert guard_secondary_clear(inputs).passed is False

    def test_a_check_that_was_not_performed_is_not_clear(self):
        # Section 4.2: 'If the check was not performed
        # (secondary_check_performed = false), treat as NOT CLEAR and escalate.'
        inputs = staged_inputs(
            secondary_check_performed=False, secondary_conjunction_clear=True
        )
        result = guard_secondary_clear(inputs)
        assert result.passed is False
        assert "not performed" in result.detail


# ---------------------------------------------------------------------------
# Section 4.3: slew feasibility
# ---------------------------------------------------------------------------


class TestSlewGuard:
    """t_now <= t_burn - (t_slew + t_settle + t_margin), with
    t_slew + t_settle = 120 s and t_margin = 30 s, so the lead time is 150 s."""

    def test_exactly_150_s_before_the_burn_is_still_feasible(self):
        inputs = staged_inputs(command=a_command(t_burn=T_NOW + timedelta(seconds=150)))
        assert guard_slew_feasible(inputs).passed is True

    def test_149_s_before_the_burn_is_infeasible(self):
        inputs = staged_inputs(command=a_command(t_burn=T_NOW + timedelta(seconds=149)))
        assert guard_slew_feasible(inputs).passed is False

    def test_the_guard_reports_the_lead_time_it_used(self):
        assert guard_slew_feasible(staged_inputs()).values["slew_lead_time_s"] == 150.0

    def test_a_burn_after_latest_burn_utc_is_infeasible(self):
        # Section 4.3: t_burn <= latest_burn_utc.
        inputs = staged_inputs(
            command=a_command(t_burn=T_NOW + timedelta(hours=4)),
            latest_burn_utc=T_NOW + timedelta(hours=3),
        )
        result = guard_slew_feasible(inputs)
        assert result.passed is False
        assert "latest_burn_utc" in result.detail

    def test_no_burn_means_no_feasible_slew(self):
        assert guard_slew_feasible(staged_inputs(command=None)).passed is False


# ---------------------------------------------------------------------------
# Section 3 and 4.1: authority and envelope
# ---------------------------------------------------------------------------


class TestAuthorityGuard:
    @pytest.mark.parametrize("level", ["L1", "L2"])
    def test_l1_and_l2_may_stage(self, level):
        assert guard_authority_l1_or_l2(staged_inputs(authority_level=level)).passed is True

    def test_l0_is_advisory_only(self):
        # Section 5: 'L0 | N/A (advisory only, no GNC command issued)'.
        assert guard_authority_l1_or_l2(staged_inputs(authority_level="L0")).passed is False

    def test_no_granted_authority_fails(self):
        assert guard_authority_l1_or_l2(staged_inputs(authority_level=None)).passed is False


class TestEnvelopeGuard:
    def test_a_satisfied_l2_envelope_passes(self):
        assert guard_envelope_satisfied(staged_inputs()).passed is True

    def test_dv_exactly_at_the_l2_cap_passes(self):
        # Section 4.1: 'max_dv cap for the authority level (L1: 2.0 m/s,
        # L2: 0.5 m/s)'. At the cap is within it.
        inputs = staged_inputs(command=a_command(dv_m_s=0.5), dv_cap_m_s=0.5)
        assert guard_envelope_satisfied(inputs).passed is True

    def test_dv_above_the_l2_cap_fails(self):
        inputs = staged_inputs(command=a_command(dv_m_s=0.5001), dv_cap_m_s=0.5)
        assert guard_envelope_satisfied(inputs).passed is False

    def test_dv_above_the_l1_cap_fails(self):
        inputs = staged_inputs(
            authority_level="L1", command=a_command(dv_m_s=2.0001), dv_cap_m_s=2.0
        )
        assert guard_envelope_satisfied(inputs).passed is False

    def test_the_raised_l2_cap_of_1_m_s_is_honoured_when_granted(self):
        # SCRUM-375's LOCKED_L2_RAISED_DV_CAP_M_S. The guard reads whatever
        # effective cap the compiled envelope handed it; it does not decide
        # which cap applies.
        inputs = staged_inputs(command=a_command(dv_m_s=0.9), dv_cap_m_s=1.0)
        assert guard_envelope_satisfied(inputs).passed is True

    def test_an_envelope_error_is_a_not_satisfied_signal(self):
        # EnvelopeApprovalError / EnvelopeActivationError arrive as text.
        inputs = staged_inputs(envelope_error="authorization-envelope keyed MAC verification failed")
        result = guard_envelope_satisfied(inputs)
        assert result.passed is False
        assert "MAC verification failed" in result.detail

    def test_no_active_envelope_fails(self):
        assert guard_envelope_satisfied(staged_inputs(envelope_version=None)).passed is False

    def test_an_expired_envelope_fails(self):
        # Section 4.1: 'Envelope version is cryptographically valid and not expired'.
        assert guard_envelope_satisfied(staged_inputs(envelope_expired=True)).passed is False

    def test_data_age_exactly_at_24_hours_is_still_fresh(self):
        # Section 4.1 / section 8: 86400 s for a Space-Track CDM.
        assert guard_envelope_satisfied(staged_inputs(data_age_s=86400.0)).passed is True

    def test_data_age_beyond_24_hours_fails(self):
        assert guard_envelope_satisfied(staged_inputs(data_age_s=86400.1)).passed is False

    def test_unknown_data_age_fails_closed(self):
        assert guard_envelope_satisfied(staged_inputs(data_age_s=None)).passed is False

    def test_propellant_must_strictly_exceed_dv_plus_reserve(self):
        # Section 4.1: 'v_remaining_m_s > dv_magnitude_m_s + v_reserved_m_s'.
        # Strictly greater, so exactly equal fails.
        exactly_equal = staged_inputs(
            command=a_command(dv_m_s=1.0), dv_cap_m_s=2.0,
            v_remaining_m_s=6.0, v_reserved_m_s=5.0,
        )
        assert guard_envelope_satisfied(exactly_equal).passed is False

        one_drop_more = staged_inputs(
            command=a_command(dv_m_s=1.0), dv_cap_m_s=2.0,
            v_remaining_m_s=6.001, v_reserved_m_s=5.0,
        )
        assert guard_envelope_satisfied(one_drop_more).passed is True


class TestCovarianceProvenanceGuard:
    """Section 4.1: 'covariance_source is real_cdm (surrogate_identity
    triggers a warning and requires explicit operator acknowledgement before
    L2 auto-execute)'."""

    def test_real_cdm_passes(self):
        assert guard_covariance_provenance(staged_inputs()).passed is True

    def test_surrogate_identity_without_acknowledgement_fails(self):
        inputs = staged_inputs(covariance_source="surrogate_identity")
        assert guard_covariance_provenance(inputs).passed is False

    def test_surrogate_identity_with_explicit_acknowledgement_passes(self):
        inputs = staged_inputs(
            covariance_source="surrogate_identity",
            operator_ack_surrogate_covariance=True,
        )
        assert guard_covariance_provenance(inputs).passed is True

    def test_an_unknown_covariance_source_fails(self):
        assert guard_covariance_provenance(staged_inputs(covariance_source="")).passed is False


# ---------------------------------------------------------------------------
# Section 3: the 4-hour TCA floor
# ---------------------------------------------------------------------------


class TestTcaFloorGuard:
    def test_exactly_four_hours_to_tca_is_still_above_the_floor(self):
        # Section 3: 'TCA < min_hours_before_tca (4.0 hrs)'. Strictly less,
        # so exactly 4.0 h has not breached it.
        inputs = staged_inputs(t_ca_utc=T_NOW + timedelta(hours=4))
        assert guard_tca_above_floor(inputs).passed is True

    def test_inside_four_hours_with_pc_above_monitor_fails(self):
        inputs = staged_inputs(t_ca_utc=T_NOW + timedelta(hours=3, minutes=59))
        assert guard_tca_above_floor(inputs).passed is False

    def test_inside_four_hours_with_pc_below_monitor_does_not_escalate(self):
        # The doc qualifies this clause: '...with Pc still above monitor
        # threshold'. A resolved event inside the floor is not an escalation.
        inputs = staged_inputs(t_ca_utc=T_NOW + timedelta(hours=1), pc=1e-9)
        assert guard_tca_above_floor(inputs).passed is True


# ---------------------------------------------------------------------------
# Sections 3 and 5: M2 approval, veto and re-veto
# ---------------------------------------------------------------------------


class TestApprovalAndVetoGuards:
    def test_l1_needs_an_accepted_approval(self):
        assert guard_l1_approval_accepted(staged_inputs()).passed is False
        accepted = staged_inputs(approval_command_received=True, approval_accepted=True)
        assert guard_l1_approval_accepted(accepted).passed is True

    def test_l1_rejection_is_not_an_approval(self):
        rejected = staged_inputs(approval_command_received=True, approval_accepted=False)
        assert guard_l1_approval_accepted(rejected).passed is False

    def test_the_l2_window_is_not_expired_one_second_early(self):
        # Section 5: the minimum veto window is 300 s, driven by t_review.
        opened = T_NOW - timedelta(seconds=299)
        inputs = staged_inputs(
            current_mode="M2", veto_window_close_utc=opened + timedelta(seconds=300)
        )
        assert guard_l2_veto_window_expired(inputs).passed is False

    def test_the_l2_window_expires_exactly_at_close(self):
        inputs = staged_inputs(current_mode="M2", veto_window_close_utc=T_NOW)
        assert guard_l2_veto_window_expired(inputs).passed is True

    def test_a_veto_command_stops_the_window_from_expiring(self):
        inputs = staged_inputs(
            current_mode="M2",
            veto_window_close_utc=T_NOW - timedelta(seconds=1),
            veto_command_received=True,
        )
        assert guard_l2_veto_window_expired(inputs).passed is False

    def test_reveto_restages_on_a_fresh_still_earned_cdm(self):
        # Section 3, M2 to M2, and section 8's re-veto recommendation.
        inputs = staged_inputs(
            current_mode="M2",
            veto_command_received=True,
            veto_accepted=True,
            fresh_cdm_available=True,
            data_age_s=600.0,
        )
        assert guard_reveto_may_restage(inputs).passed is True

    def test_reveto_without_a_fresh_cdm_does_not_restage(self):
        inputs = staged_inputs(
            current_mode="M2",
            veto_command_received=True,
            veto_accepted=True,
            fresh_cdm_available=False,
        )
        assert guard_reveto_may_restage(inputs).passed is False

    def test_reveto_with_a_stale_cdm_does_not_restage(self):
        inputs = staged_inputs(
            current_mode="M2",
            veto_command_received=True,
            veto_accepted=True,
            fresh_cdm_available=True,
            data_age_s=86400.1,
        )
        assert guard_reveto_may_restage(inputs).passed is False

    def test_reveto_with_validity_lost_on_the_new_cdm_does_not_restage(self):
        inputs = staged_inputs(
            current_mode="M2",
            veto_command_received=True,
            veto_accepted=True,
            fresh_cdm_available=True,
            data_age_s=600.0,
            validity_routing=ValidityRouting.BLOCKED,
        )
        assert guard_reveto_may_restage(inputs).passed is False

    def test_reveto_below_the_action_line_does_not_restage(self):
        inputs = staged_inputs(
            current_mode="M2",
            veto_command_received=True,
            veto_accepted=True,
            fresh_cdm_available=True,
            data_age_s=600.0,
            pc=0.99e-4,
        )
        assert guard_reveto_may_restage(inputs).passed is False


# ---------------------------------------------------------------------------
# Sections 3 and 4.5: post-burn
# ---------------------------------------------------------------------------


class TestPostBurnGuards:
    def test_a_nominal_report_passes(self):
        inputs = staged_inputs(
            current_mode="M3", gnc_report_received=True, execution_status="NOMINAL"
        )
        assert guard_execution_nominal(inputs).passed is True

    @pytest.mark.parametrize("status", ["ABORTED", "PARTIAL"])
    def test_an_aborted_or_partial_report_fails(self, status):
        inputs = staged_inputs(
            current_mode="M3", gnc_report_received=True, execution_status=status
        )
        assert guard_execution_nominal(inputs).passed is False

    def test_no_report_yet_fails(self):
        assert guard_execution_nominal(staged_inputs(current_mode="M3")).passed is False

    def test_m2_post_must_strictly_exceed_25(self):
        # Section 4.5: m2_post > m2_safe, recommended m2_safe = 25
        # (5-sigma separation in the conjunction plane).
        assert guard_m2_post_safe(staged_inputs(m2_post=25.0)).passed is False
        assert guard_m2_post_safe(staged_inputs(m2_post=25.0001)).passed is True

    def test_an_unknown_m2_post_fails_closed(self):
        assert guard_m2_post_safe(staged_inputs(m2_post=None)).passed is False


# ---------------------------------------------------------------------------
# Section 6.3: subsystem failure
# ---------------------------------------------------------------------------


class TestSubsystemGuard:
    def test_a_healthy_bus_passes(self):
        assert guard_subsystem_healthy(staged_inputs()).passed is True

    @pytest.mark.parametrize(
        "failure",
        [
            "navigation_filter_degraded",
            "thruster_fault",
            "ATTITUDE_ERROR_EXCEEDED",
            "monitor_certification_failure",
        ],
    )
    def test_any_reported_failure_fails(self, failure):
        assert guard_subsystem_healthy(staged_inputs(subsystem_failure=failure)).passed is False


# ---------------------------------------------------------------------------
# The M1 to M2 AND, built last because it is the conjunction of all the above
# ---------------------------------------------------------------------------


class TestM1ToM2:
    """Section 3: 'Pc >= pc_maneuver_threshold AND IOD confidence CONFIDENT AND
    validity EARNED AND envelope satisfied AND secondary conflict clear AND
    slew feasible AND authority L1 or L2.'"""

    def test_all_seven_guards_are_evaluated_in_the_doc_order(self):
        names = [g.name for g in m1_to_m2_guards(staged_inputs())]
        assert names[:7] == [
            "pc_above_maneuver_threshold",
            "iod_confident",
            "validity_earned",
            "envelope_satisfied",
            "secondary_conflict_clear",
            "slew_feasible",
            "authority_l1_or_l2",
        ]

    def test_the_section_3_seven_are_followed_by_the_scrum_380_baseline_floor(self):
        """SCRUM-380 adds the ground-validated-baseline floor to the AND. It is
        a MAF section 6 floor layered on top of the section 3 row, not one of
        its seven, so it sits after them rather than being interleaved."""
        names = [g.name for g in m1_to_m2_guards(staged_inputs())]

        assert names[7:] == ["authority_baseline_validated"]

    def test_the_baseline_event_satisfies_every_guard_and_stages(self):
        decision = evaluate_safety_monitor(staged_inputs())

        assert decision.mode is FlightMode.M2_STAGED
        assert decision.escalated is False
        assert all(g.passed for g in decision.guards)

    @pytest.mark.parametrize(
        "override, failing_guard",
        [
            ({"pc": 0.99e-4}, "pc_above_maneuver_threshold"),
            ({"iod_proceeds_to_validity": False}, "iod_confident"),
            ({"validity_routing": ValidityRouting.OPERATOR_REVIEW}, "validity_earned"),
            ({"dv_cap_m_s": 0.1}, "envelope_satisfied"),
            ({"secondary_check_performed": False}, "secondary_conflict_clear"),
            ({"latest_burn_utc": T_NOW}, "slew_feasible"),
            ({"authority_level": "L0"}, "authority_l1_or_l2"),
        ],
    )
    def test_breaking_any_single_guard_prevents_staging(self, override, failing_guard):
        """The AND is a real AND: one broken guard is enough."""
        decision = evaluate_safety_monitor(staged_inputs(**override))

        assert decision.mode is not FlightMode.M2_STAGED
        assert failing_guard in [g.name for g in decision.guards if not g.passed]
