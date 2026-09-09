"""
SCRUM-379 -- modes, transition table and the ESCALATE default arm.

Every expectation here is anchored to docs/scrum-333/state_machine_guards.md:
section 1 for the escalate-by-default rule, section 2 for the mode list,
section 3 for the transition table, sections 4 to 6 for the locked timings.
The values are quoted from the doc, not re-derived from the implementation.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from common.decision_state_machine import (
    AUTONOMOUS_AUTHORITY_LEVELS,
    COMMS_GAP_THRESHOLD_S,
    CONSECUTIVE_BELOW_MONITOR_TO_CLEAR,
    DATA_FRESHNESS_BOUND_S,
    ESCALATE_TRIGGER_PREFIX,
    M2_SAFE_MAHALANOBIS,
    MIN_HOURS_BEFORE_TCA,
    MIN_VETO_WINDOW_S,
    REAL_CDM_COVARIANCE_SOURCE,
    T_MARGIN_S,
    T_REVIEW_S,
    T_SLEW_SETTLE_S,
    FlightMode,
    GuardInputs,
    GuardResult,
    TRANSITION_TABLE,
    ValidityRouting,
    as_mode,
    first_failure,
    is_defined_transition,
    l2_veto_window_s,
    resolve_transition,
    slew_lead_time_s,
)

T0 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)

ALL_MODES = ("M0", "M1", "M2", "M3", "M4")


# ---------------------------------------------------------------------------
# Section 1: the default arm. This is the first test in the story for a reason.
# ---------------------------------------------------------------------------


class TestUndefinedTransitionsEscalate:
    """Section 1: 'Any undefined or ambiguous transition defaults to
    ESCALATE to M4.'"""

    def test_unknown_transition_falls_to_m4(self):
        decision = resolve_transition(FlightMode.M0_NOMINAL, FlightMode.M3_EXECUTING)

        assert decision.to_mode == FlightMode.M4_SAFE_HOLD
        assert decision.escalated is True

    def test_escalation_preserves_what_was_attempted(self):
        decision = resolve_transition(
            FlightMode.M0_NOMINAL, FlightMode.M3_EXECUTING, "operator typo"
        )

        assert decision.requested_mode == FlightMode.M3_EXECUTING
        assert ESCALATE_TRIGGER_PREFIX in decision.trigger
        assert "M0->M3" in decision.trigger
        assert "operator typo" in decision.trigger

    def test_every_undefined_pair_escalates_to_m4_not_to_a_lower_mode(self):
        """The default arm is M4, never a fall-through to M0 or M1.

        Exhaustive over the 5x5 grid, so a future edit that adds a
        'safe-looking' fall-through cannot pass by only touching one pair.
        """
        for origin in ALL_MODES:
            for target in ALL_MODES:
                decision = resolve_transition(origin, target)
                if (origin, target) in TRANSITION_TABLE:
                    continue
                assert decision.to_mode == FlightMode.M4_SAFE_HOLD, (
                    f"{origin}->{target} is undefined and must escalate to M4"
                )
                assert decision.escalated is True

    def test_m4_cannot_be_left_except_by_ground_clearance_to_m0(self):
        """Section 3, last row: 'Ground command only: explicit operator
        clearance via ARBITER.'"""
        assert is_defined_transition("M4", "M0")
        for target in ("M1", "M2", "M3", "M4"):
            assert resolve_transition("M4", target).to_mode == FlightMode.M4_SAFE_HOLD

    def test_a_defined_transition_is_not_escalated(self):
        decision = resolve_transition("M1", "M2", "pc_above_maneuver_threshold")

        assert decision.to_mode == FlightMode.M2_STAGED
        assert decision.escalated is False
        assert decision.trigger == "pc_above_maneuver_threshold"
        assert decision.is_transition is True

    def test_an_undefined_self_loop_escalates(self):
        """M2 to M2 is defined (the L2 re-veto row); M1 to M1 is not."""
        assert resolve_transition("M2", "M2").to_mode == FlightMode.M2_STAGED
        assert resolve_transition("M1", "M1").to_mode == FlightMode.M4_SAFE_HOLD

    def test_a_mode_outside_m0_to_m4_is_a_programming_error_not_an_escalation(self):
        with pytest.raises(ValueError):
            resolve_transition("M9", "M0")


# ---------------------------------------------------------------------------
# Section 2 and 3: modes and the table itself
# ---------------------------------------------------------------------------


class TestModes:
    def test_the_five_modes_are_exactly_section_2(self):
        assert [m.value for m in FlightMode] == ["M0", "M1", "M2", "M3", "M4"]

    def test_modes_compare_equal_to_their_evidence_chain_strings(self):
        # build_transition_record validates against plain strings; a mode must
        # travel to the audit chain without translation.
        assert FlightMode.M2_STAGED == "M2"
        assert as_mode("M2") is FlightMode.M2_STAGED


class TestTransitionTable:
    def test_the_table_is_exactly_the_section_3_rows(self):
        assert TRANSITION_TABLE == frozenset({
            ("M0", "M1"),
            ("M1", "M0"),
            ("M1", "M2"),
            ("M1", "M4"),
            ("M2", "M0"),
            ("M2", "M2"),
            ("M2", "M3"),
            ("M2", "M4"),
            ("M3", "M0"),
            ("M3", "M1"),
            ("M3", "M4"),
            ("M4", "M0"),
        })

    def test_no_transition_skips_a_mode_upward(self):
        """Section 3 has no M0 to M2, M0 to M3 or M1 to M3 row: staging and
        executing are always reached one mode at a time."""
        for pair in (("M0", "M2"), ("M0", "M3"), ("M1", "M3")):
            assert pair not in TRANSITION_TABLE


# ---------------------------------------------------------------------------
# Sections 4 to 6: the locked numbers
# ---------------------------------------------------------------------------


class TestLockedValues:
    """Quoted from the doc. A change to any of these escalates to Minh."""

    def test_slew_settle_and_margin_are_the_section_4_3_values(self):
        assert T_SLEW_SETTLE_S == 120.0
        assert T_MARGIN_S == 30.0

    def test_slew_lead_time_is_slew_plus_settle_plus_margin(self):
        # Section 4.3: t_now <= t_burn - (t_slew + t_settle + t_margin).
        assert slew_lead_time_s() == 150.0

    def test_l2_veto_window_is_the_section_5_maximum(self):
        # Section 5: t_veto = max(t_review, t_slew + t_settle + t_margin),
        # and the doc's own 'Minimum veto window' row reads 300 s.
        assert T_REVIEW_S == 300.0
        assert l2_veto_window_s() == 300.0
        assert MIN_VETO_WINDOW_S == 300.0

    def test_comms_gap_threshold_is_600_s(self):
        assert COMMS_GAP_THRESHOLD_S == 600.0

    def test_mahalanobis_safe_threshold_is_25(self):
        assert M2_SAFE_MAHALANOBIS == 25.0

    def test_freshness_bound_is_24_hours(self):
        assert DATA_FRESHNESS_BOUND_S == 86400.0

    def test_min_hours_before_tca_is_the_locked_four_hour_floor(self):
        assert MIN_HOURS_BEFORE_TCA == 4.0

    def test_m1_to_m0_needs_two_consecutive_evaluations(self):
        assert CONSECUTIVE_BELOW_MONITOR_TO_CLEAR == 2

    def test_l2_auto_execute_covariance_source_is_real_cdm(self):
        assert REAL_CDM_COVARIANCE_SOURCE == "real_cdm"

    def test_only_l1_and_l2_may_stage(self):
        assert AUTONOMOUS_AUTHORITY_LEVELS == ("L1", "L2")


# ---------------------------------------------------------------------------
# The inputs bundle
# ---------------------------------------------------------------------------


class TestGuardInputs:
    def test_minimum_bundle_leaves_every_optional_input_unknown(self):
        inputs = GuardInputs(
            conjunction_id="conj-2026-0302-001",
            current_mode="M0",
            t_now_utc=T0,
        )

        assert inputs.current_mode is FlightMode.M0_NOMINAL
        # Unknown means None, never a defaulted number a guard could read as a
        # measurement.
        assert inputs.pc is None
        assert inputs.iod_proceeds_to_validity is None
        assert inputs.validity_routing is None
        assert inputs.authority_level is None
        # Section 4.2: a check that was not performed is NOT CLEAR.
        assert inputs.secondary_check_performed is False
        assert inputs.secondary_conjunction_clear is False

    def test_naive_timestamps_are_rejected(self):
        with pytest.raises(ValueError):
            GuardInputs(
                conjunction_id="C1",
                current_mode="M0",
                t_now_utc=datetime(2026, 4, 1, 12, 0, 0),
            )

    def test_timestamps_are_normalised_to_utc(self):
        from datetime import timedelta

        offset = datetime(2026, 4, 1, 14, 0, 0, tzinfo=timezone(timedelta(hours=2)))
        inputs = GuardInputs(conjunction_id="C1", current_mode="M0", t_now_utc=offset)

        assert inputs.t_now_utc == T0
        assert inputs.t_now_utc.tzinfo == timezone.utc

    def test_hours_to_tca_is_measured_from_now(self):
        from datetime import timedelta

        inputs = GuardInputs(
            conjunction_id="C1",
            current_mode="M1",
            t_now_utc=T0,
            t_ca_utc=T0 + timedelta(hours=3.5),
        )

        assert inputs.hours_to_tca() == pytest.approx(3.5)

    def test_validity_routing_accepts_a_plain_routing_string(self):
        # services/validity's RoutingDecision is a str enum with these exact
        # values, so a caller may pass either. That the two enums have not
        # drifted is asserted in test_validity_seam.py, where the import of
        # services/validity belongs.
        inputs = GuardInputs(
            conjunction_id="C1",
            current_mode="M1",
            t_now_utc=T0,
            validity_routing="AUTONOMOUS",
        )

        assert inputs.validity_routing is ValidityRouting.AUTONOMOUS


class TestGuardResult:
    def test_first_failure_reports_the_earliest_failing_guard(self):
        results = [
            GuardResult("pc_above_action", True, "ok"),
            GuardResult("iod_confident", False, "DEGRADED"),
            GuardResult("validity_earned", False, "NOT_EARNED"),
        ]

        assert first_failure(results).name == "iod_confident"

    def test_first_failure_is_none_when_every_guard_passes(self):
        assert first_failure([GuardResult("a", True, "ok")]) is None

    def test_a_guard_result_carries_the_values_it_read(self):
        result = GuardResult(
            "pc_above_action", True, "Pc 3.12e-04 >= 1e-04",
            {"pc": 0.000312, "pc_maneuver_threshold": 1e-4},
        )

        assert result.to_dict()["values"]["pc"] == 0.000312
