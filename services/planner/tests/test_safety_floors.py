"""
SCRUM-380 -- the refusal floors, and the proof they are non-negotiable.

MAF v2.0 section 6 and guard doc sections 4 to 7. Three floors plus one sweep:

  * degraded or zero-filled records are rejected, not parsed
  * L3 is refused without a pre-verified safe action
  * the post-burn feasibility flag is surfaced on an m2_post failure
  * no operator-supplied or auto-derived value can loosen a locked floor

The last one is the point of the whole ticket. Every threshold below is locked;
a change to any of them escalates to Minh, and there is deliberately no config
knob to turn. The tests assert the literal doc values rather than importing the
constants, so retuning a constant fails a test instead of silently redefining
the floor.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from common.authorization_envelope import (
    LOCKED_L1_DV_CAP_M_S,
    LOCKED_L2_DV_CAP_M_S,
    LOCKED_L2_RAISED_DV_CAP_M_S,
    LOCKED_MAX_TCA_HOURS,
    LOCKED_MIN_MISS_DISTANCE_KM,
    LOCKED_MIN_TCA_HOURS,
    LOCKED_PC_ACTION_CEILING,
    AuthorityLevel,
    EnvelopeCompileError,
    ManeuverCapacityInput,
    OperatorAuthorizationProfile,
    RiskBudgetInput,
    compile_authorization_envelope,
)
from common.decision_state_machine import (
    DATA_FRESHNESS_BOUND_S,
    M2_SAFE_MAHALANOBIS,
    MIN_HOURS_BEFORE_TCA,
    FlightMode,
)
from common.safety_floors import cdm_record_rejection_reason, l3_refusal_reason
from common.safety_monitor import (
    evaluate_safety_monitor,
    guard_cdm_record_usable,
    guard_l3_pre_verified_action,
    guard_m2_post_safe,
)

from test_safety_monitor import T_NOW, a_command, staged_inputs

_GOOD_COV = [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4]


# ---------------------------------------------------------------------------
# Degraded / zero-filled record rejection
# ---------------------------------------------------------------------------


class TestRecordRejection:
    def test_a_real_record_is_usable(self):
        assert cdm_record_rejection_reason(_GOOD_COV, [0.1, 0.2, 0.3]) == ""

    def test_a_zero_filled_covariance_is_rejected(self):
        """Zero covariance is not 'no uncertainty', it is 'no covariance'.
        Scoring it yields a confident-looking Pc computed from nothing."""
        reason = cdm_record_rejection_reason([0.0] * 9, [0.1, 0.2, 0.3])

        assert "zero-filled" in reason

    def test_a_covariance_with_a_zero_variance_is_rejected(self):
        cov = list(_GOOD_COV)
        cov[4] = 0.0
        reason = cdm_record_rejection_reason(cov, [0.1, 0.2, 0.3])

        assert "non-positive variance" in reason

    def test_a_negative_variance_is_rejected(self):
        cov = list(_GOOD_COV)
        cov[8] = -1e-4
        assert cdm_record_rejection_reason(cov, [0.1, 0.2, 0.3]) != ""

    def test_a_non_finite_covariance_is_rejected(self):
        assert cdm_record_rejection_reason(
            [float("nan")] + [0.0] * 8, [0.1, 0.2, 0.3]
        ) != ""

    def test_a_zero_filled_relative_position_is_rejected(self):
        reason = cdm_record_rejection_reason(_GOOD_COV, [0.0, 0.0, 0.0])

        assert "zero-filled" in reason

    def test_a_producer_flagged_degraded_record_is_rejected(self):
        reason = cdm_record_rejection_reason(
            _GOOD_COV, [0.1, 0.2, 0.3], declared_degraded=True
        )

        assert "degraded" in reason

    def test_an_awkward_geometry_is_not_a_degraded_record(self):
        """covariance_quality reads 'degraded' or 'dilution_region' for a real
        covariance in a bad geometry. Those are legitimate, scorable
        conjunctions and this floor must not swallow them."""
        inputs = staged_inputs(current_mode="M1")

        assert guard_cdm_record_usable(inputs).passed is True

    def test_a_hollow_record_blocks_staging_and_holds_the_watch(self):
        """Section 6: hold M1 rather than staging a burn against nothing."""
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M1",
                cdm_record_rejected_reason="relative covariance is zero-filled",
            )
        )

        assert decision.mode is FlightMode.M1_WATCH
        assert "cdm_record_usable" in [g.name for g in decision.guards if not g.passed]

    def test_a_hollow_record_while_staged_escalates(self):
        """Past staging, blocking is not enough: a burn is already loaded
        against a record that turns out to be hollow."""
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M2",
                cdm_record_rejected_reason="relative covariance is zero-filled",
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_a_hollow_record_cannot_auto_execute(self):
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M2",
                veto_window_close_utc=T_NOW - timedelta(seconds=1),
                cdm_record_rejected_reason="relative position is zero-filled",
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.authorized_execution is None


# ---------------------------------------------------------------------------
# L3 refusal
# ---------------------------------------------------------------------------


class TestL3Refusal:
    def test_the_floor_does_not_apply_below_l3(self):
        assert l3_refusal_reason("L2", None) == ""
        assert l3_refusal_reason("L1", None) == ""
        assert l3_refusal_reason("L0", None) == ""

    def test_l3_without_a_pre_verified_action_is_refused(self):
        reason = l3_refusal_reason("L3", None)

        assert "refused" in reason
        assert "pre-verified safe action" in reason

    def test_a_blank_pre_verified_action_does_not_count(self):
        assert l3_refusal_reason("L3", "   ") != ""

    def test_l3_with_a_pre_verified_action_clears_the_floor(self):
        """Clearing the floor is not permission to execute L3. The execution
        path is gated to a later phase and does not exist."""
        assert l3_refusal_reason("L3", "safe-action-0001") == ""

    def test_the_guard_refuses_an_l3_request(self):
        inputs = staged_inputs(authority_level="L3")

        assert guard_l3_pre_verified_action(inputs).passed is False

    def test_an_l3_request_cannot_stage(self):
        decision = evaluate_safety_monitor(
            staged_inputs(current_mode="M1", authority_level="L3")
        )

        assert decision.mode is not FlightMode.M2_STAGED

    def test_the_guard_passes_for_every_level_the_ladder_actually_supports(self):
        for level in ("L0", "L1", "L2"):
            assert guard_l3_pre_verified_action(
                staged_inputs(authority_level=level)
            ).passed is True


# ---------------------------------------------------------------------------
# Post-burn feasibility flag (guard doc section 4.5)
# ---------------------------------------------------------------------------


class TestFeasibilityFlag:
    def test_a_safe_post_burn_separation_reports_feasible(self):
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M3", gnc_report_received=True,
                execution_status="NOMINAL", pc=1e-9, m2_post=30.0,
            )
        )

        assert decision.post_burn_feasible is True
        assert decision.to_dict()["post_burn_feasible"] is True

    def test_an_unsafe_post_burn_separation_reports_infeasible(self):
        """Section 4.5's failure has to show in the evidence package as a
        feasibility result, not only as a mode change a reader must interpret."""
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M3", gnc_report_received=True,
                execution_status="NOMINAL", pc=1e-9, m2_post=25.0,
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert decision.post_burn_feasible is False
        assert decision.to_dict()["post_burn_feasible"] is False

    def test_the_flag_is_absent_when_no_post_burn_check_ran(self):
        decision = evaluate_safety_monitor(staged_inputs(current_mode="M1"))

        assert decision.post_burn_feasible is None


# ---------------------------------------------------------------------------
# Non-negotiability sweep
# ---------------------------------------------------------------------------


def _profile(**overrides):
    base = dict(
        profile_id="p", operator_id="o", mission_class="LEO",
        risk_budget=RiskBudgetInput(risk_budget_id="rb", pc_watch_min=1e-5),
        maneuver_capacity=ManeuverCapacityInput(
            max_dv_per_burn_m_s=10.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0
        ),
        pc_action=1e-4,
        authority_level=AuthorityLevel.L2,
    )
    base.update(overrides)
    return OperatorAuthorizationProfile(**base)


class TestLockedFloorsCannotBeLoosened:
    """MAF v2.0 section 6: these are non-negotiable. An operator-supplied or
    auto-derived value must be rejected or clamped, never honoured.

    Values are written out literally rather than imported, so retuning a
    constant fails here instead of quietly redefining the floor.
    """

    def test_the_locked_values_are_the_doc_values(self):
        assert LOCKED_PC_ACTION_CEILING == 1.0e-4
        assert LOCKED_L1_DV_CAP_M_S == 2.0
        assert LOCKED_L2_DV_CAP_M_S == 0.5
        assert LOCKED_L2_RAISED_DV_CAP_M_S == 1.0
        assert LOCKED_MIN_TCA_HOURS == 4.0
        assert LOCKED_MAX_TCA_HOURS == 72.0
        assert LOCKED_MIN_MISS_DISTANCE_KM == 1.0
        assert MIN_HOURS_BEFORE_TCA == 4.0
        assert M2_SAFE_MAHALANOBIS == 25.0
        assert DATA_FRESHNESS_BOUND_S == 86400.0

    # -- pc_action ceiling ---------------------------------------------------

    def test_an_operator_cannot_raise_pc_action_above_1e_4(self):
        with pytest.raises(EnvelopeCompileError):
            _profile(pc_action=1.01e-4)

    def test_pc_action_exactly_at_the_ceiling_is_allowed(self):
        assert _profile(pc_action=1.0e-4).pc_action == 1.0e-4

    # -- dv caps -------------------------------------------------------------

    def test_a_generous_capability_cannot_raise_the_l2_dv_cap(self):
        """A satellite that can burn 10 m/s is still capped at the locked 0.5."""
        envelope = compile_authorization_envelope(_profile())

        assert envelope.effective_l2_dv_cap_m_s == 0.5

    def test_a_generous_capability_cannot_raise_the_l1_dv_cap(self):
        envelope = compile_authorization_envelope(
            _profile(authority_level=AuthorityLevel.L1)
        )

        assert envelope.effective_l1_dv_cap_m_s == 2.0

    def test_the_raised_l2_cap_is_itself_capped_at_1_m_s(self):
        envelope = compile_authorization_envelope(_profile(l2_raise_to_1_m_s=True))

        assert envelope.effective_l2_dv_cap_m_s == 1.0

    def test_a_dv_above_the_cap_is_refused_at_the_guard(self):
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M1",
                command=a_command(dv_m_s=0.5001),
                dv_cap_m_s=0.5,
            )
        )

        assert decision.mode is not FlightMode.M2_STAGED
        assert "envelope_satisfied" in [g.name for g in decision.guards if not g.passed]

    # -- freshness bound -----------------------------------------------------

    def test_data_older_than_24_hours_cannot_be_declared_fresh(self):
        """The bound is a floor on the guard, not a per-request setting: a
        request supplying a laxer bound still fails at the locked one."""
        decision = evaluate_safety_monitor(
            staged_inputs(current_mode="M1", data_age_s=86400.1)
        )

        assert decision.mode is not FlightMode.M2_STAGED

    def test_data_exactly_at_24_hours_is_still_fresh(self):
        inputs = staged_inputs(current_mode="M1", data_age_s=86400.0)

        assert evaluate_safety_monitor(inputs).mode is FlightMode.M2_STAGED

    def test_a_request_cannot_widen_the_freshness_bound(self):
        """A caller asking for a 48 hour bound would make a 36 hour old CDM
        'fresh'. The clamp holds the bound at the locked 24."""
        inputs = staged_inputs(current_mode="M1", data_freshness_bound_s=172800.0)

        assert inputs.data_freshness_bound_s == 86400.0

    def test_a_lax_freshness_request_still_fails_on_stale_data(self):
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M1",
                data_age_s=129600.0,             # 36 hours
                data_freshness_bound_s=172800.0, # asking for 48
            )
        )

        assert decision.mode is not FlightMode.M2_STAGED

    def test_a_request_may_demand_fresher_data(self):
        inputs = staged_inputs(current_mode="M1", data_freshness_bound_s=3600.0)

        assert inputs.data_freshness_bound_s == 3600.0

    # -- TCA floor -----------------------------------------------------------

    def test_an_operator_cannot_lower_the_4_hour_tca_floor(self):
        class LaxPolicy:
            operator_id = "o"
            pc_maneuver_threshold = 1e-4
            min_miss_distance_km = 1.0
            max_dv_per_event_ms = 2.0
            max_maneuvers_per_week = 3
            min_hours_before_tca = 3.0     # below the locked 4.0
            max_hours_before_tca = 72.0
            pc_monitor_threshold = 1e-5

        from common.authorization_envelope import validate_section6_safety_floors

        with pytest.raises(EnvelopeCompileError):
            validate_section6_safety_floors(LaxPolicy())

    def test_a_request_cannot_lower_the_tca_floor_below_4_hours(self):
        """The clamp SCRUM-380 adds. Before it, a request supplying
        min_hours_before_tca=1.0 was honoured and the floor was loosened by the
        very request it exists to constrain."""
        inputs = staged_inputs(current_mode="M1", min_hours_before_tca=1.0)

        assert inputs.min_hours_before_tca == 4.0

    def test_a_request_may_be_more_cautious_than_the_tca_floor(self):
        """Clamped in one direction only: more caution is the operator's to ask
        for, less is not."""
        inputs = staged_inputs(current_mode="M1", min_hours_before_tca=12.0)

        assert inputs.min_hours_before_tca == 12.0

    def test_a_lax_tca_request_still_escalates_inside_the_locked_floor(self):
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M1",
                t_ca_utc=T_NOW + timedelta(hours=3, minutes=59),
                min_hours_before_tca=1.0,   # asking for a laxer floor
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    # -- m2_safe -------------------------------------------------------------

    def test_m2_post_must_strictly_exceed_25(self):
        assert guard_m2_post_safe(staged_inputs(m2_post=25.0)).passed is False
        assert guard_m2_post_safe(staged_inputs(m2_post=25.0001)).passed is True

    def test_a_request_cannot_lower_the_m2_safe_threshold_to_pass(self):
        """A caller supplying m2_safe_threshold=1.0 would make a 2.0 separation
        read as 'safe'. Section 4.5 fixes it at 25 and the clamp holds it there."""
        lowered = staged_inputs(m2_post=2.0, m2_safe_threshold=1.0)

        assert lowered.m2_safe_threshold == 25.0
        assert guard_m2_post_safe(lowered).passed is False

    def test_a_request_may_demand_a_stricter_separation(self):
        stricter = staged_inputs(m2_post=30.0, m2_safe_threshold=50.0)

        assert stricter.m2_safe_threshold == 50.0
        assert guard_m2_post_safe(stricter).passed is False

    # -- miss distance -------------------------------------------------------

    def test_an_operator_cannot_lower_the_1_km_miss_distance_floor(self):
        class LaxPolicy:
            operator_id = "o"
            pc_maneuver_threshold = 1e-4
            min_miss_distance_km = 0.5     # below the locked 1.0
            max_dv_per_event_ms = 2.0
            max_maneuvers_per_week = 3
            min_hours_before_tca = 4.0
            max_hours_before_tca = 72.0
            pc_monitor_threshold = 1e-5

        from common.authorization_envelope import validate_section6_safety_floors

        with pytest.raises(EnvelopeCompileError):
            validate_section6_safety_floors(LaxPolicy())

    def test_an_operator_cannot_raise_the_max_dv_per_event(self):
        class LaxPolicy:
            operator_id = "o"
            pc_maneuver_threshold = 1e-4
            min_miss_distance_km = 1.0
            max_dv_per_event_ms = 2.5      # above the locked 2.0 L1 cap
            max_maneuvers_per_week = 3
            min_hours_before_tca = 4.0
            max_hours_before_tca = 72.0
            pc_monitor_threshold = 1e-5

        from common.authorization_envelope import validate_section6_safety_floors

        with pytest.raises(EnvelopeCompileError):
            validate_section6_safety_floors(LaxPolicy())

    def test_an_operator_cannot_exceed_3_maneuvers_per_week(self):
        envelope = compile_authorization_envelope(
            _profile(
                maneuver_capacity=ManeuverCapacityInput(
                    max_dv_per_burn_m_s=10.0,
                    v_remaining_m_s=50.0,
                    v_reserved_m_s=5.0,
                    max_maneuvers_per_week=10,
                )
            )
        )

        assert envelope.effective_max_maneuvers_per_week == 3

    def test_an_operator_cannot_lower_the_reserved_dv(self):
        envelope = compile_authorization_envelope(
            _profile(
                maneuver_capacity=ManeuverCapacityInput(
                    max_dv_per_burn_m_s=10.0,
                    v_remaining_m_s=50.0,
                    v_reserved_m_s=1.0,    # below the locked 5.0
                )
            )
        )

        assert envelope.effective_reserved_dv_m_s == 5.0
