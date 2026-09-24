"""
SCRUM-382 -- consuming the GNCReport and folding the post-burn outcome back in.

MAF v2.0 sections 9 and 10. Every report and ack here is validated against
openapi/gnc_interface.yaml itself.

Two boundaries the ticket draws, and both are asserted rather than assumed:
the post-maneuver Pc is computed by maneuver_scorer.compute_pc_post, not here,
and the ExecutionError block is SCRUM-365's and is carried through verbatim.

The section 10 producers under test are the three the evidence catalogue lists
against SCRUM-382: actual_vs_predicted, post_maneuver_od, residual_risk.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from common import gnc_report
from common.evidence_record import FIELD_PRODUCERS
from common.gnc_contract import validate_against_contract
from common.gnc_report import (
    assess_post_burn,
    build_report_ack,
    post_burn_covariance_km2,
)

# A LEO post-burn state with a real, readable covariance.
_P_POST = [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4]
_P_PRE = [2e-4, 0, 0, 0, 2e-4, 0, 0, 0, 2e-4]
_P_BURN = [1e-5, 0, 0, 0, 1e-5, 0, 0, 0, 1e-5]


def a_report(**overrides) -> dict:
    base = {
        "command_id": "gnc-cmd-0001",
        "conjunction_id": "conj-2026-0302-001",
        "execution_status": "NOMINAL",
        "actual_dv_m_s": 0.347,
        "actual_dv_rtn_m_s": [0.002, 0.347, -0.001],
        "attitude_error_at_burn_deg": 0.31,
        "execution_error": {
            "dv_magnitude_error_m_s": -0.003,
            "dv_magnitude_sigma_m_s": 0.007,
            "pointing_error_deg": 0.31,
            "p_burn_rtn_km2": _P_BURN,
        },
        "post_burn_state": {
            "r_sat_km": [6878.0, 0.0, 0.0],
            "v_sat_km_s": [0.0, 7.6127, 0.0],
            "p_post_km2": _P_POST,
            "epoch_utc": "2026-04-01T15:45:03Z",
            "m2_post_estimated": 31.4,
        },
        "reported_at_utc": "2026-04-01T15:45:10Z",
    }
    base.update(overrides)
    return base


def assessed(report=None, **kwargs):
    defaults = dict(
        commanded_dv_m_s=0.35,
        commanded_dv_rtn_m_s=[0.0, 0.35, 0.0],
        r_rel_km_at_tca=[0.4, 0.1, 0.0],
        v_rel_km_s_at_tca=[0.0, -7.5, 0.1],
        p_pre_km2=_P_PRE,
        hbr_m=15.0,
        pc_pre=0.000312,
        pc_monitor_threshold=1.0e-5,
    )
    defaults.update(kwargs)
    return assess_post_burn(report or a_report(), **defaults)


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------


class TestContractConformance:
    def test_the_report_under_test_is_a_real_gnc_report(self):
        validate_against_contract("GNCReport", a_report())

    def test_the_ack_validates_against_the_contract(self):
        validate_against_contract("GNCReportAck", build_report_ack(assessed()))

    def test_an_aborted_report_with_abort_detail_validates(self):
        report = a_report(
            execution_status="ABORTED",
            actual_dv_m_s=0.0,
            actual_dv_rtn_m_s=[0.0, 0.0, 0.0],
            abort_detail={
                "abort_reason": "ATTITUDE_ERROR_EXCEEDED",
                "abort_time_utc": "2026-04-01T15:45:01Z",
                "flight_mode_at_abort": "M3",
                "dv_achieved_before_abort_m_s": 0.0,
            },
        )

        validate_against_contract("GNCReport", report)

    def test_the_ack_carries_only_contract_fields(self):
        from common.gnc_contract import load_gnc_schema

        allowed = set(load_gnc_schema("GNCReportAck")["properties"])

        assert set(build_report_ack(assessed())) <= allowed


# ---------------------------------------------------------------------------
# P_post = P_pre + P_burn, and who computes the Pc
# ---------------------------------------------------------------------------


class TestPostBurnCovariance:
    def test_gncs_own_post_burn_covariance_is_authoritative(self):
        """Section 9: PostBurnState.p_post_km2 came from the flight computer's
        own OD and outranks anything assembled here."""
        p_post = post_burn_covariance_km2(a_report(), _P_PRE)

        assert p_post == pytest.approx(np.array(_P_POST).reshape(3, 3))

    def test_without_a_post_burn_covariance_it_is_p_pre_plus_p_burn(self):
        """The section 9 execution-error note: P_post = P_pre + P_burn."""
        report = a_report()
        report["post_burn_state"].pop("p_post_km2")
        p_post = post_burn_covariance_km2(report, _P_PRE)

        expected = np.array(_P_PRE).reshape(3, 3) + np.array(_P_BURN).reshape(3, 3)
        assert p_post == pytest.approx(expected)

    def test_an_unreadable_covariance_is_none_not_zeros(self):
        """A covariance of zeros would read as perfect knowledge."""
        report = a_report()
        report["post_burn_state"]["p_post_km2"] = [0.0] * 4   # wrong length
        report["execution_error"]["p_burn_rtn_km2"] = None

        assert post_burn_covariance_km2(report, None) is None

    def test_a_non_finite_covariance_is_rejected(self):
        report = a_report()
        report["post_burn_state"]["p_post_km2"] = [float("nan")] + [0.0] * 8
        report["execution_error"] = {}

        assert post_burn_covariance_km2(report, None) is None

    def test_the_source_of_the_covariance_is_recorded(self):
        with_gnc = assessed()
        assert with_gnc.post_maneuver_od["p_post_source"] == "gnc_post_burn_state"

        report = a_report()
        report["post_burn_state"].pop("p_post_km2")
        assembled = assessed(report)
        assert assembled.post_maneuver_od["p_post_source"] == "p_pre_plus_p_burn"

    def test_case_2_rotates_p_burn_from_rtn_to_eci_before_summing(self):
        """SCRUM-428. p_burn_rtn_km2 is RTN, p_pre is ECI; summing them
        naively would silently produce a frame-inconsistent P_post whenever
        the burn covariance is non-isotropic and the state is not
        axis-aligned to ECI. Neither condition holds in this file's shared
        fixture (_P_BURN is isotropic, a_report()'s r_sat/v_sat are
        axis-aligned), which is exactly why that fixture cannot exercise
        this fix -- rotating an isotropic matrix, or rotating at the
        identity, both leave a matrix unchanged either way. This test uses
        a deliberately non-isotropic p_burn and a non-axis-aligned state,
        and checks against a rotation built by hand from the RTN
        definition, not by calling aps_math.frames, so it verifies the
        actual physics rather than the module's self-consistency.
        """
        r_sat = np.array([6878.0, 500.0, 300.0])
        v_sat = np.array([-0.05, 7.5, 0.4])

        p_burn_rtn_flat = [
            4.0e-5, 1.0e-5, 0.0,
            1.0e-5, 1.0e-5, 0.0,
            0.0, 0.0, 2.0e-6,
        ]

        report = a_report()
        report["post_burn_state"]["r_sat_km"] = r_sat.tolist()
        report["post_burn_state"]["v_sat_km_s"] = v_sat.tolist()
        report["post_burn_state"].pop("p_post_km2")
        report["execution_error"]["p_burn_rtn_km2"] = p_burn_rtn_flat

        p_post = post_burn_covariance_km2(report, _P_PRE)

        # Independent rotation: R = r_hat, N = (r x v)/|r x v|, T = N x R.
        r_hat = r_sat / np.linalg.norm(r_sat)
        h = np.cross(r_sat, v_sat)
        n_hat = h / np.linalg.norm(h)
        t_hat = np.cross(n_hat, r_hat)
        rot = np.column_stack([r_hat, t_hat, n_hat])

        p_burn_rtn_matrix = np.array(p_burn_rtn_flat).reshape(3, 3)
        expected_p_burn_eci = rot @ p_burn_rtn_matrix @ rot.T
        expected = np.array(_P_PRE).reshape(3, 3) + expected_p_burn_eci

        assert p_post == pytest.approx(expected)

        # Explicitly not the naive, unrotated sum -- the trap this closes.
        naive = np.array(_P_PRE).reshape(3, 3) + p_burn_rtn_matrix
        assert not np.allclose(p_post, naive)

    def test_case_2_without_r_sat_v_sat_returns_none_rather_than_guess(self):
        """SCRUM-428. A covariance this function cannot honestly place in
        one frame is not a covariance to hand to compute_pc_post."""
        report = a_report()
        report["post_burn_state"].pop("p_post_km2")
        report["post_burn_state"].pop("r_sat_km")
        report["post_burn_state"].pop("v_sat_km_s")

        assert post_burn_covariance_km2(report, _P_PRE) is None


class TestPcPostIsDelegated:
    def test_the_post_maneuver_pc_comes_from_the_scorer(self):
        """The ticket's boundary: 382 feeds compute_pc_post, it does not
        reimplement the physics."""
        with patch.object(
            gnc_report, "compute_pc_post", return_value=4.2e-7
        ) as scorer:
            assessment = assessed()

        scorer.assert_called_once()
        assert assessment.pc_post == 4.2e-7

    def test_the_scorer_is_given_the_post_burn_state_not_the_pre_burn_one(self):
        with patch.object(gnc_report, "compute_pc_post", return_value=1e-9) as scorer:
            assessed()

        r_sat = scorer.call_args[0][0]
        assert r_sat == pytest.approx(np.array([6878.0, 0.0, 0.0]))

    def test_the_secondary_is_reconstructed_relative_to_the_post_burn_primary(self):
        """compute_pc_post takes the secondary's absolute position, and the
        report gives the primary's. r2 = r_post + r_rel."""
        with patch.object(gnc_report, "compute_pc_post", return_value=1e-9) as scorer:
            assessed()

        r_post_secondary = scorer.call_args[0][2]
        assert r_post_secondary == pytest.approx(
            np.array([6878.0, 0.0, 0.0]) + np.array([0.4, 0.1, 0.0])
        )

    def test_a_real_pc_is_established_end_to_end(self):
        """Not mocked: the real scorer on the real geometry."""
        assessment = assessed()

        assert assessment.pc_post is not None
        assert 0.0 <= assessment.pc_post <= 1.0

    def test_no_post_burn_state_means_no_pc_not_a_pc_of_zero(self):
        report = a_report(execution_status="ABORTED")
        report.pop("post_burn_state")
        assessment = assessed(report)

        assert assessment.pc_post is None

    def test_a_scorer_failure_does_not_raise(self):
        with patch.object(
            gnc_report, "compute_pc_post", side_effect=RuntimeError("boom")
        ):
            assessment = assessed()

        assert assessment.pc_post is None


# ---------------------------------------------------------------------------
# Section 10 producers
# ---------------------------------------------------------------------------


class TestSectionTenProducers:
    def test_the_three_producers_are_the_ones_the_catalogue_assigns_to_382(self):
        """evidence_record.FIELD_PRODUCERS already names them."""
        assigned = {
            field for field, owner in FIELD_PRODUCERS.items()
            if owner == "SCRUM-382"
        }

        assert set(assessed().evidence_values()) <= assigned

    def test_the_evidence_values_are_accepted_by_the_record_builder(self):
        """They must be real catalogue fields, or EvidenceRecord.build rejects
        them."""
        from common.evidence_record import GENESIS_HASH, build_decision_record

        record = build_decision_record(
            "SAT-382", 0, GENESIS_HASH, values=assessed().evidence_values()
        )

        assert record.value("residual_risk")["replan_required"] is False

    def test_actual_vs_predicted_compares_commanded_with_actual(self):
        block = assessed().actual_vs_predicted

        assert block["commanded_dv_m_s"] == 0.35
        assert block["actual_dv_m_s"] == 0.347
        assert block["dv_delta_m_s"] == pytest.approx(-0.003)
        assert block["attitude_error_at_burn_deg"] == 0.31

    def test_the_execution_error_block_is_carried_through_verbatim(self):
        """SCRUM-365's block. 382 does not compute or reshape it."""
        block = assessed().actual_vs_predicted["execution_error"]

        assert block == a_report()["execution_error"]

    def test_post_maneuver_od_carries_the_post_burn_state(self):
        block = assessed().post_maneuver_od

        assert block["epoch_utc"] == "2026-04-01T15:45:03Z"
        assert block["r_sat_km"] == [6878.0, 0.0, 0.0]
        assert block["p_post_km2"] == _P_POST
        assert block["m2_post_estimated"] == 31.4

    def test_residual_risk_carries_both_pcs_and_the_threshold(self):
        block = assessed().residual_risk

        assert block["pc_pre"] == 0.000312
        assert block["pc_post"] is not None
        assert block["pc_monitor_threshold"] == 1.0e-5

    def test_an_abort_detail_reaches_residual_risk(self):
        report = a_report(
            execution_status="ABORTED",
            abort_detail={
                "abort_reason": "WATCHDOG_EXPIRED",
                "abort_time_utc": "2026-04-01T15:45:01Z",
                "flight_mode_at_abort": "M3",
            },
        )
        block = assessed(report).residual_risk

        assert block["abort_detail"]["abort_reason"] == "WATCHDOG_EXPIRED"


# ---------------------------------------------------------------------------
# The replan decision (guard doc section 3, the M3 rows)
# ---------------------------------------------------------------------------


class TestReplanDecision:
    def test_a_nominal_burn_that_clears_the_event_needs_no_replan(self):
        """M3 to M0: nominal, and the residual is below the monitor line."""
        with patch.object(gnc_report, "compute_pc_post", return_value=1e-9):
            assessment = assessed()

        assert assessment.replan_required is False
        assert assessment.replan_note == ""

    def test_a_nominal_burn_with_pc_still_elevated_needs_a_replan(self):
        """M3 to M1: nominal, but Pc is still at or above the monitor line."""
        with patch.object(gnc_report, "compute_pc_post", return_value=3.0e-5):
            assessment = assessed()

        assert assessment.replan_required is True
        assert "monitor line" in assessment.replan_note

    def test_exactly_at_the_monitor_line_still_needs_a_replan(self):
        """Section 3 writes Pc >= pc_monitor_threshold. Inclusive."""
        with patch.object(gnc_report, "compute_pc_post", return_value=1.0e-5):
            assessment = assessed()

        assert assessment.replan_required is True

    @pytest.mark.parametrize("status", ["PARTIAL", "ABORTED"])
    def test_a_non_nominal_execution_always_needs_a_replan(self, status):
        """The commanded burn was not delivered, so the risk was not bought
        down, whatever the post-burn Pc happens to say."""
        with patch.object(gnc_report, "compute_pc_post", return_value=1e-12):
            assessment = assessed(a_report(execution_status=status))

        assert assessment.replan_required is True
        assert status in assessment.replan_note

    def test_an_abort_reason_is_named_in_the_replan_note(self):
        report = a_report(
            execution_status="ABORTED",
            abort_detail={
                "abort_reason": "PARTIAL_BURN_CUTOFF",
                "abort_time_utc": "2026-04-01T15:45:01Z",
                "flight_mode_at_abort": "M3",
            },
        )
        assessment = assessed(report)

        assert "PARTIAL_BURN_CUTOFF" in assessment.replan_note

    def test_a_nominal_burn_we_cannot_rescore_needs_a_replan(self):
        """A nominal burn whose residual risk could not be established is not a
        cleared conjunction."""
        with patch.object(gnc_report, "compute_pc_post", return_value=None):
            assessment = assessed()

        assert assessment.replan_required is True
        assert "no post-burn Pc" in assessment.replan_note

    def test_the_ack_reports_the_replan_decision(self):
        with patch.object(gnc_report, "compute_pc_post", return_value=3.0e-5):
            ack = build_report_ack(assessed())

        assert ack["replan_required"] is True
        assert ack["replan_note"]
        validate_against_contract("GNCReportAck", ack)

    def test_the_ack_echoes_the_command_and_conjunction_ids(self):
        ack = build_report_ack(assessed())

        assert ack["command_id"] == "gnc-cmd-0001"
        assert ack["conjunction_id"] == "conj-2026-0302-001"
        assert ack["received_at_utc"]
