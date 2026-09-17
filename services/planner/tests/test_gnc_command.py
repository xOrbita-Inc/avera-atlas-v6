"""
SCRUM-382 -- assembling the GNCCommand from an AuthorizedExecution.

MAF v2.0 section 9. Every command here is validated against
openapi/gnc_interface.yaml itself, not against a restatement of its field
names: the contract is final and is the authority, so a test that agreed with
the builder but disagreed with the file would be worthless.

The frame conversion is the substance. AuthorizedExecution carries dv in ECI
km/s (SCRUM-379); ManeuverSpec.dv_rtn_m_s is an RTN burn in m/s. The rotation
uses the existing convention in libs/aps_math.frames -- R = r/|r|,
N = (r x v)/|r x v|, T = N x R, which is what the contract's own frame section
states -- and is checked against an independently constructed basis rather than
against the builder's own arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest

from aps_math import frames
from common.decision_state_machine import AuthorizedExecution
from common.gnc_command import (
    GNCCommandContext,
    build_gnc_command,
    eci_to_rtn_m_s,
    may_emit_without_further_approval,
    new_command_id,
)
from common.gnc_contract import (
    GNCContractError,
    contract_covariance_source,
    contract_direction,
    validate_against_contract,
)

# A near-circular LEO state at burn time. r along x, v along y, so the RTN
# basis is R = +x, T = +y, N = +z and an RTN component maps to an obvious ECI
# axis -- which makes a rotation error visible by inspection, not only by
# assertion.
R_SAT = (6878.0, 0.0, 0.0)
V_SAT = (0.0, 7.6127, 0.0)

AUTHORIZED = AuthorizedExecution(
    conjunction_id="conj-2026-0302-001",
    dv_eci_km_s=(0.0, 0.00035, 0.0),      # +0.35 m/s along ECI y == along-track
    dv_magnitude_m_s=0.35,
    t_burn_utc="2026-04-01T15:45:00Z",
    mode="M3",
    authority_level="L2",
    envelope_version="env-v1-sha256-abc123456789",
    approval_basis="l2_veto_window_expired",
    validity_status="EARNED",
    validity_epsilon=0.73,
    covariance_source="real_cdm",
    authorized_at_utc="2026-04-01T13:45:00Z",
)


def a_context(**overrides) -> GNCCommandContext:
    base = dict(
        approving_identity="ground-ops",
        r_sat_km=R_SAT,
        v_sat_km_s=V_SAT,
        direction="prograde",
        burn_duration_s=2.4,
        epsilon_threshold=0.20,
        pc_computed=0.000312,
        m2_pre=0.064,
        data_age_s=3600.0,
        t_ca_utc="2026-04-01T21:45:00Z",
        latest_burn_utc="2026-04-01T17:45:00Z",
        veto_window_open_utc="2026-04-01T13:45:00Z",
        veto_window_close_utc="2026-04-01T13:50:00Z",
        safe_action_id="safe-passive-hold-001",
        safe_action_type="passive_hold",
    )
    base.update(overrides)
    return GNCCommandContext(**base)


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------


class TestSchemaConformance:
    def test_a_built_command_validates_against_the_contract(self):
        command = build_gnc_command(AUTHORIZED, a_context())

        validate_against_contract("GNCCommand", command)

    def test_every_required_top_level_field_is_present(self):
        """Quoted from GNCCommand.required in the contract."""
        command = build_gnc_command(AUTHORIZED, a_context())

        for field in (
            "command_id", "conjunction_id", "authority_level", "envelope_version",
            "approving_identity", "validity", "risk", "maneuver", "safe_action",
            "timing",
        ):
            assert field in command, field

    def test_the_command_carries_no_field_the_contract_does_not_define(self):
        """A stray field would pass a permissive validator and be dropped, or
        rejected, at the GNC boundary. Better to notice here."""
        from common.gnc_contract import load_gnc_schema

        allowed = set(load_gnc_schema("GNCCommand")["properties"])
        command = build_gnc_command(AUTHORIZED, a_context())

        assert set(command) <= allowed

    @pytest.mark.parametrize(
        "block, schema_name",
        [
            ("validity", "ValidityVerdict"),
            ("risk", "RiskInputs"),
            ("maneuver", "ManeuverSpec"),
            ("safe_action", "SafeActionRef"),
            ("timing", "TimingBudget"),
        ],
    )
    def test_each_nested_block_validates_on_its_own_schema(self, block, schema_name):
        command = build_gnc_command(AUTHORIZED, a_context())

        validate_against_contract(schema_name, command[block])


# ---------------------------------------------------------------------------
# The ECI to RTN rotation
# ---------------------------------------------------------------------------


class TestFrameConversion:
    def test_the_rotation_matches_an_independently_built_rtn_basis(self):
        """R = r/|r|, N = (r x v)/|r x v|, T = N x R, per the contract's own
        frame section. Built here from the definition rather than by calling
        the same helper the builder calls."""
        r = np.array(R_SAT)
        v = np.array(V_SAT)
        r_hat = r / np.linalg.norm(r)
        n_hat = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
        t_hat = np.cross(n_hat, r_hat)

        dv_eci_km_s = np.array([0.0, 0.00035, 0.0])
        expected_m_s = [
            float(np.dot(dv_eci_km_s, r_hat) * 1000.0),
            float(np.dot(dv_eci_km_s, t_hat) * 1000.0),
            float(np.dot(dv_eci_km_s, n_hat) * 1000.0),
        ]

        assert eci_to_rtn_m_s(dv_eci_km_s, R_SAT, V_SAT) == pytest.approx(expected_m_s)

    def test_an_along_track_burn_lands_in_the_transverse_component(self):
        """With r along x and v along y, a +y ECI burn is purely transverse.
        A rotation error would put the magnitude in R or N instead."""
        dv_rtn = eci_to_rtn_m_s((0.0, 0.00035, 0.0), R_SAT, V_SAT)

        assert dv_rtn[0] == pytest.approx(0.0, abs=1e-9)
        assert dv_rtn[1] == pytest.approx(0.35, abs=1e-9)
        assert dv_rtn[2] == pytest.approx(0.0, abs=1e-9)

    def test_a_radial_burn_lands_in_the_radial_component(self):
        dv_rtn = eci_to_rtn_m_s((0.00035, 0.0, 0.0), R_SAT, V_SAT)

        assert dv_rtn[0] == pytest.approx(0.35, abs=1e-9)
        assert dv_rtn[1] == pytest.approx(0.0, abs=1e-9)

    def test_a_cross_track_burn_lands_in_the_normal_component(self):
        dv_rtn = eci_to_rtn_m_s((0.0, 0.0, 0.00035), R_SAT, V_SAT)

        assert dv_rtn[2] == pytest.approx(0.35, abs=1e-9)

    def test_the_rotation_preserves_magnitude(self):
        """It is a rotation, not a projection: no delta-v may be lost in it."""
        dv_eci = (0.0001, 0.0002, -0.00015)
        dv_rtn = eci_to_rtn_m_s(dv_eci, R_SAT, V_SAT)

        assert float(np.linalg.norm(dv_rtn)) == pytest.approx(
            float(np.linalg.norm(dv_eci)) * 1000.0
        )

    def test_it_uses_the_shared_frame_convention_not_a_local_one(self):
        """The contract makes GNC responsible for rotating RTN back to ECI with
        the same r_sat_km / v_sat_km_s. Round-tripping through the shared helper
        must return the original vector, or the two sides disagree."""
        rot = frames.rtn_to_eci_rotation(np.array(R_SAT), np.array(V_SAT))
        dv_eci_km_s = np.array([0.0001, 0.0002, -0.00015])
        dv_rtn_m_s = np.array(eci_to_rtn_m_s(dv_eci_km_s, R_SAT, V_SAT))

        back_to_eci_km_s = (rot @ dv_rtn_m_s) / 1000.0
        assert back_to_eci_km_s == pytest.approx(dv_eci_km_s)

    def test_the_maneuver_magnitude_is_the_scorers_not_recomputed(self):
        """382 never re-plans: the dv is the scorer's, unchanged."""
        command = build_gnc_command(AUTHORIZED, a_context())

        assert command["maneuver"]["dv_magnitude_m_s"] == 0.35


# ---------------------------------------------------------------------------
# The vocabulary mappings
# ---------------------------------------------------------------------------


class TestVocabularyMapping:
    @pytest.mark.parametrize(
        "scorer, contract_value",
        [
            ("prograde", "prograde"),
            ("retrograde", "retrograde"),
            ("radial", "radial"),
            ("anti-radial", "anti_radial"),
            ("cross-track", "cross_track"),
            ("anti-cross", "anti_cross_track"),
        ],
    )
    def test_every_scorer_direction_maps_to_a_contract_direction(
        self, scorer, contract_value
    ):
        assert contract_direction(scorer) == contract_value

    def test_a_direction_the_contract_has_no_word_for_is_refused(self):
        """Including no-burn: there is no command to build for a burn that was
        not recommended."""
        with pytest.raises(GNCContractError):
            contract_direction("no-burn")

    def test_the_mapped_direction_survives_schema_validation(self):
        command = build_gnc_command(AUTHORIZED, a_context(direction="anti-cross"))

        assert command["maneuver"]["direction"] == "anti_cross_track"
        validate_against_contract("GNCCommand", command)

    def test_a_real_cdm_stays_real_cdm(self):
        assert contract_covariance_source("real_cdm") == "real_cdm"

    def test_the_planners_elliptical_surrogate_maps_to_the_contract_surrogate(self):
        """The planner says surrogate_elliptical; the contract admits only
        real_cdm or surrogate_identity, and GNC does not act on the ellipse's
        shape."""
        assert contract_covariance_source("surrogate_elliptical") == "surrogate_identity"

    def test_an_unknown_provenance_is_never_claimed_as_a_real_cdm(self):
        """The one direction this must not fail in."""
        for label in ("", None, "mystery", "udl"):
            assert contract_covariance_source(label) == "surrogate_identity"

    def test_a_surrogate_backed_command_still_validates(self):
        authorized = AuthorizedExecution(
            **{**AUTHORIZED.to_dict(), "covariance_source": "surrogate_elliptical",
               "dv_eci_km_s": AUTHORIZED.dv_eci_km_s}
        )
        command = build_gnc_command(authorized, a_context())

        assert command["risk"]["covariance_source"] == "surrogate_identity"
        validate_against_contract("GNCCommand", command)


# ---------------------------------------------------------------------------
# Field provenance: what comes from where
# ---------------------------------------------------------------------------


class TestFieldProvenance:
    def test_the_identity_fields_come_from_the_authorized_execution(self):
        command = build_gnc_command(AUTHORIZED, a_context())

        assert command["conjunction_id"] == "conj-2026-0302-001"
        assert command["authority_level"] == "L2"
        assert command["envelope_version"] == "env-v1-sha256-abc123456789"

    def test_the_validity_block_comes_from_the_authorized_execution(self):
        command = build_gnc_command(AUTHORIZED, a_context())

        assert command["validity"]["status"] == "EARNED"
        assert command["validity"]["epsilon"] == 0.73
        assert command["validity"]["epsilon_threshold"] == 0.20

    def test_the_burn_time_comes_from_the_authorized_execution(self):
        command = build_gnc_command(AUTHORIZED, a_context())

        assert command["maneuver"]["burn_time_utc"] == "2026-04-01T15:45:00Z"

    def test_the_approving_identity_comes_from_the_envelope_not_invented(self):
        command = build_gnc_command(AUTHORIZED, a_context())

        assert command["approving_identity"] == "ground-ops"

    def test_a_command_id_is_generated_when_none_is_supplied(self):
        first = build_gnc_command(AUTHORIZED, a_context())
        second = build_gnc_command(AUTHORIZED, a_context())

        assert first["command_id"]
        assert first["command_id"] != second["command_id"]

    def test_a_supplied_command_id_is_used_verbatim(self):
        command = build_gnc_command(AUTHORIZED, a_context(command_id="cmd-0001"))

        assert command["command_id"] == "cmd-0001"

    def test_new_command_ids_are_unique(self):
        assert len({new_command_id() for _ in range(50)}) == 50

    def test_the_timing_budget_carries_the_contract_required_instants(self):
        command = build_gnc_command(AUTHORIZED, a_context())
        timing = command["timing"]

        assert timing["t_ca_utc"] == "2026-04-01T21:45:00Z"
        assert timing["latest_burn_utc"] == "2026-04-01T17:45:00Z"
        assert timing["veto_window_close_utc"] == "2026-04-01T13:50:00Z"
        assert timing["command_issued_utc"]

    def test_the_locked_slew_and_margin_are_reported_not_reinvented(self):
        """Guard doc section 4.3 / section 5: t_slew + t_settle 120 s,
        t_margin 30 s. SCRUM-379 owns them; 382 reports them."""
        command = build_gnc_command(AUTHORIZED, a_context())

        assert command["timing"]["t_slew_s"] == 120.0
        assert command["timing"]["t_margin_s"] == 30.0


# ---------------------------------------------------------------------------
# approval_basis
# ---------------------------------------------------------------------------


class TestApprovalBasis:
    def test_an_l2_veto_expiry_may_emit_directly(self):
        assert may_emit_without_further_approval(AUTHORIZED) is True

    def test_an_l1_authorized_execution_already_carries_its_approval(self):
        """An L1 AuthorizedExecution exists only because an operator approved
        it, so it needs no second approval before emission. The gate is that
        it cannot exist without one."""
        l1 = AuthorizedExecution(
            **{**AUTHORIZED.to_dict(),
               "authority_level": "L1",
               "approval_basis": "l1_operator_approval",
               "dv_eci_km_s": AUTHORIZED.dv_eci_km_s}
        )

        assert may_emit_without_further_approval(l1) is True

    def test_an_unrecognised_approval_basis_is_refused(self):
        """Section 3 has exactly two M2 to M3 rows. A third basis is a bug, and
        emitting a burn on one would mean acting without knowing why it was
        lawful."""
        odd = AuthorizedExecution(
            **{**AUTHORIZED.to_dict(), "approval_basis": "because_i_said_so",
               "dv_eci_km_s": AUTHORIZED.dv_eci_km_s}
        )

        assert may_emit_without_further_approval(odd) is False

    def test_an_l1_command_records_a_zero_length_veto_window(self):
        """The contract requires both veto instants, and section 5 gives L1 no
        veto window. A zero-length window at the issue instant is the honest
        encoding; inventing a window L1 does not have would be worse."""
        l1 = AuthorizedExecution(
            **{**AUTHORIZED.to_dict(),
               "authority_level": "L1",
               "approval_basis": "l1_operator_approval",
               "dv_eci_km_s": AUTHORIZED.dv_eci_km_s}
        )
        command = build_gnc_command(
            l1, a_context(veto_window_open_utc="", veto_window_close_utc="")
        )

        timing = command["timing"]
        assert timing["veto_window_open_utc"] == timing["command_issued_utc"]
        assert timing["veto_window_close_utc"] == timing["command_issued_utc"]
        validate_against_contract("GNCCommand", command)
