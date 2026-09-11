"""SCRUM-383 MAF policy-flip capstone integration tests."""

from __future__ import annotations

import pytest

from common.authorization_envelope import AuthorityLevel
from common.evidence_record import FieldState
from maf_policy_flip_demo import (
    BASE_PC_ACTION,
    STRICT_PC_ACTION,
    run_demo,
)


@pytest.fixture(scope="module")
def demo():
    return run_demo()


def _case(demo, authority: str, pc_action: float):
    return next(
        case
        for case in demo["cases"]
        if case["policy"]["authority_level"] == authority
        and case["policy"]["pc_action"] == pc_action
    )


def test_demo_now_runs_through_real_authorized_execution(demo):
    assert demo["scope"] == "through-authorized-execution"
    assert demo["downstream_dependency"] == "SCRUM-382"


def test_fixed_scrum_383_scenario_is_preserved_across_policy_matrix(demo):
    physical_inputs = [
        case["physical_input"]
        for case in demo["cases"]
    ]

    assert all(
        physical == physical_inputs[0]
        for physical in physical_inputs
    )

    assert (
        demo["cases"][0]["scenario"]["conjunction_id"]
        == "SCRUM-383-POLICY-FLIP"
    )
    assert demo["cases"][0]["scenario"]["pc"] == pytest.approx(
        3.001e-5
    )

    physical = demo["cases"][0]["physical_input"]

    assert (
        physical["satellite"]["t_burn_utc"]
        == "2026-03-02T14:00:00Z"
    )
    assert (
        physical["conjunction"]["t_ca_utc"]
        == "2026-03-02T18:00:00Z"
    )


def test_pc_action_policy_flip_changes_real_planner_decision(demo):
    base = _case(
        demo,
        AuthorityLevel.L1.value,
        BASE_PC_ACTION,
    )
    strict = _case(
        demo,
        AuthorityLevel.L1.value,
        STRICT_PC_ACTION,
    )

    assert base["planner"]["direction"] == "no-burn"
    assert base["planner"]["dv_magnitude_m_s"] == 0.0

    assert strict["planner"]["direction"] != "no-burn"
    assert strict["planner"]["dv_magnitude_m_s"] > 0.0

    assert (
        base["planner"]["direction"]
        != strict["planner"]["direction"]
    )


@pytest.mark.parametrize(
    "authority",
    [
        AuthorityLevel.L0.value,
        AuthorityLevel.L1.value,
        AuthorityLevel.L2.value,
    ],
)
def test_same_policy_flip_is_visible_at_each_authority(
    demo,
    authority,
):
    base = _case(
        demo,
        authority,
        BASE_PC_ACTION,
    )
    strict = _case(
        demo,
        authority,
        STRICT_PC_ACTION,
    )

    assert base["planner"]["direction"] == "no-burn"
    assert strict["planner"]["direction"] != "no-burn"


@pytest.mark.parametrize(
    "authority",
    [
        AuthorityLevel.L0.value,
        AuthorityLevel.L1.value,
        AuthorityLevel.L2.value,
    ],
)
def test_real_authorization_envelope_is_compiled_approved_and_active(
    demo,
    authority,
):
    case = _case(
        demo,
        authority,
        STRICT_PC_ACTION,
    )

    envelope = case["authorization_envelope"]

    assert envelope["authority_level"] == authority
    assert envelope["effective_authority"] == authority

    # SCRUM-375 first-flight authentication is a keyed MAC,
    # not a digital signature.
    assert envelope["alg"] == "HMAC-SHA256"
    assert envelope["mac_verified"] is True
    assert envelope["envelope_version"]


def test_policy_change_produces_a_different_envelope_version(demo):
    base = _case(
        demo,
        AuthorityLevel.L2.value,
        BASE_PC_ACTION,
    )
    strict = _case(
        demo,
        AuthorityLevel.L2.value,
        STRICT_PC_ACTION,
    )

    assert (
        base["authorization_envelope"]["envelope_version"]
        != strict["authorization_envelope"]["envelope_version"]
    )


def test_real_validity_assessor_is_wired_into_every_run(demo):
    for case in demo["cases"]:
        validity = case["validity"]

        assert validity["status"] == "EARNED"
        assert (
            validity["epsilon"]
            > validity["epsilon_threshold"]
        )
        assert validity["epsilon_threshold"] == pytest.approx(
            0.20
        )
        assert validity["weak_directions"] == []


def test_every_run_emits_a_valid_chained_scrum_377_evidence_package(
    demo,
):
    for case in demo["cases"]:
        evidence = case["evidence"]

        assert evidence["hash_matches"] is True
        assert evidence["chain_verified"] is True
        assert evidence["record_count"] >= 2
        assert evidence["transition_record_ids"]


@pytest.mark.parametrize(
    "authority",
    [
        AuthorityLevel.L0.value,
        AuthorityLevel.L1.value,
        AuthorityLevel.L2.value,
    ],
)
def test_loose_policy_never_reaches_authorized_execution(
    demo,
    authority,
):
    case = _case(
        demo,
        authority,
        BASE_PC_ACTION,
    )

    assert case["planner"]["direction"] == "no-burn"
    assert (
        case["state_machine"]["authorized_execution"]
        is None
    )
    assert (
        case["state_machine"]["final_mode"]
        == "M1"
    )


def test_l0_strict_policy_remains_advisory(demo):
    case = _case(
        demo,
        AuthorityLevel.L0.value,
        STRICT_PC_ACTION,
    )

    assert case["planner"]["direction"] != "no-burn"
    assert (
        case["state_machine"]["authorized_execution"]
        is None
    )
    assert case["state_machine"]["final_mode"] == "M1"

    steps = case["state_machine"]["steps"]

    assert steps[0]["from_mode"] == "M0"
    assert steps[0]["to_mode"] == "M1"

    failed = [
        guard["guard"]
        for guard in steps[-1]["guards"]
        if not guard["passed"]
    ]
    assert "authority_l1_or_l2" in failed


def test_l1_strict_policy_runs_m1_to_m2_to_m3_with_operator_approval(
    demo,
):
    case = _case(
        demo,
        AuthorityLevel.L1.value,
        STRICT_PC_ACTION,
    )

    assert case["state_machine"]["final_mode"] == "M3"

    execution = case["state_machine"]["authorized_execution"]

    assert execution is not None
    assert execution["mode"] == "M3"
    assert execution["authority_level"] == "L1"
    assert (
        execution["approval_basis"]
        == "l1_operator_approval"
    )
    assert execution["validity_status"] == "EARNED"

    transitions = [
        (step["from_mode"], step["to_mode"])
        for step in case["state_machine"]["steps"]
        if step["changed_mode"]
    ]

    assert transitions == [
        ("M0", "M1"),
        ("M1", "M2"),
        ("M2", "M3"),
    ]


def test_l2_strict_policy_runs_m1_to_m2_to_m3_after_veto_window(
    demo,
):
    case = _case(
        demo,
        AuthorityLevel.L2.value,
        STRICT_PC_ACTION,
    )

    assert case["state_machine"]["final_mode"] == "M3"

    execution = case["state_machine"]["authorized_execution"]

    assert execution is not None
    assert execution["mode"] == "M3"
    assert execution["authority_level"] == "L2"
    assert (
        execution["approval_basis"]
        == "l2_veto_window_expired"
    )
    assert execution["validity_status"] == "EARNED"

    transitions = [
        (step["from_mode"], step["to_mode"])
        for step in case["state_machine"]["steps"]
        if step["changed_mode"]
    ]

    assert transitions == [
        ("M0", "M1"),
        ("M1", "M2"),
        ("M2", "M3"),
    ]


@pytest.mark.parametrize(
    "authority",
    [
        AuthorityLevel.L1.value,
        AuthorityLevel.L2.value,
    ],
)
def test_authorized_execution_preserves_the_planner_burn(
    demo,
    authority,
):
    case = _case(
        demo,
        authority,
        STRICT_PC_ACTION,
    )

    execution = case["state_machine"]["authorized_execution"]

    assert execution is not None
    assert execution["dv_magnitude_m_s"] == pytest.approx(
        case["planner"]["dv_magnitude_m_s"]
    )


def test_scrum_379_transition_evidence_is_real_not_pending(demo):
    case = _case(
        demo,
        AuthorityLevel.L2.value,
        STRICT_PC_ACTION,
    )

    evidence = case["evidence"]
    fields = evidence["fields"]

    assert fields["from_mode"]["state"] == FieldState.PRESENT
    assert fields["from_mode"]["value"] == "M2"

    assert fields["to_mode"]["state"] == FieldState.PRESENT
    assert fields["to_mode"]["value"] == "M3"

    assert fields["trigger"]["state"] == FieldState.PRESENT

    assert (
        fields["monitor_results"]["state"]
        == FieldState.PRESENT
    )
    assert fields["monitor_results"]["value"]

    assert "SCRUM-379" not in evidence["pending_producers"]


def test_scrum_382_outputs_remain_explicitly_pending(demo):
    case = _case(
        demo,
        AuthorityLevel.L2.value,
        STRICT_PC_ACTION,
    )

    evidence = case["evidence"]
    fields = evidence["fields"]

    for field_name in (
        "commands_and_acknowledgments",
        "actual_vs_predicted",
        "post_maneuver_od",
        "residual_risk",
    ):
        assert (
            fields[field_name]["state"]
            == FieldState.PRODUCER_NOT_IMPLEMENTED
        )
        assert (
            fields[field_name]["producer"]
            == "SCRUM-382"
        )

    assert "SCRUM-382" in evidence["pending_producers"]


def test_evidence_contains_real_validity_and_envelope_outputs(demo):
    case = _case(
        demo,
        AuthorityLevel.L2.value,
        STRICT_PC_ACTION,
    )

    fields = case["evidence"]["fields"]

    assert fields["validity_status"]["state"] == FieldState.PRESENT
    assert fields["validity_status"]["value"] == "EARNED"

    assert fields["validity_epsilon"]["state"] == FieldState.PRESENT
    assert fields["validity_epsilon"]["value"] > 0.20

    assert fields["authority_level"]["state"] == FieldState.PRESENT
    assert fields["authority_level"]["value"] == "L2"

    assert fields["envelope_version"]["state"] == FieldState.PRESENT
    assert fields["envelope_version"]["value"]

    assert (
        fields["envelope_approving_identity"]["state"]
        == FieldState.PRESENT
    )
