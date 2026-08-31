"""SCRUM-383 partial capstone integration tests."""

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
    assert base["planner"]["utility"] == 0.0

    assert strict["planner"]["direction"] == "prograde"
    assert strict["planner"]["dv_magnitude_m_s"] == pytest.approx(
        1.0
    )
    assert strict["planner"]["utility"] > 0.0

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
def test_same_policy_flip_is_visible_at_each_compiled_authority(
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
    assert strict["planner"]["direction"] == "prograde"


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
        assert validity["epsilon"] > validity["epsilon_threshold"]
        assert validity["epsilon_threshold"] == pytest.approx(0.20)
        assert validity["weak_directions"] == []
        assert validity["phenomenologies_used"] == ["TLE"]


def test_every_run_emits_a_valid_scrum_377_evidence_record(demo):
    for case in demo["cases"]:
        evidence = case["evidence"]

        assert evidence["hash_matches"] is True
        assert evidence["chain_verified"] is True
        assert evidence["content_hash"]


def test_evidence_contains_real_375_and_378_outputs(demo):
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
    assert (
        fields["envelope_version"]["value"]
        == case["authorization_envelope"]["envelope_version"]
    )

    assert (
        fields["envelope_approving_identity"]["state"]
        == FieldState.PRESENT
    )


def test_missing_runtime_dependencies_are_not_fabricated(demo):
    case = _case(
        demo,
        AuthorityLevel.L2.value,
        STRICT_PC_ACTION,
    )

    fields = case["evidence"]["fields"]

    # A decision record is not itself a mode transition.
    assert fields["from_mode"]["state"] == FieldState.NOT_APPLICABLE
    assert fields["to_mode"]["state"] == FieldState.NOT_APPLICABLE

    # SCRUM-379 has not supplied the runtime monitor/state-machine values.
    assert (
        fields["monitor_results"]["state"]
        == FieldState.PRODUCER_NOT_IMPLEMENTED
    )
    assert fields["monitor_results"]["producer"] == "SCRUM-379"

    # SCRUM-382 has not supplied runtime GNC execution values.
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
        assert fields[field_name]["producer"] == "SCRUM-382"


def test_completed_foundation_outputs_are_not_still_pending(demo):
    case = _case(
        demo,
        AuthorityLevel.L2.value,
        STRICT_PC_ACTION,
    )

    pending = case["evidence"]["pending_producers"]

    assert "SCRUM-375" not in pending
    assert "SCRUM-378" not in pending

    assert "SCRUM-379" in pending
    assert "SCRUM-382" in pending
