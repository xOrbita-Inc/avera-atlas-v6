from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import sys

import pytest

PLANNER_ROOT = Path(__file__).resolve().parents[1]
if str(PLANNER_ROOT) not in sys.path:
    sys.path.insert(0, str(PLANNER_ROOT))

from common.authorization_envelope import (
    LOCKED_L1_DV_CAP_M_S,
    LOCKED_L2_DV_CAP_M_S,
    LOCKED_L2_RAISED_DV_CAP_M_S,
    LOCKED_MAX_MANEUVERS_PER_WEEK,
    LOCKED_MAX_TCA_HOURS,
    LOCKED_MIN_MISS_DISTANCE_KM,
    LOCKED_MIN_TCA_HOURS,
    LOCKED_PC_ACTION_CEILING,
    LOCKED_RESERVED_DV_M_S,
    AuthorityLevel,
    AuthorizationEnvelopeManager,
    EnvelopeActivationError,
    EnvelopeApprovalError,
    EnvelopeCompileError,
    ManeuverCapacityInput,
    OperatorAuthorizationProfile,
    RiskBudgetInput,
    approve_authorization_envelope,
    compile_authorization_envelope,
    compile_from_operator_policy,
    validate_section6_safety_floors,
    verify_approved_envelope,
)

NOW = datetime(2026, 8, 7, 15, 0, tzinfo=timezone.utc)
KEY = b"scrum-375-test-key-not-for-production"
KEY_ID = "ground-envelope-key-001"


def _profile(
    *,
    pc_action: float = 1.0e-4,
    pc_watch_min: float = 1.0e-5,
    authority_level: AuthorityLevel = AuthorityLevel.L2,
    l2_raise_to_1_m_s: bool = False,
    profile_id: str = "profile-a",
    max_dv_per_burn_m_s: float = 10.0,
    v_remaining_m_s: float = 50.0,
    v_reserved_m_s: float = 5.0,
    max_maneuvers_per_week: int = 5,
) -> OperatorAuthorizationProfile:
    return OperatorAuthorizationProfile(
        profile_id=profile_id,
        operator_id="DEFAULT_LEO",
        mission_class="first_flight_leo",
        risk_budget=RiskBudgetInput(
            risk_budget_id="nominal-risk-budget",
            pc_watch_min=pc_watch_min,
        ),
        maneuver_capacity=ManeuverCapacityInput(
            max_dv_per_burn_m_s=max_dv_per_burn_m_s,
            v_remaining_m_s=v_remaining_m_s,
            v_reserved_m_s=v_reserved_m_s,
            max_maneuvers_per_week=max_maneuvers_per_week,
        ),
        pc_action=pc_action,
        authority_level=authority_level,
        l2_raise_to_1_m_s=l2_raise_to_1_m_s,
    )


def _approved(profile: OperatorAuthorizationProfile):
    envelope = compile_authorization_envelope(profile)
    approved = approve_authorization_envelope(
        envelope,
        approving_identity="operator:xorb-ops-001",
        mac_key=KEY,
        key_id=KEY_ID,
        human_review_confirmed=True,
        approved_at=NOW,
    )
    return envelope, approved


def test_compile_is_deterministic() -> None:
    first = compile_authorization_envelope(_profile())
    second = compile_authorization_envelope(_profile())
    assert first.to_dict() == second.to_dict()
    assert first.content_hash == second.content_hash
    assert first.version == second.version


def test_watch_line_uses_action_over_ten_when_higher() -> None:
    envelope = compile_authorization_envelope(
        _profile(pc_action=8.0e-5, pc_watch_min=1.0e-6)
    )
    assert envelope.pc_watch == pytest.approx(8.0e-6)
    assert envelope.pc_watch < envelope.pc_action


def test_watch_line_uses_floor_when_floor_is_higher() -> None:
    envelope = compile_authorization_envelope(
        _profile(pc_action=8.0e-5, pc_watch_min=2.0e-5)
    )
    assert envelope.pc_watch == pytest.approx(2.0e-5)
    assert envelope.pc_watch < envelope.pc_action


def test_watch_floor_meeting_action_line_is_rejected() -> None:
    with pytest.raises(EnvelopeCompileError, match="strictly below"):
        compile_authorization_envelope(
            _profile(pc_action=5.0e-5, pc_watch_min=5.0e-5)
        )


def test_pc_action_laxer_than_locked_ceiling_is_rejected() -> None:
    with pytest.raises(EnvelopeCompileError, match="locked 1e-4"):
        _profile(pc_action=LOCKED_PC_ACTION_CEILING + 1.0e-12)


def test_compiled_envelope_carries_locked_spec_values() -> None:
    envelope = compile_authorization_envelope(_profile())
    assert envelope.min_miss_distance_km == LOCKED_MIN_MISS_DISTANCE_KM
    assert envelope.min_tca_hours == LOCKED_MIN_TCA_HOURS
    assert envelope.max_tca_hours == LOCKED_MAX_TCA_HOURS
    assert envelope.l1_dv_cap_m_s == LOCKED_L1_DV_CAP_M_S
    assert envelope.l2_dv_cap_m_s == LOCKED_L2_DV_CAP_M_S
    assert envelope.max_maneuvers_per_week == LOCKED_MAX_MANEUVERS_PER_WEEK
    assert envelope.reserved_dv_m_s == LOCKED_RESERVED_DV_M_S


def test_l2_raise_uses_only_locked_raised_cap() -> None:
    envelope = compile_authorization_envelope(
        _profile(l2_raise_to_1_m_s=True)
    )
    assert envelope.l2_dv_cap_m_s == LOCKED_L2_RAISED_DV_CAP_M_S


def test_capability_can_only_tighten_execution_limits() -> None:
    envelope = compile_authorization_envelope(
        _profile(
            max_dv_per_burn_m_s=0.4,
            v_remaining_m_s=10.0,
            v_reserved_m_s=6.0,
            max_maneuvers_per_week=2,
        )
    )
    assert envelope.l1_dv_cap_m_s == 2.0
    assert envelope.l2_dv_cap_m_s == 0.5
    assert envelope.effective_l1_dv_cap_m_s == pytest.approx(0.4)
    assert envelope.effective_l2_dv_cap_m_s == pytest.approx(0.4)
    assert envelope.effective_max_maneuvers_per_week == 2
    assert envelope.effective_reserved_dv_m_s == pytest.approx(6.0)


def test_human_review_is_required_before_keyed_mac() -> None:
    envelope = compile_authorization_envelope(_profile())
    with pytest.raises(EnvelopeApprovalError, match="human review"):
        approve_authorization_envelope(
            envelope,
            approving_identity="operator:xorb-ops-001",
            mac_key=KEY,
            key_id=KEY_ID,
            human_review_confirmed=False,
            approved_at=NOW,
        )


def test_keyed_mac_verifies_and_exposes_evidence_fields() -> None:
    envelope, approved = _approved(_profile())
    assert verify_approved_envelope(approved, mac_key=KEY, expected_key_id=KEY_ID)
    assert approved.alg == "HMAC-SHA256"
    assert approved.key_id == KEY_ID
    assert envelope.evidence_fields(approved.approving_identity) == {
        "authority_level": "L2",
        "envelope_version": envelope.version,
        "envelope_approving_identity": "operator:xorb-ops-001",
    }


def test_keyed_mac_verification_fails_for_wrong_key() -> None:
    _, approved = _approved(_profile())
    assert not verify_approved_envelope(approved, mac_key=b"wrong-key")


def test_mac_payload_binds_canonical_envelope_version_and_approver() -> None:
    envelope, approved = _approved(_profile())
    payload = approved.mac_payload()

    assert payload["canonical_envelope"] == envelope.canonical_payload()
    assert payload["envelope_version"] == envelope.version
    assert payload["approving_identity"] == "operator:xorb-ops-001"
    assert payload["alg"] == "HMAC-SHA256"
    assert payload["key_id"] == KEY_ID


def test_key_id_is_authenticated_and_can_be_checked() -> None:
    _, approved = _approved(_profile())
    tampered = replace(approved, key_id="ground-envelope-key-999")

    assert not verify_approved_envelope(tampered, mac_key=KEY)
    assert not verify_approved_envelope(
        approved,
        mac_key=KEY,
        expected_key_id="ground-envelope-key-999",
    )


def test_approval_does_not_activate_authority() -> None:
    envelope, approved = _approved(_profile())
    manager = AuthorizationEnvelopeManager()
    manager.stage(envelope)
    manager.record_approval(approved, mac_key=KEY)
    assert manager.effective_authority(AuthorityLevel.L2, at_time=NOW) == AuthorityLevel.L0


def test_activation_time_and_authority_ladder_are_enforced() -> None:
    envelope, approved = _approved(_profile(authority_level=AuthorityLevel.L1))
    manager = AuthorizationEnvelopeManager()
    manager.stage(envelope)
    manager.record_approval(approved, mac_key=KEY)
    manager.activate(activation_time=NOW + timedelta(minutes=5))
    assert manager.effective_authority(AuthorityLevel.L2, at_time=NOW) == AuthorityLevel.L0
    assert manager.effective_authority(
        AuthorityLevel.L2,
        at_time=NOW + timedelta(minutes=6),
    ) == AuthorityLevel.L1


def test_activation_requires_matching_explicit_approval() -> None:
    _, approved_first = _approved(_profile(profile_id="first"))
    second = compile_authorization_envelope(_profile(profile_id="second"))
    manager = AuthorizationEnvelopeManager()
    manager.stage(second)
    with pytest.raises(EnvelopeActivationError, match="does not match"):
        manager.record_approval(approved_first, mac_key=KEY)
    with pytest.raises(EnvelopeActivationError, match="approved"):
        manager.activate(activation_time=NOW)


def test_any_content_change_drops_system_to_l0() -> None:
    first, approved_first = _approved(_profile(profile_id="first"))
    manager = AuthorizationEnvelopeManager()
    manager.stage(first)
    manager.record_approval(approved_first, mac_key=KEY)
    manager.activate(activation_time=NOW)
    assert manager.effective_authority(AuthorityLevel.L2, at_time=NOW) == AuthorityLevel.L2

    changed = compile_authorization_envelope(
        _profile(profile_id="changed", pc_action=8.0e-5)
    )
    manager.stage(changed)
    assert manager.effective_authority(AuthorityLevel.L2, at_time=NOW) == AuthorityLevel.L0


def test_rollback_is_reversible_but_requires_reapproval() -> None:
    first, approved_first = _approved(_profile(profile_id="first"))
    manager = AuthorizationEnvelopeManager()
    manager.stage(first)
    manager.record_approval(approved_first, mac_key=KEY)
    manager.activate(activation_time=NOW)

    second, approved_second = _approved(
        _profile(profile_id="second", pc_action=8.0e-5)
    )
    manager.stage(second)
    manager.record_approval(approved_second, mac_key=KEY)
    manager.activate(activation_time=NOW)

    restored = manager.rollback()
    assert restored.version == first.version
    assert manager.active is None
    assert manager.pending is not None
    assert manager.effective_authority(AuthorityLevel.L2, at_time=NOW) == AuthorityLevel.L0

    reapproved = approve_authorization_envelope(
        restored,
        approving_identity="operator:xorb-ops-002",
        mac_key=KEY,
        key_id=KEY_ID,
        human_review_confirmed=True,
        approved_at=NOW + timedelta(minutes=1),
    )
    manager.record_approval(reapproved, mac_key=KEY)
    manager.activate(activation_time=NOW + timedelta(minutes=1))
    assert manager.effective_authority(
        AuthorityLevel.L2,
        at_time=NOW + timedelta(minutes=2),
    ) == AuthorityLevel.L2

def _policy(**overrides):
    values = {
        "operator_id": "DEFAULT_LEO",
        "pc_maneuver_threshold": 1.0e-4,
        "pc_monitor_threshold": 1.0e-5,
        "min_miss_distance_km": 1.0,
        "max_dv_per_event_ms": 2.0,
        "max_maneuvers_per_week": 3,
        "min_hours_before_tca": 4.0,
        "max_hours_before_tca": 72.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_existing_operator_policy_bridge_compiles_default_locked_profile() -> None:
    envelope = compile_from_operator_policy(
        _policy(),
        profile_id="default-leo-envelope",
        mission_class="first_flight_leo",
        risk_budget_id="default-leo-risk",
        maneuver_capacity=ManeuverCapacityInput(
            max_dv_per_burn_m_s=10.0,
            v_remaining_m_s=50.0,
            v_reserved_m_s=5.0,
            max_maneuvers_per_week=3,
        ),
        authority_level=AuthorityLevel.L2,
    )
    assert envelope.pc_action == pytest.approx(1.0e-4)
    assert envelope.pc_watch == pytest.approx(1.0e-5)
    assert envelope.authority_level == AuthorityLevel.L2


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"pc_maneuver_threshold": 1.1e-4}, "1e-4 ceiling"),
        ({"min_miss_distance_km": 0.9}, "1.0 km floor"),
        ({"max_dv_per_event_ms": 2.1}, "2.0 m/s L1 cap"),
        ({"max_maneuvers_per_week": 4}, "3/week"),
        ({"min_hours_before_tca": 3.9}, "4 hour floor"),
        ({"max_hours_before_tca": 72.1}, "72 hour horizon"),
    ],
)
def test_section6_safety_floor_validation_rejects_laxer_policy(
    override,
    match,
) -> None:
    with pytest.raises(EnvelopeCompileError, match=match):
        validate_section6_safety_floors(_policy(**override))

def test_scrum479_envelope_identical_across_modes():
    from common.operator_policy import OperatorPolicy

    envelopes = []
    for mode in ("aps", "flight_rule_1e4", "flight_rule_1e5"):
        policy = OperatorPolicy(
            operator_id="DEFAULT_LEO",
            policy_version="2.5.0",
            decision_mode=mode,
        )
        envelope = compile_from_operator_policy(
            policy,
            profile_id="scrum479-envelope",
            mission_class="first_flight_leo",
            risk_budget_id="scrum479-risk",
            maneuver_capacity=ManeuverCapacityInput(
                max_dv_per_burn_m_s=10.0,
                v_remaining_m_s=50.0,
                v_reserved_m_s=5.0,
                max_maneuvers_per_week=3,
            ),
            authority_level=AuthorityLevel.L2,
        )
        assert envelope.pc_action == 1.0e-4
        envelopes.append(envelope.to_dict())

    assert envelopes[0] == envelopes[1] == envelopes[2]
