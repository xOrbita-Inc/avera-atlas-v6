"""
SCRUM-380 -- an envelope or monitor-logic change drops authority to L0 until
ground re-validates.

MAF v2.0 section 6. SCRUM-379's envelope guard already checks the envelope is
cryptographically valid and unexpired, but it does not notice that the envelope
or the software has CHANGED since ground last signed off. This is that floor,
and it is the one SCRUM-384's fifth case (tamper / unsigned envelope) waits on.

L0 is advisory only (guard doc section 5), so clamping to it blocks every
autonomous execution path. The clamp is not a preference: nothing raises it
except a ground re-validate, which changes the baseline itself.

Anchored to the guard doc and MAF section 6, not to a restatement of the code.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from common.decision_state_machine import FlightMode
from common.safety_floors import (
    DEMOTED_AUTHORITY,
    AuthorityBaseline,
    authority_demotion_reason,
    baseline_enforced,
    baseline_matches,
    monitor_logic_hash,
    revalidated_baseline,
)
from common.safety_monitor import (
    evaluate_safety_monitor,
    guard_authority_baseline_validated,
)

from test_safety_monitor import T_NOW, staged_inputs

ENVELOPE = "env-v1-sha256-abc123456789"
LOGIC = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"

VALIDATED = AuthorityBaseline(
    envelope_version=ENVELOPE,
    monitor_logic_hash=LOGIC,
    validated_by="ground-ops",
    validated_at_utc="2026-04-01T12:00:00Z",
)


def baselined(**overrides):
    """The SCRUM-379 baseline event, with the floor armed and matching."""
    base = dict(
        ground_validated_baseline=VALIDATED,
        envelope_version=ENVELOPE,
        monitor_logic_hash=LOGIC,
    )
    base.update(overrides)
    return staged_inputs(**base)


# ---------------------------------------------------------------------------
# When the floor is armed at all
# ---------------------------------------------------------------------------


class TestFloorArming:
    def test_the_floor_is_change_detection_so_it_needs_a_baseline(self):
        """With nothing on record there is nothing to have changed FROM.

        The deliberate non-fail-closed call in SCRUM-380: demoting on a missing
        baseline would clamp every not-yet-adopted caller to L0, which is an
        outage operators route around rather than a safety win. Ground
        establishing a baseline is what arms the floor.
        """
        assert baseline_enforced(None) is False
        assert authority_demotion_reason(None, ENVELOPE, LOGIC) == ""

    def test_an_unbaselined_l2_still_reaches_its_granted_authority(self):
        inputs = staged_inputs(authority_level="L2")

        assert inputs.effective_authority() == "L2"
        assert inputs.authority_demotion() == ""

    def test_a_baseline_on_record_arms_the_floor(self):
        assert baseline_enforced(VALIDATED) is True


# ---------------------------------------------------------------------------
# The two changes that demote
# ---------------------------------------------------------------------------


class TestDemotionOnChange:
    def test_a_matching_configuration_keeps_its_authority(self):
        inputs = baselined(authority_level="L2")

        assert inputs.effective_authority() == "L2"
        assert guard_authority_baseline_validated(inputs).passed is True

    def test_an_envelope_version_change_demotes_to_l0(self):
        inputs = baselined(authority_level="L2", envelope_version="env-v1-sha256-999999999999")

        assert inputs.effective_authority() == DEMOTED_AUTHORITY
        assert inputs.authority() == "L2"  # what the envelope said, recorded

    def test_a_monitor_logic_change_demotes_to_l0(self):
        inputs = baselined(authority_level="L2", monitor_logic_hash="0" * 64)

        assert inputs.effective_authority() == DEMOTED_AUTHORITY

    def test_either_change_alone_is_enough(self):
        """An operator can swap the envelope without touching the software, and
        software can change without touching the envelope. Either on its own is
        a configuration ground has not signed off."""
        envelope_only = baselined(envelope_version="env-v1-sha256-aaaaaaaaaaaa")
        logic_only = baselined(monitor_logic_hash="1" * 64)

        assert envelope_only.authority_demotion() != ""
        assert logic_only.authority_demotion() != ""

    def test_the_demotion_reason_names_what_drifted(self):
        inputs = baselined(envelope_version="env-v1-sha256-999999999999")
        reason = inputs.authority_demotion()

        assert "envelope version changed" in reason
        assert ENVELOPE in reason

    def test_a_missing_envelope_version_cannot_prove_it_is_unchanged(self):
        inputs = baselined(envelope_version=None)

        assert inputs.effective_authority() == DEMOTED_AUTHORITY

    def test_an_unreadable_monitor_hash_cannot_prove_it_is_unchanged(self):
        inputs = baselined(monitor_logic_hash="unreadable")

        assert inputs.effective_authority() == DEMOTED_AUTHORITY

    def test_baseline_matches_is_strict_once_a_baseline_exists(self):
        assert baseline_matches(VALIDATED, ENVELOPE, LOGIC) is True
        assert baseline_matches(VALIDATED, ENVELOPE, "other") is False
        assert baseline_matches(VALIDATED, "other", LOGIC) is False
        assert baseline_matches(VALIDATED, "", LOGIC) is False


# ---------------------------------------------------------------------------
# What the demotion actually blocks
# ---------------------------------------------------------------------------


class TestDemotionBlocksAutonomy:
    def test_a_drifted_l2_cannot_stage(self):
        """Section 3's M1 to M2 row requires authority L1 or L2. Clamped to L0,
        the AND cannot be satisfied."""
        decision = evaluate_safety_monitor(
            baselined(current_mode="M1", authority_level="L2",
                      envelope_version="env-v1-sha256-999999999999")
        )

        assert decision.mode is not FlightMode.M2_STAGED
        failed = [g.name for g in decision.guards if not g.passed]
        assert "authority_baseline_validated" in failed

    def test_a_drifted_l2_cannot_auto_execute_even_with_the_window_closed(self):
        """The case SCRUM-384 waits on: a tampered or unsigned envelope must not
        be able to fly a burn on its own."""
        executable = dict(
            current_mode="M2",
            authority_level="L2",
            veto_window_close_utc=T_NOW - timedelta(seconds=1),
        )
        clean = evaluate_safety_monitor(baselined(**executable))
        assert clean.mode is FlightMode.M3_EXECUTING  # baseline matches: executes

        drifted = evaluate_safety_monitor(
            baselined(envelope_version="env-v1-sha256-999999999999", **executable)
        )
        assert drifted.mode is not FlightMode.M3_EXECUTING
        assert drifted.authorized_execution is None

    def test_a_drifted_l1_cannot_execute_on_an_operator_approval(self):
        drifted = evaluate_safety_monitor(
            baselined(
                current_mode="M2",
                authority_level="L1",
                dv_cap_m_s=2.0,
                monitor_logic_hash="0" * 64,
                approval_command_received=True,
                approval_accepted=True,
            )
        )

        assert drifted.mode is not FlightMode.M3_EXECUTING
        assert drifted.authorized_execution is None

    def test_the_decision_records_both_the_granted_and_effective_authority(self):
        """So an auditor can see the envelope said L2 and the machine acted at
        L0, instead of wondering why an L2 asset declined to act."""
        payload = evaluate_safety_monitor(
            baselined(authority_level="L2",
                      envelope_version="env-v1-sha256-999999999999")
        ).to_dict()

        assert payload["authority_granted"] == "L2"
        assert payload["authority_effective"] == "L0"


# ---------------------------------------------------------------------------
# Ground re-validate restores authority
# ---------------------------------------------------------------------------


class TestGroundRevalidate:
    def test_a_re_validate_re_baselines_and_restores_authority(self):
        new_envelope = "env-v1-sha256-999999999999"
        drifted = baselined(authority_level="L2", envelope_version=new_envelope)
        assert drifted.effective_authority() == DEMOTED_AUTHORITY

        restored = staged_inputs(
            authority_level="L2",
            envelope_version=new_envelope,
            monitor_logic_hash=LOGIC,
            ground_validated_baseline=revalidated_baseline(
                new_envelope, LOGIC, "ground-ops", "2026-04-01T14:00:00Z"
            ),
        )

        assert restored.effective_authority() == "L2"
        assert restored.authority_demotion() == ""

    def test_a_re_validated_l2_can_auto_execute_again(self):
        new_envelope = "env-v1-sha256-999999999999"
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M2",
                authority_level="L2",
                envelope_version=new_envelope,
                monitor_logic_hash=LOGIC,
                ground_validated_baseline=revalidated_baseline(
                    new_envelope, LOGIC, "ground-ops", "2026-04-01T14:00:00Z"
                ),
                veto_window_close_utc=T_NOW - timedelta(seconds=1),
            )
        )

        assert decision.mode is FlightMode.M3_EXECUTING

    def test_a_re_validate_must_carry_the_validating_identity(self):
        """The spacecraft cannot decide on its own that a new configuration is
        acceptable, so the re-baseline takes a ground identity."""
        with pytest.raises(ValueError):
            revalidated_baseline(ENVELOPE, LOGIC, "", "2026-04-01T14:00:00Z")

    def test_a_re_validate_does_not_bless_a_different_configuration(self):
        """Re-validating envelope A does not authorise envelope B."""
        baseline = revalidated_baseline(
            "env-A", LOGIC, "ground-ops", "2026-04-01T14:00:00Z"
        )
        inputs = staged_inputs(
            authority_level="L2",
            envelope_version="env-B",
            monitor_logic_hash=LOGIC,
            ground_validated_baseline=baseline,
        )

        assert inputs.effective_authority() == DEMOTED_AUTHORITY


# ---------------------------------------------------------------------------
# The monitor logic hash itself
# ---------------------------------------------------------------------------


class TestMonitorLogicHash:
    def test_the_hash_is_stable_across_calls(self):
        assert monitor_logic_hash() == monitor_logic_hash()

    def test_the_hash_covers_the_modules_that_decide_whether_a_burn_happens(self):
        """Content, not a declared version string: a version number is something
        an editor forgets to bump, and this floor exists for exactly the case
        where the logic changed without anyone saying so."""
        assert len(monitor_logic_hash()) == 64

    def test_an_unreadable_module_directory_hashes_to_a_non_matching_sentinel(
        self, tmp_path
    ):
        """A monitor that cannot read its own source runs at L0 rather than
        assuming it is the software ground signed off."""
        sentinel = monitor_logic_hash(module_dir=tmp_path)

        assert sentinel == "unreadable"
        assert baseline_matches(VALIDATED, ENVELOPE, sentinel) is False

    def test_changed_content_changes_the_hash(self, tmp_path):
        (tmp_path / "decision_state_machine.py").write_text("a = 1\n")
        (tmp_path / "safety_monitor.py").write_text("b = 2\n")
        (tmp_path / "safety_floors.py").write_text("c = 3\n")
        before = monitor_logic_hash(module_dir=tmp_path)

        (tmp_path / "safety_monitor.py").write_text("b = 3\n")
        after = monitor_logic_hash(module_dir=tmp_path)

        assert before != after
