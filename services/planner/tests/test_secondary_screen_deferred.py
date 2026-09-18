"""
SCRUM-431 -- the secondary conflict screen, deferred behind a flag.

The SCRUM-381 screen ran on a Space-Track TLE catalog. A TLE carries no
covariance, so the screen could only ever produce an assumed-covariance answer,
and with Space-Track retired there is no catalog at all: it failed closed and
put VERIFICATION FAILED on every evaluate.

Disable-and-defer, not delete. The two tests the plan names are the two halves
of that decision:

  * with the screen disabled, the deferred state is produced, the secondary
    guard is absent from the staging set, and staging is not blocked by it;
  * with it enabled, the old fail-closed behaviour holds byte-for-byte.

Deferred is deliberately its own state rather than a reuse of not-performed.
Section 4.2 fails closed on not-performed, and it should: that means the screen
ought to have run and could not. Deferred means the check is out of scope for
this build. Collapsing the two would make a deliberate deferral
indistinguishable from a broken catalog.
"""
from __future__ import annotations

import pytest

from common.atlas_artifact import (
    SECONDARY_SCREEN_DEFERRED_NOTE,
    SecondaryConflictCheck,
    _build_verification_result,
    _deferred_secondary_check,
    build_atlas_artifact,
)
from common.constellation_geometry import _mean_motion_to_sma_km
from common.decision_state_machine import FlightMode
from common.maneuver_scorer import score_maneuver_candidates
from common.operator_policy import OperatorPolicy, ScoringWeights
from common.safety_monitor import evaluate_safety_monitor, m1_to_m2_guards
from common.satellite_capability import (
    ConstellationSlot,
    LifetimeProfile,
    SatelliteCapability,
)

from test_safety_monitor import staged_inputs

import numpy as np
import pytest

T_BURN = "2026-04-14T08:00:00Z"
T_CA = "2026-04-14T12:00:00Z"


@pytest.fixture
def policy():
    return OperatorPolicy(
        operator_id="TEST_OP", policy_version="2.5.0", max_dv_per_event_ms=2.0,
        mission_lifetime_days_total=1825.0,
        scoring_weights=ScoringWeights(
            lambda_dv=1.0, lambda_lifetime=0.8, lambda_slot_deviation=1.2
        ),
    )


@pytest.fixture
def cap_solo():
    return SatelliteCapability(
        sat_id="SAT-SOLO", a_ref_km=_mean_motion_to_sma_km(15.3020),
        lifetime=LifetimeProfile(
            mass_kg=100.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
            mission_lifetime_days_remaining=365.0,
        ),
        slot=ConstellationSlot(in_constellation=False),
    )


@pytest.fixture
def high_risk_scoring(cap_solo, policy):
    """A high-risk event that recommends a maneuver, same shape as
    test_atlas_artifact's fixture."""
    r_sat = np.array([_mean_motion_to_sma_km(15.3020), 0.0, 0.0])
    v_sat = np.array([0.0, 7.626, 0.0])
    r_rel = np.array([0.3, 0.0, 0.0])
    P = np.eye(3) * 0.01
    return score_maneuver_candidates(
        "CID-HR", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy
    )

DEFERRED_TEXT = "Secondary screen deferred, LeoLabs covariance integration pending."


# ---------------------------------------------------------------------------
# The deferred state itself
# ---------------------------------------------------------------------------


class TestDeferredState:
    def test_the_note_is_the_text_the_panel_must_show(self):
        assert SECONDARY_SCREEN_DEFERRED_NOTE == DEFERRED_TEXT

    def test_a_deferred_check_is_flagged_deferred(self):
        check = _deferred_secondary_check()

        assert check.screen_deferred is True
        assert check.operator_note == DEFERRED_TEXT

    def test_a_deferred_check_claims_neither_performed_nor_clear(self):
        """Factually accurate: no screen ran, so nothing was cleared. The
        deferred flag is what stops that being read as a failure."""
        check = _deferred_secondary_check()

        assert check.secondary_check_performed is False
        assert check.secondary_conjunction_clear is False
        assert check.flagged_objects == []

    def test_deferred_defaults_to_false_so_a_real_screen_is_never_mistaken_for_one(self):
        real = SecondaryConflictCheck(
            secondary_check_performed=True,
            secondary_conjunction_clear=True,
            flagged_objects=[],
            operator_note="screen clear",
        )

        assert real.screen_deferred is False


# ---------------------------------------------------------------------------
# Test 1: disabled -- deferred, absent from the staging set, not blocking
# ---------------------------------------------------------------------------


class TestDisabledScreenDoesNotBlockStaging:
    def test_the_artifact_carries_the_deferred_state(
        self, high_risk_scoring, cap_solo, policy
    ):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA,
            secondary_screen_enabled=False,
        )
        check = artifact.post_maneuver.secondary_conflict

        assert check.screen_deferred is True
        assert check.operator_note == DEFERRED_TEXT

    def test_the_secondary_guard_is_absent_from_the_staging_set(self):
        """Omitted, not passed. A guard that always passed would read, in the
        evidence record, as a screen that ran and found nothing."""
        names = [g.name for g in m1_to_m2_guards(
            staged_inputs(secondary_screen_deferred=True)
        )]

        assert "secondary_conflict_clear" not in names

    def test_the_guard_is_present_when_the_screen_is_not_deferred(self):
        names = [g.name for g in m1_to_m2_guards(
            staged_inputs(secondary_screen_deferred=False)
        )]

        assert "secondary_conflict_clear" in names

    def test_staging_is_not_blocked_by_the_deferred_screen(self):
        """The headline: an event that would otherwise stage still stages, with
        no catalog anywhere in the picture."""
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M1",
                secondary_screen_deferred=True,
                secondary_check_performed=False,
                secondary_conjunction_clear=False,
            )
        )

        assert decision.mode is FlightMode.M2_STAGED
        assert all(g.passed for g in decision.guards)

    def test_the_same_inputs_without_the_deferral_are_blocked(self):
        """The control: not-performed still fails closed, so the test above is
        measuring the deferral and not something else."""
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M1",
                secondary_screen_deferred=False,
                secondary_check_performed=False,
                secondary_conjunction_clear=False,
            )
        )

        assert decision.mode is not FlightMode.M2_STAGED
        assert "secondary_conflict_clear" in [
            g.name for g in decision.guards if not g.passed
        ]

    def test_a_deferred_screen_does_not_escalate_a_watch(self):
        """_m1_escalation_guards fires only on a performed-and-not-clear screen,
        so a deferred one is inert. If it were not, the default build would
        safehold every watch event."""
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M1",
                secondary_screen_deferred=True,
                iod_proceeds_to_validity=None,   # blocks staging, nothing else
                validity_routing=None,
                validity_evidence={},
            )
        )

        assert decision.mode is FlightMode.M1_WATCH

    def test_verification_does_not_fail_on_a_deferred_screen(
        self, high_risk_scoring, policy
    ):
        result = _build_verification_result(
            high_risk_scoring, policy, _deferred_secondary_check()
        )

        assert result.passed is True
        assert not any("Secondary" in r for r in result.failure_reasons)

    def test_the_deferred_note_is_surfaced_as_an_informational_line(
        self, high_risk_scoring, policy
    ):
        """Neutral text on the panel, appended after the verdict so it cannot
        read as a failure reason."""
        result = _build_verification_result(
            high_risk_scoring, policy, _deferred_secondary_check()
        )

        assert DEFERRED_TEXT in result.verification_note
        assert "VERIFICATION FAILED" not in result.verification_note


# ---------------------------------------------------------------------------
# Test 2: enabled -- the old fail-closed behaviour, unchanged
# ---------------------------------------------------------------------------


class TestEnabledScreenStillFailsClosed:
    def test_an_enabled_screen_with_no_catalog_is_not_performed(
        self, high_risk_scoring, cap_solo, policy
    ):
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA,
            known_objects=None, secondary_screen_enabled=True,
        )
        check = artifact.post_maneuver.secondary_conflict

        assert check.screen_deferred is False
        assert check.secondary_check_performed is False
        assert check.secondary_conjunction_clear is False

    def test_an_enabled_screen_that_could_not_run_fails_verification(
        self, high_risk_scoring, cap_solo, policy
    ):
        """Exactly today's behaviour: not-performed means safety could not be
        established, and that is a failure."""
        artifact = build_atlas_artifact(
            high_risk_scoring, cap_solo, policy, T_CA,
            known_objects=None, secondary_screen_enabled=True,
        )

        assert artifact.verification.passed is False
        assert any(
            "Secondary conjunction screen was not performed" in r
            for r in artifact.verification.failure_reasons
        )

    def test_an_enabled_screen_blocks_staging_when_not_clear(self):
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M1",
                secondary_screen_deferred=False,
                secondary_check_performed=True,
                secondary_conjunction_clear=False,
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_an_enabled_clear_screen_still_stages(self):
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M1",
                secondary_screen_deferred=False,
                secondary_check_performed=True,
                secondary_conjunction_clear=True,
            )
        )

        assert decision.mode is FlightMode.M2_STAGED

    def test_the_flag_defaults_to_off(self):
        """The whole point: the default build does not run the screen."""
        import importlib
        import server

        importlib.reload(server)
        assert server.SECONDARY_SCREEN_ENABLED is False

    def test_the_flag_turns_the_screen_on(self, monkeypatch):
        import importlib
        import server

        monkeypatch.setenv("SECONDARY_SCREEN_ENABLED", "true")
        importlib.reload(server)
        try:
            assert server.SECONDARY_SCREEN_ENABLED is True
        finally:
            monkeypatch.delenv("SECONDARY_SCREEN_ENABLED", raising=False)
            importlib.reload(server)
