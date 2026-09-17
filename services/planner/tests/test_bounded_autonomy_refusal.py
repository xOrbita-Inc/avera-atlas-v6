"""
SCRUM-384 -- bounded-autonomy adversarial refusal set.

Per John (Slack): build end to end against the real 379 monitor and state
machine, not a decoupled unit-level version, and pin down per case whether
refusal means M4 escalation or blocked M1-to-M2 staging -- both are
never-execute, but they are different modes, and confirming each case lands
where the spec says is the whole point of this ticket.

Scope, after checking test_safety_monitor.py and test_safety_monitor_modes.py
directly rather than assuming coverage:

- Validity NOT_EARNED and secondary-conflict-not-clear (performed) are
  already fully covered by TestM1::test_each_m1_to_m4_clause_escalates in
  test_safety_monitor_modes.py, correctly asserting M4. Not duplicated here.
- Delta-v over cap and secondary-conflict-not-performed are exercised by
  test_safety_monitor.py's test_breaking_any_single_guard_prevents_staging,
  but that test only asserts `decision.mode is not M2_STAGED`, which does not
  distinguish M1 (hold) from M4 (escalate). The four tests below add the
  precise mode assertion for those two cases, plus the two genuinely
  untested stale-CDM sub-cases.

Envelope tampering (the fifth case in the original ticket) is explicitly
held per John: the full tamper-to-L0-until-ground-re-validates floor is
SCRUM-380's, not built yet.

Reuses staged_inputs() from test_safety_monitor.py rather than redefining
the baseline, so this file cannot silently drift from what "everything
satisfied" means there.
"""
from __future__ import annotations

from datetime import timedelta

from common.decision_state_machine import FlightMode
from common.safety_monitor import evaluate_safety_monitor

from test_safety_monitor import T_NOW, staged_inputs


class TestDeltaVOverCapBlocksStagingNotM4:
    """Section 3, M1 to M2's envelope-satisfied clause, via section 4.1's dv
    cap. Guard failure is already covered by
    test_breaking_any_single_guard_prevents_staging; this pins the mode.
    """

    def test_dv_over_cap_holds_m1_and_does_not_escalate(self):
        # Baseline command is 0.35 m/s (a_command()'s default); a 0.1 m/s cap
        # is the same override test_safety_monitor.py's own guard-level test
        # uses to fail envelope_satisfied specifically on the dv clause.
        decision = evaluate_safety_monitor(staged_inputs(dv_cap_m_s=0.1))

        assert decision.mode is FlightMode.M1_WATCH
        assert decision.mode is not FlightMode.M4_SAFE_HOLD
        assert "envelope_satisfied" in [
            g.name for g in decision.guards if not g.passed
        ]


class TestSecondaryConflictNotPerformedBlocksStagingNotM4:
    """Section 4.2: 'if the check was not performed, treat as NOT CLEAR.'

    Distinct from the performed-and-failed case (already covered, escalates
    to M4): an unperformed check blocks staging per m1_to_m2_guards, but is
    deliberately excluded from _m1_escalation_guards -- see that function's
    own docstring. This is the interpretive line worth pinning with an
    explicit test, same as the existing
    test_an_unevaluated_gate_blocks_staging_without_safeholding does for the
    validity gate.
    """

    def test_unperformed_check_holds_m1_and_does_not_escalate(self):
        decision = evaluate_safety_monitor(
            staged_inputs(secondary_check_performed=False)
        )

        assert decision.mode is FlightMode.M1_WATCH
        assert decision.mode is not FlightMode.M4_SAFE_HOLD
        assert "secondary_conflict_clear" in [
            g.name for g in decision.guards if not g.passed
        ]


class TestStaleCdmSplitsOnIngestAvailability:
    """SCRUM-384's original 'stale CDM beyond the 24 hr freshness bound' case
    is actually two scenarios with opposite outcomes, not one -- confirmed
    directly from guard_last_cdm_still_fresh's own docstring and body.

    With ingest up, an old CDM is the envelope guard's business (section
    4.1), which blocks staging. With ingest down, guard_last_cdm_still_fresh
    (section 6.3, the ingest row) is what fires, and that one escalates.
    """

    def test_stale_cdm_with_ingest_available_holds_m1(self):
        """Ingest is up; the last CDM is just old. guard_envelope_satisfied's
        data_age_s check fails; guard_last_cdm_still_fresh does not even run
        its freshness check, since it passes immediately whenever ingest is
        available."""
        decision = evaluate_safety_monitor(
            staged_inputs(
                ingest_available=True,
                data_age_s=90_000.0,  # beyond the 86400s / 24h bound
            )
        )

        assert decision.mode is FlightMode.M1_WATCH
        assert decision.mode is not FlightMode.M4_SAFE_HOLD
        failing = [g.name for g in decision.guards if not g.passed]
        assert "envelope_satisfied" in failing
        assert "last_cdm_still_fresh" not in failing

    def test_stale_cdm_with_ingest_unavailable_escalates_to_m4(self):
        """Ingest itself is down, and the last CDM predates that outage
        beyond the freshness bound. Section 6.3's ingest row: 'if stale, go
        to M4.'"""
        decision = evaluate_safety_monitor(
            staged_inputs(
                ingest_available=False,
                data_age_s=90_000.0,
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD
        assert "last_cdm_still_fresh" in [
            g.name for g in decision.guards if not g.passed
        ]
