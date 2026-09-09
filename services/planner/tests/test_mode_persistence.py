"""
SCRUM-379 -- section 6: persistence, reboot recovery, comms gaps, and the
section 6.3 rows that are not in the section 3 table.

Anchored to docs/scrum-333/state_machine_guards.md section 6, row by row. The
600 s comms-gap threshold, the 150 s slew lead time and the 86400 s freshness
bound are written out literally rather than imported, so a retuned constant
fails a test instead of redefining the spec.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from common.decision_state_machine import FlightMode, ManeuverCommand
from common.mode_persistence import (
    FileModeStore,
    InMemoryModeStore,
    NullModeStore,
    PersistedMode,
    build_mode_store,
    recover_after_reboot,
)
from common.safety_monitor import comms_gap_status, evaluate_safety_monitor

from test_safety_monitor import T_NOW, a_command, staged_inputs

LATEST_BURN = T_NOW + timedelta(hours=1)


def persisted(mode="M2", latest_burn=LATEST_BURN, **overrides) -> PersistedMode:
    base = dict(
        sat_id="SAT-379",
        mode=mode,
        conjunction_id="conj-2026-0302-001",
        persisted_at_utc="2026-04-01T13:45:00Z",
        command=a_command(),
        latest_burn_utc=latest_burn,
        software_version="aps-3.0.0",
    )
    base.update(overrides)
    return PersistedMode(**base)


# ---------------------------------------------------------------------------
# Section 6.2: reboot recovery
# ---------------------------------------------------------------------------


class TestRebootRecovery:
    def test_nothing_persisted_escalates(self):
        """Section 6.2 step 1 is 'read persisted state'. If there is none, the
        mode is unknown, and section 1 escalates the unknown."""
        recovery = recover_after_reboot(None, T_NOW)

        assert recovery.mode is FlightMode.M4_SAFE_HOLD
        assert recovery.pre_reboot_mode is None

    @pytest.mark.parametrize("mode", ["M0", "M4"])
    def test_m0_and_m4_resume_normally(self, mode):
        recovery = recover_after_reboot(persisted(mode=mode), T_NOW)

        assert recovery.mode.value == mode
        assert recovery.pre_reboot_mode.value == mode

    def test_m1_resumes_and_re_evaluates(self):
        # Section 6.2 step 3: 're-evaluate conjunction with last known CDM'.
        recovery = recover_after_reboot(persisted(mode="M1"), T_NOW)

        assert recovery.mode is FlightMode.M1_WATCH
        assert "last known CDM" in recovery.reason

    @pytest.mark.parametrize("mode", ["M2", "M3"])
    def test_an_open_burn_window_re_enters_m2(self, mode):
        # Section 6.2 step 4: t_now < latest_burn_utc - (t_slew+t_settle+t_margin).
        recovery = recover_after_reboot(
            persisted(mode=mode, latest_burn=T_NOW + timedelta(seconds=151)), T_NOW
        )

        assert recovery.mode is FlightMode.M2_STAGED
        assert recovery.burn_window_open is True
        assert recovery.pre_reboot_mode.value == mode

    @pytest.mark.parametrize("mode", ["M2", "M3"])
    def test_a_closed_burn_window_escalates(self, mode):
        # 'If t_now >= latest_burn_utc: transition to M4.'
        recovery = recover_after_reboot(
            persisted(mode=mode, latest_burn=T_NOW), T_NOW
        )

        assert recovery.mode is FlightMode.M4_SAFE_HOLD
        assert recovery.burn_window_open is False

    def test_too_late_to_slew_escalates_rather_than_burning(self):
        """Inside the window but with less than the 150 s lead time left.
        Section 6.2 defines no branch for it, so it escalates."""
        recovery = recover_after_reboot(
            persisted(latest_burn=T_NOW + timedelta(seconds=149)), T_NOW
        )

        assert recovery.mode is FlightMode.M4_SAFE_HOLD
        assert recovery.burn_window_open is False

    def test_exactly_at_the_slew_boundary_escalates(self):
        # The condition is strict: t_now < latest_burn - 150 s.
        recovery = recover_after_reboot(
            persisted(latest_burn=T_NOW + timedelta(seconds=150)), T_NOW
        )

        assert recovery.mode is FlightMode.M4_SAFE_HOLD

    def test_a_staged_mode_with_no_persisted_window_escalates(self):
        recovery = recover_after_reboot(persisted(latest_burn=None), T_NOW)

        assert recovery.mode is FlightMode.M4_SAFE_HOLD
        assert recovery.burn_window_open is None

    def test_the_record_names_the_pre_reboot_mode(self):
        """Section 6.2 step 5: 'Log the reboot event with pre-reboot state'."""
        recovery = recover_after_reboot(persisted(mode="M3", latest_burn=T_NOW), T_NOW)

        assert recovery.to_dict()["pre_reboot_mode"] == "M3"
        assert recovery.to_dict()["event"] == "reboot_recovery"

    def test_a_naive_clock_is_rejected(self):
        with pytest.raises(ValueError):
            recover_after_reboot(persisted(), datetime(2026, 4, 1, 13, 45, 0))


# ---------------------------------------------------------------------------
# The stores
# ---------------------------------------------------------------------------


class TestStores:
    def test_the_default_store_persists_nothing(self):
        """A stateless service must not silently remember another event's mode."""
        store = NullModeStore()
        store.write(persisted())

        assert store.read("SAT-379") is None

    def test_build_mode_store_defaults_to_the_null_store(self, monkeypatch):
        monkeypatch.delenv("MODE_STATE_DIR", raising=False)

        assert isinstance(build_mode_store(), NullModeStore)

    def test_build_mode_store_honours_the_configured_directory(self, tmp_path):
        assert isinstance(build_mode_store(str(tmp_path)), FileModeStore)

    def test_in_memory_round_trip(self):
        store = InMemoryModeStore()
        store.write(persisted(mode="M2"))

        assert store.read("SAT-379").mode is FlightMode.M2_STAGED

    def test_file_store_round_trips_the_mode_and_the_command(self, tmp_path):
        store = FileModeStore(tmp_path)
        store.write(persisted(mode="M2"))

        restored = store.read("SAT-379")
        assert restored.mode is FlightMode.M2_STAGED
        assert restored.conjunction_id == "conj-2026-0302-001"
        assert restored.latest_burn_utc == LATEST_BURN
        assert isinstance(restored.command, ManeuverCommand)
        assert restored.command.dv_magnitude_m_s == pytest.approx(0.35)
        assert restored.command.t_burn_utc == a_command().t_burn_utc

    def test_an_unknown_satellite_reads_as_nothing_persisted(self, tmp_path):
        assert FileModeStore(tmp_path).read("SAT-NOBODY") is None

    def test_a_corrupt_file_reads_as_nothing_persisted(self, tmp_path):
        """Which recovery then escalates on, rather than assuming M0."""
        store = FileModeStore(tmp_path)
        store.write(persisted())
        (tmp_path / "mode_SAT-379.json").write_text("{ not json")

        assert store.read("SAT-379") is None
        assert recover_after_reboot(store.read("SAT-379"), T_NOW).mode is (
            FlightMode.M4_SAFE_HOLD
        )

    def test_a_write_replaces_the_previous_state(self, tmp_path):
        store = FileModeStore(tmp_path)
        store.write(persisted(mode="M2"))
        store.write(persisted(mode="M4"))

        assert store.read("SAT-379").mode is FlightMode.M4_SAFE_HOLD
        # One file per spacecraft, and no temporary files left behind.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["mode_SAT-379.json"]

    def test_an_awkward_satellite_id_does_not_escape_the_directory(self, tmp_path):
        store = FileModeStore(tmp_path)
        store.write(persisted(sat_id="../../etc/passwd"))

        assert [p.parent for p in tmp_path.iterdir()] == [tmp_path]
        assert store.read("../../etc/passwd").sat_id == "../../etc/passwd"


# ---------------------------------------------------------------------------
# Section 6.1: comms gaps
# ---------------------------------------------------------------------------


class TestCommsGap:
    def test_600_seconds_exactly_is_not_yet_a_gap(self):
        # Section 6.1: 'no uplink from ground for longer than 600s'.
        assert comms_gap_status("M0", comms_gap_s=600.0).in_gap is False
        assert comms_gap_status("M0", comms_gap_s=600.1).in_gap is True

    def test_an_l2_staged_burn_keeps_counting_down_through_a_gap(self):
        """Section 6.1, M2 (L2): 'Continue veto countdown with onboard clock.
        Auto-execute at veto_window_close_utc if no veto received. This is the
        intended L2 behaviour; autonomy was pre-granted for exactly this case.'"""
        status = comms_gap_status("M2", comms_gap_s=1200.0, authority_level="L2")

        assert status.may_auto_execute is True
        assert "onboard clock" in status.behaviour

    def test_an_l1_staged_burn_cannot_execute_through_a_gap(self):
        status = comms_gap_status("M2", comms_gap_s=1200.0, authority_level="L1")

        assert status.may_auto_execute is False
        assert "operator approval" in status.behaviour

    def test_no_gap_means_no_auto_execute_licence(self):
        status = comms_gap_status("M2", comms_gap_s=0.0, authority_level="L2")

        assert status.in_gap is False
        assert status.may_auto_execute is False

    def test_m4_holds_safehold_through_a_gap(self):
        status = comms_gap_status("M4", comms_gap_s=99999.0)

        assert "hold safehold" in status.behaviour
        assert status.may_auto_execute is False

    def test_stale_data_is_flagged_during_a_watch(self):
        # Section 6.1, M1: 'Flag staleness if data_age_s exceeds freshness bound.'
        assert comms_gap_status(
            "M1", comms_gap_s=1200.0, data_age_s=86400.1
        ).data_stale is True
        assert comms_gap_status(
            "M1", comms_gap_s=1200.0, data_age_s=86400.0
        ).data_stale is False

    def test_the_status_is_attached_to_every_decision(self):
        decision = evaluate_safety_monitor(staged_inputs(comms_gap_s=1200.0))

        assert decision.comms_gap.in_gap is True
        assert decision.to_dict()["comms_gap"]["comms_gap_threshold_s"] == 600.0

    def test_a_gap_alone_changes_no_mode(self):
        """Section 6.1 grants and withholds behaviours; it does not move modes.
        The guards already produce the right answer during a gap."""
        quiet = evaluate_safety_monitor(staged_inputs())
        in_gap = evaluate_safety_monitor(staged_inputs(comms_gap_s=99999.0))

        assert in_gap.mode is quiet.mode


# ---------------------------------------------------------------------------
# Section 6.3: the ingest row
# ---------------------------------------------------------------------------


class TestIngestUnavailable:
    def test_a_watch_continues_on_a_fresh_last_cdm(self):
        """Section 6.3: 'Continue monitoring with last CDM if data_age_s is
        within freshness bound.'"""
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M1", ingest_available=False, data_age_s=86400.0,
                iod_proceeds_to_validity=None,
            )
        )

        assert decision.mode is FlightMode.M1_WATCH

    def test_a_watch_on_a_stale_last_cdm_escalates(self):
        """'If stale, go to M4.'"""
        decision = evaluate_safety_monitor(
            staged_inputs(
                current_mode="M1", ingest_available=False, data_age_s=86400.1
            )
        )

        assert decision.mode is FlightMode.M4_SAFE_HOLD

    def test_a_stale_cdm_with_ingest_up_blocks_staging_without_safeholding(self):
        """With ingest available, section 4.1's freshness bound is the envelope
        guard's business: it blocks the burn, it does not safehold the bus."""
        decision = evaluate_safety_monitor(staged_inputs(data_age_s=86400.1))

        assert decision.mode is FlightMode.M1_WATCH
        failed = [g.name for g in decision.guards if not g.passed]
        assert failed == ["envelope_satisfied"]
