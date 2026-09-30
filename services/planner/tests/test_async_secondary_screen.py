"""tests/test_async_secondary_screen.py

SCRUM-456: the on-demand secondary screen runs off the evaluate request path.

The screen takes 30 s to 2 min against LeoLabs and the dashboard's evaluate times
out at 60 s, so running it inline failed the very decisions it exists to check.
Evaluate now returns immediately with the secondary check PENDING and a job id,
the screen runs on a bounded pool, and a poll endpoint reports the result.

What these tests are really guarding is the safety direction, because moving a
gate off the request path is exactly where one gets weakened by accident:

  - PENDING is never CLEAR. Not on the check, not in the store, not on the poll
    response, not through the guard.
  - A pending screen does not authorize a maneuver: it fails the M1 to M2 staging
    guard.
  - A pending screen does not escalate to M4 on its own. It holds the decision
    provisional. Only a resolved NOT CLEAR escalates.
  - error, timeout, rate-limit and no-access still fail closed, exactly as the
    inline SCRUM-442 screen did.

The async is made deterministic by injecting `submit`, so the worker runs on the
calling thread. Nothing here starts a real thread or touches LeoLabs.

Run from repo root:
    python -m pytest services/planner/tests/test_async_secondary_screen.py -v
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from common import atlas_artifact as aa
from common import leolabs_runtime, safety_monitor
from common import leolabs_screening as ls
from common import secondary_screen_async as sa
from common.decision_state_machine import GuardInputs
from common.leolabs_cdm_parser import parse_leolabs_cdm
from common.leolabs_screening import (
    LeoLabsScreeningTimeout,
    LeoLabsScreeningUnavailable,
    ScreeningResult,
    evaluate_clear_contract,
)
from common.operator_policy import OperatorPolicy

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"
_EPOCH = "2026-09-24T12:00:00Z"
_R_POST = [6838.0, 0.0, 0.0]
_V_POST = [0.0, 0.3419, 7.5754]
_P_POST = [[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.16]]

#: Run the worker on the calling thread, so ordering is deterministic.
INLINE = staticmethod(lambda job: job())


@pytest.fixture(autouse=True)
def _clean_store():
    sa.reset_store()
    yield
    sa.reset_store()


@pytest.fixture
def policy() -> OperatorPolicy:
    return OperatorPolicy(operator_id="test", policy_version="v1", decision_mode="flight_rule_1e4")


@pytest.fixture
def cap():
    return MagicMock(radius_m=1.0)


@pytest.fixture
def cdm() -> dict:
    return json.loads(_FIXTURE.read_text())


@pytest.fixture
def real_cap():
    """A real SatelliteCapability: build_atlas_artifact reads it for real."""
    from common.constellation_geometry import _mean_motion_to_sma_km
    from common.satellite_capability import (
        ConstellationSlot, LifetimeProfile, SatelliteCapability,
    )
    return SatelliteCapability(
        sat_id="SAT-456", a_ref_km=_mean_motion_to_sma_km(15.3020),
        lifetime=LifetimeProfile(
            mass_kg=100.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
            mission_lifetime_days_remaining=365.0,
        ),
        slot=ConstellationSlot(in_constellation=False),
    )


@pytest.fixture
def burn_scoring(real_cap, policy):
    """A high-risk event that recommends a maneuver, so the gate opens."""
    import numpy as np
    from common.constellation_geometry import _mean_motion_to_sma_km
    from common.maneuver_scorer import score_maneuver_candidates
    a = _mean_motion_to_sma_km(15.3020)
    return score_maneuver_candidates(
        "CID-456-BURN", np.array([a, 0.0, 0.0]), np.array([0.0, 7.626, 0.0]),
        np.array([0.3, 0.0, 0.0]), np.eye(3) * 0.01,
        "2026-09-24T12:00:00Z", "2026-09-24T18:00:00Z", real_cap, policy,
    )


@pytest.fixture
def no_burn_scoring(real_cap, policy):
    """A distant event: no maneuver recommended, so no screen is needed."""
    import numpy as np
    from common.constellation_geometry import _mean_motion_to_sma_km
    from common.maneuver_scorer import score_maneuver_candidates
    a = _mean_motion_to_sma_km(15.3020)
    return score_maneuver_candidates(
        "CID-456-CLEAR", np.array([a, 0.0, 0.0]), np.array([0.0, 7.626, 0.0]),
        np.array([500.0, 0.0, 0.0]), np.eye(3) * 0.01,
        "2026-09-24T12:00:00Z", "2026-09-24T18:00:00Z", real_cap, policy,
    )


def _guard_inputs(**overrides) -> GuardInputs:
    """GuardInputs with the required scalars filled, overrides on top.

    Mirrors the helper in test_safety_monitor so the secondary fields under test
    are the only thing varying.
    """
    from datetime import datetime, timedelta, timezone

    from common.decision_state_machine import FlightMode

    t_now = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)
    base = dict(
        conjunction_id="conj-456",
        current_mode=FlightMode.M1_WATCH,
        t_now_utc=t_now,
        pc=1.0e-3,
        pc_source="cdm",
        pc_monitor_threshold=1.0e-8,
        pc_maneuver_threshold=1.0e-4,
        latest_burn_utc=t_now + timedelta(hours=3),
        t_ca_utc=t_now + timedelta(hours=6),
        covariance_source="real_cdm",
        data_age_s=3600.0,
    )
    base.update(overrides)
    return GuardInputs(**base)


def _parsed(cdm_dict, **overrides):
    out = copy.deepcopy(cdm_dict)
    out.update(overrides)
    return parse_leolabs_cdm(out, "L2669", validate_miss_distance=False,
                             strict_psd=False)


def _start(policy, cap, *, submit=lambda job: job(), **kw):
    """Start a screen through the artifact seam, worker inline by default."""
    params = dict(
        r_post_km=_R_POST, v_post_km_s=_V_POST, p_post_eci_km2=_P_POST,
        screening_epoch_utc=_EPOCH, policy=policy, cap=cap,
        primary_catalog_number="L2669", submit=submit,
    )
    params.update(kw)
    return aa._start_on_demand_secondary_check(**params)


def _stub_screening(monkeypatch, *, conjunctions=(), raises=None,
                    screening_id="scr-456"):
    if raises is not None:
        def _run(*a, **k):
            raise raises
    else:
        def _run(*a, **k):
            return ScreeningResult(
                screening_id=screening_id,
                conjunctions=list(conjunctions),
                cdm_count=len(conjunctions),
            )
    monkeypatch.setattr(ls, "run_screening", _run)


# ---------------------------------------------------------------------------
# 1. Evaluate returns PENDING, immediately, with a job id
# ---------------------------------------------------------------------------

class TestEvaluateReturnsPending:
    def test_the_check_comes_back_pending_with_a_job_id(self, policy, cap):
        """Nothing is submitted, so the screen cannot have resolved."""
        check = _start(policy, cap, submit=lambda job: None)
        assert check.screen_pending is True
        assert check.screen_job_id
        assert check.screen_deferred is False

    def test_pending_is_not_clear_and_not_performed(self, policy, cap):
        """The two booleans the MAF gate reads. Both False while pending."""
        check = _start(policy, cap, submit=lambda job: None)
        assert check.secondary_check_performed is False
        assert check.secondary_conjunction_clear is False

    def test_the_request_thread_does_no_leolabs_work(self, monkeypatch, policy, cap):
        """The point of the ticket: no create, no poll, no retrieve inline."""
        calls = {"n": 0}

        def _run(*a, **k):
            calls["n"] += 1
            return ScreeningResult("scr-456", [], 0)

        monkeypatch.setattr(ls, "run_screening", _run)
        _start(policy, cap, submit=lambda job: None)
        assert calls["n"] == 0

    def test_the_note_tells_the_operator_the_decision_is_provisional(
        self, policy, cap
    ):
        note = _start(policy, cap, submit=lambda job: None).operator_note
        assert "provisional" in note
        assert "NOT authorized" in note

    def test_the_store_holds_it_as_pending(self, policy, cap):
        check = _start(policy, cap, submit=lambda job: None)
        entry = sa.get_entry(check.screen_job_id)
        assert entry.status == sa.STATUS_PENDING
        assert entry.is_clear is False

    def test_missing_inputs_fail_closed_immediately_rather_than_pending(
        self, policy, cap
    ):
        """A screen that can never run must not leave a decision provisional."""
        check = _start(policy, cap, r_post_km=None, submit=lambda job: None)
        assert check.screen_pending is False
        assert check.secondary_check_performed is False
        assert check.secondary_conjunction_clear is False
        assert "missing required input" in check.operator_note

    def test_the_seed_source_is_recorded_for_the_poll(self, policy, cap):
        check = _start(policy, cap, p_post_source="cdm_primary_own",
                       submit=lambda job: None)
        assert sa.get_entry(check.screen_job_id).seed_source == "cdm_primary_own"


# ---------------------------------------------------------------------------
# 2. The store resolves to CLEAR or NOT CLEAR
# ---------------------------------------------------------------------------

class TestResolution:
    def test_an_empty_screen_resolves_clear(self, monkeypatch, policy, cap):
        _stub_screening(monkeypatch, conjunctions=())
        check = _start(policy, cap)
        entry = sa.get_entry(check.screen_job_id)
        assert entry.status == sa.STATUS_CLEAR
        assert entry.is_clear is True
        assert entry.screening_id == "scr-456"

    def test_a_breaching_screen_resolves_not_clear(
        self, monkeypatch, policy, cap, cdm
    ):
        breaching = _parsed(cdm, MISS_DISTANCE=500.0)
        _stub_screening(monkeypatch, conjunctions=[breaching])
        entry = sa.get_entry(_start(policy, cap).screen_job_id)
        assert entry.status == sa.STATUS_NOT_CLEAR
        assert entry.is_clear is False
        assert entry.verdict.clear is False

    def test_the_resolved_entry_carries_the_verdict_and_conjunctions(
        self, monkeypatch, policy, cap, cdm
    ):
        breaching = _parsed(cdm, MISS_DISTANCE=500.0)
        _stub_screening(monkeypatch, conjunctions=[breaching])
        entry = sa.get_entry(_start(policy, cap).screen_job_id)
        assert entry.verdict.breaches
        assert len(entry.result.conjunctions) == 1

    def test_the_verdict_matches_the_inline_contract_exactly(
        self, monkeypatch, policy, cap, cdm
    ):
        """The worker must not judge differently from the inline path."""
        breaching = _parsed(cdm, MISS_DISTANCE=500.0)
        _stub_screening(monkeypatch, conjunctions=[breaching])
        entry = sa.get_entry(_start(policy, cap).screen_job_id)
        expected = evaluate_clear_contract([breaching], policy)
        assert entry.verdict.clear == expected.clear
        assert entry.verdict.evaluated == expected.evaluated
        assert [b["limbs"] for b in entry.verdict.breaches] == \
               [b["limbs"] for b in expected.breaches]


# ---------------------------------------------------------------------------
# 3. Failure still fails closed
# ---------------------------------------------------------------------------

class TestFailsClosed:
    @pytest.mark.parametrize("failure", [
        LeoLabsScreeningTimeout("poll timed out"),
        LeoLabsScreeningUnavailable("no on-demand access"),
        LeoLabsScreeningUnavailable("rate limited"),
        RuntimeError("something unexpected"),
    ])
    def test_every_failure_mode_resolves_error_and_not_clear(
        self, monkeypatch, policy, cap, failure
    ):
        _stub_screening(monkeypatch, raises=failure)
        entry = sa.get_entry(_start(policy, cap).screen_job_id)
        assert entry.status == sa.STATUS_ERROR
        assert entry.is_clear is False

    def test_an_error_entry_is_terminal_not_pending(self, monkeypatch, policy, cap):
        _stub_screening(monkeypatch, raises=LeoLabsScreeningTimeout("timeout"))
        entry = sa.get_entry(_start(policy, cap).screen_job_id)
        assert entry.is_terminal is True
        assert entry.status != sa.STATUS_PENDING

    def test_the_error_reason_is_kept(self, monkeypatch, policy, cap):
        _stub_screening(monkeypatch, raises=LeoLabsScreeningTimeout("poll timed out"))
        entry = sa.get_entry(_start(policy, cap).screen_job_id)
        assert "timed out" in (entry.error or "")

    def test_a_worker_that_raises_outside_the_screen_is_still_recorded(
        self, monkeypatch, policy, cap
    ):
        """The pool must never swallow a job, leaving an entry pending forever."""
        def _boom(**kw):
            raise ValueError("ephemeris exploded")

        monkeypatch.setattr(aa, "_run_on_demand_secondary_check", _boom)
        entry = sa.get_entry(_start(policy, cap).screen_job_id)
        assert entry.status == sa.STATUS_ERROR
        assert "exploded" in (entry.error or "")

    def test_a_screen_that_could_not_start_is_not_clear_and_not_pending(
        self, monkeypatch, policy, cap
    ):
        def _boom(**kw):
            raise RuntimeError("pool is gone")

        monkeypatch.setattr(sa, "start_screen", _boom)
        check = _start(policy, cap)
        assert check.screen_pending is False
        assert check.secondary_conjunction_clear is False
        assert "could not be started" in check.operator_note


# ---------------------------------------------------------------------------
# 4. PENDING is never CLEAR, and never authorizes a maneuver
# ---------------------------------------------------------------------------

class TestPendingIsNeverClear:
    def test_is_clear_status_is_an_allow_list(self):
        """A new status, a typo or a None must not read as clear."""
        assert sa.is_clear_status(sa.STATUS_CLEAR) is True
        for status in (sa.STATUS_PENDING, sa.STATUS_NOT_CLEAR, sa.STATUS_ERROR,
                       None, "", "CLEAR", "cleared", "unknown"):
            assert sa.is_clear_status(status) is False

    def test_the_staging_guard_fails_on_a_pending_screen(self):
        """So a pending screen cannot authorize a maneuver."""
        result = safety_monitor.guard_secondary_clear(_guard_inputs(
            secondary_check_performed=False,
            secondary_conjunction_clear=False,
            secondary_screen_pending=True,
        ))
        assert result.passed is False
        assert "still running" in result.detail
        assert "provisional" in result.detail

    def test_the_guard_distinguishes_pending_from_a_broken_screen(self):
        """Both fail, and the record says which."""
        broken = safety_monitor.guard_secondary_clear(_guard_inputs(
            secondary_check_performed=False, secondary_conjunction_clear=False,
        ))
        assert broken.passed is False
        assert "was not performed" in broken.detail
        assert "still running" not in broken.detail

    def test_pending_does_not_escalate_to_m4_on_its_own(self):
        """A screen that is still running is not evidence of a conflict."""
        pending = _guard_inputs(
            secondary_check_performed=False,
            secondary_conjunction_clear=False,
            secondary_screen_pending=True,
        )
        escalations = safety_monitor._m1_escalation_guards(pending)
        assert not [g for g in escalations
                    if g.name == "secondary_conflict_clear"]

    def test_a_resolved_not_clear_does_escalate(self):
        """The contrast that makes the test above meaningful."""
        resolved = _guard_inputs(
            secondary_check_performed=True,
            secondary_conjunction_clear=False,
        )
        escalations = safety_monitor._m1_escalation_guards(resolved)
        assert [g for g in escalations if g.name == "secondary_conflict_clear"]

    def test_the_staging_guard_set_includes_the_secondary_guard_when_pending(self):
        """Unlike a deferred screen, pending must not be omitted from the AND.

        Omitting it would let a pending screen stage a maneuver, which is the
        exact failure this ticket must not introduce.
        """
        guards = safety_monitor.m1_to_m2_guards(_guard_inputs(
            secondary_check_performed=False,
            secondary_conjunction_clear=False,
            secondary_screen_pending=True,
        ))
        secondary = [g for g in guards if g.name == "secondary_conflict_clear"]
        assert len(secondary) == 1
        assert secondary[0].passed is False


# ---------------------------------------------------------------------------
# 5. The poll payload
# ---------------------------------------------------------------------------

class TestPollPayload:
    def test_a_pending_payload_reports_clear_false(self, policy, cap):
        check = _start(policy, cap, submit=lambda job: None)
        payload = sa.entry_to_dict(sa.get_entry(check.screen_job_id))
        assert payload["status"] == "pending"
        assert payload["clear"] is False
        assert payload["pending"] is True
        assert payload["secondary_conjunction_clear"] is False

    def test_a_clear_payload_reports_clear_true(self, monkeypatch, policy, cap):
        _stub_screening(monkeypatch, conjunctions=())
        payload = sa.entry_to_dict(
            sa.get_entry(_start(policy, cap).screen_job_id))
        assert payload["status"] == "clear"
        assert payload["clear"] is True
        assert payload["screening_id"] == "scr-456"

    def test_a_not_clear_payload_carries_the_breaches(
        self, monkeypatch, policy, cap, cdm
    ):
        _stub_screening(monkeypatch, conjunctions=[_parsed(cdm, MISS_DISTANCE=500.0)])
        payload = sa.entry_to_dict(
            sa.get_entry(_start(policy, cap).screen_job_id))
        assert payload["clear"] is False
        assert payload["verdict"]["breaches"]
        assert payload["flagged_objects"]

    def test_an_error_payload_reports_clear_false_with_the_reason(
        self, monkeypatch, policy, cap
    ):
        _stub_screening(monkeypatch, raises=LeoLabsScreeningTimeout("poll timed out"))
        payload = sa.entry_to_dict(
            sa.get_entry(_start(policy, cap).screen_job_id))
        assert payload["status"] == "error"
        assert payload["clear"] is False
        assert "timed out" in payload["error"]

    def test_the_conjunction_list_is_capped_and_says_so(
        self, monkeypatch, policy, cap, cdm
    ):
        """A live screen returns over a thousand events; the poll must stay fast."""
        many = []
        for i in range(sa._MAX_CONJUNCTIONS_IN_RESPONSE + 25):
            row = copy.deepcopy(cdm)
            row["SAT2_OBJECT_DESIGNATOR"] = f"L{80000 + i}"
            row["TCA_ISO"] = f"2026-08-30T10:{i % 60:02d}:48.130573Z"
            many.append(_parsed(row, MISS_DISTANCE=float(20_000 + i)))
        _stub_screening(monkeypatch, conjunctions=many)
        payload = sa.entry_to_dict(
            sa.get_entry(_start(policy, cap).screen_job_id))
        assert len(payload["conjunctions"]) == sa._MAX_CONJUNCTIONS_IN_RESPONSE
        assert payload["conjunctions_total"] == len(many)
        assert payload["conjunctions_truncated"] is True

    def test_the_capped_list_keeps_the_closest_events(
        self, monkeypatch, policy, cap, cdm
    ):
        """Truncation must not drop the events an operator cares about most."""
        many = []
        for i in range(sa._MAX_CONJUNCTIONS_IN_RESPONSE + 25):
            row = copy.deepcopy(cdm)
            row["SAT2_OBJECT_DESIGNATOR"] = f"L{80000 + i}"
            row["TCA_ISO"] = f"2026-08-30T10:{i % 60:02d}:48.130573Z"
            # descending miss, so the last built is the closest
            many.append(_parsed(row, MISS_DISTANCE=float(500_000 - i * 1000)))
        _stub_screening(monkeypatch, conjunctions=many)
        payload = sa.entry_to_dict(
            sa.get_entry(_start(policy, cap).screen_job_id))
        misses = [c["miss_distance_km"] for c in payload["conjunctions"]]
        assert misses == sorted(misses)
        assert misses[0] == pytest.approx(min(
            p.miss_distance_m / 1000.0 for p in many))


# ---------------------------------------------------------------------------
# 6. The store itself
# ---------------------------------------------------------------------------

class TestStore:
    def test_an_unknown_job_id_is_none_not_clear(self):
        assert sa.get_entry("nope") is None

    def test_expired_entries_are_evicted(self, monkeypatch, policy, cap):
        check = _start(policy, cap, submit=lambda job: None)
        entry = sa.get_entry(check.screen_job_id)
        entry.created_at -= sa._EVICT_AFTER_SECONDS + 1.0
        assert sa.get_entry(check.screen_job_id) is None

    def test_the_store_is_capped(self, policy, cap):
        for _ in range(sa._MAX_ENTRIES + 10):
            _start(policy, cap, submit=lambda job: None)
        assert sa.store_size() <= sa._MAX_ENTRIES

    def test_the_cap_evicts_the_oldest_first(self, policy, cap):
        first = _start(policy, cap, submit=lambda job: None).screen_job_id
        for _ in range(sa._MAX_ENTRIES + 5):
            _start(policy, cap, submit=lambda job: None)
        assert sa.get_entry(first) is None

    def test_reset_store_empties_it(self, policy, cap):
        _start(policy, cap, submit=lambda job: None)
        assert sa.store_size() >= 1
        sa.reset_store()
        assert sa.store_size() == 0

    def test_leolabs_reset_caches_clears_the_store(self, policy, cap):
        """So a test cannot inherit a resolved screen and read it as clear."""
        _start(policy, cap, submit=lambda job: None)
        assert sa.store_size() >= 1
        leolabs_runtime.reset_caches()
        assert sa.store_size() == 0

    def test_resolving_an_evicted_job_does_not_resurrect_it(self, policy, cap):
        """A late worker writing into a cleared store must not create an entry."""
        sa._finish("never-existed", status=sa.STATUS_CLEAR)
        assert sa.get_entry("never-existed") is None


# ---------------------------------------------------------------------------
# 7. The SCRUM-442 gate still holds
# ---------------------------------------------------------------------------

class TestGateUnchanged:
    def _artifact(self, scoring, policy, cap, monkeypatch, **kw):
        started = {"n": 0}
        real = sa.start_screen

        def _count(**kwargs):
            started["n"] += 1
            return real(**kwargs)

        monkeypatch.setattr(sa, "start_screen", _count)
        params = dict(
            r_post_km=_R_POST, v_post_km_s=_V_POST,
            secondary_screen_enabled=True, p_post_eci_km2=_P_POST,
            primary_catalog_number="L2669", screen_submit=lambda job: None,
        )
        params.update(kw)
        aa.build_atlas_artifact(
            scoring, cap, policy, "2026-09-24T18:00:00Z", **params)
        return started["n"]

    def test_a_no_burn_decision_starts_no_screen(
        self, monkeypatch, policy, no_burn_scoring, real_cap
    ):
        """The rate-limit gate: 3 creates per 2 minutes, not spent on no-burns."""
        assert no_burn_scoring.is_maneuver_recommended() is False
        assert self._artifact(no_burn_scoring, policy, real_cap, monkeypatch) == 0

    def test_a_recommended_maneuver_starts_exactly_one_screen(
        self, monkeypatch, policy, burn_scoring, real_cap
    ):
        assert burn_scoring.is_maneuver_recommended() is True
        assert self._artifact(burn_scoring, policy, real_cap, monkeypatch) == 1

    def test_the_screen_off_case_is_still_a_deliberate_deferral(
        self, monkeypatch, policy, burn_scoring, real_cap
    ):
        """Turning the screen off must not fail every evaluate closed."""
        assert self._artifact(burn_scoring, policy, real_cap, monkeypatch,
                              secondary_screen_enabled=False) == 0

    def test_the_artifact_carries_the_job_id_for_the_dashboard_to_poll(
        self, monkeypatch, policy, burn_scoring, real_cap
    ):
        artifact = aa.build_atlas_artifact(
            burn_scoring, real_cap, policy, "2026-09-24T18:00:00Z",
            r_post_km=_R_POST, v_post_km_s=_V_POST,
            secondary_screen_enabled=True, p_post_eci_km2=_P_POST,
            primary_catalog_number="L2669", screen_submit=lambda job: None,
        )
        check = artifact.post_maneuver.secondary_conflict
        assert check.screen_pending is True
        assert sa.get_entry(check.screen_job_id) is not None
        # and it survives serialisation, which is what SCRUM-457 will read
        assert artifact.to_dict()["post_maneuver"]["secondary_conflict"][
            "screen_job_id"] == check.screen_job_id


# ---------------------------------------------------------------------------
# 8. The inline path is still available and unchanged
# ---------------------------------------------------------------------------

def test_the_inline_path_still_resolves_in_one_call(
    monkeypatch, policy, real_cap, burn_scoring
):
    """secondary_screen_async=False keeps the SCRUM-442 behaviour verbatim."""
    _stub_screening(monkeypatch, conjunctions=())
    artifact = aa.build_atlas_artifact(
        burn_scoring, real_cap, policy, "2026-09-24T18:00:00Z",
        r_post_km=_R_POST, v_post_km_s=_V_POST,
        secondary_screen_enabled=True, p_post_eci_km2=_P_POST,
        primary_catalog_number="L2669", secondary_screen_async=False,
    )
    check = artifact.post_maneuver.secondary_conflict
    assert check.screen_pending is False
    assert check.secondary_check_performed is True
    assert check.secondary_conjunction_clear is True


# ---------------------------------------------------------------------------
# 9. The poll endpoint over HTTP
# ---------------------------------------------------------------------------

class TestPollEndpoint:
    """GET /v1/secondary-screen/{job_id}, the route SCRUM-457 will poll."""

    def _client(self):
        from fastapi.testclient import TestClient

        import server
        return TestClient(server.svc)

    def test_an_unknown_id_is_404_and_says_it_is_not_clear(self):
        """A caller that cannot find its screen has not been told it passed."""
        resp = self._client().get("/v1/secondary-screen/does-not-exist")
        assert resp.status_code == 404
        assert "NOT a clear screen" in json.dumps(resp.json())

    def test_pending_then_resolved_over_the_wire(self, monkeypatch, policy, cap):
        """The transition the dashboard polls for."""
        held = {}
        check = _start(policy, cap, submit=lambda job: held.setdefault("job", job))

        first = self._client().get(f"/v1/secondary-screen/{check.screen_job_id}")
        assert first.status_code == 200
        assert first.json()["status"] == "pending"
        assert first.json()["clear"] is False

        _stub_screening(monkeypatch, conjunctions=())
        held["job"]()                                 # the worker completes

        second = self._client().get(f"/v1/secondary-screen/{check.screen_job_id}")
        assert second.json()["status"] == "clear"
        assert second.json()["clear"] is True

    def test_a_not_clear_screen_resolves_not_clear_over_the_wire(
        self, monkeypatch, policy, cap, cdm
    ):
        _stub_screening(monkeypatch, conjunctions=[_parsed(cdm, MISS_DISTANCE=500.0)])
        check = _start(policy, cap)
        body = self._client().get(
            f"/v1/secondary-screen/{check.screen_job_id}").json()
        assert body["status"] == "not_clear"
        assert body["clear"] is False
        assert body["verdict"]["breaches"]

    def test_an_errored_screen_resolves_error_over_the_wire(
        self, monkeypatch, policy, cap
    ):
        _stub_screening(monkeypatch, raises=LeoLabsScreeningTimeout("poll timed out"))
        check = _start(policy, cap)
        body = self._client().get(
            f"/v1/secondary-screen/{check.screen_job_id}").json()
        assert body["status"] == "error"
        assert body["clear"] is False

    def test_the_payload_is_json_serialisable_whatever_the_status(
        self, monkeypatch, policy, cap, cdm
    ):
        """The route returns it directly, so a stray object would 500."""
        _stub_screening(monkeypatch, conjunctions=[_parsed(cdm, MISS_DISTANCE=500.0)])
        check = _start(policy, cap)
        resp = self._client().get(f"/v1/secondary-screen/{check.screen_job_id}")
        assert resp.status_code == 200
        json.dumps(resp.json())
