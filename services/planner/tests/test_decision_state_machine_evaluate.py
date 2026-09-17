"""
SCRUM-379 -- /v1/evaluate runs the decision state machine and records its
transitions.

Two things are being checked. That the monitor is wired in additively, so a
request that predates 379 still evaluates and simply cannot stage. And that a
request carrying the IOD verdict, an observation arc and an L2 authorization
drives the same endpoint through the real seams to a staged or executing mode,
with a section 7 transition record appended to the evidence chain.
"""
from __future__ import annotations


from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

import server
from common.decision_state_machine import FlightMode
from common.mode_persistence import InMemoryModeStore, NullModeStore, PersistedMode

# The scenario is anchored to the real clock the adapter reads, because the
# section 4.3 slew inequality and the section 3 four-hour TCA floor are both
# measured from t_now. A fixed calendar date would drift into the past and
# quietly turn every scenario into an escalation.
_NOW = datetime.now(timezone.utc)

# The doc's own audit-log example Pc, comfortably above the 1e-4 action line.
_PC = 0.000312


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _base_body() -> dict:
    """A near-circular ~6778 km LEO primary, TCA six hours out."""
    return {
        "conjunction_id": "test-conj-379",
        "satellite": {
            "norad_id": 25544,
            "sat_id": "SAT-379",
            "r_sat_km": [6778.0, 0.0, 0.0],
            "v_sat_km_s": [0.0, 7.6686, 0.0],
            "t_burn_utc": _iso(_NOW + timedelta(hours=2)),
            "v_remaining_m_s": 50.0,
            "mass_kg": 100.0,
        },
        "policy": {
            "operator_id": "TEST",
            "policy_version": "2.5.0",
            "lambda_v": 0.01,
            "lambda_L": 0.001,
            "dv_mag_limit_m_s": 0.5,
            "a_ref_km": 6778.0,
        },
        "conjunction": {
            "primary_norad": 25544,
            "secondary_norad": 99001,
            "t_ca_utc": _iso(_NOW + timedelta(hours=6)),
            "r_rel_km": [0.1, 0.2, 0.3],
            "p_rel_km2": [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4],
            "pc_precomputed": _PC,
            "data_age_s": 3600.0,
        },
    }


def _observation_arc(n: int = 8) -> list:
    """A range-and-angles arc well enough spread to earn validity.

    Synthetic, and tuned only so far as to land on the EARNED side of the
    locked 0.20 floor -- the point is to drive the real seam, not to model a
    real sensor. The mirror case (a NOT_EARNED arc forcing M4) is
    _weak_arc below.
    """
    return [
        {
            "epoch_utc": _iso(_NOW - timedelta(minutes=5 * (n - i))),
            "r_observer_km": [6378.0, 100.0 * i, 50.0 * i],
            "ra_sigma_rad": 1e-4,
            "dec_sigma_rad": 1e-4,
            "range_sigma_km": 0.001,
        }
        for i in range(n)
    ]


def _weak_arc() -> list:
    """The same arc with no range measurement: angles only, badly observable."""
    return [
        {**entry, "range_sigma_km": None} for entry in _observation_arc()
    ]


def _live_body() -> dict:
    """The plain body plus everything a staging decision needs."""
    body = _base_body()
    body["iod"] = {"proceeds_to_validity": True, "confidence_verdict": "CONFIDENT"}
    body["authorization"] = {"authority_level": "L2"}
    body["validity"] = {
        "target_state": {
            "epoch_utc": _iso(_NOW),
            "r_km": [7000.0, 0.0, 0.0],
            "v_km_s": [0.0, 7.546, 0.0],
        },
        "observations": _observation_arc(),
        "r_rel_km_at_tca": [0.5, -0.3, 0.1],
        "v_rel_km_s_at_tca": [0.0, 0.01, 0.0],
    }
    body["conjunction"]["covariance_source"] = "real_cdm"
    body["monitor"] = {"current_mode": "M1"}
    return body


# A single catalog object in a far-away orbit, so the SCRUM-381 horizon screen
# runs for real and comes back CLEAR. Without a catalog the screen is 'not
# performed', which section 4.2 treats as NOT CLEAR -- correct, but it would
# stop every scenario here at the secondary guard and hide everything after it.
_DISTANT_CATALOG = [
    {
        "obj_id": "FAR-1",
        "r_km": [0.0, 0.0, 8000.0],
        "v_km_s": [0.0, 7.06, 0.0],
        "position_sigma_m": 100.0,
    }
]


def _ingest_get(cdm_available: bool = True, cdm_zero_filled: bool = False):
    """Stand in for the two ingest reads /v1/evaluate makes.

    The CDM read has to return a real RTN covariance, or the planner falls back
    to its elliptical surrogate and sets covariance_source to
    surrogate_elliptical -- which would then correctly block every L2
    auto-execute and hide what these tests are checking. Setting
    cdm_available False is how a test asks for exactly that surrogate path;
    covariance_source is the planner's own answer, not something a request body
    can assert.
    """

    def _get(url, *args, **kwargs):
        if "/cdm/" in url:
            if not cdm_available:
                return MagicMock(status_code=404, ok=False)
            response = MagicMock(status_code=200, ok=True)
            # SCRUM-380: a zero-filled CDM is what the record-quality floor is
            # for. It reaches the guard as a real_cdm covariance of all zeros,
            # which is how a hollow record actually arrives -- from the store,
            # not from the caller, whose covariance the adapter replaces.
            cov = (
                np.zeros((3, 3)) if cdm_zero_filled else np.diag([1e-4, 1e-4, 1e-4])
            )
            response.json.return_value = {
                "id": 4242,
                "covariance_source": "real_cdm",
                "covariance_combined_rtn": cov.tolist(),
            }
            return response
        response = MagicMock(status_code=200, ok=True)
        response.json.return_value = {"next_seq": 0, "content_hash": "0" * 64}
        return response

    return _get


def _evaluate(body: dict, catalog=None, cdm_available: bool = True,
              cdm_zero_filled: bool = False):
    posts = MagicMock(return_value=MagicMock(status_code=201))
    with patch.object(server, "UDL_ENABLED", False), \
         patch.object(server, "fetch_catalog_objects",
                      return_value=list(_DISTANT_CATALOG if catalog is None else catalog)), \
         patch.object(server.http_requests, "post", posts), \
         patch.object(server.http_requests, "get",
                      side_effect=_ingest_get(cdm_available, cdm_zero_filled)):
        response = TestClient(server.svc).post("/v1/evaluate", json=body)
    return response, posts


class TestAdditiveWiring:
    def test_a_pre_379_request_still_evaluates(self):
        response, _ = _evaluate(_base_body())

        assert response.status_code == 200
        body = response.json()
        assert "recommendation" in body
        assert "atlas_artifact" in body

    def test_a_pre_379_request_only_reaches_the_watch_mode(self):
        """Pc is above the monitor line, so the event is flagged -- and that is
        as far as a request carrying no IOD verdict and no arc can get."""
        response, _ = _evaluate(_base_body())

        decision = response.json()["decision_state_machine"]
        assert decision["from_mode"] == FlightMode.M0_NOMINAL.value
        assert decision["to_mode"] == FlightMode.M1_WATCH.value

    def test_a_pre_379_request_already_in_m1_cannot_stage(self):
        """No IOD verdict and no arc, so both gates are unknown. The monitor
        fails them closed rather than treating absence as consent."""
        body = _base_body()
        body["monitor"] = {"current_mode": "M1"}
        response, _ = _evaluate(body)

        decision = response.json()["decision_state_machine"]
        assert decision["to_mode"] != FlightMode.M2_STAGED.value
        failed = [g["guard"] for g in decision["guards"] if not g["passed"]]
        assert "iod_confident" in failed
        assert "validity_earned" in failed

    def test_a_pre_379_request_emits_no_authorized_execution(self):
        response, _ = _evaluate(_base_body())

        assert "authorized_execution" not in response.json()

    def test_a_monitor_failure_does_not_break_the_evaluate(self):
        with patch.object(
            server, "evaluate_decision_state_machine", side_effect=RuntimeError("boom")
        ):
            response, _ = _evaluate(_base_body())

        assert response.status_code == 200
        assert "recommendation" in response.json()
        assert "decision_state_machine" not in response.json()


class TestTheDecisionInTheResponse:
    def test_the_decision_names_both_modes_and_the_trigger(self):
        response, _ = _evaluate(_base_body())

        decision = response.json()["decision_state_machine"]
        assert decision["from_mode"] in {m.value for m in FlightMode}
        assert decision["to_mode"] in {m.value for m in FlightMode}
        assert decision["trigger"]
        assert decision["conjunction_id"] == "test-conj-379"

    def test_each_guard_reports_the_values_it_read(self):
        response, _ = _evaluate(_base_body())

        guards = response.json()["decision_state_machine"]["guards"]
        assert guards
        for guard in guards:
            assert set(guard) == {"guard", "passed", "detail", "values"}

    def test_an_unrecognised_current_mode_starts_from_safehold(self):
        body = _base_body()
        body["monitor"] = {"current_mode": "M9"}
        response, _ = _evaluate(body)

        decision = response.json()["decision_state_machine"]
        assert decision["from_mode"] == FlightMode.M4_SAFE_HOLD.value
        assert decision["to_mode"] == FlightMode.M4_SAFE_HOLD.value


class TestTheValiditySeamRunsForReal:
    def test_an_observable_arc_earns_validity_and_stages(self):
        """The full chain through the real seams: IOD CONFIDENT, an arc
        propagated to TCA and assessed by SCRUM-378, an L2 envelope compiled by
        SCRUM-375, and a covariance_source of real_cdm."""
        response, _ = _evaluate(_live_body())

        decision = response.json()["decision_state_machine"]
        guards = {g["guard"]: g for g in decision["guards"]}
        assert guards["validity_earned"]["values"]["validity_status"] == "EARNED"
        assert guards["validity_earned"]["values"]["epsilon_threshold"] == 0.20
        assert decision["to_mode"] == FlightMode.M2_STAGED.value

    def test_a_weakly_observable_arc_is_not_earned_and_escalates(self):
        """The mirror. Same event, angles-only arc, so the conjunction-plane
        geometry is not observable and section 3 sends it to M4."""
        body = _live_body()
        body["validity"]["observations"] = _weak_arc()
        response, _ = _evaluate(body)

        decision = response.json()["decision_state_machine"]
        guards = {g["guard"]: g for g in decision["guards"]}
        assert guards["validity_earned"]["values"]["validity_status"] == "NOT_EARNED"
        assert decision["to_mode"] == FlightMode.M4_SAFE_HOLD.value
        assert "authorized_execution" not in response.json()

    def test_a_routing_decision_in_the_request_body_is_ignored(self):
        """A caller must not be able to assert its own validity verdict."""
        body = _live_body()
        body["validity"]["routing"] = "AUTONOMOUS"
        del body["validity"]["observations"]
        response, _ = _evaluate(body)

        guards = {
            g["guard"]: g for g in response.json()["decision_state_machine"]["guards"]
        }
        assert guards["validity_earned"]["passed"] is False
        assert guards["validity_earned"]["values"]["validity_routing"] is None

    def test_an_unavailable_validity_service_is_not_earned(self):
        body = _live_body()
        body["validity"]["service_available"] = False
        response, _ = _evaluate(body)

        decision = response.json()["decision_state_machine"]
        assert decision["to_mode"] == FlightMode.M4_SAFE_HOLD.value


class TestTheExecuteEmission:
    def test_a_staged_l2_event_emits_the_382_payload_when_the_window_closes(self):
        body = _live_body()
        body["monitor"] = {
            "current_mode": "M2",
            "veto_window_close_utc": _iso(_NOW - timedelta(seconds=1)),
        }
        response, _ = _evaluate(body)

        payload = response.json()["authorized_execution"]
        assert payload["mode"] == "M3"
        assert payload["authority_level"] == "L2"
        assert payload["approval_basis"] == "l2_veto_window_expired"
        assert payload["covariance_source"] == "real_cdm"
        assert payload["validity_status"] == "EARNED"
        assert payload["dv_eci_km_s"] == response.json()["recommendation"]["dv_eci_km_s"]
        assert payload["dv_magnitude_m_s"] == (
            response.json()["recommendation"]["dv_magnitude_m_s"]
        )

    def test_a_surrogate_covariance_blocks_the_l2_auto_execute(self):
        """No CDM available, so the planner falls back to its elliptical
        surrogate. Section 4.1 will not auto-execute on that without an
        explicit operator acknowledgement."""
        body = _live_body()
        body["monitor"] = {
            "current_mode": "M2",
            "veto_window_close_utc": _iso(_NOW - timedelta(seconds=1)),
        }
        response, _ = _evaluate(body, cdm_available=False)

        assert "authorized_execution" not in response.json()
        assert (
            response.json()["decision_state_machine"]["to_mode"]
            == FlightMode.M4_SAFE_HOLD.value
        )


class TestTheEnvelopeIsCompiledFromTheRequest:
    def test_an_l2_request_compiles_an_envelope_and_a_dv_cap(self):
        response, _ = _evaluate(_live_body())

        guards = {
            g["guard"]: g for g in response.json()["decision_state_machine"]["guards"]
        }
        envelope = guards["envelope_satisfied"]
        assert envelope["values"]["envelope_version"].startswith("env-v1-sha256-")
        # SCRUM-375 locks the L2 cap at 0.5 m/s unless explicitly raised.
        assert envelope["values"]["dv_cap_m_s"] <= 0.5

    def test_the_default_authority_is_l0_and_cannot_stage(self):
        body = _base_body()
        body["monitor"] = {"current_mode": "M1"}
        response, _ = _evaluate(body)

        guards = {
            g["guard"]: g for g in response.json()["decision_state_machine"]["guards"]
        }
        assert guards["authority_l1_or_l2"]["passed"] is False
        assert guards["authority_l1_or_l2"]["values"]["authority_level"] == "L0"


class TestTransitionRecords:
    def test_a_transition_appends_an_evidence_record(self):
        body = _live_body()
        body["monitor"] = {"current_mode": "M4", "operator_clearance_received": True}
        response, posts = _evaluate(body)

        assert response.status_code == 200
        appended = [
            call for call in posts.call_args_list
            if "/evidence_record" in str(call[0])
            and call.kwargs["json"]["record"]["record_type"] == "transition"
        ]
        assert appended, "an M4 to M0 clearance must be recorded"
        fields = appended[0].kwargs["json"]["record"]["fields"]
        assert fields["from_mode"]["value"] == "M4"
        assert fields["to_mode"]["value"] == "M0"
        assert fields["trigger"]["value"] == "operator_clearance_via_arbiter"

    def test_a_hold_appends_no_transition_record(self):
        """A quiet M0 poll must not bury the transitions that matter."""
        body = _base_body()
        body["conjunction"]["pc_precomputed"] = 1e-12
        _, posts = _evaluate(body)

        transitions = [
            call for call in posts.call_args_list
            if "/evidence_record" in str(call[0])
            and call.kwargs["json"]["record"]["record_type"] == "transition"
        ]
        assert transitions == []

    def test_the_transition_record_carries_the_monitor_results(self):
        body = _live_body()
        body["monitor"] = {"current_mode": "M4", "operator_clearance_received": True}
        _, posts = _evaluate(body)

        record = [
            call.kwargs["json"]["record"] for call in posts.call_args_list
            if "/evidence_record" in str(call[0])
            and call.kwargs["json"]["record"]["record_type"] == "transition"
        ][0]
        monitor_results = record["fields"]["monitor_results"]
        assert monitor_results["state"] == "present"
        assert monitor_results["producer"] == "SCRUM-379"

    def test_an_audit_write_failure_does_not_break_the_evaluate(self):
        body = _live_body()
        body["monitor"] = {"current_mode": "M4", "operator_clearance_received": True}
        with patch.object(server, "UDL_ENABLED", False), \
             patch.object(server.http_requests, "get", side_effect=RuntimeError("ingest down")), \
             patch.object(server.http_requests, "post", MagicMock(return_value=MagicMock(status_code=201))):
            response = TestClient(server.svc).post("/v1/evaluate", json=body)

        assert response.status_code == 200
        assert "decision_state_machine" in response.json()


class TestModePersistenceThroughTheEndpoint:
    """Section 6.2, through /v1/evaluate rather than in isolation."""

    def test_nothing_is_persisted_by_default(self):
        """The service is stateless unless MODE_STATE_DIR is configured."""
        assert isinstance(server._MODE_STORE, NullModeStore)

    def test_the_decided_mode_is_persisted(self):
        store = InMemoryModeStore()
        with patch.object(server, "_MODE_STORE", store):
            response, _ = _evaluate(_live_body())

        persisted = store.read("SAT-379")
        assert persisted is not None
        assert persisted.mode.value == (
            response.json()["decision_state_machine"]["to_mode"]
        )
        assert persisted.conjunction_id == "test-conj-379"

    def test_a_persisted_mode_is_recovered_on_the_next_call(self):
        store = InMemoryModeStore()
        staged = _live_body()
        with patch.object(server, "_MODE_STORE", store):
            _evaluate(staged)  # stages, and persists M2

            # The next call carries no monitor block at all, so the persisted
            # mode is what it starts from -- via the section 6.2 ladder, since
            # this process did not make that transition itself.
            follow_up = _live_body()
            del follow_up["monitor"]
            response, _ = _evaluate(follow_up)

        decision = response.json()["decision_state_machine"]
        assert decision["reboot_recovery"]["pre_reboot_mode"] == "M2"
        assert decision["from_mode"] in ("M2", "M4")

    def test_an_explicit_mode_in_the_request_beats_the_persisted_one(self):
        store = InMemoryModeStore()
        store.write(
            PersistedMode(sat_id="SAT-379", mode="M2", latest_burn_utc=_NOW + timedelta(hours=1))
        )
        body = _live_body()
        body["monitor"] = {"current_mode": "M0"}
        with patch.object(server, "_MODE_STORE", store):
            response, _ = _evaluate(body)

        decision = response.json()["decision_state_machine"]
        assert decision["from_mode"] == "M0"
        assert decision["reboot_recovery"] is None

    def test_a_reboot_with_a_closed_burn_window_recovers_into_safehold(self):
        store = InMemoryModeStore()
        store.write(
            PersistedMode(
                sat_id="SAT-379", mode="M2", latest_burn_utc=_NOW - timedelta(minutes=1)
            )
        )
        body = _live_body()
        del body["monitor"]
        with patch.object(server, "_MODE_STORE", store):
            response, _ = _evaluate(body)

        decision = response.json()["decision_state_machine"]
        assert decision["from_mode"] == FlightMode.M4_SAFE_HOLD.value
        assert decision["to_mode"] == FlightMode.M4_SAFE_HOLD.value
        assert decision["reboot_recovery"]["burn_window_open"] is False

    def test_every_decision_reports_the_comms_gap_status(self):
        response, _ = _evaluate(_base_body())

        gap = response.json()["decision_state_machine"]["comms_gap"]
        assert gap["in_comms_gap"] is False
        assert gap["comms_gap_threshold_s"] == 600.0


class TestScrum380FloorsThroughTheEndpoint:
    """SCRUM-380 -- the abort and the L0 demotion reach /v1/evaluate, and a
    failure in either still cannot break the core evaluate."""

    def _abort_body(self):
        body = _live_body()
        body["monitor"] = {
            "current_mode": "M2",
            "veto_window_close_utc": _iso(_NOW - timedelta(seconds=1)),
            "ground_abort": {
                "abort_reason": "pass aborted by ground",
                "operator_id": "ops-jhavera",
                "command_id": "abort-42",
            },
        }
        return body

    def test_a_ground_abort_forces_m4_and_blocks_the_execute(self):
        response, _ = _evaluate(self._abort_body())

        decision = response.json()["decision_state_machine"]
        assert decision["to_mode"] == FlightMode.M4_SAFE_HOLD.value
        assert decision["aborted"] is True
        assert decision["abort"]["operator_id"] == "ops-jhavera"
        assert "authorized_execution" not in response.json()

    def test_the_abort_is_recorded_on_the_evidence_chain(self):
        _, posts = _evaluate(self._abort_body())

        transitions = [
            call.kwargs["json"]["record"] for call in posts.call_args_list
            if "/evidence_record" in str(call[0])
            and call.kwargs["json"]["record"]["record_type"] == "transition"
        ]
        assert transitions, "an abort is a transition and must be recorded"
        provenance = transitions[0]["fields"]["inputs_and_provenance"]["value"]
        assert provenance["ground_abort"]["abort_reason"] == "pass aborted by ground"
        assert provenance["ground_abort"]["operator_id"] == "ops-jhavera"

    def test_a_malformed_abort_block_does_not_break_the_evaluate(self):
        """Additive discipline: the monitor may refuse the request, the core
        evaluate still returns."""
        body = _live_body()
        body["monitor"] = {"ground_abort": {"operator_id": "ops-jhavera"}}
        response, _ = _evaluate(body)

        assert response.status_code == 200
        assert "recommendation" in response.json()
        assert "decision_state_machine" not in response.json()

    def test_a_drifted_baseline_clamps_authority_to_l0_end_to_end(self):
        body = _live_body()
        body["monitor"] = {
            "current_mode": "M2",
            "veto_window_close_utc": _iso(_NOW - timedelta(seconds=1)),
            "ground_validated_baseline": {
                "envelope_version": "env-v1-sha256-somethingelse",
                "monitor_logic_hash": "0" * 64,
                "validated_by": "ground-ops",
            },
        }
        response, _ = _evaluate(body)

        decision = response.json()["decision_state_machine"]
        assert decision["authority_effective"] == "L0"
        assert decision["to_mode"] != FlightMode.M3_EXECUTING.value
        assert "authorized_execution" not in response.json()

    def test_no_baseline_leaves_the_live_path_working_as_before(self):
        """The floor is change detection; an unbaselined caller is unaffected."""
        body = _live_body()
        body["monitor"] = {
            "current_mode": "M2",
            "veto_window_close_utc": _iso(_NOW - timedelta(seconds=1)),
        }
        response, _ = _evaluate(body)

        assert response.json()["decision_state_machine"]["authority_effective"] == "L2"
        assert "authorized_execution" in response.json()

    def test_a_zero_filled_cdm_is_refused_rather_than_scored(self):
        """A hollow record arrives from the store, not the caller: the planner
        replaces whatever covariance a request supplies."""
        body = _base_body()
        body["monitor"] = {"current_mode": "M1"}
        response, _ = _evaluate(body, cdm_zero_filled=True)

        assert response.status_code == 200
        guards = {
            g["guard"]: g for g in response.json()["decision_state_machine"]["guards"]
        }
        assert guards["cdm_record_usable"]["passed"] is False
