"""
SCRUM-382 -- the GNC endpoints on the planner, and the additive evaluate wiring.

MAF v2.0 section 9. Request and response shapes are validated against
openapi/gnc_interface.yaml.

On where these paths live: the contract declares /v1/gnc/approve and
/v1/gnc/veto on the GNC server, called by ARBITER, and /v1/gnc/report as
GNC to APS. The planner serves all three as the APS-side receiver, because an
operator decision and a post-burn report both have to reach the state machine
that acts on them. The shapes are the contract's, unchanged.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import server
from common import gnc_client
from common.decision_state_machine import ManeuverCommand
from common.gnc_contract import validate_against_contract
from common.gnc_operator import OperatorCommandStore
from common.mode_persistence import InMemoryModeStore, PersistedMode

from test_decision_state_machine_evaluate import (
    _NOW,
    _base_body,
    _evaluate,
    _iso,
    _live_body,
)
from test_gnc_report import a_report

CONJ = "conj-2026-0302-001"


def _client():
    return TestClient(server.svc)


def _staged_store(conjunction_id=CONJ, mode="M2", with_command=True):
    store = InMemoryModeStore()
    store.write(
        PersistedMode(
            sat_id="SAT-382",
            mode=mode,
            conjunction_id=conjunction_id,
            command=(
                ManeuverCommand(
                    dv_eci_km_s=(0.0, 0.00035, 0.0),
                    dv_magnitude_m_s=0.35,
                    t_burn_utc=_NOW + timedelta(hours=2),
                )
                if with_command else None
            ),
            latest_burn_utc=_NOW + timedelta(hours=3),
        )
    )
    return store


# ---------------------------------------------------------------------------
# /v1/gnc/approve
# ---------------------------------------------------------------------------

APPROVAL = {
    "command_id": "gnc-cmd-0001",
    "conjunction_id": CONJ,
    "issued_by": "ops-jhavera",
    "issued_at_utc": "2026-04-01T13:47:00Z",
}


class TestApproveEndpoint:
    def test_the_request_under_test_is_a_real_approval_command(self):
        validate_against_contract("GNCApprovalCommand", APPROVAL)

    def test_an_approval_for_a_staged_burn_is_honoured(self):
        with patch.object(server, "_MODE_STORE", _staged_store()), \
             patch.object(server, "_OPERATOR_COMMANDS", OperatorCommandStore()):
            resp = _client().post("/v1/gnc/approve", json=APPROVAL)

        assert resp.status_code == 200
        ack = resp.json()
        validate_against_contract("GNCApprovalAck", ack)
        assert ack["approval_accepted"] is True
        assert ack["flight_mode_after"] == "M3"

    def test_an_approval_with_nothing_staged_is_refused_and_says_why(self):
        with patch.object(server, "_MODE_STORE", InMemoryModeStore()), \
             patch.object(server, "_OPERATOR_COMMANDS", OperatorCommandStore()):
            resp = _client().post("/v1/gnc/approve", json=APPROVAL)

        ack = resp.json()
        validate_against_contract("GNCApprovalAck", ack)
        assert ack["approval_accepted"] is False
        assert ack["flight_mode_after"] == "M4"
        assert "nothing is staged" in ack["rejection_note"]

    def test_an_approval_against_a_non_staged_mode_is_refused(self):
        with patch.object(server, "_MODE_STORE", _staged_store(mode="M1")), \
             patch.object(server, "_OPERATOR_COMMANDS", OperatorCommandStore()):
            ack = _client().post("/v1/gnc/approve", json=APPROVAL).json()

        assert ack["approval_accepted"] is False
        assert "not M2" in ack["rejection_note"]

    def test_an_approval_is_recorded_for_the_state_machine(self):
        """The point of serving this on the planner: the decision reaches the
        machine that acts on it."""
        commands = OperatorCommandStore()
        with patch.object(server, "_MODE_STORE", _staged_store()), \
             patch.object(server, "_OPERATOR_COMMANDS", commands):
            _client().post("/v1/gnc/approve", json=APPROVAL)

        assert commands.guard_inputs(CONJ) == {
            "approval_command_received": True,
            "approval_accepted": True,
        }

    def test_a_missing_required_field_is_a_422(self):
        for field in ("command_id", "conjunction_id", "issued_by", "issued_at_utc"):
            body = {k: v for k, v in APPROVAL.items() if k != field}
            resp = _client().post("/v1/gnc/approve", json=body)
            assert resp.status_code == 422, field

    def test_an_invalid_body_is_a_422_not_a_500(self):
        resp = _client().post(
            "/v1/gnc/approve", content=b"not json",
            headers={"Content-Type": "application/json"},
        )

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /v1/gnc/veto
# ---------------------------------------------------------------------------

VETO = {
    "command_id": "gnc-cmd-0001",
    "conjunction_id": CONJ,
    "issued_by": "ops-jhavera",
    "issued_at_utc": "2026-04-01T13:48:00Z",
    "reason": "operator wants a manual review",
}


class TestVetoEndpoint:
    def test_the_request_under_test_is_a_real_veto_command(self):
        validate_against_contract("GNCVetoCommand", VETO)

    def test_a_veto_inside_the_window_is_accepted_and_restages(self):
        """Section 3's M2 to M2 row."""
        with patch.object(server, "_MODE_STORE", _staged_store()), \
             patch.object(server, "_OPERATOR_COMMANDS", OperatorCommandStore()):
            ack = _client().post("/v1/gnc/veto", json=VETO).json()

        validate_against_contract("GNCVetoAck", ack)
        assert ack["veto_accepted"] is True
        assert ack["flight_mode_after"] == "M2"

    def test_a_late_veto_is_refused_and_the_burn_proceeds(self):
        """The contract: after veto_window_close_utc, veto_accepted is false and
        the burn proceeds, with the reason in late_veto_note."""
        late = {**VETO, "window_closed": True}
        with patch.object(server, "_MODE_STORE", _staged_store()), \
             patch.object(server, "_OPERATOR_COMMANDS", OperatorCommandStore()):
            ack = _client().post("/v1/gnc/veto", json=late).json()

        validate_against_contract("GNCVetoAck", ack)
        assert ack["veto_accepted"] is False
        assert "the burn proceeds" in ack["late_veto_note"]

    def test_a_veto_against_nothing_staged_is_refused_into_safehold(self):
        """Opposite meaning to a late veto: an event nobody is tracking should
        not be left in an unknown state."""
        with patch.object(server, "_MODE_STORE", InMemoryModeStore()), \
             patch.object(server, "_OPERATOR_COMMANDS", OperatorCommandStore()):
            ack = _client().post("/v1/gnc/veto", json=VETO).json()

        assert ack["veto_accepted"] is False
        assert ack["flight_mode_after"] == "M4"

    def test_a_veto_is_recorded_for_the_state_machine(self):
        commands = OperatorCommandStore()
        with patch.object(server, "_MODE_STORE", _staged_store()), \
             patch.object(server, "_OPERATOR_COMMANDS", commands):
            _client().post("/v1/gnc/veto", json=VETO)

        assert commands.guard_inputs(CONJ) == {
            "veto_command_received": True,
            "veto_accepted": True,
        }

    def test_the_latest_operator_decision_wins(self):
        """An operator who approves then vetoes has vetoed."""
        commands = OperatorCommandStore()
        with patch.object(server, "_MODE_STORE", _staged_store()), \
             patch.object(server, "_OPERATOR_COMMANDS", commands):
            _client().post("/v1/gnc/approve", json=APPROVAL)
            _client().post("/v1/gnc/veto", json=VETO)

        assert "veto_command_received" in commands.guard_inputs(CONJ)
        assert "approval_command_received" not in commands.guard_inputs(CONJ)

    def test_a_missing_required_field_is_a_422(self):
        for field in ("command_id", "conjunction_id", "issued_by", "issued_at_utc"):
            body = {k: v for k, v in VETO.items() if k != field}
            assert _client().post("/v1/gnc/veto", json=body).status_code == 422


# ---------------------------------------------------------------------------
# /v1/gnc/report
# ---------------------------------------------------------------------------


def _report_body(**overrides):
    body = a_report(**overrides)
    body["sat_id"] = "SAT-382"
    body["aps_risk_context"] = {
        "commanded_dv_m_s": 0.35,
        "commanded_dv_rtn_m_s": [0.0, 0.35, 0.0],
        "r_rel_km_at_tca": [0.4, 0.1, 0.0],
        "v_rel_km_s_at_tca": [0.0, -7.5, 0.1],
        "p_pre_km2": [2e-4, 0, 0, 0, 2e-4, 0, 0, 0, 2e-4],
        "hbr_m": 15.0,
        "pc_pre": 0.000312,
        "pc_monitor_threshold": 1.0e-5,
    }
    return body


def _post_report(body, ingest_ok=True):
    head = MagicMock()
    head.json.return_value = {"next_seq": 0, "content_hash": "0" * 64}
    posts = MagicMock(return_value=MagicMock(status_code=201 if ingest_ok else 500))
    with patch.object(server.http_requests, "get", return_value=head), \
         patch.object(server.http_requests, "post", posts):
        resp = _client().post("/v1/gnc/report", json=body)
    return resp, posts


class TestReportEndpoint:
    def test_a_report_round_trips_to_a_contract_valid_ack(self):
        body = _report_body()
        validate_against_contract(
            "GNCReport", {k: v for k, v in body.items()
                          if k not in ("sat_id", "aps_risk_context")}
        )
        resp, _ = _post_report(body)

        assert resp.status_code == 200
        validate_against_contract("GNCReportAck", resp.json())

    def test_a_nominal_burn_that_clears_the_event_needs_no_replan(self):
        resp, _ = _post_report(_report_body())

        assert resp.json()["replan_required"] is False

    def test_an_aborted_burn_requires_a_replan(self):
        resp, _ = _post_report(
            _report_body(
                execution_status="ABORTED",
                abort_detail={
                    "abort_reason": "WATCHDOG_EXPIRED",
                    "abort_time_utc": "2026-04-01T15:45:01Z",
                    "flight_mode_at_abort": "M3",
                },
            )
        )
        ack = resp.json()

        assert ack["replan_required"] is True
        assert "WATCHDOG_EXPIRED" in ack["replan_note"]

    def test_the_section_10_producers_are_recorded_on_the_evidence_trail(self):
        _, posts = _post_report(_report_body())

        appended = [
            call.kwargs["json"]["record"] for call in posts.call_args_list
            if "/evidence_record" in str(call[0])
        ]
        assert appended, "the post-burn outcome must be auditable"
        fields = appended[0]["fields"]
        for producer in ("actual_vs_predicted", "post_maneuver_od", "residual_risk"):
            assert fields[producer]["state"] == "present"
            assert fields[producer]["producer"] == "SCRUM-382"

    def test_an_audit_write_failure_still_acks_the_report(self):
        """A GNC layer left waiting on an ack because our audit store was down
        would be a worse failure than a gap we can see in the chain."""
        resp, _ = _post_report(_report_body(), ingest_ok=False)

        assert resp.status_code == 200
        validate_against_contract("GNCReportAck", resp.json())

    def test_an_unreachable_ingest_still_acks_the_report(self):
        head = MagicMock(side_effect=ConnectionError("ingest down"))
        with patch.object(server.http_requests, "get", head), \
             patch.object(server.http_requests, "post", MagicMock()):
            resp = _client().post("/v1/gnc/report", json=_report_body())

        assert resp.status_code == 200

    def test_a_missing_required_field_is_a_422(self):
        for field in ("command_id", "conjunction_id", "execution_status"):
            body = {k: v for k, v in _report_body().items() if k != field}
            assert _client().post("/v1/gnc/report", json=body).status_code == 422

    def test_a_report_with_no_risk_context_still_acks(self):
        """GNC is not obliged to know APS's pre-burn geometry. Without it no Pc
        can be established, which means a replan, not a crash."""
        body = _report_body()
        body.pop("aps_risk_context")
        resp, _ = _post_report(body)

        assert resp.status_code == 200
        assert resp.json()["replan_required"] is True


# ---------------------------------------------------------------------------
# /gnc-status
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    def test_the_default_is_disabled(self):
        resp = _client().get("/gnc-status")

        assert resp.status_code == 200
        assert resp.json()["mode"] == "disabled"

    def test_record_only_is_reported_when_enabled_without_an_endpoint(self, monkeypatch):
        monkeypatch.setattr(gnc_client, "GNC_ENABLED", True)
        monkeypatch.setattr(gnc_client, "GNC_SERVICE_URL", "")

        assert _client().get("/gnc-status").json()["mode"] == "record_only"


# ---------------------------------------------------------------------------
# The additive evaluate wiring
# ---------------------------------------------------------------------------


class TestEvaluateWiringIsAdditive:
    def _executable(self):
        body = _live_body()
        body["monitor"] = {
            "current_mode": "M2",
            "veto_window_close_utc": _iso(_NOW - timedelta(seconds=1)),
        }
        return body

    def test_with_gnc_off_the_response_carries_no_gnc_blocks(self):
        """The pre-382 evaluate contract, unchanged."""
        response, _ = _evaluate(self._executable())

        body = response.json()
        assert "authorized_execution" in body      # 379's payload still there
        assert "gnc_command" not in body
        assert "gnc_emission" not in body

    def test_with_gnc_in_record_only_the_command_is_attached_without_an_ack(
        self, monkeypatch
    ):
        monkeypatch.setattr(gnc_client, "GNC_ENABLED", True)
        monkeypatch.setattr(gnc_client, "GNC_SERVICE_URL", "")
        response, _ = _evaluate(self._executable())

        body = response.json()
        assert body["gnc_emission"]["mode"] == "record_only"
        assert body["gnc_emission"]["acknowledged"] is False
        assert body["gnc_emission"]["accepted"] is None

    def test_the_attached_command_is_contract_valid(self, monkeypatch):
        monkeypatch.setattr(gnc_client, "GNC_ENABLED", True)
        monkeypatch.setattr(gnc_client, "GNC_SERVICE_URL", "")
        response, _ = _evaluate(self._executable())

        validate_against_contract("GNCCommand", response.json()["gnc_command"])

    def test_the_command_carries_the_scorers_dv_and_the_authorised_identity(
        self, monkeypatch
    ):
        monkeypatch.setattr(gnc_client, "GNC_ENABLED", True)
        monkeypatch.setattr(gnc_client, "GNC_SERVICE_URL", "")
        response, _ = _evaluate(self._executable())
        body = response.json()

        command = body["gnc_command"]
        assert command["maneuver"]["dv_magnitude_m_s"] == (
            body["authorized_execution"]["dv_magnitude_m_s"]
        )
        assert command["conjunction_id"] == body["authorized_execution"]["conjunction_id"]
        assert command["authority_level"] == "L2"

    def test_an_emission_failure_does_not_break_the_evaluate(self, monkeypatch):
        monkeypatch.setattr(gnc_client, "GNC_ENABLED", True)
        monkeypatch.setattr(gnc_client, "GNC_SERVICE_URL", "")
        with patch.object(server, "build_gnc_command", side_effect=RuntimeError("boom")):
            response, _ = _evaluate(self._executable())

        assert response.status_code == 200
        assert "recommendation" in response.json()
        assert "authorized_execution" in response.json()
        assert "gnc_command" not in response.json()

    def test_an_evaluate_that_authorises_nothing_emits_nothing(self, monkeypatch):
        monkeypatch.setattr(gnc_client, "GNC_ENABLED", True)
        monkeypatch.setattr(gnc_client, "GNC_SERVICE_URL", "")
        response, _ = _evaluate(_base_body())

        assert "gnc_command" not in response.json()
