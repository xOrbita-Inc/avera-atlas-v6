"""
SCRUM-382 -- the feature-flagged GNC emission adapter.

MAF v2.0 section 9. Mirrors the LeoLabs and UDL flag pattern: a missing endpoint
interpolates to inert, not to an error, so CI and the demo need no GNC service.

The distinction under test is between record-only and live. record-only means
the command was built and logged and NOTHING acknowledged it, so there is no
ack. A synthesized ack would put "accepted: true" in an audit trail for a burn
no flight computer ever saw, which is the one thing this adapter must not do.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from common import gnc_client
from common.gnc_client import (
    MODE_DISABLED,
    MODE_LIVE,
    MODE_RECORD_ONLY,
    emission_mode,
    emit_gnc_command,
    get_status,
)
from common.gnc_contract import GNC_COMMAND_PATH, validate_against_contract

COMMAND = {
    "command_id": "gnc-cmd-0001",
    "conjunction_id": "conj-2026-0302-001",
    "authority_level": "L2",
    "envelope_version": "env-v1-sha256-abc123456789",
    "approving_identity": "ground-ops",
    "validity": {"status": "EARNED", "epsilon": 0.73, "epsilon_threshold": 0.20},
    "risk": {
        "pc_computed": 0.000312, "m2_pre": 0.064,
        "covariance_source": "real_cdm", "data_age_s": 3600.0,
    },
    "maneuver": {
        "direction": "prograde", "dv_rtn_m_s": [0.0, 0.35, 0.0],
        "dv_magnitude_m_s": 0.35, "burn_time_utc": "2026-04-01T15:45:00Z",
        "burn_duration_s": 2.4,
    },
    "safe_action": {
        "safe_action_id": "safe-001", "safe_action_type": "passive_hold",
    },
    "timing": {
        "command_issued_utc": "2026-04-01T13:45:00Z",
        "veto_window_open_utc": "2026-04-01T13:45:00Z",
        "veto_window_close_utc": "2026-04-01T13:50:00Z",
        "latest_burn_utc": "2026-04-01T17:45:00Z",
        "t_ca_utc": "2026-04-01T21:45:00Z",
    },
}

ACK = {
    "command_id": "gnc-cmd-0001",
    "conjunction_id": "conj-2026-0302-001",
    "accepted": True,
    "flight_mode_after": "M2",
    "ack_time_utc": "2026-04-01T13:45:01Z",
}


class TestTheFixtureIsContractValid:
    def test_the_command_under_test_is_a_real_gnc_command(self):
        validate_against_contract("GNCCommand", COMMAND)

    def test_the_ack_under_test_is_a_real_gnc_command_ack(self):
        validate_against_contract("GNCCommandAck", ACK)


class TestModeResolution:
    def test_the_flag_off_is_disabled(self):
        assert emission_mode(enabled=False, url="http://gnc:8070") == MODE_DISABLED

    def test_the_flag_on_without_an_endpoint_is_record_only(self):
        """The normal state today: no GNC service exists yet."""
        assert emission_mode(enabled=True, url="") == MODE_RECORD_ONLY

    def test_the_flag_on_with_an_endpoint_is_live(self):
        assert emission_mode(enabled=True, url="http://gnc:8070") == MODE_LIVE

    def test_the_default_is_off(self):
        """CI and the demo need no GNC service standing."""
        assert gnc_client.GNC_ENABLED is False
        assert gnc_client.GNC_SERVICE_URL == ""


class TestDisabled:
    def test_nothing_is_posted(self):
        with patch.object(gnc_client.http_requests, "post") as post:
            emission = emit_gnc_command(COMMAND, enabled=False, url="http://gnc:8070")

        post.assert_not_called()
        assert emission.mode == MODE_DISABLED
        assert emission.posted is False
        assert emission.acknowledged is False


class TestRecordOnly:
    def test_nothing_is_posted_without_an_endpoint(self):
        with patch.object(gnc_client.http_requests, "post") as post:
            emission = emit_gnc_command(COMMAND, enabled=True, url="")

        post.assert_not_called()
        assert emission.mode == MODE_RECORD_ONLY

    def test_no_ack_is_synthesized(self):
        """Nothing acknowledged the command, so there is no ack. A manufactured
        one would claim GNC accepted a burn it never received."""
        emission = emit_gnc_command(COMMAND, enabled=True, url="")

        assert emission.ack is None
        assert emission.acknowledged is False

    def test_accepted_is_none_not_false(self):
        """A command GNC never saw was not rejected."""
        emission = emit_gnc_command(COMMAND, enabled=True, url="")

        assert emission.accepted is None

    def test_the_command_identity_is_still_recorded(self):
        emission = emit_gnc_command(COMMAND, enabled=True, url="")

        assert emission.command_id == "gnc-cmd-0001"
        assert emission.conjunction_id == "conj-2026-0302-001"
        assert emission.emitted_at_utc

    def test_record_only_is_not_an_error(self):
        """An unconfigured endpoint is a fallback, not a failure."""
        emission = emit_gnc_command(COMMAND, enabled=True, url="")

        assert emission.error == ""


class TestLive:
    def _post(self, status=200, body=None, raises=None):
        if raises is not None:
            return MagicMock(side_effect=raises)
        response = MagicMock(status_code=status)
        response.json.return_value = body if body is not None else ACK
        response.text = "detail"
        return MagicMock(return_value=response)

    def test_the_command_is_posted_to_the_contract_path(self):
        post = self._post()
        with patch.object(gnc_client.http_requests, "post", post):
            emit_gnc_command(COMMAND, enabled=True, url="http://gnc:8070")

        url = post.call_args[0][0]
        assert url == "http://gnc:8070" + GNC_COMMAND_PATH
        assert post.call_args.kwargs["json"] == COMMAND

    def test_a_trailing_slash_on_the_endpoint_does_not_double_up(self):
        post = self._post()
        with patch.object(gnc_client.http_requests, "post", post):
            emit_gnc_command(COMMAND, enabled=True, url="http://gnc:8070/")

        assert post.call_args[0][0] == "http://gnc:8070" + GNC_COMMAND_PATH

    def test_the_real_ack_is_returned(self):
        with patch.object(gnc_client.http_requests, "post", self._post()):
            emission = emit_gnc_command(COMMAND, enabled=True, url="http://gnc:8070")

        assert emission.mode == MODE_LIVE
        assert emission.posted is True
        assert emission.acknowledged is True
        assert emission.accepted is True
        assert emission.ack["flight_mode_after"] == "M2"

    def test_a_rejection_ack_is_carried_through_not_reinterpreted(self):
        """The contract lets GNC reject and move to M4. That is GNC's call."""
        rejected = {**ACK, "accepted": False, "flight_mode_after": "M4",
                    "rejection_note": "slew infeasible"}
        with patch.object(gnc_client.http_requests, "post", self._post(body=rejected)):
            emission = emit_gnc_command(COMMAND, enabled=True, url="http://gnc:8070")

        validate_against_contract("GNCCommandAck", emission.ack)
        assert emission.accepted is False
        assert emission.ack["flight_mode_after"] == "M4"

    def test_an_unreachable_endpoint_returns_an_emission_not_an_exception(self):
        post = self._post(raises=ConnectionError("no route to host"))
        with patch.object(gnc_client.http_requests, "post", post):
            emission = emit_gnc_command(COMMAND, enabled=True, url="http://gnc:8070")

        assert emission.acknowledged is False
        assert "unreachable" in emission.error

    def test_a_non_200_returns_an_emission_not_an_exception(self):
        with patch.object(gnc_client.http_requests, "post", self._post(status=500)):
            emission = emit_gnc_command(COMMAND, enabled=True, url="http://gnc:8070")

        assert emission.posted is True
        assert emission.acknowledged is False
        assert "HTTP 500" in emission.error

    def test_a_non_json_ack_returns_an_emission_not_an_exception(self):
        response = MagicMock(status_code=200)
        response.json.side_effect = ValueError("not json")
        with patch.object(gnc_client.http_requests, "post",
                          MagicMock(return_value=response)):
            emission = emit_gnc_command(COMMAND, enabled=True, url="http://gnc:8070")

        assert emission.acknowledged is False
        assert "not JSON" in emission.error

    def test_the_adapter_never_raises(self):
        """The caller is inside /v1/evaluate. Every failure path returns."""
        for raises in (ConnectionError("x"), TimeoutError("x"), RuntimeError("x")):
            post = self._post(raises=raises)
            with patch.object(gnc_client.http_requests, "post", post):
                emission = emit_gnc_command(
                    COMMAND, enabled=True, url="http://gnc:8070"
                )
            assert emission.error


class TestStatusBadge:
    def test_disabled_reads_as_disabled(self, monkeypatch):
        monkeypatch.setattr(gnc_client, "GNC_ENABLED", False)
        monkeypatch.setattr(gnc_client, "GNC_SERVICE_URL", "")

        assert get_status()["mode"] == MODE_DISABLED
        assert get_status()["label"] == "GNC DISABLED"

    def test_record_only_says_commands_carry_no_acknowledgement(self, monkeypatch):
        monkeypatch.setattr(gnc_client, "GNC_ENABLED", True)
        monkeypatch.setattr(gnc_client, "GNC_SERVICE_URL", "")
        status = get_status()

        assert status["mode"] == MODE_RECORD_ONLY
        assert status["endpoint_configured"] is False
        assert "no acknowledgement" in status["note"]

    def test_live_names_the_endpoint(self, monkeypatch):
        monkeypatch.setattr(gnc_client, "GNC_ENABLED", True)
        monkeypatch.setattr(gnc_client, "GNC_SERVICE_URL", "http://gnc:8070")
        status = get_status()

        assert status["mode"] == MODE_LIVE
        assert status["endpoint"] == "http://gnc:8070"
