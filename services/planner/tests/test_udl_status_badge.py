"""
tests/test_udl_status_badge.py

SCRUM-348: badge truth for the /udl-status endpoint.

The data-source badge must never claim "UDL LIVE" while UDL conjunctions are
not consumed by the planner. UDL being enabled and authenticated only means it
is reachable and contributing catalog elsets to secondary screening, so the
enabled+valid state reports "UDL CONNECTED (CATALOG ONLY)", not "UDL LIVE".
"""
from __future__ import annotations

import os
from unittest.mock import patch

from fastapi.testclient import TestClient

import server


def _udl_status(enabled: bool, creds: bool = True, probe_status: str = "valid"):
    probe = {"status": probe_status, "checked_at_utc": "2026-07-20T00:00:00.000000Z"}
    with patch.object(server, "UDL_ENABLED", enabled), \
         patch.object(server, "get_credential_validity", return_value=probe), \
         patch.dict(os.environ, {}, clear=False):
        os.environ.pop("UDL_USER", None)
        os.environ.pop("UDL_PASS", None)
        if creds:
            os.environ["UDL_USER"] = "u"
            os.environ["UDL_PASS"] = "p"
        return TestClient(server.svc).get("/udl-status").json()


def test_enabled_and_valid_is_catalog_only_never_live():
    """AC1: enabled + valid credentials must NOT read UDL LIVE."""
    r = _udl_status(enabled=True, creds=True, probe_status="valid")
    assert r["mode"] == "connected"
    assert r["label"] == "UDL CONNECTED (CATALOG ONLY)"
    assert "LIVE" not in r["label"]


def test_invalid_credentials_state():
    r = _udl_status(enabled=True, creds=True, probe_status="invalid")
    assert r["mode"] == "invalid"
    assert r["label"] == "UDL CREDENTIALS INVALID"


def test_unreachable_probe_is_unconfirmed():
    r = _udl_status(enabled=True, creds=True, probe_status="unreachable")
    assert r["mode"] == "unconfirmed"


def test_enabled_without_credentials_is_misconfigured():
    r = _udl_status(enabled=True, creds=False)
    assert r["mode"] == "misconfigured"


def test_disabled_state():
    r = _udl_status(enabled=False)
    assert r["mode"] == "disabled"
    assert r["label"] == "UDL DISABLED"


def test_no_state_ever_reports_udl_live():
    """AC1 guarantee across every reachable state in the current build."""
    labels = [
        _udl_status(True, True, "valid")["label"],
        _udl_status(True, True, "invalid")["label"],
        _udl_status(True, True, "unreachable")["label"],
        _udl_status(True, False)["label"],
        _udl_status(False)["label"],
    ]
    assert all(l != "UDL LIVE" for l in labels), labels
