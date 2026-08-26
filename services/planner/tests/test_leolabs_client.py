"""tests/test_leolabs_client.py

SCRUM-411: unit tests for the LeoLabs API client.

All tests are offline: the HTTP session is a mock and the clocks/sleeps are
injected, so nothing here touches the network or waits in real time. The live
end-to-end probe stays opt-in behind the credentials and out of CI
(see live_leolabs_smoke.py).

Run from repo root:
    python -m pytest services/planner/tests/test_leolabs_client.py -v
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from common.leolabs_client import (
    LeoLabsAuthError,
    LeoLabsClient,
    LeoLabsHTTPError,
    MinIntervalLimiter,
    SlidingWindowLimiter,
    _auth_header,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeClock:
    """A monotonic clock whose sleep() advances the clock instead of waiting."""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        # Real time.sleep ignores negatives; mirror that.
        if dt > 0:
            self.t += dt
            self.slept += dt


def _resp(status: int, json_body=None, text: str = "", headers=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.headers = headers or {}
    r.content = b"x" if (json_body is not None or text) else b""
    r.text = text
    if json_body is not None:
        r.json.return_value = json_body
    else:
        r.json.side_effect = ValueError("no json")
    return r


def _client(session, **kw):
    """A client with no real waits: null jitter and a zero-interval org limiter."""
    clock = FakeClock()
    return LeoLabsClient(
        session=session,
        org_limiter=MinIntervalLimiter(0.0, now=clock.now, sleep=clock.sleep),
        screening_limiter=SlidingWindowLimiter(
            3, 120.0, now=clock.now, sleep=clock.sleep
        ),
        sleep=clock.sleep,
        jitter=lambda base: base,  # deterministic
        **kw,
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_auth_header_uses_literal_basic_scheme():
    with patch.dict(os.environ, {"LEOLABS_ACCESS_KEY": "ak", "LEOLABS_SECRET_KEY": "sk"}):
        h = _auth_header()
    # Literal "basic <access>:<secret>", NOT base64 RFC-7617 basic auth.
    assert h["Authorization"] == "basic ak:sk"
    assert h["Accept"] == "application/json"


def test_auth_header_missing_credentials_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(LeoLabsAuthError):
            _auth_header()


def test_secret_not_in_auth_error_message():
    with patch.dict(os.environ, {}, clear=True):
        try:
            _auth_header()
        except LeoLabsAuthError as exc:
            assert "sk" not in str(exc)


# ---------------------------------------------------------------------------
# Rate limiters
# ---------------------------------------------------------------------------

def test_min_interval_limiter_spaces_acquires():
    clock = FakeClock()
    lim = MinIntervalLimiter(0.25, now=clock.now, sleep=clock.sleep)
    lim.acquire()          # first is free
    assert clock.slept == 0.0
    lim.acquire()          # must wait one interval
    assert clock.slept == pytest.approx(0.25)
    lim.acquire()
    assert clock.slept == pytest.approx(0.50)


def test_sliding_window_limiter_allows_burst_then_throttles():
    clock = FakeClock()
    lim = SlidingWindowLimiter(3, 120.0, now=clock.now, sleep=clock.sleep)
    for _ in range(3):
        lim.acquire()      # 3 in the window are free
    assert clock.slept == 0.0
    lim.acquire()          # the 4th waits until the oldest ages out
    assert clock.slept == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# _request: success, auth, retry, exhaustion
# ---------------------------------------------------------------------------

def _env():
    return patch.dict(
        os.environ, {"LEOLABS_ACCESS_KEY": "ak", "LEOLABS_SECRET_KEY": "sk"}
    )


def test_request_returns_json_on_200():
    session = MagicMock()
    session.request.return_value = _resp(200, json_body={"ok": True})
    with _env():
        out = _client(session)._request("GET", "/catalog/objects")
    assert out == {"ok": True}


def test_request_raises_auth_on_401():
    session = MagicMock()
    session.request.return_value = _resp(401, text="nope")
    with _env():
        with pytest.raises(LeoLabsAuthError):
            _client(session)._request("GET", "/catalog/objects")


def test_request_retries_on_429_then_succeeds():
    session = MagicMock()
    session.request.side_effect = [
        _resp(429, text="slow down", headers={"Retry-After": "2"}),
        _resp(200, json_body={"ok": 1}),
    ]
    with _env():
        out = _client(session)._request("GET", "/catalog/objects")
    assert out == {"ok": 1}
    assert session.request.call_count == 2


def test_request_retries_on_5xx_then_exhausts():
    session = MagicMock()
    session.request.return_value = _resp(503, text="unavailable")
    with _env():
        with pytest.raises(LeoLabsHTTPError) as ei:
            _client(session)._request("GET", "/catalog/objects")
    assert ei.value.status == 503
    assert session.request.call_count == 5  # _MAX_RETRIES


def test_request_retries_on_network_error_then_succeeds():
    session = MagicMock()
    session.request.side_effect = [
        requests.ConnectionError("boom"),
        _resp(200, json_body={"ok": 1}),
    ]
    with _env():
        out = _client(session)._request("GET", "/catalog/objects")
    assert out == {"ok": 1}


# ---------------------------------------------------------------------------
# Pagination and CDM search
# ---------------------------------------------------------------------------

def test_search_conjunction_cdms_follows_next_token():
    session = MagicMock()
    session.request.side_effect = [
        _resp(200, json_body={"cdms": [{"id": 1}], "nextToken": "tok"}),
        _resp(200, json_body={"cdms": [{"id": 2}]}),
    ]
    with _env():
        cdms = _client(session).search_conjunction_cdms(object1="L2669")
    assert [c["id"] for c in cdms] == [1, 2]

    # First call carries cdmSource=LeoLabs and object1; second carries nextToken.
    first_params = session.request.call_args_list[0].kwargs["params"]
    assert first_params["cdmSource"] == "LeoLabs"
    assert first_params["object1"] == "L2669"
    second_params = session.request.call_args_list[1].kwargs["params"]
    assert second_params["nextToken"] == "tok"


def test_search_conjunction_cdms_defaults_source_to_leolabs():
    session = MagicMock()
    session.request.return_value = _resp(200, json_body={"cdms": []})
    with _env():
        _client(session).search_conjunction_cdms(object1="L2669")
    params = session.request.call_args_list[0].kwargs["params"]
    # Filtering to the LeoLabs source keeps 18th Space DEFAULT-covariance CDMs out.
    assert params["cdmSource"] == "LeoLabs"


# ---------------------------------------------------------------------------
# Screenings
# ---------------------------------------------------------------------------

def test_create_screening_passes_through_screening_limiter():
    session = MagicMock()
    session.request.return_value = _resp(200, json_body={"id": "scr1"})
    clock = FakeClock()
    spy = SlidingWindowLimiter(3, 120.0, now=clock.now, sleep=clock.sleep)
    with _env():
        client = LeoLabsClient(
            session=session,
            org_limiter=MinIntervalLimiter(0.0, now=clock.now, sleep=clock.sleep),
            screening_limiter=spy,
            sleep=clock.sleep,
            jitter=lambda base: base,
        )
        for _ in range(3):
            client.create_screening({"target": "L2669"})
        assert clock.slept == 0.0
        client.create_screening({"target": "L2669"})  # 4th throttles
    assert clock.slept == pytest.approx(120.0)


def test_wait_for_screening_polls_until_complete():
    session = MagicMock()
    session.request.side_effect = [
        _resp(200, json_body={"status": "processing"}),
        _resp(200, json_body={"status": "processing"}),
        _resp(200, json_body={"status": "completed"}),
    ]
    with _env():
        final = _client(session).wait_for_screening("scr1", timeout_s=1000)
    assert final["status"] == "completed"
    assert session.request.call_count == 3


def test_wait_for_screening_times_out():
    session = MagicMock()
    session.request.return_value = _resp(200, json_body={"status": "processing"})
    with _env():
        with pytest.raises(Exception):
            _client(session).wait_for_screening("scr1", timeout_s=5)
