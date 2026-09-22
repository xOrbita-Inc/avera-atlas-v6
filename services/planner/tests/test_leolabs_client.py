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

import logging
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


class _CaptureWarnings:
    """Collect records off the planner logger directly.

    logging_setup sets propagate=False on "planner", so pytest's caplog (which
    listens on the root logger) never sees these records once another test has
    configured logging. Attaching a handler to the logger itself is independent
    of that ordering.
    """

    def __init__(self, logger_name: str = "planner") -> None:
        self._logger = logging.getLogger(logger_name)
        self.records: list[logging.LogRecord] = []

    def __enter__(self) -> "_CaptureWarnings":
        records = self.records

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        self._handler = _Handler(level=logging.WARNING)
        self._prev_level = self._logger.level
        self._logger.setLevel(logging.WARNING)
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc) -> None:
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._prev_level)

    def events(self, event: str) -> list[logging.LogRecord]:
        return [r for r in self.records if getattr(r, "event", None) == event]


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

    # First call carries cdmSource=LeoLabs and object1; second carries the cursor
    # as `token` (LeoLabs returns it as `nextToken`, expects it back as `token`).
    first_params = session.request.call_args_list[0].kwargs["params"]
    assert first_params["cdmSource"] == "LeoLabs"
    assert first_params["object1"] == "L2669"
    second_params = session.request.call_args_list[1].kwargs["params"]
    assert second_params["token"] == "tok"
    assert "nextToken" not in second_params


def test_search_conjunction_cdms_paginates_across_two_pages():
    """SCRUM-438: a 1,500-CDM result comes back whole, not capped at page one.

    Without `paginate=true` LeoLabs caps the response at 1,000 entries and omits
    nextToken, so the loop would stop after page one and silently truncate. Page
    sizes here mirror that cap: 1,000 then 500, total 1,500.
    """
    page1 = [{"id": i} for i in range(1000)]
    page2 = [{"id": i} for i in range(1000, 1500)]
    session = MagicMock()
    session.request.side_effect = [
        _resp(200, json_body={"cdms": page1, "total": 1500, "nextToken": "tok"}),
        _resp(200, json_body={"cdms": page2, "total": 1500}),
    ]
    with _env():
        cdms = _client(session).search_conjunction_cdms(object1="L2669")

    # All 1500 returned, in order, and the loop stopped once nextToken was gone.
    assert len(cdms) == 1500
    assert [c["id"] for c in cdms] == list(range(1500))
    assert session.request.call_count == 2

    # Both pages carry the flag; the second also carries the token.
    first_params = session.request.call_args_list[0].kwargs["params"]
    second_params = session.request.call_args_list[1].kwargs["params"]
    assert first_params["paginate"] == "true"
    assert second_params["paginate"] == "true"
    assert second_params["token"] == "tok"
    assert "nextToken" not in second_params
    # The filter params survive onto page two rather than being dropped.
    assert second_params["object1"] == "L2669"
    assert second_params["cdmSource"] == "LeoLabs"


def test_paginate_sends_cursor_under_token_not_next_token():
    """SCRUM-438: the cursor is echoed back as `token`, not `nextToken`.

    LeoLabs returns the cursor in the response field `nextToken` but expects it
    on the *next request* under the parameter name `token`. Sending it as
    `nextToken` is silently ignored by the API, so the loop re-fetches page one
    and never advances past the first 1,000 items. This locks the wire name so
    the two names cannot drift back together.
    """
    session = MagicMock()
    session.request.side_effect = [
        _resp(200, json_body={"cdms": [{"id": 1}], "nextToken": "CURSOR-123"}),
        _resp(200, json_body={"cdms": [{"id": 2}]}),
    ]
    with _env():
        _client(session).search_conjunction_cdms(object1="C0")

    second_params = session.request.call_args_list[1].kwargs["params"]
    # Sent under `token`, carrying the value the response gave as `nextToken`.
    assert second_params["token"] == "CURSOR-123"
    # And never echoed back under the response's own key.
    assert "nextToken" not in second_params


def test_paginate_warns_when_retrieved_count_misses_total():
    """A short read against a reported total is logged, not swallowed."""
    session = MagicMock()
    session.request.return_value = _resp(
        200, json_body={"cdms": [{"id": 1}], "total": 1500}
    )
    with _env():
        with _CaptureWarnings() as cap:
            cdms = _client(session).search_conjunction_cdms(object1="L2669")

    assert len(cdms) == 1
    mismatches = cap.events("leolabs_pagination_count_mismatch")
    assert len(mismatches) == 1
    assert mismatches[0].retrieved == 1
    assert mismatches[0].total == 1500


def test_paginate_does_not_warn_when_count_matches_total():
    session = MagicMock()
    session.request.side_effect = [
        _resp(200, json_body={"cdms": [{"id": 1}], "total": 2, "nextToken": "tok"}),
        _resp(200, json_body={"cdms": [{"id": 2}], "total": 2}),
    ]
    with _env():
        with _CaptureWarnings() as cap:
            cdms = _client(session).search_conjunction_cdms(object1="L2669")

    assert len(cdms) == 2
    assert not cap.events("leolabs_pagination_count_mismatch")


def test_list_objects_sets_paginate_flag():
    """SCRUM-438: the object listing pages fully too, not just the CDM search."""
    session = MagicMock()
    session.request.return_value = _resp(
        200, json_body={"objects": [{"catalogNumber": "L2669"}], "total": 1}
    )
    with _env():
        objects = _client(session).list_objects()
    assert len(objects) == 1
    assert session.request.call_args_list[0].kwargs["params"]["paginate"] == "true"


def test_search_conjunction_cdms_accepts_c0_and_volume_filters():
    """object1='C0' (full subscription) and the RIC volume filters pass through."""
    session = MagicMock()
    session.request.return_value = _resp(200, json_body={"cdms": [], "total": 0})
    with _env():
        _client(session).search_conjunction_cdms(
            object1="C0", maxRelativePositionR=2000, maxRelativePositionI=25000
        )
    params = session.request.call_args_list[0].kwargs["params"]
    assert params["object1"] == "C0"
    assert params["maxRelativePositionR"] == 2000
    assert params["maxRelativePositionI"] == 25000
    assert params["paginate"] == "true"


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
