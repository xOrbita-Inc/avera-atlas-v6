"""services/planner/common/leolabs_client.py

Thin, rate-limited client for the LeoLabs Pulse API (SCRUM-411).

Scope and non-goals
-------------------
This module talks to https://api.leolabs.space/v1 and returns raw JSON. It does
no physics, no frame work, and no CDM interpretation. Parsing, the RTN->ECI
rotation, and the Pc cross-check all live in leolabs_cdm_parser.py, so this layer
stays trivial and testable against recorded responses (design section 5).

Auth
----
LeoLabs uses key-pair basic auth with a *literal* scheme, not base64:

    Authorization: basic <access-key>:<secret-key>

Credentials come from the environment (LEOLABS_ACCESS_KEY, LEOLABS_SECRET_KEY),
never hardcoded and never logged. _auth_header() is the only place the secret is
read, and no log statement in this module ever includes it.

Rate limits (design section 2, from the LeoLabs API Usage Guidelines)
--------------------------------------------------------------------
- Org-wide: 4 requests/second, i.e. one every 250 ms. Enforced by a shared
  min-interval limiter that every request passes through.
- On-demand screening create: 3 requests per 2 minutes. Enforced by a separate
  sliding-window limiter, applied on top of the org-wide one.
- Exponential backoff with jitter on 429 and 5xx. Retry-After is honored when
  present.
- Screening latency is 30 s to 2 min, so status is polled (wait_for_screening),
  never awaited inline, with jittered poll times so we do not sit on the top of
  the minute.

Caveat on "org-wide": a single process cannot coordinate a true org-wide budget
with other processes or pods sharing the same key. This limiter throttles within
one process. A genuinely org-wide budget would need a shared token store (Redis
or similar); called out here rather than pretended away.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Iterator, List, Optional

import requests

log = logging.getLogger("planner")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.leolabs.space/v1"

# Org-wide limit: 4 req/s -> one every 250 ms.
_ORG_MIN_INTERVAL_S = 0.25

# Screening create limit: 3 requests per 2 minutes.
_SCREENING_MAX = 3
_SCREENING_WINDOW_S = 120.0

_DEFAULT_TIMEOUT_S = 30.0
_MAX_RETRIES = 5
_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 30.0

# Screening polling (design section 2): latency is 30 s to 2 min, so poll with a
# jittered interval and a generous ceiling rather than awaiting inline.
_SCREENING_POLL_INTERVAL_S = 20.0
_SCREENING_POLL_JITTER_S = 8.0
_SCREENING_POLL_TIMEOUT_S = 300.0

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LeoLabsError(RuntimeError):
    """Base class for LeoLabs client failures."""


class LeoLabsAuthError(LeoLabsError):
    """Missing credentials, or the API rejected them (401/403)."""


class LeoLabsHTTPError(LeoLabsError):
    """A non-retryable, non-auth HTTP error, or retries exhausted."""

    def __init__(self, status: Optional[int], message: str):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Rate limiters
# ---------------------------------------------------------------------------

class MinIntervalLimiter:
    """Allow at most one acquire() per ``min_interval`` seconds, thread-safe.

    A token bucket of size one refilling at a fixed rate. The clock and sleep
    are injectable so tests can drive it deterministically without real waits.
    """

    def __init__(
        self,
        min_interval: float,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_interval = float(min_interval)
        self._now = now
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed = float("-inf")

    def acquire(self) -> None:
        with self._lock:
            now = self._now()
            wait = self._next_allowed - now
            if wait > 0:
                self._sleep(wait)
                now = self._now()
            # Schedule the next slot from whichever is later: now, or the slot we
            # just reserved. Prevents drift and keeps spacing at min_interval.
            self._next_allowed = max(now, self._next_allowed) + self._min_interval


class SlidingWindowLimiter:
    """Allow at most ``max_events`` acquire()s per rolling ``window`` seconds.

    Used for the screening-create limit (3 per 2 minutes). Thread-safe; clock and
    sleep injectable for tests.
    """

    def __init__(
        self,
        max_events: int,
        window: float,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._max = int(max_events)
        self._window = float(window)
        self._now = now
        self._sleep = sleep
        self._lock = threading.Lock()
        self._events: deque[float] = deque()

    def acquire(self) -> None:
        with self._lock:
            now = self._now()
            self._evict(now)
            if len(self._events) >= self._max:
                # Wait until the oldest event ages out of the window.
                wait = self._events[0] + self._window - now
                if wait > 0:
                    self._sleep(wait)
                    now = self._now()
                    self._evict(now)
            self._events.append(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()


# Module-level org-wide limiter, shared by every client instance in this process
# so parallel callers respect the 4 req/s budget together rather than per client.
_ORG_LIMITER = MinIntervalLimiter(_ORG_MIN_INTERVAL_S)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _auth_header() -> Dict[str, str]:
    """Build the LeoLabs auth header from environment credentials.

    LeoLabs' scheme is the literal string ``basic <access>:<secret>``, not
    RFC-7617 base64 basic auth. Raises LeoLabsAuthError if either variable is
    missing. The secret is never logged.
    """
    access = os.environ.get("LEOLABS_ACCESS_KEY")
    secret = os.environ.get("LEOLABS_SECRET_KEY")
    if not access or not secret:
        raise LeoLabsAuthError(
            "LEOLABS_ACCESS_KEY and LEOLABS_SECRET_KEY must be set in the "
            "environment for LeoLabs API access."
        )
    return {
        "Authorization": f"basic {access}:{secret}",
        "Accept": "application/json",
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LeoLabsClient:
    """Thin HTTP client for the LeoLabs Pulse API.

    Every call passes through the shared org-wide limiter. Screening creates pass
    through an additional per-instance sliding-window limiter. All methods return
    parsed JSON (dicts/lists) and do no physics.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_S,
        session: Optional[requests.Session] = None,
        org_limiter: Optional[MinIntervalLimiter] = None,
        screening_limiter: Optional[SlidingWindowLimiter] = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Optional[Callable[[float], float]] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session or requests.Session()
        self._org_limiter = org_limiter or _ORG_LIMITER
        self._screening_limiter = screening_limiter or SlidingWindowLimiter(
            _SCREENING_MAX, _SCREENING_WINDOW_S
        )
        self._sleep = sleep
        # Jitter is injectable so tests are deterministic. Default multiplies the
        # backoff by a random factor in [0.5, 1.5). Imported lazily so a test can
        # patch it without importing random here at module load.
        if jitter is None:
            import random

            def jitter(base: float) -> float:
                return base * (0.5 + random.random())

        self._jitter = jitter

    # -- low-level request with backoff -----------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Perform one API call with rate limiting and retry/backoff.

        Retries 429 and 5xx with exponential backoff plus jitter, honoring
        Retry-After when present. Raises LeoLabsAuthError on 401/403 (never
        retried, since a bad key will not fix itself) and LeoLabsHTTPError on any
        other non-2xx or when retries are exhausted. The auth header, and thus
        the secret, is never logged.
        """
        url = f"{self._base_url}{path}"
        headers = _auth_header()  # may raise LeoLabsAuthError

        attempt = 0
        while True:
            self._org_limiter.acquire()
            try:
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=self._timeout,
                )
            except requests.RequestException as exc:
                if attempt >= _MAX_RETRIES - 1:
                    raise LeoLabsHTTPError(
                        None, f"network error after {attempt + 1} attempts: {exc}"
                    ) from exc
                self._backoff(attempt, retry_after=None)
                attempt += 1
                continue

            status = resp.status_code

            if status in (401, 403):
                # Do not log the response body; it can echo request headers.
                raise LeoLabsAuthError(
                    f"LeoLabs rejected the credentials (HTTP {status})."
                )

            if 200 <= status < 300:
                if not resp.content:
                    return None
                try:
                    return resp.json()
                except ValueError as exc:
                    raise LeoLabsHTTPError(
                        status, f"response was not valid JSON: {exc}"
                    ) from exc

            if status in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                log.warning(
                    "LeoLabs request retrying",
                    extra={
                        "event": "leolabs_retry",
                        "method": method,
                        "path": path,
                        "status": status,
                        "attempt": attempt + 1,
                    },
                )
                self._backoff(attempt, retry_after=resp.headers.get("Retry-After"))
                attempt += 1
                continue

            # Non-retryable, or retries exhausted. Truncate the body and never
            # include request headers.
            detail = (resp.text or "")[:400]
            raise LeoLabsHTTPError(
                status, f"LeoLabs request failed (HTTP {status}): {detail}"
            )

    def _backoff(self, attempt: int, retry_after: Optional[str]) -> None:
        """Sleep before the next retry: honor Retry-After, else jittered exp."""
        delay: Optional[float] = None
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                delay = None
        if delay is None:
            base = min(_BACKOFF_BASE_S * (2 ** attempt), _BACKOFF_CAP_S)
            delay = self._jitter(base)
        self._sleep(delay)

    # -- pagination -------------------------------------------------------

    def _paginate(
        self,
        path: str,
        params: Dict[str, Any],
        items_key: str,
    ) -> Iterator[Dict[str, Any]]:
        """Yield every item across pages, following nextToken.

        LeoLabs list/search endpoints return {<items_key>: [...], total, limit,
        nextToken}. Pagination is required on every search and list call
        (design section 9, item 5).
        """
        page_params = dict(params)
        while True:
            data = self._request("GET", path, params=page_params)
            if not isinstance(data, dict):
                return
            for item in data.get(items_key, []) or []:
                yield item
            next_token = data.get("nextToken")
            if not next_token:
                return
            page_params = dict(params)
            page_params["nextToken"] = next_token

    # -- identity ---------------------------------------------------------

    def list_objects(self, **params: Any) -> List[Dict[str, Any]]:
        """List catalog objects, following pagination.

        Used once to map each of our subscribed sats to its LeoLabs catalog
        number and NORAD id (design sections 3 and 6.3).
        """
        return list(self._paginate("/catalog/objects", params, "objects"))

    def get_object(self, catalog_number: str) -> Dict[str, Any]:
        """Fetch metadata for one object by its LeoLabs catalog number."""
        return self._request("GET", f"/catalog/objects/{catalog_number}")

    def list_subscribed_objects(self) -> List[Dict[str, Any]]:
        """Return the objects visible to this account (the trial subscription).

        On the trial this is the ten ILRS satellites we can screen. Thin wrapper
        over list_objects so callers read intent, not the endpoint.
        """
        return self.list_objects()

    def get_states(self, catalog_number: str, latest: bool = True, **params: Any) -> Any:
        """Fetch state vectors for an object. Optional; the CDM alone carries
        enough to parse and rotate, so this is only a full-EME2000 fallback and a
        data-quality (covariance realism) source (design Appendix A)."""
        if latest:
            params.setdefault("latest", "true")
        return self._request(
            "GET", f"/catalog/objects/{catalog_number}/states", params=params
        )

    # -- conjunctions / CDMs (read) ---------------------------------------

    def search_conjunction_cdms(
        self,
        object1: Optional[str] = None,
        object2: Optional[str] = None,
        min_tca: Optional[str] = None,
        max_tca: Optional[str] = None,
        cdm_source: str = "LeoLabs",
        **extra: Any,
    ) -> List[Dict[str, Any]]:
        """Search CDMs, filtered to a single source, following pagination.

        Uses GET /catalog/conjunctions/cdms/search, the primary read path. Its
        cdmSource filters so *all* returned CDMs come from the given source, so
        cdm_source defaults to "LeoLabs" to keep 18th Space CDMs (which can carry
        DEFAULT covariance) out of the result (design Appendix A). Each returned
        item is a field-keyed CDM JSON object for the parser to read.
        """
        params: Dict[str, Any] = {"cdmSource": cdm_source}
        if object1 is not None:
            params["object1"] = object1
        if object2 is not None:
            params["object2"] = object2
        if min_tca is not None:
            params["minTca"] = min_tca
        if max_tca is not None:
            params["maxTca"] = max_tca
        params.update(extra)
        return list(
            self._paginate("/catalog/conjunctions/cdms/search", params, "cdms")
        )

    def get_cdms(self, cdm_ids: str) -> List[Dict[str, Any]]:
        """Retrieve CDMs by id (comma-separated ids allowed)."""
        data = self._request("GET", f"/catalog/conjunctions/cdms/{cdm_ids}")
        if isinstance(data, dict):
            return data.get("cdms", []) or []
        return data or []

    # -- on-demand screenings ---------------------------------------------

    def create_screening(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Create an on-demand screening.

        Rate-limited to 3 per 2 minutes by the screening limiter, on top of the
        org-wide 4 req/s. The body carries the target object, the time/TCA
        window, and thresholds; exact field names are confirmed against the live
        doc at build time (design section 3, Appendix A).
        """
        self._screening_limiter.acquire()
        return self._request(
            "POST", "/catalog/conjunctions/screenings", json_body=body
        )

    def get_screening_status(self, screening_ids: str) -> Dict[str, Any]:
        """Fetch status for one or more screenings by id."""
        return self._request(
            "GET", f"/catalog/conjunctions/screenings/{screening_ids}"
        )

    def get_screening_cdms(self, screening_id: str) -> List[Dict[str, Any]]:
        """Fetch the result CDM list for a completed screening, paginated."""
        return list(
            self._paginate(
                f"/catalog/conjunctions/screenings/{screening_id}/cdms",
                {},
                "cdms",
            )
        )

    def wait_for_screening(
        self,
        screening_id: str,
        timeout_s: float = _SCREENING_POLL_TIMEOUT_S,
        is_complete: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """Poll a screening until it completes, with jittered intervals.

        Screening latency is 30 s to 2 min, so we poll rather than await inline,
        and we jitter the poll interval so many clients do not converge on the
        top of the minute (design section 2). Returns the final status dict.
        Raises LeoLabsError on timeout.

        is_complete decides when to stop; the default treats a status of
        "completed"/"complete"/"done" as terminal. The exact status vocabulary is
        confirmed against the live API at build time, so it is injectable.
        """
        if is_complete is None:
            def is_complete(status: Dict[str, Any]) -> bool:
                s = str(status.get("status", "")).lower()
                return s in ("completed", "complete", "done", "finished")

        waited = 0.0
        while True:
            status = self.get_screening_status(screening_id)
            record = status
            if isinstance(status, dict) and isinstance(status.get("screenings"), list):
                record = status["screenings"][0] if status["screenings"] else status
            if is_complete(record):
                return record
            if waited >= timeout_s:
                raise LeoLabsError(
                    f"screening {screening_id} did not complete within "
                    f"{timeout_s:.0f}s"
                )
            interval = self._jitter(_SCREENING_POLL_INTERVAL_S)
            interval = max(1.0, min(interval, _SCREENING_POLL_INTERVAL_S
                                    + _SCREENING_POLL_JITTER_S))
            self._sleep(interval)
            waited += interval
