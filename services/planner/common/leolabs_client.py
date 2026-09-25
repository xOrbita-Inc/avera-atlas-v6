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
from dataclasses import dataclass
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

# SCRUM-438: LeoLabs defaults the `paginate` request parameter to false, which
# caps a search/list at 1,000 entries and suppresses nextToken. Every paginated
# GET sends it explicitly, as the lowercase string the API expects (requests
# would serialize a Python bool as "True").
_PAGINATE_TRUE = "true"


def _as_int(value: Any) -> Optional[int]:
    """Parse a count that may arrive as a number or as a numeric string.

    SCRUM-459, and a bug this ticket only found by needing the number: LeoLabs
    returns the ``total`` on a paginated search as a STRING -- "22008", not 22008.
    The check here used to be isinstance(value, (int, float)), which a string never
    satisfies, so ``total`` stayed None on every live response.

    The consequence was that SCRUM-438's retrieved-versus-total cross-check -- added
    precisely because a silent truncation was the failure mode it feared -- has never
    once run against the live API. It could not fire, so its silence was not
    evidence of anything, and SCRUM-439 recorded that silence as the counts agreeing.

    bool is excluded deliberately: it is an int subclass, and True would otherwise
    read as a total of 1.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Pull reporting (SCRUM-459)
# ---------------------------------------------------------------------------

@dataclass
class CdmPullReport:
    """How a bounded CDM search went. Pass one in to find out.

    SCRUM-459. search_conjunction_cdms returns a list either way, so truncation
    cannot be inferred from the return value -- a short window and a truncated
    window are the same list. A caller that bounds the pull passes a report and
    reads `complete` off it.

    Defaults say "whole window", which is what an unbounded caller gets and what a
    test double that simply returns its fixture gets. A bounded pull that stopped
    early is the only thing that sets complete False, and it must be the caller
    that asked for the bound.
    """

    complete: bool = True
    pulled: int = 0
    window_total: Optional[int] = None
    reason: Optional[str] = None          # "deadline" | "cap", when incomplete


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
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Perform one API call with rate limiting and retry/backoff.

        Retries 429 and 5xx with exponential backoff plus jitter, honoring
        Retry-After when present. Raises LeoLabsAuthError on 401/403 (never
        retried, since a bad key will not fix itself) and LeoLabsHTTPError on any
        other non-2xx or when retries are exhausted. The auth header, and thus
        the secret, is never logged.

        Bodies, and why there are two kinds (SCRUM-441)
        ----------------------------------------------
        Most of this API takes JSON. The on-demand screening create does not: it
        is multipart form-data with a file upload, which is why our first live
        submit was rejected with 422 on the first required form field -- the
        server's form parser saw no fields at all in a JSON body.

        So when `files` (or `data`) is given, they are passed to requests and
        `json` is not: sending both would put a JSON body and a multipart body in
        the same request. Content-Type is deliberately not set here either --
        requests generates the multipart boundary, and a hand-set header would
        replace it with one that has no boundary and break the parse.

        File parts must be bytes, not open file objects. This method retries, and
        a file object would be consumed by the first attempt and upload empty on
        the second.
        """
        url = f"{self._base_url}{path}"
        headers = _auth_header()  # may raise LeoLabsAuthError
        multipart = bool(files or data)

        attempt = 0
        while True:
            self._org_limiter.acquire()
            try:
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    json=None if multipart else json_body,
                    files=files,
                    data=data,
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
        on_total: Optional[Callable[[int], None]] = None,
        should_continue: Optional[Callable[[], bool]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield every item across pages, following nextToken.

        LeoLabs list/search endpoints return {<items_key>: [...], total, limit,
        nextToken}. Pagination is required on every search and list call
        (design section 9, item 5).

        The ``paginate`` request parameter must be set explicitly (SCRUM-438).
        When it is absent LeoLabs defaults it to false, caps the response at
        1,000 entries and returns no nextToken, so this loop would stop after one
        page and silently truncate the result. It is sent as the string "true"
        rather than a Python bool because requests renders True as "True", and
        the API expects the lowercase literal. A caller can still override it
        through params if a specific endpoint ever needs paginate off.

        Because a silent truncation is the failure mode this guards against, the
        number of items yielded is cross-checked against the ``total`` the first
        page reports, and a mismatch is logged as a warning.
        """
        page_params = dict(params)
        page_params.setdefault("paginate", _PAGINATE_TRUE)

        yielded = 0
        total: Optional[int] = None
        while True:
            # SCRUM-459: asked before each request, not after. A page can take up to
            # this client's request timeout, so a caller that only checked its
            # deadline between yielded items could overshoot by a whole request it
            # had no budget for. Stopping here means the overshoot is at most the one
            # page already under way when the budget ran out.
            if should_continue is not None and not should_continue():
                break
            data = self._request("GET", path, params=page_params)
            if not isinstance(data, dict):
                break
            if total is None:
                total = _as_int(data.get("total"))
                if total is not None:
                    # SCRUM-459: hand the window's size to the caller as soon as
                    # the first page reports it. A consumer that stops early needs
                    # it to say how much of the window it actually pulled, and by
                    # then it can no longer reach the end of this generator to
                    # find out.
                    if on_total is not None:
                        try:
                            on_total(total)
                        except Exception:
                            pass
            for item in data.get(items_key, []) or []:
                yielded += 1
                yield item
            next_token = data.get("nextToken")
            if not next_token:
                break
            page_params = dict(params)
            page_params.setdefault("paginate", _PAGINATE_TRUE)
            # SCRUM-438: LeoLabs returns the cursor as "nextToken" but expects it
            # sent back on the next request as the "token" parameter, per LeoLabs
            # support guidance. Sending it under "nextToken" is ignored, so the
            # loop would re-fetch page 1 and never advance past 1,000 items.
            page_params["token"] = next_token

        if total is not None and yielded != total:
            log.warning(
                "LeoLabs pagination returned a different count than total",
                extra={
                    "event": "leolabs_pagination_count_mismatch",
                    "path": path,
                    "retrieved": yielded,
                    "total": total,
                },
            )

    # -- identity ---------------------------------------------------------

    def list_objects(self, **params: Any) -> List[Dict[str, Any]]:
        """List catalog objects, following pagination.

        Used once to map each of our subscribed sats to its LeoLabs catalog
        number and NORAD id (design sections 3 and 6.3). Pages carry
        ``paginate=true`` via _paginate, so a catalog larger than 1,000 objects
        comes back whole (SCRUM-438).
        """
        return list(self._paginate("/catalog/objects", params, "objects"))

    def get_object(self, catalog_number: str) -> Dict[str, Any]:
        """Fetch metadata for one object by its LeoLabs catalog number."""
        return self._request("GET", f"/catalog/objects/{catalog_number}")

    def ping(self) -> bool:
        """Cheap authenticated probe to confirm the credentials work.

        Returns True on 2xx. Raises LeoLabsAuthError on 401/403 and
        LeoLabsHTTPError otherwise. Used by the runtime status endpoint so it can
        report credential validity without pulling a full object list.
        """
        self._request("GET", "/catalog/objects", params={"limit": 1})
        return True

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
        deadline_s: Optional[float] = None,
        max_cdms: Optional[int] = None,
        report: Optional[CdmPullReport] = None,
        **extra: Any,
    ) -> List[Dict[str, Any]]:
        """Search CDMs, filtered to a single source, following pagination.

        Uses GET /catalog/conjunctions/cdms/search, the primary read path. Its
        cdmSource filters so *all* returned CDMs come from the given source, so
        cdm_source defaults to "LeoLabs" to keep 18th Space CDMs (which can carry
        DEFAULT covariance) out of the result (design Appendix A). Each returned
        item is a field-keyed CDM JSON object for the parser to read.

        object1 takes either a catalog number for the per-asset evaluate
        (``object1="L2669"``, what the runtime uses) or the literal ``"C0"`` to
        pull every vehicle the subscription covers in one search. ``**extra``
        passes any further query parameter straight through; the volume filters
        ``maxRelativePositionR``, ``maxRelativePositionI`` and
        ``maxRelativePositionC`` (metres, RIC) are the ones worth knowing about.

        Every page carries ``paginate=true`` via _paginate, so the result is the
        complete set rather than the first 1,000 CDMs (SCRUM-438).

        Bounding the pull (SCRUM-459)
        -----------------------------
        ``deadline_s`` and ``max_cdms`` stop the pagination early, and ``report``
        is how the caller finds out that it did. All three default to off, so every
        existing caller gets exactly the unbounded pull it always got.

        Bounding lives here because the pagination does. The densest subscribed
        asset (SWARM B, flying in formation with SWARM A and C) has a window that
        does not come back inside the ui-to-planner 60 s read timeout at all, so the
        listing asks for a bounded pull and presents the result as partial. What it
        must not do is present a truncated prefix as a whole window: CDMs arrive in
        LeoLabs order, not Pc order, so the prefix can be missing the worst
        conjunction.

        The deadline is checked before each page request rather than only between
        items, so the overshoot is at most the single page already under way when the
        budget ran out. Measured against the live API at about 10 s per page and a
        30 s per-request timeout, that puts the worst case at roughly
        ``deadline_s`` + 30 s and the typical case at ``deadline_s`` + 10 s. Pick
        ``deadline_s`` with room for that under whatever timeout is downstream.
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

        if deadline_s is None and max_cdms is None:
            # Unbounded: byte-for-byte the pre-SCRUM-459 path.
            items = list(
                self._paginate("/catalog/conjunctions/cdms/search", params, "cdms")
            )
            if report is not None:
                report.complete = True
                report.pulled = len(items)
            return items

        started = time.monotonic()
        seen_total: List[Optional[int]] = [None]
        out_of_time = [False]

        def _note_total(total: int) -> None:
            seen_total[0] = int(total)

        def _more() -> bool:
            if deadline_s is None:
                return True
            if time.monotonic() - started >= deadline_s:
                out_of_time[0] = True
                return False
            return True

        out: List[Dict[str, Any]] = []
        reason: Optional[str] = None
        for item in self._paginate(
            "/catalog/conjunctions/cdms/search", params, "cdms",
            on_total=_note_total, should_continue=_more,
        ):
            out.append(item)
            if max_cdms is not None and len(out) >= max_cdms:
                reason = "cap"
                break
        if reason is None and out_of_time[0]:
            reason = "deadline"

        if report is not None:
            report.complete = reason is None
            report.pulled = len(out)
            report.window_total = seen_total[0]
            report.reason = reason
        if reason is not None:
            log.warning(
                "LeoLabs CDM pull stopped early; the result is a partial window",
                extra={"event": "leolabs_cdm_pull_truncated", "reason": reason,
                       "pulled": len(out), "window_total": seen_total[0],
                       "elapsed_s": round(time.monotonic() - started, 2),
                       "deadline_s": deadline_s, "max_cdms": max_cdms},
            )
        return out

    def iter_conjunction_cdms(
        self,
        object1: Optional[str] = None,
        object2: Optional[str] = None,
        min_tca: Optional[str] = None,
        max_tca: Optional[str] = None,
        cdm_source: str = "LeoLabs",
        on_total: Optional[Callable[[int], None]] = None,
        **extra: Any,
    ) -> Iterator[Dict[str, Any]]:
        """search_conjunction_cdms, streamed rather than materialised. SCRUM-459.

        Same endpoint, same parameters, same order -- search_conjunction_cdms is
        now literally list() of this, so the two cannot diverge. The difference is
        that a caller can stop consuming.

        That is what lets the listing bound a cold fetch: the densest subscribed
        asset flies in formation with two others and its window does not come back
        inside the ui-to-planner 60 s timeout, so the listing pulls under a
        wall-clock deadline and an item cap and reports the result as incomplete
        rather than running on until the request fails.

        `on_total` receives the window size LeoLabs reports on the first page, so a
        caller that stops early can still say how much of the window it holds.

        Stopping early abandons the generator mid-pagination. The HTTP request in
        flight at that moment still has to come back -- each one is bounded by the
        client's own request timeout -- so a deadline here bounds the pull to
        roughly the deadline plus one request timeout, not to the deadline exactly.
        Choose the deadline with that headroom in mind.
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
        return self._paginate(
            "/catalog/conjunctions/cdms/search", params, "cdms", on_total=on_total
        )

    def get_cdm_summaries(self, cdm_ids: str) -> List[Dict[str, Any]]:
        """Look up conjunction SUMMARIES by CDM id (comma-separated ids allowed).

        Read the name carefully: despite the path, this endpoint does **not** return
        CCSDS CDMs. It returns a compact conjunction record --

            {"id": 79867238980, "conjunction": 3590491017,
             "tca": "2026-09-26T18:05:15.953564Z", "sat1": "L3969",
             "sat2": "L186018", "missDistance": 11226.646,
             "relativeSpeed": 14828.261, "collisionProbability": 7.035e-05,
             "source": "leolabs"}

        -- with no state vectors and no covariance, so parse_leolabs_cdm cannot read
        it and nothing can be scored from it. Confirmed against the live API
        (SCRUM-460); `?format=ccsds` is ignored and returns the same shape.

        What it is good for is coordinates: the tca and the two object designators
        are exactly what a narrow, one-page search needs to pull the real CDMs for
        one event, which is how leolabs_runtime resolves a clicked row without
        pulling the whole window.

        It was previously named get_cdms and read ``data["cdms"]``. The response key
        is ``conjunctions``, so it returned an empty list for every id ever passed to
        it. Nothing called it, which is why that went unnoticed; the rename is so the
        next caller is not misled into expecting a CDM.
        """
        data = self._request("GET", f"/catalog/conjunctions/cdms/{cdm_ids}")
        if isinstance(data, dict):
            return data.get("conjunctions") or data.get("cdms") or []
        return data or []

    # -- on-demand screenings ---------------------------------------------

    def create_screening(
        self,
        body: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create an on-demand screening.

        Rate-limited to 3 per 2 minutes by the screening limiter, on top of the
        org-wide 4 req/s. Both limiters and the retry/backoff are _request's, and
        are the same whichever body shape is used.

        SCRUM-441: the endpoint is multipart form-data with a file upload, not
        JSON -- "Parameters must be sent as a multipart form" in the LeoLabs
        reference. Callers pass `files` (the ephemeris part) and `data` (the flat
        form fields); leolabs_screening.build_screening_request assembles both and
        owns the field names. The positional `body` is retained only so an
        existing JSON caller keeps working, and is ignored when files/data are
        given.
        """
        self._screening_limiter.acquire()
        if files or data:
            return self._request(
                "POST", "/catalog/conjunctions/screenings", files=files, data=data
            )
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
