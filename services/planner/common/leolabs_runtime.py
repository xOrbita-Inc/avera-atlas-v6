"""services/planner/common/leolabs_runtime.py

Runtime glue that makes the planner actually use the LeoLabs live feed
(SCRUM-412, follow-on to SCRUM-411).

SCRUM-411 delivered the client, parser, asset map and evaluate-request builder,
but nothing in the running app called them. This module is the bridge: behind the
LEOLABS_ENABLED flag it fetches a live CDM for a subscribed sat, resolves our
object via the asset registry, and parses it into a ParsedLeoLabsCDM the evaluate
path can score. It mirrors the UDL_ENABLED pattern in udl_client.py.

Feature flag and precedence
---------------------------
LEOLABS_ENABLED (default false) gates all live behaviour, so CI needs no
credential. When both LEOLABS_ENABLED and UDL_ENABLED are set, LeoLabs wins:
LeoLabs replaces the UDL route, which structurally could not supply covariance.

Caching
-------
The client and the asset registry are built once per process and reused, so the
registry's list call is not repeated per evaluate and the org-wide rate limit is
respected. Call reset_caches() in tests.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional, Set

from common.leolabs_asset_map import AssetRegistry
from common.leolabs_cdm_parser import (
    LeoLabsParseError,
    ParsedLeoLabsCDM,
    parse_leolabs_cdm,
)
from common.leolabs_client import (
    LeoLabsAuthError,
    LeoLabsClient,
    LeoLabsError,
)
from common.leolabs_conjunction_list import dedupe_raw_by_event

log = logging.getLogger("planner")

# Feature flag -- default false until credentials are provisioned in the
# deployed environment and the trial is confirmed active.
LEOLABS_ENABLED = os.environ.get("LEOLABS_ENABLED", "false").lower() == "true"

# Runtime conjunction window. Only upcoming conjunctions are actionable for an
# avoidance planner, so the default window is now..now+lookahead. LeoLabs caps
# the total minTca..maxTca span at 30 days.
_DEFAULT_LOOKAHEAD_DAYS = 7
_DEFAULT_LOOKBACK_DAYS = 0

# Public aliases, so a caller can state the window it asked for without reaching
# for a private name. LeoLabs caps the total minTca..maxTca span at 30 days, so a
# caller widening the window has to stay inside MAX_WINDOW_DAYS.
DEFAULT_LOOKAHEAD_DAYS = _DEFAULT_LOOKAHEAD_DAYS
DEFAULT_LOOKBACK_DAYS = _DEFAULT_LOOKBACK_DAYS
MAX_WINDOW_DAYS = 30

# Optional allow-list of our fleet's NORAD ids, comma-separated. When set, the
# registry is restricted to these so we never treat an unrelated subscribed
# object as ours.
_ASSET_NORADS_ENV = "LEOLABS_ASSET_NORADS"

_PROBE_INTERVAL_SECONDS = 1800  # 30 minutes, matches the UDL probe cadence

# Paging (SCRUM-445). The listing used to fetch and parse every scorable CDM in
# the window in one synchronous call. Once SCRUM-438 made the pull complete, a
# dense asset (SWARM C / 39453) overran the ui-to-planner 60 s read timeout. The
# fetch of ~1,850 CDMs is the larger fixed cost (~16-20 s, measured); parsing all
# of them on top of it is what pushed the request past 60 s. A page bounds the
# parse, which is the part this module controls.
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500

# The three LeoLabs RIC volume filters, as the API spells them.
VOLUME_FILTER_PARAMS = (
    "maxRelativePositionR",
    "maxRelativePositionI",
    "maxRelativePositionC",
)

# LeoLabs' own reporting volume, in metres: it issues an initial CDM when the
# overall miss is within 2 x 50 x 50 km (radial x in-track x cross-track) and
# reissues while the overall miss is under 100 km. It is LeoLabs' own reporting
# criterion rather than a number invented here.
#
# Measured against SWARM C (39453) on 2026-09-22, this filter is NOT what makes a
# dense asset tractable: the window held 1,852 CDMs unfiltered and 1,841 inside
# this volume, collapsing to the same 96 conjunction events either way. LeoLabs
# only issues CDMs inside roughly this volume already, so the filter trims the
# margin, not the bulk. What actually bounds the work is deduping reissues before
# parsing (1,841 CDMs -> 96 events) and parsing only the returned page. The filter
# is kept because the ticket asks for it and because a narrower explicit volume is
# a useful operator control, not because it is load-bearing here.
DEFAULT_REPORTING_VOLUME_M: Dict[str, float] = {
    "maxRelativePositionR": 2_000.0,
    "maxRelativePositionI": 50_000.0,
    "maxRelativePositionC": 50_000.0,
}

# Cursor format version, so a cursor minted by an older build is rejected rather
# than silently reinterpreted as an offset into a different query.
_CURSOR_VERSION = 1


class LeoLabsRuntimeError(LeoLabsError):
    """A runtime precondition failed (e.g. an unsubscribed primary)."""


# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_client: Optional[LeoLabsClient] = None
_registry: Optional[AssetRegistry] = None
_last_fetch_utc: Optional[str] = None

_last_probe_monotonic: Optional[float] = None
_last_probe_result: Optional[Dict[str, Any]] = None


def reset_caches() -> None:
    """Drop the cached client, registry and probe result. For tests."""
    global _client, _registry, _last_fetch_utc
    global _last_probe_monotonic, _last_probe_result
    with _lock:
        _client = None
        _registry = None
        _last_fetch_utc = None
        _last_probe_monotonic = None
        _last_probe_result = None


def _our_norads() -> Optional[Set[int]]:
    raw = os.environ.get(_ASSET_NORADS_ENV)
    if not raw:
        return None
    norads: Set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                norads.add(int(part))
            except ValueError:
                log.warning(
                    "ignoring non-integer NORAD id in %s: %r",
                    _ASSET_NORADS_ENV, part,
                    extra={"event": "leolabs_bad_asset_norad", "value": part},
                )
    return norads or None


def get_client() -> LeoLabsClient:
    """Return the process-wide LeoLabs client, building it on first use."""
    global _client
    with _lock:
        if _client is None:
            _client = LeoLabsClient()
        return _client


def get_registry(client: Optional[LeoLabsClient] = None) -> AssetRegistry:
    """Return the process-wide asset registry, built once from the client.

    This is the "mapped once via list_subscribed_objects" step. Restricted to
    LEOLABS_ASSET_NORADS when that env var is set.
    """
    global _registry
    with _lock:
        if _registry is None:
            _registry = AssetRegistry.from_client(
                client or get_client(), our_norads=_our_norads()
            )
        return _registry


def last_fetch_utc() -> Optional[str]:
    return _last_fetch_utc


# ---------------------------------------------------------------------------
# Fetch + select + parse
# ---------------------------------------------------------------------------

def _risk_key(cdm: Dict[str, Any]):
    """Order CDMs by risk: highest Pc first, earliest TCA as tie-break."""
    pc = cdm.get("COLLISION_PROBABILITY") or 0.0
    try:
        pc = float(pc)
    except (TypeError, ValueError):
        pc = 0.0
    tca = str(cdm.get("TCA_ISO") or cdm.get("TCA") or "")
    return (-pc, tca)


def conjunction_window(
    now: datetime,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    lookahead_days: int = _DEFAULT_LOOKAHEAD_DAYS,
) -> tuple[str, str]:
    """The (min_tca, max_tca) TCA window a fetch searches, as LeoLabs strings.

    Returned as well as used so a listing response can state the window its rows
    came from; an empty table means something different over one hour than over
    seven days, and the caller should not have to guess which it asked for.
    """
    return (
        (now - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        (now + timedelta(days=lookahead_days)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _scorable_in_risk_order(
    primary_norad: int,
    client: Optional[LeoLabsClient],
    registry: Optional[AssetRegistry],
    now: datetime,
    lookback_days: int,
    lookahead_days: int,
) -> Iterator[ParsedLeoLabsCDM]:
    """Yield every scorable CDM for a subscribed sat, highest risk first.

    A generator on purpose. The singular fetch takes the first item and stops, so
    it still parses only as far as it needs to -- the same lazy behaviour it had
    before SCRUM-422 split this out -- while the list fetch drains it. One code
    path, two consumption patterns, no duplicated fetch/sort/skip logic.

    CDMs that fail a guard (e.g. a non-CALCULATED covariance from an 18th Space
    CDM that slipped through) are skipped, not fatal.
    """
    global _last_fetch_utc

    client = client or get_client()
    registry = registry or get_registry(client)

    catalog = registry.leolabs_for_norad(int(primary_norad))
    if catalog is None:
        raise LeoLabsRuntimeError(
            f"NORAD {primary_norad} is not in the LeoLabs subscribed-objects "
            f"registry; it cannot be screened on this account."
        )

    min_tca, max_tca = conjunction_window(now, lookback_days, lookahead_days)

    cdms = client.search_conjunction_cdms(
        object1=catalog, min_tca=min_tca, max_tca=max_tca, cdm_source="LeoLabs"
    )
    if not cdms:
        log.info(
            "LeoLabs returned no CDMs in the window",
            extra={"event": "leolabs_no_cdms", "catalog": catalog,
                   "primary_norad": primary_norad},
        )
        return

    yielded = 0
    for cdm in sorted(cdms, key=_risk_key):
        try:
            our_id = registry.resolve_our_catalog_id(cdm)
            parsed = parse_leolabs_cdm(cdm, our_id)
        except (LeoLabsParseError, LookupError) as exc:
            log.info(
                "skipping LeoLabs CDM that failed parse/guards",
                extra={"event": "leolabs_cdm_skipped", "catalog": catalog,
                       "reason": str(exc)},
            )
            continue
        _last_fetch_utc = now.isoformat().replace("+00:00", "Z")
        yielded += 1
        yield parsed

    if yielded == 0:
        log.info(
            "no scorable LeoLabs CDMs after guards",
            extra={"event": "leolabs_no_scorable_cdms", "catalog": catalog},
        )


def fetch_leolabs_conjunctions(
    primary_norad: int,
    *,
    client: Optional[LeoLabsClient] = None,
    registry: Optional[AssetRegistry] = None,
    now: Optional[datetime] = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    lookahead_days: int = _DEFAULT_LOOKAHEAD_DAYS,
) -> List[ParsedLeoLabsCDM]:
    """Fetch and parse every scorable LeoLabs CDM for a subscribed sat.

    SCRUM-422. The search was always fetching the whole window; SCRUM-412 simply
    discarded everything below the top. This returns the lot, ordered highest Pc
    first with earliest TCA as the tie-break -- the same ordering the UDL path
    selects on, so a table sorted by this list and a single evaluate agree about
    which conjunction is worst.

    Returns an empty list when there are no scorable CDMs in the window; an empty
    window is a real and unremarkable state of the world, not an error. Raises
    LeoLabsRuntimeError when the primary is not in our subscribed registry, and
    propagates LeoLabsError subclasses on transport/auth failures.

    Note that this is one entry per CDM. LeoLabs reissues CDMs for the same event
    as the solution refines, so a caller rendering a table should pass the result
    through leolabs_conjunction_list.dedupe_by_event first.
    """
    now = now or datetime.now(timezone.utc)
    return list(
        _scorable_in_risk_order(
            primary_norad, client, registry, now, lookback_days, lookahead_days
        )
    )


# ---------------------------------------------------------------------------
# Paging (SCRUM-445)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConjunctionPage:
    """One page of an asset's conjunctions, plus what the caller needs to page on.

    rows are parsed CDMs, one per conjunction event, ordered highest Pc first with
    earliest TCA as the tie-break -- the same ordering the unpaged listing used.
    total counts events across the whole in-volume set, not just this page, so a
    dashboard can say "1-100 of 1,432" honestly.
    """

    rows: List[ParsedLeoLabsCDM]
    next_cursor: Optional[str]
    total: int
    # How many CDMs the window held before reissues for one event were collapsed.
    # Shown so a caller can see that reissues were collapsed, not dropped.
    cdm_total: int
    offset: int
    page_size: int
    volume_filters: Dict[str, float] = dc_field(default_factory=dict)

    @property
    def in_volume(self) -> bool:
        """Whether a reporting-volume bound was applied to this page's fetch."""
        return bool(self.volume_filters)


class LeoLabsCursorError(LeoLabsError):
    """A paging cursor was malformed, or belongs to a different query."""


def _query_fingerprint(
    primary_norad: int,
    lookback_days: int,
    lookahead_days: int,
    page_size: int,
    volume_filters: Dict[str, float],
) -> str:
    """A short digest of everything a cursor's offset is only valid against.

    A cursor is an offset into one specific ordered result set. Replayed against a
    different asset, window, page size or volume filter it would point at an
    unrelated row and the operator would page from one asset's list into another's.
    Binding the offset to a fingerprint of the query makes that a 422 instead.

    Bound to the requested window in days, deliberately NOT to the resolved
    absolute minTca..maxTca. Those are computed from the wall clock, so they move
    between the request that mints a cursor and the request that spends it, and
    fingerprinting them would reject every "next page" click as a foreign cursor.
    The set itself does shift slightly as the clock advances -- unavoidable when
    paging a live feed -- but the offset stays an offset into the same query.
    """
    payload = json.dumps(
        {
            "n": int(primary_norad),
            "lb": int(lookback_days),
            "la": int(lookahead_days),
            "p": int(page_size),
            "v": {k: volume_filters[k] for k in sorted(volume_filters)},
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def encode_cursor(offset: int, fingerprint: str) -> str:
    """An opaque forward cursor. Opaque on purpose: it is ours, not LeoLabs'."""
    raw = json.dumps(
        {"v": _CURSOR_VERSION, "o": int(offset), "q": fingerprint},
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str, fingerprint: str) -> int:
    """The offset a cursor names, or raise LeoLabsCursorError.

    Rejects a cursor from a different query or an older cursor format rather than
    treating its offset as valid here.
    """
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        version = int(data["v"])
        offset = int(data["o"])
        query = str(data["q"])
    except Exception as exc:
        raise LeoLabsCursorError(f"malformed cursor: {exc}") from exc
    if version != _CURSOR_VERSION:
        raise LeoLabsCursorError(
            f"cursor format v{version} is not v{_CURSOR_VERSION}; re-read page one"
        )
    if query != fingerprint:
        raise LeoLabsCursorError(
            "cursor belongs to a different asset, window, page size or volume "
            "filter; re-read page one"
        )
    if offset < 0:
        raise LeoLabsCursorError("cursor offset is negative")
    return offset


def resolve_volume_filters(
    max_r_m: Optional[float] = None,
    max_i_m: Optional[float] = None,
    max_c_m: Optional[float] = None,
    *,
    in_volume: bool = True,
) -> Dict[str, float]:
    """The RIC volume filters a fetch should send.

    in_volume=True with no explicit values means LeoLabs' own reporting volume,
    which is the listing default. Any explicitly supplied axis overrides that
    axis. in_volume=False drops the bound entirely, which is the "widen" control.

    See DEFAULT_REPORTING_VOLUME_M: on a real dense asset this filter turned out
    to change the raw CDM count by well under 1% and the event count not at all,
    so do not read the default as the thing keeping the listing fast.
    """
    if not in_volume:
        filters: Dict[str, float] = {}
    else:
        filters = dict(DEFAULT_REPORTING_VOLUME_M)
    for key, value in zip(VOLUME_FILTER_PARAMS, (max_r_m, max_i_m, max_c_m)):
        if value is not None:
            filters[key] = float(value)
    return filters


def _raw_cdms_in_risk_order(
    primary_norad: int,
    client: Optional[LeoLabsClient],
    registry: Optional[AssetRegistry],
    now: datetime,
    lookback_days: int,
    lookahead_days: int,
    volume_filters: Dict[str, float],
) -> tuple[str, List[Dict[str, Any]], int]:
    """Fetch the in-volume CDMs for an asset, risk-ordered, one event each.

    Sorting and deduping happen on the RAW CDMs, before any parsing, which is the
    whole point: _risk_key reads COLLISION_PROBABILITY and the TCA straight off
    the message, and raw_event_key reads COMMENT_EVENT_ID, so the global ordering
    and the event dedupe both cost a dict lookup per CDM rather than a parse. Only
    the page that is actually returned gets parsed.

    Returns (catalog, cdms, raw_count), where raw_count is how many CDMs the
    window held before reissues were collapsed, so a response can show that
    reissues were collapsed rather than dropped. Raises LeoLabsRuntimeError for an
    unsubscribed asset, the same as the unpaged path.
    """
    client = client or get_client()
    registry = registry or get_registry(client)

    catalog = registry.leolabs_for_norad(int(primary_norad))
    if catalog is None:
        raise LeoLabsRuntimeError(
            f"NORAD {primary_norad} is not in the LeoLabs subscribed-objects "
            f"registry; it cannot be screened on this account."
        )

    min_tca, max_tca = conjunction_window(now, lookback_days, lookahead_days)
    cdms = client.search_conjunction_cdms(
        object1=catalog,
        min_tca=min_tca,
        max_tca=max_tca,
        cdm_source="LeoLabs",
        **volume_filters,
    )
    raw = list(cdms or [])
    ordered = dedupe_raw_by_event(sorted(raw, key=_risk_key))
    return catalog, ordered, len(raw)


def fetch_leolabs_conjunction_page(
    primary_norad: int,
    *,
    client: Optional[LeoLabsClient] = None,
    registry: Optional[AssetRegistry] = None,
    now: Optional[datetime] = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    lookahead_days: int = _DEFAULT_LOOKAHEAD_DAYS,
    page_size: int = DEFAULT_PAGE_SIZE,
    cursor: Optional[str] = None,
    volume_filters: Optional[Dict[str, float]] = None,
) -> ConjunctionPage:
    """One page of an asset's conjunctions, newest solution per event, worst first.

    SCRUM-445. The unpaged fetch_leolabs_conjunctions still exists and is still
    what the evaluate path uses, because a row selector has to resolve against the
    whole window no matter which page the operator clicked it on. This is the
    listing's fetch, and it differs in exactly two ways: it bounds the fetch with
    the RIC reporting volume, and it parses only the rows it returns.

    Why the cursor is ours and not LeoLabs'
    ---------------------------------------
    LeoLabs' own `token` cursor pages the API in LeoLabs' order, which is not Pc
    order. Paging on it would mean the worst conjunction in a window could land on
    page seven, and an operator reading the top of a triage table would not see
    it. Dedupe would break too, since reissues of one event can straddle an API
    page boundary. So the set is fetched whole -- cheap, because sorting and
    deduping read raw dict fields and nothing is parsed yet -- then ordered and
    deduped globally, and only then sliced. That buys a true worst-Pc-first page
    one, an exact event total, and a cursor that is simply an offset into that
    ordering.

    What this costs: every page request re-fetches the window from LeoLabs, since
    nothing is cached between requests. Measured on SWARM C (39453), the fetch is
    ~16-20 s of the ~17-25 s a page takes, comfortably inside the ui-to-planner
    60 s timeout that this story exists to stop overrunning, but not free. A
    short-TTL cache of the ordered set per query fingerprint would make page two
    onward nearly instant; deliberately not built here, since it adds cross-request
    state the story did not ask for.

    Raises LeoLabsCursorError for a cursor that does not belong to this query,
    LeoLabsRuntimeError for an unsubscribed asset, and propagates LeoLabsError
    subclasses on transport failures.
    """
    now = now or datetime.now(timezone.utc)
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    filters = dict(volume_filters or {})

    fingerprint = _query_fingerprint(
        primary_norad, lookback_days, lookahead_days, page_size, filters
    )
    offset = decode_cursor(cursor, fingerprint) if cursor else 0

    catalog, ordered, cdm_total = _raw_cdms_in_risk_order(
        primary_norad, client, registry, now, lookback_days,
        lookahead_days, filters,
    )
    total = len(ordered)

    # Parse only this page. A CDM that fails a guard is skipped exactly as the
    # unpaged path skips it, so a page can come back shorter than page_size
    # without that meaning the list ended.
    registry_for_parse = registry or get_registry(client or get_client())
    rows: List[ParsedLeoLabsCDM] = []
    global _last_fetch_utc
    for cdm in ordered[offset:offset + page_size]:
        try:
            our_id = registry_for_parse.resolve_our_catalog_id(cdm)
            rows.append(parse_leolabs_cdm(cdm, our_id))
        except (LeoLabsParseError, LookupError) as exc:
            log.info(
                "skipping LeoLabs CDM that failed parse/guards",
                extra={"event": "leolabs_cdm_skipped", "catalog": catalog,
                       "reason": str(exc)},
            )
            continue
    if rows:
        _last_fetch_utc = now.isoformat().replace("+00:00", "Z")

    next_offset = offset + page_size
    next_cursor = (
        encode_cursor(next_offset, fingerprint) if next_offset < total else None
    )

    log.info(
        "LeoLabs conjunction page fetched",
        extra={"event": "leolabs_page_fetched", "catalog": catalog,
               "primary_norad": int(primary_norad), "offset": offset,
               "page_size": page_size, "returned": len(rows), "total": total,
               "in_volume": bool(filters)},
    )
    return ConjunctionPage(
        rows=rows,
        next_cursor=next_cursor,
        total=total,
        cdm_total=cdm_total,
        offset=offset,
        page_size=page_size,
        volume_filters=filters,
    )


# ---------------------------------------------------------------------------
# Live state vectors for the globe (SCRUM-447)
# ---------------------------------------------------------------------------

# LeoLabs returns state vectors in metres and metres/second under
# frames.EME2000. Confirmed against the live API on 2026-09-23 rather than
# assumed: SWARM C came back with |r| = 6.79e6 and |v| = 7.66e3, which are only
# sane read as m and m/s. The globe and the shared propagator work in km, so
# everything crossing this boundary is divided by 1000 exactly once, here.
_M_PER_KM = 1000.0
STATE_FRAME = "EME2000"


class LeoLabsStateError(LeoLabsError):
    """An object's state vector was missing or unusable."""


def _state_vector_km(state: Dict[str, Any]) -> tuple[List[float], List[float]]:
    """Pull (r_km, v_km_s) out of one LeoLabs state record.

    Raises LeoLabsStateError rather than returning a half-built vector: a track
    drawn from a partial state is worse than no track, because it looks real.
    """
    frames = (state or {}).get("frames") or {}
    frame = frames.get(STATE_FRAME) or {}
    position = frame.get("position")
    velocity = frame.get("velocity")
    if not (isinstance(position, (list, tuple)) and len(position) == 3):
        raise LeoLabsStateError(f"state has no 3-component {STATE_FRAME} position")
    if not (isinstance(velocity, (list, tuple)) and len(velocity) == 3):
        raise LeoLabsStateError(f"state has no 3-component {STATE_FRAME} velocity")
    try:
        r_km = [float(c) / _M_PER_KM for c in position]
        v_km_s = [float(c) / _M_PER_KM for c in velocity]
    except (TypeError, ValueError) as exc:
        raise LeoLabsStateError(f"state vector is not numeric: {exc}") from exc
    if not any(r_km):
        raise LeoLabsStateError("state position is the origin")
    return r_km, v_km_s


def latest_state_km(
    catalog_number: str,
    *,
    client: Optional[LeoLabsClient] = None,
) -> tuple[List[float], List[float], Optional[str]]:
    """The object's latest state as (r_km, v_km_s, epoch_utc).

    `states` comes back as a list newest-first; [0] is the latest. Raises
    LeoLabsStateError when the list is empty or the record is unusable, and
    propagates LeoLabsError subclasses (including the 403 an unsubscribed object
    returns) untouched so a caller can tell "no state" from "not allowed".
    """
    client = client or get_client()
    data = client.get_states(str(catalog_number), latest=True)
    states = (data or {}).get("states") if isinstance(data, dict) else None
    if not states:
        raise LeoLabsStateError(f"no states returned for {catalog_number}")
    record = states[0]
    r_km, v_km_s = _state_vector_km(record)
    return r_km, v_km_s, record.get("timestamp")


def catalog_for_norad(
    primary_norad: int,
    *,
    client: Optional[LeoLabsClient] = None,
    registry: Optional[AssetRegistry] = None,
) -> str:
    """Our NORAD id to its LeoLabs catalog number, or LeoLabsRuntimeError.

    get_states is keyed on the catalog number, not the NORAD id -- passing the
    NORAD is a 404 from the API, which is how the SCRUM-447 confirm-live step
    started. Callers go through this so that mistake cannot be made twice.
    """
    client = client or get_client()
    registry = registry or get_registry(client)
    catalog = registry.leolabs_for_norad(int(primary_norad))
    if catalog is None:
        raise LeoLabsRuntimeError(
            f"NORAD {primary_norad} is not in the LeoLabs subscribed-objects "
            f"registry; it cannot be screened on this account."
        )
    return catalog


def fetch_leolabs_conjunction(
    primary_norad: int,
    *,
    client: Optional[LeoLabsClient] = None,
    registry: Optional[AssetRegistry] = None,
    now: Optional[datetime] = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    lookahead_days: int = _DEFAULT_LOOKAHEAD_DAYS,
) -> Optional[ParsedLeoLabsCDM]:
    """Fetch and parse the highest-risk LeoLabs CDM for a subscribed sat.

    The first item of fetch_leolabs_conjunctions, and nothing more. Unchanged
    behaviour from SCRUM-412: same window, same ordering, same skipping of
    guard-failing CDMs, and still stops parsing as soon as one CDM is scorable.

    Returns None when there are no scorable CDMs in the window. Raises
    LeoLabsRuntimeError when the primary is not in our subscribed registry, and
    propagates LeoLabsError subclasses on transport/auth failures.
    """
    now = now or datetime.now(timezone.utc)
    parsed = next(
        _scorable_in_risk_order(
            primary_norad, client, registry, now, lookback_days, lookahead_days
        ),
        None,
    )
    if parsed is not None:
        log.info(
            "LeoLabs conjunction selected",
            extra={"event": "leolabs_conjunction_selected",
                   "primary_norad": primary_norad,
                   "cdm_id": parsed.provenance.get("cdm_id"),
                   "event_id": parsed.provenance.get("event_id")},
        )
    return parsed


# ---------------------------------------------------------------------------
# Status / credential probe (mirrors udl_client.get_credential_validity)
# ---------------------------------------------------------------------------

def _check_credentials_live() -> Dict[str, Any]:
    checked_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        get_client().ping()
        return {"status": "valid", "checked_at_utc": checked_at}
    except LeoLabsAuthError:
        return {"status": "invalid", "checked_at_utc": checked_at}
    except Exception as exc:  # network, 5xx, etc.
        log.warning(
            "LeoLabs credential probe unreachable",
            extra={"event": "leolabs_probe_unreachable", "exc": str(exc)},
        )
        return {"status": "unreachable", "checked_at_utc": checked_at}


def get_credential_validity() -> Dict[str, Any]:
    """Throttled credential probe; reuses a cached result within the interval."""
    global _last_probe_monotonic, _last_probe_result
    now = time.monotonic()
    if _last_probe_result is not None and _last_probe_monotonic is not None:
        if now - _last_probe_monotonic < _PROBE_INTERVAL_SECONDS:
            return _last_probe_result
    result = _check_credentials_live()
    _last_probe_result = result
    _last_probe_monotonic = now
    return result


def get_status() -> Dict[str, Any]:
    """Status for the /leolabs-status endpoint and the dashboard idle badge.

    Mirrors /udl-status. Unlike UDL, LeoLabs -- when enabled and valid -- genuinely
    drives planner conjunctions, so the live-and-valid mode is "live", which the
    dashboard shows as a green LIVE badge.
    """
    credentials_set = bool(
        os.environ.get("LEOLABS_ACCESS_KEY") and os.environ.get("LEOLABS_SECRET_KEY")
    )

    credential_valid = False
    credential_check_status = "not_checked"
    last_credential_check_utc = None
    if LEOLABS_ENABLED and credentials_set:
        probe = get_credential_validity()
        credential_check_status = probe["status"]
        credential_valid = probe["status"] == "valid"
        last_credential_check_utc = probe["checked_at_utc"]

    if LEOLABS_ENABLED and credentials_set and credential_valid:
        mode, label = "live", "LEOLABS LIVE"
        note = "LeoLabs enabled and authenticated; driving planner conjunctions."
    elif LEOLABS_ENABLED and credentials_set and credential_check_status == "invalid":
        mode, label = "invalid", "LEOLABS CREDENTIALS INVALID"
        note = "Credentials are set but did not authenticate on the last check."
    elif LEOLABS_ENABLED and credentials_set and credential_check_status == "unreachable":
        mode, label = "unconfirmed", "LEOLABS UNCONFIRMED"
        note = "Could not reach LeoLabs to confirm credential validity."
    elif LEOLABS_ENABLED and not credentials_set:
        mode, label = "misconfigured", "LEOLABS MISCONFIGURED"
        note = "LEOLABS_ENABLED=true but LEOLABS_ACCESS_KEY or LEOLABS_SECRET_KEY is missing."
    else:
        mode, label = "disabled", "LEOLABS DISABLED"
        note = "LEOLABS_ENABLED=false."

    return {
        "enabled": LEOLABS_ENABLED,
        "credentials_set": credentials_set,
        "credential_valid": credential_valid,
        "credential_check_status": credential_check_status,
        "last_credential_check_utc": last_credential_check_utc,
        "mode": mode,
        "label": label,
        "note": note,
        "last_fetch_utc": _last_fetch_utc,
    }
