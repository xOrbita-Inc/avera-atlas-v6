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

import logging
import os
import threading
import time
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
