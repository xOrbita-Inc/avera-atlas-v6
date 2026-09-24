"""services/planner/common/leolabs_screening.py

SCRUM-441: submit a post-burn ephemeris for on-demand screening, poll it,
retrieve the results, and parse them into our covariance-bearing conjunction
model.

This exposes a clean call and stops. Wiring the result into
guard_secondary_clear and the maneuver decision is SCRUM-442; nothing here
touches the guard, the decision path, the SCRUM-440 builder or the internal
secondary_horizon screen.

Composition, not new transport
------------------------------
The whole lifecycle already exists on LeoLabsClient -- create_screening,
get_screening_status, get_screening_cdms (paginated through the SCRUM-438 token
path) and wait_for_screening (jittered poll, 3-creates-per-2-minutes limiter on
top of the org-wide 4 req/s). Results come back as CDMs on
/screenings/{id}/cdms, so leolabs_cdm_parser.parse_leolabs_cdm reads them the
same way it reads the live conjunction list. None of that is reimplemented here.

Failing closed
--------------
The single most important property in this module: a screen that could not run
must never look like a screen that found nothing. An empty result is a real and
unremarkable clear sky and returns an empty set. A timeout, a transport failure,
a rate-limit refusal, missing credentials or no on-demand access raise
LeoLabsScreeningError, so the caller in SCRUM-442 can fail closed on it. There is
no path through this module that turns a failure into an empty result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from aps_math import conventions
from common.leolabs_asset_map import AssetRegistry
from common.leolabs_client import (
    LeoLabsAuthError,
    LeoLabsClient,
    LeoLabsError,
    LeoLabsHTTPError,
)
from common.leolabs_cdm_parser import LeoLabsParseError, ParsedLeoLabsCDM, parse_leolabs_cdm
from common.leolabs_conjunction_list import dedupe_by_event
from common.leolabs_runtime import LEOLABS_ENABLED, get_client, get_registry

log = logging.getLogger("planner")

# HTTP 429 is the rate limiter; 402/403 is how an account without on-demand
# screening access is refused. Both are worth naming separately from a generic
# transport failure, because the operator response differs: wait, versus this
# account cannot do this at all.
_RATE_LIMIT_STATUS = 429
_NO_ACCESS_STATUS = frozenset({402, 403})


class LeoLabsScreeningError(LeoLabsError):
    """An on-demand screening could not be completed.

    Deliberately distinct from "the screening ran and found nothing". A caller
    fails closed on this; it must never be collapsed into an empty result.
    """


class LeoLabsScreeningUnavailable(LeoLabsScreeningError):
    """This account cannot run on-demand screenings, or is being rate-limited."""


class LeoLabsScreeningTimeout(LeoLabsScreeningError):
    """The screening did not reach a terminal status inside the poll budget."""


# ---------------------------------------------------------------------------
# The clear contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScreeningThresholds:
    """What we ask LeoLabs to report, read from our own clear contract.

    Every value here comes from OperatorPolicy or aps_math.conventions. None of
    them is invented in this module, because a screening threshold invented here
    would silently change which conjunctions come back -- and for a safety
    screen, a threshold that is too tight returns fewer conjunctions and reads as
    a clearer sky than the truth.

    min_probability_of_collision
        policy.pc_maneuver_threshold. The action line: at or above it, section
        4.2's secondary screen is not clear.
    max_miss_distance_km
        policy.min_miss_distance_km. The hard floor on acceptable miss distance;
        is_maneuver_required treats Pc OR miss as sufficient on its own.
    max_mahalanobis
        policy.mahalanobis_screen_threshold. Above it, an event is outside the
        risk-relevant radius.
    combined_hbr_m
        conventions.combined_hbr_m(). 15 m by screening convention, deliberately
        inflated to absorb attitude. Explicitly NOT configurable per that module:
        lowering it would reduce every Pc and suppress maneuvers.
    """

    min_probability_of_collision: float
    max_miss_distance_km: float
    max_mahalanobis: float
    combined_hbr_m: float
    hbr_source: str = "screening_convention"


def thresholds_from_policy(policy: Any, primary_radius_m: float) -> ScreeningThresholds:
    """Build the screening thresholds from the operator policy in force.

    One place, so the screen we ask LeoLabs for and the screen we judge the
    result by cannot drift apart.
    """
    hbr_m, hbr_source = conventions.combined_hbr_m(float(primary_radius_m))
    return ScreeningThresholds(
        min_probability_of_collision=float(policy.pc_maneuver_threshold),
        max_miss_distance_km=float(policy.min_miss_distance_km),
        max_mahalanobis=float(policy.mahalanobis_screen_threshold),
        combined_hbr_m=float(hbr_m),
        hbr_source=str(hbr_source),
    )


# ---------------------------------------------------------------------------
# The request body
# ---------------------------------------------------------------------------

def build_screening_request(
    ephemeris: Dict[str, Any],
    thresholds: ScreeningThresholds,
    primary_object: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The create_screening body.

    Every LeoLabs JSON field name lives in this one function, deliberately. These
    names are the open item the SCRUM-439 spike left: the lifecycle is confirmed
    but the request-body field names are not, and they are confirmed on the first
    live submit. Keeping them here means that confirmation is a change to one
    function rather than a hunt.

    What is being asked for:

      screenAgainstAllObjects  the full LeoLabs catalog, which is the wide
                               secondary-screen volume SCRUM-433 wants -- not
                               just our subscribed vehicles.
      useFileUncertainty       screen against the covariance in the uploaded
                               ephemeris, not a LeoLabs default. Our uploaded
                               covariance is the one that should decide the
                               result, and SCRUM-440's covariance floor is what
                               it is -- substituting a default would hide that.
      thresholds               our clear contract, from thresholds_from_policy.
      hardBodyRadius           the 15 m combined override.

    `extra` merges last, so the first live submit can add or correct a field
    without editing this signature.

    STATUS OF THESE FIELD NAMES: NOT CONFIRMED, and the first live submit did not
    confirm them -- it rejected the body. On 2026-09-24 three creates against the
    real API all returned

        HTTP 422 {"error": "Invalid Miss Distance"}

    including one with the `thresholds` block removed entirely and one adding a
    top-level `missDistance` in metres. So a required miss-distance parameter is
    expected under a name not in this body, and it is not derivable by probing a
    rate-limited production endpoint. It needs LeoLabs' API documentation or a
    question to Lois Reid. Full trace in docs/scrum-441/live_verification.md.

    Everything above the thresholds is therefore also unconfirmed. The names are
    isolated in this one function, and `extra` merges last, so correcting them is
    a one-line change and needs no edit here once the real schema is known.

    Also unconfirmed, and separate: whether LeoLabs combines the thresholds as AND
    or OR. Our own contract ORs them -- is_maneuver_required treats Pc or miss as
    sufficient alone -- so an AND would make the screen narrower than our contract
    and return fewer conjunctions than intended. That is the dangerous direction
    for a safety screen.
    """
    if not isinstance(ephemeris, dict) or not ephemeris.get("states"):
        raise LeoLabsScreeningError(
            "ephemeris must be a built SCRUM-440 document with at least one state"
        )
    if not primary_object:
        raise LeoLabsScreeningError("primary_object is required")

    body: Dict[str, Any] = {
        "primaryObject": str(primary_object),
        "ephemeris": ephemeris,
        # Screen Against All Objects: the full catalog, not the subscription.
        "screenAgainstAllObjects": True,
        # Use File Uncertainty: our covariance decides, not a LeoLabs default.
        "useFileUncertainty": True,
        "thresholds": {
            "minProbabilityOfCollision": thresholds.min_probability_of_collision,
            "maxMissDistanceKm": thresholds.max_miss_distance_km,
            "maxMahalanobisDistance": thresholds.max_mahalanobis,
        },
        "hardBodyRadius": thresholds.combined_hbr_m,
    }
    if extra:
        body.update(extra)
    return body


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScreeningResult:
    """One completed on-demand screening.

    `conjunctions` is one ParsedLeoLabsCDM per conjunction event, covariance
    included, deduped the same way the live listing dedupes. Empty means the
    screen ran and found nothing -- a clear sky. It never means the screen did
    not run; that raises.
    """

    screening_id: str
    conjunctions: List[ParsedLeoLabsCDM]
    cdm_count: int
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    status: Optional[Dict[str, Any]] = None

    @property
    def is_clear(self) -> bool:
        """No conjunctions came back. Only meaningful because failures raise."""
        return not self.conjunctions


def _screening_id(created: Any) -> str:
    """The id of the screening just created, whatever LeoLabs called the field.

    Another first-live-submit unknown, so several spellings are accepted rather
    than guessing one and failing on the others. If none matches, that is a hard
    error: without an id there is nothing to poll, and proceeding would produce a
    result that describes no screening.
    """
    if isinstance(created, str) and created:
        return created
    if isinstance(created, dict):
        for key in ("id", "screeningId", "screening_id"):
            value = created.get(key)
            if value not in (None, ""):
                return str(value)
        for container in ("screenings", "data"):
            items = created.get(container)
            if isinstance(items, list) and items:
                return _screening_id(items[0])
    raise LeoLabsScreeningError(
        f"screening create returned no recognisable id: {str(created)[:200]}"
    )


def _classify(exc: Exception, what: str) -> LeoLabsScreeningError:
    """Map a transport failure onto the typed failure the caller fails closed on."""
    if isinstance(exc, LeoLabsAuthError):
        return LeoLabsScreeningUnavailable(
            f"{what}: LeoLabs rejected the credentials ({exc})"
        )
    if isinstance(exc, LeoLabsHTTPError):
        status = getattr(exc, "status", None)
        if status == _RATE_LIMIT_STATUS:
            return LeoLabsScreeningUnavailable(
                f"{what}: rate-limited by LeoLabs (HTTP 429); the screening was "
                f"not run"
            )
        if status in _NO_ACCESS_STATUS:
            return LeoLabsScreeningUnavailable(
                f"{what}: this account has no on-demand screening access "
                f"(HTTP {status})"
            )
        return LeoLabsScreeningError(f"{what}: {exc}")
    return LeoLabsScreeningError(f"{what}: {exc}")


def run_screening(
    ephemeris: Dict[str, Any],
    thresholds: ScreeningThresholds,
    primary_object: str,
    *,
    client: Optional[LeoLabsClient] = None,
    registry: Optional[AssetRegistry] = None,
    timeout_s: Optional[float] = None,
    is_complete: Optional[Callable[[Dict[str, Any]], bool]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> ScreeningResult:
    """Submit, poll, retrieve and parse one on-demand screening.

    Returns a ScreeningResult whose conjunctions carry real EME2000 covariance.
    An empty conjunction list is a clear sky.

    Raises LeoLabsScreeningError (or its Unavailable/Timeout subclasses) when the
    screening could not be completed. Never returns empty to signal failure: the
    caller's whole reason for calling is to decide whether a burn is safe, and
    "we could not look" and "we looked and it is clear" must not be the same
    return value.

    is_complete is passed straight through to wait_for_screening so the terminal
    status vocabulary is injectable. The client's default accepts
    completed/complete/done/finished; the real enum is confirmed on the first
    live submit and, until then, is not hard-wired here.
    """
    if not LEOLABS_ENABLED:
        raise LeoLabsScreeningUnavailable(
            "LEOLABS_ENABLED=false; on-demand screening is unavailable"
        )

    try:
        client = client or get_client()
    except Exception as exc:
        raise _classify(exc, "on-demand screening could not build a client") from exc

    body = build_screening_request(
        ephemeris, thresholds, primary_object, extra=extra_body
    )

    # -- create ----------------------------------------------------------
    try:
        created = client.create_screening(body)
    except Exception as exc:
        raise _classify(exc, "on-demand screening create failed") from exc
    screening_id = _screening_id(created)
    log.info(
        "on-demand screening created",
        extra={"event": "leolabs_screening_created",
               "screening_id": screening_id,
               "primary_object": str(primary_object),
               "states": len(ephemeris.get("states") or [])},
    )

    # -- poll ------------------------------------------------------------
    wait_kwargs: Dict[str, Any] = {}
    if timeout_s is not None:
        wait_kwargs["timeout_s"] = float(timeout_s)
    if is_complete is not None:
        wait_kwargs["is_complete"] = is_complete
    try:
        status = client.wait_for_screening(screening_id, **wait_kwargs)
    except LeoLabsError as exc:
        # wait_for_screening raises a bare LeoLabsError on timeout. That is a
        # screen that did not run, not a clear one.
        if "did not complete" in str(exc):
            raise LeoLabsScreeningTimeout(
                f"on-demand screening {screening_id} did not reach a terminal "
                f"status in time: {exc}"
            ) from exc
        raise _classify(exc, f"on-demand screening {screening_id} poll failed") from exc
    except Exception as exc:
        raise _classify(exc, f"on-demand screening {screening_id} poll failed") from exc

    # -- retrieve --------------------------------------------------------
    try:
        cdms = client.get_screening_cdms(screening_id) or []
    except Exception as exc:
        raise _classify(
            exc, f"on-demand screening {screening_id} result retrieval failed"
        ) from exc

    # A retrieved-vs-total count mismatch is logged as a warning inside the
    # client's paginator and stays a warning: the live catalog moves under a long
    # pull, and SCRUM-439 found the warning is correct behaviour, not a fault.
    # Turning it into an error here would fail screens that are actually fine.

    # -- parse -----------------------------------------------------------
    registry = registry or get_registry(client)
    parsed: List[ParsedLeoLabsCDM] = []
    skipped: List[Dict[str, Any]] = []
    for cdm in cdms:
        try:
            our_id = _our_id_for(cdm, registry, primary_object)
            parsed.append(parse_leolabs_cdm(cdm, our_id))
        except (LeoLabsParseError, LookupError, ValueError) as exc:
            # One unparseable result CDM does not invalidate the screening, but
            # it is not silently dropped either: it is counted and returned so a
            # caller can see the set it is judging is incomplete.
            skipped.append({
                "cdm_id": cdm.get("COMMENT_ID"),
                "event_id": cdm.get("COMMENT_EVENT_ID"),
                "reason": str(exc),
            })
            log.info(
                "skipping unparseable screening result CDM",
                extra={"event": "leolabs_screening_cdm_skipped",
                       "screening_id": screening_id, "reason": str(exc)},
            )

    events = dedupe_by_event(parsed)
    log.info(
        "on-demand screening complete",
        extra={"event": "leolabs_screening_complete",
               "screening_id": screening_id, "cdms": len(cdms),
               "conjunctions": len(events), "skipped": len(skipped)},
    )
    return ScreeningResult(
        screening_id=screening_id,
        conjunctions=events,
        cdm_count=len(cdms),
        skipped=skipped,
        status=status if isinstance(status, dict) else None,
    )


def _our_id_for(
    cdm: Dict[str, Any], registry: AssetRegistry, primary_object: str
) -> str:
    """Which object in this result CDM is ours.

    The registry resolves it when the primary is one of our subscribed vehicles,
    which is the normal case. A screening of an uploaded ephemeris may instead
    name the primary by whatever we called it on submit, so that is the fallback
    rather than failing the parse -- but only when it actually matches one of the
    CDM's two objects, so this cannot mislabel someone else's object as ours.
    """
    try:
        return registry.resolve_our_catalog_id(cdm)
    except LookupError:
        wanted = str(primary_object)
        for sat_key in ("SAT1", "SAT2"):
            designator = cdm.get(f"{sat_key}_OBJECT_DESIGNATOR")
            if designator is not None and str(designator) == wanted:
                return str(designator)
        raise
