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

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

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
    # SCRUM-442: how wide the request asks LeoLabs to look. Separate from
    # max_miss_distance_km, which is the action floor the verdict is decided
    # against locally. Screen wide, decide narrow.
    screening_volume_km: float = 50.0


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
        screening_volume_km=float(policy.screening_volume_km),
    )


# ---------------------------------------------------------------------------
# The request body
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScreeningRequest:
    """A create-screening request, ready to post as multipart form-data.

    Two parts, because the endpoint takes two: `files` carries the ephemeris
    document as an uploaded file, `data` carries the flat form fields. Kept as
    one object so the builder can be unit-tested without a client and so
    run_screening hands the client exactly what it assembled.
    """

    files: Dict[str, Any]
    data: Dict[str, Any]

    @property
    def ephemeris_bytes(self) -> bytes:
        """The uploaded document, for tests and for logging its size."""
        return self.files["file"][1]


EPHEMERIS_FILENAME = "ephemeris.json"
EPHEMERIS_CONTENT_TYPE = "application/json"


def build_screening_request(
    ephemeris: Dict[str, Any],
    thresholds: ScreeningThresholds,
    primary_object: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> ScreeningRequest:
    """The multipart create-screening request.

    Every LeoLabs field name lives in this one function, deliberately.

    Transport (SCRUM-441 fix)
    ------------------------
    This endpoint is multipart form-data with a file upload, not JSON --
    "Parameters must be sent as a multipart form" in the LeoLabs reference. Our
    first live submit posted JSON and was rejected with
    `422 {"error": "Invalid Miss Distance"}` on all three probes, including one
    with every threshold removed. That was not a field-name problem: the form
    parser saw no fields at all and failed on the first required one. Hence a
    file part plus flat form fields rather than a nested body.

    The fields, and their units
    ---------------------------
      file                    the SCRUM-440 ephemeris, JSON bytes, uploaded.
      primaryCatalogNumber    the LeoLabs catalog number, e.g. L3969.
      missDistance            KILOMETRES, max 100. The WIDE screening volume
                              (policy.screening_volume_km, default 50), not the
                              action floor. No conversion: this one field is km
                              while the rest of the API is metres.
      primaryHardBodyRadius   METRES. See the open decision below.

    Screen wide, decide narrow (SCRUM-442)
    --------------------------------------
    probabilityOfCollision and mahalanobisDistance are deliberately NOT sent.
    They would act as server-side filters, and our clear contract is an OR of Pc,
    miss distance and Mahalanobis -- so a server-side Pc floor would silently drop
    events that breach on miss or Mahalanobis alone, and the guard would never see
    them. Full recall inside the volume, verdict computed locally by
    evaluate_clear_contract. This also retires the mahalanobisDistance request
    field, whose meaning SCRUM-441 could never confirm: we no longer send it.

    missDistance is the one filter kept server-side, because it is the volume
    itself rather than a decision threshold -- something has to bound the search.
    It is set wide so the bounding does not pre-empt the verdict.

    What is omitted, and why omission is the instruction
    ----------------------------------------------------
    There is no screenAgainstAllObjects flag and no useFileUncertainty flag. Both
    behaviours are selected by leaving fields out, which means an accidental
    addition silently changes the screen rather than erroring:

      secondaryCatalogNumber, secondaryFile
          omitted -> screen against the whole catalog, which is the wide
          secondary-screen volume SCRUM-433 wants.
      radialUncertainty, inTrackUncertainty, crossTrackUncertainty
          omitted -> the covariance in our uploaded file is what gets screened,
          rather than a LeoLabs default. SCRUM-440's covariance floor is what it
          is, and substituting a default would hide that.

    Deliberately never sent: the deprecated combined `hardBodyRadius`,
    `primaryObject`, an inline `ephemeris`, `screenAgainstAllObjects`,
    `useFileUncertainty`, or a nested `thresholds` object. None exists in the API.
    The offline tests assert each of those is absent.

    Two open decisions, interim defaults only -- for John or Sreejit
    ---------------------------------------------------------------
    1. Hard body radius. Our convention is one inflated 15 m *combined* value; the
       API takes primaryHardBodyRadius and secondaryHardBodyRadius separately.
       Interim: put the 15 m combined value in primaryHardBodyRadius and omit the
       secondary, so each catalog object keeps its own radius. The effective
       combined radius per event is then larger than 15 m, which raises Pc and
       returns more conjunctions -- the safe direction for a screen, but it does
       over-count. The true split is a physics call.
    2. mahalanobisDistance. The reference calls it "Supplied Mahalanobis distance
       for ephemerides file", which does not clearly match our max-Mahalanobis
       result filter. Interim: send our 4.0 threshold. The corrected live submit
       accepted it without a 422, so the contingency of dropping it was not
       needed -- but acceptance is not confirmation of meaning, and it remains a
       candidate contributor to the 0-conjunction result below.

    One more thing the live submit revealed, which the field name hides:
    missDistance is expanded by LeoLabs into a per-axis box. Sending 1.0 comes
    back echoed as missDistanceR/I/C all "1.0", i.e. a 1 x 1 x 1 km RIC box --
    not a 1 km sphere, and far tighter than the 2 x 50 x 50 km volume LeoLabs
    uses for its own reporting. policy.min_miss_distance_km is our *action*
    floor, the distance at which a maneuver is required; using it here also makes
    it the distance we look out to, which are not the same question. Recorded in
    docs/scrum-441/live_verification.md.

    `extra` merges into the form fields last, so a correction needs no edit here.
    """
    if not isinstance(ephemeris, dict) or not ephemeris.get("states"):
        raise LeoLabsScreeningError(
            "ephemeris must be a built SCRUM-440 document with at least one state"
        )
    if not primary_object:
        raise LeoLabsScreeningError("primary_object is required")

    payload = json.dumps(ephemeris).encode("utf-8")

    data: Dict[str, Any] = {
        "primaryCatalogNumber": str(primary_object),
        # SCRUM-442: the WIDE screening volume, in kilometres -- not the action
        # floor. See the note above on screening wide and deciding narrow.
        "missDistance": thresholds.screening_volume_km,
        # Metres. The 15 m combined value, per the interim decision above.
        "primaryHardBodyRadius": thresholds.combined_hbr_m,
    }
    if extra:
        data.update(extra)

    return ScreeningRequest(
        files={"file": (EPHEMERIS_FILENAME, payload, EPHEMERIS_CONTENT_TYPE)},
        data=data,
    )


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


@dataclass(frozen=True)
class ContractVerdict:
    """The SCRUM-381 clear contract applied to a screening result.

    `clear` is the boolean guard_secondary_clear ultimately reads. `breaches`
    records why each breaching event breached, so a NOT CLEAR verdict can be
    audited against the exact events and limbs it was made on rather than just
    asserted.
    """

    clear: bool
    evaluated: int
    breaches: List[Dict[str, Any]] = field(default_factory=list)
    closest_miss_km: Optional[float] = None
    closest_object_id: Optional[str] = None
    max_pc: Optional[float] = None
    flagged_objects: List[str] = field(default_factory=list)


def _mahalanobis_distance(parsed: ParsedLeoLabsCDM) -> Optional[float]:
    """Mahalanobis distance of the relative position, or None if not computable.

    Uses the same mahalanobis_sq the scorer uses, on the parsed CDM's own
    relative position and combined covariance, rather than a second formula.
    """
    try:
        from avoid.decision_model import mahalanobis_sq

        m2 = mahalanobis_sq(parsed.r_rel_km(), parsed.p_rel_eci_km2())
        if not np.isfinite(m2) or m2 < 0:
            return None
        return float(np.sqrt(m2))
    except Exception:
        return None


def evaluate_clear_contract(
    conjunctions: Sequence[ParsedLeoLabsCDM], policy: Any
) -> ContractVerdict:
    """Judge screened conjunctions against the SCRUM-381 clear contract.

    Decide narrow: the screen was submitted wide, and the verdict is made here
    against the operator policy's own thresholds. The two predicates are the
    policy's existing ones, not reimplementations --

        policy.is_maneuver_required(pc, miss_km)
            Pc >= pc_maneuver_threshold OR miss < min_miss_distance_km
        policy.passes_pre_screen(mahalanobis)
            mahalanobis <= mahalanobis_screen_threshold, i.e. inside the
            risk-relevant radius

    -- so the on-demand screen and the internal one cannot drift apart about what
    "clear" means. An event breaches if either holds. Note the plan describes the
    miss limb as "at or below" while is_maneuver_required uses a strict `<`; the
    existing implementation is used as-is rather than being adjusted to the prose,
    because one of them has been in the decision path and the other has not.

    An absent value never breaches on its own limb -- a CDM with no Pc cannot
    breach the Pc limb -- but it does not excuse the event either: the other limbs
    still apply, and the Mahalanobis limb is precisely the fallback for an event
    with no Pc.

    An empty sequence is CLEAR. That is the honest reading of a screen that ran
    and found nothing, and it is only safe because every way of *failing* to run
    raises instead of returning empty.
    """
    breaches: List[Dict[str, Any]] = []
    flagged: List[str] = []
    closest_miss_km: Optional[float] = None
    closest_object_id: Optional[str] = None
    max_pc: Optional[float] = None

    for parsed in conjunctions:
        obj_id = (parsed.secondary.object_name
                  or parsed.secondary.designator
                  or str(parsed.secondary.norad_id))
        pc = parsed.cdm_collision_probability
        miss_km = (parsed.miss_distance_m / 1000.0
                   if parsed.miss_distance_m is not None else None)
        mahalanobis = _mahalanobis_distance(parsed)

        if miss_km is not None and (closest_miss_km is None
                                    or miss_km < closest_miss_km):
            closest_miss_km = miss_km
            closest_object_id = obj_id
        if pc is not None and (max_pc is None or pc > max_pc):
            max_pc = float(pc)

        limbs: List[str] = []
        # Pc and miss share one predicate, so they are asked together and the
        # limb is attributed afterwards for the audit trail.
        if pc is not None or miss_km is not None:
            if policy.is_maneuver_required(
                float(pc) if pc is not None else 0.0,
                float(miss_km) if miss_km is not None else float("inf"),
            ):
                if pc is not None and pc >= policy.pc_maneuver_threshold:
                    limbs.append("pc")
                if miss_km is not None and miss_km < policy.min_miss_distance_km:
                    limbs.append("miss_distance")
        if mahalanobis is not None and policy.passes_pre_screen(mahalanobis):
            limbs.append("mahalanobis")

        if limbs:
            flagged.append(obj_id)
            breaches.append({
                "object_id": obj_id,
                "secondary_norad": parsed.secondary.norad_id,
                "cdm_id": parsed.provenance.get("cdm_id"),
                "event_id": parsed.provenance.get("event_id"),
                "tca_utc": parsed.t_ca_utc,
                "pc": pc,
                "miss_distance_km": miss_km,
                "mahalanobis": mahalanobis,
                "limbs": limbs,
            })

    return ContractVerdict(
        clear=not breaches,
        evaluated=len(conjunctions),
        breaches=breaches,
        closest_miss_km=closest_miss_km,
        closest_object_id=closest_object_id,
        max_pc=max_pc,
        flagged_objects=flagged,
    )


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

    request = build_screening_request(
        ephemeris, thresholds, primary_object, extra=extra_body
    )

    # -- create ----------------------------------------------------------
    # Multipart: the ephemeris as a file part, the thresholds as form fields.
    # The client's screening limiter, org limiter and backoff are unchanged.
    try:
        created = client.create_screening(files=request.files, data=request.data)
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
