"""
AVERA-ATLAS Planner Service — FastAPI wrapper for decision_model.py
APS 2.5 / 9.6 + SCRUM-341 container hardening.

Endpoints
---------
GET  /health              Liveness probe -- service version and uptime
GET  /ready               Readiness probe -- service up, policy config loaded
POST /v1/evaluate         Single conjunction evaluation (APS 2.5, emits ATLASManeuverArtifact)
POST /v1/evaluate/batch   Batch conjunction evaluation (APS 2.4 path)

SCRUM-341 changes (relative to 9.6 server.py):
  - Structured JSON logging replacing basicConfig plain-text format.
    Format: {"time": "...", "level": "INFO", "service": "planner", "msg": "..."}
  - /ready endpoint: returns 200 when service is up and operator policy
    config is loadable. API-only readiness -- no disk volume check.
  - HTTP request logging middleware (method, path, status, elapsed_ms).
  - Startup log via lifespan.
  - No changes to /v1/evaluate, /v1/evaluate/batch, covariance adapter,
    audit write, or ATLASManeuverArtifact integration.

Service port: 8060 (per k8s/06-planner.yaml and service map)
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aps_math import frames
import requests as http_requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from avoid.decision_model import (
    error_response,
    evaluate_batch,
)
from common.maneuver_scorer import (
    evaluate_conjunction_v25, _policy_from_dict, aps_decision_state, aps_pc_usable,
)
from common.evidence_record import (
    EvidenceRecord, RecordType, canonical_json, build_decision_record,
    build_transition_record, GENESIS_HASH,
)
from common.atlas_artifact import build_atlas_artifact, DecisionLog
from common.monitor_adapter import evaluate_request as evaluate_decision_state_machine
from common.mode_persistence import build_mode_store
from common import gnc_client
from common.gnc_client import emit_gnc_command
from common.gnc_command import GNCCommandContext, build_gnc_command
from common.gnc_operator import (
    OperatorCommandStore,
    approval_ack,
    record_approval,
    record_veto,
    veto_ack,
)
from common.gnc_report import (
    assess_post_burn,
    build_report_ack,
    post_burn_covariance_km2,
)
from common.satellite_capability import SatelliteCapability
from common.logging_setup import build_logger, _POLICY_CONFIG_PATH, SERVICE_NAME, SERVICE_VERSION
from common.operator_policy import OperatorPolicy, CovarianceSurrogate
from common.udl_client import UDL_ENABLED, get_conjunctions, get_credential_validity
from common import leolabs_runtime
from common.leolabs_runtime import (
    LEOLABS_ENABLED,
    LeoLabsCursorError,
    LeoLabsRuntimeError,
    LeoLabsStateError,
    catalog_for_norad,
    object_covariance_block,
    fetch_leolabs_conjunction,
    fetch_leolabs_conjunction_by_cdm_id,
    fetch_leolabs_conjunction_page,
    fetch_leolabs_conjunctions,
    latest_state_km,
)
from aps_math.orbits import propagate_two_body
from common.leolabs_conjunction_list import (
    conjunction_row,
    dedupe_by_event,
    select_conjunction,
    selector_from_conjunction_block,
)
from common.leolabs_evaluate import build_evaluate_request, analyze_leolabs_portfolio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

log = build_logger()

# Last successful UDL fetch timestamp (UTC ISO-8601). Updated by /v1/evaluate
# when UDL_ENABLED=true and get_conjunctions() returns at least one record.
_udl_last_fetch_utc: str | None = None

# SCRUM-369: loaded once at startup (see lifespan()). Used by
# _fetch_cdm_covariance for the documented surrogate covariance when no
# real CDM/UDL covariance is available. Falls back to CovarianceSurrogate()
# defaults if the policy file fails to load, so a policy load failure
# degrades gracefully rather than crashing evaluate requests.
_operator_policy: OperatorPolicy | None = None

# ---------------------------------------------------------------------------
# Ingest service URL
# ---------------------------------------------------------------------------

def _ingest_url() -> str:
    return os.environ.get("INGEST_SERVICE_URL", "http://ingest:8000")


# ---------------------------------------------------------------------------
# Covariance adapter helpers (unchanged from 9.6)
# ---------------------------------------------------------------------------

def _rtn_to_eci_rotation(r_km: np.ndarray, v_km_s: np.ndarray) -> np.ndarray:
    """Build the 3x3 RTN->ECI rotation matrix from an ECI state vector.

    SCRUM-369 follow-up: guards against a zero or degenerate state vector
    (e.g. the [0,0,0] fallback used when satellite data is missing in the
    outermost exception handler), which would otherwise divide by zero.
    Falls back to the identity matrix (no rotation -- the surrogate ellipse
    stays aligned with ECI axes rather than correctly oriented to the real
    orbit) and logs a warning, rather than crashing the request.

    SCRUM-397: the rotation itself now comes from aps_math.frames, which is the
    one definition. This wrapper survives because the logging is a service
    concern rather than a library one. A library that decides how a service logs
    is a library nobody wants to share.

    ingest/cdm_to_conjunction.py still carries its own copy. The ingest image
    builds with context ./services/ingest, so it cannot COPY libs/aps_math
    without a build-context change, and dragging a third service's image build
    into a frame-correctness fix is the wrong trade. A test asserts the two
    agree to machine precision across a range of geometries, which catches
    divergence more reliably than an import would.
    """
    if frames.is_degenerate_state(r_km, v_km_s):
        log.warning(
            "degenerate satellite state vector, using identity rotation",
            extra={"event": "rtn_rotation_degenerate"},
        )
    return frames.rtn_to_eci_rotation(r_km, v_km_s)


def _surrogate_covariance(
    r_sat_km: list,
    v_sat_km_s: list,
) -> tuple[list, str, None]:
    """SCRUM-369: documented elliptical surrogate covariance, used when no
    real CDM/UDL covariance is available. Reads sigma values from the
    loaded operator policy (falls back to CovarianceSurrogate defaults
    if the policy failed to load at startup -- see lifespan()).

    Rotated from RTN into ECI using the same rotation as the real
    covariance path, so the ellipse is correctly oriented relative to
    the satellite's actual orbit, not just a flat matrix in ECI.
    """
    cs = (
        _operator_policy.covariance_surrogate
        if _operator_policy is not None
        else CovarianceSurrogate()
    )
    cov_rtn = np.diag([
        cs.radial_sigma_km ** 2,
        cs.along_track_sigma_km ** 2,
        cs.cross_track_sigma_km ** 2,
    ])
    r = np.array(r_sat_km, dtype=float)
    v = np.array(v_sat_km_s, dtype=float)
    rot = _rtn_to_eci_rotation(r, v)
    cov_eci = rot @ cov_rtn @ rot.T
    return cov_eci.flatten().tolist(), "surrogate_elliptical", None


def _fetch_cdm_covariance(
    primary_norad: str,
    secondary_norad: str,
    r_sat_km: list,
    v_sat_km_s: list,
) -> tuple[list, str, int | None, list | None]:
    """Fetch RTN covariance from the ingest service and rotate to ECI.

    Returns (p_rel_km2, covariance_source, cdm_record_id, primary_cov_km2).

    p_rel_km2 is the COMBINED covariance, flattened, which is what the scorer's
    Pc wants. primary_cov_km2 is the primary's OWN 3x3 block in ECI, or None.

    SCRUM-453: the primary block used to be unavailable here. Ingest computed
    c_primary and c_secondary separately and then returned only their sum, so the
    primary-only block was not on the wire -- exposing it was an ingest change,
    which is why SCRUM-454 (planner-only) could not use it and the stored path
    fell back to the combined stand-in. Ingest now returns
    covariance_primary_rtn, so this reads it and the screening seed on the stored
    and reference path can prefer the primary's own covariance exactly as the
    live path already does. It stays None when the field is absent -- older rows
    assembled before that change, or a surrogate -- and the caller falls back to
    the combined stand-in as before.

    The LeoLabs path does not call this function at all -- a test asserts that --
    and takes the primary's own covariance straight off the parsed CDM.

    SCRUM-369: always returns a real, usable covariance. Falls back to
    the documented elliptical surrogate (_surrogate_covariance) on any
    failure, including missing NORAD IDs -- previously, missing NORAD
    IDs meant this function was never called at all, silently leaving
    whatever covariance the caller had already supplied (the old
    hardcoded 100 m UI value). The caller no longer supplies a
    covariance, so this function is now the single source of truth,
    called unconditionally.
    """
    if not primary_norad or not secondary_norad:
        log.info(
            "using surrogate covariance",
            extra={"event": "surrogate_covariance", "reason": "primary_or_secondary_norad_missing"},
        )
        return (*_surrogate_covariance(r_sat_km, v_sat_km_s), None)

    try:
        url = f"{_ingest_url()}/cdm/{primary_norad}/{secondary_norad}"
        resp = http_requests.get(url, timeout=5.0)
    except Exception as exc:
        log.warning("covariance fetch failed", extra={"event": "covariance_fetch_fail", "reason": "unreachable", "pair": f"{primary_norad}/{secondary_norad}", "exc": str(exc)})
        return (*_surrogate_covariance(r_sat_km, v_sat_km_s), None)

    if resp.status_code == 404:
        log.warning("covariance fetch failed", extra={"event": "covariance_fetch_fail", "reason": "not_found", "pair": f"{primary_norad}/{secondary_norad}"})
        return (*_surrogate_covariance(r_sat_km, v_sat_km_s), None)

    if resp.status_code == 503:
        log.warning("covariance fetch failed", extra={"event": "covariance_fetch_fail", "reason": "store_unavailable", "pair": f"{primary_norad}/{secondary_norad}"})
        return (*_surrogate_covariance(r_sat_km, v_sat_km_s), None)

    if not resp.ok:
        log.warning("covariance fetch failed", extra={"event": "covariance_fetch_fail", "reason": f"http_{resp.status_code}", "pair": f"{primary_norad}/{secondary_norad}"})
        return (*_surrogate_covariance(r_sat_km, v_sat_km_s), None)

    try:
        data = resp.json()
        cov_rtn = np.array(data["covariance_combined_rtn"], dtype=float)
        covariance_source = data.get("covariance_source", "real")
        cdm_record_id = data.get("id")

        r = np.array(r_sat_km, dtype=float)
        v = np.array(v_sat_km_s, dtype=float)
        rot = _rtn_to_eci_rotation(r, v)
        cov_eci = rot @ cov_rtn @ rot.T
        p_rel_km2 = cov_eci.flatten().tolist()

        # SCRUM-453: the primary's own block, rotated with the SAME rotation as
        # the combined so the two are in one frame, and left in km^2 to match the
        # seed's units. Guarded separately from the combined parse on purpose: a
        # malformed or absent primary block must cost the screen its preferred
        # seed, not cost the decision its covariance.
        primary_cov_km2 = None
        _primary_rtn = data.get("covariance_primary_rtn")
        if _primary_rtn is not None:
            try:
                _m = np.array(_primary_rtn, dtype=float)
                if _m.shape == (3, 3):
                    primary_cov_km2 = [
                        [float(c) for c in row] for row in (rot @ _m @ rot.T)
                    ]
                else:
                    log.info(
                        "primary covariance ignored, unexpected shape",
                        extra={"event": "cdm_primary_covariance_bad_shape",
                               "shape": str(_m.shape)},
                    )
            except Exception as exc:
                log.info(
                    "primary covariance could not be read from the CDM response",
                    extra={"event": "cdm_primary_covariance_unavailable",
                           "exc": str(exc)},
                )

        log.info("covariance fetched", extra={"event": "covariance_fetched", "pair": f"{primary_norad}/{secondary_norad}", "source": covariance_source, "cdm_id": cdm_record_id, "primary_covariance": primary_cov_km2 is not None})
        return p_rel_km2, covariance_source, cdm_record_id, primary_cov_km2

    except Exception as exc:
        log.warning("covariance parse failed", extra={"event": "covariance_parse_fail", "pair": f"{primary_norad}/{secondary_norad}", "exc": str(exc)})
        return (*_surrogate_covariance(r_sat_km, v_sat_km_s), None)


# ---------------------------------------------------------------------------
# Per-evaluate source (SCRUM-411 AC6)
# ---------------------------------------------------------------------------

# Conjunction data sources that represent a genuine live feed, and so light the
# green LIVE badge in the dashboard (SCRUM-348 AC3). "surrogate" is not live.
_LIVE_SOURCES = frozenset({"leolabs", "udl", "spacetrack"})


def _resolve_evaluate_source(
    body: Dict[str, Any],
    covariance_source: str,
    udl_used: bool,
) -> str:
    """Resolve the per-evaluate conjunction data source (SCRUM-411 AC6).

    This is the source of the conjunction that drove the decision, distinct from
    covariance_source (the provenance of the covariance matrix). It is what the
    dashboard badge reflects per evaluate rather than a static label.

    Resolution order:
      1. An explicit source on the conjunction block (or body). This is how a
         LeoLabs-driven evaluate reports itself: the request assembled from a
         parsed LeoLabs CDM carries source="leolabs" (design section 7).
      2. UDL, when a UDL conjunction drove the evaluate.
      3. Otherwise derived from covariance_source: a real CDM came from
         Space-Track; a surrogate is not a live source.

    Returns one of: "leolabs", "udl", "spacetrack", "surrogate".
    """
    conj = body.get("conjunction", {}) or {}
    explicit = conj.get("source") or body.get("source")
    if explicit:
        val = str(explicit).strip().lower()
        if "leolabs" in val:
            return "leolabs"
        if "udl" in val:
            return "udl"
        if "space" in val:  # space-track / spacetrack / space_track
            return "spacetrack"
        return val

    if udl_used:
        return "udl"

    cs = (covariance_source or "").lower()
    if cs in ("real_cdm", "real"):
        return "spacetrack"
    if cs == "udl":
        return "udl"
    return "surrogate"


def _persist_screening_result(
    result, verdict, decision_log_id, p_post_source: Optional[str] = None
) -> Optional[str]:
    """SCRUM-442: store the on-demand screening a decision was made on.

    NOT YET LANDING. The ingest service has no /screening/persist endpoint and no
    table to hold this -- PlannerOutput is a fixed schema with no room for a
    screening reference, and adding one is an ingest change, while SCRUM-442 is
    scoped to the planner. So this posts, 404s, logs an audit failure and returns
    None. Verified live: the evaluate still returns its decision and
    screening_record_id comes back null.

    That is the correct *failure* behaviour but it is not the feature. What is
    still needed, and is flagged in docs/scrum-442/live_verification.md: an ingest
    endpoint and table mirroring cdm_records / PlannerOutput, after which this
    function works unchanged. Until then a NOT CLEAR verdict is auditable only
    through the operator note and the service logs, not through the store.

    Mirrors _persist_leolabs_cdm: persist-on-evaluate, guarded, and never allowed
    to affect the decision. Any failure logs and returns None, and the evaluate
    returns the decision it already made -- a decision that was made must still be
    returned even if we could not write down what it was made from.

    The record is the screening id, the verdict and its breaches, and each
    returned conjunction with its covariance, tied to the decision log id so the
    CLEAR or NOT CLEAR can be audited against the exact events behind it.
    """
    if result is None:
        return None
    try:
        conjunctions = []
        for parsed in result.conjunctions:
            entry = {
                "cdm_id": parsed.provenance.get("cdm_id"),
                "event_id": parsed.provenance.get("event_id"),
                "secondary_norad": parsed.secondary.norad_id,
                "secondary_designator": parsed.secondary.designator,
                "secondary_name": parsed.secondary.object_name,
                "tca_utc": parsed.t_ca_utc,
                "miss_distance_m": parsed.miss_distance_m,
                "pc": parsed.cdm_collision_probability,
                # The covariance the verdict was judged on, m^2, EME2000.
                "cov_eci_pos_m2": [
                    [float(c) for c in row]
                    for row in parsed.secondary.cov_eci_pos_m2
                ],
            }
            conjunctions.append(entry)

        record = {
            "decision_log_id": decision_log_id,
            "screening_id": result.screening_id,
            "source": "leolabs_on_demand",
            "cdm_count": result.cdm_count,
            "conjunction_count": len(result.conjunctions),
            "skipped": result.skipped,
            "clear": bool(verdict.clear) if verdict is not None else None,
            "breaches": list(verdict.breaches) if verdict is not None else [],
            "conjunctions": conjunctions,
            # SCRUM-452: the covariance is real -- an execution-error seed grown
            # along the trajectory by the J2 state transition matrix -- so the
            # record states its provenance rather than a caveat about it.
            "covariance_model": "execution_error_seed_grown_by_stm",
            # Which quantity seeded the position block. "gnc_post_burn_covariance"
            # is the primary's own; "combined_relative_stand_in" is the
            # over-estimate used when the call carried no GNC report. Recorded so
            # an audit can tell the two apart rather than inferring it.
            "p_post_source": p_post_source,
        }
    except Exception as exc:
        log.warning(
            "screening result could not be mapped for the store",
            extra={"event": "screening_map_failed", "exc": str(exc)},
        )
        return None

    try:
        resp = http_requests.post(
            f"{_ingest_url()}/screening/persist", json=record, timeout=5.0
        )
        if resp.status_code not in (200, 201):
            _note_audit_failure(
                "screening_result", str(result.screening_id),
                f"status {resp.status_code}",
            )
            return None
        log.info(
            "on-demand screening result persisted",
            extra={"event": "screening_persisted",
                   "screening_id": result.screening_id,
                   "decision_log_id": decision_log_id,
                   "conjunctions": len(result.conjunctions)},
        )
        return str(result.screening_id)
    except Exception as exc:
        _note_audit_failure("screening_result", str(result.screening_id), str(exc))
        return None


async def _persist_resolved_screen(entry) -> None:
    """SCRUM-484: write the screening record for a resolved async screen.

    Called from the poll endpoint, which is where both halves exist at once: the
    decision log id was stamped during the evaluate, and the worker has since
    written the result and verdict onto the same entry. The inline path keeps its
    own persist and is untouched.

    Runs the POST in a worker thread. The poll handler is `async def`, the service
    runs a single uvicorn worker, and _persist_screening_result does a blocking
    HTTP POST with a five second timeout -- doing that on the event loop would
    stall every other request behind a slow store, which is exactly the failure
    SCRUM-459 fixed elsewhere. The dashboard polls this endpoint repeatedly, so it
    is the last place that can afford to block. A short POST does not justify the
    capped _LIST_EXECUTOR, which exists to keep minutes-long window fetches off
    the shared pool, so this uses the default threadpool instead.

    Everything here is best-effort by design. Any failure is logged and swallowed:
    the caller reports the verdict from the entry regardless, so whether the
    record was written cannot change whether the screen reads clear.
    """
    try:
        if not entry.is_terminal:
            return                      # still running; nothing to write yet
        if entry.screening_record_id:
            return                      # already written by an earlier poll
        if entry.result is None or not entry.decision_log_id:
            # An errored screen that never produced a result has nothing to
            # record, and a screen with no decision log id cannot be tied to one.
            return

        from starlette.concurrency import run_in_threadpool
        from common.secondary_screen_async import set_screening_record_id

        record_id = await run_in_threadpool(
            _persist_screening_result,
            entry.result,
            entry.verdict,
            entry.decision_log_id,
            p_post_source=entry.seed_source,
        )
        if record_id:
            set_screening_record_id(entry.job_id, record_id)
            entry.screening_record_id = record_id
            log.info(
                "async screening record persisted on resolution",
                extra={"event": "async_screening_persisted",
                       "job_id": entry.job_id,
                       "decision_log_id": entry.decision_log_id,
                       "screening_record_id": record_id},
            )
    except Exception as exc:
        # _persist_screening_result already swallows its own failures; this is
        # the belt for anything else -- an evicted entry, a threadpool refusal.
        log.warning(
            "async screening record could not be persisted",
            extra={"event": "async_screening_persist_failed",
                   "job_id": getattr(entry, "job_id", None), "exc": str(exc)},
        )


def _async_screen_job_id(artifact) -> Optional[str]:
    """The pending async screen's poll id for this decision, if there is one.

    Returns None whenever the screen did not run async -- deferred, inline, not
    enabled, or no maneuver recommended -- so callers do not have to know which.
    """
    try:
        secondary = artifact.post_maneuver.secondary_conflict
    except AttributeError:
        return None
    return getattr(secondary, "screen_job_id", None)


def _stamp_async_screen_decision(artifact, decision_log_id: str) -> None:
    """SCRUM-484: tie a running async screen to the decision it was run for.

    The screening record is keyed to the decision log id, and that id does not
    exist until after the artifact is built, so it cannot be handed to the worker
    when the screen starts. Stamping it on the entry here is what lets the poll
    endpoint persist the record once the screen resolves.

    Guarded like every other audit-adjacent write on this path: a store entry
    that has already been evicted, or any failure here, costs the record and
    never the decision.
    """
    job_id = _async_screen_job_id(artifact)
    if not job_id:
        return
    try:
        from common.secondary_screen_async import set_decision_log_id

        if not set_decision_log_id(job_id, decision_log_id):
            log.info(
                "async screen entry gone before it could be tied to its decision",
                extra={"event": "screening_stamp_entry_missing",
                       "job_id": job_id, "decision_log_id": decision_log_id},
            )
    except Exception as exc:
        log.warning(
            "could not tie the async screen to its decision",
            extra={"event": "screening_stamp_failed", "job_id": job_id,
                   "exc": str(exc)},
        )


def _persist_leolabs_cdm(parsed_ll) -> Optional[int]:
    """SCRUM-429: store the LeoLabs CDM this evaluate is scoring, return its id.

    Persist-on-evaluate, not a scheduler: the planner already holds a parsed,
    guard-checked CDM at this point, so writing it down costs one POST and no
    background machinery.

    Guarded like every other audit write on this service. Any failure -- a
    malformed mapping, an unreachable ingest, a store error -- logs and returns
    None, which is exactly the value this branch used to hardcode. The evaluate
    then behaves as it does today rather than failing because an audit write
    did. A decision that was made must still be returned even if we could not
    write down what it was made from.
    """
    try:
        store_dict = parsed_ll.to_store_cdm_dict()
    except Exception as exc:
        log.warning(
            "LeoLabs CDM could not be mapped for the store",
            extra={"event": "leolabs_cdm_map_failed", "exc": str(exc)},
        )
        return None

    try:
        resp = http_requests.post(
            f"{_ingest_url()}/cdm/persist", json=store_dict, timeout=5.0
        )
        if resp.status_code != 200:
            _note_audit_failure(
                "leolabs_cdm", str(store_dict.get("COMMENT_ID", "?")),
                f"status {resp.status_code}: {resp.text[:200]}",
            )
            return None
        payload = resp.json()
        record_id = payload.get("id")
        log.info(
            "LeoLabs CDM persisted",
            extra={"event": "leolabs_cdm_persisted",
                   "cdm_record_id": record_id,
                   # Not "created": that is a reserved LogRecord attribute and
                   # passing it in extra raises inside the logging call.
                   "row_created": payload.get("created"),
                   "cdm_id": store_dict.get("COMMENT_ID")},
        )
        return int(record_id) if record_id is not None else None
    except Exception as exc:
        _note_audit_failure(
            "leolabs_cdm", str(store_dict.get("COMMENT_ID", "?")), str(exc)
        )
        return None


def _post_planner_output(
    cdm_record_id: int,
    result: Dict[str, Any],
    body: Dict[str, Any],
    covariance_source: str,
    scoring: Optional[Any] = None,
) -> None:
    """Write a planner decision audit record to the ingest service.

    Fire-and-forget. Any failure is logged and silently swallowed.

    SCRUM-393: pc_computed comes from the scoring result's pc_pre, not from
    risk_surrogate_post.

    This used to read risk_surrogate_post, which meant the audit trail received
    the pre-maneuver Pc when one was supplied and an inverse square kilometre
    when one was not. That second value lands between 1e-4 and 1e-2 at the
    covariances this system produces, which is exactly where a real Pc lives, so
    nothing about the number invited anyone to check it. It was written into a
    non-nullable column described as "APS-computed Pc" and into RiskInputs in
    gnc_interface.yaml, and the SCRUM-377 trail is tamper-evident, so wrong
    entries are preserved rather than corrected.

    pc_computed belongs alongside m2_pre, covariance_source and data_age_s in
    RiskInputs, and those are all pre-maneuver inputs, so the field wants the
    pre-maneuver Pc. risk_surrogate_post was never the right source for it, and
    that stays true now that it carries a genuine post-maneuver Pc.

    None when no Pc could be established. Sent as 0.0 only because the column is
    non-nullable, which is a schema question rather than this ticket's, and is
    noted on SCRUM-396.
    """
    try:
        rec = result.get("recommendation", {})
        policy = body.get("policy", {})

        utility = float(rec.get("utility", 0.0))
        recommendation = "maneuver" if utility > 0.0 else "no_maneuver"
        delta_v_ms = float(rec.get("dv_magnitude_m_s")) if recommendation == "maneuver" else None
        pc_pre = getattr(scoring, "pc_pre", None) if scoring is not None else None
        pc_computed = float(pc_pre) if pc_pre is not None else 0.0

        payload = {
            "cdm_record_id":     cdm_record_id,
            "recommendation":    recommendation,
            "delta_v_ms":        delta_v_ms,
            "pc_computed":       pc_computed,
            "utility_value":     utility,
            "lambda_v":          float(policy.get("lambda_v", 0.0)),
            "lambda_l":          float(policy.get("lambda_L", 0.0)),
            "covariance_source": covariance_source,
        }

        url = f"{_ingest_url()}/planner_output"
        resp = http_requests.post(url, json=payload, timeout=5.0)

        if resp.status_code == 201:
            log.info("audit record written", extra={"event": "audit_written", "planner_output_id": resp.json().get("id"), "cdm_record_id": cdm_record_id})
        else:
            log.warning("audit write unexpected status", extra={"event": "audit_write_unexpected_status", "status": resp.status_code, "cdm_record_id": cdm_record_id})

    except Exception as exc:
        log.warning("audit write failed", extra={"event": "audit_write_failed", "cdm_record_id": cdm_record_id, "exc": str(exc)})


# SCRUM-379, section 6.2. Durable onboard storage has no analogue in a
# stateless planner, so persistence is opt-in: with MODE_STATE_DIR set this is a
# FileModeStore, otherwise a NullModeStore that persists nothing. A store that
# silently retained modes between unrelated requests would be worse than none,
# because a mode left over from another event would read as this one's.
# SCRUM-431: the SCRUM-381 secondary conflict screen ran on a Space-Track TLE
# catalog. That catalog carries no covariance, so the screen could only ever
# produce an assumed-covariance answer, and with Space-Track retired there is no
# catalog at all -- it failed closed and showed VERIFICATION FAILED on every
# evaluate. Deferred, not deleted: the screen logic is intact and flipping this
# flag restores today's fail-closed behaviour exactly, which is what the
# LeoLabs covariance-backed rebuild will do.
# SCRUM-442: the screen is the real LeoLabs on-demand path (SCRUM-440/441/451)
# rather than the SCRUM-431 deferral, and it fails closed on its own. The flag
# stays so it can be turned off, which returns the deliberate deferral rather than
# failing every evaluate.
#
# SCRUM-455: default OFF, and .env is the authoritative control. Both composes now
# forward SECONDARY_SCREEN_ENABLED to this service; until they did, the .env line
# did nothing and the screen ran on this default, which is how a load-bearing
# screen came to be armed by a default rather than by a decision.
#
# Off is the safe direction, not a convenient one. With the flag off no screen
# runs, the secondary check reports the SCRUM-431 deferral, and a deferral is
# never a clear screen -- guard_secondary_clear is omitted from the staging AND
# rather than passed, so nothing is certified secondary-clear on the strength of a
# screen that did not run. A missing .env line therefore costs a capability, which
# is visible, instead of silently arming or silently passing one.
#
# Consequence for deployment: the demo needs the screen, so both the local .env and
# the KVM .env must carry SECONDARY_SCREEN_ENABLED=true. See the deploy checklist.
SECONDARY_SCREEN_ENABLED = (
    os.environ.get("SECONDARY_SCREEN_ENABLED", "false").lower() == "true"
)

_MODE_STORE = build_mode_store()

# SCRUM-382: the latest operator approve/veto per conjunction, so an operator
# decision reaches the state machine that acts on it. Process-local, same
# posture as the mode store's default; an explicit flag on the evaluate request
# still wins, so this never becomes a hidden source of authority.
_OPERATOR_COMMANDS = OperatorCommandStore()

# SCRUM-377: audit writes must not fail silently. A missing evidence record is
# as damaging as a modified one, and the old fire-and-forget path swallowed
# every exception, so loss was undetectable. Failures are counted here and
# surfaced on /health so a gap is visible without reading logs.
_audit_failures: Dict[str, Any] = {
    "evidence_write_failures": 0,
    "decision_log_write_failures": 0,
    "last_error": "",
    "last_failed_record_id": "",
}


def _note_audit_failure(kind: str, record_id: str, exc: str) -> None:
    key = f"{kind}_write_failures"
    _audit_failures[key] = _audit_failures.get(key, 0) + 1
    _audit_failures["last_error"] = exc
    _audit_failures["last_failed_record_id"] = record_id
    log.error(
        "audit write failed, evidence chain may have a gap",
        extra={"event": "audit_write_failed", "kind": kind, "record_id": record_id, "exc": exc},
    )


def _evidence_values(
    artifact, decision_log, policy: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """Map what exists on main today onto the MAF section 10 field catalogue.

    Returns (values, not_applicable).

    The split matters. producer_not_implemented means the component that would
    fill this field has not shipped. not_applicable means the producer exists
    and this particular event simply had nothing to put there, e.g. an event
    for which no Pc could be established. Collapsing the two would make pending_producers()
    report "main" as an outstanding ticket, which is meaningless, and would
    hide which fields are genuinely still owed by 378 to 382.
    """
    risk = artifact.risk_summary
    values: Dict[str, Any] = {
        "conjunction_id": artifact.conjunction_id,
        "timestamp": decision_log.logged_at,
        "software_version": SERVICE_VERSION,
        "model_version": SERVICE_VERSION,
        "policy_version": str(policy.get("policy_version", "")),
        "inputs_and_provenance": {
            "sat_id": artifact.sat_id,
            "operator_id": str(policy.get("operator_id", "")),
            "evaluated_at": artifact.evaluated_at,
            "tca_utc": risk.tca_utc,
            "decision": decision_log.decision,
            "reason_code": decision_log.reason_code,
            # SCRUM-396: Pc provenance belongs with the existing evidence
            # provenance package. pc_at_transition remains the canonical
            # SCRUM-377 top-level transition field.
            "pc_source": decision_log.pc_source,
            "pc_at_decision": decision_log.pc_at_transition,
        },
        "covariance_state": {
            "covariance_quality": risk.covariance_quality,
            "mahalanobis_pre": risk.mahalanobis_pre,
        },
        "orbit_state": {
            "miss_distance_km": risk.miss_distance_km,
            "tca_utc": risk.tca_utc,
        },
        "candidate_maneuvers": _candidate_evidence(artifact),
    }
    not_applicable: List[str] = []
    if decision_log.pc_at_transition is not None:
        values["pc_at_transition"] = decision_log.pc_at_transition
    else:
        # The producer exists, but no Pc could be established for this event.
        not_applicable.append("pc_at_transition")

    return values, not_applicable


def _candidate_evidence(artifact) -> List[Dict[str, Any]]:
    """Candidates considered, and why the ones not chosen were not chosen.

    MAF section 10 requires the rejected candidates, not just the winner.
    """
    out: List[Dict[str, Any]] = []
    chosen = artifact.direction
    for c in (getattr(artifact.rationale, "all_candidates", None) or []):
        entry = dict(c) if isinstance(c, dict) else {"direction": str(c)}
        direction = entry.get("direction", "")
        entry["selected"] = (direction == chosen)
        if not entry["selected"]:
            entry.setdefault(
                "rejection_reason",
                "lower utility than the selected candidate",
            )
        out.append(entry)
    if not out and artifact.no_go is not None:
        out.append({
            "direction": "no-burn",
            "selected": True,
            "rejection_reason": artifact.no_go.reason_code,
        })
    return out


def _post_evidence_record(artifact, decision_log, policy: Dict[str, Any]) -> Optional[str]:
    """Append one decision record to this satellite's evidence chain.

    Chain head is read from ingest so seq and prev_hash are correct. Ingest
    rejects an out-of-order append, so a race produces a refusal rather than a
    corrupted chain.
    """
    chain_id = artifact.sat_id or "UNKNOWN"
    record_id = "?"
    if chain_id == "UNKNOWN":
        # Not fatal: an unidentifiable satellite still gets an audit record.
        # But every such record lands in one shared chain, so say so loudly
        # rather than letting distinct spacecraft interleave silently.
        log.warning(
            "evidence chain has no satellite identity; records will share the "
            "UNKNOWN chain",
            extra={"event": "evidence_chain_unidentified",
                   "conjunction_id": artifact.conjunction_id},
        )
    try:
        head = http_requests.get(
            f"{_ingest_url()}/store/evidence/{chain_id}/head", timeout=5.0
        ).json()
        seq = int(head.get("next_seq", 0))
        prev_hash = head.get("content_hash", GENESIS_HASH)

        ev_values, ev_na = _evidence_values(artifact, decision_log, policy)
        record = build_decision_record(
            chain_id=chain_id,
            seq=seq,
            prev_hash=prev_hash,
            values=ev_values,
            not_applicable=ev_na,
        )
        record_id = record.record_id
        payload = {
            "record": record.to_dict(),
            "canonical_payload": canonical_json(record._hashable_payload()),
        }
        resp = http_requests.post(
            f"{_ingest_url()}/evidence_record", json=payload, timeout=5.0
        )
        if resp.status_code != 201:
            _note_audit_failure("evidence", record_id, f"status {resp.status_code}: {resp.text[:200]}")
            return None
        log.info(
            "evidence record appended",
            extra={"event": "evidence_record_appended", "record_id": record_id,
                   "pending_producers": record.pending_producers()},
        )
        return record_id
    except Exception as exc:
        _note_audit_failure("evidence", record_id, str(exc))
        return None


def _transition_evidence_values(artifact, monitor, policy: Dict[str, Any]) -> Dict[str, Any]:
    """Map one state transition onto the MAF section 10 catalogue.

    SCRUM-379 owns event, from_mode, to_mode and trigger; SCRUM-378's
    validity_evidence_values supplies the four validity fields, already under
    the catalogue's own names; SCRUM-375's compiled envelope supplies the
    authority fields. monitor_results carries every guard the monitor evaluated,
    with the values it read, so a reviewer can see why a guard failed without
    re-running the evaluation.

    Only fields whose producer actually ran are included. A guard that could not
    be evaluated leaves its field producer_not_implemented rather than present
    and null, which is the distinction EvidenceRecord.build exists to keep.
    """
    inputs = monitor.inputs
    values: Dict[str, Any] = {
        "timestamp": monitor.evaluated_at_utc,
        "monitor_results": [g.to_dict() for g in monitor.guards],
        "policy_version": str(policy.get("policy_version", "")),
        "model_version": SERVICE_VERSION,
        "inputs_and_provenance": {
            "sat_id": artifact.sat_id,
            "operator_id": str(policy.get("operator_id", "")),
            "evaluated_at": artifact.evaluated_at,
            "pc_source": inputs.pc_source,
            "covariance_source": inputs.covariance_source,
            "requested_mode": monitor.transition.requested_mode.value,
            "escalated": monitor.transition.escalated,
            # SCRUM-380: the granted level and the level actually acted on. They
            # differ when a baseline drift clamped authority to L0, and an
            # auditor should not have to infer that from a declined execution.
            "authority_granted": inputs.authority(),
            "authority_effective": inputs.effective_authority(),
            "authority_demotion_reason": inputs.authority_demotion(),
            "post_burn_feasible": monitor.post_burn_feasible,
        },
    }
    # SCRUM-380: the section 7 abort entry rides in the same tamper-evident
    # record as the transition it caused, so the chain carries who commanded the
    # stop and why, not merely that the mode changed.
    abort_entry = monitor.abort_audit_entry()
    if abort_entry is not None:
        values["inputs_and_provenance"]["ground_abort"] = abort_entry
    for name in ("validity_status", "validity_epsilon", "epsilon_threshold",
                 "phenomenologies_used", "weak_directions"):
        if name in inputs.validity_evidence:
            values[name] = inputs.validity_evidence[name]
    if inputs.authority_level:
        values["authority_level"] = inputs.authority()
    if inputs.envelope_version:
        values["envelope_version"] = inputs.envelope_version
    if inputs.envelope_approving_identity:
        values["envelope_approving_identity"] = inputs.envelope_approving_identity
    return values


def _post_transition_record(artifact, monitor, policy: Dict[str, Any]) -> Optional[str]:
    """Append one transition record to this satellite's evidence chain.

    Section 7 of the guard doc: 'Every transition produces a tamper-evident
    append-only log entry.' A hold is not a transition and is not recorded --
    an M0 evaluation on a quiet sky would otherwise append a record per poll and
    bury the transitions that matter. An escalation always is, including one
    that lands in the mode it was already in.
    """
    if not monitor.changed_mode and not monitor.escalated:
        return None

    chain_id = artifact.sat_id or "UNKNOWN"
    record_id = "?"
    try:
        head = http_requests.get(
            f"{_ingest_url()}/store/evidence/{chain_id}/head", timeout=5.0
        ).json()
        record = build_transition_record(
            chain_id=chain_id,
            seq=int(head.get("next_seq", 0)),
            prev_hash=head.get("content_hash", GENESIS_HASH),
            from_mode=monitor.transition.from_mode.value,
            to_mode=monitor.transition.to_mode.value,
            trigger=monitor.transition.trigger,
            conjunction_id=monitor.conjunction_id,
            software_version=SERVICE_VERSION,
            pc_at_transition=monitor.inputs.pc,
            extra_values=_transition_evidence_values(artifact, monitor, policy),
            recorded_at=monitor.evaluated_at_utc,
        )
        record_id = record.record_id
        resp = http_requests.post(
            f"{_ingest_url()}/evidence_record",
            json={
                "record": record.to_dict(),
                "canonical_payload": canonical_json(record._hashable_payload()),
            },
            timeout=5.0,
        )
        if resp.status_code != 201:
            _note_audit_failure(
                "transition", record_id, f"status {resp.status_code}: {resp.text[:200]}"
            )
            return None
        log.info(
            "transition record appended",
            extra={"event": "transition_record_appended", "record_id": record_id,
                   "from_mode": monitor.transition.from_mode.value,
                   "to_mode": monitor.transition.to_mode.value},
        )
        return record_id
    except Exception as exc:
        _note_audit_failure("transition", record_id, str(exc))
        return None


def _post_decision_log(decision_log) -> None:
    """Persist the full DecisionLog audit record to the ingest store (SCRUM-351).

    Fire-and-forget, keyed by log_id so an operator can retrieve any past
    decision by its decision ID. Any failure is logged and swallowed.
    """
    try:
        payload = {
            "log_id":         decision_log.log_id,
            "conjunction_id": decision_log.conjunction_id,
            "sat_id":         decision_log.sat_id,
            "decision":       decision_log.decision,
            "decision_log":   vars(decision_log),
        }
        url = f"{_ingest_url()}/decision_log"
        resp = http_requests.post(url, json=payload, timeout=5.0)
        if resp.status_code != 201:
            _note_audit_failure("decision_log", decision_log.log_id, f"status {resp.status_code}")
    except Exception as exc:
        _note_audit_failure("decision_log", getattr(decision_log, "log_id", "?"), str(exc))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_start_time = time.time()


@asynccontextmanager
async def lifespan(a):
    log.info("service starting", extra={"event": "startup", "version": SERVICE_VERSION, "policy_config": str(_POLICY_CONFIG_PATH)})
    global _operator_policy
    try:
        _operator_policy = OperatorPolicy.from_yaml(str(_POLICY_CONFIG_PATH))
    except Exception as exc:
        log.warning(
            "operator policy load failed at startup, using CovarianceSurrogate defaults",
            extra={"event": "operator_policy_load_fail", "exc": str(exc)},
        )
        _operator_policy = None
    yield
    log.info("service shutting down", extra={"event": "shutdown", "version": SERVICE_VERSION})


svc = FastAPI(
    title="APS Planner - Avoidance Decision Model",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request logging middleware (SCRUM-341 AC3)
# ---------------------------------------------------------------------------

@svc.middleware("http")
async def _log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    log.info("request", extra={"method": request.method, "path": request.url.path, "status": response.status_code, "elapsed_ms": round((time.time() - t0) * 1000, 1)})
    return response


# ---------------------------------------------------------------------------
# Health and readiness (SCRUM-341 AC1, AC5)
# ---------------------------------------------------------------------------

@svc.get("/health")
async def health() -> Dict[str, Any]:
    """Liveness probe. Returns 200 with version and uptime.
    Polled by Dockerfile HEALTHCHECK and k8s liveness probe.
    """
    return {
        "status":   "ok",
        "service":  SERVICE_NAME,
        "version":  SERVICE_VERSION,
        "uptime_s": round(time.time() - _start_time, 1),
        # SCRUM-377: a failed audit write must be visible without reading logs.
        "audit": dict(_audit_failures),
    }


@svc.get("/ready")
async def ready() -> Dict[str, Any]:
    """Readiness probe. Returns 200 when the service is up and the operator
    policy config is loadable. API-only readiness -- no disk volume check.

    SCRUM-341 AC1: probes that:
      1. The service process is running (implicit -- if we get here, it is)
      2. The operator policy config file exists and parses without error

    Returns 503 if the policy file is missing or malformed, so the container
    is not marked ready before its required config is in place.
    """
    try:
        if not _POLICY_CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Operator policy config not found: {_POLICY_CONFIG_PATH}"
            )
        # Attempt a parse to catch malformed YAML early.
        OperatorPolicy.from_yaml(str(_POLICY_CONFIG_PATH))
        return {
            "status":        "ready",
            "version":       SERVICE_VERSION,
            "policy_config": str(_POLICY_CONFIG_PATH),
        }
    except Exception as exc:
        log.warning("readiness check failed", extra={"event": "readiness_fail", "exc": str(exc)})
        raise HTTPException(status_code=503, detail=str(exc))


# ---------------------------------------------------------------------------
# UDL status (SCRUM-331 AC6)
# ---------------------------------------------------------------------------

@svc.get("/udl-status")
async def udl_status() -> Dict[str, Any]:
    """UDL credential and connection status.

    SCRUM-331 AC6: reflects live credential and connection state.
    SCRUM-363: adds a real, throttled credential validity probe.

    Returns:
      enabled                    : bool -- reflects UDL_ENABLED env var
      credentials_set             : bool -- UDL_USER and UDL_PASS are present (not validated)
      credential_valid             : bool -- result of the live probe (SCRUM-363).
                                             False if never probed (e.g. UDL disabled).
      credential_check_status      : str  -- 'valid' | 'invalid' | 'unreachable' | 'not_checked'
      last_credential_check_utc    : str or None -- timestamp of the last live probe
                                             (may be older than now if the cached
                                             result was reused -- see udl_client
                                             throttle interval)
      mode             : str    -- 'connected' | 'invalid' | 'unconfirmed' | 'misconfigured' | 'disabled'
                                  (SCRUM-348: 'connected' replaces the old 'live'; a genuine
                                   UDL-driven 'live' returns with the per-evaluate source field)
      label            : str    -- human-readable label for the UI badge
      note             : str    -- additional context for the operator
      last_fetch_utc   : str or None -- last successful conjunction data fetch
                                        (distinct from last_credential_check_utc --
                                        AC3: these are never conflated)
    """
    import os
    credentials_set = bool(
        os.environ.get("UDL_USER") and os.environ.get("UDL_PASS")
    )

    # --- SCRUM-363: throttled live credential probe -----------------------
    credential_valid = False
    credential_check_status = "not_checked"
    last_credential_check_utc = None

    if UDL_ENABLED:
        try:
            probe = get_credential_validity()
            credential_check_status = probe["status"]
            credential_valid = (probe["status"] == "valid")
            last_credential_check_utc = probe["checked_at_utc"]
        except Exception as exc:
            # AC5: a probe failure must not break /udl-status. Fall back to
            # an unconfirmed state and keep returning the rest of the fields.
            log.warning(
                "credential probe raised unexpectedly",
                extra={"event": "udl_status_probe_error", "exc": str(exc)},
            )
            credential_check_status = "unreachable"
            credential_valid = False
            last_credential_check_utc = None
    # ------------------------------------------------------------------

    if UDL_ENABLED and credentials_set and credential_valid:
        # SCRUM-348: UDL being enabled and authenticated does NOT mean the
        # planner consumed a UDL conjunction. UDL currently contributes catalog
        # elsets to secondary screening only, not conjunctions to the planner,
        # so the badge must not claim "UDL LIVE". A genuine live state, tied to
        # an actual per-evaluate UDL conjunction, lands with the consumption
        # path (SCRUM-364 / AC3).
        mode = "connected"
        label = "UDL CONNECTED (CATALOG ONLY)"
        note = (
            "UDL enabled and credentials confirmed valid. UDL is contributing "
            "catalog elsets to secondary screening, but is not driving planner "
            "conjunctions. Conjunctions come from the reference or Space-Track "
            "source."
        )
    elif UDL_ENABLED and credentials_set and credential_check_status == "invalid":
        mode = "invalid"
        label = "UDL CREDENTIALS INVALID"
        note = "Credentials are set but did not authenticate on the last check. They may have been rotated or revoked."
    elif UDL_ENABLED and credentials_set and credential_check_status == "unreachable":
        mode = "unconfirmed"
        label = "UDL UNCONFIRMED"
        note = "Could not reach UDL to confirm credential validity. Credentials may still be valid; last check was inconclusive."
    elif UDL_ENABLED and not credentials_set:
        mode = "misconfigured"
        label = "UDL MISCONFIGURED"
        note = "UDL_ENABLED=true but UDL_USER or UDL_PASS is missing. Set both env vars."
    else:
        mode = "disabled"
        label = "UDL DISABLED"
        note = (
            "UDL_ENABLED=false. Set UDL_ENABLED=true after service account "
            "is confirmed to activate live UDL conjunction data."
        )

    return {
        "enabled":                    UDL_ENABLED,
        "credentials_set":            credentials_set,
        "credential_valid":           credential_valid,
        "credential_check_status":    credential_check_status,
        "last_credential_check_utc":  last_credential_check_utc,
        "mode":                       mode,
        "label":                      label,
        "note":                       note,
        "last_fetch_utc":             _udl_last_fetch_utc,
    }


# ---------------------------------------------------------------------------
# LeoLabs status (SCRUM-412)
# ---------------------------------------------------------------------------

@svc.get("/leolabs-status")
async def leolabs_status() -> Dict[str, Any]:
    """LeoLabs credential and connection status for the dashboard badge.

    Mirrors /udl-status. Unlike UDL, LeoLabs -- when enabled and authenticated --
    genuinely drives planner conjunctions, so the enabled+valid mode is "live"
    and the dashboard shows a green LIVE badge. A throttled credential probe backs
    the credential_valid field; a probe failure degrades to "unconfirmed" rather
    than breaking the endpoint.
    """
    try:
        return leolabs_runtime.get_status()
    except Exception as exc:
        log.warning(
            "leolabs status probe raised unexpectedly",
            extra={"event": "leolabs_status_error", "exc": str(exc)},
        )
        return {
            "enabled": LEOLABS_ENABLED,
            "credentials_set": bool(
                os.environ.get("LEOLABS_ACCESS_KEY")
                and os.environ.get("LEOLABS_SECRET_KEY")
            ),
            "credential_valid": False,
            "credential_check_status": "unreachable",
            "last_credential_check_utc": None,
            "mode": "unconfirmed",
            "label": "LEOLABS UNCONFIRMED",
            "note": f"status probe failed: {exc}",
            "last_fetch_utc": None,
        }


@svc.get("/gnc-status")
async def gnc_status() -> Dict[str, Any]:
    """SCRUM-382: GNC emission mode for the dashboard badge.

    Mirrors /leolabs-status and /udl-status. record_only is the normal state
    today: no GNC service exists yet, so commands are built and recorded rather
    than emitted, and they carry no acknowledgement.
    """
    try:
        return gnc_client.get_status()
    except Exception as exc:
        log.warning(
            "gnc status probe raised unexpectedly",
            extra={"event": "gnc_status_error", "exc": str(exc)},
        )
        return {"enabled": False, "mode": "disabled", "label": "GNC DISABLED",
                "note": f"status probe failed: {exc}"}


def _persisted_for_conjunction(conjunction_id: str):
    """The persisted mode record matching a conjunction, or None.

    The mode store is keyed by satellite, and an operator command arrives keyed
    by conjunction, so this walks the one to find the other. Returns None rather
    than guessing when the store holds a different conjunction: acking an
    approval against the wrong event would be worse than refusing it.
    """
    store = _MODE_STORE
    reader = getattr(store, "_states", None)
    candidates = list(reader.values()) if isinstance(reader, dict) else []
    for state in candidates:
        if state.conjunction_id == conjunction_id:
            return state
    return None


@svc.post("/v1/gnc/approve")
async def gnc_approve(request: Request) -> JSONResponse:
    """SCRUM-382: operator approval for a staged L1 burn (contract: approveGNCCommand).

    Records the approval against its conjunction so the next /v1/evaluate carries
    it into the state machine's approval inputs, and returns the contract's
    GNCApprovalAck. The approval is permission; the burn itself is still the
    state machine's decision on the next evaluation, where every M2 guard is
    re-checked.
    """
    try:
        command: Dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse(status_code=422, content=error_response("Invalid JSON body"))

    missing = [
        f for f in ("command_id", "conjunction_id", "issued_by", "issued_at_utc")
        if not command.get(f)
    ]
    if missing:
        return JSONResponse(
            status_code=422,
            content=error_response(
                f"GNCApprovalCommand is missing required field(s): {missing}"
            ),
        )

    persisted = _persisted_for_conjunction(str(command["conjunction_id"]))
    ack = approval_ack(
        command,
        current_mode=persisted.mode if persisted else None,
        has_staged_command=bool(persisted and persisted.command is not None),
    )
    try:
        record_approval(_OPERATOR_COMMANDS, command, ack["approval_accepted"])
    except Exception as exc:
        log.warning(
            "operator approval could not be recorded",
            extra={"event": "gnc_approval_record_failed", "exc": str(exc)},
        )
    log.info(
        "operator approval received",
        extra={"event": "gnc_approval_received",
               "command_id": ack["command_id"],
               "conjunction_id": ack["conjunction_id"],
               "approval_accepted": ack["approval_accepted"],
               "issued_by": command.get("issued_by")},
    )
    return JSONResponse(status_code=200, content=ack)


@svc.post("/v1/gnc/veto")
async def gnc_veto(request: Request) -> JSONResponse:
    """SCRUM-382: operator veto of a committed L2 burn (contract: vetoGNCCommand).

    A veto inside the window is accepted and re-stages per section 3's M2 to M2
    row. A veto after veto_window_close_utc is not accepted and the burn
    proceeds, with the reason in late_veto_note, per the contract.
    """
    try:
        command: Dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse(status_code=422, content=error_response("Invalid JSON body"))

    missing = [
        f for f in ("command_id", "conjunction_id", "issued_by", "issued_at_utc")
        if not command.get(f)
    ]
    if missing:
        return JSONResponse(
            status_code=422,
            content=error_response(
                f"GNCVetoCommand is missing required field(s): {missing}"
            ),
        )

    persisted = _persisted_for_conjunction(str(command["conjunction_id"]))
    window_closed = bool(command.get("window_closed", False))
    ack = veto_ack(
        command,
        current_mode=persisted.mode if persisted else None,
        window_closed=window_closed,
    )
    try:
        record_veto(_OPERATOR_COMMANDS, command, ack["veto_accepted"])
    except Exception as exc:
        log.warning(
            "operator veto could not be recorded",
            extra={"event": "gnc_veto_record_failed", "exc": str(exc)},
        )
    log.info(
        "operator veto received",
        extra={"event": "gnc_veto_received",
               "command_id": ack["command_id"],
               "conjunction_id": ack["conjunction_id"],
               "veto_accepted": ack["veto_accepted"],
               "issued_by": command.get("issued_by")},
    )
    return JSONResponse(status_code=200, content=ack)


@svc.post("/v1/gnc/report")
async def gnc_report_endpoint(request: Request) -> JSONResponse:
    """SCRUM-382: the post-burn execution report, GNC to APS (receiveGNCReport).

    Re-evaluates the risk on the state the burn actually produced, assembles the
    MAF section 10 producers SCRUM-382 owns, records them on the SCRUM-377
    evidence trail next to the decision that authorised the burn, and returns the
    contract's GNCReportAck.

    The post-maneuver Pc is computed by maneuver_scorer.compute_pc_post; the
    ExecutionError block is SCRUM-365's and is carried through unchanged.
    """
    try:
        report: Dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse(status_code=422, content=error_response("Invalid JSON body"))

    missing = [
        f for f in ("command_id", "conjunction_id", "execution_status")
        if not report.get(f)
    ]
    if missing:
        return JSONResponse(
            status_code=422,
            content=error_response(
                f"GNCReport is missing required field(s): {missing}"
            ),
        )

    risk = report.get("aps_risk_context") or {}
    try:
        assessment = assess_post_burn(
            report,
            commanded_dv_m_s=risk.get("commanded_dv_m_s"),
            commanded_dv_rtn_m_s=risk.get("commanded_dv_rtn_m_s"),
            r_rel_km_at_tca=risk.get("r_rel_km_at_tca"),
            v_rel_km_s_at_tca=risk.get("v_rel_km_s_at_tca"),
            p_pre_km2=risk.get("p_pre_km2"),
            hbr_m=risk.get("hbr_m"),
            pc_pre=risk.get("pc_pre"),
            pc_monitor_threshold=risk.get(
                "pc_monitor_threshold", OperatorPolicy.__dataclass_fields__[
                    "pc_monitor_threshold"].default
            ),
        )
    except Exception as exc:
        log.error(
            "GNC report could not be assessed",
            extra={"event": "gnc_report_assess_failed",
                   "command_id": report.get("command_id"), "exc": str(exc)},
        )
        return JSONResponse(
            status_code=500,
            content=error_response(f"report assessment failed: {exc}"),
        )

    # Additive, like every other audit write on this service: a failed evidence
    # append is logged and counted, and the ack still goes back to GNC. A GNC
    # layer left waiting on an ack because our audit store was down would be a
    # worse failure than a gap we can see in the chain.
    _post_gnc_report_record(assessment, report)

    log.info(
        "GNC report consumed",
        extra={"event": "gnc_report_consumed",
               "command_id": assessment.command_id,
               "execution_status": assessment.execution_status,
               "pc_post": assessment.pc_post,
               "replan_required": assessment.replan_required},
    )
    return JSONResponse(status_code=200, content=build_report_ack(assessment))


def _emit_to_gnc(result, monitor, scoring, artifact, sat_dict) -> None:
    """SCRUM-382: assemble the GNCCommand for an authorised burn and emit it.

    Additive throughout. When GNC is disabled nothing is attached and the
    evaluate response is byte-identical to its pre-382 shape; when GNC is in
    record-only mode the command is attached but carries no ack, because nothing
    acknowledged it.
    """
    if gnc_client.emission_mode() == gnc_client.MODE_DISABLED:
        return

    authorized = monitor.authorized_execution
    inputs = monitor.inputs
    risk = artifact.risk_summary

    context = GNCCommandContext(
        approving_identity=str(inputs.envelope_approving_identity or "unattributed"),
        r_sat_km=tuple(float(v) for v in sat_dict.get("r_sat_km", (0.0, 0.0, 0.0))),
        v_sat_km_s=tuple(float(v) for v in sat_dict.get("v_sat_km_s", (0.0, 0.0, 0.0))),
        direction=str(scoring.direction),
        # The scorer models an impulsive burn, so there is no computed duration.
        # Reported as zero rather than as a fabricated non-zero: a made-up burn
        # duration is a number GNC would plan a slew around.
        burn_duration_s=float(sat_dict.get("burn_duration_s", 0.0)),
        epsilon_threshold=float(
            inputs.validity_evidence.get("epsilon_threshold", 0.20)
        ),
        pc_computed=float(authorized.validity_epsilon if scoring.pc_pre is None
                          else scoring.pc_pre),
        m2_pre=float(scoring.m2_pre),
        data_age_s=float(inputs.data_age_s if inputs.data_age_s is not None else 0.0),
        t_ca_utc=str(risk.tca_utc or ""),
        latest_burn_utc=(
            inputs.latest_burn_utc.isoformat().replace("+00:00", "Z")
            if inputs.latest_burn_utc else authorized.t_burn_utc
        ),
        veto_window_open_utc=authorized.authorized_at_utc,
        veto_window_close_utc=(
            inputs.veto_window_close_utc.isoformat().replace("+00:00", "Z")
            if inputs.veto_window_close_utc else ""
        ),
        # The contract requires a safe-action reference. Passive hold is the
        # honest default: APS has no canned nudge defined, and naming one that
        # does not exist would tell GNC it has a fallback it does not have.
        safe_action_id=f"passive-hold-{authorized.conjunction_id}",
        safe_action_type="passive_hold",
        weak_directions=tuple(inputs.validity_evidence.get("weak_directions", ()) or ()),
        phenomenologies_used=tuple(
            inputs.validity_evidence.get("phenomenologies_used", ()) or ()
        ),
        dv_return_m_s=getattr(scoring, "dv_return_m_s", None),
    )
    gnc_command = build_gnc_command(authorized, context)
    emission = emit_gnc_command(gnc_command)

    result["gnc_command"] = gnc_command
    result["gnc_emission"] = emission.to_dict()
    log.info(
        "GNC command assembled",
        extra={"event": "gnc_command_assembled",
               "command_id": gnc_command["command_id"],
               "mode": emission.mode,
               "acknowledged": emission.acknowledged},
    )


def _post_gnc_report_record(assessment, report: Dict[str, Any]) -> Optional[str]:
    """Append the post-burn outcome to this satellite's evidence chain."""
    chain_id = str(report.get("sat_id") or assessment.conjunction_id or "UNKNOWN")
    record_id = "?"
    try:
        head = http_requests.get(
            f"{_ingest_url()}/store/evidence/{chain_id}/head", timeout=5.0
        ).json()
        values = {
            "conjunction_id": assessment.conjunction_id,
            "timestamp": report.get("reported_at_utc")
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "software_version": SERVICE_VERSION,
            **assessment.evidence_values(),
        }
        record = build_decision_record(
            chain_id=chain_id,
            seq=int(head.get("next_seq", 0)),
            prev_hash=head.get("content_hash", GENESIS_HASH),
            values=values,
        )
        record_id = record.record_id
        resp = http_requests.post(
            f"{_ingest_url()}/evidence_record",
            json={
                "record": record.to_dict(),
                "canonical_payload": canonical_json(record._hashable_payload()),
            },
            timeout=5.0,
        )
        if resp.status_code != 201:
            _note_audit_failure(
                "gnc_report", record_id, f"status {resp.status_code}"
            )
            return None
        log.info(
            "GNC report record appended",
            extra={"event": "gnc_report_record_appended", "record_id": record_id},
        )
        return record_id
    except Exception as exc:
        _note_audit_failure("gnc_report", record_id, str(exc))
        return None


# SCRUM-447 globe tracks. 90 steps is what the UI's /api/orbits has always drawn
# a revolution with, so the live globe renders at the same fidelity as the
# scenario one. The object cap is a rendering bound, not a data one: past a few
# hundred rings the globe is unreadable and the browser, not the API, is the
# limit.
# ---------------------------------------------------------------------------
# SCRUM-459: keeping a dense asset's window fetch off the event loop
# ---------------------------------------------------------------------------
#
# The list and globe routes are `async def` and called fetch_leolabs_conjunction_page
# -- which does synchronous `requests` work -- directly on the event loop. The
# planner runs one uvicorn worker, so one event loop: a single such call blocks
# *everything*, including /health and every other asset, for as long as it runs.
#
# That is worse than the ticket's premise, and the evidence that looked reassuring
# was measuring the wrong thing. The request log showed `/health` at 0.2 ms during a
# SWARM B fetch, which reads like health being fine. It is not: the logging
# middleware times the handler, starting after the event loop picks the request up.
# Measured from the client instead, `/health` during a single SWARM B list fetch did
# not answer at all -- a 120 s curl timed out, against 1.6 ms idle. The handler
# really did take 0.2 ms; it just did not get to run for two minutes.
#
# So the fetch is handed to a dedicated executor and awaited. The loop stays free,
# /health keeps answering, and other assets keep listing.
#
# A dedicated pool, not the default one Starlette offloads sync endpoints to: these
# fetches are minutes long and would otherwise occupy threads that every `def`
# endpoint in the service depends on. Two workers, because concurrent heavy pulls
# also contend for the client's 4 req/s org-wide budget, so more parallelism would
# not make any of them faster.
_LIST_EXECUTOR = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="leolabs-list",
)

# In-flight cap, matched to the pool size so the cap is what refuses work rather
# than an invisible queue behind the pool. Past it the endpoint sheds a typed 503
# instead of queueing: a request that waits behind two 25 s pulls has already lost
# its own budget, and shedding tells the operator to retry rather than timing out.
_LIST_MAX_INFLIGHT = 2

# A plain int is safe here. Every read and write happens on the event loop, and
# there is no await between the check and the increment, so no other coroutine can
# interleave. A lock would add nothing but a way to get it wrong.
_list_inflight = 0


def _list_fetch_busy() -> bool:
    return _list_inflight >= _LIST_MAX_INFLIGHT


async def _run_list_fetch(fn, *args, **kwargs):
    """Run a blocking LeoLabs window fetch off the event loop, under the cap.

    Raises ListFetchBusy when the cap is already taken, so the caller can shed.
    """
    global _list_inflight
    if _list_fetch_busy():
        raise ListFetchBusy(
            f"{_list_inflight} LeoLabs window fetches already in flight "
            f"(limit {_LIST_MAX_INFLIGHT})"
        )
    _list_inflight += 1
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _LIST_EXECUTOR, functools.partial(fn, *args, **kwargs)
        )
    finally:
        _list_inflight -= 1


class ListFetchBusy(RuntimeError):
    """The concurrent-window-fetch cap is taken. Shed, do not queue."""


GLOBE_TRACK_STEPS = 90
MIN_GLOBE_TRACK_STEPS = 8
MAX_GLOBE_TRACK_STEPS = 360
MAX_GLOBE_OBJECTS = 250



@svc.post("/v1/leolabs/portfolio")
async def leolabs_portfolio(
    request: Request,
    primary_norad: int,
    lookahead_days: int = leolabs_runtime.DEFAULT_LOOKAHEAD_DAYS,
    lookback_days: int = leolabs_runtime.DEFAULT_LOOKBACK_DAYS,
    in_volume: bool = True,
    max_relative_position_r_m: Optional[float] = None,
    max_relative_position_i_m: Optional[float] = None,
    max_relative_position_c_m: Optional[float] = None,
) -> JSONResponse:
    """SCRUM-480: read-only portfolio comparison; no authorization or emission.

    Body accepts satellite and policy blocks with the same operational inputs
    used by live evaluate. Counts and modeled avoidance-burn delta-v describe
    analyzed events only; incomplete coverage is explicitly marked partial.
    """
    from math import isfinite

    if not LEOLABS_ENABLED:
        return JSONResponse(
            status_code=503,
            content=error_response(
                "LEOLABS_ENABLED=false; the live portfolio is unavailable"
            ),
        )

    try:
        if primary_norad <= 0:
            raise ValueError("primary_norad must be positive")
        if lookback_days < 0 or lookahead_days < 0:
            raise ValueError("lookback_days and lookahead_days must be >= 0")
        if lookback_days + lookahead_days > leolabs_runtime.MAX_WINDOW_DAYS:
            raise ValueError(
                f"requested window exceeds {leolabs_runtime.MAX_WINDOW_DAYS} days"
            )

        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("request must be a JSON object")
        satellite = body.get("satellite", {})
        policy = body.get("policy", {})
        if not isinstance(satellite, dict) or not isinstance(policy, dict):
            raise ValueError("satellite and policy must be JSON objects")
        _policy_from_dict(policy)

        remaining = float(satellite.get("v_remaining_m_s", 0.0))
        if not isfinite(remaining) or remaining < 0:
            raise ValueError("v_remaining_m_s must be finite and >= 0")
        if satellite.get("a_ref_km") is not None:
            a_ref = float(satellite["a_ref_km"])
            if not isfinite(a_ref) or a_ref <= 0:
                raise ValueError("a_ref_km must be finite and positive")
        if satellite.get("t_burn_utc") is not None:
            datetime.fromisoformat(
                str(satellite["t_burn_utc"]).replace("Z", "+00:00")
            )

        for value in (
            max_relative_position_r_m,
            max_relative_position_i_m,
            max_relative_position_c_m,
        ):
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError("reporting-volume limits must be finite and positive")
        filters = leolabs_runtime.resolve_volume_filters(
            max_relative_position_r_m,
            max_relative_position_i_m,
            max_relative_position_c_m,
            in_volume=in_volume,
        )
    except (ValueError, TypeError, OverflowError) as exc:
        return JSONResponse(status_code=422, content=error_response(str(exc)))

    try:
        result = await _run_list_fetch(
            analyze_leolabs_portfolio,
            int(primary_norad),
            satellite=satellite,
            policy=policy,
            now=datetime.now(timezone.utc),
            lookback_days=lookback_days,
            lookahead_days=lookahead_days,
            volume_filters=filters,
        )
    except ListFetchBusy as exc:
        return JSONResponse(
            status_code=503,
            content=error_response(
                f"Planner busy; retry the portfolio analysis in a few seconds. ({exc})"
            ),
        )
    except LeoLabsRuntimeError as exc:
        return JSONResponse(status_code=404, content=error_response(str(exc)))
    except Exception as exc:
        log.warning(
            "LeoLabs portfolio analysis failed",
            extra={"event": "leolabs_portfolio_failed",
                   "primary_norad": primary_norad, "exc": str(exc)},
        )
        return JSONResponse(
            status_code=503,
            content=error_response(f"LeoLabs portfolio analysis failed: {exc}"),
        )

    log.info(
        "LeoLabs portfolio analysis completed",
        extra={"event": "leolabs_portfolio_completed",
               "primary_norad": primary_norad,
               "count": result["count"],
               "complete": result["complete"],
               "cost": result["cost"]},
    )
    return JSONResponse(status_code=200, content=result)


@svc.get("/v1/leolabs/conjunctions")
async def leolabs_conjunctions(
    primary_norad: int,
    lookahead_days: int = leolabs_runtime.DEFAULT_LOOKAHEAD_DAYS,
    lookback_days: int = leolabs_runtime.DEFAULT_LOOKBACK_DAYS,
    page_size: int = leolabs_runtime.DEFAULT_PAGE_SIZE,
    cursor: Optional[str] = None,
    in_volume: bool = True,
    max_relative_position_r_m: Optional[float] = None,
    max_relative_position_i_m: Optional[float] = None,
    max_relative_position_c_m: Optional[float] = None,
) -> JSONResponse:
    """SCRUM-422: list a subscribed asset's live LeoLabs conjunctions.

    A read, not a scoring run. It returns one row per conjunction event in the
    TCA window, ordered highest Pc first with earliest TCA as the tie-break, so
    the dashboard's Active Conjunctions table (SCRUM-421) can render an asset's
    real close approaches without paying for an evaluate per row. Each row
    carries a selector (cdm_id / event_id / secondary_norad) that POST
    /v1/evaluate accepts, which is what makes a row individually clickable.

    The Pc on a row is LeoLabs' own, labelled pc_source="leolabs_cdm". The
    planner's Pc for a row comes from scoring it.

    Status codes, and why each is what it is:
      200  A listing, possibly empty. An empty window is a real state of the
           world -- the asset has nothing coming up -- not an error.
      404  The NORAD id is not in this account's subscribed-objects registry, so
           there is nothing to list and never will be until it is subscribed.
      422  The requested window is wider than the LeoLabs 30-day cap.
      503  LeoLabs is disabled, or the fetch failed. Deliberately not an empty
           200: "the feed is off" and "the sky is clear" must not look alike on
           an operator's screen.

    Paging and the reporting volume (SCRUM-445)
    -------------------------------------------
    This used to return every scorable CDM in the window in one response. Once
    SCRUM-438 made the pull complete, a dense asset (SWARM C / 39453) overran the
    ui-to-planner 60 s read timeout, so a request now returns one page:

      page_size   rows per page, capped at leolabs_runtime.MAX_PAGE_SIZE.
      cursor      the next_cursor from a prior page. Opaque and forward-only. It
                  is bound to the asset, window, page size and volume filter it
                  was minted for; replayed against a different query it is a 422
                  rather than an offset into an unrelated list.
      in_volume   defaults true, meaning LeoLabs' own reporting volume
                  (2 x 50 x 50 km RIC). Pass false to widen to the whole window,
                  which is honest about being slower rather than silently capped.
      max_relative_position_{r,i,c}_m
                  override one RIC axis, in metres.

    total counts conjunction events across the whole in-volume set, not just this
    page, and rows stay ordered worst Pc first with earliest TCA as the tie-break
    across that whole set -- so page one really does hold the worst conjunctions,
    which is the only ordering a triage table can be read top-down.
    """
    if not LEOLABS_ENABLED:
        return JSONResponse(
            status_code=503,
            content=error_response(
                "LEOLABS_ENABLED=false; the live conjunction list is unavailable"
            ),
        )

    if lookback_days < 0 or lookahead_days < 0:
        return JSONResponse(
            status_code=422,
            content=error_response("lookback_days and lookahead_days must be >= 0"),
        )
    if lookback_days + lookahead_days > leolabs_runtime.MAX_WINDOW_DAYS:
        return JSONResponse(
            status_code=422,
            content=error_response(
                f"requested window spans {lookback_days + lookahead_days} days; "
                f"LeoLabs caps minTca..maxTca at {leolabs_runtime.MAX_WINDOW_DAYS}"
            ),
        )

    if page_size < 1:
        return JSONResponse(
            status_code=422,
            content=error_response("page_size must be >= 1"),
        )

    volume_filters = leolabs_runtime.resolve_volume_filters(
        max_relative_position_r_m,
        max_relative_position_i_m,
        max_relative_position_c_m,
        in_volume=in_volume,
    )

    now = datetime.now(timezone.utc)
    try:
        # SCRUM-459: off the event loop, so this fetch cannot stop /health or the
        # other assets, and under the in-flight cap so a burst sheds instead of
        # queueing past everyone's timeout.
        page = await _run_list_fetch(
            fetch_leolabs_conjunction_page,
            int(primary_norad),
            now=now,
            lookback_days=lookback_days,
            lookahead_days=lookahead_days,
            page_size=page_size,
            cursor=cursor,
            volume_filters=volume_filters,
        )
    except ListFetchBusy as exc:
        log.info(
            "shedding a LeoLabs list fetch; the in-flight cap is taken",
            extra={"event": "leolabs_list_shed", "primary_norad": primary_norad,
                   "in_flight": _list_inflight},
        )
        return JSONResponse(
            status_code=503,
            content=error_response(
                f"The planner is already pulling {_LIST_MAX_INFLIGHT} conjunction "
                f"windows. This is a busy signal, not an empty sky: retry in a few "
                f"seconds. ({exc})"
            ),
        )
    except LeoLabsCursorError as exc:
        # A cursor from a different asset, window, page size or volume filter.
        # Paging on it would walk one asset's offsets through another's list, so
        # it is a bad request and the dashboard re-reads page one.
        return JSONResponse(status_code=422, content=error_response(str(exc)))
    except LeoLabsRuntimeError as exc:
        return JSONResponse(status_code=404, content=error_response(str(exc)))
    except Exception as exc:
        log.warning(
            "LeoLabs conjunction list failed",
            extra={"event": "leolabs_list_failed",
                   "primary_norad": primary_norad, "exc": str(exc)},
        )
        return JSONResponse(
            status_code=503,
            content=error_response(f"LeoLabs fetch failed: {exc}"),
        )

    # The page's rows are already one per event -- the paged fetch dedupes the
    # whole in-volume set before it slices, so an event cannot straddle two pages.
    # Run it again anyway: it is idempotent, and it keeps the endpoint's guarantee
    # true at the endpoint rather than only upstream of it.
    events = dedupe_by_event(page.rows)
    min_tca, max_tca = leolabs_runtime.conjunction_window(
        now, lookback_days, lookahead_days
    )
    log.info(
        "LeoLabs conjunction list served",
        extra={"event": "leolabs_list_served", "primary_norad": primary_norad,
               "cdm_count": page.cdm_total, "count": len(events),
               "total": page.total, "offset": page.offset,
               "in_volume": page.in_volume},
    )
    return JSONResponse(
        status_code=200,
        content={
            "primary_norad": int(primary_norad),
            "source": "leolabs",
            # count is this page; total is every conjunction event in the window
            # under the current volume filter, so a caller can render "1-100 of
            # 1432" without a second request.
            "count": len(events),
            "total": page.total,
            "next_cursor": page.next_cursor,
            "page": {
                "size": page.page_size,
                "offset": page.offset,
                "returned": len(events),
                "has_more": page.next_cursor is not None,
            },
            # Rows are deduped to one per conjunction event; cdm_count is how
            # many CDMs those events were distilled from, so a caller can see
            # that reissues were collapsed rather than dropped.
            "cdm_count": page.cdm_total,
            # SCRUM-459. complete=false means the window behind this list was
            # truncated by the fetch deadline or cap, so `total` counts events in a
            # PREFIX of the window in LeoLabs order -- which is not Pc order, so the
            # worst conjunction may not be in it. A caller must show this as a
            # partial view. A short list presented as complete reads as a quiet sky,
            # and that is the direction that gets someone hurt.
            "complete": page.complete,
            "partial": not page.complete,
            # Named "truncation" and not "window": this dict already has a "window"
            # key for the TCA bounds, and a second one would have silently replaced
            # it -- a duplicate literal key is not an error in Python, the last one
            # simply wins.
            "truncation": {
                "cdms_pulled": page.pulled_cdms,
                "cdms_in_window": page.window_cdm_total,
                "truncated_by": page.truncation_reason,
            } if not page.complete else None,
            # What bounded the fetch. An operator reading a short list needs to
            # know whether it is short because the sky is quiet or because the
            # reporting volume cropped it.
            "volume_filter": {
                "in_volume": page.in_volume,
                "max_relative_position_r_m": page.volume_filters.get(
                    "maxRelativePositionR"
                ),
                "max_relative_position_i_m": page.volume_filters.get(
                    "maxRelativePositionI"
                ),
                "max_relative_position_c_m": page.volume_filters.get(
                    "maxRelativePositionC"
                ),
            },
            "window": {
                "min_tca_utc": min_tca,
                "max_tca_utc": max_tca,
                "lookback_days": lookback_days,
                "lookahead_days": lookahead_days,
            },
            "fetched_at_utc": now.isoformat().replace("+00:00", "Z"),
            "conjunctions": [conjunction_row(p, now=now) for p in events],
        },
    )


@svc.get("/v1/leolabs/orbits")
async def leolabs_orbits(
    primary_norad: int,
    lookahead_days: int = leolabs_runtime.DEFAULT_LOOKAHEAD_DAYS,
    lookback_days: int = leolabs_runtime.DEFAULT_LOOKBACK_DAYS,
    in_volume: bool = True,
    max_relative_position_r_m: Optional[float] = None,
    max_relative_position_i_m: Optional[float] = None,
    max_relative_position_c_m: Optional[float] = None,
    max_objects: int = MAX_GLOBE_OBJECTS,
    steps: int = GLOBE_TRACK_STEPS,
) -> JSONResponse:
    """SCRUM-447: orbit tracks for the 3D globe, from live LeoLabs states.

    Returns one closed revolution for the subscribed asset and for each object it
    has an in-volume conjunction with, plus the risk band per object, so the globe
    can draw what the Active Conjunctions table lists rather than the surrogate
    propagator artifact it drew before SCRUM-446 hid it.

    Where the states come from, and why they differ per object
    ---------------------------------------------------------
    The asset's state is live from GET /catalog/objects/<catalog>/states, which is
    the point of this ticket. A secondary's state is the SAT2 block of the CDM
    already parsed for its conjunction row. That is not a shortcut: get_states
    returns HTTP 403 for an object outside our subscription, which every
    conjunction secondary is, so the CDM is the only live source available for
    them -- and it costs no extra API call, since the CDM was already fetched.

    Epochs are therefore not aligned: the asset is at now, a secondary is at its
    TCA. The tracks are orbital rings, and the renderer already places every
    marker at a random phase along its ring, so the globe does not claim to be a
    simultaneous snapshot and this does not make it less true than it was. Each
    object carries its own epoch_utc so the limitation is legible rather than
    implied.

    Status codes, matching /v1/leolabs/conjunctions exactly:
      200  Tracks, possibly with no objects at all. An asset with nothing in its
           reporting volume is a real and unremarkable state of the world.
      404  The NORAD id is not in this account's subscribed-objects registry.
      422  The window is wider than the LeoLabs cap, or a parameter is invalid.
      503  LeoLabs is disabled, the conjunction fetch failed, or the asset has no
           usable state. Never an empty 200: an operator must not read "the feed
           is off" as "your asset has a clear sky".
    """
    if not LEOLABS_ENABLED:
        return JSONResponse(
            status_code=503,
            content=error_response(
                "LEOLABS_ENABLED=false; the live conjunction globe is unavailable"
            ),
        )

    if lookback_days < 0 or lookahead_days < 0:
        return JSONResponse(
            status_code=422,
            content=error_response("lookback_days and lookahead_days must be >= 0"),
        )
    if lookback_days + lookahead_days > leolabs_runtime.MAX_WINDOW_DAYS:
        return JSONResponse(
            status_code=422,
            content=error_response(
                f"requested window spans {lookback_days + lookahead_days} days; "
                f"LeoLabs caps minTca..maxTca at {leolabs_runtime.MAX_WINDOW_DAYS}"
            ),
        )
    if max_objects < 0:
        return JSONResponse(
            status_code=422,
            content=error_response("max_objects must be >= 0"),
        )
    if steps < MIN_GLOBE_TRACK_STEPS:
        return JSONResponse(
            status_code=422,
            content=error_response(
                f"steps must be >= {MIN_GLOBE_TRACK_STEPS} for a closed track"
            ),
        )
    max_objects = min(int(max_objects), MAX_GLOBE_OBJECTS)
    steps = min(int(steps), MAX_GLOBE_TRACK_STEPS)

    volume_filters = leolabs_runtime.resolve_volume_filters(
        max_relative_position_r_m,
        max_relative_position_i_m,
        max_relative_position_c_m,
        in_volume=in_volume,
    )

    now = datetime.now(timezone.utc)

    # -- the asset's live state. A failure here is 503, not an empty globe ----
    try:
        catalog = catalog_for_norad(int(primary_norad))
    except LeoLabsRuntimeError as exc:
        return JSONResponse(status_code=404, content=error_response(str(exc)))
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content=error_response(f"LeoLabs registry lookup failed: {exc}"),
        )

    try:
        asset_r_km, asset_v_km_s, asset_epoch = latest_state_km(catalog)
        asset_track = propagate_two_body(asset_r_km, asset_v_km_s, n_steps=steps)
    except Exception as exc:
        # Includes LeoLabsStateError and the transport errors. Without the asset
        # there is no globe, and drawing the secondaries alone would show an
        # operator a sky with no spacecraft in it.
        log.warning(
            "LeoLabs asset state unavailable",
            extra={"event": "leolabs_globe_asset_state_failed",
                   "primary_norad": primary_norad, "catalog": catalog,
                   "exc": str(exc)},
        )
        return JSONResponse(
            status_code=503,
            content=error_response(f"asset state unavailable: {exc}"),
        )

    # -- the in-volume conjunctions, via the same fetch the 2D table uses -----
    try:
        # SCRUM-459: the same fetch, so the same wedge. Selecting a dense asset on
        # the globe blocked the event loop exactly as the 2D list did, and the fix
        # has to cover both or the planner still goes down one route over. The
        # deadline and cap come for free with the shared fetch; this is the
        # off-the-loop half.
        page = await _run_list_fetch(
            fetch_leolabs_conjunction_page,
            int(primary_norad),
            now=now,
            lookback_days=lookback_days,
            lookahead_days=lookahead_days,
            page_size=max_objects if max_objects else 1,
            volume_filters=volume_filters,
        )
    except ListFetchBusy as exc:
        log.info(
            "shedding a LeoLabs globe fetch; the in-flight cap is taken",
            extra={"event": "leolabs_globe_shed", "primary_norad": primary_norad,
                   "in_flight": _list_inflight},
        )
        return JSONResponse(
            status_code=503,
            content=error_response(
                f"The planner is already pulling {_LIST_MAX_INFLIGHT} conjunction "
                f"windows. Busy, not an empty sky: retry in a few seconds. ({exc})"
            ),
        )
    except LeoLabsRuntimeError as exc:
        return JSONResponse(status_code=404, content=error_response(str(exc)))
    except Exception as exc:
        log.warning(
            "LeoLabs globe conjunction fetch failed",
            extra={"event": "leolabs_globe_fetch_failed",
                   "primary_norad": primary_norad, "exc": str(exc)},
        )
        return JSONResponse(
            status_code=503,
            content=error_response(f"LeoLabs fetch failed: {exc}"),
        )

    rows = dedupe_by_event(page.rows) if max_objects else []

    object_tracks: Dict[str, List[List[float]]] = {}
    risk: Dict[str, str] = {}
    objects: Dict[str, Dict[str, Any]] = {}
    skipped: List[Dict[str, Any]] = []
    # The parsed row behind the first drawn object, kept so the asset covariance
    # can come from its SAT1 block.
    worst_row_parsed = None

    for parsed in rows:
        # conjunction_row is reused rather than reimplemented so the globe's risk
        # band is literally the band the 2D table shows for the same object; two
        # code paths would be two chances to disagree about what is RED.
        row = conjunction_row(parsed, now=now)
        norad = row.get("secondary_norad")
        key = str(norad if norad is not None else row.get("secondary_designator"))
        if key in object_tracks:
            # One ring per object. Rows arrive worst-first, so the first sighting
            # of an object already carries its worst risk band; a second event for
            # the same object would draw a duplicate ring over the first.
            continue
        try:
            r_km = list(parsed.secondary.r_km)
            v_km_s = list(parsed.secondary.v_km_s)
            track = propagate_two_body(r_km, v_km_s, n_steps=steps)
        except Exception as exc:
            # A secondary with no usable state is skipped, not fatal. It still
            # appears in `skipped` so the globe can say the set it drew is
            # incomplete instead of quietly showing fewer objects.
            skipped.append({"secondary_norad": norad, "reason": str(exc)})
            log.info(
                "skipping globe object with no usable state",
                extra={"event": "leolabs_globe_object_skipped",
                       "secondary_norad": norad, "reason": str(exc)},
            )
            continue
        if not object_tracks:
            worst_row_parsed = parsed
        object_tracks[key] = track
        risk[key] = row["risk_level"]
        # SCRUM-448: the CDM's own position covariance for this object, as
        # principal axes and one-sigma extents. None when the covariance is
        # missing or not usable -- the key is then absent and the globe draws no
        # ellipsoid, rather than a default one an operator would read as measured.
        cov_block = object_covariance_block(
            getattr(parsed.secondary, "cov_eci_pos_m2", None)
        )
        objects[key] = {
            "secondary_norad": norad,
            "secondary_designator": row.get("secondary_designator"),
            "name": row.get("secondary_name"),
            "risk_level": row["risk_level"],
            "pc": row.get("pc"),
            "pc_display": row.get("pc_display"),
            "miss_distance_m": row.get("miss_distance_m"),
            "tca_utc": row.get("tca_utc"),
            "cdm_id": row.get("cdm_id"),
            "event_id": row.get("event_id"),
            # The CDM's own epoch for this object's state, so a consumer can see
            # that rings are not drawn at one common time.
            "epoch_utc": row.get("tca_utc"),
            "state_source": "leolabs_cdm_sat2",
        }
        if cov_block is not None:
            objects[key]["cov"] = cov_block

    # The worst conjunction is simply the first drawn object, since rows arrive
    # highest Pc first. Named explicitly so the globe does not have to re-sort.
    worst_key = next(iter(object_tracks), None)

    # SCRUM-448: one asset ellipsoid, from the SAT1 block of the worst row. Every
    # row carries a SAT1 covariance for the same asset, so taking the worst row's
    # is a choice of epoch rather than of object -- the same epoch caveat the
    # rings already carry, and asset.epoch_utc already states it.
    asset_cov = None
    if worst_row_parsed is not None:
        asset_cov = object_covariance_block(
            getattr(worst_row_parsed.primary, "cov_eci_pos_m2", None)
        )

    min_tca, max_tca = leolabs_runtime.conjunction_window(
        now, lookback_days, lookahead_days
    )
    log.info(
        "LeoLabs globe orbits served",
        extra={"event": "leolabs_globe_served", "primary_norad": primary_norad,
               "catalog": catalog, "objects": len(object_tracks),
               "skipped": len(skipped), "in_volume": page.in_volume,
               "total_events": page.total},
    )
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "primary_norad": int(primary_norad),
            "source": "leolabs",
            "asset": {
                "norad": int(primary_norad),
                "catalog_number": catalog,
                "epoch_utc": asset_epoch,
                "state_source": "leolabs_get_states",
                **({"cov": asset_cov} if asset_cov is not None else {}),
                # The asset's state is live from get_states, but its covariance
                # comes from a CDM's SAT1 block, so the two are not the same
                # epoch. Said here rather than left to be inferred.
                **({"cov_epoch_utc": (
                    objects.get(worst_key, {}).get("epoch_utc") if worst_key else None
                )} if asset_cov is not None else {}),
            },
            # The three fields the renderer consumes.
            "asset_track": asset_track,
            "object_tracks": object_tracks,
            "risk": risk,
            # Everything else is for labels, the worst-conjunction link, and
            # honesty about what was left out.
            "objects": objects,
            "worst_object": worst_key,
            "counts": {
                "drawn": len(object_tracks),
                "skipped": len(skipped),
                "events_in_window": page.total,
            },
            # SCRUM-459: same honesty as the 2D list. events_in_window counts a
            # PREFIX of the window when the pull was truncated, so a globe drawn from
            # it is a partial sky. Saying so matters more here than in the table: an
            # operator reads an uncluttered globe as a quiet sky, and this is the one
            # asset dense enough to be truncated.
            "complete": page.complete,
            "partial": not page.complete,
            "truncation": {
                "cdms_pulled": page.pulled_cdms,
                "cdms_in_window": page.window_cdm_total,
                "truncated_by": page.truncation_reason,
            } if not page.complete else None,
            "skipped": skipped,
            "volume_filter": {
                "in_volume": page.in_volume,
                "max_relative_position_r_m": page.volume_filters.get(
                    "maxRelativePositionR"
                ),
                "max_relative_position_i_m": page.volume_filters.get(
                    "maxRelativePositionI"
                ),
                "max_relative_position_c_m": page.volume_filters.get(
                    "maxRelativePositionC"
                ),
            },
            "window": {
                "min_tca_utc": min_tca,
                "max_tca_utc": max_tca,
                "lookback_days": lookback_days,
                "lookahead_days": lookahead_days,
            },
            "fetched_at_utc": now.isoformat().replace("+00:00", "Z"),
        },
    )


# ---------------------------------------------------------------------------
# Single conjunction (unchanged from 9.6 except logging converted to JSON)
# ---------------------------------------------------------------------------

@svc.post("/v1/evaluate")
async def post_evaluate(request: Request):
    """Evaluate a single conjunction event.

    9.6: calls evaluate_conjunction_v25() which returns a ManeuverScoringResult,
    then builds and emits ATLASManeuverArtifact in the response. The v2.4
    response fields are preserved unchanged for backward compatibility.
    atlas_artifact is additive and its build failure never affects the core
    recommendation.
    """
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse(
            status_code=422,
            content=error_response("Invalid JSON body"),
        )

    # --- LeoLabs live conjunction fetch (SCRUM-412) ----------------------
    # When LEOLABS_ENABLED=true, LeoLabs is the authoritative conjunction source
    # and takes precedence over UDL: LeoLabs replaces the UDL route, which could
    # not supply covariance. We fetch the highest-risk LeoLabs CDM for the
    # requested primary, resolve our object via the asset registry, parse it, and
    # rebuild the request from the parsed CDM (real per-object covariance rotated
    # to ECI, source=leolabs). Operator inputs (sat_id, v_remaining, burn time,
    # policy) are carried from the incoming request.
    leolabs_used = False
    if LEOLABS_ENABLED:
        ll_primary = (body.get("conjunction") or {}).get("primary_norad")
        # SCRUM-420: primary_norad is the switch between the two request modes.
        # Present -> authoritative live LeoLabs fetch (below). Absent -> the caller
        # supplied a self-contained conjunction block (e.g. the dashboard scenario
        # presets, which carry their own surrogate covariance); fall through and
        # evaluate that block directly. Enabling LeoLabs must not disable the
        # surrogate/scenario path, so a missing primary_norad is not a 422 here --
        # it simply means "evaluate what I gave you" and the response reports
        # source=surrogate rather than leolabs.
        #
        # SCRUM-432: use_stored_cdm is the third mode, and the only one that can
        # name a NORAD pair without asking LeoLabs about it. A caller that has
        # already put a CDM in the store -- the TIROS 4 reference CDM the E2E
        # smoke test injects -- needs the pair to resolve that row through the
        # covariance adapter, but TIROS 4 is not a subscribed LeoLabs asset, so
        # the live fetch can only fail it. Opt-in and absent from every existing
        # caller, so the live path below is reached on exactly the requests it
        # was reached on before: an unsubscribed NORAD still 503s rather than
        # silently scoring on whatever else happens to be available.
        _use_stored_cdm = bool(
            (body.get("conjunction") or {}).get("use_stored_cdm")
        )
        if ll_primary and not _use_stored_cdm:
            # SCRUM-422: an operator clicking a row in the live Active
            # Conjunctions table sends that row's selector back, and we score
            # that conjunction instead of whatever is currently top of the list.
            # With no selector this is byte-for-byte the SCRUM-412 call, so the
            # existing single-evaluate path is untouched rather than merely
            # equivalent.
            ll_selector = selector_from_conjunction_block(
                body.get("conjunction") or {}
            )
            try:
                # SCRUM-460: a clicked row carries its own cdm_id, and that names
                # one message, so it can be fetched directly -- two requests -- with
                # no whole-window pull. For SWARM B the window is 75,257 CDMs over
                # 77 pages and 773 s, so this is the difference between an evaluate
                # that works and one that takes the planner down.
                #
                # Everything else still needs the window: an event_id or a
                # secondary_norad selector, and the no-selector "evaluate the worst"
                # path, have nothing narrower to search on. Those keep the
                # whole-window search, and the point of running every branch through
                # _run_list_fetch is that even the slow one can no longer stop
                # /health or the other assets.
                if ll_selector.get("cdm_id") not in (None, ""):
                    parsed_ll = await _run_list_fetch(
                        fetch_leolabs_conjunction_by_cdm_id,
                        ll_selector["cdm_id"],
                        int(ll_primary),
                    )
                    if parsed_ll is None:
                        log.info(
                            "LeoLabs selector matched no conjunction",
                            extra={"event": "leolabs_selector_no_match",
                                   "primary_norad": ll_primary,
                                   "selector": ll_selector},
                        )
                        return JSONResponse(
                            status_code=404,
                            content=error_response(
                                f"no LeoLabs conjunction matches "
                                f"{ll_selector}; re-read /v1/leolabs/conjunctions"
                            ),
                        )
                elif ll_selector:
                    parsed_ll = select_conjunction(
                        await _run_list_fetch(
                            fetch_leolabs_conjunctions, int(ll_primary)
                        ),
                        ll_selector,
                    )
                    if parsed_ll is None:
                        # The row is gone from the window, or never existed.
                        # Silently scoring the highest-risk conjunction instead
                        # would label another object's numbers as the one the
                        # operator clicked, so this is a 404 and the dashboard
                        # re-reads the list.
                        log.info(
                            "LeoLabs selector matched no conjunction",
                            extra={"event": "leolabs_selector_no_match",
                                   "primary_norad": ll_primary,
                                   "selector": ll_selector},
                        )
                        return JSONResponse(
                            status_code=404,
                            content=error_response(
                                f"no LeoLabs conjunction in the current window matches "
                                f"{ll_selector}; re-read /v1/leolabs/conjunctions"
                            ),
                        )
                else:
                    parsed_ll = await _run_list_fetch(
                        fetch_leolabs_conjunction, int(ll_primary)
                    )
            except ListFetchBusy as exc:
                # The same cap and the same typed shed as the list (SCRUM-459). A
                # request that queues behind two window pulls has already spent its
                # own budget, and a busy planner must not look like a planner with
                # nothing to report.
                log.info(
                    "shedding a LeoLabs evaluate fetch; the in-flight cap is taken",
                    extra={"event": "leolabs_evaluate_shed",
                           "primary_norad": ll_primary,
                           "in_flight": _list_inflight},
                )
                return JSONResponse(
                    status_code=503,
                    content=error_response(
                        f"The planner is already pulling {_LIST_MAX_INFLIGHT} "
                        f"conjunction windows. Busy, not an empty sky: retry in a "
                        f"few seconds. ({exc})"
                    ),
                )
            except Exception as exc:
                log.warning(
                    "LeoLabs fetch failed",
                    extra={"event": "leolabs_fetch_failed", "primary_norad": ll_primary,
                           "exc": str(exc)},
                )
                return JSONResponse(
                    status_code=503,
                    content=error_response(f"LeoLabs fetch failed: {exc}"),
                )
            if parsed_ll is None:
                log.info(
                    "LeoLabs returned no scorable conjunctions",
                    extra={"event": "leolabs_no_conjunctions", "primary_norad": ll_primary},
                )
                return JSONResponse(
                    status_code=200,
                    content={
                        "conjunction_id": None,
                        "recommendation": {"direction": "no_maneuver_needed"},
                        "source": "leolabs",
                        "note": "LeoLabs returned no scorable conjunctions in the window. No maneuver needed.",
                    },
                )
            sat_in = body.get("satellite", {}) or {}
            pol_in = body.get("policy", {}) or {}
            body = build_evaluate_request(
                parsed_ll,
                sat_id=str(sat_in.get("sat_id", ll_primary)),
                v_remaining_m_s=float(sat_in.get("v_remaining_m_s", 0.0)),
                t_burn_utc=sat_in.get("t_burn_utc"),
                a_ref_km=sat_in.get("a_ref_km"),
                policy=pol_in or None,
                conjunction_id=body.get("conjunction_id"),
            )
            leolabs_used = True
    # ----------------------------------------------------------------------

    # --- UDL live conjunction fetch (SCRUM-331 AC5) ----------------------
    # When UDL_ENABLED=true, UDL is the authoritative conjunction source.
    # The injected body's conjunction block is replaced with the highest-risk
    # UDL record (highest Pc, earliest TCA as tie-break). Satellite state stays
    # from the request body. If UDL returns no records, treat as no-maneuver
    # needed -- do not fall back to injected data (would mislabel the source).
    udl_record_id: str | None = None
    if UDL_ENABLED and not leolabs_used:
        global _udl_last_fetch_utc
        primary_norad_udl = (body.get("conjunction") or {}).get("primary_norad")
        if not primary_norad_udl:
            return JSONResponse(
                status_code=422,
                content=error_response(
                    "UDL_ENABLED=true requires primary_norad in the conjunction block"
                ),
            )
        try:
            udl_records = get_conjunctions(sat_no=int(primary_norad_udl))
            if not udl_records:
                log.info(
                    "UDL returned no conjunction records above threshold",
                    extra={"event": "udl_no_conjunctions", "primary_norad": primary_norad_udl},
                )
                return JSONResponse(
                    status_code=200,
                    content={
                        "conjunction_id": None,
                        "recommendation": {"direction": "no_maneuver_needed"},
                        "covariance_source": "UDL",
                        "source": "udl",
                        "udl_record_id": None,
                        "note": "UDL returned no conjunction records above threshold. No maneuver needed.",
                    },
                )
            # Select highest-risk record: highest Pc, earliest TCA as tie-break.
            best = min(
                udl_records,
                key=lambda r: (-(r.get("pc_precomputed") or 0.0), r.get("t_ca_utc") or ""),
            )
            udl_record_id = best.get("obj_id") or best.get("conjunction_id")
            # Replace conjunction block fields from UDL; keep satellite state.
            body["conjunction"].update(best)
            body["conjunction"]["covariance_source"] = "UDL"
            from datetime import datetime, timezone
            _udl_last_fetch_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            log.info(
                "UDL conjunction selected",
                extra={
                    "event": "udl_conjunction_selected",
                    "primary_norad": primary_norad_udl,
                    "udl_record_id": udl_record_id,
                },
            )
        except Exception as exc:
            log.warning(
                "UDL fetch failed",
                extra={"event": "udl_fetch_failed", "exc": str(exc)},
            )
            return JSONResponse(
                status_code=503,
                content=error_response(f"UDL fetch failed: {exc}"),
            )
    # ----------------------------------------------------------------------

    # --- Covariance adapter (ADR-008) -------------------------------------
    # When UDL_ENABLED=true, UDL already supplied the covariance in the block
    # above. Skip the ingest CDM fetch so it cannot overwrite the UDL source.
    _conj_block = body.get("conjunction", {}) or {}
    _is_leolabs = str(_conj_block.get("source", "")).lower() == "leolabs"
    # SCRUM-454: set on the LeoLabs path, where a parsed CDM exists.
    # SCRUM-453: and on the stored/reference path, when ingest supplies the
    # per-object block. None otherwise, which selects the combined stand-in.
    _cdm_primary_cov_km2 = None
    if leolabs_used:
        # SCRUM-412: the LeoLabs fetch above rebuilt the request from a parsed
        # CDM, so the real per-object covariance is already in the block. Keep it
        # and take precedence over UDL when both flags are set.
        covariance_source = _conj_block.get("covariance_source") or "real_cdm"
        body["conjunction"]["covariance_source"] = covariance_source
        # SCRUM-429: persist the CDM this decision is being made from, and keep
        # its row id. Without this the store's only feeder was the injected
        # reference CDM, so no real conjunction the system acted on reached
        # ADR-008's record of truth -- and the audit write below, guarded on
        # cdm_record_id, never fired for a LeoLabs decision. The covariance is
        # untouched: it still comes from the block, still reads real_cdm. This
        # only writes the CDM down and remembers where.
        cdm_record_id = _persist_leolabs_cdm(parsed_ll)
        # SCRUM-454: the asset's OWN pre-burn position covariance, as distinct
        # from the combined relative one the scorer needs. Both are in this same
        # CDM and the parser has already separated them -- primary resolved
        # against our catalog id rather than by assuming SAT1 -- so the screening
        # seed can stop growing an over-estimate. km^2, to match the seed units.
        try:
            _cdm_primary_cov_km2 = [
                [float(c) / 1.0e6 for c in row]
                for row in parsed_ll.primary.cov_eci_pos_m2
            ]
        except Exception as exc:
            _cdm_primary_cov_km2 = None
            log.info(
                "primary-only covariance unavailable from the parsed CDM",
                extra={"event": "cdm_primary_covariance_unavailable",
                       "exc": str(exc)},
            )
    elif UDL_ENABLED:
        covariance_source = "UDL"
        cdm_record_id = None
    elif _is_leolabs and _conj_block.get("p_rel_km2"):
        # SCRUM-411 AC7: a request assembled from a parsed LeoLabs CDM already
        # carries the real per-object covariance rotated to ECI (km^2). Do NOT
        # run the ingest covariance adapter, which would overwrite it with an
        # unrelated Space-Track fetch or a surrogate. Trust the supplied matrix,
        # same as the UDL path trusts its own. covariance_source stays real_cdm.
        covariance_source = _conj_block.get("covariance_source") or "real_cdm"
        body["conjunction"]["covariance_source"] = covariance_source
        cdm_record_id = None
    else:
        try:
            conj = body.get("conjunction", {})
            sat = body.get("satellite", {})
            primary_norad = conj.get("primary_norad")
            secondary_norad = str(conj.get("secondary_norad") or conj.get("obj_id", ""))
            r_sat_km = sat.get("r_sat_km", [])
            v_sat_km_s = sat.get("v_sat_km_s", [])

            p_rel_km2, covariance_source, cdm_record_id, _primary_cov = (
                _fetch_cdm_covariance(
                    str(primary_norad) if primary_norad else "",
                    secondary_norad,
                    r_sat_km,
                    v_sat_km_s,
                )
            )
            body["conjunction"]["p_rel_km2"] = p_rel_km2
            body["conjunction"]["covariance_source"] = covariance_source
            # SCRUM-453: the stored and reference path can now seed the screen
            # from the primary's own covariance, like the live path. The
            # seed-selection block below already prefers _cdm_primary_cov_km2 and
            # labels it cdm_primary_own; until ingest exposed the per-object
            # block there was nothing to put here, so the stored path always fell
            # through to the combined stand-in. None when ingest did not supply
            # it, which keeps that fallback exactly as it was.
            _cdm_primary_cov_km2 = _primary_cov
        except Exception as exc:
            log.warning("covariance adapter error", extra={"event": "covariance_adapter_error", "exc": str(exc)})
            covariance_source = "surrogate_elliptical"
            cdm_record_id = None
            fallback_p_rel_km2, _, _ = _surrogate_covariance(
                body.get("satellite", {}).get("r_sat_km", [0.0, 0.0, 0.0]),
                body.get("satellite", {}).get("v_sat_km_s", [0.0, 0.0, 0.0]),
            )
            body["conjunction"]["p_rel_km2"] = fallback_p_rel_km2
            body["conjunction"]["covariance_source"] = covariance_source
    # ----------------------------------------------------------------------

    try:
        scoring = evaluate_conjunction_v25(body)

        # SCRUM-485: this block is the commanded burn a caller acts on, and it
        # is built from the scoring directly rather than from the artifact, so
        # gating the artifact alone would still have handed back a capped burn
        # on an event with no usable Pc. The scorer prices a candidate for every
        # event, including those priced on the unbounded delta_c_legacy
        # fallback; APS must not command one of those.
        #
        # Same gate and same helpers as the artifact, so the two cannot drift.
        # Flight-rule modes are untouched: they never consulted the basis.
        _eval_policy = _policy_from_dict(body.get("policy", {}))
        _eval_pc_usable = aps_pc_usable(scoring.pc_pre)
        _eval_required = _eval_policy.is_recommendation_required(
            scoring.pc_pre,
            float((body.get("conjunction") or {}).get("miss_distance_km") or 999.0),
            utility=scoring.utility,
            pc_usable=_eval_pc_usable,
        )
        _eval_aps_state = (
            aps_decision_state(scoring.pc_pre, bool(_eval_required))
            if _eval_policy.decision_mode == "aps"
            else ""
        )
        _aps_refuses_burn = (
            _eval_policy.decision_mode == "aps"
            and not _eval_pc_usable
            and scoring.is_maneuver_recommended()
        )
        # SCRUM-488: a flight-rule mode fires on the proximity floor regardless
        # of Pc, so a zero-Pc event under a kilometre legitimately requires a
        # maneuver -- and the delta-v reported beside it came from the same
        # unbounded fallback candidate, which means it is the policy's largest
        # permitted burn rather than anything sized against the event.
        #
        # The A/B marks such a burn unpriced. This response cannot simply null
        # it: the authorization path builds its execute payload from the scoring
        # directly, not from this block, so nulling here would make the response
        # disagree with what the system would actually fly. Sizing the burn
        # properly is the follow-on; until then the number is reported as the
        # scorer produced it and labelled as unpriced, so a reader is told it
        # was never sized rather than being quietly handed the cap.
        _eval_dv_unpriced = (
            not _aps_refuses_burn
            and not _eval_pc_usable
            and scoring.is_maneuver_recommended()
        )

        result: Dict[str, Any] = {
            "conjunction_id": scoring.conjunction_id,
            "recommendation": {
                "direction":        "no-burn" if _aps_refuses_burn else scoring.direction,
                "dv_eci_km_s":      [0.0, 0.0, 0.0] if _aps_refuses_burn else scoring.dv_eci_km_s,
                "dv_magnitude_m_s": 0.0 if _aps_refuses_burn else scoring.dv_magnitude_m_s,
                "t_burn_utc":       scoring.t_burn_utc,
                "utility":          scoring.utility,
            },
            "metrics": {
                "delta_C":             scoring.delta_C,
                "m2_pre":              scoring.m2_pre,
                "m2_post":             scoring.m2_post,
                "fuel_cost_m_s":       scoring.fuel_cost_m_s,
                "lifetime_penalty":    scoring.lifetime_penalty,
                # SCRUM-396: expose the resolved pre-maneuver Pc and the
                # provenance/risk-gate values that travel with it.
                "pc_pre":              scoring.pc_pre,
                "pc_source":           scoring.pc_source,
                # SCRUM-485: which basis the utility came from, and what APS
                # concluded. Without these a reader sees a huge utility beside a
                # zero burn and has no way to tell why.
                "utility_basis":       scoring.utility_basis,
                "aps_state":           _eval_aps_state,
                # SCRUM-488: true when the reported delta-v came from the
                # unbounded fallback basis and so was never sized against this
                # event. The maneuver verdict beside it is unaffected.
                "dv_unpriced":         bool(_eval_dv_unpriced),
                "hbr_m":               scoring.hbr_m,
                "risk_gate":           scoring.risk_gate,
                "risk_surrogate_post": scoring.risk_surrogate_post,
                # SCRUM-393: risk_surrogate_post has carried three different
                # quantities over its life and its name says none of them, so
                # the source travels with the number rather than leaving a
                # consumer to guess from the magnitude.
                "risk_surrogate_source": scoring.risk_surrogate_source,
                "pc_post": scoring.pc_post,
                "all_candidates":      scoring.all_candidates,
            },
            "covariance_source": covariance_source,
            # SCRUM-411 AC6: the conjunction data source that drove this evaluate,
            # distinct from covariance_source. Lights the dashboard LIVE badge.
            "source":            _resolve_evaluate_source(
                body, covariance_source, bool(udl_record_id)
            ),
            "udl_record_id":     udl_record_id,
            "evaluated_at":      scoring.evaluated_at,
        }

        detection_confidence = None
        conj_input = body.get("conjunction", {})

        if isinstance(conj_input, dict):
            detection_confidence = conj_input.get("detection_confidence")

            if detection_confidence is None:
                confidence_keys = (
                    "estimated_range_km",
                    "debris_size_class",
                    "range_confidence",
                )
                if any(key in conj_input for key in confidence_keys):
                    detection_confidence = {
                        key: conj_input.get(key)
                        for key in confidence_keys
                        if key in conj_input
                    }

        if detection_confidence is None:
            detection_confidence = body.get("detection_confidence")

        if detection_confidence is not None:
            result["detection_confidence"] = detection_confidence

        # --- 9.6 + SCRUM-330: ATLASManeuverArtifact + secondary conflict ----
        # Catalog retrieval remains best-effort at this layer. SCRUM-381
        # treats an unavailable/incomplete catalog as NOT CLEAR and fails closed.
        try:
            conj_dict = body.get("conjunction", {})
            sat_dict  = body.get("satellite", {})

            cap     = SatelliteCapability.from_request(sat_dict)
            policy  = _policy_from_dict(body.get("policy", {}))

            # SCRUM-431: the Space-Track TLE catalog fetch is gone with the
            # credential, so nothing fetches a catalog here.
            #
            # SCRUM-455: this comment used to say the screen was "deferred behind
            # SECONDARY_SCREEN_ENABLED (default off) pending the LeoLabs
            # covariance-backed rebuild". That rebuild has landed -- the screen is
            # the live LeoLabs on-demand path (SCRUM-440/441/451), it runs in the
            # background (SCRUM-456) and the dashboard resolves it (SCRUM-457). The
            # default is off, but as a fail-safe backstop rather than a deferral:
            # .env is the control, forwarded by both composes.
            # r_post_km is still approximated as r_sat_km -- position barely
            # changes during a short avoidance burn, only velocity does -- and
            # the A4 post-maneuver projection still uses it.
            r_sat_km_req = sat_dict.get("r_sat_km", [])
            known_objects = None
            v_sat_km_s_req = sat_dict.get("v_sat_km_s", [])
            v_post_km_s = None
            if v_sat_km_s_req and scoring.dv_eci_km_s:
                v_post_km_s = [
                    float(v_sat_km_s_req[i]) + float(scoring.dv_eci_km_s[i])
                    for i in range(3)
                ]

            # SCRUM-452: the seed for the screening ephemeris covariance.
            #
            # The ephemeris wants the PRIMARY's own post-burn position
            # covariance. post_burn_covariance_km2 is the single source of truth
            # for that -- it prefers GNC's own p_post_km2, else rebuilds
            # P_pre + P_burn frame-consistently (SCRUM-428), else returns None
            # meaning unknown.
            #
            # It takes a GNC report, and /v1/evaluate is not the GNC endpoint, so
            # it is used when the caller supplies a report-shaped block and not
            # otherwise. Field names are the ones POST /v1/gnc/report already
            # uses, so a caller has one shape to learn rather than two.
            _p_post = None
            _p_post_source = None
            _gnc = body.get("gnc_report") or body.get("gnc") or {}
            if isinstance(_gnc, dict) and _gnc:
                try:
                    _risk = _gnc.get("aps_risk_context") or {}
                    _m = post_burn_covariance_km2(
                        _gnc, p_pre_km2=_risk.get("p_pre_km2"))
                    if _m is not None:
                        _p_post = [[float(c) for c in row] for row in _m]
                        _p_post_source = "gnc_post_burn_covariance"
                except Exception as exc:
                    log.info(
                        "post-burn covariance could not be read from the GNC report",
                        extra={"event": "post_burn_covariance_unavailable",
                               "exc": str(exc)},
                    )

            if _p_post is None and _cdm_primary_cov_km2 is not None:
                # SCRUM-454: the asset's own covariance out of the same CDM this
                # decision is being made from.
                #
                # This is a PRE-burn covariance, and naming it honestly matters.
                # The CDM block is P_pre, the SCRUM-452 execution-error block is
                # what the burn adds, and the two grown together are the
                # post-burn covariance -- the same P_pre + P_burn structure
                # post_burn_covariance_km2 formalises, with P_pre coming from the
                # CDM here rather than from a GNC assessment.
                _p_post = _cdm_primary_cov_km2
                _p_post_source = "cdm_primary_own"

            if _p_post is None:
                # No GNC report and no parsed CDM. p_rel_km2 is the COMBINED
                # relative covariance, so as a primary-only seed it is an
                # over-estimate -- conservative for a screen, but not the right
                # quantity. Kept as a fallback rather than failing the screen
                # closed, and labelled so the difference is visible in the record
                # instead of implied.
                _p_rel = (body.get("conjunction") or {}).get("p_rel_km2")
                if _p_rel:
                    try:
                        _flat = [float(c) for c in _p_rel]
                        if len(_flat) == 9:
                            _p_post = [_flat[0:3], _flat[3:6], _flat[6:9]]
                            _p_post_source = "combined_relative_stand_in"
                    except (TypeError, ValueError):
                        _p_post = None

            # The screen is keyed on the LeoLabs catalog number, not the NORAD.
            _screen_catalog = None
            if SECONDARY_SCREEN_ENABLED:
                _ll_norad = (body.get("conjunction") or {}).get("primary_norad")
                if _ll_norad:
                    try:
                        _screen_catalog = catalog_for_norad(int(_ll_norad))
                    except Exception as exc:
                        log.info(
                            "secondary screen has no catalog number for the primary",
                            extra={"event": "secondary_screen_no_catalog",
                                   "primary_norad": _ll_norad, "exc": str(exc)},
                        )

            if SECONDARY_SCREEN_ENABLED and _p_post is not None:
                # Which quantity seeded the screen, at the point it is chosen.
                # Without this the only record of it is the persisted screening
                # row, and that write depends on an ingest endpoint that does not
                # exist yet (SCRUM-442), so the seed source would be invisible in
                # a running system.
                log.info(
                    "screening covariance seed selected",
                    extra={"event": "screening_seed_selected",
                           "p_post_source": _p_post_source},
                )

            _screening_capture: Dict[str, Any] = {}

            def _capture_screening(result, verdict):
                """Hold the screening result so it can be persisted with the decision."""
                _screening_capture["result"] = result
                _screening_capture["verdict"] = verdict
                _screening_capture["p_post_source"] = _p_post_source

            artifact = build_atlas_artifact(
                scoring=scoring,
                cap=cap,
                policy=policy,
                tca_utc=conj_dict.get("t_ca_utc", ""),
                pc_precomputed=conj_dict.get("pc_precomputed"),
                miss_distance_km=conj_dict.get("miss_distance_km"),
                known_objects=known_objects if known_objects else None,
                r_post_km=r_sat_km_req if r_sat_km_req else None,
                v_post_km_s=v_post_km_s,
                secondary_screen_enabled=SECONDARY_SCREEN_ENABLED,
                p_post_eci_km2=_p_post,
                primary_catalog_number=_screen_catalog,
                screening_sink=_capture_screening,
                # SCRUM-456: the screen runs in the background, so the sink above
                # fires only on the inline path. The seed provenance goes onto the
                # store entry instead, where the poll endpoint reports it.
                p_post_source=_p_post_source,
            )
            result["atlas_artifact"] = artifact.to_dict()
            # Read once: both the decision-log block and the SCRUM-379 block
            # below need it, and the second must not depend on the first having
            # got that far.
            _pol = body.get("policy", {})
            # --- DecisionLog (SCRUM-351): full audit record, retrievable by log_id ---
            try:
                decision_log = DecisionLog.from_artifact(
                    artifact,
                    operator_id=str(_pol.get("operator_id", "")),
                    policy_version=str(_pol.get("policy_version", "")),
                )
                result["decision_log_id"] = decision_log.log_id
                result["decision_log"] = vars(decision_log)
                _post_decision_log(decision_log)
                # SCRUM-442: link the screening the verdict was made on to the
                # decision, the way SCRUM-429 links a decision to its stored CDM
                # under ADR-008. A NOT CLEAR that cannot be traced back to the
                # events behind it is an assertion, not an audit record.
                _sc_id = _persist_screening_result(
                    _screening_capture.get("result"),
                    _screening_capture.get("verdict"),
                    decision_log.log_id,
                    p_post_source=_screening_capture.get("p_post_source"),
                )
                if _sc_id:
                    result["screening_record_id"] = _sc_id
                # SCRUM-484: the inline persist above fires only when the screen
                # ran inline. On the default async path the screen is still
                # running here, so there is no result to write yet -- which is
                # why a live decision returned screening_record_id null and no
                # record was ever stored. Stamp the decision onto the screen's
                # store entry instead; the poll endpoint persists once the screen
                # resolves, when both halves finally exist.
                _stamp_async_screen_decision(artifact, decision_log.log_id)
                _ev_id = _post_evidence_record(artifact, decision_log, _pol)
                if _ev_id:
                    result["evidence_record_id"] = _ev_id
            except Exception as exc:
                log.warning("decision log build failed", extra={"event": "decision_log_build_failed", "exc": str(exc)})

            # --- SCRUM-379: MAF section 8 decision state machine ------------
            # Additive, exactly like the artifact and decision-log blocks above:
            # a monitor failure is logged and the core evaluate still returns.
            # Fail-closed lives inside the monitor, not here -- a request that
            # supplies no IOD verdict and no observation arc gets a decision
            # that declines to stage, not a missing decision.
            try:
                monitor = evaluate_decision_state_machine(
                    body=body,
                    scoring=scoring,
                    artifact=artifact,
                    policy=policy,
                    cap=cap,
                    covariance_source=covariance_source,
                    store=_MODE_STORE,
                    software_version=SERVICE_VERSION,
                )
                result["decision_state_machine"] = monitor.to_dict()
                if monitor.authorized_execution is not None:
                    # The signal SCRUM-382 consumes to assemble the GNC command.
                    result["authorized_execution"] = monitor.authorized_execution.to_dict()
                    # SCRUM-382: build and emit. Nested try/except of its own so
                    # a GNC problem cannot cost us the decision record either --
                    # the authorisation stands whether or not it reached GNC.
                    try:
                        _emit_to_gnc(result, monitor, scoring, artifact, sat_dict)
                    except Exception as exc:
                        log.warning(
                            "GNC emission failed",
                            extra={"event": "gnc_emission_failed",
                                   "conjunction_id": scoring.conjunction_id,
                                   "exc": str(exc)},
                        )
                _post_transition_record(artifact, monitor, _pol)
                log.info(
                    "decision state machine evaluated",
                    extra={
                        "event": "decision_state_machine_evaluated",
                        "conjunction_id": scoring.conjunction_id,
                        "from_mode": monitor.transition.from_mode.value,
                        "to_mode": monitor.mode.value,
                        "trigger": monitor.transition.trigger,
                        "escalated": monitor.escalated,
                    },
                )
            except Exception as exc:
                log.warning(
                    "decision state machine evaluation failed",
                    extra={"event": "decision_state_machine_failed", "exc": str(exc)},
                )
            # ----------------------------------------------------------------

            log.info("atlas artifact built", extra={"event": "artifact_built", "conjunction_id": scoring.conjunction_id, "summary": artifact.operator_summary()})
        except Exception as exc:
            log.warning("atlas artifact build failed", extra={"event": "artifact_build_failed", "conjunction_id": body.get("conjunction_id", "?"), "exc": str(exc)})
        # ------------------------------------------------------------------

        # --- Audit write --------------------------------------------------
        if cdm_record_id is not None:
            _post_planner_output(cdm_record_id, result, body, covariance_source, scoring)
        # ------------------------------------------------------------------

        # SCRUM-417: on the live LeoLabs path, attach the scored conjunction's
        # identity and RTN relative geometry so the UI can name the secondary and
        # draw the true encounter. Additive, live path only.
        if leolabs_used:
            result["conjunction"] = parsed_ll.to_response_conjunction()

        return JSONResponse(status_code=200, content=result)

    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content=error_response(str(exc)),
        )
    except Exception as exc:
        log.error("evaluate error", extra={"event": "evaluate_error", "exc": str(exc)})
        return JSONResponse(
            status_code=500,
            content=error_response("Internal error: " + str(exc)),
        )


# ---------------------------------------------------------------------------
# Batch (unchanged from 9.6)
# ---------------------------------------------------------------------------

@svc.post("/v1/evaluate/batch")
async def post_evaluate_batch(request: Request):
    """Evaluate multiple conjunctions. Uses v2.4 path. No atlas_artifact."""
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse(
            status_code=422,
            content=error_response("Invalid JSON body"),
        )

    try:
        result = evaluate_batch(body)
        return JSONResponse(status_code=200, content=result)
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content=error_response(str(exc)),
        )
    except Exception as exc:
        log.error("evaluate batch error", extra={"event": "evaluate_batch_error", "exc": str(exc)})
        return JSONResponse(
            status_code=500,
            content=error_response("Internal error: " + str(exc)),
        )


# ---------------------------------------------------------------------------
# SCRUM-456: the asynchronous secondary screen
# ---------------------------------------------------------------------------

@svc.get("/v1/secondary-screen/{job_id}")
async def get_secondary_screen(job_id: str):
    """Poll one asynchronous on-demand secondary screen.

    Evaluate starts the screen in the background and returns the burn decision
    immediately with the secondary check PENDING and this job id (SCRUM-456),
    because the screen takes 30 s to 2 min and the dashboard's evaluate call times
    out at 60 s. This endpoint is read-only and does no LeoLabs work: it reports
    what the worker has written into the process-local store.

    Status is one of pending, clear, not_clear, error. `clear` is a derived
    boolean and is True for exactly one of those, so a pending or errored screen
    can never be read as a passed screen. A screen that errored, timed out, was
    rate-limited or had no access reports error, which is NOT CLEAR and escalates
    exactly as the inline SCRUM-442 screen did.

    404 for an unknown or evicted job id. That is also not clear: a caller that
    cannot find its screen has not been told the screen passed, and the decision
    still carries the pending check, which fails closed.
    """
    from common.secondary_screen_async import entry_to_dict, get_entry

    entry = get_entry(job_id)
    if entry is None:
        return JSONResponse(
            status_code=404,
            content=error_response(
                f"no secondary screen with id {job_id!r}; it never existed or has "
                f"been evicted. This is NOT a clear screen."
            ),
        )
    # SCRUM-484: a resolved screen is where the screening record finally can be
    # written -- the evaluate returned before the screen had a result. This is
    # the only side effect on this endpoint and it is deliberately incapable of
    # changing what the endpoint reports: the verdict is read off the entry
    # afterwards either way, so a store that is down costs the record and leaves
    # the clear / not-clear answer exactly as it was.
    await _persist_resolved_screen(entry)
    return JSONResponse(status_code=200, content=entry_to_dict(entry))
