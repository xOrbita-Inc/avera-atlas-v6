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

import json
import logging
import os
import time
from contextlib import asynccontextmanager
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
from common.maneuver_scorer import evaluate_conjunction_v25, _policy_from_dict
from common.evidence_record import (
    EvidenceRecord, RecordType, canonical_json, build_decision_record,
    GENESIS_HASH,
)
from common.atlas_artifact import build_atlas_artifact, DecisionLog
from common.satellite_capability import SatelliteCapability
from common.logging_setup import build_logger, _POLICY_CONFIG_PATH, SERVICE_NAME, SERVICE_VERSION
from common.operator_policy import OperatorPolicy, CovarianceSurrogate
from common.spacetrack_tle import fetch_catalog_objects
from common.udl_client import UDL_ENABLED, get_conjunctions, get_credential_validity
from common import leolabs_runtime
from common.leolabs_runtime import LEOLABS_ENABLED, fetch_leolabs_conjunction
from common.leolabs_evaluate import build_evaluate_request

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
) -> tuple[list, str, int | None]:
    """Fetch RTN covariance from the ingest service and rotate to ECI.

    Returns (p_rel_km2, covariance_source, cdm_record_id).

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
        return _surrogate_covariance(r_sat_km, v_sat_km_s)

    try:
        url = f"{_ingest_url()}/cdm/{primary_norad}/{secondary_norad}"
        resp = http_requests.get(url, timeout=5.0)
    except Exception as exc:
        log.warning("covariance fetch failed", extra={"event": "covariance_fetch_fail", "reason": "unreachable", "pair": f"{primary_norad}/{secondary_norad}", "exc": str(exc)})
        return _surrogate_covariance(r_sat_km, v_sat_km_s)

    if resp.status_code == 404:
        log.warning("covariance fetch failed", extra={"event": "covariance_fetch_fail", "reason": "not_found", "pair": f"{primary_norad}/{secondary_norad}"})
        return _surrogate_covariance(r_sat_km, v_sat_km_s)

    if resp.status_code == 503:
        log.warning("covariance fetch failed", extra={"event": "covariance_fetch_fail", "reason": "store_unavailable", "pair": f"{primary_norad}/{secondary_norad}"})
        return _surrogate_covariance(r_sat_km, v_sat_km_s)

    if not resp.ok:
        log.warning("covariance fetch failed", extra={"event": "covariance_fetch_fail", "reason": f"http_{resp.status_code}", "pair": f"{primary_norad}/{secondary_norad}"})
        return _surrogate_covariance(r_sat_km, v_sat_km_s)

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

        log.info("covariance fetched", extra={"event": "covariance_fetched", "pair": f"{primary_norad}/{secondary_norad}", "source": covariance_source, "cdm_id": cdm_record_id})
        return p_rel_km2, covariance_source, cdm_record_id

    except Exception as exc:
        log.warning("covariance parse failed", extra={"event": "covariance_parse_fail", "pair": f"{primary_norad}/{secondary_norad}", "exc": str(exc)})
        return _surrogate_covariance(r_sat_km, v_sat_km_s)


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
    and this particular event simply had nothing to put there, e.g. a request
    that supplied no Pc. Collapsing the two would make pending_producers()
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
    if risk.pc_pre is not None:
        values["pc_at_transition"] = risk.pc_pre
    else:
        # The producer exists; this request carried no Pc.
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
        if not ll_primary:
            return JSONResponse(
                status_code=422,
                content=error_response(
                    "LEOLABS_ENABLED=true requires primary_norad in the conjunction block"
                ),
            )
        try:
            parsed_ll = fetch_leolabs_conjunction(int(ll_primary))
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
    if leolabs_used:
        # SCRUM-412: the LeoLabs fetch above rebuilt the request from a parsed
        # CDM, so the real per-object covariance is already in the block. Keep it
        # and take precedence over UDL when both flags are set.
        covariance_source = _conj_block.get("covariance_source") or "real_cdm"
        body["conjunction"]["covariance_source"] = covariance_source
        cdm_record_id = None
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

            p_rel_km2, covariance_source, cdm_record_id = _fetch_cdm_covariance(
                str(primary_norad) if primary_norad else "",
                secondary_norad,
                r_sat_km,
                v_sat_km_s,
            )
            body["conjunction"]["p_rel_km2"] = p_rel_km2
            body["conjunction"]["covariance_source"] = covariance_source
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

        result: Dict[str, Any] = {
            "conjunction_id": scoring.conjunction_id,
            "recommendation": {
                "direction":        scoring.direction,
                "dv_eci_km_s":      scoring.dv_eci_km_s,
                "dv_magnitude_m_s": scoring.dv_magnitude_m_s,
                "t_burn_utc":       scoring.t_burn_utc,
                "utility":          scoring.utility,
            },
            "metrics": {
                "delta_C":             scoring.delta_C,
                "m2_pre":              scoring.m2_pre,
                "m2_post":             scoring.m2_post,
                "fuel_cost_m_s":       scoring.fuel_cost_m_s,
                "lifetime_penalty":    scoring.lifetime_penalty,
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
        # Catalog fetch is fire-and-forget: on any failure known_objects=[]
        # which preserves the not_performed fallback in atlas_artifact.py.
        try:
            conj_dict = body.get("conjunction", {})
            sat_dict  = body.get("satellite", {})

            cap     = SatelliteCapability.from_request(sat_dict)
            policy  = _policy_from_dict(body.get("policy", {}))

            # SCRUM-330: fetch TLE catalog for secondary conflict screening.
            # r_post_km approximated as r_sat_km -- position barely changes
            # during a short avoidance burn; only velocity changes.
            r_sat_km_req = sat_dict.get("r_sat_km", [])
            t_burn_utc   = sat_dict.get("t_burn_utc", "")
            known_objects = []
            try:
                known_objects = fetch_catalog_objects(
                    r_sat_km=r_sat_km_req,
                    burn_time_utc=t_burn_utc,
                )
                log.info(
                    "catalog screening complete",
                    extra={
                        "event": "catalog_screening_complete",
                        "nearby_count": len(known_objects),
                        "conjunction_id": scoring.conjunction_id,
                    },
                )
            except Exception as exc:
                log.warning(
                    "catalog fetch failed, secondary check will be not_performed",
                    extra={"event": "catalog_fetch_failed", "exc": str(exc)},
                )

            artifact = build_atlas_artifact(
                scoring=scoring,
                cap=cap,
                policy=policy,
                tca_utc=conj_dict.get("t_ca_utc", ""),
                pc_precomputed=conj_dict.get("pc_precomputed"),
                miss_distance_km=conj_dict.get("miss_distance_km"),
                known_objects=known_objects if known_objects else None,
                r_post_km=r_sat_km_req if r_sat_km_req else None,
            )
            result["atlas_artifact"] = artifact.to_dict()
            # --- DecisionLog (SCRUM-351): full audit record, retrievable by log_id ---
            try:
                _pol = body.get("policy", {})
                decision_log = DecisionLog.from_artifact(
                    artifact,
                    operator_id=str(_pol.get("operator_id", "")),
                    policy_version=str(_pol.get("policy_version", "")),
                )
                result["decision_log_id"] = decision_log.log_id
                result["decision_log"] = vars(decision_log)
                _post_decision_log(decision_log)
                _ev_id = _post_evidence_record(artifact, decision_log, _pol)
                if _ev_id:
                    result["evidence_record_id"] = _ev_id
            except Exception as exc:
                log.warning("decision log build failed", extra={"event": "decision_log_build_failed", "exc": str(exc)})
            log.info("atlas artifact built", extra={"event": "artifact_built", "conjunction_id": scoring.conjunction_id, "summary": artifact.operator_summary()})
        except Exception as exc:
            log.warning("atlas artifact build failed", extra={"event": "artifact_build_failed", "conjunction_id": body.get("conjunction_id", "?"), "exc": str(exc)})
        # ------------------------------------------------------------------

        # --- Audit write --------------------------------------------------
        if cdm_record_id is not None:
            _post_planner_output(cdm_record_id, result, body, covariance_source, scoring)
        # ------------------------------------------------------------------

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
