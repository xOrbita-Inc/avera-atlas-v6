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
from typing import Any, Dict

import numpy as np
import requests as http_requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from avoid.decision_model import (
    error_response,
    evaluate_batch,
)
from common.maneuver_scorer import evaluate_conjunction_v25, _policy_from_dict
from common.atlas_artifact import build_atlas_artifact
from common.satellite_capability import SatelliteCapability
from common.logging_setup import build_logger, _POLICY_CONFIG_PATH, SERVICE_NAME, SERVICE_VERSION
from common.spacetrack_tle import fetch_catalog_objects

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

log = build_logger()

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

    Identical math to cdm_to_conjunction._rtn_to_eci_rotation in the ingest
    service. Duplicated here because the planner cannot import from ingest.
    """
    r_hat = r_km / np.linalg.norm(r_km)
    h = np.cross(r_km, v_km_s)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)
    return np.column_stack([r_hat, t_hat, n_hat])


def _fetch_cdm_covariance(
    primary_norad: str,
    secondary_norad: str,
    r_sat_km: list,
    v_sat_km_s: list,
) -> tuple[list, str, int | None]:
    """Fetch RTN covariance from the ingest service and rotate to ECI.

    Returns (p_rel_km2, covariance_source, cdm_record_id).
    Falls back to surrogate identity matrix on any failure.
    """
    _SURROGATE = (
        [0.01, 0.0, 0.0,
         0.0, 0.01, 0.0,
         0.0, 0.0, 0.01],
        "surrogate_identity",
        None,
    )

    try:
        url = f"{_ingest_url()}/cdm/{primary_norad}/{secondary_norad}"
        resp = http_requests.get(url, timeout=5.0)
    except Exception as exc:
        log.warning("covariance fetch failed", extra={"event": "covariance_fetch_fail", "reason": "unreachable", "pair": f"{primary_norad}/{secondary_norad}", "exc": str(exc)})
        return _SURROGATE

    if resp.status_code == 404:
        log.warning("covariance fetch failed", extra={"event": "covariance_fetch_fail", "reason": "not_found", "pair": f"{primary_norad}/{secondary_norad}"})
        return _SURROGATE

    if resp.status_code == 503:
        log.warning("covariance fetch failed", extra={"event": "covariance_fetch_fail", "reason": "store_unavailable", "pair": f"{primary_norad}/{secondary_norad}"})
        return _SURROGATE

    if not resp.ok:
        log.warning("covariance fetch failed", extra={"event": "covariance_fetch_fail", "reason": f"http_{resp.status_code}", "pair": f"{primary_norad}/{secondary_norad}"})
        return _SURROGATE

    try:
        data = resp.json()
        cov_rtn = np.array(data["covariance_combined_rtn"], dtype=float)
        covariance_source = data.get("covariance_source", "surrogate_identity")
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
        return _SURROGATE


def _post_planner_output(
    cdm_record_id: int,
    result: Dict[str, Any],
    body: Dict[str, Any],
    covariance_source: str,
) -> None:
    """Write a planner decision audit record to the ingest service.

    Fire-and-forget. Any failure is logged and silently swallowed.
    """
    try:
        rec = result.get("recommendation", {})
        metrics = result.get("metrics", {})
        policy = body.get("policy", {})

        utility = float(rec.get("utility", 0.0))
        recommendation = "maneuver" if utility > 0.0 else "no_maneuver"
        delta_v_ms = float(rec.get("dv_magnitude_m_s")) if recommendation == "maneuver" else None
        pc_computed = float(metrics.get("risk_surrogate_post", 0.0))

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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_start_time = time.time()


@asynccontextmanager
async def lifespan(a):
    log.info("service starting", extra={"event": "startup", "version": SERVICE_VERSION, "policy_config": str(_POLICY_CONFIG_PATH)})
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
        from common.operator_policy import OperatorPolicy
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

    # --- Covariance adapter (ADR-008) -------------------------------------
    covariance_source = "surrogate_identity"
    cdm_record_id = None
    try:
        conj = body.get("conjunction", {})
        sat = body.get("satellite", {})
        primary_norad = conj.get("primary_norad")
        secondary_norad = str(conj.get("secondary_norad") or conj.get("obj_id", ""))

        if primary_norad and secondary_norad:
            r_sat_km = sat.get("r_sat_km", [])
            v_sat_km_s = sat.get("v_sat_km_s", [])
            p_rel_km2, covariance_source, cdm_record_id = _fetch_cdm_covariance(
                str(primary_norad),
                secondary_norad,
                r_sat_km,
                v_sat_km_s,
            )
            body["conjunction"]["p_rel_km2"] = p_rel_km2
            body["conjunction"]["covariance_source"] = covariance_source
        else:
            log.info("using surrogate covariance", extra={"event": "surrogate_covariance", "reason": "primary_norad_not_provided"})
    except Exception as exc:
        log.warning("covariance adapter error", extra={"event": "covariance_adapter_error", "exc": str(exc)})
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
                "all_candidates":      scoring.all_candidates,
            },
            "covariance_source": covariance_source,
            "evaluated_at":      scoring.evaluated_at,
        }

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
            log.info("atlas artifact built", extra={"event": "artifact_built", "conjunction_id": scoring.conjunction_id, "summary": artifact.operator_summary()})
        except Exception as exc:
            log.warning("atlas artifact build failed", extra={"event": "artifact_build_failed", "conjunction_id": body.get("conjunction_id", "?"), "exc": str(exc)})
        # ------------------------------------------------------------------

        # --- Audit write --------------------------------------------------
        if cdm_record_id is not None:
            _post_planner_output(cdm_record_id, result, body, covariance_source)
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
