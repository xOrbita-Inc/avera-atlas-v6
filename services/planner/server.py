"""
APS Planner Service - FastAPI wrapper for decision_model.py

Thin HTTP layer that maps OpenAPI endpoints to the core decision logic.
All validation and computation lives in decision_model.py.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import numpy as np
import requests as http_requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from avoid.decision_model import (
    error_response,
    evaluate_batch,
    evaluate_conjunction,
)

logger = logging.getLogger("planner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

SERVICE_VERSION = "2.4.3"


# Ingest service base URL. Resolved at call time so tests can override via env.
def _ingest_url() -> str:
    return os.environ.get("INGEST_SERVICE_URL", "http://ingest:8000")


def _rtn_to_eci_rotation(r_km: np.ndarray, v_km_s: np.ndarray) -> np.ndarray:
    """Build the 3x3 RTN->ECI rotation matrix from an ECI state vector.

    Identical math to cdm_to_conjunction._rtn_to_eci_rotation in the ingest
    service. Duplicated here because the planner cannot import from ingest.

    Columns are the R, T, N unit vectors expressed in ECI:
      R = r_hat  (radial)
      N = (r x v) / |r x v|  (cross-track / normal)
      T = N x R  (along-track / tangential)
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

    Returns (p_rel_km2, covariance_source, cdm_record_id) where:
    - p_rel_km2 is a 9-element row-major flat list (km², ECI frame)
    - covariance_source is 'real_cdm' or 'surrogate_identity'
    - cdm_record_id is the cdm_records.id FK, or None if surrogate

    Falls back to surrogate identity matrix (0.01 * I3) on any failure.
    The surrogate value matches the pre-existing demo default.

    ADR-008 reference: the RTN->ECI rotation is the planner adapter's
    responsibility. The ingest service returns RTN frame only.
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
        logger.warning(
            "[PLANNER] Ingest service unreachable for %s/%s: %s -- using surrogate",
            primary_norad, secondary_norad, exc,
        )
        return _SURROGATE

    if resp.status_code == 404:
        logger.warning(
            "[PLANNER] No CDM found for %s/%s -- using surrogate",
            primary_norad, secondary_norad,
        )
        return _SURROGATE

    if resp.status_code == 503:
        logger.warning(
            "[PLANNER] Ingest CDM store unavailable for %s/%s -- using surrogate",
            primary_norad, secondary_norad,
        )
        return _SURROGATE

    if not resp.ok:
        logger.warning(
            "[PLANNER] Unexpected ingest response %d for %s/%s -- using surrogate",
            resp.status_code, primary_norad, secondary_norad,
        )
        return _SURROGATE

    try:
        data = resp.json()
        cov_rtn = np.array(data["covariance_combined_rtn"], dtype=float)
        covariance_source = data.get("covariance_source", "surrogate_identity")
        cdm_record_id = data.get("id")  # FK for planner_outputs audit write

        r = np.array(r_sat_km, dtype=float)
        v = np.array(v_sat_km_s, dtype=float)
        rot = _rtn_to_eci_rotation(r, v)
        cov_eci = rot @ cov_rtn @ rot.T
        p_rel_km2 = cov_eci.flatten().tolist()

        logger.info(
            "[PLANNER] CDM covariance fetched for %s/%s (source: %s, id: %s)",
            primary_norad, secondary_norad, covariance_source, cdm_record_id,
        )
        return p_rel_km2, covariance_source, cdm_record_id

    except Exception as exc:
        logger.warning(
            "[PLANNER] Failed to parse CDM covariance for %s/%s: %s -- using surrogate",
            primary_norad, secondary_norad, exc,
        )
        return _SURROGATE


def _post_planner_output(
    cdm_record_id: int,
    result: Dict[str, Any],
    body: Dict[str, Any],
    covariance_source: str,
) -> None:
    """Write a planner decision audit record to the ingest service.

    Fire-and-forget: called after a successful evaluate_conjunction().
    Any failure is logged as a WARNING and silently swallowed -- the
    planner response is never affected by audit write failures.

    Recommendation mapping (utility-based, APS v2.4):
      utility > 0  -> 'maneuver'
      utility <= 0 -> 'no_maneuver'
    The 'monitor' case is reserved for Pc-threshold logic in APS v2.5.

    Only called when cdm_record_id is not None (i.e. real CDM was used).
    """
    try:
        rec = result.get("recommendation", {})
        metrics = result.get("metrics", {})
        policy = body.get("policy", {})

        utility = float(rec.get("utility", 0.0))
        recommendation = "maneuver" if utility > 0.0 else "no_maneuver"
        delta_v_ms = float(rec.get("dv_magnitude_m_s")) if recommendation == "maneuver" else None

        # pc_computed: use risk_surrogate_post as the best available proxy.
        # This is either the Space-Track published Pc (if pc_precomputed was
        # provided) or 1/m2_post from the Mahalanobis metric.
        pc_computed = float(metrics.get("risk_surrogate_post", 0.0))

        payload = {
            "cdm_record_id":    cdm_record_id,
            "recommendation":   recommendation,
            "delta_v_ms":       delta_v_ms,
            "pc_computed":      pc_computed,
            "utility_value":    utility,
            "lambda_v":         float(policy.get("lambda_v", 0.0)),
            "lambda_l":         float(policy.get("lambda_L", 0.0)),
            "covariance_source": covariance_source,
        }

        url = f"{_ingest_url()}/planner_output"
        resp = http_requests.post(url, json=payload, timeout=5.0)

        if resp.status_code == 201:
            planner_output_id = resp.json().get("id")
            logger.info(
                "[PLANNER] Audit record written: planner_outputs.id=%s for cdm_record_id=%s",
                planner_output_id, cdm_record_id,
            )
        else:
            logger.warning(
                "[PLANNER] Audit write returned unexpected status %d for cdm_record_id=%s",
                resp.status_code, cdm_record_id,
            )

    except Exception as exc:
        logger.warning(
            "[PLANNER] Audit write failed for cdm_record_id=%s: %s",
            cdm_record_id, exc,
        )


@asynccontextmanager
async def lifespan(a):
    logger.info("planner service v%s starting", SERVICE_VERSION)
    yield
    logger.info("planner service shutting down")


svc = FastAPI(
    title="APS Planner - Avoidance Decision Model",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


# -- Health -------------------------------------------------------------------

@svc.get("/health")
async def health():
    return {"status": "ok", "version": SERVICE_VERSION}


# -- Single conjunction -------------------------------------------------------

@svc.post("/v1/evaluate")
async def post_evaluate(request: Request):
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
        # Use secondary_norad if explicitly provided (real NORAD ID),
        # otherwise fall back to obj_id (may be synthetic scenario ID).
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
            logger.debug(
                "[PLANNER] primary_norad not provided -- using surrogate covariance"
            )
    except Exception as exc:
        logger.warning("[PLANNER] Covariance adapter error: %s -- using surrogate", exc)
    # ----------------------------------------------------------------------

    try:
        result = evaluate_conjunction(body)
        result["covariance_source"] = covariance_source

        # --- Audit write (ADR-008 Prompt 5) --------------------------------
        # Fire-and-forget: only write when a real CDM record was used.
        if cdm_record_id is not None:
            _post_planner_output(cdm_record_id, result, body, covariance_source)
        # -------------------------------------------------------------------

        return JSONResponse(status_code=200, content=result)
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content=error_response(str(exc)),
        )
    except Exception as exc:
        logger.exception("Unexpected error in /v1/evaluate")
        return JSONResponse(
            status_code=500,
            content=error_response("Internal error: " + str(exc)),
        )


# -- Batch --------------------------------------------------------------------

@svc.post("/v1/evaluate/batch")
async def post_evaluate_batch(request: Request):
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
        logger.exception("Unexpected error in /v1/evaluate/batch")
        return JSONResponse(
            status_code=500,
            content=error_response("Internal error: " + str(exc)),
        )
