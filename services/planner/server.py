"""
APS Planner Service - FastAPI wrapper for decision_model.py

Thin HTTP layer that maps OpenAPI endpoints to the core decision logic.
All validation and computation lives in decision_model.py.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

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

SERVICE_VERSION = "2.4.0"


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

    try:
        result = evaluate_conjunction(body)
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
