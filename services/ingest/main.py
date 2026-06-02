import os
import json
import logging
import numpy as np
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from sqlalchemy.exc import OperationalError

from spacetrack_client import SpaceTrackClient
from cdm_parser import parse_cdm_kvn
from cdm_to_conjunction import cdm_to_conjunction_state
from db import init_db, save_cdm_record, CdmRecord, PlannerOutput, get_session

# === CONFIGURATION ===
BUFFER_WINDOW_SIZE = 5
# This path must be a shared volume in the Pod
OUTPUT_DIR = "/data/planner_artifacts"
ARTIFACT_NAME = "states_multi.npz"

# SCRUM-329: Live polling gate.
# Default is false -- injected reference CDM path is used until the SSA
# Sharing Agreement with 18 SPCS is in place. Flip to true via env var
# after the agreement is active. No code changes required at that point.
_LIVE_POLLING_ENABLED = (
    os.environ.get("SPACETRACK_LIVE_POLLING_ENABLED", "false").lower() == "true"
)

app = FastAPI(title="AVERA-ATLAS Ingest Service")


@app.on_event("startup")
async def startup_event() -> None:
    init_db()
    logging.info(
        "[INGEST] Space-Track live polling: %s",
        "ENABLED" if _LIVE_POLLING_ENABLED else "DISABLED (reference CDM path active)",
    )


# --- Data Models (Matching detection.json Schema) ---
class CameraPose(BaseModel):
    position_eci_km: List[float] = Field(..., min_items=3, max_items=3)
    quaternion_eci_body: List[float] = Field(..., min_items=4, max_items=4)

class Detection(BaseModel):
    track_id: Optional[str] = None
    object_class: Literal["debris", "satellite", "star", "unknown"] = Field(..., alias="class")
    spacecraft_type: Optional[str] = None
    confidence: float
    bbox: List[float] = Field(..., min_items=4, max_items=4)

class DetectionFrame(BaseModel):
    frame_id: str
    timestamp_utc: str
    sensor_id: Optional[str] = "default_sensor"
    camera_pose: Optional[CameraPose] = None
    detections: List[Detection]

class PollRequest(BaseModel):
    """Parameters for a Space-Track CDM poll.

    Exactly one of norad_id or pc_threshold must be provided.
    """
    norad_id: Optional[int] = Field(
        None,
        description="Fetch CDMs for a specific NORAD catalog ID"
    )
    pc_threshold: Optional[float] = Field(
        None,
        description="Fetch all CDMs where Pc >= this value"
    )
    days_lookahead: int = Field(
        7,
        ge=1,
        le=30,
        description="Number of days ahead to query TCAs"
    )


class PollResponse(BaseModel):
    saved: int
    skipped: int
    errors: List[str]


detection_buffer = []
object_counters = {}


def get_object_id(det, frame_id):
    """Generate a clean object ID based on detected type."""
    global object_counters

    if det.track_id:
        return det.track_id

    if det.spacecraft_type:
        obj_type = det.spacecraft_type.replace(" ", "_")
    else:
        obj_type = det.object_class if det.object_class != "unknown" else "UNK"

    if obj_type not in object_counters:
        object_counters[obj_type] = 0
    object_counters[obj_type] += 1

    return f"{obj_type}_{object_counters[obj_type]:03d}"


def process_buffer():
    global detection_buffer
    if not detection_buffer:
        return

    print(f"[INGEST] Processing batch of {len(detection_buffer)} frames...")

    object_ids = []
    r_eci_km = []
    v_eci_km_s = []
    confidences = []

    for frame in detection_buffer:
        obs_pos = np.array(frame.camera_pose.position_eci_km) if frame.camera_pose else np.array([0, 0, 0])

        for det in frame.detections:
            if det.confidence < 0.5:
                continue

            obj_id = get_object_id(det, frame.frame_id)

            random_offset = np.random.normal(0, 50, 3)
            state_pos = obs_pos + random_offset
            state_vel = np.array([0.0, 7.6, 0.0])

            object_ids.append(obj_id)
            r_eci_km.append(state_pos)
            v_eci_km_s.append(state_vel)
            confidences.append(det.confidence)

    if object_ids:
        t0_utc = datetime.utcnow().isoformat()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, ARTIFACT_NAME)

        np.savez(
            out_path,
            object_ids=np.array(object_ids),
            r_eci_km=np.array(r_eci_km),
            v_eci_km_s=np.array(v_eci_km_s),
            confidences=np.array(confidences),
            t_window=np.array([60.0, 1440]),
            metadata=json.dumps({"source": "swir_live", "t0": t0_utc})
        )
        print(f"[INGEST] Written {out_path} ({len(object_ids)} objects)")

    detection_buffer = []


@app.post("/ingest/detection", status_code=202)
async def ingest_detection(frame: DetectionFrame, background_tasks: BackgroundTasks):
    detection_buffer.append(frame)
    if len(detection_buffer) >= BUFFER_WINDOW_SIZE:
        background_tasks.add_task(process_buffer)
    return {"status": "buffered", "count": len(detection_buffer)}


# SCRUM-329 AC5: source mode endpoint.
# Returns whether the pipeline is running on live Space-Track data or the
# injected reference CDM. The UI polls this on load to render the correct
# data source label. No code changes required to activate live mode --
# flip SPACETRACK_LIVE_POLLING_ENABLED in the deployment env.
@app.get("/cdm/source_mode", status_code=200)
async def get_source_mode() -> dict:
    """Return the current CDM data source mode.

    Response:
      live_polling_enabled : bool   -- reflects SPACETRACK_LIVE_POLLING_ENABLED
      mode                 : str    -- 'live' | 'reference'
      label                : str    -- human-readable label for the UI badge
    """
    return {
        "live_polling_enabled": _LIVE_POLLING_ENABLED,
        "mode": "live" if _LIVE_POLLING_ENABLED else "reference",
        "label": "LIVE SPACE-TRACK POLLING" if _LIVE_POLLING_ENABLED else "REFERENCE CDM (TIROS 4)",
    }


@app.post("/cdm/poll", response_model=PollResponse, status_code=200)
async def poll_cdms(req: PollRequest) -> PollResponse:
    """Trigger a Space-Track CDM fetch and persist results.

    SCRUM-329: Gated behind SPACETRACK_LIVE_POLLING_ENABLED env flag (AC1).
    When disabled, returns 503 with a descriptive message so the operator
    knows why no CDMs were fetched rather than receiving a silent empty state.

    Each parsed CDM is validated through cdm_to_conjunction_state() before
    being saved, confirming it maps cleanly to evaluate_conjunction() inputs
    without adapter code (AC4). CDMs that fail validation are skipped and
    logged, but do not abort the rest of the batch.
    """
    from fastapi import HTTPException

    # SCRUM-329 AC1: gate on env flag.
    if not _LIVE_POLLING_ENABLED:
        raise HTTPException(
            status_code=503,
            detail=(
                "Live Space-Track CDM polling is disabled. "
                "Set SPACETRACK_LIVE_POLLING_ENABLED=true to activate after the "
                "SSA Sharing Agreement with 18 SPCS is in place. "
                "Use the Inject Example CDM button to load the reference TIROS 4 CDM."
            ),
        )

    if req.norad_id is None and req.pc_threshold is None:
        raise HTTPException(
            status_code=422,
            detail="Exactly one of norad_id or pc_threshold must be provided",
        )

    client = SpaceTrackClient()
    try:
        client.login()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Space-Track login failed: {e}",
        )

    try:
        if req.norad_id is not None:
            raw_kvn = client.get_cdms_for_norad(
                req.norad_id, days_lookahead=req.days_lookahead
            )
        else:
            raw_kvn = client.get_cdms_above_pc(
                req.pc_threshold, days_lookahead=req.days_lookahead
            )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Space-Track query failed: {e}",
        )

    cdm_list = parse_cdm_kvn(raw_kvn)

    # SCRUM-329 AC3: explicit log when Space-Track returns no CDMs.
    # This is the expected state before the SSA Sharing Agreement is active.
    # A silent empty state would be indistinguishable from a real zero-conjunction
    # period, which is misleading. Log the reason so operators and support can
    # diagnose without digging through service internals.
    if not cdm_list:
        query_desc = (
            f"NORAD ID {req.norad_id}" if req.norad_id is not None
            else f"Pc >= {req.pc_threshold}"
        )
        logging.warning(
            "[INGEST] Space-Track returned 0 CDMs for query (%s, %d-day lookahead). "
            "Likely cause: no SSA Sharing Agreement with 18 SPCS, or no registered "
            "spacecraft in CDM_PUBLIC for this operator. "
            "Live polling will return results only after the agreement is active.",
            query_desc,
            req.days_lookahead,
        )
        return PollResponse(saved=0, skipped=0, errors=[])

    saved = 0
    skipped = 0
    errors: List[str] = []

    for cdm in cdm_list:
        norad1 = cdm.get("OBJECT1_OBJECT_DESIGNATOR", "?")
        norad2 = cdm.get("OBJECT2_OBJECT_DESIGNATOR", "?")

        # SCRUM-329 AC4: validate CDM maps cleanly to evaluate_conjunction()
        # inputs via cdm_to_conjunction_state() before persisting. Catches
        # malformed CDMs early -- a CDM that fails here would cause a silent
        # error later in the planner pipeline.
        try:
            cdm_to_conjunction_state(cdm)
        except Exception as e:
            skipped += 1
            msg = (
                f"cdm_to_conjunction_state validation failed for "
                f"{norad1}/{norad2}: {e} -- CDM not saved"
            )
            logging.warning("[INGEST] %s", msg)
            errors.append(msg)
            continue

        try:
            save_cdm_record(cdm)
            saved += 1
        except Exception as e:
            skipped += 1
            msg = f"save_cdm_record failed for {norad1}/{norad2}: {e}"
            logging.warning("[INGEST] %s", msg)
            errors.append(msg)

    logging.info(
        "[INGEST] Poll complete: %d saved, %d skipped from %d CDMs",
        saved, skipped, len(cdm_list),
    )
    return PollResponse(saved=saved, skipped=skipped, errors=errors)


@app.get("/cdm/{primary_norad}/{secondary_norad}", status_code=200)
async def get_cdm(
    primary_norad: str,
    secondary_norad: str,
    limit: int = 1,
) -> dict:
    """Return the most recent CDM record(s) for an object pair.

    Assembles covariance_combined_rtn as C_primary + C_secondary,
    converting from m² (stored) to km² (returned) by dividing by 1e6.
    Returns RTN frame -- NOT ECI. Caller is responsible for rotation.

    covariance_source maps the internal DB source column to the
    OpenAPI CovarianceSourceEnum (openapi/ingest.yaml):
      'space_track' -> 'real_cdm'
      'synthetic'   -> 'surrogate_identity'

    Returns 404 if no records exist for the pair.
    Returns 503 if the CDM store is unavailable.
    """
    from fastapi import HTTPException

    limit = max(1, min(limit, 10))

    _SOURCE_MAP = {
        "space_track": "real_cdm",
        "synthetic":   "surrogate_identity",
    }

    def _assemble(row: CdmRecord) -> dict:
        c_primary = [
            [row.cr_r,     row.ct_r,     row.cn_r    ],
            [row.ct_r,     row.ct_t,     row.cn_t    ],
            [row.cn_r,     row.cn_t,     row.cn_n    ],
        ]
        c_secondary = [
            [row.cr_r_sec, row.ct_r_sec, row.cn_r_sec],
            [row.ct_r_sec, row.ct_t_sec, row.cn_t_sec],
            [row.cn_r_sec, row.cn_t_sec, row.cn_n_sec],
        ]
        combined = [
            [(c_primary[i][j] + c_secondary[i][j]) / 1e6 for j in range(3)]
            for i in range(3)
        ]
        return {
            "id":                      row.id,
            "primary_norad":           row.primary_norad,
            "secondary_norad":         row.secondary_norad,
            "tca":                     row.tca,
            "miss_distance_m":         row.miss_distance_m,
            "pc_space_track":          row.pc_space_track,
            "covariance_combined_rtn": combined,
            "covariance_source":       _SOURCE_MAP.get(row.source, "surrogate_identity"),
            "ingested_at":             row.ingested_at,
        }

    try:
        with get_session() as session:
            rows = (
                session.query(CdmRecord)
                .filter(
                    CdmRecord.primary_norad == primary_norad,
                    CdmRecord.secondary_norad == secondary_norad,
                )
                .order_by(CdmRecord.ingested_at.desc())
                .limit(limit)
                .all()
            )

            if not rows:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No CDM record found for "
                        f"primary_norad={primary_norad}, secondary_norad={secondary_norad}"
                    ),
                )

            if limit == 1:
                result = _assemble(rows[0])
            else:
                result = {"records": [_assemble(r) for r in rows]}

    except HTTPException:
        raise
    except OperationalError as e:
        logging.error("[INGEST] CDM store unavailable: %s", e)
        raise HTTPException(
            status_code=503,
            detail="CDM store temporarily unavailable",
        )

    return result


@app.post("/cdm/inject", status_code=200)
async def inject_cdm(request: Request) -> dict:
    """Inject a pre-parsed CDM dict directly into the CDM store.

    Bypasses Space-Track for demo and testing purposes.
    Accepts a dict matching the cdm_parser.parse_cdm_kvn() output format
    (flat dict with CCSDS field names prefixed by OBJECT1_ / OBJECT2_).

    Returns the same PollResponse shape as POST /cdm/poll for UI consistency.
    This path is always active regardless of SPACETRACK_LIVE_POLLING_ENABLED --
    the reference CDM remains available at all times (AC1).
    """
    body = await request.json()

    cdm_list = body if isinstance(body, list) else [body]

    saved = 0
    skipped = 0
    errors: List[str] = []

    for cdm in cdm_list:
        try:
            save_cdm_record(cdm)
            with get_session() as session:
                record = session.query(CdmRecord).order_by(
                    CdmRecord.id.desc()
                ).first()
                if record:
                    record.source = "real_cdm"
                    session.commit()
            saved += 1
        except Exception as e:
            skipped += 1
            norad1 = cdm.get("OBJECT1_OBJECT_DESIGNATOR", "?")
            norad2 = cdm.get("OBJECT2_OBJECT_DESIGNATOR", "?")
            msg = f"inject failed for {norad1}/{norad2}: {e}"
            logging.warning("[INGEST] %s", msg)
            errors.append(msg)

    logging.info("[INGEST] Inject complete: %d saved, %d skipped", saved, skipped)
    return {"saved": saved, "skipped": skipped, "errors": errors}


@app.get("/store/cdm_records")
async def store_cdm_records():
    """Return all CDM records from the SQLite store."""
    try:
        with get_session() as session:
            rows = session.query(CdmRecord).order_by(
                CdmRecord.ingested_at.desc()
            ).limit(100).all()
            return [
                {
                    "id": r.id,
                    "primary_norad": r.primary_norad,
                    "secondary_norad": r.secondary_norad,
                    "tca_utc": r.tca,
                    "miss_distance_m": round(r.miss_distance_m, 1) if r.miss_distance_m else None,
                    "pc": r.pc_space_track,
                    "covariance_source": r.source,
                    "ingested_at": r.ingested_at,
                }
                for r in rows
            ]
    except Exception as e:
        logging.error("[INGEST] store_cdm_records error: %s", e)
        return []


@app.get("/store/planner_outputs")
async def store_planner_outputs():
    """Return all planner output records from the SQLite store."""
    try:
        with get_session() as session:
            rows = session.query(PlannerOutput).order_by(
                PlannerOutput.created_at.desc()
            ).limit(100).all()
            return [
                {
                    "id": r.id,
                    "cdm_record_id": r.cdm_record_id,
                    "conjunction_id": f"{r.cdm_record_id}",
                    "recommendation": r.recommendation,
                    "utility": round(r.utility_value, 4) if r.utility_value else None,
                    "dv_magnitude_m_s": round(r.delta_v_ms, 3) if r.delta_v_ms else None,
                    "covariance_source": r.covariance_source,
                    "created_at": r.created_at,
                }
                for r in rows
            ]
    except Exception as e:
        logging.error("[INGEST] store_planner_outputs error: %s", e)
        return []


@app.post("/planner_output", status_code=201)
async def save_planner_output(request: Request) -> dict:
    """Persist a planner audit record to the SQLite store."""
    body = await request.json()
    try:
        with get_session() as session:
            output = PlannerOutput(
                cdm_record_id=body.get("cdm_record_id"),
                recommendation=body.get("recommendation"),
                delta_v_ms=body.get("delta_v_ms"),
                pc_computed=body.get("pc_computed") or 0.0,
                utility_value=body.get("utility_value"),
                lambda_v=body.get("lambda_v") or 0.0,
                lambda_l=body.get("lambda_l") or 0.0,
                covariance_source=body.get("covariance_source"),
                created_at=datetime.utcnow().isoformat() + "Z",
            )
            session.add(output)
            session.commit()
            logging.info("[INGEST] Planner output saved: %s", body.get("conjunction_id"))
            return {"status": "saved"}
    except Exception as e:
        logging.error("[INGEST] save_planner_output error: %s", e)
        return {"status": "error", "error": str(e)}


@app.delete("/store/cdm_records/duplicates", status_code=200)
async def deduplicate_cdm_records() -> dict:
    """Remove duplicate CDM records keeping only the most recent per object pair."""
    try:
        with get_session() as session:
            rows = session.query(CdmRecord).order_by(
                CdmRecord.ingested_at.desc()
            ).all()
            seen = set()
            to_delete = []
            for r in rows:
                key = (r.primary_norad, r.secondary_norad)
                if key in seen:
                    to_delete.append(r.id)
                else:
                    seen.add(key)
            for rid in to_delete:
                session.query(CdmRecord).filter(CdmRecord.id == rid).delete()
            session.commit()
            logging.info("[INGEST] Deduplicated CDM records: removed %d", len(to_delete))
            return {"removed": len(to_delete), "kept": len(seen)}
    except Exception as e:
        logging.error("[INGEST] deduplicate error: %s", e)
        return {"error": str(e)}


@app.delete("/store/cdm_records/all", status_code=200)
async def clear_cdm_records() -> dict:
    """Delete all CDM records and planner outputs from the store."""
    try:
        with get_session() as session:
            cdm_count = session.query(CdmRecord).count()
            output_count = session.query(PlannerOutput).count()
            session.query(PlannerOutput).delete()
            session.query(CdmRecord).delete()
            session.commit()
            logging.info("[INGEST] Cleared store: %d CDMs, %d outputs", cdm_count, output_count)
            return {"cdm_records_deleted": cdm_count, "planner_outputs_deleted": output_count}
    except Exception as e:
        logging.error("[INGEST] clear error: %s", e)
        return {"error": str(e)}
