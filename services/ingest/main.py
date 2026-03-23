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
from db import init_db, save_cdm_record, CdmRecord, get_session

# === CONFIGURATION ===
BUFFER_WINDOW_SIZE = 5 
# This path must be a shared volume in the Pod
OUTPUT_DIR = "/data/planner_artifacts" 
ARTIFACT_NAME = "states_multi.npz"

app = FastAPI(title="AVERA-ATLAS Ingest Service")


@app.on_event("startup")
async def startup_event() -> None:
    init_db()


# --- Data Models (Matching detection.json Schema) ---
class CameraPose(BaseModel):
    position_eci_km: List[float] = Field(..., min_items=3, max_items=3)
    quaternion_eci_body: List[float] = Field(..., min_items=4, max_items=4)

class Detection(BaseModel):
    track_id: Optional[str] = None
    object_class: Literal["debris", "satellite", "star", "unknown"] = Field(..., alias="class")
    spacecraft_type: Optional[str] = None  # Specific type like "CubeSat", "Starlink", etc.
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
object_counters = {}  # Track sequential IDs per object type

def get_object_id(det, frame_id):
    """Generate a clean object ID based on detected type."""
    global object_counters
    
    # Priority: track_id > spacecraft_type > object_class > unknown
    if det.track_id:
        return det.track_id
    
    # Use spacecraft_type if available (e.g., "CubeSat", "Starlink")
    if det.spacecraft_type:
        obj_type = det.spacecraft_type.replace(" ", "_")
    else:
        # Fall back to general class
        obj_type = det.object_class if det.object_class != "unknown" else "UNK"
    
    # Get next sequential number for this type
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
    
    # 1. Convert Detections to States (Mock Logic for V2)
    for frame in detection_buffer:
        obs_pos = np.array(frame.camera_pose.position_eci_km) if frame.camera_pose else np.array([0,0,0])
        
        for det in frame.detections:
            if det.confidence < 0.5: continue
            
            obj_id = get_object_id(det, frame.frame_id)
            
            # Mock Relative -> Absolute State
            random_offset = np.random.normal(0, 50, 3)
            state_pos = obs_pos + random_offset
            state_vel = np.array([0.0, 7.6, 0.0]) # Approx LEO
            
            object_ids.append(obj_id)
            r_eci_km.append(state_pos)
            v_eci_km_s.append(state_vel)
            confidences.append(det.confidence)

    # 2. Write Artifact to Shared Volume
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


@app.post("/cdm/poll", response_model=PollResponse, status_code=200)
async def poll_cdms(req: PollRequest) -> PollResponse:
    """Trigger a Space-Track CDM fetch and persist results.

    Exactly one of norad_id or pc_threshold must be provided.
    Each parsed CDM is saved independently -- a failure on one record
    does not abort the rest of the batch.
    """
    if req.norad_id is None and req.pc_threshold is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="Exactly one of norad_id or pc_threshold must be provided"
        )

    client = SpaceTrackClient()
    try:
        client.login()
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"Space-Track login failed: {e}"
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
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"Space-Track query failed: {e}"
        )

    cdm_list = parse_cdm_kvn(raw_kvn)

    saved = 0
    skipped = 0
    errors: List[str] = []

    for cdm in cdm_list:
        try:
            save_cdm_record(cdm)
            saved += 1
        except Exception as e:
            skipped += 1
            norad1 = cdm.get("OBJECT1_OBJECT_DESIGNATOR", "?")
            norad2 = cdm.get("OBJECT2_OBJECT_DESIGNATOR", "?")
            msg = f"save_cdm_record failed for {norad1}/{norad2}: {e}"
            logging.warning("[INGEST] %s", msg)
            errors.append(msg)

    logging.info(
        "[INGEST] Poll complete: %d saved, %d skipped from %d CDMs",
        saved, skipped, len(cdm_list)
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
        # Build 3x3 RTN covariance per object (m²) then sum and convert to km².
        # Matrix layout per CCSDS 508.0-B-1:
        #   [[CR_R,  CT_R,  CN_R],
        #    [CT_R,  CT_T,  CN_T],
        #    [CN_R,  CN_T,  CN_N]]
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
                    )
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
            detail="CDM store temporarily unavailable"
        )

    return result


@app.post("/cdm/inject", status_code=200)
async def inject_cdm(request: Request) -> dict:
    """Inject a pre-parsed CDM dict directly into the CDM store.

    Bypasses Space-Track for demo and testing purposes.
    Accepts a dict matching the cdm_parser.parse_cdm_kvn() output format
    (flat dict with CCSDS field names prefixed by OBJECT1_ / OBJECT2_).

    Returns the same PollResponse shape as POST /cdm/poll for UI consistency.
    """
    body = await request.json()

    # Accept either a single CDM dict or a list
    cdm_list = body if isinstance(body, list) else [body]

    saved = 0
    skipped = 0
    errors: List[str] = []

    for cdm in cdm_list:
        try:
            save_cdm_record(cdm)
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