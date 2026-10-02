import os
import json
import logging
import numpy as np
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from sqlalchemy.exc import OperationalError

from cdm_parser import parse_cdm_kvn
from cdm_to_conjunction import cdm_to_conjunction_state
from fastapi.responses import JSONResponse
from db import (
    init_db, save_cdm_record, CdmRecord, PlannerOutput, DecisionLogRecord,
    EvidenceRecordRow, ScreeningRecord, get_session,
)

# === CONFIGURATION ===
BUFFER_WINDOW_SIZE = 5
# This path must be a shared volume in the Pod
OUTPUT_DIR = "/data/planner_artifacts"
ARTIFACT_NAME = "states_multi.npz"

# SCRUM-431: Space-Track is retired as a data source. The live polling gate
# and its client are gone; the reference CDM path below is the only ingest
# route, and it was always the active one.

app = FastAPI(title="AVERA-ATLAS Ingest Service")


@app.on_event("startup")
async def startup_event() -> None:
    init_db()
    logging.info("[INGEST] CDM store ready, reference CDM path active")


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


# SCRUM-329 AC5 / SCRUM-431: source mode endpoint.
# Space-Track is retired, so there is no live tier left to advertise: the
# reference CDM is the only ingest source. Kept as an endpoint because the
# dashboard reads it, and kept truthful rather than removed so a caller is not
# left guessing what the store holds.
@app.get("/cdm/source_mode", status_code=200)
async def get_source_mode() -> dict:
    """Return the current CDM data source mode.

    Response:
      mode  : str -- always 'reference'
      label : str -- human-readable label for the UI badge
    """
    return {
        "mode": "reference",
        "label": "REFERENCE CDM (TIROS 4)",
    }


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
      'reference_cdm' -> 'real_cdm'
      'synthetic'     -> 'surrogate_identity'

    SCRUM-431: 'space_track' is kept in the map for rows written before the
    retirement. Dropping it would make every already-stored real CDM fall
    through to the surrogate default, which would silently downgrade a real
    covariance to an assumed one.

    Returns 404 if no records exist for the pair.
    Returns 503 if the CDM store is unavailable.
    """
    from fastapi import HTTPException

    limit = max(1, min(limit, 10))

    _SOURCE_MAP = {
        "reference_cdm": "real_cdm",
        "space_track":   "real_cdm",   # pre-SCRUM-431 rows
        "synthetic":     "surrogate_identity",
        "real_cdm":      "real_cdm",
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
        def _km2(m):
            return [[m[i][j] / 1e6 for j in range(3)] for i in range(3)]

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
            # SCRUM-453 item 2: the per-object blocks, which were stored all
            # along but summed away on assembly. The combined matrix is what the
            # scorer's Pc needs -- primary plus secondary relative uncertainty --
            # but it over-states the primary's OWN uncertainty, so using it to
            # seed the secondary screen inflates the seed (SCRUM-454 measured
            # 2.72x in sigma, 7.4x in trace on a live CDM). The live LeoLabs path
            # already seeds from the primary's own block off the parsed CDM; this
            # puts the same quantity on the wire for the stored and reference
            # path. Same unit convention as combined: m^2 stored, km^2 returned.
            # combined is unchanged and is still the sum of these two.
            "covariance_primary_rtn":   _km2(c_primary),
            "covariance_secondary_rtn": _km2(c_secondary),
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

    Loads the offline reference CDM for demo and testing.
    Accepts a dict matching the cdm_parser.parse_cdm_kvn() output format
    (flat dict with CCSDS field names prefixed by OBJECT1_ / OBJECT2_).

    Returns {saved, skipped, errors}. SCRUM-431: the Space-Track poll this used
    to sit beside is retired. SCRUM-429: the source is now named through
    save_cdm_record rather than patched onto the row afterwards, and the save is
    idempotent, so re-injecting the reference CDM reuses its row.
    """
    body = await request.json()

    cdm_list = body if isinstance(body, list) else [body]

    saved = 0
    skipped = 0
    errors: List[str] = []

    for cdm in cdm_list:
        try:
            save_cdm_record(cdm, source="real_cdm")
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


@app.post("/cdm/persist", status_code=200)
async def persist_cdm(request: Request) -> dict:
    """SCRUM-429: persist a live LeoLabs CDM from the planner's evaluate path.

    Same flat OBJECTn_ shape as /cdm/inject, written with source='leolabs' so
    the store distinguishes a live conjunction from the offline reference CDM
    and from a synthetic one. ParsedLeoLabsCDM.to_store_cdm_dict() produces it.

    Idempotent by construction: save_cdm_record upserts on the originating CDM's
    id, so re-evaluating a conjunction reuses its row and returns the same id
    rather than accumulating one row per evaluate. created says which happened.

    Returns {id, created}. The planner threads id back as cdm_record_id so the
    decision's audit trail links to the exact stored CDM it was made from.
    """
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=422,
            content={"error": "expected a single CDM object"},
        )
    try:
        record_id, created = save_cdm_record(body, source="leolabs")
    except Exception as exc:
        logging.warning("[INGEST] persist failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    logging.info(
        "[INGEST] LeoLabs CDM %s (id=%d, created=%s)",
        body.get("COMMENT_ID", "?"), record_id, created,
    )
    return {"id": record_id, "created": created}


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


@app.post("/decision_log", status_code=201)
async def save_decision_log(request: Request) -> dict:
    """Persist a full DecisionLog audit record keyed by log_id (SCRUM-351)."""
    body = await request.json()
    log_id = body.get("log_id")
    if not log_id:
        return JSONResponse(status_code=400, content={"status": "error", "error": "log_id required"})
    decision_log = body.get("decision_log")
    payload_json = json.dumps(decision_log) if decision_log is not None else "{}"
    try:
        with get_session() as session:
            existing = session.query(DecisionLogRecord).filter_by(log_id=log_id).first()
            if existing is not None:
                # SCRUM-377: an audit record is not editable. Re-posting the
                # identical payload stays idempotent (retries and at-least-once
                # delivery must not fail), but a write that would CHANGE a
                # stored decision is refused rather than silently overwriting
                # it, which is what this endpoint used to do.
                if existing.decision_log_json == payload_json:
                    logging.info("[INGEST] Decision log re-post identical, no change: %s", log_id)
                    return {"status": "saved", "log_id": log_id, "changed": False}
                logging.warning(
                    "[INGEST] Refused overwrite of existing decision log: %s", log_id
                )
                return JSONResponse(
                    status_code=409,
                    content={
                        "status": "error",
                        "error": (
                            f"decision log '{log_id}' already exists with different "
                            "content; audit records are append-only and are not "
                            "overwritten"
                        ),
                        "log_id": log_id,
                    },
                )
            else:
                session.add(DecisionLogRecord(
                    log_id=log_id,
                    conjunction_id=body.get("conjunction_id"),
                    sat_id=body.get("sat_id"),
                    decision=body.get("decision"),
                    decision_log_json=payload_json,
                    created_at=datetime.utcnow().isoformat() + "Z",
                ))
            logging.info("[INGEST] Decision log saved: %s", log_id)
            return {"status": "saved", "log_id": log_id}
    except Exception as e:
        logging.error("[INGEST] save_decision_log error: %s", e)
        return {"status": "error", "error": str(e)}


@app.get("/store/decision_log/{log_id}")
async def store_decision_log(log_id: str):
    """Return the full DecisionLog JSON for a decision ID, or 404 (SCRUM-351)."""
    try:
        with get_session() as session:
            row = session.query(DecisionLogRecord).filter_by(log_id=log_id).first()
            if row is None:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"No decision log found for id '{log_id}'"},
                )
            return json.loads(row.decision_log_json)
    except Exception as e:
        logging.error("[INGEST] store_decision_log error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/screening/persist", status_code=200)
async def persist_screening(request: Request) -> dict:
    """SCRUM-453: store the on-demand secondary screening a decision was made on.

    Accepts the record the planner's _persist_screening_result already builds and
    posts. That planner code is unchanged by this ticket; it was written against
    this endpoint before the endpoint existed, so the contract here is whatever it
    sends: decision_log_id, screening_id, source, cdm_count, conjunction_count,
    skipped (a list of unparseable CDMs with reasons), clear, breaches,
    conjunctions, covariance_model, p_post_source.

    Idempotent, like /cdm/persist. The planner posts once per evaluate and
    re-evaluating the same conjunction must reuse the row rather than accumulate
    one per evaluate, so a repeat for the same decision_log_id updates in place.
    Deliberately NOT the 409-on-different-content rule /decision_log uses: that
    endpoint guards an append-only audit record of the decision itself, whereas
    this is the derived record of the screen behind it, and a retry after a
    partial write has to be able to land. The decision is never reachable from
    here -- the planner already treats any failure of this call as non-fatal.

    Returns {stored, decision_log_id, screening_id, id, created}. The planner only
    reads the status code and keeps its own screening id, so nothing here depends
    on the body shape.
    """
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=422,
            content={"error": "expected a single screening record object"},
        )

    decision_log_id = body.get("decision_log_id")
    screening_id = body.get("screening_id")
    if decision_log_id is None and screening_id is None:
        return JSONResponse(
            status_code=400,
            content={"error": "decision_log_id or screening_id required"},
        )

    def _as_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    skipped = body.get("skipped")
    if skipped is None:
        skipped = []
    elif not isinstance(skipped, list):
        # Tolerate a producer that sends a count or a flag rather than the list,
        # so an older or third-party caller still stores instead of 500ing.
        skipped = [skipped]

    breaches = body.get("breaches") or []
    conjunctions = body.get("conjunctions") or []
    clear = body.get("clear")

    try:
        with get_session() as session:
            existing = None
            if decision_log_id is not None:
                existing = (
                    session.query(ScreeningRecord)
                    .filter(ScreeningRecord.decision_log_id == decision_log_id)
                    .first()
                )
            if existing is None and screening_id is not None:
                existing = (
                    session.query(ScreeningRecord)
                    .filter(ScreeningRecord.screening_id == str(screening_id))
                    .first()
                )

            row = existing if existing is not None else ScreeningRecord()
            row.decision_log_id   = decision_log_id
            row.screening_id      = None if screening_id is None else str(screening_id)
            row.source            = body.get("source")
            row.cdm_count         = _as_int(body.get("cdm_count"))
            row.conjunction_count = _as_int(body.get("conjunction_count"))
            row.skipped_count     = len(skipped)
            row.skipped_json      = json.dumps(skipped)
            row.clear             = None if clear is None else bool(clear)
            row.covariance_model  = body.get("covariance_model")
            row.p_post_source     = body.get("p_post_source")
            row.breaches_json     = json.dumps(breaches)
            row.conjunctions_json = json.dumps(conjunctions)
            if existing is None:
                row.created_at = datetime.utcnow().isoformat() + "Z"
                session.add(row)
            session.flush()      # assigns row.id inside this transaction
            row_id = int(row.id)

        logging.info(
            "[INGEST] Screening record stored: screening_id=%s decision_log_id=%s "
            "conjunctions=%d created=%s",
            screening_id, decision_log_id, len(conjunctions), existing is None,
        )
        return {
            "stored": True,
            "decision_log_id": decision_log_id,
            "screening_id": screening_id,
            "id": row_id,
            "created": existing is None,
        }
    except Exception as exc:
        logging.error("[INGEST] persist_screening error: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/store/screening/{decision_log_id}")
async def store_screening(decision_log_id: str):
    """Return the stored screening for a decision log id, or 404 (SCRUM-453).

    Mirrors GET /store/decision_log/{log_id}. This is what makes a CLEAR or
    NOT CLEAR auditable: it hands back the exact conjunctions and breaches the
    verdict was formed over, not a summary of them.
    """
    try:
        with get_session() as session:
            row = (
                session.query(ScreeningRecord)
                .filter(ScreeningRecord.decision_log_id == decision_log_id)
                .order_by(ScreeningRecord.id.desc())
                .first()
            )
            if row is None:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": (
                            f"No screening record found for decision_log_id "
                            f"'{decision_log_id}'"
                        )
                    },
                )
            return {
                "id":                row.id,
                "decision_log_id":   row.decision_log_id,
                "screening_id":      row.screening_id,
                "source":            row.source,
                "cdm_count":         row.cdm_count,
                "conjunction_count": row.conjunction_count,
                "skipped":           json.loads(row.skipped_json or "[]"),
                "skipped_count":     row.skipped_count,
                "clear":             row.clear,
                "covariance_model":  row.covariance_model,
                "p_post_source":     row.p_post_source,
                "breaches":          json.loads(row.breaches_json or "[]"),
                "conjunctions":      json.loads(row.conjunctions_json or "[]"),
                "created_at":        row.created_at,
            }
    except Exception as exc:
        logging.error("[INGEST] store_screening error: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


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


# ---------------------------------------------------------------------------
# SCRUM-377: Evidence package. Append-only, chained audit records.
# MAF v2.0 section 10.
# ---------------------------------------------------------------------------

_GENESIS_HASH = "0" * 64


def _sha256_hex(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@app.post("/evidence_record", status_code=201)
async def save_evidence_record(request: Request) -> dict:
    """Append one evidence record to its chain (SCRUM-377).

    Append-only. Two things are refused rather than accepted:

      - a record_id that already exists with different content. Re-posting the
        identical record stays idempotent so retries are safe, but an audit
        record is never edited in place.
      - a seq that is not the next one in the chain. Enforcing contiguity at
        write time means a well-behaved writer cannot create a gap, which is
        what makes a gap discovered later actual evidence rather than noise.

    Expects the body to be an EvidenceRecord.to_dict() plus the canonical
    payload string the content_hash was computed over, so verification never
    needs a second serializer.
    """
    body = await request.json()
    record = body.get("record")
    canonical_payload = body.get("canonical_payload")

    if not isinstance(record, dict):
        return JSONResponse(status_code=400, content={"status": "error", "error": "record object required"})
    if not isinstance(canonical_payload, str) or not canonical_payload:
        return JSONResponse(status_code=400, content={"status": "error", "error": "canonical_payload required"})

    record_id = record.get("record_id")
    chain_id = record.get("chain_id")
    seq = record.get("seq")
    content_hash = record.get("content_hash")
    prev_hash = record.get("prev_hash")

    if not record_id or not chain_id or seq is None or not content_hash or not prev_hash:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "record_id, chain_id, seq, prev_hash and content_hash are required"},
        )

    # The hash must actually cover the payload we were handed. Catches a
    # mismatched or truncated write before it enters the chain.
    if _sha256_hex(canonical_payload) != content_hash:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "content_hash does not match canonical_payload"},
        )

    record_json = json.dumps(record, sort_keys=True)

    try:
        with get_session() as session:
            existing = session.query(EvidenceRecordRow).filter_by(record_id=record_id).first()
            if existing is not None:
                if existing.content_hash == content_hash and existing.record_json == record_json:
                    return {"status": "saved", "record_id": record_id, "changed": False}
                logging.warning("[INGEST] Refused overwrite of evidence record: %s", record_id)
                return JSONResponse(
                    status_code=409,
                    content={
                        "status": "error",
                        "error": (
                            f"evidence record '{record_id}' already exists with different "
                            "content; the evidence chain is append-only"
                        ),
                        "record_id": record_id,
                    },
                )

            head = (
                session.query(EvidenceRecordRow)
                .filter_by(chain_id=chain_id)
                .order_by(EvidenceRecordRow.seq.desc())
                .first()
            )
            expected_seq = 0 if head is None else head.seq + 1
            expected_prev = _GENESIS_HASH if head is None else head.content_hash

            if int(seq) != expected_seq:
                return JSONResponse(
                    status_code=409,
                    content={
                        "status": "error",
                        "error": (
                            f"out-of-order append to chain '{chain_id}': expected seq "
                            f"{expected_seq}, got {seq}"
                        ),
                        "expected_seq": expected_seq,
                    },
                )
            if prev_hash != expected_prev:
                return JSONResponse(
                    status_code=409,
                    content={
                        "status": "error",
                        "error": (
                            f"prev_hash does not match the head of chain '{chain_id}'; "
                            "the record does not link to the record before it"
                        ),
                        "expected_prev_hash": expected_prev,
                    },
                )

            session.add(EvidenceRecordRow(
                record_id=record_id,
                chain_id=chain_id,
                seq=int(seq),
                record_type=record.get("record_type", ""),
                conjunction_id=(
                    (record.get("fields", {}).get("conjunction_id") or {}).get("value")
                ),
                prev_hash=prev_hash,
                content_hash=content_hash,
                canonical_payload=canonical_payload,
                record_json=record_json,
                created_at=datetime.utcnow().isoformat() + "Z",
            ))
            logging.info("[INGEST] Evidence record appended: %s", record_id)
            return {"status": "saved", "record_id": record_id, "changed": True}
    except Exception as e:
        logging.error("[INGEST] save_evidence_record error: %s", e)
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


@app.get("/store/evidence/{chain_id}")
async def store_evidence_chain(chain_id: str):
    """Return every record in a chain, in sequence order (SCRUM-377)."""
    try:
        with get_session() as session:
            rows = (
                session.query(EvidenceRecordRow)
                .filter_by(chain_id=chain_id)
                .order_by(EvidenceRecordRow.seq.asc())
                .all()
            )
            if not rows:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"No evidence chain found for '{chain_id}'"},
                )
            return {
                "chain_id": chain_id,
                "count": len(rows),
                "records": [json.loads(r.record_json) for r in rows],
            }
    except Exception as e:
        logging.error("[INGEST] store_evidence_chain error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/store/evidence/{chain_id}/head")
async def store_evidence_head(chain_id: str):
    """Return the chain's current head so a writer knows the next seq/prev_hash.

    Returns seq -1 and the genesis hash for a chain that does not exist yet, so
    a first append needs no special case.
    """
    try:
        with get_session() as session:
            row = (
                session.query(EvidenceRecordRow)
                .filter_by(chain_id=chain_id)
                .order_by(EvidenceRecordRow.seq.desc())
                .first()
            )
            if row is None:
                return {"chain_id": chain_id, "seq": -1, "next_seq": 0,
                        "content_hash": _GENESIS_HASH, "exists": False}
            return {"chain_id": chain_id, "seq": row.seq, "next_seq": row.seq + 1,
                    "content_hash": row.content_hash, "exists": True}
    except Exception as e:
        logging.error("[INGEST] store_evidence_head error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/store/evidence/{chain_id}/verify")
async def verify_evidence_chain(chain_id: str):
    """Walk a chain and report the first break, if any (SCRUM-377).

    Detects modification (a record whose stored hash no longer covers its
    payload), deletion (a seq gap, or a prev_hash that does not link), and a
    chain that does not begin at the genesis hash.

    Honest limit: a hash chain held entirely inside the store it protects
    detects corruption, partial edits and dropped records, but not an actor who
    can rewrite every record forward from the change. Defending against that
    needs an anchor outside this database. See the note in
    services/planner/common/evidence_record.py.
    """
    try:
        with get_session() as session:
            rows = (
                session.query(EvidenceRecordRow)
                .filter_by(chain_id=chain_id)
                .order_by(EvidenceRecordRow.seq.asc())
                .all()
            )
            if not rows:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"No evidence chain found for '{chain_id}'"},
                )

            expected_seq = 0
            expected_prev = _GENESIS_HASH
            for i, row in enumerate(rows):
                if row.seq != expected_seq:
                    return {
                        "chain_id": chain_id, "ok": False, "break_seq": expected_seq,
                        "checked": i,
                        "reason": (
                            f"sequence gap: expected seq {expected_seq}, found {row.seq}. "
                            "A record is missing from this chain."
                        ),
                    }
                if _sha256_hex(row.canonical_payload) != row.content_hash:
                    return {
                        "chain_id": chain_id, "ok": False, "break_seq": row.seq,
                        "checked": i + 1,
                        "reason": (
                            f"record {row.record_id} has been modified: stored "
                            "content_hash does not cover its payload"
                        ),
                    }
                if row.prev_hash != expected_prev:
                    return {
                        "chain_id": chain_id, "ok": False, "break_seq": row.seq,
                        "checked": i + 1,
                        "reason": (
                            f"record {row.record_id} does not link to its predecessor: "
                            "prev_hash mismatch. A record was removed or reordered."
                        ),
                    }
                expected_prev = row.content_hash
                expected_seq += 1

            return {
                "chain_id": chain_id, "ok": True, "break_seq": None,
                "checked": len(rows), "reason": "", "head_hash": expected_prev,
            }
    except Exception as e:
        logging.error("[INGEST] verify_evidence_chain error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
