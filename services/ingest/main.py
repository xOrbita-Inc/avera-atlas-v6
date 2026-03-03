import os
import json
import numpy as np
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# === CONFIGURATION ===
BUFFER_WINDOW_SIZE = 5 
# This path must be a shared volume in the Pod
OUTPUT_DIR = "/data/planner_artifacts" 
ARTIFACT_NAME = "states_multi.npz"

app = FastAPI(title="AVERA-ATLAS Ingest Service")

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