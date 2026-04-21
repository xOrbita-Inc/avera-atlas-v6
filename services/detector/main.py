"""
SWIR Debris Detection Service
==============================
FastAPI microservice using YOLOv8 for spacecraft detection and classification.
Supports 11 spacecraft classes with accurate bounding box localization.

Pipeline: ingest → [DETECTOR] → tracker → propagator → viz → ui
"""

import io
import base64
import os
import httpx
import uvicorn
from datetime import datetime, timezone
from typing import List, Optional, Literal, Dict, Any
from uuid import uuid4
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict
from PIL import Image

from detector_yolo import YOLODetector

# === CONFIGURATION ===
# Tracker connection: prefer TRACKER_HOST/PORT, fall back to legacy PLANNER_HOST/PORT
TRACKER_HOST = os.getenv("TRACKER_HOST", os.getenv("PLANNER_HOST", "tracker"))
TRACKER_PORT = os.getenv("TRACKER_PORT", os.getenv("PLANNER_PORT", "8000"))
TRACKER_DETECTIONS_URL = f"http://{TRACKER_HOST}:{TRACKER_PORT}/detections"

DEFAULT_SENSOR_ID = os.getenv("DEFAULT_SENSOR_ID", "UI-UPLOAD-SWIR")

MODEL_PATH = os.getenv("MODEL_PATH", "spark_detector.onnx")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))


# === 1. API DATA MODELS (Contract Compliance) ===

# Planner only accepts: debris, satellite, star, unknown
# Map all YOLOv8 classes to these 4 categories
SpacecraftClass = Literal["debris", "satellite", "star", "unknown"]

# Mapping from model class names to Planner-accepted classes
CLASS_NAME_MAP = {
    "AcrimSat": "satellite",
    "Aquarius": "satellite",
    "Aura": "satellite",
    "Calipso": "satellite",
    "Cloudsat": "satellite",
    "CubeSat": "satellite",
    "Debris": "debris",
    "Jason": "satellite",
    "Sentinel-6": "satellite",
    "TRMM": "satellite",
    "Terra": "satellite",
    "Unknown": "unknown",
}


class CameraPose(BaseModel):
    position_eci_km: List[float] = Field(..., min_length=3, max_length=3)
    quaternion_eci_body: List[float] = Field(..., min_length=4, max_length=4)


class InferenceRequest(BaseModel):
    frame_id: Optional[str] = None
    image_id: Optional[str] = None  # Legacy UI support
    base64_data: str
    sensor_id: Optional[str] = None
    camera_pose: Optional[CameraPose] = None


class Detection(BaseModel):
    track_id: Optional[str] = None
    model_config = ConfigDict(populate_by_name=True)
    object_class: SpacecraftClass = Field(..., alias="class")
    spacecraft_type: Optional[str] = Field(None, description="Specific spacecraft type (e.g., CubeSat, Calipso)")
    confidence: float
    bbox: List[float] = Field(..., min_length=4, max_length=4)


class DetectionFrame(BaseModel):
    frame_id: str
    timestamp_utc: str
    sensor_id: str
    camera_pose: Optional[CameraPose]
    detections: List[Detection]


# === 2. INFERENCE ENGINE WRAPPER ===

class DetectorWrapper:
    """Wraps YOLODetector for FastAPI service integration."""

    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        print(f"[SYSTEM] Loading YOLOv8 model: {model_path}")
        try:
            self.detector = YOLODetector(
                model_path=model_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            print("[SYSTEM] Model loaded successfully")
        except Exception as e:
            print(f"[CRITICAL] Failed to load model: {e}")
            self.detector = None

    def predict(self, image: Image.Image) -> List[Detection]:
        """
        Run detection and convert to API format.

        Args:
            image: PIL Image

        Returns:
            List of Detection objects in API format
        """
        if self.detector is None:
            return []

        result = self.detector.detect(image)
        detections = []

        for det in result.get('all_detections', []):
            # Map class name to Planner-accepted category
            class_name = det['class_name']
            api_class = CLASS_NAME_MAP.get(class_name, "unknown")

            # bbox is already [x1, y1, x2, y2] format
            bbox = [float(v) for v in det['bbox']]

            detections.append(Detection(
                object_class=api_class,
                spacecraft_type=class_name,  # Original specific type for UI
                confidence=float(det['confidence']),
                bbox=bbox
            ))

        return detections


# === 3. TRACKER PAYLOAD HELPERS ===

def _xyxy_to_xywh(bbox: List[float]) -> Dict[str, float]:
    """Convert [x1, y1, x2, y2] to {bbox_x, bbox_y, bbox_w, bbox_h}."""
    x1, y1, x2, y2 = bbox
    return {
        "bbox_x": x1,
        "bbox_y": y1,
        "bbox_w": x2 - x1,
        "bbox_h": y2 - y1,
    }


def _bbox_center(bbox: List[float]) -> Dict[str, float]:
    """Return {pixel_u, pixel_v} at the center of [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return {
        "pixel_u": (x1 + x2) / 2.0,
        "pixel_v": (y1 + y2) / 2.0,
    }


def build_tracker_payload(frame: DetectionFrame) -> Dict[str, Any]:
    """
    Convert a DetectionFrame into Tracker's DetectionBatchInput shape.

    Produces one DetectionInput record per detection.
    """
    records = []
    for det in frame.detections:
        xywh = _xyxy_to_xywh(det.bbox)
        center = _bbox_center(det.bbox)
        records.append({
            "detection_id": str(uuid4()),
            "sensor_id": frame.sensor_id,
            "timestamp": frame.timestamp_utc,
            "pixel_u": center["pixel_u"],
            "pixel_v": center["pixel_v"],
            **xywh,
            "confidence": det.confidence,
            "object_class": det.object_class,
            "platform_state": None,
        })
    return {"detections": records}


# === 4. BACKGROUND TASKS ===

async def push_to_tracker(payload: Dict[str, Any]):
    """
    Asynchronously push detections to the Tracker service.
    Fire-and-forget; does not block the UI response.
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(TRACKER_DETECTIONS_URL, json=payload)
            if resp.status_code == 200:
                body = resp.json()
                print(
                    f"[INFO] Tracker accepted: processed={body.get('processed')}, "
                    f"iod_ready_ucts={body.get('iod_ready_ucts')}"
                )
            else:
                print(f"[WARN] Tracker returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[WARN] Failed to reach Tracker at {TRACKER_DETECTIONS_URL}: {e}")


# === 5. FASTAPI APP ===

app = FastAPI(
    title="SWIR Debris Detection Service",
    version="3.1.0",
    description="YOLOv8-based spacecraft detection for xOrbita CubeSat debris sensor."
)

# Initialize Detector
detector = DetectorWrapper(MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD)


@app.get("/health")
def health_check():
    return {
        "status": "nominal",
        "backend": "yolov8-onnxruntime",
        "model_loaded": detector.detector is not None,
        "tracker_url": TRACKER_DETECTIONS_URL,
    }


@app.post("/predict", response_model=DetectionFrame)
async def run_inference(request: InferenceRequest, background_tasks: BackgroundTasks):
    try:
        # 1. Image Decoding
        b64_clean = request.base64_data
        if "," in b64_clean:
            b64_clean = b64_clean.split(",")[1]

        img_bytes = base64.b64decode(b64_clean)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # 2. Inference
        detections = detector.predict(image)

        # 3. Create Response
        frame_id = request.frame_id or request.image_id or "unknown"
        sensor_id = request.sensor_id or DEFAULT_SENSOR_ID

        response_obj = DetectionFrame(
            frame_id=frame_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            sensor_id=sensor_id,
            camera_pose=request.camera_pose,
            detections=detections
        )

        # 4. Async Push to Tracker
        tracker_payload = build_tracker_payload(response_obj)
        background_tasks.add_task(push_to_tracker, tracker_payload)

        # 5. Return to UI
        return response_obj

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
