"""
SWIR Debris Detection Service
==============================
FastAPI microservice using YOLOv8 for spacecraft detection and classification.
Supports 11 spacecraft classes with accurate bounding box localization.
"""

import io
import base64
import os
import httpx
import uvicorn
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict
from PIL import Image

from detector_yolo import YOLODetector

# === CONFIGURATION ===
PLANNER_HOST = os.getenv("PLANNER_HOST", "planner-ingest-svc")
PLANNER_PORT = os.getenv("PLANNER_PORT", "8000")
PLANNER_URL = f"http://{PLANNER_HOST}:{PLANNER_PORT}/ingest/detection"

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
    sensor_id: str = "default_sensor"
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


# === 3. BACKGROUND TASKS ===

async def push_to_planner(payload: Dict[str, Any]):
    """
    Asynchronously pushes the result to the Planner Ingest Service.
    Fire-and-forget; does not block the UI response.
    """
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.post(PLANNER_URL, json=payload)
            if resp.status_code != 202:
                print(f"[WARN] Planner returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[WARN] Failed to reach Planner at {PLANNER_URL}: {e}")


# === 4. FASTAPI APP ===

app = FastAPI(
    title="SWIR Debris Detection Service",
    version="3.0.0",
    description="YOLOv8-based spacecraft detection for xOrbita CubeSat debris sensor."
)

# Initialize Detector
detector = DetectorWrapper(MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD)


@app.get("/health")
def health_check():
    return {
        "status": "nominal",
        "backend": "yolov8-onnxruntime",
        "model_loaded": detector.detector is not None
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
        
        response_obj = DetectionFrame(
            frame_id=frame_id,
            timestamp_utc=datetime.utcnow().isoformat() + "Z",
            sensor_id=request.sensor_id,
            camera_pose=request.camera_pose,
            detections=detections
        )
        
        # 4. Async Push to Planner
        payload_dict = response_obj.model_dump(by_alias=True)
        background_tasks.add_task(push_to_planner, payload_dict)
        
        # 5. Return to UI
        return response_obj
    
    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)