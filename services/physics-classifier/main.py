"""
Physics-Based Orbital Object Classifier Service
=================================================
FastAPI microservice — single responsibility:
classify physics-based sensor spectrograms.

Part of AVERA-ATLAS v6 microservice architecture.
Parallel to detector service (SWIR/YOLOv8).
Feeds tracker service for multi-modal data fusion.

ADR-009 Phase 1 implementation.
"""

import io
import base64
import os
import httpx
import uvicorn
from datetime import datetime
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, \
    BackgroundTasks
from pydantic import BaseModel, Field
from PIL import Image

from classifier import PhysicsClassifier, CLASS_NAMES

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/orbital_classifier.onnx"
)
TRACKER_HOST = os.getenv("TRACKER_HOST", "tracker")
TRACKER_PORT = os.getenv("TRACKER_PORT", "8000")
TRACKER_URL = (
    f"http://{TRACKER_HOST}:{TRACKER_PORT}"
    f"/detections/ingest"
)

Modality = Literal["EM", "THERMAL", "UNKNOWN"]
OrbitalClass = Literal[
    "ACTIVE_SAT", "DEAD_SAT", "DEBRIS_SMALL",
    "DEBRIS_LARGE", "MANEUVERING"
]


class ClassifyRequest(BaseModel):
    image_id:    Optional[str] = None
    base64_data: str
    modality:    Modality = "UNKNOWN"
    sensor_id:   str = "physics_sensor"


class ClassifyResult(BaseModel):
    image_id:      str
    timestamp_utc: str
    sensor_id:     str
    modality:      Modality
    class_label:   OrbitalClass
    confidence:    float = Field(ge=0.0, le=1.0)
    risk_class:    str
    all_probs:     dict
    inference_ms:  float
    detected_modality: str = Field(
        default="UNKNOWN",
        description=(
            "Auto-detected modality via FFT "
            "frequency analysis"
        )
    )
    freq_ratio: float = Field(
        default=0.0,
        description=(
            "FFT high/low frequency ratio used "
            "for modality detection"
        )
    )


async def push_to_tracker(payload: dict):
    try:
        async with httpx.AsyncClient(
            timeout=1.0
        ) as client:
            resp = await client.post(
                TRACKER_URL, json=payload
            )
            if resp.status_code not in (200, 202):
                print(
                    f"[WARN] Tracker returned "
                    f"{resp.status_code}"
                )
    except Exception as e:
        print(f"[WARN] Tracker unreachable: {e}")


app = FastAPI(
    title="Physics-Based Orbital Object Classifier",
    version="1.0.0",
    description=(
        "EM/RF and thermal spectrogram classification. "
        "DINOv2 ViT-Small, orbital-pbsdg dataset. "
        "ADR-009 Phase 1."
    )
)

classifier = PhysicsClassifier(MODEL_PATH)


@app.get("/health")
def health_check():
    return {
        "status": (
            "nominal" if classifier.ready
            else "degraded"
        ),
        "service":      "physics-classifier",
        "version":      "1.0.0",
        "model_loaded": classifier.ready,
        "model_path":   MODEL_PATH,
        "classes":      CLASS_NAMES,
        "adr":          "ADR-009 Phase 1",
    }


@app.post(
    "/predict",
    response_model=ClassifyResult
)
async def classify_spectrogram(
    request: ClassifyRequest,
    background_tasks: BackgroundTasks,
):
    if not classifier.ready:
        raise HTTPException(
            status_code=503,
            detail="Physics classifier not ready"
        )
    try:
        b64 = request.base64_data
        if "," in b64:
            b64 = b64.split(",")[1]
        img_bytes = base64.b64decode(b64)
        image = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {e}"
        )
    try:
        result = classifier.classify(image)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {e}"
        )

    image_id = (
        request.image_id
        or f"phys-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    )

    response = ClassifyResult(
        image_id=image_id,
        timestamp_utc=(
            datetime.utcnow().isoformat() + "Z"
        ),
        sensor_id=request.sensor_id,
        modality=request.modality,
        class_label=result["class_label"],
        confidence=result["confidence"],
        risk_class=result["risk_class"],
        all_probs=result["all_probs"],
        inference_ms=result["inference_ms"],
        detected_modality=result.get(
            "detected_modality", "UNKNOWN"
        ),
        freq_ratio=result.get("freq_ratio", 0.0),
    )

    background_tasks.add_task(
        push_to_tracker,
        response.model_dump()
    )

    return response


if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=8000
    )
