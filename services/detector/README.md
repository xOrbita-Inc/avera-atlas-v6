# Detector Service

YOLOv8-based SWIR debris detection and spacecraft classification.

## Role in Pipeline

```
SWIR Sensor → Ingest → [DETECT] → Track → Propagate → Plan → ATLAS
```

The Detector service runs neural network inference on SWIR camera frames to identify and classify space objects. It uses a YOLOv8 model trained on 11 spacecraft classes and outputs bounding boxes with confidence scores in the Ingest-compatible detection schema.

## Spacecraft Classes

The YOLOv8 model recognizes 11 classes, mapped to 4 planner-compatible categories:

| Model Class | Planner Category |
|-------------|-----------------|
| AcrimSat, Aquarius, Aura, Calipso, Cloudsat, CubeSat, Jason, Sentinel-6, TRMM, Terra | `satellite` |
| Debris | `debris` |
| Unknown | `unknown` |

Stars detected in the field of view are classified as `star` and excluded from conjunction assessment.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Run inference on a SWIR frame |
| `/health` | GET | Service health check |

### Inference Request

```json
{
  "frame_id": "frame_001",
  "sensor_id": "swir_001",
  "image_base64": "<base64-encoded SWIR frame>",
  "camera_pose": {
    "position_eci_km": [6878.0, 0.0, 0.0],
    "quaternion_eci_body": [1.0, 0.0, 0.0, 0.0]
  }
}
```

### Inference Response

Returns detection frame compatible with the Ingest service schema, including bounding boxes, confidence scores, and classified object types.

## Model

- **Architecture**: YOLOv8 (exported to ONNX for edge deployment)
- **Input**: SWIR grayscale frames
- **Confidence threshold**: 0.25 (configurable)
- **IoU threshold**: 0.45 (configurable)

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MODEL_PATH` | `spark_detector.onnx` | Path to ONNX model weights |
| `CONF_THRESHOLD` | `0.25` | Minimum detection confidence |
| `IOU_THRESHOLD` | `0.45` | Non-maximum suppression IoU threshold |

## Docker

```bash
docker build -t avera/detector:v6 .
docker run -p 8000:8000 avera/detector:v6
```

The Dockerfile installs PyTorch CPU wheels to keep the image lightweight. For Jetson deployment, swap to the NVIDIA L4T PyTorch base image for GPU acceleration.

## Hardware Notes

On the Jetson Orin Nano, the detector can leverage the onboard GPU for real-time inference. The IMX477 camera (CSI-2 MIPI interface) serves as a stand-in for the flight SWIR sensor during development.
