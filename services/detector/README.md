# Detector Service

YOLOv8-based SWIR debris detection and spacecraft classification.

## Role in Pipeline

```
SWIR Sensor → Ingest → [DETECT] → Tracker → Propagate → Plan → ATLAS
```

The Detector service runs neural network inference on SWIR camera frames to identify and classify space objects. It uses a YOLOv8 model trained on 11 spacecraft classes, returns bounding boxes and confidence scores synchronously to the UI, and asynchronously forwards each detection batch to the Tracker service in `DetectionBatchInput` format for correlation and IOD.

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
| `/health` | GET | Service health check (includes configured `tracker_url`) |

### Inference Request

```json
{
  "frame_id": "frame_001",
  "sensor_id": "UI-UPLOAD-SWIR",
  "base64_data": "<base64-encoded SWIR frame>",
  "camera_pose": {
    "position_eci_km": [6878.0, 0.0, 0.0],
    "quaternion_eci_body": [1.0, 0.0, 0.0, 0.0]
  }
}
```

`sensor_id` defaults to `UI-UPLOAD-SWIR` if not supplied. For programmatic callers, use `AVERA-SAT-01-SWIR` or `AVERA-SAT-02-SWIR` so the Tracker's platform-id derivation resolves to the expected mock platform.

### Inference Response

Returns a `DetectionFrame` with bounding boxes in `[x1, y1, x2, y2]` (xyxy) format, confidence scores, and object class. The UI contract is stable; response shape has been unchanged since v3.0.0.

### Tracker Forwarding

On each successful inference, the Detector translates its internal `DetectionFrame` into Tracker's `DetectionBatchInput` schema and POSTs asynchronously to `http://{TRACKER_HOST}:{TRACKER_PORT}/detections`. The forward is fire-and-forget and does not block the UI response. Translation includes:

- `bbox` xyxy → `bbox_x` / `bbox_y` / `bbox_w` / `bbox_h`
- bbox centre → `pixel_u` / `pixel_v`
- generated UUID per `detection_id`
- ISO-8601 UTC `timestamp`
- `platform_state: null` (Tracker falls back to its mock platform generator)

See `services/tracker/schemas.py :: DetectionInput` for the full downstream schema.

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
| `TRACKER_HOST` | `tracker` | Tracker service hostname |
| `TRACKER_PORT` | `8000` | Tracker service port |
| `DEFAULT_SENSOR_ID` | `UI-UPLOAD-SWIR` | Sensor ID used when the request omits one |
| `PLANNER_HOST` | n/a | Legacy fallback for `TRACKER_HOST` (pre-SCRUM-325; kept for transition) |
| `PLANNER_PORT` | n/a | Legacy fallback for `TRACKER_PORT` |

## Docker

```bash
docker build -t avera/detector:v6 .
docker run -p 8000:8000 avera/detector:v6
```

The Dockerfile installs PyTorch CPU wheels to keep the image lightweight. For Jetson deployment, swap to the NVIDIA L4T PyTorch base image for GPU acceleration.

## Hardware Notes

On the Jetson Orin Nano, the detector can leverage the onboard GPU for real-time inference. The IMX477 camera (CSI-2 MIPI interface) serves as a stand-in for the flight SWIR sensor during development.
