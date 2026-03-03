# Ingest Service

Buffers SWIR detection frames and assembles multi-object state vectors for downstream propagation.

## Role in Pipeline

```
SWIR Sensor → [INGEST] → Detect → Track → Propagate → Plan → ATLAS
```

The Ingest service is the entry point for observation data. It accepts detection frames from the SWIR sensor, buffers them over a configurable window, and writes assembled state vectors to the shared artifact volume.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest/detection` | POST | Accept a detection frame |
| `/health` | GET | Service health check |

### Detection Frame Schema

```json
{
  "frame_id": "frame_001",
  "timestamp_utc": "2024-01-15T12:00:00Z",
  "sensor_id": "swir_001",
  "camera_pose": {
    "position_eci_km": [6878.0, 0.0, 0.0],
    "quaternion_eci_body": [1.0, 0.0, 0.0, 0.0]
  },
  "detections": [
    {
      "class": "debris",
      "confidence": 0.92,
      "bbox": [120, 340, 45, 45],
      "track_id": "DEB_001"
    }
  ]
}
```

## Output Artifact

Writes `states_multi.npz` to `/data/planner_artifacts/` containing:

| Array | Shape | Description |
|-------|-------|-------------|
| `object_ids` | (N,) | String identifiers for each tracked object |
| `r_eci_km` | (N, 3) | ECI position vectors in km |
| `v_eci_km_s` | (N, 3) | ECI velocity vectors in km/s |
| `confidences` | (N,) | Detection confidence scores |
| `t_window` | (2,) | [dt_seconds, n_steps] time window parameters |
| `metadata` | JSON string | Source, timestamp, asset state |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OUTPUT_DIR` | `/data/planner_artifacts` | Shared volume path |
| `BUFFER_WINDOW_SIZE` | `5` | Frames to buffer before assembly |

## Docker

```bash
docker build -t avera/ingest:v6 .
docker run -p 8001:8000 -v planner_data:/data/planner_artifacts avera/ingest:v6
```

Port mapping: container listens on 8000, exposed as 8001 to avoid conflict with detector.
