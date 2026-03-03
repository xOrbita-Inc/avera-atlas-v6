# AVERA-ATLAS Tracker Service

Multi-sensor fusion and track management service for space debris tracking.

## Overview

The Tracker service sits between the YOLOv8 Detector and the Propagator in the AVERA-ATLAS pipeline:

```
ingest → detector → [TRACKER] → propagator → viz → ui
```

It solves the critical problem of converting 2D SWIR camera detections into 3D orbital states suitable for conjunction assessment.

## The Problem

SWIR cameras are passive 2D sensors. They output pixel coordinates, not 3D positions. With CubeSat-hosted sensors, we have a moving observer tracking a moving target. To determine where a debris object actually is in 3D space, we need:

1. Precise knowledge of where the CubeSat is (ephemeris)
2. Precise knowledge of where the CubeSat is pointing (attitude)
3. Multiple observations over time to solve the angles-only orbit determination problem

## Service Responsibilities

| Component | Function |
|-----------|----------|
| Detection Ingestion | Accept 2D detections from YOLOv8 with platform state |
| Sensor-to-Inertial Transform | Convert pixel coords → angular (RA/Dec) in ECI |
| Cross-Sensor Correlation | Associate detections across multiple platforms |
| Uncorrelated Buffer | Hold detections awaiting sufficient obs for IOD |
| Initial Orbit Determination | Compute 3D state from angles-only observations |
| Track Manager | Lifecycle management and state persistence |
| State Estimation | EKF updates for confirmed tracks |
| Propagator Export | Write states_multi.npz for downstream services |

## Track Lifecycle

```
Detection received
       │
       ▼
┌─────────────────┐
│  UNCORRELATED   │ ← Buffered, awaiting more observations
└────────┬────────┘
         │ (3+ observations, IOD attempted)
         ▼
┌─────────────────┐
│   TENTATIVE     │ ← IOD succeeded, orbit estimate uncertain
└────────┬────────┘
         │ (5+ observations, arc > threshold)
         ▼
┌─────────────────┐
│   CONFIRMED     │ ← High confidence, exported to propagator
└────────┬────────┘
         │ (no observations for 5 min)
         ▼
┌─────────────────┐
│    COASTING     │ ← Propagating without updates
└────────┬────────┘
         │ (coast timeout exceeded)
         ▼
┌─────────────────┐
│    DROPPED      │ ← Track lost
└─────────────────┘
```

## API Endpoints

### Health & Status
- `GET /health` - Basic health check
- `GET /status` - Detailed service status

### Detection Ingestion
- `POST /detections` - Ingest batch of detections
- `POST /detections/single` - Ingest single detection

### Track Management
- `GET /tracks` - List all tracks (with filters)
- `GET /tracks/{track_id}` - Get track detail
- `DELETE /tracks/{track_id}` - Drop a track

### Export
- `POST /export/states` - Write states_multi.npz for propagator

### Sensors
- `POST /sensors` - Register a sensor platform
- `GET /sensors` - List registered sensors
- `GET /sensors/{sensor_id}` - Get sensor status
- `POST /sensors/{sensor_id}/heartbeat` - Update heartbeat

### Uncorrelated Detections
- `GET /uncorrelated` - List uncorrelated buffers
- `POST /uncorrelated/{uct_id}/attempt_iod` - Trigger IOD

## Data Formats

### Input: Detection from Detector

```json
{
  "detection_id": "det-001",
  "sensor_id": "AVERA-SAT-01",
  "timestamp": "2025-01-15T12:00:00Z",
  "pixel_u": 512.5,
  "pixel_v": 384.2,
  "confidence": 0.87,
  "object_class": "Debris",
  "platform_state": {
    "epoch": "2025-01-15T12:00:00Z",
    "position_eci": [6778000.0, 0.0, 0.0],
    "velocity_eci": [0.0, 7668.0, 0.0],
    "quaternion_body_to_eci": [1.0, 0.0, 0.0, 0.0],
    ...
  }
}
```

### Output: states_multi.npz (to Propagator)

```python
{
    'object_ids': ['Debris_001', 'Debris_002', ...],
    'r_eci_km': [[x, y, z], ...],
    'v_eci_km_s': [[vx, vy, vz], ...],
    'confidences': [0.92, 0.85, ...],
    't_window': [60.0, 1440],
    'metadata': '{"source": "tracker", "t0": "...", ...}'
}
```

## Demo Mode

For demonstration purposes, the service includes a mock platform state generator that provides synthetic CubeSat ephemeris and attitude data. This allows the tracker to function without real platform telemetry.

See `mock_platform.py` for the implementation.

Default mock platforms:
- `AVERA-SAT-01` - 400 km, 51.6° inclination
- `AVERA-SAT-02` - 400 km, 51.6° inclination, different RAAN
- `AVERA-SAT-03` - 550 km, 97.5° sun-synchronous

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

API available at http://localhost:8001
OpenAPI docs at http://localhost:8001/docs

### Docker

```bash
# Build
docker build -t avera-tracker .

# Run
docker run -p 8001:8001 -v planner-data:/data/planner_artifacts avera-tracker
```

### Integration with AVERA-ATLAS

Add the tracker service to your docker-compose.yaml (see `docker-compose.tracker.yaml` for snippet).

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PORT` | 8001 | Service port |
| `DATA_DIR` | /data/planner_artifacts | Shared data directory |
| `LOG_LEVEL` | INFO | Logging level |

## Next Steps (Not Yet Implemented)

1. **Sensor-to-inertial transformation** - pixel → angular using camera model
2. **IOD algorithm** - Gooding's method for moving observer
3. **EKF state estimation** - angles-only measurement model
4. **Database persistence** - PostgreSQL/TimescaleDB for tracks
5. **Cross-sensor correlation** - temporal alignment and fusion

## Files

```
tracker-service/
├── main.py              # FastAPI application
├── models.py            # Data models (dataclasses)
├── schemas.py           # Pydantic schemas (OpenAPI)
├── mock_platform.py     # Mock ephemeris/attitude generator
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
├── docker-compose.tracker.yaml  # Integration snippet
└── README.md            # This file
```
