# Ingest Service

CDM persistence and detection-frame buffer for the AVERA-ATLAS pipeline.

## Role in Pipeline

The Ingest service has two distinct responsibilities:

1. **CDM persistence (primary role).** Fetches CCSDS 508.0-B-1 Conjunction Data Messages from Space-Track.org, parses them, and persists the covariance records to a SQLite store. Planner queries this store before running the APS decision model. Per ADR-001, Ingest is the sole writer to the CDM store; all other services are readers via this API only.

2. **Detection-frame buffer (legacy path).** Accepts SWIR detection frames at `/ingest/detection`, buffers them over a configurable window, and writes `states_multi.npz` for the propagator. This path predates the Tracker service and is not currently in the live pipeline, see "Known State" below.

## API

### CDM Endpoints (Primary)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/cdm/poll` | POST | Trigger a Space-Track CDM fetch by NORAD ID or Pc threshold |
| `/cdm/{primary_norad}/{secondary_norad}` | GET | Retrieve most recent CDM for an object pair, assembled covariance in km² |
| `/cdm/inject` | POST | Inject a CDM (CCSDS KVN) manually for testing and demo flows |
| `/planner_output` | POST | Record a planner decision against a CDM record (audit trail) |
| `/store/cdm_records` | GET | List persisted CDM records |
| `/store/planner_outputs` | GET | List planner decision records |
| `/store/cdm_records/duplicates` | DELETE | Deduplicate the CDM store |
| `/store/cdm_records/all` | DELETE | Purge all CDM records (use with caution) |

### Detection Endpoint (Legacy)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest/detection` | POST | Buffer a detection frame; writes `states_multi.npz` when buffer fills |
| `/health` | GET | Service health check |

## CDM Flow

1. Planner (or a scheduled poller) calls `POST /cdm/poll` with a target NORAD ID or Pc threshold.
2. `spacetrack_client.py` authenticates against Space-Track and pulls matching CDMs in KVN format.
3. `cdm_parser.py` parses each KVN into structured fields including the six lower-triangle RTN covariance elements for both primary and secondary objects.
4. Records are persisted to SQLite (`cdm_store` volume, `/data/cdm_store/`).
5. Planner retrieves covariance via `GET /cdm/{primary}/{secondary}`, which assembles `C_combined = C_primary + C_secondary` in km² (converted from the CCSDS m² storage unit).
6. Planner rotates RTN → ECI using the satellite r/v at TCA before passing to the APS decision model.

See `openapi/ingest.yaml` for the authoritative interface contract (v1.0.0).

### Unit Conventions

- Covariance stored in DB: m² (raw CCSDS 508.0-B-1 values)
- Covariance returned by the API: km² (divided by 1e6 on assembly, matching planner `p_rel_km2` convention)
- Miss distance: metres
- Times: UTC, ISO-8601
- Covariance frame: RTN (Radial / Transverse / Normal) per CCSDS 508.0-B-1

### Fallback Behaviour

If `GET /cdm/{primary}/{secondary}` returns 404 (no CDM on record) or 503 (store unavailable), the caller is responsible for falling back to a surrogate identity matrix and setting `covariance_source` to `surrogate_identity` in its own output. The ATLAS UI displays a visible warning when surrogate covariance is used.

## Detection Frame Schema

For reference, the `/ingest/detection` endpoint accepts:

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

The buffer processes frames into `states_multi.npz` under `/data/planner_artifacts/` when `BUFFER_WINDOW_SIZE` is reached.

## Known State

The detection-frame buffer path (`/ingest/detection` → `states_multi.npz`) is not wired into the live pipeline as of SCRUM-325. Detectors now forward detections to the Tracker service (`tracker:8000/detections`) directly. The Tracker service also writes `states_multi.npz` via its `POST /export/states` endpoint. This creates two producers of the same artifact file. Resolution of which service owns `states_multi.npz` going forward is captured under a separate architectural review; do not assume either path is canonical without checking current docker-compose wiring.

## Output Artifact

When `/ingest/detection` is used, writes `states_multi.npz` to `/data/planner_artifacts/`:

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
| `OUTPUT_DIR` | `/data/planner_artifacts` | Shared volume path for `states_multi.npz` |
| `BUFFER_WINDOW_SIZE` | `5` | Frames to buffer before assembly |
| `SPACETRACK_USER` | n/a | Space-Track.org account username (required for `/cdm/poll`) |
| `SPACETRACK_PASS` | n/a | Space-Track.org account password |

CDM store volume is mounted at `/data/cdm_store/` and persists across container restarts.

## Docker

```bash
docker build -t avera/ingest:v6 .
docker run -p 8001:8000 -v planner_data:/data/planner_artifacts -v cdm_data:/data/cdm_store avera/ingest:v6
```

Port mapping: container listens on 8000, exposed as 8001 to avoid conflict with detector.
