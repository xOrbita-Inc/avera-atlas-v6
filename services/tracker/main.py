"""
AVERA-ATLAS Tracker Service

Multi-sensor fusion and track management service for space debris tracking.
Sits between detector and propagator in the APS pipeline.

Pipeline: ingest → detector → [TRACKER] → propagator → viz → ui
"""

import os
import json
import math
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4, UUID
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    DetectionInput,
    DetectionBatchInput,
    TrackStateOutput,
    TracksOutput,
    TrackDetailOutput,
    SensorRegistration,
    SensorStatusOutput,
    ServiceStatus,
    ErrorResponse,
    TrackStatusEnum,
)
from models import SensorDetection, PlatformState, SensorConfig
from transform import SensorToInertialTransformer, CameraModel, format_ra_dec
from mock_platform import MockPlatformStateGenerator, MockPlatformConfig
from correlate import CorrelationEngine, CorrelationConfig, CorrelatedObservation
from iod import IODSolver, IODObservation, IODSolution

# =============================================================================
# Configuration
# =============================================================================

VERSION = "0.1.0"
SERVICE_NAME = "tracker"
DATA_DIR = os.getenv("DATA_DIR", "/data/planner_artifacts")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Track lifecycle parameters
MIN_OBSERVATIONS_FOR_IOD = 3          # Minimum obs before attempting IOD
MIN_OBSERVATIONS_FOR_CONFIRMED = 5    # Minimum obs for CONFIRMED status
COASTING_TIMEOUT_SECONDS = 300        # Time before CONFIRMED → COASTING
DROP_TIMEOUT_SECONDS = 600            # Time before COASTING → DROPPED

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(SERVICE_NAME)


# =============================================================================
# In-Memory State (would be database in production)
# =============================================================================

class TrackerState:
    """In-memory state for the tracker service."""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.detections_processed = 0
        
        # Registered sensors
        self.sensors: dict[str, dict] = {}
        
        # Active tracks (track_id -> track data)
        self.tracks: dict[str, dict] = {}
        
        # Track ID counter for friendly naming
        self.track_counter = 0
        
        # Sensor-to-inertial transformer
        self.transformer = SensorToInertialTransformer()
        
        # Mock platform state generator (for demo)
        self.platform_generator = MockPlatformStateGenerator()
        
        # Correlation engine for grouping observations
        self.correlation_engine = CorrelationEngine(CorrelationConfig(
            angular_gate_deg=5.0,
            temporal_window_sec=60.0,
            min_obs_for_iod=3,
            min_arc_length_deg=0.5,
        ))
        
        # IOD solver for orbit determination
        self.iod_solver = IODSolver()
    
    def get_next_track_name(self, object_class: str) -> str:
        """Generate friendly track name like Debris_001."""
        self.track_counter += 1
        return f"{object_class}_{self.track_counter:03d}"
    
    def register_sensor(self, config: SensorConfig):
        """Register sensor with transformer and platform generator."""
        # Register camera model with transformer
        camera = CameraModel.from_sensor_config(config)
        self.transformer.register_camera(config.sensor_id, camera)
        
        # Register mock platform (extract platform ID from sensor ID)
        # e.g., "AVERA-SAT-01-SWIR" -> "AVERA-SAT-01"
        platform_id = "-".join(config.sensor_id.split("-")[:-1])
        if platform_id and platform_id not in self.platform_generator.platforms:
            # Give each platform a different orbital position
            # Extract platform number to offset RAAN and true anomaly
            try:
                platform_num = int(platform_id.split("-")[-1])
            except (ValueError, IndexError):
                platform_num = 1
            
            # Offset each platform by 90° in RAAN and 45° in true anomaly
            raan_offset = (platform_num - 1) * 90.0
            ta_offset = (platform_num - 1) * 45.0
            
            self.platform_generator.add_platform(MockPlatformConfig(
                platform_id=platform_id,
                semi_major_axis_km=6778.0,
                inclination_deg=51.6,
                raan_deg=raan_offset,
                true_anomaly_deg=ta_offset,
            ))


# Global state instance
state = TrackerState()


# =============================================================================
# Lifespan (startup/shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info(f"Starting {SERVICE_NAME} service v{VERSION}")
    
    # Create data directory if needed
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info(f"Data directory: {DATA_DIR}")
    
    yield
    
    logger.info(f"Shutting down {SERVICE_NAME} service")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="AVERA-ATLAS Tracker Service",
    description="""
Multi-sensor fusion and track management for space debris tracking.

## Overview

The Tracker service sits between the YOLOv8 Detector and the Propagator in the 
AVERA-ATLAS pipeline. It performs:

- **Detection ingestion** from multiple CubeSat SWIR sensors
- **Sensor-to-inertial transformation** (pixel → angular coordinates)
- **Cross-sensor correlation** for multi-observer fusion
- **Initial Orbit Determination (IOD)** from angles-only observations
- **Track lifecycle management** (UNCORRELATED → TENTATIVE → CONFIRMED → COASTING)
- **State estimation** using Extended Kalman Filter

## Pipeline Position

```
ingest → detector → [TRACKER] → propagator → viz → ui
```

## Key Concepts

- **Detection**: Raw 2D observation from SWIR camera + YOLOv8
- **Track**: Persistent object state with 3D position/velocity
- **Uncorrelated Detection**: Observations not yet associated to a track
- **IOD**: Process of determining 3D orbit from 2D angle measurements
    """,
    version=VERSION,
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "service": SERVICE_NAME, "version": VERSION}


@app.get("/status", response_model=ServiceStatus, tags=["Health"])
async def service_status():
    """Detailed service status."""
    now = datetime.now(timezone.utc)
    
    confirmed = sum(1 for t in state.tracks.values() if t.get("status") == "CONFIRMED")
    tentative = sum(1 for t in state.tracks.values() if t.get("status") == "TENTATIVE")
    
    return ServiceStatus(
        status="healthy",
        timestamp=now,
        version=VERSION,
        active_tracks=len(state.tracks),
        confirmed_tracks=confirmed,
        tentative_tracks=tentative,
        uncorrelated_detections=len(state.correlation_engine.ucts),
        registered_sensors=len(state.sensors),
        detections_processed=state.detections_processed,
        uptime_seconds=(now - state.start_time).total_seconds()
    )


# =============================================================================
# Detection Ingestion Endpoints
# =============================================================================

@app.post("/detections", tags=["Detections"])
async def ingest_detections(batch: DetectionBatchInput):
    """
    Ingest a batch of detections from YOLOv8 detector(s).
    
    This is the primary input endpoint. Detections are:
    1. Validated and transformed to angular coordinates
    2. Correlated to existing UCTs or create new ones
    3. Check if UCTs are ready for IOD
    """
    now = datetime.now(timezone.utc)
    
    results = {
        "received": len(batch.detections),
        "processed": 0,
        "transformed": 0,
        "correlated": 0,
        "new_ucts": 0,
        "errors": 0,
        "timestamp": now.isoformat()
    }
    
    for det in batch.detections:
        try:
            state.detections_processed += 1
            results["processed"] += 1
            
            # Get platform ID from sensor ID (e.g., "AVERA-SAT-01-SWIR" -> "AVERA-SAT-01")
            platform_id = "-".join(det.sensor_id.split("-")[:-1])
            
            # Get mock platform state for this observation time
            platform_state = state.platform_generator.get_state(
                platform_id,
                det.timestamp,
                add_noise=True
            )
            
            if platform_state is None:
                logger.warning(f"No platform state for {platform_id}, skipping detection")
                results["errors"] += 1
                continue
            
            # Check if sensor is registered with transformer
            if det.sensor_id not in state.transformer.cameras:
                logger.warning(f"Sensor {det.sensor_id} not registered, skipping detection")
                results["errors"] += 1
                continue
            
            # Create SensorDetection from input
            # Generate a proper UUID if detection_id isn't a valid UUID
            try:
                det_uuid = UUID(det.detection_id)
            except (ValueError, AttributeError):
                # Use UUID5 with a namespace to create deterministic UUID from string
                det_uuid = uuid4()  # Or use uuid5 for deterministic mapping
            
            sensor_det = SensorDetection(
                detection_id=det_uuid,
                sensor_id=det.sensor_id,
                timestamp=det.timestamp,
                pixel_u=det.pixel_u,
                pixel_v=det.pixel_v,
                bbox_x=det.bbox_x,
                bbox_y=det.bbox_y,
                bbox_w=det.bbox_w,
                bbox_h=det.bbox_h,
                confidence=det.confidence,
                object_class=det.object_class,
                platform_state=platform_state,
            )
            
            # Transform to angular observation
            angular_obs = state.transformer.transform(sensor_det, platform_state)
            results["transformed"] += 1
            
            # Convert to CorrelatedObservation
            corr_obs = CorrelatedObservation.from_angular_observation(angular_obs)
            
            # Correlate with existing UCTs
            uct, is_new = state.correlation_engine.correlate(corr_obs)
            
            if is_new:
                results["new_ucts"] += 1
            else:
                results["correlated"] += 1
            
            # Update sensor detection count
            if det.sensor_id in state.sensors:
                state.sensors[det.sensor_id]["detection_count"] += 1
            
        except Exception as e:
            logger.error(f"Error processing detection {det.detection_id}: {e}")
            results["errors"] += 1
    
    # Add IOD-ready count to results
    results["iod_ready_ucts"] = len(state.correlation_engine.get_iod_ready())
    results["total_ucts"] = len(state.correlation_engine.ucts)
    
    return results


@app.post("/detections/single", tags=["Detections"])
async def ingest_single_detection(detection: DetectionInput):
    """Ingest a single detection (convenience endpoint)."""
    batch = DetectionBatchInput(detections=[detection])
    return await ingest_detections(batch)


# =============================================================================
# Track Query Endpoints
# =============================================================================

@app.get("/tracks", response_model=TracksOutput, tags=["Tracks"])
async def get_tracks(
    status: Optional[TrackStatusEnum] = Query(None, description="Filter by status"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence"),
):
    """
    Get all tracks matching filter criteria.
    
    Returns tracks in format compatible with propagator service.
    """
    now = datetime.now(timezone.utc)
    
    filtered_tracks = []
    for track_id, track in state.tracks.items():
        if status and track.get("status") != status.value:
            continue
        
        # Get epoch as datetime
        epoch_val = track.get("epoch", now)
        if isinstance(epoch_val, str):
            try:
                epoch_val = datetime.fromisoformat(epoch_val.replace('Z', '+00:00'))
            except:
                epoch_val = now
        
        filtered_tracks.append(TrackStateOutput(
            object_id=track.get("object_id", track.get("track_name", f"TRK_{track_id[:8]}")),
            track_id=track_id,
            status=TrackStatusEnum(track.get("status", "TENTATIVE")),
            epoch=epoch_val,
            r_eci_km=track.get("r_eci_km", [0, 0, 0]),
            v_eci_km_s=track.get("v_eci_km_s", [0, 0, 0]),
            covariance_km=track.get("covariance_km", [0.01] * 36),
            confidence=track.get("confidence", 0.8 if track.get("status") == "CONFIRMED" else 0.6),
            object_class=track.get("object_class", "Unknown"),
            observation_count=track.get("observations_used", track.get("observation_count", 0))
        ))
    
    confirmed = sum(1 for t in filtered_tracks if t.status == TrackStatusEnum.CONFIRMED)
    tentative = sum(1 for t in filtered_tracks if t.status == TrackStatusEnum.TENTATIVE)
    
    return TracksOutput(
        tracks=filtered_tracks,
        timestamp=now,
        total_tracks=len(filtered_tracks),
        confirmed_count=confirmed,
        tentative_count=tentative
    )


@app.get("/tracks/{track_id}", response_model=TrackDetailOutput, tags=["Tracks"])
async def get_track_detail(track_id: str):
    """Get detailed information for a specific track."""
    if track_id not in state.tracks:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")
    
    track = state.tracks[track_id]
    now = datetime.now(timezone.utc)
    
    return TrackDetailOutput(
        track_id=track_id,
        status=TrackStatusEnum(track.get("status", "TENTATIVE")),
        state_epoch=track.get("epoch", now),
        position_eci_km=track.get("r_eci_km", [0, 0, 0]),
        velocity_eci_km_s=track.get("v_eci_km_s", [0, 0, 0]),
        covariance_km=track.get("covariance_km", [0.01] * 36),
        created_at=track.get("created_at", now),
        updated_at=track.get("updated_at", now),
        last_observation=track.get("last_observation", now),
        observation_count=track.get("observation_count", 0),
        arc_length_deg=track.get("arc_length_deg", 0.0),
        confidence_score=track.get("confidence", 0.5),
        object_class=track.get("object_class", "Unknown"),
        contributing_sensors=track.get("contributing_sensors", [])
    )


@app.delete("/tracks/{track_id}", tags=["Tracks"])
async def delete_track(track_id: str):
    """Manually delete/drop a track."""
    if track_id not in state.tracks:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")
    
    del state.tracks[track_id]
    return {"status": "deleted", "track_id": track_id}


# =============================================================================
# Output to Propagator
# =============================================================================

@app.post("/export/states", tags=["Export"])
async def export_states_npz():
    """
    Export confirmed tracks to states_multi.npz for propagator.
    
    This writes the artifact that the propagator service consumes.
    """
    now = datetime.now(timezone.utc)
    
    # Filter to CONFIRMED and TENTATIVE tracks
    exportable = [
        t for t in state.tracks.values()
        if t.get("status") in ["CONFIRMED", "TENTATIVE"]
    ]
    
    if not exportable:
        return {
            "status": "no_tracks",
            "message": "No tracks available for export",
            "timestamp": now.isoformat()
        }
    
    # Build arrays matching states_multi.npz format
    object_ids = [t.get("object_id", "Unknown") for t in exportable]
    r_eci_km = np.array([t.get("r_eci_km", [0, 0, 0]) for t in exportable])
    v_eci_km_s = np.array([t.get("v_eci_km_s", [0, 0, 0]) for t in exportable])
    confidences = np.array([t.get("confidence", 0.5) for t in exportable])
    
    # Write NPZ
    out_path = os.path.join(DATA_DIR, "states_multi.npz")
    np.savez(
        out_path,
        object_ids=np.array(object_ids),
        r_eci_km=r_eci_km,
        v_eci_km_s=v_eci_km_s,
        confidences=confidences,
        t_window=np.array([60.0, 1440]),  # 1 minute steps, 24 hours
        metadata=json.dumps({
            "source": "tracker",
            "t0": now.isoformat(),
            "track_count": len(exportable)
        })
    )
    
    logger.info(f"Exported {len(exportable)} tracks to {out_path}")
    
    return {
        "status": "exported",
        "path": out_path,
        "track_count": len(exportable),
        "timestamp": now.isoformat()
    }


# =============================================================================
# Sensor Management Endpoints
# =============================================================================

@app.post("/sensors", tags=["Sensors"])
async def register_sensor(sensor: SensorRegistration):
    """Register a new sensor platform."""
    if sensor.sensor_id in state.sensors:
        raise HTTPException(
            status_code=400,
            detail=f"Sensor {sensor.sensor_id} already registered"
        )
    
    # Store sensor config
    state.sensors[sensor.sensor_id] = {
        **sensor.model_dump(),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "detection_count": 0
    }
    
    # Register with transformer and platform generator
    config = SensorConfig(
        sensor_id=sensor.sensor_id,
        platform_name=sensor.platform_name,
        focal_length_mm=sensor.focal_length_mm,
        pixel_size_um=sensor.pixel_size_um,
        resolution_x=sensor.resolution_x,
        resolution_y=sensor.resolution_y,
        fov_x_deg=sensor.fov_x_deg,
        fov_y_deg=sensor.fov_y_deg,
        boresight_body=np.array(sensor.boresight_body),
    )
    state.register_sensor(config)
    
    logger.info(f"Registered sensor: {sensor.sensor_id} ({sensor.platform_name})")
    return {"status": "registered", "sensor_id": sensor.sensor_id}


@app.get("/sensors", tags=["Sensors"])
async def list_sensors():
    """List all registered sensors."""
    return {
        "sensors": list(state.sensors.values()),
        "count": len(state.sensors)
    }


@app.get("/sensors/{sensor_id}", response_model=SensorStatusOutput, tags=["Sensors"])
async def get_sensor_status(sensor_id: str):
    """Get status of a specific sensor."""
    if sensor_id not in state.sensors:
        raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")
    
    s = state.sensors[sensor_id]
    return SensorStatusOutput(
        sensor_id=s["sensor_id"],
        platform_name=s["platform_name"],
        is_active=s.get("is_active", True),
        last_heartbeat=s.get("last_heartbeat"),
        detection_count=s.get("detection_count", 0)
    )


@app.post("/sensors/{sensor_id}/heartbeat", tags=["Sensors"])
async def sensor_heartbeat(sensor_id: str):
    """Update sensor heartbeat timestamp."""
    if sensor_id not in state.sensors:
        raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")
    
    state.sensors[sensor_id]["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
    return {"status": "ok", "sensor_id": sensor_id}


# =============================================================================
# Uncorrelated Detection Management
# =============================================================================

@app.get("/uncorrelated", tags=["Uncorrelated"])
async def get_uncorrelated():
    """Get all uncorrelated track buffers."""
    ucts = state.correlation_engine.get_all_ucts()
    
    buffers = []
    for uct in ucts:
        buffer_info = uct.to_dict()
        
        # Add first observation details
        if uct.first_observation:
            first = uct.first_observation
            buffer_info["first_obs_ra_deg"] = math.degrees(first.ra)
            buffer_info["first_obs_dec_deg"] = math.degrees(first.dec)
            buffer_info["first_obs_position_km"] = (first.observer_position_eci / 1000).tolist()
        
        # Add IOD-ready flag
        buffer_info["iod_ready"] = uct.is_ready_for_iod(state.correlation_engine.config)
        
        buffers.append(buffer_info)
    
    return {
        "count": len(ucts),
        "iod_ready_count": len(state.correlation_engine.get_iod_ready()),
        "buffers": buffers,
        "correlation_stats": state.correlation_engine.get_stats(),
    }


@app.post("/uncorrelated/{uct_id}/attempt_iod", tags=["Uncorrelated"])
async def attempt_iod(uct_id: str):
    """
    Manually trigger IOD attempt for an uncorrelated buffer.
    
    Requires minimum 3 observations spanning sufficient arc.
    """
    try:
        uct_uuid = UUID(uct_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid UCT ID format: {uct_id}")
    
    if uct_uuid not in state.correlation_engine.ucts:
        raise HTTPException(status_code=404, detail=f"UCT {uct_id} not found")
    
    uct = state.correlation_engine.ucts[uct_uuid]
    
    if not uct.is_ready_for_iod(state.correlation_engine.config):
        return {
            "status": "not_ready",
            "observation_count": uct.observation_count,
            "arc_length_deg": uct.arc_length_deg,
            "required_observations": state.correlation_engine.config.min_obs_for_iod,
            "required_arc_deg": state.correlation_engine.config.min_arc_length_deg,
        }
    
    # Convert CorrelatedObservations to IODObservations
    iod_observations = []
    for obs in uct.observations:
        iod_obs = IODObservation(
            timestamp=obs.timestamp,
            ra=obs.ra,
            dec=obs.dec,
            ra_sigma=obs.ra_sigma,
            dec_sigma=obs.dec_sigma,
            observer_position_km=obs.observer_position_eci / 1000,  # m to km
            observer_velocity_km_s=obs.observer_velocity_eci / 1000,  # m/s to km/s
        )
        iod_observations.append(iod_obs)
    
    # Perform IOD
    track_id = uuid4()
    solution = state.iod_solver.solve(iod_observations, track_id)
    
    if not solution.success:
        return {
            "status": "iod_failed",
            "error": solution.error_message,
            "observation_count": uct.observation_count,
            "arc_length_deg": uct.arc_length_deg,
        }
    
    # Success! Create track from solution
    track_name = state.get_next_track_name(uct.object_class)
    
    track_data = {
        "track_id": str(track_id),
        "track_name": track_name,
        "object_id": track_name,  # For propagator export
        "status": "TENTATIVE",
        "object_class": uct.object_class,
        "epoch": solution.epoch.isoformat(),
        "r_eci_km": solution.position_km.tolist(),
        "v_eci_km_s": solution.velocity_km_s.tolist(),
        "confidence": 0.8,  # Base confidence for newly tracked objects
        "semi_major_axis_km": solution.semi_major_axis_km,
        "eccentricity": solution.eccentricity,
        "inclination_deg": solution.inclination_deg,
        "raan_deg": solution.raan_deg,
        "arg_perigee_deg": solution.arg_perigee_deg,
        "true_anomaly_deg": solution.true_anomaly_deg,
        "rms_residual_arcsec": solution.rms_residual_arcsec,
        "observations_used": solution.observations_used,
        "sensors": list(uct.sensor_ids),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    
    state.tracks[str(track_id)] = track_data
    
    # Remove the UCT since it's now a track
    state.correlation_engine.remove_uct(uct_uuid)
    
    logger.info(f"IOD successful: {track_name} (a={solution.semi_major_axis_km:.1f} km, i={solution.inclination_deg:.1f}°)")
    
    return {
        "status": "success",
        "track_id": str(track_id),
        "track_name": track_name,
        "solution": solution.to_dict(),
    }


# =============================================================================
# Demo Mode Endpoints
# =============================================================================

@app.post("/demo/generate", tags=["Demo"])
async def generate_demo_detections(
    scenario: str = Query("conjunction", description="Demo scenario type"),
    count: int = Query(4, ge=1, le=10, description="Number of observation sets")
):
    """
    Generate synthetic detections for demo purposes.
    
    This creates realistic detection sequences that flow through the full pipeline:
    Detection → Transform → Correlate → IOD → Track → Export → Propagator
    """
    from datetime import timedelta
    
    now = datetime.now(timezone.utc)
    results = {
        "scenario": scenario,
        "generated_detections": 0,
        "ucts_created": 0,
        "tracks_created": 0,
        "timestamp": now.isoformat()
    }
    
    # Ensure we have registered sensors and platforms
    if not state.sensors:
        # Auto-register demo platforms and sensors
        for sat_id in ["AVERA-SAT-01", "AVERA-SAT-02"]:
            # Register the platform first
            if sat_id not in state.platform_generator.platforms:
                state.platform_generator.add_platform(MockPlatformConfig(
                    platform_id=sat_id,
                    semi_major_axis_km=6871.0,  # ~500 km altitude
                    eccentricity=0.001,
                    inclination_deg=51.6,  # ISS-like inclination
                    raan_deg=0.0,
                    arg_periapsis_deg=0.0,
                    true_anomaly_deg=0.0 if sat_id == "AVERA-SAT-01" else 90.0,  # Spread them out
                    epoch=now,
                    attitude_mode="nadir_pointing"
                ))
            
            # Register the sensor
            sensor_id = f"{sat_id}-SWIR"
            state.sensors[sensor_id] = {
                "sensor_id": sensor_id,
                "platform_name": sat_id,
                "focal_length_mm": 50.0,
                "pixel_size_um": 15.0,
                "resolution_x": 1024,
                "resolution_y": 768,
                "fov_x_deg": 12.0,
                "fov_y_deg": 9.0,
                "registered_at": now.isoformat()
            }
            
            camera = CameraModel(
                focal_length_mm=50.0,
                pixel_size_um=15.0,
                resolution_x=1024,
                resolution_y=768
            )
            state.transformer.register_camera(sensor_id, camera)
        
        logger.info("Auto-registered demo platforms and sensors")
    
    # Generate synthetic debris observations
    # Simulate multiple objects with different trajectories
    objects_config = [
        {"name": "Debris", "pixel_start": (300, 400), "velocity": (50, 30)},
        {"name": "CubeSat", "pixel_start": (600, 200), "velocity": (-30, 40)},
    ]
    
    if scenario == "conjunction":
        # Add a close-approach object
        objects_config.append(
            {"name": "Debris", "pixel_start": (500, 380), "velocity": (45, 35)}
        )
    
    for obj_config in objects_config:
        sensor_id = "AVERA-SAT-01-SWIR"
        platform_id = "AVERA-SAT-01"
        
        # Generate observation sequence
        for i in range(count):
            t = now + timedelta(seconds=i * 10)
            
            # Get platform state
            platform_state = state.platform_generator.get_state(platform_id, t, add_noise=True)
            
            # Calculate pixel position
            px = obj_config["pixel_start"][0] + i * obj_config["velocity"][0]
            py = obj_config["pixel_start"][1] + i * obj_config["velocity"][1]
            
            # Clamp to sensor bounds
            px = max(50, min(974, px))
            py = max(50, min(718, py))
            
            # Create detection
            detection = SensorDetection(
                detection_id=uuid4(),
                sensor_id=sensor_id,
                timestamp=t,
                pixel_u=float(px),
                pixel_v=float(py),
                bbox_x=float(px - 20),
                bbox_y=float(py - 20),
                bbox_w=40.0,
                bbox_h=40.0,
                confidence=0.85 + np.random.uniform(-0.1, 0.1),
                object_class=obj_config["name"]
            )
            
            # Process through pipeline
            try:
                # Transform to angular
                angular_obs = state.transformer.transform(detection, platform_state)
                
                # Correlate
                corr_obs = CorrelatedObservation(
                    obs_id=uuid4(),
                    detection_id=detection.detection_id,
                    sensor_id=sensor_id,
                    timestamp=t,
                    ra=angular_obs.right_ascension,
                    dec=angular_obs.declination,
                    ra_sigma=angular_obs.ra_sigma,
                    dec_sigma=angular_obs.dec_sigma,
                    observer_position_eci=angular_obs.observer_position_eci,
                    observer_velocity_eci=angular_obs.observer_velocity_eci,
                    confidence=detection.confidence,
                    object_class=detection.object_class
                )
                
                uct, is_new = state.correlation_engine.correlate(corr_obs)
                
                state.detections_processed += 1
                results["generated_detections"] += 1
                
                if is_new:
                    results["ucts_created"] += 1
                    
            except Exception as e:
                logger.warning(f"Error processing demo detection: {e}")
    
    # Attempt IOD on ready UCTs
    ready_ucts = state.correlation_engine.get_iod_ready()
    for uct in ready_ucts:
        try:
            # Build IOD observations
            iod_obs = [
                IODObservation(
                    timestamp=o.timestamp,
                    ra=o.ra,
                    dec=o.dec,
                    ra_sigma=o.ra_sigma,
                    dec_sigma=o.dec_sigma,
                    observer_position_km=o.observer_position_eci / 1000.0,
                    observer_velocity_km_s=o.observer_velocity_eci / 1000.0
                )
                for o in uct.observations
            ]
            
            # Run IOD
            track_id = uuid4()
            solution = state.iod_solver.solve(iod_obs, track_id)
            
            if solution.success:
                # Create track
                track_name = state.get_next_track_name(uct.object_class)
                track_data = {
                    "track_id": str(track_id),
                    "track_name": track_name,
                    "object_id": track_name,
                    "status": "TENTATIVE",
                    "object_class": uct.object_class,
                    "epoch": solution.epoch.isoformat(),
                    "r_eci_km": solution.position_km.tolist(),
                    "v_eci_km_s": solution.velocity_km_s.tolist(),
                    "confidence": 0.8,
                    "semi_major_axis_km": solution.semi_major_axis_km,
                    "eccentricity": solution.eccentricity,
                    "inclination_deg": solution.inclination_deg,
                    "observations_used": solution.observations_used,
                    "created_at": now.isoformat(),
                }
                state.tracks[str(track_id)] = track_data
                state.correlation_engine.remove_uct(uct.uct_id)
                results["tracks_created"] += 1
                
                logger.info(f"Demo IOD: Created {track_name}")
                
        except Exception as e:
            logger.warning(f"Demo IOD failed: {e}")
    
    # Auto-export if we have tracks
    if results["tracks_created"] > 0:
        try:
            exportable = [t for t in state.tracks.values() if t.get("status") in ["CONFIRMED", "TENTATIVE"]]
            if exportable:
                out_path = os.path.join(DATA_DIR, "states_multi.npz")
                np.savez(
                    out_path,
                    object_ids=np.array([t.get("object_id", "Unknown") for t in exportable]),
                    r_eci_km=np.array([t.get("r_eci_km", [0, 0, 0]) for t in exportable]),
                    v_eci_km_s=np.array([t.get("v_eci_km_s", [0, 0, 0]) for t in exportable]),
                    confidences=np.array([t.get("confidence", 0.5) for t in exportable]),
                    t_window=np.array([60.0, 1440]),
                    metadata=json.dumps({
                        "source": "tracker_demo",
                        "t0": now.isoformat(),
                        "track_count": len(exportable)
                    })
                )
                results["exported"] = True
                logger.info(f"Demo: Exported {len(exportable)} tracks to propagator")
        except Exception as e:
            logger.warning(f"Demo export failed: {e}")
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
