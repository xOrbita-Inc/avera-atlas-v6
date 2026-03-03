"""
AVERA-ATLAS Tracker Service - API Schemas

Pydantic models for FastAPI request/response validation.
These define the OpenAPI contract for the tracker service.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# Enums (mirrored for Pydantic)
# =============================================================================

class TrackStatusEnum(str, Enum):
    UNCORRELATED = "UNCORRELATED"
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    COASTING = "COASTING"
    DROPPED = "DROPPED"


# =============================================================================
# Platform State Schema
# =============================================================================

class PlatformStateSchema(BaseModel):
    """Platform state at observation epoch."""
    epoch: datetime
    position_eci: list[float] = Field(..., min_length=3, max_length=3, description="[x, y, z] meters in ECI J2000")
    velocity_eci: list[float] = Field(..., min_length=3, max_length=3, description="[vx, vy, vz] m/s in ECI J2000")
    position_covariance: list[float] = Field(..., min_length=9, max_length=9, description="3x3 covariance, flattened")
    quaternion_body_to_eci: list[float] = Field(..., min_length=4, max_length=4, description="[q0, q1, q2, q3] scalar-first")
    attitude_covariance: list[float] = Field(..., min_length=9, max_length=9, description="3x3 covariance, flattened")
    ephemeris_source: str = Field("mock", description="Source: gps, ground_update, propagated, mock")
    attitude_source: str = Field("mock", description="Source: star_tracker, imu, fused, mock")

    class Config:
        json_schema_extra = {
            "example": {
                "epoch": "2025-01-15T12:00:00Z",
                "position_eci": [6778000.0, 0.0, 0.0],
                "velocity_eci": [0.0, 7668.0, 0.0],
                "position_covariance": [100, 0, 0, 0, 100, 0, 0, 0, 100],
                "quaternion_body_to_eci": [1.0, 0.0, 0.0, 0.0],
                "attitude_covariance": [1e-8, 0, 0, 0, 1e-8, 0, 0, 0, 1e-8],
                "ephemeris_source": "gps",
                "attitude_source": "star_tracker"
            }
        }


# =============================================================================
# Detection Input Schema (from detector service)
# =============================================================================

class DetectionInput(BaseModel):
    """
    Single detection from YOLOv8 detector.
    POST /detections endpoint accepts a list of these.
    """
    detection_id: str = Field(..., description="Unique detection identifier")
    sensor_id: str = Field(..., description="Sensor/platform identifier")
    timestamp: datetime = Field(..., description="Observation epoch UTC")
    
    # Pixel coordinates
    pixel_u: float = Field(..., description="Pixel X coordinate")
    pixel_v: float = Field(..., description="Pixel Y coordinate")
    
    # Bounding box
    bbox_x: float = Field(..., description="Bbox top-left X")
    bbox_y: float = Field(..., description="Bbox top-left Y")
    bbox_w: float = Field(..., description="Bbox width")
    bbox_h: float = Field(..., description="Bbox height")
    
    # Detection metadata
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    object_class: str = Field(..., description="Classification: Debris, CubeSat, RocketBody, etc.")
    
    # Platform state (required for tracking)
    platform_state: Optional[PlatformStateSchema] = Field(None, description="Platform state at observation epoch")

    class Config:
        json_schema_extra = {
            "example": {
                "detection_id": "det-001-abc",
                "sensor_id": "AVERA-SAT-01",
                "timestamp": "2025-01-15T12:00:00Z",
                "pixel_u": 512.5,
                "pixel_v": 384.2,
                "bbox_x": 500.0,
                "bbox_y": 370.0,
                "bbox_w": 25.0,
                "bbox_h": 28.0,
                "confidence": 0.87,
                "object_class": "Debris",
                "platform_state": None
            }
        }


class DetectionBatchInput(BaseModel):
    """Batch of detections from one or more sensors."""
    detections: list[DetectionInput] = Field(..., description="List of detections")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detections": [
                    {
                        "detection_id": "det-001",
                        "sensor_id": "AVERA-SAT-01",
                        "timestamp": "2025-01-15T12:00:00Z",
                        "pixel_u": 512.5,
                        "pixel_v": 384.2,
                        "bbox_x": 500.0,
                        "bbox_y": 370.0,
                        "bbox_w": 25.0,
                        "bbox_h": 28.0,
                        "confidence": 0.87,
                        "object_class": "Debris"
                    }
                ]
            }
        }


# =============================================================================
# Track Output Schema (to propagator service)
# =============================================================================

class TrackStateOutput(BaseModel):
    """
    Single track state for output to propagator.
    Matches the states_multi.npz format.
    """
    object_id: str = Field(..., description="Track identifier for propagator")
    track_id: str = Field(..., description="Internal track UUID")
    status: TrackStatusEnum = Field(..., description="Track lifecycle status")
    
    # State vector (km, km/s)
    epoch: datetime = Field(..., description="State epoch UTC")
    r_eci_km: list[float] = Field(..., min_length=3, max_length=3, description="Position [x, y, z] km")
    v_eci_km_s: list[float] = Field(..., min_length=3, max_length=3, description="Velocity [vx, vy, vz] km/s")
    covariance_km: list[float] = Field(..., min_length=36, max_length=36, description="6x6 covariance, flattened")
    
    # Metadata
    confidence: float = Field(..., ge=0.0, le=1.0, description="Track confidence score")
    object_class: str = Field(..., description="Object classification")
    observation_count: int = Field(..., description="Number of observations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "object_id": "Debris_001",
                "track_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "CONFIRMED",
                "epoch": "2025-01-15T12:00:00Z",
                "r_eci_km": [6778.0, 0.0, 0.0],
                "v_eci_km_s": [0.0, 7.668, 0.0],
                "covariance_km": [0.01] * 36,
                "confidence": 0.92,
                "object_class": "Debris",
                "observation_count": 5
            }
        }


class TracksOutput(BaseModel):
    """Response containing all confirmed/tentative tracks."""
    tracks: list[TrackStateOutput] = Field(..., description="List of track states")
    timestamp: datetime = Field(..., description="Response generation time")
    total_tracks: int = Field(..., description="Total track count")
    confirmed_count: int = Field(..., description="CONFIRMED tracks")
    tentative_count: int = Field(..., description="TENTATIVE tracks")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tracks": [],
                "timestamp": "2025-01-15T12:00:00Z",
                "total_tracks": 0,
                "confirmed_count": 0,
                "tentative_count": 0
            }
        }


# =============================================================================
# Track Detail Schema (for querying individual tracks)
# =============================================================================

class TrackDetailOutput(BaseModel):
    """Detailed track information including observation history."""
    track_id: str
    status: TrackStatusEnum
    
    # Current state
    state_epoch: datetime
    position_eci_km: list[float]
    velocity_eci_km_s: list[float]
    covariance_km: list[float]
    
    # Lifecycle
    created_at: datetime
    updated_at: datetime
    last_observation: datetime
    observation_count: int
    arc_length_deg: float
    
    # Quality
    confidence_score: float
    object_class: str
    contributing_sensors: list[str]


# =============================================================================
# Sensor Registration Schema
# =============================================================================

class SensorRegistration(BaseModel):
    """Register a new sensor platform."""
    sensor_id: str = Field(..., description="Unique sensor identifier")
    platform_name: str = Field(..., description="Platform name, e.g., AVERA-SAT-01")
    
    # Camera intrinsics
    focal_length_mm: float = Field(..., gt=0, description="Focal length in mm")
    pixel_size_um: float = Field(..., gt=0, description="Pixel size in micrometers")
    resolution_x: int = Field(..., gt=0, description="Horizontal resolution")
    resolution_y: int = Field(..., gt=0, description="Vertical resolution")
    fov_x_deg: float = Field(..., gt=0, description="Horizontal FOV in degrees")
    fov_y_deg: float = Field(..., gt=0, description="Vertical FOV in degrees")
    
    # Boresight in body frame
    boresight_body: list[float] = Field([0, 0, 1], min_length=3, max_length=3, description="Boresight unit vector")

    class Config:
        json_schema_extra = {
            "example": {
                "sensor_id": "AVERA-SAT-01-SWIR",
                "platform_name": "AVERA-SAT-01",
                "focal_length_mm": 50.0,
                "pixel_size_um": 15.0,
                "resolution_x": 1024,
                "resolution_y": 768,
                "fov_x_deg": 12.0,
                "fov_y_deg": 9.0,
                "boresight_body": [0.0, 0.0, 1.0]
            }
        }


class SensorStatusOutput(BaseModel):
    """Sensor status response."""
    sensor_id: str
    platform_name: str
    is_active: bool
    last_heartbeat: Optional[datetime]
    detection_count: int = Field(0, description="Total detections received")


# =============================================================================
# Service Status Schema
# =============================================================================

class ServiceStatus(BaseModel):
    """Tracker service health status."""
    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    timestamp: datetime
    version: str
    
    # Counts
    active_tracks: int
    confirmed_tracks: int
    tentative_tracks: int
    uncorrelated_detections: int
    registered_sensors: int
    
    # Performance
    detections_processed: int
    uptime_seconds: float


# =============================================================================
# Error Response Schema
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
    timestamp: datetime
