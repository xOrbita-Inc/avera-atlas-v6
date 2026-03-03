"""
AVERA-ATLAS Tracker Service - Data Models

Defines the input/output contracts for the tracker service.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
import numpy as np


# =============================================================================
# Enums
# =============================================================================

class TrackStatus(str, Enum):
    """Track lifecycle states."""
    UNCORRELATED = "UNCORRELATED"  # Detections awaiting sufficient obs for IOD
    TENTATIVE = "TENTATIVE"        # IOD attempted, not yet confirmed
    CONFIRMED = "CONFIRMED"        # Sufficient observations, high confidence
    COASTING = "COASTING"          # No recent observations, propagating
    DROPPED = "DROPPED"            # Track lost or merged


class AttitudeSource(str, Enum):
    """Source of platform attitude data."""
    STAR_TRACKER = "star_tracker"
    IMU = "imu"
    FUSED = "fused"
    MOCK = "mock"


class EphemerisSource(str, Enum):
    """Source of platform ephemeris data."""
    GPS = "gps"
    GROUND_UPDATE = "ground_update"
    PROPAGATED = "propagated"
    MOCK = "mock"


# =============================================================================
# Platform State (CubeSat observer)
# =============================================================================

@dataclass
class PlatformState:
    """
    State of the observing CubeSat platform at a given epoch.
    Required for transforming sensor-frame detections to inertial coordinates.
    """
    epoch: datetime
    
    # Position and velocity in ECI J2000 (meters, m/s)
    position_eci: np.ndarray          # [x, y, z]
    velocity_eci: np.ndarray          # [vx, vy, vz]
    position_covariance: np.ndarray   # 3x3 covariance matrix
    
    # Attitude: quaternion body-to-ECI [q0, q1, q2, q3] (scalar-first)
    quaternion_body_to_eci: np.ndarray
    attitude_covariance: np.ndarray   # 3x3 small-angle covariance
    
    # Data provenance
    ephemeris_source: EphemerisSource
    attitude_source: AttitudeSource
    
    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch.isoformat(),
            "position_eci": self.position_eci.tolist(),
            "velocity_eci": self.velocity_eci.tolist(),
            "position_covariance": self.position_covariance.flatten().tolist(),
            "quaternion_body_to_eci": self.quaternion_body_to_eci.tolist(),
            "attitude_covariance": self.attitude_covariance.flatten().tolist(),
            "ephemeris_source": self.ephemeris_source.value,
            "attitude_source": self.attitude_source.value,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PlatformState":
        return cls(
            epoch=datetime.fromisoformat(data["epoch"]),
            position_eci=np.array(data["position_eci"]),
            velocity_eci=np.array(data["velocity_eci"]),
            position_covariance=np.array(data["position_covariance"]).reshape(3, 3),
            quaternion_body_to_eci=np.array(data["quaternion_body_to_eci"]),
            attitude_covariance=np.array(data["attitude_covariance"]).reshape(3, 3),
            ephemeris_source=EphemerisSource(data["ephemeris_source"]),
            attitude_source=AttitudeSource(data["attitude_source"]),
        )


# =============================================================================
# Input: Detection from YOLOv8 Detector
# =============================================================================

@dataclass
class SensorDetection:
    """
    Raw detection from SWIR sensor + YOLOv8.
    This is the INPUT to the tracker from the detector service.
    
    Note: SWIR cameras only provide 2D measurements (pixel coordinates).
    The 3D position must be derived through IOD using multiple observations.
    """
    detection_id: UUID
    sensor_id: str
    timestamp: datetime
    
    # 2D measurement in image frame
    pixel_u: float
    pixel_v: float
    
    # Bounding box (for context, not used in tracking)
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    
    # Detection metadata
    confidence: float              # YOLOv8 confidence [0, 1]
    object_class: str              # e.g., "Debris", "CubeSat", "RocketBody"
    
    # Platform state at observation epoch (required for IOD)
    platform_state: Optional[PlatformState] = None
    
    def to_dict(self) -> dict:
        return {
            "detection_id": str(self.detection_id),
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp.isoformat(),
            "pixel_u": self.pixel_u,
            "pixel_v": self.pixel_v,
            "bbox_x": self.bbox_x,
            "bbox_y": self.bbox_y,
            "bbox_w": self.bbox_w,
            "bbox_h": self.bbox_h,
            "confidence": self.confidence,
            "object_class": self.object_class,
            "platform_state": self.platform_state.to_dict() if self.platform_state else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SensorDetection":
        return cls(
            detection_id=UUID(data["detection_id"]),
            sensor_id=data["sensor_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            pixel_u=data["pixel_u"],
            pixel_v=data["pixel_v"],
            bbox_x=data["bbox_x"],
            bbox_y=data["bbox_y"],
            bbox_w=data["bbox_w"],
            bbox_h=data["bbox_h"],
            confidence=data["confidence"],
            object_class=data["object_class"],
            platform_state=PlatformState.from_dict(data["platform_state"]) if data.get("platform_state") else None,
        )


# =============================================================================
# Intermediate: Angular Observation (after sensor-to-inertial transform)
# =============================================================================

@dataclass
class AngularObservation:
    """
    Observation transformed from pixel coordinates to inertial angles.
    Used internally for IOD and track association.
    """
    obs_id: UUID
    detection_id: UUID              # Link back to source detection
    sensor_id: str
    timestamp: datetime
    
    # Inertial angles (radians)
    right_ascension: float          # RA in ECI J2000
    declination: float              # Dec in ECI J2000
    
    # Angular rates if available (rad/s)
    ra_rate: Optional[float] = None
    dec_rate: Optional[float] = None
    
    # Measurement uncertainty (radians, 1-sigma)
    ra_sigma: float = 0.0001        # ~20 arcsec default
    dec_sigma: float = 0.0001
    
    # Observer state at this epoch (copied from platform state)
    observer_position_eci: np.ndarray = field(default_factory=lambda: np.zeros(3))
    observer_velocity_eci: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Original detection metadata
    confidence: float = 0.0
    object_class: str = "Unknown"


# =============================================================================
# Internal: Track (persistent state)
# =============================================================================

@dataclass
class Track:
    """
    Persistent track representing a tracked object.
    This is the internal state maintained by the tracker.
    """
    track_id: UUID
    status: TrackStatus
    
    # State vector in ECI J2000
    state_epoch: datetime
    position_eci: np.ndarray        # [x, y, z] meters
    velocity_eci: np.ndarray        # [vx, vy, vz] m/s
    covariance: np.ndarray          # 6x6 state covariance
    
    # Lifecycle metadata
    created_at: datetime
    updated_at: datetime
    last_observation: datetime
    observation_count: int
    
    # Track quality
    confidence_score: float         # Composite quality metric [0, 1]
    arc_length_deg: float           # Angular arc spanned by observations
    
    # Classification (from detections)
    object_class: str
    
    # Sensor contributions
    contributing_sensors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "track_id": str(self.track_id),
            "status": self.status.value,
            "state_epoch": self.state_epoch.isoformat(),
            "position_eci": self.position_eci.tolist(),
            "velocity_eci": self.velocity_eci.tolist(),
            "covariance": self.covariance.flatten().tolist(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_observation": self.last_observation.isoformat(),
            "observation_count": self.observation_count,
            "confidence_score": self.confidence_score,
            "arc_length_deg": self.arc_length_deg,
            "object_class": self.object_class,
            "contributing_sensors": self.contributing_sensors,
        }


# =============================================================================
# Output: Track State for Propagator (matches states_multi.npz format)
# =============================================================================

@dataclass
class TrackOutput:
    """
    Output format for confirmed tracks, ready for the propagator service.
    Matches the states_multi.npz schema expected by the propagator.
    """
    object_id: str                  # e.g., "TRK_001" or "Debris_001"
    
    # State vector in ECI (km, km/s to match propagator expectations)
    r_eci_km: np.ndarray            # [x, y, z] km
    v_eci_km_s: np.ndarray          # [vx, vy, vz] km/s
    
    # Covariance (6x6, km and km/s units)
    covariance_km: np.ndarray
    
    # Metadata
    epoch: datetime
    confidence: float
    object_class: str
    track_id: str
    status: str
    
    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "r_eci_km": self.r_eci_km.tolist(),
            "v_eci_km_s": self.v_eci_km_s.tolist(),
            "covariance_km": self.covariance_km.flatten().tolist(),
            "epoch": self.epoch.isoformat(),
            "confidence": self.confidence,
            "object_class": self.object_class,
            "track_id": self.track_id,
            "status": self.status,
        }


# =============================================================================
# Uncorrelated Detection (awaiting IOD)
# =============================================================================

@dataclass
class UncorrelatedDetection:
    """
    Detection that hasn't been associated to a track yet.
    Stored in a buffer awaiting sufficient observations for IOD.
    """
    uct_id: UUID                    # Uncorrelated track ID
    observations: list[AngularObservation] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_observation: datetime = field(default_factory=datetime.utcnow)
    
    # Tentative classification (most common from detections)
    object_class: str = "Unknown"
    
    # Association metadata
    candidate_track_ids: list[UUID] = field(default_factory=list)
    
    @property
    def observation_count(self) -> int:
        return len(self.observations)
    
    @property
    def arc_length_seconds(self) -> float:
        if len(self.observations) < 2:
            return 0.0
        times = [obs.timestamp for obs in self.observations]
        return (max(times) - min(times)).total_seconds()


# =============================================================================
# Sensor Registration
# =============================================================================

@dataclass
class SensorConfig:
    """
    Configuration for a registered sensor platform.
    """
    sensor_id: str
    platform_name: str              # e.g., "AVERA-SAT-01"
    
    # Camera intrinsics
    focal_length_mm: float
    pixel_size_um: float
    resolution_x: int
    resolution_y: int
    fov_x_deg: float
    fov_y_deg: float
    
    # Mounting (body frame offset)
    boresight_body: np.ndarray      # Unit vector in body frame
    
    # Status
    is_active: bool = True
    last_heartbeat: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "platform_name": self.platform_name,
            "focal_length_mm": self.focal_length_mm,
            "pixel_size_um": self.pixel_size_um,
            "resolution_x": self.resolution_x,
            "resolution_y": self.resolution_y,
            "fov_x_deg": self.fov_x_deg,
            "fov_y_deg": self.fov_y_deg,
            "boresight_body": self.boresight_body.tolist(),
            "is_active": self.is_active,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
        }
