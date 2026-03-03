"""
AVERA-ATLAS Tracker Service - Sensor to Inertial Transformation

Transforms pixel coordinates from SWIR camera detections into
angular observations (Right Ascension / Declination) in the ECI J2000 frame.

Transformation Pipeline:
    Pixel (u, v) → Camera Frame → Body Frame → ECI Frame → RA/Dec

This is a critical step for angles-only tracking from space-based sensors.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
import numpy as np

from models import (
    SensorDetection,
    AngularObservation,
    PlatformState,
    SensorConfig,
)


# =============================================================================
# Camera Model
# =============================================================================

@dataclass
class CameraModel:
    """
    Pinhole camera model for SWIR sensor.
    
    Converts between pixel coordinates and direction vectors in camera frame.
    """
    focal_length_mm: float
    pixel_size_um: float
    resolution_x: int
    resolution_y: int
    
    # Principal point (optical center) - defaults to image center
    cx: Optional[float] = None
    cy: Optional[float] = None
    
    def __post_init__(self):
        # Default principal point to image center
        if self.cx is None:
            self.cx = self.resolution_x / 2.0
        if self.cy is None:
            self.cy = self.resolution_y / 2.0
        
        # Focal length in pixels
        self.focal_length_px = (self.focal_length_mm * 1000) / self.pixel_size_um
    
    def pixel_to_direction(self, u: float, v: float) -> np.ndarray:
        """
        Convert pixel coordinates to unit direction vector in camera frame.
        
        Camera frame convention:
            +X: right (increasing u)
            +Y: down (increasing v)
            +Z: boresight (into the scene)
        
        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            
        Returns:
            Unit direction vector [dx, dy, dz] in camera frame
        """
        # Offset from principal point
        x = u - self.cx
        y = v - self.cy
        
        # Direction vector (not normalized)
        d = np.array([x, y, self.focal_length_px])
        
        # Normalize to unit vector
        return d / np.linalg.norm(d)
    
    def direction_to_pixel(self, direction: np.ndarray) -> tuple[float, float]:
        """
        Convert direction vector in camera frame to pixel coordinates.
        
        Args:
            direction: Direction vector [dx, dy, dz] in camera frame
            
        Returns:
            (u, v) pixel coordinates
        """
        # Project onto image plane
        if direction[2] <= 0:
            raise ValueError("Direction vector points behind camera")
        
        scale = self.focal_length_px / direction[2]
        u = direction[0] * scale + self.cx
        v = direction[1] * scale + self.cy
        
        return (u, v)
    
    @classmethod
    def from_sensor_config(cls, config: SensorConfig) -> "CameraModel":
        """Create camera model from sensor configuration."""
        return cls(
            focal_length_mm=config.focal_length_mm,
            pixel_size_um=config.pixel_size_um,
            resolution_x=config.resolution_x,
            resolution_y=config.resolution_y,
        )


# =============================================================================
# Coordinate Frame Transformations
# =============================================================================

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion [q0, q1, q2, q3] (scalar-first convention)
           Represents rotation from body frame to reference frame.
           
    Returns:
        3x3 rotation matrix R such that v_ref = R @ v_body
    """
    q0, q1, q2, q3 = q / np.linalg.norm(q)  # Normalize
    
    R = np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [    2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2),     2*(q2*q3 - q0*q1)],
        [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    
    return R


def camera_to_body(
    direction_camera: np.ndarray,
    boresight_body: np.ndarray = np.array([0, 0, 1])
) -> np.ndarray:
    """
    Transform direction vector from camera frame to body frame.
    
    Assumes camera is mounted with boresight along specified body axis.
    Default is +Z body axis (nadir-pointing camera on nadir-pointing spacecraft).
    
    For more complex mounting, a full rotation matrix would be needed.
    
    Args:
        direction_camera: Unit direction vector in camera frame
        boresight_body: Camera boresight direction in body frame
        
    Returns:
        Unit direction vector in body frame
    """
    # Simple case: camera Z-axis aligned with body Z-axis
    # Camera frame: +X right, +Y down, +Z boresight
    # Body frame: +X forward, +Y right, +Z nadir (typical)
    
    # For a nadir-pointing camera on a nadir-pointing spacecraft:
    # Camera +X → Body +Y
    # Camera +Y → Body +X  
    # Camera +Z → Body +Z
    
    # This is a 90° rotation about Z
    R_cam_to_body = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    
    return R_cam_to_body @ direction_camera


def body_to_eci(
    direction_body: np.ndarray,
    quaternion_body_to_eci: np.ndarray
) -> np.ndarray:
    """
    Transform direction vector from body frame to ECI frame.
    
    Args:
        direction_body: Unit direction vector in body frame
        quaternion_body_to_eci: Attitude quaternion [q0, q1, q2, q3]
        
    Returns:
        Unit direction vector in ECI J2000 frame
    """
    R = quaternion_to_rotation_matrix(quaternion_body_to_eci)
    direction_eci = R @ direction_body
    
    return direction_eci / np.linalg.norm(direction_eci)


def eci_direction_to_ra_dec(direction_eci: np.ndarray) -> tuple[float, float]:
    """
    Convert ECI direction vector to Right Ascension and Declination.
    
    Args:
        direction_eci: Unit direction vector in ECI J2000 frame
        
    Returns:
        (ra, dec) in radians
            ra: Right Ascension [0, 2π)
            dec: Declination [-π/2, π/2]
    """
    x, y, z = direction_eci / np.linalg.norm(direction_eci)
    
    # Declination: angle from equatorial plane
    dec = math.asin(z)
    
    # Right Ascension: angle in equatorial plane from vernal equinox
    ra = math.atan2(y, x)
    if ra < 0:
        ra += 2 * math.pi
    
    return (ra, dec)


def ra_dec_to_eci_direction(ra: float, dec: float) -> np.ndarray:
    """
    Convert Right Ascension and Declination to ECI direction vector.
    
    Args:
        ra: Right Ascension in radians
        dec: Declination in radians
        
    Returns:
        Unit direction vector in ECI J2000 frame
    """
    x = math.cos(dec) * math.cos(ra)
    y = math.cos(dec) * math.sin(ra)
    z = math.sin(dec)
    
    return np.array([x, y, z])


# =============================================================================
# Angular Uncertainty Estimation
# =============================================================================

def estimate_angular_uncertainty(
    camera: CameraModel,
    pixel_u: float,
    pixel_v: float,
    detection_confidence: float,
    attitude_sigma_rad: float = 1e-4  # ~20 arcsec default
) -> tuple[float, float]:
    """
    Estimate angular measurement uncertainty.
    
    Sources of uncertainty:
    1. Pixel quantization (~0.5 pixel)
    2. Centroiding error (depends on SNR/confidence)
    3. Attitude knowledge error
    
    Args:
        camera: Camera model
        pixel_u, pixel_v: Detection pixel coordinates
        detection_confidence: YOLOv8 confidence [0, 1]
        attitude_sigma_rad: Attitude uncertainty (1-sigma, radians)
        
    Returns:
        (ra_sigma, dec_sigma) in radians
    """
    # Pixel uncertainty (centroiding error)
    # Higher confidence → better centroid → lower uncertainty
    # Typical: 0.1-1.0 pixels depending on SNR
    pixel_sigma = 0.5 / (detection_confidence + 0.1)
    
    # Convert pixel uncertainty to angular uncertainty
    # Small angle approximation: θ ≈ pixel_error / focal_length_px
    angular_sigma_pixel = pixel_sigma / camera.focal_length_px
    
    # Combine with attitude uncertainty (RSS)
    total_sigma = math.sqrt(angular_sigma_pixel**2 + attitude_sigma_rad**2)
    
    # For simplicity, assume equal uncertainty in RA and Dec
    # (In reality, this depends on position in FOV and attitude geometry)
    return (total_sigma, total_sigma)


# =============================================================================
# Main Transformation Class
# =============================================================================

class SensorToInertialTransformer:
    """
    Transforms pixel detections to angular observations in ECI frame.
    
    Usage:
        transformer = SensorToInertialTransformer()
        transformer.register_camera("AVERA-SAT-01-SWIR", camera_model)
        
        angular_obs = transformer.transform(detection, platform_state)
    """
    
    def __init__(self):
        self.cameras: dict[str, CameraModel] = {}
    
    def register_camera(self, sensor_id: str, camera: CameraModel):
        """Register a camera model for a sensor."""
        self.cameras[sensor_id] = camera
    
    def register_from_config(self, config: SensorConfig):
        """Register camera from sensor configuration."""
        self.cameras[config.sensor_id] = CameraModel.from_sensor_config(config)
    
    def transform(
        self,
        detection: SensorDetection,
        platform_state: PlatformState
    ) -> AngularObservation:
        """
        Transform a pixel detection to an angular observation.
        
        Args:
            detection: Raw sensor detection with pixel coordinates
            platform_state: Platform state at observation epoch
            
        Returns:
            AngularObservation in ECI J2000 frame
        """
        # Get camera model
        if detection.sensor_id not in self.cameras:
            raise ValueError(f"No camera model registered for sensor {detection.sensor_id}")
        
        camera = self.cameras[detection.sensor_id]
        
        # Step 1: Pixel → Camera frame direction
        direction_camera = camera.pixel_to_direction(
            detection.pixel_u,
            detection.pixel_v
        )
        
        # Step 2: Camera frame → Body frame
        direction_body = camera_to_body(direction_camera)
        
        # Step 3: Body frame → ECI frame
        direction_eci = body_to_eci(
            direction_body,
            platform_state.quaternion_body_to_eci
        )
        
        # Step 4: ECI direction → RA/Dec
        ra, dec = eci_direction_to_ra_dec(direction_eci)
        
        # Estimate uncertainties
        attitude_sigma = math.sqrt(platform_state.attitude_covariance[0, 0])
        ra_sigma, dec_sigma = estimate_angular_uncertainty(
            camera,
            detection.pixel_u,
            detection.pixel_v,
            detection.confidence,
            attitude_sigma
        )
        
        # Build angular observation
        return AngularObservation(
            obs_id=uuid4(),
            detection_id=detection.detection_id,
            sensor_id=detection.sensor_id,
            timestamp=detection.timestamp,
            right_ascension=ra,
            declination=dec,
            ra_rate=None,  # Would need consecutive frames to compute
            dec_rate=None,
            ra_sigma=ra_sigma,
            dec_sigma=dec_sigma,
            observer_position_eci=platform_state.position_eci,
            observer_velocity_eci=platform_state.velocity_eci,
            confidence=detection.confidence,
            object_class=detection.object_class,
        )
    
    def transform_batch(
        self,
        detections: list[SensorDetection],
        platform_states: dict[str, PlatformState]
    ) -> list[AngularObservation]:
        """
        Transform a batch of detections.
        
        Args:
            detections: List of sensor detections
            platform_states: Dict mapping sensor_id → platform_state
            
        Returns:
            List of angular observations
        """
        observations = []
        
        for det in detections:
            if det.sensor_id not in platform_states:
                continue  # Skip if no platform state available
            
            try:
                obs = self.transform(det, platform_states[det.sensor_id])
                observations.append(obs)
            except Exception as e:
                print(f"Warning: Failed to transform detection {det.detection_id}: {e}")
        
        return observations


# =============================================================================
# Utility Functions
# =============================================================================

def angular_separation(
    ra1: float, dec1: float,
    ra2: float, dec2: float
) -> float:
    """
    Compute angular separation between two directions.
    
    Uses the Vincenty formula for numerical stability.
    
    Args:
        ra1, dec1: First direction (radians)
        ra2, dec2: Second direction (radians)
        
    Returns:
        Angular separation in radians
    """
    delta_ra = ra2 - ra1
    
    numerator = math.sqrt(
        (math.cos(dec2) * math.sin(delta_ra))**2 +
        (math.cos(dec1) * math.sin(dec2) - 
         math.sin(dec1) * math.cos(dec2) * math.cos(delta_ra))**2
    )
    
    denominator = (
        math.sin(dec1) * math.sin(dec2) +
        math.cos(dec1) * math.cos(dec2) * math.cos(delta_ra)
    )
    
    return math.atan2(numerator, denominator)


def format_ra_dec(ra: float, dec: float) -> str:
    """
    Format RA/Dec for display.
    
    Args:
        ra: Right Ascension in radians
        dec: Declination in radians
        
    Returns:
        Formatted string "RA: HHh MMm SS.Ss, Dec: ±DD° MM' SS.S\""
    """
    # Convert RA to hours
    ra_hours = math.degrees(ra) / 15.0
    ra_h = int(ra_hours)
    ra_m = int((ra_hours - ra_h) * 60)
    ra_s = (ra_hours - ra_h - ra_m/60) * 3600
    
    # Convert Dec to degrees
    dec_deg = math.degrees(dec)
    dec_sign = '+' if dec_deg >= 0 else '-'
    dec_deg = abs(dec_deg)
    dec_d = int(dec_deg)
    dec_m = int((dec_deg - dec_d) * 60)
    dec_s = (dec_deg - dec_d - dec_m/60) * 3600
    
    return f"RA: {ra_h:02d}h {ra_m:02d}m {ra_s:05.2f}s, Dec: {dec_sign}{dec_d:02d}° {dec_m:02d}' {dec_s:04.1f}\""


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    from mock_platform import create_default_platforms
    from datetime import timezone
    
    print("=== Sensor to Inertial Transformation Test ===\n")
    
    # Create camera model
    camera = CameraModel(
        focal_length_mm=50.0,
        pixel_size_um=15.0,
        resolution_x=1024,
        resolution_y=768
    )
    
    print(f"Camera Model:")
    print(f"  Focal length: {camera.focal_length_mm} mm ({camera.focal_length_px:.1f} px)")
    print(f"  Resolution: {camera.resolution_x} x {camera.resolution_y}")
    print(f"  Principal point: ({camera.cx:.1f}, {camera.cy:.1f})")
    print()
    
    # Test pixel to direction
    test_pixels = [
        (512, 384),   # Center
        (0, 0),       # Top-left corner
        (1024, 768),  # Bottom-right corner
        (512, 0),     # Top center
    ]
    
    print("Pixel to Direction (Camera Frame):")
    for u, v in test_pixels:
        d = camera.pixel_to_direction(u, v)
        print(f"  ({u:4.0f}, {v:4.0f}) → [{d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}]")
    print()
    
    # Test full transformation with mock platform state
    print("Full Transformation Pipeline:")
    
    platform_gen = create_default_platforms()
    now = datetime.now(timezone.utc)
    
    transformer = SensorToInertialTransformer()
    transformer.register_camera("AVERA-SAT-01-SWIR", camera)
    
    # Get platform state
    platform_state = platform_gen.get_state("AVERA-SAT-01", now, add_noise=False)
    
    print(f"\nPlatform State (AVERA-SAT-01):")
    print(f"  Position: {platform_state.position_eci / 1000} km")
    print(f"  Quaternion: {platform_state.quaternion_body_to_eci}")
    print()
    
    # Create mock detection
    mock_detection = SensorDetection(
        detection_id=uuid4(),
        sensor_id="AVERA-SAT-01-SWIR",
        timestamp=now,
        pixel_u=512,
        pixel_v=384,
        bbox_x=500,
        bbox_y=370,
        bbox_w=25,
        bbox_h=28,
        confidence=0.87,
        object_class="Debris",
        platform_state=platform_state,
    )
    
    # Transform
    angular_obs = transformer.transform(mock_detection, platform_state)
    
    print("Detection → Angular Observation:")
    print(f"  Pixel: ({mock_detection.pixel_u}, {mock_detection.pixel_v})")
    print(f"  {format_ra_dec(angular_obs.right_ascension, angular_obs.declination)}")
    print(f"  RA: {math.degrees(angular_obs.right_ascension):.4f}°")
    print(f"  Dec: {math.degrees(angular_obs.declination):.4f}°")
    print(f"  Uncertainty (1σ): {math.degrees(angular_obs.ra_sigma)*3600:.1f} arcsec")
    print()
    
    # Test different pixel positions
    print("Pixel Position Sweep (same platform state):")
    for u, v in [(256, 192), (512, 384), (768, 576)]:
        det = SensorDetection(
            detection_id=uuid4(),
            sensor_id="AVERA-SAT-01-SWIR",
            timestamp=now,
            pixel_u=u, pixel_v=v,
            bbox_x=u-12, bbox_y=v-14, bbox_w=25, bbox_h=28,
            confidence=0.9,
            object_class="Debris",
        )
        obs = transformer.transform(det, platform_state)
        print(f"  ({u:4.0f}, {v:4.0f}) → RA: {math.degrees(obs.right_ascension):8.3f}°, Dec: {math.degrees(obs.declination):+8.3f}°")
    
    print()
    print("=== Test Complete ===")
