"""
AVERA-ATLAS Tracker Service - Mock Platform State Generator

Generates synthetic CubeSat ephemeris and attitude data for demo purposes.
This allows the tracker to function without real platform telemetry.

For production, this would be replaced by:
- GPS receiver data
- Star tracker attitude solutions
- Ground-uploaded ephemeris
"""

import math
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional
import numpy as np

from models import PlatformState, EphemerisSource, AttitudeSource


# =============================================================================
# Orbital Constants
# =============================================================================

MU_EARTH = 3.986004418e14  # Earth gravitational parameter (m³/s²)
R_EARTH = 6.371e6          # Earth radius (m)
J2 = 1.08263e-3            # Earth J2 perturbation coefficient


# =============================================================================
# Mock Platform Configuration
# =============================================================================

@dataclass
class MockPlatformConfig:
    """Configuration for a mock CubeSat platform."""
    platform_id: str
    
    # Orbital elements (simplified Keplerian)
    semi_major_axis_km: float = 6778.0      # ~400 km altitude
    eccentricity: float = 0.001
    inclination_deg: float = 51.6           # ISS-like inclination
    raan_deg: float = 0.0                   # Right Ascension of Ascending Node
    arg_periapsis_deg: float = 0.0
    true_anomaly_deg: float = 0.0           # Initial position
    
    # Epoch for orbital elements
    epoch: datetime = None
    
    # Attitude mode
    attitude_mode: str = "nadir_pointing"   # nadir_pointing, inertial, target_track
    
    # Uncertainties (1-sigma)
    position_sigma_m: float = 10.0          # GPS-level accuracy
    velocity_sigma_m_s: float = 0.1
    attitude_sigma_arcsec: float = 30.0     # Star tracker accuracy
    
    def __post_init__(self):
        if self.epoch is None:
            self.epoch = datetime.now(timezone.utc)


# =============================================================================
# Keplerian Propagation
# =============================================================================

def kepler_to_cartesian(
    a: float,           # semi-major axis (m)
    e: float,           # eccentricity
    i: float,           # inclination (rad)
    raan: float,        # RAAN (rad)
    omega: float,       # argument of periapsis (rad)
    nu: float           # true anomaly (rad)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Keplerian elements to ECI position and velocity.
    
    Returns:
        position: [x, y, z] in meters
        velocity: [vx, vy, vz] in m/s
    """
    # Orbital parameter
    p = a * (1 - e**2)
    
    # Position in perifocal frame
    r_mag = p / (1 + e * math.cos(nu))
    r_pqw = np.array([
        r_mag * math.cos(nu),
        r_mag * math.sin(nu),
        0.0
    ])
    
    # Velocity in perifocal frame
    h = math.sqrt(MU_EARTH * p)
    v_pqw = np.array([
        -MU_EARTH / h * math.sin(nu),
        MU_EARTH / h * (e + math.cos(nu)),
        0.0
    ])
    
    # Rotation matrix from perifocal to ECI
    cos_raan, sin_raan = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(i), math.sin(i)
    cos_omega, sin_omega = math.cos(omega), math.sin(omega)
    
    R = np.array([
        [cos_raan * cos_omega - sin_raan * sin_omega * cos_i,
         -cos_raan * sin_omega - sin_raan * cos_omega * cos_i,
         sin_raan * sin_i],
        [sin_raan * cos_omega + cos_raan * sin_omega * cos_i,
         -sin_raan * sin_omega + cos_raan * cos_omega * cos_i,
         -cos_raan * sin_i],
        [sin_omega * sin_i,
         cos_omega * sin_i,
         cos_i]
    ])
    
    position = R @ r_pqw
    velocity = R @ v_pqw
    
    return position, velocity


def propagate_kepler(
    config: MockPlatformConfig,
    target_time: datetime
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate platform state to target time using Keplerian dynamics.
    
    Returns:
        position: [x, y, z] in meters
        velocity: [vx, vy, vz] in m/s
    """
    # Convert to SI units and radians
    a = config.semi_major_axis_km * 1000
    e = config.eccentricity
    i = math.radians(config.inclination_deg)
    raan = math.radians(config.raan_deg)
    omega = math.radians(config.arg_periapsis_deg)
    
    # Time since epoch
    dt = (target_time - config.epoch).total_seconds()
    
    # Mean motion
    n = math.sqrt(MU_EARTH / a**3)
    
    # Mean anomaly at epoch (from true anomaly)
    nu0 = math.radians(config.true_anomaly_deg)
    E0 = 2 * math.atan2(
        math.sqrt(1 - e) * math.sin(nu0 / 2),
        math.sqrt(1 + e) * math.cos(nu0 / 2)
    )
    M0 = E0 - e * math.sin(E0)
    
    # Mean anomaly at target time
    M = M0 + n * dt
    M = M % (2 * math.pi)
    
    # Solve Kepler's equation for eccentric anomaly
    E = M
    for _ in range(10):
        E = M + e * math.sin(E)
    
    # True anomaly at target time
    nu = 2 * math.atan2(
        math.sqrt(1 + e) * math.sin(E / 2),
        math.sqrt(1 - e) * math.cos(E / 2)
    )
    
    return kepler_to_cartesian(a, e, i, raan, omega, nu)


# =============================================================================
# Attitude Generation
# =============================================================================

def generate_nadir_pointing_attitude(
    position: np.ndarray,
    velocity: np.ndarray
) -> np.ndarray:
    """
    Generate quaternion for nadir-pointing attitude.
    
    Body frame: +Z points to Earth (nadir), +X in velocity direction
    
    Returns:
        quaternion: [q0, q1, q2, q3] scalar-first, body-to-ECI
    """
    # Nadir direction (negative of position unit vector)
    z_body = -position / np.linalg.norm(position)
    
    # Approximate velocity direction for X axis
    v_unit = velocity / np.linalg.norm(velocity)
    
    # Y axis completes right-handed system
    y_body = np.cross(z_body, v_unit)
    y_body = y_body / np.linalg.norm(y_body)
    
    # Recompute X to ensure orthogonality
    x_body = np.cross(y_body, z_body)
    
    # Rotation matrix (body to ECI)
    R = np.column_stack([x_body, y_body, z_body])
    
    # Convert to quaternion
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q0 = 0.25 / s
        q1 = (R[2, 1] - R[1, 2]) * s
        q2 = (R[0, 2] - R[2, 0]) * s
        q3 = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q0 = (R[2, 1] - R[1, 2]) / s
            q1 = 0.25 * s
            q2 = (R[0, 1] + R[1, 0]) / s
            q3 = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q0 = (R[0, 2] - R[2, 0]) / s
            q1 = (R[0, 1] + R[1, 0]) / s
            q2 = 0.25 * s
            q3 = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q0 = (R[1, 0] - R[0, 1]) / s
            q1 = (R[0, 2] + R[2, 0]) / s
            q2 = (R[1, 2] + R[2, 1]) / s
            q3 = 0.25 * s
    
    q = np.array([q0, q1, q2, q3])
    return q / np.linalg.norm(q)


# =============================================================================
# Mock Platform State Generator
# =============================================================================

class MockPlatformStateGenerator:
    """
    Generates synthetic platform state data for demo purposes.
    
    Usage:
        generator = MockPlatformStateGenerator()
        generator.add_platform(MockPlatformConfig(
            platform_id="AVERA-SAT-01",
            semi_major_axis_km=6778.0,
            inclination_deg=51.6
        ))
        
        state = generator.get_state("AVERA-SAT-01", datetime.now(timezone.utc))
    """
    
    def __init__(self):
        self.platforms: dict[str, MockPlatformConfig] = {}
    
    def add_platform(self, config: MockPlatformConfig):
        """Register a mock platform."""
        self.platforms[config.platform_id] = config
    
    def get_state(
        self,
        platform_id: str,
        target_time: datetime,
        add_noise: bool = True
    ) -> Optional[PlatformState]:
        """
        Generate platform state at the specified time.
        
        Args:
            platform_id: Platform identifier
            target_time: Time for state
            add_noise: Whether to add realistic measurement noise
            
        Returns:
            PlatformState or None if platform not found
        """
        if platform_id not in self.platforms:
            return None
        
        config = self.platforms[platform_id]
        
        # Propagate orbit
        position, velocity = propagate_kepler(config, target_time)
        
        # Generate attitude
        quaternion = generate_nadir_pointing_attitude(position, velocity)
        
        # Add noise if requested
        if add_noise:
            position += np.random.normal(0, config.position_sigma_m, 3)
            velocity += np.random.normal(0, config.velocity_sigma_m_s, 3)
            
            # Small attitude perturbation
            attitude_noise_rad = config.attitude_sigma_arcsec * 4.848e-6  # arcsec to rad
            euler_noise = np.random.normal(0, attitude_noise_rad, 3)
            # Apply small rotation (simplified)
            quaternion[1:] += euler_noise * quaternion[0]
            quaternion = quaternion / np.linalg.norm(quaternion)
        
        # Build covariance matrices
        pos_cov = np.diag([config.position_sigma_m**2] * 3)
        att_cov = np.diag([(config.attitude_sigma_arcsec * 4.848e-6)**2] * 3)
        
        return PlatformState(
            epoch=target_time,
            position_eci=position,
            velocity_eci=velocity,
            position_covariance=pos_cov,
            quaternion_body_to_eci=quaternion,
            attitude_covariance=att_cov,
            ephemeris_source=EphemerisSource.MOCK,
            attitude_source=AttitudeSource.MOCK
        )
    
    def get_state_series(
        self,
        platform_id: str,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: float = 1.0
    ) -> list[PlatformState]:
        """Generate a time series of platform states."""
        states = []
        current = start_time
        
        while current <= end_time:
            state = self.get_state(platform_id, current)
            if state:
                states.append(state)
            current += timedelta(seconds=interval_seconds)
        
        return states


# =============================================================================
# Default Platform Configurations
# =============================================================================

def create_default_platforms() -> MockPlatformStateGenerator:
    """
    Create generator with default AVERA constellation platforms.
    
    Sets up a realistic LEO constellation geometry for demo.
    """
    generator = MockPlatformStateGenerator()
    
    # Primary observation platform
    generator.add_platform(MockPlatformConfig(
        platform_id="AVERA-SAT-01",
        semi_major_axis_km=6778.0,       # ~400 km altitude
        eccentricity=0.0001,
        inclination_deg=51.6,
        raan_deg=0.0,
        true_anomaly_deg=0.0,
        position_sigma_m=5.0,            # Good GPS
        attitude_sigma_arcsec=20.0       # Good star tracker
    ))
    
    # Secondary platform (different orbital plane)
    generator.add_platform(MockPlatformConfig(
        platform_id="AVERA-SAT-02",
        semi_major_axis_km=6778.0,
        eccentricity=0.0001,
        inclination_deg=51.6,
        raan_deg=90.0,                   # Different RAAN
        true_anomaly_deg=45.0,
        position_sigma_m=5.0,
        attitude_sigma_arcsec=20.0
    ))
    
    # Third platform (sun-synchronous orbit)
    generator.add_platform(MockPlatformConfig(
        platform_id="AVERA-SAT-03",
        semi_major_axis_km=6928.0,       # ~550 km altitude
        eccentricity=0.0001,
        inclination_deg=97.5,            # SSO
        raan_deg=180.0,
        true_anomaly_deg=120.0,
        position_sigma_m=5.0,
        attitude_sigma_arcsec=20.0
    ))
    
    return generator


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    # Test the mock generator
    generator = create_default_platforms()
    
    now = datetime.now(timezone.utc)
    
    print("=== Mock Platform State Generator Test ===\n")
    
    for platform_id in ["AVERA-SAT-01", "AVERA-SAT-02", "AVERA-SAT-03"]:
        state = generator.get_state(platform_id, now, add_noise=False)
        
        if state:
            pos_km = state.position_eci / 1000
            vel_km_s = state.velocity_eci / 1000
            alt_km = np.linalg.norm(state.position_eci) / 1000 - 6371
            
            print(f"Platform: {platform_id}")
            print(f"  Position (km): [{pos_km[0]:.1f}, {pos_km[1]:.1f}, {pos_km[2]:.1f}]")
            print(f"  Velocity (km/s): [{vel_km_s[0]:.3f}, {vel_km_s[1]:.3f}, {vel_km_s[2]:.3f}]")
            print(f"  Altitude (km): {alt_km:.1f}")
            print(f"  Speed (km/s): {np.linalg.norm(vel_km_s):.3f}")
            print(f"  Quaternion: {state.quaternion_body_to_eci}")
            print()
