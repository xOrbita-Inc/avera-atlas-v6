"""
AVERA-ATLAS Tracker Service - Detection Correlation

Correlates angular observations that are likely from the same object.
Groups observations into Uncorrelated Track (UCT) buffers for IOD.

Correlation Strategy:
1. Angular gating - observations within threshold angular distance
2. Temporal window - observations within time window
3. Motion consistency - angular rate consistent with orbital motion
4. Cross-sensor fusion - correlate observations from different platforms
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID, uuid4
import numpy as np

from transform import angular_separation


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CorrelationConfig:
    """Configuration for detection correlation."""
    
    # Angular gating threshold (radians)
    # Objects in LEO can move 0.1-2 deg/sec as seen from another LEO platform
    # For 1 second interval, allow up to 3 degrees of motion
    angular_gate_deg: float = 5.0
    
    # Temporal window for correlation (seconds)
    # Only correlate observations within this time window
    temporal_window_sec: float = 60.0
    
    # Maximum angular rate (deg/sec) for valid orbital motion
    # LEO objects typically move 0.1-2 deg/sec as seen from LEO
    max_angular_rate_deg_sec: float = 3.0
    
    # Minimum angular rate (deg/sec) - too slow might be noise
    min_angular_rate_deg_sec: float = 0.01
    
    # Minimum observations before attempting IOD
    min_obs_for_iod: int = 3
    
    # Minimum arc length (degrees) for IOD
    min_arc_length_deg: float = 0.5
    
    # Maximum time gap (seconds) before starting new UCT
    max_gap_sec: float = 120.0
    
    # Cross-sensor correlation: allow observations from different sensors
    allow_cross_sensor: bool = True
    
    @property
    def angular_gate_rad(self) -> float:
        return math.radians(self.angular_gate_deg)
    
    @property
    def max_angular_rate_rad_sec(self) -> float:
        return math.radians(self.max_angular_rate_deg_sec)
    
    @property
    def min_angular_rate_rad_sec(self) -> float:
        return math.radians(self.min_angular_rate_deg_sec)


# =============================================================================
# Observation Data Structure
# =============================================================================

@dataclass
class CorrelatedObservation:
    """
    Angular observation with correlation metadata.
    """
    obs_id: UUID
    detection_id: UUID
    sensor_id: str
    timestamp: datetime
    
    # Angular position (radians)
    ra: float
    dec: float
    
    # Uncertainty (radians)
    ra_sigma: float
    dec_sigma: float
    
    # Observer state
    observer_position_eci: np.ndarray
    observer_velocity_eci: np.ndarray
    
    # Detection metadata
    confidence: float
    object_class: str
    
    def to_dict(self) -> dict:
        return {
            "obs_id": str(self.obs_id),
            "detection_id": str(self.detection_id),
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp.isoformat(),
            "ra_deg": math.degrees(self.ra),
            "dec_deg": math.degrees(self.dec),
            "ra_sigma_arcsec": math.degrees(self.ra_sigma) * 3600,
            "dec_sigma_arcsec": math.degrees(self.dec_sigma) * 3600,
            "observer_position_km": (self.observer_position_eci / 1000).tolist(),
            "confidence": self.confidence,
            "object_class": self.object_class,
        }
    
    @classmethod
    def from_angular_observation(cls, obs) -> "CorrelatedObservation":
        """Create from AngularObservation model."""
        return cls(
            obs_id=obs.obs_id,
            detection_id=obs.detection_id,
            sensor_id=obs.sensor_id,
            timestamp=obs.timestamp,
            ra=obs.right_ascension,
            dec=obs.declination,
            ra_sigma=obs.ra_sigma,
            dec_sigma=obs.dec_sigma,
            observer_position_eci=obs.observer_position_eci,
            observer_velocity_eci=obs.observer_velocity_eci,
            confidence=obs.confidence,
            object_class=obs.object_class,
        )


# =============================================================================
# Uncorrelated Track Buffer
# =============================================================================

@dataclass
class UncorrelatedTrack:
    """
    Buffer of correlated observations awaiting IOD.
    
    Once sufficient observations are accumulated with enough arc,
    IOD can be attempted to promote this to a proper track.
    """
    uct_id: UUID
    observations: list[CorrelatedObservation] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Classification (majority vote from observations)
    object_class: str = "Unknown"
    
    # Contributing sensors
    sensor_ids: set = field(default_factory=set)
    
    def add_observation(self, obs: CorrelatedObservation):
        """Add observation and update metadata."""
        self.observations.append(obs)
        self.sensor_ids.add(obs.sensor_id)
        
        # Update object class by majority vote
        class_counts = {}
        for o in self.observations:
            class_counts[o.object_class] = class_counts.get(o.object_class, 0) + 1
        self.object_class = max(class_counts, key=class_counts.get)
    
    @property
    def observation_count(self) -> int:
        return len(self.observations)
    
    @property
    def first_observation(self) -> Optional[CorrelatedObservation]:
        return self.observations[0] if self.observations else None
    
    @property
    def last_observation(self) -> Optional[CorrelatedObservation]:
        return self.observations[-1] if self.observations else None
    
    @property
    def time_span_sec(self) -> float:
        """Time span of observations in seconds."""
        if len(self.observations) < 2:
            return 0.0
        times = [o.timestamp for o in self.observations]
        return (max(times) - min(times)).total_seconds()
    
    @property
    def arc_length_deg(self) -> float:
        """Angular arc length spanned by observations in degrees."""
        if len(self.observations) < 2:
            return 0.0
        
        # Compute max angular separation between any two observations
        max_sep = 0.0
        for i, o1 in enumerate(self.observations):
            for o2 in self.observations[i+1:]:
                sep = angular_separation(o1.ra, o1.dec, o2.ra, o2.dec)
                max_sep = max(max_sep, sep)
        
        return math.degrees(max_sep)
    
    @property
    def mean_angular_rate_deg_sec(self) -> float:
        """Mean angular rate in deg/sec."""
        if self.time_span_sec < 0.1:
            return 0.0
        return self.arc_length_deg / self.time_span_sec
    
    def is_ready_for_iod(self, config: CorrelationConfig) -> bool:
        """Check if buffer has sufficient data for IOD."""
        return (
            self.observation_count >= config.min_obs_for_iod and
            self.arc_length_deg >= config.min_arc_length_deg
        )
    
    def to_dict(self) -> dict:
        return {
            "uct_id": str(self.uct_id),
            "observation_count": self.observation_count,
            "object_class": self.object_class,
            "sensor_ids": list(self.sensor_ids),
            "time_span_sec": self.time_span_sec,
            "arc_length_deg": self.arc_length_deg,
            "mean_angular_rate_deg_sec": self.mean_angular_rate_deg_sec,
            "first_observation": self.first_observation.timestamp.isoformat() if self.first_observation else None,
            "last_observation": self.last_observation.timestamp.isoformat() if self.last_observation else None,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Correlation Engine
# =============================================================================

class CorrelationEngine:
    """
    Correlates angular observations into UCT buffers.
    
    Usage:
        engine = CorrelationEngine()
        
        # Process each observation
        uct, is_new = engine.correlate(angular_obs)
        
        # Check for IOD-ready buffers
        ready = engine.get_iod_ready()
    """
    
    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.config = config or CorrelationConfig()
        self.ucts: dict[UUID, UncorrelatedTrack] = {}
        self.stats = {
            "observations_processed": 0,
            "new_ucts_created": 0,
            "correlations_made": 0,
        }
    
    def correlate(self, obs: CorrelatedObservation) -> tuple[UncorrelatedTrack, bool]:
        """
        Correlate an observation to existing UCT or create new one.
        
        Args:
            obs: Angular observation to correlate
            
        Returns:
            (uct, is_new): The UCT buffer and whether it was newly created
        """
        self.stats["observations_processed"] += 1
        
        # Find best matching UCT
        best_uct = None
        best_score = float('inf')
        
        for uct in self.ucts.values():
            score = self._compute_correlation_score(obs, uct)
            if score is not None and score < best_score:
                best_score = score
                best_uct = uct
        
        if best_uct is not None:
            # Add to existing UCT
            best_uct.add_observation(obs)
            self.stats["correlations_made"] += 1
            return (best_uct, False)
        else:
            # Create new UCT
            uct = UncorrelatedTrack(
                uct_id=uuid4(),
                created_at=datetime.now(timezone.utc),
            )
            uct.add_observation(obs)
            self.ucts[uct.uct_id] = uct
            self.stats["new_ucts_created"] += 1
            return (uct, True)
    
    def _compute_correlation_score(
        self,
        obs: CorrelatedObservation,
        uct: UncorrelatedTrack
    ) -> Optional[float]:
        """
        Compute correlation score between observation and UCT.
        
        Returns:
            Score (lower is better) or None if correlation rejected
        """
        if not uct.observations:
            return None
        
        last_obs = uct.last_observation
        
        # Check sensor compatibility
        if not self.config.allow_cross_sensor:
            if obs.sensor_id != last_obs.sensor_id:
                return None
        
        # Check temporal window
        time_diff = abs((obs.timestamp - last_obs.timestamp).total_seconds())
        if time_diff > self.config.temporal_window_sec:
            return None
        
        # Check for time gap (might be a new pass)
        if time_diff > self.config.max_gap_sec:
            return None
        
        # Compute angular separation from last observation
        ang_sep = angular_separation(
            obs.ra, obs.dec,
            last_obs.ra, last_obs.dec
        )
        
        # Check angular gate
        if ang_sep > self.config.angular_gate_rad:
            return None
        
        # Check angular rate consistency
        if time_diff > 0.1:  # Need some time difference
            angular_rate = ang_sep / time_diff
            
            if angular_rate > self.config.max_angular_rate_rad_sec:
                return None
            
            # Very slow motion might be noise (but allow for near-stationary)
            # Don't reject on min rate for now - could be valid geometry
        
        # Predict position based on linear extrapolation from UCT
        if len(uct.observations) >= 2:
            predicted_ra, predicted_dec = self._predict_position(uct, obs.timestamp)
            if predicted_ra is not None:
                pred_sep = angular_separation(
                    obs.ra, obs.dec,
                    predicted_ra, predicted_dec
                )
                # Use prediction residual as score
                return pred_sep
        
        # Fall back to angular separation as score
        return ang_sep
    
    def _predict_position(
        self,
        uct: UncorrelatedTrack,
        target_time: datetime
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Predict angular position at target time using linear extrapolation.
        
        Returns:
            (predicted_ra, predicted_dec) in radians, or (None, None)
        """
        if len(uct.observations) < 2:
            return (None, None)
        
        # Use last two observations for linear extrapolation
        obs1 = uct.observations[-2]
        obs2 = uct.observations[-1]
        
        dt = (obs2.timestamp - obs1.timestamp).total_seconds()
        if dt < 0.01:
            return (None, None)
        
        # Angular rates
        ra_rate = (obs2.ra - obs1.ra) / dt
        dec_rate = (obs2.dec - obs1.dec) / dt
        
        # Handle RA wraparound
        if abs(ra_rate) > math.pi / dt:
            if ra_rate > 0:
                ra_rate = ra_rate - 2 * math.pi / dt
            else:
                ra_rate = ra_rate + 2 * math.pi / dt
        
        # Extrapolate
        dt_predict = (target_time - obs2.timestamp).total_seconds()
        predicted_ra = obs2.ra + ra_rate * dt_predict
        predicted_dec = obs2.dec + dec_rate * dt_predict
        
        # Normalize RA to [0, 2π)
        predicted_ra = predicted_ra % (2 * math.pi)
        
        # Clamp Dec to [-π/2, π/2]
        predicted_dec = max(-math.pi/2, min(math.pi/2, predicted_dec))
        
        return (predicted_ra, predicted_dec)
    
    def get_iod_ready(self) -> list[UncorrelatedTrack]:
        """Get UCTs ready for IOD attempt."""
        return [
            uct for uct in self.ucts.values()
            if uct.is_ready_for_iod(self.config)
        ]
    
    def get_all_ucts(self) -> list[UncorrelatedTrack]:
        """Get all UCT buffers."""
        return list(self.ucts.values())
    
    def remove_uct(self, uct_id: UUID):
        """Remove a UCT buffer (e.g., after successful IOD)."""
        if uct_id in self.ucts:
            del self.ucts[uct_id]
    
    def prune_stale(self, max_age_sec: float = 300.0):
        """Remove UCTs that haven't received observations recently."""
        now = datetime.now()
        stale_ids = []
        
        for uct_id, uct in self.ucts.items():
            if uct.last_observation:
                age = (now - uct.last_observation.timestamp).total_seconds()
                if age > max_age_sec:
                    stale_ids.append(uct_id)
        
        for uct_id in stale_ids:
            del self.ucts[uct_id]
        
        return len(stale_ids)
    
    def get_stats(self) -> dict:
        """Get correlation statistics."""
        return {
            **self.stats,
            "active_ucts": len(self.ucts),
            "iod_ready": len(self.get_iod_ready()),
        }


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    from datetime import timezone
    import random
    
    print("=== Detection Correlation Test ===\n")
    
    config = CorrelationConfig(
        angular_gate_deg=5.0,
        temporal_window_sec=60.0,
        min_obs_for_iod=3,
        min_arc_length_deg=0.5,
    )
    
    engine = CorrelationEngine(config)
    
    # Simulate observations of a single object moving across the sky
    print("Simulating single object with 5 observations...")
    base_ra = math.radians(166.0)
    base_dec = math.radians(16.0)
    ra_rate = math.radians(0.5)  # 0.5 deg/sec
    dec_rate = math.radians(-0.1)  # -0.1 deg/sec
    base_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    
    for i in range(5):
        dt = i * 1.0  # 1 second intervals
        obs = CorrelatedObservation(
            obs_id=uuid4(),
            detection_id=uuid4(),
            sensor_id="AVERA-SAT-01-SWIR",
            timestamp=base_time + timedelta(seconds=dt),
            ra=base_ra + ra_rate * dt,
            dec=base_dec + dec_rate * dt,
            ra_sigma=math.radians(0.01),
            dec_sigma=math.radians(0.01),
            observer_position_eci=np.array([6778000, 0, 0]),
            observer_velocity_eci=np.array([0, 7668, 0]),
            confidence=0.9,
            object_class="Debris",
        )
        
        uct, is_new = engine.correlate(obs)
        status = "NEW UCT" if is_new else "correlated"
        print(f"  Obs {i+1}: RA={math.degrees(obs.ra):.3f}°, Dec={math.degrees(obs.dec):.3f}° → {status}")
    
    print(f"\nUCTs after single object: {len(engine.ucts)}")
    
    # Check if ready for IOD
    ready = engine.get_iod_ready()
    print(f"IOD-ready UCTs: {len(ready)}")
    
    if ready:
        uct = ready[0]
        print(f"\nUCT details:")
        print(f"  Observations: {uct.observation_count}")
        print(f"  Arc length: {uct.arc_length_deg:.3f}°")
        print(f"  Time span: {uct.time_span_sec:.1f} sec")
        print(f"  Mean rate: {uct.mean_angular_rate_deg_sec:.3f} °/sec")
    
    # Add a second object (should create new UCT)
    print("\n\nSimulating second object (different location)...")
    base_ra2 = math.radians(200.0)  # Very different RA
    base_dec2 = math.radians(-30.0)
    
    for i in range(3):
        dt = i * 1.0
        obs = CorrelatedObservation(
            obs_id=uuid4(),
            detection_id=uuid4(),
            sensor_id="AVERA-SAT-01-SWIR",
            timestamp=base_time + timedelta(seconds=dt),
            ra=base_ra2 + ra_rate * dt,
            dec=base_dec2 + dec_rate * dt,
            ra_sigma=math.radians(0.01),
            dec_sigma=math.radians(0.01),
            observer_position_eci=np.array([6778000, 0, 0]),
            observer_velocity_eci=np.array([0, 7668, 0]),
            confidence=0.85,
            object_class="RocketBody",
        )
        
        uct, is_new = engine.correlate(obs)
        status = "NEW UCT" if is_new else "correlated"
        print(f"  Obs {i+1}: RA={math.degrees(obs.ra):.3f}°, Dec={math.degrees(obs.dec):.3f}° → {status}")
    
    print(f"\nFinal UCTs: {len(engine.ucts)}")
    print(f"IOD-ready: {len(engine.get_iod_ready())}")
    
    print("\nCorrelation stats:")
    stats = engine.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n=== Test Complete ===")
