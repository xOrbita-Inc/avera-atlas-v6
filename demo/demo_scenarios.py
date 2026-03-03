"""
AVERA-ATLAS Demo Scenario Generator

Generates realistic conjunction scenarios for testing and demonstration.
Creates synthetic debris objects with actual close approaches to the asset.

Usage:
    python demo_scenarios.py [scenario_name]
    
Scenarios:
    - nominal: All objects at safe distances (NOMINAL/GREEN)
    - warning: Mix of AMBER and GREEN alerts  
    - critical: RED alert with imminent collision risk
    - mixed: Realistic mix of all risk levels
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any

# Output configuration
# Priority: 1) DATA_DIR env var, 2) /data/planner_artifacts (Docker), 3) local data/ folder
_env_dir = os.getenv("DATA_DIR")
if _env_dir and os.path.exists(os.path.dirname(_env_dir)):
    DATA_DIR = _env_dir
elif os.path.exists("/data"):
    DATA_DIR = "/data/planner_artifacts"
else:
    # Local development - write to data/ subfolder relative to this script
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
ARTIFACT_NAME = "states_multi.npz"

# Orbital mechanics constants
MU_EARTH = 398600.4418  # km³/s² - Earth gravitational parameter
R_EARTH = 6371.0  # km - Earth radius


def orbital_velocity(altitude_km: float) -> float:
    """Calculate circular orbital velocity at given altitude."""
    r = R_EARTH + altitude_km
    return np.sqrt(MU_EARTH / r)


def orbital_period(altitude_km: float) -> float:
    """Calculate orbital period in seconds."""
    r = R_EARTH + altitude_km
    return 2 * np.pi * np.sqrt(r**3 / MU_EARTH)


def generate_eci_state(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    true_anomaly_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ECI position and velocity from orbital elements.
    Simplified for circular orbits.
    """
    r = R_EARTH + altitude_km
    v = orbital_velocity(altitude_km)
    
    # Convert angles to radians
    inc = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    ta = np.radians(true_anomaly_deg)
    
    # Position in orbital plane
    r_orbital = r * np.array([np.cos(ta), np.sin(ta), 0])
    
    # Velocity in orbital plane (circular orbit)
    v_orbital = v * np.array([-np.sin(ta), np.cos(ta), 0])
    
    # Rotation matrices
    R_raan = np.array([
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan), np.cos(raan), 0],
        [0, 0, 1]
    ])
    
    R_inc = np.array([
        [1, 0, 0],
        [0, np.cos(inc), -np.sin(inc)],
        [0, np.sin(inc), np.cos(inc)]
    ])
    
    # Transform to ECI
    R = R_raan @ R_inc
    r_eci = R @ r_orbital
    v_eci = R @ v_orbital
    
    return r_eci, v_eci


def create_debris_near_asset(
    asset_r: np.ndarray,
    asset_v: np.ndarray,
    miss_distance_km: float,
    time_to_tca_steps: int,
    dt_sec: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create debris using linear relative motion model.
    
    For close-proximity conjunctions, the relative motion between objects
    in similar orbits can be approximated as linear over short time spans.
    This ensures predictable close approaches at specified times.
    """
    # Time to TCA
    tca_time_sec = time_to_tca_steps * dt_sec
    
    # Unit vectors in the local frame
    r_hat = asset_r / np.linalg.norm(asset_r)
    v_hat = asset_v / np.linalg.norm(asset_v)
    h = np.cross(asset_r, asset_v)
    h_hat = h / np.linalg.norm(h)
    
    # Random direction for miss distance (perpendicular to approach direction)
    theta = np.random.uniform(0, 2 * np.pi)
    miss_direction = np.cos(theta) * r_hat + np.sin(theta) * h_hat
    miss_direction = miss_direction / np.linalg.norm(miss_direction)
    
    # Relative velocity (debris relative to asset)
    # Typical conjunction: 0.05 - 0.3 km/s relative velocity
    rel_vel_mag = np.random.uniform(0.05, 0.2)  # km/s
    
    # Relative velocity direction (perpendicular to miss direction at TCA)
    # This ensures minimum distance is achieved at TCA
    rel_vel_direction = np.cross(miss_direction, h_hat)
    if np.linalg.norm(rel_vel_direction) < 0.01:
        rel_vel_direction = np.cross(miss_direction, v_hat)
    rel_vel_direction = rel_vel_direction / np.linalg.norm(rel_vel_direction)
    
    # Relative velocity vector (approaching before TCA, receding after)
    v_rel = rel_vel_direction * rel_vel_mag
    
    # Position at TCA: debris is at asset position + miss_direction * miss_distance
    r_rel_at_tca = miss_direction * miss_distance_km
    
    # Position at t=0: work backwards from TCA
    # r_rel(t) = r_rel_tca + v_rel * (t - tca_time)
    # At t=0: r_rel_0 = r_rel_tca - v_rel * tca_time
    r_rel_0 = r_rel_at_tca - v_rel * tca_time_sec
    
    # Debris absolute state (approximately)
    # For demo purposes, debris has same base velocity as asset
    # plus the relative velocity
    debris_r = asset_r + r_rel_0
    debris_v = asset_v + v_rel
    
    return debris_r, debris_v


def generate_scenario(scenario_name: str = "mixed") -> Dict[str, Any]:
    """
    Generate a complete conjunction scenario.
    
    Returns dict with object states and expected outcomes.
    """
    
    # Asset orbital parameters (ISS-like)
    asset_altitude = 420  # km
    asset_inclination = 51.6  # degrees
    asset_raan = 45.0  # degrees
    asset_true_anomaly = 0.0  # degrees
    
    asset_r, asset_v = generate_eci_state(
        asset_altitude, asset_inclination, asset_raan, asset_true_anomaly
    )
    
    # Time parameters
    dt_sec = 60.0  # 1 minute steps
    n_steps = 1440  # 24 hours
    
    # Scenario definitions
    scenarios = {
        "nominal": [
            # All safe distances
            {"name": "SAT_001", "miss_km": 50.0, "tca_step": 200, "type": "satellite"},
            {"name": "DEB_001", "miss_km": 75.0, "tca_step": 400, "type": "debris"},
            {"name": "SAT_002", "miss_km": 100.0, "tca_step": 600, "type": "satellite"},
        ],
        "warning": [
            # AMBER alerts
            {"name": "DEB_001", "miss_km": 0.5, "tca_step": 150, "type": "debris"},
            {"name": "DEB_002", "miss_km": 0.8, "tca_step": 300, "type": "debris"},
            {"name": "SAT_001", "miss_km": 25.0, "tca_step": 500, "type": "satellite"},
            {"name": "DEB_003", "miss_km": 1.2, "tca_step": 700, "type": "debris"},
        ],
        "critical": [
            # RED alert - imminent collision
            {"name": "DEB_CRIT", "miss_km": 0.05, "tca_step": 30, "type": "debris"},  # 50 meters!
            {"name": "DEB_002", "miss_km": 0.3, "tca_step": 120, "type": "debris"},
            {"name": "SAT_001", "miss_km": 2.0, "tca_step": 400, "type": "satellite"},
        ],
        "mixed": [
            # Realistic operational scenario
            {"name": "Cosmos_DEB", "miss_km": 0.08, "tca_step": 45, "type": "debris"},   # RED - 80m
            {"name": "Fengyun_DEB", "miss_km": 0.4, "tca_step": 180, "type": "debris"},  # AMBER
            {"name": "CubeSat_012", "miss_km": 1.5, "tca_step": 350, "type": "satellite"},  # GREEN
            {"name": "Starlink_42", "miss_km": 15.0, "tca_step": 500, "type": "satellite"},  # NOMINAL
            {"name": "Iridium_DEB", "miss_km": 0.6, "tca_step": 720, "type": "debris"},  # AMBER
            {"name": "Unknown_001", "miss_km": 45.0, "tca_step": 900, "type": "unknown"},  # NOMINAL
        ],
    }
    
    if scenario_name not in scenarios:
        print(f"Unknown scenario: {scenario_name}. Using 'mixed'.")
        scenario_name = "mixed"
    
    objects = scenarios[scenario_name]
    
    # Generate debris states
    object_ids = []
    r_eci_km = []
    v_eci_km_s = []
    confidences = []
    expected_outcomes = []
    
    for obj in objects:
        debris_r, debris_v = create_debris_near_asset(
            asset_r, asset_v,
            miss_distance_km=obj["miss_km"],
            time_to_tca_steps=obj["tca_step"],
            dt_sec=dt_sec
        )
        
        object_ids.append(obj["name"])
        r_eci_km.append(debris_r)
        v_eci_km_s.append(debris_v)
        confidences.append(np.random.uniform(0.7, 0.98))
        
        # Predict risk level
        if obj["miss_km"] < 0.1:
            risk = "RED"
        elif obj["miss_km"] < 1.0:
            risk = "AMBER"
        elif obj["miss_km"] < 10.0:
            risk = "GREEN"
        else:
            risk = "NOMINAL"
        
        expected_outcomes.append({
            "object": obj["name"],
            "miss_km": obj["miss_km"],
            "tca_minutes": obj["tca_step"],
            "expected_risk": risk
        })
    
    return {
        "scenario": scenario_name,
        "asset": {"r_eci_km": asset_r.tolist(), "v_eci_km_s": asset_v.tolist()},
        "object_ids": object_ids,
        "r_eci_km": np.array(r_eci_km),
        "v_eci_km_s": np.array(v_eci_km_s),
        "confidences": np.array(confidences),
        "t_window": np.array([dt_sec, n_steps]),
        "expected_outcomes": expected_outcomes
    }


def write_scenario(scenario: Dict[str, Any], output_dir: str = None):
    """Write scenario to NPZ file for processing by propagator."""
    if output_dir is None:
        output_dir = DATA_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, ARTIFACT_NAME)
    
    t0_utc = datetime.utcnow().isoformat()
    
    np.savez(
        out_path,
        object_ids=np.array(scenario["object_ids"]),
        r_eci_km=scenario["r_eci_km"],
        v_eci_km_s=scenario["v_eci_km_s"],
        confidences=scenario["confidences"],
        t_window=scenario["t_window"],
        metadata=json.dumps({
            "source": "demo_generator",
            "scenario": scenario["scenario"],
            "t0": t0_utc,
            "asset_state": scenario["asset"]
        })
    )
    
    print(f"\n{'='*60}")
    print(f"DEMO SCENARIO: {scenario['scenario'].upper()}")
    print(f"{'='*60}")
    print(f"Written to: {out_path}")
    print(f"\nExpected Outcomes:")
    print(f"{'-'*60}")
    
    for outcome in scenario["expected_outcomes"]:
        risk = outcome["expected_risk"]
        emoji = {"RED": "🔴", "AMBER": "🟡", "GREEN": "🟢", "NOMINAL": "⚪"}.get(risk, "⚪")
        print(f"  {emoji} {outcome['object']:15} | Miss: {outcome['miss_km']:8.3f} km | "
              f"TCA: T+{outcome['tca_minutes']:4}min | Risk: {risk}")
    
    print(f"{'='*60}\n")
    
    return out_path


def main():
    import sys
    
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "mixed"
    
    print(f"Generating scenario: {scenario_name}")
    scenario = generate_scenario(scenario_name)
    write_scenario(scenario)
    
    print("Scenario ready. Run propagator to process.")


if __name__ == "__main__":
    main()
