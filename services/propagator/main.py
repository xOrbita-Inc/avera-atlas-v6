"""
AVERA-ATLAS Orbital Propagator with Keplerian Mechanics

Proper orbital propagation for debris objects using Keplerian two-body dynamics.
Replaces linear motion with realistic orbital trajectories.

Features:
- Keplerian orbit propagation for debris
- SGP4 propagation for the asset (from TLE)
- Proper conjunction geometry
- NASA-standard Pc calculation
- GO/NO GO decision support
"""

import time
import os
import json
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any

from sgp4.api import Satrec, WGS72

# Import Pc utilities
from pc_utils import compute_pc, default_covariance_from_uncertainty

# === CONSTANTS ===
MU_EARTH = 398600.4418  # km³/s² - Earth gravitational parameter
R_EARTH = 6371.0  # km

# === CONFIGURATION ===
DATA_DIR = os.getenv("DATA_DIR", "/data/planner_artifacts")
INPUT_FILE = "states_multi.npz"
OUTPUT_FILE = "prop_multi.npz"

HBR_M = 15.0  # Combined hard body radius (meters)
SCREENING_THRESHOLD_KM = 100.0
PC_RED_THRESHOLD = 1e-4
PC_AMBER_THRESHOLD = 1e-5
PC_GREEN_THRESHOLD = 1e-7
DEFAULT_DEBRIS_UNCERTAINTY_M = 2000.0  # 2 km uncertainty

# Asset TLE (ISS)
MY_SAT_TLE_LINE1 = "1 25544U 98067A   23321.56445781  .00018593  00000-0  34139-3 0  9995"
MY_SAT_TLE_LINE2 = "2 25544  51.6416 288.7738 0005519 253.3323 214.2882 15.50066497425769"


# =============================================================================
# Keplerian Propagation
# =============================================================================

def kepler_propagate(r0: np.ndarray, v0: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate a state vector using Keplerian two-body dynamics.
    
    Uses universal variable formulation for robustness.
    
    Parameters
    ----------
    r0 : np.ndarray
        Initial position [km]
    v0 : np.ndarray
        Initial velocity [km/s]
    dt : float
        Time step [seconds]
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (position, velocity) at time t0 + dt
    """
    # Handle dt=0 case
    if abs(dt) < 1e-10:
        return r0.copy(), v0.copy()
    
    mu = MU_EARTH
    
    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    
    # Check for degenerate cases
    if r0_mag < 100:  # Inside Earth or too close
        return r0.copy(), v0.copy()
    
    # Specific energy
    energy = v0_mag**2 / 2 - mu / r0_mag
    
    # Semi-major axis
    if abs(energy) > 1e-10:
        a = -mu / (2 * energy)
    else:
        # Parabolic - use large value
        a = 1e10
    
    # Check if hyperbolic escape (a < 0 means hyperbolic)
    # For hyperbolic orbits, use simplified linear propagation
    if a < 0 or abs(a) > 1e8:
        # Object is escaping - use linear approximation
        r_new = r0 + v0 * dt
        return r_new, v0.copy()
    
    # Initial radial velocity
    vr0 = np.dot(r0, v0) / r0_mag
    
    # Universal variable initial guess
    alpha = 1 / a
    
    if alpha > 1e-10:  # Elliptical
        chi = np.sqrt(mu) * dt * alpha
    else:  # Near-parabolic
        chi = np.sqrt(mu) * dt / r0_mag
    
    # Newton-Raphson iteration for universal anomaly
    ratio = 1
    max_iter = 50
    tol = 1e-10
    
    for iteration in range(max_iter):
        chi2 = chi * chi
        psi = chi2 * alpha
        
        # Stumpff functions
        if psi > 1e-6:
            sqrt_psi = np.sqrt(psi)
            c2 = (1 - np.cos(sqrt_psi)) / psi
            c3 = (sqrt_psi - np.sin(sqrt_psi)) / (sqrt_psi * psi)
        elif psi < -1e-6:
            sqrt_neg_psi = np.sqrt(-psi)
            c2 = (1 - np.cosh(sqrt_neg_psi)) / psi
            c3 = (np.sinh(sqrt_neg_psi) - sqrt_neg_psi) / (-psi * sqrt_neg_psi)
        else:
            c2 = 1/2
            c3 = 1/6
        
        r = chi2 * c2 + vr0 / np.sqrt(mu) * chi * (1 - psi * c3) + r0_mag * (1 - psi * c2)
        
        # Protect against division by zero
        if abs(r) < 1e-10:
            break
        
        # Time equation
        t_chi = chi**3 * c3 + vr0 / np.sqrt(mu) * chi2 * c2 + r0_mag * chi * (1 - psi * c3)
        t_chi = t_chi / np.sqrt(mu)
        
        ratio = (dt - t_chi) / r
        chi = chi + ratio
        
        if abs(ratio) < tol:
            break
    
    # If iteration didn't converge, fall back to linear
    if iteration >= max_iter - 1 or not np.isfinite(chi):
        r_new = r0 + v0 * dt
        return r_new, v0.copy()
    
    # Compute f and g functions
    chi2 = chi * chi
    psi = chi2 * alpha
    
    if psi > 1e-6:
        sqrt_psi = np.sqrt(psi)
        c2 = (1 - np.cos(sqrt_psi)) / psi
        c3 = (sqrt_psi - np.sin(sqrt_psi)) / (sqrt_psi * psi)
    elif psi < -1e-6:
        sqrt_neg_psi = np.sqrt(-psi)
        c2 = (1 - np.cosh(sqrt_neg_psi)) / psi
        c3 = (np.sinh(sqrt_neg_psi) - sqrt_neg_psi) / (-psi * sqrt_neg_psi)
    else:
        c2 = 1/2
        c3 = 1/6
    
    f = 1 - chi2 / r0_mag * c2
    g = dt - chi**3 / np.sqrt(mu) * c3
    
    r_new = f * r0 + g * v0
    r_new_mag = np.linalg.norm(r_new)
    
    # Check for valid result
    if r_new_mag < 100 or not np.all(np.isfinite(r_new)):
        r_new = r0 + v0 * dt
        return r_new, v0.copy()
    
    fdot = np.sqrt(mu) / (r_new_mag * r0_mag) * chi * (psi * c3 - 1)
    gdot = 1 - chi2 / r_new_mag * c2
    
    v_new = fdot * r0 + gdot * v0
    
    # Final sanity check
    if not np.all(np.isfinite(v_new)):
        return r_new, v0.copy()
    
    return r_new, v_new


def propagate_trajectory(r0: np.ndarray, v0: np.ndarray, times_sec: np.ndarray, use_linear: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate a trajectory over an array of times.
    
    Parameters
    ----------
    r0 : np.ndarray
        Initial position [km]
    v0 : np.ndarray
        Initial velocity [km/s]
    times_sec : np.ndarray
        Array of times from epoch [seconds]
    use_linear : bool
        If True, use linear propagation (r = r0 + v*t). Better for relative motion.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (positions, velocities) arrays of shape (n_times, 3)
    """
    n = len(times_sec)
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    
    if use_linear:
        # Linear propagation - preserves relative motion geometry
        for i, t in enumerate(times_sec):
            positions[i] = r0 + v0 * t
            velocities[i] = v0
    else:
        # Keplerian propagation - full orbital mechanics
        for i, t in enumerate(times_sec):
            positions[i], velocities[i] = kepler_propagate(r0, v0, t)
    
    return positions, velocities


# =============================================================================
# Risk Assessment
# =============================================================================

def pc_to_risk_level(pc: float) -> str:
    if pc >= PC_RED_THRESHOLD:
        return "RED"
    elif pc >= PC_AMBER_THRESHOLD:
        return "AMBER"
    elif pc >= PC_GREEN_THRESHOLD:
        return "GREEN"
    return "NOMINAL"


def evaluate_decision(pc: float, time_to_tca_min: float) -> Dict[str, Any]:
    """GO/NO GO decision based on Pc and time."""
    if time_to_tca_min < 30:
        if pc >= 1e-3:
            return {"decision": "GO", "urgency": "CRITICAL"}
        elif pc >= 1e-4:
            return {"decision": "STANDBY", "urgency": "HIGH"}
        return {"decision": "NO_GO", "urgency": "MONITOR"}
    elif time_to_tca_min < 120:
        if pc >= 1e-4:
            return {"decision": "GO", "urgency": "HIGH"}
        elif pc >= 1e-5:
            return {"decision": "STANDBY", "urgency": "ELEVATED"}
        return {"decision": "NO_GO", "urgency": "LOW"}
    else:
        if pc >= 1e-5:
            return {"decision": "GO", "urgency": "MODERATE"}
        elif pc >= 1e-6:
            return {"decision": "STANDBY", "urgency": "LOW"}
        return {"decision": "NO_GO", "urgency": "NOMINAL"}


# =============================================================================
# Main Processing Loop
# =============================================================================

def propagate_and_screen():
    """Main propagation and conjunction screening."""
    input_path = os.path.join(DATA_DIR, INPUT_FILE)
    
    if not os.path.exists(input_path):
        return
    
    print(f"[PROP] Found {INPUT_FILE}. Loading...")
    
    try:
        data = np.load(input_path, allow_pickle=True)
        obj_ids = data['object_ids']
        r_eci_init = data['r_eci_km']
        v_eci_init = data['v_eci_km_s']
        t_window = data['t_window']
        metadata = json.loads(str(data['metadata']))
        confidences = data['confidences'] if 'confidences' in data else np.full(len(obj_ids), 0.8)
    except Exception as e:
        print(f"[ERROR] Corrupt artifact: {e}")
        os.rename(input_path, input_path + ".err")
        return
    
    dt_sec = float(t_window[0])
    n_steps = int(t_window[1])
    
    # Check if this is a demo scenario with asset state
    is_demo = "asset_state" in metadata
    if is_demo:
        # Use provided asset state
        asset_r0 = np.array(metadata["asset_state"]["r_eci_km"])
        asset_v0 = np.array(metadata["asset_state"]["v_eci_km_s"])
        print(f"[PROP] Using demo scenario asset state")
        use_sgp4 = False
    else:
        # Use SGP4 for asset
        use_sgp4 = True
    
    # Time array
    times_sec = np.arange(n_steps) * dt_sec
    jd_start = 2460265.5
    times_jd = jd_start + times_sec / 86400.0
    
    # Propagate Asset
    if use_sgp4:
        my_sat = Satrec.twoline2rv(MY_SAT_TLE_LINE1, MY_SAT_TLE_LINE2, WGS72)
        fr_array = np.zeros(n_steps)
        e, r_asset, v_asset = my_sat.sgp4_array(times_jd, fr_array)
        r_asset = np.array(r_asset) if isinstance(r_asset, tuple) else r_asset
        v_asset = np.array(v_asset) if isinstance(v_asset, tuple) else v_asset
    else:
        # For demo scenarios, use linear propagation to preserve relative motion geometry
        # This matches the debris propagation model
        r_asset, v_asset = propagate_trajectory(asset_r0, asset_v0, times_sec, use_linear=is_demo)
    
    # Propagate Debris
    # For demo scenarios, use linear propagation to preserve relative motion geometry
    # For real data, use Keplerian propagation
    n_objs = len(obj_ids)
    r_debris = np.zeros((n_objs, n_steps, 3))
    v_debris = np.zeros((n_objs, n_steps, 3))
    
    prop_method = "LINEAR (relative motion)" if is_demo else "Keplerian"
    print(f"[PROP] Propagating {n_objs} objects with {prop_method} dynamics...")
    
    for i in range(n_objs):
        r_debris[i], v_debris[i] = propagate_trajectory(
            r_eci_init[i], v_eci_init[i], times_sec, use_linear=is_demo
        )
    
    # Conjunction Screening
    print(f"[PROP] Running conjunction assessment...")
    
    cov_asset = default_covariance_from_uncertainty(1000.0, cross_track_factor=0.3)
    
    results = {
        'min_miss_distances': [], 'pc_values': [], 'risk_levels': [],
        'tca_indices': [], 'relative_velocities': [], 'decisions': [],
        'decision_urgencies': [], 'propulsion_options': [], 'delta_v_estimates': []
    }
    
    red_alerts, amber_alerts, go_decisions, standby_decisions = [], [], [], []
    
    for i in range(n_objs):
        # Find TCA
        diff = r_asset - r_debris[i]
        dists = np.linalg.norm(diff, axis=1)
        
        tca_idx = np.argmin(dists)
        min_dist_km = dists[tca_idx]
        time_to_tca_s = tca_idx * dt_sec
        time_to_tca_min = time_to_tca_s / 60.0
        
        # Relative velocity at TCA
        v_rel = v_asset[tca_idx] - v_debris[i, tca_idx]
        rel_vel = np.linalg.norm(v_rel)
        
        results['min_miss_distances'].append(min_dist_km)
        results['tca_indices'].append(tca_idx)
        results['relative_velocities'].append(rel_vel)
        
        # Pc calculation
        if min_dist_km > SCREENING_THRESHOLD_KM:
            pc = 0.0
        else:
            conf = float(confidences[i]) if i < len(confidences) else 0.8
            uncertainty_m = DEFAULT_DEBRIS_UNCERTAINTY_M / max(conf, 0.1)
            cov_debris = default_covariance_from_uncertainty(uncertainty_m, cross_track_factor=0.5)
            
            try:
                result = compute_pc(
                    r_asset[tca_idx] * 1000, v_asset[tca_idx] * 1000, cov_asset,
                    r_debris[i, tca_idx] * 1000, v_debris[i, tca_idx] * 1000, cov_debris,
                    HBR_M
                )
                pc = result.Pc
            except Exception as e:
                print(f"[WARN] Pc calc failed for {obj_ids[i]}: {e}")
                pc = 0.0
        
        risk = pc_to_risk_level(pc)
        dec = evaluate_decision(pc, time_to_tca_min)
        
        # Delta-V estimate for close approaches
        if min_dist_km < 1.0 and time_to_tca_s > 0:
            delta_v = (1.0 - min_dist_km) / time_to_tca_s * 1000 * 1.5
        else:
            delta_v = 0.0
        
        # Propulsion option
        if dec['decision'] == 'NO_GO':
            prop_option = 'N/A'
        elif time_to_tca_min < 30:
            prop_option = 'A'
        elif time_to_tca_min < 120:
            prop_option = 'A/B'
        else:
            prop_option = 'B'
        
        results['pc_values'].append(pc)
        results['risk_levels'].append(risk)
        results['decisions'].append(dec['decision'])
        results['decision_urgencies'].append(dec['urgency'])
        results['propulsion_options'].append(prop_option)
        results['delta_v_estimates'].append(delta_v)
        
        # Track alerts
        if dec['decision'] == 'GO':
            go_decisions.append(i)
        elif dec['decision'] == 'STANDBY':
            standby_decisions.append(i)
        
        if risk == "RED":
            red_alerts.append(i)
            emoji = "🔴"
        elif risk == "AMBER":
            amber_alerts.append(i)
            emoji = "🟡"
        elif risk == "GREEN":
            emoji = "🟢"
        else:
            emoji = "⚪"
        
        print(f"[PROP] {emoji} {obj_ids[i]:15} | Miss: {min_dist_km*1000:10.1f}m | "
              f"Pc: {pc:.2e} | TCA: T+{time_to_tca_min:5.1f}min | {dec['decision']}")
    
    # Summary
    print(f"\n[PROP] {'='*55}")
    print(f"[PROP] CONJUNCTION ASSESSMENT COMPLETE")
    print(f"[PROP] {'-'*55}")
    print(f"[PROP] Objects: {n_objs} | 🔴 RED: {len(red_alerts)} | 🟡 AMBER: {len(amber_alerts)}")
    print(f"[PROP] Decisions: ✅ GO: {len(go_decisions)} | ⏳ STANDBY: {len(standby_decisions)}")
    print(f"[PROP] {'='*55}\n")
    
    # Write output
    out_path = os.path.join(DATA_DIR, OUTPUT_FILE)
    np.savez(
        out_path,
        t_array=times_jd, r_asset=r_asset, v_asset=v_asset,
        r_objects=r_debris, v_objects=v_debris, obj_ids=obj_ids,
        ca_table=np.array(results['min_miss_distances']),
        pc_values=np.array(results['pc_values']),
        risk_levels=np.array(results['risk_levels']),
        tca_indices=np.array(results['tca_indices']),
        relative_velocities=np.array(results['relative_velocities']),
        decisions=np.array(results['decisions']),
        decision_urgencies=np.array(results['decision_urgencies']),
        propulsion_options=np.array(results['propulsion_options']),
        delta_v_estimates=np.array(results['delta_v_estimates']),
        screening_params=json.dumps({
            'hbr_m': HBR_M, 'screening_threshold_km': SCREENING_THRESHOLD_KM,
            'pc_red_threshold': PC_RED_THRESHOLD, 'pc_amber_threshold': PC_AMBER_THRESHOLD,
            'dt_sec': dt_sec
        }),
        n_red_alerts=len(red_alerts), n_amber_alerts=len(amber_alerts),
        n_go_decisions=len(go_decisions), n_standby_decisions=len(standby_decisions)
    )
    
    print(f"[PROP] ✅ Results saved to {OUTPUT_FILE}")
    os.rename(input_path, input_path + ".processed")


if __name__ == "__main__":
    print(f"[PROP] AVERA-ATLAS Keplerian Propagator")
    print(f"[PROP] Watching {DATA_DIR}...")
    while True:
        propagate_and_screen()
        time.sleep(1)
