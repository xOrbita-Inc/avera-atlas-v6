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
from aps_math.pc_utils import compute_pc, default_covariance_from_uncertainty

# SCRUM-389 / ADR-010: values the planner and the propagator must agree on live
# in one place, so the two services cannot compute a different Pc for the same
# conjunction. Each constant carries its reasoning at the definition.
from aps_math import conventions

# === CONSTANTS ===
MU_EARTH = 398600.4418  # km³/s² - Earth gravitational parameter
R_EARTH = 6371.0  # km

# === CONFIGURATION ===
DATA_DIR = os.getenv("DATA_DIR", "/data/planner_artifacts")
INPUT_FILE = "states_multi.npz"
OUTPUT_FILE = "prop_multi.npz"

# Shared conventions (ADR-010). Re-exported under the propagator's historic
# names so existing call sites and tests keep working, but there is now exactly
# one definition and the planner imports the same one.
HBR_M = conventions.DEFAULT_COMBINED_HBR_M
SCREENING_THRESHOLD_KM = conventions.SCREENING_THRESHOLD_KM
DEFAULT_DEBRIS_UNCERTAINTY_M = conventions.DEFAULT_DEBRIS_UNCERTAINTY_M

# Risk banding is the propagator's own presentation concern, not a shared
# convention. The planner gates on the operator policy's Pc thresholds instead.
PC_RED_THRESHOLD = 1e-4
PC_AMBER_THRESHOLD = 1e-5
PC_GREEN_THRESHOLD = 1e-7

# Asset TLE (ISS)
MY_SAT_TLE_LINE1 = "1 25544U 98067A   23321.56445781  .00018593  00000-0  34139-3 0  9995"
MY_SAT_TLE_LINE2 = "2 25544  51.6416 288.7738 0005519 253.3323 214.2882 15.50066497425769"


# =============================================================================
# Keplerian Propagation
# =============================================================================

def kepler_propagate(r0: np.ndarray, v0: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate a bound elliptic state with universal-variable two-body dynamics.

    Hyperbolic and near-parabolic trajectories are deliberately unsupported
    here rather than silently replaced with straight-line motion.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    if r0.shape != (3,) or v0.shape != (3,):
        raise ValueError("r0 and v0 must each be 3-element vectors")
    if not np.all(np.isfinite(r0)) or not np.all(np.isfinite(v0)):
        raise ValueError("r0 and v0 must contain only finite values")
    if not np.isfinite(dt):
        raise ValueError("dt must be finite")

    if abs(dt) < 1e-10:
        return r0.copy(), v0.copy()

    mu = MU_EARTH
    sqrt_mu = np.sqrt(mu)
    r0_mag = float(np.linalg.norm(r0))
    if r0_mag < 100.0:
        raise ValueError("initial position magnitude is too small for Earth-orbit propagation")

    v0_sq = float(np.dot(v0, v0))
    alpha = 2.0 / r0_mag - v0_sq / mu  # reciprocal semi-major axis [1/km]

    # Preserve the old near-parabolic boundary (|a| > 1e8 km), but fail
    # explicitly instead of substituting a linear trajectory.
    if abs(alpha) < 1e-8:
        raise ValueError("near-parabolic trajectories are unsupported by kepler_propagate")
    if alpha < 0.0:
        raise ValueError("hyperbolic trajectories are unsupported by kepler_propagate")

    vr0 = float(np.dot(r0, v0) / r0_mag)
    radial_coeff = r0_mag * vr0 / sqrt_mu

    def stumpff_c2_c3(z: float) -> Tuple[float, float]:
        if z > 1e-6:
            sqrt_z = np.sqrt(z)
            c2 = (1.0 - np.cos(sqrt_z)) / z
            c3 = (sqrt_z - np.sin(sqrt_z)) / (sqrt_z * z)
        elif z < -1e-6:
            sqrt_neg_z = np.sqrt(-z)
            c2 = (np.cosh(sqrt_neg_z) - 1.0) / (-z)
            c3 = (np.sinh(sqrt_neg_z) - sqrt_neg_z) / ((-z) ** 1.5)
        else:
            z2 = z * z
            c2 = 0.5 - z / 24.0 + z2 / 720.0
            c3 = 1.0 / 6.0 - z / 120.0 + z2 / 5040.0
        return float(c2), float(c3)

    # Elliptic universal-anomaly initial guess. The sign follows dt so
    # backward propagation is handled by the same equations.
    chi = sqrt_mu * dt * alpha
    converged = False
    max_iter = 50
    tol = 1e-10

    for _ in range(max_iter):
        z = alpha * chi * chi
        c2, c3 = stumpff_c2_c3(z)

        # Universal Kepler residual F(chi) = 0 and its derivative.
        # The r0_mag factor on the radial-velocity term is required for
        # general states away from an apsis.
        F = (
            radial_coeff * chi * chi * c2
            + (1.0 - alpha * r0_mag) * chi**3 * c3
            + r0_mag * chi
            - sqrt_mu * dt
        )
        dF = (
            radial_coeff * chi * (1.0 - z * c3)
            + (1.0 - alpha * r0_mag) * chi * chi * c2
            + r0_mag
        )

        if not np.isfinite(F) or not np.isfinite(dF) or abs(dF) < 1e-12:
            raise RuntimeError("universal-variable iteration produced an invalid Newton step")

        delta_chi = F / dF
        chi -= delta_chi
        if abs(delta_chi) < tol:
            converged = True
            break

    if not converged or not np.isfinite(chi):
        raise RuntimeError("universal-variable Kepler propagation did not converge")

    z = alpha * chi * chi
    c2, c3 = stumpff_c2_c3(z)
    chi2 = chi * chi

    f = 1.0 - chi2 / r0_mag * c2
    g = dt - chi**3 / sqrt_mu * c3
    r_new = f * r0 + g * v0
    r_new_mag = float(np.linalg.norm(r_new))

    if r_new_mag < 100.0 or not np.all(np.isfinite(r_new)):
        raise RuntimeError("universal-variable propagation produced an invalid position")

    fdot = sqrt_mu / (r_new_mag * r0_mag) * chi * (z * c3 - 1.0)
    gdot = 1.0 - chi2 / r_new_mag * c2
    v_new = fdot * r0 + gdot * v0

    if not np.all(np.isfinite(v_new)):
        raise RuntimeError("universal-variable propagation produced an invalid velocity")

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

def _validate_demo_asset_state(r0: np.ndarray, v0: np.ndarray) -> None:
    """Sanity-check a demo asset state before Keplerian propagation.

    SCRUM-370 places the demo asset on its real Keplerian orbit, so the provided
    (r0, v0) must describe a sane, bound orbit. Warn loudly if it does not, so a
    bad preset surfaces instead of silently producing a garbage asset track.
    """
    r_mag = float(np.linalg.norm(r0))
    v_mag = float(np.linalg.norm(v0))
    # Radius must be above the surface and within a generous Earth-orbit bound.
    if not (R_EARTH < r_mag < 100000.0):
        print(f"[WARN] demo asset |r0|={r_mag:.1f} km is not a sane orbital radius")
    # Bound (elliptical) orbit: specific orbital energy must be negative.
    energy = v_mag**2 / 2.0 - MU_EARTH / max(r_mag, 1e-6)
    if energy >= 0.0:
        print(f"[WARN] demo asset state is not bound (specific energy={energy:.3f} >= 0); "
              f"unbound state is unsupported by kepler_propagate and will raise")


def _debris_uncertainty_m(position_sigma_m: float, confidence: float):
    """SCRUM-391: pick the debris 1-sigma position uncertainty for one object.

    Returns (uncertainty_m, source).

    A supplied ``position_sigma_m`` is used as given and is deliberately NOT
    divided by confidence. Confidence is a stand-in for how well the object is
    known; a supplied sigma *is* how well it is known. Scaling one by the other
    would count the same uncertainty twice, and would mean a tracked object with
    a 250 m covariance and a 0.85 confidence silently got 294 m.

    Absent a supplied value, the previous behaviour is unchanged:
    ``DEFAULT_DEBRIS_UNCERTAINTY_M / max(confidence, 0.1)``, which represents an
    object known only from a TLE and can only ever inflate above 2 km.
    """
    if np.isfinite(position_sigma_m) and position_sigma_m > 0.0:
        return float(position_sigma_m), "supplied"
    return DEFAULT_DEBRIS_UNCERTAINTY_M / max(confidence, 0.1), "confidence_default"


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
        # SCRUM-391: optional per-object 1-sigma position uncertainty in metres.
        # A non-finite or non-positive entry means "not supplied for this
        # object", which falls back to the confidence-scaled default below. See
        # _debris_uncertainty_m.
        position_sigma_m = (
            np.asarray(data['position_sigma_m'], dtype=float)
            if 'position_sigma_m' in data
            else np.full(len(obj_ids), np.nan)
        )
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
        asset_r0 = np.array(metadata["asset_state"]["r_eci_km"], dtype=float)
        asset_v0 = np.array(metadata["asset_state"]["v_eci_km_s"], dtype=float)
        _validate_demo_asset_state(asset_r0, asset_v0)
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
        # SCRUM-370: the demo asset rides its real Keplerian orbit. It was
        # previously propagated linearly (r = r0 + v0*t), which flew the asset
        # thousands of km off-orbit over the conjunction window and corrupted the
        # absolute state later fed to the planner. Debris is carried relative to
        # this Keplerian track below, so relative geometry is unchanged.
        r_asset, v_asset = propagate_trajectory(asset_r0, asset_v0, times_sec, use_linear=False)

    # Propagate Debris
    # For real data, use Keplerian propagation. For demo scenarios (SCRUM-370),
    # carry each debris as the asset's Keplerian trajectory plus the constant
    # initial relative offset and relative velocity, so that
    #     r_debris_i(t) - r_asset(t) == rel0_i + vrel_i * t   (exactly)
    # identical to the previous linear-everything scheme. Because miss distance,
    # TCA index, and the 2D Pc depend only on relative position/velocity (plus the
    # fixed diagonal covariances), no preset's miss/TCA/risk/badge moves. Only the
    # absolute asset and debris tracks change from straight lines to on-orbit arcs.
    n_objs = len(obj_ids)
    r_debris = np.zeros((n_objs, n_steps, 3))
    v_debris = np.zeros((n_objs, n_steps, 3))

    prop_method = "Keplerian asset + linear relative offset" if is_demo else "Keplerian"
    print(f"[PROP] Propagating {n_objs} objects with {prop_method} dynamics...")

    for i in range(n_objs):
        if is_demo:
            rel0 = np.asarray(r_eci_init[i], dtype=float) - asset_r0
            vrel = np.asarray(v_eci_init[i], dtype=float) - asset_v0
            r_debris[i] = r_asset + rel0 + np.outer(times_sec, vrel)
            v_debris[i] = v_asset + vrel
        else:
            r_debris[i], v_debris[i] = propagate_trajectory(
                r_eci_init[i], v_eci_init[i], times_sec, use_linear=False
            )
    
    # Conjunction Screening
    print(f"[PROP] Running conjunction assessment...")
    
    cov_asset = default_covariance_from_uncertainty(
        conventions.DEFAULT_ASSET_UNCERTAINTY_M,
        cross_track_factor=conventions.DEFAULT_PRIMARY_CROSS_TRACK_FACTOR,
    )
    
    results = {
        'min_miss_distances': [], 'pc_values': [], 'risk_levels': [],
        'tca_indices': [], 'relative_velocities': [], 'decisions': [],
        'decision_urgencies': [], 'propulsion_options': [], 'delta_v_estimates': [],
        # SCRUM-391: which covariance each object's Pc was computed against, so
        # an operator can tell a supplied sigma from the TLE-grade default.
        'debris_sigma_m': [], 'covariance_sources': []
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
        conf = float(confidences[i]) if i < len(confidences) else 0.8
        sigma_in = float(position_sigma_m[i]) if i < len(position_sigma_m) else float('nan')
        uncertainty_m, uncertainty_source = _debris_uncertainty_m(sigma_in, conf)
        results['debris_sigma_m'].append(uncertainty_m)
        results['covariance_sources'].append(uncertainty_source)

        if min_dist_km > SCREENING_THRESHOLD_KM:
            pc = 0.0
        else:
            cov_debris = default_covariance_from_uncertainty(
                uncertainty_m,
                cross_track_factor=conventions.DEFAULT_SECONDARY_CROSS_TRACK_FACTOR,
            )

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
        debris_sigma_m=np.array(results['debris_sigma_m']),
        covariance_sources=np.array(results['covariance_sources']),
        decisions=np.array(results['decisions']),
        decision_urgencies=np.array(results['decision_urgencies']),
        propulsion_options=np.array(results['propulsion_options']),
        delta_v_estimates=np.array(results['delta_v_estimates']),
        # SCRUM-383: preserve synthetic/live source provenance so ARBITER
        # can add constructed MAF integration inputs only for the
        # controlled demo scenario, never for live conjunctions.
        source_metadata=data.get("metadata", ""),
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
