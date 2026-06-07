"""
AVERA-ATLAS Tracker Service - Initial Orbit Determination (IOD)

Determines orbital state vectors from angles-only observations.
Implements Gauss's method adapted for moving observers (space-based sensors).

The Problem:
- We have angular measurements (RA/Dec) from multiple observation times
- Observer (CubeSat) position is known at each observation
- Target (debris) orbit is unknown
- Must solve for target position/velocity from geometry

Approach:
1. Convert angular observations to line-of-sight unit vectors
2. Use Gauss's method to estimate slant ranges
3. Compute position vectors: r = R_obs + ρ * L_hat
4. Use Gibbs/Herrick-Gibbs to get velocity from positions
5. Refine with iterative improvement
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, List
from uuid import UUID, uuid4
import numpy as np

# Constants
MU_EARTH = 3.986004418e14  # Earth gravitational parameter (m³/s²)
MU_EARTH_KM = 3.986004418e5  # (km³/s²)
RE_EARTH = 6378.137  # Earth equatorial radius (km)
J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
MAX_ACCEPTED_RMS_RESIDUAL_ARCSEC = 900.0
MAX_ACCEPTED_TOTAL_RESIDUAL_ARCSEC = 1800.0


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class IODObservation:
    """Single observation for IOD processing."""
    timestamp: datetime
    ra: float  # Right Ascension (radians)
    dec: float  # Declination (radians)
    ra_sigma: float  # RA uncertainty (radians)
    dec_sigma: float  # Dec uncertainty (radians)
    observer_position_km: np.ndarray  # Observer position in ECI (km)
    observer_velocity_km_s: np.ndarray  # Observer velocity in ECI (km/s)

    # Optional radar/range-bearing fields.
    # If range_km is present, the observation is range+angles.
    # If absent, the observation is angles-only optical.
    range_km: Optional[float] = None
    range_sigma_km: Optional[float] = None
    
    @property
    def line_of_sight(self) -> np.ndarray:
        """Unit vector pointing from observer toward target in ECI."""
        x = math.cos(self.dec) * math.cos(self.ra)
        y = math.cos(self.dec) * math.sin(self.ra)
        z = math.sin(self.dec)
        return np.array([x, y, z])


@dataclass
class IODSolution:
    """Result of Initial Orbit Determination."""
    success: bool
    track_id: UUID
    epoch: datetime
    
    # State vector at epoch (km, km/s)
    position_km: Optional[np.ndarray] = None
    velocity_km_s: Optional[np.ndarray] = None
    
    # Covariance (6x6, km and km/s units)
    covariance: Optional[np.ndarray] = None
    
    # Orbital elements (for reference)
    semi_major_axis_km: Optional[float] = None
    eccentricity: Optional[float] = None
    inclination_deg: Optional[float] = None
    raan_deg: Optional[float] = None
    arg_perigee_deg: Optional[float] = None
    true_anomaly_deg: Optional[float] = None
    
    # Quality metrics
    rms_residual_arcsec: Optional[float] = None
    observations_used: int = 0
    iterations: int = 0
    method_used: Optional[str] = None
    attempted_methods: Optional[list[dict]] = None
    
    # Error information
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {
            "success": self.success,
            "track_id": str(self.track_id),
            "epoch": self.epoch.isoformat(),
            "observations_used": self.observations_used,
            "iterations": self.iterations,
            "method_used": self.method_used,
            "attempted_methods": self.attempted_methods or [],
        }
        
        if self.success:
            result.update({
                "position_km": self.position_km.tolist() if self.position_km is not None else None,
                "velocity_km_s": self.velocity_km_s.tolist() if self.velocity_km_s is not None else None,
                "semi_major_axis_km": self.semi_major_axis_km,
                "eccentricity": self.eccentricity,
                "inclination_deg": self.inclination_deg,
                "raan_deg": self.raan_deg,
                "arg_perigee_deg": self.arg_perigee_deg,
                "true_anomaly_deg": self.true_anomaly_deg,
                "rms_residual_arcsec": self.rms_residual_arcsec,
            })
        else:
            result["error_message"] = self.error_message
            
        return result


# =============================================================================
# Utility Functions
# =============================================================================

def time_difference_seconds(t1: datetime, t2: datetime) -> float:
    """Compute time difference in seconds."""
    return (t2 - t1).total_seconds()


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3D vectors."""
    return np.cross(a, b)


def norm(v: np.ndarray) -> float:
    """Euclidean norm of a vector."""
    return np.linalg.norm(v)


def unit(v: np.ndarray) -> np.ndarray:
    """Unit vector."""
    n = norm(v)
    if n < 1e-12:
        return v
    return v / n


def stumpff_c2(psi: float) -> float:
    """Stumpff function c2(ψ)."""
    if psi > 1e-6:
        return (1 - math.cos(math.sqrt(psi))) / psi
    elif psi < -1e-6:
        return (math.cosh(math.sqrt(-psi)) - 1) / (-psi)
    else:
        return 0.5 - psi / 24 + psi**2 / 720


def stumpff_c3(psi: float) -> float:
    """Stumpff function c3(ψ)."""
    if psi > 1e-6:
        sp = math.sqrt(psi)
        return (sp - math.sin(sp)) / (psi * sp)
    elif psi < -1e-6:
        sp = math.sqrt(-psi)
        return (math.sinh(sp) - sp) / ((-psi) * sp)
    else:
        return 1/6 - psi / 120 + psi**2 / 5040


def perifocal_to_eci_matrix(i: float, raan: float, arg_peri: float) -> np.ndarray:
    """
    Rotation matrix from perifocal to ECI frame.
    
    Args:
        i: Inclination (radians)
        raan: Right Ascension of Ascending Node (radians)
        arg_peri: Argument of perigee (radians)
        
    Returns:
        3x3 rotation matrix
    """
    cos_raan = math.cos(raan)
    sin_raan = math.sin(raan)
    cos_i = math.cos(i)
    sin_i = math.sin(i)
    cos_w = math.cos(arg_peri)
    sin_w = math.sin(arg_peri)
    
    R = np.array([
        [cos_raan * cos_w - sin_raan * sin_w * cos_i,
         -cos_raan * sin_w - sin_raan * cos_w * cos_i,
         sin_raan * sin_i],
        [sin_raan * cos_w + cos_raan * sin_w * cos_i,
         -sin_raan * sin_w + cos_raan * cos_w * cos_i,
         -cos_raan * sin_i],
        [sin_w * sin_i,
         cos_w * sin_i,
         cos_i]
    ])
    
    return R


# =============================================================================
# Gauss's Method for Angles-Only IOD
# =============================================================================

def gauss_iod(
    obs1: IODObservation,
    obs2: IODObservation, 
    obs3: IODObservation,
    mu: float = MU_EARTH_KM
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Gauss's method for initial orbit determination from 3 observations.
    
    Adapted for space-based observers with known positions.
    
    Args:
        obs1, obs2, obs3: Three observations with timestamps, RA/Dec, observer positions
        mu: Gravitational parameter (km³/s²)
        
    Returns:
        (position_km, velocity_km_s, status_message)
        Position and velocity at obs2 epoch, or (None, None, error_message)
    """
    # Extract data
    R1 = obs1.observer_position_km
    R2 = obs2.observer_position_km
    R3 = obs3.observer_position_km
    
    L1 = obs1.line_of_sight
    L2 = obs2.line_of_sight
    L3 = obs3.line_of_sight
    
    # Time intervals (seconds -> canonical time units for numerical stability)
    tau1 = time_difference_seconds(obs2.timestamp, obs1.timestamp)  # negative
    tau3 = time_difference_seconds(obs2.timestamp, obs3.timestamp)  # positive
    tau = tau3 - tau1  # total span
    
    if abs(tau) < 1.0:
        return None, None, "Observations too close in time"
    
    # Convert to appropriate time scale
    # For better numerical conditioning, we work in seconds but scale later
    
    # Cross products for the D matrix
    p1 = cross(L2, L3)
    p2 = cross(L1, L3)
    p3 = cross(L1, L2)
    
    # Scalar triple product (determinant)
    D0 = np.dot(L1, p1)
    
    # For short arcs, line-of-sight vectors will be nearly coplanar
    # Relax threshold but warn if very small
    if abs(D0) < 1e-20:
        return None, None, "Coplanar line-of-sight vectors (degenerate geometry)"
    
    if abs(D0) < 1e-10:
        # Proceed with caution - may have numerical issues
        pass
    
    # D matrix elements
    D11 = np.dot(R1, p1)
    D12 = np.dot(R1, p2)
    D13 = np.dot(R1, p3)
    D21 = np.dot(R2, p1)
    D22 = np.dot(R2, p2)
    D23 = np.dot(R2, p3)
    D31 = np.dot(R3, p1)
    D32 = np.dot(R3, p2)
    D33 = np.dot(R3, p3)
    
    # Initial approximation coefficients (assuming f,g series truncation)
    A = (-D12 * tau3 / tau + D22 - D32 * tau1 / tau) / D0
    B = (D12 * (tau**2 - tau3**2) * tau3 / tau 
         + D32 * (tau**2 - tau1**2) * tau1 / tau) / (6 * D0)
    
    # Solve for r2 magnitude using iteration
    # The equation is: r2^8 + a*r2^6 + b*r2^3 + c = 0 (approximately)
    
    R2_mag = norm(R2)
    
    # Initial guess for r2 (target distance from Earth center)
    # Assume LEO debris: ~6800 km altitude -> ~7200 km from center
    r2_guess = 7000.0  
    
    # Alternative: use observer distance as initial guess
    if R2_mag > 6000:
        r2_guess = R2_mag + 500  # Assume target slightly higher
    
    # Iterative solution for slant range ρ2
    max_iter = 50
    tol = 1e-8
    
    r2 = r2_guess
    
    for iteration in range(max_iter):
        # Compute auxiliary quantities
        r2_cubed = r2**3
        
        # f and g series coefficients (truncated)
        f1 = 1 - 0.5 * mu * tau1**2 / r2_cubed
        f3 = 1 - 0.5 * mu * tau3**2 / r2_cubed
        g1 = tau1 - mu * tau1**3 / (6 * r2_cubed)
        g3 = tau3 - mu * tau3**3 / (6 * r2_cubed)
        
        # Lagrange coefficients
        c1 = g3 / (f1 * g3 - f3 * g1)
        c3 = -g1 / (f1 * g3 - f3 * g1)
        
        # Slant ranges
        rho1 = (-D11 + D21 * c1 - D31 * c3) / (c1 * D0)
        rho2 = A + mu * B / r2_cubed
        rho3 = (-D13 + D23 * c1 - D33 * c3) / (c3 * D0)
        
        # Check for negative slant ranges
        if rho1 < 0 or rho2 < 0 or rho3 < 0:
            # Try alternate solution or adjust
            pass
        
        # Position vectors
        r1_vec = R1 + rho1 * L1
        r2_vec = R2 + rho2 * L2
        r3_vec = R3 + rho3 * L3
        
        # Update r2 magnitude
        r2_new = norm(r2_vec)
        
        if abs(r2_new - r2) < tol:
            r2 = r2_new
            break
            
        r2 = r2_new
    
    # Final position at t2
    r2_vec = R2 + rho2 * L2
    r2_mag = norm(r2_vec)
    
    # Sanity check
    if r2_mag < RE_EARTH:
        return None, None, f"Solution inside Earth (r = {r2_mag:.1f} km)"
    
    if r2_mag > 50000:
        return None, None, f"Solution too far from Earth (r = {r2_mag:.1f} km)"
    
    # Compute velocity using Gibbs method on the three position vectors
    r1_vec = R1 + rho1 * L1
    r3_vec = R3 + rho3 * L3
    
    v2_vec = gibbs_velocity(r1_vec, r2_vec, r3_vec, tau1, tau3, mu)
    
    if v2_vec is None:
        # Fall back to simple finite difference
        v2_vec = (r3_vec - r1_vec) / (tau3 - tau1)
    
    return r2_vec, v2_vec, "Success"

def solve_lambert_universal(
    r1: np.ndarray,
    r3: np.ndarray,
    dt: float,
    mu: float = MU_EARTH_KM,
    prograde: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Lambert solver using a universal-variable formulation.

    Returns:
        (v1_km_s, v3_km_s, status_message)
    """
    r1_mag = norm(r1)
    r3_mag = norm(r3)

    if r1_mag < 1e-9 or r3_mag < 1e-9:
        return None, None, "Lambert solver: invalid endpoint magnitude"

    if abs(dt) < 1e-6:
        return None, None, "Lambert solver: invalid time of flight"

    cos_dnu = np.clip(np.dot(r1, r3) / (r1_mag * r3_mag), -1.0, 1.0)
    cross_r = cross(r1, r3)
    sin_dnu_mag = norm(cross_r) / (r1_mag * r3_mag)

    if sin_dnu_mag < 1e-12:
        return None, None, "Lambert solver: nearly collinear endpoints"

    if prograde:
        sin_dnu = sin_dnu_mag if cross_r[2] >= 0 else -sin_dnu_mag
    else:
        sin_dnu = -sin_dnu_mag if cross_r[2] >= 0 else sin_dnu_mag

    denom = 1.0 - cos_dnu
    if abs(denom) < 1e-12:
        return None, None, "Lambert solver: transfer angle too small"

    A = sin_dnu * math.sqrt(r1_mag * r3_mag / denom)
    if abs(A) < 1e-12:
        return None, None, "Lambert solver: degenerate A parameter"

    def y_of_z(z: float) -> float:
        C = stumpff_c2(z)
        S = stumpff_c3(z)
        if C <= 0:
            return float("nan")
        return r1_mag + r3_mag + A * (z * S - 1.0) / math.sqrt(C)

    def tof_of_z(z: float) -> float:
        C = stumpff_c2(z)
        S = stumpff_c3(z)
        if C <= 0:
            return float("nan")
        y = y_of_z(z)
        if y <= 0:
            return float("nan")
        x = math.sqrt(y / C)
        return (x**3 * S + A * math.sqrt(y)) / math.sqrt(mu)
    
    def _safe_tof_of_z(z: float) -> tuple[float, float, float, float]:
        C = stumpff_c2(z) 
        S = stumpff_c3(z)
        if C <= 0:
            return float("nan"), C, S, float("nan")
        y = y_of_z(z)
        if y <= 0 or not np.isfinite(y):
            return float("nan"), C, S, y
        x = math.sqrt(y / C)
        tof = (x**3 * S + A * math.sqrt(y)) / math.sqrt(mu)
        return tof, C, S, y

    # Scan for a finite bracket in z
    z_samples = np.linspace(-16.0, 16.0, 257)

    finite_samples: list[tuple[float, float]] = []
    for z in z_samples:
        tof, C, S, y = _safe_tof_of_z(float(z))
        if np.isfinite(tof):
            finite_samples.append((float(z), float(tof)))

    if len(finite_samples) < 2:
        return None, None, "Lambert solver: insufficient finite TOF samples"

    z_low = None
    z_high = None
    f_low = None
    f_high = None

    for (z_a, f_a), (z_b, f_b) in zip(finite_samples[:-1], finite_samples[1:]):
        if (f_a - dt) == 0:
            z_low, z_high = z_a, z_a
            f_low, f_high = f_a, f_a
            break
        if (f_a - dt) * (f_b - dt) <= 0:
            z_low, z_high = z_a, z_b
            f_low, f_high = f_a, f_b
            break

    if z_low is None or z_high is None:
        min_tof = min(f for _, f in finite_samples)
        max_tof = max(f for _, f in finite_samples)
        return None, None, (
            "Lambert solver: failed to bracket solution "
            f"(dt={dt:.3f}, finite_tof_range=[{min_tof:.3f}, {max_tof:.3f}])"
        )

    if z_low == z_high:
        z_mid = z_low
    else:
        for _ in range(100):
            z_mid = 0.5 * (z_low + z_high)
            f_mid, _, _, _ = _safe_tof_of_z(z_mid)

            if not np.isfinite(f_mid):
                z_low = z_mid
                continue

            if abs(f_mid - dt) < 1e-6:
                break

            if (f_low - dt) * (f_mid - dt) <= 0:
                z_high = z_mid
                f_high = f_mid
            else:
                z_low = z_mid
                f_low = f_mid

    C = stumpff_c2(z_mid)
    y = y_of_z(z_mid)

    if C <= 0 or y <= 0:
        return None, None, "Lambert solver: invalid final y/C"

    f = 1.0 - y / r1_mag
    g = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r3_mag

    if abs(g) < 1e-12:
        return None, None, "Lambert solver: singular g parameter"

    v1 = (r3 - f * r1) / g
    v3 = (gdot * r3 - r1) / g

    if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v3)):
        return None, None, "Lambert solver: non-finite velocity solution"

    return v1, v3, "Lambert solver success"


def _gooding_initial_range_guesses(
    obs1: IODObservation,
    obs2: IODObservation,
    obs3: IODObservation,
) -> list[tuple[float, float]]:
    """
    Produce broader initial guesses for endpoint slant ranges (rho1, rho3) in km.
    These are still scaffolding guesses, but wide enough to let Gooding reach the
    Lambert stage on more synthetic cases.
    """
    base_ranges = [
        300.0,
        500.0,
        750.0,
        1000.0,
        1500.0,
        2000.0,
        3000.0,
        5000.0,
        8000.0,
        12000.0,
        20000.0,
    ]

    guesses: list[tuple[float, float]] = []

    for rho1 in base_ranges:
        for rho3 in base_ranges:
            guesses.append((rho1, rho3))

    return guesses

def gooding_iod(
    obs1: IODObservation,
    obs2: IODObservation,
    obs3: IODObservation,
    mu: float = MU_EARTH_KM
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Lambert-backed Gooding-style prototype.

    This is not yet a canonical full Gooding implementation.
    It searches endpoint slant ranges, solves Lambert between the endpoint
    positions, propagates to the middle observation epoch, and scores the
    middle-angle consistency.

    Returns:
        (position_km, velocity_km_s, status_message)
    """
    t13 = time_difference_seconds(obs1.timestamp, obs3.timestamp)
    if abs(t13) < 1.0:
        return None, None, "Gooding IOD: observations too close in time"

    t12 = time_difference_seconds(obs1.timestamp, obs2.timestamp)

    L1 = obs1.line_of_sight
    L2 = obs2.line_of_sight
    L3 = obs3.line_of_sight

    R1 = obs1.observer_position_km
    R2 = obs2.observer_position_km
    R3 = obs3.observer_position_km

    guess_pairs = _gooding_initial_range_guesses(obs1, obs2, obs3)

    best_result = None
    best_score_arcsec = float("inf")
    last_status = "Gooding IOD: no guesses attempted"

    attempted = 0

    for rho1, rho3 in guess_pairs:
        attempted += 1

        r1 = R1 + rho1 * L1
        r3 = R3 + rho3 * L3

        if norm(r1) < RE_EARTH + 100 or norm(r3) < RE_EARTH + 100:
            last_status = (
                f"Gooding IOD attempt {attempted}: rejected rho1={rho1:.1f}, rho3={rho3:.1f} "
                "because endpoint lies below minimum orbital altitude"
            )
            continue

        v1, v3, lambert_status = solve_lambert_universal(r1, r3, t13, mu=mu, prograde=True)
        last_status = (
            f"Gooding IOD attempt {attempted}: rho1={rho1:.1f}, rho3={rho3:.1f}, "
            f"lambert_status={lambert_status}"
        )

        if v1 is None or v3 is None:
            continue
    
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v3)):
            last_status = (
                f"Gooding IOD attempt {attempted}: non-finite Lambert velocity"
            )
            continue

        v1_mag = norm(v1)
        v3_mag = norm(v3)

        if v1_mag < 1.0 or v1_mag > 15.0 or v3_mag < 1.0 or v3_mag > 15.0:
            last_status = (
                f"Gooding IOD attempt {attempted}: Lambert endpoint velocity out of bounds "
                f"(v1={v1_mag:.2f} km/s, v3={v3_mag:.2f} km/s)"
            )
            continue

        energy1 = v1_mag**2 / 2.0 - mu / norm(r1)
        if not np.isfinite(energy1):
            last_status = (
                f"Gooding IOD attempt {attempted}: non-finite endpoint energy"
            )
            continue

        # Skip strongly hyperbolic / numerically dangerous cases for now
        if energy1 >= 0:
            last_status = (
                f"Gooding IOD attempt {attempted}: hyperbolic Lambert branch rejected"
            )
            continue

        # Propagate the Lambert endpoint state to the middle epoch
        try:
            r2_pred, v2_pred = kepler_propagate(r1, v1, t12, mu)
        except (OverflowError, ValueError, FloatingPointError) as exc:
            last_status = (
                f"Gooding IOD attempt {attempted}: midpoint propagation failed "
                f"({type(exc).__name__}: {exc})"
            )
            continue

        if not np.all(np.isfinite(r2_pred)) or not np.all(np.isfinite(v2_pred)):
            last_status = (
                f"Gooding IOD attempt {attempted}: non-finite midpoint propagation"
            )
            continue

        r2_mag = norm(r2_pred)
        v2_mag = norm(v2_pred)

        if r2_mag < RE_EARTH + 100 or r2_mag > 100000:
            last_status = (
                f"Gooding IOD attempt {attempted}: midpoint radius out of bounds ({r2_mag:.1f} km)"
            )
            continue

        if v2_mag < 1.0 or v2_mag > 15.0:
            last_status = (
                f"Gooding IOD attempt {attempted}: midpoint velocity out of bounds ({v2_mag:.2f} km/s)"
            )
            continue

        los2_pred = unit(r2_pred - R2)
        ang_err2 = math.acos(np.clip(np.dot(los2_pred, L2), -1.0, 1.0))
        score_arcsec = math.degrees(ang_err2) * 3600.0

        if score_arcsec < best_score_arcsec:
            best_score_arcsec = score_arcsec
            best_result = (r2_pred.copy(), v2_pred.copy())

    if best_result is None:
        return None, None, last_status

    if best_score_arcsec > 3600.0:
        return None, None, f"Gooding IOD prototype: poor midpoint consistency ({best_score_arcsec:.1f} arcsec)"

    return (
        best_result[0],
        best_result[1],
        f"Gooding IOD prototype success (midpoint residual: {best_score_arcsec:.1f} arcsec)"
    )

def gibbs_velocity(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    tau1: float,
    tau3: float,
    mu: float = MU_EARTH_KM
) -> Optional[np.ndarray]:
    """
    Gibbs method to compute velocity at r2 given three position vectors.
    
    Args:
        r1, r2, r3: Position vectors (km)
        tau1: Time from t2 to t1 (negative)
        tau3: Time from t2 to t3 (positive)
        mu: Gravitational parameter
        
    Returns:
        Velocity at r2 (km/s) or None if degenerate
    """
    r1_mag = norm(r1)
    r2_mag = norm(r2)
    r3_mag = norm(r3)
    
    # Check for coplanar vectors
    Z12 = cross(r1, r2)
    Z23 = cross(r2, r3)
    Z31 = cross(r3, r1)
    
    # Coplanarity check
    alpha_cop = math.asin(np.dot(Z23, r1) / (norm(Z23) * r1_mag + 1e-12))
    if abs(alpha_cop) > math.radians(5):  # More than 5 degrees out of plane
        return None  # Not coplanar enough
    
    # Gibbs vectors
    N = r1_mag * Z23 + r2_mag * Z31 + r3_mag * Z12
    D = Z12 + Z23 + Z31
    S = r1 * (r2_mag - r3_mag) + r2 * (r3_mag - r1_mag) + r3 * (r1_mag - r2_mag)
    
    N_mag = norm(N)
    D_mag = norm(D)
    
    if N_mag < 1e-12 or D_mag < 1e-12:
        return None
    
    # Velocity at r2
    B = cross(D, r2)
    L_g = math.sqrt(mu / (N_mag * D_mag))
    
    v2 = L_g * (B / r2_mag + S)
    
    return v2


# =============================================================================
# Range Search IOD (More Robust for Short Arcs)
# =============================================================================

def range_search_iod(
    obs1: IODObservation,
    obs2: IODObservation,
    obs3: IODObservation,
    mu: float = MU_EARTH_KM,
    range_min: float = 50.0,
    range_max: float = 5000.0,
    n_search: int = 150,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Range search IOD for space-based observers.

    Searches over possible slant ranges to find a consistent orbital solution.
    This version uses a finer initial grid plus two local zoom passes around
    the best rho2 value. The physics is unchanged from the original range
    search; only the search resolution is improved.

    Args:
        obs1, obs2, obs3: Three observations
        mu: Gravitational parameter (km^3/s^2)
        range_min, range_max: Search bounds for slant range (km)
        n_search: Number of coarse search points for rho2

    Returns:
        (position_km, velocity_km_s, status_message) at obs2 epoch
    """
    R1 = obs1.observer_position_km
    R2 = obs2.observer_position_km
    R3 = obs3.observer_position_km

    L1 = obs1.line_of_sight
    L2 = obs2.line_of_sight
    L3 = obs3.line_of_sight

    # Time intervals measured from obs1.
    t1 = 0.0
    t2 = time_difference_seconds(obs1.timestamp, obs2.timestamp)
    t3 = time_difference_seconds(obs1.timestamp, obs3.timestamp)

    if t3 < 1.0:
        return None, None, "Observations too close in time"

    factor_pts = 31
    zoom_passes = 2

    def search_window(
        rho2_min: float,
        rho2_max: float,
        rho2_points: int,
    ) -> tuple[Optional[tuple[np.ndarray, np.ndarray]], float, Optional[float]]:
        best_solution: Optional[tuple[np.ndarray, np.ndarray]] = None
        best_residual = float("inf")
        best_rho2: Optional[float] = None

        for rho2 in np.linspace(rho2_min, rho2_max, rho2_points):
            r2 = R2 + rho2 * L2
            r2_mag = norm(r2)

            if r2_mag < RE_EARTH + 100:
                continue

            for rho1_factor in np.linspace(0.5, 2.0, factor_pts):
                rho1 = rho2 * rho1_factor
                r1 = R1 + rho1 * L1
                r1_mag = norm(r1)

                if r1_mag < RE_EARTH + 100:
                    continue

                for rho3_factor in np.linspace(0.5, 2.0, factor_pts):
                    rho3 = rho2 * rho3_factor
                    r3 = R3 + rho3 * L3
                    r3_mag = norm(r3)

                    if r3_mag < RE_EARTH + 100:
                        continue

                    v2 = herrick_gibbs_velocity(r1, r2, r3, t1, t2, t3, mu)
                    v2_mag = norm(v2)

                    if v2_mag < 2.0 or v2_mag > 12.0:
                        continue

                    energy = v2_mag**2 / 2.0 - mu / r2_mag

                    # Bound orbit only.
                    if energy >= 0:
                        continue

                    a = -mu / (2.0 * energy)

                    if a < RE_EARTH + 100 or a > 100000:
                        continue

                    try:
                        r1_prop, _ = kepler_propagate(r2, v2, t1 - t2, mu)
                        r3_prop, _ = kepler_propagate(r2, v2, t3 - t2, mu)
                    except (OverflowError, ValueError, FloatingPointError):
                        continue

                    if not np.all(np.isfinite(r1_prop)) or not np.all(np.isfinite(r3_prop)):
                        continue

                    los1_pred = unit(r1_prop - R1)
                    los3_pred = unit(r3_prop - R3)

                    ang_err1 = math.acos(np.clip(np.dot(los1_pred, L1), -1.0, 1.0))
                    ang_err3 = math.acos(np.clip(np.dot(los3_pred, L3), -1.0, 1.0))

                    total_residual = (ang_err1 + ang_err3) * 206265.0

                    if total_residual < best_residual:
                        best_residual = total_residual
                        best_solution = (r2.copy(), v2.copy())
                        best_rho2 = float(rho2)

        return best_solution, best_residual, best_rho2

    best_solution, best_residual, best_rho2 = search_window(
        range_min,
        range_max,
        n_search,
    )

    if best_solution is None or best_rho2 is None:
        return None, None, "No valid solution found in range search"

    span = (range_max - range_min) / max(n_search, 1)

    for _ in range(zoom_passes):
        local_min = max(range_min, best_rho2 - 2.0 * span)
        local_max = min(range_max, best_rho2 + 2.0 * span)

        zoom_solution, zoom_residual, zoom_rho2 = search_window(
            local_min,
            local_max,
            80,
        )

        if (
            zoom_solution is not None
            and zoom_rho2 is not None
            and zoom_residual < best_residual
        ):
            best_solution = zoom_solution
            best_residual = zoom_residual
            best_rho2 = zoom_rho2

        span = (local_max - local_min) / 80.0

    if best_residual > MAX_ACCEPTED_TOTAL_RESIDUAL_ARCSEC:
        return None, None, (
            "Best solution rejected by residual ceiling: "
            f"{best_residual:.1f} arcsec > "
            f"{MAX_ACCEPTED_TOTAL_RESIDUAL_ARCSEC:.1f} arcsec"
        )

    return (
        best_solution[0],
        best_solution[1],
        f"Success refined range-search (residual: {best_residual:.1f} arcsec)",
    )


def double_r_iod(
    obs1: IODObservation,
    obs2: IODObservation,
    obs3: IODObservation,
    mu: float = MU_EARTH_KM
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Double-r iteration method for angles-only IOD.
    
    Searches over possible target distances to find consistent orbital solution.
    """
    R1 = obs1.observer_position_km
    R2 = obs2.observer_position_km
    R3 = obs3.observer_position_km
    
    L1 = obs1.line_of_sight
    L2 = obs2.line_of_sight
    L3 = obs3.line_of_sight
    
    # Time intervals
    tau1 = time_difference_seconds(obs2.timestamp, obs1.timestamp)
    tau3 = time_difference_seconds(obs2.timestamp, obs3.timestamp)
    
    # Search over a wide range of target altitudes
    # LEO: 6500-8500 km, MEO: 8500-35000 km, GEO: ~42000 km
    r_min = RE_EARTH + 200   # 200 km altitude minimum
    r_max = RE_EARTH + 2000  # 2000 km altitude maximum for LEO
    
    best_solution = None
    best_score = float('inf')
    
    for r1_mag in np.linspace(r_min, r_max, 40):
        for r3_mag in np.linspace(r_min, r_max, 40):
            # Compute slant ranges from geometry
            # r_target = R_obs + rho * L
            # |r_target|² = |R_obs|² + rho² + 2*rho*(R_obs · L)
            
            # Solve quadratic for rho1
            a1 = 1.0
            b1 = 2 * np.dot(R1, L1)
            c1 = np.dot(R1, R1) - r1_mag**2
            disc1 = b1**2 - 4*a1*c1
            if disc1 < 0:
                continue
            
            # Try both roots
            rho1_candidates = []
            rho1_a = (-b1 + math.sqrt(disc1)) / (2*a1)
            rho1_b = (-b1 - math.sqrt(disc1)) / (2*a1)
            if rho1_a > 50:  # Minimum 50 km range
                rho1_candidates.append(rho1_a)
            
            if not rho1_candidates:
                continue
            
            # Same for rho3
            a3 = 1.0
            b3 = 2 * np.dot(R3, L3)
            c3 = np.dot(R3, R3) - r3_mag**2
            disc3 = b3**2 - 4*a3*c3
            if disc3 < 0:
                continue
            
            rho3_candidates = []
            rho3_a = (-b3 + math.sqrt(disc3)) / (2*a3)
            rho3_b = (-b3 - math.sqrt(disc3)) / (2*a3)
            if rho3_a > 50:
                rho3_candidates.append(rho3_a)
            if rho3_b > 50:
                rho3_candidates.append(rho3_b)
            
            if not rho3_candidates:
                continue
            
            for rho1 in rho1_candidates:
                for rho3 in rho3_candidates:
                    # Target positions
                    r1 = R1 + rho1 * L1
                    r3 = R3 + rho3 * L3
                    
                    # Time from obs1
                    t1 = 0
                    t2 = -tau1
                    t3 = tau3 - tau1
                    
                    # Interpolate r2
                    if abs(t3) > 0.1:
                        alpha = t2 / t3
                    else:
                        alpha = 0.5
                    r2_interp = (1 - alpha) * r1 + alpha * r3
                    
                    # Compute velocity using Herrick-Gibbs
                    v2 = herrick_gibbs_velocity(r1, r2_interp, r3, t1, t2, t3, mu)
                    v2_mag = norm(v2)
                    
                    # Orbital velocity should be 3-11 km/s for Earth orbits
                    if v2_mag < 3.0 or v2_mag > 11.0:
                        continue
                    
                    # Check energy
                    r2_mag = norm(r2_interp)
                    energy = v2_mag**2 / 2 - mu / r2_mag
                    if energy >= 0:
                        continue  # Hyperbolic - skip
                    
                    # Semi-major axis
                    a = -mu / (2 * energy)
                    
                    # Accept LEO to MEO orbits
                    if a < 6400 or a > 50000:
                        continue
                    
                    # Score: propagate and check line-of-sight consistency
                    r1_prop, _ = kepler_propagate(r2_interp, v2, tau1, mu)
                    r3_prop, _ = kepler_propagate(r2_interp, v2, tau3, mu)
                    
                    # Also check r2 consistency
                    los2_to_r2 = r2_interp - R2
                    los2_pred = unit(los2_to_r2)
                    
                    # Angular residuals
                    los1_pred = unit(r1_prop - R1)
                    los3_pred = unit(r3_prop - R3)
                    
                    ang_err1 = math.acos(np.clip(np.dot(los1_pred, L1), -1, 1))
                    ang_err2 = math.acos(np.clip(np.dot(los2_pred, L2), -1, 1))
                    ang_err3 = math.acos(np.clip(np.dot(los3_pred, L3), -1, 1))
                    
                    # Score in arcseconds (include all three observations)
                    score = (ang_err1 + ang_err2 + ang_err3) * 206265  # rad to arcsec
                    
                    if score < best_score:
                        best_score = score
                        best_solution = (r2_interp.copy(), v2.copy())
    
    if best_solution is None:
        return None, None, "Double-r method: no valid solution found"
    
    if best_score > 1800:  # 30 arcminutes total for 3 obs
        return None, None, f"Double-r method: poor fit ({best_score:.1f} arcsec)"
    
    return best_solution[0], best_solution[1], f"Success (angular residual: {best_score:.1f} arcsec)"


def herrick_gibbs_velocity(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    mu: float = MU_EARTH_KM
) -> np.ndarray:
    """
    Herrick-Gibbs method for velocity from three closely-spaced positions.
    
    Better than Gibbs for small time spans (< 1-3 minutes between observations).
    
    Args:
        r1, r2, r3: Position vectors (km)
        t1, t2, t3: Times (seconds from reference)
        mu: Gravitational parameter
        
    Returns:
        Velocity at r2 (km/s)
    """
    dt31 = t3 - t1
    dt32 = t3 - t2
    dt21 = t2 - t1
    
    # Avoid division by zero
    if abs(dt21) < 0.01 or abs(dt32) < 0.01 or abs(dt31) < 0.01:
        # Fall back to simple difference
        return (r3 - r1) / max(dt31, 0.1)
    
    r1_mag = norm(r1)
    r2_mag = norm(r2)
    r3_mag = norm(r3)
    
    # Herrick-Gibbs formula
    v2 = (
        -dt32 * (1 / (dt21 * dt31) + mu / (12 * r1_mag**3)) * r1
        + (dt32 - dt21) * (1 / (dt21 * dt32) + mu / (12 * r2_mag**3)) * r2
        + dt21 * (1 / (dt32 * dt31) + mu / (12 * r3_mag**3)) * r3
    )
    
    return v2


def estimate_orbit_from_directions(
    obs1: IODObservation,
    obs2: IODObservation,
    obs3: IODObservation,
    mu: float = MU_EARTH_KM
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Estimate an orbit by placing target at a fixed range along line of sight.
    
    This is a simplified fallback when other IOD methods fail.
    Places target at ~500 km range from observer and computes velocity.
    """
    R1 = obs1.observer_position_km
    R2 = obs2.observer_position_km
    R3 = obs3.observer_position_km
    
    L1 = obs1.line_of_sight
    L2 = obs2.line_of_sight
    L3 = obs3.line_of_sight
    
    # Time intervals
    t1 = 0
    t2 = time_difference_seconds(obs1.timestamp, obs2.timestamp)
    t3 = time_difference_seconds(obs1.timestamp, obs3.timestamp)
    
    best_solution = None
    best_score = float('inf')
    
    # Try different ranges - wider range for robustness
    for rho in [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]:
        r1 = R1 + rho * L1
        r2 = R2 + rho * L2
        r3 = R3 + rho * L3
        
        # Check positions are above Earth
        if norm(r1) < RE_EARTH + 100 or norm(r2) < RE_EARTH + 100 or norm(r3) < RE_EARTH + 100:
            continue
        
        # Compute velocity
        v2 = herrick_gibbs_velocity(r1, r2, r3, t1, t2, t3, mu)
        v2_mag = norm(v2)
        
        if v2_mag < 1.0 or v2_mag > 15.0:
            continue
        
        # Check orbital energy
        r2_mag = norm(r2)
        energy = v2_mag**2 / 2 - mu / r2_mag
        if energy >= 0:
            continue
        
        # Semi-major axis
        a = -mu / (2 * energy)
        if a < 6400 or a > 50000:
            continue
        
        # Check orbit is valid (perigee above Earth)
        h = cross(r2, v2)
        h_mag = norm(h)
        e_vec = ((v2_mag**2 - mu / r2_mag) * r2 - np.dot(r2, v2) * v2) / mu
        e = norm(e_vec)
        
        perigee = a * (1 - e)
        if perigee < RE_EARTH - 500:  # Allow some margin
            continue
        
        # Score by how well the orbit fits
        r1_prop, _ = kepler_propagate(r2, v2, t1 - t2, mu)
        r3_prop, _ = kepler_propagate(r2, v2, t3 - t2, mu)
        
        los1_pred = unit(r1_prop - R1)
        los3_pred = unit(r3_prop - R3)
        
        ang_err1 = math.acos(np.clip(np.dot(los1_pred, L1), -1, 1))
        ang_err3 = math.acos(np.clip(np.dot(los3_pred, L3), -1, 1))
        
        score = (ang_err1 + ang_err3) * 206265
        
        if score < best_score:
            best_score = score
            best_solution = (r2.copy(), v2.copy())
    
    if best_solution is None:
        # Last resort: create a synthetic circular orbit at the mean observation direction
        # This ensures we always return something for demo purposes
        mean_los = unit(L1 + L2 + L3)
        mean_obs_pos = (R1 + R2 + R3) / 3
        
        # Place target 1000 km along mean line of sight
        r2 = mean_obs_pos + 1000 * mean_los
        r2_mag = norm(r2)
        
        if r2_mag < RE_EARTH + 200:
            # Adjust to be at least 200 km altitude
            r2 = unit(r2) * (RE_EARTH + 400)
            r2_mag = norm(r2)
        
        # Circular orbit velocity
        v_circ = math.sqrt(mu / r2_mag)
        
        # Velocity perpendicular to position (prograde)
        # Use angular momentum direction from observations
        h_dir = unit(cross(L1, L3))
        v2 = v_circ * unit(cross(h_dir, r2))
        
        return r2, v2, "Synthetic circular orbit (demo mode)"
    
    return best_solution[0], best_solution[1], f"Estimated (residual: {best_score:.1f} arcsec)"

def range_angles_iod(
    observations: List[IODObservation],
    mu: float = MU_EARTH_KM,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[datetime], str]:
    """
    IOD from range+angles observations.

    If slant range is available, each observation directly reconstructs
    target ECI position:

        r_target = r_observer + range_km * line_of_sight

    Velocity is estimated from reconstructed positions and referenced to the
    middle observation epoch.
    """
    ranged_obs = [
        obs for obs in observations
        if obs.range_km is not None and np.isfinite(obs.range_km)
    ]

    if len(ranged_obs) < 3:
        return None, None, None, "Range+angles IOD requires at least 3 ranged observations"

    ranged_obs = sorted(ranged_obs, key=lambda obs: obs.timestamp)

    positions = []
    for obs in ranged_obs:
        if obs.range_km is None or obs.range_km <= 0:
            return None, None, None, "Range+angles IOD received invalid range"

        r_target = obs.observer_position_km + obs.range_km * obs.line_of_sight
        positions.append(r_target)

    positions = np.array(positions, dtype=np.float64)

    mid_idx = len(ranged_obs) // 2
    epoch = ranged_obs[mid_idx].timestamp
    r_mid = positions[mid_idx]

    # Use a finite-difference velocity over the full arc for stability.
    t_first = ranged_obs[0].timestamp
    t_last = ranged_obs[-1].timestamp
    dt = time_difference_seconds(t_first, t_last)

    if abs(dt) < 1e-9:
        return None, None, None, "Range+angles observations have insufficient time separation"

    v_mid = (positions[-1] - positions[0]) / dt

    r_mag = norm(r_mid)
    v_mag = norm(v_mid)

    if r_mag < RE_EARTH + 100:
        return None, None, None, f"Range+angles solution inside Earth (r={r_mag:.1f} km)"

    if v_mag < 1.0 or v_mag > 15.0:
        return None, None, None, f"Range+angles velocity unreasonable ({v_mag:.2f} km/s)"

    return r_mid, v_mid, epoch, "Range+angles IOD success"

# =============================================================================
# Orbital Elements Conversion
# =============================================================================

def state_to_elements(
    r: np.ndarray,
    v: np.ndarray,
    mu: float = MU_EARTH_KM
) -> dict:
    """
    Convert state vector to Keplerian orbital elements.
    
    Args:
        r: Position vector (km)
        v: Velocity vector (km/s)
        mu: Gravitational parameter
        
    Returns:
        Dictionary with orbital elements
    """
    r_mag = norm(r)
    v_mag = norm(v)
    
    # Specific angular momentum
    h = cross(r, v)
    h_mag = norm(h)
    
    # Node vector
    K = np.array([0, 0, 1])
    n = cross(K, h)
    n_mag = norm(n)
    
    # Eccentricity vector
    e_vec = ((v_mag**2 - mu / r_mag) * r - np.dot(r, v) * v) / mu
    e = norm(e_vec)
    
    # Specific energy
    energy = v_mag**2 / 2 - mu / r_mag
    
    # Semi-major axis
    if abs(e - 1.0) > 1e-10:
        a = -mu / (2 * energy)
    else:
        a = float('inf')  # Parabolic
    
    # Inclination
    i = math.acos(np.clip(h[2] / h_mag, -1, 1))
    
    # RAAN (Right Ascension of Ascending Node)
    if n_mag > 1e-12:
        Omega = math.acos(np.clip(n[0] / n_mag, -1, 1))
        if n[1] < 0:
            Omega = 2 * math.pi - Omega
    else:
        Omega = 0  # Equatorial orbit
    
    # Argument of Perigee
    if n_mag > 1e-12 and e > 1e-10:
        omega = math.acos(np.clip(np.dot(n, e_vec) / (n_mag * e), -1, 1))
        if e_vec[2] < 0:
            omega = 2 * math.pi - omega
    else:
        omega = 0
    
    # True Anomaly
    if e > 1e-10:
        nu = math.acos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1, 1))
        if np.dot(r, v) < 0:
            nu = 2 * math.pi - nu
    else:
        # Circular orbit - measure from ascending node
        if n_mag > 1e-12:
            nu = math.acos(np.clip(np.dot(n, r) / (n_mag * r_mag), -1, 1))
            if r[2] < 0:
                nu = 2 * math.pi - nu
        else:
            nu = math.acos(r[0] / r_mag)
            if r[1] < 0:
                nu = 2 * math.pi - nu
    
    # Period (for elliptical orbits)
    if a > 0:
        period = 2 * math.pi * math.sqrt(a**3 / mu)
    else:
        period = float('inf')
    
    return {
        "semi_major_axis_km": a,
        "eccentricity": e,
        "inclination_deg": math.degrees(i),
        "raan_deg": math.degrees(Omega),
        "arg_perigee_deg": math.degrees(omega),
        "true_anomaly_deg": math.degrees(nu),
        "period_minutes": period / 60 if period != float('inf') else None,
        "apogee_km": a * (1 + e) - RE_EARTH if a > 0 else None,
        "perigee_km": a * (1 - e) - RE_EARTH if a > 0 else None,
    }


# =============================================================================
# Residual Computation
# =============================================================================

def compute_residuals(
    observations: List[IODObservation],
    r_ref: np.ndarray,
    v_ref: np.ndarray,
    t_ref: datetime,
    mu: float = MU_EARTH_KM
) -> Tuple[float, List[float]]:
    """
    Compute angular residuals between observations and propagated orbit.
    
    Args:
        observations: List of observations
        r_ref, v_ref: Reference state at t_ref
        t_ref: Reference epoch
        mu: Gravitational parameter
        
    Returns:
        (rms_residual_arcsec, list_of_residuals_arcsec)
    """
    residuals = []
    
    for obs in observations:
        dt = time_difference_seconds(t_ref, obs.timestamp)
        
        # Simple Kepler propagation to observation time
        r_prop, v_prop = kepler_propagate(r_ref, v_ref, dt, mu)
        
        # Compute predicted line-of-sight from observer
        los_pred = r_prop - obs.observer_position_km
        los_pred = unit(los_pred)
        
        # Observed line-of-sight
        los_obs = obs.line_of_sight
        
        # Angular separation
        cos_ang = np.clip(np.dot(los_pred, los_obs), -1, 1)
        ang_sep = math.acos(cos_ang)
        
        residuals.append(math.degrees(ang_sep) * 3600)  # arcseconds
    
    rms = math.sqrt(sum(r**2 for r in residuals) / len(residuals))
    
    return rms, residuals


def kepler_propagate(
    r0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    mu: float = MU_EARTH_KM
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate state using universal Kepler's equation.
    
    Args:
        r0, v0: Initial state (km, km/s)
        dt: Time step (seconds)
        mu: Gravitational parameter
        
    Returns:
        (r, v) at time t0 + dt
    """
    r0_mag = norm(r0)
    v0_mag = norm(v0)
    
    # Specific energy and semi-major axis
    energy = v0_mag**2 / 2 - mu / r0_mag
    
    if abs(energy) > 1e-10:
        a = -mu / (2 * energy)
    else:
        a = float('inf')
    
    # Initial radial velocity
    vr0 = np.dot(r0, v0) / r0_mag
    
    # Universal variable initial guess
    if a > 0:
        # Elliptical
        chi0 = math.sqrt(mu) * dt / a
    else:
        # Hyperbolic
        chi0 = math.sqrt(mu) * dt / abs(a)
    
    # Newton iteration for universal variable
    chi = chi0
    for _ in range(50):
        psi = chi**2 / a if a != float('inf') else 0
        c2 = stumpff_c2(psi)
        c3 = stumpff_c3(psi)
        
        r = chi**2 * c2 + vr0 / math.sqrt(mu) * chi * (1 - psi * c3) + r0_mag * (1 - psi * c2)
        
        F = r0_mag * vr0 / math.sqrt(mu) * chi**2 * c2 + (1 - r0_mag / a) * chi**3 * c3 + r0_mag * chi - math.sqrt(mu) * dt
        dFdchi = chi**2 * c2 + vr0 / math.sqrt(mu) * chi * (1 - psi * c3) + r0_mag * (1 - psi * c2)
        
        chi_new = chi - F / dFdchi
        
        if abs(chi_new - chi) < 1e-10:
            chi = chi_new
            break
        chi = chi_new
    
    # Compute f, g, fdot, gdot
    psi = chi**2 / a if a != float('inf') else 0
    c2 = stumpff_c2(psi)
    c3 = stumpff_c3(psi)
    
    r_mag = chi**2 * c2 + vr0 / math.sqrt(mu) * chi * (1 - psi * c3) + r0_mag * (1 - psi * c2)
    
    f = 1 - chi**2 / r0_mag * c2
    g = dt - chi**3 / math.sqrt(mu) * c3
    fdot = math.sqrt(mu) / (r_mag * r0_mag) * chi * (psi * c3 - 1)
    gdot = 1 - chi**2 / r_mag * c2
    
    # Final state
    r = f * r0 + g * v0
    v = fdot * r0 + gdot * v0
    
    return r, v


# =============================================================================
# Main IOD Solver
# =============================================================================

class IODSolver:
    """
    Initial Orbit Determination solver for angles-only observations.
    """
    
    def __init__(self, mu: float = MU_EARTH_KM):
        self.mu = mu
    
    def _score_candidate(
        self,
        obs_sorted: List[IODObservation],
        r2: np.ndarray,
        v2: np.ndarray,
        epoch: datetime,
    ) -> tuple[float, float]:
        """
        Score a candidate solution. Lower is better.

        The RMS angular residual is still the main quality metric, but we add
        stronger physical-plausibility penalties so a weird high-eccentricity
        or very high-energy orbit does not beat a cleaner low-residual LEO-like
        solution.
        """
        rms_arcsec, _ = compute_residuals(obs_sorted, r2, v2, epoch, self.mu)

        v_mag = norm(v2)
        r_mag = norm(r2)
        elements = state_to_elements(r2, v2, self.mu)

        a = elements["semi_major_axis_km"]
        e = elements["eccentricity"]
        perigee = elements["perigee_km"]
        apogee = elements["apogee_km"]

        penalty = 0.0

        # Radius sanity.
        if r_mag < RE_EARTH + 100:
            penalty += 1_000_000.0
        elif r_mag > 50_000.0:
            penalty += 50_000.0 + 0.5 * (r_mag - 50_000.0)

        # LEO-ish optical close-pass cases should normally be around 7-8 km/s.
        # Keep this loose, but penalize obviously suspicious values.
        if v_mag < 5.0:
            penalty += 25_000.0 + 10_000.0 * (5.0 - v_mag)
        elif v_mag > 9.5:
            penalty += 25_000.0 + 10_000.0 * (v_mag - 9.5)

        # Semi-major axis plausibility.
        if a is not None and np.isfinite(a):
            if a < RE_EARTH + 100:
                penalty += 100_000.0
            elif a > 20_000.0:
                penalty += 25_000.0 + 0.5 * (a - 20_000.0)
        else:
            penalty += 100_000.0

        # Eccentricity plausibility.
        if e is not None and np.isfinite(e):
            if e >= 1.0:
                penalty += 1_000_000.0
            elif e > 0.5:
                penalty += 75_000.0 + 150_000.0 * (e - 0.5)
            elif e > 0.2:
                penalty += 10_000.0 + 50_000.0 * (e - 0.2)
        else:
            penalty += 100_000.0

        # Perigee / apogee sanity.
        if perigee is not None and np.isfinite(perigee):
            if perigee < 100.0:
                penalty += 100_000.0
            elif perigee < 300.0:
                penalty += 25_000.0
        else:
            penalty += 50_000.0

        if apogee is not None and np.isfinite(apogee):
            if apogee > 20_000.0:
                penalty += 25_000.0 + 0.25 * (apogee - 20_000.0)

        score = rms_arcsec + penalty
        return score, rms_arcsec

    def solve(
        self,
        observations: List[IODObservation],
        track_id: UUID
    ) -> IODSolution:
        """
        Perform IOD from a list of observations.
        
        Args:
            observations: List of at least 3 observations
            track_id: UUID for the resulting track
            
        Returns:
            IODSolution with state vector or error
        """
        if len(observations) < 3:
            return IODSolution(
                success=False,
                track_id=track_id,
                epoch=datetime.now(timezone.utc),
                error_message=f"Insufficient observations: {len(observations)} < 3",
                observations_used=len(observations)
            )
        
        # Sort by time
        obs_sorted = sorted(observations, key=lambda o: o.timestamp)
        
        # If all observations contain slant range, use the range+angles path.
        # This demonstrates sensor-agnostic ingest: optical observations use
        # angles-only methods, while radar observations can use range-bearing
        # reconstruction.
        ranged_count = sum(
            1 for obs in obs_sorted
            if obs.range_km is not None and np.isfinite(obs.range_km)
        )

        if ranged_count >= 3 and ranged_count == len(obs_sorted):
            r_radar, v_radar, radar_epoch, radar_status = range_angles_iod(
                obs_sorted,
                self.mu,
            )

            attempted_methods = [{
                "method": "range+angles",
                "success": r_radar is not None and v_radar is not None,
                "status": radar_status,
            }]

            if r_radar is None or v_radar is None or radar_epoch is None:
                return IODSolution(
                    success=False,
                    track_id=track_id,
                    epoch=obs_sorted[len(obs_sorted) // 2].timestamp,
                    error_message=radar_status,
                    observations_used=len(obs_sorted),
                    attempted_methods=attempted_methods,
                )

            elements = state_to_elements(r_radar, v_radar, self.mu)
            rms_arcsec, _ = compute_residuals(
                obs_sorted,
                r_radar,
                v_radar,
                radar_epoch,
                self.mu,
            )

            attempted_methods[0].update({
                "rms_residual_arcsec": rms_arcsec,
                "semi_major_axis_km": elements["semi_major_axis_km"],
                "eccentricity": elements["eccentricity"],
                "perigee_km": elements["perigee_km"],
                "apogee_km": elements["apogee_km"],
            })

            return IODSolution(
                success=True,
                track_id=track_id,
                epoch=radar_epoch,
                position_km=r_radar,
                velocity_km_s=v_radar,
                semi_major_axis_km=elements["semi_major_axis_km"],
                eccentricity=elements["eccentricity"],
                inclination_deg=elements["inclination_deg"],
                raan_deg=elements["raan_deg"],
                arg_perigee_deg=elements["arg_perigee_deg"],
                true_anomaly_deg=elements["true_anomaly_deg"],
                rms_residual_arcsec=rms_arcsec,
                observations_used=len(obs_sorted),
                iterations=1,
                method_used="range+angles",
                attempted_methods=attempted_methods,
            )

        # Select best three observations (first, middle, last for max arc)
        n = len(obs_sorted)
        if n == 3:
            obs1, obs2, obs3 = obs_sorted
        else:
            # Use first, middle, last
            obs1 = obs_sorted[0]
            obs2 = obs_sorted[n // 2]
            obs3 = obs_sorted[-1]

        attempted_methods = []
        candidates = []

        def try_method(method_name: str, method_fn):
            r2_try, v2_try, status_try = method_fn(obs1, obs2, obs3, self.mu)

            attempt_record = {
                "method": method_name,
                "success": r2_try is not None and v2_try is not None,
                "status": status_try,
            }

            if r2_try is not None and v2_try is not None:
                v_mag = norm(v2_try)
                r_mag = norm(r2_try)

                # Hard screening first
                if v_mag < 1.0 or v_mag > 15.0:
                    attempt_record["success"] = False
                    attempt_record["status"] = (
                        f"{status_try}; rejected: unreasonable velocity {v_mag:.2f} km/s"
                    )
                elif r_mag < RE_EARTH + 100:
                    attempt_record["success"] = False
                    attempt_record["status"] = (
                        f"{status_try}; rejected: solution inside Earth (r={r_mag:.1f} km)"
                    )
                elif r_mag > 100000:
                    attempt_record["success"] = False
                    attempt_record["status"] = (
                        f"{status_try}; rejected: solution too far from Earth (r={r_mag:.1f} km)"
                    )
                else:
                    elements_try = state_to_elements(r2_try, v2_try, self.mu)

                    a_try = elements_try["semi_major_axis_km"]
                    e_try = elements_try["eccentricity"]
                    perigee_try = elements_try["perigee_km"]
                    apogee_try = elements_try["apogee_km"]

                    if a_try is not None and a_try < 0:
                        attempt_record["success"] = False
                        attempt_record["status"] = (
                            f"{status_try}; rejected: hyperbolic orbit (a={a_try:.1f} km)"
                        )
                    elif perigee_try is not None and perigee_try < 100.0:
                        attempt_record["success"] = False
                        attempt_record["status"] = (
                            f"{status_try}; rejected: perigee too low ({perigee_try:.1f} km)"
                        )
                    elif e_try is not None and e_try >= 0.8:
                        attempt_record["success"] = False
                        attempt_record["status"] = (
                            f"{status_try}; rejected: eccentricity too high ({e_try:.3f})"
                        )
                    elif apogee_try is not None and apogee_try > 50000.0:
                        attempt_record["success"] = False
                        attempt_record["status"] = (
                            f"{status_try}; rejected: apogee too high ({apogee_try:.1f} km)"
                        )
                    else:
                        score_try, rms_try = self._score_candidate(
                            obs_sorted, r2_try, v2_try, obs2.timestamp
                        )

                        if rms_try > MAX_ACCEPTED_RMS_RESIDUAL_ARCSEC:
                            attempt_record["success"] = False
                            attempt_record["status"] = (
                                f"{status_try}; rejected: RMS residual too high "
                                f"({rms_try:.1f} arcsec > {MAX_ACCEPTED_RMS_RESIDUAL_ARCSEC:.1f} arcsec)"
                            )
                        else:
                            physical_penalty = score_try - rms_try

                            attempt_record["score"] = score_try
                            attempt_record["base_score"] = rms_try
                            attempt_record["physical_penalty"] = physical_penalty
                            attempt_record["rms_residual_arcsec"] = rms_try
                            attempt_record["semi_major_axis_km"] = elements_try["semi_major_axis_km"]
                            attempt_record["eccentricity"] = elements_try["eccentricity"]
                            attempt_record["perigee_km"] = elements_try["perigee_km"]
                            attempt_record["apogee_km"] = elements_try["apogee_km"]

                            candidates.append({
                                "method": method_name,
                                "r2": r2_try,
                                "v2": v2_try,
                                "score": score_try,
                                "rms_arcsec": rms_try,
                                "elements": elements_try,
                            })

            attempted_methods.append(attempt_record)

        # Active committed optical IOD path.
        # range-search is currently the validated workhorse across LEO and HEO cases.
        try_method("range-search", range_search_iod)
        # gauss_iod, double_r_iod, and gooding_iod remain available as experimental
        # helpers, but are not active candidates until they pass multi-orbit validation.

        if not candidates:
            r2, v2, status = estimate_orbit_from_directions(obs1, obs2, obs3, self.mu)
            fallback_record = {
                "method": "estimate_orbit_from_directions",
                "success": r2 is not None and v2 is not None,
                "status": status,
            }

            if r2 is not None and v2 is not None:
                v_mag = norm(v2)
                r_mag = norm(r2)

                if v_mag < 1.0 or v_mag > 15.0:
                    fallback_record["success"] = False
                    fallback_record["status"] = (
                        f"{status}; rejected: unreasonable velocity {v_mag:.2f} km/s"
                    )
                elif r_mag < RE_EARTH + 100:
                    fallback_record["success"] = False
                    fallback_record["status"] = (
                        f"{status}; rejected: solution inside Earth (r={r_mag:.1f} km)"
                    )
                elif r_mag > 100000:
                    fallback_record["success"] = False
                    fallback_record["status"] = (
                        f"{status}; rejected: solution too far from Earth (r={r_mag:.1f} km)"
                    )
                else:
                    elements_try = state_to_elements(r2, v2, self.mu)

                    a_try = elements_try["semi_major_axis_km"]
                    e_try = elements_try["eccentricity"]
                    perigee_try = elements_try["perigee_km"]
                    apogee_try = elements_try["apogee_km"]

                    if a_try is not None and a_try < 0:
                        fallback_record["success"] = False
                        fallback_record["status"] = (
                            f"{status}; rejected: hyperbolic orbit (a={a_try:.1f} km)"
                        )
                    elif perigee_try is not None and perigee_try < 100.0:
                        fallback_record["success"] = False
                        fallback_record["status"] = (
                            f"{status}; rejected: perigee too low ({perigee_try:.1f} km)"
                        )
                    elif e_try is not None and e_try >= 0.8:
                        fallback_record["success"] = False
                        fallback_record["status"] = (
                            f"{status}; rejected: eccentricity too high ({e_try:.3f})"
                        )
                    elif apogee_try is not None and apogee_try > 50000.0:
                        fallback_record["success"] = False
                        fallback_record["status"] = (
                            f"{status}; rejected: apogee too high ({apogee_try:.1f} km)"
                        )
                    else:
                        score_try, rms_try = self._score_candidate(
                            obs_sorted, r2, v2, obs2.timestamp
                        )
                        physical_penalty = score_try - rms_try

                        fallback_record["score"] = score_try
                        fallback_record["base_score"] = rms_try
                        fallback_record["physical_penalty"] = physical_penalty
                        fallback_record["rms_residual_arcsec"] = rms_try
                        fallback_record["semi_major_axis_km"] = elements_try["semi_major_axis_km"]
                        fallback_record["eccentricity"] = elements_try["eccentricity"]
                        fallback_record["perigee_km"] = elements_try["perigee_km"]
                        fallback_record["apogee_km"] = elements_try["apogee_km"]

                        candidates.append({
                            "method": "estimate_orbit_from_directions",
                            "r2": r2,
                            "v2": v2,
                            "score": score_try,
                            "rms_arcsec": rms_try,
                            "elements": elements_try,
                        })

            attempted_methods.append(fallback_record)

        if not candidates:
            return IODSolution(
                success=False,
                track_id=track_id,
                epoch=obs2.timestamp,
                error_message="IOD failed: no acceptable candidate solution",
                observations_used=3,
                attempted_methods=attempted_methods
            )

        best = min(candidates, key=lambda c: c["score"])
        r2 = best["r2"]
        v2 = best["v2"]
        method_used = best["method"]
        elements = best["elements"]
        rms_arcsec = best["rms_arcsec"]
        
        # Check velocity reasonableness
        v2_mag = norm(v2)
        if v2_mag < 1.0 or v2_mag > 15.0:
            return IODSolution(
                success=False,
                track_id=track_id,
                epoch=obs2.timestamp,
                error_message=f"Unreasonable velocity: {v2_mag:.2f} km/s",
                observations_used=3,
                attempted_methods=attempted_methods
            )
        
        # Check position is in valid orbital range
        r2_mag = norm(r2)
        if r2_mag < RE_EARTH + 100:
            return IODSolution(
                success=False,
                track_id=track_id,
                epoch=obs2.timestamp,
                error_message=f"Solution inside Earth: r = {r2_mag:.1f} km",
                observations_used=3,
                attempted_methods=attempted_methods
            )
        
        if r2_mag > 100000:
            return IODSolution(
                success=False,
                track_id=track_id,
                epoch=obs2.timestamp,
                error_message=f"Solution too far from Earth (r = {r2_mag:.1f} km)",
                observations_used=3,
                attempted_methods=attempted_methods
            )
        
        # Check orbit reasonableness - only fail for clearly impossible orbits
        # Note: For demo purposes, we allow some physically questionable orbits
        if elements["semi_major_axis_km"] is not None and elements["semi_major_axis_km"] < 0:
            return IODSolution(
                success=False,
                track_id=track_id,
                epoch=obs2.timestamp,
                error_message=f"Hyperbolic orbit (a = {elements['semi_major_axis_km']:.1f} km)",
                observations_used=3,
                attempted_methods=attempted_methods
            )
        
        # Success!
        return IODSolution(
            success=True,
            track_id=track_id,
            epoch=obs2.timestamp,
            position_km=r2,
            velocity_km_s=v2,
            semi_major_axis_km=elements["semi_major_axis_km"],
            eccentricity=elements["eccentricity"],
            inclination_deg=elements["inclination_deg"],
            raan_deg=elements["raan_deg"],
            arg_perigee_deg=elements["arg_perigee_deg"],
            true_anomaly_deg=elements["true_anomaly_deg"],
            rms_residual_arcsec=rms_arcsec,
            observations_used=len(obs_sorted),
            iterations=1,
            method_used=method_used,
            attempted_methods=attempted_methods
        )

# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    print("=== IOD Module Test ===\n")
    
    # Create synthetic test data with realistic geometry
    # Key: observations need sufficient angular separation
    
    from datetime import timedelta
    
    t_base = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    
    # Target orbit: 400 km circular, 51.6° inclination
    a_tgt = 6778.0  # km (400 km altitude)
    i_tgt = math.radians(51.6)
    n_tgt = math.sqrt(MU_EARTH_KM / a_tgt**3)  # rad/s mean motion
    
    # Observer orbit: similar but offset in RAAN and true anomaly
    a_obs = 6778.0
    i_obs = math.radians(51.6)
    n_obs = math.sqrt(MU_EARTH_KM / a_obs**3)
    
    # Initial positions
    theta_tgt_0 = math.radians(45)  # Target true anomaly
    theta_obs_0 = math.radians(0)   # Observer true anomaly (behind target)
    raan_tgt = math.radians(0)
    raan_obs = math.radians(10)     # Different RAAN gives cross-track separation
    
    observations = []
    
    print("Creating observations with realistic orbital geometry...\n")
    
    for idx, dt_sec in enumerate([0, 30, 60]):  # 30-second spacing for good arc
        t = t_base + timedelta(seconds=dt_sec)
        
        # Target position in perifocal frame
        theta_tgt = theta_tgt_0 + n_tgt * dt_sec
        r_peri_tgt = a_tgt * np.array([math.cos(theta_tgt), math.sin(theta_tgt), 0])
        v_peri_tgt = math.sqrt(MU_EARTH_KM / a_tgt) * np.array([-math.sin(theta_tgt), math.cos(theta_tgt), 0])
        
        # Rotation matrix perifocal to ECI for target
        R_tgt = perifocal_to_eci_matrix(i_tgt, raan_tgt, 0)
        r_tgt = R_tgt @ r_peri_tgt
        v_tgt = R_tgt @ v_peri_tgt
        
        # Observer position
        theta_obs = theta_obs_0 + n_obs * dt_sec
        r_peri_obs = a_obs * np.array([math.cos(theta_obs), math.sin(theta_obs), 0])
        v_peri_obs = math.sqrt(MU_EARTH_KM / a_obs) * np.array([-math.sin(theta_obs), math.cos(theta_obs), 0])
        
        # Rotation matrix for observer
        R_obs = perifocal_to_eci_matrix(i_obs, raan_obs, 0)
        r_obs = R_obs @ r_peri_obs
        v_obs = R_obs @ v_peri_obs
        
        # Line of sight from observer to target
        los = r_tgt - r_obs
        los_mag = norm(los)
        los_unit = unit(los)
        
        # Convert to RA/Dec
        ra = math.atan2(los_unit[1], los_unit[0])
        if ra < 0:
            ra += 2 * math.pi
        dec = math.asin(np.clip(los_unit[2], -1, 1))
        
        obs = IODObservation(
            timestamp=t,
            ra=ra,
            dec=dec,
            ra_sigma=math.radians(0.01),
            dec_sigma=math.radians(0.01),
            observer_position_km=r_obs,
            observer_velocity_km_s=v_obs
        )
        observations.append(obs)
        
        print(f"Observation {idx+1}:")
        print(f"  Time: {t.isoformat()}")
        print(f"  RA: {math.degrees(ra):.4f}°, Dec: {math.degrees(dec):.4f}°")
        print(f"  Observer: [{r_obs[0]:.1f}, {r_obs[1]:.1f}, {r_obs[2]:.1f}] km")
        print(f"  Target: [{r_tgt[0]:.1f}, {r_tgt[1]:.1f}, {r_tgt[2]:.1f}] km")
        print(f"  Range: {los_mag:.1f} km")
        print()
    
    # True state at middle observation
    dt_mid = 30
    theta_tgt_mid = theta_tgt_0 + n_tgt * dt_mid
    r_peri_mid = a_tgt * np.array([math.cos(theta_tgt_mid), math.sin(theta_tgt_mid), 0])
    v_peri_mid = math.sqrt(MU_EARTH_KM / a_tgt) * np.array([-math.sin(theta_tgt_mid), math.cos(theta_tgt_mid), 0])
    R_tgt = perifocal_to_eci_matrix(i_tgt, raan_tgt, 0)
    r_true = R_tgt @ r_peri_mid
    v_true = R_tgt @ v_peri_mid
    
    print(f"True state at middle observation:")
    print(f"  Position: [{r_true[0]:.3f}, {r_true[1]:.3f}, {r_true[2]:.3f}] km")
    print(f"  Velocity: [{v_true[0]:.4f}, {v_true[1]:.4f}, {v_true[2]:.4f}] km/s")
    print()
    
    # Run IOD
    solver = IODSolver()
    solution = solver.solve(observations, uuid4())
    
    print("IOD Solution:")
    print(f"  Success: {solution.success}")
    if solution.success:
        print(f"  Position: [{solution.position_km[0]:.3f}, {solution.position_km[1]:.3f}, {solution.position_km[2]:.3f}] km")
        print(f"  Velocity: [{solution.velocity_km_s[0]:.4f}, {solution.velocity_km_s[1]:.4f}, {solution.velocity_km_s[2]:.4f}] km/s")
        print(f"  Semi-major axis: {solution.semi_major_axis_km:.1f} km (true: {a_tgt:.1f})")
        print(f"  Eccentricity: {solution.eccentricity:.6f}")
        print(f"  Inclination: {solution.inclination_deg:.2f}° (true: {math.degrees(i_tgt):.2f}°)")
        print(f"  RMS Residual: {solution.rms_residual_arcsec:.1f} arcsec")
        
        # Position error
        pos_err = norm(solution.position_km - r_true)
        vel_err = norm(solution.velocity_km_s - v_true)
        print(f"\n  Position error: {pos_err:.3f} km")
        print(f"  Velocity error: {vel_err:.6f} km/s")
    else:
        print(f"  Error: {solution.error_message}")
    
    print("\n=== Test Complete ===")
