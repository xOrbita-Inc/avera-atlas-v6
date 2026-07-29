"""
pc_utils.py - Probability of Collision Calculation Utilities

This module provides NASA-standard Pc calculation methods for conjunction assessment:
- pc_circle: Standard 2D Pc calculation with multiple estimation modes
- frisbee_max_pc: Maximum Pc when one object's covariance is unknown

Integrated into AVERA-ATLAS planner-stack for collision risk assessment.

Based on:
- Alfano, S. 2005. "A Numerical Implementation of Spherical Object Collision Probability"
- NASA Conjunction Assessment algorithms

Original MATLAB: T. Lechtenberg, L. Baars, D. Hall, S. Es haghi (NASA)
Python conversion: Claude
"""

import numpy as np
from scipy import integrate
from scipy.special import erf
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings


# =============================================================================
# Output Data Structure
# =============================================================================

@dataclass
class PcResult:
    """Result structure for Pc calculations."""
    Pc: float                    # Probability of collision
    miss_distance: float         # Miss distance in meters
    sigma_x: float = None        # Sigma along major axis
    sigma_z: float = None        # Sigma along minor axis
    is_remediated: bool = False  # Whether covariance was remediated


# =============================================================================
# Module-level cache for Gauss-Chebyshev quadrature
# =============================================================================

_gc_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


# =============================================================================
# Main API Functions
# =============================================================================

def compute_pc(
    r1: np.ndarray,
    v1: np.ndarray,
    cov1: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    cov2: np.ndarray,
    hbr: float,
    estimation_mode: int = 64
) -> PcResult:
    """
    Compute probability of collision for a conjunction event.
    
    This is the main API for single conjunction Pc calculation.
    
    Parameters
    ----------
    r1 : np.ndarray
        Primary object position in ECI [meters], shape (3,)
    v1 : np.ndarray
        Primary object velocity in ECI [m/s], shape (3,)
    cov1 : np.ndarray
        Primary object covariance in ECI [m^2], shape (3,3) or (6,6)
    r2 : np.ndarray
        Secondary object position in ECI [meters], shape (3,)
    v2 : np.ndarray
        Secondary object velocity in ECI [m/s], shape (3,)
    cov2 : np.ndarray
        Secondary object covariance in ECI [m^2], shape (3,3) or (6,6)
    hbr : float
        Combined hard body radius [meters]
    estimation_mode : int
        Pc estimation mode:
        -1 = Circumscribing square upper bound (fast)
         0 = Equal-area square approximation
        64 = Gauss-Chebyshev quadrature order 64 (default, recommended)
         1 = Full numerical integration (slowest, most accurate)
    
    Returns
    -------
    PcResult
        Dataclass with Pc, miss_distance, sigma values, and flags
    """
    # Ensure proper array shapes
    r1 = np.asarray(r1, dtype=float).flatten()[:3]
    v1 = np.asarray(v1, dtype=float).flatten()[:3]
    r2 = np.asarray(r2, dtype=float).flatten()[:3]
    v2 = np.asarray(v2, dtype=float).flatten()[:3]
    
    # Extract 3x3 position covariance if 6x6 provided
    cov1 = np.asarray(cov1, dtype=float)
    cov2 = np.asarray(cov2, dtype=float)
    if cov1.shape[0] > 3:
        cov1 = cov1[:3, :3]
    if cov2.shape[0] > 3:
        cov2 = cov2[:3, :3]
    
    # Check if either covariance is missing/zero
    cov1_zero = np.allclose(cov1, 0)
    cov2_zero = np.allclose(cov2, 0)
    
    if cov1_zero or cov2_zero:
        # Use Frisbee max Pc method
        Pc = frisbee_max_pc(r1, v1, cov1, r2, v2, cov2, hbr)
        miss_distance = np.linalg.norm(r1 - r2)
        return PcResult(Pc=Pc, miss_distance=miss_distance, is_remediated=True)
    
    # Standard Pc calculation
    params = {'EstimationMode': estimation_mode}
    Pc_arr, out = pc_circle(
        r1.reshape(1, 3), v1.reshape(1, 3), cov1,
        r2.reshape(1, 3), v2.reshape(1, 3), cov2,
        hbr, params
    )
    
    return PcResult(
        Pc=float(Pc_arr[0]),
        miss_distance=float(np.sqrt(out.xm[0]**2 + out.zm[0]**2)) if out.xm is not None else np.linalg.norm(r1-r2),
        sigma_x=float(out.sx[0]) if out.sx is not None else None,
        sigma_z=float(out.sz[0]) if out.sz is not None else None,
        is_remediated=bool(out.IsRemediated[0]) if out.IsRemediated is not None else False
    )


def compute_pc_batch(
    r1_batch: np.ndarray,
    v1_batch: np.ndarray,
    cov1: np.ndarray,
    r2_batch: np.ndarray,
    v2_batch: np.ndarray,
    cov2_batch: np.ndarray,
    hbr: float,
    estimation_mode: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Pc for multiple conjunctions (vectorized).
    
    Parameters
    ----------
    r1_batch : np.ndarray
        Primary positions, shape (n, 3) [meters]
    v1_batch : np.ndarray
        Primary velocities, shape (n, 3) or (1, 3) [m/s]
    cov1 : np.ndarray
        Primary covariance, shape (3, 3) - same for all [m^2]
    r2_batch : np.ndarray
        Secondary positions, shape (n, 3) [meters]
    v2_batch : np.ndarray
        Secondary velocities, shape (n, 3) or (1, 3) [m/s]
    cov2_batch : np.ndarray
        Secondary covariances, shape (3, 3) or (n, 3, 3) [m^2]
    hbr : float
        Combined hard body radius [meters]
    estimation_mode : int
        Pc estimation mode (default: 64)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (Pc_values, miss_distances) arrays of shape (n,)
    """
    params = {'EstimationMode': estimation_mode}
    
    Pc_arr, out = pc_circle(
        r1_batch, v1_batch, cov1,
        r2_batch, v2_batch, cov2_batch,
        hbr, params
    )
    
    miss_distances = np.sqrt(out.xm**2 + out.zm**2) if out.xm is not None else \
                     np.linalg.norm(r1_batch - r2_batch, axis=1)
    
    return Pc_arr, miss_distances


def default_covariance_from_uncertainty(
    position_uncertainty_m: float = 1000.0,
    velocity_uncertainty_ms: float = 1.0,
    cross_track_factor: float = 0.1
) -> np.ndarray:
    """
    Generate a default diagonal covariance matrix from uncertainty estimates.
    
    Parameters
    ----------
    position_uncertainty_m : float
        1-sigma position uncertainty in along-track direction [meters]
    velocity_uncertainty_ms : float
        1-sigma velocity uncertainty [m/s] (not used in 3x3 output)
    cross_track_factor : float
        Factor for cross-track uncertainty relative to along-track
    
    Returns
    -------
    np.ndarray
        3x3 position covariance matrix [m^2]
    """
    sigma_along = position_uncertainty_m
    sigma_cross = position_uncertainty_m * cross_track_factor
    
    return np.diag([sigma_along**2, sigma_cross**2, sigma_cross**2])


# =============================================================================
# PcCircle Implementation
# =============================================================================

@dataclass
class PcCircleOutput:
    """Output structure for PcCircle function."""
    IsPosDef: np.ndarray = None
    IsRemediated: np.ndarray = None
    Amat: np.ndarray = None
    xm: np.ndarray = None
    zm: np.ndarray = None
    sx: np.ndarray = None
    sz: np.ndarray = None
    xhat: np.ndarray = None
    yhat: np.ndarray = None
    zhat: np.ndarray = None
    EigV1: np.ndarray = None
    EigV2: np.ndarray = None
    EigL1: np.ndarray = None
    EigL2: np.ndarray = None
    ClipBoundSet: np.ndarray = None


def pc_circle(
    r1: np.ndarray,
    v1: np.ndarray,
    cov1: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    cov2: np.ndarray,
    hbr: float,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, PcCircleOutput]:
    """
    Compute Pc by integrating over a circle on the conjunction plane.
    
    See module docstring for full documentation.
    """
    if params is None:
        params = {}
    
    estimation_mode = params.get('EstimationMode', 64)
    warning_level = params.get('WarningLevel', 0)
    
    # Validate EstimationMode
    if estimation_mode <= 0 and estimation_mode not in [0, -1]:
        raise ValueError("Invalid EstimationMode")
    
    # Reformat inputs
    Nvec, r1, v1 = _check_and_resize_pos_vel(r1, v1)
    Nvec2, r2, v2 = _check_and_resize_pos_vel(r2, v2)
    
    if Nvec != Nvec2:
        raise ValueError("Number of primary and secondary positions must be equal")
    
    cov1 = _check_and_resize_cov(Nvec, cov1)
    cov2 = _check_and_resize_cov(Nvec2, cov2)
    
    # Handle HBR
    hbr = np.atleast_1d(np.asarray(hbr, dtype=float))
    if hbr.size == 1 and Nvec > 1:
        hbr = np.full(Nvec, hbr[0])
    hbr = np.maximum(hbr, 0)
    
    out = PcCircleOutput()
    
    # Combine covariances
    comb_cov = cov1 + cov2
    
    # Relative position and velocity
    r = r1 - r2
    v = v1 - v2
    
    # Handle zero/small miss distance
    rmag = np.sqrt(np.sum(r**2, axis=1))
    reps = np.maximum(10 * np.finfo(float).eps * rmag, 1e-6 * hbr)
    small_rmag = rmag < reps
    
    if np.any(small_rmag):
        rsum = r1[small_rmag] + r2[small_rmag]
        rsummag = np.sqrt(np.sum(rsum**2, axis=1))
        vmag_small = np.sqrt(np.sum(v[small_rmag]**2, axis=1))
        safe_denom = np.maximum(rsummag * vmag_small, 1e-30)
        rdel = reps[small_rmag, np.newaxis] * np.cross(rsum, v[small_rmag]) / safe_denom[:, np.newaxis]
        r[small_rmag] = r[small_rmag] + rdel
    
    # Check for zero relative velocity
    vmag = np.sqrt(np.sum(v**2, axis=1))
    zero_vmag = vmag == 0
    
    # Orbit normal
    h = np.cross(r, v)
    
    # Relative encounter frame
    vmag_safe = np.where(vmag > 0, vmag, 1)[:, np.newaxis]
    y = v / vmag_safe
    
    hmag = np.sqrt(np.sum(h**2, axis=1))
    hmag_safe = np.where(hmag > 0, hmag, 1)[:, np.newaxis]
    z = h / hmag_safe
    
    x = np.cross(y, z)
    
    eci2xyz = np.column_stack([x, y, z])
    out.xhat = x
    out.yhat = y
    out.zhat = z
    
    eci2xyz_T = eci2xyz[:, [0, 3, 6, 1, 4, 7, 2, 5, 8]]
    
    # Project combined covariance onto conjunction plane
    rotated_cov = _product_3x3(eci2xyz, _product_3x3(comb_cov, eci2xyz_T))
    Amat = rotated_cov[:, [0, 2, 8]]
    out.Amat = Amat
    
    # Eigendecomposition
    V1, V2, L1, L2 = _eig2x2(Amat)
    out.EigV1 = V1
    out.EigL1 = L1
    out.EigV2 = V2
    out.EigL2 = L2
    
    if np.any(L1 <= 0):
        raise ValueError("Invalid case(s) with two non-positive eigenvalues")
    
    # Eigenvalue clipping
    finite_hbr = ~np.isinf(hbr)
    Fclip = 1e-4
    Lrem = (Fclip * hbr) ** 2
    
    is_rem1 = (L1 < Lrem) & finite_hbr
    L1 = np.where(is_rem1, Lrem, L1)
    is_rem2 = (L2 < Lrem) & finite_hbr
    L2 = np.where(is_rem2, Lrem, L2)
    
    out.IsPosDef = L2 > 0
    out.IsRemediated = is_rem1 | is_rem2
    
    # Sigma values
    sx = np.sqrt(L1)
    sz = np.sqrt(L2)
    out.sx = sx
    out.sz = sz
    
    # Miss distance in conjunction plane
    rm = np.sqrt(np.sum(r**2, axis=1))
    xm = rm * np.abs(V1[:, 0])
    zm = rm * np.abs(V1[:, 1])
    out.xm = xm
    out.zm = zm
    
    # Estimate Pc
    if estimation_mode <= 0:
        if estimation_mode == 0:
            HSQ = np.sqrt(np.pi / 4) * hbr
        else:
            HSQ = hbr
        
        sqrt2 = np.sqrt(2)
        dx = sqrt2 * sx
        dz = sqrt2 * sz
        
        Ex = _erf_vec_dif((xm + HSQ) / dx, (xm - HSQ) / dx)
        Ez = _erf_vec_dif((zm + HSQ) / dz, (zm - HSQ) / dz)
        Pc = Ex * Ez / 4
    else:
        xlo = xm - hbr
        xhi = xm + hbr
        
        Nsx = 4 * sx
        xloclip = np.where(xlo < 0, 0, xlo)
        out.ClipBoundSet = ~((xlo > -Nsx) & (xhi < xloclip + Nsx))
        
        if estimation_mode == 1:
            Iset = np.zeros(Nvec, dtype=bool)
        else:
            Iset = ~out.ClipBoundSet
        
        Pc = np.full(Nvec, np.nan)
        sqrt2 = np.sqrt(2)
        dx = sqrt2 * sx
        dz = sqrt2 * sz
        
        # Gauss-Chebyshev quadrature
        if np.any(Iset):
            if estimation_mode not in _gc_cache:
                xGC, yGC, wGC = _gen_gc_quad(estimation_mode)
                _gc_cache[estimation_mode] = (xGC, yGC, wGC)
            else:
                xGC, yGC, wGC = _gc_cache[estimation_mode]
            
            NGC = estimation_mode
            Nset = np.sum(Iset)
            
            zmrep = np.tile(zm[Iset][:, np.newaxis], (1, NGC))
            dzrep = np.tile(dz[Iset][:, np.newaxis], (1, NGC))
            Hrep = np.tile(hbr[Iset][:, np.newaxis], (1, NGC))
            xrep = np.tile(xm[Iset][:, np.newaxis], (1, NGC)) + Hrep * xGC
            Hxrep = Hrep * yGC
            
            Fint = (np.exp(-(xrep / np.tile(dx[Iset][:, np.newaxis], (1, NGC)))**2) *
                   _erf_vec_dif((zmrep + Hxrep) / dzrep, (zmrep - Hxrep) / dzrep))
            
            Psum = np.sum(wGC * Fint, axis=1)
            Pc[Iset] = (hbr[Iset] / sx[Iset]) * Psum
        
        # Numerical integration for remaining cases
        Iset_remaining = ~Iset & ~zero_vmag
        if np.any(Iset_remaining):
            xlo_adj = xlo.copy()
            xhi_adj = xhi.copy()
            
            neg_set = Iset_remaining & (xlo < 0)
            if np.any(neg_set):
                Nsx_clip = 5 * sx
                xlo_adj[neg_set & (xlo < -Nsx_clip)] = -Nsx_clip[neg_set & (xlo < -Nsx_clip)]
                xhi_adj[neg_set & (xhi > Nsx_clip)] = Nsx_clip[neg_set & (xhi > Nsx_clip)]
            
            HBR2 = hbr[Iset_remaining] ** 2
            indices = np.where(Iset_remaining)[0]
            
            for kk, k in enumerate(indices):
                def integrand(xx):
                    return _pc_2d_integrand(xx, xm[k], zm[k], dx[k], dz[k], HBR2[kk])
                
                result, _ = integrate.quad(integrand, xlo_adj[k], xhi_adj[k],
                                          epsabs=1e-300, epsrel=1e-6)
                Pc[k] = result
            
            Pc[Iset_remaining] = (Pc[Iset_remaining] / sx[Iset_remaining]) / np.sqrt(8 * np.pi)
    
    # Special cases
    Pc[~finite_hbr] = 1.0
    Pc[hbr == 0] = 0.0
    Pc[zero_vmag] = np.nan
    
    return Pc, out


# =============================================================================
# FrisbeeMaxPc Implementation
# =============================================================================

def frisbee_max_pc(
    r1: np.ndarray,
    v1: np.ndarray,
    cov1: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    cov2: np.ndarray,
    hbr: float,
    hbr_type: str = 'circle'
) -> float:
    """
    Calculate maximum possible Pc when one covariance is unknown/zero.
    
    Uses Frisbee's method to estimate upper bound Pc.
    """
    estimation_mode = _hbr_type_to_est_mode(hbr_type)
    params = {'EstimationMode': estimation_mode}
    
    r1 = np.atleast_2d(np.asarray(r1, dtype=float))
    v1 = np.atleast_2d(np.asarray(v1, dtype=float))
    r2 = np.atleast_2d(np.asarray(r2, dtype=float))
    v2 = np.atleast_2d(np.asarray(v2, dtype=float))
    
    if cov1 is not None and np.asarray(cov1).size > 0:
        cov1 = np.asarray(cov1, dtype=float)
        if cov1.ndim == 2 and cov1.shape[0] >= 3:
            cov1 = cov1[:3, :3].copy()
    else:
        cov1 = np.zeros((3, 3))
        
    if cov2 is not None and np.asarray(cov2).size > 0:
        cov2 = np.asarray(cov2, dtype=float)
        if cov2.ndim == 2 and cov2.shape[0] >= 3:
            cov2 = cov2[:3, :3].copy()
    else:
        cov2 = np.zeros((3, 3))
    
    num_r1, r1, v1 = _check_and_resize_pos_vel(r1, v1)
    num_r2, r2, v2 = _check_and_resize_pos_vel(r2, v2)
    
    cov1_flat = _check_and_resize_cov(num_r1, cov1)
    cov2_flat = _check_and_resize_cov(num_r2, cov2)
    
    # Relative encounter frame
    r = r1 - r2
    v = v1 - v2
    h = np.cross(r, v)
    
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    y = v / v_norm
    h_norm = np.linalg.norm(h, axis=1, keepdims=True)
    z = h / h_norm
    x = np.cross(y, z)
    
    eci2xyz = np.column_stack([x, y, z])
    eci2xyz_T = eci2xyz[:, [0, 3, 6, 1, 4, 7, 2, 5, 8]]
    
    first_part1 = _product_3x3(cov1_flat, eci2xyz_T)
    covcombxyz1 = _product_3x3(eci2xyz, first_part1)
    first_part2 = _product_3x3(cov2_flat, eci2xyz_T)
    covcombxyz2 = _product_3x3(eci2xyz, first_part2)
    
    Cp1 = covcombxyz1[:, [0, 2, 6, 8]]
    Cp2 = covcombxyz2[:, [0, 2, 6, 8]]
    
    C1 = _inv_2x2(Cp1)
    C2 = _inv_2x2(Cp2)
    
    x0 = np.linalg.norm(r, axis=1)
    z0 = np.zeros_like(x0)
    rrel = np.column_stack([x0, z0])
    rrel_norm = np.linalg.norm(rrel, axis=1, keepdims=True)
    urel = rrel / rrel_norm
    
    Ka = _determine_ka(rrel, C2, C1)
    
    j = np.argmax(Ka, axis=1)
    cov1_sum = np.sum(cov1_flat, axis=1)
    cov2_sum = np.sum(cov2_flat, axis=1)
    j[cov1_sum == 0] = 0
    j[cov2_sum == 0] = 1
    
    Ka_j = Ka[np.arange(len(j)), j]
    
    cov1_out = cov1_flat.copy()
    cov2_out = cov2_flat.copy()
    
    mask_ka_gt_1 = Ka_j > 1
    
    if np.any(mask_ka_gt_1):
        rrel_norm_sq = np.sum(rrel ** 2, axis=1)
        Vc = rrel_norm_sq * ((Ka_j ** 2 - 1) / Ka_j ** 2)
        
        covnew = Vc[:, np.newaxis] * np.column_stack([
            urel[:, 0] ** 2,
            urel[:, 0] * urel[:, 1],
            urel[:, 0] * urel[:, 1],
            urel[:, 1] ** 2
        ])
        
        zeros = np.zeros(len(x0))
        
        mask_j0 = (j == 0) & mask_ka_gt_1
        if np.any(mask_j0):
            covnew_3x3 = np.column_stack([
                covnew[mask_j0, 0], zeros[mask_j0], covnew[mask_j0, 1],
                zeros[mask_j0], zeros[mask_j0], zeros[mask_j0],
                covnew[mask_j0, 2], zeros[mask_j0], covnew[mask_j0, 3]
            ])
            cov1_out[mask_j0] = _product_3x3(
                eci2xyz_T[mask_j0],
                _product_3x3(covnew_3x3, eci2xyz[mask_j0])
            )
        
        mask_j1 = (j == 1) & mask_ka_gt_1
        if np.any(mask_j1):
            covnew_3x3 = np.column_stack([
                covnew[mask_j1, 0], zeros[mask_j1], covnew[mask_j1, 1],
                zeros[mask_j1], zeros[mask_j1], zeros[mask_j1],
                covnew[mask_j1, 2], zeros[mask_j1], covnew[mask_j1, 3]
            ])
            cov2_out[mask_j1] = _product_3x3(
                eci2xyz_T[mask_j1],
                _product_3x3(covnew_3x3, eci2xyz[mask_j1])
            )
    
    cov1_3x3 = cov1_out.reshape(-1, 3, 3)
    cov2_3x3 = cov2_out.reshape(-1, 3, 3)
    
    if num_r1 == 1:
        cov1_input = cov1_3x3[0]
        cov2_input = cov2_3x3[0]
    else:
        cov1_input = cov1_3x3
        cov2_input = cov2_3x3
    
    Pc2D, _ = pc_circle(r1, v1, cov1_input, r2, v2, cov2_input, hbr, params)
    
    return float(Pc2D[0]) if Pc2D.size == 1 else Pc2D


# =============================================================================
# Helper Functions
# =============================================================================

def _erf_vec_dif(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute erf(a) - erf(b)."""
    return erf(a) - erf(b)


def _eig2x2(Amat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 2x2 eigenvalue decomposition."""
    Amat = np.atleast_2d(Amat)
    n = Amat.shape[0]
    
    a = Amat[:, 0]
    b = Amat[:, 1]
    c = Amat[:, 2]
    
    trace = a + c
    det = a * c - b * b
    disc = np.sqrt(np.maximum((trace / 2)**2 - det, 0))
    
    L1 = trace / 2 + disc
    L2 = trace / 2 - disc
    
    V1 = np.zeros((n, 2))
    V2 = np.zeros((n, 2))
    
    mask_b = np.abs(b) > 1e-30
    
    if np.any(mask_b):
        v1x = b[mask_b]
        v1y = L1[mask_b] - a[mask_b]
        norm1 = np.sqrt(v1x**2 + v1y**2)
        norm1 = np.where(norm1 > 0, norm1, 1)
        V1[mask_b, 0] = v1x / norm1
        V1[mask_b, 1] = v1y / norm1
        V2[mask_b, 0] = -V1[mask_b, 1]
        V2[mask_b, 1] = V1[mask_b, 0]
    
    mask_diag = ~mask_b
    if np.any(mask_diag):
        V1[mask_diag & (a >= c), 0] = 1
        V1[mask_diag & (a >= c), 1] = 0
        V1[mask_diag & (a < c), 0] = 0
        V1[mask_diag & (a < c), 1] = 1
        V2[mask_diag & (a >= c), 0] = 0
        V2[mask_diag & (a >= c), 1] = 1
        V2[mask_diag & (a < c), 0] = 1
        V2[mask_diag & (a < c), 1] = 0
    
    return V1, V2, L1, L2


def _gen_gc_quad(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Gauss-Chebyshev quadrature points and weights."""
    k = np.arange(1, n + 1)
    theta = k * np.pi / (n + 1)
    xGC = np.cos(theta)
    yGC = np.sin(theta)
    wGC = (np.pi / (n + 1)) * yGC**2
    return xGC, yGC, wGC


def _product_3x3(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply 3x3 matrices in Nx9 format."""
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    A_3d = A.reshape(-1, 3, 3)
    B_3d = B.reshape(-1, 3, 3)
    C_3d = np.einsum('nij,njk->nik', A_3d, B_3d)
    return C_3d.reshape(-1, 9)


def _inv_2x2(mat: np.ndarray) -> np.ndarray:
    """Invert 2x2 matrices in Nx4 format."""
    mat = np.atleast_2d(mat)
    det = mat[:, 0] * mat[:, 3] - mat[:, 1] * mat[:, 2]
    det = np.where(np.abs(det) < 1e-30, 1e-30, det)
    return np.column_stack([
        mat[:, 3] / det, -mat[:, 1] / det,
        -mat[:, 2] / det, mat[:, 0] / det
    ])


def _determine_ka(rrel, C2, C1):
    """Determine Ka vectors."""
    Ka1 = np.sqrt(
        rrel[:, 0] * (rrel[:, 0] * C2[:, 0] + rrel[:, 1] * C2[:, 1]) +
        rrel[:, 1] * (rrel[:, 0] * C2[:, 2] + rrel[:, 1] * C2[:, 3])
    )
    Ka2 = np.sqrt(
        rrel[:, 0] * (rrel[:, 0] * C1[:, 0] + rrel[:, 1] * C1[:, 1]) +
        rrel[:, 1] * (rrel[:, 0] * C1[:, 2] + rrel[:, 1] * C1[:, 3])
    )
    return np.column_stack([Ka1, Ka2])


def _hbr_type_to_est_mode(hbr_type: str) -> int:
    """Convert HBR type to estimation mode."""
    hbr_type_lower = hbr_type.lower()
    if hbr_type_lower == 'circle':
        return 64
    elif hbr_type_lower == 'square':
        return -1
    elif hbr_type_lower == 'squareequarea':
        return 0
    else:
        raise ValueError(f"Incorrect HBRType: {hbr_type}")


def _pc_2d_integrand(x, xm, zm, dx, dz, R2):
    """Integrand for numerical Pc integration."""
    x = np.atleast_1d(x)
    Rx = np.sqrt(np.maximum(R2 - (x - xm)**2, 0))
    integrand = np.exp(-(x / dx)**2) * _erf_vec_dif((zm + Rx) / dz, (zm - Rx) / dz)
    return integrand[0] if integrand.size == 1 else integrand


def _check_and_resize_pos_vel(r, v):
    """Reshape position/velocity arrays."""
    r = np.atleast_2d(np.asarray(r, dtype=float))
    v = np.atleast_2d(np.asarray(v, dtype=float))
    if r.shape[1] != 3:
        r = r.T
    if v.shape[1] != 3:
        v = v.T
    num = r.shape[0]
    if v.shape[0] == 1 and num > 1:
        v = np.tile(v, (num, 1))
    return num, r, v


def _check_and_resize_cov(num, cov):
    """Reshape covariance to Nx9 format."""
    cov = np.asarray(cov, dtype=float)
    if cov.size == 0:
        return np.zeros((num, 9))
    if cov.shape == (3, 3):
        return np.tile(cov.flatten(order='F'), (num, 1))
    if cov.shape == (6, 6):
        return np.tile(cov[:3, :3].flatten(order='F'), (num, 1))
    if cov.ndim == 2 and cov.shape[1] == 9:
        return np.tile(cov, (num, 1)) if cov.shape[0] == 1 else cov
    return cov.reshape(-1, 9)
