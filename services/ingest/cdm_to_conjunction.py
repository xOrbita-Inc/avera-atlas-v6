"""Convert a parsed CDM dict into a ConjunctionState matching the planner OpenAPI schema."""

from __future__ import annotations

from typing import Any

import numpy as np


def _build_rtn_covariance(cdm: dict[str, Any], prefix: str) -> np.ndarray:
    """Reconstruct a 3×3 symmetric RTN position covariance (m²) for one object.

    The CDM upper-triangle labels map to::

        [[CR_R,  CT_R,  CN_R],
         [CT_R,  CT_T,  CN_T],
         [CN_R,  CN_T,  CN_N]]
    """
    cr_r = cdm.get(f"{prefix}_CR_R", 0.0)
    ct_r = cdm.get(f"{prefix}_CT_R", 0.0)
    ct_t = cdm.get(f"{prefix}_CT_T", 0.0)
    cn_r = cdm.get(f"{prefix}_CN_R", 0.0)
    cn_t = cdm.get(f"{prefix}_CN_T", 0.0)
    cn_n = cdm.get(f"{prefix}_CN_N", 0.0)

    return np.array(
        [
            [cr_r, ct_r, cn_r],
            [ct_r, ct_t, cn_t],
            [cn_r, cn_t, cn_n],
        ],
        dtype=np.float64,
    )


def _rtn_to_eci_rotation(r_km: np.ndarray, v_km_s: np.ndarray) -> np.ndarray:
    """Build the 3×3 RTN→ECI rotation matrix from an ECI state vector.

    Columns are the R, T, N unit vectors expressed in ECI:
      R = r̂  (radial)
      N = (r × v) / |r × v|  (cross-track / normal)
      T = N × R  (along-track / tangential)
    """
    r_hat = r_km / np.linalg.norm(r_km)
    h = np.cross(r_km, v_km_s)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)
    return np.column_stack([r_hat, t_hat, n_hat])


def _iso_tca(tca_raw: Any) -> str:
    """Normalise the TCA string to ISO-8601 with a Z suffix."""
    s = str(tca_raw).strip()
    # CCSDS day-of-year format "2010-097T04:42:19.315" — pass through as-is
    # but ensure it ends with Z for UTC.
    if not s.endswith("Z"):
        s += "Z"
    return s


def cdm_to_conjunction_state(cdm: dict[str, Any]) -> dict[str, Any]:
    """Convert a parsed CDM dict into a ConjunctionState dict.

    Returns a dict with keys:
      obj_id, t_ca_utc, r_rel_km, p_rel_km2, pc_precomputed
    matching the planner's OpenAPI v2.4.1 schema.
    """

    # ── Object IDs ────────────────────────────────────────────────
    raw_id = cdm.get("OBJECT2_OBJECT_DESIGNATOR", cdm.get("OBJECT2_OBJECT_NAME", "UNKNOWN"))
    # Designators may have been parsed as float; convert back to clean string.
    obj_id = str(int(raw_id)) if isinstance(raw_id, float) and raw_id == int(raw_id) else str(raw_id)

    # ── TCA ───────────────────────────────────────────────────────
    t_ca_utc = _iso_tca(cdm["TCA"])

    # ── Satellite (OBJECT1) ECI state ─────────────────────────────
    r1 = np.array(
        [cdm["OBJECT1_X"], cdm["OBJECT1_Y"], cdm["OBJECT1_Z"]], dtype=np.float64
    )  # km
    v1 = np.array(
        [cdm["OBJECT1_X_DOT"], cdm["OBJECT1_Y_DOT"], cdm["OBJECT1_Z_DOT"]],
        dtype=np.float64,
    )  # km/s

    # ── Rotation matrix RTN→ECI (based on OBJECT1 state) ─────────
    rot = _rtn_to_eci_rotation(r1, v1)

    # ── Relative position (ECI, km) ──────────────────────────────
    rel_r = cdm.get("RELATIVE_POSITION_R")
    if rel_r is not None:
        # CDM relative position is in RTN, metres
        dr_rtn_m = np.array(
            [
                cdm["RELATIVE_POSITION_R"],
                cdm["RELATIVE_POSITION_T"],
                cdm["RELATIVE_POSITION_N"],
            ],
            dtype=np.float64,
        )
        r_rel_km = (rot @ dr_rtn_m) / 1000.0  # m → km
    else:
        # Derive from state vectors
        r2 = np.array(
            [cdm["OBJECT2_X"], cdm["OBJECT2_Y"], cdm["OBJECT2_Z"]], dtype=np.float64
        )
        r_rel_km = r2 - r1  # already km

    # ── Covariance ────────────────────────────────────────────────
    # Step A: per-object 3×3 RTN covariance (m²)
    p1 = _build_rtn_covariance(cdm, "OBJECT1")
    p2 = _build_rtn_covariance(cdm, "OBJECT2")

    # Step B: combined relative covariance (independence assumed)
    p_rel_rtn = p1 + p2  # m²

    # Step C–D: rotate to ECI
    p_rel_eci_m2 = rot @ p_rel_rtn @ rot.T

    # Step E: m² → km²
    p_rel_eci_km2 = p_rel_eci_m2 / 1e6

    # Step F: flatten row-major
    p_rel_km2 = p_rel_eci_km2.flatten().tolist()  # 9 elements

    # ── Pc ────────────────────────────────────────────────────────
    pc_raw = cdm.get("COLLISION_PROBABILITY")
    pc_precomputed: float | None = float(pc_raw) if pc_raw is not None else None

    return {
        "obj_id": obj_id,
        "t_ca_utc": t_ca_utc,
        "r_rel_km": r_rel_km.tolist(),
        "p_rel_km2": p_rel_km2,
        "pc_precomputed": pc_precomputed,
    }
