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
      obj_id, t_ca_utc, r_rel_km, v_rel_km_s, v_rel_source, p_rel_km2,
      pc_precomputed

    SCRUM-398 added v_rel_km_s and v_rel_source. This producer CAN supply a
    relative velocity: a CCSDS CDM carries RELATIVE_VELOCITY_R/T/N in the
    relative metadata section, and both objects' state vectors at TCA, so it is
    either a direct read or a subtraction of two vectors this function already
    parses. v_rel_source records which route was taken, so "derived" and
    "as supplied by the originator" are distinguishable in the audit trail.
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

    # ── Relative velocity (ECI, km/s) ────────────────────────────
    #
    # SCRUM-398. Without this the planner cannot define an encounter plane, so
    # it cannot compute a Pc, so the Pc gate SCRUM-389 built never runs and the
    # Mahalanobis screen decides every event. The field was optional on the
    # contract and no producer filled it.
    #
    # Sign convention: secondary minus primary, matching r_rel above and
    # matching compute_pc_from_geometry, which builds the secondary as
    # v2 = v1 + v_rel. CCSDS defines RELATIVE_VELOCITY as object 2 relative to
    # object 1, the same sense. A sign error here would rotate the encounter
    # plane by 180 degrees and be invisible for a symmetric covariance, which is
    # why the tests assert the sense rather than trusting this paragraph.
    rel_v = cdm.get("RELATIVE_VELOCITY_R")
    if rel_v is not None:
        # CDM relative velocity is RTN, metres per second.
        dv_rtn_m_s = np.array(
            [
                cdm["RELATIVE_VELOCITY_R"],
                cdm["RELATIVE_VELOCITY_T"],
                cdm["RELATIVE_VELOCITY_N"],
            ],
            dtype=np.float64,
        )
        v_rel_km_s = (rot @ dv_rtn_m_s) / 1000.0  # m/s → km/s
        v_rel_source = "relative_velocity_rtn"
    elif "OBJECT2_X_DOT" in cdm:
        v2 = np.array(
            [cdm["OBJECT2_X_DOT"], cdm["OBJECT2_Y_DOT"], cdm["OBJECT2_Z_DOT"]],
            dtype=np.float64,
        )
        v_rel_km_s = v2 - v1  # already km/s, already ECI
        v_rel_source = "state_vector_difference"
    else:
        # A CDM carrying neither is malformed against CCSDS 508.0-B-1, but the
        # planner treats a missing relative velocity as "no Pc" rather than as
        # zero, so passing None through is safe and honest.
        v_rel_km_s = None
        v_rel_source = "unavailable"

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
        "v_rel_km_s": v_rel_km_s.tolist() if v_rel_km_s is not None else None,
        "v_rel_source": v_rel_source,
        "p_rel_km2": p_rel_km2,
        "pc_precomputed": pc_precomputed,
    }
