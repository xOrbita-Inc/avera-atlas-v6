from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import brentq

from aps_math import conventions
from aps_math.pc_utils import compute_pc, default_covariance_from_uncertainty
from common.orbit_propagation import kepler_propagate


# Existing ingest/propagator cadence. SCRUM-381 uses it only to bracket
# stationary points of relative distance; the final TCA is refined continuously.
_TCA_BRACKET_STEP_S = 60.0


def _parse_utc(value: str) -> datetime:
    if not value:
        raise ValueError("burn_time_utc is required for secondary horizon screening")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _catalog_initial_state(
    obj: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return one catalog object's normalized J2000 ECI state at burn epoch."""
    r_secondary_0 = np.asarray(obj.get("r_km"), dtype=float)
    v_secondary_0 = np.asarray(obj.get("v_km_s"), dtype=float)
    if r_secondary_0.shape != (3,) or v_secondary_0.shape != (3,):
        raise ValueError(
            f"catalog object {obj.get('obj_id', 'UNKNOWN')} is missing a 3D "
            "J2000 ECI position/velocity state"
        )
    if not np.all(np.isfinite(r_secondary_0)) or not np.all(
        np.isfinite(v_secondary_0)
    ):
        raise ValueError(
            f"catalog object {obj.get('obj_id', 'UNKNOWN')} has a non-finite "
            "J2000 ECI state"
        )
    return r_secondary_0, v_secondary_0


def _debris_uncertainty_m(obj: Dict[str, Any]) -> tuple[float, str]:
    """Reuse the SCRUM-391 TLE-grade secondary uncertainty convention.

    A supplied position_sigma_m is used directly. Otherwise the controlled
    screening fallback is DEFAULT_DEBRIS_UNCERTAINTY_M / max(confidence, 0.1).
    Missing confidence follows the propagator convention and defaults to 0.8.
    These are screening conventions, not measured object uncertainties.
    """
    sigma_in = float(obj.get("position_sigma_m", float("nan")))
    confidence = float(obj.get("confidence", 0.8))
    if np.isfinite(sigma_in) and sigma_in > 0.0:
        return sigma_in, "supplied"
    return (
        conventions.DEFAULT_DEBRIS_UNCERTAINTY_M / max(confidence, 0.1),
        "confidence_default",
    )


def _relative_state(
    r_primary_0_km: np.ndarray,
    v_primary_0_km_s: np.ndarray,
    r_secondary_0_km: np.ndarray,
    v_secondary_0_km_s: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_primary, v_primary = kepler_propagate(
        r_primary_0_km,
        v_primary_0_km_s,
        float(dt_s),
    )
    r_secondary, v_secondary = kepler_propagate(
        r_secondary_0_km,
        v_secondary_0_km_s,
        float(dt_s),
    )
    return r_primary, v_primary, r_secondary, v_secondary


def _distance_derivative(
    r_primary_0_km: np.ndarray,
    v_primary_0_km_s: np.ndarray,
    r_secondary_0_km: np.ndarray,
    v_secondary_0_km_s: np.ndarray,
    dt_s: float,
) -> float:
    r_primary, v_primary, r_secondary, v_secondary = _relative_state(
        r_primary_0_km,
        v_primary_0_km_s,
        r_secondary_0_km,
        v_secondary_0_km_s,
        dt_s,
    )
    r_rel = r_secondary - r_primary
    v_rel = v_secondary - v_primary
    # 0.5 * d(|r_rel|^2)/dt = r_rel dot v_rel.
    return float(np.dot(r_rel, v_rel))


def _candidate_state(
    r_primary_0_km: np.ndarray,
    v_primary_0_km_s: np.ndarray,
    r_secondary_0_km: np.ndarray,
    v_secondary_0_km_s: np.ndarray,
    dt_s: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_primary, v_primary, r_secondary, v_secondary = _relative_state(
        r_primary_0_km,
        v_primary_0_km_s,
        r_secondary_0_km,
        v_secondary_0_km_s,
        dt_s,
    )
    distance_km = float(np.linalg.norm(r_secondary - r_primary))
    if not np.isfinite(distance_km):
        raise RuntimeError("secondary TCA search produced a non-finite miss distance")
    return distance_km, r_primary, v_primary, r_secondary, v_secondary


def _find_object_tca(
    r_primary_0_km: np.ndarray,
    v_primary_0_km_s: np.ndarray,
    r_secondary_0_km: np.ndarray,
    v_secondary_0_km_s: np.ndarray,
    horizon_s: float,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("secondary screening horizon must be positive and finite")

    grid = np.arange(0.0, horizon_s, _TCA_BRACKET_STEP_S, dtype=float)
    if grid.size == 0 or grid[-1] != horizon_s:
        grid = np.append(grid, horizon_s)

    best: Optional[
        tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = None

    def consider(dt_s: float) -> None:
        nonlocal best
        distance_km, r1, v1, r2, v2 = _candidate_state(
            r_primary_0_km,
            v_primary_0_km_s,
            r_secondary_0_km,
            v_secondary_0_km_s,
            dt_s,
        )
        candidate = (distance_km, float(dt_s), r1, v1, r2, v2)
        if best is None or distance_km < best[0]:
            best = candidate

    # Horizon endpoints are valid TCAs when distance is monotonic over the window.
    consider(float(grid[0]))
    if grid[-1] != grid[0]:
        consider(float(grid[-1]))

    previous_t = float(grid[0])
    previous_g = _distance_derivative(
        r_primary_0_km,
        v_primary_0_km_s,
        r_secondary_0_km,
        v_secondary_0_km_s,
        previous_t,
    )
    if not np.isfinite(previous_g):
        raise RuntimeError("secondary TCA search produced a non-finite derivative")

    for current in grid[1:]:
        current_t = float(current)
        current_g = _distance_derivative(
            r_primary_0_km,
            v_primary_0_km_s,
            r_secondary_0_km,
            v_secondary_0_km_s,
            current_t,
        )
        if not np.isfinite(current_g):
            raise RuntimeError("secondary TCA search produced a non-finite derivative")

        # A local minimum of squared separation crosses from closing (g < 0)
        # to opening (g > 0). Refine that root rather than accepting the 60 s
        # sample-grid argmin.
        if previous_g <= 0.0 <= current_g:
            if previous_g == 0.0:
                root_t = previous_t
            elif current_g == 0.0:
                root_t = current_t
            else:
                root_t = float(
                    brentq(
                        lambda t: _distance_derivative(
                            r_primary_0_km,
                            v_primary_0_km_s,
                            r_secondary_0_km,
                            v_secondary_0_km_s,
                            t,
                        ),
                        previous_t,
                        current_t,
                        xtol=1e-6,
                        rtol=1e-12,
                        maxiter=100,
                    )
                )
            consider(root_t)

        previous_t = current_t
        previous_g = current_g

    if best is None:
        raise RuntimeError("secondary TCA search produced no candidate state")

    distance_km, dt_s, r1, v1, r2, v2 = best
    return dt_s, distance_km, r1, v1, r2, v2


def screen_secondary_catalog(
    r_post_km: List[float],
    v_post_km_s: List[float],
    known_objects: List[Dict[str, Any]],
    burn_time_utc: str,
    horizon_hours: float,
    pc_action: float,
    primary_radius_m: float,
) -> Dict[str, Any]:
    """Screen normalized J2000 catalog states across the post-burn policy horizon."""
    if not known_objects:
        raise ValueError("no catalog objects supplied for secondary horizon screening")

    r_primary_0 = np.asarray(r_post_km, dtype=float)
    v_primary_0 = np.asarray(v_post_km_s, dtype=float)
    if r_primary_0.shape != (3,) or v_primary_0.shape != (3,):
        raise ValueError(
            "post-burn primary state must contain 3 position and 3 velocity values"
        )
    if not np.all(np.isfinite(r_primary_0)) or not np.all(np.isfinite(v_primary_0)):
        raise ValueError("post-burn primary state must be finite")

    horizon_s = float(horizon_hours) * 3600.0
    threshold = float(pc_action)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("pc_action must be positive and finite")

    burn_epoch = _parse_utc(burn_time_utc)

    primary_cov = default_covariance_from_uncertainty(
        conventions.DEFAULT_ASSET_UNCERTAINTY_M,
        cross_track_factor=conventions.DEFAULT_PRIMARY_CROSS_TRACK_FACTOR,
    )
    hbr_m, _hbr_source = conventions.combined_hbr_m(primary_radius_m)

    flagged: List[str] = []
    closest_distance_km = float("inf")
    closest_object_id: Optional[str] = None
    max_pc = -1.0
    max_pc_object_id: Optional[str] = None
    max_pc_tca_utc: Optional[str] = None

    for obj in known_objects:
        obj_id = str(obj.get("obj_id", "UNKNOWN"))
        r_secondary_0, v_secondary_0 = _catalog_initial_state(obj)

        uncertainty_m, _uncertainty_source = _debris_uncertainty_m(obj)
        secondary_cov = default_covariance_from_uncertainty(
            uncertainty_m,
            cross_track_factor=conventions.DEFAULT_SECONDARY_CROSS_TRACK_FACTOR,
        )

        dt_s, distance_km, r1, v1, r2, v2 = _find_object_tca(
            r_primary_0,
            v_primary_0,
            r_secondary_0,
            v_secondary_0,
            horizon_s,
        )

        pc_result = compute_pc(
            r1 * 1000.0,
            v1 * 1000.0,
            primary_cov,
            r2 * 1000.0,
            v2 * 1000.0,
            secondary_cov,
            hbr_m,
        )
        pc = float(pc_result.Pc)
        if not np.isfinite(pc):
            raise RuntimeError(f"Pc computation returned non-finite value for {obj_id}")

        tca = burn_epoch + timedelta(seconds=dt_s)
        tca_text = tca.isoformat().replace("+00:00", "Z")

        if distance_km < closest_distance_km:
            closest_distance_km = distance_km
            closest_object_id = obj_id

        if pc > max_pc:
            max_pc = pc
            max_pc_object_id = obj_id
            max_pc_tca_utc = tca_text

        # MAF gate is strict: CLEAR only when every secondary Pc is below
        # Pc_action. Equality is therefore NOT CLEAR.
        if pc >= threshold:
            flagged.append(obj_id)

    if max_pc < 0.0:
        raise RuntimeError("secondary horizon screen computed no Pc values")

    clear = len(flagged) == 0
    if clear:
        note = (
            f"Secondary horizon screen CLEAR across {len(known_objects)} catalog "
            f"object(s) over {horizon_hours:g} h. Maximum Pc={max_pc:.6g} "
            f"for {max_pc_object_id} at {max_pc_tca_utc}, strictly below "
            f"Pc_action={threshold:.6g}."
        )
    else:
        note = (
            f"Secondary horizon screen NOT CLEAR: {len(flagged)} object(s) have "
            f"Pc >= Pc_action={threshold:.6g} over the {horizon_hours:g} h "
            f"policy horizon: {', '.join(flagged)}. Maximum Pc={max_pc:.6g} "
            f"for {max_pc_object_id} at {max_pc_tca_utc}."
        )

    return {
        "secondary_check_performed": True,
        "secondary_conjunction_clear": clear,
        "flagged_objects": flagged,
        "operator_note": note,
        "closest_approach_km": round(closest_distance_km, 6),
        "closest_object_id": closest_object_id,
        "screening_epoch_utc": burn_time_utc,
    }
