from __future__ import annotations

import math

from .config import DragConfig

RE_EARTH_KM = 6378.137


def altitude_from_radius_km(r_norm_km: float) -> float:
    return r_norm_km - RE_EARTH_KM


def exponential_density_kg_m3(altitude_km: float, drag_cfg: DragConfig) -> float:
    """
    Very simple exponential atmosphere model.
    Good enough for first-pass sandbox drag.
    """
    if altitude_km < 0:
        altitude_km = 0.0

    rho0 = drag_cfg.reference_density_kg_m3
    h0 = drag_cfg.reference_altitude_km
    H = drag_cfg.atmosphere_scale_height_km

    return rho0 * math.exp(-(altitude_km - h0) / H)