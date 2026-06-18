from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import numpy as np

from .models import SimObject
from .sensor_model import DetectionResult, SensorConfig, detect_object


ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)
DEFAULT_SUN_DIR_ECI = np.array([1.0, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class AngularObservation:
    """One simulated angular observation emitted by a host sensor."""

    host_id: str
    debris_id: str
    t_seconds: float

    detected: bool
    reason: str

    ra_rad: float | None
    dec_rad: float | None
    ra_sigma_rad: float | None
    dec_sigma_rad: float | None

    ra_rate_rad_s: float | None
    dec_rate_rad_s: float | None

    range_km: float | None
    off_boresight_deg: float | None
    sunlit: bool
    earth_limb_blocked: bool

    # Observer state is emitted in SI units for direct IOD ingestion.
    observer_eci_m: np.ndarray
    observer_eci_m_s: np.ndarray


def wrap_ra_rad(ra_rad: float) -> float:
    """Wrap right ascension into the interval [0, 2π)."""
    return ra_rad % (2.0 * math.pi)


def wrapped_ra_difference_rad(current_ra_rad: float, previous_ra_rad: float) -> float:
    """Return the shortest signed RA difference in the interval [-π, π)."""
    difference = current_ra_rad - previous_ra_rad
    return (difference + math.pi) % (2.0 * math.pi) - math.pi


def los_to_ra_dec(los_eci: np.ndarray) -> tuple[float, float]:
    """
    Convert an ECI line-of-sight unit vector to right ascension and declination.
    """
    los = np.asarray(los_eci, dtype=np.float64)
    los_norm = float(np.linalg.norm(los))

    if los_norm < 1e-12:
        raise ValueError("Cannot convert a near-zero line-of-sight vector.")

    x, y, z = los / los_norm

    ra_rad = wrap_ra_rad(math.atan2(y, x))
    dec_rad = math.asin(float(np.clip(z, -1.0, 1.0)))

    return ra_rad, dec_rad


def make_noisy_ra_dec(
    los_eci: np.ndarray,
    angular_sigma_arcsec: float,
    rng: random.Random,
) -> tuple[float, float, float, float]:
    """
    Add independent Gaussian angular noise to right ascension and declination.

    The configured angular resolution is treated as the one-sigma observation
    noise for each angular component.
    """
    if angular_sigma_arcsec < 0.0:
        raise ValueError("angular_sigma_arcsec must be non-negative.")

    true_ra_rad, true_dec_rad = los_to_ra_dec(los_eci)
    sigma_rad = angular_sigma_arcsec * ARCSEC_TO_RAD

    noisy_ra_rad = wrap_ra_rad(
        true_ra_rad + rng.gauss(0.0, sigma_rad)
    )
    noisy_dec_rad = true_dec_rad + rng.gauss(0.0, sigma_rad)
    noisy_dec_rad = float(
        np.clip(noisy_dec_rad, -math.pi / 2.0, math.pi / 2.0)
    )

    return noisy_ra_rad, noisy_dec_rad, sigma_rad, sigma_rad


def estimate_angular_rates(
    previous_observation: AngularObservation,
    current_observation: AngularObservation,
) -> tuple[float | None, float | None]:
    """
    Estimate angular rates from two chronological detected observations.

    RA differences are wrapped so a crossing through 0/2π does not create an
    artificial large rate.
    """
    if not previous_observation.detected or not current_observation.detected:
        return None, None

    if (
        previous_observation.ra_rad is None
        or previous_observation.dec_rad is None
        or current_observation.ra_rad is None
        or current_observation.dec_rad is None
    ):
        return None, None

    dt_seconds = (
        current_observation.t_seconds
        - previous_observation.t_seconds
    )

    if dt_seconds <= 0.0:
        return None, None

    delta_ra_rad = wrapped_ra_difference_rad(
        current_ra_rad=current_observation.ra_rad,
        previous_ra_rad=previous_observation.ra_rad,
    )
    delta_dec_rad = (
        current_observation.dec_rad
        - previous_observation.dec_rad
    )

    return (
        delta_ra_rad / dt_seconds,
        delta_dec_rad / dt_seconds,
    )


def generate_angular_observation(
    host: SimObject,
    debris: SimObject,
    t_seconds: float,
    sensor_cfg: SensorConfig,
    rng: random.Random,
    previous_observation: AngularObservation | None = None,
    sun_dir_eci: np.ndarray | None = None,
) -> AngularObservation:
    """
    Convert ground-truth host/debris state into one sensor observation.

    A rejected detection contains gate metadata but no angular measurement.
    An accepted detection contains noisy RA/Dec, one-sigma uncertainties,
    observer ECI state, and optional rates estimated from the previous
    detection of the same host-target pair.
    """
    if sun_dir_eci is None:
        sun_dir_eci = DEFAULT_SUN_DIR_ECI

    detection: DetectionResult = detect_object(
        host=host,
        debris=debris,
        sensor_cfg=sensor_cfg,
        sun_dir_eci=np.asarray(sun_dir_eci, dtype=np.float64),
    )

    observer_eci_m = host.r_eci_km.astype(np.float64, copy=True) * 1000.0
    observer_eci_m_s = host.v_eci_km_s.astype(np.float64, copy=True) * 1000.0

    if not detection.detected:
        return AngularObservation(
            host_id=host.object_id,
            debris_id=debris.object_id,
            t_seconds=t_seconds,
            detected=False,
            reason=detection.reason,
            ra_rad=None,
            dec_rad=None,
            ra_sigma_rad=None,
            dec_sigma_rad=None,
            ra_rate_rad_s=None,
            dec_rate_rad_s=None,
            range_km=detection.range_km,
            off_boresight_deg=detection.off_boresight_deg,
            sunlit=detection.sunlit,
            earth_limb_blocked=detection.earth_limb_blocked,
            observer_eci_m=observer_eci_m,
            observer_eci_m_s=observer_eci_m_s,
        )

    relative_position_km = debris.r_eci_km - host.r_eci_km
    relative_distance_km = float(np.linalg.norm(relative_position_km))

    if relative_distance_km < 1e-12:
        raise ValueError(
            "Detected host and debris states cannot be co-located."
        )

    los_eci = relative_position_km / relative_distance_km

    (
        ra_rad,
        dec_rad,
        ra_sigma_rad,
        dec_sigma_rad,
    ) = make_noisy_ra_dec(
        los_eci=los_eci,
        angular_sigma_arcsec=sensor_cfg.angular_resolution_arcsec,
        rng=rng,
    )

    observation = AngularObservation(
        host_id=host.object_id,
        debris_id=debris.object_id,
        t_seconds=t_seconds,
        detected=True,
        reason=detection.reason,
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        ra_sigma_rad=ra_sigma_rad,
        dec_sigma_rad=dec_sigma_rad,
        ra_rate_rad_s=None,
        dec_rate_rad_s=None,
        range_km=detection.range_km,
        off_boresight_deg=detection.off_boresight_deg,
        sunlit=detection.sunlit,
        earth_limb_blocked=detection.earth_limb_blocked,
        observer_eci_m=observer_eci_m,
        observer_eci_m_s=observer_eci_m_s,
    )

    if previous_observation is None:
        return observation

    ra_rate_rad_s, dec_rate_rad_s = estimate_angular_rates(
        previous_observation=previous_observation,
        current_observation=observation,
    )

    return replace(
        observation,
        ra_rate_rad_s=ra_rate_rad_s,
        dec_rate_rad_s=dec_rate_rad_s,
    )