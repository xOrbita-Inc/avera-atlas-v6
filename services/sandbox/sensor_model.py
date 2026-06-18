from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .models import SimObject


RE_EARTH_KM = 6378.137
DEFAULT_SUN_DIR_ECI = np.array(
    [1.0, 0.0, 0.0],
    dtype=np.float64,
)

PointingMode = Literal[
    "nadir_minus_30",
    "velocity_aligned",
]

SensorMode = Literal[
    "neuromorphic_event",
]


@dataclass(frozen=True)
class SensorConfig:
    """Configuration for the TT240-40 + Acuros CQD-CMOS sensor."""

    # Tiny Telescopes TT240-40 optical properties.
    fov_full_angle_deg: float = 3.7
    aperture_mm: float = 40.0
    focal_length_mm: float = 240.0

    # The ticket treats this resolution as one-sigma angular noise.
    angular_resolution_arcsec: float = 2.9

    # Neuromorphic event mode has no frame-cycle duty penalty.
    sensor_mode: SensorMode = "neuromorphic_event"
    duty_cycle: float = 1.0

    # Default host has fixed pointing authority.
    pointing_mode: PointingMode = "nadir_minus_30"
    nadir_offset_deg: float = 30.0

    def __post_init__(self) -> None:
        if not 0.0 < self.fov_full_angle_deg < 180.0:
            raise ValueError(
                "fov_full_angle_deg must be between 0 and 180 degrees."
            )

        if self.aperture_mm <= 0.0:
            raise ValueError("aperture_mm must be positive.")

        if self.focal_length_mm <= 0.0:
            raise ValueError("focal_length_mm must be positive.")

        if self.angular_resolution_arcsec < 0.0:
            raise ValueError(
                "angular_resolution_arcsec must be non-negative."
            )

        if self.sensor_mode != "neuromorphic_event":
            raise ValueError(
                f"Unsupported sensor mode: {self.sensor_mode}"
            )

        if not 0.0 < self.duty_cycle <= 1.0:
            raise ValueError("duty_cycle must be in the interval (0, 1].")

        if not 0.0 <= self.nadir_offset_deg <= 180.0:
            raise ValueError(
                "nadir_offset_deg must be between 0 and 180 degrees."
            )


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of applying all SCRUM-291 detection gates."""

    detected: bool
    reason: str
    range_km: float | None
    range_cutoff_km: float | None
    off_boresight_deg: float | None
    sunlit: bool
    earth_limb_blocked: bool


def norm(vector: np.ndarray) -> float:
    """Return the Euclidean norm of a vector."""
    return float(np.linalg.norm(vector))


def unit(vector: np.ndarray) -> np.ndarray:
    """Return a normalized copy of a three-element vector."""
    array = np.asarray(vector, dtype=np.float64)

    if array.shape != (3,):
        raise ValueError("Vector must contain exactly three elements.")

    magnitude = norm(array)

    if magnitude < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")

    return array / magnitude


def angle_between_deg(
    first_vector: np.ndarray,
    second_vector: np.ndarray,
) -> float:
    """Return the unsigned angle between two vectors in degrees."""
    first_unit = unit(first_vector)
    second_unit = unit(second_vector)

    cosine = float(
        np.clip(
            np.dot(first_unit, second_unit),
            -1.0,
            1.0,
        )
    )

    return math.degrees(math.acos(cosine))


def range_cutoff_km_for_size_bin(size_bin: str) -> float:
    """Return the ticket-defined hard detection range for a size class."""
    cutoffs_km = {
        "1cm": 12.0,
        "5cm": 59.0,
        "10cm": 117.0,
    }

    try:
        return cutoffs_km[size_bin]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported debris size bin: {size_bin}"
        ) from exc


def boresight_from_velocity(
    host_r_eci_km: np.ndarray,
    host_v_eci_km_s: np.ndarray,
    pointing_mode: PointingMode = "nadir_minus_30",
    nadir_offset_deg: float = 30.0,
) -> np.ndarray:
    """
    Calculate a fixed host-sensor boresight in ECI coordinates.

    ``velocity_aligned`` points directly along the host velocity.

    ``nadir_minus_30`` points from nadir toward the along-track velocity
    direction. Despite the retained mode name, the actual offset is
    configurable through ``nadir_offset_deg``.
    """
    host_position = np.asarray(
        host_r_eci_km,
        dtype=np.float64,
    )
    host_velocity = np.asarray(
        host_v_eci_km_s,
        dtype=np.float64,
    )

    velocity_hat = unit(host_velocity)

    if pointing_mode == "velocity_aligned":
        return velocity_hat

    if pointing_mode != "nadir_minus_30":
        raise ValueError(
            f"Unsupported pointing mode: {pointing_mode}"
        )

    if not 0.0 <= nadir_offset_deg <= 180.0:
        raise ValueError(
            "nadir_offset_deg must be between 0 and 180 degrees."
        )

    nadir_hat = unit(-host_position)

    # Remove radial velocity so the pointing rotation uses the local
    # along-track direction.
    tangential_velocity = (
        host_velocity
        - np.dot(host_velocity, nadir_hat) * nadir_hat
    )

    if norm(tangential_velocity) < 1e-12:
        return nadir_hat

    tangential_hat = unit(tangential_velocity)
    offset_rad = math.radians(nadir_offset_deg)

    boresight = (
        math.cos(offset_rad) * nadir_hat
        + math.sin(offset_rad) * tangential_hat
    )

    return unit(boresight)


def is_within_fov(
    boresight_eci: np.ndarray,
    line_of_sight_eci: np.ndarray,
    fov_full_angle_deg: float,
) -> tuple[bool, float]:
    """
    Determine whether a line of sight lies inside the conical sensor FOV.

    The specified 3.7-degree FOV is a full cone angle, so the detection
    threshold is half that value from the boresight.
    """
    if not 0.0 < fov_full_angle_deg < 180.0:
        raise ValueError(
            "fov_full_angle_deg must be between 0 and 180 degrees."
        )

    half_angle_deg = 0.5 * fov_full_angle_deg
    off_boresight_deg = angle_between_deg(
        boresight_eci,
        line_of_sight_eci,
    )

    return (
        off_boresight_deg <= half_angle_deg,
        off_boresight_deg,
    )


def passes_range_gate(
    host_r_eci_km: np.ndarray,
    debris_r_eci_km: np.ndarray,
    size_bin: str,
) -> tuple[bool, float, float]:
    """Apply the ticket-defined hard range cutoff for a debris size."""
    relative_position_km = (
        np.asarray(debris_r_eci_km, dtype=np.float64)
        - np.asarray(host_r_eci_km, dtype=np.float64)
    )

    range_km = norm(relative_position_km)
    cutoff_km = range_cutoff_km_for_size_bin(size_bin)

    return range_km <= cutoff_km, range_km, cutoff_km


def is_sunlit(
    object_r_eci_km: np.ndarray,
    sun_dir_eci: np.ndarray | None = None,
) -> bool:
    """
    Apply a coarse cylindrical Earth-shadow test.

    The Sun is treated as infinitely distant. An object is eclipsed when it
    lies behind Earth relative to the Sun and within Earth's shadow cylinder.
    """
    if sun_dir_eci is None:
        sun_dir_eci = DEFAULT_SUN_DIR_ECI

    sun_hat = unit(sun_dir_eci)
    object_position = np.asarray(
        object_r_eci_km,
        dtype=np.float64,
    )

    if object_position.shape != (3,):
        raise ValueError(
            "object_r_eci_km must contain exactly three elements."
        )

    parallel_distance_km = float(
        np.dot(object_position, sun_hat)
    )
    perpendicular_vector_km = (
        object_position
        - parallel_distance_km * sun_hat
    )
    perpendicular_distance_km = norm(
        perpendicular_vector_km
    )

    behind_earth = parallel_distance_km < 0.0
    inside_shadow_cylinder = (
        perpendicular_distance_km < RE_EARTH_KM
    )

    return not (
        behind_earth
        and inside_shadow_cylinder
    )


def passes_earth_limb_gate(
    host_r_eci_km: np.ndarray,
    boresight_eci: np.ndarray,
    fov_full_angle_deg: float,
) -> bool:
    """
    Reject a view when the FOV cone overlaps Earth's apparent disk.

    This is the coarse binary bright-Earth-limb model allowed by SCRUM-291.
    It does not perform radiometric background calculations.
    """
    host_position = np.asarray(
        host_r_eci_km,
        dtype=np.float64,
    )
    host_radius_km = norm(host_position)

    if host_radius_km <= RE_EARTH_KM:
        return False

    nadir_hat = unit(-host_position)

    earth_angular_radius_deg = math.degrees(
        math.asin(RE_EARTH_KM / host_radius_km)
    )
    boresight_to_nadir_deg = angle_between_deg(
        boresight_eci,
        nadir_hat,
    )
    half_fov_deg = 0.5 * fov_full_angle_deg

    # The cone is clear only when its nearest edge is outside Earth's
    # apparent angular disk.
    cone_nearest_edge_deg = (
        boresight_to_nadir_deg - half_fov_deg
    )

    return cone_nearest_edge_deg > earth_angular_radius_deg


def detect_object(
    host: SimObject,
    debris: SimObject,
    sensor_cfg: SensorConfig,
    sun_dir_eci: np.ndarray | None = None,
) -> DetectionResult:
    """
    Apply FOV, range, sunlight, and Earth-limb gates to one target.

    Gates are evaluated in a deterministic order so rejection reasons are
    stable and testable.
    """
    if host.kind != "host":
        raise ValueError(
            f"{host.object_id} is not a host object."
        )

    if debris.kind != "debris":
        raise ValueError(
            f"{debris.object_id} is not a debris object."
        )

    relative_position_km = (
        debris.r_eci_km - host.r_eci_km
    )
    range_km = norm(relative_position_km)

    if range_km < 1e-12:
        return DetectionResult(
            detected=False,
            reason="co_located",
            range_km=0.0,
            range_cutoff_km=None,
            off_boresight_deg=None,
            sunlit=False,
            earth_limb_blocked=False,
        )

    line_of_sight_hat = unit(relative_position_km)

    boresight_eci = boresight_from_velocity(
        host_r_eci_km=host.r_eci_km,
        host_v_eci_km_s=host.v_eci_km_s,
        pointing_mode=sensor_cfg.pointing_mode,
        nadir_offset_deg=sensor_cfg.nadir_offset_deg,
    )

    in_fov, off_boresight_deg = is_within_fov(
        boresight_eci=boresight_eci,
        line_of_sight_eci=line_of_sight_hat,
        fov_full_angle_deg=sensor_cfg.fov_full_angle_deg,
    )

    if not in_fov:
        return DetectionResult(
            detected=False,
            reason="outside_fov",
            range_km=range_km,
            range_cutoff_km=None,
            off_boresight_deg=off_boresight_deg,
            sunlit=False,
            earth_limb_blocked=False,
        )

    (
        inside_range,
        range_km,
        cutoff_km,
    ) = passes_range_gate(
        host_r_eci_km=host.r_eci_km,
        debris_r_eci_km=debris.r_eci_km,
        size_bin=debris.physical.size_bin,
    )

    if not inside_range:
        return DetectionResult(
            detected=False,
            reason="outside_range",
            range_km=range_km,
            range_cutoff_km=cutoff_km,
            off_boresight_deg=off_boresight_deg,
            sunlit=False,
            earth_limb_blocked=False,
        )

    sunlit = is_sunlit(
        object_r_eci_km=debris.r_eci_km,
        sun_dir_eci=sun_dir_eci,
    )

    if not sunlit:
        return DetectionResult(
            detected=False,
            reason="not_sunlit",
            range_km=range_km,
            range_cutoff_km=cutoff_km,
            off_boresight_deg=off_boresight_deg,
            sunlit=False,
            earth_limb_blocked=False,
        )

    earth_limb_clear = passes_earth_limb_gate(
        host_r_eci_km=host.r_eci_km,
        boresight_eci=boresight_eci,
        fov_full_angle_deg=sensor_cfg.fov_full_angle_deg,
    )

    if not earth_limb_clear:
        return DetectionResult(
            detected=False,
            reason="earth_limb_blocked",
            range_km=range_km,
            range_cutoff_km=cutoff_km,
            off_boresight_deg=off_boresight_deg,
            sunlit=True,
            earth_limb_blocked=True,
        )

    return DetectionResult(
        detected=True,
        reason="detected",
        range_km=range_km,
        range_cutoff_km=cutoff_km,
        off_boresight_deg=off_boresight_deg,
        sunlit=True,
        earth_limb_blocked=False,
    )