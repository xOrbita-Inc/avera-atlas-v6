from __future__ import annotations

import math
import random
from typing import Literal

import numpy as np

from .config import SimConfig
from .models import PhysicalProperties, SimObject


MU_EARTH_KM3_S2 = 3.986004418e5
RE_EARTH_KM = 6378.137

TransitCase = Literal[
    "co_orbital",
    "moderate_crossing",
    "high_crossing",
]


def norm(vector: np.ndarray) -> float:
    """Return the Euclidean norm of a vector."""
    return float(np.linalg.norm(vector))


def unit(vector: np.ndarray) -> np.ndarray:
    """Return a normalized three-element vector."""
    array = np.asarray(vector, dtype=np.float64)

    if array.shape != (3,):
        raise ValueError("Vector must contain exactly three elements.")

    magnitude = norm(array)

    if magnitude < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")

    return array / magnitude


def circular_speed_km_s(radius_km: float) -> float:
    """Return circular-orbit speed at the given Earth-centered radius."""
    if radius_km <= RE_EARTH_KM:
        raise ValueError(
            "Circular-orbit radius must be greater than Earth's radius."
        )

    return math.sqrt(MU_EARTH_KM3_S2 / radius_km)


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=np.float64,
    )


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)

    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def perifocal_to_eci(
    r_pf_km: np.ndarray,
    v_pf_km_s: np.ndarray,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate perifocal position and velocity into ECI coordinates."""
    inclination_rad = math.radians(inc_deg)
    raan_rad = math.radians(raan_deg)
    argument_of_perigee_rad = math.radians(argp_deg)

    rotation = (
        rotation_matrix_z(raan_rad)
        @ rotation_matrix_x(inclination_rad)
        @ rotation_matrix_z(argument_of_perigee_rad)
    )

    return (
        rotation @ np.asarray(r_pf_km, dtype=np.float64),
        rotation @ np.asarray(v_pf_km_s, dtype=np.float64),
    )


def make_circular_object(
    object_id: str,
    kind: str,
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    true_anomaly_deg: float,
    physical: PhysicalProperties,
) -> SimObject:
    """Create a deterministic circular-orbit simulation object."""
    radius_km = RE_EARTH_KM + altitude_km
    true_anomaly_rad = math.radians(true_anomaly_deg)

    r_pf_km = np.array(
        [
            radius_km * math.cos(true_anomaly_rad),
            radius_km * math.sin(true_anomaly_rad),
            0.0,
        ],
        dtype=np.float64,
    )

    circular_velocity_km_s = circular_speed_km_s(radius_km)

    v_pf_km_s = np.array(
        [
            -circular_velocity_km_s * math.sin(true_anomaly_rad),
            circular_velocity_km_s * math.cos(true_anomaly_rad),
            0.0,
        ],
        dtype=np.float64,
    )

    r_eci_km, v_eci_km_s = perifocal_to_eci(
        r_pf_km=r_pf_km,
        v_pf_km_s=v_pf_km_s,
        inc_deg=inclination_deg,
        raan_deg=raan_deg,
    )

    return SimObject(
        object_id=object_id,
        kind=kind,
        r_eci_km=r_eci_km,
        v_eci_km_s=v_eci_km_s,
        physical=physical,
        metadata={
            "altitude_km": altitude_km,
            "inclination_deg": inclination_deg,
            "raan_deg": raan_deg,
            "true_anomaly_deg": true_anomaly_deg,
        },
    )


def host_physical_properties() -> PhysicalProperties:
    """Return the default host-satellite physical properties."""
    return PhysicalProperties(
        size_bin="10cm",
        characteristic_size_m=1.0,
        mass_kg=12.0,
        area_m2=0.1,
        cd=2.2,
    )


def physical_properties_for_size_bin(
    size_bin: str,
) -> PhysicalProperties:
    """Return deterministic debris properties for a supported size class."""
    properties = {
        "1cm": {
            "size_m": 0.01,
            "mass_kg": 0.001,
        },
        "5cm": {
            "size_m": 0.05,
            "mass_kg": 0.05,
        },
        "10cm": {
            "size_m": 0.10,
            "mass_kg": 0.4,
        },
    }

    try:
        selected = properties[size_bin]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported size bin: {size_bin}"
        ) from exc

    size_m = selected["size_m"]
    area_m2 = math.pi * (size_m / 2.0) ** 2

    return PhysicalProperties(
        size_bin=size_bin,
        characteristic_size_m=size_m,
        mass_kg=selected["mass_kg"],
        area_m2=area_m2,
        cd=2.2,
    )


def create_hello_world_scenario(
    config: SimConfig,
) -> dict[str, SimObject]:
    """Create the original SCRUM-290 host/debris example."""
    host = make_circular_object(
        object_id="host_001",
        kind="host",
        altitude_km=config.hosts.altitude_km,
        inclination_deg=config.hosts.inclination_deg,
        raan_deg=0.0,
        true_anomaly_deg=0.0,
        physical=host_physical_properties(),
    )

    debris = make_circular_object(
        object_id="debris_001",
        kind="debris",
        altitude_km=650.0,
        inclination_deg=74.0,
        raan_deg=15.0,
        true_anomaly_deg=35.0,
        physical=physical_properties_for_size_bin("5cm"),
    )

    return {
        host.object_id: host,
        debris.object_id: debris,
    }


def create_seeded_swarm_and_hosts(
    config: SimConfig,
) -> dict[str, SimObject]:
    """Create the deterministic SCRUM-290 swarm and constellation."""
    rng = random.Random(config.seed)
    objects: dict[str, SimObject] = {}

    host_count = config.hosts.count

    for index in range(host_count):
        if config.hosts.mode == "same_host":
            raan_deg = 0.0
        else:
            raan_deg = (360.0 / host_count) * index

        true_anomaly_deg = (360.0 / host_count) * index

        host = make_circular_object(
            object_id=f"host_{index + 1:03d}",
            kind="host",
            altitude_km=config.hosts.altitude_km,
            inclination_deg=config.hosts.inclination_deg,
            raan_deg=raan_deg,
            true_anomaly_deg=true_anomaly_deg,
            physical=host_physical_properties(),
        )

        objects[host.object_id] = host

    size_bins = list(
        config.debris.size_bin_weights.keys()
    )
    size_weights = list(
        config.debris.size_bin_weights.values()
    )

    for index in range(config.debris.count):
        size_bin = rng.choices(
            size_bins,
            weights=size_weights,
            k=1,
        )[0]

        altitude_km = rng.uniform(
            config.debris.altitude_min_km,
            config.debris.altitude_max_km,
        )
        inclination_deg = rng.uniform(
            config.debris.inclination_min_deg,
            config.debris.inclination_max_deg,
        )
        raan_deg = rng.uniform(0.0, 360.0)
        true_anomaly_deg = rng.uniform(0.0, 360.0)

        debris = make_circular_object(
            object_id=f"debris_{index + 1:04d}",
            kind="debris",
            altitude_km=altitude_km,
            inclination_deg=inclination_deg,
            raan_deg=raan_deg,
            true_anomaly_deg=true_anomaly_deg,
            physical=physical_properties_for_size_bin(
                size_bin
            ),
        )

        objects[debris.object_id] = debris

    return objects


def create_forced_detection_scenario(
    *,
    altitude_km: float = 600.0,
    range_km: float = 20.0,
    size_bin: str = "5cm",
) -> dict[str, SimObject]:
    """
    Create a deterministic geometry that passes the core sensor gates.

    The host is located on the positive ECI x-axis and travels in the positive
    y direction. The debris object is placed directly along the host velocity
    vector so it lies on the boresight when using ``velocity_aligned``.

    The default 20 km separation is inside the 59 km cutoff for 5 cm debris.
    The tangential viewing direction also clears the bright Earth limb.
    """
    if range_km <= 0.0:
        raise ValueError("range_km must be positive.")

    host_radius_km = RE_EARTH_KM + altitude_km
    host_speed_km_s = circular_speed_km_s(
        host_radius_km
    )

    host_position_km = np.array(
        [host_radius_km, 0.0, 0.0],
        dtype=np.float64,
    )
    host_velocity_km_s = np.array(
        [0.0, host_speed_km_s, 0.0],
        dtype=np.float64,
    )

    host = SimObject(
        object_id="host_001",
        kind="host",
        r_eci_km=host_position_km,
        v_eci_km_s=host_velocity_km_s,
        physical=host_physical_properties(),
        metadata={
            "scenario": "forced_detection",
            "altitude_km": altitude_km,
        },
    )

    debris = SimObject(
        object_id="debris_001",
        kind="debris",
        r_eci_km=host_position_km
        + np.array(
            [0.0, range_km, 0.0],
            dtype=np.float64,
        ),
        v_eci_km_s=host_velocity_km_s.copy(),
        physical=physical_properties_for_size_bin(
            size_bin
        ),
        metadata={
            "scenario": "forced_detection",
            "initial_range_km": range_km,
        },
    )

    return {
        host.object_id: host,
        debris.object_id: debris,
    }


def create_range_gate_scenario(
    *,
    size_bin: str,
    outside_by_km: float = 0.001,
    altitude_km: float = 600.0,
) -> dict[str, SimObject]:
    """
    Create targets exactly at and just beyond one size-class range cutoff.

    Both debris objects lie along the velocity-aligned boresight.
    """
    cutoffs_km = {
        "1cm": 12.0,
        "5cm": 59.0,
        "10cm": 117.0,
    }

    try:
        cutoff_km = cutoffs_km[size_bin]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported size bin: {size_bin}"
        ) from exc

    if outside_by_km <= 0.0:
        raise ValueError(
            "outside_by_km must be positive."
        )

    base = create_forced_detection_scenario(
        altitude_km=altitude_km,
        range_km=cutoff_km,
        size_bin=size_bin,
    )

    host = base["host_001"]

    debris_at_cutoff = base["debris_001"]
    debris_at_cutoff = SimObject(
        object_id="debris_at_cutoff",
        kind="debris",
        r_eci_km=debris_at_cutoff.r_eci_km,
        v_eci_km_s=debris_at_cutoff.v_eci_km_s,
        physical=debris_at_cutoff.physical,
        metadata={
            "scenario": "range_gate",
            "expected": "detected",
            "range_km": cutoff_km,
        },
    )

    debris_outside = SimObject(
        object_id="debris_outside_cutoff",
        kind="debris",
        r_eci_km=host.r_eci_km
        + np.array(
            [
                0.0,
                cutoff_km + outside_by_km,
                0.0,
            ],
            dtype=np.float64,
        ),
        v_eci_km_s=host.v_eci_km_s.copy(),
        physical=physical_properties_for_size_bin(
            size_bin
        ),
        metadata={
            "scenario": "range_gate",
            "expected": "outside_range",
            "range_km": cutoff_km + outside_by_km,
        },
    )

    return {
        host.object_id: host,
        debris_at_cutoff.object_id: debris_at_cutoff,
        debris_outside.object_id: debris_outside,
    }


def create_sensor_transit_scenario(
    transit_case: TransitCase,
    *,
    altitude_km: float = 600.0,
    range_km: float = 20.0,
    size_bin: str = "5cm",
) -> dict[str, SimObject]:
    """
    Create deterministic transit geometries for SCRUM-291 validation.

    The target begins on the velocity-aligned boresight. Relative cross-track
    velocity controls the approximate FOV transit duration:

    - ``co_orbital``: slow relative crossing
    - ``moderate_crossing``: medium relative crossing
    - ``high_crossing``: fast relative crossing

    These scenarios are intended for repeatable comparison of transit-time
    distributions, not as a complete population model.
    """
    relative_cross_track_speed_km_s = {
        "co_orbital": 0.5,
        "moderate_crossing": 4.0,
        "high_crossing": 10.0,
    }

    try:
        cross_track_speed_km_s = (
            relative_cross_track_speed_km_s[
                transit_case
            ]
        )
    except KeyError as exc:
        raise ValueError(
            f"Unsupported transit case: {transit_case}"
        ) from exc

    base = create_forced_detection_scenario(
        altitude_km=altitude_km,
        range_km=range_km,
        size_bin=size_bin,
    )

    host = base["host_001"]
    base_debris = base["debris_001"]

    debris = SimObject(
        object_id="debris_001",
        kind="debris",
        r_eci_km=base_debris.r_eci_km.copy(),
        v_eci_km_s=host.v_eci_km_s
        + np.array(
            [
                0.0,
                0.0,
                cross_track_speed_km_s,
            ],
            dtype=np.float64,
        ),
        physical=base_debris.physical,
        metadata={
            "scenario": "sensor_transit",
            "transit_case": transit_case,
            "initial_range_km": range_km,
            "relative_cross_track_speed_km_s": (
                cross_track_speed_km_s
            ),
        },
    )

    return {
        host.object_id: host,
        debris.object_id: debris,
    }
