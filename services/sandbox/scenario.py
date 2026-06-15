from __future__ import annotations

import math
import random

import numpy as np

from .config import SimConfig
from .models import PhysicalProperties, SimObject


MU_EARTH_KM3_S2 = 3.986004418e5
RE_EARTH_KM = 6378.137


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n < 1e-12:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def circular_speed_km_s(radius_km: float) -> float:
    return math.sqrt(MU_EARTH_KM3_S2 / radius_km)


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
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
    inc = math.radians(inc_deg)
    raan = math.radians(raan_deg)
    argp = math.radians(argp_deg)

    rot = rotation_matrix_z(raan) @ rotation_matrix_x(inc) @ rotation_matrix_z(argp)
    return rot @ r_pf_km, rot @ v_pf_km_s


def make_circular_object(
    object_id: str,
    kind: str,
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    true_anomaly_deg: float,
    physical: PhysicalProperties,
) -> SimObject:
    radius_km = RE_EARTH_KM + altitude_km
    nu = math.radians(true_anomaly_deg)

    r_pf = np.array(
        [radius_km * math.cos(nu), radius_km * math.sin(nu), 0.0],
        dtype=np.float64,
    )
    v_circ = circular_speed_km_s(radius_km)
    v_pf = np.array(
        [-v_circ * math.sin(nu), v_circ * math.cos(nu), 0.0],
        dtype=np.float64,
    )

    r_eci, v_eci = perifocal_to_eci(
        r_pf_km=r_pf,
        v_pf_km_s=v_pf,
        inc_deg=inclination_deg,
        raan_deg=raan_deg,
    )

    return SimObject(
        object_id=object_id,
        kind=kind,
        r_eci_km=r_eci,
        v_eci_km_s=v_eci,
        physical=physical,
        metadata={
            "altitude_km": altitude_km,
            "inclination_deg": inclination_deg,
            "raan_deg": raan_deg,
            "true_anomaly_deg": true_anomaly_deg,
        },
    )


def physical_properties_for_size_bin(size_bin: str) -> PhysicalProperties:
    if size_bin == "1cm":
        size_m = 0.01
        mass_kg = 0.001
    elif size_bin == "5cm":
        size_m = 0.05
        mass_kg = 0.05
    elif size_bin == "10cm":
        size_m = 0.10
        mass_kg = 0.4
    else:
        raise ValueError(f"Unsupported size bin: {size_bin}")

    area_m2 = math.pi * (size_m / 2.0) ** 2
    return PhysicalProperties(
        size_bin=size_bin,
        characteristic_size_m=size_m,
        mass_kg=mass_kg,
        area_m2=area_m2,
        cd=2.2,
    )


def create_hello_world_scenario(config: SimConfig) -> dict[str, SimObject]:
    host_physical = PhysicalProperties(
        size_bin="10cm",
        characteristic_size_m=1.0,
        mass_kg=12.0,
        area_m2=0.1,
        cd=2.2,
    )
    debris_physical = physical_properties_for_size_bin("5cm")

    host = make_circular_object(
        object_id="host_001",
        kind="host",
        altitude_km=config.hosts.altitude_km,
        inclination_deg=config.hosts.inclination_deg,
        raan_deg=0.0,
        true_anomaly_deg=0.0,
        physical=host_physical,
    )

    debris = make_circular_object(
        object_id="debris_001",
        kind="debris",
        altitude_km=650.0,
        inclination_deg=74.0,
        raan_deg=15.0,
        true_anomaly_deg=35.0,
        physical=debris_physical,
    )

    return {
        host.object_id: host,
        debris.object_id: debris,
    }


def create_seeded_swarm_and_hosts(config: SimConfig) -> dict[str, SimObject]:
    rng = random.Random(config.seed)
    objects: dict[str, SimObject] = {}

    host_physical = PhysicalProperties(
        size_bin="10cm",
        characteristic_size_m=1.0,
        mass_kg=12.0,
        area_m2=0.1,
        cd=2.2,
    )

    host_count = config.hosts.count
    for i in range(host_count):
        if config.hosts.mode == "same_host":
            raan_deg = 0.0
        else:
            raan_deg = (360.0 / host_count) * i

        true_anomaly_deg = (360.0 / host_count) * i

        host = make_circular_object(
            object_id=f"host_{i + 1:03d}",
            kind="host",
            altitude_km=config.hosts.altitude_km,
            inclination_deg=config.hosts.inclination_deg,
            raan_deg=raan_deg,
            true_anomaly_deg=true_anomaly_deg,
            physical=host_physical,
        )
        objects[host.object_id] = host

    bins = list(config.debris.size_bin_weights.keys())
    weights = list(config.debris.size_bin_weights.values())

    for i in range(config.debris.count):
        size_bin = rng.choices(bins, weights=weights, k=1)[0]
        physical = physical_properties_for_size_bin(size_bin)

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
            object_id=f"debris_{i + 1:04d}",
            kind="debris",
            altitude_km=altitude_km,
            inclination_deg=inclination_deg,
            raan_deg=raan_deg,
            true_anomaly_deg=true_anomaly_deg,
            physical=physical,
        )
        objects[debris.object_id] = debris

    return objects