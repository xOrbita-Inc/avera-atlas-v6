from __future__ import annotations

import numpy as np

from .atmosphere import RE_EARTH_KM, altitude_from_radius_km, exponential_density_kg_m3
from .config import DragConfig
from .models import SimObject

MU_EARTH_KM3_S2 = 3.986004418e5
J2 = 1.08262668e-3
OMEGA_EARTH_RAD_S = 7.2921159e-5


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def two_body_accel_km_s2(r_eci_km: np.ndarray) -> np.ndarray:
    r = norm(r_eci_km)
    if r < 1e-12:
        raise ValueError("Position norm too small for gravity computation.")
    return -MU_EARTH_KM3_S2 * r_eci_km / (r**3)


def j2_accel_km_s2(r_eci_km: np.ndarray) -> np.ndarray:
    x, y, z = r_eci_km
    r = norm(r_eci_km)
    if r < 1e-12:
        raise ValueError("Position norm too small for J2 computation.")

    r2 = r * r
    z2 = z * z
    factor = 1.5 * J2 * MU_EARTH_KM3_S2 * (RE_EARTH_KM**2) / (r**5)

    common = 5.0 * z2 / r2
    ax = factor * x * (common - 1.0)
    ay = factor * y * (common - 1.0)
    az = factor * z * (common - 3.0)

    return np.array([ax, ay, az], dtype=np.float64)


def drag_accel_km_s2(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    sim_object: SimObject,
    drag_cfg: DragConfig,
) -> np.ndarray:
    if not drag_cfg.enabled:
        return np.zeros(3, dtype=np.float64)

    r_norm_km = norm(r_eci_km)
    altitude_km = altitude_from_radius_km(r_norm_km)
    rho_kg_m3 = exponential_density_kg_m3(altitude_km, drag_cfg)

    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S], dtype=np.float64)
    r_eci_m = r_eci_km * 1000.0
    v_eci_m_s = v_eci_km_s * 1000.0

    v_atm_m_s = np.cross(omega_vec, r_eci_m)
    v_rel_m_s = v_eci_m_s - v_atm_m_s
    v_rel_norm = float(np.linalg.norm(v_rel_m_s))

    if v_rel_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)

    area_m2 = sim_object.physical.area_m2
    mass_kg = sim_object.physical.mass_kg
    cd = sim_object.physical.cd

    accel_m_s2 = -0.5 * rho_kg_m3 * cd * area_m2 / mass_kg * v_rel_norm * v_rel_m_s
    return accel_m_s2 / 1000.0


def total_accel_km_s2(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    sim_object: SimObject,
    drag_cfg: DragConfig,
) -> np.ndarray:
    return (
        two_body_accel_km_s2(r_eci_km)
        + j2_accel_km_s2(r_eci_km)
        + drag_accel_km_s2(r_eci_km, v_eci_km_s, sim_object, drag_cfg)
    )


def state_derivative(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    sim_object: SimObject,
    drag_cfg: DragConfig,
) -> tuple[np.ndarray, np.ndarray]:
    a_eci_km_s2 = total_accel_km_s2(r_eci_km, v_eci_km_s, sim_object, drag_cfg)
    return v_eci_km_s, a_eci_km_s2


def rk4_step(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    dt_s: float,
    sim_object: SimObject,
    drag_cfg: DragConfig,
) -> tuple[np.ndarray, np.ndarray]:
    k1_r, k1_v = state_derivative(r_eci_km, v_eci_km_s, sim_object, drag_cfg)

    k2_r, k2_v = state_derivative(
        r_eci_km + 0.5 * dt_s * k1_r,
        v_eci_km_s + 0.5 * dt_s * k1_v,
        sim_object,
        drag_cfg,
    )

    k3_r, k3_v = state_derivative(
        r_eci_km + 0.5 * dt_s * k2_r,
        v_eci_km_s + 0.5 * dt_s * k2_v,
        sim_object,
        drag_cfg,
    )

    k4_r, k4_v = state_derivative(
        r_eci_km + dt_s * k3_r,
        v_eci_km_s + dt_s * k3_v,
        sim_object,
        drag_cfg,
    )

    r_next = r_eci_km + (dt_s / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_next = v_eci_km_s + (dt_s / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return r_next.astype(np.float64), v_next.astype(np.float64)


def _two_body_accel_batch_km_s2(r_eci_km: np.ndarray) -> np.ndarray:
    radii = np.linalg.norm(r_eci_km, axis=1)
    if np.any(radii < 1e-12):
        raise ValueError("Position norm too small for gravity computation.")

    return -MU_EARTH_KM3_S2 * r_eci_km / (radii[:, None] ** 3)


def _j2_accel_batch_km_s2(r_eci_km: np.ndarray) -> np.ndarray:
    x = r_eci_km[:, 0]
    y = r_eci_km[:, 1]
    z = r_eci_km[:, 2]

    radii = np.linalg.norm(r_eci_km, axis=1)
    if np.any(radii < 1e-12):
        raise ValueError("Position norm too small for J2 computation.")

    r2 = radii * radii
    z2 = z * z
    factor = 1.5 * J2 * MU_EARTH_KM3_S2 * (RE_EARTH_KM**2) / (radii**5)
    common = 5.0 * z2 / r2

    accel = np.empty_like(r_eci_km, dtype=np.float64)
    accel[:, 0] = factor * x * (common - 1.0)
    accel[:, 1] = factor * y * (common - 1.0)
    accel[:, 2] = factor * z * (common - 3.0)

    return accel


def _drag_accel_batch_km_s2(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    area_m2: np.ndarray,
    mass_kg: np.ndarray,
    cd: np.ndarray,
    drag_cfg: DragConfig,
) -> np.ndarray:
    if not drag_cfg.enabled:
        return np.zeros_like(r_eci_km, dtype=np.float64)

    radii_km = np.linalg.norm(r_eci_km, axis=1)
    altitude_km = np.maximum(radii_km - RE_EARTH_KM, 0.0)

    rho_kg_m3 = drag_cfg.reference_density_kg_m3 * np.exp(
        -(altitude_km - drag_cfg.reference_altitude_km) / drag_cfg.atmosphere_scale_height_km
    )

    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S], dtype=np.float64)
    r_eci_m = r_eci_km * 1000.0
    v_eci_m_s = v_eci_km_s * 1000.0

    v_atm_m_s = np.cross(omega_vec, r_eci_m)
    v_rel_m_s = v_eci_m_s - v_atm_m_s
    v_rel_norm = np.linalg.norm(v_rel_m_s, axis=1)

    accel_m_s2 = np.zeros_like(r_eci_km, dtype=np.float64)
    moving = v_rel_norm >= 1e-12

    if np.any(moving):
        coeff = (
            -0.5
            * rho_kg_m3[moving]
            * cd[moving]
            * area_m2[moving]
            / mass_kg[moving]
            * v_rel_norm[moving]
        )
        accel_m_s2[moving] = coeff[:, None] * v_rel_m_s[moving]

    return accel_m_s2 / 1000.0


def _total_accel_batch_km_s2(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    area_m2: np.ndarray,
    mass_kg: np.ndarray,
    cd: np.ndarray,
    drag_cfg: DragConfig,
) -> np.ndarray:
    return (
        _two_body_accel_batch_km_s2(r_eci_km)
        + _j2_accel_batch_km_s2(r_eci_km)
        + _drag_accel_batch_km_s2(
            r_eci_km=r_eci_km,
            v_eci_km_s=v_eci_km_s,
            area_m2=area_m2,
            mass_kg=mass_kg,
            cd=cd,
            drag_cfg=drag_cfg,
        )
    )


def _state_derivative_batch(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    area_m2: np.ndarray,
    mass_kg: np.ndarray,
    cd: np.ndarray,
    drag_cfg: DragConfig,
) -> tuple[np.ndarray, np.ndarray]:
    a_eci_km_s2 = _total_accel_batch_km_s2(
        r_eci_km=r_eci_km,
        v_eci_km_s=v_eci_km_s,
        area_m2=area_m2,
        mass_kg=mass_kg,
        cd=cd,
        drag_cfg=drag_cfg,
    )
    return v_eci_km_s, a_eci_km_s2


def rk4_step_batch(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    dt_s: float,
    area_m2: np.ndarray,
    mass_kg: np.ndarray,
    cd: np.ndarray,
    drag_cfg: DragConfig,
) -> tuple[np.ndarray, np.ndarray]:
    k1_r, k1_v = _state_derivative_batch(
        r_eci_km, v_eci_km_s, area_m2, mass_kg, cd, drag_cfg
    )

    k2_r, k2_v = _state_derivative_batch(
        r_eci_km + 0.5 * dt_s * k1_r,
        v_eci_km_s + 0.5 * dt_s * k1_v,
        area_m2,
        mass_kg,
        cd,
        drag_cfg,
    )

    k3_r, k3_v = _state_derivative_batch(
        r_eci_km + 0.5 * dt_s * k2_r,
        v_eci_km_s + 0.5 * dt_s * k2_v,
        area_m2,
        mass_kg,
        cd,
        drag_cfg,
    )

    k4_r, k4_v = _state_derivative_batch(
        r_eci_km + dt_s * k3_r,
        v_eci_km_s + dt_s * k3_v,
        area_m2,
        mass_kg,
        cd,
        drag_cfg,
    )

    r_next = r_eci_km + (dt_s / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_next = v_eci_km_s + (dt_s / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return r_next.astype(np.float64), v_next.astype(np.float64)