from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from sgp4.api import Satrec, jday

from iod import (
    IODObservation,
    IODSolver,
    MU_EARTH_KM,
    RE_EARTH,
    state_to_elements,
    norm,
    unit,
)


J2 = 1.08262668e-3


@dataclass
class TLECase:
    name: str
    line1: str
    line2: str
    offsets_s: list[float]


def datetime_to_jd(dt: datetime) -> tuple[float, float]:
    dt = dt.astimezone(timezone.utc)
    jd, fr = jday(
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second + dt.microsecond * 1e-6,
    )
    return jd, fr


def sgp4_state_km(tle: TLECase, dt: datetime) -> tuple[np.ndarray, np.ndarray]:
    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    jd, fr = datetime_to_jd(dt)
    error_code, r_km, v_km_s = sat.sgp4(jd, fr)

    if error_code != 0:
        raise RuntimeError(
            f"SGP4 failed for {tle.name} at {dt.isoformat()} with code {error_code}"
        )

    return np.array(r_km, dtype=np.float64), np.array(v_km_s, dtype=np.float64)


def ra_dec_from_los(los_eci: np.ndarray) -> tuple[float, float]:
    los_hat = unit(los_eci)
    x, y, z = los_hat

    ra = math.atan2(y, x)
    if ra < 0.0:
        ra += 2.0 * math.pi

    dec = math.asin(float(np.clip(z, -1.0, 1.0)))
    return ra, dec


def make_synthetic_observer_state(
    target_r_km: np.ndarray,
    target_v_km_s: np.ndarray,
    range_km: float = 900.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic observer near the target in a repeatable local orbital frame.
    """
    r_hat = unit(target_r_km)
    v_hat = unit(target_v_km_s)
    h_hat = unit(np.cross(target_r_km, target_v_km_s))

    offset_dir = unit(-0.65 * v_hat + 0.65 * h_hat + 0.25 * r_hat)

    observer_r_km = target_r_km + range_km * offset_dir
    observer_v_km_s = target_v_km_s + 0.015 * h_hat - 0.010 * v_hat

    return observer_r_km, observer_v_km_s


def make_observations_from_tle(
    tle: TLECase,
    t0: datetime,
) -> tuple[list[IODObservation], list[tuple[datetime, np.ndarray, np.ndarray]]]:
    observations: list[IODObservation] = []
    truth_states: list[tuple[datetime, np.ndarray, np.ndarray]] = []

    for offset_s in tle.offsets_s:
        timestamp = t0 + timedelta(seconds=offset_s)

        target_r_km, target_v_km_s = sgp4_state_km(tle, timestamp)
        observer_r_km, observer_v_km_s = make_synthetic_observer_state(
            target_r_km,
            target_v_km_s,
        )

        los = target_r_km - observer_r_km
        ra, dec = ra_dec_from_los(los)

        observations.append(
            IODObservation(
                timestamp=timestamp,
                ra=ra,
                dec=dec,
                ra_sigma=math.radians(2.9 / 3600.0),
                dec_sigma=math.radians(2.9 / 3600.0),
                observer_position_km=observer_r_km,
                observer_velocity_km_s=observer_v_km_s,
            )
        )

        truth_states.append((timestamp, target_r_km, target_v_km_s))

    return observations, truth_states


def gravity_j2_drag_accel_km_s2(
    r_km: np.ndarray,
    v_km_s: np.ndarray,
    cd: float = 2.2,
    area_m2: float = 0.01,
    mass_kg: float = 1.0,
) -> np.ndarray:
    """
    Acceleration model in km/s^2:
    - central Earth gravity
    - J2 perturbation
    - simple exponential atmospheric drag
    """
    r_norm = norm(r_km)
    x, y, z = r_km

    # Central gravity.
    a_grav = -MU_EARTH_KM * r_km / r_norm**3

    # J2 perturbation.
    z2 = z * z
    r2 = r_norm * r_norm
    factor = 1.5 * J2 * MU_EARTH_KM * RE_EARTH**2 / r_norm**5

    a_j2 = factor * np.array(
        [
            x * (5.0 * z2 / r2 - 1.0),
            y * (5.0 * z2 / r2 - 1.0),
            z * (5.0 * z2 / r2 - 3.0),
        ],
        dtype=np.float64,
    )

    # Coarse exponential atmosphere for validation only.
    altitude_km = r_norm - RE_EARTH
    rho0 = 3.614e-13  # kg/m^3 near 700 km
    h0_km = 700.0
    scale_height_km = 88.667

    rho = rho0 * math.exp(-(altitude_km - h0_km) / scale_height_km)
    rho = max(0.0, min(rho, 1e-8))

    v_m_s = v_km_s * 1000.0
    v_m_s_norm = norm(v_m_s)

    if v_m_s_norm < 1e-12:
        a_drag = np.zeros(3, dtype=np.float64)
    else:
        ballistic = cd * area_m2 / mass_kg
        a_drag_m_s2 = -0.5 * rho * ballistic * v_m_s_norm * v_m_s
        a_drag = a_drag_m_s2 / 1000.0  # km/s^2

    return a_grav + a_j2 + a_drag


def rk4_step(
    r_km: np.ndarray,
    v_km_s: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    def deriv(state: np.ndarray) -> np.ndarray:
        r = state[:3]
        v = state[3:]
        a = gravity_j2_drag_accel_km_s2(r, v)
        return np.hstack((v, a))

    y = np.hstack((r_km, v_km_s))

    k1 = deriv(y)
    k2 = deriv(y + 0.5 * dt_s * k1)
    k3 = deriv(y + 0.5 * dt_s * k2)
    k4 = deriv(y + dt_s * k3)

    y_next = y + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return y_next[:3], y_next[3:]


def propagate_one_period_rk4(
    r0_km: np.ndarray,
    v0_km_s: np.ndarray,
    step_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Propagate one orbital period with RK4 + central gravity + J2 + drag.
    """
    elements = state_to_elements(r0_km, v0_km_s)
    a_km = elements["semi_major_axis_km"]

    if a_km is None or not np.isfinite(a_km) or a_km <= 0:
        raise ValueError("Cannot propagate one period for non-elliptic orbit.")

    period_s = 2.0 * math.pi * math.sqrt(a_km**3 / MU_EARTH_KM)

    r = r0_km.copy()
    v = v0_km_s.copy()

    positions = [r.copy()]
    velocities = [v.copy()]

    n_steps = int(math.ceil(period_s / step_s))
    for step_idx in range(n_steps):
        remaining_s = period_s - step_idx * step_s
        this_step_s = min(step_s, remaining_s)
        r, v = rk4_step(r, v, this_step_s)

        positions.append(r.copy())
        velocities.append(v.copy())

    return np.array(positions), np.array(velocities), period_s


def plot_validation_case(
    case_name: str,
    truth_states: list[tuple[datetime, np.ndarray, np.ndarray]],
    observations: list[IODObservation],
    solution_r_km: np.ndarray,
    propagated_positions_km: np.ndarray,
    output_dir: Path,
) -> Path:
    """
    Save a 3D validation plot:
    - Earth wireframe
    - TLE truth observation points
    - synthetic observer points
    - recovered IOD state
    - RK4/J2/drag propagated trajectory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = (
        case_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )
    out_path = output_dir / f"{safe_name}_rk4_validation.png"

    truth_positions = np.array([state[1] for state in truth_states])
    observer_positions = np.array([obs.observer_position_km for obs in observations])

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Earth wireframe.
    u = np.linspace(0.0, 2.0 * math.pi, 48)
    v = np.linspace(0.0, math.pi, 24)
    x = RE_EARTH * np.outer(np.cos(u), np.sin(v))
    y = RE_EARTH * np.outer(np.sin(u), np.sin(v))
    z = RE_EARTH * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, linewidth=0.3, alpha=0.25)

    ax.plot(
        propagated_positions_km[:, 0],
        propagated_positions_km[:, 1],
        propagated_positions_km[:, 2],
        linewidth=1.5,
        label="RK4/J2/drag propagated IOD trajectory",
    )

    ax.scatter(
        truth_positions[:, 0],
        truth_positions[:, 1],
        truth_positions[:, 2],
        marker="o",
        s=35,
        label="TLE truth observation points",
    )

    ax.scatter(
        observer_positions[:, 0],
        observer_positions[:, 1],
        observer_positions[:, 2],
        marker="^",
        s=35,
        label="Synthetic observer points",
    )

    ax.scatter(
        [solution_r_km[0]],
        [solution_r_km[1]],
        [solution_r_km[2]],
        marker="x",
        s=80,
        label="Recovered IOD state",
    )

    all_points = np.vstack(
        [
            propagated_positions_km,
            truth_positions,
            observer_positions,
            solution_r_km.reshape(1, 3),
        ]
    )
    center = all_points.mean(axis=0)
    max_range = np.max(np.ptp(all_points, axis=0))
    half = max_range / 2.0

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    ax.set_title(f"IOD RK4/J2/drag Validation: {case_name}")
    ax.set_xlabel("ECI X (km)")
    ax.set_ylabel("ECI Y (km)")
    ax.set_zlabel("ECI Z (km)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return out_path


def print_elements(prefix: str, r_km: np.ndarray, v_km_s: np.ndarray) -> None:
    elements = state_to_elements(r_km, v_km_s)

    print(prefix)
    print(f"  position_km: {r_km}")
    print(f"  velocity_km_s: {v_km_s}")
    print(f"  semi_major_axis_km: {elements['semi_major_axis_km']}")
    print(f"  eccentricity: {elements['eccentricity']}")
    print(f"  inclination_deg: {elements['inclination_deg']}")
    print(f"  raan_deg: {elements['raan_deg']}")
    print(f"  arg_perigee_deg: {elements['arg_perigee_deg']}")
    print(f"  true_anomaly_deg: {elements['true_anomaly_deg']}")
    print(f"  perigee_km: {elements['perigee_km']}")
    print(f"  apogee_km: {elements['apogee_km']}")
    print()


def run_case(tle: TLECase) -> None:
    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    epoch_dt = datetime.fromtimestamp(
        (sat.jdsatepoch + sat.jdsatepochF - 2440587.5) * 86400.0,
        tz=timezone.utc,
    )

    observations, truth_states = make_observations_from_tle(tle, epoch_dt)
    _, truth_r_km, truth_v_km_s = truth_states[len(truth_states) // 2]

    solution = IODSolver().solve(observations, uuid4())

    print("=" * 80)
    print(f"IOD propagation validation case: {tle.name}")
    print(f"TLE epoch: {epoch_dt.isoformat()}")
    print(f"Observations: {len(observations)}")
    print()

    print_elements("Truth middle state:", truth_r_km, truth_v_km_s)

    print("IOD solution:")
    print(f"  success: {solution.success}")

    if not solution.success:
        print(f"  error_message: {solution.error_message}")
        print("  attempted_methods:")
        for attempt in solution.attempted_methods or []:
            print(f"    - {attempt}")
        print()
        return

    print(f"  method_used: {solution.method_used}")
    print(f"  rms_residual_arcsec: {solution.rms_residual_arcsec}")
    print_elements("Recovered IOD state:", solution.position_km, solution.velocity_km_s)

    pos_error_km = norm(solution.position_km - truth_r_km)
    vel_error_km_s = norm(solution.velocity_km_s - truth_v_km_s)

    print("Truth comparison:")
    print(f"  position_error_km: {pos_error_km}")
    print(f"  velocity_error_km_s: {vel_error_km_s}")
    print()

    propagated_positions, propagated_velocities, period_s = propagate_one_period_rk4(
        solution.position_km,
        solution.velocity_km_s,
        step_s=10.0,
    )

    final_r_km = propagated_positions[-1]
    final_v_km_s = propagated_velocities[-1]

    print("RK4 + J2 + drag propagation summary:")
    print(f"  propagated_duration_s: {period_s}")
    print(f"  propagated_duration_min: {period_s / 60.0}")
    print(f"  final_position_km: {final_r_km}")
    print(f"  final_velocity_km_s: {final_v_km_s}")
    print(f"  final_radius_km: {norm(final_r_km)}")

    plot_path = plot_validation_case(
        case_name=tle.name,
        truth_states=truth_states,
        observations=observations,
        solution_r_km=solution.position_km,
        propagated_positions_km=propagated_positions,
        output_dir=Path("outputs"),
    )

    print(f"  plot_path: {plot_path}")
    print()


def main() -> None:
    cases = [
        TLECase(
            name="NOAA 15 LEO",
            line1=(
                "1 25338U 98030A   24001.50000000  .00000092  "
                "00000-0  80852-4 0  9997"
            ),
            line2=(
                "2 25338  98.7342  40.3121 0010732  95.2104 "
                "265.0298 14.25900000300000"
            ),
            offsets_s=[0.0, 60.0, 120.0, 180.0, 240.0],
        ),
        TLECase(
            name="ISS LEO",
            line1=(
                "1 25544U 98067A   24001.50000000  .00016717  "
                "00000-0  10270-3 0  9000"
            ),
            line2=(
                "2 25544  51.6416 247.4627 0006703 130.5360 "
                "325.0288 15.49514239000010"
            ),
            offsets_s=[0.0, 60.0, 120.0, 180.0, 240.0],
        ),
        TLECase(
            name="Molniya HEO",
            line1=(
                "1 44249U 19035A   24001.50000000 -.00000273  "
                "00000-0  00000-0 0  9990"
            ),
            line2=(
                "2 44249  63.4521 206.7395 7141832 268.5743 "
                "16.0050  2.00612897000017"
            ),
            offsets_s=[0.0, 60.0, 120.0, 180.0, 240.0],
        ),
    ]

    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()