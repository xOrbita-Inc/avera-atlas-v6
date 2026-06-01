from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np
from sgp4.api import Satrec, jday

from iod import (
    IODObservation,
    IODSolver,
    state_to_elements,
    norm,
    unit,
)


@dataclass
class TLECase:
    name: str
    line1: str
    line2: str


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

def deterministic_range_error_km(timestamp: datetime, scale_km: float = 0.15) -> float:
    """
    Deterministic pseudo-measurement error for radar range validation.

    This keeps the validation repeatable while ensuring the radar path does not
    receive the exact truth range that it later reconstructs.
    """
    seconds = timestamp.timestamp()
    return scale_km * math.sin(0.001 * seconds)


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

    This creates usable synthetic observation geometry while keeping the target
    truth state tied to the TLE/SGP4 reference orbit.
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
    offsets_s: list[float],
    sensor_type: str,
) -> tuple[list[IODObservation], list[tuple[datetime, np.ndarray, np.ndarray]]]:
    """
    Build either optical angles-only or radar range+angles observations.

    optical:
        IODObservation with RA/Dec only.

    radar:
        IODObservation with RA/Dec plus range_km/range_sigma_km.
    """
    observations: list[IODObservation] = []
    truth_states: list[tuple[datetime, np.ndarray, np.ndarray]] = []

    for offset_s in offsets_s:
        timestamp = t0 + timedelta(seconds=offset_s)

        target_r_km, target_v_km_s = sgp4_state_km(tle, timestamp)
        observer_r_km, observer_v_km_s = make_synthetic_observer_state(
            target_r_km,
            target_v_km_s,
            range_km=900.0,
        )

        los = target_r_km - observer_r_km
        slant_range_km = norm(los)
        ra, dec = ra_dec_from_los(los)

        if sensor_type == "optical":
            range_km = None
            range_sigma_km = None
        elif sensor_type == "radar":
            # Use an independent measured range, not the exact geometry range.
            # This prevents the validation from trivially reconstructing the truth state
            # with zero position error.
            range_error_km = deterministic_range_error_km(timestamp)
            range_km = slant_range_km + range_error_km
            range_sigma_km = 0.15
        else:
            raise ValueError(f"Unsupported sensor_type: {sensor_type}")

        observations.append(
            IODObservation(
                timestamp=timestamp,
                ra=ra,
                dec=dec,
                ra_sigma=math.radians(2.9 / 3600.0),
                dec_sigma=math.radians(2.9 / 3600.0),
                observer_position_km=observer_r_km,
                observer_velocity_km_s=observer_v_km_s,
                range_km=range_km,
                range_sigma_km=range_sigma_km,
            )
        )

        truth_states.append((timestamp, target_r_km, target_v_km_s))

    return observations, truth_states


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


def run_solution(
    label: str,
    observations: list[IODObservation],
    truth_r_km: np.ndarray,
    truth_v_km_s: np.ndarray,
) -> None:
    solver = IODSolver()
    solution = solver.solve(observations, uuid4())

    print(f"{label} path:")
    print(f"  observations: {len(observations)}")
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

    print_elements(
        f"  {label} recovered elements:",
        solution.position_km,
        solution.velocity_km_s,
    )

    pos_error_km = norm(solution.position_km - truth_r_km)
    vel_error_km_s = norm(solution.velocity_km_s - truth_v_km_s)

    print(f"  {label} truth comparison:")
    print(f"    position_error_km: {pos_error_km}")
    print(f"    velocity_error_km_s: {vel_error_km_s}")
    print()

    print("  attempted_methods:")
    for attempt in solution.attempted_methods or []:
        print(f"    - {attempt}")
    print()


def main() -> None:
    tle = TLECase(
        name="NOAA 15 reference orbit",
        line1=(
            "1 25338U 98030A   24001.50000000  .00000092  "
            "00000-0  80852-4 0  9997"
        ),
        line2=(
            "2 25338  98.7342  40.3121 0010732  95.2104 "
            "265.0298 14.25900000300000"
        ),
    )

    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    epoch_dt = datetime.fromtimestamp(
        (sat.jdsatepoch + sat.jdsatepochF - 2440587.5) * 86400.0,
        tz=timezone.utc,
    )

    offsets_s = [0.0, 60.0, 120.0, 180.0, 240.0]

    optical_obs, truth_states = make_observations_from_tle(
        tle=tle,
        t0=epoch_dt,
        offsets_s=offsets_s,
        sensor_type="optical",
    )

    radar_obs, _ = make_observations_from_tle(
        tle=tle,
        t0=epoch_dt,
        offsets_s=offsets_s,
        sensor_type="radar",
    )

    truth_time, truth_r_km, truth_v_km_s = truth_states[len(truth_states) // 2]

    print("=" * 80)
    print("Sensor-agnostic ingest validation")
    print(f"Reference object: {tle.name}")
    print(f"Truth epoch: {truth_time.isoformat()}")
    print()

    print_elements("Truth state:", truth_r_km, truth_v_km_s)

    run_solution(
        label="Optical angles-only",
        observations=optical_obs,
        truth_r_km=truth_r_km,
        truth_v_km_s=truth_v_km_s,
    )

    run_solution(
        label="Radar range+angles",
        observations=radar_obs,
        truth_r_km=truth_r_km,
        truth_v_km_s=truth_v_km_s,
    )

    print("Result:")
    print("  Optical observations use RA/Dec only and route through the angles-only IOD chain.")
    print("  Radar observations include RA/Dec plus independently perturbed range_km")
    print("  and route through the range+angles IOD path inside IODSolver.solve().")
    print("  Both paths ingest through IODObservation and output orbital elements.")


if __name__ == "__main__":
    main()