from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from observation_loader import load_observations_multi, to_iod_observations
from iod import IODSolver


def norm(v: np.ndarray) -> float:
    return np.linalg.norm(v)


def main() -> None:
    obs_path = "services/sandbox/sandbox_observations_multi.npz"
    truth_path = None  

    batch = load_observations_multi(obs_path)
    iod_observations = to_iod_observations(batch)

    print("Converted observations for IOD:")
    print(f"Count: {len(iod_observations)}")
    print()

    solver = IODSolver()
    solution = solver.solve(iod_observations, uuid4())

    print("IOD solution:")
    print(f"success: {solution.success}")
    if not solution.success:
        print(f"error_message: {solution.error_message}")
        print("attempted_methods:")
        for attempt in solution.attempted_methods or []:
            print(f"  - {attempt}")
        return

    print(f"epoch: {solution.epoch.isoformat()}")
    print(f"method_used: {solution.method_used}")
    print(f"position_km: {solution.position_km}")
    print(f"velocity_km_s: {solution.velocity_km_s}")
    print(f"semi_major_axis_km: {solution.semi_major_axis_km}")
    print(f"eccentricity: {solution.eccentricity}")
    print(f"inclination_deg: {solution.inclination_deg}")
    print(f"raan_deg: {solution.raan_deg}")
    print(f"arg_perigee_deg: {solution.arg_perigee_deg}")
    print(f"true_anomaly_deg: {solution.true_anomaly_deg}")
    print(f"rms_residual_arcsec: {solution.rms_residual_arcsec}")
    print()

    print("attempted_methods:")
    for attempt in solution.attempted_methods or []:
        print(f"  - {attempt}")
    print()

    if truth_path is None:
        return

    truth_file = Path(truth_path)
    if not truth_file.exists():
        print(f"Truth file not found, skipping truth comparison: {truth_file}")
        return

    truth = json.loads(truth_file.read_text())
    truth_mid = truth["middle_observation_truth"]

    truth_position_km = np.array(truth_mid["target_position_km"], dtype=np.float64)
    truth_velocity_km_s = np.array(truth_mid["target_velocity_km_s"], dtype=np.float64)

    pos_err_km = norm(solution.position_km - truth_position_km)
    vel_err_km_s = norm(solution.velocity_km_s - truth_velocity_km_s)

    print("Truth comparison:")
    print(f"truth_position_km: {truth_position_km}")
    print(f"truth_velocity_km_s: {truth_velocity_km_s}")
    print(f"position_error_km: {pos_err_km}")
    print(f"velocity_error_km_s: {vel_err_km_s}")


if __name__ == "__main__":
    main()