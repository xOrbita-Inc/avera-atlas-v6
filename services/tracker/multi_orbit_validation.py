from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np
from sgp4.api import Satrec

from iod import IODObservation, IODSolver, state_to_elements, norm
from sensor_agnostic_ingest import (
    TLECase,
    sgp4_state_km,
    make_observations_from_tle,
)


CASES = [
    TLECase(
        name="NOAA 15 (sun-sync LEO, near-circular)",
        line1="1 25338U 98030A   24001.50000000  .00000092  00000-0  80852-4 0  9997",
        line2="2 25338  98.7342  40.3121 0010732  95.2104 265.0298 14.25900000300000",
    ),
    TLECase(
        name="ISS (low-incl LEO, near-circular)",
        line1="1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9000",
        line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49514239000010",
    ),
    TLECase(
        name="Molniya-type (HEO, high eccentricity)",
        line1="1 44249U 19035A   24001.50000000 -.00000273  00000-0  00000-0 0  9990",
        line2="2 44249  63.4521 206.7395 7141832 268.5743  16.0050  2.00612897000017",
    ),
]

OFFSET_SETS = {
    "4-min arc (60s steps)": [0.0, 60.0, 120.0, 180.0, 240.0],
    "10-min arc (150s steps)": [0.0, 150.0, 300.0, 450.0, 600.0],
}


def evaluate(tle: TLECase, offsets, sensor_type: str):
    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    epoch_dt = datetime.fromtimestamp(
        (sat.jdsatepoch + sat.jdsatepochF - 2440587.5) * 86400.0,
        tz=timezone.utc,
    )
    obs, truth_states = make_observations_from_tle(tle, epoch_dt, offsets, sensor_type)
    _, truth_r, truth_v = truth_states[len(truth_states) // 2]

    sol = IODSolver().solve(obs, uuid4())
    if not sol.success:
        return {"success": False, "error": sol.error_message,
                "attempts": [(a["method"], a["success"]) for a in (sol.attempted_methods or [])]}

    pos_err = norm(sol.position_km - truth_r)
    vel_err = norm(sol.velocity_km_s - truth_v)
    truth_el = state_to_elements(truth_r, truth_v)
    return {
        "success": True,
        "method": sol.method_used,
        "rms_arcsec": sol.rms_residual_arcsec,
        "pos_err_km": pos_err,
        "vel_err_km_s": vel_err,
        "a_est": sol.semi_major_axis_km,
        "a_truth": truth_el["semi_major_axis_km"],
        "e_est": sol.eccentricity,
        "e_truth": truth_el["eccentricity"],
        "i_est": sol.inclination_deg,
        "i_truth": truth_el["inclination_deg"],
        "attempts": [(a["method"], a["success"]) for a in (sol.attempted_methods or [])],
    }


def main():
    for tle in CASES:
        print("=" * 78)
        print(tle.name)
        print("=" * 78)
        for arc_label, offsets in OFFSET_SETS.items():
            for sensor in ("optical", "radar"):
                r = evaluate(tle, offsets, sensor)
                tag = f"  [{arc_label}] {sensor:7s}"
                if not r["success"]:
                    print(f"{tag} FAILED: {r['error']}")
                    print(f"            attempts: {r['attempts']}")
                    continue
                print(f"{tag} method={r['method']:10s} "
                      f"pos_err={r['pos_err_km']:9.2f} km  vel_err={r['vel_err_km_s']:.4f} km/s")
                print(f"            a: est {r['a_est']:.1f} vs truth {r['a_truth']:.1f} km   "
                      f"e: est {r['e_est']:.4f} vs truth {r['e_truth']:.4f}   "
                      f"i: est {r['i_est']:.2f} vs truth {r['i_truth']:.2f} deg")
                if sensor == "optical":
                    print(f"            optical attempts: {r['attempts']}")
        print()


if __name__ == "__main__":
    main()
