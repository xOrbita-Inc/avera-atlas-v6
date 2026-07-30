"""AVERA-ATLAS demo scenario writer.

Writes ``states_multi.npz`` for one demo scenario so the propagator can pick it
up. The same thing the UI's ``POST /api/scenarios/run`` endpoint does, from a
terminal.

Usage:
    python demo/demo_scenarios.py [scenario]

Scenarios: nominal, warning, critical, mixed, demo. Default is mixed.

SCRUM-392: what changed and why
-------------------------------
This file used to be a third, independent scenario generator. It had its own
object names, its own miss distances in km, unseeded ``np.random`` for both the
relative velocity direction and the approach speed, and its own risk prediction
computed from hardcoded miss-distance bands: under 0.1 km printed RED, under
1.0 km printed AMBER. That rule never touched ``compute_pc`` or
``pc_to_risk_level``, so the "Expected Outcomes" table it printed next to real
propagator output was not a prediction of anything the pipeline would produce.
It also disagreed with the UI's presets, which are what the demo actually shows.

Now it builds from ``services/ui/app/demo_presets.py``, the single definition,
and it reports geometry only. Risk is determined downstream by the propagator
from Pc, and this script does not guess at it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DP_PATH = _REPO_ROOT / "services" / "ui" / "app" / "demo_presets.py"
_spec = importlib.util.spec_from_file_location("demo_presets", _DP_PATH)
demo_presets = importlib.util.module_from_spec(_spec)
# Registered before exec because the module defines dataclasses, and @dataclass
# resolves its own __module__ out of sys.modules during class-body processing.
sys.modules["demo_presets"] = demo_presets
_spec.loader.exec_module(demo_presets)

ARTIFACT_NAME = "states_multi.npz"


def _resolve_data_dir() -> str:
    """DATA_DIR env var, then the Docker volume, then a local data/ folder."""
    env_dir = os.getenv("DATA_DIR")
    if env_dir and os.path.exists(os.path.dirname(env_dir)):
        return env_dir
    if os.path.exists("/data"):
        return "/data/planner_artifacts"
    return str(Path(__file__).resolve().parent / "data")


def write_scenario(scenario: str, output_dir: str | None = None) -> str:
    obj_ids, r_list, v_list, confidences = demo_presets.build_scenario(scenario)

    out_dir = output_dir or _resolve_data_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, ARTIFACT_NAME)

    np.savez(
        out_path,
        object_ids=np.array(obj_ids),
        r_eci_km=np.array(r_list, dtype=float),
        v_eci_km_s=np.array(v_list, dtype=float),
        confidences=np.array(confidences, dtype=float),
        t_window=np.array([demo_presets.SAMPLE_DT_S, demo_presets.N_STEPS]),
        metadata=json.dumps({
            "source": "demo_scenarios_cli",
            "scenario": scenario,
            "t0": datetime.utcnow().isoformat(),
            "asset_state": {
                "r_eci_km": demo_presets.ASSET_R_ECI_KM,
                "v_eci_km_s": demo_presets.ASSET_V_ECI_KM_S,
            },
        }),
    )

    specs = demo_presets.specs_for(scenario)
    print()
    print("=" * 68)
    print(f"DEMO SCENARIO: {scenario.upper()}")
    print("=" * 68)
    print(f"Written to: {out_path}")
    print()
    print("Geometry (what the preset asks for):")
    print("-" * 68)
    print(f"  {'object':18} {'miss at TCA':>14}  {'TCA':>11}  {'confidence':>10}")
    for obj_id, s in zip(obj_ids, specs):
        tca_min = s.t_star_s / 60.0
        print(f"  {obj_id:18} {s.miss_km * 1000:11.1f} m  T+{tca_min:6.1f}min  "
              f"{s.confidence:10.3f}")
    print("-" * 68)
    print("Risk level is not predicted here. The propagator computes Pc from this")
    print("geometry and the covariance model, and pc_to_risk_level assigns the")
    print("band. Run the propagator to see it.")
    print("=" * 68)
    print()
    return out_path


def main() -> int:
    scenario = sys.argv[1] if len(sys.argv) > 1 else demo_presets.DEFAULT_SCENARIO
    if scenario not in demo_presets.SCENARIOS:
        print(f"Unknown scenario '{scenario}'. "
              f"Choose one of: {', '.join(demo_presets.SCENARIOS)}", file=sys.stderr)
        return 2
    write_scenario(scenario)
    print("Scenario ready. Run the propagator to process it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
