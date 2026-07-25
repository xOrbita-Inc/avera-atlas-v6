"""SCRUM-370: the demo asset must ride its real Keplerian orbit while the
relative geometry of every preset is preserved exactly.

Background. In the demo path of ``propagate_and_screen`` the asset and debris
were previously both propagated with a straight-line model, which flew the asset
hundreds of thousands of km off-orbit over the conjunction window and corrupted
the absolute state later fed to the planner. The fix propagates the asset with
``kepler_propagate`` and carries each debris as that Keplerian asset track plus a
constant relative offset ``rel0 + vrel * t``. Because miss distance, TCA index,
and the 2D Pc depend only on relative position/velocity (plus the fixed diagonal
covariances), no preset's miss / TCA / risk / badge may move.

This module is a regression guard. ``BASELINE`` below was captured from the
pre-change (linear-everything) demo path. If these assertions fail, the
relative-geometry invariant has been broken, not "fixed" -- investigate rather
than updating the numbers.

The four preset scenarios are built in-code here (geometry + deterministic
confidences) to mirror ``services/ui/app/main.py::_generate_synthetic_scenario``.
Input states are generated deterministically rather than committed as ``.npz``
fixtures, which the repo gitignores.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# The propagator service module is named 'main', which collides with the other
# services' main.py when the full repo suite runs. Load it by explicit path under
# a unique module name so this test is import-order independent (SCRUM-351).
_PROP_ROOT = Path(__file__).resolve().parents[1]
if str(_PROP_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROP_ROOT))
_spec = importlib.util.spec_from_file_location("propagator_main", _PROP_ROOT / "main.py")
propagator_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(propagator_main)

# --- Preset definitions, mirroring the UI synthetic-scenario generator ---------
_MU = 398600.4418          # km^3/s^2
_R_EARTH = 6371.0          # km
_R_MAG = _R_EARTH + 500.0  # 6871 km circular asset
_V_CIRC = float(np.sqrt(_MU / _R_MAG))
_ASSET_R = [_R_MAG, 0.0, 0.0]
_ASSET_V = [0.0, _V_CIRC, 0.0]
_SEED = 42  # matches the confidences captured in BASELINE

# (start_dist_km, miss_y_km, miss_z_km, v_approach_km_s) per debris object.
_PRESETS = {
    "nominal":  [(200.0, 50.0, 0.0, 0.2), (150.0, 30.0, 10.0, 0.15), (300.0, 80.0, 5.0, 0.25)],
    "warning":  [(50.0, 2.2, 0.3, 0.05), (80.0, 2.8, 0.2, 0.08), (30.0, 3.0, 0.1, 0.03), (60.0, 2.5, 0.0, 0.06)],
    "critical": [(20.0, 0.02, 0.01, 0.02), (40.0, 0.05, 0.0, 0.04), (60.0, 0.3, 0.1, 0.06), (30.0, 0.1, 0.0, 0.03)],
    "mixed":    [(25.0, 0.05, 0.01, 0.025), (50.0, 2.5, 0.2, 0.05), (80.0, 3.5, 0.0, 0.08), (40.0, 1.5, 0.0, 0.04), (200.0, 50.0, 0.0, 0.2)],
}
_LABEL = {"nominal": "NOM", "warning": "WRN", "critical": "CRT", "mixed": "MIX"}

# Pre-change baseline (linear-everything demo path), captured on these exact
# deterministic states before the SCRUM-370 refactor. The refactor must preserve
# these values.
BASELINE = {
    "nominal":  {"risk": ["NOMINAL", "NOMINAL", "NOMINAL"], "tca": [17, 17, 20]},
    "warning":  {"risk": ["AMBER", "GREEN", "GREEN", "AMBER"], "tca": [17, 17, 17, 17]},
    "critical": {"risk": ["RED", "RED", "RED", "RED"], "tca": [17, 17, 17, 17]},
    "mixed":    {"risk": ["RED", "AMBER", "GREEN", "RED", "NOMINAL"], "tca": [17, 17, 17, 17, 17]},
}


def _write_states(preset: str, data_dir: Path) -> dict:
    """Write a deterministic states_multi.npz for one preset and return its arrays."""
    objs = _PRESETS[preset]
    asset_r0 = np.array(_ASSET_R, dtype=float)
    asset_v0 = np.array(_ASSET_V, dtype=float)
    r_list = np.array([[_ASSET_R[0] + sd, _ASSET_R[1] + my, _ASSET_R[2] + mz] for sd, my, mz, va in objs])
    v_list = np.array([[_ASSET_V[0] - va, _ASSET_V[1], _ASSET_V[2]] for sd, my, mz, va in objs])
    ids = np.array([f"OBJ-{_LABEL[preset]}-{i:03d}" for i in range(len(objs))])
    # RandomState(SEED) reproduces np.random.seed(SEED); np.random.uniform(...)
    conf = np.random.RandomState(_SEED).uniform(0.75, 0.98, len(objs))
    np.savez(
        data_dir / "states_multi.npz",
        object_ids=ids, r_eci_km=r_list, v_eci_km_s=v_list, confidences=conf,
        t_window=np.array([60.0, 1440]),
        metadata=json.dumps({
            "source": "test", "scenario": preset,
            "asset_state": {"r_eci_km": _ASSET_R, "v_eci_km_s": _ASSET_V},
        }),
    )
    return {"r": r_list, "v": v_list, "asset_r0": asset_r0, "asset_v0": asset_v0}


def _run_demo_preset(preset: str, data_dir: Path):
    """Build a preset's states and run it through the demo path; return output npz."""
    _write_states(preset, data_dir)
    # propagate_and_screen reads the module-global DATA_DIR at call time.
    propagator_main.DATA_DIR = str(data_dir)
    propagator_main.propagate_and_screen()
    return np.load(data_dir / "prop_multi.npz", allow_pickle=True)


@pytest.mark.parametrize("preset", ["nominal", "warning", "critical", "mixed"])
def test_demo_preset_risk_and_tca_unchanged(preset, tmp_path):
    """Every preset's per-object risk level and TCA index must match the
    pre-change baseline exactly."""
    d = _run_demo_preset(preset, tmp_path)
    risk = [str(x) for x in d["risk_levels"]]
    tca = [int(x) for x in d["tca_indices"]]
    assert risk == BASELINE[preset]["risk"], (
        f"{preset}: risk levels moved {BASELINE[preset]['risk']} -> {risk}"
    )
    assert tca == BASELINE[preset]["tca"], (
        f"{preset}: TCA indices moved {BASELINE[preset]['tca']} -> {tca}"
    )


def test_demo_schema_unchanged(tmp_path):
    """The saved prop_multi.npz must keep the fields the UI/planner consume."""
    d = _run_demo_preset("critical", tmp_path)
    for key in ("r_asset", "v_asset", "r_objects", "v_objects",
                "ca_table", "risk_levels", "tca_indices"):
        assert key in d.files, f"missing output field: {key}"


def test_demo_asset_stays_on_orbit(tmp_path):
    """The demo asset must stay near-LEO and roughly constant in radius (the
    presets use a circular asset), not drift off-orbit as the pre-change linear
    propagation did (which reached ~6.5e5 km)."""
    d = _run_demo_preset("critical", tmp_path)
    mag = np.linalg.norm(d["r_asset"], axis=1)
    assert mag.min() > 6000.0 and mag.max() < 8000.0, (
        f"asset left LEO band: min={mag.min():.1f} max={mag.max():.1f} km"
    )
    # Circular asset -> radius essentially constant across the window.
    assert (mag.max() - mag.min()) < 1.0, (
        f"asset radius not constant: spread={mag.max() - mag.min():.3f} km"
    )


def test_demo_relative_motion_invariant(tmp_path):
    """r_debris(t) - r_asset(t) must equal rel0 + vrel*t exactly, and
    v_debris(t) - v_asset(t) must be the constant vrel, identical to the old
    linear relative model."""
    states = _write_states("critical", tmp_path)
    propagator_main.DATA_DIR = str(tmp_path)
    propagator_main.propagate_and_screen()
    d = np.load(tmp_path / "prop_multi.npz", allow_pickle=True)

    times = np.arange(1440) * 60.0
    for i in range(len(states["r"])):
        rel0 = states["r"][i] - states["asset_r0"]
        vrel = states["v"][i] - states["asset_v0"]
        expected_rel = rel0 + np.outer(times, vrel)
        actual_rel = d["r_objects"][i] - d["r_asset"]
        assert np.allclose(actual_rel, expected_rel, atol=1e-6), (
            f"object {i}: relative position deviates from rel0 + vrel*t"
        )
        actual_vrel = d["v_objects"][i] - d["v_asset"]
        assert np.allclose(actual_vrel, vrel, atol=1e-9), (
            f"object {i}: relative velocity is not constant vrel"
        )
