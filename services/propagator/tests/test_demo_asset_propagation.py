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

This module is a regression guard. If these assertions fail, the relative-geometry
invariant has been broken, not "fixed" -- investigate rather than updating the
numbers.

The four preset scenarios are built in-code here (geometry + deterministic
confidences) to mirror ``services/ui/app/main.py::_generate_synthetic_scenario``.
Input states are generated deterministically rather than committed as ``.npz``
fixtures, which the repo gitignores.

SCRUM-390: why BASELINE moved, once
-----------------------------------
The original BASELINE was captured from the pre-change demo path while
``compute_pc`` was still running its broken default mode, which overestimated Pc
by about 4.2554x. Correcting the Pc changed the input to ``pc_to_risk_level``, so
the risk labels moved. The relative-geometry invariant did not move: miss
distances, TCA indices and relative velocities are all bit-identical, and the
three geometry tests below still pass untouched. Only the labels derived from Pc
changed, which is the change the fix was supposed to produce.

The baseline now pins Pc itself, not just the label it falls into. A colour
bucket spans a decade, so the old label-only baseline would have accepted a
fresh 4x error without complaint. It did, for as long as the defect was live.

What the corrected numbers say about the presets
------------------------------------------------
No preset reaches RED any more, and none can. ``PC_RED_THRESHOLD`` is 1e-4, and
at the demo's own inputs -- ``HBR_M`` 15 m, ``DEFAULT_DEBRIS_UNCERTAINTY_M``
2000 m scaled by the seeded confidences -- the encounter-plane sigma is roughly
1.1 km, which caps Pc at about 9.7e-5 even for an object placed exactly on the
asset. That is a covariance ceiling, not a geometry shortfall, so moving the
objects closer cannot restore RED; ``test_red_is_unreachable_at_the_demo_covariance``
asserts the ceiling so that nobody tries. Restoring a RED demo case needs a real
covariance for the secondary, which is tracked separately.
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

# Baseline captured on these exact deterministic states. TCA indices are the
# pre-change values and have never moved. Risk and Pc were recaptured under
# SCRUM-390, once, for the reason recorded in the module docstring.
BASELINE = {
    "nominal": {
        "risk": ["NOMINAL", "NOMINAL", "NOMINAL"],
        "tca": [17, 17, 20],
        "pc": [0.000000e+00, 2.639157e-194, 0.000000e+00],
    },
    "warning": {
        "risk": ["AMBER", "GREEN", "GREEN", "GREEN"],
        "tca": [17, 17, 17, 17],
        "pc": [1.052580e-05, 1.063763e-06, 2.241367e-06, 4.889303e-06],
    },
    "critical": {
        "risk": ["AMBER", "AMBER", "AMBER", "AMBER"],
        "tca": [17, 17, 17, 17],
        "pc": [7.018983e-05, 7.371553e-05, 4.822367e-05, 7.224133e-05],
    },
    "mixed": {
        "risk": ["AMBER", "GREEN", "GREEN", "AMBER", "NOMINAL"],
        "tca": [17, 17, 17, 17, 17],
        "pc": [6.809577e-05, 4.155794e-06, 2.658082e-07, 2.858628e-05, 0.000000e+00],
    },
}

# Miss distance in metres at TCA, per preset, from the same run. Pinned because
# it is the one quantity the SCRUM-370 refactor was required to leave alone, and
# because it is what makes the Pc figures above checkable by hand.
BASELINE_MISS_M = {
    "nominal":  [50159.7448, 31764.7603, 80156.0977],
    "warning":  [2435.1591, 3231.0989, 3061.0456, 2773.0849],
    "critical": [400.6245, 801.5610, 1240.9674, 608.2763],
    "mixed":    [502.5933, 2700.0000, 3848.3763, 1700.0000, 50159.7448],
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
    baseline exactly."""
    d = _run_demo_preset(preset, tmp_path)
    risk = [str(x) for x in d["risk_levels"]]
    tca = [int(x) for x in d["tca_indices"]]
    assert risk == BASELINE[preset]["risk"], (
        f"{preset}: risk levels moved {BASELINE[preset]['risk']} -> {risk}"
    )
    assert tca == BASELINE[preset]["tca"], (
        f"{preset}: TCA indices moved {BASELINE[preset]['tca']} -> {tca}"
    )


@pytest.mark.parametrize("preset", ["nominal", "warning", "critical", "mixed"])
def test_demo_preset_pc_values_unchanged(preset, tmp_path):
    """Pin Pc itself, not the colour it falls into.

    A risk label covers a decade of Pc, so a label-only baseline accepts a
    multiplicative error silently. That is exactly how the 4.2554x default-mode
    error in SCRUM-390 survived: every preset kept its label while the number
    behind it was wrong.
    """
    d = _run_demo_preset(preset, tmp_path)
    pc = [float(x) for x in d["pc_values"]]
    expected = BASELINE[preset]["pc"]
    assert pc == pytest.approx(expected, rel=1e-5, abs=1e-12), (
        f"{preset}: Pc moved {expected} -> {pc}"
    )


@pytest.mark.parametrize("preset", ["nominal", "warning", "critical", "mixed"])
def test_demo_preset_miss_distance_unchanged(preset, tmp_path):
    """Miss distance is the invariant the SCRUM-370 refactor had to preserve,
    and it is unaffected by any Pc change. It must never move."""
    d = _run_demo_preset(preset, tmp_path)
    miss_m = [float(x) * 1000.0 for x in d["ca_table"]]
    assert miss_m == pytest.approx(BASELINE_MISS_M[preset], rel=1e-6), (
        f"{preset}: miss distances moved {BASELINE_MISS_M[preset]} -> {miss_m}"
    )


def test_red_is_unreachable_at_the_demo_covariance(tmp_path):
    """SCRUM-390. RED is out of reach for the demo presets by construction, so
    nobody should try to recover it by moving objects closer.

    Place an object essentially on top of the asset, which is the most dangerous
    geometry the preset format can express, and show it still lands short of
    PC_RED_THRESHOLD. The limit is the covariance: HBR_M is 15 m while the
    encounter-plane sigma is about 1.1 km, and Pc for a near-zero miss is
    approximately HBR^2 / (2 sigma_x sigma_z), which no amount of closing the
    miss distance can raise.

    If this test starts failing, something changed HBR_M, the debris uncertainty
    model, or the thresholds. Any of those is worth reading about before the
    baseline above is touched.
    """
    # Sub-metre miss, TCA on the 60 s sample grid so the geometry is exact.
    on_top = [(0.02 * 1020.0, 0.0005, 0.0, 0.02)]
    _PRESETS["_ceiling"] = on_top
    _LABEL["_ceiling"] = "CEIL"
    try:
        d = _run_demo_preset("_ceiling", tmp_path)
    finally:
        del _PRESETS["_ceiling"], _LABEL["_ceiling"]

    miss_m = float(d["ca_table"][0]) * 1000.0
    pc = float(d["pc_values"][0])
    assert miss_m < 1.0, f"probe did not close the miss: {miss_m:.3f} m"
    assert pc < propagator_main.PC_RED_THRESHOLD, (
        f"RED is now reachable at the demo covariance: Pc={pc:.4e} at a "
        f"{miss_m:.3f} m miss. Read the SCRUM-390 note in this module's "
        f"docstring before updating BASELINE."
    )
    # And it is close to the threshold, not orders below, which is why the
    # original inflated numbers read as RED so convincingly.
    assert pc > 0.5 * propagator_main.PC_RED_THRESHOLD, (
        f"ceiling moved far from the threshold: Pc={pc:.4e}"
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
