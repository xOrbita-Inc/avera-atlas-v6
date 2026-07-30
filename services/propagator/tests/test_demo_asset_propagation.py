"""SCRUM-370: the demo asset must ride its real Keplerian orbit while the
relative geometry of every preset is preserved exactly.

Background. In the demo path of ``propagate_and_screen`` the asset and debris
were previously both propagated with a straight-line model, which flew the asset
hundreds of thousands of km off-orbit over the conjunction window and corrupted
the absolute state later fed to the planner. The fix propagates the asset with
``kepler_propagate`` and carries each debris as that Keplerian asset track plus a
constant relative offset ``rel0 + vrel * t``. Because miss distance, TCA index,
and the 2D Pc depend only on relative position/velocity (plus the fixed diagonal
covariances), the relative-geometry invariant must hold across any change here.

Where the presets come from
---------------------------
SCRUM-392. This module no longer carries its own copy of the preset geometry. It
imports ``services/ui/app/demo_presets.py``, which is the definition the live UI
scenario endpoint builds from. Before that, the geometry existed as two
hand-copied literals that happened to agree, so tuning the live preset would
have left this baseline silently describing a demo that no longer existed. The
``demo`` scenario is now covered here too; it previously had no test at all.

BASELINE changelog
------------------
Every entry is a deliberate recapture with its reason. Do not add a fourth
without one. A baseline that moves silently stops being a tripwire and teaches
the next reader to update the numbers instead of investigating them.

- Original: captured from the pre-SCRUM-370 demo path (linear everything), to
  prove the Keplerian-asset refactor preserved relative geometry.
- 2026-07-29, SCRUM-390: risk labels recaptured. ``compute_pc``'s default mode
  had been overestimating Pc by 4.25539x, so correcting it changed the input to
  ``pc_to_risk_level``. Geometry did not move; only the labels derived from Pc
  did. Pc itself was added to the baseline at this point, because a colour
  bucket spans a decade and the label-only baseline had accepted the 4x error in
  silence for as long as it was live.
- 2026-07-30, SCRUM-392: miss distances and Pc recaptured. Every preset's time of
  closest approach was off the 60 s sample grid, so the propagator's argmin
  landed 20 s away from true TCA and the leftover separation along the approach
  axis dominated the reported miss. The critical preset's first object asked for
  a 22 m miss and this test recorded 400 m. Aligning TCA onto the grid makes
  stated and produced miss agree exactly. Risk labels did not move.

The covariance ceiling
----------------------
No preset reaches RED, and none can. ``PC_RED_THRESHOLD`` is 1e-4, and at the
demo's own inputs (``HBR_M`` 15 m, ``DEFAULT_DEBRIS_UNCERTAINTY_M`` 2000 m scaled
by confidence) the encounter-plane sigma is roughly 1.1 km, which caps Pc at
about 9.7e-5 even for an object placed on the asset. That is a covariance
ceiling, not a geometry shortfall, so moving objects closer cannot restore RED.
``test_red_is_unreachable_at_the_demo_covariance`` asserts it. SCRUM-391 carries
the real fix, a per-object covariance in the scenario spec.
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

# SCRUM-392: the live preset definitions, loaded from the UI service by path.
# Registered in sys.modules before exec because the module defines dataclasses,
# and @dataclass resolves its own __module__ out of sys.modules while the class
# body is being processed.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DP_PATH = _REPO_ROOT / "services" / "ui" / "app" / "demo_presets.py"
_dp_spec = importlib.util.spec_from_file_location("demo_presets", _DP_PATH)
demo_presets = importlib.util.module_from_spec(_dp_spec)
sys.modules["demo_presets"] = demo_presets
_dp_spec.loader.exec_module(demo_presets)

ALL_SCENARIOS = list(demo_presets.PRESETS) + ["demo"]

BASELINE = {
    "nominal": {
        "risk": ["NOMINAL", "NOMINAL", "NOMINAL"],
        "tca": [17, 17, 20],
        "pc": [0.000000e+00, 1.295545e-192, 0.000000e+00],
        "miss_m": [50000.0000, 31622.7766, 80156.0977],
    },
    "warning": {
        "risk": ["AMBER", "GREEN", "GREEN", "GREEN"],
        "tca": [17, 17, 17, 17],
        "pc": [1.462442e-05, 3.219644e-06, 2.580999e-06, 8.304638e-06],
        "miss_m": [2220.3603, 2807.1338, 3001.6662, 2500.0000],
    },
    "critical": {
        "risk": ["AMBER", "AMBER", "AMBER", "AMBER"],
        "tca": [17, 17, 17, 17],
        "pc": [7.398193e-05, 9.722978e-05, 8.479370e-05, 8.247144e-05],
        "miss_m": [22.3607, 50.0000, 316.2278, 100.0000],
    },
    "mixed": {
        "risk": ["AMBER", "GREEN", "GREEN", "AMBER", "NOMINAL"],
        "tca": [17, 17, 17, 17, 17],
        "pc": [7.393086e-05, 6.405165e-06, 7.249460e-07, 3.617531e-05, 0.0],
        "miss_m": [50.9902, 2507.9872, 3500.0000, 1500.0000, 50000.0000],
    },
    "demo": {
        "risk": ["AMBER", "AMBER", "GREEN"],
        "tca": [120, 60, 60],
        "pc": [7.972491e-05, 5.028529e-05, 1.779663e-06],
        "miss_m": [316.2278, 412.3106, 3006.6593],
    },
}


def _write_states(scenario: str, data_dir: Path) -> dict:
    """Write states_multi.npz for one scenario and return its arrays."""
    obj_ids, r_list, v_list, confidences = demo_presets.build_scenario(scenario)
    asset_r0 = np.array(demo_presets.ASSET_R_ECI_KM, dtype=float)
    asset_v0 = np.array(demo_presets.ASSET_V_ECI_KM_S, dtype=float)
    r_list = np.array(r_list, dtype=float)
    v_list = np.array(v_list, dtype=float)
    np.savez(
        data_dir / "states_multi.npz",
        object_ids=np.array(obj_ids),
        r_eci_km=r_list, v_eci_km_s=v_list,
        confidences=np.array(confidences, dtype=float),
        t_window=np.array([demo_presets.SAMPLE_DT_S, demo_presets.N_STEPS]),
        metadata=json.dumps({
            "source": "test", "scenario": scenario,
            "asset_state": {
                "r_eci_km": demo_presets.ASSET_R_ECI_KM,
                "v_eci_km_s": demo_presets.ASSET_V_ECI_KM_S,
            },
        }),
    )
    return {"r": r_list, "v": v_list, "asset_r0": asset_r0, "asset_v0": asset_v0}


def _run_demo_preset(scenario: str, data_dir: Path):
    """Build a scenario's states and run it through the demo path."""
    _write_states(scenario, data_dir)
    # propagate_and_screen reads the module-global DATA_DIR at call time.
    propagator_main.DATA_DIR = str(data_dir)
    propagator_main.propagate_and_screen()
    return np.load(data_dir / "prop_multi.npz", allow_pickle=True)


# ---------------------------------------------------------------------------
# SCRUM-392: the preset definitions themselves
# ---------------------------------------------------------------------------

class TestPresetDefinitions:
    def test_this_test_uses_the_live_definition(self):
        """The point of SCRUM-392. If this file ever grows its own copy of the
        geometry again, this fails."""
        assert demo_presets.build_scenario is not None
        assert not hasattr(sys.modules[__name__], "_PRESETS"), (
            "this module defines its own presets again; import demo_presets instead"
        )
        # The module loaded is the one the UI serves from, not a fixture.
        assert _DP_PATH.is_file()
        assert _DP_PATH.parts[-3:] == ("ui", "app", "demo_presets.py")

    @pytest.mark.parametrize("scenario,spec", list(demo_presets.all_specs()))
    def test_every_tca_is_on_the_sample_grid(self, scenario, spec):
        """A target TCA between samples cannot be observed.

        The propagator finds closest approach by argmin over samples at
        SAMPLE_DT_S. If true TCA falls between two of them, the nearest sample
        still carries separation along the approach axis, and that residual is
        what gets reported as the miss. Every preset was off-grid before
        SCRUM-392, by 20 s, which inflated the critical preset's first miss from
        22 m to 400 m.
        """
        remainder = spec.t_star_s % demo_presets.SAMPLE_DT_S
        assert remainder == pytest.approx(0.0, abs=1e-9), (
            f"{scenario}: t_star_s={spec.t_star_s} is not a multiple of "
            f"{demo_presets.SAMPLE_DT_S} s, so its stated miss will not be produced"
        )

    @pytest.mark.parametrize("scenario", ALL_SCENARIOS)
    def test_stated_miss_is_the_miss_produced(self, scenario, tmp_path):
        """The assertion that makes the preset geometry mean something.

        Each spec declares a miss. The propagator must report that miss, to
        within the propagation's own tolerance. Without this, the miss_y and
        miss_z values are decoration and nobody finds out.
        """
        d = _run_demo_preset(scenario, tmp_path)
        produced_m = [float(x) * 1000.0 for x in d["ca_table"]]
        stated_m = [s.miss_km * 1000.0 for s in demo_presets.specs_for(scenario)]
        assert produced_m == pytest.approx(stated_m, rel=1e-6), (
            f"{scenario}: stated miss {stated_m} but propagator produced {produced_m}"
        )

    @pytest.mark.parametrize("scenario", ALL_SCENARIOS)
    def test_declared_tca_index_is_the_index_produced(self, scenario, tmp_path):
        d = _run_demo_preset(scenario, tmp_path)
        produced = [int(x) for x in d["tca_indices"]]
        declared = [s.tca_index for s in demo_presets.specs_for(scenario)]
        assert produced == declared, f"{scenario}: declared {declared}, produced {produced}"

    def test_confidences_are_fixed_rather_than_drawn(self):
        """They used to be np.random.uniform in the UI and a seeded draw here, so
        the live demo's Pc varied run to run and this baseline could not have
        described it. Building twice must give the same values."""
        for scenario in ALL_SCENARIOS:
            first = demo_presets.build_scenario(scenario)[3]
            second = demo_presets.build_scenario(scenario)[3]
            assert first == second, f"{scenario}: confidences are not deterministic"


# ---------------------------------------------------------------------------
# SCRUM-370 regression guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_demo_preset_risk_and_tca_unchanged(scenario, tmp_path):
    """Every object's risk level and TCA index must match the baseline."""
    d = _run_demo_preset(scenario, tmp_path)
    risk = [str(x) for x in d["risk_levels"]]
    tca = [int(x) for x in d["tca_indices"]]
    assert risk == BASELINE[scenario]["risk"], (
        f"{scenario}: risk levels moved {BASELINE[scenario]['risk']} -> {risk}"
    )
    assert tca == BASELINE[scenario]["tca"], (
        f"{scenario}: TCA indices moved {BASELINE[scenario]['tca']} -> {tca}"
    )


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_demo_preset_pc_values_unchanged(scenario, tmp_path):
    """Pin Pc itself, not the colour it falls into.

    A risk label covers a decade of Pc, so a label-only baseline accepts a
    multiplicative error silently. That is exactly how the 4.2554x default-mode
    error in SCRUM-390 survived: every preset kept its label while the number
    behind it was wrong.
    """
    d = _run_demo_preset(scenario, tmp_path)
    pc = [float(x) for x in d["pc_values"]]
    expected = BASELINE[scenario]["pc"]
    assert pc == pytest.approx(expected, rel=1e-5, abs=1e-12), (
        f"{scenario}: Pc moved {expected} -> {pc}"
    )


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_demo_preset_miss_distance_unchanged(scenario, tmp_path):
    """Miss distance is the invariant the SCRUM-370 refactor had to preserve.
    Pinned separately from the spec comparison above so a change to the specs
    and a change to the propagator are distinguishable."""
    d = _run_demo_preset(scenario, tmp_path)
    miss_m = [float(x) * 1000.0 for x in d["ca_table"]]
    assert miss_m == pytest.approx(BASELINE[scenario]["miss_m"], rel=1e-6), (
        f"{scenario}: miss distances moved {BASELINE[scenario]['miss_m']} -> {miss_m}"
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

    times = np.arange(demo_presets.N_STEPS) * demo_presets.SAMPLE_DT_S
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
    baseline above is touched. SCRUM-391 is the intended way for it to change.
    """
    on_top = [demo_presets.DebrisSpec(
        t_star_s=17 * demo_presets.SAMPLE_DT_S,
        miss_y_km=0.0005, miss_z_km=0.0,
        v_approach_km_s=0.02, confidence=0.836144,
    )]
    demo_presets.PRESETS["_ceiling"] = on_top
    demo_presets.PRESET_LABELS["_ceiling"] = "CEIL"
    try:
        d = _run_demo_preset("_ceiling", tmp_path)
    finally:
        del demo_presets.PRESETS["_ceiling"], demo_presets.PRESET_LABELS["_ceiling"]

    miss_m = float(d["ca_table"][0]) * 1000.0
    pc = float(d["pc_values"][0])
    assert miss_m < 1.0, f"probe did not close the miss: {miss_m:.3f} m"
    assert pc < propagator_main.PC_RED_THRESHOLD, (
        f"RED is now reachable at the demo covariance: Pc={pc:.4e} at a "
        f"{miss_m:.3f} m miss. Read the ceiling note in this module's docstring "
        f"before updating BASELINE."
    )
    # And it is close to the threshold, not orders below, which is why the
    # original inflated numbers read as RED so convincingly.
    assert pc > 0.5 * propagator_main.PC_RED_THRESHOLD, (
        f"ceiling moved far from the threshold: Pc={pc:.4e}"
    )
