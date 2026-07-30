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
- 2026-07-30, SCRUM-391: risk and Pc recaptured for critical and mixed only. Two
  objects in critical and one in mixed now carry a supplied position covariance
  and reach RED. Every object without one is bit-identical to the SCRUM-392
  entry, which is the evidence that the fallback path did not change.

The covariance ceiling, and how SCRUM-391 got past it
-----------------------------------------------------
``PC_RED_THRESHOLD`` is 1e-4 and ``HBR_M`` is 15 m. An object with no supplied
covariance inherits ``DEFAULT_DEBRIS_UNCERTAINTY_M / confidence``, a TLE-grade
2000 m that can only inflate above that, putting the encounter-plane sigma near
1.1 km and capping Pc around 9.7e-5. So RED was unreachable for every preset, and
no amount of moving objects closer could change it.

SCRUM-391 did not move the thresholds or the hard-body radius. It let a scenario
state what it actually knows about an object. A secondary with a tracked
covariance of a few hundred metres reaches RED at the same geometry that stays
AMBER at TLE grade, which is the honest version of the demo's top tier and a
better thing to show than a RED the numbers do not support.

``test_red_is_still_unreachable_without_a_supplied_covariance`` keeps the ceiling
pinned for the fallback path, so the old failure mode cannot return quietly.
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
        "sigma_m": [2391.9325, 2064.6994, 2177.7976],
        "cov_source": ["confidence_default"] * 3,
    },
    "warning": {
        "risk": ["AMBER", "GREEN", "GREEN", "GREEN"],
        "tca": [17, 17, 17, 17],
        "pc": [1.462442e-05, 3.219644e-06, 2.580999e-06, 8.304638e-06],
        "miss_m": [2220.3603, 2807.1338, 3001.6662, 2500.0000],
        "sigma_m": [2391.9325, 2064.6994, 2177.7976, 2253.0362],
        "cov_source": ["confidence_default"] * 4,
    },
    # SCRUM-391: objects 0 and 1 are tracked and reach RED. Objects 2 and 3 are
    # the same close geometry at TLE grade and stay AMBER. Their Pc is unchanged
    # from the SCRUM-392 entry, to the digit.
    "critical": {
        "risk": ["RED", "RED", "AMBER", "AMBER"],
        "tca": [17, 17, 17, 17],
        "pc": [1.062006e-03, 1.052005e-03, 8.479370e-05, 8.247144e-05],
        "miss_m": [22.3607, 50.0000, 316.2278, 100.0000],
        "sigma_m": [250.0000, 250.0000, 2177.7976, 2253.0362],
        "cov_source": ["supplied", "supplied",
                       "confidence_default", "confidence_default"],
    },
    # Spans all four tiers now that object 0 is tracked.
    "mixed": {
        "risk": ["RED", "GREEN", "GREEN", "AMBER", "NOMINAL"],
        "tca": [17, 17, 17, 17, 17],
        "pc": [1.051507e-03, 6.405165e-06, 7.249460e-07, 3.617531e-05, 0.0],
        "miss_m": [50.9902, 2507.9872, 3500.0000, 1500.0000, 50000.0000],
        "sigma_m": [250.0000, 2064.6994, 2177.7976, 2253.0362, 2544.9048],
        "cov_source": ["supplied"] + ["confidence_default"] * 4,
    },
    "demo": {
        "risk": ["AMBER", "AMBER", "GREEN"],
        "tca": [120, 60, 60],
        "pc": [7.972491e-05, 5.028529e-05, 1.779663e-06],
        "miss_m": [316.2278, 412.3106, 3006.6593],
        "sigma_m": [2222.2222, 2222.2222, 2222.2222],
        "cov_source": ["confidence_default"] * 3,
    },
}


def _write_states(scenario: str, data_dir: Path, include_sigma: bool = True) -> dict:
    """Write states_multi.npz for one scenario and return its arrays.

    include_sigma=False omits position_sigma_m entirely, which is how a producer
    predating SCRUM-391 writes the artifact. Used to prove the fallback path.
    """
    obj_ids, r_list, v_list, confidences, sigma_m = demo_presets.build_scenario(scenario)
    asset_r0 = np.array(demo_presets.ASSET_R_ECI_KM, dtype=float)
    asset_v0 = np.array(demo_presets.ASSET_V_ECI_KM_S, dtype=float)
    r_list = np.array(r_list, dtype=float)
    v_list = np.array(v_list, dtype=float)
    np.savez(
        data_dir / "states_multi.npz",
        object_ids=np.array(obj_ids),
        r_eci_km=r_list, v_eci_km_s=v_list,
        confidences=np.array(confidences, dtype=float),
        **({"position_sigma_m": np.array(sigma_m, dtype=float)} if include_sigma else {}),
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


def _run_demo_preset(scenario: str, data_dir: Path, include_sigma: bool = True):
    """Build a scenario's states and run it through the demo path."""
    _write_states(scenario, data_dir, include_sigma=include_sigma)
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
def test_demo_preset_covariance_unchanged(scenario, tmp_path):
    """SCRUM-391: pin the covariance each Pc was computed against, and its
    source. Without this, a Pc could move because the geometry changed or because
    the covariance changed and the baseline could not tell you which."""
    d = _run_demo_preset(scenario, tmp_path)
    sigma = [float(x) for x in d["debris_sigma_m"]]
    source = [str(x) for x in d["covariance_sources"]]
    assert sigma == pytest.approx(BASELINE[scenario]["sigma_m"], rel=1e-6), (
        f"{scenario}: debris sigma moved {BASELINE[scenario]['sigma_m']} -> {sigma}"
    )
    assert source == BASELINE[scenario]["cov_source"], (
        f"{scenario}: covariance source moved {BASELINE[scenario]['cov_source']} -> {source}"
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
                "ca_table", "risk_levels", "tca_indices",
                # SCRUM-391 additions.
                "debris_sigma_m", "covariance_sources"):
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


class TestSuppliedCovariance:
    """SCRUM-391. A scenario can state what it actually knows about an object."""

    def test_supplied_sigma_is_used_and_recorded(self, tmp_path):
        d = _run_demo_preset("critical", tmp_path)
        sigma = [float(x) for x in d["debris_sigma_m"]]
        source = [str(x) for x in d["covariance_sources"]]
        assert sigma == pytest.approx(BASELINE["critical"]["sigma_m"], rel=1e-6)
        assert source == BASELINE["critical"]["cov_source"]

    def test_supplied_sigma_is_not_scaled_by_confidence(self, tmp_path):
        """Confidence is a stand-in for how well an object is known. A supplied
        sigma is how well it is known. Multiplying them counts it twice, and a
        250 m covariance at 0.836 confidence would silently become 299 m."""
        d = _run_demo_preset("critical", tmp_path)
        tracked = float(d["debris_sigma_m"][0])
        assert tracked == pytest.approx(demo_presets.TRACKED_SECONDARY_SIGMA_M, rel=1e-12)

    @pytest.mark.parametrize("scenario", ALL_SCENARIOS)
    def test_omitting_the_field_entirely_changes_nothing(self, scenario, tmp_path):
        """AC2. An artifact written by a producer that predates SCRUM-391 has no
        position_sigma_m key at all, and must behave exactly as it did before.

        Compared against the confidence-scaled default rather than against the
        baseline, because the baseline now contains supplied values for some
        objects. Asserted rather than assumed, since the whole point of the
        fallback is that nothing outside the demo notices this change.
        """
        d = _run_demo_preset(scenario, tmp_path, include_sigma=False)
        sigma = [float(x) for x in d["debris_sigma_m"]]
        source = [str(x) for x in d["covariance_sources"]]
        expected = [
            propagator_main.DEFAULT_DEBRIS_UNCERTAINTY_M / max(spec.confidence, 0.1)
            for spec in demo_presets.specs_for(scenario)
        ]
        assert sigma == pytest.approx(expected, rel=1e-9)
        assert source == ["confidence_default"] * len(expected)

    def test_untracked_objects_keep_their_pre_391_pc(self, tmp_path):
        """The other half of AC2, at the level that matters.

        critical objects 2 and 3 have no supplied covariance, so their Pc must be
        exactly what SCRUM-392 recorded. If this drifts, the fallback changed and
        every real, non-demo input changed with it.
        """
        d = _run_demo_preset("critical", tmp_path)
        pc = [float(x) for x in d["pc_values"]]
        assert pc[2] == pytest.approx(8.479370e-05, rel=1e-5)
        assert pc[3] == pytest.approx(8.247144e-05, rel=1e-5)

    def test_the_same_geometry_grades_differently_by_covariance(self, tmp_path):
        """The point of the preset, stated as an assertion.

        critical object 1 sits 50 m away and is RED. Object 3 sits 100 m away and
        is AMBER. Two miss distances of the same order, a decade apart in Pc,
        because one object is tracked and the other is known from a TLE. If this
        ever collapses to a single band, the demo has stopped making its point.
        """
        d = _run_demo_preset("critical", tmp_path)
        risk = [str(x) for x in d["risk_levels"]]
        pc = [float(x) for x in d["pc_values"]]
        assert risk[1] == "RED" and risk[3] == "AMBER"
        assert pc[1] / pc[3] > 10.0


def test_red_is_still_unreachable_without_a_supplied_covariance(tmp_path):
    """SCRUM-390's ceiling, kept pinned for the fallback path.

    Replaces test_red_is_unreachable_at_the_demo_covariance, which asserted the
    ceiling for every object because no object could supply a covariance. That is
    still true of the fallback, and it is the part worth guarding: an object known
    only from a TLE cannot reach RED at HBR_M 15 m, whatever its geometry.

    Place an object essentially on top of the asset with no supplied sigma. Pc
    for a near-zero miss is about HBR^2 / (2 sigma_x sigma_z), and at a 2 km
    uncertainty that is around 1e-4 at best, so closing the miss cannot get there.

    If this starts failing, something changed HBR_M, DEFAULT_DEBRIS_UNCERTAINTY_M,
    or the thresholds. Read the ceiling note in this module's docstring first.
    """
    on_top = [demo_presets.DebrisSpec(
        t_star_s=17 * demo_presets.SAMPLE_DT_S,
        miss_y_km=0.0005, miss_z_km=0.0,
        v_approach_km_s=0.02, confidence=0.836144,
        position_sigma_m=None,
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
    assert str(d["covariance_sources"][0]) == "confidence_default"
    assert pc < propagator_main.PC_RED_THRESHOLD, (
        f"RED is now reachable without a supplied covariance: Pc={pc:.4e} at a "
        f"{miss_m:.3f} m miss. Read the ceiling note in this module's docstring "
        f"before updating BASELINE."
    )
    # Close to the threshold, not orders below, which is why the pre-SCRUM-390
    # inflated numbers read as RED so convincingly.
    assert pc > 0.5 * propagator_main.PC_RED_THRESHOLD, (
        f"ceiling moved far from the threshold: Pc={pc:.4e}"
    )


def test_thresholds_and_hard_body_radius_are_unchanged(tmp_path):
    """SCRUM-391 AC5. RED was recovered by describing the covariance honestly,
    not by moving the goalposts. Both of these are physical or policy quantities
    that exist for reasons unrelated to the demo, so pin them here where anyone
    tempted to nudge one will see why not."""
    assert propagator_main.HBR_M == 15.0
    assert propagator_main.PC_RED_THRESHOLD == 1e-4
    assert propagator_main.PC_AMBER_THRESHOLD == 1e-5
    assert propagator_main.PC_GREEN_THRESHOLD == 1e-7
    assert propagator_main.DEFAULT_DEBRIS_UNCERTAINTY_M == 2000.0
