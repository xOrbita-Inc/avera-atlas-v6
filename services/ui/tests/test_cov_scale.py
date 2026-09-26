"""tests/test_cov_scale.py

SCRUM-463: the covariance exaggeration slider on the globe.

cov_scale_harness.mjs extracts the real functions from index.html and checks the
maths and the caption: 1x equals the SCRUM-448 true scale computed independently,
Nx is exactly N times it, the floor holds at every value, the rescale reads the
stored base sigmas so a second drag is not compounded, and COVARIANCE off hides
everything at any slider value.

This file runs that harness under pytest (skipped without node) and adds
template-wiring checks that need no node. The split is the SCRUM-461 lesson: a
correct scale function that no slider calls would pass a pure harness and ship a
control that does nothing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_HARNESS = Path(__file__).parent / "cov_scale_harness.mjs"
_TEMPLATE = Path(__file__).parent.parent / "app" / "templates" / "index.html"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not available; the scale harness needs it")
def test_the_cov_scale_harness_passes():
    assert _HARNESS.exists(), f"missing harness at {_HARNESS}"
    proc = subprocess.run(
        ["node", str(_HARNESS)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        "the covariance scale harness failed:\n" + proc.stdout + "\n" + proc.stderr)
    assert "checks passed" in proc.stdout


@pytest.fixture(scope="module")
def html() -> str:
    return _TEMPLATE.read_text()


def _fn(html: str, name: str) -> str:
    m = re.search(r"^function " + name + r"\(.*?^\}", html, re.S | re.M)
    assert m, f"{name} is missing"
    return m.group(0)


class TestStateNotConstant:
    def test_the_exaggeration_is_mutable_state(self, html):
        assert re.search(r"^let globeCovExaggeration=1\s*;?\s*$", html, re.M), (
            "globeCovExaggeration is not a mutable let defaulting to 1")

    def test_the_old_constant_is_gone(self, html):
        """A surviving constant would be a second source of truth for the scale."""
        assert "GLOBE_COV_EXAGGERATION=" not in html
        assert "GLOBE_COV_EXAGGERATION " not in html

    def test_the_default_is_true_scale(self, html):
        """The globe opens honest; exaggeration is opt-in."""
        assert re.search(r"^let globeCovExaggeration=1", html, re.M)
        slider = re.search(r'<input type="range" id="globeCovScaleInput"[^>]*>', html)
        assert slider, "the slider input is missing"
        assert 'value="1"' in slider.group(0)

    def test_the_untouched_constants_are_untouched(self, html):
        assert "const GLOBE_COV_SIGMA_MULTIPLIER=3;" in html
        assert "const GLOBE_COV_MIN_SCALE=1e-6;" in html


class TestOneFormula:
    """One helper, called by both the builder and the slider, so they cannot drift."""

    def test_the_helper_exists_and_carries_the_floor(self, html):
        fn = _fn(html, "covAxisScale")
        assert "GLOBE_COV_SIGMA_MULTIPLIER" in fn
        assert "globeCovExaggeration" in fn
        assert "EARTH_RADIUS_KM" in fn
        assert "GLOBE_COV_MIN_SCALE" in fn
        assert "Math.max" in fn

    def test_the_builder_uses_the_helper(self, html):
        fn = _fn(html, "buildCovEllipsoid")
        assert fn.count("covAxisScale(") == 3, "the builder does not scale all three axes through the helper"
        assert "GLOBE_COV_SIGMA_MULTIPLIER" not in fn, (
            "the builder still computes its own scale; that is a second formula")

    def test_the_slider_handler_uses_the_helper(self, html):
        fn = _fn(html, "setGlobeCovExaggeration")
        assert fn.count("covAxisScale(") == 3
        assert "GLOBE_COV_SIGMA_MULTIPLIER" not in fn

    def test_there_is_exactly_one_place_the_scale_is_computed(self, html):
        """Outside the helper, nothing multiplies sigma by the 3-sigma constant."""
        others = [m.start() for m in re.finditer(r"GLOBE_COV_SIGMA_MULTIPLIER", html)]
        helper = _fn(html, "covAxisScale")
        helper_at = html.index(helper)
        inside = [i for i in others if helper_at <= i < helper_at + len(helper)]
        # the declaration, the caption (which prints "3σ"), and the helper
        assert len(others) - len(inside) <= 2, (
            "GLOBE_COV_SIGMA_MULTIPLIER is used in more places than the helper, "
            "its declaration and the caption")


class TestLiveRescaleNotRebuild:
    def test_the_base_sigmas_ride_on_the_mesh(self, html):
        fn = _fn(html, "buildCovEllipsoid")
        assert "userData.sigmas_m" in fn

    def test_the_handler_walks_the_existing_meshes(self, html):
        fn = _fn(html, "setGlobeCovExaggeration")
        assert "globeCovParts" in fn
        assert "scale.set(" in fn

    def test_the_handler_rebuilds_nothing(self, html):
        """A rebuild would drop the camera framing and churn geometry every drag."""
        fn = _fn(html, "setGlobeCovExaggeration")
        for forbidden in ("buildCovEllipsoid", "SphereGeometry", "new THREE.Mesh",
                          "fetch(", "updateGlobeOrbits"):
            assert forbidden not in fn, f"the slider handler calls {forbidden}"

    def test_the_handler_reads_the_stored_sigmas(self, html):
        """Rescaling from the current scale would compound across drags."""
        fn = _fn(html, "setGlobeCovExaggeration")
        assert "userData" in fn and "sigmas_m" in fn


class TestCaptionCannotDisagree:
    def test_the_caption_reads_the_live_value(self, html):
        fn = _fn(html, "renderGlobeCovNote")
        assert "globeCovExaggeration" in fn
        assert "true scale" in fn
        assert "not to scale" in fn

    def test_the_handler_refreshes_the_caption(self, html):
        """One value drives both, so they are updated together or not at all."""
        assert "renderGlobeCovNote()" in _fn(html, "setGlobeCovExaggeration")

    def test_the_caption_still_states_the_sigma_count(self, html):
        assert "GLOBE_COV_SIGMA_MULTIPLIER" in _fn(html, "renderGlobeCovNote")


class TestSliderWiring:
    @pytest.fixture(scope="class")
    def slider(self, html) -> str:
        m = re.search(r'<input type="range" id="globeCovScaleInput".*?>', html, re.S)
        assert m, "the slider input is missing"
        return m.group(0)

    def test_the_range_is_one_to_fifty_by_one(self, slider):
        assert 'min="1"' in slider
        assert 'max="50"' in slider
        assert 'step="1"' in slider

    def test_it_drives_the_exaggeration_state(self, slider):
        assert "setGlobeCovExaggeration(this.value)" in slider
        assert "oninput" in slider, "on change only would not rescale as it is dragged"

    def test_it_has_an_accessible_label(self, slider):
        assert "aria-label" in slider

    def test_it_sits_in_the_globe_control_cluster(self, html):
        cluster = html.split('id="globeControls"')[1].split("</div>")[0]
        assert 'id="globeCovScale"' in cluster or 'globeCovScaleInput' in html.split(
            'id="globeControls"')[1][:2000]

    def test_it_sits_next_to_the_covariance_button(self, html):
        cov_at = html.index('id="globeCovBtn"')
        slider_at = html.index('id="globeCovScale"')
        focus_at = html.index('id="globeFocusWorstBtn"')
        assert cov_at < slider_at < focus_at, (
            "the slider is not between COVARIANCE and FOCUS WORST")

    def test_it_starts_disabled_and_dimmed(self, html):
        """COVARIANCE defaults off, so the slider must not look live."""
        wrap = re.search(r'<span class="([^"]*)" id="globeCovScale"', html)
        assert wrap and "off" in wrap.group(1)
        slider = re.search(r'<input type="range" id="globeCovScaleInput".*?>',
                           html, re.S).group(0)
        assert "disabled" in slider

    def test_the_readout_element_exists(self, html):
        assert 'id="globeCovScaleVal"' in html
        assert "globeCovScaleVal" in _fn(html, "setGlobeCovExaggeration")


class TestTheCaptionDoesNotCollideWithTheControls:
    """A regression from this ticket, caught live and fixed.

    SCRUM-448 parked the caption at a fixed bottom:46px, just clear of a
    single-row control cluster. The slider widened that cluster enough to wrap to
    three rows, and the caption then rendered on top of the buttons -- measured
    overlapping on a 1440px window. A fixed offset cannot track a cluster whose
    height depends on how it wraps, so the caption is now a row of the cluster.
    """

    def test_the_caption_lives_inside_the_control_cluster(self, html):
        cluster_at = html.index('id="globeControls"')
        note_at = html.index('id="globeCovNote"')
        # the first control in the cluster
        rotate_at = html.index('id="globeRotateBtn"')
        assert cluster_at < note_at < rotate_at, (
            "the caption is not the first child of the control cluster")

    def test_it_is_a_full_width_row_above_the_controls(self, html):
        style = re.search(r"\.globe-cov-note\{([^}]*)\}", html).group(1)
        assert "flex:0 0 100%" in style, "the caption does not take its own row"
        assert "order:-1" in style, "the caption is not ordered above the controls"

    def test_it_no_longer_floats_at_a_fixed_offset(self, html):
        """The fixed offset is what collided once the cluster could wrap."""
        style = re.search(r"\.globe-cov-note\{([^}]*)\}", html).group(1)
        assert "position:absolute" not in style
        # The `bottom` PROPERTY, at a declaration boundary -- a bare substring
        # test also matches margin-bottom, which is fine and present.
        decls = [d.strip().split(":")[0].strip()
                 for d in style.replace("\n", " ").split(";") if ":" in d]
        assert "bottom" not in decls, f"still positioned by offset: {decls}"
        assert "top" not in decls


class TestToggleGovernsTheSlider:
    def test_the_toggle_syncs_the_control(self, html):
        assert "syncGlobeCovScaleControl()" in _fn(html, "toggleGlobeCovariance")

    def test_the_sync_disables_and_dims(self, html):
        fn = _fn(html, "syncGlobeCovScaleControl")
        assert "disabled" in fn
        assert "globeShowCovariance" in fn
        assert "'off'" in fn

    def test_the_toggle_still_hides_every_ellipsoid(self, html):
        """The slider must not have weakened what COVARIANCE off means."""
        fn = _fn(html, "toggleGlobeCovariance")
        assert "globeCovParts.forEach" in fn
        assert "visible=globeShowCovariance" in fn

    def test_a_rebuild_resyncs_the_control(self, html):
        """So the control cannot drift from the toggle after a fetch."""
        assert html.count("syncGlobeCovScaleControl()") >= 2
