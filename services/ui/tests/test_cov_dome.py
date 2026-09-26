"""tests/test_cov_dome.py

SCRUM-464: covariance drawn as one wireframe dome around the asset.

cov_dome_harness.mjs extracts the real normalizer from index.html and checks the
split the feature rests on: the shape is the covariance's (ratios preserved), the
size is not (major axis pinned to a fixed readable value), the floor is kept, and a
degenerate covariance is handled without a divide by zero.

This file runs that harness under pytest and adds template-wiring checks that need
no node: exactly one covariance mesh, no per-object loop left behind, a wireframe
material, the toggle driving it, and a caption that never claims true scale.

It replaces test_cov_scale.py, which tested the abandoned SCRUM-463 slider.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_HARNESS = Path(__file__).parent / "cov_dome_harness.mjs"
_TEMPLATE = Path(__file__).parent.parent / "app" / "templates" / "index.html"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not available; the dome harness needs it")
def test_the_cov_dome_harness_passes():
    assert _HARNESS.exists(), f"missing harness at {_HARNESS}"
    proc = subprocess.run(
        ["node", str(_HARNESS)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        "the covariance dome harness failed:\n" + proc.stdout + "\n" + proc.stderr)
    assert "checks passed" in proc.stdout


@pytest.fixture(scope="module")
def html() -> str:
    return _TEMPLATE.read_text()


def _fn(html: str, name: str) -> str:
    m = re.search(r"^function " + name + r"\(.*?^\}", html, re.S | re.M)
    assert m, f"{name} is missing"
    return m.group(0)


class TestThePerObjectEllipsoidsAreGone:
    """SCRUM-448's filled shell on every object is what this replaces."""

    @pytest.mark.parametrize("leftover", ["buildCovEllipsoid", "globeCovParts"])
    def test_no_trace_of_the_old_builder_or_list(self, html, leftover):
        assert leftover not in html, (
            f"{leftover} survives; a per-object ellipsoid would draw alongside "
            f"the dome")

    def test_the_abandoned_exaggeration_slider_is_not_reintroduced(self, html):
        for name in ("GLOBE_COV_EXAGGERATION", "globeCovExaggeration",
                     "globeCovScaleInput", "covAxisScale"):
            assert name not in html, f"{name} is SCRUM-463, which was abandoned"

    def test_the_object_loop_builds_no_covariance(self, html):
        """The per-object branch read objects[objId].cov; it must be gone."""
        assert "objects[objId].cov" not in html
        assert "objCov" not in html


class TestExactlyOneDome:
    def test_the_state_is_a_single_mesh_not_a_list(self, html):
        assert re.search(r"^let globeCovDome=null;", html, re.M), (
            "the covariance state is not a single nullable mesh")

    def test_only_one_place_builds_it(self, html):
        """Two build sites would put two domes on the same persistent marker."""
        calls = re.findall(r"buildAssetCovDome\(", html)
        # the definition plus exactly one call site
        assert len(calls) == 2, f"expected one call site, found {len(calls) - 1}"

    def test_it_is_built_from_the_asset_covariance(self, html):
        assert "buildAssetCovDome(orbits.asset.cov,assetMarker)" in html

    def test_it_is_parented_to_the_asset_marker(self, html):
        """So it follows the asset without per-frame position code."""
        fn = _fn(html, "buildAssetCovDome")
        assert "marker.add(mesh)" in fn

    def test_it_is_detached_on_a_rebuild(self, html):
        """The asset marker is persistent, so a stale dome would accumulate."""
        clear = _fn(html, "clearGlobeTracks")
        assert "globeCovDome" in clear
        assert "parent.remove(globeCovDome)" in clear
        assert "globeCovDome=null" in clear


class TestItLooksLikeTheConcept:
    @pytest.fixture(scope="class")
    def builder(self, html) -> str:
        return _fn(html, "buildAssetCovDome")

    def test_it_is_a_wireframe(self, builder):
        assert "wireframe:true" in builder

    def test_it_is_cyan_and_translucent(self, builder):
        assert "0x4ff8e8" in builder
        assert "transparent:true" in builder
        assert re.search(r"opacity:0\.1[0-9]", builder), "opacity is not ~0.14"

    def test_it_is_a_sphere_scaled_into_an_ellipsoid(self, builder):
        assert "SphereGeometry(1,20,20)" in builder
        assert "mesh.scale.set(" in builder

    def test_it_does_not_write_depth(self, builder):
        """So the rings and markers inside it stay visible."""
        assert "depthWrite:false" in builder


class TestTheAspectCompressionIsDeclared:
    """The departure from "real shape", and why it has to be visible in the code.

    A tracked LEO asset's position covariance is 299:1 to 15,764:1 along-track, so
    the true ratios at a readable major axis draw a sub-pixel needle. The minor axes
    are compressed to make it a dome. That is a rendering choice, not a measurement,
    so it is a named constant with the measurements in the comment rather than a
    magic number, and it is reversible.
    """

    def test_the_compression_is_named_and_tunable(self, html):
        assert re.search(r"^const GLOBE_COV_ASPECT_GAMMA=", html, re.M)
        assert re.search(r"^const GLOBE_COV_MIN_AXIS_RATIO=", html, re.M)

    def test_the_reason_is_recorded_with_the_numbers(self, html):
        """So the next reader does not "restore" the real ratios and get a needle."""
        block = html.split("const GLOBE_COV_ASPECT_GAMMA=")[0][-2000:]
        assert "15,764" in block, "the measured aspect ratios are not recorded"
        assert "sub-pixel" in block or "needle" in block

    def test_the_compression_is_monotonic_in_the_code(self, html):
        """A power law, not a clamp: ordering and relative roundness survive."""
        fn = _fn(html, "covDomeAxisScales")
        assert "Math.pow(ratio,GLOBE_COV_ASPECT_GAMMA)" in fn.replace(" ", "")

    def test_it_can_be_turned_off(self, html):
        """Gamma 1 and ratio 0 give the raw ratios back, and that is documented."""
        block = html.split("const GLOBE_COV_ASPECT_GAMMA=")[0][-1200:]
        assert "GAMMA to 1" in block


class TestOrientationIsRealSizeIsNot:
    def test_the_orientation_comes_from_the_covariance_axes(self, html):
        fn = _fn(html, "buildAssetCovDome")
        assert "cov.axes.map" in fn
        assert "eciToScene" in fn, "axes must be rotated into scene space"
        assert "makeBasis" in fn
        assert "setFromRotationMatrix" in fn

    def test_the_size_is_normalized_not_physical(self, html):
        fn = _fn(html, "covDomeAxisScales")
        assert "GLOBE_COV_READABLE_MAJOR" in fn
        # the SCRUM-448 physical conversion must not be here any more
        assert "EARTH_RADIUS_KM" not in fn, (
            "the normalizer still converts metres to scene units; the size would "
            "depend on magnitude again")

    def test_the_readable_major_is_a_named_constant(self, html):
        assert re.search(r"^const GLOBE_COV_READABLE_MAJOR=", html, re.M)

    def test_the_floor_survives(self, html):
        assert re.search(r"^const GLOBE_COV_MIN_SCALE=1e-6;", html, re.M)
        assert "GLOBE_COV_MIN_SCALE" in _fn(html, "covDomeAxisScales")


class TestTheBreathingCannotDistortTheShape:
    """The pulse is a rendering flourish; the shape is the measurement."""

    def test_it_multiplies_every_axis_by_one_factor(self, html):
        loop = html.split("function startGlobeLoop()")[1].split("\n}")[0]
        assert "globeCovDome" in loop
        pulse = loop[loop.index("globeCovDome"):]
        # one scalar, applied to all three base components
        assert "b[0]*cs" in pulse and "b[1]*cs" in pulse and "b[2]*cs" in pulse

    def test_it_rescales_from_the_base_not_the_current_scale(self, html):
        """Compounding would make the dome drift in size over time."""
        assert "userData.baseScale" in _fn(html, "buildAssetCovDome")
        loop = html.split("function startGlobeLoop()")[1].split("\n}")[0]
        assert "userData.baseScale" in loop


class TestToggleAndCaption:
    def test_the_toggle_drives_the_dome_visibility(self, html):
        fn = _fn(html, "toggleGlobeCovariance")
        assert "globeCovDome.visible=globeShowCovariance" in fn

    def test_a_new_dome_honours_the_current_toggle(self, html):
        assert "mesh.visible=globeShowCovariance" in _fn(html, "buildAssetCovDome")

    def test_the_caption_never_claims_true_scale(self, html):
        fn = _fn(html, "renderGlobeCovNote")
        assert "true scale" not in fn, (
            "the caption claims true scale, but the size is normalized")
        assert "not to scale" in fn

    def test_the_caption_disclaims_the_shape_too(self, html):
        """The aspect is compressed, so claiming a true shape would be a lie.

        The orientation is the only part of the drawing that is a measurement,
        and the caption has to be precise about that rather than generically
        hedging on "scale".
        """
        fn = _fn(html, "renderGlobeCovNote")
        assert "shape and size not to scale" in fn

    def test_the_caption_says_which_part_is_real(self, html):
        fn = _fn(html, "renderGlobeCovNote")
        assert "orientation true" in fn

    def test_the_caption_shows_only_with_a_dome_and_the_toggle_on(self, html):
        fn = _fn(html, "renderGlobeCovNote")
        assert "!globeShowCovariance||!globeCovDome" in fn.replace(" ", "")
