"""tests/test_encounter_view_resize.py

SCRUM-483: the 2D encounter view comes back from the globe centred and clean.

Coming back from 3D left the encounter off-centre, and every pan or double-click
centre smeared a ghost of the previous render across it. Both symptoms had one
cause. While the globe is up the 2D wrap is display:none, so its clientWidth and
clientHeight are 0; resize() bails on a zero measurement, so every window-resize
during 3D was discarded. Returning to 2D un-hid the canvas and drew straight onto
it without ever re-measuring, leaving encounterW/encounterH and the backing store
describing the element as it used to be. The centre came from the stale width, and
the clear -- clearRect(0,0,encounterW,encounterH) applied through the
devicePixelRatio scale -- no longer reached the edges of the canvas, so whatever
sat in the uncovered strip stayed on screen.

The fix is in two parts and this suite pins both:

  1. setView's 2D branch schedules resizeEncounter() in a requestAnimationFrame, so
     the canvas is re-measured once the un-hidden wrap has actually been laid out.
     This is the real fix; it repairs the centring and the clear together, because
     assigning canvas.width rebuilds the backing store and resets the transform.
  2. drawEncounter and drawEncounterEmpty clear the whole backing store under an
     identity transform. This is defence in depth: it bounds the damage to nothing
     even in the frame before the rAF callback runs, and it keeps the clear correct
     if the globals are ever stale again for some other reason.

Both matter. Testing only the first would let a future edit quietly reintroduce a
short clear; testing only the second would leave the off-centre render unguarded,
since a full clear does not fix a centre computed from a stale width.

Run from repo root:
    python -m pytest services/ui/tests/test_encounter_view_resize.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_TEMPLATE = Path(__file__).parent.parent / "app" / "templates" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return _TEMPLATE.read_text()


def _fn(html: str, name: str) -> str:
    m = re.search(r"^function " + name + r"\(.*?^\}", html, re.S | re.M)
    assert m, f"{name} is missing"
    return m.group(0)


def _two_d_branch(html: str) -> str:
    """setView's else branch -- the path taken when returning to the 2D view."""
    set_view = _fn(html, "setView")
    m = re.search(r"\n\s*\}\s*else\s*\{(.*)", set_view, re.S)
    assert m, "setView no longer has an else branch"
    return m.group(1)


def _three_d_branch(html: str) -> str:
    set_view = _fn(html, "setView")
    m = re.search(r"if\s*\(\s*v\s*===\s*'3d'\s*\)\s*\{(.*?)\n\s*\}\s*else\s*\{", set_view, re.S)
    assert m, "setView no longer has a 3d branch"
    return m.group(1)


# A full-backing-store clear: identity transform, then clearRect over canvas.width
# and canvas.height, with the transform saved and restored around it.
_IDENTITY = r"setTransform\(\s*1\s*,\s*0\s*,\s*0\s*,\s*1\s*,\s*0\s*,\s*0\s*\)"
_FULL_CLEAR = re.compile(
    r"\.save\(\)\s*;?\s*"
    r"(?:\w+\.)?" + _IDENTITY + r"\s*;?\s*"
    r"(?:\w+\.)?clearRect\(\s*0\s*,\s*0\s*,\s*[\w.]*canvas\.width\s*,\s*[\w.]*canvas\.height\s*\)\s*;?\s*"
    r"(?:\w+\.)?restore\(\)",
    re.S,
)


class TestReturningToTheTwoDViewReMeasuresTheCanvas:
    def test_the_2d_branch_schedules_a_resize(self, html):
        branch = _two_d_branch(html)
        assert re.search(r"requestAnimationFrame\s*\(", branch), (
            "setView's 2D branch does not schedule anything; the canvas will keep "
            "the size it had before the globe was opened"
        )

    def test_the_resize_is_inside_the_animation_frame_not_called_directly(self, html):
        """A direct call would measure the wrap before it has been laid out."""
        branch = _two_d_branch(html)
        m = re.search(r"requestAnimationFrame\(\s*function\s*\(\s*\)\s*\{(.*?)\}\s*\)", branch, re.S)
        assert m, "no requestAnimationFrame(function(){...}) in the 2D branch"
        assert re.search(r"\bresizeEncounter\s*\(\s*\)", m.group(1)), (
            f"the scheduled callback does not call resizeEncounter(): {m.group(1)!r}"
        )

    def test_it_matches_the_portlet_pattern_already_in_the_file(self, html):
        """The same idiom the expand/restore handlers use, so there is one way to do this."""
        pattern = r"requestAnimationFrame\(function\(\)\{\s*resizeEncounter\(\);\s*\}\);"
        assert len(re.findall(pattern, html)) >= 3, (
            "expected the 2D branch to reuse the portlet expand/restore idiom"
        )

    def test_the_synchronous_draw_is_kept_so_there_is_no_empty_flash(self, html):
        """The rAF is a frame away; something has to be on screen in the meantime."""
        branch = _two_d_branch(html)
        assert re.search(r"\bdrawEncounter\s*\(", branch)
        assert re.search(r"\bdrawEncounterEmpty\s*\(", branch)

    def test_the_3d_branch_is_untouched_by_this_fix(self, html):
        """Out of scope, and SCRUM-482's work lives there."""
        assert not re.search(r"\bresizeEncounter\s*\(", _three_d_branch(html))


class TestTheClearCoversTheWholeBackingStore:
    @pytest.mark.parametrize("name", ["drawEncounter", "drawEncounterEmpty"])
    def test_it_clears_under_an_identity_transform(self, html, name):
        assert _FULL_CLEAR.search(_fn(html, name)), (
            f"{name} does not clear the full backing store under an identity "
            f"transform; a stale-dimension frame can leave a ghost trail"
        )

    @pytest.mark.parametrize("name", ["drawEncounter", "drawEncounterEmpty"])
    def test_it_does_not_clear_only_the_css_sized_region(self, html, name):
        """clearRect(0,0,encounterW,encounterH) is the bug, in either spelling."""
        body = _fn(html, name)
        for bad in (r"clearRect\(\s*0\s*,\s*0\s*,\s*encounterW\s*,\s*encounterH\s*\)",
                    r"clearRect\(\s*0\s*,\s*0\s*,\s*W\s*,\s*H\s*\)"):
            assert not re.search(bad, body), f"{name} still clears a stale-dimension region"

    @pytest.mark.parametrize("name", ["drawEncounter", "drawEncounterEmpty"])
    def test_the_clear_comes_before_any_drawing(self, html, name):
        """A clear after the first stroke would wipe the frame it just drew."""
        body = _fn(html, name)
        clear_at = _FULL_CLEAR.search(body).start()
        first_draw = re.search(r"\b(?:drawGrid|fillText|beginPath|stroke|fillRect)\s*\(", body)
        assert first_draw, f"{name} draws nothing at all?"
        assert clear_at < first_draw.start(), f"{name} clears after it starts drawing"

    @pytest.mark.parametrize("name", ["drawEncounter", "drawEncounterEmpty"])
    def test_the_dpr_transform_is_restored_for_the_drawing(self, html, name):
        """Drawing math is in CSS pixels; it must not run under the identity transform."""
        body = _fn(html, name)
        m = _FULL_CLEAR.search(body)
        assert ".restore()" in body[m.start():m.end() + 2]


class TestTheFixStaysInItsLane:
    def test_resize_itself_is_unchanged(self, html):
        """The ticket scopes the change to setView and the two draw functions."""
        init = _fn(html, "initEncounter")
        assert "encounterW=wrap.clientWidth;encounterH=wrap.clientHeight;" in init
        assert re.search(r"if\(encounterW===0\|\|encounterH===0\)return;", init), (
            "resize()'s zero-size guard is gone"
        )
        assert re.search(r"canvas\.width=encounterW\*window\.devicePixelRatio;", init)

    def test_the_globe_autorotate_behaviour_is_not_touched(self, html):
        """Explicitly out of scope on the ticket."""
        assert re.search(r"function toggleGlobeAutoRotate", html)
