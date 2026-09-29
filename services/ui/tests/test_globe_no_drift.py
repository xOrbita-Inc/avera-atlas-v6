"""tests/test_globe_no_drift.py

SCRUM-482: the 3D globe holds still.

The globe used to drift up and down and around on its own, separate from the rotate
toggle. The cause was a soft camera-follow that armed when the view opened and lerped
the OrbitControls target toward the asset marker every frame. The marker is animated
around its orbit, so the target chased a moving point forever, and that perpetual
chase was the drift. Focus Worst re-armed the same follow when its swing finished, so
the view slid off the conjunction the operator had just framed.

The fix is that the follow is never armed. These tests pin that: nothing in the
template sets cameraFollowActive true, the 3D-open path leaves it off, and the Focus
Worst swing does not re-arm it. The rotate toggle (OrbitControls autoRotate) is a
separate mechanism and is deliberately left alone.
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


class TestTheGlobeHoldsStill:
    def test_the_follow_is_never_armed(self, html):
        """No path in the template turns the soft camera-follow on."""
        assert re.search(r"cameraFollowActive\s*=\s*true", html) is None

    def test_opening_the_3d_view_leaves_the_follow_off(self, html):
        """setView's 3d branch opens the globe stationary."""
        set_view = _fn(html, "setView")
        assert re.search(r"cameraFollowActive\s*=\s*true", set_view) is None
        assert re.search(r"cameraFollowActive\s*=\s*false", set_view)

    def test_focus_worst_does_not_re_arm_the_follow(self, html):
        """The swing animates to a fixed point and holds; it does not resume the chase."""
        focus = _fn(html, "globeFocusOnConj")
        assert re.search(r"cameraFollowActive\s*=\s*true", focus) is None

    def test_the_rotate_toggle_is_untouched(self, html):
        """autoRotate is a separate mechanism and must still be wired."""
        toggle = _fn(html, "toggleGlobeAutoRotate")
        assert "autoRotate" in toggle
