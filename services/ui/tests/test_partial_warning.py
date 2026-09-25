"""tests/test_partial_warning.py

SCRUM-462: the partial-view warning as one toast plus one persistent chip.

partial_warning_harness.mjs extracts the real functions from index.html and drives
them against a stub DOM with controllable timers, so it can fire the auto-dismiss
and check the chip survives it. That is the property the redesign turns on: the
toast is allowed to disappear only because the chip is not.

This file runs that harness under pytest (skipped without node) and adds
template-wiring checks that need no node. The split is deliberate and was the point
SCRUM-461 made: a perfect render function that nothing calls still passes a pure
harness, and would ship a dashboard that stays silent on a truncated view.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_HARNESS = Path(__file__).parent / "partial_warning_harness.mjs"
_TEMPLATE = Path(__file__).parent.parent / "app" / "templates" / "index.html"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not available; the warning harness needs it")
def test_the_partial_warning_harness_passes():
    assert _HARNESS.exists(), f"missing harness at {_HARNESS}"
    proc = subprocess.run(
        ["node", str(_HARNESS)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        "the partial-warning harness failed:\n" + proc.stdout + "\n" + proc.stderr)
    assert "checks passed" in proc.stdout


@pytest.fixture(scope="module")
def html() -> str:
    return _TEMPLATE.read_text()


class TestTheOldBannersAreGone:
    """One signal, not three. The redesign must remove what it replaces."""

    @pytest.mark.parametrize("leftover", [
        "conjPartialBanner", "globePartialBanner", "partial-banner",
        "renderPartialBanner", "hidePartialBanner",
    ])
    def test_no_trace_of_the_scrum_461_banners(self, html, leftover):
        assert leftover not in html, (
            f"{leftover} survives; the old banner would show alongside the toast")


class TestSourceNeutralCopy:
    """The copy outlives today's provider, so it must not name it."""

    @pytest.fixture(scope="class")
    def copy_src(self, html) -> str:
        """The sentence builder, the copy wrapper and the head constant."""
        parts = []
        for fn in ("partialViewSentence", "partialViewCopy", "renderPartialChip"):
            m = re.search(r"^function " + fn + r"\(.*?^\}", html, re.S | re.M)
            assert m, f"{fn} is missing"
            parts.append(m.group(0))
        head = re.search(r"^const PARTIAL_HEAD = .*$", html, re.M)
        assert head
        parts.append(head.group(0))
        return "\n".join(parts)

    @pytest.mark.parametrize("name", ["LeoLabs", "leolabs", "LEOLABS", "Pulse"])
    def test_no_source_name_in_the_partial_copy(self, copy_src, name):
        assert name not in copy_src, (
            f"the partial-view copy names {name!r}; the warning will outlive the feed")

    def test_it_says_messages_are_not_in_risk_order(self, copy_src):
        assert "not in risk order" in copy_src

    def test_it_reads_only_the_three_backend_fields(self, copy_src):
        for field in ("cdms_pulled", "cdms_in_window", "truncated_by"):
            assert field in copy_src, f"{field} is not read"

    def test_both_truncation_reasons_are_translated(self, copy_src):
        assert "'deadline'" in copy_src and "time limit" in copy_src
        assert "'cap'" in copy_src and "size limit" in copy_src

    def test_the_head_names_the_risk(self, html):
        head = re.search(r"^const PARTIAL_HEAD = (.*)$", html, re.M).group(1)
        assert "PARTIAL VIEW" in head
        assert "WORST CONJUNCTION MAY NOT BE SHOWN" in head


class TestElementsAndStyle:
    def test_one_toast_and_one_chip_exist(self, html):
        assert html.count('id="partialToast"') == 1
        assert html.count('id="partialChip"') == 1

    def test_the_toast_is_app_level_not_per_panel(self, html):
        """One toast for the app: fixed to the viewport, declared once, at body top.

        Two toasts would let the table and the globe show different warnings about
        the same fetch, which is what the redesign exists to stop.
        """
        style = re.search(r"\.partial-toast\{([^}]*)\}", html).group(1)
        assert "position:fixed" in style
        body_at = html.index("<body>")
        toast_at = html.index('id="partialToast"')
        conj_at = html.index('id="conjToolbar"')
        assert body_at < toast_at < conj_at, "the toast is not at the top of the app"

    def test_the_chip_sits_with_the_pager(self, html):
        """The pager line is on screen in both the 2D and the globe view."""
        toolbar = html.split('id="conjToolbar"')[1].split("</div>\n        </div>")[0]
        assert 'id="partialChip"' in toolbar
        assert 'id="conjRange"' in toolbar

    def test_both_are_hidden_until_shown(self, html):
        toast = re.search(r"\.partial-toast\{([^}]*)\}", html).group(1)
        chip = re.search(r"\.partial-chip\{([^}]*)\}", html).group(1)
        # The toast is parked off-screen rather than display:none so it can slide.
        assert "translateY(-130%)" in toast
        assert "display:none" in chip
        assert ".partial-toast.show{" in html
        assert ".partial-chip.show{display:inline-flex}" in html

    def test_the_toast_has_a_close_control(self, html):
        copy_fn = re.search(r"^function partialViewCopy\(.*?^\}", html,
                            re.S | re.M).group(0)
        assert "dismissPartialToast()" in copy_fn
        assert "pt-close" in copy_fn


class TestTheChipIsTheDurableOne:
    """The structural half of the safety argument.

    The harness proves the behaviour; these prove the code is shaped so the
    behaviour cannot be undone by a later edit that looks harmless.
    """

    def test_dismissing_the_toast_does_not_touch_the_chip(self, html):
        fn = re.search(r"^function dismissPartialToast\(.*?^\}", html,
                       re.S | re.M).group(0)
        assert "partialChip" not in fn
        assert "renderPartialChip" not in fn
        assert "partialViewState" not in fn.replace("partialToastDismissed", "")

    def test_the_auto_dismiss_only_hides_the_toast(self, html):
        fn = re.search(r"^function showPartialToast\(.*?^\}", html,
                       re.S | re.M).group(0)
        timeout = fn[fn.index("setTimeout"):]
        assert "partialChip" not in timeout, (
            "the auto-dismiss reaches the chip; the condition would vanish with it")

    def test_the_chip_renders_from_the_state_not_from_the_toast(self, html):
        fn = re.search(r"^function renderPartialChip\(.*?^\}", html,
                       re.S | re.M).group(0)
        assert "partialViewState" in fn
        assert "partialToast" not in fn

    def test_the_chip_tooltip_carries_the_counts(self, html):
        fn = re.search(r"^function renderPartialChip\(.*?^\}", html,
                       re.S | re.M).group(0)
        assert "partialViewSentence" in fn
        assert ".title" in fn


class TestWiring:
    """Driven from the responses, and cleared on everything that changes the view."""

    def test_the_list_fetch_sets_it_from_its_response(self, html):
        fetch = html.split("async function fetchLiveConjunctions")[1].split("\n}")[0]
        assert "setPartialView(data)" in fetch

    def test_the_globe_fetch_sets_it_from_its_response(self, html):
        fetch = html.split("async function fetchLiveGlobeOrbits")[1].split("\n}")[0]
        assert "setPartialView(data)" in fetch

    @pytest.mark.parametrize("fn_name", [
        "async function fetchLiveConjunctions",
        "async function fetchLiveGlobeOrbits",
        "function clearConjunctionsForSourceSwitch",
        "function clearAll",
    ])
    def test_everything_that_changes_the_view_clears_it(self, html, fn_name):
        body = html.split(fn_name)[1].split("\n}")[0]
        assert "clearPartialView()" in body, f"{fn_name} does not clear the warning"

    def test_the_list_clears_before_it_fetches(self, html):
        """The error paths return early, so clearing has to come first."""
        fetch = html.split("async function fetchLiveConjunctions")[1].split("\n}")[0]
        assert fetch.index("clearPartialView()") < fetch.index("await fetch(")

    def test_the_globe_clears_before_it_fetches(self, html):
        fetch = html.split("async function fetchLiveGlobeOrbits")[1].split("\n}")[0]
        assert fetch.index("clearPartialView()") < fetch.index("await fetch(")

    def test_the_rows_are_not_gated_behind_the_warning(self, html):
        """The rows that came back are real; the warning sits with them."""
        fetch = html.split("async function fetchLiveConjunctions")[1].split("\n}")[0]
        table_at = fetch.index("updateConjunctionTable(rows)")
        warn_at = fetch.index("setPartialView(data)")
        assert table_at < warn_at
        assert "else" not in fetch[table_at:warn_at]

    def test_partial_is_read_as_an_explicit_flag(self, html):
        """SCRUM-461's invariant: absent fields mean complete, not partial."""
        fn = re.search(r"^function setPartialView\(.*?^\}", html, re.S | re.M).group(0)
        assert "data.partial === true" in fn
        assert "data.complete === false" in fn
