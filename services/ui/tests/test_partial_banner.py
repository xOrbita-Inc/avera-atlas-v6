"""tests/test_partial_banner.py

SCRUM-461: run the partial-view banner harness under pytest.

partial_banner_harness.mjs extracts the real banner functions from index.html and
drives them against a stub DOM, asserting the honesty rule: a truncated window
announces itself, says the worst conjunction may not be shown, carries the
backend's counts, and never invents a denominator. Skipped where node is absent,
the same as the SCRUM-457 row harness.

There is also a template check below that does not need node: the markup and the
wiring have to exist for the harness's conclusions to mean anything at runtime.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_HARNESS = Path(__file__).parent / "partial_banner_harness.mjs"
_TEMPLATE = Path(__file__).parent.parent / "app" / "templates" / "index.html"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not available; the banner harness needs it")
def test_the_partial_banner_harness_passes():
    assert _HARNESS.exists(), f"missing harness at {_HARNESS}"
    proc = subprocess.run(
        ["node", str(_HARNESS)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        "the partial-banner harness failed:\n" + proc.stdout + "\n" + proc.stderr)
    assert "checks passed" in proc.stdout


class TestTemplateWiring:
    """The harness proves the functions behave; this proves they are connected.

    A perfect render function nothing calls would pass the harness and ship a
    dashboard that still renders a truncated window as complete.
    """

    @pytest.fixture(scope="class")
    def html(self) -> str:
        return _TEMPLATE.read_text()

    def test_both_banner_elements_exist(self, html):
        assert 'id="conjPartialBanner"' in html
        assert 'id="globePartialBanner"' in html

    def test_they_share_one_style_class(self, html):
        assert html.count('class="partial-banner"') >= 1
        assert 'class="partial-banner on-globe"' in html
        assert ".partial-banner{" in html
        # One definition of the loud style, not two that can drift apart.
        assert html.count(".partial-banner{") == 1

    def test_the_banner_is_hidden_until_shown(self, html):
        """A banner that defaults to visible would cry partial on every view."""
        style = re.search(r"\.partial-banner\{([^}]*)\}", html).group(1)
        assert "display:none" in style
        assert ".partial-banner.show{display:flex}" in html

    def test_the_list_fetch_renders_it_from_the_response(self, html):
        assert "renderPartialBanner('conjPartialBanner', data)" in html

    def test_the_globe_fetch_renders_it_from_the_response(self, html):
        assert "renderPartialBanner('globePartialBanner', data)" in html

    def test_both_fetches_clear_it_before_they_start(self, html):
        """So an error path cannot leave the previous asset's banner standing."""
        list_fn = html.split("async function fetchLiveConjunctions")[1].split(
            "\n}")[0]
        globe_fn = html.split("async function fetchLiveGlobeOrbits")[1].split(
            "\n}")[0]
        assert "hidePartialBanner('conjPartialBanner')" in list_fn
        assert "hidePartialBanner('globePartialBanner')" in globe_fn

    def test_the_rows_are_not_gated_behind_the_banner(self, html):
        """The rows that came back are real; the banner sits with them, not instead.

        updateConjunctionTable must run whether or not the view is partial, so the
        banner render comes after it and not in an else.
        """
        fetch = html.split("async function fetchLiveConjunctions")[1].split(
            "\n}")[0]
        table_at = fetch.index("updateConjunctionTable(rows)")
        banner_at = fetch.index("renderPartialBanner('conjPartialBanner', data)")
        assert table_at < banner_at
        assert "else" not in fetch[table_at:banner_at]

    def test_the_banner_markup_is_outside_the_table_body(self, html):
        """It must not land inside <tbody>, where a browser would hoist it out."""
        banner_at = html.index('id="conjPartialBanner"')
        split_at = html.index('class="conj-split"')
        assert banner_at < split_at

    def test_clearing_the_dashboard_hides_both(self, html):
        clear = html.split("function clearAll()")[1].split("\n}")[0]
        assert "hidePartialBanner('conjPartialBanner')" in clear
        assert "hidePartialBanner('globePartialBanner')" in clear

    def test_switching_asset_hides_both(self, html):
        """The banner describes one asset's window and must not outlive it."""
        switch = html.split("function clearConjunctionsForSourceSwitch()")[1].split(
            "\n}")[0]
        assert "hidePartialBanner('conjPartialBanner')" in switch
        assert "hidePartialBanner('globePartialBanner')" in switch

    def test_the_copy_names_the_backends_fields_and_no_others(self, html):
        """Invented fields would silently render nothing on a real response."""
        copy_fn = html.split("function partialViewCopy(truncation)")[1].split(
            "\n}")[0]
        for field in ("cdms_pulled", "cdms_in_window", "truncated_by"):
            assert field in copy_fn, f"{field} is not read"
        assert "cdm_total" not in copy_fn
        assert "total" not in copy_fn.replace("unreported total", "")
