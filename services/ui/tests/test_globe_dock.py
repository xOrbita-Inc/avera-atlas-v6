"""tests/test_globe_dock.py

SCRUM-465: the globe's bottom-right dock -- asset switch, worst-conjunction card,
control row -- rebuilt to match the Live Conjunction Globe concept.

The one thing here that is a correctness question rather than a layout question is
the asset selection. The dock adds a second control over a state the LIVE ASSET
dropdown already owns, and two controls writing the same state independently is how
they drift. So the tests below pin that there is exactly ONE selection path: the
dock writes the dropdown and calls the dropdown's own handler, and nothing else
assigns the selection.

The rest is wiring: the dock holds the three panels in the concept's order, the card
is bound to the worst-conjunction data the page already has, the counts come only
from real responses, and the SCRUM-461/462 partial-view signal is untouched.
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


class TestTheDockHoldsTheConceptsThreePanels:
    def test_the_dock_exists(self, html):
        assert 'id="globeDock"' in html
        assert re.search(r"^\.globe-dock\{", html, re.M)

    def test_it_is_a_bottom_right_column(self, html):
        style = re.search(r"\.globe-dock\{([^}]*)\}", html).group(1)
        assert "flex-direction:column" in style
        assert "position:absolute" in style
        assert "right:" in style and "bottom:" in style

    def test_the_panels_are_in_the_concepts_order(self, html):
        """Asset switch, then the worst card, then the controls."""
        strip = html.index('id="globeAssetStrip"')
        card = html.index('id="globeWorstCard"')
        controls = html.index('id="globeControls"')
        assert strip < card < controls

    def test_the_control_row_keeps_its_id_and_buttons(self, html):
        """Restyled and re-parented, not rebuilt: the handlers are unchanged."""
        for btn in ("globeRotateBtn", "globeLabelsBtn", "globeNominalBtn",
                    "globeCovBtn", "globeFocusWorstBtn"):
            assert f'id="{btn}"' in html
        for handler in ("toggleGlobeAutoRotate()", "toggleGlobeLabels()",
                        "toggleGlobeNominal()", "toggleGlobeCovariance()",
                        "focusWorstConjunction()"):
            assert handler in html

    def test_the_dock_is_what_the_view_shows_and_hides(self, html):
        """One container toggled, so the panels cannot get out of step."""
        assert "getElementById('globeDock')?.classList.add('visible')" in html
        assert "getElementById('globeDock')?.classList.remove('visible')" in html
        assert "getElementById('globeControls')?.classList.add('visible')" not in html

    def test_the_covariance_caption_moved_into_the_dock(self, html):
        """It floated at a fixed offset that a multi-panel dock would cover."""
        note_at = html.index('id="globeCovNote"')
        dock_at = html.index('id="globeDock"')
        wrap_end = html.index('id="globeCanvas"')
        assert dock_at < note_at, "the caption is still outside the dock"
        style = re.search(r"\.globe-cov-note\{([^}]*)\}", html).group(1)
        assert "position:absolute" not in style
        decls = [d.split(":")[0].strip()
                 for d in style.replace("\n", " ").split(";") if ":" in d]
        assert "bottom" not in decls


class TestOneAssetSelectionPath:
    """The correctness question: two controls, one state."""

    def test_the_dock_writes_the_dropdown_and_calls_its_handler(self, html):
        fn = _fn(html, "selectGlobeAsset")
        assert "getElementById('liveAssetSelect')" in fn
        assert "sel.value=" in fn.replace(" ", "")
        assert "onLiveAssetChange()" in fn, (
            "the dock does not go through the dropdown's handler")

    def test_the_dock_never_assigns_the_selection_itself(self, html):
        """Assigning liveAssetNorad here would be the second path."""
        fn = _fn(html, "selectGlobeAsset")
        assert "liveAssetNorad=" not in fn.replace(" ", "")
        assert "liveAssetName=" not in fn.replace(" ", "")

    def test_only_the_dropdown_handler_assigns_the_selection(self, html):
        """Pins it globally: exactly one function writes liveAssetNorad."""
        assigns = [m.start() for m in
                   re.finditer(r"liveAssetNorad\s*=\s*(?!=)", html)]
        # the declaration, plus the single assignment in onLiveAssetChange
        handler = _fn(html, "onLiveAssetChange")
        handler_at = html.index(handler)
        inside = [i for i in assigns
                  if handler_at <= i < handler_at + len(handler)]
        decl = [i for i in assigns
                if html[max(0, i - 4):i].strip().endswith("let")]
        assert len(assigns) - len(inside) - len(decl) == 0, (
            "liveAssetNorad is assigned outside onLiveAssetChange; that is a "
            "second selection path")

    def test_the_dropdown_updates_the_dock(self, html):
        """Selecting in the dropdown must move the dock's highlight."""
        fn = _fn(html, "onLiveAssetChange")
        assert "renderGlobeAssetStrip()" in fn

    def test_selecting_in_the_dock_refreshes_the_globe(self, html):
        fn = _fn(html, "selectGlobeAsset")
        assert "fetchAndUpdateGlobeOrbits()" in fn

    def test_reselecting_the_current_asset_is_a_no_op(self, html):
        """Otherwise every click refetches a window the page already has."""
        fn = _fn(html, "selectGlobeAsset")
        assert "return" in fn.split("sel.value=")[0]

    def test_the_highlight_is_derived_from_the_shared_state(self, html):
        fn = _fn(html, "renderGlobeAssetStrip")
        assert "liveAssetNorad" in fn
        assert 'aria-pressed' in fn


class TestTheAssetStripDoesNotInventCounts:
    def test_counts_are_written_only_from_a_response(self, html):
        # `=` not followed by `=`, so the `=== undefined` comparison in the empty
        # state is not mistaken for an assignment.
        writes = re.findall(r"globeAssetCounts\[[^\]]*\]\s*=(?!=)", html)
        assert len(writes) == 1, f"counts written in {len(writes)} places"
        fetch = html.split("async function fetchLiveGlobeOrbits")[1].split("\n}")[0]
        assert "globeAssetCounts[String(liveAssetNorad)]=data.counts.events_in_window" \
            in fetch.replace(" ", "")

    def test_an_unvisited_asset_shows_no_count(self, html):
        """Ten assets cannot be counted without ten window pulls."""
        fn = _fn(html, "renderGlobeAssetStrip").replace(" ", "")
        assert "typeofn==='number'" in fn

    def test_the_strip_is_built_from_the_subscribed_list(self, html):
        assert "LEOLABS_ASSETS" in _fn(html, "renderGlobeAssetStrip")

    def test_the_strip_scrolls_rather_than_wrapping(self, html):
        style = re.search(r"\.dock-assets\{([^}]*)\}", html).group(1)
        assert "overflow-x:auto" in style


class TestTheWorstCardIsBoundToLiveData:
    @pytest.fixture(scope="class")
    def card(self, html) -> str:
        return _fn(html, "renderGlobeWorstCard")

    def test_it_reads_the_object_focus_worst_frames(self, card):
        assert "globeWorstMeta" in card

    @pytest.mark.parametrize("field", [
        "name", "risk_level", "miss_distance_m", "tca_utc",
    ])
    def test_it_binds_the_concepts_fields(self, card, field):
        assert field in card

    def test_it_shows_pc(self, card):
        assert "pc_display" in card or "m.pc" in card
        assert "Pc (LeoLabs)" in card

    def test_it_labels_the_asset_row(self, card):
        assert "liveAssetName" in card

    def test_the_covariance_badge_is_honest(self, card):
        """Real when the CDM carried one; not-reported when it did not."""
        assert "m.cov" in card
        assert "cov-badge" in card
        assert "REAL CDM" in card
        assert "NOT REPORTED" in card

    def test_the_count_comes_from_the_response(self, card):
        """On a truncated window this is the honest short count (SCRUM-459)."""
        assert "counts.events_in_window" in card
        assert "reporting volume" in card

    def test_not_yet_fetched_is_not_reported_as_empty_sky(self, card):
        """"Nothing out there" and "we have not looked" are different facts."""
        assert "No data yet" in card
        assert "No conjunction in the reporting volume" in card
        assert "globeAssetCounts" in card, (
            "the empty state cannot tell the two apart without knowing whether a "
            "fetch has happened")

    def test_switching_asset_drops_the_previous_assets_conjunction(self, html):
        """Caught live: the card kept the old object under the new asset's name.

        renderGlobeWorstCard reads globeWorstMeta and labels the Asset row from
        liveAssetName. If the meta survives a switch, the card shows the previous
        asset's encounter relabelled as this one's -- the same failure the 2D table
        clears itself to avoid. The state is cleared on the shared selection path,
        so both the dropdown and the dock get it.
        """
        fn = _fn(html, "onLiveAssetChange")
        assert "globeWorstMeta=null" in fn.replace(" ", ""), (
            "the worst conjunction survives an asset switch")
        assert "globeWorstNorad=null" in fn.replace(" ", "")

    def test_it_is_refreshed_on_fetch_and_on_switch(self, html):
        fetch = html.split("async function fetchLiveGlobeOrbits")[1].split("\n}")[0]
        assert "renderGlobeWorstCard(" in fetch
        assert "renderGlobeWorstCard(" in _fn(html, "onLiveAssetChange")
        assert "renderGlobeWorstCard(" in _fn(html, "selectGlobeAsset")


class TestThePartialViewHonestyIsIntact:
    """SCRUM-461/462 must survive the relayout."""

    def test_the_chip_and_toast_still_exist(self, html):
        assert 'id="partialChip"' in html
        assert 'id="partialToast"' in html

    def test_the_globe_fetch_still_drives_them(self, html):
        fetch = html.split("async function fetchLiveGlobeOrbits")[1].split("\n}")[0]
        assert "setPartialView(data)" in fetch
        assert "clearPartialView()" in fetch

    def test_the_dock_does_not_sit_on_the_chip(self, html):
        """The chip lives with the pager in the Active Conjunctions panel, which
        is outside the globe pane, so the dock cannot cover it."""
        chip_at = html.index('id="partialChip"')
        dock_at = html.index('id="globeDock"')
        wrap_at = html.index('<div id="globeWrap">')
        wrap_end = html.index('<div class="risk-bar">')
        assert wrap_at < dock_at < wrap_end, "the dock is not inside the globe pane"
        assert not (wrap_at < chip_at < wrap_end), (
            "the partial chip is inside the globe pane, where the dock could cover it")

    def test_switching_asset_still_clears_the_warning(self, html):
        """A dock switch goes through onLiveAssetChange, which clears it."""
        assert "clearPartialView()" in _fn(html, "clearConjunctionsForSourceSwitch")
        assert "clearConjunctionsForSourceSwitch()" in _fn(html, "onLiveAssetChange")
