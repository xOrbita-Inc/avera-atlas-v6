"""tests/test_leolabs_attribution.py

SCRUM-466: the LeoLabs attribution banner in the ARBITER header.

This suite is unlike the rest of the ui tests. The others pin behaviour, and a
reasonable future change is allowed to alter them. This one pins a CONTRACTUAL
STRING: LeoLabs require that any public material featuring their results state,
verbatim, "utilizing LeoLabs Pulse space safety service". So the assertions here
are deliberately literal, and a failure is not a layout nit -- it means the
dashboard is out of compliance and must not be shown.

Three things can silently break the credit, and there is a test for each:

  1. The words get edited, abbreviated or reworded. Pinned by comparing against a
     literal phrase spelled out in this file, so a change has to be made twice,
     in the template and here, to pass.
  2. The phrase gets split across elements -- a <b> around "LeoLabs Pulse", say.
     The rendered substring test would still pass while the element's text is no
     longer the required string, so the pill's text content is reconstructed from
     the parsed DOM and compared whole.
  3. The pill gets gated -- to live mode, to a LeoLabs fetch, to a feature flag --
     and is therefore absent exactly when results are on screen. Pinned by
     asserting the header block carries no Jinja conditional, that no script
     touches the element, and that no rule hides it.

Assertions are against the RENDERED page from the real app, not the template
source, because the rendered bytes are what a LeoLabs reviewer would see.

Run from repo root:
    python -m pytest services/ui/tests/test_leolabs_attribution.py -v
"""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

import app.main as ui_main

# The contract, spelled out. Do not soften this to a regex or a fragment.
REQUIRED_PHRASE = "utilizing LeoLabs Pulse space safety service"

_TEMPLATE = Path(__file__).parent.parent / "app" / "templates" / "index.html"


class _Header(HTMLParser):
    """Walks the document and records what the header actually contains.

    Collects, for the #leolabsAttrib element: its ancestor chain, and all text
    inside it concatenated in document order -- i.e. its text content, which is
    what has to equal the required phrase.

    Also records the order of identified elements inside .header-right, to check
    placement structurally rather than by comparing string offsets.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[tuple[str, dict]] = []
        self._capture_depth: int | None = None
        self.pill_ancestors: list[tuple[str, dict]] | None = None
        self.pill_attrs: dict | None = None
        self.pill_text: list[str] = []
        self.header_right_ids: list[str] = []

    @staticmethod
    def _classes(attrs: dict) -> set[str]:
        return set((attrs.get("class") or "").split())

    def _in_header_right(self) -> bool:
        return any("header-right" in self._classes(a) for _, a in self._stack)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id") == "leolabsAttrib":
            self.pill_ancestors = list(self._stack)
            self.pill_attrs = a
            self._capture_depth = len(self._stack)
        if self._in_header_right() and a.get("id"):
            self.header_right_ids.append(a["id"])
        # Void elements never nest, so they must not grow the stack.
        if tag not in ("br", "img", "input", "hr", "meta", "link", "source"):
            self._stack.append((tag, a))

    def handle_endtag(self, tag):
        while self._stack:
            popped, _ = self._stack.pop()
            if popped == tag:
                break
        if self._capture_depth is not None and len(self._stack) <= self._capture_depth:
            self._capture_depth = None

    def handle_data(self, data):
        if self._capture_depth is not None:
            self.pill_text.append(data)


@pytest.fixture(scope="module")
def rendered() -> str:
    """The dashboard as it ships: the real template, through the app's own Jinja
    environment, with the context the index endpoint actually passes.

    Not via TestClient. This checkout's local starlette (1.0) dropped the legacy
    positional TemplateResponse(name, context) call that main.py uses, so GET /
    raises here while working in the image, which pins fastapi 0.115.6. That
    mismatch is a local environment matter and out of scope for a template-only
    ticket, so the test renders through the version-stable half -- the same
    template, loader and environment the endpoint renders with.
    """
    ctx = {
        "request": None,
        "aps_version": ui_main.APS_VERSION,
        "leolabs_assets": ui_main.leolabs_subscribed_assets(),
    }
    # The loader's search path is the relative "app/templates", which resolves
    # only from the service root -- correct in the container, where that is the
    # WORKDIR. conftest.py does the same dance for the import.
    previous = os.getcwd()
    os.chdir(Path(__file__).parent.parent)
    try:
        return ui_main.templates.env.get_template("index.html").render(**ctx)
    finally:
        os.chdir(previous)


@pytest.fixture(scope="module")
def header(rendered) -> _Header:
    p = _Header()
    p.feed(rendered)
    return p


@pytest.fixture(scope="module")
def template() -> str:
    return _TEMPLATE.read_text()


class TestThePhraseIsStatedVerbatim:
    def test_the_rendered_page_contains_the_required_phrase(self, rendered):
        assert REQUIRED_PHRASE.lower() in rendered.lower(), (
            "The LeoLabs attribution phrase is not in the rendered dashboard. "
            "This string is contractually required on any public material."
        )

    def test_it_is_one_contiguous_run_not_assembled_from_pieces(self, rendered):
        # Collapsing whitespace first would let "utilizing  LeoLabs" or a run
        # broken over two source lines pass, so the check is on the raw bytes.
        assert rendered.lower().count(REQUIRED_PHRASE.lower()) == 1

    def test_the_pill_exists(self, header):
        assert header.pill_attrs is not None, "#leolabsAttrib is missing"
        assert "attrib-pill" in (header.pill_attrs.get("class") or "").split()

    def test_the_pills_text_content_is_exactly_the_phrase(self, header):
        # The load-bearing assertion. Guards against the phrase being split
        # across child elements, which a substring test cannot see.
        text = "".join(header.pill_text).strip()
        assert text.lower() == REQUIRED_PHRASE.lower(), repr(text)

    def test_the_phrase_is_not_abbreviated_anywhere(self, rendered):
        # The forms a well-meaning edit would reach for under space pressure.
        for wrong in ("LeoLabs Pulse SSS", "UTILIZING LEOLABS PULSE", "LeoLabs Pulse."):
            hits = [m.start() for m in re.finditer(re.escape(wrong.lower()), rendered.lower())]
            for at in hits:
                run = rendered[at : at + len(REQUIRED_PHRASE)]
                assert run.lower() == REQUIRED_PHRASE.lower(), (
                    f"found a truncated form of the credit: {rendered[at:at+70]!r}"
                )


class TestPlacementMatchesTheApprovedMockup:
    def test_the_pill_is_a_child_of_header_right(self, header):
        assert header.pill_ancestors is not None
        parent_tag, parent_attrs = header.pill_ancestors[-1]
        assert "header-right" in (parent_attrs.get("class") or "").split(), (
            f"#leolabsAttrib's parent is <{parent_tag} {parent_attrs}>, not .header-right"
        )

    def test_the_pill_sits_inside_the_header(self, header):
        assert any(
            "header" in (a.get("class") or "").split() for _, a in header.pill_ancestors
        ), "the pill is not inside .header"

    def test_it_comes_before_the_status_pill(self, header):
        ids = header.header_right_ids
        assert "leolabsAttrib" in ids and "sysStatus" in ids, ids
        assert ids.index("leolabsAttrib") < ids.index("sysStatus"), ids

    def test_it_carries_the_accent_dot_from_the_mockup(self, header, rendered):
        assert re.search(r"^\.attrib-dot\{", rendered, re.M), "the .attrib-dot rule is missing"
        assert 'class="attrib-dot"' in rendered

    def test_the_pill_is_styled_in_the_accent_not_the_status_green(self, rendered):
        rule = re.search(r"\.attrib-pill\{([^}]*)\}", rendered, re.S)
        assert rule, "the .attrib-pill rule is missing"
        body = rule.group(1)
        assert "--accent-dim" in body, body
        assert "--green" not in body, "the credit must not read as a second status light"
        # nowrap is what keeps the phrase from being broken mid-string.
        assert "white-space:nowrap" in body.replace(" ", "")


class TestItIsAlwaysOn:
    """A credit that can be absent will be absent during the one live moment."""

    def test_the_header_block_has_no_jinja_conditional(self, template):
        m = re.search(r'<div class="header">.*?\n  </div>', template, re.S)
        assert m, "could not locate the header block in the template"
        block = m.group(0)
        assert "leolabsAttrib" in block, "the pill is not in the block that was checked"
        # {{ expressions }} are fine; {% if %} / {% for %} would make it conditional.
        assert "{%" not in block, f"the header carries a Jinja statement tag:\n{block}"

    def test_no_script_touches_the_element(self, template):
        script_area = template[template.index("leolabsAttrib") + 1 :]
        for pattern in (
            r"getElementById\(\s*['\"]leolabsAttrib",
            r"querySelector\(\s*['\"][^'\"]*attrib-pill",
            r"\.attrib-pill['\"]\s*\)",
        ):
            assert not re.search(pattern, script_area), (
                f"the attribution pill is manipulated by script ({pattern}); "
                "it must be static so it cannot be hidden or rewritten at runtime"
            )

    def test_nothing_hides_it(self, rendered):
        assert "display:none" not in (rendered[
            rendered.index('id="leolabsAttrib"') - 200 : rendered.index('id="leolabsAttrib"') + 200
        ])
        # Including responsively: the narrow-window treatment must move the credit,
        # never remove it.
        for rule in re.finditer(r"([^{}]*attrib-pill[^{}]*|[^{}]*#leolabsAttrib[^{}]*)\{([^}]*)\}", rendered):
            assert "display:none" not in rule.group(2), rule.group(0)

    def test_the_element_is_not_marked_hidden(self, header):
        assert "hidden" not in header.pill_attrs
        assert "display:none" not in (header.pill_attrs.get("style") or "").replace(" ", "")

    def test_it_does_not_depend_on_a_feature_flag(self, template):
        # Every flag the template reads, to make sure none of them gate the pill.
        line = next(l for l in template.splitlines() if "leolabsAttrib" in l)
        assert "{{" not in line and "{%" not in line, line
