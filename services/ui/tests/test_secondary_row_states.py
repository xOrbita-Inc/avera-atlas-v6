"""tests/test_secondary_row_states.py

SCRUM-457: run the Secondary conflict row harness under pytest.

row_state_harness.mjs extracts the real row-rendering and polling functions from
index.html and drives them against a stub DOM, asserting the honesty rule in every
state: only a screen that came back clear may paint CLEAR, and screening, error,
a cap timeout, an unreadable answer and a stale result for a superseded evaluate
all read SCREENING or NOT CLEAR.

The plan expected these to be covered live rather than offline, because the
dashboard has no JS unit harness. This is that harness. It does not replace seeing
the row repaint in a browser -- it cannot catch a CSS or layout fault -- but it does
pin the logic that decides CLEAR versus NOT CLEAR, which is the part that carries
the safety meaning, and it keeps that pinned for the next person to touch this row.

Skipped when node is not on PATH: node is not a dependency of the ui service (the
image serves a template, it does not build JS), so this runs where a developer has
node and stays out of the way where the container does not.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_HARNESS = Path(__file__).parent / "row_state_harness.mjs"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not available; the row harness needs it")
def test_the_secondary_conflict_row_harness_passes():
    assert _HARNESS.exists(), f"missing harness at {_HARNESS}"
    proc = subprocess.run(
        ["node", str(_HARNESS)],
        capture_output=True, text=True, timeout=120,
    )
    # The harness prints one line per check and exits non-zero on any failure, so
    # its own output is the failure message worth seeing.
    assert proc.returncode == 0, (
        "the Secondary conflict row harness failed:\n"
        + proc.stdout + "\n" + proc.stderr
    )
    assert "checks passed" in proc.stdout
