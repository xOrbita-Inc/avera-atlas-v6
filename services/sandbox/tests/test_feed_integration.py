"""
Integration test for SCRUM-373 AC2: proves the sandbox feed actually
produces something downstream in a real tracker, not just that a 200 came
back over the wire (John's review: "A 200 tells us the wire works, not
that the observation went anywhere").

Stands up the real tracker FastAPI app via uvicorn in a background
thread, bound to a free localhost port. This is a genuine request over a
real socket through the real app, not a mock and not an in-process ASGI
shortcut.

Requires uvicorn and the tracker service's own dependencies (fastapi,
pydantic, numpy) to be importable in whatever environment runs this test.
That is a real, separate question (does this run inside sandbox's own
minimal container, which currently only has pytest/numpy/httpx?) --
flagged, not silently resolved by expanding sandbox's Dockerfile to carry
another service's full dependency set.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest
import uvicorn

from services.sandbox import schema_emission
from services.sandbox.feed import feed

# services/tracker/main.py does a bare `from schemas import (...)`, which
# only resolves when services/tracker/ itself is on sys.path. That's true
# when pytest runs tests inside services/tracker/tests/, but not here in
# services/sandbox/tests/. sys.path insertion mirrors services/detector/
# tests/test_tracker_link.py's fix for the identical cross-service import
# problem, rather than changing tracker/main.py's import style (that file
# is AC5 signed-off; not touching it for an unrelated reason).
_TRACKER_DIR = Path(__file__).resolve().parents[3] / "services" / "tracker"

_TRACKER_MAIN_MODULE_NAME = "tracker_main_for_sandbox_feed_integration_test"


def _import_tracker_main() -> ModuleType:
    """
    Import services/tracker/main.py under a unique module name via
    importlib, rather than a bare `import main`.

    test_tracker_link.py (services/detector/tests/) does its own bare
    `import main` for DETECTOR's main.py, using the same sys.path-insert
    trick. Bare imports share Python's global sys.modules['main'] cache
    across the whole pytest process: if both test files run in one
    session (a full repo-root pytest run, which this ticket has done
    repeatedly), whichever imports first wins that cache slot and the
    other test silently gets the wrong service's main.py. A unique name
    here avoids that collision entirely, regardless of import order.
    """
    if _TRACKER_MAIN_MODULE_NAME in sys.modules:
        return sys.modules[_TRACKER_MAIN_MODULE_NAME]

    if str(_TRACKER_DIR) not in sys.path:
        sys.path.insert(0, str(_TRACKER_DIR))  # still needed for main.py's own bare `from schemas import ...`

    spec = importlib.util.spec_from_file_location(
        _TRACKER_MAIN_MODULE_NAME, _TRACKER_DIR / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_TRACKER_MAIN_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_tracker(monkeypatch, tmp_path):
    """
    Run the real tracker app in a background thread and point
    schema_emission's live-call target at it for the duration of the test.
    """
    tracker_main = _import_tracker_main()

    # Same DATA_DIR fix as test_v1_observations_iod.py's client fixture:
    # the production default is a container mount path, not writable here.
    monkeypatch.setattr(tracker_main, "DATA_DIR", str(tmp_path))

    # Fresh state for this test, regardless of what ran before it.
    tracker_main.state.correlation_engine.ucts.clear()
    tracker_main.state.observations.clear()
    tracker_main.state.tracks.clear()

    port = _free_port()
    config = uvicorn.Config(
        tracker_main.app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 5.0
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "tracker did not start within 5s"

    monkeypatch.setattr(schema_emission, "TRACKER_HOST", "127.0.0.1")
    monkeypatch.setattr(schema_emission, "TRACKER_PORT", str(port))
    monkeypatch.setattr(
        schema_emission,
        "TRACKER_OBSERVATIONS_URL",
        f"http://127.0.0.1:{port}/v1/observations",
    )

    yield tracker_main.state

    server.should_exit = True
    thread.join(timeout=5.0)


class TestFeedAgainstRealTracker:
    def test_feed_is_accepted_by_the_real_tracker(self, live_tracker):
        result = feed()
        assert result.accepted is True
        assert result.observation_count == 3

    def test_feed_produces_a_correlated_track_or_iod_result(self, live_tracker):
        """
        The bar John set: not just a 200, but proof the observation went
        somewhere. Assert the real tracker state shows either a
        correlated UCT (tracking in progress) or a completed track
        (IOD already closed) -- either is legitimate evidence the feed
        worked end to end, not just that the wire was up.
        """
        state = live_tracker

        feed()

        produced_uct = len(state.correlation_engine.ucts) >= 1
        produced_track = len(state.tracks) >= 1
        assert produced_uct or produced_track, (
            "feed() was accepted but produced neither a correlated UCT "
            "nor a track in the live tracker's state -- the wire works "
            "but nothing downstream happened"
        )

    def test_correlated_observations_are_tagged_v1_observations(self, live_tracker):
        """Provenance must carry through the real feed path, not just in
        the mocked unit tests."""
        state = live_tracker

        feed()

        assert len(state.correlation_engine.ucts) >= 1, (
            "expected at least one correlated UCT for this assertion to "
            "be meaningful"
        )
        uct = list(state.correlation_engine.ucts.values())[0]
        assert all(
            obs.ingest_path == "v1_observations" for obs in uct.observations
        )

    def test_observations_are_persisted_in_tracker_state(self, live_tracker):
        state = live_tracker

        result = feed()

        for observation_id in result.observation_ids:
            assert observation_id in state.observations
