"""
services/detector/conftest.py

SCRUM-352: let `python3 -m pytest` from the repo root collect and run this
service's tests without the cross-service top-level module-name collision.

detector, ingest and tracker each ship a top-level main.py. Under pytest's
default "prepend" import mode every test that does `from main import ...` binds
to a module cached in sys.modules under the bare name "main", so whichever
service is collected first wins and the others import the wrong main.py.

Two things are needed and both stay in test infrastructure only:

  1. At collect time, anchor this service root on sys.path and evict a sibling
     service's cached "main" so `from main import ...` binds to our module.
  2. At run time, restore our "main" into sys.modules before each test, because
     run-time string lookups such as unittest.mock.patch("main....") resolve
     sys.modules["main"] when the test executes, by which point a sibling
     service collected later would otherwise have replaced it.

No test files or production code are modified.
"""

import importlib
import sys
from pathlib import Path

_SVC_ROOT = str(Path(__file__).resolve().parent)

# Bare top-level module names this service shares with a sibling service.
# Only "main" collides today; listed explicitly so the intent is obvious.
_COLLIDING = ("main",)


def _anchor() -> None:
    """Put this service root ahead of sibling roots and drop foreign collisions."""
    if _SVC_ROOT in sys.path:
        sys.path.remove(_SVC_ROOT)
    sys.path.insert(0, _SVC_ROOT)
    for name in _COLLIDING:
        cached = sys.modules.get(name)
        if cached is not None and not (getattr(cached, "__file__", "") or "").startswith(_SVC_ROOT):
            del sys.modules[name]


_anchor()

# Hold this service's colliding modules so both import-time symbol binding and
# run-time sys.modules lookups refer to the same objects throughout our tests.
_OWNED = {}
for _name in _COLLIDING:
    try:
        _OWNED[_name] = importlib.import_module(_name)
    except Exception:
        # Service may not import cleanly at collect time; its own tests surface
        # that. Do not break collection of the rest of the suite.
        pass


def pytest_runtest_setup(item):
    """Restore this service's colliding modules before each of its tests run."""
    for _name, _mod in _OWNED.items():
        sys.modules[_name] = _mod
