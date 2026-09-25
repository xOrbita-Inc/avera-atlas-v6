"""
services/ui/conftest.py

Adds services/ui to sys.path so test files can use
    from app.main import app
without a manual PYTHONPATH export. Mirrors services/planner/conftest.py.

It also imports app.main here, with the working directory temporarily set to
services/ui. main.py mounts its static files at the relative path "app/static",
which resolves only from that directory -- correct in the container, where the
WORKDIR is the service root, but not when pytest runs from the repo root. Doing
the import here and restoring the working directory immediately keeps the fix in
the test harness rather than changing how the service mounts its assets, and
leaves no global cwd change behind for other suites in the same session.

pytest auto-discovers this when running from the repo root:
    python -m pytest services/ui/tests/ -v
"""

import os
import sys
from pathlib import Path

_UI_ROOT = Path(__file__).parent.resolve()
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))

_PREVIOUS_CWD = os.getcwd()
os.chdir(_UI_ROOT)
try:
    import app.main  # noqa: F401  (imported for its side effect: the app object)
finally:
    os.chdir(_PREVIOUS_CWD)
