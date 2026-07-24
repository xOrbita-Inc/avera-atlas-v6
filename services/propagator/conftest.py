"""
services/propagator/conftest.py

Adds services/propagator to sys.path so test files can import the sibling
modules (pc_utils, and main.py loaded by path) when the suite runs from the
repo root:

    python -m pytest services/propagator/tests/ -v

Mirrors services/planner/conftest.py.
"""

import sys
from pathlib import Path

_PROPAGATOR_ROOT = Path(__file__).parent
if str(_PROPAGATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROPAGATOR_ROOT))
