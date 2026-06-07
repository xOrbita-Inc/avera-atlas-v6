"""
services/planner/conftest.py

Adds services/planner to sys.path so that test files can use
    from common.X import ...
    from avoid.X import ...
without a manual PYTHONPATH export.

pytest auto-discovers this file when running from the repo root:
    python -m pytest services/planner/tests/ -v
"""

import sys
from pathlib import Path

# Insert services/planner at the front of sys.path.
# __file__ is services/planner/conftest.py, so parent is services/planner.
_PLANNER_ROOT = Path(__file__).parent
if str(_PLANNER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLANNER_ROOT))
