"""
Repo-root conftest.

SCRUM-388: puts the repository root on sys.path so `libs/aps_math` is importable
from every service's test suite the same way it is importable inside each
service image, where the build context is the repo root and the package is
copied to /app/aps_math.

Service-local conftest.py files still add their own service directory, so
`from common.X import ...` continues to work unchanged.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent
_LIBS = _REPO_ROOT / "libs"
for p in (str(_REPO_ROOT), str(_LIBS)):
    if p not in sys.path:
        sys.path.insert(0, p)
