"""Ingest test config: put the service root on sys.path and point the store
at a throwaway SQLite DB so the persistence layer can be exercised without
the production /data volume (SCRUM-351)."""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DBFILE = "/tmp/avera_ingest_pytest.db"
os.environ["AVERA_DB_URL"] = f"sqlite:///{_DBFILE}"
if os.path.exists(_DBFILE):
    os.remove(_DBFILE)
