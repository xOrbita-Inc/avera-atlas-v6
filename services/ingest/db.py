"""CDM persistence layer for the AVERA-ATLAS ingest service.

Architecture: ADR-008 (CDM Data Persistence and Service Ownership).
- This module is the sole writer to the CDM store.
- All other services are readers via the ingest REST API only.
- Stage 1: SQLite on a named Docker volume at /data/cdm_store/avera_atlas.db
- Stage 2 (post-preseed): swap engine URL to PostgreSQL; no other changes needed.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from sqlalchemy import create_engine, Column, Integer, Float, String, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# StaticPool is required for SQLite with a single connection shared across
# threads (FastAPI background tasks run on a thread pool).
_DB_URL = "sqlite:////data/cdm_store/avera_atlas.db"
engine = create_engine(
    _DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class CdmRecord(Base):
    """One row per ingested CCSDS 508.0-B-1 CDM event.

    Column names match the ADR-008 schema exactly and must not be renamed
    without a corresponding update to the ingest OpenAPI spec v1.0.0.

    Covariance elements are stored in m² (raw CCSDS values).
    The GET /cdm endpoint divides by 1e6 on assembly to return km².
    """
    __tablename__ = "cdm_records"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    primary_norad   = Column(String, nullable=False)
    secondary_norad = Column(String, nullable=False)
    tca             = Column(String, nullable=False)   # ISO 8601 UTC text
    miss_distance_m = Column(Float,  nullable=False)   # metres
    pc_space_track  = Column(Float,  nullable=True)    # Space-Track published Pc; null if absent

    # Primary object RTN covariance elements (m²)
    cr_r   = Column(Float, nullable=False)
    ct_r   = Column(Float, nullable=False)
    ct_t   = Column(Float, nullable=False)
    cn_r   = Column(Float, nullable=False)
    cn_t   = Column(Float, nullable=False)
    cn_n   = Column(Float, nullable=False)

    # Secondary object RTN covariance elements (m²)
    cr_r_sec = Column(Float, nullable=False)
    ct_r_sec = Column(Float, nullable=False)
    ct_t_sec = Column(Float, nullable=False)
    cn_r_sec = Column(Float, nullable=False)
    cn_t_sec = Column(Float, nullable=False)
    cn_n_sec = Column(Float, nullable=False)

    source      = Column(String, nullable=False)  # 'space_track' or 'synthetic'
    ingested_at = Column(String, nullable=False)  # ISO 8601 UTC text


class PlannerOutput(Base):
    """One row per APS planner decision linked to a CDM record.

    Written by the planner service via POST /planner_output (Prompt 5).
    Provides the audit trail required for LeoLabs and Satlyt demo validation.
    """
    __tablename__ = "planner_outputs"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    cdm_record_id    = Column(Integer, nullable=False)   # FK to cdm_records.id
    recommendation   = Column(String,  nullable=False)   # 'maneuver' | 'no_maneuver' | 'monitor'
    delta_v_ms       = Column(Float,   nullable=True)    # m/s; null when no maneuver recommended
    pc_computed      = Column(Float,   nullable=False)   # APS-computed Pc
    utility_value    = Column(Float,   nullable=False)
    lambda_v         = Column(Float,   nullable=False)
    lambda_l         = Column(Float,   nullable=False)
    covariance_source = Column(String, nullable=False)   # 'real_cdm' | 'surrogate_identity'
    created_at       = Column(String,  nullable=False)   # ISO 8601 UTC text


def init_db() -> None:
    """Create tables if they do not exist. Safe to call on every startup.

    Existing tables and their data are never modified or dropped.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("[DB] CDM store ready at %s", _DB_URL)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session; commit on success, rollback on exception."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_cdm_record(cdm: dict[str, Any]) -> None:
    """Write one CdmRecord row from a parsed CDM dict.

    The dict is the output of cdm_parser.parse_cdm_kvn() -- a flat dict
    with CCSDS field names prefixed by OBJECT1_ / OBJECT2_.

    OBJECT_DESIGNATOR values are parsed as floats by cdm_parser (e.g. 12345.0).
    This function casts them to clean NORAD ID strings.

    Call site: services/ingest/main.py CDM polling loop (added in Prompt 2).
    This function is intentionally not called from anywhere in Prompt 1.
    """
    def _norad(raw: Any) -> str:
        if isinstance(raw, float) and raw == int(raw):
            return str(int(raw))
        return str(raw)

    def _get_float(key: str, default: float = 0.0) -> float:
        val = cdm.get(key, default)
        return float(val) if val is not None else default

    record = CdmRecord(
        primary_norad   = _norad(cdm.get("OBJECT1_OBJECT_DESIGNATOR", "UNKNOWN")),
        secondary_norad = _norad(cdm.get("OBJECT2_OBJECT_DESIGNATOR", "UNKNOWN")),
        tca             = str(cdm.get("TCA", "")),
        miss_distance_m = _get_float("MISS_DISTANCE"),
        pc_space_track  = float(cdm["COLLISION_PROBABILITY"]) if cdm.get("COLLISION_PROBABILITY") is not None else None,
        cr_r    = _get_float("OBJECT1_CR_R"),
        ct_r    = _get_float("OBJECT1_CT_R"),
        ct_t    = _get_float("OBJECT1_CT_T"),
        cn_r    = _get_float("OBJECT1_CN_R"),
        cn_t    = _get_float("OBJECT1_CN_T"),
        cn_n    = _get_float("OBJECT1_CN_N"),
        cr_r_sec = _get_float("OBJECT2_CR_R"),
        ct_r_sec = _get_float("OBJECT2_CT_R"),
        ct_t_sec = _get_float("OBJECT2_CT_T"),
        cn_r_sec = _get_float("OBJECT2_CN_R"),
        cn_t_sec = _get_float("OBJECT2_CN_T"),
        cn_n_sec = _get_float("OBJECT2_CN_N"),
        source      = "space_track",
        ingested_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    with get_session() as session:
        session.add(record)
