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

import os

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, text, UniqueConstraint, Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# StaticPool is required for SQLite with a single connection shared across
# threads (FastAPI background tasks run on a thread pool).
# Default is the production volume path. Overridable via AVERA_DB_URL so the
# store can be exercised in tests and local runs (SCRUM-351).
_DB_URL = os.environ.get("AVERA_DB_URL", "sqlite:////data/cdm_store/avera_atlas.db")
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
    pc_space_track  = Column(Float,  nullable=True)    # published Pc from the CDM; null if absent

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

    source      = Column(String, nullable=False)  # 'reference_cdm' | 'leolabs' | 'synthetic'
    ingested_at = Column(String, nullable=False)  # ISO 8601 UTC text

    # SCRUM-429: the originating CDM's own identifiers, nullable because rows
    # written before this change and any producer that has no such id still
    # have to store. cdm_id is the idempotency key for persist-on-evaluate:
    # re-evaluating one conjunction must reuse its row, not accumulate one per
    # evaluate.
    cdm_id   = Column(String, nullable=True, index=True)
    event_id = Column(String, nullable=True)


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


class DecisionLogRecord(Base):
    """Full DecisionLog audit record keyed by log_id (SCRUM-351).

    Persists the complete JSON audit record produced by the planner so an
    operator can retrieve any past decision by its decision ID (log_id) from
    the ATLAS UI. Created automatically by init_db()'s create_all; existing
    tables are never altered.
    """
    __tablename__ = "decision_logs"

    log_id            = Column(String, primary_key=True)   # decision ID
    conjunction_id    = Column(String, nullable=True)
    sat_id            = Column(String, nullable=True)
    decision          = Column(String, nullable=True)
    decision_log_json = Column(String, nullable=False)     # full DecisionLog.to_json()
    created_at        = Column(String, nullable=False)     # ISO 8601 UTC text


class EvidenceRecordRow(Base):
    """One append-only evidence record (SCRUM-377, MAF v2.0 section 10).

    Separate from decision_logs on purpose. decision_logs is SCRUM-351's
    retrieval-by-decision-ID store and keeps working unchanged; this table is
    the append-only chained audit trail. ADR-008 says existing tables are never
    altered, so the new semantics get a new table rather than a migration.

    Append-only is enforced in two places: the (chain_id, seq) unique constraint
    below, and the write path in main.py, which refuses a seq that is not the
    next one in the chain. A gap therefore cannot be created by a well-behaved
    writer, which is what makes a gap found later meaningful.

    canonical_payload is the exact string content_hash was computed over. Storing
    it means verification never has to re-derive a canonical form, so there is no
    second serializer to drift out of sync with the planner's.
    """
    __tablename__ = "evidence_records"

    record_id         = Column(String, primary_key=True)   # '{chain_id}#{seq}'
    chain_id          = Column(String, nullable=False)
    seq               = Column(Integer, nullable=False)
    record_type       = Column(String, nullable=False)     # 'decision' | 'transition'
    conjunction_id    = Column(String, nullable=True)
    prev_hash         = Column(String, nullable=False)
    content_hash      = Column(String, nullable=False)
    canonical_payload = Column(String, nullable=False)     # exact bytes hashed
    record_json       = Column(String, nullable=False)     # full record for retrieval
    created_at        = Column(String, nullable=False)     # ISO 8601 UTC text

    __table_args__ = (
        UniqueConstraint("chain_id", "seq", name="uq_evidence_chain_seq"),
        Index("ix_evidence_chain_seq", "chain_id", "seq"),
    )


def init_db() -> None:
    """Create tables if they do not exist. Safe to call on every startup.

    Existing tables and their data are never modified or dropped.
    """
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
    logger.info("[DB] CDM store ready at %s", _DB_URL)


# SCRUM-429: columns added after the table already existed somewhere.
# create_all does not alter an existing table, so a store file created before
# this change would keep the old shape and every insert naming these columns
# would fail. Additive only: ADD COLUMN on a nullable column rewrites no rows
# and cannot lose data, and an already-present column is skipped rather than
# retried. Deliberately not a migration framework -- one nullable column pair
# does not justify one, and this runs on a store whose only writer is us.
_ADDED_COLUMNS = (
    ("cdm_records", "cdm_id", "VARCHAR"),
    ("cdm_records", "event_id", "VARCHAR"),
)


def _add_missing_columns() -> None:
    """Add nullable columns an older store file predates. Never drops or alters."""
    from sqlalchemy import inspect

    inspector = inspect(engine)
    for table, column, sql_type in _ADDED_COLUMNS:
        try:
            existing = {c["name"] for c in inspector.get_columns(table)}
        except Exception:
            continue
        if column in existing:
            continue
        try:
            with engine.begin() as connection:
                connection.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
                )
            logger.info("[DB] added column %s.%s", table, column)
        except Exception as exc:
            # A store that cannot take the column still works: the upsert falls
            # back to (primary, secondary, tca). Logged loudly rather than
            # raised, because failing startup over an optional dedup key would
            # take the whole store down for a degraded feature.
            logger.warning(
                "[DB] could not add column %s.%s (%s); dedup will fall back to "
                "(primary_norad, secondary_norad, tca)", table, column, exc,
            )


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


def save_cdm_record(
    cdm: dict[str, Any], source: str = "reference_cdm"
) -> tuple[int, bool]:
    """Write one CdmRecord row from a parsed CDM dict, or reuse an existing one.

    The dict is the output of cdm_parser.parse_cdm_kvn() -- a flat dict with
    CCSDS field names prefixed by OBJECT1_ / OBJECT2_ -- which is also what
    ParsedLeoLabsCDM.to_store_cdm_dict() emits (SCRUM-429).

    OBJECT_DESIGNATOR values are parsed as floats by cdm_parser (e.g. 12345.0).
    This function casts them to clean NORAD ID strings.

    Returns (row_id, created). created is False when an existing row matched, so
    a caller can tell a fresh insert from a reuse without querying again.

    Idempotency (SCRUM-429). Persist-on-evaluate offers the same conjunction to
    the store on every evaluate, so this upserts rather than appending. The key
    is the originating CDM's own id when it has one, since that identifies the
    exact message. Without one it falls back to (primary_norad, secondary_norad,
    tca), which identifies the encounter -- weaker, because two revisions of one
    conjunction share it, but that is the right trade: reusing a row is
    recoverable, while duplicating one per evaluate is the unbounded growth this
    exists to prevent.
    """
    def _norad(raw: Any) -> str:
        if isinstance(raw, float) and raw == int(raw):
            return str(int(raw))
        return str(raw)

    def _get_float(key: str, default: float = 0.0) -> float:
        val = cdm.get(key, default)
        return float(val) if val is not None else default

    def _opt_str(key: str):
        val = cdm.get(key)
        return None if val is None else str(val)

    primary_norad = _norad(cdm.get("OBJECT1_OBJECT_DESIGNATOR", "UNKNOWN"))
    secondary_norad = _norad(cdm.get("OBJECT2_OBJECT_DESIGNATOR", "UNKNOWN"))
    tca = str(cdm.get("TCA", ""))
    cdm_id = _opt_str("COMMENT_ID")

    with get_session() as session:
        if cdm_id:
            existing = (
                session.query(CdmRecord).filter(CdmRecord.cdm_id == cdm_id).first()
            )
        else:
            existing = (
                session.query(CdmRecord)
                .filter(
                    CdmRecord.primary_norad == primary_norad,
                    CdmRecord.secondary_norad == secondary_norad,
                    CdmRecord.tca == tca,
                )
                .first()
            )
        if existing is not None:
            return int(existing.id), False

        record = CdmRecord(
            primary_norad   = primary_norad,
            secondary_norad = secondary_norad,
            tca             = tca,
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
            # SCRUM-431 retired Space-Track, so a stored real CDM reports its
            # actual origin. SCRUM-429: the caller names it, because the store
            # now has two real feeders -- the injected reference CDM and the
            # live LeoLabs path. The pc_space_track COLUMN keeps its name: it is
            # the published Pc from the originating CDM, and renaming it is a
            # migration.
            source      = source,
            cdm_id      = cdm_id,
            event_id    = _opt_str("COMMENT_EVENT_ID"),
            ingested_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        session.add(record)
        session.flush()          # assigns record.id inside this transaction
        return int(record.id), True
