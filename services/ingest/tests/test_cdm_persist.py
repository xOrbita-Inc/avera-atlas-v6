"""
SCRUM-429 -- save_cdm_record's source parameter, id return, and upsert.

Persist-on-evaluate offers the same conjunction to the store on every evaluate,
so the save has to be idempotent or the store grows a row per evaluate. The key
is the originating CDM's own id; without one it falls back to the encounter
(primary, secondary, tca).

The store is ADR-008's record of truth, so these tests are about what ends up in
it: how many rows, which id comes back, and what source each row claims.
"""
from __future__ import annotations

import pytest

from db import CdmRecord, get_session, init_db, save_cdm_record


@pytest.fixture(autouse=True)
def _clean_store():
    init_db()
    with get_session() as session:
        session.query(CdmRecord).delete()
    yield
    with get_session() as session:
        session.query(CdmRecord).delete()


def a_cdm(**overrides) -> dict:
    """A LeoLabs-shaped store dict, as to_store_cdm_dict emits."""
    base = {
        "OBJECT1_OBJECT_DESIGNATOR": "36508",
        "OBJECT2_OBJECT_DESIGNATOR": "270302",
        "TCA": "2026-08-30T10:24:48.130573Z",
        "MISS_DISTANCE": 20115.063,
        "COLLISION_PROBABILITY": 4.3858e-13,
        "COMMENT_ID": "76592783517",
        "COMMENT_EVENT_ID": "3538687541",
        "OBJECT1_CR_R": 1798279.0, "OBJECT1_CT_R": 877.2,
        "OBJECT1_CT_T": 8978958.0, "OBJECT1_CN_R": -4.4,
        "OBJECT1_CN_T": -79.0, "OBJECT1_CN_N": 2355285.0,
        "OBJECT2_CR_R": 1427261.0, "OBJECT2_CT_R": -11074.1,
        "OBJECT2_CT_T": 133839500.0, "OBJECT2_CN_R": 52.0,
        "OBJECT2_CN_T": 952.6, "OBJECT2_CN_N": 25724680.0,
    }
    base.update(overrides)
    return base


def _rows():
    """Rows as plain dicts. ORM instances detach when the session closes, so
    reading an attribute afterwards raises rather than returning the value."""
    with get_session() as session:
        return [
            {
                "id": r.id,
                "primary_norad": r.primary_norad,
                "secondary_norad": r.secondary_norad,
                "tca": r.tca,
                "miss_distance_m": r.miss_distance_m,
                "pc_space_track": r.pc_space_track,
                "cr_r": r.cr_r,
                "cn_n_sec": r.cn_n_sec,
                "source": r.source,
                "cdm_id": r.cdm_id,
                "event_id": r.event_id,
            }
            for r in session.query(CdmRecord).all()
        ]


# ---------------------------------------------------------------------------
# The source parameter and the id return
# ---------------------------------------------------------------------------


class TestSourceAndIdReturn:
    def test_a_save_returns_its_row_id_and_created(self):
        record_id, created = save_cdm_record(a_cdm(), source="leolabs")

        assert isinstance(record_id, int)
        assert created is True

    def test_the_returned_id_addresses_the_row_that_was_written(self):
        record_id, _ = save_cdm_record(a_cdm(), source="leolabs")

        with get_session() as session:
            row = session.query(CdmRecord).filter(CdmRecord.id == record_id).one()
            assert row.primary_norad == "36508"

    def test_the_caller_names_the_source(self):
        """The store has two real feeders now: the injected reference CDM and
        the live LeoLabs path."""
        save_cdm_record(a_cdm(), source="leolabs")

        assert _rows()[0]["source"] == "leolabs"

    def test_the_default_source_is_the_reference_cdm(self):
        save_cdm_record(a_cdm())

        assert _rows()[0]["source"] == "reference_cdm"

    def test_the_dedup_identifiers_are_stored(self):
        save_cdm_record(a_cdm(), source="leolabs")
        row = _rows()[0]

        assert row["cdm_id"] == "76592783517"
        assert row["event_id"] == "3538687541"

    def test_the_cdm_payload_lands_in_the_columns(self):
        save_cdm_record(a_cdm(), source="leolabs")
        row = _rows()[0]

        assert row["secondary_norad"] == "270302"
        assert row["miss_distance_m"] == pytest.approx(20115.063)
        assert row["pc_space_track"] == pytest.approx(4.3858e-13)
        assert row["cr_r"] == pytest.approx(1798279.0)
        assert row["cn_n_sec"] == pytest.approx(25724680.0)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestUpsertIdempotency:
    def test_persisting_the_same_cdm_twice_leaves_one_row(self):
        """The reason this exists: an evaluate can run many times against one
        conjunction, and each must not add a row."""
        first, created_first = save_cdm_record(a_cdm(), source="leolabs")
        second, created_second = save_cdm_record(a_cdm(), source="leolabs")

        assert len(_rows()) == 1
        assert first == second
        assert created_first is True
        assert created_second is False

    def test_a_different_cdm_id_inserts_a_second_row(self):
        first, _ = save_cdm_record(a_cdm(), source="leolabs")
        second, created = save_cdm_record(
            a_cdm(COMMENT_ID="99999999999"), source="leolabs"
        )

        assert len(_rows()) == 2
        assert first != second
        assert created is True

    def test_a_revised_cdm_for_the_same_event_is_its_own_row(self):
        """cdm_id identifies the message, event_id the event. Two revisions of
        one conjunction are distinct messages and both are worth keeping."""
        save_cdm_record(a_cdm(), source="leolabs")
        save_cdm_record(
            a_cdm(COMMENT_ID="76592783518", MISS_DISTANCE=19000.0), source="leolabs"
        )

        rows = _rows()
        assert len(rows) == 2
        assert {r["event_id"] for r in rows} == {"3538687541"}

    def test_the_reused_row_keeps_its_original_source(self):
        """A reuse returns the existing row untouched; it does not relabel it."""
        save_cdm_record(a_cdm(), source="leolabs")
        save_cdm_record(a_cdm(), source="real_cdm")

        assert len(_rows()) == 1
        assert _rows()[0]["source"] == "leolabs"

    def test_without_a_cdm_id_it_dedups_on_the_encounter(self):
        """The documented fallback: primary, secondary and TCA identify the
        encounter. Weaker than a message id, but it still stops a row per
        evaluate, which is what this is for."""
        payload = a_cdm()
        payload.pop("COMMENT_ID")
        first, created_first = save_cdm_record(payload, source="leolabs")
        second, created_second = save_cdm_record(payload, source="leolabs")

        assert len(_rows()) == 1
        assert first == second
        assert (created_first, created_second) == (True, False)

    def test_the_encounter_fallback_separates_different_encounters(self):
        payload = a_cdm()
        payload.pop("COMMENT_ID")
        other = dict(payload, OBJECT2_OBJECT_DESIGNATOR="11111")
        save_cdm_record(payload, source="leolabs")
        save_cdm_record(other, source="leolabs")

        assert len(_rows()) == 2

    def test_the_encounter_fallback_separates_different_tcas(self):
        payload = a_cdm()
        payload.pop("COMMENT_ID")
        later = dict(payload, TCA="2026-08-31T10:24:48Z")
        save_cdm_record(payload, source="leolabs")
        save_cdm_record(later, source="leolabs")

        assert len(_rows()) == 2

    def test_a_reference_cdm_is_also_idempotent(self):
        """Re-injecting TIROS reuses its row rather than stacking copies."""
        first, _ = save_cdm_record(a_cdm(COMMENT_ID=None), source="real_cdm")
        second, created = save_cdm_record(a_cdm(COMMENT_ID=None), source="real_cdm")

        assert len(_rows()) == 1
        assert first == second and created is False


# ---------------------------------------------------------------------------
# The add-column migration
# ---------------------------------------------------------------------------


class TestGuardedMigration:
    def test_the_columns_exist_after_init(self):
        from sqlalchemy import inspect
        from db import engine

        columns = {c["name"] for c in inspect(engine).get_columns("cdm_records")}

        assert {"cdm_id", "event_id"} <= columns

    def test_init_db_is_repeatable(self):
        """It runs on every startup, so adding an existing column must be a
        no-op rather than an error."""
        init_db()
        init_db()

        record_id, _ = save_cdm_record(a_cdm(), source="leolabs")
        assert isinstance(record_id, int)
