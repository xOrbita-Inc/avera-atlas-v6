"""SCRUM-453 item 2: GET /cdm exposes the per-object covariances, not just the sum.

The two blocks were stored separately all along -- cr_r.. and cr_r_sec.. -- but
assembly summed them away, so the only covariance on the wire was
covariance_combined_rtn. That matrix is the right input for the scorer's Pc,
which wants the relative uncertainty of the pair, but it over-states the
PRIMARY's own uncertainty, and the secondary screen seeds from the primary's own.
So a decision from a stored or reference CDM had nothing to seed with and fell
back to the combined stand-in (SCRUM-454 measured that stand-in at 2.72x in sigma
and 7.4x in trace on a live CDM).

These tests pin the new fields and, more importantly, pin that they stay
consistent with the existing one: primary + secondary must still equal combined,
so nothing downstream that trusts combined is changed by splitting it out.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_INGEST_ROOT = Path(__file__).resolve().parents[1]
if str(_INGEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_INGEST_ROOT))
_spec = importlib.util.spec_from_file_location("ingest_main", _INGEST_ROOT / "main.py")
ingest_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingest_main)
app = ingest_main.app

from db import CdmRecord, get_session, init_db, save_cdm_record  # noqa: E402

_PRIMARY = "36508"
_SECONDARY = "270302"

# Deliberately asymmetric between the two objects, and with real off-diagonal
# terms, so a test cannot pass by accident on a symmetric or diagonal matrix.
_P = {"CR_R": 1798279.0, "CT_R": 877.2, "CT_T": 8978958.0,
      "CN_R": -4.4, "CN_T": -79.0, "CN_N": 2355285.0}
_S = {"CR_R": 1427261.0, "CT_R": -11074.1, "CT_T": 133839500.0,
      "CN_R": 52.0, "CN_T": 952.6, "CN_N": 25724680.0}


@pytest.fixture(autouse=True)
def _store_with_one_cdm():
    init_db()
    with get_session() as session:
        session.query(CdmRecord).delete()
    save_cdm_record({
        "OBJECT1_OBJECT_DESIGNATOR": _PRIMARY,
        "OBJECT2_OBJECT_DESIGNATOR": _SECONDARY,
        "TCA": "2026-09-24T10:24:48.130573Z",
        "MISS_DISTANCE": 20115.063,
        "COLLISION_PROBABILITY": 4.3858e-13,
        "COMMENT_ID": "cov-split-test",
        **{f"OBJECT1_{k}": v for k, v in _P.items()},
        **{f"OBJECT2_{k}": v for k, v in _S.items()},
    }, source="leolabs")
    yield
    with get_session() as session:
        session.query(CdmRecord).delete()


def _get():
    with TestClient(app) as c:
        r = c.get(f"/cdm/{_PRIMARY}/{_SECONDARY}")
    assert r.status_code == 200, r.text
    return r.json()


def _expected(d, key_pair):
    """The stored RTN block for one object, in km^2, as the endpoint returns it."""
    (rr, tr, tt, nr, nt, nn) = (d["CR_R"], d["CT_R"], d["CT_T"],
                                d["CN_R"], d["CN_T"], d["CN_N"])
    m = [[rr, tr, nr], [tr, tt, nt], [nr, nt, nn]]
    return [[v / 1e6 for v in row] for row in m]


class TestTheFieldsArePresent:
    def test_the_primary_block_is_returned(self):
        body = _get()
        assert "covariance_primary_rtn" in body
        assert body["covariance_primary_rtn"] == _expected(_P, None)

    def test_the_secondary_block_is_returned(self):
        body = _get()
        assert "covariance_secondary_rtn" in body
        assert body["covariance_secondary_rtn"] == _expected(_S, None)

    def test_the_combined_block_is_still_returned(self):
        """Nothing that already reads this endpoint may break."""
        body = _get()
        assert "covariance_combined_rtn" in body
        assert len(body["covariance_combined_rtn"]) == 3

    def test_every_block_is_three_by_three(self):
        body = _get()
        for key in ("covariance_primary_rtn", "covariance_secondary_rtn",
                    "covariance_combined_rtn"):
            m = body[key]
            assert len(m) == 3, key
            assert all(len(row) == 3 for row in m), key


class TestTheBlocksAreConsistent:
    """The new fields must not drift from the one that was already shipping."""

    def test_primary_plus_secondary_equals_combined(self):
        body = _get()
        p = body["covariance_primary_rtn"]
        s = body["covariance_secondary_rtn"]
        c = body["covariance_combined_rtn"]
        for i in range(3):
            for j in range(3):
                assert c[i][j] == pytest.approx(p[i][j] + s[i][j], rel=1e-12), (i, j)

    def test_the_primary_is_not_the_combined(self):
        """If it were, the seed would be no better than the stand-in it replaces."""
        body = _get()
        assert body["covariance_primary_rtn"] != body["covariance_combined_rtn"]

    def test_the_primary_trace_is_smaller_than_the_combined_trace(self):
        """Why this ticket exists: the combined over-states the primary's own
        uncertainty whenever the secondary carries real covariance."""
        body = _get()
        tr = lambda m: sum(m[i][i] for i in range(3))
        assert tr(body["covariance_secondary_rtn"]) > 0.0
        assert tr(body["covariance_primary_rtn"]) < tr(body["covariance_combined_rtn"])

    def test_the_blocks_are_symmetric(self):
        body = _get()
        for key in ("covariance_primary_rtn", "covariance_secondary_rtn"):
            m = body[key]
            for i in range(3):
                for j in range(3):
                    assert m[i][j] == pytest.approx(m[j][i]), (key, i, j)

    def test_the_unit_convention_matches_combined(self):
        """Stored m^2, returned km^2 -- the same divide the combined block uses."""
        body = _get()
        assert body["covariance_primary_rtn"][0][0] == pytest.approx(
            _P["CR_R"] / 1e6, rel=1e-12)
