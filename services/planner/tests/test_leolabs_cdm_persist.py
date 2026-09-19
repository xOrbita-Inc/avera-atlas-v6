"""
SCRUM-429 -- persist the LeoLabs CDM on evaluate and thread its store id.

Before this, the store's only feeder was the injected TIROS reference CDM, so no
real conjunction the system acted on reached ADR-008's record of truth, and the
audit write -- guarded on cdm_record_id -- never fired for a LeoLabs decision.

Three things under test, matching the plan:
  * the mapping from a parsed LeoLabs CDM to the store's flat CCSDS shape;
  * the upsert's idempotency, since persist-on-evaluate offers the same
    conjunction to the store on every evaluate;
  * the evaluate path threading the id, and staying unaffected when the persist
    fails.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from common.leolabs_cdm_parser import parse_leolabs_cdm

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"


@pytest.fixture
def cdm() -> dict:
    return json.loads(_FIXTURE.read_text())


@pytest.fixture
def parsed(cdm):
    return parse_leolabs_cdm(cdm, "L2669")


# ---------------------------------------------------------------------------
# 1. The mapping
# ---------------------------------------------------------------------------


class TestToStoreCdmDict:
    def test_the_designators_carry_the_norad_ids(self, parsed):
        """The store is keyed and looked up by NORAD
        (GET /cdm/{primary}/{secondary}), so that is what the designator field
        has to hold."""
        record = parsed.to_store_cdm_dict()

        assert record["OBJECT1_OBJECT_DESIGNATOR"] == "36508"
        assert record["OBJECT2_OBJECT_DESIGNATOR"] == "270302"

    def test_an_object_without_a_norad_id_falls_back_to_its_designator(self, cdm):
        """Better a real LeoLabs catalog id than a placeholder."""
        without = dict(cdm)
        without.pop("SAT2_COMMENT_NORAD_ID")
        record = parse_leolabs_cdm(without, "L2669").to_store_cdm_dict()

        assert record["OBJECT2_OBJECT_DESIGNATOR"] == "L143957"

    def test_the_encounter_scalars_are_carried(self, parsed, cdm):
        record = parsed.to_store_cdm_dict()

        assert record["TCA"] == parsed.t_ca_utc
        assert record["MISS_DISTANCE"] == pytest.approx(cdm["MISS_DISTANCE"])
        assert record["COLLISION_PROBABILITY"] == pytest.approx(
            cdm["COLLISION_PROBABILITY"]
        )

    def test_the_pc_is_the_cdms_own_not_the_planners(self, parsed, cdm):
        """The store column holds the published Pc from the originating CDM."""
        record = parsed.to_store_cdm_dict()

        assert record["COLLISION_PROBABILITY"] == parsed.cdm_collision_probability
        assert record["COLLISION_PROBABILITY"] == pytest.approx(4.3858e-13)

    @pytest.mark.parametrize("sat, prefix", [("SAT1", "OBJECT1"), ("SAT2", "OBJECT2")])
    def test_the_six_position_covariance_elements_map_across(self, parsed, cdm, sat, prefix):
        """A SATn_ to OBJECTn_ rename, not a conversion: same six lower-triangle
        position elements, same m^2 units."""
        record = parsed.to_store_cdm_dict()

        for element in ("CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N"):
            assert record[f"{prefix}_{element}"] == pytest.approx(
                cdm[f"{sat}_{element}"]
            ), f"{prefix}_{element}"

    def test_the_covariance_matches_the_parsed_matrix(self, parsed):
        """Read off the parsed 6x6, whose axis order is R,T,N,RDOT,TDOT,NDOT --
        so these are the lower triangle of its top-left 3x3 position block."""
        record = parsed.to_store_cdm_dict()
        cov = parsed.primary.cov_rtn_m2

        assert record["OBJECT1_CR_R"] == pytest.approx(cov[0, 0])
        assert record["OBJECT1_CT_R"] == pytest.approx(cov[1, 0])
        assert record["OBJECT1_CT_T"] == pytest.approx(cov[1, 1])
        assert record["OBJECT1_CN_R"] == pytest.approx(cov[2, 0])
        assert record["OBJECT1_CN_T"] == pytest.approx(cov[2, 1])
        assert record["OBJECT1_CN_N"] == pytest.approx(cov[2, 2])

    def test_the_dedup_identifiers_are_carried(self, parsed, cdm):
        record = parsed.to_store_cdm_dict()

        assert record["COMMENT_ID"] == cdm["COMMENT_ID"]
        assert record["COMMENT_EVENT_ID"] == cdm["COMMENT_EVENT_ID"]

    def test_the_shape_is_what_the_store_consumes(self, parsed):
        """Every key the store reads, flat and OBJECTn_-prefixed, same as
        cdm_parser.parse_cdm_kvn output."""
        record = parsed.to_store_cdm_dict()

        for key in (
            "OBJECT1_OBJECT_DESIGNATOR", "OBJECT2_OBJECT_DESIGNATOR", "TCA",
            "MISS_DISTANCE", "COLLISION_PROBABILITY",
            "OBJECT1_CR_R", "OBJECT1_CT_R", "OBJECT1_CT_T",
            "OBJECT1_CN_R", "OBJECT1_CN_T", "OBJECT1_CN_N",
            "OBJECT2_CR_R", "OBJECT2_CT_R", "OBJECT2_CT_T",
            "OBJECT2_CN_R", "OBJECT2_CN_T", "OBJECT2_CN_N",
        ):
            assert key in record, key

    def test_it_is_json_serialisable(self, parsed):
        """It is POSTed, so a stray numpy scalar would fail at the wire."""
        json.dumps(parsed.to_store_cdm_dict())

    def test_no_numpy_scalars_leak_into_the_mapping(self, parsed):
        record = parsed.to_store_cdm_dict()

        for key, value in record.items():
            assert not isinstance(value, np.generic), key


# ---------------------------------------------------------------------------
# 3. The evaluate path
# ---------------------------------------------------------------------------

# Mirrors test_leolabs_runtime's shape: LEOLABS_ENABLED on and the fetch
# returning a parsed CDM is what makes leolabs_used true, which is the only
# branch that persists.
_BODY = {
    "conjunction_id": "persist-1",
    "satellite": {
        "sat_id": "XORB-001",
        "r_sat_km": [6778.0, 0.0, 0.0],
        "v_sat_km_s": [0.0, 7.66, 0.0],
        "v_remaining_m_s": 25.0,
    },
    "policy": {
        "operator_id": "O", "policy_version": "2.5.7",
        "lambda_v": 0.01, "lambda_L": 0.001, "dv_mag_limit_m_s": 0.5,
    },
    "conjunction": {"primary_norad": 36508, "obj_id": "unknown"},
}


class TestEvaluateThreadsTheRecordId:
    def _run(self, parsed, monkeypatch, persist_response=None, persist_side_effect=None,
             persist_status=200):
        """A real LeoLabs evaluate with /cdm/persist mocked, capturing every POST."""
        posts = []

        def _post(url, *args, **kwargs):
            posts.append((url, kwargs.get("json")))
            if "/cdm/persist" in url:
                if persist_side_effect is not None:
                    raise persist_side_effect
                response = MagicMock(status_code=persist_status)
                response.text = "store error"
                response.json.return_value = (
                    persist_response if persist_response is not None
                    else {"id": 4242, "created": True}
                )
                return response
            return MagicMock(status_code=201)

        head = MagicMock()
        head.json.return_value = {"next_seq": 0, "content_hash": "0" * 64}
        monkeypatch.setattr(server, "LEOLABS_ENABLED", True)
        monkeypatch.setattr(server, "fetch_leolabs_conjunction", lambda n: parsed)
        monkeypatch.setattr(server.http_requests, "post", _post)
        monkeypatch.setattr(server.http_requests, "get", lambda *a, **k: head)
        response = TestClient(server.svc).post("/v1/evaluate", json=_BODY)
        return response, posts

    def _planner_output(self, posts):
        for url, payload in posts:
            if "/planner_output" in url:
                return payload
        return None

    def test_the_evaluate_really_is_on_the_leolabs_path(self, parsed, monkeypatch):
        """Guards the rest of this class: if leolabs_used were false these
        assertions would be measuring nothing."""
        response, _ = self._run(parsed, monkeypatch)

        assert response.status_code == 200
        assert response.json()["source"] == "leolabs"
        assert response.json()["covariance_source"] == "real_cdm"

    def test_the_leolabs_cdm_is_offered_to_the_store(self, parsed, monkeypatch):
        response, posts = self._run(parsed, monkeypatch)

        persisted = [p for url, p in posts if "/cdm/persist" in url]
        assert response.status_code == 200
        assert persisted, "a LeoLabs evaluate must offer its CDM to the store"
        assert persisted[0]["OBJECT1_OBJECT_DESIGNATOR"] == "36508"
        assert persisted[0]["COMMENT_ID"] is not None

    def test_the_returned_id_is_threaded_as_cdm_record_id(self, parsed, monkeypatch):
        """The whole point: the audit write is guarded on cdm_record_id, so
        without this a LeoLabs decision linked to nothing."""
        _, posts = self._run(parsed, monkeypatch, persist_response={"id": 4242, "created": True})

        output = self._planner_output(posts)
        assert output is not None, "the audit write must now fire for LeoLabs"
        assert output["cdm_record_id"] == 4242

    def test_a_reused_row_threads_the_same_id(self, parsed, monkeypatch):
        """created=False is a reuse, not a failure; the link is just as valid."""
        _, posts = self._run(parsed, monkeypatch, persist_response={"id": 99, "created": False})

        assert self._planner_output(posts)["cdm_record_id"] == 99

    def test_a_persist_that_raises_leaves_the_evaluate_intact(self, parsed, monkeypatch):
        """The guard. An audit write must never cost us a decision."""
        response, posts = self._run(
            parsed, monkeypatch, persist_side_effect=ConnectionError("ingest unreachable")
        )

        assert response.status_code == 200
        assert "recommendation" in response.json()
        assert self._planner_output(posts) is None

    def test_a_persist_error_status_leaves_the_evaluate_intact(self, parsed, monkeypatch):
        response, posts = self._run(parsed, monkeypatch, persist_status=500)

        assert response.status_code == 200
        assert "recommendation" in response.json()
        assert self._planner_output(posts) is None

    def test_a_persist_returning_no_id_leaves_the_link_absent(self, parsed, monkeypatch):
        """A malformed ack is not an id. Better no link than a wrong one."""
        _, posts = self._run(parsed, monkeypatch, persist_response={"created": True})

        assert self._planner_output(posts) is None

    def test_the_covariance_path_is_untouched(self, parsed, monkeypatch):
        """SCRUM-429 writes the CDM down; it does not change where the
        covariance comes from."""
        response, _ = self._run(parsed, monkeypatch)

        assert response.json()["covariance_source"] == "real_cdm"
        assert response.json()["source"] == "leolabs"

    def test_a_non_leolabs_evaluate_persists_nothing(self, monkeypatch):
        """Surrogate rows are not real conjunctions and must not enter the
        store. Only the LeoLabs branch persists."""
        posts = []

        def _post(url, *args, **kwargs):
            posts.append(url)
            return MagicMock(status_code=201)

        head = MagicMock()
        head.json.return_value = {"next_seq": 0, "content_hash": "0" * 64}
        monkeypatch.setattr(server, "LEOLABS_ENABLED", False)
        monkeypatch.setattr(server, "UDL_ENABLED", False)
        monkeypatch.setattr(server.http_requests, "post", _post)
        monkeypatch.setattr(server.http_requests, "get", lambda *a, **k: head)
        # A self-contained surrogate body: with LeoLabs off there is no fetch to
        # supply the geometry, so the request has to carry its own.
        surrogate = {
            **_BODY,
            "satellite": {**_BODY["satellite"], "t_burn_utc": "2026-06-22T08:00:00Z"},
            "conjunction": {
                "obj_id": "OBJ-1",
                "t_ca_utc": "2026-06-22T10:00:00Z",
                "r_rel_km": [0.1, 0.2, 0.3],
                "v_rel_km_s": [0.0, -0.5, 0.1],
                "p_rel_km2": [1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4],
            },
        }
        response = TestClient(server.svc).post("/v1/evaluate", json=surrogate)

        assert response.status_code == 200
        assert not [u for u in posts if "/cdm/persist" in u]


class TestPersistHelper:
    def test_a_mapping_failure_returns_none_rather_than_raising(self):
        broken = MagicMock()
        broken.to_store_cdm_dict.side_effect = ValueError("bad covariance")

        assert server._persist_leolabs_cdm(broken) is None

    def test_an_unreachable_ingest_returns_none(self, parsed):
        with patch.object(
            server.http_requests, "post", side_effect=ConnectionError("down")
        ):
            assert server._persist_leolabs_cdm(parsed) is None

    def test_a_good_response_returns_the_id(self, parsed):
        response = MagicMock(status_code=200)
        response.json.return_value = {"id": 7, "created": True}
        with patch.object(server.http_requests, "post", return_value=response):
            assert server._persist_leolabs_cdm(parsed) == 7

    def test_it_posts_the_mapped_dict_to_the_persist_route(self, parsed):
        response = MagicMock(status_code=200)
        response.json.return_value = {"id": 1, "created": True}
        with patch.object(
            server.http_requests, "post", return_value=response
        ) as post:
            server._persist_leolabs_cdm(parsed)

        url = post.call_args[0][0]
        assert url.endswith("/cdm/persist")
        assert post.call_args.kwargs["json"]["COMMENT_ID"] is not None
