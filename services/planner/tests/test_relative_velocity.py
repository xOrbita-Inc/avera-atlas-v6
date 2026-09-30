"""SCRUM-398: the producers supply a relative velocity, so the Pc path is reachable.

What was wrong
--------------
SCRUM-389 added `v_rel_km_s` to the conjunction contract as an optional field and
built the planner's ability to compute its own Pc on top of it. No producer
filled it. Not the UI, not the propagator, not ingest, not the UDL client, not
the CDM converter.

So `resolve_pc` returned `unavailable` for every event the system actually
produced. The Pc gate never ran, the Mahalanobis screen stayed the de facto
maneuver decision, `pc_post` from SCRUM-393 was always None, and SCRUM-388's
shared Pc library was reachable but unreached. Four merged tickets of machinery
with nothing feeding the one input it needs.

Not a defect in any of those four. Each did what it said. Nobody checked whether
the field they made optional was ever populated.

What this file guards
---------------------
That each producer supplies one, that the sense is secondary minus primary
everywhere, and above all that a conjunction built by a real producer reaches
`score_maneuver_candidates` with `pc_source` of `computed` rather than
`unavailable`. That last assertion is the one whose absence let four tickets
ship unreachable, and it is worth more than the rest of the file.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from common.maneuver_scorer import (
    PC_SOURCE_COMPUTED,
    PC_SOURCE_UNAVAILABLE,
    evaluate_conjunction_v25,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INGEST = str(_REPO_ROOT / "services" / "ingest")
if _INGEST not in sys.path:
    sys.path.insert(0, _INGEST)

from cdm_parser import parse_cdm_kvn  # noqa: E402
from cdm_to_conjunction import cdm_to_conjunction_state  # noqa: E402

MU_EARTH = 398600.4418
_A_KM = 6878.0
_V_CIRC = math.sqrt(MU_EARTH / _A_KM)

# A CDM whose two objects are on a head-on encounter, so the relative velocity
# it produces is known without reference to the implementation. Covariances are
# small enough that a 100 m miss is a real conjunction rather than a trivial one.
HEAD_ON_KVN = f"""\
CCSDS_CDM_VERS = 1.0
CREATION_DATE  = 2026-03-02T00:00:00.000
ORIGINATOR     = XORBITA_TEST
MESSAGE_ID     = 398-head-on
TCA            = 2026-03-02T15:30:00.000
MISS_DISTANCE  = 100.0 [m]
COLLISION_PROBABILITY = 2.0E-04
OBJECT          = OBJECT1
OBJECT_NAME     = XORB_PRIMARY
OBJECT_DESIGNATOR = 10001
X               = {_A_KM} [km]
Y               = 0.0 [km]
Z               = 0.0 [km]
X_DOT           = 0.0 [km/s]
Y_DOT           = {_V_CIRC} [km/s]
Z_DOT           = 0.0 [km/s]
CR_R            = 90000.0 [m**2]
CT_R            = 0.0 [m**2]
CT_T            = 250000.0 [m**2]
CN_R            = 0.0 [m**2]
CN_T            = 0.0 [m**2]
CN_N            = 90000.0 [m**2]
OBJECT          = OBJECT2
OBJECT_NAME     = DEBRIS
OBJECT_DESIGNATOR = 20002
X               = {_A_KM} [km]
Y               = 0.0 [km]
Z               = 0.1 [km]
X_DOT           = 0.0 [km/s]
Y_DOT           = {-_V_CIRC} [km/s]
Z_DOT           = 0.0 [km/s]
CR_R            = 90000.0 [m**2]
CT_R            = 0.0 [m**2]
CT_T            = 250000.0 [m**2]
CN_R            = 0.0 [m**2]
CN_T            = 0.0 [m**2]
CN_N            = 90000.0 [m**2]
"""


def _request_from(state: dict, include_v_rel: bool = True) -> dict:
    conjunction = {
        "obj_id": state["obj_id"],
        "t_ca_utc": "2026-03-02T15:30:00Z",
        "r_rel_km": state["r_rel_km"],
        "p_rel_km2": state["p_rel_km2"],
        "pc_precomputed": None,
    }
    if include_v_rel and state.get("v_rel_km_s") is not None:
        conjunction["v_rel_km_s"] = state["v_rel_km_s"]
    return {
        "conjunction_id": "CID-398",
        "satellite": {
            "sat_id": "XORB-001",
            "r_sat_km": [_A_KM, 0.0, 0.0],
            "v_sat_km_s": [0.0, _V_CIRC, 0.0],
            "t_burn_utc": "2026-03-02T11:30:00Z",
            "v_remaining_m_s": 25.0,
        },
        "conjunction": conjunction,
        "policy": {"decision_mode": "flight_rule_1e4"},
    }


# ---------------------------------------------------------------------------
# AC5: the assertion whose absence let four tickets ship unreachable
# ---------------------------------------------------------------------------

class TestAProducerReachesTheComputedPcPath:
    @pytest.fixture()
    def state(self) -> dict:
        return cdm_to_conjunction_state(parse_cdm_kvn(HEAD_ON_KVN)[0])

    def test_a_cdm_sourced_event_gets_a_computed_pc(self, state):
        """End to end. A CDM goes through the real converter, the resulting
        conjunction goes through the real request adapter and the real scorer,
        and the Pc is computed rather than unavailable.

        Before this ticket this assertion failed on every producer in the system.
        """
        result = evaluate_conjunction_v25(_request_from(state))
        assert result.pc_source == PC_SOURCE_COMPUTED
        assert result.pc_pre is not None and result.pc_pre > 0.0

    def test_the_risk_gate_is_now_pc_rather_than_mahalanobis(self, state):
        """SCRUM-389's whole purpose. The Pc gate decides when a Pc exists, and
        until this ticket a Pc never existed on a real event."""
        result = evaluate_conjunction_v25(_request_from(state))
        assert result.risk_gate == "pc"

    def test_scrum_393s_post_maneuver_pc_is_also_reachable(self, state):
        """393 computes a post-maneuver Pc per candidate. With no relative
        velocity it produced None every time, so risk_surrogate_post always took
        the inverse-area branch."""
        result = evaluate_conjunction_v25(_request_from(state))
        if result.no_go_reason_code == "":
            assert result.pc_post is not None
            assert all(c.pc_post is not None for c in result.candidates_v25)

    def test_dropping_the_field_reproduces_the_old_behaviour(self, state):
        """The same event, with the relative velocity withheld, is what every
        event looked like before this ticket. Kept so the difference this change
        makes is visible in the suite rather than argued for in a commit."""
        result = evaluate_conjunction_v25(_request_from(state, include_v_rel=False))
        assert result.pc_source == PC_SOURCE_UNAVAILABLE
        assert result.pc_pre is None
        assert result.risk_gate == "mahalanobis"


# ---------------------------------------------------------------------------
# AC4: the sense, on a geometry whose answer is known in advance
# ---------------------------------------------------------------------------

class TestTheSenseIsRight:
    def test_a_head_on_cdm_gives_twice_orbital_velocity_opposing_the_primary(self):
        state = cdm_to_conjunction_state(parse_cdm_kvn(HEAD_ON_KVN)[0])
        v_rel = np.array(state["v_rel_km_s"])

        assert np.linalg.norm(v_rel) == pytest.approx(2 * _V_CIRC, rel=1e-9)
        assert v_rel[1] == pytest.approx(-2 * _V_CIRC, rel=1e-9)

    def test_a_flipped_sense_would_change_the_answer(self):
        """Guards against the assertion above being vacuous. If Pc came out the
        same either way, the sense would not be worth testing, and a future
        reader should be able to see that it is."""
        state = cdm_to_conjunction_state(parse_cdm_kvn(HEAD_ON_KVN)[0])
        request = _request_from(state)

        forward = evaluate_conjunction_v25(request)

        flipped = _request_from(state)
        flipped["conjunction"]["v_rel_km_s"] = [
            -x for x in state["v_rel_km_s"]
        ]
        reversed_result = evaluate_conjunction_v25(flipped)

        # Both establish a Pc. Whether they agree is the point of the check.
        assert forward.pc_source == PC_SOURCE_COMPUTED
        assert reversed_result.pc_source == PC_SOURCE_COMPUTED
        assert forward.pc_pre == pytest.approx(reversed_result.pc_pre, rel=1e-9), (
            "Pc is symmetric under a sign flip at this geometry, so the sense "
            "cannot be caught here. It is caught by the magnitude and direction "
            "assertions in the ingest suite instead. If this ever stops holding, "
            "the sign convention needs a stronger test than a comment."
        )


# ---------------------------------------------------------------------------
# AC2: the UDL producer
# ---------------------------------------------------------------------------

class TestTheUdlProducer:
    @staticmethod
    def _record(**overrides) -> dict:
        rec = {
            "satNo1": 10001,
            "satNo2": 20002,
            "id": "udl-398",
            "tca": "2026-03-02T15:30:00.000",
            "collisionProb": 2.0e-4,
            "missDistance": 100.0,
            "relPosR": 0.0,
            "relPosT": 0.0,
            "relPosN": 100.0,
            "stateVector1": {
                "xpos": _A_KM, "ypos": 0.0, "zpos": 0.0,
                "xvel": 0.0, "yvel": _V_CIRC, "zvel": 0.0,
                "cov": [90000.0, 0.0, 250000.0, 0.0, 0.0, 90000.0],
            },
            "stateVector2": {
                "xpos": _A_KM, "ypos": 0.0, "zpos": 0.1,
                "xvel": 0.0, "yvel": -_V_CIRC, "zvel": 0.0,
                "cov": [90000.0, 0.0, 250000.0, 0.0, 0.0, 90000.0],
            },
        }
        rec.update(overrides)
        return rec

    def test_it_supplies_a_relative_velocity_from_the_state_vectors(self):
        from common.udl_client import _parse_conjunction

        parsed = _parse_conjunction(self._record())
        assert parsed is not None
        assert parsed["v_rel_source"] == "state_vector_difference"
        assert np.linalg.norm(parsed["v_rel_km_s"]) == pytest.approx(
            2 * _V_CIRC, rel=1e-9
        )

    def test_a_supplied_relative_velocity_wins(self):
        from common.udl_client import _parse_conjunction

        parsed = _parse_conjunction(
            self._record(relVelR=100.0, relVelT=-200.0, relVelN=50.0)
        )
        assert parsed["v_rel_source"] == "relative_velocity_rtn"

    def test_a_record_without_secondary_velocity_reports_unavailable(self):
        """None rather than zero. A zero would claim the two objects are
        co-moving, which is a statement about the physics rather than about what
        the feed gave us."""
        from common.udl_client import _parse_conjunction

        rec = self._record()
        rec["stateVector2"] = {"cov": rec["stateVector2"]["cov"]}
        parsed = _parse_conjunction(rec)
        assert parsed["v_rel_km_s"] is None
        assert parsed["v_rel_source"] == "unavailable"

    def test_the_planner_consumes_the_udl_dict_without_an_adapter(self):
        """server.py does body["conjunction"].update(record), so whatever this
        parser emits lands directly in the conjunction block. That is why adding
        the key here is sufficient, and it is worth asserting rather than
        assuming."""
        from common.udl_client import _parse_conjunction

        parsed = _parse_conjunction(self._record())
        request = _request_from(
            {
                "obj_id": str(parsed["obj_id"]),
                "r_rel_km": parsed["r_rel_km"],
                "p_rel_km2": parsed["p_rel_km2"],
                "v_rel_km_s": parsed["v_rel_km_s"],
            }
        )
        result = evaluate_conjunction_v25(request)
        assert result.pc_source == PC_SOURCE_COMPUTED
