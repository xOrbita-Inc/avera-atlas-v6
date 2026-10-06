"""tests/test_aps_zero_pc_gate.py

SCRUM-485: APS does not recommend a maneuver off the delta_c_legacy fallback.

On a live window, APS returned MANEUVER REQUIRED with a capped burn on hundreds
of conjunctions carrying no collision risk at all. The Pc-zero values behind them
were correct: negligible-risk in-volume passes whose geometric Pc underflows to
exactly 0.0. The defect was downstream of that.

The scorer prices a burn on one of two bases. It trades in Pc when a positive
geometric Pc exists; otherwise it falls back to the raw Mahalanobis separation
gain, whose benefit term the scorer's own comment calls unbounded. A zero Pc took
that fallback exactly as a missing Pc would, and the fallback produces utilities
in the tens of millions against roughly 2 for a real event. The APS trigger was
`utility > 0.0` with no reference to where the utility came from, so every one of
those events recommended a maneuver -- and since benefit rose with separation
without bound, the winning candidate was always the largest burn permitted.

So the tests here are mostly about one thing: the trigger must look at the BASIS,
not the number. A test that only checked "zero Pc does not maneuver" could pass
against a gate keyed on `pc > 0`, and that gate would still command a burn on an
event whose supplied Pc is nonzero while its geometric Pc underflows -- the same
fallback, the same unbounded utility. The basis is what distinguishes them.

The two non-maneuver outcomes are kept apart deliberately. "Assessed and safe" is
a claim; "no usable Pc" is the absence of one. Collapsing them would tell an
operator an unassessed event had been cleared.

Run from repo root:
    python -m pytest services/planner/tests/test_aps_zero_pc_gate.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from common.maneuver_scorer import (
    APS_STATE_NOT_REQUIRED,
    APS_STATE_NO_USABLE_PC,
    APS_STATE_REQUIRED,
    UTILITY_BASIS_DELTA_C,
    UTILITY_BASIS_PC,
    analyze_conjunction_modes,
    aps_decision_state,
    aps_pc_usable,
)
from common.operator_policy import OperatorPolicy

_FLIGHT_RULE_MODES = ("flight_rule_1e4", "flight_rule_1e5")

_SAT = {
    "sat_id": "485",
    "r_sat_km": [6853.0, 0.0, 0.0],
    "v_sat_km_s": [0.0, 7.62, 0.0],
    "t_burn_utc": "2026-04-14T08:00:00Z",
    "v_remaining_m_s": 50.0,
    "a_ref_km": 6853.0,
}
_TIGHT_COV = (np.eye(3) * 1.0e-4).flatten().tolist()


def _request(*, miss_km: float, with_rel_velocity: bool, pc_precomputed=None) -> dict:
    """An evaluate request whose geometry decides which basis the scorer takes.

    A wide separation against a tight covariance underflows the geometric Pc to
    exactly zero; a close one leaves a real Pc to trade. Dropping the relative
    velocity is what makes resolve_pc report unavailable rather than zero.
    """
    conjunction = {
        "t_ca_utc": "2026-04-14T12:00:00Z",
        "r_rel_km": [miss_km, 0.0, 0.0],
        "miss_distance_km": miss_km,
        "p_rel_km2": _TIGHT_COV,
    }
    if with_rel_velocity:
        conjunction["v_rel_km_s"] = [0.0, 10.0, 0.0]
    if pc_precomputed is not None:
        conjunction["pc_precomputed"] = pc_precomputed
    return {"conjunction_id": "485", "satellite": dict(_SAT),
            "policy": {}, "conjunction": conjunction}


@pytest.fixture(scope="module")
def pc_zero():
    """Computed Pc of exactly zero: a negligible-risk in-volume pass."""
    return analyze_conjunction_modes(_request(miss_km=10.0, with_rel_velocity=True))


@pytest.fixture(scope="module")
def pc_unavailable():
    """No relative velocity, so no Pc can be established at all."""
    return analyze_conjunction_modes(_request(miss_km=10.0, with_rel_velocity=False))


@pytest.fixture(scope="module")
def pc_traded():
    """A real, nonzero Pc the burn can trade down."""
    return analyze_conjunction_modes(_request(miss_km=0.05, with_rel_velocity=True))


# ---------------------------------------------------------------------------
# The defect, reproduced
# ---------------------------------------------------------------------------

class TestTheFallbackStillProducesTheHugeUtility:
    """This ticket gates the fallback; it does not fix or bound it.

    Pinned because it is what makes the rest of the suite non-vacuous. If the
    fallback ever stopped producing an enormous utility, every assertion below
    would still pass while testing nothing.
    """

    def test_a_zero_pc_event_takes_the_fallback_basis(self, pc_zero):
        assert pc_zero["pc_pre"] == 0.0
        assert pc_zero["pc_source"] == "computed"
        assert pc_zero["utility_basis"] == UTILITY_BASIS_DELTA_C

    def test_its_utility_is_enormous(self, pc_zero):
        """Tens of millions, against about 2 for a real event."""
        assert pc_zero["utility"] > 1.0e6

    def test_a_real_event_scores_orders_of_magnitude_lower(self, pc_traded):
        assert pc_traded["utility_basis"] == UTILITY_BASIS_PC
        assert pc_traded["utility"] < 1.0e3

    def test_so_a_utility_only_trigger_would_still_fire(self, pc_zero):
        """The old condition, stated explicitly: it is true on this event."""
        assert pc_zero["utility"] > 0.0


# ---------------------------------------------------------------------------
# Acceptance 1: a computed Pc of zero is NOT REQUIRED, with a zero burn
# ---------------------------------------------------------------------------

class TestComputedZeroPcIsNotRequired:
    def test_aps_does_not_require_a_maneuver(self, pc_zero):
        assert pc_zero["modes"]["aps"]["maneuver_required"] is False

    def test_the_commanded_burn_is_zero(self, pc_zero):
        assert pc_zero["modes"]["aps"]["dv_m_s"] == 0.0

    def test_the_state_says_assessed_and_safe(self, pc_zero):
        assert pc_zero["modes"]["aps"]["state"] == APS_STATE_NOT_REQUIRED

    def test_the_event_was_still_priced(self, pc_zero):
        """Not required is not the same as not analysable; the rollup counts on
        pricing_available staying true so the window is not reported partial."""
        assert pc_zero["modes"]["aps"]["pricing_available"] is True


# ---------------------------------------------------------------------------
# Acceptance 2: an unavailable Pc is its own state, and is not called safe
# ---------------------------------------------------------------------------

class TestUnavailablePcIsADistinctState:
    def test_no_pc_was_established(self, pc_unavailable):
        assert pc_unavailable["pc_pre"] is None
        assert pc_unavailable["pc_source"] == "unavailable"

    def test_it_is_not_a_maneuver(self, pc_unavailable):
        assert pc_unavailable["modes"]["aps"]["maneuver_required"] is False
        assert pc_unavailable["modes"]["aps"]["dv_m_s"] == 0.0

    def test_it_is_not_labelled_not_required(self, pc_unavailable):
        """The load-bearing distinction. NOT REQUIRED would tell an operator the
        event had been assessed and cleared, and it has not been assessed."""
        assert pc_unavailable["modes"]["aps"]["state"] == APS_STATE_NO_USABLE_PC
        assert pc_unavailable["modes"]["aps"]["state"] != APS_STATE_NOT_REQUIRED

    def test_the_two_zero_burn_states_are_distinguishable(
            self, pc_zero, pc_unavailable):
        """Both carry maneuver_required False and a zero burn, so the bool alone
        cannot tell them apart. The state is the only thing that can."""
        aps_zero = pc_zero["modes"]["aps"]
        aps_none = pc_unavailable["modes"]["aps"]
        assert aps_zero["maneuver_required"] == aps_none["maneuver_required"]
        assert aps_zero["dv_m_s"] == aps_none["dv_m_s"]
        assert aps_zero["state"] != aps_none["state"]


# ---------------------------------------------------------------------------
# Acceptance 4: a real Pc is untouched
# ---------------------------------------------------------------------------

class TestRealPcEventsAreUnchanged:
    def test_a_pc_traded_event_still_requires_a_maneuver(self, pc_traded):
        assert pc_traded["utility_basis"] == UTILITY_BASIS_PC
        assert pc_traded["modes"]["aps"]["maneuver_required"] is True
        assert pc_traded["modes"]["aps"]["state"] == APS_STATE_REQUIRED

    def test_it_still_carries_a_nonzero_burn(self, pc_traded):
        assert pc_traded["modes"]["aps"]["dv_m_s"] > 0.0

    def test_the_gate_is_what_lets_it_through(self, pc_traded):
        assert pc_traded["pc_usable"] is True


# ---------------------------------------------------------------------------
# Acceptance 5: flight-rule modes are untouched
# ---------------------------------------------------------------------------

class TestFlightRuleModesAreUnchanged:
    @pytest.mark.parametrize("mode", _FLIGHT_RULE_MODES)
    def test_a_zero_pc_event_was_never_a_flight_rule_maneuver(self, pc_zero, mode):
        """Pc zero is below every threshold, so these never fired on it. Pinned
        so the APS gate cannot be mistaken for having changed them."""
        assert pc_zero["modes"][mode]["maneuver_required"] is False

    @pytest.mark.parametrize("mode", _FLIGHT_RULE_MODES)
    def test_an_unavailable_pc_is_not_a_flight_rule_maneuver(
            self, pc_unavailable, mode):
        assert pc_unavailable["modes"][mode]["maneuver_required"] is False

    @pytest.mark.parametrize("mode", _FLIGHT_RULE_MODES)
    def test_a_real_pc_event_still_fires_the_flight_rules(self, pc_traded, mode):
        assert pc_traded["modes"][mode]["maneuver_required"] is True
        assert pc_traded["modes"][mode]["dv_m_s"] > 0.0

    @pytest.mark.parametrize("mode", _FLIGHT_RULE_MODES)
    def test_flight_rule_modes_carry_no_aps_state(self, pc_zero, mode):
        """A fixed-threshold mode makes no judgement about a utility basis, so
        reporting one of these states there would describe a decision it did
        not make."""
        assert "state" not in pc_zero["modes"][mode]

    @pytest.mark.parametrize("usable", [True, False])
    @pytest.mark.parametrize("mode", _FLIGHT_RULE_MODES)
    def test_the_gate_argument_cannot_move_a_flight_rule_verdict(self, mode, usable):
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0",
                                decision_mode=mode)
        threshold = policy.flight_rule_pc_threshold
        assert policy.is_recommendation_required(
            threshold, 2.0, utility=-1.0, pc_usable=usable) is True
        assert policy.is_recommendation_required(
            threshold * 0.5, 2.0, utility=1.0e9, pc_usable=usable) is False


# ---------------------------------------------------------------------------
# The gate keys off the basis, not the Pc value
# ---------------------------------------------------------------------------

class TestTheGateKeysOffThePcNotTheBasis:
    """Where the gate is drawn, and why it is not drawn at the basis.

    The fallback basis is where the huge utilities come from, and on every event
    in the reported window the two coincide: a computed Pc of zero takes the
    fallback. They come apart only when a Pc is supplied externally while the
    geometric Pc underflows.

    Gating on the basis there would refuse a burn on an event with a real,
    authority-supplied probability -- possibly above the maneuver threshold --
    and it would not even stop the burn, because the mode machine authorizes off
    the Pc threshold and would execute anyway while the recommendation beside it
    read no-burn. A gate that makes the response disagree with what the system
    does is worse than no gate, so the Pc is the test.
    """

    @pytest.fixture(scope="class")
    def supplied_pc_on_fallback(self):
        return analyze_conjunction_modes(
            _request(miss_km=10.0, with_rel_velocity=True, pc_precomputed=5.0e-3))

    def test_the_case_really_is_the_one_this_class_is_about(
            self, supplied_pc_on_fallback):
        """A real supplied Pc whose own geometric Pc underflows. Otherwise this
        class is testing nothing.

        It no longer takes the FALLBACK basis -- SCRUM-487 gave it a Pc-traded
        path of its own -- but it is still the case where the Pc and the basis
        would have disagreed, which is what the gate's design turns on.
        """
        assert supplied_pc_on_fallback["pc_pre"] == 5.0e-3
        assert supplied_pc_on_fallback["utility_basis"] == UTILITY_BASIS_PC

    def test_a_real_supplied_pc_keeps_its_maneuver(self, supplied_pc_on_fallback):
        """It has a real probability attached and is above the maneuver
        threshold; suppressing the burn would be the unsafe direction."""
        assert supplied_pc_on_fallback["pc_usable"] is True
        assert supplied_pc_on_fallback["modes"]["aps"]["maneuver_required"] is True
        assert supplied_pc_on_fallback["modes"]["aps"]["dv_m_s"] > 0.0

    def test_the_flight_rules_agree_with_that(self, supplied_pc_on_fallback):
        """5e-3 is above both thresholds. APS refusing while the flight rules
        maneuver would be the incoherence this gate must not create."""
        for mode in _FLIGHT_RULE_MODES:
            assert supplied_pc_on_fallback["modes"][mode]["maneuver_required"] is True

    def test_the_burn_size_is_now_fixed_too(self, supplied_pc_on_fallback):
        """This test used to assert the opposite.

        SCRUM-485 left the SIZE of this burn open and said so here: it was
        chosen by the unbounded fallback and pinned to the delta-v cap, which is
        a sizing defect on an event that does warrant some burn rather than the
        485 defect of commanding one on an event that warrants none. SCRUM-487
        closed it by pricing this case on the Pc-traded basis with the supplied
        Pc as the level, so the burn is now sized to the risk and sits well
        below the cap. Asserting the cap here now would pin a defect that has
        been fixed.
        """
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0")
        dv = supplied_pc_on_fallback["modes"]["aps"]["dv_m_s"]
        assert 0.0 < dv < policy.max_dv_per_event_ms

    def test_pc_usability_is_what_the_gate_reads(self):
        assert aps_pc_usable(1.0e-9) is True
        assert aps_pc_usable(0.0) is False
        assert aps_pc_usable(None) is False


class TestTheStateHelper:
    def test_a_required_event_reports_required(self):
        assert aps_decision_state(1e-4, True) == APS_STATE_REQUIRED

    def test_a_zero_pc_reports_not_required(self):
        """A probability was established and it says there is no risk."""
        assert aps_decision_state(0.0, False) == APS_STATE_NOT_REQUIRED

    def test_an_absent_pc_reports_no_usable_pc(self):
        assert aps_decision_state(None, False) == APS_STATE_NO_USABLE_PC

    def test_a_real_pc_that_did_not_justify_a_burn_is_not_required(self):
        """Assessed, and safe: there was a real Pc and the burn was not worth it."""
        assert aps_decision_state(1e-9, False) == APS_STATE_NOT_REQUIRED

    def test_an_absent_pc_is_never_called_not_required(self):
        """The distinction the whole state exists for."""
        assert aps_decision_state(None, False) != APS_STATE_NOT_REQUIRED


# ---------------------------------------------------------------------------
# The commanded burn, on the live evaluate path
# ---------------------------------------------------------------------------

class TestTheLiveEvaluateCommandsNoBurn:
    """The headline harm: APS COMMANDS a burn on these events.

    The verdict and the burn are produced in two different places. The evaluate
    response builds its recommendation straight off the scoring, and the
    artifact builds its own; gating only the maneuver_required flag left both of
    them still carrying a delta-v pinned to the policy cap. An operator reading
    either one would see a burn to fly.
    """

    @staticmethod
    def _evaluate(conjunction: dict, mode: str = "aps") -> dict:
        from unittest.mock import MagicMock, patch
        from fastapi.testclient import TestClient
        import server

        body = {
            "conjunction_id": "485-live",
            "satellite": dict(_SAT),
            "conjunction": conjunction,
            "policy": {"decision_mode": mode},
        }

        def _get(url, *args, **kwargs):
            resp = MagicMock(status_code=200, ok=True)
            resp.json.return_value = {"next_seq": 0, "content_hash": "0" * 64}
            return resp

        with patch.object(server, "UDL_ENABLED", False), \
             patch.object(server, "LEOLABS_ENABLED", False), \
             patch.object(server, "SECONDARY_SCREEN_ENABLED", False), \
             patch.object(server.http_requests, "post",
                          MagicMock(return_value=MagicMock(status_code=201))), \
             patch.object(server.http_requests, "get", side_effect=_get):
            response = TestClient(server.svc).post("/v1/evaluate", json=body)
        assert response.status_code == 200, response.text
        return response.json()

    def _zero_pc(self):
        return self._evaluate(_request(
            miss_km=10.0, with_rel_velocity=True)["conjunction"])

    def _unavailable(self):
        return self._evaluate(_request(
            miss_km=10.0, with_rel_velocity=False)["conjunction"])

    def _real_pc(self, mode="aps"):
        return self._evaluate(_request(
            miss_km=0.05, with_rel_velocity=True)["conjunction"], mode=mode)

    def test_a_zero_pc_event_commands_no_burn(self):
        rec = self._zero_pc()["recommendation"]
        assert rec["dv_magnitude_m_s"] == 0.0
        assert rec["dv_eci_km_s"] == [0.0, 0.0, 0.0]
        assert rec["direction"] == "no-burn"

    def test_an_unavailable_pc_event_commands_no_burn(self):
        rec = self._unavailable()["recommendation"]
        assert rec["dv_magnitude_m_s"] == 0.0
        assert rec["direction"] == "no-burn"

    def test_the_artifact_carries_no_recommendation_either(self):
        artifact = self._zero_pc()["atlas_artifact"]
        assert artifact["recommendation"] is None
        assert artifact["risk_summary"]["maneuver_required"] is False

    def test_the_artifact_says_which_conclusion_it_reached(self):
        zero = self._zero_pc()["atlas_artifact"]
        assert zero["risk_summary"]["aps_state"] == APS_STATE_NOT_REQUIRED
        assert zero["no_go"]["reason_code"] == "pc_zero_negligible_risk"

        unavailable = self._unavailable()["atlas_artifact"]
        assert unavailable["risk_summary"]["aps_state"] == APS_STATE_NO_USABLE_PC
        assert unavailable["no_go"]["reason_code"] == "no_usable_pc"

    def test_the_unassessed_event_is_not_described_as_safe(self):
        """Its no-go text must not read as a clearance."""
        no_go = self._unavailable()["atlas_artifact"]["no_go"]
        assert "NOT been assessed as safe" in no_go["human_readable"]

    def test_the_response_explains_why_the_burn_is_zero(self):
        """A huge utility beside a zero burn is unreadable without this."""
        metrics = self._zero_pc()["metrics"]
        assert metrics["aps_state"] == APS_STATE_NOT_REQUIRED
        assert metrics["utility_basis"] == UTILITY_BASIS_DELTA_C

    def test_a_real_pc_event_still_commands_its_burn(self):
        rec = self._real_pc()["recommendation"]
        assert rec["dv_magnitude_m_s"] > 0.0
        assert rec["direction"] != "no-burn"
        assert self._real_pc()["atlas_artifact"]["recommendation"] is not None

    @pytest.mark.parametrize("mode", _FLIGHT_RULE_MODES)
    def test_flight_rule_modes_are_untouched_on_the_live_path(self, mode):
        rec = self._real_pc(mode=mode)["recommendation"]
        assert rec["dv_magnitude_m_s"] > 0.0
        assert self._evaluate(
            _request(miss_km=10.0, with_rel_velocity=True)["conjunction"],
            mode=mode,
        )["recommendation"]["dv_magnitude_m_s"] == 0.0

    @pytest.mark.parametrize("mode", _FLIGHT_RULE_MODES)
    def test_flight_rule_modes_report_no_aps_state(self, mode):
        assert self._real_pc(mode=mode)["metrics"]["aps_state"] == ""


# ---------------------------------------------------------------------------
# Acceptance 3 and 6: the rollup
# ---------------------------------------------------------------------------

class TestThePortfolioRollup:
    """Both non-maneuver states have to drop out of the APS count and total."""

    def _summary(self, rows):
        """The rollup arithmetic, over controlled rows."""
        from common.maneuver_scorer import (
            APS_STATE_NOT_REQUIRED as NR, APS_STATE_NO_USABLE_PC as NU,
        )
        verdicts = [row["modes"]["aps"] for row in rows]
        return {
            "maneuver_count": sum(v["maneuver_required"] for v in verdicts),
            "known_dv_m_s": sum(v["dv_m_s"] for v in verdicts
                                if v["dv_m_s"] is not None),
            "not_required_count": sum(v.get("state") == NR for v in verdicts),
            "no_usable_pc_count": sum(v.get("state") == NU for v in verdicts),
        }

    def test_a_window_of_zero_pc_events_collapses_to_nothing(
            self, pc_zero, pc_unavailable):
        """The reported case: hundreds of events, none of them a real maneuver."""
        rows = [pc_zero] * 200 + [pc_unavailable] * 60
        summary = self._summary(rows)
        assert summary["maneuver_count"] == 0
        assert summary["known_dv_m_s"] == 0.0
        assert summary["not_required_count"] == 200
        assert summary["no_usable_pc_count"] == 60

    def test_the_real_events_survive_the_collapse(
            self, pc_zero, pc_unavailable, pc_traded):
        """Acceptance 6: the count reflects only the real-Pc events, and they
        are still there -- the fix must not zero the whole window."""
        rows = [pc_zero] * 200 + [pc_unavailable] * 60 + [pc_traded] * 4
        summary = self._summary(rows)
        assert summary["maneuver_count"] == 4
        assert summary["known_dv_m_s"] == pytest.approx(
            4 * pc_traded["modes"]["aps"]["dv_m_s"])
        assert summary["not_required_count"] == 200
        assert summary["no_usable_pc_count"] == 60

    def test_every_event_is_still_accounted_for(
            self, pc_zero, pc_unavailable, pc_traded):
        """A window that collapses from hundreds to a handful must not look like
        events went missing."""
        rows = [pc_zero] * 200 + [pc_unavailable] * 60 + [pc_traded] * 4
        summary = self._summary(rows)
        assert (summary["maneuver_count"] + summary["not_required_count"]
                + summary["no_usable_pc_count"]) == len(rows)


# ---------------------------------------------------------------------------
# Acceptance 8: the safety surface is untouched
# ---------------------------------------------------------------------------

class TestNoSafetySourceChanged:
    def test_is_maneuver_required_is_unchanged(self):
        """The hard trigger is not this ticket's business, in either direction."""
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0")
        assert policy.is_maneuver_required(1.0e-4, 2.0) is True
        assert policy.is_maneuver_required(1.0e-5, 2.0) is False
        assert policy.is_maneuver_required(0.0, 0.5) is True      # miss floor
        assert policy.is_maneuver_required(0.0, 2.0) is False

    def test_the_fixed_thresholds_are_unchanged(self):
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0")
        assert policy.pc_maneuver_threshold == 1.0e-4
        assert policy.pc_monitor_threshold == 1.0e-5
        for mode, threshold in (("flight_rule_1e4", 1.0e-4),
                                ("flight_rule_1e5", 1.0e-5)):
            assert OperatorPolicy(
                operator_id="T", policy_version="2.5.0", decision_mode=mode,
            ).flight_rule_pc_threshold == threshold

    def test_the_monitor_band_is_unchanged(self):
        policy = OperatorPolicy(operator_id="T", policy_version="2.5.0")
        assert policy.is_monitor_only(5.0e-5) is True
        assert policy.is_monitor_only(5.0e-4) is False
