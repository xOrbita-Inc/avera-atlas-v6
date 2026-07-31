"""SCRUM-393: risk_surrogate_post reports the burn's own Pc.

What was wrong
--------------
score_maneuver_candidates assigned pc_precomputed to risk_surrogate_post
whenever an external Pc was supplied. That is the PRE-maneuver probability
under a post-maneuver name, so the field reported that the recommended burn
reduced risk by exactly zero, and it reported it most confidently on the events
carrying the most real data. Events without an external Pc got 1/m2_post, which
does respond to the burn but is an inverse square kilometre rather than a
probability.

Why it mattered more than a reported metric usually does
--------------------------------------------------------
server.py mapped risk_surrogate_post onto pc_computed in the audit write to
ingest /planner_output. pc_computed is a non-nullable column described as
"APS-computed Pc" and a required field of RiskInputs in gnc_interface.yaml. So
both wrong values were persisted into the SCRUM-377 evidence trail, which is
tamper-evident, meaning wrong entries are preserved rather than corrected. The
inverse-area value lands between 1e-4 and 1e-2 at the covariances this system
produces, which is exactly where a real Pc lives, so nothing about the number
invited anyone to check it.

What this file guards
---------------------
That the field carries the burn's own Pc, that the Pc is a real one rather than
a plausible-looking number, that every candidate carries its own so SCRUM-387
can trade in Pc, that Q_exec is included, that the approximation in
compute_pc_post is within the bound its docstring claims, and that the audit
write can no longer receive an inverse area.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aps_math import conventions

from avoid.decision_model import cw_phi_rv
from common.maneuver_scorer import (
    RISK_SURROGATE_INVERSE_M2,
    RISK_SURROGATE_PC_POST,
    RISK_SURROGATE_PC_PRE_NO_BURN,
    compute_pc_from_geometry,
    compute_pc_post,
    resolve_hard_body_radius,
    score_maneuver_candidates,
)
from common.operator_policy import OperatorPolicy
from common.satellite_capability import (
    LifetimeProfile,
    PropulsionProfile,
    SatelliteCapability,
)

MU_EARTH = 398600.4418
_A_KM = 6378.137 + 550.0
_V_CIRC = math.sqrt(MU_EARTH / _A_KM)

# Axis-aligned on purpose. R_hat is +x, T_hat is +y, N_hat is +z, so the RTN
# ordering the CW blocks are written in coincides with ECI here. That keeps this
# file measuring Pc rather than measuring a frame conversion.
R_SAT = np.array([_A_KM, 0.0, 0.0])
V_SAT = np.array([0.0, _V_CIRC, 0.0])
V_REL_HEAD_ON = np.array([0.0, -2.0 * _V_CIRC, 0.0])

R_REL = np.array([0.0, 0.0, 0.3])
P_SURROGATE = np.diag([0.3 ** 2, 2.5 ** 2, 0.5 ** 2])

T_CA = "2026-03-02T15:30:00Z"

# A ten-minute lead with a small delta-v ceiling. Chosen so the burn moves the
# object enough to change Pc measurably but not enough to drive it to underflow.
# At the four-hour lead used elsewhere the recommended burn takes Pc to exactly
# 0.0 in double precision, which is a true answer but a useless test: it passes
# against any implementation that returns something small.
T_BURN_SHORT = "2026-03-02T15:20:00Z"
DV_CEILING_SMALL = 0.02

T_BURN_LONG = "2026-03-02T11:30:00Z"


def _cap(**propulsion) -> SatelliteCapability:
    base = dict(min_dv_m_s=0.001)
    base.update(propulsion)
    return SatelliteCapability(
        sat_id="TEST-6U",
        a_ref_km=_A_KM,
        propulsion=PropulsionProfile(**base),
        lifetime=LifetimeProfile(
            mass_kg=12.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
            mission_lifetime_days_remaining=365.0,
        ),
    )


def _policy(**kw) -> OperatorPolicy:
    base = dict(
        operator_id="TEST", policy_version="2.5.0",
        pc_maneuver_threshold=1.0e-4, pc_monitor_threshold=1.0e-5,
        mahalanobis_screen_threshold=4.0,
    )
    base.update(kw)
    return OperatorPolicy(**base)


def _score(cap=None, t_burn=T_BURN_SHORT, dv_ceiling=DV_CEILING_SMALL,
           v_rel=V_REL_HEAD_ON, r_rel=R_REL, **kw):
    return score_maneuver_candidates(
        "CID-393", R_SAT, V_SAT, r_rel, P_SURROGATE, t_burn, T_CA,
        cap or _cap(), _policy(max_dv_per_event_ms=dv_ceiling),
        v_rel_km_s=v_rel, **kw
    )


# ---------------------------------------------------------------------------
# AC1 and AC2: the field stops echoing the pre-maneuver value
# ---------------------------------------------------------------------------

class TestTheFieldRespondsToTheBurn:
    def test_the_fixture_actually_recommends_a_burn(self):
        """Every assertion below is about a recommended maneuver. If the
        geometry ever stops producing one they would pass vacuously, so fail
        here instead."""
        result = _score()
        assert result.no_go_reason_code == ""
        assert result.direction != "no-burn"

    def test_a_supplied_pc_is_no_longer_echoed_into_the_post_field(self):
        """AC2. This is the assertion whose absence let the defect through.

        Before the fix, risk_surrogate_post equalled pc_precomputed exactly, so
        the burn appeared to reduce risk by zero.
        """
        supplied = 2.5e-4
        result = _score(pc_precomputed=supplied)

        assert result.pc_pre == pytest.approx(supplied)
        assert result.risk_surrogate_post != pytest.approx(supplied, rel=1e-9)
        assert result.risk_surrogate_post == pytest.approx(result.pc_post)
        assert result.risk_surrogate_source == RISK_SURROGATE_PC_POST

    def test_a_supplied_pre_maneuver_pc_still_wins_for_pc_pre(self):
        """AC3. SCRUM-389 gave an external Pc precedence for the pre-maneuver
        figure, because a CDM or UDL Pc comes from an authority with more
        information than we have. That precedence is unchanged, and it does not
        extend to the post-maneuver figure, because nobody external has computed
        the Pc after a burn we have not flown."""
        supplied = 2.5e-4
        result = _score(pc_precomputed=supplied)
        assert result.pc_pre == pytest.approx(supplied)
        assert result.pc_post != pytest.approx(supplied, rel=1e-9)

    def test_the_recommended_burn_reduces_pc(self):
        result = _score()
        assert result.pc_post < result.pc_pre
        assert result.pc_post > 0.0, (
            "underflowed to zero, so this fixture no longer discriminates "
            "between implementations"
        )


# ---------------------------------------------------------------------------
# The number is a real Pc, not a plausible-looking one
# ---------------------------------------------------------------------------

class TestPcPostIsAnchored:
    def test_it_matches_an_independent_computation_on_the_post_burn_geometry(self):
        """Rebuilds the post-burn relative position from the winning candidate's
        own delta-v and the CW block, then computes Pc through the same helper
        the pre-maneuver path uses.

        Deliberately reuses the scorer's delta-v and CW propagation. This test
        is scoped to the Pc computation given a post-burn geometry, not to
        whether the post-burn geometry is right. The CW propagation has its own
        defect, recorded on SCRUM-387, and pinning it here would freeze it.
        """
        result = _score()
        winner = next(
            c for c in result.candidates_v25 if c.direction == result.direction
        )

        dt_s = 600.0  # T_BURN_SHORT to T_CA
        phi_rv = cw_phi_rv(_A_KM, dt_s)
        r_post = R_REL - phi_rv @ np.asarray(winner.dv_eci_km_s, dtype=float)

        hbr_m, _ = resolve_hard_body_radius(_cap())
        expected = compute_pc_from_geometry(
            R_SAT, V_SAT, r_post, V_REL_HEAD_ON, P_SURROGATE, hbr_m,
        )
        assert result.pc_post == pytest.approx(expected, rel=1e-12)

    def test_it_uses_the_same_hard_body_radius_as_the_pre_maneuver_pc(self):
        """A different radius on either side would make the before and after
        numbers incomparable, and Pc scales as the square of it."""
        result = _score()
        assert result.hbr_m == conventions.DEFAULT_COMBINED_HBR_M

        hbr_m, _ = resolve_hard_body_radius(_cap())
        direct = compute_pc_post(
            R_SAT, V_SAT, R_REL, V_REL_HEAD_ON, P_SURROGATE, hbr_m,
        )
        assert direct == pytest.approx(result.pc_pre, rel=1e-12), (
            "compute_pc_post on the UNMOVED geometry must reproduce the "
            "pre-maneuver Pc; if it does not, the two ends use different inputs"
        )

    def test_a_bigger_burn_gives_a_smaller_pc(self):
        small = _score(dv_ceiling=0.005)
        large = _score(dv_ceiling=0.02)
        assert large.pc_post < small.pc_post


# ---------------------------------------------------------------------------
# AC5: every candidate carries its own, for SCRUM-387
# ---------------------------------------------------------------------------

class TestEveryCandidateCarriesItsOwnPc:
    def test_all_candidates_have_a_pc(self):
        result = _score()
        assert len(result.candidates_v25) == 6
        for c in result.candidates_v25:
            assert c.pc_post is not None, f"{c.direction} has no pc_post"
            assert 0.0 <= c.pc_post <= 1.0

    def test_the_ordering_by_pc_matches_the_ordering_by_separation(self):
        """Independent cross-check. Pc and Mahalanobis are different quantities
        computed from different code, but at a fixed covariance and radius they
        have to rank the candidates the same way. A covariance mixed up between
        the two would break this even where both numbers still look sane."""
        result = _score()
        by_pc = [c.direction for c in sorted(result.candidates_v25, key=lambda c: c.pc_post)]
        by_m2 = [c.direction for c in sorted(result.candidates_v25, key=lambda c: -c.m2_post)]
        assert by_pc == by_m2

    def test_the_recommended_candidate_is_the_lowest_pc_one(self):
        """True today because the utility maximises the Mahalanobis gain and the
        two orderings agree. SCRUM-387 changes the utility to trade in Pc, so if
        this ever stops holding it is a deliberate decision that ticket has to
        record rather than a silent drift."""
        result = _score()
        best_by_pc = min(result.candidates_v25, key=lambda c: c.pc_post)
        assert result.direction == best_by_pc.direction


# ---------------------------------------------------------------------------
# Execution error reaches the post-maneuver Pc
# ---------------------------------------------------------------------------

class TestExecutionErrorIsIncluded:
    """SCRUM-365 adds Q_exec to the covariance before m2_post. pc_post uses the
    same covariance, so an imperfectly executed burn must not be scored as a
    perfect one.

    Note on the direction of the effect. Adding covariance does NOT always raise
    Pc, and asserting that it does would be wrong. Pc is not monotonic in
    covariance:

        Pc ~ (HBR^2 / 2 sigma_x sigma_z) * exp(-m^2 / 2)

    Widening the covariance shrinks the prefactor and raises the exponential.
    Measured on this geometry the crossover sits between m^2 of 1.44 and 4.00,
    so inside roughly 1.4 sigma more uncertainty LOWERS Pc and outside it raises
    Pc. That is the probability-dilution effect the codebase already names in
    _covariance_quality, where m^2 < 1 is flagged as dilution_region following
    Hejduk and NASA CARA.

    So these tests assert Q_exec reaches the answer, and assert the sign
    separately in each regime, rather than assuming one sign everywhere.
    """

    def test_execution_error_changes_the_post_maneuver_pc(self):
        perfect = _score(cap=_cap())
        sloppy = _score(cap=_cap(thrust_misalignment_deg=2.0, dv_magnitude_sigma=0.05))

        assert perfect.execution_error_modelled is False
        assert sloppy.execution_error_modelled is True
        assert sloppy.pc_post != pytest.approx(perfect.pc_post, rel=1e-12), (
            "pc_post ignores Q_exec, so it reports a perfect-burn probability "
            "for a burn that is not perfect"
        )

    @pytest.mark.parametrize(
        "miss_km, expect_higher",
        [
            (0.1, False),   # m^2 = 0.04, deep inside: dilution lowers Pc
            (0.6, False),   # m^2 = 1.44, still inside the crossover
            (1.0, True),    # m^2 = 4.00, outside: extra spread reaches the target
            (2.0, True),    # m^2 = 16.0
        ],
    )
    def test_added_covariance_moves_pc_the_way_the_geometry_says(
        self, miss_km, expect_higher
    ):
        """Pins the non-monotonicity so nobody later "fixes" it into a
        one-directional assumption. The crossover is a property of the Pc
        integral, not of this fixture."""
        hbr_m, _ = resolve_hard_body_radius(_cap())
        r_rel = np.array([0.0, 0.0, miss_km])
        added = np.diag([0.05 ** 2, 0.05 ** 2, 0.05 ** 2])

        base = compute_pc_post(R_SAT, V_SAT, r_rel, V_REL_HEAD_ON, P_SURROGATE, hbr_m)
        wider = compute_pc_post(
            R_SAT, V_SAT, r_rel, V_REL_HEAD_ON, P_SURROGATE + added, hbr_m
        )

        if expect_higher:
            assert wider > base
        else:
            assert wider < base


# ---------------------------------------------------------------------------
# The approximation in compute_pc_post is bounded, and the bound is measured
# ---------------------------------------------------------------------------

def _cw_phi_vv(a_km: float, dt_s: float) -> np.ndarray:
    """CW velocity-to-velocity block, written here rather than imported.

    Independent reference implementation. The production code deliberately does
    not carry one, so this exists to measure what leaving it out costs rather
    than to trust the docstring that says it is small.

        Phi_vv = [[ cos(nt),     2 sin(nt),      0       ],
                  [ -2 sin(nt),  4 cos(nt) - 3,  0       ],
                  [ 0,           0,              cos(nt) ]]
    """
    n = math.sqrt(MU_EARTH / (a_km ** 3))
    c = math.cos(n * dt_s)
    s = math.sin(n * dt_s)
    return np.array([
        [c, 2.0 * s, 0.0],
        [-2.0 * s, 4.0 * c - 3.0, 0.0],
        [0.0, 0.0, c],
    ], dtype=float)


class TestHoldingRelativeVelocityFixed:
    """compute_pc_post neglects the burn's effect on the relative velocity, and
    so on the orientation of the encounter plane. Its docstring bounds that at
    about 7.3 times the burn magnitude, from the entries of Phi_vv. These tests
    measure it rather than accept it."""

    @pytest.mark.parametrize("dt_s", [600.0, 3600.0, 4.0 * 3600.0, 72.0 * 3600.0])
    def test_the_phi_vv_norm_bound_holds_across_the_whole_tca_window(self, dt_s):
        """The bound has to be independent of propagation time, otherwise it is
        only true for the lead times we happen to test. The entries are
        trigonometric and do not grow, unlike Phi_rv's -3nt term."""
        phi_vv = _cw_phi_vv(_A_KM, dt_s)
        assert np.linalg.norm(phi_vv, 2) <= 7.3

    def test_holding_v_rel_fixed_is_below_the_stated_bound(self):
        """Computes Pc both ways at the policy delta-v ceiling and asserts the
        answers agree to well inside anything a decision turns on."""
        result = _score()
        winner = next(
            c for c in result.candidates_v25 if c.direction == result.direction
        )
        dv = np.asarray(winner.dv_eci_km_s, dtype=float)
        dt_s = 600.0

        r_post = R_REL - cw_phi_rv(_A_KM, dt_s) @ dv
        v_rel_post = V_REL_HEAD_ON - _cw_phi_vv(_A_KM, dt_s) @ dv

        hbr_m, _ = resolve_hard_body_radius(_cap())
        held_fixed = compute_pc_post(
            R_SAT, V_SAT, r_post, V_REL_HEAD_ON, P_SURROGATE, hbr_m,
        )
        propagated = compute_pc_post(
            R_SAT, V_SAT, r_post, v_rel_post, P_SURROGATE, hbr_m,
        )

        assert held_fixed == pytest.approx(result.pc_post, rel=1e-12)
        assert held_fixed == pytest.approx(propagated, rel=1.0e-6), (
            "the encounter-plane rotation neglected by compute_pc_post has "
            "grown large enough to matter; if the delta-v ceiling has risen, "
            "Phi_vv belongs in the production path"
        )


# ---------------------------------------------------------------------------
# The fallbacks, and what they say about themselves
# ---------------------------------------------------------------------------

class TestTheSourceIsAlwaysRecorded:
    def test_no_relative_velocity_falls_back_to_the_inverse_area_and_says_so(self):
        """No relative velocity means no encounter plane, so there is no Pc at
        either end. 1/m2_post is the only thing left, and it is an inverse area
        rather than a probability, which is exactly why the source travels with
        it."""
        result = _score(v_rel=None)
        assert result.no_go_reason_code == ""
        assert result.pc_post is None
        assert result.risk_surrogate_source == RISK_SURROGATE_INVERSE_M2
        assert result.risk_surrogate_post == pytest.approx(1.0 / result.m2_post)

    def test_a_no_go_says_post_equals_pre_because_there_was_no_burn(self):
        """On a no-go the post-maneuver risk genuinely is the pre-maneuver risk.
        The rule is that a pre-maneuver value is never carried under a
        post-maneuver name SILENTLY, not that the two can never be equal."""
        result = _score(r_rel=np.array([0.0, 0.0, 1.2]), t_burn=T_BURN_LONG)
        assert result.no_go_reason_code == "pc_below_threshold"
        assert result.pc_post is None
        assert result.risk_surrogate_source == RISK_SURROGATE_PC_PRE_NO_BURN
        assert result.risk_surrogate_post == pytest.approx(result.pc_pre)

    @pytest.mark.parametrize("supplied", [None, 2.5e-4])
    def test_a_source_is_recorded_on_every_path(self, supplied):
        result = _score(pc_precomputed=supplied)
        assert result.risk_surrogate_source in (
            RISK_SURROGATE_PC_POST,
            RISK_SURROGATE_INVERSE_M2,
            RISK_SURROGATE_PC_PRE_NO_BURN,
        )
        assert result.risk_surrogate_source != ""


# ---------------------------------------------------------------------------
# AC4: an inverse area can no longer reach the audit trail
# ---------------------------------------------------------------------------

class TestTheAuditWriteSendsThePreManeuverPc:
    """server._post_planner_output writes pc_computed into the planner_output
    table and into RiskInputs. It used to read risk_surrogate_post, so the
    tamper-evident SCRUM-377 trail received either a pre-maneuver Pc under a
    name that does not say pre-maneuver, or an inverse square kilometre.

    pc_computed sits alongside m2_pre, covariance_source and data_age_s, all of
    which are pre-maneuver inputs, so the field wants pc_pre. That stays true
    now that risk_surrogate_post carries a genuine post-maneuver Pc.
    """

    @staticmethod
    def _captured_payload(monkeypatch, scoring):
        import server

        sent = {}

        class _Resp:
            status_code = 201

            @staticmethod
            def json():
                return {"id": 1}

        def _fake_post(url, json=None, timeout=None):
            sent.update(json or {})
            return _Resp()

        monkeypatch.setattr(server.http_requests, "post", _fake_post)
        server._post_planner_output(
            cdm_record_id=1,
            result=scoring.to_v24_response(),
            body={"policy": {}},
            covariance_source="test",
            scoring=scoring,
        )
        return sent

    def test_pc_computed_is_the_pre_maneuver_pc(self, monkeypatch):
        scoring = _score()
        payload = self._captured_payload(monkeypatch, scoring)
        assert payload["pc_computed"] == pytest.approx(scoring.pc_pre)
        assert payload["pc_computed"] != pytest.approx(
            scoring.risk_surrogate_post, rel=1e-9
        )

    def test_an_inverse_area_can_no_longer_reach_pc_computed(self, monkeypatch):
        """The event with no relative velocity is the one that used to send an
        inverse square kilometre. It now sends 0.0, meaning no Pc was
        established, which is wrong only in that the column cannot hold null.
        That is a schema question, recorded on SCRUM-396."""
        scoring = _score(v_rel=None)
        assert scoring.risk_surrogate_source == RISK_SURROGATE_INVERSE_M2

        payload = self._captured_payload(monkeypatch, scoring)
        assert payload["pc_computed"] == 0.0
        assert payload["pc_computed"] != pytest.approx(
            scoring.risk_surrogate_post, rel=1e-9
        )

    def test_a_supplied_pc_reaches_the_audit_record_unchanged(self, monkeypatch):
        scoring = _score(pc_precomputed=2.5e-4)
        payload = self._captured_payload(monkeypatch, scoring)
        assert payload["pc_computed"] == pytest.approx(2.5e-4)
