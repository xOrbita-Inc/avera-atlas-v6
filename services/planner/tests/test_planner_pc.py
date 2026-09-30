"""SCRUM-389: the planner computes its own Pc, and the two risk gates are reconciled.

What was wrong
--------------
The operator policy expresses risk in Pc. Both thresholds are described in
operator_policy.py as physics-based gates. But the Pc gate only ran when an
external CDM or UDL supplied a Pc, and the Mahalanobis pre-screen ran always and
first. So for every event without an external Pc, a filter documented as a
pre-screen was silently the maneuver decision, and the two could disagree.

The planner could not compute its own Pc for two reasons. compute_pc lived in
the propagator and was unreachable, fixed in SCRUM-388. And the conjunction
contract carried no relative velocity, so there was no encounter plane to
integrate over, fixed here by making v_rel_km_s an optional input.

What this file guards
---------------------
That a Pc is computed when it can be, that a supplied Pc still wins, that the
source of both the Pc and the hard-body radius is always recorded, that the
right gate decides, and that the hard-body radius cannot diverge from the
propagator's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aps_math import conventions
from aps_math.pc_utils import compute_pc

from common.maneuver_scorer import (
    PC_SOURCE_COMPUTED,
    PC_SOURCE_SUPPLIED,
    PC_SOURCE_UNAVAILABLE,
    _passes_feasibility,
    compute_pc_from_geometry,
    resolve_hard_body_radius,
    resolve_pc,
    score_maneuver_candidates,
)
from common.atlas_artifact import DecisionLog, build_atlas_artifact
from common.evidence_record import GENESIS_HASH, FieldState, build_decision_record
from common.operator_policy import OperatorPolicy
from server import _evidence_values
from common.satellite_capability import (
    LifetimeProfile,
    PropulsionProfile,
    SatelliteCapability,
)

MU_EARTH = 398600.4418
_A_KM = 6378.137 + 550.0
_V_CIRC = math.sqrt(MU_EARTH / _A_KM)

R_SAT = np.array([_A_KM, 0.0, 0.0])
V_SAT = np.array([0.0, _V_CIRC, 0.0])
V_REL_HEAD_ON = np.array([0.0, -2.0 * _V_CIRC, 0.0])

T_BURN = "2026-03-02T11:30:00Z"
T_CA = "2026-03-02T15:30:00Z"

# SCRUM-369 documented surrogate, RTN 1-sigma 0.3 / 2.5 / 0.5 km.
P_SURROGATE = np.diag([0.3 ** 2, 2.5 ** 2, 0.5 ** 2])


def _cap(radius_m: float | None = None) -> SatelliteCapability:
    kwargs = {} if radius_m is None else {"radius_m": radius_m}
    return SatelliteCapability(
        sat_id="TEST-6U",
        a_ref_km=_A_KM,
        propulsion=PropulsionProfile(min_dv_m_s=0.01),
        lifetime=LifetimeProfile(
            mass_kg=12.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
            mission_lifetime_days_remaining=365.0,
        ),
        **kwargs,
    )


def _policy(**kw) -> OperatorPolicy:
    base = dict(
        operator_id="TEST", policy_version="2.5.0",
        decision_mode="flight_rule_1e4",
        pc_maneuver_threshold=1.0e-4, pc_monitor_threshold=1.0e-5,
        mahalanobis_screen_threshold=4.0,
    )
    base.update(kw)
    return OperatorPolicy(**base)


# ---------------------------------------------------------------------------
# AC6: the computed Pc is anchored, not merely plausible
# ---------------------------------------------------------------------------

class TestComputedPcIsCorrect:
    """Anchored the same way SCRUM-388 anchored the library: against a case with
    an exact closed-form answer, not against a value this code produced."""

    @pytest.mark.parametrize("sigma_km,hbr_m", [
        (0.05, 5.0), (0.05, 10.0), (0.1, 10.0), (0.25, 15.0), (1.0, 20.0),
    ])
    def test_matches_the_rayleigh_closed_form(self, sigma_km, hbr_m):
        """Zero miss, isotropic covariance. The encounter-plane projection is an
        isotropic 2D Gaussian on the origin, so Pc is exactly 1 - exp(-R^2/2s^2).
        """
        p_rel = np.eye(3) * sigma_km ** 2
        r_rel = np.array([0.0, 0.0, 1.0e-9])  # numerically zero miss
        pc = compute_pc_from_geometry(R_SAT, V_SAT, r_rel, V_REL_HEAD_ON, p_rel, hbr_m)
        sigma_m = sigma_km * 1000.0
        exact = 1.0 - math.exp(-(hbr_m ** 2) / (2.0 * sigma_m ** 2))
        assert pc == pytest.approx(exact, rel=1e-6)

    @pytest.mark.parametrize("miss_km", [0.001, 0.3, 1.0, 3.0, 8.0])
    def test_the_split_matches_a_hand_split_at_every_miss(self, miss_km):
        """Checked across misses, not at one geometry, and that matters.

        The first version of compute_pc_from_geometry passed the whole combined
        covariance as the primary's and zero as the secondary's. compute_pc
        detects a zero covariance and silently routes to frisbee_max_pc, an
        upper bound rather than the Pc. The bound agrees to three figures at a
        300 m miss and is four orders of magnitude high by four sigma, so a
        single small-miss assertion passed and the defect shipped as far as this
        test file before anything caught it.
        """
        r_rel = np.array([0.0, 0.0, miss_km])
        r1 = R_SAT * 1000.0
        v1 = V_SAT * 1000.0
        r2 = r1 + r_rel * 1000.0
        v2 = v1 + V_REL_HEAD_ON * 1000.0
        cov = P_SURROGATE * 1.0e6
        split = compute_pc(r1, v1, cov * 0.5, r2, v2, cov * 0.5, hbr=15.0).Pc
        assert compute_pc_from_geometry(
            R_SAT, V_SAT, r_rel, V_REL_HEAD_ON, P_SURROGATE, 15.0
        ) == pytest.approx(split, rel=1e-12)

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_pc_falls_off_as_a_gaussian_not_as_a_bound(self, k):
        """Names the wrong answer so a revert fails loudly.

        A real Pc falls as exp(-k^2/2) in Mahalanobis distance. The Frisbee
        upper bound falls as roughly 1/k. At k=4 that is the difference between
        9.4e-09 and 4.3e-06.
        """
        sigma_km = 2.0
        p_rel = np.eye(3) * sigma_km ** 2
        at_zero = compute_pc_from_geometry(
            R_SAT, V_SAT, np.array([0.0, 0.0, 1e-9]), V_REL_HEAD_ON, p_rel, 15.0)
        at_k = compute_pc_from_geometry(
            R_SAT, V_SAT, np.array([0.0, 0.0, k * sigma_km]), V_REL_HEAD_ON,
            p_rel, 15.0)
        assert at_k / at_zero == pytest.approx(math.exp(-(k ** 2) / 2.0), rel=2e-3)

    def test_pc_falls_as_the_miss_grows(self):
        pcs = [
            compute_pc_from_geometry(R_SAT, V_SAT, np.array([0.0, 0.0, m]),
                                     V_REL_HEAD_ON, P_SURROGATE, 15.0)
            for m in (0.0001, 0.3, 1.0, 3.0)
        ]
        assert pcs == sorted(pcs, reverse=True)


# ---------------------------------------------------------------------------
# AC1 and AC3: provenance, and precedence
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_supplied_pc_wins_over_a_computable_one(self):
        """AC3. A CDM or UDL Pc comes from an authority with more information
        than we have. We compute one only when nobody has."""
        supplied = 7.7e-6
        pc, source, _, _ = resolve_pc(
            _cap(), R_SAT, V_SAT, np.array([0.0, 0.0, 0.3]), V_REL_HEAD_ON,
            P_SURROGATE, supplied,
        )
        assert source == PC_SOURCE_SUPPLIED
        assert pc == supplied

        computed, csource, _, _ = resolve_pc(
            _cap(), R_SAT, V_SAT, np.array([0.0, 0.0, 0.3]), V_REL_HEAD_ON,
            P_SURROGATE, None,
        )
        assert csource == PC_SOURCE_COMPUTED
        assert computed != pytest.approx(supplied, rel=1e-3), (
            "the two paths should give different numbers here, or this test "
            "cannot tell precedence from coincidence"
        )

    def test_no_relative_velocity_means_no_pc_not_a_zero_pc(self):
        """Reporting zero would claim the event is safe. Unavailable says we do
        not know, which is the true statement."""
        pc, source, _, _ = resolve_pc(
            _cap(), R_SAT, V_SAT, np.array([0.0, 0.0, 0.3]), None,
            P_SURROGATE, None,
        )
        assert pc is None
        assert source == PC_SOURCE_UNAVAILABLE

    def test_degenerate_geometry_yields_unavailable_not_an_exception(self):
        """Zero relative velocity has no encounter plane. The caller gets no Pc
        rather than a traceback or an invented number."""
        pc, source, _, _ = resolve_pc(
            _cap(), R_SAT, V_SAT, np.array([0.0, 0.0, 0.3]),
            np.zeros(3), P_SURROGATE, None,
        )
        assert pc is None
        assert source == PC_SOURCE_UNAVAILABLE

    @pytest.mark.parametrize("supplied", [None, 5.0e-4])
    def test_hbr_source_is_always_recorded(self, supplied):
        """AC1, no silent default. Whatever route the Pc took, the hard-body
        radius says where it came from."""
        _, _, hbr, hbr_source = resolve_pc(
            _cap(), R_SAT, V_SAT, np.array([0.0, 0.0, 0.3]), V_REL_HEAD_ON,
            P_SURROGATE, supplied,
        )
        assert hbr > 0.0
        assert hbr_source in {"screening_convention", "physical_sum"}


# ---------------------------------------------------------------------------
# AC1: the hard-body radius rule
# ---------------------------------------------------------------------------

class TestHardBodyRadius:
    def test_a_6u_gets_the_screening_convention(self):
        """0.60 m deployed plus a 2 m secondary is 2.6 m, well under the 15 m
        floor, so the convention decides. This is the case we actually fly."""
        hbr, source = resolve_hard_body_radius(_cap())
        assert hbr == conventions.DEFAULT_COMBINED_HBR_M
        assert source == "screening_convention"

    def test_a_large_spacecraft_exceeds_the_floor(self):
        """The convention is a floor, not a replacement for physics. A big pair
        must not be screened as though it were small."""
        hbr, source = resolve_hard_body_radius(_cap(radius_m=40.0))
        assert source == "physical_sum"
        assert hbr == pytest.approx(40.0 + conventions.DEFAULT_SECONDARY_RADIUS_M)

    def test_the_planner_uses_the_shared_convention(self):
        """Half of the anti-divergence property. The other half is asserted in
        the propagator suite, which pins its HBR_M to the same constant. Two
        services, one definition, neither free to drift."""
        hbr, _ = resolve_hard_body_radius(_cap())
        assert hbr is conventions.DEFAULT_COMBINED_HBR_M or hbr == conventions.DEFAULT_COMBINED_HBR_M

    def test_radius_is_a_property_of_the_spacecraft_not_the_policy(self):
        """ADR-010. A larger spacecraft changes the answer; an operator policy
        has no way to express a hard-body radius at all, deliberately."""
        small, _ = resolve_hard_body_radius(_cap(radius_m=0.22))
        large, _ = resolve_hard_body_radius(_cap(radius_m=60.0))
        assert large > small
        assert not hasattr(_policy(), "hbr_m")
        assert not hasattr(_policy(), "hard_body_radius_m")


# ---------------------------------------------------------------------------
# AC4: which gate decides
# ---------------------------------------------------------------------------

class TestRiskGate:
    def test_pc_decides_when_a_pc_exists(self):
        passes, code, _, gate = _passes_feasibility(
            _cap(), _policy(), 2.0, 5.0e-4, PC_SOURCE_COMPUTED, 0.5, 2.0
        )
        assert gate == "pc" and passes is True

    def test_mahalanobis_decides_only_when_no_pc_exists(self):
        passes, code, _, gate = _passes_feasibility(
            _cap(), _policy(), 2.0, None, PC_SOURCE_UNAVAILABLE, None, 10.0
        )
        assert gate == "mahalanobis"
        assert passes is False and code == "trivial_event"

    def test_a_high_pc_is_not_discarded_by_a_wide_mahalanobis(self):
        """The defect this reordering fixes. Before SCRUM-389 the Mahalanobis
        screen ran first and rejected outright, so an event with a maneuver-level
        Pc could be thrown away as 'trivial' without its Pc ever being read."""
        passes, code, _, gate = _passes_feasibility(
            _cap(), _policy(), 2.0, 9.9e-3, PC_SOURCE_COMPUTED, 0.1, 50.0
        )
        assert passes is True, (
            "an event with Pc 9.9e-3 was discarded because its Mahalanobis "
            "distance was wide"
        )
        assert gate == "pc"

    @pytest.mark.parametrize("sigma_m,expected_max_pc", [
        (2000.0, 1e-8), (500.0, 2e-7), (250.0, 7e-7), (100.0, 4e-6),
    ])
    def test_the_mahalanobis_fallback_is_conservative_at_our_covariances(
        self, sigma_m, expected_max_pc
    ):
        """The screen rejects at MD 4.0. At every covariance this system
        produces, the Pc it is throwing away is far below pc_monitor_threshold
        of 1e-5, so the fallback discards nothing that matters.

        It stops being conservative near a 65 m encounter-plane sigma, which is
        tighter than anything we produce today but reachable for a very
        well-tracked object close to TCA. The next test pins that boundary so it
        cannot move silently.
        """
        p_rel = np.eye(3) * (sigma_m / 1000.0) ** 2
        r_rel = np.array([4.0 * sigma_m / 1000.0, 0.0, 0.0])  # exactly MD 4
        pc = compute_pc_from_geometry(R_SAT, V_SAT, r_rel, V_REL_HEAD_ON,
                                      p_rel, conventions.DEFAULT_COMBINED_HBR_M)
        assert pc < expected_max_pc
        assert pc < 1e-5, "the Mahalanobis fallback would discard a monitor-level event"

    def test_where_the_mahalanobis_fallback_stops_being_conservative(self):
        """Named so it is a known boundary rather than a surprise. At a 50 m
        isotropic covariance and a 15 m hard-body radius, Pc at MD 4.0 is above
        the monitor threshold, so the fallback would discard a real event.

        Nothing produces a covariance that tight today. If something starts to,
        this test still passes but the one above will fail, which is the alarm.
        """
        p_rel = np.eye(3) * (50.0 / 1000.0) ** 2
        r_rel = np.array([4.0 * 50.0 / 1000.0, 0.0, 0.0])
        pc = compute_pc_from_geometry(R_SAT, V_SAT, r_rel, V_REL_HEAD_ON,
                                      p_rel, conventions.DEFAULT_COMBINED_HBR_M)
        assert pc > 1e-5


# ---------------------------------------------------------------------------
# Backward compatibility, and AC5
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_omitting_relative_velocity_behaves_exactly_as_before(self):
        """A caller that predates SCRUM-389 sends no v_rel_km_s. It must get the
        Mahalanobis gate and the same outcome it always got."""
        r_rel = np.array([0.0, 0.0, 0.3])
        result = score_maneuver_candidates(
            "CID-COMPAT", R_SAT, V_SAT, r_rel, P_SURROGATE, T_BURN, T_CA,
            _cap(), _policy(),
        )
        assert result.pc_source == PC_SOURCE_UNAVAILABLE
        assert result.pc_pre is None
        assert result.risk_gate == "mahalanobis"

    def test_supplying_relative_velocity_produces_a_computed_pc(self):
        r_rel = np.array([0.0, 0.0, 0.3])
        result = score_maneuver_candidates(
            "CID-PC", R_SAT, V_SAT, r_rel, P_SURROGATE, T_BURN, T_CA,
            _cap(), _policy(), v_rel_km_s=V_REL_HEAD_ON,
        )
        assert result.pc_source == PC_SOURCE_COMPUTED
        assert result.pc_pre is not None and result.pc_pre > 0.0
        assert result.risk_gate == "pc"
        assert result.hbr_m == conventions.DEFAULT_COMBINED_HBR_M

    def test_the_v24_path_still_applies_no_screen_deliberately(self):
        """AC5. decision_model.evaluate_conjunction, behind /v1/evaluate/batch,
        screens nothing and is not changed here.

        It is a backward-compatibility contract with its own callers, and adding
        a risk gate to it would change what existing integrations receive. The
        difference is intentional and recorded rather than reconciled. If the
        v2.4 path ever grows a screen, this assertion is where to say so.
        """
        from avoid import decision_model

        source = decision_model.evaluate_conjunction.__doc__ or ""
        assert "_passes_feasibility" not in dir(decision_model), (
            "the v2.4 path has grown a feasibility screen; update AC5 and this test"
        )
        assert "screen" not in source.lower()


# ---------------------------------------------------------------------------
# SCRUM-395: a no-go result has to carry the Pc that produced it
# ---------------------------------------------------------------------------

class TestNoGoResultsCarryTheResolvedPc:
    """SCRUM-389 added pc_pre as a parameter of _build_nogo_result carrying the
    resolved Pc, supplied or computed. A leftover local assignment inside that
    function overwrote it with pc_precomputed, which is None for every computed
    Pc. Two things followed.

    The event that the Pc threshold rejected reported no Pc at all, so the one
    number that explains the decision was missing from the record of it.

    And risk_surrogate_post fell through to the 1/m^2 branch. That branch exists
    only for events with no Pc, and its units are inverse square kilometres. It
    publishes about 1.7e-01 where the Pc branch publishes 4.2e-05, on the field
    the UI reads as risk. Wrong quantity, wrong units, three orders of magnitude
    high, and only on rejected events, which are the majority.

    Geometry below is chosen so the computed Pc lands between the monitor and
    maneuver thresholds. That is the escalate-to-watch case, the one an operator
    is most likely to look at closely.
    """

    R_REL_BELOW_THRESHOLD = np.array([0.0, 0.0, 1.2])

    def _result(self, **kw):
        return score_maneuver_candidates(
            "CID-NOGO", R_SAT, V_SAT, self.R_REL_BELOW_THRESHOLD, P_SURROGATE,
            T_BURN, T_CA, _cap(), _policy(), v_rel_km_s=V_REL_HEAD_ON, **kw
        )

    def test_the_geometry_still_produces_the_no_go_this_class_tests(self):
        """If a later change moves the Pc across a threshold, the assertions
        below would pass vacuously against a go result. Fail here instead."""
        assert self._result().no_go_reason_code == "pc_below_threshold"

    def test_a_rejected_event_reports_the_pc_that_rejected_it(self):
        result = self._result()
        assert result.pc_source == PC_SOURCE_COMPUTED
        assert result.pc_pre is not None, (
            "the Pc that made the decision was dropped from the result"
        )

        hbr_m, _hbr_source = resolve_hard_body_radius(_cap())
        expected = compute_pc_from_geometry(
            R_SAT, V_SAT, self.R_REL_BELOW_THRESHOLD, V_REL_HEAD_ON,
            P_SURROGATE, hbr_m,
        )
        assert result.pc_pre == pytest.approx(expected, rel=1e-12)

    def test_risk_surrogate_post_is_the_pc_and_not_the_inverse_square_branch(self):
        result = self._result()
        assert result.risk_surrogate_post == pytest.approx(result.pc_pre, rel=1e-12)

        inverse_m2 = 1.0 / result.m2_pre
        assert result.risk_surrogate_post != pytest.approx(inverse_m2, rel=1e-6)
        assert result.risk_surrogate_post < 1.0, (
            "risk_surrogate_post is carrying inverse square kilometres, not a "
            "probability"
        )

    def test_the_reported_pc_sits_between_the_two_thresholds(self):
        """Confirms the number is the escalate-to-watch case it claims to be,
        so the human-readable string and the field agree."""
        policy = _policy()
        result = self._result()
        assert policy.pc_monitor_threshold < result.pc_pre < policy.pc_maneuver_threshold
        assert "watch" in result.no_go_human_readable.lower()

    def test_a_supplied_pc_on_a_rejected_event_is_still_reported(self):
        """The path that always worked. Guards it while fixing the other one."""
        result = self._result(pc_precomputed=5.0e-5)
        assert result.pc_source == PC_SOURCE_SUPPLIED
        assert result.pc_pre == pytest.approx(5.0e-5)
        assert result.risk_surrogate_post == pytest.approx(5.0e-5)

    def test_an_event_with_no_pc_still_uses_the_inverse_square_branch(self):
        """The fallback is not removed. An event with no relative velocity has
        no Pc, and risk_surrogate_post has nothing else to carry."""
        r_rel = np.array([0.0, 0.0, 3.0])
        result = score_maneuver_candidates(
            "CID-NOGO-NOPC", R_SAT, V_SAT, r_rel, P_SURROGATE,
            T_BURN, T_CA, _cap(), _policy(),
        )
        assert result.no_go_reason_code == "trivial_event"
        assert result.pc_pre is None
        assert result.risk_surrogate_post == pytest.approx(1.0 / result.m2_pre)


# ---------------------------------------------------------------------------
# SCRUM-396: resolved Pc survives the layers above the scorer
# ---------------------------------------------------------------------------

class TestResolvedPcPropagation:
    def test_computed_pc_survives_artifact_decision_log_and_evidence(self):
        policy = _policy()
        r_rel = np.array([0.0, 0.0, 1.2])
        scoring = score_maneuver_candidates(
            "CID-396", R_SAT, V_SAT, r_rel, P_SURROGATE,
            T_BURN, T_CA, _cap(), policy, v_rel_km_s=V_REL_HEAD_ON,
        )

        assert scoring.pc_source == PC_SOURCE_COMPUTED
        assert scoring.pc_pre is not None

        artifact = build_atlas_artifact(
            scoring, _cap(), policy, T_CA,
            pc_precomputed=None,
            miss_distance_km=1.2,
        )

        # A1 must report the same resolved Pc the scorer actually used.
        assert artifact.risk_summary.pc_pre == pytest.approx(scoring.pc_pre, rel=1e-12)
        assert artifact.risk_summary.pc_source == PC_SOURCE_COMPUTED
        assert artifact.risk_summary.maneuver_required == policy.is_maneuver_required(
            scoring.pc_pre, 1.2
        )
        assert artifact.risk_summary.monitor_only == policy.is_monitor_only(scoring.pc_pre)

        # A5 must not lose the Pc simply because ingest did not supply it.
        assert artifact.no_go is not None
        assert artifact.no_go.pc_at_decision == pytest.approx(scoring.pc_pre, rel=1e-12)

        decision_log = DecisionLog.from_artifact(
            artifact, policy.operator_id, policy.policy_version
        )
        assert decision_log.pc_at_transition == pytest.approx(scoring.pc_pre, rel=1e-12)
        assert decision_log.pc_source == PC_SOURCE_COMPUTED

        values, not_applicable = _evidence_values(
            artifact,
            decision_log,
            {"operator_id": policy.operator_id, "policy_version": policy.policy_version},
        )
        assert values["pc_at_transition"] == pytest.approx(scoring.pc_pre, rel=1e-12)
        assert "pc_at_transition" not in not_applicable
        assert values["inputs_and_provenance"]["pc_source"] == PC_SOURCE_COMPUTED
        assert values["inputs_and_provenance"]["pc_at_decision"] == pytest.approx(
            scoring.pc_pre, rel=1e-12
        )

        # AC4: assert against the actual SCRUM-377 EvidenceRecord, not only
        # the intermediate values passed to its builder.
        record = build_decision_record(
            "SAT-396", 0, GENESIS_HASH, values,
            not_applicable=not_applicable,
        )
        assert record.fields["pc_at_transition"]["value"] == pytest.approx(
            scoring.pc_pre, rel=1e-12
        )
        provenance = record.fields["inputs_and_provenance"]["value"]
        assert provenance["pc_source"] == PC_SOURCE_COMPUTED
        assert provenance["pc_at_decision"] == pytest.approx(
            scoring.pc_pre, rel=1e-12
        )

    def test_computed_maneuver_pc_cannot_contradict_the_artifact(self):
        # AC2: this geometry produces a planner-computed Pc comfortably above
        # the maneuver threshold and a real burn recommendation.
        policy = _policy()
        r_rel = np.array([0.0, 0.0, 0.5])
        scoring = score_maneuver_candidates(
            "CID-396-GO", R_SAT, V_SAT, r_rel, P_SURROGATE,
            T_BURN, T_CA, _cap(), policy, v_rel_km_s=V_REL_HEAD_ON,
        )

        assert scoring.pc_source == PC_SOURCE_COMPUTED
        assert scoring.pc_pre is not None
        assert scoring.pc_pre > policy.pc_maneuver_threshold
        assert scoring.is_maneuver_recommended()

        artifact = build_atlas_artifact(
            scoring, _cap(), policy, T_CA,
            pc_precomputed=None,
            miss_distance_km=0.5,
        )

        assert artifact.is_maneuver_recommended()
        assert artifact.risk_summary.pc_pre == pytest.approx(
            scoring.pc_pre, rel=1e-12
        )
        assert artifact.risk_summary.pc_source == PC_SOURCE_COMPUTED
        assert artifact.risk_summary.maneuver_required is True
        assert artifact.risk_summary.monitor_only is False
        assert "pc_threshold_exceeded" in artifact.rationale.policy_constraints_applied

    def test_only_a_genuinely_unavailable_pc_is_not_applicable(self):
        policy = _policy()
        scoring = score_maneuver_candidates(
            "CID-396-NOPC", R_SAT, V_SAT, np.array([0.0, 0.0, 3.0]),
            P_SURROGATE, T_BURN, T_CA, _cap(), policy,
        )

        assert scoring.pc_source == PC_SOURCE_UNAVAILABLE
        assert scoring.pc_pre is None

        artifact = build_atlas_artifact(
            scoring, _cap(), policy, T_CA,
            pc_precomputed=None,
            miss_distance_km=3.0,
        )
        decision_log = DecisionLog.from_artifact(
            artifact, policy.operator_id, policy.policy_version
        )

        values, not_applicable = _evidence_values(
            artifact,
            decision_log,
            {"operator_id": policy.operator_id, "policy_version": policy.policy_version},
        )

        assert decision_log.pc_at_transition is None
        assert decision_log.pc_source == PC_SOURCE_UNAVAILABLE
        assert "pc_at_transition" in not_applicable
        assert "pc_at_transition" not in values
        assert values["inputs_and_provenance"]["pc_source"] == PC_SOURCE_UNAVAILABLE
        assert values["inputs_and_provenance"]["pc_at_decision"] is None

        record = build_decision_record(
            "SAT-396-NOPC", 0, GENESIS_HASH, values,
            not_applicable=not_applicable,
        )
        assert record.fields["pc_at_transition"]["state"] == FieldState.NOT_APPLICABLE
        provenance = record.fields["inputs_and_provenance"]["value"]
        assert provenance["pc_source"] == PC_SOURCE_UNAVAILABLE
        assert provenance["pc_at_decision"] is None
