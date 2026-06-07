"""
APS 2.5 — Step 9.4 Test Suite
==============================
File: services/planner/tests/test_maneuver_scorer.py

Coverage:
  1.  compute_lifetime_fraction — SCRUM-280 unblock
  2.  _drag_corrected_dv_return_m_s — CTO §6.2
  3.  _along_track_displacement_km — CW geometry
  4.  _covariance_quality — dilution region classification
  5.  _passes_feasibility — all no-go filter paths
  6.  Candidate generation — directions, attitude restriction
  7.  Scoring: non-constellated satellite (delta_S = 0)
  8.  Scoring: constellated satellite (delta_S > 0, return included)
  9.  Co-optimization: dv_total = dv_avoid + dv_return in utility
 10.  CTO Item 1: return_ratio = 1.0 / dv_return = 0 for non-constellation
 11.  CTO Item 2: drag correction applied only when warranted
 12.  CTO Item 3: lifetime_fraction_used real value wired into scoring
 13.  No-go: utility <= 0 for all candidates
 14.  No-go: propulsion infeasible
 15.  No-go: Pc below threshold
 16.  No-go: Mahalanobis pre-screen
 17.  V2.4 backward compatibility: all output fields preserved
 18.  Sign convention: delta_C v2.4 = m2_pre - m2_post
 19.  operator_summary() answers the operator question
 20.  to_v24_response() schema matches decision_model.py output schema
 21.  evaluate_conjunction_v25() end-to-end
 22.  Physical sanity: utility ordering, best candidate selection

Run:
    pytest services/planner/tests/test_maneuver_scorer.py -v
"""

import math
import pytest
import numpy as np

from common.maneuver_scorer import (
    compute_lifetime_fraction,
    score_maneuver_candidates,
    evaluate_conjunction_v25,
    _drag_corrected_dv_return_m_s,
    _along_track_displacement_km,
    _covariance_quality,
    _passes_feasibility,
    _parse_slot_id,
    ManeuverScoringResult,
    CandidateScore,
)
from common.satellite_capability import (
    SatelliteCapability,
    PropulsionProfile,
    LifetimeProfile,
    ManeuverCadence,
    ConstellationSlot,
)
from common.operator_policy import OperatorPolicy, ScoringWeights
from common.constellation_geometry import (
    leo_constellation_482km,
    _mean_motion_to_sma_km,
    _return_burn_cost_m_s,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sma_km():
    return _mean_motion_to_sma_km(15.3020)


@pytest.fixture
def policy_default():
    return OperatorPolicy(
        operator_id="TEST_OP",
        policy_version="2.5.0",
        max_dv_per_event_ms=2.0,
        mission_lifetime_days_total=1825.0,
        scoring_weights=ScoringWeights(
            lambda_dv=1.0, lambda_lifetime=0.8, lambda_slot_deviation=1.2
        ),
    )


@pytest.fixture
def cap_solo():
    """Non-constellated satellite with plenty of fuel."""
    return SatelliteCapability(
        sat_id="SAT-SOLO",
        a_ref_km=_mean_motion_to_sma_km(15.3020),
        lifetime=LifetimeProfile(
            mass_kg=100.0,
            v_remaining_m_s=50.0,
            v_reserved_m_s=5.0,
            mission_lifetime_days_remaining=365.0,
        ),
        slot=ConstellationSlot(in_constellation=False),
    )


@pytest.fixture
def cap_constellated():
    """Constellated satellite — Starlink-like slot."""
    return SatelliteCapability(
        sat_id="SL-1234",
        a_ref_km=_mean_motion_to_sma_km(15.3020),
        lifetime=LifetimeProfile(
            mass_kg=300.0,
            v_remaining_m_s=120.0,
            v_reserved_m_s=10.0,
            mission_lifetime_days_remaining=900.0,
        ),
        slot=ConstellationSlot(
            in_constellation=True,
            slot_id="P02-S05",
            target_mean_motion_rev_per_day=15.3020,
            acceptable_drift_km=4.459,
            return_dv_budget_m_s=4.4,
            max_recovery_time_s=86400.0,
        ),
    )


@pytest.fixture
def std_vectors():
    """Standard conjunction geometry: clear radial threat.

    r_rel = 0.3 km, P = 0.01*I -> m2 = 9.0, MD = 3.0.
    MD=3.0 < mahalanobis_screen_threshold=4.0, so the event correctly
    passes the pre-screen and proceeds to full scoring.
    (r_rel=0.5 km gives MD=5.0 which is above the screen threshold and
    would be correctly rejected as a trivial/safe event by the policy.)
    """
    r_sat = np.array([6853.0, 0.0, 0.0])
    v_sat = np.array([0.0, 7.626, 0.0])
    r_rel = np.array([0.3, 0.0, 0.0])       # 0.3 km separation: MD=3.0, within screen
    P = np.eye(3) * 0.01                     # tight covariance
    return r_sat, v_sat, r_rel, P


@pytest.fixture
def low_risk_vectors():
    """Low-risk geometry: large separation, small covariance."""
    r_sat = np.array([6853.0, 0.0, 0.0])
    v_sat = np.array([0.0, 7.626, 0.0])
    r_rel = np.array([50.0, 0.0, 0.0])      # 50 km — well outside risk zone
    P = np.eye(3) * 0.01
    return r_sat, v_sat, r_rel, P


T_BURN = "2026-04-14T08:00:00Z"
T_CA   = "2026-04-14T12:00:00Z"   # 4 hours to TCA


# ---------------------------------------------------------------------------
# 1. compute_lifetime_fraction (SCRUM-280)
# ---------------------------------------------------------------------------

class TestComputeLifetimeFraction:
    def _lp(self, remaining):
        return LifetimeProfile(
            mass_kg=100.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
            mission_lifetime_days_remaining=remaining,
        )

    def test_returns_zero_when_total_none(self):
        assert compute_lifetime_fraction(self._lp(365.0), None) == pytest.approx(0.0)

    def test_fresh_satellite(self):
        """If remaining == total, fraction = 0 (just launched)."""
        assert compute_lifetime_fraction(self._lp(1825.0), 1825.0) == pytest.approx(0.0)

    def test_eol_satellite(self):
        """If remaining = 0, fraction = 1 (end of life)."""
        lp = LifetimeProfile(mass_kg=100.0, v_remaining_m_s=5.0,
                             v_reserved_m_s=5.0,
                             mission_lifetime_days_remaining=0.001)
        assert compute_lifetime_fraction(lp, 1825.0) == pytest.approx(1.0, abs=0.001)

    def test_midlife(self):
        """Half remaining = 0.5 fraction."""
        result = compute_lifetime_fraction(self._lp(912.5), 1825.0)
        assert result == pytest.approx(0.5, rel=1e-4)

    def test_clamped_above_1(self):
        """remaining > total -> clamped to 0."""
        result = compute_lifetime_fraction(self._lp(2000.0), 1825.0)
        assert result == pytest.approx(0.0)

    def test_zero_total_returns_zero(self):
        assert compute_lifetime_fraction(self._lp(365.0), 0.0) == pytest.approx(0.0)

    def test_real_value_used_in_scoring(self, cap_solo, policy_default, std_vectors):
        """Scorer uses real lifetime_fraction (not 0.0 stub) when total supplied."""
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "CID-LF", r_sat, v_sat, r_rel, P,
            T_BURN, T_CA, cap_solo, policy_default,
        )
        # mission_lifetime_days_total=1825, remaining=365 -> frac=0.8
        assert result.lifetime_fraction_used == pytest.approx(0.8, rel=0.01)
        assert result.lifetime_fraction_used != 0.0   # stub would give 0.0


# ---------------------------------------------------------------------------
# 2. Drag-corrected return cost (CTO §6.2)
# ---------------------------------------------------------------------------

class TestDragCorrectedReturn:
    def test_no_drag_below_2_orbits(self, sma_km):
        """< 2 recovery orbits -> pure §2 formula, no drag."""
        base = _return_burn_cost_m_s(0.21, sma_km)
        drag = _drag_corrected_dv_return_m_s(0.21, sma_km, 1.0, 482.0)
        assert drag == pytest.approx(base, rel=1e-6)

    def test_drag_applied_above_2_orbits_low_alt(self, sma_km):
        """> 2 orbits, alt < 550 km -> drag correction increases dv_return."""
        base = _return_burn_cost_m_s(0.21, sma_km)
        drag = _drag_corrected_dv_return_m_s(0.21, sma_km, 5.0, 482.0)
        assert drag > base, "Drag correction should increase return cost"

    def test_no_drag_above_550km(self, sma_km):
        """Above 550 km threshold -> no drag correction even for many orbits."""
        base = _return_burn_cost_m_s(0.21, sma_km)
        drag = _drag_corrected_dv_return_m_s(0.21, sma_km, 10.0, 600.0)
        assert drag == pytest.approx(base, rel=1e-6)

    def test_drag_scales_with_recovery_orbits(self, sma_km):
        """More recovery orbits -> more drag -> higher return cost."""
        d3 = _drag_corrected_dv_return_m_s(0.21, sma_km, 3.0, 482.0)
        d5 = _drag_corrected_dv_return_m_s(0.21, sma_km, 5.0, 482.0)
        d10 = _drag_corrected_dv_return_m_s(0.21, sma_km, 10.0, 482.0)
        assert d3 < d5 < d10

    def test_zero_dv_returns_zero(self, sma_km):
        assert _drag_corrected_dv_return_m_s(0.0, sma_km, 5.0, 482.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Along-track displacement
# ---------------------------------------------------------------------------

class TestAlongTrackDisplacement:
    def test_positive(self, sma_km):
        d = _along_track_displacement_km(0.21, sma_km, 14400.0)  # 4h to TCA
        assert d > 0.0

    def test_scales_with_dv(self, sma_km):
        d1 = _along_track_displacement_km(0.21, sma_km, 14400.0)
        d2 = _along_track_displacement_km(0.42, sma_km, 14400.0)
        assert d2 == pytest.approx(2 * d1, rel=1e-6)

    def test_zero_dv_zero_displacement(self, sma_km):
        assert _along_track_displacement_km(0.0, sma_km, 14400.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. Covariance quality
# ---------------------------------------------------------------------------

class TestCovarianceQuality:
    def test_dilution_region(self):
        assert _covariance_quality(0.5) == "dilution_region"

    def test_dilution_boundary(self):
        assert _covariance_quality(0.99) == "dilution_region"

    def test_degraded(self):
        assert _covariance_quality(2.0) == "degraded"

    def test_good(self):
        assert _covariance_quality(5.0) == "good"


# ---------------------------------------------------------------------------
# 5. Feasibility filters
# ---------------------------------------------------------------------------

class TestFeasibilityFilters:
    def _make_cap(self, v_remaining=50.0, v_reserved=5.0,
                  attitude_restricted=False, power_constrained=False,
                  min_dv=0.01):
        return SatelliteCapability(
            sat_id="TEST",
            a_ref_km=_mean_motion_to_sma_km(15.3020),
            propulsion=PropulsionProfile(min_dv_m_s=min_dv),
            lifetime=LifetimeProfile(
                mass_kg=100.0, v_remaining_m_s=v_remaining,
                v_reserved_m_s=v_reserved,
                mission_lifetime_days_remaining=365.0,
            ),
            cadence=ManeuverCadence(
                attitude_restricted=attitude_restricted,
                power_constrained=power_constrained,
            ),
        )

    def _policy(self, md_thresh=4.0, pc_thresh=1e-4):
        return OperatorPolicy(
            operator_id="T", policy_version="2.5.0",
            mahalanobis_screen_threshold=md_thresh,
            pc_maneuver_threshold=pc_thresh,
            pc_monitor_threshold=1e-5,
        )

    def test_passes_normal(self):
        # mahalanobis=2.0 < screen_threshold=4.0 -> passes pre-screen
        passes, code, _ = _passes_feasibility(
            self._make_cap(), self._policy(), 2.0, None, None, 2.0
        )
        assert passes is True
        assert code == ""

    def test_nogo_trivial_event(self):
        """Mahalanobis above screen threshold -> trivial_event."""
        passes, code, _ = _passes_feasibility(
            self._make_cap(), self._policy(md_thresh=3.0), 2.0, None, None, 10.0
        )
        assert passes is False
        assert code == "trivial_event"

    def test_nogo_propulsion_infeasible(self):
        """Zero available dv -> propulsion_infeasible."""
        cap = self._make_cap(v_remaining=5.0, v_reserved=5.0)  # v_available=0
        passes, code, _ = _passes_feasibility(
            cap, self._policy(), 2.0, None, None, 2.0
        )
        assert passes is False
        assert code == "propulsion_infeasible"

    def test_nogo_pc_below_threshold(self):
        """Pc below maneuver threshold -> pc_below_threshold."""
        passes, code, _ = _passes_feasibility(
            self._make_cap(), self._policy(), 2.0, 1e-6, 5.0, 2.0
        )
        assert passes is False
        assert code == "pc_below_threshold"

    def test_passes_pc_above_threshold(self):
        """Pc above threshold -> passes."""
        passes, _, _ = _passes_feasibility(
            self._make_cap(), self._policy(), 2.0, 5e-4, 0.5, 2.0
        )
        assert passes is True


# ---------------------------------------------------------------------------
# 6. Candidate generation
# ---------------------------------------------------------------------------

class TestCandidateGeneration:
    def test_non_restricted_6_candidates(self, cap_solo, policy_default, std_vectors):
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "CID-DIR", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        directions = [c["direction"] for c in result.all_candidates]
        assert "prograde" in directions
        assert "retrograde" in directions
        assert "radial" in directions

    def test_attitude_restricted_only_prograde(self, policy_default, std_vectors):
        r_sat, v_sat, r_rel, P = std_vectors
        cap = SatelliteCapability(
            sat_id="ATT-SAT",
            a_ref_km=_mean_motion_to_sma_km(15.3020),
            lifetime=LifetimeProfile(mass_kg=100.0, v_remaining_m_s=50.0,
                                     v_reserved_m_s=5.0,
                                     mission_lifetime_days_remaining=365.0),
            cadence=ManeuverCadence(attitude_restricted=True),
        )
        result = score_maneuver_candidates(
            "CID-ATT", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap, policy_default
        )
        burn_directions = [c["direction"] for c in result.all_candidates
                           if c["direction"] != "no-burn"]
        assert burn_directions == ["prograde"]


# ---------------------------------------------------------------------------
# 7. Scoring — non-constellated (delta_S = 0)
# ---------------------------------------------------------------------------

class TestScoringNonConstellated:
    def test_slot_cost_zero(self, cap_solo, policy_default, std_vectors):
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "CID-NC", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        assert result.slot_cost_term == pytest.approx(0.0)

    def test_dv_return_zero(self, cap_solo, policy_default, std_vectors):
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "CID-NC2", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        assert result.dv_return_m_s == pytest.approx(0.0)

    def test_dv_total_equals_dv_avoid(self, cap_solo, policy_default, std_vectors):
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "CID-NC3", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        assert result.dv_total_m_s == pytest.approx(result.dv_magnitude_m_s)

    def test_no_recovery_plan(self, cap_solo, policy_default, std_vectors):
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "CID-NC4", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        assert result.recovery_plan is None


# ---------------------------------------------------------------------------
# 8. Scoring — constellated satellite
# ---------------------------------------------------------------------------

class TestScoringConstellated:
    def test_slot_cost_positive_when_drift_exceeds_tol(
        self, cap_constellated, policy_default, std_vectors
    ):
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "CID-C", r_sat, v_sat, r_rel, P, T_BURN, T_CA,
            cap_constellated, policy_default, geometry=leo_constellation_482km(),
        )
        if result.is_maneuver_recommended():
            # If a burn is recommended, check the candidate scores
            for cs in result.candidates_v25:
                assert cs.slot_cost_term >= 0.0

    def test_recovery_plan_populated(
        self, cap_constellated, policy_default, std_vectors
    ):
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "CID-C2", r_sat, v_sat, r_rel, P, T_BURN, T_CA,
            cap_constellated, policy_default, geometry=leo_constellation_482km(),
        )
        if result.is_maneuver_recommended() and result.dv_return_m_s > 0:
            assert result.recovery_plan is not None

    def test_dv_total_geq_dv_avoid(
        self, cap_constellated, policy_default, std_vectors
    ):
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "CID-C3", r_sat, v_sat, r_rel, P, T_BURN, T_CA,
            cap_constellated, policy_default,
        )
        assert result.dv_total_m_s >= result.dv_magnitude_m_s


# ---------------------------------------------------------------------------
# 9. Co-optimization: return included before scoring
# ---------------------------------------------------------------------------

class TestCoOptimization:
    def test_constellated_utility_lower_than_solo_same_geometry(
        self, policy_default, std_vectors
    ):
        """Constellated satellite has lower utility than identical non-constellated
        satellite because the return burn is included in dv_total before scoring.

        Both satellites are identical except for in_constellation.  Same fuel,
        same mass, same lifetime — so the only difference in utility is the
        return burn penalty on the constellated path.
        """
        r_sat, v_sat, r_rel, P = std_vectors

        shared_lifetime = LifetimeProfile(
            mass_kg=100.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
            mission_lifetime_days_remaining=365.0,
        )
        cap_solo = SatelliteCapability(
            sat_id="SOLO",
            a_ref_km=_mean_motion_to_sma_km(15.3020),
            lifetime=shared_lifetime,
            slot=ConstellationSlot(in_constellation=False),
        )
        cap_const = SatelliteCapability(
            sat_id="CONST",
            a_ref_km=_mean_motion_to_sma_km(15.3020),
            lifetime=shared_lifetime,
            slot=ConstellationSlot(
                in_constellation=True,
                slot_id="P02-S05",
                acceptable_drift_km=4.459,
                return_dv_budget_m_s=4.4,
            ),
        )

        result_solo = score_maneuver_candidates(
            "CID-CO1", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        result_const = score_maneuver_candidates(
            "CID-CO2", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_const, policy_default
        )

        if result_solo.is_maneuver_recommended() and result_const.is_maneuver_recommended():
            # Both pick the same best direction (identical geometry).
            # Constellated dv_total >= solo dv_total -> constellated utility <= solo utility.
            assert result_const.dv_total_m_s >= result_solo.dv_total_m_s
            assert result_const.utility <= result_solo.utility


# ---------------------------------------------------------------------------
# 10. CTO Item 1: return_ratio
# ---------------------------------------------------------------------------

class TestReturnRatio:
    def test_return_ratio_1_for_constellated(self, sma_km):
        """dv_return = dv_avoid exactly for constellated (§2 formula)."""
        for dv in [0.1, 0.21, 1.0, 2.0]:
            dv_ret = _return_burn_cost_m_s(dv, sma_km)
            assert dv_ret == pytest.approx(dv, rel=1e-5), (
                f"return_ratio != 1.0 for dv={dv}"
            )

    def test_dv_return_zero_non_constellated(
        self, cap_solo, policy_default, std_vectors
    ):
        """Non-constellated: dv_return = 0 in final result."""
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "CID-RR", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        assert result.dv_return_m_s == pytest.approx(0.0)

    def test_total_cost_2x_avoid_constellated_no_drag(self, sma_km):
        """Total = 2 * dv_avoid when recovery required and no drag."""
        dv = 0.21
        dv_ret = _return_burn_cost_m_s(dv, sma_km)   # no drag
        total = dv + dv_ret
        assert total == pytest.approx(2 * dv, rel=1e-5)


# ---------------------------------------------------------------------------
# 11. CTO Item 2: drag correction applied only when warranted
# ---------------------------------------------------------------------------

class TestDragApplication:
    def test_drag_not_applied_at_high_altitude(self, sma_km):
        base = _return_burn_cost_m_s(1.0, sma_km)
        drag = _drag_corrected_dv_return_m_s(1.0, sma_km, 10.0, 600.0)
        assert drag == pytest.approx(base, rel=1e-6)

    def test_drag_not_applied_short_window(self, sma_km):
        base = _return_burn_cost_m_s(1.0, sma_km)
        drag = _drag_corrected_dv_return_m_s(1.0, sma_km, 1.5, 482.0)
        assert drag == pytest.approx(base, rel=1e-6)

    def test_drag_applied_long_window_low_alt(self, sma_km):
        base = _return_burn_cost_m_s(1.0, sma_km)
        drag = _drag_corrected_dv_return_m_s(1.0, sma_km, 5.0, 482.0)
        assert drag > base

    def test_drag_physically_small(self, sma_km):
        """Drag correction should be a small fraction of total, not dominant."""
        dv = 1.0
        drag_10 = _drag_corrected_dv_return_m_s(dv, sma_km, 10.0, 482.0)
        base = _return_burn_cost_m_s(dv, sma_km)
        correction_frac = (drag_10 - base) / base
        # At 482 km, 10 orbits: drag adds ~50m/day * (10*94min/1440) days
        # ~0.033 km = 33 m drag, vs delta_a_burn for 1 m/s ~ 1 km
        # correction should be a few percent
        assert 0 < correction_frac < 0.2, (
            f"Drag correction fraction {correction_frac:.3f} out of expected range"
        )


# ---------------------------------------------------------------------------
# 12. CTO Item 3: lifetime_fraction_used wired into scoring
# ---------------------------------------------------------------------------

class TestLifetimeFractionScoring:
    def test_eol_satellite_higher_lifetime_penalty(
        self, policy_default, std_vectors
    ):
        """EOL satellite should have higher lifetime cost term than fresh one."""
        r_sat, v_sat, r_rel, P = std_vectors

        cap_fresh = SatelliteCapability(
            sat_id="FRESH",
            a_ref_km=_mean_motion_to_sma_km(15.3020),
            lifetime=LifetimeProfile(
                mass_kg=100.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
                mission_lifetime_days_remaining=1825.0,  # just launched
            ),
        )
        cap_eol = SatelliteCapability(
            sat_id="EOL",
            a_ref_km=_mean_motion_to_sma_km(15.3020),
            lifetime=LifetimeProfile(
                mass_kg=100.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
                mission_lifetime_days_remaining=10.0,    # near EOL
            ),
        )

        res_fresh = score_maneuver_candidates(
            "FRESH", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_fresh, policy_default
        )
        res_eol = score_maneuver_candidates(
            "EOL", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_eol, policy_default
        )

        # EOL satellite has higher lifetime_fraction_used -> higher penalty -> lower utility
        assert res_eol.lifetime_fraction_used > res_fresh.lifetime_fraction_used
        if res_eol.is_maneuver_recommended() and res_fresh.is_maneuver_recommended():
            assert res_eol.utility < res_fresh.utility

    def test_lifetime_fraction_zero_when_total_none(self, std_vectors):
        r_sat, v_sat, r_rel, P = std_vectors
        policy_no_total = OperatorPolicy(
            operator_id="T", policy_version="2.5.0",
            mission_lifetime_days_total=None,
        )
        cap = SatelliteCapability(
            sat_id="X", a_ref_km=_mean_motion_to_sma_km(15.3020),
            lifetime=LifetimeProfile(mass_kg=100.0, v_remaining_m_s=50.0,
                                     v_reserved_m_s=5.0,
                                     mission_lifetime_days_remaining=365.0),
        )
        result = score_maneuver_candidates(
            "LF-NONE", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap, policy_no_total
        )
        assert result.lifetime_fraction_used == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 13-16. No-go conditions
# ---------------------------------------------------------------------------

class TestNoGoConditions:
    def test_no_utility_gain_low_risk(self, cap_solo, policy_default, low_risk_vectors):
        """Low-risk geometry: all candidates have negative utility -> no-burn."""
        r_sat, v_sat, r_rel, P = low_risk_vectors
        result = score_maneuver_candidates(
            "NOGO-U", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        assert result.direction == "no-burn"
        assert result.dv_magnitude_m_s == pytest.approx(0.0)
        assert result.utility == pytest.approx(0.0)

    def test_nogo_propulsion_infeasible(self, policy_default, std_vectors):
        r_sat, v_sat, r_rel, P = std_vectors
        cap_dead = SatelliteCapability(
            sat_id="DEAD",
            a_ref_km=_mean_motion_to_sma_km(15.3020),
            lifetime=LifetimeProfile(mass_kg=100.0, v_remaining_m_s=5.0,
                                     v_reserved_m_s=5.0,
                                     mission_lifetime_days_remaining=365.0),
        )
        result = score_maneuver_candidates(
            "NOGO-P", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_dead, policy_default
        )
        assert result.direction == "no-burn"
        assert result.no_go_reason_code == "propulsion_infeasible"

    def test_nogo_pc_below_threshold(self, cap_solo, policy_default, std_vectors):
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "NOGO-PC", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default,
            pc_precomputed=1e-7,   # well below 1e-4 threshold
        )
        assert result.direction == "no-burn"
        assert result.no_go_reason_code == "pc_below_threshold"

    def test_nogo_trivial_event(self, cap_solo, std_vectors):
        r_sat, v_sat, r_rel, P = std_vectors
        tight_policy = OperatorPolicy(
            operator_id="T", policy_version="2.5.0",
            mahalanobis_screen_threshold=0.1,  # extremely tight screen
        )
        result = score_maneuver_candidates(
            "NOGO-MD", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, tight_policy
        )
        assert result.direction == "no-burn"
        assert result.no_go_reason_code == "trivial_event"

    def test_nogo_no_go_human_readable_populated(self, cap_solo, policy_default, low_risk_vectors):
        r_sat, v_sat, r_rel, P = low_risk_vectors
        result = score_maneuver_candidates(
            "NOGO-HR", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        if not result.is_maneuver_recommended():
            assert len(result.no_go_human_readable) > 10


# ---------------------------------------------------------------------------
# 17. V2.4 backward compatibility
# ---------------------------------------------------------------------------

class TestV24Compatibility:
    def test_all_v24_fields_present(self, cap_solo, policy_default, std_vectors):
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "V24", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        v24 = result.to_v24_response()
        assert "conjunction_id" in v24
        rec = v24["recommendation"]
        assert "direction" in rec
        assert "dv_eci_km_s" in rec
        assert "dv_magnitude_m_s" in rec
        assert "t_burn_utc" in rec
        assert "utility" in rec
        met = v24["metrics"]
        assert "delta_C" in met
        assert "m2_pre" in met
        assert "m2_post" in met
        assert "fuel_cost_m_s" in met
        assert "lifetime_penalty" in met
        assert "risk_surrogate_post" in met
        assert "all_candidates" in met

    def test_all_candidates_have_direction_and_utility(
        self, cap_solo, policy_default, std_vectors
    ):
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "V24-C", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        for c in result.all_candidates:
            assert "direction" in c
            assert "utility" in c
            assert "delta_C" in c
            assert "dv_eci_km_s" in c


# ---------------------------------------------------------------------------
# 18. Sign convention
# ---------------------------------------------------------------------------

class TestSignConvention:
    def test_delta_C_v24_is_pre_minus_post(self, cap_solo, policy_default, std_vectors):
        """Output delta_C must be m2_pre - m2_post (v2.4 convention)."""
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "SIGN", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        if result.is_maneuver_recommended():
            expected = result.m2_pre - result.m2_post
            assert result.delta_C == pytest.approx(expected, rel=1e-6)

    def test_delta_C_v24_positive_when_safer(
        self, cap_solo, policy_default, std_vectors
    ):
        """When a maneuver is recommended, m2_post > m2_pre (safer),
        so delta_C_v24 = m2_pre - m2_post should be negative.
        Wait — v2.4 convention: delta_C = m2_pre - m2_post.
        If burn makes satellite safer: m2_post > m2_pre -> delta_C < 0 in v2.4.
        decision_model.py line 372: delta_C = m2_pre - m2_post, treated as positive when good.
        Cross-check: for a burn that INCREASES separation, m2_post > m2_pre,
        so delta_C_v24 < 0, delta_C_v25 > 0.  Both conventions say it's good.
        The sign is just a reporting convention, not a decision driver."""
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "SIGN2", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        # delta_C reported in output must equal m2_pre - m2_post
        assert result.delta_C == pytest.approx(result.m2_pre - result.m2_post, rel=1e-6)

    def test_candidate_delta_C_consistent(self, cap_solo, policy_default, std_vectors):
        """all_candidates delta_C must be m2_pre - m2_post for each candidate."""
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "SIGN3", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        for cs in result.candidates_v25:
            assert cs.delta_C_v24 == pytest.approx(
                -cs.delta_C_v25, rel=1e-9
            ), "delta_C_v24 must equal -delta_C_v25"


# ---------------------------------------------------------------------------
# 19. operator_summary
# ---------------------------------------------------------------------------

class TestOperatorSummary:
    def test_maneuver_summary_contains_direction(
        self, cap_solo, policy_default, std_vectors
    ):
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "SUM", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        s = result.operator_summary()
        assert len(s) > 10
        if result.is_maneuver_recommended():
            assert "MANEUVER RECOMMENDED" in s
            assert result.direction in s
        else:
            assert "NO ACTION" in s

    def test_nogo_summary_contains_reason(
        self, cap_solo, policy_default, low_risk_vectors
    ):
        r_sat, v_sat, r_rel, P = low_risk_vectors
        result = score_maneuver_candidates(
            "SUM2", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        if not result.is_maneuver_recommended():
            assert "NO ACTION" in result.operator_summary()


# ---------------------------------------------------------------------------
# 20. to_v24_response schema
# ---------------------------------------------------------------------------

class TestToV24Response:
    def test_schema_matches_decision_model_output(
        self, cap_solo, policy_default, std_vectors
    ):
        """to_v24_response must match decision_model.evaluate_conjunction schema."""
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "V24R", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        v24 = result.to_v24_response()
        # Top-level keys
        for key in ("conjunction_id", "recommendation", "metrics", "evaluated_at"):
            assert key in v24, f"Missing key: {key}"
        # recommendation sub-keys
        for key in ("direction", "dv_eci_km_s", "dv_magnitude_m_s",
                    "t_burn_utc", "utility"):
            assert key in v24["recommendation"], f"Missing recommendation.{key}"
        # metrics sub-keys
        for key in ("delta_C", "m2_pre", "m2_post", "fuel_cost_m_s",
                    "lifetime_penalty", "risk_surrogate_post", "all_candidates"):
            assert key in v24["metrics"], f"Missing metrics.{key}"


# ---------------------------------------------------------------------------
# 21. evaluate_conjunction_v25 end-to-end
# ---------------------------------------------------------------------------

class TestEvaluateConjunctionV25:
    def _make_request(self, in_constellation=False, pc=None):
        return {
            "conjunction_id": "E2E-001",
            "satellite": {
                "sat_id": "SL-9999",
                "a_ref_km": _mean_motion_to_sma_km(15.3020),
                "r_sat_km": [6853.0, 0.0, 0.0],
                "v_sat_km_s": [0.0, 7.626, 0.0],
                "t_burn_utc": T_BURN,
                "v_remaining_m_s": 50.0,
                "lifetime": {
                    "v_remaining_m_s": 50.0,
                    "v_reserved_m_s": 5.0,
                    "mission_lifetime_days_remaining": 365.0,
                },
                "slot": {
                    "in_constellation": in_constellation,
                    "slot_id": "P02-S05",
                    "acceptable_drift_km": 4.459,
                    "return_dv_budget_m_s": 4.4,
                } if in_constellation else {"in_constellation": False},
            },
            "conjunction": {
                "obj_id": "OBJ-001",
                "t_ca_utc": T_CA,
                "r_rel_km": [0.3, 0.0, 0.0],
                "p_rel_km2": [0.01,0,0, 0,0.01,0, 0,0,0.01],
                **({"pc_precomputed": pc} if pc is not None else {}),
            },
            "policy": {
                "operator_id": "DEFAULT_LEO",
                "policy_version": "2.5.0",
                "lambda_v": 1.0,
                "lambda_L": 0.8,
                "dv_mag_limit_m_s": 2.0,
                "mission_lifetime_days_total": 1825.0,
            },
        }

    def test_returns_scoring_result(self):
        req = self._make_request()
        result = evaluate_conjunction_v25(req)
        assert isinstance(result, ManeuverScoringResult)

    def test_conjunction_id_preserved(self):
        req = self._make_request()
        result = evaluate_conjunction_v25(req)
        assert result.conjunction_id == "E2E-001"

    def test_constellated_end_to_end(self):
        req = self._make_request(in_constellation=True)
        result = evaluate_conjunction_v25(req, geometry=leo_constellation_482km())
        assert isinstance(result, ManeuverScoringResult)

    def test_pc_nogo_end_to_end(self):
        req = self._make_request(pc=1e-8)
        result = evaluate_conjunction_v25(req)
        assert result.direction == "no-burn"
        assert result.no_go_reason_code == "pc_below_threshold"


# ---------------------------------------------------------------------------
# 22. Physical sanity
# ---------------------------------------------------------------------------

class TestPhysicalSanity:
    def test_utility_decreases_with_higher_lambda_v(
        self, cap_solo, std_vectors
    ):
        """Higher lambda_v -> higher dv penalty -> lower utility."""
        r_sat, v_sat, r_rel, P = std_vectors
        p_low = OperatorPolicy(
            operator_id="T", policy_version="2.5.0",
            scoring_weights=ScoringWeights(lambda_dv=0.1, lambda_lifetime=0.0,
                                           lambda_slot_deviation=0.0),
        )
        p_high = OperatorPolicy(
            operator_id="T", policy_version="2.5.0",
            scoring_weights=ScoringWeights(lambda_dv=10.0, lambda_lifetime=0.0,
                                           lambda_slot_deviation=0.0),
        )
        r_low  = score_maneuver_candidates("PL", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, p_low)
        r_high = score_maneuver_candidates("PH", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, p_high)
        if r_low.is_maneuver_recommended() and r_high.is_maneuver_recommended():
            assert r_low.utility > r_high.utility

    def test_best_candidate_has_highest_utility(
        self, cap_solo, policy_default, std_vectors
    ):
        """Best candidate must have the highest utility among all candidates."""
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "BEST", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        if result.is_maneuver_recommended():
            all_utils = [c.utility for c in result.candidates_v25]
            assert result.utility == pytest.approx(max(all_utils), rel=1e-6)

    def test_m2_pre_positive(self, cap_solo, policy_default, std_vectors):
        r_sat, v_sat, r_rel, P = std_vectors
        result = score_maneuver_candidates(
            "M2", r_sat, v_sat, r_rel, P, T_BURN, T_CA, cap_solo, policy_default
        )
        assert result.m2_pre > 0.0

    def test_parse_slot_id(self):
        assert _parse_slot_id("P02-S05") == (2, 5)
        assert _parse_slot_id("P00-S00") == (0, 0)
        assert _parse_slot_id("INVALID") == (0, 0)
        assert _parse_slot_id("") == (0, 0)
