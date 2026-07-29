"""
services/planner/tests/test_execution_error.py

SCRUM-365: tests for execution-error covariance modelling (Q_exec).

Covers AC1 (optional fields, nominal default preserves old behavior),
AC2 (Q_exec added to S before m2_post), AC3 (transparency flag reflects
reality), and AC5's three required cases: perfect burn (no change),
a misaligned burn, and a magnitude-uncertain burn.

Note on AC1: the full existing test_maneuver_scorer.py suite (which
never sets thrust_misalignment_deg or dv_magnitude_sigma on any
fixture) continues to pass unmodified against this change -- that is
itself evidence AC1 holds for every pre-existing scenario, not just
the ones added here.

Run with (from repo root -- conftest.py handles PYTHONPATH):
    python -m pytest services/planner/tests/test_execution_error.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.satellite_capability import (
    SatelliteCapability,
    PropulsionProfile,
    LifetimeProfile,
    ManeuverCadence,
    ConstellationSlot,
)
from common.operator_policy import OperatorPolicy, ScoringWeights
from common.maneuver_scorer import score_maneuver_candidates
from avoid.decision_model import compute_q_exec_km2, cw_phi_rv
from common.constellation_geometry import _mean_motion_to_sma_km


# ---------------------------------------------------------------------------
# PropulsionProfile: new optional fields
# ---------------------------------------------------------------------------

class TestPropulsionProfileExecutionErrorFields:
    def test_thrust_misalignment_defaults_to_none(self):
        p = PropulsionProfile()
        assert p.thrust_misalignment_deg is None

    def test_dv_magnitude_sigma_defaults_to_none(self):
        p = PropulsionProfile()
        assert p.dv_magnitude_sigma is None

    def test_valid_values_accepted(self):
        p = PropulsionProfile(thrust_misalignment_deg=1.0, dv_magnitude_sigma=0.02)
        assert p.thrust_misalignment_deg == 1.0
        assert p.dv_magnitude_sigma == 0.02

    def test_negative_thrust_misalignment_raises(self):
        with pytest.raises(ValueError):
            PropulsionProfile(thrust_misalignment_deg=-0.1)

    def test_dv_magnitude_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            PropulsionProfile(dv_magnitude_sigma=-0.01)

    def test_dv_magnitude_sigma_of_one_raises(self):
        """Upper bound is exclusive -- 100% uncertainty is not a valid fraction."""
        with pytest.raises(ValueError):
            PropulsionProfile(dv_magnitude_sigma=1.0)

    def test_dv_magnitude_sigma_zero_is_valid_boundary(self):
        p = PropulsionProfile(dv_magnitude_sigma=0.0)
        assert p.dv_magnitude_sigma == 0.0

    def test_thrust_misalignment_zero_is_valid_boundary(self):
        p = PropulsionProfile(thrust_misalignment_deg=0.0)
        assert p.thrust_misalignment_deg == 0.0


# ---------------------------------------------------------------------------
# compute_q_exec_km2: the physics primitive itself
# ---------------------------------------------------------------------------

class TestComputeQExecKm2:
    """Real orbit, real burn geometry -- same setup used throughout
    tonight's other physics checks (550 km circular LEO)."""

    MU_EARTH = 398600.4418
    A_KM = 6378.137 + 550.0
    DT_S = 6 * 3600.0
    DIRECTION = np.array([0.0, 1.0, 0.0])
    DV_MAG_KM_S = 0.5 / 1000.0  # 0.5 m/s, typical per operator policy

    @pytest.fixture
    def phi_rv(self):
        return cw_phi_rv(self.A_KM, self.DT_S)

    def test_both_none_returns_zero_matrix(self, phi_rv):
        """AC1: no parameters -> no contribution, exactly."""
        q = compute_q_exec_km2(self.DIRECTION, self.DV_MAG_KM_S, None, None, phi_rv)
        assert np.allclose(q, 0.0)

    def test_result_is_symmetric(self, phi_rv):
        q = compute_q_exec_km2(self.DIRECTION, self.DV_MAG_KM_S, 1.0, 0.02, phi_rv)
        assert np.allclose(q, q.T)

    def test_result_is_positive_semidefinite(self, phi_rv):
        """A covariance matrix cannot have negative eigenvalues."""
        q = compute_q_exec_km2(self.DIRECTION, self.DV_MAG_KM_S, 1.0, 0.02, phi_rv)
        eigvals = np.linalg.eigvalsh(q)
        assert np.all(eigvals >= -1e-15)

    def test_misalignment_only_is_nonzero(self, phi_rv):
        q = compute_q_exec_km2(self.DIRECTION, self.DV_MAG_KM_S, 1.0, None, phi_rv)
        assert not np.allclose(q, 0.0)

    def test_magnitude_sigma_only_is_nonzero(self, phi_rv):
        """Uses a tight explicit tolerance (not np.allclose's default
        atol=1e-8), since this particular direction/geometry produces a
        genuinely real but very small value (~1e-10) that the default
        tolerance would incorrectly treat as zero."""
        q = compute_q_exec_km2(self.DIRECTION, self.DV_MAG_KM_S, None, 0.02, phi_rv)
        assert not np.allclose(q, 0.0, atol=1e-13)

    def test_independent_sources_add_in_trace(self, phi_rv):
        """Trace is basis-independent, so this checks the two error
        sources compose correctly regardless of the local burn frame."""
        q_misalign = compute_q_exec_km2(self.DIRECTION, self.DV_MAG_KM_S, 1.0, None, phi_rv)
        q_magnitude = compute_q_exec_km2(self.DIRECTION, self.DV_MAG_KM_S, None, 0.02, phi_rv)
        q_both = compute_q_exec_km2(self.DIRECTION, self.DV_MAG_KM_S, 1.0, 0.02, phi_rv)
        assert math.isclose(
            np.trace(q_both), np.trace(q_misalign) + np.trace(q_magnitude), rel_tol=1e-9
        )

    def test_larger_misalignment_gives_larger_covariance(self, phi_rv):
        """Sanity: more pointing error must never produce less uncertainty."""
        q_small = compute_q_exec_km2(self.DIRECTION, self.DV_MAG_KM_S, 0.5, None, phi_rv)
        q_large = compute_q_exec_km2(self.DIRECTION, self.DV_MAG_KM_S, 2.0, None, phi_rv)
        assert np.trace(q_large) > np.trace(q_small)

    def test_zero_burn_magnitude_gives_zero_matrix(self, phi_rv):
        """No burn commanded -- there is no execution to have an error in."""
        q = compute_q_exec_km2(self.DIRECTION, 0.0, 1.0, 0.02, phi_rv)
        assert np.allclose(q, 0.0)


# ---------------------------------------------------------------------------
# Full integration: score_maneuver_candidates with real profiles
# ---------------------------------------------------------------------------

class TestExecutionErrorScoringIntegration:
    """Uses the same std_vectors-style geometry established in
    test_maneuver_scorer.py: r_rel=[0.3,0,0] km, P=0.01*I, a real
    conjunction that passes the pre-screen (MD=3.0 < 4.0 threshold)."""

    R_SAT = np.array([6853.0, 0.0, 0.0])
    V_SAT = np.array([0.0, 7.626, 0.0])
    R_REL = np.array([0.3, 0.0, 0.0])
    P_COV = np.eye(3) * 0.01
    T_BURN = "2026-04-14T08:00:00Z"
    T_CA = "2026-04-14T12:00:00Z"

    @pytest.fixture
    def policy(self):
        return OperatorPolicy(
            operator_id="TEST_OP", policy_version="2.5.0",
            max_dv_per_event_ms=2.0, mission_lifetime_days_total=1825.0,
            scoring_weights=ScoringWeights(lambda_dv=1.0, lambda_lifetime=0.8, lambda_slot_deviation=1.2),
        )

    def _make_cap(self, thrust_misalignment_deg=None, dv_magnitude_sigma=None):
        return SatelliteCapability(
            sat_id="SAT-EXEC-TEST",
            a_ref_km=_mean_motion_to_sma_km(15.3020),
            propulsion=PropulsionProfile(
                thrust_misalignment_deg=thrust_misalignment_deg,
                dv_magnitude_sigma=dv_magnitude_sigma,
            ),
            lifetime=LifetimeProfile(
                mass_kg=100.0, v_remaining_m_s=50.0, v_reserved_m_s=5.0,
                mission_lifetime_days_remaining=365.0,
            ),
            slot=ConstellationSlot(in_constellation=False),
        )

    def _score(self, cap, policy):
        return score_maneuver_candidates(
            "CID-EXEC", self.R_SAT, self.V_SAT, self.R_REL, self.P_COV,
            self.T_BURN, self.T_CA, cap, policy,
        )

    def test_perfect_burn_execution_error_modelled_is_false(self, policy):
        """AC1/AC3: no profile parameters -> flag correctly reports False."""
        cap = self._make_cap()
        result = self._score(cap, policy)
        assert result.execution_error_modelled is False

    def test_misaligned_burn_execution_error_modelled_is_true(self, policy):
        cap = self._make_cap(thrust_misalignment_deg=1.0)
        result = self._score(cap, policy)
        assert result.execution_error_modelled is True

    def test_magnitude_uncertain_burn_execution_error_modelled_is_true(self, policy):
        cap = self._make_cap(dv_magnitude_sigma=0.02)
        result = self._score(cap, policy)
        assert result.execution_error_modelled is True

    def test_misaligned_burn_m2_post_lower_than_perfect_burn(self, policy):
        """AC2, the core physics fix: adding execution-error uncertainty
        must make the post-maneuver separation estimate MORE conservative
        (lower m2_post), correcting the optimistic bias the ticket
        describes -- not leave it unchanged or make it look safer.

        1.0 deg matches the realistic bus range in gnc_interface.yaml
        (typical 0.5-1.0 deg). Independently verified against SCRUM-386's
        re-measurement: this scenario gives approximately -99.7% change
        in m2_post, not a marginal effect.
        """
        perfect = self._score(self._make_cap(), policy)
        misaligned = self._score(self._make_cap(thrust_misalignment_deg=1.0), policy)
        assert misaligned.m2_post < perfect.m2_post - 1e-6

    def test_magnitude_uncertain_m2_post_lower_than_perfect_burn(self, policy):
        """0.02 (2%) matches the realistic bus range in gnc_interface.yaml
        (typical 1-5%). See test_misaligned_burn_m2_post_lower_than_perfect_burn."""
        perfect = self._score(self._make_cap(), policy)
        uncertain = self._score(self._make_cap(dv_magnitude_sigma=0.02), policy)
        assert uncertain.m2_post < perfect.m2_post - 1e-6

    def test_both_together_lower_than_either_alone(self, policy):
        """Combined error sources should be at least as conservative as
        either alone (more uncertainty added, never less)."""
        misaligned_only = self._score(self._make_cap(thrust_misalignment_deg=1.0), policy)
        both = self._score(
            self._make_cap(thrust_misalignment_deg=1.0, dv_magnitude_sigma=0.02), policy
        )
        assert both.m2_post <= misaligned_only.m2_post

    def test_perfect_burn_matches_pre_scrum_365_baseline(self, policy):
        """Regression guard: with no execution-error parameters, m2_post
        must be numerically identical to a plain Mahalanobis calculation
        against the raw p_rel_km2 alone (Q_exec truly zero, not just
        small) -- confirms AC1 exactly, not approximately."""
        from avoid.decision_model import mahalanobis_sq, cw_phi_rv as cw
        cap = self._make_cap()
        result = self._score(cap, policy)

        # Recompute the expected value directly, independent of the
        # scorer's internals, using the same real burn direction/dt it
        # would have chosen (the result tells us which one won).
        phi_rv = cw(cap.a_ref_km, 4 * 3600.0)  # 4 hr: T_BURN to T_CA
        dv_vec = np.array(result.dv_eci_km_s)
        delta_r = phi_rv @ dv_vec
        r_post = self.R_REL - delta_r
        expected_m2_post = mahalanobis_sq(r_post, self.P_COV)

        assert math.isclose(result.m2_post, expected_m2_post, rel_tol=1e-9)

    def test_no_go_result_reflects_profile_not_hardcoded_false(self, policy):
        """Regression guard (John's PR #52 review): a no-go result must
        report execution_error_modelled based on the satellite's actual
        propulsion profile, not a hardcoded False. Previously, any
        no-go result (feasibility pre-screen failure, or no positive-
        utility candidate) always reported False regardless of the
        profile, which would misleadingly suggest a satellite that
        genuinely carries thrust_misalignment_deg/dv_magnitude_sigma
        does not model execution error, just because this particular
        event did not produce a recommended burn.

        Forces the no_utility_gain no-go path with an extreme lambda_dv,
        so no candidate burn can ever have positive utility, on a
        satellite whose profile does carry execution-error parameters.
        """
        extreme_policy = OperatorPolicy(
            operator_id="TEST_OP", policy_version="2.5.0",
            max_dv_per_event_ms=2.0, mission_lifetime_days_total=1825.0,
            scoring_weights=ScoringWeights(
                lambda_dv=1.0e6, lambda_lifetime=0.8, lambda_slot_deviation=1.2
            ),
        )
        cap = self._make_cap(thrust_misalignment_deg=1.0)
        result = self._score(cap, extreme_policy)

        assert result.no_go_reason_code != ""  # confirms this actually hit the no-go path
        assert result.execution_error_modelled is True
