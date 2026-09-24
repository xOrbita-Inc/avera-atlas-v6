"""tests/test_post_burn_covariance.py

SCRUM-452: the real post-burn covariance -- an execution-error seed grown along
the trajectory by the state transition matrix.

This replaces two approximations that made the secondary screen falsely
confident: a covariance that was constant across 72 h, and one that was the
combined relative covariance rather than the primary's own. The tests below are
mostly about growth, because growth is the thing that was missing.

Run from repo root:
    python -m pytest services/planner/tests/test_post_burn_covariance.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.leolabs_ephemeris import (
    DEFAULT_STEP_S,
    LeoLabsEphemerisError,
    build_screening_ephemeris,
    execution_error_velocity_covariance_km2_s2,
    propagate_covariance,
    seed_post_burn_covariance_km2,
    state_transition_matrices,
)
from common.orbit_propagation import MU_EARTH

_A_KM = 6838.0
_V_CIRC = math.sqrt(MU_EARTH / _A_KM)
_R0 = [_A_KM, 0.0, 0.0]
_V0 = [0.0, _V_CIRC, 0.0]
_EPOCH = "2026-09-24T12:00:00Z"

# 10 m position one-sigma, expressed in km^2.
_P_POS = np.diag([1.0e-4, 1.0e-4, 1.0e-4])
# 0.1 m/s prograde burn.
_DV = [0.0, 1.0e-4, 0.0]
_F = 0.02                        # 2% magnitude one-sigma
_THETA = math.radians(1.0)       # 1 degree pointing one-sigma


# ---------------------------------------------------------------------------
# The execution-error velocity covariance (Gates)
# ---------------------------------------------------------------------------

class TestExecutionErrorCovariance:
    def test_magnitude_variance_lies_along_the_burn_direction(self):
        cov = execution_error_velocity_covariance_km2_s2(_DV, _F, _THETA)
        dv_hat = np.asarray(_DV) / np.linalg.norm(_DV)
        sigma_along = math.sqrt(float(dv_hat @ cov @ dv_hat))
        assert sigma_along == pytest.approx(_F * np.linalg.norm(_DV), rel=1e-12)

    def test_pointing_variance_lies_in_the_perpendicular_plane(self):
        cov = execution_error_velocity_covariance_km2_s2(_DV, _F, _THETA)
        dv_hat = np.asarray(_DV) / np.linalg.norm(_DV)
        expected = _THETA * float(np.linalg.norm(_DV))
        for axis in np.eye(3):
            perp = axis - float(axis @ dv_hat) * dv_hat
            if float(np.linalg.norm(perp)) < 1e-12:
                continue
            perp /= np.linalg.norm(perp)
            assert math.sqrt(float(perp @ cov @ perp)) == pytest.approx(
                expected, rel=1e-9)

    def test_it_rotates_correctly_for_a_non_axis_aligned_burn(self):
        """The case a burn-frame-only implementation would get wrong."""
        dv = [3.0e-5, -4.0e-5, 1.2e-4]
        cov = execution_error_velocity_covariance_km2_s2(dv, _F, _THETA)
        dv_hat = np.asarray(dv) / np.linalg.norm(dv)
        dv_mag = float(np.linalg.norm(dv))

        assert math.sqrt(float(dv_hat @ cov @ dv_hat)) == pytest.approx(
            _F * dv_mag, rel=1e-9)
        # Eigenvalues are the burn-frame variances regardless of orientation.
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))
        expected = np.sort([(_F * dv_mag) ** 2] + [(_THETA * dv_mag) ** 2] * 2)
        assert np.allclose(eigenvalues, expected, rtol=1e-9)

    def test_the_result_is_symmetric_and_psd(self):
        cov = execution_error_velocity_covariance_km2_s2(
            [3.0e-5, -4.0e-5, 1.2e-4], _F, _THETA)
        assert np.allclose(cov, cov.T)
        assert float(np.linalg.eigvalsh(cov).min()) >= -1e-30

    def test_a_zero_burn_has_no_execution_error(self):
        """Correct, not degenerate: no burn, no execution error."""
        cov = execution_error_velocity_covariance_km2_s2([0.0, 0.0, 0.0], _F, _THETA)
        assert np.all(cov == 0.0)

    def test_the_two_parameters_scale_the_two_axes_independently(self):
        doubled_f = execution_error_velocity_covariance_km2_s2(_DV, 2 * _F, _THETA)
        base = execution_error_velocity_covariance_km2_s2(_DV, _F, _THETA)
        dv_hat = np.asarray(_DV) / np.linalg.norm(_DV)
        assert math.sqrt(float(dv_hat @ doubled_f @ dv_hat)) == pytest.approx(
            2 * math.sqrt(float(dv_hat @ base @ dv_hat)), rel=1e-9)

    @pytest.mark.parametrize("bad_dv", [[1.0, 2.0], [float("nan"), 0.0, 0.0]])
    def test_an_unusable_delta_v_raises(self, bad_dv):
        with pytest.raises(LeoLabsEphemerisError):
            execution_error_velocity_covariance_km2_s2(bad_dv, _F, _THETA)

    def test_negative_parameters_raise(self):
        with pytest.raises(LeoLabsEphemerisError):
            execution_error_velocity_covariance_km2_s2(_DV, -0.01, _THETA)


# ---------------------------------------------------------------------------
# The seed
# ---------------------------------------------------------------------------

class TestSeed:
    def test_the_position_block_is_the_supplied_p_post(self):
        p0 = seed_post_burn_covariance_km2(_P_POS, _DV, _F, _THETA)
        assert np.allclose(p0[:3, :3], _P_POS)

    def test_the_velocity_block_is_the_execution_error(self):
        p0 = seed_post_burn_covariance_km2(_P_POS, _DV, _F, _THETA)
        expected = execution_error_velocity_covariance_km2_s2(_DV, _F, _THETA)
        assert np.allclose(p0[3:, 3:], expected)

    def test_cross_terms_are_zero_at_the_burn_epoch(self):
        """Position and velocity are uncorrelated at the instant of the burn.

        The correlations are not assumed here; they develop under propagation.
        """
        p0 = seed_post_burn_covariance_km2(_P_POS, _DV, _F, _THETA)
        assert np.all(p0[:3, 3:] == 0.0)
        assert np.all(p0[3:, :3] == 0.0)

    def test_an_absent_position_covariance_is_still_refused(self):
        """SCRUM-440's rule survives: None means unknown, not zero."""
        with pytest.raises(LeoLabsEphemerisError):
            seed_post_burn_covariance_km2(None, _DV, _F, _THETA)


# ---------------------------------------------------------------------------
# The state transition matrix and growth
# ---------------------------------------------------------------------------

class TestStateTransitionMatrix:
    def test_phi_at_zero_is_the_identity(self):
        phis = state_transition_matrices(_R0, _V0, np.array([0.0, 3600.0]))
        assert np.allclose(phis[0], np.eye(6), atol=1e-6)

    def test_a_pure_velocity_seed_grows_along_track_position_variance(self):
        """The load-bearing behaviour.

        A velocity one-sigma integrates into along-track position uncertainty.
        The growth is faster than the naive sigma_v * t, because an along-track
        velocity change alters the semi-major axis and therefore the period,
        which drifts the object along-track secularly -- the familiar factor of
        about three. Both bounds are asserted so a kinematics-only implementation
        (which would give exactly sigma_v * t) fails here.
        """
        sigma_v = 1.0e-6                      # km/s, 1 mm/s
        p0 = np.zeros((6, 6))
        p0[4, 4] = sigma_v ** 2               # along-track at t=0

        times = np.array([0.0, 3600.0, 24 * 3600.0, 72 * 3600.0])
        covs = propagate_covariance(p0, state_transition_matrices(_R0, _V0, times))

        sigmas = [math.sqrt(float(np.trace(c[:3, :3]))) for c in covs]
        assert sigmas[0] == pytest.approx(0.0, abs=1e-12)
        for index, t in enumerate(times[1:], start=1):
            naive = sigma_v * float(t)
            assert sigmas[index] > naive
            assert sigmas[index] < 6.0 * naive

    def test_growth_is_quadratic_in_time_to_the_expected_order(self):
        """Doubling the elapsed time roughly doubles the along-track sigma.

        Linear in sigma, quadratic in variance, which is what the secular drift
        term gives once it dominates.
        """
        p0 = np.zeros((6, 6))
        p0[4, 4] = (1.0e-6) ** 2
        times = np.array([0.0, 24 * 3600.0, 48 * 3600.0])
        covs = propagate_covariance(p0, state_transition_matrices(_R0, _V0, times))
        s24 = math.sqrt(float(np.trace(covs[1][:3, :3])))
        s48 = math.sqrt(float(np.trace(covs[2][:3, :3])))
        assert s48 / s24 == pytest.approx(2.0, rel=0.2)

    def test_propagated_covariances_stay_symmetric_and_psd(self):
        p0 = seed_post_burn_covariance_km2(_P_POS, _DV, _F, _THETA)
        times = np.linspace(0.0, 72 * 3600.0, 25)
        for cov in propagate_covariance(p0, state_transition_matrices(_R0, _V0, times)):
            assert np.allclose(cov, cov.T, atol=1e-18)
            assert float(np.linalg.eigvalsh(cov).min()) > -1e-12

    def test_the_j2_propagator_is_used_not_replaced(self, monkeypatch):
        """The STM must differentiate the same dynamics the states come from."""
        import common.leolabs_ephemeris as le

        calls = {"n": 0}
        real = le.propagate_j2_series

        def _counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(le, "propagate_j2_series", _counting)
        state_transition_matrices(_R0, _V0, np.array([0.0, 3600.0]))
        # One nominal plus six perturbed passes.
        assert calls["n"] == 7


# ---------------------------------------------------------------------------
# The emitted ephemeris
# ---------------------------------------------------------------------------

def _grown(**kw):
    params = dict(
        epoch_utc=_EPOCH, r_sat_km=_R0, v_sat_km_s=_V0, p_post_eci_km2=_P_POS,
        dv_eci_km_s=_DV, execution_error_magnitude_fraction=_F,
        execution_error_pointing_sigma_rad=_THETA,
    )
    params.update(kw)
    return build_screening_ephemeris(**params)


class TestEmittedEphemeris:
    def test_every_state_carries_its_own_6x6(self):
        states = _grown(horizon_hours=6.0, step_s=1800.0)["states"]
        for state in states:
            cov = np.array(state["covariance"], dtype=float)
            assert cov.shape == (6, 6)
        assert states[-1]["covariance"] != states[0]["covariance"]

    def test_the_trace_grows_across_the_horizon(self):
        states = _grown()["states"]
        traces = [float(np.trace(np.array(s["covariance"], dtype=float)))
                  for s in states]
        assert traces[-1] > traces[0]
        assert all(b >= a * 0.999 for a, b in zip(traces, traces[1:]))

    def test_the_first_state_is_exactly_the_seed(self):
        """Phi(0) is the identity, so this must hold to numerical precision."""
        first = np.array(_grown()["states"][0]["covariance"], dtype=float)
        seed_m = seed_post_burn_covariance_km2(_P_POS, _DV, _F, _THETA) * 1.0e6
        assert np.allclose(first, seed_m, rtol=1e-9, atol=1e-6)

    def test_the_seed_position_block_is_the_supplied_p_post(self):
        """Not p_rel: the primary's own covariance is what seeds this."""
        first = np.array(_grown()["states"][0]["covariance"], dtype=float)
        assert np.allclose(first[:3, :3], np.asarray(_P_POS) * 1.0e6,
                           rtol=1e-9, atol=1e-6)

    def test_the_fan_out_is_material_over_72_hours(self):
        """The property the constant floor did not have, stated in metres."""
        states = _grown()["states"]
        sigma_0 = math.sqrt(float(np.trace(
            np.array(states[0]["covariance"], dtype=float)[:3, :3])))
        sigma_end = math.sqrt(float(np.trace(
            np.array(states[-1]["covariance"], dtype=float)[:3, :3])))
        assert sigma_0 < 100.0            # metres, the 10 m seed
        assert sigma_end > 1000.0         # kilometres-scale by 72 h
        assert sigma_end > 50.0 * sigma_0

    def test_the_execution_error_contributes_to_the_growth(self):
        """The burn must move the answer, or the velocity seed is doing nothing."""
        with_burn = float(np.trace(np.array(_grown()["states"][-1]["covariance"])))
        without = float(np.trace(np.array(build_screening_ephemeris(
            _EPOCH, _R0, _V0, _P_POS)["states"][-1]["covariance"])))
        assert with_burn > without

    def test_a_realistic_maneuver_makes_the_execution_error_dominant(self):
        """Measured, and narrower than the plan's framing.

        Both the position seed and the execution error grow the same way: each
        perturbs the semi-major axis, which drifts the object along-track
        secularly. Which dominates is simply which produces the larger delta-a.

        Against a 10 m position seed, a 0.1 m/s burn adds only a couple of
        percent at 72 h, while a 1 m/s burn roughly doubles the uncertainty and a
        5 m/s burn is an order of magnitude. So the execution-error parameters
        matter most for real avoidance burns, and the quality of the position
        seed matters most for small ones -- both are worth having, and neither is
        universally 'the load-bearing part'.
        """
        def sigma_72h(dv_m_s):
            eph = _grown(dv_eci_km_s=[0.0, dv_m_s / 1000.0, 0.0])
            cov = np.array(eph["states"][-1]["covariance"], dtype=float)
            return math.sqrt(float(np.trace(cov[:3, :3])))

        baseline = math.sqrt(float(np.trace(np.array(
            build_screening_ephemeris(_EPOCH, _R0, _V0, _P_POS)
            ["states"][-1]["covariance"], dtype=float)[:3, :3])))

        assert sigma_72h(0.1) < 1.2 * baseline      # small burn: minor addition
        assert sigma_72h(1.0) > 1.5 * baseline      # realistic burn: dominant
        assert sigma_72h(5.0) > 5.0 * baseline      # large burn: overwhelming

    def test_the_units_are_still_metres_squared(self):
        first = np.array(_grown()["states"][0]["covariance"], dtype=float)
        # 1e-4 km^2 is a 10 m one-sigma, i.e. 100 m^2.
        assert math.sqrt(first[0][0]) == pytest.approx(10.0, rel=1e-6)

    def test_the_document_shape_is_unchanged(self):
        """SCRUM-452 changes the numbers, not the schema LeoLabs receives."""
        eph = _grown(horizon_hours=2.0, step_s=1800.0)
        assert sorted(eph) == ["covarianceFrame", "frame", "states"]
        assert eph["frame"] == eph["covarianceFrame"] == "EME2000"
        for state in eph["states"]:
            assert sorted(state) == [
                "covariance", "position", "timestamp", "velocity"]


class TestCaveatsAreGone:
    """SCRUM-440 and SCRUM-442 both shipped covariance caveats. The covariance is
    real now, and a stale caveat would understate the screen."""

    def test_the_ephemeris_module_no_longer_calls_it_a_floor(self):
        from pathlib import Path
        source = Path(
            "services/planner/common/leolabs_ephemeris.py").read_text()
        assert "constant position-only floor" not in source
        assert "tracked fast-follow" not in source

    def test_the_artifact_operator_note_carries_no_caveat(self):
        from pathlib import Path
        source = Path("services/planner/common/atlas_artifact.py").read_text()
        assert "constant position-only floor" not in source
        assert "tracked fast-follow" not in source

    def test_the_persisted_record_carries_no_caveat(self):
        from pathlib import Path
        source = Path("services/planner/server.py").read_text()
        assert "covariance_caveat" not in source
        assert "tracked fast-follow" not in source
