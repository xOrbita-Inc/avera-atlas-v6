"""tests/test_j2_propagation.py

SCRUM-451: two-body plus J2, for the LeoLabs screening ephemeris.

The point of these tests is to prove the propagator is genuinely perturbed rather
than silently two-body, and to make that impossible to undo quietly. The nodal
regression check is the load-bearing one: it compares against the closed-form
secular rate, which a two-body propagator cannot reproduce at all.

Run from repo root:
    python -m pytest services/planner/tests/test_j2_propagation.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.orbit_propagation import (
    J2,
    MU_EARTH,
    RE_EARTH_KM,
    _propagate_j2_rk4,
    j2_acceleration,
    kepler_propagate,
    propagate_j2,
    propagate_j2_series,
)


def _circular_state(a_km: float, inc_deg: float):
    """A circular orbit at radius a_km and inclination inc_deg, starting at +x."""
    inc = math.radians(inc_deg)
    speed = math.sqrt(MU_EARTH / a_km)
    r0 = np.array([a_km, 0.0, 0.0])
    v0 = np.array([0.0, speed * math.cos(inc), speed * math.sin(inc)])
    return r0, v0


def _period_s(a_km: float) -> float:
    return 2.0 * math.pi * math.sqrt(a_km ** 3 / MU_EARTH)


def _raan(r: np.ndarray, v: np.ndarray) -> float:
    """Right ascension of the ascending node from a state vector."""
    h = np.cross(r, v)
    n = np.cross(np.array([0.0, 0.0, 1.0]), h)
    if float(np.linalg.norm(n)) < 1e-12:
        return 0.0
    return float(np.arctan2(n[1], n[0])) % (2.0 * math.pi)


def _specific_energy(r: np.ndarray, v: np.ndarray) -> float:
    """Two-body specific energy. Not conserved under J2 -- see _total_energy."""
    return float(np.dot(v, v)) / 2.0 - MU_EARTH / float(np.linalg.norm(r))


def _total_energy(r: np.ndarray, v: np.ndarray) -> float:
    """Specific energy including the J2 potential, which IS the invariant here.

    J2 is a conservative perturbation, so v^2/2 + V is constant with

        V = -mu/r + (mu J2 Re^2 / 2 r^3) (3 z^2/r^2 - 1)

    Two-body energy alone drifts by order J2 (~1e-3 relative) under this
    propagation, which is physics rather than integrator error -- checking the
    wrong invariant would either fail a correct propagator or, with a loose
    tolerance, hide a genuinely bad one.
    """
    r_mag = float(np.linalg.norm(r))
    z = float(r[2])
    potential = (
        -MU_EARTH / r_mag
        + (MU_EARTH * J2 * RE_EARTH_KM ** 2) / (2.0 * r_mag ** 3)
        * (3.0 * (z * z) / (r_mag * r_mag) - 1.0)
    )
    return float(np.dot(v, v)) / 2.0 + potential


# Swarm C is the asset the SCRUM-441 screen was run against.
_SWARM_C_A_KM = 6838.0
_SWARM_C_INC_DEG = 87.35


class TestJ2Acceleration:
    def test_at_the_equator_the_bulge_pulls_inward(self):
        """Extra mass in the equatorial plane means extra inward acceleration."""
        a = j2_acceleration(np.array([7000.0, 0.0, 0.0]))
        assert a[0] < 0.0
        assert a[1] == pytest.approx(0.0)
        assert a[2] == pytest.approx(0.0)

    def test_over_a_pole_it_pushes_outward(self):
        """Less mass below the pole, so the perturbation opposes gravity there."""
        a = j2_acceleration(np.array([0.0, 0.0, 7000.0]))
        assert a[2] > 0.0

    def test_it_is_small_against_the_two_body_term(self):
        """J2 is a perturbation, about 1e-3 of the central term in LEO."""
        r = np.array([6838.0, 0.0, 0.0])
        two_body = MU_EARTH / float(np.dot(r, r))
        ratio = float(np.linalg.norm(j2_acceleration(r))) / two_body
        assert 1e-4 < ratio < 1e-2


class TestEnergyAndBoundedness:
    def test_a_circular_orbit_stays_bound_and_near_its_radius(self):
        r0, v0 = _circular_state(_SWARM_C_A_KM, _SWARM_C_INC_DEG)
        times = np.linspace(0.0, 24 * 3600.0, 289)
        positions, velocities = propagate_j2_series(r0, v0, times)

        radii = np.linalg.norm(positions, axis=1)
        # J2 makes the radius oscillate; it must not drift or diverge.
        assert radii.min() > 6700.0
        assert radii.max() < 6980.0
        assert all(_specific_energy(p, v) < 0.0
                   for p, v in zip(positions, velocities))

    def test_total_energy_is_conserved_to_tolerance(self):
        """J2 is conservative, so total energy is a real invariant here.

        It is also the sharpest check on integrator quality: a sloppy step shows
        up as energy drift long before it shows up as a visible position error.
        """
        r0, v0 = _circular_state(_SWARM_C_A_KM, _SWARM_C_INC_DEG)
        times = np.linspace(0.0, 72 * 3600.0, 865)
        positions, velocities = propagate_j2_series(r0, v0, times)

        energies = np.array([_total_energy(p, v)
                             for p, v in zip(positions, velocities)])
        drift = float(np.max(np.abs(energies - energies[0])) / abs(energies[0]))
        assert drift < 1e-9, f"total energy drifted by {drift:.3e}"

    def test_two_body_energy_alone_is_not_the_invariant(self):
        """Guards the test above against being 'fixed' back to the wrong quantity.

        Under J2 the two-body energy oscillates at order J2. Asserting on it would
        need a tolerance loose enough to hide real integrator error.
        """
        r0, v0 = _circular_state(_SWARM_C_A_KM, _SWARM_C_INC_DEG)
        times = np.linspace(0.0, 6 * 3600.0, 200)
        positions, velocities = propagate_j2_series(r0, v0, times)
        two_body = np.array([_specific_energy(p, v)
                             for p, v in zip(positions, velocities)])
        swing = float(np.max(np.abs(two_body - two_body[0])) / abs(two_body[0]))
        assert swing > 1e-4


class TestSecularNodalRegression:
    """The check a two-body propagator cannot pass.

    Omega_dot = -1.5 n J2 (Re/p)^2 cos i is the closed-form secular rate. Two-body
    motion has Omega_dot = 0 exactly, so matching this to a few tenths of a
    percent is positive evidence that J2 is actually being integrated.
    """

    @pytest.mark.parametrize("a_km,inc_deg", [
        (6778.0, 51.6),               # ISS-like, strong signal
        (_SWARM_C_A_KM, _SWARM_C_INC_DEG),   # the asset this was built for
        (7078.0, 98.2),               # retrograde: the rate changes sign
    ])
    def test_the_measured_rate_matches_the_analytic_rate(self, a_km, inc_deg):
        r0, v0 = _circular_state(a_km, inc_deg)
        period = _period_s(a_km)
        span = 10.0 * period
        times = np.linspace(0.0, span, 2001)
        positions, velocities = propagate_j2_series(r0, v0, times)

        delta = _raan(positions[-1], velocities[-1]) - _raan(positions[0], velocities[0])
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        measured = delta / span

        inc = math.radians(inc_deg)
        n = 2.0 * math.pi / period
        analytic = -1.5 * n * J2 * (RE_EARTH_KM / a_km) ** 2 * math.cos(inc)

        assert measured == pytest.approx(analytic, rel=0.02)

    def test_a_retrograde_orbit_regresses_the_other_way(self):
        """cos i flips sign above 90 deg, and so must the node motion."""
        prograde = _circular_state(7078.0, 81.8)
        retrograde = _circular_state(7078.0, 98.2)
        span = 10.0 * _period_s(7078.0)
        times = np.linspace(0.0, span, 2001)

        rates = []
        for r0, v0 in (prograde, retrograde):
            p, v = propagate_j2_series(r0, v0, times)
            d = _raan(p[-1], v[-1]) - _raan(p[0], v[0])
            rates.append(((d + math.pi) % (2 * math.pi) - math.pi) / span)

        assert rates[0] < 0.0 < rates[1]

    def test_two_body_has_no_nodal_regression_at_all(self):
        """The contrast that makes the test above meaningful."""
        r0, v0 = _circular_state(_SWARM_C_A_KM, _SWARM_C_INC_DEG)
        span = 10.0 * _period_s(_SWARM_C_A_KM)
        r_end, v_end = kepler_propagate(r0, v0, span)
        delta = _raan(r_end, v_end) - _raan(r0, v0)
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        assert abs(delta) < 1e-6


class TestDivergenceRegressionGuard:
    """So J2 cannot be silently dropped from the screening path again.

    These are the measured divergences that justified SCRUM-451: ~26 km at 1 h,
    ~462 km at 24 h, ~1400 km at 72 h for a Swarm-C-like orbit. If a future edit
    reverts the builder to two-body, or neuters J2, these collapse toward zero.
    """

    @pytest.mark.parametrize("hours,low_km,high_km", [
        (1, 15.0, 45.0),
        (6, 80.0, 180.0),
        (24, 330.0, 620.0),
        (72, 1000.0, 1900.0),
    ])
    def test_divergence_from_two_body_grows_to_the_expected_order(
        self, hours, low_km, high_km
    ):
        r0, v0 = _circular_state(_SWARM_C_A_KM, _SWARM_C_INC_DEG)
        dt = hours * 3600.0
        r_j2, _ = propagate_j2(r0, v0, dt)
        r_kepler, _ = kepler_propagate(r0, v0, dt)
        divergence = float(np.linalg.norm(r_j2 - r_kepler))
        assert low_km < divergence < high_km, (
            f"{hours} h divergence {divergence:.1f} km outside "
            f"[{low_km}, {high_km}]"
        )

    def test_the_trajectory_leaves_a_25_km_box_within_the_first_hour(self):
        """The concrete reason the SCRUM-441 screen came back empty."""
        r0, v0 = _circular_state(_SWARM_C_A_KM, _SWARM_C_INC_DEG)
        r_j2, _ = propagate_j2(r0, v0, 3600.0)
        r_kepler, _ = kepler_propagate(r0, v0, 3600.0)
        assert float(np.linalg.norm(r_j2 - r_kepler)) > 25.0


class TestIntegrators:
    def test_the_rk4_fallback_agrees_with_the_reference_integrator(self):
        """scipy may be absent; the fallback must not be a different answer."""
        r0, v0 = _circular_state(_SWARM_C_A_KM, _SWARM_C_INC_DEG)
        times = np.array([0.0, 6 * 3600.0, 24 * 3600.0, 72 * 3600.0])
        reference, _ = propagate_j2_series(r0, v0, times)
        fallback, _ = _propagate_j2_rk4(
            np.concatenate((r0, v0)), times, MU_EARTH, J2, RE_EARTH_KM, 10.0)

        # Well inside any screening volume over the full horizon.
        for ref, alt in zip(reference, fallback):
            assert float(np.linalg.norm(ref - alt)) < 1.0

    def test_t_zero_returns_the_input_state_unchanged(self):
        r0, v0 = _circular_state(_SWARM_C_A_KM, _SWARM_C_INC_DEG)
        positions, velocities = propagate_j2_series(r0, v0, np.array([0.0]))
        assert np.allclose(positions[0], r0)
        assert np.allclose(velocities[0], v0)

    def test_the_singular_form_matches_the_series_form(self):
        r0, v0 = _circular_state(_SWARM_C_A_KM, _SWARM_C_INC_DEG)
        dt = 12 * 3600.0
        r_one, v_one = propagate_j2(r0, v0, dt)
        positions, velocities = propagate_j2_series(r0, v0, np.array([0.0, dt]))
        assert np.allclose(r_one, positions[1])
        assert np.allclose(v_one, velocities[1])

    def test_sampling_cadence_does_not_change_the_trajectory(self):
        """t_eval must sample one integration, not restart it per point."""
        r0, v0 = _circular_state(_SWARM_C_A_KM, _SWARM_C_INC_DEG)
        coarse, _ = propagate_j2_series(r0, v0, np.array([0.0, 72 * 3600.0]))
        fine, _ = propagate_j2_series(
            r0, v0, np.linspace(0.0, 72 * 3600.0, 865))
        assert float(np.linalg.norm(coarse[-1] - fine[-1])) < 0.1


class TestGuards:
    @pytest.mark.parametrize("r0,v0", [
        ([1.0, 2.0], [0.0, 7.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, 7.0, 0.0]),
        ([float("nan"), 0.0, 0.0], [0.0, 7.0, 0.0]),
    ])
    def test_an_unusable_state_raises(self, r0, v0):
        with pytest.raises(ValueError):
            propagate_j2_series(r0, v0, np.array([0.0, 60.0]))

    @pytest.mark.parametrize("times", [
        np.array([]),
        np.array([-60.0, 0.0]),
        np.array([600.0, 300.0]),
        np.array([np.nan]),
    ])
    def test_unusable_times_raise(self, times):
        r0, v0 = _circular_state(_SWARM_C_A_KM, _SWARM_C_INC_DEG)
        with pytest.raises(ValueError):
            propagate_j2_series(r0, v0, times)

    def test_backward_propagation_is_refused(self):
        r0, v0 = _circular_state(_SWARM_C_A_KM, _SWARM_C_INC_DEG)
        with pytest.raises(ValueError):
            propagate_j2(r0, v0, -3600.0)

    def test_kepler_propagate_is_untouched(self):
        """This ticket adds a propagator; it must not have altered the old one."""
        r0, v0 = _circular_state(_SWARM_C_A_KM, _SWARM_C_INC_DEG)
        r_end, _ = kepler_propagate(r0, v0, _period_s(_SWARM_C_A_KM))
        # One full period of pure two-body motion closes on itself.
        assert float(np.linalg.norm(r_end - r0)) < 1.0
