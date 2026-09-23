"""libs/aps_math/tests/test_orbits.py

SCRUM-447: the shared two-body propagator the 3D globe draws its rings with.

It moved here from services/ui/app/main.py so the planner and the UI cannot draw
the same orbit two different ways. These tests pin the properties the globe
actually relies on -- a track that closes, a sane period, and a radius that does
not drift -- rather than exact point values, which are an RK4 implementation
detail.
"""

from __future__ import annotations

import math

import pytest

from aps_math.orbits import (
    MU_EARTH_KM3_S2,
    DEFAULT_STEPS,
    orbital_period_s,
    propagate_two_body,
)

# A circular LEO state: 6790 km radius, speed from sqrt(mu/r).
_R0 = [6790.0, 0.0, 0.0]
_V_CIRC = math.sqrt(MU_EARTH_KM3_S2 / 6790.0)
_V0 = [0.0, _V_CIRC * math.cos(math.radians(49.0)),
       _V_CIRC * math.sin(math.radians(49.0))]


def _mag(v):
    return math.sqrt(sum(c * c for c in v))


class TestPeriod:
    def test_a_leo_period_is_about_ninety_minutes(self):
        minutes = orbital_period_s(_R0) / 60.0
        assert 88.0 < minutes < 96.0

    def test_period_grows_with_radius(self):
        assert orbital_period_s([7500.0, 0, 0]) > orbital_period_s(_R0)


class TestPropagate:
    def test_returns_one_more_point_than_steps(self):
        assert len(propagate_two_body(_R0, _V0)) == DEFAULT_STEPS + 1
        assert len(propagate_two_body(_R0, _V0, n_steps=24)) == 25

    def test_the_track_closes_on_itself(self):
        """The globe appends the first point to close the ring, so the last point
        has to land back near the first or the ring shows a visible seam."""
        track = propagate_two_body(_R0, _V0)
        gap = math.dist(track[0], track[-1])
        assert gap < 0.02 * _mag(_R0)

    def test_a_circular_orbit_keeps_its_radius(self):
        track = propagate_two_body(_R0, _V0)
        radii = [_mag(p) for p in track]
        assert max(radii) - min(radii) < 0.01 * _mag(_R0)

    def test_the_orbit_stays_in_its_plane(self):
        """Two-body motion conserves the angular-momentum direction, so every
        point lies in one plane. A track that wanders out of plane would draw a
        ring the operator could not trust as an orbit."""
        track = propagate_two_body(_R0, _V0)
        r0, v0 = track[0], track[1]
        nx = r0[1] * v0[2] - r0[2] * v0[1]
        ny = r0[2] * v0[0] - r0[0] * v0[2]
        nz = r0[0] * v0[1] - r0[1] * v0[0]
        n = [nx, ny, nz]
        nmag = _mag(n)
        for p in track:
            out_of_plane = abs(sum(a * b for a, b in zip(p, n))) / nmag
            assert out_of_plane < 1.0  # km

    def test_units_are_km_so_a_metres_caller_is_obviously_wrong(self):
        """LeoLabs returns metres. Feeding those in unconverted gives a radius
        1000x too large, and the period explodes -- which is the failure this
        propagator's callers must convert to avoid."""
        metres = [c * 1000.0 for c in _R0]
        assert orbital_period_s(metres) > 1000 * orbital_period_s(_R0)


class TestGuards:
    @pytest.mark.parametrize("r0,v0", [
        ([1.0, 2.0], [0.0, 7.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, 7.0, 0.0]),
        ([float("nan"), 0.0, 0.0], [0.0, 7.0, 0.0]),
        ([6790.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]),
    ])
    def test_an_unusable_state_raises(self, r0, v0):
        with pytest.raises(ValueError):
            propagate_two_body(r0, v0)
