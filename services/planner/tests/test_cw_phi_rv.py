"""
services/planner/tests/test_cw_phi_rv.py

SCRUM-386: guard cw_phi_rv against silently becoming Phi_rr again.

The function returned the Phi_rr (position-to-position) block for a long time
while its docstring described Phi_rv. Nothing caught it because no test looked
at the matrix itself, only at scenario outcomes, and the scenario expectations
had been fitted to the wrong matrix. So this file asserts the block directly
against the closed form, and asserts the two properties that make the
substitution impossible to repeat: the units, and the fact that it is not
Phi_rr.

SCRUM-378: cw_phi_rv is now a thin wrapper around aps_math.frames.cw_phi_full.
MU_EARTH moved with it, so this file imports MU_EARTH from aps_math.frames
instead of avoid.decision_model. Every assertion below is unchanged; only
the import line moved.

Run with (from repo root -- conftest.py handles PYTHONPATH):
    python -m pytest services/planner/tests/test_cw_phi_rv.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aps_math.frames import MU_EARTH
from avoid.decision_model import cw_phi_rv


A_KM = 6378.137 + 550.0      # 550 km circular LEO
DT_S = 4 * 3600.0            # 4 hours to TCA


def _n(a_km: float) -> float:
    """Mean motion [rad/s]."""
    return math.sqrt(MU_EARTH / (a_km ** 3))


def _phi_rv_closed_form(a_km: float, dt_s: float) -> np.ndarray:
    """Clohessy-Wiltshire Phi_rv, written out independently of the module."""
    n = _n(a_km)
    c = math.cos(n * dt_s)
    s = math.sin(n * dt_s)
    return np.array([
        [s / n,            2.0 * (1.0 - c) / n,               0.0],
        [-2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * n * dt_s) / n, 0.0],
        [0.0,              0.0,                               s / n],
    ])


def _phi_rr_closed_form(a_km: float, dt_s: float) -> np.ndarray:
    """The block that used to be returned. Present so the test can assert
    the implementation is NOT this."""
    n = _n(a_km)
    c = math.cos(n * dt_s)
    s = math.sin(n * dt_s)
    return np.array([
        [4.0 - 3.0 * c,        0.0, 0.0],
        [6.0 * (s - n * dt_s), 1.0, 0.0],
        [0.0,                  0.0, c],
    ])


class TestMatchesClosedForm:
    @pytest.mark.parametrize("a_km", [6928.137, 7078.137, 6778.137])
    @pytest.mark.parametrize("dt_s", [600.0, 3600.0, 4 * 3600.0, 24 * 3600.0])
    def test_matches_phi_rv_across_geometries(self, a_km, dt_s):
        assert np.allclose(
            cw_phi_rv(a_km, dt_s), _phi_rv_closed_form(a_km, dt_s), rtol=1e-12
        )

    def test_is_not_phi_rr(self):
        """The specific regression. Phi_rr is dimensionless; applying it to a
        delta-v in km/s yields km/s read as km."""
        assert not np.allclose(
            cw_phi_rv(A_KM, DT_S), _phi_rr_closed_form(A_KM, DT_S)
        )


class TestUnitsAndPhysics:
    def test_entries_scale_like_seconds(self):
        """Phi_rv maps km/s to km, so its entries carry units of seconds.

        Checked structurally rather than by inspection: every entry of Phi_rv
        scales as 1/n, so halving the mean motion doubles the matrix at a
        fixed number of orbits. Phi_rr, being dimensionless, would not move.
        """
        a1, a2 = 6928.137, 6928.137 * (2.0 ** (2.0 / 3.0))   # n2 = n1 / 2
        n1, n2 = _n(a1), _n(a2)
        assert math.isclose(n1 / n2, 2.0, rel_tol=1e-9)

        # Same phase angle on both orbits, so only the 1/n factor differs.
        phase = 1.7
        m1 = cw_phi_rv(a1, phase / n1)
        m2 = cw_phi_rv(a2, phase / n2)
        assert np.allclose(m2, 2.0 * m1, rtol=1e-9)

    def test_along_track_secular_term_dominates_at_long_lead(self):
        """Why prograde wins for collision avoidance.

        The (1,1) entry carries -3*dt, which grows without bound, while every
        other entry is bounded by a multiple of 1/n. Over hours of lead time
        the along-track response is far larger than radial or cross-track.
        """
        m = cw_phi_rv(A_KM, DT_S)
        along_track = abs(m[1, 1])
        others = [abs(m[0, 0]), abs(m[0, 1]), abs(m[1, 0]), abs(m[2, 2])]
        assert along_track > 10.0 * max(others)

    def test_a_realistic_burn_moves_kilometres_not_millimetres(self):
        """Anchors the magnitude that made the original bug obvious.

        A 2 m/s along-track burn 4 hours before TCA. Phi_rr gave 2 metres of
        separation change. Phi_rv gives tens of kilometres.
        """
        dv_km_s = np.array([0.0, 0.002, 0.0])
        delta_r = cw_phi_rv(A_KM, DT_S) @ dv_km_s
        assert 50.0 < np.linalg.norm(delta_r) < 200.0

    def test_zero_dt_is_the_zero_map(self):
        """No time to act, no displacement."""
        assert np.allclose(cw_phi_rv(A_KM, 0.0), np.zeros((3, 3)))

    def test_cross_track_is_decoupled(self):
        """Out-of-plane motion does not couple into the orbital plane."""
        m = cw_phi_rv(A_KM, DT_S)
        assert m[2, 0] == 0.0 and m[2, 1] == 0.0
        assert m[0, 2] == 0.0 and m[1, 2] == 0.0


class TestGuards:
    @pytest.mark.parametrize("bad_a", [0.0, -1.0])
    def test_non_positive_sma_raises(self, bad_a):
        with pytest.raises(ValueError):
            cw_phi_rv(bad_a, DT_S)
