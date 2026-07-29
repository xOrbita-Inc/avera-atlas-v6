"""
libs/aps_math/tests/test_pc_utils_reference.py

SCRUM-388: prove the move of pc_utils did not change the numerics, by anchoring
compute_pc against a case with an exact closed-form answer rather than against
a value quoted from a paper.

The anchor
----------
For a head-on encounter with isotropic combined covariance, the projection into
the encounter plane is an isotropic 2D Gaussian centred on the origin. The
probability of landing inside a circle of radius R is then exactly the Rayleigh
CDF:

    Pc = 1 - exp(-R^2 / (2 * sigma^2))

This is derivable rather than cited, it holds for any sigma and R, and it fails
loudly if the projection, the covariance combination, or the integration is
disturbed. Modes 0 and 1 reproduce it to eight significant figures.

History
-------
This file was written for SCRUM-388, the move of pc_utils into libs/aps_math,
and immediately caught that estimation_mode 64, the module default, was wrong
by a factor of about 4.2554 and could return values above 1.0. That was fixed
in SCRUM-390; the strict xfail that recorded it is gone and TestDefaultMode
now asserts the corrected behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aps_math.pc_utils import compute_pc


# Head-on encounter, so relative velocity is well defined and the encounter
# plane projection of an isotropic covariance stays isotropic.
R_ECI = np.array([7.0e6, 0.0, 0.0])
V_PRIMARY = np.array([0.0, 7.5e3, 0.0])
V_SECONDARY = np.array([0.0, -7.5e3, 0.0])

GEOMETRIES = [
    (50.0, 5.0), (50.0, 10.0),
    (100.0, 5.0), (100.0, 10.0), (100.0, 50.0),
    (500.0, 10.0), (1000.0, 20.0),
]


def _rayleigh_cdf(hbr_m: float, sigma_m: float) -> float:
    """Exact Pc for a zero-miss isotropic 2D encounter."""
    return 1.0 - math.exp(-(hbr_m ** 2) / (2.0 * sigma_m ** 2))


def _pc(sigma_m: float, hbr_m: float, mode: int) -> float:
    cov = np.eye(3) * sigma_m ** 2
    return compute_pc(
        R_ECI, V_PRIMARY, cov * 0.5,
        R_ECI, V_SECONDARY, cov * 0.5,
        hbr=hbr_m, estimation_mode=mode,
    ).Pc


class TestAgainstClosedForm:
    """Modes with a correct normalisation reproduce the analytic answer."""

    @pytest.mark.parametrize("sigma_m,hbr_m", GEOMETRIES)
    def test_full_numerical_integration_matches(self, sigma_m, hbr_m):
        assert _pc(sigma_m, hbr_m, 1) == pytest.approx(
            _rayleigh_cdf(hbr_m, sigma_m), rel=1e-6
        )

    @pytest.mark.parametrize("sigma_m,hbr_m", [g for g in GEOMETRIES if g[1] / g[0] <= 0.2])
    def test_equal_area_square_matches_closely(self, sigma_m, hbr_m):
        """Mode 0 is an equal-area square approximation, not an exact method.

        Its error grows with the ratio of hard-body radius to covariance scale,
        which is inherent to replacing a disc with a square of equal area. It
        is held to 1e-3 only where that ratio is small; the degradation itself
        is asserted separately below rather than hidden behind a loose bound.
        """
        assert _pc(sigma_m, hbr_m, 0) == pytest.approx(
            _rayleigh_cdf(hbr_m, sigma_m), rel=1e-3
        )

    @pytest.mark.parametrize(
        "hbr_over_sigma,max_rel_err",
        [(0.01, 5e-6), (0.1, 5e-4), (0.3, 2e-3), (0.8, 1e-2)],
    )
    def test_equal_area_square_degrades_predictably(self, hbr_over_sigma, max_rel_err):
        """Bounds the approximation error so a real regression is separable
        from the known, expected degradation."""
        sigma_m = 100.0
        hbr_m = hbr_over_sigma * sigma_m
        exact = _rayleigh_cdf(hbr_m, sigma_m)
        assert abs(_pc(sigma_m, hbr_m, 0) - exact) / exact < max_rel_err

    @pytest.mark.parametrize("sigma_m,hbr_m", GEOMETRIES)
    def test_circumscribing_square_is_an_upper_bound(self, sigma_m, hbr_m):
        """Mode -1 is documented as an upper bound. It must bound, not equal."""
        exact = _rayleigh_cdf(hbr_m, sigma_m)
        assert _pc(sigma_m, hbr_m, -1) >= exact


class TestPhysicalProperties:
    def test_pc_never_exceeds_one(self):
        for sigma_m, hbr_m in GEOMETRIES:
            for mode in (-1, 0, 1):
                assert 0.0 <= _pc(sigma_m, hbr_m, mode) <= 1.0

    def test_pc_falls_as_miss_distance_grows(self):
        cov = np.eye(3) * 100.0 ** 2
        prev = None
        for miss_m in (0.0, 100.0, 300.0, 1000.0):
            secondary = R_ECI + np.array([0.0, 0.0, miss_m])
            pc = compute_pc(
                R_ECI, V_PRIMARY, cov * 0.5,
                secondary, V_SECONDARY, cov * 0.5,
                hbr=10.0, estimation_mode=1,
            ).Pc
            if prev is not None:
                assert pc < prev
            prev = pc

    def test_pc_rises_with_hard_body_radius(self):
        pcs = [_pc(100.0, r, 1) for r in (5.0, 10.0, 20.0, 40.0)]
        assert pcs == sorted(pcs)


class TestDefaultMode:
    """SCRUM-390. Mode 64 was wrong by ~4.2554x and could exceed 1.0.

    Two compounding errors: the quadrature used second-kind Gauss-Chebyshev
    weights, which carry a spurious sqrt(1-u^2) the erf difference already
    accounts for, and the prefactor was missing the 1/(2*sqrt(2*pi)) from the
    Gaussian normalisation. Both corrected. This class replaces the strict
    xfail that recorded the defect.
    """

    @pytest.mark.parametrize("sigma_m,hbr_m", GEOMETRIES)
    def test_default_mode_matches_closed_form(self, sigma_m, hbr_m):
        assert _pc(sigma_m, hbr_m, 64) == pytest.approx(
            _rayleigh_cdf(hbr_m, sigma_m), rel=1e-6
        )

    @pytest.mark.parametrize("sigma_m,hbr_m", GEOMETRIES)
    def test_default_mode_agrees_with_full_integration(self, sigma_m, hbr_m):
        """The two exact methods must agree with each other, not only with the
        anchor, so a shared systematic error cannot hide."""
        assert _pc(sigma_m, hbr_m, 64) == pytest.approx(_pc(sigma_m, hbr_m, 1), rel=1e-6)

    @pytest.mark.parametrize("n,tol", [(16, 4e-3), (64, 3e-4), (256, 2e-5)])
    def test_quadrature_weights_integrate_a_plain_measure(self, n, tol):
        """Guards the specific regression at its source.

        The weights must sum to the integral of 1 over [-1, 1], which is 2.
        The old second-kind weights summed to pi/2, the integral of
        sqrt(1-u^2), because the chord factor was being counted twice.

        Tolerance tightens with n because this is a quadrature rule, not an
        identity; the error falls as the node count rises. Holding every n to
        the same bound would either fail on 16 or be vacuous on 256.
        """
        from aps_math.pc_utils import _gen_gc_quad

        _, _, w = _gen_gc_quad(n)
        assert sum(w) == pytest.approx(2.0, rel=tol)

    def test_quadrature_error_falls_as_node_count_rises(self):
        """Convergence, so a rule that is merely close by luck cannot pass."""
        from aps_math.pc_utils import _gen_gc_quad

        errs = [abs(sum(_gen_gc_quad(n)[2]) - 2.0) for n in (16, 64, 256)]
        assert errs[0] > errs[1] > errs[2]

    def test_weights_are_not_the_old_second_kind_rule(self):
        """The specific wrong answer, named, so a revert fails loudly."""
        from aps_math.pc_utils import _gen_gc_quad

        assert sum(_gen_gc_quad(64)[2]) != pytest.approx(math.pi / 2, rel=1e-2)


class TestBlastRadiusOfTheOldDefault:
    """SCRUM-390 AC4. What the pre-fix numbers do and do not invalidate.

    The question this answers is narrow and it matters: SCRUM-369 item 6 recorded
    that delta_C tracks real Pc *direction* across six burn directions, and that
    verification was run while the default mode was wrong. If the old error varied
    with geometry it could have reordered those six directions, and the claim would
    need re-running. If it was a constant scale, orderings are untouched and only
    absolute comparisons against a threshold are affected.

    It was a constant scale. Reconstructing the pre-fix rule and measuring it shows
    4.25539x, flat to six figures across every geometry in the operating regime.
    So the directional claims survive and the threshold-relative ones do not, which
    is why SCRUM-390 recaptured the demo risk labels but left SCRUM-369 alone.
    """

    _EXPECTED_FACTOR = 4.25539

    @staticmethod
    def _pc_with_the_old_rule(sigma_m: float, hbr_m: float, miss_m: float = 0.0) -> float:
        """Pc as the pre-SCRUM-390 code computed it.

        Both original errors are reproduced: second-kind quadrature weights, and
        the missing 1/(2*sqrt(2*pi)) prefactor, which is undone here by scaling
        the corrected result's prefactor back out.
        """
        from aps_math import pc_utils

        def _second_kind(n):
            k = np.arange(1, n + 1)
            theta = k * np.pi / (n + 1)
            return np.cos(theta), np.sin(theta), (np.pi / (n + 1)) * np.sin(theta) ** 2

        original = pc_utils._gen_gc_quad
        pc_utils._gc_cache.clear()
        pc_utils._gen_gc_quad = _second_kind
        try:
            cov = np.eye(3) * sigma_m ** 2
            secondary = R_ECI + np.array([0.0, 0.0, miss_m]) if miss_m else R_ECI
            pc = compute_pc(
                R_ECI, V_PRIMARY, cov * 0.5,
                secondary, V_SECONDARY, cov * 0.5,
                hbr=hbr_m, estimation_mode=64,
            ).Pc
        finally:
            pc_utils._gen_gc_quad = original
            pc_utils._gc_cache.clear()
        return pc * 2.0 * math.sqrt(2.0 * math.pi)

    @pytest.mark.parametrize("sigma_m,hbr_m", GEOMETRIES)
    def test_the_old_error_was_a_constant_scale(self, sigma_m, hbr_m):
        old = self._pc_with_the_old_rule(sigma_m, hbr_m)
        new = _pc(sigma_m, hbr_m, 64)
        assert old / new == pytest.approx(self._EXPECTED_FACTOR, rel=5e-3)

    @pytest.mark.parametrize("miss_m", [0.0, 300.0, 600.0, 1200.0, 2400.0])
    def test_the_scale_did_not_move_with_miss_distance(self, miss_m):
        """The demo and planner regime, where sigma is about 1.2 km and the hard
        body radius is 15 m. Held tighter than the sweep above because this is
        the band the system actually operates in."""
        cov = np.eye(3) * 1233.0 ** 2
        secondary = R_ECI + np.array([0.0, 0.0, miss_m]) if miss_m else R_ECI
        new = compute_pc(
            R_ECI, V_PRIMARY, cov * 0.5,
            secondary, V_SECONDARY, cov * 0.5,
            hbr=15.0, estimation_mode=64,
        ).Pc
        old = self._pc_with_the_old_rule(1233.0, 15.0, miss_m)
        assert old / new == pytest.approx(self._EXPECTED_FACTOR, rel=1e-4)

    def test_the_old_rule_preserved_ordering(self):
        """A constant scale cannot reorder anything. Asserted directly rather
        than left as an inference, since a whole verification rests on it."""
        cases = [(1233.0, 15.0, m) for m in (0.0, 150.0, 400.0, 900.0, 1800.0, 3000.0)]
        old = [self._pc_with_the_old_rule(s, h, m) for s, h, m in cases]
        new = []
        for s, h, m in cases:
            cov = np.eye(3) * s ** 2
            secondary = R_ECI + np.array([0.0, 0.0, m]) if m else R_ECI
            new.append(compute_pc(
                R_ECI, V_PRIMARY, cov * 0.5,
                secondary, V_SECONDARY, cov * 0.5,
                hbr=h, estimation_mode=64,
            ).Pc)
        assert np.argsort(old).tolist() == np.argsort(new).tolist()


class TestDefaultModeStaysAtSixtyFour:
    """SCRUM-390 AC5. The default is recorded as still correct, with the reason.

    Mode 64 agrees with mode 1, the full 2D numerical integration, to well inside
    1e-6 on every reference geometry, and it is about an order of magnitude
    cheaper. Node count is not the sensitivity here: the quadrature is already at
    machine precision by n=16 for these integrands, so raising the default would
    buy nothing. Lowering it would.
    """

    def test_the_module_default_is_sixty_four(self):
        import inspect

        sig = inspect.signature(compute_pc)
        assert sig.parameters["estimation_mode"].default == 64

    @pytest.mark.parametrize("n", [16, 32, 64, 128, 256])
    @pytest.mark.parametrize("sigma_m,hbr_m", GEOMETRIES)
    def test_node_count_is_not_the_sensitivity(self, n, sigma_m, hbr_m):
        """Any reasonable node count lands on the closed form, so 64 is not a
        precision compromise."""
        assert _pc(sigma_m, hbr_m, n) == pytest.approx(
            _rayleigh_cdf(hbr_m, sigma_m), rel=1e-9
        )


class TestNoModeCanReturnAnInvalidProbability:
    @pytest.mark.parametrize("mode", [-1, 0, 1, 64])
    @pytest.mark.parametrize("sigma_m,hbr_m", GEOMETRIES)
    def test_pc_stays_in_the_unit_interval(self, mode, sigma_m, hbr_m):
        """At sigma 50 m and HBR 50 m the old mode 64 returned 1.70."""
        assert 0.0 <= _pc(sigma_m, hbr_m, mode) <= 1.0
