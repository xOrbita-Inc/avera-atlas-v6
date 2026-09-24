"""tests/test_leolabs_covariance.py

SCRUM-448: the covariance eigendecomposition behind the globe's uncertainty
ellipsoids.

The decomposition is done server-side in numpy rather than in the browser, so it
is worth testing away from the route. What matters is that the axes really are an
orthonormal right-handed basis, the sigmas really are sqrt(eigenvalue) in metres,
and that anything not usable as a covariance is rejected outright instead of
being coerced into a plausible-looking ellipsoid.

Run from repo root:
    python -m pytest services/planner/tests/test_leolabs_covariance.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.leolabs_runtime import (
    LeoLabsCovarianceError,
    cov_eigen_axes_sigmas,
    object_covariance_block,
)


def _basis(axes):
    """Axes as columns, the way a rotation matrix is built from them."""
    return np.array(axes, dtype=float).T


class TestDiagonalCovariance:
    def test_a_diagonal_covariance_gives_axis_aligned_unit_vectors(self):
        axes, sigmas = cov_eigen_axes_sigmas(np.diag([9.0, 4.0, 1.0]))

        # Largest sigma first, and sigma is sqrt of the variance.
        assert sigmas == pytest.approx([3.0, 2.0, 1.0])
        # Each axis is a unit vector along a coordinate direction.
        for axis, expected in zip(axes, ([1, 0, 0], [0, 1, 0], [0, 0, 1])):
            assert np.allclose(np.abs(axis), expected)
            assert math.isclose(np.linalg.norm(axis), 1.0, rel_tol=1e-9)

    def test_sigmas_are_sorted_largest_first_whatever_the_input_order(self):
        _, sigmas = cov_eigen_axes_sigmas(np.diag([1.0, 100.0, 25.0]))
        assert sigmas == pytest.approx([10.0, 5.0, 1.0])
        assert sigmas == sorted(sigmas, reverse=True)


class TestBasisProperties:
    def test_the_axes_are_orthonormal(self):
        cov = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        axes, _ = cov_eigen_axes_sigmas(cov)
        basis = _basis(axes)
        assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-9)

    def test_the_basis_is_right_handed(self):
        """The renderer builds a rotation from these; det -1 is not a rotation."""
        cov = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        axes, _ = cov_eigen_axes_sigmas(cov)
        assert float(np.linalg.det(_basis(axes))) == pytest.approx(1.0, abs=1e-9)

    def test_the_decomposition_reconstructs_the_input(self):
        """A x = lambda x for each pair, i.e. these really are its eigenpairs."""
        cov = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        axes, sigmas = cov_eigen_axes_sigmas(cov)
        basis = _basis(axes)
        lam = np.diag([s ** 2 for s in sigmas])
        assert np.allclose(basis @ lam @ basis.T, cov, atol=1e-9)


class TestRealisticConjunctionCovariance:
    """The real thing is extremely anisotropic, and that must survive intact."""

    def test_a_highly_elongated_covariance_keeps_its_axis_ratio(self):
        # Close to the fixture CDM's SAT2 block: metres-squared, in-track dominant.
        cov = np.diag([251.0, 1060.0, 1.6099e8])
        axes, sigmas = cov_eigen_axes_sigmas(cov)

        assert sigmas[0] == pytest.approx(math.sqrt(1.6099e8), rel=1e-9)
        assert sigmas[2] == pytest.approx(math.sqrt(251.0), rel=1e-9)
        # ~800:1. If a renderer ever normalises this away, the picture stops
        # saying "the uncertainty is along-track", which is the useful part.
        assert sigmas[0] / sigmas[2] > 500

    def test_the_dominant_axis_points_along_the_dominant_direction(self):
        cov = np.diag([251.0, 1060.0, 1.6099e8])
        axes, _ = cov_eigen_axes_sigmas(cov)
        assert abs(axes[0][2]) == pytest.approx(1.0, abs=1e-9)


class TestRejection:
    @pytest.mark.parametrize("bad,reason", [
        (None, "absent"),
        ([[1.0, 0.0], [0.0, 1.0]], "wrong shape"),
        (np.full((3, 3), np.nan), "non-finite"),
        (np.diag([np.inf, 1.0, 1.0]), "non-finite"),
        (np.array([[1.0, 5.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "asymmetric"),
        (np.diag([-9.0, 1.0, 1.0]), "not PSD"),
    ])
    def test_an_unusable_covariance_is_rejected(self, bad, reason):
        with pytest.raises(LeoLabsCovarianceError):
            cov_eigen_axes_sigmas(bad)

    def test_a_tiny_negative_eigenvalue_is_rounding_and_is_tolerated(self):
        """Clamped to zero, not rejected: a PSD matrix can round just below zero."""
        cov = np.diag([1.0e8, 1.0e4, -1.0e-6])
        axes, sigmas = cov_eigen_axes_sigmas(cov)
        assert sigmas[2] == 0.0
        assert sigmas[0] == pytest.approx(1.0e4)

    def test_a_zero_covariance_is_degenerate_but_not_an_error(self):
        """All-zero is PSD. It yields zero sigmas, which the renderer can skip."""
        _, sigmas = cov_eigen_axes_sigmas(np.zeros((3, 3)))
        assert sigmas == [0.0, 0.0, 0.0]


class TestResponseBlock:
    def test_a_usable_covariance_becomes_a_labelled_block(self):
        block = object_covariance_block(np.diag([9.0, 4.0, 1.0]))
        assert sorted(block) == ["axes", "frame", "sigmas_m", "source"]
        assert block["sigmas_m"] == pytest.approx([3.0, 2.0, 1.0])
        assert block["frame"] == "EME2000"
        # Labelled as the CDM's own covariance so nobody reads it as synthetic.
        assert block["source"] == "leolabs_cdm_covariance"
        assert all(isinstance(c, float) for axis in block["axes"] for c in axis)

    @pytest.mark.parametrize("bad", [
        None,
        np.diag([-9.0, 1.0, 1.0]),
        np.full((3, 3), np.nan),
        [[1.0, 0.0], [0.0, 1.0]],
    ])
    def test_an_unusable_covariance_yields_no_block_rather_than_a_fake_one(self, bad):
        """No ellipsoid is the honest answer; a default one reads as measurement."""
        assert object_covariance_block(bad) is None
