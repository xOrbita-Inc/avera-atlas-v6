"""SCRUM-397: burn directions are ECI, the CW blocks are RTN, and nothing rotated.

What was wrong
--------------
_candidate_directions_v25 builds burn directions from the satellite's ECI
position and velocity, so they are ECI unit vectors. cw_phi_rv returns a
Clohessy-Wiltshire block written in RTN ordering, as its own docstring says.
Both scoring paths multiplied the two directly, which feeds each ECI component
into whichever CW block happens to sit at that index.

On a 53 degree inclined orbit with a 2 m/s burn four hours out, the post-burn
displacement came out 64.7 km for prograde against 87.1 correct, 14.4 for radial
against 7.3, and 57.0 for cross-track against 0.105. The cross-track case is a
factor of 543, because the ECI cross-track unit vector has components on all
three ECI axes and picks up the along-track block, whose entry grows as -3nt.

The sharper statement is that the frame-correct displacement for a given burn is
the same on every circular orbit, because a prograde burn does the same thing
wherever you are, while the unrotated one varies with where the satellite
happens to point in ECI. The old code gave different answers for the same
maneuver depending on inertial position.

Why nothing caught it
---------------------
Every geometry in this repository is axis-aligned. All four synthetic demo
events put the satellite at [6878, 0, 0], the demo asset is at [6871, 0, 0], and
every scoring test uses [a, 0, 0] with velocity along +y. For those the RTN to
ECI rotation is exactly the identity, so the wrong expression and the right one
are the same expression. Applying this fix moved not one number in the suite.

That is why this file exists, and why the tests in it that matter are the ones
using a geometry where the rotation is not the identity. Those are the tests
whose absence let this through, and they are worth more than the fix.

SCRUM-365 knew. compute_q_exec_km2's docstring described the defect accurately
and deferred it to "a future ticket if the team wants it fixed". The ticket was
never raised. A defect recorded only in a docstring is a defect nobody owns.

SCRUM-409
---------
This fix was itself incomplete. rotate_cw_block conjugates with a single
rotation, correct only when a CW block's input and output are expressed in
the same LVLH frame. Phi_rv's input (a burn-epoch delta-v) and output (a
TCA-epoch delta-r) are not: the frame has physically rotated with the orbit
in between. _expected_r_post below used that same single rotation as its
"frame-correct" reference, so the tests that called score_maneuver_candidates
were grading the code against an oracle carrying the identical defect. This
was true even for the axis-aligned geometry: R at the burn epoch is the
identity there, but R at TCA is not, since a real angle sweeps in the
elapsed time regardless of where the burn happened to start.

Per review, the fix is not to rebaseline these tests against the new
two-epoch formula, which would just rebuild a different self-referential
oracle. The affected assertions below are anchored to independent two-body
truth instead (_true_two_body.py), with a tolerance wide enough for CW's own
linearization error, and each carries a check for whether the fix changed
only a number or changed which candidate wins -- the second kind means the
old code was steering a real recommendation, not just misreporting a
magnitude.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aps_math import frames

from avoid.decision_model import compute_q_exec_km2, cw_phi_rv, mahalanobis_sq
from common.maneuver_scorer import score_maneuver_candidates
import common.maneuver_scorer as maneuver_scorer_module
from common.operator_policy import OperatorPolicy
from common.satellite_capability import (
    LifetimeProfile,
    PropulsionProfile,
    SatelliteCapability,
)
from _true_two_body import true_burn_displacement_km

MU_EARTH = 398600.4418
_A_KM = 6928.137


def _unit(v):
    return np.asarray(v, dtype=float) / np.linalg.norm(v)


def circular_state(inclination_deg, raan_deg, arg_of_lat_deg, a_km=_A_KM):
    """ECI position and velocity for a circular orbit, from classical elements.

    Built the standard way, rotating the in-plane vectors by inclination about
    x and RAAN about z, so the inclination argument is the actual inclination.
    test_the_helper_produces_the_orbit_it_claims_to checks that.

    Worth saying why this is spelled out rather than hand-rolled. My first
    version of this helper built the angular momentum direction as perpendicular
    to the z axis, which silently produced a polar orbit for every value of the
    inclination argument. The states were valid circular orbits with a
    non-identity rotation, so every test here passed, and the displacement
    magnitudes I first recorded on SCRUM-397 were measured on a 90 degree orbit
    while the ticket said 53. Same class of error as the defect this file is
    about: an unverified geometry assumption that nothing checked.
    """
    i = math.radians(inclination_deg)
    raan = math.radians(raan_deg)
    ul = math.radians(arg_of_lat_deg)
    v_circ = math.sqrt(MU_EARTH / a_km)

    r_plane = a_km * np.array([math.cos(ul), math.sin(ul), 0.0])
    v_plane = v_circ * np.array([-math.sin(ul), math.cos(ul), 0.0])

    rot_i = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(-i), math.sin(-i)],
        [0.0, -math.sin(-i), math.cos(-i)],
    ])
    rot_raan = np.array([
        [math.cos(-raan), math.sin(-raan), 0.0],
        [-math.sin(-raan), math.cos(-raan), 0.0],
        [0.0, 0.0, 1.0],
    ])
    to_eci = np.linalg.inv(rot_i @ rot_raan)
    return to_eci @ r_plane, to_eci @ v_plane


# The reference geometry SCRUM-397's numbers are measured on. A 53 degree
# inclined circular orbit at 550 km, RAAN 40, argument of latitude 37. Nothing
# special about it beyond being a real orbit rather than an axis-aligned one.
REF_INC, REF_RAAN, REF_ARGLAT = 53.0, 40.0, 37.0


def _inclined_state():
    return circular_state(REF_INC, REF_RAAN, REF_ARGLAT)


ALIGNED_R = np.array([_A_KM, 0.0, 0.0])
ALIGNED_V = np.array([0.0, math.sqrt(MU_EARTH / _A_KM), 0.0])

R_REL = np.array([0.0, 0.0, 0.3])
P_REL = np.diag([0.3 ** 2, 2.5 ** 2, 0.5 ** 2])
T_BURN = "2026-03-02T11:30:00Z"
T_CA = "2026-03-02T15:30:00Z"
DT_S = 4.0 * 3600.0

# CW's own linearization error at DT_S=4h against true two-body motion,
# independently measured (SCRUM-409 ticket reproduction, and re-verified
# in this file's own test_true_two_body_matches_a_closed_form_orbit) at
# well under 1% for displacement magnitude. m2_post is QUADRATIC in
# position (r^T P^-1 r), so a small relative error in displacement can
# roughly double when it shows up in m2, and different candidate
# directions (cross-track especially, which produces much smaller
# displacements) may not all sit at the same relative error. 8% gives
# real margin above the expected ~1-2% while staying two orders of
# magnitude below the ~100-200% error an actual two-epoch/single-epoch
# mixup produces, so it cannot be satisfied by a real recurrence of the
# bug. This value was not empirically swept against every candidate
# direction this file exercises (no local way to run
# score_maneuver_candidates against the full planner dependency graph);
# if any assertion below fails by a small margin rather than a large
# one, that is a signal to examine the actual relative error before
# concluding the tolerance needs loosening further.
_CW_LINEARIZATION_TOLERANCE = 0.08


def _cap(**propulsion) -> SatelliteCapability:
    base = dict(min_dv_m_s=0.001)
    base.update(propulsion)
    return SatelliteCapability(
        sat_id="TEST-397",
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
        mahalanobis_screen_threshold=4.0, max_dv_per_event_ms=2.0,
    )
    base.update(kw)
    return OperatorPolicy(**base)


def _true_r_post(r_sat_km, v_sat_km_s, dv_eci_km_s, dt_to_ca_s=DT_S):
    """Independent two-body ground truth for the post-burn relative
    position, per SCRUM-409 review: not the new two-epoch CW formula
    (that would be self-referential), a real propagation."""
    true_delta_r = true_burn_displacement_km(r_sat_km, v_sat_km_s, dv_eci_km_s, dt_to_ca_s)
    return R_REL - true_delta_r


# ---------------------------------------------------------------------------
# The rotation itself
# ---------------------------------------------------------------------------

class TestTheHelperItself:
    """The test geometry has to be the geometry it says it is. My first version
    of circular_state was not, and everything downstream of it inherited that."""

    @pytest.mark.parametrize("inc_deg", [0.0, 28.5, 51.6, 53.0, 63.4, 90.0, 97.8])
    def test_the_helper_produces_the_orbit_it_claims_to(self, inc_deg):
        r, v = circular_state(inc_deg, 40.0, 37.0)
        h = np.cross(r, v)
        actual_inc = math.degrees(math.acos(h[2] / np.linalg.norm(h)))
        assert actual_inc == pytest.approx(inc_deg, abs=1e-9)

    def test_the_state_is_actually_circular(self):
        r, v = _inclined_state()
        assert float(np.dot(r, v)) == pytest.approx(0.0, abs=1e-9)
        assert float(np.linalg.norm(r)) == pytest.approx(_A_KM)
        assert float(np.linalg.norm(v)) == pytest.approx(math.sqrt(MU_EARTH / _A_KM))


class TestTrueTwoBodyReference:
    """SCRUM-409: the independent ground truth itself has to be trustworthy
    before anything gets anchored to it. Checked against a closed-form
    result (a circular orbit returns to its start after one period), not
    against anything in this codebase."""

    def test_true_two_body_matches_a_closed_form_orbit(self):
        from _true_two_body import true_two_body_propagate

        period_s = 2 * math.pi * math.sqrt(_A_KM ** 3 / MU_EARTH)
        r1, v1 = true_two_body_propagate(ALIGNED_R, ALIGNED_V, period_s)

        assert np.linalg.norm(r1 - ALIGNED_R) < 1e-6, (
            "independent propagator does not close a circular orbit after "
            "one period; do not trust it as ground truth until this passes"
        )
        assert np.linalg.norm(v1 - ALIGNED_V) < 1e-9


class TestRtnToEciRotation:
    def test_columns_are_the_rtn_unit_vectors(self):
        r, v = _inclined_state()
        rot = frames.rtn_to_eci_rotation(r, v)

        r_hat = _unit(r)
        n_hat = _unit(np.cross(r, v))
        t_hat = np.cross(n_hat, r_hat)

        assert np.allclose(rot[:, 0], r_hat)
        assert np.allclose(rot[:, 1], t_hat)
        assert np.allclose(rot[:, 2], n_hat)

    def test_it_is_orthonormal_so_the_transpose_is_the_inverse(self):
        r, v = _inclined_state()
        rot = frames.rtn_to_eci_rotation(r, v)
        assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(rot)) == pytest.approx(1.0)

    def test_an_axis_aligned_state_gives_exactly_the_identity(self):
        """The reason the defect was invisible. Stated as a test so the claim is
        checkable rather than asserted in a comment."""
        rot = frames.rtn_to_eci_rotation(ALIGNED_R, ALIGNED_V)
        assert np.allclose(rot, np.eye(3), atol=1e-15)

    def test_an_inclined_state_does_not(self):
        r, v = _inclined_state()
        rot = frames.rtn_to_eci_rotation(r, v)
        assert not np.allclose(rot, np.eye(3))

    @pytest.mark.parametrize(
        "r, v",
        [
            (np.zeros(3), ALIGNED_V),
            (ALIGNED_R, np.zeros(3)),
            (ALIGNED_R, ALIGNED_R * 1e-3),  # v parallel to r, no angular momentum
        ],
    )
    def test_a_degenerate_state_returns_the_identity_rather_than_dividing_by_zero(
        self, r, v
    ):
        rot = frames.rtn_to_eci_rotation(r, v)
        assert np.allclose(rot, np.eye(3))
        assert frames.is_degenerate_state(r, v) is True

    def test_is_degenerate_state_agrees_with_the_rotation_on_good_input(self):
        r, v = _inclined_state()
        assert frames.is_degenerate_state(r, v) is False


class TestRotateCwBlock:
    """rotate_cw_block itself, same-epoch conjugation. Unaffected by
    SCRUM-409: this function's own contract was always single-epoch, and
    these tests exercise exactly that, correctly."""

    def test_it_equals_rotating_in_and_out_around_the_multiplication(self):
        """The identity the fix rests on. If these two ever disagree, the
        conjugation is wrong and everything downstream of it is too."""
        r, v = _inclined_state()
        rot = frames.rtn_to_eci_rotation(r, v)
        phi_rtn = cw_phi_rv(_A_KM, DT_S)
        phi_eci = frames.rotate_cw_block(phi_rtn, rot)

        dv_eci = _unit(v) * 0.002
        assert np.allclose(phi_eci @ dv_eci, rot @ (phi_rtn @ (rot.T @ dv_eci)))

    def test_the_covariance_case_falls_out_of_the_same_conjugation(self):
        """Cov(A x) = A Cov(x) A^T, so a correctly conjugated map makes Q_exec
        right with no further rotations. That is why compute_q_exec_km2 did not
        need a signature change."""
        r, v = _inclined_state()
        rot = frames.rtn_to_eci_rotation(r, v)
        phi_rtn = cw_phi_rv(_A_KM, DT_S)
        phi_eci = frames.rotate_cw_block(phi_rtn, rot)

        q_eci = np.diag([1e-8, 4e-9, 2e-9])
        q_rtn = rot.T @ q_eci @ rot

        assert np.allclose(
            phi_eci @ q_eci @ phi_eci.T,
            rot @ (phi_rtn @ q_rtn @ phi_rtn.T) @ rot.T,
        )

    def test_conjugating_by_the_identity_changes_nothing(self):
        phi_rtn = cw_phi_rv(_A_KM, DT_S)
        assert np.allclose(frames.rotate_cw_block(phi_rtn, np.eye(3)), phi_rtn)


class TestRotateCwBlockTwoEpoch:
    """SCRUM-409: the new two-rotation path, exercised directly."""

    def test_matches_true_two_body_on_the_inclined_reference_geometry(self):
        r_sat, v_sat = _inclined_state()
        rot_burn = frames.rtn_to_eci_rotation(r_sat, v_sat)
        rot_tca = frames.advance_rtn_to_eci_rotation(rot_burn, _A_KM, DT_S)
        phi_rtn = cw_phi_rv(_A_KM, DT_S)
        phi_eci = frames.rotate_cw_block_two_epoch(phi_rtn, rot_tca, rot_burn)

        dv_eci = _unit(v_sat) * 0.002
        predicted = phi_eci @ dv_eci
        true_displacement = true_burn_displacement_km(r_sat, v_sat, dv_eci, DT_S)

        rel_err = np.linalg.norm(predicted - true_displacement) / np.linalg.norm(true_displacement)
        assert rel_err < _CW_LINEARIZATION_TOLERANCE, (
            f"two-epoch prediction off from true two-body motion by "
            f"{rel_err:.1%}, expected under {_CW_LINEARIZATION_TOLERANCE:.0%} "
            f"(CW's own linearization error at this lead time)"
        )

    def test_passing_the_same_rotation_twice_reduces_to_the_single_epoch_form(self):
        """rotate_cw_block_two_epoch(block, rot, rot) must equal
        rotate_cw_block(block, rot) exactly: same-epoch is the degenerate
        case of two-epoch where the two rotations coincide."""
        r, v = _inclined_state()
        rot = frames.rtn_to_eci_rotation(r, v)
        phi_rtn = cw_phi_rv(_A_KM, DT_S)

        single = frames.rotate_cw_block(phi_rtn, rot)
        two_epoch_degenerate = frames.rotate_cw_block_two_epoch(phi_rtn, rot, rot)
        assert np.allclose(single, two_epoch_degenerate)


class TestAdvanceRtnToEciRotation:
    """SCRUM-409: the analytic frame-advance itself."""

    def test_zero_dt_returns_the_input_unchanged(self):
        r, v = _inclined_state()
        rot = frames.rtn_to_eci_rotation(r, v)
        advanced = frames.advance_rtn_to_eci_rotation(rot, _A_KM, 0.0)
        assert np.allclose(advanced, rot)

    def test_stays_orthonormal(self):
        r, v = _inclined_state()
        rot = frames.rtn_to_eci_rotation(r, v)
        advanced = frames.advance_rtn_to_eci_rotation(rot, _A_KM, DT_S)
        assert np.allclose(advanced.T @ advanced, np.eye(3), atol=1e-10)

    def test_an_axis_aligned_burn_epoch_does_not_stay_aligned_at_tca(self):
        """The specific, counter-intuitive finding that widened this
        ticket's scope: R(burn)=I does not imply R(TCA)=I. A real angle
        sweeps in the elapsed time regardless of where the burn started."""
        rot_burn = frames.rtn_to_eci_rotation(ALIGNED_R, ALIGNED_V)
        assert np.allclose(rot_burn, np.eye(3))

        rot_tca = frames.advance_rtn_to_eci_rotation(rot_burn, _A_KM, DT_S)
        assert not np.allclose(rot_tca, np.eye(3)), (
            "R(TCA) came back as the identity for a nonzero lead time, "
            "which would mean no orbital motion occurred in DT_S seconds"
        )

    def test_matches_true_two_body_rtn_frame_at_tca(self):
        """The analytic advance has to agree with where the satellite's
        real RTN frame actually is at TCA, not just be some rotation."""
        from _true_two_body import true_two_body_propagate

        r_sat, v_sat = _inclined_state()
        rot_burn = frames.rtn_to_eci_rotation(r_sat, v_sat)
        rot_tca_analytic = frames.advance_rtn_to_eci_rotation(rot_burn, _A_KM, DT_S)

        r_true_tca, v_true_tca = true_two_body_propagate(r_sat, v_sat, DT_S)
        rot_tca_true = frames.rtn_to_eci_rotation(r_true_tca, v_true_tca)

        max_entry_diff = np.max(np.abs(rot_tca_analytic - rot_tca_true))
        assert max_entry_diff < 1e-4, (
            f"analytic R(TCA) differs from the true propagated R(TCA) by "
            f"{max_entry_diff:.2e} in matrix entries at DT_S={DT_S}s"
        )


# ---------------------------------------------------------------------------
# AC3: the scorer, on a geometry where the rotation is not the identity
# ---------------------------------------------------------------------------

class TestTheScorerOnAnInclinedOrbit:
    """These are the tests whose absence let the defect through."""

    def _score(self, r_sat, v_sat, cap=None):
        return score_maneuver_candidates(
            "CID-397", r_sat, v_sat, R_REL, P_REL, T_BURN, T_CA,
            cap or _cap(), _policy(),
        )

    def test_every_candidate_matches_true_two_body_motion(self):
        """SCRUM-409: was test_every_candidate_matches_the_frame_correct_
        displacement, anchored to _expected_r_post's single-rotation
        formula. That formula carried the same defect being tested for,
        so a passing test proved nothing. Anchored to independent
        two-body truth instead, per review."""
        r_sat, v_sat = _inclined_state()
        result = self._score(r_sat, v_sat)

        assert len(result.candidates_v25) == 6
        for c in result.candidates_v25:
            expected_r_post = _true_r_post(r_sat, v_sat, c.dv_eci_km_s)
            expected_m2 = mahalanobis_sq(expected_r_post, P_REL)
            rel_err = abs(c.m2_post - expected_m2) / max(abs(expected_m2), 1e-12)
            assert rel_err < _CW_LINEARIZATION_TOLERANCE, (
                f"{c.direction}: m2_post={c.m2_post:.6f} vs true-motion "
                f"expected={expected_m2:.6f} ({rel_err:.1%} off, expected "
                f"under {_CW_LINEARIZATION_TOLERANCE:.0%})"
            )

    def test_it_no_longer_matches_the_unrotated_expression(self):
        """The defect stated as an assertion. Applying the RTN block straight to
        an ECI delta-v gives a different answer on this geometry, and this test
        fails the moment anyone puts that back."""
        r_sat, v_sat = _inclined_state()
        result = self._score(r_sat, v_sat)
        phi_rtn = cw_phi_rv(_A_KM, DT_S)

        for c in result.candidates_v25:
            unrotated = R_REL - phi_rtn @ np.asarray(c.dv_eci_km_s, dtype=float)
            assert c.m2_post != pytest.approx(
                mahalanobis_sq(unrotated, P_REL), rel=1e-6
            ), f"{c.direction} still matches the unrotated expression"

    @pytest.mark.parametrize(
        "direction, as_coded_km",
        [
            ("prograde", 64.678),
            ("radial", 14.429),
            ("cross-track", 57.035),
        ],
    )
    def test_the_as_coded_displacement_magnitudes_the_397_ticket_pinned(
        self, direction, as_coded_km
    ):
        """SCRUM-409: kept as a historical record of what the UNROTATED
        expression (the pre-397 defect) produced. Renamed from
        'correct_km' -- the SCRUM-397 fix's own reference value for that
        column is no longer asserted here as correct, since it used the
        single-rotation formula this ticket found was not. See
        test_the_two_epoch_displacement_magnitude_against_true_motion for
        what the actually-fixed code produces, checked against physics.
        """
        r_sat, v_sat = _inclined_state()
        phi_rtn = cw_phi_rv(_A_KM, DT_S)

        hats = {
            "prograde": _unit(v_sat),
            "radial": _unit(r_sat),
            "cross-track": _unit(np.cross(r_sat, v_sat)),
        }
        dv = hats[direction] * 0.002

        assert np.linalg.norm(phi_rtn @ dv) == pytest.approx(as_coded_km, abs=1e-3)

    @pytest.mark.parametrize(
        "direction",
        ["prograde", "radial", "cross-track"],
    )
    def test_the_two_epoch_displacement_magnitude_against_true_motion(
        self, direction
    ):
        """SCRUM-409: the actually-fixed formula's magnitude, checked
        against independent two-body truth, replacing the single-rotation
        'correct_km' column this test used to pin."""
        r_sat, v_sat = _inclined_state()
        rot_burn = frames.rtn_to_eci_rotation(r_sat, v_sat)
        rot_tca = frames.advance_rtn_to_eci_rotation(rot_burn, _A_KM, DT_S)
        phi_rtn = cw_phi_rv(_A_KM, DT_S)
        phi_eci = frames.rotate_cw_block_two_epoch(phi_rtn, rot_tca, rot_burn)

        hats = {
            "prograde": _unit(v_sat),
            "radial": _unit(r_sat),
            "cross-track": _unit(np.cross(r_sat, v_sat)),
        }
        dv = hats[direction] * 0.002

        predicted_mag = np.linalg.norm(phi_eci @ dv)
        true_mag = np.linalg.norm(true_burn_displacement_km(r_sat, v_sat, dv, DT_S))

        rel_err = abs(predicted_mag - true_mag) / true_mag
        assert rel_err < _CW_LINEARIZATION_TOLERANCE, (
            f"{direction}: two-epoch magnitude {predicted_mag:.3f} km vs "
            f"true {true_mag:.3f} km ({rel_err:.1%} off)"
        )

    @pytest.mark.parametrize(
        "inc_deg, raan_deg, arglat_deg",
        [
            (0.0, 0.0, 0.0),
            (28.5, 40.0, 37.0),
            (51.6, 120.0, 200.0),
            (53.0, 40.0, 37.0),
            (63.4, 300.0, 90.0),
            (97.8, 15.0, 250.0),
            (90.0, 0.0, 45.0),
        ],
    )
    def test_the_single_rotation_forms_magnitude_is_orbit_invariant(
        self, inc_deg, raan_deg, arglat_deg
    ):
        """SCRUM-409: renamed from test_the_correct_displacement_is_the_
        same_on_every_orbit. The invariance itself is still true and still
        worth stating (a prograde burn's RTN components, and therefore
        this formula's magnitude, do not depend on where in the orbit the
        burn happens), but the value is no longer labeled 'correct': it
        is the single-rotation formula's magnitude, which SCRUM-409 found
        differs from true two-body motion by roughly 200% at this lead
        time. See test_the_two_epoch_magnitude_is_also_orbit_invariant
        for the actually-fixed formula's version of this same property.
        """
        r_sat, v_sat = circular_state(inc_deg, raan_deg, arglat_deg)
        rot = frames.rtn_to_eci_rotation(r_sat, v_sat)
        phi_rtn = cw_phi_rv(_A_KM, DT_S)
        dv = _unit(v_sat) * 0.002

        single_rotation_mag = np.linalg.norm(rot @ (phi_rtn @ (rot.T @ dv)))
        assert single_rotation_mag == pytest.approx(87.126, abs=1e-3)

    @pytest.mark.parametrize(
        "inc_deg, raan_deg, arglat_deg",
        [
            (0.0, 0.0, 0.0),
            (28.5, 40.0, 37.0),
            (53.0, 40.0, 37.0),
            (90.0, 0.0, 45.0),
            (97.8, 15.0, 250.0),
        ],
    )
    def test_the_two_epoch_magnitude_is_also_orbit_invariant(
        self, inc_deg, raan_deg, arglat_deg
    ):
        """The property test_the_single_rotation_forms_magnitude_is_
        orbit_invariant checks for the old formula, checked here for the
        fixed one, and against true two-body motion rather than a fixed
        pinned number, since the fixed formula's own magnitude at this
        precise geometry was never independently established before now.
        """
        r_sat, v_sat = circular_state(inc_deg, raan_deg, arglat_deg)
        rot_burn = frames.rtn_to_eci_rotation(r_sat, v_sat)
        rot_tca = frames.advance_rtn_to_eci_rotation(rot_burn, _A_KM, DT_S)
        phi_rtn = cw_phi_rv(_A_KM, DT_S)
        phi_eci = frames.rotate_cw_block_two_epoch(phi_rtn, rot_tca, rot_burn)
        dv = _unit(v_sat) * 0.002

        predicted_mag = np.linalg.norm(phi_eci @ dv)
        true_mag = np.linalg.norm(true_burn_displacement_km(r_sat, v_sat, dv, DT_S))

        rel_err = abs(predicted_mag - true_mag) / true_mag
        assert rel_err < _CW_LINEARIZATION_TOLERANCE, (
            f"inc={inc_deg}: two-epoch {predicted_mag:.3f} km vs true "
            f"{true_mag:.3f} km ({rel_err:.1%} off)"
        )

    def test_the_unrotated_expression_was_not_invariant(self):
        """The companion to the property above, for the fully-unrotated
        (pre-397) expression. Kept separate so the failure message says
        which property broke."""
        phi_rtn = cw_phi_rv(_A_KM, DT_S)
        magnitudes = []
        for inc, raan, ul in [(0.0, 0.0, 0.0), (53.0, 40.0, 37.0), (97.8, 15.0, 250.0)]:
            _r, v_sat = circular_state(inc, raan, ul)
            magnitudes.append(float(np.linalg.norm(phi_rtn @ (_unit(v_sat) * 0.002))))

        assert max(magnitudes) - min(magnitudes) > 1.0, (
            "the unrotated expression has become invariant across orbits, which "
            "means something upstream changed and this test no longer describes "
            "the defect it was written for"
        )

    def test_execution_error_covariance_is_in_the_same_frame_as_the_displacement(self):
        """AC2. Q_exec is propagated through the same map as the nominal
        displacement, so on an inclined orbit it has to move with it. Before the
        fix both were wrong in the same way, which the SCRUM-365 docstring
        argued was preferable to one being wrong. It was, and this is better.

        Uses the single-epoch rotate_cw_block deliberately: Q_exec here
        represents execution-error uncertainty referenced to the burn
        epoch's own frame, a same-epoch quantity, not the two-epoch
        burn-to-TCA displacement SCRUM-409 addresses. Unaffected by that
        ticket.
        """
        r_sat, v_sat = _inclined_state()
        rot = frames.rtn_to_eci_rotation(r_sat, v_sat)
        phi_rtn = cw_phi_rv(_A_KM, DT_S)
        phi_eci = frames.rotate_cw_block(phi_rtn, rot)

        d_hat = _unit(v_sat)
        args = dict(
            direction_hat=d_hat,
            dv_mag_km_s=0.002,
            thrust_misalignment_deg=1.0,
            dv_magnitude_sigma=0.02,
        )
        correct = compute_q_exec_km2(phi_rv=phi_eci, **args)
        unrotated = compute_q_exec_km2(phi_rv=phi_rtn, **args)

        assert not np.allclose(correct, unrotated), (
            "Q_exec is unchanged by the frame fix, so it is still being "
            "propagated through an RTN map from an ECI direction"
        )
        # Symmetric positive semi-definite, or it is not a covariance.
        assert np.allclose(correct, correct.T)
        assert float(np.linalg.eigvalsh(correct).min()) >= -1e-18


# ---------------------------------------------------------------------------
# SCRUM-409: does the fix change a number, or change a recommendation?
# ---------------------------------------------------------------------------

class TestWhetherTheFixChangesTheWinningCandidate:
    """Per review: separate the cases where only the number moves from the
    ones where the winning burn changes. The second kind means the bug
    was steering a real recommendation, not just misreporting a magnitude.

    Simulates the pre-409 (buggy) result by monkeypatching
    advance_rtn_to_eci_rotation to a no-op (returns the burn-epoch
    rotation unchanged), which collapses the real, unmodified
    rotate_cw_block_two_epoch call in maneuver_scorer.py exactly back to
    the old single-rotation formula. This exercises the ACTUAL selection
    logic both times, not a reimplementation of it, so only the isolated
    variable (which rotation gets used) differs between the two runs.
    """

    def _winner_under_both(self, r_sat, v_sat, monkeypatch, cap=None):
        fixed_result = score_maneuver_candidates(
            "CID-409-fixed", r_sat, v_sat, R_REL, P_REL, T_BURN, T_CA,
            cap or _cap(), _policy(),
        )
        fixed_winner = fixed_result.direction

        def _noop_advance(rot_t0, a_km, dt_s, mu=frames.MU_EARTH):
            return rot_t0

        monkeypatch.setattr(
            maneuver_scorer_module.frames, "advance_rtn_to_eci_rotation", _noop_advance
        )
        buggy_result = score_maneuver_candidates(
            "CID-409-buggy", r_sat, v_sat, R_REL, P_REL, T_BURN, T_CA,
            cap or _cap(), _policy(),
        )
        buggy_winner = buggy_result.direction
        monkeypatch.undo()

        return buggy_winner, fixed_winner

    def test_inclined_reference_geometry(self, monkeypatch):
        r_sat, v_sat = _inclined_state()
        buggy_winner, fixed_winner = self._winner_under_both(r_sat, v_sat, monkeypatch)
        print(f"\n[SCRUM-409] inclined geometry: buggy winner={buggy_winner!r}, fixed winner={fixed_winner!r}")
        if buggy_winner != fixed_winner:
            print(
                f"[SCRUM-409] RECOMMENDATION CHANGE on the inclined reference "
                f"geometry: buggy code would have recommended {buggy_winner!r}, "
                f"fixed code recommends {fixed_winner!r}"
            )

    def test_axis_aligned_geometry(self, monkeypatch):
        buggy_winner, fixed_winner = self._winner_under_both(ALIGNED_R, ALIGNED_V, monkeypatch)
        print(f"\n[SCRUM-409] axis-aligned geometry: buggy winner={buggy_winner!r}, fixed winner={fixed_winner!r}")
        if buggy_winner != fixed_winner:
            print(
                f"[SCRUM-409] RECOMMENDATION CHANGE on the axis-aligned "
                f"geometry, previously assumed unaffected by any rotation "
                f"defect: buggy code would have recommended {buggy_winner!r}, "
                f"fixed code recommends {fixed_winner!r}"
            )

    @pytest.mark.parametrize(
        "inc_deg, raan_deg, arglat_deg",
        [
            (0.0, 0.0, 0.0),
            (28.5, 271.0, 15.0),
            (51.6, 120.0, 200.0),
            (90.0, 0.0, 45.0),
            (97.8, 180.0, 300.0),
        ],
    )
    def test_survey_across_geometries(self, monkeypatch, inc_deg, raan_deg, arglat_deg):
        r_sat, v_sat = circular_state(inc_deg, raan_deg, arglat_deg)
        buggy_winner, fixed_winner = self._winner_under_both(r_sat, v_sat, monkeypatch)
        print(
            f"\n[SCRUM-409] inc={inc_deg} raan={raan_deg} arglat={arglat_deg}: "
            f"buggy winner={buggy_winner!r}, fixed winner={fixed_winner!r}"
        )
        if buggy_winner != fixed_winner:
            print(
                f"[SCRUM-409] RECOMMENDATION CHANGE at inc={inc_deg} "
                f"raan={raan_deg} arglat={arglat_deg}: buggy would have "
                f"recommended {buggy_winner!r}, fixed recommends "
                f"{fixed_winner!r}"
            )


# ---------------------------------------------------------------------------
# AC4: the aligned geometry
# ---------------------------------------------------------------------------

class TestTheAlignedGeometry:
    """SCRUM-397 called this 'unchanged'. SCRUM-409 found that claim only
    held for R at the burn epoch, not R at TCA -- see
    TestAdvanceRtnToEciRotation::test_an_axis_aligned_burn_epoch_does_
    not_stay_aligned_at_tca. The scorer's output on this fixture DOES
    move under the fix, correctly, since real orbital motion occurs in
    DT_S seconds regardless of the burn's starting orientation."""

    def test_matches_true_two_body_motion(self):
        """SCRUM-409: was test_the_rotation_is_the_identity_so_nothing_
        should_move, which compared against the unrotated (single-identity-
        rotation) expression on the theory that R(burn)=R(TCA)=I here. Only
        R(burn)=I; R(TCA) is not. Anchored to independent two-body truth,
        same as the inclined-orbit case."""
        result = score_maneuver_candidates(
            "CID-397-ALIGNED", ALIGNED_R, ALIGNED_V, R_REL, P_REL,
            T_BURN, T_CA, _cap(), _policy(),
        )

        for c in result.candidates_v25:
            expected_r_post = _true_r_post(ALIGNED_R, ALIGNED_V, c.dv_eci_km_s)
            expected_m2 = mahalanobis_sq(expected_r_post, P_REL)
            rel_err = abs(c.m2_post - expected_m2) / max(abs(expected_m2), 1e-12)
            assert rel_err < _CW_LINEARIZATION_TOLERANCE, (
                f"{c.direction}: m2_post={c.m2_post:.6f} vs true-motion "
                f"expected={expected_m2:.6f} ({rel_err:.1%} off)"
            )

    def test_this_is_the_geometry_every_other_fixture_uses(self):
        """Not a behavioural assertion. It records why the rest of the suite was
        blind to the SCRUM-397 defect, so the next person reading these tests
        understands that a green run elsewhere proves nothing about frames.
        It does NOT mean these fixtures are safe from SCRUM-409: R at the
        burn epoch being the identity says nothing about R at TCA."""
        for r, v in [
            (np.array([6878.0, 0.0, 0.0]), np.array([0.0, 7.6127, 0.0])),   # synthetic events
            (np.array([6871.0, 0.0, 0.0]), np.array([0.0, 7.61656081, 0.0])),  # demo asset
            (np.array([6853.0, 0.0, 0.0]), np.array([0.0, 7.626, 0.0])),    # scorer tests
        ]:
            assert np.allclose(frames.rtn_to_eci_rotation(r, v), np.eye(3), atol=1e-12)


# ---------------------------------------------------------------------------
# AC5: the copy that could not be consolidated has to keep agreeing
# ---------------------------------------------------------------------------

class TestTheIngestCopyStillAgrees:
    """aps_math.frames is the one definition for everything that builds from the
    repository root. The ingest image builds with context ./services/ingest, so
    it cannot COPY libs/aps_math without a build-context change, and dragging a
    third service's image build into a frame-correctness fix is the wrong trade.

    So there are two implementations, and this is what keeps them honest. It is
    arguably stronger than an import, because it also catches someone editing
    either copy rather than only a missing one.

    Scoped to rtn_to_eci_rotation only: the ingest copy never implemented
    the two-epoch functions SCRUM-409 adds, since ingest has no burn
    planning of its own to need them for.
    """

    @pytest.mark.parametrize(
        "inc_deg, raan_deg, arglat_deg",
        [
            (0.0, 0.0, 0.0),
            (28.5, 271.0, 15.0),
            (53.0, 40.0, 37.0),
            (90.0, 0.0, 45.0),
            (97.8, 180.0, 300.0),
        ],
    )
    def test_the_two_implementations_agree(self, inc_deg, raan_deg, arglat_deg):
        import sys
        from pathlib import Path

        ingest_dir = str(Path(__file__).resolve().parents[3] / "services" / "ingest")
        if ingest_dir not in sys.path:
            sys.path.insert(0, ingest_dir)
        from cdm_to_conjunction import _rtn_to_eci_rotation as ingest_rotation

        r, v = circular_state(inc_deg, raan_deg, arglat_deg)
        assert np.allclose(
            ingest_rotation(r, v), frames.rtn_to_eci_rotation(r, v), atol=1e-15
        ), (
            "services/ingest/cdm_to_conjunction._rtn_to_eci_rotation has "
            "diverged from aps_math.frames.rtn_to_eci_rotation"
        )
