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
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aps_math import frames

from avoid.decision_model import compute_q_exec_km2, cw_phi_rv, mahalanobis_sq
from common.maneuver_scorer import score_maneuver_candidates
from common.operator_policy import OperatorPolicy
from common.satellite_capability import (
    LifetimeProfile,
    PropulsionProfile,
    SatelliteCapability,
)

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


# ---------------------------------------------------------------------------
# AC3: the scorer, on a geometry where the rotation is not the identity
# ---------------------------------------------------------------------------

class TestTheScorerOnAnInclinedOrbit:
    """These are the tests whose absence let the defect through."""

    @staticmethod
    def _expected_r_post(r_sat, v_sat, dv_eci):
        rot = frames.rtn_to_eci_rotation(r_sat, v_sat)
        phi_rtn = cw_phi_rv(_A_KM, DT_S)
        delta_r = rot @ (phi_rtn @ (rot.T @ np.asarray(dv_eci, dtype=float)))
        return R_REL - delta_r

    def _score(self, r_sat, v_sat, cap=None):
        return score_maneuver_candidates(
            "CID-397", r_sat, v_sat, R_REL, P_REL, T_BURN, T_CA,
            cap or _cap(), _policy(),
        )

    def test_every_candidate_matches_the_frame_correct_displacement(self):
        r_sat, v_sat = _inclined_state()
        result = self._score(r_sat, v_sat)

        assert len(result.candidates_v25) == 6
        for c in result.candidates_v25:
            expected = self._expected_r_post(r_sat, v_sat, c.dv_eci_km_s)
            assert c.m2_post == pytest.approx(
                mahalanobis_sq(expected, P_REL), rel=1e-12
            ), f"{c.direction} does not match the frame-correct post-burn position"

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
        "direction, as_coded_km, correct_km",
        [
            ("prograde", 64.678, 87.126),
            ("radial", 14.429, 7.302),
            ("cross-track", 57.035, 0.105),
        ],
    )
    def test_the_displacement_magnitudes_reported_on_the_ticket(
        self, direction, as_coded_km, correct_km
    ):
        """Pins the numbers SCRUM-397 carries, so the ticket and the code cannot
        drift apart. A 2 m/s burn, four hours out, on the reference geometry.

        The cross-track row is the mechanism in one line: the code claimed 57 km
        of separation change from a burn that produces 105 metres.
        """
        r_sat, v_sat = _inclined_state()
        rot = frames.rtn_to_eci_rotation(r_sat, v_sat)
        phi_rtn = cw_phi_rv(_A_KM, DT_S)

        hats = {
            "prograde": _unit(v_sat),
            "radial": _unit(r_sat),
            "cross-track": _unit(np.cross(r_sat, v_sat)),
        }
        dv = hats[direction] * 0.002

        assert np.linalg.norm(phi_rtn @ dv) == pytest.approx(as_coded_km, abs=1e-3)
        assert np.linalg.norm(
            rot @ (phi_rtn @ (rot.T @ dv))
        ) == pytest.approx(correct_km, abs=1e-3)

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
    def test_the_correct_displacement_is_the_same_on_every_orbit(
        self, inc_deg, raan_deg, arglat_deg
    ):
        """The sharpest statement of the defect, and better than pinning three
        numbers.

        A prograde burn of a given size does the same thing to your orbit
        wherever you are. Its RTN components are the same on every circular
        orbit, so the frame-correct displacement has to be identical across all
        of these. The unrotated expression is not, because it depends on where
        the satellite happens to be pointing in ECI, which is physically
        meaningless.

        So the old code gave a different answer for the same maneuver depending
        on the satellite's position in inertial space. There is no reading of
        the physics under which that is right.
        """
        r_sat, v_sat = circular_state(inc_deg, raan_deg, arglat_deg)
        rot = frames.rtn_to_eci_rotation(r_sat, v_sat)
        phi_rtn = cw_phi_rv(_A_KM, DT_S)
        dv = _unit(v_sat) * 0.002

        correct = np.linalg.norm(rot @ (phi_rtn @ (rot.T @ dv)))
        assert correct == pytest.approx(87.126, abs=1e-3)

    def test_the_unrotated_expression_was_not_invariant(self):
        """The companion to the test above. Kept separate so the failure message
        says which property broke."""
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
# AC4: the aligned geometry is untouched
# ---------------------------------------------------------------------------

class TestTheAlignedGeometryIsUnchanged:
    def test_the_rotation_is_the_identity_so_nothing_should_move(self):
        result = score_maneuver_candidates(
            "CID-397-ALIGNED", ALIGNED_R, ALIGNED_V, R_REL, P_REL,
            T_BURN, T_CA, _cap(), _policy(),
        )
        phi_rtn = cw_phi_rv(_A_KM, DT_S)

        for c in result.candidates_v25:
            unrotated = R_REL - phi_rtn @ np.asarray(c.dv_eci_km_s, dtype=float)
            assert c.m2_post == pytest.approx(
                mahalanobis_sq(unrotated, P_REL), rel=1e-12
            ), (
                f"{c.direction} moved on an axis-aligned geometry, where the "
                f"rotation is the identity and nothing should have"
            )

    def test_this_is_the_geometry_every_other_fixture_uses(self):
        """Not a behavioural assertion. It records why the rest of the suite was
        blind to this, so the next person reading these tests understands that a
        green run elsewhere proves nothing about frames."""
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
