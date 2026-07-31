"""Tests for cdm_to_conjunction.cdm_to_conjunction_state."""

import pytest

from cdm_parser import parse_cdm_kvn
from cdm_to_conjunction import cdm_to_conjunction_state

# Re-use the Blue Book fixture from the parser tests.
SAMPLE_KVN = """\
CCSDS_CDM_VERS = 1.0
CREATION_DATE  = 2010-03-12T22:31:12.000
ORIGINATOR     = JSPOC
MESSAGE_ID     = 201113719185
TCA            = 2010-097T04:42:19.315
MISS_DISTANCE  = 715.0 [m]
COLLISION_PROBABILITY = 4.835E-05
OBJECT          = OBJECT1
OBJECT_NAME     = SATELLITE_A
OBJECT_DESIGNATOR = 12345
X               = 2238.400000 [km]
Y               = 1952.900000 [km]
Z               = 6060.700000 [km]
X_DOT           = -4.100000 [km/s]
Y_DOT           = -5.900000 [km/s]
Z_DOT           = 3.300000 [km/s]
CR_R            = 100.0 [m**2]
CT_R            = -10.0 [m**2]
CT_T            = 200.0 [m**2]
CN_R            = 5.0 [m**2]
CN_T            = -3.0 [m**2]
CN_N            = 50.0 [m**2]
OBJECT          = OBJECT2
OBJECT_NAME     = DEBRIS_B
OBJECT_DESIGNATOR = 67890
X               = 2569.540800 [km]
Y               = 2245.100000 [km]
Z               = 6281.600000 [km]
X_DOT           = -2.900000 [km/s]
Y_DOT           = -6.000000 [km/s]
Z_DOT           = 3.300000 [km/s]
CR_R            = 1337.0 [m**2]
CT_R            = -48060.0 [m**2]
CT_T            = 2492000.0 [m**2]
CN_R            = -32.98 [m**2]
CN_T            = -758.88 [m**2]
CN_N            = 71.05 [m**2]
"""


@pytest.fixture()
def state() -> dict:
    cdm = parse_cdm_kvn(SAMPLE_KVN)[0]
    return cdm_to_conjunction_state(cdm)


class TestConjunctionStateSchema:
    """Verify the output matches the planner's ConjunctionState schema."""

    def test_keys_present(self, state: dict) -> None:
        assert set(state.keys()) == {
            "obj_id",
            "t_ca_utc",
            "r_rel_km",
            # SCRUM-398. Without these the planner has no encounter plane, so no
            # Pc, so the Pc gate SCRUM-389 built never runs on a CDM-sourced
            # event and the Mahalanobis screen decides it instead.
            "v_rel_km_s",
            "v_rel_source",
            "p_rel_km2",
            "pc_precomputed",
        }

    def test_obj_id(self, state: dict) -> None:
        assert state["obj_id"] == "67890"

    def test_tca_utc_ends_with_z(self, state: dict) -> None:
        assert state["t_ca_utc"].endswith("Z")

    def test_pc_precomputed(self, state: dict) -> None:
        assert state["pc_precomputed"] == pytest.approx(4.835e-05)


class TestRelativePosition:
    """Verify r_rel_km."""

    def test_is_3_element_list(self, state: dict) -> None:
        assert isinstance(state["r_rel_km"], list)
        assert len(state["r_rel_km"]) == 3

    def test_values_reasonable(self, state: dict) -> None:
        """Relative position should be on the order of the miss distance (~0.7 km)."""
        import numpy as np

        norm = np.linalg.norm(state["r_rel_km"])
        # Miss distance is 715 m ≈ 0.715 km; allow generous bounds.
        assert 0.1 < norm < 500.0


class TestCovariance:
    """Verify p_rel_km2 properties."""

    def test_is_9_element_list(self, state: dict) -> None:
        assert isinstance(state["p_rel_km2"], list)
        assert len(state["p_rel_km2"]) == 9

    def test_symmetric(self, state: dict) -> None:
        """Row-major 3×3 must be symmetric: p[1]≈p[3], p[2]≈p[6], p[5]≈p[7]."""
        p = state["p_rel_km2"]
        assert p[1] == pytest.approx(p[3], abs=1e-12)
        assert p[2] == pytest.approx(p[6], abs=1e-12)
        assert p[5] == pytest.approx(p[7], abs=1e-12)

    def test_values_in_km2_range(self, state: dict) -> None:
        """For LEO objects all covariance elements should be < 10 km²."""
        for val in state["p_rel_km2"]:
            assert abs(val) < 10.0, f"Covariance element {val} km² out of range"

    def test_diagonal_positive(self, state: dict) -> None:
        """Diagonal elements (variances) must be non-negative."""
        p = state["p_rel_km2"]
        assert p[0] >= 0.0
        assert p[4] >= 0.0
        assert p[8] >= 0.0


# ---------------------------------------------------------------------------
# SCRUM-398: relative velocity
# ---------------------------------------------------------------------------

import numpy as np

from cdm_to_conjunction import _rtn_to_eci_rotation


class TestRelativeVelocity:
    """SCRUM-389 gave the planner the ability to compute its own Pc from an
    encounter plane. The encounter plane is perpendicular to the relative
    velocity at TCA, and no producer supplied one, so the Pc gate never ran on
    a real event and the Mahalanobis screen decided everything.

    This producer can supply one. A CCSDS CDM carries both objects' state
    vectors at TCA, and optionally RELATIVE_VELOCITY_R/T/N. Neither needed
    fetching; both were already parsed and discarded.
    """

    def test_it_is_present_and_finite(self, state: dict) -> None:
        v = state["v_rel_km_s"]
        assert v is not None
        assert len(v) == 3
        assert all(np.isfinite(v))

    def test_the_source_is_recorded(self, state: dict) -> None:
        """Whether the originator gave us the figure or we derived it changes
        how much weight it deserves, so the record says which."""
        assert state["v_rel_source"] in (
            "relative_velocity_rtn", "state_vector_difference", "unavailable"
        )

    def test_the_blue_book_sample_derives_from_state_vectors(self, state: dict) -> None:
        """The CCSDS Blue Book sample carries no relative metadata section, so
        the subtraction branch is the one under test here."""
        assert state["v_rel_source"] == "state_vector_difference"

    def test_the_sense_is_secondary_minus_primary(self, state: dict) -> None:
        """The assertion that matters most, and the one a comment cannot make.

        compute_pc_from_geometry builds the secondary as v2 = v1 + v_rel, so
        v_rel must be secondary minus primary. Getting it backwards rotates the
        encounter plane by 180 degrees, which is invisible for a symmetric
        covariance and wrong for every real one.
        """
        v1 = np.array([-4.1, -5.9, 3.3])   # OBJECT1 from SAMPLE_KVN
        v2 = np.array([-2.9, -6.0, 3.3])   # OBJECT2
        assert np.allclose(state["v_rel_km_s"], v2 - v1)
        assert not np.allclose(state["v_rel_km_s"], v1 - v2)

    def test_relative_position_and_relative_velocity_use_the_same_sense(
        self, state: dict
    ) -> None:
        """Both must be object 2 relative to object 1. If one flipped and the
        other did not, the miss and the encounter plane would disagree and Pc
        would be computed on a geometry that does not exist."""
        r1 = np.array([2238.400000, 1952.900000, 6060.700000])
        r2 = np.array([2569.540800, 2245.100000, 6281.600000])
        assert np.allclose(state["r_rel_km"], r2 - r1)

    def test_a_head_on_conjunction_gives_about_twice_orbital_velocity(self) -> None:
        """AC4. Sense and magnitude together, on a geometry whose answer is
        known without reference to the implementation."""
        v_orb = 7.5
        kvn = SAMPLE_KVN.replace(
            "X_DOT           = -2.900000 [km/s]\nY_DOT           = -6.000000 [km/s]\nZ_DOT           = 3.300000 [km/s]",
            f"X_DOT           = 0.0 [km/s]\nY_DOT           = {-v_orb} [km/s]\nZ_DOT           = 0.0 [km/s]",
        ).replace(
            "X_DOT           = -4.100000 [km/s]\nY_DOT           = -5.900000 [km/s]\nZ_DOT           = 3.300000 [km/s]",
            f"X_DOT           = 0.0 [km/s]\nY_DOT           = {v_orb} [km/s]\nZ_DOT           = 0.0 [km/s]",
        )
        s = cdm_to_conjunction_state(parse_cdm_kvn(kvn)[0])
        assert np.linalg.norm(s["v_rel_km_s"]) == pytest.approx(2 * v_orb, rel=1e-9)
        # Opposing the primary's motion, which is what head-on means.
        assert s["v_rel_km_s"][1] == pytest.approx(-2 * v_orb, rel=1e-9)

    def test_a_supplied_relative_velocity_wins_over_the_subtraction(self) -> None:
        """The originator's own figure at TCA beats our difference of two state
        vectors, the same precedence r_rel already uses for RELATIVE_POSITION
        and the same precedence SCRUM-389 gave a supplied Pc."""
        kvn = SAMPLE_KVN.replace(
            "COLLISION_PROBABILITY = 4.835E-05",
            "COLLISION_PROBABILITY = 4.835E-05\n"
            "RELATIVE_VELOCITY_R = 100.0 [m/s]\n"
            "RELATIVE_VELOCITY_T = -200.0 [m/s]\n"
            "RELATIVE_VELOCITY_N = 50.0 [m/s]",
        )
        cdm = parse_cdm_kvn(kvn)[0]
        s = cdm_to_conjunction_state(cdm)

        assert s["v_rel_source"] == "relative_velocity_rtn"

        r1 = np.array([cdm["OBJECT1_X"], cdm["OBJECT1_Y"], cdm["OBJECT1_Z"]])
        v1 = np.array([cdm["OBJECT1_X_DOT"], cdm["OBJECT1_Y_DOT"], cdm["OBJECT1_Z_DOT"]])
        expected = _rtn_to_eci_rotation(r1, v1) @ np.array([100.0, -200.0, 50.0]) / 1000.0
        assert np.allclose(s["v_rel_km_s"], expected)

    def test_the_rtn_branch_is_rotated_not_passed_through(self) -> None:
        """A CDM's relative velocity is RTN and the planner wants ECI. Passing
        the components straight through would be the SCRUM-397 defect again, one
        service upstream."""
        kvn = SAMPLE_KVN.replace(
            "COLLISION_PROBABILITY = 4.835E-05",
            "COLLISION_PROBABILITY = 4.835E-05\n"
            "RELATIVE_VELOCITY_R = 100.0 [m/s]\n"
            "RELATIVE_VELOCITY_T = -200.0 [m/s]\n"
            "RELATIVE_VELOCITY_N = 50.0 [m/s]",
        )
        s = cdm_to_conjunction_state(parse_cdm_kvn(kvn)[0])
        assert not np.allclose(s["v_rel_km_s"], np.array([100.0, -200.0, 50.0]) / 1000.0)

    def test_a_cdm_with_neither_reports_unavailable_rather_than_zero(self) -> None:
        """None means no Pc can be established. Zero would claim the two objects
        are co-moving, which is a statement about the physics rather than about
        what we know."""
        kvn = "\n".join(
            line for line in SAMPLE_KVN.splitlines()
            if not line.startswith(("X_DOT", "Y_DOT", "Z_DOT"))
        )
        # Primary velocity is still needed for the RTN rotation, so put it back
        # on OBJECT1 only.
        kvn = kvn.replace(
            "OBJECT_DESIGNATOR = 12345",
            "OBJECT_DESIGNATOR = 12345\n"
            "X_DOT           = -4.100000 [km/s]\n"
            "Y_DOT           = -5.900000 [km/s]\n"
            "Z_DOT           = 3.300000 [km/s]",
        )
        s = cdm_to_conjunction_state(parse_cdm_kvn(kvn)[0])
        assert s["v_rel_km_s"] is None
        assert s["v_rel_source"] == "unavailable"
