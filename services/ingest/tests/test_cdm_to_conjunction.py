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
