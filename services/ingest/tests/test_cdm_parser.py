"""Tests for cdm_parser.parse_cdm_kvn."""

import pytest

from cdm_parser import parse_cdm_kvn

# ---------------------------------------------------------------------------
# CCSDS 508.0-B-1 Blue Book example (trimmed to relevant fields)
# ---------------------------------------------------------------------------
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


class TestParseCdmKvn:
    """Verify KVN parsing against the CCSDS Blue Book example."""

    @pytest.fixture()
    def cdm(self) -> dict:
        blocks = parse_cdm_kvn(SAMPLE_KVN)
        assert len(blocks) == 1
        return blocks[0]

    def test_tca_parsed(self, cdm: dict) -> None:
        assert cdm["TCA"] == "2010-097T04:42:19.315"

    def test_collision_probability(self, cdm: dict) -> None:
        assert cdm["COLLISION_PROBABILITY"] == pytest.approx(4.835e-05)

    def test_miss_distance(self, cdm: dict) -> None:
        assert cdm["MISS_DISTANCE"] == pytest.approx(715.0)

    def test_object1_state(self, cdm: dict) -> None:
        assert cdm["OBJECT1_X"] == pytest.approx(2238.4)
        assert cdm["OBJECT1_Y"] == pytest.approx(1952.9)
        assert cdm["OBJECT1_Z"] == pytest.approx(6060.7)

    def test_object2_covariance(self, cdm: dict) -> None:
        assert cdm["OBJECT2_CR_R"] == pytest.approx(1337.0)
        assert cdm["OBJECT2_CT_R"] == pytest.approx(-48060.0)
        assert cdm["OBJECT2_CT_T"] == pytest.approx(2492000.0)
        assert cdm["OBJECT2_CN_R"] == pytest.approx(-32.98)
        assert cdm["OBJECT2_CN_T"] == pytest.approx(-758.88)
        assert cdm["OBJECT2_CN_N"] == pytest.approx(71.05)

    def test_object_designators(self, cdm: dict) -> None:
        # Parsed as floats because they look numeric
        assert cdm["OBJECT1_OBJECT_DESIGNATOR"] == 12345
        assert cdm["OBJECT2_OBJECT_DESIGNATOR"] == 67890

    def test_units_stripped(self, cdm: dict) -> None:
        """Ensure no value retains bracket-enclosed unit text."""
        for v in cdm.values():
            if isinstance(v, str):
                assert "[" not in v and "]" not in v

    def test_multiple_blocks(self) -> None:
        """Two concatenated blocks should produce two dicts."""
        two_blocks = SAMPLE_KVN + "\n" + SAMPLE_KVN
        blocks = parse_cdm_kvn(two_blocks)
        assert len(blocks) == 2

    def test_empty_input(self) -> None:
        assert parse_cdm_kvn("") == []
        assert parse_cdm_kvn("   \n\n  ") == []
