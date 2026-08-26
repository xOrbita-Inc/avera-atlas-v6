"""tests/test_leolabs_asset_map.py

SCRUM-411 AC5: our asset identity <-> LeoLabs catalog number mapping, and
resolving which CDM object is ours.

Run from repo root:
    python -m pytest services/planner/tests/test_leolabs_asset_map.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from common.leolabs_asset_map import AssetMapping, AssetRegistry

_FIXTURE = Path(__file__).parent / "fixtures" / "leolabs_cdm_sample.json"

# A subscribed-objects list in the documented LeoLabs shape. CRYOSAT 2 is ours
# (L2669 / 36508); the others are unrelated subscribed vehicles.
_OBJECTS = [
    {"catalogNumber": "L2669", "noradCatalogNumber": 36508, "name": "CRYOSAT 2"},
    {"catalogNumber": "L335", "noradCatalogNumber": 12345, "name": "ILRS SAT B"},
    {"catalogNumber": "L999", "noradCatalogNumber": 54321, "name": "ILRS SAT C"},
]


@pytest.fixture
def cdm() -> dict:
    return json.loads(_FIXTURE.read_text())


# ---------------------------------------------------------------------------
# Construction and lookups
# ---------------------------------------------------------------------------

def test_from_objects_builds_bidirectional_map():
    reg = AssetRegistry.from_objects(_OBJECTS)
    assert len(reg) == 3
    assert reg.leolabs_for_norad(36508) == "L2669"
    assert reg.norad_for_leolabs("L2669") == 36508
    assert "L2669" in reg
    assert "L000" not in reg


def test_from_objects_filters_to_our_fleet():
    reg = AssetRegistry.from_objects(_OBJECTS, our_norads={36508})
    assert len(reg) == 1
    assert reg.leolabs_for_norad(36508) == "L2669"
    assert reg.norad_for_leolabs("L335") is None  # filtered out


def test_from_client_makes_one_list_call():
    client = MagicMock()
    client.list_subscribed_objects.return_value = _OBJECTS
    reg = AssetRegistry.from_client(client)
    client.list_subscribed_objects.assert_called_once()
    assert reg.norad_for_leolabs("L2669") == 36508


def test_tolerant_field_extraction():
    reg = AssetRegistry.from_objects([
        {"catalog_number": "L7", "noradId": "7777", "objectName": "ALT KEYS"},
    ])
    assert reg.leolabs_for_norad(7777) == "L7"  # string norad coerced to int
    assert reg.get("L7").name == "ALT KEYS"


def test_missing_catalog_is_skipped():
    reg = AssetRegistry.from_objects([{"noradCatalogNumber": 1}, {"catalogNumber": "L1"}])
    assert len(reg) == 1
    assert "L1" in reg


# ---------------------------------------------------------------------------
# CDM object resolution (design 6.3): do not assume SAT1 is ours
# ---------------------------------------------------------------------------

def test_resolve_picks_our_object_from_cdm(cdm):
    """On the sample, L2669 (SAT1) is ours and L143957 (SAT2) is not."""
    reg = AssetRegistry.from_objects(_OBJECTS)
    assert reg.resolve_our_catalog_id(cdm) == "L2669"


def test_resolve_picks_our_object_when_ours_is_sat2(cdm):
    """If only the SAT2 object is in our registry, SAT2's id is returned."""
    reg = AssetRegistry.from_objects([
        {"catalogNumber": "L143957", "noradCatalogNumber": 270302, "name": "OURS"},
    ])
    assert reg.resolve_our_catalog_id(cdm) == "L143957"


def test_resolve_by_norad_fallback(cdm):
    """A registry keyed only by NORAD still resolves via COMMENT_NORAD_ID."""
    reg = AssetRegistry.from_objects([
        {"catalogNumber": "SOME_OTHER_ID", "noradCatalogNumber": 36508},
    ])
    # SAT1_COMMENT_NORAD_ID is 36508; the CDM's own designator (L2669) is returned.
    assert reg.resolve_our_catalog_id(cdm) == "L2669"


def test_resolve_raises_when_no_object_is_ours(cdm):
    reg = AssetRegistry.from_objects([
        {"catalogNumber": "L000", "noradCatalogNumber": 111},
    ])
    with pytest.raises(LookupError):
        reg.resolve_our_catalog_id(cdm)


def test_resolve_feeds_parser(cdm):
    """The resolved id is exactly what parse_leolabs_cdm expects."""
    from common.leolabs_cdm_parser import parse_leolabs_cdm

    reg = AssetRegistry.from_objects(_OBJECTS)
    our_id = reg.resolve_our_catalog_id(cdm)
    parsed = parse_leolabs_cdm(cdm, our_id)
    assert parsed.primary.designator == "L2669"
    assert parsed.provenance["primary_norad_id"] == 36508
