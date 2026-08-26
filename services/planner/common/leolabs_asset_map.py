"""services/planner/common/leolabs_asset_map.py

Map our asset identity to LeoLabs catalog identifiers (SCRUM-411 AC5).

LeoLabs identifies objects by its own catalog number (e.g. L2669, L335), not by
NORAD id, and its CDMs put OBJECT_DESIGNATOR = the LeoLabs catalog number. Our
stack identifies assets by NORAD id everywhere else (primary_norad, the UI's
asset NORAD field). This module bridges the two.

It is built once from the subscribed-objects list (design section 6.3: "mapped
once via list_subscribed_objects"), then used to:
  - translate a NORAD id to its LeoLabs catalog number for a CDM/screening query,
  - translate a LeoLabs catalog number back to a NORAD id for our records,
  - decide which object in a CDM is ours, so parse_leolabs_cdm can be told which
    object is the primary without assuming SAT1.

The registry does no network I/O itself beyond from_client(), which makes exactly
one list call; everything else is in-memory lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

# LeoLabs object JSON field names vary slightly across endpoints and doc
# versions, so extraction is tolerant. The design flags exact field names as
# "confirm from sample" (Appendix A); these cover the documented spellings.
_CATALOG_KEYS = ("catalogNumber", "catalog_number", "leolabsCatalogNumber",
                 "objectId", "id")
_NORAD_KEYS = ("noradCatalogNumber", "noradCatalogNum", "noradId", "norad_id",
               "norad")
_NAME_KEYS = ("name", "objectName", "commonName")


def _extract(obj: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            return obj[k]
    return None


def _as_norad(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class AssetMapping:
    """One subscribed object's identity across the two catalogs."""

    leolabs_catalog: str
    norad_id: Optional[int]
    name: Optional[str] = None


class AssetRegistry:
    """In-memory map between our asset identity and LeoLabs catalog numbers."""

    def __init__(self, mappings: Iterable[AssetMapping]) -> None:
        self._by_leolabs: Dict[str, AssetMapping] = {}
        self._by_norad: Dict[int, AssetMapping] = {}
        for m in mappings:
            self._by_leolabs[m.leolabs_catalog] = m
            if m.norad_id is not None:
                self._by_norad[m.norad_id] = m

    # -- construction -----------------------------------------------------

    @classmethod
    def from_objects(
        cls,
        objects: Iterable[Dict[str, Any]],
        our_norads: Optional[Set[int]] = None,
    ) -> "AssetRegistry":
        """Build from a list of LeoLabs object records.

        If our_norads is given, only objects whose NORAD id is in that set are
        kept, so the registry reflects our actual fleet rather than every object
        the account can see.
        """
        mappings: List[AssetMapping] = []
        for obj in objects:
            catalog = _extract(obj, _CATALOG_KEYS)
            if catalog is None:
                continue
            norad = _as_norad(_extract(obj, _NORAD_KEYS))
            if our_norads is not None and norad not in our_norads:
                continue
            mappings.append(
                AssetMapping(
                    leolabs_catalog=str(catalog),
                    norad_id=norad,
                    name=_extract(obj, _NAME_KEYS),
                )
            )
        return cls(mappings)

    @classmethod
    def from_client(
        cls,
        client: Any,
        our_norads: Optional[Set[int]] = None,
    ) -> "AssetRegistry":
        """Build from the subscribed-objects list via one list call.

        `client` is a LeoLabsClient (or anything exposing list_subscribed_objects).
        This is the "mapped once" step; hold the returned registry rather than
        calling per CDM.
        """
        objects = client.list_subscribed_objects()
        return cls.from_objects(objects, our_norads=our_norads)

    # -- lookups ----------------------------------------------------------

    def leolabs_for_norad(self, norad_id: int) -> Optional[str]:
        m = self._by_norad.get(int(norad_id))
        return m.leolabs_catalog if m else None

    def norad_for_leolabs(self, leolabs_catalog: str) -> Optional[int]:
        m = self._by_leolabs.get(str(leolabs_catalog))
        return m.norad_id if m else None

    def get(self, leolabs_catalog: str) -> Optional[AssetMapping]:
        return self._by_leolabs.get(str(leolabs_catalog))

    def __contains__(self, leolabs_catalog: object) -> bool:
        return str(leolabs_catalog) in self._by_leolabs

    def __len__(self) -> int:
        return len(self._by_leolabs)

    def catalog_numbers(self) -> List[str]:
        return list(self._by_leolabs)

    # -- CDM object resolution (design 6.3) -------------------------------

    def resolve_our_catalog_id(self, cdm: Dict[str, Any]) -> str:
        """Return the LeoLabs catalog id of whichever CDM object is ours.

        Checks each object's OBJECT_DESIGNATOR against the registry, and its
        COMMENT_NORAD_ID as a fallback, so a match works whether the CDM is keyed
        by LeoLabs id or NORAD. The returned id is what parse_leolabs_cdm takes as
        our_catalog_id, which then becomes the primary.

        Raises LookupError if no object is ours. If both objects are ours (an
        intra-fleet conjunction), SAT1 is returned as the primary and the caller
        can re-resolve with the other id if it wants the reverse framing.
        """
        matches: List[str] = []
        for sat_key in ("SAT1", "SAT2"):
            designator = cdm.get(f"{sat_key}_OBJECT_DESIGNATOR")
            norad = _as_norad(cdm.get(f"{sat_key}_COMMENT_NORAD_ID"))
            if designator is not None and str(designator) in self._by_leolabs:
                matches.append(str(designator))
            elif norad is not None and norad in self._by_norad:
                # Registry knows this NORAD; use the CDM's own designator so the
                # returned id matches what the parser will read from the CDM.
                matches.append(str(designator) if designator is not None
                               else self._by_norad[norad].leolabs_catalog)

        if not matches:
            raise LookupError(
                "no CDM object matches our asset registry "
                f"(SAT1={cdm.get('SAT1_OBJECT_DESIGNATOR')!r}, "
                f"SAT2={cdm.get('SAT2_OBJECT_DESIGNATOR')!r}); "
                "the registry may be stale or this event is not ours."
            )
        return matches[0]
