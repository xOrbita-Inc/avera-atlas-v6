"""Parse CCSDS 508.0-B-1 KVN Conjunction Data Messages into Python dicts."""

from __future__ import annotations

import re
from typing import Any


# Regex to strip trailing unit annotations like " [km]" or " [m**2]"
_UNIT_RE = re.compile(r"\s*\[.*?\]\s*$")


def _try_numeric(value: str) -> float | str:
    """Return *value* as a float if it looks numeric, otherwise as a string."""
    try:
        return float(value)
    except ValueError:
        return value


def parse_cdm_kvn(kvn_text: str) -> list[dict[str, Any]]:
    """Parse one or more KVN CDM blocks from *kvn_text*.

    Each block starts with a ``CCSDS_CDM_VERS`` line.  Within a block the
    two object sections (``OBJECT = OBJECT1`` / ``OBJECT = OBJECT2``) are
    detected, and their keys are prefixed with ``OBJECT1_`` or ``OBJECT2_``
    so they coexist in a single flat dict.

    Units in square brackets (e.g. ``[m**2]``) are stripped and numeric
    values are converted to ``float``.
    """
    blocks: list[dict[str, Any]] = []

    # Split on "CCSDS_CDM_VERS" — the first element before the first
    # occurrence is discarded (it is empty or whitespace).
    raw_blocks = re.split(r"(?=CCSDS_CDM_VERS)", kvn_text)

    for raw in raw_blocks:
        raw = raw.strip()
        if not raw:
            continue

        record: dict[str, Any] = {}
        current_object: str | None = None  # "OBJECT1" or "OBJECT2"

        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("COMMENT"):
                continue

            # KVN lines are "KEY = VALUE"
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = _UNIT_RE.sub("", value.strip())

            # Track which object section we are in.
            if key == "OBJECT":
                current_object = value.strip()
                continue

            # Convert value
            converted = _try_numeric(value)

            # Prefix per-object keys
            if current_object is not None:
                key = f"{current_object}_{key}"

            record[key] = converted

        if record:
            blocks.append(record)

    return blocks
