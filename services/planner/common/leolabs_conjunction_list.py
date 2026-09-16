"""services/planner/common/leolabs_conjunction_list.py

SCRUM-422: turn parsed LeoLabs CDMs into table rows, and pick one back out again.

SCRUM-412 already fetched every CDM in the window and kept only the highest-risk
one. This module is the other half: the row shape the dashboard's Active
Conjunctions table (SCRUM-421) renders, and the selector that lets an operator
click a row and score that specific conjunction rather than whatever happens to
be top of the list.

Two things here are worth reading before trusting a number out of this module.

The Pc in a row is LeoLabs' Pc, not ours. The parser is deliberate about this:
to_conjunction_state() sets pc_precomputed to None because "the CDM Pc is a
cross-check, not our Pc. The planner computes Pc from this real geometry." A row
carries the CDM Pc because a triage table needs something to sort and colour by
before anything has been scored, and it is labelled pc_source="leolabs_cdm" so a
consumer cannot mistake it for a planner result. The authoritative Pc for a row
arrives when that row is scored through /v1/evaluate.

Rows are one per conjunction EVENT, not one per CDM. LeoLabs reissues a CDM for
the same event as the solution refines, all sharing a COMMENT_EVENT_ID. A table
listing the same secondary five times is unusable, so dedupe_by_event keeps the
highest-risk CDM per event. Selection still runs against the full undeduped list,
so a cdm_id that lost its dedupe still resolves.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from common.leolabs_cdm_parser import ParsedLeoLabsCDM

# Display risk bands. RED and AMBER are not invented here: they are the locked
# Pc_action (1e-4, authorization_envelope.LOCKED_PC_ACTION_CEILING, and the
# operator policy's pc_maneuver_threshold default) and the pc_monitor_threshold
# default (1e-5). So RED means "at or above the action line" and AMBER means "at
# or above the watch line", which is what an operator triaging a table wants the
# colours to mean. GREEN's 1e-7 floor is display-only and mirrors
# services/propagator/main.py's pc_to_risk_level, so the dashboard's existing
# risk-dot classes keep working unchanged for live rows.
PC_RED_THRESHOLD = 1.0e-4
PC_AMBER_THRESHOLD = 1.0e-5
PC_GREEN_THRESHOLD = 1.0e-7

RISK_LEVELS = ("RED", "AMBER", "GREEN", "NOMINAL")

# The conjunction-block fields /v1/evaluate accepts as a row selector, in
# precedence order: most specific first. cdm_id names one message, event_id names
# one conjunction event across its reissues, secondary_norad names the object.
SELECTOR_FIELDS = ("cdm_id", "event_id", "secondary_norad")


def pc_to_risk_level(pc: Optional[float]) -> str:
    """Map a collision probability onto the dashboard's four risk bands.

    A CDM with no Pc at all is NOMINAL here, and its row carries pc=None with
    pc_display="n/a" so the absence is visible rather than rendered as a very
    small number. The row's authoritative risk comes from scoring it.
    """
    if pc is None:
        return "NOMINAL"
    if pc >= PC_RED_THRESHOLD:
        return "RED"
    if pc >= PC_AMBER_THRESHOLD:
        return "AMBER"
    if pc >= PC_GREEN_THRESHOLD:
        return "GREEN"
    return "NOMINAL"


def _pc_display(pc: Optional[float]) -> str:
    """Mirrors the dashboard's own formatting in services/ui/app/main.py."""
    if pc is None:
        return "n/a"
    if pc > 0:
        return f"{pc:.2e}"
    return "< 1e-10"


def _parse_tca(t_ca_utc: str) -> Optional[datetime]:
    try:
        tca = datetime.fromisoformat(str(t_ca_utc).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return tca if tca.tzinfo else tca.replace(tzinfo=timezone.utc)


def event_key(parsed: ParsedLeoLabsCDM) -> str:
    """The identity a row is deduped on.

    event_id first, since that is what LeoLabs reissues a CDM under. Falling back
    to cdm_id and then the secondary's designator means a CDM missing an event id
    still gets its own row instead of silently collapsing into another event's.
    """
    prov = parsed.provenance
    for key in ("event_id", "cdm_id"):
        value = prov.get(key)
        if value is not None:
            return f"{key}:{value}"
    return f"secondary:{parsed.secondary.designator}"


def conjunction_row(
    parsed: ParsedLeoLabsCDM, now: Optional[datetime] = None
) -> Dict[str, Any]:
    """One table row: enough to render it, and enough to score it.

    now is passed in rather than read here so every row in a listing is measured
    against the same clock; otherwise two rows in one response could disagree
    about how far away "now" is.
    """
    prov = parsed.provenance
    pc = parsed.cdm_collision_probability
    miss_m = parsed.miss_distance_m

    tca = _parse_tca(parsed.t_ca_utc)
    time_to_tca_s: Optional[float] = None
    if tca is not None:
        reference = now or datetime.now(timezone.utc)
        time_to_tca_s = (tca - reference.astimezone(timezone.utc)).total_seconds()

    return {
        # -- selectors: any one of these scores this row via /v1/evaluate ---
        "cdm_id": prov.get("cdm_id"),
        "event_id": prov.get("event_id"),
        "secondary_norad": parsed.secondary.norad_id,
        # -- identity ------------------------------------------------------
        "secondary_designator": parsed.secondary.designator,
        "secondary_name": parsed.secondary.object_name,
        "primary_norad": parsed.primary.norad_id,
        "primary_designator": parsed.primary.designator,
        "primary_name": parsed.primary.object_name,
        # -- the numbers the table shows -----------------------------------
        "tca_utc": parsed.t_ca_utc,
        "time_to_tca_s": time_to_tca_s,
        "time_to_tca_min": (
            round(time_to_tca_s / 60.0, 1) if time_to_tca_s is not None else None
        ),
        "miss_distance_m": round(miss_m, 1) if miss_m is not None else None,
        "miss_distance_km": round(miss_m / 1000.0, 4) if miss_m is not None else None,
        "pc": pc,
        "pc_display": _pc_display(pc),
        "risk_level": pc_to_risk_level(pc),
        # -- provenance, so a consumer cannot mistake whose Pc this is ------
        "pc_source": "leolabs_cdm",
        "pc_method": parsed.cdm_collision_probability_method,
        "covariance_source": "real_cdm",
        "source": "leolabs",
    }


def dedupe_by_event(
    conjunctions: Sequence[ParsedLeoLabsCDM],
) -> List[ParsedLeoLabsCDM]:
    """One entry per conjunction event, keeping the first seen.

    Callers pass an already risk-ordered sequence, so "first seen" is the
    highest-risk CDM for that event, which is the one a triage table should show.
    Order is otherwise preserved.
    """
    seen = set()
    kept: List[ParsedLeoLabsCDM] = []
    for parsed in conjunctions:
        key = event_key(parsed)
        if key in seen:
            continue
        seen.add(key)
        kept.append(parsed)
    return kept


def _matches(parsed: ParsedLeoLabsCDM, field: str, wanted: Any) -> bool:
    if field == "secondary_norad":
        actual = parsed.secondary.norad_id
    else:
        actual = parsed.provenance.get(field)
    if actual is None:
        return False
    # Compared as strings: a NORAD id arrives as an int from the parser and
    # usually as a string from a JSON request body, and "270302" and 270302 are
    # the same object.
    return str(actual) == str(wanted)


def select_conjunction(
    conjunctions: Sequence[ParsedLeoLabsCDM],
    selector: Dict[str, Any],
) -> Optional[ParsedLeoLabsCDM]:
    """Find the conjunction a selector names, or None.

    Fields are tried in SELECTOR_FIELDS order, most specific first, and the first
    field actually present in the selector decides the match. Returning None for
    an unmatched selector is deliberate: falling back to the highest-risk
    conjunction would hand the operator another object's numbers under the label
    of the one they clicked.

    Matching runs against the full, undeduped list so a cdm_id from an earlier
    listing still resolves after a newer CDM has taken its event's row.
    """
    for field in SELECTOR_FIELDS:
        wanted = selector.get(field)
        if wanted in (None, ""):
            continue
        for parsed in conjunctions:
            if _matches(parsed, field, wanted):
                return parsed
        return None
    return None


def selector_from_conjunction_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the row selector out of an /v1/evaluate conjunction block.

    Returns only the keys actually supplied, so an empty dict means "no row was
    picked" and the caller keeps today's highest-risk behaviour.

    Note that build_evaluate_request already writes cdm_id, event_id and
    secondary_norad into the conjunction block it produces, so these are the
    field names the LeoLabs path already speaks -- the selector reuses them
    rather than inventing a parallel vocabulary.
    """
    return {
        field: block[field]
        for field in SELECTOR_FIELDS
        if block.get(field) not in (None, "")
    }
