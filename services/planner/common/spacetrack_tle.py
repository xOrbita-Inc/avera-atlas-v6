"""
services/planner/common/spacetrack_tle.py

Space-Track TLE catalog client for secondary conflict screening (SCRUM-330).

Fetches public TLE data from Space-Track (no SSA agreement required),
propagates catalog objects to the screening epoch using SGP4, and returns
a list of known_objects dicts ready for _run_secondary_conflict_check()
in atlas_artifact.py.

Design
------
- Auth follows the session-cookie pattern established in services/ingest/
  spacetrack_client.py (John's pattern on the VPS). Duplicated here because
  the planner cannot import from the ingest service.
- TLE query is filtered to LEO objects (period < 128 min) to keep the
  catalog size manageable (~5,000-8,000 objects vs 25,000+ total).
- Objects are propagated to a single screening epoch (burn time) using SGP4.
  Proximity is checked at that epoch only -- not over a time window.
  Full time-window propagation is APS 3.0 scope.
- On any failure (network, auth, parse), returns an empty list with a
  logged reason. Caller receives known_objects=[] which preserves the
  not_performed fallback in _run_secondary_conflict_check().

Screening radius
----------------
Default 5 km. Deliberately conservative for a single-epoch check --
the actual miss distance at TCA may differ due to relative motion.
Operator should treat any flagged object as a screening hit requiring
manual verification, not a confirmed conjunction.

Dependencies
------------
sgp4>=2.22  (add to services/planner/requirements.txt)
requests    (already in requirements.txt)
numpy       (already in requirements.txt)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import requests

log = logging.getLogger("planner")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://www.space-track.org"
_LOGIN_URL = f"{_BASE_URL}/ajaxauth/login"

# LEO filter: period < 128 min covers up to ~2000 km altitude.
# Keeps catalog to ~5,000-8,000 objects rather than the full 25,000+.
_TLE_QUERY_URL = (
    f"{_BASE_URL}/basicspacedata/query/class/gp"
    "/PERIOD/<128"
    "/EPOCH/>now-30"
    "/orderby/NORAD_CAT_ID"
    "/format/tle"
)

# Default screening radius [km] -- conservative single-epoch proximity threshold.
_DEFAULT_SCREENING_RADIUS_KM = 5.0


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _login(session: requests.Session) -> None:
    """Authenticate with Space-Track and store session cookie.

    Raises RuntimeError on auth failure.
    """
    username = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASS")

    if not username or not password:
        raise RuntimeError(
            "SPACETRACK_USER and SPACETRACK_PASS environment variables are required "
            "for Space-Track TLE catalog access."
        )

    resp = session.post(
        _LOGIN_URL,
        data={"identity": username, "password": password},
        timeout=15.0,
    )
    resp.raise_for_status()

    body = resp.text
    if not body or "failed" in body.lower():
        raise RuntimeError(f"Space-Track login failed: {body[:200]}")


# ---------------------------------------------------------------------------
# TLE fetch
# ---------------------------------------------------------------------------

def _fetch_tle_text(session: requests.Session) -> str:
    """Fetch raw TLE text for LEO catalog objects.

    Returns raw two-line element text. Empty string on failure.
    """
    resp = session.get(_TLE_QUERY_URL, timeout=30.0)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# TLE parse + propagate
# ---------------------------------------------------------------------------

def _parse_and_propagate_tle(
    tle_text: str,
    epoch_utc: datetime,
    r_sat_km: np.ndarray,
    screening_radius_km: float,
) -> List[Dict[str, Any]]:
    """Parse TLE text, propagate each object to epoch_utc, return nearby objects.

    Only returns objects within screening_radius_km of r_sat_km to avoid
    building a full catalog list in memory.

    Parameters
    ----------
    tle_text : str
        Raw two-line element text from Space-Track.
    epoch_utc : datetime
        Screening epoch (burn time).
    r_sat_km : np.ndarray
        Satellite ECI position at screening epoch [km].
    screening_radius_km : float
        Proximity threshold [km].

    Returns
    -------
    list[dict]
        Each dict has keys: obj_id (str), r_km (list[float])
        Format matches known_objects expected by _run_secondary_conflict_check().
    """
    try:
        from sgp4.api import Satrec, jday
    except ImportError:
        raise RuntimeError(
            "sgp4 library not installed. Add sgp4>=2.22 to requirements.txt."
        )

    lines = [l.strip() for l in tle_text.splitlines() if l.strip()]

    # TLE format: alternating line 1 (starts with '1 ') and line 2 (starts with '2 ')
    # Some formats include a name line before each pair -- handle both.
    pairs: list[tuple[str, str, str]] = []  # (name, line1, line2)
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            # No name line
            pairs.append(("", lines[i], lines[i + 1]))
            i += 2
        elif not lines[i].startswith("1 ") and not lines[i].startswith("2 "):
            # Name line -- next two should be TLE lines
            if i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
                pairs.append((lines[i], lines[i + 1], lines[i + 2]))
                i += 3
            else:
                i += 1
        else:
            i += 1

    # Compute Julian date for screening epoch
    jd, fr = jday(
        epoch_utc.year, epoch_utc.month, epoch_utc.day,
        epoch_utc.hour, epoch_utc.minute,
        epoch_utc.second + epoch_utc.microsecond / 1e6,
    )

    nearby = []
    errors = 0

    for name, line1, line2 in pairs:
        try:
            sat = Satrec.twoline2rv(line1, line2)
            e, r, v = sat.sgp4(jd, fr)

            if e != 0:
                # SGP4 error code -- object decay or bad elements
                continue

            r_obj = np.array(r, dtype=float)  # km, ECI (TEME)
            sep = float(np.linalg.norm(r_sat_km - r_obj))

            if sep <= screening_radius_km:
                # Extract NORAD ID from line 1 (columns 2-7)
                norad_id = line1[2:7].strip()
                obj_label = name.strip() if name.strip() else norad_id
                nearby.append({
                    "obj_id": obj_label,
                    "r_km": r_obj.tolist(),
                    "separation_km": round(sep, 3),
                    "norad_id": norad_id,
                })

        except Exception:
            errors += 1
            continue

    if errors > 0:
        log.warning(
            "spacetrack_tle: %d TLE propagation errors (bad elements or decayed objects)",
            errors,
            extra={"event": "tle_propagation_errors", "count": errors},
        )

    return nearby


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_catalog_objects(
    r_sat_km: List[float],
    burn_time_utc: str,
    screening_radius_km: float = _DEFAULT_SCREENING_RADIUS_KM,
) -> List[Dict[str, Any]]:
    """Fetch LEO TLE catalog and return objects near the post-burn position.

    This is the primary entry point called from server.py before
    build_atlas_artifact(). Returns a list of nearby catalog objects
    suitable for passing as known_objects to _run_secondary_conflict_check().

    On any failure (missing credentials, network error, parse error),
    returns an empty list and logs the reason. The caller's fallback to
    not_performed is preserved automatically since known_objects=[] triggers
    the not_performed branch in _run_secondary_conflict_check().

    Parameters
    ----------
    r_sat_km : list[float]
        Post-burn satellite ECI position [x, y, z] km.
        For a short burn, this equals r_sat_km from the request (position
        barely changes during the burn; only velocity changes).
    burn_time_utc : str
        Burn execution time (ISO-8601 UTC). Used as the screening epoch.
    screening_radius_km : float
        Proximity threshold [km]. Default 5 km.

    Returns
    -------
    list[dict]
        List of dicts with keys: obj_id, r_km, separation_km, norad_id.
        Empty list on any failure.
    """
    # Parse screening epoch
    try:
        # Handle both 'Z' suffix and '+00:00' offset
        ts = burn_time_utc.replace("Z", "+00:00")
        epoch_utc = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except Exception as exc:
        log.warning(
            "spacetrack_tle: could not parse burn_time_utc %r: %s -- skipping catalog fetch",
            burn_time_utc, exc,
            extra={"event": "tle_epoch_parse_fail", "burn_time_utc": burn_time_utc},
        )
        return []

    r_post = np.array(r_sat_km, dtype=float)

    # SCRUM-331: prefer UDL elsets when UDL is enabled.
    # UDL returns line1/line2 TLE strings directly -- same format as Space-Track.
    try:
        from common.udl_client import UDL_ENABLED, get_elsets
        if UDL_ENABLED:
            tle_text = get_elsets(epoch_window_days=7)
            if tle_text:
                log.info(
                    "spacetrack_tle: using UDL elsets for catalog screening",
                    extra={"event": "tle_source_udl"},
                )
                try:
                    nearby = _parse_and_propagate_tle(
                        tle_text, epoch_utc, r_post, screening_radius_km
                    )
                    log.info(
                        "spacetrack_tle: catalog screening complete (UDL), %d nearby objects",
                        len(nearby),
                        extra={"event": "tle_screening_complete", "nearby_count": len(nearby),
                               "source": "udl", "screening_radius_km": screening_radius_km},
                    )
                    return nearby
                except Exception as exc:
                    log.warning(
                        "spacetrack_tle: UDL propagation failed: %s -- falling back to Space-Track",
                        exc,
                        extra={"event": "tle_udl_propagation_fail", "reason": str(exc)},
                    )
    except ImportError:
        pass

    # Fall back to Space-Track TLE catalog.
    session = requests.Session()
    try:
        _login(session)
    except Exception as exc:
        log.warning(
            "spacetrack_tle: login failed: %s -- secondary conflict check will be not_performed",
            exc,
            extra={"event": "tle_login_fail", "reason": str(exc)},
        )
        return []

    try:
        tle_text = _fetch_tle_text(session)
    except Exception as exc:
        log.warning(
            "spacetrack_tle: TLE fetch failed: %s -- secondary conflict check will be not_performed",
            exc,
            extra={"event": "tle_fetch_fail", "reason": str(exc)},
        )
        return []

    if not tle_text.strip():
        log.warning(
            "spacetrack_tle: Space-Track returned empty TLE catalog -- "
            "secondary conflict check will be not_performed",
            extra={"event": "tle_empty_catalog"},
        )
        return []

    try:
        nearby = _parse_and_propagate_tle(
            tle_text, epoch_utc, r_post, screening_radius_km
        )
    except Exception as exc:
        log.warning(
            "spacetrack_tle: propagation failed: %s -- secondary conflict check will be not_performed",
            exc,
            extra={"event": "tle_propagation_fail", "reason": str(exc)},
        )
        return []

    log.info(
        "spacetrack_tle: catalog screening complete, %d nearby objects within %.1f km",
        len(nearby), screening_radius_km,
        extra={
            "event": "tle_screening_complete",
            "nearby_count": len(nearby),
            "screening_radius_km": screening_radius_km,
        },
    )
    return nearby
