"""
services/planner/common/udl_client.py

Unified Data Library (UDL) client for SCRUM-331.
Provides conjunction data retrieval for APS 2.5.

Authentication
--------------
HTTP Basic auth using UDL_USER and UDL_PASS environment variables.
No session cookies -- simpler than the Space-Track pattern.

AC2: CDM retrieval
------------------
get_conjunctions() fetches conjunction records for a primary NORAD ID
and parses them into ConjunctionState dicts that pass directly into
evaluate_conjunction() with no adapter code.

Field mapping (UDL -> evaluate_conjunction):
  tca                          -> t_ca_utc
  satNo2                       -> obj_id
  collisionProb                -> pc_precomputed
  missDistance / 1000          -> miss_distance_km  (UDL: metres, we need km)
  [relPosR, relPosT, relPosN]  -> RTN relative position (metres) -> ECI (km)
  stateVector1.{xpos,ypos,zpos} -> r_sat_km  (already km, J2000)
  stateVector1.{xvel,yvel,zvel} -> v_sat_km_s
  stateVector1.cov + stateVector2.cov -> combined 3x3 RTN covariance
                                         -> rotated to ECI -> p_rel_km2

Unit notes
----------
- relPosR/T/N:  metres  -> divide by 1000 for km
- missDistance: metres  -> divide by 1000 for km
- stateVector positions: km (J2000) -- no conversion needed
- cov: 6-element upper triangle [cr_r, ct_r, ct_t, cn_r, cn_t, cn_n] in m^2
  Expand to 3x3, sum object1 + object2, rotate RTN->ECI, convert m^2->km^2

AC3 (state vector / elset retrieval for secondary conflict screening)
is deferred -- Elset schema not yet captured. Stub included.

Feature flag
------------
Gated behind UDL_ENABLED env var (default false), same pattern as
SPACETRACK_LIVE_POLLING_ENABLED in SCRUM-329.

When both UDL_ENABLED and SPACETRACK_LIVE_POLLING_ENABLED are true,
UDL takes precedence as the primary source per SCRUM-331 scope.
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import requests

log = logging.getLogger("planner")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://unifieddatalibrary.com"
_CONJUNCTION_URL = f"{_BASE_URL}/udl/conjunction"
_ELSET_URL = f"{_BASE_URL}/udl/elset"

# Feature flag -- default false until service account is confirmed and
# SSA agreement is in place for registered spacecraft CDM access.
UDL_ENABLED = os.environ.get("UDL_ENABLED", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _auth_header() -> Dict[str, str]:
    """Build HTTP Basic auth header from UDL_USER and UDL_PASS env vars.

    Raises RuntimeError if either variable is missing.
    """
    username = os.environ.get("UDL_USER")
    password = os.environ.get("UDL_PASS")

    if not username or not password:
        raise RuntimeError(
            "UDL_USER and UDL_PASS environment variables are required "
            "for UDL API access."
        )

    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Accept": "application/json",
    }


# ---------------------------------------------------------------------------
# RTN -> ECI rotation (same math as cdm_to_conjunction.py and server.py)
# ---------------------------------------------------------------------------

def _rtn_to_eci_rotation(r_km: np.ndarray, v_km_s: np.ndarray) -> np.ndarray:
    """Build the 3x3 RTN->ECI rotation matrix from an ECI state vector.

    R = r_hat  (radial)
    N = (r x v) / |r x v|  (cross-track / normal)
    T = N x R  (along-track / tangential)
    """
    r_hat = r_km / np.linalg.norm(r_km)
    h = np.cross(r_km, v_km_s)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)
    return np.column_stack([r_hat, t_hat, n_hat])


def _expand_cov_upper_triangle(cov6: List[float]) -> np.ndarray:
    """Expand a 6-element upper triangle covariance to a 3x3 symmetric matrix.

    UDL cov layout: [cr_r, ct_r, ct_t, cn_r, cn_t, cn_n]
    Maps to:
        [[cr_r, ct_r, cn_r],
         [ct_r, ct_t, cn_t],
         [cn_r, cn_t, cn_n]]
    """
    if len(cov6) < 6:
        return np.zeros((3, 3), dtype=float)

    cr_r, ct_r, ct_t, cn_r, cn_t, cn_n = [float(x) for x in cov6[:6]]
    return np.array([
        [cr_r, ct_r, cn_r],
        [ct_r, ct_t, cn_t],
        [cn_r, cn_t, cn_n],
    ], dtype=float)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_conjunction(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a single UDL conjunction record into a ConjunctionState dict.

    Returns a dict with keys matching evaluate_conjunction() inputs:
      obj_id, t_ca_utc, r_rel_km, p_rel_km2, pc_precomputed, miss_distance_km

    Returns None if required fields are missing or parsing fails.

    AC2: no adapter code -- the parsed dict passes directly into
    evaluate_conjunction() without any intermediate transformation.
    """
    try:
        # --- Object ID ---
        obj_id = str(record.get("satNo2") or record.get("idOnOrbit2") or "UNKNOWN")

        # --- TCA ---
        tca_raw = record.get("tca")
        if not tca_raw:
            return None
        t_ca_utc = str(tca_raw)
        if not t_ca_utc.endswith("Z"):
            t_ca_utc += "Z"

        # --- Collision probability ---
        pc_precomputed = record.get("collisionProb")
        if pc_precomputed is not None:
            pc_precomputed = float(pc_precomputed)

        # --- Miss distance (metres -> km) ---
        miss_m = record.get("missDistance")
        miss_distance_km = float(miss_m) / 1000.0 if miss_m is not None else None

        # --- State vector 1 (primary satellite) ---
        sv1 = record.get("stateVector1") or {}
        r_sat_km = [
            float(sv1.get("xpos", 0.0)),
            float(sv1.get("ypos", 0.0)),
            float(sv1.get("zpos", 0.0)),
        ]
        v_sat_km_s = [
            float(sv1.get("xvel", 0.0)),
            float(sv1.get("yvel", 0.0)),
            float(sv1.get("zvel", 0.0)),
        ]

        r_sat = np.array(r_sat_km, dtype=float)
        v_sat = np.array(v_sat_km_s, dtype=float)

        # --- Relative position RTN (metres) -> ECI (km) ---
        rel_r_m = record.get("relPosR", 0.0)
        rel_t_m = record.get("relPosT", 0.0)
        rel_n_m = record.get("relPosN", 0.0)

        dr_rtn_m = np.array([float(rel_r_m), float(rel_t_m), float(rel_n_m)])

        if np.linalg.norm(r_sat) > 0 and np.linalg.norm(v_sat) > 0:
            rot = _rtn_to_eci_rotation(r_sat, v_sat)
            r_rel_km = (rot @ dr_rtn_m / 1000.0).tolist()
        else:
            # State vector missing -- cannot compute r_rel_km or rotate covariance.
            # Return None so the caller discards this record rather than passing
            # fabricated zero position and covariance to evaluate_conjunction().
            log.warning(
                "UDL conjunction %s has no state vector -- record discarded",
                record.get("id", "?"),
                extra={"event": "udl_missing_state_vector", "id": record.get("id")},
            )
            return None

        # --- Covariance (RTN, m^2) -> ECI (km^2) ---
        sv2 = record.get("stateVector2") or {}
        cov1_raw = sv1.get("cov") or []
        cov2_raw = sv2.get("cov") or []

        p1_rtn = _expand_cov_upper_triangle(cov1_raw)  # m^2
        p2_rtn = _expand_cov_upper_triangle(cov2_raw)  # m^2
        p_rel_rtn = p1_rtn + p2_rtn                    # combined, m^2

        if np.linalg.norm(r_sat) > 0 and np.linalg.norm(v_sat) > 0:
            p_rel_eci_m2 = rot @ p_rel_rtn @ rot.T
        else:
            p_rel_eci_m2 = p_rel_rtn

        p_rel_km2 = (p_rel_eci_m2 / 1e6).flatten().tolist()  # m^2 -> km^2

        return {
            "obj_id":          obj_id,
            "t_ca_utc":        t_ca_utc,
            "r_rel_km":        r_rel_km,
            "p_rel_km2":       p_rel_km2,
            "pc_precomputed":  pc_precomputed,
            "miss_distance_km": miss_distance_km,
            # Carry through for planner request assembly
            "r_sat_km":        r_sat_km,
            "v_sat_km_s":      v_sat_km_s,
            "satNo1":          record.get("satNo1"),
            "satNo2":          record.get("satNo2"),
            "udl_id":          record.get("id"),
        }

    except Exception as exc:
        log.warning(
            "UDL conjunction parse failed for record %s: %s",
            record.get("id", "?"), exc,
            extra={"event": "udl_parse_failed", "id": record.get("id"), "exc": str(exc)},
        )
        return None


# ---------------------------------------------------------------------------
# Public interface -- AC2
# ---------------------------------------------------------------------------

def get_conjunctions(
    sat_no: int,
    days_lookahead: int = 7,
    pc_threshold: float = 1e-5,
) -> List[Dict[str, Any]]:
    """Fetch conjunction records for a primary NORAD ID from UDL.

    Queries GET /udl/conjunction with satNo1 and tca filters.
    Each record is parsed into a ConjunctionState dict that passes
    directly into evaluate_conjunction() with no adapter code (AC2).

    On any failure (missing credentials, network error, parse error),
    returns an empty list and logs the reason. The caller falls back
    to the injected reference CDM path.

    Parameters
    ----------
    sat_no : int
        NORAD catalog number of the primary (protected) satellite.
    days_lookahead : int
        Number of days ahead to query TCAs. Default 7.
    pc_threshold : float
        Minimum collision probability to include. Default 1e-5.

    Returns
    -------
    list[dict]
        List of parsed ConjunctionState dicts. Empty list on any failure.
    """
    if not UDL_ENABLED:
        log.info(
            "UDL disabled -- skipping conjunction fetch",
            extra={"event": "udl_disabled"},
        )
        return []

    try:
        headers = _auth_header()
    except RuntimeError as exc:
        log.warning(
            "UDL credentials missing: %s -- conjunction fetch skipped",
            exc,
            extra={"event": "udl_credentials_missing", "reason": str(exc)},
        )
        return []

    # Build TCA window: now to now + days_lookahead
    now = datetime.now(timezone.utc)
    tca_from = now.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
    tca_to = (now + timedelta(days=days_lookahead)).strftime("%Y-%m-%dT%H:%M:%S.000000Z")

    params = {
        "satNo1":        sat_no,
        "tca":           f">{tca_from}",
        "collisionProb": f">={pc_threshold}",
    }

    try:
        resp = requests.get(
            _CONJUNCTION_URL,
            headers=headers,
            params=params,
            timeout=30.0,
        )
    except Exception as exc:
        log.warning(
            "UDL conjunction fetch failed (network): %s",
            exc,
            extra={"event": "udl_fetch_failed", "reason": str(exc)},
        )
        return []

    if resp.status_code == 401:
        log.warning(
            "UDL conjunction fetch: invalid credentials (401)",
            extra={"event": "udl_auth_failed"},
        )
        return []

    if resp.status_code == 403:
        log.warning(
            "UDL conjunction fetch: not authorized (403) -- "
            "account may lack required data access role",
            extra={"event": "udl_not_authorized"},
        )
        return []

    if not resp.ok:
        log.warning(
            "UDL conjunction fetch: unexpected status %d",
            resp.status_code,
            extra={"event": "udl_unexpected_status", "status": resp.status_code},
        )
        return []

    try:
        records = resp.json()
    except Exception as exc:
        log.warning(
            "UDL conjunction fetch: JSON parse failed: %s",
            exc,
            extra={"event": "udl_json_parse_failed", "exc": str(exc)},
        )
        return []

    if not records:
        log.info(
            "UDL conjunction fetch: 0 records returned for satNo1=%s "
            "(no active conjunctions above Pc threshold, or no data access)",
            sat_no,
            extra={"event": "udl_empty_response", "sat_no": sat_no},
        )
        return []

    parsed = []
    for record in records:
        result = _parse_conjunction(record)
        if result is not None:
            parsed.append(result)

    log.info(
        "UDL conjunction fetch complete: %d/%d records parsed for satNo1=%s",
        len(parsed), len(records), sat_no,
        extra={
            "event": "udl_fetch_complete",
            "parsed": len(parsed),
            "total": len(records),
            "sat_no": sat_no,
        },
    )
    return parsed


# ---------------------------------------------------------------------------
# AC3 stub -- state vector / elset retrieval
# Deferred: Elset schema not yet captured from UDL portal.
# ---------------------------------------------------------------------------

def get_elsets(
    sat_no: Optional[int] = None,
    epoch_window_days: int = 7,
) -> str:
    """Fetch elset (TLE-equivalent) data from UDL and return as TLE text.

    AC3: state vector / ephemeris retrieval for secondary conflict screening.

    Queries GET /udl/elset filtered by epoch window. If sat_no is provided,
    fetches elsets for that specific satellite. Otherwise fetches all LEO
    elsets within the epoch window for catalog screening.

    The response contains line1 and line2 TLE fields which are assembled
    into raw TLE text and returned. This text is compatible with
    _parse_and_propagate_tle() in spacetrack_tle.py -- no new propagation
    code required.

    Parameters
    ----------
    sat_no : int or None
        NORAD catalog number to filter by. None fetches all objects.
    epoch_window_days : int
        Fetch elsets with epoch within the last N days. Default 7.

    Returns
    -------
    str
        Raw TLE text (name\nline1\nline2\n per object).
        Empty string on any failure -- caller falls back gracefully.
    """
    if not UDL_ENABLED:
        log.info(
            "UDL disabled -- skipping elset fetch",
            extra={"event": "udl_disabled"},
        )
        return ""

    try:
        headers = _auth_header()
    except RuntimeError as exc:
        log.warning(
            "UDL credentials missing: %s -- elset fetch skipped",
            exc,
            extra={"event": "udl_credentials_missing", "reason": str(exc)},
        )
        return ""

    now = datetime.now(timezone.utc)
    epoch_from = (now - timedelta(days=epoch_window_days)).strftime(
        "%Y-%m-%dT%H:%M:%S.000000Z"
    )

    params: Dict[str, Any] = {"epoch": f">{epoch_from}"}
    if sat_no is not None:
        params["satNo"] = sat_no

    try:
        resp = requests.get(
            _ELSET_URL,
            headers=headers,
            params=params,
            timeout=30.0,
        )
    except Exception as exc:
        log.warning(
            "UDL elset fetch failed (network): %s",
            exc,
            extra={"event": "udl_elset_fetch_failed", "reason": str(exc)},
        )
        return ""

    if resp.status_code == 401:
        log.warning(
            "UDL elset fetch: invalid credentials (401)",
            extra={"event": "udl_elset_auth_failed"},
        )
        return ""

    if resp.status_code == 403:
        log.warning(
            "UDL elset fetch: not authorized (403)",
            extra={"event": "udl_elset_not_authorized"},
        )
        return ""

    if not resp.ok:
        log.warning(
            "UDL elset fetch: unexpected status %d",
            resp.status_code,
            extra={"event": "udl_elset_unexpected_status", "status": resp.status_code},
        )
        return ""

    try:
        records = resp.json()
    except Exception as exc:
        log.warning(
            "UDL elset fetch: JSON parse failed: %s",
            exc,
            extra={"event": "udl_elset_json_parse_failed", "exc": str(exc)},
        )
        return ""

    if not records:
        log.info(
            "UDL elset fetch: 0 records returned (sat_no=%s)",
            sat_no,
            extra={"event": "udl_elset_empty_response", "sat_no": sat_no},
        )
        return ""

    # Assemble TLE text from line1/line2 fields.
    # Format: optional name line + line1 + line2 per object.
    # Compatible with spacetrack_tle._parse_and_propagate_tle().
    tle_lines = []
    skipped = 0
    for record in records:
        line1 = record.get("line1", "").strip()
        line2 = record.get("line2", "").strip()
        if not line1 or not line2:
            skipped += 1
            continue
        sat_no_str = str(record.get("satNo", ""))
        tle_lines.extend([sat_no_str, line1, line2])

    if skipped > 0:
        log.warning(
            "UDL elset fetch: %d records missing line1/line2 -- skipped",
            skipped,
            extra={"event": "udl_elset_missing_tle_lines", "skipped": skipped},
        )

    tle_text = "\n".join(tle_lines)
    log.info(
        "UDL elset fetch complete: %d TLEs assembled (sat_no=%s)",
        len(tle_lines) // 3, sat_no,
        extra={
            "event": "udl_elset_fetch_complete",
            "tle_count": len(tle_lines) // 3,
            "sat_no": sat_no,
        },
    )
    return tle_text
