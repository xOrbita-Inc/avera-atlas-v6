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
is implemented in get_elsets() below. Fully live on the screening path.

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
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from aps_math import frames
import requests

log = logging.getLogger("planner")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://unifieddatalibrary.com"
_CONJUNCTION_URL = f"{_BASE_URL}/udl/conjunction"
_ELSET_URL = f"{_BASE_URL}/udl/elset"
_ELSET_QUERYHELP_URL = f"{_BASE_URL}/udl/elset/queryhelp"

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

    SCRUM-397: delegates to aps_math.frames. This was one of three copies of
    the same function in the repo, and the one place that needed it most called
    none of them. Kept as a name so this module's callers do not change.

    Behaviour change: the shared version returns the identity for a degenerate
    state vector rather than dividing by zero. That is strictly safer here.
    """
    return frames.rtn_to_eci_rotation(r_km, v_km_s)


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

        if not cov1_raw and not cov2_raw:
            # Covariance fields missing -- cannot compute p_rel_km2.
            # Return None so the caller discards this record rather than passing
            # an all-zero covariance to evaluate_conjunction(). Mirrors the
            # missing-state-vector guard above.
            log.warning(
                "UDL conjunction %s has no covariance fields -- record discarded",
                record.get("id", "?"),
                extra={"event": "udl_missing_covariance", "id": record.get("id")},
            )
            return None

        p1_rtn = _expand_cov_upper_triangle(cov1_raw)  # m^2
        p2_rtn = _expand_cov_upper_triangle(cov2_raw)  # m^2
        p_rel_rtn = p1_rtn + p2_rtn                    # combined, m^2
        p_rel_eci_m2 = rot @ p_rel_rtn @ rot.T

        p_rel_km2 = (p_rel_eci_m2 / 1e6).flatten().tolist()  # m^2 -> km^2

        # --- Relative velocity (ECI, km/s) -------------------------------
        #
        # SCRUM-398. sv2's velocity was already being fetched and thrown away,
        # since this function only read its covariance. Without a relative
        # velocity the planner has no encounter plane, so no Pc, so the Pc gate
        # from SCRUM-389 never runs on a UDL-sourced event.
        #
        # Sign convention is secondary minus primary, matching r_rel_km above
        # and matching compute_pc_from_geometry's v2 = v1 + v_rel.
        #
        # UDL's relVelR/T/N, when present, is RTN in metres per second, the same
        # shape as relPosR/T/N which this function already rotates. Preferred
        # over the state-vector difference because it is the originator's own
        # figure at TCA rather than ours.
        rel_vel_r = record.get("relVelR")
        if rel_vel_r is not None:
            dv_rtn_m_s = np.array([
                float(rel_vel_r),
                float(record.get("relVelT", 0.0)),
                float(record.get("relVelN", 0.0)),
            ])
            v_rel_km_s = (rot @ dv_rtn_m_s / 1000.0).tolist()
            v_rel_source = "relative_velocity_rtn"
        elif sv2.get("xvel") is not None:
            v_sec = np.array([
                float(sv2.get("xvel", 0.0)),
                float(sv2.get("yvel", 0.0)),
                float(sv2.get("zvel", 0.0)),
            ])
            v_rel_km_s = (v_sec - v_sat).tolist()   # already km/s, already ECI
            v_rel_source = "state_vector_difference"
        else:
            # None rather than zero. The planner reads a missing relative
            # velocity as "no Pc could be established", which is correct. A zero
            # would claim the two objects are co-moving.
            v_rel_km_s = None
            v_rel_source = "unavailable"

        return {
            "obj_id":          obj_id,
            "t_ca_utc":        t_ca_utc,
            "r_rel_km":        r_rel_km,
            "v_rel_km_s":      v_rel_km_s,
            "v_rel_source":    v_rel_source,
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
    tca_to_dt = now + timedelta(days=days_lookahead)
    tca_to = tca_to_dt.strftime("%Y-%m-%dT%H:%M:%S.000000Z")

    params = {
        "satNo1":        sat_no,
        # NOTE (SCRUM-364): tca_to was calculated but never applied here --
        # only the lower bound (tca_from) was sent to UDL, so days_lookahead
        # was silently ignored and every query searched an unbounded future
        # window. Found during live testing. UDL's query syntax for a
        # combined upper+lower bound on one field was not verified against
        # the live service, so rather than guess at new syntax, the upper
        # bound is enforced in Python below after the response comes back.
        "tca":           f">{tca_from}",
        # NOTE (SCRUM-364): UDL's query parser rejects the >= operator entirely
        # (confirmed via live testing -- returns 400 "Query parameter exceeds
        # maximum allowed value"). Only > is supported. Using > instead of >=
        # means a record with collisionProb exactly equal to pc_threshold is
        # excluded -- an accepted, minor tradeoff, not a functional gap, since
        # pc_threshold is a conservative screening floor, not an exact cutoff.
        #
        # Also force plain decimal formatting (not Python's default float
        # repr) since small thresholds like 1e-05 would otherwise render in
        # scientific notation, which UDL also rejects.
        "collisionProb": f">{pc_threshold:.10f}",
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

    # NOTE (SCRUM-364): enforce the days_lookahead upper bound here, in
    # Python, since it is not sent to UDL as part of the query (see note
    # above params["tca"]). Parsed as real datetimes, not compared as raw
    # strings, since UDL's returned tca precision (decimal places) isn't
    # guaranteed to match tca_to's fixed 6-digit format.
    before_filter = len(parsed)
    def _tca_within_window(record: Dict[str, Any]) -> bool:
        try:
            tca_dt = datetime.fromisoformat(record["t_ca_utc"].replace("Z", "+00:00"))
            return tca_dt <= tca_to_dt
        except Exception as exc:
            # Don't silently drop the record, but DO log -- a silent except
            # here previously masked a real bug (wrong dict key) that let
            # every record through unfiltered. Found via unit testing.
            log.warning(
                "UDL conjunction: could not evaluate tca lookahead window "
                "for a record, letting it through unfiltered: %s",
                exc,
                extra={"event": "udl_tca_window_check_failed", "reason": str(exc)},
            )
            return True
    parsed = [r for r in parsed if _tca_within_window(r)]
    if len(parsed) < before_filter:
        log.info(
            "UDL conjunction fetch: %d record(s) beyond %d-day lookahead window excluded",
            before_filter - len(parsed), days_lookahead,
            extra={
                "event": "udl_tca_lookahead_filtered",
                "excluded": before_filter - len(parsed),
                "days_lookahead": days_lookahead,
            },
        )

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
# AC3 -- state vector / elset retrieval
# Implemented: get_elsets() is fully live on the secondary conflict screening path.
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
    the retired spacetrack_tle._parse_and_propagate_tle (SCRUM-431) -- no new propagation
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
    # Same convention the retired spacetrack_tle helper used (SCRUM-431).
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


# ---------------------------------------------------------------------------
# Credential validity probe (SCRUM-363)
# ---------------------------------------------------------------------------
#
# Uses GET /udl/elset/queryhelp -- authenticated, no query params, returns
# schema/help metadata instead of records. Confirms the credentials
# authenticate without pulling real data. Does NOT confirm data-read
# authorization for any specific record type (see team note on SCRUM-363).
#
# This is the RAW probe: one live call, no caching or throttling. The
# throttled wrapper that /udl-status actually calls is added in step 3.

def check_credential_validity() -> Dict[str, Any]:
    """Perform a single live probe against UDL to check credential validity.

    Returns a dict:
      status      : 'valid' | 'invalid' | 'unreachable'
      http_status : int or None (the raw HTTP status code, if one was received)
      checked_at_utc : str -- ISO-8601 UTC timestamp of this probe attempt

    status meanings:
      'valid'       -- 200 OK, credentials authenticate right now
      'invalid'     -- 401 or 403, credentials rejected or rotated
      'unreachable' -- network error, timeout, or any other unexpected
                       response (5xx etc). Not a credentials problem.
    """
    checked_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    try:
        headers = _auth_header()
    except RuntimeError as exc:
        log.warning(
            "UDL credential probe: credentials missing: %s",
            exc,
            extra={"event": "udl_probe_credentials_missing", "reason": str(exc)},
        )
        return {
            "status": "invalid",
            "http_status": None,
            "checked_at_utc": checked_at_utc,
        }

    try:
        resp = requests.get(
            _ELSET_QUERYHELP_URL,
            headers=headers,
            timeout=10.0,
        )
    except Exception as exc:
        log.warning(
            "UDL credential probe failed (network): %s",
            exc,
            extra={"event": "udl_probe_unreachable", "reason": str(exc)},
        )
        return {
            "status": "unreachable",
            "http_status": None,
            "checked_at_utc": checked_at_utc,
        }

    if resp.status_code == 200:
        log.info(
            "UDL credential probe: valid",
            extra={"event": "udl_probe_valid"},
        )
        return {
            "status": "valid",
            "http_status": 200,
            "checked_at_utc": checked_at_utc,
        }

    if resp.status_code in (401, 403):
        log.warning(
            "UDL credential probe: invalid (%d)",
            resp.status_code,
            extra={"event": "udl_probe_invalid", "http_status": resp.status_code},
        )
        return {
            "status": "invalid",
            "http_status": resp.status_code,
            "checked_at_utc": checked_at_utc,
        }

    log.warning(
        "UDL credential probe: unreachable, unexpected status %d",
        resp.status_code,
        extra={"event": "udl_probe_unexpected_status", "http_status": resp.status_code},
    )
    return {
        "status": "unreachable",
        "http_status": resp.status_code,
        "checked_at_utc": checked_at_utc,
    }


# ---------------------------------------------------------------------------
# Throttled probe wrapper (SCRUM-363 AC2)
# ---------------------------------------------------------------------------
_PROBE_INTERVAL_SECONDS = 1800  # 30 minutes

# Stopwatch reference point (time.monotonic()) for the last real probe.
# None until the first probe ever runs.
_last_probe_monotonic: float | None = None

# The last result returned by check_credential_validity(), reused for any
# /udl-status call that arrives before _PROBE_INTERVAL_SECONDS has elapsed.
_last_probe_result: Dict[str, Any] | None = None


def get_credential_validity() -> Dict[str, Any]:
    """Return credential validity, using a cached result if the throttle
    interval has not yet elapsed. Runs a fresh live probe otherwise.

    This is the function /udl-status should call, not check_credential_validity()
    directly, since that always makes a live UDL call with no throttling.
    """
    global _last_probe_monotonic, _last_probe_result

    now = time.monotonic()

    if _last_probe_result is not None and _last_probe_monotonic is not None:
        elapsed = now - _last_probe_monotonic
        if elapsed < _PROBE_INTERVAL_SECONDS:
            remaining = _PROBE_INTERVAL_SECONDS - elapsed
            log.info(
                "UDL credential probe: cached result reused",
                extra={
                    "event": "udl_probe_cache_hit",
                    "last_checked_at_utc": _last_probe_result["checked_at_utc"],
                    "status": _last_probe_result["status"],
                    "next_live_probe_in_seconds": round(remaining, 1),
                },
            )
            return _last_probe_result

    # Throttle interval elapsed, or this is the very first call -- run a
    # real probe against UDL.
    result = check_credential_validity()
    _last_probe_result = result
    _last_probe_monotonic = now

    log.info(
        "UDL credential probe: live probe ran",
        extra={
            "event": "udl_probe_cache_miss",
            "checked_at_utc": result["checked_at_utc"],
            "status": result["status"],
            "next_live_probe_in_seconds": _PROBE_INTERVAL_SECONDS,
        },
    )
    return result
