"""services/planner/common/leolabs_evaluate.py

Assemble a /v1/evaluate request from a parsed LeoLabs CDM (SCRUM-411 AC7).

This is the glue between the parser and the planner: it turns a ParsedLeoLabsCDM
into the exact request body /v1/evaluate expects, so a real LeoLabs CDM drives a
recommendation on real per-object covariance.

What comes from the CDM vs the operator
----------------------------------------
From the CDM (via the parser):
  - the satellite (primary) ECI state at TCA -> satellite.r_sat_km / v_sat_km_s
  - the relative geometry and the real combined ECI covariance -> conjunction.*
  - source="leolabs" and covariance_source="real_cdm", which together light the
    dashboard LIVE badge and keep the covariance adapter from overwriting the
    real matrix (see server.post_evaluate, the AC7 bypass branch).
  - the primary's own hard-body radius -> satellite.radius_m. The scorer applies
    the ADR-010 combined-HBR floor to this; it does not use the CDM's combined
    radius directly, by design. The CDM combined radius is still recorded in
    provenance and is what the parser's Pc cross-check uses.

From the operator (arguments here):
  - sat_id, v_remaining_m_s, t_burn_utc, and the policy. These are operational
    inputs the CDM cannot supply. t_burn_utc defaults to TCA when not given;
    a real planning run should pass an operationally meaningful burn time.

The satellite state is the primary's state at TCA on purpose: the scorer
reconstructs the secondary as r2 = r_sat + r_rel and v2 = v_sat + v_rel, so the
encounter geometry is exact only when r_sat is the primary at the same epoch as
r_rel.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from common.leolabs_cdm_parser import ParsedLeoLabsCDM

# Default lead time of the planned burn before TCA when the caller does not give
# one. The scorer requires t_burn_utc strictly before TCA; six hours is a plain,
# operationally-sane pre-TCA default for a validation run. A real planning run
# should pass its own burn window.
_DEFAULT_BURN_LEAD_S = 6 * 3600

# A minimal, sensible default policy. Callers with an operator policy should pass
# their own; these are the same weights the planner tests use and are safe for a
# validation run.
_DEFAULT_POLICY: Dict[str, Any] = {
    "operator_id": "LEOLABS-AC7",
    "policy_version": "leolabs",
    "lambda_v": 0.01,
    "lambda_L": 0.001,
    "dv_mag_limit_m_s": 0.5,
}


def _burn_before_tca(t_ca_utc: str, lead_s: float) -> str:
    """Return an ISO-8601 UTC burn time `lead_s` seconds before TCA.

    The scorer requires the burn strictly before TCA. Falls back to a plain
    string subtraction only if TCA cannot be parsed, which should not happen for
    a well-formed CDM.
    """
    s = t_ca_utc.strip().replace("Z", "+00:00")
    tca = datetime.fromisoformat(s)
    if tca.tzinfo is None:
        tca = tca.replace(tzinfo=timezone.utc)
    burn = tca - timedelta(seconds=lead_s)
    return burn.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_evaluate_request(
    parsed: ParsedLeoLabsCDM,
    *,
    sat_id: str,
    v_remaining_m_s: float,
    t_burn_utc: Optional[str] = None,
    burn_lead_s: float = _DEFAULT_BURN_LEAD_S,
    a_ref_km: Optional[float] = None,
    policy: Optional[Dict[str, Any]] = None,
    conjunction_id: Optional[str] = None,
    satellite_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the /v1/evaluate request body for a parsed LeoLabs CDM.

    Parameters
    ----------
    parsed
        The ParsedLeoLabsCDM from parse_leolabs_cdm().
    sat_id
        Our satellite identifier for the evaluate request.
    v_remaining_m_s
        Remaining maneuver authority (m/s), an operator input.
    t_burn_utc
        Planned burn time (UTC, ISO-8601). Defaults to burn_lead_s before TCA;
        pass a real burn window for an operational run.
    burn_lead_s
        Seconds before TCA to place the default burn when t_burn_utc is omitted.
        The scorer requires the burn strictly before TCA. Default six hours.
    a_ref_km
        Reference semi-major axis (km). If omitted, APS 2.5 derives an osculating
        value from the satellite state (SCRUM-400).
    policy
        Operator policy dict. Defaults to _DEFAULT_POLICY.
    conjunction_id
        Identifier for this evaluate. Defaults to "leolabs-<event_id>".
    satellite_extra
        Optional extra satellite/capability fields merged into the satellite
        block (e.g. mass_kg, thrust parameters).

    Returns
    -------
    dict
        A request body ready to POST to /v1/evaluate.
    """
    prov = parsed.provenance
    if conjunction_id is None:
        event_id = prov.get("event_id") or prov.get("cdm_id") or "unknown"
        conjunction_id = f"leolabs-{event_id}"

    satellite: Dict[str, Any] = {
        "sat_id": sat_id,
        "r_sat_km": parsed.primary.r_km.tolist(),
        "v_sat_km_s": parsed.primary.v_km_s.tolist(),
        "t_burn_utc": t_burn_utc or _burn_before_tca(parsed.t_ca_utc, burn_lead_s),
        "v_remaining_m_s": float(v_remaining_m_s),
        # The asset's own radius, honestly reported. The scorer floors it per
        # the ADR-010 convention; a small primary is still screened at the floor.
        "radius_m": parsed.primary.radius_m,
    }
    if a_ref_km is not None:
        satellite["a_ref_km"] = float(a_ref_km)
    if satellite_extra:
        satellite.update(satellite_extra)

    # Start from the parser's ConjunctionState contract, then add the LeoLabs
    # provenance the evaluate path and audit trail want. source="leolabs" and a
    # real covariance trigger the AC7 bypass so the matrix is not overwritten.
    conjunction: Dict[str, Any] = dict(parsed.to_conjunction_state())
    conjunction["source"] = "leolabs"
    conjunction["covariance_source"] = "real_cdm"
    conjunction["miss_distance_km"] = (
        parsed.miss_distance_m / 1000.0 if parsed.miss_distance_m is not None else None
    )
    # Recorded for the audit trail (AC4). Not used to fetch covariance: the AC7
    # bypass skips the ingest adapter for a leolabs source that already carries
    # its covariance, so these NORAD ids never trigger an overwrite.
    if prov.get("primary_norad_id") is not None:
        conjunction["primary_norad"] = str(prov["primary_norad_id"])
    if prov.get("secondary_norad_id") is not None:
        conjunction["secondary_norad"] = str(prov["secondary_norad_id"])
    conjunction["cdm_id"] = prov.get("cdm_id")
    conjunction["event_id"] = prov.get("event_id")

    return {
        "conjunction_id": conjunction_id,
        "satellite": satellite,
        "conjunction": conjunction,
        "policy": dict(policy) if policy is not None else dict(_DEFAULT_POLICY),
    }
