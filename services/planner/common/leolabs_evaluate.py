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


def analyze_leolabs_portfolio(
    primary_norad: int, *, satellite=None, policy=None,
    now=None, lookback_days=1, lookahead_days=7,
    volume_filters=None, client=None, registry=None,
    processing_budget_s=25.0,
) -> Dict[str, Any]:
    """SCRUM-480: bounded, read-only comparison over one asset window.

    Fetch bounds and event deduplication are shared with the live list.
    The processing budget is checked between events; an individual scoring
    call is allowed to finish. Partial totals cover only analyzed events.
    """
    from time import perf_counter
    from common import leolabs_runtime as runtime
    from common.leolabs_cdm_parser import parse_leolabs_cdm, LeoLabsParseError
    from common.maneuver_scorer import analyze_conjunction_modes

    now = now or datetime.now(timezone.utc)
    satellite = dict(satellite or {})
    filters = dict(volume_filters or {})
    started = perf_counter()
    client = client or runtime.get_client()
    registry = registry or runtime.get_registry(client)

    catalog, ordered, cdm_count, pull = runtime._raw_cdms_in_risk_order(
        primary_norad, client, registry, now,
        lookback_days, lookahead_days, filters,
        deadline_s=runtime._LIST_FETCH_DEADLINE_S,
        max_cdms=runtime._LIST_FETCH_MAX_CDMS,
    )
    fetched = perf_counter()
    rows = []
    failures = []
    attempted = 0
    score_calls = 0
    processing_truncated = False

    for cdm in ordered:
        if perf_counter() - fetched >= processing_budget_s:
            processing_truncated = True
            break
        attempted += 1
        identity = {
            "cdm_id": cdm.get("CDM_ID"),
            "event_id": cdm.get("COMMENT_EVENT_ID"),
        }
        stage = "parse"
        try:
            our_id = registry.resolve_our_catalog_id(cdm)
            parsed = parse_leolabs_cdm(cdm, our_id)
            identity = {
                "cdm_id": parsed.provenance.get("cdm_id"),
                "event_id": parsed.provenance.get("event_id"),
            }
            stage = "score"
            req = build_evaluate_request(
                parsed,
                sat_id=str(satellite.get("sat_id", primary_norad)),
                v_remaining_m_s=float(satellite.get("v_remaining_m_s", 0.0)),
                t_burn_utc=satellite.get("t_burn_utc"),
                a_ref_km=satellite.get("a_ref_km"),
                policy=policy or None,
            )
            score_calls += 1
            row = analyze_conjunction_modes(req)
            row.update(identity)
            row["tca_utc"] = parsed.t_ca_utc
            row["secondary_norad"] = parsed.secondary.norad_id
            rows.append(row)
        except (LeoLabsParseError, LookupError, ValueError, TypeError) as exc:
            failures.append({
                **identity, "stage": stage, "reason": str(exc),
            })

    finished = perf_counter()
    summaries = {}
    for mode in ("aps", "flight_rule_1e4", "flight_rule_1e5"):
        verdicts = [row["modes"][mode] for row in rows]
        missing = sum(not verdict["pricing_available"] for verdict in verdicts)
        subtotal = sum(
            verdict["dv_m_s"] for verdict in verdicts
            if verdict["dv_m_s"] is not None
        )
        summaries[mode] = {
            "maneuver_count": sum(verdict["maneuver_required"] for verdict in verdicts),
            "total_dv_m_s": subtotal if missing == 0 else None,
            "known_dv_m_s": subtotal,
            "unpriced_maneuver_count": missing,
        }

    reasons = []
    if not pull.complete:
        reasons.append("fetch_" + (pull.reason or "incomplete"))
    if processing_truncated:
        reasons.append("processing_deadline")
    if failures:
        reasons.append("event_failures")
    min_tca, max_tca = runtime.conjunction_window(
        now, lookback_days, lookahead_days,
    )
    return {
        "primary_norad": int(primary_norad),
        "catalog_number": catalog,
        "source": "leolabs",
        "analysis_only": True,
        "dv_basis": "best_candidate_avoidance_burn",
        "complete": not reasons,
        "partial": bool(reasons),
        "partial_reasons": reasons,
        "count": len(rows),
        "events_in_fetched_set": len(ordered),
        "events_attempted": attempted,
        "events_not_attempted": len(ordered) - attempted,
        "cdm_count": cdm_count,
        "cdms_in_window": pull.window_total,
        "window": {
            "min_tca_utc": min_tca,
            "max_tca_utc": max_tca,
            "lookback_days": lookback_days,
            "lookahead_days": lookahead_days,
        },
        "volume_filters": filters,
        "modes": summaries,
        "conjunctions": rows,
        "failures": failures,
        "cost": {
            "fetch_ms": (fetched - started) * 1000.0,
            "processing_ms": (finished - fetched) * 1000.0,
            "total_ms": (finished - started) * 1000.0,
            "scoring_calls": score_calls,
            "processing_budget_s": processing_budget_s,
        },
        "analyzed_at_utc": now.isoformat().replace("+00:00", "Z"),
    }
