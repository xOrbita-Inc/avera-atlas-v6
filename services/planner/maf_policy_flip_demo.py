"""
SCRUM-383 MAF policy-flip integration demo.

Runs one deterministic collision-avoidance scenario through the assembled MAF
decision loop using the production planner, artifact, validity, authorization,
state-machine, safety-monitor, and evidence components.

Implemented here:
  1. Deterministic representative CA scenario derived from the existing
     AMBER-001 regression fixture.
  2. Same physical conjunction evaluated under Pc_action 1e-4 and 1e-5.
  3. L0, L1, and L2 authorization envelopes using SCRUM-375.
  4. Real SCRUM-378 validity assessment through the production validity seam.
  5. Real SCRUM-379 M0-M3 decision-state-machine and safety-monitor execution.
  6. L1 operator-approval and L2 veto-window execution authorization.
  7. Real SCRUM-377 decision and transition evidence records.
  8. Real SCRUM-382 GNCCommand construction and record-only emission.
  9. Constructed contract-valid GNCReport consumed by the real SCRUM-382
     post-burn assessment and evidence producers.

The demo has no live GNC endpoint. Authorized commands therefore use
SCRUM-382 record-only emission and no GNCCommandAck is fabricated. The
GNCReport is a constructed simulation fixture, not measured flight data.

SCRUM-377 provides a tamper-evident hash-chained evidence trail. SCRUM-375
provides HMAC-SHA256 authentication for the approved authorization envelope.
HMAC-SHA256 is a keyed MAC, not a digital signature.

All geometry and observation data in this demo are constructed integration
fixtures. They are not measured flight data.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

for path in (
    REPO_ROOT / "libs",
    REPO_ROOT / "services" / "validity",
    REPO_ROOT / "services" / "planner",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


from common.atlas_artifact import build_atlas_artifact
from common.authorization_envelope import (
    AuthorityLevel,
    AuthorizationEnvelopeManager,
    ManeuverCapacityInput,
    approve_authorization_envelope,
    compile_from_operator_policy,
    verify_approved_envelope,
)
from common.decision_state_machine import FlightMode
from common.evidence_record import (
    GENESIS_HASH,
    build_decision_record,
    build_transition_record,
    verify_chain,
)
from common.logging_setup import SERVICE_VERSION
from aps_math import conventions
from common import gnc_client
from common.gnc_contract import load_gnc_schema, validate_against_contract
from common.gnc_report import assess_post_burn, build_report_ack
import server as planner_server
from common.maneuver_scorer import evaluate_conjunction_v25, _policy_from_dict
from common.monitor_adapter import (
    evaluate_request as evaluate_decision_state_machine,
)
from common.satellite_capability import SatelliteCapability


EVENTS_PATH = (
    Path(__file__).resolve().parent
    / "tests"
    / "synthetic_conjunction_demo_events.json"
)

SOURCE_SCENARIO_ID = "AMBER-001"
SCENARIO_ID = "SCRUM-383-POLICY-FLIP"
DEMO_TCA_UTC = "2026-03-02T18:00:00Z"

BASE_PC_ACTION = 1.0e-4
STRICT_PC_ACTION = 1.0e-5

# The monitor runs six hours before TCA and two hours before the constructed
# burn, comfortably inside the first-flight 4-72 hour TCA decision horizon.
DEMO_DECISION_DT = datetime(
    2026, 3, 2, 12, 0, 0, tzinfo=timezone.utc
)
DEMO_DECISION_TIME = "2026-03-02T12:00:00Z"

DEMO_APPROVAL_TIME = datetime(
    2026, 3, 2, 11, 50, 0, tzinfo=timezone.utc
)

# Constructed demo credential only. This is not a production secret.
DEMO_MAC_KEY = b"scrum-383-demo-envelope-key"
DEMO_KEY_ID = "scrum-383-demo-key"
DEMO_APPROVER = "operator:scrum-383-demo"


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_scenario() -> Dict[str, Any]:
    """Build the deterministic SCRUM-383 CA fixture from AMBER-001."""

    with EVENTS_PATH.open("r", encoding="utf-8") as handle:
        events = json.load(handle)

    for event in events:
        if event["conjunction_id"] != SOURCE_SCENARIO_ID:
            continue

        scenario = copy.deepcopy(event)
        scenario["conjunction_id"] = SCENARIO_ID
        scenario["conjunction"]["obj_id"] = "OBJ-SCRUM-383-POLICY-FLIP"
        scenario["conjunction"]["t_ca_utc"] = DEMO_TCA_UTC

        # The existing SCRUM-383 fixture defines this Pc explicitly. Carry it
        # through the v2.5 production scorer as a supplied Pc rather than asking
        # a different geometry path to reproduce the fixture value.
        scenario["conjunction"]["pc_precomputed"] = float(scenario["Pc"])
        scenario["conjunction"]["covariance_source"] = "real_cdm"
        scenario["conjunction"]["data_age_s"] = 3600.0

        scenario["scenario"] = (
            "Constructed SCRUM-383 policy-flip case derived from "
            "AMBER-001 geometry at the locked 4-hour burn-to-TCA horizon"
        )
        return scenario

    raise RuntimeError(
        f"{SOURCE_SCENARIO_ID} was not found in {EVENTS_PATH}"
    )


def _monitor_validity_block() -> Dict[str, Any]:
    """Construct the observation arc consumed by the real SCRUM-378 seam."""

    n = 8
    observations = []

    for i in range(n):
        observations.append(
            {
                "epoch_utc": _iso(
                    DEMO_DECISION_DT
                    - timedelta(minutes=5 * (n - i))
                ),
                "r_observer_km": [
                    6378.0,
                    100.0 * i,
                    50.0 * i,
                ],
                "ra_sigma_rad": 1.0e-4,
                "dec_sigma_rad": 1.0e-4,
                "range_sigma_km": 0.001,
            }
        )

    return {
        "service_available": True,
        "target_state": {
            "epoch_utc": _iso(DEMO_DECISION_DT),
            "r_km": [7000.0, 0.0, 0.0],
            "v_km_s": [0.0, 7.546, 0.0],
        },
        "observations": observations,
        "r_rel_km_at_tca": [0.5, -0.3, 0.1],
        "v_rel_km_s_at_tca": [0.0, 0.01, 0.0],
        "phenomenologies_used": ["TLE"],
    }


def _capacity_input(cap: SatelliteCapability) -> ManeuverCapacityInput:
    lifetime = getattr(cap, "lifetime", None)
    propulsion = getattr(cap, "propulsion", None)

    max_dv_per_burn_m_s = float(
        getattr(propulsion, "max_dv_per_burn_m_s", 0.0) or 0.0
    )
    v_remaining_m_s = float(
        getattr(lifetime, "v_remaining_m_s", 0.0) or 0.0
    )
    v_reserved_m_s = float(
        getattr(lifetime, "v_reserved_m_s", 5.0) or 5.0
    )

    return ManeuverCapacityInput(
        max_dv_per_burn_m_s=max_dv_per_burn_m_s,
        v_remaining_m_s=v_remaining_m_s,
        v_reserved_m_s=v_reserved_m_s,
        max_maneuvers_per_week=3,
    )


def _compile_and_activate_envelope(
    *,
    policy: Any,
    cap: SatelliteCapability,
    authority_level: AuthorityLevel,
):
    """Compile, approve, verify, and activate a real SCRUM-375 envelope."""

    profile_id = (
        f"scrum-383-{authority_level.value.lower()}-policy"
    )

    envelope = compile_from_operator_policy(
        policy,
        profile_id=profile_id,
        mission_class="first_flight_leo",
        risk_budget_id="scrum-383-demo-risk",
        maneuver_capacity=_capacity_input(cap),
        authority_level=authority_level,
        l2_raise_to_1_m_s=False,
    )

    approved = approve_authorization_envelope(
        envelope,
        approving_identity=DEMO_APPROVER,
        mac_key=DEMO_MAC_KEY,
        key_id=DEMO_KEY_ID,
        human_review_confirmed=True,
        approved_at=DEMO_APPROVAL_TIME,
    )

    mac_verified = verify_approved_envelope(
        approved,
        mac_key=DEMO_MAC_KEY,
        expected_key_id=DEMO_KEY_ID,
    )

    manager = AuthorizationEnvelopeManager()
    manager.stage(envelope)
    manager.record_approval(
        approved,
        mac_key=DEMO_MAC_KEY,
    )
    manager.activate(
        activation_time=DEMO_APPROVAL_TIME,
    )

    effective_authority = manager.effective_authority(
        AuthorityLevel.L2,
        at_time=DEMO_APPROVAL_TIME,
    )

    return envelope, approved, mac_verified, effective_authority


def _distant_catalog() -> List[Dict[str, Any]]:
    """One intentionally distant object so SCRUM-381 actually runs CLEAR."""

    return [
        {
            "obj_id": "SCRUM-383-FAR-OBJECT",
            "r_km": [0.0, 0.0, 8000.0],
            "v_km_s": [0.0, 7.06, 0.0],
            "position_sigma_m": 100.0,
        }
    ]


def _build_real_artifact(
    *,
    scenario: Dict[str, Any],
    scoring: Any,
    cap: SatelliteCapability,
    policy: Any,
):
    sat = scenario["satellite"]
    conj = scenario["conjunction"]

    r_post_km = sat.get("r_sat_km")
    v_post_km_s = None

    v_sat = sat.get("v_sat_km_s")
    if v_sat and scoring.dv_eci_km_s:
        v_post_km_s = [
            float(v_sat[i]) + float(scoring.dv_eci_km_s[i])
            for i in range(3)
        ]

    return build_atlas_artifact(
        scoring=scoring,
        cap=cap,
        policy=policy,
        tca_utc=conj["t_ca_utc"],
        pc_precomputed=conj.get("pc_precomputed"),
        miss_distance_km=conj.get("miss_distance_km"),
        known_objects=_distant_catalog(),
        r_post_km=r_post_km,
        v_post_km_s=v_post_km_s,
    )


def _monitor_body(
    scenario: Dict[str, Any],
    authority_level: AuthorityLevel,
) -> Dict[str, Any]:
    body = copy.deepcopy(scenario)

    body["iod"] = {
        "proceeds_to_validity": True,
        "confidence_verdict": "CONFIDENT",
    }

    body["validity"] = _monitor_validity_block()

    body["authorization"] = {
        "authority_level": authority_level.value,
        "profile_id": (
            f"scrum-383-{authority_level.value.lower()}-policy"
        ),
        "mission_class": "first_flight_leo",
        "risk_budget_id": "scrum-383-demo-risk",
        "approving_identity": DEMO_APPROVER,
    }

    body["monitor"] = {}
    return body


def _run_state_machine(
    *,
    scenario: Dict[str, Any],
    scoring: Any,
    artifact: Any,
    policy: Any,
    cap: SatelliteCapability,
    authority_level: AuthorityLevel,
) -> List[Any]:
    """Drive one case through the real SCRUM-379 monitor."""

    body = _monitor_body(scenario, authority_level)
    decisions: List[Any] = []

    # M0 -> M1 when the event crosses the watch threshold.
    first = evaluate_decision_state_machine(
        body=body,
        scoring=scoring,
        artifact=artifact,
        policy=policy,
        cap=cap,
        covariance_source="real_cdm",
        current_mode=FlightMode.M0_NOMINAL,
        t_now_utc=DEMO_DECISION_DT,
        software_version=SERVICE_VERSION,
    )
    decisions.append(first)

    if first.mode is not FlightMode.M1_WATCH:
        return decisions

    # M1 either stages a valid actionable maneuver or holds watch.
    second = evaluate_decision_state_machine(
        body=body,
        scoring=scoring,
        artifact=artifact,
        policy=policy,
        cap=cap,
        covariance_source="real_cdm",
        current_mode=FlightMode.M1_WATCH,
        t_now_utc=DEMO_DECISION_DT + timedelta(seconds=1),
        software_version=SERVICE_VERSION,
    )
    decisions.append(second)

    if second.mode is not FlightMode.M2_STAGED:
        return decisions

    # M2 -> M3 differs by authority level.
    if authority_level is AuthorityLevel.L1:
        body["monitor"] = {
            "approval_command_received": True,
            "approval_accepted": True,
        }
    elif authority_level is AuthorityLevel.L2:
        body["monitor"] = {
            "veto_window_close_utc": _iso(
                DEMO_DECISION_DT + timedelta(seconds=1)
            ),
            "veto_command_received": False,
        }
    else:
        return decisions

    third = evaluate_decision_state_machine(
        body=body,
        scoring=scoring,
        artifact=artifact,
        policy=policy,
        cap=cap,
        covariance_source="real_cdm",
        current_mode=FlightMode.M2_STAGED,
        t_now_utc=DEMO_DECISION_DT + timedelta(seconds=2),
        software_version=SERVICE_VERSION,
    )
    decisions.append(third)

    return decisions




def _contract_fixture_value(
    schema: Dict[str, Any],
    field_name: str,
) -> Any:
    """Minimal constructed value for a required GNC contract field."""

    if "default" in schema:
        return copy.deepcopy(schema["default"])

    enum = schema.get("enum")
    if enum:
        return enum[0]

    schema_type = schema.get("type")

    if schema_type == "string":
        if schema.get("format") == "date-time" or field_name.endswith("_utc"):
            return _iso(DEMO_DECISION_DT + timedelta(seconds=10))
        return f"scrum-383-{field_name}"

    if schema_type == "number":
        return 0.0

    if schema_type == "integer":
        return 0

    if schema_type == "boolean":
        return False

    if schema_type == "array":
        item_schema = schema.get("items", {})
        count = int(schema.get("minItems", 0) or 0)
        return [
            _contract_fixture_value(item_schema, field_name)
            for _ in range(count)
        ]

    if schema_type == "object":
        properties = schema.get("properties", {})
        return {
            name: _contract_fixture_value(
                properties.get(name, {}),
                name,
            )
            for name in schema.get("required", [])
        }

    return None


def _contract_block(
    schema_name: str,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Build and validate one constructed block from gnc_interface.yaml."""

    schema = load_gnc_schema(schema_name)
    properties = schema.get("properties", {})

    block: Dict[str, Any] = {}

    for field_name in schema.get("required", []):
        if field_name in overrides:
            block[field_name] = copy.deepcopy(
                overrides[field_name]
            )
        else:
            block[field_name] = _contract_fixture_value(
                properties.get(field_name, {}),
                field_name,
            )

    for field_name, value in overrides.items():
        if field_name in properties:
            block[field_name] = copy.deepcopy(value)

    validate_against_contract(schema_name, block)
    return block


def _real_post_burn_assessment(
    *,
    report: Dict[str, Any],
    scenario: Dict[str, Any],
    scoring: Any,
    pc_action: float,
) -> Any:
    """Feed the constructed report through SCRUM-382 assess_post_burn."""

    conj = scenario["conjunction"]

    available = {
        "conjunction_id": scenario["conjunction_id"],
        "pc_pre": scoring.pc_pre,
        "pc_precomputed": scoring.pc_pre,
        "r_rel_km": conj.get("r_rel_km"),
        "r_rel_km_at_tca": conj.get("r_rel_km"),
        "v_rel_km_s": conj.get("v_rel_km_s"),
        "v_rel_km_s_at_tca": conj.get("v_rel_km_s"),
        "p_pre_km2": conj.get("p_rel_km2"),
        "p_rel_km2": conj.get("p_rel_km2"),
        "p_rel_pre_km2": conj.get("p_rel_km2"),
        "hbr_m": float(
            conventions.DEFAULT_COMBINED_HBR_M
        ),
        "pc_monitor_threshold": pc_action / 10.0,
        "pc_action_threshold": pc_action,
        "pc_maneuver_threshold": pc_action,
    }

    signature = inspect.signature(assess_post_burn)
    kwargs: Dict[str, Any] = {}

    # First positional argument is the GNC report itself.
    parameters = list(signature.parameters.items())[1:]

    for name, parameter in parameters:
        if name in available:
            kwargs[name] = available[name]
        elif parameter.default is inspect._empty:
            raise RuntimeError(
                "SCRUM-383 missing required assess_post_burn "
                f"input: {name}"
            )

    return assess_post_burn(report, **kwargs)


def _run_scrum_382(
    *,
    scenario: Dict[str, Any],
    scoring: Any,
    artifact: Any,
    decisions: List[Any],
    pc_action: float,
) -> Optional[Dict[str, Any]]:
    """Run real SCRUM-382 command + report processing for an authorized burn."""

    final_monitor = decisions[-1]

    if final_monitor.authorized_execution is None:
        return None

    result: Dict[str, Any] = {}

    # The capstone has no external GNC service. Use 382's real
    # record-only adapter instead of manufacturing a fake endpoint/ack.
    old_enabled = gnc_client.GNC_ENABLED
    old_url = gnc_client.GNC_SERVICE_URL

    try:
        gnc_client.GNC_ENABLED = True
        gnc_client.GNC_SERVICE_URL = ""

        planner_server._emit_to_gnc(
            result,
            final_monitor,
            scoring,
            artifact,
            scenario["satellite"],
        )
    finally:
        gnc_client.GNC_ENABLED = old_enabled
        gnc_client.GNC_SERVICE_URL = old_url

    command = result["gnc_command"]
    emission = result["gnc_emission"]

    validate_against_contract(
        "GNCCommand",
        command,
    )

    if emission["mode"] != gnc_client.MODE_RECORD_ONLY:
        raise RuntimeError(
            "SCRUM-383 expected record-only GNC emission"
        )

    if emission["acknowledged"]:
        raise RuntimeError(
            "record-only mode must not fabricate a GNCCommandAck"
        )

    maneuver = command["maneuver"]
    sat = scenario["satellite"]
    conj = scenario["conjunction"]

    # Constructed perfect-execution report for the simulation.
    # These are not measured flight values.
    r_post_km = [
        float(v)
        for v in sat["r_sat_km"]
    ]

    v_post_km_s = [
        float(sat["v_sat_km_s"][i])
        + float(scoring.dv_eci_km_s[i])
        for i in range(3)
    ]

    execution_error = _contract_block(
        "ExecutionError",
        {
            "sigma_mag": 0.0,
            "pointing_error_deg": 0.0,
            "p_burn_rtn_km2": [0.0] * 9,
        },
    )

    post_burn_state = _contract_block(
        "PostBurnState",
        {
            "r_sat_km": r_post_km,
            "v_sat_km_s": v_post_km_s,
            "p_post_km2": [
                float(v)
                for v in conj["p_rel_km2"]
            ],
            "epoch_utc": _iso(
                DEMO_DECISION_DT
                + timedelta(seconds=5)
            ),
            "m2_post_estimated": (
                float(scoring.m2_pre) + 1.0
            ),
        },
    )

    report = _contract_block(
        "GNCReport",
        {
            "command_id": command["command_id"],
            "conjunction_id": scenario["conjunction_id"],
            "execution_status": "NOMINAL",
            "actual_dv_m_s": float(
                maneuver["dv_magnitude_m_s"]
            ),
            "actual_dv_rtn_m_s": list(
                maneuver["dv_rtn_m_s"]
            ),
            "attitude_error_deg": 0.0,
            "execution_error": execution_error,
            "post_burn_state": post_burn_state,
            "reported_at_utc": _iso(
                DEMO_DECISION_DT
                + timedelta(seconds=10)
            ),
        },
    )

    assessment = _real_post_burn_assessment(
        report=report,
        scenario=scenario,
        scoring=scoring,
        pc_action=pc_action,
    )

    report_ack = build_report_ack(assessment)

    validate_against_contract(
        "GNCReportAck",
        report_ack,
    )

    evidence_values = {
        "commands_and_acknowledgments": {
            "command": command,
            "emission": emission,
            "command_acknowledgment": None,
            "report_acknowledgment": report_ack,
        },
        **assessment.evidence_values(),
    }

    return {
        "command": command,
        "emission": emission,
        "command_ack": None,
        "report": report,
        "report_ack": report_ack,
        "assessment": {
            "execution_status": (
                assessment.execution_status
            ),
            "pc_post": assessment.pc_post,
            "replan_required": (
                assessment.replan_required
            ),
        },
        "evidence_values": evidence_values,
    }



def _transition_extra_values(
    *,
    artifact: Any,
    monitor: Any,
    policy_version: str,
    operator_id: str,
) -> Dict[str, Any]:
    """Mirror the production SCRUM-379 transition-evidence mapping."""

    inputs = monitor.inputs

    values: Dict[str, Any] = {
        "timestamp": monitor.evaluated_at_utc,
        "monitor_results": [
            guard.to_dict()
            for guard in monitor.guards
        ],
        "policy_version": policy_version,
        "model_version": SERVICE_VERSION,
        "inputs_and_provenance": {
            "sat_id": artifact.sat_id,
            "operator_id": operator_id,
            "evaluated_at": artifact.evaluated_at,
            "pc_source": inputs.pc_source,
            "covariance_source": inputs.covariance_source,
            "requested_mode": (
                monitor.transition.requested_mode.value
            ),
            "escalated": monitor.transition.escalated,
        },
    }

    for name in (
        "validity_status",
        "validity_epsilon",
        "epsilon_threshold",
        "phenomenologies_used",
        "weak_directions",
    ):
        if name in inputs.validity_evidence:
            values[name] = inputs.validity_evidence[name]

    if inputs.authority_level:
        values["authority_level"] = inputs.authority()

    if inputs.envelope_version:
        values["envelope_version"] = inputs.envelope_version

    if inputs.envelope_approving_identity:
        values["envelope_approving_identity"] = (
            inputs.envelope_approving_identity
        )

    return values


def _build_evidence_package(
    *,
    scenario: Dict[str, Any],
    scoring: Any,
    artifact: Any,
    envelope: Any,
    approved: Any,
    decisions: List[Any],
    authority_level: AuthorityLevel,
    pc_action: float,
    gnc_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build one chained decision + transition evidence package."""

    final_monitor = decisions[-1]
    validity_evidence = dict(
        final_monitor.inputs.validity_evidence
        if final_monitor.inputs is not None
        else {}
    )

    policy_version = (
        f"scrum-383-demo-pc-action-{pc_action:.0e}"
    )

    evidence_values: Dict[str, Any] = {
        "conjunction_id": scenario["conjunction_id"],
        "timestamp": DEMO_DECISION_TIME,
        "pc_at_transition": scoring.pc_pre,
        "software_version": SERVICE_VERSION,
        "model_version": SERVICE_VERSION,
        "policy_version": policy_version,
        "inputs_and_provenance": {
            "scenario_source": str(
                EVENTS_PATH.relative_to(REPO_ROOT)
            ).replace("\\", "/"),
            "scenario_id": SCENARIO_ID,
            "source_scenario_id": SOURCE_SCENARIO_ID,
            "scenario_horizon": (
                "constructed at locked 4-hour minimum "
                "from burn to TCA"
            ),
            "pc_action": pc_action,
            "pc_source": scoring.pc_source,
            "covariance_source": "real_cdm",
            "integration_scope": (
                "SCRUM-383 end-to-end simulated MAF loop through "
                "SCRUM-382 command emission and post-burn assessment"
            ),
        },
        "orbit_state": {
            "r_sat_km": scenario["satellite"]["r_sat_km"],
            "v_sat_km_s": scenario["satellite"]["v_sat_km_s"],
            "t_burn_utc": scenario["satellite"]["t_burn_utc"],
            "t_ca_utc": scenario["conjunction"]["t_ca_utc"],
            "r_rel_km": scenario["conjunction"]["r_rel_km"],
            "v_rel_km_s": scenario["conjunction"].get(
                "v_rel_km_s"
            ),
        },
        "covariance_state": {
            "p_rel_km2": scenario["conjunction"]["p_rel_km2"],
        },
        "candidate_maneuvers": scoring.all_candidates,
    }

    evidence_values.update(validity_evidence)
    evidence_values.update(
        envelope.evidence_fields(
            approved.approving_identity
        )
    )

    chain_id = (
        f"SCRUM-383-{authority_level.value}-"
        f"{pc_action:.0e}"
    )

    decision_record = build_decision_record(
        chain_id=chain_id,
        seq=0,
        prev_hash=GENESIS_HASH,
        values=evidence_values,
        recorded_at=DEMO_DECISION_TIME,
    )

    records = [decision_record]

    for monitor in decisions:
        if not monitor.changed_mode and not monitor.escalated:
            continue

        transition_record = build_transition_record(
            chain_id=chain_id,
            seq=len(records),
            prev_hash=records[-1].content_hash,
            from_mode=monitor.transition.from_mode.value,
            to_mode=monitor.transition.to_mode.value,
            trigger=monitor.transition.trigger,
            conjunction_id=monitor.conjunction_id,
            software_version=SERVICE_VERSION,
            pc_at_transition=monitor.inputs.pc,
            extra_values=_transition_extra_values(
                artifact=artifact,
                monitor=monitor,
                policy_version=policy_version,
                operator_id=str(
                    scenario.get("policy", {}).get("operator_id", "")
                ),
            ),
            recorded_at=monitor.evaluated_at_utc,
        )
        records.append(transition_record)

    # SCRUM-382 post-burn outcome is a later event in the evidence chain,
    # not something retroactively attached to the earlier M2 -> M3 record.
    if gnc_result is not None:
        post_values = dict(evidence_values)

        post_values.update(
            gnc_result["evidence_values"]
        )

        final_monitor = decisions[-1]

        post_values.update(
            {
                "timestamp": (
                    gnc_result["report"]["reported_at_utc"]
                ),
                "from_mode": (
                    final_monitor.transition.from_mode.value
                ),
                "to_mode": final_monitor.mode.value,
                "trigger": (
                    "gnc_report_consumed_post_burn"
                ),
                "monitor_results": [
                    guard.to_dict()
                    for guard in final_monitor.guards
                ],
            }
        )

        post_record = build_decision_record(
            chain_id=chain_id,
            seq=len(records),
            prev_hash=records[-1].content_hash,
            values=post_values,
            recorded_at=(
                gnc_result["report"]["reported_at_utc"]
            ),
        )

        records.append(post_record)

    verification = verify_chain(records)

    serialized = [
        json.loads(record.to_json())
        for record in records
    ]

    final_record = records[-1]
    final_blob = serialized[-1]

    return {
        "record_id": decision_record.record_id,
        "content_hash": final_record.content_hash,
        "hash_matches": all(
            record.hash_matches()
            for record in records
        ),
        "chain_verified": verification.ok,
        "record_count": len(records),
        "transition_record_ids": [
            record.record_id
            for record in records[1:]
        ],
        "pending_producers": sorted(
            final_record.pending_producers()
        ),
        "fields": final_blob["fields"],
        "records": serialized,
    }


def _validity_summary(decisions: List[Any]) -> Dict[str, Any]:
    evidence = dict(
        decisions[-1].inputs.validity_evidence
        if decisions[-1].inputs is not None
        else {}
    )

    return {
        "status": evidence.get("validity_status"),
        "epsilon": evidence.get("validity_epsilon"),
        "epsilon_threshold": evidence.get(
            "epsilon_threshold"
        ),
        "weak_directions": evidence.get(
            "weak_directions",
            [],
        ),
        "phenomenologies_used": evidence.get(
            "phenomenologies_used",
            [],
        ),
    }


def run_case(
    pc_action: float,
    authority_level: AuthorityLevel,
) -> Dict[str, Any]:
    """Run one SCRUM-383 CA case through every available real producer."""

    scenario = _load_scenario()

    # Preserve every physical conjunction input. Only operator policy changes.
    scenario["policy"]["pc_maneuver_threshold"] = pc_action
    scenario["policy"]["pc_monitor_threshold"] = pc_action / 10.0
    scenario["policy"]["policy_version"] = (
        f"scrum-383-demo-pc-action-{pc_action:.0e}"
    )

    scoring = evaluate_conjunction_v25(scenario)
    cap = SatelliteCapability.from_request(
        scenario["satellite"]
    )
    policy = _policy_from_dict(
        scenario["policy"]
    )

    artifact = _build_real_artifact(
        scenario=scenario,
        scoring=scoring,
        cap=cap,
        policy=policy,
    )

    (
        envelope,
        approved,
        mac_verified,
        effective_authority,
    ) = _compile_and_activate_envelope(
        policy=policy,
        cap=cap,
        authority_level=authority_level,
    )

    decisions = _run_state_machine(
        scenario=scenario,
        scoring=scoring,
        artifact=artifact,
        policy=policy,
        cap=cap,
        authority_level=authority_level,
    )

    authorized_execution = None
    for decision in decisions:
        if decision.authorized_execution is not None:
            authorized_execution = (
                decision.authorized_execution.to_dict()
            )

    gnc_result = _run_scrum_382(
        scenario=scenario,
        scoring=scoring,
        artifact=artifact,
        decisions=decisions,
        pc_action=pc_action,
    )

    evidence = _build_evidence_package(
        scenario=scenario,
        scoring=scoring,
        artifact=artifact,
        envelope=envelope,
        approved=approved,
        decisions=decisions,
        authority_level=authority_level,
        pc_action=pc_action,
        gnc_result=gnc_result,
    )

    return {
        "scenario": {
            "conjunction_id": scenario["conjunction_id"],
            "pc": float(scenario["Pc"]),
            "pc_basis": scenario.get("pc_basis"),
        },
        "physical_input": {
            "satellite": copy.deepcopy(
                scenario["satellite"]
            ),
            "conjunction": copy.deepcopy(
                scenario["conjunction"]
            ),
        },
        "policy": {
            "pc_action": pc_action,
            "pc_watch": pc_action / 10.0,
            "authority_level": authority_level.value,
        },
        "planner": {
            "direction": scoring.direction,
            "dv_magnitude_m_s": scoring.dv_magnitude_m_s,
            "utility": scoring.utility,
            "pc_pre": scoring.pc_pre,
            "pc_source": scoring.pc_source,
        },
        "validity": _validity_summary(decisions),
        "authorization_envelope": {
            "authority_level": envelope.authority_level.value,
            "effective_authority": effective_authority.value,
            "envelope_version": envelope.version,
            "alg": approved.alg,
            "key_id": approved.key_id,
            "mac_verified": mac_verified,
        },
        "state_machine": {
            "steps": [
                decision.to_dict()
                for decision in decisions
            ],
            "final_mode": decisions[-1].mode.value,
            "authorized_execution": authorized_execution,
        },
        "gnc": gnc_result,
        "evidence": evidence,
    }


def run_demo() -> Dict[str, Any]:
    """Run the L0-L2 loose/strict policy matrix."""

    cases = []

    for authority_level in (
        AuthorityLevel.L0,
        AuthorityLevel.L1,
        AuthorityLevel.L2,
    ):
        for pc_action in (
            BASE_PC_ACTION,
            STRICT_PC_ACTION,
        ):
            cases.append(
                run_case(
                    pc_action=pc_action,
                    authority_level=authority_level,
                )
            )

    return {
        "story": "SCRUM-383",
        "scope": "end-to-end-sim",
        "software_version": SERVICE_VERSION,
        "downstream_dependency": None,
        "cases": cases,
    }


def _case(
    result: Dict[str, Any],
    authority: str,
    pc_action: float,
) -> Dict[str, Any]:
    for case in result["cases"]:
        if (
            case["policy"]["authority_level"] == authority
            and case["policy"]["pc_action"] == pc_action
        ):
            return case

    raise KeyError((authority, pc_action))


def _print_summary(result: Dict[str, Any]) -> None:
    print("SCRUM-383 MAF policy-flip demo")
    print(
        f"Planner software version: "
        f"{result['software_version']}"
    )
    print()

    first = result["cases"][0]
    print(
        f"Scenario: "
        f"{first['scenario']['conjunction_id']}"
    )
    print(
        f"Fixed scenario Pc: "
        f"{first['scenario']['pc']:.6g}"
    )
    print()

    for case in result["cases"]:
        policy = case["policy"]
        planner = case["planner"]
        validity = case["validity"]
        envelope = case["authorization_envelope"]
        state_machine = case["state_machine"]
        evidence = case["evidence"]

        execution = state_machine["authorized_execution"]
        approval_basis = (
            execution["approval_basis"]
            if execution is not None
            else "N/A"
        )

        print(
            f"{policy['authority_level']} "
            f"Pc_action={policy['pc_action']:.0e} "
            f"decision={planner['direction']} "
            f"dv={planner['dv_magnitude_m_s']:.3f} m/s "
            f"validity={validity['status']} "
            f"mode={state_machine['final_mode']} "
            f"approval_basis={approval_basis} "
            f"MAC={'VERIFIED' if envelope['mac_verified'] else 'FAILED'} "
            f"evidence_chain="
            f"{'VERIFIED' if evidence['chain_verified'] else 'FAILED'}"
        )

    print()

    for authority in ("L0", "L1", "L2"):
        base = _case(
            result,
            authority,
            BASE_PC_ACTION,
        )
        strict = _case(
            result,
            authority,
            STRICT_PC_ACTION,
        )

        flipped = (
            base["planner"]["direction"]
            != strict["planner"]["direction"]
        )

        print(
            f"{authority} policy decision flipped: "
            f"{flipped}"
        )

    print()
    print(
        "Scope note: the simulated MAF loop is exercised through "
        "SCRUM-382 GNCCommand record-only emission and a constructed "
        "contract-valid GNCReport consumed by the real post-burn "
        "assessment. No live GNC acknowledgement is fabricated."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the complete machine-readable demo result.",
    )

    args = parser.parse_args()
    result = run_demo()

    if args.json:
        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_summary(result)


if __name__ == "__main__":
    main()
