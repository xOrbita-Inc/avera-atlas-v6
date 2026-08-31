"""
SCRUM-383 partial MAF policy-flip integration demo.

This is intentionally a partial capstone while SCRUM-379, SCRUM-380, and
SCRUM-382 runtime dependencies are not available on main.

Implemented here:
  1. Deterministic representative CA scenario using the existing AMBER-001
     planner regression fixture.
  2. Same physical conjunction evaluated under Pc_action 1e-4 and 1e-5.
  3. L0, L1, and L2 authorization envelopes using SCRUM-375.
  4. Real SCRUM-378 validity assessment over the existing constructed
     validity wiring fixture.
  5. Real SCRUM-377 tamper-evident evidence records.

Not implemented here:
  * SCRUM-379 runtime M0-M4 state-machine execution.
  * SCRUM-380 behavior beyond safety-floor enforcement already consumed by
    the authorization-envelope compiler.
  * SCRUM-382 APS-to-GNC command, approval, veto, execution, or report flow.
  * Additional evidence-package signing. SCRUM-377 provides the hash-chained
    evidence record; SCRUM-375 provides HMAC-SHA256 authentication for the
    approved authorization envelope.

The validity observation arc below is a constructed simulation fixture. It is
not measured flight data and is not claimed to be physically exact.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

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


from avoid.decision_model import evaluate_conjunction
from common.authorization_envelope import (
    AuthorityLevel,
    AuthorizationEnvelopeManager,
    ManeuverCapacityInput,
    approve_authorization_envelope,
    compile_from_operator_policy,
    verify_approved_envelope,
)
from common.evidence_record import (
    GENESIS_HASH,
    build_decision_record,
    verify_chain,
)
from common.logging_setup import SERVICE_VERSION
from wrapper import (
    ObservationEpochState,
    build_validity_verdict_for_arc,
    validity_evidence_values,
)


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

DEMO_APPROVAL_TIME = datetime(
    2026,
    3,
    2,
    13,
    50,
    0,
    tzinfo=timezone.utc,
)

DEMO_DECISION_TIME = "2026-03-02T13:55:00Z"

# Constructed demo credential only. This is not a production secret.
DEMO_MAC_KEY = b"scrum-383-demo-envelope-key"
DEMO_KEY_ID = "scrum-383-demo-key"
DEMO_APPROVER = "operator:scrum-383-demo"


def _load_scenario() -> Dict[str, Any]:
    """Build the deterministic SCRUM-383 CA fixture from AMBER-001 geometry.

    The relative geometry and covariance come from the existing AMBER-001
    regression fixture. SCRUM-383 assigns a TCA exactly four hours after the
    burn so the constructed demo lies on the locked first-flight minimum
    planning horizon. This is a synthetic integration fixture, not a measured
    conjunction event.
    """

    with EVENTS_PATH.open("r", encoding="utf-8") as handle:
        events = json.load(handle)

    for event in events:
        if event["conjunction_id"] == SOURCE_SCENARIO_ID:
            scenario = copy.deepcopy(event)
            scenario["conjunction_id"] = SCENARIO_ID
            scenario["conjunction"]["obj_id"] = "OBJ-SCRUM-383-POLICY-FLIP"
            scenario["conjunction"]["t_ca_utc"] = DEMO_TCA_UTC
            scenario["scenario"] = (
                "Constructed SCRUM-383 policy-flip case derived from "
                "AMBER-001 geometry at the locked 4-hour minimum horizon"
            )
            return scenario

    raise RuntimeError(
        f"{SOURCE_SCENARIO_ID} was not found in {EVENTS_PATH}"
    )


def _build_validity_verdict():
    """Run the real SCRUM-378 assessor over its constructed wiring geometry."""

    a_km = 7000.0
    r_tca = np.array([7000.0, 0.0, 0.0])
    v_tca = np.array([0.0, 7.5, 0.0])

    observations = []

    for i in range(3):
        observations.append(
            ObservationEpochState(
                r_target_km=r_tca
                + np.array(
                    [
                        -5.0 * (i + 1),
                        -20.0 * (i + 1),
                        0.0,
                    ]
                ),
                v_target_km_s=v_tca
                + np.array(
                    [
                        0.02 * (i + 1),
                        0.0,
                        0.0,
                    ]
                ),
                dt_s=-600.0 + i * 200.0,
                r_observer_km=np.array(
                    [
                        6378.0,
                        50.0 * i,
                        0.0,
                    ]
                ),
                ra_sigma_rad=1.0e-5,
                dec_sigma_rad=1.0e-5,
                range_sigma_km=0.01,
            )
        )

    return build_validity_verdict_for_arc(
        r_target_tca_km=r_tca,
        v_target_tca_km_s=v_tca,
        a_km=a_km,
        observation_epochs=observations,
        r_rel_km_at_tca=np.array([0.5, -0.3, 0.1]),
        v_rel_km_s_at_tca=np.array([0.0, 0.01, 0.0]),
        epsilon_threshold=0.20,
    )


def _operator_policy_for_envelope(pc_action: float) -> SimpleNamespace:
    """Build the existing planner-policy surface consumed by SCRUM-375."""

    return SimpleNamespace(
        operator_id="SCRUM-383-DEMO",
        pc_maneuver_threshold=pc_action,
        pc_monitor_threshold=pc_action / 10.0,
        min_miss_distance_km=1.0,
        max_dv_per_event_ms=2.0,
        max_maneuvers_per_week=3,
        min_hours_before_tca=4.0,
        max_hours_before_tca=72.0,
    )


def _compile_and_activate_envelope(
    pc_action: float,
    authority_level: AuthorityLevel,
    v_remaining_m_s: float,
):
    """Compile, approve, verify, and activate a real SCRUM-375 envelope."""

    policy = _operator_policy_for_envelope(pc_action)

    # Keep profile identity fixed across the Pc policy comparison so a
    # different envelope version is attributable to changed policy content,
    # not to changing the profile identifier itself.
    profile_id = (
        f"scrum-383-{authority_level.value.lower()}-policy"
    )

    envelope = compile_from_operator_policy(
        policy,
        profile_id=profile_id,
        mission_class="first_flight_leo",
        risk_budget_id="scrum-383-demo-risk",
        maneuver_capacity=ManeuverCapacityInput(
            max_dv_per_burn_m_s=10.0,
            v_remaining_m_s=v_remaining_m_s,
            v_reserved_m_s=5.0,
            max_maneuvers_per_week=3,
        ),
        authority_level=authority_level,
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


def run_case(
    pc_action: float,
    authority_level: AuthorityLevel,
) -> Dict[str, Any]:
    """Run one currently-available portion of the SCRUM-383 CA loop."""

    scenario = _load_scenario()

    # Preserve all physical conjunction inputs. Only the operator action line
    # changes between policy-flip cases.
    scenario["policy"]["pc_maneuver_threshold"] = pc_action

    planner_result = evaluate_conjunction(scenario)

    validity_verdict = _build_validity_verdict()

    (
        envelope,
        approved,
        mac_verified,
        effective_authority,
    ) = _compile_and_activate_envelope(
        pc_action=pc_action,
        authority_level=authority_level,
        v_remaining_m_s=float(
            scenario["satellite"]["v_remaining_m_s"]
        ),
    )

    evidence_values: Dict[str, Any] = {
        "conjunction_id": scenario["conjunction_id"],
        "timestamp": DEMO_DECISION_TIME,
        "pc_at_transition": float(scenario["Pc"]),
        "software_version": SERVICE_VERSION,
        "model_version": SERVICE_VERSION,
        "policy_version": (
            f"scrum-383-demo-pc-action-{pc_action:.0e}"
        ),
        "inputs_and_provenance": {
            "scenario_source": str(
                EVENTS_PATH.relative_to(REPO_ROOT)
            ).replace("\\", "/"),
            "scenario_id": SCENARIO_ID,
            "source_scenario_id": SOURCE_SCENARIO_ID,
            "scenario_horizon": (
                "constructed at locked 4-hour minimum from burn to TCA"
            ),
            "pc_action": pc_action,
            "validity_fixture": (
                "constructed SCRUM-378 wiring fixture; "
                "not measured flight data and not claimed "
                "to be physically exact"
            ),
            "integration_scope": (
                "partial SCRUM-383 integration; "
                "SCRUM-379 and SCRUM-382 runtime producers "
                "are not implemented here"
            ),
        },
        "orbit_state": {
            "r_sat_km": scenario["satellite"]["r_sat_km"],
            "v_sat_km_s": scenario["satellite"]["v_sat_km_s"],
            "t_burn_utc": scenario["satellite"]["t_burn_utc"],
            "t_ca_utc": scenario["conjunction"]["t_ca_utc"],
            "r_rel_km": scenario["conjunction"]["r_rel_km"],
            "v_rel_km_s": scenario["conjunction"]["v_rel_km_s"],
        },
        "covariance_state": {
            "p_rel_km2": scenario["conjunction"]["p_rel_km2"],
        },
        "candidate_maneuvers": planner_result["metrics"][
            "all_candidates"
        ],
    }

    # SCRUM-378 owns these exact evidence-catalogue field names.
    evidence_values.update(
        validity_evidence_values(validity_verdict)
    )

    # SCRUM-375 owns authority, envelope version, and approving identity.
    evidence_values.update(
        envelope.evidence_fields(
            approved.approving_identity
        )
    )

    chain_id = (
        f"SCRUM-383-{authority_level.value}-"
        f"{pc_action:.0e}"
    )

    evidence_record = build_decision_record(
        chain_id=chain_id,
        seq=0,
        prev_hash=GENESIS_HASH,
        values=evidence_values,
        recorded_at=DEMO_DECISION_TIME,
    )

    chain_verification = verify_chain(
        [evidence_record]
    )

    evidence_blob = json.loads(
        evidence_record.to_json()
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
            "direction": planner_result[
                "recommendation"
            ]["direction"],
            "dv_magnitude_m_s": planner_result[
                "recommendation"
            ]["dv_magnitude_m_s"],
            "utility": planner_result[
                "recommendation"
            ]["utility"],
        },
        "validity": validity_verdict.to_dict(),
        "authorization_envelope": {
            "authority_level": envelope.authority_level.value,
            "effective_authority": effective_authority.value,
            "envelope_version": envelope.version,
            "alg": approved.alg,
            "key_id": approved.key_id,
            "mac_verified": mac_verified,
        },
        "evidence": {
            "record_id": evidence_record.record_id,
            "content_hash": evidence_record.content_hash,
            "hash_matches": evidence_record.hash_matches(),
            "chain_verified": chain_verification.ok,
            "pending_producers": sorted(
                evidence_record.pending_producers()
            ),
            "fields": evidence_blob["fields"],
        },
    }


def run_demo() -> Dict[str, Any]:
    """Run the available L0-L2, loose/strict policy matrix."""

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
        "scope": "partial-capstone",
        "software_version": SERVICE_VERSION,
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
    print("SCRUM-383 partial MAF policy-flip demo")
    print(f"Planner software version: {result['software_version']}")
    print()

    first = result["cases"][0]

    print(
        f"Scenario: {first['scenario']['conjunction_id']}"
    )
    print(
        f"Fixed scenario Pc: {first['scenario']['pc']:.6g}"
    )
    print()

    for case in result["cases"]:
        policy = case["policy"]
        planner = case["planner"]
        validity = case["validity"]
        envelope = case["authorization_envelope"]
        evidence = case["evidence"]

        print(
            f"{policy['authority_level']} "
            f"Pc_action={policy['pc_action']:.0e} "
            f"decision={planner['direction']} "
            f"dv={planner['dv_magnitude_m_s']:.3f} m/s "
            f"validity={validity['status']} "
            f"effective_authority="
            f"{envelope['effective_authority']} "
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
            f"{authority} policy decision flipped: {flipped}"
        )

    print()
    print(
        "Scope note: this is not yet the complete MAF authority loop. "
        "SCRUM-379 state-machine execution and SCRUM-382 GNC runtime "
        "wiring remain outside this partial integration."
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
