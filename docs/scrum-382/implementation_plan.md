# SCRUM-382 Implementation Plan — APS to GNC emission and post-burn report wiring

## Intent
MAF v2.0 section 9. Wire APS to emit the GNCCommand when the decision state machine authorises a burn, and to consume the GNCReport that comes back, against the delivered contract openapi/gnc_interface.yaml (final on main). This is the output side of the loop: 379 decides and authorises, 380 guards the floors, 382 carries an authorised decision out to GNC as a command and folds the post-burn report back in. Depends on 379 (the authorised-execution payload, merged and shipped) and consumes the SCRUM-365 execution-error and post-maneuver-Pc physics that already live in maneuver_scorer.

## What already exists — do NOT re-implement
- The authorised-to-execute payload. `AuthorizedExecution` in decision_state_machine.py, emitted on `MonitorDecision.authorized_execution`, surfaced at /v1/evaluate. server.py at the evaluate hook already marks it: "The signal SCRUM-382 consumes to assemble the GNC command." Twelve flat fields, documented in docs/scrum-379/authorized_execution_payload.md. The dv travels as the scorer produced it; 382 never re-plans a maneuver.
- The contract. openapi/gnc_interface.yaml, final, with four endpoints (/v1/gnc/command, /v1/gnc/approve, /v1/gnc/veto, /v1/gnc/report) and every schema 382 needs (GNCCommand, GNCCommandAck, ValidityVerdict, RiskInputs, ManeuverSpec, SafeActionRef, TimingBudget, GNCApprovalCommand/Ack, GNCVetoCommand/Ack, GNCReport/Ack, ExecutionError, PostBurnState, AbortDetail). 382 conforms to it exactly; it does not redesign the schemas or edit the file.
- The execution-error and post-maneuver covariance physics. maneuver_scorer.py already has compute_pc_post(...) (P_post = P_pre + P_burn) and the SCRUM-365 execution-error machinery. 382 feeds these from the report; it does not compute the physics. The ExecutionError block is explicitly SCRUM-365's, per the ticket.
- The operator command model. The state machine already takes approval_command_received / approval_accepted (L1) and veto_command_received (L2) as GuardInputs. 382's /v1/gnc/approve and /v1/gnc/veto endpoints translate GNC operator commands into those inputs; it does not re-model approval or veto.

## 382 scope — the delta

### 1. Assemble the GNCCommand (headline)
Build a GNCCommand from the AuthorizedExecution payload plus the blocks the contract adds: command_id, approving_identity, validity (ValidityVerdict), risk (RiskInputs), maneuver (ManeuverSpec, an RTN burn), safe_action (SafeActionRef), timing (TimingBudget). Conform field-for-field to gnc_interface.yaml. Note the frame conversion: AuthorizedExecution carries dv_eci_km_s (ECI), the ManeuverSpec is an RTN burn, so the builder rotates ECI to RTN at the burn state using the existing frame convention (libs/aps_math / the test_frames convention), never a new one. Respect approval_basis: an L2 authorised_execution (l2_veto_window_expired) is emitted directly; an L1 one (l1_operator_approval) only exists after approval.

### 2. The GNC emission adapter, feature-flagged
A client that POSTs the command to the configured GNC endpoint (/v1/gnc/command) and handles the GNCCommandAck. Behind a feature flag (e.g. GNC_ENABLED plus a GNC base URL) with a graceful record-or-log fallback when the endpoint is unconfigured, mirroring the LeoLabs and UDL pattern: a missing GNC config interpolates to inert, not an error. Additive try/except so an emission failure never breaks /v1/evaluate.

### 3. Operator approve / veto endpoints
POST /v1/gnc/approve (GNCApprovalCommand, L1) and POST /v1/gnc/veto (GNCVetoCommand, L2), each returning its ack and translating into the state machine's approval / veto inputs. The veto path covers a committed L2 burn per the contract.

### 4. Consume the GNCReport
POST /v1/gnc/report (receiveGNCReport). Parse execution_status, actual delta-v, attitude error, the ExecutionError block (365), and PostBurnState. Feed P_post = P_pre + P_burn via compute_pc_post for the post-maneuver covariance, and assemble the section-10 producers actual_vs_predicted, post_maneuver_od and residual_risk. Return the GNCReportAck. Record the report on the SCRUM-377 evidence trail so the post-burn outcome is auditable next to the authorising decision.

### 5. Wire into /v1/evaluate
At the server.py hook, when authorized_execution is present, assemble and emit through the adapter (flag permitting), and attach the command and ack to the response additively, so the existing evaluate contract is unchanged when GNC is off.

## Boundaries
- Do not compute the ExecutionError physics or the post-maneuver Pc; consume compute_pc_post and the 365 machinery. ExecutionError is SCRUM-365's.
- Do not re-plan the maneuver; the dv is the scorer's, unchanged.
- Do not touch 379's decision logic or 380's floors.
- Conform to openapi/gnc_interface.yaml exactly; do not edit the contract.
- Units and frames per section 9: ECI km and km/s, delta-v m/s, RTN burn vectors, covariance km2 row-major, UTC ISO-8601.
- Emission is inert until configured; a missing GNC endpoint is a fallback, not a failure.

## Test discipline
- Anchor tests to openapi/gnc_interface.yaml (schema-validate every command, ack and report round-trip against it) and to MAF v2.0 section 9/10, not to a restatement of the code.
- The command builder produces a schema-valid GNCCommand from a known AuthorizedExecution, with the ECI-to-RTN dv rotation checked against the frame convention.
- L2 emits directly; L1 emits only after approval; a veto on a committed L2 is accepted and acknowledged.
- A GNCReport round-trips, drives compute_pc_post for P_post, and the residual-risk and post-maneuver producers match section 10.
- Emission-off and emission-misconfigured leave /v1/evaluate unchanged (additive try/except).
- Run the planner suite from the repo root: python -m pytest services/planner/tests/ and libs/aps_math. Full suite green is the exit bar.

## Build order
1. GNCCommand builder from AuthorizedExecution, with the ECI-to-RTN rotation and full schema conformance. Test first against the contract.
2. The feature-flagged emission adapter and the GNCCommandAck path. Test the inert-when-unconfigured fallback.
3. The /v1/gnc/approve and /v1/gnc/veto endpoints into the state machine inputs, with acks.
4. The /v1/gnc/report consumer, the compute_pc_post feed, the section-10 producers, the evidence-trail record.
5. Wire into /v1/evaluate additively; full suite green.
