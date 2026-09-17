Implement SCRUM-382, MAF v2.0 section 9, APS to GNC emission and post-burn report wiring, in this repo.
Ticket: https://xorbita.atlassian.net/browse/SCRUM-382

READ FIRST: docs/scrum-382/implementation_plan.md, docs/scrum-379/authorized_execution_payload.md (the input), and openapi/gnc_interface.yaml (the contract, final, do not edit it).

379 already emits the authorized-execution payload and 380 guards the floors. Do NOT re-implement the decision or floor logic, and do NOT compute the ExecutionError physics or the post-maneuver Pc. Those live in maneuver_scorer (compute_pc_post and the SCRUM-365 machinery); consume them.

Branch off main: feature/scrum-382-gnc-emission.

Build, test-first, in this order (detail in the plan):
1. Assemble a schema-valid GNCCommand from AuthorizedExecution plus command_id, approving_identity, validity, risk, maneuver (RTN), safe_action, timing. Rotate the dv from ECI to RTN using the existing frame convention, never a new one. Respect approval_basis (L2 emits directly, L1 only after approval).
2. A feature-flagged GNC emission adapter that POSTs to /v1/gnc/command and handles the ack, inert with a record-or-log fallback when the GNC endpoint is unconfigured, same pattern as LeoLabs and UDL. Additive try/except so a GNC failure never breaks /v1/evaluate.
3. The /v1/gnc/approve (L1) and /v1/gnc/veto (L2) endpoints, translating into the state machine's approval and veto inputs and returning the acks.
4. The /v1/gnc/report consumer: parse the report, feed compute_pc_post for P_post = P_pre + P_burn, assemble the section-10 actual_vs_predicted / post_maneuver_od / residual_risk producers, record on the SCRUM-377 evidence trail, return the ack.
5. Wire into /v1/evaluate additively so the response is unchanged when GNC is off.

Constraints:
- Conform to openapi/gnc_interface.yaml exactly; schema-validate every command, ack and report round-trip in tests. Do not edit the contract.
- Units and frames per section 9: ECI km and km/s, dv m/s, RTN burn vectors, covariance km2 row-major, UTC ISO-8601.
- Additive try/except: a GNC emission or report failure must never break /v1/evaluate.
- Anchor tests to the contract and MAF v2.0 section 9/10, not to a restatement of your own code.
- Run the planner suite from the repo root: python -m pytest services/planner/tests/ and libs/aps_math. Full suite green is the exit bar.
- NO AI attribution in commits or the PR (repo rule).

One open decision to surface, do not guess: whether there is a real GNC service endpoint to POST to for the demo, or emission should be record-only (contract-validated, logged or stored) until a GNC endpoint exists. Default to the record-only fallback behind the flag and call it out in the PR so John can point it at a real endpoint when there is one.

When done, open a PR against main and report the URL, the new and changed modules, the new endpoints, and the command-builder and report-consumer test names, so I can review verification-first.
