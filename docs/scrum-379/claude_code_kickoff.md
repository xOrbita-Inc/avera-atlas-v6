Implement SCRUM-379, the MAF decision state machine and safety monitor (M0 to M4), in this repo.

Read these two files first, in order:
- docs/scrum-379/implementation_plan.md  (build sequence and seam map)
- docs/scrum-333/state_machine_guards.md  (authoritative MAF v2.0 section 8 guard values and timings; locked, do not retune, a floor change escalates to Minh)

Scope. Build a new pure module (services/planner/common/decision_state_machine.py plus a safety monitor) and invoke it from services/planner/server.py in the /v1/evaluate path, right after the scoring result and ATLAS artifact are built (around the build_atlas_artifact call). The monitor must be a pure function over a fully populated inputs bundle that returns a mode plus a transition record, so every guard is testable without the service running.

The guard inputs already have homes in the tree. Do not reinvent them:
- Resolved Pc and provenance: ManeuverScoringResult.pc_pre / pc_source (services/planner/common/maneuver_scorer.py). Read the resolved Pc from the scoring result, not RiskSummary.pc_pre.
- IOD CONFIDENT: services/tracker/iod.py, proceeds_to_validity.
- Validity EARNED: services/validity, build_validity_verdict_for_arc then route_validity_verdict. This is the one genuinely new piece of engineering. 378 does no propagation and imports nothing from tracker, so you must propagate the arc and the target to TCA before calling it, and the target state must be at TCA, not the IOD epoch. Test that epoch requirement explicitly, mixing epochs is a silent-wrong bug.
- Envelope and authority L1/L2 with dv caps: services/planner/common/authorization_envelope.py.
- Secondary clear: the SCRUM-381 horizon secondary-conflict gate.
- Slew feasibility: the timing inequality in the plan (120 s slew+settle per sat, 30 s margin).

Build order (from the plan):
1. Mode enum, transition table with ESCALATE (M4) as the default arm, guard-inputs dataclass. First test: an unknown transition falls to M4.
2. The monitor, one guard at a time, each with a section 8 anchored test. Do M1 to M2 last, it is the AND of all guards.
3. The validity seam: propagate arc and target to TCA, call build_validity_verdict_for_arc, route the verdict.
4. Integrate into server.py behind the existing additive pattern so a monitor failure never breaks the core evaluate.
5. Transition audit logging into the evidence package (reuse evidence_record and DecisionLog).
6. Recovery and mode persistence: comms gap beyond 600 s (L2 continues on onboard clock), reboot restore else M4, any partial subsystem failure to M4.

Also define the authorized-to-execute payload that 379 hands to SCRUM-382 (GNC emission): the scorer's dv_eci_km_s / ManeuverSpec plus mode and authority context. Keep it small and explicit. This is the payload nothing in the repo currently assembles.

Testing discipline. Anchor every guard test to the section 8 example values, not to a restatement of your own code. Run the planner suite from the repo root: python -m pytest services/planner/tests/ and libs/aps_math. Full suite green is the exit bar. Include one test that drives a live-style event (covariance_source real_cdm, IOD CONFIDENT, validity EARNED, envelope L2) all the way to an execute emission, and its mirror that forces M4 on a NOT_EARNED re-check in M2.

Workflow. Work on a branch feature/scrum-379-decision-state-machine, commit incrementally with clear messages, keep the suite green as you go. When done, push the branch and open a PR against main for review. Do not merge or close anything. John is the only one who merges and closes.
