# SCRUM-379 implementation plan, MAF decision state machine and safety monitor

Implements MAF v2.0 section 8. The authoritative guard values and timings live in
docs/scrum-333/state_machine_guards.md; this plan is the build sequence and the
seam map against the current tree. Every dependency this story names already
exists and is merged, so 379 is unblocked. It is the keystone that activates the
Validity Assessor (378), unblocks Tylen's 383, and produces the authorized-to-execute
payload that 382 emits to GNC (the thing Sreerjit flagged nothing in the repo assembles).

## Where it lives

Greenfield. No state-machine scaffold exists today. Build it as a new pure module,
suggested services/planner/common/decision_state_machine.py plus a safety_monitor,
and invoke it from services/planner/server.py in the /v1/evaluate path right after the
scoring result and the ATLAS artifact are built (server.py around line 1046, where
build_atlas_artifact runs and DecisionLog.from_artifact and the evidence record post).
Keep the monitor pure: it takes a fully populated inputs bundle and returns a mode plus
a transition record. That keeps every guard testable without standing up the service.

## States

M0 nominal monitor. M1 elevated watch. M2 armed and evaluating for autonomous action.
M3 executing or committed. M4 escalate to operator and hold. Hard rule: any undefined or
ambiguous transition defaults to ESCALATE (M4). Build the transition table so the default
arm is M4, not a fall-through to a lower mode, and make that the first test.

## Transition guards and where each input comes from

M0 to M1. Pc at or above the derived watch line. Pc is the resolved value from the scorer
(ManeuverScoringResult.pc_pre / pc_source). Read the resolved Pc from the scoring result
directly, not RiskSummary.pc_pre (that field only became reliable after SCRUM-396).

M1 to M2 requires all of these together:
- Pc at or above the action threshold.
- IOD CONFIDENT. services/tracker/iod.py, classify_iod_confidence returns
  IODConfidenceVerdict.CONFIDENT and the gate exposes proceeds_to_validity. Use
  proceeds_to_validity as the single boolean.
- Validity EARNED. The 378 seam. Call services/validity build_validity_verdict_for_arc(
  r_target_tca_km, v_target_tca_km_s, a_km, observation_epochs, r_rel_km_at_tca,
  v_rel_km_s_at_tca, epsilon_threshold, phenomenologies_used) then route_validity_verdict(verdict).
  AUTONOMOUS clears the guard; OPERATOR_REVIEW and BLOCKED (NOT_EARNED) route to M4.
  Critical: 378 does no propagation and imports nothing from tracker. Every state must be at
  the epoch it is needed at, and the target state must genuinely be at TCA, not at the IOD
  solution epoch (conjunction_plane_epsilon is a silent-wrong bug if epochs are mixed). The
  caller (this monitor, or a thin adapter in server.py) owns propagating the arc and target
  to TCA before the call. That propagation seam is the main new code 379 adds around 378.
- Envelope satisfied. services/planner/common/authorization_envelope.py,
  CompiledAuthorizationEnvelope with AuthorityLevel L1 or L2 and dv within the locked cap
  (L1 2.0 m/s, L2 0.5 m/s, L2 raised 1.0 m/s). EnvelopeApprovalError / EnvelopeActivationError
  are the not-satisfied signals.
- Secondary clear. The SCRUM-381 horizon secondary-conflict gate. A conflicting secondary
  routes to M4.
- Slew feasible. t_now at or before t_burn minus (t_slew + t_settle + t_margin). Model
  t_slew + t_settle at 120 s per satellite and t_margin at 30 s. Pure timing check.
- Authority L1 or L2. From the compiled envelope.

## M2 dynamics that are easy to miss

Validity is re-checked on every new CDM while in M2. A drop to NOT_EARNED forces M4
immediately, no grace. covariance_source must be real_cdm for an L2 auto-execute, which is
exactly the field the LeoLabs live path now sets, so a live LeoLabs event is eligible and a
surrogate one is not. The L2 veto window is max(t_review, t_slew + t_settle + t_margin) with
t_review 300 s.

## Recovery

Comms gap beyond 600 s, an L2 continues on the onboard clock. Reboot, restore the persisted
mode, else M4. Any partial subsystem failure goes to M4. Persistence of the current mode is
part of this story.

## Audit logging

Every transition logs in full and feeds the section 10 evidence package. Reuse the existing
evidence_record and DecisionLog machinery that /v1/evaluate already posts. Each transition
record carries mode-from, mode-to, the guard that fired or failed, and the evidence values
including validity_evidence_values(verdict) from 378.

## What 379 hands to 382

On reaching the execute state with authority satisfied, 379 emits the authorized-to-execute
signal plus the command payload (the scorer's dv_eci_km_s / ManeuverSpec plus mode and
authority context). 382 consumes that. Defining this payload shape cleanly here is what
unblocks 382 and closes the gap Sreerjit flagged. Keep the shape small and explicit.

## Dependencies, all present

Envelope compiler (375, authorization_envelope.py), Validity Assessor (378, services/validity),
IOD gate (tracker/iod.py), secondary-clear gate (381). Nothing waits on another ticket. One
governance rule: any guard change that touches a floor escalates to Minh, so build to the
delivered thresholds in docs/scrum-333/state_machine_guards.md and do not retune them here.

## Build order

1. Mode enum, transition table with ESCALATE (M4) as the default arm, guard-inputs dataclass.
   First test: default-to-M4 on an unknown transition.
2. The safety monitor as a pure function over the inputs bundle, one guard at a time, each
   with a section 8 anchored test. Do M1 to M2 last since it is the AND of all guards.
3. The validity invocation seam: propagate arc and target to TCA, call build_validity_verdict_for_arc,
   route the verdict. Test the TCA-epoch requirement explicitly.
4. Integrate into server.py /v1/evaluate after the artifact build, behind the existing additive
   pattern so a monitor failure never breaks the core evaluate.
5. Transition audit logging into the evidence package.
6. Recovery and mode persistence: comms gap, reboot restore, subsystem-failure to M4.

Anchor every guard test to the section 8 example values, not a restatement of the code, the same
discipline as the 409 review. Full suite green is the exit bar. Add at least one test that drives
a live-style event (covariance_source real_cdm, IOD CONFIDENT, validity EARNED, envelope L2) all
the way to an execute emission, and its mirror that forces M4 on a NOT_EARNED re-check.
