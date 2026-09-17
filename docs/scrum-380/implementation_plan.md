# SCRUM-380 Implementation Plan — Safety floors, refusal gates, and ground-commanded abort

## Intent
MAF v2.0 section 6. Enforce the non-negotiable refusal floors as hard, non-operator-tunable limits; add the envelope/monitor-logic-change demotion to L0 until ground re-validates; and build the ground-commanded emergency abort that can interrupt an in-progress sequence during a pass. Depends on SCRUM-379 (decision state machine and safety monitor), which is merged and shipped.

## What 379 already provides — do NOT re-implement
379 already built the guard floors and the locked thresholds. Scope 380 as the delta on top, not a rebuild.

- `safety_monitor.py` guards already enforce: validity EARNED (BLOCKED routes to M4), IOD CONFIDENT, secondary-conflict clear (not-performed is treated as not-clear), TCA above the 4.0 hr floor, CDM freshness, envelope satisfied, covariance provenance, m2_post safe (> 25), subsystem healthy, watchdog. `m1_to_m2_guards` is the AND of the seven; the M1 escalation set routes each failure to M4.
- `authorization_envelope.py` holds the locked values and the compiler already REJECTS operator values that exceed them: LOCKED_MIN_TCA_HOURS 4.0, LOCKED_L1_DV_CAP 2.0, LOCKED_L2_DV_CAP 0.5, LOCKED_L2_RAISED_DV_CAP 1.0, LOCKED_PC_ACTION_CEILING 1e-4, LOCKED_MIN_MISS_DISTANCE_KM 1.0, LOCKED_MAX_TCA_HOURS 72.0, LOCKED_RESERVED_DV 5.0, LOCKED_MAX_MANEUVERS_PER_WEEK 3. `decision_state_machine.py` already states these are locked and a floor change escalates to Minh.
- The transition table already carries ESCALATE to M4 as the default arm, the M4 to M0 ground-clearance row, the M3 to M4 ABORTED/PARTIAL row, and the section 6.3 subsystem-failure to M4 routing. `mode_persistence.py` covers the section 6.2 reboot ladder.

So the refusal floors themselves are in place. 380 adds the genuinely-new pieces below plus a verification pass proving the floors are truly non-negotiable.

## 380 scope — the delta

### 1. Ground-commanded emergency abort (headline, new)
There is no ground abort command today (execution_status ABORTED is a GNC report outcome, not a ground command). Build a ground-commanded abort that:
- Introduces a ground abort command input (an abort command on the state machine's command channel / an abort field on GuardInputs), carrying abort_reason and operator identity.
- Forces an immediate transition to M4 from any active mode — M1, M2, and critically M3 (in-progress). It PREEMPTS all other guards: evaluate the abort at the very top of transition resolution, before the normal table, so it interrupts a staged or executing sequence during a pass.
- Is the highest-priority transition; nothing overrides it, and it never auto-clears (M4 holds until the existing M4 to M0 ground clearance).
- Persists via mode_persistence so an abort survives a reboot mid-sequence.
- Emits a tamper-evident abort log entry (section 7 shape) with from_mode, abort_reason, operator.

### 2. Envelope or monitor-logic change drops to L0 until ground re-validates (new; unblocks 384)
379 checks the envelope is cryptographically valid and unexpired, but does not demote on CHANGE. Add: when the authorization envelope OR the monitor logic changes from the last ground-validated baseline (envelope_version change, or monitor software_version / logic hash change), force authority to L0 (advisory only, no autonomous execution) and HOLD there until an explicit ground re-validation clears the new baseline. This is the tamper/unsigned-envelope floor SCRUM-384's fifth case waits on. Include baseline tracking, change detection, the L0 demotion (authority clamped to L0 regardless of the envelope's stated level), the ground re-validate command that re-baselines, and audit entries.

### 3. Degraded / zero-filled data rejection (data-quality floor)
The freshness guard checks age. Add the reject-not-parse rule: degraded or zero-filled CDM records are rejected outright rather than parsed and evaluated. If not already enforced upstream, add a floor that refuses a degraded/zero-filled record (hold M1, or route to M4 if staged) rather than letting a hollow record through the guards.

### 4. L3 pre-verified-safe-action gate (light)
L3 is gated to a later phase. Add the floor only: an L3 authorization is refused unless it carries a pre-verified safe action. Keep it a guarded stub; do not build the L3 execution path.

### 5. Post-burn feasibility flag (small)
`guard_m2_post_safe` already enforces m2_post > 25 on M3 to M0, and the M3 to M4 elevated-residual row exists. Surface the feasibility flag on failure if it is not already emitted, so a post-burn residual failure shows in the evidence package, not only as a mode change.

## Non-negotiability verification
For each floor already in 379, add or confirm a test that an operator-supplied or auto-derived value cannot loosen it: dv above the level cap, freshness beyond 24 hr, TCA floor below 4.0 hr, m2_safe below 25, pc_action above 1e-4 must all be rejected or clamped, never honored. These are locked; a change escalates to Minh. Do not add config to tune them.

## Test discipline
- Anchor tests to guard-doc sections 6 and 7 (docs/scrum-333/state_machine_guards.md) and MAF v2.0 section 6, not to a restatement of the code.
- Abort: a ground abort from M2 and from M3 both land in M4 immediately, preempting the normal table; abort persists across a simulated reboot; abort is audit-logged.
- L0 demotion: an envelope-version change and a monitor-logic-hash change each force L0 and block L2 auto-execute until a ground re-validate; re-validate restores authority.
- Degraded record: a zero-filled/degraded CDM is rejected, not scored.
- Floors: each non-negotiability test above.
- Keep the additive try/except discipline so a monitor or abort failure never breaks /v1/evaluate.
- Run the planner suite from the repo root: `python -m pytest services/planner/tests/` and `libs/aps_math`. Full suite green is the exit bar.

## Boundaries
- Do not re-implement 379's guards or the locked envelope values; extend them.
- Do not build the L3 execution path or GNC transport; SCRUM-382 owns emission.
- Every floor stays locked; a threshold change is a Minh escalation, not a config knob.

## Build order
1. Ground-commanded abort: command input, top-of-resolution preempt to M4 from M1/M2/M3, persistence, audit. Test first.
2. Envelope/monitor-logic-change to L0 demotion with ground re-validate. Test.
3. Degraded/zero-filled record rejection. Test.
4. L3 pre-verified-safe-action refusal stub. Test.
5. Feasibility-flag surfacing plus the non-negotiability test sweep. Full suite green.
