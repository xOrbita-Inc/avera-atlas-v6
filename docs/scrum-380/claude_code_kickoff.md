Implement SCRUM-380, MAF safety floors, refusal gates, and ground-commanded abort, in this repo.
Ticket: https://xorbita.atlassian.net/browse/SCRUM-380

READ FIRST: docs/scrum-380/implementation_plan.md. It scopes 380 as the delta on top of 379.
379 already built the guard floors and the locked envelope values — do NOT re-implement them.

Branch off main: feature/scrum-380-safety-floors-abort.

Build, test-first, in this order (detail in the plan):
1. Ground-commanded emergency abort. A ground abort command that PREEMPTS the normal transition
   table and forces M4 immediately from M1, M2 and M3 (interrupts an in-progress sequence during
   a pass), never auto-clears, persists across reboot, and is tamper-evident logged. There is no
   abort command today; this is the headline new piece.
2. Envelope or monitor-logic change drops authority to L0 until ground re-validates. Detect a
   change from the last ground-validated baseline and clamp to L0 (no autonomous execution) until
   an explicit re-validate. This is the floor SCRUM-384's fifth case waits on.
3. Degraded or zero-filled CDM records rejected, not parsed.
4. L3 refused unless it carries a pre-verified safe action (guarded stub; do not build L3 execution).
5. Surface the post-burn feasibility flag on m2_post failure, and add the non-negotiability sweep
   proving no operator or auto value can loosen a locked floor (dv cap, 24 hr freshness, 4.0 hr TCA,
   m2_safe > 25, pc_action <= 1e-4).

Constraints:
- Additive try/except discipline: a monitor or abort failure must never break /v1/evaluate.
- Floors are locked; a threshold change escalates to Minh, not a config knob. Do not add tuning.
- Anchor every test to the guard-doc sections (docs/scrum-333/state_machine_guards.md sec 6 and 7)
  and MAF v2.0 section 6, not to a restatement of your own code.
- Run the planner suite from the repo root: python -m pytest services/planner/tests/ and libs/aps_math.
  Full suite green is the exit bar.
- NO AI attribution in commits or the PR (repo rule).

When done, open a PR against main and report the URL, the new/changed modules, and the abort and
L0-demotion test names, so I can review verification-first.
