# SCRUM-456 implementation plan: run the secondary screen asynchronously, off the evaluate path

Ticket: https://xorbita.atlassian.net/browse/SCRUM-456
Branch off main. Backend only, planner. Part of SCRUM-433. Relates to SCRUM-442.
Blocks SCRUM-457.

## What this is

Today evaluate runs the on-demand screen inline, and the screen takes 30 s to
2 min against LeoLabs (300 s poll budget). The dashboard's evaluate call times out
at 60 s, so a maneuver-recommended decision fails. Decouple the screen from the
evaluate response: evaluate returns the burn decision immediately with the
secondary check PENDING and a screening id, the screen runs in the background, and
a poll endpoint returns the result when it completes. SCRUM-457 makes the dashboard
poll it.

## The safety invariants, first, because they constrain everything

- PENDING must never read as CLEAR. A provisional screen is not a passed screen.
- A maneuver is not authorized as secondary-clear while the screen is PENDING. The
  decision is provisional until the screen resolves.
- A screen that errors, times out, is rate-limited, or has no access fails closed
  exactly as SCRUM-442 built it: NOT CLEAR, escalate. Async does not soften this.
- No decision is ever finalized secondary-clear on a screen that did not return
  CLEAR.

These are the same properties SCRUM-442 gave the inline screen. Moving the screen
off the evaluate path must not weaken any of them.

## The seam as it is

build_atlas_artifact drives the secondary-conflict path, which today calls
run_screening inline and sets secondary_check_performed and
secondary_conjunction_clear on the artifact, gated on secondary_screen_enabled and
scoring.is_maneuver_recommended(). guard_secondary_clear reads those two booleans
and fails closed when the check was not performed. Trace the exact functions rather
than assuming their shape; they are the ones SCRUM-442 and SCRUM-458 touched.

## The change

1. A third secondary state, PENDING. Add it alongside not-performed and clear.
   PENDING means the screen was started and has not resolved. It is distinct from
   not-performed (which fails closed) and from clear.

2. Evaluate starts the screen in the background. When secondary_screen_enabled and
   a maneuver is recommended, evaluate no longer calls run_screening inline.
   Instead it generates a screening id, submits the screen to a background worker,
   and returns the artifact with the secondary check PENDING and the id. Evaluate
   does no LeoLabs polling on the request path. Whether the LeoLabs create happens
   synchronously (fast, ~1 s, gives a real screening id) or in the worker (evaluate
   does zero network) is your call; if the create runs in the worker, a create
   failure surfaces as the store's error status, which the guard fails closed on.

3. An in-process result store, keyed by the screening id, holds the status
   (pending, clear, not_clear, error), the parsed conjunctions, the verdict, the
   seed source, the LeoLabs screening id, and a created-at for eviction. Guard it
   with a lock and evict old entries, following the leolabs_runtime cache pattern.
   In-process is fine for the single planner process; durable persistence is
   SCRUM-453.

4. The background worker runs the existing path unchanged: build the SCRUM-440
   ephemeris, run_screening (create, wait_for_screening, retrieve, parse with the
   SCRUM-458 repair, dedupe with the screening key, evaluate the clear contract),
   and write the result and verdict into the store. Bound concurrency with a small
   pool so many evaluates cannot spawn unbounded threads. The screen's covariance,
   seed and parse (SCRUM-452, 454, 458) are unchanged; only where and when it runs
   moves.

5. A poll endpoint, GET /v1/secondary-screen/{id} (confirm the name), returns the
   store entry: {status, conjunctions, verdict, seed_source, screening_id}. It is
   read-only and fast. This is what SCRUM-457 polls.

6. The decision treats PENDING as provisional. In the decision state machine,
   PENDING is not-yet-authorized, not not-performed: the maneuver is held provisional
   rather than escalated to M4 on pending alone, and it resolves to secondary-clear
   or escalates when the screen returns. Keep the SCRUM-442 fail-closed for error
   and timeout. Trace the actual state machine and implement PENDING consistently
   with how M0 to M4 already work; do not invent mode transitions.

## Do not

- Do not change the screen's covariance, seed priority or parse (SCRUM-452, 454,
  458). This ticket moves execution, not the physics or the parsing.
- Do not touch the dashboard; that is SCRUM-457.
- Do not weaken any fail-closed path. Error, timeout, rate-limit and no-access
  remain NOT CLEAR.
- Do not let PENDING be read as CLEAR anywhere.

## Tests

Offline, mocked client and a stubbed background worker (run it synchronously in
tests, or inject the worker, so the async is deterministic):
- A maneuver-recommended evaluate returns fast with the secondary PENDING and a
  screening id, and does not call the LeoLabs poll on the request path.
- The poll endpoint returns PENDING before the worker finishes, then CLEAR or
  NOT CLEAR after, from the store.
- A worker that raises the typed screening failure writes error, and the guard and
  the poll endpoint both treat it as NOT CLEAR, never clear.
- PENDING is never clear: a decision with a pending screen is provisional, not
  authorized secondary-clear, and not escalated to M4 on pending alone.
- The store evicts old entries and reset_caches clears it.
- A no-maneuver evaluate starts no screen (the SCRUM-442 gate stays).

Run python3 -m pytest services/planner and confirm green.

## Live acceptance

Rebuild the planner. On the stack, a maneuver-recommended evaluate returns in the
normal fast time with the secondary PENDING and an id; polling the endpoint shows
PENDING then resolving to CLEAR or NOT CLEAR once LeoLabs completes; a forced
failure fails closed. Respect 3 creates per 2 minutes. Record in
docs/scrum-456/live_verification.md the evaluate latency, the poll transitions, and
the fail-closed check.

## Commit

Branch off main. Include docs/scrum-456/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it confirms
evaluate no longer blocks on the screen, that PENDING is never read as CLEAR and a
pending screen never authorizes a maneuver, that error and timeout fail closed, and
that the screen's physics and parse are unchanged, then reviews the live poll
transitions.
