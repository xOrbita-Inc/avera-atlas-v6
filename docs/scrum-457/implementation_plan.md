# SCRUM-457 implementation plan: dashboard polls and resolves the secondary conflict row

Ticket: https://xorbita.atlassian.net/browse/SCRUM-457
Branch off main. UI only. Part of SCRUM-433. Depends on SCRUM-456 (merged).
Relates to SCRUM-442.

## What this is

SCRUM-456 made the screen run in the background: evaluate returns immediately with
the secondary check pending and a job id, and the planner exposes
GET /v1/secondary-screen/{id} that resolves to clear, not_clear or error. This
ticket makes the dashboard show that: the Secondary conflict row goes from
screening in progress to CLEAR or NOT CLEAR when the screen resolves, without the
evaluate call ever waiting on it.

## What SCRUM-456 gives us to build on

- The evaluate response carries the pending state at
  atlas_artifact.post_maneuver.secondary_conflict.screen_pending (the flag) and
  .screen_job_id (the poll key). A test pins that both survive serialisation.
- The poll endpoint GET /v1/secondary-screen/{id} returns
  {status: pending|clear|not_clear|error, clear (an allow-list bool, only true on
  a real clear), conjunctions (capped at 50, closest first), error, ...}. It is a
  fast read, a few milliseconds.

## The change, all in the ui service

1. A proxy route in services/ui/app/main.py, matching the existing planner proxies:
   GET /api/planner/secondary-screen/{id} to PLANNER_SERVICE_URL/v1/secondary-screen/{id},
   with a short timeout since the endpoint is a fast read. On a connection error or
   an upstream failure, return a body the front end reads as NOT CLEAR, never a
   shape that could look clear, so a planner hiccup fails closed in the ui the same
   way the guard does.

2. In services/ui/app/templates/index.html, after an evaluate response, read
   secondary_conflict.screen_pending and screen_job_id. When pending, render the
   Secondary conflict row as screening in progress, and poll
   /api/planner/secondary-screen/{id} on an interval (a few seconds) until the
   status is clear, not_clear or error. On resolution, fill the row: clear shows
   CLEAR, not_clear shows NOT CLEAR with the breaching events, error and timeout
   show NOT CLEAR with a note that the screen could not complete. Stop polling on
   any resolution.

3. Bound the polling. The screen's own budget is 300 s; cap the ui polling at a
   little beyond that and, if it is still pending when the cap is hit, show NOT
   CLEAR with a could-not-complete note rather than polling forever. Never leave a
   spinner that cannot end.

4. Latest selection wins. Reuse the SCRUM-449 pattern, the evalSeq token and
   AbortController that already guards the encounter panel, so that starting a new
   evaluate or selecting a different conjunction cancels any in-flight secondary
   poll. A stale screen result must never overwrite the row for the decision now
   on screen. This is the same race SCRUM-449 fixed for the geometry pane; apply
   it to the secondary poll.

5. Honesty in the states, never a false clear in the ui: pending is not clear and
   is shown as screening, not as a provisional pass; only a status of clear with
   the allow-list clear flag true renders CLEAR. Anything else renders NOT CLEAR or
   screening.

## Do not

- Do not change the planner, the poll endpoint, the screen, or the evaluate
  contract. This is the dashboard consuming what SCRUM-456 already returns.
- Do not raise the evaluate proxy timeout or put the screen back on the evaluate
  path. The whole point is that evaluate no longer waits.
- Do not let pending or error render as CLEAR.

## Tests

- A proxy-route test in the ui, like the existing planner-proxy tests: the poll
  route forwards to the planner and, on a planner connection error, returns the
  NOT CLEAR shape rather than an error that the front end could misread.
- The JS row-resolution and the race cancel are verified live (the dashboard has
  no JS unit harness); cover them in the live verification rather than asserting
  them offline.

Run the existing ui tests and confirm green.

## Live acceptance

Rebuild the ui container. On the stack, run a maneuver-recommended decision on the
demo asset: the burn shows immediately, the Secondary conflict row shows screening
in progress, and it resolves to CLEAR or NOT CLEAR when LeoLabs completes (about a
minute and a half after the response, per SCRUM-456). Confirm a forced screen
failure resolves the row to NOT CLEAR, and that selecting a different conjunction
mid-screen cancels the old poll so the row tracks the current decision. Record in
docs/scrum-457/live_verification.md, ideally with a short GIF of the row resolving.

## Commit

Branch off main. Include docs/scrum-457/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it checks that
the proxy fails closed, that pending and error never render as CLEAR, that the poll
is bounded and the SCRUM-449 race cancel is applied, and reviews the live row
transitions.
