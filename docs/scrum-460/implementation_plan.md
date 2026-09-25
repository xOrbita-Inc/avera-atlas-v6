# SCRUM-460 implementation plan: evaluate a clicked conjunction without a whole-window fetch on the event loop

Ticket: https://xorbita.atlassian.net/browse/SCRUM-460
Branch off main. Backend only, planner. Reliability, safety-relevant. Part of
SCRUM-433. Relates to SCRUM-459 and SCRUM-412.

## The bug

SCRUM-459 bounded and offloaded the conjunction LIST fetch, so a dense asset no
longer takes the planner down from the list. Evaluate was out of scope there and is
still exposed. /v1/evaluate resolves the clicked conjunction by pulling the whole
in-volume window from LeoLabs and filtering it to the selector, synchronously on the
event loop. For SWARM B that is the same 773 second, 75,000 CDM pull SCRUM-459
measured, so clicking a SWARM B row to evaluate it wedges the single planner worker
exactly as the list did before the fix. The list is safe now; evaluate is not. This
is the one that still takes the planner red end to end, so it is the priority.

## The seam as it is

In server.py the LeoLabs evaluate path resolves the selector with
select_leolabs_conjunction(fetch_leolabs_conjunctions(primary), selector), and the
no-selector path calls fetch_leolabs_conjunction(primary). Both go through the
unbounded _scorable_in_risk_order whole-window pull (leolabs_runtime), on the event
loop. SCRUM-459 deliberately left that pull unbounded, because it searches the whole
window to find the operator's selection, and a truncated window would 404 that
selection. Trace the exact functions; do not assume their shape.

## The fix

Two parts, the first the real one.

1. Resolve a clicked row by its cdm_id, with no whole-window pull. The list rows the
   operator clicks carry a cdm_id (and event_id). When the selector is a cdm_id,
   fetch that one CDM directly (client.get_cdms takes a cdm id), parse it with
   parse_leolabs_cdm, and score it. No window pull, no pagination, one request. This
   is fast and exact, and it is the path the dashboard actually takes, because the
   operator clicks a row. Confirm get_cdms returns the same CDM shape parse_leolabs_cdm
   expects, and that a row's cdm_id round-trips into the same parsed conjunction the
   window search would have produced for that row, so the score is identical to
   today's for the same click.

2. Keep the window search as a fallback, off the event loop. A selector that is not a
   cdm_id, and the no-selector "evaluate the worst" path, still need the window. Keep
   them, but run the blocking LeoLabs work off the event loop the same way SCRUM-459
   offloaded the list: reuse its dedicated executor and in-flight cap
   (_run_list_fetch / _LIST_EXECUTOR / ListFetchBusy in server.py) rather than adding
   a second mechanism, and shed the same typed 503 when the cap is taken. The point is
   that even the slow fallback can never stop /health or the other assets. The
   no-selector dense case is still slow because it needs the whole window, but off the
   loop it is survivable rather than fatal; note it, do not bound it here (bounding the
   worst-selection would change which conjunction gets evaluated, which is a different
   decision).

## What the fix must guarantee

- Evaluating a clicked SWARM B conjunction returns within the ui timeout and never
  takes the planner red.
- No evaluate, on any asset, blocks /health or the other routes while its LeoLabs
  fetch runs.
- The score for a given clicked row is unchanged: the cdm_id path must produce the
  same parsed conjunction and the same decision as the window-search path did for the
  same row.

## Do not

- Do not change scoring, the covariance seed, the secondary screen (SCRUM-456/457),
  or the clear contract. This moves how the conjunction is fetched, not how it is
  judged.
- Do not bound the worst-selection fetch into truncation. Evaluate needs the real
  conjunction the operator chose; a truncated pull could evaluate the wrong one. The
  list is what is allowed to be partial, not the decision.
- Do not add a second offload mechanism. Reuse the SCRUM-459 executor and cap.

## Tests

Offline, with a fake client:
- A cdm_id selector fetches exactly that CDM (assert get_cdms called with the id) and
  makes no window search (assert the search endpoint saw zero calls), and the parsed
  conjunction and the decision match what the window-search path produced for the same
  row fixture.
- A non-cdm_id selector and the no-selector path still resolve via the window search,
  now off the loop.
- Concurrency, the SCRUM-459 style: a slow evaluate fetch does not block a concurrent
  /health-style request, and past the in-flight cap evaluate sheds the typed 503
  rather than queueing. Drive the ASGI app through the async transport so a blocking
  call would really block, and time the quick request from before the evaluate starts.
- The existing evaluate tests still pass unchanged.

Run python3 -m pytest services/planner and confirm green.

## Live acceptance

On the local stack, click a SWARM B conjunction and evaluate it: it returns within
the budget, the decision is for the exact conjunction clicked, and the planner health
stays green with other assets still listing throughout. Confirm from the logs that the
cdm_id path made no window search. Record in docs/scrum-460/live_verification.md,
including the evaluate latency for a SWARM B click and a /health sample taken during
it. Respect the LeoLabs rate limit. Do not deploy to the KVM; that is John's manual
step.

## Commit

Branch off main. Include docs/scrum-460/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it confirms a
clicked cdm_id evaluates with no whole-window pull and an unchanged decision, that the
fallback and no-selector paths run off the event loop, that a slow evaluate cannot
starve /health or the other assets, and that scoring is untouched, then reviews the
live SWARM B evaluate.
