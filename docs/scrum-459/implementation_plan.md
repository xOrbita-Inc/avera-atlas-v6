# SCRUM-459 implementation plan: list live conjunctions without a fetch that can take the planner down

Ticket: https://xorbita.atlassian.net/browse/SCRUM-459
Branch off main. Backend only, planner. Reliability, safety-relevant. Part of
SCRUM-433. Relates to SCRUM-445, SCRUM-450, SCRUM-422.

## The bug

Selecting SWARM B (39451) on the live-asset view runs the conjunction-list fetch
past the 60 second ui-to-planner read timeout and fails (the ui shows
`HTTPConnectionPool(host='planner', port=8060): Read timed out (read timeout=60)`).
Worse, when it is in that state the planner stops serving the other assets too, so
one dense asset takes the whole live-asset view down. The planner keeps answering
`/health` in 0.2 ms while a single SWARM B fetch is pending, so the fetch is slow,
not the event loop cleanly wedged; but under more than one heavy fetch the planner
goes red and everything fails. SWARM B flies in formation with SWARM A and C, so it
is the densest subscribed asset and the first to cross the line.

## What already exists, so we do not rebuild it

- A short-TTL window cache already exists: `_CDM_CACHE_TTL_SECONDS = 45.0`,
  `_cdm_cache`, `_search_cdms_cached` (SCRUM-450), cleared by `reset_caches`. The
  list path uses it: `_raw_cdms_in_risk_order` calls `_search_cdms_cached`, so a
  repeat select of the same asset within 45 s is already a cache hit. Do NOT add a
  second cache. The cache is not the gap.
- The response is already paged (SCRUM-445): `fetch_leolabs_conjunction_page`
  fetches the whole in-volume window, orders and dedupes on raw dicts, parses only
  the returned page. The paging fixed the response size, not the upstream fetch
  cost.

The gap is the COLD fetch (a cache miss) for a dense asset, which pulls the whole
window from LeoLabs and for SWARM B now takes longer than 60 seconds, plus the fact
that these heavy fetches can starve the planner for everything else.

## Confirm the concurrency model first

The route is `async def leolabs_conjunctions` and it calls the blocking
`fetch_leolabs_conjunction_page` (which does synchronous `requests` calls) on the
request path. The planner runs one uvicorn worker (no `--workers`). Before changing
anything, establish how a blocking call in this route actually behaves here: whether
Starlette is offloading it, why `/health` still answered during a single SWARM B
fetch, and what happens to `/health` and other assets under two or three concurrent
SWARM B fetches. Fix against what is really there, not against an assumption.

## What the fix must guarantee

1. No single list fetch may exceed a hard budget well under 60 seconds. The endpoint
   always returns within that budget.
2. No number of concurrent heavy fetches may keep the planner from serving `/health`
   or from listing the other assets. One dense asset can never take the view down.
3. A fetch that could not pull the whole window must return a result explicitly
   marked incomplete, and the ui must never read an incomplete or truncated list as
   the full, clear picture. A short list that hides the worst conjunction is the
   dangerous direction, exactly as a dropped conjunction was in SCRUM-458.

## The change

### A. Bound the cold fetch with a hard deadline and a cap

The cost is in the whole-window pull inside `_search_cdms_cached` ->
`search_conjunction_cdms` (the paginated LeoLabs `while True` in the client). Give
that pull a hard wall-clock deadline (a few seconds of headroom under 60, propose
about 25 to 30 s, confirm at review) and a cap on how many CDMs / pages it will
draw. When the deadline or cap is hit, stop and return what was pulled so far,
tagged incomplete, rather than running on toward the ui timeout.

Because CDMs come back in LeoLabs order rather than Pc order, a truncated pull can
miss the worst conjunction. So the incomplete result is not "the top of the list",
it is "a partial view, and we are telling you it is partial." Carry an explicit
`complete: false` (and how many were pulled versus the reporting-volume total when
known) on `ConjunctionPage` so the endpoint and the dashboard can say so. Never let
incomplete render as a normal, full listing.

Cache the bounded result the same way the full result is cached (SCRUM-450), keyed
so a completed result supersedes an earlier incomplete one for the same query, and
so an incomplete result does not pin a wrong "total".

### B. Keep heavy fetches from starving the planner

Take the blocking window fetch off the path that serves `/health` and the other
assets, and bound how many run at once, so a dense asset cannot consume the worker.
Run the fetch in a small, dedicated bounded executor (not the shared default
threadpool that other endpoints depend on), and cap concurrent LeoLabs list fetches
with a semaphore. When the cap is already taken, shed load with a typed 503 that the
dashboard shows as "busy, try again", rather than queueing without limit until
everything times out. `/health` must stay green no matter how many list fetches are
in flight.

### C. Do not weaken correctness for the completed case

When the fetch does complete within budget, behaviour is unchanged: worst Pc first,
one event per reissue, exact total, the SCRUM-450 cache serving repeats. This ticket
only bounds the pathological case and isolates it; it does not change what a
complete listing contains or its ordering.

## Out of scope

- The evaluate path and the on-demand secondary screen (SCRUM-456/457) are not
  touched. The unpaged `fetch_leolabs_conjunctions` the evaluate selector uses is a
  separate concern; note it, do not refactor it here unless the deadline/executor
  work is shared cleanly.
- No new cross-request cache beyond what SCRUM-450 already provides.

## Tests

Offline, with a fake LeoLabs client so nothing hits the network:
- A slow client (each page sleeps) makes the list endpoint return within the budget
  with `complete: false`, not after 60 s.
- An oversized set (more CDMs than the cap) returns a capped, incomplete page, and
  the incomplete flag is set and preserved through the response.
- A normal small set returns complete, worst Pc first, unchanged from today.
- Concurrency: several simultaneous slow fetches do not block a `/health`-style
  quick request, and past the concurrency cap the endpoint sheds with 503 rather
  than hanging. Assert the quick request stays fast while heavy fetches run.
- The SCRUM-450 cache still serves a repeat query, and `reset_caches` clears
  anything new this ticket adds.
- An incomplete result never presents as complete: a test pins that the flag rides
  through `ConjunctionPage` to the JSON body.

Run `python3 -m pytest services/planner` and confirm green.

## Live acceptance

On the local stack, select SWARM B: the list returns within the budget with no
60 s timeout, marked complete or incomplete honestly, and the planner health never
goes red. Select other assets while a SWARM B fetch is in flight and confirm they
still return and `/health` stays green. Repeat-select the same asset and confirm the
SCRUM-450 cache serves it fast. Record the SWARM B window size (the
`leolabs_page_fetched total=` and the raw count) so we know how dense it is, in
docs/scrum-459/live_verification.md. Do not deploy to the KVM; that is John's manual
step once this merges.

## Commit

Branch off main. Include docs/scrum-459/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it confirms the
cold fetch is bounded and returns within budget, that an incomplete result can never
read as a complete or clear listing, that concurrent heavy fetches cannot starve
`/health` or the other assets, and that the completed-case ordering and the SCRUM-450
cache are unchanged, then reviews the live SWARM B behaviour.
