# SCRUM-459 live verification — a dense asset can no longer take the planner down

Run 2026-09-25 on the local stack against the real LeoLabs API. Planner rebuilt
from this branch. No KVM deploy; that is John's manual step.

## How dense SWARM B actually is

Measured directly against the API, in-volume, 1 day back / 7 days ahead:

| asset | catalog | raw CDMs | pages | wall clock | per page |
|---|---|---|---|---|---|
| **SWARM B** | **L5429** | **75,257** | **77** | **773 s** | 10.0 s |
| SWARM A | L3972 | 1,766 | 3 | 20.3 s | 6.8 s |
| SWARM C | L3969 | 1,519 | 3 | 17.6 s | 5.9 s |

SWARM B flies in formation with the other two, so it sees **43x** their
conjunctions and its window takes **773 seconds** — nearly **13x** the
ui-to-planner 60 s read timeout. This was never a request that was going to
finish. On the endpoint's own default window (0 back / 7 ahead) LeoLabs reports
**53,034** CDMs in the window.

## The concurrency model, which the ticket asked me to establish first

The plan's premise was that `/health` still answered in 0.2 ms during a single
SWARM B fetch, so the fetch was "slow, not the event loop cleanly wedged", and that
the planner only went red under two or three concurrent fetches.

**That is not what happens, and the 0.2 ms is a measurement artifact.**

Both list routes are `async def` and called the blocking
`fetch_leolabs_conjunction_page` directly. The planner runs `uvicorn server:svc`
with no `--workers`, so one worker, one event loop. A blocking call in a coroutine
holds that loop for its whole duration. Measured from a client:

    /health, planner idle                      1.2 - 1.6 ms
    /health, during ONE SWARM B list fetch     never answered (120 s curl timeout)

And in the planner's own log for that same request:

    {"path": "/health", "status": 200, "elapsed_ms": 0.2}

Both numbers are real. The request-logging middleware starts its clock inside
`call_next`, after the event loop has picked the request up, so it measures the
handler and not the wait in front of it. The handler really did take 0.2 ms; it
just did not get to run for two minutes. A single fetch wedges everything — not
two or three.

That changes the emphasis of the fix. Taking the fetch off the loop is not a
defence against concurrency, it is what makes even one fetch survivable.

I nearly repeated the error in my own test: the first version of
`test_health_answers_while_a_heavy_fetch_runs` started its timer *after* the heavy
fetch was under way and so passed against the blocking code. It now times from
before the fetch starts and asserts `/health` returns while the fetch is still
running. Verified by reverting the offload: 4 of the 6 concurrency tests fail,
including that one and "another asset still lists".

## SWARM B now returns, and says what it is

    GET /v1/leolabs/conjunctions?primary_norad=39451&page_size=25

    http=200 in 35.8 s        (and 36.2 s on a second cold run)

    complete    False
    partial     True
    truncation  {cdms_pulled: 3000, cdms_in_window: 53034, truncated_by: "deadline"}
    total       382 events, from the prefix
    count       25 rows on this page

Inside the 60 s timeout, with the 25 s deadline plus the one page already under way
(~10 s) — exactly the overshoot the deadline's design accounts for.

**3,000 of 53,034 is 5.7% of the window, and the response says so.** That is the
whole point: CDMs arrive in LeoLabs order, not Pc order, so this prefix may not
contain the worst conjunction. A short list presented as complete would read as a
quiet sky, which is the SCRUM-458 failure in a new place.

## /health and the other assets, during a SWARM B fetch

Planner restarted for a cold cache, SWARM B fetch started, then probed while it ran:

    /health  during the fetch    2.7 / 2.4 / 2.2 / 2.0 / 2.2 ms
    SWARM C  during the fetch    http=200 in 14.6 s, complete=True, 97 events

So `/health` is unaffected and another asset lists normally and completely while
the dense one is still pulling. Before this change, both were impossible.

## Load shedding at the cap

Three cold fetches started 0.4 s apart against a cap of two, then a fourth:

    asset 39453 (3rd)  http=503 in 3.2 ms
    asset 36508 (4th)  http=503 in 2.3 ms
    /health throughout        1.7 ms

The shed body:

> The planner is already pulling 2 conjunction windows. This is a busy signal, not
> an empty sky: retry in a few seconds. (2 LeoLabs window fetches already in flight
> (limit 2))

Shed immediately rather than queued — 2 to 3 ms, not a wait behind two 25 s pulls
that would spend the new request's whole budget. And the body carries **no**
`conjunctions` key, so a refused fetch can never be read as an empty sky.

## The SCRUM-450 cache still does its job

    repeat select of SWARM B    http=200 in 11.8 ms

No second cache was added. The bounded result is cached the same way, flagged
incomplete, so a repeat within the TTL serves the partial view rather than spending
another deadline to arrive at another prefix.

## A pre-existing bug this ticket surfaced

I needed the window size for the "3,000 of 53,034" denominator, and it kept coming
back null. The cause:

    total value = '22008'   type = str

**LeoLabs returns `total` on a paginated search as a string.** The client tested
`isinstance(raw_total, (int, float))`, which a string never satisfies, so `total`
stayed `None` on every live response.

The consequence is not cosmetic. SCRUM-438 added a retrieved-versus-total
cross-check *because* silent truncation was the failure mode it feared, and that
check has therefore **never run against the live API**. It could not fire, so its
silence was not evidence that the counts agreed — and SCRUM-439 recorded that
silence as exactly that, calling the absent warning correct behaviour under a long
pull.

Fixed with a parser that accepts a numeric string and rejects bool (an int
subclass, which would otherwise read as a total of 1). Four tests cover it,
including the end-to-end string path into the response.

## What is unchanged

- **The completed case.** SWARM A and C still come back complete, worst Pc first,
  exact event totals, one event per reissue. Asserted offline, and SWARM C returned
  `complete=True` with 97 events live.
- **The evaluate path.** Its window pull is deliberately left unbounded: a row
  selector must resolve against the whole window, and a truncated one would 404 the
  operator's own selection or resolve it to a different event. Bounded and unbounded
  pulls are separate cache entries so one can never be served to the other, and a
  test asserts the evaluate path asks for no bounds.
- **The secondary screen (SCRUM-456/457) and the dashboard.** Untouched.

## One deliberate extension beyond the ticket's wording

The ticket names the conjunction list. `/v1/leolabs/orbits` — the 3D globe's
backend — calls **the same** `fetch_leolabs_conjunction_page` on the same event
loop, so selecting SWARM B on the globe wedged the planner identically. Bounding it
came free with the shared fetch; I also put it behind the same executor and cap, and
gave its response the same `complete` / `partial` / `truncation` fields. Leaving the
identical wedge one route over would not have met the ticket's own guarantee that
one dense asset can never take the view down. Flagging it because it is a route the
kickoff did not name.

The globe needed the honesty fields more than the table, not less: an operator
reads an uncluttered globe as a quiet sky.

## Still on the event loop, and out of scope

`/v1/evaluate` is also `async def` and also does blocking LeoLabs work directly, so
it still blocks `/health` for its duration (~17 s measured in SCRUM-456, not the
minutes SWARM B took). The ticket puts the evaluate path out of scope and I have
left it alone. It is the same class of defect and worth its own ticket; the
`_run_list_fetch` helper here is the shape that would fix it.

## Numbers to revisit first

SWARM A completes in 20.3 s against a 25 s deadline — 4.7 s of headroom, thinner
than is comfortable. A slower day at LeoLabs would have SWARM A report partial where
it used to report whole. It would say so, which is the point, but if partial views
start appearing on the sparse assets this is the number to look at. Raising the
deadline is not free: the worst case is the deadline plus the client's 30 s
per-request timeout, so 35 s would put it past 60 s. A larger budget needs a shorter
per-request timeout for this pull first. Noted rather than done.

## Tests

    python3 -m pytest services/planner services/ui/tests    1820 passed, 2 skipped

New `test_bounded_list_fetch.py`, 39 cases, in eight groups: the deadline, the cap,
incomplete never presenting as complete (including through the JSON body and the
globe response), the completed case unchanged, the client's bounding contract, the
cache carrying completeness, concurrency, and the string-total parsing. The
concurrency group drives the real ASGI app over httpx's async transport, because a
sync test client cannot show a blocked event loop.
