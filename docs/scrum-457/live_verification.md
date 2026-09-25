# SCRUM-457 live verification — the dashboard resolves the Secondary conflict row

Run 2026-09-25 against the real LeoLabs API on the local stack, ui rebuilt from
this branch. Asset **SWARM A**, LeoLabs **L3972**, NORAD **39452**.

## One gap up front

**The Chrome extension was not connected on this run**, so there is no GIF and no
in-browser screenshot of the row repainting. What that leaves unverified is
specifically the *visual* result: CSS, layout, and that the repaint is perceptible
to a human. Everything behind it was verified two other ways, both reproducible:

- the real HTTP path, end to end through the actual ui proxy against the live
  planner and LeoLabs (below), and
- `services/ui/tests/row_state_harness.mjs`, which extracts the real row-rendering
  and polling functions out of `index.html` and drives them under node against a
  stub DOM, asserting every state including the race cancel — **48/48 checks**.

The plan expected the row states to be covered live rather than offline. With the
browser unavailable I built the offline harness instead of leaving them unchecked,
and it is wired into pytest (skipped where node is absent). The browser check is
still worth doing before merge, and it is the one thing I could not do.

## Evaluate returns immediately, with the row pending

    POST /api/planner/evaluate (through the ui proxy)   http 200 in 17.9 s

    screen_pending   True
    screen_job_id    5b55e9d8d0aa4c288404ddc7d7e7e0ac

Comfortably inside the ui proxy's unchanged 60 s evaluate timeout. The screen is
not on this path at all — that was SCRUM-456's work, and this ticket does not put
any of it back.

## Poll transitions, through the ui proxy

`GET /api/planner/secondary-screen/{id}` at the dashboard's 5 s interval:

    02:45:11  http=200  status=pending    clear=False  screening_id=None    rtt=3 ms
    02:45:16  http=200  status=pending    clear=False  screening_id=None    rtt=4 ms
    ...                (21 polls in total)
    02:46:53  http=200  status=pending    clear=False  screening_id=None    rtt=4 ms
    02:46:58  http=200  status=not_clear  clear=False  screening_id=603659  rtt=5 ms
                                                       evaluated=1223

Pending for ~107 s after the evaluate returned, then resolved. **2 to 6 ms per
poll** through the proxy, which is what the short timeout is there to protect: the
planner side is a store read, so anything slow is a fault, not work in progress.

`clear` is `False` on every single line, including every pending one. Pending is
never a provisional pass.

## The row, in every state

From `row_state_harness.mjs`, driving the real extracted functions:

| state | row | class |
|---|---|---|
| polling | `SCREENING IN PROGRESS` + spinner | amber |
| resolved, a real clear | `CLEAR` | green |
| resolved `not_clear` | `NOT CLEAR` + breaching objects, limb, count, screening id, closest approach | red |
| resolved `error` | `NOT CLEAR` | red |
| 360 s cap hit | `NOT CLEAR` + what the bound was, no spinner left behind | red |
| pending artifact with no live poll | `NOT CLEAR` | red |
| stale result for a superseded evaluate | does not paint | — |

The clear case is written as an allow-list mirroring the planner's
`is_clear_status`, and the harness checks the cases that matter:
`{status:'clear',clear:false}`, `{status:'CLEAR',clear:true}`,
`{status:'cleared',clear:true}`, `{clear:true}` with no status,
`{status:'clear'}` with no flag, `{}`, `null` and `undefined` **all** read not
clear. Only `{status:'clear',clear:true}` paints CLEAR.

The four pre-async shapes (DEFERRED, NOT CHECKED, CLEAR, CONFLICT DETECTED) are
asserted unchanged, so a scenario preset or a screen-off build renders exactly as
it did.

## Bounded polling

Cap 360 s against the planner's own 300 s screening budget. Past the cap the
harness confirms the poll **issues no further request** and the row reads NOT CLEAR
with a note naming the bound. There is no state in which a spinner can outlive the
cap.

## Latest selection wins

The SCRUM-449 `evalSeq` token and an AbortController, applied to this row:

- `cancelSecondaryScreenPoll()` runs at the top of `evaluateConjunction`, beside
  the existing `evalAbort.abort()`, clearing the timer, aborting the in-flight poll
  and dropping the row state.
- `pollSecondaryScreen` re-checks `seq !== evalSeq` *after* the await, the same
  rule SCRUM-449 established, so a screen result that lands after the operator
  moved on cannot repaint the row.
- The row itself only honours poll state whose `seq` matches the current
  `evalSeq`.

Harness checks: a poll for a superseded evaluate **issues no request at all**; an
aborted poll is not treated as a failure and does not repaint; and a stale `CLEAR`
for a superseded evaluate does not paint the row — it falls back to the fail-closed
reading. That last one is the dangerous case, and it is the one most worth having a
test for.

## Failing closed, forced live

**1. Unknown or evicted job id**, through the real proxy:

    HTTP 404
    status = error | clear = False | pending = False
    error  = Planner returned no screen status for job deadbeefdeadbeef (HTTP 404).
             no secondary screen with id 'deadbeefdeadbeef'; it never existed or
             has been evicted. This is NOT a clear screen.
    conjunctions = []   verdict = None

The planner answers a 404 with an error envelope that carries no `status`. Relaying
that unexamined would leave the front end guessing, so the proxy converts anything
that is not a recognisable poll payload into the not-clear shape.

**2. Planner stopped mid-poll** — `docker compose stop planner`, then poll a job
that had already resolved:

    planner up:      status = not_clear | clear = False
    planner stopped: HTTP 503
                     status = error | clear = False | pending = False
                     error  = Planner service unavailable.
                     note   = Secondary screen result could not be retrieved, so
                              safety could not be established. Treating as NOT CLEAR.

This is the failure most likely to happen for real — a planner restart during a
screen — and it reads NOT CLEAR rather than hanging or erroring into something the
row could misread. Planner restarted and healthy afterwards.

Worth noting as a property rather than a defect: SCRUM-456's store is in-process,
so a planner restart loses pending screens and they then read as unknown, which is
NOT CLEAR. That is the safe direction, and durable persistence is SCRUM-453.

A third forced failure is asserted offline rather than live: a genuine screen
`error` payload (LeoLabs rejecting the create) resolves the row to NOT CLEAR. The
payload shape for that was verified live in SCRUM-456; here the harness checks the
row's reading of it. I attempted to force one inside the planner via
`docker compose exec`, and it 404'd — the store is per-process and `exec` starts a
separate one, so the running server never saw it. Recording that because it looks
like a bug and is not.

## Not changed

No planner change. The evaluate proxy timeout is still 60 s, the screen is still
off the evaluate path, and the poll endpoint, the screen and the evaluate contract
are untouched. The verification panel's own "Secondary conflict clear" row still
comes from the planner's verification result, which SCRUM-456 already made honest
about the provisional case; this ticket deliberately leaves it alone.

## Tests

    python3 -m pytest services/ui/tests      15 passed
    python3 -m pytest services/planner       1760 passed, 2 skipped
    node services/ui/tests/row_state_harness.mjs    48/48 checks passed

These are the first tests in the ui service — there was no suite here before, so
`services/ui/conftest.py` mirrors the planner's. It also imports `app.main` with the
working directory set to `services/ui`, because main.py mounts its static files at
a cwd-relative path that only resolves from there; the fix stays in the harness
rather than changing how the service mounts its assets.
