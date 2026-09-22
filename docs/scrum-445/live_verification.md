# SCRUM-445 live verification — SWARM C (39453)

Run on 2026-09-22 against the real LeoLabs API, planner started directly with
uvicorn on port 8391 using the repo `.env` credentials. Docker was not available
on this machine, so the UI container was not part of this run: the planner
endpoint was exercised directly with curl. The frontend still needs the manual
check described in the kickoff.

## Acceptance: the timeout is gone

| request | status | wall time |
|---|---|---|
| `page_size=100` (in-volume default), cold | 200 | 25.2 s |
| `page_size=25` page 1, warm | 200 | 21.9 s |
| `page_size=25` page 2 via cursor | 200 | 16.7 s |
| `page_size=100&in_volume=false` | 200 | 19.1 s |

Previously this request overran the ui-to-planner 60 s read timeout and the
dashboard showed "Read timed out (read timeout=60)". Every page now returns
inside the timeout with roughly 2.5-3.5x margin. The 60 s proxy timeout was not
raised.

## What the numbers say

    in-volume : 1,841 CDMs -> 96 conjunction events
    widened   : 1,852 CDMs -> 96 conjunction events

Worst row: STARLINK-38108, Pc 4.69e-4, RED. Rows carry
`relative_position_rtn_m` with `relative_state_source: "leolabs_cdm_rtn"`.

Two things worth recording, because one of them contradicts the implementation
plan's reasoning:

1. **The reporting-volume filter is not what makes a dense asset tractable.**
   The plan assumed it was ("the reporting-volume filter provides that bound").
   Measured, it removed 11 of 1,852 raw CDMs and zero events. LeoLabs evidently
   only issues CDMs inside roughly this volume already. The filter is still
   implemented, defaulted on, and surfaced in the UI, because the ticket asks for
   it and an explicit narrower volume is a useful operator control — but it is not
   load-bearing for the timeout.

2. **Deduping before parsing is what does the work.** 1,841 CDMs collapse to 96
   events, a 19x reduction, and only the returned page is parsed. The old path
   parsed all 1,841 and deduped afterwards.

A consequence: for SWARM C the event total (96) is below the default page size
(100), so at the default the listing is a single page and `next_cursor` is null.
Paging was verified with `page_size=25`.

## Cursor paging, live

`page_size=25` page 1 -> page 2 via `next_cursor`: offsets 0 and 25, totals 96 and
96, no `cdm_id` on both pages, and Pc descending across the two pages combined
(page 1's worst >= page 2's worst). Global worst-Pc-first holds across the page
boundary, which is the property that makes the top of the table readable as the
worst conjunctions.

## Status contract, live

| case | expected | got |
|---|---|---|
| valid request | 200 | 200 |
| cursor replayed with a different `page_size` | 422 | 422 |
| garbage cursor | 422 | 422 |
| NORAD 99999, not subscribed | 404 | 404 |

Feed-off (503) was not re-exercised live — it is covered by the offline route
test, and turning the feed off on a live run proves nothing new.

## Known cost, not fixed here

Every page request re-fetches the window from LeoLabs; nothing is cached between
requests, so page two pays the same ~16-20 s fetch as page one. That is inside
the timeout but it is not "instant". A short-TTL cache of the ordered raw set,
keyed by the cursor's query fingerprint, would make page two onward nearly free.
Deliberately not built here: it adds cross-request state the story did not ask
for, and it wants its own decision about staleness on a live feed.

## Frontend verification

Docker was unavailable, and this machine's local `uvicorn app.main:app` fails in
Starlette's `get_template` with `TypeError: unhashable type: 'dict'` — a
Starlette/Jinja2 version mismatch in the host environment, thrown before the
template is parsed and unrelated to this change. So the page was rendered with
Jinja directly, served statically, and driven in Chrome with a stubbed page-1
payload (25 rows carrying RTN state, `total: 96`, a next cursor). That exercises
every part of the frontend change except the real network round-trip.

Verified:

- **Expanded split.** The existing maximize control opens the panel at 94vw/88vh
  with the table at 53% and the encounter geometry at 45%, side by side. The live
  `.encounter-canvas-wrap` is moved into the pane and moved back on restore
  (placeholder removed, panel restored, canvas back in the dashboard) — one
  canvas and one renderer, never a second copy.
- **Row to geometry, no recompute.** Selecting a row redraws the encounter from
  the row's own `relative_position_rtn_m`, with the header reading
  "<object> — CDM RTN geometry, unscored" and the risk colour taken from the row.
  No burn arrow is drawn, because nothing has been scored.
- **Pager.** Reads "1-25 of 96"; PREV disabled on page one, NEXT enabled.
  Forward/back over a three-page walk issues cursors `c1, c2` forward and
  `c1, null` back — a correct forward-only walk with prior cursors cached.
- **Volume control.** Defaults on, labelled "IN VOLUME" with "2 × 50 × 50 km RIC";
  toggling resets to page one, since the volume is part of what a cursor is
  valid against.

Two bugs were found by this check and fixed, neither of which any test caught:

1. `drawEncounter` dereferenced `result.recommendation` and `result.metrics`
   unguarded, so the new `drawEncounter(conj, null)` row-highlight path threw
   `TypeError: Cannot read properties of null` on **every** row click. The burn
   vector and post-maneuver envelope are scored-result concepts, so they are now
   guarded and simply not drawn for an unscored row.
2. The geometry pane ignored its 46% flex basis (flex items default to
   `min-width: auto`, so the canvas's intrinsic width won), squeezing the table to
   33%. Fixed with `min-width: 0`.

Still not verified here: the real browser-to-UI-to-planner round trip, i.e. the
dashboard fetching a live page for SWARM C through the UI proxy. That needs the
Docker stack and is the manual check the kickoff assigns to the reviewer.
