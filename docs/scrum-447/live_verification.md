# SCRUM-447 live verification — 3D globe on live LeoLabs data

Run 2026-09-23 on the local Docker stack (planner and ui rebuilt from this
branch) against the real LeoLabs API. Asset: SWARM C (39453).

## Confirm-live, before any endpoint code

Recorded in full at the end of `implementation_plan.md`. Three assumptions in the
plan did not survive it:

1. `get_states("39453")` is a **404** — the call takes a catalog number, not a
   NORAD. `get_states("L3969")` works.
2. Position and velocity are under `frames.EME2000` in a `states` **list**, and
   are in **metres**, not km. Built against the assumed km, every track would
   have been 1000x too large.
3. `get_states` on a conjunction secondary is **403** — the subscription does not
   cover arbitrary catalog objects. Secondary states therefore come from the
   already-parsed CDM SAT2 block, which also means zero extra API calls and no
   need for the cache the plan proposed.

## Planner route, live

    GET /v1/leolabs/orbits?primary_norad=39453   ->  200 in 20.9 s

    asset        L3969, epoch 2026-09-23T19:00:17Z, source leolabs_get_states
    asset_track  91 points, |r| 6794.5 km
    rings drawn  18   (from 116 conjunction events)
    risk bands   1 RED, 1 AMBER, 2 GREEN, 14 NOMINAL
    worst        58483 STARLINK-30994, Pc 1.39e-04, miss 4906.6 m
    skipped      0

**Ring count cross-checked against the 2D table.** The conjunctions route
returned 116 events over 18 distinct secondaries (the busiest object appears 29
times), and the globe drew exactly those 18 — so "18 from 116" is dedupe, not
truncation. Risk bands were compared object by object between the two endpoints:
**zero mismatches**. That is acceptance 1's "same risk bands as the 2D panel",
measured rather than asserted.

## Status contract, live

| case | expected | got |
|---|---|---|
| valid request | 200 | 200 |
| NORAD 99999, unsubscribed | 404 | 404 |
| lookahead_days=60 | 422 | 422 |
| steps=2 | 422 | 422 |
| max_objects=-1 | 422 | 422 |
| `LEOLABS_ENABLED=false` | 503 | 503 |

Feed-off was exercised against a real planner container started with the flag
off, not only in the unit test: both `/v1/leolabs/orbits` and
`/v1/leolabs/conjunctions` returned 503 with an explanatory body and no track
data. An empty globe under a 200 never occurs.

## UI proxy and the browser

`GET /api/orbits/live?primary_norad=39453` through the ui container: 200 in
24.0 s, same payload.

Driven in Chrome against the rebuilt stack, Live Asset = SWARM C:

| acceptance | result |
|---|---|
| 1. asset + conjunctions from live states, risk-banded | 18 rings + 18 markers, asset track 92 pts (91 + closing point), bands match the table |
| 2. nominal haze, toggle drops it; labels on asset + RED/AMBER only | 28 haze parts (14 NOMINAL x line+marker) at opacity 0.12 vs 0.85 for RED; toggle 28 -> 0 -> 28 and never touches a non-nominal ring; 2 object labels drawn out of 18 slots, plus the asset label |
| 3. worst conjunction carries the dashed link | `globeWorstLink` present, index 6, re-pointed each frame |
| 4. 2D stays default, globe fetched only on switching to 3D | after load and asset select: `currentView` `'2d'`, no `/api/orbits*` fetch, `globeInitialized` false |
| 5. feed-off is 503, not an empty globe | see above |

The operator path was exercised end to end by clicking the real `3D GLOBE`
button, not by calling `setView` directly: header reads `2D | 3D GLOBE | NOMINAL
| ORBIT SPEED`, the button takes the active class, and the globe draws.

`/api/orbits` (scenario mode) still returns 91-point tracks at the same radii
after the propagator moved, so the shared helper serves both paths.

## Suites

    python3 -m pytest services/planner        1428 passed, 2 skipped  (+24)
    python3 -m pytest libs                     230 passed             (+11)
    python3 -m pytest libs services/planner services/propagator
                                              1736 passed, 2 skipped

## Not verified

- **Anything but SWARM C.** One asset was exercised live. The other nine
  subscribed assets share the code path but were not individually checked.
- **A secondary genuinely lacking a usable state.** SWARM C skipped zero objects,
  so the skip path is covered by the offline test only; it has not fired against
  real data.
- **Sustained rendering cost.** 18 rings animate smoothly; a much denser asset
  drawing at the 250-object cap was not profiled in the browser.
