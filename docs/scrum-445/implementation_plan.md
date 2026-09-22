# SCRUM-445 implementation plan — expandable Active Conjunctions panel with paging, volume filters, and row-to-geometry

## Why

The Active Conjunctions listing fetches every scorable CDM for an asset in one
synchronous call. Since SCRUM-438 made the pull complete (no longer capped at
1,000), a dense asset returns a large set, and the ui-to-planner request
overruns its 60 s read timeout. SWARM C (39453) currently fails with
"Read timed out (read timeout=60)". The fix is to bound each request with
paging and the reporting-volume filters, and to present the list as an
expandable in-dashboard panel rather than one unbounded table.

## The seams (already located)

- Planner route: `GET /v1/leolabs/conjunctions` in `services/planner/server.py`
  (around line 1281). Params today: `primary_norad`, `lookahead_days`,
  `lookback_days`. It calls `fetch_leolabs_conjunctions` and returns the whole
  parsed list.
- Runtime: `fetch_leolabs_conjunctions` in
  `services/planner/common/leolabs_runtime.py` (around line 247), which fetches
  and parses every scorable CDM in the window via `_scorable_in_risk_order`,
  ordered highest Pc first, earliest TCA as tie-break.
- Client: `search_conjunction_cdms` in
  `services/planner/common/leolabs_client.py` already accepts the `token` cursor
  and the `maxRelativePositionR/I/C` volume filters via passthrough (SCRUM-438),
  and `_paginate` sends the cursor correctly. No new client work.
- Row builder: `conjunction_row` in
  `services/planner/common/leolabs_conjunction_list.py`. A row already carries
  its selectors (`cdm_id`, `event_id`, `secondary_norad`), identity, the display
  numbers (miss, Pc, TCA), and provenance. It does NOT currently carry the
  relative encounter geometry.
- UI proxy: `GET /api/planner/v1/leolabs/conjunctions` in
  `services/ui/app/main.py` (around line 124). It already relays all query
  params to the planner, so it forwards new params without change, but it must
  return the new `next_cursor`/`total` fields to the frontend and keeps the
  status-code passthrough (503 vs empty 200) intact.
- Frontend: the Active Conjunctions panel and the encounter-geometry viz in
  `services/ui/app/templates/index.html`.

## Server side

1. Add optional params to the planner route: `page_size` (default a sane page,
   e.g. 100, capped), `cursor` (the forward token from a prior page), and the
   three `maxRelativePosition*` volume filters. Thread them into
   `fetch_leolabs_conjunctions`.
2. In `fetch_leolabs_conjunctions`, pass the volume filters and the cursor down
   to `search_conjunction_cdms`, and return one page of parsed rows plus the
   next cursor and the reported total, rather than the whole set. Parse only the
   page's CDMs, not the whole catalog window, so cost is bounded.
3. Ordering: keep highest Pc first, earliest TCA tie-break, within the returned
   page. A true global Pc sort needs the bounded set; the reporting-volume
   filter provides that bound. For a typical asset the in-volume set is small
   enough to fetch whole and page client-side while keeping the global sort; if
   an asset's in-volume set is still large, page at the API and sort within the
   page. Either is acceptable as long as the first page renders fast; pick the
   simpler one that meets the SWARM C acceptance and note which.
4. Keep `dedupe_by_event` behavior. Keep the exact status-code contract: 200 for
   a listing (possibly empty), 404 for an unsubscribed NORAD, 422 for a window
   over the 30-day cap, 503 for feed-off or fetch failure. Do not collapse
   feed-off into an empty 200.
5. Response shape: `{ rows: [...], next_cursor: <str|null>, total: <int|null>,
   in_volume: <bool> }` (final field names at the implementer's discretion, but
   the frontend needs rows, a next cursor, and the total).

## Frontend

6. Make Active Conjunctions an expandable panel using the existing maximize
   control. Collapsed, it shows the top rows by Pc. Expanded, it opens as a
   large split that keeps the encounter-geometry viz visible, with a paged,
   sortable, filterable table.
7. Load-next-page via the returned cursor (forward-only; cache prior cursors to
   step back). Default the reporting-volume filter on, with a control to widen.
   Sort worst Pc first within the returned set.
8. Row-to-geometry: selecting a row highlights that conjunction's encounter
   geometry in the viz. This is select-and-highlight of geometry already
   available for the row, NOT a recompute or a fresh propagation. The row does
   not currently carry the relative state the viz needs, so add the minimum to
   the row payload (the relative position, and velocity if the viz uses it, at
   TCA, from the parsed CDM) so the viz can draw the encounter without a
   server round-trip on click. Keep this addition small and provenance-labeled.

## Do not touch

- The decision and secondary-screen path. It still uses the complete in-volume
  set and runs off the blocking request path. This story is the display and the
  listing endpoint only.
- The 60 s proxy timeout is not the fix; do not just raise it. With paging each
  request is one bounded page and returns well inside it.

## Tests

- Planner route: a page returns `next_cursor` and `total`; a second call with
  that cursor returns the next page; the `maxRelativePosition*` filters reach the
  client; ordering is worst-Pc-first within the page; the 200/404/422/503
  contract is unchanged (extend the existing SCRUM-422 route tests).
- Runtime: `fetch_leolabs_conjunctions` passes the cursor and volume filters
  through and parses only the page.
- Row: the added relative-state fields are present and provenance-labeled.
- Run the planner suite from the repo root and confirm green:
  `python3 -m pytest services/planner`.
- Frontend: manual check that SWARM C (39453) renders its first page quickly and
  does not time out, paging works, and selecting a row highlights its geometry.

## Git

- Branch `scrum-445-conjunctions-paging`. Include this `docs/scrum-445/` folder
  in the commit. No AI attribution in the commit message or PR body.
- Push and open the PR against main; report the PR number and head SHA, and post
  the review verdict on the PR itself, not only in chat.
