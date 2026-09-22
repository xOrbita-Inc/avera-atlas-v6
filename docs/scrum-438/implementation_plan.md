# SCRUM-438 implementation plan — pull all LeoLabs CDMs (paginate + C0)

## The defect (precise)

The LeoLabs client already has the pagination scaffolding but never turns it on.
`LeoLabsClient._paginate` (services/planner/common/leolabs_client.py) correctly
follows `nextToken` across pages, but no caller sets the `paginate` request
parameter. Per LeoLabs (Lois Reid, 2026-08-31 and again 2026-09-22): when
`paginate` is absent it defaults to false, the endpoint returns at most 1,000
entries and no `nextToken`, so `_paginate`'s loop terminates after the first page
and silently caps at 1,000. Both `search_conjunction_cdms` and `list_objects`
build their params without `paginate`, so a subscribed asset with more than 1,000
matching CDMs comes back truncated. That is exactly what Lois observed.

Not the bug: the runtime (`leolabs_runtime.py`) fetches per asset with
`object1=<catalog>`, which is correct for the per-asset evaluate. The
`object1='C0'` option (pull every accessible vehicle in one call) already works
through the method's passthrough, and `maxRelativePositionR/I/C` volume filters
already pass through via `**extra`. Neither is the defect.

## Scope

1. Set `paginate=true` on the paginated LeoLabs GET requests. Cleanest at the
   source: have `_paginate` include `paginate: "true"` in its request params so
   both CDM search and object listing page fully. Confirm the value format
   LeoLabs expects (string "true" vs boolean) against the API.
2. Cross-check completeness: read `total` from the first response and warn if the
   number of items yielded does not match, so a silent truncation cannot recur.
3. Keep `object1='C0'` reachable for a full-subscription pull (already works via
   the signature). Add a thin convenience only if a caller needs the bulk pull;
   otherwise leave the passthrough. Do not change the per-asset evaluate, which
   correctly uses `object1=<catalog>`.
4. Keep `maxRelativePositionR/I/C` available as optional filters (already via
   `**extra`); document them.

## Tests (services/planner/tests/test_leolabs_client.py)

- A mocked LeoLabs API where page 1 returns `{cdms:[...1000], total:1500,
  nextToken:"t"}` and page 2 returns `{cdms:[...500], total:1500}`. Assert
  `search_conjunction_cdms` returns all 1500, that the outgoing request carried
  `paginate=true`, and that the loop terminates when `nextToken` is absent (no
  infinite paging).
- Assert the same `paginate` flag is present on `list_objects`.
- Leave the per-asset live path unchanged (object1=<catalog> still works).

## Acceptance (from the ticket)

A live pull returns the full CDM set for a subscribed asset (retrieved count
matches `total`); the pagination loop terminates cleanly; the dashboard live
conjunction list is complete. Prerequisite for the on-demand screen (SCRUM-433)
so the plan is built against the complete CDM set.

## Files

- services/planner/common/leolabs_client.py (`_paginate` and/or the search/list methods)
- services/planner/tests/test_leolabs_client.py (new pagination test)
- services/planner/tests/test_leolabs_runtime.py only if a runtime fetch assertion needs updating

Design note: claude/leolabs-on-demand-screening-433-design-2026-09-22.
