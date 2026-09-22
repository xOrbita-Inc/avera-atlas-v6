# SCRUM-438 — kickoff for Claude Code

Implement SCRUM-438 in this repo. Planner-side LeoLabs client fix; the UI is
untouched. Ticket: https://xorbita.atlassian.net/browse/SCRUM-438

Context: the LeoLabs client already paginates via `LeoLabsClient._paginate`
(it follows `nextToken`), but no request sets the `paginate` parameter, so
LeoLabs defaults it to false, caps every search/list at 1,000 entries, and
returns no `nextToken` — so `_paginate` stops after one page and the CDM pull is
silently truncated. LeoLabs flagged this twice (Lois Reid, 2026-08-31 and
2026-09-22). Full analysis in docs/scrum-438/implementation_plan.md and the
design note claude/leolabs-on-demand-screening-433-design-2026-09-22.md.

Run in order:

1. Read docs/scrum-438/implementation_plan.md.
2. Set `paginate=true` on the paginated LeoLabs GET requests (in
   `LeoLabsClient._paginate` in services/planner/common/leolabs_client.py, or on
   the search/list methods), confirming the value format LeoLabs expects. This
   fixes both `search_conjunction_cdms` and `list_objects`.
3. Cross-check the retrieved count against the response `total` and log a warning
   on mismatch.
4. Leave `object1='C0'` reachable (already via the signature) for a future
   full-subscription pull; do NOT change the per-asset evaluate
   (`object1=<catalog>`). Keep `maxRelativePositionR/I/C` available.
5. Add tests in services/planner/tests/test_leolabs_client.py: a mocked two-page
   CDM search (total 1500 across two pages) returns all 1500, the request carries
   `paginate=true`, and the loop terminates cleanly. Assert the same flag on
   `list_objects`.
6. Run the planner suite from the repo root and confirm green:
   `python3 -m pytest services/planner`
   The change touches no UI and no propagator, so those suites are unaffected.
7. Commit on a branch named `scrum-438-cdm-pagination`, including this
   `docs/scrum-438/` folder in the commit (matches the other scrum-NNN docs).
   No AI attribution in the commit or PR body.
8. Push and open the PR against main, then report the PR number and head SHA.

Cowork will do the verification-first review (read the committed code, run the
planner suite on the branch, confirm the paginate flag and the two-page test)
before John merges.
