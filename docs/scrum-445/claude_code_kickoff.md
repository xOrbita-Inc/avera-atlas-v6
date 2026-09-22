# SCRUM-445 — kickoff for Claude Code

Implement SCRUM-445 in this repo: make the Active Conjunctions listing paged and
volume-filtered so a dense asset no longer times out, present it as an expandable
in-dashboard panel, and link a selected row to its encounter geometry. Ticket:
https://xorbita.atlassian.net/browse/SCRUM-445

Context: the listing fetches every scorable CDM for an asset in one synchronous
call. Since SCRUM-438 made the pull complete, a dense asset (SWARM C, 39453)
overruns the 60 s ui-to-planner read timeout. Full analysis, the exact seams, and
the design decisions are in docs/scrum-445/implementation_plan.md. Read it first.

Run in order:

1. Read docs/scrum-445/implementation_plan.md. It names every seam
   (planner route, leolabs_runtime, the client, the row builder, the UI proxy,
   the frontend panel and viz) with file and approximate line.
2. Server side: add page_size, a forward cursor, and the maxRelativePositionR/I/C
   volume filters to GET /v1/leolabs/conjunctions and thread them through
   fetch_leolabs_conjunctions to search_conjunction_cdms (which already accepts
   the token cursor and the volume filters from SCRUM-438). Return one page plus
   the next cursor and the total. Keep dedupe_by_event, keep worst-Pc-first
   within the page, and keep the exact 200/404/422/503 status contract, including
   feed-off as 503 and never as an empty 200.
3. Frontend: make Active Conjunctions an expandable panel using the existing
   maximize control, opening as a large split that keeps the encounter-geometry
   viz visible. Page via the returned cursor (forward-only; cache prior cursors
   for stepping back), default the reporting-volume filter on with a widen
   control, sort worst Pc first. Selecting a row highlights that conjunction's
   geometry in the viz as select-and-highlight only, no recompute or fresh
   propagation; add the minimum relative-state fields to the row payload (from
   the parsed CDM) so the viz can draw the encounter without a round-trip.
4. Do NOT touch the decision or secondary-screen path, and do NOT just raise the
   60 s proxy timeout. With paging each request is one bounded page.
5. Tests: extend the SCRUM-422 planner route tests for paging (next_cursor and
   total present, a second call with the cursor returns the next page, volume
   filters reach the client, worst-Pc-first within the page, status contract
   unchanged) and the runtime and row builder as the plan describes. Run the
   planner suite from the repo root and confirm green:
   `python3 -m pytest services/planner`
   The frontend change is verified manually: SWARM C (39453) renders its first
   page fast and does not time out, paging works, and a row selection highlights
   its geometry. Development and testing are on the local stack only.
6. Commit on a branch named `scrum-445-conjunctions-paging`, including this
   docs/scrum-445/ folder in the commit. No AI attribution in the commit message
   or PR body.
7. Push and open the PR against main, then report the PR number and head SHA, and
   post the verification-first review verdict on the PR itself, not only in chat.

Cowork will do the verification-first review (read the committed diff, run the
planner suite on the branch, confirm the paging and the status contract, and
check the frontend against the SWARM C acceptance on local) before John merges.
John is the only one who merges.
