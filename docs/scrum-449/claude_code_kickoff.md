# SCRUM-449 — kickoff for Claude Code

Fix the encounter selection race: selecting an Active Conjunctions row shows the
wrong (worst) conjunction because a stale live-evaluate repaints the panel over the
selection. Ticket: https://xorbita.atlassian.net/browse/SCRUM-449

Branch scrum-449-encounter-selection-race off main. Frontend only —
services/ui/app/templates/index.html. No planner change.

Read docs/scrum-449/implementation_plan.md first; it has the line anchors and the
exact fix.

Run in order:

1. Read docs/scrum-449/implementation_plan.md.
2. In evaluateConjunction (index.html:1680), add a monotonic sequence token (module
   `let evalSeq = 0;`, `const seq = ++evalSeq;` at the top) and, after every await,
   `if (seq !== evalSeq) return;` before any UI update — on the success path and the
   error/404 paths both. Latest selection wins, not latest network response.
3. Add an AbortController (module `let evalAbort = null;`), abort the previous one at
   the start of each evaluate, pass its signal to the /api/planner/evaluate fetch,
   and swallow AbortError silently so a cancelled evaluate never shows EVALUATE
   FAILED.
4. Leave autoEvaluateSelectedOrFirstPlannable, highlightRowGeometry and the request
   shape unchanged.
5. Rebuild only the ui container (docker compose up -d --build ui) and do the browser
   check in the plan on the running stack with SWARM C: select non-worst rows, click
   rapidly, toggle 2D/3D, and confirm the panel always follows the last selection and
   never flashes EVALUATE FAILED. Run python3 -m pytest services/planner once to
   confirm no regression (1456 / 2).
6. Commit on scrum-449-encounter-selection-race, include docs/scrum-449/ in the
   commit, push, and open a PR against main. No AI attribution. Report the PR number
   and head SHA.

Then Cowork runs the verification review before John merges, driving the browser to
confirm last-selection-wins on the rebuilt stack.
