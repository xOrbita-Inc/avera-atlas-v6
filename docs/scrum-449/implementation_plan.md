# SCRUM-449 — implementation plan: fix the encounter selection race

Ticket: https://xorbita.atlassian.net/browse/SCRUM-449
Branch off main. Frontend only: services/ui/app/templates/index.html. No planner change.

## The bug

In Live Asset mode the 2D and 3D encounter header and risk chip show the worst
conjunction (STARLINK-30994 / 58483) no matter which Active Conjunctions row is
selected. Selection is correct; the display is not. Diagnosed live: a real click on
row 3 scores the right CDM, but a stale evaluate repaints the panel over it.

Why: evaluateConjunction repaints the panel unconditionally at the end of its
round-trip (updateRiskBar(result, conj) and drawEncounter(conj, result) near
index.html:1770), with no check that conj is still selected and no request
sequencing. autoEvaluateSelectedOrFirstPlannable() fires an evaluate for the worst
row (index 0) after every fetch (index.html:1232) and on setView (index.html:3296).
A live evaluate re-fetches the whole LeoLabs window (~13 s, measured, serialized), so
the worst-row evaluate lands 13 to 26 s after a click and repaints over the
selection. Reproduced: firing the worst then selecting row 3 painted 58483 first,
then corrected to STARLINK-38092 thirteen seconds later.

## Fix (all in evaluateConjunction, index.html:1680, and its module scope)

1. Sequence token. Add a module var `let evalSeq = 0;`. At the top of
   evaluateConjunction: `const seq = ++evalSeq;`. After EVERY await in the function
   (the fetch and the response .json()), before touching any UI, bail if stale:
   `if (seq !== evalSeq) return;`. Apply it on the success path AND the error/404
   paths — a stale error must not repaint or show EVALUATE FAILED. This makes the
   latest selection the only evaluate that can paint.

2. Abort the prior in-flight request. Add a module var `let evalAbort = null;`. At
   the top of evaluateConjunction: `if (evalAbort) evalAbort.abort(); evalAbort = new
   AbortController();`, and pass `signal: evalAbort.signal` to the
   fetch('/api/planner/evaluate', ...). Swallow an AbortError silently — a request
   cancelled because the operator moved on is not a failure and must not call
   showEvaluateError. The seq guard already stops a stale success from painting;
   aborting also unqueues the planner so the new click is not stuck behind ~13 s of
   stale work.

## Do not

- Do not change the planner, the endpoint, or the evaluate request shape.
- Do not remove the pre-score row geometry (highlightRowGeometry) — that instant draw
  is correct; it is the post-await repaint that needs guarding.
- No need to gate setView's auto-evaluate: with the seq guard its worst-row evaluate
  simply loses to any later selection, and once a row is picked it auto-evaluates the
  user's own selectedConjunction anyway.

## Verify (frontend only)

- Rebuild only the ui container: docker compose up -d --build ui. Hard-refresh.
- Live Asset SWARM C, fetch conjunctions, 2D view. Select a non-worst row (a
  STARLINK-38092) and confirm the header, risk chip, decision card and geometry
  settle on THAT object and stay — the worst-row auto-evaluate must not take it back
  to 58483.
- Rapidly select several different rows; confirm the panel ends on the last one
  selected, with no prior object's numbers left behind.
- Toggle 2D and 3D; confirm the shown conjunction does not change.
- Confirm no EVALUATE FAILED flashes from an aborted request.
- No JS test harness exists in the repo, so the browser check above is the
  verification. Run python3 -m pytest services/planner once to confirm no regression
  (still 1456 passed / 2 skipped) — it should be untouched.

## Commit

Branch scrum-449-encounter-selection-race off main. Include docs/scrum-449/ in the
commit. No AI attribution. PR against main; report the PR number and head SHA.

Cowork runs the verification review before John merges, and will drive the browser to
confirm last-selection-wins on the rebuilt stack.
