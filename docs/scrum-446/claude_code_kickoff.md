# SCRUM-446 — kickoff for Claude Code

Disable the 3D globe encounter view until it can be driven by live data. Ticket:
https://xorbita.atlassian.net/browse/SCRUM-446

The change is already made in the working tree by Cowork. Do NOT re-derive it.
`git diff` against main (0163fa8) shows exactly one file:

- services/ui/app/templates/index.html: the "3D GLOBE" toggle button (id="btn3D")
  now carries style="display:none" with a four-line comment above it explaining
  why (the globe renders surrogate propagator orbits, not live conjunctions) and
  how to re-enable (unhide the button). Nothing else is changed. The 2D view
  (btn2D) stays active, currentView still defaults to '2d', and the globe code
  (setView, initGlobe, the globe loop, fetchAndUpdateGlobeOrbits) is left intact
  and dormant. Since the only entry point to the 3D view is that button, hiding it
  fully disables the globe.

Why this is correct: currentView is '2d' by default and the only caller of
setView('3d') is btn3D, so with the button hidden the globe is unreachable. The
SCRUM-445 expandable panel and the row-to-geometry highlight are untouched.

Run in order:

1. Review the `git diff` for services/ui/app/templates/index.html and confirm it
   is the single btn3D change described above, nothing else.
2. This is UI-only, no .py in the diff, so the planner suite is main's by
   construction. Optionally run `python3 -m pytest services/planner` to confirm
   green; the change cannot affect it.
3. Commit on a branch named `scrum-446-disable-3d-globe`, including this
   docs/scrum-446/ folder in the commit. No AI attribution in the commit message
   or PR body.
4. Push and open the PR against main, then report the PR number and head SHA, and
   post the verification-first review verdict on the PR itself, not only in chat.

Cowork will review before John merges: read the committed diff, confirm the
toggle is hidden and the globe is unreachable, and check on the rebuilt local
stack that the encounter panel shows the 2D view only with no "3D GLOBE" toggle,
and that the 2D encounter and the SCRUM-445 panel still work. John is the only
one who merges. Development and testing stay on the local stack.
