# SCRUM-483 implementation plan: 2D encounter view off-center with a ghost trail

Ticket: https://xorbita.atlassian.net/browse/SCRUM-483
Branch off main. UI only. Independent of SCRUM-482.

> Note: `docs/scrum-483/` did not exist in the repo when this ticket was picked up —
> no `implementation_plan.md` had been committed on any branch. This file is the plan
> as recorded on the Jira ticket by John (2026-09-29), reproduced here so the folder
> is self-describing and matches the convention of the other scrum-NNN folders. The
> kickoff given to Claude Code and the ticket body agree; nothing here was invented.

## Symptom

Switching from the 3D globe back to the 2D encounter view leaves the encounter
off-center, and panning or double-click-center leaves a residual ghost trail of the
previous render.

## Cause

setView's 2D branch un-hides the canvas and draws directly, but never re-measures it.
The only resizeEncounter() calls are at startup and in the portlet expand/restore
handlers (both wrapped in requestAnimationFrame). While the 2D view is hidden during
3D, its element has zero size, so after a 2D->3D->2D round trip the canvas backing
store and the encounterW/encounterH used for clearing and centering are out of sync
with the element's real size. The center is computed from a stale width (off-center),
and clearRect(0,0,encounterW,encounterH) under the devicePixelRatio-scaled transform
no longer covers the full canvas, so each pan leaves an uncleared ghost.

Not caused by SCRUM-482. That change touched only the 3D camera-follow; the 2D path
is unchanged. This is a pre-existing issue surfaced by toggling views.

## Fix, UI only, services/ui/app/templates/index.html

- In setView's 2D branch, after un-hiding the 2D wrap, schedule resizeEncounter()
  inside a requestAnimationFrame, matching the portlet expand/restore pattern.
  resize() re-measures the element, rebuilds the backing store (setting canvas.width
  clears it and resets the transform to a single correct dpr scale), and redraws.
  This fixes both the off-center render and the ghost.
- Defense in depth: in drawEncounter and drawEncounterEmpty, clear the full backing
  store under an identity transform (save, setTransform(1,0,0,1,0,0),
  clearRect(0,0,canvas.width,canvas.height), restore) so a stale-dimension frame can
  never leave a trail even before the rAF fires.

Keep the existing synchronous draw in the 2D branch so there is no empty flash.
Do not change resize() itself, the planner, any response, or the starfield/autoRotate
behavior.

## Acceptance

Toggle 2D->3D->2D several times; the encounter renders centered every time, and
panning and double-click-center show no ghost trail. Existing UI tests stay green.
No planner or response change.

## Out of scope, noted separately

With autoRotate on, the starfield appears to rotate with the globe because autoRotate
orbits the camera around the whole scene. Cosmetic, tracked on its own if we decide
to change it.
