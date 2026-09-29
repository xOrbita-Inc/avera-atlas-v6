# SCRUM-483 live verification: 2D encounter view after a 3D round trip

Branch `scrum-483-encounter-view-resize`, off main at `ca95d99`.
Verified 2026-09-29 against the local Docker stack, ui rebuilt from this branch.
Not deployed to the KVM.

## What changed

`services/ui/app/templates/index.html` only, three edits. No planner change, no
response change, no change to resize() itself, the 3D globe, or autoRotate.

1. `setView`'s 2D branch schedules `requestAnimationFrame(function(){ resizeEncounter(); });`
   after un-hiding the wrap, reusing the idiom the portlet expand/restore handlers
   already use. The synchronous `drawEncounter`/`drawEncounterEmpty` call is kept, so
   nothing flashes empty while the frame is pending.
2. `drawEncounter` and `drawEncounterEmpty` clear the full backing store under an
   identity transform (`save` / `setTransform(1,0,0,1,0,0)` /
   `clearRect(0,0,canvas.width,canvas.height)` / `restore`) instead of clearing
   `encounterW x encounterH` through the dpr scale.

## The mechanism, observed directly

While the globe is up, the 2D wrap really does measure zero, and the globals keep
the old numbers. Read live from the running page with the view in 3D:

    element (wrap.clientWidth/Height):  [0, 0]
    encounterW / encounterH:            [1910, 1832]

That is the whole bug in two lines: `resize()` bails on a zero measurement, so every
window-resize seen during 3D was discarded, and the globals describe an element that
no longer matches the canvas.

## The ghost, reproduced and eliminated

The decisive test. With the globals deliberately left stale at 60% of the element
(what a round trip leaves behind), the entire backing store was filled magenta and
then the view was drawn, sampling two points that a short clear cannot reach:

| sample point | shipped clear | the clear this replaces |
|---|---|---|
| just outside the stale region (968,1061) | `rgba(0,0,0,0)` | `rgba(255,0,255,255)` |
| far corner (1594,1749) | `rgba(0,0,0,0)` | `rgba(255,0,255,255)` |

Canvas backing 1597x1752, element 1065x1168, dpr 1.5. The old clear was applied by
hand under identical conditions for the comparison, so the two columns differ only in
the clear. The shipped code wipes both points; the old one leaves the fill behind —
that retained fill is exactly the ghost users saw.

## The off-center render, repaired

Starting from the same deliberately stale globals and then running the callback that
setView now schedules:

| | encounterW/H | centre the draw would use |
|---|---|---|
| stale (the bug) | 639 x 701 | (242.8, 336.5) |
| after the scheduled resize | 1065 x 1168 | (404.7, 560.6) |

The correct centre for a 1065x1168 element is (404.7, 560.6). The resize moves the
render back by 161.9px horizontally. Backing store and globals both end in sync.

(The centre is `W*0.38, H*0.48`, not `W/2` — the asset is deliberately left of centre
so the burn arrow has room, per the comment already in `drawEncounter`.)

## Round trips

Three consecutive 2D->3D->2D round trips, each followed by a pan and a
double-click-centre:

| trip | rAF callbacks scheduled by the 2D branch | globals in sync | backing store in sync |
|---|---|---|---|
| 1 | 1 | yes | yes |
| 2 | 1 | yes | yes |
| 3 | 1 | yes | yes |

Exactly one resize is scheduled per return to 2D — it is not double-scheduled, and
not scheduled from the 3D branch.

## No ink accumulates

End-to-end check on the real render: count the non-transparent pixels after each of
several pans. A ghost means the previous frame is still there, so the count climbs.

| pan | non-transparent pixels |
|---|---|
| (0,0) | 301,220 |
| (220,-140) | 300,236 |
| (-260,160) | 300,367 |
| (180,200) | 300,507 |
| (0,0) | 301,418 |

Flat within 0.4% across the whole sequence, and returning to the origin reproduces
the starting count to within 0.07%. Nothing is left behind.

Screenshot of the encounter canvas after a round trip, a pan and a re-centre:
`docs/scrum-483/encounter-2d-after-round-trip.png` (the canvas' own pixels, so what
is in the file is exactly what was rasterised). One render, grid drawn to all four
edges, no smear.

## How this was measured, and one honest limitation

The Chrome automation tab runs hidden, and a hidden tab never fires
`requestAnimationFrame`. Two attempts to verify by waiting on real frames hung and
were abandoned. The round-trip figures above were therefore produced by capturing the
callback that `setView`'s 2D branch passes to `requestAnimationFrame` and invoking it
directly — the real scheduled callback, delivered by hand rather than by the
compositor. The pixel and ink measurements above needed no frames at all and are
unaffected.

Worth stating plainly because it is a real property of the fix: while a tab is
hidden the resize is deferred until it is shown, since rAF is the scheduling
mechanism. That is correct — nothing is being presented to anyone in the meantime,
and the frame fires on the first paint after the tab becomes visible — and it is the
same behaviour the existing portlet expand/restore handlers already have. The
defence-in-depth full-store clear means that even that first pre-resize frame cannot
smear.

A side effect worth recording: two abandoned scripts from those timed-out attempts
kept running in the page and resumed when a screenshot briefly brought the tab to the
foreground, flipping the view mid-measurement. The page was reloaded to clear them
before the final measurements; nothing in the application caused that.

## Test

`services/ui/tests/test_encounter_view_resize.py`, 15 tests, in the style of
`test_globe_no_drift.py`. It pins both halves of the fix — the scheduled resize and
the full-store clear — plus the scope boundary (resize()'s zero-size guard and its
backing-store rebuild are unchanged, the 3D branch gains no resize call).

Mutation-checked; every regression is caught:

| mutation | result |
|---|---|
| rAF removed from the 2D branch | 3 failed |
| `resizeEncounter()` called directly instead of in rAF | 3 failed |
| rAF moved to the 3D branch | 3 failed |
| `drawEncounter` reverted to `clearRect(0,0,W,H)` | 4 failed |
| `drawEncounterEmpty` reverted to the old clear | 4 failed |
| clear moved after the first drawing call | 1 failed |
| synchronous draw removed (empty flash) | 1 failed |
| resize()'s zero-size guard deleted | 1 failed |

## Suites

    services/ui/tests   145 passed
