# SCRUM-463 live verification — covariance exaggeration slider

Run 2026-09-25 on the local stack against the real LeoLabs API, ui rebuilt from this
branch and driven in Chrome. SWARM B (39451), 45 ellipsoids. No KVM deploy.

## 1x is the SCRUM-448 picture, to the digit

Not asserted — computed. The SCRUM-448 formula was written out independently in the
page console and compared against what the meshes actually carry:

    const scrum448 = s => Math.max(s * 3 * 1.0 / (EARTH_RADIUS_KM * 1000), 1e-6);

    cov_on: true, visible_ellipsoids: 45
    at_1x_matches_scrum448: true
    caption: "3σ CDM position uncertainty — true scale"
    readout: "1×"
    biggest axis at 1x: 0.2409 Earth radii

`docs/scrum-463/cov-slider-1x.jpg` — the SCALE slider at 1× in the control cluster,
COVARIANCE on, caption reading true scale, and the ellipsoids barely larger than the
marker dots. That last part is the SCRUM-448 problem this ticket exists to give the
operator a handle on, and at 1x it is unchanged.

## The slider rescales live, and the caption follows

Driven through the real `<input type=range>` with a dispatched `input` event, not by
calling the handler:

    exaggeration: 20
    readout:  "20×"
    caption:  "3σ CDM position uncertainty — not to scale — ×20"
    biggest axis: 0.241 -> 4.82 Earth radii
    meshes rescaled: 45

One value moved both the meshes and the caption, which is the property that keeps
the note from claiming a scale that is not drawn.

## The floor, and a check of mine that was wrong

A first pass asserted `drawn_at_20x == drawn_at_1x * 20` and it came back **false**.
Worth recording, because the code is right and the check was not.

Per-axis on a real mesh (sigmas 27,829 m / 6.80 m / 1.77 m):

| axis | true 1x, unfloored | floored at 1x | drawn at 20x | expected |
|---|---|---|---|---|
| x | 1.3104e-2 | no | 2.62089e-1 | 2.62089e-1 |
| y | 3.2031e-6 | no | 6.40612e-5 | 6.40612e-5 |
| z | **8.3130e-7** | **yes** | 1.66259e-5 | 1.66259e-5 |

The z axis is below the `GLOBE_COV_MIN_SCALE` floor of 1e-6 at 1x, so it is drawn at
the floor there. The floor is applied **after** the exaggeration, so at 20x the axis
un-floors onto its real exaggerated size — 1.66e-5, not 20 x 1e-6 = 2e-5. "Nx is N
times 1x" therefore holds only above the floor, which is exactly what the harness
asserts and what my live one-liner ignored.

Checked properly against `max(sigma * 3 * N / (R*1000), floor)`:

    every_mesh_correct: true      (all 45 meshes, all three axes)

## COVARIANCE still governs everything

    slider starts disabled and dimmed          (COVARIANCE defaults off)
    COVARIANCE on  -> slider enabled, undimmed, 45 ellipsoids visible
    COVARIANCE off at 50x -> every ellipsoid hidden, caption hidden,
                             slider disabled and dimmed, value kept at 50
    COVARIANCE on again  -> ellipsoids back at 50x, caption back at ×50

The slider keeps its value across the toggle, so turning the ellipsoids back on
restores the operator's chosen scale rather than snapping to 1x.

## FOCUS WORST still works

    focus_worst_worked: true, camera moved to (-0.001, -0.001, 3.496),
    worst: 67348, exaggeration still 20, caption still ×20

## A regression this ticket introduced, caught live and fixed

The first live run showed the caption rendering **on top of** the control buttons.
Measured rather than eyeballed:

    controls: top 1085, bottom 1145   (60px tall — wrapped to three rows)
    cov_note: top 1087, bottom 1109
    overlapping: true

Cause: SCRUM-448 parked the caption at a fixed `bottom:46px`, which cleared a
single-row cluster. Adding the slider widened the cluster enough to wrap, and a
fixed offset cannot track a height that depends on wrapping.

Fix: the caption is now a full-width row **inside** the cluster (`flex:0 0 100%`,
`order:-1`), so it is always directly above the controls it describes, at any wrap.
Re-measured after the fix:

    caption_overlaps_a_button: false
    caption_above_buttons:     true
    caption_inside_cluster:    true

Visible in `cov-slider-1x.jpg`. `cov-slider-1x-before-caption-fix.jpg` is kept as
the before shot, since the collision is only obvious side by side. Three tests now
pin the new placement so the fixed offset cannot come back.

## What I could not capture

**There is no 20x screenshot.** The Chrome capture frame clipped the viewport
inconsistently between calls — 773x691 on a 1333x1192 window on several attempts,
so the control cluster fell outside the captured region — and resizing the window,
zooming the page and scrolling the element into view did not move it. I got one
clean frame on a fresh tab, which is the 1x shot above, and stopped rather than keep
retrying.

So the 20x evidence in this document is numeric rather than visual: the readout, the
caption, the 45 rescaled meshes and the per-axis comparison against the formula, all
read out of the live page. The visual check at 20x is the one thing a reviewer
should do by hand, and it is a drag of one slider.

## Tests

    python3 -m pytest services/ui/tests services/planner    1904 passed, 2 skipped
    node services/ui/tests/cov_scale_harness.mjs            67/67 checks passed

`cov_scale_harness.mjs` extracts the real functions from `index.html` and checks the
maths against an independently written SCRUM-448 formula: 1x equals it for six real
sigma values including 5.8 km and 1,569 km, Nx is exactly N times it above the floor
for N in 2/5/20/50, the floor holds at every value, the rescale reads the stored base
sigmas so two drags are not compounded, returning to 1x returns exactly to true
scale, the value is clamped and integral (0, -5, 999, 7.4, NaN, "abc" all handled),
the caption flips at exactly 1, and COVARIANCE off hides everything at 50x while
keeping the value.

`test_cov_scale.py` runs that under pytest and adds wiring checks needing no node:
the exaggeration is mutable state defaulting to 1 and the old constant is gone, the
builder and the slider both go through `covAxisScale` and neither computes its own
scale, the handler rebuilds nothing (no `buildCovEllipsoid`, no `SphereGeometry`, no
fetch), the base sigmas ride on the mesh, the slider is 1..50 step 1 wired to
`oninput`, it sits between COVARIANCE and FOCUS WORST, it starts disabled and dimmed,
the toggle syncs it, and the caption placement above.

## Not changed

No planner change, no change to the responses or the fetch. `GLOBE_COV_SIGMA_MULTIPLIER`
and `GLOBE_COV_MIN_SCALE` are untouched, and 1x draws what SCRUM-448 drew.
