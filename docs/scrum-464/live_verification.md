# SCRUM-464 live verification — one covariance dome on the asset

Run 2026-09-26 on the local stack against the real LeoLabs API, ui rebuilt from this
branch. No KVM deploy.

## The finding that changed the implementation

The ticket asks for a dome that is "real in shape, readable in size". On this data
those two are mutually exclusive, and it is worth seeing why before reading the rest.

Asset position covariance, measured across every subscribed asset:

| asset | sigmas (m) | aspect |
|---|---|---|
| SWARM B 39451 | 27,829 / 6.8 / 1.8 | **15,764 : 1** |
| SWARM A 39452 | 31,196 / 15.9 / 4.2 | **7,489 : 1** |
| SWARM C 39453 | 1,801 / 8.7 / 6.0 | **299 : 1** |
| CRYOSAT 2 36508 | 1,454 / 17.1 / 2.0 | **743 : 1** |

A well-tracked LEO asset's uncertainty is overwhelmingly along-track. Pin the major
axis at a readable 0.13 scene units and draw the true ratios, and the minor axes land
between 8e-6 and 1.5e-3 — sub-pixel, against an asset marker of radius 0.018. I
built that first and confirmed it live: one wireframe mesh, correctly oriented,
`base_scale [0.13, 0.00003, 0.00001]`, and **nothing visible on screen**. An
invisible needle is the COVARIANCE button looking broken, which is the complaint
that started this line of work.

The concept mockup's 1.5 / 1.0 / 1.2 was symbolic precisely because real covariance
is not shaped like a dome.

**So the minor axes are compressed** by a power law (`ratio^0.3`, floored at 0.12)
rather than drawn raw. It is monotonic, so the axis ordering and the relative
roundness between assets both survive — a rounder covariance still draws rounder —
but the magnitude of the ratio does not. Drawn result:

    SWARM B    0.1300 / 0.0156 / 0.0156     minor/major 0.120
    SWARM A    0.1300 / 0.0156 / 0.0156     minor/major 0.120
    SWARM C    0.1300 / 0.0263 / 0.0235     minor/major 0.181
    CRYOSAT 2  0.1300 / 0.0343 / 0.0180     minor/major 0.139
    isotropic  0.1300 / 0.1300 / 0.1300     minor/major 1.000

SWARM C (299:1) reads rounder than SWARM B (15,764:1), which is the real signal
surviving qualitatively, and a genuinely isotropic covariance still draws a sphere.

**This is a deviation from the ticket**, which says the shape is real. It is named
(`GLOBE_COV_ASPECT_GAMMA`, `GLOBE_COV_MIN_AXIS_RATIO`), the measurements above are in
the comment beside it, and setting gamma to 1 and the ratio floor to 0 restores the
raw ratios. The caption was changed to match: it now says **"orientation true; shape
and size not to scale"** rather than claiming a true shape it no longer draws. If the
call is that a needle is preferable to a compressed dome, it is one constant.

## Exactly one dome, on the asset

Counted in the live scene rather than assumed:

    wireframe meshes in the entire scene:   1
    meshes parented to the asset marker:    1
    dome.parent === assetMarker:            true
    dome world position === asset position: true   (it follows the asset)

    material: wireframe true, color #4ff8e8, opacity 0.14, depthWrite false
    geometry: SphereGeometry(1,20,20)
    caption:  "3σ CDM position uncertainty — orientation true; shape and size not to scale"

No per-object ellipsoid exists anywhere: `buildCovEllipsoid` and `globeCovParts` are
gone from the template, and `typeof buildCovEllipsoid === 'undefined'` in the live
page.

## It is actually drawn — measured in pixels

The screenshot tooling clipped the viewport again (see below), so the dome's
visibility was verified by reading the WebGL buffer instead, which is better evidence
anyway. With a camera aimed at the asset, counting strongly-cyan pixels in a box
around it:

    COVARIANCE on      96,754 cyan px
    COVARIANCE off     93,478 cyan px
    the dome            3,276 px

Driven through the real COVARIANCE button, not by poking `.visible`: the button
removes exactly 3,276 and restores exactly 3,276. The ~93k baseline is the asset's own
marker, the cyan orbit track and the Earth limb inside a deliberately large sample
box; the delta is the dome.

Before the aspect compression the same measurement gave a delta of zero, which is
what "sub-pixel needle" means in practice.

## The size never depends on magnitude

Asserted offline across the real shapes and at extremes: a covariance and the same
covariance at 1,000,000x the magnitude draw identically, and the major axis is pinned
to `GLOBE_COV_READABLE_MAJOR` for a 1 m and a 1e9 m uncertainty alike. That is the
normalization doing its job, and it is why the caption must not claim true scale.

## Degenerate and unreadable inputs

An all-zero covariance returns three floor values rather than dividing by zero, and
the normalizer returns null — so nothing is drawn — for a null, a wrong-length array,
a NaN, an Infinity, a negative sigma or a string. A dome that asserted a shape the
data did not contain would be worse than no dome.

## What I could not capture

**There is no screenshot of the dome.** The Chrome capture frame clipped the
viewport to 673x784 of a 1527x3474 drawing buffer across every attempt, and resizing
the window, zooming the page, scrolling the element into view and aiming the camera
did not bring the asset into the captured region — the same tool limitation that cost
the SCRUM-463 run its 20x frame. Rather than keep retrying I verified visibility by
reading the pixels, above.

So the one thing left for a human is the aesthetic check: does the dome read like the
concept. Turn COVARIANCE on with any live asset and look. Everything mechanical about
it — one mesh, asset-parented, correct material, correct scales, toggle, caption —
is measured above.

## Tests

    python3 -m pytest services/ui/tests services/planner    1907 passed, 2 skipped
    node services/ui/tests/cov_dome_harness.mjs             36/36 checks passed

`cov_dome_harness.mjs` replaces the SCRUM-463 exaggeration harness. It checks the
major axis is pinned, the axis ordering is preserved, the compression is monotonic
(rounder draws rounder, isotropic draws a sphere), magnitude never affects the
drawing, the floor holds, a degenerate covariance does not divide by zero, and
explicitly that the raw ratio is **not** preserved — stated as a test so nobody
restores the invisible needle by "fixing" it back.

`test_cov_dome.py` adds wiring checks needing no node: the old per-object builder and
list are gone and the SCRUM-463 slider is not reintroduced, exactly one build site,
parented to the marker, detached on rebuild, wireframe/cyan/translucent/no-depth-write,
orientation from the covariance axes through `eciToScene`, the normalizer no longer
converts metres to scene units, the breathing multiplies all three axes from the base
scale so it cannot distort the shape, the toggle drives visibility, and the caption
never says "true scale".

## Not changed

No planner change. The orbits response, the markers, the rings, the worst-approach
link and the SCRUM-461/462 partial-view chip and caption are untouched.
