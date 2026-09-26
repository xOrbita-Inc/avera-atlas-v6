# SCRUM-463 implementation plan: covariance exaggeration slider on the globe, live rescale

Ticket: https://xorbita.atlassian.net/browse/SCRUM-463
Branch off main. UI only. Presentation change, demo readiness. Relates to SCRUM-448.

## What this is

SCRUM-448 draws each object's 3-sigma CDM position uncertainty as a translucent
ellipsoid on the globe, at true scale on purpose. True scale is honest but it does
not demo: a well-tracked object's 3-sigma position ellipsoid runs from a few metres
to a few km, so at globe scale most ellipsoids are the size of the marker dot or
smaller, and the COVARIANCE toggle looks like it does nothing except on the worst
outliers. The scale factor is 3 sigma over one Earth radius, so an along-track
1-sigma of 5.8 km draws about 0.15 of a marker radius, the median about twice it,
and only the worst (about 1,569 km 1-sigma) draws a large cigar of roughly 0.74
Earth radii. The measurement is correct; the picture is just too small to read.

SCRUM-448 anticipated this. It left `GLOBE_COV_EXAGGERATION` as a named constant and
a caption that already flips from "true scale" to "not to scale — xN" when the
constant is above one. The only thing missing is that changing it means editing code
and redeploying. This ticket turns that constant into an operator control: a slider
from 1x (true scale) to 50x, next to the COVARIANCE button, that rescales the drawn
ellipsoids live.

## The honesty rule, unchanged

The drawn size at 1x must still equal the real 3-sigma. The caption must always read
the exaggeration actually applied, so it can never claim a scale that is not drawn.
Exaggeration is a reading aid for a demo, never a change to the measurement, and the
default is 1x so the globe opens honest.

## The change, all in services/ui/app/templates/index.html

1. Turn the constant into state. Replace `const GLOBE_COV_EXAGGERATION=1.0` with a
   mutable `let globeCovExaggeration=1` (default 1x, true scale). Keep
   `GLOBE_COV_SIGMA_MULTIPLIER` and `GLOBE_COV_MIN_SCALE` exactly as they are.

2. Add the slider. In the globe overlay control row that holds AUTO-ROTATE, LABELS,
   NOMINAL, COVARIANCE, FOCUS WORST, add a range input from 1 to 50, step 1, default
   1, with a small readout of the current value (for example "1x" / "20x"). It is
   only meaningful while COVARIANCE is on, so show or enable it with the ellipsoids
   and dim or hide it when COVARIANCE is off. Match the existing control styling.

3. Rescale live, without a rebuild. When `buildCovEllipsoid` builds a mesh, store its
   base one-sigma extents on the mesh (for example `mesh.userData.sigmas_m`). The
   slider handler then walks `globeCovParts` and, for each mesh, recomputes
   `scale_axis = max(sigma_axis * GLOBE_COV_SIGMA_MULTIPLIER * globeCovExaggeration /
   (EARTH_RADIUS_KM*1000), GLOBE_COV_MIN_SCALE)` and reapplies it with
   `mesh.scale.set(...)`. No re-fetch, no new geometry. Factor the scale math out of
   `buildCovEllipsoid` into one helper both the builder and the slider call, so there
   is one formula, not two that can drift.

4. Caption tracks the live value. `renderGlobeCovNote` already reads the exaggeration;
   point it at `globeCovExaggeration` so it says "true scale" at 1x and "not to
   scale — xN" above it, and update it on every slider input.

5. Survives fetch and switch. A new fetch clears and rebuilds the ellipsoids;
   `buildCovEllipsoid` reads the current `globeCovExaggeration`, so the exaggeration
   the operator set carries across an asset switch or a refresh with no extra wiring.
   COVARIANCE off still hides every ellipsoid regardless of the slider, and turning it
   back on restores them at the current slider value.

## Do not

- Do not change the planner, the responses, or the fetch. This reads the same
  SCRUM-448 covariance blocks and only rescales what is drawn.
- Do not change what 1x draws. True scale at 1x must be bit-for-bit the SCRUM-448
  size; the slider only multiplies above it.
- Do not let the caption and the drawn scale disagree. One exaggeration value drives
  both.
- Do not drop the `GLOBE_COV_MIN_SCALE` floor; a degenerate axis must still give a
  well-formed transform at every slider value.

## Tests

- A node-harness or template-wiring test in the SCRUM-448/462 pattern: the shared
  scale helper returns the SCRUM-448 true-scale value at exaggeration 1 and exactly
  N times that at exaggeration N (above the floor), and never below `GLOBE_COV_MIN_SCALE`.
- The caption builder reads the live exaggeration: "true scale" at 1, "not to scale
  — xN" above it.
- Template-wiring asserts the slider exists, is driven into the exaggeration state,
  and that the handler rescales the meshes in `globeCovParts` rather than rebuilding
  (the SCRUM-461 lesson that a function nothing calls still passes a pure harness).
- COVARIANCE off hides the ellipsoids at any slider value; a rebuild on fetch honours
  the current slider value.
- Run the existing ui tests and confirm green.

## Live acceptance

Rebuild the ui. On SWARM B with COVARIANCE on: at 1x the ellipsoids match the
SCRUM-448 true-scale picture and the caption reads "true scale"; sliding up to a demo
value grows them in place, smoothly, with no redeploy and no code edit, and the
caption reads "not to scale — xN"; FOCUS WORST still frames the worst conjunction.
Turn COVARIANCE off and confirm the slider stops mattering and no ellipsoid shows.
Capture the slider at 1x and at a demo value. Record in
docs/scrum-463/live_verification.md. Do not deploy to the KVM.

## Commit

Branch off main. Include docs/scrum-463/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it confirms the
drawn size at 1x is unchanged from SCRUM-448, that one exaggeration value drives both
the scale and the caption so they cannot disagree, that the rescale is live and not a
rebuild, and that COVARIANCE off still hides everything, then reviews the live slider.
