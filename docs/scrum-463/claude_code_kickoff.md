# SCRUM-463 kickoff for Claude Code

Turn the globe's covariance exaggeration from a build-time constant into an operator
slider, 1x (true scale) to 50x, that rescales the drawn 3-sigma ellipsoids live. The
COVARIANCE button still shows and hides them; this only adds a scale control next to
it. SCRUM-448 already left the constant and a caption that reads "not to scale — xN"
above 1x, so this is small. Ticket:
https://xorbita.atlassian.net/browse/SCRUM-463

Branch off main. UI only. No planner changes.

Read docs/scrum-463/implementation_plan.md first. The honesty rule is the point: 1x
must draw the real 3-sigma, and the caption must always read the exaggeration
actually applied.

Run in order:

1. Read docs/scrum-463/implementation_plan.md.

2. Find the SCRUM-448 code in services/ui/app/templates/index.html: the constants
   GLOBE_COV_SIGMA_MULTIPLIER, GLOBE_COV_EXAGGERATION, GLOBE_COV_MIN_SCALE; the state
   globeShowCovariance / globeCovParts; toggleGlobeCovariance(); renderGlobeCovNote();
   buildCovEllipsoid(); and the globe overlay control row that holds AUTO-ROTATE,
   LABELS, NOMINAL, COVARIANCE, FOCUS WORST.

3. State, not constant. Replace `const GLOBE_COV_EXAGGERATION=1.0` with a mutable
   `let globeCovExaggeration=1`. Keep GLOBE_COV_SIGMA_MULTIPLIER and
   GLOBE_COV_MIN_SCALE unchanged.

4. One scale formula. Factor the axis-scale math out of buildCovEllipsoid into a
   helper, for example covAxisScale(sigma_m) returning
   `max(sigma_m * GLOBE_COV_SIGMA_MULTIPLIER * globeCovExaggeration /
   (EARTH_RADIUS_KM*1000), GLOBE_COV_MIN_SCALE)`, and have both the builder and the
   slider handler call it, so there is one formula not two.

5. Store the base sigmas. In buildCovEllipsoid, keep the mesh's base one-sigma extents
   on the mesh (mesh.userData.sigmas_m) so the slider can rescale without rebuilding.

6. Add the slider. In the control row, a range input 1 to 50, step 1, default 1, with
   a small readout ("1x" / "20x"). On input, set globeCovExaggeration, walk
   globeCovParts and reapply covAxisScale to each mesh's three axes with
   mesh.scale.set(...), then call renderGlobeCovNote(). No re-fetch, no new geometry.
   The slider is only meaningful while COVARIANCE is on: show or enable it with the
   ellipsoids and dim or hide it when off, wired through toggleGlobeCovariance.

7. Caption tracks it. Point renderGlobeCovNote at globeCovExaggeration so it reads
   "true scale" at 1x and "not to scale — xN" above it, and update it on every slider
   input. One value drives both the drawn scale and the caption.

8. Survives fetch and switch. A new fetch already clears and rebuilds the ellipsoids;
   buildCovEllipsoid reads the current globeCovExaggeration through covAxisScale, so
   the operator's setting carries across an asset switch or a refresh. COVARIANCE off
   still hides everything regardless of the slider.

9. Do not change the planner, the responses or the fetch, do not change what 1x draws
   (it must equal the SCRUM-448 true-scale size), do not let the caption and the drawn
   scale disagree, and do not drop the GLOBE_COV_MIN_SCALE floor.

10. Tests: the shared scale helper returns the SCRUM-448 true-scale value at
    exaggeration 1 and N times it at exaggeration N, never below the floor; the caption
    reads "true scale" at 1 and "not to scale — xN" above; template-wiring asserts the
    slider drives the exaggeration state and the handler rescales globeCovParts rather
    than rebuilding; COVARIANCE off hides the ellipsoids at any slider value and a
    rebuild on fetch honours the current value. Run the existing ui tests and confirm
    green.

11. Rebuild the ui. On SWARM B with COVARIANCE on: at 1x the picture matches SCRUM-448
    and the caption reads "true scale"; sliding up grows the ellipsoids in place with
    no redeploy and no code edit, caption "not to scale — xN"; FOCUS WORST still frames
    the worst conjunction; COVARIANCE off stops the slider mattering. Capture the slider
    at 1x and at a demo value. Record in docs/scrum-463/live_verification.md. Do not
    deploy to the KVM.

12. Commit off main, include docs/scrum-463/, push, open a PR against main. No
    attribution trailers, author John Avera <javera@xorbita.com>. Report the PR number
    and head SHA.

Then Cowork runs the verification-first review before John merges: it confirms the 1x
picture is unchanged from SCRUM-448, that one exaggeration value drives both the scale
and the caption, that the rescale is live and not a rebuild, and that COVARIANCE off
still hides everything, then reviews the live slider.
