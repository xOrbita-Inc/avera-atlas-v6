# SCRUM-464 kickoff for Claude Code

Bring the globe's covariance back to the Live Conjunction Globe concept: one cyan
wireframe uncertainty dome around the asset, real in shape, readable in size. Remove
the SCRUM-448 filled per-object ellipsoids. This replaces the abandoned SCRUM-463
slider (PR #127, being closed). Ticket:
https://xorbita.atlassian.net/browse/SCRUM-464

Branch off main. UI only. No planner changes.

Read docs/scrum-464/implementation_plan.md first. The point: the shape and orientation
are the real CDM covariance, the size is a fixed readable normalization, and there is
exactly one dome, on the asset.

Run in order:

1. Read docs/scrum-464/implementation_plan.md.

2. Find the SCRUM-448 covariance code in services/ui/app/templates/index.html:
   buildCovEllipsoid, globeCovParts, the per-object build loop and the asset-cov build
   in updateGlobeOrbits, GLOBE_COV_* constants, toggleGlobeCovariance, renderGlobeCovNote.

3. Remove the per-object ellipsoids: the per-object buildCovEllipsoid calls and the
   globeCovParts list of many meshes. One covariance mesh remains, the asset dome.

4. Build the asset dome from asset.cov = { axes, sigmas_m } on the orbits response.
   Orient by asset.cov.axes through eciToScene (the SCRUM-448 basis-from-axes rotation).
   Size: major = max(sigmas_m); each axis scale = READABLE_MAJOR * sigmas_m[i] / major,
   with READABLE_MAJOR a named constant (propose 0.13), floored by GLOBE_COV_MIN_SCALE,
   guarding major === 0. Shape and tilt real, overall size fixed and readable. Not true
   scale.

5. Style like the concept: wireframe SphereGeometry(1,20,20), MeshBasicMaterial cyan
   (0x4ff8e8) transparent opacity ~0.14, depthWrite false, parented to the asset marker
   so it follows the asset. Optional gentle breathing that multiplies the whole scale
   (never per-axis, so it cannot distort the shape).

6. Toggle with the existing COVARIANCE button (show/hide the one dome). Rebuild the dome
   from the new asset.cov on fetch and asset switch, where the old ellipsoids rebuilt.
   Default off is fine; flag on-by-default as a review question.

7. Caption honest: no true-scale claim. Propose "3-sigma CDM position uncertainty —
   orientation true, size not to scale". Shows only while COVARIANCE is on and a dome
   exists. No slider, no exaggeration factor.

8. Do not touch the planner, the orbits response, the markers, the rings, the
   worst-approach link, or the SCRUM-461/462 partial-view chip and caption. Do not draw
   any per-object ellipsoid. Do not reintroduce the SCRUM-463 slider.

9. Tests: a harness that the size normalizer preserves the sigma ratios, pins the major
   axis to READABLE_MAJOR, floors at GLOBE_COV_MIN_SCALE, and handles an all-zero
   covariance; template-wiring that exactly one covariance mesh is built, no per-object
   loop remains, the material is wireframe, and COVARIANCE drives its visibility; the
   caption never says "true scale". Run the existing ui tests and confirm green.

10. Rebuild the ui. On a live asset, COVARIANCE on shows one cyan wireframe dome around
    the asset, real in shape, readable in size, following the asset; no per-object
    ellipsoids, no streak; caption says size not to scale; off hides it; FOCUS WORST
    still frames the worst. Screenshot next to the concept dome. Record in
    docs/scrum-464/live_verification.md. Do not deploy to the KVM.

11. Commit off main, include docs/scrum-464/, push, open a PR against main. No
    attribution trailers, author John Avera <javera@xorbita.com>. Report the PR number
    and head SHA.

Then Cowork runs the verification-first review before John merges.
