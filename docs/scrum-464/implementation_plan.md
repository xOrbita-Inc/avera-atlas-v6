# SCRUM-464 implementation plan: covariance as a real-shaped wireframe dome around the asset

Ticket: https://xorbita.atlassian.net/browse/SCRUM-464
Branch off main. UI only. Relates to SCRUM-447, SCRUM-448. Supersedes SCRUM-463.

## What this is, and why

The Live Conjunction Globe concept draws covariance as ONE cyan wireframe ellipsoid,
a dome, around the operator asset, at a readable size. The shipped globe (SCRUM-448)
instead draws each object's real 3-sigma CDM covariance as a filled translucent
ellipsoid on every object, at true scale. True scale is invisible, and the abandoned
SCRUM-463 slider made it worse: exaggerating along-track-dominated ellipsoids turns
the worst one into a long streak across the globe. That is what looked wrong. This
ticket brings the rendering back to the concept: one real-shaped, readable-size
wireframe dome around the asset, and nothing else.

## The concept, for reference

From the mockup source, the covariance mesh is:

    var cov = new THREE.Mesh(
      new THREE.SphereGeometry(1,20,20),
      new THREE.MeshBasicMaterial({color:0x4ff8e8, transparent:true, opacity:0.14, wireframe:true})
    );
    // each frame, parented in effect to the asset:
    cov.position.copy(asset.world);
    var cs = 0.11 + 0.02*Math.sin(t*2);
    cov.scale.set(cs*1.5, cs, cs*1.2);   // fixed readable size, gentle breathing

So: wireframe, cyan, opacity 0.14, asset-centred, roughly 0.11 to 0.16 scene units,
anisotropic, breathing. The concept's shape is symbolic (fixed 1.5/1.0/1.2). We
improve on it in one way: drive the shape from the real covariance while keeping the
readable fixed size.

## The change, all in services/ui/app/templates/index.html

1. Remove the per-object ellipsoids. Delete the per-object use of buildCovEllipsoid
   and the globeCovParts list of per-object meshes (the SCRUM-448 loop that built a
   filled shell for the asset and every object). There is one covariance mesh now.

2. Build the asset dome from the real covariance. The orbits response already carries
   asset.cov = { axes, sigmas_m } (principal axes in ECI, one-sigma extents in
   metres). Orient the ellipsoid by asset.cov.axes run through eciToScene exactly as
   SCRUM-448 did (rotation from a basis of the three normalized axes). Take the axis
   RATIOS from asset.cov.sigmas_m so the dome's elongation and tilt are the true
   uncertainty.

3. Normalize the SIZE to a fixed readable magnitude. Let major = max(sigmas_m). Scale
   each axis to `READABLE_MAJOR * sigmas_m[i] / major`, where READABLE_MAJOR is a
   fixed scene size (propose 0.13, tune at review; the asset marker is ~0.03, so this
   is a few marker radii, matching the concept). Every asset's dome is then about the
   same overall size, real in shape, readable at globe zoom. This is deliberately not
   true scale. Guard major === 0 (degenerate) with the existing GLOBE_COV_MIN_SCALE
   floor so the transform stays well-formed.

4. Style and parent it like the concept. Wireframe SphereGeometry(1,20,20),
   MeshBasicMaterial cyan (var --accent / 0x4ff8e8), opacity ~0.14, depthWrite false.
   Parent it to the asset marker so it follows the asset every frame with no per-frame
   position code. A gentle breathing pulse (cs +/- ~15%) is a nice-to-have, not
   required; if added, it multiplies the whole scale so it cannot distort the shape.

5. Toggle with the existing COVARIANCE button. The button now shows/hides the single
   asset dome. Default off is fine; confirm at review whether the demo wants it on by
   default. Rebuild the dome on a fetch or asset switch from the new asset.cov, the
   same point in updateGlobeOrbits the old ellipsoids were rebuilt.

6. Caption, honest. The size is normalized, so the caption must not claim true scale.
   Propose "3-sigma CDM position uncertainty — orientation true, size not to scale"
   (final wording at review). No exaggeration factor, no slider, no true-scale claim.
   The note shows only while COVARIANCE is on and a dome exists.

## Do not

- Do not change the planner, the orbits response, the markers, the orbit rings, the
  worst-approach link, or the SCRUM-461/462 partial-view chip and caption. This changes
  only how the covariance is drawn.
- Do not draw a per-object ellipsoid anywhere. Asset only.
- Do not claim true scale in the caption, since the size is normalized. The shape and
  orientation are real; the size is not.
- Do not reintroduce the SCRUM-463 exaggeration slider.

## Tests

- A node harness / template-wiring test in the SCRUM-448/462 pattern: the normalizer
  maps sigmas to axis scales whose ratios equal the sigma ratios, whose major axis
  equals READABLE_MAJOR, and never below GLOBE_COV_MIN_SCALE; a degenerate (all-zero)
  covariance is handled by the floor rather than a divide-by-zero.
- Template-wiring: exactly one covariance mesh is built (the asset dome), no per-object
  covariance loop remains, the mesh material is wireframe, and the COVARIANCE toggle
  drives its visibility.
- The caption reads the normalized wording and never "true scale".
- Run the existing ui tests and confirm green.

## Live acceptance

Rebuild the ui. On a live asset with COVARIANCE on: exactly one cyan wireframe dome
around the asset, oriented and shaped by the real CDM covariance, at a fixed readable
size, following the asset as it moves; no filled per-object ellipsoids and no streak at
any zoom or asset; the caption states the size is not to scale; COVARIANCE off hides
it; FOCUS WORST still frames the worst conjunction. Capture a screenshot next to the
concept dome. Record in docs/scrum-464/live_verification.md. Do not deploy to the KVM.

## Commit

Branch off main. Include docs/scrum-464/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it confirms one
asset-only wireframe dome whose shape and orientation come from the real covariance and
whose size is a fixed readable normalization, that no per-object ellipsoid or streak
remains, that the caption does not claim true scale, and that COVARIANCE and FOCUS WORST
still behave, then reviews the live dome against the concept.
