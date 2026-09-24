# SCRUM-448 — implementation plan: real-CDM covariance ellipsoids on the globe

Ticket: https://xorbita.atlassian.net/browse/SCRUM-448
Builds on: SCRUM-447 (PR #108, the globe control cluster). Merge 447 first, then
branch 448 off main. The COVARIANCE control is the fifth button in the cluster 447
added.

## What this is

Add a COVARIANCE toggle to the globe that draws a real per-object position
uncertainty ellipsoid from the CDM covariance the parser already computes.
Exaggerated to be visible, labelled not-to-scale, true sigma shown in the object
panel. Default off.

## Data already in the tree

- services/planner/common/leolabs_cdm_parser.py: ParsedObject carries cov_eci_m2
  (6x6, m^2) and the property cov_eci_pos_m2 (top-left 3x3 position block, m^2), for
  both parsed.primary (asset / SAT1) and parsed.secondary (SAT2). It is rotated
  RTN->ECI at parse and guarded symmetric-PSD, so it is already a usable ECI position
  covariance. No new physics.
- The /v1/leolabs/orbits route already iterates the parsed rows and reads
  parsed.secondary state per object, so it reads the covariance off the same object
  with no extra fetch.

## Seams

1. Endpoint (services/planner/server.py, leolabs_orbits). For each drawn object emit
   its position uncertainty as an eigendecomposition, computed server-side with numpy
   (already a dependency), not the raw 3x3 — a JS symmetric-3x3 eig is avoidable error
   surface:
     cov: { axes: [[x,y,z],[x,y,z],[x,y,z]], sigmas_m: [s1,s2,s3] }
   where axes are the unit eigenvectors (ECI) and sigmas_m are sqrt(eigenvalue) in
   metres (one sigma per principal axis). Attach under objects[key].cov, and under
   asset.cov.
   - Asset covariance from the primary (SAT1) block of the WORST row (first drawn,
     highest Pc), so there is one asset ellipsoid. Same epoch caveat as the ring; the
     response already carries asset.epoch_utc.
   - An object whose covariance is missing, non-finite or non-PSD gets NO cov field,
     never a fabricated one. The renderer then draws no ellipsoid for it, the same
     honesty rule as a stateless secondary being skipped today.
   - Contract unchanged: still 200/404/422/503, feed-off still 503.
   - Put the eigendecomposition in a small tested helper, e.g.
     leolabs_runtime.cov_eigen_axes_sigmas(cov3), so the numpy path is unit-tested
     away from the route.

2. Renderer (services/ui/app/templates/index.html).
   - Add the fifth control to #globeControls: COVARIANCE, id globeCovBtn, onclick
     toggleGlobeCovariance(). Default off (no .on class).
   - State: globeShowCovariance=false, globeCovParts=[] (same pattern as
     globeNominalParts and globeLabelParts). clearGlobeTracks resets it.
   - Build one ellipsoid per object that has cov: a unit THREE.SphereGeometry scaled by
     sigmas_m * EXAGGERATION along the three principal axes, oriented by the eigenvector
     basis (a Matrix4/Quaternion from the three axes), positioned at the object's
     marker point (the same scene point the marker uses). Colour by the object's risk
     band at low opacity (~0.15-0.25), depthWrite off so overlaps still read.
   - EXAGGERATION is a single constant chosen so a few-hundred-metre sigma is legible at
     default zoom. While COVARIANCE is on, the globe carries a persistent "uncertainty
     not to scale" note (reuse #globeMsg styling or a small caption). The true sigma
     (3 sigma = max(sigmas_m) * 3, or per-axis) shows in the object panel when an object
     is focused or selected.
   - Visibility rules: COVARIANCE toggles all ellipsoids. An ellipsoid follows its
     object, so a nominal object's ellipsoid is hidden with the NOMINAL haze:
     toggleGlobeNominal must also set the nominal ellipsoids' .visible, and a refetch
     must honour both globeShowCovariance and globeShowNominal, the same way labels now
     honour their toggle on refetch.

## Tests

- services/planner: extend tests/test_leolabs_globe_orbits.py (the fixture CDM already
  carries covariance): asset.cov and each objects[*].cov carry axes (3 unit vectors)
  and sigmas_m (3 positive floats); an object whose CDM covariance is stripped or
  invalid is emitted with no cov field and still drawn (marker, no ellipsoid); the
  200/404/422/503 contract is unchanged. Unit-test cov_eigen_axes_sigmas directly: a
  known diagonal covariance returns axis-aligned unit vectors and the right sigmas, and
  a non-PSD input is rejected. Run python3 -m pytest services/planner green.
- Renderer is a manual browser check on a FOREGROUND tab (background tabs throttle
  requestAnimationFrame to zero, which misreads as no motion): COVARIANCE off by
  default; on draws oriented ellipsoids; NOMINAL off drops the nominal ellipsoids with
  the haze; the not-to-scale label shows; the object panel shows the true sigma.

## Verify / build

- The endpoint changed this time, so rebuild BOTH: docker compose up -d --build ui
  planner. Confirm on the rebuilt stack with SWARM C.

## Commit

- Branch scrum-448-globe-covariance off main (after 447 merges). New PR against main.
  Include docs/scrum-448/ in the commit. No AI attribution.
- Cowork runs the verification-first review before John merges.
