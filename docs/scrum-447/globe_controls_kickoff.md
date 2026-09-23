# SCRUM-447 follow-up — globe control cluster to match the mockup

Same branch: scrum-447-globe-live-data. This is pre-merge polish on PR #108, not a
new branch and not a new ticket. Goal: the live globe should match its own
reference mockup ("Live Conjunction Globe", artifact v3) in its controls.

## What to change

Today the globe's only control is the NOMINAL toggle, sitting in the header next to
2D | 3D GLOBE (index.html:519). The mockup instead has a floating control cluster
in the bottom-right corner over the globe. Build that cluster and move NOMINAL into
it.

Cluster contents, matching the mockup's top row:
- AUTO-ROTATE
- LABELS
- NOMINAL  (moved from the header, behaviour unchanged)
- FOCUS WORST

Do NOT build COVARIANCE here. The mockup shows a COVARIANCE button, but a real
globe covariance ellipsoid is a separate feature: the endpoint has to emit the
per-object ECI covariance the parser already computes (cov_eci_m2), and the
renderer has to draw an ellipsoid. That is its own follow-on ticket. Leave it out,
and do not add a dead COVARIANCE button.

## Placement and styling

- The cluster is an absolutely-positioned child of #globeWrap, which is already
  position:relative and holds #globeCanvas and the bottom-left #globeMsg
  (index.html:270). Put the cluster bottom-right, mirroring how #globeMsg sits
  bottom-left.
- Reuse the existing .globe-toggle button class (index.html:278); .globe-toggle.on
  is the active/cyan state. AUTO-ROTATE, LABELS and NOMINAL are on-state toggles;
  FOCUS WORST is a momentary action button.
- The cluster is visible only in the 3D view. setView('3d') already adds .visible
  to the nominal button (index.html:3300) and removes it for 2D (3311); tie the
  whole cluster to that same show/hide so it appears only on the globe.

## Wiring (every hook already exists except the label array)

- AUTO-ROTATE: set globeControls.autoRotate on/off. globeControls is the
  OrbitControls at index.html:3332, and the render loop already calls
  globeControls.update() every frame (3747), so auto-rotate animates with no other
  change. Use a slow autoRotateSpeed (~0.5) so it reads as a gentle drift.
- LABELS: labels are Sprites created at index.html:3435 and added to the scene, but
  there is no toggle yet. Collect the label sprites into an array as they are
  created, the same pattern globeNominalParts uses (index.html:3537), and have
  LABELS flip their .visible. Default on.
- NOMINAL: keep toggleGlobeNominal() and globeShowNominal exactly as they are
  (index.html:3542). Only the button's location moves.
- FOCUS WORST: globeFocusOnConj(conj) already exists (index.html:3829) and the
  payload's worst_object is already stored on the globe state (index.html:3511).
  Focus the worst object the same way the 2D path focuses a selected conjunction
  (index.html:1528). If focusing needs a lookup from worst_object to its stored
  track/marker, add that small map rather than re-deriving it.

## Scope guard

Front-end only. No change to /v1/leolabs/orbits, leolabs_runtime, or any planner
code, so no new planner tests and no planner rebuild. Do not touch the risk bands,
the 503 contract, or the live-vs-scenario switch.

## Verify

- Rebuild only the ui container (planner is unchanged this time):
    docker compose up -d --build ui
- Hard-refresh, switch to 3D GLOBE with SWARM C selected and conjunctions fetched.
  Confirm: the bottom-right cluster shows AUTO-ROTATE, LABELS, NOMINAL, FOCUS WORST;
  the header no longer carries NOMINAL; AUTO-ROTATE drifts the view; LABELS hides
  and shows the labels; NOMINAL still drops the haze; FOCUS WORST frames COSMOS 1408
  DEB (the worst for SWARM C). 2D stays the default and the cluster hides in 2D.

## Commit

Commit onto scrum-447-globe-live-data and push, which updates PR #108 in place. No
new branch, no new PR. No AI attribution in the message.

Then Cowork re-reviews the delta before John merges.
