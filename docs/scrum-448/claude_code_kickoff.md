# SCRUM-448 — kickoff for Claude Code

Add real-CDM covariance ellipsoids to the live conjunction globe.
Ticket: https://xorbita.atlassian.net/browse/SCRUM-448

Prereq: SCRUM-447 (PR #108) merges first. Branch this off main; the COVARIANCE
control is the fifth button in the globe control cluster 447 added.

Read docs/scrum-448/implementation_plan.md first — it names the parser fields, the
endpoint contract, the renderer seams and the tests. The covariance data already
exists in the parser; this is plumbing it to the globe plus one ellipsoid renderer.

Run in order:

1. Read docs/scrum-448/implementation_plan.md.
2. Confirm the data first. In the planner container, parse the fixture CDM
   (services/planner/tests/fixtures/leolabs_cdm_sample.json) and confirm
   parsed.primary.cov_eci_pos_m2 and parsed.secondary.cov_eci_pos_m2 are 3x3,
   symmetric-PSD, in m^2. Build against the real shape, not an assumption.
3. Add cov_eigen_axes_sigmas(cov3) to leolabs_runtime: numpy eigendecomposition,
   returns unit eigenvector axes and sqrt(eigenvalue) sigmas in metres. Unit-test it.
4. Emit per-object cov {axes, sigmas_m} from /v1/leolabs/orbits for each drawn object,
   and asset.cov from the worst row's SAT1 block. Missing or invalid covariance means
   no cov field and no fabricated ellipsoid. Keep the 200/404/422/503 contract and the
   503 feed-off exactly as they are.
5. Add the COVARIANCE button to #globeControls (fifth control, default off) and the
   renderer: ellipsoids built from axes + sigmas, exaggerated to a visible minimum,
   oriented in ECI, coloured by risk at low opacity, positioned at each object's
   marker. A persistent "uncertainty not to scale" note on the globe while COVARIANCE
   is on, and the true sigma in the object panel.
6. Wire visibility: COVARIANCE toggles all ellipsoids; nominal objects' ellipsoids
   follow the NOMINAL toggle; a refetch honours both toggles (same pattern the labels
   now use).
7. Tests: extend services/planner/tests/test_leolabs_globe_orbits.py per the plan and
   unit-test the eigen helper. Run python3 -m pytest services/planner and confirm green.
8. Rebuild BOTH containers, since the endpoint changed: docker compose up -d --build ui
   planner. Do the manual browser check per the plan on the rebuilt stack with SWARM C,
   on a foreground tab.
9. Commit on scrum-448-globe-covariance, include docs/scrum-448/ in the commit, push,
   and open a PR against main. No AI attribution. Report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: confirm the
covariance is real (from cov_eci_pos_m2, not synthetic), the no-covariance object is
drawn without a fake ellipsoid, the contract is unchanged, the planner suite is green,
and the ellipsoids render and toggle correctly on the rebuilt stack.
