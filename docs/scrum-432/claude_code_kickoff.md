# SCRUM-432 kickoff for Claude Code

Make the TIROS 4 E2E smoke test evaluate against the real stored covariance and
assert it. Full spec in docs/scrum-432/implementation_plan.md. Read it first.

Branch: scrum-432-tiros-e2e-real-covariance off current main (67d3a49).

Do:
1. In services/ui/app/templates/index.html runTiros4SmokeTest(): add
   primary_norad: '226' and secondary_norad: '35929' to evalReq.conjunction,
   and remove the hand-rolled p_rel_km2 and pc_precomputed: 0.0 (keep r_rel_km).
   With the pair, the covariance adapter resolves the injected TIROS CDM and
   the evaluate reports covariance_source real_cdm with a linked cdm_record_id.
2. Add an assertion step to the smoke test that evalResult.covariance_source is
   'real_cdm' (not a surrogate); renumber the following step labels. The result
   exposes covariance_source at top level.
3. Add services/planner/tests/test_tiros_e2e_real_covariance.py: a request with
   the 226/35929 pair reports covariance_source real_cdm with a non-None
   cdm_record_id (mock the ingest covariance fetch to return a real_cdm
   covariance and an id), and the no-pair control reports surrogate. Run the
   planner suite from the repo root.

Do not: change the inject/store/decision-log steps or any other evaluate path.
No AI attribution. Do not merge; open the PR for John. Since this touches the
dashboard, browser-verify: run RUN TIROS 4 E2E TEST and confirm the evaluate and
new assertion steps read real_cdm with zero console errors.

When done, report: files changed with line counts, the suite result on the
branch, and confirmation that the smoke test's evaluate now reports real_cdm
with the assertion passing and the no-pair control still surrogate.
