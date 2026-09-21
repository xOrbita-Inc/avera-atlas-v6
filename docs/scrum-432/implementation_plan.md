# SCRUM-432 implementation plan

Make the TIROS 4 E2E smoke test evaluate against the real stored covariance
instead of a surrogate, and assert it, so the real-covariance path cannot
silently regress. Small, UI-focused, plus a backend regression test.

## The gap (confirmed in code, main 67d3a49)

RUN TIROS 4 E2E TEST runs runTiros4SmokeTest() in
services/ui/app/templates/index.html. Step 1 injects the real TIROS 4 (226) vs
IRIDIUM 33 DEB (35929) CDM into the store (source real_cdm). Step 2 then builds
its own /v1/evaluate request whose conjunction block carries obj_id
'IRIDIUM 33 DEB' (a name, not a NORAD), a hand-rolled p_rel_km2, and
pc_precomputed 0.0, and no primary_norad / secondary_norad.

In server.py the covariance adapter runs its else branch for this request
(not leolabs_used, not UDL, not a leolabs source), calling
_fetch_cdm_covariance(primary_norad="", secondary_norad="IRIDIUM 33 DEB", ...).
That pair cannot resolve the stored CDM (GET /cdm//IRIDIUM 33 DEB), so it falls
back to surrogate and overwrites the supplied p_rel_km2. The evaluate therefore
reports covariance_source=surrogate and never exercises the real stored
covariance, even though inject and store are real. All 13 steps pass because
none of them checks the covariance source.

## The fix

The covariance adapter resolves the stored CDM by the NORAD pair, so the smoke
test only needs to send it.

### 1. UI: send the pair, drop the hand-rolled covariance, assert the source

services/ui/app/templates/index.html, runTiros4SmokeTest():

- In evalReq.conjunction, add primary_norad: '226' and secondary_norad:
  '35929' (the injected pair). Remove p_rel_km2 and pc_precomputed: 0.0 so the
  request does not carry a covariance or a pinned Pc; the store-resolved real
  covariance should drive both. Keep r_rel_km, it is used for the 2D geometry.
- With the pair present, the adapter's else branch calls
  _fetch_cdm_covariance('226', '35929', ...), which resolves the injected row
  (source real_cdm -> covariance_source real_cdm via the ingest _SOURCE_MAP) and
  returns a cdm_record_id, so the evaluate scores the real covariance and the
  audit write links to the stored CDM.
- Add an assertion step (a new Step between the current evaluate and artifact
  steps, renumber the labels): the evaluate result's covariance_source is
  'real_cdm', not a surrogate. The response exposes it at top level
  (result.covariance_source, which the dashboard already reads in
  renderCovSourceBadge and covLabelLine). Fail the step if it is anything other
  than real_cdm, so a regression that drops back to surrogate is visible in the
  smoke output rather than silent.

### 2. Backend: a durable regression test

The smoke test is browser-only and not in the pytest suite, so add a small
planner test that pins the wiring:

- services/planner/tests/test_tiros_e2e_real_covariance.py.
- A TIROS-shaped /v1/evaluate request carrying the 226 / 35929 pair drives the
  covariance adapter's store-fetch branch and reports covariance_source
  real_cdm with a non-None cdm_record_id (mock _fetch_cdm_covariance or the
  ingest GET to return a real_cdm covariance and an id for that pair, so the
  test asserts the request routes to the real path, not that the store is up).
- The control: the same request without the pair (obj_id name only, as today)
  reports a surrogate covariance_source. This is what makes the first assertion
  non-vacuous and documents the exact bug this ticket fixes.

## Keep intact

The inject, store, and decision-log steps of the smoke test, and every other
evaluate path. This only changes what identifiers the TIROS smoke test sends
and adds assertions.

## Acceptance

- Running RUN TIROS 4 E2E TEST reports covariance_source real_cdm on the
  evaluate step, with the new assertion step passing, and the Pc is computed
  from the real covariance rather than pinned to the old pc_precomputed 0.0.
- The backend test asserts the pair-carrying request reports real_cdm and the
  no-pair control reports surrogate.
- Planner suite green.

## Verification hints

- Browser: run the smoke test, confirm the evaluate step and the new assertion
  step both read real_cdm, and the covariance badge shows real covariance, not
  surrogate. Re-run it and confirm the store does not gain a duplicate TIROS row
  (the 429 upsert dedups the inject on the encounter key).
- Prove the control in the backend test: no pair -> surrogate, pair -> real_cdm.

## Files

- services/ui/app/templates/index.html (runTiros4SmokeTest: add the pair, drop
  p_rel_km2 and pc_precomputed, add the covariance-source assertion step)
- services/planner/tests/test_tiros_e2e_real_covariance.py (new)
