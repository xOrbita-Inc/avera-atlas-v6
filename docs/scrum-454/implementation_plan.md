# SCRUM-454 implementation plan: seed the screen from the primary's own CDM covariance

Ticket: https://xorbita.atlassian.net/browse/SCRUM-454
Branch off main. Backend only, planner. Part of SCRUM-433. Relates to SCRUM-442
and SCRUM-452. Worth having before the Disrupt demo.

## What this is

SCRUM-452 made the screening covariance real and growing, but on the demo path it
still seeds the growth from the wrong quantity, the combined relative covariance
p_rel, because that is what the seam had. The primary's own covariance is already
in the same CDM. This ticket seeds from it, so the screen grows the asset's own
uncertainty rather than an over-estimate. Small, planner-only.

## Why not the GNC-report route

The demo runs UI to /v1/evaluate, and the post-burn covariance depends on the burn
evaluate itself chooses, so the UI cannot pre-compute a GNC report and thread it
in. The SCRUM-452 seam already uses post_burn_covariance_km2 when a request carries
a GNC report, and that path stays for callers that have one. It just does not
engage on the evaluate path, which is the demo path.

## The quantity is already parsed

The LeoLabs CDM carries per-object covariances and the parser separates them:
- parsed.primary.cov_eci_pos_m2 is our asset's own ECI position covariance, with
  primary resolved against our catalog id, not by assuming SAT1.
- p_rel_eci_km2() is primary plus secondary, the combined relative covariance.

_fetch_cdm_covariance in server.py currently returns only the combined p_rel_km2,
so the seam never sees the primary-only block. This is the whole gap.

## The change

1. _fetch_cdm_covariance also returns the primary's own position covariance
   alongside p_rel_km2, from parsed.primary.cov_eci_pos_m2. Keep p_rel for the
   scoring path that already uses it; add the primary-only block for the seed.

2. The SCRUM-452 seam (the _p_post / _p_post_source block in server.py) uses this
   seed priority:
   - GNC report P_post from post_burn_covariance_km2 when a GNC report is on the
     request. Unchanged. p_post_source "gnc_post_burn_covariance".
   - Else the CDM primary's own covariance as the pre-burn seed. New.
     p_post_source "cdm_primary_own".
   - Else the labelled combined p_rel stand-in. Existing, demoted to here.
     p_post_source "combined_relative_stand_in".
   - Else the surrogate, when there is no CDM at all.
   In every case the SCRUM-452 execution-error velocity growth is applied on top,
   so the emitted covariance is the pre-burn primary covariance plus the burn
   execution error, grown along the trajectory. That is the post-burn primary
   covariance, computed on the evaluate path with no cross-service plumbing and no
   server state.

3. The seed is a pre-burn covariance now, so name it honestly in the code. The
   primary's CDM covariance is the pre-burn state, and the execution-error block is
   what the burn adds; the two together are the post-burn covariance. This is the
   same P_pre plus P_burn structure post_burn_covariance_km2 formalises, with
   P_pre here coming from the CDM rather than a GNC assessment.

## Do not

- Do not change the SCRUM-452 growth math, the execution-error model, the STM, the
  screening request, the clear contract, the guard, or the ingest path.
- Do not change the scoring path's use of p_rel; it still wants the combined
  covariance for Pc.
- Do not remove the p_rel stand-in or the surrogate; they stay as lower-priority
  fallbacks.

## Tests

Offline:
- _fetch_cdm_covariance returns the primary-only block and it equals
  parsed.primary.cov_eci_pos_m2, distinct from p_rel.
- The seed priority resolves in order: GNC report wins when present, else the CDM
  primary block, else the p_rel stand-in, else the surrogate, with p_post_source
  labelled correctly at each step.
- With a real CDM and no GNC report, the seed position block is the primary's own
  covariance, not p_rel, and it is strictly smaller than p_rel for a case where
  the secondary carries real covariance.
- The SCRUM-452 growth still applies on top of the new seed.

Run python3 -m pytest services/planner and confirm green.

## Live acceptance

Rebuild the planner. Run one evaluate on the demo path for a real asset with a
LeoLabs CDM and confirm the decision records p_post_source as the CDM-primary
source, the seed is the primary's own covariance, and the screen result is still
sane and catalog-consistent. Respect 3 creates per 2 minutes. Record in
docs/scrum-454/live_verification.md.

## Commit

Branch off main. Include docs/scrum-454/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it confirms the
primary-only fetch, the seed priority and its labels against real code, that the
scoring path's p_rel use is untouched, runs the offline suite, and reviews the
live evaluate showing the CDM-primary seed.
