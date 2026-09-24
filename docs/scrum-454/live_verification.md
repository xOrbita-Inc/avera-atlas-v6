# SCRUM-454 live verification — seeding the screen from the primary's own covariance

Run 2026-09-24 against the real LeoLabs API, planner rebuilt from this branch.
Asset SWARM C / 39453, real CDM, demo path (UI shape: `/v1/evaluate` with a
`cdm_id` selector, no GNC report).

## The seed source, confirmed live

    "event": "screening_seed_selected", "p_post_source": "cdm_primary_own"

So on the demo path the screen now grows the asset's **own** covariance. Before
this ticket the same path logged `combined_relative_stand_in`.

## How large the over-estimate was, on this CDM

Measured from the parsed CDM the decision was made on (secondary
STARLINK-30994):

| block | position 1-sigma |
|---|---|
| primary (SWARM C, our asset) | **2,300.9 m** |
| secondary (STARLINK-30994) | 5,822.0 m |
| combined `p_rel` (what was seeded before) | 6,260.2 m |

**2.72x over-stated in sigma, 7.40x in trace.** The secondary's own uncertainty
is larger than our asset's, which is what one would expect -- LeoLabs tracks a
subscribed asset better than an arbitrary Starlink -- and it was being attributed
to our asset.

Carried through the SCRUM-452 growth to the end of the window:

| seed | emitted 1-sigma at 72 h |
|---|---|
| primary's own (now) | **38,057 m** |
| combined `p_rel` (before) | 121,233 m |

So the screen was drawing an uncertainty volume around our asset roughly **three
times larger than the asset actually has** at the far end of the horizon.
Conservative, but wrong, and it is the quantity the clear contract's Mahalanobis
limb is judged against.

## The screen is still sane and catalog-consistent

    screening 603150, evaluate ~110 s
    secondary_check_performed     True
    secondary_conjunction_clear   True
    closest_approach_km           39.87   (STARLINK-38309)
    verification.secondary_clear  True

In line with the runs on this asset and window across SCRUM-442 (48.55 / 47.48 km)
and SCRUM-452 (46.67 km). The smaller seed did not destabilise the result.

## SCRUM-452's growth still applies on top

Unchanged by this ticket, and asserted offline: the seed is still the pre-burn
covariance plus the burn execution error, grown along the trajectory by the J2
state transition matrix. What changed is only which position block seeds it.

Named honestly in the code: the CDM block is **P_pre**, the execution-error block
is what the burn adds, and the two grown together are the post-burn covariance --
the same P_pre + P_burn structure `post_burn_covariance_km2` formalises, with
P_pre coming from the CDM here rather than from a GNC assessment.

## One deviation from the plan, and why

The plan's step 2 says to change `_fetch_cdm_covariance` to also return
`parsed.primary.cov_eci_pos_m2`. That function cannot supply it, for two
independent reasons found by tracing it:

1. **It has no parsed CDM.** It fetches a *stored* record from ingest
   (`/cdm/{primary}/{secondary}`), which returns `covariance_combined_rtn` -- a
   single combined matrix. The ingest endpoint does compute `c_primary` and
   `c_secondary` separately and then returns only their sum, so the primary-only
   block is not on the wire at all. Exposing it is an **ingest** change, and this
   ticket is planner-only.
2. **It is not on the demo path.** An existing test asserts
   `_fetch_cdm_covariance` *must not run* for a LeoLabs source -- the LeoLabs
   branch rebuilds the request from its own parsed CDM and deliberately does not
   call the ingest adapter.

So the primary-only covariance is taken from `parsed_ll.primary.cov_eci_pos_m2`
where the parsed CDM is actually in scope, which is the same quantity the plan
names and the same rationale it gives. `_fetch_cdm_covariance` is documented with
why it cannot contribute one rather than left looking like an oversight.

**Consequence:** the stored/reference-CDM path (non-LeoLabs) still falls back to
the labelled `combined_relative_stand_in`. Fixing that needs the ingest endpoint
to return the primary block.

## Seed priority as shipped

    1. gnc_post_burn_covariance     GNC report on the request (unchanged)
    2. cdm_primary_own              the parsed CDM's primary block (new)
    3. combined_relative_stand_in   p_rel, demoted (kept as fallback)
    4. surrogate                    no CDM at all (kept)

The order is asserted against the source itself, so a reordering that let the
over-estimate win again fails the suite.

## Not changed

The scoring path still uses the combined `p_rel` for Pc, which is correct -- Pc
needs primary + secondary. Asserted offline against
`parsed.to_conjunction_state()`.

## Quota

One `create_screening` (603150). Read-only GETs for everything else.

## Suite

    python3 -m pytest services/planner    1669 passed, 2 skipped   (+14)
