# SCRUM-452 live verification — real post-burn covariance, grown along the trajectory

Run 2026-09-24 against the real LeoLabs API, planner rebuilt from this branch.
Asset SWARM C / L3969, real conjunction cdm_id 79669937218.

## Execution-error parameters used

    execution_error_magnitude_fraction   0.02        (2% of |dv|, 1-sigma)
    execution_error_pointing_sigma_rad   0.0174533   (1 degree, 1-sigma)

Both are the documented defaults added to OperatorPolicy by this ticket. **They
are a starting point for John to confirm, not a measured property of this
spacecraft**, and they are settable per request.

## The covariance now fans out

Measured on the ephemeris the live path builds, from SWARM C's real state and its
real position covariance:

| t | position 1-sigma |
|---|---|
| epoch | 9.1 m |
| +1 h | 69.9 m |
| +6 h | 367.3 m |
| +24 h | 1,443.6 m |
| +72 h | **4,332.4 m** |

**475x growth from epoch to 72 h**, across 865 states, with the first state
exactly equal to the seed (Phi(0) is the identity). Before this ticket the same
file carried one covariance repeated 865 times.

## The live screen is still sane and catalog-consistent

    screening 603119, evaluate 97 s
    secondary_check_performed     True
    secondary_conjunction_clear   True
    closest_approach_km           46.67   (STARLINK-32185)
    verification.secondary_clear  True

Consistent with the SCRUM-442 runs on the same asset and window (48.55 km /
47.48 km closest), so the larger covariance did not destabilise the result.

## The caveats are gone

The operator note now reads, in full:

    On-demand secondary screen clear: 1 conjunction(s) returned within the 50 km
    screening volume, none breaching the clear contract (Pc >= 1e-06, miss < 1 km,
    or Mahalanobis <= 4). Screening id 603119.

No "constant position-only floor", no "tracked fast-follow" -- verified live and
asserted in the offline suite against the module sources, the operator note and
the persisted record.

## A measured finding that narrows the plan's framing

The plan says the velocity seed is "the load-bearing part" and that "over 72 h
even a small execution error dominates the position uncertainty". Measured, that
is true for real avoidance burns but not universally.

Both mechanisms are the same physics: each perturbs the semi-major axis, which
drifts the object along-track secularly. Which dominates is simply which produces
the larger delta-a. Against a 10 m position seed, at 72 h:

| burn | 72 h sigma | vs no burn (8,692 m) |
|---|---|---|
| 0.1 m/s | 8,830 m | +2% |
| 1 m/s | 17,791 m | 2.0x |
| 5 m/s | 78,098 m | 9.0x |

So the execution-error parameters matter most for real avoidance burns, and the
quality of the position seed matters most for small ones. Both are worth having.
The live decision above commanded only **0.01 m/s**, so its growth is dominated by
the position seed -- which is why the 475x above is mostly seed propagation, not
execution error.

This is asserted in the suite (`test_a_realistic_maneuver_makes_the_execution_error_dominant`)
rather than left as prose, so the crossover is pinned.

## One thing this ticket could not fully fix, and why

The plan's step 5 asks the server seam to source the primary's P_post from
`gnc_report.post_burn_covariance_km2`. Traced: that function takes a **GNC report
dict**, and `/v1/evaluate` is not the GNC endpoint -- the report arrives at
`POST /v1/gnc/report`, and the evaluate body has no `post_burn_state` block.

So the seam now does both:

- **When the call carries a GNC report** (`gnc_report` or `gnc`, the field names
  the real endpoint already uses), `post_burn_covariance_km2` is called and the
  seed is the primary's own P_post. Recorded as
  `p_post_source: gnc_post_burn_covariance`.
- **Otherwise** it falls back to `p_rel_km2`, the combined relative covariance,
  which as a primary-only seed is an over-estimate -- conservative for a screen,
  but not the right quantity. Recorded as
  `p_post_source: combined_relative_stand_in`.

Failing closed instead would have disabled the screen on every evaluate that does
not carry a GNC report, which is every evaluate today. The fallback keeps the
path working and the record says which quantity was used, so an audit can tell
them apart rather than inferring it. **The remaining work is upstream**: have the
maneuver-planning path carry a GNC report, or supply P_post directly.

The second of SCRUM-442's two approximations is therefore fixed *when a GNC
report is present* and labelled when it is not. The first -- constant in time --
is fixed unconditionally, and it was the larger of the two.

## Quota

One `create_screening` (603119). Read-only GETs for everything else.

## Suite

    python3 -m pytest services/planner    1655 passed, 2 skipped   (+30)
