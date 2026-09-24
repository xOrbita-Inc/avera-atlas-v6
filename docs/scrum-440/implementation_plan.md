# SCRUM-440 implementation plan: post-burn trajectory ephemeris for LeoLabs on-demand screening

Ticket: https://xorbita.atlassian.net/browse/SCRUM-440
Branch off main. Backend only, planner. Part of the SCRUM-433 on-demand secondary
screen. This is the ephemeris builder; submission and polling are SCRUM-441.

## What this is

Given the planner's post-burn state and its covariance, emit a LeoLabs JSON
ephemeris of the post-burn trajectory that SCRUM-441 will submit for on-demand
screening. 440 builds and unit-tests the file. The live "LeoLabs accepts it" check
happens on the first real submit in 441, per the SCRUM-439 spike.

## What the SCRUM-439 spike already settled (do not re-derive)

- Ephemeris format is confirmed: LeoLabs JSON with `frame` and `covarianceFrame`
  both EME2000, and a `states` array of { timestamp, position in metres, velocity
  in metres/second, 6x6 position-velocity covariance }. LeoLabs' own state vectors
  expose the same EME2000 6x6, so what we emit matches what LeoLabs returns.
- The screening lifecycle already exists on the client (create_screening,
  get_screening_status, wait_for_screening, get_screening_cdms). None of that is
  440. 440 stops at the JSON.
- The only doc gap is the screening request-body JSON field names and the status
  enum, and those are 441 concerns, confirmed on the first live submit. The
  ephemeris format itself is not part of that gap.

## Inputs (already produced in the planner)

The post-burn assessment carries everything the builder needs:
- epoch_utc, r_sat_km, v_sat_km_s on the post-burn state (assess_post_burn's
  post_maneuver_od block; the same fields gnc_report reads).
- The post-burn ECI position covariance from gnc_report.post_burn_covariance_km2,
  a 3x3 in km^2, in EME2000. SCRUM-428 just made its case-2 fallback
  frame-consistent, so this is genuinely EME2000 whichever case produced it.

## The one design decision: 3x3 position P_post to a 6x6

post_burn_covariance_km2 returns a 3x3 ECI position covariance. The ephemeris
format wants a 6x6 position-velocity covariance. We do not have a post-burn
velocity covariance in this codebase (p_post_km2 is read as a 3x3 position block).

Recommended default, to implement and flag for confirmation at review: put the 3x3
position P_post into the top-left position block of the 6x6, converted km^2 to m^2,
and set the velocity block and the position-velocity cross terms to zero. Emit the
same covariance on each state. This is honest: we provide the position uncertainty
we actually have, we do not fabricate a velocity covariance, and LeoLabs treats an
absent uncertainty as zero anyway. Keep the 3x3-to-6x6 construction in one small,
documented function so the convention can be swapped without touching the builder.

This is the single physics call in 440. Two better options exist and should be
named in the code comment as future work, not built now: provide a real 6x6 if the
GNC OD ever supplies one, or propagate P_post forward along the trajectory with the
state transition matrix (aps_math.cw_phi_full) so the emitted covariance grows
per state. Cowork will raise this decision explicitly at review so John or Sreejit
can confirm or override it before 441 relies on it.

## Propagation and horizon

- Propagate with kepler_propagate from common.orbit_propagation, the planner's own
  Keplerian propagator that secondary_horizon.py already uses. Do not use the
  globe's viz two-body or a cross-service call to the propagator service.
- Match the horizon and step to the SCRUM-381 secondary screen window that
  secondary_horizon.py defines, so the ephemeris covers exactly what the screen
  looks across. Read the window from that module rather than picking a new number.
  A coarse step is fine, since LeoLabs re-screens continuously across the states we
  submit; do not emit a needlessly dense file.

## Seam

New module services/planner/common/leolabs_ephemeris.py:
- build_screening_ephemeris(epoch_utc, r_sat_km, v_sat_km_s, p_post_eci_km2,
  horizon, step) -> dict, returning the LeoLabs JSON as a Python dict.
- A small documented helper that builds the 6x6 from the 3x3 position covariance
  (the decision above).
- Units: km to m (x1000) on position, km/s to m/s (x1000) on velocity, km^2 to m^2
  (x1e6) on covariance. These conversions are where a bug would hide, so test each.
- Match the exact JSON key names (frame, covarianceFrame, states, timestamp,
  position, velocity, covariance) and the covariance nesting to LeoLabs' own
  get_states EME2000 representation, which the 439 note confirms we already
  receive, so the emitted shape equals the received shape.

## Do not

- Do not submit, poll, or call create_screening. That is 441.
- Do not invent a velocity covariance beyond the documented zero-block default.
- Do not add a new propagator or change kepler_propagate.

## Tests (new services/planner/tests/test_leolabs_ephemeris.py)

Offline, no network:
- For a known post-burn state and P_post, the builder returns a dict with frame and
  covarianceFrame EME2000 and a states array spanning the horizon at the step.
- Units: position and velocity are the km inputs times 1000; the covariance
  top-left 3x3 equals P_post times 1e6 to tolerance; the velocity block is zero.
- The first state is the post-burn r/v at epoch_utc; timestamps are ISO8601 Z and
  monotonic across the horizon.
- A degenerate or missing P_post is handled the way the rest of this path handles
  it, rather than emitting a fabricated covariance.

Run python3 -m pytest services/planner and confirm green.

## Verify

Backend only. Rebuild the planner container to run the suite. The live "LeoLabs
accepts the file" acceptance is confirmed on the first submit in SCRUM-441; 440's
own acceptance is the offline builder tests plus Cowork reading the emitted JSON
against the confirmed format.

## Commit

Branch scrum-440-post-burn-ephemeris off main. Include docs/scrum-440/ in the
commit. No attribution trailers, author John Avera <javera@xorbita.com>. PR against
main; report the PR number and head SHA.

Cowork runs the verification-first review before John merges, checks the emitted
JSON against the confirmed LeoLabs format and the unit conversions, and raises the
6x6 covariance decision for John or Sreejit to confirm.
