# SCRUM-452 implementation plan: real post-burn covariance, execution-error 6x6 grown along the trajectory

Ticket: https://xorbita.atlassian.net/browse/SCRUM-452
Branch off main. Backend only, planner. Part of SCRUM-433. Relates to SCRUM-440,
SCRUM-451, SCRUM-442. Must land before the Tech Crunch Disrupt demo.

## What this is

The covariance we submit with the screening ephemeris is a caveated stand-in with
two approximations. This ticket replaces both with an honest post-burn covariance
that grows across the window, so the "clears a maneuver against real covariance"
demo claim is true.

The two approximations, from the SCRUM-441 review and the SCRUM-442 build:

1. Constant in time. SCRUM-440 puts the same covariance on all 865 states across
   72 h. Real uncertainty fans out, most of it along-track.
2. Wrong quantity. The SCRUM-442 wiring passes p_rel_km2, the combined relative
   covariance, where the ephemeris wants the primary object's own post-burn
   position covariance. Over-stated, conservative, but not the right input.

## The model

Seed a post-burn 6 by 6 P0 in EME2000, then grow it along the trajectory.

Seed P0:
- Position block: the primary's own post-burn position covariance from
  gnc_report.post_burn_covariance_km2 (a 3x3 in km^2, EME2000, made
  frame-consistent by SCRUM-428), converted km^2 to m^2. This is the quantity
  SCRUM-440 originally intended, and it replaces the p_rel_km2 stand-in that
  SCRUM-442 wired in.
- Velocity block: from the burn execution error, the Gates model. In the burn
  frame with dv_hat the burn direction, a magnitude 1-sigma of f times the
  delta-v magnitude along dv_hat, and a pointing 1-sigma of theta times the
  delta-v magnitude in each of the two directions perpendicular to dv_hat.
  So velocity covariance in the burn frame is
  diag((f |dv|)^2, (theta |dv|)^2, (theta |dv|)^2), rotated into EME2000 by the
  frame aligned with dv_hat.
- Cross terms zero at the epoch. Position and velocity are uncorrelated at the
  burn; the correlations develop under propagation and the growth step produces
  them.

Growth:
- Build the state transition matrix Phi(t) by finite-differencing the SCRUM-451 J2
  propagator: perturb each of the six initial state components by a small epsilon,
  propagate with propagate_j2_series to all output times, and form Phi's columns
  from the differences. One nominal plus six perturbed passes, cheap for an
  865-state file.
- Propagate the covariance as P(t) = Phi(t) P0 Phi(t) transpose, symmetrising
  (P + P transpose) / 2 at each step to hold symmetry. Emit the 6 by 6 per state.
- The along-track growth comes from the velocity block. A velocity 1-sigma
  integrates into an along-track position 1-sigma of order sigma_v times t, so
  over 72 h even a small execution error dominates the position uncertainty. This
  is exactly why the position-only floor barely moved and why the velocity seed is
  the load-bearing part.

## Inputs, for John to set or confirm

Add to OperatorPolicy: the execution-error magnitude fraction (magnitude 1-sigma
as a fraction of delta-v) and the pointing 1-sigma angle. Documented defaults to
start from, to be confirmed: 2 percent magnitude 1-sigma and 1 degree pointing
1-sigma, typical small-thruster values. John owns these numbers. Sreejit is not in
this decision.

## Seam

- leolabs_ephemeris.py: build_screening_ephemeris takes the seed inputs (the
  primary's P_post, the delta-v vector, and the execution-error parameters) and
  emits a per-state growing 6 by 6, replacing the constant
  covariance_6x6_from_position_3x3 path. Keep that helper for the seed's position
  block, drop its use as the whole-file covariance.
- A new helper builds the execution-error velocity covariance in EME2000 from the
  delta-v and the policy parameters.
- A new helper builds Phi(t) from propagate_j2_series and returns P(t).
- The SCRUM-442 wiring point in server.py that currently sets _p_post from
  p_rel_km2 changes to source the primary's own P_post from
  gnc_report.post_burn_covariance_km2, and to pass the delta-v so the velocity
  seed can be built. Trace the exact call rather than assuming its shape.
- Remove the covariance caveats added in SCRUM-440 and SCRUM-442 from the code,
  the operator note and the persisted record, since the covariance is now real.

## Do not

- Do not change kepler_propagate or the SCRUM-451 J2 propagator; use the J2
  propagator for the STM.
- Do not change the screening request, the clear contract, the guard, or the
  ingest path.
- Do not model process noise, drag or higher zonals in the covariance now; the
  execution-error seed grown under J2 dynamics is the model for this ticket.

## Tests

Offline:
- The execution-error velocity covariance places the magnitude variance along
  dv_hat and the pointing variance in the perpendicular plane, and rotates
  correctly for a non-axis-aligned dv.
- Phi(0) is the identity to tolerance, and for a pure velocity seed the
  along-track position variance grows quadratically in time to the expected order.
- build_screening_ephemeris emits a per-state 6 by 6 whose trace increases across
  the horizon, the first state equals the seed, and the seed position block is the
  primary's P_post, not p_rel.
- The SCRUM-440 and SCRUM-442 covariance caveats are gone.

Run python3 -m pytest services/planner and confirm green.

## Live acceptance

Rebuild the planner. Run one on-demand screen, respecting 3 creates per 2 minutes,
and confirm the result is still sane and catalog-consistent, and that the emitted
covariance at 72 h is materially larger than at the epoch, the fanning the floor
did not have. Record in docs/scrum-452/live_verification.md, including the
execution-error parameters used.

## Commit

Branch off main. Include docs/scrum-452/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it checks the
execution-error covariance and the STM growth against a reference, confirms the
seed is the primary's P_post and the covariance grows, confirms the caveats are
removed, runs the offline suite, and reviews the live screen.
