# SCRUM-451 implementation plan: propagate the screening ephemeris with J2, not two-body

Ticket: https://xorbita.atlassian.net/browse/SCRUM-451
Branch off main. Backend only, planner. Part of SCRUM-433. Modifies the SCRUM-440
screening-ephemeris builder. Blocks SCRUM-442.

## What this is

The merged SCRUM-441 on-demand screen returns a falsely clear result because the
ephemeris we submit is propagated with two-body dynamics. This ticket gives the
screening-ephemeris path a J2-perturbed propagation, so the trajectory we hand
LeoLabs is where the asset will actually be. The change is scoped to that one
path.

## Why, with the evidence

On the first accepted live submit the screen returned 0 conjunctions at a 1 km
miss distance and 0 again at 25 km. The catalog's own screening of the same asset
over the same 72 h window has 64 events, closest miss 1.214 km, one above our 1e-4
Pc floor. Miss distance is a geometric filter, independent of covariance, so the
covariance floor cannot cause a miss at a 25 km box. The cause is the propagator.

Measured two-body versus two-body-plus-J2 divergence for a Swarm-C-like orbit
(a about 6838 km, i 87.35 deg, near-circular), RK4 at a 5 s step:

    t + 1 h     26 km
    t + 6 h    122 km
    t + 24 h   462 km
    t + 72 h  1400 km

The submitted two-body trajectory leaves even a 25 km box within the first hour,
so the screen finds almost nothing where the real asset has conjunctions.

## The change

Add a new J2 propagator that takes an osculating position and velocity and returns
the propagated position and velocity at a requested time under two-body plus J2.
Put it beside kepler_propagate, as a new function, in
services/planner/common/orbit_propagation.py or in libs/aps_math. Do not modify
kepler_propagate.

Then have SCRUM-440's build_screening_ephemeris
(services/planner/common/leolabs_ephemeris.py) generate each state across the
72 h / 300 s window with the new J2 propagator instead of kepler_propagate.
Everything else in the builder stays exactly as it is: the EME2000 frame, the 3x3
to 6x6 covariance construction, the unit conversions, and the JSON shape.

## Propagator design

- Integrate the two-body plus J2 acceleration in ECI. Prefer scipy's solve_ivp
  with DOP853 and tight tolerances if scipy is available in the service, sampled
  at the 300 s output cadence; otherwise a fixed-step RK4 at a small internal step
  (10 s or finer) sampled to 300 s. Keep it deterministic and numpy or scipy only,
  matching kepler_propagate's dependency-light style.
- Units and constants match kepler_propagate: km, km/s, s, and MU_EARTH from the
  module. Add J2 = 1.08262668e-3 and Re = 6378.137 km.
- Frame note: the J2 bulge is aligned with the Earth equator of date, while our
  states are EME2000. The J2000-to-date pole offset is small and its effect over
  72 h is well inside the screening volume, so treat EME2000 as the integration
  frame and record this approximation in a comment. Capturing the J2 secular
  effect is what matters here and it dwarfs the pole-frame subtlety.
- Accuracy: the J2 propagation should reproduce an independent finer-step
  reference to well within the screening volume over 72 h, and its nodal
  regression rate should match the analytic secular rate (below).

## Do not

- Do not change kepler_propagate or any other consumer of it. secondary_horizon,
  maneuver_scorer, the globe viz and the propagator service keep two-body. This is
  the fidelity of the file we submit to LeoLabs, not the planner's internal
  dynamics.
- Do not add drag, solar radiation pressure or higher zonal harmonics now. Name
  them in a comment as later refinement. J2 removes the full 1400 km error above;
  the residual is dominated by drag and is out of scope.
- Do not change the covariance construction, the window or step, the frame, or the
  JSON shape in the builder.
- Do not touch the SCRUM-441 screening request or the missDistance and volume
  policy. That is SCRUM-442.

## Tests

Offline, no network:
- A circular-orbit sanity case: the propagated state stays on a bound orbit and
  the specific energy is conserved to tolerance.
- The J2 secular nodal regression rate over several orbits matches the analytic
  Omega_dot = -1.5 n J2 (Re/p)^2 cos i to tolerance, which proves the propagator
  is actually perturbed rather than silently two-body.
- A regression guard that the two-body-vs-J2 position divergence grows to the
  expected order over 24 to 72 h for a Swarm-C-like orbit, so a future edit cannot
  quietly drop J2.
- build_screening_ephemeris now uses the J2 propagator: the first state is still
  the epoch state, and a state near the end of the window differs from the
  two-body result by the expected order.

Run python3 -m pytest services/planner and confirm green.

## Live acceptance

Rebuild the planner container. Run one on-demand screen, deliberately and
respecting the 3-per-2-minutes limit, with a wide missDistance box (for example
50 km) for the same asset and window used in SCRUM-441's live_verification. Confirm
it now recovers catalog events where two-body returned zero. Record in
docs/scrum-451/live_verification.md the events recovered and the residual sanity.
Using a wide box validates the propagator fix independent of the SCRUM-442 volume
decision.

## Commit

Branch scrum-451-j2-screening-ephemeris off main. Include docs/scrum-451/ in the
commit. No attribution trailers, author John Avera <javera@xorbita.com>. PR against
main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it checks the
J2 propagator against the analytic secular rate and a reference, confirms the
builder path uses it and nothing else changed, runs the offline suite, and reviews
the live screen recovering the catalog events.
