# SCRUM-442 implementation plan: wire the secondary-clear guard to the on-demand screen, screen wide and decide narrow

Ticket: https://xorbita.atlassian.net/browse/SCRUM-442
Branch off main. Backend only, planner. Part of SCRUM-433. Depends on SCRUM-440
(ephemeris), SCRUM-441 (client and parser), SCRUM-451 (J2 propagation), all merged.

## What this is

Feed the SCRUM-381 secondary-clear contract from the real on-demand screening
result, remove the SCRUM-431 deferral stub, and re-enable SECONDARY_SCREEN_ENABLED.
The guard itself already fails closed on two booleans. This ticket is the upstream
wiring that runs the screen, judges the returned conjunctions against our clear
contract, produces those two booleans, persists the result, and flips the screen
on.

The decision on how wide to screen is made and recorded in
claude/SCRUM-433_secondary_screen_decisions_2026-09-24.md. Read it first. This plan
bakes it in: screen wide, decide narrow.

## The seam, as it actually is

- guard_secondary_clear in safety_monitor.py reads secondary_check_performed and
  secondary_conjunction_clear from GuardInputs and fails closed when the check was
  not performed. It does not change.
- build_atlas_artifact in atlas_artifact.py sets those two booleans. Today, with
  the SCRUM-431 deferral, it sets secondary_check_performed=False whenever the
  screen is deferred or disabled, which keeps the guard inert.
- server.py builds the artifact with secondary_screen_enabled=SECONDARY_SCREEN_ENABLED
  (default off) and, per SCRUM-431, fetches no catalog.

So the work lands where the two booleans are produced, not in the guard.

## Screen wide, decide narrow (Decision 1, baked in)

The screen submits a wide volume and the decision is made locally on the returned
events.

- Submit missDistance at a wide screening volume, 50 km, not
  policy.min_miss_distance_km. Add a screening_volume_km to OperatorPolicy
  (default 50.0, capped at the API maximum of 100) and use it for the request's
  missDistance. This is separate from min_miss_distance_km, which stays the action
  floor the guard decides against.
- Stop sending probabilityOfCollision and mahalanobisDistance as server-side
  filters on the request. Our clear contract is an OR of Pc, miss and Mahalanobis,
  so a server-side Pc floor would drop events that breach on miss or Mahalanobis
  alone. Full recall in the volume, decision made locally. This also retires the
  unconfirmed-meaning mahalanobisDistance request field, since we no longer send
  it.
- Keep primaryHardBodyRadius on the request. It drives the Pc LeoLabs computes and
  returns, which the guard then reads.

This changes SCRUM-441's build_screening_request body shape. Update it and its
offline tests: missDistance is the wide screening volume, probabilityOfCollision
and mahalanobisDistance are absent, primaryHardBodyRadius stays.

## Producing the two booleans

In the path that builds the secondary block for build_atlas_artifact, when
SECONDARY_SCREEN_ENABLED:

1. Build the post-burn ephemeris (SCRUM-440) from the post-burn state and P_post,
   and run the on-demand screen with run_screening (SCRUM-441) at the wide volume.
2. Evaluate each returned conjunction against the SCRUM-381 clear contract, the
   same Pc-or-miss-or-Mahalanobis logic the internal decision already uses. Do not
   invent new threshold logic. Trace the existing evaluation and reuse it, the way
   SCRUM-441 traced the thresholds to OperatorPolicy rather than inventing them.
   An event breaches if Pc is at or above pc_maneuver_threshold, or miss is at or
   below min_miss_distance_km, or Mahalanobis is at or below
   mahalanobis_screen_threshold.
3. Set secondary_check_performed=True and secondary_conjunction_clear=(no returned
   event breaches). If any event breaches, the screen is not clear.
4. Fail closed on any failure. run_screening raises a typed LeoLabsScreeningError
   on timeout, transport error, rate-limit refusal or no access. Catch it and set
   secondary_check_performed=False so the guard treats it as NOT CLEAR and
   escalates to M4. A screen that could not run must never read as a clear screen.
   An empty result, by contrast, is a real clear sky: check_performed=True,
   conjunction_clear=True.

## Latency

On-demand screening takes 30 s to 2 min and run_screening polls to completion.
This runs on the maneuver-planning path, which tolerates that latency, and it stays
behind SECONDARY_SCREEN_ENABLED so the default fast path is unchanged. Do not block
an unrelated fast endpoint on the poll. If the planning call cannot absorb a
multi-minute wait, fail closed on a bounded timeout rather than hanging, and record
the timeout as NOT CLEAR.

## Persistence

Persist the screening result alongside the decision, the way SCRUM-429 links a
decision to its stored CDM under ADR-008. The stored record is the screening id and
the returned conjunctions with covariance, tied to the decision log id, so the
NOT CLEAR or CLEAR verdict can be audited against the exact events it was made on.

## The covariance floor stays as it is (Decision 2 is a fast-follow)

The submitted covariance is still SCRUM-440's constant position-only floor. Per the
decision memo, 442 ships with that behind an explicit caveat and does not implement
the execution-error covariance. Add a one-line note in the code and the live
verification that the primary covariance is a floor and the honest growth model is
the tracked fast-follow. Do not build it here.

## Re-enable the screen

Remove the SCRUM-431 deferral stub so the live on-demand path is the real path, and
default SECONDARY_SCREEN_ENABLED on for the demo once the path is proven on the
stack. Keep the flag so it can be turned off.

## Do not

- Do not change guard_secondary_clear or the fail-closed contract.
- Do not implement the execution-error covariance (Decision 2 fast-follow).
- Do not change the J2 propagator or the ephemeris builder from SCRUM-451.
- Do not pre-filter the screen on Pc or Mahalanobis server-side.

## Tests

Offline, mocked screening client:
- A maneuver that creates a conjunction breaching the contract yields
  secondary_conjunction_clear=False and the artifact escalates to NOT CLEAR / M4.
- A clean maneuver, empty screen result, yields check_performed=True,
  conjunction_clear=True, and clears.
- A screening failure (timeout, transport, rate-limit, no access) yields
  check_performed=False and fails closed to NOT CLEAR, never a silent clear.
- The request carries the wide missDistance and no Pc or Mahalanobis filter.
- The decision persists the screening result tied to the decision log id.

Run python3 -m pytest services/planner and confirm green.

## Live acceptance

Rebuild the planner. On the stack with SECONDARY_SCREEN_ENABLED on, run the
maneuver-planning path for a real asset: confirm a maneuver that creates a
contract-breaching conjunction is held NOT CLEAR and escalates, a clean case
clears, and the screening result persists with the decision. Respect the 3 creates
per 2 minutes. Record in docs/scrum-442/live_verification.md.

## Commit

Branch off main. Include docs/scrum-442/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it checks the
wide-volume request and the local clear-contract evaluation against real code,
confirms the fail-closed behavior on a screen that could not run, runs the offline
suite, and reviews the live NOT CLEAR and CLEAR cases with the persisted result.
