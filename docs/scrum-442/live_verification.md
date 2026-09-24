# SCRUM-442 live verification — the secondary-clear guard on the real screen

Run 2026-09-24 against the real LeoLabs API, planner rebuilt from this branch,
SECONDARY_SCREEN_ENABLED on. Asset SWARM C / 39453, real conjunction cdm_id
79669937218 (STARLINK-30994).

## Both verdicts proven end to end

**CLEAR** — screening `603055`, evaluate 97.7 s

    secondary_check_performed     True
    secondary_conjunction_clear   True
    screen_deferred               False
    closest_approach_km           48.55   (STARLINK-36825)
    verification.secondary_clear  True

    note: On-demand secondary screen clear: 1 conjunction(s) returned within the
    50 km screening volume, none breaching the clear contract (Pc >= 1e-06,
    miss < 1 km, or Mahalanobis <= 4). Screening id 603055. Submitted covariance
    is a constant position-only floor; the growth model is a tracked fast-follow.

**NOT CLEAR** — screening `603057`, evaluate 108.5 s

Same asset, same real screen, with `min_miss_distance_km` raised to 50 km so the
returned event breaches the miss limb. A real screening result judged against a
stricter floor, rather than fabricated conjunction data:

    secondary_check_performed     True
    secondary_conjunction_clear   False
    screen_deferred               False
    flagged_objects               ['STARLINK-32174']
    closest_approach_km           47.48
    verification.passed           False
    verification.failure_reasons  ['Secondary conjunction detected with: STARLINK-32174.']

    note: ... NOT CLEAR: 1 of 1 returned conjunction(s) breach the clear contract
    on miss_distance. MAF requires M4 safe hold. Screening id 603057. ...

So the screen runs live, the local clear contract decides, the verdict reaches
`verification`, and both outcomes carry the covariance caveat.

**Fail-closed** is covered offline across all four failure modes (timeout,
transport, rate-limit, no access) plus every missing input, each asserting
`performed=False, clear=False, deferred=False`. It was also observed
incidentally: before the gating fix below, evaluates without LeoLabs credentials
produced `holding M1: secondary_conflict_clear not satisfied`, which is the guard
doing exactly its job.

## Three things found while wiring this, worth review

**1. The screen was running on no-burn decisions.** The first live evaluate spent
97.8 s and a rate-limited screening create on a decision that recommended
no-burn — and then discarded the result, because the A4 post-maneuver projection
that consumes it is only built when a maneuver is recommended. At 3 creates per
2 minutes, a handful of routine evaluates would exhaust the quota, after which
the screen fails closed for the decisions that actually need it. The screen is
now gated on `scoring.is_maneuver_recommended()`. Not in the plan; added because
the live run surfaced it.

**2. Default-on has a wide blast radius.** With `SECONDARY_SCREEN_ENABLED`
defaulting to true, any environment that cannot reach LeoLabs cannot stage a
maneuver at all — a screen that could not run is NOT CLEAR, so the decision holds
M1. That is the correct safety posture and it is the point of the guard, but it
is a real behavioural change: two existing state-machine tests began failing
because they exercise the validity seam without LeoLabs. They now disable the
screen explicitly, which is honest about what they test. Flagged so the default
is a deliberate choice rather than a side effect.

**3. The persistence does not land.** The planner side is implemented, guarded
and verified to fail safely, but the ingest service has no `/screening/persist`
endpoint and no table for it: `PlannerOutput` is a fixed schema with no room for
a screening reference, and adding one is an ingest change while this ticket is
scoped to the planner.

Observed live: the POST 404s, an `audit_write_failed` with
`kind: screening_result, record_id: 603055, exc: status 404` is logged, and the
evaluate still returns its decision with `screening_record_id: null`. That is the
correct failure behaviour, but it is not the feature.

**Still needed:** an ingest endpoint and table mirroring `cdm_records` /
`PlannerOutput`, after which `_persist_screening_result` works unchanged. Until
then a NOT CLEAR verdict is auditable through the operator note and the service
logs, but not through the store — which is short of what ADR-008 asks for and
short of what this ticket asked for.

## The covariance, twice over

Per the decision memo this ships with SCRUM-440's floor, caveated not fixed. Two
separate approximations are in play and both are stated in the code, the operator
note and the persisted record:

- **Constant in time.** The submitted covariance does not grow across the 72 h
  horizon. It is a floor; the growth model is the tracked fast-follow.
- **Wrong quantity.** The ephemeris wants the *primary's* post-burn position
  covariance. What the evaluate path holds is `p_rel_km2`, the *combined*
  relative covariance. Passing it over-states the primary's own uncertainty,
  which is the conservative direction for a screen -- a larger covariance lowers
  Mahalanobis distance, so more events fall inside the risk-relevant radius and
  more breach. It is a stand-in, and it is the second half of what the
  fast-follow should replace.

## Request shape, as submitted

    primaryCatalogNumber  L3969
    missDistance          50.0      (the wide screening volume, not the 1 km floor)
    primaryHardBodyRadius 15.0

No `probabilityOfCollision` and no `mahalanobisDistance`: our contract ORs Pc,
miss and Mahalanobis, so a server-side Pc floor would have dropped the very event
that produced the live NOT CLEAR, which breached on miss alone with a negligible
Pc. Screen wide, decide narrow — and the NOT CLEAR case is the evidence that the
OR matters.

## Quota

Three `create_screening` calls, spread deliberately and inside the 3-per-2-minute
limit: one discarded no-burn run that prompted finding 1, then 603055 and 603057.

## Suite

    python3 -m pytest services/planner    1624 passed, 2 skipped   (+28)
