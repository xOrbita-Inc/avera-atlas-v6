# SCRUM-456 live verification — the secondary screen runs off the evaluate path

Run 2026-09-24 against the real LeoLabs API, planner rebuilt from this branch.
Asset **SWARM A**, LeoLabs **L3972**, NORAD **39452**.

## Evaluate no longer blocks on the screen

    POST /v1/evaluate    http 200 in 20.0 s

Against **112 s** for the same request on main (SCRUM-458's live run, same asset
and window). The dashboard's evaluate times out at 60 s, so this is the difference
between a maneuver-recommended decision arriving and failing.

The 20 s that remain are the LeoLabs conjunction-window fetch on the evaluate path
(13 to 20 s, cached by SCRUM-450), not the screen. The screen contributes nothing
to the response now: the worker does the create, the poll and the retrieve.

The secondary check as returned:

    screen_pending                True
    screen_job_id                 9596f2a685a84f6cae69aa107ecec383
    secondary_check_performed     False
    secondary_conjunction_clear   False
    screen_deferred               False
    operator_note                 On-demand secondary screen running in the
                                  background; the maneuver is provisional and is
                                  NOT authorized as secondary-clear until it
                                  resolves. Poll GET /v1/secondary-screen/...

Both MAF booleans are False while pending, which is the literal truth and is what
keeps a pending screen from authorizing anything.

## Poll transitions

`GET /v1/secondary-screen/{job_id}`, one line per poll:

    23:16:39   status=pending     clear=False  screening_id=None    n=0      2 ms
    23:16:51   status=pending     clear=False  screening_id=None    n=0      3 ms
    23:17:03   status=pending     clear=False  screening_id=None    n=0      3 ms
    23:17:15   status=pending     clear=False  screening_id=None    n=0      2 ms
    23:17:27   status=pending     clear=False  screening_id=None    n=0      2 ms
    23:17:39   status=pending     clear=False  screening_id=None    n=0      2 ms
    23:17:51   status=pending     clear=False  screening_id=None    n=0      3 ms
    23:18:03   status=not_clear   clear=False  screening_id=603390  n=1210   4 ms

Pending for ~84 s after the evaluate returned, then resolved. **2 to 4 ms per
poll** — read-only, as the plan requires, with no LeoLabs work on the poll path.

Total screen wall-clock was ~94 s from the start of the evaluate. That is why it
could not live inside a 60 s request.

The resolved payload:

    status                  not_clear
    clear                   False
    pending                 False
    screening_id            603390
    seed_source             cdm_primary_own
    conjunctions_total      1210
    conjunctions_truncated  True      (50 returned, closest first)
    closest_approach_km     4.396523  (STARLINK-1661)
    verdict.evaluated       1210
    verdict.breaches        8
    flagged                 STARLINK-1260, ELECTRON R/B, FLOCK 4BE 36,
                            SHIYAN 32-01, STARLINK-2421, R5-S4,
                            STARLINK-38112, SHIYAN 32-03

The screen's own numbers are unchanged by this ticket, as they should be: 1210
conjunctions, 329 on repaired covariance, 0 skipped, seeded from
`cdm_primary_own`. Same physics (SCRUM-440/451/452), same seed (454), same parse
and dedupe (458), same clear contract — the worker calls the very same
`_run_on_demand_secondary_check`, so there is no second implementation to drift.

    {"event": "leolabs_screening_complete", "screening_id": "603390",
     "cdms": 1261, "conjunctions": 1210, "skipped": 0,
     "covariance_repaired": 329}

## PENDING did not escalate, and did not read as clear

From the same live response:

    decision_state_machine   M0 -> M1   escalated: False   trigger: pc_above_monitor_threshold
    verification.passed      False
    verification.secondary_clear   False
    decision_log             decision=maneuver_recommended
                             secondary_check_performed=False
                             secondary_conjunction_clear=False

Exactly the intended shape: the decision is **provisional** — not verified
secondary-clear, so it cannot be staged — and **not safeheld** on pending alone.
`escalated: False` is the load-bearing observation. It falls out of the existing
semantics rather than a new rule: the M1 to M4 clause fires only on a *performed*
check that came back not clear, and pending is not performed, while the M1 to M2
staging AND includes the secondary guard and fails on it.

## Fail-closed, forced live

Two forced failures, both through the real path:

**1. Ephemeris cannot be built** (inputs present but unusable):

    STATUS error | CLEAR False
    Reason: ephemeris could not be built: ...

**2. LeoLabs rejects the create** — a bogus catalog number, so the request really
goes to LeoLabs and comes back 400:

    leolabs_ephemeris_built   states 865, dynamics two_body_plus_j2
    secondary_screen_resolved job_id a292d394..., status error, clear false
    STATUS error | CLEAR False | terminal True
    Reason: on-demand screening create failed: LeoLabs request failed (HTTP 400)
    poll payload: clear=False, status=error, pending=False

So a screen that cannot run resolves `error`, which is not clear, terminal, and
reported as such on the poll. Async did not soften the SCRUM-442 fail-closed: it
is the same `_run_on_demand_secondary_check` fail-closed path, plus a catch in the
worker so nothing a job raises can leave an entry pending forever.

## Two defects found and fixed on the way

**1. My own new module logged to nowhere.** `secondary_screen_async` used
`logging.getLogger(__name__)`. The service configures structured JSON logging on
the logger named `"planner"` with `propagate=False`, so a module-named logger has
no handler and its records are dropped. `secondary_screen_started` and
`secondary_screen_resolved` produced no output at all until this was fixed. After:

    {"event": "secondary_screen_started",  "job_id": "...", "seed_source": "probe"}
    {"event": "secondary_screen_resolved", "job_id": "...", "status": "error",
     "clear": false}

**2. The same defect was already shipped in SCRUM-458.** `leolabs_cdm_parser` was
given `getLogger(__name__)` in that ticket, so its
`leolabs_covariance_untrusted` warning — the one SCRUM-458 exists to emit — never
reached the service log. It was invisible in that ticket's live run because no
covariance was untrusted (`covariance_untrusted: 0` on both runs), and invisible
to its tests because `caplog` captures at the root regardless of handlers. Fixed
here, one line, and the test now asserts through the real `"planner"` logger so it
means "the warning reaches the service log". Verified emitting:

    {"level": "WARNING", "event": "leolabs_covariance_untrusted",
     "sat_key": "SAT1", "designator": "L2669", "min_eigenvalue_m2": -4596208.9,
     "relative_to_max": 0.35, "detail": "... 35.000% of the largest, beyond the
     1.0% repair band"}

**3. One inconsistency I introduced, caught by reading the live response.** The
verification note still said "Secondary conjunction screen was not performed;
safety could not be established and M4 safe hold is required" for a pending
screen, while the state machine correctly reported `escalated: False`. The text
contradicted the behaviour. A pending screen now gets its own line saying the
maneuver is provisional and where to poll; it still fails verification, because
the burn genuinely is not verified secondary-clear yet.

## Design choices worth a reviewer's attention

**The LeoLabs create runs in the worker, not on the request thread.** The plan
left this open. Consequence: evaluate does *zero* network for the screen, so its
latency is independent of LeoLabs entirely, and the poll key is our own job id
(a uuid) rather than the LeoLabs screening id, which does not exist until the
create returns. The LeoLabs id appears on the entry once it does — `603390` above.
A create that fails surfaces as status `error`, which fails closed, as forced
above.

**The poll caps the conjunction list at 50, closest first.** A live screen returns
over a thousand events (1210 here), and the plan requires this endpoint be fast.
`conjunctions_total` and `conjunctions_truncated` report the full picture, and
every breaching event is in `verdict.breaches` regardless of the cap, so nothing
that drove the verdict is hidden by it.

**The inline path is kept**, behind `secondary_screen_async=False`. The SCRUM-442
suite exercises it, and it is the right shape for any caller that wants a resolved
verdict in one call. The default is async.

**The screening sink no longer fires on the async path**, because there is no
result at artifact-build time. `_persist_screening_result` already 404s against
ingest (documented in SCRUM-442: no endpoint, no table), so nothing that was
landing has stopped landing. The store entry holds everything that record needs
for SCRUM-453 to persist it.

## Quota

One `create_screening` (603390) plus one rejected create (HTTP 400, the forced
failure). Well inside 3 per 2 minutes.

## Suite

    python3 -m pytest services/planner    1760 passed, 2 skipped   (+48)
