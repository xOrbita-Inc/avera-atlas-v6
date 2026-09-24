# SCRUM-441 live verification — first on-demand screening submit

Run 2026-09-24 against the real LeoLabs API from inside the rebuilt planner
container, asset SWARM C / L3969.

## Outcome: the submit was rejected. The request body is NOT confirmed.

The plan expected the first live submit to confirm the request-body field names
and the terminal status enum. It did not confirm them — it rejected the body, and
the three things step 7 was meant to answer are consequently still open.

This is recorded as a blocking finding rather than worked around. **SCRUM-442 must
not make this screen load-bearing until the request body is confirmed.**

## What was attempted

Three `create_screening` calls, the full quota for one 2-minute window, run
deliberately and not in a loop. Each used a real ephemeris built by the SCRUM-440
builder from SWARM C's live state and its own position covariance:

    asset            L3969, epoch 2026-09-24T11:34:28.729701Z
    P_post (km^2)    diag [1.182e-4, 2.767e-5, 9.74e-6]
    position sigmas  10.87, 5.26, 3.12 m
    ephemeris        865 states, 460,923 bytes,
                     2026-09-24T11:34:28Z .. 2026-09-27T11:34:28Z

| # | body | response |
|---|---|---|
| 1 | as `build_screening_request` emits it | `422 {"error": "Invalid Miss Distance"}` |
| 2 | same, with the whole `thresholds` block **removed** | `422 {"error": "Invalid Miss Distance"}` |
| 3 | same as 1, plus a top-level `"missDistance": 1000.0` (metres) | `422 {"error": "Invalid Miss Distance"}` |

Probe 2 is the informative one: removing our `thresholds.maxMissDistanceKm`
entirely did not change the error, so the rejection is **not** our thresholds
block. A required miss-distance parameter is expected under a name that is not in
our body at all.

Probe 3 tested the most plausible inference — every other distance in this API is
metres (position, velocity, covariance, and the `maxRelativePosition*` filters),
so a top-level `missDistance` in metres was the informed guess. Also rejected.

## Why probing stopped there

`create_screening` is limited to 3 per 2 minutes on top of the org-wide 4 req/s,
against the company's production trial account. Converging on an unknown schema by
guessing field names would mean many more creates with no guarantee of
convergence, which is not a reasonable use of a rate-limited production endpoint.

Read-only discovery was tried first and found nothing: `/openapi.json`,
`/swagger.json`, `/docs`, `/schema` and `/catalog/conjunctions/screenings/schema`
all 404. A `GET /catalog/conjunctions/screenings` returns a LeoLabs-shaped
`{"error":"HTTP error code 404"}`, so the path exists but does not serve a schema.

**What is needed:** LeoLabs' on-demand screening API documentation, or the field
list from Lois Reid. It is one question, and it unblocks the whole path.

## What this DID confirm

- **The transport and auth are right.** A 422 with a LeoLabs-shaped JSON error
  body means the request reached the screening endpoint, authenticated, and was
  parsed. It is a schema rejection, not an access or transport failure.
- **The account is not blocked from on-demand screening.** A missing entitlement
  would be 402/403; this is 422 on content.
- **The SCRUM-440 ephemeris builds and serialises fine** at 865 states / 461 KB
  from real live state and real covariance, and was accepted as far as the body
  schema check.
- **The module fails closed on exactly this.** A rejected body raises
  `LeoLabsScreeningError` and never returns an empty conjunction set. The real
  422 is pinned as a test
  (`test_the_real_422_from_the_first_live_submit_fails_closed`), including that it
  is *not* classified as `Unavailable`, because the account has access and it was
  the body that was wrong.

## Still open — the three step-7 questions, none answered

1. **The request-body field names.** Rejected, see above. Needs the schema.
2. **The terminal status enum.** Not reached; nothing was ever created to poll.
   `is_complete` remains injectable and the client's default vocabulary
   (completed/complete/done/finished) is unverified.
3. **Whether LeoLabs accepted the 440 ephemeris, and its interpolation at the
   300 s step.** Not reached. The SCRUM-440 step-size question is therefore still
   open exactly as 440 left it.

## The SCRUM-440 covariance floor — still for John or Sreejit

SCRUM-440 flagged the constant, position-only covariance (velocity block zero,
same covariance on all 865 states across 72 h) for confirmation here, from what a
live result showed. **No live result exists**, so nothing here informs that
decision and it is handed on unchanged.

Worth noting for whoever picks it up: the real covariance used in these probes was
tiny — 3 to 11 m position sigmas from LeoLabs' own OD of a tracked asset. If a
real post-burn P_post is of that order, a covariance that does not grow across 72 h
is a much stronger assumption than it would be with kilometre-scale uncertainty,
because the screen would be drawing a very tight volume around a trajectory whose
true uncertainty is fanning out. That is an argument for the
`aps_math.cw_phi_full` option 440 named, not a decision.

## Suite

    python3 -m pytest services/planner    1547 passed, 2 skipped   (+38)

All offline, mocked client, no network. The offline suite is complete and green;
it is the live acceptance that is blocked.
