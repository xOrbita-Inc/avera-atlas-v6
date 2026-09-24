# SCRUM-441 live verification — on-demand screening, first and corrected submits

Asset SWARM C / L3969, from inside the rebuilt planner container against the real
LeoLabs API.

## Summary

The corrected multipart submit is **accepted**. Transport, field names, the
screening id and the terminal status enum are all confirmed, and LeoLabs accepted
the 865-state 300 s ephemeris without complaint.

But the screening returns **0 conjunctions where the catalog has 64**, and that
discrepancy is not explained by the code. It points at the two physics questions
SCRUM-440 flagged. **SCRUM-442 must not make this screen load-bearing until that
is resolved** — a screen that returns nothing is indistinguishable from a clear
sky, which is the exact failure this path is supposed to prevent.

---

## 1. The original failure, explained (2026-09-24, first round)

Three JSON submits, all `422 {"error": "Invalid Miss Distance"}`, including one
with the thresholds block removed and one adding a top-level `missDistance`.

Root cause, confirmed: the endpoint is **multipart form-data with a file upload**,
not JSON. The form parser saw no fields in a JSON body and failed on the first
required one, which is `missDistance` — so every probe failed identically
regardless of its JSON content. Not a field rename; a transport change.

## 2. The corrected submit — accepted

    POST /catalog/conjunctions/screenings   multipart/form-data
    file  ephemeris.json  application/json  865 states, 460,923 bytes
    form  {"primaryCatalogNumber": "L3969", "missDistance": 1.0,
           "probabilityOfCollision": "0.0001", "mahalanobisDistance": 4.0,
           "primaryHardBodyRadius": 15.0}

    -> ACCEPTED in 0.9 s, screening id 602816

**Confirmed, previously open:**

- **Request body and field names.** All five fields accepted, and echoed back in
  `screeningParameters` unchanged.
- **Screening id.** Integer under key `id` (602816). `_screening_id` handles it.
- **Terminal status enum.** Observed `pending` → `complete`. **"complete"** is
  already in the client's default `is_complete` vocabulary, so the default is
  correct — now verified rather than assumed. Completion took ~23 s.
- **The SCRUM-440 ephemeris is accepted at 865 states / 300 s.** No complaint
  about density, step or size, and `filename: ephemeris.json` is echoed back.

**A finding from the echo worth knowing:** our scalar `missDistance` is expanded
by LeoLabs into a per-axis box —

    "missDistance": "1.0", "missDistanceR": "1.0",
    "missDistanceI": "1.0", "missDistanceC": "1.0"

So a scalar becomes a **1 x 1 x 1 km RIC box**, not a 1 km radius sphere and not
the 2 x 50 x 50 km volume LeoLabs uses for its own reporting. That is a much
tighter screen than the number suggests.

## 3. The result does not match the catalog

| submit | missDistance | conjunctionsCount |
|---|---|---|
| 602816 | 1 km | **0** |
| second | 25 km | **0** |

Cross-checked against `/v1/leolabs/conjunctions` for the same asset over the same
72 h window (read-only, no screening quota):

    events in window        64
    max Pc                  1.723e-04
    events with Pc >= 1e-4  1
    closest miss            1.214 km
    within 1 km             0
    within 25 km            64

So:

- **At 1 km, 0 is correct.** The closest real approach in the window is 1.214 km,
  outside a 1 km box. The screen agrees with the catalog.
- **At 25 km, 0 is a real discrepancy.** 64 events are inside 25 km and one of
  them carries Pc 1.72e-4, above our `probabilityOfCollision` floor of 1e-4. That
  event should plausibly have come back, and did not.

## 4. Two candidate causes, both physics calls — FOR JOHN OR SREEJIT

These cannot be separated without more live creates, and the choice between them
is not a call to make in this ticket.

**(a) The SCRUM-440 covariance floor.** The ephemeris carries LeoLabs' own OD
covariance for SWARM C — position sigmas of 3 to 11 m — held **constant across all
865 states over 72 h**. The catalog's Pc of 1.72e-4 is computed against LeoLabs'
propagated, *growing* covariance for both objects. Screening with an
unrealistically tight and non-growing covariance shrinks the computed Pc, and a
Pc pushed below the 1e-4 filter returns nothing. This is exactly the concern
SCRUM-440 flagged, now with live evidence behind it, and it argues for the
`aps_math.cw_phi_full` option 440 named.

**(b) The propagator over this horizon.** The ephemeris is propagated with
`kepler_propagate`, pure two-body, no J2. Over 72 h in LEO, J2 alone precesses the
orbit plane by several degrees per day, so by the end of the window the submitted
trajectory is materially not where SWARM C will be. A screen against a diverged
trajectory would not line up with real conjunctions regardless of covariance.

Both point the same way: the screen is currently returning a *falsely clear*
answer, which is the dangerous direction. Neither is a code defect in 441 — the
client does what it was asked — and neither should be decided silently.

## 5. The two decisions the fix plan flagged, still interim

- **Hard body radius split.** `primaryHardBodyRadius: 15.0` (our inflated combined
  value) with `secondaryHardBodyRadius` omitted, so each catalog secondary keeps
  its own. The effective combined radius per event is therefore larger than 15 m,
  which raises Pc and returns more conjunctions — the safe direction, but it does
  over-count. Accepted by the API. The true split is a physics call.
- **mahalanobisDistance.** Sent as our 4.0 threshold. **Accepted without a 422**,
  so the fix plan's contingency (drop it if the field is named in an error) was
  not needed. But acceptance is not confirmation of meaning: the reference calls
  it "Supplied Mahalanobis distance for ephemerides file", which still does not
  clearly match our max-Mahalanobis result filter, and it is a candidate
  contributor to the 0-conjunction result.

## 6. Quota

Five `create_screening` calls total across the day: three rejected JSON probes in
the first round, two accepted multipart submits in the second. Both rounds
respected the 3-per-2-minute limit and neither ran in a loop. Read-only GETs were
used for every question that did not require a create.

## Suite

    python3 -m pytest services/planner    1565 passed, 2 skipped   (+56 over main)

All offline with a mocked client, including the multipart wire shape, both
limiters still applying to a multipart post, retry re-sending identical bytes, and
a regression guard on the historical 422.
