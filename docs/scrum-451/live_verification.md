# SCRUM-451 live verification — J2 screening ephemeris recovers catalog events

Run 2026-09-24 against the real LeoLabs API from the rebuilt planner container.
Asset SWARM C / L3969, same asset and window as SCRUM-441's live verification.

## Result: the screen now finds conjunctions where two-body found none

    screening 603002, accepted in 1.0 s, pending -> complete at t+23 s
    missDistance 50 km (wide, deliberately -- see below)

    conjunctionsCount              5        (SCRUM-441 with two-body: 0)
    maxCollisionProbability        5.04e-4  (above our 1e-4 floor)
    aggregateCollisionProbability  1.18e-3
    minMissDistance                950.6 m
    minMahalanobisDistance         0.792

| secondary | TCA | miss | Pc |
|---|---|---|---|
| STARLINK-38092 | 2026-09-26T16:32:24Z | 950.6 m | 5.04e-4 |
| SWARM A | 2026-09-27T11:40:36Z | 15,732 m | 3.42e-4 |
| SWARM A | 2026-09-27T05:28:58Z | 20,972 m | 1.83e-4 |
| SWARM A | 2026-09-27T03:56:02Z | 22,636 m | 1.55e-4 |
| STARLINK-2153 | 2026-09-26T12:36:28Z | 43,995 m | null |

A wide 50 km box was used on purpose, per the plan: it validates the propagator
fix independently of the SCRUM-442 volume decision.

## The recovered events are real

Cross-checked read-only against `/v1/leolabs/conjunctions` for the same asset and
window (67 catalog events):

| object | in catalog? | catalog closest miss | our screen |
|---|---|---|---|
| STARLINK-38092 | yes, 30 events | 1,214 m | 950.6 m |
| SWARM A | yes, 14 events | 17,127 m | 15,732 m |
| STARLINK-2153 | not in the deduped list | — | 43,995 m |

The two objects the screen is most confident about are objects LeoLabs' own
catalog also has in this window, at comparable geometry. The differences of a few
hundred metres are expected: the catalog screens LeoLabs' own orbit determination,
ours screens the trajectory we submitted.

Magnitudes agree at the top end too: our screen's maximum Pc is 5.04e-4 against
the catalog's 1.723e-4 for the window. Same order.

## How far off the old file was

At this epoch the J2 trajectory departs from the two-body one by **2,812 km at
t+72 h** — larger than the 1,400 km in the synthetic Swarm-C case, because the
real orbit's phase differs. That is the error SCRUM-441 was submitting.

## A correction to SCRUM-441's verification

SCRUM-441's live doc concluded that the 0-conjunction result at a **1 km** box was
"correct", reasoning that the catalog's closest approach in the window was 1,214 m
and therefore outside a 1 km box.

**That conclusion was wrong.** On the J2 trajectory the closest approach is
**950.6 m**, which is inside a 1 km box. So the SCRUM-441 screen at 1 km was also a
false clear — it just happened to agree with a catalog number computed from a
different trajectory, which made it look consistent. Both the 1 km and the 25 km
results in 441 were the same failure.

The 441 doc's reasoning that a *geometric* miss-distance filter could not be
explained by the covariance floor still holds, and it is what pointed at the
propagator. It was the "0 at 1 km is correct" line that was mistaken.

## Residual sanity, and what this does NOT settle

- **The covariance floor is still open (SCRUM-440).** Note `minMahalanobisDistance
  0.792` against a 950 m miss: that implies a combined covariance of roughly
  kilometre scale, far larger than the 3-to-11 m sigmas we submitted. The
  secondary's own uncertainty dominates the combination, which dilutes the effect
  of our constant position-only floor — it does not remove it. Still for John or
  Sreejit, unchanged by this ticket.
- **One result carries a null Pc** (`includesNullCollisionProbabilities: true`),
  STARLINK-2153 at 44 km. Expected for an object without enough covariance to
  compute one, and the reason SCRUM-441's parse counts unparseable results rather
  than dropping them.
- **Drag, SRP and higher zonals are still unmodelled.** J2 removes the bulk of a
  1,400-to-2,800 km error; the residual over 72 h in LEO is dominated by drag and
  is named as later refinement, not fixed here.
- **The missDistance volume policy is untouched**, as the plan requires. 50 km here
  is a test choice, not a change to what the system sends. That is SCRUM-442.

## Quota

One `create_screening` in this window, inside the 3-per-2-minute limit, not in a
loop. Every other question was answered with read-only GETs.

## Suite

    python3 -m pytest services/planner    1596 passed, 2 skipped   (+31)

Offline throughout. The J2 propagator is checked against the closed-form secular
nodal regression rate at three inclinations including the retrograde sign flip,
total energy conservation to 1e-9, agreement between the scipy and RK4
integrators, and a divergence-versus-two-body guard at 1, 6, 24 and 72 h so J2
cannot be silently dropped from this path again.
