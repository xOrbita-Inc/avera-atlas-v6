# SCRUM-458 live verification — non-PSD result covariance no longer drops conjunctions

Run 2026-09-24 against the real LeoLabs API, planner rebuilt from this branch.
Asset **SWARM A**, LeoLabs catalog **L3972**, NORAD **39452** — the asset that
produced screening 603312.

## SAT1 identity, confirmed

The plan asked whether our primary is consistently SAT1 in screening results. On
all 1255 result CDMs of screening 603312:

| field | value |
|---|---|
| `SAT1_OBJECT_DESIGNATOR` | `L3972`, **1 distinct value** |
| `SAT1_EPHEMERIS_NAME` | `ephemeris.json`, 1 distinct value (our uploaded file) |
| `SAT2_OBJECT_DESIGNATOR` | **718 distinct** secondaries |

So in an on-demand screening result SAT1 is always our submitted ephemeris and
SAT2 is always the catalog object. The log messages name the right satellite.

The parser still resolves the roles by matching our catalog id against both
designators rather than assuming SAT1, which stays correct and is unchanged; the
labels in the warnings come from the resolved `sat_key`, so they are right either
way.

## REPAIR_RTOL, chosen from the data rather than proposed

Measured over all 1255 result CDMs of screening 603312, on the rotated 6x6:

| quantity | value |
|---|---|
| SAT1 covariances with a negative minimum eigenvalue | **1249 of 1255** |
| `|min_eig| / max|eig|` — median | 2.0e-9 |
| `|min_eig| / max|eig|` — **worst** | **5.37e-3** (0.54%) |
| worst absolute minimum eigenvalue | -3.14e5 m^2 |
| SAT2 (catalog secondaries) worst ratio | 4.3e-9 |

**`_PSD_REPAIR_RTOL = 1.0e-2`**, which covers every observed case with **1.86x**
in hand while staying two orders of magnitude below the anisotropy itself.

The band has to be relative, and this data shows why: the worst absolute negative
is -3.14e5 m^2, which sounds enormous, and is 0.54% of a largest eigenvalue of
~5.8e7 m^2. An absolute threshold would either reject all of these or accept a
genuinely broken covariance on a small object.

The negatives are ours, not the catalog's, exactly as the plan predicted: SAT2's
worst ratio is 4.3e-9, inside the old 1e-8 tolerance, so no catalog secondary was
ever being rejected. It is our own SCRUM-452 anisotropy that does not survive the
CDM's fixed precision.

## The safety direction, checked rather than assumed

Clipping a negative eigenvalue to zero claims more certainty along that axis,
which raises the Mahalanobis distance and pushes toward CLEAR. So the clip was
compared against the conservative alternative — reflecting the negatives to
`|lambda|`, which cannot reduce any uncertainty — over all 1249 non-PSD CDMs:

| | |
|---|---|
| events where the two choices differ at all | 1249 |
| max relative difference in Mahalanobis | 0.99 (about 2x) |
| **events where the choice flips the Mahalanobis limb** | **0** |

Not one event changes risk-relevance between the two. The large Mahalanobis values
come from the real anisotropy, not from the clip: `cond(p_rel)` after repair peaks
at 2.9e7, a sigma ratio of ~5,400, which is the km-along-track versus
metres-radially shape SCRUM-452 submits. The clip is safe on this data, and the
offline suite asserts the property so it stays that way.

## Why the repair runs after the diagonal-comment guard

Clipping changes the diagonal. Measured over the same set:

| | |
|---|---|
| diagonal entries shifted more than `_DIAG_RTOL` (0.1%) | **1992 of 7494** |
| worst relative shift | **40x** |

Had the repair run before `_assert_diagonal_matches_comments`, that guard would
have rejected these events and this ticket would have fixed nothing. The PSD limb
therefore runs last, and the diagonal guard keeps checking the rotation exactly as
the CDM states it, which is what it was written to measure. A test asserts both
halves: that a repaired CDM passes the diagonal guard, and that the clip really
does move a diagonal past the tolerance, so the first test cannot go vacuous.

## Second finding: the dedupe key collapsed the entire screening

Fixing the PSD drop exposed a second and more severe instance of the same bug.
Screening result CDMs do not carry a usable event identity:

| field | distinct values over 1255 CDMs |
|---|---|
| `COMMENT_EVENT_ID` | **1** — the constant string `"0"` |
| `COMMENT_ID` | **1** — `"603312"`, the *screening* id, not a CDM id |
| `MESSAGE_ID` | 1201 |
| `SAT2_OBJECT_DESIGNATOR` | 718 |
| `(SAT2, TCA)` pairs | ~1021 |

`leolabs_conjunction_list.event_key` prefers `event_id`, so it returned
`"event_id:0"` for **every** CDM in the screening and `dedupe_by_event` collapsed
the whole screened set to **one conjunction**. That is a silent drop of an entire
result, and it was invisible while the PSD guard was already rejecting nearly every
CDM before dedupe ran — there was nothing left to collapse.

Repairing the covariance without this would have delivered nothing observable: one
conjunction before, one after, still CLEAR.

The fix is narrow. `dedupe_by_event` takes an optional key function, defaulting to
`event_key` so the live feed is untouched, and the screen passes
`screening_event_key` = `(secondary designator, TCA)`. The live feed keeps
`event_key`, because its `COMMENT_EVENT_ID` is a real event id.

Why that key: a conjunction within one screening is one close approach, one
secondary at one TCA. The same secondary genuinely recurs — **185 secondaries have
more than one TCA across the 72 h horizon, one has 29** — so deduping on the
secondary alone would discard **303 real conjunctions**. Identical `(secondary,
TCA)` CDMs do collapse, which is the intent: 217 pairs appear more than once.

One imprecision, in the safe direction: the parser's TCA is microsecond-resolution
`TCA_ISO`, so a reissue whose TCA is refined by microseconds counts as two events
rather than one. That is why the re-parse below yields 1186 rather than the ~1021
second-resolution pairs. It over-counts rather than drops, so it cannot hide a
conflict. Collapsing on a second-rounded TCA would be the tighter choice and is
left for review rather than slipped in here.

## Screening 603312 re-parsed: the verdict flips

Same stored result, same asset, read-only (no create spent):

| | before this branch | after |
|---|---|---|
| result CDMs retrieved | 1255 | 1255 |
| CDMs skipped | **1249** | **0** |
| conjunctions judged | **1** | **1186** |
| covariance repaired | n/a | 323 |
| covariance untrusted | n/a | 0 |
| **verdict** | **CLEAR** | **NOT CLEAR** |
| breaching events | 0 | **7**, all on Mahalanobis |
| closest approach | 48.571 km | **4.397 km** (STARLINK-1661) |

The events that were being dropped, closest first:

    STARLINK-38112    miss  4.799 km   mahalanobis 1.387
    STARLINK-1260     miss 10.994 km   mahalanobis 2.905
    FLOCK 4BE 36      miss 14.431 km   mahalanobis 3.141
    ELECTRON R/B      miss 16.794 km   mahalanobis 3.364
    R5-S4             miss 15.797 km   mahalanobis 3.891
    SHIYAN 32-03      miss 24.280 km   mahalanobis 3.820

Against a 4.0 threshold. Seven risk-relevant conjunctions were absent from a
result that read CLEAR. That is the false CLEAR the ticket describes, reproduced
and then removed.

Note none of them breaches the 1 km miss limb — the closest is 4.4 km. The
covariance limb is the only thing that catches these, which is precisely why
dropping events on a covariance technicality was dangerous.

## Live acceptance: one fresh on-demand screen

`POST /v1/evaluate`, SWARM A / 39452, live LeoLabs, 112 s:

    {"event": "leolabs_screening_complete", "screening_id": "603365",
     "cdms": 1253, "conjunctions": 1195, "skipped": 0,
     "covariance_repaired": 332, "covariance_untrusted": 0}

**`skipped: 0`** — the acceptance criterion. 1195 conjunctions judged, 332 of them
on a repaired covariance.

The decision:

    secondary_check_performed     True
    secondary_conjunction_clear   False
    closest_approach_km           4.396523   (STARLINK-1661)
    flagged                       FLOCK 4BE 36, SHIYAN 32-03, SHIYAN 32-01,
                                  ELECTRON R/B, STARLINK-38112, R5-S4
    screening id                  603365

Operator note, as recorded:

> On-demand secondary screen NOT CLEAR: 6 of 1195 returned conjunction(s) breach
> the clear contract on mahalanobis. MAF requires M4 safe hold. Screening id
> 603365. 332 of 1195 carried a covariance repaired for numerical non-PSD.

So the screen now fails closed on this asset and window, the guard escalates to M4
safe hold, and the repair count is in the record rather than implied. This is a
real change of outcome: the same request previously produced a clear screen.

## The flag path did not trigger live, by design

`covariance_untrusted: 0` on both runs. Every non-PSD covariance observed live is
numerical, well inside the band, and gets repaired. The flag-and-fail-closed path
is therefore exercised by the offline suite rather than by this run — a covariance
non-PSD by 35% of its largest eigenvalue is carried, marked, and breaches on the
`covariance_untrusted` limb, with its Mahalanobis deliberately not computed.

## Numbers that differ from the ticket

The ticket reports "skipped 348 of 349". On screening 603312 as it stands I measure
1249 of 1255 CDMs rejected on non-PSD, deduping to 1 conjunction. The mechanism is
exactly the one described; the counts differ, so the earlier observation was
probably a different screening or window. Recording what I measured rather than
restating the ticket's figures.

Also, repeated `get_screening_cdms("603312")` calls returned slightly different
distinct-secondary counts (712 then 718). That is the retrieved-versus-total
pagination drift SCRUM-439 documented as correct behaviour under a long pull, not a
new fault, and it does not affect any conclusion here.

## Invariant, as shipped

No conjunction the screen could not fully assess is silently absent from a result
that reads CLEAR. Three ways that could happen, all closed:

1. Non-PSD of numerical scale — repaired, event kept and judged normally.
2. Non-PSD beyond the band — kept, marked untrusted, breaches the contract, and
   its Mahalanobis is not computed since inverting that matrix can read as
   arbitrarily many sigma, i.e. as clear.
3. A CDM that could not be parsed at all — the set is incomplete, so
   `_run_on_demand_secondary_check` refuses to certify CLEAR over it and fails
   closed naming the reason. This closes the same hole reached by a different skip
   reason.

Symmetry stays a hard error, unrepaired: an asymmetric covariance is a real
convention fault with no benign reading.

The live listing and evaluate paths keep the strict raise they have today
(`strict_psd=True` by default), so only the screen carries unassessable events.

## Quota

One `create_screening` (603365). Everything else was read-only GETs against the
stored 603312 result.

## Suite

    python3 -m pytest services/planner    1712 passed, 2 skipped   (+43)

No existing test changed.
