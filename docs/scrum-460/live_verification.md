# SCRUM-460 live verification — evaluating a clicked conjunction without the window

Run 2026-09-25 on the local stack against the real LeoLabs API. Planner rebuilt from
this branch. No KVM deploy; that is John's manual step.

## The seam, confirmed

`/v1/evaluate` resolved the operator's clicked row with
`select_conjunction(fetch_leolabs_conjunctions(primary), selector)`, and the
no-selector path with `fetch_leolabs_conjunction(primary)`. Both go through
`_scorable_in_risk_order`, the unbounded whole-window pull, called directly on the
event loop. SCRUM-459 left that pull unbounded on purpose — a truncated window would
404 the operator's own selection — and left evaluate out of scope. So for SWARM B
this was still the 75,257-CDM, 77-page, **773 s** pull, on the single worker.

## The plan's premise was wrong, and the API says so

The plan says to fetch the one CDM with `client.get_cdms`, which "takes a cdm id".
It does not work: `/catalog/conjunctions/cdms/{id}` returns a **conjunction summary,
not a CCSDS CDM**:

    {"id": 79867238980, "conjunction": 3590491017,
     "tca": "2026-09-26T18:05:15.953564Z", "sat1": "L3969", "sat2": "L186018",
     "missDistance": 11226.646, "relativeSpeed": 14828.261,
     "collisionProbability": 7.035e-05, "source": "leolabs"}

No state vectors, no covariance, so `parse_leolabs_cdm` cannot read it and nothing
can be scored from it. `?format=ccsds` is ignored and returns the same shape.
`/catalog/conjunctions/{conj}/cdms` returns 83 more of the same summaries.
`search?id={cdm_id}` ignores the `id` and returns the whole window.

`get_cdms` also read `data["cdms"]` while the response key is `conjunctions`, so it
returned an empty list for **every id ever passed to it**. Nothing called it, which
is why that went unnoticed. Renamed to `get_cdm_summaries` so the next reader is not
misled into expecting a CDM.

**What does work** is a narrow search: the summary gives the TCA and the two object
designators, and a search pinned to that object pair inside a TCA bracket returns
the real CCSDS CDMs for that one event in a single page. Two requests instead of 77
pages.

## The decision is bit-identical

Proved on SWARM C, where the window pull is affordable enough to compare against:

    window path:   1,311 conjunctions in 19.5 s; worst cdm_id=79867238980
    summary:       sat1=L3969 sat2=L186018 tca=2026-09-26T18:05:15.953564Z
    narrow search: 83 CDMs in one page, exact id matches=1, 4.06 s total

    scalar mismatches:                NONE
    r_rel_km bit-identical:           True
    v_rel_km_s bit-identical:         True
    p_rel_km2 bit-identical:          True
    to_conjunction_state identical:   True

`to_conjunction_state()` is the whole of what the scorer reads, so an identical one
means an identical decision. This changes how the conjunction is found and nothing
about how it is judged.

## A clicked SWARM B conjunction, live

The row, from the (bounded, SCRUM-459) list:

    cdm_id=79868420478  STARLINK-36467  miss 3.393 km  pc 3.16e-04

Evaluating it:

    POST /v1/evaluate   http=200 in 8.6 s

    source            leolabs          covariance_source  real_cdm
    recommendation    prograde
    primary           SWARM B 39451
    secondary         STARLINK-36467 67348
    miss_distance_km  3.393156
    tca_utc           2026-09-30T02:14:31.577634Z
    pc_pre            1.1857e-04

**8.6 s against 773 s**, and the decision is for the exact conjunction clicked —
same secondary, same miss distance to six decimals, same TCA as the row.

(`pc_pre` 1.19e-4 differs from the row's `pc` 3.16e-4 by design and not by this
change: the row shows the CDM's own COLLISION_PROBABILITY, while `pc_pre` is the
planner's Pc computed from the real geometry. The parser has always left
`pc_precomputed` None for exactly this reason.)

From the log, the same request:

    {"event": "leolabs_cdm_id_resolved", "cdm_id": "79868420478",
     "primary_norad": 39451, "candidates_in_bracket": 5,
     "secondary": "L150849", "tca_utc": "2026-09-30T02:14:31.577634Z"}

**5 CDMs examined, not 75,257**, and window-search markers
(`leolabs_cdm_cache_miss`, `leolabs_cdm_pull_truncated`) in that window: **0**. The
clicked path made no window search at all.

## /health and the other assets

`/health` sampled once a second through the SWARM B evaluate:

    1.5 / 2.2 / 2.3 / 2.5 / 2.8 ms      (idle baseline is 1.2 - 1.6 ms)

And with a SWARM B evaluate and a cold SWARM C list overlapping:

    SWARM B evaluate   http=200 in 10.1 s
    SWARM C list       http=200 in 21.8 s
    /health throughout 1.9 / 1.9 / 2.5 ms

So an evaluate no longer blocks anything, and the dense asset's evaluate and another
asset's listing run at the same time.

## A bug in my own implementation, caught live

The first live attempt returned **503, "LeoLabs rejected the credentials (HTTP
403)"** — which reads like an auth problem and is nothing of the kind.

The summary for that SWARM B row came back:

    sat1 = L150849   (a Starlink)
    sat2 = L5429     (SWARM B, our asset)

I had passed `object1=sat1`, assuming the summary puts our asset first. It does not.
LeoLabs requires `object1` to be an object the subscription covers, so passing the
Starlink refuses the entire request with a 403. It worked on SWARM C only because
that row happened to have our asset in `sat1`.

This is precisely the trap `leolabs_cdm_parser._resolve_roles` was written to avoid
— "Do not assume SAT1 is our asset" — and I walked into it one module over. Fixed by
always sending our catalog as `object1` and whichever of the pair is not ours as
`object2`, with a test that runs both orderings and asserts the query is the same.

Worth recording because the symptom is so misleading: a 403 on a live credential
that works for every other call.

## Routing, and what still needs the window

| selector | seam | window pull |
|---|---|---|
| `cdm_id` (what a click sends) | summary + narrow search | **none** |
| `event_id` | whole-window search | yes, off the loop |
| `secondary_norad` | whole-window search | yes, off the loop |
| none ("evaluate the worst") | `fetch_leolabs_conjunction` | yes, off the loop |

The three fallbacks have nothing narrower to search on — an event id is not a
LeoLabs query parameter, and "the worst conjunction" is only knowable from the whole
window. They keep the full pull and are deliberately **not** bounded: a truncated
window could hand back a different conjunction, and the list is what is allowed to
be partial, not the decision. What changed for them is that they run on the
SCRUM-459 executor under the same in-flight cap, so the slow fallback is survivable
rather than fatal. A no-selector SWARM B evaluate is still slow — it needs the whole
773 s window — but it no longer takes the planner with it.

The dashboard always sends `cdm_id` for a clicked row, so the fast path is the one
the demo takes.

## The trade-off in reusing the SCRUM-459 cap

Evaluate now shares the list's in-flight cap of two, so a clicked evaluate can be
shed with the typed 503 while two window pulls are running. That is the honest
behaviour — the shed body says busy and carries no `recommendation`, so it cannot be
mistaken for a decision — and the retry is immediate, but it is a real behaviour
change worth a reviewer's attention. The alternative was a second cap for cheap
work, which the plan rules out and which would have meant two mechanisms to reason
about.

## Not changed

Scoring, the covariance seed, the secondary screen (SCRUM-456/457), the clear
contract, and the dashboard. The window pull for the fallbacks is unbounded exactly
as SCRUM-459 left it.

## Tests

    python3 -m pytest services/planner services/ui/tests    1843 passed, 2 skipped

New `test_evaluate_cdm_id_resolve.py`, 24 cases: the resolve makes one narrow search
and zero window searches; the bracket is centred on the event's own TCA; our asset is
`object1` under either summary ordering; an unresolvable, wrong-asset, bad-TCA or
guard-failing id returns None rather than the worst conjunction; the parsed
conjunction and the end-to-end decision match the window path; each selector routes
to the seam it should; and six concurrency cases driven through the async transport,
timed from before the evaluate starts.

The concurrency cases were verified to fail when the offload is reverted — 4 of 6,
the two counter-leak ones being independent of it — so they are testing the thing
they claim to.
