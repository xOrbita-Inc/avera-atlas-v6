# SCRUM-450 live verification — short-TTL conjunction-window cache

Run 2026-09-24 against the real LeoLabs API with the planner container rebuilt
from this branch. Asset: SWARM C (39453). Backend only; the UI container was not
rebuilt and needed no change.

## Latency, measured

**Listing path** (`GET /v1/leolabs/conjunctions`, reporting volume applied):

| call | time | total |
|---|---|---|
| 1 (cold) | **13.784 s** | 103 |
| 2 | **0.004 s** | 103 |
| 3 | **0.004 s** | 103 |

**Evaluate path** (`POST /v1/evaluate` with a `cdm_id` selector — what selecting a
row actually does, and the thing the ticket is about):

| selection | time | result |
|---|---|---|
| 1 (cold) | **8.423 s** | ok |
| 2 | **0.188 s** | ok |
| 3 | **0.195 s** | ok |

So the first selection after a window is pulled still pays the full fetch, and
every following selection inside the TTL returns in well under a second. The
`total` is identical across cold and warm calls, which is the point: the cache
changes how often the fetch happens, not what it returns.

## The TTL really expires

Waiting past 45 s and re-requesting:

    after TTL   8.625 s   total=103     (re-fetched)
    immediately 0.005 s                 (warm again)

So the staleness is bounded by the TTL rather than pinned for the life of the
process.

## The two paths never share an entry

The planner's own cache logging shows it directly — the evaluate path fetches
unfiltered and the listing path fetches with the reporting volume, and they cache
different sets:

    leolabs_cdm_cache_miss  catalog L3969  cdms 812  in_volume true
    leolabs_cdm_cache_hit   catalog L3969  cdms 812  in_volume true   age 0.06 s
    leolabs_cdm_cache_miss  catalog L3969  cdms 855  in_volume false
    leolabs_cdm_cache_hit   catalog L3969  cdms 855  in_volume false  age 0.26 s

812 vs 855 CDMs is the volume filter doing its job, and each is served only to
the path that asked for it. Over the session: 5 hits, 3 misses.

## Contract unchanged

| case | expected | got |
|---|---|---|
| valid listing | 200 | 200 |
| NORAD 99999, unsubscribed | 404 | 404 |
| lookahead_days=60 | 422 | 422 |
| `LEOLABS_ENABLED=false`, conjunctions | 503 | 503 |
| `LEOLABS_ENABLED=false`, orbits | 503 | 503 |

Feed-off was re-checked against a real container started with the flag off. The
flag is read before any fetch, so the cache cannot reach it, but it is the
contract the repo cares most about and it was cheap to confirm rather than
assume.

## Suite

    python3 -m pytest services/planner    1469 passed, 2 skipped   (+11)

The new cases count client calls rather than timing anything: a hit inside the
TTL fetches once, a changed asset / window-in-days / volume filter is a miss,
expiry is driven by monkeypatching `time.monotonic` rather than sleeping,
`reset_caches()` empties the cache, and the cached result is asserted equal to
the uncached one.

## Notes and limits

- **Test isolation now depends on `reset_caches()`.** The three test files that
  reach a real fetch (`test_leolabs_runtime`, `test_leolabs_conjunction_list`,
  `test_leolabs_globe_orbits`) all already call it in an autouse fixture, so this
  was checked rather than assumed. The other files that touch LeoLabs patch at
  the `server` level and never reach the cache.
- **The cached list is returned, not copied.** Every caller sorts into a new list
  and treats it as read-only; copying ~1,850 dicts per call would give back part
  of what the cache is for. A future caller that mutates must copy first — noted
  in the docstring.
- **Two callers racing a cold key may both fetch.** The lock is not held across
  the 13-to-20 s call, deliberately: holding it would serialise every caller,
  including ones asking for a different asset. Both then store the same thing, so
  the result is correct and the waste is bounded.
- **45 s is a starting point, not a tuned value.** It was chosen to cover an
  operator working through a list of rows while staying well inside the cadence
  at which LeoLabs refines CDMs. Nothing here measures the right number.
- One asset, one session. The cache is per-process, so a multi-worker or
  multi-pod planner gets one window per worker, not one shared window.
