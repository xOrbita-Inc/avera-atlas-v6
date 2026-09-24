# SCRUM-450 implementation plan: cache the LeoLabs conjunction window (short TTL)

Ticket: https://xorbita.atlassian.net/browse/SCRUM-450
Branch off main. Backend only: services/planner/common/leolabs_runtime.py plus a
test. No UI change.

## What this is

Every live evaluate re-fetches the whole LeoLabs window (~1,850 CDMs, ~13 to 20 s)
before selecting the one CDM the row names. SCRUM-449 made selection correct; this
makes it fast. Add a short-TTL, process-level cache of the raw CDM set so repeated
pulls within the TTL hit the cache instead of re-fetching.

## The one fetch to cache

client.search_conjunction_cdms is the ~16 to 20 s call, and it happens in exactly
two places in leolabs_runtime.py:

- line 259, in _scorable_in_risk_order (the evaluate path, fetched without a volume
  filter)
- line 493, in _raw_cdms_in_risk_order (the listing and globe path, fetched with the
  reporting volume filter)

Both should route their search_conjunction_cdms call through one cached helper.

## The key is the requested window, not the resolved timestamps

conjunction_window derives min_tca and max_tca from the wall clock on every call, so
two calls seconds apart carry slightly different absolute timestamps. Keying on those
would make every call a miss. Key instead on the stable query intent: catalog (or
primary NORAD), lookback_days, lookahead_days, and the volume filters. This is the
same reasoning _query_fingerprint already documents for the paging cursor, which
binds to the requested window in days rather than the resolved minTca..maxTca. Within
the TTL the absolute window shifts by a few seconds and the cached set is served
anyway, which is acceptable for a live feed on this horizon.

Because the evaluate path fetches with no volume filter and the listing fetches with
the reporting volume, the filters are part of the key, so the two are separate cache
entries and never cross. The win is still real: the first selection after a fetch is
a miss (~13 to 20 s), and every following row selection within the TTL is a hit.

## Seam

Add a module-level cache and helper in leolabs_runtime.py, beside the existing
_client/_registry singletons and _lock:

- _CDM_CACHE_TTL_SECONDS, a module constant (start at 45).
- _cdm_cache: Dict[str, Tuple[float, List[Dict]]], mapping a fingerprint to
  (monotonic timestamp, cdms).
- _search_cdms_cached(client, catalog, min_tca, max_tca, volume_filters, *,
  lookback_days, lookahead_days): under _lock, build the key from catalog +
  lookback_days + lookahead_days + volume_filters; return the cached list if
  now - ts < TTL; otherwise call client.search_conjunction_cdms(...) once, store
  (time.monotonic(), result), and return it. Use time.monotonic, matching the
  existing probe throttle already in this module.
- reset_caches() also clears _cdm_cache.

Route both call sites (259 and 493) through _search_cdms_cached, passing the volume
filters each already uses (empty for the evaluate path, the reporting volume for the
listing path).

## Do not

- Do not change what is fetched or parsed, the risk ordering, the dedupe, or the
  200/404/422/503 contract. Feed-off stays 503.
- Do not cache parsed results or the ordered set. Cache the raw CDM list so both
  consumers share one fetch and each still parses its own page.
- Do not key on the absolute min_tca/max_tca.

## Tests (extend services/planner/tests/test_leolabs_runtime.py)

Use a fake client whose search_conjunction_cdms counts calls, and call reset_caches()
in setup so entries do not leak between tests.

- A second fetch for the same asset and window within the TTL calls the client once,
  and returns the same CDMs.
- A different asset, a different window in days, or a different volume filter is a
  miss and calls the client again.
- After the TTL, a fetch re-calls the client. Drive this by monkeypatching
  time.monotonic (or _CDM_CACHE_TTL_SECONDS) rather than sleeping.
- reset_caches() empties the cache.
- The cached path returns exactly what the uncached path returned, so nothing
  downstream changes.

Run python3 -m pytest services/planner and confirm green (currently 1456 passed /
2 skipped, plus the new cases).

## Verify

- Backend only, but rebuild the planner container to run it:
  docker compose up -d --build planner.
- On the running stack, Live Asset SWARM C, fetch conjunctions, then select several
  rows one after another. The first selection still takes ~13 to 20 s; each following
  selection within the TTL should return in well under that. Cowork will measure this
  before and after in its review.

## Commit

Branch scrum-450-cache-conjunction-window off main. Include docs/scrum-450/ in the
commit. No attribution trailers, author John Avera <javera@xorbita.com>. PR against
main; report the PR number and head SHA.

Cowork runs the verification-first review before John merges, measures the selection
latency on the rebuilt stack, and gives the verdict to post on the PR.
