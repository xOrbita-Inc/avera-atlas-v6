# SCRUM-450 kickoff for Claude Code

Cache the LeoLabs conjunction window with a short TTL so row selection is fast after
the first fetch. Ticket: https://xorbita.atlassian.net/browse/SCRUM-450

Branch scrum-450-cache-conjunction-window off main. Backend only,
services/planner/common/leolabs_runtime.py plus a test. No UI change.

Read docs/scrum-450/implementation_plan.md first; it names the two fetch call sites
and the exact cache key.

Run in order:

1. Read docs/scrum-450/implementation_plan.md.
2. Add a short-TTL cache in leolabs_runtime.py beside the existing _client/_registry
   singletons and _lock: a module constant _CDM_CACHE_TTL_SECONDS (start at 45), a
   _cdm_cache dict keyed by catalog + lookback_days + lookahead_days + volume_filters,
   and a _search_cdms_cached helper that returns the cached CDM list within the TTL or
   calls client.search_conjunction_cdms once and stores it, using time.monotonic under
   _lock. reset_caches() clears it.
3. Key on the requested window in days and the volume filters, not on the resolved
   min_tca/max_tca, so the live feed's clock drift does not miss every time. The plan
   explains why; it mirrors _query_fingerprint.
4. Route both search_conjunction_cdms call sites through the helper: line 259 in
   _scorable_in_risk_order (evaluate path, no volume filter) and line 493 in
   _raw_cdms_in_risk_order (listing and globe path, with the reporting volume). Do not
   change what is fetched, parsed, ordered or deduped, and keep the 200/404/422/503
   contract and feed-off 503.
5. Extend services/planner/tests/test_leolabs_runtime.py: a second call within the TTL
   fetches once, a changed asset/window/filter is a miss, expiry after the TTL
   re-fetches (monkeypatch time.monotonic, do not sleep), reset_caches empties it, and
   the cached result equals the uncached one. Run python3 -m pytest services/planner
   and confirm green.
6. Rebuild the planner container (docker compose up -d --build planner) and check on
   the running stack with SWARM C that the first row selection still takes ~13 to 20 s
   and each following selection within the TTL returns much faster.
7. Commit on scrum-450-cache-conjunction-window, include docs/scrum-450/ in the
   commit, push, open a PR against main. No attribution trailers, author John Avera.
   Report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges, measuring the
selection latency on the rebuilt stack and giving the verdict to post on the PR.
