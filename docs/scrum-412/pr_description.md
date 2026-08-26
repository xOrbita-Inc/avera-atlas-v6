# SCRUM-412: Wire the LeoLabs live feed into the runtime evaluate and UI flow

Follow-on to SCRUM-411. SCRUM-411 delivered the LeoLabs client, CCSDS parser,
asset-identity map, evaluate request-assembly glue, the per-evaluate source field
and LIVE badge, and a live-validated end-to-end run — but only tests and the
opt-in smoke called them. This PR makes the running planner and dashboard
actually use the LeoLabs feed.

**Stacks on the SCRUM-411 branch** (`scrum-411-leolabs-client-parser`); merge
SCRUM-411 first, or merge them together.

## What changed

- **`services/planner/common/leolabs_runtime.py`** (new): the runtime bridge.
  - `LEOLABS_ENABLED` feature flag (default false), mirroring `UDL_ENABLED`.
  - Process-cached client and asset registry — the registry's
    `list_subscribed_objects` call happens once, not per evaluate, so the org-wide
    rate limit is respected. `reset_caches()` for tests.
  - `fetch_leolabs_conjunction(primary_norad)`: maps NORAD → LeoLabs catalog via
    the registry, searches CDMs filtered to the LeoLabs source in a
    now..now+7d window, and returns the highest-risk CDM that parses and passes
    the parser's guards (guard-failing CDMs are skipped, not fatal). Returns None
    when nothing is scorable; raises for an unsubscribed primary.
  - Throttled credential/status probe (`get_status`), mirroring `udl_client`.
  - Optional `LEOLABS_ASSET_NORADS` allow-list to restrict the registry to our
    fleet.
- **`services/planner/server.py`**:
  - A LeoLabs branch in `/v1/evaluate` **before** the UDL branch (precedence):
    fetch → resolve our object → parse → rebuild the request from the parsed CDM
    via `build_evaluate_request` (real per-object covariance rotated to ECI,
    `source=leolabs`, `covariance_source=real_cdm`). Operator inputs (sat_id,
    v_remaining, burn time, policy) are carried from the incoming request.
  - No scorable CDMs → `no_maneuver_needed`; fetch failure → 503; missing
    `primary_norad` → 422. The covariance section keeps the real matrix and takes
    precedence over UDL.
  - New `/leolabs-status` endpoint (undocumented in OpenAPI, matching
    `/udl-status`).
- **UI**: `/api/planner/leolabs-status` proxy in `main.py`; `loadSourceMode`
  checks LeoLabs first for the idle badge. The per-evaluate LIVE badge from
  SCRUM-411 AC6 already handles the live source once an evaluate returns
  `source=leolabs`.
- **`services/planner/common/leolabs_client.py`**: `ping()` for the status probe.
- **`openapi/planner.yaml`** and `SERVICE_VERSION` → 2.5.7 (runtime feature +
  endpoint; no request/response schema change — the `source` field from 2.5.6
  carries the result).

## Acceptance criteria

1. LEOLABS_ENABLED=true → evaluate returns `source=leolabs`,
   `covariance_source=real_cdm`, on real per-object covariance. ✅ (offline via
   mocked fetch; live-provable with the flag + keys)
2. LEOLABS_ENABLED=false → runtime unchanged. ✅
3. LeoLabs precedence over UDL when both flags set. ✅ (UDL never consulted)
4. `/leolabs-status` reflects enabled/credential/mode; dashboard badge lights. ✅
5. Unit tests cover the runtime path with a mocked client; registry cached, limiter
   respected. ✅ Full suite green.
6. Degrades cleanly (no-record → no-maneuver; failure → 503; unsubscribed primary
   raises) rather than erroring. ✅

## Tests

`test_leolabs_runtime.py` (14): fetch returns/None/skips-guard-failing/
unsubscribed-raises, registry built once, evaluate-uses-leolabs, no-record,
requires-primary-norad (422), fetch-failure (503), **precedence over UDL**,
disabled path unchanged, status disabled/misconfigured/live.

Full suite (planner + aps_math + ingest): 929 passed, 2 skipped (opt-in live
smoke). No regressions.

## Known limitations / follow-ups

- The client's rate limiter is per-process, not truly org-wide across replicas. A
  multi-replica deployment would need a shared token store (Redis or similar) —
  flagged in the ticket as a further follow-on.
- Screening request-body field names and status vocabulary remain "confirm against
  live API"; the client's `wait_for_screening` predicate is injectable for that.
- The trial covers ten ILRS sats through 19 September; the runtime degrades cleanly
  on trial expiry (fetch failure → 503, no CDMs → no-maneuver).
