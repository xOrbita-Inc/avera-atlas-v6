# SCRUM-411: LeoLabs Pulse client and CDM parser, with evaluate wiring and live validation

Stands up the live conjunction data path for the planner using LeoLabs Pulse:
a thin rate-limited API client, a CCSDS CDM parser producing real per-object
covariance in ECI, the `/v1/evaluate` wiring and source badge, the asset-identity
mapping, and an end-to-end run validated against the live API.

Built to the design doc (SCRUM-411_LeoLabs_Client_and_Parser_Design). Reuses the
committed shared math (`aps_math.frames.rtn_to_eci_rotation`, `aps_math.pc_utils`)
— no second copy of any rotation or Pc math.

## Commits

- `add LeoLabs Pulse client and CCSDS CDM parser` — client + parser + tests + fixture
- `wire per-evaluate source field and LIVE badge` — `source` on /v1/evaluate, badge, OpenAPI 2.5.6
- `assemble evaluate request from parsed LeoLabs CDM (AC7)` — request glue + covariance bypass
- `map asset identity to LeoLabs catalog numbers (AC5)` — asset registry + CDM object resolution

## What's included

- **Client** (`services/planner/common/leolabs_client.py`): literal `basic
  <access>:<secret>` auth from `LEOLABS_ACCESS_KEY`/`LEOLABS_SECRET_KEY` (never
  logged); shared 4 req/s limiter + 3-per-2-min screening limiter; backoff with
  jitter honoring Retry-After; `nextToken` pagination; CDM search, object list,
  states, and screening create/poll/retrieve.
- **Parser** (`services/planner/common/leolabs_cdm_parser.py`): reads the
  `SAT1_`/`SAT2_` flattened CDM, resolves primary/secondary by catalog id (no
  SAT1 assumption), rotates each object's RTN 6×6 to ECI with its own state,
  guards on CALCULATED / EME2000 / symmetric+PSD / diagonal-vs-comment /
  miss-distance, and emits the existing ConjunctionState contract plus provenance.
- **Asset map** (`services/planner/common/leolabs_asset_map.py`): NORAD ↔ LeoLabs
  catalog registry built once from `list_subscribed_objects`, and CDM object
  resolution.
- **Evaluate glue** (`services/planner/common/leolabs_evaluate.py`) + `server.py`:
  assembles the `/v1/evaluate` request from a parsed CDM (`source=leolabs`,
  `covariance_source=real_cdm`), and a bypass so the real covariance is not
  overwritten by the ingest adapter.
- **Source field + badge**: `source` on every `/v1/evaluate` response;
  `SourceEnum` and the field added to `EvaluateResponse` and `ConjunctionState`;
  the dashboard badge reflects the per-evaluate source (activating the dormant
  SCRUM-348 AC3 LIVE branch). `planner.yaml` and `SERVICE_VERSION` bumped to 2.5.6.

## Acceptance criteria

All 8 satisfied. See `docs/scrum-411/live_validation.md` for the live run.

- AC1–2 client/auth/limits/retrieval; AC3 parser + frame guard; AC4
  CALCULATED-vs-DEFAULT guard + provenance; AC5 identity map + resolution; AC6
  source field + badge + versioned OpenAPI; AC7 live end-to-end evaluate on real
  covariance (validated 2026-08-26); AC8 full suite green.

## Golden checks

Both pass on the fixture and on live data:
- Rotated ECI diagonal vs the CDM's EME2000 comments — fixture max rel err ~1e-6.
- `pc_utils` Pc vs CDM `COLLISION_PROBABILITY` — fixture 0.016%; live `1.816016e-219`
  vs `1.8161e-219` (~4.6e-5).

## Tests

Full suite: 868 passed, 2 skipped (opt-in live smoke, which passed live).
New: `test_leolabs_cdm_parser`, `test_leolabs_client`, `test_leolabs_asset_map`,
`test_leolabs_evaluate_source`, `test_leolabs_evaluate_e2e`,
`test_leolabs_live_smoke` (opt-in, gated by keys + `LEOLABS_LIVE_SMOKE=1`, so CI
never needs a credential).

## Deviations / decisions

- Modules live under `services/planner/common/` (mirrors `udl_client.py`); the
  design left the service open.
- `pc_precomputed` is left `None` in the parsed record so the planner computes Pc
  from real geometry; the CDM Pc is kept in provenance as a cross-check (design
  6.1/6.5).
- Operational Pc uses the ADR-010 ~15 m combined-HBR floor, not the CDM's
  combined radius; the parser's cross-check uses the CDM radius.
- The client's org-wide limiter is per-process; a true org-wide budget would need
  a shared token store (documented in the module).
- Screening request-body field names and status vocabulary are marked "confirm
  against live API"; the `wait_for_screening` completion predicate is injectable.

## Not included

- Wiring the LeoLabs registry/fetch into the non-test evaluate/UI runtime flow
  (the glue and bypass exist and are exercised by the opt-in smoke).
