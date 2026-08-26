# SCRUM-411 — LeoLabs live validation results

AC7 asks for an end-to-end evaluate on a real LeoLabs CDM, with results recorded.
This is that record. The run hit the live LeoLabs API against the trial account
(ten ILRS satellites) using CRYOSAT 2 (LeoLabs `L2669`, NORAD 36508) as the
subscribed asset.

## Run

- Date: 2026-08-26 (04:27 UTC)
- Command:
  ```
  export LEOLABS_ACCESS_KEY=... LEOLABS_SECRET_KEY=...
  export LEOLABS_LIVE_SMOKE=1
  export INGEST_SERVICE_URL=http://localhost:8001   # so the audit write lands
  python3 -m pytest services/planner/tests/test_leolabs_live_smoke.py -v -s
  ```
- Result: `2 passed in 34.06s`. Python 3.11.9.

## Golden check on live data (parser)

Live CDM `76633250244` for `L2669`, a benign conjunction (Pc ~1e-219):

| Quantity | Value |
| --- | --- |
| Our Pc (aps_math.pc_utils on parsed ECI states + rotated covariances) | `1.816016e-219` |
| CDM `COLLISION_PROBABILITY` (ALFANO-2005) | `1.8161e-219` |
| Relative error | ~`4.6e-5` (tolerance `5e-3`) |

This agrees to ~5 significant figures at a magnitude 200+ orders away from the
saved fixture's `4.3858e-13`, so units, frame, the per-object RTN→ECI rotation,
and the Pc math are all validated across an enormous dynamic range. (The rotated
ECI diagonal vs the CDM's EME2000 comment diagonals is checked inside the parser
on every parse, so it also held on this live CDM.)

## End-to-end evaluate on real covariance (AC7)

Live CDM event `3539915022` for `L2669`, driven through `/v1/evaluate`:

| Field | Value |
| --- | --- |
| `source` | `leolabs` |
| `covariance_source` | `real_cdm` |
| `recommendation.direction` | `no-burn` |
| Computed Pc (planner, 15 m HBR screening floor) | `1.55e-218` |
| Maneuver threshold | `1.0e-04` |
| Artifact summary | `NO ACTION — Pc 1.55e-218 (computed) is below maneuver threshold 1.0e-04. No maneuver warranted.` |
| HTTP status / latency | `200` / ~23 ms |
| Audit | evidence record appended, `record_id L2669#0` (ingest up) |

Notes:
- The planner's own Pc (`1.55e-218`) is computed from the real per-object
  covariance but with the ADR-010 combined hard-body-radius floor (~15 m), which
  differs from the CDM's own combined radius; this is by design. The parser's
  cross-check above uses the CDM's combined radius, which is why it matches the
  CDM Pc while the operational Pc does not have to.
- The Space-Track secondary-conflict screening reported `not_performed` (no
  `SPACETRACK_USER`/`SPACETRACK_PASS` in this run). It is fire-and-forget and does
  not affect the recommendation.
- The covariance-adapter bypass held: the real LeoLabs covariance reached the
  scorer unmodified rather than being overwritten by an ingest fetch.

## Acceptance criteria

All eight ACs are satisfied. Full offline suite: 868 passed, 2 skipped (the two
opt-in live-smoke tests, which passed in this live run).
