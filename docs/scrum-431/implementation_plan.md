# SCRUM-431 implementation plan

Retire the Space-Track code path, and disable-and-defer the secondary
conflict screen so it no longer runs, fails, or blocks a decision, while
keeping the SCRUM-381 screen logic intact for a later LeoLabs-backed rebuild.

## Why this shape (read first)

Space-Track is retired as a data source. The only live consumer of Space-Track
data left in the planner is the secondary conflict screen (SCRUM-330/381),
which fetches the LEO TLE catalog through spacetrack_tle.py on every evaluate.
A TLE catalog carries no covariance, so that screen has only ever produced an
assumed-covariance result, and with no catalog available it fails closed and
shows VERIFICATION FAILED. The covariance-true rebuild of that screen belongs
on LeoLabs data and is a separate post-Disrupt story (see the new LeoLabs
secondary-screen ticket). This ticket removes the Space-Track code and takes
the screen out of the live path cleanly, so nothing dormant references a
revoked credential and the decision path does not fail on a check that is out
of scope for this build.

Decision made with John: disable-and-defer, not delete. Keep the screen logic
(secondary_horizon.py, guard_secondary_clear, SecondaryConflictCheck, the
screen orchestration). Add an explicit disabled state, default off, so the
guard is skipped rather than failed, verification does not count it, and
staging is not blocked. When the LeoLabs screen ships, the flag flips on and
the fail-closed semantics return exactly as they are today.

## Part A. Retire the ingest Space-Track CDM poll

- services/ingest/main.py: remove the POST /cdm/poll endpoint and its handler,
  the `from spacetrack_client import SpaceTrackClient` import, the
  SPACETRACK_LIVE_POLLING_ENABLED read, and the SpaceTrackClient construction.
  Remove PollRequest/PollResponse only if nothing else uses them (the
  always-active reference/inject path around the former line 416 returns the
  same shape; keep that path working, keep the store write).
- Simplify GET /cdm/source_mode so it reports the reference-CDM path only and
  no longer advertises a live Space-Track tier or the polling flag.
- services/ingest/spacetrack_client.py: remove the file.
- services/ingest/db.py: keep the store schema and the stored-CDM read path.
  Change the source label a stored real CDM reports from "spacetrack" /
  "space_track" to "reference_cdm" so the UI stops showing a Space-Track
  origin (this folds in the naming leftover noted on SCRUM-419). Do not rename
  the pc_space_track column; a column migration is out of scope here, leave the
  column name as-is and only change the outward source label.
- services/ui/app/main.py: remove the /api/ingest/poll proxy route.

## Part B. Disable-and-defer the secondary screen

Add an explicit deferred state rather than leaning on not-performed.

- Config: SECONDARY_SCREEN_ENABLED, default "false", read in
  services/planner/server.py alongside the other env reads.
- SecondaryConflictCheck (services/planner/common/atlas_artifact.py): add
  `screen_deferred: bool = False`. A deferred instance has
  secondary_check_performed=False, secondary_conjunction_clear=False,
  screen_deferred=True, flagged_objects=[], and
  operator_note="Secondary screen deferred, LeoLabs covariance integration
  pending."
- build_atlas_artifact: add a parameter `secondary_screen_enabled: bool =
  False`. When False, construct the deferred SecondaryConflictCheck directly
  and skip _run_secondary_conflict_check entirely (do not fetch or require a
  catalog). When True, keep today's behavior.
- Verification builder _build_verification_result: when
  secondary.screen_deferred is True, do not append a failure. Surface the
  deferred operator_note as a non-failing informational line so the panel
  shows the neutral text and the pass/fail tally excludes the screen.
- GuardInputs (services/planner/common/safety_monitor.py): add
  `secondary_screen_deferred: bool = False`.
- monitor_adapter.py _secondary_check: read the deferred flag off the artifact
  and carry it into GuardInputs (extend the return or set the field), so the
  monitor knows the screen was deferred rather than merely not performed.
- m1_to_m2_guards (safety_monitor.py ~734-749): include guard_secondary_clear
  only when not inputs.secondary_screen_deferred. When deferred, omit it so it
  cannot block staging.
- _m1_escalation_guards (safety_monitor.py ~788): already escalates only when
  the check was performed and not clear, so a deferred screen is inert here.
  Add a one-line comment noting the deferred case is intentionally inert.

Net effect with the flag off: no catalog fetch, the A4 line and verification
read "Secondary screen deferred, LeoLabs covariance integration pending", the
screen never fails verification, and it never blocks staging. With the flag on,
behavior is identical to today.

## Part C. Remove the Space-Track TLE catalog fetch

- services/planner/common/spacetrack_tle.py: remove the file. It is the
  Space-Track-specific catalog fetch and the last code using SPACETRACK_USER /
  SPACETRACK_PASS in the planner, so it must go with the credential revoke
  (SCRUM-430).
- services/planner/server.py: remove `from common.spacetrack_tle import
  fetch_catalog_objects` and the known_objects fetch block in the evaluate
  flow. Keep the r_post / v_post computation, which the A4 post-maneuver
  projection still uses.
- Note on UDL: spacetrack_tle.fetch_catalog_objects held the only UDL-elset
  branch that fed the secondary catalog. Removing this file removes that
  branch too. udl_client.py itself is untouched and out of scope. Re-sourcing
  the secondary catalog, from LeoLabs or UDL, is the separate LeoLabs
  secondary-screen story.
- Tests: trim the fetch_catalog_objects cases in
  services/planner/tests/test_secondary_conflict.py (the missing-credentials,
  network-failure, and bad-epoch cases that exercise the Space-Track fetch).
  Keep the screen-logic tests (screen_secondary_catalog and the horizon
  screen). Fix any docstring or comment references to
  spacetrack_tle._parse_and_propagate_tle in test_udl_client.py and
  udl_client.py. After removal, grep the planner for spacetrack_tle and
  confirm no remaining import.

## Part D. Config and deployment

- docker-compose.yaml: remove SPACETRACK_USER and SPACETRACK_PASS from the
  ingest service env.
- docker-compose.prod.yaml: remove the SPACETRACK_USER / SPACETRACK_PASS lines
  and the SPACETRACK_* comment header.
- k8s/01-ingest.yaml: remove the spacetrack-credentials secret refs and the
  header comment.
- Prod .env on the KVM: stripping SPACETRACK_* there is SCRUM-430. Coordinate
  so the ingest service is recreated after both land.

## Keep intact

The CDM store and its schema (ADR-008), the inject / TIROS-4 reference path,
the decision log and audit records, the secondary screen logic modules
(secondary_horizon.py, guard_secondary_clear, screen_secondary_catalog,
SecondaryConflictCheck), and udl_client.py.

## Acceptance

- No Space-Track CDM poll code and no SPACETRACK_* config remain in the ingest
  service, compose, or k8s. spacetrack_tle.py is gone and nothing imports it.
- With SECONDARY_SCREEN_ENABLED off (the default), evaluate runs no catalog
  fetch, the secondary line reads the deferred text, verification does not fail
  on it, and an otherwise-clean maneuver stages and clears through the MAF path.
- With SECONDARY_SCREEN_ENABLED on, the screen behaves exactly as today
  (fail-closed on a missing or not-clear catalog).
- The store, inject, and audit paths still work. A stored real CDM reports
  source "reference_cdm", not a Space-Track origin.
- Planner suite green. Add a test that with the screen disabled the deferred
  state is produced, the secondary guard is absent from the staging set, and
  staging is not blocked by it; and a test that with it enabled the old
  fail-closed behavior holds.

## Verification hints

- Prove the deferred state does not block staging: an input that would
  otherwise stage to M2 still stages with the screen disabled, and the
  secondary guard is not in the guard set.
- Prove the enabled path still fails closed: with the flag on and no catalog,
  the screen is not performed and staging is blocked, as today.
- Browser-check the dashboard: the A4 secondary line and the verification panel
  read the neutral deferred text, no VERIFICATION FAILED from the secondary
  check, and a scenario maneuver clears end to end with zero console errors.
