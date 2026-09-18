# SCRUM-431 kickoff for Claude Code

Retire the Space-Track code path and disable-and-defer the secondary conflict
screen. Full spec in docs/scrum-431/implementation_plan.md. Read it first.

Branch: scrum-431-retire-spacetrack off current main.

Do:
1. Remove the ingest Space-Track poll: the POST /cdm/poll endpoint, the
   SpaceTrackClient import and use, the SPACETRACK_LIVE_POLLING_ENABLED flag,
   and simplify /cdm/source_mode to reference-CDM only. Delete
   services/ingest/spacetrack_client.py. Keep the store and the inject path.
   Change the stored-CDM outward source label from spacetrack/space_track to
   reference_cdm; do not rename the pc_space_track column.
2. Remove the ui /api/ingest/poll proxy (services/ui/app/main.py).
3. Disable-and-defer the secondary screen behind SECONDARY_SCREEN_ENABLED
   (default off). Add screen_deferred to SecondaryConflictCheck and
   secondary_screen_deferred to GuardInputs, give build_atlas_artifact a
   secondary_screen_enabled parameter that builds the deferred check and skips
   the fetch when off, skip guard_secondary_clear in m1_to_m2_guards when
   deferred, and make verification treat the deferred state as a non-failing
   informational line reading "Secondary screen deferred, LeoLabs covariance
   integration pending." Keep the enabled path byte-identical to today.
4. Delete services/planner/common/spacetrack_tle.py and its import and call in
   server.py. Trim the fetch_catalog_objects tests in test_secondary_conflict.py,
   keep the screen-logic tests, and fix the spacetrack_tle references in
   udl_client.py and test_udl_client.py. Grep to confirm no remaining import.
5. Remove SPACETRACK_* from docker-compose.yaml, docker-compose.prod.yaml, and
   k8s/01-ingest.yaml. Prod .env stripping is SCRUM-430, not this branch.
6. Run the planner suite from the repo root. Add the two tests named in the plan
   (deferred does not block staging; enabled still fails closed).

Do not: touch the CDM store schema, the TIROS/inject path, the decision log,
the secondary screen logic modules, or udl_client.py beyond the comment fix.
No AI attribution in commits or the PR. Do not merge; open the PR for John.

When done, report: files changed with line counts, the suite result on the
branch, and confirmation that with the flag off a scenario maneuver clears with
the secondary line showing the deferred text and no failed verification.
