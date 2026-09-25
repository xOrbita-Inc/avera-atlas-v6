# SCRUM-455 implementation plan: make .env control the screen in both composes, fix the prod ui build context

Ticket: https://xorbita.atlassian.net/browse/SCRUM-455
Branch off main. Deploy and config hygiene, no feature change. Part of SCRUM-433.
Relates to SCRUM-442 and SCRUM-447.

## What this is

The 2026-09-24 KVM promotion exposed two things. The .env does not actually
control the secondary screen, and docker-compose.prod.yaml has a stale ui build
context that fails the prod ui build. This ticket makes .env authoritative for the
screen in both composes, flips the code default to fail safe, and fixes the prod
build context. The principle is that local and prod differ only by the compose
file, and both read the same .env.

Note the screen is now live: SCRUM-456 (async) and SCRUM-457 (dashboard poll) are
merged and the on-demand screen returns a real result on local. This ticket is not
about keeping the screen off; it is about making .env the single control, on both
local and prod, and unbreaking the prod ui build so a clean redeploy works.

## The current state, confirmed on main

- services/planner/server.py line 543 to 544:
  `SECONDARY_SCREEN_ENABLED = os.environ.get("SECONDARY_SCREEN_ENABLED", "true").lower() == "true"`.
  The default is true. There is also a stale comment near line 2237 that describes
  the screen as default off, which contradicts the code.
- docker-compose.yaml (dev) planner environment forwards INGEST, UDL_USER,
  UDL_PASS, UDL_ENABLED, LEOLABS_ENABLED, LEOLABS_ACCESS_KEY, LEOLABS_SECRET_KEY,
  LEOLABS_ASSET_NORADS. It does NOT forward SECONDARY_SCREEN_ENABLED.
- docker-compose.prod.yaml planner environment forwards INGEST, LEOLABS_ENABLED,
  LEOLABS_ACCESS_KEY, LEOLABS_SECRET_KEY, LEOLABS_ASSET_NORADS. It forwards neither
  SECONDARY_SCREEN_ENABLED nor the three UDL vars the dev compose has.
- docker-compose.prod.yaml ui is `build: {context: ./services/ui}`. The dev compose
  ui is already `build: {context: ., dockerfile: services/ui/Dockerfile}` from
  SCRUM-447. The prod one is stale and fails the ui build for lack of libs/aps_math.

## The change

1. Forward the var in both composes. Add
   `- SECONDARY_SCREEN_ENABLED=${SECONDARY_SCREEN_ENABLED}` to the planner
   environment in docker-compose.yaml and docker-compose.prod.yaml, so the .env
   line is what turns the screen on or off in local and prod alike. This is the
   correction to the earlier mis-scope: the dev compose is missing the var too, not
   only prod.

2. Flip the code default to false. In server.py change the default from "true" to
   "false", so a missing .env line fails safe rather than arming a load-bearing
   screen by accident. With the var forwarded and .env carrying it, the default is
   only a backstop. Reconcile the stale near-2237 comment so the code and the
   comment agree that the default is off and .env is authoritative.

3. Fix the prod ui build context. Change docker-compose.prod.yaml ui to
   `build: {context: ., dockerfile: services/ui/Dockerfile}` to match the dev
   compose and SCRUM-447.

4. Reconcile the two composes. Compare planner and ui across both files. The
   SECONDARY_SCREEN_ENABLED addition and the ui context fix are the load-bearing
   ones. The dev planner also carries UDL_USER, UDL_PASS, UDL_ENABLED that prod does
   not. Do not blindly copy them. Determine whether UDL is used on prod, and if it
   is, forward the three vars in prod too; if it is dev-only, leave prod without
   them and say so in the PR. The instruction is to report and fix real gaps, not to
   make the files textually identical. Image tags, ports and DATA_DIR are allowed to
   differ.

## The consequence to make explicit

After this change the screen is off unless .env sets SECONDARY_SCREEN_ENABLED=true.
The TCD demo needs it on, so both the local .env and the KVM .env must carry
SECONDARY_SCREEN_ENABLED=true. The deploy checklist must say this in the manual
steps, because a promotion with the line missing now yields a silent no-screen
rather than a screen-on-by-default.

## Do not

- Do not change the screen itself, the async work, the parser or the dashboard.
  This ticket moves no logic; it only makes configuration authoritative and fixes a
  build context.
- Do not weaken any fail-closed path. Off means the screen does not run and the
  secondary check reports not-performed, which the guard already fails closed on.
  Off is not a false clear.
- Do not make the two composes identical for its own sake. Preserve the intended
  differences.

## Tests

- A small guard test that parses both compose files and asserts the planner service
  forwards SECONDARY_SCREEN_ENABLED in each, so this regression cannot return
  silently. A plain YAML parse and a membership check is enough; no stack needed.
- A test that with SECONDARY_SCREEN_ENABLED unset in the environment the server
  reads the flag as false, and that the existing SCRUM-442 state-machine tests,
  which patch the flag explicitly, still pass with the new default.
- Run python3 -m pytest services/planner and confirm green.

## Local verification, the load-bearing proof

This is a local-provable ticket. On the local stack, with the dev compose:

- `docker compose config` resolves SECONDARY_SCREEN_ENABLED on the planner from
  .env (show the resolved planner environment).
- With SECONDARY_SCREEN_ENABLED=true in .env, rebuild the planner and confirm a
  maneuver-recommended evaluate arms the secondary screen (screen_pending with a
  job id, as SCRUM-456 and 457 show).
- With SECONDARY_SCREEN_ENABLED=false in .env, rebuild the planner and confirm the
  same evaluate runs no screen and the secondary check reports not-performed, and
  that this reads NOT CLEAR or not-checked in the dashboard, never CLEAR.
- Confirm the code default alone (no .env line) is now off.

For the prod build context, confirm locally that
`docker compose -f docker-compose.prod.yaml build ui` resolves the root context and
builds, rather than failing on the missing libs/aps_math. Building the prod ui image
locally proves the context fix without the KVM.

Record all of this in docs/scrum-455/live_verification.md.

## Prod, which is John's manual step

The real prod deploy runs on the KVM, which Claude Code and Cowork cannot reach from
here. Do not attempt it. This ticket lands the fix and proves it locally. The KVM
promotion that consumes it is the next manual step, gated on this PR and on the KVM
.env carrying SECONDARY_SCREEN_ENABLED=true.

## Deploy checklist

Update the repo deploy checklist (find it near scripts/deploy.sh or docs). Add that
.env must carry SECONDARY_SCREEN_ENABLED, that the default is now off, and that a
promotion must confirm the resolved planner environment shows the var. Include
docs/scrum-455/.

## Commit

Branch off main. Include docs/scrum-455/. No attribution trailers, author John
Avera <javera@xorbita.com>. PR against main; report the PR number and head SHA.

Then Cowork runs the verification-first review before John merges: it confirms both
composes forward the var, the default is now false and fails safe, the prod ui
context is fixed, the compose reconciliation is reported with a real decision on
UDL, and the local toggle proof holds.
