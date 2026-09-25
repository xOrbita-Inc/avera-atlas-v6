# SCRUM-455 local verification — .env is now the single control for the screen

Run 2026-09-25 on the local stack. No KVM access from here; the prod deploy is
John's manual step and this PR proves the fix locally.

## The current state, confirmed on main before changing anything

Checked against the files, not the plan's line numbers, and all three matched:

| claim | found |
|---|---|
| `server.py` default is `"true"` | line 543-544, confirmed |
| stale comment calling the screen default off | line 2237, confirmed |
| dev compose planner: no `SECONDARY_SCREEN_ENABLED` | confirmed (has INGEST, 3x UDL, 4x LEOLABS) |
| prod compose planner: no `SECONDARY_SCREEN_ENABLED`, no UDL | confirmed |
| prod compose ui: `build: {context: ./services/ui}` | confirmed |

And the root cause in one line: `.env` **already carried**
`SECONDARY_SCREEN_ENABLED=true`. It was simply never forwarded, so the screen ran
on the code default and the `.env` line did nothing.

## 1. `docker compose config` now resolves the var, in both composes

Dev, resolved planner environment (secrets redacted):

    INGEST_SERVICE_URL       = 'http://ingest:8000'
    LEOLABS_ACCESS_KEY       = '<redacted>'
    LEOLABS_ASSET_NORADS     = '36508,43476,43477,27944,41335,43437,7646,39452,39451,39453'
    LEOLABS_ENABLED          = 'true'
    LEOLABS_SECRET_KEY       = '<redacted>'
    SECONDARY_SCREEN_ENABLED = 'true'          <- new
    UDL_ENABLED              = ''
    UDL_PASS                 = '<redacted>'
    UDL_USER                 = ''

Prod, resolved planner environment:

    INGEST_SERVICE_URL       = 'http://ingest:8000'
    LEOLABS_ACCESS_KEY       = '<redacted>'
    LEOLABS_ASSET_NORADS     = '36508,43476,...'
    LEOLABS_ENABLED          = 'true'
    LEOLABS_SECRET_KEY       = '<redacted>'
    SECONDARY_SCREEN_ENABLED = 'true'          <- new

Prod, resolved ui build:

    {'context': '/Users/jhavera/avera-atlas-v6', 'dockerfile': 'services/ui/Dockerfile'}

## 2. The toggle, proved end to end on the running planner

**`.env SECONDARY_SCREEN_ENABLED=true`**, planner rebuilt:

    container env SECONDARY_SCREEN_ENABLED = 'true'
    server.SECONDARY_SCREEN_ENABLED        = True

    POST /v1/evaluate    http 200 in 16.3 s
      screen_pending              = True
      screen_job_id               = a28e800c36494f1392899655934a9858
      screen_deferred             = False
      secondary_check_performed   = False
      secondary_conjunction_clear = False
      note: On-demand secondary screen running in the background; the maneuver is
            provisional and is NOT authorized as secondary-clear until it resolves.

**`.env SECONDARY_SCREEN_ENABLED=false`**, planner recreated:

    container env = 'false'
    server flag   = False

    POST /v1/evaluate    http 200 in 15.5 s
      screen_pending              = False
      screen_job_id               = None
      screen_deferred             = True
      secondary_check_performed   = False
      secondary_conjunction_clear = False
      note: Secondary screen deferred, LeoLabs covariance integration pending.
      verification.secondary_clear = False
      decision_state_machine       M0 -> M1, escalated: False

    screen-start log lines in the window: 0

So off really means off: no screen started, no LeoLabs create spent, and the
secondary check reports the SCRUM-431 deferral.

**Off is never a false clear.** With the screen off,
`secondary_conjunction_clear` is `False`, `verification.secondary_clear` is
`False`, and the dashboard row renders **DEFERRED** with a neutral class — not
green, not CLEAR. Being precise rather than overclaiming: the row shows DEFERRED,
not the words "NOT CLEAR". That is the pre-existing SCRUM-431 state and it is
honest — the screen was deliberately not run, as distinct from having run and
failed — and the SCRUM-457 harness asserts DEFERRED renders neutral rather than
green, so it cannot be mistaken for a pass. The staging guard is omitted rather
than passed, so nothing is certified secondary-clear on the strength of it.

`escalated: False` is also correct and deliberate: a deferral does not safehold,
which is the SCRUM-431 design, otherwise turning the screen off would M4 every
watch event.

## 3. The code default alone is off

`.env` line deleted entirely, planner recreated:

    container env = ''
    server flag   = False          <- code default alone

One honest detail worth recording rather than glossing: with the var absent from
`.env`, compose forwards `SECONDARY_SCREEN_ENABLED=''` — an **empty string**, not
an absent variable. So the flag is off because `'' != "true"`, not because the
`"false"` default was consulted. Both roads lead to off, which is the point, and
the genuinely-unset case (where the `"false"` default really is what decides) is
covered by the offline test that deletes the var from the environment before
reloading the module.

`.env` was restored byte-identically afterwards and the planner rebuilt; the stack
is back to `SECONDARY_SCREEN_ENABLED=true`.

## 4. The prod ui build context, proved both ways

**With the fix**, `docker compose -f docker-compose.prod.yaml build ui`:

    #12 exporting to image ... naming to docker.io/avera/ui:v6 done
    Image avera/ui:v6 Built

**With the old context**, the same Dockerfile built from `./services/ui` through a
throwaway compose file (created outside the repo, used, and removed):

    failed to solve: failed to compute cache key: failed to calculate checksum of
    ref ...: "/services/ui": not found

So the fix is load-bearing, not cosmetic. One correction to the plan's wording:
it predicted the failure would be "for lack of libs/aps_math". It actually fails
one line earlier, on `COPY services/ui/requirements.txt` — with the context set to
`services/ui`, that path does not exist *inside* the context. Same root cause, and
`COPY libs/aps_math/` would fail immediately after.

## 5. Compose reconciliation, and the UDL decision

A structural diff of both files, service by service, found exactly two differences
in planner and ui, plus two legitimately prod-only services:

| | |
|---|---|
| planner | dev has `UDL_USER`, `UDL_PASS`, `UDL_ENABLED`; prod does not |
| ui | the build context (fixed) |
| prod-only services | `dmui`, `voice` — intended, left alone |
| image tags, ports, `DATA_DIR` | identical across both |

**Decision on UDL: leave prod without the three vars.** Not a guess; four pieces
of evidence:

1. `UDL_ENABLED` defaults to **false** in `udl_client.py`, with the comment
   "default false until service account is confirmed and SSA agreement is in place
   for registered spacecraft CDM access". The account was never confirmed.
2. **`.env` carries no UDL vars at all** — not on this box, and there is nothing
   for prod to forward. The dev compose's three entries resolve to empty strings,
   which is the source of the `"UDL_USER" variable is not set` warnings on every
   compose command in this repo.
3. LeoLabs superseded it: `server.py` states LeoLabs "takes precedence over UDL:
   LeoLabs replaces the UDL route, which could not supply covariance."
4. Nothing but the dev planner references UDL in either compose.

So forwarding them to prod would add three empty variables and three warnings and
enable nothing. If UDL is ever commissioned, the credentials go into `.env` and the
vars into both composes at that point. The dev compose's entries are left as they
are: harmless, and removing them is a different (cosmetic) change.

I also checked whether any **other** service has the same stale-context bug, by
reading each Dockerfile for `libs/` rather than assuming. Only `ui`, `planner` and
`propagator` need the root context; `planner` and `propagator` already had it in
both files, and no other service copies `libs/`. A test now enforces this across
both composes by walking the Dockerfiles, so a new service that copies `libs/` and
forgets the context is caught here rather than on a prod build.

## 6. The deploy checklist

There was no checklist document in the repo — `Claude outputs/deploy-checklist-*.md`
is untracked, so it is not the repo's. Rather than add prose that can drift from
reality, the checklist went into `scripts/deploy.sh`, which is the tracked artifact
a deployer actually runs:

- a **MANUAL CHECKLIST BEFORE A PROMOTION** block in the header: `.env` must carry
  `SECONDARY_SCREEN_ENABLED`, the default is now off, how to check the resolved
  value by hand, and what a correct post-deploy evaluate looks like
  (`screen_pending` with a job id, not `screen_deferred`);
- an **executable pre-flight** beside the existing compose validation that prints
  the resolved value and warns when it is missing or not true. It warns rather than
  fails: turning the screen off is a legitimate choice, and a deploy should not be
  blocked by it. Not *knowing* is the thing that went wrong.

Exercised in all five states it can meet:

    prod compose, .env=true      [OK]   Secondary screen ON  (SECONDARY_SCREEN_ENABLED: "true")
    dev compose,  .env=true      [OK]   Secondary screen ON  (SECONDARY_SCREEN_ENABLED: "true")
    .env=false                   [WARN] Secondary screen OFF (SECONDARY_SCREEN_ENABLED: "false")
    .env line deleted            [WARN] Secondary screen OFF (SECONDARY_SCREEN_ENABLED: "")
    var absent from the compose   [WARN] SECONDARY_SCREEN_ENABLED is not in the
                                        planner's resolved environment.

The last row is the 2026-09-24 promotion exactly. The pre-flight would have caught
it.

## Not done, on purpose

No KVM deploy. The promotion is John's manual step and the box is out of reach
from here. It is gated on this PR and on the KVM `.env` carrying
`SECONDARY_SCREEN_ENABLED=true`.

No change to the screen, the async work, the parser or the dashboard. The only
behaviour change is the default, and it moves toward safe.

## Tests

    python3 -m pytest services/planner services/ui/tests    1781 passed, 2 skipped

New: `services/planner/tests/test_compose_screen_flag.py` — 6 cases. Both composes
forward the var; neither pins a literal value instead of interpolating from `.env`
(which would be the same bug in a different hat); the ui build context matches
across both and is the repo root; and no service that copies `libs/` builds from
anywhere else.

Updated: `test_the_flag_defaults_to_on` became `test_the_flag_defaults_to_off`,
with the three-step history in its docstring — SCRUM-431 off as a deferral,
SCRUM-442 on, SCRUM-455 off as a fail-safe backstop — so the next reader does not
mistake this for a regression of SCRUM-442. It also now unsets the variable
explicitly, so the assertion is about the code default rather than whatever the
developer's shell exports.
