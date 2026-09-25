#!/usr/bin/env bash
# =============================================================
# AVERA-ATLAS VPS Deployment Script  (SCRUM-355 rework)
# =============================================================
# Deploys origin/main to the production VPS by actually moving
# the branch pointer, then rebuilding and restarting containers.
#
# Usage:
#   ./scripts/deploy.sh              deploy all services
#   ./scripts/deploy.sh ui           rebuild/restart one service
#   ./scripts/deploy.sh ui planner   ...or several
#
# What changed vs the old script (the bugs SCRUM-355 fixes):
#   - Old step 1 was `git checkout origin/main -- services/`,
#     which overwrote the working tree but NEVER moved HEAD, so
#     the deployed SHA was unknowable and everything outside
#     services/ (openapi, README, scripts) silently drifted.
#     This version does `git pull --ff-only`, so the branch
#     actually advances and the whole tree is in sync.
#   - The old confirm prompt ran AFTER the checkout and aborted
#     with exit 0, so an aborted run reported success having
#     already mutated the tree. Here the confirm is before any
#     mutation and an abort changes nothing.
#   - Before/after SHAs are printed, so a no-op cannot hide.
#
# Requirements:
#   - Run from the repo root on the VPS
#   - docker-compose.prod.yaml must exist
#   - traefik-proxy Docker network must exist
#
# -------------------------------------------------------------
# MANUAL CHECKLIST BEFORE A PROMOTION  (SCRUM-455)
# -------------------------------------------------------------
# The .env on this box is the single control for feature flags.
# It is never in git, so a promotion can carry new code that
# reads a var the box's .env does not have.
#
#   1. .env must carry SECONDARY_SCREEN_ENABLED.
#
#      The on-demand secondary screen (the LeoLabs post-burn
#      conjunction screen) is OFF unless .env says:
#
#          SECONDARY_SCREEN_ENABLED=true
#
#      The code default is now false, deliberately: until
#      SCRUM-455 neither compose forwarded this var, so the
#      .env line did nothing and the screen ran on the code
#      default. Both composes forward it now, and the default
#      is only the backstop for a missing line.
#
#      The demo needs the screen ON. A promotion with the line
#      missing yields a silent no-screen, not a screen by
#      default. Off is safe -- no screen runs, the secondary
#      check reports deferred, and a deferral is never a clear
#      screen -- but it is a lost capability, so check it.
#
#   2. Step 5 below prints the resolved value and warns if it is
#      missing or empty. Read that line. To check by hand:
#
#          docker compose -f docker-compose.prod.yaml config \
#            | sed -n '/^  planner:/,/^  [a-z-]*:$/p' \
#            | grep SECONDARY_SCREEN_ENABLED
#
#   3. After the deploy, a maneuver-recommended evaluate should
#      come back with screen_pending true and a screen_job_id
#      (SCRUM-456/457). If it comes back screen_deferred, the
#      flag did not reach the container.
#
# DO NOT run this locally. VPS only.
# =============================================================

set -euo pipefail

COMPOSE_FILE="docker-compose.prod.yaml"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="origin"
BRANCH="main"

# The single working-tree file allowed to differ on the VPS.
# docker-compose.yaml is a deliberate production stub and is
# intentionally never overwritten by a deploy.
ALLOWED_DIRTY="docker-compose.yaml"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[DEPLOY]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*" >&2; exit 1; }

# =============================================================
# GUARDS
# =============================================================
cd "$REPO_DIR"

[[ -f "$COMPOSE_FILE" ]] || \
  fail "$COMPOSE_FILE not found. Run this from the repo root on the VPS."

git rev-parse --git-dir >/dev/null 2>&1 || \
  fail "$REPO_DIR is not a git repository."

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "$CURRENT_BRANCH" == "$BRANCH" ]] || \
  fail "On branch '$CURRENT_BRANCH', expected '$BRANCH'. Refusing to deploy from a detached HEAD or feature branch."

# =============================================================
# STEP 1 — Fetch and figure out where we are vs origin
# =============================================================
log "Fetching $REMOTE/$BRANCH..."
git fetch --quiet "$REMOTE" "$BRANCH"

BEFORE="$(git rev-parse HEAD)"
TARGET="$(git rev-parse "$REMOTE/$BRANCH")"

# =============================================================
# STEP 2 — Pre-flight safety checks (before touching anything)
# =============================================================

# 2a. Working tree must be clean except for the allowed compose stub.
#     Any other local change means the box is in an unexpected state.
DIRTY="$(git status --porcelain --untracked-files=no \
         | sed 's/^...//' \
         | grep -v -x "$ALLOWED_DIRTY" || true)"
if [[ -n "$DIRTY" ]]; then
  fail "Working tree has unexpected local changes:
$(printf '  %s\n' $DIRTY)
Only '$ALLOWED_DIRTY' may differ on the VPS. Resolve these before deploying
(commit and push them, or discard them) so nothing is silently clobbered."
fi

# 2b. A clean fast-forward must be possible. If local HEAD is not an
#     ancestor of origin/main, the VPS has commits that never made it
#     to origin, or histories diverged. Stop rather than clobber them.
if [[ "$BEFORE" != "$TARGET" ]] \
   && ! git merge-base --is-ancestor "$BEFORE" "$TARGET"; then
  fail "Local HEAD ($(git rev-parse --short "$BEFORE")) is not an ancestor of $REMOTE/$BRANCH ($(git rev-parse --short "$TARGET")).
The VPS has commits not on origin/main, or the histories diverged, so a
fast-forward is not possible. Investigate before deploying."
fi

# =============================================================
# STEP 3 — Show the plan and confirm BEFORE any mutation
# =============================================================
if [[ "$BEFORE" == "$TARGET" ]]; then
  log "Already at $REMOTE/$BRANCH ($(git rev-parse --short "$BEFORE")). No new code to pull."
  log "This will be a rebuild/restart only."
else
  log "Update available, will fast-forward:"
  log "  before: $(git rev-parse --short "$BEFORE")"
  log "  after:  $(git rev-parse --short "$TARGET")"
  echo   "  ------------------------------------------------"
  git --no-pager log --oneline "$BEFORE".."$TARGET" | sed 's/^/    /'
  echo   "  ------------------------------------------------"
fi

SERVICES=("$@")
if [[ ${#SERVICES[@]} -eq 0 ]]; then
  warn "No service specified. This will rebuild and restart ALL services."
else
  log "Target service(s): ${SERVICES[*]}"
fi

read -r -p "Proceed with deploy? [y/N] " CONFIRM
[[ "$CONFIRM" =~ ^[Yy]$ ]] || { log "Aborted. Nothing changed."; exit 0; }

# =============================================================
# STEP 4 — Move the branch (the actual fix)
# --ff-only guarantees no merge commit and no clobber; it can
# only advance HEAD to origin/main.
#
# The allowed compose stub is set aside first, because an
# upstream commit that edits docker-compose.yaml would otherwise
# block the fast-forward (git will not overwrite the local
# modification). The VPS copy is restored afterward: the VPS
# stub intentionally wins, and production runs from
# docker-compose.prod.yaml regardless, so upstream edits to the
# dev compose file are deliberately not adopted on the box.
# =============================================================
STUB_SAVED=0
if [[ "$BEFORE" != "$TARGET" ]]; then
  if [[ -n "$(git status --porcelain -- "$ALLOWED_DIRTY")" ]]; then
    log "Setting aside local $ALLOWED_DIRTY (VPS stub) across the pull..."
    cp -- "$ALLOWED_DIRTY" "$ALLOWED_DIRTY.deploybak"
    git checkout -- "$ALLOWED_DIRTY"
    STUB_SAVED=1
  fi

  log "Fast-forwarding $BRANCH to $REMOTE/$BRANCH..."
  git pull --ff-only "$REMOTE" "$BRANCH"

  if [[ "$STUB_SAVED" -eq 1 ]]; then
    mv -- "$ALLOWED_DIRTY.deploybak" "$ALLOWED_DIRTY"
    warn "Restored VPS $ALLOWED_DIRTY stub. Upstream changes to that file were not applied on the box (by design)."
  fi
fi
AFTER="$(git rev-parse HEAD)"

# =============================================================
# STEP 5 — Validate compose config
# Tolerates the two known-benign warnings (unset runtime vars
# resolved by the environment, and the external traefik network).
# =============================================================
log "Validating $COMPOSE_FILE..."
VAL_ERR="$(docker compose -f "$COMPOSE_FILE" config 2>&1 >/dev/null \
           | grep -v "variable is not set" \
           | grep -v "traefik-proxy" || true)"
[[ -z "$VAL_ERR" ]] || fail "Compose config validation failed:
$VAL_ERR"
ok "Compose config valid"

# SCRUM-455: report the resolved secondary-screen flag rather than leaving it to
# be discovered after the deploy. This warns and does not fail: turning the screen
# off is a legitimate choice, and a deploy should not be blocked by it. What is not
# legitimate is not knowing, which is exactly what happened on the 2026-09-24
# promotion -- the var was set in .env, neither compose forwarded it, and nothing
# said so.
SCREEN_LINE="$(docker compose -f "$COMPOSE_FILE" config 2>/dev/null \
               | sed -n '/^  planner:/,/^  [a-z-]*:$/p' \
               | grep 'SECONDARY_SCREEN_ENABLED' \
               | sed 's/^ *//' || true)"
if [[ -z "$SCREEN_LINE" ]]; then
  warn "SECONDARY_SCREEN_ENABLED is not in the planner's resolved environment.
  The on-demand secondary screen will be OFF. If that is not intended, add
  SECONDARY_SCREEN_ENABLED=true to .env on this box and redeploy the planner."
elif [[ "$SCREEN_LINE" == *'"true"'* || "$SCREEN_LINE" == *"'true'"* || "$SCREEN_LINE" == *": true"* ]]; then
  ok "Secondary screen ON  ($SCREEN_LINE)"
else
  warn "Secondary screen OFF ($SCREEN_LINE).
  The demo needs it ON. Set SECONDARY_SCREEN_ENABLED=true in .env if that is wrong."
fi

# =============================================================
# STEP 6 — Build and deploy against the production compose file
# =============================================================
if [[ ${#SERVICES[@]} -eq 0 ]]; then
  log "Building all services..."
  docker compose -f "$COMPOSE_FILE" build
  log "Restarting all services..."
  docker compose -f "$COMPOSE_FILE" up -d
  ok "All services deployed"
else
  for SERVICE in "${SERVICES[@]}"; do
    log "Building $SERVICE..."
    docker compose -f "$COMPOSE_FILE" build "$SERVICE"
    log "Restarting $SERVICE..."
    docker compose -f "$COMPOSE_FILE" up -d "$SERVICE"
    ok "$SERVICE deployed"
  done
fi

# =============================================================
# STEP 7 — Report, with before/after SHAs so a no-op is obvious
# =============================================================
echo ""
log "Container status:"
docker compose -f "$COMPOSE_FILE" ps \
  --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

echo ""
ok "Deployment complete"
log "  code before: $(git rev-parse --short "$BEFORE")"
log "  code after:  $(git rev-parse --short "$AFTER")"
if [[ "$BEFORE" == "$AFTER" ]]; then
  warn "Code SHA unchanged. This was a rebuild/restart only, no new code was pulled."
fi
