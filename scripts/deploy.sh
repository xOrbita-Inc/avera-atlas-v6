#!/usr/bin/env bash
# =============================================================
# AVERA-ATLAS VPS Deployment Script
# =============================================================
# Usage:
#   ./scripts/deploy.sh              — deploy all services
#   ./scripts/deploy.sh ui           — deploy single service
#   ./scripts/deploy.sh ui planner   — deploy multiple
#
# This script:
#   1. Pulls latest service code from GitHub
#   2. Validates docker-compose.prod.yaml
#   3. Builds and restarts the specified service(s)
#
# Requirements:
#   - Run from /avera-atlas-v6 on the VPS
#   - docker-compose.prod.yaml must exist
#   - traefik-proxy Docker network must exist
#
# DO NOT run this script locally — it is VPS only.
# =============================================================

set -euo pipefail

COMPOSE_FILE="docker-compose.prod.yaml"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[DEPLOY]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# =============================================================
# GUARDS
# =============================================================

# Must run from the repo root
cd "$REPO_DIR"

# Compose file must exist
[[ -f "$COMPOSE_FILE" ]] || \
  fail "$COMPOSE_FILE not found. Run from /avera-atlas-v6"

# =============================================================
# STEP 1 — Pull latest service code
# Pulls services/ directory only.
# Never touches docker-compose.yaml or
# docker-compose.prod.yaml on the VPS.
# =============================================================

log "Fetching latest code from origin/main..."
git fetch origin main

log "Pulling services/ directory..."
git checkout origin/main -- services/

ok "Service code updated"

# =============================================================
# STEP 2 — Validate compose config
# =============================================================

log "Validating $COMPOSE_FILE..."
if docker compose -f "$COMPOSE_FILE" config --quiet \
   2>&1 | grep -v "variable is not set" \
        | grep -v "traefik-proxy" \
        | grep -qi "error"; then
  fail "Compose config validation failed"
fi
ok "Compose config valid"

# =============================================================
# STEP 3 — Build and deploy
# =============================================================

SERVICES=("$@")

if [[ ${#SERVICES[@]} -eq 0 ]]; then
  warn "No service specified — deploying ALL services"
  warn "This will restart every container."
  read -r -p "Continue? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || \
    { log "Aborted."; exit 0; }

  log "Building all services..."
  docker compose -f "$COMPOSE_FILE" build

  log "Restarting all services..."
  docker compose -f "$COMPOSE_FILE" up -d

  ok "All services deployed"
else
  for SERVICE in "${SERVICES[@]}"; do
    log "Building $SERVICE..."
    docker compose -f "$COMPOSE_FILE" \
      build "$SERVICE"

    log "Restarting $SERVICE..."
    docker compose -f "$COMPOSE_FILE" \
      up -d "$SERVICE"

    ok "$SERVICE deployed"
  done
fi

# =============================================================
# STEP 4 — Status report
# =============================================================

log "Container status:"
docker compose -f "$COMPOSE_FILE" ps \
  --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

echo ""
ok "Deployment complete"
