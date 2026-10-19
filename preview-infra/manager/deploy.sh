#!/bin/bash
# /opt/postgres-ai-previews/manager/deploy.sh

set -euo pipefail

PREVIEW_BASE_DIR="/opt/postgres-ai-previews"

# Required variables (fail if missing)
: "${BRANCH_SLUG:?BRANCH_SLUG is required}"
: "${COMMIT_SHA:?COMMIT_SHA is required}"

BRANCH_NAME="${BRANCH_NAME:-$BRANCH_SLUG}"
UPDATE_MODE="${1:-}"
PREVIEW_DIR="${PREVIEW_BASE_DIR}/previews/${BRANCH_SLUG}"
MAX_PREVIEWS=2
GLOBAL_LOCK="${PREVIEW_BASE_DIR}/manager/.global.lock"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Validate BRANCH_SLUG format (security)
if ! [[ "$BRANCH_SLUG" =~ ^[a-z0-9-]{1,63}$ ]]; then
  log "ERROR: Invalid BRANCH_SLUG format: $BRANCH_SLUG"
  exit 1
fi

# Validate PREVIEW_DIR is within expected path (safety)
REAL_PREVIEW_DIR=$(realpath -m "$PREVIEW_DIR")
REAL_BASE="${PREVIEW_BASE_DIR}/previews"
if [[ "$REAL_PREVIEW_DIR" != "${REAL_BASE}/"* ]]; then
  log "ERROR: PREVIEW_DIR outside expected path: $REAL_PREVIEW_DIR"
  exit 1
fi

# =============================================================================
# GLOBAL LOCK for quota/resource checks
# =============================================================================
exec 200>"$GLOBAL_LOCK"
flock -w 30 200 || { log "ERROR: Could not acquire global lock"; exit 1; }

# =============================================================================
# QUOTA CHECK (skip for updates)
# =============================================================================
if [ "$UPDATE_MODE" != "--update" ]; then
  # Count RUNNING preview containers (not directories)
  CURRENT=$(docker ps --filter 'label=pgai.preview=true' --format '{{.ID}}' | wc -l)
  # Divide by ~9 services per preview to get preview count (updated from 7)
  CURRENT_PREVIEWS=$((CURRENT / 9))

  if [ "$CURRENT_PREVIEWS" -ge "$MAX_PREVIEWS" ]; then
    log "ERROR: Maximum preview limit ($MAX_PREVIEWS) reached. Running: $CURRENT_PREVIEWS"
    log "Active previews:"
    docker ps --filter 'label=pgai.preview=true' --format '{{.Labels}}' | grep 'com.docker.compose.project=' | sort -u
    exit 1
  fi

  # Check disk space (fail if < 5GB free)
  FREE_GB=$(df -BG "${PREVIEW_BASE_DIR}" | awk 'NR==2 {print $4}' | tr -d 'G')
  if [ "$FREE_GB" -lt 5 ]; then
    log "ERROR: Insufficient disk space. Free: ${FREE_GB}GB, Required: 5GB"
    exit 1
  fi

  # Check memory (fail if < 1GB free)
  FREE_MEM_MB=$(free -m | awk '/^Mem:/ {print $7}')
  if [ "$FREE_MEM_MB" -lt 1024 ]; then
    log "ERROR: Insufficient memory. Available: ${FREE_MEM_MB}MB, Required: 1024MB"
    exit 1
  fi
fi

# Release global lock
flock -u 200

# =============================================================================
# PER-PREVIEW LOCK
# =============================================================================
mkdir -p "${PREVIEW_DIR}"
exec 201>"${PREVIEW_DIR}/.lock"
flock -w 60 201 || { log "ERROR: Could not acquire preview lock"; exit 1; }

log "Deploying preview: $BRANCH_SLUG (commit: ${COMMIT_SHA:0:8})"
cd "${PREVIEW_DIR}"

# =============================================================================
# CREDENTIALS AND REGISTRY SETTINGS
# =============================================================================
# PGAI_REGISTRY and PGAI_TAG from CI take precedence over .env values
CI_PGAI_REGISTRY="${PGAI_REGISTRY:-}"
CI_PGAI_TAG="${PGAI_TAG:-}"

if [ ! -f ".env" ]; then
  GRAFANA_PASSWORD=$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 16)
  PGAI_REGISTRY="${CI_PGAI_REGISTRY:-registry.gitlab.com/postgres-ai/postgresai}"
  PGAI_TAG="${CI_PGAI_TAG:-${BRANCH_SLUG}}"
  cat > ".env" << EOF
BRANCH_SLUG=${BRANCH_SLUG}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
PGAI_REGISTRY=${PGAI_REGISTRY}
PGAI_TAG=${PGAI_TAG}
EOF
  chmod 600 .env
  log "Generated new credentials"
else
  # Source existing .env to get GRAFANA_PASSWORD
  source .env
  # CI values override stored values
  PGAI_REGISTRY="${CI_PGAI_REGISTRY:-${PGAI_REGISTRY:-registry.gitlab.com/postgres-ai/postgresai}}"
  PGAI_TAG="${CI_PGAI_TAG:-${PGAI_TAG:-${BRANCH_SLUG}}}"
  # Update .env with current values
  sed -i "s|^PGAI_REGISTRY=.*|PGAI_REGISTRY=${PGAI_REGISTRY}|" .env
  sed -i "s|^PGAI_TAG=.*|PGAI_TAG=${PGAI_TAG}|" .env
fi

log "Using registry: ${PGAI_REGISTRY}, tag: ${PGAI_TAG}"

# Update COMMIT_SHA in .env
if grep -q '^COMMIT_SHA=' .env 2>/dev/null; then
  sed -i "s/^COMMIT_SHA=.*/COMMIT_SHA=${COMMIT_SHA}/" .env
else
  echo "COMMIT_SHA=${COMMIT_SHA}" >> .env
fi

# =============================================================================
# COPY SHARED FILES
# =============================================================================
mkdir -p workload
cp -r "${PREVIEW_BASE_DIR}/shared/workload/"* workload/ 2>/dev/null || true
cp "${PREVIEW_BASE_DIR}/shared/instances.yml" instances.yml 2>/dev/null || true

# =============================================================================
# GENERATE COMPOSE FILE
# =============================================================================
[ -n "${GRAFANA_PASSWORD:-}" ] || { log "ERROR: GRAFANA_PASSWORD is empty or missing in .env"; exit 1; }
export BRANCH_SLUG COMMIT_SHA GRAFANA_PASSWORD PGAI_REGISTRY PGAI_TAG
envsubst '$BRANCH_SLUG $GRAFANA_PASSWORD $PGAI_REGISTRY $PGAI_TAG' \
  < "${PREVIEW_BASE_DIR}/shared/docker-compose.preview.template.yml" \
  > docker-compose.yml

# =============================================================================
# STATE FILE (NO secrets - only metadata)
# =============================================================================
# Read old created_at before overwriting state.json
OLD_CREATED_AT=$(jq -r '.created_at // empty' state.json 2>/dev/null || echo "")
CREATED_AT="${OLD_CREATED_AT:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
cat > state.json << EOF
{
  "branch": "${BRANCH_NAME}",
  "branch_slug": "${BRANCH_SLUG}",
  "commit_sha": "${COMMIT_SHA}",
  "created_at": "${CREATED_AT}",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# =============================================================================
# DNS RECORD (only on fresh deploy)
# =============================================================================
if [ "$UPDATE_MODE" != "--update" ]; then
  log "Creating DNS record..."
  "${PREVIEW_BASE_DIR}/scripts/cloudflare-dns.sh" create "preview-${BRANCH_SLUG}" || \
    log "WARNING: DNS record creation failed (may already exist)"
fi

# =============================================================================
# DEPLOY
# =============================================================================
# For fresh deploys, clean up all volumes
if [ "$UPDATE_MODE" != "--update" ]; then
  log "Cleaning up existing deployment (if any)..."
  docker compose -p "preview-${BRANCH_SLUG}" down -v --remove-orphans 2>/dev/null || true
fi

# Always delete Grafana data volume to ensure password sync
# (Grafana stores password in database on first init, env var is ignored after)
log "Resetting Grafana data for password sync..."
docker compose -p "preview-${BRANCH_SLUG}" stop grafana 2>/dev/null || true
docker compose -p "preview-${BRANCH_SLUG}" rm -f grafana 2>/dev/null || true

# Find and delete Grafana volume by pattern (handles naming variations)
GRAFANA_VOL=$(docker volume ls -q --filter "name=preview-${BRANCH_SLUG}" | grep grafana || true)
if [ -n "$GRAFANA_VOL" ]; then
  log "Deleting Grafana volume: $GRAFANA_VOL"
  docker volume rm "$GRAFANA_VOL" || log "WARNING: Could not delete volume $GRAFANA_VOL"
else
  log "No Grafana volume found to delete"
fi

log "Pulling images..."
docker compose -p "preview-${BRANCH_SLUG}" pull --quiet 2>/dev/null || true

log "Starting services..."
docker compose -p "preview-${BRANCH_SLUG}" up -d --force-recreate --remove-orphans

# =============================================================================
# HEALTH CHECK (using docker inspect, not wget inside container)
# =============================================================================
log "Waiting for Grafana to be healthy..."
GRAFANA_CONTAINER="preview-${BRANCH_SLUG}-grafana-1"

for i in $(seq 1 60); do
  HEALTH=$(docker inspect --format='{{.State.Health.Status}}' "$GRAFANA_CONTAINER" 2>/dev/null || echo "starting")
  if [ "$HEALTH" = "healthy" ]; then
    log "Grafana is healthy"
    break
  fi
  if [ $i -eq 60 ]; then
    log "ERROR: Grafana failed to become healthy (status: $HEALTH)"
    log "Container status:"
    docker compose -p "preview-${BRANCH_SLUG}" ps
    log "Grafana logs:"
    docker compose -p "preview-${BRANCH_SLUG}" logs grafana --tail=50

    # Rollback
    log "Rolling back..."
    docker compose -p "preview-${BRANCH_SLUG}" down -v --remove-orphans
    "${PREVIEW_BASE_DIR}/scripts/cloudflare-dns.sh" delete "preview-${BRANCH_SLUG}" || true
    rm -rf "${PREVIEW_DIR}"
    exit 1
  fi
  sleep 2
done

# =============================================================================
# SUCCESS
# =============================================================================
log "SUCCESS: Preview deployed"
log "URL: https://preview-${BRANCH_SLUG}.pgai.watch"
log "Username: monitor"
log "Password: retrieve from the preview VM's restricted .env file"
