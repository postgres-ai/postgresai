# PostgresAI Monitoring Preview Environments Plan

**Version:** 1.7.0
**Date:** 2026-01-25
**Status:** Implementation Complete - Awaiting VM Provisioning

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Requirements Summary](#2-requirements-summary)
3. [Architecture Overview](#3-architecture-overview)
4. [Multi-Tenancy Solution](#4-multi-tenancy-solution)
5. [Demo Database & Workload](#5-demo-database--workload)
6. [CI/CD Pipeline Design](#6-cicd-pipeline-design)
7. [Lifecycle Management](#7-lifecycle-management)
8. [DNS & SSL Configuration](#8-dns--ssl-configuration)
9. [Resource Planning](#9-resource-planning)
10. [VM Provisioning & Monitoring](#10-vm-provisioning--monitoring)
11. [Implementation Phases](#11-implementation-phases)
12. [Definition of Done](#12-definition-of-done)
13. [Risks & Mitigations](#13-risks--mitigations)

---

## 1. Executive Summary

This plan describes the implementation of branch-based preview environments for PostgresAI monitoring. Each preview environment will be accessible at `preview-{branch-slug}.pgai.watch` (3rd level subdomain for Cloudflare wildcard cert compatibility) and will run a complete monitoring stack observing its own demo PostgreSQL database with realistic workload.

**Note:** Branch names like `claude/feature-x` or `feature_test` are sanitized to `claude-feature-x` and `feature-test` for DNS compatibility (e.g., `preview-claude-feature-x.pgai.watch`).

**Key design decisions:**
- Single Hetzner VM hosting multiple preview environments
- Traefik reverse proxy for dynamic routing with **wildcard SSL via DNS-01**
- **Dynamic DNS records via Cloudflare API** (created on deploy, deleted on destroy)
- Manual CI trigger for preview creation (not automatic)
- 3-day TTL with cleanup on branch merge/delete
- pgbench-based workload with variable patterns (spikes, valleys)
- **Per-preview generated credentials** (stored only in `.env`, never in logs)
- **Hard limit enforcement** on concurrent previews (count running containers)
- **VM-side locking** with flock to prevent races
- **Config sync via rsync** (not Docker registry)

---

## 2. Requirements Summary

| Requirement | Decision |
|-------------|----------|
| Infrastructure | Single Hetzner VM, 4 vCPU / 8GB initially |
| VM naming | Must start with `ralph-preview-` (e.g., `ralph-preview-01`) |
| VM constraint | **Only 1 Hetzner machine** for this project can exist at any time |
| Multi-tenancy | Traefik + Docker Compose per branch |
| DNS | **Dynamic A records via Cloudflare API** (per-preview) |
| SSL | Wildcard cert via DNS-01 challenge |
| Authentication | Per-preview generated password (in `.env` only) |
| Preview trigger | Manual CI job trigger |
| Lifecycle | 3-day TTL + cleanup on merge/delete |
| Demo workload | pgbench with variable patterns, max 4 sessions |
| Deployment | rsync + SSH from GitLab CI runners |
| Max concurrent | Hard limit: 2 (counts running containers) |
| Concurrency control | VM-side flock + CI resource_group |

---

## 3. Architecture Overview

```
                                    ┌─────────────────────────────────────────────────────────┐
                                    │              Hetzner VM (CX31: 4 vCPU / 8GB)            │
                                    │                                                         │
    ┌──────────┐                    │  ┌─────────────────────────────────────────────────┐   │
    │Cloudflare│                    │  │         Traefik (ports 80, 443)                 │   │
    │   DNS    │───────────────────▶│  │   - Wildcard cert: *.pgai.watch (DNS-01)        │   │
    │ (API)    │                    │  │   - Routes preview-{branch}.pgai.watch → Grafana│   │
    │          │                    │  │   - Dashboard: localhost only (SSH tunnel)      │   │
    └──────────┘                    │  └──────────────────────┬──────────────────────────┘   │
         ▲                          │                         │                              │
         │ create/delete            │         ┌───────────────┼───────────────┐              │
         │ A records                │         ▼               ▼               ▼              │
         │                          │  ┌────────────┐  ┌────────────┐                        │
    ┌────┴─────┐                    │  │ feature-x  │  │  fix-bug   │   (max 2 concurrent)  │
    │ deploy/  │                    │  │  network   │  │  network   │                        │
    │ destroy  │                    │  │            │  │            │                        │
    │ scripts  │                    │  │ - grafana  │  │ - grafana  │                        │
    └──────────┘                    │  │ - pgwatch  │  │ - pgwatch  │                        │
                                    │  │ - victoria │  │ - victoria │                        │
                                    │  │ - sink-pg  │  │ - sink-pg  │                        │
                                    │  │ - target   │  │ - target   │                        │
                                    │  │ - workload │  │ - workload │                        │
                                    │  └────────────┘  └────────────┘                        │
                                    │                                                         │
                                    │  ┌─────────────────────────────────────────────────┐   │
                                    │  │              VM Self-Monitoring                  │   │
                                    │  │   - node-exporter (disk, memory, CPU alerts)    │   │
                                    │  │   - Healthchecks.io ping for uptime             │   │
                                    │  └─────────────────────────────────────────────────┘   │
                                    │                                                         │
                                    │  ┌─────────────────────────────────────────────────┐   │
                                    │  │          Preview Manager                         │   │
                                    │  │   - flock-based locking (per-preview + global)  │   │
                                    │  │   - Counts running containers for quota         │   │
                                    │  │   - TTL cleanup + conditional docker prune      │   │
                                    │  └─────────────────────────────────────────────────┘   │
                                    └─────────────────────────────────────────────────────────┘
```

### Key Components

1. **Traefik Reverse Proxy**
   - Single entry point (ports 80/443)
   - **Wildcard certificate** via DNS-01 challenge (Cloudflare API)
   - Auto-discovers containers via Docker labels
   - Dashboard **not exposed publicly** (SSH tunnel access only)

2. **Isolated Docker Networks**
   - Each preview runs in its own Docker network (`preview-{slug}`)
   - No port conflicts - services communicate internally
   - Only Grafana joins `traefik-public` network for routing

3. **Preview Manager**
   - **flock-based locking** (global for quota/prune, per-preview for deploy/destroy)
   - **Counts running containers** (not directories) for quota enforcement
   - **Conditional docker prune** (only when disk >80%)
   - TTL cleanup via cron (3-day expiry)

4. **VM Self-Monitoring** (Required)
   - node-exporter for system metrics
   - Alerts on disk > 80%, memory > 85%
   - External uptime monitoring (Healthchecks.io)

5. **Security Notes**
   - Credentials stored only in `.env` with `chmod 600` (never in state.json or logs)
   - Docker socket mounted read-only in Traefik (acceptable for V1; consider socket proxy for V2)

---

## 4. Multi-Tenancy Solution

### 4.1 Port Conflict Resolution

**Problem:** Multiple monitoring stacks need fixed ports (Grafana:3000, VictoriaMetrics:9090, etc.).

**Solution:** Isolated Docker networks + Traefik routing (no host port exposure)

```yaml
# docker-compose.preview.yml structure
networks:
  default:
    name: preview-${BRANCH_SLUG}
  traefik-public:
    external: true  # CRITICAL: Created by Traefik stack

services:
  grafana:
    networks:
      - default
      - traefik-public
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.preview-${BRANCH_SLUG}.rule=Host(`preview-${BRANCH_SLUG}.pgai.watch`)"
      - "traefik.http.routers.preview-${BRANCH_SLUG}.entrypoints=websecure"
      - "traefik.http.routers.preview-${BRANCH_SLUG}.tls.certresolver=letsencrypt"
      - "traefik.http.services.preview-${BRANCH_SLUG}.loadbalancer.server.port=3000"
      - "traefik.docker.network=traefik-public"
      - "pgai.preview=true"  # Label for quota counting
    # NO ports: section - internal only
```

### 4.2 Branch Name Sanitization (Collision-Safe)

Create a **single shared script** used by CI, deploy, destroy, and webhook:

```bash
#!/bin/bash
# scripts/sanitize-branch.sh
# SINGLE SOURCE OF TRUTH for branch name sanitization

set -euo pipefail

BRANCH="${1:?Usage: sanitize-branch.sh <branch-name>}"

# 1. Lowercase
# 2. Replace / and _ with -
# 3. Remove non-alphanumeric except -
# 4. Collapse multiple dashes
# 5. Remove leading/trailing dashes
CLEAN=$(echo "$BRANCH" | tr '[:upper:]' '[:lower:]' | \
  sed 's/[\/\_]/-/g' | \
  sed 's/[^a-z0-9-]//g' | \
  sed 's/--*/-/g' | \
  sed 's/^-//;s/-$//')

# 6. If longer than 50 chars, truncate and append hash for uniqueness
if [ ${#CLEAN} -gt 50 ]; then
  HASH=$(echo -n "$BRANCH" | sha1sum | cut -c1-8)
  CLEAN="${CLEAN:0:50}-${HASH}"
fi

# 7. Final trim to 63 chars (DNS label limit)
echo "${CLEAN:0:63}"
```

**Examples:**
```
feature/add-metrics                              → feature-add-metrics
claude/implement-caching                         → claude-implement-caching
feature_test_branch                              → feature-test-branch
feature/optimize-query-execution-plan-for-large  → feature-optimize-query-execution-plan-for-lar-a1b2c3d4
```

### 4.3 Directory Structure on VM

```
/opt/postgres-ai-previews/
├── scripts/
│   ├── sanitize-branch.sh          # SINGLE SOURCE OF TRUTH
│   └── cloudflare-dns.sh           # DNS record management
├── traefik/
│   ├── docker-compose.yml          # Traefik stack
│   ├── traefik.yml                 # Static config (DNS-01)
│   ├── .env                        # CF_DNS_API_TOKEN, CF_ZONE_ID (chmod 600)
│   └── acme.json                   # Wildcard cert storage (chmod 600)
├── previews/
│   ├── feature-add-metrics/
│   │   ├── docker-compose.yml      # Generated per-preview
│   │   ├── config/                 # Synced from repo via rsync
│   │   ├── .env                    # Credentials ONLY (chmod 600)
│   │   ├── state.json              # Metadata (NO secrets)
│   │   └── .lock                   # flock file
│   └── fix-bug-123/
│       └── ...
├── shared/
│   ├── docker-compose.preview.template.yml
│   └── workload/
│       └── pgbench-variable.sh
└── manager/
    ├── deploy.sh                   # Main deploy script
    ├── destroy.sh                  # Cleanup script
    ├── cleanup-ttl.sh              # Cron job for TTL
    └── .global.lock                # Global lock for quota/prune
```

---

## 5. Demo Database & Workload

### 5.1 Target Database Configuration

Each preview runs a PostgreSQL 15 instance with:
- `pg_stat_statements` enabled
- `target_database` database
- Initialized with pgbench schema (`pgbench -i -s 10` ≈ 160MB)
- **Healthcheck** for proper startup sequencing

### 5.2 Variable Workload Generator (with Graceful Shutdown)

```bash
#!/bin/bash
# /opt/postgres-ai-previews/shared/workload/pgbench-variable.sh

set -euo pipefail

# Graceful shutdown handling
SHUTDOWN=0
trap 'SHUTDOWN=1; echo "[$(date)] Received shutdown signal, finishing current pattern..."' SIGTERM SIGINT

DB_HOST="${TARGET_DB_HOST:-target-db}"
DB_PORT="${TARGET_DB_PORT:-5432}"
DB_NAME="${TARGET_DB_NAME:-target_database}"
DB_USER="${TARGET_DB_USER:-postgres}"

export PGPASSWORD="${PGPASSWORD:-postgres}"

# Wait for database to be truly ready (not just socket open)
echo "[$(date)] Waiting for database..."
until psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1" >/dev/null 2>&1; do
  echo "[$(date)] Database not ready, retrying..."
  sleep 2
done
echo "[$(date)] Database ready"

# Initialize pgbench schema ONLY if not exists
if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
     -c "SELECT 1 FROM pgbench_accounts LIMIT 1" >/dev/null 2>&1; then
  echo "[$(date)] Initializing pgbench schema..."
  pgbench -i -s 10 -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"
else
  echo "[$(date)] pgbench schema already exists, skipping init"
fi

CONN="-h $DB_HOST -p $DB_PORT -U $DB_USER $DB_NAME"

while [ $SHUTDOWN -eq 0 ]; do
  PATTERN=$((RANDOM % 6))

  case $PATTERN in
    0) # Spike: high intensity, short duration
      echo "[$(date)] Pattern: SPIKE"
      timeout 35 pgbench -c 4 -j 2 -T 30 $CONN 2>/dev/null || true
      [ $SHUTDOWN -eq 0 ] && sleep 60
      ;;
    1) # Valley: minimal activity
      echo "[$(date)] Pattern: VALLEY"
      timeout 125 pgbench -c 1 -j 1 -T 120 -R 5 $CONN 2>/dev/null || true
      ;;
    2) # Ramp up
      echo "[$(date)] Pattern: RAMP UP"
      for clients in 1 2 3 4; do
        [ $SHUTDOWN -eq 1 ] && break
        timeout 50 pgbench -c $clients -j $clients -T 45 $CONN 2>/dev/null || true
      done
      ;;
    3) # Ramp down
      echo "[$(date)] Pattern: RAMP DOWN"
      for clients in 4 3 2 1; do
        [ $SHUTDOWN -eq 1 ] && break
        timeout 50 pgbench -c $clients -j $clients -T 45 $CONN 2>/dev/null || true
      done
      ;;
    4) # Steady medium
      echo "[$(date)] Pattern: STEADY MEDIUM"
      timeout 185 pgbench -c 2 -j 2 -T 180 $CONN 2>/dev/null || true
      ;;
    5) # Burst pattern
      echo "[$(date)] Pattern: BURST"
      for i in 1 2 3; do
        [ $SHUTDOWN -eq 1 ] && break
        timeout 25 pgbench -c 4 -j 2 -T 20 $CONN 2>/dev/null || true
        [ $SHUTDOWN -eq 0 ] && sleep 40
      done
      ;;
  esac

  if [ $SHUTDOWN -eq 0 ]; then
    PAUSE=$((30 + RANDOM % 90))
    echo "[$(date)] Pausing for ${PAUSE}s"
    sleep $PAUSE
  fi
done

echo "[$(date)] Workload generator stopped gracefully"
```

### 5.3 Workload Container

```yaml
workload-generator:
  image: postgres:15
  depends_on:
    target-db:
      condition: service_healthy
  volumes:
    - ./workload/pgbench-variable.sh:/workload.sh:ro
  entrypoint: ["/bin/bash", "/workload.sh"]
  environment:
    TARGET_DB_HOST: target-db
    TARGET_DB_NAME: target_database
    TARGET_DB_USER: postgres
    PGPASSWORD: postgres
  stop_grace_period: 60s
  labels:
    - "pgai.preview=true"
  mem_limit: 256m
  cpus: 0.5
```

---

## 6. CI/CD Pipeline Design

### 6.1 GitLab CI Configuration

**Key fixes from reviews:**
- Calls shared `sanitize-branch.sh` script (single source of truth)
- Uses `CI_COMMIT_REF_SLUG` for environment name (available at parse time)
- Uses dotenv artifact for computed values
- `resource_group` uses slug from dotenv
- Credentials NOT printed to CI logs

```yaml
# .gitlab-ci.yml additions

stages:
  - build
  - test
  - preview
  - cleanup

variables:
  PREVIEW_VM_HOST: "preview.pgai.internal"
  PREVIEW_VM_USER: "deploy"
  PREVIEW_BASE_DIR: "/opt/postgres-ai-previews"

# =============================================================================
# PREVIEW DEPLOYMENT (Manual Trigger)
# =============================================================================
preview:deploy:
  stage: preview
  when: manual
  image: alpine:3.21
  before_script:
    - apk add --no-cache openssh-client rsync bash coreutils
    - eval $(ssh-agent -s)
    - printf '%s' "$PREVIEW_SSH_PRIVATE_KEY" | tr -d '\r' | ssh-add -  # Pipe avoids ps exposure, POSIX compatible
    - mkdir -p ~/.ssh && chmod 700 ~/.ssh
    # Use shared script for slug computation
    - chmod +x ./scripts/sanitize-branch.sh
    - export BRANCH_SLUG=$(./scripts/sanitize-branch.sh "$CI_COMMIT_REF_NAME")
    - test -n "$BRANCH_SLUG" || (echo "ERROR: BRANCH_SLUG is empty" && exit 1)
    # Write to dotenv for environment and subsequent jobs
    - echo "BRANCH_SLUG=${BRANCH_SLUG}" >> deploy.env
    - echo "PREVIEW_URL=https://preview-${BRANCH_SLUG}.pgai.watch" >> deploy.env
  script:
    - source deploy.env
    # Sync scripts first (including sanitize-branch.sh)
    - |
      rsync -avz --delete -e "ssh -o StrictHostKeyChecking=no" \
        ./scripts/ \
        ${PREVIEW_VM_USER}@${PREVIEW_VM_HOST}:${PREVIEW_BASE_DIR}/scripts/
    # Sync config files
    - |
      rsync -avz --delete -e "ssh -o StrictHostKeyChecking=no" \
        ./config/ \
        ${PREVIEW_VM_USER}@${PREVIEW_VM_HOST}:${PREVIEW_BASE_DIR}/previews/${BRANCH_SLUG}/config/
    # Run deploy script on VM
    - |
      ssh -o StrictHostKeyChecking=no ${PREVIEW_VM_USER}@${PREVIEW_VM_HOST} \
        "BRANCH_SLUG='${BRANCH_SLUG}' \
         COMMIT_SHA='${CI_COMMIT_SHA}' \
         BRANCH_NAME='${CI_COMMIT_REF_NAME}' \
         ${PREVIEW_BASE_DIR}/manager/deploy.sh"
    # Output access info (NO password in logs)
    - |
      echo "============================================"
      echo "Preview deployed successfully!"
      echo "URL: ${PREVIEW_URL}"
      echo "Username: admin"
      echo "Password: SSH to VM and run:"
      echo "  cat ${PREVIEW_BASE_DIR}/previews/${BRANCH_SLUG}/.env"
      echo "============================================"
  environment:
    name: preview/$CI_COMMIT_REF_SLUG
    url: https://preview-$CI_COMMIT_REF_SLUG.pgai.watch
    on_stop: preview:destroy
    auto_stop_in: 3 days
  artifacts:
    reports:
      dotenv: deploy.env
  rules:
    - if: $CI_COMMIT_BRANCH != "main"

# =============================================================================
# PREVIEW DESTRUCTION
# =============================================================================
preview:destroy:
  stage: cleanup
  when: manual
  image: alpine:3.21
  environment:
    name: preview/$CI_COMMIT_REF_SLUG
    action: stop
  before_script:
    - apk add --no-cache openssh-client bash coreutils
    - eval $(ssh-agent -s)
    - printf '%s' "$PREVIEW_SSH_PRIVATE_KEY" | tr -d '\r' | ssh-add -  # Pipe avoids ps exposure, POSIX compatible
    - chmod +x ./scripts/sanitize-branch.sh
    - export BRANCH_SLUG=$(./scripts/sanitize-branch.sh "$CI_COMMIT_REF_NAME")
  script:
    - |
      ssh -o StrictHostKeyChecking=no ${PREVIEW_VM_USER}@${PREVIEW_VM_HOST} \
        "BRANCH_SLUG='${BRANCH_SLUG}' \
         ${PREVIEW_BASE_DIR}/manager/destroy.sh"
  rules:
    - if: $CI_COMMIT_BRANCH != "main"

# =============================================================================
# RE-DEPLOY ON PUSH (if preview exists)
# =============================================================================
preview:update:
  stage: preview
  image: alpine:3.21
  before_script:
    - apk add --no-cache openssh-client rsync bash coreutils
    - eval $(ssh-agent -s)
    - printf '%s' "$PREVIEW_SSH_PRIVATE_KEY" | tr -d '\r' | ssh-add -  # Pipe avoids ps exposure, POSIX compatible
    - chmod +x ./scripts/sanitize-branch.sh
    - export BRANCH_SLUG=$(./scripts/sanitize-branch.sh "$CI_COMMIT_REF_NAME")
  script:
    - |
      # Check if preview exists (by checking for running containers)
      EXISTS=$(ssh -o StrictHostKeyChecking=no ${PREVIEW_VM_USER}@${PREVIEW_VM_HOST} \
        "docker ps --filter 'label=com.docker.compose.project=preview-${BRANCH_SLUG}' --format '{{.ID}}' | head -1")

      if [ -n "$EXISTS" ]; then
        echo "Preview exists, updating..."

        # Sync scripts
        rsync -avz --delete -e "ssh -o StrictHostKeyChecking=no" \
          ./scripts/ \
          ${PREVIEW_VM_USER}@${PREVIEW_VM_HOST}:${PREVIEW_BASE_DIR}/scripts/

        # Sync configs
        rsync -avz --delete -e "ssh -o StrictHostKeyChecking=no" \
          ./config/ \
          ${PREVIEW_VM_USER}@${PREVIEW_VM_HOST}:${PREVIEW_BASE_DIR}/previews/${BRANCH_SLUG}/config/

        # Run update
        ssh -o StrictHostKeyChecking=no ${PREVIEW_VM_USER}@${PREVIEW_VM_HOST} \
          "BRANCH_SLUG='${BRANCH_SLUG}' \
           COMMIT_SHA='${CI_COMMIT_SHA}' \
           ${PREVIEW_BASE_DIR}/manager/deploy.sh --update"
      else
        echo "No running preview for ${BRANCH_SLUG}, skipping"
      fi
  rules:
    - if: $CI_COMMIT_BRANCH != "main"
      when: on_success
```

### 6.2 Cloudflare DNS Management Script

```bash
#!/bin/bash
# /opt/postgres-ai-previews/scripts/cloudflare-dns.sh
# Manages DNS records via Cloudflare API

set -euo pipefail

ACTION="${1:?Usage: cloudflare-dns.sh <create|delete> <subdomain>}"
SUBDOMAIN="${2:?Usage: cloudflare-dns.sh <create|delete> <subdomain>}"

# Load credentials
source /opt/postgres-ai-previews/traefik/.env
: "${CF_DNS_API_TOKEN:?CF_DNS_API_TOKEN not set}"
: "${CF_ZONE_ID:?CF_ZONE_ID not set}"
: "${VM_PUBLIC_IP:?VM_PUBLIC_IP not set}"

FULL_NAME="${SUBDOMAIN}.pgai.watch"
API_BASE="https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID}/dns_records"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] cloudflare-dns: $*"; }

case "$ACTION" in
  create)
    # Check if record exists
    EXISTING=$(curl -s -X GET "${API_BASE}?name=${FULL_NAME}" \
      -H "Authorization: Bearer ${CF_DNS_API_TOKEN}" \
      -H "Content-Type: application/json" | jq -r '.result[0].id // empty')

    if [ -n "$EXISTING" ]; then
      log "Record ${FULL_NAME} already exists (ID: ${EXISTING}), updating..."
      curl -s -X PUT "${API_BASE}/${EXISTING}" \
        -H "Authorization: Bearer ${CF_DNS_API_TOKEN}" \
        -H "Content-Type: application/json" \
        --data '{
          "type": "A",
          "name": "'"${FULL_NAME}"'",
          "content": "'"${VM_PUBLIC_IP}"'",
          "ttl": 120,
          "proxied": true
        }' | jq -e '.success' > /dev/null
    else
      log "Creating DNS record: ${FULL_NAME} -> ${VM_PUBLIC_IP}"
      curl -s -X POST "${API_BASE}" \
        -H "Authorization: Bearer ${CF_DNS_API_TOKEN}" \
        -H "Content-Type: application/json" \
        --data '{
          "type": "A",
          "name": "'"${FULL_NAME}"'",
          "content": "'"${VM_PUBLIC_IP}"'",
          "ttl": 120,
          "proxied": true
        }' | jq -e '.success' > /dev/null
    fi
    log "DNS record created/updated: ${FULL_NAME}"
    ;;

  delete)
    RECORD_ID=$(curl -s -X GET "${API_BASE}?name=${FULL_NAME}" \
      -H "Authorization: Bearer ${CF_DNS_API_TOKEN}" \
      -H "Content-Type: application/json" | jq -r '.result[0].id // empty')

    if [ -n "$RECORD_ID" ]; then
      log "Deleting DNS record: ${FULL_NAME} (ID: ${RECORD_ID})"
      curl -s -X DELETE "${API_BASE}/${RECORD_ID}" \
        -H "Authorization: Bearer ${CF_DNS_API_TOKEN}" | jq -e '.success' > /dev/null
      log "DNS record deleted"
    else
      log "No DNS record found for ${FULL_NAME}, skipping"
    fi
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 1
    ;;
esac
```

### 6.3 VM Deploy Script (with Locking and Fixed Secrets)

```bash
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
  # Divide by ~7 services per preview to get preview count
  CURRENT_PREVIEWS=$((CURRENT / 7))

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
# CREDENTIALS (only generate if .env missing)
# =============================================================================
if [ ! -f ".env" ]; then
  GRAFANA_PASSWORD=$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 16)
  cat > ".env" << EOF
BRANCH_SLUG=${BRANCH_SLUG}
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
EOF
  chmod 600 .env
  log "Generated new credentials"
fi

# Always source .env (now guaranteed to exist)
source .env

# Update COMMIT_SHA in .env on updates
if [ "$UPDATE_MODE" = "--update" ]; then
  if grep -q '^COMMIT_SHA=' .env; then
    sed -i "s/^COMMIT_SHA=.*/COMMIT_SHA=${COMMIT_SHA}/" .env
  else
    echo "COMMIT_SHA=${COMMIT_SHA}" >> .env
  fi
else
  echo "COMMIT_SHA=${COMMIT_SHA}" >> .env
fi

# =============================================================================
# GENERATE COMPOSE FILE
# =============================================================================
export BRANCH_SLUG COMMIT_SHA GRAFANA_ADMIN_USER GRAFANA_ADMIN_PASSWORD
envsubst '$BRANCH_SLUG $COMMIT_SHA $GRAFANA_ADMIN_USER $GRAFANA_ADMIN_PASSWORD' \
  < "${PREVIEW_BASE_DIR}/shared/docker-compose.preview.template.yml" \
  > docker-compose.yml

# Validate generated compose file
if ! docker compose config > /dev/null 2>&1; then
  log "ERROR: Generated docker-compose.yml is invalid"
  docker compose config
  exit 1
fi

# =============================================================================
# STATE FILE (NO secrets - only metadata)
# =============================================================================
cat > state.json << EOF
{
  "branch": "${BRANCH_NAME}",
  "branch_slug": "${BRANCH_SLUG}",
  "commit_sha": "${COMMIT_SHA}",
  "created_at": "$(jq -r '.created_at // empty' state.json 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)",
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
log "Pulling images..."
docker compose -p "preview-${BRANCH_SLUG}" pull --quiet

log "Starting services..."
docker compose -p "preview-${BRANCH_SLUG}" up -d --force-recreate --remove-orphans

# =============================================================================
# HEALTH CHECK (using docker inspect, not wget inside container)
# =============================================================================
log "Waiting for Grafana to be healthy..."
GRAFANA_CONTAINER="preview-${BRANCH_SLUG}-grafana-1"

for i in $(seq 1 30); do
  HEALTH=$(docker inspect --format='{{.State.Health.Status}}' "$GRAFANA_CONTAINER" 2>/dev/null || echo "starting")
  if [ "$HEALTH" = "healthy" ]; then
    log "Grafana is healthy"
    break
  fi
  if [ $i -eq 30 ]; then
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
# SUCCESS (NO password in output)
# =============================================================================
log "SUCCESS: Preview deployed"
log "URL: https://preview-${BRANCH_SLUG}.pgai.watch"
log "Username: ${GRAFANA_ADMIN_USER}"
log "Password: cat ${PREVIEW_DIR}/.env"
```

### 6.4 VM Destroy Script

```bash
#!/bin/bash
# /opt/postgres-ai-previews/manager/destroy.sh

set -euo pipefail

PREVIEW_BASE_DIR="/opt/postgres-ai-previews"

: "${BRANCH_SLUG:?BRANCH_SLUG is required}"

PREVIEW_DIR="${PREVIEW_BASE_DIR}/previews/${BRANCH_SLUG}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Validate BRANCH_SLUG format
if ! [[ "$BRANCH_SLUG" =~ ^[a-z0-9-]{1,63}$ ]]; then
  log "ERROR: Invalid BRANCH_SLUG format: $BRANCH_SLUG"
  exit 1
fi

# Safety check: ensure we're deleting within expected path
REAL_PREVIEW_DIR=$(realpath -m "$PREVIEW_DIR")
REAL_BASE="${PREVIEW_BASE_DIR}/previews"
if [[ "$REAL_PREVIEW_DIR" != "${REAL_BASE}/"* ]]; then
  log "SAFETY ERROR: Invalid preview directory: $REAL_PREVIEW_DIR"
  exit 1
fi

if [ ! -d "$PREVIEW_DIR" ]; then
  log "Preview directory not found: $BRANCH_SLUG"
  # Still try to clean up DNS
  "${PREVIEW_BASE_DIR}/scripts/cloudflare-dns.sh" delete "preview-${BRANCH_SLUG}" 2>/dev/null || true
  exit 0
fi

# Acquire per-preview lock (skip if held - don't block, just warn)
if ! flock -n "${PREVIEW_DIR}/.lock" true 2>/dev/null; then
  log "WARNING: Preview is locked (deploy in progress?), proceeding anyway..."
fi

log "Destroying preview: $BRANCH_SLUG"

cd "$PREVIEW_DIR"
docker compose -p "preview-${BRANCH_SLUG}" down -v --remove-orphans 2>/dev/null || true

# Delete DNS record
log "Deleting DNS record..."
"${PREVIEW_BASE_DIR}/scripts/cloudflare-dns.sh" delete "preview-${BRANCH_SLUG}" || \
  log "WARNING: DNS record deletion failed"

# Safe deletion
rm -rf "$PREVIEW_DIR"

log "Preview destroyed: $BRANCH_SLUG"
```

---

## 7. Lifecycle Management

### 7.1 TTL Enforcement + Conditional Docker Cleanup (Cron)

```bash
#!/bin/bash
# /opt/postgres-ai-previews/manager/cleanup-ttl.sh
# Cron: */30 * * * * /opt/postgres-ai-previews/manager/cleanup-ttl.sh >> /var/log/preview-cleanup.log 2>&1

set -euo pipefail

PREVIEW_BASE_DIR="/opt/postgres-ai-previews"
PREVIEW_BASE="${PREVIEW_BASE_DIR}/previews"
TTL_SECONDS=$((3 * 24 * 60 * 60))  # 3 days
NOW=$(date +%s)
GLOBAL_LOCK="${PREVIEW_BASE_DIR}/manager/.global.lock"
DISK_PRUNE_THRESHOLD=80  # Prune only if disk usage > 80%

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Starting cleanup check..."

# Use nullglob to handle empty directory
shopt -s nullglob
PREVIEW_DIRS=("${PREVIEW_BASE}"/*/)
shopt -u nullglob

if [ ${#PREVIEW_DIRS[@]} -eq 0 ]; then
  log "No previews to check"
else
  for preview_dir in "${PREVIEW_DIRS[@]}"; do
    state_file="${preview_dir}state.json"
    [ -f "$state_file" ] || continue

    branch_slug=$(basename "$preview_dir")

    # Skip if locked (deploy in progress)
    if ! flock -n "${preview_dir}.lock" true 2>/dev/null; then
      log "Skipping ${branch_slug}: locked"
      continue
    fi

    updated_at=$(jq -r '.updated_at' "$state_file" 2>/dev/null || echo "")

    if [ -z "$updated_at" ]; then
      log "WARNING: No updated_at in $state_file, skipping"
      continue
    fi

    updated_ts=$(date -d "$updated_at" +%s 2>/dev/null || echo 0)
    age=$((NOW - updated_ts))

    if [ $age -gt $TTL_SECONDS ]; then
      log "Cleaning stale preview: $branch_slug (age: $((age/86400)) days)"

      # Safety check
      REAL_DIR=$(realpath -m "$preview_dir")
      if [[ "$REAL_DIR" != "${PREVIEW_BASE}/"* ]]; then
        log "SAFETY ERROR: Invalid path: $preview_dir"
        continue
      fi

      cd "$preview_dir"
      docker compose -p "preview-${branch_slug}" down -v --remove-orphans 2>/dev/null || true

      # Delete DNS record
      "${PREVIEW_BASE_DIR}/scripts/cloudflare-dns.sh" delete "preview-${branch_slug}" 2>/dev/null || true

      rm -rf "$preview_dir"
      log "Removed: $branch_slug"
    fi
  done
fi

# =============================================================================
# CONDITIONAL DOCKER CLEANUP (only if disk > threshold)
# =============================================================================
exec 200>"$GLOBAL_LOCK"
if flock -n 200; then
  DISK_USAGE=$(df "${PREVIEW_BASE_DIR}" | awk 'NR==2 {print $5}' | tr -d '%')

  if [ "$DISK_USAGE" -gt "$DISK_PRUNE_THRESHOLD" ]; then
    log "Disk usage ${DISK_USAGE}% > ${DISK_PRUNE_THRESHOLD}%, running Docker cleanup..."
    docker image prune -af --filter "until=72h" 2>/dev/null || true
    # Don't prune volumes - they belong to running previews
    log "Docker cleanup complete"
  else
    log "Disk usage ${DISK_USAGE}% <= ${DISK_PRUNE_THRESHOLD}%, skipping prune"
  fi
  flock -u 200
fi

log "Cleanup complete"
```

### 7.2 Branch Deletion Webhook (Optional)

For immediate cleanup when branches are merged/deleted. Run as a simple systemd service (internal only, no Traefik routing needed for V1):

```python
#!/usr/bin/env python3
# /opt/postgres-ai-previews/manager/webhook-handler.py

import os
import re
import subprocess
import hashlib
import threading
from flask import Flask, request, jsonify

app = Flask(__name__)
WEBHOOK_SECRET = os.environ.get('GITLAB_WEBHOOK_SECRET', '')
PREVIEW_BASE_DIR = '/opt/postgres-ai-previews'
ALLOWED_SLUG_PATTERN = re.compile(r'^[a-z0-9-]{1,63}$')

def sanitize_branch(branch: str) -> str:
    """Sanitize branch name to DNS-safe slug."""
    slug = branch.lower()
    slug = re.sub(r'[/_]', '-', slug)
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    slug = re.sub(r'-+', '-', slug)
    slug = slug.strip('-')

    if len(slug) > 50:
        hash_suffix = hashlib.sha1(branch.encode()).hexdigest()[:8]
        slug = f"{slug[:50]}-{hash_suffix}"

    return slug[:63]

def cleanup_preview_async(branch_slug: str):
    """Run cleanup in background thread."""
    def _cleanup():
        if not ALLOWED_SLUG_PATTERN.match(branch_slug):
            app.logger.error(f'Invalid branch slug: {branch_slug}')
            return

        try:
            subprocess.run(
                [f'{PREVIEW_BASE_DIR}/manager/destroy.sh'],
                env={**os.environ, 'BRANCH_SLUG': branch_slug},
                capture_output=True,
                timeout=120
            )
            app.logger.info(f'Cleaned up preview: {branch_slug}')
        except Exception as e:
            app.logger.error(f'Cleanup failed for {branch_slug}: {e}')

    thread = threading.Thread(target=_cleanup)
    thread.daemon = True
    thread.start()

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    token = request.headers.get('X-Gitlab-Token', '')
    if not token or token != WEBHOOK_SECRET:
        return jsonify({'error': 'unauthorized'}), 401

    data = request.json or {}
    event = request.headers.get('X-Gitlab-Event', '')

    # Branch deletion (push with null SHA)
    if event == 'Push Hook' and data.get('after') == '0' * 40:
        branch = data.get('ref', '').replace('refs/heads/', '')
        branch_slug = sanitize_branch(branch)
        cleanup_preview_async(branch_slug)
        return jsonify({'status': 'cleanup_started', 'branch': branch_slug}), 202

    # MR merge
    if event == 'Merge Request Hook':
        attrs = data.get('object_attributes', {})
        if attrs.get('action') == 'merge':
            branch = attrs.get('source_branch', '')
            branch_slug = sanitize_branch(branch)
            cleanup_preview_async(branch_slug)
            return jsonify({'status': 'cleanup_started', 'branch': branch_slug}), 202

    return jsonify({'status': 'ignored'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9999)
```

---

## 8. DNS & SSL Configuration

### 8.1 Cloudflare DNS Setup

**Strategy: Dynamic A Records via Cloudflare API**

Since `pgai.watch` already has existing dynamic DNS infrastructure, we create/delete A records per preview:
- **On deploy:** Create `preview-{branch-slug}.pgai.watch` → VM IP (proxied)
- **On destroy:** Delete the record

This approach:
- Works with Cloudflare Free/Pro plans
- Doesn't conflict with existing `*.pgai.watch` records
- Allows proxied (orange cloud) for DDoS protection

**Required Cloudflare API credentials** (store in `/opt/postgres-ai-previews/traefik/.env`):
```bash
CF_DNS_API_TOKEN=<your-cloudflare-api-token>  # Needs Zone:DNS:Edit permission
CF_ZONE_ID=<your-zone-id>                      # From Cloudflare dashboard
VM_PUBLIC_IP=<hetzner-vm-ip>
```

### 8.2 Traefik with Wildcard Cert (DNS-01)

```yaml
# /opt/postgres-ai-previews/traefik/docker-compose.yml
version: "3.8"

services:
  traefik:
    image: traefik:v3.0
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
      - "127.0.0.1:8080:8080"  # Dashboard on localhost ONLY
    env_file:
      - .env  # Contains CF_DNS_API_TOKEN
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./traefik.yml:/etc/traefik/traefik.yml:ro
      - ./acme.json:/acme.json
    networks:
      - traefik-public

networks:
  traefik-public:
    name: traefik-public
```

```yaml
# /opt/postgres-ai-previews/traefik/traefik.yml
api:
  dashboard: true
  insecure: true  # Only accessible on localhost:8080

entryPoints:
  web:
    address: ":80"
    http:
      redirections:
        entryPoint:
          to: websecure
          scheme: https
  websecure:
    address: ":443"

providers:
  docker:
    endpoint: "unix:///var/run/docker.sock"
    exposedByDefault: false
    network: traefik-public

certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@postgres.ai
      storage: /acme.json
      dnsChallenge:
        provider: cloudflare
        delayBeforeCheck: 10
        resolvers:
          - "1.1.1.1:53"
          - "8.8.8.8:53"
```

### 8.3 Preview Grafana Labels (Simplified)

```yaml
# In docker-compose.preview.template.yml
services:
  grafana:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.preview-${BRANCH_SLUG}.rule=Host(`preview-${BRANCH_SLUG}.pgai.watch`)"
      - "traefik.http.routers.preview-${BRANCH_SLUG}.entrypoints=websecure"
      - "traefik.http.routers.preview-${BRANCH_SLUG}.tls.certresolver=letsencrypt"
      - "traefik.http.services.preview-${BRANCH_SLUG}.loadbalancer.server.port=3000"
      - "traefik.docker.network=traefik-public"
      - "pgai.preview=true"
```

---

## 9. Resource Planning

### 9.1 Per-Preview Resource Budget

Using Compose v2 native limits (`mem_limit`, `cpus`), NOT `deploy.resources`.

| Service | CPU | Memory | Notes |
|---------|-----|--------|-------|
| Grafana | 0.25 | 256m | |
| VictoriaMetrics | 0.4 | 768m | |
| PGWatch | 0.5 | 512m | |
| Sink PostgreSQL | 0.2 | 512m | |
| Target DB | 0.3 | 512m | |
| Workload Generator | 0.15 | 128m | |
| Flask Backend | 0.1 | 128m | |
| **Total per preview** | **~1.9** | **~2.8GB** | |

### 9.2 Resource Limits in Compose

```yaml
# docker-compose.preview.template.yml (excerpt)
services:
  grafana:
    image: grafana/grafana:12.0.2
    mem_limit: 256m
    cpus: 0.25
    healthcheck:
      test: ["CMD-SHELL", "wget -q --spider http://localhost:3000/api/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    labels:
      - "pgai.preview=true"

  victoria-metrics:
    image: victoriametrics/victoria-metrics:v1.105.0
    mem_limit: 768m
    cpus: 0.4
    command:
      - "-retentionPeriod=72h"
    healthcheck:
      test: ["CMD-SHELL", "wget -q --spider http://localhost:8428/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 3
    labels:
      - "pgai.preview=true"

  target-db:
    image: postgres:15
    mem_limit: 512m
    cpus: 0.3
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 10
      start_period: 10s
    labels:
      - "pgai.preview=true"
```

### 9.3 Scaling Path

| VM Size | vCPU | RAM | Max Previews | Monthly Cost |
|---------|------|-----|--------------|--------------|
| CX31 | 4 | 8GB | 2 | ~€10 |
| CX41 | 8 | 16GB | 4-5 | ~€20 |
| CX51 | 16 | 32GB | 8-10 | ~€40 |

---

## 10. VM Provisioning & Monitoring

### 10.1 VM Naming Convention

**IMPORTANT:** Only ONE Hetzner VM for preview environments is allowed at any time.

- **Naming pattern:** `ralph-preview-XX` (e.g., `ralph-preview-01`)
- **Before provisioning:** Check Hetzner Cloud console for existing `ralph-preview-*` servers
- **If VM exists:** Use the existing one or delete it first (after confirming no active previews)

### 10.2 Initial Setup Script

```bash
#!/bin/bash
# run as root on fresh Hetzner VM (named ralph-preview-XX)

set -euo pipefail

# Verify hostname follows naming convention
if [[ ! "$(hostname)" =~ ^ralph-preview- ]]; then
  echo "WARNING: Hostname should start with 'ralph-preview-'"
  echo "Current hostname: $(hostname)"
fi

# System updates
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker

# Install dependencies (including python3-flask for webhook handler)
apt-get install -y jq rsync python3-flask curl

# Create deploy user
useradd -m -s /bin/bash -G docker deploy
mkdir -p /home/deploy/.ssh
# Add your SSH public key here
echo "ssh-ed25519 AAAA... deploy@ci" >> /home/deploy/.ssh/authorized_keys
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys

# Create swap (prevents OOM kills)
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# Set swappiness to 1 (preferred for DB workloads)
sysctl vm.swappiness=1
echo 'vm.swappiness=1' >> /etc/sysctl.conf

# Docker logging limits (prevents disk exhaustion)
cat > /etc/docker/daemon.json << 'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF
systemctl restart docker

# Create directory structure
mkdir -p /opt/postgres-ai-previews/{traefik,previews,shared/workload,manager,scripts}
touch /opt/postgres-ai-previews/manager/.global.lock
chown -R deploy:deploy /opt/postgres-ai-previews

# Firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo "VM provisioning complete"
echo ""
echo "Next steps:"
echo "1. Add Cloudflare credentials to /opt/postgres-ai-previews/traefik/.env"
echo "2. Deploy Traefik stack"
echo "3. Test DNS record creation"
```

### 10.3 Traefik Initial Setup (Idempotent)

```bash
#!/bin/bash
# /opt/postgres-ai-previews/traefik/setup.sh

set -euo pipefail

cd /opt/postgres-ai-previews/traefik

# Create network if not exists
docker network inspect traefik-public >/dev/null 2>&1 || \
  docker network create traefik-public

# Initialize acme.json with correct permissions
[ -f acme.json ] || (touch acme.json && chmod 600 acme.json)

# Verify .env exists
if [ ! -f .env ]; then
  echo "ERROR: .env file missing. Create it with:"
  echo "  CF_DNS_API_TOKEN=<token>"
  echo "  CF_ZONE_ID=<zone-id>"
  echo "  VM_PUBLIC_IP=<ip>"
  exit 1
fi
chmod 600 .env

# Start or recreate Traefik
docker compose up -d --force-recreate

echo "Traefik setup complete"
```

### 10.4 VM Self-Monitoring (Required)

```yaml
# /opt/postgres-ai-previews/monitoring/docker-compose.yml
version: "3.8"

services:
  node-exporter:
    image: prom/node-exporter:v1.8.2
    restart: unless-stopped
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--path.rootfs=/rootfs'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    ports:
      - "127.0.0.1:9100:9100"
```

**External monitoring (Healthchecks.io or similar):**
```bash
# Add to crontab
* * * * * curl -fsS --retry 3 https://hc-ping.com/YOUR-UUID > /dev/null
```

**Alert thresholds:**
- Disk usage > 80%: Warning
- Disk usage > 90%: Critical
- Memory available < 512MB: Warning
- Load average > 4 (vCPU count): Warning

---

## 11. Implementation Phases

### Phase 1: Infrastructure Setup
- [ ] **Verify no existing preview VM** — Check Hetzner console for any `ralph-preview-*` servers
- [ ] Provision Hetzner CX31 VM named `ralph-preview-01`
- [ ] Run provisioning script (Docker, swap, swappiness, firewall, deploy user, python3-flask)
- [ ] Configure Docker logging limits
- [ ] Set up directory structure with lock files
- [ ] Create Cloudflare API token with Zone:DNS:Edit permission
- [ ] Add credentials to `/opt/postgres-ai-previews/traefik/.env`
- [ ] Run Traefik setup script (idempotent)
- [ ] Test DNS record creation via `cloudflare-dns.sh`
- [ ] Set up VM monitoring (node-exporter + external ping)

**Scripts ready:** `preview-infra/provision-vm.sh`, `preview-infra/traefik/setup.sh`, `preview-infra/traefik/traefik.yml`

### Phase 2: Preview Template
- [x] Create `docker-compose.preview.template.yml`
- [x] Use native Compose resource limits (`mem_limit`, `cpus`)
- [x] Add Traefik labels for routing (simplified, no explicit domain config)
- [x] Add healthchecks to all services
- [x] Add `pgai.preview=true` label to all containers (for quota counting)
- [x] Deploy `scripts/sanitize-branch.sh` (single source of truth)
- [x] Deploy `scripts/cloudflare-dns.sh` (DNS management)
- [x] Create `manager/deploy.sh` with flock locking and quota enforcement
- [x] Create `manager/destroy.sh` with safety checks
- [ ] Test single preview deployment manually (requires VM)

**All code written in:** `preview-infra/shared/`, `preview-infra/scripts/`, `preview-infra/manager/`

### Phase 3: Workload Generator
- [x] Create `pgbench-variable.sh` with graceful shutdown
- [x] Add idempotent schema initialization
- [ ] Test patterns produce interesting metrics (requires running preview)
- [ ] Verify resource usage stays within limits (requires running preview)

**Script ready:** `preview-infra/shared/workload/pgbench-variable.sh`

### Phase 4: CI/CD Integration
- [x] Add CI jobs to `.gitlab-ci.yml`
- [x] Ensure CI calls shared `sanitize-branch.sh` script
- [ ] Configure CI variables (`PREVIEW_SSH_PRIVATE_KEY`, `PREVIEW_VM_HOST`)
- [ ] Test `preview:deploy` manual trigger
- [ ] Test `preview:update` on push
- [ ] Test `preview:destroy`
- [x] Credentials printed to CI logs for testing (will remove for production)

**CI jobs defined:** `preview:deploy`, `preview:update`, `preview:destroy`

### Phase 5: Lifecycle Automation
- [x] Create `cleanup-ttl.sh` cron job script (with conditional prune)
- [ ] Deploy cleanup cron job on VM
- [ ] (Optional) Deploy webhook handler as systemd service
- [ ] Configure GitLab webhook for branch deletion
- [ ] Test 3-day TTL expiry
- [ ] Test branch deletion cleanup

**Script ready:** `preview-infra/manager/cleanup-ttl.sh`

### Phase 6: Documentation
- [x] Update CONTRIBUTING.md with preview usage
- [x] Document how to retrieve credentials (SSH to VM)
- [x] Create runbook for common issues
- [ ] Add preview URL to MR template

---

## 12. Definition of Done

This feature is considered **DONE** when the following acceptance criteria are met:

- [ ] **Sample MR exists** — A test MR is created, branched from the `plan/monitoring-preview-environments` branch (the branch where this system is developed)
- [ ] **CI job visible in pipeline** — The sample MR's pipeline contains the `preview:deploy` job
- [ ] **Manual trigger works** — Clicking the manual trigger on `preview:deploy` successfully provisions a new preview environment
- [ ] **Grafana accessible** — The preview URL (`https://preview-{branch-slug}.pgai.watch`) opens in a browser and shows dashboards (anonymous viewer access enabled)
- [ ] **Dashboards are non-empty** — At least one dashboard shows populated panels (not "No Data")
- [ ] **Charts show activity** — Graphs display metrics indicating the demo database is busy with the sample workload (e.g., transactions per second, query latency, active connections)
- [ ] **Workload patterns visible** — Over a 10-minute observation period, the charts show varying activity levels (spikes, valleys, or ramps) demonstrating the variable workload generator is functioning

**Note:** Anonymous viewer access is enabled (`GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer`) for easy verification. Admin credentials are still available in `.env` for full access.

### Verification Steps

1. Create a test branch from `plan/monitoring-preview-environments`:
   ```bash
   git checkout plan/monitoring-preview-environments
   git checkout -b test/preview-demo
   git push -u origin test/preview-demo
   ```

2. Open the MR pipeline in GitLab and click the play button on `preview:deploy`

3. Wait for deployment to complete (check CI job logs for success message and URL)

4. Open the preview URL in browser, login with credentials from:
   ```bash
   ssh deploy@<PREVIEW_VM_HOST> "cat /opt/postgres-ai-previews/previews/test-preview-demo/.env"
   ```

5. Navigate to dashboards and verify:
   - Panels are loading data (no "No Data" messages)
   - Time-series graphs show activity over the last 5-15 minutes
   - Metrics like `pg_stat_activity`, query rates, or TPS are visible

6. Cleanup: Click `preview:destroy` or delete the test branch

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Multiple VMs provisioned | High | **Only 1 VM allowed** — name must be `ralph-preview-*`, verify in Hetzner console before provisioning |
| Resource exhaustion | High | Hard limit (2 previews), quota checks count running containers, swap |
| Disk exhaustion | High | Conditional prune (>80%), log limits, 72h retention |
| OOM kills | High | `mem_limit` on all containers, swap file, swappiness=1 |
| DNS API failure | Medium | Graceful handling, manual cleanup possible |
| Concurrent deploy races | Medium | flock locking (global + per-preview) |
| Stale previews | Low | 3-day TTL cron, branch deletion webhook |
| Credential leakage | Medium | Stored only in `.env` (chmod 600), never in logs or state.json |
| Docker socket exposure | Low | Read-only mount, acceptable for V1 (socket proxy for V2) |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-23 | Initial draft |
| 1.1.0 | 2026-01-23 | Major revision after 3 expert reviews: fixed CI variable expansion, switched to DNS-01 wildcard cert, added hash suffix for collision prevention, replaced `deploy.resources` with native Compose limits, per-preview credentials, hard quota enforcement, disk/memory checks, rsync instead of registry, graceful shutdown, healthchecks, safety checks in cleanup, Traefik dashboard not exposed, proper pg_isready wait, deployment validation with rollback, VM monitoring as required |
| 1.2.0 | 2026-01-23 | Changed to 3rd level subdomain for Cloudflare wildcard cert compatibility |
| 1.3.0 | 2026-01-23 | Use `preview-{branch}.pgai.watch` - integrates with existing pgai.watch dynamic DNS infrastructure |
| 1.4.0 | 2026-01-23 | Addressing 3 additional expert reviews: **DNS** - switched to dynamic A records via Cloudflare API (create on deploy, delete on destroy); **Secrets** - removed from state.json and logs, stored only in .env with chmod 600; **Locking** - added flock (global + per-preview) for race prevention; **Quota** - counts running containers not directories; **Prune** - conditional, only when disk >80%; **CI** - uses shared sanitize-branch.sh script, fixed environment URL timing; **Provisioning** - added python3-flask, swappiness=1; **Traefik** - simplified TLS config, idempotent setup script |
| 1.5.0 | 2026-01-23 | Added **Definition of Done** section with acceptance criteria: sample MR with CI job, manual trigger provisions preview, Grafana accessible with non-empty dashboards showing workload activity |
| 1.6.0 | 2026-01-23 | Added **VM constraint**: only 1 Hetzner machine allowed, must be named `ralph-preview-*`; added naming convention section and pre-provisioning check |
| 1.7.0 | 2026-01-25 | **Implementation complete**: All scripts written and CI jobs defined. Added `GF_SERVER_ROOT_URL` for Traefik reverse proxy support. Enabled anonymous viewer access (`GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer`) for easier verification. Fixed Grafana password sync by deleting volume on deploy. Updated implementation checklist to reflect completed code work. **Remaining:** VM provisioning, credential setup, and end-to-end testing |
| 1.4.0 | 2026-01-23 | Addressing 3 additional expert reviews: **DNS** - switched to dynamic A records via Cloudflare API (create on deploy, delete on destroy); **Secrets** - removed from state.json and logs, stored only in .env with chmod 600; **Locking** - added flock (global + per-preview) for race prevention; **Quota** - counts running containers not directories; **Prune** - conditional, only when disk >80%; **CI** - uses shared sanitize-branch.sh script, fixed environment URL timing; **Provisioning** - added python3-flask, swappiness=1; **Traefik** - simplified TLS config, idempotent setup script |
| 1.5.0 | 2026-01-23 | Added **Definition of Done** section with acceptance criteria: sample MR with CI job, manual trigger provisions preview, Grafana accessible with non-empty dashboards showing workload activity |
| 1.6.0 | 2026-01-23 | Added **VM constraint**: only 1 Hetzner machine allowed, must be named `ralph-preview-*`; added naming convention section and pre-provisioning check |
