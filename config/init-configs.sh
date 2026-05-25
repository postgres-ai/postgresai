#!/bin/sh
set -e

# PostgresAI Monitoring Config Initializer
# Copies configuration files to mounted volumes

# Allow overriding the version source for testability; defaults to image /VERSION.
VERSION_FILE="${VERSION_FILE:-/VERSION}"
BUILD_TS_FILE="${BUILD_TS_FILE:-/BUILD_TS}"

SOURCE_VERSION="$(cat "$VERSION_FILE" 2>/dev/null || echo unknown)"
SOURCE_BUILD_TS="$(cat "$BUILD_TS_FILE" 2>/dev/null || echo unknown)"

echo "PostgresAI configs v${SOURCE_VERSION}"
echo "Build: ${SOURCE_BUILD_TS}"
echo ""

# Default target is /target, can be overridden
TARGET_DIR="${TARGET_DIR:-/target}"
SOURCE_DIR="${SOURCE_DIR:-/postgres_ai_configs}"

if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Target directory $TARGET_DIR does not exist"
  exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: Source directory $SOURCE_DIR does not exist"
  exit 1
fi

# Idempotency guard: short-circuit when the target was already populated by
# init-configs at the same image version. This preserves user edits to
# volume-backed config files (e.g. pgwatch metrics.yml) across routine
# `docker compose up` invocations. To force re-initialization (e.g. to pull
# new image defaults without recreating the volume), remove the marker file:
#   docker exec config-init rm /target/.pgai-configs-version
TARGET_VERSION_FILE="${TARGET_DIR}/.pgai-configs-version"

if [ -f "$TARGET_VERSION_FILE" ] && \
   [ "$(cat "$TARGET_VERSION_FILE" 2>/dev/null)" = "$SOURCE_VERSION" ]; then
  echo "Configs already initialized at version $SOURCE_VERSION; skipping."
  exit 0
fi

echo "Initializing configs (target version: $SOURCE_VERSION)..."

# Grafana dashboards are managed by file-based provisioning and must mirror the
# image exactly: stale files cause "the same UID is used more than once" warnings
# that block the provisioner from writing ANY dashboard. cp -r only adds/overwrites,
# so an in-place upgrade where a dashboard was renamed (e.g. 0.14 ->
# 0.15: Dashboard_7_Autovacuum_and_bloat.json -> Dashboard_7_Autovacuum_and_xmin_horizon.json,
# both with the same top-level uid) would leave both files in the volume and
# trigger a collision. Clean the dashboards directory before re-copy. We only
# clean dashboards/ - other config files (pgwatch metrics.yml, etc.) may have
# been edited by the operator and must be preserved.
DASHBOARDS_DIR="${TARGET_DIR}/grafana/dashboards"
if [ -d "$DASHBOARDS_DIR" ]; then
  echo "Cleaning stale dashboards from $DASHBOARDS_DIR..."
  find "$DASHBOARDS_DIR" -mindepth 1 -delete
fi

# Copy all configs preserving structure
cp -r "$SOURCE_DIR"/* "$TARGET_DIR/"

echo "$SOURCE_VERSION" > "$TARGET_VERSION_FILE"

echo "Done. Copied:"
find "$TARGET_DIR" -type f | wc -l | xargs echo "  - files:"
find "$TARGET_DIR" -type d | wc -l | xargs echo "  - directories:"

echo ""
echo "Config initialization complete."
