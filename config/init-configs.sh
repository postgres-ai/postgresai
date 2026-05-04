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

# Copy all configs preserving structure
cp -r "$SOURCE_DIR"/* "$TARGET_DIR/"

echo "$SOURCE_VERSION" > "$TARGET_VERSION_FILE"

echo "Done. Copied:"
find "$TARGET_DIR" -type f | wc -l | xargs echo "  - files:"
find "$TARGET_DIR" -type d | wc -l | xargs echo "  - directories:"

echo ""
echo "Config initialization complete."
