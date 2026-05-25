# Patched pgwatch build
#
# Fixes:
#   1. "unexpected extension X version input: 0.0" error that kills all metric
#      gathering when the monitored DB has extensions whose version parses to 0
#      (e.g. supabase-dbdev 0.0.4 → regex extracts "0.0" → VersionToInt returns
#      0 → pgwatch treats it as invalid and aborts). The one-line fix: skip the
#      extension instead of returning a fatal error from FetchRuntimeInfo.
#
#   2. Prometheus sink wipes its per-DB metric cache on every scrape (regression
#      introduced upstream in v3.6.0, commit fb7abf39 / PR #790). This turns the
#      /pgwatch endpoint into a transient drain: scrapes between collector polls
#      return zero pg metrics; scrapes after multiple polls return the union and
#      can exceed Prometheus sample_limit. Both halves are the same bug. Fix:
#      remove the 3-line wipe so the cache holds the latest sample per metric
#      until the next poll overwrites it. See gitlab.com/postgres-ai/postgresai#195.
#
# Based on: cybertec-postgresql/pgwatch v3.7.0

# ---- Stage 1: build WebUI ----
# Pin to BUILDPLATFORM so the WebUI build (yarn install + react-scripts build)
# runs on the runner's native arch. The output is static JS/CSS/HTML — fully
# arch-independent — so a single build serves both linux/amd64 and linux/arm64
# final images. Without this, `docker buildx --platform linux/amd64,linux/arm64`
# would run the WebUI build under QEMU emulation for arm64, taking 30+ min and
# overrunning the GitLab runner's 1h job timeout.
FROM --platform=$BUILDPLATFORM node:22 AS uibuilder

# Pin to the exact commit SHA of the v3.7.0 tag in upstream. Tags are
# mutable (force-push etc.), so we verify the checkout resolves to this SHA
# and fail the build if it ever drifts.
ARG PGWATCH_VERSION=v3.7.0
ARG PGWATCH_SHA=2995dbec0486dea5c5e7dcd502b94fbafbbe2fa5

RUN git clone --depth 1 --branch "${PGWATCH_VERSION}" \
      https://github.com/cybertec-postgresql/pgwatch.git /src \
    && actual_sha="$(git -C /src rev-parse HEAD)" \
    && [ "${actual_sha}" = "${PGWATCH_SHA}" ] \
      || (echo "ERROR: pgwatch ${PGWATCH_VERSION} resolved to ${actual_sha}, expected ${PGWATCH_SHA}"; exit 1)

RUN cd /src/internal/webui && yarn install --network-timeout 100000 && yarn build

# ---- Stage 2: patch & build Go binary ----
# Pin to BUILDPLATFORM and use Go's native cross-compilation via GOOS/GOARCH
# (BuildKit injects TARGETOS/TARGETARCH from the --platform flag). Cgo is
# already disabled below, so cross-compile produces a working static binary
# in seconds for any target arch — no QEMU, no toolchain setup.
FROM --platform=$BUILDPLATFORM golang:1.24 AS builder
ARG TARGETOS
ARG TARGETARCH

COPY --from=uibuilder /src /pgwatch
COPY --from=uibuilder /src/internal/webui/build /pgwatch/internal/webui/build

# Patch 1: skip extensions with unparseable versions instead of aborting.
# pgwatch's regex extracts only major.minor from extension versions. For
# extensions like supabase-dbdev (0.0.4), this yields "0.0" which
# VersionToInt() maps to 0 — treated as invalid, killing all metrics.
# Fix: return nil (continue to next extension) instead of a fatal error.
RUN grep -q 'return fmt.Errorf("unexpected extension %s version input: %s", ext, ver)' \
      /pgwatch/internal/sources/conn.go \
    || (echo "ERROR: sed patch target not found in /pgwatch/internal/sources/conn.go — upstream may have changed"; exit 1)
RUN sed -i 's|return fmt.Errorf("unexpected extension %s version input: %s", ext, ver)|return nil /* skip unparseable extension version */|' \
    /pgwatch/internal/sources/conn.go

# Patch 2: keep the Prometheus sink's per-DB metric cache across scrapes.
# Upstream commit fb7abf39 (v3.6.0, "improve Prometheus scrapping (#790)") added
# a wipe of promAsyncMetricCache[dbname] at the end of every Collect(), turning
# the cache into a transient drain. Effect: VM scrapes landing between collector
# polls return zero pg metrics (empty Grafana); scrapes landing after multiple
# poll cycles return the union of all of them at once and routinely exceed the
# configured Prometheus sample_limit (rejected scrapes). Both halves are the
# same bug.
# Fix: remove the 3-line wipe so the cache holds the latest sample per metric
# until overwritten by the next poll. Samples are emitted with their original
# collection epoch via NewMetricWithTimestamp(), so VM deduplicates repeats at
# storage time — re-emitting the same (metric, timestamp) across scrapes is a
# no-op. The 10-min promScrapingStalenessHardDropLimit guard in
# MetricStoreMessageToPromMetrics already covers the "collection stalled, stop
# emitting" case the wipe was defending against.
# Refs: gitlab.com/postgres-ai/postgresai#195
RUN grep -Fq 'clear the cache for this db after metrics are collected' \
      /pgwatch/internal/sinks/prometheus.go \
    || (echo "ERROR: drain patch target not found in /pgwatch/internal/sinks/prometheus.go — upstream may have changed"; exit 1)
RUN grep -Fq 'for dbname, metricsMessages := range promAsyncMetricCache' \
      /pgwatch/internal/sinks/prometheus.go \
    || (echo "ERROR: drain patch loop header not found in /pgwatch/internal/sinks/prometheus.go — upstream may have changed"; exit 1)
# Remove the 3-line wipe (Lock + assignment-with-unique-comment + Unlock).
RUN sed -i '/promAsyncMetricCacheLock\.Lock()$/{N;N;/clear the cache for this db after metrics are collected/d;}' \
    /pgwatch/internal/sinks/prometheus.go
# Removing the wipe leaves the `dbname` loop variable unused — rename to `_`.
RUN sed -i 's|for dbname, metricsMessages := range promAsyncMetricCache|for _, metricsMessages := range promAsyncMetricCache|' \
    /pgwatch/internal/sinks/prometheus.go
RUN ! grep -Fq 'clear the cache for this db after metrics are collected' /pgwatch/internal/sinks/prometheus.go \
    || (echo "ERROR: drain patch applied but wipe comment still present in /pgwatch/internal/sinks/prometheus.go"; exit 1)
RUN ! grep -Fq 'for dbname, metricsMessages := range promAsyncMetricCache' /pgwatch/internal/sinks/prometheus.go \
    || (echo "ERROR: drain patch applied but dbname loop var still present in /pgwatch/internal/sinks/prometheus.go"; exit 1)

RUN cd /pgwatch && CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH go build \
      -ldflags "-X 'main.version=3.7.0-patched'" \
      ./cmd/pgwatch

# ---- Stage 3: production image ----
FROM alpine:3.22

RUN addgroup -S pgwatch && adduser -S -G pgwatch -u 1001 pgwatch

COPY --from=builder --chown=pgwatch:pgwatch /pgwatch/pgwatch /pgwatch/
COPY --from=builder --chown=pgwatch:pgwatch /pgwatch/internal/metrics/metrics.yaml /pgwatch/metrics/metrics.yaml

USER pgwatch

EXPOSE 8080

ENTRYPOINT ["/pgwatch/pgwatch"]
