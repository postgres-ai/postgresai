# Patched pgwatch build
#
# Fixes: "unexpected extension X version input: 0.0" error that kills all
# metric gathering when the monitored DB has extensions whose version
# parses to 0 (e.g. supabase-dbdev 0.0.4 → regex extracts "0.0"
# → VersionToInt returns 0 → pgwatch treats it as invalid and aborts).
#
# The one-line fix: skip the extension instead of returning a fatal
# error from FetchRuntimeInfo.
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

# Apply the fix: skip extensions with unparseable versions instead of aborting.
# pgwatch's regex extracts only major.minor from extension versions. For
# extensions like supabase-dbdev (0.0.4), this yields "0.0" which
# VersionToInt() maps to 0 — treated as invalid, killing all metrics.
# Fix: return nil (continue to next extension) instead of a fatal error.
RUN grep -q 'return fmt.Errorf("unexpected extension %s version input: %s", ext, ver)' \
      /pgwatch/internal/sources/conn.go \
    || (echo "ERROR: sed patch target not found in /pgwatch/internal/sources/conn.go — upstream may have changed"; exit 1)
RUN sed -i 's|return fmt.Errorf("unexpected extension %s version input: %s", ext, ver)|return nil /* skip unparseable extension version */|' \
    /pgwatch/internal/sources/conn.go

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
