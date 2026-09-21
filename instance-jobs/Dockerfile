# instance-jobs: the monitoring instance's outbound job runner.
#
# It asks the platform for collection work, runs it against the local metric
# store and posts the results back. No inbound port, no docker socket, no host
# filesystem -- see instance-jobs/README.md.

# ---- Stage 1: build ----
# Pinned to BUILDPLATFORM with Go's own cross-compilation (BuildKit injects
# TARGETOS/TARGETARCH from --platform). CGO is off, so both arches build in
# seconds without QEMU -- the same pattern as pgwatch/Dockerfile.
FROM --platform=$BUILDPLATFORM golang:1.24-alpine AS builder
ARG TARGETOS
ARG TARGETARCH
ARG VERSION

WORKDIR /src
COPY . /src

# No third-party dependencies: `go mod download` would be a no-op and the build
# never reaches the network.
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build \
      -trimpath \
      -ldflags "-s -w -X main.version=${VERSION}" \
      -o /out/instance-jobs ./cmd/instance-jobs

# ---- Stage 2: production image ----
FROM alpine:3.22

ARG VERSION
RUN test -n "${VERSION}" || (echo "VERSION build arg is required" && exit 1)
ARG BUILD_TS
RUN test -n "${BUILD_TS}" || (echo "BUILD_TS build arg is required" && exit 1)

LABEL org.opencontainers.image.title="PostgresAI instance jobs"
LABEL org.opencontainers.image.description="Runs collection jobs on a monitoring instance and posts the results out"
LABEL org.opencontainers.image.vendor="PostgresAI"
LABEL org.opencontainers.image.source="https://gitlab.com/postgres-ai/postgresai"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_TS}"

# ca-certificates: the platform is reached over https.
RUN apk add --no-cache ca-certificates \
  && addgroup -S pgai && adduser -S -G pgai -u 1001 pgai

COPY --from=builder /out/instance-jobs /usr/local/bin/instance-jobs

RUN printf '%s' "${VERSION}" > /VERSION \
  && printf '%s' "${BUILD_TS}" > /BUILD_TS

USER pgai

# The loop records its verdict in a file rather than on a port, because this
# service is defined with no ports at all. `healthcheck` fails whenever the loop
# reports itself unwell -- an unusable config, N failed polls, N failed jobs --
# or has stopped writing altogether, so `mon health` sees a dead channel instead
# of a container that has idled since it started.
HEALTHCHECK --interval=30s --start-interval=5s --timeout=10s --start-period=15s --retries=2 \
  CMD ["/usr/local/bin/instance-jobs", "healthcheck"]

ENTRYPOINT ["/usr/local/bin/instance-jobs"]
