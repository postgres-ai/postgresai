# syntax=docker/dockerfile:1.7
# pgai-mon image. Build from the repository root:
#   docker build -f mon/Dockerfile -t pgai-mon .
#   docker buildx build --platform linux/amd64,linux/arm64 -f mon/Dockerfile .
# mon/Dockerfile.dockerignore keeps the build context to the files below.
#
# The build stage runs on the BUILD platform and cross-compiles for the target
# one: building DuckDB's C++ under QEMU emulation would take hours.

FROM --platform=$BUILDPLATFORM rust:1.94-bookworm AS build
ARG TARGETARCH
ARG BUILDARCH
# Cross toolchains for whichever architecture is not the build host's. Debian
# bookworm on both sides, so the binary needs the runtime image's glibc 2.36 and
# libstdc++ (DuckDB is C++, built from source: duckdb crate feature "bundled").
RUN set -eux; \
    case "$TARGETARCH" in \
      amd64) target=x86_64-unknown-linux-gnu; pkgs="g++-x86-64-linux-gnu libc6-dev-amd64-cross" ;; \
      arm64) target=aarch64-unknown-linux-gnu; pkgs="g++-aarch64-linux-gnu libc6-dev-arm64-cross" ;; \
      *) echo "unsupported TARGETARCH: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    if [ "$TARGETARCH" != "$BUILDARCH" ]; then \
      apt-get update; \
      apt-get install -y --no-install-recommends $pkgs; \
      rm -rf /var/lib/apt/lists/*; \
    fi; \
    rustup target add "$target"; \
    echo "$target" > /rust-target
# Compiler, archiver and linker per target triple (cc-rs and cargo read these;
# the native toolchain has the same triplet-prefixed names).
ENV CC_x86_64_unknown_linux_gnu=x86_64-linux-gnu-gcc \
    CXX_x86_64_unknown_linux_gnu=x86_64-linux-gnu-g++ \
    AR_x86_64_unknown_linux_gnu=x86_64-linux-gnu-ar \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc \
    CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc \
    CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++ \
    AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
WORKDIR /src/mon
COPY mon/Cargo.toml mon/Cargo.lock mon/rustfmt.toml ./
COPY mon/src ./src
COPY mon/schema ./schema
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,id=pgai-mon-target-${TARGETARCH},target=/src/mon/target \
    target="$(cat /rust-target)" \
    && cargo build --release --locked --target "$target" --bin pgai-mon \
    && install -D -m 0755 "target/$target/release/pgai-mon" /out/pgai-mon

FROM debian:bookworm-slim
ARG VERSION=dev
ARG BUILD_TS=unknown
LABEL org.opencontainers.image.title="pgai-mon" \
      org.opencontainers.image.description="PostgresAI monitoring: collector, Parquet store and SQL endpoint" \
      org.opencontainers.image.vendor="PostgresAI" \
      org.opencontainers.image.source="https://gitlab.com/postgres-ai/postgres_ai" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_TS}"
# ca-certificates: TLS to S3 and to monitored Postgres (rustls reads the system
# roots); libstdc++6: DuckDB. bash: the compose healthcheck (no HTTP client here).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 pgai \
    && useradd --system --uid 10001 --gid pgai --home-dir /var/lib/pgai-mon --shell /usr/sbin/nologin pgai \
    && install -d -o pgai -g pgai -m 0750 /var/lib/pgai-mon
COPY --from=build /out/pgai-mon /usr/local/bin/pgai-mon
COPY mon/metrics.yml /etc/pgai-mon/metrics.yml
COPY mon/docker/pgai-mon.yml /etc/pgai-mon/pgai-mon.yml
COPY mon/examples/pgai-mon.example.yml /usr/share/pgai-mon/pgai-mon.example.yml
COPY mon/sql/macros.sql /usr/share/pgai-mon/macros.sql
# Fails the build if the binary cannot load here (glibc, libstdc++).
RUN /usr/local/bin/pgai-mon --version

USER pgai:pgai
WORKDIR /var/lib/pgai-mon
VOLUME ["/var/lib/pgai-mon"]
# HTTP: /sql, /schema, /health and push ingest need the bearer token; /livez does not.
EXPOSE 8765
ENTRYPOINT ["/usr/local/bin/pgai-mon"]
CMD ["run", "--config", "/etc/pgai-mon/pgai-mon.yml"]
