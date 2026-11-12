# syntax=docker/dockerfile:1.7
# pgai-mon image. Build from the repository root (metrics.yml lives outside mon/):
#   docker build -f mon/Dockerfile -t pgai-mon .
# mon/Dockerfile.dockerignore keeps the build context to the files below.

FROM rust:1.94-bookworm AS build
# DuckDB is built from source (duckdb crate feature "bundled"): needs a C++ toolchain (in the
# image) and takes a few minutes on a cold cache.
WORKDIR /src/mon
COPY mon/Cargo.toml mon/Cargo.lock mon/rustfmt.toml ./
COPY mon/src ./src
COPY mon/schema ./schema
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/mon/target \
    cargo build --release --locked --bin pgai-mon \
    && install -D -m 0755 target/release/pgai-mon /out/pgai-mon

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 pgai \
    && useradd --system --uid 10001 --gid pgai --home-dir /var/lib/pgai-mon --shell /usr/sbin/nologin pgai \
    && install -d -o pgai -g pgai -m 0750 /var/lib/pgai-mon
COPY --from=build /out/pgai-mon /usr/local/bin/pgai-mon
COPY config/pgwatch-prometheus/metrics.yml /etc/pgai-mon/metrics.yml
COPY mon/examples/pgai-mon.example.yml /etc/pgai-mon/pgai-mon.yml
COPY mon/sql/macros.sql /usr/share/pgai-mon/macros.sql

USER pgai:pgai
WORKDIR /var/lib/pgai-mon
VOLUME ["/var/lib/pgai-mon"]
# HTTP endpoint (/sql, /schema, /health, ingest). Every route but /healthz needs credentials
# (PGAI_MON_AUTH_* or http.auth; see README).
EXPOSE 8765
ENTRYPOINT ["/usr/local/bin/pgai-mon"]
CMD ["run", "--config", "/etc/pgai-mon/pgai-mon.yml"]
