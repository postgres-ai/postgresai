# Build from the repository root (the dashboards live in mon/grafana/):
#   docker build -f config/Dockerfile --build-arg VERSION=dev --build-arg BUILD_TS=now .
# config/Dockerfile.dockerignore keeps the build context to the files below.
FROM alpine:3.22.0

ARG VERSION
RUN test -n "${VERSION}" || (echo "VERSION build arg is required" && exit 1)
ARG BUILD_TS
RUN test -n "${BUILD_TS}" || (echo "BUILD_TS build arg is required" && exit 1)

LABEL org.opencontainers.image.title="PostgresAI configs: Grafana provisioning and dashboards, demo DB scripts"
LABEL org.opencontainers.image.description="Configuration files for PostgresAI monitoring stack"
LABEL org.opencontainers.image.vendor="PostgresAI"
LABEL org.opencontainers.image.source="https://gitlab.com/postgres-ai/postgres_ai"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_TS}"

# Embed build metadata
RUN printf '%s' "${VERSION}" > /VERSION \
  && printf '%s' "${BUILD_TS}" > /BUILD_TS

# Grafana: provisioning (datasources, dashboard provider, grafana.ini) and the
# pgai-mon SQL dashboards. pgai-mon's own config, metrics.yml and macros ship in
# the pgai-mon image.
COPY config/grafana/provisioning/ /postgres_ai_configs/grafana/provisioning/
COPY mon/grafana/dashboards/ /postgres_ai_configs/grafana/dashboards/
# Demo DB – used only for `mon local-install --demo`
COPY config/target-db/ /postgres_ai_configs/target-db/
# Entrypoints: demo DB, reporter
COPY config/scripts/postgres-entrypoint.sh config/scripts/postgres-reports.sh /postgres_ai_configs/scripts/

# Copy script that initializes volumes
COPY --chmod=755 config/init-configs.sh /init-configs.sh

ENTRYPOINT ["/init-configs.sh"]
