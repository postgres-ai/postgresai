# rds-host-stats: polls CloudWatch and Enhanced Monitoring for one RDS or
# Aurora instance and writes host_* series to VictoriaMetrics. No inbound
# port -- see rds-host-stats/README.md.
FROM oven/bun:1 AS deps
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile --production

FROM oven/bun:1

# Same contract as the sibling images: CI passes both and the build fails
# without them, so a tag never ships unlabelled.
ARG VERSION
RUN test -n "${VERSION}" || (echo "VERSION build arg is required" && exit 1)
ARG BUILD_TS
RUN test -n "${BUILD_TS}" || (echo "BUILD_TS build arg is required" && exit 1)

LABEL org.opencontainers.image.title="PostgresAI RDS host stats"
LABEL org.opencontainers.image.description="Polls RDS/Aurora host metrics from AWS and writes them to VictoriaMetrics"
LABEL org.opencontainers.image.vendor="PostgresAI"
LABEL org.opencontainers.image.source="https://gitlab.com/postgres-ai/postgresai"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_TS}"

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY package.json tsconfig.json ./
COPY bin ./bin
COPY lib ./lib
RUN printf '%s' "${VERSION}" > /VERSION \
  && printf '%s' "${BUILD_TS}" > /BUILD_TS
USER bun
CMD ["bun", "run", "./bin/rds-host-stats.ts"]
