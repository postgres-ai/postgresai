FROM oven/bun:1 AS deps
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

FROM oven/bun:1
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY package.json tsconfig.json ./
COPY bin ./bin
COPY lib ./lib

# /proc and /var/lib/docker (or /) need to be visible from inside the container.
# Compose / k8s manifest must mount:
#   - /proc            -> /host/proc    (readonly)
#   - / (or data vol)  -> /host/disk    (readonly)
#   - /var/run/docker.sock -> /var/run/docker.sock  (so `docker ps` works)
#
# Mounting the docker socket grants root-equivalent access to the host. Where
# possible, prefer a docker-socket proxy restricted to GET /containers/json.
# See telemetry/README.md for the deployment threat model.
ENV PGAI_TELEMETRY_MEMINFO_PATH=/host/proc/meminfo \
    PGAI_TELEMETRY_DISK_PATH=/host/disk

CMD ["bun", "run", "./bin/telemetry.ts"]
