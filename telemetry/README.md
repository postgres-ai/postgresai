# @postgresai/telemetry

Telemetry reporter for PostgresAI monitoring instances. A small TS+Bun
service that runs on each monitoring host and posts an hourly
mini-healthcheck to the platform.

## What it collects

Each tick gathers four signals and POSTs them to the platform:

| Signal | Source |
|---|---|
| OOM events in the lookback window | `journalctl -k --since "<lookback>"` |
| Faulty containers (exited / dead / restarting / unhealthy) | `docker ps -a --format '{{json .}}'` |
| Free RAM | `MemAvailable` from `/proc/meminfo` (falls back to `MemFree`) |
| Free disk | `fs.statfs` on the configured mount |

The companion platform-side hypertable, RPC, alert evaluator, and
dispatcher live in `postgres-ai/platform-all!365`.

## Configuration

| Var | Required | Default |
|---|---|---|
| `PGAI_PLATFORM_API_URL` | yes | — |
| `PGAI_API_TOKEN` | yes | — |
| `PGAI_MONITORING_INSTANCE_ID` | yes | — |
| `PGAI_TELEMETRY_DISK_PATH` | no | `/` |
| `PGAI_TELEMETRY_MEMINFO_PATH` | no | `/proc/meminfo` |
| `PGAI_TELEMETRY_OOM_LOOKBACK` | no | `24 hours ago` |
| `PGAI_TELEMETRY_INTERVAL_SEC` | no | `3600` (min `60`) |

`PGAI_API_TOKEN` is the existing PostgresAI checkup double-base64 token.

## Build and run locally

```sh
cd telemetry
bun install --frozen-lockfile
bun test
bun run typecheck
bun run start          # actually starts reporting
```

## Run in a container

```sh
docker build -t postgresai-telemetry telemetry
docker run --rm \
  -e PGAI_PLATFORM_API_URL=https://postgres.ai/api/v1 \
  -e PGAI_API_TOKEN=... \
  -e PGAI_MONITORING_INSTANCE_ID=... \
  --read-only \
  -v /proc:/host/proc:ro \
  -v /:/host/disk:ro \
  -v /var/run/docker.sock:/var/run/docker.sock \
  postgresai-telemetry
```

## Deployment requirements

The agent must read host kernel logs, host memory, the host filesystem,
and ask the local Docker daemon for its container list. Mount these:

| Host path | Container path | Mode |
|---|---|---|
| `/proc` | `/host/proc` | read-only |
| `/` (or data volume) | `/host/disk` | read-only |
| `/var/run/docker.sock` | `/var/run/docker.sock` | read-write |

## Threat model

Mounting `/var/run/docker.sock` is **root-equivalent on the host**.
Anyone who execs into this container — or compromises any of its
dependencies — can launch privileged containers and take over the
monitoring host.

Recommended mitigations:

- Prefer a docker-socket proxy (e.g.
  [`tecnativa/docker-socket-proxy`](https://github.com/Tecnativa/docker-socket-proxy))
  restricted to `CONTAINERS=1` so only `GET /containers/json` is exposed.
- Drop all Linux capabilities the agent doesn't need
  (`--cap-drop ALL --cap-add ...`).
- Run as a non-root UID inside the container; the `oven/bun` base image
  ships a `bun` user.
- `PGAI_TELEMETRY_MEMINFO_PATH` and `PGAI_TELEMETRY_DISK_PATH` are
  security-sensitive: an actor who can flip them can turn the heartbeat
  into an arbitrary-file-read primitive. Keep them under config-management
  control.

## API contract

`POST /rpc/monitoring_instance_telemetry_report` with a JSON body:

```json
{
  "api_token": "<double-base64 token>",
  "instance_id": "<uuid>",
  "oom_count_24h": 0,
  "faulty_containers": ["cadvisor"],
  "free_ram_bytes": 8589934592,
  "free_disk_bytes": 100000000000,
  "metadata": { "collected_at": "2026-04-28T09:00:00.000Z" }
}
```

All `*_bytes` fields are byte counts. `metadata` is an open-ended
JSON object that today carries `collected_at` (ISO 8601 UTC).

## Operational notes

- **Startup tick**: the agent reports once on startup, then on each
  `PGAI_TELEMETRY_INTERVAL_SEC` boundary.
- **Graceful shutdown**: SIGTERM / SIGINT cancel the in-flight sleep
  immediately. Shutdown latency is bounded by the current tick (not the
  interval).
- **Per-collector failure isolation**: each of the four collectors logs
  a warning on failure and reports a safe default (`0` / `[]`) so a
  single broken signal doesn't silence the heartbeat.
