# Docker Infrastructure Review

**Date**: 2026-03-12
**Scope**: docker-compose.yml, docker-compose.local.yml, docker-compose.override.example.yml, all Dockerfiles, config/ directory, Makefile, instances.demo.yml, preview-infra/

---

## 1. Resource Limits

**Verdict: Reasonable, well-documented.**

The header comment in `docker-compose.yml` documents the 4 vCPU / 8 GiB target and explains the intentional CPU overcommit. Summing the allocations:

| Service | CPU | Memory |
|---------|-----|--------|
| target-db | 0.2 | 768 MiB |
| sink-postgres | 0.4 | 1024 MiB |
| sink-prometheus (VM) | 0.75 | 1536 MiB |
| pgwatch-postgres | 0.35 | 512 MiB |
| pgwatch-prometheus | 1.5 | 1024 MiB |
| grafana | 0.5 | 512 MiB |
| monitoring_flask_backend | 0.1 | 192 MiB |
| postgres-reports | 1.0 | 1792 MiB |
| self-cadvisor | 0.15 | 192 MiB |
| self-node-exporter | 0.05 | 96 MiB |
| self-postgres-exporter | 0.1 | 128 MiB |
| **Total** | **5.1** | **7776 MiB** |

The CPU overcommit (5.1 / 4.0 = 1.275x) is acceptable since burst workloads are non-overlapping. The memory total (~7.6 GiB) leaves ~400 MiB for the host OS, which is tight but workable on a dedicated monitoring VM. The comment that pgwatch-prometheus gets 1.5 CPU due to observed pg_stat_statements spikes is a good operational note.

**Suggestion (non-blocking):** If this stack shares the host with other workloads, the memory headroom is slim. Consider documenting "dedicated host" as a requirement or adding `mem_reservation` hints.

---

## 2. Dockerfile Quality

### config/Dockerfile
- Good: Minimal Alpine base, no unnecessary packages, clean layer structure.
- Good: OCI labels, build-arg validation with early `exit 1`.
- Note: Not a multi-stage build, but the image is purely a file-copy container, so multi-stage would add complexity without benefit.

### monitoring_flask_backend/Dockerfile
- **Fixed: `gcc` was left installed after pip build.** Combined the install/build/purge into a single layer to reduce image size.
- **Fixed: `pytest==8.3.4` was in production `requirements.txt`.** Removed -- no tests exist in this directory and pytest should not ship in the production image.
- Good: Uses `python:3.11-slim`, `--no-cache-dir`.
- Note: Not a multi-stage build. For a Flask API this size, single-stage with cleanup is sufficient.
- Observation: `COPY app.py .` copies a single file -- if the app grows, a broader COPY may be needed.

### reporter/Dockerfile
- Good: Clean structure, slim base.
- Minor: `COPY . /app/reporter` copies the entire build context including the Dockerfile itself and any local artifacts. A `.dockerignore` would help, though the practical impact is small.

---

## 3. Health Checks

**Fixed: Added health checks to the main `docker-compose.yml`.**

Previously, the main compose file had zero health checks -- only the preview template (`docker-compose.preview.template.yml`) defined them. This meant that `depends_on` conditions could not use `service_healthy`, leading to race conditions where services would start before their dependencies were actually ready.

Added health checks for:
- `target-db`: `pg_isready -U postgres`
- `sink-postgres`: `pg_isready -U postgres`
- `sink-prometheus`: `wget` against `/metrics` endpoint

Services that lack health checks but are less critical (init containers, exporters, cAdvisor) are fine without them -- they either run-to-completion or are self-recovering.

---

## 4. Startup Ordering

**Fixed: Upgraded `depends_on` conditions from `service_started` to `service_healthy`.**

Before this change, multiple services used `condition: service_started` for their database/metrics-store dependencies, which means they would attempt to connect immediately after the container process started, not after the service was actually accepting connections. This caused race conditions, especially on slower hosts or cold starts.

Changes made:
- `pgwatch-postgres` now waits for `sink-postgres: service_healthy`
- `pgwatch-prometheus` now waits for `sink-prometheus: service_healthy`
- `grafana` now waits for both sinks as `service_healthy`
- `monitoring_flask_backend` now uses structured `depends_on` with `service_healthy` (was using bare list form)
- `postgres-reports` now waits for `sink-prometheus: service_healthy`

The `config-init` -> `sources-generator` -> `pgwatch-*` chain uses `service_completed_successfully`, which is correct for init containers.

**Remaining concern:** `pgwatch-prometheus` does not have a health check itself, so `postgres-reports` still depends on it with `service_started`. pgwatch does expose an HTTP web interface on `:8089`, so a health check could be added if needed, but the 30-minute initial delay in the reporter script mitigates this.

---

## 5. Volume Management

**Verdict: Good.**

- All volumes are **named** (no anonymous volumes), making them easy to identify and manage.
- Five named volumes: `postgres_ai_configs`, `target_db_data`, `sink_postgres_data`, `victoria_metrics_data`, `grafana_data`.
- Config volumes are mounted `:ro` where appropriate (pgwatch, grafana, reports).
- The `postgres_ai_configs` volume is writable only for init containers, read-only for consumers.
- `docker compose down` does NOT remove volumes (requires `-v`), so data persists across restarts.
- The Makefile `down` target does not pass `-v`, which is the safe default.

**Suggestion (non-blocking):** Consider adding a `make clean` target that runs `docker compose down -v` for full teardown.

---

## 6. Config Init Pattern

**Verdict: Robust, with one minor edge case.**

The pattern is:
1. `config-init` (run-to-completion): copies config files from image to shared volume
2. `sources-generator` (run-to-completion): reads `instances.yml`, generates `sources.yml` files via sed substitution
3. All other services mount the volume read-only

Strengths:
- `service_completed_successfully` ensures configs exist before consumers start.
- The generator has a fallback: if `instances.yml` is missing, it writes a demo default.
- The script uses `set -Eeuo pipefail` for strict error handling.

Edge case: If the `postgres_ai_configs` volume already contains stale files from a previous version, `config-init` does `cp -r` which will overwrite files but NOT delete removed files. This is documented in `docker-compose.override.example.yml` as intentional (to preserve generated `sources.yml`). In practice, a `docker compose down -v && up` resolves any staleness. This is acceptable.

---

## 7. Security

### Running as root
- PostgreSQL containers (postgres:15) run as the `postgres` user internally (the official image handles this).
- Grafana runs as the `grafana` user (official image default).
- VictoriaMetrics runs as a non-root user by default.
- `self-cadvisor` runs as **privileged: true** with access to `/`, `/sys`, `/var/run`, `/var/lib/docker`. This is required for container metrics and is a known requirement. Documented in compose file.
- `monitoring_flask_backend` and `reporter` Dockerfiles do not set a `USER` directive, so they run as root inside the container.

**Suggestion (non-blocking):** Add `USER nobody` or create a dedicated user in the Flask backend and reporter Dockerfiles. These services do not need root.

### Secret handling
- Database passwords are hardcoded (`postgres`/`postgres`, `pgwatchadmin`) but only used for internal Docker network communication. No ports are exposed by default except Grafana (3000) and VictoriaMetrics (59090).
- `sink-postgres` uses `POSTGRES_HOST_AUTH_METHOD: trust` for internal container-to-container connections. The `00-configure-pg-hba.sh` script is well-documented explaining why this is acceptable.
- Grafana admin password defaults to `demo` but is configurable via env var.
- VM auth credentials are passed via env vars (standard Docker practice).
- OAuth secrets are passed via env vars (acceptable for Docker Compose; Swarm/K8s would use secrets).
- **Fixed:** Quoted `${PGDATA}` in `00-configure-pg-hba.sh` to prevent word-splitting.

### Exposed ports
- `3000` (Grafana) -- expected, user-facing
- `59090` (VictoriaMetrics) -- useful for debugging; `BIND_HOST` can restrict it
- Local dev overrides bind to `127.0.0.1` -- correct practice

---

## 8. Reproducibility

**Verdict: Good, with one friction point.**

A fresh clone can run the stack with:
```bash
cp .env.example .env
cp instances.demo.yml instances.yml
docker compose up
```

The `.env.example` provides `PGAI_TAG` (required). The `instances.demo.yml` provides a ready-to-use instance config. The `sources-generator` falls back to demo defaults if `instances.yml` is missing.

**Friction point:** `PGAI_TAG` is required (enforced via `${PGAI_TAG:?}`), so running without `.env` fails with an error. This is intentional -- it prevents accidentally running with an unspecified version. The error message is clear.

The `docker-compose.local.yml` overlay handles local development builds well, and `docker-compose.override.example.yml` provides a copy-and-customize workflow. The Makefile targets (`up`, `up-local`, `down`, `logs`) cover the common workflows.

---

## Summary of Changes Made

| File | Change |
|------|--------|
| `docker-compose.yml` | Added health checks for target-db, sink-postgres, sink-prometheus |
| `docker-compose.yml` | Upgraded 5 `depends_on` conditions from `service_started` to `service_healthy` |
| `docker-compose.yml` | Changed monitoring_flask_backend from list-form to structured `depends_on` |
| `monitoring_flask_backend/Dockerfile` | Combined gcc install/purge into single layer; removed gcc from final image |
| `monitoring_flask_backend/requirements.txt` | Removed `pytest==8.3.4` (dev dependency, no tests in this service) |
| `config/sink-postgres/00-configure-pg-hba.sh` | Quoted `${PGDATA}` to prevent word-splitting |

## Open Items (Non-blocking)

1. **Non-root users in Flask/reporter Dockerfiles** -- Both run as root. Adding a `USER` directive would improve defense-in-depth.
2. **`.dockerignore` for reporter** -- `COPY . /app/reporter` includes the Dockerfile and any local artifacts.
3. **`make clean` target** -- Would be useful for full teardown (`docker compose down -v`).
4. **Memory headroom** -- 400 MiB for host OS is tight if other processes run on the same host.
5. **pgwatch health checks** -- pgwatch exposes HTTP on `:8080`/`:8089` which could be used for health checks if downstream services need guaranteed readiness.
