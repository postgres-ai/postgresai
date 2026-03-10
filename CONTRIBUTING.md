# Contributing (Local Development)

This document describes how to run Postgres AI Monitoring locally for development.

## Prerequisites

- Docker + Docker Compose plugin (`docker compose`)
- Git
- Python 3.11+ (for running the reporter on your host)
- Node.js (optional, for `npx postgresai ...` helpers)

## Repo setup

If you cloned the repo with submodules, make sure the `.cursor` submodule is initialized:

```bash
git submodule update --init --recursive
```

### Pre-commit hooks

This repo uses [pre-commit](https://pre-commit.com/) with [gitleaks](https://github.com/gitleaks/gitleaks) to catch secrets before they are committed.

```bash
# Install pre-commit
pip install pre-commit
# or: brew install pre-commit

# Install the hooks (one-time, after cloning)
pre-commit install
```

## Local development workflow (no-commit, debugger-friendly)

This workflow lets you:

- run the monitoring stack via Docker Compose
- iterate on **custom code** without rebuilding images or committing changes
- run the **reporter on your host** (recommended) and debug it
- optionally debug the Flask backend running in Docker

### What runs where (quick mental model)

- **Docker**: pgwatch collectors + sinks + Grafana (+ optional Flask dev container)
- **Host**: `reporter/postgres_reports.py` (recommended for iteration & debugging)

### One-time local setup (no commits)

#### Create `.env` (required)

`docker-compose.yml` requires `PGAI_TAG`. Copy the example and edit as needed:

```bash
cp .env.example .env
# edit .env and set at least PGAI_TAG=...
```

#### Enable the dev override (recommended)

The repo ships an example override file. Copy it to the standard Compose override filename:

```bash
cp docker-compose.override.example.yml docker-compose.override.yml
```

This enables:

- using local `./config/**` (Prometheus/Grafana/pgwatch configs) instead of published config images
- Flask bind-mount + optional debugpy
- exposing `sink-postgres` on localhost for host-run reporter
- (optional) an alternate mode to run the reporter *inside Docker* (commented in the example override). **Host-run reporter is the primary workflow.**

#### Create `.pgwatch-config` (optional but recommended)

The compose stack bind-mounts this file into the reporter container. Create it to avoid Docker creating a directory by accident:

```bash
: > .pgwatch-config
```

Optional (enables uploading reports to PostgresAI):

```bash
echo "api_key=YOUR_KEY" >> .pgwatch-config
```

#### Local-only files are excluded from git

This repo is already configured (locally) to ignore these files via `.git/info/exclude`:

- `.env`
- `.pgwatch-config`
- `docker-compose.override.yml`
- `.vscode/`

That means you can iterate without polluting `git status`.

### Start the stack

`docker compose` automatically loads:

- `docker-compose.yml`
- `docker-compose.override.yml` (if present)

#### Local configs are used (recommended for dev)

This repo includes a `docker-compose.override.yml` that overrides `config-init` to **copy configs from your local working tree** (`./config`) into the shared `postgres_ai_configs` Docker volume.

That means edits to:

- `config/prometheus/prometheus.yml`
- `config/grafana/dashboards/*.json`
- `config/pgwatch-*/metrics.yml`
- `config/scripts/*.sh`

can be picked up locally (see “Applying config changes” below).

#### 1) Choose what Postgres you want to monitor

The pgwatch collectors read targets from `instances.yml`. **By default, the included demo target is disabled**, so you must enable at least one instance.

##### Option A (fastest): monitor the included demo DB (`target-db`)

Edit `instances.yml` and set:

- `is_enabled: true` for `target-database`
- keep `conn_str: postgresql://monitor:monitor_pass@target-db:5432/target_database`

The demo DB is initialized by `config/target-db/init.sql` and creates the `monitor` user/password used above.

##### Option B: monitor your own Postgres instance

- Ensure your DB is reachable **from Docker containers** (network + `pg_hba.conf`).
- Create/prepare a monitoring role (recommended, idempotent):

```bash
PGPASSWORD='...' npx postgresai prepare-db postgresql://admin@host:5432/dbname
```

- Add your instance to `instances.yml` (or use the CLI: `postgresai mon targets add 'postgresql://user:pass@host:port/db' my-db`)
- Make sure `is_enabled: true` for that instance.

#### 2) Generate pgwatch sources from `instances.yml`

This turns `instances.yml` into pgwatch `sources.yml` files inside the shared configs volume:

```bash
docker compose run --rm sources-generator
```

#### 3) Start the stack

Start everything:

```bash
docker compose up -d
```

If you edited `instances.yml` after the stack was already running, restart pgwatch to pick up the new sources:

```bash
docker compose restart pgwatch-postgres pgwatch-prometheus
```

Useful URLs/ports:

- **Grafana**: `http://localhost:3000`
- **VictoriaMetrics (Prometheus API)**: `http://localhost:59090`
- **sink-postgres** (exposed to host by override): `postgresql://pgwatch@127.0.0.1:55433/measurements`

Quick sanity checks:

```bash
docker compose ps
curl -fsS http://localhost:59090/metrics >/dev/null && echo "victoriametrics ok"
curl -fsS http://localhost:3000/api/health || true
```

### Applying config changes (Prometheus/Grafana/pgwatch)

When you change files under `config/`, re-run `config-init` to copy them into the shared volume:

```bash
docker compose up -d --force-recreate config-init
```

Then restart the affected services:

- **Prometheus scrape config changed** (`config/prometheus/prometheus.yml`):

```bash
docker compose restart sink-prometheus
```

- **pgwatch metrics changed** (`config/pgwatch-*/metrics.yml`):

```bash
docker compose restart pgwatch-postgres pgwatch-prometheus
```

- **Grafana dashboards/provisioning changed** (`config/grafana/**`):
  - dashboards are file-provisioned and often reload automatically, but if in doubt:

```bash
docker compose restart grafana
```

### Run reporter on your host (recommended)

#### Install deps (host)

Use whichever venv you prefer; `.venv` is a common convention:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r reporter/requirements.txt
```

#### Run without debugger (prints-only)

Use the helper script:

```bash
./scripts/run_reporter_local.sh
```

Or run directly:

```bash
orig_dir="$(pwd)"
ts="$(date -u +%Y%m%d_%H%M%S)"
out_dir="${orig_dir}/dev_reports/dev_report_${ts}"
mkdir -p "${out_dir}"
cd "${out_dir}"
PYTHONPATH="${orig_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
python -m reporter.postgres_reports \
  --prometheus-url http://127.0.0.1:59090 \
  --postgres-sink-url postgresql://pgwatch@127.0.0.1:55433/measurements \
  --no-upload \
  --output -
echo "Wrote reports to: ${out_dir}"
cd "${orig_dir}"
```

#### Debug on host (Cursor/VS Code)

This repo includes `.vscode/launch.json` with a config:

- **Run Reporter (local)**

Use **Run and Debug** → select **Run Reporter (local)**.

### Debug Flask backend in Docker (optional)

The override file bind-mounts `./monitoring_flask_backend` into the container for fast iteration.

#### Run without debugger

```bash
docker compose up -d --force-recreate monitoring_flask_backend
```

#### Enable debugpy (attach debugger)

```bash
DEBUGPY_FLASK=1 docker compose up -d --force-recreate monitoring_flask_backend
```

Then attach from Cursor/VS Code:

- **Attach (Flask in Docker: debugpy 5678)**

The Flask service (gunicorn) is exposed on:

- `http://localhost:55000`

### (Optional) Debug reporter in Docker

This is usually slower than host-run debugging, but it exists for parity:

```bash
DEBUGPY_REPORTER=1 docker compose up -d --force-recreate postgres-reports
```

Then attach:

- **Attach (Reporter in Docker: debugpy 5679)**

### Troubleshooting

- **Compose says `PGAI_TAG is required`**: set it in `.env` (see above).
- **Host reporter can’t connect to sink-postgres**: make sure the stack is up and `sink-postgres` is bound to `127.0.0.1:55433` (it is in `docker-compose.override.yml`).
- **Need to regenerate pgwatch sources after editing `instances.yml`**:

```bash
docker compose run --rm sources-generator
docker compose restart pgwatch-postgres pgwatch-prometheus
```

## Alternative local development (always rebuild images)

The default `docker-compose.yml` uses published images. For local development you can opt-in to building services from source via `docker-compose.local.yml`.

### Make targets (optional)

```bash
make up
make up-local
```

### Option A: Run via the CLI (recommended)

Use the CLI with `COMPOSE_FILE` to include the local compose override:

```bash
COMPOSE_FILE="docker-compose.yml:docker-compose.local.yml" postgresai mon local-install --demo -y
```

To rebuild on every run:

```bash
COMPOSE_FILE="docker-compose.yml:docker-compose.local.yml" \
  docker compose -f docker-compose.yml -f docker-compose.local.yml build --no-cache

COMPOSE_FILE="docker-compose.yml:docker-compose.local.yml" postgresai mon restart
```

### Option B: Run Docker Compose directly

Bring the stack up and **force rebuild**:

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build --force-recreate
```

If you want to rebuild everything without cache (slow, but deterministic):

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml build --no-cache --pull
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --force-recreate
```

## Common workflows

### Reset everything

```bash
postgresai mon reset
```

### View logs

```bash
postgresai mon logs
postgresai mon logs grafana
postgresai mon logs monitoring_flask_backend
```

### Stop / start

```bash
postgresai mon stop
postgresai mon start
```

## Preview Environments

Preview environments allow you to test monitoring changes in isolated, publicly-accessible deployments before merging.

### Overview

- Each preview runs at `https://preview-{branch-slug}.pgai.watch`
- Includes: PostgreSQL target database, pgwatch collector, VictoriaMetrics, Grafana, and variable workload generator
- Auto-expires after 3 days of inactivity
- Maximum 2 concurrent previews

### Creating a Preview

1. Push your branch to GitLab
2. Open the merge request pipeline
3. Click the **Play** button on `preview:deploy` (manual trigger)
4. Wait for deployment to complete (~2-3 minutes)
5. Access the preview URL shown in the job output

### Accessing Grafana

**URL:** `https://preview-{branch-slug}.pgai.watch`

**Credentials:** The password is generated per-preview and stored on the VM. To retrieve it:

```bash
# SSH to the preview VM (requires access)
ssh deploy@<PREVIEW_VM_HOST> "cat /opt/postgres-ai-previews/previews/{branch-slug}/.env"
```

Username: `monitor`

### Updating a Preview

When you push new commits to a branch with an active preview:

- The `preview:update` job runs automatically
- Grafana dashboards and configs are refreshed
- No need to destroy and recreate

### Destroying a Preview

Previews are automatically cleaned up when:

- The branch is merged or deleted
- The 3-day TTL expires

To manually destroy:

1. Open the merge request pipeline
2. Click `preview:destroy`

### Branch Name Sanitization

Branch names are sanitized for DNS compatibility:

| Original | Sanitized |
|----------|-----------|
| `claude/feature-x` | `claude-feature-x` |
| `feature_test` | `feature-test` |
| `UPPERCASE-Branch` | `uppercase-branch` |

### Troubleshooting Previews

**Preview won't deploy:**
- Check if the maximum concurrent previews limit (2) is reached
- Verify disk/memory on the preview VM

**Grafana shows "No Data":**
- Wait 1-2 minutes for metrics to be collected
- Check if pgwatch container is running

**Can't access the preview URL:**
- DNS propagation may take a few minutes
- Verify the SSL certificate is valid (Let's Encrypt DNS-01 challenge)

