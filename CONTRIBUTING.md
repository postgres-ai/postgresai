# Contributing (local development)

This document describes how to run postgres_ai monitoring locally for development.

## Prerequisites

- Docker + Docker Compose plugin (`docker compose`)
- Git
- Bun 1.x (for the CLI workflow; CI uses `oven/bun:1`). Install with `curl -fsSL https://bun.sh/install | bash`.
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

### What runs where (quick mental model)

- **Docker**: pgai-mon (collection, storage, SQL) + Grafana + the reporter
- **Host**: `reporter/postgres_reports.py` (recommended for iteration & debugging)
- **Not started by default**: `instance-jobs`, behind a compose profile of the
  same name (see `instance-jobs/README.md`). Its tests need no stack at all:

  ```bash
  cd instance-jobs && go vet ./... && go test ./... -race
  ```

  To run the container from your working tree, build the image under the exact
  name compose resolves — registry and tag both come from `.env`, which compose
  reads by itself but your shell does not:

  ```bash
  [ -f .env ] || cp .env.example .env   # the one-time setup below, if you have not done it
  set -a; . ./.env; set +a
  docker build --build-arg VERSION="$PGAI_TAG" \
    --build-arg BUILD_TS="$(date -u '+%Y-%m-%d %H:%M:%S UTC')" \
    -t "${PGAI_REGISTRY:-postgresai}/instance-jobs:$PGAI_TAG" instance-jobs
  docker compose --profile instance-jobs up -d instance-jobs
  ```

  `--profile` here is a one-off for your shell. On a real box the profile is a
  value in `.env` (`COMPOSE_PROFILES=instance-jobs`, written by
  `postgresai mon local-install --instance-jobs`), so every later compose
  command covers the service without the flag.

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

- using the local Grafana provisioning (`config/grafana/provisioning/`), dashboards
  (`mon/grafana/dashboards/`) and demo-DB scripts instead of the published config image
- (optional, commented) running pgai-mon from a locally built image or your own config
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

The override overrides `config-init` to **copy configs from your local working tree** (`./config` and `mon/grafana/dashboards`) into the shared `postgres_ai_configs` Docker volume.

That means edits to:

- `config/grafana/provisioning/**`
- `mon/grafana/dashboards/*.json`
- `config/target-db/**`
- `config/scripts/*.sh`

can be picked up locally (see “Applying config changes” below).

#### 1) Choose what Postgres you want to monitor

pgai-mon reads targets from `instances.yml`. **By default, the included demo target is disabled**, so you must enable at least one instance.

##### Option A (fastest): monitor the included demo DB (`target-db`)

Edit `instances.yml` and set:

- `is_enabled: true` for `target-database`
- keep `conn_str: postgresql://monitor:monitor_pass@target-db:5432/target_database`

The demo DB is initialized by `config/target-db/init.sql` and creates the `monitor` user/password used above.

The demo DB containers (`target-db`, `target-standby`) belong to the `demo` compose profile, so turn it on before starting the stack: add `COMPOSE_PROFILES=demo` to `.env`, or pass `--profile demo` to `docker compose`.

##### Option B: monitor your own Postgres instance

- Ensure your DB is reachable **from Docker containers** (network + `pg_hba.conf`).
- Create/prepare a monitoring role (recommended, idempotent):

```bash
PGPASSWORD='...' npx postgresai prepare-db postgresql://admin@host:5432/dbname
```

- Add your instance to `instances.yml` (or use the CLI: `postgresai mon targets add 'postgresql://user:pass@host:port/db' my-db`)
- Make sure `is_enabled: true` for that instance.

#### 2) Start the stack

Start everything:

```bash
docker compose up -d
```

pgai-mon picks up a changed `instances.yml` by itself:

- **Edited in place** (an editor that writes the file in place, `cat > instances.yml`, `postgresai mon targets add/remove`): re-read within 30 s, no restart.
- **Replaced** (`sed -i`, Ansible's `template` or `copy`, `mv`, editors that save to a new file): compose mounts the file on its own, and such a mount keeps the old file. pgai-mon notices within 30 s, logs why, and exits; Docker restarts it with the new file. The reporter does the same before its next run, and instance-jobs (`.pgwatch-config`) before its next poll. A job that replaces the file on every run (a cron `sed -i`) restarts pgai-mon every time.
- The new file must stay readable by the install user: a `sudo mv` of a root-owned 0600 file leaves pgai-mon collecting nothing (its log names the file) until the owner is fixed.
- **Removed**: the services that mount it stop at their next check, and Docker creates a root-owned directory in its place. Remove that directory (`sudo rmdir instances.yml`), restore the file, then `docker compose up -d`. (`postgresai mon targets add` also replaces the directory, with a file holding only the target it adds; `docker compose up -d` is still needed.)

Useful URLs/ports:

- **Grafana**: `http://localhost:3000`
- **pgai-mon** (read-only SQL at `POST /sql`, bearer token `PGAI_MON_TOKEN` from `.env`): `http://127.0.0.1:8765`

Quick sanity checks:

```bash
docker compose ps
curl -fsS http://127.0.0.1:8765/livez >/dev/null && echo "pgai-mon ok"
curl -fsS http://localhost:3000/api/health || true
```

### Applying config changes (Grafana, demo DB, scripts)

When you change files under `config/` or `mon/grafana/dashboards/`, re-run `config-init` to copy them into the shared volume:

```bash
docker compose up -d --force-recreate config-init
```

Then restart the affected services:

- **Collected metrics changed** (`mon/metrics.yml`): the file is baked into the
  pgai-mon image; rebuild it (see "Option A" below) and recreate the service.

- **Grafana dashboards/provisioning changed** (`mon/grafana/dashboards/**`, `config/grafana/provisioning/**`):
  - dashboards are file-provisioned and often reload automatically, but if in doubt:

```bash
docker compose restart grafana
```

#### Dashboard conventions (read before editing a dashboard)

`mon/grafana/dashboards/README.md` is the source of truth for tags, panel
titles, terminology, units, and **legend sorting** (rate panels sort by `Mean`,
per-call/latency by `Max`, levels by `Last`/`Last *` — always descending).
Consistency with the existing dashboards comes first: see README.md §
"Consistency first".

`tests/grafana_dashboards/` enforces the mechanical parts on every MR via the
`quality:grafana-dashboards-lint` job. Run it locally before pushing:

```bash
python -m pytest tests/grafana_dashboards -q
```

There is no Helm chart for the `pgai-mon` stack. The chart of the previous stack is under
`old/postgres_ai_helm/`; it is not built, tested or released. Its
`config/grafana/dashboards/*.json` are symlinks to `old/config/grafana/dashboards/`, which is
the input of `mon/grafana/compile.py`: edit the canonical file, never write through the link.

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
export PGAI_MON_TOKEN="$(sed -n 's/^PGAI_MON_TOKEN=//p' "${orig_dir}/.env" | tail -n 1)"
PYTHONPATH="${orig_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
python -m reporter.postgres_reports \
  --mon-url http://127.0.0.1:8765 \
  --sources-path "${orig_dir}/instances.yml" \
  --no-upload \
  --output -
echo "Wrote reports to: ${out_dir}"
cd "${orig_dir}"
```

#### Debug on host (Cursor/VS Code)

This repo includes `.vscode/launch.json` with a config:

- **Run Reporter (local)**

Use **Run and Debug** → select **Run Reporter (local)**.

### (Optional) Debug reporter in Docker

This is usually slower than host-run debugging, but it exists for parity:

```bash
DEBUGPY_REPORTER=1 docker compose up -d --force-recreate postgres-reports
```

Then attach:

- **Attach (Reporter in Docker: debugpy 5679)**

### Troubleshooting

- **Compose says `PGAI_TAG is required`**: set it in `.env` (see above).
- **Host reporter can’t connect to pgai-mon**: make sure the stack is up (`docker-compose.yml` publishes pgai-mon on `127.0.0.1:8765`) and `PGAI_MON_TOKEN` matches the one in `.env`.
- **Edited `instances.yml`**: pgai-mon reloads it by itself (see above); `docker compose logs pgai-mon` shows the new source, or the restart when the file was replaced.

## Alternative local development (always rebuild images)

The default `docker-compose.yml` uses published images. For local development you can opt-in to building services from source via `docker-compose.local.yml`.

### Make targets (optional)

```bash
make up
make up-local
```

### Option A: Run via the CLI (recommended)

The CLI always runs `docker compose -f docker-compose.yml` in the project
directory, so `COMPOSE_FILE` and `docker-compose.local.yml` do not apply to it.
Instead, build the images under the names `docker-compose.yml` resolves
(`postgresai/<image>:$PGAI_TAG`; the CLI writes `PGAI_TAG` = the version in
`cli/package.json`, `0.0.0-dev.0` in a checkout) and run the CLI from the
repository root. The default install needs these three project images; build the fourth,
`instance-jobs`, only when using `--instance-jobs` (see above). A cold `pgai-mon`
build takes about 20 minutes on a 2 vCPU box (it compiles DuckDB).

```bash
cd cli && bun install && bun run embed-all && cd ..   # generated files are not in git
TAG=$(sed -n 's/.*"version": "\(.*\)".*/\1/p' cli/package.json)
TS="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
docker build -f config/Dockerfile --build-arg VERSION="$TAG" --build-arg BUILD_TS="$TS" -t "postgresai/postgres-ai-configs:$TAG" .
docker build -f mon/Dockerfile -t "postgresai/pgai-mon:$TAG" .
docker build --build-arg VERSION="$TAG" --build-arg BUILD_TS="$TS" -t "postgresai/reporter:$TAG" reporter
bun cli/bin/postgres-ai.ts mon local-install --demo -y
```

Run it inside a git checkout: in a tree without `.git` (a tarball, `git archive`)
the CLI replaces `docker-compose.yml` with the one from GitLab. Configs built
with a `0.0.0-dev*` version are re-provisioned on every start, so rebuilt
dashboards reach Grafana without removing the configs volume. After a rebuild,
`docker compose up -d --force-recreate` (from the repository root, which holds
the `.env` the install wrote) puts the new images in place; `mon restart` does
not, as it restarts the existing containers.

The CLI keeps the Grafana admin password in `.pgwatch-config`, not in `.env`, and
the compose file needs it for every command (`down` and `ps` too). For plain
`docker compose` commands on a stack the CLI installed, export it first:

```bash
export GF_SECURITY_ADMIN_PASSWORD="$(sed -n 's/^grafana_password=//p' .pgwatch-config)"
```

#### Monitoring v4 beside the old stack on one host

You can run the branch beside an installed old stack (`postgresai@0.18.0`, pgwatch +
VictoriaMetrics) and point both at the same database. The containers of monitoring v4 have
no fixed names: compose names them after the project (`<project>-<service>-1`), and the
volumes and the network too (`<project>_<name>`). So a second project name and a second
Grafana port are enough. The old stack keeps its fixed container names, Grafana on port
3000 and its project `monitoring` (its directory is `~/.config/postgresai/monitoring`).

1. Leave the old stack as it is. Run its commands from outside the checkout, for example
   `cd ~ && npx postgresai@0.18.0 mon health`. Inside the checkout the CLI uses the
   checkout's `docker-compose.yml`.
2. In the checkout root, before the first install, set the project name and the ports in
   `.env`. `mon local-install` keeps these keys:

   ```bash
   printf '%s\n' COMPOSE_PROJECT_NAME=pgai-v4 GRAFANA_PORT=3001 >> .env
   ```

   The old stack does not publish 8765, so pgai-mon can keep it. For a third stack, set
   `PGAI_MON_PORT` too.
3. Build the images and install as in Option A above, with the database the old stack
   monitors:

   ```bash
   bun cli/bin/postgres-ai.ts mon local-install --db-url 'postgresql://postgres_ai_mon:<password>@<host>:5432/<db>' -y
   ```

4. Check that both run: `docker ps --format '{{.Names}}\t{{.Ports}}'` lists the old
   stack's `grafana-with-datasources` on port 3000 and `pgai-v4-grafana-1` on
   127.0.0.1:3001. Monitoring v4 is at http://localhost:3001, the old stack at
   http://localhost:3000. `bun cli/bin/postgres-ai.ts mon health` (from the checkout) looks
   only at the `pgai-v4` containers.
5. To remove monitoring v4 and keep the old stack, run `bun cli/bin/postgres-ai.ts mon
   reset` from the checkout and answer `y`. It removes the containers, volumes and network
   of the `pgai-v4` project only. `mon clean` there also keeps all historical
   `monitoring_*` and `postgres_ai_configs_*` volumes, even if the other install is
   stopped, and prints the command to remove them on purpose. With either default
   project name (`monitoring` or `postgres_ai_configs`), `mon clean` removes listed
   historical volumes from both names. It cleans only this project's unused networks;
   of images it removes dangling ones only. `--keep-volumes` keeps all volumes.

Host scripts read `PGAI_MON_PORT` and `GRAFANA_PORT` from the environment, then the
project's `.env`, with defaults 8765 and 3000. `scripts/run_reporter_local.sh` keeps
an explicit `PGAI_MON_URL`; `tests/e2e.cli.sh` accepts `GRAFANA_URL` and otherwise uses
`DOCKER_HOST` for Grafana's host.

Monitoring v4 defaults to 2.0 CPUs for pgai-mon (`PGAI_MON_CPUS`), matching the combined budget
of pgwatch-postgres, pgwatch-prometheus, sink-prometheus and sink-postgres. Automatic query
concurrency honors Linux cgroup CPU quotas: normally two slots with this limit. Grafana defaults
to 768 MiB (`GRAFANA_MEM=805306368`). Override these limits in `.env`.

Both stacks collect from the database, so it serves two collectors. Their memory limits
add up (about 4.7 GiB for monitoring v4 and 7.3 GiB for the old stack); the host needs
room for both.

### Option B: Run Docker Compose directly

An Option A install prepares everything below. Without one, do it by hand from the
repository root before the first `docker compose up`:

```bash
cp .env.example .env                 # then fill in the three required values (below)
cp instances.demo.yml instances.yml  # the targets pgai-mon collects; edit for your own
mkdir -p host-metrics                # scrape files for host metrics; may stay empty
touch .pgwatch-config                # read by the reporter; may stay empty
```

`.env` must set `PGAI_TAG`, `PGAI_MON_TOKEN` (`openssl rand -hex 32`) and
`GF_SECURITY_ADMIN_PASSWORD`; compose stops with a message while one is missing. The demo
databases start only with `COMPOSE_PROFILES=demo`, which also needs `REPLICATOR_PASSWORD`.
Compose bind-mounts `instances.yml`, `host-metrics/` and `.pgwatch-config`; if a path is
missing, Docker creates a directory in its place and the containers fail to read it.

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
postgresai mon logs pgai-mon
```

### Stop / start

```bash
postgresai mon stop
postgresai mon start
```

## Preview environments

Preview environments let you test monitoring changes in isolated, publicly accessible deployments before merging.

### Overview

- Each preview runs at `https://preview-{branch-slug}.pgai.watch`
- Includes: PostgreSQL target database, pgai-mon, Grafana, the reporter, and variable workload generator
- Auto-expires after 3 days of inactivity
- Maximum 2 concurrent previews

### Creating a preview

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

### Updating a preview

When you push new commits to a branch with an active preview:

- The `preview:update` job runs automatically
- Grafana dashboards and configs are refreshed
- No need to destroy and recreate

### Destroying a preview

Previews are automatically cleaned up when:

- The branch is merged or deleted
- The 3-day TTL expires

To manually destroy:

1. Open the merge request pipeline
2. Click `preview:destroy`

### Branch name sanitization

Branch names are sanitized for DNS compatibility:

| Original | Sanitized |
|----------|-----------|
| `claude/feature-x` | `claude-feature-x` |
| `feature_test` | `feature-test` |
| `UPPERCASE-Branch` | `uppercase-branch` |

### Troubleshooting previews

**Preview won't deploy:**
- Check if the maximum concurrent previews limit (2) is reached
- Verify disk/memory on the preview VM

**Grafana shows "No Data":**
- Wait 1-2 minutes for metrics to be collected
- Check if the pgai-mon container is running

**Can't access the preview URL:**
- DNS propagation may take a few minutes
- Verify the SSL certificate is valid (Let's Encrypt DNS-01 challenge)

## Wait-event colors

Follow [docs/COLOR_SCHEME.md](docs/COLOR_SCHEME.md) for monitoring charts and
all supporting materials. Use the canonical palette, regenerate dashboard
overrides with `python3 scripts/sync-wait-event-colors.py`, and run the dashboard
lint tests. Include focused before/after screenshots for visual changes.
