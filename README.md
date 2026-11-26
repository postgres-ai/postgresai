<div align="center">

# postgresai

### AI-native PostgreSQL observability

[![npm](https://img.shields.io/npm/v/postgresai?style=flat-square&logo=npm)](https://www.npmjs.com/package/postgresai)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](LICENSE)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14--18-336791?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![CLI Coverage](https://img.shields.io/gitlab/pipeline-coverage/postgres-ai%2Fpostgresai?branch=main&job_name=cli%3Anode%3Atests&label=CLI%20coverage&style=flat-square)](https://gitlab.com/postgres-ai/postgresai/-/pipelines)
[![Reporter Coverage](https://img.shields.io/gitlab/pipeline-coverage/postgres-ai%2Fpostgresai?branch=main&job_name=reporter%3Atests&label=Reporter%20coverage&style=flat-square)](https://gitlab.com/postgres-ai/postgresai/-/pipelines)

**Monitoring, health checks, and root cause analysis — built for humans and AI agents**

[Live Demo](https://demo.postgres.ai) ·
[Documentation](https://postgres.ai/docs) ·
[Get started](#get-started)

</div>

<div align="center">
<img src="assets/postgresai.png" alt="postgresai" width="500">
</div>


Battle-tested in PostgresAI team's work with [companies like GitLab, Miro, Chewy, Suno, Supabase, Gadget, and more](https://postgres.ai) — now packaged for easy use by humans and AI agents.

## Why postgresai?

Traditional monitoring tools give you dashboards. **`postgresai` gives AI agents the context they need to actually fix problems.**

- **Structured for AI** — Reports and metrics designed for LLM consumption
- **Issues workflow** — Track problems from detection to resolution
- **45+ health checks** — Bloat, indexes, queries, settings, security
- **Active Session History** — Postgres's answer to Oracle ASH
- **Expert dashboards** — Four Golden Signals methodology

Part of [Self-Driving Postgres](https://postgres.ai/blog/20250725-self-driving-postgres) — PostgresAI's open-source initiative to make Postgres autonomous.

## Get started

### 1. Sign up

Create a free account at [postgres.ai](https://postgres.ai)

### 2. Authenticate

```bash
npx postgresai auth
```

This opens your browser to log in and saves your API key locally.

### 3. Run health checks

```bash
PGPASSWORD=secret npx postgresai checkup postgresql://user@host:5432/dbname
```

### 4. View results

Open [console.postgres.ai](https://console.postgres.ai) to see:
- Detailed reports with suggested fixes
- Issues workflow to track remediation
- Historical data across all your projects

> **Offline mode:** Add `--no-upload` to run locally without an account or sending report data to the PostgresAI API.

<details open>
<summary>See demo</summary>
<div align="center">
<img src="assets/demo-checkup.gif" alt="postgresai checkup demo" width="700">
</div>
</details>

## Express Checkup

Run specific checks or work offline:

```bash
# Run a specific check
npx postgresai checkup --check-id H002 postgresql://...

# Local JSON output only (no upload)
npx postgresai checkup --no-upload --check-id H002 postgresql://...
```

Checkup reports are JSON. Use `--json` to print them or `--output <dir>` to save
them; uploaded reports show up in the Console.

> **Tips:** `npx pgai checkup` also works. `bunx postgresai` if you prefer Bun.

## Full monitoring stack

For continuous monitoring with dashboards, install the full stack on a Linux machine with Docker:

```bash
# Quick demo with sample data
npx postgresai mon local-install --demo
# → Open http://localhost:3000

# Production setup (Linux + Docker required)
npx postgresai prepare-db postgresql://admin@host:5432/dbname  # Create monitoring role with minimal permissions
npx postgresai mon local-install --api-key=YOUR_TOKEN --db-url="postgresql://..."
```

Get your API key at [console.postgres.ai](https://console.postgres.ai) — or use the fully managed version there.

### Production-safe

All diagnostic queries are carefully designed to avoid the [observer effect](https://en.wikipedia.org/wiki/Observer_effect_(information_technology)) — they use timeouts, row limits, and non-blocking approaches. Battle-tested on production databases with dozens of TiB of data, hundreds of kTPS, and millions of DB objects.

Preview the setup SQL before running:

```bash
npx postgresai prepare-db --print-sql postgresql://...   # Review what will be created
```

The `prepare-db` step creates a read-only `postgres_ai_mon` user with minimal permissions, enables `pg_stat_statements`, and creates `postgres_ai` schema with a few helper views.

## What's inside

| Component | Purpose |
|-----------|---------|
| **Express checkup** | Instant health checks — no setup needed |
| **Grafana dashboards** | 13+ expert views: Node Overview, Query Analysis, Wait Events, Indexes, Tables, Replication, and more |
| **Scheduled reports** | 45+ metrics collected continuously via pgai-mon |
| **Metrics collection** | `pgai-mon`: one Rust binary that collects (pgwatch v3 metric semantics), stores history as Parquet and serves read-only SQL |
| **AI-ready output** | Structured JSON for Claude, GPT, and other agents |
| **Claude Code plugin** | Slash commands for health checks in your IDE |
| **MCP server** | Work with Issues from [console.postgres.ai](https://console.postgres.ai) in AI coding tools |

## For AI developers

`postgresai` is designed to feed AI agents with structured PostgreSQL diagnostics.

### Claude Code plugin

Install from the marketplace:

```bash
claude plugin marketplace add postgres-ai/postgresai
claude plugin install pgai@postgresai
```

Then ask Claude to check your database. The `postgresai` skill
([SKILL.md](.claude/skills/postgresai/SKILL.md)) deploys PostgresAI
monitoring and reports the problems with exact fixes. Other agents
install the skill with `npx skills add postgres-ai/postgresai --skill postgresai`.

Work with Issues from [console.postgres.ai](https://console.postgres.ai):

```
/pgai:issues
```

<details>
<summary>See demo</summary>
<div align="center">
<img src="assets/demo-claude-code.gif" alt="Claude Code plugin demo" width="700">
</div>
</details>

### MCP server (Claude Code, Cursor, Windsurf)

```bash
# Install MCP server for your AI coding tool
npx postgresai mcp install
```

<details>
<summary>See demo</summary>
<div align="center">
<img src="assets/demo-mcp.gif" alt="MCP install demo" width="700">
</div>
</details>

This enables AI agents to work with Issues and Action Items from [console.postgres.ai](https://console.postgres.ai) directly within your IDE.

### CLI + LLM

```bash
# Pipe checkup output to any LLM
npx postgresai checkup --json postgresql://... | llm -s "analyze this Postgres health report"

# Or use with Claude directly
npx postgresai checkup --json postgresql://... | claude -p "find issues and suggest fixes"
```

<details>
<summary>See demo</summary>
<div align="center">
<img src="assets/demo-llm.gif" alt="CLI + LLM demo" width="700">
</div>
</details>

### Sample AI workflow

1. AI agent detects slow query from application logs
2. Runs `postgresai checkup --check-id H002` → finds unused indexes
3. Gets structured JSON with analysis
4. Suggests `DROP INDEX` or creates PR

## Requirements

**For checkup (express mode):**
- Node.js 18+ (includes npm/npx)
- Network access to PostgreSQL 14-18

**For full monitoring stack:**
- Linux machine with Docker
- `pg_stat_statements` extension enabled

## Tips

- **Short alias:** `npx pgai checkup` works too
- **Can I use it without registration on console.postgres.ai?** Yes, both express checkup and full-fledged monitoring are open-source and can be used without any connection to console.postgres.ai. In instructions above, just skip `npx postgresai auth` and:
    - Express checkup: use `--no-upload`
    - Full monitoring: omit `--api-key`
- **Managed version:** Express checkup finds problems. The managed version at [console.postgres.ai](https://console.postgres.ai) also explains how to fix them and provides an Issues workflow to track fixes

## Links

| | |
|---|---|
| **Demo** | [demo.postgres.ai](https://demo.postgres.ai) (login: `demo` / `demo`) |
| **Docs** | [postgres.ai/docs](https://postgres.ai/docs) |
| **Issues** | [GitLab](https://gitlab.com/postgres-ai/postgresai/-/issues) |
| **Community** | [Postgres.FM](https://postgres.fm) · [Postgres.TV](https://postgres.tv) |

---

<div align="center">

**[PostgresAI](https://postgres.ai)** — Self-Driving Postgres

</div>

## 🎯 Use cases

**For developers:**
```bash
postgresai mon local-install --demo
```
Get a complete monitoring setup with demo data in under 2 minutes.

**For production:**
```bash
postgresai mon local-install --api-key=your_key
# Then add your databases
postgresai mon targets add "postgresql://user:pass@host:port/DB"
```

## 🔧 Management commands

```bash
# Instance management
postgresai mon targets add "postgresql://user:pass@host:port/DB"
postgresai mon targets list
postgresai mon targets test my-DB

# Service management
postgresai mon status
postgresai mon logs
postgresai mon restart

# Health check
postgresai mon health
```

## 🔄 Upgrading

> ⚠️ **Upgrading from the pgwatch + VictoriaMetrics stack starts metric history fresh.**
>
> `postgresai mon update` switches an install of the old stack to `pgai-mon`: it removes the old
> stack's containers and starts the new one. Nothing is migrated from the old metric store. The
> old stack's volumes are kept untouched; `mon update` prints their names and the
> `docker volume rm` command to remove them once you no longer need them.

To upgrade postgres_ai monitoring to a newer version:

### Step 1: Update the CLI

```bash
npm install -g postgresai@0.15.0
```

Or if you're using npx:
```bash
npx postgresai@0.15.0 --version  # verify the new version
```

### Step 2: Run `mon update`

Do not stop the stack first. `mon update` starts the new stack only when the old one was
running; after `mon stop` it leaves the box stopped and you start it with `postgresai mon start`.

```bash
postgresai mon update
```

What `mon update` does:

| Step | Detail |
|---|---|
| Migrates `.env` | Adds keys the new stack requires. Existing values, secrets and `instances.yml` are kept. |
| Refreshes `docker-compose.yml` | On installs that are not a git checkout. The old file is saved as `docker-compose.yml.bak-<oldtag>-<hash>`. |
| Sets `PGAI_TAG` | To this CLI's version, as `mon local-install` does, and prints `PGAI_TAG in .env: <old> -> <new>`. `--tag <tag>` runs another tag (the compose file still moves to the CLI's version). On a git checkout, or with `PGAI_SKIP_COMPOSE_REFRESH=1`, the compose file is not the CLI's, so the tag changes only with `--tag`. |
| Pulls images | For that `PGAI_TAG`. If the pull fails, the old `docker-compose.yml` and `PGAI_TAG` are put back and no container is re-created. |
| Applies them | A running stack: `docker compose up -d` re-creates the services that changed and those that depend on them, and starts any service of the stack that is stopped. Data volumes are kept. A stopped stack stays stopped: `postgresai mon start` starts it on the new images. Old stack to `pgai-mon`: see below. |

If the `docker-compose.yml` of the new version cannot be fetched, `mon update` keeps the compose
file and `PGAI_TAG`, pulls nothing and fails: new images with the old compose file would not work
together.

### Upgrading from the pgwatch + VictoriaMetrics stack

When the install still runs the old stack, `mon update` switches it to `pgai-mon`:

1. It replaces `docker-compose.yml` (backup: `docker-compose.yml.bak-<oldtag>-<hash>`) and pulls
   the images of this CLI's version.
2. It removes the old stack's containers and, if the old stack was running, starts the new one.
3. It prints the names of the old data volumes, the command that removes them, and the
   commands that bring the old stack back.

What to expect:

- **History starts empty.** Nothing is copied from VictoriaMetrics. Dashboards fill from the
  moment of the switch.
- **The old data stays on disk** in the old stack's Docker volumes until you remove it with the
  `docker volume rm ...` command that `mon update` printed.
- **Grafana listens on `127.0.0.1:3000` only** unless `GRAFANA_BIND_HOST` is set in `.env`. The
  old stack published it on every interface.
- If the image pull fails, the old stack keeps running, its `docker-compose.yml` and `PGAI_TAG`
  are put back, and you can run `mon update` again.

To go back to the old stack (`mon update` prints these commands with your paths filled in):

```bash
cd ~/.postgres_ai                     # your monitoring directory
postgresai mon stop                   # stop the new stack; volumes stay
docker volume rm <project>_postgres_ai_configs   # the old stack fills it again
cp docker-compose.yml.bak-<oldtag>-<hash> docker-compose.yml
# edit .env: set PGAI_TAG=<the old version>
docker compose up -d
```

The `postgres_ai_configs` volume holds Grafana's provisioning files. Without the `docker volume
rm`, the old Grafana finds the new stack's datasource beside its own and keeps restarting.
After a switch made by `mon local-install` (not `mon update`), restore `.env` from the
`.env.bak-<oldtag>` copy instead of editing `PGAI_TAG`: that command removed the old stack's
`VM_AUTH_*` keys from `.env`.

This works while the old volumes exist. Data that `pgai-mon` collected in the meantime is
not carried back; it stays in the `pgai_mon_data` volume.

> **Note:** The `.env` file contains configuration for the monitoring stack, including `PGAI_TAG` (the Docker image version tag), `REPLICATOR_PASSWORD` (generated password for the demo standby replication user), `PGAI_MON_TOKEN` (the bearer token `pgai-mon` requires; Grafana, the reporter and instance-jobs send it), and optionally `PGAI_MON_STORE` (where metric history goes, e.g. `s3://<bucket>/<prefix>`; unset means the local `pgai_mon_data` volume), `GF_SECURITY_ADMIN_PASSWORD` (Grafana admin password) and `PGAI_REGISTRY` (custom Docker registry). `postgresai mon local-install` preserves existing `REPLICATOR_PASSWORD` and `PGAI_MON_TOKEN` values or generates new ones when they are missing.

> **In-place upgrade note:** Newer stack versions can require both additional `.env` keys and a matching `docker-compose.yml`. `postgresai mon local-install -y`, `postgresai mon update`, and `postgresai mon update-config` all perform a purely-additive `.env` migration on every run (existing values preserved verbatim; newly-required keys such as `PGAI_MON_TOKEN` appended with safe random defaults) **and** refresh `docker-compose.yml` to the new stack version on non-git installs (npx / `npm install -g`), backing up the old compose as `docker-compose.yml.bak-<oldtag>-<hash>`. The old stack's `VM_AUTH_*` and `VM_*_AUTH_KEY` keys are no longer read: `mon local-install` drops them when it rewrites `.env`, and `mon update` leaves them in place. If you run `docker compose` directly and maintain `.env` yourself, run `postgresai mon update-config` once to have the CLI add `PGAI_MON_TOKEN` (and refresh the compose) for you.

**Alternative: Manual upgrade**

If you prefer more control:

```bash
# Update the PGAI_TAG in .env to match your target stack version
# Edit .env and set PGAI_TAG=0.15.0

# Migrate .env to add any newly-required keys (e.g. PGAI_MON_TOKEN)
postgresai mon update-config

# Pull new images
docker compose pull

# Start services
postgresai mon start
```

### Verify the upgrade

After upgrading, verify services are running correctly:

```bash
postgresai mon status
postgresai mon health
```

Check Grafana dashboards at http://localhost:3000 to confirm metrics are being collected.

## Storage, retention and security

| Topic | Default | How to change |
|---|---|---|
| Where metrics are stored | The `pgai_mon_data` Docker volume on the host. | `PGAI_MON_STORE=s3://bucket/prefix` in `.env` stores the Parquet files in S3. The volume then holds only the write-ahead log and a cache. Credentials come from the AWS environment or the instance role. |
| Retention of metrics | 14 days | `PGAI_MON_RETENTION_DAYS` in `.env`. `0` keeps everything. |
| Retention of query texts | 30 days | `PGAI_MON_TEXTS_RETENTION_DAYS` in `.env`. `0` keeps everything. |
| Ports | `pgai-mon` on `127.0.0.1:8765`, Grafana on `127.0.0.1:3000`. Neither is reachable from other hosts. | `GRAFANA_PORT` and `PGAI_MON_PORT` change the host ports. `GRAFANA_BIND_HOST=0.0.0.0:` publishes Grafana on every interface. Put TLS in front of it first. |
| Container names | Set by compose from the project: `<project>-<service>-1` (for example `monitoring-pgai-mon-1`). | `COMPOSE_PROJECT_NAME` in `.env`. Unset: the name of the install directory. Two stacks on one host need two project names and two Grafana ports. |
| `PGAI_MON_TOKEN` | Generated by `mon local-install`. Required. | Full access to `pgai-mon`: queries and push ingest. Whoever holds it can read every metric and query text. |
| `PGAI_MON_READ_TOKEN` | Empty: readers use `PGAI_MON_TOKEN`. | A second token that can query but not write. Grafana, the reporter and instance-jobs use it when it is set. |
| `GF_SECURITY_ADMIN_PASSWORD` | Generated by `mon local-install`. Required: there is no default password. | Set it in `.env` when you run `docker compose` yourself. |

Files older than the retention are deleted from the store, in the S3 bucket too. With an S3
store, a lost VM loses at most the last 5 minutes of data (the upload interval) plus one upload.
[`mon/README.md`](./mon/README.md) has the full configuration reference and
[`mon/ARCHITECTURE.md`](./mon/ARCHITECTURE.md) explains how the store works.

## ☸️ Kubernetes deployment

There is no Helm chart for the current (`pgai-mon`) stack yet. The Helm chart of the previous
pgwatch + VictoriaMetrics stack is kept in [`old/postgres_ai_helm`](./old/postgres_ai_helm) for
reference only: it is unmaintained and its images are no longer built. To run the current stack on
Kubernetes you have to write your own manifests for the `pgai-mon` image.

## 📋 Checkup reports

postgres_ai monitoring generates automated health check reports based on [postgres-checkup](https://gitlab.com/postgres-ai/postgres-checkup). Each report has a unique check ID and title:

### A. General / Infrastructural
| Check ID | Title |
|----------|-------|
| A001 | System information |
| A002 | Version information |
| A003 | Postgres settings |
| A004 | Cluster information |
| A005 | Extensions |
| A006 | Postgres setting deviations |
| A007 | Altered settings |
| A008 | Disk usage and file system type |

A005 reads every database that accepts connections, each over its own connection with
the checkup's credentials (at most 100 databases, within 60 seconds): installed
extensions with versions, index counts per access method and the extension of their
operator classes, row security policies, extended statistics and deferrable unique or
primary-key constraints. From the connected
database it adds the logical slots with their plugin, the non-superuser roles with
CREATE on each database, and the roles that can use replication. MD5 password hashes
(pg_authid) and pg_hba.conf rules need a superuser: for any other role they are
recorded as unknown, with the reason, never as none. A database that is not read is
listed with the reason.

A monitoring box writes A005 too: its reporter reads each monitored node once a day with
the connection pgai-mon collects with (the node's `conn_str`), each query in its own read-only
transaction. A node it cannot reach is listed with the reason and no database.

### D. Monitoring / Troubleshooting
| Check ID | Title |
|----------|-------|
| D004 | pg_stat_statements and pg_stat_kcache settings |

### F. Autovacuum, Bloat
| Check ID | Title |
|----------|-------|
| F001 | Autovacuum: current settings |
| F004 | Autovacuum: heap bloat (estimated) |
| F005 | Autovacuum: index bloat (estimated) |
| F009 | Xmin horizon and blockers |

F009 includes a 500-character query preview only when the checkup role can read all
statistics. Disable preview capture with
`PGOPTIONS='-c postgresai.capture_query_preview=off' postgresai checkup ...`.

### G. Performance / Connections / Memory-related settings
| Check ID | Title |
|----------|-------|
| G001 | Memory-related settings |

### H. Index analysis
| Check ID | Title |
|----------|-------|
| H001 | Invalid indexes |
| H002 | Unused indexes |
| H004 | Redundant indexes |

### K. SQL query analysis
| Check ID | Title |
|----------|-------|
| K001 | Globally aggregated query metrics |
| K003 | Top queries by total time (total_exec_time + total_plan_time) |
| K004 | Top queries by temp bytes written |
| K005 | Top queries by WAL generation |
| K006 | Top queries by shared blocks read |
| K007 | Top queries by shared blocks hit |

### M. SQL query analysis (top queries)
| Check ID | Title |
|----------|-------|
| M001 | Top queries by mean execution time |
| M002 | Top queries by rows (I/O intensity) |
| M003 | Top queries by I/O time |

### N. Wait events analysis
| Check ID | Title |
|----------|-------|
| N001 | Wait events grouped by type and query |

## 🌐 Access points

After running local-install:

- **🚀 MAIN: Grafana Dashboard**: http://localhost:3000, or the `GRAFANA_PORT` set in `.env` (login: `monitor`; password is shown at the end of local-install)

Technical access (for advanced users):
- **pgai-mon SQL API**: http://127.0.0.1:8765, or `PGAI_MON_PORT` (bearer token `PGAI_MON_TOKEN` from `.env`); `postgresai mon sql --local "<query>"` uses it
- **Demo DB** (`--demo`): no published port; `docker exec -it <project>-target-db-1 psql -U postgres target_database` (`local-install` prints the exact command)

## 📖 Help

```bash
postgresai --help
postgresai mon --help
```

## 🔑 PostgresAI access token
Get your access token at [PostgresAI](https://postgres.ai) for automated report uploads and advanced analysis.

## 🛣️ Roadmap

- Host stats for on-premise and managed Postgres setups
- `pg_wait_sampling` and `pg_stat_kcache` extension support
- Additional expert dashboards: autovacuum, checkpointer, lock analysis
- Query plan analysis and automated recommendations
- Enhanced AI integration capabilities

## 🧪 Testing

Python-based report generation lives under `reporter/` and now ships with a pytest suite.

### Installation

Install dev dependencies (includes `pytest`, `pytest-postgresql`, `psycopg`, etc.):
```bash
python3 -m pip install -r reporter/requirements-dev.txt
```

### Running tests

#### Unit tests only (fast, no external services required)

Run only unit tests (external services are mocked):
```bash
pytest tests/reporter
```

This automatically skips integration tests. Or run specific test files:
```bash
pytest tests/reporter/test_generators_unit.py -v
pytest tests/reporter/test_formatters.py -v
```

#### All tests: unit + integration (requires PostgreSQL)

Run the complete test suite (both unit and integration tests):
```bash
pytest tests/reporter --run-integration
```

Integration tests create a temporary PostgreSQL instance automatically and require PostgreSQL binaries (`initdb`, `postgres`) on your PATH. No manual database setup or environment variables are required - the tests create and destroy their own temporary PostgreSQL instances.

**Summary:**
- `pytest tests/reporter` → **Unit tests only** (integration tests skipped)
- `pytest tests/reporter --run-integration` → **Both unit and integration tests**

### Test coverage

Generate coverage report:
```bash
pytest tests/reporter -m unit --cov=reporter --cov-report=html
```

View the coverage report by opening `htmlcov/index.html` in your browser.

## 🤝 Contributing

We welcome contributions from Postgres experts! Please check our [GitLab repository](https://gitlab.com/postgres-ai/postgresai) for:
- Code standards and review process
- Dashboard design principles
- Testing requirements for monitoring components

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🏢 About PostgresAI

postgres_ai monitoring is developed by [PostgresAI](https://postgres.ai), bringing years of Postgres expertise into automated monitoring and analysis tools. We provide enterprise consulting and advanced Postgres solutions for fast-growing companies.

## 📞 Support & community

- 💬 [Get support](https://postgres.ai/contact)
- 📺 [Postgres.TV (YouTube)](https://postgres.tv)
- 🎙️ [Postgres FM Podcast](https://postgres.fm)
- 🐛 [Report issues](https://gitlab.com/postgres-ai/postgresai/-/issues)
- 📧 [Enterprise support](https://postgres.ai/consulting)

## Visualization standard

Wait-event charts in monitoring and supporting materials follow the
[PostgresAI color scheme](docs/COLOR_SCHEME.md).
