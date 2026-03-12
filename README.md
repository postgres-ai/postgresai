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
[Get Started](#get-started)

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

## Get Started

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

> **Offline mode:** Add `--no-upload` to run locally without an account.

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
| **Scheduled reports** | 45+ metrics collected continuously via pgwatch |
| **Metrics collection** | pgwatch v3 + Victoria Metrics |
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

### MCP Server (Claude Code, Cursor, Windsurf)

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

To upgrade postgres_ai monitoring to a newer version:

### Step 1: Update the CLI

```bash
npm install -g postgresai@latest
```

Or if you're using npx:
```bash
npx postgresai@latest --version  # verify the new version
```

### Step 2: Stop running services

```bash
postgresai mon stop
```

### Step 3: Pull new Docker images and restart

The simplest approach is to re-run local-install, which updates the image tag and pulls new images:

```bash
postgresai mon local-install -y
```

This will:
- Update the `PGAI_TAG` in `.env` (located in your monitoring directory, typically `~/.postgres_ai/` or your current working directory) to match the new CLI version
- Pull the latest Docker images
- Start the services with the new images

> **Note:** The `.env` file contains configuration for the monitoring stack, including `PGAI_TAG` (the Docker image version tag) and optionally `GF_SECURITY_ADMIN_PASSWORD` (Grafana admin password) and `PGAI_REGISTRY` (custom Docker registry).

**Alternative: Manual upgrade**

If you prefer more control:

```bash
# Update the PGAI_TAG in .env to match your CLI version
postgresai --version  # check your CLI version
# Edit .env and set PGAI_TAG to the version number

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

- **🚀 MAIN: Grafana Dashboard**: http://localhost:3000 (login: `monitoring`; password is shown at the end of local-install)

Technical URLs (for advanced users):
- **Demo DB**: postgresql://postgres:postgres@localhost:55432/target_database
- **Monitoring**: http://localhost:58080 (PGWatch)
- **Metrics**: http://localhost:59090 (Victoria Metrics)

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

### Running Tests

#### Unit Tests Only (Fast, No External Services Required)

Run only unit tests with mocked Prometheus interactions:
```bash
pytest tests/reporter
```

This automatically skips integration tests. Or run specific test files:
```bash
pytest tests/reporter/test_generators_unit.py -v
pytest tests/reporter/test_formatters.py -v
```

#### All Tests: Unit + Integration (Requires PostgreSQL)

Run the complete test suite (both unit and integration tests):
```bash
pytest tests/reporter --run-integration
```

Integration tests create a temporary PostgreSQL instance automatically and require PostgreSQL binaries (`initdb`, `postgres`) on your PATH. No manual database setup or environment variables are required - the tests create and destroy their own temporary PostgreSQL instances.

**Summary:**
- `pytest tests/reporter` → **Unit tests only** (integration tests skipped)
- `pytest tests/reporter --run-integration` → **Both unit and integration tests**

### Test Coverage

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

</div>
