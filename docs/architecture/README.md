# PostgresAI — C4 Architecture Model

This directory holds the [C4 model](https://c4model.com/) for PostgresAI, an
AI-native PostgreSQL observability platform (monitoring, health checks, and root
cause analysis).

The model is maintained in two complementary forms:

| File | Purpose |
|------|---------|
| [`workspace.dsl`](./workspace.dsl) | **Source of truth.** [Structurizr DSL](https://docs.structurizr.com/dsl) describing people, systems, containers, components and their relationships. Renders all C4 levels and stays in sync as a single definition. |
| `README.md` (this file) | GitHub-renderable [Mermaid](https://mermaid.js.org/syntax/c4.html) diagrams for quick reading without tooling. |

The C4 model describes architecture at four levels of zoom: **System Context**
(L1) → **Containers** (L2) → **Components** (L3) → Code (L4, left to the source).

## How to view the Structurizr model

```bash
# Render/edit interactively with Structurizr Lite
docker run -it --rm -p 8080:8080 \
  -v "$(pwd)/docs/architecture:/usr/local/structurizr" \
  structurizr/lite
# then open http://localhost:8080
```

---

## Level 1 — System Context

How PostgresAI fits among its users and the external systems it talks to.

```mermaid
C4Context
    title System Context — PostgresAI

    Person(dba, "Engineer / DBA", "Runs health checks, reviews dashboards, resolves issues")
    Person(ai, "AI Coding Agent", "Claude Code, Cursor, Windsurf — consumes reports & MCP tools")

    System(pgai, "PostgresAI", "AI-native PostgreSQL observability platform")

    System_Ext(target, "Target PostgreSQL", "Databases being observed (self-hosted, RDS/Aurora, CloudSQL, Supabase)")
    System_Ext(console, "console.postgres.ai", "Managed cloud: UI, Issues API, report/file storage, auth")
    System_Ext(supabase, "Supabase Management API", "Executes SQL on Supabase databases")
    System_Ext(aws, "AWS / Amazon Managed Prometheus", "RDS/Aurora + managed metrics")
    System_Ext(llm, "LLM Provider", "Anthropic Claude / OpenAI GPT")

    Rel(dba, pgai, "Runs checkups, views dashboards, manages issues")
    Rel(ai, pgai, "Reads reports, calls MCP tools")
    Rel(pgai, target, "Connects to & queries", "SQL")
    Rel(pgai, console, "Uploads reports, syncs issues, authenticates", "HTTPS")
    Rel(pgai, supabase, "Executes SQL", "HTTPS")
    Rel(pgai, aws, "Reads metrics", "HTTPS")
    Rel(ai, llm, "Analyzes JSON reports")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="2")
```

---

## Level 2 — Containers

The separately deployable/runnable units inside PostgresAI. The CLI delivers the
zero-setup express checkup; the rest form the optional full monitoring stack
(`docker-compose.yml` / Helm chart).

```mermaid
C4Container
    title Containers — PostgresAI

    Person(dba, "Engineer / DBA")
    Person(ai, "AI Coding Agent")

    System_Ext(target, "Target PostgreSQL")
    System_Ext(console, "console.postgres.ai")
    System_Ext(supabase, "Supabase Management API")
    System_Ext(aws, "Amazon Managed Prometheus")

    System_Boundary(pgai, "PostgresAI") {
        Container(cli, "CLI", "Node.js / TypeScript", "Express checkup, stack install, prepare-db, auth (postgresai / pgai)")
        Container(mcp, "MCP Server", "Node.js / TypeScript", "Exposes Issues tools to AI agents")
        Container(reporter, "Reporter", "Python", "Generates 45+ structured health-check reports")
        Container(flask, "Monitoring Backend", "Python / Flask", "Query history + PromQL proxy API")
        Container(pgwPg, "pgwatch → Postgres", "Go", "Collects metrics into the Postgres sink")
        Container(pgwProm, "pgwatch → Prometheus", "Go", "Collects metrics into VictoriaMetrics")
        ContainerDb(vm, "VictoriaMetrics", "TSDB", "Prometheus-compatible metrics store")
        ContainerDb(sink, "Postgres Sink", "PostgreSQL 17", "Historical metrics & check data")
        Container(grafana, "Grafana", "Grafana 12", "13+ observability dashboards")
        Container(telemetry, "Telemetry Agent", "Node.js", "Hourly system telemetry")
        Container(pilot, "pg_index_pilot", "SQL / PL-pgSQL", "Autonomous index lifecycle")
    }

    Rel(dba, cli, "Runs commands")
    Rel(dba, grafana, "Views dashboards", "HTTPS")
    Rel(ai, mcp, "Calls tools", "MCP/stdio")

    Rel(cli, target, "Health-check & prepare-db SQL", "SQL")
    Rel(cli, console, "Uploads reports, syncs issues, auth", "HTTPS")
    Rel(cli, supabase, "Runs queries", "HTTPS")
    Rel(cli, reporter, "Triggers report generation")
    Rel(mcp, console, "Reads/writes issues", "HTTPS")

    Rel(pgwPg, target, "Scrapes", "SQL")
    Rel(pgwProm, target, "Scrapes", "SQL")
    Rel(pgwPg, sink, "Writes metrics", "SQL")
    Rel(pgwProm, vm, "Writes metrics", "HTTP")

    Rel(reporter, vm, "Reads metrics", "PromQL")
    Rel(reporter, sink, "Reads history", "SQL")
    Rel(reporter, aws, "Reads metrics", "HTTPS")
    Rel(reporter, console, "Uploads reports", "HTTPS")

    Rel(flask, vm, "Proxies PromQL", "HTTP")
    Rel(flask, sink, "Reads query history", "SQL")
    Rel(grafana, vm, "Queries metrics", "PromQL")
    Rel(grafana, sink, "Queries history", "SQL")
    Rel(grafana, flask, "Calls backend API", "HTTP")

    Rel(telemetry, console, "Posts telemetry", "HTTPS")
    Rel(pilot, target, "Manages indexes", "SQL")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

## Level 3 — Components: CLI

The CLI is the primary entry point and the most component-rich container
(`cli/lib/*.ts`).

```mermaid
C4Component
    title Components — CLI

    Person(dba, "Engineer / DBA")
    System_Ext(target, "Target PostgreSQL")
    System_Ext(console, "console.postgres.ai")
    System_Ext(supabase, "Supabase Management API")

    Container_Boundary(cli, "CLI") {
        Component(cmd, "Command Dispatch", "postgres-ai.ts", "Parses commands/flags, orchestrates workflows")
        Component(checkup, "Checkup Engine", "checkup.ts", "Runs 45+ health checks, builds JSON reports")
        Component(checkupApi, "Checkup API Client", "checkup-api.ts", "Uploads/fetches reports")
        Component(init, "DB Init / prepare-db", "init.ts", "Creates monitoring role, schema, permissions")
        Component(instances, "Targets / Instances", "instances.ts", "Manages monitored targets")
        Component(issues, "Issues Client", "issues.ts", "Console Issues API CRUD")
        Component(auth, "Auth (OAuth2/PKCE)", "auth-server.ts, pkce.ts", "Login & API-key handling")
        Component(storage, "Storage Client", "storage.ts", "File upload/download")
        Component(sb, "Supabase Client", "supabase.ts", "Runs SQL via Supabase API")
        Component(cfg, "Local Config", "config.ts, util.ts", "~/.postgres-ai settings")
    }

    Rel(dba, cmd, "Runs commands")
    Rel(cmd, checkup, "Invokes")
    Rel(cmd, init, "Invokes")
    Rel(cmd, instances, "Invokes")
    Rel(cmd, issues, "Invokes")
    Rel(cmd, auth, "Invokes")

    Rel(checkup, target, "Runs SQL", "SQL")
    Rel(checkup, checkupApi, "Uploads report")
    Rel(checkup, cfg, "Reads settings")
    Rel(checkupApi, console, "HTTPS")
    Rel(init, target, "Creates role/schema", "SQL")
    Rel(issues, console, "HTTPS")
    Rel(auth, console, "OAuth2/PKCE", "HTTPS")
    Rel(storage, console, "Uploads files", "HTTPS")
    Rel(sb, supabase, "HTTPS")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

## Level 3 — Components: Reporter

```mermaid
C4Component
    title Components — Reporter

    System_Ext(aws, "Amazon Managed Prometheus")
    System_Ext(console, "console.postgres.ai")
    ContainerDb(vm, "VictoriaMetrics")
    ContainerDb(sink, "Postgres Sink")

    Container_Boundary(reporter, "Reporter") {
        Component(gen, "Report Generators", "postgres_reports.py", "Per-check report logic (A/D/F/H/I/K/M/N series)")
        Component(schemas, "Report Schemas", "schemas/*.schema.json", "29 JSON Schemas validating output")
    }

    Rel(gen, vm, "Reads metrics", "PromQL")
    Rel(gen, sink, "Reads history", "SQL")
    Rel(gen, aws, "Reads metrics", "HTTPS")
    Rel(gen, schemas, "Validates against")
    Rel(gen, console, "Uploads reports", "HTTPS")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

---

## Maintaining this model

- Treat `workspace.dsl` as the source of truth; update it when containers,
  components, or integrations change.
- Keep the Mermaid diagrams above in sync for at-a-glance reading on GitHub.
- Useful references: [c4model.com](https://c4model.com/),
  [Structurizr DSL docs](https://docs.structurizr.com/dsl),
  [Mermaid C4 syntax](https://mermaid.js.org/syntax/c4.html).
