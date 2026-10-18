workspace "PostgresAI" "AI-native PostgreSQL observability: monitoring, health checks, and root cause analysis." {

    !identifiers hierarchical

    model {
        # ---------------------------------------------------------------
        # People / external actors
        # ---------------------------------------------------------------
        dba         = person "Engineer / DBA"  "Runs health checks, reviews dashboards, and resolves issues."
        aiAgent     = person "AI Coding Agent" "Claude Code, Cursor, Windsurf, etc. Consumes structured reports and the MCP server." "AI"

        # ---------------------------------------------------------------
        # External software systems
        # ---------------------------------------------------------------
        targetPg    = softwareSystem "Target PostgreSQL"      "The PostgreSQL database(s) being observed (self-hosted, RDS/Aurora, CloudSQL, Supabase, etc.)." "External, Database"
        console     = softwareSystem "console.postgres.ai"    "PostgresAI managed cloud: web UI, Issues API, report & file storage, auth (OAuth2/PKCE)." "External"
        supabaseApi = softwareSystem "Supabase Management API" "Executes SQL against Supabase-hosted databases."                                        "External"
        awsApm      = softwareSystem "AWS / Amazon Managed Prometheus" "RDS, Aurora and Amazon Managed Prometheus metrics sources."                     "External"
        llm         = softwareSystem "LLM Provider"           "Anthropic Claude / OpenAI GPT used to analyze JSON reports."                             "External, AI"

        # ---------------------------------------------------------------
        # The PostgresAI system and its containers
        # ---------------------------------------------------------------
        pgai = softwareSystem "PostgresAI" "AI-native PostgreSQL observability platform." {

            cli = container "CLI" "Express checkup, stack install, target & DB prep, auth. Binaries: postgresai / pgai." "Node.js / TypeScript" "EntryPoint" {
                cmd          = component "Command Dispatch" "Parses commands and flags, orchestrates workflows." "postgres-ai.ts"
                checkup      = component "Checkup Engine"   "Runs 45+ health checks and builds structured JSON reports." "checkup.ts"
                checkupApi   = component "Checkup API Client" "Uploads/fetches reports to/from console.postgres.ai." "checkup-api.ts"
                init         = component "DB Init / prepare-db" "Creates monitoring role, schema and permissions on targets." "init.ts"
                instances    = component "Targets / Instances" "Manages monitored targets (instances.yml)." "instances.ts"
                issues       = component "Issues Client" "CRUD against the console Issues API." "issues.ts"
                auth         = component "Auth (OAuth2/PKCE)" "Browser-based login and API-key handling." "auth-server.ts, pkce.ts"
                storage      = component "Storage Client" "Uploads/downloads files to PostgresAI storage." "storage.ts"
                supabase     = component "Supabase Client" "Runs queries via the Supabase Management API." "supabase.ts"
                config       = component "Local Config" "Reads/writes ~/.postgres-ai settings." "config.ts, util.ts"
            }

            mcp = container "MCP Server" "Model Context Protocol server exposing Issues tools to AI coding agents." "Node.js / TypeScript" "EntryPoint" {
                mcpTools = component "MCP Tools" "list_issues, view_issue, create_issue, update_issue, ..." "mcp-server.ts"
            }

            pgwatchPg   = container "pgwatch (Postgres sink)"   "Collects pg_stat_statements, wait events, table/index stats; writes to the Postgres sink." "Go (patched v3.7.0)"
            pgwatchProm = container "pgwatch (Prometheus sink)" "Same collector, writes metrics to VictoriaMetrics." "Go (patched v3.7.0)"

            reporter = container "Reporter" "Generates 45+ structured health-check reports (A/D/F/H/I/K/M/N series) from metrics & history." "Python" {
                reportGen = component "Report Generators" "Per-check report logic." "postgres_reports.py"
                schemas   = component "Report Schemas" "29 JSON Schemas validating report output." "schemas/*.schema.json"
            }

            flask = container "Monitoring Backend" "API for pg_stat_statements query history and PromQL proxying." "Python / Flask + Gunicorn" {
                api      = component "API Endpoints" "Query history, query-id mapping, PromQL proxy." "app.py"
                promql   = component "PromQL Utils" "Escaping/formatting PromQL." "promql_utils.py"
            }

            grafana    = container "Grafana" "13+ dashboards (Four Golden Signals, wait events, indexes, replication, ...)." "Grafana 12.x" "WebUI"
            vmetrics   = container "VictoriaMetrics" "Prometheus-compatible time-series store for metrics." "VictoriaMetrics" "Database"
            sinkPg     = container "Postgres Sink" "Stores historical metrics and check data." "PostgreSQL 17" "Database"

            telemetry  = container "Telemetry Agent" "Hourly system metrics (OOM, free RAM/disk) posted to the platform." "Node.js / TypeScript"
            indexPilot = container "pg_index_pilot" "Autonomous index lifecycle: bloat estimation, recommendations, reindexing." "SQL / PL-pgSQL + Bash"
        }

        # ---------------------------------------------------------------
        # Relationships — Context level
        # ---------------------------------------------------------------
        dba     -> pgai "Runs checkups, views dashboards, manages issues"
        aiAgent -> pgai "Reads reports, calls MCP tools"
        pgai    -> targetPg    "Connects to and queries"
        pgai    -> console     "Uploads reports, syncs issues, authenticates" "HTTPS"
        pgai    -> supabaseApi "Executes SQL"  "HTTPS"
        pgai    -> awsApm      "Reads metrics" "HTTPS"
        aiAgent -> llm         "Analyzes JSON reports"

        # ---------------------------------------------------------------
        # Relationships — Container level
        # ---------------------------------------------------------------
        dba -> pgai.cli     "Runs commands"
        dba -> pgai.grafana "Views dashboards" "HTTPS"
        aiAgent -> pgai.mcp "Calls tools" "MCP/stdio"

        pgai.cli -> targetPg     "Runs health-check & prepare-db SQL" "SQL/TCP"
        pgai.cli -> console      "Uploads reports, syncs issues, auth" "HTTPS"
        pgai.cli -> supabaseApi  "Runs queries" "HTTPS"
        pgai.cli -> pgai.reporter "Triggers report generation"

        pgai.mcp -> console "Reads/writes issues" "HTTPS"

        pgai.pgwatchPg   -> targetPg   "Scrapes metrics" "SQL/TCP"
        pgai.pgwatchProm -> targetPg   "Scrapes metrics" "SQL/TCP"
        pgai.pgwatchPg   -> pgai.sinkPg   "Writes metrics" "SQL/TCP"
        pgai.pgwatchProm -> pgai.vmetrics "Writes metrics" "HTTP"

        pgai.reporter -> pgai.vmetrics "Reads metrics" "PromQL/HTTP"
        pgai.reporter -> pgai.sinkPg   "Reads history" "SQL/TCP"
        pgai.reporter -> awsApm        "Reads metrics" "HTTPS"
        pgai.reporter -> console       "Uploads reports" "HTTPS"

        pgai.flask    -> pgai.vmetrics "Proxies PromQL" "HTTP"
        pgai.flask    -> pgai.sinkPg   "Reads query history" "SQL/TCP"
        pgai.grafana  -> pgai.vmetrics "Queries metrics" "PromQL/HTTP"
        pgai.grafana  -> pgai.sinkPg   "Queries history" "SQL/TCP"
        pgai.grafana  -> pgai.flask    "Calls backend API" "HTTP"

        pgai.telemetry  -> console   "Posts system telemetry" "HTTPS"
        pgai.indexPilot -> targetPg  "Manages indexes" "SQL/TCP"

        # ---------------------------------------------------------------
        # Relationships — Component level (CLI)
        # ---------------------------------------------------------------
        pgai.cli.cmd -> pgai.cli.checkup   "Invokes"
        pgai.cli.cmd -> pgai.cli.init      "Invokes"
        pgai.cli.cmd -> pgai.cli.instances "Invokes"
        pgai.cli.cmd -> pgai.cli.issues    "Invokes"
        pgai.cli.cmd -> pgai.cli.auth      "Invokes"
        pgai.cli.checkup    -> targetPg          "Runs SQL" "SQL/TCP"
        pgai.cli.checkup    -> pgai.cli.checkupApi "Uploads report"
        pgai.cli.checkupApi -> console           "HTTPS"
        pgai.cli.init       -> targetPg          "Creates role/schema" "SQL/TCP"
        pgai.cli.issues     -> console           "HTTPS"
        pgai.cli.auth       -> console           "OAuth2/PKCE" "HTTPS"
        pgai.cli.storage    -> console           "Uploads files" "HTTPS"
        pgai.cli.supabase   -> supabaseApi       "HTTPS"
        pgai.cli.checkup    -> pgai.cli.config   "Reads settings"
    }

    views {
        systemContext pgai "SystemContext" {
            include *
            autolayout lr
            description "C4 Level 1 — PostgresAI in context."
        }

        container pgai "Containers" {
            include *
            autolayout lr
            description "C4 Level 2 — containers inside PostgresAI."
        }

        component pgai.cli "CliComponents" {
            include *
            autolayout lr
            description "C4 Level 3 — components inside the CLI."
        }

        component pgai.reporter "ReporterComponents" {
            include *
            autolayout lr
            description "C4 Level 3 — components inside the Reporter."
        }

        styles {
            element "Person"   { shape person background #08427b color #ffffff }
            element "AI"       { background #6b3fa0 color #ffffff }
            element "Software System" { background #1168bd color #ffffff }
            element "External" { background #999999 color #ffffff }
            element "Container" { background #438dd5 color #ffffff }
            element "Component" { background #85bbf0 color #000000 }
            element "Database" { shape cylinder }
            element "WebUI"    { shape webBrowser }
            element "EntryPoint" { shape roundedBox }
        }

        theme default
    }
}
