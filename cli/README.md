# PostgresAI CLI

Command-line interface for PostgresAI monitoring and database management.

## Installation

### From npm

```bash
npm install -g postgresai
```

For reproducible installs, pin the 0.15 release explicitly:
```bash
npm install -g postgresai@0.15.0
```

Note: in this repository, `cli/package.json` uses a placeholder version (`0.0.0-dev.0`). The real published version is set by the git tag in CI when publishing to npm.

### From Homebrew (macOS)

```bash
# Add the PostgresAI tap
brew tap postgres-ai/tap https://gitlab.com/postgres-ai/homebrew-tap.git

# Install postgresai
brew install postgresai
```

## Usage

The `postgresai` package provides two command aliases:
```bash
postgresai --help   # Canonical, discoverable
pgai --help         # Short and convenient
```

You can also run it without installing via `npx`:

```bash
npx postgresai --help
```

### Optional shorthand: `pgai`

If you want `npx pgai ...` as a shorthand for `npx postgresai ...`, install the separate `pgai` wrapper package:

```bash
npx pgai --help
```

## prepare-db (create monitoring user in Postgres)

This command creates (or updates) the `postgres_ai_mon` user, creates the required view(s), and grants the permissions described in the root `README.md` (it is idempotent). Where supported, it also enables observability extensions described there.

Run without installing (positional connection string):

```bash
npx postgresai prepare-db postgresql://admin@host:5432/dbname
```

It also accepts libpq "conninfo" syntax:

```bash
npx postgresai prepare-db "dbname=dbname host=host user=admin"
```

And psql-like options:

```bash
npx postgresai prepare-db -h host -p 5432 -U admin -d dbname
```

Password input options (in priority order):
- `--password <password>`
- `PGAI_MON_PASSWORD` environment variable
- if not provided: a strong password is generated automatically

By default, the generated password is printed **only in interactive (TTY) mode**. In non-interactive mode, you must either provide the password explicitly, or opt-in to printing it:
- `--print-password` (dangerous in CI logs)

Optional permissions (RDS/self-managed extras from the root `README.md`) are enabled by default. To skip them:

```bash
npx postgresai prepare-db postgresql://admin@host:5432/dbname --skip-optional-permissions
```

### Print SQL / dry run

To see what SQL would be executed (passwords redacted by default):

```bash
npx postgresai prepare-db postgresql://admin@host:5432/dbname --print-sql
```

### Supabase mode

For Supabase projects, you can use the Management API instead of direct PostgreSQL connections. This is useful when direct database access is restricted.

```bash
# Using environment variables
export SUPABASE_ACCESS_TOKEN='your_management_api_token'
export SUPABASE_PROJECT_REF='your_project_ref'
npx postgresai prepare-db --supabase

# Using command-line options
npx postgresai prepare-db --supabase \
  --supabase-access-token 'your_token' \
  --supabase-project-ref 'your_project_ref'

# Auto-detect project ref from a Supabase database URL
npx postgresai prepare-db postgresql://postgres:password@db.abc123.supabase.co:5432/postgres \
  --supabase --supabase-access-token 'your_token'
```

The Supabase access token can be created at https://supabase.com/dashboard/account/tokens.

Options:
- `--supabase` - Enable Supabase Management API mode
- `--supabase-access-token <token>` - Supabase Management API access token (or use `SUPABASE_ACCESS_TOKEN` env var)
- `--supabase-project-ref <ref>` - Supabase project reference (or use `SUPABASE_PROJECT_REF` env var)

Notes:
- The project reference can be auto-detected from Supabase database URLs
- All standard options work with Supabase mode (`--verify`, `--print-sql`, `--skip-optional-permissions`, etc.)
- When using `--verify`, the tool checks if all required setup is in place

### Verify and password reset

Verify that everything is configured as expected (no changes):

```bash
npx postgresai prepare-db postgresql://admin@host:5432/dbname --verify
```

Reset monitoring user password only (no other changes):

```bash
npx postgresai prepare-db postgresql://admin@host:5432/dbname --reset-password --password 'new_password'
```

## Quick start

### Authentication

Authenticate via browser to obtain API key:
```bash
postgresai auth
```

This will:
- Open your browser for authentication
- Prompt you to select an organization — or **All my organizations**, which mints a *global token*
- Automatically save your API key to `~/.config/postgresai/config.json`

#### Token kinds

|  | Per-organization token | Global token (`pai_global_…`) |
|---|---|---|
| Reaches | one organization | every organization you belong to |
| Org selection | implied by the token | **named on every org-specific command** |
| Lifetime | 1 year by default¹ | **at most** 1 year² |

¹ The Console pre-fills one year from today and accepts a later date; the
platform enforces no maximum on per-organization tokens.

² A hard cap: the platform refuses a global token expiring more than a year
out ("A global token may not be valid for more than 1 year"). Both limits are
the platform's, not the CLI's — see
[platform-all#629](https://gitlab.com/postgres-ai/platform-all/-/issues/629) /
[!709](https://gitlab.com/postgres-ai/platform-all/-/merge_requests/709).

A global token is bound to *you*, not to an organization, so working across
several companies no longer means re-running the browser login each time you
switch. The trade-off is that it cannot guess which organization a command
means, so it makes you say:

```bash
pgai issues list --org acme          # by alias
pgai reports list --org-id 5225      # by numeric id — same on every org-scoped command
PGAI_ORG=acme pgai checkup ...       # for scripts and agents

pgai orgs                            # which organizations can this token reach?
```

`--org` always means an alias and `--org-id` always means a numeric id.
Organization aliases have no format restriction, so an all-digit alias is
legal — one flag accepting both forms would have to guess, and guessing wrong
writes into the wrong organization.

Nothing is stored: there is no "current organization" to get out of sync, and
no shared state for parallel agents to race. Per-organization tokens are
unaffected and need no flag.

The MCP server follows the same rule — under a global token, `org_id` becomes a
required tool argument instead of falling back to a stored default.

### Start monitoring

Start monitoring with demo database:
```bash
postgresai mon local-install --demo
```

Start monitoring with your own database:
```bash
postgresai mon local-install --db-url postgresql://user:pass@host:5432/db
```

Complete automated setup with API key and database:
```bash
postgresai mon local-install --api-key your_key --db-url postgresql://user:pass@host:5432/db -y
```

This will:
- Configure API key for automated report uploads (if provided)
- Add PostgreSQL instance to monitor (if provided)
- Generate secure Grafana and replication passwords
- Start all monitoring services
- Open Grafana at http://localhost:3000

## Commands

### Monitoring services management (`mon` group)

#### Service lifecycle
```bash
# Complete setup with various options
postgresai mon local-install                                  # Interactive setup for production
postgresai mon local-install --demo                           # Demo mode with sample database
postgresai mon local-install --api-key <key>                  # Setup with API key
postgresai mon local-install --db-url <url>                   # Setup with database URL
postgresai mon local-install --api-key <key> --db-url <url>   # Complete automated setup
postgresai mon local-install -y                               # Auto-accept all defaults

# Service management
postgresai mon start                  # Start monitoring services
postgresai mon stop                   # Stop monitoring services
postgresai mon restart [service]      # Restart all or specific monitoring service
postgresai mon status                 # Show monitoring services status
postgresai mon health [--wait <sec>]  # Check monitoring services health
```

##### local-install options
- `--demo` - Demo mode with sample database (testing only, cannot use with --api-key)
- `--api-key <key>` - PostgresAI API key for automated report uploads
- `--db-url <url>` - PostgreSQL connection URL to monitor (format: `postgresql://user:pass@host:port/db`)
- `--instance-id <uuid>` - Adopt a console-provisioned monitoring instance (also via the `PGAI_INSTANCE_ID` env var)
- `-y, --yes` - Accept all defaults and skip interactive prompts

When `--instance-id <uuid>` (or `PGAI_INSTANCE_ID`) is set, `local-install` forwards the id to the platform, which **adopts** the already-provisioned monitoring instance instead of self-registering a duplicate under an auto-created `postgres-ai-monitoring` project. The CLI then persists the adopted instance's real project to `.pgwatch-config`, so checkup reports upload alongside the rest of that instance's health data. Adoption is awaited (with one automatic retry); if it fails, the CLI warns and reports fall back to the default project until you re-run `local-install`. Without the flag, the legacy self-registration path is byte-for-byte unchanged.

`local-install` writes `.env` in the monitoring directory. It preserves existing `REPLICATOR_PASSWORD` and `VM_AUTH_*` values or generates new random ones when missing; `VM_AUTH_USERNAME` defaults to `vmauth` when absent. The replication password is used by the demo PostgreSQL standby replication user, and the VM auth credentials are required before Docker Compose can provision Grafana datasources. If you run `docker compose` directly or maintain `.env` yourself, set both VM auth values before upgrading. For rotation, run `VM_AUTH_PASSWORD="$(openssl rand -base64 18)" ./scripts/rotate-vm-auth.sh` from the monitoring directory so `.env`, `sink-prometheus`, and `grafana` update together.

#### Monitoring target databases (`mon targets` subgroup)
```bash
postgresai mon targets list                       # List databases to monitor
postgresai mon targets add <conn-string> <name>   # Add database to monitor
postgresai mon targets remove <name>              # Remove monitoring target
postgresai mon targets test <name>                # Test target connectivity
```

#### Configuration and maintenance
```bash
postgresai mon config                         # Show monitoring configuration
postgresai mon update-config                  # Apply configuration changes
postgresai mon update                         # Update monitoring stack
postgresai mon reset [service]                # Reset service data
postgresai mon clean                          # Cleanup artifacts
postgresai mon check                          # System readiness check
postgresai mon shell <service>                # Open shell to monitoring service
```

### MCP server (`mcp` group)

```bash
postgresai mcp start                 # Start MCP stdio server exposing tools
```

Cursor configuration example (Settings → MCP):

```json
{
  "mcpServers": {
    "PostgresAI": {
      "command": "postgresai",
      "args": ["mcp", "start"],
      "env": {
        "PGAI_API_BASE_URL": "https://postgres.ai/api/general/"
      }
    }
  }
}
```

Every org-scoped tool takes an `org_id` argument. Under a per-organization
token it is optional (the token supplies the org); under a **global token it is
required** — the server will not assume an organization. Call `orgs_list` (or
run `pgai orgs`) to discover the available ids.

Tools exposed:
- `list_issues`: returns the same JSON as `postgresai issues list` (args: `{ org_id, status?, limit?, offset?, debug? }`).
- `view_issue`: view a single issue with its comments (args: `{ issue_id, org_id, debug? }`).
- `create_issue`: create a new issue (args: `{ title, description?, org_id, attachments?, debug? }`).
- `update_issue`: update title/description/status/labels (args: `{ issue_id, org_id, title?, description?, status?, labels?, attachments?, debug? }`).
- `post_issue_comment`: post a comment (args: `{ issue_id, org_id, content?, parent_comment_id?, attachments?, debug? }`).
- `update_issue_comment`: update an existing comment (args: `{ comment_id, org_id, content?, attachments?, debug? }`).
- `upload_file`: upload a local file and return the storage URL plus a ready-to-paste markdown link (args: `{ path, org_id, debug? }`).
- `download_file`: download a file from storage (args: `{ url, org_id, output_path?, debug? }`).

#### `attachments` parameter (issue/comment tools)

`create_issue`, `update_issue`, `post_issue_comment`, and `update_issue_comment` accept an
optional `attachments: string[]` of local file paths. Each file is uploaded to PostgresAI
storage and the resulting markdown link is appended to the comment body or issue
description (image extensions — `.png .jpg .jpeg .gif .webp .svg .bmp .ico` — render
inline as `![](url)`; everything else as `[](url)`).

For `post_issue_comment` and `update_issue_comment`, either `content` or `attachments`
must be non-empty (attachments alone are allowed). For `update_issue` with `attachments`
but no `description`, the existing description is fetched first and the new links are
appended to it.

#### Threat model

The MCP server runs in your local user account with your PostgresAI API key. It
treats the connected MCP client (the LLM agent) as **trusted** — the same way the
CLI treats you when you type a command. In particular:

- `upload_file` and the `attachments: string[]` parameter on the issue/comment tools
  read **any local file the CLI process can read**, including secrets like
  `~/.ssh/id_rsa`, `~/.aws/credentials`, or `~/.config/postgresai/config.json` (which
  contains your own API key). The file's bytes are uploaded to PostgresAI storage
  and the resulting URL becomes visible to anyone with read access to the issue or
  comment it ends up in.
- `download_file` writes to **any path the CLI process can write to** when
  `output_path` is supplied (`~/.ssh/authorized_keys`, `~/.bashrc`, etc. are all
  fair game). When `output_path` is omitted, downloads are restricted to the
  current working directory.

This is fine when the agent and the upstream context the agent is reading are
trusted. It is **not** safe to run this MCP server against an agent that is
processing untrusted text (issue bodies, comments, web pages, third-party docs)
without additional sandboxing — a prompt-injection in any input the agent reads
could be used to exfiltrate local secrets or write arbitrary files. If you need
to expose this MCP server to such an agent, run the agent (and this server) in a
container or restricted user account that doesn't have access to anything
sensitive.

### Issues management (`issues` group)

```bash
postgresai issues list                                       # List issues (shows: id, title, status, created_at)
postgresai issues view <issueId>                             # View issue details and comments
postgresai issues create --org-id <id> --title <t>           # Create a new issue
postgresai issues update <issueId> [--title ... --status ...]# Update an existing issue
postgresai issues post-comment <issueId> <content>           # Post a comment to an issue
postgresai issues update-comment <commentId> <content>       # Update an existing comment
postgresai issues files upload <path>                        # Upload a file, print URL + markdown
postgresai issues files download <url> [-o <path>]           # Download a file
# Common options:
#   --parent <uuid>  Parent comment ID (for replies on post-comment)
#   --debug          Enable debug output
#   --json           Output raw JSON (overrides default YAML)
```

#### Attaching files to issues and comments (`--attach`)

`create`, `update`, `post-comment`, and `update-comment` accept a repeatable
`--attach <path>` flag. Each file is uploaded to PostgresAI storage and a
markdown link is appended to the comment body (or issue description). Image
extensions — `.png .jpg .jpeg .gif .webp .svg .bmp .ico` — render inline as
`![](url)`; everything else as `[](url)`. Multiple `--attach` flags preserve
order; each link goes on its own line.

```bash
# Attach a screenshot to a new comment
postgresai issues post-comment <issueId> "Saw this in prod" --attach screenshot.png

# Attach multiple files to a new issue
postgresai issues create --org-id 4 --title "Slow query" \
  --description "Plan attached" --attach plan.txt --attach flame.svg

# Attach a file to an existing issue without changing the description.
# The current description is fetched and the link is appended to it.
postgresai issues update <issueId> --attach trace.log
```

#### Output format for issues commands

By default, issues commands print human-friendly YAML when writing to a terminal. For scripting, you can:

- Use `--json` to force JSON output:

```bash
postgresai issues list --json | jq '.[] | {id, title}'
```

- Rely on auto-detection: when stdout is not a TTY (e.g., piped or redirected), output is JSON automatically:

```bash
postgresai issues view <issueId> > issue.json
```

#### Grafana management
```bash
postgresai mon generate-grafana-password      # Generate new Grafana password
postgresai mon show-grafana-credentials       # Show Grafana credentials
```

### Authentication and API key management
```bash
postgresai auth                    # Authenticate via browser (OAuth)
postgresai auth --set-key <key>    # Store API key directly
postgresai show-key                # Show stored key (masked)
postgresai remove-key              # Remove stored key
```

## Configuration

The CLI stores configuration in `~/.config/postgresai/config.json` including:
- API key
- Base URL
- Organization ID

### Configuration priority

API key resolution order:
1. Command line option (`--api-key`)
2. Environment variable (`PGAI_API_KEY`)
3. User config file (`~/.config/postgresai/config.json`)
4. Legacy project config (`.pgwatch-config`)

Base URL resolution order:
- API base URL (`apiBaseUrl`):
  1. Command line option (`--api-base-url`)
  2. Environment variable (`PGAI_API_BASE_URL`)
  3. User config file `baseUrl` (`~/.config/postgresai/config.json`)
  4. Default: `https://postgres.ai/api/general/`
- UI base URL (`uiBaseUrl`):
  1. Command line option (`--ui-base-url`)
  2. Environment variable (`PGAI_UI_BASE_URL`)
  3. Default: `https://console.postgres.ai`

Normalization:
- A single trailing `/` is removed to ensure consistent path joining.

### Environment variables

- `PGAI_API_KEY` - API key for PostgresAI services
- `PGAI_ORG` - organization alias for org-specific commands (equivalent to `--org`; required with a global token)
- `PGAI_ORG_ID` - organization id for org-specific commands (equivalent to `--org-id`)
- `PGAI_API_BASE_URL` - API endpoint for backend RPC (default: `https://postgres.ai/api/general/`)
- `PGAI_UI_BASE_URL` - UI endpoint for browser routes (default: `https://console.postgres.ai`)

A flag beats the matching environment variable. Setting both `PGAI_ORG` and
`PGAI_ORG_ID` (or passing both flags) is an error rather than a silent
precedence win.

### Per-command options

Placed **after** the subcommand (`pgai projects --org acme`), not before it —
they are registered on each org-scoped command, so `pgai --org acme projects`
is an error.

- `--org <alias>` - organization alias; overrides `PGAI_ORG`
- `--org-id <id>` - organization id; overrides `PGAI_ORG_ID`

### CLI options

- `--api-base-url <url>` - overrides `PGAI_API_BASE_URL`
- `--ui-base-url <url>` - overrides `PGAI_UI_BASE_URL`

### Examples

For production (uses default URLs):

```bash
# Production auth - uses console.postgres.ai by default
postgresai auth --debug
```

For staging/development environments:

```bash
# Linux/macOS (bash/zsh)
export PGAI_API_BASE_URL=https://v2.postgres.ai/api/general/
export PGAI_UI_BASE_URL=https://console-dev.postgres.ai
postgresai auth --debug
```

```powershell
# Windows PowerShell
$env:PGAI_API_BASE_URL = "https://v2.postgres.ai/api/general/"
$env:PGAI_UI_BASE_URL = "https://console-dev.postgres.ai"
postgresai auth --debug
```

Via CLI options (overrides env):

```bash
postgresai auth --debug \
  --api-base-url https://v2.postgres.ai/api/general/ \
  --ui-base-url https://console-dev.postgres.ai
```

Notes:
- If `PGAI_UI_BASE_URL` is not set, the default is `https://console.postgres.ai`.

## Embedding checkup

Express checkup is designed to be embedded by **host applications** — for
example Rails/Django/Node applications and database-diagnostics admin UIs that
want to run PostgreSQL health checks and render the findings themselves. The
supported integration surface is the CLI's machine contract: run the checkup
command, read a single JSON document from stdout, and parse it against the
published JSON schemas.

### The `--no-upload --json` ABI

```bash
PGPASSWORD=... postgresai checkup \
  postgresql://monitoring_user@host:5432/dbname \
  --no-upload --json
```

- **stdout** carries **exactly one JSON object**, keyed by check ID:

  ```json
  {
    "H002": { "contract_version": "1.0.0", "checkId": "H002", "...": "..." },
    "F003": { "contract_version": "1.0.0", "checkId": "F003", "...": "..." }
  }
  ```

  It is a single document (not newline-delimited JSON). Each value is a report
  that validates against the schema shipped at
  `postgresai/schemas/<CHECK_ID>.schema.json`.
- Restrict the run to one check with `--check-id <ID>` (or the positional form
  `postgresai checkup <ID> <conn>`); stdout is then a one-key object.
- **stderr** carries only human-readable diagnostics (progress, warnings,
  errors). It never contains report JSON — machine consumers should read stdout
  only. Do not parse stderr as JSON.
- **Exit codes**: `0` on success; non-zero when the run fails (connection
  failure, insufficient permissions, an unknown/unavailable check ID, or a
  failing check). On a non-zero exit, no JSON report object is written to
  stdout.
- Pass `--no-upload` to keep the run fully local (no network calls to the
  PostgresAI API and no API key required).

### Passing credentials

Pass the database password via the **`PGPASSWORD`** environment variable (the
libpq standard), never on the command line. Credentials in `argv` are visible to
other processes (e.g. `ps`); `PGPASSWORD` is not. All other libpq environment
variables (`PGHOST`, `PGPORT`, `PGUSER`, `PGDATABASE`, `PGSSLMODE`) are also
honored.

### Permissions the connection needs

The checkup command runs a permissions preflight and expects a prepared
monitoring role. Provision it once with `prepare-db` (run as an admin/superuser):

```bash
PGPASSWORD=<admin-pw> postgresai prepare-db \
  postgresql://admin@host:5432/dbname \
  --monitoring-user postgres_ai_mon \
  --password <monitoring-pw>
```

This creates the monitoring role, the `postgres_ai` schema and helper
function(s), and grants the required read-only privileges (`pg_monitor`
membership, `SELECT` on the relevant catalogs/views, and the appropriate
`search_path`). Host applications then run checkup as that role. See
[prepare-db](#prepare-db-create-monitoring-user-in-postgres) for details and
provider-specific behavior.

### What runs locally vs. server-side

| Output | Where it is produced | Available offline (`--no-upload`) |
|--------|----------------------|-----------------------------------|
| Schema-valid JSON reports | Local (CLI) | Yes |
| Severity summaries (`summary`: `status` + `message`) | Local (CLI) | Yes |
| Local conclusions/recommendations for the checks that implement them (e.g. F003, H001) | Local (CLI) | Yes |
| Rich markdown analysis and prose recommendations | **Server-side** (PostgresAI API, via `--markdown`) | No — requires a network call and, for full detail, an API key |

In short: the local embed gives you structured JSON, per-check severity, and
the conclusions each check implements today. The full narrative analysis is an
API-side capability. Embedders that only need structured findings and severity
never have to call the API.

For every new or updated check, local JSON must include `conclusions` and
`recommendations`. Server-side markdown analysis can enrich those verdicts, but
must never be their only source. The coverage checklist in
[work item #285](https://gitlab.com/postgres-ai/postgresai/-/work_items/285)
tracks which existing checks have completed this migration.

### Loading a check schema

Every published `postgresai` package contains the JSON Schemas used by that
exact CLI build. Resolve schemas through the public package subpath instead of
depending on the tarball layout:

```js
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const schemaPath = require.resolve("postgresai/schemas/H002.schema.json");
const schema = JSON.parse(await readFile(schemaPath, "utf8"));
```

Schemas are generated during build and package lifecycle hooks. For an unbuilt
linked checkout or workspace dependency, run `npm run sync-schemas` first.

Replace `H002` with the report's `checkId`. The package ships one
`*.schema.json` file per check, plus `query.schema.json`. These files follow the
`contract_version` policy below: additive schema changes require a minor
contract-version bump, while breaking changes require a major bump.

### The versioned JSON contract

Every report envelope carries a **`contract_version`** (semver). This is the
public compatibility surface — the report envelope plus the per-check JSON
schemas shipped in `postgresai/schemas/` plus this stdout/stderr/exit-code ABI. It is
independent of the CLI/package `version`: the CLI can be released many times
without the contract changing.

Compatibility policy (semver applied to the contract, not the code):

- **PATCH** (`x.y.Z`) — editorial/no-op changes that cannot affect a consumer.
- **MINOR** (`x.Y.0`) — **additive, backward-compatible** changes: new optional
  fields in the envelope or a report, new checks, new schema files. Existing
  valid reports stay valid and existing consumers keep working untouched.
- **MAJOR** (`X.0.0`) — **breaking** changes: removing/renaming a field,
  tightening a type, making an optional field required, or changing the JSON ABI
  in a way that could break a consumer parsing the previous format.

A consumer should accept any report whose `contract_version` shares its **major**
and has a **minor ≥** the minimum it was built against. Pin the major, tolerate
additive minors, and treat a major bump as a required review.

The current contract version is **`1.0.0`**.

### Envelope fields

Beyond the check-specific `results`, every report includes:

| Field | Meaning |
|-------|---------|
| `contract_version` | Version of the JSON report contract (see above). |
| `checkId` / `checkTitle` | The check identifier and its human title. |
| `generation_mode` | `"express"` for CLI-generated reports. |
| `summary` | Optional `{ "status": "ok" \| "warning" \| "info", "message": string }` severity summary. |
| `timestamptz` | Report generation time (ISO 8601). |
| `nodes` | `{ "primary": string, "standbys": string[] }`. |
| `version` / `build_ts` | CLI/package version and build timestamp (may be null). |

## Development

### Testing

The CLI uses [Bun](https://bun.sh/) as the test runner with built-in coverage reporting.

```bash
# Run tests with coverage (default)
bun run test

# Run tests without coverage (faster iteration during development)
bun run test:fast

# Run tests with coverage and show report location
bun run test:coverage
```

Coverage configuration is in `bunfig.toml`. Reports are generated in `coverage/` directory:
- `coverage/lcov-report/index.html` - HTML coverage report
- `coverage/lcov.info` - LCOV format for CI integration

## Requirements

- Node.js 18 or higher
- Docker and Docker Compose

## Feedback

Have an idea or found a rough edge? Run `pgai feedback` (or `pgai feedback --open`) or share it directly at https://gitlab.com/postgres-ai/postgresai/-/work_items/300. Set `PGAI_NO_FEEDBACK_TIP=1` to silence the occasional in-CLI reminder.

## Learn more

- Documentation: https://postgres.ai/docs
- Issues: https://gitlab.com/postgres-ai/postgres_ai/-/issues
- Ideas / feedback: https://gitlab.com/postgres-ai/postgresai/-/work_items/300
