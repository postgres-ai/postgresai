# PostgresAI CLI

Command-line interface for PostgresAI monitoring and database management.

## Installation

### From npm

```bash
npm install -g postgresai
```

Or install the latest beta release explicitly:
```bash
npm install -g postgresai@beta
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
- Prompt you to select an organization
- Automatically save your API key to `~/.config/postgresai/config.json`

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
- Generate secure Grafana password
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
- `--api-key <key>` - Postgres AI API key for automated report uploads
- `--db-url <url>` - PostgreSQL connection URL to monitor (format: `postgresql://user:pass@host:port/db`)
- `-y, --yes` - Accept all defaults and skip interactive prompts

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

Tools exposed:
- list_issues: returns the same JSON as `postgresai issues list`.
- view_issue: view a single issue with its comments (args: { issue_id, debug? })
- post_issue_comment: post a comment (args: { issue_id, content, parent_comment_id?, debug? })

### Issues management (`issues` group)

```bash
postgresai issues list                                  # List issues (shows: id, title, status, created_at)
postgresai issues view <issueId>                        # View issue details and comments
postgresai issues post_comment <issueId> <content>      # Post a comment to an issue
# Options:
#   --parent <uuid>  Parent comment ID (for replies)
#   --debug          Enable debug output
#   --json           Output raw JSON (overrides default YAML)
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
- `PGAI_API_BASE_URL` - API endpoint for backend RPC (default: `https://postgres.ai/api/general/`)
- `PGAI_UI_BASE_URL` - UI endpoint for browser routes (default: `https://console.postgres.ai`)

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

## Learn more

- Documentation: https://postgres.ai/docs
- Issues: https://gitlab.com/postgres-ai/postgresai/-/issues
