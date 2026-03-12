#!/bin/bash

set -euo pipefail

# pg_index_pilot unified installer
#
# Subcommands:
#   install-control   - Create control DB, install schema and functions
#   register-target   - Create FDW server and register a target DB
#   verify            - Run version/permissions/environment checks
#   uninstall         - Remove schema and related servers (best effort)
#
# Common options:
#   -H, --host            PostgreSQL host
#   -P, --port            PostgreSQL port
#   -U, --user            PostgreSQL user
#   -W, --password        PostgreSQL password (env var preferred)
#   -C, --control-db      Control database name
#   --fdw-host            Hostname to use for FDW servers (default: same as --host)
#   --no-create-db        Do not attempt to create the control DB
#   -q, --quiet           Less verbose psql output
#
# register-target options:
#   -T, --target-db       Target database name (required for register-target)
#   --server-name         FDW server name to create (default: target_<target-db>)
#   --force               Recreate FDW server and update registration
#
# uninstall options:
#   --drop-servers        Attempt to drop FDW servers referenced in target_databases
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly RED='\033[0;31m'
readonly NC='\033[0m'

######################## Functions ########################

# Description: Print informational message to STDOUT in yellow.
# Globals: YELLOW, NC
# Args: $*  Message text
# Outputs: Writes colored message to STDOUT
# Returns: 0
print_info() { echo -e "${YELLOW}$*${NC}"; }

# Description: Print success message to STDOUT in green.
# Globals: GREEN, NC
# Args: $*  Message text
# Outputs: Writes colored message to STDOUT
# Returns: 0
print_ok() { echo -e "${GREEN}$*${NC}"; }

# Description: Print error message to STDERR in red.
# Globals: RED, NC
# Args: $*  Message text
# Outputs: Writes colored message to STDERR
# Returns: 0
print_err() { echo -e "${RED}$*${NC}" 1>&2; }

##
# Description: Show usage information for the installer.
# Globals: none
# Args: none
# Outputs: Usage text to STDOUT
# Returns: 0
usage() {
  cat << USAGE
Usage: $0 <subcommand> [options]

Subcommands:
  install-control     Install schema/functions into control DB
  register-target     Register a target DB via postgres_fdw
  verify              Verify version, permissions, FDW and environment
  uninstall           Uninstall from control DB (and optionally drop servers)

Common options:
  -H, --host HOST           PostgreSQL host
  -P, --port PORT           PostgreSQL port
  -U, --user USER           PostgreSQL user
  -W, --password PASS       PostgreSQL password (prefer PGPASSWORD env)
  -C, --control-db NAME     Control database name
  --fdw-host HOST           Hostname to use inside FDW (default: same as --host)
  --no-create-db            Do not create control DB if missing
  -q, --quiet               Less verbose psql output

register-target options:
  -T, --target-db NAME      Target database name (required)
  --server-name NAME        FDW server name (default: target_<target-db>)
  --force                   Recreate FDW server and upsert registration

uninstall options:
  --drop-servers            Attempt to drop FDW servers from target_databases

Environment:
  PGPASSWORD                PostgreSQL password (safer than -W)

Examples:
  $0 install-control -H db.example.com -U postgres -C index_pilot_control
  $0 register-target -H db.example.com -U postgres -C index_pilot_control -T appdb --fdw-host 127.0.0.1
  $0 verify -H db.example.com -U postgres -C index_pilot_control
  $0 uninstall -H db.example.com -U postgres -C index_pilot_control --drop-servers
USAGE
}

# Defaults
DB_HOST="localhost"
DB_PORT="5432"
DB_USER="${USER:-postgres}"
DB_PASS="${PGPASSWORD:-}"
CONTROL_DB="index_pilot_control"
FDW_HOST_OVERRIDE=""
QUIET_ARGS=("-X")
CREATE_DB=true

# register-target vars
TARGET_DB=""
SERVER_NAME=""
FORCE=false

# uninstall vars
DROP_SERVERS=false

##
# Description: Ensure psql is available in PATH.
# Globals: none
# Args: none
# Outputs: Error to STDERR if psql is missing
# Returns: Exits 1 on error
require_psql() {
  if ! command -v psql > /dev/null 2>&1; then
    print_err "psql not found in PATH"
    exit 1
  fi
}

##
# Description: Export PGPASSWORD if a password is provided.
# Globals: DB_PASS
# Args: none
# Outputs: Sets environment variable PGPASSWORD when DB_PASS is non-empty
# Returns: 0
export_password_if_set() {
  if [[ -n "${DB_PASS}" ]]; then
    export PGPASSWORD="${DB_PASS}"
  fi
}

##
# Description: Execute a single SQL command via psql with standard flags.
# Globals: QUIET_ARGS, DB_HOST, DB_PORT, DB_USER
# Args: $1 database name; $2 SQL; $@ optional extra psql args
# Outputs: psql output to STDOUT/STDERR
# Returns: psql's exit status
psql_cmd() {
  local db="$1"
  shift
  local sql="$1"
  shift || true
  # Allow passing psql -v vars via "$@"; psql expects options before -c
  psql "${QUIET_ARGS[@]}" -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${db}" "$@" -v ON_ERROR_STOP=on -At -c "${sql}"
}

##
# Description: Execute a SQL file via psql with standard flags.
# Globals: QUIET_ARGS, DB_HOST, DB_PORT, DB_USER
# Args: $1 database name; $2 path to SQL file; $@ optional extra psql args
# Outputs: psql output to STDOUT/STDERR
# Returns: psql's exit status
psql_file() {
  local db="$1"
  shift
  local file="$1"
  shift
  psql "${QUIET_ARGS[@]}" -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${db}" -v ON_ERROR_STOP=on -f "${file}" "$@"
}

##
# Description: Create control database if it does not exist and allowed.
# Globals: CREATE_DB, CONTROL_DB
# Args: none
# Outputs: Status messages
# Returns: 0 on success
ensure_control_db() {
  if [[ "${CREATE_DB}" != true ]]; then
    return 0
  fi
  local exists
  exists=$(psql_cmd postgres "select 1 from pg_database where datname = '${CONTROL_DB//\'/\'\'}'") || true
  if [[ -z "${exists}" ]]; then
    print_info "Creating control database ${CONTROL_DB}"
    psql_cmd postgres "create database \"${CONTROL_DB}\";"
    print_ok "Control database created: ${CONTROL_DB}"
  else
    print_ok "Control database exists: ${CONTROL_DB}"
  fi
}

##
# Description: Install schema, functions, and FDW in control database.
# Globals: DB_HOST, DB_PORT, DB_USER, DB_PASS, CONTROL_DB, SCRIPT_DIR
# Args: none
# Outputs: Progress and errors
# Returns: Exits on error
install_control() {
  # Hard fail early if required inputs are missing (before any DB calls)
  if [[ -z "${DB_HOST}" ]]; then
    print_err "Host is required (-H)."
    exit 1
  fi
  if [[ -z "${DB_PORT}" || ! "${DB_PORT}" =~ ^[0-9]+$ ]]; then
    print_err "Valid port is required (-P)."
    exit 1
  fi
  if [[ -z "${DB_USER}" ]]; then
    print_err "User is required (-U)."
    exit 1
  fi
  if [[ -z "${CONTROL_DB}" ]]; then
    print_err "Control database name is required (-C)."
    exit 1
  fi
  if [[ -z "${DB_PASS}" ]]; then
    print_err "Password is required for install-control. Provide -W or set PGPASSWORD."
    exit 1
  fi
  ensure_control_db

  print_info "Ensuring required extensions in ${CONTROL_DB}"
  psql_cmd "${CONTROL_DB}" "create extension if not exists dblink;"
  psql_cmd "${CONTROL_DB}" "create extension if not exists postgres_fdw;"

  print_info "Installing schema and functions into ${CONTROL_DB}"
  psql_file "${CONTROL_DB}" "${SCRIPT_DIR}/index_pilot_tables.sql"
  psql_file "${CONTROL_DB}" "${SCRIPT_DIR}/index_pilot_functions.sql"
  psql_file "${CONTROL_DB}" "${SCRIPT_DIR}/index_pilot_fdw.sql"

  print_info "Verifying installation"
  psql_cmd "${CONTROL_DB}" "select 'Version: '||index_pilot.version();"
  psql_cmd "${CONTROL_DB}" "select * from index_pilot.check_fdw_security_status();" | sed 's/^/  /'
  psql_cmd "${CONTROL_DB}" "select * from index_pilot.check_environment();" | sed 's/^/  /'
  print_ok "Installation complete in ${CONTROL_DB}"
}

##
# Description: Sanitize server name to contain only [a-zA-Z0-9_].
# Globals: none
# Args: $1 raw name
# Outputs: Sanitized name to STDOUT
# Returns: 0
sanitize_server_name() {
  local name="$1"
  name="${name//[^a-zA-Z0-9_]/_}"
  echo "${name}"
}

##
# Description: Create/ensure FDW server, user mapping, and register target DB.
# Globals: TARGET_DB, DB_HOST, DB_PORT, DB_USER, DB_PASS, CONTROL_DB, FDW_HOST_OVERRIDE
# Args: none
# Outputs: Progress and errors
# Returns: Exits on error when required inputs missing or test fails
register_target() {
  if [[ -z "${TARGET_DB}" ]]; then
    print_err "--target-db is required for register-target"
    exit 1
  fi
  if [[ -z "${DB_HOST}" ]]; then
    print_err "Host is required (-H)."
    exit 1
  fi
  if [[ -z "${DB_PORT}" || ! "${DB_PORT}" =~ ^[0-9]+$ ]]; then
    print_err "Valid port is required (-P)."
    exit 1
  fi
  if [[ -z "${DB_USER}" ]]; then
    print_err "User is required (-U)."
    exit 1
  fi
  if [[ -z "${CONTROL_DB}" ]]; then
    print_err "Control database name is required (-C)."
    exit 1
  fi
  if [[ -z "${DB_PASS}" ]]; then
    print_err "Password is required for register-target. Provide -W or set PGPASSWORD."
    exit 1
  fi
  local fdw_host
  fdw_host="${FDW_HOST_OVERRIDE:-${DB_HOST}}"

  local server
  if [[ -n "${SERVER_NAME}" ]]; then
    server="${SERVER_NAME}"
  else
    server="target_$(sanitize_server_name "${TARGET_DB}")"
  fi

  print_info "Configuring FDW server '${server}' to ${fdw_host}:${DB_PORT} dbname=${TARGET_DB}"
  # Ensure current user has FDW usage (useful on managed services)
  psql_cmd "${CONTROL_DB}" "grant usage on foreign data wrapper postgres_fdw to \"${DB_USER}\";" || true
  if [[ "${FORCE}" == true ]]; then
    psql_cmd "${CONTROL_DB}" "drop server if exists ${server} cascade;" || true
  fi

  # Create or replace server
  psql_cmd "${CONTROL_DB}" "do \$\$ begin
    if exists (select 1 from pg_foreign_server where srvname = '${server}') then
      perform 1; -- exists, keep as-is
    else
      execute format('create server %I foreign data wrapper postgres_fdw options (host %L, port %L, dbname %L)', '${server}', '${fdw_host}', '${DB_PORT}', '${TARGET_DB}');
    end if;
  end \$\$;"

  # Upsert into target_databases
  psql_cmd "${CONTROL_DB}" "insert into index_pilot.target_databases(database_name, host, port, fdw_server_name, enabled)
    values ('${TARGET_DB}', '${fdw_host}', ${DB_PORT}, '${server}', true)
    on conflict (database_name) do update set host=excluded.host, port=excluded.port, fdw_server_name=excluded.fdw_server_name, enabled=true;"

  # User mapping for current_user (managed-services friendly)
  if [[ -n "${DB_PASS}" ]]; then
    print_info "Ensuring user mapping for current_user on server ${server}"
    psql_cmd "${CONTROL_DB}" "drop user mapping if exists for current_user server ${server};" || true
    psql_cmd "${CONTROL_DB}" "create user mapping for current_user server ${server} options (user '${DB_USER}', password '${DB_PASS}');" || true
  else
    print_info "Skipping user mapping password setup (PGPASSWORD not set)"
  fi

  print_info "Testing secure FDW connection via internal helper"
  if psql_cmd "${CONTROL_DB}" "select index_pilot._connect_securely('${TARGET_DB}'::name);" > /dev/null 2>&1; then
    print_ok "FDW connection OK for target database '${TARGET_DB}'"
  else
    print_err "FDW connection test failed. Verify user mapping and network access."
    exit 1
  fi
}

##
# Description: Verify version, permissions, FDW security and environment.
# Globals: DB_HOST, DB_PORT, DB_USER, DB_PASS, CONTROL_DB
# Args: none
# Outputs: Checks summary
# Returns: Exits on error when required inputs missing
verify_all() {
  if [[ -z "${DB_HOST}" ]]; then
    print_err "Host is required (-H)."
    exit 1
  fi
  if [[ -z "${DB_PORT}" || ! "${DB_PORT}" =~ ^[0-9]+$ ]]; then
    print_err "Valid port is required (-P)."
    exit 1
  fi
  if [[ -z "${DB_USER}" ]]; then
    print_err "User is required (-U)."
    exit 1
  fi
  if [[ -z "${CONTROL_DB}" ]]; then
    print_err "Control database name is required (-C)."
    exit 1
  fi
  if [[ -z "${DB_PASS}" ]]; then
    print_err "Password is required for verify. Provide -W or set PGPASSWORD."
    exit 1
  fi
  print_info "Checking version"
  psql_cmd "${CONTROL_DB}" "select index_pilot.version();"

  print_info "Checking permissions"
  psql_cmd "${CONTROL_DB}" "select * from index_pilot.check_permissions();" | sed 's/^/  /'

  print_info "Checking FDW security status"
  psql_cmd "${CONTROL_DB}" "select * from index_pilot.check_fdw_security_status();" | sed 's/^/  /'

  print_info "Environment check"
  psql_cmd "${CONTROL_DB}" "select * from index_pilot.check_environment();" | sed 's/^/  /'

  print_ok "Verification complete"
}

##
# Description: Uninstall schema from control DB and optionally drop FDW servers.
# Globals: DB_HOST, DB_PORT, DB_USER, DB_PASS, CONTROL_DB, DROP_SERVERS, SCRIPT_DIR
# Args: none
# Outputs: Progress and errors
# Returns: Exits on error when required inputs missing
uninstall_all() {
  if [[ -z "${DB_HOST}" ]]; then
    print_err "Host is required (-H)."
    exit 1
  fi
  if [[ -z "${DB_PORT}" || ! "${DB_PORT}" =~ ^[0-9]+$ ]]; then
    print_err "Valid port is required (-P)."
    exit 1
  fi
  if [[ -z "${DB_USER}" ]]; then
    print_err "User is required (-U)."
    exit 1
  fi
  if [[ -z "${CONTROL_DB}" ]]; then
    print_err "Control database name is required (-C)."
    exit 1
  fi
  if [[ -z "${DB_PASS}" ]]; then
    print_err "Password is required for uninstall. Provide -W or set PGPASSWORD."
    exit 1
  fi
  print_info "Running uninstall in ${CONTROL_DB}"
  psql_file "${CONTROL_DB}" "${SCRIPT_DIR}/uninstall.sql"

  if [[ "${DROP_SERVERS}" == true ]]; then
    print_info "Dropping FDW servers referenced by index_pilot.target_databases (best effort)"
    psql_cmd "${CONTROL_DB}" "do \$\$ declare r record; begin
      for r in select fdw_server_name from index_pilot.target_databases loop
        begin
          execute format('drop server if exists %I cascade', r.fdw_server_name);
        exception when others then
          perform 1;
        end;
      end loop;
    exception when undefined_table then
      perform 1;
    end \$\$;" || true
    print_ok "FDW servers drop attempted"
  fi

  print_ok "Uninstall complete"
}

##
# Description: Parse CLI arguments to set globals and subcommand.
# Globals: SUBCOMMAND, DB_HOST, DB_PORT, DB_USER, DB_PASS, CONTROL_DB, FDW_HOST_OVERRIDE,
#          CREATE_DB, QUIET_ARGS, TARGET_DB, SERVER_NAME, FORCE, DROP_SERVERS
# Args: $@ CLI arguments
# Outputs: May print usage or errors
# Returns: Exits on unknown args
parse_args() {
  local positional=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      install-control | register-target | verify | uninstall)
        SUBCOMMAND="$1"
        shift
        break
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *) break ;;
    esac
  done

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -H | --host)
        DB_HOST="$2"
        shift 2
        ;;
      -P | --port)
        DB_PORT="$2"
        shift 2
        ;;
      -U | --user)
        DB_USER="$2"
        shift 2
        ;;
      -W | --password)
        DB_PASS="$2"
        shift 2
        ;;
      -C | --control-db)
        CONTROL_DB="$2"
        shift 2
        ;;
      --fdw-host)
        FDW_HOST_OVERRIDE="$2"
        shift 2
        ;;
      --no-create-db)
        CREATE_DB=false
        shift
        ;;
      -q | --quiet)
        QUIET_ARGS=("-q" "-X")
        shift
        ;;
      -T | --target-db)
        TARGET_DB="$2"
        shift 2
        ;;
      --server-name)
        SERVER_NAME="$2"
        shift 2
        ;;
      --force)
        FORCE=true
        shift
        ;;
      --drop-servers)
        DROP_SERVERS=true
        shift
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        positional+=("$1")
        shift
        ;;
    esac
  done

  if [[ ${#positional[@]} -gt 0 ]]; then
    print_err "Unexpected arguments: ${positional[*]}"
    usage
    exit 1
  fi
}

##
# Description: Entry point. Parses args, validates environment, dispatches subcommands.
# Globals: SUBCOMMAND
# Args: $@ CLI arguments
# Outputs: Progress and errors
# Returns: Exits non-zero on failure
main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi
  parse_args "$@"
  require_psql
  export_password_if_set

  case "${SUBCOMMAND:-}" in
    install-control)
      install_control
      ;;
    register-target)
      register_target
      ;;
    verify)
      verify_all
      ;;
    uninstall)
      uninstall_all
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
