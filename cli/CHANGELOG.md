# Changelog

## Unreleased

### Changed

- `issues list` now shows **open issues only** by default. Closed issues need
  an explicit `--status closed` (or `--status all` for both); an unknown
  `--status` value is rejected instead of silently listing everything.
  The MCP `list_issues` tool keeps its "omit status = all" behaviour.
- `issues list` / `issues view` and the MCP `list_issues` / `view_issue` tools
  render `status` as `open` / `closed` instead of the raw `0` / `1`. Scripts
  that compared `.status == 0` in the JSON output need updating. (#367)

### Added

- `pgai promql <expr>` runs a PromQL query on a monitoring instance through the
  platform and prints the result — `--range --start --end --step` for a matrix,
  `--at` for an instant query at a given time, `--json` for every point. The
  instance is taken from `--instance`, or from `.pgwatch-config` when the
  command runs on the box itself. The wait is sized from the pacing estimate the
  platform returns, so a 5s-polling instance answers in about a minute while the
  600s fleet default waits longer; `--timeout` overrides it. A result larger than
  the box's byte budget comes back trimmed and flagged `truncated`, not refused.
  **Requires platform-all !809**
  (https://gitlab.com/postgres-ai/platform-all/-/merge_requests/809), which adds
  `v1.instance_query_enqueue` / `v1.instance_query_result`. Until that is
  deployed the command fails with a PGRST202 404 rather than degrading. (#378)

- `issues create --hidden` creates a staff-only hidden issue, and
  `issues update --hidden` / `--no-hidden` hides or unhides an existing one
  (MCP: optional `is_hidden` on `create_issue` / `update_issue`). The CLI does
  no staff detection: the platform's `user_is_staff()` guard on
  `issue_create` / `issue_update` (platform-all #562) refuses a visibility
  change for a non-staff credential and the CLI surfaces that error as-is. An update that
  does not mention the flag never touches it (postgresai #365).

- `issues list` and `issues view` (and their MCP counterparts `list_issues` /
  `view_issue`) now surface the staff-only hidden-issue flag. It is rendered
  **only when true**: hidden issues are filtered out server-side for everyone
  but PostgresAI staff, so a non-staff response can only ever carry
  `is_hidden: false`, and printing that would disclose that the mechanism
  exists. `issues list --hidden-only` (MCP: `list_issues` with
  `hidden_only: true`) lists just the hidden ones.

  Requires platform-all !712, which resolves staff from the access-token
  header. Until it is deployed the CLI degrades silently rather than erroring:
  `--hidden-only` returns an empty list and `is_hidden` never appears. The
  same silence applies to a credential that does not qualify as staff — the
  token must be personal, live, and, if it is a per-organization token, issued
  on or after 2026-08-14.

  Issue requests now carry `x-pgai-include-hidden`, the server's opt-in for
  hidden rows. It is a client capability declaration — "this client will mark
  hidden issues" — not a user preference. The platform default-excludes hidden
  rows from any token caller that omits it, so older CLI versions (and curl,
  scripts, MCP clients) keep seeing exactly what they see today rather than
  receiving staff-internal issues they would render as ordinary ones.

### Fixed

- `mon targets add` / `mon targets remove` now leave `instances.yml` owner-only
  (`0600`) on every write, tightening a pre-existing looser file before the new
  content lands. The file holds password-bearing `conn_str` values and was
  previously created at the ambient umask. A chmod the CLI is not permitted to
  perform (foreign-owned file) is warned about, not fatal.

- `checkup --markdown` previously performed server-side conversion and sent the
  full report JSON to the PostgresAI API even when `--no-upload` was set. The
  flags are now mutually exclusive, and `--no-upload` prevents report data from
  being sent to the PostgresAI API. Use `--json` or `--output` for local-only
  output.
