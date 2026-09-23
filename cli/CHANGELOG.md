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

- `mon local-install --instance-jobs` / `--no-instance-jobs` (or
  `PGAI_INSTANCE_JOBS`) turns the outbound collection channel on or off for a
  machine. It writes `COMPOSE_PROFILES` into the stack `.env` rather than
  passing `--profile` on every command: compose reads that key itself, so a
  plain `up -d` / `pull` / `down` then covers the `instance-jobs` container —
  `mon stop` removes it and `mon start` brings it back, and `mon update` pulls
  its image and starts it with a scoped `up -d --no-deps` (`pull` cannot create
  a service that was not there before, and `mon restart` never creates a
  missing container). Disabling also removes the container by name: taking the
  profile out of `.env` does not stop one that is already running, and a later
  `up -d --force-recreate` leaves it up. Do not rely on `down` to reach a
  profile-gated container — it was observed both reaching and not reaching one
  on the same engine and compose versions.

  Passing neither the flag nor the env var expresses no opinion: the existing
  value is carried over and any other profile listed there survives, so an
  upgrade or a routine re-install never turns the channel on or off. Enabling
  on a fleet machine still goes through the ansible playbook
  (`instance_jobs_enabled`), because a hand-run install without the instance id
  self-registers a duplicate instance. The platform's
  `app.settings.instance_jobs_enabled` remains a separate gate. (#381)

- `mon health` reports the `instance-jobs` container as a fault — `enabled but
  not running` — when the compose profile is on and the container is absent,
  instead of `- not enabled`. `mon stop` deleting that container is the most
  likely way the channel stops, and the health line used to be blind to it.
  With the profile off it is still skipped. (#381)

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

- `mon local-install` no longer throws away what the platform says about AAS
  auto-collection. A refused registration now reports the platform's own reason
  (`platform returned HTTP 400 — PT400: <detail>`) instead of a bare status —
  the error text is scrubbed of the API key and any `glsa_` service-account
  token, flattened to one line and capped, so a platform that echoes the request
  cannot leak either. A successful one reports what was actually armed: when the
  platform has no vCPU count for the source database it now says
  `collection stays OFF until a source-DB vCPU count is known` instead of a bare
  "registered", because the producer skips those instances with `no_vcpus`
  indefinitely, and it names which channel was armed (`pull` vs the instance job
  channel). An older platform that reports neither field behaves as before.
  (#348, #382, postgres-ai/platform-all#778)

- The platform error text printed by `mon local-install` is normalised before it
  is scrubbed, and credentials are matched whitespace-tolerantly. An error body
  that echoed a credential with a line break inside it previously defeated the
  scrub and was then rejoined into one line, leaving the token one space-deletion
  from usable, and a credential sitting behind a credential-named key
  (`sa_token: <wrapped value>`) no longer kept its tail. Control characters (ESC,
  BEL, NUL, DEL, C1) are also neutralised, so platform-supplied text can no
  longer repaint the operator's terminal. (#382)

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
