# Changelog

## Unreleased

### Added

- "Seen by" on issues (https://gitlab.com/postgres-ai/platform-all/-/work_items/1085): `pgai issues view` records that
  the token owner opened the issue, via CLI, or via AI agent when an AI coding agent runs pgai (`AI_AGENT`,
  `CLAUDECODE`, `CURSOR_AGENT`, `GEMINI_CLI`, `CODEX_SANDBOX` or `CODEX_THREAD_ID` set; `PGAI_VIA=cli|agent` decides
  when set). It prints `issue.seen_by` (`user_id`, `name`, `via` = console, cli or agent, `first_seen_at`; first view
  per person, oldest first; newer platforms add `last_seen_at` and `last_via`, the latest open). The MCP `view_issue`
  tool does the same as an AI agent.
  Shared (non-personal) org tokens are not recorded. On a platform without this feature `seen_by` is omitted and the
  issue prints as before.
- `pgai billing status` says whether the organization has a payment method on file
  (https://gitlab.com/postgres-ai/postgresai/-/work_items/462): `--json` gives `org`, `has_payment_method` and
  `billing_url`; exit 0 yes, 3 no, 1 error or unknown. It is read-only and checks no promo code. The `postgresai` agent
  skill uses it to wait for the card the user is adding. Before, the skill re-ran the deploy with `--coupon`, which
  used up the platform's coupon checks within minutes and then read the refusal as "card added".
- `pgai mon deploy` monitors an Amazon RDS that is not publicly accessible over AWS PrivateLink
  (https://gitlab.com/postgres-ai/postgresai/-/work_items/454), as the console's wizard does: the price, a CloudFormation
  link that creates the stack in your AWS account (`--terraform` writes the same as Terraform), the `postgres_ai_mon`
  role with the organization's ownership marker, and the box on AWS in the RDS's region. From a machine that can't reach
  the database, it prints the setup SQL to run where the database is reachable, and the box checks it over PrivateLink
  (https://gitlab.com/postgres-ai/postgresai/-/work_items/459). With the aws CLI it reads the instance and the stack (read-only; `--aws-profile`, `--no-aws`), fills the
  link in, and waits for the stack at a terminal; otherwise it asks public DNS whether the RDS is public, and for the
  stack's `VpceServiceName` (`--vpce-service`). Scripts and agents get exit 3 with `"step":"privatelink_stack"`
  (`cfn_url`) or `"step":"prepare_db"` (`sql`, `nonce`, `monitoring_password`; resume with `PGAI_MON_PASSWORD` and
  `PGAI_MON_NONCE`). `--host-stats` adds host CPU, memory, disk and IOPS through a read-only role the stack creates.
  `--privatelink` / `--no-privatelink` decide instead. The MCP `deploy_monitoring` tool takes the same arguments.
  `pgai mon instances list` shows a PrivateLink box's `privatelink` (region, service, verification), `watch` says why
  a box failed its check of the database, and `delete` names the stack to delete afterwards.
  Deploy finds the stack under any name by its RDS; with several for the RDS, a person picks one and a script gets them
  listed (`stacks`). A stack with deploy's name for another RDS is never used.

### Changed

- A publicly accessible RDS is monitored directly (as `--provider generic`), not handed off to the console. There the
  RDS master user (a member of `rds_superuser`) counts as an admin before `pg_stat_statements` is installed, as it can
  create it.
- `--monitoring-user` with a private RDS is refused (over PrivateLink the box uses `postgres_ai_mon`); it used to be
  ignored with a note.

### Removed

- Checkup reports are JSON only (https://gitlab.com/postgres-ai/postgresai/-/work_items/458): `pgai checkup
  --markdown` is gone and now fails at once with a message pointing to `--json` and `--output`. `pgai reports files`
  and `pgai reports data` return JSON files by default in every mode (the terminal default was markdown; `--json` and
  `--output` used to return every type); `--type md` still returns markdown that the legacy postgres-checkup tool
  uploaded. The MCP report tools default to JSON and point to the issues tools for recommendations. Uploads no longer
  send `generate_issue`.

### Security

- `prepare-db` refuses a preexisting `postgres_ai` schema unless the connecting administrator owns it, no other role
  (including PUBLIC) has CREATE there, and every object in it belongs to that administrator. This includes operators,
  types, and other object kinds beyond tables and functions, including objects owned by predefined roles that have no
  ownership dependency row. Extensions installed inside `postgres_ai` are also refused. Existing installations using a `pg_database_owner`-owned
  schema may need a DBA to inspect the schema and transfer ownership before running `prepare-db`; the CLI prints the
  specific owner, grantee, object or extension and a recovery hint. Do not transfer ownership before inspecting existing objects
  and grants.

### Fixed

- `pgai mon deploy` over AWS PrivateLink provisions the box with the nonce of the setup SQL that ran, so a newer SQL
  issued since (another run, the console wizard) no longer makes the box fail its ownership check
  (https://gitlab.com/postgres-ai/platform-all/-/work_items/1058). `pgai mon instances list`, `watch` and `delete` read a
  PrivateLink box's route from the list, and a box made in the console shows its server and database. Needs the platform
  release with that change.

- `pgai mon deploy` no longer reports "✗ Create the monitoring box … request timed out after 25000ms" when the box
  request is slow but the platform creates the box (https://gitlab.com/postgres-ai/postgresai/-/work_items/443). The
  box request may take 55 s; with no answer (a timeout, a proxy's 502/504/524) deploy says it is still creating the box
  and looks for it for up to 5 minutes (30 s from the MCP tool), whatever `--wait` is, then follows it as usual. A box
  that does not appear ends with `action_required` (exit 3) and the next step, instead of `failed`. A 5xx the platform
  raised (with its SQLSTATE, such as PT503) is still reported at once, as before.

### Added

- `pgai mon deploy --cloud aws|hetzner` (https://gitlab.com/postgres-ai/platform-all/-/issues/972): the platform decides
  where the monitoring box runs, as the Console does: the database's cloud and region (a Supabase project's region,
  an RDS, Aurora, EC2 or pooler host), else AWS us-east-1. Deploy prints `Monitoring box: <cloud>, <location> (<why>)`
  before the price, and the result carries `cloud`, `location` and `location_reason`. `--location` is a location of
  that cloud (an AWS region, or `fsn1`, `nbg1`, `hel1`, `ash`, `hil` on Hetzner). A cloud that is not available, or a
  location of another cloud, is refused before anything is created, with the flag to use. `--location` now works for
  Supabase too.
- `pgai mon deploy` asks before it creates a box in PostgresAI Cloud, free or billed (a platform without the cloud
  choice still asks only for a billed box): at a terminal "Deploy <db> here? [y]es / [c]hoose
  another location / [N]o" (`c` lists the clouds and their locations), or "Deploy <db> on <cloud>, <location>? [y/N]"
  with `--cloud`/`--location`. Not at a terminal (piped, `--json`, the MCP tool) and without `--yes`, nothing is
  created: it stops (exit 3) with the place, `location_reason_text`, the price and `choices`. Before, a box on a free
  slot was created without a question. The MCP tool `deploy_monitoring` gains `cloud` and `location`.

- `pgai mon deploy --provider supabase` without a database URL (https://gitlab.com/postgres-ai/postgresai/-/work_items/438),
  like the Console: it connects the Supabase organization if needed, then lists its projects. At a terminal you pick
  one from a numbered list; agents and scripts pass `--supabase-project <name or project ID>` (it implies
  `--provider supabase`). Without either, with `--yes`, or not at a terminal, it stops (exit 3) with the list in the JSON result
  (`supabase_org`, and `projects` with `ref`, `name`, `region`, `status`, `selectable`, `reason`, `monitored`).
  Paused and not-ready projects are listed but can't be picked; an already-monitored project returns its
  monitoring. A URL still works as before. Wording: the consent failure now reads "Supabase did not connect:
  <why>. Re-run to try again", and the missing-URL error mentions `--provider supabase`.
  The `postgresai` agent skill asks which Supabase project from this list instead of reading `DATABASE_URL`.

- `pgai mon deploy` for Supabase from the CLI (https://gitlab.com/postgres-ai/platform-all/-/work_items/890): no
  password; the CLI opens the Supabase consent page (the link is printed too, `auth_url` in the JSON events) and
  waits for the approval, then the box prepares the database through the Supabase Management API. Agents and
  `--wait 0` get `action_required` (exit 3) with the link in `next`. A wait that ends before the box is ready is
  `action_required` (exit 3) with the resume command; with `--json` the event stream ends with a `done` event.
  `--monitoring-user` other than `postgres_ai_mon` and `--reset-password` are refused there (PostgresAI
  creates `postgres_ai_mon` through the Supabase Management API).

- `--provider generic` names the direct PostgreSQL path in `pgai mon deploy`, `pgai dblab deploy`,
  `prepare-db` and `unprepare-db` (postgresai!480). It is an alias of `self-managed`: both spellings
  work, and results still report `self-managed`. `--provider` now ignores case and surrounding spaces
  in every command that takes it, so `prepare-db --provider Generic` no longer warns "Unknown provider".
  The direct path needs a direct endpoint, not a pooler.

- `pgai mon deploy --monitoring-user <name>`: a monitoring role other than `postgres_ai_mon` (postgres-ai/internal#432),
  as `pgai prepare-db` already allows. The role is created or checked under that name and the box logs in as it;
  it carries its own password (PostgresAI keeps only `postgres_ai_mon`'s), so creating or resetting it needs
  `PGAI_MON_PASSWORD`. `--reset-password` sets a new one, under the same server lock and checks as for
  `postgres_ai_mon`. Refused: an empty name, one over 63 bytes (Postgres would truncate it), one with `'`, `$`,
  `%`, backslash or a control character, and a role that is a superuser or has CREATEDB, CREATEROLE, REPLICATION or BYPASSRLS,
  is a member of a role other than `pg_monitor`, `pg_read_all_stats`, `pg_read_all_settings` and `pg_stat_scan_tables` (directly or through other roles,
  also through `pg_monitor` itself), owns objects, has grants in another database of the server, or has
  grants in the database beyond CONNECT, USAGE on schemas other than pg_toast, SELECT outside pg_catalog
  and pg_toast, and the grants pgai makes, or holds any grant WITH GRANT OPTION or a membership WITH ADMIN OPTION. A custom role is used for one database per server.
  The name is trimmed. MCP `deploy_monitoring` takes `monitoring_user`.
- `pgai mon deploy` and `pgai dblab deploy` share one surface (postgresai#412): the URL as the argument or
  `--db-url`, `--name`, `--location`, `--wait <minutes>` / `--no-wait`, `--yes`, `--json`, one set of
  `status` words (`ready`, `in_progress`, `deleting`, `deleted`, `inactive`, `action_required`, `failed`)
  with exit codes 0 / 1 / 3, and `instances list | watch (status) | delete` by id or name. Progress goes
  to stderr; stdout is the result. The console-path `pgai mon deploy` of the DBLab deploy release is gone.
- `pgai mon instances set <id-or-name> --vcpus <n>`: add or change a monitoring instance's vCPUs after
  deploy (an org admin), so AAS is collected for an instance created without them. `mon deploy` on a
  database already monitored sets missing vCPUs from `--vcpus` (or asks at a terminal) and says when
  there is no AAS without them. `mon instances list` shows `vcpus`.
- `pgai mon deploy --vcpus <n>`: the database server's vCPUs, which AAS (database load) needs. At a terminal a
  missing value is asked for; without one, the run says there will be no AAS data. MCP `deploy_monitoring` takes `vcpus`.
- `pgai mon deploy <database-url>`: put a database under PostgresAI Cloud monitoring in one command.
  Signs in if needed, prepares the database (an admin URL creates `postgres_ai_mon`; otherwise the
  SQL is printed), provisions the monitoring box, waits, and prints the project's Health Matrix
  (`health_url`, the next step) and Grafana, signed in with the PostgresAI account (`dashboard_url`). ClickHouse,
  RDS and Supabase are detected from the host; `--clickhouse-key <id>:<secret>` adds CPU, memory and
  disk; RDS points to its console flow; `--self-hosted` runs `mon local-install`.
  JSON (`status`, `health_url`, `dashboard_url`, `next`) when stdout is not a TTY; exit 0 / 1 / 3 (action required),
  and 130 for a `pgai init` cancelled at a prompt.
  An existing `postgres_ai_mon` keeps its password: `PGAI_MON_PASSWORD` is checked by logging in.
  Another database on a server PostgresAI already monitors for the organization needs no
  `PGAI_MON_PASSWORD`: with `sslmode=require` or `verify-*` in the URL, PostgresAI fills in the
  password it keeps for that server.
  On a server without TLS, `mon deploy` says the kept password cannot be used there and names the way
  out, to an agent too (`PGAI_MON_PASSWORD`, TLS, or the instances to delete before
  `--reset-password`, quoted for the shell).
  A failed run can be re-run: the ClickHouse key is checked before the database is touched, and a role
  created with a generated password is dropped again when the launch is refused.
  A role that is not a superuser creates `postgres_ai_mon` only if it can run the whole preparation;
  `host` / `port` in the URL's query string is refused; certificate files in the URL are used from
  this machine only (the monitoring box does not get them).
  `mon local-install` reads `PGAI_DB_URL` like `--db-url`, and `PGAI_API_KEY` only together with it.
  `mon targets add [name]` reads `PGAI_DB_URL` when argv has no `postgres://` / `postgresql://` URL;
  with it set, a lone argument other than a plain name (ASCII letters, digits, `.`, `_`, `=`, `-`,
  with no `password=` / `pwd=`) is refused, not saved as the name. Under sudo, pass it on stdin
  (`targets add --help`): sudo logs a variable kept with `--preserve-env`, password included.
  Where sudoers enables I/O logging (`log_input`), which records stdin, read it from a file of
  mode 0600 inside the root shell. `Using PGAI_DB_URL` shows the user only for an accepted URL.
  The default name takes the host and database as WHATWG and pgx read them. Refused: no password
  in the URL, an IPv6 host, a raw `/`, `?` or `#` in the password (part of it would be read as the
  host, and end up in the name), an `@` in the database name or the query, a user info character
  that pgx refuses (a space, `"`, non-ASCII, a `%` without two hex digits), a user name with an
  encoded `:` (`monitor%3A<password>`, user and password encoded as one unit, which pgx reads as
  the user name), and a control character anywhere in the URL (such as a CR from a CRLF file).
  `mon deploy --self-hosted` percent-encodes an `@` in the query it passes on. `targets test` decodes
  a percent-encoded database name in full (`my%40db` is `my@db`), as pgwatch does. A host or
  database name with characters other than ASCII letters and digits may get a different default
  name than before (`база` gives `db-example--D0-B1-D0-B0-D0-B7-D0-B0`, not `db-example-----`): a
  re-run of `targets add <url>` without a name then adds a second target.
  Also `pgai init` (first run at a terminal), `pgai mon instances list`, `pgai mon instances watch <id-or-name>`, `pgai mon instances delete <id-or-name>`, and the MCP
  tool `deploy_monitoring`.
  `mon deploy` runs no checkup of its own: the box runs the first one about 30 minutes after it is ready
  (`first_checkup_eta`) and saves it as the database's first report (`pgai reports list`). Each step and
  each change of the box's state is shown once, with the time since the start (JSON: one event a line
  on stderr; any other text there, such as an error, a warning, `--debug` or a line of
  `mon local-install`, is `{"event":"log","level":"error"|"warn"|"info"|"debug","message":...}`, with
  the Grafana and VictoriaMetrics logins masked). A `--clickhouse-key` on a re-run is checked: a
  rejected key is exit 3, not `ready`. `pgai mon instances list` uses the `status` words of `mon deploy` and
  `mon instances watch`. `pgai init` does not echo the URL. `--reset-password` gives an existing `postgres_ai_mon`
  a new password.

- `prepare-db` / `unprepare-db --provider clickhouse` for ClickHouse Managed Postgres, auto-detected
  from `*.pg.clickhouse.cloud` hosts (positional URI, conninfo, `--db-url`, `--host`, `PGHOST`).
  Before any grant runs, a `-- scope:` line lists what the run grants the monitoring
  role, derived from the plan steps about to run (stderr under `--json`, omitted on `--reset-password`).
  `channel_binding=require` in a URI or conninfo string is honoured: SCRAM-SHA-256-PLUS is
  preferred, the plaintext retry is disabled, and `sslmode=disable` is rejected.

- ClickHouse Cloud host metrics: `mon local-install --db-url` and `mon targets add` write a
  VictoriaMetrics scrape job for the service's Prometheus endpoint when `CLICKHOUSE_ORG_ID`,
  `CLICKHOUSE_KEY_ID` and `CLICKHOUSE_KEY_SECRET` are set. The key secret is kept in
  `host-metrics/clickhouse-<name>.secret` (0600, directory 0700); a Basic Service API Reader key is enough.
  For a Supabase target, while `PGAI_SUPABASE_HOST_METRICS` is true, they write the relay's scrape
  job; for an RDS instance endpoint, the instance, region and labels `rds-host-stats` reads from `.env`.

### Changed

- `unprepare-db` no longer transfers or deletes objects owned by the monitoring role before
  dropping it (postgresai#432). If that role owns an unrelated object or has another dependency,
  the command keeps the role, exits 1, and reports the dependency for an administrator to resolve.
  A normal role drop still removes the known `public` and `pg_stat_statements` schema grants.
- `prepare-db`, `unprepare-db`, `checkup` and `pgai mon deploy` warn on stderr before an admin
  connection that is plaintext, or TLS without a verified server certificate (no sslmode, `prefer`,
  `allow`, `require`), and say when the session fell back to plaintext (postgresai#428). The warning
  follows the TLS settings pg uses, URI parameters such as `sslrootcert` included; `verify-full`,
  `verify-ca` and Unix sockets get none. In `pgai mon deploy` it is a step event, said once a run.

- `pgai mon deploy` (and MCP `deploy_monitoring`) no longer runs the express checkup while the box
  starts, as 0.18.0-dev builds did (postgresai#425): no `checkup` key in the result, no `checkup`
  event on stderr, and no report uploaded by the deploy. The box's first report (`first_checkup_eta`)
  brings the first findings. `pgai checkup` is unchanged.
- `mon deploy`: `first_checkup_eta` is 3 minutes after the box is ready, not 30: the earliest time
  of the first checkup. The reporter then waits until every target has data (at most 30 minutes),
  so on a slow first collection the report comes later than the ETA.
- `mon targets add` restarts `postgres-reports`, so a target added later gets its first checkup
  report after the reporter's initial delay and data wait, not the next day. `remove` does not.

- `sslmode` in a connection URL now wins over `PGSSLMODE`, as in libpq (`prepare-db`, `unprepare-db`,
  `checkup`, `mon deploy`). Before, an exported `PGSSLMODE=disable` turned `?sslmode=verify-full` into
  a plaintext connection. `PGSSLMODE` still applies to a URL without `sslmode`.

- `prepare-db --verify --json` reports `provider` only when it was given explicitly or auto-detected,
  as before; it is no longer filled with `self-managed`.

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

- `pgai mon instances delete` says "Delete the ClickHouse Cloud API key you gave us" only for a ClickHouse box. The
  platform now reports `host_metrics` for Supabase and RDS boxes too, with `host_metrics_source`
  (`clickhouse` | `supabase` | `rds`) right after it in `pgai mon instances list`
  (https://gitlab.com/postgres-ai/platform-all/-/issues/992).
- `pgai mon deploy` / `mon instances watch` / `list`: when a box is removed because Stripe refused the promo code at
  the first charge (for example, a code made for another organization), `next` now says to re-run without `--coupon`
  or with a code this organization can use, instead of "Update the payment method"
  (https://gitlab.com/postgres-ai/platform-all/-/work_items/971). The platform now rejects such a code at the quote,
  before anything is created.

- `prepare-db` removes `postgres_ai.explain_generic(text,text,text)`, a retired SECURITY DEFINER helper that
  older CLI versions installed (#426). Only its owner or a superuser can drop it; otherwise `prepare-db` names it
  in text and JSON, and `--verify` exits 1 while it or a SECURITY DEFINER `postgres_ai.table_describe(text)`
  (0.14.0 to 0.17.0) is installed. `--verify` lists a missing `table_describe` instead of failing with 42883.
- `pgai mon deploy --self-hosted` (or `--provider self-managed`) through the Supabase pooler: the box's URL keeps
  the project suffix of the URL's user (`postgres.<ref>` gives `postgres_ai_mon.<ref>`), so the pooler can route it
  (postgres-ai/internal#432). A cloud deploy of a Supabase host is finished in the Console, not by this code.
- The CLI and MCP server work on a slow link. When the TCP connect to postgres.ai took longer than
  Node's per-address limit (250 ms on Node 20–24, 500 ms from Node 25) — a VPN, mobile, satellite, a
  congested Wi-Fi — every request failed with a bare `fetch failed` while curl worked. The CLI now
  allows 2 s per address on Node, unless the user set
  `--network-family-autoselection-attempt-timeout` themselves. (#421)

- `pgai joe` works against an *inverted* Joe instance — one whose platform agent
  polls the job channel and has no URL for the platform to dial. Every verb used
  to fail with `HTTP 426 Upgrade Required`, because the run rpc will only enqueue
  a command for a client that has declared it understands the async handle. The
  CLI now declares it, takes the handle, and polls the same `command_id` it
  always polled; a platform that predates the handle is retried once without the
  flag, and a command is never re-sent. The wait on that route is sized from the
  handle's own `expires_in_s` (clamped to the platform's [10 min, 1 h] window)
  rather than the 25s one-shot, because an enqueued command waits for its box to
  poll — `--budget` still overrides it, the polls are paced by a backoff ladder,
  and a command that outlives the wait still prints the
  `pgai joe result <id>` resume line. (#402, postgres-ai/platform-all#840)

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
