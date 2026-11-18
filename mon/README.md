# pgai-mon

PostgresAI monitoring collector, Parquet store and SQL endpoint in one Rust binary with
embedded DuckDB.

[ARCHITECTURE.md](ARCHITECTURE.md) describes the processes, the module map, the write and
read paths, the files on disk, the invariants and the security model. Read it first.

- Collects every metric in `mon/metrics.yml` with pgwatch v3 semantics
  (same metric names and labels as the pgwatch + VictoriaMetrics stack).
- Tails PostgreSQL jsonlog / auto_explain output, and snapshots query texts and object
  definitions.
- Writes a WAL, flushes it to Parquet, compacts closed days and builds 1 h / 1 d rollups.
- Serves read-only SQL over HTTP. Consumers query only the views in `schema/views.sql`; the
  PromQL-shaped table functions (`prom_rate`, `prom_topk`, ...) are in `src/eval/` and
  `sql/macros.sql`.

Every HTTP route requires credentials except the liveness probe `GET /livez`: a bearer
token, HTTP basic auth, or either. They come from the environment or files
(`PGAI_MON_TOKEN` / `PGAI_MON_TOKEN_FILE`, `PGAI_MON_AUTH_USERNAME` +
`PGAI_MON_AUTH_PASSWORD` / `PGAI_MON_AUTH_PASSWORD_FILE`, or `http.auth` in the config); secrets
are at least 16 characters. A second, read-only token (`PGAI_MON_READ_TOKEN` /
`PGAI_MON_READ_TOKEN_FILE`) may make any GET and `POST /sql`, and gets 403 on writes (push
ingest): it is the one to give Grafana and other readers. Without credentials the server refuses to start, unless
authentication is turned off explicitly for local development (`PGAI_MON_AUTH_DISABLED=1`,
`http.auth.disabled: true`, or `serve --no-auth`).

## Ingest

`POST /api/v1/import/prometheus` takes Prometheus text exposition with millisecond timestamps,
as VictoriaMetrics' endpoint of the same path does, so a pusher (rds-host-stats) only changes its
URL and credentials. `?extra_label=name=value` adds a label to every sample. The body is parsed
and validated whole first: one bad line answers 400 and nothing is written. Samples older than
`ingest.max_age_s` (1 day) or more than `ingest.max_future_s` (10 min) ahead, and new series past
`ingest.max_series` (200 000 tracked; a series idle for an hour is forgotten when full) are
dropped and counted (`GET /ingest/stats`). 204 once the samples are in the WAL. Pushed series get
no staleness markers (as with VictoriaMetrics' import); in-process producers (scraper, host rules)
write markers per scrape target or rule group, as the collector does.

```bash
curl -s -H "$H" --data-binary @samples.prom localhost:8765/api/v1/import/prometheus
```

## Scrape

`scrape.files` lists Prometheus scrape-config files (`*` allowed in the file name), e.g. the
`host-metrics/*.yml` the CLI writes for Supabase (through the instance-jobs relay) and ClickHouse
Cloud; `scrape.jobs` takes the same configs inline (the image's `docker/pgai-mon.yml` scrapes the
monitoring host's own node-exporter and cAdvisor that way, for the self-monitoring dashboard). Files are re-read every
`scrape.reload_interval_s` (10 s, no SIGHUP), secret files at every scrape. Supported: static
targets, `scheme`, `metrics_path`, `params`, `scrape_interval` / `scrape_timeout`, `sample_limit`,
`honor_labels`, `honor_timestamps`, `basic_auth` (password / password_file), `bearer_token(_file)`,
`authorization` (credentials / credentials_file), and `relabel_configs` /
`metric_relabel_configs` with keep, drop, replace, labeldrop, labelkeep. A file with anything
else is skipped whole and reported. Series get `job`, `instance` and the static labels
(`__`-prefixed ones such as the CLI's `__pgai_rev` stay off the series), plus `up`,
`scrape_duration_seconds`, `scrape_samples_scraped`, `scrape_samples_post_metric_relabeling`, as
Prometheus writes them; series that disappear, every series of a failed scrape and of a removed
target get staleness markers. `GET /scrape/targets` lists the targets with all their labels and
their last scrape.

```yaml
scrape:
  files: [/etc/pgai/host-metrics/*.yml]
```

## Host rules

`sql/host_rules.sql` holds the 38 recording rules that mapped Supabase (node_exporter through the
instance-jobs relay) and ClickHouse Cloud (`PostgresServer_*`) series to the `host_*` vocabulary
of `docs/host-metrics.md` (formerly vmalert's `old/config/prometheus/host_rules.yml`). pgai-mon
evaluates it every `rules.interval_s` (60 s) at the minute, `rules.delay_s` (5 s) later, after a
store flush, and writes the rows back as `host_*{cluster, node_name}` series (`provider` on
`host_provider_info`); a `host_*` series a later evaluation no longer produces gets a staleness
marker. It runs when `scrape` is configured (`rules.host: true|false` forces it). The results
are materialised rather than computed by views at query time: every consumer keeps reading
`host_*` like other series (and RDS writes the same names directly), rollups apply, and long
ranges read one series per node instead of re-deriving rates of every per-CPU and per-device
counter. `tests/host_rules.rs` runs the vmalert golden cases against it.

## Logs

pgai-mon can read PostgreSQL logs written in the `jsonlog` format. What exists now:

| Feature | State |
|---|---|
| Tail a directory of jsonlog files (`logs[]` in the config) | Supported. pgai-mon must be able to read the files on its own filesystem. Positions are kept in the catalog, so a restart continues where it stopped. |
| Offline import: `pgai-mon load-logs --store DIR --input FILE_OR_DIR` | Supported. Resumable. Run it while no `run` or `serve` has the store open. |
| auto_explain plans found in log lines | Parsed into the `plans` and `plan_events` views. Literals in plans are redacted. |
| Log lines | In the `logs` view. Parameter values are dropped. |
| Logs of managed services (RDS, Supabase and others) | Not supported: nothing fetches them. |
| The compose stack | Does not tail logs: `docker/pgai-mon.yml` has no `logs` entry and compose mounts no log directory. |

## PromQL to SQL

Conventions of the `prom_*` functions:

- Times (`start`, `end`, `step`, `window`) are Unix milliseconds. `epoch_ms(now())` gives the
  current time. Below, `S` and `E` stand for the start and end of the range.
- Matchers are a list of `[label, operator, value]` triples; operators are `=`, `!=`, `=~`,
  `!~`. `[]` selects every series of the metric. Regular expressions are anchored, as in
  PromQL.
- The result columns are `series_id`, `labels` (a map: `labels['datname']`), `t` (the step
  time, ms) and `value`. `prom_agg` returns `labels`, `t`, `value`.
- An instant query at time `T` is a range query with `start = end = T` and `step = 300000`.

Bare selectors for metrics collected more than 300 seconds apart fetch at least the dictionary interval
(`metric_dict.interval_s`, the maximum across sources) plus 12.5%: 2 h 15 min for a two-hour job.
The last value stays current for at least that interval plus margin, for instant and range queries,
even when restarts or retries shorten the observed sample spacing. A longer estimated lookback
still applies. Staleness markers still end a series. Explicit windows and raw SQL are
unchanged. Native target removal does not yet write markers, so a removed target's slow-job
series can remain visible for up to this lookback.

| PromQL | SQL |
|---|---|
| `m{cluster="c"}` at time T | `SELECT labels, value FROM prom_last('m', [['cluster','=','c']], T, T, 300000)` |
| `m` as a range query, step 1m | `SELECT labels, t, value FROM prom_last('m', [], S, E, 60000)` |
| `m{datname=~"a\|b", mode!="idle"}` | matchers `[['datname','=~','a\|b'], ['mode','!=','idle']]` |
| `rate(m[5m])`, step 1m | `SELECT labels, t, value FROM prom_rate('m', [], S, E, 60000, 300000)` |
| `irate(m[5m])`, `increase(m[1h])`, `delta(m[1h])` | `prom_irate`, `prom_increase`, `prom_delta`, same arguments as `prom_rate` |
| `max_over_time(m[1h])` | `SELECT labels, t, value FROM prom_max_over_time('m', [], S, E, 60000, 3600000)` |
| `avg_`, `min_`, `sum_`, `count_`, `stddev_`, `last_`, `first_over_time` | `prom_avg_over_time`, ... same arguments |
| `quantile_over_time(0.95, m[1h])` | `SELECT labels, t, value FROM prom_quantile_over_time(0.95, 'm', [], S, E, 60000, 3600000)` |
| `sum by (datname) (rate(m[5m]))` | `SELECT labels['datname'] AS datname, t, value FROM prom_agg('sum', 'rate', 'm', [], S, E, 60000, 300000, p_by := ['datname'])` |
| `avg`, `min`, `max`, `count by (...)` | `prom_agg` with `'avg'`, `'min'`, `'max'`, `'count'` |
| `topk(10, rate(m[5m]))` | `SELECT labels, t, value FROM prom_topk(10, 'rate', 'm', [], S, E, 60000, 300000)` |
| `bottomk(10, ...)` | `prom_topk(..., p_bottom := true)` |
| last value of every series now | `SELECT labels, value FROM prom_last('m', [], epoch_ms(now()), epoch_ms(now()), 300000)` |
| raw samples (no PromQL equivalent) | `SELECT ts, labels, value FROM pgai_samples('m', [], S, E) ORDER BY ts` |
| label values: `label_values(m, datname)` | `SELECT DISTINCT labels['datname'] FROM series WHERE metric = 'm'` |
| metric names with type and interval | `SELECT metric, type, interval_s FROM metric_dict` |
| `histogram_quantile(...)` | Not available. |

A window of `0` means no `[d]` was given. Expressions over several series (a ratio of two
rates, for example) are written as an SQL join of two calls on `labels` and `t`.

## Layout

| Path | What |
|---|---|
| `src/` | the crate (`pgai-mon` binary, `pgai_mon` library) |
| `tests/` | integration tests; `tests/fixtures/` holds sample jsonlog lines, the host rule golden cases, RDS host stats recordings and scrape files |
| `schema/views.sql` | the consumer views; embedded in the binary (`include_str!`) |
| `sql/macros.sql` | SQL macros loaded after the views (`macros_sql` / `--macros-sql`) |
| `sql/reports/` | ported dashboard queries |
| `grafana/` | dashboard compiler and the 16 compiled SQL dashboards (see `grafana/README.md`) |
| `examples/pgai-mon.example.yml` | example config (the demo target) |
| `Dockerfile` | container image |

## Build

Rust 1.94 (`rust-version` in `Cargo.toml`); dependencies are pinned and `Cargo.lock` is
checked in. DuckDB is compiled from source (feature `bundled`), so a cold build needs a C++
compiler and takes a few minutes.

```bash
cd mon
cargo build --release          # target/release/pgai-mon
```

Image (build context is the repository root, because the image ships `metrics.yml`):

```bash
docker build -f mon/Dockerfile -t pgai-mon .
docker run -v pgai-mon-data:/var/lib/pgai-mon -v $PWD/my.yml:/etc/pgai-mon/pgai-mon.yml -p 8765:8765 pgai-mon
```

## Run

Run from the repository root (the default `metrics_yml` is `mon/metrics.yml`,
relative to the working directory).

```bash
B=mon/target/release/pgai-mon

# collect + store + compact + serve
$B run --config my.yml

# SQL over an existing store, no collection
$B serve --store DIR --listen 127.0.0.1:8765 \
    --metrics-yml mon/metrics.yml --macros-sql mon/sql/macros.sql
export PGAI_MON_TOKEN=$(openssl rand -hex 24)   # or --no-auth for local development
H="Authorization: Bearer $PGAI_MON_TOKEN"
curl -s -H "$H" localhost:8765/sql -d "select metric, count(*) from series group by 1 order by 2 desc limit 5"
curl -s -H "$H" localhost:8765/sql -H 'content-type: application/json' -d '{"sql":"select 1"}'
curl -s -H "$H" 'localhost:8765/sql?format=objects' -d "select 1 as x"   # the format Grafana's Infinity reads
curl -s -H "$H" localhost:8765/schema    # views, columns, metric names with type/interval/description
curl -s localhost:8765/livez   # liveness, no credentials
curl -s -H "$H" localhost:8765/health    # ingest counters, files, bytes, reader state

# one read-only query, same guardrails as POST /sql, no server
$B query --store DIR "select ..."

# import a VictoriaMetrics export (/api/v1/export JSON lines; "-" = stdin)
$B load --store DIR --input export.jsonl
# import jsonlog files offline (resumable)
$B load-logs --store DIR --input /path/to/pglog
# compact now (stop run/serve first)
$B compact --store DIR --all
```

`$B --help` and `$B <command> --help` list every command and option, including the
crash-test helpers (`crashtest-write`, `crashtest-verify`).

## Config

YAML, loaded by `run --config`. `store_dir` is the only required key (the environment
variable `PGAI_MON_STORE` may replace its value); without `sources` or `sources_file` nothing
is collected. Unknown keys are rejected, except inside a source, where they are reported and
ignored. `examples/pgai-mon.example.yml` is a working example; `docker/pgai-mon.yml` is the
config of the compose image.

Top level:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `store_dir` | path or `s3://bucket/prefix` | required | Root of the store. |
| `metrics_yml` | path | `mon/metrics.yml` | Metric definitions (pgwatch format). Relative to the working directory. |
| `preset` | string | `full` | Preset in `metrics.yml` that gives the metric list and intervals. |
| `sources` | list | empty | Monitored nodes, inline. See the next table. |
| `sources_file` | path | none | A pgwatch `sources.yml` or the CLI's `instances.yml`. Re-read on SIGHUP and every `collector.reload_interval_s`. A missing file counts as empty. |
| `logs` | list | empty | jsonlog directories to tail: `dir`, `suffix` (`.json`), `start_at_end` (`false`). |
| `views_sql` | path | embedded copy | The consumer views (`schema/views.sql`). |
| `macros_sql` | path | none | Extra SQL run after the views (`sql/macros.sql`). |
| `read_only` | bool | `false` | Open the store without WAL, compaction or ingest. Not allowed with sources or logs. |
| `store`, `http`, `collector`, `ingest`, `scrape`, `rules` | maps | | See below. |
| `rds` | list | empty | RDS or Aurora instances polled for host metrics: `instance_id`, `region`, `cluster`, `node_name`, `role_arn`, `external_id`, `poll_interval_s` (1 to 300, default 60). |

`sources[]`:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | required | Source name; becomes the `dbname` label. |
| `conn_str` | string | required | libpq URL or `key=value` string. |
| `custom_tags` | map | empty | Labels added to every series. |
| `discover_databases` | bool | `false` | Also collect per-database metrics on every other connectable database, named `<name>_<datname>`. |
| `kind` | string | `postgres` | `postgres` or `postgres-continuous-discovery` (which turns discovery on). Other kinds are skipped. |
| `include_pattern`, `exclude_pattern` | regex | none | Filters on discovered database names. |
| `preset_metrics`, `custom_metrics` | string, map | none | Per-source preset, or metric to interval in seconds. Neither: the global `preset`. |
| `preset_metrics_standby`, `custom_metrics_standby` | string, map | none | What a standby runs instead. |
| `is_enabled` | bool | `true` | `false`: not collected. |
| `only_if_master` | bool | `false` | Collect only while the node is a primary. |
| `scrape_labels` | map | `{job: pgwatch-prometheus, instance: <name>}` | Extra labels (`job`, `instance`) that the old stack added at scrape time. With a `job`, the source also writes `up`, which the Active session history panels divide by. A non-empty map replaces the default. |
| `group` | string | none | Informational. |

`store`:

| Key | Default | Meaning |
|---|---|---|
| `flush_interval_s` | 60 | Seconds between flushes of the WAL to Parquet. New data is visible to queries after a flush. |
| `max_buffer_bytes` | 32 MiB | Flush early once this much is buffered. |
| `write_chunk_bytes` | 8 MiB | Bytes per write batch inside a flush. |
| `writer_memory_limit` | `64MB` | DuckDB memory limit of the flush. |
| `writer_threads` | 1 | DuckDB threads of the flush. |
| `compact_memory_limit` | `32MB` | DuckDB memory limit of sample compaction. |
| `compact_events_memory_limit` | `96MB` | The same for logs, plans and other non-sample kinds. |
| `compact_grace_s` | 300 | Seconds after a day ends before it is compacted as closed. |
| `delete_grace_s` | 300 | Seconds a replaced or expired file stays on disk. |
| `closed_lag_ms` | 5000 | A sample older than the newest sample minus this counts as late. |
| `zstd_level` | 9 | Compression level of compacted files. |
| `row_group_size` | 122,880 | Rows per Parquet row group. |
| `retention_days` | 0 | Days of samples, series, rollups, logs and plan events kept. 0 keeps everything. |
| `texts_retention_days` | 0 | Days of query texts kept. 0 keeps everything. |
| `s3.local_dir` | `/var/lib/pgai-mon` | Local working directory of an S3 store. |
| `s3.upload_interval_s` | 300 | Seconds between commits to the bucket. |
| `s3.lease_intervals` | 3 | A writer silent for this many upload intervals may be taken over. |
| `s3.takeover` | `false` | Start even when another writer's lease is fresh. |
| `s3.keep_local_days` | 2 | Committed files of this many days stay on local disk. |
| `s3.sweep_interval_s` | 86,400 | Seconds between sweeps of objects no catalog lists. |
| `s3.delete_grace_s` | 600 | Seconds a dropped file stays in the bucket. |
| `s3.part_mb` | 8 | Multipart upload part size, MB. |

`http`:

| Key | Default | Meaning |
|---|---|---|
| `listen` | `127.0.0.1:8765` | Address and port. |
| `timeout_ms` | 15,000 | Per-query timeout. |
| `max_rows` | 100,000 | Rows per result; more sets `truncated`. With `format=objects` (what Grafana panels read) it is a `result_too_large` error, so a chart never stops halfway unnoticed. |
| `max_bytes` | 16 MiB | Bytes per result; same behaviour as `max_rows`. |
| `max_concurrent` | 0 | Query slots. 0: one per core, at most 4. |
| `queue_max` | 64 | Queries that may wait for a slot. |
| `queue_wait_ms` | 30,000 | Longest wait for a slot; then `busy`. |
| `short_query_ms` | 50 | A query expected to finish under this is short and may use the reserved slot. |
| `heavy_mem_gate_pct` | 50 | A heavy query starts beside another only while the reader's RSS is under this share of the cap. 0: off. |
| `memory_limit` | `96MB` | DuckDB memory limit of the reader. |
| `threads` | 1 | DuckDB threads of the reader. 0: `max_concurrent`. |
| `max_temp_directory_size` | `2GB` | Reader spill limit. |
| `reader_rss_cap_mb` | 192 | The child is killed above this RSS. |
| `reader_data_limit_mb` | 1024 | Linux `RLIMIT_DATA` of the child. 0: none. |
| `reader_drain_pct` | 70 | Replace the child once its RSS passed this share of the cap during a query. |
| `footer_cache_mb` | 16 | Parquet footer cache in the reader. 0: off. |
| `row_cache_mb` | 24 | Cache of evaluated rows in the reader. 0: off. |
| `cache_mb` | 16 | Result cache in the parent, compressed. 0: off. |
| `cache_max_entry_kb` | 2048 | Largest compressed result the cache stores. |
| `eval_workers` | 1 | Evaluator threads shared by running queries. 1: off. 0: `max_concurrent`. |
| `eval_worker_mb_per_day` | 3 | Memory one evaluator worker may add per day scanned. |
| `auth.token_file` | none | File with the full bearer token. |
| `auth.read_token_file` | none | File with the read-only bearer token. |
| `auth.username`, `auth.password_file` | none | Basic auth. |
| `auth.disabled` | `false` | Serve without authentication. Local development only. |
| `s3.concurrency` | 16 | Range GETs in flight. |
| `s3.inflight_mb` | 8 | Prefetch window. |
| `s3.coalesce_kb` | 256 | Merge two planned ranges into one GET across a gap up to this. |
| `s3.cache_dir`, `s3.cache_gb` | none, 2.0 | Local cache of fetched chunks and its size. |
| `s3.catalog_ttl_ms` | 30,000 | How often a reader re-reads `catalog.json` from the bucket. |
| `s3.footer_mb` | 8 | Raw footers kept in memory. |
| `s3.gets_warn` | 64 | A query with more GETs gets a warning in its reply. |
| `s3.window_soft_pct` | 85 | The prefetch window shrinks so RSS stays under this share of the cap. 0: off. |
| `s3.mirror_mb` | 2048 | Budget of the local copies of the small kinds that DuckDB reads. A kind over the budget fails when queried, with an error naming this setting; `s3_mirror` in `GET /health` lists such kinds. |

`collector`:

| Key | Default | Meaning |
|---|---|---|
| `max_sessions_per_node` | 4 | Sessions to one Postgres node across all its databases. With more monitored databases than this on a node, sessions are closed and reopened to take turns; raise it there. |
| `default_statement_timeout_s` | 15 | Statement timeout of a metric that sets none. |
| `lock_timeout_ms` | 100 | `lock_timeout` of every session. |
| `search_path` | none | Default: `pg_catalog`, the schemas of installed extensions, `public`. |
| `max_rows_per_query` | 100,000 | Rows one metric query may return; the rest is not fetched and is counted. |
| `max_series_per_query` | 20,000 | Series one metric query may produce per run; the rest is cut and counted. |
| `drop_numeric` | `true` | Drop NUMERIC columns, as pgwatch's Prometheus sink did. |
| `text_columns` | empty | Per metric and column: `label` or `drop`, for text columns without the `tag_` prefix. |
| `query_texts_interval_s` | 300 | Seconds between query text snapshots. |
| `objects_interval_s` | 600 | Seconds between object definition snapshots. |
| `objects_statement_timeout_s`, `objects_max_rows` | 60, 500,000 | Limits of the object snapshot. |
| `stats_reset_interval_s` | 60 | Seconds between `stats_reset` snapshots. |
| `reconnect_min_s`, `reconnect_max_s` | 2, 120 | Reconnect backoff range. |
| `timeout_retry_min_s`, `timeout_retry_max_s` | 30, 600 | Lock-timeout retry backoff base and cap, in seconds. Each delay has 50–100% jitter. |
| `reload_interval_s` | 30 | Seconds between re-reads of the config, sources file and `metrics.yml`. 0: SIGHUP only. |
| `discovery_interval_s` | 120 | Seconds between database discovery runs. |
| `role_check_interval_s` | 30 | Seconds between primary or standby checks. |
| `self_metrics_interval_s` | 15 | Seconds between writes of the collector's own series. |

Jobs with intervals over 60 seconds retry lock timeouts (`55P03`) with exponential backoff.
Statement timeouts or cancels (`57014`) get at most two retries per slot, after 5 minutes
and then 15 minutes more. A retry must fall before the next regular slot. Success clears
retries; the next regular slot replaces any pending retry. Intervals of 60 seconds or less
keep their normal cadence. Retries use the same session limits and heavy-job permit.
The job's `/health` status includes `retries`, also emitted as `pgai_mon_retries_total` with
the same labels as `pgai_mon_timeouts_total`.

`ingest`, `scrape`, `rules`:

| Key | Default | Meaning |
|---|---|---|
| `ingest.max_body_bytes` | 16 MiB | Largest push request. |
| `ingest.max_samples_per_request` | 500,000 | Samples per push request. |
| `ingest.max_series` | 200,000 | Ingested series tracked at once; new ones beyond it are dropped and counted. |
| `ingest.max_age_s`, `ingest.max_future_s` | 86,400, 600 | Samples older or further ahead are dropped. |
| `ingest.max_labels`, `ingest.max_label_value_bytes` | 64, 4096 | Per-series limits. |
| `scrape.files` | empty | Scrape-config files; `*` allowed in the file name. |
| `scrape.jobs` | empty | Scrape configs inline. |
| `scrape.reload_interval_s` | 10 | Seconds between re-reads of `files`. |
| `scrape.max_body_bytes` | 32 MiB | Largest scrape response; larger is a failed scrape. |
| `scrape.default_sample_limit` | 50,000 | `sample_limit` of jobs that set none. 0: none. |
| `rules.host` | unset | Evaluate the host rules. Unset: only when scraping is configured. |
| `rules.interval_s`, `rules.delay_s` | 60, 5 | Evaluation period, and the delay after each minute. |

### Environment variables

| Variable | Effect |
|---|---|
| `PGAI_MON_STORE` | Replaces `store_dir` when set and not empty. |
| `PGAI_MON_RETENTION_DAYS`, `PGAI_MON_TEXTS_RETENTION_DAYS` | Replace `store.retention_days` and `store.texts_retention_days`. A value that is not a whole number stops the start. Empty counts as unset. |
| `VM_RETENTION_PERIOD`, `QUERYID_RETENTION_HOURS` | The old stack's retention settings. Used only when the variable above is unset; rounded up to days. A value that is not understood is ignored with a message. |
| `PGAI_MON_TOKEN`, `PGAI_MON_TOKEN_FILE` | The full bearer token, or a file that holds it. |
| `PGAI_MON_READ_TOKEN`, `PGAI_MON_READ_TOKEN_FILE` | The read-only bearer token. |
| `PGAI_MON_AUTH_USERNAME`, `PGAI_MON_AUTH_PASSWORD`, `PGAI_MON_AUTH_PASSWORD_FILE` | Basic auth. |
| `PGAI_MON_AUTH_DISABLED` | `1`, `true`, `yes` or `on`: serve without authentication. |
| `AWS_*` (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`, ...) | Credentials and region for an S3 store and for RDS host stats. Without keys, the instance or container role is used. |
| `RDS_DB_INSTANCE_IDENTIFIER`, `RDS_POLL_INTERVAL_SECONDS`, `RDS_ROLE_ARN`, `RDS_EXTERNAL_ID`, `PGAI_CLUSTER`, `PGAI_NODE_NAME` | One RDS host stats target, used when the `rds` list is empty. |
| `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`, `PGPASSFILE`, `PGOPTIONS`, `PGSSLMODE`, `PGSSLROOTCERT`, `PGSSLCERT`, `PGSSLKEY`, `PGHOSTADDR`, `PGTARGETSESSIONATTRS` | libpq defaults for values a `conn_str` leaves out. |
| `PGAI_NO_COMPACT` | Set: the compactor thread does not run. |
| `PGAI_QCACHE=0` | Turns the result cache off. |
| `PGAI_TEST_PG_URL` | Tests only: a throwaway Postgres for `tests/collector_pg.rs`. |

Variables named `PGAI_EVAL_*`, `PGAI_QUERY_TIMING`, `PGAI_NO_ROLLUP*`, `PGAI_ROLLUP_*` and
`PGAI_TOPK_*` are diagnostics for developers and tests; see their use in `src/eval/`.
`PGAI_QUERY_TIMING=1` logs read-only checking, preparation, execution, Arrow fetching, JSON
encoding and sample-file opening times. Scan times are part of execution/fetch, not additional
elapsed time. Measure HTTP time separately: cached bodies retain the original `elapsed_ms`.

### Retention

`store.retention_days` covers samples, series, rollups, logs and plan events;
`store.texts_retention_days` covers query texts. 0, the default, keeps everything. The compose
image's config sets 14 and 30. The environment variables `PGAI_MON_RETENTION_DAYS` and
`PGAI_MON_TEXTS_RETENTION_DAYS` replace them; without those, the old stack's
`VM_RETENTION_PERIOD` and `QUERYID_RETENTION_HOURS` apply, rounded up to days. Each compaction
pass drops the files whose last day is older from the catalog and deletes them after
`store.delete_grace_s`, in the bucket after its next commit. A month file (1 d and 1 h rollups,
monthly series) goes once its whole month is older. The series registry, object definitions and
plan definitions are not per day and are kept. A query's text is written again on each UTC day
it is seen, so every day's file holds the texts of that day's queries.

### Object store

A store root may also be `s3://bucket/prefix` (credentials and region from the AWS
environment). The environment variable `PGAI_MON_STORE`, when set, replaces `store_dir` (compose
and Terraform select the store with it). `serve` reads an object store. `run` writes it with
`store.s3.local_dir` (default `/var/lib/pgai-mon`) as its local working directory (WAL, files not
uploaded yet, cache) and commits to the bucket every `store.s3.upload_interval_s` (default 300 s)
on its own thread: a lost machine loses at most that interval plus one upload round. One writer
per prefix: a writer that has not committed for `store.s3.lease_intervals` (default 3) upload
intervals is taken over at start; a fresher one refuses the start (`store.s3.takeover: true`
forces it). A clean stop releases the lease with its last commit, so a new machine on the same
prefix starts at once. A copy or an older snapshot of a writer's directory counts as another
writer while that writer's lease is fresh. At start the store must answer within 30 s, or the process exits with the
reason (credentials, region, network). A writer whose commit finds another owner exits with
code 4; started again on the same directory once the store is free, it adopts the owner's
catalog and adds what it had flushed and not uploaded. `file:///dir` works as a
single-host bucket (tests, dev). See `src/bucket.rs`; `/health` shows the write path under
`s3_write`.

Operator facts for an object store:

- Worst-case loss when the machine dies (S3 healthy): the data acknowledged in the last
  `upload_interval_s` plus one upload round, about 300 s + 1-3 s with the defaults. A process
  crash with the local disk intact loses nothing. During an S3 outage data stays on the local
  disk and the bound grows with the outage.
- Local disk: plan about 3 GB. The store files take about 200-250 MB for a store of ~76 M
  samples/day with `store.s3.keep_local_days: 2` (about 100 MB with 0, which works: jobs fetch
  what they need); most of the rest is the reader's local copy of the small kinds (logs history),
  capped by `http.s3.mirror_mb` (default 2048). Lower that cap for a smaller disk.

## Tests

```bash
cd mon
cargo test             # unit + integration tests
python3 grafana/test_compile.py       # offline dashboard checks
```

The tests need no network, and no PostgreSQL except `tests/collector_pg.rs`, which runs only
when `PGAI_TEST_PG_URL` names a throwaway server and prints a skip message otherwise. The
others build their stores in temp dirs (and in-memory object stores). `kill9.rs` and `rowcache.rs` run the built binary.
