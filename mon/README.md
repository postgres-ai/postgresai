# pgai-mon

PostgresAI monitoring collector, Parquet store and SQL endpoint in one Rust binary with
embedded DuckDB.

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
Cloud; `scrape.jobs` takes the same configs inline. Files are re-read every
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

`sql/host_rules.sql` holds the 37 recording rules that mapped Supabase (node_exporter through the
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

## Layout

| Path | What |
|---|---|
| `src/` | the crate (`pgai-mon` binary, `pgai_mon` library) |
| `tests/` | integration tests (`tests/fixtures/`: sample jsonlog lines) |
| `schema/views.sql` | the consumer views; embedded in the binary (`include_str!`) |
| `sql/macros.sql` | SQL macros loaded after the views (`macros_sql` / `--macros-sql`) |
| `sql/reports/` | ported dashboard queries |
| `grafana/` | dashboard compiler and the 14 compiled SQL dashboards (see `grafana/README.md`) |
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

YAML; the reference is `src/config.rs`. Only `store_dir` and `sources` are required; every
other field has a default. Unknown fields are rejected. See
`examples/pgai-mon.example.yml`:

- `sources[]`: `name`, `conn_str`, `custom_tags`, `discover_databases`, `scrape_labels`;
- `metrics_yml`, `preset` (default `full`);
- `store`: flush interval, writer/compaction memory limits, compaction settings;
- `http`: `listen` (default `127.0.0.1:8765`), reader memory and RSS caps, timeouts, row and
  byte limits, caches, query slots, `s3` read path, `auth` (`token_file`, `username`,
  `password_file`, `disabled`);
- `collector`: series and session limits, statement timeout, snapshot intervals;
- `logs[]`: jsonlog directories to tail;
- `views_sql` (default: the embedded copy), `macros_sql`, `read_only`;
- `scrape`: `files`, `jobs`, `reload_interval_s`, `max_body_bytes`, `default_sample_limit`;
- `rules`: `host` (default: on when `scrape` is configured), `interval_s`, `delay_s`;
- `ingest`: push limits (`max_body_bytes`, `max_samples_per_request`, `max_series`, `max_age_s`,
  `max_future_s`, `max_labels`, `max_label_value_bytes`).

Retention: `store.retention_days` (metrics: samples, series, rollups, logs, plan events) and
`store.texts_retention_days` (query texts); 0, the default, keeps everything, and the compose
image's config sets 14 and 30. The environment variables `PGAI_MON_RETENTION_DAYS` and
`PGAI_MON_TEXTS_RETENTION_DAYS` replace them; without those, the old stack's
`VM_RETENTION_PERIOD` and `QUERYID_RETENTION_HOURS` apply, rounded up to days. Each compaction
pass drops the files whose last day is older from the catalog and deletes them after
`store.delete_grace_s`, in the bucket after its next commit. A month file (1 d and 1 h rollups,
monthly series) goes once its whole month is older. The series registry, object definitions and
plan definitions are not per day and are kept. A query's text is written again on each UTC day
it is seen, so every day's file holds the texts of that day's queries.

A store root may also be `s3://bucket/prefix` (credentials and region from the AWS
environment). The environment variable `PGAI_MON_STORE`, when set, replaces `store_dir` (compose
and Terraform select the store with it). `serve` reads an object store. `run` writes it with
`store.s3.local_dir` (default `/var/lib/pgai-mon`) as its local working directory (WAL, files not
uploaded yet, cache) and commits to the bucket every `store.s3.upload_interval_s` (default 300 s)
on its own thread: a lost machine loses at most that interval plus one upload round. One writer
per prefix: a writer that has not committed for `store.s3.lease_intervals` (default 3) upload
intervals is taken over at start; a fresher one refuses the start (`store.s3.takeover: true`
forces it). A clean stop releases the lease with its last commit, so a new machine on the same
prefix starts at once. At start the store must answer within 30 s, or the process exits with the
reason (credentials, region, network). A writer whose commit finds another owner exits with code 4. `file:///dir` works as a
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
cargo test -- --ignored --nocapture   # manual measurements (not gates)
python3 grafana/test_compile.py       # offline dashboard checks
```

The tests need no PostgreSQL and no network: they build their stores in temp dirs (and
in-memory object stores). `kill9.rs` and `rowcache.rs` run the built binary.
