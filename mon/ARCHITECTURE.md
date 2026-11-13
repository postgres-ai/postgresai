# pgai-mon architecture

pgai-mon is one Rust binary. It collects metrics from PostgreSQL, stores them as Parquet
files in a directory or an S3 bucket, and answers read-only SQL over HTTP through an embedded
DuckDB. This document describes the processes, the modules, the write path, the read path,
the files on disk, the invariants and the security model. Module names are files under
`src/`; each file starts with a `//!` header that says more.

Terms used below:

| Term | Meaning |
|---|---|
| sample | one value of one series at one time |
| series | a metric name plus a set of labels; identified by `series_id` |
| store | the directory (or bucket prefix) that holds every file pgai-mon writes |
| catalog | `catalog.json` in the store: the list of Parquet files that exist |
| WAL | write-ahead log: every record is appended and fsync'ed here before anything else |
| kind | a class of data with its own directory: `samples`, `series`, `logs`, ... |
| rollup | per series and time bucket, a summary of the samples (first, last, count, sum, min, max, counter resets) |
| reader | the DuckDB instance that runs user SQL |

## Processes and threads

`pgai-mon run --config x.yml` starts everything. `pgai-mon serve --store DIR` starts only
the read side over an existing store.

| Unit | Kind | What it does |
|---|---|---|
| server | parent process | HTTP, authentication, admission, result cache, and every writer below. It never runs user SQL. |
| SQL worker | child process (`pgai-mon sql-worker`) | The reader: DuckDB plus the Rust table functions. One child serves all query slots. The parent restarts it when it dies. |
| `store` | thread | The only owner of the WAL and of the flush to Parquet. Receives records over a channel (`Msg` in `lib.rs`). |
| `compactor` | thread | Every 60 s: merges files, builds rollups, applies retention, deletes files past their grace time. Waits while queries run, for at most 10 minutes. |
| uploader | thread, S3 store only | Every `store.s3.upload_interval_s`: uploads new files and commits `catalog.json` to the bucket. |
| `tailer` | thread per `logs[]` entry | Reads new jsonlog lines once a second. |
| collector manager | tokio task | One task per source, per database and per metric run. Re-reads the config and the sources file every `collector.reload_interval_s` and on SIGHUP. |
| scraper, host rules, RDS host stats, self metrics | tokio tasks | Other producers of samples; they write through `ingest.rs`. |
| flush ticker | tokio task | Asks the store thread to flush every `store.flush_interval_s` (60 s). |

With `read_only: true`, or `serve` on an `s3://` store, there is no store thread, no
compaction and no ingest route.

On SIGTERM or Ctrl-C the server stops accepting requests, flushes the WAL to Parquet, and on
an S3 store uploads and commits once more and releases the writer lease.

## Module map

| Role | Module | One line |
|---|---|---|
| Entry | `main.rs` | Commands (`run`, `serve`, `query`, `load`, `load-logs`, `compact`, crash-test helpers), thread and task start-up, stop order. |
| | `lib.rs` | Module list and the `Msg` type sent to the store thread. |
| | `config.rs` | The YAML config, its defaults, and the environment overrides. |
| Producers | `collector.rs` | Runs the SQL of `metrics.yml` per source and database; emits samples with pgwatch's names and labels. |
| | `metricdefs.rs` | Parses `metrics.yml` (SQL per Postgres major version, gauges, presets, intervals). |
| | `pgconn.rs` | Connection strings, `PG*` variables, `.pgpass`, TLS with libpq's `sslmode` values. |
| | `structure.rs` | Snapshots of object definitions (changes only) and `pg_stat_statements` query texts. |
| | `logs.rs` | Tails PostgreSQL jsonlog files and parses auto_explain plans. |
| | `ingest.rs` | Push endpoint `POST /api/v1/import/prometheus` and the in-process API the next four use. |
| | `scrape.rs` | Scrapes Prometheus text endpoints listed in scrape-config files. |
| | `rules.rs` | Evaluates `sql/host_rules.sql` every minute and writes the rows back as `host_*` series. |
| | `hoststats/` | Host metrics of RDS and Aurora from AWS APIs (`rds.rs`: one poll; `aws.rs`: signing, credentials, calls). |
| | `selfmon.rs` | Writes what `GET /health` reports as `pgai_mon_*` series. |
| Store | `model.rs` | The record types that flow from producers through the WAL into Parquet. |
| | `wal.rs` | Append-only log in numbered segments, CRC per frame, fsync per append. |
| | `store.rs` | WAL replay, buffers, flush to Parquet, compaction, retention, purge. |
| | `catalog.rs` | `catalog.json`: file list, flush watermark, pending deletes, writer lease. Atomic rewrite. |
| | `kmerge.rs` | Streaming merge of sorted sample files, used by compaction with bounded memory. |
| | `registry.rs` | Series registry: one row per series ever seen, so the `series` view needs no dedupe. |
| | `bucket.rs` | S3 write path: upload, conditional commit, fencing, garbage collection, local eviction. |
| | `objstore.rs` | S3 read path: planned range GETs, prefetch window, local chunk cache. |
| | `tools.rs` | Loader for VictoriaMetrics exports and jsonlog files; crash-test writer and verifier. |
| Read | `query.rs` | HTTP routes, the reader (DuckDB session, views, lock-down), the read-only check, limits. |
| | `auth.rs` | Bearer tokens and basic auth; what the read-only token may do. |
| | `admission.rs` | Which waiting query gets the next slot. |
| | `qcache.rs` | Result cache in the parent, keyed by SQL text and catalog generation. |
| | `sqlproc.rs` | The reader as a child process with a memory cap; the parent-child protocol. |
| | `rowjson.rs` | Result rows from Arrow to JSON. |
| Evaluator (`eval/`) | `mod.rs`, `eval.sql` | The `prom_*` SQL macros and the handle they share. |
| | `vtab.rs`, `agg.rs`, `topk.rs` | The table functions `prom_eval_rs`, `prom_agg_rs`, `prom_topk_rs` and the raw-sample functions. |
| | `kernels.rs` | The rollup functions (rate, irate, increase, `*_over_time`, ...), matched to VictoriaMetrics. |
| | `scan.rs`, `index.rs` | Streaming scan of sample files per series; in-memory label index for selectors. |
| | `rollup.rs` | Builds 1 h and 1 d rollups and serves long windows from them. |
| | `footer.rs`, `rowcache.rs`, `par.rs` | Parquet footer cache; cache of evaluated rows; worker threads inside one query. |

## Write path

```
collector ─┐
log tailer ─┤
structure  ─┤                 store thread                      compactor thread
push ingest ┼─► channel ─► WAL append + fsync ─► buffers      ┌─ open day: L0 → L1 (hourly)
scraper    ─┤   (Msg)                              │ flush     ├─ closed day: → L2 + 1 h rollup
host rules ─┤                                      ▼           ├─ month files (1 d rollup, ...)
RDS stats  ─┘                         Parquet L0 file per kind ├─ retention: drop from catalog
                                      and day, then catalog ──►└─ purge after grace time
                                                   │
                                  S3 store only:   ▼
                                  uploader: PUT files, then conditional PUT of catalog.json
```

1. **Produce.** A producer sends a record to the store thread. Push ingest and the log tailer
   wait for the acknowledgement; push ingest answers 204 only after it.
2. **WAL.** The store thread appends the record and fsyncs before it enters the in-memory
   buffer (`wal.rs`). If the append fails, the process exits with code 3: it must not
   acknowledge data it cannot keep.
3. **Flush.** Every `store.flush_interval_s`, or when `store.max_buffer_bytes` are buffered,
   the buffers become level-0 Parquet files, one per kind and UTC day. Data is visible to
   queries after the flush. If a flush fails, the records stay in the WAL and the next flush
   retries.
4. **Compaction.** In the open day, flush files of finished hours are merged into one level-1
   file. Once a day is closed (`store.compact_grace_s` after midnight UTC) its files are
   merged into level-2 files sorted by `(series_id, ts)`, and the day's 1 h rollup is built
   from them. The day's new series go to the series registry.
5. **Month files.** The 1 d rollup and a per-month copy of the 1 h rollup are built from the
   closed days' 1 h rollups. Long dashboard ranges read these instead of raw samples.
6. **Retention.** A file whose last day is older than `store.retention_days`
   (`store.texts_retention_days` for query texts) leaves the catalog. A month file leaves once
   its whole month is older. Object definitions, plan definitions and the series registry are
   kept.
7. **Purge.** A file that left the catalog (replaced by compaction or expired) is listed in
   `pending_delete` and deleted `store.delete_grace_s` later, so queries in flight can finish.

## Read path

```
client ─► POST /sql ─► auth ─► result cache ─► admission ─► SQL worker (child process)
                       401/403   hit: reply      429 busy     │ 1. parse: one SELECT only,
                                                              │    table functions on the allowlist
                                                              │ 2. DuckDB over views of the catalog's
                                                              │    Parquet files + Rust table functions
                                                              │ 3. timeout, row cap, byte cap
                                                              ▼
                                               JSON reply (rows | columnar | objects)
```

1. **Auth** (`auth.rs`). Every route except `GET /livez` needs credentials.
2. **Result cache** (`qcache.rs`). Key: SQL text, response format, and a hash of the views.
   An entry is valid for one catalog generation, so it never outlives a flush or compaction.
   Identical requests that arrive while one runs wait for it. Queries that call clock
   functions are not cached. The response header `x-pgai-cache` says how it was served.
3. **Admission** (`admission.rs`). `http.max_concurrent` queries run at once (default: one per
   core, at most 4). One slot is kept for short queries. Up to `http.queue_max` (64) wait for
   at most `http.queue_wait_ms` (30 s); beyond that the answer is `busy`, HTTP 429.
4. **Read-only check** (`Reader::check_read_only` in `query.rs`). DuckDB's own parser
   serialises the statement; anything but exactly one SELECT is refused, and so is a table
   function outside the allowlist.
5. **Execution.** The reader defines one internal view per kind over the file list in the
   catalog (no directory listing) and re-points them when the catalog generation changes. The
   public views are in `schema/views.sql`. The `prom_*` macros call Rust table functions that
   read the Parquet files directly, one series at a time, and use rollups for closed days.
6. **Limits.** See the table.

| Limit | Config key | Default | When reached |
|---|---|---|---|
| Query time | `http.timeout_ms` | 15 s | error `timeout`, HTTP 408 |
| Result rows | `http.max_rows` | 100,000 | reply cut, `"truncated": true`; `format=objects` (dashboards): `result_too_large` (413) |
| Result bytes | `http.max_bytes` | 16 MiB | reply cut, `"truncated": true`; `format=objects`: `result_too_large` (413) |
| DuckDB memory | `http.memory_limit` | 96 MB | error `out_of_memory`, HTTP 413 |
| Reader process RSS | `http.reader_rss_cap_mb` | 192 MB | child killed; queries in flight are retried once, then `out_of_memory` |
| Spill to disk | `http.max_temp_directory_size` | 2 GB | query fails |

Other routes: `GET /schema` (views, columns, metric dictionary), `GET /health` (counters and
state as JSON), `GET /ingest/stats`, `GET /scrape/targets`.

On an `s3://` store DuckDB never sees the bucket. The Rust table functions read S3 with
range GETs. The small kinds the views need (series, query texts, objects, plans, logs) are
copied to a local mirror within `http.s3.mirror_mb`. A kind that does not fit the budget (or
whose download failed) raises an error naming the setting when queried: it never answers as
if it were empty. The `samples` and `metrics` views raise an error there; use
`pgai_samples(...)`.

## Files

Local store directory:

```
<store_dir>/
  catalog.json                                  the file list; rewritten atomically
  wal/<20-digit segment number>.wal             records not yet in Parquet
  tmp/                                          scratch; removed at start
  <kind>/date=YYYY-MM-DD/L<level>-<id>.parquet  level 0 flush, 1 hourly merge, 2 daily
  rollup_1h/date=YYYY-MM-DD/R-<id>.parquet      1 h rollup of a closed day
  series_reg/date=YYYY-MM-DD/G-<id>.parquet     series registry
  series_month/date=YYYY-MM-DD/M-<id>.parquet   series of a month
  rollup_1d/month=YYYY-MM/D-<id>.parquet        1 d rollup
  rollup_1h_month/month=YYYY-MM/H-<id>.parquet  1 h rollup, one file per month
```

Kinds with level files: `samples`, `series`, `logs`, `plan_events`, `plan_defs`,
`query_texts`, `objects`. `<id>` is an 8-digit number from `next_file_id` in the catalog; a
path is never reused.

`catalog.json` holds: `generation` (bumped on every change), `files` (kind, path, day, level,
rows, bytes, time range), `flushed_through_seg` (the highest WAL segment that is fully in
Parquet), `pending_delete` (path and delete time), `log_offsets` (how far each log file was
read), `data_epoch`, and on S3 the `writer` lease.

S3 bucket, under the prefix of `s3://bucket/prefix`: the same `catalog.json` and the same
relative Parquet paths. There is no WAL in the bucket. The writer keeps a local directory
(`store.s3.local_dir`, default `/var/lib/pgai-mon`) with the layout above: the WAL, files not
uploaded yet, and committed files of the last `store.s3.keep_local_days` days.

The `writer` lease in the bucket's catalog names the one process that may commit: `id`,
`epoch`, `host`, `heartbeat_ms` (time of its last commit), `released` (set by a clean stop).

## Invariants

| Invariant | How it is enforced | Where | Test |
|---|---|---|---|
| Exactly-once flush: a crash at any point leaves neither a gap nor a duplicate. | Flush rotates the WAL, writes Parquet under temp names and renames, rewrites the catalog with the new files and `flushed_through_seg`, then deletes the segments. At start, Parquet files the catalog does not list are deleted and segments above the watermark are replayed. | `store.rs`, `wal.rs` | `tests/kill9.rs` (kill -9 at random points), `tests/store.rs` (crash between each step, torn WAL tail) |
| An acknowledged record survives a crash. | fsync per append before the record is buffered; a short or corrupt last frame ends replay. | `wal.rs` | `tests/kill9.rs`, `tests/store.rs` |
| Readers see only whole files. | Readers learn files only from the catalog, which is replaced by tmp + fsync + rename. | `catalog.rs` | `tests/store.rs` |
| One writer per S3 prefix. | A writer takes the lease with a conditional PUT of `catalog.json` (epoch + 1). Every commit is conditional on the version this writer wrote last. A failed precondition means another writer took over: the process exits with code 4. File ids of epoch e start at e × 2^32, so a fenced writer never writes a key the new owner uses. | `bucket.rs` | `tests/bucket.rs`, `tests/run_bucket.rs` |
| S3 commit is atomic. | Files are uploaded first; the catalog PUT is the commit. Objects are immutable. Objects no catalog lists are removed by a LIST sweep. | `bucket.rs` | `tests/bucket.rs` (crash between upload and commit, partial uploads) |
| Delete only after commit plus a grace time. | Replaced and expired files go to `pending_delete` with a time; `purge` deletes them after it. In the bucket the grace (`store.s3.delete_grace_s`) counts from the commit that dropped the file. | `store.rs`, `bucket.rs` | `tests/retention.rs`, `tests/bucket.rs` |
| User SQL cannot write, load code or read outside the store. | Single SELECT check, table function allowlist, `enable_external_access=false`, `allowed_directories` = the store (or the mirror) and scratch, `lock_configuration=true`. | `query.rs` | `tests/guardrails.rs`, `tests/reader_settings.rs` |
| A query cannot take the collector down. | User SQL runs only in the child. A watchdog samples the child's RSS every 20 ms and kills it above the cap; on Linux `RLIMIT_DATA` also bounds it. | `sqlproc.rs`, `admission.rs` | `tests/guardrails.rs` |
| A cached reply equals a fresh one. | The cache key includes the catalog generation; errors and volatile queries are not stored. | `qcache.rs` | `tests/qcache.rs`, `tests/rowcache.rs` |
| Rust functions give the same numbers as the SQL reference. | `sql/macros.sql` keeps a pure-SQL implementation of each function (`prom_<fn>_sql`). | `eval/` | `tests/eval.rs`, `tests/topk_expr.rs`, `tests/rollup_month.rs` |
| A path in a remote catalog cannot point outside the store. | Every path must be plain and relative, or the whole catalog is refused. | `catalog.rs` (`plain_rel`), `query.rs`, `objstore.rs` | `tests/eval.rs` |

## Security model

Credentials come from environment variables or files, never from the YAML text. Secrets are
at least 16 characters, are kept as SHA-256 digests and are compared in constant time. With no
credentials configured the server refuses to start, unless authentication is switched off
explicitly (`PGAI_MON_AUTH_DISABLED=1`, `http.auth.disabled: true`, `serve --no-auth`).

| Credential | Variable | May do |
|---|---|---|
| Full token | `PGAI_MON_TOKEN` or `PGAI_MON_TOKEN_FILE` | Every route, including push ingest. |
| Read-only token | `PGAI_MON_READ_TOKEN` or `PGAI_MON_READ_TOKEN_FILE` | Any GET, and `POST /sql`. Any other request gets 403. Must differ from the full token. |
| Basic auth | `PGAI_MON_AUTH_USERNAME` with `PGAI_MON_AUTH_PASSWORD` or `PGAI_MON_AUTH_PASSWORD_FILE` | Same as the full token. |
| none | | `GET /livez` only, which returns no data. Everything else gets 401. |

What `POST /sql` refuses:

| Request | Error kind | HTTP |
|---|---|---|
| Anything that is not a single SELECT (or `WITH ... SELECT`): DDL, DML, `SET`, `COPY`, `ATTACH`, `INSTALL`, `LOAD`, two statements | `not_read_only` | 403 |
| A table function outside the allowlist, for example `read_csv`, `read_text`, `glob`, `query` | `forbidden` | 403 |
| A file outside the store, named in an allowed function such as `read_parquet` | `forbidden` | 403 |

Allowed table functions (`table_fn_allowed` in `query.rs`): names starting with `prom_`,
`pgai_` or `parquet_`, and `range`, `generate_series`, `unnest`, `read_parquet`, `json_each`,
`json_tree`, `duckdb_columns`, `duckdb_functions`, `duckdb_memory`, `duckdb_settings`,
`duckdb_tables`, `duckdb_types`, `duckdb_views`.

The SQL endpoint gives a reader every row in the store, including query texts and log lines.
There is no per-user or per-database access control: a token grants the whole store.

The collector sets a `lock_timeout` (100 ms) and a statement timeout (15 s unless the metric
sets one) on every session, and opens at most `collector.max_sessions_per_node` (4) sessions
per node.

## Known limits

- One writer per store. There is no replication between pgai-mon processes; durability beyond
  one disk comes from the S3 store.
- With an S3 store, a lost machine loses the data acknowledged since the last commit: up to
  `store.s3.upload_interval_s` (300 s) plus one upload round.
- No TLS in the server. It listens on `127.0.0.1:8765` by default; put a proxy in front for
  remote access.
- PromQL is not parsed. Queries are SQL; the `prom_*` functions reproduce single PromQL
  functions, and combinations are written as SQL joins and aggregates. There is no
  `histogram_quantile`.
- Source kinds other than `postgres` and `postgres-continuous-discovery` (pgbouncer, pgpool,
  patroni) are skipped with a message.
- `pgconn.rs` does not support service files, `sslcrl`, `sslpassword` or GSS; the keys are
  ignored with a warning.
- A node with more databases than `collector.max_sessions_per_node` (4) gets reconnects: an
  idle session is reused only for its own database, so the oldest is closed to make room.
  Raise the setting for nodes with many monitored databases.
- A database that goes away gets no staleness markers: its series end at the last sample.
- Log tailing reads files from a local directory. It does not fetch logs from managed
  services.
- New data is visible to queries after the next flush, up to `store.flush_interval_s` (60 s)
  later.
- All times in the `prom_*` functions are Unix milliseconds.
