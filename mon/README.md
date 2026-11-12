# pgai-mon

PostgresAI monitoring collector, Parquet store and SQL endpoint in one Rust binary with
embedded DuckDB.

- Collects every metric in `config/pgwatch-prometheus/metrics.yml` with pgwatch v3 semantics
  (same metric names and labels as the pgwatch + VictoriaMetrics stack).
- Tails PostgreSQL jsonlog / auto_explain output, and snapshots query texts and object
  definitions.
- Writes a WAL, flushes it to Parquet, compacts closed days and builds 1 h / 1 d rollups.
- Serves read-only SQL over HTTP. Consumers query only the views in `schema/views.sql`; the
  PromQL-shaped table functions (`prom_rate`, `prom_topk`, ...) are in `src/eval/` and
  `sql/macros.sql`.

The HTTP endpoint has no authentication. Keep it on an internal network.

## Layout

| Path | What |
|---|---|
| `src/` | the crate (`pgai-mon` binary, `pgai_mon` library) |
| `tests/` | integration tests (`tests/fixtures/`: sample jsonlog lines) |
| `schema/views.sql` | the consumer views; embedded in the binary (`include_str!`) |
| `sql/macros.sql` | SQL macros loaded after the views (`macros_sql` / `--macros-sql`) |
| `sql/reports/` | ported reporter and dashboard queries (`proposed/`: evaluator variants of R1 and R9) |
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

Run from the repository root (the default `metrics_yml` is `config/pgwatch-prometheus/metrics.yml`,
relative to the working directory).

```bash
B=mon/target/release/pgai-mon

# collect + store + compact + serve
$B run --config my.yml

# SQL over an existing store, no collection
$B serve --store DIR --listen 127.0.0.1:8765 \
    --metrics-yml config/pgwatch-prometheus/metrics.yml --macros-sql mon/sql/macros.sql
curl -s localhost:8765/sql -d "select metric, count(*) from series group by 1 order by 2 desc limit 5"
curl -s localhost:8765/sql -H 'content-type: application/json' -d '{"sql":"select 1"}'
curl -s 'localhost:8765/sql?format=objects' -d "select 1 as x"   # the format Grafana's Infinity reads
curl -s localhost:8765/schema    # views, columns, metric names with type/interval/description
curl -s localhost:8765/health    # ingest counters, files, bytes, reader state

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
benchmark and crash-test helpers (`tile`, `layout-bench`, `crashtest-write`,
`crashtest-verify`, `synth-*`).

## Config

YAML; the reference is `src/config.rs`. Only `store_dir` and `sources` are required; every
other field has a default. Unknown fields are rejected. See
`examples/pgai-mon.example.yml`:

- `sources[]`: `name`, `conn_str`, `custom_tags`, `discover_databases`, `scrape_labels`;
- `metrics_yml`, `preset` (default `full`);
- `store`: flush interval, writer/compaction memory limits, compaction settings;
- `http`: `listen` (default `127.0.0.1:8765`), reader memory and RSS caps, timeouts, row and
  byte limits, caches, query slots, `s3` read path;
- `collector`: series and session limits, statement timeout, snapshot intervals;
- `logs[]`: jsonlog directories to tail;
- `views_sql` (default: the embedded copy), `macros_sql`, `read_only`.

A store root may also be `s3://bucket/prefix` (credentials and region from the AWS
environment). `serve` reads it. `run` writes it with `store.s3.local_dir` as its local working
directory (WAL, files not uploaded yet, cache) and commits to the bucket every
`store.s3.upload_interval_s` (default 300 s; a lost machine loses at most about that much). One
writer per prefix: a second `run` is refused while the first one's lease is live
(`store.s3.takeover: true` forces it, and the first is fenced at its next commit). See
`src/bucket.rs`. `/health` shows the write path under `s3_write`.

## Tests

```bash
cd mon
cargo test             # unit + integration tests
cargo test -- --ignored --nocapture   # manual measurements (not gates)
python3 grafana/test_compile.py       # offline dashboard checks
```

The tests need no PostgreSQL and no network: they build their stores in temp dirs (and
in-memory object stores). `kill9.rs` and `rowcache.rs` run the built binary.
