//! pgai-mon. One process: collector (pgwatch metrics.yml), jsonlog/auto_explain
//! tailer, structure snapshots, WAL + Parquet store with compaction, and a read-only SQL
//! endpoint over embedded DuckDB.
use clap::{Parser, Subcommand};
use pgai_mon::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "pgai-mon",
    version,
    about = "PostgresAI monitoring: collector, Parquet store and SQL endpoint (Rust + DuckDB)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Collect, store, compact and serve SQL (the product shape).
    Run {
        #[arg(long)]
        config: PathBuf,
    },
    /// Serve SQL over an existing store (no collection). Compaction still runs.
    Serve {
        #[arg(long)]
        store: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8765")]
        listen: String,
        #[arg(long)]
        metrics_yml: Option<PathBuf>,
        #[arg(long)]
        views_sql: Option<PathBuf>,
        /// SQL run after views.sql (e.g. sql/macros.sql).
        #[arg(long)]
        macros_sql: Option<PathBuf>,
        /// Reader DuckDB memory limit (all concurrent queries share it).
        #[arg(long, default_value = "96MB")]
        memory_limit: String,
        /// Reader DuckDB threads (1 = each query runs on its own slot's thread; 0 = max_concurrent).
        #[arg(long, alias = "duckdb-threads", default_value_t = 1)]
        threads: i64,
        /// Query slots (0 = one per core, at most 4).
        #[arg(long, default_value_t = 0)]
        max_concurrent: usize,
        /// A heavy query starts only while the reader RSS is under this % of its cap (0 = off).
        #[arg(long, default_value_t = 50)]
        heavy_mem_gate_pct: u64,
        /// Evaluator threads shared by running queries (0 = max_concurrent; 1, the default, = off).
        #[arg(long, default_value_t = 1)]
        eval_workers: usize,
        /// Memory one evaluator worker may add per day of scan, MB (lent within RSS headroom).
        #[arg(long, default_value_t = 3)]
        eval_worker_mb_per_day: u64,
        #[arg(long, default_value_t = 15000)]
        timeout_ms: u64,
        /// SQL reader child: RSS cap (watchdog kill) and Linux RLIMIT_DATA, MB.
        #[arg(long, default_value_t = 192)]
        reader_rss_cap_mb: u64,
        #[arg(long, default_value_t = 1024)]
        reader_data_limit_mb: u64,
        /// Evaluated-row cache in the reader (prom_topk rows shared by panels), MB; 0 = off.
        #[arg(long, default_value_t = 24)]
        row_cache_mb: u64,
        /// Run SQL in-process (no child; no OS memory cap).
        #[arg(long)]
        in_process: bool,
        /// Don't run background compaction (benchmarks of a fixed store).
        #[arg(long)]
        no_compact: bool,
        /// Reader spill cap.
        #[arg(long, default_value = "2GB")]
        max_temp_directory_size: String,
        /// `--store s3://bucket/prefix` (read-only; credentials and region from the AWS
        /// environment): range GETs in flight.
        #[arg(long, default_value_t = 16)]
        s3_concurrency: usize,
        /// Prefetch window: bytes requested or fetched and not yet decoded, MB. In the reader it
        /// is also kept under its RSS headroom (--s3-window-soft-pct).
        #[arg(long, default_value_t = 8)]
        s3_inflight_mb: u64,
        /// Merge planned ranges into one GET across gaps up to this, KB.
        #[arg(long, default_value_t = 256)]
        s3_coalesce_kb: u64,
        /// Local cache of fetched column chunks and footers, bounded by --s3-cache-gb. On an
        /// s3:// store it defaults to $TMPDIR/pgai-mon-s3cache/<xxh3(store url)> (kept across
        /// restarts); `none` (or '') turns it off.
        #[arg(long)]
        s3_cache_dir: Option<String>,
        #[arg(long, default_value_t = 2.0)]
        s3_cache_gb: f64,
        /// How often the reader re-reads catalog.json from the bucket, ms.
        #[arg(long, default_value_t = 30_000)]
        s3_catalog_ttl_ms: u64,
        /// Raw Parquet footers kept in memory, MB (also on disk under --s3-cache-dir, where they
        /// survive reader restarts): a footer evicted from the parsed cache costs a parse, not GETs.
        #[arg(long, default_value_t = 8)]
        s3_footer_mb: u64,
        /// A query that issues more GETs than this gets a warning in its response stats.
        #[arg(long, default_value_t = 64)]
        s3_gets_warn: u64,
        /// The reader bounds its S3 prefetch window to the room left under this share of
        /// --reader-rss-cap-mb, %: near the cap it prefetches less instead of being killed.
        /// 0 turns the bound off.
        #[arg(long, default_value_t = 85, value_parser = clap::value_parser!(u64).range(0..=100))]
        s3_window_soft_pct: u64,
        /// SQL over an s3:// store: the small kinds DuckDB's views read (series registry and
        /// tail, plans, query texts, objects, then logs) are mirrored to local files (under
        /// --s3-cache-dir, else the scratch dir), all of them together within this budget; a
        /// kind that does not fit is left empty. Samples are never mirrored (`samples`/`metrics`
        /// are empty; prom_* read S3).
        #[arg(long, default_value_t = 2048)]
        s3_mirror_mb: u64,
    },
    /// Build an N-day store from a recorded one (long-window benchmarks): it ends at the
    /// source's last day; earlier days repeat the source's full days with time columns shifted
    /// by whole days. Rollup tiers (1 h, and 1 d when built) and the series registry come from
    /// the store's own compaction. Writes `format_version` to catalog.json and a FORMAT file.
    Tile {
        #[arg(long, required_unless_present = "format_version")]
        src: Option<PathBuf>,
        #[arg(long, required_unless_present = "format_version")]
        out: Option<PathBuf>,
        #[arg(long, required_unless_present = "format_version")]
        days: Option<usize>,
        /// Print the store format this binary writes (v<rollup format>-base or -tier1d, e.g. v4-tier1d) and exit.
        #[arg(long)]
        format_version: bool,
        /// Parallel file rewrites (0 = cores, at most 4).
        #[arg(long, default_value_t = 0)]
        jobs: usize,
        #[arg(long, default_value_t = 9)]
        zstd_level: i32,
    },
    /// Import a VictoriaMetrics export (/api/v1/export JSON lines; "-" = stdin), and/or query
    /// texts and object definitions as JSON lines (test fixtures; see structure::load_jsonl).
    Load {
        #[arg(long)]
        store: PathBuf,
        #[arg(long, required_unless_present_any = ["query_texts", "objects"])]
        input: Option<PathBuf>,
        /// JSON lines: {"queryid","dbname","query","first_seen_ms","last_seen_ms"[,"source","cluster","node_name"]}
        #[arg(long)]
        query_texts: Option<PathBuf>,
        /// JSON lines: {"ts_ms","dbname","kind","schema","name","definition"[,"source","cluster","node_name"]}
        #[arg(long)]
        objects: Option<PathBuf>,
        #[arg(long, default_value = store::DEFAULT_LAYOUT)]
        layout: String,
        #[arg(long, default_value_t = 1_000_000)]
        chunk: usize,
        #[arg(long, default_value = "64MB")]
        writer_memory_limit: String,
        /// Leave staging files uncompacted (run `compact --all` later).
        #[arg(long)]
        no_compact: bool,
    },
    /// Import jsonlog files offline (a file or a directory), chunked and resumable.
    LoadLogs {
        #[arg(long)]
        store: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = ".json")]
        suffix: String,
        #[arg(long, default_value = "64MB")]
        writer_memory_limit: String,
        #[arg(long)]
        no_compact: bool,
    },
    /// Run compaction now (`--all` treats every day, including today, as closed).
    Compact {
        #[arg(long)]
        store: PathBuf,
        #[arg(long)]
        all: bool,
        #[arg(long, default_value = "32MB")]
        compact_memory_limit: String,
        #[arg(long, default_value_t = 9)]
        zstd_level: i32,
        #[arg(long, default_value_t = 3_500_000)]
        part_rows: u64,
    },
    /// One read-only query through the same guardrails as POST /sql.
    Query {
        #[arg(long)]
        store: PathBuf,
        #[arg(long)]
        metrics_yml: Option<PathBuf>,
        sql: String,
    },
    /// Rewrite a store's samples in every layout variant and report bytes/sample.
    LayoutBench {
        #[arg(long)]
        store: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 9)]
        zstd_level: i32,
    },
    /// Crash test writer: append deterministic samples and flush until killed.
    CrashtestWrite {
        #[arg(long)]
        store: PathBuf,
        #[arg(long, default_value_t = 100)]
        series: u64,
        #[arg(long, default_value_t = 50)]
        flush_every: u64,
        #[arg(long, default_value_t = u64::MAX / 2)]
        steps: u64,
        /// Leave the last records only in the WAL (for the negative test).
        #[arg(long)]
        no_final_flush: bool,
    },
    /// Rewrite series files as per-day files (stores written before per-day series).
    MigrateSeries {
        #[arg(long)]
        store: PathBuf,
    },
    /// Test data: add N never-sampled series of a metric (queryid churn) over the last D days.
    SynthChurn {
        #[arg(long)]
        store: PathBuf,
        /// metric, or a comma list to spread the series over
        #[arg(long, default_value = "pgwatch_pg_stat_statements_calls")]
        metric: String,
        #[arg(long)]
        series: u64,
        #[arg(long, default_value_t = 395)]
        days: i64,
        /// last day (days since 1970-01-01); default: the newest samples day
        #[arg(long)]
        last_day: Option<i64>,
    },
    /// Review B6 gate (b): churned pgss series with samples, written as final day files.
    SynthSampled {
        #[arg(long)]
        store: PathBuf,
        #[arg(
            long,
            default_value = "pgwatch_pg_stat_statements_calls,pgwatch_pg_stat_statements_exec_time_total,pgwatch_pg_stat_statements_plan_time_total"
        )]
        metrics: String,
        #[arg(long, default_value_t = 30)]
        days: i64,
        /// last day (days since 1970-01-01)
        #[arg(long)]
        last_day: i64,
        #[arg(long, default_value_t = 2000)]
        base: u64,
        #[arg(long, default_value_t = 4000)]
        new_per_day: u64,
        #[arg(long, default_value_t = 30)]
        step_s: i64,
    },
    /// Internal: the SQL reader child process (spawned by run/serve).
    #[command(hide = true)]
    SqlWorker {
        #[arg(long)]
        spec: PathBuf,
    },
    /// Crash test verifier: no gap, no duplicate.
    CrashtestVerify {
        #[arg(long)]
        store: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let r = main_cmd();
    // a read-only open's scratch dir (crashes and kills are swept by the next open)
    store::remove_own_scratch();
    r
}

fn main_cmd() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run { config } => run(config::Config::load(&config)?),
        Cmd::Serve {
            store,
            listen,
            metrics_yml,
            views_sql,
            macros_sql,
            memory_limit,
            threads,
            max_concurrent,
            heavy_mem_gate_pct,
            eval_workers,
            eval_worker_mb_per_day,
            timeout_ms,
            reader_rss_cap_mb,
            reader_data_limit_mb,
            row_cache_mb,
            in_process,
            mut no_compact,
            max_temp_directory_size,
            s3_concurrency,
            s3_inflight_mb,
            s3_coalesce_kb,
            s3_cache_dir,
            s3_cache_gb,
            s3_catalog_ttl_ms,
            s3_footer_mb,
            s3_gets_warn,
            s3_window_soft_pct,
            s3_mirror_mb,
        } => {
            let s3 = objstore::S3Cfg {
                concurrency: s3_concurrency,
                inflight_mb: s3_inflight_mb,
                coalesce_kb: s3_coalesce_kb,
                cache_dir: match s3_cache_dir.as_deref() {
                    Some("") | Some("none") => None,
                    Some(d) => Some(PathBuf::from(d)),
                    None if objstore::is_remote(&store) => {
                        Some(objstore::make_default_cache_dir(&store.to_string_lossy())?)
                    }
                    None => None,
                },
                cache_gb: s3_cache_gb,
                catalog_ttl_ms: s3_catalog_ttl_ms,
                footer_mb: s3_footer_mb,
                gets_warn: s3_gets_warn,
                window_soft_pct: s3_window_soft_pct,
                mirror_mb: s3_mirror_mb,
            };
            if objstore::is_remote(&store) {
                // nothing to compact into: an object store is served read-only
                no_compact = true;
                objstore::open(&store.to_string_lossy(), &s3)?;
            }
            if no_compact {
                std::env::set_var("PGAI_NO_COMPACT", "1");
            }
            let cfg = config::Config {
                store_dir: store,
                metrics_yml: metrics_yml.unwrap_or_else(|| PathBuf::from("/nonexistent")),
                preset: "full".into(),
                sources: vec![],
                sources_file: None,
                config_path: None,
                store: Default::default(),
                http: config::HttpCfg {
                    listen,
                    memory_limit,
                    threads,
                    max_concurrent,
                    heavy_mem_gate_pct,
                    eval_workers,
                    eval_worker_mb_per_day,
                    timeout_ms,
                    reader_rss_cap_mb,
                    reader_data_limit_mb,
                    row_cache_mb,
                    reader_process: !in_process,
                    max_temp_directory_size,
                    s3,
                    ..Default::default()
                },
                collector: Default::default(),
                logs: vec![],
                views_sql,
                macros_sql,
                read_only: no_compact,
            };
            run(cfg)
        }
        Cmd::Load {
            store,
            input,
            query_texts,
            objects,
            layout,
            chunk,
            writer_memory_limit,
            no_compact,
        } => {
            let cfg = config::StoreCfg {
                samples_layout: layout,
                writer_memory_limit,
                ..Default::default()
            };
            let mut core = store::StoreCore::open(&store, &cfg)?;
            if query_texts.is_some() || objects.is_some() {
                let st = structure::load_jsonl(
                    &mut core,
                    query_texts.as_deref(),
                    objects.as_deref(),
                    !no_compact && input.is_none(),
                )?;
                println!("{}", serde_json::json!({ "load_structure": st }));
            }
            let Some(input) = input else { return Ok(()) };
            let mut rd: Box<dyn std::io::BufRead> = if input.as_os_str() == "-" {
                Box::new(std::io::BufReader::with_capacity(1 << 20, std::io::stdin()))
            } else {
                Box::new(std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(&input)?))
            };
            let st = tools::load_vm_export(&mut core, &mut rd, chunk, !no_compact)?;
            let bytes: u64 = core
                .shared
                .catalog
                .lock()
                .unwrap()
                .files_of("samples")
                .map(|f| f.bytes)
                .sum();
            println!(
                "{}",
                serde_json::json!({"load": st, "samples_parquet_bytes": bytes,
                    "bytes_per_sample": bytes as f64 / st.samples.max(1) as f64})
            );
            Ok(())
        }
        Cmd::Tile {
            src,
            out,
            days,
            format_version,
            jobs,
            zstd_level,
        } => {
            if format_version {
                println!("{}", tile::format_version()?);
                return Ok(());
            }
            let r = tile::tile(&tile::Opts {
                src: src.unwrap(),
                out: out.unwrap(),
                days: days.unwrap(),
                jobs,
                zstd_level,
                verbose: true,
            })?;
            println!(
                "{} days ({} .. {}), {} files, {} bytes, format {} (tiers {}), {} merges",
                r.map.len(),
                model::day_str(r.map.last().unwrap().0),
                model::day_str(r.map[0].0),
                r.files,
                r.bytes,
                r.format,
                r.tiers.join(","),
                r.merges
            );
            Ok(())
        }
        Cmd::Compact {
            store,
            all,
            compact_memory_limit,
            zstd_level,
            part_rows,
        } => {
            let cfg = config::StoreCfg {
                compact_memory_limit,
                zstd_level,
                compact_part_rows: part_rows,
                ..Default::default()
            };
            let core = store::StoreCore::open(&store, &cfg)?;
            let n = core.compactor()?.run_once(model::now_ms(), all)?;
            println!("{n} merges");
            Ok(())
        }
        Cmd::Query {
            store,
            metrics_yml,
            sql,
        } => {
            // read-only: no WAL, no tmp in the store (it may be someone else's or read-only)
            let shared = store::Shared::open_read_only(&store)?;
            let mf = metrics_yml.and_then(|p| metricdefs::load(&p).ok());
            let r = query::Reader::new(
                shared.clone(),
                Default::default(),
                query::EMBEDDED_VIEWS_SQL.to_string(),
                query::build_dict(mf.as_ref(), &mf.as_ref().map(|m| m.dict_intervals(&["full"], &[])).unwrap_or_default()),
            )?;
            match r.run(&sql) {
                Ok(res) => println!("{}", serde_json::to_string(&res)?),
                Err(e) => {
                    println!("{}", serde_json::json!({"error": e}));
                    store::remove_own_scratch();
                    std::process::exit(2);
                }
            }
            Ok(())
        }
        Cmd::LoadLogs {
            store,
            input,
            suffix,
            writer_memory_limit,
            no_compact,
        } => {
            let cfg = config::StoreCfg {
                writer_memory_limit,
                ..Default::default()
            };
            let mut core = store::StoreCore::open(&store, &cfg)?;
            let st = tools::load_logs(&mut core, &input, &suffix, !no_compact)?;
            println!("{}", serde_json::json!({ "load_logs": st }));
            Ok(())
        }
        Cmd::SqlWorker { spec } => sqlproc::child_main(&spec),
        Cmd::SynthSampled {
            store,
            metrics,
            days,
            last_day,
            base,
            new_per_day,
            step_s,
        } => {
            let ms: Vec<&str> = metrics.split(',').collect();
            println!(
                "{}",
                tools::synth_sampled(&store, &ms, days, last_day, base, new_per_day, step_s)?
            );
            Ok(())
        }
        Cmd::MigrateSeries { store } => {
            let mut core = store::StoreCore::open(&store, &Default::default())?;
            println!("{}", tools::migrate_series_per_day(&mut core)?);
            println!("{}", tools::series_fix_and_register(&mut core)?);
            Ok(())
        }
        Cmd::SynthChurn {
            store,
            metric,
            series,
            days,
            last_day,
        } if metric.contains(',') => {
            // spread over several metrics (e.g. every pg_stat_statements metric)
            let mut core = store::StoreCore::open(&store, &Default::default())?;
            let ld = match last_day {
                Some(d) => d,
                None => core
                    .shared
                    .catalog
                    .lock()
                    .unwrap()
                    .files_of("samples")
                    .map(|f| f.day)
                    .max()
                    .unwrap_or(0),
            };
            let ms: Vec<&str> = metric.split(',').collect();
            for m in &ms {
                println!(
                    "{m}: {}",
                    tools::synth_churn(&mut core, m, series / ms.len() as u64, days, ld)?
                );
            }
            Ok(())
        }
        Cmd::SynthChurn {
            store,
            metric,
            series,
            days,
            last_day,
        } => {
            let mut core = store::StoreCore::open(&store, &Default::default())?;
            let ld = match last_day {
                Some(d) => d,
                None => core
                    .shared
                    .catalog
                    .lock()
                    .unwrap()
                    .files_of("samples")
                    .map(|f| f.day)
                    .max()
                    .unwrap_or(0),
            };
            println!("{}", tools::synth_churn(&mut core, &metric, series, days, ld)?);
            Ok(())
        }
        Cmd::LayoutBench { store, out, zstd_level } => layout_bench(&store, &out, zstd_level),
        Cmd::CrashtestWrite {
            store,
            series,
            flush_every,
            steps,
            no_final_flush,
        } => {
            let mut core = store::StoreCore::open(&store, &Default::default())?;
            tools::crashtest_write(&mut core, series, flush_every, steps, !no_final_flush)
        }
        Cmd::CrashtestVerify { store } => {
            // Open = the same recovery a restart does (WAL replay), then flush it.
            let cfg = config::StoreCfg {
                writer_memory_limit: "2GB".into(),
                ..Default::default()
            };
            let mut core = store::StoreCore::open(&store, &cfg)?;
            core.flush()?;
            let v = tools::crashtest_verify(&core)?;
            println!("{v}");
            if v["pass"].as_bool() != Some(true) {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

fn layout_bench(store_dir: &std::path::Path, out: &std::path::Path, zstd_level: i32) -> anyhow::Result<()> {
    let core = store::StoreCore::open(store_dir, &Default::default())?;
    let cat = core.shared.catalog.lock().unwrap().clone();
    let files: Vec<_> = cat.files_of("samples").collect();
    anyhow::ensure!(!files.is_empty(), "no samples");
    let src = format!("read_parquet({})", store::file_list_sql(&core.shared.root, &files));
    let db = core.conn();
    db.execute_batch(&format!(
        "CREATE OR REPLACE TEMP TABLE buf_samples AS SELECT series_id, epoch_ms(ts) AS ts, value, stale FROM ({}) ORDER BY series_id, ts",
        store::samples_raw_select(&cat.samples_layout, &src)
    ))?;
    let n: i64 = db.query_row("SELECT count(*) FROM buf_samples", [], |r| r.get(0))?;
    std::fs::create_dir_all(out)?;
    let mut res = Vec::new();
    for layout in store::LAYOUTS {
        let cfg = config::StoreCfg {
            zstd_level,
            ..Default::default()
        };
        let p = out.join(format!("{layout}.parquet"));
        let proj = match *layout {
            l if l.contains("split") && l.ends_with("_tz") => "series_id, make_timestamptz(ts * 1000) AS ts, CASE WHEN value = trunc(value) AND abs(value) < 9.0e15 THEN value::BIGINT END AS ival, CASE WHEN value = trunc(value) AND abs(value) < 9.0e15 THEN NULL ELSE value END AS fval",
            l if l.contains("split") => "series_id, epoch_ms(ts)::TIMESTAMP_MS AS ts, CASE WHEN value = trunc(value) AND abs(value) < 9.0e15 THEN value::BIGINT END AS ival, CASE WHEN value = trunc(value) AND abs(value) < 9.0e15 THEN NULL ELSE value END AS fval",
            l if l.ends_with("_tz") => "series_id, make_timestamptz(ts * 1000) AS ts, value",
            _ => "series_id, epoch_ms(ts)::TIMESTAMP_MS AS ts, value",
        };
        let t0 = std::time::Instant::now();
        let bytes = store::copy_atomic(
            db,
            &cfg,
            &format!("SELECT {proj} FROM buf_samples ORDER BY series_id, ts"),
            &p,
            layout,
        )?;
        let cols: Vec<(String, i64, String)> = {
            let mut st = db.prepare(&format!(
                "SELECT path_in_schema, sum(total_compressed_size)::BIGINT, string_agg(DISTINCT encodings, ' ') \
                 FROM parquet_metadata('{}') GROUP BY 1 ORDER BY 1",
                p.display()
            ))?;
            let rows = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
            rows.collect::<Result<_, _>>()?
        };
        let col_bps: serde_json::Map<String, serde_json::Value> = cols
            .iter()
            .map(|(c, b, e)| {
                (
                    c.clone(),
                    serde_json::json!({"bytes_per_sample": *b as f64 / n as f64, "encodings": e}),
                )
            })
            .collect();
        res.push(serde_json::json!({
            "layout": layout, "bytes": bytes, "samples": n,
            "bytes_per_sample": bytes as f64 / n as f64, "write_ms": t0.elapsed().as_millis() as u64,
            "columns": col_bps,
        }));
    }
    println!("{}", serde_json::to_string_pretty(&res)?);
    Ok(())
}

fn run(mut cfg: config::Config) -> anyhow::Result<()> {
    cfg.http = cfg.http.clone().resolve();
    if cfg.read_only {
        anyhow::ensure!(
            !cfg.has_sources() && cfg.logs.is_empty(),
            "read-only: no sources or logs"
        );
    }
    // store_dir s3://bucket/prefix: the writer works in store.s3.local_dir and commits to the
    // bucket (bucket.rs); its reader reads the bucket with that directory as the overlay
    let bucket = if objstore::is_remote(&cfg.store_dir) && !cfg.read_only {
        let local = cfg
            .store
            .s3
            .local_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(bucket::DEFAULT_LOCAL_DIR));
        let b = bucket::Bucket::open(&cfg.store_dir.to_string_lossy(), &local, cfg.store.s3.clone(), &cfg.http.s3)?;
        b.attach(&cfg.store.samples_layout)?;
        eprintln!(
            "pgai-mon: writing {} (epoch {}, local dir {}, upload every {} s)",
            b.url,
            b.epoch(),
            b.local.display(),
            b.cfg.upload_interval_s
        );
        Some(b)
    } else {
        None
    };
    let core = if cfg.read_only {
        None
    } else {
        let root = bucket.as_ref().map_or(cfg.store_dir.clone(), |b| b.local.clone());
        Some(store::StoreCore::open(&root, &cfg.store)?)
    };
    let shared = match &core {
        Some(c) => c.shared.clone(),
        None => store::Shared::open_read_only(&cfg.store_dir)?,
    };
    // what the reader reads: the bucket URL (same catalog and stats, local files win)
    let reader_shared = match &bucket {
        Some(b) => store::Shared {
            root: PathBuf::from(&b.url),
            ..shared.clone()
        },
        None => shared.clone(),
    };
    let log_offsets = core.as_ref().map(|c| c.log_offsets()).unwrap_or_default();
    let mf = metricdefs::load(&cfg.metrics_yml).ok();
    if mf.is_none() && cfg.has_sources() {
        anyhow::bail!("cannot read metrics.yml at {}", cfg.metrics_yml.display());
    }
    let mut views_sql = match &cfg.views_sql {
        Some(p) => std::fs::read_to_string(p)?,
        None => query::EMBEDDED_VIEWS_SQL.to_string(),
    };
    if let Some(p) = &cfg.macros_sql {
        views_sql.push_str("\n;\n");
        views_sql.push_str(&std::fs::read_to_string(p)?);
    }
    // intervals from the global preset and every source's own lists (pg_index_pilot, ...)
    let dict_iv = match &mf {
        Some(m) => {
            let sources = if cfg.has_sources() { config::collect_sources(&cfg).unwrap_or_default() } else { vec![] };
            m.dict_intervals(&[cfg.preset.as_str()], &sources)
        }
        None => Default::default(),
    };
    let dict = query::build_dict(mf.as_ref(), &dict_iv);
    let views_text = format!("{views_sql}\n;\n{}", eval::EVAL_SQL);
    let backend = if cfg.http.reader_process {
        let spec = sqlproc::Spec {
            store: if objstore::is_remote(&reader_shared.root) {
                reader_shared.root.clone()
            } else {
                reader_shared.root.canonicalize()?
            },
            overlay: bucket.as_ref().map(|b| b.local.clone()),
            scratch: shared.scratch.clone(),
            http: cfg.http.clone(),
            views_sql,
            dict,
        };
        let spec_path = shared.scratch.join("reader_spec.json");
        std::fs::create_dir_all(spec_path.parent().unwrap())?;
        query::Backend::Proc(sqlproc::ReaderProc::new(&spec, spec_path)?)
    } else {
        query::Backend::InProc(Arc::new(query::Reader::new(
            reader_shared.clone(),
            cfg.http.clone(),
            views_sql,
            dict,
        )?))
    };
    let objects = structure::load_current_objects(&shared)?;

    let (tx, mut rx) = tokio::sync::mpsc::channel::<Msg>(4096);

    // Read-only: no store thread; drain the (unused) channel and acknowledge flushes.
    let Some(core) = core else {
        std::thread::Builder::new().name("store-ro".into()).spawn(move || {
            while let Some(m) = rx.blocking_recv() {
                if let Msg::Flush(Some(a)) = m {
                    let _ = a.send(());
                }
            }
        })?;
        return serve_loop(cfg, mf, objects, shared, backend, tx, None, views_text);
    };
    // Store thread: sole owner of the WAL and the flush path.
    let compactor = core.compactor()?;
    let upload_compactor = if bucket.is_some() { Some(core.upload_compactor()?) } else { None };
    let store_thread = std::thread::Builder::new().name("store".into()).spawn(move || {
        let mut core = core;
        // Flush whatever the WAL replay produced right away.
        if let Err(e) = core.flush() {
            eprintln!("store: initial flush: {e}");
        }
        while let Some(m) = rx.blocking_recv() {
            match m {
                Msg::Rec(r) => {
                    if let Err(e) = core.append(r) {
                        // Without a durable WAL we must not acknowledge data: stop.
                        eprintln!("store: WAL append failed: {e}; exiting");
                        std::process::exit(3);
                    }
                    if let Err(e) = core.flush_if_full() {
                        eprintln!("store: flush failed (data stays in WAL): {e}");
                    }
                }
                Msg::RecAck(r, ack) => {
                    if let Err(e) = core.append(r) {
                        eprintln!("store: WAL append failed: {e}; exiting");
                        std::process::exit(3);
                    }
                    if let Err(e) = core.flush_if_full() {
                        eprintln!("store: flush failed (data stays in WAL): {e}");
                    }
                    let _ = ack.send(());
                }
                Msg::Flush(ack) => {
                    if let Err(e) = core.flush() {
                        eprintln!("store: flush failed (data stays in WAL): {e}");
                    }
                    if let Some(a) = ack {
                        let _ = a.send(());
                    }
                }
            }
        }
        let _ = core.flush();
    })?;
    let compact_enabled = std::env::var_os("PGAI_NO_COMPACT").is_none();
    std::thread::Builder::new().name("compactor".into()).spawn(move || {
        if !compact_enabled {
            return;
        }
        // Defer while SQL queries run (their memory and compaction's would add up), but
        // never more than 10 minutes.
        let mut deferred_since: Option<std::time::Instant> = None;
        loop {
            std::thread::sleep(Duration::from_secs(if deferred_since.is_some() { 5 } else { 60 }));
            let busy = compactor
                .shared
                .active_queries
                .load(std::sync::atomic::Ordering::Relaxed)
                > 0;
            if busy && deferred_since.map_or(true, |t| t.elapsed() < Duration::from_secs(600)) {
                deferred_since.get_or_insert_with(std::time::Instant::now);
                continue;
            }
            deferred_since = None;
            if let Err(e) = compactor.run_once(model::now_ms(), false) {
                eprintln!("compactor: {e}");
            }
        }
    })?;
    // Uploads on their own thread: never deferred by queries nor delayed by a compaction job.
    if let (Some(b), Some(premerge)) = (&bucket, upload_compactor) {
        let ftx = tx.clone();
        bucket::spawn_uploader(
            b.clone(),
            shared.clone(),
            premerge,
            Box::new(move || {
                // the WAL's records to local Parquet first, so the round commits all acked data
                let (a, r) = tokio::sync::oneshot::channel();
                if ftx.blocking_send(Msg::Flush(Some(a))).is_ok() {
                    let _ = r.blocking_recv();
                }
            }),
            Box::new(|e| {
                eprintln!("bucket: {e}");
                std::process::exit(4);
            }),
            Duration::from_secs(30),
        )?;
    }

    // Log tailers.
    for ls in &cfg.logs {
        let mut t = logs::Tailer {
            dir: ls.dir.clone(),
            suffix: ls.suffix.clone(),
            offsets: log_offsets.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            start_at_end: ls.start_at_end,
            known_plans: Default::default(),
            fp_map: shared.fp_to_queryid.clone(),
            max_read: 4 << 20,
        };
        let tx = tx.clone();
        std::thread::Builder::new().name("tailer".into()).spawn(move || loop {
            let mut got = false;
            match t.poll() {
                Ok(recs) => {
                    for r in recs {
                        got = true;
                        let (a, b) = tokio::sync::oneshot::channel();
                        if tx.blocking_send(Msg::RecAck(r, a)).is_err() || b.blocking_recv().is_err() {
                            return;
                        }
                    }
                }
                Err(e) => eprintln!("tailer {}: {e}", t.dir.display()),
            }
            if !got {
                std::thread::sleep(Duration::from_secs(1));
            }
        })?;
    }

    let final_tx = tx.clone();
    let res = serve_loop(cfg, mf, objects, shared.clone(), backend, tx, Some(store_thread), views_text);
    // Stop order: serve_loop has stopped HTTP, flushed, and dropped its runtime (the collector
    // manager and its tasks with it). Flush once more what they sent meanwhile, then commit,
    // so a clean stop leaves nothing flushed and not uploaded (log tailers resume from the
    // committed offsets).
    if let Some(b) = &bucket {
        let (a, r) = tokio::sync::oneshot::channel();
        if final_tx.blocking_send(Msg::Flush(Some(a))).is_ok() {
            let _ = r.blocking_recv();
        }
        if let Err(e) = b.sync(&shared, None) {
            eprintln!("bucket: final upload failed (data stays in {}): {e:#}", b.local.display());
        }
    }
    res
}

#[allow(clippy::too_many_arguments)]
fn serve_loop(
    cfg: config::Config,
    mf: Option<metricdefs::MetricsFile>,
    objects: std::collections::HashMap<String, std::collections::HashMap<structure::ObjKey, u64>>,
    shared: store::Shared,
    backend: query::Backend,
    tx: tokio::sync::mpsc::Sender<Msg>,
    store_thread: Option<std::thread::JoinHandle<()>>,
    views_text: String,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(cfg.http.max_concurrent + 4)
        .enable_all()
        .build()?;
    rt.block_on(async move {
        if mf.is_some() && cfg.has_sources() {
            let ctx = collector::CollectorCtx {
                em: collector::Emitter::new(shared.clone(), tx.clone(), cfg.collector.clone()),
                objects: Arc::new(std::sync::Mutex::new(objects)),
            };
            tokio::spawn(collector::run_manager(cfg.clone(), ctx));
        }
        // Flush ticker.
        {
            let tx = tx.clone();
            let every = Duration::from_secs(cfg.store.flush_interval_s.max(1));
            tokio::spawn(async move {
                let mut iv = tokio::time::interval(every);
                iv.tick().await;
                loop {
                    iv.tick().await;
                    if tx.send(Msg::Flush(None)).await.is_err() {
                        break;
                    }
                }
            });
        }
        let app =
            query::router(query::AppState::new(backend, cfg.http.clone(), shared.clone()).with_cache(&views_text));
        let listener = tokio::net::TcpListener::bind(&cfg.http.listen).await?;
        eprintln!("pgai-mon: serving on http://{}", cfg.http.listen);
        let shutdown = async {
            let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {},
                _ = term.recv() => {},
            }
        };
        axum::serve(listener, app).with_graceful_shutdown(shutdown).await?;
        let (a, b) = tokio::sync::oneshot::channel();
        let _ = tx.send(Msg::Flush(Some(a))).await;
        let _ = b.await;
        anyhow::Ok(())
    })?;
    drop(store_thread);
    Ok(())
}
