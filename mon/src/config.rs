//! Runtime configuration (YAML). Everything has a default so `pgai-mon run --config x.yml`
//! needs only the sources.
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Root of the store (WAL, Parquet, catalog).
    pub store_dir: PathBuf,
    /// pgwatch metrics.yml, read unchanged.
    #[serde(default = "d_metrics_yml")]
    pub metrics_yml: PathBuf,
    /// Preset from metrics.yml that gives the metric list and intervals.
    #[serde(default = "d_preset")]
    pub preset: String,
    #[serde(default)]
    pub sources: Vec<Source>,
    #[serde(default)]
    pub store: StoreCfg,
    #[serde(default)]
    pub http: HttpCfg,
    #[serde(default)]
    pub collector: CollectorCfg,
    /// jsonlog directories to tail, one per source name (optional).
    #[serde(default)]
    pub logs: Vec<LogSource>,
    /// Path to views.sql (the consumer contract). Embedded copy used when absent.
    #[serde(default)]
    pub views_sql: Option<PathBuf>,
    /// Extra SQL run after views.sql on the reader (e.g. sql/macros.sql: CREATE MACRO only).
    #[serde(default)]
    pub macros_sql: Option<PathBuf>,
    /// Open the store read-only (no WAL, no compaction, no ingest, nothing written in the store).
    #[serde(default)]
    pub read_only: bool,
}

fn d_metrics_yml() -> PathBuf {
    PathBuf::from("config/pgwatch-prometheus/metrics.yml")
}
fn d_preset() -> String {
    "full".into()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    /// pgwatch source name: becomes the `dbname` label, exactly as pgwatch does.
    pub name: String,
    pub conn_str: String,
    /// Labels added to every series (pgwatch `custom_tags`).
    #[serde(default)]
    pub custom_tags: BTreeMap<String, String>,
    /// Also collect per-database metrics on every other connectable, non-template
    /// database of the node (pgwatch "continuous discovery" naming: `<name>_<datname>`).
    #[serde(default)]
    pub discover_databases: bool,
    /// Labels pgwatch never sees but VictoriaMetrics adds at scrape time (job, instance).
    /// Kept configurable so live series can be made label-identical to the reference.
    #[serde(default)]
    pub scrape_labels: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogSource {
    pub dir: PathBuf,
    /// Glob-free suffix filter, e.g. ".json".
    #[serde(default = "d_log_suffix")]
    pub suffix: String,
    /// Start at the end of existing files on first run (true) or read them whole (false).
    #[serde(default)]
    pub start_at_end: bool,
}
fn d_log_suffix() -> String {
    ".json".into()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct StoreCfg {
    /// Seconds between WAL -> Parquet flushes (query visibility latency).
    pub flush_interval_s: u64,
    /// Samples older than (newest sample - closed_lag_ms) count as late: data_epoch is bumped.
    pub closed_lag_ms: i64,
    /// Memory limit of the writer DuckDB instance (WAL -> Parquet flush, loader chunks).
    pub writer_memory_limit: String,
    /// Memory limit of the compaction DuckDB instance (its own instance, so the limits
    /// don't compete). RSS of a compaction is roughly this + ~100 MB of reader buffers.
    pub compact_memory_limit: String,
    /// Compaction memory for non-sample kinds (streaming copy of wide rows: logs, plans).
    pub compact_events_memory_limit: String,
    pub writer_threads: i64,
    /// Days after which a closed day is compacted (0 = as soon as the day ends + grace).
    pub compact_grace_s: u64,
    /// Seconds a replaced file stays on disk so in-flight readers can finish.
    pub delete_grace_s: u64,
    /// Physical samples layout variant (see README): "v2_split" (default) etc.
    pub samples_layout: String,
    pub zstd_level: i32,
    pub row_group_size: u64,
    /// Compaction sorts at most this many sample rows at once: a bigger day is written as
    /// several files split by series_id range (bounded memory; still <= 10 files/day).
    pub compact_part_rows: u64,
    /// Merge sample days with the streaming k-way merge (bounded memory) instead of a DuckDB sort.
    pub compact_kmerge: bool,
    /// Flush as soon as this many WAL bytes are buffered (bursts, catch-up, WAL replay).
    pub max_buffer_bytes: usize,
    /// Max bytes per DuckDB write batch inside a flush (one Parquet file per batch and day).
    pub write_chunk_bytes: usize,
    /// Write path of a store whose `store_dir` is `s3://bucket/prefix` (bucket.rs).
    pub s3: crate::bucket::BucketCfg,
}
impl Default for StoreCfg {
    fn default() -> Self {
        Self {
            flush_interval_s: 60,
            closed_lag_ms: 5000,
            writer_memory_limit: "64MB".into(),
            compact_memory_limit: "32MB".into(),
            compact_events_memory_limit: "96MB".into(),
            writer_threads: 1,
            compact_grace_s: 300,
            delete_grace_s: 300,
            samples_layout: crate::store::DEFAULT_LAYOUT.into(),
            zstd_level: 9,
            row_group_size: 122_880,
            compact_part_rows: 3_500_000,
            compact_kmerge: true,
            max_buffer_bytes: 32 << 20,
            write_chunk_bytes: 8 << 20,
            s3: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct HttpCfg {
    pub listen: String,
    pub memory_limit: String,
    /// Reader DuckDB threads. 1 (the default): the thread that runs a query executes its plan
    /// itself, and concurrency comes from the query slots (max_concurrent). More threads add
    /// DuckDB background workers, and the caller then polls for the result in a busy loop that
    /// took half the CPU of a dashboard refresh of table-function queries (D2 profile); heavy
    /// ad-hoc SQL may still gain from them. 0 = max_concurrent.
    pub threads: i64,
    pub max_concurrent: usize,
    pub timeout_ms: u64,
    pub max_rows: usize,
    pub max_bytes: usize,
    pub max_temp_directory_size: String,
    /// Admission: queries beyond max_concurrent wait in a bounded queue (FIFO) for up to
    /// queue_wait_ms; a full queue or a longer wait answers `busy`.
    pub queue_max: usize,
    pub queue_wait_ms: u64,
    /// Run SQL in a child process (true) so no query can take the collector down.
    pub reader_process: bool,
    /// Child RSS above this is killed by the watchdog (typed out_of_memory, child restarted).
    pub reader_rss_cap_mb: u64,
    /// Linux RLIMIT_DATA of the child (0 = none): allocations past it fail in the child.
    pub reader_data_limit_mb: u64,
    /// Drain the reader once its RSS has passed this % of the cap during a query: it takes no
    /// new requests, finishes the ones in flight, and is replaced. The next query then starts on
    /// a fresh reader instead of inheriting a near-cap one and being killed for someone else.
    pub reader_drain_pct: u64,
    /// Admission (admission.rs): a query expected to finish under this many ms is short and
    /// may use the slot kept for short queries.
    pub short_query_ms: u64,
    /// A heavy query starts only while the reader's RSS is under this % of reader_rss_cap_mb
    /// (or when no other heavy query runs). 0 = no gate.
    pub heavy_mem_gate_pct: u64,
    /// Parquet footer cache in the reader (eval/footer.rs), MB; 0 = off.
    pub footer_cache_mb: u64,
    /// Evaluated-row cache in the reader (eval/rowcache.rs), MB; 0 = off.
    pub row_cache_mb: u64,
    /// L1 result cache in the parent (qcache.rs), MB of compressed bodies; 0 = off.
    pub cache_mb: u64,
    /// Largest compressed body L1 stores, KB.
    pub cache_max_entry_kb: u64,
    /// Evaluator threads in all, shared by the running queries: a query with a large
    /// selection borrows the free ones (eval/par.rs). 0 = max_concurrent; 1 (default) = off.
    pub eval_workers: usize,
    /// Memory one evaluator worker may add per day its scan spans, MB (plus 4 MB): workers are
    /// lent only while the reader's RSS plus theirs stays under 65% of its cap (PAR_MEM_PCT).
    pub eval_worker_mb_per_day: u64,
    /// Object-store read path, for a store root `s3://bucket/prefix` (objstore.rs).
    pub s3: crate::objstore::S3Cfg,
}

impl HttpCfg {
    /// Fill the automatic values: max_concurrent 0 = one slot per core, at most 4;
    /// threads 0 = max_concurrent.
    pub fn resolve(mut self) -> Self {
        if self.max_concurrent == 0 {
            let cores = std::thread::available_parallelism().map_or(2, |n| n.get());
            self.max_concurrent = cores.clamp(1, 4);
        }
        if self.threads <= 0 {
            self.threads = self.max_concurrent as i64;
        }
        if self.eval_workers == 0 {
            self.eval_workers = self.max_concurrent;
        }
        self
    }
}
impl Default for HttpCfg {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:8765".into(),
            memory_limit: "96MB".into(),
            threads: 1,
            max_concurrent: 0,
            timeout_ms: 15_000,
            max_rows: 100_000,
            max_bytes: 16 << 20,
            max_temp_directory_size: "2GB".into(),
            queue_max: 64,
            queue_wait_ms: 30_000,
            reader_process: true,
            reader_rss_cap_mb: 192,
            reader_data_limit_mb: 1024,
            reader_drain_pct: 70,
            short_query_ms: 50,
            heavy_mem_gate_pct: 50,
            footer_cache_mb: 16,
            row_cache_mb: 24,
            cache_mb: 16,
            cache_max_entry_kb: 2048,
            eval_workers: 1,
            eval_worker_mb_per_day: 3,
            s3: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CollectorCfg {
    /// Unique series a single metric query may produce per run; the rest is cut and counted.
    pub max_series_per_query: usize,
    /// Max sessions to one Postgres node across all databases.
    pub max_sessions_per_node: usize,
    pub default_statement_timeout_s: u64,
    /// pgwatch's Prometheus sink drops NUMERIC columns (pgx decodes them as pgtype.Numeric,
    /// which the sink skips). Keep that behaviour for label/series parity.
    pub drop_numeric: bool,
    /// Seconds between pg_stat_statements text snapshots.
    pub query_texts_interval_s: u64,
    /// Seconds between object-definition snapshots.
    pub objects_interval_s: u64,
    /// Seconds between stats_reset snapshots (pgai_mon_stats_reset_epoch_s).
    pub stats_reset_interval_s: u64,
    /// pgwatch's sink turns non-`tag_` text columns into labels too (seen in VictoriaMetrics:
    /// settings.unit_normalized, pg_not_valid_constraints.constraint_definition). That is
    /// the default here, for parity. This checked-in classification overrides it per
    /// metric/column: "label" | "drop". Query text must never be a label: list such columns
    /// as "drop" (they belong in query_texts).
    pub text_columns: BTreeMap<String, BTreeMap<String, String>>,
}
impl Default for CollectorCfg {
    fn default() -> Self {
        Self {
            max_series_per_query: 20_000,
            max_sessions_per_node: 4,
            default_statement_timeout_s: 15,
            drop_numeric: true,
            query_texts_interval_s: 300,
            objects_interval_s: 600,
            stats_reset_interval_s: 60,
            text_columns: BTreeMap::new(),
        }
    }
}

/// Environment variable that selects the store, overriding `store_dir` (compose and Terraform
/// set it: `s3://bucket/prefix`, or unset for the local volume).
pub const STORE_ENV: &str = "PGAI_MON_STORE";

impl Config {
    /// Read the YAML file, then apply `PGAI_MON_STORE` (`with_store_env`).
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let c: Config = serde_yaml::from_str(&s)?;
        Ok(c.with_store_env(std::env::var(STORE_ENV).ok()))
    }

    /// `env` (the value of `PGAI_MON_STORE`), when set and not empty, replaces `store_dir`. An
    /// object-store `store_dir` without `store.s3.local_dir` works in `bucket::DEFAULT_LOCAL_DIR`.
    pub fn with_store_env(mut self, env: Option<String>) -> Self {
        if let Some(v) = env.map(|v| v.trim().to_string()).filter(|v| !v.is_empty()) {
            self.store_dir = PathBuf::from(v);
        }
        if crate::objstore::is_remote(&self.store_dir) && self.store.s3.local_dir.is_none() {
            self.store.s3.local_dir = Some(PathBuf::from(crate::bucket::DEFAULT_LOCAL_DIR));
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(yaml: &str, env: Option<&str>) -> Config {
        serde_yaml::from_str::<Config>(yaml).unwrap().with_store_env(env.map(str::to_string))
    }

    #[test]
    fn store_env_overrides_store_dir() {
        let y = "store_dir: /data/local\n";
        let c = cfg(y, None);
        assert_eq!(c.store_dir, PathBuf::from("/data/local"));
        assert_eq!(c.store.s3.local_dir, None);
        assert_eq!(cfg(y, Some("")).store_dir, PathBuf::from("/data/local"));
        let c = cfg(y, Some("s3://bkt/mon"));
        assert_eq!(c.store_dir, PathBuf::from("s3://bkt/mon"));
        assert_eq!(c.store.s3.local_dir, Some(PathBuf::from("/var/lib/pgai-mon")));
        let c = cfg("store_dir: /x\nstore:\n  s3:\n    local_dir: /wd\n", Some("s3://bkt/mon"));
        assert_eq!(c.store.s3.local_dir, Some(PathBuf::from("/wd")));
        let c = cfg("store_dir: s3://a/b\n", None);
        assert_eq!(c.store.s3.local_dir, Some(PathBuf::from("/var/lib/pgai-mon")));
        assert_eq!(cfg("store_dir: s3://a/b\n", Some("/local")).store_dir, PathBuf::from("/local"));
    }
}
