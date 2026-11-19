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
    /// pgwatch sources.yml / the CLI's instances.yml, read as is (`~sink_type~` becomes
    /// `prometheus`); re-read with the config on SIGHUP and every collector.reload_interval_s.
    #[serde(default)]
    pub sources_file: Option<PathBuf>,
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
    /// The file this config was loaded from (re-read on reload).
    #[serde(skip)]
    pub config_path: Option<PathBuf>,
    /// Externally produced samples: push endpoint limits (ingest.rs).
    #[serde(default)]
    pub ingest: crate::ingest::IngestCfg,
    /// Prometheus scrape targets (scrape.rs): files such as host-metrics/*.yml, inline jobs.
    #[serde(default)]
    pub scrape: crate::scrape::ScrapeCfg,
    /// Recording rules materialised at ingest (rules.rs): host_* from scraped provider series.
    #[serde(default)]
    pub rules: crate::rules::RulesCfg,
    /// RDS/Aurora instances whose host stats are polled from AWS (hoststats/). Empty: the
    /// rds-host-stats environment (RDS_DB_INSTANCE_IDENTIFIER, ...) when it is set.
    #[serde(default)]
    pub rds: Vec<crate::hoststats::RdsTarget>,
}

fn d_metrics_yml() -> PathBuf {
    PathBuf::from("mon/metrics.yml")
}
fn d_preset() -> String {
    "full".into()
}

/// One monitored node. The native fields plus everything pgwatch's sources.yml (and the
/// instances.yml the CLI writes, cli/lib/instances.ts) carries, so that file loads as is.
/// Unknown fields are reported and ignored.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Source {
    /// pgwatch source name: becomes the `dbname` label, exactly as pgwatch does.
    pub name: String,
    #[serde(default)]
    pub conn_str: String,
    /// Labels added to every series (pgwatch `custom_tags`). Non-string YAML values are
    /// written as text; null values are left out.
    #[serde(default, deserialize_with = "de_tags")]
    pub custom_tags: BTreeMap<String, String>,
    /// Also collect per-database metrics on every other connectable, non-template
    /// database of the node (pgwatch "continuous discovery" naming: `<name>_<datname>`).
    /// `kind: postgres-continuous-discovery` sets it too.
    #[serde(default)]
    pub discover_databases: bool,
    /// Labels pgwatch never sees but VictoriaMetrics adds at scrape time (job, instance).
    /// Kept configurable so live series can be made label-identical to the reference.
    /// Empty: collect_sources sets the default (DEFAULT_SCRAPE_JOB, instance = name).
    #[serde(default)]
    pub scrape_labels: BTreeMap<String, String>,
    /// pgwatch: preset from metrics.yml; wins over custom_metrics. Neither: the global preset.
    #[serde(default)]
    pub preset_metrics: Option<String>,
    /// pgwatch: metric -> interval (s). Used when preset_metrics is empty.
    #[serde(default)]
    pub custom_metrics: Option<BTreeMap<String, f64>>,
    /// pgwatch: the metrics a standby runs instead (only when this resolves to a non-empty list).
    #[serde(default)]
    pub preset_metrics_standby: Option<String>,
    #[serde(default)]
    pub custom_metrics_standby: Option<BTreeMap<String, f64>>,
    /// false: not collected. Default true (pgwatch's file defaults it to false, but the CLI
    /// always writes it, and native configs never had it).
    #[serde(default = "d_true")]
    pub is_enabled: bool,
    /// pgwatch source group; informational.
    #[serde(default)]
    pub group: Option<String>,
    /// pgwatch source kind: postgres (default) or postgres-continuous-discovery. Others
    /// (pgbouncer, pgpool, patroni*) are not supported: such a source is skipped.
    #[serde(default)]
    pub kind: Option<String>,
    /// Discovery filters on database names (regular expressions, pgwatch semantics).
    #[serde(default)]
    pub include_pattern: Option<String>,
    #[serde(default)]
    pub exclude_pattern: Option<String>,
    /// pgwatch: collect only while the node is a primary.
    #[serde(default)]
    pub only_if_master: bool,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_yaml::Value>,
}

fn d_true() -> bool {
    true
}

fn de_tags<'de, D: serde::Deserializer<'de>>(d: D) -> Result<BTreeMap<String, String>, D::Error> {
    let m: Option<BTreeMap<String, serde_yaml::Value>> = Deserialize::deserialize(d)?;
    let mut out = BTreeMap::new();
    for (k, v) in m.unwrap_or_default() {
        let s = match v {
            serde_yaml::Value::Null => continue,
            serde_yaml::Value::String(s) => s,
            serde_yaml::Value::Bool(b) => b.to_string(),
            serde_yaml::Value::Number(n) => n.to_string(),
            other => serde_yaml::to_string(&other).unwrap_or_default().trim().to_string(),
        };
        out.insert(k, s);
    }
    Ok(out)
}

/// The value the old stack's generate-pgwatch-sources.sh substituted for `~sink_type~` in
/// instances.yml for the Prometheus pgwatch, whose series pgai-mon reproduces.
pub const SINK_TYPE: &str = "prometheus";

impl Source {
    /// Problems that make this source unusable (it is skipped with the message), and
    /// warnings about fields that are ignored.
    pub fn check(&self) -> (Option<String>, Vec<String>) {
        let warn: Vec<String> = self
            .unknown
            .keys()
            .map(|k| format!("source {}: unknown field {k} ignored", self.name))
            .collect();
        if self.name.is_empty() {
            return (Some("source without a name".into()), warn);
        }
        if self.conn_str.trim().is_empty() {
            return (Some(format!("source {}: no conn_str", self.name)), warn);
        }
        match self.kind.as_deref() {
            None | Some("") | Some("postgres") | Some("postgres-continuous-discovery") => {}
            Some(k) => return (Some(format!("source {}: kind {k} is not supported", self.name)), warn),
        }
        for p in [&self.include_pattern, &self.exclude_pattern].into_iter().flatten() {
            if let Err(e) = regex::Regex::new(p) {
                return (Some(format!("source {}: bad pattern {p}: {e}", self.name)), warn);
            }
        }
        if let Err(e) = crate::pgconn::ConnParams::parse(&self.conn_str) {
            return (Some(format!("source {}: conn_str: {e}", self.name)), warn);
        }
        (None, warn)
    }

    pub fn discovers(&self) -> bool {
        self.discover_databases || self.kind.as_deref() == Some("postgres-continuous-discovery")
    }

    /// Whether a discovered database is collected (include_pattern / exclude_pattern).
    pub fn wants_db(&self, datname: &str) -> bool {
        let m = |p: &Option<String>| {
            p.as_deref()
                .filter(|p| !p.is_empty())
                .map(|p| regex::Regex::new(p).map(|r| r.is_match(datname)).unwrap_or(false))
        };
        m(&self.include_pattern).unwrap_or(true) && !m(&self.exclude_pattern).unwrap_or(false)
    }
}

/// Parse a pgwatch sources.yml / CLI instances.yml (a YAML list; empty or null: none).
pub fn parse_sources_file(text: &str) -> anyhow::Result<Vec<Source>> {
    let text = text.replace("~sink_type~", SINK_TYPE);
    let v: Option<Vec<Source>> = serde_yaml::from_str(&text).map_err(|e| {
        // serde_yaml messages quote the offending value (conn_str with a password): keep
        // only the position.
        match e.location() {
            Some(l) => anyhow::anyhow!("sources file: invalid YAML at line {} column {}", l.line(), l.column()),
            None => anyhow::anyhow!("sources file: invalid YAML"),
        }
    })?;
    Ok(v.unwrap_or_default())
}

/// The `job` a source without `scrape_labels` gets: the one the dashboards query
/// (`up{job="pgwatch-prometheus"}`, the Active session history divisor).
pub const DEFAULT_SCRAPE_JOB: &str = "pgwatch-prometheus";

/// Every source to collect: the config's own `sources` plus `sources_file`. Unusable and
/// disabled sources are left out (reported); a duplicate name keeps the first. A source
/// without `scrape_labels` gets `{job: DEFAULT_SCRAPE_JOB, instance: <name>}`; `instance`
/// is the name, unique per source, because each source writes its own `up`.
pub fn collect_sources(cfg: &Config) -> anyhow::Result<Vec<Source>> {
    let mut all = cfg.sources.clone();
    if let Some(f) = &cfg.sources_file {
        let text = match std::fs::read_to_string(f) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => anyhow::bail!("sources file {}: {e}", f.display()),
        };
        all.extend(parse_sources_file(&text)?);
    }
    let mut out: Vec<Source> = Vec::new();
    for mut s in all {
        let (bad, warns) = s.check();
        for w in warns {
            eprintln!("collector: {w}");
        }
        if let Some(b) = bad {
            eprintln!("collector: {b}; skipped");
            continue;
        }
        if !s.is_enabled {
            continue;
        }
        if out.iter().any(|o| o.name == s.name) {
            eprintln!("collector: source {} listed twice; the first is used", s.name);
            continue;
        }
        if s.scrape_labels.is_empty() {
            s.scrape_labels = BTreeMap::from([
                ("job".to_string(), DEFAULT_SCRAPE_JOB.to_string()),
                ("instance".to_string(), s.name.clone()),
            ]);
        }
        out.push(s);
    }
    Ok(out)
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
    pub zstd_level: i32,
    pub row_group_size: u64,
    /// Flush as soon as this many WAL bytes are buffered (bursts, catch-up, WAL replay).
    pub max_buffer_bytes: usize,
    /// Max bytes per DuckDB write batch inside a flush (one Parquet file per batch and day).
    pub write_chunk_bytes: usize,
    /// Days of metrics kept (samples, series, rollups, logs, plan events); 0 keeps everything.
    /// A file leaves the store once its last day is older (month files: once the month is).
    pub retention_days: u32,
    /// Days of query texts kept; 0 keeps everything.
    pub texts_retention_days: u32,
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
            zstd_level: 9,
            row_group_size: 122_880,
            max_buffer_bytes: 32 << 20,
            write_chunk_bytes: 8 << 20,
            retention_days: 0,
            texts_retention_days: 0,
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
    /// Encoded bytes for chart formats (objects and csv); oversized charts fail with 413.
    pub chart_max_bytes: usize,
    pub max_temp_directory_size: String,
    /// Admission: queries beyond max_concurrent wait in a bounded queue (FIFO) for up to
    /// queue_wait_ms; a full queue or a longer wait answers `busy`.
    pub queue_max: usize,
    pub queue_wait_ms: u64,
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
    /// Credentials every route but `/livez` requires (auth.rs).
    pub auth: crate::auth::AuthCfg,
}

impl HttpCfg {
    /// Fill the automatic values: max_concurrent 0 = one slot per core, at most 4;
    /// threads 0 = max_concurrent.
    pub fn resolve(mut self) -> Self {
        if self.max_concurrent == 0 {
            // On Linux this includes cgroup v2 cpu.max and affinity, not just host cores.
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
            chart_max_bytes: 5 << 20,
            max_temp_directory_size: "2GB".into(),
            queue_max: 64,
            queue_wait_ms: 30_000,
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
            auth: Default::default(),
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
    /// lock_timeout of every collector session, ms (pgwatch: SET LOCAL lock_timeout '100ms').
    /// A catalog query queued behind DDL gives up instead of blocking the DDL's waiters.
    pub lock_timeout_ms: u64,
    /// search_path of every collector session. None: pg_catalog, the schemas of installed
    /// extensions (pg_stat_statements may live outside public), then public; no "$user".
    pub search_path: Option<String>,
    /// Rows one metric query may return; the rest is not fetched (portal with a row limit)
    /// and the cut is counted in pgai_mon_rows_cut_total.
    pub max_rows_per_query: usize,
    /// statement_timeout and row limit of the object-definition snapshot (catalog-wide).
    pub objects_statement_timeout_s: u64,
    pub objects_max_rows: usize,
    /// Reconnect backoff after a failed connection: doubles from min to max, with jitter.
    pub reconnect_min_s: u64,
    pub reconnect_max_s: u64,
    /// Lock-timeout retry backoff: doubles from min to max, with jitter.
    pub timeout_retry_min_s: u64,
    pub timeout_retry_max_s: u64,
    /// Seconds between re-reads of the config, sources file and metrics.yml (also on SIGHUP);
    /// changed sources restart, removed ones stop. 0 = SIGHUP only.
    pub reload_interval_s: u64,
    /// Seconds between database discovery runs (sources with discovery).
    pub discovery_interval_s: u64,
    /// Seconds between primary/standby (and version) re-checks.
    pub role_check_interval_s: u64,
    /// Seconds between writes of the collector's own series (pgai_mon_source_up, ...).
    pub self_metrics_interval_s: u64,
}
impl Default for CollectorCfg {
    fn default() -> Self {
        Self {
            max_series_per_query: 20_000,
            max_sessions_per_node: 4,
            default_statement_timeout_s: 15,
            lock_timeout_ms: 100,
            search_path: None,
            max_rows_per_query: 100_000,
            objects_statement_timeout_s: 60,
            objects_max_rows: 500_000,
            reconnect_min_s: 2,
            reconnect_max_s: 120,
            timeout_retry_min_s: 30,
            timeout_retry_max_s: 600,
            reload_interval_s: 30,
            discovery_interval_s: 120,
            role_check_interval_s: 30,
            self_metrics_interval_s: 15,
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
/// Environment variables that set the retention, overriding `store.retention_days` and
/// `store.texts_retention_days` (the compose image's config has 14 and 30 days).
pub const RETENTION_ENV: &str = "PGAI_MON_RETENTION_DAYS";
pub const TEXTS_RETENTION_ENV: &str = "PGAI_MON_TEXTS_RETENTION_DAYS";
/// The old stack's retention variables (VictoriaMetrics' period, the query text sink's hours).
pub const LEGACY_RETENTION_ENV: &str = "VM_RETENTION_PERIOD";
pub const LEGACY_TEXTS_RETENTION_ENV: &str = "QUERYID_RETENTION_HOURS";

/// A VictoriaMetrics `-retentionPeriod` value in days, rounded up: a number with the unit
/// s, h, d, w or y, or without a unit for months (31 days).
pub fn vm_period_days(v: &str) -> Option<u32> {
    let i = v.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(v.len());
    let n: f64 = v[..i].parse().ok()?;
    let hours = n * match &v[i..] {
        "s" => 1.0 / 3600.0,
        "h" => 1.0,
        "d" => 24.0,
        "w" => 168.0,
        "y" => 8760.0,
        "" => 744.0,
        _ => return None,
    };
    (hours > 0.0).then(|| (hours / 24.0).ceil() as u32)
}

impl Config {
    /// Read the YAML file, then apply `PGAI_MON_STORE` (`with_store_env`) and the retention
    /// variables (`with_retention_env`).
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("config {}: {e}", path.display()))?;
        let mut c: Self = serde_yaml::from_str(&s).map_err(|e| anyhow::anyhow!("config {}: {e}", path.display()))?;
        c.config_path = Some(path.to_path_buf());
        c.with_store_env(std::env::var(STORE_ENV).ok())
            .with_retention_env(|name| std::env::var(name).ok())
    }

    /// Retention from the environment (`env` looks a variable up), replacing
    /// `store.retention_days` and `store.texts_retention_days`. `PGAI_MON_RETENTION_DAYS` and
    /// `PGAI_MON_TEXTS_RETENTION_DAYS` win; a value that is not a whole number of days is an
    /// error (a typo must not silently keep or delete data). Without them, the old stack's
    /// `VM_RETENTION_PERIOD` and `QUERYID_RETENTION_HOURS` apply, rounded up to days, so an
    /// upgraded install keeps the retention its operator chose. Empty values count as unset.
    pub fn with_retention_env(mut self, env: impl Fn(&str) -> Option<String>) -> anyhow::Result<Self> {
        let get = |name: &str| env(name).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let days = |name: &str| -> anyhow::Result<Option<u32>> {
            get(name)
                .map(|v| {
                    v.parse().map_err(|_| {
                        anyhow::anyhow!("{name}={v:?}: expected a whole number of days (0 keeps everything)")
                    })
                })
                .transpose()
        };
        let legacy = |name: &str, to_days: fn(&str) -> Option<u32>| {
            let v = get(name)?;
            let d = to_days(&v);
            match d {
                Some(d) => eprintln!("pgai-mon: {name}={v} (old stack) taken as a retention of {d} days"),
                None => eprintln!("pgai-mon: {name}={v} (old stack) is not understood and is ignored"),
            }
            d
        };
        if let Some(d) = days(RETENTION_ENV)?.or_else(|| legacy(LEGACY_RETENTION_ENV, vm_period_days)) {
            self.store.retention_days = d;
        }
        let hours_days = |v: &str| v.parse::<u32>().ok().filter(|h| *h > 0).map(|h| h.div_ceil(24));
        if let Some(d) = days(TEXTS_RETENTION_ENV)?.or_else(|| legacy(LEGACY_TEXTS_RETENTION_ENV, hours_days)) {
            self.store.texts_retention_days = d;
        }
        Ok(self)
    }

    /// Whether the collector has anything to run (inline sources or a sources file).
    pub fn has_sources(&self) -> bool {
        !self.sources.is_empty() || self.sources_file.is_some()
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

    #[test]
    fn timeout_retry_defaults_and_overrides() {
        let cfg: CollectorCfg = serde_yaml::from_str("{}").unwrap();
        assert_eq!((cfg.timeout_retry_min_s, cfg.timeout_retry_max_s), (30, 600));
        let cfg: CollectorCfg = serde_yaml::from_str("timeout_retry_min_s: 1\ntimeout_retry_max_s: 2").unwrap();
        assert_eq!((cfg.timeout_retry_min_s, cfg.timeout_retry_max_s), (1, 2));
    }

    #[test]
    fn retention_variables_override_the_file() {
        let load = |vars: &[(&str, &str)]| {
            let vars: Vec<(String, String)> = vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            let c: Config = serde_yaml::from_str("store_dir: /s\nstore: {retention_days: 7}").unwrap();
            c.with_retention_env(|name| vars.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone()))
                .map(|c| (c.store.retention_days, c.store.texts_retention_days))
        };
        assert_eq!(load(&[]).unwrap(), (7, 0));
        assert_eq!(
            load(&[(RETENTION_ENV, " 14 "), (TEXTS_RETENTION_ENV, "30")]).unwrap(),
            (14, 30)
        );
        // empty (compose passes unset variables as empty) changes nothing; 0 keeps everything
        assert_eq!(
            load(&[(RETENTION_ENV, ""), (TEXTS_RETENTION_ENV, "0")]).unwrap(),
            (7, 0)
        );
        let e = load(&[(RETENTION_ENV, "14d")]).unwrap_err();
        assert!(e.to_string().contains("PGAI_MON_RETENTION_DAYS"), "{e}");
        // the old stack's variables apply when the new ones are unset, rounded up to days
        let old = [(LEGACY_RETENTION_ENV, "336h"), (LEGACY_TEXTS_RETENTION_ENV, "9600")];
        assert_eq!(load(&old).unwrap(), (14, 400));
        assert_eq!(load(&[old[0], old[1], (RETENTION_ENV, "3")]).unwrap(), (3, 400));
        assert_eq!(
            load(&[(LEGACY_RETENTION_ENV, "13months")]).unwrap(),
            (7, 0),
            "not understood"
        );
        let days: Vec<_> = ["1", "2w", "1y", "25h", "1.5d", "0", "h"]
            .iter()
            .map(|v| vm_period_days(v))
            .collect();
        assert_eq!(days, [Some(31), Some(14), Some(365), Some(2), Some(2), None, None]);
    }

    fn cfg(yaml: &str, env: Option<&str>) -> Config {
        serde_yaml::from_str::<Config>(yaml)
            .unwrap()
            .with_store_env(env.map(str::to_string))
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
        let c = cfg(
            "store_dir: /x\nstore:\n  s3:\n    local_dir: /wd\n",
            Some("s3://bkt/mon"),
        );
        assert_eq!(c.store.s3.local_dir, Some(PathBuf::from("/wd")));
        let c = cfg("store_dir: s3://a/b\n", None);
        assert_eq!(c.store.s3.local_dir, Some(PathBuf::from("/var/lib/pgai-mon")));
        assert_eq!(
            cfg("store_dir: s3://a/b\n", Some("/local")).store_dir,
            PathBuf::from("/local")
        );
    }

    /// The file `pgai mon local-install --demo` writes (instances.demo.yml), as is.
    const DEMO: &str = r#"
# Demo PostgreSQL instance
- name: target_database
  conn_str: postgresql://monitor:monitor_pass@target-db:5432/target_database
  preset_metrics: full
  custom_metrics:
  is_enabled: true
  group: default
  custom_tags:
    env: demo
    cluster: local
    node_name: node-01
    sink_type: ~sink_type~  # substituted by generate-pgwatch-sources.sh
"#;

    #[test]
    fn instances_yml_from_the_cli_loads_as_is() {
        let v = parse_sources_file(DEMO).unwrap();
        assert_eq!(v.len(), 1);
        let s = &v[0];
        assert_eq!(s.name, "target_database");
        assert_eq!(s.preset_metrics.as_deref(), Some("full"));
        assert_eq!(s.custom_metrics, None);
        assert!(s.is_enabled);
        assert_eq!(s.group.as_deref(), Some("default"));
        assert_eq!(s.custom_tags["sink_type"], "prometheus");
        assert_eq!(s.custom_tags["node_name"], "node-01");
        assert!(s.unknown.is_empty());
        assert_eq!(s.check(), (None, vec![]));
    }

    #[test]
    fn sources_file_fields() {
        let v = parse_sources_file(
            r#"
- name: a
  conn_str: host=h dbname=d
  custom_metrics: {wait_events: 15, db_size: 300.0, off: 0}
  custom_tags: {port: 5432, primary: true, nothing: null}
  kind: postgres-continuous-discovery
  include_pattern: ^app
  exclude_pattern: _tmp$
  only_if_master: true
  host_config: {x: 1}
  something_new: 1
- name: b
  conn_str: host=h
  is_enabled: false
- name: c
  conn_str: host=h
  kind: pgbouncer
- name: d
"#,
        )
        .unwrap();
        let a = &v[0];
        assert_eq!(a.custom_metrics.as_ref().unwrap()["db_size"], 300.0);
        assert_eq!(a.custom_tags.get("port").map(|s| s.as_str()), Some("5432"));
        assert_eq!(a.custom_tags.get("primary").map(|s| s.as_str()), Some("true"));
        assert!(!a.custom_tags.contains_key("nothing"));
        assert!(a.discovers() && a.only_if_master);
        assert!(a.wants_db("app1") && !a.wants_db("app_tmp") && !a.wants_db("other"));
        let (bad, warn) = a.check();
        assert_eq!(bad, None);
        assert_eq!(warn.len(), 2, "{warn:?}"); // host_config, something_new
        assert!(v[2].check().0.unwrap().contains("not supported"));
        assert!(v[3].check().0.unwrap().contains("no conn_str"));

        let cfg = Config {
            sources_file: None,
            sources: v,
            ..serde_yaml::from_str("store_dir: /x").unwrap()
        };
        let names: Vec<String> = collect_sources(&cfg).unwrap().into_iter().map(|s| s.name).collect();
        assert_eq!(
            names,
            vec!["a"],
            "disabled, unsupported and incomplete sources are left out"
        );
    }

    #[test]
    fn empty_and_broken_sources_files() {
        assert!(parse_sources_file("").unwrap().is_empty());
        assert!(parse_sources_file("[]").unwrap().is_empty());
        assert!(parse_sources_file("# nothing\n").unwrap().is_empty());
        let e = parse_sources_file("- name: x\n  conn_str: postgresql://u:secretpw@h/d\n  custom_tags: [").unwrap_err();
        assert!(!e.to_string().contains("secretpw"), "{e}");
    }

    #[test]
    fn native_config_and_sources_file_together() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("instances.yml");
        std::fs::write(&f, DEMO).unwrap();
        let cfg_path = dir.path().join("pgai-mon.yml");
        std::fs::write(
            &cfg_path,
            format!(
                "store_dir: /x\nsources_file: {}\nsources:\n  - name: native\n    conn_str: host=h\n    scrape_labels: {{job: j}}\n  - name: target_database\n    conn_str: host=dup\n",
                f.display()
            ),
        )
        .unwrap();
        let cfg = Config::load(&cfg_path).unwrap();
        assert_eq!(cfg.config_path.as_deref(), Some(cfg_path.as_path()));
        let s = collect_sources(&cfg).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].name, "native");
        assert_eq!(s[1].conn_str, "host=dup", "the first of two same-named sources wins");
        // explicit scrape_labels are kept as they are; none: the default job, instance = name
        let labels = |i: usize| s[i].scrape_labels.clone().into_iter().collect::<Vec<_>>();
        assert_eq!(labels(0), vec![("job".to_string(), "j".to_string())]);
        assert_eq!(
            labels(1),
            vec![
                ("instance".to_string(), "target_database".to_string()),
                ("job".to_string(), DEFAULT_SCRAPE_JOB.to_string())
            ]
        );
        // a missing sources file is no sources (the CLI writes it later)
        std::fs::remove_file(&f).unwrap();
        assert_eq!(collect_sources(&Config::load(&cfg_path).unwrap()).unwrap().len(), 2);
    }
}
