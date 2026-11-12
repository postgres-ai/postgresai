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
}

fn d_metrics_yml() -> PathBuf {
    PathBuf::from("config/pgwatch-prometheus/metrics.yml")
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

/// Every source to collect: the config's own `sources` plus `sources_file`. Unusable and
/// disabled sources are left out (reported); a duplicate name keeps the first.
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
    for s in all {
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

impl Config {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let mut c: Self = serde_yaml::from_str(&s)?;
        c.config_path = Some(path.to_path_buf());
        Ok(c)
    }

    /// Whether the collector has anything to run (inline sources or a sources file).
    pub fn has_sources(&self) -> bool {
        !self.sources.is_empty() || self.sources_file.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // a missing sources file is no sources (the CLI writes it later)
        std::fs::remove_file(&f).unwrap();
        assert_eq!(collect_sources(&Config::load(&cfg_path).unwrap()).unwrap().len(), 2);
    }
}
