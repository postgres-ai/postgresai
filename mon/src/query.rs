//! Read-only SQL over the store: a separate in-process DuckDB instance (its own memory limit,
//! thread count and temp space) so a bad query cannot starve ingest, with guardrails:
//! - only a single SELECT is accepted (checked by DuckDB's own parser via json_serialize_sql);
//! - enable_external_access=false, allowed_directories = the store, lock_configuration=true:
//!   no file/network table functions outside the store, no INSTALL/LOAD, no SET;
//! - an object store (`s3://`) never reaches DuckDB (no httpfs): the small kinds its views read
//!   are mirrored to local files by Rust (`Mirror`), allowed_directories lists only that
//!   mirror and the scratch dir, and the raw samples view raises (prom_* and pgai_samples read S3 in Rust).
//! - per-query timeout (interrupt), row and byte caps on the result, a concurrency cap;
//! - typed JSON errors.
use crate::catalog::FileEntry;
use crate::config::HttpCfg;
use crate::metricdefs::MetricsFile;
use crate::store::{empty_select, samples_raw_select, Shared, KINDS};
use duckdb::Connection;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

pub const EMBEDDED_VIEWS_SQL: &str = include_str!("../schema/views.sql");

/// `_raw_samples` on an object store: raw samples are never mirrored, so the `samples` and
/// `metrics` views fail with a pointer to what does work, where an empty answer would mislead.
const REMOTE_SAMPLES_ERROR: &str = "the samples and metrics views are not available on an object store: use \
     pgai_samples(metric, matchers, start_ms, end_ms) or the prom_* functions";

/// The body of a view that cannot be read: a query on it fails with `msg`, whatever it selects
/// or filters on. Every column raises, not only the filter: over constant NULL columns the
/// optimizer answers `WHERE ts > ...` with no rows and never runs the filter that raises.
fn failing_select(kind: &str, msg: &str) -> String {
    let e = format!("CASE WHEN i >= 0 THEN error('{}') END", msg.replace('\'', "''"));
    empty_select(kind)
        .replace("NULL::", &format!("{e}::"))
        .replace(" WHERE false", &format!(" FROM range(1) t(i) WHERE {e}::BOOLEAN"))
}

/// On an object store the SQL macros of sql/macros.sql read raw samples through the Rust scan.
const REMOTE_PROM_RAW: &str =
    "\n;\nCREATE OR REPLACE MACRO prom_raw(p_metric, p_matchers, p_fetch_start, p_end) AS TABLE \
     SELECT series_id, ts_ms, value AS v, stale FROM pgai_samples_rs(p_metric, CAST(to_json(p_matchers) AS VARCHAR), \
     CAST(p_fetch_start AS BIGINT), CAST(p_end AS BIGINT));\n";

#[derive(Debug, Serialize, Clone)]
pub struct QueryError {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub message: String,
}

impl QueryError {
    fn new(kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
    pub fn http_status(&self) -> u16 {
        match self.kind {
            "not_read_only" | "forbidden" => 403,
            "parse_error" | "binder_error" | "catalog_error" | "invalid_input" | "conversion_error" => 400,
            "timeout" => 408,
            "busy" => 429,
            "reader_restarted" => 503,
            "out_of_memory" | "result_too_large" => 413,
            _ => 500,
        }
    }
}

fn classify(e: &duckdb::Error) -> QueryError {
    let m = e.to_string();
    let kind = if m.contains("INTERRUPT") || m.contains("Interrupted") {
        "timeout"
    } else if m.starts_with("Out of Memory") || m.contains("Out of Memory Error") || m.contains("could not allocate") {
        "out_of_memory"
    } else if m.contains("Permission Error")
        || m.contains("disabled by configuration")
        || m.contains("enable_external_access")
    {
        "forbidden"
    } else if m.contains("Parser Error") {
        "parse_error"
    } else if m.contains("Binder Error") {
        "binder_error"
    } else if m.contains("Catalog Error") {
        "catalog_error"
    } else if m.contains("Conversion Error") {
        "conversion_error"
    } else if m.contains("Invalid Input Error") {
        "invalid_input"
    } else {
        "execution_error"
    };
    QueryError::new(kind, m)
}

/// Response format of `/sql` (`?format=`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fmt {
    /// `rows`: each row a JSON array (the default).
    Rows,
    /// `columnar`: one array (or dictionary) per column in `data`.
    Columnar,
    /// `objects`: each row a JSON object `{column: value}`, for clients that map fields by name
    /// (Grafana's Infinity datasource reads it with no per-column configuration).
    Objects,
    /// RFC 4180 CSV with a header, for chart clients.
    Csv,
}

impl Fmt {
    /// From the `format` parameter (None or "" is rows).
    pub fn parse(f: Option<&str>) -> Option<Fmt> {
        match f {
            None | Some("") | Some("rows") => Some(Fmt::Rows),
            Some("columnar") => Some(Fmt::Columnar),
            Some("objects") => Some(Fmt::Objects),
            Some("csv") => Some(Fmt::Csv),
            Some(_) => None,
        }
    }
    /// The child protocol's op for a query in this format.
    pub fn op(self) -> &'static str {
        match self {
            Fmt::Rows => "sql",
            Fmt::Columnar => "sql_columnar",
            Fmt::Objects => "sql_objects",
            Fmt::Csv => "sql_csv",
        }
    }
    pub fn from_op(op: &str) -> Fmt {
        match op {
            "sql_columnar" => Fmt::Columnar,
            "sql_objects" => Fmt::Objects,
            "sql_csv" => Fmt::Csv,
            _ => Fmt::Rows,
        }
    }
    /// Part of the result cache key.
    pub fn key(self) -> u8 {
        self as u8
    }
}

#[derive(Debug)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub types: Vec<String>,
    /// Each row is a JSON array (DuckDB's own JSON conversion of the values), or a JSON object
    /// keyed by column name in the objects format. Empty in the columnar format, which fills
    /// `data` instead.
    pub rows: Vec<Box<serde_json::value::RawValue>>,
    /// Response format the rows were built in.
    pub fmt: Fmt,
    /// Columnar format (`?format=columnar`): one entry per column, either a JSON array of its
    /// values or `{"dict":[distinct values],"idx":[index per row]}` when values repeat.
    pub data: Option<Box<serde_json::value::RawValue>>,
    /// Complete CSV body; JSON formats leave this empty.
    pub csv: Option<String>,
    pub row_count: usize,
    pub truncated: bool,
    pub elapsed_ms: u64,
    /// Catalog generation the views pointed at for this query (sent in the reply envelope,
    /// not in the body: the parent's result cache stores a body only if gen is current).
    pub gen: u64,
    /// Whether the parent's result cache may store this result (`Reader::cacheable`).
    pub cacheable: bool,
    pub range: Option<crate::readset::ReadRange>,
    pub dict_version: u64,
    /// Object-store reads during the query (on an object store).
    pub s3: Option<QueryS3>,
}

/// Object-store counters over one query: the process's counters before and after it, so they
/// include other queries' reads unless `exclusive` (no other query ran meanwhile).
#[derive(Debug, Clone, Serialize)]
pub struct QueryS3 {
    pub gets: u64,
    pub bytes: u64,
    pub demand_gets: u64,
    pub footer_gets: u64,
    pub exclusive: bool,
    /// Set when `gets` passed --s3-gets-warn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

/// Queries running in this reader, and queries started (for `QueryS3::exclusive`).
static ACTIVE: AtomicU64 = AtomicU64::new(0);
static STARTED: AtomicU64 = AtomicU64::new(0);

/// Row format: `{"columns","types","rows","row_count","truncated","elapsed_ms"}` (unchanged),
/// plus `"s3"` (QueryS3) on an object store.
/// Columnar: `{"columns","types","format":"columnar","data","row_count","truncated",
/// "elapsed_ms"}`. Objects: `{"columns","types","format":"objects","rows":[{column: value}],
/// "row_count","truncated","elapsed_ms"}`.
impl Serialize for QueryResult {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        // The child protocol carries CSV as a JSON string, decoded by the parent.
        if let Some(csv) = &self.csv {
            return ser.serialize_str(csv);
        }
        use serde::ser::SerializeStruct;
        let mut m = ser.serialize_struct("QueryResult", 7)?;
        m.serialize_field("columns", &self.columns)?;
        m.serialize_field("types", &self.types)?;
        match &self.data {
            Some(d) => {
                m.serialize_field("format", "columnar")?;
                m.serialize_field("data", d)?;
            }
            None => {
                if self.fmt == Fmt::Objects {
                    m.serialize_field("format", "objects")?;
                }
                m.serialize_field("rows", &self.rows)?
            }
        }
        m.serialize_field("row_count", &self.row_count)?;
        m.serialize_field("truncated", &self.truncated)?;
        m.serialize_field("elapsed_ms", &self.elapsed_ms)?;
        if let Some(s) = &self.s3 {
            m.serialize_field("s3", s)?;
        }
        m.end()
    }
}

/// Count the full JSON envelope without allocating another response body.
struct EncodedSize(usize);
impl std::io::Write for EncodedSize {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 += buf.len();
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// One column of a columnar result: distinct value texts and an index per row.
#[derive(Default)]
struct ColBuilder {
    ids: std::collections::HashMap<Box<str>, u32>,
    vals: Vec<Box<str>>,
    idx: Vec<u32>,
}

impl ColBuilder {
    /// Add a value's JSON text; returns the bytes it adds to the response (about).
    fn push(&mut self, v: &str) -> usize {
        if let Some(&i) = self.ids.get(v) {
            self.idx.push(i);
            return 4;
        }
        let i = self.vals.len() as u32;
        self.ids.insert(v.into(), i);
        self.vals.push(v.into());
        self.idx.push(i);
        v.len() + 5
    }
    /// Dictionary form when it is smaller: few distinct values over many rows.
    fn write(&self, out: &mut String) {
        let n = self.idx.len();
        if n >= 4 && self.vals.len() * 2 <= n {
            out.push_str("{\"dict\":[");
            for (k, v) in self.vals.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                out.push_str(v);
            }
            out.push_str("],\"idx\":[");
            for (k, i) in self.idx.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                let _ = std::fmt::Write::write_fmt(out, format_args!("{i}"));
            }
            out.push_str("]}");
        } else {
            out.push('[');
            for (k, i) in self.idx.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                out.push_str(&self.vals[*i as usize]);
            }
            out.push(']');
        }
    }
}

pub struct Reader {
    shared: Shared,
    admin: Mutex<Connection>,
    pool: Mutex<Vec<Connection>>,
    /// The catalog generation the views are at (u64::MAX: not created yet).
    view_gen: AtomicU64,
    /// Held by the one `refresh` that runs.
    refreshing: Mutex<()>,
    query_snapshot: RwLock<()>,
    pub cfg: HttpCfg,
    views_sql: String,
    pub eval: crate::eval::EvalHandle,
    dict: Mutex<crate::dictionary::Dictionary>,
    bounded_macros: std::sync::OnceLock<(std::collections::HashSet<String>, std::collections::HashSet<String>)>,
    /// Scalar functions whose result can change between runs of the same SQL at the same
    /// catalog generation (DuckDB's own stability metadata plus clock functions it labels
    /// CONSISTENT): a query calling one is not cached.
    volatile: std::collections::HashSet<String>,
    /// Remote store: where DuckDB's views read their local copies (None: a local store).
    mirror: Option<Mirror>,
}

/// Local copies, for DuckDB, of the files its views read on a remote store. Catalog paths are
/// immutable (new data gets new paths), so a copy of the right size is current.
struct Mirror {
    dir: std::path::PathBuf,
    cap_bytes: u64,
    /// For /health: files and bytes mirrored now, bytes fetched, kinds left empty, last error.
    stat: Mutex<serde_json::Value>,
}

/// The mirror path of catalog path `rel`, which must be relative and plain (a tampered
/// catalog.json must not write outside the mirror).
fn mirror_path(dir: &std::path::Path, rel: &str) -> anyhow::Result<std::path::PathBuf> {
    anyhow::ensure!(
        crate::catalog::plain_rel(rel),
        "catalog path {rel:?}: not a plain relative path"
    );
    Ok(dir.join(rel))
}

/// Delete mirror files not in `keep` and older than `min_age` (another process sharing the
/// mirror may still read a file of an older generation).
fn mirror_gc(dir: &std::path::Path, keep: &std::collections::HashSet<std::path::PathBuf>, min_age: Duration) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let Ok(m) = e.metadata() else { continue };
        if m.is_dir() {
            mirror_gc(&p, keep, min_age);
            let _ = std::fs::remove_dir(&p); // only when empty
        } else if !keep.contains(&p)
            && m.modified()
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|a| a > min_age)
        {
            let _ = std::fs::remove_file(&p);
        }
    }
}

fn sql_list(paths: &[String]) -> String {
    format!("[{}]", paths.iter().map(|p| sql_str(p)).collect::<Vec<_>>().join(","))
}

/// Table functions whose output depends only on their arguments and the catalog generation.
/// Any other (glob, read_*, duckdb_*, query, ...) can read state outside the catalog.
fn table_fn_cacheable(name: &str) -> bool {
    name.starts_with("prom_")
        || name.starts_with("pgai_")
        || matches!(name, "range" | "generate_series" | "unnest" | "unnest_rs")
}

/// Table functions a caller's SQL may name. DuckDB accepts `SELECT * FROM f()` as a read-only
/// SELECT for functions that change state or read arbitrary files (enable_logging writes log
/// files where it is told to, query() runs a string this check never sees, read_text and glob
/// expose every file under the store), so anything not listed is refused.
fn table_fn_allowed(name: &str) -> bool {
    table_fn_cacheable(name)
        || name.starts_with("parquet_")
        || matches!(
            name,
            "read_parquet"
                | "json_each"
                | "json_tree"
                | "duckdb_columns"
                | "duckdb_functions"
                | "duckdb_memory"
                | "duckdb_settings"
                | "duckdb_tables"
                | "duckdb_types"
                | "duckdb_views"
        )
}

/// The first table function in a json_serialize_sql tree that `table_fn_allowed` refuses.
fn refused_table_fn(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::Object(m) => {
            if m.get("type").and_then(|c| c.as_str()) == Some("TABLE_FUNCTION") {
                let f = m["function"]["function_name"]
                    .as_str()
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if !table_fn_allowed(&f) {
                    return Some(f);
                }
            }
            m.values().find_map(refused_table_fn)
        }
        serde_json::Value::Array(a) => a.iter().find_map(refused_table_fn),
        _ => None,
    }
}

/// Clock functions DuckDB 1.4 labels CONSISTENT although their value moves with time.
const CLOCK_FNS: &[&str] = &[
    "current_localtimestamp",
    "current_localtime",
    "localtimestamp",
    "localtime",
    "current_timestamp",
    "current_time",
    "current_date",
    "get_current_time",
    "get_current_timestamp",
    "today",
    "now",
    "transaction_timestamp",
    "epoch_ms_now",
    // one-argument age(ts) measures from now (the two-argument form is not cached either)
    "age",
];

/// Walk a json_serialize_sql tree: false when it calls a volatile scalar function, a table
/// function outside the allowlist, or scans a file by name (`FROM 'x.parquet'`).
fn ast_cacheable(v: &serde_json::Value, volatile: &std::collections::HashSet<String>) -> bool {
    match v {
        serde_json::Value::Object(m) => {
            match m.get("class").and_then(|c| c.as_str()) {
                Some("FUNCTION") => {
                    let f = m.get("function_name").and_then(|f| f.as_str()).unwrap_or("");
                    if volatile.contains(&f.to_ascii_lowercase()) {
                        return false;
                    }
                }
                // bare CURRENT_TIMESTAMP / CURRENT_DATE / LOCALTIME parse as column refs and
                // bind to functions later (a real column with such a name just isn't cached)
                Some("COLUMN_REF") => {
                    let names = m.get("column_names").and_then(|c| c.as_array());
                    if let Some([c]) = names.map(|a| a.as_slice()) {
                        if volatile.contains(&c.as_str().unwrap_or("").to_ascii_lowercase()) {
                            return false;
                        }
                    }
                }
                _ => {}
            }
            match m.get("type").and_then(|c| c.as_str()) {
                Some("TABLE_FUNCTION") => {
                    let f = m["function"]["function_name"].as_str().unwrap_or("");
                    if !table_fn_cacheable(&f.to_ascii_lowercase()) {
                        return false;
                    }
                }
                Some("BASE_TABLE") => {
                    let t = m.get("table_name").and_then(|t| t.as_str()).unwrap_or("");
                    if t.contains(['/', '.', '*', '\\']) {
                        return false;
                    }
                }
                _ => {}
            }
            // USING SAMPLE / TABLESAMPLE: a different sample each run
            if m.get("sample").is_some_and(|x| !x.is_null()) {
                return false;
            }
            m.values().all(|x| ast_cacheable(x, volatile))
        }
        serde_json::Value::Array(a) => a.iter().all(|x| ast_cacheable(x, volatile)),
        _ => true,
    }
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct DictEntry {
    pub group: String,
    pub prefix: String,
    pub interval_s: Option<u64>,
    pub instance_level: bool,
    pub gauges: Vec<String>,
    pub description: String,
}

/// The metric dictionary: every metric of metrics.yml, with the interval it is collected at
/// (`MetricsFile::dict_intervals`: the global preset and every source's lists) where it is.
pub fn build_dict(mf: Option<&MetricsFile>, intervals: &std::collections::BTreeMap<String, u64>) -> Vec<DictEntry> {
    let Some(mf) = mf else { return vec![] };
    mf.metrics
        .iter()
        .map(|(name, d)| DictEntry {
            group: name.clone(),
            prefix: format!("pgwatch_{}", d.storage_name(name)),
            interval_s: intervals.get(name).copied(),
            instance_level: d.is_instance_level,
            gauges: d.gauges.clone().unwrap_or_default(),
            description: d.description.clone().unwrap_or_default().trim().to_string(),
        })
        .collect()
}

/// DuckDB's JSON writes NaN/Infinity/-Infinity as bare tokens (invalid JSON). Rewrite them,
/// outside string literals, as the Prometheus API strings "NaN", "+Inf", "-Inf".
pub fn sanitize_non_finite(s: String) -> String {
    if !s.contains("NaN") && !s.contains("Infinity") {
        return s;
    }
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 16);
    let mut i = 0;
    let mut in_str = false;
    while i < b.len() {
        let c = b[i];
        if in_str {
            out.push(c as char);
            if c == b'\\' && i + 1 < b.len() {
                out.push(b[i + 1] as char);
                i += 2;
                continue;
            }
            if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if c == b'"' {
            in_str = true;
            out.push('"');
            i += 1;
            continue;
        }
        let rest = &s[i..];
        if rest.starts_with("NaN") {
            out.push_str("\"NaN\"");
            i += 3;
        } else if rest.starts_with("-Infinity") {
            out.push_str("\"-Inf\"");
            i += 9;
        } else if rest.starts_with("Infinity") {
            out.push_str("\"+Inf\"");
            i += 8;
        } else {
            // copy one UTF-8 char
            let ch = rest.chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

#[derive(Default)]
struct MirrorRun {
    keep: std::collections::HashSet<std::path::PathBuf>,
    bytes: u64,
    fetched: u64,
    skipped: Vec<String>,
    error: Option<String>,
}

fn sql_str(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// Evaluator workers are lent while the reader's RSS plus their estimate stays under this % of
/// its cap: above the heavy-query gate (a heavy query already admitted is running), below the
/// 70% at which a big shape is unmarked.
const PAR_MEM_PCT: u64 = 65;

impl Reader {
    pub fn new(shared: Shared, cfg: HttpCfg, views_sql: String, dict: Vec<DictEntry>) -> anyhow::Result<Self> {
        let cfg = cfg.resolve();
        crate::eval::footer::set_cap((cfg.footer_cache_mb as usize) << 20);
        crate::eval::rowcache::set_cap((cfg.row_cache_mb as usize) << 20);
        crate::eval::par::set_slots(
            cfg.eval_workers,
            shared.active_queries.clone(),
            (cfg.reader_rss_cap_mb << 20) * PAR_MEM_PCT / 100,
            cfg.eval_worker_mb_per_day,
        );
        let root = if crate::objstore::is_remote(&shared.root) {
            shared.root.clone()
        } else {
            shared.root.canonicalize()?
        };
        let tmp = shared.scratch.join("reader");
        std::fs::create_dir_all(&tmp)?;
        let db = Connection::open_in_memory()?;
        db.execute_batch(&format!(
            "SET memory_limit='{}'; SET threads={}; SET temp_directory={}; SET max_temp_directory_size='{}'; \
             SET GLOBAL TimeZone='UTC'; SET preserve_insertion_order=false; SET allocator_flush_threshold='16MB'; SET allocator_bulk_deallocation_flush_threshold='16MB'; SET autoinstall_known_extensions=false; SET autoload_known_extensions=false; \
             SET allow_community_extensions=false; SET allow_unsigned_extensions=false;",
            cfg.memory_limit,
            cfg.threads,
            sql_str(&tmp.to_string_lossy()),
            cfg.max_temp_directory_size
        ))?;
        // Metric dictionary: a plain table, created before access is locked down.
        db.execute_batch(
            "CREATE TABLE _metric_dict(metric_group VARCHAR, prefix VARCHAR, interval_s BIGINT, \
             instance_level BOOLEAN, gauges VARCHAR[], description VARCHAR)",
        )?;
        {
            let mut st = db.prepare("INSERT INTO _metric_dict VALUES (?, ?, ?, ?, string_split(?, ','), ?)")?;
            for d in &dict {
                st.execute(duckdb::params![
                    d.group,
                    d.prefix,
                    d.interval_s.map(|v| v as i64),
                    d.instance_level,
                    d.gauges.join(","),
                    d.description
                ])?;
            }
        }
        let scratch = tmp.canonicalize().unwrap_or(tmp.clone());
        // DuckDB reads local paths only: the store's directory, or on a remote store the local
        // mirror of its small kinds; never the s3:// URL (httpfs is not loaded)
        let mirror = if crate::objstore::is_remote(&root) {
            // one directory per store URL: readers of different stores may share a cache dir,
            // and their catalogs reuse the same relative paths
            let dir = match &cfg.s3.cache_dir {
                Some(d) => d.join("mirror"),
                None => shared.scratch.join("mirror"),
            }
            .join(format!(
                "{:016x}",
                xxhash_rust::xxh3::xxh3_64(root.to_string_lossy().as_bytes())
            ));
            std::fs::create_dir_all(&dir)?;
            Some(Mirror {
                dir: dir.canonicalize()?,
                cap_bytes: cfg.s3.mirror_mb << 20,
                stat: Mutex::new(serde_json::Value::Null),
            })
        } else {
            None
        };
        let readable = mirror.as_ref().map_or(&root, |m| &m.dir);
        db.execute_batch(&format!(
            "SET allowed_directories=[{}, {}]; SET enable_external_access=false; SET lock_configuration=true;",
            sql_str(&format!("{}/", readable.display())),
            sql_str(&format!("{}/", scratch.display()))
        ))?;
        // Rust evaluator table function (before access is locked; it reads Parquet itself).
        let eval = crate::eval::empty_handle(root.clone());
        db.register_table_function_with_extra_info::<crate::eval::vtab::PromEval, crate::eval::EvalHandle>(
            "prom_eval_rs",
            &eval,
        )?;
        db.register_table_function_with_extra_info::<crate::eval::vtab::PromSeries, crate::eval::EvalHandle>(
            "prom_series_rs",
            &eval,
        )?;
        db.register_table_function_with_extra_info::<crate::eval::topk::PromTopK, crate::eval::EvalHandle>(
            "prom_topk_rs",
            &eval,
        )?;
        db.register_table_function_with_extra_info::<crate::registry::TailIds, crate::eval::EvalHandle>(
            "pgai_series_tail_ids",
            &eval,
        )?;
        db.register_table_function_with_extra_info::<crate::eval::agg::PromAgg, crate::eval::EvalHandle>(
            "prom_agg_rs",
            &eval,
        )?;
        db.register_table_function_with_extra_info::<crate::eval::vtab::PgaiSamples, crate::eval::EvalHandle>(
            "pgai_samples_rs",
            &eval,
        )?;
        db.register_table_function_with_extra_info::<crate::eval::vtab::PgaiSeries, crate::eval::EvalHandle>(
            "pgai_series_rs",
            &eval,
        )?;
        // eval.sql loads last: it binds the generic prom_<fn> names to Rust. macros.sql ships
        // the SQL implementations as prom_<fn>_sql (the oracle), so nothing is shadowed.
        let mut views_sql = format!("{views_sql}\n;\n{}", crate::eval::EVAL_SQL);
        if mirror.is_some() {
            views_sql.push_str(REMOTE_PROM_RAW);
        }
        let mut volatile: std::collections::HashSet<String> = CLOCK_FNS.iter().map(|s| s.to_string()).collect();
        {
            let mut q = db.prepare(
                "SELECT DISTINCT lower(function_name) FROM duckdb_functions() \
                 WHERE stability IN ('VOLATILE', 'CONSISTENT_WITHIN_QUERY')",
            )?;
            for n in q.query_map([], |r| r.get::<_, String>(0))? {
                volatile.insert(n?);
            }
            // DuckDB's built-in SQL macros have no stability (ago(i) := current_timestamp - i,
            // pg_postmaster_start_time() := current_timestamp): a macro whose definition
            // names a volatile function is volatile, to a fixed point. This runs
            // before views.sql/macros.sql load, so only DuckDB's own macros are scanned.
            let mut q = db.prepare(
                "SELECT DISTINCT lower(function_name), lower(macro_definition) FROM duckdb_functions() \
                 WHERE function_type IN ('macro', 'table_macro') AND macro_definition IS NOT NULL",
            )?;
            let macros: Vec<(String, String)> = q
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<Result<_, _>>()?;
            loop {
                let before = volatile.len();
                for (name, def) in &macros {
                    if !volatile.contains(name)
                        && def
                            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                            .any(|w| volatile.contains(w))
                    {
                        volatile.insert(name.clone());
                    }
                }
                if volatile.len() == before {
                    break;
                }
            }
        }
        let r = Self {
            volatile,
            eval,
            shared: Shared { root, ..shared },
            admin: Mutex::new(db),
            pool: Default::default(),
            view_gen: AtomicU64::new(u64::MAX),
            refreshing: Mutex::new(()),
            query_snapshot: RwLock::new(()),
            cfg,
            views_sql,
            dict: Mutex::new(crate::dictionary::Dictionary::new(dict)),
            bounded_macros: Default::default(),
            mirror,
        };
        r.refresh()?;
        r.init_bounded_macros()?;
        Ok(r)
    }

    pub fn dictionary_version(&self) -> anyhow::Result<u64> {
        Ok(crate::dictionary::Dictionary::load(&self.shared.scratch)?
            .map_or_else(|| self.dict.lock().unwrap().version, |d| d.version))
    }

    /// Refresh the catalog and dictionary as one query snapshot. Publication waits for
    /// running queries; a dictionary change is picked up before the next query.
    /// Queries can use the previous snapshot during remote copies. Copies stop after half
    /// the query timeout; failed copies are retried on the next refresh.
    pub fn refresh(&self) -> anyhow::Result<()> {
        let g = self.shared.generation.load(Ordering::Acquire);
        let next_dict = crate::dictionary::Dictionary::load(&self.shared.scratch)?;
        let changed = || {
            next_dict
                .as_ref()
                .is_some_and(|d| d.version != self.dict.lock().unwrap().version)
        };
        if self.view_gen.load(Ordering::Acquire) == g && !changed() {
            return Ok(());
        }
        // Keep the dictionary, views and bound evaluator context on one snapshot per query.
        let _one = if changed() {
            self.refreshing.lock().unwrap()
        } else {
            match self.refreshing.try_lock() {
                Ok(one) => one,
                Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => return Ok(()),
            }
        };
        let next_dict = crate::dictionary::Dictionary::load(&self.shared.scratch)?;
        let changed = || {
            next_dict
                .as_ref()
                .is_some_and(|d| d.version != self.dict.lock().unwrap().version)
        };
        let vg = self.view_gen.load(Ordering::Acquire);
        if vg == g && !changed() {
            return Ok(());
        }
        let copy_until = Instant::now() + Duration::from_millis(self.cfg.timeout_ms / 2);
        let cat = self.shared.catalog.lock().unwrap().clone();
        let g = cat.generation;
        let files = &cat.files;
        let root = &self.shared.root;
        // DuckDB's file list for each view: the store's own paths, or the local mirror's
        let mut ms = MirrorRun::default();
        let mut paths = |kind: &str, fs: &[&FileEntry]| -> Option<Vec<String>> {
            let Some(m) = &self.mirror else {
                return Some(
                    fs.iter()
                        .map(|f| root.join(&f.path).to_string_lossy().into_owned())
                        .collect(),
                );
            };
            // one budget per refresh across every mirrored kind, in real bytes: a copy is used only
            // when its size equals the catalog's, and download() refuses any other object size
            let total = fs.iter().fold(0u64, |a, f| a.saturating_add(f.bytes));
            if kind == "samples" || ms.bytes.saturating_add(total) > m.cap_bytes {
                // raw samples are never copied (the whole store); prom_* read them from S3
                if !fs.is_empty() {
                    ms.skipped.push(kind.to_string());
                }
                return None;
            }
            let mut out = Vec::with_capacity(fs.len());
            for f in fs {
                let res = mirror_path(&m.dir, &f.path).and_then(|dst| {
                    let fresh =
                        std::fs::symlink_metadata(&dst).is_ok_and(|x| x.is_file() && f.bytes > 0 && x.len() == f.bytes);
                    if !fresh {
                        anyhow::ensure!(Instant::now() < copy_until, "not copied yet; the next query continues");
                        std::fs::create_dir_all(dst.parent().unwrap())?;
                        ms.fetched += crate::objstore::download(&root.join(&f.path), &dst, f.bytes)?;
                    }
                    Ok(dst)
                });
                match res {
                    Ok(dst) => {
                        ms.bytes += f.bytes;
                        out.push(dst.to_string_lossy().into_owned());
                        ms.keep.insert(dst);
                    }
                    Err(e) => {
                        // this kind stays empty until a refresh mirrors it (retried next query)
                        eprintln!("reader: mirror {kind} {}: {e:#}", f.path);
                        ms.error = Some(format!("{kind}: {e:#}"));
                        ms.skipped.push(kind.to_string());
                        return None;
                    }
                }
            }
            Some(out)
        };
        // files exist but are not mirrored (over the budget, a failed download, or not copied
        // yet): an empty view would answer 0 rows as if there were no data
        let unavailable = |kind: &str| {
            failing_select(
                kind,
                &format!(
                    "{kind} is not available right now: its files on the object store are over the mirror budget \
                     (http.s3.mirror_mb), could not be downloaded or are still being copied; see s3_mirror in GET \
                     /health"
                ),
            )
        };
        let mut sql = String::new();
        // Series registry (registry.rs): disjoint registry files, and the unregistered tail.
        for (name, fs) in [
            (
                "_raw_series_reg",
                files
                    .iter()
                    .filter(|f| f.kind == crate::registry::KIND)
                    .collect::<Vec<&FileEntry>>(),
            ),
            (
                "_raw_series_tail",
                files.iter().filter(|f| f.kind == "series" && !f.registered).collect(),
            ),
        ] {
            let body = match paths(name, &fs) {
                Some(ps) if !ps.is_empty() => {
                    let mut b = format!("SELECT * FROM read_parquet({}, filename = true)", sql_list(&ps));
                    if self.mirror.is_some() {
                        // `filename` names the store's file (pgai_series_tail_ids joins on it)
                        let arms: String = ps
                            .iter()
                            .zip(&fs)
                            .map(|(l, f)| {
                                format!(
                                    " WHEN {} THEN {}",
                                    sql_str(l),
                                    sql_str(&root.join(&f.path).to_string_lossy())
                                )
                            })
                            .collect();
                        b = format!("SELECT * EXCLUDE (filename), CASE filename{arms} END AS filename FROM ({b})");
                    }
                    b
                }
                None if !fs.is_empty() => {
                    format!("SELECT *, NULL::VARCHAR AS filename FROM ({})", unavailable("series"))
                }
                _ => format!("SELECT *, NULL::VARCHAR AS filename FROM ({})", empty_select("series")),
            };
            sql.push_str(&format!("CREATE OR REPLACE VIEW {name} AS {body};\n"));
        }
        // small kinds first, so on a remote store the mirror budget goes to them before logs
        let mut order: Vec<&str> = KINDS.to_vec();
        order.sort_by_key(|k| match *k {
            "logs" => 1,
            "samples" => 2,
            _ => 0,
        });
        for kind in &order {
            let fs: Vec<&FileEntry> = files.iter().filter(|f| f.kind == *kind).collect();
            let body = if *kind == "series" && self.mirror.is_some() {
                // every day's series file would be most of the series data: on a remote store
                // `_raw_series` is the registry plus the unregistered tail (the same series)
                "SELECT series_id, metric, labels, first_seen FROM _raw_series_reg \
                 UNION ALL SELECT series_id, metric, labels, first_seen FROM _raw_series_tail"
                    .to_string()
            } else {
                match paths(kind, &fs) {
                    Some(ps) if !ps.is_empty() => {
                        let src = crate::store::parquet_src(kind, &sql_list(&ps));
                        if *kind == "samples" {
                            samples_raw_select(&src)
                        } else {
                            format!("SELECT * FROM {src}")
                        }
                    }
                    _ if *kind == "samples" && self.mirror.is_some() => failing_select(kind, REMOTE_SAMPLES_ERROR),
                    None if self.mirror.is_some() && !fs.is_empty() => unavailable(kind),
                    _ => empty_select(kind).to_string(),
                }
            };
            sql.push_str(&format!("CREATE OR REPLACE VIEW _raw_{kind} AS {body};\n"));
        }
        // Copy remote files before taking the query snapshot lock. Other queries can keep
        // using the previous context while a slow mirror copy runs.
        // Queue publication after copying so overlapping queries cannot starve refresh.
        let _snapshot = self.query_snapshot.write().unwrap();
        let admin = self.admin.lock().unwrap();
        let mut dict = self.dict.lock().unwrap();
        let dict_changed = next_dict.as_ref().is_some_and(|d| d.version != dict.version);
        if let Some(d) = next_dict.as_ref().filter(|_| dict_changed) {
            sql.push_str("DELETE FROM _metric_dict;\n");
            for e in &d.entries {
                sql.push_str(&format!(
                    "INSERT INTO _metric_dict VALUES ({}, {}, {}, {}, string_split({}, ','), {});\n",
                    sql_str(&e.group),
                    sql_str(&e.prefix),
                    e.interval_s.map_or("NULL".into(), |n| n.to_string()),
                    e.instance_level,
                    sql_str(&e.gauges.join(",")),
                    sql_str(&e.description)
                ));
            }
        }
        // one transaction: a query preparing on another connection sees all the old views or all
        // the new ones, never `_raw_series_reg` from one catalog with `_raw_series_tail` from another
        if let Err(e) = admin.execute_batch(&format!("BEGIN;\n{sql}COMMIT;")) {
            let _ = admin.execute_batch("ROLLBACK");
            return Err(e.into());
        }
        if vg == u64::MAX {
            admin.execute_batch(&self.views_sql)?;
        }
        if let Some(m) = &self.mirror {
            mirror_gc(&m.dir, &ms.keep, Duration::from_secs(600));
            *m.stat.lock().unwrap() = serde_json::json!({
                "dir": m.dir, "files": ms.keep.len(), "bytes": ms.bytes, "fetched_bytes": ms.fetched,
                "empty_kinds": ms.skipped, "error": ms.error,
            });
        }
        // Evaluator context: the file list (series are read per call, for its window).
        {
            let live: std::collections::HashSet<std::path::PathBuf> =
                files.iter().map(|f| self.shared.root.join(&f.path)).collect();
            crate::eval::footer::retain(|p| live.contains(p));
        }
        let mut ctx = crate::eval::EvalCtx::from_catalog(self.shared.root.clone(), &cat);
        ctx.generation = g;
        if dict_changed {
            *dict = next_dict.unwrap();
        }
        ctx.dict = dict.entries.clone();
        ctx.dict_version = dict.version;
        *self.eval.write().unwrap() = Arc::new(ctx);
        // a failed mirror download is retried by the next refresh
        if ms.error.is_none() {
            self.view_gen.store(g, Ordering::Release);
        }
        Ok(())
    }

    fn init_bounded_macros(&self) -> anyhow::Result<()> {
        let db = self.admin.lock().unwrap();
        let mut q = db.prepare("SELECT lower(function_name), macro_definition, function_type FROM duckdb_functions() WHERE function_type IN ('macro', 'table_macro')")?;
        let defs = q
            .query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let names = defs.iter().map(|(n, _, _)| n.clone()).collect();
        let mut trees = Vec::new();
        for (name, def, kind) in defs {
            let sql = if kind == "table_macro" {
                def
            } else {
                format!("SELECT {def}")
            };
            let json: String = db.query_row("SELECT json_serialize_sql(?::VARCHAR)::VARCHAR", [sql], |r| r.get(0))?;
            let tree: serde_json::Value = serde_json::from_str(&json)?;
            trees.push((name, tree));
        }
        let mut safe = std::collections::HashSet::new();
        loop {
            let before = safe.len();
            for name in &names {
                // A safe overload must not bless an unsafe overload or schema sibling.
                if trees.iter().filter(|(n, _)| n == name).all(|(_, tree)| {
                    tree["error"] == false
                        && ast_cacheable(tree, &self.volatile)
                        && crate::readset::bounded_sql(tree, &names, &safe, &Default::default())
                }) {
                    safe.insert(name.clone());
                }
            }
            if safe.len() == before {
                break;
            }
        }
        let _ = self.bounded_macros.set((names, safe));
        Ok(())
    }

    /// Check that `sql` is exactly one SELECT, using DuckDB's parser.
    /// Ok((cacheable, bounded evaluator dependencies)) for a single read-only SELECT.
    fn check_read_only(&self, conn: &Connection, sql: &str) -> Result<(bool, bool), QueryError> {
        let j: String = conn
            .query_row("SELECT json_serialize_sql(?::VARCHAR)::VARCHAR", [sql], |r| r.get(0))
            .map_err(|e| classify(&e))?;
        let v: serde_json::Value = serde_json::from_str(&j).map_err(|e| QueryError::new("internal", e.to_string()))?;
        if v["error"].as_bool() == Some(true) {
            let msg = v["error_message"].as_str().unwrap_or("").to_string();
            let et = v["error_type"].as_str().unwrap_or("");
            return Err(if msg.contains("Out of Memory") || et.contains("Out of Memory") {
                QueryError::new("out_of_memory", msg)
            } else if et == "not implemented" || msg.contains("Only SELECT") {
                QueryError::new(
                    "not_read_only",
                    "only a single read-only SELECT (or WITH ... SELECT) is accepted",
                )
            } else {
                QueryError::new("parse_error", msg)
            });
        }
        let n = v["statements"].as_array().map(|a| a.len()).unwrap_or(0);
        if n != 1 {
            return Err(QueryError::new(
                "not_read_only",
                format!("exactly one statement expected, got {n}"),
            ));
        }
        if let Some(f) = refused_table_fn(&v["statements"]) {
            return Err(QueryError::new(
                "forbidden",
                format!("table function {f} is not available here (see GET /schema for the tables and functions)"),
            ));
        }
        let bounded = self.bounded_macros.get().is_some_and(|(names, safe)| {
            crate::readset::bounded_sql(&v["statements"], names, safe, &Default::default())
        });
        Ok((ast_cacheable(&v["statements"], &self.volatile), bounded))
    }

    /// Run one read-only query under the limits. Blocking; call from a blocking thread.
    pub fn run(&self, sql: &str) -> Result<QueryResult, QueryError> {
        self.run_fmt(sql, Fmt::Rows)
    }

    /// Run one read-only query in the given response format.
    pub fn run_fmt(&self, sql: &str, fmt: Fmt) -> Result<QueryResult, QueryError> {
        struct Active<'a>(&'a std::sync::atomic::AtomicUsize);
        impl Drop for Active<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }
        self.shared.active_queries.fetch_add(1, Ordering::Relaxed);
        let _active = Active(&self.shared.active_queries);
        let t0 = Instant::now();
        self.refresh()
            .map_err(|e| QueryError::new("internal", format!("view refresh: {e}")))?;
        let _snapshot = self.query_snapshot.read().unwrap();
        let gen = self.view_gen.load(Ordering::Acquire);
        let dict_version = self.dict.lock().unwrap().version;
        // Connections are reused (a pool); a fresh one is cloned from the admin connection.
        let conn = match self.pool.lock().unwrap().pop() {
            Some(c) => c,
            None => self.admin.lock().unwrap().try_clone().map_err(|e| classify(&e))?,
        };
        struct Back<'a>(&'a Mutex<Vec<Connection>>, Option<Connection>);
        impl Drop for Back<'_> {
            fn drop(&mut self) {
                if let Some(c) = self.1.take() {
                    let mut p = self.0.lock().unwrap();
                    if p.len() < 8 {
                        p.push(c);
                    }
                }
            }
        }
        let conn = Back(&self.pool, Some(conn));
        let conn = conn.1.as_ref().unwrap();
        let body = sql
            .trim()
            .trim_end_matches(|c: char| c == ';' || c.is_whitespace())
            .to_string();
        if body.is_empty() {
            return Err(QueryError::new("parse_error", "empty query"));
        }
        let t_chk = Instant::now();
        let (cacheable, bounded) = self.check_read_only(conn, &body)?;
        let t_chk = t_chk.elapsed();

        // Timeout: a watchdog thread interrupts the connection at the deadline.
        let handle = conn.interrupt_handle();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        // from the query's arrival: the refresh above is part of its time
        let timeout = Duration::from_millis(self.cfg.timeout_ms).saturating_sub(t0.elapsed());
        let watchdog = std::thread::spawn(move || {
            if done_rx.recv_timeout(timeout).is_err() {
                handle.interrupt();
                true
            } else {
                false
            }
        });
        let t_run = Instant::now();
        let s3_0 = crate::objstore::totals();
        let me = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        let alone = ACTIVE.fetch_add(1, Ordering::AcqRel) == 0;
        crate::readset::reset();
        let res = self.run_inner(conn, &body, fmt);
        let range = crate::readset::take().filter(|_| bounded);
        ACTIVE.fetch_sub(1, Ordering::AcqRel);
        let s3 = match (s3_0, crate::objstore::totals()) {
            (Some(a), Some(b)) => {
                // (the totals go down when a store is closed meanwhile)
                let d = |i: usize| b[i].saturating_sub(a[i]);
                let gets = d(0);
                let warn = crate::objstore::gets_warn();
                Some(QueryS3 {
                    gets,
                    bytes: d(1),
                    demand_gets: d(2),
                    footer_gets: d(3),
                    exclusive: alone && STARTED.load(Ordering::Acquire) == me,
                    warning: (warn > 0 && gets > warn)
                        .then(|| format!("{gets} object-store GETs, over --s3-gets-warn {warn}")),
                })
            }
            _ => None,
        };
        if std::env::var_os("PGAI_QUERY_TIMING").is_some() {
            eprintln!(
                "query timing: total {:?} check {:?} run {:?}",
                t0.elapsed(),
                t_chk,
                t_run.elapsed()
            );
        }
        let _ = done_tx.send(());
        let timed_out = watchdog.join().unwrap_or(false);
        match res {
            Ok(mut r) => {
                r.elapsed_ms = t0.elapsed().as_millis() as u64;
                r.s3 = s3;
                r.gen = gen;
                r.cacheable = cacheable;
                r.range = range;
                r.dict_version = dict_version;
                if fmt == Fmt::Objects {
                    let mut size = EncodedSize(0);
                    serde_json::to_writer(&mut size, &r).unwrap();
                    if size.0 > self.cfg.chart_max_bytes {
                        return Err(self.chart_too_large());
                    }
                }
                Ok(r)
            }
            Err(e) if timed_out => Err(QueryError::new(
                "timeout",
                format!(
                    "query exceeded {} ms and was interrupted ({})",
                    self.cfg.timeout_ms, e.message
                ),
            )),
            Err(e) => Err(e),
        }
    }

    fn run_inner(&self, conn: &Connection, body: &str, fmt: Fmt) -> Result<QueryResult, QueryError> {
        let columnar = fmt == Fmt::Columnar;
        let chart = matches!(fmt, Fmt::Objects | Fmt::Csv);
        let max_bytes = if chart {
            self.cfg.chart_max_bytes
        } else {
            self.cfg.max_bytes
        };
        let mut csv = (fmt == Fmt::Csv).then(String::new);
        // One prepare: the query is bound once (it was DESCRIBE + a wrapping prepare, so every
        // table function bound twice), and its Arrow output is written as JSON in Rust
        // (rowjson.rs, the same text DuckDB's to_json gave). Streaming: DuckDB produces chunk by
        // chunk and stops when reading stops at the row/byte cap.
        let t_prep = Instant::now();
        let mut st = conn.prepare(body).map_err(|e| classify(&e))?;
        if std::env::var_os("PGAI_QUERY_TIMING").is_some() {
            eprintln!("query timing: prepare {:?}", t_prep.elapsed());
        }
        let mut out = Vec::new();
        let mut cols_out: Vec<ColBuilder> = Vec::new();
        let mut nrows = 0usize;
        let mut bytes = 0usize;
        let mut truncated = false;
        let columns: Vec<String>;
        let arrow_types: Vec<String>;
        let timing = std::env::var_os("PGAI_QUERY_TIMING").is_some();
        let t_exec = Instant::now();
        let mut t_fetch = Duration::ZERO;
        let mut t_encode = Duration::ZERO;
        {
            let mut stream = st.stream_arrow([]).map_err(|e| classify(&e))?;
            if timing {
                eprintln!("query timing: execute {:?}", t_exec.elapsed());
            }
            let schema = stream.get_schema();
            columns = schema.fields().iter().map(|f| f.name().clone()).collect();
            arrow_types = schema.fields().iter().map(|f| format!("{}", f.data_type())).collect();
            if let Some(csv) = &mut csv {
                for (i, name) in columns.iter().enumerate() {
                    if i > 0 {
                        csv.push(',');
                    }
                    crate::rowcsv::push_field(csv, name);
                }
                csv.push_str("\r\n");
                bytes = csv.len();
                if bytes > max_bytes {
                    return Err(self.chart_too_large());
                }
            }
            // objects format: each column's `"name":` prefix, JSON-escaped once
            let keys: Vec<String> = if fmt == Fmt::Objects {
                columns
                    .iter()
                    .map(|c| format!("{}:", serde_json::Value::String(c.clone())))
                    .collect()
            } else {
                vec![]
            };
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), QueryError> {
                let mut s = String::new();
                loop {
                    let t = timing.then(Instant::now);
                    let batch = stream.next();
                    if let Some(t) = t {
                        t_fetch += t.elapsed();
                    }
                    let Some(batch) = batch else { break };
                    let t = timing.then(Instant::now);
                    let cols = batch.columns();
                    if columnar && cols_out.is_empty() {
                        cols_out.resize_with(cols.len(), Default::default);
                    }
                    for i in 0..batch.num_rows() {
                        if nrows >= self.cfg.max_rows {
                            truncated = true;
                            break;
                        }
                        if let Some(csv) = &mut csv {
                            s.clear();
                            for (j, c) in cols.iter().enumerate() {
                                if j > 0 {
                                    s.push(',');
                                }
                                crate::rowcsv::push_value(&mut s, c.as_ref(), i);
                            }
                            if s.is_empty() {
                                // one column, NULL or '': a blank line is no row to a CSV reader
                                s.push_str("\"\"");
                            }
                            s.push_str("\r\n");
                            bytes += s.len();
                            if bytes > max_bytes {
                                truncated = true;
                                break;
                            }
                            csv.push_str(&s);
                            nrows += 1;
                            continue;
                        }
                        if columnar {
                            let mut add = 0;
                            for (j, c) in cols.iter().enumerate() {
                                s.clear();
                                crate::rowjson::push_value(&mut s, c.as_ref(), i);
                                add += cols_out[j].push(&s);
                            }
                            bytes += add;
                            if bytes > max_bytes {
                                // drop the row just added: the response holds whole rows only
                                for c in cols_out.iter_mut() {
                                    c.idx.pop();
                                }
                                truncated = true;
                                break;
                            }
                            nrows += 1;
                            continue;
                        }
                        if out.len() >= self.cfg.max_rows {
                            truncated = true;
                            break;
                        }
                        s.clear();
                        s.push(if keys.is_empty() { '[' } else { '{' });
                        for (j, c) in cols.iter().enumerate() {
                            if j > 0 {
                                s.push(',');
                            }
                            if let Some(k) = keys.get(j) {
                                s.push_str(k);
                            }
                            crate::rowjson::push_value(&mut s, c.as_ref(), i);
                        }
                        s.push(if keys.is_empty() { ']' } else { '}' });
                        bytes += s.len() + 1;
                        if bytes > max_bytes {
                            truncated = true;
                            break;
                        }
                        out.push(
                            serde_json::value::RawValue::from_string(s.clone())
                                .map_err(|e| QueryError::new("internal", e.to_string()))?,
                        );
                        nrows += 1;
                    }
                    if let Some(t) = t {
                        t_encode += t.elapsed();
                    }
                    if truncated {
                        break;
                    }
                }
                Ok(())
            }));
            match res {
                Ok(r) => r?,
                // duckdb-rs's ArrowStream panics on a fetch error (interrupt, out of memory, ...)
                Err(p) => {
                    let m = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "query failed while streaming".into());
                    let kind = if m.contains("INTERRUPT") || m.contains("Interrupted") {
                        "timeout"
                    } else if m.contains("Out of Memory") {
                        "out_of_memory"
                    } else {
                        "execution_error"
                    };
                    return Err(QueryError::new(kind, m));
                }
            }
        }
        if timing {
            eprintln!("query timing: fetch {t_fetch:?} encode {t_encode:?} rows {nrows} bytes {bytes}");
        }
        // A cut chart looks like a collector outage; return an error instead.
        if truncated && chart {
            return Err(self.chart_too_large());
        }
        // DuckDB type names from the executed statement's logical types.
        let types = (0..columns.len())
            .map(|i| {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::rowjson::type_name(&st.column_logical_type(i))
                }))
                .unwrap_or_else(|_| arrow_types[i].clone())
            })
            .collect();
        let data = if columnar {
            if cols_out.is_empty() {
                cols_out.resize_with(columns.len(), Default::default);
            }
            let mut d = String::with_capacity(bytes + 16);
            d.push('[');
            for (j, c) in cols_out.iter().enumerate() {
                if j > 0 {
                    d.push(',');
                }
                c.write(&mut d);
            }
            d.push(']');
            drop(cols_out);
            Some(serde_json::value::RawValue::from_string(d).map_err(|e| QueryError::new("internal", e.to_string()))?)
        } else {
            None
        };
        let result = QueryResult {
            row_count: nrows,
            columns,
            types,
            rows: out,
            fmt,
            csv,
            data,
            truncated,
            elapsed_ms: 0,
            gen: 0,
            cacheable: false,
            range: None,
            dict_version: 0,
            s3: None,
        };
        Ok(result)
    }

    fn chart_too_large(&self) -> QueryError {
        QueryError::new(
            "result_too_large",
            format!(
            "the result exceeds http.max_rows ({}) or http.chart_max_bytes ({}): narrow the time range or the filter; \
             or raise the limit together with Grafana memory (GRAFANA_MEM)",
            self.cfg.max_rows, self.cfg.chart_max_bytes),
        )
    }

    /// Reader-side counters (the `stats` op of the child protocol).
    pub fn stats(&self) -> serde_json::Value {
        let (n, b, h, m) = crate::eval::footer::stats();
        serde_json::json!({
            "footer_cache": {"entries": n, "bytes": b, "hits": h, "misses": m},
            "row_cache": crate::eval::rowcache::stats(),
            "s3": crate::objstore::stats_json(),
            "s3_mirror": self.mirror.as_ref().map(|m| m.stat.lock().unwrap().clone()),
            "pooled_connections": self.pool.lock().unwrap().len(),
            "json_fallback_values": crate::rowjson::FALLBACK.load(Ordering::Relaxed),
            "parallel_queries": crate::eval::par::PAR_QUERIES.load(Ordering::Relaxed),
            "parallel_workers_lent": crate::eval::par::PAR_WORKERS.load(Ordering::Relaxed),
            "parallel_denied_memory": crate::eval::par::PAR_DENIED_MEM.load(Ordering::Relaxed),
        })
    }

    pub fn schema(&self) -> Result<serde_json::Value, QueryError> {
        let views = self.run(
            "SELECT v.view_name, list({'name': c.column_name, 'type': c.data_type} ORDER BY c.column_index) AS columns \
             FROM duckdb_views() v JOIN duckdb_columns() c ON c.table_name = v.view_name \
             WHERE NOT v.internal AND v.view_name NOT LIKE '\\_%' ESCAPE '\\' GROUP BY v.view_name ORDER BY v.view_name",
        )?;
        let metrics = self.run(
            "SELECT md.metric, md.type, c.series, md.metric_group, md.interval_s, md.description \
             FROM metric_dict md JOIN (SELECT metric, count(*) AS series FROM series GROUP BY metric) c USING (metric) \
             ORDER BY md.metric",
        )?;
        Ok(serde_json::json!({
            "views": views.rows,
            "metrics_columns": metrics.columns,
            "metrics": metrics.rows,
            "notes": "Query with POST /sql (body: SQL text or {\"sql\": ...}). Only one SELECT per request. \
                      metric names and labels are exactly those of pgwatch's Prometheus sink; samples exist \
                      only when the metric was collected (see metric_dict.interval_s for the lookback to use).",
        }))
    }
}

// ---------------------------------------------------------------- HTTP

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

/// A reader's answer: the JSON body, and the catalog generation it was computed at (0 for
/// non-query ops).
#[derive(Debug, Clone)]
pub struct Reply {
    pub body: String,
    pub gen: u64,
    /// The reader found nothing in the SQL that can change between runs at one generation.
    pub cacheable: bool,
    pub range: Option<crate::readset::ReadRange>,
    pub dict_version: u64,
}

/// Where SQL runs: in this process (tests, `query`), or in the capped child process.
#[derive(Clone)]
pub enum Backend {
    InProc(Arc<Reader>),
    Proc(Arc<crate::sqlproc::ReaderProc>),
}

impl Backend {
    pub fn dictionary_version(&self) -> anyhow::Result<u64> {
        match self {
            Self::InProc(r) => r.dictionary_version(),
            Self::Proc(p) => p.dictionary_version(),
        }
    }

    pub async fn call(&self, op: &'static str, sql: String) -> Result<Reply, QueryError> {
        match self {
            Backend::Proc(p) => p.call(op, &sql).await,
            Backend::InProc(r) => {
                let r = r.clone();
                tokio::task::spawn_blocking(move || match op {
                    "schema" => r.schema().map(|v| Reply {
                        body: v.to_string(),
                        gen: 0,
                        cacheable: false,
                        range: None,
                        dict_version: 0,
                    }),
                    "stats" => Ok(Reply {
                        body: r.stats().to_string(),
                        gen: 0,
                        cacheable: false,
                        range: None,
                        dict_version: 0,
                    }),
                    op => r.run_fmt(&sql, Fmt::from_op(op)).map(|mut x| Reply {
                        body: x.csv.take().unwrap_or_else(|| serde_json::to_string(&x).unwrap()),
                        gen: x.gen,
                        cacheable: x.cacheable,
                        range: x.range,
                        dict_version: x.dict_version,
                    }),
                })
                .await
                .unwrap_or_else(|e| Err(QueryError::new("internal", format!("query thread panicked: {e}"))))
            }
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub backend: Backend,
    pub cfg: HttpCfg,
    pub adm: Arc<crate::admission::Admission>,
    pub cache: Arc<crate::qcache::QueryCache>,
    pub shared: Shared,
}

impl AppState {
    pub fn new(backend: Backend, cfg: HttpCfg, shared: Shared) -> Self {
        let cfg = cfg.resolve();
        // The memory gate reads the reader child's RSS as its watchdog last sampled it.
        let (probe, gate): (Option<crate::admission::MemProbe>, u64) = match &backend {
            Backend::Proc(p) => {
                let r = p.last_rss.clone();
                (
                    Some(Arc::new(move || match r.load(Ordering::Relaxed) {
                        0 => None,
                        v => Some(v),
                    })),
                    p.rss_cap * cfg.heavy_mem_gate_pct / 100,
                )
            }
            Backend::InProc(_) => (None, 0),
        };
        Self {
            backend,
            adm: crate::admission::Admission::new(cfg.max_concurrent, cfg.short_query_ms as f64, gate, probe),
            cache: Arc::new(crate::qcache::QueryCache::new(0, 0, "")),
            cfg,
            shared,
        }
    }

    /// Turn on the L1 result cache (`cache_mb`; `PGAI_QCACHE=0` turns it off). `views_text`:
    /// everything that defines what SQL means (views, macros, eval.sql), hashed into the key.
    pub fn with_cache(mut self, views_text: &str) -> Self {
        let off = std::env::var("PGAI_QCACHE").is_ok_and(|v| v == "0");
        let cap = if off { 0 } else { (self.cfg.cache_mb as usize) << 20 };
        self.cache = Arc::new(crate::qcache::QueryCache::new(
            cap,
            (self.cfg.cache_max_entry_kb as usize) << 10,
            views_text,
        ));
        self
    }
}

impl AppState {
    /// Run a big shape (alone among heavy queries) and watch the reader's peak RSS meanwhile:
    /// a run that stays under 70 % of the cap clears the shape's big mark.
    async fn run_big(&self, cost: &crate::admission::Cost, op: &'static str, sql: String) -> Result<Reply, QueryError> {
        let Backend::Proc(p) = &self.backend else {
            return self.backend.call(op, sql).await;
        };
        p.peak_rss.store(p.last_rss.load(Ordering::Relaxed), Ordering::Relaxed);
        let r = self.backend.call(op, sql).await;
        if r.is_ok() && p.peak_rss.load(Ordering::Relaxed) < p.rss_cap * 70 / 100 {
            self.adm.unmark_big(cost);
        }
        r
    }
}

fn err_response(e: QueryError) -> Response {
    let code = StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let reason = reason_phrase(&e);
    let mut r = (code, Json(serde_json::json!({ "error": e }))).into_response();
    if let Some(reason) = reason {
        r.extensions_mut().insert(reason);
    }
    r
}

/// The error's type and message as the HTTP/1 reason phrase, next to the JSON body. Grafana's
/// Infinity datasource drops the body of a failed request and shows the status line only, so a
/// panel would say "413 Payload Too Large" and nothing of what to do about it.
fn reason_phrase(e: &QueryError) -> Option<hyper::ext::ReasonPhrase> {
    let text = format!("{}: {}", e.kind, e.message);
    let ascii: String = text
        .chars()
        .map(|c| if c.is_ascii_graphic() { c } else { ' ' })
        .collect();
    let line: String = ascii.split_whitespace().collect::<Vec<_>>().join(" ");
    hyper::ext::ReasonPhrase::try_from(line.chars().take(300).collect::<String>()).ok()
}

fn json_body(s: String) -> Response {
    ([(axum::http::header::CONTENT_TYPE, "application/json")], s).into_response()
}

async fn post_sql(
    State(st): State<AppState>,
    axum::extract::RawQuery(query): axum::extract::RawQuery,
    headers: HeaderMap,
    body: String,
) -> Response {
    let is_json = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("json"));
    // response format: `?format=rows|columnar|objects|csv` (or "format" in a JSON body); rows by default
    let mut format = query
        .as_deref()
        .unwrap_or("")
        .split('&')
        .find_map(|kv| kv.strip_prefix("format="))
        .map(|v| v.to_string());
    let sql = if is_json || body.trim_start().starts_with('{') {
        match serde_json::from_str::<serde_json::Value>(&body) {
            Ok(v) => {
                if let Some(f) = v["format"].as_str() {
                    format.get_or_insert_with(|| f.to_string());
                }
                v["sql"].as_str().unwrap_or("").to_string()
            }
            Err(e) => return err_response(QueryError::new("invalid_input", format!("bad JSON body: {e}"))),
        }
    } else {
        body
    };
    let Some(f) = Fmt::parse(format.as_deref()) else {
        return err_response(QueryError::new(
            "invalid_input",
            format!(
                "unknown format {:?}: use rows, columnar, objects or csv",
                format.unwrap_or_default()
            ),
        ));
    };
    let (op, fmt) = (f.op(), f.key());
    // L1 result cache (qcache.rs): a hit or a coalesced follower needs no admission slot.
    let dict_version = match st.backend.dictionary_version() {
        Ok(v) => v,
        Err(e) => return err_response(QueryError::new("internal", format!("dictionary reload: {e}"))),
    };
    let snapshot = st.cache.snapshot(&st.shared.catalog.lock().unwrap(), dict_version);
    let (res, served) = st
        .cache
        .get_or_run(&sql, fmt, snapshot, {
            // the run is a spawned flight that owns its admission permit: a client that goes away
            // does not free the slot early
            let (st, sql) = (st.clone(), sql.clone());
            move || {
                let (st, sql) = (st.clone(), sql.clone());
                async move { run_admitted(&st, op, sql).await }
            }
        })
        .await;
    let mut r = match res {
        Ok(rep) => {
            let body = axum::body::Body::from(bytes::Bytes::from_owner(ReplyBody(rep)));
            let content_type = if f == Fmt::Csv {
                "text/csv; charset=utf-8"
            } else {
                "application/json"
            };
            ([(axum::http::header::CONTENT_TYPE, content_type)], body).into_response()
        }
        Err(e) => err_response(e),
    };
    r.headers_mut()
        .insert("x-pgai-cache", axum::http::HeaderValue::from_static(served.as_str()));
    r
}

/// A cached or shared reply as a response body without copying it.
struct ReplyBody(Arc<Reply>);
impl AsRef<[u8]> for ReplyBody {
    fn as_ref(&self) -> &[u8] {
        self.0.body.as_bytes()
    }
}

/// Admission, the reader call and the out-of-memory retries: everything a cache miss runs.
/// `op`: "sql" (row format), "sql_columnar", "sql_objects" or "sql_csv".
async fn run_admitted(st: &AppState, op: &'static str, sql: String) -> Result<Reply, QueryError> {
    let cfg = &st.cfg;
    let busy = |why: String| Err(QueryError::new("busy", why));
    // Admission (admission.rs): at most max_concurrent run; short queries bypass heavy ones;
    // up to queue_max wait for at most queue_wait_ms.
    if st.adm.waiting() >= cfg.queue_max {
        return busy(format!("queue full ({} waiting); retry later", cfg.queue_max));
    }
    let wait = Duration::from_millis(cfg.queue_wait_ms);
    let cost = st.adm.cost(&sql);
    let Some(permit) = st.adm.acquire(&cost, false, wait).await else {
        return busy(format!("waited {} ms for a query slot; retry later", cfg.queue_wait_ms));
    };
    struct Active<'a>(&'a std::sync::atomic::AtomicUsize);
    impl Drop for Active<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::Relaxed);
        }
    }
    st.shared.active_queries.fetch_add(1, Ordering::Relaxed);
    let _active = Active(&st.shared.active_queries);
    let t0 = Instant::now();
    let res = if cost.big {
        st.run_big(&cost, op, sql.clone()).await
    } else {
        st.backend.call(op, sql.clone()).await
    };
    st.adm.observe(&cost, t0.elapsed().as_secs_f64() * 1000.0);
    drop(permit);
    // Out of memory under concurrency gets one retry, within the queue wait (best effort):
    // - a reader kill fails every query in flight, culprit unknown: the shape is
    //   marked big (it runs alone among heavy queries from now on) and retried that way;
    // - DuckDB's own OOM: retried alone once every slot is free.
    match res {
        Err(e) if e.kind == "out_of_memory" && e.message.starts_with("the SQL reader process") => {
            let big = st.adm.mark_big(&cost);
            match st.adm.acquire(&big, false, wait).await {
                Some(p) => {
                    let r = st.run_big(&big, op, sql).await;
                    drop(p);
                    r
                }
                None => Err(e),
            }
        }
        Err(e) if e.kind == "out_of_memory" && cfg.max_concurrent > 1 => {
            match st.adm.acquire(&cost, true, wait).await {
                Some(p) => {
                    let r = st.backend.call(op, sql).await;
                    drop(p);
                    r
                }
                None => Err(e),
            }
        }
        other => other,
    }
}

async fn get_schema(State(st): State<AppState>) -> Response {
    match st.backend.call("schema", String::new()).await {
        Ok(s) => json_body(s.body),
        Err(e) => err_response(e),
    }
}

async fn get_health(State(st): State<AppState>) -> Response {
    Json(health(&st).await).into_response()
}

/// The body of GET /health: the process's counters and current values (selfmon.rs also
/// writes them as `pgai_mon_*` series).
pub async fn health(st: &AppState) -> serde_json::Value {
    let mut reader_stats = st
        .backend
        .call("stats", String::new())
        .await
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s.body).ok());
    // A process reader's own counters restart with it: report the parent's totals across
    // restarts instead (the child's are kept under "child").
    if let (Backend::Proc(p), Some(rs)) = (&st.backend, reader_stats.as_mut()) {
        if let (Some([g, b, d, f]), Some(s3)) = (p.s3.get(), rs.get_mut("s3")) {
            if let Some(o) = s3.as_object_mut() {
                let child = serde_json::json!({
                    "gets": o.get("gets"), "bytes": o.get("bytes"), "demand_gets": o.get("demand_gets"),
                });
                o.insert("gets".into(), g.into());
                o.insert("bytes".into(), b.into());
                o.insert("demand_gets".into(), d.into());
                if let Some(fo) = o.get_mut("footers").and_then(|x| x.as_object_mut()) {
                    fo.insert("gets".into(), f.into());
                }
                o.insert("child".into(), child);
                o.insert("since".into(), "parent_start".into());
            }
        }
    }
    let s = &st.shared.stats;
    let (files, bytes, gen) = {
        let c = st.shared.catalog.lock().unwrap();
        (
            c.files.len(),
            c.files.iter().map(|f| f.bytes).sum::<u64>(),
            c.generation,
        )
    };
    serde_json::json!({
        "version": crate::VERSION,
        "samples_in": s.samples_in.load(Ordering::Relaxed),
        "records_in": s.records_in.load(Ordering::Relaxed),
        "flushes": s.flushes.load(Ordering::Relaxed),
        "flush_ms_last": s.flush_ms_last.load(Ordering::Relaxed),
        "compactions": s.compactions.load(Ordering::Relaxed),
        "wal_bytes": s.wal_bytes.load(Ordering::Relaxed),
        "files": files,
        "parquet_bytes": bytes,
        "catalog_generation": gen,
        "series_known": st.shared.known_series.lock().unwrap().len(),
        "rss_bytes": crate::sqlproc::rss_bytes(std::process::id()),
        "cpu_seconds": crate::selfmon::cpu_seconds(),
        "admission": st.adm.stats(),
        "collector": crate::collector::health_json(),
        "late_flushes": s.late_flushes.load(Ordering::Relaxed),
        "s3_write": crate::bucket::health_json(),
        "cache": {"l1": st.cache.stats_json(), "reader": reader_stats},
        "reader": match &st.backend {
            Backend::Proc(p) => {
                let pid = p.pid.load(Ordering::Relaxed);
                serde_json::json!({
                    "process": true, "pid": pid,
                    "rss_bytes": if pid != 0 { crate::sqlproc::rss_bytes(pid) } else { None },
                    "rss_cap_bytes": p.rss_cap,
                    "restarts": p.restarts.load(Ordering::Relaxed),
                    "memory_kills": p.mem_kills.load(Ordering::Relaxed),
                    "idle_recycles": p.recycles.load(Ordering::Relaxed),
                })
            }
            Backend::InProc(_) => serde_json::json!({"process": false}),
        },
    })
}

/// The whole HTTP surface: the SQL routes plus `extra` (ingest), all behind `auth`, and the
/// liveness probe `/livez`, which needs no credentials and tells nothing.
pub fn app(state: AppState, extra: Router, auth: Arc<crate::auth::Auth>) -> Router {
    router(state)
        .merge(extra)
        .layer(axum::middleware::from_fn_with_state(auth, crate::auth::require))
        .route("/livez", get(crate::auth::livez))
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/sql", post(post_sql))
        .route("/schema", get(get_schema))
        .route("/health", get(get_health))
        .with_state(state)
}
