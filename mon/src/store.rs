//! The store: WAL -> in-memory buffers -> Parquet (per day, ZSTD, sorted) -> compaction.
//!
//! Exactly-once rule: a flush rotates the WAL (closing segment S), writes Parquet files to
//! temp names and renames them, then atomically rewrites the catalog with the new files and
//! `flushed_through_seg = S`, then deletes segments <= S. On startup, Parquet files that the
//! catalog doesn't list are orphans of an interrupted flush and are deleted; segments
//! <= flushed_through_seg are deleted; the rest are replayed. So a crash at any point yields
//! neither a gap nor a duplicate.
use crate::catalog::{Catalog, FileEntry};
use crate::config::StoreCfg;
use crate::model::*;
use crate::wal::{self, Wal};
use duckdb::{params, Connection};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub const DEFAULT_LAYOUT: &str = "v2_split_tz";
pub const LAYOUTS: &[&str] = &[
    "v1_double",
    "v2_double",
    "v1_split",
    "v2_split",
    "v2_split_tz",
    "v2_double_tz",
];
pub const KINDS: &[&str] = &[
    "samples",
    "series",
    "logs",
    "plan_events",
    "plan_defs",
    "query_texts",
    "objects",
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Failpoint {
    /// Crash after the Parquet files were renamed into place, before the catalog update.
    AfterParquet,
    /// Crash after the catalog update, before the WAL segment is deleted.
    AfterCatalog,
}

#[derive(Default)]
pub struct Stats {
    pub samples_in: AtomicU64,
    pub records_in: AtomicU64,
    pub flushes: AtomicU64,
    pub flush_ms_last: AtomicU64,
    pub compactions: AtomicU64,
    pub wal_bytes: AtomicU64,
    /// Flushes that carried a sample older than the closed watermark (data_epoch bumped).
    pub late_flushes: AtomicU64,
}

#[derive(Default)]
pub struct Buffers {
    pub samples: Vec<Sample>,
    pub series: Vec<SeriesDef>,
    pub logs: Vec<LogRow>,
    pub plan_events: Vec<PlanEvent>,
    pub plan_defs: Vec<PlanDef>,
    pub query_texts: Vec<QueryText>,
    pub objects: Vec<ObjectEvent>,
    pub log_offsets: BTreeMap<String, u64>,
    /// Encoded size of what is buffered (WAL bytes since the last commit).
    pub bytes: usize,
}

impl Buffers {
    fn is_empty(&self) -> bool {
        self.samples.is_empty()
            && self.series.is_empty()
            && self.logs.is_empty()
            && self.plan_events.is_empty()
            && self.plan_defs.is_empty()
            && self.query_texts.is_empty()
            && self.objects.is_empty()
            && self.log_offsets.is_empty()
    }
    fn push(&mut self, r: Record, size: usize) {
        self.bytes += size;
        match r {
            Record::Samples(v) => self.samples.extend(v),
            Record::Series(v) => self.series.extend(v),
            Record::Logs {
                rows,
                plan_events,
                plan_defs,
                file,
                offset,
            } => {
                self.logs.extend(rows);
                self.plan_events.extend(plan_events);
                self.plan_defs.extend(plan_defs);
                let e = self.log_offsets.entry(file).or_insert(0);
                *e = (*e).max(offset);
            }
            Record::PlanEvents(v) => self.plan_events.extend(v),
            Record::PlanDefs(v) => self.plan_defs.extend(v),
            Record::QueryTexts(v) => self.query_texts.extend(v),
            Record::Objects(v) => self.objects.extend(v),
        }
    }
}

/// Shared, read-mostly state used by producers and readers.
#[derive(Clone)]
pub struct Shared {
    pub root: PathBuf,
    pub catalog: Arc<Mutex<Catalog>>,
    pub generation: Arc<AtomicU64>,
    pub stats: Arc<Stats>,
    /// (day, series id) already written: producers emit a SeriesDef once per series per day, so
    /// the series files of a day list exactly the series active that day (time-scoped lookups).
    pub known_series: Arc<Mutex<HashSet<(i64, u64)>>>,
    /// Queries running in the reader; background compaction defers while > 0.
    pub active_queries: Arc<std::sync::atomic::AtomicUsize>,
    /// Normalized pg_stat_statements text -> queryid, to link plans on PG14/15.
    pub fp_to_queryid: Arc<Mutex<std::collections::HashMap<String, i64>>>,
    /// Scratch space (reader spill, reader spec): store/tmp, or an OS temp dir for a read-only
    /// open, which writes nothing into the store.
    pub scratch: PathBuf,
}

impl Shared {
    /// Open a store for reading only: catalog from disk, no WAL, no orphan cleanup, nothing
    /// written under `root` (query, serve --no-compact).
    pub fn open_read_only(root: &Path) -> anyhow::Result<Self> {
        anyhow::ensure!(
            crate::objstore::is_remote(root) || crate::catalog::path(root).exists(),
            "no store at {} (catalog.json missing)",
            root.display()
        );
        let cat = Catalog::load_or_new(root, DEFAULT_LAYOUT)
            .map_err(|e| anyhow::anyhow!("no store at {}: {e}", root.display()))?;
        let scratch = own_scratch_in(&std::env::temp_dir())?;
        Ok(Shared {
            root: root.to_path_buf(),
            generation: Arc::new(AtomicU64::new(cat.generation)),
            catalog: Arc::new(Mutex::new(cat)),
            stats: Default::default(),
            known_series: Default::default(),
            fp_to_queryid: Default::default(),
            active_queries: Default::default(),
            scratch,
        })
    }
}

/// A read-only open's scratch dir: `<tmp>/pgai-mon-<pid>`. `main` removes it on a clean exit
/// (`remove_own_scratch`); a crash or kill leaves it behind, so every open first sweeps the dirs
/// of processes that no longer run.
pub fn own_scratch_in(tmp: &Path) -> anyhow::Result<PathBuf> {
    sweep_dead_scratch(tmp);
    let d = tmp.join(format!("pgai-mon-{}", std::process::id()));
    std::fs::create_dir_all(&d)?;
    let _ = OWN_SCRATCH.set(d.clone());
    Ok(d)
}

static OWN_SCRATCH: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Remove this process's scratch dir, if it made one (a no-op in the sql-worker child, which
/// uses its parent's).
pub fn remove_own_scratch() {
    if let Some(d) = OWN_SCRATCH.get() {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// Remove `<tmp>/pgai-mon-<pid>` dirs whose process is gone; returns how many. Only all-digit
/// suffixes (not pgai-mon-tile-fmt-*), only dirs, never this process's own.
pub fn sweep_dead_scratch(tmp: &Path) -> usize {
    let Ok(rd) = std::fs::read_dir(tmp) else {
        return 0;
    };
    let me = std::process::id();
    let mut n = 0;
    for e in rd.flatten() {
        let name = e.file_name();
        let Some(pid) = name
            .to_str()
            .and_then(|s| s.strip_prefix("pgai-mon-"))
            .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|s| s.parse::<u32>().ok())
        else {
            continue;
        };
        if pid == me || !e.file_type().is_ok_and(|t| t.is_dir()) || pid_alive(pid) {
            continue;
        }
        if std::fs::remove_dir_all(e.path()).is_ok() {
            n += 1;
        }
    }
    n
}

/// Whether a process with this pid exists (EPERM: it does, owned by someone else).
fn pid_alive(pid: u32) -> bool {
    let Ok(p) = libc::pid_t::try_from(pid) else {
        return false;
    };
    if p <= 0 {
        return true;
    }
    let alive = unsafe { libc::kill(p, 0) } == 0;
    alive || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

pub struct StoreCore {
    pub shared: Shared,
    cfg: StoreCfg,
    db: Connection,
    wal: Wal,
    buf: Buffers,
    pub failpoint: Option<Failpoint>,
}

pub fn open_writer_db(root: &Path, cfg: &StoreCfg) -> anyhow::Result<Connection> {
    open_db(root, "writer", &cfg.writer_memory_limit, cfg.writer_threads)
}

fn open_db(root: &Path, name: &str, memory_limit: &str, threads: i64) -> anyhow::Result<Connection> {
    let tmp = root.join("tmp").join(name);
    std::fs::create_dir_all(&tmp)?;
    let db = Connection::open_in_memory()?;
    db.execute_batch(&format!(
        "SET memory_limit='{}'; SET threads={}; SET preserve_insertion_order=false; \
         SET temp_directory='{}'; SET TimeZone='UTC'; SET autoinstall_known_extensions=false; \
         SET autoload_known_extensions=false;",
        memory_limit,
        threads,
        tmp.display()
    ))?;
    Ok(db)
}

fn sql_str(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

pub fn file_list_sql(root: &Path, files: &[&FileEntry]) -> String {
    let v: Vec<String> = files
        .iter()
        .map(|f| sql_str(&root.join(&f.path).to_string_lossy()))
        .collect();
    format!("[{}]", v.join(","))
}

fn parquet_opts(cfg: &StoreCfg, layout: &str) -> String {
    let v = if layout.starts_with("v1") { "V1" } else { "V2" };
    // Non-sample kinds ("v2") have wide rows (log lines, plans): cap row groups by bytes, or
    // the writer buffers 122,880 log lines (~250 MB) per row group.
    let rg = if layout == "v2" {
        format!("ROW_GROUP_SIZE {}, ROW_GROUP_SIZE_BYTES '4MB'", cfg.row_group_size)
    } else {
        format!("ROW_GROUP_SIZE {}", cfg.row_group_size)
    };
    format!(
        "FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL {}, PARQUET_VERSION {v}, {rg}",
        cfg.zstd_level
    )
}

/// SELECT list that converts the buffer table `buf_samples(series_id, ts BIGINT ms, value)`
/// into the physical layout.
fn samples_projection(layout: &str) -> &'static str {
    // Staleness markers (buffer column `stale`) are stored as ival = 1 AND fval = NaN, a
    // combination real values never produce (integral -> ival only, other -> fval only), so a
    // real NaN stays distinguishable from a marker (VictoriaMetrics' StaleNaN vs NaN).
    let tz = layout.ends_with("_tz");
    let split = layout.contains("split");
    match (split, tz) {
        (false, false) => "series_id, epoch_ms(ts)::TIMESTAMP_MS AS ts, value",
        (false, true) => "series_id, make_timestamptz(ts * 1000) AS ts, value",
        (true, false) => {
            "series_id, epoch_ms(ts)::TIMESTAMP_MS AS ts, \
             CASE WHEN stale THEN 1 WHEN value = trunc(value) AND abs(value) < 9.0e15 THEN value::BIGINT END AS ival, \
             CASE WHEN stale THEN 'nan'::DOUBLE WHEN value = trunc(value) AND abs(value) < 9.0e15 THEN NULL ELSE value END AS fval"
        }
        (true, true) => {
            "series_id, make_timestamptz(ts * 1000) AS ts, \
             CASE WHEN stale THEN 1 WHEN value = trunc(value) AND abs(value) < 9.0e15 THEN value::BIGINT END AS ival, \
             CASE WHEN stale THEN 'nan'::DOUBLE WHEN value = trunc(value) AND abs(value) < 9.0e15 THEN NULL ELSE value END AS fval"
        }
    }
}

/// Logical relation (series_id UBIGINT, ts TIMESTAMPTZ, value DOUBLE) over physical files.
pub fn samples_raw_select(layout: &str, src: &str) -> String {
    let (value, stale) = if layout.contains("split") {
        (
            "coalesce(fval, ival::DOUBLE)",
            "(ival IS NOT NULL AND fval IS NOT NULL)",
        )
    } else {
        ("value", "isnan(value)")
    };
    let ts = if layout.ends_with("_tz") {
        "ts"
    } else {
        "ts::TIMESTAMPTZ"
    };
    format!("SELECT series_id, {ts} AS ts, {value} AS value, {stale} AS stale FROM {src}")
}

/// Typed empty relation per kind (used when the kind has no files yet), matching the
/// logical schema the raw views expose.
/// Row group size of series (and registry) files. Readers decode a whole row group's labels
/// column at once (arrow-rs reads nested columns per column chunk): with 122,880 rows that was
/// ~50 MiB for one selection on a burst day, 8,192 rows keep it at a few MiB.
pub const SERIES_ROW_GROUP: usize = 8192;
/// One deduplicated series file per closed month: the series with samples in that month. A
/// selection over a window that covers whole months reads these instead of their day files.
pub const SERIES_MONTH: &str = "series_month";

/// First day (days since epoch) of the UTC month containing `day`, and of the next month.
pub fn month_of(day: i64) -> (i64, i64) {
    let d = chrono_free::civil_from_days(day);
    let first = chrono_free::days_from_civil(d.0, d.1, 1);
    let (ny, nm) = if d.1 == 12 { (d.0 + 1, 1) } else { (d.0, d.1 + 1) };
    (first, chrono_free::days_from_civil(ny, nm, 1))
}

/// Civil-calendar conversions (Howard Hinnant's algorithms), no date crate needed.
pub mod chrono_free {
    pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
        let y = if m <= 2 { y - 1 } else { y };
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let mp = (m + 9) % 12;
        let doy = (153 * mp + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }
    pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
        let z = z + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        (if m <= 2 { y + 1 } else { y }, m, d)
    }
}

pub fn empty_select(kind: &str) -> &'static str {
    match kind {
        "samples" => "SELECT NULL::UBIGINT AS series_id, NULL::TIMESTAMPTZ AS ts, NULL::DOUBLE AS value, NULL::BOOLEAN AS stale WHERE false",
        "series" => "SELECT NULL::UBIGINT AS series_id, NULL::VARCHAR AS metric, NULL::MAP(VARCHAR, VARCHAR) AS labels, NULL::TIMESTAMPTZ AS first_seen WHERE false",
        "logs" => "SELECT NULL::TIMESTAMPTZ AS ts, NULL::VARCHAR AS level, NULL::VARCHAR AS sqlstate, NULL::BIGINT AS pid, NULL::VARCHAR AS dbname, NULL::VARCHAR AS username, NULL::VARCHAR AS application, NULL::VARCHAR AS message, NULL::VARCHAR AS query_fingerprint, NULL::VARCHAR AS plan_hash, NULL::BIGINT AS queryid, NULL::VARCHAR AS raw WHERE false",
        "plan_events" => "SELECT NULL::TIMESTAMPTZ AS ts, NULL::VARCHAR AS plan_hash, NULL::BIGINT AS queryid, NULL::VARCHAR AS dbname, NULL::DOUBLE AS total_ms, NULL::VARCHAR AS top_node, NULL::BIGINT AS shared_hit, NULL::BIGINT AS shared_read, NULL::BIGINT AS shared_dirtied, NULL::BIGINT AS shared_written, NULL::BIGINT AS temp_read, NULL::BIGINT AS temp_written WHERE false",
        "plan_defs" => "SELECT NULL::VARCHAR AS plan_hash, NULL::TIMESTAMPTZ AS first_seen, NULL::VARCHAR AS plan WHERE false",
        "query_texts" => "SELECT NULL::BIGINT AS queryid, NULL::VARCHAR AS dbname, NULL::UBIGINT AS query_hash, NULL::VARCHAR AS query, NULL::TIMESTAMPTZ AS first_seen, NULL::TIMESTAMPTZ AS last_seen, NULL::VARCHAR AS source, NULL::VARCHAR AS cluster, NULL::VARCHAR AS node_name WHERE false",
        "objects" => "SELECT NULL::TIMESTAMPTZ AS ts, NULL::VARCHAR AS dbname, NULL::VARCHAR AS kind, NULL::VARCHAR AS schema, NULL::VARCHAR AS name, NULL::VARCHAR AS definition, NULL::VARCHAR AS source, NULL::VARCHAR AS cluster, NULL::VARCHAR AS node_name WHERE false",
        _ => unreachable!(),
    }
}

/// Merge SQL used by compaction: same physical columns, sorted (and deduplicated where the
/// kind is a registry).
/// `read_parquet` over a kind's files. query_texts and objects files written before the
/// source columns (source, cluster, node_name) were added lack them: union_by_name reads
/// those as NULL, so old and new files mix in views and compaction (compaction then writes
/// the new layout).
pub fn parquet_src(kind: &str, list: &str) -> String {
    match kind {
        "query_texts" | "objects" => format!("read_parquet({list}, union_by_name = true)"),
        _ => format!("read_parquet({list})"),
    }
}

pub fn merge_select(kind: &str, src: &str) -> String {
    match kind {
        "samples" => format!("SELECT * FROM {src} ORDER BY series_id, ts"),
        "series" => format!("SELECT DISTINCT ON (series_id) * FROM {src} ORDER BY series_id, first_seen"),
        "plan_defs" => format!("SELECT DISTINCT ON (plan_hash) * FROM {src} ORDER BY plan_hash, first_seen"),
        "query_texts" => format!(
            "SELECT queryid, dbname, query_hash, max(query) AS query, min(first_seen) AS first_seen, \
             max(last_seen) AS last_seen, source, cluster, node_name FROM {src} GROUP BY ALL \
             ORDER BY dbname, queryid, query_hash"
        ),
        // Event kinds: inputs are each time-sorted; a streaming copy (no global sort) keeps
        // compaction memory flat for large log days.
        _ => format!("SELECT * FROM {src}"),
    }
}

impl StoreCore {
    pub fn open(root: &Path, cfg: &StoreCfg) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !crate::objstore::is_remote(root),
            "{}: a store on an object store is written only by `run` (with store.s3.local_dir)",
            root.display()
        );
        std::fs::create_dir_all(root)?;
        let mut cat = Catalog::load_or_new(root, &cfg.samples_layout)?;
        if cat.samples_layout.is_empty() {
            cat.samples_layout = cfg.samples_layout.clone();
        }
        anyhow::ensure!(
            LAYOUTS.contains(&cat.samples_layout.as_str()),
            "unknown layout {}",
            cat.samples_layout
        );
        if cat.samples_layout != cfg.samples_layout {
            eprintln!(
                "warn: store layout is {} (config says {}); keeping the store's",
                cat.samples_layout, cfg.samples_layout
            );
        }
        remove_orphans(root, &cat)?;
        let db = open_writer_db(root, cfg)?;

        // Series written for today and yesterday (older days are never written again).
        let mut known = HashSet::new();
        {
            let today = day_of(now_ms());
            let files: Vec<&FileEntry> = cat.files_of("series").filter(|f| f.day >= today - 1).collect();
            if !files.is_empty() {
                let mut st = db.prepare(&format!(
                    "SELECT series_id, epoch_ms(first_seen) FROM read_parquet({})",
                    file_list_sql(root, &files)
                ))?;
                let mut rows = st.query([])?;
                while let Some(r) = rows.next()? {
                    let (id, fs): (u64, i64) = (r.get(0)?, r.get(1)?);
                    known.insert((day_of(fs), id));
                }
            }
        }

        // WAL recovery.
        let wal_dir = root.join("wal");
        let segs = wal::list_segments(&wal_dir)?;
        let max_seg = segs
            .iter()
            .map(|s| *s as i64)
            .max()
            .unwrap_or(-1)
            .max(cat.flushed_through_seg);
        let wal = Wal::open(&wal_dir, (max_seg + 1) as u64)?;
        let generation = Arc::new(AtomicU64::new(cat.generation));
        let flushed = cat.flushed_through_seg;
        let shared = Shared {
            root: root.to_path_buf(),
            catalog: Arc::new(Mutex::new(cat)),
            generation,
            stats: Arc::new(Stats::default()),
            known_series: Arc::new(Mutex::new(known)),
            fp_to_queryid: Arc::new(Mutex::new(Default::default())),
            active_queries: Arc::new(Default::default()),
            scratch: root.join("tmp"),
        };
        let mut core = Self {
            shared,
            cfg: cfg.clone(),
            db,
            wal,
            buf: Buffers::default(),
            failpoint: None,
        };
        // Replay segment by segment; commit whenever the buffer passes the size cap, so a
        // large backlog (a burst, an outage) never has to fit in memory at once.
        for s in segs {
            if (s as i64) <= flushed || s as i64 > max_seg || s == core.wal.seg() {
                if (s as i64) <= flushed {
                    let _ = std::fs::remove_file(wal::seg_path(&wal_dir, s));
                }
                continue;
            }
            let (recs, valid) = wal::read_segment(&wal::seg_path(&wal_dir, s))?;
            let len = std::fs::metadata(wal::seg_path(&wal_dir, s))?.len();
            if valid < len {
                eprintln!("wal: segment {s}: torn tail, truncating {len} -> {valid}");
                Wal::truncate_segment(&wal_dir, s, valid)?;
            }
            for (r, size) in recs {
                if let Record::Series(v) = &r {
                    let mut k = core.shared.known_series.lock().unwrap();
                    for d in v {
                        k.insert((day_of(d.first_seen), d.series_id));
                    }
                }
                core.buf.push(r, size);
            }
            if core.buf.bytes >= core.cfg.max_buffer_bytes {
                core.commit(s)?;
            }
        }
        Ok(core)
    }

    pub fn layout(&self) -> String {
        self.shared.catalog.lock().unwrap().samples_layout.clone()
    }

    pub fn buffered_samples(&self) -> usize {
        self.buf.samples.len()
    }

    pub fn max_buffered_ts(&self) -> Option<i64> {
        self.buf.samples.iter().map(|s| s.ts).max()
    }

    /// Tail positions covered by flushed data plus replayed WAL.
    pub fn log_offsets(&self) -> BTreeMap<String, u64> {
        let mut m = self.shared.catalog.lock().unwrap().log_offsets.clone();
        for (k, v) in &self.buf.log_offsets {
            let e = m.entry(k.clone()).or_insert(0);
            *e = (*e).max(*v);
        }
        m
    }

    pub fn conn(&self) -> &Connection {
        &self.db
    }

    /// Durably append a record (fsync) and buffer it.
    pub fn append(&mut self, rec: Record) -> anyhow::Result<()> {
        let size = self.wal.append(&rec)?;
        self.shared.stats.records_in.fetch_add(1, Ordering::Relaxed);
        self.shared.stats.wal_bytes.store(self.wal.bytes, Ordering::Relaxed);
        if let Record::Samples(v) = &rec {
            self.shared
                .stats
                .samples_in
                .fetch_add(v.len() as u64, Ordering::Relaxed);
        }
        self.buf.push(rec, size);
        Ok(())
    }

    /// Bytes buffered since the last commit.
    pub fn buffered_bytes(&self) -> usize {
        self.buf.bytes
    }

    /// Flush when the buffer passed the size cap (bounded memory under bursts).
    pub fn flush_if_full(&mut self) -> anyhow::Result<()> {
        if self.buf.bytes >= self.cfg.max_buffer_bytes {
            self.flush()?;
        }
        Ok(())
    }

    /// Write buffered data to Parquet and advance the catalog. See module docs.
    pub fn flush(&mut self) -> anyhow::Result<usize> {
        if self.buf.is_empty() {
            return Ok(0);
        }
        let seg = self.wal.rotate()?;
        match self.commit(seg) {
            Ok(n) => Ok(n),
            Err(e) => {
                // Don't retry the same batch shape forever: halve the write chunk.
                self.cfg.write_chunk_bytes = (self.cfg.write_chunk_bytes / 2).max(256 << 10);
                Err(e)
            }
        }
    }

    /// Write the buffers (everything in WAL segments <= seg) to Parquet, commit the catalog
    /// with flushed_through_seg = seg, then delete those segments.
    fn commit(&mut self, seg: u64) -> anyhow::Result<usize> {
        let t0 = std::time::Instant::now();
        // Buffers are cleared only after the catalog commit: on error they stay and the next
        // flush (which closes a later segment) covers them again.
        let buf = &self.buf;
        let layout = self.layout();
        let mut entries = Vec::new();
        let mut ids = { self.shared.catalog.lock().unwrap().next_file_id };
        let mut written = 0usize;
        {
            let mut alloc = || {
                ids += 1;
                ids
            };
            written += buf.samples.len();
            entries.extend(self.write_samples(&buf.samples, &layout, &mut alloc)?);
            entries.extend(self.write_generic("series", &buf.series, &mut alloc)?);
            entries.extend(self.write_generic("logs", &buf.logs, &mut alloc)?);
            entries.extend(self.write_generic("plan_events", &buf.plan_events, &mut alloc)?);
            entries.extend(self.write_generic("plan_defs", &buf.plan_defs, &mut alloc)?);
            entries.extend(self.write_generic("query_texts", &buf.query_texts, &mut alloc)?);
            entries.extend(self.write_generic("objects", &buf.objects, &mut alloc)?);
        }
        if self.failpoint == Some(Failpoint::AfterParquet) {
            anyhow::bail!("failpoint AfterParquet");
        }
        {
            let mut cat = self.shared.catalog.lock().unwrap();
            // A sample at or before the closed watermark (hwm - closed_lag_ms) changes data a
            // cache may already hold: bump data_epoch in the same catalog save.
            let lag = self.cfg.closed_lag_ms;
            if cat.closed_lag_ms != lag {
                if cat.closed_lag_ms != 0 {
                    cat.data_epoch += 1;
                }
                cat.closed_lag_ms = lag;
            }
            let hwm = cat.files.iter().filter(|f| f.kind == "samples").map(|f| f.max_ts).max();
            let batch_min = buf.samples.iter().map(|s| s.ts).min();
            if let (Some(h), Some(b)) = (hwm, batch_min) {
                if b <= h - lag {
                    cat.data_epoch += 1;
                    self.shared.stats.late_flushes.fetch_add(1, Ordering::Relaxed);
                }
            }
            cat.next_file_id = cat.next_file_id.max(ids);
            cat.files.extend(entries);
            cat.flushed_through_seg = seg as i64;
            for (f, o) in &buf.log_offsets {
                let e = cat.log_offsets.entry(f.clone()).or_insert(0);
                *e = (*e).max(*o);
            }
            cat.save(&self.shared.root)?;
            self.shared.generation.store(cat.generation, Ordering::Release);
        }
        self.buf = Buffers::default();
        if self.failpoint == Some(Failpoint::AfterCatalog) {
            anyhow::bail!("failpoint AfterCatalog");
        }
        for s in wal::list_segments(&self.shared.root.join("wal"))? {
            if s <= seg {
                self.wal.remove_segment(s);
            }
        }
        self.shared.stats.flushes.fetch_add(1, Ordering::Relaxed);
        self.shared
            .stats
            .flush_ms_last
            .store(t0.elapsed().as_millis() as u64, Ordering::Relaxed);
        Ok(written)
    }

    fn out_path(&self, kind: &str, day: i64, level: u8, id: u64) -> (String, PathBuf) {
        let rel = format!("{kind}/date={}/L{level}-{id:08}.parquet", day_str(day));
        let abs = self.shared.root.join(&rel);
        (rel, abs)
    }

    fn write_samples(
        &self,
        rows: &[Sample],
        layout: &str,
        alloc: &mut dyn FnMut() -> u64,
    ) -> anyhow::Result<Vec<FileEntry>> {
        let mut out = Vec::new();
        for c in rows.chunks(1_000_000) {
            out.extend(self.write_samples_chunk(c, layout, alloc)?);
        }
        Ok(out)
    }

    fn write_samples_chunk(
        &self,
        rows: &[Sample],
        layout: &str,
        alloc: &mut dyn FnMut() -> u64,
    ) -> anyhow::Result<Vec<FileEntry>> {
        if rows.is_empty() {
            return Ok(vec![]);
        }
        self.db.execute_batch(
            "CREATE OR REPLACE TEMP TABLE buf_samples(series_id UBIGINT, ts BIGINT, value DOUBLE, stale BOOLEAN)",
        )?;
        {
            let mut app = self.db.appender("buf_samples")?;
            for s in rows {
                app.append_row(params![s.series_id, s.ts, s.value, is_stale(s.value)])?;
            }
            app.flush()?;
        }
        let mut days: BTreeMap<i64, (u64, i64, i64)> = BTreeMap::new();
        for s in rows {
            let e = days.entry(day_of(s.ts)).or_insert((0, i64::MAX, i64::MIN));
            e.0 += 1;
            e.1 = e.1.min(s.ts);
            e.2 = e.2.max(s.ts);
        }
        let mut out = Vec::new();
        for (day, (n, mn, mx)) in days {
            let id = alloc();
            let (rel, abs) = self.out_path("samples", day, 0, id);
            let sel = format!(
                "SELECT {} FROM buf_samples WHERE ts >= {} AND ts < {} ORDER BY series_id, ts",
                samples_projection(layout),
                day * DAY_MS,
                (day + 1) * DAY_MS
            );
            let bytes = self.copy_atomic(&sel, &abs, layout)?;
            out.push(FileEntry {
                kind: "samples".into(),
                path: rel,
                day,
                level: 0,
                rows: n,
                bytes,
                min_ts: mn,
                max_ts: mx,
                created_ms: now_ms(),
                registered: false,
                ver: 0,
            });
        }
        self.db.execute_batch("DROP TABLE buf_samples")?;
        Ok(out)
    }

    /// COPY a SELECT to `abs` via a temp name + fsync + rename. Returns the file size.
    pub fn copy_atomic(&self, select: &str, abs: &Path, layout: &str) -> anyhow::Result<u64> {
        copy_atomic(&self.db, &self.cfg, select, abs, layout)
    }

    /// Writes in chunks of at most `write_chunk_bytes`, so DuckDB never holds a whole burst.
    fn write_generic<T: Appendable>(
        &self,
        kind: &str,
        rows: &[T],
        alloc: &mut dyn FnMut() -> u64,
    ) -> anyhow::Result<Vec<FileEntry>> {
        let mut out = Vec::new();
        let mut start = 0;
        let mut acc = 0usize;
        for (i, r) in rows.iter().enumerate() {
            acc += r.approx_bytes();
            if acc >= self.cfg.write_chunk_bytes {
                out.extend(self.write_generic_chunk(kind, &rows[start..=i], alloc)?);
                start = i + 1;
                acc = 0;
            }
        }
        if start < rows.len() {
            out.extend(self.write_generic_chunk(kind, &rows[start..], alloc)?);
        }
        Ok(out)
    }

    fn write_generic_chunk<T: Appendable>(
        &self,
        kind: &str,
        rows: &[T],
        alloc: &mut dyn FnMut() -> u64,
    ) -> anyhow::Result<Vec<FileEntry>> {
        if rows.is_empty() {
            return Ok(vec![]);
        }
        let tbl = format!("buf_{kind}");
        self.db
            .execute_batch(&format!("CREATE OR REPLACE TEMP TABLE {tbl}({})", T::buf_ddl()))?;
        {
            let mut app = self.db.appender(&tbl)?;
            for r in rows {
                r.append(&mut app)?;
            }
            app.flush()?;
        }
        let mut days: BTreeMap<i64, (u64, i64, i64)> = BTreeMap::new();
        for r in rows {
            let ts = r.ts();
            let e = days.entry(day_of(ts)).or_insert((0, i64::MAX, i64::MIN));
            e.0 += 1;
            e.1 = e.1.min(ts);
            e.2 = e.2.max(ts);
        }
        let mut out = Vec::new();
        for (day, (n, mn, mx)) in days {
            let id = alloc();
            let (rel, abs) = self.out_path(kind, day, 0, id);
            let sel = format!(
                "SELECT {} FROM {tbl} WHERE {} >= {} AND {} < {} ORDER BY {}",
                T::projection(),
                T::ts_col(),
                day * DAY_MS,
                T::ts_col(),
                (day + 1) * DAY_MS,
                T::order()
            );
            let bytes = self.copy_atomic(&sel, &abs, "v2")?;
            out.push(FileEntry {
                kind: kind.into(),
                path: rel,
                day,
                level: 0,
                rows: n,
                bytes,
                min_ts: mn,
                max_ts: mx,
                created_ms: now_ms(),
                registered: false,
                ver: 0,
            });
        }
        self.db.execute_batch(&format!("DROP TABLE {tbl}"))?;
        Ok(out)
    }

    /// Bulk path for the loader: write samples/series straight to staging Parquet (level 0)
    /// and catalog them, bypassing the WAL (the input file is the durable copy).
    pub fn bulk_write(&mut self, samples: &[Sample], series: &[SeriesDef]) -> anyhow::Result<()> {
        let layout = self.layout();
        let mut ids = { self.shared.catalog.lock().unwrap().next_file_id };
        let mut entries = Vec::new();
        {
            let mut alloc = || {
                ids += 1;
                ids
            };
            entries.extend(self.write_samples(samples, &layout, &mut alloc)?);
            entries.extend(self.write_generic("series", series, &mut alloc)?);
        }
        let mut cat = self.shared.catalog.lock().unwrap();
        cat.next_file_id = cat.next_file_id.max(ids);
        cat.files.extend(entries);
        cat.save(&self.shared.root)?;
        self.shared.generation.store(cat.generation, Ordering::Release);
        self.shared
            .stats
            .samples_in
            .fetch_add(samples.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    pub fn compactor(&self) -> anyhow::Result<Compactor> {
        let db = open_db(
            &self.shared.root,
            "compact",
            &self.cfg.compact_memory_limit,
            self.cfg.writer_threads,
        )?;
        Ok(Compactor {
            shared: self.shared.clone(),
            cfg: self.cfg.clone(),
            db,
        })
    }

    /// The uploader's own compactor (bucket.rs: the merge of not-yet-uploaded L0 files before
    /// each upload). Its inputs are a few minutes of flushes: a small DuckDB instance.
    pub fn upload_compactor(&self) -> anyhow::Result<Compactor> {
        let mut cfg = self.cfg.clone();
        cfg.compact_memory_limit = "32MB".into();
        cfg.compact_events_memory_limit = "32MB".into();
        let db = open_db(&self.shared.root, "upload", &cfg.compact_memory_limit, 1)?;
        Ok(Compactor {
            shared: self.shared.clone(),
            cfg,
            db,
        })
    }
}

/// Test hook: compaction merges (level 1 and 2) of the store at this root sleep this many ms
/// first, as a slow job would take (tests/bucket.rs).
pub static TEST_MERGE_DELAY: Mutex<Option<(PathBuf, u64)>> = Mutex::new(None);

pub fn copy_atomic(db: &Connection, cfg: &StoreCfg, select: &str, abs: &Path, layout: &str) -> anyhow::Result<u64> {
    std::fs::create_dir_all(abs.parent().unwrap())?;
    let tmp = abs.with_extension("parquet.tmp");
    db.execute_batch(&format!(
        "COPY ({select}) TO {} ({})",
        sql_str(&tmp.to_string_lossy()),
        parquet_opts(cfg, layout)
    ))?;
    std::fs::File::open(&tmp)?.sync_all()?;
    std::fs::rename(&tmp, abs)?;
    if let Ok(d) = std::fs::File::open(abs.parent().unwrap()) {
        let _ = d.sync_all();
    }
    Ok(std::fs::metadata(abs)?.len())
}

/// Delete Parquet files the catalog doesn't know (leftovers of an interrupted flush or
/// compaction) and temp files. Only runs at startup, the one time we list directories.
fn remove_orphans(root: &Path, cat: &Catalog) -> anyhow::Result<()> {
    let mut known: BTreeSet<PathBuf> = cat.files.iter().map(|f| root.join(&f.path)).collect();
    for (p, _) in &cat.pending_delete {
        known.insert(root.join(p));
    }
    for kind in KINDS
        .iter()
        .chain([crate::eval::rollup::KIND, crate::eval::rollup::KIND_1D, crate::eval::rollup::KIND_1HM, crate::registry::KIND].iter())
    {
        let kd = root.join(kind);
        if !kd.exists() {
            continue;
        }
        for day in std::fs::read_dir(&kd)? {
            let day = day?.path();
            if !day.is_dir() {
                continue;
            }
            for f in std::fs::read_dir(&day)? {
                let p = f?.path();
                let name = p.to_string_lossy();
                if name.ends_with(".tmp") || (name.ends_with(".parquet") && !known.contains(&p)) {
                    eprintln!("store: removing orphan {}", p.display());
                    std::fs::remove_file(&p)?;
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(root.join("tmp"));
    Ok(())
}

/// Row types that the flush path can append to a DuckDB buffer table.
pub trait Appendable {
    fn buf_ddl() -> &'static str;
    /// SELECT list from the buffer table to the physical file.
    fn projection() -> &'static str;
    fn ts_col() -> &'static str;
    fn order() -> &'static str;
    fn ts(&self) -> i64;
    fn append(&self, app: &mut duckdb::Appender) -> duckdb::Result<()>;
    fn approx_bytes(&self) -> usize {
        64
    }
}

impl Appendable for SeriesDef {
    fn approx_bytes(&self) -> usize {
        64 + self.metric.len() + self.labels.iter().map(|(k, v)| k.len() + v.len() + 8).sum::<usize>()
    }
    fn buf_ddl() -> &'static str {
        "series_id UBIGINT, metric VARCHAR, labels_json VARCHAR, first_seen BIGINT"
    }
    fn projection() -> &'static str {
        "series_id, metric, labels_json::JSON::MAP(VARCHAR, VARCHAR) AS labels, make_timestamptz(first_seen * 1000) AS first_seen"
    }
    fn ts_col() -> &'static str {
        "first_seen"
    }
    fn order() -> &'static str {
        "series_id"
    }
    fn ts(&self) -> i64 {
        self.first_seen
    }
    fn append(&self, app: &mut duckdb::Appender) -> duckdb::Result<()> {
        let mut m = serde_json::Map::new();
        for (k, v) in &self.labels {
            m.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        let j = serde_json::Value::Object(m).to_string();
        app.append_row(params![self.series_id, self.metric, j, self.first_seen])
    }
}

impl Appendable for LogRow {
    fn approx_bytes(&self) -> usize {
        96 + self.raw.len() + self.message.len()
    }
    fn buf_ddl() -> &'static str {
        "ts BIGINT, level VARCHAR, sqlstate VARCHAR, pid BIGINT, dbname VARCHAR, username VARCHAR, \
         application VARCHAR, message VARCHAR, query_fingerprint VARCHAR, plan_hash VARCHAR, queryid BIGINT, raw VARCHAR"
    }
    fn projection() -> &'static str {
        "make_timestamptz(ts * 1000) AS ts, level, sqlstate, pid, dbname, username, application, message, \
         query_fingerprint, plan_hash, queryid, raw"
    }
    fn ts_col() -> &'static str {
        "ts"
    }
    fn order() -> &'static str {
        "ts"
    }
    fn ts(&self) -> i64 {
        self.ts
    }
    fn append(&self, app: &mut duckdb::Appender) -> duckdb::Result<()> {
        app.append_row(params![
            self.ts,
            self.level,
            self.sqlstate,
            self.pid,
            self.dbname,
            self.username,
            self.application,
            self.message,
            self.query_fingerprint,
            self.plan_hash,
            self.queryid,
            self.raw
        ])
    }
}

impl Appendable for PlanEvent {
    fn buf_ddl() -> &'static str {
        "ts BIGINT, plan_hash VARCHAR, queryid BIGINT, dbname VARCHAR, total_ms DOUBLE, top_node VARCHAR, \
         shared_hit BIGINT, shared_read BIGINT, shared_dirtied BIGINT, shared_written BIGINT, temp_read BIGINT, temp_written BIGINT"
    }
    fn projection() -> &'static str {
        "make_timestamptz(ts * 1000) AS ts, plan_hash, queryid, dbname, total_ms, top_node, shared_hit, shared_read, \
         shared_dirtied, shared_written, temp_read, temp_written"
    }
    fn ts_col() -> &'static str {
        "ts"
    }
    fn order() -> &'static str {
        "ts"
    }
    fn ts(&self) -> i64 {
        self.ts
    }
    fn append(&self, app: &mut duckdb::Appender) -> duckdb::Result<()> {
        app.append_row(params![
            self.ts,
            self.plan_hash,
            self.queryid,
            self.dbname,
            self.total_ms,
            self.top_node,
            self.shared_hit,
            self.shared_read,
            self.shared_dirtied,
            self.shared_written,
            self.temp_read,
            self.temp_written
        ])
    }
}

impl Appendable for PlanDef {
    fn approx_bytes(&self) -> usize {
        64 + self.plan.len()
    }
    fn buf_ddl() -> &'static str {
        "plan_hash VARCHAR, first_seen BIGINT, plan VARCHAR"
    }
    fn projection() -> &'static str {
        "plan_hash, make_timestamptz(first_seen * 1000) AS first_seen, plan"
    }
    fn ts_col() -> &'static str {
        "first_seen"
    }
    fn order() -> &'static str {
        "plan_hash"
    }
    fn ts(&self) -> i64 {
        self.first_seen
    }
    fn append(&self, app: &mut duckdb::Appender) -> duckdb::Result<()> {
        app.append_row(params![self.plan_hash, self.first_seen, self.plan])
    }
}

impl Appendable for QueryText {
    fn approx_bytes(&self) -> usize {
        64 + self.query.as_ref().map_or(0, |q| q.len())
    }
    fn buf_ddl() -> &'static str {
        "queryid BIGINT, dbname VARCHAR, query_hash UBIGINT, query VARCHAR, first_seen BIGINT, last_seen BIGINT, \
         source VARCHAR, cluster VARCHAR, node_name VARCHAR"
    }
    fn projection() -> &'static str {
        "queryid, dbname, query_hash, query, make_timestamptz(first_seen * 1000) AS first_seen, \
         make_timestamptz(last_seen * 1000) AS last_seen, source, cluster, node_name"
    }
    fn ts_col() -> &'static str {
        "last_seen"
    }
    fn order() -> &'static str {
        "dbname, queryid"
    }
    fn ts(&self) -> i64 {
        self.last_seen
    }
    fn append(&self, app: &mut duckdb::Appender) -> duckdb::Result<()> {
        app.append_row(params![
            self.queryid,
            self.dbname,
            self.query_hash,
            self.query,
            self.first_seen,
            self.last_seen,
            self.src.source,
            self.src.cluster,
            self.src.node_name
        ])
    }
}

impl Appendable for ObjectEvent {
    fn approx_bytes(&self) -> usize {
        96 + self.definition.as_ref().map_or(0, |q| q.len())
    }
    fn buf_ddl() -> &'static str {
        "ts BIGINT, dbname VARCHAR, kind VARCHAR, schema VARCHAR, name VARCHAR, definition VARCHAR, \
         source VARCHAR, cluster VARCHAR, node_name VARCHAR"
    }
    fn projection() -> &'static str {
        "make_timestamptz(ts * 1000) AS ts, dbname, kind, schema, name, definition, source, cluster, node_name"
    }
    fn ts_col() -> &'static str {
        "ts"
    }
    fn order() -> &'static str {
        "ts"
    }
    fn ts(&self) -> i64 {
        self.ts
    }
    fn append(&self, app: &mut duckdb::Appender) -> duckdb::Result<()> {
        app.append_row(params![
            self.ts,
            self.dbname,
            self.kind,
            self.schema,
            self.name,
            self.definition,
            self.src.source,
            self.src.cluster,
            self.src.node_name
        ])
    }
}

/// Merges small files: the open day's flush files into one file per hour (level 1), and a
/// closed day into one file (level 2). Runs on its own thread and DuckDB connection.
pub struct Compactor {
    pub shared: Shared,
    cfg: StoreCfg,
    db: Connection,
}

struct Job {
    kind: String,
    day: i64,
    level: u8,
    inputs: Vec<FileEntry>,
}

impl Compactor {
    fn plan(&self, now: i64, all_closed: bool) -> Vec<Job> {
        let cat = self.shared.catalog.lock().unwrap();
        let today = day_of(now);
        let hour_start = now - now.rem_euclid(3_600_000);
        let mut groups: BTreeMap<(String, i64), Vec<FileEntry>> = BTreeMap::new();
        for f in &cat.files {
            groups.entry((f.kind.clone(), f.day)).or_default().push(f.clone());
        }
        let mut jobs = Vec::new();
        for ((kind, day), files) in groups {
            let day_closed =
                all_closed || (day < today && now >= (day + 1) * DAY_MS + self.cfg.compact_grace_s as i64 * 1000);
            if day_closed {
                let needs_rollup = kind == "samples"
                    && self.cfg.compact_kmerge
                    && !cat
                        .files
                        .iter()
                        .any(|f| f.kind == crate::eval::rollup::KIND && f.day == day);
                if (files.len() > 1 && !files.iter().all(|f| f.level == 2))
                    || (needs_rollup && kind != crate::eval::rollup::KIND)
                {
                    jobs.push(Job {
                        kind,
                        day,
                        level: 2,
                        inputs: files,
                    });
                }
            } else {
                // Open day: merge flush files from finished hours into one level-1 file.
                let l0: Vec<FileEntry> = files
                    .into_iter()
                    .filter(|f| f.level == 0 && f.created_ms < hour_start)
                    .collect();
                if l0.len() > 1 {
                    jobs.push(Job {
                        kind,
                        day,
                        level: 1,
                        inputs: l0,
                    });
                }
            }
        }
        jobs
    }

    /// One compaction pass. `all_closed` treats every past day as closed regardless of grace
    /// (the loader uses it). Returns the number of merges done.
    pub fn run_once(&self, now: i64, all_closed: bool) -> anyhow::Result<usize> {
        // on an object store: no eviction while the jobs read local files
        let bucket = crate::bucket::of(&self.shared.root);
        let _jobs = bucket.as_ref().map(|b| b.jobs.read().unwrap());
        let jobs = self.plan(now, all_closed);
        let mut done = 0;
        for job in jobs {
            let what = format!(
                "compaction of {} {} ({} files)",
                job.kind,
                day_str(job.day),
                job.inputs.len()
            );
            self.merge(job).map_err(|e| e.context(what))?;
            done += 1;
        }
        done += self.register(now, all_closed)?;
        done += self.series_months(now, all_closed)?;
        if self.cfg.compact_kmerge {
            // on an object store, the month jobs' inputs may have left the local directory
            if crate::bucket::of(&self.shared.root).is_some() {
                let files = self.shared.catalog.lock().unwrap().files.clone();
                for kind in [crate::eval::rollup::KIND_1D, crate::eval::rollup::KIND_1HM] {
                    for job in crate::eval::rollup::jobs_tier(&files, kind) {
                        let refs: Vec<&FileEntry> = job.inputs.iter().map(|x| &x.0).chain(job.old.iter()).collect();
                        crate::bucket::ensure_local(&self.shared.root, &refs)?;
                    }
                }
            }
            // 1 d rollup month files from the closed days' 1 h rollups
            done += crate::eval::rollup::update_1d(&self.shared, self.cfg.zstd_level, self.cfg.delete_grace_s as i64 * 1000)?;
        }
        self.purge(now)?;
        Ok(done)
    }

    /// Before an upload (bucket.rs): merge the level-0 files of each open (kind, day) written
    /// this hour and not in `uploaded` into one level-0 file, so a round PUTs one file per kind and day instead of
    /// one per flush. Returns the number of merges.
    pub fn merge_unuploaded(&self, uploaded: &HashSet<String>) -> anyhow::Result<usize> {
        let groups: BTreeMap<(String, i64), Vec<FileEntry>> = {
            let cat = self.shared.catalog.lock().unwrap();
            let mut g: BTreeMap<(String, i64), Vec<FileEntry>> = BTreeMap::new();
            // only what the compactor leaves alone: open days' files of the current hour (it
            // merges finished hours and closed days; at a boundary its merge check discards
            // whichever of two merges of the same inputs commits second)
            let now = now_ms();
            let (today, hour_start) = (day_of(now), now - now.rem_euclid(3_600_000));
            let open = |day: i64| day >= today || now < (day + 1) * DAY_MS + self.cfg.compact_grace_s as i64 * 1000;
            for f in cat.files.iter().filter(|f| {
                f.level == 0 && !uploaded.contains(&f.path) && f.created_ms >= hour_start && open(f.day)
            }) {
                g.entry((f.kind.clone(), f.day)).or_default().push(f.clone());
            }
            g
        };
        let mut n = 0;
        for ((kind, day), inputs) in groups {
            if inputs.len() < 2 {
                continue;
            }
            self.merge(Job {
                kind,
                day,
                level: 0,
                inputs,
            })?;
            n += 1;
        }
        Ok(n)
    }

    fn merge(&self, job: Job) -> anyhow::Result<()> {
        let layout = { self.shared.catalog.lock().unwrap().samples_layout.clone() };
        let refs: Vec<&FileEntry> = job.inputs.iter().collect();
        crate::bucket::ensure_local(&self.shared.root, &refs)?;
        let delay = TEST_MERGE_DELAY
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|(r, ms)| (*r == self.shared.root).then_some(*ms));
        if let (Some(ms), true) = (delay, job.level > 0) {
            std::thread::sleep(std::time::Duration::from_millis(ms));
        }
        let src = parquet_src(&job.kind, &file_list_sql(&self.shared.root, &refs));
        let file_layout = if job.kind == "samples" { layout.as_str() } else { "v2" };
        let total_rows: u64 = job.inputs.iter().map(|f| f.rows).sum();
        // Samples sort (bounded by splitting into parts); other kinds stream wide rows
        // (log lines, plans) and need more room per row group.
        let lim = if job.kind == "samples" {
            &self.cfg.compact_memory_limit
        } else {
            &self.cfg.compact_events_memory_limit
        };
        self.db.execute_batch(&format!("SET memory_limit='{lim}'"))?;
        // Samples: streaming k-way merge (inputs are each sorted), one file, O(files) memory.
        // The DuckDB sort split by series_id range remains only as a fallback (kmerge off).
        let kmerge = (job.kind == "samples" || job.kind == "series") && self.cfg.compact_kmerge;
        let parts = if !kmerge && job.kind == "samples" && total_rows > self.cfg.compact_part_rows {
            ((total_rows + self.cfg.compact_part_rows - 1) / self.cfg.compact_part_rows).min(10)
        } else {
            1
        };
        let mut ranges: Vec<String> = vec![String::new()];
        if parts > 1 {
            let qs: Vec<String> = (1..parts).map(|i| format!("{}", i as f64 / parts as f64)).collect();
            let b: String = self.db.query_row(
                &format!(
                    "SELECT approx_quantile(series_id, [{}])::VARCHAR FROM {src}",
                    qs.join(",")
                ),
                [],
                |r| r.get(0),
            )?;
            let mut bounds: Vec<u64> = b
                .trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            bounds.dedup();
            ranges.clear();
            let mut lo: Option<u64> = None;
            for hi in bounds.iter().copied().chain(std::iter::once(u64::MAX)) {
                let cond = match (lo, hi) {
                    (None, h) if h != u64::MAX => format!("series_id < {h}"),
                    (Some(l), u64::MAX) => format!("series_id >= {l}"),
                    (Some(l), h) => format!("series_id >= {l} AND series_id < {h}"),
                    (None, _) => "true".to_string(),
                };
                ranges.push(cond);
                lo = Some(hi);
            }
        }
        let mut outs = Vec::new();
        for cond in &ranges {
            let id = {
                let mut c = self.shared.catalog.lock().unwrap();
                c.alloc_id()
            };
            let rel = format!("{}/date={}/L{}-{id:08}.parquet", job.kind, day_str(job.day), job.level);
            let abs = self.shared.root.join(&rel);
            let from = if cond.is_empty() {
                src.clone()
            } else {
                format!("(SELECT * FROM {src} WHERE {cond})")
            };
            let bytes = if kmerge {
                std::fs::create_dir_all(abs.parent().unwrap())?;
                let tmp = abs.with_extension("parquet.tmp");
                let paths: Vec<PathBuf> = job.inputs.iter().map(|f| self.shared.root.join(&f.path)).collect();
                // series: dedupe by id (a series has one row per day; files may repeat it)
                crate::kmerge::merge_sorted_opts(
                    &paths,
                    &tmp,
                    self.cfg.zstd_level,
                    if job.kind == "series" {
                        SERIES_ROW_GROUP
                    } else {
                        self.cfg.row_group_size as usize
                    },
                    job.kind == "series",
                )?;
                std::fs::File::open(&tmp)?.sync_all()?;
                std::fs::rename(&tmp, &abs)?;
                if let Ok(d) = std::fs::File::open(abs.parent().unwrap()) {
                    let _ = d.sync_all();
                }
                std::fs::metadata(&abs)?.len()
            } else {
                copy_atomic(&self.db, &self.cfg, &merge_select(&job.kind, &from), &abs, file_layout)?
            };
            let rows: i64 = self.db.query_row(
                &format!(
                    "SELECT coalesce(sum(num_rows), 0)::BIGINT FROM parquet_file_metadata({})",
                    sql_str(&abs.to_string_lossy())
                ),
                [],
                |r| r.get(0),
            )?;
            outs.push((
                abs,
                FileEntry {
                    kind: job.kind.clone(),
                    path: rel,
                    day: job.day,
                    level: job.level,
                    rows: rows as u64,
                    bytes,
                    min_ts: job.inputs.iter().map(|f| f.min_ts).min().unwrap_or(0),
                    max_ts: job.inputs.iter().map(|f| f.max_ts).max().unwrap_or(0),
                    created_ms: now_ms(),
                    registered: job.kind == "series" && job.inputs.iter().all(|f| f.registered),
                    ver: 0,
                },
            ));
        }
        // 1 h rollup of a closed day's samples (served to aligned increase/rate/... windows).
        let rollup_entry = if job.kind == "samples" && job.level == 2 && outs.len() == 1 {
            let id = {
                let mut c = self.shared.catalog.lock().unwrap();
                c.alloc_id()
            };
            let rel = format!(
                "{}/date={}/R-{id:08}.parquet",
                crate::eval::rollup::KIND,
                day_str(job.day)
            );
            let abs = self.shared.root.join(&rel);
            std::fs::create_dir_all(abs.parent().unwrap())?;
            let tmp = abs.with_extension("parquet.tmp");
            let rows = crate::eval::rollup::build(&outs[0].0, &tmp, self.cfg.zstd_level)?;
            std::fs::File::open(&tmp)?.sync_all()?;
            std::fs::rename(&tmp, &abs)?;
            Some(FileEntry {
                kind: crate::eval::rollup::KIND.into(),
                path: rel,
                day: job.day,
                level: 2,
                rows,
                bytes: std::fs::metadata(&abs)?.len(),
                min_ts: outs[0].1.min_ts,
                max_ts: outs[0].1.max_ts,
                created_ms: now_ms(),
                registered: false,
                ver: crate::eval::rollup::VER,
            })
        } else {
            None
        };
        // Content checksum (order-independent): every input row is in the output exactly once.
        let dedups = matches!(job.kind.as_str(), "series" | "plan_defs" | "query_texts");
        // (The k-way merge verifies its own output in Rust, streaming; see kmerge.rs.)
        if !dedups && !kmerge {
            let sum = |list: String| -> anyhow::Result<(i64, String)> {
                Ok(self.db.query_row(
                    &format!(
                        "SELECT count(*), coalesce(sum(hash(t)::HUGEINT), 0)::VARCHAR FROM {} t",
                        parquet_src(&job.kind, &list)
                    ),
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )?)
            };
            let out_list = format!(
                "[{}]",
                outs.iter()
                    .map(|(a, _)| sql_str(&a.to_string_lossy()))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let a = sum(file_list_sql(&self.shared.root, &refs))?;
            let b = sum(out_list)?;
            if a != b {
                for (abs, _) in &outs {
                    let _ = std::fs::remove_file(abs);
                }
                anyhow::bail!(
                    "compaction of {} {} discarded: checksum {a:?} != {b:?}",
                    job.kind,
                    day_str(job.day)
                );
            }
        }
        let mut cat = self.shared.catalog.lock().unwrap();
        let input_paths: HashSet<&str> = job.inputs.iter().map(|f| f.path.as_str()).collect();
        let present = cat
            .files
            .iter()
            .filter(|f| input_paths.contains(f.path.as_str()))
            .count();
        let out_rows: u64 = outs.iter().map(|(_, e)| e.rows).sum();
        if present != job.inputs.len() || (!dedups && out_rows != total_rows) {
            drop(cat);
            for (abs, _) in &outs {
                let _ = std::fs::remove_file(abs);
            }
            anyhow::bail!(
                "compaction of {} {} discarded (inputs changed or row count {out_rows} != {total_rows})",
                job.kind,
                day_str(job.day)
            );
        }
        cat.files.retain(|f| !input_paths.contains(f.path.as_str()));
        if let Some(r) = rollup_entry {
            // a day's rollup is rebuilt with its samples: retire the old one
            let old: Vec<String> = cat
                .files
                .iter()
                .filter(|f| f.kind == crate::eval::rollup::KIND && f.day == job.day)
                .map(|f| f.path.clone())
                .collect();
            cat.files
                .retain(|f| !(f.kind == crate::eval::rollup::KIND && f.day == job.day));
            let after = now_ms() + self.cfg.delete_grace_s as i64 * 1000;
            for p in old {
                cat.pending_delete.push((p, after));
            }
            cat.files.push(r);
        }
        cat.files.extend(outs.into_iter().map(|(_, e)| e));
        let after = now_ms() + self.cfg.delete_grace_s as i64 * 1000;
        for p in input_paths {
            cat.pending_delete.push((p.to_string(), after));
        }
        cat.save(&self.shared.root)?;
        self.shared.generation.store(cat.generation, Ordering::Release);
        self.shared.stats.compactions.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Register closed days' series files (registry.rs), oldest day first: their series not yet
    /// in the registry go to a new registry file. A day is taken once its compaction is done
    /// (one file, or all level 2). Returns the number of files registered.
    pub fn register(&self, now: i64, all_closed: bool) -> anyhow::Result<usize> {
        let today = day_of(now);
        let todo: Vec<FileEntry> = {
            let cat = self.shared.catalog.lock().unwrap();
            let mut by_day: BTreeMap<i64, Vec<&FileEntry>> = BTreeMap::new();
            for f in cat.files.iter().filter(|f| f.kind == "series") {
                by_day.entry(f.day).or_default().push(f);
            }
            by_day
                .into_iter()
                .filter(|(day, fs)| {
                    let closed = all_closed
                        || (*day < today && now >= (day + 1) * DAY_MS + self.cfg.compact_grace_s as i64 * 1000);
                    closed && (fs.len() == 1 || fs.iter().all(|f| f.level == 2))
                })
                .flat_map(|(_, fs)| fs.into_iter().filter(|f| !f.registered).cloned())
                .collect()
        };
        let mut n = 0;
        for f in todo {
            let reg: Vec<FileEntry> = {
                let cat = self.shared.catalog.lock().unwrap();
                cat.files
                    .iter()
                    .filter(|x| x.kind == crate::registry::KIND)
                    .cloned()
                    .collect()
            };
            let regs: Vec<&FileEntry> = reg.iter().collect();
            crate::bucket::ensure_local(&self.shared.root, &[&f])?;
            // registry files are read where they are (the bucket's, unless still local)
            let fresh = crate::registry::new_ids(&crate::bucket::read_root(&self.shared.root), &[&f], &regs)?;
            let entry = if fresh.is_empty() {
                None
            } else {
                let ids: HashSet<u64> = fresh.keys().copied().collect();
                let id = {
                    let mut c = self.shared.catalog.lock().unwrap();
                    c.alloc_id()
                };
                let rel = format!("{}/date={}/G-{id:08}.parquet", crate::registry::KIND, day_str(f.day));
                let abs = self.shared.root.join(&rel);
                std::fs::create_dir_all(abs.parent().unwrap())?;
                let tmp = abs.with_extension("parquet.tmp");
                let rows =
                    crate::registry::write_subset(&self.shared.root.join(&f.path), &ids, &tmp, self.cfg.zstd_level)?;
                std::fs::File::open(&tmp)?.sync_all()?;
                std::fs::rename(&tmp, &abs)?;
                if rows != ids.len() as u64 {
                    let _ = std::fs::remove_file(&abs);
                    anyhow::bail!("registry of {}: wrote {rows} rows for {} new series", f.path, ids.len());
                }
                Some(FileEntry {
                    kind: crate::registry::KIND.into(),
                    path: rel,
                    day: f.day,
                    level: 2,
                    rows,
                    bytes: std::fs::metadata(&abs)?.len(),
                    min_ts: f.min_ts,
                    max_ts: f.max_ts,
                    created_ms: now_ms(),
                    registered: false,
                    ver: 0,
                })
            };
            let mut cat = self.shared.catalog.lock().unwrap();
            let Some(pos) = cat.files.iter().position(|x| x.path == f.path) else {
                // compacted away meanwhile: drop our file, the output is registered later
                if let Some(e) = entry {
                    let _ = std::fs::remove_file(self.shared.root.join(&e.path));
                }
                continue;
            };
            cat.files[pos].registered = true;
            if let Some(e) = entry {
                cat.files.push(e);
            }
            cat.save(&self.shared.root)?;
            self.shared.generation.store(cat.generation, Ordering::Release);
            n += 1;
        }
        Ok(n)
    }

    /// Build (or rebuild, when a day file of the month is newer) the monthly series file of every
    /// closed month whose days are all compacted: a dedupe k-way merge of the day files.
    pub fn series_months(&self, now: i64, all_closed: bool) -> anyhow::Result<usize> {
        let today = day_of(now);
        let jobs: Vec<(i64, Vec<FileEntry>, Vec<FileEntry>)> = {
            let cat = self.shared.catalog.lock().unwrap();
            let mut by_month: BTreeMap<i64, (Vec<FileEntry>, Vec<FileEntry>)> = BTreeMap::new();
            for f in &cat.files {
                if f.kind == "series" || f.kind == SERIES_MONTH {
                    let e = by_month.entry(month_of(f.day).0).or_default();
                    if f.kind == "series" {
                        e.0.push(f.clone());
                    } else {
                        e.1.push(f.clone());
                    }
                }
            }
            by_month
                .into_iter()
                .filter(|(m, (days, olds))| {
                    let (_, next) = month_of(*m);
                    let closed = all_closed || next <= today;
                    // every day compacted: one file, or only level-2 files (as registration)
                    let mut per_day: BTreeMap<i64, (usize, bool)> = BTreeMap::new();
                    for f in days {
                        let e = per_day.entry(f.day).or_insert((0, true));
                        e.0 += 1;
                        e.1 &= f.level == 2;
                    }
                    let settled = per_day.values().all(|(n, l2)| *n == 1 || *l2);
                    let newest = days.iter().map(|f| f.created_ms).max().unwrap_or(0);
                    let fresh = olds.iter().any(|o| o.created_ms >= newest);
                    closed && settled && !days.is_empty() && !fresh
                })
                .map(|(m, (days, olds))| (m, days, olds))
                .collect()
        };
        let mut n = 0;
        for (m, days, olds) in jobs {
            crate::bucket::ensure_local(&self.shared.root, &days.iter().collect::<Vec<_>>())?;
            let id = self.shared.catalog.lock().unwrap().alloc_id();
            let rel = format!("{SERIES_MONTH}/date={}/M-{id:08}.parquet", day_str(m));
            let abs = self.shared.root.join(&rel);
            std::fs::create_dir_all(abs.parent().unwrap())?;
            let tmp = abs.with_extension("parquet.tmp");
            let paths: Vec<PathBuf> = days.iter().map(|f| self.shared.root.join(&f.path)).collect();
            crate::kmerge::merge_sorted_opts(&paths, &tmp, self.cfg.zstd_level, SERIES_ROW_GROUP, true)?;
            std::fs::File::open(&tmp)?.sync_all()?;
            std::fs::rename(&tmp, &abs)?;
            let rows = crate::registry::ids_of(&abs)?.len() as u64;
            let mut cat = self.shared.catalog.lock().unwrap();
            // the inputs must still be the month's files (a compaction may have run meanwhile)
            let still = days.iter().all(|d| cat.files.iter().any(|f| f.path == d.path));
            if !still {
                drop(cat);
                let _ = std::fs::remove_file(&abs);
                continue;
            }
            let after = now_ms() + self.cfg.delete_grace_s as i64 * 1000;
            for o in &olds {
                cat.files.retain(|f| f.path != o.path);
                cat.pending_delete.push((o.path.clone(), after));
            }
            cat.files.push(FileEntry {
                kind: SERIES_MONTH.into(),
                path: rel,
                day: m,
                level: 2,
                rows,
                bytes: std::fs::metadata(&abs)?.len(),
                min_ts: days.iter().map(|f| f.min_ts).min().unwrap_or(0),
                max_ts: days.iter().map(|f| f.max_ts).max().unwrap_or(0),
                created_ms: now_ms(),
                registered: false,
                ver: 0,
            });
            cat.save(&self.shared.root)?;
            self.shared.generation.store(cat.generation, Ordering::Release);
            n += 1;
        }
        Ok(n)
    }

    fn purge(&self, now: i64) -> anyhow::Result<()> {
        let mut cat = self.shared.catalog.lock().unwrap();
        let (due, keep): (Vec<_>, Vec<_>) = cat.pending_delete.drain(..).partition(|(_, t)| *t <= now);
        cat.pending_delete = keep;
        if due.is_empty() {
            return Ok(());
        }
        cat.save(&self.shared.root)?;
        self.shared.generation.store(cat.generation, Ordering::Release);
        drop(cat);
        for (p, _) in due {
            let _ = std::fs::remove_file(self.shared.root.join(p));
        }
        Ok(())
    }
}
