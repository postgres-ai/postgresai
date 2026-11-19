//! In-memory label index over `series`: selector -> sorted series ids (+ labels).
//! Strings are interned once; a series' labels are (key id, value id) pairs, 8 bytes each,
//! so 50k series x 12 labels cost ~5 MB.
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::arrow::ProjectionMask;
use regex::Regex;
use std::sync::Arc;

/// Labels of one series as interned (key, value) ids, sorted by key string.
pub type Labels = Vec<(u32, u32)>;

/// The series one call selected, with labels interned per call (8 B per label pair; the strings
/// are shared), so a window with many series stays small.
#[derive(Default)]
pub struct Selection {
    strs: Vec<Box<str>>,
    /// (series id, index of the metric in the request, labels), sorted by id
    pub entries: Vec<(u64, usize, Labels)>,
}

impl Selection {
    pub fn str(&self, i: u32) -> &str {
        &self.strs[i as usize]
    }
    pub fn strs_len(&self) -> usize {
        self.strs.len()
    }
    /// id of an interned string (label names and values are interned once per selection)
    pub fn str_id(&self, s: &str) -> Option<u32> {
        self.strs.iter().position(|x| &**x == s).map(|i| i as u32)
    }
    pub fn label<'a>(&'a self, l: &Labels, k: &str) -> Option<&'a str> {
        l.iter().find(|(x, _)| self.str(*x) == k).map(|(_, v)| self.str(*v))
    }
    /// DuckDB's CAST(MAP AS VARCHAR): {k=v, k2=v2}, keys sorted.
    pub fn text(&self, l: &Labels) -> String {
        let parts: Vec<String> = l
            .iter()
            .map(|(k, v)| format!("{}={}", self.str(*k), self.str(*v)))
            .collect();
        format!("{{{}}}", parts.join(", "))
    }
    pub fn json(&self, l: &Labels) -> String {
        let m: serde_json::Map<String, serde_json::Value> = l
            .iter()
            .map(|(k, v)| {
                (
                    self.str(*k).to_string(),
                    serde_json::Value::String(self.str(*v).to_string()),
                )
            })
            .collect();
        serde_json::Value::Object(m).to_string()
    }
    pub fn ids(&self, metric_idx: usize) -> Vec<u64> {
        self.entries.iter().filter(|e| e.1 == metric_idx).map(|e| e.0).collect()
    }
}

pub enum MOp {
    Eq(String),
    Ne(String),
    Re(Regex),
    Nre(Regex),
}

pub struct Matcher {
    pub label: String,
    pub op: MOp,
}

impl Matcher {
    /// PromQL: a missing label matches as ''. Regexes are fully anchored (RE2 syntax).
    pub fn parse(label: &str, op: &str, value: &str) -> anyhow::Result<Self> {
        let anchored = || Regex::new(&format!("^(?:{value})$"));
        let op = match op {
            "=" => MOp::Eq(value.to_string()),
            "!=" => MOp::Ne(value.to_string()),
            "=~" => MOp::Re(anchored()?),
            "!~" => MOp::Nre(anchored()?),
            o => anyhow::bail!("unknown matcher op {o}"),
        };
        Ok(Self {
            label: label.to_string(),
            op,
        })
    }

    fn test(&self, v: &str) -> bool {
        match &self.op {
            MOp::Eq(x) => v == x,
            MOp::Ne(x) => v != x,
            MOp::Re(r) => r.is_match(v),
            MOp::Nre(r) => !r.is_match(v),
        }
    }
}

/// Matchers from the macros' JSON form: [["label","op","value"], ...].
pub fn parse_matchers(json: &str) -> anyhow::Result<Vec<Matcher>> {
    let v: Vec<Vec<String>> = if json.trim().is_empty() || json == "null" || json == "NULL" {
        vec![]
    } else {
        serde_json::from_str(json)?
    };
    v.iter()
        .map(|m| {
            anyhow::ensure!(m.len() == 3, "matcher must be [label, op, value]");
            Matcher::parse(&m[0], &m[1], &m[2])
        })
        .collect()
}

/// Series held by the selection cache in all (about 100 B each with labels interned).
const SEL_CACHE_SERIES: usize = 20_000;
/// Selections held by the cache at most. The dashboards' panels use a few hundred distinct
/// selectors per window, and a refresh walks them in order: with fewer slots than selectors an
/// LRU evicts each one just before its next use (at 64, every D2 tail call missed and re-read
/// the series files at bind, about 1.2 ms of a 2.6 ms laptop p50). SEL_CACHE_SERIES still
/// bounds the memory; an entry beyond its series is a key of a few hundred bytes.
const SEL_CACHE_ENTRIES: usize = 512;

/// A cached selection by key, moved to the most recently used end.
fn sel_cache_get(c: &mut Vec<(String, Arc<Selection>)>, key: &str) -> Option<Arc<Selection>> {
    let i = c.iter().position(|(k, _)| k == key)?;
    let e = c.remove(i);
    let s = e.1.clone();
    c.push(e);
    Some(s)
}

/// Add a selection as the most recently used and evict the least recently used past
/// SEL_CACHE_ENTRIES entries or SEL_CACHE_SERIES series (the newest entry always stays).
fn sel_cache_put(c: &mut Vec<(String, Arc<Selection>)>, key: String, s: Arc<Selection>) {
    c.push((key, s));
    let mut series: usize = c.iter().map(|(_, s)| s.entries.len()).sum();
    while c.len() > 1 && (c.len() > SEL_CACHE_ENTRIES || series > SEL_CACHE_SERIES) {
        series -= c.remove(0).1.entries.len();
    }
}

/// The series of each of `metrics` matching `matchers` that are active in [t0, t1], through the
/// context's selection cache (see `select_series_uncached`). A query binds each table function
/// twice, and the macros call prom_eval_rs and prom_series_rs with the same selector.
pub fn select_series(
    ctx: &super::EvalCtx,
    metrics: &[&str],
    matchers_json: &str,
    t0: i64,
    t1: i64,
) -> anyhow::Result<Arc<Selection>> {
    crate::readset::record(metrics, t0, t1);
    let key = format!(
        "{metrics:?}|{matchers_json}|{}|{}",
        crate::model::day_of(t0),
        crate::model::day_of(t1)
    );
    if let Some(s) = sel_cache_get(&mut ctx.sel_cache.lock().unwrap(), &key) {
        return Ok(s);
    }
    let s = Arc::new(select_series_uncached(
        ctx,
        metrics,
        &parse_matchers(matchers_json)?,
        t0,
        t1,
    )?);
    sel_cache_put(&mut ctx.sel_cache.lock().unwrap(), key, s.clone());
    Ok(s)
}

/// The series of each of `metrics` matching `matchers` that are active in [t0, t1]: read from
/// the series files of the days in that window only (each day's series files list the series with
/// samples that day), never the whole registry. Row groups are pruned by series_id: a metric's ids
/// share the high 16 bits, and series files are sorted by id. Matchers are tested before any label
pub fn select_series_uncached(
    ctx: &super::EvalCtx,
    metrics: &[&str],
    matchers: &[Matcher],
    t0: i64,
    t1: i64,
) -> anyhow::Result<Selection> {
    use arrow::array::{Array, MapArray, StringArray, UInt64Array};
    use std::collections::HashMap;
    let ranges: Vec<(u64, u64)> = metrics
        .iter()
        .map(|m| {
            let p = (xxhash_rust::xxh3::xxh3_64(m.as_bytes()) >> 48) << 48;
            (p, p | 0x0000_FFFF_FFFF_FFFF)
        })
        .collect();
    let mut sel = Selection::default();
    let mut intern: HashMap<String, u32> = HashMap::new();
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut scratch: Vec<(String, String)> = Vec::new();
    let keep = |mn: u64, mx: u64| ranges.iter().any(|(lo, hi)| mx >= *lo && mn <= *hi);
    for r in series_readers(ctx, t0, t1, &keep)? {
        let r = r?;
        for batch in r {
            let batch = batch?;
            let ids = batch.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
            let ms = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let lm = batch.column(2).as_any().downcast_ref::<MapArray>().unwrap();
            for i in 0..batch.num_rows() {
                let id = ids.value(i);
                if !ranges.iter().any(|(lo, hi)| id >= *lo && id <= *hi) {
                    continue;
                }
                let Some(mi) = metrics.iter().position(|m| *m == ms.value(i)) else {
                    continue;
                };
                if seen.contains(&id) {
                    continue;
                }
                let m = lm.value(i);
                let ks = m.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                let vs = m.column(1).as_any().downcast_ref::<StringArray>().unwrap();
                let get = |k: &str| {
                    (0..m.len())
                        .find(|&j| ks.value(j) == k)
                        .map(|j| if vs.is_null(j) { "" } else { vs.value(j) })
                };
                if !matchers.iter().all(|mt| mt.test(get(&mt.label).unwrap_or(""))) {
                    continue;
                }
                seen.insert(id);
                scratch.clear();
                for j in 0..m.len() {
                    scratch.push((
                        ks.value(j).to_string(),
                        if vs.is_null(j) {
                            String::new()
                        } else {
                            vs.value(j).to_string()
                        },
                    ));
                }
                scratch.sort();
                let mut l: Labels = Vec::with_capacity(scratch.len());
                for (k, v) in scratch.drain(..) {
                    let mut id_of = |x: String| -> u32 {
                        if let Some(i) = intern.get(&x) {
                            return *i;
                        }
                        let i = sel.strs.len() as u32;
                        sel.strs.push(x.clone().into_boxed_str());
                        intern.insert(x, i);
                        i
                    };
                    let ki = id_of(k);
                    let vi = id_of(v);
                    l.push((ki, vi));
                }
                sel.entries.push((id, mi, l));
            }
        }
    }
    sel.entries.sort_by_key(|e| e.0);
    Ok(sel)
}

/// Lean selection for top-k: no labels kept. Per series 8 bytes (the id; the metric
/// is its 16-bit prefix), plus 8 for a hash of its sorted label pairs when a denominator metric
/// must be paired with it (same labels, other metric). Sorted by id.
pub struct IdSel {
    /// sorted ids; a metric's ids are contiguous (they share its 16-bit prefix)
    ids: std::sync::Arc<[u64]>,
    /// label-set hash per id (two-metric selections only)
    hashes: Vec<u64>,
    /// metric prefixes, in request order (metric index = position of the id's prefix)
    pub prefixes: Vec<u64>,
}

impl Default for IdSel {
    fn default() -> Self {
        IdSel {
            ids: std::sync::Arc::from(Vec::new()),
            hashes: vec![],
            prefixes: vec![],
        }
    }
}

impl IdSel {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    /// The ids of one metric, as a shared range (no copy).
    pub fn ids_of(&self, metric_idx: usize) -> super::scan::IdList {
        let p = self.prefixes[metric_idx];
        let lo = self.ids.partition_point(|x| *x < p);
        let hi = self.ids.partition_point(|x| *x <= (p | 0x0000_FFFF_FFFF_FFFF));
        super::scan::IdList::range(self.ids.clone(), lo, hi)
    }
    /// label-set hash of a selected series (two-metric selections only)
    pub fn lhash(&self, id: u64) -> Option<u64> {
        if self.hashes.is_empty() {
            return None;
        }
        self.ids.binary_search(&id).ok().map(|i| self.hashes[i])
    }
}

/// The window's series files: for every closed month the window covers entirely, its monthly
/// file (one row per series active that month); day files for the rest.
pub fn window_files(ctx: &super::EvalCtx, t0: i64, t1: i64) -> Vec<&crate::catalog::FileEntry> {
    use crate::store::{month_of, SERIES_MONTH};
    let (d0, d1) = (crate::model::day_of(t0), crate::model::day_of(t1));
    let months: std::collections::HashMap<i64, &crate::catalog::FileEntry> = ctx
        .files
        .iter()
        .filter(|f| f.kind == SERIES_MONTH)
        .filter(|f| {
            let (m0, m1) = month_of(f.day);
            m0 >= d0 && m1 - 1 <= d1
        })
        .map(|f| (f.day, f))
        .collect();
    let mut out: Vec<&crate::catalog::FileEntry> = months.values().copied().collect();
    out.extend(
        ctx.files
            .iter()
            .filter(|f| f.kind == "series" && f.day >= d0 && f.day <= d1 && !months.contains_key(&month_of(f.day).0)),
    );
    out
}

/// Read the series rows of the window's files whose id passes `want_id` (row groups pruned by
/// id stats with `want_range`), calling `each(id, metric, keys, values)` per row.
fn scan_series_rows(
    ctx: &super::EvalCtx,
    t0: i64,
    t1: i64,
    want_range: &dyn Fn(u64, u64) -> bool,
    want_id: &dyn Fn(u64) -> bool,
    each: &mut dyn FnMut(u64, &str, &arrow::array::StringArray, &arrow::array::StringArray),
) -> anyhow::Result<()> {
    use arrow::array::{Array, MapArray, StringArray, UInt64Array};
    for r in series_readers(ctx, t0, t1, want_range)? {
        let r = r?;
        for batch in r {
            let batch = batch?;
            let ids = batch.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
            let ms = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let lm = batch.column(2).as_any().downcast_ref::<MapArray>().unwrap();
            for i in 0..batch.num_rows() {
                let id = ids.value(i);
                if !want_id(id) {
                    continue;
                }
                let m = lm.value(i);
                let ks = m.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                let vs = m.column(1).as_any().downcast_ref::<StringArray>().unwrap();
                each(id, ms.value(i), ks, vs);
            }
        }
    }
    Ok(())
}

/// A fixed-capacity buffer in its own anonymous mapping: freed with munmap, so its pages leave
/// the process at once. (A large Vec freed on macOS stays in RSS, and the selection's scan
/// buffer is the biggest transient allocation of a long-window top-k.)
struct MapBuf<T: Copy> {
    ptr: *mut T,
    cap: usize,
    len: usize,
}
impl<T: Copy> MapBuf<T> {
    fn new(cap: usize) -> anyhow::Result<Self> {
        let cap = cap.max(1);
        let bytes = cap * std::mem::size_of::<T>();
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        anyhow::ensure!(p != libc::MAP_FAILED, "mmap of {bytes} bytes failed");
        Ok(MapBuf {
            ptr: p as *mut T,
            cap,
            len: 0,
        })
    }
    fn push(&mut self, x: T) {
        assert!(self.len < self.cap, "MapBuf over its row-group bound");
        unsafe { self.ptr.add(self.len).write(x) };
        self.len += 1;
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl<T: Copy> Drop for MapBuf<T> {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.cap * std::mem::size_of::<T>()) };
    }
}

/// A reader over one series file: the row groups whose series_id stats `keep` admits (all of
/// them without stats), columns (series_id, metric, labels); None when no row group is kept.
fn series_reader(
    ctx: &super::EvalCtx,
    f: &crate::catalog::FileEntry,
    keep: &dyn Fn(u64, u64) -> bool,
) -> anyhow::Result<Option<(ParquetRecordBatchReader, crate::objstore::Planned)>> {
    let (b, src) = super::footer::builder_src(&ctx.root.join(&f.path))?;
    let pq = b.parquet_schema();
    let sc = pq.columns().iter().position(|c| c.name() == "series_id");
    let mut rgs = Vec::new();
    for (i, rg) in b.metadata().row_groups().iter().enumerate() {
        let k = match sc.and_then(|c| rg.column(c).statistics()) {
            Some(st) => {
                let u = |x: Option<&[u8]>| {
                    x.filter(|x| x.len() == 8)
                        .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
                };
                match (u(st.min_bytes_opt()), u(st.max_bytes_opt())) {
                    (Some(mn), Some(mx)) if mn <= mx => keep(mn, mx),
                    _ => true,
                }
            }
            None => true,
        };
        if k {
            rgs.push(i);
        }
    }
    if rgs.is_empty() {
        return Ok(None);
    }
    let names: Vec<String> = b.schema().fields().iter().map(|f| f.name().clone()).collect();
    anyhow::ensure!(
        names.iter().take(3).eq(["series_id", "metric", "labels"].iter()),
        "unexpected series schema {names:?}"
    );
    let mask = ProjectionMask::roots(pq, [0usize, 1, 2]);
    let plan = src.plan(b.metadata(), &rgs, Some(&mask));
    let r = b
        .with_row_groups(rgs)
        .with_projection(mask)
        .with_batch_size(4096)
        .build()?;
    Ok(Some((r, plan)))
}

/// Readers over the window's series files (`series_reader`). Local files open one at a time;
/// on an object store every file's reads are planned and queued up front, interleaved.
fn series_readers<'a>(
    ctx: &'a super::EvalCtx,
    t0: i64,
    t1: i64,
    keep: &'a dyn Fn(u64, u64) -> bool,
) -> anyhow::Result<Box<dyn Iterator<Item = anyhow::Result<ParquetRecordBatchReader>> + 'a>> {
    let wf = window_files(ctx, t0, t1);
    if !crate::objstore::is_remote(&ctx.root) {
        return Ok(Box::new(
            wf.into_iter()
                .filter_map(move |f| series_reader(ctx, f, keep).transpose())
                .map(|r| r.map(|x| x.0)),
        ));
    }
    super::footer::warm(&wf.iter().map(|f| ctx.root.join(&f.path)).collect::<Vec<_>>());
    let (mut out, mut plans) = (Vec::new(), Vec::new());
    for f in wf {
        if let Some((r, p)) = series_reader(ctx, f, keep)? {
            out.push(Ok(r));
            plans.push(p);
        }
    }
    crate::objstore::submit(plans);
    Ok(Box::new(out.into_iter()))
}

/// Rows in the window's series files' row groups that `want_range` keeps (footers only): an
/// upper bound on the rows a scan will pass on, to allocate once instead of doubling.
pub fn rows_upper_bound(
    ctx: &super::EvalCtx,
    t0: i64,
    t1: i64,
    want_range: &dyn Fn(u64, u64) -> bool,
) -> anyhow::Result<usize> {
    let mut n = 0usize;
    let wf = window_files(ctx, t0, t1);
    super::footer::warm(&wf.iter().map(|f| ctx.root.join(&f.path)).collect::<Vec<_>>());
    for f in wf {
        let p = ctx.root.join(&f.path);
        let arm = super::footer::metadata_of(&p)?;
        let md = arm.metadata();
        let sc = md
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .position(|c| c.name() == "series_id");
        for rg in md.row_groups() {
            let keep = match sc.and_then(|c| rg.column(c).statistics()) {
                Some(st) => {
                    let u = |x: Option<&[u8]>| {
                        x.filter(|x| x.len() == 8)
                            .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
                    };
                    match (u(st.min_bytes_opt()), u(st.max_bytes_opt())) {
                        (Some(mn), Some(mx)) if mn <= mx => want_range(mn, mx),
                        _ => true,
                    }
                }
                None => true,
            };
            if keep {
                n += rg.num_rows() as usize;
            }
        }
    }
    Ok(n)
}

/// Upper bound on the series rows a selection of `metrics` over [t0, t1] reads (footers only).
pub fn selection_bound(ctx: &super::EvalCtx, metrics: &[&str], t0: i64, t1: i64) -> anyhow::Result<usize> {
    let prefixes: Vec<u64> = metrics
        .iter()
        .map(|m| (xxhash_rust::xxh3::xxh3_64(m.as_bytes()) >> 48) << 48)
        .collect();
    rows_upper_bound(ctx, t0, t1, &|mn, mx| {
        prefixes.iter().any(|p| mx >= *p && mn <= (*p | 0x0000_FFFF_FFFF_FFFF))
    })
}

/// Ids (+ metric index, label hash) of the series of `metrics` matching `matchers` that are
/// active in [t0, t1]. Same files and pruning as select_series; dedupes as it goes (a series has
/// a row in every day it was active), so memory is ~2x the distinct series, not the rows read.
pub fn select_ids(
    ctx: &super::EvalCtx,
    metrics: &[&str],
    matchers: &[Matcher],
    t0: i64,
    t1: i64,
) -> anyhow::Result<IdSel> {
    use arrow::array::Array;
    let prefixes: Vec<u64> = metrics
        .iter()
        .map(|m| (xxhash_rust::xxh3::xxh3_64(m.as_bytes()) >> 48) << 48)
        .collect();
    anyhow::ensure!(
        prefixes.iter().collect::<std::collections::HashSet<_>>().len() == prefixes.len(),
        "metrics {metrics:?} share an id prefix"
    );
    let need_hash = metrics.len() > 1;
    let want_range = |mn: u64, mx: u64| prefixes.iter().any(|p| mx >= *p && mn <= (*p | 0x0000_FFFF_FFFF_FFFF));
    // Allocated once from the row-group bound (a doubling Vec peaked at 2.6x the final size),
    // deduped in place at the end (a series has a row in every day it was active).
    let bound = rows_upper_bound(ctx, t0, t1, &want_range)?;
    let mut rows: MapBuf<(u64, u64)> = MapBuf::new(if need_hash { bound } else { 0 })?;
    let mut ids: MapBuf<u64> = MapBuf::new(if need_hash { 0 } else { bound })?;
    let in_range = |id: u64| prefixes.contains(&((id >> 48) << 48));
    scan_series_rows(ctx, t0, t1, &want_range, &in_range, &mut |id, metric, ks, vs| {
        if !metrics.contains(&metric) {
            return;
        }
        let get = |k: &str| {
            (0..ks.len())
                .find(|&j| ks.value(j) == k)
                .map(|j| if vs.is_null(j) { "" } else { vs.value(j) })
        };
        if !matchers.iter().all(|mt| mt.test(get(&mt.label).unwrap_or(""))) {
            return;
        }
        if need_hash {
            let mut pairs: Vec<(&str, &str)> = (0..ks.len())
                .map(|j| (ks.value(j), if vs.is_null(j) { "" } else { vs.value(j) }))
                .collect();
            pairs.sort_unstable();
            let mut h = xxhash_rust::xxh3::Xxh3::new();
            for (k, v) in &pairs {
                h.update(k.as_bytes());
                h.update(&[0]);
                h.update(v.as_bytes());
                h.update(&[0]);
            }
            rows.push((id, h.digest()));
        } else {
            ids.push(id);
        }
    })?;
    let out = if need_hash {
        let r = rows.as_mut_slice();
        r.sort_unstable_by_key(|x| x.0);
        let mut n = 0;
        for i in 0..r.len() {
            if i == 0 || r[i].0 != r[n - 1].0 {
                r[n] = r[i];
                n += 1;
            }
        }
        let r = &r[..n];
        IdSel {
            ids: r.iter().map(|x| x.0).collect(),
            hashes: r.iter().map(|x| x.1).collect(),
            prefixes,
        }
    } else {
        let v = ids.as_mut_slice();
        v.sort_unstable();
        let mut n = 0;
        for i in 0..v.len() {
            if i == 0 || v[i] != v[n - 1] {
                v[n] = v[i];
                n += 1;
            }
        }
        IdSel {
            ids: v[..n].iter().copied().collect(),
            hashes: vec![],
            prefixes,
        }
    };
    drop(rows);
    drop(ids);
    Ok(out)
}

/// `select_ids` through the context's small cache (a query binds each table function twice).
pub fn select_ids_cached(
    ctx: &super::EvalCtx,
    metrics: &[&str],
    matchers_json: &str,
    t0: i64,
    t1: i64,
) -> anyhow::Result<std::sync::Arc<IdSel>> {
    crate::readset::record(metrics, t0, t1);
    let key = format!(
        "{metrics:?}|{matchers_json}|{}|{}",
        crate::model::day_of(t0),
        crate::model::day_of(t1)
    );
    if let Some((_, s)) = ctx.ids_cache.lock().unwrap().iter().find(|(k, _)| *k == key) {
        return Ok(s.clone());
    }
    let s = std::sync::Arc::new(select_ids(ctx, metrics, &parse_matchers(matchers_json)?, t0, t1)?);
    let mut c = ctx.ids_cache.lock().unwrap();
    c.push((key, s.clone()));
    if c.len() > 2 {
        c.remove(0);
    }
    Ok(s)
}

/// Labels (sorted by key) of the given series, from the window's series files: the second phase
/// of a lean top-k, for its finalists only.
pub fn labels_of(
    ctx: &super::EvalCtx,
    ids: &[u64],
    t0: i64,
    t1: i64,
) -> anyhow::Result<std::collections::HashMap<u64, Vec<(String, String)>>> {
    use arrow::array::Array;
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut out: std::collections::HashMap<u64, Vec<(String, String)>> = Default::default();
    if sorted.is_empty() {
        return Ok(out);
    }
    let any_in = |mn: u64, mx: u64| {
        let i = sorted.partition_point(|x| *x < mn);
        i < sorted.len() && sorted[i] <= mx
    };
    let mut each = |id: u64, _m: &str, ks: &arrow::array::StringArray, vs: &arrow::array::StringArray| {
        out.entry(id).or_insert_with(|| {
            let mut l: Vec<(String, String)> = (0..ks.len())
                .map(|j| {
                    (
                        ks.value(j).to_string(),
                        if vs.is_null(j) {
                            String::new()
                        } else {
                            vs.value(j).to_string()
                        },
                    )
                })
                .collect();
            l.sort();
            l
        });
    };
    // short windows: their own day files are few; long ones: the registry (each series once)
    let days = crate::model::day_of(t1) - crate::model::day_of(t0) + 1;
    if ctx.reg_files.is_empty() || days <= 31 {
        scan_series_rows(ctx, t0, t1, &any_in, &|id| sorted.binary_search(&id).is_ok(), &mut each)?;
    } else {
        // every series is in exactly one registry file (or in the tail): Bloom filters and id
        // stats skip almost all row groups, instead of re-reading every day of the window
        let files: Vec<&crate::catalog::FileEntry> = ctx.reg_files.iter().chain(ctx.tail_files.iter()).collect();
        let want: std::collections::HashSet<u64> = sorted.iter().copied().collect();
        crate::registry::scan_rows_with_ids(&ctx.root, &files, &want, &mut each)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sel(n: usize) -> Arc<Selection> {
        Arc::new(Selection {
            strs: vec![],
            entries: (0..n as u64).map(|i| (i, 0, vec![])).collect(),
        })
    }

    #[test]
    fn a_refresh_with_more_selectors_than_the_old_cap_hits_on_its_second_pass() {
        // A refresh walks its selectors in order; at 64 slots, 80 selectors all missed every pass.
        let mut c = vec![];
        let keys: Vec<String> = (0..200).map(|i| format!("sel{i}")).collect();
        for k in &keys {
            assert!(sel_cache_get(&mut c, k).is_none());
            sel_cache_put(&mut c, k.clone(), sel(10));
        }
        for k in &keys {
            assert!(sel_cache_get(&mut c, k).is_some(), "{k} missed on the second pass");
        }
    }

    #[test]
    fn the_cache_evicts_the_least_recently_used_past_either_bound() {
        // the entry bound
        let mut c = vec![];
        for i in 0..=SEL_CACHE_ENTRIES {
            sel_cache_put(&mut c, format!("k{i}"), sel(1));
        }
        assert_eq!(c.len(), SEL_CACHE_ENTRIES);
        assert!(sel_cache_get(&mut c, "k0").is_none());
        assert!(sel_cache_get(&mut c, "k1").is_some());
        // the series bound: a selection that fills it evicts everything older than itself
        sel_cache_put(&mut c, "big".into(), sel(SEL_CACHE_SERIES));
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].0, "big");
        sel_cache_put(&mut c, "huge".into(), sel(SEL_CACHE_SERIES + 1));
        assert_eq!(c.len(), 1, "the newest entry always stays");
        assert_eq!(c[0].0, "huge");
        // a hit moves an entry to the most recently used end
        let mut c = vec![];
        for k in ["a", "b", "c"] {
            sel_cache_put(&mut c, k.into(), sel(SEL_CACHE_SERIES / 3));
        }
        assert!(sel_cache_get(&mut c, "a").is_some());
        sel_cache_put(&mut c, "d".into(), sel(SEL_CACHE_SERIES / 3)); // 4 x 1/3 > the bound
        assert!(sel_cache_get(&mut c, "b").is_none());
        assert!(sel_cache_get(&mut c, "a").is_some());
    }
}
