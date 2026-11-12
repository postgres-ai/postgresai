//! `prom_topk_rs(k, fn, metric, matchers_json, start, end, step, window, den_metric, bottom)`
//!   -> (series_id, t, value): per step the k largest (bottom: smallest) values of
//!   fn(metric{matchers}) — or of fn(metric) / fn(den_metric) matched on identical label sets
//!   (D1's per-call panels) — with NaN dropped. Ties break like the ported D1 SQL:
//!   value, then labels['queryid'], then the label set as text. Memory: one series at a time,
//!   a k-heap per step, and (for a ratio) the denominator's grid rows.
//!
//! Named parameter `terms` (JSON `[{"m": metric, "n": a, "d": b}, ...]`, replaces `metric`): rank
//! an expression over several metrics matched one-to-one on identical label sets,
//! `Σ a_i f(m_i)`, or `Σ a_i f(m_i) / Σ b_i f(m_i)` when any b is non-zero (`x - y`, `a + b + c`,
//! `h / (r + h)`); a label set has a value at a step only when every term's series has one there.
//! The terms are read in one scan, and only per-label-set sums for the steps are held (time
//! chunks above a budget), never every (series, step) row.
use super::index::{self, Selection};
use super::kernels::{eval_series, Func, Grid};
use super::scan::{Scan, SeriesRows};
use super::{EvalCtx, EvalHandle};
use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, Mutex};

pub struct Bind {
    k: usize,
    f: Func,
    grid: Grid,
    num: super::scan::IdList,
    den: Option<super::scan::IdList>,
    bottom: bool,
    ctx: Arc<EvalCtx>,
    /// `sum by (labels)` before the top-k (empty: per series)
    by: Vec<String>,
    /// by-mode: every selected series with interned labels (groups need their key values)
    sel: Option<std::sync::Arc<Selection>>,
    /// series id -> index in sel.entries
    pos: HashMap<u64, usize>,
    /// per-series mode: ids and label hashes only (B6); labels are read for finalists only
    lean: Arc<index::IdSel>,
    /// `terms`: an expression over several metrics (num = the first term's series)
    expr: Option<Expr>,
}

/// An expression over several metrics, one-to-one on identical label sets (see the header).
pub struct Expr {
    n: Vec<f64>,
    d: Vec<f64>,
    has_den: bool,
    /// every term's series, sorted
    all: super::scan::IdList,
    /// series id -> term index
    term: HashMap<u64, usize>,
}

#[derive(serde::Deserialize)]
struct Term {
    m: String,
    #[serde(default)]
    n: f64,
    #[serde(default)]
    d: f64,
}

/// Output rows: series id, t, value, labels JSON (shared per series).
type Row = (u64, i64, f64, Arc<str>);

pub struct Init {
    st: Mutex<(Option<Vec<Row>>, usize)>,
}

/// Evaluate `f` over `ids`, calling `each(series_id, grid_row)` with NaN for absent points.
pub(super) fn for_each_series(
    ctx: &EvalCtx,
    f: Func,
    g: &Grid,
    ids: &super::scan::IdList,
    mut each: impl FnMut(u64, &[f64]),
) -> anyhow::Result<()> {
    // Large selections are split across free workers (eval/par.rs); the sink still sees the
    // series in id order.
    if let Some(plan) = super::rollup::plan(ctx, f, g) {
        return super::par::for_each_ordered(
            ids,
            g.points(),
            g.end - g.fetch_start(f),
            |sub, emit| {
                let mut row = vec![f64::NAN; g.points()];
                let mut cur: Option<u64> = None;
                super::rollup::eval(ctx, f, g, sub, plan.clone(), |id, t, y| {
                    if cur != Some(id) {
                        if let Some(c) = cur {
                            emit(c, &row);
                        }
                        row.iter_mut().for_each(|x| *x = f64::NAN);
                        cur = Some(id);
                    }
                    row[((t - g.start) / g.step) as usize] = y;
                })?;
                if let Some(c) = cur {
                    emit(c, &row);
                }
                Ok(())
            },
            &mut each,
        );
    }
    super::par::for_each_ordered(
        ids,
        g.points(),
        g.end - g.fetch_start(f),
        |sub, emit| raw_series(ctx, f, g, sub, emit),
        &mut each,
    )
}

fn raw_series(
    ctx: &EvalCtx,
    f: Func,
    g: &Grid,
    ids: &super::scan::IdList,
    each: &mut dyn FnMut(u64, &[f64]),
) -> anyhow::Result<()> {
    let t_o = std::time::Instant::now();
    let mut scan = Scan::open(&ctx.root, &ctx.files, ids.clone(), g.fetch_start(f), g.end)?;
    if std::env::var_os("PGAI_QUERY_TIMING").is_some() {
        eprintln!("query timing: scan open {} ids {:?}", ids.len(), t_o.elapsed());
    }
    let (mut rows, mut c, mut ts, mut v) = (SeriesRows::default(), vec![], vec![], vec![]);
    let mut row = vec![f64::NAN; g.points()];
    while scan.next_series(&mut rows)? {
        ts.clear();
        v.clear();
        for i in 0..rows.ts.len() {
            if !rows.v[i].is_nan() || (f.keep_stale() && rows.stale[i]) {
                ts.push(rows.ts[i]);
                v.push(rows.v[i]);
            }
        }
        row.iter_mut().for_each(|x| *x = f64::NAN);
        let mut any = false;
        eval_series(f, g, &ts, &v, &mut c, |t, y| {
            row[((t - g.start) / g.step) as usize] = y;
            any = true;
        });
        if any {
            each(rows.id, &row);
        }
    }
    Ok(())
}

struct Cand {
    v: f64,
    qid: Option<String>,
    text: Arc<str>,
    id: u64,
    bottom: bool,
}
impl Cand {
    /// "better" = ranks first: value desc (asc for bottom), queryid asc (NULLs last), text asc.
    fn rank_cmp(&self, o: &Self) -> Ordering {
        let byv = if self.bottom {
            self.v.total_cmp(&o.v)
        } else {
            o.v.total_cmp(&self.v)
        };
        byv.then_with(|| match (&self.qid, &o.qid) {
            (Some(a), Some(b)) => a.cmp(b),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        })
        .then_with(|| self.text.cmp(&o.text))
    }
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.rank_cmp(o) == Ordering::Equal
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    // max-heap of the worst kept candidate on top
    fn cmp(&self, o: &Self) -> Ordering {
        self.rank_cmp(o)
    }
}

/// topk(k, sum by (by) (f(metric))): sum per group and step, then the per-step k-heap over
/// groups. Each group is reported with a representative series id (its labels reduced to `by`
/// in SQL). Memory: groups x points.
fn compute_by(b: &Bind) -> anyhow::Result<Vec<Row>> {
    let np = b.grid.points();
    let sel = b.sel.as_ref().unwrap();
    // group key -> (representative id, sums, present)
    let mut groups: HashMap<Vec<String>, (u64, Vec<f64>, Vec<bool>)> = HashMap::new();
    let set_key = set_keys(sel);
    let lkey = |id: u64| b.pos.get(&id).map(|&pi| set_key[&sel.entries[pi].2]);
    ratio_rows(b, &lkey, &mut |id, off, row| {
        let Some(&pi) = b.pos.get(&id) else { return };
        let l = &sel.entries[pi].2;
        let key: Vec<String> = b.by.iter().map(|k| sel.label(l, k).unwrap_or("").to_string()).collect();
        let g = groups
            .entry(key)
            .or_insert_with(|| (id, vec![0.0; np], vec![false; np]));
        g.0 = g.0.min(id);
        for (j, x) in row.iter().enumerate() {
            if !x.is_nan() {
                g.1[off + j] += x;
                g.2[off + j] = true;
            }
        }
    })?;
    let mut heaps: Vec<BinaryHeap<Cand>> = (0..np).map(|_| BinaryHeap::with_capacity(b.k + 1)).collect();
    for (key, (rep, sums, has)) in &groups {
        // tie order of the ported R1: value, then the group's labels (queryid first if grouped by it)
        let qid = b.by.iter().position(|k| k == "queryid").map(|i| key[i].clone());
        let text: Arc<str> = Arc::from(
            b.by.iter()
                .zip(key)
                .filter(|(_, v)| !v.is_empty())
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        for j in 0..np {
            if !has[j] || sums[j].is_nan() {
                continue;
            }
            let c = Cand {
                v: sums[j],
                qid: qid.clone(),
                text: text.clone(),
                id: *rep,
                bottom: b.bottom,
            };
            let h = &mut heaps[j];
            if h.len() < b.k {
                h.push(c);
            } else if c.rank_cmp(h.peek().unwrap()) == Ordering::Less {
                h.pop();
                h.push(c);
            }
        }
    }
    let mut json: HashMap<u64, Arc<str>> = HashMap::new();
    let mut out = Vec::new();
    for (j, h) in heaps.into_iter().enumerate() {
        let t = b.grid.start + j as i64 * b.grid.step;
        for c in h.into_sorted_vec() {
            let l = json
                .entry(c.id)
                .or_insert_with(|| Arc::from(sel.json(&sel.entries[b.pos[&c.id]].2)))
                .clone();
            out.push((c.id, t, c.v, l));
        }
    }
    Ok(out)
}

/// Denominator rows in one arena: per series an (offset, len, dense) slot. A dense row is one
/// value per step (8 B); a sparse one the steps with a value (12 B each), for rows with values in
/// at most half the steps (churned series in a long window). No allocation per series.
#[derive(Default)]
struct DenRows {
    ix: Vec<u32>,
    vs: Vec<f64>,
}
#[derive(Clone, Copy)]
struct Sparse {
    off_v: u32,
    off_i: u32,
    len: u32,
    dense: bool,
}
impl DenRows {
    fn add(&mut self, row: &[f64]) -> Sparse {
        let n = row.iter().filter(|x| !x.is_nan()).count();
        let (off_v, off_i) = (self.vs.len() as u32, self.ix.len() as u32);
        let dense = n * 2 > row.len();
        if dense {
            self.vs.extend_from_slice(row);
        } else {
            for (j, x) in row.iter().enumerate() {
                if !x.is_nan() {
                    self.ix.push(j as u32);
                    self.vs.push(*x);
                }
            }
        }
        Sparse {
            off_v,
            off_i,
            len: self.vs.len() as u32 - off_v,
            dense,
        }
    }
    /// Value at step j (NaN when absent); `i` is a cursor for ascending j.
    fn at(&self, d: &Sparse, i: &mut usize, j: usize) -> f64 {
        let (ov, oi, n) = (d.off_v as usize, d.off_i as usize, d.len as usize);
        if d.dense {
            return self.vs[ov + j];
        }
        while *i < n && (self.ix[oi + *i] as usize) < j {
            *i += 1;
        }
        if *i < n && self.ix[oi + *i] as usize == j {
            self.vs[ov + *i]
        } else {
            f64::NAN
        }
    }
}

/// Denominator rows are bounded to this many bytes (as if dense): a ratio over more series x
/// steps is evaluated in time chunks.
const DEN_BUDGET: usize = 48 << 20;
/// Days of steps per chunk at most (each open day file costs a reader per column; time_chunks).
const CHUNK_DAYS: i64 = 3;

/// Like for_each_series, but with the scrape-interval estimate of each series given (`si`, from
/// the whole grid's fetched range) and samples fetched from `from`: one time chunk of a larger
/// grid evaluates exactly as in the whole grid. Raw samples only (chunked grids are the ones
/// rollups can't serve).
fn for_each_series_si(
    ctx: &EvalCtx,
    f: Func,
    g: &Grid,
    ids: &super::scan::IdList,
    si: &HashMap<u64, i64>,
    from: i64,
    mut each: impl FnMut(u64, &[f64]),
) -> anyhow::Result<()> {
    super::par::for_each_ordered(
        ids,
        g.points(),
        g.end - from,
        |sub, emit| raw_series_si(ctx, f, g, sub, si, from, emit),
        &mut each,
    )
}

fn raw_series_si(
    ctx: &EvalCtx,
    f: Func,
    g: &Grid,
    ids: &super::scan::IdList,
    si: &HashMap<u64, i64>,
    from: i64,
    each: &mut dyn FnMut(u64, &[f64]),
) -> anyhow::Result<()> {
    let mut scan = Scan::open(&ctx.root, &ctx.files, ids.clone(), from, g.end)?;
    let (mut rows, mut c, mut ts, mut v) = (SeriesRows::default(), vec![], vec![], vec![]);
    let mut row = vec![f64::NAN; g.points()];
    while scan.next_series(&mut rows)? {
        ts.clear();
        v.clear();
        for i in 0..rows.ts.len() {
            if !rows.v[i].is_nan() || (f.keep_stale() && rows.stale[i]) {
                ts.push(rows.ts[i]);
                v.push(rows.v[i]);
            }
        }
        if ts.is_empty() {
            continue;
        }
        c.clear();
        let mut adj = 0.0;
        for i in 0..ts.len() {
            if i > 0 {
                adj += super::kernels::reset_adj(v[i], v[i - 1]);
            }
            c.push(v[i] + adj);
        }
        row.iter_mut().for_each(|x| *x = f64::NAN);
        let mut any = false;
        super::kernels::eval_prepared(f, g, &ts, &v, &c, si.get(&rows.id).copied(), |t, y| {
            row[((t - g.start) / g.step) as usize] = y;
            any = true;
        });
        if any {
            each(rows.id, &row);
        }
    }
    Ok(())
}

/// Funcs whose points a time chunk of a grid evaluates as the whole grid does, given the whole
/// grid's scrape intervals and the samples a point can look back to: irate, the bare selector,
/// last_over_time and window aggregates. Not rate: with no sample in a window but one before it,
/// rate is 0 only when a sample follows anywhere up to the grid's end (has_prev), which a chunk
/// can't see; nor increase/delta and friends, which look back to the query's first sample.
fn chunkable(f: Func) -> bool {
    matches!(f, Func::Irate | Func::Last | Func::LastOverTime) || f.is_window_agg()
}

/// How a grid reads its series: `None` the grid itself; `Some` a time chunk of a larger grid,
/// with that grid's scrape intervals (`si`) and samples from `from`.
struct ChunkSrc {
    si: Arc<HashMap<u64, i64>>,
    from: i64,
}

/// for_each_series for one (chunk) grid: rollups where they serve it (RollupStream with the
/// whole grid's scrape intervals), else raw samples.
fn chunk_series(
    ctx: &EvalCtx,
    f: Func,
    g: &Grid,
    ids: &super::scan::IdList,
    src: Option<&ChunkSrc>,
    mut each: impl FnMut(u64, &[f64]),
) -> anyhow::Result<()> {
    let Some(src) = src else {
        if super::rowcache::enabled() {
            // the whole grid: shared across queries (num, den, terms) through the row cache
            let k = super::rowcache::key(ctx, f, g, ids);
            return super::rowcache::get_or_compute(k, g.points(), &mut each, |sink| {
                for_each_series(ctx, f, g, ids, |id, row| sink(id, row))
            });
        }
        return for_each_series(ctx, f, g, ids, each);
    };
    let Some(plan) = super::rollup::plan(ctx, f, g) else {
        if f.is_window_agg() {
            // no scrape intervals: the chunk's own window is all a point reads
            return for_each_series(ctx, f, g, ids, each);
        }
        return for_each_series_si(ctx, f, g, ids, &src.si, src.from, each);
    };
    super::par::for_each_ordered(
        ids,
        g.points(),
        g.end - src.from,
        |sub, emit| {
            let mut row = vec![f64::NAN; g.points()];
            let mut cur: Option<u64> = None;
            let mut st = super::rollup::RollupStream::open_with(
                ctx,
                f,
                g,
                sub,
                plan.clone(),
                Some(src.si.clone()),
                Some(src.from),
            )?;
            while st.next(|id, t, y| {
                if cur != Some(id) {
                    if let Some(c) = cur {
                        emit(c, &row);
                    }
                    row.iter_mut().for_each(|x| *x = f64::NAN);
                    cur = Some(id);
                }
                row[((t - g.start) / g.step) as usize] = y;
            })? {}
            if let Some(c) = cur {
                emit(c, &row);
            }
            Ok(())
        },
        &mut each,
    )
}

/// The time chunks a top-k evaluates b.grid in: (offset, grid) and how they read (None: one chunk,
/// the grid itself). A grid is cut into chunks (of at most CHUNK_DAYS days) when `live` series
/// x steps x `row_bytes` (rows held across the selection: a ratio's denominators, an
/// expression's sums)
/// pass the budget. Only funcs a chunk evaluates exactly (chunkable).
fn time_chunks(
    b: &Bind,
    ids: &[u64],
    live: usize,
    row_bytes: usize,
) -> anyhow::Result<(Vec<(usize, Grid)>, Option<ChunkPlan>)> {
    let np = b.grid.points();
    let whole = (vec![(0, b.grid)], None);
    // tests: PGAI_TOPK_CHUNK_ALWAYS=1 chunks every chunkable grid of >= 4 steps
    let always = knob("PGAI_TOPK_CHUNK_ALWAYS", 0) == 1;
    if !chunkable(b.f) || np < 4 {
        return Ok(whole);
    }
    let budget = knob("PGAI_TOPK_DEN_BUDGET", DEN_BUDGET);
    // Only the row budget chunks: the reader closes each file's row group before the next
    // (footer.rs RgReader) and reads only the columns it needs, so the number of files a grid
    // opens no longer drives memory, and re-opening every file per chunk cost more than it saved
    // (D2 at 30 d:1 h on a month store: 180 s chunked, 35 s whole, peak 127-143 MiB, no kills).
    let max_days = knob("PGAI_TOPK_CHUNK_DAYS", CHUNK_DAYS as usize).max(1) as i64;
    let over_budget = live.saturating_mul(np).saturating_mul(row_bytes) > budget;
    if !(always || over_budget) {
        return Ok(whole);
    }
    // exact per-series scrape intervals over the whole grid's fetched range
    let t_si = std::time::Instant::now();
    let si = super::rollup::si_map(
        &b.ctx,
        ids,
        b.grid.fetch_start(b.f),
        b.grid.end,
        CHUNK_DAYS,
        b.f.keep_stale(),
    )?;
    let maxprev = si
        .values()
        .map(|s| super::kernels::max_prev(*s))
        .max()
        .unwrap_or(0)
        .max(super::kernels::max_prev(b.grid.step));
    let per = (budget / row_bytes.max(1) / live.max(1))
        .min((max_days * crate::model::DAY_MS / b.grid.step) as usize)
        .max(2);
    if std::env::var_os("PGAI_EVAL_TRACE").is_some() {
        eprintln!(
            "topk: chunks of {per} steps, si for {} series in {:?}",
            si.len(),
            t_si.elapsed()
        );
    }
    let mut out = vec![];
    let mut c0 = 0;
    while c0 < np {
        let mut n = per.min(np - c0);
        if np - c0 - n == 1 {
            n += 1; // a one-point chunk would be an instant query (different window rules)
        }
        out.push((
            c0,
            Grid {
                start: b.grid.start + c0 as i64 * b.grid.step,
                end: b.grid.start + (c0 + n - 1) as i64 * b.grid.step,
                ..b.grid
            },
        ));
        c0 += n;
    }
    Ok((
        out,
        Some(ChunkPlan {
            si: Arc::new(si),
            maxprev,
            floor: b.grid.fetch_start(b.f),
        }),
    ))
}

/// Chunked evaluation: the whole grid's scrape intervals, the farthest previous sample any
/// series accepts, and the whole grid's first fetched ms (no chunk sees an earlier sample).
struct ChunkPlan {
    si: Arc<HashMap<u64, i64>>,
    maxprev: i64,
    floor: i64,
}

/// How chunk `g` reads: every sample one of its points can look at (its window, the 5 min of
/// silence, and the previous sample within maxprev), but nothing the whole grid would not read:
/// a bare selector's window is maxPrev(si), which may reach before the grid's fetch_start.
fn src_of(src: &Option<ChunkPlan>, b: &Bind, g: &Grid) -> Option<ChunkSrc> {
    src.as_ref().map(|s| ChunkSrc {
        si: s.si.clone(),
        from: (g.fetch_start(b.f) - s.maxprev).max(s.floor),
    })
}

/// The values a top-k ranks, per numerator series: f(num), or f(num) / f(den) paired by `key`
/// (same labels). Calls `each(id, offset, row)` with the row of steps offset.. (NaN = no point).
/// Long grids, and ratios whose denominator rows would pass DEN_BUDGET, are evaluated in time
/// chunks (time_chunks).
fn ratio_rows(
    b: &Bind,
    key: &dyn Fn(u64) -> Option<u64>,
    each: &mut dyn FnMut(u64, usize, &[f64]),
) -> anyhow::Result<()> {
    if let Some(e) = &b.expr {
        return expr_rows(b, e, key, each);
    }
    let none = super::scan::IdList::from(Vec::new());
    let den_ids = b.den.as_ref().unwrap_or(&none);
    let days = ((b.grid.end - b.grid.fetch_start(b.f)) / crate::model::DAY_MS + 1).max(1) as usize;
    let live = (den_ids.len() / days * 2).clamp(1, den_ids.len().max(1));
    let mut all: Vec<u64> = den_ids.iter().chain(b.num.iter()).copied().collect();
    all.sort_unstable();
    let (chunks, src) = time_chunks(b, &all, live, if b.den.is_some() { 8 } else { 0 })?;
    drop(all);
    let mut y = vec![f64::NAN; b.grid.points()];
    let ratio = b.den.is_some();
    let trace = std::env::var_os("PGAI_EVAL_TRACE").is_some();
    for (c0, g) in chunks {
        let td = std::time::Instant::now();
        let s = src_of(&src, b, &g);
        let mut den: HashMap<u64, Sparse> = HashMap::new();
        let mut arena = DenRows::default();
        if ratio {
            chunk_series(&b.ctx, b.f, &g, den_ids, s.as_ref(), |id, row| {
                if let Some(k) = key(id) {
                    den.insert(k, arena.add(row));
                }
            })?;
        }
        chunk_series(&b.ctx, b.f, &g, &b.num, s.as_ref(), |id, row| {
            if !ratio {
                return each(id, c0, row);
            }
            let Some(k) = key(id) else { return };
            let Some(d) = den.get(&k) else {
                return; // no matching denominator series: the join drops it
            };
            let mut di = 0;
            let y = &mut y[..row.len()];
            for (j, x) in row.iter().enumerate() {
                let dj = arena.at(d, &mut di, j);
                y[j] = if dj.is_nan() && !x.is_nan() { f64::NAN } else { x / dj };
            }
            each(id, c0, y);
        })?;
        if trace && src.is_some() {
            eprintln!(
                "topk: chunk {c0}+{} den {} rows in {:?}",
                g.points(),
                den.len(),
                td.elapsed()
            );
        }
    }
    Ok(())
}

/// Numerator and denominator (or an expression's terms) pair on identical label sets: one key
/// per distinct set of the selection.
fn set_keys(sel: &Selection) -> HashMap<&index::Labels, u64> {
    let mut set_key: HashMap<&index::Labels, u64> = HashMap::new();
    for e in &sel.entries {
        let n = set_key.len() as u64;
        set_key.entry(&e.2).or_insert(n);
    }
    set_key
}

/// Per label set and step of one (chunk) grid: Σ n_i x_i, Σ d_i x_i and the count of terms with
/// a value (17 B per label set and step).
#[derive(Default)]
struct ExprAcc {
    np: usize,
    slot: HashMap<u64, usize>,
    /// the first term's series of each label set (0: none, the set is never complete)
    rep: Vec<u64>,
    num: Vec<f64>,
    den: Vec<f64>,
    cnt: Vec<u8>,
}

impl ExprAcc {
    fn new(np: usize) -> Self {
        ExprAcc {
            np,
            ..Default::default()
        }
    }
    fn add(&mut self, e: &Expr, k: u64, id: u64, row: &[f64]) {
        let Some(&ti) = e.term.get(&id) else { return };
        let np = self.np;
        let s = *self.slot.entry(k).or_insert_with(|| {
            self.rep.push(0);
            self.num.resize(self.num.len() + np, 0.0);
            self.den.resize(self.den.len() + np, 0.0);
            self.cnt.resize(self.cnt.len() + np, 0);
            self.rep.len() - 1
        });
        if ti == 0 {
            self.rep[s] = id;
        }
        let (n, d) = (e.n[ti], e.d[ti]);
        let o = s * np;
        for (j, x) in row.iter().enumerate() {
            if !x.is_nan() {
                self.cnt[o + j] += 1;
                if n != 0.0 {
                    self.num[o + j] += n * x;
                }
                if d != 0.0 {
                    self.den[o + j] += d * x;
                }
            }
        }
    }
    /// Each complete label set's row (NaN: incomplete, or 0/0), under its first term's series id.
    fn emit(&self, e: &Expr, off: usize, each: &mut dyn FnMut(u64, usize, &[f64])) {
        let (np, nt) = (self.np, e.n.len() as u8);
        let mut y = vec![f64::NAN; np];
        let mut order: Vec<(u64, usize)> = self
            .slot
            .iter()
            .map(|(_, &s)| (self.rep[s], s))
            .filter(|x| x.0 != 0)
            .collect();
        order.sort_unstable();
        for (id, s) in order {
            let o = s * np;
            if !self.cnt[o..o + np].contains(&nt) {
                continue; // never complete: no row (a group's representative is one with values)
            }
            for (j, yj) in y.iter_mut().enumerate().take(np) {
                *yj = if self.cnt[o + j] != nt {
                    f64::NAN
                } else if e.has_den {
                    self.num[o + j] / self.den[o + j]
                } else {
                    self.num[o + j]
                };
            }
            each(id, off, &y);
        }
    }
}

/// The expression's rows per label set (see the header), one scan over every term's series, in
/// time chunks as ratio_rows (the budget: live label sets x steps x 17 B).
fn expr_rows(
    b: &Bind,
    e: &Expr,
    key: &dyn Fn(u64) -> Option<u64>,
    each: &mut dyn FnMut(u64, usize, &[f64]),
) -> anyhow::Result<()> {
    // one metric: no label hashes in the lean selection, and every id is its own label set
    let key = |id: u64| key(id).or(Some(id));
    let days = ((b.grid.end - b.grid.fetch_start(b.f)) / crate::model::DAY_MS + 1).max(1) as usize;
    let sets = (e.all.len() / e.n.len()).max(1);
    let live = (sets / days * 2).clamp(1, sets);
    let (chunks, src) = time_chunks(b, &e.all, live, 17)?;
    for (c0, g) in chunks {
        let s = src_of(&src, b, &g);
        let mut acc = ExprAcc::new(g.points());
        chunk_series(&b.ctx, b.f, &g, &e.all, s.as_ref(), |id, row| {
            if let Some(k) = key(id) {
                acc.add(e, k, id, row)
            }
        })?;
        acc.emit(e, c0, each);
    }
    Ok(())
}

/// Exact per-step k-heaps over the interned selection (small selections).
fn compute_full(b: &Bind) -> anyhow::Result<Vec<Row>> {
    let sel = b.sel.as_ref().unwrap();
    let set_key = set_keys(sel);
    let key = |id: u64| b.pos.get(&id).map(|&pi| set_key[&sel.entries[pi].2]);
    let mut heaps: Vec<BinaryHeap<Cand>> = (0..b.grid.points())
        .map(|_| BinaryHeap::with_capacity(b.k + 1))
        .collect();
    let mut cur: Option<(u64, Option<String>, Arc<str>)> = None;
    ratio_rows(b, &key, &mut |id, off, row| {
        let Some(&pi) = b.pos.get(&id) else { return };
        for (j, y) in row.iter().enumerate() {
            if y.is_nan() {
                continue;
            }
            // A value strictly worse than a full heap's worst ranks after it whatever its labels:
            // skip it before building the series' label text (most series of a refresh never
            // enter a heap, and the text cost ~2 ms per D1 call for 2,000 series).
            if let Some(w) = heaps[off + j].peek().filter(|_| heaps[off + j].len() >= b.k) {
                let worse = if b.bottom { y.total_cmp(&w.v) } else { w.v.total_cmp(y) };
                if worse == Ordering::Greater {
                    continue;
                }
            }
            if cur.as_ref().map(|c| c.0) != Some(id) {
                let l = &sel.entries[pi].2;
                cur = Some((
                    id,
                    sel.label(l, "queryid").map(|s| s.to_string()),
                    Arc::from(sel.text(l)),
                ));
            }
            let (_, qid, text) = cur.as_ref().unwrap();
            let c = Cand {
                v: *y,
                qid: qid.clone(),
                text: text.clone(),
                id,
                bottom: b.bottom,
            };
            let h = &mut heaps[off + j];
            if h.len() < b.k {
                h.push(c);
            } else if c.rank_cmp(h.peek().unwrap()) == Ordering::Less {
                h.pop();
                h.push(c);
            }
        }
    })?;
    let mut json: HashMap<u64, Arc<str>> = HashMap::new();
    let mut out = Vec::new();
    for (j, h) in heaps.into_iter().enumerate() {
        let t = b.grid.start + j as i64 * b.grid.step;
        for c in h.into_sorted_vec() {
            let l = json
                .entry(c.id)
                .or_insert_with(|| Arc::from(sel.json(&sel.entries[b.pos[&c.id]].2)))
                .clone();
            out.push((c.id, t, c.v, l));
        }
    }
    Ok(out)
}

/// Per step: the k best values so far plus every candidate tied with the k-th (phase A of the
/// lean top-k). Ties are resolved in phase B with labels, as rank_cmp does.
struct TopTies {
    b: std::collections::BTreeMap<Key, Vec<u64>>,
    n: usize,
}
#[derive(Clone, Copy)]
struct Key(f64);
impl PartialEq for Key {
    fn eq(&self, o: &Self) -> bool {
        self.0.total_cmp(&o.0) == Ordering::Equal
    }
}
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}
impl TopTies {
    fn worst(&self, bottom: bool) -> Option<Key> {
        if bottom {
            self.b.keys().next_back().copied()
        } else {
            self.b.keys().next().copied()
        }
    }
    fn push(&mut self, v: f64, id: u64, k: usize, bottom: bool) {
        if k == 0 {
            return;
        }
        let key = Key(v);
        if self.n >= k {
            let w = self.worst(bottom).unwrap();
            let c = if bottom { w.cmp(&key) } else { key.cmp(&w) }; // Greater: better than worst
            match c {
                Ordering::Less => return,
                Ordering::Equal => {
                    self.b.get_mut(&w).unwrap().push(id);
                    self.n += 1;
                    return;
                }
                Ordering::Greater => {}
            }
        }
        self.b.entry(key).or_default().push(id);
        self.n += 1;
        loop {
            let w = self.worst(bottom).unwrap();
            let wl = self.b[&w].len();
            if self.n - wl >= k {
                self.b.remove(&w);
                self.n -= wl;
            } else {
                break;
            }
        }
    }
}

/// Selections whose series files hold more rows than this (an upper bound from footers) use the
/// lean path. Below it the interned labels fit easily (~500 B/series) and k-heaps are exact with
/// no tie bookkeeping (a metric that is 0 for most series ties at every step).
const LEAN_ABOVE_ROWS: usize = 100_000;

/// Tie keys of a series (queryid, label text, labels JSON), from its sorted labels.
type TieKey = (Option<String>, Arc<str>, Arc<str>);
fn tie_key(l: &[(String, String)]) -> TieKey {
    let qid = l.iter().find(|(k, _)| k == "queryid").map(|(_, v)| v.clone());
    let text = format!(
        "{{{}}}",
        l.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(", ")
    );
    let json: serde_json::Map<String, serde_json::Value> = l
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();
    (
        qid,
        Arc::from(text),
        Arc::from(serde_json::Value::Object(json).to_string()),
    )
}

impl TopTies {
    /// Cut to the exact k best under the full order (value, queryid, labels). Safe at any time:
    /// the order is total and fixed, so a candidate outside the current k best never returns.
    fn compact(&mut self, k: usize, bottom: bool, keys: &HashMap<u64, TieKey>, empty: &TieKey) {
        if self.n <= k {
            return;
        }
        let mut c: Vec<Cand> = std::mem::take(&mut self.b)
            .into_iter()
            .flat_map(|(v, ids)| ids.into_iter().map(move |id| (v.0, id)))
            .map(|(v, id)| {
                let m = keys.get(&id).unwrap_or(empty);
                Cand {
                    v,
                    qid: m.0.clone(),
                    text: m.1.clone(),
                    id,
                    bottom,
                }
            })
            .collect();
        c.sort_by(|x, y| x.rank_cmp(y));
        c.truncate(k);
        self.n = c.len();
        for x in c {
            self.b.entry(Key(x.v)).or_default().push(x.id);
        }
    }
}

/// Candidates held across all steps before ties are resolved with labels and every step is cut
/// back to k (a metric that is 0 for most series ties at every step).
const COMPACT_ABOVE: usize = 300_000;

/// Thresholds, overridable for tests (PGAI_TOPK_LEAN_ABOVE, PGAI_TOPK_COMPACT_ABOVE) so the
/// oracle cases also run through the lean path and its compaction.
fn knob(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Bound on candidates kept for tie resolution across all steps (k per step plus ties).
const MAX_FINALISTS: usize = 4_000_000;

fn compute(b: &Bind) -> anyhow::Result<Vec<Row>> {
    if !b.by.is_empty() {
        return compute_by(b);
    }
    if b.sel.is_some() {
        return compute_full(b);
    }
    let ls = &b.lean;
    let trace = std::env::var("PGAI_EVAL_TRACE").is_ok();
    let t0 = std::time::Instant::now();
    let np = b.grid.points();
    let mut tops: Vec<TopTies> = (0..np)
        .map(|_| TopTies {
            b: Default::default(),
            n: 0,
        })
        .collect();
    let mut kept: isize = 0; // candidates held across all steps
    let mut too_many = false;
    let mut keys: HashMap<u64, TieKey> = HashMap::new();
    let empty: TieKey = (None, Arc::from("{}"), Arc::from("{}"));
    let mut err: Option<anyhow::Error> = None;
    let mut compactions = 0;
    let compact_above = knob("PGAI_TOPK_COMPACT_ABOVE", COMPACT_ABOVE);
    let key = |id: u64| ls.lhash(id);
    ratio_rows(b, &key, &mut |id, off, row| {
        for (j, y) in row.iter().enumerate() {
            if y.is_nan() {
                continue;
            }
            let j = off + j;
            let before = tops[j].n as isize;
            tops[j].push(*y, id, b.k, b.bottom);
            kept += tops[j].n as isize - before;
        }
        if kept > compact_above as isize && err.is_none() {
            // labels for the held ids we have no tie keys for yet, then cut every step to k
            let mut need: Vec<u64> = tops.iter().flat_map(|t| t.b.values().flatten().copied()).collect();
            need.sort_unstable();
            need.dedup();
            need.retain(|id| !keys.contains_key(id));
            match index::labels_of(&b.ctx, &need, b.grid.fetch_start(b.f), b.grid.end) {
                Ok(l) => {
                    for (id, l) in l {
                        keys.insert(id, tie_key(&l));
                    }
                    for t in tops.iter_mut() {
                        t.compact(b.k, b.bottom, &keys, &empty);
                    }
                    kept = tops.iter().map(|t| t.n as isize).sum();
                    compactions += 1;
                }
                Err(e) => err = Some(e),
            }
        }
        if kept > MAX_FINALISTS as isize {
            too_many = true;
        }
    })?;
    if let Some(e) = err {
        return Err(e);
    }
    anyhow::ensure!(
        !too_many,
        "prom_topk: more than {MAX_FINALISTS} candidates tied at the k-th value across steps"
    );
    // phase B: labels of the finalists only (those not fetched by a compaction), then the
    // exact order (value, queryid, labels)
    let mut fin: Vec<u64> = tops.iter().flat_map(|t| t.b.values().flatten().copied()).collect();
    fin.sort_unstable();
    fin.dedup();
    fin.retain(|id| !keys.contains_key(id));
    if trace {
        eprintln!(
            "topk: num {} series, new finalists {} after {compactions} compactions {:?}",
            b.num.len(),
            fin.len(),
            t0.elapsed()
        );
    }
    for (id, l) in index::labels_of(&b.ctx, &fin, b.grid.fetch_start(b.f), b.grid.end)? {
        keys.insert(id, tie_key(&l));
    }
    let meta = keys;
    let mut out = Vec::new();
    for (j, t) in tops.into_iter().enumerate() {
        let ts = b.grid.start + j as i64 * b.grid.step;
        let mut c: Vec<(Cand, Arc<str>)> =
            t.b.into_iter()
                .flat_map(|(v, ids)| ids.into_iter().map(move |id| (v.0, id)))
                .map(|(v, id)| {
                    let m = meta.get(&id).unwrap_or(&empty);
                    (
                        Cand {
                            v,
                            qid: m.0.clone(),
                            text: m.1.clone(),
                            id,
                            bottom: b.bottom,
                        },
                        m.2.clone(),
                    )
                })
                .collect();
        c.sort_by(|x, y| x.0.rank_cmp(&y.0));
        for (cand, l) in c.into_iter().take(b.k) {
            out.push((cand.id, ts, cand.v, l));
        }
    }
    Ok(out)
}

pub struct PromTopK;

impl VTab for PromTopK {
    type InitData = Init;
    type BindData = Bind;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        bind.add_result_column("series_id", LogicalTypeHandle::from(LogicalTypeId::UBigint));
        bind.add_result_column("t", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column("value", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("labels", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        let k = bind.get_parameter(0).to_int64().max(0) as usize;
        let f = Func::parse(&bind.get_parameter(1).to_string())?;
        let metric = bind.get_parameter(2).to_string();
        let matchers = bind.get_parameter(3).to_string();
        let grid = Grid {
            start: bind.get_parameter(4).to_int64(),
            end: bind.get_parameter(5).to_int64(),
            step: bind.get_parameter(6).to_int64(),
            window: bind.get_parameter(7).to_int64(),
            phi: 0.0,
        };
        if grid.step <= 0 {
            return Err("step must be > 0".into());
        }
        let den_metric = bind.get_parameter(8).to_string();
        let bottom = bind.get_parameter(9).to_string() == "true";
        let by_json = bind.get_parameter(10).to_string();
        let by: Vec<String> = if by_json.is_empty() || by_json == "NULL" || by_json == "null" {
            vec![]
        } else {
            serde_json::from_str(&by_json)?
        };
        let handle = unsafe { &*bind.get_extra_info::<EvalHandle>() };
        let ctx = handle.read().unwrap().clone();
        let has_den = !(den_metric.is_empty() || den_metric == "NULL");
        let terms: Vec<Term> = match bind.get_named_parameter("terms").map(|v| v.to_string()) {
            Some(j) if !(j.is_empty() || j == "NULL" || j == "null" || j == "[]") => serde_json::from_str(&j)?,
            _ => vec![],
        };
        let metrics: Vec<&str> = if !terms.is_empty() {
            terms.iter().map(|t| t.m.as_str()).collect()
        } else if has_den {
            vec![&metric, &den_metric]
        } else {
            vec![&metric]
        };
        if !by.is_empty() && has_den {
            return Err("prom_topk: p_by and p_den_metric together are not supported".into());
        }
        let expr_den = terms.iter().any(|t| t.d != 0.0);
        if !terms.is_empty() {
            if has_den {
                return Err("prom_topk: terms and p_den_metric together are not supported".into());
            }
            if !by.is_empty() && expr_den {
                return Err("prom_topk: p_by over a ratio of terms is not supported".into());
            }
            if terms.len() > 64 {
                return Err("prom_topk: at most 64 terms".into());
            }
            let mut seen = std::collections::HashSet::new();
            if !terms.iter().all(|t| seen.insert(t.m.as_str())) {
                return Err("prom_topk: a metric appears twice in terms (give it one n and d)".into());
            }
        }
        // Exact k-heaps with interned labels when the selection is small enough; the lean
        // two-phase path (ids only, labels for the finalists) for long windows under churn.
        let timing = std::env::var_os("PGAI_QUERY_TIMING").is_some();
        let t_b = std::time::Instant::now();
        let bound = index::selection_bound(&ctx, &metrics, grid.fetch_start(f), grid.end)?;
        let t_bound = t_b.elapsed();
        let (sel, pos, lean, num, den) = if !by.is_empty() || bound <= knob("PGAI_TOPK_LEAN_ABOVE", LEAN_ABOVE_ROWS) {
            let t_s = std::time::Instant::now();
            let sel = index::select_series(&ctx, &metrics, &matchers, grid.fetch_start(f), grid.end)?;
            if timing {
                eprintln!(
                    "query timing: topk bind: selection_bound {t_bound:?}, select_series {} entries {:?}",
                    sel.entries.len(),
                    t_s.elapsed()
                );
            }
            let pos: HashMap<u64, usize> = sel.entries.iter().enumerate().map(|(i, e)| (e.0, i)).collect();
            let num: super::scan::IdList = sel.ids(0).into();
            let den: Option<super::scan::IdList> = if has_den { Some(sel.ids(1).into()) } else { None };
            (Some(sel), pos, Arc::new(index::IdSel::default()), num, den)
        } else {
            let ts = std::time::Instant::now();
            let lean = index::select_ids_cached(&ctx, &metrics, &matchers, grid.fetch_start(f), grid.end)?;
            if std::env::var("PGAI_EVAL_TRACE").is_ok() {
                eprintln!("topk: select_ids {} in {:?}", lean.len(), ts.elapsed());
            }
            let num = lean.ids_of(0);
            let den = if has_den { Some(lean.ids_of(1)) } else { None };
            (None, HashMap::new(), lean, num, den)
        };
        let expr = if terms.is_empty() {
            None
        } else {
            let mut term: HashMap<u64, usize> = HashMap::new();
            for ti in 0..terms.len() {
                let ids: Vec<u64> = match &sel {
                    Some(s) => s.ids(ti),
                    None => lean.ids_of(ti).to_vec(),
                };
                term.extend(ids.into_iter().map(|id| (id, ti)));
            }
            let mut all: Vec<u64> = term.keys().copied().collect();
            all.sort_unstable();
            Some(Expr {
                n: terms.iter().map(|t| t.n).collect(),
                d: terms.iter().map(|t| t.d).collect(),
                has_den: expr_den,
                all: all.into(),
                term,
            })
        };
        Ok(Bind {
            k,
            f,
            grid,
            num,
            den,
            bottom,
            ctx,
            by,
            sel,
            pos,
            lean,
            expr,
        })
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        init.set_max_threads(1);
        Ok(Init {
            st: Mutex::new((None, 0)),
        })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        let b = func.get_bind_data();
        let mut st = func.get_init_data().st.lock().unwrap();
        if st.0.is_none() {
            let t = std::time::Instant::now();
            st.0 = Some(compute(b)?);
            if std::env::var_os("PGAI_QUERY_TIMING").is_some() {
                eprintln!(
                    "query timing: topk compute {:?} ({} num, {} den series, {} points)",
                    t.elapsed(),
                    b.num.len(),
                    b.den.as_ref().map_or(0, |d| d.len()),
                    b.grid.points()
                );
            }
        }
        let pos = st.1;
        let rows = st.0.as_ref().unwrap();
        let cap = output.flat_vector(0).capacity();
        let k = (rows.len() - pos).min(cap);
        let mut a = output.flat_vector(0);
        let mut t = output.flat_vector(1);
        let mut v = output.flat_vector(2);
        let (a, t, v) = unsafe {
            (
                a.as_mut_slice_with_len::<u64>(k),
                t.as_mut_slice_with_len::<i64>(k),
                v.as_mut_slice_with_len::<f64>(k),
            )
        };
        for (j, r) in rows[pos..pos + k].iter().enumerate() {
            a[j] = r.0;
            t[j] = r.1;
            v[j] = r.2;
        }
        let lv = output.flat_vector(3);
        for (j, r) in rows[pos..pos + k].iter().enumerate() {
            lv.insert(j, &*r.3);
        }
        st.1 = pos + k;
        output.set_len(k);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        let v = || LogicalTypeHandle::from(LogicalTypeId::Varchar);
        let i = || LogicalTypeHandle::from(LogicalTypeId::Bigint);
        Some(vec![i(), v(), v(), v(), i(), i(), i(), i(), v(), v(), v()])
    }

    fn named_parameters() -> Option<Vec<(String, LogicalTypeHandle)>> {
        Some(vec![("terms".into(), LogicalTypeHandle::from(LogicalTypeId::Varchar))])
    }
}
