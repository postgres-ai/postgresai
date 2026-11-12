//! Streaming scan of selected series over the catalog's sample files, series at a time.
//! Files are sorted by (series_id, ts); row groups are pruned by series_id and ts statistics;
//! a heap merges the files (days, flush files) into one (series_id, ts) stream.
use crate::catalog::FileEntry;
use arrow::array::{Array, Float64Array, Int64Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, TimeUnit};
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::arrow::ProjectionMask;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;

/// One series' samples in time order. `stale` marks staleness markers (value NaN).
#[derive(Default)]
pub struct SeriesRows {
    pub id: u64,
    pub ts: Vec<i64>, // ms
    pub v: Vec<f64>,
    pub stale: Vec<bool>,
}

pub struct Rows {
    pub sid: Vec<u64>,
    pub ts: Vec<i64>,
    pub v: Vec<f64>,
    pub stale: Vec<bool>,
}

struct Cursor {
    reader: ParquetRecordBatchReader,
    rows: Rows,
    pos: usize,
    ts_div: i64, // file ts unit -> ms
}

/// A sorted id list shared without copying: a range of an Arc'd array (one metric's ids inside a
/// two-metric selection, or a whole list).
#[derive(Clone)]
pub struct IdList {
    all: std::sync::Arc<[u64]>,
    lo: usize,
    hi: usize,
}
impl IdList {
    pub fn range(all: std::sync::Arc<[u64]>, lo: usize, hi: usize) -> Self {
        IdList { all, lo, hi }
    }
    /// Elements lo..hi of this list (no copy).
    pub fn sub(&self, lo: usize, hi: usize) -> Self {
        IdList {
            all: self.all.clone(),
            lo: self.lo + lo,
            hi: self.lo + hi,
        }
    }
}
impl std::ops::Deref for IdList {
    type Target = [u64];
    fn deref(&self) -> &[u64] {
        &self.all[self.lo..self.hi]
    }
}
impl From<Vec<u64>> for IdList {
    fn from(v: Vec<u64>) -> Self {
        let n = v.len();
        IdList {
            all: v.into(),
            lo: 0,
            hi: n,
        }
    }
}
impl From<std::sync::Arc<[u64]>> for IdList {
    fn from(a: std::sync::Arc<[u64]>) -> Self {
        let n = a.len();
        IdList { all: a, lo: 0, hi: n }
    }
}

pub struct Scan {
    ids: IdList, // sorted (shared with the caller: no copy per scan)
    t0: i64,
    t1: i64,
    cursors: Vec<Cursor>,
    heap: BinaryHeap<Reverse<(u64, i64, usize)>>,
    skip: Option<(i64, i64)>,
    st: ScanStats,
}

#[derive(Default)]
struct ScanStats {
    rgs: usize,
    rows_read: usize,
    rows_kept: usize,
    t_read: std::time::Duration,
    t_decode: std::time::Duration,
}

impl Drop for Scan {
    fn drop(&mut self) {
        if std::env::var_os("PGAI_QUERY_TIMING").is_some() && self.st.rows_read > 0 {
            eprintln!(
                "query timing: scan {} files {} row groups, rows read {} kept {}, parquet {:?}, filter {:?}",
                self.cursors.len(),
                self.st.rgs,
                self.st.rows_read,
                self.st.rows_kept,
                self.st.t_read,
                self.st.t_decode
            );
        }
    }
}

fn stat_u64(b: Option<&[u8]>) -> Option<u64> {
    b.filter(|x| x.len() == 8)
        .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
}
fn stat_i64(b: Option<&[u8]>) -> Option<i64> {
    b.filter(|x| x.len() == 8)
        .map(|x| i64::from_le_bytes(x.try_into().unwrap()))
}

impl Scan {
    /// `ids` sorted ascending; samples with t0 <= ts <= t1 (ms).
    pub fn open(root: &Path, files: &[FileEntry], ids: impl Into<IdList>, t0: i64, t1: i64) -> anyhow::Result<Self> {
        Self::open_skip(root, files, ids.into(), t0, t1, None)
    }

    /// Like `open`, without the samples in (p, q] (served from rollups instead).
    pub fn open_skip(
        root: &Path,
        files: &[FileEntry],
        ids: impl Into<IdList>,
        t0: i64,
        t1: i64,
        skip: Option<(i64, i64)>,
    ) -> anyhow::Result<Self> {
        let mut s = Scan {
            ids: ids.into(),
            t0,
            t1,
            cursors: vec![],
            heap: BinaryHeap::new(),
            skip,
            st: Default::default(),
        };
        if s.ids.is_empty() {
            return Ok(s);
        }
        let (lo, hi) = (s.ids[0], *s.ids.last().unwrap());
        let skip = s.skip;
        let files: Vec<&FileEntry> = files
            .iter()
            .filter(|f| {
                f.kind == "samples"
                    && f.max_ts >= t0
                    && f.min_ts <= t1
                    && !skip.is_some_and(|(p, q)| f.min_ts > p && f.max_ts <= q)
            })
            .collect();
        let _no_insert = super::footer::NoInsert::new(files.len() > super::footer::SCAN_CACHE_FILES);
        super::footer::warm(&files.iter().map(|f| root.join(&f.path)).collect::<Vec<_>>());
        let mut plans = Vec::new();
        for f in files {
            let (b, src) = super::footer::builder_src(&root.join(&f.path))?;
            let schema = b.schema().clone();
            let pq = b.parquet_schema();
            let want: Vec<&str> = ["series_id", "ts", "ival", "fval", "value"]
                .into_iter()
                .filter(|c| schema.index_of(c).is_ok())
                .collect();
            let mask = ProjectionMask::columns(pq, want.iter().copied());
            let ts_div = match schema.field_with_name("ts")?.data_type() {
                DataType::Timestamp(TimeUnit::Microsecond, _) => 1000,
                DataType::Timestamp(TimeUnit::Nanosecond, _) => 1_000_000,
                _ => 1,
            };
            let col = |name: &str| pq.columns().iter().position(|c| c.name() == name);
            let (sc, tc) = (col("series_id"), col("ts"));
            let md = b.metadata().clone();
            let mut rgs = Vec::new();
            for (i, rg) in md.row_groups().iter().enumerate() {
                let mut keep = true;
                if let Some(sc) = sc {
                    if let Some(st) = rg.column(sc).statistics() {
                        if let (Some(mn), Some(mx)) = (stat_u64(st.min_bytes_opt()), stat_u64(st.max_bytes_opt())) {
                            if mn <= mx {
                                // any selected id inside [mn, mx]?
                                let k = s.ids.partition_point(|x| *x < mn);
                                keep &= k < s.ids.len() && s.ids[k] <= mx && mx >= lo && mn <= hi;
                            }
                        }
                    }
                }
                if let (true, Some(tc)) = (keep, tc) {
                    if let Some(st) = rg.column(tc).statistics() {
                        if let (Some(mn), Some(mx)) = (stat_i64(st.min_bytes_opt()), stat_i64(st.max_bytes_opt())) {
                            keep &= mx / ts_div >= t0 - 1 && mn / ts_div <= t1 + 1;
                            if let Some((p, q)) = skip {
                                keep &= !(mn / ts_div > p && mx / ts_div <= q);
                            }
                        }
                    }
                }
                if keep {
                    rgs.push(i);
                }
            }
            if rgs.is_empty() {
                continue;
            }
            s.st.rgs += rgs.len();
            plans.push(src.plan(&md, &rgs, Some(&mask)));
            let reader = super::footer::builder_rgs(b, &src, &rgs)?
                .with_projection(mask)
                .with_batch_size(8192)
                .build()?;
            s.cursors.push(Cursor {
                reader,
                rows: Rows {
                    sid: vec![],
                    ts: vec![],
                    v: vec![],
                    stale: vec![],
                },
                pos: 0,
                ts_div,
            });
        }
        crate::objstore::submit(plans);
        for i in 0..s.cursors.len() {
            if s.fill(i)? {
                let c = &s.cursors[i];
                s.heap.push(Reverse((c.rows.sid[c.pos], c.rows.ts[c.pos], i)));
            }
        }
        Ok(s)
    }

    /// Load the next batch of cursor i with at least one selected row. false at EOF.
    fn fill(&mut self, i: usize) -> anyhow::Result<bool> {
        loop {
            let t = std::time::Instant::now();
            let Some(b) = self.cursors[i].reader.next() else {
                return Ok(false);
            };
            let b = b?;
            self.st.t_read += t.elapsed();
            let t = std::time::Instant::now();
            let rows = decode(&b, self.cursors[i].ts_div, Some(&self.ids), self.t0, self.t1, self.skip)?;
            self.st.t_decode += t.elapsed();
            self.st.rows_read += b.num_rows();
            self.st.rows_kept += rows.sid.len();
            if !rows.sid.is_empty() {
                let c = &mut self.cursors[i];
                c.rows = rows;
                c.pos = 0;
                return Ok(true);
            }
        }
    }

    /// The next series (ascending id) with all its samples in [t0, t1], merged across files.
    pub fn next_series(&mut self, out: &mut SeriesRows) -> anyhow::Result<bool> {
        out.ts.clear();
        out.v.clear();
        out.stale.clear();
        let Some(Reverse((sid, _, _))) = self.heap.peek().copied() else {
            return Ok(false);
        };
        out.id = sid;
        while let Some(Reverse((s, _, i))) = self.heap.peek().copied() {
            if s != sid {
                break;
            }
            self.heap.pop();
            // drain this cursor's run of `sid` (rows are sorted by sid, ts)
            loop {
                let c = &mut self.cursors[i];
                while c.pos < c.rows.sid.len() && c.rows.sid[c.pos] == sid {
                    out.ts.push(c.rows.ts[c.pos]);
                    out.v.push(c.rows.v[c.pos]);
                    out.stale.push(c.rows.stale[c.pos]);
                    c.pos += 1;
                }
                if c.pos < c.rows.sid.len() {
                    self.heap.push(Reverse((c.rows.sid[c.pos], c.rows.ts[c.pos], i)));
                    break;
                }
                if !self.fill(i)? {
                    break;
                }
                let c = &self.cursors[i];
                if c.rows.sid[0] != sid {
                    self.heap.push(Reverse((c.rows.sid[0], c.rows.ts[0], i)));
                    break;
                }
            }
        }
        // several files may hold the same series (days, flush files): restore time order
        if !out.ts.windows(2).all(|w| w[0] <= w[1]) {
            let mut idx: Vec<usize> = (0..out.ts.len()).collect();
            idx.sort_by_key(|&k| out.ts[k]);
            out.ts = idx.iter().map(|&k| out.ts[k]).collect();
            out.v = idx.iter().map(|&k| out.v[k]).collect();
            out.stale = idx.iter().map(|&k| out.stale[k]).collect();
        }
        Ok(true)
    }
}

/// End (exclusive) of the run of `sids[i]` starting at `i` (`sids` sorted).
fn run_end(sids: &[u64], i: usize) -> usize {
    let s = sids[i];
    let mut step = 1;
    let mut lo = i;
    while lo + step < sids.len() && sids[lo + step] == s {
        lo += step;
        step *= 2;
    }
    let hi = (lo + step).min(sids.len());
    lo + 1 + sids[lo + 1..hi].partition_point(|&x| x == s)
}

/// Decode a samples batch into rows: selected ids (None = all), t0 <= ts <= t1, not in `skip`
/// (an (exclusive, inclusive] range covered by rollups).
pub fn decode(
    b: &RecordBatch,
    ts_div: i64,
    ids: Option<&[u64]>,
    t0: i64,
    t1: i64,
    skip: Option<(i64, i64)>,
) -> anyhow::Result<Rows> {
    let sid = b
        .column_by_name("series_id")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| anyhow::anyhow!("series_id column"))?;
    let tsc = b.column_by_name("ts").ok_or_else(|| anyhow::anyhow!("ts column"))?;
    // Timestamp(unit) and Int64 share the physical layout: reinterpret, don't cast (a cast
    // copied the whole column per batch).
    let ts: Int64Array = match tsc.data_type() {
        DataType::Int64 => tsc.as_any().downcast_ref::<Int64Array>().unwrap().clone(),
        DataType::Timestamp(_, _) => Int64Array::from(tsc.to_data().into_builder().data_type(DataType::Int64).build()?),
        _ => arrow::compute::cast(tsc, &DataType::Int64)?
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .clone(),
    };
    let ival = b
        .column_by_name("ival")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>().cloned());
    let fval = b
        .column_by_name("fval")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>().cloned());
    let value = b
        .column_by_name("value")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>().cloned());
    if ival.is_none() && value.is_none() {
        anyhow::bail!("no value column");
    }
    let mut r = Rows {
        sid: vec![],
        ts: vec![],
        v: vec![],
        stale: vec![],
    };
    let (sids, tsv) = (sid.values(), ts.values());
    let n = b.num_rows();
    let mut i = 0;
    while i < n {
        let s = sids[i];
        // end of this series' run: gallop, then binary search (rows are sorted by sid, ts)
        let end = run_end(sids, i);
        if ids.is_none_or(|ids| ids.binary_search(&s).is_ok()) {
            // ts is sorted within a run: only the rows in [t0, t1] are visited
            let run = &tsv[i..end];
            let lo = i + run.partition_point(|&x| x / ts_div < t0);
            let hi = i + run.partition_point(|&x| x / ts_div <= t1);
            for k in lo..hi {
                let t = tsv[k] / ts_div;
                if skip.is_some_and(|(p, q)| t > p && t <= q) {
                    continue;
                }
                let (v, st) = match (&ival, &fval, &value) {
                    (Some(iv), Some(fv), _) => {
                        let (inull, fnull) = (iv.is_null(k), fv.is_null(k));
                        if !fnull {
                            (fv.value(k), !inull) // ival+fval together = staleness marker
                        } else if !inull {
                            (iv.value(k) as f64, false)
                        } else {
                            (f64::NAN, false)
                        }
                    }
                    (_, _, Some(vv)) => {
                        let x = if vv.is_null(k) { f64::NAN } else { vv.value(k) };
                        (x, x.is_nan())
                    }
                    _ => unreachable!(),
                };
                r.sid.push(s);
                r.ts.push(t);
                r.v.push(v);
                r.stale.push(st);
            }
        }
        i = end;
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, TimestampMillisecondArray};
    use std::sync::Arc;

    /// The per-row filter this replaced (piece 6): the reference for `decode`.
    fn naive(
        sid: &[u64],
        ts: &[i64],
        v: &[Option<i64>],
        f: &[Option<f64>],
        ids: Option<&[u64]>,
        t0: i64,
        t1: i64,
        skip: Option<(i64, i64)>,
    ) -> Vec<(u64, i64, u64, bool)> {
        let mut out = vec![];
        for k in 0..sid.len() {
            if ids.is_some_and(|ids| ids.binary_search(&sid[k]).is_err()) {
                continue;
            }
            let t = ts[k];
            if t < t0 || t > t1 || skip.is_some_and(|(p, q)| t > p && t <= q) {
                continue;
            }
            let (x, st) = match (v[k], f[k]) {
                (iv, Some(fv)) => (fv, iv.is_some()),
                (Some(iv), None) => (iv as f64, false),
                (None, None) => (f64::NAN, false),
            };
            out.push((sid[k], t, x.to_bits(), st));
        }
        out
    }

    #[test]
    fn decode_matches_the_row_by_row_filter() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rnd = |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        for case in 0..200 {
            let (mut sid, mut ts, mut iv, mut fv) = (vec![], vec![], vec![], vec![]);
            let mut s = rnd(5);
            for _ in 0..1 + rnd(40) {
                s += 1 + rnd(3);
                let len = 1 + rnd(if case % 3 == 0 { 300 } else { 12 });
                let mut t = rnd(1000) as i64;
                for _ in 0..len {
                    t += 1 + rnd(30) as i64;
                    sid.push(s);
                    ts.push(t);
                    match rnd(4) {
                        0 => {
                            iv.push(Some(rnd(100) as i64));
                            fv.push(None)
                        }
                        1 => {
                            iv.push(None);
                            fv.push(Some(rnd(1000) as f64 / 7.0))
                        }
                        2 => {
                            iv.push(Some(1));
                            fv.push(Some(f64::NAN))
                        }
                        _ => {
                            iv.push(None);
                            fv.push(None)
                        }
                    }
                }
            }
            let all: Vec<u64> = {
                let mut a = sid.clone();
                a.dedup();
                a
            };
            let ids: Vec<u64> = all.iter().copied().filter(|_| rnd(3) > 0).collect();
            let (t0, t1) = (rnd(3000) as i64, rnd(9000) as i64);
            let skip = (rnd(2) == 0).then(|| {
                let p = rnd(4000) as i64;
                (p, p + rnd(2000) as i64)
            });
            let batch = RecordBatch::try_from_iter(vec![
                ("series_id", Arc::new(UInt64Array::from(sid.clone())) as ArrayRef),
                ("ts", Arc::new(TimestampMillisecondArray::from(ts.clone())) as ArrayRef),
                ("ival", Arc::new(Int64Array::from(iv.clone())) as ArrayRef),
                ("fval", Arc::new(Float64Array::from(fv.clone())) as ArrayRef),
            ])
            .unwrap();
            for sel in [Some(ids.as_slice()), None] {
                let got = decode(&batch, 1, sel, t0, t1, skip).unwrap();
                let got: Vec<_> = (0..got.sid.len())
                    .map(|k| (got.sid[k], got.ts[k], got.v[k].to_bits(), got.stale[k]))
                    .collect();
                assert_eq!(got, naive(&sid, &ts, &iv, &fv, sel, t0, t1, skip), "case {case}");
            }
        }
    }

    #[test]
    fn run_end_finds_each_run() {
        let v: Vec<u64> = [1u64, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4].to_vec();
        let mut i = 0;
        let mut ends = vec![];
        while i < v.len() {
            i = run_end(&v, i);
            ends.push(i);
        }
        assert_eq!(ends, vec![3, 4, 13, 14]);
    }
}
