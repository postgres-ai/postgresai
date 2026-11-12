//! Rollups: per series and bucket (kW - W, kW] (right-closed, like PromQL windows), the first,
//! second, second-to-last and last real sample, the count, sum, min and max, the counter-reset
//! corrections inside the bucket, and where the bucket's staleness markers are. Two tiers:
//! - 1 h (`rollup_1h`), one file per closed day, built at closed-day compaction from the day's
//!   merged samples file;
//! - 1 d (`rollup_1d`), one file per month, built from the 1 h rollups (`update_1d`).
//!
//! For a grid whose window ends t and starts t - w on bucket boundaries, every quantity the
//! edge rollups read is exact from these rows: the previous sample (a bucket's last), the
//! first/second/last in the window, n (only compared with 0, 1, 2), and corrected values
//! (raw + the stored in-bucket corrections + the boundary correction recomputed between
//! buckets). So a compressed sequence (first, second, last per bucket) fed to the same kernel
//! gives the raw result exactly. NaN is not in rollups. Staleness markers are kept only as
//! positions (`a_first`, `a_last`, `last_stale`, `ivs`): the bare selector (`last`) needs the
//! last element <= t, marker or not, and the scrape interval of all its elements; every other
//! served function skips them, as the raw path does.
//!
//! Bucket sums are added per bucket, then across buckets: `sum_over_time`/`avg_over_time` can
//! differ from the raw left-to-right sum in the last bits (equal for integer-valued series).
use super::kernels::{eval_prepared, reset_adj, Func, Grid};
use super::scan::{decode, Scan, SeriesRows};
use super::EvalCtx;
use crate::catalog::FileEntry;
use crate::model::DAY_MS;
use arrow::array::{Array, BooleanArray, Float64Array, Int64Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use parquet::schema::types::ColumnPath;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const KIND: &str = "rollup_1h";
/// The 1 d tier: one file per month (FileEntry.day = the month's first day; min_ts/max_ts are
/// the (exclusive, inclusive] range of bucket ends it holds, not sample times).
pub const KIND_1D: &str = "rollup_1d";
/// The 1 h tier in month files (same buckets as the per-day `rollup_1h` files, merged per month
/// like the 1 d tier, with the same day range and freshness rule): a 30 d panel opens ~2 files
/// instead of 30 (CONSULT_S3.md on branch spike-monitoring-v2-duckdb 1.4). Served in place of the day files it covers.
pub const KIND_1HM: &str = "rollup_1h_month";
pub const B: i64 = 3_600_000;
pub const D: i64 = DAY_MS;
/// Rollup file format (catalog FileEntry.ver): 2 adds the second-to-last sample of each bucket
/// (prev_ts, prev_v), which makes irate servable; 3 adds the bucket's last (up to) 20 sample
/// intervals (`iv`), so the scrape-interval estimate comes from rollups, not a raw scan; 4 adds
/// sum/min/max and the staleness-marker positions (bare selector, *_over_time) and stale-only
/// buckets (n = 0). Rollups older than VER_EDGE are not served (`migrate-series` rebuilds them).
pub const VER: u8 = 4;
/// Oldest 1 h format served to the edge functions and irate (the columns they read).
const VER_EDGE: u8 = 3;
/// Rows per row group: a query keeps one open row group per day file (30+ for a month), and
/// arrow-rs holds a row group's column chunks while it reads them. 16k rows keep a day file's
/// footer to ~52 row groups (~80 KiB on disk, ~460 KiB parsed; 4096 rows: 430 KiB / 1.8 MB),
/// fetched and parsed per cold file on S3.
const ROLLUP_ROW_GROUP: usize = 16384;
/// Month files: ~13 open for a year.
const ROLLUP_1D_ROW_GROUP: usize = 16384;
/// 1 h month files: a month of 1 h buckets is ~30x a day file's rows; 128k-row groups keep the
/// footer (fetched and parsed per cold file on S3) to ~140 row groups, under one suffix GET.
const ROLLUP_1HM_ROW_GROUP: usize = 131072;

/// Plans served per tier (tests and debugging): [1 h only, 1 d (+ 1 h tail)].
pub static SERVED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// Plans that read at least one 1 h month file.
pub static SERVED_1HM: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default)]
pub struct Bucket {
    pub k: i64, // bucket end, ms
    /// real (non-NaN) samples; 0 for a bucket holding only staleness markers
    pub n: i64,
    pub first_ts: i64,
    pub first_v: f64,
    pub second_ts: i64,
    pub second_v: f64,
    pub last_ts: i64,
    pub last_v: f64,
    /// second-to-last sample (= first when n == 1): with last, the two samples irate needs
    pub prev_ts: i64,
    pub prev_v: f64,
    /// the last iv_n (<= 20) intervals between consecutive samples in the bucket, oldest first
    pub iv: [i32; 20],
    pub iv_n: u8,
    pub adj_12: f64,    // correction first -> second
    pub adj_inner: f64, // all corrections inside the bucket
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    /// all elements, staleness markers included (the bare selector's sample list): first and
    /// last time, whether the last is a marker, and the last (<= 20) intervals. Equal to the
    /// real-sample fields when the bucket has no marker.
    pub a_first: i64,
    pub a_last: i64,
    pub last_stale: bool,
    pub has_stale: bool,
    pub ivs: [i32; 20],
    pub ivs_n: u8,
}

fn push20(iv: &mut [i32; 20], n: &mut u8, d: i32) {
    if *n as usize == 20 {
        iv.copy_within(1.., 0);
        iv[19] = d;
    } else {
        iv[*n as usize] = d;
        *n += 1;
    }
}

impl Bucket {
    fn start(ts: i64, v: f64) -> Self {
        Bucket {
            k: bucket_end(ts),
            n: 1,
            first_ts: ts,
            first_v: v,
            second_ts: ts,
            second_v: v,
            last_ts: ts,
            last_v: v,
            prev_ts: ts,
            prev_v: v,
            sum: v,
            min: v,
            max: v,
            a_first: ts,
            a_last: ts,
            ..Default::default()
        }
    }
    fn start_stale(ts: i64) -> Self {
        Bucket {
            n: 0,
            first_v: f64::NAN,
            second_v: f64::NAN,
            last_v: f64::NAN,
            prev_v: f64::NAN,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            last_stale: true,
            has_stale: true,
            ..Bucket::start(ts, 0.0)
        }
    }
    fn push(&mut self, ts: i64, v: f64) {
        push20(&mut self.ivs, &mut self.ivs_n, (ts - self.a_last) as i32);
        self.a_last = ts;
        self.last_stale = false;
        if self.n == 0 {
            let s = Bucket::start(ts, v);
            self.set_real(&s);
            return;
        }
        let a = reset_adj(v, self.last_v);
        if self.n == 1 {
            self.second_ts = ts;
            self.second_v = v;
            self.adj_12 = a;
        }
        self.adj_inner += a;
        push20(&mut self.iv, &mut self.iv_n, (ts - self.last_ts) as i32);
        self.prev_ts = self.last_ts;
        self.prev_v = self.last_v;
        self.last_ts = ts;
        self.last_v = v;
        self.n += 1;
        self.sum += v;
        self.min = self.min.min(v);
        self.max = self.max.max(v);
    }
    fn push_stale(&mut self, ts: i64) {
        push20(&mut self.ivs, &mut self.ivs_n, (ts - self.a_last) as i32);
        self.a_last = ts;
        self.last_stale = true;
        self.has_stale = true;
    }
    /// Copy the real-sample fields of `o` (this bucket has none yet).
    fn set_real(&mut self, o: &Bucket) {
        self.n = o.n;
        self.first_ts = o.first_ts;
        self.first_v = o.first_v;
        self.second_ts = o.second_ts;
        self.second_v = o.second_v;
        self.last_ts = o.last_ts;
        self.last_v = o.last_v;
        self.prev_ts = o.prev_ts;
        self.prev_v = o.prev_v;
        self.iv = o.iv;
        self.iv_n = o.iv_n;
        self.adj_12 = o.adj_12;
        self.adj_inner = o.adj_inner;
        self.sum = o.sum;
        self.min = o.min;
        self.max = o.max;
    }
    /// Append the elements of `o`, which all come after this bucket's (the later part of the
    /// same bucket split across two day files, or the next bucket when building a coarser tier).
    fn merge(&mut self, o: &Bucket) {
        push20(&mut self.ivs, &mut self.ivs_n, (o.a_first - self.a_last) as i32);
        for i in 0..o.ivs_n as usize {
            push20(&mut self.ivs, &mut self.ivs_n, o.ivs[i]);
        }
        self.a_last = o.a_last;
        self.last_stale = o.last_stale;
        self.has_stale |= o.has_stale;
        if o.n == 0 {
            return;
        }
        if self.n == 0 {
            self.set_real(o);
            return;
        }
        push20(&mut self.iv, &mut self.iv_n, (o.first_ts - self.last_ts) as i32);
        for i in 0..o.iv_n as usize {
            push20(&mut self.iv, &mut self.iv_n, o.iv[i]);
        }
        let a = reset_adj(o.first_v, self.last_v);
        if self.n == 1 {
            self.second_ts = o.first_ts;
            self.second_v = o.first_v;
            self.adj_12 = a;
        }
        self.adj_inner += a + o.adj_inner;
        if o.n >= 2 {
            self.prev_ts = o.prev_ts;
            self.prev_v = o.prev_v;
        } else {
            self.prev_ts = self.last_ts;
            self.prev_v = self.last_v;
        }
        self.last_ts = o.last_ts;
        self.last_v = o.last_v;
        self.n += o.n;
        self.sum += o.sum;
        self.min = self.min.min(o.min);
        self.max = self.max.max(o.max);
    }
}

pub fn bucket_end(ts: i64) -> i64 {
    ceil_to(ts, B)
}

/// ceil(ts / w) * w: the end of ts's bucket (kw - w, kw]
fn ceil_to(ts: i64, w: i64) -> i64 {
    -((-ts).div_euclid(w)) * w
}

/// Rollup columns a reader decodes (empty: all). Every set keeps the key (series_id, k) and
/// what the scrape-interval estimate and merging split buckets read (n, first/last ts, iv,
/// a_first/a_last, ivs); a column left out decodes as a default its consumer never reads.
const COLS_SI: &[&str] = &["series_id", "k", "n", "first_ts", "last_ts", "iv", "a_first", "a_last", "ivs"];
const COLS_LAST: &[&str] =
    &["series_id", "k", "n", "first_ts", "last_ts", "iv", "a_first", "a_last", "ivs", "last_v", "last_stale"];
/// irate reads only each bucket's last two samples (next_edges, `narrow`), and its scrape
/// interval walks real samples only (si_walk: `iv`, never the marker list `ivs`)
const COLS_IRATE: &[&str] =
    &["series_id", "k", "n", "first_ts", "last_ts", "iv", "a_first", "a_last", "last_v", "prev_ts", "prev_v"];
const COLS_WINDOW: &[&str] = &["series_id", "k", "n", "first_ts", "last_ts", "a_first", "a_last", "v_sum", "v_min", "v_max"];
const COLS_EDGES: &[&str] = &[
    "series_id", "k", "n", "first_ts", "first_v", "second_ts", "second_v", "last_ts", "last_v", "prev_ts", "prev_v",
    "iv", "adj_12", "adj_inner", "a_first", "a_last", "ivs",
];

/// The columns RollupStream needs for `f` (next_last / next_window / next_edges).
fn cols_for(f: Func) -> &'static [&'static str] {
    if f == Func::Last {
        COLS_LAST
    } else if f.is_window_agg() {
        COLS_WINDOW
    } else if f == Func::Irate {
        COLS_IRATE
    } else {
        COLS_EDGES
    }
}

fn schema() -> Arc<Schema> {
    let f = |n: &str, t: DataType| Field::new(n, t, false);
    let list = || DataType::List(Arc::new(Field::new("item", DataType::Int32, false)));
    Arc::new(Schema::new(vec![
        f("series_id", DataType::UInt64),
        f("k", DataType::Int64),
        f("n", DataType::Int64),
        f("first_ts", DataType::Int64),
        f("first_v", DataType::Float64),
        f("second_ts", DataType::Int64),
        f("second_v", DataType::Float64),
        f("last_ts", DataType::Int64),
        f("last_v", DataType::Float64),
        f("prev_ts", DataType::Int64),
        f("prev_v", DataType::Float64),
        Field::new("iv", list(), false),
        f("adj_12", DataType::Float64),
        f("adj_inner", DataType::Float64),
        f("v_sum", DataType::Float64),
        f("v_min", DataType::Float64),
        f("v_max", DataType::Float64),
        // null unless the bucket has a staleness marker
        Field::new("a_first", DataType::Int64, true),
        Field::new("a_last", DataType::Int64, true),
        Field::new("last_stale", DataType::Boolean, true),
        Field::new("ivs", list(), true),
    ]))
}

struct Writer {
    w: ArrowWriter<std::fs::File>,
    rows: Vec<(u64, Bucket)>,
    total: u64,
}

impl Writer {
    fn new(out: &Path, zstd_level: i32, rg_rows: usize) -> anyhow::Result<Self> {
        let mut props = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(zstd_level)?))
            // statistics only where readers prune (series_id, k): every other column's min/max
            // only grew the footer (on S3 each footer is fetched and parsed per cold file)
            .set_statistics_enabled(EnabledStatistics::None)
            .set_column_statistics_enabled(ColumnPath::from("series_id"), EnabledStatistics::Chunk)
            .set_column_statistics_enabled(ColumnPath::from("k"), EnabledStatistics::Chunk)
            .set_max_row_group_row_count(Some(rg_rows));
        for c in ["k", "n", "first_ts", "second_ts", "last_ts", "prev_ts", "a_first", "a_last"] {
            props = props
                .set_column_dictionary_enabled(ColumnPath::from(c), false)
                .set_column_encoding(ColumnPath::from(c), Encoding::DELTA_BINARY_PACKED);
        }
        for c in [
            "first_v", "second_v", "last_v", "prev_v", "adj_12", "adj_inner", "v_sum", "v_min", "v_max",
        ] {
            props = props
                .set_column_dictionary_enabled(ColumnPath::from(c), false)
                .set_column_encoding(ColumnPath::from(c), Encoding::BYTE_STREAM_SPLIT);
        }
        Ok(Writer {
            w: ArrowWriter::try_new(std::fs::File::create(out)?, schema(), Some(props.build()))?,
            rows: vec![],
            total: 0,
        })
    }
    fn push(&mut self, sid: u64, b: Bucket) -> anyhow::Result<()> {
        self.rows.push((sid, b));
        self.total += 1;
        if self.rows.len() >= 8192 {
            self.flush()?;
        }
        Ok(())
    }
    fn flush(&mut self) -> anyhow::Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let r = &self.rows;
        let u =
            |f: fn(&(u64, Bucket)) -> i64| Arc::new(Int64Array::from_iter_values(r.iter().map(f))) as Arc<dyn Array>;
        let d =
            |f: fn(&(u64, Bucket)) -> f64| Arc::new(Float64Array::from_iter_values(r.iter().map(f))) as Arc<dyn Array>;
        let st = |f: fn(&Bucket) -> i64| {
            Arc::new(Int64Array::from_iter(r.iter().map(|x| x.1.has_stale.then(|| f(&x.1))))) as Arc<dyn Array>
        };
        let list = |stale_only: bool| {
            let mut lb = arrow::array::ListBuilder::new(arrow::array::Int32Builder::new())
                .with_field(Arc::new(Field::new("item", DataType::Int32, false)));
            for x in r.iter() {
                if stale_only {
                    if x.1.has_stale {
                        lb.values().append_slice(&x.1.ivs[..x.1.ivs_n as usize]);
                        lb.append(true);
                    } else {
                        lb.append_null();
                    }
                } else {
                    lb.values().append_slice(&x.1.iv[..x.1.iv_n as usize]);
                    lb.append(true);
                }
            }
            Arc::new(lb.finish()) as Arc<dyn Array>
        };
        let cols: Vec<Arc<dyn Array>> = vec![
            Arc::new(UInt64Array::from_iter_values(r.iter().map(|x| x.0))),
            u(|x| x.1.k),
            u(|x| x.1.n),
            u(|x| x.1.first_ts),
            d(|x| x.1.first_v),
            u(|x| x.1.second_ts),
            d(|x| x.1.second_v),
            u(|x| x.1.last_ts),
            d(|x| x.1.last_v),
            u(|x| x.1.prev_ts),
            d(|x| x.1.prev_v),
            list(false),
            d(|x| x.1.adj_12),
            d(|x| x.1.adj_inner),
            d(|x| x.1.sum),
            d(|x| x.1.min),
            d(|x| x.1.max),
            st(|b| b.a_first),
            st(|b| b.a_last),
            Arc::new(BooleanArray::from_iter(
                r.iter().map(|x| x.1.has_stale.then_some(x.1.last_stale)),
            )),
            list(true),
        ];
        self.w.write(&RecordBatch::try_new(schema(), cols)?)?;
        self.rows.clear();
        Ok(())
    }
    fn close(mut self) -> anyhow::Result<u64> {
        self.flush()?;
        self.w.close()?;
        Ok(self.total)
    }
}

/// Build the 1 h rollup of one samples file (sorted by series_id, ts) into `out`. Returns rows.
pub fn build(samples: &Path, out: &Path, zstd_level: i32) -> anyhow::Result<u64> {
    let b = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(samples)?)?;
    let ts_div = match b.schema().field_with_name("ts")?.data_type() {
        DataType::Timestamp(TimeUnit::Microsecond, _) => 1000,
        DataType::Timestamp(TimeUnit::Nanosecond, _) => 1_000_000,
        _ => 1,
    };
    let reader = b.with_batch_size(8192).build()?;
    let mut w = Writer::new(out, zstd_level, ROLLUP_ROW_GROUP)?;
    let mut cur: Option<(u64, Bucket)> = None;
    for batch in reader {
        let r = decode(&batch?, ts_div, None, i64::MIN, i64::MAX, None)?;
        for i in 0..r.sid.len() {
            let stale = r.stale[i];
            if r.v[i].is_nan() && !stale {
                continue; // NaN is skipped by every served function
            }
            let (s, t, v) = (r.sid[i], r.ts[i], r.v[i]);
            match &mut cur {
                Some((cs, cb)) if *cs == s && cb.k == bucket_end(t) => {
                    if stale {
                        cb.push_stale(t)
                    } else {
                        cb.push(t, v)
                    }
                }
                _ => {
                    if let Some((cs, cb)) = cur.take() {
                        w.push(cs, cb)?;
                    }
                    cur = Some((s, if stale { Bucket::start_stale(t) } else { Bucket::start(t, v) }));
                }
            }
        }
    }
    if let Some((cs, cb)) = cur {
        w.push(cs, cb)?;
    }
    w.close()
}

// ------------------------------------------------------------------ query side

/// Selections up to this many series read rollup files in FEW_SERIES_BATCH-row batches, wider
/// ones in WIDE_BATCH (measured at 30 d, peak RSS of a fresh reader: few series do best with
/// fewer, larger decodes; a wide scan holds less with small batches). Not tied to the writer's
/// row-group size: arrow sizes its buffers by the batch size.
const FEW_SERIES: usize = 256;
const FEW_SERIES_BATCH: usize = 4096;
const WIDE_BATCH: usize = 1024;

/// `Bucket::iv_n` of a bucket read without its `iv` column (a lazy cursor, RollupScan::load_iv).
const IV_LAZY: u8 = u8::MAX;
/// Rows of `iv` load_iv decodes at once from a series' first row (bounded memory with large row
/// groups; the next series usually falls in the same window).
const IV_WINDOW: usize = 8192;
/// load_iv calls that read a file's `iv` (tests: the lazy path ran).
pub static IV_LOADS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct Cursor {
    r: super::footer::RgReader,
    /// the file, and whether `iv` was left out of its projection (read on demand, load_iv)
    path: std::path::PathBuf,
    lazy: bool,
    /// load_iv: every row group's (series_id, k) ranges and rows from the footer; the series
    /// ids of the last row group it read; and that group's last window of rows (group, first
    /// row, the rows' (k, iv, iv_n))
    iv_ranges: Option<Vec<(u64, u64, i64, i64, usize)>>,
    iv_ids: Option<(usize, Vec<u64>)>,
    iv_rg: Option<(usize, usize, Vec<(i64, [i32; 20], u8)>)>,
    rows: Vec<(u64, Bucket)>,
    pos: usize,
    /// buckets with p < k <= q
    p: i64,
    q: i64,
}

/// Rollup rows of selected series (None = all), each input file cut to its own (p, q] of bucket
/// ends, merged across files in (series_id, k) order.
struct RollupScan {
    ids: Option<super::scan::IdList>,
    cursors: Vec<Cursor>,
    heap: BinaryHeap<Reverse<(u64, i64, usize)>>,
    /// the cursor each bucket of the last next_series came from (before any merge)
    src: Vec<usize>,
}

/// A row group's (min, max) of an 8-byte integer column from its chunk statistics.
fn stat_range(rg: &parquet::file::metadata::RowGroupMetaData, col: Option<usize>) -> Option<(i64, i64)> {
    let st = rg.column(col?).statistics()?;
    let u = |b: Option<&[u8]>| b.filter(|x| x.len() == 8).map(|x| i64::from_le_bytes(x.try_into().unwrap()));
    Some((u(st.min_bytes_opt())?, u(st.max_bytes_opt())?))
}

impl RollupScan {
    /// `want`: the columns to decode (empty: all; cols_for). `lazy_iv`: files whose rows all
    /// end before this time leave `iv` out (their buckets get IV_LAZY; load_iv reads it when a
    /// scrape-interval walk reaches them).
    fn open(
        root: &Path,
        inputs: &[(&FileEntry, i64, i64)],
        ids: Option<super::scan::IdList>,
        want: &[&str],
        lazy_iv: Option<i64>,
    ) -> anyhow::Result<Self> {
        let mut s = RollupScan {
            ids,
            cursors: vec![],
            heap: BinaryHeap::new(),
            src: vec![],
        };
        if s.ids.as_ref().is_some_and(|x| x.is_empty()) {
            return Ok(s);
        }
        let _no_insert = super::footer::NoInsert::new(inputs.len() > super::footer::SCAN_CACHE_FILES);
        super::footer::warm(&inputs.iter().map(|(f, _, _)| root.join(&f.path)).collect::<Vec<_>>());
        let batch = if s.ids.as_ref().is_some_and(|x| x.len() <= FEW_SERIES) {
            FEW_SERIES_BATCH
        } else {
            WIDE_BATCH
        };
        let mut plans = Vec::new();
        for &(f, p, q) in inputs {
            let (b, src) = super::footer::builder_src(&root.join(&f.path))?;
            // prune row groups by series_id (any selected id inside [min, max]) and by k
            let md = b.metadata().clone();
            let cols = b.parquet_schema().columns();
            let sc = cols.iter().position(|c| c.name() == "series_id");
            let kc = cols.iter().position(|c| c.name() == "k");
            let mut rgs = Vec::new();
            for (i, rg) in md.row_groups().iter().enumerate() {
                let mut keep = true;
                if let (Some(ids), Some((mn, mx))) = (&s.ids, stat_range(rg, sc)) {
                    let (mn, mx) = (mn as u64, mx as u64);
                    if mn <= mx {
                        let k = ids.partition_point(|x| *x < mn);
                        keep = k < ids.len() && ids[k] <= mx;
                    }
                }
                if let Some((kmn, kmx)) = stat_range(rg, kc) {
                    keep &= kmx > p && kmn <= q;
                }
                if keep {
                    rgs.push(i);
                }
            }
            if rgs.is_empty() {
                continue;
            }
            let lazy = want.contains(&"iv") && lazy_iv.is_some_and(|t| f.max_ts.min(q) < t);
            let mask = (!want.is_empty()).then(|| {
                let schema = b.schema().clone();
                let roots: Vec<usize> = want
                    .iter()
                    .filter(|c| !(lazy && **c == "iv"))
                    .filter_map(|c| schema.index_of(c).ok())
                    .collect();
                parquet::arrow::ProjectionMask::roots(b.parquet_schema(), roots)
            });
            plans.push(src.plan(&md, &rgs, mask.as_ref()));
            let r = super::footer::RgReader::new(b, src, &rgs, batch, mask)?;
            s.cursors.push(Cursor {
                r,
                path: root.join(&f.path),
                lazy,
                iv_ranges: None,
                iv_ids: None,
                iv_rg: None,
                rows: vec![],
                pos: 0,
                p,
                q,
            });
        }
        crate::objstore::submit(plans);
        for i in 0..s.cursors.len() {
            if s.fill(i)? {
                let c = &s.cursors[i];
                s.heap.push(Reverse((c.rows[0].0, c.rows[0].1.k, i)));
            }
        }
        Ok(s)
    }

    fn fill(&mut self, i: usize) -> anyhow::Result<bool> {
        let (p, q, lazy) = (self.cursors[i].p, self.cursors[i].q, self.cursors[i].lazy);
        loop {
            let Some(b) = self.cursors[i].r.next_batch()? else {
                return Ok(false);
            };
            let col = |n: &str| b.column_by_name(n).unwrap().clone();
            let sid = col("series_id");
            let sid = sid.as_any().downcast_ref::<UInt64Array>().unwrap();
            let i64c = |n: &str| col(n).as_any().downcast_ref::<Int64Array>().unwrap().clone();
            let i64o = |n: &str| b.column_by_name(n).and_then(|c| c.as_any().downcast_ref::<Int64Array>().cloned());
            let f64o = |n: &str| b.column_by_name(n).and_then(|c| c.as_any().downcast_ref::<Float64Array>().cloned());
            let listo = |n: &str| {
                b.column_by_name(n)
                    .and_then(|c| c.as_any().downcast_ref::<arrow::array::ListArray>().cloned())
            };
            let k = i64c("k");
            // columns a projection left out read as defaults (cols_for: never read downstream)
            let (n, ft, fv, st, sv, lt, lv, a12, ai) = (
                i64o("n"),
                i64o("first_ts"),
                f64o("first_v"),
                i64o("second_ts"),
                f64o("second_v"),
                i64o("last_ts"),
                f64o("last_v"),
                f64o("adj_12"),
                f64o("adj_inner"),
            );
            let gi = |c: &Option<Int64Array>, r: usize, d: i64| c.as_ref().map_or(d, |c| c.value(r));
            let gf = |c: &Option<Float64Array>, r: usize| c.as_ref().map_or(f64::NAN, |c| c.value(r));
            // older formats lack later columns (plan() serves each function only from files
            // that have what it reads)
            let (pt, pv) = (i64o("prev_ts"), f64o("prev_v"));
            let (sum, mn, mx) = (f64o("v_sum"), f64o("v_min"), f64o("v_max"));
            let (af, al) = (i64o("a_first"), i64o("a_last"));
            let ls = b
                .column_by_name("last_stale")
                .and_then(|c| c.as_any().downcast_ref::<BooleanArray>().cloned());
            let (ivc, ivsc) = (listo("iv"), listo("ivs"));
            // a list's values by offsets (ListArray::value would build an array per row)
            let flat = |l: &arrow::array::ListArray| {
                let v = l.values().as_any().downcast_ref::<arrow::array::Int32Array>().unwrap().values().clone();
                (l.value_offsets().to_vec(), v)
            };
            let (ivf, ivsf) = (ivc.as_ref().map(flat), ivsc.as_ref().map(flat));
            let take20 = |(o, v): &(Vec<i32>, arrow::buffer::ScalarBuffer<i32>), r: usize| {
                let (a, b) = (o[r] as usize, o[r + 1] as usize);
                let m = (b - a).min(20);
                let mut iv = [0i32; 20];
                iv[..m].copy_from_slice(&v[b - m..b]);
                (iv, m as u8)
            };
            let mut rows = std::mem::take(&mut self.cursors[i].rows);
            rows.clear();
            for r in 0..b.num_rows() {
                let s = sid.value(r);
                let kk = k.value(r);
                if kk <= p || kk > q || self.ids.as_ref().is_some_and(|ids| ids.binary_search(&s).is_err()) {
                    continue;
                }
                let (iv, mut iv_n) = ivf.as_ref().map_or(([0i32; 20], 0), |l| take20(l, r));
                if lazy && gi(&n, r, 0) >= 2 {
                    iv_n = IV_LAZY; // has intervals, not read yet
                }
                let mut bk = Bucket {
                    k: kk,
                    n: gi(&n, r, 0),
                    first_ts: gi(&ft, r, 0),
                    first_v: gf(&fv, r),
                    second_ts: gi(&st, r, 0),
                    second_v: gf(&sv, r),
                    last_ts: gi(&lt, r, 0),
                    last_v: gf(&lv, r),
                    prev_ts: pt.as_ref().map_or(i64::MIN, |c| c.value(r)),
                    prev_v: pv.as_ref().map_or(f64::NAN, |c| c.value(r)),
                    iv,
                    iv_n,
                    adj_12: a12.as_ref().map_or(0.0, |c| c.value(r)),
                    adj_inner: ai.as_ref().map_or(0.0, |c| c.value(r)),
                    sum: sum.as_ref().map_or(f64::NAN, |c| c.value(r)),
                    min: mn.as_ref().map_or(f64::NAN, |c| c.value(r)),
                    max: mx.as_ref().map_or(f64::NAN, |c| c.value(r)),
                    a_first: gi(&ft, r, 0),
                    a_last: gi(&lt, r, 0),
                    last_stale: false,
                    has_stale: false,
                    ivs: iv,
                    ivs_n: iv_n,
                };
                if let (Some(af), Some(al)) = (&af, &al) {
                    if af.is_valid(r) && al.is_valid(r) {
                        bk.has_stale = true;
                        bk.a_first = af.value(r);
                        bk.a_last = al.value(r);
                        bk.last_stale = ls.as_ref().is_some_and(|c| c.is_valid(r) && c.value(r));
                        if let (Some(l), Some(f)) = (ivsc.as_ref(), ivsf.as_ref()) {
                            if l.is_valid(r) {
                                (bk.ivs, bk.ivs_n) = take20(f, r);
                            }
                        }
                    }
                }
                rows.push((s, bk));
            }
            self.cursors[i].rows = rows;
            self.cursors[i].pos = 0;
            if !self.cursors[i].rows.is_empty() {
                return Ok(true);
            }
        }
    }

    /// Next series id present, with its buckets (sorted by k, split buckets merged).
    fn next_series(&mut self, out: &mut Vec<Bucket>) -> anyhow::Result<Option<u64>> {
        out.clear();
        self.src.clear();
        let Some(Reverse((sid, _, _))) = self.heap.peek().copied() else {
            return Ok(None);
        };
        // buckets straight into `out`, file by file in order of their first bucket; sorted and
        // merged only when that is not already strictly increasing in k (a bucket split across
        // day files): otherwise the sort would keep the order and nothing would merge
        while let Some(Reverse((s, _, i))) = self.heap.peek().copied() {
            if s != sid {
                break;
            }
            self.heap.pop();
            loop {
                let c = &mut self.cursors[i];
                while c.pos < c.rows.len() && c.rows[c.pos].0 == sid {
                    out.push(c.rows[c.pos].1);
                    self.src.push(i);
                    c.pos += 1;
                }
                if c.pos < c.rows.len() {
                    self.heap.push(Reverse((c.rows[c.pos].0, c.rows[c.pos].1.k, i)));
                    break;
                }
                if !self.fill(i)? {
                    break;
                }
                let c = &self.cursors[i];
                if c.rows[0].0 != sid {
                    self.heap.push(Reverse((c.rows[0].0, c.rows[0].1.k, i)));
                    break;
                }
            }
        }
        if !out.windows(2).all(|w| w[0].k < w[1].k) {
            // merging reads both pieces' intervals
            while self.load_iv(sid, out)? {}
            let mut got = std::mem::take(out);
            got.sort_by_key(|b| (b.k, b.a_first));
            for b in got {
                match out.last_mut() {
                    Some(l) if l.k == b.k => l.merge(&b),
                    _ => out.push(b),
                }
            }
        }
        Ok(Some(sid))
    }

    /// Read `iv` for the newest IV_LAZY buckets of `sid` (those of one file: the cursor of the
    /// newest such bucket) from that file; false when no bucket is lazy. `out` is what the last
    /// next_series gave for `sid`, unmerged when it holds a lazy bucket.
    fn load_iv(&mut self, sid: u64, out: &mut [Bucket]) -> anyhow::Result<bool> {
        let Some(j) = (0..out.len()).rev().find(|&j| out[j].iv_n == IV_LAZY) else {
            return Ok(false);
        };
        anyhow::ensure!(self.src.len() == out.len(), "load_iv: buckets without their cursor");
        let ci = self.src[j];
        IV_LOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let c = &mut self.cursors[ci];
        let (p, q) = (c.p, c.q);
        if c.iv_ranges.is_none() {
            let b = super::footer::builder(&c.path)?;
            let md = b.metadata();
            let cols = b.parquet_schema().columns();
            let (sc, kc) = (
                cols.iter().position(|x| x.name() == "series_id"),
                cols.iter().position(|x| x.name() == "k"),
            );
            c.iv_ranges = Some(
                md.row_groups()
                    .iter()
                    .map(|rg| {
                        let (a, b) = stat_range(rg, sc).map_or((0, u64::MAX), |(a, b)| (a as u64, b as u64));
                        let (ka, kb) = stat_range(rg, kc).unwrap_or((i64::MIN, i64::MAX));
                        (a, b, ka, kb, rg.num_rows().max(0) as usize)
                    })
                    .collect(),
            );
        }
        let rgs: Vec<usize> = c
            .iv_ranges
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, &(a, b, ka, kb, _))| a <= sid && sid <= b && kb > p && ka <= q)
            .map(|(g, _)| g)
            .collect();
        for g in rgs {
            let n = c.iv_ranges.as_ref().unwrap()[g].4;
            // the row group's series ids (cheap), then `k` and `iv` of a window of rows from
            // this series' first: memory stays bounded however large the row group
            if c.iv_ids.as_ref().is_none_or(|(at, _)| *at != g) {
                let mut ids = Vec::with_capacity(n);
                for bt in read_rg(&c.path, g, &["series_id"], None, n)? {
                    ids.extend_from_slice(bt.column(0).as_any().downcast_ref::<UInt64Array>().unwrap().values());
                }
                c.iv_ids = Some((g, ids));
            }
            let ids = &c.iv_ids.as_ref().unwrap().1;
            let (a, b) = (ids.partition_point(|x| *x < sid), ids.partition_point(|x| *x <= sid));
            if a == b {
                continue;
            }
            if c.iv_rg.as_ref().is_none_or(|(at, from, rows)| *at != g || a < *from || b > *from + rows.len()) {
                // (tests: PGAI_ROLLUP_IV_WINDOW, small windows over small row groups)
                let w = std::env::var("PGAI_ROLLUP_IV_WINDOW").ok().and_then(|v| v.parse().ok()).unwrap_or(IV_WINDOW);
                let end = b.max(a + w).min(ids.len());
                let sel = parquet::arrow::arrow_reader::RowSelection::from(vec![
                    parquet::arrow::arrow_reader::RowSelector::skip(a),
                    parquet::arrow::arrow_reader::RowSelector::select(end - a),
                ]);
                let mut rows = Vec::with_capacity(end - a);
                for bt in read_rg(&c.path, g, &["k", "iv"], Some(sel), end - a)? {
                    let ks = bt.column_by_name("k").unwrap().as_any().downcast_ref::<Int64Array>().unwrap().clone();
                    let l = bt
                        .column_by_name("iv")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<arrow::array::ListArray>()
                        .unwrap()
                        .clone();
                    let o = l.value_offsets();
                    let v = l.values().as_any().downcast_ref::<arrow::array::Int32Array>().unwrap().values().clone();
                    for r in 0..bt.num_rows() {
                        let (x, e) = (o[r] as usize, o[r + 1] as usize);
                        let m = (e - x).min(20);
                        let mut iv = [0i32; 20];
                        iv[..m].copy_from_slice(&v[e - m..e]);
                        rows.push((ks.value(r), iv, m as u8));
                    }
                }
                anyhow::ensure!(rows.len() == end - a, "load_iv: {} rows, want {}", rows.len(), end - a);
                c.iv_rg = Some((g, a, rows));
            }
            let (_, from, rows) = c.iv_rg.as_ref().unwrap();
            // both in k order: walk them together
            let mut it = rows[a - from..b - from].iter().filter(|x| x.0 > p && x.0 <= q).peekable();
            for (bk, _) in out.iter_mut().zip(&self.src).filter(|(_, s)| **s == ci) {
                while it.peek().is_some_and(|x| x.0 < bk.k) {
                    it.next();
                }
                if bk.iv_n == IV_LAZY && it.peek().is_some_and(|x| x.0 == bk.k) {
                    let x = it.peek().unwrap();
                    bk.iv = x.1;
                    bk.iv_n = x.2;
                    if bk.ivs_n == IV_LAZY {
                        // no marker list read: the bucket's element intervals are its samples'
                        bk.ivs = x.1;
                        bk.ivs_n = x.2;
                    }
                }
            }
        }
        // every lazy bucket of this cursor is now read (a row the file does not have: none)
        for (bk, _) in out.iter_mut().zip(&self.src).filter(|(bk, s)| **s == ci && bk.iv_n == IV_LAZY) {
            bk.iv_n = 0;
            if bk.ivs_n == IV_LAZY {
                bk.ivs_n = 0;
            }
        }
        Ok(true)
    }
}

/// The batches of row group `g` of the file at `path`: columns `cols` (in file order), the rows
/// `sel` picks (None: all), in batches of `batch` rows.
fn read_rg(
    path: &Path,
    g: usize,
    cols: &[&str],
    sel: Option<parquet::arrow::arrow_reader::RowSelection>,
    batch: usize,
) -> anyhow::Result<Vec<RecordBatch>> {
    let (b, src) = super::footer::builder_src(path)?;
    let schema = b.schema().clone();
    let roots: Vec<usize> = cols.iter().filter_map(|x| schema.index_of(x).ok()).collect();
    let mask = parquet::arrow::ProjectionMask::roots(b.parquet_schema(), roots);
    let mut b = super::footer::builder_rgs(b, &src, &[g])?.with_projection(mask).with_batch_size(batch.max(1));
    if let Some(s) = sel {
        b = b.with_row_selection(s);
    }
    Ok(b.build()?.collect::<Result<_, _>>()?)
}

/// How a function can be served from rollups.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Class {
    /// edge rollups: the window edges t and t - w must be bucket boundaries
    Edge,
    /// only t on a boundary, any window: the last one or two elements <= t decide
    GridOnly,
    /// window aggregates over (t - w, t], w = window or step: per-bucket n/sum/min/max
    WinAgg,
}

fn class(f: Func) -> Option<Class> {
    Some(match f {
        Func::Increase
        | Func::Rate
        | Func::IncreasePure
        | Func::Delta
        | Func::FirstOverTime => Class::Edge,
        // the last real sample in (t - w, t]: the last one <= t (a bucket's last, in the
        // synthetic sequence) and whether it is inside, for any w, like `last`
        Func::Irate | Func::Idelta | Func::Last | Func::Timestamp | Func::LastOverTime | Func::TlastOverTime => {
            Class::GridOnly
        }
        Func::SumOverTime | Func::MinOverTime | Func::MaxOverTime | Func::AvgOverTime | Func::CountOverTime => {
            Class::WinAgg
        }
        Func::StddevOverTime | Func::QuantileOverTime => return None,
    })
}

/// The 1 h format a function needs.
fn min_ver(f: Func) -> u8 {
    if f == Func::Last || class(f) == Some(Class::WinAgg) {
        VER
    } else {
        VER_EDGE
    }
}

/// Rollup segments in time order, contiguous: the coarsest first. (p, q] is their union, the
/// part of the fetched range read from rollups; the rest is raw.
#[derive(Clone, Debug)]
pub struct Plan {
    pub p: i64,
    pub q: i64,
    /// the tier the grid is aligned to (B or D)
    pub tier: i64,
    pub segs: Vec<(i64, i64, Vec<FileEntry>)>,
}

/// When a call can be served from rollups: the functions whose inputs are exact at bucket
/// boundaries, every window edge on a bucket boundary (a single point, or an aligned grid).
/// The 1 d tier is used when the grid (and, where it matters, the window) is on day boundaries,
/// with the 1 h tier after it to the end of the covered range.
pub fn plan(ctx: &EvalCtx, f: Func, g: &Grid) -> Option<Plan> {
    let r = plan_inner(ctx, f, g);
    if let Some(p) = &r {
        SERVED[(p.tier == D) as usize].fetch_add(1, Ordering::Relaxed);
        if p.segs.iter().any(|s| s.2.iter().any(|f| f.kind == KIND_1HM)) {
            SERVED_1HM.fetch_add(1, Ordering::Relaxed);
        }
    }
    if std::env::var_os("PGAI_EVAL_DEBUG").is_some() {
        eprintln!(
            "eval: {f:?} {g:?} rollup={:?}",
            r.as_ref()
                .map(|p| (p.tier, p.segs.iter().map(|s| (s.0, s.1, s.2.len())).collect::<Vec<_>>()))
        );
    }
    r
}

fn plan_inner(ctx: &EvalCtx, f: Func, g: &Grid) -> Option<Plan> {
    if std::env::var_os("PGAI_NO_ROLLUP").is_some() {
        return None;
    }
    let cl = class(f)?;
    if cl == Class::Edge && g.window <= 0 {
        return None;
    }
    let single = g.points() == 1;
    // the window a window aggregate uses
    let ww = if g.window > 0 { g.window } else { g.step };
    let aligned = |w: i64| -> bool {
        let grid = g.start.rem_euclid(w) == 0 && (single || g.step % w == 0);
        match cl {
            // irate/idelta: the last two samples <= t and whether each is inside (t - w, t]
            // (or, with one inside, the one before it): every bucket's last and second-to-last
            // cover that for any w. last/timestamp: the last element <= t and whether it is
            // inside. So only the grid must be aligned.
            Class::GridOnly => grid,
            // one point: the window edges fall in the raw head and tail
            Class::Edge => single || (grid && g.window % w == 0),
            Class::WinAgg => single || (grid && ww % w == 0),
        }
    };
    // where rollups may start: one point's window edge t - w must stay raw (it need not be on a
    // boundary); an aligned grid's edges are all boundaries
    let from = match cl {
        Class::GridOnly => g.fetch_start(f),
        Class::Edge if single => g.start - g.window,
        Class::WinAgg if single => g.start - ww,
        _ => g.fetch_start(f),
    };
    let ver = min_ver(f);
    // the 1 d tier: aligned grids of >= 2 points only (one point reads <= 2 raw hours at its
    // edges with the 1 h tier, against <= 2 raw days with this one)
    if !single && aligned(D) {
        if let Some(p) = plan_1d(ctx, from, g.end, ver) {
            return Some(p);
        }
    }
    if !aligned(B) {
        return None;
    }
    let (p, q, files) = coverage_v(ctx, from, g.end, ver)?;
    Some(Plan {
        p,
        q,
        tier: B,
        segs: months_1h(ctx, p, q, files),
    })
}

/// 1 d buckets for the first run the month files cover in [from, to], then 1 h buckets from its
/// end while those are valid.
fn plan_1d(ctx: &EvalCtx, from: i64, to: i64, ver: u8) -> Option<Plan> {
    let (p, q1, files) = coverage_1d(ctx, from, to)?;
    let mut segs = vec![(p, q1, files)];
    let mut q = q1;
    if let Some((p2, q2, f2)) = coverage_v(ctx, q1, to, ver) {
        if p2 == q1 {
            segs.extend(months_1h(ctx, p2, q2, f2));
            q = q2;
        }
    }
    Some(Plan { p, q, tier: D, segs })
}

/// Split a run (p, q] of valid 1 h buckets (`days`: its day rollups) into contiguous segments:
/// the buckets a current 1 h month file holds are read from it (it holds them exactly as the day
/// files do: no day it spans was recompacted after it was built), the rest from the day files.
fn months_1h(ctx: &EvalCtx, p: i64, q: i64, days: Vec<FileEntry>) -> Vec<(i64, i64, Vec<FileEntry>)> {
    if std::env::var_os("PGAI_NO_ROLLUP_1HM").is_some() {
        return vec![(p, q, days)];
    }
    let months: Vec<&FileEntry> = ctx
        .files
        .iter()
        .filter(|f| f.kind == KIND_1HM && f.ver >= VER && f.max_ts > p && f.min_ts < q)
        .collect();
    if months.is_empty() {
        return vec![(p, q, days)];
    }
    let valid = valid_days(&ctx.files, VER);
    let day = |ms: i64| crate::model::day_of(ms);
    let fresh = |m: &FileEntry, d: i64| valid.get(&d).is_some_and(|r| r.created_ms <= m.created_ms);
    let month_of = |k: i64| -> Option<usize> {
        months
            .iter()
            .position(|m| m.min_ts < k && k <= m.max_ts && fresh(m, day(k - B + 1)) && fresh(m, day(k)))
    };
    // runs of buckets by source (Some(month) or None = day files)
    let mut runs: Vec<(i64, i64, Option<usize>)> = vec![];
    let mut k = p + B;
    while k <= q {
        let src = month_of(k);
        match runs.last_mut() {
            Some(r) if r.2 == src => r.1 = k,
            _ => runs.push((k - B, k, src)),
        }
        k += B;
    }
    runs.into_iter()
        .map(|(a, b, src)| match src {
            Some(i) => (a, b, vec![months[i].clone()]),
            None => (
                a,
                b,
                days.iter().filter(|f| f.max_ts >= a - B && f.min_ts <= b + B).cloned().collect(),
            ),
        })
        .collect()
}

/// Days whose samples are exactly one closed (level 2) file and that have a 1 h rollup of at
/// least `ver`: day -> the rollup.
fn valid_days(files: &[FileEntry], ver: u8) -> HashMap<i64, &FileEntry> {
    let mut samples_per_day: HashMap<i64, (usize, bool)> = HashMap::new();
    for fe in files.iter().filter(|f| f.kind == "samples") {
        let e = samples_per_day.entry(fe.day).or_insert((0, true));
        e.0 += 1;
        e.1 &= fe.level == 2;
    }
    files
        .iter()
        .filter(|f| f.kind == KIND && f.ver >= ver)
        .filter(|f| samples_per_day.get(&f.day).map_or(false, |(n, l2)| *n == 1 && *l2))
        .map(|f| (f.day, f))
        .collect()
}

/// The first run (p, q] of at least two `w`-buckets inside [from, to] (cut to boundaries) for
/// which `ok(k)` holds.
fn first_run(from: i64, to: i64, w: i64, ok: impl Fn(i64) -> bool) -> Option<(i64, i64)> {
    let lo = ceil_to(from, w);
    let hi = to.div_euclid(w) * w;
    if hi - lo < 2 * w {
        return None;
    }
    let mut p = lo;
    while p < hi && !ok(p + w) {
        p += w;
    }
    let mut q = p;
    while q < hi && ok(q + w) {
        q += w;
    }
    (q - p >= 2 * w).then_some((p, q))
}

/// The first run (p, q] of fully valid 1 h buckets inside [from, to] (cut to bucket boundaries):
/// a bucket is valid when its days have one closed samples file and a current-format rollup.
pub fn coverage(ctx: &EvalCtx, from: i64, to: i64) -> Option<(i64, i64, Vec<FileEntry>)> {
    coverage_v(ctx, from, to, VER_EDGE)
}

fn coverage_v(ctx: &EvalCtx, from: i64, to: i64, ver: u8) -> Option<(i64, i64, Vec<FileEntry>)> {
    let valid = valid_days(&ctx.files, ver);
    let day = |ms: i64| crate::model::day_of(ms);
    // a bucket (k - B, k] is fully valid if the days of its first and last ms are valid
    let (p, q) = first_run(from, to, B, |k| {
        valid.contains_key(&day(k - B + 1)) && valid.contains_key(&day(k))
    })?;
    let files: Vec<FileEntry> = valid
        .values()
        .filter(|f| f.max_ts >= p - B && f.min_ts <= q + B)
        .map(|f| (*f).clone())
        .collect();
    Some((p, q, files))
}

/// The first run (p, q] of valid 1 d buckets inside [from, to]: in a month file's range, and
/// both days the bucket spans have valid 1 h rollups no newer than that file (a day recompacted
/// after the month file was built is not served from it).
fn coverage_1d(ctx: &EvalCtx, from: i64, to: i64) -> Option<(i64, i64, Vec<FileEntry>)> {
    let months: Vec<&FileEntry> = ctx.files.iter().filter(|f| f.kind == KIND_1D && f.ver >= VER).collect();
    if months.is_empty() {
        return None;
    }
    let valid = valid_days(&ctx.files, VER);
    let day = |ms: i64| crate::model::day_of(ms);
    let fresh = |m: &FileEntry, d: i64| valid.get(&d).is_some_and(|r| r.created_ms <= m.created_ms);
    let (p, q) = first_run(from, to, D, |k| {
        months
            .iter()
            .any(|m| m.min_ts < k && k <= m.max_ts && fresh(m, day(k - D + 1)) && fresh(m, day(k)))
    })?;
    let files = months
        .iter()
        .filter(|m| m.max_ts > p && m.min_ts < q)
        .map(|m| (*m).clone())
        .collect();
    Some((p, q, files))
}

/// The scrape-interval estimate (last up to 20 sample intervals) of a series whose samples are a
/// raw head (ts <= p), rollup buckets in (p, q] and a raw tail (ts > q), all NaN-free.
pub fn si_walk(head: &[i64], buckets: &[Bucket], tail: &[i64]) -> Option<i64> {
    si_walk_by(head, buckets, tail, false)
}

/// Same; `all`: staleness markers count as samples (the bare selector's list; head and tail
/// must then include them too).
fn si_walk_by(head: &[i64], buckets: &[Bucket], tail: &[i64], all: bool) -> Option<i64> {
    si_walk_try(head, buckets, tail, all).expect("si walk over a bucket whose intervals were not read")
}

/// si_walk_by; Err when the walk needs the intervals of an IV_LAZY bucket (RollupScan::load_iv).
fn si_walk_try(head: &[i64], buckets: &[Bucket], tail: &[i64], all: bool) -> Result<Option<i64>, ()> {
    let mut d: Vec<i64> = Vec::with_capacity(20);
    let mut cur: Option<i64> = None;
    let take = |cur: &mut Option<i64>, t: i64, d: &mut Vec<i64>| {
        if let Some(c) = *cur {
            if d.len() < 20 {
                d.push(c - t);
            }
        }
        *cur = Some(t);
    };
    for &t in tail.iter().rev() {
        take(&mut cur, t, &mut d);
    }
    for b in buckets.iter().rev() {
        if d.len() >= 20 {
            break;
        }
        let (first, last, iv) = if all {
            (b.a_first, b.a_last, &b.ivs[..b.ivs_n as usize])
        } else if b.n > 0 {
            if b.iv_n == IV_LAZY {
                take(&mut cur, b.last_ts, &mut d);
                if d.len() < 20 {
                    return Err(());
                }
                cur = Some(b.first_ts);
                continue;
            }
            (b.first_ts, b.last_ts, &b.iv[..b.iv_n as usize])
        } else {
            continue;
        };
        take(&mut cur, last, &mut d);
        for &x in iv.iter().rev() {
            if d.len() < 20 {
                d.push(x as i64);
            }
        }
        // a bucket's intervals are all known when it has <= 21 samples; otherwise 20 were taken
        cur = Some(first);
    }
    for &t in head.iter().rev() {
        if d.len() >= 20 {
            break;
        }
        take(&mut cur, t, &mut d);
    }
    Ok(super::kernels::si_of(d))
}

/// Exact scrape-interval estimates over the samples in [from, to] (what one evaluation of a grid
/// fetching that range computes per series). Read backwards in groups of `group_days` days: a
/// group that rollups cover reads their buckets (each keeps its last 20 intervals), others raw
/// samples; at most `group_days` files are open at once. A series is done once its 20 newest
/// intervals are known (ended series in the group where they lived). Lets a grid be evaluated in
/// time chunks with the estimates of the whole range.
/// `stale`: staleness markers count as samples (the bare selector's estimate, as next_last).
pub fn si_map(
    ctx: &EvalCtx,
    ids: &[u64],
    from: i64,
    to: i64,
    group_days: i64,
    stale: bool,
) -> anyhow::Result<HashMap<u64, i64>> {
    // per pending series: intervals newest first (<= 20) and the oldest sample time seen
    let mut st: HashMap<u64, (Vec<i64>, Option<i64>)> = HashMap::new();
    let add = |e: &mut (Vec<i64>, Option<i64>), t: i64| {
        if let Some(c) = e.1 {
            if e.0.len() < 20 {
                e.0.push(c - t);
            }
        }
        e.1 = Some(t);
    };
    let mut todo: Vec<u64> = ids.to_vec();
    let mut hi = to;
    let (mut rows, mut buckets) = (SeriesRows::default(), Vec::new());
    while hi >= from && !todo.is_empty() {
        let lo = ((hi.div_euclid(DAY_MS) - group_days + 1) * DAY_MS).max(from);
        let cov = coverage(ctx, lo - 1, hi).filter(|(p, q, _)| *p <= lo - 1 + B && *q >= hi - B);
        let group_ids: super::scan::IdList = std::mem::take(&mut todo).into();
        match cov {
            // rollups for (p, q], raw for the edges of the group outside it
            Some((p, q, files)) => {
                let mut raw = Scan::open_skip(&ctx.root, &ctx.files, group_ids.clone(), lo, hi, Some((p, q)))?;
                let inputs: Vec<(&FileEntry, i64, i64)> = files.iter().map(|f| (f, p, q)).collect();
                let mut ru = RollupScan::open(&ctx.root, &inputs, Some(group_ids.clone()), COLS_SI, None)?;
                let mut raw_next = if raw.next_series(&mut rows)? {
                    Some(rows.id)
                } else {
                    None
                };
                let mut ru_next = ru.next_series(&mut buckets)?;
                loop {
                    let id = match (raw_next, ru_next) {
                        (None, None) => break,
                        (Some(a), None) => a,
                        (None, Some(b)) => b,
                        (Some(a), Some(b)) => a.min(b),
                    };
                    let e = st.entry(id).or_default();
                    let raw_ts: Vec<i64> = if raw_next == Some(id) {
                        (0..rows.ts.len())
                            .filter(|&i| !rows.v[i].is_nan() || (stale && rows.stale[i]))
                            .map(|i| rows.ts[i])
                            .collect()
                    } else {
                        vec![]
                    };
                    for &t in raw_ts.iter().rev().filter(|t| **t > q) {
                        add(e, t);
                    }
                    if ru_next == Some(id) {
                        for b in buckets.iter().rev() {
                            if e.0.len() >= 20 {
                                break;
                            }
                            // as si_walk_by: markers count with `stale`
                            let (first, last, iv) = if stale {
                                (b.a_first, b.a_last, &b.ivs[..b.ivs_n as usize])
                            } else if b.n > 0 {
                                (b.first_ts, b.last_ts, &b.iv[..b.iv_n as usize])
                            } else {
                                continue;
                            };
                            add(e, last);
                            for &x in iv.iter().rev() {
                                if e.0.len() < 20 {
                                    e.0.push(x as i64);
                                }
                            }
                            e.1 = Some(first);
                        }
                    }
                    for &t in raw_ts.iter().rev().filter(|t| **t <= p) {
                        add(e, t);
                    }
                    if raw_next == Some(id) {
                        raw_next = if raw.next_series(&mut rows)? {
                            Some(rows.id)
                        } else {
                            None
                        };
                    }
                    if ru_next == Some(id) {
                        ru_next = ru.next_series(&mut buckets)?;
                    }
                }
            }
            None => {
                let mut sc = Scan::open(&ctx.root, &ctx.files, group_ids.clone(), lo, hi)?;
                while sc.next_series(&mut rows)? {
                    let e = st.entry(rows.id).or_default();
                    for i in (0..rows.ts.len()).rev() {
                        if e.0.len() >= 20 {
                            break;
                        }
                        if !rows.v[i].is_nan() || (stale && rows.stale[i]) {
                            add(e, rows.ts[i]);
                        }
                    }
                }
            }
        }
        todo = group_ids
            .iter()
            .copied()
            .filter(|id| st.get(id).map_or(true, |e| e.0.len() < 20))
            .collect();
        drop(group_ids);
        hi = lo - 1;
    }
    Ok(st
        .into_iter()
        .filter_map(|(id, (d, _))| super::kernels::si_of(d).map(|s| (id, s)))
        .collect())
}

/// Series-at-a-time evaluation with rollups for (p, q] and raw samples elsewhere. Streaming:
/// one series in memory at a time (a result Vec for the whole call would be held at peak).
pub struct RollupStream {
    f: Func,
    g: Grid,
    q: i64,
    raw: Scan,
    ru: RollupScan,
    rows: SeriesRows,
    buckets: Vec<Bucket>,
    raw_next: Option<u64>,
    ru_next: Option<u64>,
    ts: Vec<i64>,
    v: Vec<f64>,
    c: Vec<f64>,
    adjs: Vec<Option<f64>>,
    // window aggregates: per segment (a raw sample or a bucket) count, sum, min, max
    wn: Vec<f64>,
    wmin: Vec<f64>,
    wmax: Vec<f64>,
    /// scrape intervals given by the caller (a time chunk of a larger grid: the whole grid's)
    si_fixed: Option<std::sync::Arc<HashMap<u64, i64>>>,
}

impl RollupStream {
    pub fn open(ctx: &EvalCtx, f: Func, g: &Grid, ids: &super::scan::IdList, plan: Plan) -> anyhow::Result<Self> {
        Self::open_with(ctx, f, g, ids, plan, None, None)
    }

    /// `g` as one time chunk of a larger grid: `si` the larger grid's scrape intervals (si_map),
    /// raw samples fetched from `from` (before g's own fetch start: every sample a point of the
    /// chunk can look back to), so each point evaluates as in the whole grid.
    pub fn open_with(
        ctx: &EvalCtx,
        f: Func,
        g: &Grid,
        ids: &super::scan::IdList,
        plan: Plan,
        si: Option<std::sync::Arc<HashMap<u64, i64>>>,
        from: Option<i64>,
    ) -> anyhow::Result<Self> {
        let (p, q) = (plan.p, plan.q);
        let fetch_start = from.map_or(g.fetch_start(f), |x| x.min(g.fetch_start(f)));
        let mut raw = Scan::open_skip(&ctx.root, &ctx.files, ids.clone(), fetch_start, g.end, Some((p, q)))?;
        let inputs: Vec<(&FileEntry, i64, i64)> = plan
            .segs
            .iter()
            .flat_map(|(sp, sq, files)| files.iter().map(move |fe| (fe, *sp, *sq)))
            .collect();
        // irate reads `iv` (the scrape-interval walk) only from the files of the last two days
        // eagerly: the walk takes a series' newest 20 intervals, older files' only when the
        // newer ones lack them (load_iv); with the whole grid's `si` given, from none
        // (PGAI_ROLLUP_EAGER_IV=1: every file's `iv` up front, as before; tests compare)
        let eager = std::env::var_os("PGAI_ROLLUP_EAGER_IV").is_some_and(|v| v == "1");
        let lazy_iv = (f == Func::Irate && !eager).then(|| {
            if si.is_some() {
                i64::MAX
            } else {
                inputs.iter().map(|(fe, _, sq)| fe.max_ts.min(*sq)).max().unwrap_or(0) - 2 * DAY_MS
            }
        });
        let mut ru = RollupScan::open(&ctx.root, &inputs, Some(ids.clone()), cols_for(f), lazy_iv)?;
        let mut rows = SeriesRows::default();
        let mut buckets = Vec::new();
        let raw_next = if raw.next_series(&mut rows)? {
            Some(rows.id)
        } else {
            None
        };
        let ru_next = ru.next_series(&mut buckets)?;
        Ok(Self {
            f,
            g: *g,
            q,
            raw,
            ru,
            rows,
            buckets,
            raw_next,
            ru_next,
            ts: vec![],
            v: vec![],
            c: vec![],
            adjs: vec![],
            wn: vec![],
            wmin: vec![],
            wmax: vec![],
            si_fixed: si,
        })
    }

    /// Evaluate the next series; false when done.
    pub fn next(&mut self, mut each: impl FnMut(u64, i64, f64)) -> anyhow::Result<bool> {
        let id = match (self.raw_next, self.ru_next) {
            (None, None) => return Ok(false),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (Some(a), Some(b)) => a.min(b),
        };
        let use_raw = self.raw_next == Some(id);
        let use_ru = self.ru_next == Some(id);
        if self.f == Func::Last {
            self.next_last(id, use_raw, use_ru, &mut each);
        } else if self.f.is_window_agg() {
            self.next_window(id, use_raw, use_ru, &mut each);
        } else {
            self.next_edges(id, use_raw, use_ru, &mut each)?;
        }
        if use_raw {
            self.raw_next = if self.raw.next_series(&mut self.rows)? {
                Some(self.rows.id)
            } else {
                None
            };
        }
        if use_ru {
            self.ru_next = self.ru.next_series(&mut self.buckets)?;
        }
        Ok(true)
    }

    /// Edge rollups, irate/idelta, timestamp: a compressed real-sample sequence.
    fn next_edges(
        &mut self,
        id: u64,
        use_raw: bool,
        use_ru: bool,
        each: &mut impl FnMut(u64, i64, f64),
    ) -> anyhow::Result<()> {
        let q = self.q;
        let (ts, v, c, adjs) = (&mut self.ts, &mut self.v, &mut self.c, &mut self.adjs);
        ts.clear();
        v.clear();
        adjs.clear();
        // raw head (ts <= p), synthetic (p, q], raw tail (ts > q)
        let rows = &self.rows;
        let mut tail_from = 0;
        if use_raw {
            for i in 0..rows.ts.len() {
                if rows.ts[i] > q {
                    break;
                }
                tail_from = i + 1;
                if rows.v[i].is_nan() {
                    continue;
                }
                ts.push(rows.ts[i]);
                v.push(rows.v[i]);
                adjs.push(None);
            }
        }
        if use_ru {
            let two_last = matches!(self.f, Func::Irate | Func::Idelta);
            // irate (COLS_IRATE: no first/second/adj columns): each bucket's last two samples,
            // or its only one. On a grid of bucket boundaries irate only ever differences two
            // adjacent samples (a bucket's prev and last, or a 1-sample bucket's last and the
            // previous bucket's last), and the sample just before a window holding one sample is
            // one of these; corrections between other neighbours are never read.
            let narrow = self.f == Func::Irate;
            for b in self.buckets.iter().filter(|b| b.n > 0 && narrow) {
                if b.n >= 2 {
                    ts.push(b.prev_ts);
                    v.push(b.prev_v);
                    adjs.push(None);
                }
                ts.push(b.last_ts);
                v.push(b.last_v);
                adjs.push(None);
            }
            for b in self.buckets.iter().filter(|b| b.n > 0 && !narrow) {
                ts.push(b.first_ts);
                v.push(b.first_v);
                adjs.push(None);
                if two_last && b.n >= 4 && b.prev_ts > b.second_ts {
                    // first, second, second-to-last, last: the last two are adjacent samples
                    let a_last = reset_adj(b.last_v, b.prev_v);
                    ts.push(b.second_ts);
                    v.push(b.second_v);
                    adjs.push(Some(b.adj_12));
                    ts.push(b.prev_ts);
                    v.push(b.prev_v);
                    adjs.push(Some(b.adj_inner - b.adj_12 - a_last));
                    ts.push(b.last_ts);
                    v.push(b.last_v);
                    adjs.push(Some(a_last));
                } else if b.n >= 3 {
                    ts.push(b.second_ts);
                    v.push(b.second_v);
                    adjs.push(Some(b.adj_12));
                    ts.push(b.last_ts);
                    v.push(b.last_v);
                    adjs.push(Some(b.adj_inner - b.adj_12));
                } else if b.n == 2 {
                    ts.push(b.last_ts);
                    v.push(b.last_v);
                    adjs.push(Some(b.adj_inner));
                }
            }
        }
        if use_raw {
            for i in tail_from..rows.ts.len() {
                if rows.v[i].is_nan() || rows.ts[i] <= q {
                    continue;
                }
                ts.push(rows.ts[i]);
                v.push(rows.v[i]);
                adjs.push(None);
            }
        }
        c.clear();
        let mut a = 0.0;
        for i in 0..ts.len() {
            if i > 0 {
                a += adjs[i].unwrap_or_else(|| reset_adj(v[i], v[i - 1]));
            }
            c.push(v[i] + a);
        }
        // si: the last (up to) 20 sample intervals of the fetched samples, from the raw tail back
        // through the buckets (each keeps its last 20 intervals) into the raw head
        let s = if let Some(m) = &self.si_fixed {
            m.get(&id).copied()
        } else if self.g.start < self.g.end {
            let (head, tail) = self.raw_split(use_raw, false);
            loop {
                match si_walk_try(&head, if use_ru { &self.buckets } else { &[] }, &tail, false) {
                    Ok(s) => break s,
                    // the walk reached buckets whose intervals were not read: read them, again
                    Err(()) => anyhow::ensure!(self.ru.load_iv(id, &mut self.buckets)?, "si walk: no lazy bucket"),
                }
            }
        } else {
            None
        };
        eval_prepared(self.f, &self.g, &self.ts, &self.v, &self.c, s, |t, y| each(id, t, y));
        Ok(())
    }

    /// The raw samples' times split at q (head <= q < tail); `stale`: markers count.
    fn raw_split(&self, use_raw: bool, stale: bool) -> (Vec<i64>, Vec<i64>) {
        if !use_raw {
            return (vec![], vec![]);
        }
        let r = &self.rows;
        (0..r.ts.len())
            .filter(|&i| !r.v[i].is_nan() || (stale && r.stale[i]))
            .map(|i| r.ts[i])
            .partition(|t| *t <= self.q)
    }

    /// Bare selector: the last element <= t (a staleness marker hides the series) if it is
    /// inside (t - w, t]. Per bucket only its last element is needed, for any w, when t is on a
    /// boundary; the scrape interval (window when step < maxPrev) counts markers as samples.
    fn next_last(&mut self, id: u64, use_raw: bool, use_ru: bool, each: &mut impl FnMut(u64, i64, f64)) {
        let q = self.q;
        let (ts, v) = (&mut self.ts, &mut self.v);
        ts.clear();
        v.clear();
        let rows = &self.rows;
        let keep = |i: usize| !rows.v[i].is_nan() || rows.stale[i];
        if use_raw {
            for i in (0..rows.ts.len()).filter(|&i| rows.ts[i] <= q && keep(i)) {
                ts.push(rows.ts[i]);
                v.push(rows.v[i]);
            }
        }
        if use_ru {
            for b in &self.buckets {
                ts.push(b.a_last);
                v.push(if b.last_stale { crate::model::stale_nan() } else { b.last_v });
            }
        }
        if use_raw {
            for i in (0..rows.ts.len()).filter(|&i| rows.ts[i] > q && keep(i)) {
                ts.push(rows.ts[i]);
                v.push(rows.v[i]);
            }
        }
        let s = if let Some(m) = &self.si_fixed {
            m.get(&id).copied()
        } else if self.g.start < self.g.end {
            let (head, tail) = self.raw_split(use_raw, true);
            si_walk_by(&head, if use_ru { &self.buckets } else { &[] }, &tail, true)
        } else {
            None
        };
        eval_prepared(self.f, &self.g, &self.ts, &self.v, &[], s, |t, y| each(id, t, y));
    }

    /// Window aggregates: raw samples and whole buckets as segments (count, sum, min, max).
    fn next_window(&mut self, id: u64, use_raw: bool, use_ru: bool, each: &mut impl FnMut(u64, i64, f64)) {
        let q = self.q;
        let (pos, sum, n, mn, mx) = (&mut self.ts, &mut self.v, &mut self.wn, &mut self.wmin, &mut self.wmax);
        pos.clear();
        sum.clear();
        n.clear();
        mn.clear();
        mx.clear();
        let rows = &self.rows;
        let mut raw = |i: usize| {
            pos.push(rows.ts[i]);
            sum.push(rows.v[i]);
            n.push(1.0);
            mn.push(rows.v[i]);
            mx.push(rows.v[i]);
        };
        if use_raw {
            for i in (0..rows.ts.len()).filter(|&i| rows.ts[i] <= q && !rows.v[i].is_nan()) {
                raw(i);
            }
        }
        if use_ru {
            for b in self.buckets.iter().filter(|b| b.n > 0) {
                pos.push(b.last_ts);
                sum.push(b.sum);
                n.push(b.n as f64);
                mn.push(b.min);
                mx.push(b.max);
            }
        }
        if use_raw {
            for i in (0..rows.ts.len()).filter(|&i| rows.ts[i] > q && !rows.v[i].is_nan()) {
                pos.push(rows.ts[i]);
                sum.push(rows.v[i]);
                n.push(1.0);
                mn.push(rows.v[i]);
                mx.push(rows.v[i]);
            }
        }
        let g = &self.g;
        let w = if g.window > 0 { g.window } else { g.step };
        let total = pos.len();
        let (mut a, mut b) = (0usize, 0usize);
        let mut t = g.start;
        while t <= g.end {
            while a < total && pos[a] <= t - w {
                a += 1;
            }
            while b < total && pos[b] <= t {
                b += 1;
            }
            if b > a {
                let cnt: f64 = n[a..b].iter().sum();
                let y = match self.f {
                    Func::SumOverTime => sum[a..b].iter().sum::<f64>(),
                    Func::AvgOverTime => sum[a..b].iter().sum::<f64>() / cnt,
                    Func::MinOverTime => mn[a..b].iter().copied().fold(f64::INFINITY, f64::min),
                    Func::MaxOverTime => mx[a..b].iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    Func::CountOverTime => cnt,
                    _ => unreachable!("not served from rollups"),
                };
                each(id, t, y);
            }
            t += g.step;
        }
    }
}

/// Evaluate f over ids with rollups; calls each(id, t, v), series by series.
pub fn eval(
    ctx: &EvalCtx,
    f: Func,
    g: &Grid,
    ids: &super::scan::IdList,
    plan: Plan,
    mut each: impl FnMut(u64, i64, f64),
) -> anyhow::Result<()> {
    let mut st = RollupStream::open(ctx, f, g, ids, plan)?;
    while st.next(&mut each)? {}
    Ok(())
}

// ------------------------------------------------------------------ 1 d tier (monthly files)

/// Build a coarser tier: the buckets of `inputs` (rollup files, each cut to its (p, q] of bucket
/// ends, all series) merged into `w`-buckets, sorted by (series_id, k). Returns rows.
pub fn build_tier(root: &Path, inputs: &[(&FileEntry, i64, i64)], w: i64, out: &Path, zstd_level: i32) -> anyhow::Result<u64> {
    let mut scan = RollupScan::open(root, inputs, None, &[], None)?;
    let rg = if w == B { ROLLUP_1HM_ROW_GROUP } else { ROLLUP_1D_ROW_GROUP };
    let mut wr = Writer::new(out, zstd_level, rg)?;
    let mut bs = Vec::new();
    while let Some(sid) = scan.next_series(&mut bs)? {
        let mut cur: Option<Bucket> = None;
        for b in &bs {
            let k = ceil_to(b.k, w);
            match &mut cur {
                Some(c) if c.k == k => c.merge(b),
                _ => {
                    if let Some(c) = cur.take() {
                        wr.push(sid, c)?;
                    }
                    cur = Some(Bucket { k, ..*b });
                }
            }
        }
        if let Some(c) = cur {
            wr.push(sid, c)?;
        }
    }
    wr.close()
}

/// One month file to (re)build: buckets (p, q] from `inputs`, replacing `old`.
pub struct Job1d {
    pub month: i64,
    pub p: i64,
    pub q: i64,
    pub inputs: Vec<(FileEntry, i64, i64)>,
    pub old: Option<FileEntry>,
}

/// Month files that are missing, short or older than a 1 h rollup they cover. A day's 1 d
/// bucket (X, X + 1 d] needs the 1 h rollups of day X and day X + 1 (the sample at exactly
/// midnight). Each month file holds one run of consecutive buildable days (the longest; the
/// latest of equals). A file that only needs more days at its end is extended from itself.
pub fn jobs_1d(files: &[FileEntry]) -> Vec<Job1d> {
    jobs_tier(files, KIND_1D)
}

/// `jobs_1d` for the month files of `kind` (KIND_1D or KIND_1HM: same days, same rule).
pub fn jobs_tier(files: &[FileEntry], kind: &str) -> Vec<Job1d> {
    let valid = valid_days(files, VER);
    let mut by_month: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    for &d in valid.keys() {
        if valid.contains_key(&(d + 1)) {
            by_month.entry(crate::store::month_of(d).0).or_default().push(d);
        }
    }
    let mut jobs = vec![];
    for (month, mut days) in by_month {
        days.sort_unstable();
        // longest run of consecutive days, the latest on ties
        let (mut best, mut s) = ((days[0], days[0]), days[0]);
        for i in 0..days.len() {
            if i > 0 && days[i] != days[i - 1] + 1 {
                s = days[i];
            }
            if days[i] - s >= best.1 - best.0 {
                best = (s, days[i]);
            }
        }
        let (s, e) = best;
        let (p, q) = (s * D, (e + 1) * D);
        let created = |d: i64| valid[&d].created_ms;
        let newest = (s..=e + 1).map(created).max().unwrap_or(0);
        let old = files.iter().find(|f| f.kind == kind && f.day == month).cloned();
        let hours = |d0: i64, lo: i64| -> Vec<(FileEntry, i64, i64)> {
            (d0..=e + 1).map(|d| (valid[&d].clone(), lo, q)).collect()
        };
        let inputs = match &old {
            Some(o) if o.ver >= VER && o.min_ts == p && o.max_ts == q && o.created_ms >= newest => continue,
            // still valid and only short: extend it with the new days' 1 h buckets
            Some(o)
                if o.ver >= VER
                    && o.min_ts == p
                    && o.max_ts > p
                    && o.max_ts < q
                    && o.max_ts % D == 0
                    && (s..=o.max_ts / D).all(|d| created(d) <= o.created_ms) =>
            {
                let mut v = vec![(o.clone(), p, o.max_ts)];
                v.extend(hours(o.max_ts / D, o.max_ts));
                v
            }
            _ => hours(s, p),
        };
        jobs.push(Job1d {
            month,
            p,
            q,
            inputs,
            old,
        });
    }
    jobs
}

/// Bring the 1 d month files up to date with the 1 h rollups (after closed-day compaction).
/// Returns the number of month files written.
pub fn update_1d(shared: &crate::store::Shared, zstd_level: i32, delete_grace_ms: i64) -> anyhow::Result<usize> {
    Ok(update_tier(shared, KIND_1D, zstd_level, delete_grace_ms)? + update_tier(shared, KIND_1HM, zstd_level, delete_grace_ms)?)
}

/// Bring the month files of `kind` (KIND_1D: 1 d buckets; KIND_1HM: 1 h buckets) up to date.
pub fn update_tier(shared: &crate::store::Shared, kind: &str, zstd_level: i32, delete_grace_ms: i64) -> anyhow::Result<usize> {
    let (w, letter) = if kind == KIND_1HM { (B, "H") } else { (D, "D") };
    let jobs = jobs_tier(&shared.catalog.lock().unwrap().files, kind);
    let mut done = 0;
    for job in jobs {
        let id = shared.catalog.lock().unwrap().alloc_id();
        let rel = format!(
            "{kind}/month={}/{letter}-{id:08}.parquet",
            &crate::model::day_str(job.month)[..7]
        );
        let abs = shared.root.join(&rel);
        std::fs::create_dir_all(abs.parent().unwrap())?;
        let tmp = abs.with_extension("parquet.tmp");
        let refs: Vec<(&FileEntry, i64, i64)> = job.inputs.iter().map(|(f, p, q)| (f, *p, *q)).collect();
        let rows = build_tier(&shared.root, &refs, w, &tmp, zstd_level)?;
        std::fs::File::open(&tmp)?.sync_all()?;
        std::fs::rename(&tmp, &abs)?;
        let mut cat = shared.catalog.lock().unwrap();
        // inputs replaced meanwhile (a day recompacted): drop this build, the next run redoes it
        let same = job
            .inputs
            .iter()
            .all(|(f, _, _)| cat.files.iter().any(|x| x.path == f.path && x.created_ms == f.created_ms));
        if !same {
            drop(cat);
            let _ = std::fs::remove_file(&abs);
            continue;
        }
        let old: Vec<String> = cat
            .files
            .iter()
            .filter(|f| f.kind == kind && f.day == job.month)
            .map(|f| f.path.clone())
            .collect();
        cat.files.retain(|f| !(f.kind == kind && f.day == job.month));
        let after = crate::model::now_ms() + delete_grace_ms;
        for p in old {
            cat.pending_delete.push((p, after));
        }
        cat.files.push(FileEntry {
            kind: kind.into(),
            path: rel,
            day: job.month,
            level: 2,
            rows,
            bytes: std::fs::metadata(&abs)?.len(),
            min_ts: job.p,
            max_ts: job.q,
            created_ms: crate::model::now_ms(),
            registered: false,
            ver: VER,
        });
        cat.save(&shared.root)?;
        shared.generation.store(cat.generation, Ordering::Release);
        done += 1;
    }
    Ok(done)
}
