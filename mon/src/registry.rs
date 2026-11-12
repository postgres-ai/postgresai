//! Series registry (review B7): one row per series ever seen, disjoint across files, so the
//! contract `series` view needs no `DISTINCT ON` over every per-day row of the retention.
//!
//! - Per-day series files (`series/`) stay as they are: the evaluator reads them per window.
//! - A closed day's series file is *registered* at compaction: its series not yet in the registry
//!   go to a new `series_reg/` file (sorted by id, with a Bloom filter on series_id), and the
//!   file is flagged `registered` in the catalog. Registry files never overlap.
//! - The rest (open day, days not yet compacted, late files) is the *tail*. The reader resolves
//!   which tail series are new to the registry (`tail_new`: Bloom filters first, then the id
//!   column of the few row groups that may hold them) and the view adds just those rows, once.
//!
//! So `series` = registry UNION ALL (tail rows of series new to the registry, one file each),
//! no dedupe of the whole history, and filters on metric push down into the registry Parquet.
use crate::catalog::FileEntry;
use arrow::array::{Array, BooleanArray, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use parquet::schema::types::ColumnPath;
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub const KIND: &str = "series_reg";

fn id_leaf<T: parquet::file::reader::ChunkReader + 'static>(
    b: &ParquetRecordBatchReaderBuilder<T>,
) -> anyhow::Result<usize> {
    b.parquet_schema()
        .columns()
        .iter()
        .position(|c| c.path().string() == "series_id")
        .ok_or_else(|| anyhow::anyhow!("no series_id column"))
}

/// A row group's [min, max] series_id from its stats (the full range without them).
fn id_bounds(md: &parquet::file::metadata::ParquetMetaData, rg: usize, leaf: usize) -> (u64, u64) {
    match md.row_group(rg).column(leaf).statistics() {
        Some(parquet::file::statistics::Statistics::Int64(s)) => (
            s.min_opt().map(|v| *v as u64).unwrap_or(0),
            s.max_opt().map(|v| *v as u64).unwrap_or(u64::MAX),
        ),
        _ => (0, u64::MAX),
    }
}

/// Every series_id of a file (id column only).
pub fn ids_of(p: &Path) -> anyhow::Result<Vec<u64>> {
    let (b, src) = crate::eval::footer::builder_src(p)?;
    let leaf = id_leaf(&b)?;
    let mask = ProjectionMask::leaves(b.parquet_schema(), [leaf]);
    src.prefetch(
        b.metadata(),
        &(0..b.metadata().num_row_groups()).collect::<Vec<_>>(),
        Some(&mask),
    );
    let r = b.with_projection(mask).with_batch_size(65_536).build()?;
    let mut out = Vec::new();
    for batch in r {
        let batch = batch?;
        let a = batch.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
        out.extend(a.values().iter().copied());
    }
    Ok(out)
}

/// Remove from `cand` every id present in registry file `p`. Per row group: skip it when no
/// candidate is within its [min, max] id stats or its Bloom filter rejects them all; otherwise
/// read its id column. Memory: one row group's ids.
pub fn remove_present(p: &Path, cand: &mut HashSet<u64>) -> anyhow::Result<()> {
    if cand.is_empty() {
        return Ok(());
    }
    let (mut b, src) = crate::eval::footer::builder_src(p)?;
    let leaf = id_leaf(&b)?;
    let n_rg = b.metadata().num_row_groups();
    src.prefetch_blooms(
        b.metadata(),
        (0..n_rg).filter(|&rg| {
            let (lo, hi) = id_bounds(b.metadata(), rg, leaf);
            cand.iter().any(|&x| x >= lo && x <= hi)
        }),
        leaf,
    );
    let mut keep_rg = Vec::new();
    for rg in 0..n_rg {
        let md = b.metadata().row_group(rg).column(leaf);
        let (lo, hi) = match md.statistics() {
            Some(parquet::file::statistics::Statistics::Int64(s)) => (
                s.min_opt().map(|v| *v as u64).unwrap_or(0),
                s.max_opt().map(|v| *v as u64).unwrap_or(u64::MAX),
            ),
            _ => (0, u64::MAX),
        };
        let in_range: Vec<u64> = cand.iter().copied().filter(|&x| x >= lo && x <= hi).collect();
        if in_range.is_empty() {
            continue;
        }
        match b.get_row_group_column_bloom_filter(rg, leaf)? {
            Some(sbbf) => {
                if in_range.iter().any(|x| sbbf.check(x)) {
                    keep_rg.push(rg);
                }
            }
            None => keep_rg.push(rg),
        }
    }
    if keep_rg.is_empty() {
        return Ok(());
    }
    let mask = ProjectionMask::leaves(b.parquet_schema(), [leaf]);
    src.prefetch(b.metadata(), &keep_rg, Some(&mask));
    b = b.with_projection(mask).with_row_groups(keep_rg).with_batch_size(65_536);
    for batch in b.build()? {
        let batch = batch?;
        let a = batch.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
        for x in a.values().iter() {
            cand.remove(x);
        }
        if cand.is_empty() {
            break;
        }
    }
    Ok(())
}

/// Rows (id, metric, label keys, label values) of `files` whose id is in `want`; row groups are
/// skipped by id stats and Bloom filter. For a few ids over many files (a top-k's finalists).
pub fn scan_rows_with_ids(
    root: &Path,
    files: &[&FileEntry],
    want: &HashSet<u64>,
    each: &mut dyn FnMut(u64, &str, &arrow::array::StringArray, &arrow::array::StringArray),
) -> anyhow::Result<()> {
    use arrow::array::{MapArray, StringArray};
    if want.is_empty() {
        return Ok(());
    }
    let mut left: HashSet<u64> = want.clone();
    for f in files {
        if left.is_empty() {
            break;
        }
        let (mut b, src) = crate::eval::footer::builder_src(&root.join(&f.path))?;
        let leaf = id_leaf(&b)?;
        src.prefetch_blooms(
            b.metadata(),
            (0..b.metadata().num_row_groups()).filter(|&rg| {
                let (lo, hi) = id_bounds(b.metadata(), rg, leaf);
                left.iter().any(|&x| x >= lo && x <= hi)
            }),
            leaf,
        );
        let mut keep = Vec::new();
        for rg in 0..b.metadata().num_row_groups() {
            let md = b.metadata().row_group(rg).column(leaf);
            let (lo, hi) = match md.statistics() {
                Some(parquet::file::statistics::Statistics::Int64(s)) => (
                    s.min_opt().map(|v| *v as u64).unwrap_or(0),
                    s.max_opt().map(|v| *v as u64).unwrap_or(u64::MAX),
                ),
                _ => (0, u64::MAX),
            };
            let cand: Vec<u64> = left.iter().copied().filter(|&x| x >= lo && x <= hi).collect();
            if cand.is_empty() {
                continue;
            }
            let hit = match b.get_row_group_column_bloom_filter(rg, leaf)? {
                Some(sbbf) => cand.iter().any(|x| sbbf.check(x)),
                None => true,
            };
            if hit {
                keep.push(rg);
            }
        }
        if keep.is_empty() {
            continue;
        }
        let names: Vec<String> = b.schema().fields().iter().map(|f| f.name().clone()).collect();
        anyhow::ensure!(
            names.iter().take(3).eq(["series_id", "metric", "labels"].iter()),
            "unexpected series schema {names:?}"
        );
        let mask = ProjectionMask::roots(b.parquet_schema(), [0usize, 1, 2]);
        src.prefetch(b.metadata(), &keep, Some(&mask));
        b = b.with_row_groups(keep).with_projection(mask).with_batch_size(4096);
        for batch in b.build()? {
            let batch = batch?;
            let ids = batch.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
            let ms = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            let lm = batch.column(2).as_any().downcast_ref::<MapArray>().unwrap();
            for i in 0..batch.num_rows() {
                let id = ids.value(i);
                if !left.remove(&id) {
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

/// Ids of `files` (in order) that are in none of the registry files, each with the index of the
/// first file holding it.
pub fn new_ids(root: &Path, files: &[&FileEntry], reg: &[&FileEntry]) -> anyhow::Result<HashMap<u64, usize>> {
    let mut first: HashMap<u64, usize> = HashMap::new();
    for (i, f) in files.iter().enumerate() {
        for id in ids_of(&root.join(&f.path))? {
            first.entry(id).or_insert(i);
        }
    }
    let mut cand: HashSet<u64> = first.keys().copied().collect();
    for r in reg {
        remove_present(&root.join(&r.path), &mut cand)?;
    }
    first.retain(|id, _| cand.contains(id));
    Ok(first)
}

/// Write the rows of series file `src` whose id is in `ids` (each once) to `out`, as a registry
/// file (same columns, Bloom filter on series_id). Returns the rows written.
pub fn write_subset(src: &Path, ids: &HashSet<u64>, out: &Path, zstd_level: i32) -> anyhow::Result<u64> {
    let b = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(src)?)?.with_batch_size(8192);
    let schema = b.schema().clone();
    let r = b.build()?;
    let sid = schema.index_of("series_id")?;
    let props = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(zstd_level)?))
        .set_max_row_group_row_count(Some(crate::store::SERIES_ROW_GROUP))
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_column_bloom_filter_enabled(ColumnPath::from("series_id"), true)
        .set_column_bloom_filter_fpp(ColumnPath::from("series_id"), 0.01)
        .set_column_dictionary_enabled(ColumnPath::from("series_id"), false)
        .build();
    let mut w = ArrowWriter::try_new(std::fs::File::create(out)?, schema, Some(props))?;
    let mut seen: HashSet<u64> = HashSet::new();
    let mut rows = 0u64;
    for batch in r {
        let batch = batch?;
        let a = batch.column(sid).as_any().downcast_ref::<UInt64Array>().unwrap();
        let keep: BooleanArray = (0..a.len())
            .map(|i| Some(!a.is_null(i) && ids.contains(&a.value(i)) && seen.insert(a.value(i))))
            .collect();
        let fb = arrow::compute::filter_record_batch(&batch, &keep)?;
        if fb.num_rows() > 0 {
            rows += fb.num_rows() as u64;
            w.write(&fb)?;
        }
    }
    w.close()?;
    Ok(rows)
}

// ------------------------------------------------------------------ reader side

use crate::eval::{EvalCtx, EvalHandle};
use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use std::sync::{Arc, Mutex};

/// Tail series new to the registry, computed once per catalog generation (EvalCtx).
pub fn tail_new(ctx: &EvalCtx) -> Result<Arc<Vec<(u64, u32)>>, String> {
    ctx.tail_new
        .get_or_init(|| {
            let tail: Vec<&FileEntry> = ctx.tail_files.iter().collect();
            let reg: Vec<&FileEntry> = ctx.reg_files.iter().collect();
            let rows: u64 = tail.iter().map(|f| f.rows).sum();
            if rows > 4_000_000 {
                return Err(format!(
                    "{rows} series rows are not in the series registry yet (store not compacted?); \
                     run `pgai-mon compact` on it"
                ));
            }
            let m = new_ids(&ctx.root, &tail, &reg).map_err(|e| e.to_string())?;
            let mut v: Vec<(u64, u32)> = m.into_iter().map(|(id, i)| (id, i as u32)).collect();
            v.sort_unstable();
            Ok(Arc::new(v))
        })
        .clone()
}

/// `pgai_series_tail_ids()` -> (series_id, file): which tail rows the `series` view adds.
pub struct TailIds;
pub struct TailBind {
    ids: Arc<Vec<(u64, u32)>>,
    files: Vec<String>,
}
pub struct TailInit {
    pos: Mutex<usize>,
}

impl VTab for TailIds {
    type InitData = TailInit;
    type BindData = TailBind;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        bind.add_result_column("series_id", LogicalTypeHandle::from(LogicalTypeId::UBigint));
        bind.add_result_column("file", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        let handle = unsafe { &*bind.get_extra_info::<EvalHandle>() };
        let ctx = handle.read().unwrap().clone();
        let ids = tail_new(&ctx)?;
        let files = ctx
            .tail_files
            .iter()
            .map(|f| ctx.root.join(&f.path).to_string_lossy().into_owned())
            .collect();
        Ok(TailBind { ids, files })
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        init.set_max_threads(1);
        Ok(TailInit { pos: Mutex::new(0) })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        let b = func.get_bind_data();
        let mut pos = func.get_init_data().pos.lock().unwrap();
        let cap = output.flat_vector(0).capacity();
        let k = (b.ids.len() - *pos).min(cap);
        {
            let mut ids = output.flat_vector(0);
            let s = unsafe { ids.as_mut_slice_with_len::<u64>(k) };
            for (j, sj) in s.iter_mut().enumerate() {
                *sj = b.ids[*pos + j].0;
            }
        }
        let fv = output.flat_vector(1);
        for j in 0..k {
            fv.insert(j, b.files[b.ids[*pos + j].1 as usize].as_str());
        }
        *pos += k;
        output.set_len(k);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![])
    }
}
