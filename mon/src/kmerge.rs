//! Streaming k-way merge of sample files that are each sorted by (series_id, ts).
//!
//! Compaction used to `COPY (SELECT ... ORDER BY series_id, ts)`: a full sort whose memory
//! DuckDB doesn't keep inside `memory_limit` (185 MiB for a 132 M-sample day at 64 MB, OOM at
//! 32 MB). Every input is already sorted, so a heap merge needs only one small batch per input
//! plus one output row group: O(files), independent of rows.
use arrow::array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, SchemaRef, TimeUnit};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use parquet::schema::types::ColumnPath;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;

const IN_BATCH: usize = 2048;

/// Order-independent checksum of a batch: sum (wrapping) of a per-row hash over all columns.
fn batch_checksum(b: &RecordBatch) -> u64 {
    use arrow::array::*;
    let n = b.num_rows();
    let mut acc = vec![0x9E37_79B9_7F4A_7C15u64; n];
    for c in b.columns() {
        let mix = |acc: &mut [u64], i: usize, v: u64| {
            acc[i] = xxhash_rust::xxh3::xxh3_64_with_seed(&v.to_le_bytes(), acc[i]);
        };
        macro_rules! prim {
            ($t:ty, $f:expr) => {{
                let a = c.as_any().downcast_ref::<$t>().unwrap();
                for i in 0..n {
                    let v = if a.is_null(i) {
                        0xDEAD_BEEF_u64
                    } else {
                        $f(a.value(i))
                    };
                    mix(&mut acc, i, v);
                }
            }};
        }
        match c.data_type() {
            DataType::UInt64 => prim!(UInt64Array, |v: u64| v),
            DataType::Int64 => prim!(Int64Array, |v: i64| v as u64),
            DataType::Float64 => prim!(Float64Array, |v: f64| v.to_bits()),
            DataType::Timestamp(TimeUnit::Microsecond, _) => prim!(TimestampMicrosecondArray, |v: i64| v as u64),
            DataType::Timestamp(TimeUnit::Millisecond, _) => prim!(TimestampMillisecondArray, |v: i64| v as u64),
            t => panic!("checksum: unsupported column type {t:?}"),
        }
    }
    acc.into_iter().fold(0u64, |s, h| s.wrapping_add(h))
}

/// (rows, checksum) of one Parquet file, streaming.
pub fn file_checksum(p: &Path) -> anyhow::Result<(u64, u64)> {
    let r = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(p)?)?
        .with_batch_size(8192)
        .build()?;
    let (mut rows, mut sum) = (0u64, 0u64);
    for b in r {
        let b = b?;
        rows += b.num_rows() as u64;
        sum = sum.wrapping_add(batch_checksum(&b));
    }
    Ok((rows, sum))
}
const OUT_BATCH: usize = 8192;

struct Input {
    reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    batch: Option<RecordBatch>,
    row: usize,
    slot: usize, // index of the current batch in the interleave slot list
    rows: u64,
    sum: u64,
}

fn ts_value(a: &ArrayRef, i: usize) -> i64 {
    match a.data_type() {
        DataType::Timestamp(TimeUnit::Microsecond, _) => a
            .as_any()
            .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
            .unwrap()
            .value(i),
        DataType::Timestamp(TimeUnit::Millisecond, _) => a
            .as_any()
            .downcast_ref::<arrow::array::TimestampMillisecondArray>()
            .unwrap()
            .value(i),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => a
            .as_any()
            .downcast_ref::<arrow::array::TimestampNanosecondArray>()
            .unwrap()
            .value(i),
        DataType::Int64 => a.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap().value(i),
        t => panic!("unexpected ts type {t:?}"),
    }
}

/// Inputs open at once. Memory is per open input (a decoded page per column), so more files
/// than this are merged in passes through intermediate files next to `out`.
pub const FAN_IN: usize = 12;

/// Merge `inputs` into `out` (a temp name; the caller renames it), with at most FAN_IN files
/// open at once. Returns rows written.
pub fn merge_sorted(
    inputs: &[std::path::PathBuf],
    out: &Path,
    zstd_level: i32,
    row_group: usize,
) -> anyhow::Result<u64> {
    merge_sorted_opts(inputs, out, zstd_level, row_group, false)
}

/// `dedupe`: keep only the first row per series_id (series registry files, sorted by id then
/// first_seen). Without it every input row is kept and checksummed.
pub fn merge_sorted_opts(
    inputs: &[std::path::PathBuf],
    out: &Path,
    zstd_level: i32,
    row_group: usize,
    dedupe: bool,
) -> anyhow::Result<u64> {
    if inputs.len() <= FAN_IN {
        return merge_pass(inputs, out, zstd_level, row_group, dedupe);
    }
    let mut level = inputs.to_vec();
    let mut temps: Vec<std::path::PathBuf> = Vec::new();
    let mut pass = 0;
    let res = (|| {
        while level.len() > FAN_IN {
            let mut next = Vec::new();
            for (i, chunk) in level.chunks(FAN_IN).enumerate() {
                let t = out.with_extension(format!("p{pass}-{i}.tmp"));
                // intermediate passes: fast zstd, the final pass sets the real level
                merge_pass(chunk, &t, 1, row_group, dedupe)?;
                temps.push(t.clone());
                next.push(t);
            }
            // previous intermediates are no longer needed
            for p in &level {
                if temps.contains(p) {
                    let _ = std::fs::remove_file(p);
                }
            }
            level = next;
            pass += 1;
        }
        merge_pass(&level, out, zstd_level, row_group, dedupe)
    })();
    for t in temps {
        let _ = std::fs::remove_file(t);
    }
    res
}

fn merge_pass(
    inputs: &[std::path::PathBuf],
    out: &Path,
    zstd_level: i32,
    row_group: usize,
    dedupe: bool,
) -> anyhow::Result<u64> {
    let mut ins: Vec<Input> = Vec::with_capacity(inputs.len());
    let mut schema: Option<SchemaRef> = None;
    for p in inputs {
        let b = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(p)?)?.with_batch_size(IN_BATCH);
        if schema.is_none() {
            schema = Some(b.schema().clone());
        } else if b.schema().fields() != schema.as_ref().unwrap().fields() {
            anyhow::bail!("schema of {} differs", p.display());
        }
        ins.push(Input {
            reader: b.build()?,
            batch: None,
            row: 0,
            slot: 0,
            rows: 0,
            sum: 0,
        });
    }
    let schema = schema.ok_or_else(|| anyhow::anyhow!("no inputs"))?;
    let sid_col = schema.index_of("series_id")?;
    let ts_col = schema.index_of("ts").or_else(|_| schema.index_of("first_seen"))?;

    let mut props = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(zstd_level)?))
        .set_max_row_group_row_count(Some(row_group))
        .set_statistics_enabled(EnabledStatistics::Chunk);
    if schema.field(ts_col).name() == "ts" {
        // samples: statistics only where readers prune (series_id, ts); the value columns'
        // min/max were a quarter of the footer, fetched per cold file on S3. Series files keep
        // them all (small; SQL views filter on their labels)
        props = props
            .set_statistics_enabled(EnabledStatistics::None)
            .set_column_statistics_enabled(ColumnPath::from("series_id"), EnabledStatistics::Chunk)
            .set_column_statistics_enabled(ColumnPath::from("ts"), EnabledStatistics::Chunk);
    }
    for (name, enc) in [
        ("ts", Encoding::DELTA_BINARY_PACKED),
        ("ival", Encoding::DELTA_BINARY_PACKED),
        ("fval", Encoding::BYTE_STREAM_SPLIT),
        ("value", Encoding::BYTE_STREAM_SPLIT),
    ] {
        if schema.index_of(name).is_ok() {
            let c = ColumnPath::from(name);
            props = props
                .set_column_dictionary_enabled(c.clone(), false)
                .set_column_encoding(c, enc);
        }
    }
    let mut w = ArrowWriter::try_new(std::fs::File::create(out)?, schema.clone(), Some(props.build()))?;

    // slots: batches referenced by pending output indices
    let mut slots: Vec<RecordBatch> = Vec::new();
    let mut idx: Vec<(usize, usize)> = Vec::with_capacity(OUT_BATCH);
    let mut heap: BinaryHeap<Reverse<(u64, i64, usize)>> = BinaryHeap::new();

    let advance = |inp: &mut Input, slots: &mut Vec<RecordBatch>| -> anyhow::Result<bool> {
        loop {
            match inp.reader.next() {
                Some(b) => {
                    let b = b?;
                    if b.num_rows() == 0 {
                        continue;
                    }
                    if !dedupe {
                        inp.sum = inp.sum.wrapping_add(batch_checksum(&b));
                    }
                    inp.rows += b.num_rows() as u64;
                    slots.push(b.clone());
                    inp.slot = slots.len() - 1;
                    inp.batch = Some(b);
                    inp.row = 0;
                    return Ok(true);
                }
                None => {
                    inp.batch = None;
                    return Ok(false);
                }
            }
        }
    };
    let key = |inp: &Input| -> (u64, i64) {
        let b = inp.batch.as_ref().unwrap();
        let sid = b
            .column(sid_col)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(inp.row);
        (sid, ts_value(b.column(ts_col), inp.row))
    };
    for (i, inp) in ins.iter_mut().enumerate() {
        if advance(inp, &mut slots)? {
            let (s, t) = key(inp);
            heap.push(Reverse((s, t, i)));
        }
    }
    let mut rows = 0u64;
    let flush = |idx: &mut Vec<(usize, usize)>,
                 slots: &mut Vec<RecordBatch>,
                 ins: &mut Vec<Input>,
                 w: &mut ArrowWriter<std::fs::File>|
     -> anyhow::Result<()> {
        if idx.is_empty() {
            return Ok(());
        }
        let cols: Vec<ArrayRef> = (0..schema.fields().len())
            .map(|c| {
                let arrs: Vec<&dyn Array> = slots.iter().map(|b| b.column(c).as_ref()).collect();
                arrow::compute::interleave(&arrs, idx)
            })
            .collect::<Result<_, _>>()?;
        w.write(&RecordBatch::try_new(schema.clone(), cols)?)?;
        idx.clear();
        // keep only the batches inputs are still positioned in
        let mut kept: Vec<RecordBatch> = Vec::new();
        for inp in ins.iter_mut() {
            if let Some(b) = &inp.batch {
                kept.push(b.clone());
                inp.slot = kept.len() - 1;
            }
        }
        *slots = kept;
        Ok(())
    };
    let mut last: Option<u64> = None;
    while let Some(Reverse((sid, _, i))) = heap.pop() {
        {
            let inp = &mut ins[i];
            if !(dedupe && last == Some(sid)) {
                idx.push((inp.slot, inp.row));
                rows += 1;
            }
            last = Some(sid);
            inp.row += 1;
        }
        let more = {
            let inp = &mut ins[i];
            if inp.row < inp.batch.as_ref().unwrap().num_rows() {
                true
            } else {
                advance(inp, &mut slots)?
            }
        };
        if more {
            let (s, t) = key(&ins[i]);
            heap.push(Reverse((s, t, i)));
        }
        if idx.len() >= OUT_BATCH {
            flush(&mut idx, &mut slots, &mut ins, &mut w)?;
        }
    }
    flush(&mut idx, &mut slots, &mut ins, &mut w)?;
    w.close()?;
    // Verify: the file as written holds exactly the rows read from the inputs.
    let in_rows: u64 = ins.iter().map(|i| i.rows).sum();
    if dedupe {
        // rows shrink by design; check counts only
        let n: i64 = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(out)?)?
            .metadata()
            .file_metadata()
            .num_rows();
        anyhow::ensure!(
            n as u64 == rows && rows <= in_rows,
            "dedupe merge: wrote {n}, merged {rows}, read {in_rows}"
        );
        return Ok(rows);
    }
    let in_sum = ins.iter().fold(0u64, |s, i| s.wrapping_add(i.sum));
    let (out_rows, out_sum) = file_checksum(out)?;
    anyhow::ensure!(
        in_rows == rows && out_rows == rows && out_sum == in_sum,
        "k-way merge checksum mismatch: in {in_rows}/{in_sum:x}, merged {rows}, out {out_rows}/{out_sum:x}"
    );
    Ok(rows)
}
