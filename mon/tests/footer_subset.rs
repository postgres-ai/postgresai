//! Readers over a subset of row groups (eval/footer.rs): `builder_rgs` and `RgReader` hold the
//! metadata of the selected row groups only and must read exactly the rows a reader over the
//! whole footer reads for the same row groups.
use arrow::array::{Array, Int64Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use pgai_mon::eval::footer;
use std::sync::Arc;

/// A file of `n_rg` row groups of 100 rows: series_id = row / 10, v = row.
fn write_file(dir: &std::path::Path, n_rg: usize) -> std::path::PathBuf {
    write_rows(dir, n_rg, 100)
}

/// A file of `n_rg` row groups of `rg_rows` rows: series_id = row / 10, v = row.
fn write_rows(dir: &std::path::Path, n_rg: usize, rg_rows: usize) -> std::path::PathBuf {
    let p = dir.join("t.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("series_id", DataType::UInt64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(rg_rows))
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();
    let mut w = ArrowWriter::try_new(std::fs::File::create(&p).unwrap(), schema.clone(), Some(props)).unwrap();
    let n = (n_rg * rg_rows) as u64;
    let b = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from_iter_values((0..n).map(|i| i / 10))),
            Arc::new(Int64Array::from_iter_values((0..n as i64))),
        ],
    )
    .unwrap();
    w.write(&b).unwrap();
    w.close().unwrap();
    p
}

fn values(batches: impl IntoIterator<Item = RecordBatch>) -> Vec<(u64, i64)> {
    let mut out = vec![];
    for b in batches {
        let s = b.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
        let v = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..b.num_rows() {
            out.push((s.value(i), v.value(i)));
        }
    }
    out
}

fn reference(p: &std::path::Path, rgs: &[usize]) -> Vec<(u64, i64)> {
    let r = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(p).unwrap())
        .unwrap()
        .with_row_groups(rgs.to_vec())
        .build()
        .unwrap();
    values(r.map(|b| b.unwrap()))
}

#[test]
fn subset_reads_the_selected_row_groups() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(dir.path(), 12);
    for rgs in [
        vec![0],
        vec![3, 7, 11],
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        (0..12).collect::<Vec<_>>(),
    ] {
        let want = reference(&p, &rgs);
        assert_eq!(want.len(), rgs.len() * 100);

        let (b, src) = footer::builder_src(&p).unwrap();
        let b = footer::builder_rgs(b, &src, &rgs).unwrap();
        // only the selected row groups' metadata (all of it when every row group is selected)
        assert_eq!(b.metadata().num_row_groups(), rgs.len(), "{rgs:?}");
        let got = values(b.with_batch_size(37).build().unwrap().map(|x| x.unwrap()));
        assert_eq!(got, want, "builder_rgs {rgs:?}");

        let (b, src) = footer::builder_src(&p).unwrap();
        let mut r = footer::RgReader::new(b, src, &rgs, 37, None).unwrap();
        assert_eq!(r.row_groups_held(), rgs.len());
        let mut got = vec![];
        while let Some(b) = r.next_batch().unwrap() {
            // small row groups are decoded whole, their reader closed at once
            assert!(!r.streaming());
            assert!(b.num_rows() <= 37);
            got.extend(values([b]));
        }
        assert_eq!(got, want, "RgReader {rgs:?}");
        assert!(r.next_batch().unwrap().is_none());
    }
}

#[test]
fn rg_reader_with_no_row_groups_is_empty() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(dir.path(), 3);
    let (b, src) = footer::builder_src(&p).unwrap();
    let mut r = footer::RgReader::new(b, src, &[], 1024, None).unwrap();
    assert_eq!(r.row_groups_held(), 0);
    assert!(r.next_batch().unwrap().is_none());
}

#[test]
fn rg_reader_applies_the_projection() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(dir.path(), 4);
    let (b, src) = footer::builder_src(&p).unwrap();
    let mask = parquet::arrow::ProjectionMask::roots(b.parquet_schema(), [1]);
    let mut r = footer::RgReader::new(b, src, &[1, 3], 1024, Some(mask)).unwrap();
    let mut got = vec![];
    while let Some(b) = r.next_batch().unwrap() {
        assert_eq!(b.num_columns(), 1);
        assert_eq!(b.schema().field(0).name(), "v");
        got.extend(
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .iter()
                .copied(),
        );
    }
    assert_eq!(got, (100..200).chain(300..400).collect::<Vec<i64>>());
}

#[test]
fn rg_reader_streams_big_row_groups() {
    // row groups decoding to more per column than an open reader holds stream; smaller ones
    // (every other test here) are decoded whole
    for (rows, stream) in [(200_000usize, true), (16_384, false)] {
        assert_eq!((rows * 8) as i64 > footer::RG_EAGER_BYTES_PER_COLUMN, stream);
        let dir = tempfile::tempdir().unwrap();
        let p = write_rows(dir.path(), 3, rows);
        let rgs = vec![0, 2];
        let want = reference(&p, &rgs);
        for proj in [false, true] {
            let (b, src) = footer::builder_src(&p).unwrap();
            let mask = proj.then(|| parquet::arrow::ProjectionMask::roots(b.parquet_schema(), [0, 1]));
            let mut r = footer::RgReader::new(b, src, &rgs, 4096, mask).unwrap();
            let mut got = vec![];
            let mut streamed = 0;
            while let Some(b) = r.next_batch().unwrap() {
                streamed += r.streaming() as usize;
                assert!(b.num_rows() <= 4096);
                got.extend(values([b]));
            }
            assert_eq!(got, want, "rows {rows} proj {proj}");
            assert_eq!(
                streamed,
                if stream { rgs.len() * rows.div_ceil(4096) } else { 0 },
                "rows {rows}"
            );
            assert!(!r.streaming());
        }
    }
}
