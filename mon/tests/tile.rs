//! `pgai-mon tile`: the day map, shifted content, rollups equal to what compaction builds from
//! the shifted samples, and the format marker.
use arrow::array::{Array, Date32Array, RecordBatch};
use pgai_mon::catalog::Catalog;
use pgai_mon::config::StoreCfg;
use pgai_mon::eval::scan::{Scan, SeriesRows};
use pgai_mon::model::*;
use pgai_mon::store::StoreCore;
use pgai_mon::tile;
use std::path::Path;

const DAY0: i64 = 1_794_009_600_000 - 4 * DAY_MS; // a UTC midnight

/// Four days of 8 counters at a jittered 60 s: day 0 starts at 06:00 (partial), days 1 and 2
/// are full, day 3 ends at 15:00 (the recording's end). Resets, NaN and staleness markers.
fn source(dir: &Path) -> Vec<u64> {
    let mut core = StoreCore::open(dir, &StoreCfg::default()).unwrap();
    let mut seed = 5u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (start, end) = (DAY0 + 6 * 3_600_000, DAY0 + 3 * DAY_MS + 15 * 3_600_000);
    let mut ids = vec![];
    let mut samples = vec![];
    let mut defs = std::collections::BTreeMap::new();
    for s in 0..8u64 {
        let labels = vec![("i".to_string(), s.to_string())];
        let id = series_id("x_total", &labels);
        ids.push(id);
        let mut t = start + (rnd() % 60_000) as i64;
        let mut v = 0.0;
        let mut k = 0;
        while t < end {
            v += (rnd() % 10) as f64;
            if k % 900 == 400 {
                v = 0.0;
            }
            let value = if k % 333 == 50 {
                stale_nan()
            } else if k % 577 == 60 {
                f64::NAN
            } else {
                v
            };
            samples.push(Sample {
                series_id: id,
                ts: t,
                value,
            });
            defs.entry((id, day_of(t))).or_insert(SeriesDef {
                series_id: id,
                metric: "x_total".into(),
                labels: labels.clone(),
                first_seen: t,
            });
            t += 60_000 + (rnd() % 200) as i64 - 100;
            k += 1;
        }
    }
    core.append(Record::Series(defs.into_values().collect())).unwrap();
    for c in samples.chunks(20_000) {
        core.append(Record::Samples(c.to_vec())).unwrap();
        core.flush().unwrap();
    }
    ids.sort();
    ids
}

fn day_rows(root: &Path, cat: &Catalog, ids: &[u64], day: i64) -> Vec<(u64, Vec<i64>, Vec<u64>, Vec<bool>)> {
    let files: Vec<_> = cat.files.iter().filter(|f| f.kind == "samples").cloned().collect();
    let mut sc = Scan::open(root, &files, ids.to_vec(), day * DAY_MS, (day + 1) * DAY_MS - 1).unwrap();
    let mut r = SeriesRows::default();
    let mut out = vec![];
    while sc.next_series(&mut r).unwrap() {
        out.push((
            r.id,
            r.ts.clone(),
            r.v.iter().map(|v| v.to_bits()).collect(),
            r.stale.clone(),
        ));
    }
    out
}

fn read_all(p: &Path) -> Vec<RecordBatch> {
    let b = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(p).unwrap())
        .unwrap();
    let schema = b.schema().clone();
    let batches: Vec<RecordBatch> = b.build().unwrap().map(|x| x.unwrap()).collect();
    vec![arrow::compute::concat_batches(&schema, &batches).unwrap()]
}

#[test]
fn a_tiled_store_repeats_full_days_back_from_the_recorded_end() {
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("src");
    let ids = source(&src);
    let src_cat_before = std::fs::read(src.join("catalog.json")).unwrap();
    let out = d.path().join("out");
    let r = tile::tile(&tile::Opts {
        src: src.clone(),
        out: out.clone(),
        days: 7,
        jobs: 2,
        zstd_level: 3,
        verbose: false,
    })
    .unwrap();
    let day0 = day_of(DAY0);
    let (d1, d2, d3) = (day0 + 1, day0 + 2, day0 + 3);
    // the recorded end stays last; full days at their dates, then repeated back; day 0 (partial)
    // is not used
    assert_eq!(
        r.map,
        vec![
            (d3, d3),
            (d2, d2),
            (d1, d1),
            (day0, d2),
            (day0 - 1, d1),
            (day0 - 2, d2),
            (day0 - 3, d1)
        ]
    );
    assert_eq!(
        std::fs::read(src.join("catalog.json")).unwrap(),
        src_cat_before,
        "source untouched"
    );

    let cat = Catalog::load_or_new(&out, "").unwrap();
    for (od, _) in &r.map {
        let n = |k: &str| cat.files.iter().filter(|f| f.kind == k && f.day == *od).count();
        assert_eq!(n("samples"), 1, "one samples file on {}", day_str(*od));
        assert_eq!(n(pgai_mon::eval::rollup::KIND), 1, "one 1 h rollup on {}", day_str(*od));
        assert_eq!(n("series"), 1);
    }
    for f in cat
        .files
        .iter()
        .filter(|f| ["samples", "series", "logs", "plan_events"].contains(&f.kind.as_str()))
    {
        assert!(
            r.map.iter().any(|(od, _)| *od == f.day),
            "{} outside the window",
            f.path
        );
    }
    for f in &cat.files {
        assert!(out.join(&f.path).exists(), "{} listed and present", f.path);
    }
    assert!(!out.join("tmp").exists() && !out.join("wal").exists());

    // a shifted day holds its source day's samples, times moved by whole days
    let src_cat = Catalog::load_or_new(&src, "").unwrap();
    for &(od, sd) in &r.map[3..5] {
        let want: Vec<_> = day_rows(&src, &src_cat, &ids, sd)
            .into_iter()
            .map(|(id, ts, v, st)| (id, ts.iter().map(|t| t + (od - sd) * DAY_MS).collect::<Vec<_>>(), v, st))
            .collect();
        assert!(!want.is_empty());
        assert_eq!(day_rows(&out, &cat, &ids, od), want, "day {}", day_str(od));
    }
    // days kept at their date are unchanged
    assert_eq!(day_rows(&out, &cat, &ids, d2), day_rows(&src, &src_cat, &ids, d2));

    // per-day series files: Date32 `date` and first_seen moved too
    let od = day0 - 1;
    let sf = cat.files.iter().find(|f| f.kind == "series" && f.day == od).unwrap();
    let b = &read_all(&out.join(&sf.path))[0];
    if let Some(c) = b.column_by_name("date") {
        let dates = c.as_any().downcast_ref::<Date32Array>().unwrap();
        assert!((0..dates.len()).all(|i| dates.value(i) as i64 == od));
    }
    let fs = arrow::compute::cast(
        b.column_by_name("first_seen").unwrap(),
        &arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
    )
    .unwrap();
    let fs = fs
        .as_any()
        .downcast_ref::<arrow::array::TimestampMillisecondArray>()
        .unwrap();
    assert!(
        fs.len() > 0 && (0..fs.len()).all(|i| day_of(fs.value(i)) == od),
        "first_seen on its day"
    );

    // the shifted 1 h rollup equals the one compaction builds from the shifted samples
    let ru = cat
        .files
        .iter()
        .find(|f| f.kind == pgai_mon::eval::rollup::KIND && f.day == od)
        .unwrap();
    let sm = cat.files.iter().find(|f| f.kind == "samples" && f.day == od).unwrap();
    let x = d.path().join("x");
    std::fs::create_dir_all(x.join("samples")).unwrap();
    std::fs::copy(out.join(&sm.path), x.join("samples/day.parquet")).unwrap();
    let mut xc = Catalog::load_or_new(&x, &cat.samples_layout).unwrap();
    xc.samples_layout = cat.samples_layout.clone();
    xc.next_file_id = 1000;
    xc.files.push(pgai_mon::catalog::FileEntry {
        path: "samples/day.parquet".into(),
        ..sm.clone()
    });
    xc.save(&x).unwrap();
    {
        let core = StoreCore::open(&x, &StoreCfg::default()).unwrap();
        core.compactor().unwrap().run_once(now_ms(), true).unwrap();
    }
    let xc = Catalog::load_or_new(&x, "").unwrap();
    let built = xc
        .files
        .iter()
        .find(|f| f.kind == pgai_mon::eval::rollup::KIND)
        .unwrap();
    assert_eq!(built.ver, ru.ver);
    assert_eq!(
        (built.min_ts, built.max_ts, built.rows),
        (ru.min_ts, ru.max_ts, ru.rows)
    );
    assert_eq!(read_all(&x.join(&built.path)), read_all(&out.join(&ru.path)));

    // the 1 d tier, when this build has one, covers the tiled days too (built after they are in)
    for (kind, built) in [
        ("rollup_1d", r.format.ends_with("tier1d") || r.format.ends_with("tier1hm")),
        ("rollup_1h_month", r.format.ends_with("tier1hm")),
    ] {
    let d1: Vec<_> = cat.files.iter().filter(|f| f.kind == kind).collect();
    if built {
        // month files: together they span (first tiled day, last closed day]
        let first = r.map.last().unwrap().0 * DAY_MS;
        let mut spans: Vec<(i64, i64)> = d1.iter().map(|f| (f.min_ts, f.max_ts)).collect();
        spans.sort();
        let mut reach = first;
        for (p, q) in &spans {
            assert!(*p <= reach, "gap in the 1 d tier before {p}: {spans:?}");
            reach = reach.max(*q);
        }
        assert!(reach >= d3 * DAY_MS, "{kind} ends at {reach}: {spans:?}");
    } else {
        assert!(d1.is_empty(), "{kind}");
    }
    }

    // the format marker
    let fmt = r.format.clone();
    let v = pgai_mon::eval::rollup::VER;
    assert!(fmt == format!("v{v}-base") || fmt == format!("v{v}-tier1d") || fmt == format!("v{v}-tier1hm"), "{fmt}");
    assert_eq!(fmt, format!("v{v}-tier1hm"), "this build writes both month tiers");
    let vers: std::collections::BTreeSet<u8> = cat
        .files
        .iter()
        .filter(|f| f.kind == pgai_mon::eval::rollup::KIND)
        .map(|f| f.ver)
        .collect();
    assert_eq!(vers, [v].into(), "every 1 h rollup at the current format");
    assert_eq!(cat.format_version.as_deref(), Some(fmt.as_str()));
    assert_eq!(tile::format_version().unwrap(), fmt);
    let marker = std::fs::read_to_string(out.join(tile::FORMAT_FILE)).unwrap();
    assert!(marker.starts_with(&format!("format={fmt}\n")), "{marker}");
    assert!(marker.contains("days=7\n"));
}

#[test]
fn a_window_shorter_than_the_recording_keeps_only_its_last_days() {
    let mut s = std::collections::BTreeMap::new();
    for d in 10..20i64 {
        s.insert(d, (d * DAY_MS, (d + 1) * DAY_MS - 1));
    }
    s.insert(20, (20 * DAY_MS, 20 * DAY_MS + 3_600_000));
    let m = tile::day_map(&s, 3).unwrap();
    assert_eq!(m, vec![(20, 20), (19, 19), (18, 18)]);
    // no full day to repeat
    let mut one = std::collections::BTreeMap::new();
    one.insert(5, (5 * DAY_MS + 7_200_000, 6 * DAY_MS - 1));
    one.insert(6, (6 * DAY_MS, 6 * DAY_MS + 1000));
    assert!(tile::day_map(&one, 3).is_err());
    assert_eq!(tile::day_map(&one, 1).unwrap(), vec![(6, 6)]);
}
