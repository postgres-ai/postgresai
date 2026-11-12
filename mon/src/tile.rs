//! `pgai-mon tile`: an N-day store built from a recorded one, for long-window benchmarks.
//!
//! The output ends where the recording ends: its last day is the source's last (partial) day,
//! and the days before it are the source's full days, each kept at its own date while the
//! window reaches back that far, then repeated cyclically further back with every time column
//! shifted by whole days. A partial first source day is not used.
//!
//! Derived files (rollups of every tier, the series registry, monthly series files) are not
//! copied. They come from the store's own compaction (`Compactor::run_once`), the code path the
//! product runs: once over the source days kept at their dates (which also rewrites their
//! samples like a closed-day compaction), and once after the shifted days are in. A day's 1 h
//! rollup is shifted with its samples rather than rebuilt: a rollup of a day shifted by whole
//! days is the source day's rollup with its time columns shifted (tests/tile.rs checks this
//! against a rollup compaction builds from the shifted samples). Tiers built from the 1 h
//! rollups (`update_1d`, when present) are built by the final compaction pass.
//!
//! The output gets `format_version` in catalog.json and a `FORMAT` file:
//! `v<N>-base` (1 h rollups at rollup format N), `v<N>-tier1d` (also the 1 d tier) or
//! `v<N>-tier1hm` (also the 1 d tier and 1 h month files), e.g. `v4-tier1hm`.
//! `tile --format-version` prints
//! what this binary writes, by tiling a tiny synthetic store.
use crate::catalog::{Catalog, FileEntry};
use crate::config::StoreCfg;
use crate::model::{day_str, now_ms, DAY_MS};
use arrow::array::{Array, ArrayRef, Date32Array, Date64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, TimeUnit};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub const FORMAT_FILE: &str = "FORMAT";

/// Kinds with one set of files per day, rewritten for every tiled day.
fn per_day_kind(kind: &str, series_per_day: bool) -> bool {
    matches!(kind, "samples" | "logs" | "plan_events" | "plan_defs") || (kind == "series" && series_per_day)
}

/// Kinds compaction derives from the others: never copied.
fn derived_kind(kind: &str) -> bool {
    kind.starts_with("rollup_") || kind == crate::registry::KIND || kind == crate::store::SERIES_MONTH
}

#[derive(Debug)]
pub struct Report {
    /// (output day, source day), newest first.
    pub map: Vec<(i64, i64)>,
    pub files: usize,
    pub bytes: u64,
    pub format: String,
    pub tiers: Vec<String>,
    pub merges: usize,
}

pub struct Opts {
    pub src: PathBuf,
    pub out: PathBuf,
    pub days: usize,
    /// Parallel file rewrites (0 = cores, at most 4).
    pub jobs: usize,
    pub zstd_level: i32,
    pub verbose: bool,
}

/// Output day -> source day: newest first, `days` of them, ending at the source's last day.
pub fn day_map(samples: &BTreeMap<i64, (i64, i64)>, days: usize) -> anyhow::Result<Vec<(i64, i64)>> {
    let (&last, _) = samples
        .iter()
        .next_back()
        .ok_or_else(|| anyhow::anyhow!("source has no samples"))?;
    let full = |d: i64, (mn, mx): (i64, i64)| mn <= d * DAY_MS + 3_600_000 && mx >= (d + 1) * DAY_MS - 3_600_000;
    let pool: Vec<i64> = samples
        .iter()
        .filter(|(&d, &r)| d < last && full(d, r))
        .map(|(&d, _)| d)
        .collect();
    anyhow::ensure!(days >= 1, "--days must be at least 1");
    anyhow::ensure!(
        days == 1 || !pool.is_empty(),
        "source has no full day before its last to repeat"
    );
    let p = pool.len();
    Ok((0..days)
        .map(|j| {
            let d = last - j as i64;
            let s = if j == 0 || pool.binary_search(&d).is_ok() {
                d
            } else {
                pool[(p - 1) - ((j - 1) % p)]
            };
            (d, s)
        })
        .collect())
}

fn compact_all(out: &Path, zstd_level: i32) -> anyhow::Result<usize> {
    let cfg = StoreCfg {
        delete_grace_s: 0,
        zstd_level,
        ..Default::default()
    };
    let core = crate::store::StoreCore::open(out, &cfg)?;
    let c = core.compactor()?;
    let mut total = 0;
    for _ in 0..8 {
        let n = c.run_once(now_ms(), true)?;
        total += n;
        if n == 0 {
            break;
        }
    }
    Ok(total)
}

fn copy_writable(src: &Path, dst: &Path) -> anyhow::Result<u64> {
    if let Some(d) = dst.parent() {
        std::fs::create_dir_all(d)?;
    }
    let n = std::fs::copy(src, dst)?;
    let mut perm = std::fs::metadata(dst)?.permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    perm.set_readonly(false);
    std::fs::set_permissions(dst, perm)?;
    Ok(n)
}

/// `kind/date=<day>/<prefix>-<id>.parquet` for a source file name `<prefix>-<old id>.parquet`.
fn tiled_path(f: &FileEntry, day: i64, id: u64) -> String {
    let name = f.path.rsplit('/').next().unwrap_or("L2-0.parquet");
    let prefix = name.split('-').next().unwrap_or("L2");
    format!("{}/date={}/{prefix}-{id:08}.parquet", f.kind, day_str(day))
}

pub fn tile(o: &Opts) -> anyhow::Result<Report> {
    let say = |m: String| {
        if o.verbose {
            eprintln!("tile: {m}");
        }
    };
    anyhow::ensure!(
        crate::catalog::path(&o.src).exists(),
        "no store at {} (catalog.json missing)",
        o.src.display()
    );
    if o.out.exists() {
        anyhow::ensure!(
            std::fs::read_dir(&o.out)?.next().is_none(),
            "{} exists and is not empty",
            o.out.display()
        );
    }
    std::fs::create_dir_all(&o.out)?;
    let src = Catalog::load_or_new(&o.src, crate::store::DEFAULT_LAYOUT)?;
    let mut samples: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    for f in src.files.iter().filter(|f| f.kind == "samples") {
        let e = samples.entry(f.day).or_insert((i64::MAX, i64::MIN));
        e.0 = e.0.min(f.min_ts);
        e.1 = e.1.max(f.max_ts);
    }
    let map = day_map(&samples, o.days)?;
    let kept: BTreeSet<i64> = map.iter().filter(|(d, s)| d == s).map(|x| x.0).collect();
    let per_day = |k: &str| per_day_kind(k, src.series_per_day);

    // A: the source days kept at their dates, and the files that are not per day, as they are
    let mut base = src.clone();
    base.files.clear();
    base.pending_delete.clear();
    let mut copied = 0u64;
    for f in &src.files {
        if derived_kind(&f.kind) || (per_day(&f.kind) && !kept.contains(&f.day)) {
            continue;
        }
        copied += copy_writable(&o.src.join(&f.path), &o.out.join(&f.path))?;
        let mut g = f.clone();
        g.registered = false;
        base.files.push(g);
    }
    base.save(&o.out)?;
    say(format!(
        "{} source files ({copied} bytes) kept at their dates",
        base.files.len()
    ));

    // B: compaction over them (closed-day merge, 1 h rollups, registry, monthly series)
    let t = std::time::Instant::now();
    let mut merges = compact_all(&o.out, o.zstd_level)?;
    say(format!(
        "first compaction: {merges} merges in {:.1} s",
        t.elapsed().as_secs_f64()
    ));

    // C: the other days, shifted copies of their source day's files and 1 h rollup
    let mut cat = Catalog::load_or_new(&o.out, crate::store::DEFAULT_LAYOUT)?;
    let mut work: Vec<(PathBuf, PathBuf, i64, FileEntry)> = Vec::new();
    for &(d, s) in map.iter().filter(|(d, s)| d != s) {
        let delta = (d - s) * DAY_MS;
        let of_day: Vec<FileEntry> = cat
            .files
            .iter()
            .filter(|f| f.day == s && (per_day(&f.kind) || f.kind == crate::eval::rollup::KIND))
            .cloned()
            .collect();
        anyhow::ensure!(
            of_day.iter().any(|f| f.kind == crate::eval::rollup::KIND),
            "day {} has no 1 h rollup after compaction",
            day_str(s)
        );
        for f in of_day {
            let id = cat.alloc_id();
            let rel = tiled_path(&f, d, id);
            let g = FileEntry {
                path: rel.clone(),
                day: d,
                min_ts: f.min_ts + delta,
                max_ts: f.max_ts + delta,
                created_ms: now_ms(),
                registered: false,
                ..f.clone()
            };
            work.push((o.out.join(&f.path), o.out.join(&rel), delta, g));
        }
    }
    let t = std::time::Instant::now();
    let jobs = if o.jobs == 0 {
        std::thread::available_parallelism().map_or(2, |n| n.get()).clamp(1, 4)
    } else {
        o.jobs
    };
    let next = AtomicUsize::new(0);
    let sizes: Vec<std::sync::Mutex<anyhow::Result<u64>>> = work.iter().map(|_| std::sync::Mutex::new(Ok(0))).collect();
    std::thread::scope(|sc| {
        for _ in 0..jobs.min(work.len().max(1)) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some((a, b, delta, g)) = work.get(i) else { break };
                let r = shift_file(a, b, *delta, g.kind == crate::eval::rollup::KIND, o.zstd_level)
                    .map_err(|e| e.context(format!("shifting {}", a.display())));
                *sizes[i].lock().unwrap() = r;
            });
        }
    });
    for ((_, _, _, mut g), r) in work.into_iter().zip(sizes) {
        g.bytes = r.into_inner().unwrap()?;
        cat.files.push(g);
    }
    cat.save(&o.out)?;
    say(format!(
        "{} days shifted in {:.1} s ({jobs} jobs)",
        map.iter().filter(|(d, s)| d != s).count(),
        t.elapsed().as_secs_f64()
    ));

    // D: tiers built from the 1 h rollups, registry and monthly series of the new days
    let t = std::time::Instant::now();
    let n = compact_all(&o.out, o.zstd_level)?;
    merges += n;
    say(format!(
        "final compaction: {n} merges in {:.1} s",
        t.elapsed().as_secs_f64()
    ));

    let mut cat = Catalog::load_or_new(&o.out, crate::store::DEFAULT_LAYOUT)?;
    let tiers: Vec<String> = cat
        .files
        .iter()
        .filter(|f| f.kind.starts_with("rollup_"))
        .map(|f| f.kind.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    // the 1 h rollup format, lowest over the store: a stale source rollup shows as an old version
    let ver = cat
        .files
        .iter()
        .filter(|f| f.kind == crate::eval::rollup::KIND)
        .map(|f| f.ver)
        .min()
        .unwrap_or(0);
    anyhow::ensure!(
        ver == crate::eval::rollup::VER,
        "1 h rollups at format {ver}, this binary writes {}; run migrate-series on the source",
        crate::eval::rollup::VER
    );
    let format = format_of(ver, &tiers);
    cat.format_version = Some(format.clone());
    cat.save(&o.out)?;
    let (first, last) = (map.last().unwrap().0, map[0].0);
    std::fs::write(
        o.out.join(FORMAT_FILE),
        format!(
            "format={format}\ntiers={}\ndays={}\nfirst_day={}\nlast_day={}\nsource={}\n",
            tiers.join(","),
            o.days,
            day_str(first),
            day_str(last),
            o.src.display()
        ),
    )?;
    // the writer's scratch and (empty) WAL are not part of a served store
    let _ = std::fs::remove_dir_all(o.out.join("tmp"));
    let _ = std::fs::remove_dir_all(o.out.join("wal"));
    Ok(Report {
        map,
        files: cat.files.len(),
        bytes: cat.files.iter().map(|f| f.bytes).sum(),
        format,
        tiers,
        merges,
    })
}

/// `v<rollup format>-base` (1 h rollups), `v<rollup format>-tier1d` (also the 1 d tier) or
/// `v<rollup format>-tier1hm` (also the 1 d tier and the 1 h tier in month files).
fn format_of(ver: u8, tiers: &[String]) -> String {
    let has = |k: &str| tiers.iter().any(|t| t == k);
    let t = if has(crate::eval::rollup::KIND_1HM) && has(crate::eval::rollup::KIND_1D) {
        "tier1hm"
    } else if has(crate::eval::rollup::KIND_1D) {
        "tier1d"
    } else {
        "base"
    };
    format!("v{ver}-{t}")
}

fn shift_col(a: &ArrayRef, delta_ms: i64, int64_ms: bool) -> anyhow::Result<Option<ArrayRef>> {
    let reinterpret = |a: &ArrayRef, to: DataType, by: i64| -> anyhow::Result<ArrayRef> {
        let data = a.to_data().into_builder().data_type(DataType::Int64).build()?;
        let shifted = Int64Array::from(data).unary::<_, arrow::datatypes::Int64Type>(|v| v + by);
        Ok(arrow::array::make_array(
            shifted.to_data().into_builder().data_type(to).build()?,
        ))
    };
    Ok(match a.data_type() {
        DataType::Timestamp(u, _) => {
            let by = match u {
                TimeUnit::Second => delta_ms / 1000,
                TimeUnit::Millisecond => delta_ms,
                TimeUnit::Microsecond => delta_ms * 1000,
                TimeUnit::Nanosecond => delta_ms * 1_000_000,
            };
            Some(reinterpret(a, a.data_type().clone(), by)?)
        }
        DataType::Date32 => {
            let d = a.as_any().downcast_ref::<Date32Array>().unwrap();
            let by = (delta_ms / DAY_MS) as i32;
            Some(Arc::new(d.unary::<_, arrow::datatypes::Date32Type>(|v| v + by)))
        }
        DataType::Date64 => {
            let d = a.as_any().downcast_ref::<Date64Array>().unwrap();
            Some(Arc::new(d.unary::<_, arrow::datatypes::Date64Type>(|v| v + delta_ms)))
        }
        DataType::Int64 if int64_ms => Some(reinterpret(a, DataType::Int64, delta_ms)?),
        _ => None,
    })
}

/// Rewrite one Parquet file with every time column shifted by `delta_ms` (whole days):
/// Timestamp, Date32 and Date64 columns; in rollup files also the Int64 columns holding epoch
/// milliseconds (bucket ends and sample times: every value in [2001, 5138], from the row groups'
/// stats or, for columns written without stats, from the data).
/// Row-group size, codecs, value encodings, dictionary use, statistics level and Bloom filters
/// follow the source file's metadata. Returns the new file's size.
pub fn shift_file(src: &Path, dst: &Path, delta_ms: i64, rollup: bool, zstd_level: i32) -> anyhow::Result<u64> {
    anyhow::ensure!(delta_ms % DAY_MS == 0, "shift by whole days only");
    let b = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(src)?)?;
    let md = b.metadata().clone();
    let schema = b.schema().clone();
    let rg_rows = md.row_groups().iter().map(|r| r.num_rows()).max().unwrap_or(1).max(1) as usize;
    let fm = md.file_metadata();
    let mut props = WriterProperties::builder()
        .set_writer_version(if fm.version() >= 2 {
            WriterVersion::PARQUET_2_0
        } else {
            WriterVersion::PARQUET_1_0
        })
        .set_max_row_group_row_count(Some(rg_rows));
    // epoch-ms Int64 leaves (rollups), by leaf index
    let mut ms_leaf = vec![false; fm.schema_descr().num_columns()];
    // rollup leaves without statistics (rollups keep them only on series_id and k): their range
    // comes from the data
    let mut unknown = vec![false; ms_leaf.len()];
    if let Some(rg0) = md.row_groups().first() {
        for c in rg0.columns() {
            let path = c.column_path().clone();
            let codec = match c.compression() {
                Compression::ZSTD(_) => Compression::ZSTD(ZstdLevel::try_new(zstd_level)?),
                x => x,
            };
            props = props.set_column_compression(path.clone(), codec);
            let encs: Vec<Encoding> = c.encodings().collect();
            let dict = encs
                .iter()
                .any(|e| matches!(e, Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY));
            props = props.set_column_dictionary_enabled(path.clone(), dict);
            if !dict {
                if let Some(e) = encs.iter().find(|e| {
                    matches!(
                        e,
                        Encoding::DELTA_BINARY_PACKED
                            | Encoding::BYTE_STREAM_SPLIT
                            | Encoding::DELTA_LENGTH_BYTE_ARRAY
                            | Encoding::DELTA_BYTE_ARRAY
                    )
                }) {
                    props = props.set_column_encoding(path.clone(), *e);
                }
            }
            let stats = if c.column_index_offset().is_some() {
                EnabledStatistics::Page
            } else if c.statistics().is_some() {
                EnabledStatistics::Chunk
            } else {
                EnabledStatistics::None
            };
            props = props.set_column_statistics_enabled(path.clone(), stats);
            if c.bloom_filter_offset().is_some() {
                props = props
                    .set_column_bloom_filter_enabled(path.clone(), true)
                    .set_column_bloom_filter_ndv(path.clone(), rg_rows as u64);
            }
        }
        if rollup {
            for (j, l) in ms_leaf.iter_mut().enumerate() {
                let (mut mn, mut mx, mut all) = (i64::MAX, i64::MIN, true);
                for rg in md.row_groups() {
                    match rg.column(j).statistics() {
                        Some(parquet::file::statistics::Statistics::Int64(s)) => match (s.min_opt(), s.max_opt()) {
                            (Some(a), Some(b)) => {
                                mn = mn.min(*a);
                                mx = mx.max(*b);
                            }
                            _ => {}
                        },
                        _ => all = false,
                    }
                }
                *l = all && is_ms(mn, mx);
                unknown[j] = !all;
            }
        }
    }
    // top-level fields -> their (single) leaf, for the Int64 rule
    let roots: Vec<Option<usize>> = {
        let sd = fm.schema_descr();
        let mut first: Vec<Option<usize>> = vec![None; schema.fields().len()];
        let mut count = vec![0usize; schema.fields().len()];
        for leaf in 0..sd.num_columns() {
            let r = sd.get_column_root_idx(leaf);
            if r < first.len() {
                first[r].get_or_insert(leaf);
                count[r] += 1;
            }
        }
        first
            .into_iter()
            .zip(count)
            .map(|(f, n)| f.filter(|_| n == 1))
            .collect()
    };
    if unknown.iter().any(|x| *x) {
        let mut rng: Vec<(i64, i64)> = vec![(i64::MAX, i64::MIN); ms_leaf.len()];
        let scan = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(src)?)?;
        for batch in scan.with_batch_size(65_536).build()? {
            let batch = batch?;
            for (i, a) in batch.columns().iter().enumerate() {
                let Some(l) = roots.get(i).copied().flatten().filter(|l| unknown[*l]) else { continue };
                if let Some(x) = a.as_any().downcast_ref::<Int64Array>() {
                    if let (Some(a), Some(b)) = (arrow::compute::min(x), arrow::compute::max(x)) {
                        rng[l] = (rng[l].0.min(a), rng[l].1.max(b));
                    }
                }
            }
        }
        for (l, (mn, mx)) in rng.into_iter().enumerate() {
            if unknown[l] {
                ms_leaf[l] = is_ms(mn, mx);
            }
        }
    }
    if let Some(d) = dst.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = dst.with_extension("parquet.tmp");
    let mut w = ArrowWriter::try_new(std::fs::File::create(&tmp)?, schema.clone(), Some(props.build()))?;
    for batch in b.with_batch_size(rg_rows.min(131_072)).build()? {
        let batch = batch?;
        let mut cols = Vec::with_capacity(batch.num_columns());
        for (i, a) in batch.columns().iter().enumerate() {
            let ms = roots.get(i).copied().flatten().is_some_and(|l| ms_leaf[l]);
            cols.push(shift_col(a, delta_ms, ms)?.unwrap_or_else(|| a.clone()));
        }
        w.write(&RecordBatch::try_new(schema.clone(), cols)?)?;
    }
    w.close()?;
    std::fs::rename(&tmp, dst)?;
    Ok(std::fs::metadata(dst)?.len())
}

/// Epoch milliseconds (2001..5138) over the whole range.
fn is_ms(mn: i64, mx: i64) -> bool {
    mn >= 1_000_000_000_000 && mx < 100_000_000_000_000 && mn <= mx
}

/// What this binary's tiler writes (`tile --format-version`): tiles a tiny synthetic store.
pub fn format_version() -> anyhow::Result<String> {
    use crate::model::{series_id, Record, Sample, SeriesDef};
    let dir = std::env::temp_dir().join(format!("pgai-mon-tile-fmt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let res = (|| {
        let src = dir.join("src");
        {
            let mut core = crate::store::StoreCore::open(&src, &StoreCfg::default())?;
            let day0 = 1_794_009_600_000i64 - 3 * DAY_MS;
            let labels = vec![("i".to_string(), "0".to_string())];
            let id = series_id("t_total", &labels);
            core.append(Record::Series(vec![SeriesDef {
                series_id: id,
                metric: "t_total".into(),
                labels,
                first_seen: day0,
            }]))?;
            let samples: Vec<Sample> = (0..(2 * 24 * 60 + 600))
                .map(|k| Sample {
                    series_id: id,
                    ts: day0 + k * 60_000,
                    value: k as f64,
                })
                .collect();
            core.append(Record::Samples(samples))?;
            core.flush()?;
        }
        let r = tile(&Opts {
            src,
            out: dir.join("out"),
            days: 4,
            jobs: 1,
            zstd_level: 3,
            verbose: false,
        })?;
        Ok(r.format)
    })();
    let _ = std::fs::remove_dir_all(&dir);
    res
}
