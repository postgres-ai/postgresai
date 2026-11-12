//! Loader for VictoriaMetrics exports, and the crash-test writer/verifier.
use crate::model::*;
use crate::store::{file_list_sql, samples_raw_select, StoreCore};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::BufRead;

#[derive(Deserialize)]
#[serde(untagged)]
enum Num {
    F(f64),
    S(String),
    Null(()),
}

#[derive(Deserialize)]
struct ExportLine {
    metric: BTreeMap<String, String>,
    values: Vec<Num>,
    timestamps: Vec<i64>,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct LoadStats {
    pub lines: u64,
    pub series_new: u64,
    pub samples: u64,
    pub skipped_values: u64,
    /// VM staleness markers (exported as null), kept as NaN samples.
    pub stale_markers: u64,
    pub secs: f64,
}

/// Import `/api/v1/export` JSON lines. Labels are kept verbatim (`__name__` -> metric).
pub fn load_vm_export(
    store: &mut StoreCore,
    input: &mut dyn BufRead,
    chunk: usize,
    compact: bool,
) -> anyhow::Result<LoadStats> {
    let t0 = std::time::Instant::now();
    let mut st = LoadStats::default();
    let mut samples: Vec<Sample> = Vec::with_capacity(chunk + 50_000);
    let mut series: Vec<SeriesDef> = Vec::new();
    let known = store.shared.known_series.clone();
    let mut line = String::new();
    loop {
        line.clear();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let l: ExportLine = serde_json::from_str(&line)?;
        st.lines += 1;
        let mut metric = String::new();
        let mut labels = Vec::with_capacity(l.metric.len());
        for (k, v) in l.metric {
            if k == "__name__" {
                metric = v;
            } else {
                labels.push((k, v));
            }
        }
        let id = series_id(&metric, &labels);
        // one SeriesDef per day the series has samples in (per-day series files)
        let mut first_by_day: std::collections::BTreeMap<i64, i64> = std::collections::BTreeMap::new();
        for ts in &l.timestamps {
            let e = first_by_day.entry(day_of(*ts)).or_insert(*ts);
            *e = (*e).min(*ts);
        }
        {
            let mut k = known.lock().unwrap();
            for (day, first) in first_by_day {
                if k.insert((day, id)) {
                    st.series_new += 1;
                    series.push(SeriesDef {
                        series_id: id,
                        metric: metric.clone(),
                        labels: labels.clone(),
                        first_seen: first,
                    });
                }
            }
        }
        for (v, ts) in l.values.iter().zip(l.timestamps.iter()) {
            let value = match v {
                Num::F(f) => *f,
                Num::S(s) => match s.parse::<f64>() {
                    Ok(f) => f,
                    Err(_) => {
                        st.skipped_values += 1;
                        continue;
                    }
                },
                // VictoriaMetrics exports staleness markers as null. Keep them as NaN: PromQL
                // ends a series at the marker (bare selectors), and the SQL macros do too.
                Num::Null(_) => {
                    st.stale_markers += 1;
                    stale_nan()
                }
            };
            samples.push(Sample {
                series_id: id,
                ts: *ts,
                value,
            });
        }
        if samples.len() >= chunk {
            st.samples += samples.len() as u64;
            store.bulk_write(&samples, &series)?;
            samples.clear();
            series.clear();
            eprintln!(
                "load: {} lines, {} samples, {:.0}s",
                st.lines,
                st.samples,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    if !samples.is_empty() || !series.is_empty() {
        st.samples += samples.len() as u64;
        store.bulk_write(&samples, &series)?;
    }
    if compact {
        eprintln!("load: compacting");
        store.compactor()?.run_once(now_ms(), true)?;
    }
    st.secs = t0.elapsed().as_secs_f64();
    Ok(st)
}

/// Crash-test writer: appends one record per step with `n` series at ts = base + step*1000,
/// value = step, flushing every `flush_every` steps, until killed. Resumes after the last
/// step found in the store (Parquet + replayed WAL).
/// Side file with every acknowledged step (O_APPEND + fsync after each append returns), so
/// verify can tell "nothing acknowledged was lost" from "no holes in what survived".
pub fn acked_path(store: &StoreCore) -> std::path::PathBuf {
    store.shared.root.join("crashtest_acked.log")
}

pub fn crashtest_write(
    store: &mut StoreCore,
    n: u64,
    flush_every: u64,
    steps: u64,
    final_flush: bool,
) -> anyhow::Result<()> {
    use std::io::Write;
    let mut acked = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(acked_path(store))?;
    let base: i64 = 1_800_000_000_000;
    let in_store = max_ts(store)?;
    let resume = in_store.into_iter().chain(store.max_buffered_ts()).max();
    let mut step = match resume {
        Some(t) => ((t - base) / 1000 + 1) as u64,
        None => 0,
    };
    eprintln!("crashtest: resuming at step {step}");
    let ids: Vec<u64> = (0..n)
        .map(|i| series_id("crashtest_value", &[("i".to_string(), i.to_string())]))
        .collect();
    let first = {
        let mut k = store.shared.known_series.lock().unwrap();
        ids.iter()
            .enumerate()
            .filter(|(_, id)| k.insert((day_of(base), **id)))
            .map(|(i, id)| SeriesDef {
                series_id: *id,
                metric: "crashtest_value".into(),
                labels: vec![("i".into(), i.to_string())],
                first_seen: base,
            })
            .collect::<Vec<_>>()
    };
    if !first.is_empty() {
        store.append(Record::Series(first))?;
    }
    let end = step + steps;
    while step < end {
        let ts = base + step as i64 * 1000;
        store.append(Record::Samples(
            ids.iter()
                .map(|id| Sample {
                    series_id: *id,
                    ts,
                    value: step as f64,
                })
                .collect(),
        ))?;
        // append() returned: the step is acknowledged (durable in the WAL)
        writeln!(acked, "{step}")?;
        acked.sync_data()?;
        step += 1;
        if step % flush_every == 0 {
            store.flush()?;
        }
    }
    if final_flush {
        store.flush()?;
    }
    Ok(())
}

fn max_ts(store: &StoreCore) -> anyhow::Result<Option<i64>> {
    let cat = store.shared.catalog.lock().unwrap().clone();
    let files: Vec<_> = cat.files_of("samples").collect();
    if files.is_empty() {
        return Ok(None);
    }
    let src = format!("read_parquet({})", file_list_sql(&store.shared.root, &files));
    let q = format!(
        "SELECT epoch_ms(max(ts)) FROM ({})",
        samples_raw_select(&cat.samples_layout, &src)
    );
    Ok(store.conn().query_row(&q, [], |r| r.get::<_, Option<i64>>(0))?)
}

/// Verifies the crash-test invariant: every series has each step exactly once, no gaps.
pub fn crashtest_verify(store: &StoreCore) -> anyhow::Result<serde_json::Value> {
    let last_acked: Option<f64> = std::fs::read_to_string(acked_path(store))
        .ok()
        .and_then(|s| s.lines().filter_map(|l| l.trim().parse::<f64>().ok()).next_back());
    let cat = store.shared.catalog.lock().unwrap().clone();
    let files: Vec<_> = cat.files_of("samples").collect();
    anyhow::ensure!(!files.is_empty(), "no samples in store");
    let src = format!("read_parquet({})", file_list_sql(&store.shared.root, &files));
    let q = format!(
        "WITH s AS ({}), per AS (SELECT series_id, count(*) n, count(DISTINCT ts) d, \
           (epoch_ms(max(ts)) - epoch_ms(min(ts))) / 1000 + 1 AS span, min(value) mn, max(value) mx, \
           bool_and(value = (epoch_ms(ts) - 1800000000000) / 1000) AS val_ok FROM s GROUP BY 1) \
         SELECT count(*) AS series, sum(n) AS samples, sum(n - d) AS duplicates, sum(span - d) AS gaps, \
           min(mx) AS min_last_step, max(mx) AS max_last_step, bool_and(val_ok) AS values_ok FROM per",
        samples_raw_select(&cat.samples_layout, &src)
    );
    let v = store.conn().query_row(&q, [], |r| {
        Ok(serde_json::json!({
            "series": r.get::<_, i64>(0)?,
            "samples": r.get::<_, f64>(1)?,
            "duplicates": r.get::<_, f64>(2)?,
            "gaps": r.get::<_, f64>(3)?,
            "min_last_step": r.get::<_, f64>(4)?,
            "max_last_step": r.get::<_, f64>(5)?,
            "values_ok": r.get::<_, bool>(6)?,
            "files": files.len(),
        }))
    })?;
    let mut v = v;
    let max_last = v["max_last_step"].as_f64().unwrap_or(-1.0);
    let min_last = v["min_last_step"].as_f64().unwrap_or(-1.0);
    let lost = last_acked.is_some_and(|a| min_last < a);
    v["last_acked_step"] = serde_json::json!(last_acked);
    v["acked_lost"] = serde_json::json!(lost);
    v["pass"] = serde_json::json!(
        !lost
            && v["duplicates"].as_f64() == Some(0.0)
            && v["gaps"].as_f64() == Some(0.0)
            && v["values_ok"].as_bool() == Some(true)
            && min_last == max_last
    );
    Ok(v)
}

#[derive(Debug, Default, serde::Serialize)]
pub struct LogLoadStats {
    pub lines: u64,
    pub plans: u64,
    pub bytes_read: u64,
    pub flushes: u64,
    pub secs: f64,
}

/// Offline jsonlog import (a file, or every `*suffix` file of a directory) through the same
/// path as live tailing: bounded reads, WAL, size-capped flushes. Resumable (offsets are in
/// the catalog), so rerunning after a kill continues where it stopped.
pub fn load_logs(
    store: &mut StoreCore,
    input: &std::path::Path,
    suffix: &str,
    compact: bool,
) -> anyhow::Result<LogLoadStats> {
    let t0 = std::time::Instant::now();
    let (dir, suffix) = if input.is_file() {
        (
            input.parent().unwrap_or(std::path::Path::new(".")).to_path_buf(),
            format!("/{}", input.file_name().unwrap().to_string_lossy()),
        )
    } else {
        (input.to_path_buf(), suffix.to_string())
    };
    let mut t = crate::logs::Tailer {
        dir,
        suffix,
        offsets: store.log_offsets().into_iter().collect(),
        start_at_end: false,
        known_plans: Default::default(),
        fp_map: store.shared.fp_to_queryid.clone(),
        max_read: 4 << 20,
    };
    let mut st = LogLoadStats::default();
    let flushes0 = store.shared.stats.flushes.load(std::sync::atomic::Ordering::Relaxed);
    loop {
        let recs = t.poll()?;
        if recs.is_empty() {
            break;
        }
        for r in recs {
            if let Record::Logs { rows, plan_events, .. } = &r {
                st.lines += rows.len() as u64;
                st.plans += plan_events.len() as u64;
                st.bytes_read += rows.iter().map(|r| r.raw.len() as u64 + 1).sum::<u64>();
            }
            store.append(r)?;
            store.flush_if_full()?;
        }
        if st.lines % 200_000 < 20_000 {
            eprintln!("load-logs: {} lines, {:.0}s", st.lines, t0.elapsed().as_secs_f64());
        }
    }
    store.flush()?;
    st.flushes = store.shared.stats.flushes.load(std::sync::atomic::Ordering::Relaxed) - flushes0;
    if compact {
        store.compactor()?.run_once(now_ms(), true)?;
    }
    st.secs = t0.elapsed().as_secs_f64();
    Ok(st)
}

/// Rewrite a store's series files as per-day files: for every day with samples, the series that
/// have samples that day (labels from the old registry). Sets `series_per_day`. For stores
/// written before per-day series.
pub fn migrate_series_per_day(store: &mut StoreCore) -> anyhow::Result<serde_json::Value> {
    let cat = store.shared.catalog.lock().unwrap().clone();
    if cat.series_per_day {
        return Ok(serde_json::json!({"already": true}));
    }
    let root = store.shared.root.clone();
    let old: Vec<_> = cat.files_of("series").cloned().collect();
    let reg = format!(
        "read_parquet({})",
        file_list_sql(&root, &old.iter().collect::<Vec<_>>())
    );
    let mut days: std::collections::BTreeMap<i64, Vec<crate::catalog::FileEntry>> = Default::default();
    for f in cat.files_of("samples") {
        days.entry(f.day).or_default().push(f.clone());
    }
    let mut new_entries = Vec::new();
    let db = store.conn();
    db.execute_batch(&format!(
        "CREATE OR REPLACE TEMP TABLE reg AS SELECT DISTINCT ON (series_id) series_id, metric, labels FROM {reg}"
    ))?;
    let mut rows_total = 0u64;
    for (day, files) in days {
        let id = {
            let mut c = store.shared.catalog.lock().unwrap();
            c.alloc_id()
        };
        let rel = format!("series/date={}/L2-{id:08}.parquet", day_str(day));
        let abs = root.join(&rel);
        let src = file_list_sql(&root, &files.iter().collect::<Vec<_>>());
        let sel = format!(
            "SELECT r.series_id, r.metric, r.labels, make_timestamptz({}::BIGINT * 1000) AS first_seen \
             FROM (SELECT DISTINCT series_id FROM read_parquet({src})) d JOIN reg r USING (series_id) ORDER BY 1",
            day * DAY_MS
        );
        let bytes = store.copy_atomic(&sel, &abs, "v2")?;
        let rows: i64 = db.query_row(
            &format!("SELECT count(*) FROM read_parquet('{}')", abs.display()),
            [],
            |r| r.get(0),
        )?;
        rows_total += rows as u64;
        new_entries.push(crate::catalog::FileEntry {
            kind: "series".into(),
            path: rel,
            day,
            level: 2,
            rows: rows as u64,
            bytes,
            min_ts: day * DAY_MS,
            max_ts: day * DAY_MS,
            created_ms: now_ms(),
            registered: false,
            ver: 0,
        });
    }
    let mut c = store.shared.catalog.lock().unwrap();
    c.files.retain(|f| f.kind != "series");
    c.files.extend(new_entries.clone());
    for f in &old {
        c.pending_delete.push((f.path.clone(), now_ms()));
    }
    c.series_per_day = true;
    c.save(&root)?;
    Ok(serde_json::json!({"days": new_entries.len(), "rows": rows_total, "old_files": old.len()}))
}

/// Churn test data: add `n` registered series of `metric` that never get samples (new queryid
/// values on real label sets of that metric), spread over `days` per-day series files ending at
/// `last_day` (so a 24 h window sees n/days of them). Reproduces "a year of queryid churn".
pub fn synth_churn(
    store: &mut StoreCore,
    metric: &str,
    n: u64,
    days: i64,
    last_day: i64,
) -> anyhow::Result<serde_json::Value> {
    let root = store.shared.root.clone();
    let cat = store.shared.catalog.lock().unwrap().clone();
    let series: Vec<_> = cat.files_of("series").collect();
    let reg = format!("read_parquet({})", file_list_sql(&root, &series));
    let db = store.conn();
    db.execute_batch(&format!(
        "CREATE OR REPLACE TEMP TABLE tmpl AS SELECT DISTINCT ON (series_id) series_id, labels FROM {reg} \
         WHERE metric = '{metric}' AND labels['queryid'] IS NOT NULL"
    ))?;
    let nt: i64 = db.query_row("SELECT count(*) FROM tmpl", [], |r| r.get(0))?;
    anyhow::ensure!(nt > 0, "no {metric} series with a queryid to copy");
    let per_day = (n as i64 + days - 1) / days;
    let mut entries = Vec::new();
    for k in 0..days {
        let day = last_day - (days - 1 - k);
        let id = {
            let mut c = store.shared.catalog.lock().unwrap();
            c.alloc_id()
        };
        let rel = format!("series/date={}/C-{id:08}.parquet", day_str(day));
        let abs = root.join(&rel);
        // ids keep the metric's 16-bit prefix (as real ids do); the low 48 bits are a hash
        let sel = format!(
            "WITH t AS (SELECT row_number() OVER () - 1 AS j, series_id, labels FROM tmpl), \
                  g AS (SELECT range AS i FROM range({per_day})) \
             SELECT ((t.series_id >> 48) << 48) | (hash(g.i, {day}::BIGINT, 'churn') & 281474976710655::UBIGINT) AS series_id, \
                    '{metric}' AS metric, \
                    map_from_entries(list_transform(map_entries(t.labels), e -> \
                      CASE WHEN e.key = 'queryid' THEN {{'key': 'queryid', 'value': ('-9' || {day}::VARCHAR || lpad(g.i::VARCHAR, 9, '0'))}} \
                      ELSE {{'key': e.key, 'value': e.value}} END)) AS labels, \
                    make_timestamptz({}::BIGINT * 1000) AS first_seen \
             FROM g JOIN t ON t.j = g.i % {nt} ORDER BY 1",
            day * DAY_MS
        );
        let bytes = store.copy_atomic(&sel, &abs, "v2")?;
        entries.push(crate::catalog::FileEntry {
            kind: "series".into(),
            path: rel,
            day,
            level: 0, // compaction folds them into the day's series file
            rows: per_day as u64,
            bytes,
            min_ts: day * DAY_MS,
            max_ts: day * DAY_MS,
            created_ms: now_ms(),
            registered: false,
            ver: 0,
        });
    }
    let bytes: u64 = entries.iter().map(|e| e.bytes).sum();
    let mut c = store.shared.catalog.lock().unwrap();
    c.files.extend(entries);
    c.save(&root)?;
    Ok(serde_json::json!({"added": per_day * days, "days": days, "per_day": per_day, "bytes": bytes}))
}

/// Rewrite series files whose row groups are larger than SERIES_ROW_GROUP (written by older
/// builds or external generators), then register every closed day's series (registry.rs).
/// Part of `migrate-series`; idempotent.
pub fn series_fix_and_register(store: &mut StoreCore) -> anyhow::Result<serde_json::Value> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    let root = store.shared.root.clone();
    let files: Vec<crate::catalog::FileEntry> = store
        .shared
        .catalog
        .lock()
        .unwrap()
        .files_of("series")
        .cloned()
        .collect();
    let mut rewritten = 0;
    for f in files {
        let abs = root.join(&f.path);
        let big = {
            let r = SerializedFileReader::new(std::fs::File::open(&abs)?)?;
            r.metadata()
                .row_groups()
                .iter()
                .any(|g| g.num_rows() as usize > crate::store::SERIES_ROW_GROUP)
        };
        if !big {
            continue;
        }
        let id = store.shared.catalog.lock().unwrap().alloc_id();
        let rel = format!("series/date={}/L2-{id:08}.parquet", crate::model::day_str(f.day));
        let out = root.join(&rel);
        let tmp = out.with_extension("parquet.tmp");
        crate::kmerge::merge_sorted_opts(
            std::slice::from_ref(&abs),
            &tmp,
            9,
            crate::store::SERIES_ROW_GROUP,
            true,
        )?;
        std::fs::File::open(&tmp)?.sync_all()?;
        std::fs::rename(&tmp, &out)?;
        let mut cat = store.shared.catalog.lock().unwrap();
        if let Some(e) = cat.files.iter_mut().find(|x| x.path == f.path) {
            e.path = rel;
            e.bytes = std::fs::metadata(&out)?.len();
            e.created_ms = now_ms();
            e.level = 2;
        }
        cat.pending_delete.push((f.path.clone(), 0));
        cat.save(&root)?;
        rewritten += 1;
    }
    let comp = store.compactor()?;
    let t = std::time::Instant::now();
    let registered = comp.register(now_ms(), false)?;
    let reg_s = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let months = comp.series_months(now_ms(), false)?;
    let months_s = t.elapsed().as_secs_f64();
    // rollups of an older format (no second-to-last sample: irate can't use them): rebuild from
    // the day's single closed samples file
    let t = std::time::Instant::now();
    let (mut rebuilt, mut bytes_before, mut bytes_after, mut samples) = (0, 0u64, 0u64, 0u64);
    let old: Vec<crate::catalog::FileEntry> = {
        let c = store.shared.catalog.lock().unwrap();
        c.files
            .iter()
            .filter(|f| {
                f.kind == crate::eval::rollup::KIND
                    && (f.ver < crate::eval::rollup::VER || std::env::var_os("PGAI_REBUILD_ROLLUPS").is_some())
            })
            .cloned()
            .collect()
    };
    for r in old {
        let src: Vec<crate::catalog::FileEntry> = {
            let c = store.shared.catalog.lock().unwrap();
            c.files
                .iter()
                .filter(|f| f.kind == "samples" && f.day == r.day)
                .cloned()
                .collect()
        };
        if src.len() != 1 || src[0].level != 2 {
            continue;
        }
        let id = store.shared.catalog.lock().unwrap().alloc_id();
        let rel = format!(
            "{}/date={}/R-{id:08}.parquet",
            crate::eval::rollup::KIND,
            crate::model::day_str(r.day)
        );
        let out = root.join(&rel);
        let tmp = out.with_extension("parquet.tmp");
        let rows = crate::eval::rollup::build(&root.join(&src[0].path), &tmp, 9)?;
        std::fs::File::open(&tmp)?.sync_all()?;
        std::fs::rename(&tmp, &out)?;
        let mut cat = store.shared.catalog.lock().unwrap();
        if let Some(e) = cat.files.iter_mut().find(|x| x.path == r.path) {
            bytes_before += e.bytes;
            e.path = rel;
            e.rows = rows;
            e.bytes = std::fs::metadata(&out)?.len();
            e.ver = crate::eval::rollup::VER;
            e.created_ms = now_ms();
            bytes_after += e.bytes;
            samples += src[0].rows;
        }
        cat.pending_delete.push((r.path.clone(), 0));
        cat.save(&root)?;
        rebuilt += 1;
    }
    let per = |b: u64| if samples > 0 { b as f64 / samples as f64 } else { 0.0 };
    Ok(serde_json::json!({
        "rewritten_row_groups": rewritten, "registered_files": registered, "register_s": reg_s,
        "series_months": months, "series_months_s": months_s,
        "rollups_rebuilt": rebuilt, "rollup_rebuild_s": t.elapsed().as_secs_f64(),
        "rollup_bytes_per_sample_before": per(bytes_before), "rollup_bytes_per_sample_after": per(bytes_after),
    }))
}

/// Gate (b) of review B6: churned pg_stat_statements series that carry samples, written straight
/// to final day files (one samples file and one series file per day, as compaction leaves them;
/// run `compact` after for rollups and the registry). `base` queryids live all `days`; each day
/// `new_per_day` new queryids appear at a random time and live until the end of that day (so a
/// churned series has ~half a day of samples). One sample per `step_s` per series. Counters:
/// calls and exec_time grow; plan_time stays 0 for 70 % of the queryids (ties at every step).
/// Ids are series_id(metric, labels) like the collector's.
pub fn synth_sampled(
    root: &std::path::Path,
    metrics: &[&str],
    days: i64,
    last_day: i64,
    base: u64,
    new_per_day: u64,
    step_s: i64,
) -> anyhow::Result<serde_json::Value> {
    use arrow::array::{
        Array, ArrayRef, Float64Array, Int64Array, MapBuilder, RecordBatch, StringBuilder, TimestampMicrosecondArray,
        UInt64Array,
    };
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::{Compression, Encoding, ZstdLevel};
    use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
    use parquet::schema::types::ColumnPath;
    use std::sync::Arc;
    let store = StoreCore::open(root, &Default::default())?;
    {
        let mut c = store.shared.catalog.lock().unwrap();
        anyhow::ensure!(c.files.is_empty(), "synth-sampled wants an empty store");
        c.series_per_day = true;
    }
    let tz: Arc<str> = Arc::from("UTC");
    let s_schema = Arc::new(Schema::new(vec![
        Field::new("series_id", DataType::UInt64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz.clone())),
            false,
        ),
        Field::new("ival", DataType::Int64, true),
        Field::new("fval", DataType::Float64, true),
    ]));
    let mut rnd_state = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = move || {
        rnd_state ^= rnd_state << 13;
        rnd_state ^= rnd_state >> 7;
        rnd_state ^= rnd_state << 17;
        rnd_state
    };
    let labels_of = |qid: i64, db: &str| -> Vec<(String, String)> {
        let mut l: Vec<(String, String)> = [
            ("cluster", "local"),
            ("datname", db),
            ("dbname", db),
            ("env", "demo"),
            ("instance", "pgwatch:9091"),
            ("job", "pgwatch-prometheus"),
            ("node_name", "node-01"),
            ("real_dbname", db),
            ("sink_type", "prometheus"),
            ("sys_id", "7705963719539601447"),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        l.push(("queryid".into(), qid.to_string()));
        l.sort();
        l
    };
    // queryid -> (db, born ms, dies ms, plan_zero)
    struct Q {
        qid: i64,
        db: &'static str,
        from: i64,
        to: i64,
        plan_zero: bool,
        rate: f64,
    }
    let first_day = last_day - days + 1;
    let mut qs: Vec<Q> = (0..base)
        .map(|i| Q {
            qid: 1_000_000_000_000 + i as i64,
            db: if i % 10 == 0 { "postgres" } else { "target_database" },
            from: first_day * DAY_MS,
            to: (last_day + 1) * DAY_MS,
            plan_zero: i % 10 < 7,
            rate: 1.0 + (i % 50) as f64,
        })
        .collect();
    for d in first_day..=last_day {
        for j in 0..new_per_day {
            let r = rnd();
            qs.push(Q {
                qid: -(d * 1_000_000 + j as i64),
                db: if j % 10 == 0 { "postgres" } else { "target_database" },
                from: d * DAY_MS + (r % DAY_MS as u64) as i64,
                to: (d + 1) * DAY_MS,
                plan_zero: j % 10 < 7,
                rate: 0.5 + (r % 20) as f64,
            });
        }
    }
    let (mut n_samples, mut n_series) = (0u64, 0u64);
    let step = step_s * 1000;
    let mut entries = Vec::new();
    for d in first_day..=last_day {
        let (d0, d1) = (d * DAY_MS, (d + 1) * DAY_MS);
        // (id, metric idx, q idx) for every series alive this day, sorted by id
        let mut live: Vec<(u64, usize, usize)> = Vec::new();
        for (qi, q) in qs.iter().enumerate() {
            if q.to <= d0 || q.from >= d1 {
                continue;
            }
            let l = labels_of(q.qid, q.db);
            for (mi, m) in metrics.iter().enumerate() {
                live.push((series_id(m, &l), mi, qi));
            }
        }
        live.sort_unstable();
        // samples
        let id = store.shared.catalog.lock().unwrap().alloc_id();
        let rel = format!("samples/date={}/L2-{id:08}.parquet", day_str(d));
        let abs = root.join(&rel);
        std::fs::create_dir_all(abs.parent().unwrap())?;
        let mut props = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(9)?))
            .set_statistics_enabled(EnabledStatistics::Chunk)
            .set_max_row_group_row_count(Some(122_880));
        for (c, e) in [
            ("ts", Encoding::DELTA_BINARY_PACKED),
            ("ival", Encoding::DELTA_BINARY_PACKED),
            ("fval", Encoding::BYTE_STREAM_SPLIT),
        ] {
            props = props
                .set_column_dictionary_enabled(ColumnPath::from(c), false)
                .set_column_encoding(ColumnPath::from(c), e);
        }
        let mut w = ArrowWriter::try_new(std::fs::File::create(&abs)?, s_schema.clone(), Some(props.build()))?;
        let (mut sid, mut ts, mut iv, mut fv) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut min_ts, mut max_ts, mut rows) = (i64::MAX, i64::MIN, 0u64);
        let flush = |sid: &mut Vec<u64>,
                     ts: &mut Vec<i64>,
                     iv: &mut Vec<Option<i64>>,
                     fv: &mut Vec<Option<f64>>,
                     w: &mut ArrowWriter<std::fs::File>|
         -> anyhow::Result<()> {
            if sid.is_empty() {
                return Ok(());
            }
            let cols: Vec<ArrayRef> = vec![
                Arc::new(UInt64Array::from(std::mem::take(sid))),
                Arc::new(TimestampMicrosecondArray::from(std::mem::take(ts)).with_timezone("UTC")),
                Arc::new(Int64Array::from(std::mem::take(iv))),
                Arc::new(Float64Array::from(std::mem::take(fv))),
            ];
            w.write(&RecordBatch::try_new(s_schema.clone(), cols)?)?;
            Ok(())
        };
        for &(id, mi, qi) in &live {
            let q = &qs[qi];
            let from = q.from.max(d0);
            let to = q.to.min(d1);
            // first sample on the series' own phase, then every step
            let phase = (id % step as u64) as i64;
            let mut t = from - from.rem_euclid(step) + phase;
            if t < from {
                t += step;
            }
            // counter value at t: rate * seconds alive (+ a deterministic wobble)
            while t < to {
                let alive_s = (t - q.from) / 1000;
                let v: f64 = match mi {
                    0 => (q.rate * alive_s as f64 / 30.0).floor(),                 // calls
                    1 => q.rate * 0.37 * alive_s as f64 + ((t / step) % 7) as f64, // exec_time ms
                    _ => {
                        if q.plan_zero {
                            0.0
                        } else {
                            q.rate * 0.05 * alive_s as f64
                        }
                    } // plan_time
                };
                sid.push(id);
                ts.push(t * 1000);
                if v.fract() == 0.0 && v.abs() < 9e15 {
                    iv.push(Some(v as i64));
                    fv.push(None);
                } else {
                    iv.push(None);
                    fv.push(Some(v));
                }
                min_ts = min_ts.min(t);
                max_ts = max_ts.max(t);
                rows += 1;
                if sid.len() >= 65_536 {
                    flush(&mut sid, &mut ts, &mut iv, &mut fv, &mut w)?;
                }
                t += step;
            }
        }
        flush(&mut sid, &mut ts, &mut iv, &mut fv, &mut w)?;
        w.close()?;
        n_samples += rows;
        entries.push(crate::catalog::FileEntry {
            kind: "samples".into(),
            path: rel,
            day: d,
            level: 2,
            rows,
            bytes: std::fs::metadata(&abs)?.len(),
            min_ts,
            max_ts,
            created_ms: now_ms(),
            registered: false,
            ver: 0,
        });
        // series rows of the day (labels map), sorted by id
        let id = store.shared.catalog.lock().unwrap().alloc_id();
        let rel = format!("series/date={}/L2-{id:08}.parquet", day_str(d));
        let abs = root.join(&rel);
        std::fs::create_dir_all(abs.parent().unwrap())?;
        let mut mb = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
        let mut ids = Vec::new();
        let mut ms = StringBuilder::new();
        let mut fs = Vec::new();
        for &(id, mi, qi) in &live {
            let q = &qs[qi];
            for (k, v) in labels_of(q.qid, q.db) {
                mb.keys().append_value(k);
                mb.values().append_value(v);
            }
            mb.append(true)?;
            ids.push(id);
            ms.append_value(metrics[mi]);
            fs.push(q.from.max(d0) * 1000);
        }
        let labels = mb.finish();
        let sc = Arc::new(Schema::new(vec![
            Field::new("series_id", DataType::UInt64, false),
            Field::new("metric", DataType::Utf8, false),
            Field::new("labels", labels.data_type().clone(), true),
            Field::new(
                "first_seen",
                DataType::Timestamp(TimeUnit::Microsecond, Some(tz.clone())),
                false,
            ),
        ]));
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(9)?))
            .set_statistics_enabled(EnabledStatistics::Chunk)
            .set_max_row_group_row_count(Some(crate::store::SERIES_ROW_GROUP))
            .build();
        let mut w = ArrowWriter::try_new(std::fs::File::create(&abs)?, sc.clone(), Some(props))?;
        let n = ids.len() as u64;
        w.write(&RecordBatch::try_new(
            sc,
            vec![
                Arc::new(UInt64Array::from(ids)),
                Arc::new(ms.finish()),
                Arc::new(labels),
                Arc::new(TimestampMicrosecondArray::from(fs).with_timezone("UTC")),
            ],
        )?)?;
        w.close()?;
        n_series += n;
        entries.push(crate::catalog::FileEntry {
            kind: "series".into(),
            path: rel,
            day: d,
            level: 2,
            rows: n,
            bytes: std::fs::metadata(&abs)?.len(),
            min_ts: d0,
            max_ts: d0,
            created_ms: now_ms(),
            registered: false,
            ver: 0,
        });
        eprintln!("synth-sampled: day {} series {} samples {}", day_str(d), n, rows);
    }
    let mut c = store.shared.catalog.lock().unwrap();
    c.files.extend(entries);
    c.save(root)?;
    Ok(serde_json::json!({"days": days, "series_rows": n_series, "samples": n_samples, "queryids": qs.len()}))
}
