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
