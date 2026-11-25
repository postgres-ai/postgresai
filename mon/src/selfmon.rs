//! pgai-mon's own state as series: what GET /health reports (query::health), written as
//! `pgai_mon_*` samples through the ingest path every `collector.self_metrics_interval_s`
//! (0: every 60 s), so the monitoring system's own health has a history. Counters end in
//! `_total`; units are in the name (`_bytes`, `_seconds`). The series carry no instance labels:
//! the config has no identity for a pgai-mon process, and each process writes its own store.
use crate::ingest::{ExtSample, Ingest};
use crate::query::AppState;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// Staleness group of these series: one that a later write lacks (a reader that became
/// in-process, a second bucket gone) gets a staleness marker.
const GROUP: &str = "pgai_mon_self";

/// CPU time (user + system) this process used, in seconds.
pub fn cpu_seconds() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        // /proc/self/stat: fields 14 and 15 (utime, stime) in clock ticks, counted after the
        // command name, which may hold spaces and parentheses (so split after the last ')').
        let s = std::fs::read_to_string("/proc/self/stat").ok()?;
        let rest = &s[s.rfind(')')? + 1..];
        let mut f = rest.split_whitespace().skip(11);
        let ut: u64 = f.next()?.parse().ok()?;
        let st: u64 = f.next()?.parse().ok()?;
        let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        (hz > 0).then(|| (ut + st) as f64 / hz as f64)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
            return None;
        }
        let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
        Some(tv(ru.ru_utime) + tv(ru.ru_stime))
    }
}

/// The samples for one /health body at `t_ms`. A value /health lacks (null: no reader
/// process, no bucket, no commit yet) gives no sample.
pub fn samples(h: &Value, t_ms: i64) -> Vec<ExtSample> {
    let mut out = Vec::new();
    let mut put = |metric: &str, labels: &[(&str, &str)], v: Option<f64>| {
        if let Some(value) = v {
            out.push(ExtSample {
                metric: metric.to_string(),
                labels: labels.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                ts: t_ms,
                value,
            });
        }
    };
    let num = |v: &Value, path: &[&str]| path.iter().try_fold(v, |v, k| v.get(k)).and_then(Value::as_f64);

    // the release this process runs (crate::VERSION), as Prometheus' `*_build_info`: the version
    // is the label, the value is 1
    if let Some(v) = h["version"].as_str() {
        put("pgai_mon_build_info", &[("version", v)], Some(1.0));
    }
    put("pgai_mon_ingest_samples_total", &[], num(h, &["samples_in"]));
    put("pgai_mon_ingest_records_total", &[], num(h, &["records_in"]));
    put("pgai_mon_flushes_total", &[], num(h, &["flushes"]));
    put("pgai_mon_late_flushes_total", &[], num(h, &["late_flushes"]));
    put(
        "pgai_mon_flush_duration_seconds",
        &[],
        num(h, &["flush_ms_last"]).map(|ms| ms / 1000.0),
    );
    put("pgai_mon_compactions_total", &[], num(h, &["compactions"]));
    put("pgai_mon_wal_bytes", &[], num(h, &["wal_bytes"]));
    put("pgai_mon_store_files", &[], num(h, &["files"]));
    put("pgai_mon_store_parquet_bytes", &[], num(h, &["parquet_bytes"]));
    put("pgai_mon_series_known", &[], num(h, &["series_known"]));
    put("pgai_mon_process_resident_memory_bytes", &[], num(h, &["rss_bytes"]));
    put("pgai_mon_process_cpu_seconds_total", &[], num(h, &["cpu_seconds"]));

    if let Some(r) = h.get("reader").filter(|r| r["process"] == true) {
        put("pgai_mon_reader_resident_memory_bytes", &[], num(r, &["rss_bytes"]));
        put("pgai_mon_reader_restarts_total", &[], num(r, &["restarts"]));
        put("pgai_mon_reader_memory_kills_total", &[], num(r, &["memory_kills"]));
    }

    let a = &h["admission"];
    put("pgai_mon_admission_running", &[], num(a, &["running"]));
    put("pgai_mon_admission_waiting", &[], num(a, &["waiting"]));
    put(
        "pgai_mon_admission_started_total",
        &[("class", "short")],
        num(a, &["started_short"]),
    );
    put(
        "pgai_mon_admission_started_total",
        &[("class", "heavy")],
        num(a, &["started_heavy"]),
    );
    put(
        "pgai_mon_admission_memory_deferred_total",
        &[],
        num(a, &["mem_deferred"]),
    );

    // query: the HTTP result cache (L1); footer and row: the reader's (in a reader process
    // they restart from 0 with it, which rate() reads as a counter reset).
    let c = &h["cache"];
    for (name, path) in [
        ("query", &["l1"][..]),
        ("footer", &["reader", "footer_cache"][..]),
        ("row", &["reader", "row_cache"][..]),
    ] {
        let Some(s) = path.iter().try_fold(c, |v, k| v.get(k)) else {
            continue;
        };
        put("pgai_mon_cache_hits_total", &[("cache", name)], num(s, &["hits"]));
        put("pgai_mon_cache_misses_total", &[("cache", name)], num(s, &["misses"]));
    }

    let buckets: Vec<&Value> = match &h["s3_write"] {
        Value::Array(a) => a.iter().collect(),
        Value::Null => vec![],
        o => vec![o],
    };
    for b in &buckets {
        // a store label only when there are several (one process normally writes one store)
        let url = b["url"].as_str().unwrap_or_default();
        let l: &[(&str, &str)] = if buckets.len() > 1 { &[("store", url)] } else { &[] };
        put("pgai_mon_s3_last_commit_age_seconds", l, num(b, &["last_commit_age_s"]));
        put("pgai_mon_s3_pending_files", l, num(b, &["pending_files"]));
        put("pgai_mon_s3_pending_bytes", l, num(b, &["pending_bytes"]));
        put("pgai_mon_s3_commits_total", l, num(b, &["commits"]));
        put("pgai_mon_s3_upload_errors_total", l, num(b, &["upload_errors"]));
    }
    out
}

/// Write the samples of the current /health once, at `t_ms`.
pub async fn write_once(state: &AppState, ing: &Ingest, t_ms: i64) -> anyhow::Result<()> {
    let h = crate::query::health(state).await;
    let s = ing.validate(samples(&h, t_ms)).map_err(|e| anyhow::anyhow!(e))?;
    ing.write(s, Some((GROUP, t_ms))).await?;
    Ok(())
}

/// Write at start, then at every interval boundary.
pub async fn run(state: AppState, ing: Arc<Ingest>, interval_s: u64) {
    let iv = if interval_s == 0 {
        60_000
    } else {
        interval_s as i64 * 1000
    };
    let mut t = crate::model::now_ms();
    loop {
        if let Err(e) = write_once(&state, &ing, t).await {
            eprintln!("selfmon: {e:#}");
        }
        let now = crate::model::now_ms();
        t = now.div_euclid(iv) * iv + iv;
        tokio::time::sleep(Duration::from_millis((t - now).max(0) as u64)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn get<'a>(s: &'a [ExtSample], metric: &str, labels: &[(&str, &str)]) -> Option<&'a ExtSample> {
        s.iter().find(|x| {
            x.metric == metric
                && x.labels.len() == labels.len()
                && labels
                    .iter()
                    .all(|(k, v)| x.labels.iter().any(|(a, b)| a == k && b == v))
        })
    }

    #[test]
    fn health_maps_to_series() {
        let h = json!({
            "version": "4.0.0-dev.1",
            "samples_in": 10, "records_in": 3, "flushes": 2, "flush_ms_last": 250, "late_flushes": 0,
            "compactions": 1, "wal_bytes": 4096, "files": 5, "parquet_bytes": 1000, "series_known": 7,
            "rss_bytes": 123, "cpu_seconds": 1.5,
            "admission": {"running": 1, "waiting": 0, "started_short": 4, "started_heavy": 2, "mem_deferred": 0},
            "cache": {"l1": {"hits": 9, "misses": 1}, "reader": {"footer_cache": {"hits": 3, "misses": 4},
                      "row_cache": {"hits": 0, "misses": 2}}},
            "reader": {"process": true, "rss_bytes": null, "restarts": 1, "memory_kills": 0},
            "s3_write": {"url": "s3://b/p", "last_commit_age_s": null, "pending_files": 2,
                         "pending_bytes": 300, "commits": 0, "upload_errors": 3},
        });
        let s = samples(&h, 1000);
        assert!(s.iter().all(|x| x.ts == 1000));
        let v = |m: &str, l: &[(&str, &str)]| get(&s, m, l).map(|x| x.value);
        assert_eq!(v("pgai_mon_build_info", &[("version", "4.0.0-dev.1")]), Some(1.0));
        assert_eq!(v("pgai_mon_flush_duration_seconds", &[]), Some(0.25));
        assert_eq!(v("pgai_mon_process_cpu_seconds_total", &[]), Some(1.5));
        assert_eq!(v("pgai_mon_admission_started_total", &[("class", "heavy")]), Some(2.0));
        assert_eq!(v("pgai_mon_cache_hits_total", &[("cache", "query")]), Some(9.0));
        assert_eq!(v("pgai_mon_cache_misses_total", &[("cache", "footer")]), Some(4.0));
        assert_eq!(v("pgai_mon_reader_restarts_total", &[]), Some(1.0));
        assert_eq!(v("pgai_mon_s3_upload_errors_total", &[]), Some(3.0));
        // null: no sample
        assert_eq!(v("pgai_mon_reader_resident_memory_bytes", &[]), None);
        assert_eq!(v("pgai_mon_s3_last_commit_age_seconds", &[]), None);
        assert_eq!(s.len(), 30); // 32 at most: two of them null here
    }

    #[test]
    fn absent_parts_give_no_series() {
        let h = json!({"samples_in": 1, "reader": {"process": false}, "s3_write": null,
                       "cache": {"l1": {"hits": 0, "misses": 0}, "reader": null}});
        let s = samples(&h, 0);
        assert!(s
            .iter()
            .all(|x| !x.metric.starts_with("pgai_mon_reader_") && !x.metric.starts_with("pgai_mon_s3_")));
        assert!(get(&s, "pgai_mon_cache_hits_total", &[("cache", "footer")]).is_none());
        assert!(get(&s, "pgai_mon_cache_hits_total", &[("cache", "query")]).is_some());
        assert!(s.iter().all(|x| x.metric != "pgai_mon_build_info"));
    }

    #[test]
    fn several_buckets_are_labelled_by_store() {
        let b = |u: &str| json!({"url": u, "pending_files": 1});
        let s = samples(&json!({"s3_write": [b("s3://a"), b("s3://b")]}), 0);
        assert!(get(&s, "pgai_mon_s3_pending_files", &[("store", "s3://a")]).is_some());
        assert!(get(&s, "pgai_mon_s3_pending_files", &[("store", "s3://b")]).is_some());
    }

    #[test]
    fn cpu_seconds_grows() {
        let a = cpu_seconds().unwrap();
        let t = std::time::Instant::now();
        let mut x = 0u64;
        while t.elapsed() < Duration::from_millis(50) {
            x = x.wrapping_add(std::hint::black_box(1));
        }
        std::hint::black_box(x);
        let b = cpu_seconds().unwrap();
        assert!(b > a, "{a} -> {b}");
    }
}
