//! The binary writes its own state (GET /health) as pgai_mon_* series: they exist, counters
//! do not go down, and the process's RSS and CPU time are non-zero.
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::time::{Duration, Instant};

struct Kill(std::process::Child);
impl Drop for Kill {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn sql(port: u16, q: &str) -> Option<Value> {
    let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    c.set_read_timeout(Some(Duration::from_secs(10))).ok()?;
    write!(
        c,
        "POST /sql HTTP/1.1\r\nHost: x\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{q}",
        q.len()
    )
    .ok()?;
    let mut raw = String::new();
    c.read_to_string(&mut raw).ok()?;
    let (head, body) = raw.split_once("\r\n\r\n")?;
    if !head.starts_with("HTTP/1.1 200") || head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        return None;
    }
    serde_json::from_str(body).ok()
}

#[test]
fn binary_writes_its_own_series() {
    let d = tempfile::tempdir().unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let cfg = d.path().join("pgai-mon.yml");
    std::fs::write(
        &cfg,
        format!(
            "store_dir: {}\nstore: {{flush_interval_s: 1}}\ncollector: {{self_metrics_interval_s: 1}}\n\
             http: {{listen: \"127.0.0.1:{port}\", auth: {{disabled: true}}}}\n",
            d.path().join("s").display()
        ),
    )
    .unwrap();
    let _child = Kill(
        std::process::Command::new(env!("CARGO_BIN_EXE_pgai-mon"))
            .args(["run", "--config", cfg.to_str().unwrap()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    // metric, labels -> values in time order
    let q = "SELECT r.metric, array_to_string(list_transform(list_sort(map_entries(r.labels)), e -> e.key || '=' || e.value), ','), \
             list(s.value ORDER BY s.ts) FROM samples s JOIN series r USING (series_id) \
             WHERE r.metric LIKE 'pgai_mon_%' AND NOT s.stale GROUP BY ALL ORDER BY 1, 2";
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut got: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    while Instant::now() < deadline {
        if let Some(r) = sql(port, q) {
            got = r["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| {
                    let vals = row[2].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
                    (
                        format!("{}{{{}}}", row[0].as_str().unwrap(), row[1].as_str().unwrap()),
                        vals,
                    )
                })
                .collect();
            // three samples, the last one taken after a flush was counted (the first ones can
            // be taken before that)
            if got
                .get("pgai_mon_flushes_total{}")
                .is_some_and(|v| v.len() >= 3 && v[v.len() - 1] >= 1.0)
            {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let v = |k: &str| got.get(k).unwrap_or_else(|| panic!("{k} missing: {:?}", got.keys()));
    for k in [
        "pgai_mon_ingest_samples_total{}",
        "pgai_mon_ingest_records_total{}",
        "pgai_mon_flushes_total{}",
        "pgai_mon_late_flushes_total{}",
        "pgai_mon_compactions_total{}",
        "pgai_mon_process_cpu_seconds_total{}",
        "pgai_mon_admission_started_total{class=short}",
        "pgai_mon_cache_hits_total{cache=query}",
        "pgai_mon_cache_misses_total{cache=query}",
    ] {
        let s = v(k);
        assert!(s.windows(2).all(|w| w[1] >= w[0]), "{k} not monotonic: {s:?}");
    }
    assert!(v("pgai_mon_flushes_total{}").len() >= 3, "{got:?}");
    assert!(*v("pgai_mon_flushes_total{}").last().unwrap() >= 1.0);
    // the series' own samples are ingested: later writes count the earlier ones
    assert!(*v("pgai_mon_ingest_samples_total{}").last().unwrap() > 0.0);
    assert!(v("pgai_mon_process_resident_memory_bytes{}").iter().all(|&x| x > 1e6));
    assert!(*v("pgai_mon_process_cpu_seconds_total{}").last().unwrap() > 0.0);
    // 0 at the first write (at start), then at least these series
    assert!(*v("pgai_mon_series_known{}").last().unwrap() >= 20.0);
    for k in [
        "pgai_mon_wal_bytes{}",
        "pgai_mon_store_files{}",
        "pgai_mon_store_parquet_bytes{}",
        "pgai_mon_flush_duration_seconds{}",
        "pgai_mon_admission_waiting{}",
    ] {
        assert!(v(k).iter().all(|&x| x >= 0.0 && x.is_finite()), "{k}");
    }
    // a local store: no S3 series
    assert!(!got.keys().any(|k| k.starts_with("pgai_mon_s3_")), "{got:?}");
}
