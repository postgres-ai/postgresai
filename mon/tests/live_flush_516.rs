//! Reproduction/measurement harness: see evidence/516/live_manifest.py.
use pgai_mon::{
    config::{HttpCfg, StoreCfg},
    model::{Record, Sample},
    query::{AppState, Backend, Reader, EMBEDDED_VIEWS_SQL},
    store::StoreCore,
};
use std::sync::Arc;

#[tokio::test]
#[ignore = "manual D02 live-flush measurement; requires PGAI_516_MANIFEST"]
async fn dashboard_02_across_live_flushes() {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("PGAI_516_MANIFEST").unwrap()).unwrap()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    let data = manifest["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| format!("{r}\n"))
        .collect::<String>();
    pgai_mon::tools::load_vm_export(&mut core, &mut std::io::Cursor::new(data), 1_000_000, true).unwrap();
    let end = manifest["end"].as_i64().unwrap();
    let ids: Vec<_> = manifest["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let m = r["metric"].as_object().unwrap();
            let labels = m
                .iter()
                .filter(|(k, _)| k.as_str() != "__name__")
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                .collect::<Vec<_>>();
            pgai_mon::model::series_id(m["__name__"].as_str().unwrap(), &labels)
        })
        .collect();
    // Reopen the loader's compacted current day before priming the cache.
    core.append(Record::Samples(
        ids.iter()
            .map(|&series_id| Sample {
                series_id,
                ts: end + 1_800_000,
                value: 122.0,
            })
            .collect(),
    ))
    .unwrap();
    core.flush().unwrap();
    let mf = pgai_mon::metricdefs::load(std::path::Path::new("metrics.yml")).unwrap();
    let b = Backend::InProc(Arc::new(
        Reader::new(
            core.shared.clone(),
            HttpCfg::default(),
            format!("{EMBEDDED_VIEWS_SQL}\n{}", include_str!("../sql/macros.sql")),
            pgai_mon::query::build_dict(Some(&mf), &mf.dict_intervals(&["full"], &[])),
        )
        .unwrap(),
    ));
    let mut state = AppState::new(b.clone(), HttpCfg::default(), core.shared.clone());
    state.cache = Arc::new(pgai_mon::qcache::QueryCache::new(16 << 20, 2 << 20, EMBEDDED_VIEWS_SQL));
    let cache = state.cache.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/sql?format=csv", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, pgai_mon::query::router(state)).await.unwrap() });
    let client = reqwest::Client::new();
    for round in 0..6 {
        if round > 0 {
            let ts = if round == 5 {
                end - 60_000
            } else {
                end + 1_800_000 + round * 60_000
            };
            core.append(Record::Samples(
                ids.iter()
                    .map(|&series_id| Sample {
                        series_id,
                        ts,
                        value: 123.0,
                    })
                    .collect(),
            ))
            .unwrap();
            core.flush().unwrap();
            let cat = core.shared.catalog.lock().unwrap();
            let f = cat.files_of("samples").last().unwrap();
            println!(
                "flush {round}: min_ts={} max_ts={} window_end={end}",
                f.min_ts, f.max_ts
            );
        }
        let started = std::time::Instant::now();
        let mut hits = 0;
        let mut rows = 0;
        let mut http_time = std::time::Duration::ZERO;
        for q in manifest["queries"].as_array().unwrap() {
            let sql = q["sql"].as_str().unwrap();
            let request_started = std::time::Instant::now();
            let response = client.post(&url).body(sql.to_string()).send().await.unwrap();
            assert_eq!(response.status(), 200, "panel {}", q["panel"]);
            hits += usize::from(response.headers()["x-pgai-cache"] == "hit");
            let body = response.text().await.unwrap();
            http_time += request_started.elapsed();
            rows += body.lines().count().saturating_sub(1);
            // Expand exactly as the HTTP handler does before a cache-free execution.
            let expanded = pgai_mon::grafana::expand(sql).unwrap();
            assert_eq!(
                body,
                b.call("sql_csv", expanded).await.unwrap().body,
                "panel {}",
                q["panel"]
            );
        }
        if (1..5).contains(&round) && std::env::var_os("PGAI_516_EXPECT_HITS").is_some() {
            assert_eq!(hits, manifest["queries"].as_array().unwrap().len());
        }
        if round == 5 {
            assert_eq!(hits, 0, "late samples must invalidate");
        }
        println!(
            "round {round}: hits={hits}/{} rows={rows} http={http_time:?} elapsed_with_fresh_checks={:?} stats={}",
            manifest["queries"].as_array().unwrap().len(),
            started.elapsed(),
            cache.stats_json()
        );
    }
    server.abort();
}

#[test]
#[ignore = "manual 128-file open-day scan measurement"]
fn open_day_file_cost() {
    use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
    use pgai_mon::eval::{
        footer,
        scan::{Scan, SeriesRows},
    };
    use std::time::Instant;
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    let day = pgai_mon::model::day_of(pgai_mon::model::now_ms()) * pgai_mon::model::DAY_MS;
    let mut ids = (0..42_000)
        .map(|i| {
            pgai_mon::model::series_id(
                "pgwatch_pg_stat_statements_calls_total",
                &[("queryid".into(), i.to_string())],
            )
        })
        .collect::<Vec<_>>();
    ids.sort_unstable();
    for minute in 0..128 {
        core.append(Record::Samples(
            ids.iter()
                .enumerate()
                .map(|(i, &series_id)| Sample {
                    series_id,
                    ts: day + minute * 60_000,
                    value: (i + minute as usize) as f64,
                })
                .collect(),
        ))
        .unwrap();
        core.flush().unwrap();
    }
    for compacted in [false, true] {
        if compacted {
            core.compactor()
                .unwrap()
                .run_once(day + pgai_mon::model::DAY_MS, true)
                .unwrap();
        }
        let files = core
            .shared
            .catalog
            .lock()
            .unwrap()
            .files_of("samples")
            .cloned()
            .collect::<Vec<_>>();
        let mut open_time = std::time::Duration::ZERO;
        let mut footer_time = std::time::Duration::ZERO;
        let mut writers = std::collections::BTreeSet::new();
        for f in &files {
            let t = Instant::now();
            let file = std::fs::File::open(dir.path().join(&f.path)).unwrap();
            open_time += t.elapsed();
            let t = Instant::now();
            let md = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new()).unwrap();
            footer_time += t.elapsed();
            writers.insert(
                md.metadata()
                    .file_metadata()
                    .created_by()
                    .unwrap_or("unknown")
                    .to_string(),
            );
        }
        println!(
            "files={} writers={writers:?} open={open_time:?} uncached_footer={footer_time:?}",
            files.len()
        );
        for run in 0..4 {
            if run == 0 {
                footer::retain(|_| false);
            }
            let t = Instant::now();
            let mut scan = Scan::open(
                dir.path(),
                &files,
                (0..10).map(|i| ids[i * 4200]).collect::<Vec<_>>(),
                day,
                day + 127 * 60_000,
            )
            .unwrap();
            let open = t.elapsed();
            let mut row = SeriesRows::default();
            let mut n = 0;
            while scan.next_series(&mut row).unwrap() {
                n += row.ts.len();
            }
            assert_eq!(n, 1280);
            println!(
                "files={} run={run} scan_open={open:?} total={:?} rows={n}",
                files.len(),
                t.elapsed()
            );
        }
    }
}
