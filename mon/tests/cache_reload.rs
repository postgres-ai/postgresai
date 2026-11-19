//! Real evaluator results across file commits and dictionary-only reloads.
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::dictionary::Dictionary;
use pgai_mon::model::{series_id, Record, Sample, SeriesDef, DAY_MS};
use pgai_mon::qcache::{QueryCache, Served};
use pgai_mon::query::{Backend, DictEntry, Reader, Reply, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::StoreCore;
use std::sync::Arc;

const END: i64 = 1_794_009_600_000 + 3_600_000;

fn append(core: &mut StoreCore, ts: i64, value: f64) {
    core.append(Record::Samples(vec![Sample {
        series_id: series_id("pgwatch_slow_v", &[]),
        ts,
        value,
    }]))
    .unwrap();
    core.flush().unwrap();
}

fn seed(core: &mut StoreCore) {
    core.append(Record::Series(vec![SeriesDef {
        series_id: series_id("pgwatch_slow_v", &[]),
        metric: "pgwatch_slow_v".into(),
        labels: vec![],
        first_seen: END - 1_800_000,
    }]))
    .unwrap();
    append(core, END - 1_800_000, 42.0);
}

fn dict(interval_s: u64) -> Dictionary {
    Dictionary::new(vec![DictEntry {
        group: "slow".into(),
        prefix: "pgwatch_slow".into(),
        interval_s: Some(interval_s),
        instance_level: false,
        gauges: vec![],
        description: String::new(),
    }])
}

fn query(end: i64) -> String {
    format!("SELECT value FROM prom_last('pgwatch_slow_v', [], {end}, {end}, 30000) ORDER BY value")
}

fn cache() -> Arc<QueryCache> {
    Arc::new(QueryCache::new(1 << 20, 1 << 20, EMBEDDED_VIEWS_SQL))
}

async fn cached(c: &Arc<QueryCache>, b: &Backend, core: &StoreCore, sql: &str, legacy: bool) -> (Arc<Reply>, Served) {
    let snap = c.snapshot(&core.shared.catalog.lock().unwrap(), b.dictionary_version().unwrap());
    let (b, sql_owned) = (b.clone(), sql.to_string());
    let (r, how) = c
        .get_or_run(sql, 3, snap, move || {
            let (b, sql) = (b.clone(), sql_owned.clone());
            async move {
                let mut r = b.call("sql_csv", sql).await?;
                if legacy {
                    r.range = None;
                }
                Ok(r)
            }
        })
        .await;
    (r.unwrap(), how)
}

async fn check(c: &Arc<QueryCache>, b: &Backend, core: &StoreCore, sql: &str, want: Served) -> String {
    let (r, how) = cached(c, b, core, sql, false).await;
    assert_eq!(how, want, "{sql}");
    let fresh = b.call("sql_csv", sql.into()).await.unwrap();
    assert_eq!(r.body, fresh.body, "cached bytes must equal a fresh execution");
    r.body.clone()
}

#[tokio::test]
async fn historical_hits_across_flushes_and_invalidates_on_overlap_compaction_retention() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = StoreCfg {
        retention_days: 1,
        ..Default::default()
    };
    let mut core = StoreCore::open(dir.path(), &cfg).unwrap();
    seed(&mut core);
    let b = Backend::InProc(Arc::new(
        Reader::new(
            core.shared.clone(),
            HttpCfg::default(),
            EMBEDDED_VIEWS_SQL.into(),
            dict(3600).entries,
        )
        .unwrap(),
    ));
    let (before, after) = (cache(), cache());
    let sql = query(END);
    assert_eq!(cached(&before, &b, &core, &sql, true).await.1, Served::Miss);
    assert!(check(&after, &b, &core, &sql, Served::Miss).await.contains("42.0"));
    check(&after, &b, &core, &sql, Served::Hit).await;
    let (mut before_hits, mut after_hits) = (0, 0);
    for i in 1..=10 {
        append(&mut core, END + i, 100.0 + i as f64);
        before_hits += usize::from(cached(&before, &b, &core, &sql, true).await.1 == Served::Hit);
        after_hits += usize::from(cached(&after, &b, &core, &sql, false).await.1 == Served::Hit);
        check(&after, &b, &core, &sql, Served::Hit).await;
    }
    assert_eq!((before_hits, after_hits), (0, 10));
    println!("identical historical requests after 10 real flushes: generation-only {before_hits}/10 hits; range-aware {after_hits}/10 hits; fresh CSV bytes equal every time");
    // Inclusive end: a sample exactly at the endpoint changes the answer.
    append(&mut core, END, 99.0);
    assert!(check(&after, &b, &core, &sql, Served::Miss).await.contains("99.0"));
    // A late append inside the lookback (before the requested grid start) also invalidates.
    append(&mut core, END - 60_000, 77.0);
    check(&after, &b, &core, &sql, Served::Miss).await;
    assert!(core.compactor().unwrap().run_once(END + DAY_MS, true).unwrap() > 0);
    check(&after, &b, &core, &sql, Served::Miss).await;
    append(&mut core, END + 3 * DAY_MS, 100.0);
    check(&after, &b, &core, &sql, Served::Hit).await;
    assert!(core.compactor().unwrap().expire(END + 3 * DAY_MS).unwrap() > 0);
    let empty = check(&after, &b, &core, &sql, Served::Miss).await;
    assert_eq!(empty, "value\r\n");
}

#[tokio::test]
async fn dictionary_version_isolates_cached_results_without_a_flush() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    seed(&mut core);
    let b = Backend::InProc(Arc::new(
        Reader::new(
            core.shared.clone(),
            HttpCfg::default(),
            EMBEDDED_VIEWS_SQL.into(),
            dict(30).entries,
        )
        .unwrap(),
    ));
    let c = cache();
    let sql = query(END);
    assert_eq!(check(&c, &b, &core, &sql, Served::Miss).await, "value\r\n");
    check(&c, &b, &core, &sql, Served::Hit).await;
    dict(3600).publish(&core.shared.scratch).unwrap();
    assert!(check(&c, &b, &core, &sql, Served::Miss).await.contains("42.0"));
    check(&c, &b, &core, &sql, Served::Hit).await;
    dict(30).publish(&core.shared.scratch).unwrap();
    assert_eq!(check(&c, &b, &core, &sql, Served::Hit).await, "value\r\n");
}

#[tokio::test]
async fn mixed_sql_and_new_same_day_series_keep_conservative_validity() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    seed(&mut core);
    let views = format!("{EMBEDDED_VIEWS_SQL}\nCREATE MACRO hidden_raw() AS (SELECT count(*) FROM samples);\nCREATE MACRO overloaded() AS 1, (x) AS (SELECT count(*) FROM samples);");
    let b = Backend::InProc(Arc::new(
        Reader::new(core.shared.clone(), HttpCfg::default(), views, dict(3600).entries).unwrap(),
    ));
    let c = cache();
    let sql = query(END);
    let queries = [
        format!("WITH e AS ({sql}) SELECT * FROM e"),
        format!("SELECT *, (SELECT count(*) FROM samples) AS n FROM ({sql})"),
        format!("SELECT *, hidden_raw() AS n FROM ({sql})"),
        "SELECT count(*) FROM samples".into(),
        format!("SELECT *, overloaded(1) AS n FROM ({sql})"),
        // An inner CTE named samples must not hide the outer real table dependency.
        format!("SELECT (WITH samples AS ({sql}) SELECT count(*) FROM samples), (SELECT count(*) FROM samples)"),
    ];
    for sql in &queries {
        check(&c, &b, &core, sql, Served::Miss).await;
    }
    append(&mut core, END + 1, 123.0);
    for (i, sql) in queries.iter().enumerate() {
        check(&c, &b, &core, sql, if i == 0 { Served::Hit } else { Served::Miss }).await;
    }
    let labels = format!(
        "SELECT labels FROM prom_series_rs('last', 'pgwatch_slow_v', '[]', {END}, {END}, 30000, 0) ORDER BY labels"
    );
    let old = check(&c, &b, &core, &labels, Served::Miss).await;
    let tags = vec![("new".into(), "series".into())];
    core.append(Record::Series(vec![SeriesDef {
        series_id: series_id("pgwatch_slow_v", &tags),
        metric: "pgwatch_slow_v".into(),
        labels: tags,
        first_seen: END + 10,
    }]))
    .unwrap();
    core.flush().unwrap();
    let new = check(&c, &b, &core, &labels, Served::Miss).await;
    assert_ne!(old, new);
}

#[test]
fn child_reloads_dictionary_and_returns_fetch_metadata() {
    use std::io::{BufRead, Write};
    use std::process::{Command, Stdio};
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    seed(&mut core);
    let spec = pgai_mon::sqlproc::Spec {
        store: dir.path().into(),
        scratch: core.shared.scratch.clone(),
        http: HttpCfg::default().resolve(),
        views_sql: EMBEDDED_VIEWS_SQL.into(),
        dict: dict(30).entries,
        overlay: None,
    };
    let path = dir.path().join("spec.json");
    std::fs::write(&path, serde_json::to_vec(&spec).unwrap()).unwrap();
    struct Worker(std::process::Child);
    impl Drop for Worker {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let mut child = Worker(
        Command::new(env!("CARGO_BIN_EXE_pgai-mon"))
            .arg("sql-worker")
            .arg("--spec")
            .arg(&path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let mut input = child.0.stdin.take().unwrap();
    let mut output = std::io::BufReader::new(child.0.stdout.take().unwrap());
    let mut run = |id| {
        writeln!(
            input,
            "{}",
            serde_json::json!({"id": id, "op": "sql_csv", "sql": query(END)})
        )
        .unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        output.read_line(&mut line).unwrap();
        pgai_mon::sqlproc::parse_reply_s3(line.trim_end().into())
            .unwrap()
            .2
            .unwrap()
    };
    let old = run(1);
    assert_eq!(old.body, "value\r\n");
    let updated = dict(3600);
    updated.publish(&core.shared.scratch).unwrap();
    let new = run(2);
    assert!(new.body.contains("42.0"));
    assert_eq!(new.dict_version, updated.version);
    assert_ne!(old.dict_version, new.dict_version);
    assert_eq!(old.gen, new.gen);
    let range = new.range.unwrap();
    assert!(range.start <= END - 1_800_000);
    assert_eq!(
        range.prefixes,
        Some(vec![(series_id("pgwatch_slow_v", &[]) >> 48) as u16])
    );
    drop(input);
    assert!(child.0.wait().unwrap().success());
}

#[tokio::test]
async fn http_cache_checks_dictionary_before_serving_and_keeps_historical_flush_hits() {
    use pgai_mon::query::AppState;
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    seed(&mut core);
    let b = Backend::InProc(Arc::new(
        Reader::new(
            core.shared.clone(),
            HttpCfg::default(),
            EMBEDDED_VIEWS_SQL.into(),
            dict(30).entries,
        )
        .unwrap(),
    ));
    let mut state = AppState::new(b.clone(), HttpCfg::default(), core.shared.clone());
    state.cache = cache();
    let app = pgai_mon::query::router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/sql?format=csv", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let sql = query(END);
    for (step, want) in ["miss", "hit", "miss", "hit"].into_iter().enumerate() {
        if step == 2 {
            dict(3600).publish(&core.shared.scratch).unwrap();
        }
        if step == 3 {
            append(&mut core, END + 1, 100.0);
        }
        let response = client.post(&url).body(sql.clone()).send().await.unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.headers()["x-pgai-cache"], want);
        let body = response.text().await.unwrap();
        assert_eq!(body, b.call("sql_csv", sql.clone()).await.unwrap().body);
        assert_eq!(body.contains("42.0"), step >= 2);
    }
    let health: serde_json::Value = serde_json::from_str(
        &client
            .get(url.replace("/sql?format=csv", "/health"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(health["cache"]["l1"]["invalidations"]["rollup_eligibility"], 0);
    server.abort();
}

#[tokio::test]
async fn multiple_evaluator_windows_are_unioned() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    seed(&mut core);
    let b = Backend::InProc(Arc::new(
        Reader::new(
            core.shared.clone(),
            HttpCfg::default(),
            EMBEDDED_VIEWS_SQL.into(),
            dict(3600).entries,
        )
        .unwrap(),
    ));
    let c = cache();
    let sql = format!(
        "SELECT * FROM ({}) UNION ALL SELECT * FROM ({}) ORDER BY value",
        query(END),
        query(END + 600_000)
    );
    let old = check(&c, &b, &core, &sql, Served::Miss).await;
    let range = b.call("sql_csv", sql.clone()).await.unwrap().range.unwrap();
    assert_eq!(range.end, END + 600_000);
    assert!(range.start < END - 1_800_000);
    append(&mut core, END + 300_000, 99.0);
    let new = check(&c, &b, &core, &sql, Served::Miss).await;
    assert_ne!(old, new);
}

/// Grafana's moving raw times identify the executed window, including any expanded fetch end.
#[tokio::test]
async fn http_grafana_windows_share_only_exact_results() {
    use pgai_mon::query::AppState;
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    seed(&mut core);
    let b = Backend::InProc(Arc::new(
        Reader::new(
            core.shared.clone(),
            HttpCfg::default(),
            EMBEDDED_VIEWS_SQL.into(),
            dict(3600).entries,
        )
        .unwrap(),
    ));
    let mut state = AppState::new(b.clone(), HttpCfg::default(), core.shared.clone());
    state.cache = cache();
    let app = pgai_mon::query::router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/sql?format=csv", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let send = |sql: String, fresh: String, want: &'static str| {
        let (client, url, b) = (client.clone(), url.clone(), b.clone());
        async move {
            let response = client.post(url).body(sql.clone()).send().await.unwrap();
            let status = response.status();
            if status != 200 {
                panic!("{sql}: {status}: {}", response.text().await.unwrap());
            }
            assert_eq!(response.headers()["x-pgai-cache"], want);
            let body = response.text().await.unwrap();
            assert_eq!(body, b.call("sql_csv", fresh).await.unwrap().body);
            body
        }
    };
    let window =
        |raw: i64, mode: &str, body: &str| format!("-- pgai-window-v1 {} {raw} 30000 {mode}\n{body}", raw - 60_000);
    let body = "SELECT t, value FROM prom_last('pgwatch_slow_v', [], $__pgai_start_ms, $__pgai_end_ms, $__pgai_step_ms) ORDER BY t";
    let literal = |end: i64, instant: bool| {
        format!(
            "SELECT t, value FROM prom_last('pgwatch_slow_v', [], {}, {end}, {}) ORDER BY t",
            if instant { end } else { end - 60_000 },
            if instant { 300_000 } else { 30_000 }
        )
    };
    let first = send(window(END + 1, "range", body), literal(END, false), "miss").await;
    let hit = send(window(END + 2, "range", body), literal(END, false), "hit").await;
    assert_eq!(first, hit);
    append(&mut core, END + 3, 99.0);
    send(window(END + 4, "range", body), literal(END, false), "hit").await;
    let next = send(
        window(END + 30_001, "range", body),
        literal(END + 30_000, false),
        "miss",
    )
    .await;
    assert_ne!(first, next);
    send(window(END + 5, "instant", body), literal(END, true), "miss").await;
    send(window(END + 6, "instant", body), literal(END, true), "hit").await;

    // A subquery can fetch the last bucket after the aligned end. Such a flush must miss.
    let extended = "SELECT t, value FROM prom_last('pgwatch_slow_v', [], $__pgai_start_ms, $__pgai_end_ms + $__pgai_step_ms, $__pgai_step_ms) ORDER BY t";
    let fresh = format!(
        "SELECT t, value FROM prom_last('pgwatch_slow_v', [], {}, {}, 30000) ORDER BY t",
        END - 60_000,
        END + 30_000
    );
    let old = send(window(END + 7, "range", extended), fresh.clone(), "miss").await;
    append(&mut core, END + 10_000, 123.0);
    let new = send(window(END + 8, "range", extended), fresh.clone(), "miss").await;
    assert_ne!(old, new);
    append(&mut core, END + 30_001, 321.0);
    send(window(END + 9, "range", extended), fresh, "hit").await;

    // Raw tables and independent raw end labels must remain distinct, even within one step.
    for raw in [END + 11, END + 12] {
        let table = format!("SELECT {raw} AS time, count(*) AS n FROM samples WHERE epoch_ms(ts) <= {raw}");
        send(table.clone(), table, "miss").await;
        let body = format!("SELECT $__pgai_end_ms AS aligned, {raw} AS raw");
        send(
            window(raw, "instant", &body),
            format!("SELECT {END}::BIGINT AS aligned, {raw} AS raw"),
            "miss",
        )
        .await;
    }
    // The same grid with a changed $__range still has a different result and key.
    for (span, rounded) in [(60_499, 60_000), (60_500, 61_000)] {
        let sql = format!(
            "-- pgai-window-v1 {} {} 30000 range\nSELECT $__pgai_range_ms AS duration",
            END - span,
            END
        );
        send(sql, format!("SELECT {rounded}::BIGINT AS duration"), "miss").await;
    }
    let response = client
        .post(url)
        .body("-- pgai-window-v1 0 1 0 range\nSELECT 1")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    server.abort();
}

/// Repeat the same aligned requests with real self-monitoring flushes inside their window.
#[tokio::test]
async fn unrelated_self_scrape_flushes_keep_identical_results() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    seed(&mut core);
    let b = Backend::InProc(Arc::new(
        Reader::new(
            core.shared.clone(),
            HttpCfg::default(),
            EMBEDDED_VIEWS_SQL.into(),
            dict(3600).entries,
        )
        .unwrap(),
    ));
    let c = cache();
    let queries: Vec<_> = (0..3).map(|i| query(END - i * 30_000)).collect();
    let mut bodies = Vec::new();
    for sql in &queries {
        bodies.push(check(&c, &b, &core, sql, Served::Miss).await);
    }
    let mut hits = 0;
    for i in 0..10 {
        let ts = END - 600_000 + i * 30_000;
        let samples = pgai_mon::selfmon::samples(
            &serde_json::json!({
                "admission": {"running": 0, "waiting": 0, "started_short": 0, "started_heavy": i},
            }),
            ts,
        );
        assert!(!samples.is_empty());
        core.append(Record::Series(
            samples
                .iter()
                .map(|s| SeriesDef {
                    series_id: series_id(&s.metric, &s.labels),
                    metric: s.metric.clone(),
                    labels: s.labels.clone(),
                    first_seen: ts,
                })
                .collect(),
        ))
        .unwrap();
        core.append(Record::Samples(
            samples
                .iter()
                .map(|s| Sample {
                    series_id: series_id(&s.metric, &s.labels),
                    ts: s.ts,
                    value: s.value,
                })
                .collect(),
        ))
        .unwrap();
        core.flush().unwrap();
        for (sql, body) in queries.iter().zip(&bodies) {
            let (r, how) = cached(&c, &b, &core, sql, false).await;
            hits += usize::from(how == Served::Hit);
            assert_eq!(&r.body, body);
            assert_eq!(r.body, b.call("sql_csv", sql.clone()).await.unwrap().body);
        }
    }
    println!("identical aligned requests with 10 in-window self-scrape flushes: {hits}/30 hits; cached and fresh CSV bytes equal");
    assert_eq!(hits, 30);
    append(&mut core, END, 99.0);
    assert!(check(&c, &b, &core, &queries[0], Served::Miss).await.contains("99.0"));
}

#[tokio::test]
async fn sample_additions_on_rollup_days_invalidate_cached_results() {
    let hour = 3_600_000;
    let day = END - hour;
    let end = day + 3 * hour;
    let m = series_id("m", &[]);
    let unrelated = series_id("unrelated", &[]);
    assert_ne!(m >> 48, unrelated >> 48);
    // Cover both keep exemptions: disjoint prefixes and samples after the fetch end.
    for (id, ts) in [(unrelated, day + 2 * hour + 3000), (m, end + 1000)] {
        let dir = tempfile::tempdir().unwrap();
        let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
        core.append(Record::Series(vec![SeriesDef {
            series_id: m,
            metric: "m".into(),
            labels: vec![],
            first_seen: day + hour + 1000,
        }]))
        .unwrap();
        core.append(Record::Samples(
            [(hour + 1000, 1e16), (2 * hour + 1000, -1e16), (2 * hour + 2000, 1.0)]
                .into_iter()
                .map(|(offset, value)| Sample {
                    series_id: m,
                    ts: day + offset,
                    value,
                })
                .collect(),
        ))
        .unwrap();
        core.flush().unwrap();
        core.compactor().unwrap().run_once(day + 2 * DAY_MS, true).unwrap();
        {
            let cat = core.shared.catalog.lock().unwrap();
            let samples: Vec<_> = cat.files_of("samples").collect();
            assert_eq!(samples.len(), 1);
            assert_eq!(samples[0].level, 2);
            assert_eq!(cat.files_of(pgai_mon::eval::rollup::KIND).count(), 1);
        }
        let b = Backend::InProc(Arc::new(
            Reader::new(
                core.shared.clone(),
                HttpCfg::default(),
                EMBEDDED_VIEWS_SQL.into(),
                vec![],
            )
            .unwrap(),
        ));
        let c = cache();
        let sql = format!(
            "SELECT value FROM prom_sum_over_time('m', [], {end}, {end}, {hour}, {})",
            2 * hour
        );
        let predicate = format!("{sql} WHERE value > 0");
        assert_eq!(check(&c, &b, &core, &sql, Served::Miss).await, "value\r\n0.0\r\n");
        assert_eq!(check(&c, &b, &core, &predicate, Served::Miss).await, "value\r\n");
        // A sample addition on a later day does not affect this day's rollup eligibility.
        core.append(Record::Samples(vec![Sample {
            series_id: m,
            ts: day + DAY_MS + hour,
            value: 42.0,
        }]))
        .unwrap();
        core.flush().unwrap();
        for q in [&sql, &predicate] {
            check(&c, &b, &core, q, Served::Hit).await;
        }
        core.append(Record::Samples(vec![Sample {
            series_id: id,
            ts,
            value: 42.0,
        }]))
        .unwrap();
        core.flush().unwrap();
        for q in [&sql, &predicate] {
            assert_eq!(check(&c, &b, &core, q, Served::Miss).await, "value\r\n1.0\r\n");
            check(&c, &b, &core, q, Served::Hit).await;
        }
    }
}

#[tokio::test]
async fn dependencies_include_every_metric_even_when_no_series_match() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    seed(&mut core);
    let b = Backend::InProc(Arc::new(
        Reader::new(
            core.shared.clone(),
            HttpCfg::default(),
            EMBEDDED_VIEWS_SQL.into(),
            dict(3600).entries,
        )
        .unwrap(),
    ));
    let c = cache();
    let missing = "pgwatch_other_v";
    let sql = format!(
        "{} UNION ALL SELECT value FROM prom_last('{missing}', [], {END}, {END}, 30000)",
        query(END).replace(" ORDER BY value", "")
    );
    let old = check(&c, &b, &core, &sql, Served::Miss).await;
    let range = b.call("sql_csv", sql.clone()).await.unwrap().range.unwrap();
    let prefixes = range.prefixes.unwrap();
    for metric in ["pgwatch_slow_v", missing] {
        assert!(prefixes.contains(&((series_id(metric, &[]) >> 48) as u16)));
    }
    core.append(Record::Series(vec![SeriesDef {
        series_id: series_id(missing, &[]),
        metric: missing.into(),
        labels: vec![],
        first_seen: END,
    }]))
    .unwrap();
    core.append(Record::Samples(vec![Sample {
        series_id: series_id(missing, &[]),
        ts: END,
        value: 123.0,
    }]))
    .unwrap();
    core.flush().unwrap();
    let new = check(&c, &b, &core, &sql, Served::Miss).await;
    assert_ne!(old, new);
    assert!(new.contains("123.0"));
}

/// A loaded or reopened day can retain rollup files after it has switched back to raw.
#[tokio::test]
async fn newer_flushes_on_an_already_raw_rollup_day_keep_past_results() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    seed(&mut core);
    core.compactor().unwrap().run_once(END + DAY_MS, true).unwrap();
    append(&mut core, END + 1_800_000, 100.0);
    let b = Backend::InProc(Arc::new(
        Reader::new(
            core.shared.clone(),
            HttpCfg::default(),
            EMBEDDED_VIEWS_SQL.into(),
            dict(3600).entries,
        )
        .unwrap(),
    ));
    let c = cache();
    let sql = query(END);
    check(&c, &b, &core, &sql, Served::Miss).await;
    for minute in 1..=3 {
        append(&mut core, END + 1_800_000 + minute * 60_000, 101.0);
        let f = core
            .shared
            .catalog
            .lock()
            .unwrap()
            .files_of("samples")
            .last()
            .unwrap()
            .clone();
        assert_eq!(f.min_ts, END + 1_800_000 + minute * 60_000);
        assert_eq!(f.min_ts, f.max_ts);
        let (r, how) = cached(&c, &b, &core, &sql, false).await;
        assert_eq!(how, Served::Hit, "{}", c.stats_json());
        assert_eq!(r.body, b.call("sql_csv", sql.clone()).await.unwrap().body);
    }
    assert_eq!(c.stats_json()["invalidations"]["rollup_eligibility"], 0);
    append(&mut core, END, 99.0);
    assert!(check(&c, &b, &core, &sql, Served::Miss).await.contains("99.0"));
    assert_eq!(c.stats_json()["invalidations"]["added_samples"], 1);
}

#[tokio::test]
async fn dashboard_text_join_survives_samples_but_not_text_changes() {
    use pgai_mon::model::QueryText;
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    seed(&mut core);
    let b = Backend::InProc(Arc::new(
        Reader::new(
            core.shared.clone(),
            HttpCfg::default(),
            format!("{EMBEDDED_VIEWS_SQL}\nCREATE MACRO text_count() AS (SELECT count(*) FROM query_texts);"),
            dict(3600).entries,
        )
        .unwrap(),
    ));
    let c = cache();
    let queries = [
        format!(
            "SELECT value, (SELECT count(*) FROM query_texts) n FROM ({})",
            query(END)
        ),
        format!("SELECT value, text_count() n FROM ({})", query(END)),
    ];
    for q in &queries {
        check(&c, &b, &core, q, Served::Miss).await;
    }
    for minute in 1..=3 {
        append(&mut core, END + 1_800_000 + minute * 60_000, 100.0);
        for q in &queries {
            check(&c, &b, &core, q, Served::Hit).await;
        }
    }
    core.append(Record::QueryTexts(vec![QueryText {
        queryid: 42,
        dbname: "db".into(),
        query_hash: 42,
        query: Some("SELECT 42".into()),
        first_seen: END + 3_600_000,
        last_seen: END + 3_600_000,
        src: Default::default(),
    }]))
    .unwrap();
    core.flush().unwrap();
    for q in &queries {
        assert!(check(&c, &b, &core, q, Served::Miss).await.contains("42.0,1"));
    }
    assert_eq!(c.stats_json()["invalidations"]["added_query_texts"], 2);
    {
        let mut cat = core.shared.catalog.lock().unwrap();
        cat.files.retain(|f| f.kind != "query_texts");
        cat.generation += 1;
        cat.save(dir.path()).unwrap();
        core.shared
            .generation
            .store(cat.generation, std::sync::atomic::Ordering::Release);
    }
    for q in &queries {
        assert!(check(&c, &b, &core, q, Served::Miss).await.contains("42.0,0"));
    }
    assert_eq!(c.stats_json()["invalidations"]["removed_files"], 2);
}

#[tokio::test]
async fn overridden_text_views_keep_generation_only_validity() {
    for custom in [
        "CREATE OR REPLACE VIEW query_texts AS SELECT count(*) n FROM samples;",
        "CREATE TEMP VIEW query_texts AS SELECT count(*) n FROM samples;",
        "CREATE TEMP TABLE Query_Texts AS SELECT count(*) n FROM samples;",
        "CREATE OR REPLACE VIEW _raw_query_texts AS SELECT 1 n; CREATE OR REPLACE VIEW query_texts AS SELECT * FROM _raw_query_texts;",
        "CREATE OR REPLACE VIEW _raw_query_texts AS SELECT 1::BIGINT queryid, 'db' dbname, 'sql' query, min(ts) first_seen, max(ts) last_seen, 1::UBIGINT query_hash, '' AS source, '' AS cluster, '' AS node_name FROM samples;",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
        seed(&mut core);
        let b = Backend::InProc(Arc::new(Reader::new(core.shared.clone(), HttpCfg::default(), format!("{EMBEDDED_VIEWS_SQL}\n{custom}"), dict(3600).entries).unwrap()));
        let c = cache();
        let sql = format!("SELECT value, (SELECT count(*) FROM query_texts) n FROM ({})", query(END));
        check(&c, &b, &core, &sql, Served::Miss).await;
        append(&mut core, END + 1, 100.0);
        check(&c, &b, &core, &sql, Served::Miss).await;
        assert_eq!(c.stats_json()["invalidations"]["unbounded"], 1);
    }
}

#[tokio::test]
async fn text_view_function_shadowing_is_not_bounded() {
    use pgai_mon::model::QueryText;
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    seed(&mut core);
    core.append(Record::QueryTexts(vec![QueryText {
        queryid: 42,
        dbname: "db".into(),
        query_hash: 42,
        query: Some("SELECT 42".into()),
        first_seen: END,
        last_seen: END,
        src: Default::default(),
    }]))
    .unwrap();
    core.flush().unwrap();
    let views = format!("CREATE MACRO max(x) AS (SELECT count(*) FROM _raw_samples);\n{EMBEDDED_VIEWS_SQL}");
    let b = Backend::InProc(Arc::new(
        Reader::new(core.shared.clone(), HttpCfg::default(), views, dict(3600).entries).unwrap(),
    ));
    let c = cache();
    let sql = format!("SELECT value, (SELECT query FROM query_texts) n FROM ({})", query(END));
    let old = check(&c, &b, &core, &sql, Served::Miss).await;
    append(&mut core, END + 1, 100.0);
    let new = check(&c, &b, &core, &sql, Served::Miss).await;
    assert_ne!(old, new);
    assert_eq!(c.stats_json()["invalidations"]["unbounded"], 1);
}
