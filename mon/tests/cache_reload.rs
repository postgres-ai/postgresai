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
    assert!(new.range.unwrap().start <= END - 1_800_000);
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
