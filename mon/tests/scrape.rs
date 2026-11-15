//! The scraper against local fake endpoints, driven by the scrape files the CLI writes
//! (golden copies in tests/fixtures/scrape/, from cli/test/fixtures/) and the recorded Supabase
//! and ClickHouse expositions.
mod ingest_common;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use pgai_mon::auth::Auth;
use pgai_mon::ingest::{Ingest, IngestCfg};
use pgai_mon::scrape::{load_targets, scrape_once, ScrapeCfg, Scraper};
use std::sync::{Arc, Mutex};

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!("{}/tests/fixtures/scrape/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

/// A fake endpoint: path -> (body, required Authorization header). The body can be swapped.
struct Fake {
    addr: String,
    bodies: Arc<Mutex<Vec<(String, String, Option<String>)>>>,
}

async fn fake(routes: Vec<(&str, String, Option<String>)>) -> Fake {
    let bodies = Arc::new(Mutex::new(
        routes
            .into_iter()
            .map(|(p, b, a)| (p.to_string(), b, a))
            .collect::<Vec<_>>(),
    ));
    let b2 = bodies.clone();
    let app = axum::Router::new().fallback(move |req: Request| {
        let b = b2.clone();
        async move {
            let path = req.uri().path().to_string();
            let authz = req
                .headers()
                .get("authorization")
                .map(|v| v.to_str().unwrap().to_string());
            let r = b.lock().unwrap().iter().find(|(p, _, _)| *p == path).cloned();
            match r {
                None => (StatusCode::NOT_FOUND, "not found".to_string()).into_response(),
                Some((_, _, Some(want))) if authz.as_deref() != Some(want.as_str()) => {
                    (StatusCode::UNAUTHORIZED, "no".to_string()).into_response()
                }
                Some((_, body, _)) => ([("content-type", "text/plain; version=0.0.4")], body).into_response(),
            }
        }
    });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap().to_string();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    Fake { addr, bodies }
}

async fn setup(dir: &std::path::Path) -> (ingest_common::Server, Arc<Ingest>) {
    let slot = Arc::new(Mutex::new(None));
    let s2 = slot.clone();
    let s = ingest_common::serve(dir, Auth::disabled(), move |tx, shared| {
        let ing = Ingest::new(shared, tx, IngestCfg::default());
        *s2.lock().unwrap() = Some(ing.clone());
        axum::Router::new()
    })
    .await;
    let ing = slot.lock().unwrap().take().unwrap();
    (s, ing)
}

/// (metric, labels k=v sorted, value) of the non-stale samples at ts.
fn rows_at(s: &ingest_common::Server, ts: i64) -> Vec<(String, String, f64)> {
    s.sql(&format!(
        "SELECT r.metric, array_to_string(list_transform(list_sort(map_entries(r.labels)), e -> e.key || '=' || e.value), ','),
                CAST(s.value AS VARCHAR)
         FROM samples s JOIN series r USING (series_id) WHERE NOT s.stale AND epoch_ms(s.ts) = {ts} ORDER BY 1, 2"
    ))
    .into_iter()
    .map(|r| (r[0].as_str().unwrap().into(), r[1].as_str().unwrap().into(), r[2].as_str().unwrap().parse().unwrap()))
    .collect()
}

fn stale_at(s: &ingest_common::Server, ts: i64) -> Vec<String> {
    s.sql(&format!(
        "SELECT r.metric FROM samples s JOIN series r USING (series_id) WHERE s.stale AND epoch_ms(s.ts) = {ts} ORDER BY 1"
    ))
    .into_iter()
    .map(|r| r[0].as_str().unwrap().into())
    .collect()
}

fn files_cfg(dir: &std::path::Path) -> ScrapeCfg {
    ScrapeCfg {
        files: vec![format!("{}/*.yml", dir.display())],
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn supabase_relay_with_the_cli_scrape_file() {
    let exposition = fixture("supabase_metrics.prom")
        + "go_goroutines 7\n"
        + "node_time_seconds{supabase_project_ref=\"abcdefghijklmnopqrst\",supabase_identifier=\"abcdefghijklmnopqrst\",service_type=\"db\"} 1.7907312e+09\n"
        + "node_time_seconds{supabase_project_ref=\"zzzzzzzzzzzzzzzzzzzz\",supabase_identifier=\"zzzzzzzzzzzzzzzzzzzz\",service_type=\"db\"} 1.7907312e+09\n"
        + "node_load1{supabase_project_ref=\"zzzzzzzzzzzzzzzzzzzz\",supabase_identifier=\"zzzzzzzzzzzzzzzzzzzz\",service_type=\"db\"} 9\n";
    let f = fake(vec![("/supabase/metrics", exposition.clone(), None)]).await;
    let d = tempfile::tempdir().unwrap();
    let conf = tempfile::tempdir().unwrap();
    std::fs::write(
        conf.path().join("supabase-sb.yml"),
        fixture("supabase-scrape.golden.yml").replace("instance-jobs:9188", &f.addr),
    )
    .unwrap();
    let (targets, errors) = load_targets(&files_cfg(conf.path()));
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(targets.len(), 1);
    let t = &targets[0];
    assert_eq!(t.url, format!("http://{}/supabase/metrics", f.addr));
    assert_eq!(t.interval.as_secs(), 60);
    assert_eq!(t.timeout.as_secs(), 20);
    assert_eq!(t.all_labels["__pgai_rev"], "r55af63516e7615b5");

    let (s, ing) = setup(d.path()).await;
    let ts = pgai_mon::model::now_ms() - 1000;
    let st = scrape_once(&reqwest::Client::new(), &ing, t, 1 << 20, ts).await;
    assert_eq!(st.health, "up", "{st:?}");
    s.flush().await;
    let rows = rows_at(&s, ts);
    let base = format!(
        "cluster=default,instance={},job=supabase-host-metrics,node_name=sb",
        f.addr
    );
    // every node_* family of the target's project, with the target's labels; no go_*, no other project
    let load: Vec<_> = rows.iter().filter(|r| r.0 == "node_load1").collect();
    assert_eq!(load.len(), 1);
    assert_eq!(
        load[0].1,
        format!(
            "{},service_type=db,supabase_identifier=abcdefghijklmnopqrst,supabase_project_ref=abcdefghijklmnopqrst",
            base
        )
    );
    assert_eq!(rows.iter().filter(|r| r.0 == "node_time_seconds").count(), 1);
    assert!(!rows.iter().any(|r| r.0.starts_with("go_")));
    assert!(!rows.iter().any(|r| r.1.contains("zzzz")));
    assert!(!rows
        .iter()
        .any(|r| r.1.contains("__pgai_rev") || r.1.contains("device_error")));
    // the /data filesystem keeps its labels; an empty label value is no label
    assert!(rows
        .iter()
        .any(|r| r.0 == "node_filesystem_avail_bytes" && r.1.contains("mountpoint=/data") && r.2 == 1.808470016e9));
    let kept: usize = rows.iter().filter(|r| r.0.starts_with("node_")).count();
    let synth = |m: &str| rows.iter().find(|r| r.0 == m).map(|r| (r.1.clone(), r.2)).unwrap();
    assert_eq!(synth("up"), (base.clone(), 1.0));
    let n_lines = exposition
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .count();
    assert_eq!(synth("scrape_samples_scraped").1, n_lines as f64);
    assert_eq!(synth("scrape_samples_post_metric_relabeling").1, kept as f64);
    assert!(synth("scrape_duration_seconds").1 >= 0.0);
    assert_eq!(st.samples, kept);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clickhouse_cloud_with_basic_auth_from_the_password_file() {
    let path = "/v1/organizations/ca04a310-730d-4ce0-93dd-39f2cd2d5e6f/postgres/0c330583-6396-86d0-82cd-ed0f23b0d38c/prometheus";
    let secret = "fixture-only-secret";
    use base64::Engine;
    let want = format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode(format!("test-key-id:{secret}"))
    );
    let f = fake(vec![(path, fixture("clickhouse-prometheus.txt"), Some(want))]).await;
    let conf = tempfile::tempdir().unwrap();
    let pw = conf.path().join("clickhouse-my-postgres.secret");
    std::fs::write(&pw, format!("{secret}\n")).unwrap();
    std::fs::write(
        conf.path().join("clickhouse-my-postgres.yml"),
        fixture("clickhouse-scrape.golden.yml")
            .replace("scheme: https", "scheme: http")
            .replace("api.clickhouse.cloud", &f.addr)
            .replace(
                "/etc/pgai/host-metrics/clickhouse-my-postgres.secret",
                pw.to_str().unwrap(),
            ),
    )
    .unwrap();
    let (targets, errors) = load_targets(&files_cfg(conf.path()));
    assert!(errors.is_empty(), "{errors:?}");
    let t = &targets[0];
    assert_eq!(t.timeout.as_secs(), 30);
    let d = tempfile::tempdir().unwrap();
    let (s, ing) = setup(d.path()).await;
    let c = reqwest::Client::new();
    let ts = pgai_mon::model::now_ms() - 10_000;
    let st = scrape_once(&c, &ing, t, 1 << 20, ts).await;
    assert_eq!(st.health, "up", "{st:?}");
    // the secret is re-read at every scrape: a rotated key that the endpoint refuses is up = 0
    std::fs::write(&pw, "rotated-wrong").unwrap();
    let st2 = scrape_once(&c, &ing, t, 1 << 20, ts + 5000).await;
    assert_eq!(st2.health, "down");
    assert!(st2.last_error.as_deref().unwrap().contains("401"), "{st2:?}");
    s.flush().await;
    let rows = rows_at(&s, ts);
    let base = format!(
        "cluster=default,instance={},job=clickhouse-my-postgres,node_name=my-postgres",
        f.addr
    );
    let info: Vec<_> = rows.iter().filter(|r| r.0 == "PostgresServiceInfo").collect();
    assert_eq!(info.len(), 1);
    assert!(info[0]
        .1
        .starts_with("clickhouse_org=ca04a310-730d-4ce0-93dd-39f2cd2d5e6f,cluster=default,instance="));
    assert!(rows
        .iter()
        .any(|r| r.0 == "PostgresServer_MemoryLimitBytes" && r.1.contains(&base[base.find("instance").unwrap()..])));
    // every PostgresServer_* line of the recording is kept (keep regex), nothing else
    let fx = fixture("clickhouse-prometheus.txt");
    let want_n = fx
        .lines()
        .filter(|l| l.starts_with("PostgresServer_") || l.starts_with("PostgresServiceInfo"))
        .count();
    assert_eq!(rows.iter().filter(|r| r.0.starts_with("Postgres")).count(), want_n);
    // the failed scrape: up 0 and a staleness marker for every series the target had
    let after = rows_at(&s, ts + 5000);
    assert_eq!(after.iter().find(|r| r.0 == "up").unwrap().2, 0.0);
    let stale = stale_at(&s, ts + 5000);
    // the scrape_* series stay (scrape_samples_scraped 0), as Prometheus writes them
    assert_eq!(stale.len(), want_n, "every Postgres series ends");
    assert!(stale.iter().all(|m| m.starts_with("Postgres")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disappearing_series_labels_limits_and_header_auth() {
    let f = fake(vec![
        (
            "/m",
            "a{x=\"1\"} 1\na{x=\"2\"} 2\nb{job=\"inner\",instance=\"i\"} 3 1000\n".into(),
            Some("Bearer tok-123".into()),
        ),
        ("/big", "c 1\n".repeat(100), None),
    ])
    .await;
    let conf = tempfile::tempdir().unwrap();
    let tokf = conf.path().join("tok");
    std::fs::write(&tokf, "tok-123").unwrap();
    std::fs::write(
        conf.path().join("j.yml"),
        format!(
            "- job_name: j\n  metrics_path: /m\n  scrape_interval: 15s\n  honor_timestamps: false\n  authorization:\n    credentials_file: {}\n  static_configs:\n    - targets: ['{}']\n      labels: {{env: test}}\n\
             - job_name: big\n  metrics_path: /big\n  sample_limit: 50\n  bearer_token: unused\n  static_configs:\n    - targets: ['{}']\n",
            tokf.display(),
            f.addr,
            f.addr
        ),
    )
    .unwrap();
    let (targets, errors) = load_targets(&files_cfg(conf.path()));
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(targets.len(), 2);
    let (j, big) = (&targets[0], &targets[1]);
    assert_eq!(j.timeout.as_secs(), 10, "default timeout: min(interval, 10s)");
    let d = tempfile::tempdir().unwrap();
    let (s, ing) = setup(d.path()).await;
    let c = reqwest::Client::new();
    let t0 = pgai_mon::model::now_ms() - 60_000;
    assert_eq!(scrape_once(&c, &ing, j, 1 << 20, t0).await.health, "up");
    // a{x="2"} disappears
    f.bodies.lock().unwrap()[0].1 = "a{x=\"1\"} 5\nb{job=\"inner\",instance=\"i\"} 3\n".into();
    assert_eq!(scrape_once(&c, &ing, j, 1 << 20, t0 + 15_000).await.health, "up");
    // over sample_limit: a failed scrape
    let st = scrape_once(&c, &ing, big, 1 << 20, t0).await;
    assert_eq!(st.health, "down");
    assert!(st.last_error.unwrap().contains("sample_limit"));
    // over the body limit: a failed scrape
    let st = scrape_once(&c, &ing, big, 64, t0 + 1).await;
    assert!(st.last_error.unwrap().contains("limit"));
    s.flush().await;
    let rows = rows_at(&s, t0);
    let base = format!("env=test,instance={},job=j", f.addr);
    // colliding exposition labels become exported_*; honor_timestamps false: scrape time
    assert!(rows.contains(&(
        "b".into(),
        format!(
            "env=test,exported_instance=i,exported_job=inner,instance={},job=j",
            f.addr
        ),
        3.0
    )));
    assert!(rows.contains(&("a".into(), format!("{base},x=2"), 2.0)));
    assert_eq!(stale_at(&s, t0 + 15_000), vec!["a".to_string()]);
    let later = rows_at(&s, t0 + 15_000);
    assert!(later.contains(&("a".into(), format!("{base},x=1"), 5.0)));
    assert!(!later.iter().any(|r| r.1.ends_with("x=2")));
    assert!(rows.contains(&("up".into(), format!("instance={},job=big", f.addr), 0.0)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reload_starts_and_stops_targets_and_reports_bad_files() {
    let f = fake(vec![("/m", "a 1\n".into(), None)]).await;
    let conf = tempfile::tempdir().unwrap();
    let file = conf.path().join("t.yml");
    std::fs::write(
        &file,
        format!("- job_name: t\n  metrics_path: /m\n  scrape_interval: 1s\n  static_configs:\n    - targets: ['{}']\n      labels: {{__pgai_rev: r1}}\n", f.addr),
    )
    .unwrap();
    // unknown fields are an error for that file, never ignored
    std::fs::write(
        conf.path().join("bad.yml"),
        "- job_name: x\n  tls_config: {insecure_skip_verify: true}\n",
    )
    .unwrap();
    std::fs::write(conf.path().join("note.txt"), "not a scrape file").unwrap();
    let d = tempfile::tempdir().unwrap();
    let (s, ing) = setup(d.path()).await;
    let sc = Scraper::new(files_cfg(conf.path()), ing.clone()).unwrap();
    assert_eq!(sc.reconcile().await, (1, 0));
    assert_eq!(sc.reconcile().await, (0, 0), "unchanged files: nothing restarts");
    let j = sc.targets_json();
    assert_eq!(j["targets"].as_array().unwrap().len(), 1);
    assert_eq!(j["targets"][0]["labels"]["__pgai_rev"], "r1");
    let errs = j["errors"].as_array().unwrap();
    assert_eq!(errs.len(), 1);
    assert!(errs[0].as_str().unwrap().contains("bad.yml") && errs[0].as_str().unwrap().contains("tls_config"));
    // wait for a scrape
    for _ in 0..60 {
        if sc.targets_json()["targets"][0]["status"]["health"] == "up" {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert_eq!(sc.targets_json()["targets"][0]["status"]["health"], "up");
    // a changed revision restarts the target (same series); removing the file stops it and
    // ends its series
    std::fs::write(&file, std::fs::read_to_string(&file).unwrap().replace("r1", "r2")).unwrap();
    assert_eq!(sc.reconcile().await, (1, 0));
    std::fs::remove_file(&file).unwrap();
    assert_eq!(sc.reconcile().await, (0, 1));
    assert!(sc.targets_json()["targets"].as_array().unwrap().is_empty());
    s.flush().await;
    let stale: Vec<String> = s
        .sql("SELECT DISTINCT r.metric FROM samples s JOIN series r USING (series_id) WHERE s.stale ORDER BY 1")
        .into_iter()
        .map(|r| r[0].as_str().unwrap().into())
        .collect();
    assert_eq!(
        stale,
        vec![
            "a",
            "scrape_duration_seconds",
            "scrape_samples_post_metric_relabeling",
            "scrape_samples_scraped",
            "up"
        ]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_file_that_stops_parsing_keeps_its_last_good_targets() {
    let conf = tempfile::tempdir().unwrap();
    let file = conf.path().join("t.yml");
    let good = "- job_name: t\n  static_configs:\n    - targets: ['127.0.0.1:9', '127.0.0.1:10']\n";
    std::fs::write(&file, good).unwrap();
    let d = tempfile::tempdir().unwrap();
    let (_s, ing) = setup(d.path()).await;
    let sc = Scraper::new(files_cfg(conf.path()), ing).unwrap();
    assert_eq!(sc.reconcile().await, (2, 0));
    // caught half-written: the targets keep running and the error says so
    std::fs::write(&file, &good[..good.len() - 20]).unwrap();
    assert_eq!(sc.reconcile().await, (0, 0));
    let j = sc.targets_json();
    assert_eq!(j["targets"].as_array().unwrap().len(), 2);
    let e = j["errors"][0].as_str().unwrap();
    assert!(
        e.contains("t.yml") && e.contains("its 2 running targets keep the last good version"),
        "{e}"
    );
    std::fs::write(&file, good).unwrap();
    assert_eq!(sc.reconcile().await, (0, 0), "parses again: nothing restarts");
    std::fs::remove_file(&file).unwrap();
    assert_eq!(sc.reconcile().await, (0, 2), "a removed file still stops its targets");
}

#[test]
fn invalid_jobs_are_refused() {
    let conf = tempfile::tempdir().unwrap();
    for (name, body, why) in [
        ("1.yml", "- job_name: a\n  scheme: ftp\n  static_configs: [{targets: [h]}]\n", "scheme"),
        ("2.yml", "- job_name: a\n  scrape_interval: 10s\n  scrape_timeout: 20s\n  static_configs: [{targets: [h]}]\n", "longer"),
        ("3.yml", "- job_name: a\n  basic_auth: {username: u}\n  static_configs: [{targets: [h]}]\n", "password"),
        ("4.yml", "- job_name: a\n  static_configs: [{targets: ['h/x']}]\n", "host[:port]"),
        ("5.yml", "- job_name: a\n  metric_relabel_configs: [{action: hashmod}]\n  static_configs: [{targets: [h]}]\n", "hashmod"),
        ("6.yml", "- job_name: a\n  bearer_token: x\n  basic_auth: {username: u, password: p}\n  static_configs: [{targets: [h]}]\n", "at most one"),
    ] {
        std::fs::write(conf.path().join(name), body).unwrap();
        let (t, e) = load_targets(&ScrapeCfg { files: vec![conf.path().join(name).display().to_string()], ..Default::default() });
        assert!(t.is_empty(), "{name}");
        assert!(e.len() == 1 && e[0].contains(why), "{name}: {e:?}");
    }
}
