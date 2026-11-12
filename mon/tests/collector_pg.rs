//! Collector against a real, throwaway Postgres. Runs only when PGAI_TEST_PG_URL is set to a
//! superuser connection string of a server the tests may reconfigure (ALTER SYSTEM, files in
//! its data directory) and whose sessions they may terminate; skips with a message otherwise.
//!
//! CI: a `postgres:17` service container, e.g.
//!   services: [{name: postgres:17, alias: pg, command: [postgres, -c, shared_preload_libraries=pg_stat_statements]}]
//!   variables: {POSTGRES_PASSWORD: postgres, PGAI_TEST_PG_URL: "postgresql://postgres:postgres@pg:5432/postgres"}
//! Locally: docker run -d --rm -e POSTGRES_PASSWORD=postgres -p 127.0.0.1:55432:5432 postgres:17
//!   PGAI_TEST_PG_URL=postgresql://postgres:postgres@127.0.0.1:55432/postgres cargo test --test collector_pg
use pgai_mon::pgconn::ConnParams;
use std::sync::OnceLock;
use tokio_postgres::{Client, NoTls};

fn url() -> Option<String> {
    let u = std::env::var("PGAI_TEST_PG_URL").ok().filter(|s| !s.is_empty());
    if u.is_none() {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| eprintln!("PGAI_TEST_PG_URL not set: skipping collector tests against Postgres"));
    }
    u
}

macro_rules! need_pg {
    () => {
        match url() {
            Some(u) => u,
            None => {
                eprintln!("skipped: PGAI_TEST_PG_URL not set");
                return;
            }
        }
    };
}

async fn admin(url: &str) -> Client {
    let (c, conn) = tokio_postgres::connect(url, NoTls).await.expect("admin connection");
    tokio::spawn(conn);
    c
}

/// One statement per round trip (ALTER SYSTEM refuses the implicit transaction of a batch).
async fn exec_each(c: &Client, sqls: &[&str]) {
    for s in sqls {
        c.batch_execute(s).await.unwrap_or_else(|e| panic!("{s}: {e}"));
    }
}

/// Tests that reconfigure the server run one at a time.
fn server_lock() -> &'static tokio::sync::Mutex<()> {
    static L: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    L.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Write a PEM file into the server's data directory, mode 0600 as Postgres requires of a key
/// file (COPY TO PROGRAM runs as the server's OS user).
async fn put_server_file(c: &Client, name: &str, pem: &str) {
    let arr = pem
        .lines()
        .map(|l| format!("'{}'", l.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(",");
    let dir: String = c.query_one("show data_directory", &[]).await.unwrap().get(0);
    c.batch_execute(&format!(
        "copy (select l from unnest(array[{arr}]::text[]) with ordinality t(l, n) order by n) \
         to program 'rm -f \"{dir}/{name}\" && umask 077 && cat > \"{dir}/{name}\"'"
    ))
    .await
    .unwrap();
}

struct Pki {
    ca_pem: String,
    ca: rcgen::Certificate,
    ca_key: rcgen::KeyPair,
}

fn pki() -> Pki {
    let ca_key = rcgen::KeyPair::generate().unwrap();
    let mut p = rcgen::CertificateParams::new(Vec::<String>::new()).unwrap();
    p.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    p.distinguished_name.push(rcgen::DnType::CommonName, "pgai-mon test CA");
    let ca = p.self_signed(&ca_key).unwrap();
    Pki {
        ca_pem: ca.pem(),
        ca,
        ca_key,
    }
}

fn server_cert(pki: &Pki, names: &[&str]) -> (String, String) {
    let key = rcgen::KeyPair::generate().unwrap();
    let p = rcgen::CertificateParams::new(names.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap();
    let c = p.signed_by(&key, &pki.ca, &pki.ca_key).unwrap();
    (c.pem(), key.serialize_pem())
}

async fn install_cert(c: &Client, cert: &str, key: &str) {
    put_server_file(c, "pgai_test_server.crt", cert).await;
    put_server_file(c, "pgai_test_server.key", key).await;
    exec_each(
        c,
        &[
            "alter system set ssl = on",
            "alter system set ssl_cert_file = 'pgai_test_server.crt'",
            "alter system set ssl_key_file = 'pgai_test_server.key'",
        ],
    )
    .await;
    reload(c).await;
}

/// pg_reload_conf() and wait until new sessions run with the reloaded configuration.
async fn reload(c: &Client) {
    let load_time = "select pg_conf_load_time()::text";
    let before: String = c.query_one(load_time, &[]).await.unwrap().get(0);
    c.batch_execute("select pg_reload_conf()").await.unwrap();
    for _ in 0..100 {
        let c2 = admin(&url().unwrap()).await;
        let t: String = c2.query_one(load_time, &[]).await.unwrap().get(0);
        if t != before {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("configuration not reloaded");
}

async fn uses_ssl(conn_str: &str) -> anyhow::Result<bool> {
    let c = pgai_mon::pgconn::connect(conn_str, "pgai-mon-test").await?;
    Ok(
        c.query_one("select ssl from pg_stat_ssl where pid = pg_backend_pid()", &[])
            .await?
            .get(0),
    )
}

#[tokio::test]
async fn tls_sslmodes_against_a_self_signed_ca() {
    let url = need_pg!();
    let _g = server_lock().lock().await;
    let a = admin(&url).await;
    let base = ConnParams::parse(&url).unwrap();
    let host = base.get("host").unwrap_or("localhost").to_string();
    let dir = tempfile::tempdir().unwrap();
    let pki = pki();
    let ca_file = dir.path().join("ca.pem");
    std::fs::write(&ca_file, &pki.ca_pem).unwrap();
    let other = self::pki();
    let other_ca = dir.path().join("other.pem");
    std::fs::write(&other_ca, &other.ca_pem).unwrap();
    let with = |mode: &str, root: Option<&std::path::Path>| {
        let mut p = base.clone();
        p.set("sslmode", mode);
        if let Some(r) = root {
            p.set("sslrootcert", r.to_string_lossy());
        }
        p.to_kv_string()
    };

    // a certificate for the host the tests connect to
    let (cert, key) = server_cert(&pki, &[&host]);
    install_cert(&a, &cert, &key).await;
    assert!(!uses_ssl(&with("disable", None)).await.unwrap());
    assert!(uses_ssl(&with("prefer", None)).await.unwrap());
    assert!(uses_ssl(&with("require", None)).await.unwrap());
    assert!(uses_ssl(&with("verify-ca", Some(&ca_file))).await.unwrap());
    assert!(uses_ssl(&with("verify-full", Some(&ca_file))).await.unwrap());
    let e = uses_ssl(&with("verify-ca", Some(&other_ca))).await.unwrap_err();
    assert!(format!("{e:#}").to_lowercase().contains("certificate"), "{e:#}");
    // require with a root certificate verifies the chain (libpq compatibility)
    assert!(uses_ssl(&with("require", Some(&other_ca))).await.is_err());

    // a certificate for another name: the chain is fine, the name is not
    let (cert, key) = server_cert(&pki, &["not-this-host.invalid"]);
    install_cert(&a, &cert, &key).await;
    assert!(uses_ssl(&with("verify-ca", Some(&ca_file))).await.unwrap());
    let e = uses_ssl(&with("verify-full", Some(&ca_file))).await.unwrap_err();
    assert!(format!("{e:#}").to_lowercase().contains("not valid for name"), "{e:#}");

    exec_each(
        &a,
        &[
            "alter system reset ssl",
            "alter system reset ssl_cert_file",
            "alter system reset ssl_key_file",
        ],
    )
    .await;
    reload(&a).await;
}

// ---------------------------------------------------------------------------------------
// The collector itself.

use pgai_mon::collector::{self, CollectorCtx, Emitter, MetricState};
use pgai_mon::config::{CollectorCfg, Source};
use pgai_mon::metricdefs::Scheduled;
use pgai_mon::model::Record;
use pgai_mon::Msg;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What the collector sent to the store: series names by id and the last value of each.
#[derive(Clone, Default)]
struct Sink {
    names: Arc<Mutex<HashMap<u64, (String, Vec<(String, String)>)>>>,
    last: Arc<Mutex<HashMap<u64, f64>>>,
}

impl Sink {
    fn value(&self, metric: &str, label: (&str, &str)) -> Option<f64> {
        let names = self.names.lock().unwrap();
        let last = self.last.lock().unwrap();
        names
            .iter()
            .find(|(_, (m, l))| m == metric && l.iter().any(|(k, v)| k == label.0 && v == label.1))
            .and_then(|(id, _)| last.get(id).copied())
    }

    /// Label sets of a metric's series.
    fn label_sets(&self, metric: &str) -> Vec<Vec<(String, String)>> {
        let names = self.names.lock().unwrap();
        names
            .values()
            .filter(|(m, _)| m == metric)
            .map(|(_, l)| l.clone())
            .collect()
    }
}

struct Harness {
    ctx: CollectorCtx,
    sink: Sink,
    _core: pgai_mon::store::StoreCore,
    _dir: tempfile::TempDir,
}

fn harness(cfg: CollectorCfg) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let core = pgai_mon::store::StoreCore::open(dir.path(), &Default::default()).unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Msg>(4096);
    let sink = Sink::default();
    let s = sink.clone();
    tokio::spawn(async move {
        while let Some(m) = rx.recv().await {
            match m {
                Msg::Rec(Record::Series(defs)) => {
                    let mut n = s.names.lock().unwrap();
                    for d in defs {
                        n.insert(d.series_id, (d.metric, d.labels));
                    }
                }
                Msg::Rec(Record::Samples(v)) => {
                    let mut l = s.last.lock().unwrap();
                    for x in v {
                        l.insert(x.series_id, x.value);
                    }
                }
                _ => {}
            }
        }
    });
    Harness {
        ctx: CollectorCtx {
            em: Emitter::new(core.shared.clone(), tx, cfg),
            objects: Default::default(),
        },
        sink,
        _core: core,
        _dir: dir,
    }
}

/// A new, empty database on the test server; its connection string.
async fn fresh_db(a: &Client, url: &str, name: &str) -> String {
    a.batch_execute(&format!("drop database if exists {name} with (force)"))
        .await
        .unwrap();
    a.batch_execute(&format!("create database {name}")).await.unwrap();
    ConnParams::parse(url).unwrap().with_dbname(name).to_kv_string()
}

fn metric(name: &str, interval_s: u64, def: &str) -> Scheduled {
    Scheduled {
        name: name.into(),
        interval_s,
        def: serde_yaml::from_str(def).unwrap(),
    }
}

fn source(name: &str, conn_str: &str) -> Source {
    Source {
        name: name.into(),
        conn_str: conn_str.into(),
        is_enabled: true,
        ..Default::default()
    }
}

fn job(src: &str, db: &str, job: &str) -> collector::JobStatus {
    collector::source_status(src)
        .and_then(|s| s.databases.get(db).cloned())
        .and_then(|d| d.jobs.get(job).cloned())
        .unwrap_or_default()
}

async fn wait_for(what: &str, secs: u64, mut f: impl FnMut() -> bool) {
    let t = Instant::now();
    while !f() {
        assert!(t.elapsed() < Duration::from_secs(secs), "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn code(e: &anyhow::Error) -> String {
    e.downcast_ref::<tokio_postgres::Error>()
        .and_then(|e| e.code())
        .map(|c| c.code().to_string())
        .unwrap_or_else(|| format!("{e:#}"))
}

#[tokio::test]
async fn sessions_are_read_only_with_lock_timeout_and_pinned_search_path() {
    let url = need_pg!();
    let cfg = CollectorCfg::default();
    let c = collector::connect(&url, &cfg).await.unwrap();
    let show = |k: &'static str| {
        let c = &c;
        async move {
            c.query_one(&format!("show {k}"), &[])
                .await
                .unwrap()
                .get::<_, String>(0)
        }
    };
    assert_eq!(show("default_transaction_read_only").await, "on");
    assert_eq!(show("lock_timeout").await, "100ms");
    assert_eq!(show("statement_timeout").await, "15s");
    assert_eq!(show("application_name").await, "pgai-mon");
    let sp = show("search_path").await;
    assert!(
        sp.starts_with("pg_catalog") && sp.ends_with("public") && !sp.contains("$user"),
        "{sp}"
    );
    let e = c
        .batch_execute("create table pgai_mon_must_not_exist(x int)")
        .await
        .unwrap_err();
    assert_eq!(e.code().unwrap().code(), "25006", "read-only transaction");
    // an explicit search_path wins
    let c2 = collector::connect(
        &url,
        &CollectorCfg {
            search_path: Some("pg_catalog".into()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let sp: String = c2.query_one("show search_path", &[]).await.unwrap().get(0);
    assert_eq!(sp, "pg_catalog");
}

#[tokio::test]
async fn lock_timeout_statement_timeout_and_row_cap() {
    let url = need_pg!();
    let a = admin(&url).await;
    let db = fresh_db(&a, &url, "pgai_t_timeouts").await;
    let a2 = admin(&db).await;
    a2.batch_execute("create table t(x int); insert into t select generate_series(1, 1000)")
        .await
        .unwrap();
    let mut c = collector::connect(&db, &CollectorCfg::default()).await.unwrap();

    // a table locked by DDL: the collector gives up after lock_timeout instead of queueing
    let locker = admin(&db).await;
    locker
        .batch_execute("begin; lock table t in access exclusive mode")
        .await
        .unwrap();
    let t0 = Instant::now();
    let e = collector::query_capped(&mut c, "select count(*) from t", &[], 10)
        .await
        .unwrap_err();
    assert_eq!(code(&e), "55P03");
    assert!(t0.elapsed() < Duration::from_secs(3), "{:?}", t0.elapsed());
    locker.batch_execute("rollback").await.unwrap();

    // statement_timeout per job
    let mut cur = 15;
    collector::set_timeout(&c, &mut cur, 1).await.unwrap();
    let e = collector::query_capped(&mut c, "select pg_sleep(3)", &[], 10)
        .await
        .unwrap_err();
    assert_eq!(code(&e), "57014");
    collector::set_timeout(&c, &mut cur, 15).await.unwrap();

    // the row cap stops the fetch, and the session stays usable
    let (rows, cut) = collector::query_capped(&mut c, "select x from t order by x", &[], 10)
        .await
        .unwrap();
    assert_eq!((rows.len(), cut), (10, true));
    let (rows, cut) = collector::query_capped(&mut c, "select x from t", &[], 1000)
        .await
        .unwrap();
    assert_eq!((rows.len(), cut), (1000, false));

    // through collect_metric: the cut is counted and exported
    let h = harness(CollectorCfg {
        max_rows_per_query: 5,
        ..Default::default()
    });
    let m = metric("many", 60, "sqls: {11: 'select x as tag_x, x as v from t'}");
    let mut st = MetricState::default();
    let base = vec![("dbname".to_string(), "rows".to_string())];
    let n = collector::collect_metric(&mut c, &m, m.def.sql_for(17).unwrap(), &base, &h.ctx.em, &mut st)
        .await
        .unwrap();
    assert_eq!((n, st.rows_cut), (5, 1));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(h.sink.value("pgai_mon_rows_cut_total", ("metric", "many")), Some(1.0));
}

#[tokio::test]
async fn slow_metrics_do_not_delay_fast_ones() {
    let url = need_pg!();
    let a = admin(&url).await;
    let db = fresh_db(&a, &url, "pgai_t_cadence").await;
    let h = harness(CollectorCfg {
        self_metrics_interval_s: 1,
        ..Default::default()
    });
    let primary = vec![
        metric("fast", 1, "sqls: {11: 'select 1 as v'}"),
        metric("slow", 1, "sqls: {11: 'select 1 as v from pg_sleep(4)'}"),
        metric(
            "timed_out",
            1,
            "{sqls: {11: 'select 1 as v from pg_sleep(3)'}, statement_timeout_seconds: 1}",
        ),
    ];
    let (stop, rx) = tokio::sync::watch::channel(false);
    tokio::spawn(collector::run_source(
        source("cadence", &db),
        primary,
        vec![],
        h.ctx.clone(),
        rx,
    ));
    tokio::time::sleep(Duration::from_secs(7)).await;
    let fast = job("cadence", "cadence", "fast");
    let slow = job("cadence", "cadence", "slow");
    let to = job("cadence", "cadence", "timed_out");
    assert!(
        fast.runs >= 5,
        "fast ran {} times in 7 s while slow was running",
        fast.runs
    );
    assert!(slow.runs <= 2 && slow.skipped >= 3, "slow: {slow:?}");
    assert!(to.timeouts >= 1 && to.errors >= to.timeouts, "timed_out: {to:?}");
    assert_eq!(fast.errors, 0);
    // own series: up, errors and timeouts, skipped slots
    assert_eq!(h.sink.value("pgai_mon_source_up", ("dbname", "cadence")), Some(1.0));
    assert!(
        h.sink
            .value("pgai_mon_timeouts_total", ("metric", "timed_out"))
            .unwrap_or(0.0)
            >= 1.0
    );
    assert!(
        h.sink
            .value("pgai_mon_skipped_slots_total", ("metric", "slow"))
            .unwrap_or(0.0)
            >= 1.0
    );
    assert!(h.sink.value("pgwatch_fast_v", ("dbname", "cadence")).is_some());
    let _ = stop.send(true);
}

#[tokio::test]
async fn reconnects_after_its_sessions_are_killed() {
    let url = need_pg!();
    let a = admin(&url).await;
    let db = fresh_db(&a, &url, "pgai_t_reconnect").await;
    let h = harness(CollectorCfg {
        reconnect_min_s: 1,
        reconnect_max_s: 2,
        role_check_interval_s: 1,
        ..Default::default()
    });
    let (stop, rx) = tokio::sync::watch::channel(false);
    tokio::spawn(collector::run_source(
        source("reconnect", &db),
        vec![metric("fast", 1, "sqls: {11: 'select 1 as v'}")],
        vec![],
        h.ctx.clone(),
        rx,
    ));
    wait_for("first runs", 10, || job("reconnect", "reconnect", "fast").runs >= 2).await;
    let killed: i64 = a
        .query_one(
            "select count(pg_terminate_backend(pid)) from pg_stat_activity \
             where datname = 'pgai_t_reconnect' and application_name = 'pgai-mon'",
            &[],
        )
        .await
        .unwrap()
        .get(0);
    assert!(killed >= 1);
    let after = Instant::now();
    let ok_before = job("reconnect", "reconnect", "fast").last_success_ms.unwrap();
    wait_for("collection after the kill", 15, || {
        job("reconnect", "reconnect", "fast").last_success_ms.unwrap_or(0) > ok_before + 1500
    })
    .await;
    assert!(after.elapsed() < Duration::from_secs(10));
    let st = collector::source_status("reconnect").unwrap();
    assert!(st.databases["reconnect"].up);
    let _ = stop.send(true);
}

#[tokio::test]
async fn an_unreachable_source_backs_off_and_reports_down() {
    let _ = need_pg!();
    let h = harness(CollectorCfg {
        reconnect_min_s: 1,
        reconnect_max_s: 2,
        self_metrics_interval_s: 1,
        ..Default::default()
    });
    let (stop, rx) = tokio::sync::watch::channel(false);
    // nothing listens on port 1
    tokio::spawn(collector::run_source(
        source(
            "nowhere",
            "host=127.0.0.1 port=1 user=x dbname=x sslmode=disable connect_timeout=2",
        ),
        vec![metric("fast", 1, "sqls: {11: 'select 1 as v'}")],
        vec![],
        h.ctx.clone(),
        rx,
    ));
    tokio::time::sleep(Duration::from_secs(6)).await;
    let st = collector::source_status("nowhere").unwrap();
    assert_eq!(st.state, "starting");
    let d = &st.databases["nowhere"];
    assert!(!d.up);
    assert!(
        d.last_error.as_deref().unwrap_or("").contains("refused"),
        "{:?}",
        d.last_error
    );
    assert_eq!(h.sink.value("pgai_mon_source_up", ("dbname", "nowhere")), Some(0.0));
    let _ = stop.send(true);
}

/// The collector's own series carry the source identity (cluster, node_name) besides dbname.
/// Needs no Postgres: nothing listens on port 1.
#[tokio::test]
async fn own_series_carry_the_source_identity() {
    let h = harness(CollectorCfg {
        reconnect_min_s: 1,
        reconnect_max_s: 2,
        self_metrics_interval_s: 1,
        ..Default::default()
    });
    let (stop, rx) = tokio::sync::watch::channel(false);
    let mut src = source(
        "ident",
        "host=127.0.0.1 port=1 user=x dbname=x sslmode=disable connect_timeout=2",
    );
    src.custom_tags = [("cluster", "c1"), ("node_name", "n1"), ("env", "e")]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    tokio::spawn(collector::run_source(
        src,
        vec![metric("fast", 1, "sqls: {11: 'select 1 as v'}")],
        vec![],
        h.ctx.clone(),
        rx,
    ));
    let t0 = Instant::now();
    while h.sink.label_sets("pgai_mon_connect_failures_total").is_empty() && t0.elapsed() < Duration::from_secs(10) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let want = vec![
        ("cluster".to_string(), "c1".to_string()),
        ("dbname".to_string(), "ident".to_string()),
        ("node_name".to_string(), "n1".to_string()),
    ];
    for m in ["pgai_mon_source_up", "pgai_mon_connect_failures_total"] {
        assert_eq!(h.sink.label_sets(m), vec![want.clone()], "{m}");
    }
    let _ = stop.send(true);
}

#[tokio::test]
async fn discovery_over_many_databases_stays_within_the_session_cap() {
    let url = need_pg!();
    let a = admin(&url).await;
    let main = fresh_db(&a, &url, "pgai_t_disc_0").await;
    for i in 1..=6 {
        fresh_db(&a, &url, &format!("pgai_t_disc_{i}")).await;
    }
    let h = harness(CollectorCfg {
        max_sessions_per_node: 4,
        discovery_interval_s: 1,
        role_check_interval_s: 1,
        ..Default::default()
    });
    let mut src = source("disc", &main);
    src.discover_databases = true;
    src.include_pattern = Some("^pgai_t_disc_".into());
    let (stop, rx) = tokio::sync::watch::channel(false);
    tokio::spawn(collector::run_source(
        src,
        vec![metric("per_db", 1, "sqls: {11: 'select 1 as v'}")],
        vec![],
        h.ctx.clone(),
        rx,
    ));
    let t = Instant::now();
    let mut max_sessions = 0i64;
    while t.elapsed() < Duration::from_secs(6) {
        let n: i64 = a
            .query_one(
                "select count(*) from pg_stat_activity where application_name = 'pgai-mon' and datname like 'pgai\\_t\\_disc\\_%'",
                &[],
            )
            .await
            .unwrap()
            .get(0);
        max_sessions = max_sessions.max(n);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(max_sessions <= 4, "{max_sessions} sessions");
    let st = collector::source_status("disc").unwrap();
    assert_eq!(st.databases.len(), 7, "{:?}", st.databases.keys());
    for (label, d) in &st.databases {
        assert!(
            d.jobs.get("per_db").is_some_and(|j| j.runs >= 1),
            "{label}: {:?}",
            d.jobs.get("per_db")
        );
    }
    // a dropped database leaves at the next discovery
    a.batch_execute("drop database pgai_t_disc_6 with (force)")
        .await
        .unwrap();
    wait_for("pgai_t_disc_6 to leave", 10, || {
        !collector::source_status("disc")
            .unwrap()
            .databases
            .contains_key("disc_pgai_t_disc_6")
    })
    .await;
    let _ = stop.send(true);
}

#[tokio::test]
async fn sources_file_changes_are_picked_up() {
    let url = need_pg!();
    let a = admin(&url).await;
    let db = fresh_db(&a, &url, "pgai_t_reload").await;
    let dir = tempfile::tempdir().unwrap();
    let mf = dir.path().join("metrics.yml");
    std::fs::write(
        &mf,
        "metrics:\n  fast:\n    sqls: {11: 'select 1 as v'}\n  other:\n    sqls: {11: 'select 2 as v'}\n\
         presets:\n  full:\n    metrics: {fast: 1}\n  second:\n    metrics: {other: 1}\n",
    )
    .unwrap();
    let inst = dir.path().join("instances.yml");
    let entry = |name: &str, enabled: bool, preset: &str| {
        format!(
            "- name: {name}\n  conn_str: \"{}\"\n  preset_metrics: {preset}\n  custom_metrics:\n  is_enabled: {enabled}\n  group: default\n  custom_tags:\n    sink_type: ~sink_type~\n",
            db.replace('"', "\\\"")
        )
    };
    std::fs::write(&inst, entry("reload_a", true, "full")).unwrap();
    let cfg_path = dir.path().join("pgai-mon.yml");
    std::fs::write(
        &cfg_path,
        format!(
            "store_dir: {}\nmetrics_yml: {}\nsources_file: {}\ncollector: {{reload_interval_s: 1}}\n",
            dir.path().join("store").display(),
            mf.display(),
            inst.display()
        ),
    )
    .unwrap();
    let cfg = pgai_mon::config::Config::load(&cfg_path).unwrap();
    let h = harness(cfg.collector.clone());
    let mgr = tokio::spawn(collector::run_manager(cfg, h.ctx.clone()));
    wait_for("reload_a", 10, || job("reload_a", "reload_a", "fast").runs >= 1).await;
    assert_eq!(h.sink.value("pgwatch_fast_v", ("sink_type", "prometheus")), Some(1.0));
    // disable a, add b with another preset
    std::fs::write(
        &inst,
        entry("reload_a", false, "full") + &entry("reload_b", true, "second"),
    )
    .unwrap();
    wait_for("reload_b", 10, || job("reload_b", "reload_b", "other").runs >= 1).await;
    wait_for("reload_a to stop", 5, || collector::source_status("reload_a").is_none()).await;
    mgr.abort();
}

#[tokio::test]
async fn column_types_map_like_pgwatch() {
    let url = need_pg!();
    let c = collector::connect(&url, &CollectorCfg::default()).await.unwrap();
    let r = c
        .query_one(
            "select 1::int2 as small, 2::int4 as i4, 3::int8 as i8, 1.5::float4 as f4, true as b, \
             2.5::numeric as num, 16384::oid as o, '0/16B3748'::pg_lsn as lsn, 'x'::name as n, \
             repeat('d', 400) as long_def, now() as ts, \
             7::int2 as tag_small, 16384::oid as tag_o, '2024-01-02 03:04:05.1234+00'::timestamptz as tag_t, \
             1234567.0::float8 as tag_f",
            &[],
        )
        .await
        .unwrap();
    let cols: Vec<(&str, collector::PgVal)> = r
        .columns()
        .iter()
        .enumerate()
        .map(|(i, col)| (col.name(), r.get(i)))
        .collect();
    let mut ts = 0;
    let (labels, fields) = collector::row_fields("m", &cols, &Default::default(), true, &mut ts);
    let names: Vec<&str> = fields.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        vec!["i4", "i8", "f4", "b"],
        "int2, numeric, oid, timestamptz are not samples"
    );
    assert_eq!(labels["lsn"], "0/16B3748");
    assert_eq!(labels["n"], "x");
    assert_eq!(labels["long_def"].len(), 400, "long text stays a label");
    assert_eq!(labels["small"], "7");
    assert_eq!(labels["o"], "16384");
    assert_eq!(labels["t"], "2024-01-02 03:04:05.1234 +0000 UTC");
    assert_eq!(labels["f"], "1.234567e+06");
    assert!(!labels.contains_key("ts") && !labels.contains_key("num"));
}
