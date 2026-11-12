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
    Ok(c.query_one("select ssl from pg_stat_ssl where pid = pg_backend_pid()", &[]).await?.get(0))
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
