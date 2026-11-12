//! The real `run` loop on an object store: the built binary with a `file://` bucket
//! (objstore::LocalCas, LocalFileSystem with a conditional PUT), log lines as the data source,
//! short intervals. Covers what tests/bucket.rs cannot: PGAI_MON_STORE, the uploader thread,
//! the final upload on SIGTERM, the lease at start, and the fenced writer's exit code.
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_pgai-mon");

/// `n` jsonlog lines starting at line `from`, appended to `<dir>/pg.json`.
fn write_log(dir: &Path, from: usize, n: usize) {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("pg.json"))
        .unwrap();
    for k in from..from + n {
        writeln!(
            f,
            r#"{{"timestamp":"2026-11-09 00:{:02}:{:02}.000 UTC","pid":{k},"error_severity":"LOG","message":"line {k}","backend_type":"client backend"}}"#,
            (k / 60) % 60,
            k % 60
        )
        .unwrap();
    }
}

/// A config whose store_dir the environment overrides (PGAI_MON_STORE).
fn config(dir: &Path, name: &str, local: &Path, logs: Option<&Path>, upload_s: u64) -> PathBuf {
    let logs = logs.map_or(String::new(), |d| format!("logs:\n  - dir: {}\n", d.display()));
    let y = format!(
        "store_dir: /nonexistent/overridden\nmetrics_yml: /nonexistent/metrics.yml\n{logs}\
         store:\n  flush_interval_s: 1\n  s3:\n    local_dir: {}\n    upload_interval_s: {upload_s}\n    \
         delete_grace_s: 1\nhttp:\n  listen: 127.0.0.1:0\n",
        local.display()
    );
    let p = dir.join(format!("{name}.yml"));
    std::fs::write(&p, y).unwrap();
    p
}

fn spawn(cfg: &Path, url: &str) -> Child {
    Command::new(BIN)
        .args(["run", "--config", cfg.to_str().unwrap()])
        .env("PGAI_MON_STORE", url)
        .env("PGAI_NO_COMPACT", "1")
        .env("PGAI_MON_TOKEN", "test-token-0123456789")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

fn catalog(p: &Path) -> Option<serde_json::Value> {
    serde_json::from_slice(&std::fs::read(p).ok()?).ok()
}

fn log_rows(cat: &serde_json::Value) -> u64 {
    cat["files"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["kind"] == "logs")
        .map(|f| f["rows"].as_u64().unwrap())
        .sum()
}

fn wait_for(what: &str, secs: u64, mut ok: impl FnMut() -> bool) {
    let t = Instant::now();
    while !ok() {
        assert!(t.elapsed() < Duration::from_secs(secs), "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn stderr(c: Child) -> (std::process::ExitStatus, String) {
    let out = c.wait_with_output().unwrap();
    (out.status, String::from_utf8_lossy(&out.stderr).into_owned())
}

fn signal(c: &Child, sig: i32) {
    assert_eq!(unsafe { libc::kill(c.id() as i32, sig) }, 0);
}

/// Every file the bucket catalog lists is an object of exactly its size.
fn intact(bucket: &Path, cat: &serde_json::Value) {
    for f in cat["files"].as_array().unwrap() {
        let p = bucket.join(f["path"].as_str().unwrap());
        let n = std::fs::metadata(&p)
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
            .len();
        assert_eq!(n, f["bytes"].as_u64().unwrap(), "{}", p.display());
    }
}

/// PGAI_MON_STORE selects the bucket; with an upload interval of an hour only the start's
/// commit happens during the run, so everything ingested reaches the bucket by the final
/// upload of a graceful stop (SIGTERM).
#[test]
fn sigterm_uploads_everything_flushed() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().canonicalize().unwrap();
    let (logs, local, bucket) = (root.join("logs"), root.join("local"), root.join("bucket"));
    std::fs::create_dir_all(&logs).unwrap();
    std::fs::create_dir_all(&bucket).unwrap();
    let url = format!("file://{}/st", bucket.display());
    write_log(&logs, 0, 300);
    let cfg = config(&root, "a", &local, Some(&logs), 3600);
    let c = spawn(&cfg, &url);
    let bcat = bucket.join("st/catalog.json");
    wait_for("the start's commit", 30, || catalog(&bcat).is_some());
    write_log(&logs, 300, 200);
    wait_for("500 log rows flushed locally", 30, || {
        catalog(&local.join("catalog.json")).is_some_and(|c| log_rows(&c) == 500)
    });
    assert_eq!(
        log_rows(&catalog(&bcat).unwrap()),
        0,
        "nothing uploaded before the stop"
    );
    signal(&c, libc::SIGTERM);
    let (st, err) = stderr(c);
    assert!(st.success(), "{st:?}\n{err}");
    let cat = catalog(&bcat).unwrap();
    assert_eq!(log_rows(&cat), 500, "{err}");
    intact(&bucket.join("st"), &cat);
}

/// Two writers on one prefix through the real loop: a second start is refused while the first
/// commits; once the first stops committing (SIGSTOP) for lease_intervals (3) x the upload
/// interval, a start takes the store over with no flag; the first wakes up (SIGCONT), its
/// next commit is refused, it exits with code 4, and the bucket stays the new writer's.
#[test]
fn stale_lease_is_taken_over_and_the_zombie_exits_4() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().canonicalize().unwrap();
    let (logs, bucket) = (root.join("logs"), root.join("bucket"));
    std::fs::create_dir_all(&logs).unwrap();
    std::fs::create_dir_all(&bucket).unwrap();
    let url = format!("file://{}/st", bucket.display());
    let bcat = bucket.join("st/catalog.json");
    write_log(&logs, 0, 100);
    let a = spawn(&config(&root, "a", &root.join("la"), Some(&logs), 1), &url);
    wait_for("A's logs in the bucket", 30, || {
        catalog(&bcat).is_some_and(|c| log_rows(&c) == 100)
    });
    let id_a = std::fs::read_to_string(root.join("la/writer.id")).unwrap();

    // fresh lease: refused
    let cfg_b = config(&root, "b", &root.join("lb"), None, 1);
    let (st, err) = stderr(spawn(&cfg_b, &url));
    assert!(
        !st.success() && err.contains("refusing to start a second writer"),
        "{st:?}\n{err}"
    );

    // A stops committing; after the lease (3 s) B takes over without takeover: true
    signal(&a, libc::SIGSTOP);
    std::thread::sleep(Duration::from_millis(3_500));
    let b = spawn(&cfg_b, &url);
    wait_for("B's commit", 30, || {
        let id_b = std::fs::read_to_string(root.join("lb/writer.id")).unwrap_or_default();
        !id_b.is_empty()
            && catalog(&bcat).is_some_and(|c| c["writer"]["id"] == id_b.as_str() && c["writer"]["epoch"] == 2)
    });
    let id_b = std::fs::read_to_string(root.join("lb/writer.id")).unwrap();
    assert_ne!(id_a, id_b);

    // the zombie wakes up with new data and is fenced at its next commit
    write_log(&logs, 100, 50);
    signal(&a, libc::SIGCONT);
    let t = Instant::now();
    let (st, err) = stderr(a);
    assert_eq!(st.code(), Some(4), "{err}");
    assert!(err.contains("is fenced"), "{err}");
    assert!(t.elapsed() < Duration::from_secs(20));
    let cat = catalog(&bcat).unwrap();
    assert_eq!(cat["writer"]["id"], id_b.as_str());
    assert_eq!(log_rows(&cat), 100, "the zombie's rows never got in");
    intact(&bucket.join("st"), &cat);
    // B goes on committing
    let hb = cat["writer"]["heartbeat_ms"].as_i64().unwrap();
    wait_for("B's next commit", 10, || {
        catalog(&bcat).is_some_and(|c| c["writer"]["heartbeat_ms"].as_i64().unwrap() > hb)
    });
    signal(&b, libc::SIGTERM);
    let (st, err) = stderr(b);
    assert!(st.success(), "{err}");
    intact(&bucket.join("st"), &catalog(&bcat).unwrap());
}
