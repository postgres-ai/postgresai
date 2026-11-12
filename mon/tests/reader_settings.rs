//! The reader's DuckDB session: `threads` is 1 by default (the query's own thread runs its plan;
//! no background workers for the caller to poll), the override reaches DuckDB (0 =
//! max_concurrent, N = N), and the access lockdown is the same whatever the thread count.
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::query::{Reader, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::StoreCore;

fn settings(threads: Option<i64>, max_concurrent: usize) -> Vec<String> {
    let dir = tempfile::tempdir().unwrap();
    let core = StoreCore::open(&dir.path().join("s"), &StoreCfg::default()).unwrap();
    let mut cfg = HttpCfg {
        memory_limit: "256MB".into(),
        max_concurrent,
        ..Default::default()
    };
    if let Some(t) = threads {
        cfg.threads = t;
    }
    let r = Reader::new(core.shared.clone(), cfg, EMBEDDED_VIEWS_SQL.to_string(), vec![]).unwrap();
    let names = [
        "threads",
        "enable_external_access",
        "lock_configuration",
        "allow_community_extensions",
        "allow_unsigned_extensions",
        "autoinstall_known_extensions",
        "autoload_known_extensions",
    ];
    let sql = format!(
        "SELECT {}",
        names
            .iter()
            .map(|n| format!("CAST(current_setting('{n}') AS VARCHAR) AS {n}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let res = r.run(&sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    let row: serde_json::Value = serde_json::from_str(res.rows[0].get()).unwrap();
    // the lockdown holds: a query cannot raise the thread count
    let err = r.run("SET threads = 8");
    assert!(err.is_err(), "SET threads succeeded on a locked reader");
    names
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let v = if row.is_array() { &row[i] } else { &row[*n] };
            v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())
        })
        .collect()
}

#[test]
fn reader_duckdb_threads_default_one_and_override() {
    assert_eq!(HttpCfg::default().resolve().threads, 1);
    let locked = |s: &[String]| s[1..].to_vec();
    let want_locked: Vec<String> = ["false", "true", "false", "false", "false", "false"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let d = settings(None, 3);
    assert_eq!(d[0], "1", "default threads: {d:?}");
    assert_eq!(locked(&d), want_locked, "lockdown at threads 1: {d:?}");
    let auto = settings(Some(0), 3);
    assert_eq!(auto[0], "3", "threads 0 = max_concurrent: {auto:?}");
    assert_eq!(locked(&auto), want_locked, "lockdown at threads 0: {auto:?}");
    let two = settings(Some(2), 4);
    assert_eq!(two[0], "2", "{two:?}");
    assert_eq!(locked(&two), want_locked, "lockdown at threads 2: {two:?}");
}
