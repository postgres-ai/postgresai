//! L1 result cache (qcache.rs): coalescing, generations, cancellation, errors, eviction, cache
//! keys and cacheability. (Byte identity against PGAI_QCACHE=0 over the dashboard and diff sets is
//! checked by a script: README.md.)
use pgai_mon::qcache::{normalize, QueryCache, Served, ENTRY_OVERHEAD};
use pgai_mon::query::{QueryError, Reply};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn cache() -> Arc<QueryCache> {
    Arc::new(QueryCache::new(16 << 20, 2 << 20, "views"))
}

fn reply(body: &str, gen: u64) -> Reply {
    Reply {
        body: body.into(),
        gen,
        cacheable: true,
        range: None,
        dict_version: 0,
    }
}

/// N identical concurrent requests run the backend once; the next one is a hit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn identical_concurrent_requests_run_once() {
    let c = cache();
    let runs = Arc::new(AtomicUsize::new(0));
    let mut hs = vec![];
    for _ in 0..8 {
        let (c, runs) = (c.clone(), runs.clone());
        hs.push(tokio::spawn(async move {
            c.get_or_run("SELECT 1", 0, 7, move || {
                let runs = runs.clone();
                async move {
                    runs.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    Ok(reply("[1]", 7))
                }
            })
            .await
        }));
    }
    let mut served = vec![];
    for h in hs {
        let (r, s) = h.await.unwrap();
        assert_eq!(r.unwrap().body, "[1]");
        served.push(s);
    }
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert_eq!(served.iter().filter(|s| **s == Served::Miss).count(), 1);
    assert_eq!(served.iter().filter(|s| **s == Served::Coalesced).count(), 7);
    let (r, s) = c.get_or_run("SELECT 1 ;", 0, 7, || async { panic!("must hit") }).await;
    assert_eq!((r.unwrap().body.as_str(), s), ("[1]", Served::Hit));
    // a different format or SQL text is a different key
    let (_, s) = c.get_or_run("SELECT 1", 1, 7, || async { Ok(reply("x", 7)) }).await;
    assert_eq!(s, Served::Miss);
    // a reply the reader marks not cacheable (volatile SQL) is returned, never stored
    for _ in 0..2 {
        let (_, s) = c
            .get_or_run("SELECT now()", 0, 7, || async {
                Ok(Reply {
                    body: "t".into(),
                    gen: 7,
                    cacheable: false,
                    range: None,
                    dict_version: 0,
                })
            })
            .await;
        assert_eq!(s, Served::Miss);
    }
    assert_eq!(c.stats_json()["not_stored_volatile"], 2);
}

/// The generation changes during a flight: the result is returned, not stored; and a new
/// generation drops what was stored.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generation_change_during_flight_is_not_stored() {
    let c = cache();
    let c2 = c.clone();
    let lead = tokio::spawn(async move {
        c2.get_or_run("SELECT 2", 0, 7, || async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(reply("old", 7))
        })
        .await
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    let (_, s) = c.get_or_run("SELECT 3", 0, 8, || async { Ok(reply("x", 8)) }).await;
    assert_eq!(s, Served::Miss);
    let (r, s) = lead.await.unwrap();
    assert_eq!((r.unwrap().body.as_str(), s), ("old", Served::Miss));
    let (r, s) = c.get_or_run("SELECT 2", 0, 8, || async { Ok(reply("new", 8)) }).await;
    assert_eq!(
        (r.unwrap().body.as_str(), s),
        ("new", Served::Miss),
        "not stored under gen 8"
    );
    let (_, s) = c.get_or_run("SELECT 2", 0, 8, || async { Ok(reply("n2", 8)) }).await;
    assert_eq!(s, Served::Hit);
    let (r, s) = c.get_or_run("SELECT 2", 0, 9, || async { Ok(reply("g9", 9)) }).await;
    assert_eq!(
        (r.unwrap().body.as_str(), s),
        ("g9", Served::Miss),
        "gen 9 drops gen 8 entries"
    );
}

/// The leader's client goes away. The flight keeps running (it owns the
/// admission permit), the follower gets its result, and the query runs once, not twice.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_leader_does_not_rerun_the_query() {
    let c = cache();
    let runs = Arc::new(AtomicUsize::new(0));
    let mk = |runs: Arc<AtomicUsize>| {
        move || {
            let runs = runs.clone();
            async move {
                runs.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(150)).await;
                Ok(reply("done", 1))
            }
        }
    };
    let (c1, r1) = (c.clone(), runs.clone());
    let leader = tokio::spawn(async move { c1.get_or_run("SELECT 4", 0, 1, mk(r1)).await });
    tokio::time::sleep(Duration::from_millis(30)).await;
    let (c2, r2) = (c.clone(), runs.clone());
    let follower = tokio::spawn(async move { c2.get_or_run("SELECT 4", 0, 1, mk(r2)).await });
    tokio::time::sleep(Duration::from_millis(30)).await;
    leader.abort();
    let (r, s) = follower.await.unwrap();
    assert_eq!(r.unwrap().body, "done");
    assert_eq!(s, Served::Coalesced);
    assert_eq!(runs.load(Ordering::SeqCst), 1, "no second run");
    let (_, s) = c.get_or_run("SELECT 4", 0, 1, || async { panic!("stored") }).await;
    assert_eq!(s, Served::Hit);

    // a sole leader that goes away: the run completes and is stored, no slot is left
    let c1 = c.clone();
    let h = tokio::spawn(async move {
        c1.get_or_run("SELECT 5", 0, 1, || async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(reply("y", 1))
        })
        .await
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    h.abort();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(c.stats_json()["in_flight"], 0);
    let (r, s) = c.get_or_run("SELECT 5", 0, 1, || async { panic!("stored") }).await;
    assert_eq!((r.unwrap().body.as_str(), s), ("y", Served::Hit));
}

/// A panicking run fails its waiters with `internal`, frees the slot; a follower's own retry
/// then runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn leader_panic_frees_the_flight() {
    let c = cache();
    let c1 = c.clone();
    let lp = tokio::spawn(async move {
        c1.get_or_run("SELECT 3", 0, 1, || async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            panic!("leader boom");
            #[allow(unreachable_code)]
            Ok(reply("x", 1))
        })
        .await
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    let c2 = c.clone();
    let f = tokio::spawn(async move { c2.get_or_run("SELECT 3", 0, 1, || async { Ok(reply("f", 1)) }).await });
    let (r, s) = lp.await.unwrap();
    assert_eq!((r.unwrap_err().kind, s), ("internal", Served::Miss));
    let (r, _) = f.await.unwrap();
    assert_eq!(r.unwrap().body, "f");
    assert_eq!(c.stats_json()["in_flight"], 0);
}

/// After a transient error, followers re-coalesce behind one retry; a
/// deterministic error is shared; neither is stored.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn errors_are_shared_only_when_deterministic_and_never_stored() {
    for (kind, n, want_runs, want_oks) in [
        ("out_of_memory", 4, 2, 3),
        ("out_of_memory", 21, 2, 20),
        ("binder_error", 4, 1, 0),
    ] {
        let c = cache();
        let runs = Arc::new(AtomicUsize::new(0));
        let mut hs = vec![];
        for i in 0..n {
            let (c, runs) = (c.clone(), runs.clone());
            hs.push(tokio::spawn(async move {
                if i > 0 {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                c.get_or_run("SELECT 5", 0, 1, move || {
                    let runs = runs.clone();
                    async move {
                        let k = runs.fetch_add(1, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(80)).await;
                        if k == 0 || kind == "binder_error" {
                            Err(QueryError {
                                kind,
                                message: "x".into(),
                            })
                        } else {
                            Ok(reply("retried", 1))
                        }
                    }
                })
                .await
            }));
        }
        let mut oks = 0;
        for h in hs {
            let (r, _) = h.await.unwrap();
            oks += r.is_ok() as usize;
        }
        assert_eq!((runs.load(Ordering::SeqCst), oks), (want_runs, want_oks), "{kind} x{n}");
        if kind == "binder_error" {
            let (_, s) = c.get_or_run("SELECT 5", 0, 1, || async { Ok(reply("y", 1)) }).await;
            assert_eq!(s, Served::Miss, "{kind}: errors are never stored");
        } else {
            let (_, s) = c.get_or_run("SELECT 5", 0, 1, || async { panic!("stored") }).await;
            assert_eq!(s, Served::Hit, "the coalesced retry's result is stored");
        }
    }
}

/// Entries are charged their overhead, the cap holds, and eviction is LRU.
#[tokio::test]
async fn eviction_keeps_the_cap_and_is_lru() {
    let c = Arc::new(QueryCache::new(64 << 10, 4096, "v"));
    for i in 0..2000 {
        let body: String = (0..100).map(|k| ((i * 7 + k * 13) % 251) as u8 as char).collect();
        let _ = c
            .get_or_run(&format!("SELECT {i}"), 0, 1, move || {
                let b = body.clone();
                async move { Ok(reply(&b, 1)) }
            })
            .await;
        // keep SELECT 0 recently used
        let (_, s) = c.get_or_run("SELECT 0", 0, 1, || async { Ok(reply("again", 1)) }).await;
        assert_eq!(s, Served::Hit, "i={i}: the most recently used entry survives");
    }
    let s = c.stats_json();
    let (bytes, entries) = (s["bytes"].as_u64().unwrap(), s["entries"].as_u64().unwrap());
    assert!(bytes <= 64 << 10, "{s}");
    assert!(bytes >= entries * ENTRY_OVERHEAD as u64, "overhead counted: {s}");
    assert!(s["evictions"].as_u64().unwrap() > 1000, "{s}");
}

/// A request that loaded an older generation runs without touching the cache.
#[tokio::test]
async fn older_generation_does_not_move_the_cache_back() {
    let c = cache();
    let _ = c.get_or_run("SELECT 1", 0, 6, || async { Ok(reply("g6", 6)) }).await;
    let (_, s) = c.get_or_run("SELECT 9", 0, 5, || async { Ok(reply("g5", 5)) }).await;
    assert_eq!(s, Served::Bypass);
    assert_eq!(c.stats_json()["generation"], 6);
    let (_, s) = c.get_or_run("SELECT 1", 0, 6, || async { panic!("kept") }).await;
    assert_eq!(s, Served::Hit);
}

/// No folding: SQL texts with different results never share a key.
#[test]
fn keys_are_the_trimmed_text() {
    for (a, b) in [
        (
            "SELECT prom_align_down(-5, 3) AS v",
            "SELECT prom_align_down(-6, 3) AS v",
        ),
        ("SELECT prom_align_down(7, 5)", "SELECT prom_align_down(9, 5)"),
        (
            "SELECT prom_align_down(1794206627 * 1000, 30000) AS e",
            "SELECT prom_align_down(1794206627000, 30000) AS e",
        ),
    ] {
        assert_ne!(normalize(a), normalize(b));
    }
    assert_eq!(normalize("  SELECT 1 ;\n"), "SELECT 1");
}

/// Cacheability is decided on DuckDB's parse tree in the reader.
#[test]
fn reader_marks_volatile_and_external_sql_not_cacheable() {
    use pgai_mon::config::{HttpCfg, StoreCfg};
    use pgai_mon::query::{Reader, EMBEDDED_VIEWS_SQL};
    use pgai_mon::store::StoreCore;
    let d = tempfile::tempdir().unwrap();
    let core = StoreCore::open(d.path(), &StoreCfg::default()).unwrap();
    let r = Reader::new(
        core.shared.clone(),
        HttpCfg::default(),
        EMBEDDED_VIEWS_SQL.into(),
        vec![],
    )
    .unwrap();
    for (sql, want) in [
        ("SELECT 1 AS x", true),
        ("SELECT sum(i) AS s FROM range(10) t(i)", true),
        ("SELECT count(*) AS n FROM samples", true),
        ("SELECT 'now()' AS s /* random() */", true),
        ("SELECT now() AS t", false),
        (r#"SELECT "random"() AS r"#, false),
        (r#"SELECT "now"() AS t"#, false),
        (r#"SELECT E'\'' AS a, now() AS t"#, false),
        ("SELECT $a$ $$ $a$ AS s, random() AS r", false),
        ("SELECT uuidv7() AS u", false),
        ("SELECT uuidv4() AS u", false),
        ("SELECT current_localtimestamp() AS t", false),
        ("SELECT current_localtime() AS t", false),
        ("SELECT CURRENT_TIMESTAMP AS t", false),
        ("SELECT * FROM duckdb_memory()", false),
        // built-in macros over clock functions, one-argument age, samples
        ("SELECT ago(INTERVAL 1 HOUR) AS t", false),
        (
            "SELECT count(*) AS n FROM samples WHERE ts > ago(INTERVAL 1 HOUR)",
            false,
        ),
        ("SELECT pg_postmaster_start_time() AS t", false),
        ("SELECT age(TIMESTAMP '2026-11-10 08:00:00') AS a", false),
        ("SELECT count(*) AS n FROM range(1000) t(i) USING SAMPLE 10%", false),
        ("SELECT version() AS v", true),
    ] {
        match r.run(sql) {
            Ok(x) => assert_eq!(x.cacheable, want, "{sql}"),
            // a reader without some function: only acceptable for non-cacheable cases
            Err(e) => assert!(!want, "{sql}: {e:?}"),
        }
    }
}

/// A finished flight leaves nothing behind: no Flight, channel or reply, and no
/// extra reference to the cache.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finished_flights_release_their_reply() {
    let c = cache();
    let body = Arc::new(());
    for i in 0..50 {
        let b = body.clone();
        let (r, _) = c
            .get_or_run(&format!("SELECT {i}"), 0, 1, move || {
                let _keep = b.clone();
                async move { Ok(reply("x", 1)) }
            })
            .await;
        drop(r);
    }
    // non-cacheable and error replies too
    let (_, _) = c
        .get_or_run("SELECT now()", 0, 1, || async {
            Ok(Reply {
                body: "t".into(),
                gen: 1,
                cacheable: false,
                range: None,
                dict_version: 0,
            })
        })
        .await;
    let (_, _) = c
        .get_or_run("SELECT err", 0, 1, || async {
            Err(QueryError {
                kind: "binder_error",
                message: "x".into(),
            })
        })
        .await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(Arc::strong_count(&c), 1, "flights hold no reference to the cache");
    assert_eq!(Arc::strong_count(&body), 1, "run closures are released");
    assert_eq!(c.stats_json()["in_flight"], 0);
}

#[tokio::test]
async fn a_reply_from_a_new_dictionary_is_not_stored_under_the_old_version() {
    use pgai_mon::qcache::Snapshot;
    let c = cache();
    let mut old: Snapshot = 1.into();
    old.dict_version = 10;
    let mut new = old.clone();
    new.dict_version = 20;
    let (_, served) = c
        .get_or_run("SELECT interval_s", 0, old.clone(), || async {
            Ok(Reply {
                dict_version: 20,
                ..reply("new", 1)
            })
        })
        .await;
    assert_eq!(served, Served::Miss);
    let (_, served) = c
        .get_or_run("SELECT interval_s", 0, old, || async {
            Ok(Reply {
                dict_version: 10,
                ..reply("old", 1)
            })
        })
        .await;
    assert_eq!(served, Served::Miss);
    let (r, served) = c
        .get_or_run("SELECT interval_s", 0, new, || async {
            Ok(Reply {
                dict_version: 20,
                ..reply("new", 1)
            })
        })
        .await;
    assert_eq!(served, Served::Miss);
    assert_eq!(r.unwrap().body, "new");
}
