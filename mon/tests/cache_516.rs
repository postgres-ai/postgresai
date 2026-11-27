//! #516: cached answers survive catalog changes that cannot change them (a merge of the files
//! they read, a text snapshot that only refreshes last_seen) and still go when an answer can
//! change. Every check compares the cached bytes with a fresh execution.
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::dictionary::Dictionary;
use pgai_mon::model::{now_ms, series_id, QueryText, Record, Sample, SeriesDef, SourceId, DAY_MS};
use pgai_mon::qcache::{QueryCache, Served};
use pgai_mon::query::{Backend, DictEntry, Reader, Reply, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::StoreCore;
use std::sync::Arc;

const M: &str = "pgwatch_slow_v";
const OTHER: &str = "pgwatch_other_v";

/// A day ahead of the wall clock, so the compactor treats it as open whatever the time of day.
fn day0() -> i64 {
    (now_ms().div_euclid(DAY_MS) + 2) * DAY_MS
}

fn dict() -> Dictionary {
    Dictionary::new(vec![DictEntry {
        group: "slow".into(),
        prefix: "pgwatch_".into(),
        interval_s: Some(60),
        instance_level: false,
        gauges: vec![],
        description: String::new(),
    }])
}

fn series(core: &mut StoreCore, metric: &str, first: i64) {
    core.append(Record::Series(vec![SeriesDef {
        series_id: series_id(metric, &[]),
        metric: metric.into(),
        labels: vec![],
        first_seen: first,
    }]))
    .unwrap();
}

fn sample(core: &mut StoreCore, metric: &str, ts: i64, value: f64) {
    core.append(Record::Samples(vec![Sample {
        series_id: series_id(metric, &[]),
        ts,
        value,
    }]))
    .unwrap();
}

fn text(core: &mut StoreCore, queryid: i64, hash: u64, query: Option<&str>, at: i64) {
    core.append(Record::QueryTexts(vec![QueryText {
        queryid,
        dbname: "db".into(),
        query_hash: hash,
        query: query.map(String::from),
        first_seen: at,
        last_seen: at,
        src: SourceId::default(),
    }]))
    .unwrap();
}

/// A store with `M` sampled every minute of the day's first two hours, one flush per 10 minutes
/// (12 level-0 files), and one statement text.
fn store(dir: &std::path::Path) -> StoreCore {
    let mut core = StoreCore::open(dir, &StoreCfg::default()).unwrap();
    let d = day0();
    series(&mut core, M, d);
    series(&mut core, OTHER, d);
    text(&mut core, 42, 1, Some("SELECT   42"), d);
    for i in 0..120 {
        sample(&mut core, M, d + i * 60_000, i as f64);
        if i % 10 == 9 {
            core.flush().unwrap();
        }
    }
    core.flush().unwrap();
    core
}

fn reader(core: &StoreCore) -> Backend {
    Backend::InProc(Arc::new(
        Reader::new(
            core.shared.clone(),
            HttpCfg::default(),
            EMBEDDED_VIEWS_SQL.into(),
            dict().entries,
        )
        .unwrap(),
    ))
}

fn cache() -> Arc<QueryCache> {
    Arc::new(QueryCache::new(1 << 20, 1 << 20, EMBEDDED_VIEWS_SQL))
}

async fn cached(c: &Arc<QueryCache>, b: &Backend, core: &StoreCore, sql: &str) -> (Arc<Reply>, Served) {
    let snap = c.snapshot(&core.shared.catalog.lock().unwrap(), b.dictionary_version().unwrap());
    let (b, sql_owned) = (b.clone(), sql.to_string());
    let (r, how) = c
        .get_or_run(sql, 3, snap, move || {
            let (b, sql) = (b.clone(), sql_owned.clone());
            async move { b.call("sql_csv", sql).await }
        })
        .await;
    (r.unwrap(), how)
}

async fn check(c: &Arc<QueryCache>, b: &Backend, core: &StoreCore, sql: &str, want: Served) -> String {
    let (r, how) = cached(c, b, core, sql).await;
    assert_eq!(how, want, "{sql}");
    let fresh = b.call("sql_csv", sql.into()).await.unwrap();
    assert_eq!(r.body, fresh.body, "cached bytes must equal a fresh execution");
    r.body.clone()
}

/// The window [d + 30 min, d + 90 min] of `M`, every 5 minutes.
fn window() -> String {
    let d = day0();
    format!(
        "SELECT t, value FROM prom_last('{M}', [], {}, {}, 300000) ORDER BY t",
        d + 1_800_000,
        d + 5_400_000
    )
}

/// The same window joined to a statement text through `view`.
fn with_text(view: &str) -> String {
    let col = if view == "query_texts" {
        "max(last_seen)"
    } else {
        "max(query)"
    };
    format!(
        "SELECT w.*, (SELECT {col} FROM {view} WHERE queryid = 42) txt FROM ({}) w",
        window()
    )
}

fn inv(c: &QueryCache, reason: &str) -> u64 {
    c.stats_json()["invalidations"][reason].as_u64().unwrap()
}

fn kinds(core: &StoreCore, kind: &str) -> Vec<(u8, usize)> {
    let cat = core.shared.catalog.lock().unwrap();
    cat.files
        .iter()
        .filter(|f| f.kind == kind)
        .map(|f| (f.level, f.replaces.len()))
        .collect()
}

/// Merge the open day's level-0 files (their hours have passed for this `now`).
fn merge_hour(core: &StoreCore) -> usize {
    core.compactor()
        .unwrap()
        .run_once(day0() + 12 * 3_600_000, false)
        .unwrap()
}

#[tokio::test]
async fn an_hourly_merge_of_the_files_an_answer_read_keeps_it() {
    let dir = tempfile::tempdir().unwrap();
    let core = store(dir.path());
    let (b, c) = (reader(&core), cache());
    let sqls = [window(), with_text("query_text_names"), with_text("query_texts")];
    for q in &sqls {
        check(&c, &b, &core, q, Served::Miss).await;
    }
    assert!(kinds(&core, "samples").iter().filter(|k| k.0 == 0).count() > 1);
    assert!(merge_hour(&core) > 0);
    // the samples merged into one level-1 file that names its 12 inputs (one per flush)
    let merged = kinds(&core, "samples");
    assert_eq!(merged, vec![(1, 12)], "{merged:?}");
    for q in &sqls {
        check(&c, &b, &core, q, Served::Hit).await;
    }
    let s = c.stats_json();
    assert!(s["kept_replaced"].as_u64().unwrap() >= 3, "{s}");
    for reason in [
        "added_samples",
        "added_series",
        "added_query_texts",
        "added_other",
        "removed_files",
    ] {
        assert_eq!(inv(&c, reason), 0, "{reason}: {s}");
    }
}

#[tokio::test]
async fn a_merge_with_an_input_the_old_snapshot_did_not_have_invalidates() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = store(dir.path());
    let (b, c) = (reader(&core), cache());
    let q = window();
    check(&c, &b, &core, &q, Served::Miss).await;
    // a file the cache never saw (another metric, later), merged with the ones it read
    sample(&mut core, OTHER, day0() + 7_000_000, 1.0);
    core.flush().unwrap();
    assert!(merge_hour(&core) > 0);
    check(&c, &b, &core, &q, Served::Miss).await;
    assert_eq!(inv(&c, "added_samples"), 1);
}

#[tokio::test]
async fn a_refresh_only_text_snapshot_keeps_answers_that_cannot_see_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = store(dir.path());
    let (b, c) = (reader(&core), cache());
    let (none, names, all) = (window(), with_text("query_text_names"), with_text("query_texts"));
    for q in [&none, &names, &all] {
        check(&c, &b, &core, q, Served::Miss).await;
    }
    text(&mut core, 42, 1, None, day0() + 3_600_000);
    core.flush().unwrap();
    assert!(core
        .shared
        .catalog
        .lock()
        .unwrap()
        .files
        .iter()
        .any(|f| f.kind == "query_texts" && f.texts_refresh_only));
    check(&c, &b, &core, &none, Served::Hit).await;
    assert!(check(&c, &b, &core, &names, Served::Hit).await.contains("SELECT   42"));
    // query_texts exposes last_seen: the refresh changes that answer
    check(&c, &b, &core, &all, Served::Miss).await;
    assert_eq!(inv(&c, "added_query_texts"), 1);
}

#[tokio::test]
async fn a_new_text_reruns_text_readers_only() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = store(dir.path());
    let (b, c) = (reader(&core), cache());
    let (none, names) = (window(), with_text("query_text_names"));
    check(&c, &b, &core, &none, Served::Miss).await;
    assert!(check(&c, &b, &core, &names, Served::Miss).await.contains("SELECT   42"));
    // the statement's text changed (a new hash), in one delta with a refresh of the old one
    text(&mut core, 42, 1, None, day0() + 3_600_000);
    text(&mut core, 42, 2, Some("SELECT 42 /* v2 */"), day0() + 3_600_000);
    core.flush().unwrap();
    check(&c, &b, &core, &none, Served::Hit).await;
    assert!(check(&c, &b, &core, &names, Served::Miss).await.contains("v2"));
}

#[tokio::test]
async fn a_merge_of_text_files_keeps_text_readers_and_their_removal_does_not() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = store(dir.path());
    text(&mut core, 42, 1, None, day0() + 3_600_000);
    core.flush().unwrap();
    let (b, c) = (reader(&core), cache());
    let (none, names, all) = (window(), with_text("query_text_names"), with_text("query_texts"));
    for q in [&none, &names, &all] {
        check(&c, &b, &core, q, Served::Miss).await;
    }
    assert!(merge_hour(&core) > 0);
    assert!(kinds(&core, "query_texts").iter().all(|k| k.0 == 1 && k.1 >= 2));
    for q in [&none, &names, &all] {
        check(&c, &b, &core, q, Served::Hit).await;
    }
    // texts retention: the files go without a replacement
    {
        let mut cat = core.shared.catalog.lock().unwrap();
        cat.files.retain(|f| f.kind != "query_texts");
        cat.generation += 1;
        cat.save(dir.path()).unwrap();
        core.shared
            .generation
            .store(cat.generation, std::sync::atomic::Ordering::Release);
    }
    check(&c, &b, &core, &none, Served::Hit).await;
    for q in [&names, &all] {
        assert!(!check(&c, &b, &core, q, Served::Miss).await.contains("42"));
    }
    assert_eq!(inv(&c, "removed_files"), 2);
}

#[tokio::test]
async fn an_older_catalog_without_replaces_keeps_the_conservative_rule() {
    let dir = tempfile::tempdir().unwrap();
    let core = store(dir.path());
    let (b, c) = (reader(&core), cache());
    let q = window();
    check(&c, &b, &core, &q, Served::Miss).await;
    assert!(merge_hour(&core) > 0);
    // as written by a binary before #516: the same merge without `replaces`
    {
        let mut cat = core.shared.catalog.lock().unwrap();
        for f in cat.files.iter_mut() {
            f.replaces.clear();
        }
        cat.generation += 1;
        cat.save(dir.path()).unwrap();
        core.shared
            .generation
            .store(cat.generation, std::sync::atomic::Ordering::Release);
    }
    check(&c, &b, &core, &q, Served::Miss).await;
    assert_eq!(c.stats_json()["kept_replaced"], 0);
}

#[tokio::test]
async fn equal_timestamps_give_one_answer_whatever_the_file_layout() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    let d = day0();
    series(&mut core, M, d);
    for i in 0..10 {
        sample(&mut core, M, d + i * 60_000, i as f64);
    }
    core.flush().unwrap();
    // the same timestamp twice, the larger value in the earlier file
    sample(&mut core, M, d + 10 * 60_000, 20.0);
    core.flush().unwrap();
    sample(&mut core, M, d + 10 * 60_000, 10.0);
    core.flush().unwrap();
    let b = reader(&core);
    let qs = [
        format!(
            "SELECT t, value FROM prom_last('{M}', [], {0}, {0}, 60000)",
            d + 10 * 60_000
        ),
        format!(
            "SELECT t, value FROM prom_irate('{M}', [], {0}, {0}, 60000, 180000)",
            d + 11 * 60_000
        ),
        format!(
            "SELECT t, value FROM prom_max_over_time('{M}', [], {0}, {0}, 60000, 600000)",
            d + 10 * 60_000
        ),
    ];
    let mut before = vec![];
    for q in &qs {
        before.push(b.call("sql_csv", q.clone()).await.unwrap().body);
    }
    // canonical order: the later of two equal-time samples is the larger value
    assert!(before[0].ends_with(",20.0\r\n"), "{}", before[0]);
    assert!(core.compactor().unwrap().run_once(d + 12 * 3_600_000, false).unwrap() > 0);
    for (q, want) in qs.iter().zip(&before) {
        assert_eq!(&b.call("sql_csv", q.clone()).await.unwrap().body, want, "{q}");
    }
}

#[tokio::test]
async fn a_macro_over_the_text_views_counts_as_reading_every_text() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = store(dir.path());
    let views = format!(
        "{EMBEDDED_VIEWS_SQL}\nCREATE MACRO txt42() AS (SELECT max(query) FROM query_text_names WHERE queryid = 42);"
    );
    let b = Backend::InProc(Arc::new(
        Reader::new(core.shared.clone(), HttpCfg::default(), views, dict().entries).unwrap(),
    ));
    let c = cache();
    let q = format!("SELECT w.*, txt42() txt FROM ({}) w", window());
    check(&c, &b, &core, &q, Served::Miss).await;
    // bounded (a sample flush after the window keeps it) ...
    sample(&mut core, M, day0() + 20_000_000, 1.0);
    core.flush().unwrap();
    check(&c, &b, &core, &q, Served::Hit).await;
    // ... but a refresh-only text file drops it: macros are not proven to read names only
    text(&mut core, 42, 1, None, day0() + 3_600_000);
    core.flush().unwrap();
    check(&c, &b, &core, &q, Served::Miss).await;
}

#[tokio::test]
async fn a_shadowed_arg_max_leaves_the_text_views_unbounded() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = store(dir.path());
    let views = format!("CREATE MACRO arg_max(a, b) AS (SELECT count(*) FROM _raw_samples);\n{EMBEDDED_VIEWS_SQL}");
    let b = Backend::InProc(Arc::new(
        Reader::new(core.shared.clone(), HttpCfg::default(), views, dict().entries).unwrap(),
    ));
    let c = cache();
    let q = with_text("query_text_names");
    check(&c, &b, &core, &q, Served::Miss).await;
    sample(&mut core, M, day0() + 20_000_000, 1.0);
    core.flush().unwrap();
    check(&c, &b, &core, &q, Served::Miss).await;
    assert_eq!(inv(&c, "unbounded"), 1);
}

#[test]
fn a_reply_without_the_text_key_reads_every_text() {
    // an older reader child: no "rt" after the range
    let old = r#"{"id":7,"gen":3,"c":1,"dv":0,"r":[1,2],"rp":[5],"rc":0,"ok":"x"}"#.to_string();
    let (_, _, r) = pgai_mon::sqlproc::parse_reply_s3(old).unwrap();
    assert_eq!(r.unwrap().range.unwrap().texts, pgai_mon::readset::TEXTS_ALL);
    let new = r#"{"id":7,"gen":3,"c":1,"dv":0,"r":[1,2],"rp":[5],"rt":1,"rc":0,"ok":"x"}"#.to_string();
    let (_, _, r) = pgai_mon::sqlproc::parse_reply_s3(new).unwrap();
    let range = r.unwrap().range.unwrap();
    assert_eq!(
        (range.texts, range.prefixes),
        (pgai_mon::readset::TEXTS_NAMES, Some(vec![5]))
    );
}

#[test]
fn flushed_text_files_say_whether_they_only_refresh() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    let d = day0();
    text(&mut core, 1, 1, None, d);
    text(&mut core, 2, 2, None, d);
    core.flush().unwrap();
    text(&mut core, 1, 1, None, d + 1);
    text(&mut core, 3, 3, Some("SELECT 3"), d + 1);
    core.flush().unwrap();
    let cat = core.shared.catalog.lock().unwrap();
    let flags: Vec<bool> = cat
        .files
        .iter()
        .filter(|f| f.kind == "query_texts")
        .map(|f| f.texts_refresh_only)
        .collect();
    assert_eq!(flags, vec![true, false]);
}

#[test]
fn the_current_hours_flush_files_merge_once_there_are_enough() {
    for (setting, files, merged) in [(8, 8, true), (8, 7, false), (0, 12, false)] {
        // stay inside one wall-clock hour (the run below is 6 s ahead): else the hourly
        // level-1 merge takes every file
        let to_hour = 3_600_000 - now_ms().rem_euclid(3_600_000);
        if to_hour < 15_000 {
            std::thread::sleep(std::time::Duration::from_millis(to_hour as u64 + 100));
        }
        let dir = tempfile::tempdir().unwrap();
        let cfg = StoreCfg {
            open_merge_files: setting,
            ..Default::default()
        };
        let mut core = StoreCore::open(dir.path(), &cfg).unwrap();
        let now = now_ms();
        series(&mut core, M, now);
        for i in 0..files {
            sample(&mut core, M, now + i as i64, i as f64);
            core.flush().unwrap();
        }
        // the newest flush files are too young for this pass; 6 s later they are taken
        core.compactor().unwrap().run_once(now_ms() + 6_000, false).unwrap();
        let k = kinds(&core, "samples");
        if merged {
            assert_eq!(k.len(), 1, "{k:?}");
            assert_eq!(k[0].1, files, "{k:?}");
        } else {
            assert_eq!(k.len(), files, "{setting} {files}: {k:?}");
        }
    }
}

#[tokio::test]
async fn following_every_generation_keeps_entries_across_a_merge_of_later_flushes() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = store(dir.path());
    let (b, c) = (reader(&core), cache());
    let q = window();
    check(&c, &b, &core, &q, Served::Miss).await;
    // live: a flush after the request (unrelated here: another metric, later), seen by the
    // server's catalog follower, then the merge of it with the files the entry read
    sample(&mut core, OTHER, day0() + 7_000_000, 1.0);
    core.flush().unwrap();
    follow(&c, &b, &core);
    assert!(merge_hour(&core) > 0);
    follow(&c, &b, &core);
    check(&c, &b, &core, &q, Served::Hit).await;
    assert!(c.stats_json()["kept_replaced"].as_u64().unwrap() > 0);
    // a related flush inside the window, followed the same way, still drops it
    sample(&mut core, M, day0() + 3_000_000, 5.0);
    core.flush().unwrap();
    follow(&c, &b, &core);
    check(&c, &b, &core, &q, Served::Miss).await;
}

fn follow(c: &Arc<QueryCache>, b: &Backend, core: &StoreCore) {
    c.observe(c.snapshot(&core.shared.catalog.lock().unwrap(), b.dictionary_version().unwrap()));
}

#[tokio::test]
async fn an_overridden_query_text_names_is_not_bounded() {
    for custom in [
        "CREATE OR REPLACE VIEW query_text_names AS SELECT 42::BIGINT queryid, 'db' dbname, count(*)::VARCHAR query FROM samples;",
        "CREATE TEMP VIEW query_text_names AS SELECT 42::BIGINT queryid, 'db' dbname, count(*)::VARCHAR query FROM samples;",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut core = store(dir.path());
        let views = format!("{EMBEDDED_VIEWS_SQL}\n{custom}");
        let b = Backend::InProc(Arc::new(Reader::new(core.shared.clone(), HttpCfg::default(), views, dict().entries).unwrap()));
        let c = cache();
        let q = with_text("query_text_names");
        check(&c, &b, &core, &q, Served::Miss).await;
        // an unrelated flush after the window: a bounded entry would survive it
        sample(&mut core, M, day0() + 20_000_000, 1.0);
        core.flush().unwrap();
        check(&c, &b, &core, &q, Served::Miss).await;
        assert_eq!(inv(&c, "unbounded"), 1, "{custom}");
    }
}

#[tokio::test]
async fn negative_values_and_markers_at_one_timestamp_keep_their_order_across_a_merge() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    let d = day0();
    series(&mut core, M, d);
    for i in 0..10 {
        sample(&mut core, M, d + i * 60_000, -(i as f64));
    }
    core.flush().unwrap();
    // at one timestamp, across three files: a staleness marker, -1.0 and -2.0
    let t = d + 10 * 60_000;
    for v in [pgai_mon::model::stale_nan(), -1.0, -2.0] {
        sample(&mut core, M, t, v);
        core.flush().unwrap();
    }
    let b = reader(&core);
    let qs = [
        format!("SELECT t, value FROM prom_last('{M}', [], {t}, {t}, 60000)"),
        format!("SELECT t, value FROM prom_min_over_time('{M}', [], {t}, {t}, 60000, 30000)"),
        format!(
            "SELECT t, value FROM prom_irate('{M}', [], {0}, {0}, 60000, 180000)",
            t + 60_000
        ),
    ];
    let mut before = vec![];
    for q in &qs {
        before.push(b.call("sql_csv", q.clone()).await.unwrap().body);
    }
    // real values first, the marker last: the series ends at t (no last value), and the
    // minimum over (t - 30 s, t] sees both real values
    assert_eq!(before[0], "t,value\r\n");
    assert!(before[1].ends_with(",-2.0\r\n"), "{}", before[1]);
    assert!(core.compactor().unwrap().run_once(d + 12 * 3_600_000, false).unwrap() > 0);
    for (q, want) in qs.iter().zip(&before) {
        assert_eq!(&b.call("sql_csv", q.clone()).await.unwrap().body, want, "{q}");
    }
}

#[test]
fn the_open_hour_merge_waits_for_flush_files_younger_than_five_seconds() {
    let dir = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(dir.path(), &StoreCfg::default()).unwrap();
    let now = now_ms();
    series(&mut core, M, now);
    for i in 0..8 {
        sample(&mut core, M, now + i, i as f64);
        core.flush().unwrap();
    }
    // at once: every file is younger than 5 s, nothing merges
    core.compactor().unwrap().run_once(now_ms(), false).unwrap();
    assert_eq!(kinds(&core, "samples").len(), 8);
}
