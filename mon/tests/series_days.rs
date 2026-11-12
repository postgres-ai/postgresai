//! Live ingest path (collector Emitter -> WAL -> flush -> compaction -> registry) across a day
//! boundary: every (series, day) with samples has a row in that day's series files, at every
//! stage (open day L0, open-day L1 merge, closed-day compaction), and the `series` view
//! (registry + tail) holds each series exactly once.
use pgai_mon::collector::Emitter;
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::model::*;
use pgai_mon::query::{Reader, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::StoreCore;
use pgai_mon::Msg;

fn one(r: &Reader, sql: &str) -> i64 {
    let q = r.run(sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    let v: (i64,) = serde_json::from_str(q.rows[0].get()).unwrap();
    v.0
}

fn check(r: &Reader, stage: &str, expect_series: i64) {
    let missing = one(
        r,
        "SELECT count(*) FROM (SELECT DISTINCT series_id, ts::DATE AS d FROM samples) s \
         WHERE NOT EXISTS (SELECT 1 FROM _raw_series x WHERE x.series_id = s.series_id AND x.first_seen::DATE = s.d)",
    );
    assert_eq!(
        missing, 0,
        "{stage}: (series, day) pairs with samples but no series row that day"
    );
    assert_eq!(
        one(r, "SELECT count(*) FROM series"),
        expect_series,
        "{stage}: series view"
    );
    assert_eq!(
        one(r, "SELECT count(DISTINCT series_id) FROM series"),
        expect_series,
        "{stage}: series view has duplicates"
    );
}

#[test]
fn per_day_series_rows_across_midnight_flush_and_compaction() {
    let d = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(d.path(), &StoreCfg::default()).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Msg>(1 << 16);
    let mut em = Emitter {
        shared: core.shared.clone(),
        tx,
        cfg: Default::default(),
        cut: Default::default(),
        errors: Default::default(),
        seen_text: Default::default(),
        prev: Default::default(),
    };
    let r = Reader::new(
        core.shared.clone(),
        HttpCfg::default(),
        EMBEDDED_VIEWS_SQL.into(),
        vec![],
    )
    .unwrap();
    let def = |q: u64| {
        let labels = vec![("queryid".to_string(), q.to_string())];
        (series_id("m_total", &labels), "m_total".to_string(), labels)
    };
    // day 0 22:00 .. day 1 02:00, one scrape every 10 min. Series 0-2 always; 10+k only in
    // day 0 (k < 3); 20+k appear after midnight (churn).
    let day1 = 1_794_009_600_000i64; // 2026-11-11 00:00 UTC
    let mut ts = day1 - 2 * 3_600_000;
    let mut k = 0;
    while ts <= day1 + 2 * 3_600_000 {
        let mut qs: Vec<u64> = vec![0, 1, 2];
        if ts < day1 {
            qs.extend(10..13);
        } else {
            qs.push(20 + (k % 5));
        }
        let defs: Vec<_> = qs.iter().map(|q| def(*q)).collect();
        let vals = defs.iter().map(|(id, _, _)| (*id, k as f64)).collect();
        rt.block_on(em.emit(defs, ts, vals));
        while let Ok(m) = rx.try_recv() {
            if let Msg::Rec(rec) = m {
                core.append(rec).unwrap();
            }
        }
        if k % 4 == 3 {
            core.flush().unwrap(); // several L0 files per day, a flush straddling midnight
        }
        ts += 600_000;
        k += 1;
    }
    core.flush().unwrap();
    let total = 3 + 3 + 5;
    check(&r, "L0 only (open day 1)", total);

    // open-day merge of finished hours on day 1, day 0 closed (grace passed) -> registered
    let comp = core.compactor().unwrap();
    comp.run_once(day1 + 3 * 3_600_000, false).unwrap();
    check(&r, "day 0 compacted + registered, day 1 L1", total);
    let reg = one(&r, "SELECT count(*) FROM _raw_series_reg");
    assert_eq!(reg, 6, "day 0's six series are in the registry");

    // a late series row for day 0 after registration (late scrape) must still show up once
    let late = def(99);
    rt.block_on(em.emit(vec![late.clone()], day1 - 60_000, vec![(late.0, 1.0)]));
    while let Ok(Msg::Rec(rec)) = rx.try_recv() {
        core.append(rec).unwrap();
    }
    core.flush().unwrap();
    check(&r, "late day-0 file (tail)", total + 1);

    // everything closed: all series files registered, the tail is empty
    comp.run_once(day1 + 30 * 3_600_000, true).unwrap();
    check(&r, "all closed", total + 1);
    assert_eq!(one(&r, "SELECT count(*) FROM _raw_series_reg"), total + 1);
    assert_eq!(one(&r, "SELECT count(*) FROM _raw_series_tail"), 0);
}

/// Monthly series files: a window covering whole closed months selects from them (one row per
/// series and month) instead of the day files; results equal the oracle's, which reads `series`.
#[test]
fn monthly_series_files_serve_long_windows() {
    let d = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(d.path(), &StoreCfg::default()).unwrap();
    let day0 = 20_330i64; // 2025-08-30 (days since epoch): spans Aug, Sep, Oct
    let (mut defs, mut samples) = (vec![], vec![]);
    for q in 0..60u64 {
        let labels = vec![("queryid".to_string(), q.to_string())];
        let id = series_id("m_total", &labels);
        // q < 20 live all 60 days; others live 3 days starting on day q
        let (from, to) = if q < 20 { (0, 60) } else { (q as i64, q as i64 + 3) };
        for day in from..to {
            let t0 = (day0 + day) * 86_400_000;
            defs.push(SeriesDef {
                series_id: id,
                metric: "m_total".into(),
                labels: labels.clone(),
                first_seen: t0,
            });
            for h in 0..24 {
                let t = t0 + h * 3_600_000 + (q as i64) * 1000;
                samples.push(Sample {
                    series_id: id,
                    ts: t,
                    value: ((day * 24 + h) * (q as i64 + 1)) as f64,
                });
            }
        }
    }
    core.append(Record::Series(defs)).unwrap();
    core.append(Record::Samples(samples)).unwrap();
    core.flush().unwrap();
    core.compactor().unwrap().run_once(now_ms(), true).unwrap();
    let months = core.shared.catalog.lock().unwrap().files_of("series_month").count();
    assert_eq!(months, 3, "Aug, Sep, Oct");
    let views = format!("{EMBEDDED_VIEWS_SQL}\n;\n{}", include_str!("../sql/macros.sql"));
    let r = Reader::new(
        core.shared.clone(),
        HttpCfg {
            memory_limit: "1GB".into(),
            ..Default::default()
        },
        views,
        vec![],
    )
    .unwrap();
    let end = (day0 + 59) * 86_400_000;
    for (start, step, f) in [
        ((day0 + 3) * 86_400_000, 86_400_000i64, "increase"), // covers all of September
        ((day0 + 2) * 86_400_000, 21_600_000, "irate"),
    ] {
        for q in [
            format!("SELECT series_id, t, value FROM prom_{f}('m_total', [], {start}, {end}, {step}, {step}) ORDER BY 1, 2"),
            format!("SELECT series_id, t, value FROM prom_topk(5, '{f}', 'm_total', [], {start}, {end}, {step}, {step}) ORDER BY 2, 3, 1"),
        ] {
            let a = r.run(&q).unwrap();
            let b = r.run(&q.replace(&format!("prom_{f}("), &format!("prom_{f}_sql(")).replace("prom_topk(", "prom_topk_sql(")).unwrap();
            assert_eq!(a.row_count, b.row_count, "{q}");
            assert!(a.row_count > 0, "{q}");
            for (x, y) in a.rows.iter().zip(b.rows.iter()) {
                let x: (u64, i64, f64) = serde_json::from_str(x.get()).unwrap();
                let y: (u64, i64, f64) = serde_json::from_str(y.get()).unwrap();
                assert_eq!((x.0, x.1), (y.0, y.1), "{q}");
                assert!((x.2 - y.2).abs() <= 1e-9 * y.2.abs().max(1e-12), "{q}");
            }
        }
    }
}
