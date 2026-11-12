//! The 1 h tier in month files (`rollup_1h_month`): built by compaction next to the 1 d tier,
//! extended from itself, rebuilt when a day it covers is recompacted, and served in place of the
//! per-day 1 h files with bit-identical results.
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::eval::kernels::{Func, Grid};
use pgai_mon::eval::rollup::{self, KIND, KIND_1D, KIND_1HM, SERVED_1HM};
use pgai_mon::model::*;
use pgai_mon::query::{Reader, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::StoreCore;
use std::sync::atomic::Ordering;

const MACROS: &str = include_str!("../sql/macros.sql");

fn rows(r: &Reader, sql: &str) -> Vec<String> {
    let res = r.run(sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    res.rows.iter().map(|x| x.get().to_string()).collect()
}

fn months(core: &StoreCore, kind: &str) -> Vec<pgai_mon::catalog::FileEntry> {
    let mut v: Vec<_> = core.shared.catalog.lock().unwrap().files_of(kind).cloned().collect();
    v.sort_by_key(|f| f.min_ts);
    v
}

/// Every hour-aligned call shape the dashboards use, on the month tier and on the day files
/// (PGAI_NO_ROLLUP_1HM): the same rows, bit for bit. Returns (rows, calls served from months).
fn check(r: &Reader, cases: &[(i64, i64, i64)]) -> (usize, u64) {
    let h = 3_600_000i64;
    let mut calls = vec![];
    for &(s, e, step) in cases {
        for f in ["increase", "rate", "delta", "increase_pure", "last_over_time", "first_over_time"] {
            for w in [step, 6 * h] {
                calls.push(format!("prom_{f}('d_total', [], {s}, {e}, {step}, {w})"));
            }
        }
        for f in ["irate", "idelta"] {
            calls.push(format!("prom_{f}('d_total', [], {s}, {e}, {step}, {})", step + 5_000));
        }
        for f in ["last_over_time", "tlast_over_time"] {
            calls.push(format!("prom_{f}('d_total', [], {s}, {e}, {step}, 600000)"));
        }
        for m in ["d_total", "e_total"] {
            for f in ["last", "timestamp"] {
                calls.push(format!("prom_{f}('{m}', [], {s}, {e}, {step})"));
            }
        }
        for f in ["sum", "min", "max", "avg", "count"] {
            calls.push(format!("prom_{f}_over_time('d_total', [], {s}, {e}, {step}, {step})"));
        }
    }
    let (mut n, before) = (0, SERVED_1HM.load(Ordering::Relaxed));
    for c in &calls {
        let q = format!("SELECT series_id, t, value FROM {c} ORDER BY 1, 2");
        let a = rows(r, &q);
        std::env::set_var("PGAI_NO_ROLLUP_1HM", "1");
        let b = rows(r, &q);
        std::env::remove_var("PGAI_NO_ROLLUP_1HM");
        assert_eq!(a.len(), b.len(), "{c}");
        assert_eq!(a, b, "{c}: month tier vs day files");
        n += a.len();
    }
    (n, SERVED_1HM.load(Ordering::Relaxed) - before)
}

#[test]
fn rollup_1h_month_tier_matches_the_day_files() {
    let d = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(
        d.path(),
        &StoreCfg {
            delete_grace_s: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let day = DAY_MS;
    let h = 3_600_000i64;
    let day0 = pgai_mon::store::chrono_free::days_from_civil(2026, 9, 20) * day;
    let ndays = 40;
    let mut defs = vec![];
    let mut samples = vec![];
    let mut seed = 31u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for s in 0..6u64 {
        let labels = vec![("i".to_string(), s.to_string())];
        let id = series_id("d_total", &labels);
        defs.push((id, "d_total", labels));
        let aligned = s % 3 == 0; // samples exactly on every hour and midnight
        let mut t = day0 + if aligned { 0 } else { (rnd() % 300_000) as i64 };
        let mut v = (rnd() % 100) as f64;
        let mut k = 0;
        while t < day0 + ndays * day {
            v += (rnd() % 20) as f64 + 0.25;
            if k % 1700 == 900 {
                v = 1.5; // reset
            }
            if s == 4 && k == 3000 {
                t += day + 7_200_000; // a gap over a midnight
            }
            let value = if k % 333 == 50 {
                stale_nan()
            } else if k % 517 == 60 {
                f64::NAN
            } else {
                v
            };
            samples.push(Sample { series_id: id, ts: t, value });
            t += if aligned { 300_000 } else { 300_000 + (rnd() % 120_000) as i64 - 60_000 };
            k += 1;
        }
    }
    // sparse, with markers: bare selectors look back further than a step
    for s in 0..3u64 {
        let labels = vec![("i".to_string(), s.to_string())];
        let id = series_id("e_total", &labels);
        defs.push((id, "e_total", labels));
        let mut t = day0 + (rnd() % 3_600_000) as i64;
        let mut k = 0;
        while t < day0 + ndays * day {
            let value = if (k + s) % 4 == 1 { stale_nan() } else { (rnd() % 1000) as f64 };
            samples.push(Sample { series_id: id, ts: t, value });
            t += 4_000_000 + (rnd() % 5_000_000) as i64;
            k += 1;
        }
    }
    let mut per_day: std::collections::BTreeMap<(u64, i64), i64> = Default::default();
    for x in &samples {
        let e = per_day.entry((x.series_id, day_of(x.ts))).or_insert(x.ts);
        *e = (*e).min(x.ts);
    }
    let meta: std::collections::HashMap<u64, (&str, Vec<(String, String)>)> =
        defs.into_iter().map(|(id, m, l)| (id, (m, l))).collect();
    let series = |lo: i64, hi: i64| -> Vec<SeriesDef> {
        per_day
            .iter()
            .filter(|((_, d), _)| *d >= lo && *d < hi)
            .map(|((id, _), first)| SeriesDef {
                series_id: *id,
                metric: meta[id].0.into(),
                labels: meta[id].1.clone(),
                first_seen: *first,
            })
            .collect()
    };
    // phase 1: days [0, 20); phase 2: the rest (month files built, then extended from themselves)
    let d_mid = day_of(day0) + 20;
    let mut after_phase1 = vec![];
    for (lo, hi) in [(i64::MIN, d_mid), (d_mid, i64::MAX)] {
        core.append(Record::Series(series(lo, hi))).unwrap();
        let part: Vec<Sample> = samples.iter().filter(|x| day_of(x.ts) >= lo && day_of(x.ts) < hi).cloned().collect();
        for chunk in part.chunks(50_000) {
            core.append(Record::Samples(chunk.to_vec())).unwrap();
            core.flush().unwrap();
        }
        core.compactor().unwrap().run_once(now_ms(), true).unwrap();
        if lo == i64::MIN {
            after_phase1 = months(&core, KIND_1HM);
        }
    }
    // same months and spans as the 1 d tier; October's file extended in phase 2
    let m1h = months(&core, KIND_1HM);
    let m1d = months(&core, KIND_1D);
    assert_eq!(m1h.len(), 2, "{m1h:?}");
    let span = |v: &[pgai_mon::catalog::FileEntry]| v.iter().map(|f| (f.day, f.min_ts, f.max_ts)).collect::<Vec<_>>();
    assert_eq!(span(&m1h), span(&m1d));
    assert!(m1h.iter().all(|f| f.ver == rollup::VER && f.path.starts_with("rollup_1h_month/month=")));
    assert_eq!(after_phase1.len(), 2);
    assert!(m1h[1].max_ts > after_phase1[1].max_ts, "October extended");
    let nday = core.shared.catalog.lock().unwrap().files_of(KIND).count();
    // the month files hold exactly the day files' buckets in their range
    let day_rows: u64 = core
        .shared
        .catalog
        .lock()
        .unwrap()
        .files_of(KIND)
        .map(|f| f.rows)
        .sum();
    eprintln!("1 h: {nday} day files, {day_rows} rows; month files {:?}", m1h.iter().map(|f| f.rows).collect::<Vec<_>>());

    // a 30 d panel at a 1 h step reads the two month files (plus day files outside them)
    let ctx = pgai_mon::eval::EvalCtx::from_catalog(d.path().to_path_buf(), &core.shared.catalog.lock().unwrap().clone());
    let g = Grid {
        start: day0 + 5 * day,
        end: day0 + 35 * day,
        step: h,
        window: h,
        phi: 0.0,
    };
    let plan = rollup::plan(&ctx, Func::Increase, &g).expect("rollup-served");
    let files: Vec<&str> = plan.segs.iter().flat_map(|s| s.2.iter().map(|f| f.kind.as_str())).collect();
    let nm = files.iter().filter(|k| **k == KIND_1HM).count();
    assert_eq!(nm, 2, "{:?}", plan.segs.iter().map(|s| (s.0, s.1, s.2.len())).collect::<Vec<_>>());
    assert!(files.len() <= 4, "30 d at 1 h opens {} rollup files: {files:?}", files.len());

    let views = format!("{EMBEDDED_VIEWS_SQL}\n;\n{MACROS}");
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
    let cases = [
        (day0 + 2 * day, day0 + 38 * day, h),           // across the month end, both files
        (day0 + 3 * day + 5 * h, day0 + 10 * day, 6 * h), // inside September's file
        (day0 + 12 * day, day0 + 39 * day + 7 * h, h),   // past the month files' end: day files, raw
        (day0 + 15 * day + 13 * h, day0 + 15 * day + 13 * h, h), // one point
    ];
    let (n, served) = check(&r, &cases);
    eprintln!("1 h month tier: {n} rows bit-identical to the day files; {served} calls read month files");
    assert!(n > 5_000, "{n}");
    assert!(served > 30, "{served}");

    // a late sample recompacts a day: the month file is stale for that day's buckets (served from
    // the day file), results still equal; the same compaction rebuilds it
    let late = day0 + 15 * day + 12 * h + 30_000;
    let id1 = series_id("d_total", &[("i".to_string(), "1".to_string())]);
    core.append(Record::Samples(vec![Sample { series_id: id1, ts: late, value: 1e6 }])).unwrap();
    core.flush().unwrap();
    {
        // before compaction: the day has a level-0 file, so it is not valid: neither tier serves it
        r.refresh().ok();
        let (n2, _) = check(&r, &cases[..1]);
        assert!(n2 > 500);
    }
    // recompact the day without rebuilding the month files (update_1d runs only with compact_kmerge)
    let stale_before = months(&core, KIND_1HM);
    core.compactor().unwrap().run_once(now_ms(), true).unwrap();
    let rebuilt = months(&core, KIND_1HM);
    assert_ne!(stale_before[1].path, rebuilt[1].path, "October's file rebuilt");
    assert_eq!(stale_before[0].path, rebuilt[0].path, "September's file kept");
    assert_eq!(span(&stale_before), span(&rebuilt));
    r.refresh().ok();
    let (n3, served3) = check(&r, &cases[..1]);
    assert!(n3 > 500 && served3 > 0);
    // a stale month file (a day newer than it): its buckets on that day come from the day file
    let mut cat = core.shared.catalog.lock().unwrap().clone();
    for f in cat.files.iter_mut().filter(|f| f.kind == KIND_1HM) {
        f.created_ms = 0;
    }
    let mut ctx = pgai_mon::eval::EvalCtx::from_catalog(d.path().to_path_buf(), &cat);
    ctx.generation = 1;
    let plan = rollup::plan(&ctx, Func::Increase, &g).expect("rollup-served");
    assert!(
        plan.segs.iter().all(|s| s.2.iter().all(|f| f.kind == KIND)),
        "stale month files are not served"
    );
}
