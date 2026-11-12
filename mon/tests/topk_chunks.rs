//! Top-k over long ranges evaluated in time chunks (PGAI_TOPK_CHUNK_ALWAYS / CHUNK_DAYS) with
//! closed days served from 1 h rollups: each chunk opens only its own days' files, with the
//! scrape intervals (si) of the whole range, so per-step results equal the unchunked
//! evaluation and the SQL oracle. Covers irate, last (staleness markers counted in si, sparse
//! s_total), last_over_time and window aggregates, plain, bottom-k, sum by, ratio and
//! prom_topk_expr; and rate, which is never chunked (evaluated whole under the same knobs).
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::model::*;
use pgai_mon::query::{QueryResult, Reader, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::StoreCore;

const MACROS: &str = include_str!("../sql/macros.sql");

/// Copied from tests/eval.rs: three closed days of r_total (24 counters, 60 s, resets, 3 h gaps,
/// NaN, staleness markers) and s_total (4 sparse series, frequent markers), compacted.
fn rollup_store(dir: &std::path::Path) -> StoreCore {
    let mut core = StoreCore::open(
        dir,
        &StoreCfg {
            delete_grace_s: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let day0 = 1_794_009_600_000i64 - 86_400_000; // a UTC midnight
    let mut defs = vec![];
    let mut samples = vec![];
    let mut seed = 11u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for s in 0..24u64 {
        let labels = vec![("i".to_string(), s.to_string())];
        let id = series_id("r_total", &labels);
        defs.push(SeriesDef {
            series_id: id,
            metric: "r_total".into(),
            labels,
            first_seen: day0,
        });
        let aligned = s % 3 == 0; // exact 60 s grid: samples on every hour/day boundary
        let mut t = day0 + if aligned { 0 } else { (rnd() % 60_000) as i64 };
        let mut v = (rnd() % 100) as f64;
        let mut k = 0;
        while t < day0 + 3 * 86_400_000 {
            v += (rnd() % 20) as f64;
            if k % 1500 == 700 {
                v = 1.0; // reset
            }
            if k % 997 == 300 {
                t += 3 * 3_600_000; // 3 h gap
            }
            let value = if k % 701 == 50 {
                stale_nan()
            } else if k % 911 == 60 {
                f64::NAN
            } else {
                v
            };
            samples.push(Sample {
                series_id: id,
                ts: t,
                value,
            });
            t += if aligned {
                60_000
            } else {
                60_000 + (rnd() % 300) as i64 - 150
            };
            k += 1;
        }
    }
    // s_total: sparse (75-165 min apart) with frequent staleness markers, some back to back:
    // a bare selector's window is maxPrev(si) > step (not on a boundary), si counting markers
    for s in 0..4u64 {
        let labels = vec![("i".to_string(), s.to_string())];
        let id = series_id("s_total", &labels);
        defs.push(SeriesDef {
            series_id: id,
            metric: "s_total".into(),
            labels,
            first_seen: day0,
        });
        let mut t = day0 + (rnd() % 3_600_000) as i64;
        let mut k = 0;
        while t < day0 + 3 * 86_400_000 {
            let value = match (k + s) % 7 {
                2 | 5 => stale_nan(),
                3 if s == 1 => stale_nan(),
                _ => (rnd() % 1000) as f64,
            };
            samples.push(Sample { series_id: id, ts: t, value });
            t += 4_500_000 + (rnd() % 5_400_000) as i64;
            k += 1;
        }
    }
    // like the producers: one SeriesDef per series per day it has samples (per-day series files)
    let mut per_day: std::collections::BTreeMap<(u64, i64), i64> = Default::default();
    for x in &samples {
        let e = per_day.entry((x.series_id, day_of(x.ts))).or_insert(x.ts);
        *e = (*e).min(x.ts);
    }
    let by_id: std::collections::HashMap<u64, SeriesDef> = defs.into_iter().map(|d| (d.series_id, d)).collect();
    let defs: Vec<SeriesDef> = per_day
        .into_iter()
        .map(|((id, _), first)| SeriesDef {
            first_seen: first,
            ..by_id[&id].clone()
        })
        .collect();
    core.append(Record::Series(defs)).unwrap();
    for chunk in samples.chunks(50_000) {
        core.append(Record::Samples(chunk.to_vec())).unwrap();
        core.flush().unwrap();
    }
    core.compactor().unwrap().run_once(now_ms(), true).unwrap();
    let rollups = core.shared.catalog.lock().unwrap().files_of("rollup_1h").count();
    assert_eq!(rollups, 3, "one rollup per closed day");
    core
}

/// (t, sorted values) per step: chunks restart counter corrections, so values agree to
/// rounding, and values equal up to rounding may rank in another order
fn by_step(x: &QueryResult) -> std::collections::BTreeMap<i64, Vec<f64>> {
    let mut m: std::collections::BTreeMap<i64, Vec<f64>> = Default::default();
    for r in &x.rows {
        let v: (i64, serde_json::Value) = serde_json::from_str(r.get()).unwrap();
        m.entry(v.0).or_default().push(v.1.as_f64().unwrap_or(f64::NAN));
    }
    m.values_mut().for_each(|v| v.sort_by(|a, b| a.total_cmp(b)));
    m
}

fn same(tag: &str, a: &QueryResult, b: &QueryResult) {
    let (ma, mb) = (by_step(a), by_step(b));
    assert_eq!(ma.keys().collect::<Vec<_>>(), mb.keys().collect::<Vec<_>>(), "{tag}");
    for (t, (va, vb)) in ma.keys().zip(ma.values().zip(mb.values())) {
        assert_eq!(va.len(), vb.len(), "{tag} t={t}");
        for (x, y) in va.iter().zip(vb) {
            assert!(
                (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-9 * y.abs().max(1e-12),
                "{tag} t={t}: {x} vs {y}"
            );
        }
    }
}

#[test]
fn chunked_topk_over_rollups_matches_unchunked_and_oracle() {
    let d = tempfile::tempdir().unwrap();
    let core = rollup_store(d.path());
    let day0 = 1_794_009_600_000i64 - 86_400_000;
    let h = 3_600_000i64;
    let r = Reader::new(
        core.shared.clone(),
        HttpCfg {
            memory_limit: "1GB".into(),
            ..Default::default()
        },
        format!("{EMBEDDED_VIEWS_SQL}\n;\n{MACROS}"),
        vec![],
    )
    .unwrap();
    // every run must scan (the served check), not replay rows the row cache kept
    pgai_mon::eval::rowcache::set_cap(0);
    let grids = [
        (day0 + 5 * h, day0 + 70 * h, h, h + 5_000),         // hourly: rollup-served
        (day0 + 6 * h, day0 + 71 * h, 2 * h, 2 * h + 5_000), // 2 h step
        (day0 + 2 * h + 30_000, day0 + 70 * h, 600_000, 65_000), // sub-hour: raw day files
    ];
    let mut cases: Vec<(String, String)> = vec![];
    for (start, end, step, window) in grids {
        // rate is served from rollups on whole buckets (window = step), irate on step + 5 s;
        // rate is never chunked (its 0 on an empty window depends on later samples): the
        // knobs must leave it evaluated whole
        for (f, window) in [("irate", window), ("rate", if step >= h { step } else { window })] {
            for extra in ["", ", p_bottom := true", ", p_by := ['i']"] {
                let args = format!("4, '{f}', 'r_total', [], {start}, {end}, {step}, {window}{extra}");
                cases.push((format!("prom_topk({args})"), format!("prom_topk_sql({args})")));
            }
        }
        // window aggregates: rollup segments on whole buckets, raw samples on the sub-hour grid
        for f in ["avg_over_time", "max_over_time"] {
            let w = if step >= h { step } else { 1_800_000 };
            let args = format!("4, '{f}', 'r_total', [], {start}, {end}, {step}, {w}");
            cases.push((format!("prom_topk({args})"), format!("prom_topk_sql({args})")));
        }
        for m in ["r_total", "s_total"] {
            for bottom in ["false", "true"] {
                let args = format!("3, 'last', '{m}', [], {start}, {end}, {step}, 0, p_bottom := {bottom}");
                cases.push((format!("prom_topk({args})"), format!("prom_topk_sql({args})")));
                let args = format!("3, 'last_over_time', '{m}', [], {start}, {end}, {step}, {}, p_bottom := {bottom}", 6 * h);
                cases.push((format!("prom_topk({args})"), format!("prom_topk_sql({args})")));
            }
        }
        // r / (r + s) on the four label sets both metrics have, and r - s
        for t in [
            "[{'m': 'r_total', 'n': 1.0, 'd': 1.0}, {'m': 's_total', 'n': 0.0, 'd': 1.0}]",
            "[{'m': 'r_total', 'n': 1.0, 'd': 0.0}, {'m': 's_total', 'n': -1.0, 'd': 0.0}]",
        ] {
            let args = format!("2, 'irate', {t}, [], {start}, {end}, {step}, {}", window.max(4 * h));
            cases.push((format!("prom_topk_expr({args})"), format!("prom_topk_expr_sql({args})")));
        }
    }
    let q = |src: &str| format!("SELECT t, value FROM {src} ORDER BY 1, 2");
    let run = |src: &str| r.run(&q(src)).unwrap_or_else(|e| panic!("{src}: {e:?}"));
    let served = || {
        use pgai_mon::eval::rollup::SERVED;
        SERVED.iter().map(|x| x.load(std::sync::atomic::Ordering::Relaxed)).sum::<u64>()
    };
    let mut checked = 0usize;
    for (rust, sql) in &cases {
        // the knobs are read at bind time; this binary's only test sets them
        std::env::set_var("PGAI_TOPK_CHUNK_ALWAYS", "0");
        std::env::set_var("PGAI_TOPK_CHUNK_DAYS", "100");
        let whole = run(rust);
        // one-day chunks: three or more chunks over three day files
        std::env::set_var("PGAI_TOPK_CHUNK_ALWAYS", "1");
        std::env::set_var("PGAI_TOPK_CHUNK_DAYS", "1");
        let before = served();
        let chunked = run(rust);
        if !rust.contains(", 600000, ") {
            assert!(served() > before, "{rust}: chunks not served from rollups");
        }
        let oracle = run(sql);
        if std::env::var_os("TOPK_CHUNKS_DEBUG").is_some() {
            let first = |x: &QueryResult| x.rows.iter().take(4).map(|r| r.get().to_string()).collect::<Vec<_>>();
            eprintln!("{rust}\n whole {:?}\n chunk {:?}\n oracle {:?}", first(&whole), first(&chunked), first(&oracle));
        }
        same(&format!("chunked vs oracle: {rust}"), &chunked, &oracle);
        same(&format!("chunked vs whole: {rust}"), &chunked, &whole);
        assert!(chunked.row_count > 0, "{rust}: empty");
        checked += chunked.row_count;
    }
    std::env::remove_var("PGAI_TOPK_CHUNK_ALWAYS");
    std::env::remove_var("PGAI_TOPK_CHUNK_DAYS");
    assert!(checked > 3_000, "{checked}");
}
