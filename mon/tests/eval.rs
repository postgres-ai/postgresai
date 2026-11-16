//! Evaluator: kernel properties, and Rust table functions vs the SQL-macro oracle on synthetic
//! series with counter resets, gaps, NaN, staleness markers and jittered scrapes.
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::eval::kernels::{eval_series, Func, Grid};
use pgai_mon::model::*;
use pgai_mon::query::{Fmt, Reader, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::StoreCore;

const MACROS: &str = include_str!("../sql/macros.sql");

/// The top-k knobs are process environment variables: tests that set them take this lock.
static KNOBS: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn run(f: Func, g: Grid, ts: &[i64], v: &[f64]) -> Vec<(i64, f64)> {
    let mut out = vec![];
    eval_series(f, &g, ts, v, &mut vec![], |t, y| out.push((t, y)));
    out
}

#[test]
fn kernel_counter_reset_is_removed() {
    // 30 s scrapes, counter 0,10,20,5(reset),15: rate over the whole range = 40 / 120 s
    let ts: Vec<i64> = (0..5).map(|i| 1_000_000 + i * 30_000).collect();
    let v = [0.0, 10.0, 20.0, 5.0, 15.0];
    let g = Grid {
        start: 1_120_000,
        end: 1_120_000,
        step: 300_000,
        window: 150_000,
        phi: 0.0,
    };
    let r = run(Func::Increase, g, &ts, &v);
    assert_eq!(r, vec![(1_120_000, 35.0)]); // 20 + 5 + 10 counted, the drop to 5 counts from 0
    let r = run(Func::Rate, g, &ts, &v);
    assert!((r[0].1 - 35.0 / 120.0).abs() < 1e-12, "{r:?}");
}

#[test]
fn kernel_gap_longer_than_window_and_single_sample() {
    let ts = [0i64, 30_000, 600_000];
    let v = [1.0, 2.0, 3.0];
    let g = Grid {
        start: 300_000,
        end: 300_000,
        step: 60_000,
        window: 60_000,
        phi: 0.0,
    };
    assert!(run(Func::Irate, g, &ts, &v).is_empty(), "nothing in (240s, 300s]");
    assert!(run(Func::Irate, g, &[0], &[1.0]).is_empty());
}

fn build_store(dir: &std::path::Path) -> StoreCore {
    let mut core = StoreCore::open(dir, &StoreCfg::default()).unwrap();
    let t0 = 1_794_000_000_000i64;
    let mut defs = vec![];
    let mut samples = vec![];
    let mut seed = 7u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for s in 0..40u64 {
        let labels = vec![
            ("i".to_string(), s.to_string()),
            ("job".to_string(), if s % 2 == 0 { "a" } else { "b" }.to_string()),
        ];
        let id = series_id("m_total", &labels);
        defs.push(SeriesDef {
            series_id: id,
            metric: "m_total".into(),
            labels,
            first_seen: t0,
        });
        let mut v = (rnd() % 1000) as f64;
        let mut t = t0 + (rnd() % 30_000) as i64;
        for k in 0..400 {
            t += 30_000 + (rnd() % 400) as i64 - 200; // jittered 30 s
            if k % 97 == 50 {
                t += 400_000; // a gap longer than the windows
            }
            v += (rnd() % 50) as f64 + 0.25;
            if k % 113 == 60 {
                v = (rnd() % 5) as f64; // counter reset
            }
            if k % 151 == 70 {
                v *= 0.95; // partial reset (< 1/8 drop)
            }
            let value = if k % 89 == 30 {
                stale_nan()
            } else if k % 131 == 40 {
                f64::NAN
            } else {
                v
            };
            samples.push(Sample {
                series_id: id,
                ts: t,
                value,
            });
        }
    }
    // m2_total: same label sets as m_total (D1's per-call denominator)
    let n = defs.len();
    for k in 0..n {
        let d = defs[k].clone();
        let id2 = series_id("m2_total", &d.labels);
        defs.push(SeriesDef {
            series_id: id2,
            metric: "m2_total".into(),
            labels: d.labels.clone(),
            first_seen: t0,
        });
        let mine: Vec<Sample> = samples.iter().filter(|s| s.series_id == d.series_id).cloned().collect();
        for (j, s) in mine.into_iter().enumerate() {
            let v = if s.value.is_nan() {
                s.value
            } else {
                (j as f64) * 2.0 + (k as f64)
            };
            samples.push(Sample {
                series_id: id2,
                ts: s.ts + 10,
                value: v,
            });
        }
    }
    core.append(Record::Series(defs)).unwrap();
    core.append(Record::Samples(samples)).unwrap();
    core.flush().unwrap();
    core
}

/// prom_irate_k / prom_labels_k (label-set keyed, for sums and ratios over several metrics) give
/// their macro twins' rows, and m_total and m2_total (same label sets) share keys; a query of
/// several table-function calls gives each call's own result (each call's scan is released when
/// it ends, not at the end of the query).
#[test]
fn label_keyed_irate_matches_sql_oracle() {
    let d = tempfile::tempdir().unwrap();
    let core = build_store(d.path());
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
    let t0 = 1_794_000_000_000i64;
    let rows = |sql: &str| -> Vec<serde_json::Value> {
        let res = r.run(sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
        res.rows
            .iter()
            .map(|x| serde_json::from_str(x.get()).unwrap())
            .collect()
    };
    let args = |m: &str| format!("'{m}', [], {}, {}, 60000, 65000", t0 + 600_000, t0 + 10_000_000);
    for m in ["m_total", "m2_total"] {
        let q = |f: &str| format!("SELECT lk, t, value FROM {f}({}) ORDER BY 1, 2", args(m));
        let (a, b) = (rows(&q("prom_irate_k")), rows(&q("prom_irate_k_sql")));
        assert!(a.len() > 1_000, "{}", a.len());
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!((&x[0], &x[1]), (&y[0], &y[1]));
            let (p, q) = (x[2].as_f64().unwrap(), y[2].as_f64().unwrap());
            assert!((p - q).abs() <= 1e-9 * q.abs().max(1e-12), "{x:?} vs {y:?}");
        }
        let q = |f: &str| format!("SELECT lk, CAST(labels AS VARCHAR) FROM {f}({}) ORDER BY 1", args(m));
        assert_eq!(rows(&q("prom_labels_k")), rows(&q("prom_labels_k_sql")));
    }
    // both metrics' series pair up on lk, and a UNION ALL of calls equals the calls one by one
    let one = |m: &str| rows(&format!("SELECT count(*), sum(value) FROM prom_irate_k({})", args(m)));
    let both = rows(&format!(
        "SELECT count(*), sum(value), count(DISTINCT lk) FROM (SELECT lk, value FROM prom_irate_k({}) \
         UNION ALL SELECT lk, value FROM prom_irate_k({}))",
        args("m_total"),
        args("m2_total")
    ));
    let (a, b) = (one("m_total"), one("m2_total"));
    assert_eq!(
        both[0][0].as_i64().unwrap(),
        a[0][0].as_i64().unwrap() + b[0][0].as_i64().unwrap()
    );
    assert_eq!(
        both[0][2].as_i64().unwrap(),
        40,
        "40 label sets shared by the two metrics"
    );
}

#[test]
fn rust_functions_match_sql_oracle() {
    let d = tempfile::tempdir().unwrap();
    let core = build_store(d.path());
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
    let t0 = 1_794_000_000_000i64;
    let mut checked = 0;
    let six = [
        "rate",
        "irate",
        "increase",
        "delta",
        "increase_pure",
        "idelta",
        "last_over_time",
        "tlast_over_time",
        "first_over_time",
        "avg_over_time",
        "sum_over_time",
        "min_over_time",
        "max_over_time",
        "count_over_time",
        "stddev_over_time",
    ];
    let grids = [
        (t0 + 600_000, t0 + 10_000_000, 60_000, 0),
        (t0 + 600_000, t0 + 10_000_000, 30_000, 35_000),
        (t0 + 1_000_000, t0 + 12_000_000, 300_000, 900_000),
        (t0 + 5_000_000, t0 + 5_000_000, 300_000, 0), // instant
    ];
    let mut cases: Vec<(String, String)> = vec![];
    for (start, end, step, window) in grids {
        for m in ["[]", "[['job','=','a']]", "[['i','=~','1.*']]"] {
            for f in six {
                let args = format!("'m_total', {m}, {start}, {end}, {step}, {window}");
                cases.push((format!("prom_{f}({args})"), format!("prom_{f}_sql({args})")));
            }
            for f in ["last", "timestamp"] {
                let args = format!("'m_total', {m}, {start}, {end}, {step}");
                cases.push((format!("prom_{f}({args})"), format!("prom_{f}_sql({args})")));
            }
            let args = format!("0.9, 'm_total', {m}, {start}, {end}, {step}, {window}");
            cases.push((
                format!("prom_quantile_over_time({args})"),
                format!("prom_quantile_over_time_sql({args})"),
            ));
            for f in ["irate", "rate", "avg_over_time"] {
                let args =
                    format!("3, '{f}', 'm_total', {m}, {start}, {end}, {step}, {window}, p_den_metric := 'm2_total'");
                cases.push((format!("prom_topk({args})"), format!("prom_topk_sql({args})")));
                let args = format!("2, '{f}', 'm_total', {m}, {start}, {end}, {step}, {window}, p_bottom := true");
                cases.push((format!("prom_topk({args})"), format!("prom_topk_sql({args})")));
            }
            // idelta of m2_total is 2 for every series: ties at every step, resolved by labels
            for bottom in ["false", "true"] {
                let args =
                    format!("3, 'idelta', 'm2_total', {m}, {start}, {end}, {step}, {window}, p_bottom := {bottom}");
                cases.push((format!("prom_topk({args})"), format!("prom_topk_sql({args})")));
            }
        }
    }
    // every top-k case again through the lean path (ids only, labels for finalists), with
    // compaction forced after a handful of candidates
    // ... and the ratio cases once more evaluated in time chunks of 2-3 steps (both paths)
    let lean: Vec<(String, String)> = cases
        .iter()
        .filter(|(r, _)| r.starts_with("prom_topk("))
        .map(|(r, q)| (format!("{r} /*lean*/"), q.clone()))
        .collect();
    let chunk: Vec<(String, String)> = cases
        .iter()
        .filter(|(r, _)| r.starts_with("prom_topk(") && r.contains("p_den_metric"))
        .flat_map(|(r, q)| {
            [
                (format!("{r} /*chunk*/"), q.clone()),
                (format!("{r} /*leanchunk*/"), q.clone()),
            ]
        })
        .collect();
    cases.extend(lean);
    cases.extend(chunk);
    let _knobs = KNOBS.lock().unwrap_or_else(|e| e.into_inner());
    for (rust, sql) in cases {
        let lean = rust.ends_with("/*lean*/") || rust.ends_with("/*leanchunk*/");
        let chunk = rust.ends_with("chunk*/");
        // the knobs are read at bind time; other tests in this binary only compare with oracles too
        std::env::set_var("PGAI_TOPK_LEAN_ABOVE", if lean { "0" } else { "100000" });
        std::env::set_var("PGAI_TOPK_COMPACT_ABOVE", if lean { "50" } else { "300000" });
        std::env::set_var("PGAI_TOPK_DEN_BUDGET", if chunk { "1000" } else { "50331648" });
        let tag = rust.clone();
        let rust = rust.split(" /*").next().unwrap().to_string();
        let q = |src: &str| format!("SELECT series_id, t, value FROM {src} ORDER BY 1, 2");
        let a = r.run(&q(&rust)).unwrap_or_else(|e| panic!("{rust}: {e:?}"));
        let b = r.run(&q(&sql)).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
        if std::env::var("DBG_DIFF").is_ok() && a.rows.iter().map(|x| x.get()).ne(b.rows.iter().map(|x| x.get())) {
            let sa: std::collections::BTreeSet<String> = a.rows.iter().map(|x| x.get().to_string()).collect();
            let sb: std::collections::BTreeSet<String> = b.rows.iter().map(|x| x.get().to_string()).collect();
            eprintln!(
                "DIFF {tag}\n only rust: {:?}\n only sql: {:?}",
                sa.difference(&sb).take(6).collect::<Vec<_>>(),
                sb.difference(&sa).take(6).collect::<Vec<_>>()
            );
        }
        assert_eq!(a.row_count, b.row_count, "{tag}");
        if chunk {
            // time chunks accumulate counter corrections from the chunk's first sample, so values
            // agree to rounding and series whose values are equal up to rounding may rank in
            // another order: compare each step's sorted values
            let by_t = |rows: &[Box<serde_json::value::RawValue>]| {
                let mut m: std::collections::BTreeMap<i64, Vec<f64>> = Default::default();
                for r in rows {
                    let x: (u64, i64, serde_json::Value) = serde_json::from_str(r.get()).unwrap();
                    m.entry(x.1).or_default().push(x.2.as_f64().unwrap_or(f64::NAN));
                }
                m.values_mut().for_each(|v| v.sort_by(|a, b| a.total_cmp(b)));
                m
            };
            let (ma, mb) = (by_t(&a.rows), by_t(&b.rows));
            assert_eq!(ma.keys().collect::<Vec<_>>(), mb.keys().collect::<Vec<_>>(), "{tag}");
            for (va, vb) in ma.values().zip(mb.values()) {
                assert_eq!(va.len(), vb.len(), "{tag}");
                for (x, y) in va.iter().zip(vb) {
                    assert!(
                        x.is_nan() && y.is_nan() || (x - y).abs() <= 1e-9 * y.abs().max(1e-12),
                        "{tag}: {x} vs {y}"
                    );
                }
            }
            checked += a.row_count;
            continue;
        }
        for (x, y) in a.rows.iter().zip(b.rows.iter()) {
            let x: (u64, i64, serde_json::Value) = serde_json::from_str(x.get()).unwrap();
            let y: (u64, i64, serde_json::Value) = serde_json::from_str(y.get()).unwrap();
            assert_eq!((x.0, x.1), (y.0, y.1), "{tag}");
            match (x.2.as_f64(), y.2.as_f64()) {
                (Some(xv), Some(yv)) => assert!((xv - yv).abs() <= 1e-9 * yv.abs().max(1e-12), "{tag} {x:?} vs {y:?}"),
                _ => assert_eq!(x.2, y.2, "{tag}"),
            }
        }
        checked += a.row_count;
    }
    assert!(checked > 10_000, "{checked}");

    // prom_agg (Rust group sink) = GROUP BY over the oracle's per-series output
    let mut agg = 0;
    for (start, end, step, window) in [grids[1], grids[2]] {
        for op in ["sum", "avg", "count", "min", "max"] {
            for by in ["CAST([] AS VARCHAR[])", "['job']", "['i', 'job']"] {
                let a = r
                    .run(&format!(
                        "SELECT labels::VARCHAR, t, value FROM prom_agg('{op}', 'irate', 'm_total', [], {start}, {end}, {step}, {window}, p_by := {by}) ORDER BY 1, 2"
                    ))
                    .unwrap();
                let b = r
                    .run(&format!(
                        "SELECT prom_by_rs(labels, {by})::VARCHAR, t, {op}(value) FROM prom_irate_sql('m_total', [], {start}, {end}, {step}, {window}) \
                         WHERE NOT isnan(value) GROUP BY 1, 2 ORDER BY 1, 2"
                    ))
                    .unwrap();
                assert_eq!(a.row_count, b.row_count, "{op} {by}");
                for (x, y) in a.rows.iter().zip(b.rows.iter()) {
                    let x: (String, i64, f64) = serde_json::from_str(x.get()).unwrap();
                    let y: (String, i64, f64) = serde_json::from_str(y.get()).unwrap();
                    assert_eq!((&x.0, x.1), (&y.0, y.1), "{op} {by}");
                    assert!(
                        (x.2 - y.2).abs() <= 1e-9 * y.2.abs().max(1e-12),
                        "{op} {by}: {x:?} vs {y:?}"
                    );
                }
                agg += a.row_count;
            }
        }
    }
    assert!(agg > 1_000, "{agg}");
}

/// A Rust call against the SQL oracle and against the Rust raw path (PGAI_NO_ROLLUP): same
/// (series, t) rows, values within 1e-9 relative. `tier`: the call must be served from rollups
/// (0: 1 h, 1: 1 d). Returns (rows, rows bit-identical to the raw path). Caller holds KNOBS.
fn parity(r: &Reader, call: &str, oracle: &str, tier: Option<usize>) -> (usize, usize) {
    use pgai_mon::eval::rollup::SERVED;
    use std::sync::atomic::Ordering;
    let q = |c: &str| format!("SELECT series_id, t, value FROM {c} ORDER BY 1, 2");
    let rows = |c: &str| -> Vec<(u64, i64, f64)> {
        let res = r.run(&q(c)).unwrap_or_else(|e| panic!("{c}: {e:?}"));
        res.rows
            .iter()
            .map(|x| serde_json::from_str(x.get()).unwrap())
            .collect()
    };
    let before = tier.map(|t| SERVED[t].load(Ordering::Relaxed));
    let a = rows(call);
    if let (Some(t), Some(b)) = (tier, before) {
        assert!(
            SERVED[t].load(Ordering::Relaxed) > b,
            "{call}: not served from tier {t}"
        );
    }
    let o = rows(oracle);
    std::env::set_var("PGAI_NO_ROLLUP", "1");
    let raw = rows(call);
    std::env::remove_var("PGAI_NO_ROLLUP");
    let mut exact = 0;
    for (name, other) in [("oracle", &o), ("raw", &raw)] {
        assert_eq!(a.len(), other.len(), "{call} vs {name}");
        for (x, y) in a.iter().zip(other.iter()) {
            assert_eq!((x.0, x.1), (y.0, y.1), "{call} vs {name}");
            assert!(
                (x.2 - y.2).abs() <= 1e-9 * y.2.abs().max(1e-12),
                "{call} vs {name}: {x:?} vs {y:?}"
            );
            if name == "raw" && x.2.to_bits() == y.2.to_bits() {
                exact += 1;
            }
        }
    }
    (a.len(), exact)
}

/// Rollups (built at closed-day compaction) must give the raw result exactly wherever they are
/// used: aligned range grids, instant points with long windows; samples exactly on hour and day
/// boundaries (buckets split across day files), resets, gaps, NaN and staleness markers.
/// Three closed days of 24 counters (60 s, some on an exact grid) with resets, 3 h gaps, NaN and
/// staleness markers, compacted: day files, per-day series files and 1 h rollups.
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
            samples.push(Sample {
                series_id: id,
                ts: t,
                value,
            });
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

#[test]
fn rollup_served_functions_match_sql_oracle() {
    // PGAI_NO_ROLLUP / top-k knobs are process-wide: hold the lock for the whole test
    let _knobs = KNOBS.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    let core = rollup_store(d.path());
    let day0 = 1_794_009_600_000i64 - 86_400_000; // a UTC midnight
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
    let h = 3_600_000i64;
    let mut checked = 0;
    for f in [
        "increase",
        "rate",
        "increase_pure",
        "delta",
        "last_over_time",
        "first_over_time",
        "tlast_over_time",
    ] {
        for (start, end, step, window) in [
            (day0 + 5 * h, day0 + 70 * h, h, h),                         // hourly, 1 h windows
            (day0 + 6 * h, day0 + 71 * h, 2 * h, 6 * h),                 // 2 h step, 6 h window
            (day0 + 72 * h - 1, day0 + 72 * h - 1, 300_000, 840 * h),    // instant, window > data
            (day0 + 50 * h + 777, day0 + 50 * h + 777, 300_000, 30 * h), // instant, unaligned
            (day0 + 5 * h + 1, day0 + 70 * h + 1, h, h),                 // unaligned grid: raw path
        ] {
            let args = format!("'r_total', [], {start}, {end}, {step}, {window}");
            let q = |name: &str| format!("SELECT series_id, t, value FROM {name}({args}) ORDER BY 1, 2");
            let a = r.run(&q(&format!("prom_{f}"))).unwrap();
            let b = r.run(&q(&format!("prom_{f}_sql"))).unwrap();
            assert_eq!(a.row_count, b.row_count, "{f} {args}");
            for (x, y) in a.rows.iter().zip(b.rows.iter()) {
                let x: (u64, i64, f64) = serde_json::from_str(x.get()).unwrap();
                let y: (u64, i64, f64) = serde_json::from_str(y.get()).unwrap();
                assert_eq!((x.0, x.1), (y.0, y.1), "{f} {args}");
                assert!(
                    (x.2 - y.2).abs() <= 1e-9 * y.2.abs().max(1e-12),
                    "{f} {args}: {x:?} vs {y:?}"
                );
            }
            checked += a.row_count;
        }
    }
    assert!(checked > 5_000, "{checked}");

    // irate from rollups (format 2: each bucket's second-to-last sample): exact for any window
    // once the grid is on bucket boundaries, e.g. Grafana's $__rate_interval = step + 5 s
    let mut irate = 0;
    for (start, end, step, window) in [
        (day0 + 5 * h, day0 + 70 * h, h, h + 5_000),
        (day0 + 6 * h, day0 + 71 * h, 2 * h, 2 * h + 5_000),
        (day0 + 5 * h, day0 + 70 * h, h, 1_800_000), // window shorter than a bucket
        (day0 + 5 * h, day0 + 70 * h, h, 60_000),    // ~one sample per window
        (day0 + 72 * h, day0 + 72 * h, 300_000, 6 * h + 5_000), // instant on a boundary
        (day0 + 5 * h + 1, day0 + 70 * h + 1, h, h + 5_000), // unaligned grid: raw path
    ] {
        let args = format!("'r_total', [], {start}, {end}, {step}, {window}");
        let q = |name: &str| format!("SELECT series_id, t, value FROM {name}({args}) ORDER BY 1, 2");
        let a = r.run(&q("prom_irate")).unwrap();
        let b = r.run(&q("prom_irate_sql")).unwrap();
        assert_eq!(a.row_count, b.row_count, "irate {args}");
        for (x, y) in a.rows.iter().zip(b.rows.iter()) {
            let x: (u64, i64, f64) = serde_json::from_str(x.get()).unwrap();
            let y: (u64, i64, f64) = serde_json::from_str(y.get()).unwrap();
            assert_eq!((x.0, x.1), (y.0, y.1), "irate {args}");
            assert!(
                (x.2 - y.2).abs() <= 1e-9 * y.2.abs().max(1e-12),
                "irate {args}: {x:?} vs {y:?}"
            );
        }
        irate += a.row_count;
    }
    assert!(irate > 1_000, "{irate}");

    // bare selector (staleness markers end the series; window = max(step, maxPrev(si)) with
    // markers counted in si), timestamp, idelta: only the grid must be on bucket boundaries
    let (mut n, mut exact) = (0, 0);
    for (start, end, step) in [
        (day0 + 5 * h, day0 + 70 * h, h),
        (day0 + 6 * h, day0 + 71 * h, 2 * h),
        (day0 + 72 * h, day0 + 72 * h, 300_000), // instant on a boundary
    ] {
        for (f, m) in [
            ("last", "r_total"),
            ("timestamp", "r_total"),
            ("last", "s_total"),
            ("timestamp", "s_total"),
        ] {
            let args = format!("'{m}', [], {start}, {end}, {step}");
            // an instant reads only step + 5 m of history: raw
            let tier = (start < end).then_some(0);
            let (a, b) = parity(&r, &format!("prom_{f}({args})"), &format!("prom_{f}_sql({args})"), tier);
            n += a;
            exact += b;
        }
        for window in [h + 5_000, 60_000, 5 * h] {
            let args = format!("'r_total', [], {start}, {end}, {step}, {window}");
            let tier = (start < end || window > 2 * h).then_some(0);
            let (a, b) = parity(
                &r,
                &format!("prom_idelta({args})"),
                &format!("prom_idelta_sql({args})"),
                tier,
            );
            n += a;
            exact += b;
        }
    }
    // unaligned grid: raw path
    let args = format!("'r_total', [], {}, {}, {h}", day0 + 5 * h + 1, day0 + 70 * h + 1);
    parity(
        &r,
        &format!("prom_last({args})"),
        &format!("prom_last_sql({args})"),
        None,
    );
    assert!(n > 3_000, "{n}");
    eprintln!("parity 1 h, last/timestamp/idelta: {n} rows, {exact} bit-identical to raw");
    assert_eq!(n, exact, "last/timestamp/idelta: not bit-identical to raw");

    // window aggregates: per-bucket count/sum/min/max, window = step when 0
    let (mut n, mut exact) = (0, 0);
    for f in ["sum", "min", "max", "avg", "count"] {
        for (start, end, step, window) in [
            (day0 + 5 * h, day0 + 70 * h, h, 0),
            (day0 + 6 * h, day0 + 71 * h, 2 * h, 6 * h),
            (day0 + 72 * h - 1, day0 + 72 * h - 1, 300_000, 40 * h), // instant, unaligned
        ] {
            let args = format!("'r_total', [], {start}, {end}, {step}, {window}");
            let (a, b) = parity(
                &r,
                &format!("prom_{f}_over_time({args})"),
                &format!("prom_{f}_over_time_sql({args})"),
                Some(0),
            );
            n += a;
            exact += b;
        }
    }
    assert!(n > 3_000, "{n}");
    eprintln!("parity 1 h, *_over_time: {n} rows, {exact} bit-identical to raw");
    assert_eq!(
        n, exact,
        "*_over_time on integer-valued series: not bit-identical to raw"
    );

    // a ratio top-k on a sub-hour grid evaluated in time chunks, with scrape intervals of the
    // whole range from rollups (closed days) and raw edges: per-step values as the oracle's
    std::env::set_var("PGAI_TOPK_DEN_BUDGET", "1000");
    std::env::set_var("PGAI_TOPK_CHUNK_ALWAYS", "1");
    for (start, end, step, window) in [
        (day0 + 2 * h + 30_000, day0 + 70 * h, 60_000, 65_000),
        (day0 + 5 * h, day0 + 71 * h, 300_000, 60_000),
    ] {
        let args = format!("5, 'irate', 'r_total', [], {start}, {end}, {step}, {window}");
        let q = |name: &str| format!("SELECT t, value FROM {name}({args}) ORDER BY 1, 2");
        let a = r.run(&q("prom_topk")).unwrap();
        let b = r.run(&q("prom_topk_sql")).unwrap();
        assert_eq!(a.row_count, b.row_count, "chunked {args}");
        for (x, y) in a.rows.iter().zip(b.rows.iter()) {
            let x: (i64, serde_json::Value) = serde_json::from_str(x.get()).unwrap();
            let y: (i64, serde_json::Value) = serde_json::from_str(y.get()).unwrap();
            assert_eq!(x.0, y.0, "chunked {args}");
            match (x.1.as_f64(), y.1.as_f64()) {
                (Some(p), Some(q)) => assert!((p - q).abs() <= 1e-9 * q.abs().max(1e-12), "chunked {args}"),
                _ => assert_eq!(x.1, y.1, "chunked {args}"),
            }
        }
        assert!(a.row_count > 1000, "{}", a.row_count);
    }
    std::env::set_var("PGAI_TOPK_DEN_BUDGET", "50331648");
    std::env::set_var("PGAI_TOPK_CHUNK_ALWAYS", "0");

    // topk(k, sum by (job) (increase)) in Rust vs the same in SQL over the oracle's increase
    let (start, end) = (day0 + 5 * h, day0 + 70 * h);
    let rs = r
        .run(&format!(
            "SELECT labels['i'] AS g, t, value FROM prom_topk(3, 'increase', 'r_total', [], {start}, {end}, {h}, {h}, p_by := ['i']) ORDER BY 2, 3 DESC, 1"
        ))
        .unwrap();
    let sq = r
        .run(&format!(
            "SELECT labels['i'] AS g, t, sum(value) AS value FROM prom_increase_sql('r_total', [], {start}, {end}, {h}, {h}) \
             WHERE NOT isnan(value) GROUP BY 1, 2 \
             QUALIFY row_number() OVER (PARTITION BY t ORDER BY sum(value) DESC, labels['i']) <= 3 ORDER BY 2, 3 DESC, 1"
        ))
        .unwrap();
    assert_eq!(rs.row_count, sq.row_count);
    for (x, y) in rs.rows.iter().zip(sq.rows.iter()) {
        let x: (String, i64, f64) = serde_json::from_str(x.get()).unwrap();
        let y: (String, i64, f64) = serde_json::from_str(y.get()).unwrap();
        assert_eq!((&x.0, x.1), (&y.0, y.1));
        assert!((x.2 - y.2).abs() <= 1e-9 * y.2.abs().max(1e-12), "{x:?} vs {y:?}");
    }
}

/// The 1 d tier (month files built from the 1 h rollups): served when the grid is on day
/// boundaries, with the 1 h tier and raw after it; equal to raw. 40 days across a month end,
/// compacted in two phases (the month file is built, then extended from itself), then a late
/// sample recompacts a day (the month file is stale, not served, and rebuilt).
#[test]
fn rollup_1d_tier_matches_raw() {
    use pgai_mon::eval::rollup::KIND_1D;
    let _knobs = KNOBS.lock().unwrap_or_else(|e| e.into_inner());
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
    let day0 = pgai_mon::store::chrono_free::days_from_civil(2026, 9, 20) * day;
    let ndays = 40;
    let mut defs = vec![];
    let mut samples = vec![];
    let mut seed = 23u64;
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
        let aligned = s % 3 == 0; // exact 10 min grid: samples at every midnight
        let mut t = day0 + if aligned { 0 } else { (rnd() % 600_000) as i64 };
        let mut v = (rnd() % 100) as f64;
        let mut k = 0;
        while t < day0 + ndays * day {
            v += (rnd() % 20) as f64;
            if k % 800 == 400 {
                v = 1.0; // reset
            }
            if s == 4 && k == 1000 {
                t += 2 * day + 7_200_000; // a gap of two days
            }
            let value = if k % 333 == 50 || (s == 5 && k == 2000) {
                stale_nan()
            } else if k % 517 == 60 {
                f64::NAN
            } else {
                v
            };
            samples.push(Sample {
                series_id: id,
                ts: t,
                value,
            });
            if s == 5 && k == 2000 {
                t += 3 * day; // went away (marker), back three days later
            }
            t += if aligned {
                600_000
            } else {
                600_000 + (rnd() % 300_000) as i64 - 150_000
            };
            k += 1;
        }
    }
    // e_total: 20-30 h apart, markers: with a 1 d step a bare selector looks back maxPrev(si) > 1 d
    for s in 0..3u64 {
        let labels = vec![("i".to_string(), s.to_string())];
        let id = series_id("e_total", &labels);
        defs.push((id, "e_total", labels));
        let mut t = day0 + (rnd() % 86_400_000) as i64;
        let mut k = 0;
        while t < day0 + ndays * day {
            let value = if (k + s) % 4 == 1 {
                stale_nan()
            } else {
                (rnd() % 1000) as f64
            };
            samples.push(Sample {
                series_id: id,
                ts: t,
                value,
            });
            t += 72_000_000 + (rnd() % 36_000_000) as i64;
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
    let d_mid = day_of(day0) + 20;
    let n1d = |core: &StoreCore| -> Vec<pgai_mon::catalog::FileEntry> {
        core.shared.catalog.lock().unwrap().files_of(KIND_1D).cloned().collect()
    };
    // phase 1: days [0, 20); phase 2: the rest
    for (lo, hi) in [(i64::MIN, d_mid), (d_mid, i64::MAX)] {
        core.append(Record::Series(series(lo, hi))).unwrap();
        let part: Vec<Sample> = samples
            .iter()
            .filter(|x| day_of(x.ts) >= lo && day_of(x.ts) < hi)
            .cloned()
            .collect();
        for chunk in part.chunks(50_000) {
            core.append(Record::Samples(chunk.to_vec())).unwrap();
            core.flush().unwrap();
        }
        core.compactor().unwrap().run_once(now_ms(), true).unwrap();
        if lo == i64::MIN {
            // September from day 0 (its last bucket ends at Oct 1 00:00); October up to day 18's
            // bucket (day 19's needs day 20's midnight sample)
            let mut m: Vec<(i64, i64)> = n1d(&core).iter().map(|f| (f.min_ts, f.max_ts)).collect();
            m.sort();
            assert_eq!(m, vec![(day0, day0 + 11 * day), (day0 + 11 * day, day0 + 19 * day)]);
        }
    }
    let months = n1d(&core);
    assert_eq!(months.len(), 2, "one file per month: {months:?}");
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
    let h = 3_600_000i64;
    // e0: the 1 d tier ends a day earlier (day 39's bucket needs day 40), then 1 h, then raw
    let (s0, e0) = (day0 + 3 * day, day0 + ndays * day);
    let check = |r: &Reader, cases: &[(i64, i64, i64)]| -> (usize, usize) {
        let (mut n, mut exact) = (0, 0);
        for &(start, end, step) in cases {
            let tier = if start % day == 0 && step % day == 0 { 1 } else { 0 };
            let mut one = |call: String, oracle: String| {
                let (a, b) = parity(r, &call, &oracle, Some(tier));
                n += a;
                exact += b;
            };
            for f in [
                "increase",
                "rate",
                "delta",
                "increase_pure",
                "last_over_time",
                "first_over_time",
                "tlast_over_time",
            ] {
                for w in [step, 7 * day] {
                    let args = format!("'d_total', [], {start}, {end}, {step}, {w}");
                    one(format!("prom_{f}({args})"), format!("prom_{f}_sql({args})"));
                }
            }
            // last_over_time / tlast_over_time need only an aligned grid: any window, here
            // shorter than a bucket and not a multiple of one
            for f in ["last_over_time", "tlast_over_time"] {
                for w in [600_000, 5_400_000] {
                    let args = format!("'d_total', [], {start}, {end}, {step}, {w}");
                    one(format!("prom_{f}({args})"), format!("prom_{f}_sql({args})"));
                }
            }
            for f in ["irate", "idelta"] {
                let args = format!("'d_total', [], {start}, {end}, {step}, {}", step + 5_000);
                one(format!("prom_{f}({args})"), format!("prom_{f}_sql({args})"));
            }
            for m in ["d_total", "e_total"] {
                for f in ["last", "timestamp"] {
                    let args = format!("'{m}', [], {start}, {end}, {step}");
                    one(format!("prom_{f}({args})"), format!("prom_{f}_sql({args})"));
                }
            }
            for f in ["sum", "min", "max", "avg", "count"] {
                for w in [0, 7 * day] {
                    let args = format!("'d_total', [], {start}, {end}, {step}, {w}");
                    one(
                        format!("prom_{f}_over_time({args})"),
                        format!("prom_{f}_over_time_sql({args})"),
                    );
                }
            }
        }
        (n, exact)
    };
    let cases = [
        (s0, e0, day),                     // 1 d tier, then 1 h, then raw
        (s0 + day, e0 - 3 * day, 2 * day), // inside the 1 d tier
        (s0 + 6 * h, e0 + 6 * h, day),     // hour-aligned only: the 1 h tier
    ];
    let (n, exact) = check(&r, &cases);
    assert!(n > 2_000, "{n}");
    eprintln!("parity 1 d tier: {n} rows, {exact} bit-identical to raw");
    assert_eq!(n, exact, "1 d tier on integer-valued series: not bit-identical to raw");

    // a late sample recompacts a day: its newer 1 h rollup makes October's file stale (not
    // served), and the same compaction run rebuilds it
    let late = day0 + 15 * day + 12 * h + 30_000;
    let id1 = series_id("d_total", &[("i".to_string(), "1".to_string())]);
    core.append(Record::Samples(vec![Sample {
        series_id: id1,
        ts: late,
        value: 1e6,
    }]))
    .unwrap();
    core.flush().unwrap();
    core.compactor().unwrap().run_once(now_ms(), true).unwrap();
    let after = n1d(&core);
    assert_eq!(after.len(), 2);
    let oct = |v: &[pgai_mon::catalog::FileEntry]| v.iter().find(|f| f.day == day_of(day0) + 11).cloned();
    let (a, b) = (oct(&months).unwrap(), oct(&after).unwrap());
    assert_ne!(a.path, b.path, "October's file rebuilt");
    assert_eq!((a.min_ts, a.max_ts), (b.min_ts, b.max_ts));
    r.refresh().ok();
    let (n, exact) = check(&r, &cases[..1]);
    eprintln!("parity 1 d tier after a rebuild: {n} rows, {exact} bit-identical to raw");
    assert!(n > 500 && n == exact, "{n} {exact}");
}

/// sum/avg_over_time from rollups add per bucket, then across buckets, so on non-integer values
/// they differ from raw's left-to-right sum in the last bits. Bounded here, on the 1 h and 1 d
/// tiers: positive values over 12 decades (relative error), and mixed signs (error against the
/// sum of magnitudes, since a cancelling sum has no useful relative bound).
#[test]
fn rollup_float_sums_are_close_to_raw() {
    let _knobs = KNOBS.lock().unwrap_or_else(|e| e.into_inner());
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
    let day0 = pgai_mon::store::chrono_free::days_from_civil(2026, 8, 20) * day;
    let ndays = 35;
    let mut seed = 99u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 // [0, 1)
    };
    let (mut defs, mut samples) = (vec![], vec![]);
    for s in 0..4u64 {
        let labels = vec![("i".to_string(), s.to_string())];
        let ids = ["g", "gm", "ga"].map(|m| series_id(m, &labels));
        let mut t = day0 + (rnd() * 600_000.0) as i64;
        while t < day0 + ndays * day {
            let g = (rnd() * 28.0 - 7.0).exp(); // 1e-3 .. 1e9
            let gm = (rnd() - 0.5) * 2e6 + rnd();
            for (id, v) in ids.iter().zip([g, gm, gm.abs()]) {
                samples.push(Sample {
                    series_id: *id,
                    ts: t,
                    value: v,
                });
            }
            t += 600_000 + (rnd() * 300_000.0) as i64 - 150_000;
        }
        for (m, id) in ["g", "gm", "ga"].iter().zip(ids) {
            for dd in 0..ndays {
                defs.push(SeriesDef {
                    series_id: id,
                    metric: m.to_string(),
                    labels: labels.clone(),
                    first_seen: day0 + dd * day,
                });
            }
        }
    }
    core.append(Record::Series(defs)).unwrap();
    for chunk in samples.chunks(50_000) {
        core.append(Record::Samples(chunk.to_vec())).unwrap();
        core.flush().unwrap();
    }
    core.compactor().unwrap().run_once(now_ms(), true).unwrap();
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
    // (label i, t) -> value; raw: the Rust raw path
    let vals = |call: &str, raw: bool| -> std::collections::BTreeMap<(String, i64), f64> {
        if raw {
            std::env::set_var("PGAI_NO_ROLLUP", "1");
        }
        let res = r.run(&format!("SELECT labels['i'], t, value FROM {call}"));
        std::env::remove_var("PGAI_NO_ROLLUP");
        let res = res.unwrap_or_else(|e| panic!("{call}: {e:?}"));
        res.rows
            .iter()
            .map(|x| {
                let (i, t, v): (String, i64, f64) = serde_json::from_str(x.get()).unwrap();
                ((i, t), v)
            })
            .collect()
    };
    let h = 3_600_000i64;
    let (s0, e0) = (day0 + 2 * day, day0 + (ndays - 1) * day);
    for (tier, step, ws) in [(0usize, h, [0, 6 * h]), (1, day, [0, 7 * day])] {
        let (mut rel, mut scaled, mut n, mut exact) = (0f64, 0f64, 0, 0);
        for w in ws {
            for f in ["sum", "avg"] {
                let call = |m: &str| format!("prom_{f}_over_time('{m}', [], {s0}, {e0}, {step}, {w})");
                let before = pgai_mon::eval::rollup::SERVED[tier].load(std::sync::atomic::Ordering::Relaxed);
                let (a, b) = (vals(&call("g"), false), vals(&call("g"), true));
                assert!(pgai_mon::eval::rollup::SERVED[tier].load(std::sync::atomic::Ordering::Relaxed) > before);
                assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
                for (k, x) in &a {
                    let y = b[k];
                    rel = rel.max((x - y).abs() / y.abs());
                    exact += (x.to_bits() == y.to_bits()) as usize;
                    n += 1;
                }
                let (a, b) = (vals(&call("gm"), false), vals(&call("gm"), true));
                let mag = vals(&format!("prom_{f}_over_time('ga', [], {s0}, {e0}, {step}, {w})"), true);
                assert_eq!(a.len(), b.len());
                for (k, x) in &a {
                    scaled = scaled.max((x - b[k]).abs() / mag[k]);
                    exact += (x.to_bits() == b[k].to_bits()) as usize;
                    n += 1;
                }
            }
        }
        eprintln!(
            "float sums, {} tier: {n} points, {exact} bit-identical, max relative error {rel:e} (positive), \
             max error / sum of magnitudes {scaled:e} (mixed signs)",
            ["1 h", "1 d"][tier]
        );
        assert!(n > 1_000, "{n}");
        assert!(rel <= 1e-12 && scaled <= 1e-12, "{rel:e} {scaled:e}");
    }
}

/// Piece 4: a selection split across worker threads (eval/par.rs) feeds every sink in id order,
/// so results are bit-identical to one thread: plain and ratio top-k, aggregation, bare eval,
/// raw and chunked paths.
#[test]
fn parallel_evaluation_is_bit_identical() {
    let _knobs = KNOBS.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    let core = build_store(d.path());
    let r = Reader::new(
        core.shared.clone(),
        HttpCfg::default(),
        EMBEDDED_VIEWS_SQL.into(),
        vec![],
    )
    .unwrap();
    let t0 = 1_794_000_000_000i64;
    let (s, e) = (t0 + 600_000, t0 + 10_000_000);
    let mut qs = vec![];
    for f in ["irate", "rate", "avg_over_time"] {
        qs.push(format!("SELECT series_id, t, value FROM prom_topk(3, '{f}', 'm_total', [], {s}, {e}, 30000, 35000, p_den_metric := 'm2_total') ORDER BY 1, 2"));
        qs.push(format!("SELECT series_id, t, value FROM prom_topk(2, '{f}', 'm_total', [], {s}, {e}, 60000, 120000, p_bottom := true) ORDER BY 1, 2"));
        qs.push(format!("SELECT labels::VARCHAR, t, value FROM prom_agg('sum', '{f}', 'm_total', [], {s}, {e}, 30000, 35000, p_by := ['job']) ORDER BY 1, 2"));
        qs.push(format!(
            "SELECT series_id, t, value FROM prom_{f}('m_total', [], {s}, {e}, 30000, 35000) ORDER BY 1, 2"
        ));
    }
    let run_all = |r: &Reader| -> Vec<Vec<String>> {
        qs.iter()
            .map(|q| {
                r.run(q)
                    .unwrap_or_else(|x| panic!("{q}: {x:?}"))
                    .rows
                    .iter()
                    .map(|x| x.get().to_string())
                    .collect()
            })
            .collect()
    };
    let one = run_all(&r);
    std::env::set_var("PGAI_EVAL_PAR_MIN", "5");
    std::env::set_var("PGAI_EVAL_FORCE_WORKERS", "3");
    let par = run_all(&r);
    std::env::set_var("PGAI_TOPK_DEN_BUDGET", "1000"); // the chunked ratio path too
    let chunked_par = run_all(&r);
    std::env::remove_var("PGAI_EVAL_FORCE_WORKERS");
    let chunked_one = run_all(&r);
    std::env::remove_var("PGAI_EVAL_PAR_MIN");
    std::env::remove_var("PGAI_TOPK_DEN_BUDGET");
    assert!(pgai_mon::eval::par::PAR_QUERIES.load(std::sync::atomic::Ordering::Relaxed) > 0);
    for (i, q) in qs.iter().enumerate() {
        assert!(!one[i].is_empty(), "{q}");
        assert_eq!(one[i], par[i], "{q}");
        assert_eq!(chunked_one[i], chunked_par[i], "chunked: {q}");
    }
}

/// Piece 5: `?format=columnar` decodes to exactly the rows of the default format (same value
/// texts, same order), with repeated values (label maps) dictionary-encoded.
#[test]
fn columnar_format_decodes_to_the_row_format() {
    let d = tempfile::tempdir().unwrap();
    let core = build_store(d.path());
    let r = Reader::new(
        core.shared.clone(),
        HttpCfg::default(),
        EMBEDDED_VIEWS_SQL.into(),
        vec![],
    )
    .unwrap();
    let t0 = 1_794_000_000_000i64;
    let (s, e) = (t0 + 600_000, t0 + 10_000_000);
    let qs = [
        format!("SELECT labels, t, value FROM prom_rate('m_total', [], {s}, {e}, 30000, 35000) ORDER BY t, labels::VARCHAR"),
        format!("SELECT labels::VARCHAR AS l, t, value FROM prom_agg('sum', 'irate', 'm_total', [], {s}, {e}, 30000, 35000, p_by := ['job']) ORDER BY 1, 2"),
        "SELECT i, i % 3 AS k, 'x' || (i % 2) AS s, CASE WHEN i % 4 = 0 THEN NULL ELSE i / 7.0 END AS f, \
         'NaN'::DOUBLE AS n FROM range(50) t(i) ORDER BY i".to_string(),
        "SELECT 1 AS one WHERE false".to_string(),
        "SELECT [1, 2] AS l, {'a': 1} AS st".to_string(),
    ];
    for q in &qs {
        let rows = r.run(q).unwrap();
        let col = r.run_fmt(q, Fmt::Columnar).unwrap();
        let rj: serde_json::Value = serde_json::to_value(&rows).unwrap();
        let cj: serde_json::Value = serde_json::from_str(&serde_json::to_string(&col).unwrap()).unwrap();
        assert_eq!(cj["format"], "columnar", "{q}");
        assert!(cj.get("rows").is_none() && rj.get("format").is_none(), "{q}");
        for k in ["columns", "types", "row_count", "truncated"] {
            assert_eq!(cj[k], rj[k], "{q}: {k}");
        }
        let n = cj["row_count"].as_u64().unwrap() as usize;
        let data = cj["data"].as_array().unwrap();
        assert_eq!(data.len(), cj["columns"].as_array().unwrap().len(), "{q}");
        let cols: Vec<Vec<serde_json::Value>> = data
            .iter()
            .map(|c| match c {
                serde_json::Value::Array(a) => a.clone(),
                c => {
                    let dict = c["dict"].as_array().unwrap();
                    c["idx"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|i| dict[i.as_u64().unwrap() as usize].clone())
                        .collect()
                }
            })
            .collect();
        let decoded: Vec<serde_json::Value> = (0..n)
            .map(|i| serde_json::Value::Array(cols.iter().map(|c| c[i].clone()).collect()))
            .collect();
        assert_eq!(serde_json::Value::Array(decoded), rj["rows"], "{q}");
    }
    // the label-map column of a range query is dictionary-encoded
    let col = r.run_fmt(&qs[0], Fmt::Columnar).unwrap();
    let cj: serde_json::Value = serde_json::from_str(&serde_json::to_string(&col).unwrap()).unwrap();
    assert!(cj["data"][0].get("dict").is_some(), "labels dict-encoded");
}

/// `?format=objects`: each row is `{column: value}` with exactly the row format's values, in
/// the same order, and the top-level keys of the row format plus `"format":"objects"`.
#[test]
fn objects_format_matches_the_row_format_by_column_name() {
    let d = tempfile::tempdir().unwrap();
    let core = build_store(d.path());
    let r = Reader::new(
        core.shared.clone(),
        HttpCfg::default(),
        EMBEDDED_VIEWS_SQL.into(),
        vec![],
    )
    .unwrap();
    let t0 = 1_794_000_000_000i64;
    let (s, e) = (t0 + 600_000, t0 + 10_000_000);
    let qs = [
        format!(
            "SELECT labels, t, value FROM prom_rate('m_total', [], {s}, {e}, 30000, 35000) ORDER BY t, labels::VARCHAR"
        ),
        // names that need escaping, a NULL, NaN, nested values
        "SELECT i AS \"a \"\"q\"\" \\\\ b\", CASE WHEN i % 4 = 0 THEN NULL ELSE i / 7.0 END AS f, \
         'NaN'::DOUBLE AS n, [i, 2] AS l, {'x': i} AS st, 'é\n' AS s FROM range(7) t(i) ORDER BY i"
            .to_string(),
        "SELECT 1 AS one WHERE false".to_string(),
    ];
    for q in &qs {
        let rows = r.run(q).unwrap();
        let obj = r.run_fmt(q, Fmt::Objects).unwrap();
        let rj: serde_json::Value = serde_json::to_value(&rows).unwrap();
        let oj: serde_json::Value = serde_json::from_str(&serde_json::to_string(&obj).unwrap()).unwrap();
        assert_eq!(oj["format"], "objects", "{q}");
        for k in ["columns", "types", "row_count", "truncated"] {
            assert_eq!(oj[k], rj[k], "{q}: {k}");
        }
        let cols: Vec<String> = oj["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_string())
            .collect();
        let orows = oj["rows"].as_array().unwrap();
        let rrows = rj["rows"].as_array().unwrap();
        assert_eq!(orows.len(), rrows.len(), "{q}");
        for (o, a) in orows.iter().zip(rrows) {
            let o = o.as_object().unwrap();
            assert_eq!(o.keys().count(), cols.len(), "{q}");
            for (j, c) in cols.iter().enumerate() {
                assert_eq!(o[c], a[j], "{q}: column {c}");
            }
        }
    }
}

/// The store read through the object-store path (an InMemory bucket) gives exactly what the
/// local files give: series selection, raw scans, rollup-served evaluation and the registry
/// tail. GETs stay few (planned and coalesced, not one per column chunk).
#[test]
fn an_object_store_copy_reads_exactly_like_the_local_store() {
    use object_store::{memory::InMemory, ObjectStoreExt, PutPayload};
    use pgai_mon::eval::{index, rollup, scan::Scan, scan::SeriesRows, EvalCtx};
    use pgai_mon::objstore;
    use std::sync::atomic::Ordering;
    let d = tempfile::tempdir().unwrap();
    let core = rollup_store(d.path());
    let cat = core.shared.catalog.lock().unwrap().clone();
    // copy the store into the bucket: catalog.json and every file it lists
    let store = std::sync::Arc::new(InMemory::new());
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let mut names: Vec<String> = cat.files.iter().map(|f| f.path.clone()).collect();
    names.push("catalog.json".into());
    for n in &names {
        let b = std::fs::read(d.path().join(n)).unwrap();
        rt.block_on(store.put(&object_store::path::Path::from(format!("st/{n}")), PutPayload::from(b)))
            .unwrap();
    }
    let remote = objstore::register("mem://parity/st", store, objstore::S3Cfg::default()).unwrap();
    let rroot = std::path::PathBuf::from("mem://parity/st");
    let rcat = pgai_mon::catalog::Catalog::load_or_new(&rroot, "").unwrap();
    assert_eq!(rcat.files, cat.files, "catalog.json read from the bucket");
    let local = EvalCtx::from_catalog(d.path().to_path_buf(), &cat);
    let rem = EvalCtx::from_catalog(rroot, &rcat);
    let day0 = 1_794_009_600_000i64 - 86_400_000;
    let h = 3_600_000i64;
    let (t0, t1) = (day0, day0 + 72 * h);

    let sel = |c: &EvalCtx| {
        let s = index::select_series(c, &["r_total"], "[]", t0, t1).unwrap();
        s.entries.iter().map(|(id, _, l)| (*id, s.text(l))).collect::<Vec<_>>()
    };
    let ids_l = sel(&local);
    assert_eq!(ids_l.len(), 24);
    assert_eq!(sel(&rem), ids_l);
    let ids: Vec<u64> = ids_l.iter().map(|x| x.0).collect();

    let raw = |c: &EvalCtx| {
        let mut sc = Scan::open(&c.root, &c.files, ids.clone(), t0, t1).unwrap();
        let mut rows = SeriesRows::default();
        let mut out = Vec::new();
        while sc.next_series(&mut rows).unwrap() {
            out.push((
                rows.id,
                rows.ts.clone(),
                rows.v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                rows.stale.clone(),
            ));
        }
        out
    };
    let raw_l = raw(&local);
    assert_eq!(raw_l.len(), 24);
    assert_eq!(raw(&rem), raw_l);

    let g = Grid {
        start: day0 + 5 * h,
        end: day0 + 70 * h,
        step: h,
        window: 6 * h,
        phi: 0.0,
    };
    let ru = |c: &EvalCtx| {
        let plan = rollup::plan(c, Func::Increase, &g).expect("rollup-served");
        let mut out = Vec::new();
        rollup::eval(c, Func::Increase, &g, &ids.clone().into(), plan, |id, t, v| {
            out.push((id, t, v.to_bits()))
        })
        .unwrap();
        out
    };
    let ru_l = ru(&local);
    assert!(ru_l.len() > 1000);
    assert_eq!(ru(&rem), ru_l);

    let tail = |c: &EvalCtx| pgai_mon::registry::tail_new(c).unwrap();
    assert_eq!(tail(&rem), tail(&local));

    let gets = remote.stats.gets.load(Ordering::Relaxed);
    let files = cat.files.len() as u64;
    // per file: a footer and a few coalesced data GETs; never one per column chunk
    assert!(gets > 0 && gets <= 4 * files + 1, "{gets} GETs for {files} files");
    assert_eq!(remote.stats.errors.load(Ordering::Relaxed), 0);
}

/// Copy a store (catalog.json and every file it lists) into a fresh InMemory bucket registered
/// as `url` (an s3://-like root).
fn to_bucket(dir: &std::path::Path, url: &str) -> std::sync::Arc<pgai_mon::objstore::Remote> {
    to_bucket_with(dir, url, |_| {})
}

/// Same, with catalog.json edited by `edit` before the upload (a tampered catalog). Objects keep
/// their real names and contents.
fn to_bucket_with(
    dir: &std::path::Path,
    url: &str,
    edit: impl FnOnce(&mut pgai_mon::catalog::Catalog),
) -> std::sync::Arc<pgai_mon::objstore::Remote> {
    use object_store::{memory::InMemory, ObjectStoreExt, PutPayload};
    let mut cat = pgai_mon::catalog::Catalog::load_or_new(dir, "").unwrap();
    let store = std::sync::Arc::new(InMemory::new());
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let put = |n: &str, b: Vec<u8>| {
        rt.block_on(store.put(&object_store::path::Path::from(format!("st/{n}")), PutPayload::from(b)))
            .unwrap();
    };
    for f in &cat.files {
        put(&f.path, std::fs::read(dir.join(&f.path)).unwrap());
    }
    edit(&mut cat);
    put("catalog.json", serde_json::to_vec(&cat).unwrap());
    pgai_mon::objstore::register(url, store.clone(), pgai_mon::objstore::S3Cfg::default()).unwrap()
}

fn sql_rows(r: &Reader, sql: &str) -> Vec<String> {
    let res = r.run(sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    res.rows.iter().map(|x| x.get().to_string()).collect()
}

/// `serve --store s3://...` end to end through the SQL path (Reader on an InMemory bucket, no
/// httpfs): rollup-served prom_last (1 h tier) and a 1 d-tier increase give exactly the local
/// store's rows; the `series` view reads the local mirror of the registry and tail; the raw
/// samples view raises. GETs stay bounded.
#[test]
fn sql_over_an_object_store_matches_the_local_store() {
    use pgai_mon::eval::rollup::{KIND_1D, SERVED};
    use std::sync::atomic::Ordering;
    let _knobs = KNOBS.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    let core = rollup_store(d.path());
    let n1d = core.shared.catalog.lock().unwrap().files_of(KIND_1D).count();
    assert!(n1d >= 1, "the 1 d tier is built");
    let nfiles = core.shared.catalog.lock().unwrap().files.len() as u64;
    let views = format!("{EMBEDDED_VIEWS_SQL}\n;\n{MACROS}");
    let cfg = HttpCfg {
        memory_limit: "1GB".into(),
        ..Default::default()
    };
    let local = Reader::new(core.shared.clone(), cfg.clone(), views.clone(), vec![]).unwrap();
    let remote = to_bucket(d.path(), "mem://sqlpath/st");
    let rshared = pgai_mon::store::Shared::open_read_only(std::path::Path::new("mem://sqlpath/st")).unwrap();
    let rem = Reader::new(rshared, cfg, views, vec![]).unwrap();

    let day0 = 1_794_009_600_000i64 - 86_400_000;
    let (h, day) = (3_600_000i64, 86_400_000i64);
    let cases = [
        (
            format!("SELECT series_id, t, value FROM prom_last('r_total', [], {}, {}, {h}) ORDER BY 1, 2", day0 + 5 * h, day0 + 70 * h),
            Some(0),
        ),
        (
            format!(
                "SELECT series_id, t, value FROM prom_increase('r_total', [], {}, {}, {day}, {day}) ORDER BY 1, 2",
                day0 + day,
                day0 + 3 * day
            ),
            Some(1),
        ),
        (
            format!("SELECT series_id, t, value FROM prom_rate('r_total', [['i', '=~', '1.*']], {}, {}, 300000, 600000) ORDER BY 1, 2", day0 + 30 * h, day0 + 34 * h),
            None,
        ),
        (
            format!(
                "SELECT series_id, labels, ts, value, stale FROM pgai_samples('r_total', [['i', '=~', '1.*']], {}, {}) ORDER BY 1, 3",
                day0 + 30 * h,
                day0 + 40 * h
            ),
            None,
        ),
        ("SELECT series_id, metric, labels FROM series ORDER BY 1".to_string(), None),
        ("SELECT metric, type FROM metric_dict ORDER BY 1".to_string(), None),
    ];
    let g0 = remote.stats.gets.load(Ordering::Relaxed);
    for (q, tier) in &cases {
        let before = tier.map(|t| SERVED[t].load(Ordering::Relaxed));
        let a = sql_rows(&rem, q);
        if let (Some(t), Some(b)) = (tier, before) {
            assert!(SERVED[*t].load(Ordering::Relaxed) > b, "{q}: not served from tier {t}");
        }
        let b = sql_rows(&local, q);
        assert!(!b.is_empty(), "{q}");
        assert_eq!(a, b, "{q}");
    }
    // per-query object-store stats ride in the response
    let (g, by) = (
        remote.stats.gets.load(Ordering::Relaxed),
        remote.stats.bytes.load(Ordering::Relaxed),
    );
    let r = rem.run(&cases[2].0).unwrap();
    let s3 = r.s3.clone().expect("s3 stats on an object store");
    assert!(s3.gets <= remote.stats.gets.load(Ordering::Relaxed) - g || !s3.exclusive);
    assert!(s3.bytes >= remote.stats.bytes.load(Ordering::Relaxed) - by || !s3.exclusive);
    let body = serde_json::to_value(&r).unwrap();
    assert_eq!(body["s3"]["gets"], serde_json::json!(s3.gets), "{body}");
    let series = sql_rows(&rem, "SELECT count(*) FROM series WHERE metric = 'r_total'");
    assert_eq!(series, vec!["[24]".to_string()]);
    // raw samples are not mirrored (prom_* read them from the bucket): the view says what to
    // use, also under a filter, which over constant columns would be answered with no rows
    for q in [
        "SELECT count(*) FROM samples",
        "SELECT count(*) FROM samples WHERE ts > now() - INTERVAL 1 HOUR",
        "SELECT ts, value FROM samples WHERE series_id = 1 AND NOT stale",
        "SELECT count(*) FROM metrics WHERE ts > now() - INTERVAL 1 HOUR",
    ] {
        let e = rem.run(q).expect_err("raw view on an object store");
        assert!(e.message.contains("pgai_samples"), "{q}: {e:?}");
    }
    let gets = remote.stats.gets.load(Ordering::Relaxed) - g0;
    assert!(
        gets <= 6 * nfiles,
        "{gets} GETs for {nfiles} files over {} queries",
        cases.len()
    );
    assert_eq!(remote.stats.errors.load(Ordering::Relaxed), 0);
    let h = rem.stats();
    eprintln!("{gets} GETs, {nfiles} files; mirror {}", h["s3_mirror"]);
    assert!(h["s3_mirror"]["files"].as_u64().unwrap() >= 1, "{h}");
    assert_eq!(h["s3_mirror"]["empty_kinds"], serde_json::json!(["samples"]), "{h}");
}

/// A slow copy of a remote store's files for the SQL views holds up neither the other queries
/// (they run on the views as they are) nor, past half the query timeout, the query that started
/// it: a kind not copied yet answers with an error, and the next queries continue the copy.
#[test]
fn a_slow_mirror_copy_does_not_block_queries() {
    use pgai_mon::objstore::TEST_DOWNLOAD_DELAY;
    let d = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(d.path(), &StoreCfg::default()).unwrap();
    // six flushes: six series files to copy
    for k in 0..6u64 {
        let labels = vec![("i".to_string(), k.to_string())];
        let series_id = series_id("c_total", &labels);
        let (ts, value) = (1_794_000_000_000 + k as i64, 1.0);
        let def = SeriesDef {
            series_id,
            metric: "c_total".into(),
            labels,
            first_seen: ts,
        };
        core.append(Record::Series(vec![def])).unwrap();
        core.append(Record::Samples(vec![Sample { series_id, ts, value }]))
            .unwrap();
        core.flush().unwrap();
    }
    let url = "mem://slowmirror/st";
    to_bucket(d.path(), url);
    let shared = pgai_mon::store::Shared::open_read_only(std::path::Path::new(url)).unwrap();
    // 400 ms a file, and the copies of one refresh stop after 800 ms: two files a refresh
    *TEST_DOWNLOAD_DELAY.lock().unwrap() = Some((url.into(), 400));
    let cfg = HttpCfg {
        timeout_ms: 1600,
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let rem = Reader::new(shared, cfg, format!("{EMBEDDED_VIEWS_SQL}\n;\n{MACROS}"), vec![]).unwrap();
    assert!(
        t.elapsed().as_millis() < 1600,
        "the first refresh took {:?}",
        t.elapsed()
    );
    let q = "SELECT count(*) FROM series";
    std::thread::scope(|s| {
        // this query's refresh copies for 800 ms; a second query does not wait for it
        let slow = s.spawn(|| rem.run(q));
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(sql_rows(&rem, "SELECT 42"), vec!["[42]".to_string()]);
        assert!(!slow.is_finished(), "the query that copies was still running");
        let e = slow.join().unwrap().expect_err("series is not copied yet");
        assert!(e.message.contains("still being copied"), "{e:?}");
    });
    let t = std::time::Instant::now();
    assert_eq!(
        sql_rows(&rem, q),
        vec!["[6]".to_string()],
        "the last two files are copied"
    );
    assert!(t.elapsed().as_millis() < 1600, "{:?}", t.elapsed());
    *TEST_DOWNLOAD_DELAY.lock().unwrap() = None;
}

/// An s3:// store adds nothing DuckDB can reach: local files outside the allowed (local)
/// directories, IMDS, the bucket URL, the chunk cache dir and the mirror's parent all fail with
/// a permission error, and configuration stays locked. Positive control: a mirror file reads.
#[test]
fn sql_over_an_object_store_stays_locked_down() {
    let d = tempfile::tempdir().unwrap();
    let core = build_store(d.path());
    let _remote = to_bucket(d.path(), "s3://locked/st");
    let shared = pgai_mon::store::Shared::open_read_only(std::path::Path::new("s3://locked/st")).unwrap();
    let cache = tempfile::tempdir().unwrap();
    let cache_dir = cache.path().canonicalize().unwrap();
    std::fs::write(cache_dir.join("0123456789abcdef-0-10"), b"chunk-data").unwrap();
    let mut cfg = HttpCfg {
        memory_limit: "1GB".into(),
        ..Default::default()
    };
    cfg.s3.cache_dir = Some(cache_dir.clone());
    let views = format!("{EMBEDDED_VIEWS_SQL}\n;\n{MACROS}");
    let r = Reader::new(shared, cfg.clone(), views.clone(), vec![]).unwrap();
    // nothing compacted: `series` is all tail (mirrored, filename mapped back to store paths)
    let local = Reader::new(core.shared.clone(), cfg, views, vec![]).unwrap();
    let q = "SELECT series_id, metric, labels FROM series ORDER BY 1";
    let rows = sql_rows(&r, q);
    assert!(rows.len() > 10, "{rows:?}");
    assert_eq!(rows, sql_rows(&local, q));
    let st = r.stats();
    let mirror = std::path::PathBuf::from(st["s3_mirror"]["dir"].as_str().unwrap());
    assert_eq!(mirror.parent(), Some(cache_dir.join("mirror").as_path()), "{st}");
    // positive control: the probes below fail for permission, not for a missing file
    let tail = core
        .shared
        .catalog
        .lock()
        .unwrap()
        .files_of("series")
        .next()
        .unwrap()
        .path
        .clone();
    let mf = mirror.join(&tail);
    assert!(mf.is_file());
    let ok = format!("SELECT count(*) FROM read_parquet('{}')", mf.display());
    assert!(!sql_rows(&r, &ok).is_empty());
    let probes = [
        "SELECT * FROM read_text('/etc/passwd')".to_string(),
        "SELECT * FROM read_csv('http://169.254.169.254/latest/meta-data/')".into(),
        "SELECT * FROM read_parquet('s3://locked/st/catalog.json')".into(),
        "SELECT * FROM read_text('s3://locked/st/catalog.json')".into(),
        "SELECT * FROM glob('/*')".into(),
        "SELECT * FROM read_text('/tmp/../etc/passwd')".into(),
        // the chunk cache (Rust only) and the mirror's parent
        format!(
            "SELECT * FROM read_text('{}/0123456789abcdef-0-10')",
            cache_dir.display()
        ),
        format!("SELECT * FROM glob('{}/*')", cache_dir.display()),
        format!(
            "SELECT * FROM read_text('{}/../../0123456789abcdef-0-10')",
            mirror.display()
        ),
        format!("SELECT * FROM glob('{}/../*')", mirror.display()),
        // the local store's own files (the remote reader must not reach them)
        format!(
            "SELECT * FROM read_parquet('{}')",
            d.path().canonicalize().unwrap().join(&tail).display()
        ),
    ];
    for q in &probes {
        let e = r.run(q).err().unwrap_or_else(|| panic!("{q}: allowed"));
        assert!(
            e.kind == "forbidden"
                && (e.message.contains("Permission Error")
                    || e.message.contains("disabled by configuration")
                    || e.message.contains("is not available here")),
            "{q}: {} {}",
            e.kind,
            e.message
        );
    }
    for q in [
        "SET enable_external_access=true",
        "SET allowed_directories=['/']",
        "RESET lock_configuration",
        "LOAD httpfs",
        "INSTALL httpfs",
        "SELECT 1; SET enable_external_access=true",
    ] {
        let e = r.run(q).err().unwrap_or_else(|| panic!("{q}: allowed"));
        assert!(
            e.kind == "not_read_only" || e.message.contains("lock_configuration"),
            "{q}: {} {}",
            e.kind,
            e.message
        );
    }
}

/// A tampered remote catalog: a file path that would leave the store root (`../`, absolute,
/// `a:b`, empty segment, backslash) refuses the whole catalog, before any reader joins it to the
/// root; an oversized catalog.json is refused unread; a file whose catalog
/// size is wrong or 0 is not mirrored.
#[test]
fn a_tampered_remote_catalog_is_refused() {
    use pgai_mon::store::Shared;
    use std::path::Path;
    let d = tempfile::tempdir().unwrap();
    let core = build_store(d.path());
    drop(core);
    for (i, bad) in [
        "../x.parquet",
        "/etc/passwd",
        "a:b/x.parquet",
        "series//x.parquet",
        "series\\x.parquet",
        "",
        "./x",
    ]
    .iter()
    .enumerate()
    {
        let url = format!("mem://bad{i}/st");
        let bad = bad.to_string();
        to_bucket_with(d.path(), &url, move |c| c.files[0].path = bad);
        let e = Shared::open_read_only(Path::new(&url)).err().expect("refused");
        assert!(format!("{e:#}").contains("not a plain relative path"), "{e:#}");
        let e = pgai_mon::catalog::Catalog::load_or_new(Path::new(&url), "").expect_err("refused");
        assert!(format!("{e:#}").contains("not a plain relative path"), "{e:#}");
    }
    // a catalog.json over 64 MB is refused before it is read
    {
        use object_store::{ObjectStoreExt, PutPayload};
        let store = std::sync::Arc::new(object_store::memory::InMemory::new());
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let big = vec![b' '; (pgai_mon::catalog::MAX_REMOTE_BYTES + 1) as usize];
        rt.block_on(store.put(
            &object_store::path::Path::from("st/catalog.json"),
            PutPayload::from(big),
        ))
        .unwrap();
        let r = pgai_mon::objstore::register("mem://big/st", store, Default::default()).unwrap();
        let e = Shared::open_read_only(Path::new("mem://big/st"))
            .err()
            .expect("refused");
        assert!(format!("{e:#}").contains("limit"), "{e:#}");
        assert!(
            r.stats.bytes.load(std::sync::atomic::Ordering::Relaxed) < 1 << 20,
            "body not read"
        );
    }
    // files that exist and are over the mirror budget: the view fails, it does not answer 0 rows
    {
        let d2 = tempfile::tempdir().unwrap();
        let mut core = build_store(d2.path());
        core.append(Record::QueryTexts(vec![QueryText {
            queryid: 1,
            dbname: "db1".into(),
            query_hash: 1,
            query: Some("select 1".into()),
            first_seen: 0,
            last_seen: 0,
            src: Default::default(),
        }]))
        .unwrap();
        core.flush().unwrap();
        drop(core);
        let url = "mem://overbudget/st";
        to_bucket_with(d2.path(), url, |c| {
            for f in c.files.iter_mut().filter(|f| f.kind == "query_texts") {
                f.bytes = u64::MAX / 2;
            }
        });
        let shared = Shared::open_read_only(Path::new(url)).unwrap();
        let views = format!("{EMBEDDED_VIEWS_SQL}\n;\n{MACROS}");
        let r = Reader::new(shared, HttpCfg::default(), views, vec![]).unwrap();
        for q in [
            "SELECT count(*) FROM query_texts",
            "SELECT query FROM query_texts WHERE queryid = 1",
            "SELECT count(*) FROM query_texts WHERE dbname = 'db1'",
        ] {
            let e = r.run(q).expect_err("not an empty answer");
            assert!(e.message.contains("mirror budget"), "{q}: {}", e.message);
        }
        // other views still answer
        assert!(!sql_rows(&r, "SELECT count(*) FROM series").is_empty());
    }
    // a catalog whose `bytes` lies (too small, too big, 0) about a series file: that copy
    // is refused, the view fails (it does not answer 0 rows), the mirror holds no such file,
    // and the error shows
    for (i, lie) in [1u64, u64::MAX / 2, 0].into_iter().enumerate() {
        let url = format!("mem://lie{i}/st");
        let remote = to_bucket_with(d.path(), &url, move |c| {
            for f in c.files.iter_mut().filter(|f| f.kind == "series") {
                f.bytes = if lie == 0 {
                    0
                } else if lie == 1 {
                    f.bytes - 1
                } else {
                    lie
                };
            }
        });
        let shared = Shared::open_read_only(Path::new(&url)).unwrap();
        let r = Reader::new(
            shared,
            HttpCfg {
                memory_limit: "1GB".into(),
                ..Default::default()
            },
            format!("{EMBEDDED_VIEWS_SQL}\n;\n{MACROS}"),
            vec![],
        )
        .unwrap();
        for q in [
            "SELECT count(*) FROM series",
            "SELECT labels FROM series WHERE metric = 'r_total'",
        ] {
            let e = r.run(q).expect_err("not an empty answer");
            assert!(
                e.message.contains("series is not available right now"),
                "{q}: {}",
                e.message
            );
        }
        let st = r.stats();
        let m = &st["s3_mirror"];
        if lie == 1 {
            // a size mismatch is caught on the first response, before the body is written
            assert!(
                m["error"].as_str().is_some_and(|e| e.contains("the catalog says")),
                "{st}"
            );
            assert!(remote.stats.errors.load(std::sync::atomic::Ordering::Relaxed) > 0);
        } else if lie == 0 {
            assert!(
                m["error"].as_str().is_some_and(|e| e.contains("catalog size 0")),
                "{st}"
            );
        } else {
            // over the budget: not fetched at all
            assert!(
                m["empty_kinds"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|k| k == "_raw_series_tail"),
                "{st}"
            );
        }
        assert_eq!(m["fetched_bytes"].as_u64(), Some(0), "{st}");
        let mdir = std::path::PathBuf::from(m["dir"].as_str().unwrap());
        let n = walk_files(&mdir);
        assert_eq!(n, 0, "nothing mirrored under {}", mdir.display());
    }
}

fn walk_files(p: &std::path::Path) -> usize {
    let Ok(rd) = std::fs::read_dir(p) else { return 0 };
    rd.flatten()
        .map(|e| if e.path().is_dir() { walk_files(&e.path()) } else { 1 })
        .sum()
}

/// `pgai_samples` (the Rust scan) returns exactly the rows of the `samples` view joined to
/// `series` for its metric, matchers and inclusive window: day files, a later flush file,
/// samples on the window's edges, NaN values and staleness markers.
#[test]
fn pgai_samples_matches_the_samples_view() {
    let d = tempfile::tempdir().unwrap();
    let mut core = rollup_store(d.path());
    let day0 = 1_794_009_600_000i64 - 86_400_000;
    let h = 3_600_000i64;
    // an uncompacted flush file on top of the compacted days (a series already in them)
    let labels = vec![("i".to_string(), "3".to_string())];
    let id = series_id("r_total", &labels);
    core.append(Record::Samples(
        (0..10)
            .map(|k| Sample {
                series_id: id,
                ts: day0 + 40 * h + 30_000 + k * 60_000,
                value: if k == 4 { stale_nan() } else { 1e9 + k as f64 },
            })
            .collect(),
    ))
    .unwrap();
    core.flush().unwrap();
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
    let mut total = 0;
    for (t0, t1) in [
        (day0 + 5 * h, day0 + 41 * h),
        (day0, day0 + 3 * 86_400_000),
        (day0 + 40 * h + 30_000, day0 + 40 * h + 30_000),
    ] {
        for (m, sqlm) in [
            ("[]", "true"),
            ("[['i', '=~', '1.*']]", "regexp_full_match(r.labels['i'], '1.*')"),
            ("[['i', '=', '3']]", "r.labels['i'] = '3'"),
        ] {
            let a = sql_rows(
                &r,
                &format!(
                    "SELECT series_id, epoch_ms(ts), value, stale, labels FROM pgai_samples('r_total', {m}, {t0}, {t1}) ORDER BY 1, 2, 4, 3"
                ),
            );
            let b = sql_rows(
                &r,
                &format!(
                    "SELECT s.series_id, epoch_ms(s.ts), s.value, s.stale, r.labels FROM samples s JOIN series r USING (series_id) \
                     WHERE r.metric = 'r_total' AND {sqlm} AND s.ts >= make_timestamptz({t0} * 1000) AND s.ts <= make_timestamptz({t1} * 1000) \
                     ORDER BY 1, 2, 4, 3"
                ),
            );
            assert_eq!(a.len(), b.len(), "{t0} {t1} {m}");
            assert_eq!(a, b, "{t0} {t1} {m}");
            total += a.len();
        }
    }
    let marks = sql_rows(&r, &format!("SELECT count(*) FILTER (stale), count(*) FILTER (isnan(value) AND NOT stale) FROM pgai_samples('r_total', [], {day0}, {})", day0 + 3 * 86_400_000));
    eprintln!("pgai_samples parity: {total} rows; markers, NaN: {marks:?}");
    assert!(total > 100_000, "{total}");
    assert_ne!(marks, vec!["[0,0]".to_string()]);
    assert!(r.run("SELECT * FROM pgai_samples('r_total', [], 10, 5)").is_err());
}
