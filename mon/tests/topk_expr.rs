//! prom_topk_expr (prom_topk_rs with `terms`: an expression over several metrics, one-to-one on
//! label sets) and prom_topk with fn 'last' against their SQL-macro oracles, through the exact,
//! lean and time-chunked paths.
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::model::*;
use pgai_mon::query::{Reader, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::StoreCore;

const MACROS: &str = include_str!("../sql/macros.sql");
const T0: i64 = 1_794_000_000_000;

/// m_total (40 counters with resets, gaps, NaN, staleness markers, jittered 30 s scrapes),
/// m2_total (the same label sets), m3_total (every third label set only: the others never have
/// all three terms), g_bytes (a gauge with ties: value = i % 4).
fn build_store(dir: &std::path::Path) -> StoreCore {
    let mut core = StoreCore::open(dir, &StoreCfg::default()).unwrap();
    let mut defs = vec![];
    let mut samples = vec![];
    let mut seed = 11u64;
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
            ("grp".to_string(), (s % 5).to_string()),
        ];
        let metrics: &[&str] = if s % 3 == 0 {
            &["m_total", "m2_total", "m3_total", "g_bytes"]
        } else {
            &["m_total", "m2_total", "g_bytes"]
        };
        for (mi, m) in metrics.iter().enumerate() {
            let id = series_id(m, &labels);
            defs.push(SeriesDef {
                series_id: id,
                metric: m.to_string(),
                labels: labels.clone(),
                first_seen: T0,
            });
            let mut v = (rnd() % 1000) as f64;
            let mut t = T0 + (rnd() % 30_000) as i64 + mi as i64 * 7;
            for k in 0..400 {
                t += 30_000 + (rnd() % 400) as i64 - 200;
                if k % 97 == 50 + mi {
                    t += 400_000; // a gap longer than the windows
                }
                if *m == "g_bytes" {
                    v = (s % 4) as f64 * 1000.0 + if k % 50 < 25 { 0.0 } else { (k % 7) as f64 };
                } else {
                    v += (rnd() % 50) as f64 + 0.25;
                    if k % 113 == 60 {
                        v = (rnd() % 5) as f64; // counter reset
                    }
                }
                let value = if k % 89 == 30 + mi {
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
    }
    core.append(Record::Series(defs)).unwrap();
    core.append(Record::Samples(samples)).unwrap();
    core.flush().unwrap();
    core
}

#[test]
fn topk_expr_and_last_match_sql_oracle() {
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
    let grids = [
        (T0 + 600_000, T0 + 10_000_000, 60_000, 65_000),
        (T0 + 1_000_000, T0 + 12_000_000, 300_000, 900_000),
        (T0 + 5_000_000, T0 + 5_000_000, 300_000, 0), // instant
    ];
    let terms = [
        // a - b
        "[{'m': 'm_total', 'n': 1.0, 'd': 0.0}, {'m': 'm2_total', 'n': -1.0, 'd': 0.0}]",
        // a + b + c (c on a third of the label sets)
        "[{'m': 'm_total', 'n': 1.0, 'd': 0.0}, {'m': 'm2_total', 'n': 1.0, 'd': 0.0}, {'m': 'm3_total', 'n': 1.0, 'd': 0.0}]",
        // a / (a + b)
        "[{'m': 'm_total', 'n': 1.0, 'd': 1.0}, {'m': 'm2_total', 'n': 0.0, 'd': 1.0}]",
        // b / (a + b + c)
        "[{'m': 'm2_total', 'n': 1.0, 'd': 1.0}, {'m': 'm_total', 'n': 0.0, 'd': 1.0}, {'m': 'm3_total', 'n': 0.0, 'd': 1.0}]",
        // one term
        "[{'m': 'm_total', 'n': 2.0, 'd': 0.0}]",
    ];
    let mut cases: Vec<(String, String)> = vec![];
    for (start, end, step, window) in grids {
        for m in ["[]", "[['job','=','a']]"] {
            for fn_ in ["irate", "rate"] {
                for (ti, t) in terms.iter().enumerate() {
                    for (k, extra) in [(3, ""), (2, ", p_bottom := true"), (3, ", p_by := ['grp']")] {
                        if (2..=3).contains(&ti) && extra.contains("p_by") {
                            continue; // sum by over a ratio: not supported
                        }
                        let args = format!("{k}, '{fn_}', {t}, {m}, {start}, {end}, {step}, {window}{extra}");
                        cases.push((format!("prom_topk_expr({args})"), format!("prom_topk_expr_sql({args})")));
                    }
                }
            }
            for bottom in ["false", "true"] {
                let args = format!("3, 'last', 'g_bytes', {m}, {start}, {end}, {step}, 0, p_bottom := {bottom}");
                cases.push((format!("prom_topk({args})"), format!("prom_topk_sql({args})")));
            }
        }
    }
    let mut checked = 0usize;
    let mut oracle: std::collections::HashMap<String, pgai_mon::query::QueryResult> = Default::default();
    let timing = std::env::var_os("TOPK_EXPR_TIMING").is_some();
    for path in ["full", "lean", "chunk", "leanchunk"] {
        let lean = path.starts_with("lean");
        let chunk = path.ends_with("chunk");
        // the knobs are read at bind time; this binary's only test sets them
        std::env::set_var("PGAI_TOPK_LEAN_ABOVE", if lean { "0" } else { "100000" });
        std::env::set_var("PGAI_TOPK_COMPACT_ABOVE", if lean { "50" } else { "300000" });
        std::env::set_var("PGAI_TOPK_DEN_BUDGET", if chunk { "1000" } else { "50331648" });
        // force chunks on every chunkable grid of >= 4 steps (irate, last; rate is never chunked
        // and checks the whole-grid fallback)
        std::env::set_var("PGAI_TOPK_CHUNK_ALWAYS", if chunk { "1" } else { "0" });
        for (rust, sql) in &cases {
            let q = |src: &str| format!("SELECT series_id, t, value, CAST(labels AS VARCHAR) FROM {src} ORDER BY 1, 2");
            let t = std::time::Instant::now();
            let a = r.run(&q(rust)).unwrap_or_else(|e| panic!("{rust}: {e:?}"));
            let ta = t.elapsed();
            let b = oracle
                .entry(sql.clone())
                .or_insert_with(|| r.run(&q(sql)).unwrap_or_else(|e| panic!("{sql}: {e:?}")));
            if timing {
                eprintln!("{path} rust {ta:?} total {:?}: {rust}", t.elapsed());
            }
            let tag = format!("{path}: {rust}");
            let rows = |x: &pgai_mon::query::QueryResult| -> Vec<(u64, i64, f64, String)> {
                x.rows
                    .iter()
                    .map(|r| {
                        let v: (u64, i64, serde_json::Value, String) = serde_json::from_str(r.get()).unwrap();
                        (v.0, v.1, v.2.as_f64().unwrap_or(f64::NAN), v.3)
                    })
                    .collect()
            };
            let (ra, rb) = (rows(&a), rows(b));
            assert_eq!(ra.len(), rb.len(), "{tag}");
            if chunk {
                // chunks restart counter corrections: values agree to rounding, and values equal
                // up to rounding may rank in another order: compare each step's sorted values
                let by_t = |rows: &[(u64, i64, f64, String)]| {
                    let mut m: std::collections::BTreeMap<i64, Vec<f64>> = Default::default();
                    for x in rows {
                        m.entry(x.1).or_default().push(x.2);
                    }
                    m.values_mut().for_each(|v| v.sort_by(|a, b| a.total_cmp(b)));
                    m
                };
                let (ma, mb) = (by_t(&ra), by_t(&rb));
                assert_eq!(ma.keys().collect::<Vec<_>>(), mb.keys().collect::<Vec<_>>(), "{tag}");
                for (va, vb) in ma.values().zip(mb.values()) {
                    assert_eq!(va.len(), vb.len(), "{tag}");
                    for (x, y) in va.iter().zip(vb) {
                        assert!((x - y).abs() <= 1e-9 * y.abs().max(1e-12), "{tag}: {x} vs {y}");
                    }
                }
            } else {
                for (x, y) in ra.iter().zip(&rb) {
                    assert_eq!((x.0, x.1, &x.3), (y.0, y.1, &y.3), "{tag}");
                    assert!(
                        (x.2 - y.2).abs() <= 1e-9 * y.2.abs().max(1e-12),
                        "{tag}: {x:?} vs {y:?}"
                    );
                }
            }
            checked += ra.len();
        }
    }
    assert!(checked > 10_000, "{checked}");
    // a label set lacking a term never appears: a + b + c only ranks every third set
    let n = r
        .run(&format!(
            "SELECT count(DISTINCT labels['i']) FROM prom_topk_expr(100, 'irate', {}, [], {}, {}, 60000, 65000)",
            terms[1],
            T0 + 600_000,
            T0 + 10_000_000
        ))
        .unwrap();
    assert_eq!(n.rows[0].get(), "[14]", "label sets i % 3 == 0 only");
    // misuse is an error, not a wrong answer
    for bad in [
        "prom_topk_expr(3, 'irate', [{'m': 'm_total', 'n': 1.0, 'd': 1.0}, {'m': 'm2_total', 'n': 0.0, 'd': 1.0}], [], 0, 600000, 60000, 65000, p_by := ['grp'])",
        "prom_topk_expr(3, 'irate', [{'m': 'm_total', 'n': 1.0, 'd': 0.0}, {'m': 'm_total', 'n': 1.0, 'd': 0.0}], [], 0, 600000, 60000, 65000)",
    ] {
        assert!(r.run(&format!("SELECT * FROM {bad}")).is_err(), "{bad}");
    }
}
