//! irate over rollups reads the sample-interval lists (`iv`) of the last two days' files up
//! front and of older files only when a series' scrape-interval walk reaches them
//! (rollup.rs RollupScan::load_iv). Results are bit-identical to reading every file's `iv`
//! (PGAI_ROLLUP_EAGER_IV=1) and to reading them in windows of a few rows, and match the raw
//! path and the SQL oracle, for series that end early, sparse ones, jittered and gapped ones,
//! with resets, NaN and staleness markers.
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::model::*;
use pgai_mon::query::{Reader, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::StoreCore;
use std::sync::atomic::Ordering;

const MACROS: &str = include_str!("../sql/macros.sql");
const H: i64 = 3_600_000;
const DAY: i64 = 86_400_000;

/// Five closed days of `c_total` series, compacted (day files and 1 h rollups):
/// dense (60 s), jittered, sparse (40-55 min apart: 20 intervals span about two days), ended on
/// day 1 or day 2 (their newest buckets are in files read lazily), with resets, a 5 h gap,
/// NaN and staleness markers.
fn store(dir: &std::path::Path) -> (StoreCore, i64) {
    let mut core = StoreCore::open(
        dir,
        &StoreCfg {
            delete_grace_s: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let day0 = 1_794_009_600_000i64 - 3 * DAY; // a UTC midnight
    let mut seed = 5u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut defs = vec![];
    let mut samples = vec![];
    for s in 0..20u64 {
        let labels = vec![("i".to_string(), s.to_string())];
        let id = series_id("c_total", &labels);
        let (gap, end) = match s % 5 {
            0 => (60_000, 5 * DAY),            // dense
            1 => (60_000, DAY + 13 * H),       // ends on day 1
            2 => (2_400_000, 5 * DAY),         // sparse
            3 => (2_400_000, 2 * DAY + 7 * H), // sparse, ends on day 2
            _ => (60_000, 2 * DAY + 20 * H),   // ends on day 2
        };
        let jitter = if s % 2 == 1 { 900_000 } else { 300 };
        let mut t = day0 + (rnd() % 60_000) as i64;
        let mut v = (rnd() % 100) as f64;
        let mut k = 0;
        while t < day0 + end {
            v += (rnd() % 30) as f64;
            if k % 400 == 150 {
                v = 2.0; // reset
            }
            if k % 600 == 90 {
                t += 5 * H; // gap
            }
            let value = if k % 333 == 40 {
                stale_nan()
            } else if k % 517 == 70 {
                f64::NAN
            } else {
                v
            };
            samples.push(Sample {
                series_id: id,
                ts: t,
                value,
            });
            t += gap + (rnd() % jitter) as i64;
            k += 1;
        }
        defs.push(SeriesDef {
            series_id: id,
            metric: "c_total".into(),
            labels,
            first_seen: day0,
        });
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
    assert_eq!(rollups, 5, "one rollup per closed day");
    (core, day0)
}

#[test]
fn irate_reads_old_intervals_on_demand_bit_identically() {
    use pgai_mon::eval::rollup::{IV_LOADS, SERVED};
    let d = tempfile::tempdir().unwrap();
    let (core, day0) = store(d.path());
    let r = Reader::new(
        core.shared.clone(),
        HttpCfg {
            memory_limit: "1GB".into(),
            row_cache_mb: 0,
            ..Default::default()
        },
        format!("{EMBEDDED_VIEWS_SQL}\n;\n{MACROS}"),
        vec![],
    )
    .unwrap();
    let rows = |sql: &str| -> Vec<(u64, i64, f64)> {
        let res = r.run(sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
        res.rows
            .iter()
            .map(|x| serde_json::from_str(x.get()).unwrap())
            .collect()
    };
    // the 1 h day files (PGAI_NO_ROLLUP_1HM: older days read lazily), then the month file
    // (the compactor merged these days into one: read up front)
    for days in [true, false] {
        if days {
            std::env::set_var("PGAI_NO_ROLLUP_1HM", "1");
        } else {
            std::env::remove_var("PGAI_NO_ROLLUP_1HM");
        }
        let loads0 = IV_LOADS.load(Ordering::Relaxed);
        let mut checked = 0;
        for (start, end, step, window) in [
            (day0 + 5 * H, day0 + 118 * H, H, H + 5_000),          // whole range
            (day0 + 2 * H, day0 + 60 * H, H, H + 5_000),           // ends inside the older days
            (day0 + 30 * H, day0 + 110 * H, 2 * H, 2 * H + 5_000), // 2 h step
            (day0 + 5 * H, day0 + 118 * H, H, 3 * H),              // window over several buckets
            (day0 + 119 * H, day0 + 119 * H, 300_000, 9 * H),      // instant on a boundary
        ] {
            let args = format!("'c_total', [], {start}, {end}, {step}, {window}");
            let q = |name: &str| format!("SELECT series_id, t, value FROM {name}({args}) ORDER BY 1, 2");
            let served = SERVED[0].load(Ordering::Relaxed);
            let lazy = rows(&q("prom_irate"));
            assert!(
                SERVED[0].load(Ordering::Relaxed) > served,
                "{args}: not served from rollups"
            );
            std::env::set_var("PGAI_ROLLUP_EAGER_IV", "1");
            let eager = rows(&q("prom_irate"));
            std::env::remove_var("PGAI_ROLLUP_EAGER_IV");
            // windows of a few rows: each series reads its own, some span two
            std::env::set_var("PGAI_ROLLUP_IV_WINDOW", "7");
            let small = rows(&q("prom_irate"));
            std::env::remove_var("PGAI_ROLLUP_IV_WINDOW");
            assert!(
                lazy.len() == small.len()
                    && lazy
                        .iter()
                        .zip(&small)
                        .all(|(x, y)| (x.0, x.1, x.2.to_bits()) == (y.0, y.1, y.2.to_bits())),
                "window 7 {args}"
            );
            assert_eq!(lazy.len(), eager.len(), "lazy vs eager {args}");
            for (x, y) in lazy.iter().zip(&eager) {
                assert_eq!(
                    (x.0, x.1, x.2.to_bits()),
                    (y.0, y.1, y.2.to_bits()),
                    "lazy vs eager {args}"
                );
            }
            std::env::set_var("PGAI_NO_ROLLUP", "1");
            let raw = rows(&q("prom_irate"));
            std::env::remove_var("PGAI_NO_ROLLUP");
            let oracle = rows(&q("prom_irate_sql"));
            for (name, other) in [("raw", &raw), ("oracle", &oracle)] {
                assert_eq!(lazy.len(), other.len(), "lazy vs {name} {args}");
                for (x, y) in lazy.iter().zip(other.iter()) {
                    assert_eq!((x.0, x.1), (y.0, y.1), "lazy vs {name} {args}");
                    assert!(
                        (x.2 - y.2).abs() <= 1e-9 * y.2.abs().max(1e-12),
                        "lazy vs {name} {args}: {x:?} {y:?}"
                    );
                }
            }
            checked += lazy.len();
        }
        assert!(checked > 1_000, "{checked}");
        if days {
            assert!(
                IV_LOADS.load(Ordering::Relaxed) > loads0,
                "no walk read an older file's intervals"
            );
        }
    }
    std::env::remove_var("PGAI_NO_ROLLUP_1HM");
}
