//! Retention (Compactor::expire) on a local store: which files leave the catalog for a given
//! "now", that they are deleted from disk after the grace, and that what is kept still answers.
use pgai_mon::collector::Emitter;
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::eval::rollup;
use pgai_mon::model::*;
use pgai_mon::query::{Reader, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::{StoreCore, SERIES_MONTH};
use pgai_mon::Msg;
use std::collections::BTreeSet;

const DAY: i64 = 86_400_000;
/// 2026-11-07, as a day number.
const NOV7: i64 = 20_764;

fn one(r: &Reader, sql: &str) -> i64 {
    let q = r.run(sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    let v: (i64,) = serde_json::from_str(q.rows[0].get()).unwrap();
    v.0
}

/// Days (relative to Nov 7) of the catalog's files of `kind`.
fn days(core: &StoreCore, kind: &str) -> BTreeSet<i64> {
    let cat = core.shared.catalog.lock().unwrap();
    cat.files_of(kind).map(|f| f.day - NOV7).collect()
}

#[test]
fn files_past_their_retention_leave_the_catalog_and_the_disk() {
    let d = tempfile::tempdir().unwrap();
    let keep_all = StoreCfg {
        delete_grace_s: 0,
        ..Default::default()
    };
    let mut core = StoreCore::open(d.path(), &keep_all).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Msg>(1 << 16);
    let em = Emitter::new(core.shared.clone(), tx, Default::default());
    // Sep 23-24, Oct 18-19, Nov 6-7: six scrapes of three series a day, and one query text
    // on the first day of each pair
    for back in [45, 44, 20, 19, 1, 0] {
        let t0 = (NOV7 - back) * DAY + 3_600_000;
        for k in 0..6 {
            let defs: Vec<_> = (0..3u64)
                .map(|q| {
                    let labels = vec![("queryid".to_string(), q.to_string())];
                    (series_id("m_total", &labels), "m_total".to_string(), labels)
                })
                .collect();
            let vals = defs.iter().map(|(id, _, _)| (*id, k as f64)).collect();
            rt.block_on(em.emit(defs, t0 + k * 600_000, vals));
        }
        while let Ok(m) = rx.try_recv() {
            if let Msg::Rec(rec) = m {
                core.append(rec).unwrap();
            }
        }
        if back % 2 == 1 {
            core.append(Record::QueryTexts(vec![QueryText {
                queryid: back,
                dbname: "db1".into(),
                query_hash: back as u64,
                query: Some(format!("select {back}")),
                first_seen: t0,
                last_seen: t0,
                src: Default::default(),
            }]))
            .unwrap();
        }
        core.flush().unwrap();
    }
    let now = NOV7 * DAY + 12 * 3_600_000;
    let comp = core.compactor().unwrap();
    comp.run_once(now, true).unwrap();
    assert_eq!(comp.expire(now).unwrap(), 0, "retention 0 keeps everything");
    let all: BTreeSet<i64> = [-45, -44, -20, -19, -1, 0].into();
    assert_eq!(days(&core, "samples"), all);
    assert_eq!(days(&core, "query_texts"), [-45, -19, -1].into());
    let (sep1, oct1) = (-67, -37);
    for kind in [rollup::KIND_1D, rollup::KIND_1HM, SERIES_MONTH] {
        let months = days(&core, kind);
        assert!(months.contains(&sep1) && months.contains(&oct1), "{kind}: {months:?}");
    }
    let before = core.shared.catalog.lock().unwrap().clone();
    drop(comp);
    drop(core);

    // 14 days of metrics (cut: Oct 24), 30 days of query texts (cut: Oct 8)
    let cfg = StoreCfg {
        retention_days: 14,
        texts_retention_days: 30,
        ..keep_all
    };
    let core = StoreCore::open(d.path(), &cfg).unwrap();
    let comp = core.compactor().unwrap();
    let n = comp.expire(now).unwrap();
    let after = core.shared.catalog.lock().unwrap().clone();
    assert_eq!(n, before.files.len() - after.files.len());
    assert!(after.data_epoch > before.data_epoch);
    for kind in ["samples", "series", rollup::KIND] {
        assert_eq!(days(&core, kind), [-1, 0].into(), "{kind}");
    }
    assert_eq!(days(&core, "query_texts"), [-19, -1].into());
    // a month file is kept until its whole month is past the retention: October's last day is
    // Oct 31, September's is not
    for kind in [rollup::KIND_1D, rollup::KIND_1HM, SERIES_MONTH] {
        let months = days(&core, kind);
        assert!(!months.contains(&sep1) && months.contains(&oct1), "{kind}: {months:?}");
    }
    // the series registry is not per day: it is kept
    assert_eq!(
        after.files_of(pgai_mon::registry::KIND).count(),
        before.files_of(pgai_mon::registry::KIND).count()
    );

    // the next pass, past the delete grace, removes the files and builds nothing from the gap
    comp.run_once(now + 1000, true).unwrap();
    let kept: BTreeSet<String> = core
        .shared
        .catalog
        .lock()
        .unwrap()
        .files
        .iter()
        .map(|f| f.path.clone())
        .collect();
    assert_eq!(kept, after.files.iter().map(|f| f.path.clone()).collect());
    let gone: Vec<&String> = before
        .files
        .iter()
        .map(|f| &f.path)
        .filter(|p| !kept.contains(*p))
        .collect();
    assert_eq!(gone.len(), n);
    for p in gone {
        assert!(!d.path().join(p).exists(), "{p} still on disk");
    }
    for p in &kept {
        assert!(d.path().join(p).exists(), "{p} missing");
    }
    let r = Reader::new(
        core.shared.clone(),
        HttpCfg::default(),
        EMBEDDED_VIEWS_SQL.into(),
        vec![],
    )
    .unwrap();
    assert_eq!(one(&r, "SELECT count(*) FROM samples"), 2 * 6 * 3);
    assert_eq!(one(&r, "SELECT count(DISTINCT queryid) FROM query_texts"), 2);
    assert_eq!(one(&r, "SELECT count(*) FROM series"), 3);
}
