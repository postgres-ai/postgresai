//! WAL crash path and compaction tests.
use pgai_mon::config::StoreCfg;
use pgai_mon::model::*;
use pgai_mon::store::{file_list_sql, samples_raw_select, Failpoint, StoreCore};

fn series(n: u64) -> Vec<SeriesDef> {
    (0..n)
        .map(|i| {
            let labels = vec![("i".to_string(), i.to_string())];
            SeriesDef {
                series_id: series_id("t_metric", &labels),
                metric: "t_metric".into(),
                labels,
                first_seen: 0,
            }
        })
        .collect()
}

fn batch(ids: &[u64], ts: i64, v: f64) -> Record {
    Record::Samples(
        ids.iter()
            .map(|id| Sample {
                series_id: *id,
                ts,
                value: v,
            })
            .collect(),
    )
}

/// (rows, distinct (series, ts), checksum) over the store's samples.
fn stats(core: &StoreCore) -> (i64, i64, f64) {
    let cat = core.shared.catalog.lock().unwrap().clone();
    let files: Vec<_> = cat.files_of("samples").collect();
    if files.is_empty() {
        return (0, 0, 0.0);
    }
    let src = format!("read_parquet({})", file_list_sql(&core.shared.root, &files));
    let q = format!(
        "SELECT count(*), count(DISTINCT (series_id, ts)), sum((hash(series_id, ts, value) % 1000003)::DOUBLE) FROM ({})",
        samples_raw_select(&src)
    );
    core.conn()
        .query_row(&q, [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap()
}

fn cfg() -> StoreCfg {
    StoreCfg {
        delete_grace_s: 0,
        ..Default::default()
    }
}

const T0: i64 = 1_794_000_000_000;

#[test]
fn crash_after_parquet_before_catalog_no_gap_no_dup() {
    let dir = tempfile::tempdir().unwrap();
    let s = series(50);
    let ids: Vec<u64> = s.iter().map(|d| d.series_id).collect();
    {
        let mut core = StoreCore::open(dir.path(), &cfg()).unwrap();
        core.append(Record::Series(s)).unwrap();
        for k in 0..10 {
            core.append(batch(&ids, T0 + k * 30_000, k as f64)).unwrap();
        }
        core.flush().unwrap(); // a clean flush first
        for k in 10..20 {
            core.append(batch(&ids, T0 + k * 30_000, k as f64)).unwrap();
        }
        core.failpoint = Some(Failpoint::AfterParquet);
        assert!(core.flush().is_err()); // "killed": Parquet written, catalog not updated
    }
    let mut core = StoreCore::open(dir.path(), &cfg()).unwrap(); // removes orphans, replays WAL
    core.flush().unwrap();
    let (rows, distinct, _) = stats(&core);
    assert_eq!(rows, 50 * 20);
    assert_eq!(distinct, rows);
}

fn parquet_files(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| match e.path() {
            p if p.is_dir() => parquet_files(&p),
            p => usize::from(p.extension().is_some_and(|x| x == "parquet")),
        })
        .sum()
}

/// A flush whose catalog save fails, in a process that keeps running: the catalog in memory
/// and the directory go back to what they were, and the retry records each sample once.
#[test]
fn a_failed_catalog_save_leaves_nothing_behind() {
    let dir = tempfile::tempdir().unwrap();
    let s = series(50);
    let ids: Vec<u64> = s.iter().map(|d| d.series_id).collect();
    let mut core = StoreCore::open(dir.path(), &cfg()).unwrap();
    core.append(Record::Series(s)).unwrap();
    for k in 0..10 {
        core.append(batch(&ids, T0 + k * 30_000, k as f64)).unwrap();
    }
    core.flush().unwrap();
    let (files, on_disk) = (
        core.shared.catalog.lock().unwrap().files.len(),
        parquet_files(dir.path()),
    );
    for k in 10..20 {
        core.append(batch(&ids, T0 + k * 30_000, k as f64)).unwrap();
    }
    // the save writes catalog.json.tmp: a directory of that name makes it fail
    let block = dir.path().join("catalog.json.tmp");
    std::fs::create_dir(&block).unwrap();
    for _ in 0..2 {
        assert!(core.flush().is_err());
        assert_eq!(core.shared.catalog.lock().unwrap().files.len(), files);
        assert_eq!(parquet_files(dir.path()), on_disk);
    }
    std::fs::remove_dir(&block).unwrap();
    core.flush().unwrap();
    let (rows, distinct, _) = stats(&core);
    assert_eq!((rows, distinct), (50 * 20, 50 * 20));
}

#[test]
fn crash_after_catalog_before_wal_delete_no_dup() {
    let dir = tempfile::tempdir().unwrap();
    let s = series(20);
    let ids: Vec<u64> = s.iter().map(|d| d.series_id).collect();
    {
        let mut core = StoreCore::open(dir.path(), &cfg()).unwrap();
        core.append(Record::Series(s)).unwrap();
        for k in 0..7 {
            core.append(batch(&ids, T0 + k * 30_000, k as f64)).unwrap();
        }
        core.failpoint = Some(Failpoint::AfterCatalog);
        assert!(core.flush().is_err()); // catalog committed, WAL segment still on disk
    }
    let mut core = StoreCore::open(dir.path(), &cfg()).unwrap();
    core.flush().unwrap();
    let (rows, distinct, _) = stats(&core);
    assert_eq!(rows, 20 * 7);
    assert_eq!(distinct, rows);
}

#[test]
fn torn_wal_tail_is_truncated_and_prefix_kept() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let s = series(5);
    let ids: Vec<u64> = s.iter().map(|d| d.series_id).collect();
    {
        let mut core = StoreCore::open(dir.path(), &cfg()).unwrap();
        core.append(Record::Series(s)).unwrap();
        for k in 0..3 {
            core.append(batch(&ids, T0 + k * 30_000, k as f64)).unwrap();
        }
    } // no flush: everything only in the WAL
      // simulate a torn write: half a frame at the tail
    let seg = pgai_mon::wal::list_segments(&dir.path().join("wal")).unwrap();
    let p = pgai_mon::wal::seg_path(&dir.path().join("wal"), *seg.last().unwrap());
    std::fs::OpenOptions::new()
        .append(true)
        .open(&p)
        .unwrap()
        .write_all(&[200, 0, 0, 0, 1, 2, 3])
        .unwrap();
    let mut core = StoreCore::open(dir.path(), &cfg()).unwrap();
    core.flush().unwrap();
    let (rows, distinct, _) = stats(&core);
    assert_eq!((rows, distinct), (15, 15));
}

#[test]
fn compaction_preserves_rows_and_checksum() {
    let dir = tempfile::tempdir().unwrap();
    let s = series(30);
    let ids: Vec<u64> = s.iter().map(|d| d.series_id).collect();
    let mut core = StoreCore::open(dir.path(), &cfg()).unwrap();
    core.append(Record::Series(s)).unwrap();
    // 3 days, a flush every 40 steps -> many small files per day
    for k in 0..(3 * 2880) {
        core.append(batch(
            &ids,
            T0 + k * 30_000,
            (k * 7 % 1000) as f64 + 0.25 * (k % 3) as f64,
        ))
        .unwrap();
        if k % 400 == 399 {
            core.flush().unwrap();
        }
    }
    core.flush().unwrap();
    let before = stats(&core);
    let files_before = core.shared.catalog.lock().unwrap().files_of("samples").count();
    let n = core.compactor().unwrap().run_once(now_ms(), true).unwrap();
    assert!(n > 0);
    let after = stats(&core);
    assert_eq!(before, after);
    let cat = core.shared.catalog.lock().unwrap().clone();
    let mut per_day = std::collections::BTreeMap::new();
    for f in cat.files_of("samples") {
        *per_day.entry(f.day).or_insert(0) += 1;
    }
    assert!(files_before > per_day.len());
    // the streaming k-way merge (default) writes one sorted file per day, whatever the part limit
    assert!(per_day.values().all(|c| *c == 1), "{per_day:?}");
    // a second pass leaves finished days alone
    assert_eq!(core.compactor().unwrap().run_once(now_ms(), true).unwrap(), 0);
    // pending deletes purged (grace 0) on the next pass
    core.compactor().unwrap().run_once(now_ms() + 1, true).unwrap();
    assert!(core.shared.catalog.lock().unwrap().pending_delete.is_empty());
}
