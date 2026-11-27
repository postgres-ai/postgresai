//! Object-store write path (bucket.rs) against an InMemory bucket and a fault-injecting
//! wrapper: commit, fencing, crash between upload and commit, failed and partial uploads,
//! restart from an empty disk, compaction over evicted inputs. No network.
use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream::BoxStream;
use object_store::memory::InMemory;
use object_store::path::Path as OPath;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore, ObjectStoreExt,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, UploadPart,
};
use pgai_mon::bucket::{Bucket, BucketCfg, Failpoint, Fenced};
use pgai_mon::catalog::Catalog;
use pgai_mon::collector::Emitter;
use pgai_mon::config::{CollectorCfg, HttpCfg, StoreCfg};
use pgai_mon::model::*;
use pgai_mon::query::{Reader, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::{Shared, StoreCore};
use pgai_mon::Msg;
use std::collections::BTreeSet;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

// ------------------------------------------------------------------ fault-injecting store

/// InMemory with injected failures: data PUTs (`.parquet` keys) fail while `fail_data_puts`;
/// multipart uploads fail at part `fail_part` (1-based; -1 = never); the next catalog.json PUT
/// lands and its answer is lost when `lose_answer` is set: 1 = the caller gets a failed
/// precondition (what the client's retry of a conditional PUT meets), 2 = a plain error (the
/// client gave up); 3 = it fails and does not land.
#[derive(Debug, Default)]
struct Faulty {
    inner: InMemory,
    fail_data_puts: AtomicBool,
    fail_part: AtomicI64,
    lose_answer: AtomicI64,
}

impl std::fmt::Display for Faulty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Faulty({})", self.inner)
    }
}

fn injected(what: &str) -> object_store::Error {
    object_store::Error::Generic {
        store: "faulty",
        source: format!("injected failure: {what}").into(),
    }
}

#[derive(Debug)]
struct FaultyUpload {
    up: Box<dyn MultipartUpload>,
    fail_part: i64,
    n: i64,
}

#[async_trait]
impl MultipartUpload for FaultyUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        self.n += 1;
        if self.fail_part >= 0 && self.n >= self.fail_part {
            return Box::pin(async { Err(injected("part")) });
        }
        self.up.put_part(data)
    }
    async fn complete(&mut self) -> object_store::Result<PutResult> {
        self.up.complete().await
    }
    async fn abort(&mut self) -> object_store::Result<()> {
        self.up.abort().await
    }
}

#[async_trait]
impl ObjectStore for Faulty {
    async fn put_opts(
        &self,
        location: &OPath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        if self.fail_data_puts.load(Ordering::SeqCst) && location.as_ref().ends_with(".parquet") {
            return Err(injected("put"));
        }
        if location.as_ref().ends_with("catalog.json") {
            let how = self.lose_answer.swap(0, Ordering::SeqCst);
            if how == 3 {
                return Err(injected("put"));
            }
            if how != 0 {
                self.inner.put_opts(location, payload, opts).await?;
                return Err(match how {
                    1 => object_store::Error::Precondition {
                        path: location.to_string(),
                        source: "the retry of a PUT that landed".into(),
                    },
                    _ => injected("answer lost"),
                });
            }
        }
        self.inner.put_opts(location, payload, opts).await
    }
    async fn put_multipart_opts(
        &self,
        location: &OPath,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        if self.fail_data_puts.load(Ordering::SeqCst) && location.as_ref().ends_with(".parquet") {
            return Err(injected("multipart"));
        }
        let up = self.inner.put_multipart_opts(location, opts).await?;
        Ok(Box::new(FaultyUpload {
            up,
            fail_part: self.fail_part.load(Ordering::SeqCst),
            n: 0,
        }))
    }
    async fn get_opts(&self, location: &OPath, options: GetOptions) -> object_store::Result<GetResult> {
        self.inner.get_opts(location, options).await
    }
    async fn get_ranges(&self, location: &OPath, ranges: &[Range<u64>]) -> object_store::Result<Vec<Bytes>> {
        self.inner.get_ranges(location, ranges).await
    }
    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<OPath>>,
    ) -> BoxStream<'static, object_store::Result<OPath>> {
        self.inner.delete_stream(locations)
    }
    fn list(&self, prefix: Option<&OPath>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }
    fn list_with_offset(
        &self,
        prefix: Option<&OPath>,
        offset: &OPath,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }
    async fn list_with_delimiter(&self, prefix: Option<&OPath>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
    async fn copy_opts(&self, from: &OPath, to: &OPath, options: CopyOptions) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
    async fn rename_opts(&self, from: &OPath, to: &OPath, options: RenameOptions) -> object_store::Result<()> {
        self.inner.rename_opts(from, to, options).await
    }
}

fn faulty() -> Arc<Faulty> {
    Arc::new(Faulty {
        fail_part: AtomicI64::new(-1),
        ..Default::default()
    })
}

// ------------------------------------------------------------------ helpers

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// Every object key under the prefix "st" (relative to it), sorted.
fn objects(store: &Arc<Faulty>) -> BTreeSet<String> {
    rt().block_on(async {
        let mut s = store.list(Some(&OPath::from("st")));
        let mut v = BTreeSet::new();
        while let Some(m) = std::future::poll_fn(|cx| s.as_mut().poll_next(cx)).await {
            v.insert(m.unwrap().location.as_ref().strip_prefix("st/").unwrap().to_string());
        }
        v
    })
}

/// The catalog the bucket holds now (read past any overlay).
fn bucket_catalog(store: &Arc<Faulty>) -> Option<Catalog> {
    rt().block_on(async {
        match store.get(&OPath::from("st/catalog.json")).await {
            Ok(g) => Some(serde_json::from_slice(&g.bytes().await.unwrap()).unwrap()),
            Err(_) => None,
        }
    })
}

/// Every file the bucket's catalog lists is in the bucket.
fn assert_listed_objects_exist(store: &Arc<Faulty>) {
    let keys = objects(store);
    for f in &bucket_catalog(store).unwrap().files {
        assert!(keys.contains(&f.path), "{} is listed and not in the bucket", f.path);
    }
}

/// Rows of `m_a` in the hour from `t0` that a reader of the bucket sees (20 per scrape).
fn bucket_rows(url: &str, t0: i64) -> String {
    let q = format!("SELECT count(*) FROM pgai_samples('m_a', [], {t0}, {})", t0 + 3_600_000);
    rows(&bucket_reader(url), &q).join("")
}

fn bcfg() -> BucketCfg {
    BucketCfg {
        delete_grace_s: 0,
        ..Default::default()
    }
}

fn scfg() -> StoreCfg {
    StoreCfg {
        delete_grace_s: 0,
        ..Default::default()
    }
}

/// A "process": a bucket attached to `dir` and the store opened on it.
fn start(url: &str, store: &Arc<Faulty>, dir: &Path, cfg: BucketCfg) -> anyhow::Result<(Arc<Bucket>, StoreCore)> {
    let b = Bucket::with_store(url, store.clone(), dir, cfg)?;
    if let Err(e) = b.attach(pgai_mon::store::DEFAULT_LAYOUT) {
        b.close();
        return Err(e);
    }
    let core = StoreCore::open(&b.local, &scfg())?;
    Ok((b, core))
}

/// Start even when another writer's lease is fresh.
fn forced() -> BucketCfg {
    BucketCfg {
        takeover: true,
        ..bcfg()
    }
}

/// A start on `dir` that stops hard right after its takeover PUT, before it saves its catalog.
fn cut_short(url: &str, store: &Arc<Faulty>, dir: &Path, cfg: BucketCfg) {
    let b = Bucket::with_store(url, store.clone(), dir, cfg).unwrap();
    *b.failpoint.lock().unwrap() = Some(Failpoint::AfterTakeover);
    assert!(b.attach(pgai_mon::store::DEFAULT_LAYOUT).is_err());
    b.close();
}

const METRICS: &[&str] = &["m_a", "m_b"];

/// Ingest `n` scrapes at `t0, t0 + 15 s, ...` through the collector's Emitter (series first,
/// once per series and day) into `core`. Returns the WAL records the Emitter produced.
fn ingest(core: &mut StoreCore, t0: i64, n: i64, seed: u64) -> (usize, usize) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Msg>(1 << 16);
    let em = Emitter::new(core.shared.clone(), tx, CollectorCfg::default());
    let r = rt();
    for k in 0..n {
        let ts = t0 + k * 15_000;
        for m in METRICS {
            let mut defs = vec![];
            let mut vals = vec![];
            for i in 0..20u64 {
                let labels = vec![
                    ("dbname".to_string(), "db1".to_string()),
                    ("i".to_string(), i.to_string()),
                ];
                let id = series_id(m, &labels);
                defs.push((id, m.to_string(), labels));
                vals.push((id, ((seed + i) * 1_000_000 + (ts / 15_000) as u64 % 1_000_000) as f64));
            }
            r.block_on(em.emit(defs, ts, vals));
        }
    }
    drop(em);
    let (mut series, mut samples) = (0, 0);
    while let Ok(m) = rx.try_recv() {
        if let Msg::Rec(rec) = m {
            match &rec {
                Record::Series(v) => series += v.len(),
                Record::Samples(v) => samples += v.len(),
                _ => {}
            }
            core.append(rec).unwrap();
        }
    }
    (series, samples)
}

fn views() -> String {
    format!("{EMBEDDED_VIEWS_SQL}\n;\n{}", include_str!("../sql/macros.sql"))
}

fn rows(r: &Reader, sql: &str) -> Vec<String> {
    let res = r.run(sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    res.rows.iter().map(|x| x.get().to_string()).collect()
}

fn queries(t0: i64, t1: i64) -> Vec<String> {
    vec![
        "SELECT series_id, metric, labels FROM series ORDER BY 1".into(),
        format!("SELECT series_id, ts, value FROM pgai_samples('m_a', [], {t0}, {t1}) ORDER BY 1, 2"),
        format!("SELECT series_id, ts, value FROM pgai_samples('m_b', [['i', '=~', '1.*']], {t0}, {t1}) ORDER BY 1, 2"),
        format!("SELECT series_id, t, value FROM prom_last('m_a', [], {t0}, {t1}, 60000) ORDER BY 1, 2"),
        // the SQL macros the reporter compiles to read raw samples through prom_raw
        format!("SELECT series_id, ts_ms, v, c FROM prom_select('m_a', [], {t0}, {t1}) ORDER BY 1, 2"),
        format!("SELECT series_id, t, count(*), max(v) FROM prom_window('m_b', [], {t0}, {t1}, 60000, 120000) GROUP BY 1, 2 ORDER BY 1, 2"),
        format!("SELECT series_id, t, n, first_v, last_v FROM prom_edges('m_a', [], {t0}, {t1}, 60000, 120000, true, false) ORDER BY 1, 2"),
    ]
}

/// A reader of what the bucket holds (no overlay: the "other machine" view).
fn bucket_reader(url: &str) -> Reader {
    pgai_mon::objstore::set_overlay(url, None).unwrap();
    let sh = Shared::open_read_only(Path::new(url)).unwrap();
    Reader::new(sh, HttpCfg::default(), views(), vec![]).unwrap()
}

/// `back_min` minutes ago (to the minute), but never before today's first minute: the test's
/// data stays in today, whose series the restart rebuilds.
fn start_ts(back_min: i64) -> i64 {
    let n = now_ms();
    (n - n % 60_000 - back_min * 60_000).max(n - n % 86_400_000 + 60_000)
}

// ------------------------------------------------------------------ tests

/// End to end: ingest through the Emitter, flush, commit; more data flushed but not uploaded is
/// visible to the writer's own reader (overlay); then "lose the VM" (new process state, empty
/// local dir): the live lease refuses a plain start, a takeover adopts the bucket's catalog,
/// rebuilds the known series from the series files (no SeriesDef is emitted again), and SQL
/// over the store gives exactly what the same ingest gives on a local-disk store.
#[test]
fn lose_the_vm_and_restart_from_the_bucket() {
    let url = "mem://e2e/st";
    let store = faulty();
    let (d1, d2, dref) = (
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
    );
    let t0 = start_ts(40);
    let (t_end, t_lost) = (t0 + 80 * 15_000, t0 + 120 * 15_000);
    let (b1, mut c1) = start(url, &store, d1.path(), bcfg()).unwrap();
    assert_eq!(b1.epoch(), 1);
    let (ns, _) = ingest(&mut c1, t0, 40, 1);
    assert_eq!(ns, 40, "one SeriesDef per series and day");
    c1.flush().unwrap();
    ingest(&mut c1, t0 + 40 * 15_000, 40, 1);
    c1.flush().unwrap();
    let comp = c1.compactor().unwrap();
    b1.sync(&c1.shared, Some(&comp)).unwrap();
    // the pre-upload merge left one samples file for the day (two if an hour began between
    // the flushes: it merges the current hour's files only)
    let bc = bucket_catalog(&store).unwrap();
    assert!(
        bc.files.iter().filter(|f| f.kind == "samples").count() <= 2,
        "{:?}",
        bc.files
    );
    assert_eq!(bc.files.iter().filter(|f| f.kind == "series").count(), 1);
    assert_eq!(bc.writer.as_ref().unwrap().id, b1.id);
    for f in &bc.files {
        assert!(objects(&store).contains(&f.path), "{} listed but not uploaded", f.path);
    }

    // data after the commit: flushed locally, not uploaded
    ingest(&mut c1, t_end, 40, 1);
    c1.flush().unwrap();
    let own = Reader::new(
        Shared {
            root: PathBuf::from(url),
            ..c1.shared.clone()
        },
        HttpCfg::default(),
        views(),
        vec![],
    )
    .unwrap();
    let q_all = format!("SELECT count(*) FROM pgai_samples('m_a', [], {t0}, {t_lost})");
    assert_eq!(
        rows(&own, &q_all),
        vec!["[2400]".to_string()],
        "own reader sees un-uploaded files"
    );
    let other = bucket_reader(url);
    assert_eq!(
        rows(&other, &q_all),
        vec!["[1600]".to_string()],
        "the bucket has the committed data only"
    );
    drop(own);

    // the reference: the same committed ingest on local disk
    let mut cref = StoreCore::open(dref.path(), &scfg()).unwrap();
    ingest(&mut cref, t0, 80, 1);
    cref.flush().unwrap();
    let local_ref = Reader::new(cref.shared.clone(), HttpCfg::default(), views(), vec![]).unwrap();

    // lose the VM
    drop(comp);
    drop(c1);
    b1.close();
    let err = start(url, &store, d2.path(), bcfg())
        .err()
        .expect("a live lease refuses the start");
    assert!(
        format!("{err:#}").contains("refusing to start a second writer"),
        "{err:#}"
    );
    let (b2, mut c2) = start(
        url,
        &store,
        d2.path(),
        BucketCfg {
            takeover: true,
            ..bcfg()
        },
    )
    .unwrap();
    assert_eq!(b2.epoch(), 2);
    assert_ne!(b2.id, b1.id);
    assert!(c2.shared.catalog.lock().unwrap().next_file_id >= 2 << 32);
    // known series rebuilt from the bucket's series file: the Emitter sends no SeriesDef again
    assert_eq!(c2.shared.known_series.lock().unwrap().len(), 40);
    let (ns, nsamp) = ingest(&mut c2, t_lost + 60_000, 2, 1);
    assert_eq!((ns, nsamp), (0, 80));

    let other = bucket_reader(url);
    for q in queries(t0, t_lost) {
        let a = rows(&other, &q);
        assert!(!a.is_empty(), "{q}");
        assert_eq!(a, rows(&local_ref, &q), "{q}");
    }
    // the raw views say what to use; an empty answer would read as "no data"
    for q in [
        "SELECT count(*) FROM samples",
        "SELECT count(*) FROM metrics",
        "SELECT count(*) FROM samples WHERE ts > now() - INTERVAL 1 HOUR",
    ] {
        let e = other
            .run(q)
            .err()
            .unwrap_or_else(|| panic!("{q} answered on an object store"));
        assert!(e.message.contains("pgai_samples"), "{e:?}");
    }
    // the new writer goes on: its commit lands, the old writer's objects are untouched
    c2.flush().unwrap();
    let comp2 = c2.compactor().unwrap();
    b2.sync(&c2.shared, Some(&comp2)).unwrap();
    let after = bucket_reader(url);
    assert_eq!(
        rows(
            &after,
            &format!(
                "SELECT count(*) FROM pgai_samples('m_a', [], {t0}, {})",
                t_lost + 3 * 60_000
            )
        ),
        vec!["[1640]".to_string()]
    );
    b2.close();
}

/// A crash after the uploads and before the catalog PUT: readers of the bucket see the previous
/// catalog; the same local dir resumes and commits; an empty-disk restart sweeps the orphans.
#[test]
fn crash_between_upload_and_commit() {
    let url = "mem://crash/st";
    let store = faulty();
    let (d1, d2) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let t0 = start_ts(30);
    let (b, mut c) = start(url, &store, d1.path(), bcfg()).unwrap();
    ingest(&mut c, t0, 20, 2);
    c.flush().unwrap();
    b.sync(&c.shared, None).unwrap();
    let committed = bucket_catalog(&store).unwrap();
    ingest(&mut c, t0 + 20 * 15_000, 20, 2);
    c.flush().unwrap();
    *b.failpoint.lock().unwrap() = Some(Failpoint::AfterUpload);
    assert!(b.sync(&c.shared, None).is_err());
    let now = bucket_catalog(&store).unwrap();
    assert_eq!(now.files, committed.files, "nothing committed");
    let listed: BTreeSet<String> = now.files.iter().map(|f| f.path.clone()).collect();
    let orphans: Vec<String> = objects(&store)
        .into_iter()
        .filter(|k| k.ends_with(".parquet") && !listed.contains(k))
        .collect();
    assert!(!orphans.is_empty(), "the round's uploads are in the bucket, unlisted");
    let q = format!("SELECT count(*) FROM pgai_samples('m_a', [], {t0}, {})", t0 + 3_600_000);
    assert_eq!(rows(&bucket_reader(url), &q), vec!["[400]".to_string()]);

    // process restart on the same disk: the local catalog continues, its files get committed
    drop(c);
    b.close();
    let (b, c) = start(url, &store, d1.path(), bcfg()).unwrap();
    b.sync(&c.shared, None).unwrap();
    assert_eq!(rows(&bucket_reader(url), &q), vec!["[800]".to_string()]);

    // crash again mid-round, then the machine is lost: the new writer sweeps the orphans
    let mut c = c;
    ingest(&mut c, t0 + 40 * 15_000, 20, 2);
    c.flush().unwrap();
    *b.failpoint.lock().unwrap() = Some(Failpoint::AfterUpload);
    assert!(b.sync(&c.shared, None).is_err());
    drop(c);
    b.close();
    let before = objects(&store);
    let (b2, _c2) = start(
        url,
        &store,
        d2.path(),
        BucketCfg {
            takeover: true,
            ..bcfg()
        },
    )
    .unwrap();
    let cat = bucket_catalog(&store).unwrap();
    let keep: BTreeSet<String> = cat
        .files
        .iter()
        .map(|f| f.path.clone())
        .chain(cat.pending_delete.iter().map(|x| x.0.clone()))
        .collect();
    let left: Vec<String> = objects(&store)
        .into_iter()
        .filter(|k| k.ends_with(".parquet") && !keep.contains(k))
        .collect();
    assert!(left.is_empty(), "orphans left: {left:?}");
    assert!(before.len() > objects(&store).len());
    assert_eq!(rows(&bucket_reader(url), &q), vec!["[800]".to_string()]);
    b2.close();
}

/// A crash right after a restart's takeover PUT (before the local catalog is saved): the next
/// start still continues from the local directory, so its flushed but not uploaded files are
/// kept and committed.
#[test]
fn crash_after_the_takeover_put_keeps_local_files() {
    let url = "mem://takeover-crash/st";
    let store = faulty();
    let d = tempfile::tempdir().unwrap();
    let t0 = start_ts(30);
    let (b, mut c) = start(url, &store, d.path(), bcfg()).unwrap();
    ingest(&mut c, t0, 20, 6);
    c.flush().unwrap();
    b.sync(&c.shared, None).unwrap();
    ingest(&mut c, t0 + 20 * 15_000, 20, 6);
    c.flush().unwrap(); // local only
    drop(c);
    b.close();
    let b = Bucket::with_store(url, store.clone(), d.path(), bcfg()).unwrap();
    *b.failpoint.lock().unwrap() = Some(Failpoint::AfterTakeover);
    assert!(b.attach(pgai_mon::store::DEFAULT_LAYOUT).is_err());
    b.close();
    assert_eq!(bucket_catalog(&store).unwrap().writer.unwrap().epoch, 2);
    let (b, c) = start(url, &store, d.path(), bcfg()).unwrap();
    assert_eq!(b.epoch(), 3);
    b.sync(&c.shared, None).unwrap();
    let q = format!("SELECT count(*) FROM pgai_samples('m_a', [], {t0}, {})", t0 + 3_600_000);
    assert_eq!(rows(&bucket_reader(url), &q), vec!["[800]".to_string()]);
    b.close();
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &to.join(e.file_name()));
        } else {
            std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
        }
    }
}

/// The local directory is restored from an older disk snapshot (same writer id, catalog behind
/// the bucket). While the lease is fresh the start is refused as a second writer's: the
/// snapshot has no record of the bucket's commit, so it may be a copy of a directory in use.
/// Forced, the bucket's catalog wins, and nothing committed since the snapshot is dropped.
#[test]
fn a_directory_restored_from_an_old_snapshot_does_not_roll_the_bucket_back() {
    let url = "mem://old-snapshot/st";
    let store = faulty();
    let (d, snap) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let t0 = start_ts(30);
    let (b, mut c) = start(url, &store, d.path(), bcfg()).unwrap();
    ingest(&mut c, t0, 20, 6);
    c.flush().unwrap();
    b.sync(&c.shared, None).unwrap();
    copy_dir(d.path(), snap.path());
    ingest(&mut c, t0 + 20 * 15_000, 20, 6);
    c.flush().unwrap();
    b.sync(&c.shared, None).unwrap();
    let committed = objects(&store);
    drop(c);
    b.close();

    let e = start(url, &store, snap.path(), bcfg()).err().expect("a copy");
    assert!(format!("{e:#}").contains("a copy or an older state"), "{e:#}");
    let (b, c) = start(url, &store, snap.path(), forced()).unwrap();
    b.sync(&c.shared, None).unwrap();
    assert!(
        objects(&store).is_superset(&committed),
        "no committed object is deleted"
    );
    let q = format!("SELECT count(*) FROM pgai_samples('m_a', [], {t0}, {})", t0 + 3_600_000);
    assert_eq!(rows(&bucket_reader(url), &q), vec!["[800]".to_string()]);
    b.close();
}

/// A commit lands without its answer, the commits after it do not reach the store, and the
/// process dies: the directory still knows the bucket's commit as its own and continues from
/// its own catalog, with every file flushed since.
#[test]
fn a_restart_after_many_commits_without_an_answer_keeps_local_data() {
    let url = "mem://unanswered/st";
    let store = faulty();
    let d = tempfile::tempdir().unwrap();
    let (b, mut c) = start(url, &store, d.path(), bcfg()).unwrap();
    let t0 = start_ts(120);
    for k in 0..12 {
        ingest(&mut c, t0 + k * STEP, 10, 3);
        c.flush().unwrap();
        store.lose_answer.store(if k == 0 { 2 } else { 3 }, Ordering::SeqCst);
        assert!(b.sync(&c.shared, None).is_err(), "round {k}");
    }
    assert!(!b.is_fenced());
    drop(c);
    b.close();

    let (b, c) = start(url, &store, d.path(), bcfg()).unwrap();
    b.sync(&c.shared, None).unwrap();
    assert_listed_objects_exist(&store);
    assert_eq!(bucket_rows(url, t0), "[2400]");
    b.close();
}

/// The local directory is restored from a disk snapshot taken before a compaction, with no
/// flush since: the bucket's catalog is one this directory never sent, so it is adopted.
/// Continuing from the directory would commit files the bucket has deleted since.
#[test]
fn a_directory_restored_from_before_a_compaction_adopts_the_bucket() {
    let day = 86_400_000i64;
    let d0 = (now_ms() / day - 5) * day;
    let cfg = BucketCfg {
        keep_local_days: 0,
        ..forced()
    };
    // (also when the first start on the snapshot stops hard right after its takeover)
    for cut in [false, true] {
        let url = format!("mem://old-snapshot-compacted-{cut}/st");
        let store = faulty();
        let (d, snap) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let (b, mut c) = start(&url, &store, d.path(), cfg.clone()).unwrap();
        for k in 0..2 {
            ingest(&mut c, d0 + 3_600_000 + k * 600_000, 20, 13);
            c.flush().unwrap();
        }
        b.sync(&c.shared, None).unwrap(); // committed, and evicted from the directory
        copy_dir(d.path(), snap.path());
        c.compactor().unwrap().run_once(now_ms(), true).unwrap(); // the closed day becomes one file
        b.sync(&c.shared, None).unwrap();
        b.sync(&c.shared, None).unwrap(); // the replaced files leave the bucket
        let committed = bucket_catalog(&store).unwrap().files;
        drop(c);
        b.close();

        if cut {
            cut_short(&url, &store, snap.path(), cfg.clone());
        }
        let (b, c) = start(&url, &store, snap.path(), cfg.clone()).unwrap();
        b.sync(&c.shared, None).unwrap();
        assert_eq!(bucket_catalog(&store).unwrap().files, committed, "cut: {cut}");
        assert_listed_objects_exist(&store);
        let q = format!("SELECT count(*) FROM pgai_samples('m_a', [], {d0}, {})", d0 + day);
        assert_eq!(rows(&bucket_reader(&url), &q), vec!["[800]".to_string()]);
        b.close();
    }
}

/// Two writers on one prefix: the second is refused while the first's lease is live; a forced
/// takeover fences the first, whose next commit fails with a clear error and changes nothing.
#[test]
fn a_second_writer_is_refused_and_a_takeover_fences_the_first() {
    let url = "mem://fence/st";
    let store = faulty();
    let (d1, d2) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let t0 = start_ts(30);
    let (a, mut ca) = start(url, &store, d1.path(), bcfg()).unwrap();
    ingest(&mut ca, t0, 10, 3);
    ca.flush().unwrap();
    a.sync(&ca.shared, None).unwrap();

    let err = Bucket::with_store(url, store.clone(), d2.path(), bcfg())
        .unwrap()
        .attach(pgai_mon::store::DEFAULT_LAYOUT)
        .err()
        .unwrap();
    assert!(
        format!("{err:#}").contains("refusing to start a second writer"),
        "{err:#}"
    );
    a.probe(std::time::Duration::from_secs(5)).unwrap();
    // a clean stop releases the lease: the same refused directory now starts, with no wait
    {
        let (url, store) = ("mem://released/st", faulty());
        let (a, ca) = start(url, &store, d1.path(), bcfg()).unwrap();
        a.release(&ca.shared).unwrap();
        assert!(bucket_catalog(&store).unwrap().writer.unwrap().released);
        let (b, cb) = start(url, &store, d2.path(), bcfg()).unwrap();
        assert_eq!(b.epoch(), a.epoch() + 1);
        b.sync(&cb.shared, None).unwrap();
        assert!(!bucket_catalog(&store).unwrap().writer.unwrap().released);
    }
    let (b, cb) = start(
        url,
        &store,
        d2.path(),
        BucketCfg {
            takeover: true,
            ..bcfg()
        },
    )
    .unwrap();
    assert_eq!(b.epoch(), a.epoch() + 1);
    let owner = bucket_catalog(&store).unwrap();

    ingest(&mut ca, t0 + 10 * 15_000, 10, 3);
    ca.flush().unwrap();
    let err = a.sync(&ca.shared, None).expect_err("fenced");
    assert!(err.downcast_ref::<Fenced>().is_some(), "{err:#}");
    assert!(
        format!("{err:#}").contains(&b.id),
        "the error names the other writer: {err:#}"
    );
    assert!(a.is_fenced());
    assert_eq!(a.remote().stats.cas_failures.load(Ordering::Relaxed), 1);
    assert!(a.sync(&ca.shared, None).is_err(), "a fenced writer stays fenced");
    let cat = bucket_catalog(&store).unwrap();
    assert_eq!(cat.writer, owner.writer);
    assert_eq!(cat.files, owner.files);
    // the fenced writer's file ids are below the new epoch's, so its stray uploads never
    // overwrite a key the new owner uses
    assert!(ca.shared.catalog.lock().unwrap().next_file_id < (b.epoch() << 32));
    // the new owner commits fine, and its sweep removes the fenced writer's stray uploads
    b.sync(&cb.shared, None).unwrap();
    b.sweep().unwrap();
    let cat = bucket_catalog(&store).unwrap();
    let keep: BTreeSet<String> = cat.files.iter().map(|f| f.path.clone()).collect();
    assert!(objects(&store)
        .iter()
        .filter(|k| k.ends_with(".parquet"))
        .all(|k| keep.contains(k)));
    a.close();
    b.close();
}

/// A failed PUT and a multipart upload that dies after its first part: nothing is committed,
/// no partial object is visible, and the next round uploads and commits everything.
#[test]
fn failed_and_partial_uploads_commit_nothing() {
    let url = "mem://partial/st";
    let store = faulty();
    let d = tempfile::tempdir().unwrap();
    let cfg = BucketCfg { part_mb: 5, ..bcfg() };
    let (b, mut c) = start(url, &store, d.path(), cfg).unwrap();
    let t0 = start_ts(50);
    ingest(&mut c, t0, 10, 4);
    c.flush().unwrap();
    b.sync(&c.shared, None).unwrap();
    let committed = bucket_catalog(&store).unwrap();

    // a large samples file (random values do not compress): uploaded in parts
    let labels = [("i".to_string(), "big".to_string())];
    let ids: Vec<u64> = (0..100u64)
        .map(|k| series_id("m_big", &[labels[0].clone(), ("k".into(), k.to_string())]))
        .collect();
    let mut x = 88172645463325252u64;
    let mut samples = Vec::new();
    for (j, id) in ids.iter().enumerate() {
        for k in 0..9000i64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            samples.push(Sample {
                series_id: *id,
                ts: t0 + k * 100 + j as i64,
                value: f64::from_bits((x >> 12) | 0x3ff0_0000_0000_0000),
            });
        }
    }
    c.append(Record::Samples(samples)).unwrap();
    c.flush().unwrap();
    let big = c
        .shared
        .catalog
        .lock()
        .unwrap()
        .files
        .iter()
        .filter(|f| f.kind == "samples")
        .map(|f| f.bytes)
        .max()
        .unwrap();
    assert!(big > 5 << 20, "the file needs parts: {big} bytes");

    store.fail_part.store(2, Ordering::SeqCst);
    let e = b.sync(&c.shared, None).expect_err("part 2 fails");
    assert!(format!("{e:#}").contains("injected"), "{e:#}");
    assert_eq!(bucket_catalog(&store).unwrap().files, committed.files);
    let listed: BTreeSet<String> = committed.files.iter().map(|f| f.path.clone()).collect();
    assert!(
        objects(&store)
            .iter()
            .filter(|k| k.ends_with(".parquet"))
            .all(|k| listed.contains(k)),
        "no partial object"
    );
    store.fail_part.store(-1, Ordering::SeqCst);
    store.fail_data_puts.store(true, Ordering::SeqCst);
    assert!(b.sync(&c.shared, None).is_err());
    assert_eq!(bucket_catalog(&store).unwrap().files, committed.files);
    let h = b.health();
    assert!(h["last_error"].as_str().unwrap().contains("injected"), "{h}");
    assert!(h["pending_files"].as_u64().unwrap() >= 1, "{h}");
    let errors = h["upload_errors"].as_u64().unwrap();
    assert!(errors >= 1, "{h}");

    store.fail_data_puts.store(false, Ordering::SeqCst);
    b.sync(&c.shared, None).unwrap();
    let cat = bucket_catalog(&store).unwrap();
    let local = c.shared.catalog.lock().unwrap().files.clone();
    assert_eq!(cat.files, local);
    let h = b.health();
    assert!(
        h["puts"].as_u64().unwrap() >= 5 && h["put_bytes"].as_u64().unwrap() > big,
        "{h}"
    );
    assert_eq!(h["last_error"], serde_json::Value::Null);
    assert_eq!(h["upload_errors"].as_u64(), Some(errors), "a success counts no error");
    // the big file went up in parts and reads back exactly
    let q = format!(
        "SELECT count(*), sum(value) FROM pgai_samples('m_big', [], {t0}, {})",
        t0 + 3_600_000
    );
    let own = Reader::new(c.shared.clone(), HttpCfg::default(), views(), vec![]).unwrap();
    assert_eq!(rows(&bucket_reader(url), &q), rows(&own, &q));
    b.close();
}

/// Closed days compacted after their files left the local dir: the jobs fetch their inputs,
/// the outputs (L2, 1 h rollups, registry, month files) are uploaded, the replaced files are
/// deleted from the bucket after the grace, and the data equals a local-disk store's.
#[test]
fn compaction_with_evicted_inputs() {
    let url = "mem://compact/st";
    let store = faulty();
    let (d, dref) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let cfg = BucketCfg {
        keep_local_days: 0,
        ..bcfg()
    };
    let (b, mut c) = start(url, &store, d.path(), cfg).unwrap();
    let mut cref = StoreCore::open(dref.path(), &scfg()).unwrap();
    let day = 86_400_000i64;
    let d0 = (now_ms() / day - 40) * day;
    for core in [&mut c, &mut cref] {
        for k in 0..3 {
            // two flushes a day, each crossing nothing: several L0 files per closed day
            ingest(core, d0 + k * day + 3_600_000, 120, 5);
            core.flush().unwrap();
            ingest(core, d0 + k * day + 7_200_000, 120, 5);
            core.flush().unwrap();
        }
    }
    // upload, no merge before it: every L0 file goes up; then they leave the local dir
    b.sync(&c.shared, None).unwrap();
    let l0: Vec<String> = c
        .shared
        .catalog
        .lock()
        .unwrap()
        .files
        .iter()
        .map(|f| f.path.clone())
        .collect();
    assert!(l0.iter().all(|p| !d.path().join(p).exists()), "evicted");
    let comp = c.compactor().unwrap();
    let n = comp.run_once(now_ms(), true).unwrap();
    assert!(n > 0);
    cref.compactor().unwrap().run_once(now_ms(), true).unwrap();
    let h = b.health();
    assert!(h["fetched"].as_u64().unwrap() >= l0.len() as u64 / 2, "{h}");
    b.sync(&c.shared, None).unwrap();
    // the replaced files left the catalog; with no grace they are deleted on the next round
    b.sync(&c.shared, None).unwrap();
    let cat = bucket_catalog(&store).unwrap();
    let kinds = |cat: &Catalog| {
        cat.files
            .iter()
            .map(|f| (f.kind.clone(), f.day, f.level))
            .collect::<BTreeSet<_>>()
    };
    let rcat = cref.shared.catalog.lock().unwrap().clone();
    assert_eq!(kinds(&cat), kinds(&rcat), "same files as the local store");
    for k in [
        pgai_mon::eval::rollup::KIND,
        pgai_mon::eval::rollup::KIND_1D,
        pgai_mon::registry::KIND,
    ] {
        assert!(cat.files.iter().any(|f| f.kind == k), "{k} built");
    }
    let keys = objects(&store);
    let listed: BTreeSet<String> = cat.files.iter().map(|f| f.path.clone()).collect();
    let replaced: Vec<&String> = l0.iter().filter(|p| !listed.contains(*p)).collect();
    assert!(replaced.len() > l0.len() / 2, "{replaced:?}");
    for p in replaced {
        assert!(!keys.contains(p), "{p} still in the bucket");
    }
    assert_eq!(
        keys.iter()
            .filter(|k| k.ends_with(".parquet"))
            .cloned()
            .collect::<BTreeSet<_>>(),
        listed
    );
    let reference = Reader::new(cref.shared.clone(), HttpCfg::default(), views(), vec![]).unwrap();
    let rem = bucket_reader(url);
    // prom_edges compares the last day's data (two L0 files compacted) over 3 hours at the same
    // 60 s step, not the 3 days: on the bucket reader its ASOF joins run as nested loops (#542),
    // ~16 s for the 3 days' 86,420 rows, past the reader's 15 s timeout (#541). The 3 days can
    // come back once #542 is fixed; the raw rows of all 3 days are compared by prom_select.
    let edges = |q: &String| q.contains("prom_edges(");
    let (e0, e1) = (d0 + 2 * day, d0 + 2 * day + 3 * 3_600_000);
    for q in queries(d0, d0 + 3 * day)
        .into_iter()
        .filter(|q| !edges(q))
        .chain(queries(e0, e1).into_iter().filter(edges))
        .chain([format!(
            "SELECT series_id, t, value FROM prom_increase('m_a', [], {}, {}, {day}, {day}) ORDER BY 1, 2",
            d0 + day,
            d0 + 3 * day
        )])
    {
        let a = rows(&rem, &q);
        assert!(!a.is_empty(), "{q}");
        if edges(&q) {
            // steps with samples in their window, not only the grid around the fetched rows
            let n = |r: &String| serde_json::from_str::<serde_json::Value>(r).unwrap()[2].as_i64();
            assert!(a.iter().any(|r| n(r) > Some(0)), "{q}");
        }
        assert_eq!(a, rows(&reference, &q), "{q}");
    }
    b.close();
}

/// Every commit is a whole snapshot, also while compaction replaces files between commits:
/// after each one the bucket's catalog lists only objects that exist and reads back the rows
/// ingested so far, once each. The objects of the commit before are still there, for a reader
/// that has not seen the new catalog yet (the delete grace).
#[test]
fn every_commit_between_compactions_is_a_whole_snapshot() {
    let url = "mem://snapshots/st";
    let store = faulty();
    let d = tempfile::tempdir().unwrap();
    let cfg = BucketCfg {
        delete_grace_s: 3600,
        keep_local_days: 0,
        ..Default::default()
    };
    let (b, mut c) = start(url, &store, d.path(), cfg).unwrap();
    let (comp, pre) = (c.compactor().unwrap(), c.upload_compactor().unwrap());
    let day = 86_400_000i64;
    let d0 = (now_ms() / day - 5) * day;
    let q = format!("SELECT count(*) FROM pgai_samples('m_a', [], {d0}, {})", d0 + 2 * day);
    let mut before = BTreeSet::new();
    for k in 0..6 {
        // two closed days, three flushes each
        ingest(&mut c, d0 + k * day / 3, 10, 5);
        c.flush().unwrap();
        // the commit of the flush; every other round a compaction and its commit
        for compact in [false, true].into_iter().take(1 + (k % 2) as usize) {
            if compact {
                assert!(comp.run_once(now_ms(), true).unwrap() > 0);
            }
            b.sync(&c.shared, Some(&pre)).unwrap();
            assert_listed_objects_exist(&store);
            let keys = objects(&store);
            assert!(before.is_subset(&keys), "gone early: {:?}", before.difference(&keys));
            assert_eq!(
                rows(&bucket_reader(url), &q),
                vec![format!("[{}]", (k + 1) * 200)],
                "round {k}"
            );
            before = bucket_catalog(&store)
                .unwrap()
                .files
                .into_iter()
                .map(|f| f.path)
                .collect();
        }
    }
    b.close();
}

/// Retention on an object store: the files the writer expires leave the bucket's catalog at the
/// next commit and the bucket after the delete grace; what is kept still answers from the bucket.
#[test]
fn retention_deletes_from_the_bucket() {
    let url = "mem://retention/st";
    let store = faulty();
    let d = tempfile::tempdir().unwrap();
    let b = Bucket::with_store(url, store.clone(), d.path(), bcfg()).unwrap();
    b.attach(pgai_mon::store::DEFAULT_LAYOUT).unwrap();
    let cfg = StoreCfg {
        retention_days: 14,
        ..scfg()
    };
    let mut c = StoreCore::open(&b.local, &cfg).unwrap();
    let day = 86_400_000i64;
    let today = now_ms() / day;
    let (old, t0) = ((today - 40) * day + 3_600_000, start_ts(30));
    ingest(&mut c, old, 120, 5);
    ingest(&mut c, old + day, 120, 5);
    ingest(&mut c, t0, 40, 5);
    c.flush().unwrap();
    b.sync(&c.shared, None).unwrap();
    let day_kinds = ["samples", "series", pgai_mon::eval::rollup::KIND];
    let old_files = |cat: &Catalog| -> BTreeSet<String> {
        let f = cat
            .files
            .iter()
            .filter(|f| day_kinds.contains(&f.kind.as_str()) && f.day < today - 14);
        f.map(|f| f.path.clone()).collect()
    };
    let up = old_files(&bucket_catalog(&store).unwrap());
    assert!(
        !up.is_empty() && up.iter().all(|p| objects(&store).contains(p)),
        "old days uploaded"
    );
    let (t1, q) = (t0 + 40 * 15_000, |a: i64, z: i64| {
        format!("SELECT count(*) FROM pgai_samples('m_a', [], {a}, {z})")
    });

    // a pass compacts the old days, then expires them; two commits later they are deleted
    assert!(c.compactor().unwrap().run_once(now_ms(), true).unwrap() > 0);
    b.sync(&c.shared, None).unwrap();
    b.sync(&c.shared, None).unwrap();
    let cat = bucket_catalog(&store).unwrap();
    assert_eq!(old_files(&cat), BTreeSet::new());
    let listed: BTreeSet<String> = cat.files.iter().map(|f| f.path.clone()).collect();
    let keys: BTreeSet<String> = objects(&store)
        .into_iter()
        .filter(|k| k.ends_with(".parquet"))
        .collect();
    assert_eq!(keys, listed, "the bucket holds exactly the catalog's files");
    let rem = bucket_reader(url);
    assert_eq!(rows(&rem, &q(t0, t1)), ["[800]"], "40 scrapes of 20 series kept");
    assert_eq!(rows(&rem, &q(old, old + 2 * day)), ["[0]"]);
    b.close();
}

const STEP: i64 = 10 * 15_000;

/// The store applies a commit and its answer is lost. The client's retry of the conditional
/// PUT then meets its own write (a failed precondition), or the client gives up (an error).
/// Either way no other writer exists: the writer is not fenced, and the next round goes on.
#[test]
fn a_commit_whose_answer_is_lost_does_not_fence_the_writer() {
    let url = "mem://lost-answer/st";
    let store = faulty();
    let d = tempfile::tempdir().unwrap();
    let t0 = start_ts(30);
    let (b, mut c) = start(url, &store, d.path(), bcfg()).unwrap();
    ingest(&mut c, t0, 10, 7);
    c.flush().unwrap();
    store.lose_answer.store(1, Ordering::SeqCst);
    b.sync(&c.shared, None).expect("the retry met this writer's own commit");
    assert_eq!(bucket_rows(url, t0), "[200]");

    ingest(&mut c, t0 + STEP, 10, 7);
    c.flush().unwrap();
    store.lose_answer.store(2, Ordering::SeqCst);
    let e = b.sync(&c.shared, None).expect_err("the client gave up");
    assert!(e.downcast_ref::<Fenced>().is_none() && !b.is_fenced(), "{e:#}");
    assert_eq!(bucket_rows(url, t0), "[400]", "the commit landed all the same");

    ingest(&mut c, t0 + 2 * STEP, 10, 7);
    c.flush().unwrap();
    b.sync(&c.shared, None)
        .expect("the next round finds its own commit in the bucket and goes on");
    assert!(!b.is_fenced());
    assert_eq!(bucket_rows(url, t0), "[600]");
    assert_listed_objects_exist(&store);
    b.close();
}

/// A writer's sweep runs after another writer took the store over and uploaded files for its
/// first commit: the sweep leaves them alone (a file id carries its writer's epoch). Otherwise
/// the new owner commits a catalog that lists deleted objects.
#[test]
fn a_sweep_never_deletes_a_newer_writers_objects() {
    let url = "mem://sweep-race/st";
    let store = faulty();
    let (d1, d2) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let t0 = start_ts(30);
    let (a, mut ca) = start(url, &store, d1.path(), bcfg()).unwrap();
    ingest(&mut ca, t0, 10, 8);
    ca.flush().unwrap();
    a.sync(&ca.shared, None).unwrap();
    let takeover = BucketCfg {
        takeover: true,
        ..bcfg()
    };
    let (b, mut cb) = start(url, &store, d2.path(), takeover).unwrap();
    ingest(&mut cb, t0 + STEP, 10, 8);
    cb.flush().unwrap();
    // the new owner has uploaded and not committed; the old one has not noticed the takeover
    *b.failpoint.lock().unwrap() = Some(Failpoint::AfterUpload);
    assert!(b.sync(&cb.shared, None).is_err());
    a.sweep().unwrap();
    *b.failpoint.lock().unwrap() = None;
    b.sync(&cb.shared, None).unwrap();
    assert_listed_objects_exist(&store);
    assert_eq!(bucket_rows(url, t0), "[400]");
    a.close();
    b.close();
}

/// A writer is fenced while it holds data it flushed and did not upload, and later starts again
/// on the same directory: it adopts the new owner's catalog and adds exactly the files that
/// none of its commits carried into the bucket. Nothing is lost and nothing is there twice,
/// also when its last commit landed without its answer. A compaction pass in between does not
/// touch those files: on an object store a job waits until the bucket has all its inputs.
#[test]
fn a_fenced_writer_that_starts_again_keeps_what_it_had_not_uploaded() {
    for lost_answer in [false, true] {
        let url = if lost_answer {
            "mem://rescue-lost/st"
        } else {
            "mem://rescue/st"
        };
        let store = faulty();
        let (d1, d2) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let t0 = start_ts(40);
        let (a, mut ca) = start(url, &store, d1.path(), bcfg()).unwrap();
        ingest(&mut ca, t0, 10, 9);
        ca.flush().unwrap();
        a.sync(&ca.shared, None).unwrap();
        ingest(&mut ca, t0 + STEP, 10, 9);
        ca.flush().unwrap();
        if lost_answer {
            // this commit lands and the writer never learns it
            store.lose_answer.store(2, Ordering::SeqCst);
            assert!(a.sync(&ca.shared, None).is_err());
        }
        for k in 2..4 {
            ingest(&mut ca, t0 + k * STEP, 10, 9);
            ca.flush().unwrap();
        }
        // two hours on, these files are from finished hours
        let _ = ca.compactor().unwrap().run_once(now_ms() + 2 * 3_600_000, false);

        let takeover = BucketCfg {
            takeover: true,
            ..bcfg()
        };
        let (b, mut cb) = start(url, &store, d2.path(), takeover).unwrap();
        ingest(&mut cb, t0 + 4 * STEP, 10, 9);
        cb.flush().unwrap();
        b.sync(&cb.shared, None).unwrap();
        let err = a.sync(&ca.shared, None).expect_err("fenced");
        assert!(err.downcast_ref::<Fenced>().is_some(), "{err:#}");
        ingest(&mut ca, t0 + 5 * STEP, 10, 9); // acked, in the WAL only
        drop(ca);
        a.close();
        b.release(&cb.shared).unwrap();
        drop(cb);
        b.close();

        let (a, mut ca) = start(url, &store, d1.path(), bcfg()).unwrap();
        ca.flush().unwrap();
        a.sync(&ca.shared, None).unwrap();
        assert_eq!(bucket_rows(url, t0), "[1200]", "lost_answer: {lost_answer}");
        assert_listed_objects_exist(&store);
        a.close();
    }
}

/// A writer takes the store over from another directory and stops hard before its first
/// commit. The takeover put nothing but the lease into the bucket, so the directory still is
/// the continuation of the bucket: what it flushed and what its WAL holds are kept.
#[test]
fn a_restart_before_the_first_commit_after_a_takeover_keeps_local_data() {
    let url = "mem://takeover-restart/st";
    let store = faulty();
    let (d1, d2) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let t0 = start_ts(40);
    let (a, mut ca) = start(url, &store, d1.path(), bcfg()).unwrap();
    for k in 0..3 {
        ingest(&mut ca, t0 + k * STEP, 10, 10);
        ca.flush().unwrap();
    }
    a.release(&ca.shared).unwrap();
    drop(ca);
    a.close();
    let (b, mut cb) = start(url, &store, d2.path(), bcfg()).unwrap();
    ingest(&mut cb, t0 + 3 * STEP, 10, 10);
    cb.flush().unwrap(); // flushed, not uploaded
    ingest(&mut cb, t0 + 4 * STEP, 10, 10); // in the WAL only
    drop(cb);
    b.close();
    let (b, mut cb) = start(url, &store, d2.path(), bcfg()).unwrap();
    cb.flush().unwrap();
    b.sync(&cb.shared, None).unwrap();
    assert_eq!(bucket_rows(url, t0), "[1000]");
    b.close();
}

/// A fenced writer starts again and stops hard right after its takeover PUT, before it saved
/// the adopted catalog. Its directory still holds its old catalog, which lacks everything the
/// other writer committed: the next start adopts the bucket again and does not continue from
/// the directory. The other writer's compaction replaced the files the first one had
/// committed: the directory still has them, and they are not added again.
#[test]
fn a_takeover_cut_short_does_not_drop_the_other_writers_commits() {
    let url = "mem://takeover-cut/st";
    let store = faulty();
    let (d1, d2) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let t0 = start_ts(40);
    let (a, mut ca) = start(url, &store, d1.path(), bcfg()).unwrap();
    for k in 0..3 {
        ingest(&mut ca, t0 + k * STEP, 10, 12);
        ca.flush().unwrap();
    }
    a.sync(&ca.shared, None).unwrap();
    ingest(&mut ca, t0 + 3 * STEP, 10, 12);
    ca.flush().unwrap(); // flushed, not uploaded
    let (b, mut cb) = start(url, &store, d2.path(), forced()).unwrap();
    ingest(&mut cb, t0 + 4 * STEP, 10, 12);
    cb.flush().unwrap();
    b.sync(&cb.shared, None).unwrap();
    // two hours on, the hour of these files is finished
    assert!(
        cb.compactor()
            .unwrap()
            .run_once(now_ms() + 2 * 3_600_000, false)
            .unwrap()
            > 0
    );
    b.release(&cb.shared).unwrap();
    drop(cb);
    b.close();
    drop(ca);
    a.close();

    cut_short(url, &store, d1.path(), bcfg());
    let (a, ca) = start(url, &store, d1.path(), bcfg()).unwrap();
    a.sync(&ca.shared, None).unwrap();
    assert_eq!(bucket_rows(url, t0), "[1000]");
    assert_listed_objects_exist(&store);
    a.close();
}

/// A writer starts again on its directory and stops hard right after the takeover; another
/// writer takes the store over; the first one comes back. The lease the store was taken over
/// from is of a later epoch than the catalog in its directory, which still holds a file it
/// never uploaded: the file is kept.
#[test]
fn a_takeover_cut_short_and_then_taken_over_keeps_local_files() {
    let url = "mem://takeover-cut-taken/st";
    let store = faulty();
    let (d1, d2) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let t0 = start_ts(40);
    let (a, mut ca) = start(url, &store, d1.path(), bcfg()).unwrap();
    ingest(&mut ca, t0, 10, 14);
    ca.flush().unwrap();
    a.sync(&ca.shared, None).unwrap();
    ingest(&mut ca, t0 + STEP, 10, 14);
    ca.flush().unwrap(); // flushed, not uploaded
    drop(ca);
    a.close();
    cut_short(url, &store, d1.path(), bcfg());
    let (b, mut cb) = start(url, &store, d2.path(), forced()).unwrap();
    ingest(&mut cb, t0 + 2 * STEP, 10, 14);
    cb.flush().unwrap();
    b.release(&cb.shared).unwrap();
    drop(cb);
    b.close();

    let (a, ca) = start(url, &store, d1.path(), bcfg()).unwrap();
    a.sync(&ca.shared, None).unwrap();
    assert_eq!(bucket_rows(url, t0), "[600]");
    a.close();
}

/// A local store is pointed at an empty prefix and the process stops hard right after the
/// takeover, which put a catalog with no files there: the next start continues from the
/// directory, with every file.
#[test]
fn a_first_start_on_an_empty_prefix_cut_short_keeps_the_local_store() {
    let url = "mem://first-cut/st";
    let store = faulty();
    let d = tempfile::tempdir().unwrap();
    let t0 = start_ts(40);
    let mut c = StoreCore::open(d.path(), &scfg()).unwrap();
    ingest(&mut c, t0, 10, 16);
    c.flush().unwrap();
    drop(c);
    cut_short(url, &store, d.path(), bcfg());
    let (b, c) = start(url, &store, d.path(), bcfg()).unwrap();
    b.sync(&c.shared, None).unwrap();
    assert_eq!(bucket_rows(url, t0), "[200]");
    b.close();
}

/// A local store is pointed at an empty prefix, and another directory takes the store over
/// before the first upload commits. The bucket never had any of the store's files: when its
/// directory starts again, the files of every level are kept, not the level-0 ones alone.
#[test]
fn a_local_store_taken_over_before_its_first_upload_keeps_every_file() {
    let url = "mem://moved-to-bucket/st";
    let store = faulty();
    let (d1, d2) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let day = 86_400_000i64;
    let d0 = (now_ms() / day - 5) * day;
    let mut c = StoreCore::open(d1.path(), &scfg()).unwrap();
    for k in 0..2 {
        ingest(&mut c, d0 + 3_600_000 + k * 600_000, 20, 19);
        c.flush().unwrap();
    }
    assert!(c.compactor().unwrap().run_once(now_ms(), true).unwrap() > 0); // the closed day becomes one file
    drop(c);
    let (a, ca) = start(url, &store, d1.path(), bcfg()).unwrap();
    let (b, cb) = start(url, &store, d2.path(), forced()).unwrap();
    b.release(&cb.shared).unwrap();
    drop(cb);
    b.close();
    assert!(a.sync(&ca.shared, None).is_err(), "fenced");
    drop(ca);
    a.close();

    let (a, ca) = start(url, &store, d1.path(), bcfg()).unwrap();
    a.sync(&ca.shared, None).unwrap();
    let q = format!("SELECT count(*) FROM pgai_samples('m_a', [], {d0}, {})", d0 + day);
    assert_eq!(rows(&bucket_reader(url), &q), vec!["[800]".to_string()]);
    assert_listed_objects_exist(&store);
    a.close();
}

/// A copy of a writer's directory from an older snapshot (the same writer id) takes the store
/// over while the writer is down, and the writer comes back. The store's history still has the
/// writer's own last commit: the writer keeps what it had flushed since and what its WAL holds.
#[test]
fn a_writer_keeps_its_data_after_a_copy_of_its_directory_owned_the_store() {
    let url = "mem://copy-owned/st";
    let store = faulty();
    let (d, copy) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let t0 = start_ts(40);
    let (a, mut ca) = start(url, &store, d.path(), bcfg()).unwrap();
    ingest(&mut ca, t0, 10, 17);
    ca.flush().unwrap();
    a.sync(&ca.shared, None).unwrap();
    copy_dir(d.path(), copy.path());
    ingest(&mut ca, t0 + STEP, 10, 17);
    ca.flush().unwrap();
    a.sync(&ca.shared, None).unwrap();
    ingest(&mut ca, t0 + 2 * STEP, 10, 17);
    ca.flush().unwrap(); // flushed, not uploaded
    ingest(&mut ca, t0 + 3 * STEP, 10, 17); // in the WAL only
    drop(ca);
    a.close();

    // the copy's WAL numbering goes past the writer's
    let (x, mut cx) = start(url, &store, copy.path(), forced()).unwrap();
    for k in 4..6 {
        ingest(&mut cx, t0 + k * STEP, 10, 17);
        cx.flush().unwrap();
    }
    x.release(&cx.shared).unwrap();
    drop(cx);
    x.close();

    let (a, mut ca) = start(url, &store, d.path(), bcfg()).unwrap();
    ca.flush().unwrap();
    a.sync(&ca.shared, None).unwrap();
    assert_eq!(bucket_rows(url, t0), "[1200]");
    a.close();
}

/// A writer starts again on its directory, which puts only the lease into the bucket, and its
/// disk is lost before the next upload. A snapshot of the disk from after the last upload is
/// restored: what its WAL holds was in no commit, and is kept.
#[test]
fn a_takeover_does_not_count_unsent_data_as_committed() {
    let url = "mem://takeover-mark/st";
    let store = faulty();
    let (d, snap) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let t0 = start_ts(40);
    let (a, mut ca) = start(url, &store, d.path(), bcfg()).unwrap();
    ingest(&mut ca, t0, 10, 18);
    ca.flush().unwrap();
    a.sync(&ca.shared, None).unwrap();
    ingest(&mut ca, t0 + STEP, 10, 18); // in the WAL only
    copy_dir(d.path(), snap.path());
    ca.flush().unwrap(); // flushed, not uploaded
    drop(ca);
    a.close();
    let (a, ca) = start(url, &store, d.path(), bcfg()).unwrap();
    drop(ca);
    a.close();

    let (x, mut cx) = start(url, &store, snap.path(), forced()).unwrap();
    cx.flush().unwrap();
    x.sync(&cx.shared, None).unwrap();
    assert_eq!(bucket_rows(url, t0), "[400]");
    x.close();
}

/// A directory restored from an older copy, after another writer owned the store, adds nothing
/// a second time. Copy 1 has WAL records that its writer flushed and committed after the copy
/// was taken. Copy 2 was taken after its writer was fenced and has a file that the real
/// directory handed to the store when it started again.
#[test]
fn a_restored_directory_adds_nothing_twice() {
    let t0 = start_ts(40);
    let dirs: Vec<tempfile::TempDir> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
    let (d1, d2, copy) = (dirs[0].path(), dirs[1].path(), dirs[2].path());
    for second in [false, true] {
        let url = format!("mem://restored-{second}/st");
        let store = faulty();
        for d in [d1, d2, copy] {
            std::fs::remove_dir_all(d).unwrap();
            std::fs::create_dir(d).unwrap();
        }
        let (a, mut ca) = start(&url, &store, d1, bcfg()).unwrap();
        ingest(&mut ca, t0, 10, 15);
        ca.flush().unwrap();
        a.sync(&ca.shared, None).unwrap();
        ingest(&mut ca, t0 + STEP, 10, 15); // in the WAL only
        if !second {
            copy_dir(d1, copy);
        }
        ca.flush().unwrap();
        if !second {
            a.sync(&ca.shared, None).unwrap();
        }
        let (b, mut cb) = start(&url, &store, d2, forced()).unwrap();
        ingest(&mut cb, t0 + 2 * STEP, 10, 15);
        cb.flush().unwrap();
        b.release(&cb.shared).unwrap();
        drop(cb);
        b.close();
        assert!(a.sync(&ca.shared, None).is_err(), "fenced");
        drop(ca);
        a.close();
        if second {
            copy_dir(d1, copy);
            // the real directory starts again and hands the file it had not uploaded to the store
            let (a, ca) = start(&url, &store, d1, bcfg()).unwrap();
            a.release(&ca.shared).unwrap();
            drop(ca);
            a.close();
        }
        assert_eq!(bucket_rows(&url, t0), "[600]", "second: {second}");

        let (x, mut cx) = start(&url, &store, copy, bcfg()).unwrap();
        cx.flush().unwrap();
        x.sync(&cx.shared, None).unwrap();
        assert_eq!(bucket_rows(&url, t0), "[600]", "second: {second}");
        x.close();
    }
}

/// A flush (store thread) and the pre-upload merge (uploader thread) run at the same time:
/// each takes its file ids under the catalog lock, so neither writes the other's file.
#[test]
fn a_flush_and_the_pre_upload_merge_never_share_a_file() {
    let d = tempfile::tempdir().unwrap();
    let mut c = StoreCore::open(d.path(), &scfg()).unwrap();
    let merger = c.upload_compactor().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let h = std::thread::spawn({
        let stop = stop.clone();
        move || {
            while !stop.load(Ordering::SeqCst) {
                let _ = merger.merge_unuploaded(&Default::default());
            }
        }
    });
    let t0 = start_ts(50);
    for k in 0..150 {
        ingest(&mut c, t0 + k * 15_000, 1, 11);
        // a second kind: the merge of its files comes first and takes its id while the next
        // flush is already writing
        c.append(Record::QueryTexts(vec![QueryText {
            queryid: k,
            dbname: "db1".into(),
            query_hash: k as u64,
            query: Some(format!("select {k}")),
            first_seen: t0,
            last_seen: t0,
            src: Default::default(),
        }]))
        .unwrap();
        c.flush().unwrap();
    }
    stop.store(true, Ordering::SeqCst);
    h.join().unwrap();
    let files = c.shared.catalog.lock().unwrap().files.clone();
    let paths: BTreeSet<&str> = files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(paths.len(), files.len(), "two catalog entries name one file");
    let r = Reader::new(c.shared.clone(), HttpCfg::default(), views(), vec![]).unwrap();
    let q = format!("SELECT count(*) FROM pgai_samples('m_a', [], {t0}, {})", t0 + 3_600_000);
    assert_eq!(rows(&r, &q), vec!["[3000]".to_string()]);
}

/// The local-disk store does not touch the write path.
#[test]
fn a_local_store_has_no_bucket() {
    let d = tempfile::tempdir().unwrap();
    let c = StoreCore::open(d.path(), &scfg()).unwrap();
    assert!(pgai_mon::bucket::of(&c.shared.root).is_none());
    assert_eq!(pgai_mon::bucket::read_root(&c.shared.root), c.shared.root);
    assert!(StoreCore::open(Path::new("s3://b/p"), &scfg()).is_err());
}

/// The uploader has its own thread: while a slow compaction job runs (each merge held for
/// 6 s), every flushed file still reaches the bucket within an upload interval plus one round,
/// and eviction waits for the job instead of removing its inputs.
#[test]
fn uploads_stay_on_time_during_a_slow_compaction() {
    let url = "mem://slowjob/st";
    let store = faulty();
    let d = tempfile::tempdir().unwrap();
    let cfg = BucketCfg {
        upload_interval_s: 1,
        keep_local_days: 0,
        ..bcfg()
    };
    let (b, mut c) = start(url, &store, d.path(), cfg).unwrap();
    let day = 86_400_000i64;
    let d0 = (now_ms() / day - 5) * day;
    for k in 0..3 {
        ingest(&mut c, d0 + 3_600_000 + k * 600_000, 20, 7);
        c.flush().unwrap();
    }
    *pgai_mon::store::TEST_MERGE_DELAY.lock().unwrap() = Some((c.shared.root.clone(), 6_000));
    let up = pgai_mon::bucket::spawn_uploader(
        b.clone(),
        c.shared.clone(),
        c.upload_compactor().unwrap(),
        Box::new(|| {}),
        Box::new(|e| panic!("fenced: {e}")),
        std::time::Duration::from_secs(1),
    )
    .unwrap();
    // a compaction job waits until the bucket has its inputs: the first round commits them
    let t = std::time::Instant::now();
    while bucket_catalog(&store).unwrap().files_of("samples").count() < 3 {
        assert!(t.elapsed() < std::time::Duration::from_secs(5), "the first round");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let comp = c.compactor().unwrap();
    let job = std::thread::spawn(move || comp.run_once(now_ms(), true).unwrap());
    std::thread::sleep(std::time::Duration::from_millis(500));
    let t_start = start_ts(20);
    let mut worst = std::time::Duration::ZERO;
    for k in 0..3 {
        ingest(&mut c, t_start + k * 300_000, 4, 7);
        c.flush().unwrap();
        let newest = c
            .shared
            .catalog
            .lock()
            .unwrap()
            .files
            .iter()
            .max_by_key(|f| f.created_ms)
            .unwrap()
            .path
            .clone();
        let t = std::time::Instant::now();
        while !bucket_catalog(&store).unwrap().files.iter().any(|f| f.path == newest) {
            assert!(t.elapsed() < std::time::Duration::from_secs(5), "{newest} not uploaded");
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        worst = worst.max(t.elapsed());
    }
    assert!(!job.is_finished(), "the compaction job was still running");
    // interval 1 s + one round (a few small files) + the poll
    assert!(worst < std::time::Duration::from_millis(2_500), "upload took {worst:?}");
    job.join().unwrap();
    *pgai_mon::store::TEST_MERGE_DELAY.lock().unwrap() = None;
    // the job's outputs go up too, and the evicted closed day reads back intact
    std::thread::sleep(std::time::Duration::from_millis(2_500));
    let cat = bucket_catalog(&store).unwrap();
    assert!(
        cat.files.iter().any(|f| f.day == d0 / day && f.level == 2),
        "{:?}",
        cat.files
    );
    let q = format!("SELECT count(*) FROM pgai_samples('m_a', [], {d0}, {})", d0 + day);
    assert_eq!(rows(&bucket_reader(url), &q), vec!["[1200]".to_string()]);
    b.close();
    drop(up);
}
