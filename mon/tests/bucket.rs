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
/// multipart uploads fail at part `fail_part` (1-based; -1 = never).
#[derive(Debug, Default)]
struct Faulty {
    inner: InMemory,
    fail_data_puts: AtomicBool,
    fail_part: AtomicI64,
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
    let err = a.sync(&ca.shared, None).err().expect("fenced");
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
    let labels = vec![("i".to_string(), "big".to_string())];
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
    let e = b.sync(&c.shared, None).err().expect("part 2 fails");
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
    for q in queries(d0, d0 + 3 * day).into_iter().chain([format!(
        "SELECT series_id, t, value FROM prom_increase('m_a', [], {}, {}, {day}, {day}) ORDER BY 1, 2",
        d0 + day,
        d0 + 3 * day
    )]) {
        let a = rows(&rem, &q);
        assert!(!a.is_empty(), "{q}");
        assert_eq!(a, rows(&reference, &q), "{q}");
    }
    b.close();
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

/// Measurement (not a gate): one hour at the default intervals (60 s flushes, a commit every
/// 5 flushes) with a load like the live benchmark store (~880 samples/s, log lines and
/// auto_explain plan events every flush, query texts every 5 min), then the day's closing
/// compaction. Prints PUTs per round and the file counts before and after compaction.
/// `cargo test --test bucket measure_puts -- --ignored --nocapture`
#[test]
#[ignore]
fn measure_puts_per_hour_and_compaction() {
    let url = "mem://measure/st";
    let store = faulty();
    let d = tempfile::tempdir().unwrap();
    let (b, mut c) = start(url, &store, d.path(), bcfg()).unwrap();
    let comp = c.compactor().unwrap();
    // today, an open day: the merge before upload takes the open days' current-hour files
    let t0 = start_ts(70);
    let ids: Vec<u64> = (0..13_000u64)
        .map(|i| series_id("m", &[("i".into(), i.to_string())]))
        .collect();
    c.append(Record::Series(
        ids.iter()
            .enumerate()
            .map(|(i, id)| SeriesDef {
                series_id: *id,
                metric: "m".into(),
                labels: vec![("i".into(), i.to_string())],
                first_seen: t0,
            })
            .collect(),
    ))
    .unwrap();
    let puts = |b: &Bucket| b.remote().stats.puts.load(Ordering::Relaxed);
    let mut per_round = vec![];
    let t_start = std::time::Instant::now();
    for round in 0..12i64 {
        let p0 = puts(&b);
        for f in 0..5i64 {
            let tf = t0 + (round * 5 + f) * 60_000;
            for s in 0..4i64 {
                let ts = tf + s * 15_000;
                c.append(Record::Samples(
                    ids.iter()
                        .map(|id| Sample {
                            series_id: *id,
                            ts,
                            value: ((*id % 1000) as i64 + ts / 15_000) as f64,
                        })
                        .collect(),
                ))
                .unwrap();
            }
            let rows: Vec<LogRow> = (0..50)
                .map(|k| LogRow {
                    ts: tf + k * 1000,
                    level: Some("LOG".into()),
                    sqlstate: None,
                    pid: Some(100 + k),
                    dbname: Some("db1".into()),
                    username: Some("u".into()),
                    application: None,
                    message: format!("duration: {k}.5 ms  plan: ..."),
                    query_fingerprint: None,
                    plan_hash: Some(format!("{:x}", k % 7)),
                    queryid: Some(k % 7),
                    raw: "{\"x\":1}".repeat(20),
                })
                .collect();
            let pe: Vec<PlanEvent> = (0..20)
                .map(|k| PlanEvent {
                    ts: tf + k * 2000,
                    plan_hash: format!("{:x}", k % 7),
                    queryid: Some(k % 7),
                    dbname: Some("db1".into()),
                    total_ms: k as f64,
                    top_node: "Seq Scan".into(),
                    shared_hit: k,
                    shared_read: 0,
                    shared_dirtied: 0,
                    shared_written: 0,
                    temp_read: 0,
                    temp_written: 0,
                })
                .collect();
            c.append(Record::Logs {
                rows,
                plan_events: pe,
                plan_defs: vec![],
                file: "pg.json".into(),
                offset: ((round * 5 + f) * 1000) as u64,
            })
            .unwrap();
            if f == 4 {
                c.append(Record::QueryTexts(
                    (0..30)
                        .map(|q| QueryText {
                            queryid: q,
                            dbname: "db1".into(),
                            query_hash: q as u64,
                            query: Some(format!("select {q}")),
                            first_seen: t0,
                            last_seen: tf,
                        })
                        .collect(),
                ))
                .unwrap();
            }
            c.flush().unwrap();
        }
        b.sync(&c.shared, Some(&comp)).unwrap();
        per_round.push(puts(&b) - p0);
    }
    let hour_files = bucket_catalog(&store).unwrap().files.len();
    let p_hour = puts(&b);
    let bytes_hour = b.remote().stats.put_bytes.load(Ordering::Relaxed);
    comp.run_once(now_ms(), true).unwrap();
    let p0 = puts(&b);
    b.sync(&c.shared, Some(&comp)).unwrap();
    let p_compact = puts(&b) - p0;
    let cat = bucket_catalog(&store).unwrap();
    let mut by_kind = std::collections::BTreeMap::new();
    for f in &cat.files {
        *by_kind.entry(f.kind.clone()).or_insert(0) += 1;
    }
    println!(
        "PUTs per round (5 flushes, incl. catalog): {per_round:?}\n\
         PUTs for the hour: {p_hour} ({} MB); files in the bucket after the hour: {hour_files}\n\
         closing compaction: {p_compact} PUTs; files after: {} {by_kind:?}; deletes: {}; wall {:?}",
        bytes_hour >> 20,
        cat.files.len(),
        b.remote().stats.deletes.load(Ordering::Relaxed),
        t_start.elapsed()
    );
    b.close();
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
