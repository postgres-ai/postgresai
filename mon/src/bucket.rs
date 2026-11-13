//! Object-store write path: `run` with `store_dir: s3://bucket/prefix`.
//!
//! The writer keeps a local working directory (`store.s3.local_dir`) with exactly the layout of a
//! local store (WAL, Parquet, catalog.json): flush, compaction, registry, month files and rollups
//! run on it unchanged. The bucket is the durable copy and the local directory a cache:
//!
//! - Every `upload_interval_s` (`sync`, on the compactor thread so it never races compaction or
//!   eviction): the L0 files not uploaded yet are merged per (kind, day), every catalog file not
//!   in the bucket is PUT, then catalog.json is PUT conditionally on the ETag of the catalog this
//!   writer wrote last. That PUT is the commit point. Objects are immutable and keys are never
//!   reused, so a reader sees the old catalog with its files or the new one with its files.
//! - Fencing: catalog.json names its writer (`WriterLease`). A writer starts by taking the lease
//!   over (epoch + 1, conditional PUT); a live lease of another writer refuses the start unless
//!   `takeover` is set. A commit whose precondition fails means another writer took over: this
//!   one is fenced and stops. File ids of epoch e start at e << 32, so a fenced writer that has
//!   not noticed yet never PUTs a key the new owner uses.
//! - GC: a file that leaves the committed file list goes to the bucket catalog's
//!   `pending_delete` with a grace time counted from that commit, and is DELETEd after it.
//!   Objects that no catalog lists (an upload whose commit never happened) are swept by a LIST
//!   at start (after the takeover) and every `sweep_interval_s`.
//! - Eviction: after a commit, committed files older than `keep_local_days` leave the local
//!   directory; a job that needs one again fetches it (`ensure_local`).
//! - Restart: when the bucket's last commit came from this local directory (its writer id),
//!   the local directory is the continuation of that commit and stays authoritative (it holds what was not uploaded
//!   yet). Otherwise (an empty disk, or another writer committed since) the bucket's catalog is
//!   adopted and the files a writer reads at start are fetched. Local WAL segments are replayed
//!   either way: they are in no Parquet file.
//!
//! The process's own reader reads the store through its URL with this directory as the
//! objstore overlay (`objstore::set_overlay`): local files, including catalog.json, win.
use crate::catalog::{self, Catalog, FileEntry, WriterLease};
use crate::model::{day_of, now_ms};
use crate::objstore::{self, Remote};
use object_store::path::Path as OPath;
use object_store::{GetOptions, ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload, UpdateVersion};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, RwLock};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct BucketCfg {
    /// Local working directory of a store whose `store_dir` is an object-store URL (WAL,
    /// Parquet not uploaded yet, cache). Default: `DEFAULT_LOCAL_DIR`.
    pub local_dir: Option<PathBuf>,
    /// Seconds between commits to the bucket. A lost machine loses at most the data acked in
    /// the last interval plus one upload round (the uploader has its own thread and flushes the
    /// WAL before each round).
    pub upload_interval_s: u64,
    /// A writer that has not committed for this many upload intervals has lost the store: a
    /// new writer takes it over at start. A fresher lease of another writer refuses the start.
    pub lease_intervals: u64,
    /// Start even when another writer's lease is fresh (it is fenced at its next commit).
    pub takeover: bool,
    /// Committed files of the last this-many days stay in the local directory.
    pub keep_local_days: i64,
    /// Seconds between LIST sweeps of objects no catalog lists.
    pub sweep_interval_s: u64,
    /// Seconds a file left out of the committed catalog stays in the bucket: readers re-read
    /// catalog.json every `http.s3.catalog_ttl_ms` and may hold the previous one for a query.
    pub delete_grace_s: u64,
    /// Files larger than this are uploaded in parts of this size, MB (memory per upload).
    pub part_mb: u64,
}

impl Default for BucketCfg {
    fn default() -> Self {
        Self {
            local_dir: None,
            upload_interval_s: 300,
            lease_intervals: 3,
            takeover: false,
            keep_local_days: 2,
            sweep_interval_s: 86_400,
            delete_grace_s: 600,
            part_mb: 8,
        }
    }
}

impl BucketCfg {
    pub fn lease_ms(&self) -> i64 {
        (self.lease_intervals.max(1) * self.upload_interval_s.max(1)) as i64 * 1000
    }
}

/// Local working directory of an object-store store when `store.s3.local_dir` is not set.
pub const DEFAULT_LOCAL_DIR: &str = "/var/lib/pgai-mon";

/// Kinds whose files cover a month (`day` is the month's first day).
pub(crate) const MONTH_KINDS: &[&str] = &[
    crate::store::SERIES_MONTH,
    crate::eval::rollup::KIND_1D,
    crate::eval::rollup::KIND_1HM,
];

/// Every kind directory a store may hold (the LIST sweep looks only under these).
fn kind_dirs() -> Vec<&'static str> {
    let mut v: Vec<&str> = crate::store::KINDS.to_vec();
    v.extend([
        crate::eval::rollup::KIND,
        crate::eval::rollup::KIND_1D,
        crate::eval::rollup::KIND_1HM,
        crate::registry::KIND,
        crate::store::SERIES_MONTH,
    ]);
    v
}

/// Error of a writer whose commit was refused because another writer owns the store now.
#[derive(Debug)]
pub struct Fenced(pub String);

impl std::fmt::Display for Fenced {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Fenced {}

#[derive(Default)]
struct St {
    /// Version of the catalog.json this writer wrote last (None: none yet, create it).
    version: Option<UpdateVersion>,
    epoch: u64,
    remote_gen: u64,
    /// Files of the last committed catalog (path -> bytes).
    committed: HashMap<String, u64>,
    /// Objects this writer knows are in the bucket (committed or uploaded for the next commit).
    uploaded: HashMap<String, u64>,
    /// Bucket deletes: (path, not before ms), listed in the bucket catalog's pending_delete.
    gc: Vec<(String, i64)>,
    last_commit_ms: i64,
    last_sweep_ms: i64,
    commits: u64,
    uploads: u64,
    evicted: u64,
    fetched: u64,
    swept: u64,
    pending_files: usize,
    pending_bytes: u64,
    fenced: Option<String>,
    last_error: Option<String>,
    /// Upload rounds (sync) that failed.
    upload_errors: u64,
}

/// Fault injection for tests: stop a sync at a point, as a crash would.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Failpoint {
    /// After the uploads, before the catalog PUT.
    AfterUpload,
    /// In `attach`, after the takeover PUT, before the local catalog is saved.
    AfterTakeover,
}

pub struct Bucket {
    /// `s3://bucket/prefix`, no trailing slash.
    pub url: String,
    pub local: PathBuf,
    remote: Arc<Remote>,
    pub cfg: BucketCfg,
    /// This local directory's writer id (`writer.id` in it).
    pub id: String,
    pub host: String,
    st: Mutex<St>,
    sync_lock: Mutex<()>,
    /// Set by `release`: commits from now on mark the lease released.
    releasing: std::sync::atomic::AtomicBool,
    /// Held (read) by a background job while it reads local files; eviction runs only when it
    /// can take it (write), so it never removes a file a job has just fetched.
    pub jobs: RwLock<()>,
    pub failpoint: Mutex<Option<Failpoint>>,
}

static BUCKETS: RwLock<Vec<Arc<Bucket>>> = RwLock::new(Vec::new());

/// The bucket behind a local store root (None: a plain local store).
pub fn of(local_root: &Path) -> Option<Arc<Bucket>> {
    let g = BUCKETS.read().unwrap();
    if g.is_empty() {
        return None;
    }
    let canon = local_root.canonicalize().ok();
    g.iter()
        .find(|b| b.local == local_root || canon.as_ref().is_some_and(|c| *c == b.local))
        .cloned()
}

/// Root the background jobs read inputs from: the bucket URL (local files win through the
/// overlay) for a bucket-backed store, else the local root.
pub fn read_root(local_root: &Path) -> PathBuf {
    match of(local_root) {
        Some(b) => PathBuf::from(&b.url),
        None => local_root.to_path_buf(),
    }
}

/// Make the files present in a bucket-backed store's local directory, fetching the missing
/// ones (no-op for a local store). Returns the number fetched.
pub fn ensure_local(local_root: &Path, files: &[&FileEntry]) -> anyhow::Result<usize> {
    match of(local_root) {
        Some(b) => b.ensure_local(files),
        None => Ok(0),
    }
}

/// /health: the write path of every bucket-backed store of this process (Null: none).
pub fn health_json() -> serde_json::Value {
    let g = BUCKETS.read().unwrap();
    match g.len() {
        0 => serde_json::Value::Null,
        1 => g[0].health(),
        _ => serde_json::Value::Array(g.iter().map(|b| b.health()).collect()),
    }
}

fn hostname() -> String {
    let mut buf = [0u8; 256];
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
    if rc != 0 {
        return String::new();
    }
    let n = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

/// The writer id of a local directory: made once, kept in `<dir>/writer.id`.
fn writer_id(dir: &Path) -> anyhow::Result<String> {
    let p = dir.join("writer.id");
    if let Ok(s) = std::fs::read_to_string(&p) {
        let s = s.trim().to_string();
        if !s.is_empty() {
            return Ok(s);
        }
    }
    let seed = format!(
        "{}-{}-{}-{:?}",
        hostname(),
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
        dir
    );
    let id = format!("{:016x}", xxhash_rust::xxh3::xxh3_64(seed.as_bytes()));
    let tmp = dir.join("writer.id.tmp");
    std::fs::write(&tmp, &id)?;
    std::fs::File::open(&tmp)?.sync_all()?;
    std::fs::rename(&tmp, &p)?;
    Ok(id)
}

fn local_catalog(dir: &Path) -> anyhow::Result<Option<Catalog>> {
    let p = catalog::path(dir);
    if !p.exists() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_slice(&std::fs::read(&p)?)?))
}

/// Write a catalog to the local directory as it is (generation included).
fn write_local_catalog(dir: &Path, c: &Catalog) -> anyhow::Result<()> {
    let mut c = c.clone();
    c.generation = c.generation.saturating_sub(1);
    c.save(dir)
}

/// The day after which a file is no longer written or read by the background jobs.
pub fn last_day(f: &FileEntry) -> i64 {
    if MONTH_KINDS.contains(&f.kind.as_str()) {
        crate::store::month_of(f.day).1 - 1
    } else {
        f.day.max(day_of(f.max_ts))
    }
}

impl Bucket {
    /// Open (once per process) the bucket at `url` with the AWS environment's credentials.
    pub fn open(url: &str, local: &Path, cfg: BucketCfg, s3: &objstore::S3Cfg) -> anyhow::Result<Arc<Self>> {
        let remote = objstore::open(url, s3)?;
        Self::with_remote(remote, local, cfg)
    }

    /// Open over an already registered object store (tests register an InMemory one).
    pub fn with_store(
        url: &str,
        store: Arc<dyn ObjectStore>,
        local: &Path,
        cfg: BucketCfg,
    ) -> anyhow::Result<Arc<Self>> {
        let remote = objstore::register(url, store, objstore::S3Cfg::default())?;
        Self::with_remote(remote, local, cfg)
    }

    fn with_remote(remote: Arc<Remote>, local: &Path, cfg: BucketCfg) -> anyhow::Result<Arc<Self>> {
        std::fs::create_dir_all(local)?;
        let local = local.canonicalize()?;
        let id = writer_id(&local)?;
        objstore::set_overlay(&remote.url, Some(local.clone()))?;
        let b = Arc::new(Bucket {
            url: remote.url.clone(),
            local: local.clone(),
            remote,
            cfg,
            id,
            host: hostname(),
            st: Default::default(),
            sync_lock: Mutex::new(()),
            releasing: Default::default(),
            jobs: RwLock::new(()),
            failpoint: Mutex::new(None),
        });
        let mut g = BUCKETS.write().unwrap();
        g.retain(|x| x.local != local);
        g.push(b.clone());
        Ok(b)
    }

    /// Forget this bucket (tests: a "new process" opens it again).
    pub fn close(&self) {
        BUCKETS.write().unwrap().retain(|x| x.local != self.local);
        let _ = objstore::set_overlay(&self.url, None);
    }

    pub fn remote(&self) -> &Arc<Remote> {
        &self.remote
    }

    fn store(&self) -> Arc<dyn ObjectStore> {
        self.remote.object_store()
    }

    pub fn epoch(&self) -> u64 {
        self.st.lock().unwrap().epoch
    }

    pub fn is_fenced(&self) -> bool {
        self.st.lock().unwrap().fenced.is_some()
    }

    /// Files the last commit listed.
    pub fn committed(&self) -> BTreeSet<String> {
        self.st.lock().unwrap().committed.keys().cloned().collect()
    }

    /// GET catalog.json with its version (None: no store in the bucket yet).
    fn get_catalog(&self) -> anyhow::Result<Option<(Catalog, UpdateVersion)>> {
        let store = self.store();
        let key = self.remote.key_of("catalog.json");
        let r = self.remote.clone();
        let got = self.remote.run(move |_| async move {
            match store.get_opts(&key, GetOptions::default()).await {
                Ok(g) => {
                    anyhow::ensure!(
                        g.meta.size <= catalog::MAX_REMOTE_BYTES,
                        "catalog.json: {} bytes, over the limit",
                        g.meta.size
                    );
                    let v = UpdateVersion {
                        e_tag: g.meta.e_tag.clone(),
                        version: g.meta.version.clone(),
                    };
                    let b = g.bytes().await?;
                    r.stats.meta_gets.fetch_add(1, Ordering::Relaxed);
                    r.stats.gets.fetch_add(1, Ordering::Relaxed);
                    r.stats.bytes.fetch_add(b.len() as u64, Ordering::Relaxed);
                    Ok(Some((b, v)))
                }
                Err(object_store::Error::NotFound { .. }) => Ok(None),
                Err(e) => Err(e.into()),
            }
        })?;
        match got {
            Some((b, v)) => Ok(Some((catalog::parse_remote(&b)?, v))),
            None => Ok(None),
        }
    }

    /// PUT catalog.json conditionally on the version this writer wrote last (create when none).
    /// A refused precondition fences this writer.
    fn put_catalog(&self, st: &mut St, c: &Catalog) -> anyhow::Result<()> {
        let body = serde_json::to_vec(c)?;
        let n = body.len() as u64;
        let mode = match &st.version {
            Some(v) => PutMode::Update(v.clone()),
            None => PutMode::Create,
        };
        let store = self.store();
        let key = self.remote.key_of("catalog.json");
        let res = self.remote.run(move |_| async move {
            Ok(store
                .put_opts(
                    &key,
                    PutPayload::from(body),
                    PutOptions {
                        mode,
                        ..Default::default()
                    },
                )
                .await)
        })?;
        match res {
            Ok(pr) => {
                self.remote.stats.puts.fetch_add(1, Ordering::Relaxed);
                self.remote.stats.put_bytes.fetch_add(n, Ordering::Relaxed);
                st.version = Some(UpdateVersion {
                    e_tag: pr.e_tag,
                    version: pr.version,
                });
                Ok(())
            }
            Err(object_store::Error::Precondition { .. }) | Err(object_store::Error::AlreadyExists { .. }) => {
                self.remote.stats.cas_failures.fetch_add(1, Ordering::Relaxed);
                let who = match self.get_catalog() {
                    Ok(Some((c, _))) => match c.writer {
                        Some(w) => format!(
                            "writer {} on {:?} (epoch {}, last commit {} s ago)",
                            w.id,
                            w.host,
                            w.epoch,
                            (now_ms() - w.heartbeat_ms) / 1000
                        ),
                        None => "a writer without a lease".into(),
                    },
                    Ok(None) => "a writer that deleted catalog.json".into(),
                    Err(e) => format!("unknown ({e:#})"),
                };
                let msg = format!(
                    "{}: catalog.json was changed by another writer, {who}; this writer (id {}, epoch {}) \
                     is fenced and stops writing. Data not uploaded since its last commit stays only in {}",
                    self.url,
                    self.id,
                    st.epoch,
                    self.local.display()
                );
                st.fenced = Some(msg.clone());
                Err(Fenced(msg).into())
            }
            Err(e) => {
                self.remote.stats.errors.fetch_add(1, Ordering::Relaxed);
                Err(e.into())
            }
        }
    }

    fn lease(&self, epoch: u64) -> WriterLease {
        WriterLease {
            id: self.id.clone(),
            epoch,
            host: self.host.clone(),
            heartbeat_ms: now_ms(),
            released: self.releasing.load(Ordering::Relaxed),
        }
    }

    /// Read catalog.json within `timeout`, so a store that cannot be reached (no credentials,
    /// wrong region, no route) fails the start with the reason; the client's own retries take
    /// minutes and log nothing.
    pub fn probe(self: &Arc<Self>, timeout: std::time::Duration) -> anyhow::Result<()> {
        let (tx, rx) = std::sync::mpsc::channel();
        let b = self.clone();
        std::thread::spawn(move || tx.send(b.get_catalog().map(|_| ())));
        match rx.recv_timeout(timeout) {
            Ok(r) => r,
            Err(_) => anyhow::bail!(
                "{}: no answer from the object store in {} s. Check the credentials (AWS_ACCESS_KEY_ID and \
                 AWS_SECRET_ACCESS_KEY, or an instance role; in a container on EC2 the role needs an instance \
                 metadata hop limit of 2), AWS_REGION and the network path",
                self.url,
                timeout.as_secs()
            ),
        }
    }

    /// The last upload round of a clean stop: as `sync`, and the commit marks the lease released,
    /// so a writer starting on this prefix (a new VM, a wiped local directory) takes over at once.
    pub fn release(&self, shared: &crate::store::Shared) -> anyhow::Result<()> {
        // under the lock: a round the uploader thread is in finishes as an ordinary commit
        let _g = self.sync_lock.lock().unwrap();
        self.releasing.store(true, Ordering::Relaxed);
        self.sync_locked(shared, None)
    }

    /// Take the store over before `StoreCore::open` runs on the local directory: check the lease,
    /// commit epoch + 1, and leave in the local directory the catalog to continue from plus the
    /// files a writer reads at start (today's and yesterday's series files, the objects files).
    /// `layout`: the samples layout of a new store.
    pub fn attach(&self, layout: &str) -> anyhow::Result<()> {
        let _g = self.sync_lock.lock().unwrap();
        let mut st = self.st.lock().unwrap();
        let local = local_catalog(&self.local)?;
        let now = now_ms();
        let (mut cat, remote_cat) = match self.get_catalog()? {
            None => {
                // a new store in the bucket: the local one (if any) is uploaded by the first sync
                let mut c = local.clone().unwrap_or_else(|| Catalog {
                    samples_layout: layout.to_string(),
                    flushed_through_seg: -1,
                    ..Default::default()
                });
                c.pending_delete.clear();
                st.epoch = c.writer.as_ref().map_or(0, |w| w.epoch) + 1;
                let mut r = c.clone();
                r.files.clear();
                r.flushed_through_seg = -1;
                r.log_offsets.clear();
                st.version = None;
                (c, r)
            }
            Some((r, ver)) => {
                if let Some(w) = &r.writer {
                    let age = now - w.heartbeat_ms;
                    if w.id != self.id && !w.released && age < self.cfg.lease_ms() && !self.cfg.takeover {
                        anyhow::bail!(
                            "{}: the store is written by another pgai-mon (writer {} on {:?}, last commit {} s ago); \
                             refusing to start a second writer on the same prefix. Stop the other writer, or wait \
                             {} s: a lease without a commit for store.s3.lease_intervals x upload_interval_s is \
                             taken over at start (store.s3.takeover: true forces it now)",
                            self.url,
                            w.id,
                            w.host,
                            age / 1000,
                            (self.cfg.lease_ms() - age) / 1000 + 1
                        );
                    }
                    if w.id != self.id {
                        eprintln!(
                            "bucket: {}: taking the store over from writer {} on {:?} (epoch {}, last commit {} s ago{})",
                            self.url,
                            w.id,
                            w.host,
                            w.epoch,
                            age / 1000,
                            if w.released {
                                ", released"
                            } else if age < self.cfg.lease_ms() {
                                ", forced"
                            } else {
                                ", lease expired"
                            }
                        );
                    }
                }
                // The bucket's last commit came from this directory (its id): every commit is a
                // snapshot of the local catalog, which is saved on every change, so the local
                // catalog is a superset (up to compaction) even when a crash came between the
                // takeover PUT and the local save below.
                // Not when the directory is behind the bucket (restored from an older disk
                // snapshot): committing it would drop every file added since, so the bucket's
                // catalog is adopted instead.
                let ours = r.writer.as_ref().is_some_and(|w| w.id == self.id);
                let same = match &local {
                    Some(l) => {
                        // (file ids cannot tell: a takeover moves the bucket's ahead)
                        ours && l.writer.as_ref().is_some_and(|lw| lw.id == self.id)
                            && l.flushed_through_seg >= r.flushed_through_seg
                    }
                    None => false,
                };
                st.epoch = r.writer.as_ref().map_or(0, |w| w.epoch) + 1;
                st.version = Some(ver);
                st.remote_gen = r.generation;
                st.gc = r.pending_delete.clone();
                for f in &r.files {
                    st.committed.insert(f.path.clone(), f.bytes);
                    st.uploaded.insert(f.path.clone(), f.bytes);
                }
                let c = if same {
                    // this directory made the last commit: it is a superset of the bucket
                    let mut l = local.clone().unwrap();
                    l.generation = l.generation.max(r.generation);
                    l
                } else {
                    let mut c = r.clone();
                    c.pending_delete.clear();
                    // local WAL numbering: segments this directory already flushed are gone. A
                    // directory of the same lineage that is behind the bucket keeps the bucket's
                    // mark: its older WAL segments are already in the bucket's files.
                    c.flushed_through_seg = local.as_ref().map_or(-1, |l| l.flushed_through_seg);
                    if ours && local.is_some() {
                        c.flushed_through_seg = c.flushed_through_seg.max(r.flushed_through_seg);
                    }
                    if let Some(l) = &local {
                        c.generation = c.generation.max(l.generation);
                    }
                    c
                };
                (c, r)
            }
        };
        let epoch = st.epoch;
        cat.writer = Some(self.lease(epoch));
        cat.next_file_id = cat.next_file_id.max(epoch << 32);
        let mut rc = remote_cat;
        rc.writer = cat.writer.clone();
        rc.next_file_id = rc.next_file_id.max(cat.next_file_id);
        rc.pending_delete = st.gc.clone();
        rc.generation = rc.generation.max(st.remote_gen) + 1;
        self.put_catalog(&mut st, &rc)?;
        st.remote_gen = rc.generation;
        st.last_commit_ms = now;
        st.commits += 1;
        if *self.failpoint.lock().unwrap() == Some(Failpoint::AfterTakeover) {
            anyhow::bail!("failpoint AfterTakeover");
        }
        // the local catalog may list more than the bucket's: never the same generation
        cat.generation = cat.generation.max(rc.generation + 1);
        write_local_catalog(&self.local, &cat)?;
        drop(st);
        let today = day_of(now);
        let need: Vec<&FileEntry> = cat
            .files
            .iter()
            .filter(|f| (f.kind == "series" && f.day >= today - 1) || f.kind == "objects")
            .collect();
        self.ensure_local(&need)?;
        if let Err(e) = self.sweep() {
            eprintln!("bucket: sweep: {e:#}");
        }
        Ok(())
    }

    /// Fetch the files missing from the local directory. Returns how many were fetched.
    pub fn ensure_local(&self, files: &[&FileEntry]) -> anyhow::Result<usize> {
        let mut n = 0;
        for f in files {
            let dst = self.local.join(&f.path);
            if dst.is_file() {
                continue;
            }
            anyhow::ensure!(catalog::plain_rel(&f.path), "bad catalog path {:?}", f.path);
            std::fs::create_dir_all(dst.parent().unwrap())?;
            let src = PathBuf::from(format!("{}/{}", self.url, f.path));
            objstore::download(&src, &dst, f.bytes)?;
            n += 1;
        }
        if n > 0 {
            self.st.lock().unwrap().fetched += n as u64;
        }
        Ok(n)
    }

    /// PUT one local file (multipart past `part_mb`). Memory: one part.
    fn upload(&self, rel: &str, bytes: u64) -> anyhow::Result<()> {
        let path = self.local.join(rel);
        let mut f = std::fs::File::open(&path)?;
        let n = f.metadata()?.len();
        anyhow::ensure!(n == bytes, "{rel}: local file is {n} bytes, the catalog says {bytes}");
        let part = (self.cfg.part_mb.max(5) << 20) as usize;
        let store = self.store();
        let key = self.remote.key_of(rel);
        let stats = &self.remote.stats;
        if n as usize <= part {
            let mut buf = Vec::with_capacity(n as usize);
            f.read_to_end(&mut buf)?;
            self.remote.run(move |_| async move {
                store.put(&key, PutPayload::from(buf)).await?;
                Ok(())
            })?;
            stats.puts.fetch_add(1, Ordering::Relaxed);
            stats.put_bytes.fetch_add(n, Ordering::Relaxed);
            return Ok(());
        }
        let k = key.clone();
        let st2 = store.clone();
        let up = self
            .remote
            .run(move |_| async move { Ok(st2.put_multipart(&k).await?) })?;
        let mut w = Some(object_store::WriteMultipart::new_with_chunk_size(up, part));
        let mut sent = 0u64;
        let res = (|| -> anyhow::Result<()> {
            let mut buf = vec![0u8; part];
            loop {
                let mut got = 0;
                while got < part {
                    let r = f.read(&mut buf[got..])?;
                    if r == 0 {
                        break;
                    }
                    got += r;
                }
                if got == 0 {
                    break;
                }
                let mut wm = w.take().unwrap();
                let chunk = bytes::Bytes::copy_from_slice(&buf[..got]);
                wm = self.remote.run(move |_| async move {
                    wm.wait_for_capacity(1).await?;
                    wm.put(chunk);
                    Ok(wm)
                })?;
                w = Some(wm);
                sent += got as u64;
                stats.puts.fetch_add(1, Ordering::Relaxed);
                if got < part {
                    break;
                }
            }
            anyhow::ensure!(sent == n, "{rel}: read {sent} of {n} bytes");
            let wm = w.take().unwrap();
            self.remote.run(move |_| async move {
                wm.finish().await?;
                Ok(())
            })?;
            Ok(())
        })();
        match res {
            Ok(()) => {
                // create + complete
                stats.puts.fetch_add(2, Ordering::Relaxed);
                stats.put_bytes.fetch_add(n, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                if let Some(wm) = w.take() {
                    let _ = self.remote.run(move |_| async move {
                        let _ = wm.abort().await;
                        Ok(())
                    });
                }
                stats.errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    fn delete(&self, rel: &str) -> anyhow::Result<()> {
        let store = self.store();
        let key = self.remote.key_of(rel);
        self.remote.run(move |_| async move {
            match store.delete(&key).await {
                Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
                Err(e) => Err(e.into()),
            }
        })?;
        self.remote.stats.deletes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// One upload round: merge the L0 files not uploaded yet (with `compactor`), upload what
    /// the local catalog lists and the bucket lacks, commit catalog.json, delete what is due,
    /// evict old committed files locally, sweep when due. The local catalog is `shared`'s.
    pub fn sync(
        &self,
        shared: &crate::store::Shared,
        compactor: Option<&crate::store::Compactor>,
    ) -> anyhow::Result<()> {
        let _g = self.sync_lock.lock().unwrap();
        self.sync_locked(shared, compactor)
    }

    fn sync_locked(
        &self,
        shared: &crate::store::Shared,
        compactor: Option<&crate::store::Compactor>,
    ) -> anyhow::Result<()> {
        if let Some(m) = &self.st.lock().unwrap().fenced {
            return Err(Fenced(m.clone()).into());
        }
        let res = self.sync_inner(shared, compactor);
        let mut st = self.st.lock().unwrap();
        st.last_error = res.as_ref().err().map(|e| format!("{e:#}"));
        st.upload_errors += res.is_err() as u64;
        res
    }

    fn sync_inner(
        &self,
        shared: &crate::store::Shared,
        compactor: Option<&crate::store::Compactor>,
    ) -> anyhow::Result<()> {
        if let Some(c) = compactor {
            let up: HashSet<String> = self.st.lock().unwrap().uploaded.keys().cloned().collect();
            if let Err(e) = c.merge_unuploaded(&up) {
                eprintln!("bucket: merge before upload: {e:#}");
            }
        }
        let snap = shared.catalog.lock().unwrap().clone();
        let todo: Vec<(String, u64)> = {
            let st = self.st.lock().unwrap();
            snap.files
                .iter()
                .filter(|f| !st.uploaded.contains_key(&f.path))
                .map(|f| (f.path.clone(), f.bytes))
                .collect()
        };
        {
            let mut st = self.st.lock().unwrap();
            st.pending_files = todo.len();
            st.pending_bytes = todo.iter().map(|x| x.1).sum();
        }
        for (rel, bytes) in &todo {
            self.upload(rel, *bytes)?;
            let mut st = self.st.lock().unwrap();
            st.uploaded.insert(rel.clone(), *bytes);
            st.uploads += 1;
            st.pending_files -= 1;
            st.pending_bytes -= bytes;
        }
        if *self.failpoint.lock().unwrap() == Some(Failpoint::AfterUpload) {
            anyhow::bail!("failpoint AfterUpload");
        }
        let now = now_ms();
        let mut st = self.st.lock().unwrap();
        let live: HashSet<&str> = snap.files.iter().map(|f| f.path.as_str()).collect();
        let grace = self.cfg.delete_grace_s as i64 * 1000;
        let mut gc = st.gc.clone();
        let listed: HashSet<String> = gc.iter().map(|x| x.0.clone()).collect();
        for p in st.committed.keys() {
            if !live.contains(p.as_str()) && !listed.contains(p) {
                gc.push((p.clone(), now + grace));
            }
        }
        // uploaded for an earlier commit that never happened, and gone from the catalog since:
        // no reader ever saw them
        for p in st.uploaded.keys() {
            if !live.contains(p.as_str()) && !st.committed.contains_key(p) && !listed.contains(p) {
                gc.push((p.clone(), now));
            }
        }
        let mut rc = snap.clone();
        rc.pending_delete = gc.clone();
        rc.writer = Some(self.lease(st.epoch));
        rc.generation = rc.generation.max(st.remote_gen + 1);
        self.put_catalog(&mut st, &rc)?;
        st.remote_gen = rc.generation;
        st.gc = gc;
        st.committed = snap.files.iter().map(|f| (f.path.clone(), f.bytes)).collect();
        st.last_commit_ms = now;
        st.commits += 1;
        let due: Vec<String> = st.gc.iter().filter(|x| x.1 <= now).map(|x| x.0.clone()).collect();
        drop(st);
        let mut deleted = HashSet::new();
        for p in due {
            match self.delete(&p) {
                Ok(()) => {
                    deleted.insert(p);
                }
                Err(e) => eprintln!("bucket: delete {p}: {e:#}"),
            }
        }
        {
            let mut st = self.st.lock().unwrap();
            st.gc.retain(|x| !deleted.contains(&x.0));
            for p in &deleted {
                st.uploaded.remove(p);
            }
        }
        self.evict(&snap);
        let due_sweep = {
            let st = self.st.lock().unwrap();
            now - st.last_sweep_ms >= self.cfg.sweep_interval_s as i64 * 1000
        };
        // not on the releasing round: the next writer may already be uploading
        if due_sweep && !self.releasing.load(Ordering::Relaxed) {
            if let Err(e) = self.sweep() {
                eprintln!("bucket: sweep: {e:#}");
            }
        }
        Ok(())
    }

    /// Remove committed files whose days are past `keep_local_days` from the local directory
    /// (never `objects`: the collector reads them all at start).
    fn evict(&self, snap: &Catalog) {
        // a compaction job is reading local files: evict next round
        let Ok(_w) = self.jobs.try_write() else {
            return;
        };
        let today = day_of(now_ms());
        let committed: HashSet<String> = self.st.lock().unwrap().committed.keys().cloned().collect();
        let mut n = 0;
        for f in &snap.files {
            if f.kind == "objects" || !committed.contains(&f.path) || last_day(f) >= today - self.cfg.keep_local_days {
                continue;
            }
            if std::fs::remove_file(self.local.join(&f.path)).is_ok() {
                n += 1;
            }
        }
        self.st.lock().unwrap().evicted += n;
    }

    /// Delete Parquet objects that neither the committed catalog nor its pending deletes list
    /// and that this writer did not upload for its next commit. Returns how many.
    pub fn sweep(&self) -> anyhow::Result<usize> {
        let keep: HashSet<String> = {
            let st = self.st.lock().unwrap();
            st.committed
                .keys()
                .chain(st.uploaded.keys())
                .cloned()
                .chain(st.gc.iter().map(|x| x.0.clone()))
                .collect()
        };
        let mut orphans = Vec::new();
        for kind in kind_dirs() {
            let store = self.store();
            let prefix = self.remote.key_of(kind);
            let metas: Vec<OPath> = self.remote.run(move |_| async move {
                let mut s = store.list(Some(&prefix));
                let mut v = Vec::new();
                while let Some(m) = std::future::poll_fn(|cx| s.as_mut().poll_next(cx)).await {
                    v.push(m?.location);
                }
                Ok(v)
            })?;
            self.remote.stats.lists.fetch_add(1, Ordering::Relaxed);
            for k in metas {
                let Some(rel) = self.remote.rel_of(&k) else { continue };
                if rel.ends_with(".parquet") && !keep.contains(&rel) {
                    orphans.push(rel);
                }
            }
        }
        let mut n = 0;
        for rel in &orphans {
            eprintln!("bucket: removing orphan object {rel}");
            if self.delete(rel).is_ok() {
                n += 1;
            }
        }
        let mut st = self.st.lock().unwrap();
        st.last_sweep_ms = now_ms();
        st.swept += n as u64;
        Ok(n)
    }

    pub fn health(&self) -> serde_json::Value {
        let st = self.st.lock().unwrap();
        let s = &self.remote.stats;
        let l = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
        serde_json::json!({
            "url": self.url,
            "local_dir": self.local,
            "writer_id": self.id,
            "epoch": st.epoch,
            "commits": st.commits,
            "last_commit_age_s": if st.last_commit_ms > 0 { Some((now_ms() - st.last_commit_ms) / 1000) } else { None },
            "upload_interval_s": self.cfg.upload_interval_s,
            "uploads": st.uploads,
            "pending_files": st.pending_files,
            "pending_bytes": st.pending_bytes,
            "committed_files": st.committed.len(),
            "gc_pending": st.gc.len(),
            "evicted": st.evicted,
            "fetched": st.fetched,
            "swept": st.swept,
            "puts": l(&s.puts),
            "put_bytes": l(&s.put_bytes),
            "deletes": l(&s.deletes),
            "lists": l(&s.lists),
            "cas_failures": l(&s.cas_failures),
            "fenced": st.fenced,
            "last_error": st.last_error,
            "upload_errors": st.upload_errors,
        })
    }
}

/// Milliseconds until the next sync is due (0: now).
pub fn next_sync_in_ms(b: &Bucket) -> i64 {
    let st = b.st.lock().unwrap();
    (st.last_commit_ms + b.cfg.upload_interval_s as i64 * 1000 - now_ms()).max(0)
}

/// The uploader thread: every `upload_interval_s`, `flush` (the WAL to local Parquet; `run`
/// asks the store thread and waits) and then `sync` with `premerge` (a compactor of its own:
/// compaction jobs on the compactor thread never delay an upload). A failed round is retried
/// after `retry`; a fenced writer calls `on_fenced` and the thread ends.
pub fn spawn_uploader(
    b: Arc<Bucket>,
    shared: crate::store::Shared,
    premerge: crate::store::Compactor,
    flush: Box<dyn Fn() + Send>,
    on_fenced: Box<dyn Fn(&str) + Send>,
    retry: std::time::Duration,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new().name("uploader".into()).spawn(move || {
        let mut retry_at = std::time::Instant::now();
        loop {
            let wait = (next_sync_in_ms(&b) as u64).max(
                retry_at
                    .saturating_duration_since(std::time::Instant::now())
                    .as_millis() as u64,
            );
            std::thread::sleep(std::time::Duration::from_millis(wait.clamp(20, 5_000)));
            if next_sync_in_ms(&b) > 0 || std::time::Instant::now() < retry_at {
                continue;
            }
            flush();
            if let Err(e) = b.sync(&shared, Some(&premerge)) {
                if b.is_fenced() {
                    on_fenced(&format!("{e:#}"));
                    return;
                }
                eprintln!(
                    "bucket: upload failed, retried in {} s (data stays local): {e:#}",
                    retry.as_secs()
                );
                retry_at = std::time::Instant::now() + retry;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DAY_MS;

    #[test]
    fn last_day_of_month_files_is_month_end() {
        let f = |kind: &str, day: i64| FileEntry {
            kind: kind.into(),
            path: "x".into(),
            day,
            level: 2,
            rows: 1,
            bytes: 1,
            min_ts: day * DAY_MS,
            max_ts: day * DAY_MS,
            created_ms: 0,
            registered: false,
            ver: 0,
        };
        let jan1 = crate::store::chrono_free::days_from_civil(2026, 1, 1);
        assert_eq!(last_day(&f(crate::eval::rollup::KIND_1D, jan1)), jan1 + 30);
        assert_eq!(last_day(&f("samples", jan1 + 3)), jan1 + 3);
    }

    #[test]
    fn lease_default_is_three_intervals() {
        let c = BucketCfg::default();
        assert_eq!(c.lease_ms(), 900_000);
        let c = BucketCfg {
            upload_interval_s: 60,
            lease_intervals: 5,
            ..c
        };
        assert_eq!(c.lease_ms(), 300_000);
    }
}
