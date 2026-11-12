//! Object-store read path: `serve --store s3://bucket/prefix`.
//!
//! A remote store root is kept as a path-like string (`s3://bucket/prefix`), so every
//! `root.join(&file.path)` in the readers still names one object. `lookup` maps such a path to
//! its registered store and key; local paths never match and keep using `std::fs`.
//!
//! The Parquet readers are synchronous and read column chunks one at a time, which over S3
//! would serialise one round trip per chunk. Instead each reader plans its reads up front
//! (`Src::prefetch`: the column chunks of the row groups and columns it will decode). The plan
//! is coalesced into range GETs (adjacent chunks closer than `coalesce_kb`, at most
//! `MAX_PART` per request) and queued; a small tokio runtime issues them with up to
//! `concurrency` requests in flight, while the queued and unread bytes stay under
//! `inflight_mb` (the prefetch window). A decode thread that needs a range blocks only for that
//! range, and a range it needs that is still queued jumps the queue and ignores the byte bound,
//! so a reader can never wait on a window filled by others. A range is released once the reader
//! has read its last byte, or when the file handle is dropped. Reads outside any plan (footers
//! on a cache miss, and anything a caller did not plan) are fetched on demand.
//!
//! Memory: a GET's body lives in its own anonymous mapping (`MapBuf`), not on the malloc heap,
//! so a released range goes back to the OS at once and the reader's RSS tracks the window. In
//! the reader process the window is also bounded by memory headroom (`set_window_limit`, from
//! `sqlproc`): near its RSS cap the reader prefetches less, down to one range at a time, instead
//! of being killed.
//!
//! With `cache_dir`, every fetched column chunk is also written to a local LRU cache bounded by
//! `cache_gb`, and a later plan reads cached chunks from disk instead of GETting them.
use bytes::Bytes;
use object_store::path::Path as OPath;
use object_store::{GetOptions, GetRange, ObjectStore, ObjectStoreExt};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Instant;

/// Largest planned range GET; bigger runs are split so they use the parallelism. A GET is held
/// until every chunk in it is read, so this also bounds what one range adds over the window.
pub const MAX_PART: u64 = 4 << 20;
/// Range GET size of a mirror copy (`download`), written to disk as it arrives.
const COPY_PART: u64 = 8 << 20;
/// Readahead of an unplanned read (page headers, footers past the suffix GET).
const DEMAND_MIN: u64 = 64 << 10;
/// Suffix GET that fetches a footer (and the object length) in one request. Every file kind's
/// footer is written under it (rollup and sample writers keep stats only on pruning columns and
/// size row groups for it): a longer footer costs a second GET.
pub const FOOTER_SUFFIX: u64 = 256 << 10;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct S3Cfg {
    /// Range GETs in flight at once.
    pub concurrency: usize,
    /// Prefetch window: bytes requested or fetched but not yet read, MB.
    pub inflight_mb: u64,
    /// Coalesce two planned ranges into one GET when the gap between them is at most this, KB.
    pub coalesce_kb: u64,
    /// Local cache of fetched column chunks (None = off) and its size bound.
    pub cache_dir: Option<PathBuf>,
    pub cache_gb: f64,
    /// How often the reader re-reads catalog.json from the store, ms.
    pub catalog_ttl_ms: u64,
    /// Raw footer bytes kept in memory (they survive the parsed-footer cache's evictions; with
    /// `cache_dir` they are also kept on disk and survive reader restarts), MB. Counted as the
    /// reader's cache memory (the idle recycle discounts it).
    pub footer_mb: u64,
    /// A query that issues more GETs than this gets a warning in its response stats.
    pub gets_warn: u64,
    /// The reader bounds its prefetch window so its RSS stays under this share of its cap, %
    /// (sqlproc::spawn_window_governor); 0 = no bound.
    pub window_soft_pct: u64,
    /// SQL on a remote store: budget of the local copies DuckDB reads (all kinds together), MB.
    pub mirror_mb: u64,
}

impl Default for S3Cfg {
    fn default() -> Self {
        Self {
            concurrency: 16,
            inflight_mb: 8,
            coalesce_kb: 256,
            cache_dir: None,
            cache_gb: 2.0,
            catalog_ttl_ms: 30_000,
            footer_mb: 8,
            gets_warn: 64,
            window_soft_pct: 85,
            mirror_mb: 2048,
        }
    }
}

#[derive(Default)]
pub struct Stats {
    pub gets: AtomicU64,
    pub bytes: AtomicU64,
    /// Range GETs of Parquet data not covered by a plan (a reader asked for bytes nobody planned).
    pub demand_gets: AtomicU64,
    /// Whole-object reads outside the query path: catalog.json reloads and mirror copies.
    pub meta_gets: AtomicU64,
    /// Times the window stopped short of `inflight_mb` on the memory bound (`set_window_limit`).
    pub window_limited: AtomicU64,
    pub errors: AtomicU64,
    pub inflight_peak: AtomicU64,
    pub held_peak_bytes: AtomicU64,
    /// Time decode threads spent blocked on a range, and summed GET latency, ns.
    pub wait_ns: AtomicU64,
    pub get_ns: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub cache_hit_bytes: AtomicU64,
    /// Footers: served from the raw-footer cache (memory, disk) and fetched (with how many GETs).
    pub footer_mem_hits: AtomicU64,
    pub footer_disk_hits: AtomicU64,
    pub footer_fetches: AtomicU64,
    pub footer_gets: AtomicU64,
}

enum PState {
    Queued,
    /// In the local cache: read from disk when first needed.
    Cached,
    Running,
    Ready(Bytes),
    Failed(String),
    /// Read to the end, or its file was dropped.
    Done,
}

struct PInner {
    state: PState,
    /// Counted in the scheduler's window (`Sched::held`).
    charged: bool,
    cancelled: bool,
}

struct Part {
    key: Arc<OPath>,
    start: u64,
    end: u64,
    /// Column-chunk ranges inside this GET, for the local cache.
    chunks: Vec<(u64, u64)>,
    /// Queued by a plan (not an unplanned read).
    planned: bool,
    /// Kept until `clear` or drop even when read to its end (a footer, read several times).
    sticky: bool,
    /// Ends of the column chunks not read to their end yet. A coalesced GET holds several
    /// chunks read side by side (one column reader each, page by page), so reaching the GET's
    /// last byte does not mean the other chunks are done: it is released when all are.
    open_ends: Mutex<Vec<u64>>,
    demanded: AtomicBool,
    inner: Mutex<PInner>,
    cv: Condvar,
}

impl Part {
    fn new(key: Arc<OPath>, start: u64, end: u64, chunks: Vec<(u64, u64)>, state: PState) -> Arc<Self> {
        Arc::new(Part {
            key,
            start,
            end,
            planned: !chunks.is_empty(),
            sticky: false,
            open_ends: Mutex::new({
                let mut v: Vec<u64> = chunks.iter().map(|c| c.1).collect();
                v.dedup();
                v
            }),
            chunks,
            demanded: AtomicBool::new(false),
            inner: Mutex::new(PInner {
                state,
                charged: false,
                cancelled: false,
            }),
            cv: Condvar::new(),
        })
    }
    fn len(&self) -> u64 {
        self.end - self.start
    }
}

#[derive(Default)]
struct Sched {
    queue: VecDeque<Arc<Part>>,
    running: usize,
    held: u64,
}

pub struct Remote {
    /// The root as given, without a trailing slash: `s3://bucket/prefix`.
    pub url: String,
    store: Arc<dyn ObjectStore>,
    prefix: String,
    rt: tokio::runtime::Runtime,
    pub cfg: S3Cfg,
    sched: Mutex<Sched>,
    cache: Option<DiskCache>,
    footers: Mutex<FooterMem>,
    /// Memory bound on the window, bytes (`u64::MAX`: none; `set_window_limit`).
    window_limit: AtomicU64,
    pub stats: Stats,
}

/// Raw footer bytes by object key: (object length, offset of the bytes, bytes, last use).
#[derive(Default)]
struct FooterMem {
    map: HashMap<String, (u64, u64, Bytes, u64)>,
    bytes: u64,
    tick: u64,
}

static REMOTES: RwLock<Vec<Arc<Remote>>> = RwLock::new(Vec::new());

/// Bound the prefetch window of every store to `bytes`, below `inflight_mb` (the reader sets
/// it from its RSS headroom). A range a reader waits on is still fetched, and an empty window
/// always admits one range, so a low bound slows a query but never stalls it.
pub fn set_window_limit(bytes: u64) {
    let g = REMOTES.read().unwrap_or_else(|e| e.into_inner());
    for r in g.iter() {
        r.set_window_limit(bytes);
    }
}

/// Bytes in every store's prefetch window now (queued GETs started, and ranges not read yet).
pub fn window_bytes() -> u64 {
    REMOTES
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|r| r.sched.lock().unwrap_or_else(|e| e.into_inner()).held)
        .sum()
}

/// Bodies at least this long get their own mapping (`MapBuf`); shorter ones use the heap.
const MAP_MIN: usize = 64 << 10;

/// A GET body in its own anonymous mapping, unmapped when the last `Bytes` over it drops.
/// From the heap, glibc would serve these 64 KiB-8 MiB bodies from a tokio thread's arena
/// once its dynamic mmap threshold has risen past them, and keep their pages resident after
/// they are freed (until the reader's malloc_trim after the query): the S3 path's RSS would
/// run ahead of its live bytes inside every query.
struct MapBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: an exclusively owned mapping; written only before it is shared as `Bytes`.
unsafe impl Send for MapBuf {}
unsafe impl Sync for MapBuf {}

impl MapBuf {
    fn new(len: usize) -> std::io::Result<Self> {
        assert!(len > 0);
        // SAFETY: a fresh private anonymous mapping; checked for failure below.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(MapBuf { ptr: p as *mut u8, len })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `len` bytes mapped read-write, owned by `self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl AsRef<[u8]> for MapBuf {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: as above.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MapBuf {
    fn drop(&mut self) {
        // SAFETY: the mapping made in `new`, unmapped once.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

/// A GET's body of exactly `len` bytes: in a `MapBuf` when long enough, else on the heap.
async fn body(g: object_store::GetResult, len: u64) -> anyhow::Result<Bytes> {
    let len = len as usize;
    if len < MAP_MIN {
        return Ok(g.bytes().await?);
    }
    let mut buf = MapBuf::new(len)?;
    let mut s = g.into_stream();
    let mut off = 0usize;
    while let Some(c) = std::future::poll_fn(|cx| s.as_mut().poll_next(cx)).await {
        let c = c?;
        anyhow::ensure!(c.len() <= len - off, "body longer than the {len} bytes asked for");
        buf.as_mut_slice()[off..off + c.len()].copy_from_slice(&c);
        off += c.len();
    }
    anyhow::ensure!(off == len, "short body: {off} of {len} bytes");
    Ok(Bytes::from_owner(buf))
}

/// `len` bytes of a local file, in a `MapBuf` when long enough (the chunk cache's reads).
fn read_file(path: &Path, len: u64) -> std::io::Result<Bytes> {
    let mut f = std::fs::File::open(path)?;
    if f.metadata()?.len() != len {
        return Err(std::io::Error::other("size changed"));
    }
    if (len as usize) < MAP_MIN {
        let mut v = Vec::with_capacity(len as usize);
        f.read_to_end(&mut v)?;
        return Ok(Bytes::from(v));
    }
    let mut buf = MapBuf::new(len as usize)?;
    f.read_exact(buf.as_mut_slice())?;
    Ok(Bytes::from_owner(buf))
}

/// A store root (or a path under one) that names an object store, not a directory.
pub fn is_remote(p: &Path) -> bool {
    p.to_str().is_some_and(|s| s.contains("://"))
}

/// Open the object store behind `url` (once per process) with this configuration.
/// `s3://bucket[/prefix]` uses the AWS default chain read from the environment
/// (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN, web identity, ECS or EC2
/// instance credentials; AWS_REGION; AWS_ENDPOINT and AWS_ALLOW_HTTP for MinIO).
pub fn open(url: &str, cfg: &S3Cfg) -> anyhow::Result<Arc<Remote>> {
    let url = url.trim_end_matches('/');
    if let Some(r) = REMOTES.read().unwrap().iter().find(|r| r.url == url) {
        return Ok(r.clone());
    }
    let (scheme, rest) = url.split_once("://").unwrap_or(("", url));
    let (bucket, _) = rest.split_once('/').unwrap_or((rest, ""));
    let store: Arc<dyn ObjectStore> = match scheme {
        "s3" => Arc::new(
            object_store::aws::AmazonS3Builder::from_env()
                .with_bucket_name(bucket)
                .build()?,
        ),
        _ => anyhow::bail!("unsupported store URL {url}: use s3://bucket/prefix"),
    };
    register(url, store, cfg.clone())
}

/// Register `store` under `url` (`scheme://bucket/prefix`; the bucket part is not part of the
/// key). Tests register an InMemory store this way.
pub fn register(url: &str, store: Arc<dyn ObjectStore>, cfg: S3Cfg) -> anyhow::Result<Arc<Remote>> {
    let url = url.trim_end_matches('/').to_string();
    let rest = url.split_once("://").map(|x| x.1).unwrap_or(&url);
    let prefix = rest
        .split_once('/')
        .map(|x| x.1.trim_matches('/').to_string())
        .unwrap_or_default();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("objstore")
        .enable_all()
        .build()?;
    let cache = match &cfg.cache_dir {
        Some(d) => Some(DiskCache::open(d, (cfg.cache_gb * (1u64 << 30) as f64) as u64)?),
        None => None,
    };
    let r = Arc::new(Remote {
        url: url.clone(),
        store,
        prefix,
        rt,
        cfg,
        sched: Default::default(),
        cache,
        footers: Default::default(),
        window_limit: AtomicU64::new(u64::MAX),
        stats: Default::default(),
    });
    let mut g = REMOTES.write().unwrap();
    g.retain(|x| x.url != url);
    g.push(r.clone());
    Ok(r)
}

/// The registered store and object key behind a path under a remote root.
pub fn lookup(p: &Path) -> Option<(Arc<Remote>, OPath)> {
    let s = p.to_str()?;
    if !s.contains("://") {
        return None;
    }
    let g = REMOTES.read().unwrap();
    let r = g
        .iter()
        .filter(|r| s.starts_with(r.url.as_str()) && s[r.url.len()..].starts_with('/'))
        .max_by_key(|r| r.url.len())?;
    let rel = s[r.url.len()..].trim_start_matches('/');
    let key = if r.prefix.is_empty() {
        rel.to_string()
    } else {
        format!("{}/{rel}", r.prefix)
    };
    Some((r.clone(), OPath::from_iter(key.split('/').filter(|x| !x.is_empty()))))
}

fn remote_of(p: &Path) -> anyhow::Result<(Arc<Remote>, OPath)> {
    lookup(p).ok_or_else(|| anyhow::anyhow!("no object store registered for {}", p.display()))
}

/// Whole file: `std::fs::read` for local paths, one GET for remote ones.
pub fn read(p: &Path) -> anyhow::Result<Vec<u8>> {
    if !is_remote(p) {
        return Ok(std::fs::read(p)?);
    }
    let (r, key) = remote_of(p)?;
    let b = r.run(move |r| async move {
        let t = Instant::now();
        let res = r.store.get_opts(&key, GetOptions::default()).await?.bytes().await?;
        r.count_meta_get(res.len() as u64, t);
        Ok(res)
    })?;
    Ok(b.to_vec())
}

/// Whole file, refused (before its body is read) when larger than `max` bytes.
pub fn read_capped(p: &Path, max: u64) -> anyhow::Result<Vec<u8>> {
    if !is_remote(p) {
        let n = std::fs::metadata(p)?.len();
        anyhow::ensure!(n <= max, "{}: {n} bytes, over the {max}-byte limit", p.display());
        return Ok(std::fs::read(p)?);
    }
    let (r, key) = remote_of(p)?;
    let shown = p.display().to_string();
    let b = r.run(move |r| async move {
        let t = Instant::now();
        let g = r.store.get_opts(&key, GetOptions::default()).await?;
        let n = g.meta.size;
        anyhow::ensure!(n <= max, "{shown}: {n} bytes, over the {max}-byte limit");
        let res = g.bytes().await?;
        r.count_meta_get(res.len() as u64, t);
        anyhow::ensure!(res.len() as u64 <= max, "{shown}: over the {max}-byte limit");
        Ok(res)
    })?;
    Ok(b.to_vec())
}

/// Copy a remote file to the local path `dst` (written to a temporary name, then renamed), in
/// `COPY_PART` range GETs so the body is never held whole. Returns the bytes written.
/// The object must be exactly `expect` bytes (the catalog's size), checked on the first response
/// before anything is written: a catalog cannot make the mirror copy more than it accounted for.
pub fn download(p: &Path, dst: &Path, expect: u64) -> anyhow::Result<u64> {
    anyhow::ensure!(expect > 0, "{}: catalog size 0", p.display());
    use std::io::Write;
    let (r, key) = remote_of(p)?;
    let tmp = dst.with_extension(format!("part{}", std::process::id()));
    let res = (|| -> anyhow::Result<u64> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        let (mut off, mut size) = (0u64, None::<u64>);
        loop {
            let k = key.clone();
            let (sz, b) = r.run(move |r| async move {
                let t = Instant::now();
                let g = r
                    .store
                    .get_opts(
                        &k,
                        GetOptions {
                            range: Some(GetRange::Bounded(off..off + COPY_PART)),
                            ..Default::default()
                        },
                    )
                    .await;
                let g = match g {
                    Ok(g) => g,
                    Err(e) => {
                        r.stats.errors.fetch_add(1, Ordering::Relaxed);
                        return Err(e.into());
                    }
                };
                let sz = g.meta.size;
                if sz != expect {
                    r.stats.errors.fetch_add(1, Ordering::Relaxed);
                    anyhow::bail!("{k}: object is {sz} bytes, the catalog says {expect}");
                }
                let b = g.bytes().await?;
                r.count_meta_get(b.len() as u64, t);
                Ok((sz, b))
            })?;
            let size = *size.get_or_insert(sz);
            out.write_all(&b)?;
            off += b.len() as u64;
            if off >= size || b.is_empty() {
                anyhow::ensure!(off == size, "{}: short read ({off} of {size} bytes)", p.display());
                break;
            }
        }
        out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        Ok(off)
    })();
    match res {
        Ok(n) => {
            std::fs::rename(&tmp, dst)?;
            Ok(n)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Whether the file exists (local) or the object can be read (remote).
pub fn exists(p: &Path) -> bool {
    if !is_remote(p) {
        return p.exists();
    }
    let Ok((r, key)) = remote_of(p) else {
        return false;
    };
    r.run(move |r| async move { Ok(r.store.head(&key).await?) }).is_ok()
}

impl Remote {
    /// Run `f` on this store's runtime and wait for it from a plain thread (never `block_on`:
    /// callers may sit on another runtime's blocking pool).
    fn run<T, F, Fut>(self: &Arc<Self>, f: F) -> anyhow::Result<T>
    where
        T: Send + 'static,
        F: FnOnce(Arc<Self>) -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<T>> + Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let fut = f(self.clone());
        self.rt.spawn(async move {
            let _ = tx.send(fut.await);
        });
        rx.recv().map_err(|_| anyhow::anyhow!("object store task dropped"))?
    }

    /// A whole-object GET outside the query path (catalog, mirror): not a demand GET.
    fn count_meta_get(&self, bytes: u64, t: Instant) {
        self.count_get(bytes, t, false);
        self.stats.meta_gets.fetch_add(1, Ordering::Relaxed);
    }

    fn count_get(&self, bytes: u64, t: Instant, demand: bool) {
        self.stats.gets.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes.fetch_add(bytes, Ordering::Relaxed);
        self.stats
            .get_ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if demand {
            self.stats.demand_gets.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Suffix GET of `n` bytes: (object length, offset of the returned bytes, bytes).
    pub fn get_suffix(self: &Arc<Self>, key: &OPath, n: u64) -> anyhow::Result<(u64, u64, Bytes)> {
        let key = key.clone();
        self.run(move |r| async move {
            let t = Instant::now();
            let g = r
                .store
                .get_opts(
                    &key,
                    GetOptions {
                        range: Some(GetRange::Suffix(n)),
                        ..Default::default()
                    },
                )
                .await?;
            let (size, start) = (g.meta.size, g.range.start);
            let b = g.bytes().await?;
            r.count_get(b.len() as u64, t, true);
            Ok((size, start, b))
        })
    }

    /// Bound the window to `bytes` (see the free function); a raised bound starts queued GETs.
    pub fn set_window_limit(self: &Arc<Self>, bytes: u64) {
        if bytes > self.window_limit.swap(bytes, Ordering::AcqRel) {
            self.pump();
        }
    }

    /// Start whatever the window and the request limit allow.
    fn pump(self: &Arc<Self>) {
        let full = self.cfg.inflight_mb.max(1) << 20;
        let budget = full.min(self.window_limit.load(Ordering::Acquire));
        let mut issue = Vec::new();
        {
            let mut s = self.sched.lock().unwrap_or_else(|e| e.into_inner());
            while s.running < self.cfg.concurrency.max(1) {
                let Some(p) = s.queue.front() else { break };
                let demanded = p.demanded.load(Ordering::Acquire);
                if !demanded && s.held > 0 && s.held + p.len() > budget {
                    if s.held + p.len() <= full {
                        self.stats.window_limited.fetch_add(1, Ordering::Relaxed);
                    }
                    break;
                }
                let p = s.queue.pop_front().unwrap();
                {
                    let mut i = p.inner.lock().unwrap_or_else(|e| e.into_inner());
                    if i.cancelled || !matches!(i.state, PState::Queued) {
                        continue;
                    }
                    i.state = PState::Running;
                    i.charged = true;
                }
                s.running += 1;
                s.held += p.len();
                issue.push(p);
            }
            self.stats.inflight_peak.fetch_max(s.running as u64, Ordering::Relaxed);
            self.stats.held_peak_bytes.fetch_max(s.held, Ordering::Relaxed);
        }
        for p in issue {
            let r = self.clone();
            self.rt.spawn(async move { r.fetch(p).await });
        }
    }

    async fn fetch(self: Arc<Self>, p: Arc<Part>) {
        let t = Instant::now();
        let res: anyhow::Result<Bytes> = async {
            let g = self
                .store
                .get_opts(
                    &p.key,
                    GetOptions {
                        range: Some(GetRange::Bounded(p.start..p.end)),
                        ..Default::default()
                    },
                )
                .await?;
            let b = body(g, p.len()).await?;
            anyhow::ensure!(b.len() as u64 == p.len(), "short read of {}", p.key);
            Ok(b)
        }
        .await;
        self.count_get(p.len(), t, !p.planned);
        if let (Ok(b), Some(_)) = (&res, &self.cache) {
            let (r, p, b) = (self.clone(), p.clone(), b.clone());
            self.rt.spawn_blocking(move || {
                let c = r.cache.as_ref().unwrap();
                for &(s, e) in &p.chunks {
                    c.put(&p.key, s, &b[(s - p.start) as usize..(e - p.start) as usize]);
                }
            });
        }
        let release = {
            let mut i = p.inner.lock().unwrap_or_else(|e| e.into_inner());
            match res {
                Ok(b) => i.state = PState::Ready(b),
                Err(e) => {
                    self.stats.errors.fetch_add(1, Ordering::Relaxed);
                    i.state = PState::Failed(format!("GET {} [{}, {}): {e}", p.key, p.start, p.end));
                }
            }
            let rel = i.charged && (i.cancelled || matches!(i.state, PState::Failed(_)));
            if rel {
                i.charged = false;
                if i.cancelled {
                    i.state = PState::Done;
                }
            }
            rel
        };
        p.cv.notify_all();
        {
            let mut s = self.sched.lock().unwrap_or_else(|e| e.into_inner());
            s.running -= 1;
            if release {
                s.held -= p.len();
            }
        }
        self.pump();
    }

    /// Move a queued part to the head of the queue, past the window (a reader waits on it).
    fn promote(self: &Arc<Self>, p: &Arc<Part>) {
        if !p.demanded.swap(true, Ordering::AcqRel) {
            let mut s = self.sched.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(i) = s.queue.iter().position(|q| Arc::ptr_eq(q, p)) {
                let q = s.queue.remove(i).unwrap();
                s.queue.push_front(q);
            } else if matches!(p.inner.lock().unwrap_or_else(|e| e.into_inner()).state, PState::Queued) {
                // planned but not submitted yet (`submit` skips it once it has run)
                s.queue.push_front(p.clone());
            }
        }
        self.pump();
    }

    /// Release a part: read to the end (`done`), or its file was dropped.
    fn release(self: &Arc<Self>, p: &Arc<Part>) {
        let free = {
            let mut i = p.inner.lock().unwrap_or_else(|e| e.into_inner());
            match i.state {
                PState::Queued | PState::Cached => i.state = PState::Done,
                PState::Running => {
                    i.cancelled = true;
                    return;
                }
                _ => i.state = PState::Done,
            }
            i.cancelled = true;
            std::mem::replace(&mut i.charged, false)
        };
        if free {
            self.sched.lock().unwrap_or_else(|e| e.into_inner()).held -= p.len();
            self.pump();
        }
    }

    /// The part's bytes, waiting for its GET (or reading it from the local cache).
    fn take(self: &Arc<Self>, p: &Arc<Part>) -> anyhow::Result<Bytes> {
        let mut t0: Option<Instant> = None;
        let mut i = p.inner.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match &i.state {
                PState::Ready(b) => {
                    let b = b.clone();
                    if let Some(t) = t0 {
                        self.stats
                            .wait_ns
                            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    return Ok(b);
                }
                PState::Failed(e) => anyhow::bail!("{e}"),
                PState::Done => anyhow::bail!("range [{}, {}) of {} read after release", p.start, p.end, p.key),
                PState::Cached => {
                    let got = self.cache.as_ref().and_then(|c| c.get(&p.key, p.start, p.len()));
                    match got {
                        Some(b) => {
                            self.stats.cache_hit_bytes.fetch_add(b.len() as u64, Ordering::Relaxed);
                            i.state = PState::Ready(b);
                        }
                        None => {
                            // evicted since planning: fetch it now
                            i.state = PState::Queued;
                            drop(i);
                            p.demanded.store(true, Ordering::Release);
                            self.sched
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .queue
                                .push_front(p.clone());
                            self.pump();
                            i = p.inner.lock().unwrap_or_else(|e| e.into_inner());
                        }
                    }
                }
                PState::Queued => {
                    t0.get_or_insert_with(Instant::now);
                    drop(i);
                    self.promote(p);
                    i = p.inner.lock().unwrap_or_else(|e| e.into_inner());
                    if matches!(i.state, PState::Queued) {
                        i = p.cv.wait(i).unwrap();
                    }
                }
                PState::Running => {
                    t0.get_or_insert_with(Instant::now);
                    i = p.cv.wait(i).unwrap();
                }
            }
        }
    }

    pub fn stats_json(&self) -> serde_json::Value {
        let s = &self.stats;
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let (hits, misses) = (l(&s.cache_hits), l(&s.cache_misses));
        let (held, queued) = {
            let g = self.sched.lock().unwrap_or_else(|e| e.into_inner());
            (g.held, g.queue.len())
        };
        serde_json::json!({
            "url": self.url,
            "gets": l(&s.gets),
            "bytes": l(&s.bytes),
            "demand_gets": l(&s.demand_gets),
            "meta_gets": l(&s.meta_gets),
            "window_limit_bytes": match self.window_limit.load(Ordering::Relaxed) {
                u64::MAX => serde_json::Value::Null,
                b => b.into(),
            },
            "window_limited": l(&s.window_limited),
            "errors": l(&s.errors),
            "inflight_peak": l(&s.inflight_peak),
            "window_peak_bytes": l(&s.held_peak_bytes),
            "window_bytes": held,
            "queued": queued,
            "wait_ms": l(&s.wait_ns) / 1_000_000,
            "get_ms_total": l(&s.get_ns) / 1_000_000,
            "footers": {
                "mem_hits": l(&s.footer_mem_hits), "disk_hits": l(&s.footer_disk_hits),
                "fetches": l(&s.footer_fetches), "gets": l(&s.footer_gets),
                "mem_bytes": self.footers.lock().unwrap_or_else(|e| e.into_inner()).bytes,
            },
            "cache": self.cache.as_ref().map(|c| {
                let (n, b) = c.size();
                serde_json::json!({
                    "chunks": n, "bytes": b, "hits": hits, "misses": misses,
                    "hit_rate": if hits + misses > 0 { hits as f64 / (hits + misses) as f64 } else { 0.0 },
                    "hit_bytes": l(&s.cache_hit_bytes),
                })
            }),
        })
    }
}

/// GETs, bytes, demand GETs and footer GETs over every registered store, since this process
/// started (`None` with no store registered). The SQL child sends them with each reply, so the
/// parent can keep totals across child restarts.
pub fn totals() -> Option<[u64; 4]> {
    let g = REMOTES.read().unwrap();
    if g.is_empty() {
        return None;
    }
    let mut t = [0u64; 4];
    for r in g.iter() {
        let s = &r.stats;
        for (i, a) in [&s.gets, &s.bytes, &s.demand_gets, &s.footer_gets].into_iter().enumerate() {
            t[i] += a.load(Ordering::Relaxed);
        }
    }
    Some(t)
}

/// The disk cache of a store served without --s3-cache-dir: under the system temp dir, one
/// directory per store URL, the same across restarts (the scratch dir is per process).
pub fn default_cache_dir(url: &str) -> PathBuf {
    std::env::temp_dir()
        .join("pgai-mon-s3cache")
        .join(format!("{:016x}", xxhash_rust::xxh3::xxh3_64(url.trim_end_matches('/').as_bytes())))
}

/// Create `default_cache_dir(url)` and its parent owner-only (0700; the system temp dir is
/// shared) and return it.
pub fn make_default_cache_dir(url: &str) -> anyhow::Result<PathBuf> {
    let d = default_cache_dir(url);
    for p in [d.parent().unwrap(), d.as_path()] {
        let mut b = std::fs::DirBuilder::new();
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut b, 0o700);
        match b.create(p) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(anyhow::anyhow!("{}: {e}", p.display())),
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            let m = std::fs::symlink_metadata(p)?;
            anyhow::ensure!(m.is_dir(), "{}: not a directory", p.display());
            anyhow::ensure!(m.uid() == unsafe { libc::geteuid() }, "{}: owned by another user", p.display());
            std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o700))?;
        }
    }
    Ok(d)
}

/// Raw footer bytes held in memory over every registered store.
pub fn footer_mem_bytes() -> u64 {
    REMOTES
        .read()
        .unwrap()
        .iter()
        .map(|r| r.footers.lock().unwrap_or_else(|e| e.into_inner()).bytes)
        .sum()
}

/// The GETs-per-query warning threshold (the first store's; 0 with no store).
pub fn gets_warn() -> u64 {
    REMOTES.read().unwrap().first().map_or(0, |r| r.cfg.gets_warn)
}

/// Counters of every registered store (for /health).
pub fn stats_json() -> serde_json::Value {
    let g = REMOTES.read().unwrap();
    match g.len() {
        0 => serde_json::Value::Null,
        1 => g[0].stats_json(),
        _ => serde_json::Value::Array(g.iter().map(|r| r.stats_json()).collect()),
    }
}

/// Coalesce sorted, non-overlapping ranges: merge across gaps of at most `gap`, keep each GET
/// under `max` (a single range longer than `max` is split). Returns (start, end, ranges inside).
pub fn coalesce(ranges: &[(u64, u64)], gap: u64, max: u64) -> Vec<(u64, u64, Vec<(u64, u64)>)> {
    let mut out: Vec<(u64, u64, Vec<(u64, u64)>)> = Vec::new();
    for &(s, e) in ranges {
        if let Some(last) = out.last_mut() {
            if s <= last.1 + gap && e - last.0 <= max {
                last.1 = last.1.max(e);
                last.2.push((s, e));
                continue;
            }
        }
        if e - s <= max {
            out.push((s, e, vec![(s, e)]));
        } else {
            // too big for one GET: pieces, none of them cacheable as a chunk
            let mut a = s;
            while a < e {
                let b = (a + max).min(e);
                out.push((a, b, vec![]));
                a = b;
            }
        }
    }
    out
}

/// GETs planned for one file, not queued yet (`submit`).
#[derive(Default)]
pub struct Planned {
    remote: Option<Arc<Remote>>,
    gets: Vec<Arc<Part>>,
}

/// Queue the planned GETs of several files, interleaved: every file's first range, then every
/// file's second, ... A scan merges its files by series_id, so it needs the head of each file
/// first; file after file would hold the merge until the last file's first range arrived.
pub fn submit(plans: Vec<Planned>) {
    let mut by: Vec<(Arc<Remote>, Vec<std::vec::IntoIter<Arc<Part>>>)> = Vec::new();
    for p in plans {
        let Some(r) = p.remote else { continue };
        match by.iter_mut().find(|x| Arc::ptr_eq(&x.0, &r)) {
            Some(x) => x.1.push(p.gets.into_iter()),
            None => by.push((r, vec![p.gets.into_iter()])),
        }
    }
    for (r, mut its) in by {
        {
            let mut s = r.sched.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                let mut any = false;
                for it in its.iter_mut() {
                    if let Some(p) = it.next() {
                        s.queue.push_back(p);
                        any = true;
                    }
                }
                if !any {
                    break;
                }
            }
        }
        r.pump();
    }
}

/// A remote Parquet file: its length and the planned (or fetched-on-demand) ranges.
pub struct RFile {
    remote: Arc<Remote>,
    key: Arc<OPath>,
    len: u64,
    /// Sorted by start, non-overlapping.
    parts: Mutex<Vec<Arc<Part>>>,
}

impl RFile {
    pub fn new(remote: Arc<Remote>, key: OPath, len: u64) -> Arc<Self> {
        Arc::new(RFile {
            remote,
            key: Arc::new(key),
            len,
            parts: Default::default(),
        })
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    /// Bytes already in hand (a footer's suffix GET).
    pub fn insert_ready(&self, start: u64, b: Bytes) {
        let mut p = Part::new(
            self.key.clone(),
            start,
            start + b.len() as u64,
            vec![],
            PState::Ready(b),
        );
        Arc::get_mut(&mut p).unwrap().sticky = true;
        self.insert(vec![p]);
    }

    fn insert(&self, new: Vec<Arc<Part>>) {
        let mut g = self.parts.lock().unwrap_or_else(|e| e.into_inner());
        g.extend(new);
        g.sort_by_key(|p| p.start);
    }

    /// Drop every part (the footer's bytes after the metadata is parsed).
    pub fn clear(&self) {
        let old = std::mem::take(&mut *self.parts.lock().unwrap_or_else(|e| e.into_inner()));
        for p in &old {
            self.remote.release(p);
        }
    }

    /// Plan these ranges (column chunks) and queue their GETs now.
    pub fn plan(&self, ranges: Vec<(u64, u64)>) {
        submit(vec![self.plan_parts(ranges)]);
    }

    /// Plan these ranges: skip what is planned already, take what the local cache holds,
    /// coalesce the rest into GETs. They are queued by `submit`.
    fn plan_parts(&self, mut ranges: Vec<(u64, u64)>) -> Planned {
        ranges.retain(|r| r.1 > r.0);
        ranges.sort();
        ranges.dedup();
        // runs of uncached ranges: a GET never spans a cached chunk (it would fetch bytes the
        // disk holds, and overlap that chunk's part)
        let mut fresh: Vec<Vec<(u64, u64)>> = vec![Vec::new()];
        let mut cached = Vec::new();
        {
            let g = self.parts.lock().unwrap_or_else(|e| e.into_inner());
            for (s, e) in ranges {
                let covered = g.iter().any(|p| p.start < e && s < p.end);
                if covered {
                    continue;
                }
                match &self.remote.cache {
                    Some(c) if c.has(&self.key, s, e - s) => {
                        self.remote.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                        cached.push(Part::new(self.key.clone(), s, e, vec![], PState::Cached));
                        if !fresh.last().unwrap().is_empty() {
                            fresh.push(Vec::new());
                        }
                    }
                    c => {
                        if c.is_some() {
                            self.remote.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
                        }
                        fresh.last_mut().unwrap().push((s, e));
                    }
                }
            }
        }
        let cfg = &self.remote.cfg;
        let gets: Vec<Arc<Part>> = fresh
            .iter()
            .flat_map(|run| coalesce(run, cfg.coalesce_kb << 10, MAX_PART))
            .map(|(s, e, ch)| Part::new(self.key.clone(), s, e, ch, PState::Queued))
            .collect();
        cached.extend(gets.iter().cloned());
        if !cached.is_empty() {
            self.insert(cached);
        }
        Planned {
            remote: Some(self.remote.clone()),
            gets,
        }
    }

    /// The part holding `pos`, or the start of the next part.
    fn find(&self, pos: u64) -> (Option<Arc<Part>>, u64) {
        let g = self.parts.lock().unwrap_or_else(|e| e.into_inner());
        let i = g.partition_point(|p| p.start <= pos);
        // the latest-starting live part over `pos` (parts may nest: a demand read inside a
        // released range, or a GET planned over an older one)
        for p in g[..i].iter().rev().take(64) {
            if p.end > pos && !matches!(p.inner.lock().unwrap_or_else(|e| e.into_inner()).state, PState::Done) {
                return (Some(p.clone()), 0);
            }
        }
        // a released part over `pos` is gone for good: refetch past it
        (None, g[i..].iter().map(|p| p.start).next().unwrap_or(self.len))
    }

    /// The part holding `pos`: planned, or fetched now (at least `want` bytes, if possible).
    fn part_at(&self, pos: u64, want: u64) -> anyhow::Result<Arc<Part>> {
        anyhow::ensure!(
            pos < self.len,
            "read at {pos} past the end of {} ({})",
            self.key,
            self.len
        );
        let (p, next) = self.find(pos);
        if let Some(p) = p {
            return Ok(p);
        }
        let end = (pos + want.max(DEMAND_MIN)).min(self.len).min(next.max(pos + 1));
        let p = Part::new(self.key.clone(), pos, end, vec![], PState::Queued);
        p.demanded.store(true, Ordering::Release);
        {
            let mut g = self.parts.lock().unwrap_or_else(|e| e.into_inner());
            // drop a released part over this range (it is being replaced)
            g.retain(|q| !(q.start < end && pos < q.end));
            let i = g.partition_point(|q| q.start <= pos);
            g.insert(i, p.clone());
        }
        self.remote
            .sched
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .queue
            .push_front(p.clone());
        self.remote.pump();
        Ok(p)
    }

    /// A read of `p` ended at `pos`: release the part once all of it that will be read is
    /// (every chunk read to its end; a part without chunks, read to its own end).
    fn read_to(&self, p: &Arc<Part>, pos: u64) {
        if p.sticky {
            return;
        }
        let done = {
            let mut e = p.open_ends.lock().unwrap_or_else(|e| e.into_inner());
            if e.is_empty() {
                pos == p.end
            } else {
                e.retain(|&x| x != pos);
                e.is_empty()
            }
        };
        if done {
            self.remote.release(p);
        }
    }

    pub fn get_bytes(&self, start: u64, len: usize) -> anyhow::Result<Bytes> {
        let end = start + len as u64;
        anyhow::ensure!(
            end <= self.len,
            "read [{start}, {end}) past the end of {} ({})",
            self.key,
            self.len
        );
        if len == 0 {
            return Ok(Bytes::new());
        }
        let p = self.part_at(start, end - start)?;
        let b = self.remote.take(&p)?;
        if end <= p.end {
            let out = b.slice((start - p.start) as usize..(end - p.start) as usize);
            self.read_to(&p, end);
            return Ok(out);
        }
        let mut out = Vec::with_capacity(len);
        let mut pos = start;
        let mut cur = Some((p, b));
        while pos < end {
            let (p, b) = match cur.take() {
                Some(x) => x,
                None => {
                    let p = self.part_at(pos, end - pos)?;
                    let b = self.remote.take(&p)?;
                    (p, b)
                }
            };
            let n = end.min(p.end) - pos;
            out.extend_from_slice(&b[(pos - p.start) as usize..(pos - p.start + n) as usize]);
            pos += n;
            self.read_to(&p, pos);
        }
        Ok(Bytes::from(out))
    }
}

impl Drop for RFile {
    fn drop(&mut self) {
        for p in self.parts.get_mut().unwrap().drain(..) {
            self.remote.release(&p);
        }
    }
}

/// `Read` from a remote file at a position (page headers).
pub struct RRead {
    f: Arc<RFile>,
    pos: u64,
}

impl Read for RRead {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.f.len || buf.is_empty() {
            return Ok(0);
        }
        let p = self
            .f
            .part_at(self.pos, buf.len() as u64)
            .map_err(std::io::Error::other)?;
        let b = self.f.remote.take(&p).map_err(std::io::Error::other)?;
        let n = ((p.end - self.pos) as usize).min(buf.len());
        let o = (self.pos - p.start) as usize;
        buf[..n].copy_from_slice(&b[o..o + n]);
        self.pos += n as u64;
        self.f.read_to(&p, self.pos);
        Ok(n)
    }
}

/// What the Parquet readers read: a local file, or a remote one. Cloning shares the handle.
#[derive(Clone)]
pub enum Src {
    Local(Arc<std::fs::File>),
    Remote(Arc<RFile>),
}

impl Src {
    /// Plan the column chunks of `row_groups` x the leaves `mask` keeps and queue their GETs
    /// (no-op for local files).
    pub fn prefetch(
        &self,
        md: &parquet::file::metadata::ParquetMetaData,
        row_groups: &[usize],
        mask: Option<&parquet::arrow::ProjectionMask>,
    ) {
        submit(vec![self.plan(md, row_groups, mask)]);
    }

    /// Like `prefetch`, queued later by `submit` with other files' plans.
    pub fn plan(
        &self,
        md: &parquet::file::metadata::ParquetMetaData,
        row_groups: &[usize],
        mask: Option<&parquet::arrow::ProjectionMask>,
    ) -> Planned {
        let Src::Remote(f) = self else {
            return Planned::default();
        };
        let mut ranges = Vec::new();
        for &i in row_groups {
            let Some(rg) = md.row_groups().get(i) else { continue };
            for (j, c) in rg.columns().iter().enumerate() {
                if mask.is_none_or(|m| m.leaf_included(j)) {
                    let (s, l) = c.byte_range();
                    ranges.push((s, s + l));
                }
            }
        }
        f.plan_parts(ranges)
    }
}

impl Src {
    /// Plan the Bloom filters of `leaf` in these row groups (those with a recorded length).
    pub fn prefetch_blooms(
        &self,
        md: &parquet::file::metadata::ParquetMetaData,
        row_groups: impl Iterator<Item = usize>,
        leaf: usize,
    ) {
        let Src::Remote(f) = self else { return };
        let ranges = row_groups
            .filter_map(|i| {
                let c = md.row_groups().get(i)?.columns().get(leaf)?;
                let (o, l) = (c.bloom_filter_offset()?, c.bloom_filter_length()?);
                Some((o as u64, o as u64 + l as u64))
            })
            .collect();
        f.plan(ranges);
    }
}

impl parquet::file::reader::Length for Src {
    fn len(&self) -> u64 {
        match self {
            Src::Local(f) => f.as_ref().len(),
            Src::Remote(f) => f.len,
        }
    }
}

pub enum SrcRead {
    Local(std::io::BufReader<std::fs::File>),
    Remote(RRead),
}

impl Read for SrcRead {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            SrcRead::Local(r) => r.read(buf),
            SrcRead::Remote(r) => r.read(buf),
        }
    }
}

impl parquet::file::reader::ChunkReader for Src {
    type T = SrcRead;

    fn get_read(&self, start: u64) -> parquet::errors::Result<SrcRead> {
        match self {
            Src::Local(f) => Ok(SrcRead::Local(f.as_ref().get_read(start)?)),
            Src::Remote(f) => Ok(SrcRead::Remote(RRead {
                f: f.clone(),
                pos: start,
            })),
        }
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        match self {
            Src::Local(f) => f.as_ref().get_bytes(start, length),
            Src::Remote(f) => f
                .get_bytes(start, length)
                .map_err(|e| parquet::errors::ParquetError::External(e.into())),
        }
    }
}

/// Open a file for reading. Remote: a suffix GET that brings the object length and (usually)
/// the whole footer; the caller drops it with `RFile::clear` once the metadata is parsed.
pub fn open_src(p: &Path) -> anyhow::Result<Src> {
    if !is_remote(p) {
        return Ok(Src::Local(Arc::new(std::fs::File::open(p)?)));
    }
    open_many(&[p.to_path_buf()]).pop().unwrap()
}

/// Open a remote file whose length is known (a footer-cache hit): no request.
pub fn open_known(p: &Path, len: u64) -> anyhow::Result<Src> {
    let (r, key) = remote_of(p)?;
    Ok(Src::Remote(RFile::new(r, key, len)))
}

/// Open these files with their footers in memory: remote footers come from the raw-footer cache
/// (memory, then disk) or are fetched concurrently, each whole (a suffix GET, and when the
/// footer is longer than the suffix a second GET for the rest, in the same task). Returns the
/// opened files in order; local paths are opened directly.
pub fn open_many(paths: &[PathBuf]) -> Vec<anyhow::Result<Src>> {
    let mut out: Vec<Option<anyhow::Result<Src>>> = (0..paths.len()).map(|_| None).collect();
    let mut by_store: HashMap<String, Vec<(usize, Arc<Remote>, OPath)>> = HashMap::new();
    for (i, p) in paths.iter().enumerate() {
        if !is_remote(p) {
            continue;
        }
        match remote_of(p) {
            Ok((r, k)) => {
                if let Some((len, start, b)) = r.footer_cached(&k) {
                    let f = RFile::new(r.clone(), k, len);
                    f.insert_ready(start, b);
                    out[i] = Some(Ok(Src::Remote(f)));
                } else {
                    by_store.entry(r.url.clone()).or_default().push((i, r, k));
                }
            }
            Err(e) => out[i] = Some(Err(e)),
        }
    }
    for (_, items) in by_store {
        let r = items[0].1.clone();
        let keys: Vec<(usize, OPath)> = items.into_iter().map(|(i, _, k)| (i, k)).collect();
        let res = r.run(move |r| async move {
            let sem = Arc::new(tokio::sync::Semaphore::new(r.cfg.concurrency.max(1)));
            let mut hs = Vec::new();
            for (i, k) in keys {
                let (r, sem) = (r.clone(), sem.clone());
                hs.push(tokio::spawn(async move {
                    let _g = sem.acquire().await;
                    let res = r.fetch_footer(&k).await;
                    (i, k, res)
                }));
            }
            let mut v = Vec::new();
            for h in hs {
                v.push(h.await?);
            }
            Ok(v)
        });
        match res {
            Ok(v) => {
                for (i, k, res) in v {
                    out[i] = Some(res.map(|(len, start, b)| {
                        r.footer_put(&k, len, start, b.clone());
                        let f = RFile::new(r.clone(), k, len);
                        f.insert_ready(start, b);
                        Src::Remote(f)
                    }));
                }
            }
            Err(e) => {
                let msg = e.to_string();
                for o in out.iter_mut().filter(|o| o.is_none()) {
                    *o = Some(Err(anyhow::anyhow!("{msg}")));
                }
            }
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(i, o)| o.unwrap_or_else(|| Ok(Src::Local(Arc::new(std::fs::File::open(&paths[i])?)))))
        .collect()
}

impl Remote {
    /// GET `range` of `key` (a footer read): counted as an unplanned GET.
    async fn get_range(self: &Arc<Self>, key: &OPath, range: GetRange) -> anyhow::Result<(u64, u64, Bytes)> {
        let t = Instant::now();
        let g = match self
            .store
            .get_opts(
                key,
                GetOptions {
                    range: Some(range),
                    ..Default::default()
                },
            )
            .await
        {
            Ok(g) => g,
            Err(e) => {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
                return Err(e.into());
            }
        };
        let (size, start) = (g.meta.size, g.range.start);
        let b = g.bytes().await?;
        self.count_get(b.len() as u64, t, false);
        self.stats.footer_gets.fetch_add(1, Ordering::Relaxed);
        Ok((size, start, b))
    }

    /// The whole footer of `key` (the Parquet metadata, its length and the magic): a suffix GET,
    /// then, when the footer is longer, one GET for the rest. Returns (object length, offset of
    /// the bytes, bytes), cut to the footer.
    async fn fetch_footer(self: &Arc<Self>, key: &OPath) -> anyhow::Result<(u64, u64, Bytes)> {
        self.stats.footer_fetches.fetch_add(1, Ordering::Relaxed);
        let (size, start, b) = self.get_range(key, GetRange::Suffix(FOOTER_SUFFIX)).await?;
        let n = b.len();
        if n < 8 || &b[n - 4..] != b"PAR1" {
            // not Parquet: the suffix is all there is to keep (the parser reports the error)
            return Ok((size, start, b));
        }
        let meta = u32::from_le_bytes(b[n - 8..n - 4].try_into().unwrap()) as u64;
        let need = meta + 8;
        anyhow::ensure!(need <= size, "{key}: footer length {meta} beyond the object ({size} bytes)");
        let fstart = size - need;
        if fstart >= start {
            // a copy: a slice would pin the whole suffix (up to FOOTER_SUFFIX) in the footer
            // cache, which counts only the footer's bytes
            return Ok((size, fstart, Bytes::copy_from_slice(&b[(fstart - start) as usize..])));
        }
        let (_, s2, head) = self.get_range(key, GetRange::Bounded(fstart..start)).await?;
        anyhow::ensure!(s2 == fstart && head.len() as u64 == start - fstart, "{key}: short footer read");
        let mut v = Vec::with_capacity(need as usize);
        v.extend_from_slice(&head);
        v.extend_from_slice(&b);
        Ok((size, fstart, Bytes::from(v)))
    }

    fn footer_disk_name(key: &OPath) -> String {
        format!("{:016x}-footer", xxhash_rust::xxh3::xxh3_64(key.as_ref().as_bytes()))
    }

    /// A cached footer (memory, then disk): (object length, offset, bytes).
    fn footer_cached(&self, key: &OPath) -> Option<(u64, u64, Bytes)> {
        {
            let mut g = self.footers.lock().unwrap_or_else(|e| e.into_inner());
            g.tick += 1;
            let t = g.tick;
            if let Some(e) = g.map.get_mut(key.as_ref()) {
                e.3 = t;
                self.stats.footer_mem_hits.fetch_add(1, Ordering::Relaxed);
                return Some((e.0, e.1, e.2.clone()));
            }
        }
        let c = self.cache.as_ref()?;
        let raw = c.get_named(&Self::footer_disk_name(key))?;
        if raw.len() < 16 {
            return None;
        }
        let len = u64::from_le_bytes(raw[0..8].try_into().unwrap());
        let start = u64::from_le_bytes(raw[8..16].try_into().unwrap());
        let b = Bytes::from(raw).slice(16..);
        if start + b.len() as u64 != len || b.len() < 8 || &b[b.len() - 4..] != b"PAR1" {
            return None;
        }
        self.stats.footer_disk_hits.fetch_add(1, Ordering::Relaxed);
        self.footer_mem_put(key, len, start, b.clone());
        Some((len, start, b))
    }

    fn footer_mem_put(&self, key: &OPath, len: u64, start: u64, b: Bytes) {
        let cap = self.cfg.footer_mb << 20;
        if b.len() as u64 > cap / 4 {
            return;
        }
        let mut g = self.footers.lock().unwrap_or_else(|e| e.into_inner());
        g.tick += 1;
        let t = g.tick;
        let n = b.len() as u64;
        if let Some(old) = g.map.insert(key.as_ref().to_string(), (len, start, b, t)) {
            g.bytes -= old.2.len() as u64;
        }
        g.bytes += n;
        while g.bytes > cap && !g.map.is_empty() {
            let k = g.map.iter().min_by_key(|(_, v)| v.3).map(|(k, _)| k.clone()).unwrap();
            let (_, _, b, _) = g.map.remove(&k).unwrap();
            g.bytes -= b.len() as u64;
        }
    }

    /// Keep a fetched footer: in memory, and on disk with `cache_dir`.
    fn footer_put(&self, key: &OPath, len: u64, start: u64, b: Bytes) {
        if let Some(c) = &self.cache {
            let mut v = Vec::with_capacity(16 + b.len());
            v.extend_from_slice(&len.to_le_bytes());
            v.extend_from_slice(&start.to_le_bytes());
            v.extend_from_slice(&b);
            c.put_named(&Self::footer_disk_name(key), &v);
        }
        self.footer_mem_put(key, len, start, b);
    }
}

/// Local LRU cache of fetched column chunks: one file per (object, start, length).
struct DiskCache {
    dir: PathBuf,
    cap: u64,
    idx: Mutex<CacheIdx>,
}

#[derive(Default)]
struct CacheIdx {
    map: HashMap<String, (u64, u64)>, // name -> (bytes, last use)
    total: u64,
    tick: u64,
}

impl DiskCache {
    fn open(dir: &Path, cap: u64) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".tmp") {
                let _ = std::fs::remove_file(e.path());
                continue;
            }
            let m = e.metadata()?;
            if !m.is_file() {
                continue; // mirror/ (query.rs): DuckDB's local copies, not cached chunks
            }
            files.push((m.modified().unwrap_or(std::time::UNIX_EPOCH), name, m.len()));
        }
        files.sort();
        let mut idx = CacheIdx::default();
        for (_, name, len) in files {
            idx.tick += 1;
            idx.total += len;
            idx.map.insert(name, (len, idx.tick));
        }
        let c = DiskCache {
            dir: dir.to_path_buf(),
            cap,
            idx: Mutex::new(idx),
        };
        c.evict(&mut c.idx.lock().unwrap_or_else(|e| e.into_inner()));
        Ok(c)
    }

    fn name(key: &OPath, start: u64, len: u64) -> String {
        format!(
            "{:016x}-{start}-{len}",
            xxhash_rust::xxh3::xxh3_64(key.as_ref().as_bytes())
        )
    }

    fn has(&self, key: &OPath, start: u64, len: u64) -> bool {
        self.idx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .map
            .contains_key(&Self::name(key, start, len))
    }

    fn size(&self) -> (usize, u64) {
        let g = self.idx.lock().unwrap_or_else(|e| e.into_inner());
        (g.map.len(), g.total)
    }

    fn get(&self, key: &OPath, start: u64, len: u64) -> Option<Bytes> {
        let name = Self::name(key, start, len);
        {
            let mut g = self.idx.lock().unwrap_or_else(|e| e.into_inner());
            g.tick += 1;
            let t = g.tick;
            g.map.get_mut(&name)?.1 = t;
        }
        match read_file(&self.dir.join(&name), len) {
            Ok(b) => Some(b),
            _ => {
                let mut g = self.idx.lock().unwrap_or_else(|e| e.into_inner());
                if let Some((n, _)) = g.map.remove(&name) {
                    g.total -= n;
                }
                None
            }
        }
    }

    /// A whole cache file by name (footers: `<xxh3(key)>-footer`).
    fn get_named(&self, name: &str) -> Option<Vec<u8>> {
        {
            let mut g = self.idx.lock().unwrap_or_else(|e| e.into_inner());
            g.tick += 1;
            let t = g.tick;
            g.map.get_mut(name)?.1 = t;
        }
        match std::fs::read(self.dir.join(name)) {
            Ok(b) => Some(b),
            Err(_) => {
                let mut g = self.idx.lock().unwrap_or_else(|e| e.into_inner());
                if let Some((n, _)) = g.map.remove(name) {
                    g.total -= n;
                }
                None
            }
        }
    }

    fn put_named(&self, name: &str, b: &[u8]) {
        self.put_as(name.to_string(), b)
    }

    fn put(&self, key: &OPath, start: u64, b: &[u8]) {
        self.put_as(Self::name(key, start, b.len() as u64), b)
    }

    fn put_as(&self, name: String, b: &[u8]) {
        let len = b.len() as u64;
        if len > self.cap / 8 {
            return;
        }
        if self
            .idx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .map
            .contains_key(&name)
        {
            return;
        }
        let tmp = self.dir.join(format!("{name}.{}.tmp", std::process::id()));
        if std::fs::write(&tmp, b).is_err() || std::fs::rename(&tmp, self.dir.join(&name)).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        let mut g = self.idx.lock().unwrap_or_else(|e| e.into_inner());
        g.tick += 1;
        let t = g.tick;
        if g.map.insert(name, (len, t)).is_none() {
            g.total += len;
        }
        self.evict(&mut g);
    }

    fn evict(&self, g: &mut CacheIdx) {
        while g.total > self.cap && !g.map.is_empty() {
            let k = g.map.iter().min_by_key(|(_, v)| v.1).map(|(k, _)| k.clone()).unwrap();
            let (n, _) = g.map.remove(&k).unwrap();
            g.total -= n;
            let _ = std::fs::remove_file(self.dir.join(&k));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use object_store::PutPayload;

    fn data(n: usize) -> Vec<u8> {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    /// An InMemory store under `url` holding `objects` (key relative to the url's prefix).
    fn mem(url: &str, cfg: S3Cfg, objects: &[(&str, &[u8])]) -> Arc<Remote> {
        let store = Arc::new(InMemory::new());
        let prefix = url.split_once("://").unwrap().1.split_once('/').map_or("", |x| x.1);
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        for (k, b) in objects {
            let key = OPath::from(format!("{prefix}/{k}"));
            rt.block_on(store.put(&key, PutPayload::from(b.to_vec()))).unwrap();
        }
        register(url, store, cfg).unwrap()
    }

    fn quiet(r: &Remote) {
        // wait for GETs of released parts still running
        for _ in 0..500 {
            if r.sched.lock().unwrap_or_else(|e| e.into_inner()).running == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    #[test]
    fn coalescing_merges_small_gaps_and_splits_long_runs() {
        let r = coalesce(&[(0, 10), (15, 20), (100, 120), (2000, 2100)], 10, 100);
        let spans: Vec<(u64, u64)> = r.iter().map(|x| (x.0, x.1)).collect();
        assert_eq!(spans, vec![(0, 20), (100, 120), (2000, 2100)]);
        assert_eq!(r[0].2, vec![(0, 10), (15, 20)]);
        // the size cap ends a merge
        let r = coalesce(&[(0, 60), (60, 120)], 10, 100);
        assert_eq!(r.len(), 2);
        // one range longer than the cap: pieces, none cacheable as a chunk
        let r = coalesce(&[(0, 250)], 0, 100);
        let spans: Vec<(u64, u64)> = r.iter().map(|x| (x.0, x.1)).collect();
        assert_eq!(spans, vec![(0, 100), (100, 200), (200, 250)]);
        assert!(r.iter().all(|x| x.2.is_empty()));
    }

    #[test]
    fn reads_match_the_object_planned_or_not() {
        let d = data(3_000_000);
        let r = mem("mem://t1/pre", S3Cfg::default(), &[("a/f.bin", &d)]);
        let p = Path::new("mem://t1/pre/a/f.bin");
        assert!(exists(p));
        assert!(!exists(Path::new("mem://t1/pre/a/missing")));
        assert_eq!(read(p).unwrap(), d);
        let Src::Remote(f) = open_src(p).unwrap() else { panic!() };
        assert_eq!(f.len(), d.len() as u64);
        f.clear();
        f.plan(vec![(1000, 5000), (5100, 9000), (2_000_000, 2_100_000)]);
        // inside a part, across two parts and a gap, in no part, and at the very end
        for (s, n) in [
            (1000u64, 4000usize),
            (1500, 7000),
            (4000, 2_100_000),
            (2_500_000, 10),
            (2_999_990, 10),
        ] {
            let got = f.get_bytes(s, n).unwrap();
            assert_eq!(&got[..], &d[s as usize..s as usize + n], "[{s}, +{n})");
        }
        // a part read to its end is released; reading it again refetches it
        assert_eq!(&f.get_bytes(5100, 3900).unwrap()[..], &d[5100..9000]);
        assert_eq!(&f.get_bytes(5100, 100).unwrap()[..], &d[5100..5200]);
        let mut rr = RRead {
            f: f.clone(),
            pos: 123_456,
        };
        let mut buf = vec![0u8; 300_000];
        rr.read_exact(&mut buf).unwrap();
        assert_eq!(&buf[..], &d[123_456..423_456]);
        assert!(f.get_bytes(2_999_999, 2).is_err(), "past the end");
        drop((f, rr));
        quiet(&r);
        assert_eq!(
            r.sched.lock().unwrap_or_else(|e| e.into_inner()).held,
            0,
            "every byte of the window released"
        );
    }

    #[test]
    fn the_window_bounds_prefetch_and_a_waiting_reader_jumps_it() {
        let d = data(8 << 20);
        let cfg = S3Cfg {
            concurrency: 4,
            inflight_mb: 1,
            coalesce_kb: 0,
            ..Default::default()
        };
        let r = mem("mem://t2/p", cfg, &[("f", &d)]);
        let f = RFile::new(r.clone(), OPath::from("p/f"), d.len() as u64);
        // 32 chunks of 200 KB with gaps: 6.4 MB planned against a 1 MB window
        let chunks: Vec<(u64, u64)> = (0..32u64).map(|i| (i * 256 << 10, (i * 256 << 10) + 200_000)).collect();
        f.plan(chunks.clone());
        // read back to front: each read needs a part the window has not reached yet
        for &(s, e) in chunks.iter().rev() {
            let got = f.get_bytes(s, (e - s) as usize).unwrap();
            assert_eq!(&got[..], &d[s as usize..e as usize]);
        }
        let peak = r.stats.held_peak_bytes.load(Ordering::Relaxed);
        // the window, plus the one part a reader waits on
        assert!(peak <= (1 << 20) + 200_000, "window peak {peak}");
        assert_eq!(r.stats.gets.load(Ordering::Relaxed), 32);
        assert_eq!(r.stats.demand_gets.load(Ordering::Relaxed), 0);
        drop(f);
        quiet(&r);
        let s = r.sched.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(s.held, 0);
    }

    #[test]
    fn dropping_a_file_cancels_its_queued_gets() {
        let d = data(4 << 20);
        let cfg = S3Cfg {
            concurrency: 1,
            inflight_mb: 1,
            coalesce_kb: 0,
            ..Default::default()
        };
        let r = mem("mem://t3/p", cfg, &[("f", &d)]);
        let f = RFile::new(r.clone(), OPath::from("p/f"), d.len() as u64);
        f.plan((0..16u64).map(|i| (i << 18, (i << 18) + 100_000)).collect());
        drop(f);
        quiet(&r);
        r.pump();
        quiet(&r);
        let s = r.sched.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(s.held, 0);
        assert!(s.queue.is_empty());
        assert!(r.stats.gets.load(Ordering::Relaxed) < 16, "queued GETs were not issued");
    }

    #[test]
    fn a_memory_bound_shrinks_the_window_but_never_stalls_a_reader() {
        let d = data(8 << 20);
        let cfg = S3Cfg {
            concurrency: 8,
            inflight_mb: 4,
            coalesce_kb: 0,
            ..Default::default()
        };
        let r = mem("mem://t9/p", cfg, &[("f", &d)]);
        // the reader near its RSS cap: room for one 200 KB part
        r.set_window_limit(250_000);
        let f = RFile::new(r.clone(), OPath::from("p/f"), d.len() as u64);
        let chunks: Vec<(u64, u64)> = (0..16u64).map(|i| (i * 256 << 10, (i * 256 << 10) + 200_000)).collect();
        f.plan(chunks.clone());
        quiet(&r);
        // without the bound the 4 MB window would have prefetched every part
        assert_eq!(r.stats.gets.load(Ordering::Relaxed), 1);
        assert!(r.stats.window_limited.load(Ordering::Relaxed) > 0);
        for &(s, e) in chunks.iter().rev() {
            let got = f.get_bytes(s, (e - s) as usize).unwrap();
            assert_eq!(&got[..], &d[s as usize..e as usize]);
        }
        let peak = r.stats.held_peak_bytes.load(Ordering::Relaxed);
        // the one part the window admits, plus the one a reader waits on
        assert!(peak <= 400_000, "window peak {peak}");
        assert_eq!(r.stats.gets.load(Ordering::Relaxed), 16);
        // a raised bound starts the queued GETs at once
        let f = RFile::new(r.clone(), OPath::from("p/f"), d.len() as u64);
        f.plan(chunks.clone());
        quiet(&r);
        r.set_window_limit(u64::MAX);
        quiet(&r);
        assert!(r.stats.gets.load(Ordering::Relaxed) >= 16 + 16, "queued GETs were not started");
        drop(f);
        quiet(&r);
        assert_eq!(r.sched.lock().unwrap_or_else(|e| e.into_inner()).held, 0);
    }

    #[test]
    fn whole_object_reads_are_meta_gets_not_demand_gets() {
        let d = data(300_000);
        let r = mem("mem://t10/p", S3Cfg::default(), &[("catalog.json", &d)]);
        let got = read(Path::new("mem://t10/p/catalog.json")).unwrap();
        assert_eq!(got, d);
        assert_eq!(r.stats.meta_gets.load(Ordering::Relaxed), 1);
        assert_eq!(r.stats.demand_gets.load(Ordering::Relaxed), 0);
        assert_eq!(r.stats.gets.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn mapped_bodies_hold_exactly_the_range() {
        let d = data(1 << 20);
        let r = mem("mem://t11/p", S3Cfg::default(), &[("f", &d)]);
        let f = RFile::new(r.clone(), OPath::from("p/f"), d.len() as u64);
        // one part far above MAP_MIN, one below it
        f.plan(vec![(0, 700_000), (900_000, 900_000 + 1000)]);
        assert_eq!(&f.get_bytes(10, 600_000).unwrap()[..], &d[10..600_010]);
        assert_eq!(&f.get_bytes(600_010, 99_990).unwrap()[..], &d[600_010..700_000]);
        assert_eq!(&f.get_bytes(900_000, 1000).unwrap()[..], &d[900_000..901_000]);
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let g = rt
            .block_on(r.store.get_opts(&OPath::from("p/f"), GetOptions { range: Some(GetRange::Bounded(0..200_000)), ..Default::default() }))
            .unwrap();
        // a body that does not match the length asked for is an error, not a short buffer
        assert!(rt.block_on(body(g, 300_000)).is_err());
    }

    #[test]
    fn the_disk_cache_serves_a_second_plan_and_stays_under_its_bound() {
        let d = data(2 << 20);
        let dir = tempfile::tempdir().unwrap();
        let cfg = S3Cfg {
            coalesce_kb: 1024,
            cache_dir: Some(dir.path().to_path_buf()),
            cache_gb: 1.0,
            ..Default::default()
        };
        let r = mem("mem://t4/p", cfg, &[("f", &d)]);
        let chunks = vec![(0u64, 50_000u64), (60_000, 90_000), (500_000, 700_000)];
        let read_all = |f: &RFile| {
            for &(s, e) in &chunks {
                assert_eq!(
                    &f.get_bytes(s, (e - s) as usize).unwrap()[..],
                    &d[s as usize..e as usize]
                );
            }
        };
        let f = RFile::new(r.clone(), OPath::from("p/f"), d.len() as u64);
        f.plan(chunks.clone());
        read_all(&f);
        drop(f);
        assert_eq!(r.stats.gets.load(Ordering::Relaxed), 1, "one coalesced GET");
        // cache writes run in the background
        for _ in 0..500 {
            if r.cache.as_ref().unwrap().size().0 == 3 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(r.cache.as_ref().unwrap().size(), (3, 280_000));
        let f = RFile::new(r.clone(), OPath::from("p/f"), d.len() as u64);
        f.plan(chunks.clone());
        read_all(&f);
        assert_eq!(r.stats.gets.load(Ordering::Relaxed), 1, "served from the cache");
        assert_eq!(r.stats.cache_hits.load(Ordering::Relaxed), 3);
        // a smaller bound evicts the least recently used on reopen
        drop(f);
        let c = DiskCache::open(dir.path(), 250_000).unwrap();
        let (n, b) = c.size();
        assert!(b <= 250_000 && n == 2, "{n} chunks, {b} bytes");
        assert!(!c.has(&OPath::from("p/f"), 0, 50_000), "oldest evicted");
    }

    #[test]
    fn a_parquet_file_reads_the_same_from_the_store_with_one_data_get() {
        use arrow::array::{Float64Array, Int64Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::arrow::{ArrowWriter, ProjectionMask};
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
            Field::new("c", DataType::Int64, false),
        ]));
        let mut buf = Vec::new();
        let props = parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_row_count(Some(10_000))
            .build();
        let mut w = ArrowWriter::try_new(&mut buf, schema.clone(), Some(props)).unwrap();
        for k in 0..5i64 {
            let a = Int64Array::from_iter_values((0..10_000).map(|i| k * 10_000 + i));
            let b = Float64Array::from_iter_values((0..10_000).map(|i| (i * 7 % 13) as f64 / 3.0));
            let c = Int64Array::from_iter_values((0..10_000).map(|i| i * i));
            w.write(&RecordBatch::try_new(schema.clone(), vec![Arc::new(a), Arc::new(b), Arc::new(c)]).unwrap())
                .unwrap();
        }
        w.close().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let local = dir.path().join("x.parquet");
        std::fs::write(&local, &buf).unwrap();
        let r = mem("mem://t5/st", S3Cfg::default(), &[("d/x.parquet", &buf)]);
        let remote = Path::new("mem://t5/st/d/x.parquet");
        let read = |p: &Path| {
            let (b, src) = crate::eval::footer::builder_src(p).unwrap();
            let mask = ProjectionMask::leaves(b.parquet_schema(), [0, 2]);
            let rgs = vec![1usize, 3];
            src.prefetch(b.metadata(), &rgs, Some(&mask));
            let r = b.with_row_groups(rgs).with_projection(mask).build().unwrap();
            r.map(|x| x.unwrap()).collect::<Vec<_>>()
        };
        let want = read(&local);
        assert_eq!(want.iter().map(|b| b.num_rows()).sum::<usize>(), 20_000);
        assert_eq!(read(remote), want);
        // the suffix GET (footer, read twice by the parser) and one GET for columns a and c of
        // row groups 1 and 3 (the gaps are under the 1 MB coalescing gap)
        assert_eq!(r.stats.gets.load(Ordering::Relaxed), 2);
        assert_eq!(r.stats.footer_gets.load(Ordering::Relaxed), 1);
        assert_eq!(r.stats.demand_gets.load(Ordering::Relaxed), 0);
        // footer cached: a second open costs only the data
        assert_eq!(read(remote), want);
        assert_eq!(r.stats.gets.load(Ordering::Relaxed), 3);
        // a plan-less reader still reads correctly (on-demand GETs)
        let b = ParquetRecordBatchReaderBuilder::try_new(open_src(remote).unwrap()).unwrap();
        let n: usize = b.build().unwrap().map(|x| x.unwrap().num_rows()).sum();
        assert_eq!(n, 50_000);
        quiet(&r);
        assert_eq!(r.sched.lock().unwrap_or_else(|e| e.into_inner()).held, 0);
    }

    #[test]
    fn the_default_cache_dir_is_per_store_and_stable() {
        let a = default_cache_dir("s3://b/st");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let u = format!("s3://test-{}/st", std::process::id());
            let d = make_default_cache_dir(&u).unwrap();
            assert_eq!(std::fs::metadata(&d).unwrap().permissions().mode() & 0o777, 0o700);
            assert_eq!(std::fs::metadata(d.parent().unwrap()).unwrap().permissions().mode() & 0o777, 0o700);
            make_default_cache_dir(&u).unwrap(); // again: fine
            std::fs::remove_dir(&d).unwrap();
        }
        assert_eq!(a, default_cache_dir("s3://b/st/"));
        assert_ne!(a, default_cache_dir("s3://b/other"));
        assert!(a.starts_with(std::env::temp_dir().join("pgai-mon-s3cache")));
    }

    #[test]
    fn a_get_never_spans_a_cached_chunk() {
        // the middle chunk is on disk: the chunks around it are two GETs, not one over it (that
        // GET overlapped the cached chunk's part, and once that part was read every later read
        // inside the GET went out on demand)
        let d = data(1 << 20);
        let dir = tempfile::tempdir().unwrap();
        let cfg = S3Cfg {
            coalesce_kb: 1024,
            cache_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        };
        let r = mem("mem://t11/p", cfg, &[("f", &d)]);
        let f = RFile::new(r.clone(), OPath::from("p/f"), d.len() as u64);
        f.plan(vec![(200_000, 300_000)]);
        f.get_bytes(200_000, 100_000).unwrap();
        drop(f);
        for _ in 0..500 {
            if r.cache.as_ref().unwrap().size().0 == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(r.cache.as_ref().unwrap().size().0, 1);
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let (g0, b0) = (l(&r.stats.gets), l(&r.stats.bytes));
        let f = RFile::new(r.clone(), OPath::from("p/f"), d.len() as u64);
        let chunks = [(0u64, 150_000u64), (200_000, 300_000), (350_000, 500_000)];
        f.plan(chunks.to_vec());
        // page by page, all three side by side
        for i in 0..10u64 {
            for &(s, e) in &chunks {
                let (a, b) = (s + (e - s) * i / 10, s + (e - s) * (i + 1) / 10);
                assert_eq!(&f.get_bytes(a, (b - a) as usize).unwrap()[..], &d[a as usize..b as usize]);
            }
        }
        assert_eq!(l(&r.stats.gets) - g0, 2);
        assert_eq!(l(&r.stats.bytes) - b0, 300_000, "nothing the disk holds");
        assert_eq!(l(&r.stats.demand_gets), 0);
    }

    #[test]
    fn coalesced_chunks_read_side_by_side_stay_until_each_is_read() {
        // two column chunks in one GET, read page by page in turn (as arrow reads a row group):
        // the second chunk reaches the GET's last byte while the first still has pages left
        let d = data(400_000);
        let r = mem("mem://t9/pre", S3Cfg::default(), &[("f.bin", &d)]);
        let Src::Remote(f) = open_src(Path::new("mem://t9/pre/f.bin")).unwrap() else { panic!() };
        f.clear();
        quiet(&r);
        let (g0, d0) = (r.stats.gets.load(Ordering::Relaxed), r.stats.demand_gets.load(Ordering::Relaxed));
        f.plan(vec![(0, 100_000), (100_000, 200_000)]);
        let pages = |c: u64| (0..4u64).map(move |i| (c + i * 25_000, 25_000usize));
        for ((a, n), (b, m)) in pages(0).zip(pages(100_000)) {
            // a page header through `Read`, then the page
            let mut h = [0u8; 1];
            let mut rr = RRead { f: f.clone(), pos: a };
            rr.read_exact(&mut h).unwrap();
            assert_eq!(h[0], d[a as usize]);
            assert_eq!(&f.get_bytes(a + 1, n - 1).unwrap()[..], &d[a as usize + 1..a as usize + n]);
            assert_eq!(&f.get_bytes(b, m).unwrap()[..], &d[b as usize..b as usize + m]);
        }
        quiet(&r);
        assert_eq!(r.stats.gets.load(Ordering::Relaxed) - g0, 1, "one coalesced GET");
        assert_eq!(r.stats.demand_gets.load(Ordering::Relaxed) - d0, 0, "no unplanned reads");
        // read to the end of both chunks: released, its window freed
        assert!(f.parts.lock().unwrap().iter().all(|p| matches!(p.inner.lock().unwrap().state, PState::Done)));
    }

    /// A Parquet-shaped object: `body` bytes, a `meta`-byte footer, its length and the magic.
    fn parquet_like(body: usize, meta: usize) -> Vec<u8> {
        let mut v = data(body + meta);
        v.extend_from_slice(&(meta as u32).to_le_bytes());
        v.extend_from_slice(b"PAR1");
        v
    }

    #[test]
    fn footers_take_one_round_then_none_from_memory_or_disk() {
        let dir = tempfile::tempdir().unwrap();
        // longer than the suffix GET (footers this big come only from older stores)
        let obj = parquet_like(1_000_000, 300_000);
        let cfg = S3Cfg {
            cache_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        };
        let objs: Vec<(String, Vec<u8>)> = (0..6).map(|i| (format!("f{i}.parquet"), obj.clone())).collect();
        let refs: Vec<(&str, &[u8])> = objs.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        let r = mem("mem://t10/pre", cfg.clone(), &refs);
        let paths: Vec<PathBuf> = (0..6).map(|i| PathBuf::from(format!("mem://t10/pre/f{i}.parquet"))).collect();
        let check = |src: &Src| {
            let Src::Remote(f) = src else { panic!() };
            let n = obj.len();
            // the whole footer is in hand: reading it needs no GET
            let b = f.get_bytes((n - 300_008) as u64, 300_008).unwrap();
            assert_eq!(&b[..], &obj[n - 300_008..]);
        };
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        for s in open_many(&paths) {
            check(&s.unwrap());
        }
        // the suffix plus one GET for the rest of each footer; nothing on demand
        assert_eq!(l(&r.stats.gets), 12);
        assert_eq!(l(&r.stats.footer_gets), 12);
        assert_eq!(l(&r.stats.demand_gets), 0);
        // again (the parsed-metadata cache evicted it): from memory
        for s in open_many(&paths) {
            check(&s.unwrap());
        }
        assert_eq!(l(&r.stats.gets), 12);
        assert_eq!(l(&r.stats.footer_mem_hits), 6);
        // a fresh process (new Remote, same cache dir): from disk
        let objs1: Vec<(&str, &[u8])> = refs.clone();
        let r2 = mem("mem://t10/pre", cfg, &objs1);
        for s in open_many(&paths) {
            check(&s.unwrap());
        }
        assert_eq!(l(&r2.stats.gets), 0);
        assert_eq!(l(&r2.stats.footer_disk_hits), 6);
        // a footer under the suffix: one GET
        let small = parquet_like(1_000_000, 200_000);
        let r3 = mem("mem://t12/pre", S3Cfg::default(), &[("g.parquet", &small)]);
        let Src::Remote(f) = open_src(Path::new("mem://t12/pre/g.parquet")).unwrap() else { panic!() };
        let n = small.len();
        assert_eq!(&f.get_bytes((n - 200_008) as u64, 200_008).unwrap()[..], &small[n - 200_008..]);
        assert_eq!(l(&r3.stats.gets), 1);
    }
}
