//! The SQL reader as a child process with an OS-enforced memory cap.
//!
//! DuckDB's `memory_limit` bounds buffer-managed memory (hash tables, sorts) but not the
//! heap of scalar results (`repeat('x', 7e8)`, `unnest(range(1e8))`), which reached 7 GB in
//! testing. So the reader runs in its own process:
//! - Linux: `RLIMIT_DATA` (private writable mappings, incl. anonymous mmaps since 4.7) makes
//!   allocations past the limit fail inside the child;
//! - everywhere: a watchdog samples the child's RSS every 10 ms and kills it above the cap.
//!
//! The parent (collector, WAL, store) never runs user SQL. When the child dies, every request it
//! held gets a typed error (`out_of_memory` if the watchdog or the limit killed it,
//! `reader_restarted` otherwise) and the next request starts a fresh child (~100 ms).
//!
//! Protocol (child stdin/stdout, one JSON object per line):
//!   -> {"id": 1, "op": "sql" | "sql_columnar" | "sql_objects" | "sql_csv" | "schema" | "stats", "sql": "..."}
//!   <- {"id": 1, "ok": <result JSON>}  |  {"id": 1, "err": {"type": .., "message": ..}}
//!      (`"gen"`, `"c"`, `"rc"` and, on an object store, `"s3":[gets,bytes,demand,footer gets]` — the
//!      child's running totals — come before `"ok"` / `"err"`)
use crate::catalog::Catalog;
use crate::config::HttpCfg;
use crate::query::{DictEntry, QueryError, Reader, Reply};
use crate::store::Shared;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Secrets of the server process that the SQL worker is started without: it runs callers' SQL,
/// reads files and (for a bucket) fetches objects, and never authenticates anyone or connects
/// to Postgres.
const WORKER_ENV_REMOVED: [&str; 6] = [
    "PGAI_MON_TOKEN",
    "PGAI_MON_READ_TOKEN",
    "PGAI_MON_AUTH_PASSWORD",
    "PGPASSWORD",
    "REPLICATOR_PASSWORD",
    "GF_SECURITY_ADMIN_PASSWORD",
];

#[derive(Serialize, Deserialize)]
pub struct Spec {
    pub store: PathBuf,
    pub scratch: PathBuf,
    pub http: HttpCfg,
    pub views_sql: String,
    pub dict: Vec<DictEntry>,
    /// The writer's local directory for an object-store `store` (objstore::set_overlay).
    #[serde(default)]
    pub overlay: Option<PathBuf>,
}

/// Resident set size of a process, bytes.
pub fn rss_bytes(pid: u32) -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string(format!("/proc/{pid}/statm")).ok()?;
        let pages: u64 = s.split_whitespace().nth(1)?.parse().ok()?;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        Some(pages * page)
    }
    #[cfg(target_os = "macos")]
    {
        let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
        let n = unsafe {
            libc::proc_pidinfo(
                pid as libc::c_int,
                libc::PROC_PIDTASKINFO,
                0,
                &mut info as *mut _ as *mut libc::c_void,
                size,
            )
        };
        if n == size {
            Some(info.pti_resident_size)
        } else {
            None
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = pid;
        None
    }
}

// ------------------------------------------------------------------ child side

/// Hand freed heap back to the OS after each query, so RSS tracks live memory and the
/// watchdog doesn't kill a reader for memory it has already freed.
pub fn release_free_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut libc::c_void, goal: libc::size_t) -> libc::size_t;
        }
        malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
    }
}

/// Bytes the reader's row cache (eval/rowcache.rs) and raw S3 footer cache (objstore, bounded by
/// --s3-footer-mb) held at its last reply. The idle recycle
/// counts the reader's RSS without them: the reader drops the cache itself first when that
/// alone would not bring it under the threshold (see child_main).
static READER_CACHE_BYTES: AtomicU64 = AtomicU64::new(0);

/// The prefetch window's memory bound: what it holds now plus the headroom between `rss` and
/// `soft_pct` of `cap` (the window's own bytes are already in `rss`).
fn window_limit(rss: u64, held: u64, cap: u64, soft_pct: u64) -> u64 {
    held.saturating_add((cap * soft_pct / 100).saturating_sub(rss))
}

/// On an S3 store, keep the reader's prefetch window inside its RSS headroom under `soft_pct`
/// of `cap` (--s3-window-soft-pct; objstore::set_window_limit), sampled every 10 ms: near the
/// cap it prefetches less instead of being killed. `soft_pct` 0 leaves the window unbounded.
pub fn spawn_window_governor(cap: u64, soft_pct: u64) {
    if soft_pct == 0 {
        return;
    }
    let pid = std::process::id();
    let _ = std::thread::Builder::new()
        .name("s3-window".into())
        .spawn(move || loop {
            if let Some(rss) = rss_bytes(pid) {
                crate::objstore::set_window_limit(window_limit(rss, crate::objstore::window_bytes(), cap, soft_pct));
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        });
}

/// An idle reader above this share of the RSS cap is recycled (the cache bytes excluded).
const IDLE_RECYCLE_PCT: u64 = 55;
/// The reader clears its row cache before a query when its RSS is above this share of the cap.
const CACHE_CLEAR_PCT: u64 = 57;

/// An idle reader at `rss` holding `cache` bytes of row cache is recycled when what is left
/// without the cache is still above IDLE_RECYCLE_PCT of `cap`.
fn recycle_due(rss: u64, cache: u64, cap: u64) -> bool {
    rss.saturating_sub(cache) > cap * IDLE_RECYCLE_PCT / 100
}

#[derive(Deserialize)]
struct Req {
    id: u64,
    op: String,
    #[serde(default)]
    sql: String,
}

/// Re-read catalog.json when its generation changed (the parent rewrites it atomically).
/// A remote catalog is re-read at most every `ttl_ms` (a GET per query would cost a round trip).
/// When the remote catalog was last read (the reader's start loads it too).
static CATALOG_READ: Mutex<Option<std::time::Instant>> = Mutex::new(None);

fn reload_catalog(shared: &Shared, ttl_ms: u64) {
    if crate::objstore::is_remote(&shared.root) {
        let mut g = CATALOG_READ.lock().unwrap();
        if g.is_some_and(|t| t.elapsed().as_millis() < ttl_ms as u128) {
            return;
        }
        *g = Some(std::time::Instant::now());
    }
    let c = if crate::objstore::is_remote(&shared.root) {
        // bounded read, every file path checked (a bad catalog keeps the previous one)
        match crate::catalog::load_remote(&shared.root) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("sql-worker: catalog not reloaded: {e:#}");
                return;
            }
        }
    } else {
        let Ok(bytes) = std::fs::read(crate::catalog::path(&shared.root)) else {
            return;
        };
        let Ok(c) = serde_json::from_slice::<Catalog>(&bytes) else {
            return;
        };
        c
    };
    if c.generation != shared.generation.load(Ordering::Acquire) {
        let g = c.generation;
        *shared.catalog.lock().unwrap() = c;
        shared.generation.store(g, Ordering::Release);
    }
}

pub fn child_main(spec_path: &Path) -> anyhow::Result<()> {
    let spec: Spec = serde_json::from_slice(&std::fs::read(spec_path)?)?;
    if crate::objstore::is_remote(&spec.store) {
        crate::objstore::open(&spec.store.to_string_lossy(), &spec.http.s3)?;
        if spec.overlay.is_some() {
            crate::objstore::set_overlay(&spec.store.to_string_lossy(), spec.overlay.clone())?;
        }
        spawn_window_governor(spec.http.reader_rss_cap_mb << 20, spec.http.s3.window_soft_pct);
    }
    let cat = Catalog::load_or_new(&spec.store, crate::store::DEFAULT_LAYOUT)?;
    // just read: the first query does not read it again (one GET per reader start, not two)
    *CATALOG_READ.lock().unwrap() = Some(std::time::Instant::now());
    let shared = Shared {
        root: spec.store.clone(),
        generation: Arc::new(AtomicU64::new(cat.generation)),
        catalog: Arc::new(Mutex::new(cat)),
        stats: Default::default(),
        known_series: Default::default(),
        fp_to_queryid: Default::default(),
        active_queries: Default::default(),
        scratch: spec.scratch.clone(),
    };
    let reader = Arc::new(Reader::new(
        shared.clone(),
        spec.http.clone(),
        spec.views_sql,
        spec.dict,
    )?);
    let out = Arc::new(Mutex::new(std::io::stdout()));
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        let Ok(req) = serde_json::from_str::<Req>(&line) else {
            continue;
        };
        reload_catalog(&shared, spec.http.s3.catalog_ttl_ms);
        let reader = reader.clone();
        let out = out.clone();
        let cap = spec.http.reader_rss_cap_mb << 20;
        static INFLIGHT: AtomicU64 = AtomicU64::new(0);
        INFLIGHT.fetch_add(1, Ordering::AcqRel);
        // a high reader drops its row cache before the query, not after a kill
        let rss_now = || rss_bytes(std::process::id()).unwrap_or(0);
        if crate::eval::rowcache::BYTES.load(Ordering::Relaxed) > 0 && rss_now() > cap * CACHE_CLEAR_PCT / 100 {
            crate::eval::rowcache::clear();
            release_free_memory();
        }
        std::thread::spawn(move || {
            // The result is serialised straight into the pipe: building it as a String and then
            // a framed line kept two extra copies of a max_bytes result inside the capped child.
            let (mut gen, mut cacheable) = (0u64, false);
            let res: Result<Box<dyn erased::Ser>, QueryError> = match req.op.as_str() {
                "schema" => reader.schema().map(|v| Box::new(v) as Box<dyn erased::Ser>),
                "stats" => Ok(Box::new(reader.stats()) as Box<dyn erased::Ser>),
                op => reader.run_fmt(&req.sql, crate::query::Fmt::from_op(op)).map(|r| {
                    gen = r.gen;
                    cacheable = r.cacheable;
                    Box::new(r) as Box<dyn erased::Ser>
                }),
            };
            // the last query out: if the reader would be recycled even without its row cache,
            // drop the cache now (the recycle loses it anyway, and the drop may avoid it)
            if INFLIGHT.fetch_sub(1, Ordering::AcqRel) == 1 {
                let cb = crate::eval::rowcache::BYTES.load(Ordering::Relaxed);
                if cb > 0 && recycle_due(rss_now(), cb + crate::objstore::footer_mem_bytes(), cap) {
                    crate::eval::rowcache::clear();
                    release_free_memory();
                }
            }
            // the reader's caches: row cache and raw S3 footers (the idle recycle discounts them)
            let rc = crate::eval::rowcache::BYTES.load(Ordering::Relaxed) + crate::objstore::footer_mem_bytes();
            {
                let mut o = out.lock().unwrap();
                let mut w = std::io::BufWriter::with_capacity(64 << 10, &mut *o);
                let s3 = match crate::objstore::totals() {
                    Some([g, b, d, f]) => format!(",\"s3\":[{g},{b},{d},{f}]"),
                    None => String::new(),
                };
                let _ = match res {
                    Ok(v) => write!(
                        w,
                        "{{\"id\":{},\"gen\":{gen},\"c\":{},\"rc\":{rc}{s3},\"ok\":",
                        req.id, cacheable as u8
                    )
                    .and_then(|_| v.write_to(&mut w)),
                    Err(e) => write!(w, "{{\"id\":{}{s3},\"err\":", req.id)
                        .and_then(|_| serde_json::to_writer(&mut w, &e).map_err(std::io::Error::from)),
                };
                let _ = w.write_all(b"}\n");
                let _ = w.flush();
            }
            release_free_memory();
        });
    }
    Ok(())
}

mod erased {
    pub trait Ser: Send {
        fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()>;
    }
    impl<T: serde::Serialize + Send> Ser for T {
        fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
            serde_json::to_writer(w, self).map_err(std::io::Error::from)
        }
    }
}

// ------------------------------------------------------------------ parent side

type Pending = Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<Result<Reply, QueryError>>>>>;

pub struct ReaderProc {
    spec_path: PathBuf,
    spec_json: Vec<u8>,
    pub rss_cap: u64,
    data_limit: u64,
    timeout: Duration,
    stdin: Arc<Mutex<Option<(ChildStdin, u32, Pending)>>>,
    next_id: AtomicU64,
    pub pid: Arc<AtomicU32>,
    pub restarts: Arc<AtomicU64>,
    pub mem_kills: Arc<AtomicU64>,
    pub recycles: Arc<AtomicU64>,
    /// The child's RSS at the watchdog's last sample (0 = no child), for admission's memory gate.
    pub last_rss: Arc<AtomicU64>,
    /// Highest RSS the watchdog saw since the last reset (admission resets it around a big
    /// shape's solo run to see what that shape needs).
    pub peak_rss: Arc<AtomicU64>,
    /// Replies relayed (the parent trims its heap every so often; see `trim_loop`).
    pub relayed: Arc<AtomicU64>,
    /// Object-store counters across child restarts.
    pub s3: Arc<S3Totals>,
    me: Mutex<std::sync::Weak<ReaderProc>>,
}

#[derive(Deserialize)]
struct QueryErrorOwned {
    #[serde(rename = "type")]
    kind: String,
    message: String,
}

fn static_kind(k: &str) -> &'static str {
    const KINDS: &[&str] = &[
        "not_read_only",
        "forbidden",
        "parse_error",
        "binder_error",
        "catalog_error",
        "invalid_input",
        "conversion_error",
        "timeout",
        "busy",
        "out_of_memory",
        "result_too_large",
        "execution_error",
        "reader_restarted",
    ];
    KINDS.iter().find(|x| **x == k).copied().unwrap_or("internal")
}

/// Object-store counters of the reader, kept by the parent across child restarts: what dead
/// children reached (`base`) plus the live child's latest totals (`last`). A child's GETs after
/// its last reply (the query that got it killed) are not counted.
#[derive(Default)]
pub struct S3Totals {
    base: [AtomicU64; 4],
    last: [AtomicU64; 4],
    seen: AtomicBool,
}

impl S3Totals {
    fn update(&self, t: [u64; 4]) {
        self.seen.store(true, Ordering::Relaxed);
        for (a, v) in self.last.iter().zip(t) {
            // replies are written by several threads: keep the highest (counters only grow)
            a.fetch_max(v, Ordering::Relaxed);
        }
    }

    /// The child exited: fold its totals into the base.
    fn child_gone(&self) {
        for (b, l) in self.base.iter().zip(&self.last) {
            b.fetch_add(l.swap(0, Ordering::Relaxed), Ordering::Relaxed);
        }
    }

    /// [gets, bytes, demand GETs, footer GETs] since the parent started; `None` before any
    /// reply carried them (no object store).
    pub fn get(&self) -> Option<[u64; 4]> {
        if !self.seen.load(Ordering::Relaxed) {
            return None;
        }
        let mut t = [0u64; 4];
        for (i, x) in t.iter_mut().enumerate() {
            *x = self.base[i].load(Ordering::Relaxed) + self.last[i].load(Ordering::Relaxed);
        }
        Some(t)
    }
}

#[cfg(test)]
fn parse_reply(line: String) -> Option<(u64, Result<Reply, QueryError>)> {
    parse_reply_s3(line).map(|(id, _, v)| (id, v))
}

/// Split a reply line `{"id":N,"gen":G,"c":0|1,"rc":B,"s3":[..],"ok":BODY}` / `{"id":N,"err":{..}}`
/// without parsing BODY; the child's object-store totals come back when the line has them.
pub fn parse_reply_s3(mut line: String) -> Option<(u64, Option<[u64; 4]>, Result<Reply, QueryError>)> {
    let num = |s: &str| -> Option<(u64, usize)> {
        let n = s.bytes().take_while(|b| b.is_ascii_digit()).count();
        Some((s[..n].parse().ok()?, n))
    };
    let rest = line.strip_prefix("{\"id\":")?;
    let (id, n) = num(rest)?;
    let mut after = &rest[n..];
    let mut gen = 0;
    if let Some(g) = after.strip_prefix(",\"gen\":") {
        let (v, n) = num(g)?;
        gen = v;
        after = &g[n..];
    }
    let mut cacheable = false;
    if let Some(c) = after.strip_prefix(",\"c\":") {
        let (v, n) = num(c)?;
        cacheable = v == 1;
        after = &c[n..];
    }
    if let Some(c) = after.strip_prefix(",\"rc\":") {
        let (v, n) = num(c)?;
        READER_CACHE_BYTES.store(v, Ordering::Relaxed);
        after = &c[n..];
    }
    let mut s3 = None;
    if let Some(x) = after.strip_prefix(",\"s3\":[") {
        let end = x.find(']')?;
        let mut t = [0u64; 4];
        let mut it = x[..end].split(',');
        for v in t.iter_mut() {
            *v = it.next()?.parse().ok()?;
        }
        s3 = Some(t);
        after = &x[end + 1..];
    }
    if let Some(body) = after.strip_prefix(",\"ok\":") {
        if !body.ends_with('}') {
            return None;
        }
        let start = line.len() - body.len();
        line.pop(); // the closing brace of the envelope
        line.drain(..start);
        // CSV travels as a JSON string so embedded newlines cannot split the pipe frame.
        if line.starts_with('"') {
            line = serde_json::from_str(&line).ok()?;
        }
        return Some((
            id,
            s3,
            Ok(Reply {
                body: line,
                gen,
                cacheable,
            }),
        ));
    }
    let e = after.strip_prefix(",\"err\":")?.strip_suffix('}')?;
    let e: QueryErrorOwned = serde_json::from_str(e).ok()?;
    Some((
        id,
        s3,
        Err(QueryError {
            kind: static_kind(&e.kind),
            message: e.message,
        }),
    ))
}

impl ReaderProc {
    /// Return freed heap to the OS when replies were relayed since the last pass: a 16 MB
    /// result body is freed after the HTTP layer sends it, and macOS malloc otherwise keeps it
    /// (the parent reached 300 MiB RSS relaying 24 h dashboards).
    pub fn trim_loop(self: &Arc<Self>) {
        let relayed = self.relayed.clone();
        let _ = std::thread::Builder::new().name("parent-trim".into()).spawn(move || {
            let mut seen = 0;
            loop {
                std::thread::sleep(Duration::from_millis(500));
                let n = relayed.load(Ordering::Relaxed);
                if n != seen {
                    seen = n;
                    release_free_memory();
                }
            }
        });
    }

    pub fn new(spec: &Spec, spec_path: PathBuf) -> anyhow::Result<Arc<Self>> {
        let spec_json = serde_json::to_vec(spec)?;
        let p = Arc::new(Self {
            spec_path,
            spec_json,
            rss_cap: spec.http.reader_rss_cap_mb << 20,
            data_limit: spec.http.reader_data_limit_mb << 20,
            timeout: Duration::from_millis(spec.http.timeout_ms + 5_000),
            stdin: Default::default(),
            next_id: AtomicU64::new(1),
            pid: Default::default(),
            restarts: Default::default(),
            mem_kills: Default::default(),
            recycles: Default::default(),
            last_rss: Default::default(),
            peak_rss: Default::default(),
            relayed: Default::default(),
            s3: Default::default(),
            me: Mutex::new(std::sync::Weak::new()),
        });
        *p.me.lock().unwrap() = Arc::downgrade(&p);
        p.spawn()?;
        p.trim_loop();
        Ok(p)
    }

    fn spawn(&self) -> anyhow::Result<()> {
        // (Re)write the spec every time: another process opening the store (a CLI `compact` or
        // `query`) clears store/tmp, and a restart must not depend on the old file.
        std::fs::create_dir_all(self.spec_path.parent().unwrap())?;
        std::fs::write(&self.spec_path, &self.spec_json)?;
        let mut cmd = Command::new(std::env::current_exe()?);
        cmd.arg("sql-worker")
            .arg("--spec")
            .arg(&self.spec_path)
            .env("MALLOC_ARENA_MAX", "2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        // The only process that runs callers' SQL does not need the server's secrets.
        for k in WORKER_ENV_REMOVED {
            cmd.env_remove(k);
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt;
            let lim = self.data_limit;
            unsafe {
                cmd.pre_exec(move || {
                    let r = libc::rlimit {
                        rlim_cur: lim as libc::rlim_t,
                        rlim_max: lim as libc::rlim_t,
                    };
                    if lim > 0 && libc::setrlimit(libc::RLIMIT_DATA, &r) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = self.data_limit;
        let mut child: Child = cmd.spawn()?;
        let pid = child.id();
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let pending: Pending = Default::default();
        *self.stdin.lock().unwrap() = Some((stdin, pid, pending.clone()));
        self.pid.store(pid, Ordering::Relaxed);
        READER_CACHE_BYTES.store(0, Ordering::Relaxed); // a new reader starts empty
        let killed_for_mem = Arc::new(AtomicBool::new(false));
        let recycled = Arc::new(AtomicBool::new(false));
        // Watchdog: RSS above the cap -> SIGKILL. An idle reader above IDLE_RECYCLE_PCT of the
        // cap, not counting its row cache (READER_CACHE_BYTES), is
        // recycled (memory the allocator keeps after queries, notably on macOS, would otherwise
        // count against the next query's headroom); no request is affected.
        {
            let cap = self.rss_cap;
            let k = killed_for_mem.clone();
            let mem_kills = self.mem_kills.clone();
            let recycles = self.recycles.clone();
            let idle_pending = pending.clone();
            let rc = recycled.clone();
            let last = self.last_rss.clone();
            let peak = self.peak_rss.clone();
            std::thread::Builder::new()
                .name("reader-watchdog".into())
                .spawn(move || loop {
                    let rss = rss_bytes(pid);
                    last.store(rss.unwrap_or(0), Ordering::Relaxed);
                    peak.fetch_max(rss.unwrap_or(0), Ordering::Relaxed);
                    match rss {
                        Some(r)
                            if recycle_due(r, READER_CACHE_BYTES.load(Ordering::Relaxed), cap)
                                && idle_pending.lock().unwrap().is_empty() =>
                        {
                            // hold the pending lock so no request slips in while we stop it
                            let g = idle_pending.lock().unwrap();
                            if g.is_empty() {
                                recycles.fetch_add(1, Ordering::Relaxed);
                                rc.store(true, Ordering::Release);
                                unsafe {
                                    libc::kill(pid as libc::pid_t, libc::SIGKILL);
                                }
                                drop(g);
                                return;
                            }
                        }
                        Some(r) if r > cap => {
                            k.store(true, Ordering::Release);
                            mem_kills.fetch_add(1, Ordering::Relaxed);
                            unsafe {
                                libc::kill(pid as libc::pid_t, libc::SIGKILL);
                            }
                            return;
                        }
                        Some(_) => std::thread::sleep(Duration::from_millis(10)),
                        None => return,
                    }
                })?;
        }
        // Response dispatcher; on EOF, fail everything pending and mark the child gone.
        let restarts = self.restarts.clone();
        let relayed = self.relayed.clone();
        let s3 = self.s3.clone();
        let mypid = self.pid.clone();
        let me = self.me.lock().unwrap().clone();
        let slot = self.stdin.clone();
        std::thread::Builder::new().name("reader-io".into()).spawn(move || {
            let rd = std::io::BufReader::with_capacity(256 << 10, stdout);
            // Zero-copy relay: the child writes `{"id":N,"ok":<body>}`; the body is cut out of
            // the line in place (no parse of the body, no second copy), and that one String is
            // what the HTTP response sends.
            let mut rd = rd;
            loop {
                let mut line = String::new();
                match std::io::BufRead::read_line(&mut rd, &mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
                if line.ends_with('\n') {
                    line.pop();
                }
                let Some((id, t, v)) = parse_reply_s3(line) else {
                    continue;
                };
                if let Some(t) = t {
                    s3.update(t);
                }
                let tx = pending.lock().unwrap().remove(&id);
                if let Some(tx) = tx {
                    let _ = tx.send(v);
                }
                relayed.fetch_add(1, Ordering::Relaxed);
            }
            let status = child.wait().ok();
            let mem = killed_for_mem.load(Ordering::Acquire);
            let (kind, msg): (&'static str, String) = if mem {
                (
                    "out_of_memory",
                    "the SQL reader process passed its memory cap and was restarted".into(),
                )
            } else {
                (
                    "reader_restarted",
                    format!("the SQL reader process exited ({status:?}) and was restarted"),
                )
            };
            restarts.fetch_add(1, Ordering::Relaxed);
            s3.child_gone();
            // Clear the stdin slot if it still points at this child.
            {
                let mut s = slot.lock().unwrap();
                if s.as_ref().map(|(_, p, _)| *p) == Some(pid) {
                    *s = None;
                    mypid.store(0, Ordering::Relaxed);
                }
            }
            let drained: Vec<_> = pending.lock().unwrap().drain().collect();
            for (_, tx) in drained {
                let _ = tx.send(Err(QueryError {
                    kind,
                    message: msg.clone(),
                }));
            }
            // an idle recycle starts the next reader now, so no query pays its startup
            if recycled.load(Ordering::Acquire) {
                if let Some(p) = me.upgrade() {
                    let need = p.stdin.lock().unwrap().is_none();
                    if need {
                        let _ = p.spawn();
                    }
                }
            }
        })?;
        Ok(())
    }

    /// Send one request to the child (starting it if needed) and wait for the answer. A
    /// `reader_restarted` (the reader went away: an idle recycle racing the request, or a crash) is
    /// retried once on a fresh reader. Once only: the query that crashes the reader cannot be told
    /// from the queries that were running beside it, and each retry of it takes those down again.
    /// A query killed for memory is not retried.
    pub async fn call(&self, op: &str, sql: &str) -> Result<Reply, QueryError> {
        match self.call_once(op, sql).await {
            Err(e) if e.kind == "reader_restarted" => {
                tokio::time::sleep(Duration::from_millis(50)).await;
                self.call_once(op, sql).await
            }
            r => r,
        }
    }

    async fn call_once(&self, op: &str, sql: &str) -> Result<Reply, QueryError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = tokio::sync::oneshot::channel();
        let line = format!("{}\n", serde_json::json!({"id": id, "op": op, "sql": sql}));
        let sent = {
            let mut s = self.stdin.lock().unwrap();
            if s.is_none() {
                drop(s);
                if let Err(e) = self.spawn() {
                    return Err(QueryError {
                        kind: "internal",
                        message: format!("cannot start reader: {e}"),
                    });
                }
                s = self.stdin.lock().unwrap();
            }
            match s.as_mut() {
                Some((w, _, pending)) => {
                    pending.lock().unwrap().insert(id, tx);
                    w.write_all(line.as_bytes()).and_then(|_| w.flush()).is_ok()
                }
                None => false,
            }
        };
        if !sent {
            return Err(QueryError {
                kind: "reader_restarted",
                message: "the SQL reader was restarting; retry".into(),
            });
        }
        match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(r)) => r,
            Ok(Err(_)) => Err(QueryError {
                kind: "reader_restarted",
                message: "reader went away".into(),
            }),
            Err(_) => {
                // The child's own watchdog should have interrupted it; it is wedged: kill it.
                let pid = self.pid.load(Ordering::Relaxed);
                if pid != 0 {
                    unsafe {
                        libc::kill(pid as libc::pid_t, libc::SIGKILL);
                    }
                }
                Err(QueryError {
                    kind: "timeout",
                    message: "the reader did not answer in time and was restarted".into(),
                })
            }
        }
    }
}

#[cfg(test)]
mod s3_tests {
    use super::*;

    #[test]
    fn the_window_bound_is_the_rss_headroom_under_the_soft_cap() {
        let cap = 192u64 << 20;
        let pct = crate::objstore::S3Cfg::default().window_soft_pct;
        assert_eq!(pct, 85);
        let soft = cap * pct / 100;
        // far under: the headroom plus what the window holds
        assert_eq!(
            window_limit(100 << 20, 10 << 20, cap, pct),
            (10 << 20) + soft - (100 << 20)
        );
        // at or past the soft cap: no prefetch beyond what is held (the scheduler still admits
        // one range into an empty window, and every range a reader waits on)
        assert_eq!(window_limit(soft, 3 << 20, cap, pct), 3 << 20);
        assert_eq!(window_limit(cap, 0, cap, pct), 0);
    }

    #[test]
    fn s3_totals_ride_on_replies_and_survive_child_restarts() {
        let (id, t, v) = parse_reply_s3(r#"{"id":7,"gen":3,"c":1,"s3":[10,2000,1,4],"ok":{"a":1}}"#.into()).unwrap();
        assert_eq!((id, t), (7, Some([10, 2000, 1, 4])));
        let r = v.unwrap();
        assert_eq!((r.body.as_str(), r.gen, r.cacheable), (r#"{"a":1}"#, 3, true));
        let (_, t, v) =
            parse_reply_s3(r#"{"id":8,"s3":[12,2100,1,4],"err":{"type":"timeout","message":"m"}}"#.into()).unwrap();
        assert_eq!(t, Some([12, 2100, 1, 4]));
        assert_eq!(v.unwrap_err().kind, "timeout");
        // a local store: no counters
        let (_, t, _) = parse_reply_s3(r#"{"id":9,"gen":0,"c":0,"ok":[]}"#.into()).unwrap();
        assert_eq!(t, None);

        let s = S3Totals::default();
        assert_eq!(s.get(), None);
        s.update([12, 2100, 1, 4]);
        s.update([10, 2000, 1, 4]); // an older snapshot written late
        assert_eq!(s.get(), Some([12, 2100, 1, 4]));
        s.child_gone();
        s.update([3, 300, 0, 2]); // the next child starts from zero
        assert_eq!(s.get(), Some([15, 2400, 1, 6]));
    }
}

#[cfg(test)]
mod recycle_tests {
    use super::*;

    #[test]
    fn idle_recycle_does_not_count_the_row_cache() {
        let cap = 192u64 << 20;
        let line = format!("{{\"id\":7,\"gen\":3,\"c\":1,\"rc\":{},\"ok\":[]}}", 20u64 << 20);
        let (id, r) = parse_reply(line).unwrap();
        assert_eq!(id, 7);
        assert!(r.is_ok());
        let cb = READER_CACHE_BYTES.load(Ordering::Relaxed);
        assert_eq!(cb, 20 << 20);
        // 115 MiB with a 20 MiB cache: 95 MiB left, under 55% of 192 (105.6): kept
        assert!(!recycle_due(115 << 20, cb, cap));
        // 130 MiB with the same cache: 110 left, still above: recycled
        assert!(recycle_due(130 << 20, cb, cap));
        // without the cache 115 MiB is above: recycled
        assert!(recycle_due(115 << 20, 0, cap));
        // a reply from an older reader without "rc" still parses
        assert!(parse_reply("{\"id\":8,\"gen\":3,\"c\":0,\"ok\":[]}".into()).is_some());
    }
}
