//! L1 result cache in the parent process.
//!
//! Whole response bodies, keyed by the SQL text (trimmed, trailing `;` removed) plus the
//! response format and a hash of the views/macros/evaluator version, valid for one catalog
//! generation. Identical requests that arrive while one is running wait for it (coalescing)
//! instead of running again. A hit never reaches admission or the reader child.
//!
//! - Exact: a hit returns the bytes a run at the same generation produced. Only `elapsed_ms` in
//!   the body is that of the original run. Errors are never stored, and neither is a reply the
//!   reader marks not cacheable (volatile functions, table functions outside the allowlist,
//!   file scans: decided on DuckDB's parse tree in the child, not on the SQL text).
//! - Each run is a spawned task (a flight) that owns its admission permit: a client that goes
//!   away neither frees the slot early nor makes a follower run the query again.
//! - Followers that get a transient error re-enter once, so their retries coalesce into one new
//!   flight instead of each running the query again.
//! - Bodies are kept zstd level 1 compressed; each entry is charged its compressed size plus
//!   `ENTRY_OVERHEAD`; least-recently-used out above `cap_bytes` (an ordered tick index, O(log n));
//!   an entry over `max_entry_bytes` (compressed) is not stored.
use crate::query::{QueryError, Reply};
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use xxhash_rust::xxh3::{xxh3_128, xxh3_64};

/// Bump on any change to the evaluator or result format that changes response bytes.
pub const EVAL_SEMVER: u32 = 1;
/// Bytes charged per entry besides its compressed body: key, map slot, LRU index node, `Entry`
/// and `Arc` headers, allocator slack. Measured: 150k tiny entries cost about 430 B each in RSS
/// with ~110 B bodies.
pub const ENTRY_OVERHEAD: usize = 320;
/// Bodies larger than this are decompressed on a blocking thread, not the async worker.
const INLINE_DECOMPRESS: usize = 256 << 10;

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub struct Key {
    text: u128,
    len: u32,
    fmt: u8,
    views: u64,
}

struct Entry {
    zbody: Box<[u8]>,
    raw_len: usize,
    gen: u64,
}

type Res = Result<Arc<Reply>, QueryError>;

struct Flight {
    rx: tokio::sync::watch::Receiver<Option<Res>>,
}

enum Slot {
    Ready(Arc<Entry>, u64),
    InFlight(Arc<Flight>),
}

#[derive(Default)]
struct L1State {
    gen: u64,
    map: HashMap<Key, Slot>,
    lru: BTreeMap<u64, Key>,
    bytes: usize,
    tick: u64,
}

impl L1State {
    fn touch(&mut self, key: &Key) {
        self.tick += 1;
        let t = self.tick;
        if let Some(Slot::Ready(_, old)) = self.map.get_mut(key) {
            self.lru.remove(old);
            *old = t;
            self.lru.insert(t, *key);
        }
    }
    fn clear(&mut self) {
        // flights keep running for their waiters; they find their slot gone and don't store
        self.map.clear();
        self.lru.clear();
        self.bytes = 0;
    }
}

#[derive(Default)]
pub struct L1Stats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub coalesced: AtomicU64,
    pub bypass: AtomicU64,
    pub stores: AtomicU64,
    pub evictions: AtomicU64,
    pub follower_retries: AtomicU64,
    pub not_stored_gen: AtomicU64,
    pub not_stored_size: AtomicU64,
    pub not_stored_volatile: AtomicU64,
    pub stale_gen_bypass: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Served {
    Hit,
    Miss,
    Coalesced,
    Bypass,
}

impl Served {
    pub fn as_str(self) -> &'static str {
        match self {
            Served::Hit => "hit",
            Served::Miss => "miss",
            Served::Coalesced => "coalesced",
            Served::Bypass => "bypass",
        }
    }
}

pub struct QueryCache {
    cap_bytes: usize,
    max_entry_bytes: usize,
    views_hash: u64,
    st: Mutex<L1State>,
    pub stats: L1Stats,
}

/// Errors a follower must not take from the leader: they may be about the leader's run (a
/// reader kill fails every query in flight), not about this SQL.
fn transient(e: &QueryError) -> bool {
    matches!(
        e.kind,
        "out_of_memory" | "timeout" | "busy" | "reader_restarted" | "internal" | "execution_error"
    )
}

/// Run `fut` as its own task, so it completes (and releases what it holds) even if the caller
/// goes away. A panic comes back as an `internal` error.
async fn detached<Fut>(fut: Fut) -> Result<Reply, QueryError>
where
    Fut: Future<Output = Result<Reply, QueryError>> + Send + 'static,
{
    tokio::spawn(fut).await.unwrap_or_else(|e| {
        Err(QueryError {
            kind: "internal",
            message: format!("query task failed: {e}"),
        })
    })
}

impl QueryCache {
    /// `cap_bytes` 0 disables L1 (every request is `Bypass`). `views_text` is everything that
    /// defines what SQL means (views.sql + macros + eval.sql).
    pub fn new(cap_bytes: usize, max_entry_bytes: usize, views_text: &str) -> Self {
        let mut v = views_text.as_bytes().to_vec();
        v.extend_from_slice(&EVAL_SEMVER.to_le_bytes());
        Self {
            cap_bytes,
            max_entry_bytes,
            views_hash: xxh3_64(&v),
            st: Default::default(),
            stats: Default::default(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.cap_bytes > 0
    }

    pub fn key(&self, normalized: &str, fmt: u8) -> Key {
        Key {
            text: xxh3_128(normalized.as_bytes()),
            len: normalized.len() as u32,
            fmt,
            views: self.views_hash,
        }
    }

    pub fn stats_json(&self) -> serde_json::Value {
        let s = &self.stats;
        let st = self.st.lock().unwrap();
        let ready = st.lru.len();
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        serde_json::json!({
            "enabled": self.enabled(),
            "cap_bytes": self.cap_bytes,
            "bytes": st.bytes,
            "entries": ready,
            "in_flight": st.map.len() - ready,
            "generation": st.gen,
            "hits": l(&s.hits),
            "misses": l(&s.misses),
            "coalesced": l(&s.coalesced),
            "bypass": l(&s.bypass),
            "stores": l(&s.stores),
            "evictions": l(&s.evictions),
            "follower_retries": l(&s.follower_retries),
            "not_stored_generation": l(&s.not_stored_gen),
            "not_stored_size": l(&s.not_stored_size),
            "not_stored_volatile": l(&s.not_stored_volatile),
            "stale_generation_bypass": l(&s.stale_gen_bypass),
        })
    }

    /// Serve `sql` from the cache, from a flight already running, or by starting one that runs
    /// `run` as a spawned task. `run` may be called again by a follower after a transient error.
    pub async fn get_or_run<F, Fut>(self: &Arc<Self>, sql: &str, fmt: u8, gen_now: u64, run: F) -> (Res, Served)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Reply, QueryError>> + Send + 'static,
    {
        if !self.enabled() {
            self.stats.bypass.fetch_add(1, Ordering::Relaxed);
            return (detached(run()).await.map(Arc::new), Served::Bypass);
        }
        let key = self.key(&normalize(sql), fmt);
        let run = Arc::new(run);
        for attempt in 0..2 {
            enum Next {
                Stale,
                Hit(Arc<Entry>),
                Wait(tokio::sync::watch::Receiver<Option<Res>>, bool),
            }
            let next = {
                let mut st = self.st.lock().unwrap();
                if gen_now > st.gen {
                    st.clear();
                    st.gen = gen_now;
                }
                if gen_now < st.gen {
                    Next::Stale
                } else {
                    match st.map.get(&key) {
                        Some(Slot::Ready(e, _)) => {
                            let e = e.clone();
                            st.touch(&key);
                            Next::Hit(e)
                        }
                        Some(Slot::InFlight(f)) => Next::Wait(f.rx.clone(), false),
                        None => {
                            let (tx, rx) = tokio::sync::watch::channel(None);
                            let f = Arc::new(Flight { rx: rx.clone() });
                            st.map.insert(key, Slot::InFlight(f.clone()));
                            let (me, run) = (self.clone(), run.clone());
                            tokio::spawn(async move {
                                // a panic in run() drops tx: waiters see a closed channel, and
                                // the guard frees the slot
                                let guard = SlotGuard(me.clone(), key, f.clone());
                                let res = run().await.map(Arc::new);
                                me.finish(key, &f, &res).await;
                                // finish stored or removed the slot: dropping the guard is a
                                // no-op then, and it must be dropped, not forgotten (forgetting
                                // would keep the Flight, its channel and the reply alive)
                                drop(guard);
                                let _ = tx.send(Some(res));
                            });
                            Next::Wait(rx, true)
                        }
                    }
                }
            };
            let (rx, led) = match next {
                Next::Stale => {
                    // loaded the generation just before a flush: run, don't touch the cache
                    self.stats.stale_gen_bypass.fetch_add(1, Ordering::Relaxed);
                    self.stats.bypass.fetch_add(1, Ordering::Relaxed);
                    return (detached(run()).await.map(Arc::new), Served::Bypass);
                }
                Next::Hit(e) => {
                    self.stats.hits.fetch_add(1, Ordering::Relaxed);
                    return (decompress(e).await, Served::Hit);
                }
                Next::Wait(rx, led) => (rx, led),
            };
            let res = wait(rx).await;
            if led {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                return (res, Served::Miss);
            }
            self.stats.coalesced.fetch_add(1, Ordering::Relaxed);
            match &res {
                Err(e) if transient(e) && attempt == 0 => {
                    self.stats.follower_retries.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                _ => return (res, Served::Coalesced),
            }
        }
        unreachable!()
    }

    fn remove_flight(&self, key: Key, flight: &Arc<Flight>) {
        let mut st = self.st.lock().unwrap();
        if matches!(st.map.get(&key), Some(Slot::InFlight(f)) if Arc::ptr_eq(f, flight)) {
            st.map.remove(&key);
        }
    }

    /// The flight stores its result (if cacheable and still current) or clears its slot.
    async fn finish(&self, key: Key, flight: &Arc<Flight>, res: &Res) {
        let z = match res {
            Ok(r) if r.cacheable => {
                let r = r.clone();
                tokio::task::spawn_blocking(move || {
                    zstd::bulk::compress(r.body.as_bytes(), 1)
                        .ok()
                        .map(|z| (z, r.body.len(), r.gen))
                })
                .await
                .ok()
                .flatten()
            }
            Ok(_) => {
                self.stats.not_stored_volatile.fetch_add(1, Ordering::Relaxed);
                None
            }
            Err(_) => None,
        };
        let mut st = self.st.lock().unwrap();
        let ours = matches!(st.map.get(&key), Some(Slot::InFlight(f)) if Arc::ptr_eq(f, flight));
        if !ours {
            return;
        }
        match z {
            Some((z, raw_len, gen)) if gen == st.gen && z.len() <= self.max_entry_bytes => {
                st.tick += 1;
                let t = st.tick;
                let e = Arc::new(Entry {
                    raw_len,
                    gen,
                    zbody: z.into_boxed_slice(),
                });
                st.bytes += e.zbody.len() + ENTRY_OVERHEAD;
                st.map.insert(key, Slot::Ready(e, t));
                st.lru.insert(t, key);
                self.stats.stores.fetch_add(1, Ordering::Relaxed);
                while st.bytes > self.cap_bytes {
                    let Some((_, k)) = st.lru.pop_first() else {
                        break;
                    };
                    if let Some(Slot::Ready(e, _)) = st.map.remove(&k) {
                        st.bytes -= e.zbody.len() + ENTRY_OVERHEAD;
                        self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            Some((_, _, gen)) => {
                if gen != st.gen {
                    self.stats.not_stored_gen.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.stats.not_stored_size.fetch_add(1, Ordering::Relaxed);
                }
                st.map.remove(&key);
            }
            None => {
                st.map.remove(&key);
            }
        }
    }
}

/// Frees a flight's slot if its task ends without finishing (panic).
struct SlotGuard(Arc<QueryCache>, Key, Arc<Flight>);
impl Drop for SlotGuard {
    fn drop(&mut self) {
        self.0.remove_flight(self.1, &self.2);
    }
}

async fn wait(mut rx: tokio::sync::watch::Receiver<Option<Res>>) -> Res {
    match rx.wait_for(|v| v.is_some()).await {
        Ok(v) => v.clone().unwrap(),
        Err(_) => Err(QueryError {
            kind: "internal",
            message: "query task failed".into(),
        }),
    }
}

async fn decompress(e: Arc<Entry>) -> Res {
    if e.raw_len > INLINE_DECOMPRESS {
        tokio::task::spawn_blocking(move || decompress_now(&e))
            .await
            .unwrap_or_else(|_| Err(corrupt()))
    } else {
        decompress_now(&e)
    }
}

fn corrupt() -> QueryError {
    QueryError {
        kind: "internal",
        message: "result cache: corrupt entry".into(),
    }
}

fn decompress_now(e: &Entry) -> Res {
    zstd::bulk::decompress(&e.zbody, e.raw_len)
        .ok()
        .and_then(|b| String::from_utf8(b).ok())
        .map(|body| {
            Arc::new(Reply {
                body,
                gen: e.gen,
                cacheable: true,
            })
        })
        .ok_or_else(corrupt)
}

/// The cache key text: trimmed, trailing `;` and whitespace removed. Nothing else is rewritten:
/// folding calls such as `prom_align_down` would give SQL texts with different results one key.
/// Whether a result may be stored is decided by the reader on the parse tree (`Reply.cacheable`).
pub fn normalize(sql: &str) -> String {
    sql.trim()
        .trim_end_matches(|c: char| c == ';' || c.is_whitespace())
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_only_trims() {
        assert_eq!(normalize("SELECT 1 ;\n"), "SELECT 1");
        for q in [
            "SELECT prom_align_down(-5, 3) AS v",
            "SELECT prom_align_down(1794206627 * 1000, 30000) AS e",
            "SELECT prom_align_down(7, 5)",
        ] {
            assert_eq!(normalize(q), q, "no folding");
        }
        assert_ne!(
            normalize("SELECT prom_align_down(-5, 3) AS v"),
            normalize("SELECT prom_align_down(-6, 3) AS v")
        );
    }
}
