//! L1 result cache in the parent process.
//!
//! Grafana window parameters are expanded by the HTTP handler before keying and execution.
//! Whole response bodies, keyed by the SQL text (trimmed, trailing `;` removed) plus the
//! response format, dictionary version and a hash of the views/macros/evaluator version.
//! Bounded evaluator results survive catalog changes proven outside their fetch window.
//! Identical requests that arrive while one runs wait for it (coalescing) instead of running again. A hit never reaches admission or the reader child.
//!
//! - Exact: a hit returns bytes from a run whose dependencies still hold. Only `elapsed_ms` in
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
    dict: u64,
}

struct Entry {
    zbody: Box<[u8]>,
    raw_len: usize,
    gen: u64,
    range: Option<crate::readset::ReadRange>,
    dict_version: u64,
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
    snapshot: Option<Snapshot>,
    map: HashMap<Key, Slot>,
    lru: BTreeMap<u64, Key>,
    bytes: usize,
    tick: u64,
}

/// The parent's catalog and dictionary at request admission. Tests may supply just a generation.
#[derive(Clone, Default)]
pub struct Snapshot {
    pub gen: u64,
    pub dict_version: u64,
    files: Option<Arc<Vec<crate::catalog::FileEntry>>>,
    layout: String,
    data_epoch: u64,
    closed_lag_ms: i64,
}

impl From<u64> for Snapshot {
    fn from(gen: u64) -> Self {
        Self {
            gen,
            ..Default::default()
        }
    }
}

fn reads(f: &crate::catalog::FileEntry, r: crate::readset::ReadRange) -> bool {
    match f.kind.as_str() {
        "series" => f.day >= crate::model::day_of(r.start) && f.day <= crate::model::day_of(r.end),
        crate::store::SERIES_MONTH => {
            let (a, b) = crate::store::month_of(f.day);
            a <= crate::model::day_of(r.end) && b > crate::model::day_of(r.start)
        }
        crate::registry::KIND => true,
        "samples" | crate::eval::rollup::KIND | crate::eval::rollup::KIND_1D | crate::eval::rollup::KIND_1HM => {
            f.min_ts <= r.end && f.max_ts >= r.start
        }
        _ => false,
    }
}

impl Snapshot {
    fn changes(&self, old: &Self) -> Option<(Vec<crate::catalog::FileEntry>, Vec<crate::catalog::FileEntry>)> {
        if self.layout != old.layout || self.closed_lag_ms != old.closed_lag_ms {
            return None;
        }
        let (new, old) = (self.files.as_ref()?, old.files.as_ref()?);
        let by_path: HashMap<_, _> = old.iter().map(|f| (&f.path, f)).collect();
        let added = new
            .iter()
            .filter(|f| by_path.get(&f.path).is_none_or(|p| *p != *f))
            .cloned()
            .collect();
        let by_path: HashMap<_, _> = new.iter().map(|f| (&f.path, f)).collect();
        let removed = old
            .iter()
            .filter(|f| by_path.get(&f.path).is_none_or(|p| *p != *f))
            .cloned()
            .collect();
        Some((added, removed))
    }
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
            dict: 0,
        }
    }

    /// Reuse the file snapshot within a generation rather than copying it for every hit.
    pub fn snapshot(&self, cat: &crate::catalog::Catalog, dict_version: u64) -> Snapshot {
        if !self.enabled() {
            return Snapshot {
                gen: cat.generation,
                dict_version,
                ..Default::default()
            };
        }
        let st = self.st.lock().unwrap();
        if let Some(s) = st.snapshot.as_ref().filter(|s| s.gen == cat.generation) {
            return Snapshot {
                dict_version,
                ..s.clone()
            };
        }
        Snapshot {
            gen: cat.generation,
            dict_version,
            files: Some(Arc::new(cat.files.clone())),
            layout: cat.samples_layout.clone(),
            data_epoch: cat.data_epoch,
            closed_lag_ms: cat.closed_lag_ms,
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
    pub async fn get_or_run<F, Fut>(
        self: &Arc<Self>,
        sql: &str,
        fmt: u8,
        snapshot: impl Into<Snapshot>,
        run: F,
    ) -> (Res, Served)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Reply, QueryError>> + Send + 'static,
    {
        if !self.enabled() {
            self.stats.bypass.fetch_add(1, Ordering::Relaxed);
            return (detached(run()).await.map(Arc::new), Served::Bypass);
        }
        let snapshot = snapshot.into();
        let gen_now = snapshot.gen;
        let mut key = self.key(&normalize(sql), fmt);
        key.dict = snapshot.dict_version;
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
                    let changes = st.snapshot.as_ref().and_then(|old| snapshot.changes(old));
                    if let Some((added, removed)) = changes {
                        let old_epoch = st.snapshot.as_ref().unwrap().data_epoch;
                        // An unexplained data epoch change cannot be proved harmless.
                        let known = old_epoch == snapshot.data_epoch || !added.is_empty() || !removed.is_empty();
                        let keep: Vec<_> = st
                            .map
                            .iter()
                            .filter_map(|(k, slot)| {
                                let Slot::Ready(e, _) = slot else { return None };
                                let r = e.range?;
                                (known
                                    && k.dict == snapshot.dict_version
                                    && added.iter().all(|f| {
                                        !matches!(
                                            f.kind.as_str(),
                                            crate::eval::rollup::KIND
                                                | crate::eval::rollup::KIND_1D
                                                | crate::eval::rollup::KIND_1HM
                                        ) && f.min_ts > r.end
                                            && (!matches!(
                                                f.kind.as_str(),
                                                "series" | crate::store::SERIES_MONTH | crate::registry::KIND
                                            ) || !reads(f, r))
                                    })
                                    && removed.iter().all(|f| {
                                        !matches!(
                                            f.kind.as_str(),
                                            "samples"
                                                | "series"
                                                | crate::store::SERIES_MONTH
                                                | crate::registry::KIND
                                                | crate::eval::rollup::KIND
                                                | crate::eval::rollup::KIND_1D
                                                | crate::eval::rollup::KIND_1HM
                                        )
                                    }))
                                .then_some(*k)
                            })
                            .collect();
                        let mut kept = HashMap::new();
                        for k in keep {
                            kept.insert(k, st.map.remove(&k).unwrap());
                        }
                        st.clear();
                        for (k, slot) in kept {
                            if let Slot::Ready(e, t) = &slot {
                                st.bytes += e.zbody.len() + ENTRY_OVERHEAD;
                                st.lru.insert(*t, k);
                            }
                            st.map.insert(k, slot);
                        }
                    } else {
                        st.clear();
                    }
                    st.gen = gen_now;
                    st.snapshot = Some(snapshot.clone());
                }
                if st.snapshot.is_none() && gen_now == st.gen {
                    st.snapshot = Some(snapshot.clone());
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
                        .map(|z| (z, r.body.len(), r.gen, r.range, r.dict_version))
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
            Some((z, raw_len, gen, range, dict_version))
                if gen == st.gen && key.dict == dict_version && z.len() <= self.max_entry_bytes =>
            {
                st.tick += 1;
                let t = st.tick;
                let e = Arc::new(Entry {
                    raw_len,
                    gen,
                    range,
                    dict_version,
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
            Some((_, _, gen, _, dict_version)) => {
                if gen != st.gen || key.dict != dict_version {
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
                range: e.range,
                dict_version: e.dict_version,
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
