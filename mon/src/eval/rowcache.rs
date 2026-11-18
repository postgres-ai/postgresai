//! Evaluated-row cache in the reader: the per-step rows a function gives a selection of series
//! on one whole grid (prom_topk's num, den and expression terms), so dashboard panels that
//! evaluate the same metric on the same grid (D2: per second and per call; eleven panels over
//! one `calls` denominator) compute it once, with the L1 response cache off and from cold.
//!
//! Key: everything the rows depend on: function, grid (including the metric interval), the
//! dictionary version, exact series ids, catalog generation and data epoch, newest sample and the
//! closed-day lag (which days are served from rollups and which from raw samples), and the
//! rollup / raw switch. Single flight: a second caller for a key being computed waits for the
//! first and replays its rows. Bounded in bytes (`set_cap`, --row-cache-mb; 0 = off), least
//! recently used out first, an entry over half the cap is not kept. `clear` drops everything
//! (the reader clears it before a query when its RSS is high, and before an idle recycle).
use super::kernels::{Func, Grid};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

static CAP: AtomicUsize = AtomicUsize::new(0);
pub static HITS: AtomicU64 = AtomicU64::new(0);
pub static MISSES: AtomicU64 = AtomicU64::new(0);
pub static WAITS: AtomicU64 = AtomicU64::new(0);
pub static EVICTIONS: AtomicU64 = AtomicU64::new(0);
/// Bytes held (mirrors the map, readable without the lock: the reader reports it per reply).
pub static BYTES: AtomicU64 = AtomicU64::new(0);

pub fn set_cap(bytes: usize) {
    CAP.store(bytes, Ordering::Relaxed);
    if bytes == 0 {
        clear();
    }
}

pub fn enabled() -> bool {
    CAP.load(Ordering::Relaxed) > 0
}

/// Rows of one evaluation, sparse: per series its id and its points (step index, value).
#[derive(Default)]
pub struct Rows {
    np: usize,
    ids: Vec<u64>,
    /// end offset of each series' points in `ix` / `vs`
    ends: Vec<u32>,
    ix: Vec<u32>,
    vs: Vec<f64>,
}

impl Rows {
    pub fn new(np: usize) -> Self {
        Rows {
            np,
            ..Default::default()
        }
    }
    pub fn push(&mut self, id: u64, row: &[f64]) {
        for (j, y) in row.iter().enumerate() {
            if !y.is_nan() {
                self.ix.push(j as u32);
                self.vs.push(*y);
            }
        }
        self.ids.push(id);
        self.ends.push(self.ix.len() as u32);
    }
    pub fn bytes(&self) -> usize {
        self.ids.len() * 12 + self.ix.len() * 12 + 64
    }
    /// Calls `each(id, row)` in the order the rows were pushed, NaN where absent.
    pub fn replay(&self, mut each: impl FnMut(u64, &[f64])) {
        let mut row = vec![f64::NAN; self.np];
        let mut a = 0usize;
        for (i, &id) in self.ids.iter().enumerate() {
            let b = self.ends[i] as usize;
            row.iter_mut().for_each(|x| *x = f64::NAN);
            for p in a..b {
                row[self.ix[p] as usize] = self.vs[p];
            }
            each(id, &row);
            a = b;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Key(u64, u64);

/// The cache key of evaluating `f` over `ids` on `g` in `ctx` (see the module doc).
pub fn key(ctx: &super::EvalCtx, f: Func, g: &Grid, ids: &[u64]) -> Key {
    let mut a = std::collections::hash_map::DefaultHasher::new();
    let mut b = std::collections::hash_map::DefaultHasher::new();
    for h in [&mut a, &mut b] {
        format!("{f:?}").hash(h);
        (g.start, g.end, g.step, g.window, g.interval, g.phi.to_bits()).hash(h);
        ctx.dict_version.hash(h);
        (ctx.generation, ctx.data_epoch, ctx.hwm_ms, ctx.closed_lag_ms).hash(h);
        std::env::var_os("PGAI_NO_ROLLUP").is_some().hash(h);
        ids.len().hash(h);
    }
    // two independent 64-bit digests of the id list (a collision would need both)
    0xA5u8.hash(&mut b);
    ids.hash(&mut a);
    for id in ids.iter().rev() {
        id.hash(&mut b);
    }
    Key(a.finish(), b.finish())
}

enum Slot {
    Ready(Arc<Rows>, u64),
    Flight(Arc<(Mutex<Option<Option<Arc<Rows>>>>, Condvar)>),
}

#[derive(Default)]
struct State {
    map: HashMap<Key, Slot>,
    bytes: usize,
    tick: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut g = STATE.lock().unwrap();
    f(g.get_or_insert_with(Default::default))
}

/// Drop every ready entry (flights in progress finish and are not kept if the cap is 0).
pub fn clear() {
    with(|s| {
        s.map.retain(|_, v| matches!(v, Slot::Flight(_)));
        s.bytes = 0;
    });
    BYTES.store(0, Ordering::Relaxed);
}

pub fn stats() -> serde_json::Value {
    let (entries, bytes) = with(|s| (s.map.values().filter(|v| matches!(v, Slot::Ready(..))).count(), s.bytes));
    serde_json::json!({
        "cap_bytes": CAP.load(Ordering::Relaxed),
        "entries": entries,
        "bytes": bytes,
        "hits": HITS.load(Ordering::Relaxed),
        "misses": MISSES.load(Ordering::Relaxed),
        "waits": WAITS.load(Ordering::Relaxed),
        "evictions": EVICTIONS.load(Ordering::Relaxed),
    })
}

/// Rows for `k`: replayed from the cache (or from a concurrent caller's flight), else
/// computed by `compute(sink)` (which calls `sink(id, row)` per series) and kept.
/// `each` sees the rows either way, in the same order.
pub fn get_or_compute(
    k: Key,
    np: usize,
    each: &mut dyn FnMut(u64, &[f64]),
    compute: impl FnOnce(&mut dyn FnMut(u64, &[f64])) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    enum Got {
        Hit(Arc<Rows>),
        Wait(Arc<(Mutex<Option<Option<Arc<Rows>>>>, Condvar)>),
        Lead(Arc<(Mutex<Option<Option<Arc<Rows>>>>, Condvar)>),
    }
    let got = with(|s| {
        s.tick += 1;
        let t = s.tick;
        match s.map.get_mut(&k) {
            Some(Slot::Ready(r, used)) => {
                *used = t;
                Got::Hit(r.clone())
            }
            Some(Slot::Flight(f)) => Got::Wait(f.clone()),
            None => {
                let f = Arc::new((Mutex::new(None), Condvar::new()));
                s.map.insert(k, Slot::Flight(f.clone()));
                Got::Lead(f)
            }
        }
    });
    let flight = match got {
        Got::Hit(r) => {
            HITS.fetch_add(1, Ordering::Relaxed);
            r.replay(each);
            return Ok(());
        }
        Got::Wait(f) => {
            WAITS.fetch_add(1, Ordering::Relaxed);
            let (m, cv) = &*f;
            let mut g = m.lock().unwrap();
            while g.is_none() {
                g = cv.wait(g).unwrap();
            }
            if let Some(Some(r)) = g.as_ref() {
                HITS.fetch_add(1, Ordering::Relaxed);
                r.replay(each);
                return Ok(());
            }
            // the leader failed or kept nothing: compute without the cache
            drop(g);
            MISSES.fetch_add(1, Ordering::Relaxed);
            return compute(each);
        }
        Got::Lead(f) => f,
    };
    MISSES.fetch_add(1, Ordering::Relaxed);
    let cap = CAP.load(Ordering::Relaxed);
    let mut rows = Some(Rows::new(np));
    let res = compute(&mut |id, row| {
        if let Some(r) = rows.as_mut() {
            r.push(id, row);
            if r.bytes() > cap / 2 {
                rows = None; // too big to keep: stop collecting
            }
        }
        each(id, row);
    });
    let keep = match (&res, rows) {
        (Ok(()), Some(r)) if CAP.load(Ordering::Relaxed) > 0 => Some(Arc::new(r)),
        _ => None,
    };
    with(|s| {
        s.map.remove(&k);
        if let Some(r) = &keep {
            let b = r.bytes();
            s.bytes += b;
            s.tick += 1;
            s.map.insert(k, Slot::Ready(r.clone(), s.tick));
            let cap = CAP.load(Ordering::Relaxed);
            while s.bytes > cap {
                let old = s
                    .map
                    .iter()
                    .filter_map(|(k, v)| match v {
                        Slot::Ready(r, t) => Some((*t, *k, r.bytes())),
                        _ => None,
                    })
                    .min();
                let Some((_, ok, ob)) = old else { break };
                s.map.remove(&ok);
                s.bytes -= ob;
                EVICTIONS.fetch_add(1, Ordering::Relaxed);
            }
        }
        BYTES.store(s.bytes as u64, Ordering::Relaxed);
    });
    let (m, cv) = &*flight;
    *m.lock().unwrap() = Some(keep);
    cv.notify_all();
    res
}
