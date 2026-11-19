//! Query admission in the parent: which waiting query gets the next reader slot.
//!
//! - `slots` queries run at once (default one per core, at most 4: `HttpCfg::resolve`).
//! - **Short queries bypass heavy ones.** A query is short when its expected service time
//!   (`Cost`) is under `short_ms`. With at least three slots, one is kept for short queries.
//!   Smaller hosts let heavy queries use every slot; waiting short queries get the next slot.
//! - **Order among waiters** is arrival time plus expected service time: shortest first, with
//!   ageing. On one/two-slot hosts, short-query priority overrides this order across classes.
//! - **Memory gate.** A heavy query starts only while the reader's RSS is under `mem_gate`
//!   bytes, or when no other heavy query runs. All slots share one reader process and one
//!   RSS cap (sqlproc.rs), so this keeps a second and third heavy query from starting on top of
//!   a reader that is already near the cap; the cap stays the hard bound.
//! - **Big shapes run alone among heavy queries.** A shape whose run ended in a reader kill
//!   (every query in flight in the killed reader is marked, since the culprit is unknown) is
//!   "big": it starts only when no other heavy query runs, and no heavy query starts beside
//!   it. Short queries still run. The killed queries are retried once under this rule
//!   (query.rs), so a kill costs latency instead of an error after the first time. A big
//!   shape whose solo run peaks under 70 % of the cap was a bystander and is unmarked.
//! - **Exclusive** (the DuckDB out-of-memory retry): waits until nothing runs and holds every slot.
//!
//! Expected service time comes from a small history keyed by the query's shape (`shape_key`:
//! the SQL with epoch literals removed, plus the magnitude of the time span they cover), so a
//! panel whose window slides keeps its history. Unknown shapes are guessed from the text.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use xxhash_rust::xxh3::xxh3_64;

pub type MemProbe = Arc<dyn Fn() -> Option<u64> + Send + Sync>;

pub struct Admission {
    pub slots: usize,
    pub short_ms: f64,
    pub mem_gate: u64,
    mem: Option<MemProbe>,
    st: Mutex<State>,
    t0: Instant,
}

#[derive(Default)]
struct State {
    running: usize,
    running_heavy: usize,
    exclusive: bool,
    waiters: Vec<Waiter>,
    seq: u64,
    hist: HashMap<u64, f64>,
    big: std::collections::HashSet<u64>,
    running_big: usize,
    /// Counters for /health.
    started_short: u64,
    started_heavy: u64,
    mem_deferred: u64,
    marked_big: u64,
    unmarked_big: u64,
}

struct Waiter {
    prio: f64, // ms since t0: arrival + expected service
    seq: u64,
    heavy: bool,
    big: bool,
    exclusive: bool,
    tx: tokio::sync::oneshot::Sender<Permit>,
}

/// A running slot; dropping it frees the slot and admits the next waiter.
pub struct Permit {
    adm: Arc<Admission>,
    heavy: bool,
    big: bool,
    exclusive: bool,
    armed: bool,
}

impl Drop for Permit {
    fn drop(&mut self) {
        if self.armed {
            self.adm.release(self.heavy, self.big, self.exclusive);
        }
    }
}

/// What the caller learns about a query before running it.
#[derive(Clone, Copy, Debug)]
pub struct Cost {
    pub key: u64,
    pub expected_ms: f64,
    pub heavy: bool,
    /// Ran alone among heavy queries (its shape was in flight when the reader was killed).
    pub big: bool,
}

impl Admission {
    pub fn new(slots: usize, short_ms: f64, mem_gate: u64, mem: Option<MemProbe>) -> Arc<Self> {
        Arc::new(Self {
            slots: slots.max(1),
            short_ms,
            mem_gate,
            mem,
            st: Default::default(),
            t0: Instant::now(),
        })
    }

    /// Keep a short lane only when it leaves at least two heavy slots.
    fn heavy_slots(&self) -> usize {
        if self.slots >= 3 {
            self.slots - 1
        } else {
            self.slots
        }
    }

    pub fn cost(&self, sql: &str) -> Cost {
        let key = shape_key(sql);
        let st = self.st.lock().unwrap();
        let expected_ms = st.hist.get(&key).copied().unwrap_or_else(|| guess_ms(sql));
        let big = st.big.contains(&key);
        Cost {
            key,
            expected_ms,
            heavy: big || expected_ms >= self.short_ms,
            big,
        }
    }

    /// Record a finished query's service time (exponentially weighted, recent runs count most).
    pub fn observe(&self, c: &Cost, ms: f64) {
        let mut st = self.st.lock().unwrap();
        if st.hist.len() >= 8192 && !st.hist.contains_key(&c.key) {
            st.hist.clear(); // crude bound; shapes are few in practice
        }
        let e = st.hist.entry(c.key).or_insert(ms);
        *e = 0.5 * *e + 0.5 * ms;
    }

    /// The query was in flight when the reader was killed for memory: run its shape alone
    /// among heavy queries from now on. Returns the cost to retry with.
    pub fn mark_big(&self, c: &Cost) -> Cost {
        let mut st = self.st.lock().unwrap();
        if st.big.len() >= 4096 {
            st.big.clear();
        }
        if st.big.insert(c.key) {
            st.marked_big += 1;
        }
        Cost {
            heavy: true,
            big: true,
            ..*c
        }
    }

    /// A big shape that ran alone among heavy queries and kept the reader under the gate was
    /// an innocent victim of someone else's kill: clear its mark.
    pub fn unmark_big(&self, c: &Cost) {
        let mut st = self.st.lock().unwrap();
        if st.big.remove(&c.key) {
            st.unmarked_big += 1;
        }
    }

    /// Wait for a slot. `None` when the wait timed out (the caller answers `busy`).
    pub async fn acquire(self: &Arc<Self>, c: &Cost, exclusive: bool, wait: std::time::Duration) -> Option<Permit> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut st = self.st.lock().unwrap();
            let now = self.t0.elapsed().as_secs_f64() * 1000.0;
            st.seq += 1;
            let seq = st.seq;
            st.waiters.push(Waiter {
                prio: now + c.expected_ms,
                seq,
                heavy: c.heavy,
                big: c.big,
                exclusive,
                tx,
            });
            self.dispatch(&mut st);
        }
        match tokio::time::timeout(wait, rx).await {
            Ok(Ok(p)) => Some(p),
            // Timed out: the waiter is still queued (dispatch skips it once the receiver is
            // gone) or a permit raced in and is dropped here, freeing its slot again.
            _ => None,
        }
    }

    pub fn waiting(&self) -> usize {
        self.st.lock().unwrap().waiters.len()
    }

    pub fn stats(&self) -> serde_json::Value {
        let st = self.st.lock().unwrap();
        serde_json::json!({
            "slots": self.slots,
            "heavy_slots": self.heavy_slots(),
            "short_ms": self.short_ms,
            "mem_gate_bytes": self.mem_gate,
            "running": st.running,
            "running_heavy": st.running_heavy,
            "waiting": st.waiters.len(),
            "started_short": st.started_short,
            "started_heavy": st.started_heavy,
            "mem_deferred": st.mem_deferred,
            "big_shapes": st.big.len(),
            "marked_big": st.marked_big,
            "unmarked_big": st.unmarked_big,
            "running_big": st.running_big,
            "shapes": st.hist.len(),
        })
    }

    fn release(self: &Arc<Self>, heavy: bool, big: bool, exclusive: bool) {
        let mut st = self.st.lock().unwrap();
        if exclusive {
            st.exclusive = false;
            st.running = 0;
            st.running_heavy = 0;
        } else {
            st.running -= 1;
            if heavy {
                st.running_heavy -= 1;
            }
            if big {
                st.running_big -= 1;
            }
        }
        self.dispatch(&mut st);
    }

    fn dispatch(self: &Arc<Self>, st: &mut State) {
        st.waiters.retain(|w| !w.tx.is_closed());
        loop {
            if st.exclusive || st.running >= self.slots {
                return;
            }
            st.waiters
                .sort_by(|a, b| a.prio.total_cmp(&b.prio).then(a.seq.cmp(&b.seq)));
            // An exclusive waiter blocks every later start until it runs.
            if let Some(i) = st.waiters.iter().position(|w| w.exclusive) {
                if st.running == 0 {
                    let w = st.waiters.remove(i);
                    st.exclusive = true;
                    st.running = self.slots;
                    let p = Permit {
                        adm: self.clone(),
                        heavy: true,
                        big: false,
                        exclusive: true,
                        armed: true,
                    };
                    if let Err(mut p) = w.tx.send(p) {
                        p.armed = false;
                        st.exclusive = false;
                        st.running = 0;
                        continue;
                    }
                }
                return;
            }
            let big_ok = st.running_heavy == 0;
            // A big shape at the head of the heavy queue that can't start yet stops new heavy
            // starts, so the running ones drain and it gets its turn. Priority is
            // arrival + expected time, so it reaches the head within its expected time.
            let big_first = st.waiters.iter().find(|w| w.heavy).is_some_and(|w| w.big && !big_ok);
            let heavy_ok = st.running_heavy < self.heavy_slots() && st.running_big == 0 && !big_first;
            let mem_ok = st.running_heavy == 0
                || self.mem_gate == 0
                || self.mem.as_ref().and_then(|m| m()).is_none_or(|r| r < self.mem_gate);
            // Without a reserved lane, a queued short query gets the next free slot,
            // even when a heavy waiter has aged ahead of it.
            let short_first = (self.slots < 3)
                .then(|| st.waiters.iter().position(|w| !w.heavy))
                .flatten();
            let pick = short_first.or_else(|| {
                st.waiters
                    .iter()
                    .position(|w| !w.heavy || if w.big { big_ok } else { heavy_ok && mem_ok })
            });
            let Some(i) = pick else {
                if heavy_ok && !mem_ok && !st.waiters.is_empty() {
                    st.mem_deferred += 1;
                }
                return;
            };
            let w = st.waiters.remove(i);
            st.running += 1;
            if w.heavy {
                st.running_heavy += 1;
                st.started_heavy += 1;
            } else {
                st.started_short += 1;
            }
            if w.big {
                st.running_big += 1;
            }
            let p = Permit {
                adm: self.clone(),
                heavy: w.heavy,
                big: w.big,
                exclusive: false,
                armed: true,
            };
            if let Err(mut p) = w.tx.send(p) {
                // the waiter gave up between the retain and now
                p.armed = false;
                st.running -= 1;
                if w.heavy {
                    st.running_heavy -= 1;
                }
                if w.big {
                    st.running_big -= 1;
                }
            }
        }
    }
}

/// The query's shape: its text with epoch-like integer literals (>= 1e9: seconds or ms since
/// 1970) removed, plus log2 of the span they cover. A sliding dashboard window keeps its key;
/// the same panel at 1 h and 14 d gets two keys.
pub fn shape_key(sql: &str) -> u64 {
    let b = sql.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let (mut lo, mut hi) = (u64::MAX, 0u64);
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_')) {
            let s = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            let n: u64 = sql[s..i].parse().unwrap_or(0);
            if i - s >= 10 && n >= 1_000_000_000 {
                // normalise seconds to ms so a mix of both still spans correctly
                let ms = if n < 100_000_000_000 { n * 1000 } else { n };
                lo = lo.min(ms);
                hi = hi.max(ms);
                out.push(b'#');
            } else {
                out.extend_from_slice(&b[s..i]);
            }
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    let span = if hi > lo { 64 - (hi - lo).leading_zeros() } else { 0 };
    out.extend_from_slice(format!("|span{span}").as_bytes());
    xxh3_64(&out)
}

/// First-time guess, before any history: queries that call the evaluator or read samples,
/// metrics or logs are assumed heavy; anything else (constants, catalog lookups) is short.
fn guess_ms(sql: &str) -> f64 {
    let s = sql.to_ascii_lowercase();
    let heavy = [
        "prom_",
        "samples",
        "metrics",
        "logs",
        "plan",
        "series",
        "query_texts",
        "objects",
    ];
    if heavy.iter().any(|w| s.contains(w)) {
        1000.0
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_key_ignores_window_position_but_not_size() {
        let q = |s: u64, e: u64| format!("SELECT * FROM prom_rate('m', [], {s}, {e}, 30000, 60000)");
        let h = 3_600_000;
        assert_eq!(
            shape_key(&q(1794200000000, 1794200000000 + h)),
            shape_key(&q(1794200030000, 1794200030000 + h))
        );
        assert_ne!(
            shape_key(&q(1794200000000, 1794200000000 + h)),
            shape_key(&q(1794200000000, 1794200000000 + 336 * h))
        );
        assert_ne!(shape_key("SELECT 1"), shape_key("SELECT 2"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn two_heavy_queries_run_and_waiting_short_gets_next_slot() {
        let a = Admission::new(2, 50.0, 0, None);
        let heavy = Cost {
            key: 1,
            expected_ms: 1000.0,
            heavy: true,
            big: false,
        };
        let short = Cost {
            key: 2,
            expected_ms: 1.0,
            heavy: false,
            big: false,
        };
        let w = std::time::Duration::from_secs(30);
        let h1 = a.acquire(&heavy, false, w).await.unwrap();
        let h2 = a.acquire(&heavy, false, w);
        tokio::pin!(h2);
        let std::task::Poll::Ready(Some(h2)) = futures_util::poll!(&mut h2) else {
            panic!("both heavy queries must run without waiting");
        };
        assert_eq!(a.stats()["running_heavy"], 2);
        let h3 = a.acquire(&heavy, false, w);
        let h4 = a.acquire(&heavy, false, w);
        tokio::pin!(h3, h4);
        assert!(futures_util::poll!(&mut h3).is_pending());
        assert!(futures_util::poll!(&mut h4).is_pending());
        // Even aged heavy waiters cannot take the next slot from a waiting short query.
        for waiter in &mut a.st.lock().unwrap().waiters {
            waiter.prio = -1.0;
        }
        let s1 = a.acquire(&short, false, w);
        tokio::pin!(s1);
        assert!(futures_util::poll!(&mut s1).is_pending());
        drop(h1);
        let std::task::Poll::Ready(Some(s1)) = futures_util::poll!(&mut s1) else {
            panic!("short query must get the next free slot");
        };
        assert!(futures_util::poll!(&mut h3).is_pending());
        assert!(futures_util::poll!(&mut h4).is_pending());
        drop(s1);
        let h3 = h3.await.unwrap();
        assert!(futures_util::poll!(&mut h4).is_pending());
        drop(h2);
        let _h4 = h4.await.unwrap();
        drop(h3);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn larger_hosts_keep_a_short_slot() {
        let a = Admission::new(3, 50.0, 0, None);
        let heavy = a.cost("SELECT * FROM samples");
        let short = a.cost("SELECT 1");
        let w = std::time::Duration::from_secs(30);
        let _h1 = a.acquire(&heavy, false, w).await.unwrap();
        let _h2 = a.acquire(&heavy, false, w).await.unwrap();
        let h3 = a.acquire(&heavy, false, w);
        tokio::pin!(h3);
        assert!(futures_util::poll!(&mut h3).is_pending());
        let s = a.acquire(&short, false, w);
        tokio::pin!(s);
        assert!(futures_util::poll!(&mut s).is_ready());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn memory_gate_holds_a_second_heavy_query() {
        let rss = Arc::new(std::sync::atomic::AtomicU64::new(150 << 20));
        let r = rss.clone();
        let probe: MemProbe = Arc::new(move || Some(r.load(std::sync::atomic::Ordering::Relaxed)));
        let a = Admission::new(2, 50.0, 100 << 20, Some(probe));
        let heavy = Cost {
            key: 1,
            expected_ms: 5000.0,
            heavy: true,
            big: false,
        };
        let w = std::time::Duration::from_millis(50);
        let _h1 = a
            .acquire(&heavy, false, w)
            .await
            .expect("first heavy runs regardless of RSS");
        assert!(a.acquire(&heavy, false, w).await.is_none(), "RSS over the gate");
        rss.store(50 << 20, std::sync::atomic::Ordering::Relaxed);
        let _h2 = a.acquire(&heavy, false, w).await.expect("RSS under the gate");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn big_shapes_run_alone_among_heavy_queries() {
        let a = Admission::new(4, 50.0, 0, None);
        let heavy = Cost {
            key: 1,
            expected_ms: 5000.0,
            heavy: true,
            big: false,
        };
        let big = a.mark_big(&Cost { key: 2, ..heavy });
        assert!(a.cost("x").key != 2 && big.big && big.heavy);
        let short = Cost {
            key: 3,
            expected_ms: 1.0,
            heavy: false,
            big: false,
        };
        let w = std::time::Duration::from_millis(50);
        let h = a.acquire(&heavy, false, w).await.unwrap();
        assert!(a.acquire(&big, false, w).await.is_none(), "big waits for heavy to end");
        drop(h);
        let b = a.acquire(&big, false, w).await.expect("alone now");
        assert!(a.acquire(&heavy, false, w).await.is_none(), "no heavy beside a big one");
        let _s = a.acquire(&short, false, w).await.expect("short still runs");
        drop(b);
    }

    /// Heavy queries keep overlapping; a big shape must still get its turn.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn big_shape_is_not_starved_by_overlapping_heavy_queries() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::{Duration, Instant};
        let a = Admission::new(4, 50.0, 0, None);
        let heavy = Cost {
            key: 1,
            expected_ms: 1000.0,
            heavy: true,
            big: false,
        };
        let big = a.mark_big(&Cost {
            key: 2,
            expected_ms: 1000.0,
            heavy: true,
            big: false,
        });
        let h0 = a.acquire(&heavy, false, Duration::from_secs(1)).await.unwrap();
        let a2 = a.clone();
        let t0 = Instant::now();
        let bigw = tokio::spawn(async move {
            let p = a2.acquire(&big, false, Duration::from_secs(5)).await;
            (p.is_some(), t0.elapsed())
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        let started = std::sync::Arc::new(AtomicUsize::new(0));
        let mut hs = vec![];
        let mut first = Some(h0);
        for _ in 0..30 {
            let (a3, st) = (a.clone(), started.clone());
            hs.push(tokio::spawn(async move {
                if let Some(p) = a3.acquire(&heavy, false, Duration::from_secs(30)).await {
                    st.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(250)).await;
                    drop(p);
                }
            }));
            tokio::time::sleep(Duration::from_millis(100)).await;
            first.take();
        }
        let (got, waited) = bigw.await.unwrap();
        assert!(got, "big shape starved for {waited:?}");
        assert!(waited < Duration::from_millis(2500), "big shape waited {waited:?}");
        for h in hs {
            let _ = h.await;
        }
    }
}
