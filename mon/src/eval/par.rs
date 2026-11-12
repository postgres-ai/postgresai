//! Intra-query parallelism for the evaluator (piece 4): a large selection is split into
//! contiguous id ranges; extra worker threads scan and evaluate later ranges while the calling
//! thread does the first, and the caller's sink sees every series **in id order**, exactly as
//! the one-thread path would. So aggregation, top-k and ratios need no merge step, and the
//! output is bit-identical whatever the number of workers.
//!
//! - Workers are taken only from free query slots: the budget is `slots - queries running -
//!   workers already lent` (`set_slots`, from `HttpCfg::eval_workers`), never blocking; with
//!   nothing free the query runs on its own thread as before.
//! - Memory: a worker's rows wait in a channel of at most `BUF_BYTES` (4 MB) of grid rows (a
//!   worker that gets ahead blocks), plus its own scan cursors: decoded row groups of every file
//!   it has open. That grows with the days the scan spans: measured on a 14 d D1 panel, about
//!   4 MB + 3 MB per day per worker (24 h: 8, 7 d: 26, 14 d: 45 MiB). Workers are lent only
//!   within the RSS headroom (`borrow`): the reader's budget (65% of its RSS cap) minus its
//!   current RSS minus what workers already lent may still add, at `worker_bytes(span, points)`
//!   each: the worker's channel (bounded in bytes, so at least one row of `points` values)
//!   plus `PER_DAY` per day of span. Check and reservation happen under one lock, so concurrent
//!   queries cannot each take the same headroom (review PIECE4 H1, M1).
//! - Selections under `MIN_PER_PART` series per worker are not split.
use super::scan::IdList;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

static SLOTS: AtomicUsize = AtomicUsize::new(1);
static LENT: AtomicUsize = AtomicUsize::new(0);
static ACTIVE: OnceLock<Arc<AtomicUsize>> = OnceLock::new();
/// Lend workers only while this process's RSS leaves room: `MEM_BUDGET` bytes (a share of the
/// reader's RSS cap) minus the current RSS minus `LENT_BYTES`. 0 = no memory check.
static MEM_BUDGET: AtomicU64 = AtomicU64::new(0);
static LENT_BYTES: AtomicU64 = AtomicU64::new(0);
static PER_DAY: AtomicU64 = AtomicU64::new(DEFAULT_PER_DAY_MB << 20);
pub static PAR_DENIED_MEM: AtomicUsize = AtomicUsize::new(0);
pub static PAR_QUERIES: AtomicUsize = AtomicUsize::new(0);
pub static PAR_WORKERS: AtomicUsize = AtomicUsize::new(0);

const MIN_PER_PART: usize = 128;
const BUF_BYTES: usize = 4 << 20;
/// Memory one more worker may add: its channel (`channel_rows` x row bytes, plus the row in
/// flight), `WORKER_BASE` (buffers) and `PER_DAY` for each
/// day its scan spans (a decoded row group per open day file, malloc slack). Measured: module doc.
const WORKER_BASE: u64 = 1 << 20;
const DEFAULT_PER_DAY_MB: u64 = 3;

/// Bytes of one grid row in a worker's channel.
fn row_bytes(points: usize) -> usize {
    points * 8 + 64
}

/// A worker's channel capacity in rows: `BUF_BYTES` worth, at least one.
fn channel_rows(points: usize) -> usize {
    (BUF_BYTES / row_bytes(points)).max(1)
}

/// Estimated bytes one worker adds for a scan over `span_ms` with `points` grid points.
pub fn worker_bytes(span_ms: i64, points: usize) -> u64 {
    let days = (span_ms.max(0) as u64).div_ceil(crate::model::DAY_MS as u64).max(1);
    let channel = ((channel_rows(points) + 2) * row_bytes(points)) as u64;
    WORKER_BASE + channel + days * PER_DAY.load(Ordering::Relaxed)
}

/// Check-and-reserve of memory and slots is one critical section (review PIECE4 M1).
static BORROW: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Worker budget: `slots` threads in all for evaluation, shared with the running queries
/// (`active` counts them; each query's own thread is one).
pub fn set_slots(slots: usize, active: Arc<AtomicUsize>, mem_budget: u64, per_day_mb: u64) {
    SLOTS.store(slots.max(1), Ordering::Relaxed);
    let _ = ACTIVE.set(active);
    MEM_BUDGET.store(mem_budget, Ordering::Relaxed);
    if per_day_mb > 0 {
        PER_DAY.store(per_day_mb << 20, Ordering::Relaxed);
    }
}

/// How many workers of `each` bytes the memory headroom allows (usize::MAX when unchecked).
fn mem_allows(each: u64) -> usize {
    let budget = MEM_BUDGET.load(Ordering::Relaxed);
    if budget == 0 {
        return usize::MAX;
    }
    let Some(rss) = crate::sqlproc::rss_bytes(std::process::id()) else {
        return 0;
    };
    let room = budget
        .saturating_sub(rss)
        .saturating_sub(LENT_BYTES.load(Ordering::Relaxed));
    (room / each.max(1)) as usize
}

struct Lent(usize, u64);
impl Drop for Lent {
    fn drop(&mut self) {
        LENT.fetch_sub(self.0, Ordering::Relaxed);
        LENT_BYTES.fetch_sub(self.1, Ordering::Relaxed);
    }
}

fn borrow(want: usize, each: u64) -> Lent {
    // tests: PGAI_EVAL_FORCE_WORKERS=n lends n workers regardless of the budget
    if let Some(n) = std::env::var("PGAI_EVAL_FORCE_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        LENT.fetch_add(want.min(n), Ordering::Relaxed);
        return Lent(want.min(n), 0);
    }
    if SLOTS.load(Ordering::Relaxed) <= 1 {
        return Lent(0, 0); // off (the default): no RSS probe on every call
    }
    let _one = BORROW.lock().unwrap_or_else(|e| e.into_inner());
    let mem = mem_allows(each);
    if mem == 0 {
        PAR_DENIED_MEM.fetch_add(1, Ordering::Relaxed);
        return Lent(0, 0);
    }
    let want = want.min(mem);
    let active = ACTIVE.get().map_or(1, |a| a.load(Ordering::Relaxed).max(1));
    let slots = SLOTS.load(Ordering::Relaxed);
    let mut got = 0;
    let _ = LENT.fetch_update(Ordering::AcqRel, Ordering::Relaxed, |lent| {
        let free = slots.saturating_sub(active + lent);
        got = want.min(free);
        (got > 0).then_some(lent + got)
    });
    let bytes = if mem == usize::MAX { 0 } else { got as u64 * each };
    if std::env::var_os("PGAI_EVAL_TRACE").is_some() {
        eprintln!(
            "par: want {want} mem_allows {mem} each {} MB active {active} slots {slots} got {got}",
            each >> 20
        );
    }
    LENT_BYTES.fetch_add(bytes, Ordering::Relaxed);
    Lent(got, bytes)
}

/// Run `produce` over `ids`, possibly split across workers, calling `each(id, row)` in id order.
/// `produce(sub, emit)` must emit the series of `sub` in id order (a Scan does).
pub fn for_each_ordered<P>(
    ids: &IdList,
    points: usize,
    span_ms: i64,
    produce: P,
    each: &mut dyn FnMut(u64, &[f64]),
) -> anyhow::Result<()>
where
    P: Fn(&IdList, &mut dyn FnMut(u64, &[f64])) -> anyhow::Result<()> + Sync,
{
    let min_per = std::env::var("PGAI_EVAL_PAR_MIN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(MIN_PER_PART)
        .max(1);
    let max_parts = ids.len() / min_per;
    let lent = if max_parts >= 2 {
        borrow(max_parts - 1, worker_bytes(span_ms, points))
    } else {
        Lent(0, 0)
    };
    if lent.0 == 0 {
        return produce(ids, each);
    }
    let parts = lent.0 + 1;
    PAR_QUERIES.fetch_add(1, Ordering::Relaxed);
    PAR_WORKERS.fetch_add(lent.0, Ordering::Relaxed);
    let n = ids.len();
    let bounds: Vec<(usize, usize)> = (0..parts).map(|p| (n * p / parts, n * (p + 1) / parts)).collect();
    let cap_rows = channel_rows(points);
    std::thread::scope(|s| -> anyhow::Result<()> {
        let mut rxs = Vec::new();
        let mut hs = Vec::new();
        for &(lo, hi) in &bounds[1..] {
            let (tx, rx) = std::sync::mpsc::sync_channel::<(u64, Vec<f64>)>(cap_rows);
            let sub = ids.sub(lo, hi);
            let produce = &produce;
            hs.push(s.spawn(move || -> anyhow::Result<()> {
                let mut gone = false;
                produce(&sub, &mut |id, row| {
                    if !gone && tx.send((id, row.to_vec())).is_err() {
                        gone = true; // the consumer stopped (error): finish quietly
                    }
                })
            }));
            rxs.push(rx);
        }
        let (lo, hi) = bounds[0];
        let first = produce(&ids.sub(lo, hi), each);
        let mut res = first;
        for rx in rxs.iter() {
            if res.is_err() {
                break;
            }
            for (id, row) in rx.iter() {
                each(id, &row);
            }
        }
        drop(rxs);
        for h in hs {
            let r = h
                .join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("evaluator worker panicked")));
            if res.is_ok() {
                res = r;
            }
        }
        res
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review PIECE4 H1: the charge covers the channel, whose floor is one row of `points`.
    #[test]
    fn worker_bytes_covers_the_channel() {
        for points in [100usize, 1_000, 32_000, 160_000, 400_000] {
            let row = (points * 8 + 64) as u64;
            let cap = channel_rows(points) as u64;
            assert!(
                cap * row <= (BUF_BYTES as u64).max(row),
                "{points}: channel bounded in bytes"
            );
            assert!(
                worker_bytes(crate::model::DAY_MS, points) >= cap * row + row,
                "{points}"
            );
        }
        // a 400k-point grid: one channel row is 3.2 MB, charged with two in flight
        assert!(worker_bytes(crate::model::DAY_MS, 400_000) >= 3 * 3_200_064);
        // days still count
        assert!(worker_bytes(14 * crate::model::DAY_MS, 100) >= worker_bytes(crate::model::DAY_MS, 100) + (39 << 20));
    }
}
