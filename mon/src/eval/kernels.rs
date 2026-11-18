//! Rollup kernels, ported from sql/macros.sql (prom_select + prom_edges and the
//! per-function CASEs), which reproduce VictoriaMetrics v1.140.0 rollup.go. Times are ms.

/// The function a table-function call evaluates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Func {
    Rate,
    Irate,
    Increase,
    Delta,
    IncreasePure,
    Idelta,
    /// bare selector (default_rollup): last raw sample, staleness markers end the series
    Last,
    /// timestamp(x): last sample's time in seconds
    Timestamp,
    LastOverTime,
    TlastOverTime,
    FirstOverTime,
    // window aggregates over (t - w, t] (prom_window): no previous-sample logic
    AvgOverTime,
    SumOverTime,
    MinOverTime,
    MaxOverTime,
    CountOverTime,
    StddevOverTime,
    QuantileOverTime,
}

impl Func {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "rate" => Func::Rate,
            "irate" => Func::Irate,
            "increase" => Func::Increase,
            "delta" => Func::Delta,
            "increase_pure" => Func::IncreasePure,
            "idelta" => Func::Idelta,
            "last" => Func::Last,
            "timestamp" => Func::Timestamp,
            "last_over_time" => Func::LastOverTime,
            "tlast_over_time" => Func::TlastOverTime,
            "first_over_time" => Func::FirstOverTime,
            "avg_over_time" => Func::AvgOverTime,
            "sum_over_time" => Func::SumOverTime,
            "min_over_time" => Func::MinOverTime,
            "max_over_time" => Func::MaxOverTime,
            "count_over_time" => Func::CountOverTime,
            "stddev_over_time" => Func::StddevOverTime,
            "quantile_over_time" => Func::QuantileOverTime,
            f => anyhow::bail!("unsupported function {f}"),
        })
    }
    /// needSilenceIntervalForRollupFunc: fetch 5 m more history.
    fn silence(self) -> bool {
        matches!(
            self,
            Func::Rate | Func::Irate | Func::Increase | Func::Delta | Func::IncreasePure | Func::Idelta | Func::Last
        )
    }
    /// rollupFuncsCanAdjustWindow: window = max(step, maxPrev) when no [d] is given.
    fn adjust(self) -> bool {
        matches!(self, Func::Rate | Func::Irate | Func::Last | Func::Timestamp)
    }
    /// Keeps staleness markers in the sample list (only a bare selector).
    pub fn keep_stale(self) -> bool {
        matches!(self, Func::Last)
    }
    /// prom_window family: plain window aggregate, window = step when no [d].
    pub fn is_window_agg(self) -> bool {
        matches!(
            self,
            Func::AvgOverTime
                | Func::SumOverTime
                | Func::MinOverTime
                | Func::MaxOverTime
                | Func::CountOverTime
                | Func::StddevOverTime
                | Func::QuantileOverTime
        )
    }
    /// Uses counter-reset-corrected values.
    pub fn corrected(self) -> bool {
        matches!(self, Func::Rate | Func::Irate | Func::Increase | Func::IncreasePure)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub start: i64,
    pub end: i64,
    pub step: i64,
    pub window: i64,
    /// Dictionary interval for a slow bare selector's lookback floor, in ms (0: disabled).
    pub interval: i64,
    /// quantile_over_time's phi.
    pub phi: f64,
}

impl Grid {
    /// First ms that prom_select fetches for `f`.
    pub fn fetch_start(&self, f: Func) -> i64 {
        if f.is_window_agg() {
            let w = if self.window > 0 { self.window } else { self.step };
            return self.start - w.max(self.step);
        }
        let history = if f.silence() { 300_000 } else { 0 } + self.window.max(self.step);
        self.start.saturating_sub(history.max(self.last_lookback(f)))
    }
    pub fn last_lookback(&self, f: Func) -> i64 {
        if matches!(f, Func::Last | Func::Timestamp) && self.window == 0 && self.interval > 300_000 {
            max_prev(self.interval)
        } else {
            0
        }
    }
    pub fn points(&self) -> usize {
        if self.end < self.start || self.step <= 0 {
            0
        } else {
            ((self.end - self.start) / self.step + 1) as usize
        }
    }
}

/// getMaxPrevInterval().
pub fn max_prev(si: i64) -> i64 {
    if si <= 2000 {
        si + 4 * si
    } else if si <= 4000 {
        si + 2 * si
    } else if si <= 8000 {
        si + si
    } else if si <= 16000 {
        si + si / 2
    } else if si <= 32000 {
        si + si / 4
    } else {
        si.saturating_add(si / 8)
    }
}

/// removeCounterResets() correction given the previous raw value.
pub fn reset_adj(v: f64, pv: f64) -> f64 {
    if v >= pv {
        0.0
    } else if (pv - v) * 8.0 < pv {
        pv - v
    } else {
        pv
    }
}

/// DuckDB quantile_cont (linear interpolation) then trunc, as prom_select's `si`.
pub fn scrape_interval(ts: &[i64]) -> Option<i64> {
    let n = ts.len();
    if n < 2 {
        return None;
    }
    // dts of rows rn > n - 20 with a previous sample
    let from = n.saturating_sub(20).max(1);
    si_of((from..n).map(|i| ts[i] - ts[i - 1]).collect())
}

/// The scrape-interval estimate from the last (up to) 20 intervals, in any order.
pub fn si_of(mut d: Vec<i64>) -> Option<i64> {
    if d.is_empty() {
        return None;
    }
    d.sort_unstable();
    let pos = 0.6 * (d.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let x = d[lo] as f64 + (pos - lo as f64) * (d[hi] - d[lo]) as f64;
    Some(x.trunc() as i64)
}

/// Evaluate `f` for one series. `ts`/`v` are the series' samples in [fetch_start, end] in time
/// order, staleness markers and NaN already removed. Calls `emit(t, value)` per grid point that
/// has a value. `c` is scratch for corrected values.
pub fn eval_series(f: Func, g: &Grid, ts: &[i64], v: &[f64], c: &mut Vec<f64>, emit: impl FnMut(i64, f64)) {
    let n_total = ts.len();
    if n_total == 0 {
        return;
    }
    if f.is_window_agg() {
        return window_agg(f, g, ts, v, c, emit);
    }
    c.clear();
    let mut adj = 0.0;
    for i in 0..n_total {
        if i > 0 {
            adj += reset_adj(v[i], v[i - 1]);
        }
        c.push(v[i] + adj);
    }
    eval_prepared(f, g, ts, v, c, scrape_interval(ts), emit)
}

/// The edge-rollup core on a prepared sequence: `c` = counter-reset-corrected values, `si` = the
/// scrape-interval estimate of the fetched samples. The rollup path passes a compressed sequence
/// (first/second/last per bucket) whose positions and corrected values are exact at bucket
/// boundaries, with `si` taken from raw samples.
pub fn eval_prepared(
    f: Func,
    g: &Grid,
    ts: &[i64],
    v: &[f64],
    c: &[f64],
    si: Option<i64>,
    mut emit: impl FnMut(i64, f64),
) {
    let n_total = ts.len();
    if n_total == 0 {
        return;
    }
    let x: &[f64] = if f.corrected() { c } else { v };
    let maxprev = if g.start < g.end {
        max_prev(si.unwrap_or(g.step))
    } else {
        g.step
    }
    .max(g.last_lookback(f));
    let w = if g.window > 0 {
        g.window
    } else if f.adjust() && g.step < maxprev {
        maxprev
    } else {
        g.step
    };
    let mut a = 0usize; // # samples with ts <= t - w   (a.rn)
    let mut b = 0usize; // # samples with ts <= t       (b.rn)
    let mut t = g.start;
    while t <= g.end {
        let ta = t - w;
        while a < n_total && ts[a] <= ta {
            a += 1;
        }
        while b < n_total && ts[b] <= t {
            b += 1;
        }
        let n = b - a;
        let has_real_prev = a > 0;
        let has_prev = a > 0 && a < n_total && ts[a - 1] > ta - maxprev;
        let prev = |arr: &[f64]| arr[a - 1];
        let val: Option<f64> = match f {
            Func::Rate => {
                if has_prev {
                    Some(if n == 0 {
                        0.0
                    } else {
                        (x[b - 1] - prev(x)) / ((ts[b - 1] - ts[a - 1]) as f64 / 1e3)
                    })
                } else if n < 2 {
                    None
                } else {
                    Some((x[b - 1] - x[a]) / ((ts[b - 1] - ts[a]) as f64 / 1e3))
                }
            }
            Func::Irate => {
                if n == 0 {
                    None
                } else if n == 1 {
                    if has_prev {
                        Some((x[b - 1] - prev(x)) / ((ts[b - 1] - ts[a - 1]) as f64 / 1e3))
                    } else {
                        None
                    }
                } else {
                    Some((x[b - 1] - x[b - 2]) / ((ts[b - 1] - ts[b - 2]) as f64 / 1e3))
                }
            }
            Func::Increase | Func::Delta => {
                if has_prev {
                    Some(if n == 0 { 0.0 } else { x[b - 1] - prev(x) })
                } else if n == 0 {
                    None
                } else if has_real_prev {
                    Some(x[b - 1] - prev(x))
                } else {
                    let first = x[a];
                    let d = if n > 1 {
                        x[a + 1] - first
                    } else if b < n_total {
                        x[b] - first
                    } else {
                        0.0
                    };
                    Some(if first.abs() < 10.0 * (d.abs() + 1.0) {
                        x[b - 1]
                    } else {
                        x[b - 1] - first
                    })
                }
            }
            Func::IncreasePure => {
                if has_prev {
                    Some(if n == 0 { 0.0 } else { x[b - 1] - prev(x) })
                } else if n == 0 {
                    None
                } else if has_real_prev {
                    Some(x[b - 1] - prev(x))
                } else {
                    Some(x[b - 1])
                }
            }
            Func::Last => {
                if n > 0 && !x[b - 1].is_nan() {
                    Some(x[b - 1])
                } else {
                    None
                }
            }
            Func::Timestamp | Func::TlastOverTime => (n > 0).then(|| ts[b - 1] as f64 / 1e3),
            Func::LastOverTime => (n > 0).then(|| x[b - 1]),
            Func::FirstOverTime => (n > 0).then(|| x[a]),
            Func::AvgOverTime
            | Func::SumOverTime
            | Func::MinOverTime
            | Func::MaxOverTime
            | Func::CountOverTime
            | Func::StddevOverTime
            | Func::QuantileOverTime => unreachable!(),
            Func::Idelta => {
                if n == 0 {
                    if has_prev {
                        Some(0.0)
                    } else {
                        None
                    }
                } else if n == 1 {
                    Some(if has_prev { x[b - 1] - prev(x) } else { x[b - 1] })
                } else {
                    Some(x[b - 1] - x[b - 2])
                }
            }
        };
        if let Some(y) = val {
            emit(t, y);
        }
        t += g.step;
    }
}

/// prom_window + GROUP BY: aggregate the raw samples in (t - w, t], w = window or step.
fn window_agg(f: Func, g: &Grid, ts: &[i64], v: &[f64], scratch: &mut Vec<f64>, mut emit: impl FnMut(i64, f64)) {
    let w = if g.window > 0 { g.window } else { g.step };
    let n_total = ts.len();
    let (mut a, mut b) = (0usize, 0usize);
    let mut t = g.start;
    while t <= g.end {
        while a < n_total && ts[a] <= t - w {
            a += 1;
        }
        while b < n_total && ts[b] <= t {
            b += 1;
        }
        if b > a {
            let xs = &v[a..b];
            let n = xs.len() as f64;
            let y = match f {
                Func::SumOverTime => xs.iter().sum::<f64>(),
                Func::AvgOverTime => xs.iter().sum::<f64>() / n,
                Func::MinOverTime => xs.iter().copied().fold(f64::INFINITY, f64::min),
                Func::MaxOverTime => xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                Func::CountOverTime => n,
                Func::StddevOverTime => {
                    // Welford, population form (DuckDB stddev_pop)
                    let (mut mean, mut m2) = (0.0, 0.0);
                    for (i, x) in xs.iter().enumerate() {
                        let d = x - mean;
                        mean += d / (i + 1) as f64;
                        m2 += d * (x - mean);
                    }
                    (m2 / n).sqrt()
                }
                Func::QuantileOverTime => {
                    scratch.clear();
                    scratch.extend_from_slice(xs);
                    scratch.sort_by(|p, q| p.total_cmp(q));
                    let pos = g.phi * (scratch.len() - 1) as f64;
                    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
                    scratch[lo] + (pos - lo as f64) * (scratch[hi] - scratch[lo])
                }
                _ => unreachable!(),
            };
            emit(t, y);
        }
        t += g.step;
    }
}
