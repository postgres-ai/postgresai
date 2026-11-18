//! `prom_eval_rs(fn, metric, matchers_json, start_ms, end_ms, step_ms, window_ms)`
//!   -> (series_id UBIGINT, t BIGINT, value DOUBLE), streamed series by series.
use super::kernels::{eval_series, Func, Grid};
use super::scan::{Scan, SeriesRows};
use super::{index, EvalHandle};
use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use std::sync::Mutex;

pub struct Bind {
    f: Func,
    grid: Grid,
    ids: super::scan::IdList,
    ctx: std::sync::Arc<super::EvalCtx>,
}

pub struct State {
    scan: Option<Scan>,
    roll: Option<super::rollup::RollupStream>,
    rows: SeriesRows,
    scratch: Vec<f64>,
    fts: Vec<i64>,
    fv: Vec<f64>,
    out: Vec<(u64, i64, f64)>,
    pos: usize,
    done: bool,
}

pub struct Init {
    st: Mutex<State>,
}

impl State {
    fn release(&mut self) {
        self.scan = None;
        self.roll = None;
        self.rows = SeriesRows::default();
        self.scratch = vec![];
        self.fts = vec![];
        self.fv = vec![];
        self.out = vec![];
        self.pos = 0;
    }
}

pub struct PromEval;

impl VTab for PromEval {
    type InitData = Init;
    type BindData = Bind;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        bind.add_result_column("series_id", LogicalTypeHandle::from(LogicalTypeId::UBigint));
        bind.add_result_column("t", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column("value", LogicalTypeHandle::from(LogicalTypeId::Double));
        let f = Func::parse(&bind.get_parameter(0).to_string())?;
        let metric = bind.get_parameter(1).to_string();
        let matchers = bind.get_parameter(2).to_string();
        let mut grid = Grid {
            start: bind.get_parameter(3).to_int64(),
            end: bind.get_parameter(4).to_int64(),
            step: bind.get_parameter(5).to_int64(),
            window: bind.get_parameter(6).to_int64(),
            interval: 0,
            phi: bind.get_parameter(7).to_double(),
        };
        if f == Func::QuantileOverTime && !(0.0..=1.0).contains(&grid.phi) {
            return Err("quantile_over_time: phi must be in [0, 1]".into());
        }
        if grid.step <= 0 {
            return Err("step must be > 0".into());
        }
        let handle = unsafe { &*bind.get_extra_info::<EvalHandle>() };
        let ctx = handle.read().unwrap().clone();
        grid.interval = ctx.interval(&metric);
        let ids: super::scan::IdList =
            index::select_series(&ctx, &[&metric], &matchers, grid.fetch_start(f), grid.end)?
                .ids(0)
                .into();
        Ok(Bind { f, grid, ids, ctx })
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        init.set_max_threads(1);
        Ok(Init {
            st: Mutex::new(State {
                scan: None,
                roll: None,
                rows: SeriesRows::default(),
                scratch: vec![],
                fts: vec![],
                fv: vec![],
                out: vec![],
                pos: 0,
                done: false,
            }),
        })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        let b = func.get_bind_data();
        let mut st = func.get_init_data().st.lock().unwrap();
        let st = &mut *st;
        if st.scan.is_none() && !st.done {
            if let Some(plan) = super::rollup::plan(&b.ctx, b.f, &b.grid) {
                st.roll = Some(super::rollup::RollupStream::open(&b.ctx, b.f, &b.grid, &b.ids, plan)?);
                st.scan = Some(Scan::open(&b.ctx.root, &[], vec![], 0, 0)?);
            } else {
                st.scan = Some(Scan::open(
                    &b.ctx.root,
                    &b.ctx.files,
                    b.ids.clone(),
                    b.grid.fetch_start(b.f),
                    b.grid.end,
                )?);
            }
        }
        let cap = output.flat_vector(0).capacity();
        let mut n = 0usize;
        let mut sid = output.flat_vector(0);
        let mut tv = output.flat_vector(1);
        let mut vv = output.flat_vector(2);
        let (sid_s, t_s, v_s) = unsafe {
            (
                sid.as_mut_slice_with_len::<u64>(cap),
                tv.as_mut_slice_with_len::<i64>(cap),
                vv.as_mut_slice_with_len::<f64>(cap),
            )
        };
        while n < cap {
            if st.pos < st.out.len() {
                let k = (cap - n).min(st.out.len() - st.pos);
                for (j, (s, t, v)) in st.out[st.pos..st.pos + k].iter().enumerate() {
                    sid_s[n + j] = *s;
                    t_s[n + j] = *t;
                    v_s[n + j] = *v;
                }
                st.pos += k;
                n += k;
                continue;
            }
            if st.done {
                // DuckDB keeps a table function's state until the whole query ends: release the
                // scan (open readers, decoded batches, footers) and the buffers now, or every
                // finished call in a query (a UNION ALL of several) keeps them (~4 MB each)
                st.release();
                break;
            }
            // next series
            if let Some(roll) = st.roll.as_mut() {
                st.out.clear();
                st.pos = 0;
                let out = &mut st.out;
                if !roll.next(|id, t, v| out.push((id, t, v)))? {
                    st.done = true;
                }
                continue;
            }
            let scan = st.scan.as_mut().unwrap();
            if !scan.next_series(&mut st.rows)? {
                st.done = true;
                continue;
            }
            st.out.clear();
            st.pos = 0;
            st.fts.clear();
            st.fv.clear();
            for i in 0..st.rows.ts.len() {
                // rollups skip NaN values and staleness markers (VM drops them before rollups);
                // a bare selector keeps markers so a trailing one hides the series at once
                if !st.rows.v[i].is_nan() || (b.f.keep_stale() && st.rows.stale[i]) {
                    st.fts.push(st.rows.ts[i]);
                    st.fv.push(st.rows.v[i]);
                }
            }
            let id = st.rows.id;
            let out = &mut st.out;
            eval_series(b.f, &b.grid, &st.fts, &st.fv, &mut st.scratch, |t, v| {
                out.push((id, t, v))
            });
        }
        output.set_len(n);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        let v = || LogicalTypeHandle::from(LogicalTypeId::Varchar);
        let i = || LogicalTypeHandle::from(LogicalTypeId::Bigint);
        Some(vec![
            v(),
            v(),
            v(),
            i(),
            i(),
            i(),
            i(),
            LogicalTypeHandle::from(LogicalTypeId::Double),
        ])
    }
}

/// `prom_series_rs(fn, metric, matchers_json, start, end, step, window)` -> (series_id, labels
/// JSON): the selected series with their labels, read from the series files of the call's window
/// (the same selection `prom_eval_rs` evaluates). The macros join this, not the `series` view,
/// so a call never touches the whole registry.
pub struct PromSeries;

pub struct SeriesBind {
    rows: Vec<(u64, String)>,
}

pub struct SeriesInit {
    pos: Mutex<usize>,
}

impl VTab for PromSeries {
    type InitData = SeriesInit;
    type BindData = SeriesBind;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        bind.add_result_column("series_id", LogicalTypeHandle::from(LogicalTypeId::UBigint));
        bind.add_result_column("labels", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        let f = Func::parse(&bind.get_parameter(0).to_string())?;
        let metric = bind.get_parameter(1).to_string();
        let matchers = bind.get_parameter(2).to_string();
        let mut grid = Grid {
            start: bind.get_parameter(3).to_int64(),
            end: bind.get_parameter(4).to_int64(),
            step: bind.get_parameter(5).to_int64().max(1),
            window: bind.get_parameter(6).to_int64(),
            interval: 0,
            phi: 0.0,
        };
        let handle = unsafe { &*bind.get_extra_info::<EvalHandle>() };
        let ctx = handle.read().unwrap().clone();
        grid.interval = ctx.interval(&metric);
        let sel = index::select_series(&ctx, &[&metric], &matchers, grid.fetch_start(f), grid.end)?;
        let rows = sel.entries.iter().map(|e| (e.0, sel.json(&e.2))).collect();
        Ok(SeriesBind { rows })
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        init.set_max_threads(1);
        Ok(SeriesInit { pos: Mutex::new(0) })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        let rows = &func.get_bind_data().rows;
        let mut pos = func.get_init_data().pos.lock().unwrap();
        let cap = output.flat_vector(0).capacity();
        let k = (rows.len() - *pos).min(cap);
        {
            let mut ids = output.flat_vector(0);
            let s = unsafe { ids.as_mut_slice_with_len::<u64>(k) };
            for j in 0..k {
                s[j] = rows[*pos + j].0;
            }
        }
        let lv = output.flat_vector(1);
        for j in 0..k {
            lv.insert(j, rows[*pos + j].1.as_str());
        }
        *pos += k;
        output.set_len(k);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        let v = || LogicalTypeHandle::from(LogicalTypeId::Varchar);
        let i = || LogicalTypeHandle::from(LogicalTypeId::Bigint);
        Some(vec![v(), v(), v(), i(), i(), i(), i()])
    }
}

/// `pgai_series_rs(metric, matchers_json, start_ms, end_ms)` -> (series_id, labels JSON): the
/// series `pgai_samples_rs` reads for the same arguments (active in [start, end]).
pub struct PgaiSeries;

impl VTab for PgaiSeries {
    type InitData = SeriesInit;
    type BindData = SeriesBind;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        bind.add_result_column("series_id", LogicalTypeHandle::from(LogicalTypeId::UBigint));
        bind.add_result_column("labels", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        let (metric, matchers) = (bind.get_parameter(0).to_string(), bind.get_parameter(1).to_string());
        let (t0, t1) = (bind.get_parameter(2).to_int64(), bind.get_parameter(3).to_int64());
        let handle = unsafe { &*bind.get_extra_info::<EvalHandle>() };
        let ctx = handle.read().unwrap().clone();
        let sel = index::select_series(&ctx, &[&metric], &matchers, t0, t1)?;
        let rows = sel.entries.iter().map(|e| (e.0, sel.json(&e.2))).collect();
        Ok(SeriesBind { rows })
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        <PromSeries as VTab>::init(init)
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        let rows = &func.get_bind_data().rows;
        let mut pos = func.get_init_data().pos.lock().unwrap();
        let cap = output.flat_vector(0).capacity();
        let k = (rows.len() - *pos).min(cap);
        {
            let mut ids = output.flat_vector(0);
            let s = unsafe { ids.as_mut_slice_with_len::<u64>(k) };
            for j in 0..k {
                s[j] = rows[*pos + j].0;
            }
        }
        let lv = output.flat_vector(1);
        for j in 0..k {
            lv.insert(j, rows[*pos + j].1.as_str());
        }
        *pos += k;
        output.set_len(k);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        let v = || LogicalTypeHandle::from(LogicalTypeId::Varchar);
        let i = || LogicalTypeHandle::from(LogicalTypeId::Bigint);
        Some(vec![v(), v(), i(), i()])
    }
}

/// `pgai_samples_rs(metric, matchers_json, start_ms, end_ms)` -> (series_id UBIGINT, ts_ms
/// BIGINT, value DOUBLE, stale BOOLEAN): every raw sample of the selected series with
/// start <= ts <= end, NaN values and staleness markers included (as the `samples` view), read by
/// the Rust scan, so it works on local and object stores alike. Ordered by (series_id, ts).
pub struct PgaiSamples;

pub struct SamplesBind {
    ids: super::scan::IdList,
    t0: i64,
    t1: i64,
    ctx: std::sync::Arc<super::EvalCtx>,
}

pub struct SamplesState {
    scan: Option<Scan>,
    rows: SeriesRows,
    pos: usize,
    done: bool,
}

pub struct SamplesInit {
    st: Mutex<SamplesState>,
}

impl VTab for PgaiSamples {
    type InitData = SamplesInit;
    type BindData = SamplesBind;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        bind.add_result_column("series_id", LogicalTypeHandle::from(LogicalTypeId::UBigint));
        bind.add_result_column("ts_ms", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column("value", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("stale", LogicalTypeHandle::from(LogicalTypeId::Boolean));
        let (metric, matchers) = (bind.get_parameter(0).to_string(), bind.get_parameter(1).to_string());
        let (t0, t1) = (bind.get_parameter(2).to_int64(), bind.get_parameter(3).to_int64());
        if t1 < t0 {
            return Err("pgai_samples: end_ms must be >= start_ms".into());
        }
        let handle = unsafe { &*bind.get_extra_info::<EvalHandle>() };
        let ctx = handle.read().unwrap().clone();
        let ids: super::scan::IdList = index::select_series(&ctx, &[&metric], &matchers, t0, t1)?.ids(0).into();
        Ok(SamplesBind { ids, t0, t1, ctx })
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        init.set_max_threads(1);
        Ok(SamplesInit {
            st: Mutex::new(SamplesState {
                scan: None,
                rows: SeriesRows::default(),
                pos: 0,
                done: false,
            }),
        })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        let b = func.get_bind_data();
        let mut st = func.get_init_data().st.lock().unwrap();
        let st = &mut *st;
        if st.scan.is_none() && !st.done {
            st.scan = Some(Scan::open(&b.ctx.root, &b.ctx.files, b.ids.clone(), b.t0, b.t1)?);
        }
        let cap = output.flat_vector(0).capacity();
        let mut n = 0usize;
        let mut sid = output.flat_vector(0);
        let mut tv = output.flat_vector(1);
        let mut vv = output.flat_vector(2);
        let mut sv = output.flat_vector(3);
        let (sid_s, t_s, v_s, s_s) = unsafe {
            (
                sid.as_mut_slice_with_len::<u64>(cap),
                tv.as_mut_slice_with_len::<i64>(cap),
                vv.as_mut_slice_with_len::<f64>(cap),
                sv.as_mut_slice_with_len::<bool>(cap),
            )
        };
        while n < cap {
            let r = &st.rows;
            if st.pos < r.ts.len() {
                let k = (cap - n).min(r.ts.len() - st.pos);
                for j in 0..k {
                    sid_s[n + j] = r.id;
                    t_s[n + j] = r.ts[st.pos + j];
                    v_s[n + j] = r.v[st.pos + j];
                    s_s[n + j] = r.stale[st.pos + j];
                }
                st.pos += k;
                n += k;
                continue;
            }
            if st.done {
                // release the scan now (see PromEval::func)
                st.scan = None;
                st.rows = SeriesRows::default();
                st.pos = 0;
                break;
            }
            st.pos = 0;
            if !st.scan.as_mut().unwrap().next_series(&mut st.rows)? {
                st.rows = SeriesRows::default();
                st.done = true;
            }
        }
        output.set_len(n);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        let v = || LogicalTypeHandle::from(LogicalTypeId::Varchar);
        let i = || LogicalTypeHandle::from(LogicalTypeId::Bigint);
        Some(vec![v(), v(), i(), i()])
    }
}
