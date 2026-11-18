//! `prom_agg_rs(op, fn, metric, matchers_json, start, end, step, window, by_json)`
//!   -> (series_id, t, value): `op by (by) (fn(metric{matchers}))` per step, op one of sum, avg,
//!   count, min, max. series_id is the group's smallest member id, so the wrapper can attach the
//!   group's labels (prom_by_rs). A group has a point at a step when any member has one (NaN
//!   points are dropped, as in PromQL aggregation).
//!   Memory: one series at a time plus one accumulator row per group, not every point. This is
//!   the sink for D2's sum(irate(...)) panels and 48A/48B's ratio of two sums, which otherwise
//!   materialise every series' grid in DuckDB before GROUP BY t.
use super::index::{self, Selection};
use super::kernels::{Func, Grid};
use super::topk::for_each_series;
use super::{EvalCtx, EvalHandle};
use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Sum,
    Avg,
    Count,
    Min,
    Max,
}

impl Op {
    fn parse(s: &str) -> anyhow::Result<Op> {
        Ok(match s {
            "sum" => Op::Sum,
            "avg" => Op::Avg,
            "count" => Op::Count,
            "min" => Op::Min,
            "max" => Op::Max,
            _ => anyhow::bail!("prom_agg: unsupported op {s:?} (sum, avg, count, min, max)"),
        })
    }
}

pub struct Bind {
    op: Op,
    f: Func,
    grid: Grid,
    ids: super::scan::IdList,
    by: Vec<String>,
    ctx: Arc<EvalCtx>,
    sel: Arc<Selection>,
    pos: HashMap<u64, usize>,
}

/// Output cursor: the groups (sorted by representative id) and the next (group, step).
/// Rows are emitted straight from the accumulators, never copied into a row list.
pub struct Init {
    st: Mutex<Option<(Vec<Acc>, usize, usize)>>,
}

struct Acc {
    rep: u64,
    v: Vec<f64>,
    n: Vec<u32>,
}

fn compute(b: &Bind) -> anyhow::Result<Vec<Acc>> {
    let np = b.grid.points();
    let mut groups: HashMap<Vec<u32>, Acc> = HashMap::new();
    // label values are interned in the selection, so the group key is a vector of string ids
    let keys: Vec<Option<u32>> = b.by.iter().map(|k| b.sel.str_id(k)).collect();
    for_each_series(&b.ctx, b.f, &b.grid, &b.ids, |id, row| {
        let Some(&pi) = b.pos.get(&id) else { return };
        let l = &b.sel.entries[pi].2;
        let key: Vec<u32> = keys
            .iter()
            .map(|k| {
                k.and_then(|k| l.iter().find(|(n, _)| *n == k).map(|(_, v)| *v))
                    .filter(|v| !b.sel.str(*v).is_empty())
                    .unwrap_or(u32::MAX)
            })
            .collect();
        let init = if matches!(b.op, Op::Min | Op::Max) {
            f64::NAN
        } else {
            0.0
        };
        let g = groups.entry(key).or_insert_with(|| Acc {
            rep: id,
            v: vec![init; np],
            n: vec![0; np],
        });
        g.rep = g.rep.min(id);
        for (j, &x) in row.iter().enumerate() {
            if x.is_nan() {
                continue;
            }
            g.v[j] = match b.op {
                Op::Sum | Op::Avg => g.v[j] + x,
                Op::Count => g.v[j] + 1.0,
                Op::Min => {
                    if g.n[j] == 0 || x < g.v[j] {
                        x
                    } else {
                        g.v[j]
                    }
                }
                Op::Max => {
                    if g.n[j] == 0 || x > g.v[j] {
                        x
                    } else {
                        g.v[j]
                    }
                }
            };
            g.n[j] += 1;
        }
    })?;
    let mut gs: Vec<Acc> = groups.into_values().collect();
    gs.sort_by_key(|g| g.rep);
    Ok(gs)
}

pub struct PromAgg;

impl VTab for PromAgg {
    type InitData = Init;
    type BindData = Bind;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        bind.add_result_column("series_id", LogicalTypeHandle::from(LogicalTypeId::UBigint));
        bind.add_result_column("t", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column("value", LogicalTypeHandle::from(LogicalTypeId::Double));
        let op = Op::parse(&bind.get_parameter(0).to_string())?;
        let f = Func::parse(&bind.get_parameter(1).to_string())?;
        let metric = bind.get_parameter(2).to_string();
        let matchers = bind.get_parameter(3).to_string();
        let mut grid = Grid {
            start: bind.get_parameter(4).to_int64(),
            end: bind.get_parameter(5).to_int64(),
            step: bind.get_parameter(6).to_int64(),
            window: bind.get_parameter(7).to_int64(),
            interval: 0,
            phi: 0.0,
        };
        if grid.step <= 0 {
            return Err("step must be > 0".into());
        }
        let by_json = bind.get_parameter(8).to_string();
        let by: Vec<String> = if by_json.is_empty() || by_json == "NULL" || by_json == "null" {
            vec![]
        } else {
            serde_json::from_str(&by_json)?
        };
        let handle = unsafe { &*bind.get_extra_info::<EvalHandle>() };
        let ctx = handle.read().unwrap().clone();
        grid.interval = ctx.interval(&metric);
        let sel = index::select_series(&ctx, &[&metric], &matchers, grid.fetch_start(f), grid.end)?;
        let ids: super::scan::IdList = sel.ids(0).into();
        let pos: HashMap<u64, usize> = sel.entries.iter().enumerate().map(|(i, e)| (e.0, i)).collect();
        Ok(Bind {
            op,
            f,
            grid,
            ids,
            by,
            ctx,
            sel,
            pos,
        })
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        init.set_max_threads(1);
        Ok(Init { st: Mutex::new(None) })
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        let b = func.get_bind_data();
        let mut st = func.get_init_data().st.lock().unwrap();
        if st.is_none() {
            *st = Some((compute(b)?, 0, 0));
        }
        let (gs, gi, j) = st.as_mut().unwrap();
        let cap = output.flat_vector(0).capacity();
        let np = b.grid.points();
        let mut a = output.flat_vector(0);
        let mut t = output.flat_vector(1);
        let mut v = output.flat_vector(2);
        let (a, t, v) = unsafe {
            (
                a.as_mut_slice_with_len::<u64>(cap),
                t.as_mut_slice_with_len::<i64>(cap),
                v.as_mut_slice_with_len::<f64>(cap),
            )
        };
        let mut k = 0;
        while k < cap && *gi < gs.len() {
            let g = &mut gs[*gi];
            if *j >= np {
                // done with this group: free its accumulators
                g.v = Vec::new();
                g.n = Vec::new();
                *gi += 1;
                *j = 0;
                continue;
            }
            if g.n[*j] > 0 {
                a[k] = g.rep;
                t[k] = b.grid.start + *j as i64 * b.grid.step;
                v[k] = if b.op == Op::Avg {
                    g.v[*j] / g.n[*j] as f64
                } else {
                    g.v[*j]
                };
                k += 1;
            }
            *j += 1;
        }
        output.set_len(k);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        let v = || LogicalTypeHandle::from(LogicalTypeId::Varchar);
        let i = || LogicalTypeHandle::from(LogicalTypeId::Bigint);
        Some(vec![v(), v(), v(), v(), i(), i(), i(), i(), v()])
    }
}
