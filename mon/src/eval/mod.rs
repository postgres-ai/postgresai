//! Series-at-a-time PromQL-shaped evaluator.
//! Exposed to SQL as table functions in the reader: `prom_eval_rs(...)`, wrapped by macros with
//! the same names and signatures as sql/macros.sql (which become the `*_sql` oracle).
pub mod agg;
pub mod footer;
pub mod index;
pub mod kernels;
pub mod par;
pub mod rollup;
pub mod rowcache;
pub mod scan;
pub mod topk;
pub mod vtab;

use crate::catalog::FileEntry;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// What the table functions read: the catalog's sample, rollup and series files. Swapped
/// atomically by the reader's view refresh; a running query keeps the Arc it started with.
/// There is no in-memory series index: each call reads the series files of its time window.
pub struct EvalCtx {
    pub root: PathBuf,
    pub files: Vec<FileEntry>,
    /// Catalog generation and data epoch this context was built from; the newest sample's
    /// time and the closed-watermark lag (for the cache below the SQL layer, not built yet).
    pub generation: u64,
    pub data_epoch: u64,
    pub hwm_ms: i64,
    pub closed_lag_ms: i64,
    /// Last few series selections (a query binds each table function twice, and the macros call
    /// prom_eval_rs and prom_series_rs with the same selector): key -> selection.
    pub sel_cache: std::sync::Mutex<Vec<(String, Arc<index::Selection>)>>,
    /// Same for the lean (ids-only) selections of top-k.
    pub ids_cache: std::sync::Mutex<Vec<(String, Arc<index::IdSel>)>>,
    /// Series registry files and the series files not yet registered (registry.rs).
    pub reg_files: Vec<FileEntry>,
    pub tail_files: Vec<FileEntry>,
    /// Tail series new to the registry: id -> index into tail_files of the first file holding
    /// it. Computed on first use per catalog generation.
    pub tail_new: std::sync::OnceLock<Result<Arc<Vec<(u64, u32)>>, String>>,
}

impl EvalCtx {
    /// The context for a catalog's files (the reader's view refresh; tests).
    pub fn from_catalog(root: PathBuf, c: &crate::catalog::Catalog) -> EvalCtx {
        let files = &c.files;
        let hwm_ms = files
            .iter()
            .filter(|f| f.kind == "samples")
            .map(|f| f.max_ts)
            .max()
            .unwrap_or(0);
        EvalCtx {
            reg_files: files
                .iter()
                .filter(|f| f.kind == crate::registry::KIND)
                .cloned()
                .collect(),
            tail_files: files
                .iter()
                .filter(|f| f.kind == "series" && !f.registered)
                .cloned()
                .collect(),
            tail_new: Default::default(),
            root,
            files: files
                .iter()
                .filter(|f| {
                    f.kind == "samples"
                        || f.kind == "series"
                        || f.kind == crate::store::SERIES_MONTH
                        || f.kind == rollup::KIND
                        || f.kind == rollup::KIND_1D
                        || f.kind == rollup::KIND_1HM
                })
                .cloned()
                .collect(),
            generation: c.generation,
            data_epoch: c.data_epoch,
            hwm_ms,
            closed_lag_ms: c.closed_lag_ms,
            sel_cache: Default::default(),
            ids_cache: Default::default(),
        }
    }
}

pub type EvalHandle = Arc<RwLock<Arc<EvalCtx>>>;

pub fn empty_handle(root: PathBuf) -> EvalHandle {
    Arc::new(RwLock::new(Arc::new(EvalCtx {
        root,
        files: vec![],
        generation: 0,
        data_epoch: 0,
        hwm_ms: 0,
        closed_lag_ms: 0,
        sel_cache: Default::default(),
        ids_cache: Default::default(),
        reg_files: vec![],
        tail_files: vec![],
        tail_new: Default::default(),
    })))
}

/// SQL run after views.sql and macros.sql: the Rust-backed functions under the macros' names.
pub const EVAL_SQL: &str = include_str!("eval.sql");

/// Generic names eval.sql binds to Rust (loaded last, after views.sql and --macros-sql).
pub const RUST_FUNCS: &[&str] = &[
    "rate",
    "irate",
    "increase",
    "delta",
    "increase_pure",
    "idelta",
    "last",
    "timestamp",
    "last_over_time",
    "tlast_over_time",
    "first_over_time",
    "avg_over_time",
    "sum_over_time",
    "min_over_time",
    "max_over_time",
    "count_over_time",
    "stddev_over_time",
    "quantile_over_time",
    "topk",
    "agg",
];
