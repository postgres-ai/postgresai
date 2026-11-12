//! pgai-mon library: everything the binary runs, exposed for integration tests.
pub mod admission;
pub mod bucket;
pub mod catalog;
pub mod collector;
pub mod config;
pub mod eval;
pub mod kmerge;
pub mod logs;
pub mod metricdefs;
pub mod model;
pub mod objstore;
pub mod pgconn;
pub mod qcache;
pub mod query;
pub mod registry;
pub mod rowjson;
pub mod sqlproc;
pub mod store;
pub mod structure;
pub mod tile;
pub mod tools;
pub mod wal;

/// Messages to the store thread.
pub enum Msg {
    Rec(model::Record),
    /// Append, then acknowledge once durable (backpressure for bulk producers).
    RecAck(model::Record, tokio::sync::oneshot::Sender<()>),
    Flush(Option<tokio::sync::oneshot::Sender<()>>),
}
