//! pgai-mon library: everything the binary runs, exposed for integration tests.

/// The release this binary was built for: the image build passes the tag in `PGAI_MON_VERSION`;
/// a plain `cargo build` reports the crate version.
pub const VERSION: &str = match option_env!("PGAI_MON_VERSION") {
    Some(v) => v,
    None => env!("CARGO_PKG_VERSION"),
};

pub mod admission;
pub mod auth;
pub mod bucket;
pub mod catalog;
pub mod collector;
pub mod config;
pub mod eval;
pub mod hoststats;
pub mod ingest;
pub mod kmerge;
pub mod logs;
pub mod metricdefs;
pub mod model;
pub mod objstore;
pub mod pgconn;
pub mod qcache;
pub mod query;
pub mod registry;
pub mod rowcsv;
pub mod rowjson;
pub mod rules;
pub mod scrape;
pub mod selfmon;
pub mod sqlproc;
pub mod store;
pub mod structure;
pub mod tools;
pub mod wal;

/// Messages to the store thread.
pub enum Msg {
    Rec(model::Record),
    /// Append, then acknowledge once durable (backpressure for bulk producers).
    RecAck(model::Record, tokio::sync::oneshot::Sender<()>),
    Flush(Option<tokio::sync::oneshot::Sender<()>>),
}

pub mod dictionary;
pub mod readset;
