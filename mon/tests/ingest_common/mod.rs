//! Shared by the HTTP tests: a store with its store thread (as `run` has it), an in-process SQL
//! reader, and the full HTTP app served on 127.0.0.1 with an ephemeral port.
#![allow(dead_code)]
use pgai_mon::auth::Auth;
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::query::{AppState, Backend, Reader, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::{Shared, StoreCore};
use pgai_mon::Msg;
use std::sync::Arc;

pub struct Server {
    pub base: String,
    pub shared: Shared,
    pub tx: tokio::sync::mpsc::Sender<Msg>,
    pub reader: Arc<Reader>,
    _store: std::thread::JoinHandle<()>,
}

/// The store thread of `run`: append every record, flush on request.
pub fn store_thread(mut core: StoreCore, mut rx: tokio::sync::mpsc::Receiver<Msg>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while let Some(m) = rx.blocking_recv() {
            match m {
                Msg::Rec(r) => core.append(r).unwrap(),
                Msg::RecAck(r, a) => {
                    core.append(r).unwrap();
                    let _ = a.send(());
                }
                Msg::Flush(a) => {
                    core.flush().unwrap();
                    if let Some(a) = a {
                        let _ = a.send(());
                    }
                }
            }
        }
    })
}

/// Open a store under `dir`, start its store thread and serve the app built by `extra`
/// (given the store's channel and shared state) behind `auth`.
pub async fn serve(
    dir: &std::path::Path,
    auth: Auth,
    extra: impl FnOnce(tokio::sync::mpsc::Sender<Msg>, Shared) -> axum::Router,
) -> Server {
    let core = StoreCore::open(dir, &StoreCfg::default()).unwrap();
    let shared = core.shared.clone();
    let (tx, rx) = tokio::sync::mpsc::channel::<Msg>(1024);
    let st = store_thread(core, rx);
    let http = HttpCfg { reader_process: false, ..Default::default() }.resolve();
    let reader = Arc::new(Reader::new(shared.clone(), http.clone(), EMBEDDED_VIEWS_SQL.into(), vec![]).unwrap());
    let state = AppState::new(Backend::InProc(reader.clone()), http, shared.clone());
    let app = pgai_mon::query::app(state, extra(tx.clone(), shared.clone()), Arc::new(auth));
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    Server { base, shared, tx, reader, _store: st }
}

impl Server {
    /// Flush the store and wait for it (ingested data becomes queryable).
    pub async fn flush(&self) {
        let (a, b) = tokio::sync::oneshot::channel();
        self.tx.send(Msg::Flush(Some(a))).await.unwrap();
        b.await.unwrap();
    }

    /// Run SQL on the reader; rows as JSON arrays.
    pub fn sql(&self, sql: &str) -> Vec<serde_json::Value> {
        let r = self.reader.run(sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
        r.rows.iter().map(|x| serde_json::from_str(x.get()).unwrap()).collect()
    }
}
