//! The barrier is inside DuckDB execution, not around the calls into the reader.
use super::*;
use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use std::sync::{atomic::AtomicBool, mpsc, Condvar};

struct Gate {
    entered: mpsc::Sender<i64>,
    released: Mutex<bool>,
    wake: Condvar,
}

impl Gate {
    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.wake.notify_all();
    }
}

struct Overlap;

impl VTab for Overlap {
    type BindData = (Arc<Gate>, i64);
    type InitData = AtomicBool;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        bind.add_result_column("id", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        let gate = unsafe { &*bind.get_extra_info::<Arc<Gate>>() }.clone();
        Ok((gate, bind.get_parameter(0).to_int64()))
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        init.set_max_threads(1);
        Ok(AtomicBool::new(false))
    }

    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<(), Box<dyn std::error::Error>> {
        if func.get_init_data().swap(true, Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }
        let (gate, id) = func.get_bind_data();
        gate.entered.send(*id)?;
        let released = gate.released.lock().unwrap();
        // Only a deadlock guard; success depends on both entry events before release.
        let (released, _) = gate
            .wake
            .wait_timeout_while(released, Duration::from_secs(60), |released| !*released)
            .unwrap();
        if !*released {
            return Err("execution barrier was not released".into());
        }
        unsafe { output.flat_vector(0).as_mut_slice::<i64>()[0] = *id };
        output.set_len(1);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Bigint)])
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_heavy_http_queries_enter_execution_before_either_finishes() {
    let dir = tempfile::tempdir().unwrap();
    let core = crate::store::StoreCore::open(dir.path(), &crate::config::StoreCfg::default()).unwrap();
    let cfg = HttpCfg {
        max_concurrent: 2,
        timeout_ms: 60_000,
        queue_wait_ms: 60_000,
        ..Default::default()
    };
    assert_eq!((cfg.threads, cfg.eval_workers), (1, 1));
    let reader = Arc::new(Reader::new(core.shared.clone(), cfg.clone(), EMBEDDED_VIEWS_SQL.into(), vec![]).unwrap());
    let (entered, rx) = mpsc::channel();
    let gate = Arc::new(Gate {
        entered,
        released: Mutex::new(false),
        wake: Condvar::new(),
    });
    reader
        .admin
        .lock()
        .unwrap()
        .register_table_function_with_extra_info::<Overlap, Arc<Gate>>("prom_overlap_test", &gate)
        .unwrap();
    let state = AppState::new(Backend::InProc(reader), cfg, core.shared.clone());
    let admission = state.adm.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/sql", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });
    let client = reqwest::Client::new();
    let requests: Vec<_> = (1..=2)
        .map(|id| {
            let (client, url) = (client.clone(), url.clone());
            tokio::spawn(async move {
                client
                    .post(url)
                    .body(format!("SELECT * FROM prom_overlap_test({id})"))
                    .send()
                    .await
                    .unwrap()
            })
        })
        .collect();
    let entries = tokio::task::spawn_blocking(move || {
        (0..2)
            .map(|_| rx.recv_timeout(Duration::from_secs(30)))
            .collect::<Result<Vec<_>, _>>()
    })
    .await
    .unwrap();
    let running = admission.stats();
    // Release even on failure so a serialized reader cannot hang the test runtime.
    gate.release();
    let mut responses = Vec::new();
    for request in requests {
        let response = request.await.unwrap();
        responses.push((response.status(), response.text().await.unwrap()));
    }
    server.abort();
    let mut entries = entries.expect("both queries must reach the execution barrier before either is released");
    entries.sort();
    assert_eq!(entries, [1, 2]);
    assert_eq!(running["running_heavy"], 2);
    for (status, body) in responses {
        assert_eq!(status, 200, "{body}");
        let body: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["rows"].as_array().unwrap().len(), 1);
    }
}
