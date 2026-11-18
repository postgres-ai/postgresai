//! Chart encodings, limits and cache identity through the HTTP app.
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::query::{AppState, Backend, Fmt, Reader, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::StoreCore;
use std::sync::Arc;

fn reader(core: &StoreCore, cfg: HttpCfg) -> Reader {
    Reader::new(core.shared.clone(), cfg, EMBEDDED_VIEWS_SQL.into(), vec![]).unwrap()
}

#[test]
fn csv_encodes_values_and_headers() {
    let d = tempfile::tempdir().unwrap();
    let core = StoreCore::open(d.path(), &StoreCfg::default()).unwrap();
    let r = reader(&core, HttpCfg::default());
    let result = r
        .run_fmt(
            "SELECT NULL AS nil, '' AS empty, 'a,b' AS \"a,b\", 'say \"hi\"' AS quoted, \
         'one' || chr(13) || chr(10) || 'two' AS multiline, 'é' AS unicode, \
         '-9223372036854775808'::BIGINT AS integer, 18446744073709551615::UBIGINT AS unsigned, \
         1.25::DOUBLE AS float, 1e100::DOUBLE AS big, true AS yes, false AS no, \
         'NaN'::DOUBLE AS nan, 'Infinity'::DOUBLE AS inf, '-Infinity'::DOUBLE AS neg_inf, \
         TIMESTAMP '1970-01-01 00:00:01.234' AS ts, \
         '1970-01-01 00:00:02'::TIMESTAMP_S AS ts_s, \
         '1970-01-01 00:00:01.234'::TIMESTAMP_MS AS ts_ms, \
         '1970-01-01 00:00:01.234567890'::TIMESTAMP_NS AS ts_ns",
            Fmt::Csv,
        )
        .unwrap();
    let body = result.csv.as_ref().unwrap();
    assert_eq!(body, concat!(
        "nil,empty,\"a,b\",quoted,multiline,unicode,integer,unsigned,float,big,yes,no,nan,inf,neg_inf,ts,ts_s,ts_ms,ts_ns\r\n",
        ",,\"a,b\",\"say \"\"hi\"\"\",\"one\r\ntwo\",é,-9223372036854775808,18446744073709551615,1.25,1e+100,true,false,NaN,+Inf,-Inf,1234,2000,1234,1234\r\n"
    ));
    // The process protocol must preserve embedded newlines and quotes in its one-line envelope.
    let wire = format!(
        "{{\"id\":1,\"gen\":0,\"c\":1,\"rc\":0,\"ok\":{}}}",
        serde_json::to_string(&result).unwrap()
    );
    let (_, _, reply) = pgai_mon::sqlproc::parse_reply_s3(wire).unwrap();
    assert_eq!(reply.unwrap().body, *body);
    assert_eq!(
        r.run_fmt("SELECT 1 AS x WHERE false", Fmt::Csv).unwrap().csv.unwrap(),
        "x\r\n"
    );
    // One empty field is written as "": a CSV reader skips a blank line.
    let one = |sql: &str| r.run_fmt(sql, Fmt::Csv).unwrap().csv.unwrap();
    assert_eq!(
        one("SELECT * FROM (VALUES (NULL), (''), ('a')) v(x)"),
        "x\r\n\"\"\r\n\"\"\r\na\r\n"
    );
    // Infinite timestamps are not a number of milliseconds.
    assert_eq!(
        one("SELECT 'infinity'::TIMESTAMP AS a, '-infinity'::TIMESTAMP AS b, 'infinity'::TIMESTAMP_S AS c"),
        "a,b,c\r\ninfinity,-infinity,infinity\r\n"
    );
}

#[test]
fn chart_limit_includes_headers_and_json_envelope() {
    let d = tempfile::tempdir().unwrap();
    let core = StoreCore::open(d.path(), &StoreCfg::default()).unwrap();
    let cfg = HttpCfg {
        chart_max_bytes: 4,
        ..Default::default()
    };
    let r = reader(&core, cfg);
    assert_eq!(
        r.run_fmt("SELECT 1 AS long_name WHERE false", Fmt::Csv)
            .unwrap_err()
            .kind,
        "result_too_large"
    );
    assert_eq!(
        r.run_fmt("SELECT 1 AS x WHERE false", Fmt::Objects).unwrap_err().kind,
        "result_too_large"
    );
    // An exact CSV byte limit succeeds; one extra byte fails.
    assert_eq!(
        r.run_fmt("SELECT '' AS x", Fmt::Csv).unwrap_err().kind,
        "result_too_large"
    );
    assert_eq!(
        r.run_fmt("SELECT 1 AS xy WHERE false", Fmt::Csv)
            .unwrap()
            .csv
            .unwrap()
            .len(),
        4
    );
}

async fn json(response: reqwest::Response) -> serde_json::Value {
    serde_json::from_str(&response.text().await.unwrap()).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_formats_errors_limits_and_cache() {
    let d = tempfile::tempdir().unwrap();
    let core = StoreCore::open(d.path(), &StoreCfg::default()).unwrap();
    let cfg = HttpCfg {
        chart_max_bytes: 256,
        max_bytes: 2048,
        max_rows: 20,
        ..Default::default()
    }
    .resolve();
    let r = Arc::new(reader(&core, cfg.clone()));
    let mut state = AppState::new(Backend::InProc(r), cfg, core.shared.clone());
    // Enable explicitly: an inherited PGAI_QCACHE=0 must not disable this test.
    state.cache = Arc::new(pgai_mon::qcache::QueryCache::new(1 << 20, 1 << 16, "test"));
    let app = pgai_mon::query::app(state, axum::Router::new(), Arc::new(pgai_mon::auth::Auth::disabled()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let post = |fmt: &str, sql: &str| {
        client
            .post(format!("{base}/sql?format={fmt}"))
            .body(sql.to_owned())
            .send()
    };
    let sql = "SELECT 1234::BIGINT AS time, 1.25 AS value, 'IO, read' AS series";
    for served in ["miss", "hit"] {
        for fmt in ["csv", "objects"] {
            let res = post(fmt, sql).await.unwrap();
            assert_eq!(res.status(), 200);
            assert_eq!(res.headers()["x-pgai-cache"], served);
            if fmt == "csv" {
                assert_eq!(res.headers()["content-type"], "text/csv; charset=utf-8");
                assert_eq!(
                    res.text().await.unwrap(),
                    "time,value,series\r\n1234,1.25,\"IO, read\"\r\n"
                );
            } else {
                assert_eq!(res.headers()["content-type"], "application/json");
                assert_eq!(json(res).await["rows"][0]["series"], "IO, read");
            }
        }
    }
    let res = post("csv", "SELECT missing_column").await.unwrap();
    assert_eq!(res.status(), 400);
    assert_eq!(res.headers()["content-type"], "application/json");
    assert_eq!(json(res).await["error"]["type"], "binder_error");
    for fmt in ["csv", "objects"] {
        for sql in ["SELECT repeat('x', 300) AS x", "SELECT i FROM range(21) r(i)"] {
            let res = post(fmt, sql).await.unwrap();
            assert_eq!(res.status(), 413, "{fmt}: {sql}");
            assert_eq!(res.headers()["content-type"], "application/json");
            let error = json(res).await;
            assert_eq!(error["error"]["type"], "result_too_large");
            let message = error["error"]["message"].as_str().unwrap();
            for hint in [
                "http.chart_max_bytes",
                "http.max_rows",
                "time range",
                "filter",
                "GRAFANA_MEM",
            ] {
                assert!(message.contains(hint), "{message}");
            }
        }
    }
    // Infinity shows only the status line of a failed request: the error's text is there too.
    let addr = base.trim_start_matches("http://").to_owned();
    let status_line = tokio::task::spawn_blocking(move || {
        use std::io::{BufRead, BufReader, Write};
        let mut s = std::net::TcpStream::connect(addr).unwrap();
        let sql = "SELECT i FROM range(21) r(i)";
        let head = "POST /sql?format=csv HTTP/1.1\r\nHost: x\r\nConnection: close\r\n";
        write!(s, "{head}Content-Length: {}\r\n\r\n{sql}", sql.len()).unwrap();
        let mut line = String::new();
        BufReader::new(s).read_line(&mut line).unwrap();
        line
    })
    .await
    .unwrap();
    assert!(
        status_line.starts_with("HTTP/1.1 413 result_too_large: the result exceeds http.max_rows (20) "),
        "{status_line}"
    );
    assert!(status_line.contains("narrow the time range"), "{status_line}");
    let res = post("csv", "SELECT 'a' ||\n chr(233) FROM missing_table")
        .await
        .unwrap();
    let reason = res
        .extensions()
        .get::<hyper::ext::ReasonPhrase>()
        .unwrap()
        .as_bytes()
        .to_vec();
    let reason = String::from_utf8(reason).unwrap();
    assert!(reason.starts_with("catalog_error: ") && reason.len() <= 300, "{reason}");
    assert!(reason.chars().all(|c| c.is_ascii_graphic() || c == ' '), "{reason}");
    for fmt in ["rows", "columnar"] {
        let res = post(fmt, "SELECT repeat('x', 300) AS x").await.unwrap();
        assert_eq!(res.status(), 200);
        assert_eq!(json(res).await["truncated"], false);
        let res = post(fmt, "SELECT repeat('x', 3000) AS x").await.unwrap();
        assert_eq!(res.status(), 200);
        let result = json(res).await;
        assert_eq!(result["truncated"], true);
        assert_eq!(result["row_count"], 0);
        let result = json(post(fmt, "SELECT i FROM range(21) r(i)").await.unwrap()).await;
        assert_eq!(result["truncated"], true);
        assert_eq!(result["row_count"], 20);
    }
    task.abort();
}
