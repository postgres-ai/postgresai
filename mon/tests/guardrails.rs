//! Read-only SQL guardrails.
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::model::*;
use pgai_mon::query::{Reader, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::StoreCore;

fn reader(dir: &std::path::Path, http: HttpCfg) -> (StoreCore, Reader) {
    let mut core = StoreCore::open(dir, &StoreCfg::default()).unwrap();
    let labels = vec![("a".to_string(), "b".to_string())];
    let id = series_id("m", &labels);
    core.append(Record::Series(vec![SeriesDef {
        series_id: id,
        metric: "m".into(),
        labels,
        first_seen: 0,
    }]))
    .unwrap();
    core.append(Record::Samples(
        (0..1000)
            .map(|k| Sample {
                series_id: id,
                ts: 1_794_000_000_000 + k * 1000,
                value: k as f64,
            })
            .collect(),
    ))
    .unwrap();
    core.flush().unwrap();
    let r = Reader::new(core.shared.clone(), http, EMBEDDED_VIEWS_SQL.into(), vec![]).unwrap();
    (core, r)
}

fn err(r: &Reader, sql: &str) -> &'static str {
    match r.run(sql) {
        Ok(_) => panic!("accepted: {sql}"),
        Err(e) => e.kind,
    }
}

#[test]
fn refuses_writes_and_side_effects() {
    let d = tempfile::tempdir().unwrap();
    let (_c, r) = reader(d.path(), HttpCfg::default());
    for sql in [
        "COPY (SELECT 1) TO '/tmp/pgai_x.csv'",
        "CREATE TABLE x AS SELECT 1",
        "INSERT INTO _metric_dict VALUES (NULL, NULL, NULL, NULL, NULL, NULL)",
        "ATTACH '/tmp/pgai_x.db' AS x",
        "INSTALL httpfs",
        "LOAD httpfs",
        "SET enable_external_access = true",
        "PRAGMA enable_profiling",
        "SELECT 1; DROP VIEW samples",
        "EXPORT DATABASE '/tmp/pgai_x'",
    ] {
        assert_eq!(err(&r, sql), "not_read_only", "{sql}");
    }
    // DuckDB takes these for read-only SELECTs; they change state, run a string the check never
    // sees, or read files that are not data.
    let store = d.path().display();
    for sql in [
        format!("SELECT * FROM enable_logging(storage := 'file', storage_path := '{store}/log')"),
        "SELECT * FROM truncate_duckdb_logs()".to_string(),
        "SELECT * FROM checkpoint()".to_string(),
        "SELECT * FROM query('SELECT * FROM enable_logging()')".to_string(),
        "SELECT 1 WHERE EXISTS (SELECT * FROM disable_logging())".to_string(),
        format!("SELECT * FROM read_text('{store}/catalog.json')"),
        format!("SELECT * FROM glob('{store}/**')"),
    ] {
        assert_eq!(err(&r, &sql), "forbidden", "{sql}");
    }
    assert!(!d.path().join("log").exists());
    r.run("SELECT count(*) FROM duckdb_views()").unwrap();
    for sql in [
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM read_text('/etc/hosts')",
        "SELECT * FROM read_parquet('/tmp/*.parquet')",
        "SELECT * FROM read_json('https://example.com/x.json')",
    ] {
        let k = err(&r, sql);
        assert!(k == "forbidden" || k == "catalog_error", "{sql}: {k}");
    }
}

#[test]
fn answers_and_caps_rows() {
    let d = tempfile::tempdir().unwrap();
    let (_c, r) = reader(
        d.path(),
        HttpCfg {
            max_rows: 100,
            ..Default::default()
        },
    );
    let res = r
        .run("SELECT metric, labels['a'] AS a, count(*) AS n, max(value) AS mx FROM metrics GROUP BY ALL")
        .unwrap();
    assert_eq!(res.columns, vec!["metric", "a", "n", "mx"]);
    assert_eq!(res.rows[0].get(), r#"["m","b",1000,999.0]"#);
    let res = r.run("SELECT * FROM samples").unwrap();
    assert_eq!(res.row_count, 100);
    assert!(res.truncated);
    // the format charts read has no place for the flag: a cut result is an error there
    let e = r
        .run_fmt("SELECT * FROM samples", pgai_mon::query::Fmt::Objects)
        .err()
        .unwrap();
    assert_eq!(e.kind, "result_too_large");
    assert!(r
        .run_fmt("SELECT * FROM samples LIMIT 100", pgai_mon::query::Fmt::Objects)
        .is_ok());
}

#[test]
fn timeout_interrupts_and_reader_survives() {
    let d = tempfile::tempdir().unwrap();
    let (_c, r) = reader(
        d.path(),
        HttpCfg {
            timeout_ms: 300,
            ..Default::default()
        },
    );
    let t = std::time::Instant::now();
    let k = err(
        &r,
        "SELECT count(*) FROM range(100000000) a, range(100000) b WHERE a.range + b.range = -1",
    );
    assert_eq!(k, "timeout");
    assert!(t.elapsed().as_millis() < 3000, "{:?}", t.elapsed());
    assert!(r.run("SELECT count(*) FROM samples").is_ok());
}

#[test]
fn memory_limit_fails_the_query_not_the_process() {
    let d = tempfile::tempdir().unwrap();
    let (_c, r) = reader(
        d.path(),
        HttpCfg {
            memory_limit: "64MB".into(),
            max_temp_directory_size: "0B".into(),
            timeout_ms: 20_000,
            ..Default::default()
        },
    );
    let k = err(&r, "SELECT list(range || repeat('x', 100)) FROM range(10000000)");
    assert_eq!(k, "out_of_memory");
    assert!(r.run("SELECT count(*) FROM samples").is_ok());
}

#[test]
fn non_finite_values_are_prometheus_strings() {
    let d = tempfile::tempdir().unwrap();
    let (_c, r) = reader(d.path(), HttpCfg::default());
    let res = r
        .run("SELECT 'nan'::DOUBLE AS a, 'inf'::DOUBLE AS b, '-inf'::DOUBLE AS c, ['nan'::DOUBLE, 1.5] AS d, 'NaN Infinity' AS e")
        .unwrap();
    assert_eq!(res.rows[0].get(), r#"["NaN","+Inf","-Inf",["NaN",1.5],"NaN Infinity"]"#);
}

/// Rows are written in Rust from Arrow (rowjson.rs). Each cell must be the JSON value DuckDB's
/// own `to_json` gives (compared as parsed JSON: the number text may differ, e.g. 1e+20 vs
/// 1e20), and each type name what DESCRIBE prints.
#[test]
fn rust_row_json_matches_duckdb_to_json() {
    let d = tempfile::tempdir().unwrap();
    let (_c, r) = reader(d.path(), HttpCfg::default());
    let exprs = [
        "42::TINYINT",
        "-7::SMALLINT",
        "123456::INTEGER",
        "-9007199254740993::BIGINT",
        "18446744073709551615::UBIGINT",
        "200::UTINYINT",
        "true",
        "NULL::INTEGER",
        "0.1::DOUBLE",
        "1e20::DOUBLE",
        "1.5e-7::DOUBLE",
        "1.0::DOUBLE",
        "-0.0::DOUBLE",
        "123.456::DECIMAL(9,3)",
        "-0.05::DECIMAL(4,2)",
        "0.1::FLOAT",
        "3.25::FLOAT",
        "'he said \"hi\"\n\\ tab\t'",
        "'ünï ☃'",
        "''",
        "DATE '2026-11-09'",
        "'infinity'::DATE",
        "'-infinity'::DATE",
        "'infinity'::TIMESTAMP",
        "'-infinity'::TIMESTAMP",
        "'infinity'::TIMESTAMPTZ",
        "DATE '1969-12-31'",
        "TIMESTAMP '2026-11-09 10:00:00'",
        "TIMESTAMP '2026-11-09 10:00:00.5'",
        "TIMESTAMP '2026-11-09 10:00:00.123456'",
        "TIMESTAMPTZ '2026-11-09 10:00:00.25+00'",
        "TIMESTAMP_MS '2026-11-09 10:00:00.007'",
        "TIMESTAMP_S '1970-01-01 00:00:01'",
        "TIME '13:14:15.5'",
        "[1, 2, NULL]",
        "['a', 'b']",
        "[[1.5], []]",
        "{'x': 1, 'y': 'z'}",
        "{'Up': [1]}",
        "MAP {'k': 'v', 'q': 'r'}",
        "MAP {1: 2.5}",
        "MAP {'a': [1, 2]}",
        "to_timestamp(1794000000.123)",
        "range(3)",
    ];
    for e in exprs {
        let res = r
            .run(&format!("SELECT {e} AS v"))
            .unwrap_or_else(|x| panic!("{e}: {x:?}"));
        let got: serde_json::Value = serde_json::from_str(res.rows[0].get()).unwrap();
        let want = r
            .run(&format!("SELECT to_json({e})::VARCHAR AS j, typeof({e}) AS t"))
            .unwrap();
        let w: serde_json::Value = serde_json::from_str(want.rows[0].get()).unwrap();
        let wj: serde_json::Value = match &w[0] {
            serde_json::Value::Null => serde_json::Value::Null,
            v => serde_json::from_str(&pgai_mon::query::sanitize_non_finite(v.as_str().unwrap().to_string())).unwrap(),
        };
        assert_eq!(got[0], wj, "value of {e}: got {}", res.rows[0].get());
        assert_eq!(res.types[0], w[1].as_str().unwrap(), "type of {e}");
    }
    let res = r.run("SELECT 1 AS a, 'x' AS b, 2.5 AS c").unwrap();
    assert_eq!(res.columns, vec!["a", "b", "c"]);
    assert_eq!(res.rows[0].get(), r#"[1,"x",2.5]"#);
}
