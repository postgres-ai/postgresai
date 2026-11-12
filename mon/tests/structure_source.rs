//! query_texts and objects carry the node they came from (source, cluster, node_name), and
//! stores written before those columns existed still read and compact: their rows show NULL.
use pgai_mon::config::{HttpCfg, StoreCfg};
use pgai_mon::model::*;
use pgai_mon::query::{Reader, EMBEDDED_VIEWS_SQL};
use pgai_mon::store::StoreCore;

const T0: i64 = 1_793_923_200_000; // 2026-11-10 00:00 UTC

fn rows(r: &Reader, sql: &str) -> Vec<serde_json::Value> {
    let q = r.run(sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    q.rows.iter().map(|x| serde_json::from_str(x.get()).unwrap()).collect()
}

fn sid(source: &str, cluster: &str) -> SourceId {
    SourceId {
        source: Some(source.into()),
        cluster: Some(cluster.into()),
        node_name: Some(format!("{source}-node")),
    }
}

fn text(queryid: i64, query: &str, ts: i64, src: SourceId) -> QueryText {
    QueryText {
        queryid,
        dbname: "postgres".into(),
        query_hash: xxhash_rust::xxh3::xxh3_64(query.as_bytes()),
        query: Some(query.into()),
        first_seen: ts,
        last_seen: ts,
        src,
    }
}

fn object(ts: i64, def: Option<&str>, src: SourceId) -> ObjectEvent {
    ObjectEvent {
        ts,
        dbname: "postgres".into(),
        kind: "index".into(),
        schema: "public".into(),
        name: "i1".into(),
        definition: def.map(|s| s.to_string()),
        src,
    }
}

/// Rewrite every file of a kind without the source columns: what a store written before them
/// holds.
fn strip_source_columns(core: &StoreCore, kind: &str) {
    let files: Vec<_> = core.shared.catalog.lock().unwrap().files_of(kind).cloned().collect();
    assert!(!files.is_empty());
    let db = duckdb::Connection::open_in_memory().unwrap();
    for f in files {
        let p = core.shared.root.join(&f.path);
        let tmp = p.with_extension("old.parquet");
        db.execute_batch(&format!(
            "COPY (SELECT * EXCLUDE (source, cluster, node_name) FROM read_parquet('{}')) TO '{}' (FORMAT parquet)",
            p.display(),
            tmp.display()
        ))
        .unwrap();
        std::fs::rename(&tmp, &p).unwrap();
    }
}

#[test]
fn texts_and_objects_are_scoped_by_source_and_old_files_still_read() {
    let d = tempfile::tempdir().unwrap();
    let mut core = StoreCore::open(d.path(), &StoreCfg::default()).unwrap();
    // an "old" flush: no source columns in its files
    core.append(Record::QueryTexts(vec![text(1, "select old", T0, SourceId::default())]))
        .unwrap();
    core.append(Record::Objects(vec![object(
        T0,
        Some("CREATE INDEX i1 ON t (a)"),
        SourceId::default(),
    )]))
    .unwrap();
    core.flush().unwrap();
    strip_source_columns(&core, "query_texts");
    strip_source_columns(&core, "objects");
    // two sources with the same queryid and the same index name in a database of the same name
    core.append(Record::QueryTexts(vec![
        text(1, "select a", T0 + 1000, sid("node_a", "c1")),
        text(1, "select b", T0 + 1000, sid("node_b", "c2")),
    ]))
    .unwrap();
    core.append(Record::Objects(vec![
        object(T0 + 1000, Some("CREATE INDEX i1 ON t (a, b)"), sid("node_a", "c1")),
        object(T0 + 1000, Some("CREATE INDEX i1 ON u (x)"), sid("node_b", "c2")),
        object(T0 + 2000, None, sid("node_b", "c2")),
    ]))
    .unwrap();
    core.flush().unwrap();

    let check = |stage: &str| {
        let r = Reader::new(
            core.shared.clone(),
            HttpCfg::default(),
            EMBEDDED_VIEWS_SQL.into(),
            vec![],
        )
        .unwrap();
        let t = rows(
            &r,
            "SELECT queryid, query, source, cluster, node_name FROM query_texts ORDER BY source NULLS FIRST",
        );
        assert_eq!(
            t,
            vec![
                serde_json::json!([1, "select old", null, null, null]),
                serde_json::json!([1, "select a", "node_a", "c1", "node_a-node"]),
                serde_json::json!([1, "select b", "node_b", "c2", "node_b-node"]),
            ],
            "{stage}"
        );
        assert_eq!(
            rows(
                &r,
                "SELECT query FROM query_texts WHERE source = 'node_b' AND queryid = 1"
            ),
            vec![serde_json::json!(["select b"])],
            "{stage}"
        );
        // validity ranges per source: node_b's drop does not end node_a's definition
        let o = rows(
            &r,
            "SELECT source, definition, valid_to IS NULL FROM objects ORDER BY source NULLS FIRST, valid_from",
        );
        assert_eq!(
            o,
            vec![
                serde_json::json!([null, "CREATE INDEX i1 ON t (a)", true]),
                serde_json::json!(["node_a", "CREATE INDEX i1 ON t (a, b)", true]),
                serde_json::json!(["node_b", "CREATE INDEX i1 ON u (x)", false]),
            ],
            "{stage}"
        );
        // the collector's restart state, keyed by source and database
        let cur = pgai_mon::structure::load_current_objects(&core.shared).unwrap();
        let mut keys: Vec<String> = cur.keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                pgai_mon::structure::objects_key(None, "postgres"),
                pgai_mon::structure::objects_key(Some("node_a"), "postgres"),
            ],
            "{stage}"
        );
    };
    check("old and new files side by side");
    let n = core.compactor().unwrap().run_once(now_ms(), true).unwrap();
    assert!(n > 0);
    check("after compaction");
    let files: Vec<_> = core
        .shared
        .catalog
        .lock()
        .unwrap()
        .files_of("query_texts")
        .cloned()
        .collect();
    assert_eq!(files.len(), 1, "one compacted file");
}

#[test]
fn load_query_texts_and_objects_from_json_lines() {
    let d = tempfile::tempdir().unwrap();
    let texts = d.path().join("texts.jsonl");
    let objects = d.path().join("objects.jsonl");
    std::fs::write(
        &texts,
        format!(
            "{{\"queryid\": -8854122569253870148, \"dbname\": \"postgres\", \"query\": \"select $1\", \"first_seen_ms\": {T0}, \"last_seen_ms\": {}}}\n\n\
             {{\"queryid\": 7, \"dbname\": \"app\", \"query\": \"select 2\", \"first_seen_ms\": {T0}, \"last_seen_ms\": {T0}, \"source\": \"s1\", \"cluster\": \"c\", \"node_name\": \"n\"}}\n",
            T0 + 60_000
        ),
    )
    .unwrap();
    std::fs::write(
        &objects,
        format!(
            "{{\"ts_ms\": {T0}, \"kind\": \"index\", \"schema\": \"app\", \"name\": \"t10_b\", \"definition\": \"CREATE INDEX t10_b ON app.t10 (b)\", \"dbname\": \"target_database\"}}\n\
             {{\"ts_ms\": {}, \"kind\": \"index\", \"schema\": \"app\", \"name\": \"t10_b\", \"definition\": null, \"dbname\": \"target_database\"}}\n",
            T0 + 1000
        ),
    )
    .unwrap();
    let store = d.path().join("store");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_pgai-mon"))
        .args(["load", "--store"])
        .arg(&store)
        .arg("--query-texts")
        .arg(&texts)
        .arg("--objects")
        .arg(&objects)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let summary: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        summary,
        serde_json::json!({"load_structure": {"query_texts": 2, "objects": 2}})
    );

    let core = StoreCore::open(&store, &StoreCfg::default()).unwrap();
    let r = Reader::new(
        core.shared.clone(),
        HttpCfg::default(),
        EMBEDDED_VIEWS_SQL.into(),
        vec![],
    )
    .unwrap();
    assert_eq!(
        rows(
            &r,
            "SELECT queryid, dbname, query, epoch_ms(first_seen), epoch_ms(last_seen), source FROM query_texts ORDER BY queryid"
        ),
        vec![
            serde_json::json!([-8854122569253870148i64, "postgres", "select $1", T0, T0 + 60_000, null]),
            serde_json::json!([7, "app", "select 2", T0, T0, "s1"]),
        ]
    );
    assert_eq!(
        rows(
            &r,
            "SELECT name, definition, epoch_ms(valid_from), epoch_ms(valid_to) FROM objects"
        ),
        vec![serde_json::json!([
            "t10_b",
            "CREATE INDEX t10_b ON app.t10 (b)",
            T0,
            T0 + 1000
        ])]
    );

    // a typo in a field name is an error, with the line
    std::fs::write(
        &texts,
        "{\"queryid\": 1, \"dbname\": \"x\", \"querry\": \"s\", \"first_seen_ms\": 1, \"last_seen_ms\": 1}\n",
    )
    .unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_pgai-mon"))
        .args(["load", "--no-compact", "--store"])
        .arg(&store)
        .arg("--query-texts")
        .arg(&texts)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("texts.jsonl:1"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
