//! Structure snapshots: object definitions (only changes are written) and
//! pg_stat_statements query texts (deduplicated).
use crate::model::*;
use crate::store::{file_list_sql, Shared};
use std::collections::{HashMap, HashSet};
use tokio_postgres::Client;
use xxhash_rust::xxh3::xxh3_64;

pub type ObjKey = (String, String, String); // kind, schema, name

const OBJECTS_SQL: &str = r#"
with rels as (
  select c.oid, n.nspname, c.relname, c.relkind, c.reloptions, c.relpersistence
  from pg_class c join pg_namespace n on n.oid = c.relnamespace
  where n.nspname not in ('pg_catalog', 'information_schema') and n.nspname not like 'pg_toast%'
    and n.nspname not like 'pg_temp%'
)
select 'table' as kind, r.nspname, r.relname,
  'CREATE ' || case r.relpersistence when 'u' then 'UNLOGGED ' else '' end
  || case r.relkind when 'f' then 'FOREIGN TABLE ' else 'TABLE ' end
  || quote_ident(r.nspname) || '.' || quote_ident(r.relname) || ' ('
  || coalesce((select string_agg(quote_ident(a.attname) || ' ' || format_type(a.atttypid, a.atttypmod)
        || case when a.attnotnull then ' NOT NULL' else '' end
        || coalesce(' DEFAULT ' || pg_get_expr(d.adbin, d.adrelid), ''), ', ' order by a.attnum)
      from pg_attribute a left join pg_attrdef d on d.adrelid = a.attrelid and d.adnum = a.attnum
      where a.attrelid = r.oid and a.attnum > 0 and not a.attisdropped), '') || ')'
  || coalesce(' PARTITION BY ' || pg_get_partkeydef(r.oid), '')
  || coalesce(' WITH (' || array_to_string(r.reloptions, ', ') || ')', '') as def
from rels r where r.relkind in ('r', 'p', 'f')
union all
select 'index', r.nspname, r.relname, pg_get_indexdef(r.oid) from rels r where r.relkind in ('i', 'I')
union all
select 'constraint', n.nspname, cl.relname || '.' || co.conname,
  'ALTER TABLE ' || quote_ident(n.nspname) || '.' || quote_ident(cl.relname) || ' ADD CONSTRAINT '
  || quote_ident(co.conname) || ' ' || pg_get_constraintdef(co.oid)
from pg_constraint co join pg_class cl on cl.oid = co.conrelid join pg_namespace n on n.oid = cl.relnamespace
where n.nspname not in ('pg_catalog', 'information_schema') and n.nspname not like 'pg_toast%'
union all
select case r.relkind when 'v' then 'view' else 'matview' end, r.nspname, r.relname, pg_get_viewdef(r.oid)
from rels r where r.relkind in ('v', 'm')
union all
select 'sequence', r.nspname, r.relname, 'CREATE SEQUENCE ' || quote_ident(r.nspname) || '.' || quote_ident(r.relname)
from rels r where r.relkind = 'S'
union all
select 'function', n.nspname, p.proname || '(' || pg_get_function_identity_arguments(p.oid) || ')',
  pg_get_functiondef(p.oid)
from pg_proc p join pg_namespace n on n.oid = p.pronamespace
where n.nspname not in ('pg_catalog', 'information_schema') and p.prokind in ('f', 'p')
  and not exists (select 1 from pg_depend dp where dp.objid = p.oid and dp.deptype = 'e')
"#;

/// Key of one database's object state: the source name and the database name. Rows written
/// before the source was recorded have no source: their key has an empty one, so the first
/// snapshot after an upgrade writes every object of that database once more, with its source.
pub fn objects_key(source: Option<&str>, dbname: &str) -> String {
    format!("{}\u{1f}{dbname}", source.unwrap_or(""))
}

/// Current (latest, not dropped) object definition hashes per `objects_key`, read from the
/// store at startup so a restart doesn't rewrite every object.
pub fn load_current_objects(shared: &Shared) -> anyhow::Result<HashMap<String, HashMap<ObjKey, u64>>> {
    let files: Vec<_> = {
        let c = shared.catalog.lock().unwrap();
        c.files.iter().filter(|f| f.kind == "objects").cloned().collect()
    };
    let mut out: HashMap<String, HashMap<ObjKey, u64>> = HashMap::new();
    // only the collector diffs against these; a remote store is served read-only
    if files.is_empty() || crate::objstore::is_remote(&shared.root) {
        return Ok(out);
    }
    let db = duckdb::Connection::open_in_memory()?;
    db.execute_batch("SET memory_limit='64MB'; SET threads=1;")?;
    let refs: Vec<_> = files.iter().collect();
    let mut st = db.prepare(&format!(
        "SELECT dbname, kind, schema, name, definition, source FROM {} \
         QUALIFY row_number() OVER (PARTITION BY source, dbname, kind, schema, name ORDER BY ts DESC) = 1",
        crate::store::parquet_src("objects", &file_list_sql(&shared.root, &refs))
    ))?;
    let mut rows = st.query([])?;
    while let Some(r) = rows.next()? {
        let def: Option<String> = r.get(4)?;
        if let Some(d) = def {
            let source: Option<String> = r.get(5)?;
            let dbname: String = r.get(0)?;
            out.entry(objects_key(source.as_deref(), &dbname))
                .or_default()
                .insert((r.get(1)?, r.get(2)?, r.get(3)?), xxh3_64(d.as_bytes()));
        }
    }
    Ok(out)
}

/// Snapshot definitions and return only changes (new/changed/dropped). At most `max_rows`
/// definitions are read; a cut snapshot reports no drops (objects past the cut were not seen).
pub async fn snapshot_objects(
    c: &mut Client,
    src: &SourceId,
    dbname: &str,
    state: &mut HashMap<ObjKey, u64>,
    max_rows: usize,
) -> anyhow::Result<Vec<ObjectEvent>> {
    let ts = now_ms();
    let (rows, cut) = crate::collector::query_capped(c, OBJECTS_SQL, &[], max_rows).await?;
    if cut {
        eprintln!("objects {dbname}: more than {max_rows} definitions, snapshot cut (collector.objects_max_rows)");
    }
    let mut seen: HashSet<ObjKey> = HashSet::with_capacity(rows.len());
    let mut out = Vec::new();
    for r in rows {
        let key: ObjKey = (r.get(0), r.get(1), r.get(2));
        let def: Option<String> = r.get(3);
        let def = def.unwrap_or_default();
        let h = xxh3_64(def.as_bytes());
        if state.get(&key) != Some(&h) {
            out.push(ObjectEvent {
                ts,
                dbname: dbname.to_string(),
                kind: key.0.clone(),
                schema: key.1.clone(),
                name: key.2.clone(),
                definition: Some(def),
                src: src.clone(),
            });
            state.insert(key.clone(), h);
        }
        seen.insert(key);
    }
    let dropped: Vec<ObjKey> = if cut {
        Vec::new()
    } else {
        state.keys().filter(|k| !seen.contains(*k)).cloned().collect()
    };
    for k in dropped {
        state.remove(&k);
        out.push(ObjectEvent {
            ts,
            dbname: dbname.to_string(),
            kind: k.0,
            schema: k.1,
            name: k.2,
            definition: None,
            src: src.clone(),
        });
    }
    Ok(out)
}

/// (queryid, dbname) -> (text hash, last time a row was written)
pub type TextState = HashMap<(i64, String), (u64, i64)>;

/// pg_stat_statements texts: list ids without text (cheap), fetch text only for new ids,
/// and refresh last_seen at most hourly per id. Texts are pgss-normalized ($1), so no values.
pub async fn snapshot_query_texts(
    c: &Client,
    src: &SourceId,
    state: &mut TextState,
    fp: &std::sync::Mutex<HashMap<String, i64>>,
) -> anyhow::Result<Vec<QueryText>> {
    let now = now_ms();
    let ids = c
        .query(
            "select s.queryid, d.datname::text from pg_stat_statements(false) s \
             join pg_database d on d.oid = s.dbid where s.queryid is not null group by 1, 2",
            &[],
        )
        .await?;
    let mut new_ids: Vec<i64> = Vec::new();
    let mut out = Vec::new();
    for r in &ids {
        let key: (i64, String) = (r.get(0), r.get(1));
        match state.get_mut(&key) {
            None => new_ids.push(key.0),
            Some((h, last)) if now - *last >= 3_600_000 => {
                out.push(QueryText {
                    queryid: key.0,
                    dbname: key.1.clone(),
                    query_hash: *h,
                    query: None,
                    first_seen: now,
                    last_seen: now,
                    src: src.clone(),
                });
                *last = now;
            }
            _ => {}
        }
    }
    if !new_ids.is_empty() {
        let rows = c
            .query(
                "select s.queryid, d.datname::text, min(s.query) from pg_stat_statements(true) s \
                 join pg_database d on d.oid = s.dbid where s.queryid = any($1) group by 1, 2",
                &[&new_ids],
            )
            .await?;
        for r in rows {
            let key: (i64, String) = (r.get(0), r.get(1));
            if state.contains_key(&key) {
                continue;
            }
            let q: Option<String> = r.get(2);
            let q = q.unwrap_or_default();
            let h = xxh3_64(q.as_bytes());
            fp.lock().unwrap().insert(crate::logs::fingerprint_text(&q), key.0);
            out.push(QueryText {
                queryid: key.0,
                dbname: key.1.clone(),
                query_hash: h,
                query: Some(q),
                first_seen: now,
                last_seen: now,
                src: src.clone(),
            });
            state.insert(key, (h, now));
        }
    }
    Ok(out)
}

/// One `pgai-mon load --query-texts` line.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TextLine {
    queryid: i64,
    dbname: String,
    query: Option<String>,
    first_seen_ms: i64,
    last_seen_ms: i64,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    cluster: Option<String>,
    #[serde(default)]
    node_name: Option<String>,
}

/// One `pgai-mon load --objects` line (definition null = dropped).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ObjectLine {
    ts_ms: i64,
    dbname: String,
    kind: String,
    schema: String,
    name: String,
    definition: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    cluster: Option<String>,
    #[serde(default)]
    node_name: Option<String>,
}

fn read_jsonl<T: serde::de::DeserializeOwned>(
    path: &std::path::Path,
    mut f: impl FnMut(T) -> anyhow::Result<()>,
) -> anyhow::Result<u64> {
    use std::io::BufRead;
    let rd: Box<dyn BufRead> = if path.as_os_str() == "-" {
        Box::new(std::io::BufReader::new(std::io::stdin()))
    } else {
        Box::new(std::io::BufReader::new(std::fs::File::open(path)?))
    };
    let mut n = 0;
    for (i, line) in rd.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: T = serde_json::from_str(&line).map_err(|e| anyhow::anyhow!("{}:{}: {e}", path.display(), i + 1))?;
        f(v)?;
        n += 1;
    }
    Ok(n)
}

/// Load query texts and object events from JSON lines into a store (test fixtures):
///   query texts: {"queryid", "dbname", "query", "first_seen_ms", "last_seen_ms"}
///   objects:     {"ts_ms", "dbname", "kind", "schema", "name", "definition"}
/// Both optionally with "source", "cluster", "node_name". Rows go through the WAL and a
/// flush like collected ones; `compact` then compacts every day.
pub fn load_jsonl(
    store: &mut crate::store::StoreCore,
    query_texts: Option<&std::path::Path>,
    objects: Option<&std::path::Path>,
    compact: bool,
) -> anyhow::Result<serde_json::Value> {
    const CHUNK: usize = 10_000;
    let mut texts = 0;
    if let Some(p) = query_texts {
        let mut buf = Vec::new();
        texts = read_jsonl(p, |l: TextLine| {
            buf.push(QueryText {
                queryid: l.queryid,
                query_hash: xxh3_64(l.query.as_deref().unwrap_or("").as_bytes()),
                dbname: l.dbname,
                query: l.query,
                first_seen: l.first_seen_ms,
                last_seen: l.last_seen_ms,
                src: SourceId {
                    source: l.source,
                    cluster: l.cluster,
                    node_name: l.node_name,
                },
            });
            if buf.len() >= CHUNK {
                store.append(Record::QueryTexts(std::mem::take(&mut buf)))?;
                store.flush_if_full()?;
            }
            Ok(())
        })?;
        if !buf.is_empty() {
            store.append(Record::QueryTexts(buf))?;
        }
    }
    let mut objs = 0;
    if let Some(p) = objects {
        let mut buf = Vec::new();
        objs = read_jsonl(p, |l: ObjectLine| {
            buf.push(ObjectEvent {
                ts: l.ts_ms,
                dbname: l.dbname,
                kind: l.kind,
                schema: l.schema,
                name: l.name,
                definition: l.definition,
                src: SourceId {
                    source: l.source,
                    cluster: l.cluster,
                    node_name: l.node_name,
                },
            });
            if buf.len() >= CHUNK {
                store.append(Record::Objects(std::mem::take(&mut buf)))?;
                store.flush_if_full()?;
            }
            Ok(())
        })?;
        if !buf.is_empty() {
            store.append(Record::Objects(buf))?;
        }
    }
    store.flush()?;
    if compact {
        store.compactor()?.run_once(now_ms(), true)?;
    }
    Ok(serde_json::json!({"query_texts": texts, "objects": objs}))
}
