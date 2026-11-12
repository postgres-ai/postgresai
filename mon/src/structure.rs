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

/// Current (latest, not dropped) object definition hashes per database, read from the store
/// at startup so a restart doesn't rewrite every object.
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
        "SELECT dbname, kind, schema, name, definition FROM read_parquet({}) \
         QUALIFY row_number() OVER (PARTITION BY dbname, kind, schema, name ORDER BY ts DESC) = 1",
        file_list_sql(&shared.root, &refs)
    ))?;
    let mut rows = st.query([])?;
    while let Some(r) = rows.next()? {
        let def: Option<String> = r.get(4)?;
        if let Some(d) = def {
            out.entry(r.get(0)?)
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
            });
            state.insert(key, (h, now));
        }
    }
    Ok(out)
}
