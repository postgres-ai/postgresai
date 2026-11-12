//! Metric collector: runs metrics.yml SQL with pgwatch semantics and emits samples with the
//! names and labels pgwatch's Prometheus sink produces.
use crate::config::{CollectorCfg, Source};
use crate::metricdefs::{prom_name, Scheduled};
use crate::model::*;
use crate::store::Shared;
use crate::Msg;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Semaphore};
use tokio_postgres::types::{FromSql, Type};
use tokio_postgres::{Client, NoTls};

/// A Postgres value decoded the way pgwatch (pgx) would see it.
#[derive(Debug, Clone)]
pub enum PgVal {
    Null,
    Int(i64),
    Float(f64),
    Bool(bool),
    Numeric(f64),
    Text(String),
    /// Types pgwatch's sink never turns into a series (oid, xid, timestamps, ...);
    /// the string is used only when the column is a `tag_`.
    Other(String),
}

fn decode_numeric(raw: &[u8]) -> f64 {
    if raw.len() < 8 {
        return f64::NAN;
    }
    let nd = i16::from_be_bytes([raw[0], raw[1]]) as usize;
    let weight = i16::from_be_bytes([raw[2], raw[3]]) as i32;
    let sign = u16::from_be_bytes([raw[4], raw[5]]);
    if sign == 0xC000 {
        return f64::NAN;
    }
    if sign == 0xD000 {
        return f64::INFINITY;
    }
    if sign == 0xF000 {
        return f64::NEG_INFINITY;
    }
    let mut v = 0f64;
    for i in 0..nd {
        let o = 8 + i * 2;
        if o + 2 > raw.len() {
            break;
        }
        let d = i16::from_be_bytes([raw[o], raw[o + 1]]) as f64;
        v += d * 10000f64.powi(weight - i as i32);
    }
    if sign == 0x4000 {
        -v
    } else {
        v
    }
}

impl<'a> FromSql<'a> for PgVal {
    fn from_sql(ty: &Type, raw: &'a [u8]) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        Ok(match *ty {
            Type::INT2 => PgVal::Int(i16::from_sql(ty, raw)? as i64),
            Type::INT4 => PgVal::Int(i32::from_sql(ty, raw)? as i64),
            Type::INT8 => PgVal::Int(i64::from_sql(ty, raw)?),
            Type::FLOAT4 => PgVal::Float(f32::from_sql(ty, raw)? as f64),
            Type::FLOAT8 => PgVal::Float(f64::from_sql(ty, raw)?),
            Type::BOOL => PgVal::Bool(bool::from_sql(ty, raw)?),
            Type::NUMERIC => PgVal::Numeric(decode_numeric(raw)),
            Type::TEXT | Type::VARCHAR | Type::NAME | Type::BPCHAR | Type::UNKNOWN => {
                PgVal::Text(String::from_utf8_lossy(raw).into_owned())
            }
            Type::OID | Type::XID | Type::REGCLASS | Type::REGPROC | Type::REGTYPE => PgVal::Other(
                u32::from_sql(&Type::OID, raw)
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
            ),
            Type::XID8 | Type::PG_LSN => PgVal::Other(if raw.len() == 8 {
                u64::from_be_bytes(raw.try_into().unwrap()).to_string()
            } else {
                String::new()
            }),
            _ => PgVal::Other(String::from_utf8_lossy(raw).into_owned()),
        })
    }
    fn from_sql_null(_: &Type) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        Ok(PgVal::Null)
    }
    fn accepts(_: &Type) -> bool {
        true
    }
}

impl PgVal {
    fn as_label(&self) -> Option<String> {
        match self {
            PgVal::Null => None,
            PgVal::Int(v) => Some(v.to_string()),
            PgVal::Float(v) | PgVal::Numeric(v) => Some(v.to_string()),
            PgVal::Bool(b) => Some(b.to_string()),
            PgVal::Text(s) | PgVal::Other(s) => (!s.is_empty()).then(|| s.clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub major: u32,
    pub in_recovery: bool,
    pub sys_id: String,
}

pub async fn connect(conn_str: &str, app: &str) -> anyhow::Result<Client> {
    let mut cfg: tokio_postgres::Config = conn_str.parse()?;
    cfg.application_name(app);
    cfg.connect_timeout(Duration::from_secs(10));
    let (client, conn) = cfg.connect(NoTls).await?;
    tokio::spawn(async move {
        if let Err(e) = conn.await {
            eprintln!("pg connection closed: {e}");
        }
    });
    Ok(client)
}

/// Connection string for another database on the same node.
pub fn conn_str_for_db(base: &str, db: &str) -> anyhow::Result<String> {
    // Works for URL-style strings: replace the path.
    if let Some(rest) = base
        .strip_prefix("postgresql://")
        .or_else(|| base.strip_prefix("postgres://"))
    {
        let (hostpart, query) = match rest.split_once('?') {
            Some((h, q)) => (h, Some(q)),
            None => (rest, None),
        };
        let host = hostpart.split('/').next().unwrap_or(hostpart);
        let mut s = format!("postgresql://{host}/{db}");
        if let Some(q) = query {
            s.push('?');
            s.push_str(q);
        }
        Ok(s)
    } else {
        Ok(format!("{base} dbname={db}"))
    }
}

pub async fn node_info(c: &Client) -> anyhow::Result<NodeInfo> {
    let r = c
        .query_one(
            "select current_setting('server_version_num')::int, pg_is_in_recovery(), \
             (select system_identifier::text from pg_control_system())",
            &[],
        )
        .await?;
    let num: i32 = r.get(0);
    Ok(NodeInfo {
        major: (num / 10000) as u32,
        in_recovery: r.get(1),
        sys_id: r.get::<_, Option<String>>(2).unwrap_or_default(),
    })
}

/// What one database worker needs.
pub struct DbTarget {
    pub source_label: String, // pgwatch `dbname` label
    pub datname: String,
    pub conn_str: String,
    pub is_main: bool,
    /// Labels of the `up` series this worker writes (`up_series`): the main database of a
    /// source whose scrape labels name a job; None otherwise.
    pub up_labels: Option<Vec<(String, String)>>,
}

/// The `up{job, instance}` series a Prometheus scrape of pgwatch writes, for a source with
/// `scrape_labels`: (series id, metric, labels). Only the scrape labels, as Prometheus's
/// target labels. None without a `job` label (nothing to stand in for).
pub fn up_series(scrape_labels: &BTreeMap<String, String>) -> Option<(u64, String, Vec<(String, String)>)> {
    if !scrape_labels.contains_key("job") {
        return None;
    }
    let labels: Vec<(String, String)> = scrape_labels.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    Some((series_id("up", &labels), "up".to_string(), labels))
}

/// How often `up` is written: the interval of the shortest-interval metric this worker runs
/// (wait_events, 15 s, in the full preset). Panels that divide by `sum_over_time(up[i])`
/// (Active session history: wait-event samples per scrape) then get one `up` per collection
/// of their numerator, as one VictoriaMetrics scrape per pgwatch collection gives.
pub fn up_interval_s(metrics: &[Scheduled]) -> Option<u64> {
    metrics.iter().map(|m| m.interval_s).filter(|&i| i > 0).min()
}

pub struct Emitter {
    pub shared: Shared,
    pub tx: mpsc::Sender<Msg>,
    pub cfg: CollectorCfg,
    /// (metric group) -> series cut so far (exported as pgai_mon_series_cut_total).
    pub cut: HashMap<String, u64>,
    pub errors: HashMap<String, u64>,
    pub seen_text: std::collections::HashSet<String>,
    /// Series of each metric in its previous successful run (for staleness markers).
    pub prev: HashMap<String, std::collections::HashSet<u64>>,
}

impl Emitter {
    /// Register new series, then send the samples. Series first, so the WAL never holds a
    /// sample whose series is unknown.
    pub async fn emit(&mut self, defs: Vec<(u64, String, Vec<(String, String)>)>, ts: i64, vals: Vec<(u64, f64)>) {
        let mut new = Vec::new();
        {
            let mut k = self.shared.known_series.lock().unwrap();
            let day = day_of(ts);
            if k.len() > 100_000 && k.iter().any(|(d, _)| *d < day - 1) {
                k.retain(|(d, _)| *d >= day - 1); // older days are never written again
            }
            for (id, metric, labels) in defs {
                if k.insert((day, id)) {
                    new.push(SeriesDef {
                        series_id: id,
                        metric,
                        labels,
                        first_seen: ts,
                    });
                }
            }
        }
        if !new.is_empty() {
            let _ = self.tx.send(Msg::Rec(Record::Series(new))).await;
        }
        if !vals.is_empty() {
            let s = vals
                .into_iter()
                .map(|(series_id, value)| Sample { series_id, ts, value })
                .collect();
            let _ = self.tx.send(Msg::Rec(Record::Samples(s))).await;
        }
    }
}

/// Run one metric query and turn its rows into series/samples the pgwatch way.
pub async fn collect_metric(
    client: &Client,
    m: &Scheduled,
    sql: &str,
    base_labels: &[(String, String)],
    em: &mut Emitter,
) -> anyhow::Result<usize> {
    let t0 = std::time::Instant::now();
    let start_ms = now_ms();
    let rows = client.query(sql, &[]).await?;
    let mut ts = start_ms;
    let storage = m.def.storage_name(&m.name).to_string();
    let mut defs = Vec::new();
    let mut vals: HashMap<u64, f64> = HashMap::new();
    let mut cut = 0u64;
    for row in &rows {
        let mut labels: BTreeMap<String, String> = BTreeMap::new();
        let mut fields: Vec<(String, f64)> = Vec::new();
        for (i, col) in row.columns().iter().enumerate() {
            let name = col.name();
            let v: PgVal = row.try_get(i).unwrap_or(PgVal::Null);
            if name == "epoch_ns" {
                if let PgVal::Int(ns) = v {
                    ts = ns / 1_000_000;
                }
                continue;
            }
            if let Some(tag) = name.strip_prefix("tag_") {
                if let Some(s) = v.as_label() {
                    labels.insert(tag.to_string(), s);
                }
                continue;
            }
            match v {
                PgVal::Int(x) => fields.push((name.to_string(), x as f64)),
                PgVal::Float(x) => fields.push((name.to_string(), x)),
                PgVal::Bool(b) => fields.push((name.to_string(), if b { 1.0 } else { 0.0 })),
                PgVal::Numeric(x) if !em.cfg.drop_numeric => fields.push((name.to_string(), x)),
                PgVal::Text(t) => {
                    let class = em
                        .cfg
                        .text_columns
                        .get(&m.name)
                        .and_then(|c| c.get(name))
                        .map(|s| s.as_str());
                    match class {
                        Some("drop") => {}
                        // pgwatch parity: a non-tag text column is a label. A long value
                        // that looks like SQL is refused as a label regardless.
                        _ if t.len() > 256 && class.is_none() => {
                            if em.seen_text.insert(format!("{}.{}", m.name, name)) {
                                eprintln!("collector: {}.{}: long text not used as a label", m.name, name);
                            }
                        }
                        _ => {
                            if !t.is_empty() {
                                labels.insert(name.to_string(), t);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        // pgwatch order: dbname first, row tags, then custom tags override.
        let mut all: BTreeMap<String, String> = BTreeMap::new();
        all.insert(
            "dbname".into(),
            base_labels
                .iter()
                .find(|(k, _)| k == "dbname")
                .map(|(_, v)| v.clone())
                .unwrap_or_default(),
        );
        all.extend(labels);
        for (k, v) in base_labels {
            if k != "dbname" {
                all.insert(k.clone(), v.clone());
            }
        }
        let labels: Vec<(String, String)> = all.into_iter().collect();
        for (f, x) in fields {
            let metric = prom_name(&storage, &f);
            let id = series_id(&metric, &labels);
            if !vals.contains_key(&id) && vals.len() >= em.cfg.max_series_per_query {
                cut += 1;
                continue;
            }
            if vals.insert(id, x).is_none() {
                defs.push((id, metric, labels.clone()));
            }
        }
    }
    let n = vals.len();
    // Staleness markers, as Prometheus/VictoriaMetrics scraping writes them: a series present
    // in the previous run of this metric and absent now gets one NaN sample at this run's ts.
    let cur: std::collections::HashSet<u64> = vals.keys().copied().collect();
    let mut out: Vec<(u64, f64)> = vals.into_iter().collect();
    if let Some(prev) = em.prev.get(&m.name) {
        out.extend(prev.difference(&cur).map(|id| (*id, stale_nan())));
    }
    em.prev.insert(m.name.clone(), cur);
    em.emit(defs, ts, out).await;
    // Self-metrics, same label base.
    let mut selfm = Vec::new();
    let mut selfv = Vec::new();
    let mut lab: Vec<(String, String)> = base_labels.to_vec();
    lab.push(("metric".into(), m.name.clone()));
    lab.sort();
    let dur_id = series_id("pgai_mon_metric_duration_ms", &lab);
    selfm.push((dur_id, "pgai_mon_metric_duration_ms".to_string(), lab.clone()));
    selfv.push((dur_id, t0.elapsed().as_secs_f64() * 1000.0));
    if cut > 0 || em.cut.contains_key(&m.name) {
        let c = em.cut.entry(m.name.clone()).or_insert(0);
        *c += cut;
        let id = series_id("pgai_mon_series_cut_total", &lab);
        selfm.push((id, "pgai_mon_series_cut_total".to_string(), lab.clone()));
        selfv.push((id, *c as f64));
        if cut > 0 {
            eprintln!(
                "collector: {}: cut {cut} series over the {} cap",
                m.name, em.cfg.max_series_per_query
            );
        }
    }
    em.emit(selfm, start_ms, selfv).await;
    Ok(n)
}

enum Job {
    Metric(Scheduled),
    QueryTexts,
    Objects,
    StatsResets,
    /// `up` = 1 while the worker reaches its database (DbTarget::up_labels)
    Up,
}

/// stats_reset of every cumulative-stats view the node has, plus postmaster start, as
/// `pgai_mon_stats_reset_epoch_s{view, key}`: counters' reset points without changing
/// pgwatch's label sets (a stats_reset label would split every series at each reset).
fn stats_reset_sql(major: u32, pgss: bool) -> String {
    let mut parts = vec![
        "select 'postmaster'::text as view, ''::text as key, extract(epoch from pg_postmaster_start_time())::float8 as v".to_string(),
        "select 'pg_stat_database', coalesce(datname, ''), extract(epoch from stats_reset)::float8 from pg_stat_database where stats_reset is not null".into(),
        "select 'pg_stat_bgwriter', '', extract(epoch from stats_reset)::float8 from pg_stat_bgwriter".into(),
        "select 'pg_stat_archiver', '', extract(epoch from stats_reset)::float8 from pg_stat_archiver".into(),
        "select 'pg_stat_slru', name, extract(epoch from stats_reset)::float8 from pg_stat_slru".into(),
    ];
    if major >= 14 {
        parts.push("select 'pg_stat_wal', '', extract(epoch from stats_reset)::float8 from pg_stat_wal".into());
        parts.push("select 'pg_stat_replication_slots', slot_name::text, extract(epoch from stats_reset)::float8 from pg_stat_replication_slots where stats_reset is not null".into());
        if pgss {
            parts.push("select 'pg_stat_statements_info', '', extract(epoch from stats_reset)::float8 from pg_stat_statements_info".into());
        }
    }
    if major >= 16 {
        parts.push("select 'pg_stat_io', concat_ws('/', backend_type, object, context), extract(epoch from stats_reset)::float8 from pg_stat_io where stats_reset is not null".into());
    }
    if major >= 17 {
        parts.push(
            "select 'pg_stat_checkpointer', '', extract(epoch from stats_reset)::float8 from pg_stat_checkpointer"
                .into(),
        );
    }
    parts.join("\nunion all ")
}

async fn collect_stats_resets(
    c: &Client,
    major: u32,
    base: &[(String, String)],
    em: &mut Emitter,
) -> anyhow::Result<()> {
    let pgss = c
        .query_opt("select 1 from pg_extension where extname = 'pg_stat_statements'", &[])
        .await?
        .is_some();
    let rows = c.query(&stats_reset_sql(major, pgss), &[]).await?;
    let ts = now_ms();
    let mut defs = Vec::new();
    let mut vals = Vec::new();
    for r in rows {
        let (view, key, v): (String, String, Option<f64>) = (r.get(0), r.get(1), r.get(2));
        let Some(v) = v else { continue };
        let mut lab: Vec<(String, String)> = base.to_vec();
        lab.push(("view".into(), view));
        if !key.is_empty() {
            lab.push(("key".into(), key));
        }
        lab.sort();
        let id = series_id("pgai_mon_stats_reset_epoch_s", &lab);
        defs.push((id, "pgai_mon_stats_reset_epoch_s".to_string(), lab));
        vals.push((id, v));
    }
    em.emit(defs, ts, vals).await;
    Ok(())
}

struct Due {
    job: Job,
    interval_s: u64,
    next: tokio::time::Instant,
}

/// One worker per database. The main database runs instance-level and per-database
/// metrics plus the query-text snapshot; every database runs per-database metrics and the
/// object-definition snapshot.
pub async fn run_db_worker(
    target: DbTarget,
    metrics: Arc<Vec<Scheduled>>,
    base_labels: Vec<(String, String)>,
    mut em: Emitter,
    sessions: Arc<Semaphore>,
    persistent: bool,
    mut objects_state: HashMap<crate::structure::ObjKey, u64>,
) {
    let start = tokio::time::Instant::now();
    let mut due: Vec<Due> = metrics
        .iter()
        .filter(|m| target.is_main || !m.def.is_instance_level)
        .map(|m| Due {
            job: Job::Metric(m.clone()),
            interval_s: m.interval_s,
            next: start,
        })
        .collect();
    if target.is_main && em.cfg.query_texts_interval_s > 0 {
        due.push(Due {
            job: Job::QueryTexts,
            interval_s: em.cfg.query_texts_interval_s,
            next: start,
        });
    }
    if em.cfg.objects_interval_s > 0 {
        due.push(Due {
            job: Job::Objects,
            interval_s: em.cfg.objects_interval_s,
            next: start,
        });
    }
    if target.is_main && em.cfg.stats_reset_interval_s > 0 {
        due.push(Due {
            job: Job::StatsResets,
            interval_s: em.cfg.stats_reset_interval_s,
            next: start,
        });
    }
    if let (Some(_), Some(i)) = (&target.up_labels, up_interval_s(&metrics)) {
        due.push(Due {
            job: Job::Up,
            interval_s: i,
            next: start,
        });
    }
    let mut texts: crate::structure::TextState = HashMap::new();
    let mut role_checked: Option<tokio::time::Instant> = None;
    let mut skipped_total: HashMap<String, u64> = HashMap::new();
    let mut client: Option<(Client, NodeInfo, tokio::sync::OwnedSemaphorePermit)> = None;
    let mut cur_timeout: u64 = 0;
    loop {
        let next = due.iter().map(|d| d.next).min().unwrap();
        tokio::time::sleep_until(next).await;
        if client.is_none() {
            let permit = sessions.clone().acquire_owned().await.unwrap();
            match connect(&target.conn_str, "pgai-mon").await {
                Ok(c) => match node_info(&c).await {
                    Ok(info) => {
                        client = Some((c, info, permit));
                        cur_timeout = 0;
                    }
                    Err(e) => eprintln!("collector {}: node info: {e}", target.source_label),
                },
                Err(e) => {
                    eprintln!("collector {}: connect: {e}", target.source_label);
                }
            }
            if client.is_none() {
                for d in due.iter_mut() {
                    d.next = tokio::time::Instant::now() + Duration::from_secs(10);
                }
                continue;
            }
        }
        let now = tokio::time::Instant::now();
        let mut broken = false;
        // Re-check the node role at most every 30 s: a failover/promotion flips which
        // primary-only / standby-only metrics run, without waiting for a reconnect.
        if role_checked.map_or(true, |t| now.duration_since(t) >= Duration::from_secs(30)) {
            if let Some((c, info, _)) = client.as_mut() {
                match c.query_one("select pg_is_in_recovery()", &[]).await {
                    Ok(r) => {
                        let rec: bool = r.get(0);
                        if rec != info.in_recovery {
                            eprintln!("collector {}: role changed, in_recovery={rec}", target.source_label);
                            info.in_recovery = rec;
                        }
                        role_checked = Some(now);
                    }
                    Err(_) => broken = true,
                }
            }
        }
        let mut skipped: Vec<(String, u64)> = Vec::new();
        for d in due.iter_mut().filter(|d| d.next <= now) {
            if broken {
                break;
            }
            let mut n = 0u64;
            while d.next <= now {
                d.next += Duration::from_secs(d.interval_s.max(1));
                n += 1;
            }
            if n > 1 {
                // slots that passed while earlier work ran: skipped, and counted
                if let Job::Metric(m) = &d.job {
                    skipped.push((m.name.clone(), n - 1));
                }
            }
            let (c, info, _) = client.as_ref().unwrap();
            let (name, res) = match &d.job {
                Job::Metric(m) => {
                    match m.def.node_status.as_deref() {
                        Some("primary") if info.in_recovery => continue,
                        Some("standby") if !info.in_recovery => continue,
                        _ => {}
                    }
                    let Some(sql) = m.def.sql_for(info.major) else { continue };
                    let to = m
                        .def
                        .statement_timeout_seconds
                        .unwrap_or(em.cfg.default_statement_timeout_s);
                    if to != cur_timeout {
                        if let Err(e) = c.batch_execute(&format!("set statement_timeout = '{to}s'")).await {
                            eprintln!("collector {}: {e}", target.source_label);
                            broken = true;
                            break;
                        }
                        cur_timeout = to;
                    }
                    (
                        m.name.clone(),
                        collect_metric(c, m, sql, &base_labels, &mut em).await.map(|_| ()),
                    )
                }
                Job::QueryTexts => {
                    let r = async {
                        let v = crate::structure::snapshot_query_texts(c, &mut texts, &em.shared.fp_to_queryid).await?;
                        if !v.is_empty() {
                            let _ = em.tx.send(Msg::Rec(Record::QueryTexts(v))).await;
                        }
                        anyhow::Ok(())
                    }
                    .await;
                    ("query_texts".to_string(), r)
                }
                Job::StatsResets => (
                    "stats_resets".to_string(),
                    collect_stats_resets(c, info.major, &base_labels, &mut em).await,
                ),
                Job::Up => {
                    let (id, metric, labels) = (
                        series_id("up", target.up_labels.as_ref().unwrap()),
                        "up".to_string(),
                        target.up_labels.clone().unwrap(),
                    );
                    em.emit(vec![(id, metric, labels)], now_ms(), vec![(id, 1.0)]).await;
                    ("up".to_string(), Ok(()))
                }
                Job::Objects => {
                    let r = async {
                        let v = crate::structure::snapshot_objects(c, &target.datname, &mut objects_state).await?;
                        if !v.is_empty() {
                            let _ = em.tx.send(Msg::Rec(Record::Objects(v))).await;
                        }
                        anyhow::Ok(())
                    }
                    .await;
                    ("objects".to_string(), r)
                }
            };
            if let Err(e) = res {
                *em.errors.entry(name.clone()).or_insert(0) += 1;
                let n = em.errors[&name];
                if n <= 3 || n % 100 == 0 {
                    let detail = e
                        .downcast_ref::<tokio_postgres::Error>()
                        .and_then(|pe| pe.as_db_error())
                        .map(|d| format!("{}: {}", d.code().code(), d.message()))
                        .unwrap_or_else(|| e.to_string());
                    eprintln!("collector {} {}: {detail} (x{n})", target.source_label, name);
                }
                if c.is_closed() {
                    broken = true;
                    break;
                }
            }
        }
        if !skipped.is_empty() {
            let ts = now_ms();
            let mut defs = Vec::new();
            let mut vals = Vec::new();
            for (name, n) in skipped {
                let t = skipped_total.entry(name.clone()).or_insert(0);
                *t += n;
                let mut lab = base_labels.clone();
                lab.push(("metric".into(), name));
                lab.sort();
                let id = series_id("pgai_mon_skipped_slots_total", &lab);
                defs.push((id, "pgai_mon_skipped_slots_total".to_string(), lab));
                vals.push((id, *t as f64));
            }
            em.emit(defs, ts, vals).await;
        }
        if broken || !persistent {
            client = None; // drops connection and session permit
            role_checked = None;
        }
    }
}

/// Start the workers for one source (node).
pub async fn run_source(
    src: Source,
    metrics: Arc<Vec<Scheduled>>,
    shared: Shared,
    tx: mpsc::Sender<Msg>,
    cfg: CollectorCfg,
    mut objects: HashMap<String, HashMap<crate::structure::ObjKey, u64>>,
) -> anyhow::Result<()> {
    let main = loop {
        match connect(&src.conn_str, "pgai-mon").await {
            Ok(c) => break c,
            Err(e) => {
                eprintln!("collector {}: {e}; retrying", src.name);
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
        }
    };
    let info = node_info(&main).await?;
    let main_db: String = main.query_one("select current_database()", &[]).await?.get(0);
    let mut dbs = vec![main_db.clone()];
    if src.discover_databases {
        for r in main
            .query(
                "select datname from pg_database where datallowconn and not datistemplate \
                 and datname <> current_database() order by datname",
                &[],
            )
            .await?
        {
            dbs.push(r.get(0));
        }
    }
    drop(main);
    eprintln!(
        "collector {}: PG{} {} sys_id={} databases={:?}",
        src.name,
        info.major,
        if info.in_recovery { "standby" } else { "primary" },
        info.sys_id,
        dbs
    );
    let persistent = dbs.len() <= cfg.max_sessions_per_node;
    let sessions = Arc::new(Semaphore::new(cfg.max_sessions_per_node.max(1)));
    for (i, db) in dbs.iter().enumerate() {
        let is_main = i == 0;
        let label = if is_main {
            src.name.clone()
        } else {
            format!("{}_{}", src.name, db)
        };
        let mut base: BTreeMap<String, String> = BTreeMap::new();
        base.insert("dbname".into(), label.clone());
        base.extend(src.custom_tags.clone());
        base.insert("real_dbname".into(), db.clone());
        base.insert("sys_id".into(), info.sys_id.clone());
        base.extend(src.scrape_labels.clone());
        let target = DbTarget {
            source_label: label,
            datname: db.clone(),
            conn_str: if is_main {
                src.conn_str.clone()
            } else {
                conn_str_for_db(&src.conn_str, db)?
            },
            is_main,
            up_labels: if is_main {
                up_series(&src.scrape_labels).map(|(_, _, l)| l)
            } else {
                None
            },
        };
        let em = Emitter {
            shared: shared.clone(),
            tx: tx.clone(),
            cfg: cfg.clone(),
            cut: HashMap::new(),
            errors: HashMap::new(),
            seen_text: Default::default(),
            prev: HashMap::new(),
        };
        tokio::spawn(run_db_worker(
            target,
            metrics.clone(),
            base.into_iter().collect(),
            em,
            sessions.clone(),
            persistent || is_main,
            objects.remove(db).unwrap_or_default(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sched(name: &str, interval_s: u64) -> Scheduled {
        Scheduled {
            name: name.into(),
            interval_s,
            def: serde_yaml::from_str("sqls: {11: 'select 1'}").unwrap(),
        }
    }

    #[test]
    fn up_is_written_only_for_sources_with_a_scrape_job() {
        let mut l = BTreeMap::new();
        assert!(up_series(&l).is_none());
        l.insert("instance".to_string(), "pgwatch:9091".to_string());
        assert!(up_series(&l).is_none(), "an instance alone names no scrape job");
        l.insert("job".to_string(), "pgwatch-prometheus".to_string());
        let (id, metric, labels) = up_series(&l).unwrap();
        assert_eq!(metric, "up");
        // Prometheus's target labels only: no dbname, sys_id or custom tags
        assert_eq!(
            labels,
            vec![
                ("instance".to_string(), "pgwatch:9091".to_string()),
                ("job".to_string(), "pgwatch-prometheus".to_string())
            ]
        );
        assert_eq!(id, series_id("up", &labels));
    }

    #[test]
    fn up_follows_the_shortest_metric_interval() {
        assert_eq!(up_interval_s(&[]), None);
        assert_eq!(
            up_interval_s(&[sched("db_stats", 30), sched("wait_events", 15), sched("x", 0)]),
            Some(15)
        );
    }
}
