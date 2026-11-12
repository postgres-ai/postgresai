//! Metric collector: runs metrics.yml SQL with pgwatch semantics and emits samples with the
//! names and labels pgwatch's Prometheus sink produces.
use crate::config::{CollectorCfg, Config, Source};
use crate::metricdefs::{prom_name, Scheduled};
use crate::model::*;
use crate::store::Shared;
use crate::Msg;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::sync::{mpsc, watch, OwnedSemaphorePermit, Semaphore};
use tokio_postgres::types::{FromSql, Type};
use tokio_postgres::Client;

/// A Postgres value decoded the way pgwatch v3.7 sees it: pgx v5 decodes each column to a Go
/// value and the Prometheus sink keeps int/int32/int64/float32/float64 and bool as samples,
/// strings as labels (whatever their length), and skips every other Go type. `tag_` columns
/// of any type become labels, formatted with Go's `%v`.
///
/// Per Postgres type (pgx v5 default type map):
/// - int4, int8, float4, float8, bool: samples; numeric (pgtype.Numeric): skipped unless
///   collector.drop_numeric is off.
/// - int2 (Go int16), oid, xid (uint32), xid8 (uint64): not samples; numbers as tags.
/// - text, varchar, name, bpchar, unknown: labels.
/// - pg_lsn: not in pgx's type map, so pgx reads it as text (`0/16B3748`): a label, as in
///   pgwatch.
/// - timestamptz, timestamp, date (Go time.Time): not samples; as tags Go's time format in
///   UTC (`2024-01-02 03:04:05.123456 +0000 UTC`), as pgwatch in a UTC container prints it.
/// - Everything else (reg* types, json, arrays, intervals, ...): not samples; as tags the raw
///   value. No metric in metrics.yml returns one of these outside a cast.
#[derive(Debug, Clone)]
pub enum PgVal {
    Null,
    Int(i64),
    Float(f64),
    Bool(bool),
    Numeric(f64),
    Text(String),
    /// Types pgwatch's sink never turns into a series; the string is used only when the
    /// column is a `tag_`.
    Other(String),
}

/// Go's `time.Time` `%v` format for a UTC time given in microseconds since 1970-01-01.
pub fn go_time_utc(micros: i64) -> String {
    let secs = micros.div_euclid(1_000_000);
    let frac = micros.rem_euclid(1_000_000);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    // civil_from_days (H. Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let mut out = format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}",
        sod / 3600,
        sod / 60 % 60,
        sod % 60
    );
    if frac != 0 {
        out.push_str(format!(".{frac:06}").trim_end_matches('0'));
    }
    out.push_str(" +0000 UTC");
    out
}

/// Go's `%v` for a float64 (`strconv` 'g' with the shortest representation: exponent form
/// below 1e-4 and from 1e6, e.g. `1.5e+06`).
pub fn go_float(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "+Inf".into() } else { "-Inf".into() };
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let e = format!("{v:e}"); // shortest round-trip digits, e.g. 1.5e-7
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if exp < -4 || exp >= 6 {
        format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        format!("{v}")
    }
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
            Type::INT2 => PgVal::Other(i16::from_sql(ty, raw)?.to_string()),
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
            Type::XID8 => PgVal::Other(if raw.len() == 8 {
                u64::from_be_bytes(raw.try_into().unwrap()).to_string()
            } else {
                String::new()
            }),
            Type::PG_LSN if raw.len() == 8 => {
                let v = u64::from_be_bytes(raw.try_into().unwrap());
                PgVal::Text(format!("{:X}/{:X}", v >> 32, v & 0xffff_ffff))
            }
            Type::TIMESTAMPTZ | Type::TIMESTAMP if raw.len() == 8 => {
                let us = i64::from_be_bytes(raw.try_into().unwrap());
                PgVal::Other(match us {
                    i64::MAX => "infinity".into(),
                    i64::MIN => "-infinity".into(),
                    // Postgres counts from 2000-01-01
                    us => go_time_utc(us + 946_684_800_000_000),
                })
            }
            Type::DATE if raw.len() == 4 => {
                let d = i32::from_be_bytes(raw.try_into().unwrap()) as i64;
                PgVal::Other(go_time_utc((d + 10_957) * 86_400_000_000))
            }
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
            PgVal::Float(v) | PgVal::Numeric(v) => Some(go_float(*v)),
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

pub const APPLICATION_NAME: &str = "pgai-mon";

/// Connect (libpq-style string, TLS per sslmode: pgconn.rs) and apply the session settings.
pub async fn connect(conn_str: &str, cc: &CollectorCfg) -> anyhow::Result<Client> {
    let client = crate::pgconn::connect(conn_str, APPLICATION_NAME).await?;
    setup_session(&client, cc).await?;
    Ok(client)
}

/// Session settings for monitoring a production database, in one round trip:
/// read-only transactions, a lock_timeout, a pinned search_path and the default
/// statement_timeout (every job sets its own before it runs, see `set_timeout`).
pub async fn setup_session(c: &Client, cc: &CollectorCfg) -> anyhow::Result<()> {
    let sql = "select pg_catalog.set_config('default_transaction_read_only', 'on', false), \
         pg_catalog.set_config('lock_timeout', $1, false), \
         pg_catalog.set_config('statement_timeout', $2, false), \
         pg_catalog.set_config('application_name', $3, false), \
         pg_catalog.set_config('search_path', coalesce($4, 'pg_catalog' || coalesce(( \
             select ', ' || pg_catalog.string_agg(pg_catalog.quote_ident(nspname), ', ' order by nspname) \
             from (select distinct n.nspname::text as nspname from pg_catalog.pg_extension e \
                   join pg_catalog.pg_namespace n on n.oid = e.extnamespace \
                   where n.nspname not in ('pg_catalog', 'public')) s), '') || ', public'), false)";
    let lock = format!("{}ms", cc.lock_timeout_ms);
    let st = format!("{}s", cc.default_statement_timeout_s);
    c.query_one(sql, &[&lock, &st, &APPLICATION_NAME, &cc.search_path])
        .await?;
    Ok(())
}

/// The statement_timeout a connection was last set to (a SET only when it changes).
pub async fn set_timeout(c: &Client, cur: &mut u64, secs: u64) -> anyhow::Result<()> {
    if *cur != secs {
        c.batch_execute(&format!("set statement_timeout = '{secs}s'")).await?;
        *cur = secs;
    }
    Ok(())
}

/// Run one query in a read-only transaction and fetch at most `cap` rows. The rows come
/// through a portal with a row limit, so the server stops at the cap instead of the client
/// buffering the whole result. Returns the rows and whether more were left unread.
pub async fn query_capped(
    c: &mut Client,
    sql: &str,
    params: &[&(dyn tokio_postgres::types::ToSql + Sync)],
    cap: usize,
) -> anyhow::Result<(Vec<tokio_postgres::Row>, bool)> {
    let tx = c.build_transaction().read_only(true).start().await?;
    let portal = tx.bind(sql, params).await?;
    let limit = cap.saturating_add(1).min(i32::MAX as usize) as i32;
    let mut rows = tx.query_portal(&portal, limit).await?;
    let cut = rows.len() > cap;
    rows.truncate(cap);
    drop(portal);
    tx.commit().await?;
    Ok((rows, cut))
}

/// Connection string for another database on the same node (key=value form; every other
/// option, TLS included, kept).
pub fn conn_str_for_db(base: &str, db: &str) -> anyhow::Result<String> {
    Ok(crate::pgconn::ConnParams::parse(base)?.with_dbname(db).to_kv_string())
}

pub async fn node_info(c: &Client) -> anyhow::Result<NodeInfo> {
    let r = c
        .query_one(
            // pg_control_system() needs EXECUTE (superuser or an explicit grant): without it
            // sys_id is empty instead of the source failing.
            "select pg_catalog.current_setting('server_version_num')::int, pg_catalog.pg_is_in_recovery(), \
             case when pg_catalog.has_function_privilege('pg_catalog.pg_control_system()', 'execute') \
             then (select system_identifier::text from pg_catalog.pg_control_system()) end",
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

/// Sends what the collector produces to the store thread.
#[derive(Clone)]
pub struct Emitter {
    pub shared: Shared,
    pub tx: mpsc::Sender<Msg>,
    pub cfg: CollectorCfg,
}

impl Emitter {
    pub fn new(shared: Shared, tx: mpsc::Sender<Msg>, cfg: CollectorCfg) -> Self {
        Self { shared, tx, cfg }
    }

    /// Register new series, then send the samples. Series first, so the WAL never holds a
    /// sample whose series is unknown.
    pub async fn emit(&self, defs: Vec<(u64, String, Vec<(String, String)>)>, ts: i64, vals: Vec<(u64, f64)>) {
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
        if !new.is_empty() && self.tx.send(Msg::Rec(Record::Series(new))).await.is_err() {
            return; // store thread gone: the process is shutting down
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

/// What one metric keeps between its runs.
#[derive(Debug, Default)]
pub struct MetricState {
    /// Series of the previous successful run (for staleness markers).
    pub prev: HashSet<u64>,
    /// Series cut over max_series_per_query so far (pgai_mon_series_cut_total).
    pub cut: u64,
    /// Runs whose result hit max_rows_per_query (pgai_mon_rows_cut_total).
    pub rows_cut: u64,
}

/// One row turned into labels and numeric fields, the way pgwatch's Prometheus sink does it.
pub fn row_fields(
    metric: &str,
    cols: &[(&str, PgVal)],
    text_columns: &BTreeMap<String, BTreeMap<String, String>>,
    drop_numeric: bool,
    ts: &mut i64,
) -> (BTreeMap<String, String>, Vec<(String, f64)>) {
    let mut labels: BTreeMap<String, String> = BTreeMap::new();
    let mut fields: Vec<(String, f64)> = Vec::new();
    for (name, v) in cols {
        let name = *name;
        if name == "epoch_ns" {
            if let PgVal::Int(ns) = v {
                *ts = ns / 1_000_000;
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
            PgVal::Int(x) => fields.push((name.to_string(), *x as f64)),
            PgVal::Float(x) => fields.push((name.to_string(), *x)),
            PgVal::Bool(b) => fields.push((name.to_string(), if *b { 1.0 } else { 0.0 })),
            PgVal::Numeric(x) if !drop_numeric => fields.push((name.to_string(), *x)),
            PgVal::Text(t) => {
                let class = text_columns.get(metric).and_then(|c| c.get(name)).map(|s| s.as_str());
                if class != Some("drop") && !t.is_empty() {
                    labels.insert(name.to_string(), t.clone());
                }
            }
            _ => {}
        }
    }
    (labels, fields)
}

/// The label set of a row's series: dbname first, the row's tags, then the base labels
/// (custom tags, real_dbname, sys_id, scrape labels) override, as pgwatch orders them.
pub fn series_labels(base_labels: &[(String, String)], row: BTreeMap<String, String>) -> Vec<(String, String)> {
    let mut all: BTreeMap<String, String> = BTreeMap::new();
    all.insert(
        "dbname".into(),
        base_labels
            .iter()
            .find(|(k, _)| k == "dbname")
            .map(|(_, v)| v.clone())
            .unwrap_or_default(),
    );
    all.extend(row);
    for (k, v) in base_labels {
        if k != "dbname" {
            all.insert(k.clone(), v.clone());
        }
    }
    all.into_iter().collect()
}

/// Run one metric query and turn its rows into series/samples the pgwatch way.
pub async fn collect_metric(
    client: &mut Client,
    m: &Scheduled,
    sql: &str,
    base_labels: &[(String, String)],
    em: &Emitter,
    st: &mut MetricState,
) -> anyhow::Result<usize> {
    let t0 = std::time::Instant::now();
    let start_ms = now_ms();
    let (rows, rows_cut) = query_capped(client, sql, &[], em.cfg.max_rows_per_query).await?;
    let mut ts = start_ms;
    let storage = m.def.storage_name(&m.name).to_string();
    let mut defs = Vec::new();
    let mut vals: HashMap<u64, f64> = HashMap::new();
    let mut cut = 0u64;
    for row in &rows {
        let cols: Vec<(&str, PgVal)> = row
            .columns()
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name(), row.try_get(i).unwrap_or(PgVal::Null)))
            .collect();
        let (labels, fields) = row_fields(&m.name, &cols, &em.cfg.text_columns, em.cfg.drop_numeric, &mut ts);
        let labels = series_labels(base_labels, labels);
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
    let cur: HashSet<u64> = vals.keys().copied().collect();
    let mut out: Vec<(u64, f64)> = vals.into_iter().collect();
    out.extend(st.prev.difference(&cur).map(|id| (*id, stale_nan())));
    st.prev = cur;
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
    if cut > 0 || st.cut > 0 {
        st.cut += cut;
        let id = series_id("pgai_mon_series_cut_total", &lab);
        selfm.push((id, "pgai_mon_series_cut_total".to_string(), lab.clone()));
        selfv.push((id, st.cut as f64));
        if cut > 0 {
            eprintln!(
                "collector: {}: cut {cut} series over the {} cap",
                m.name, em.cfg.max_series_per_query
            );
        }
    }
    if rows_cut || st.rows_cut > 0 {
        st.rows_cut += rows_cut as u64;
        let id = series_id("pgai_mon_rows_cut_total", &lab);
        selfm.push((id, "pgai_mon_rows_cut_total".to_string(), lab.clone()));
        selfv.push((id, st.rows_cut as f64));
        if rows_cut {
            eprintln!(
                "collector: {}: result cut at {} rows (collector.max_rows_per_query)",
                m.name, em.cfg.max_rows_per_query
            );
        }
    }
    em.emit(selfm, start_ms, selfv).await;
    Ok(n)
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
    c: &mut Client,
    major: u32,
    base: &[(String, String)],
    em: &Emitter,
) -> anyhow::Result<()> {
    let pgss = c
        .query_opt("select 1 from pg_extension where extname = 'pg_stat_statements'", &[])
        .await?
        .is_some();
    let (rows, _) = query_capped(c, &stats_reset_sql(major, pgss), &[], 100_000).await?;
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

// ---------------------------------------------------------------------------------------
// Status: what /health and the collector's own series report.

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct JobStatus {
    pub runs: u64,
    pub errors: u64,
    /// statement_timeout (57014) and lock_timeout (55P03) errors, also counted in errors.
    pub timeouts: u64,
    /// Slots not run: the previous run was still going, or its time passed while waiting.
    pub skipped: u64,
    pub last_success_ms: Option<i64>,
    pub last_duration_ms: Option<u64>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct DbStatus {
    pub datname: Option<String>,
    /// A job reached the database at its last attempt.
    pub up: bool,
    pub last_success_ms: Option<i64>,
    pub connect_failures: u64,
    pub last_error: Option<String>,
    pub last_error_ms: Option<i64>,
    pub jobs: BTreeMap<String, JobStatus>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SourceStatus {
    /// starting (no successful connection yet), running, stopped
    pub state: String,
    pub major: Option<u32>,
    pub in_recovery: Option<bool>,
    pub sys_id: Option<String>,
    pub metrics: usize,
    pub standby_metrics: usize,
    /// by `dbname` label
    pub databases: BTreeMap<String, DbStatus>,
}

static STATUS: OnceLock<Mutex<BTreeMap<String, SourceStatus>>> = OnceLock::new();

fn status() -> &'static Mutex<BTreeMap<String, SourceStatus>> {
    STATUS.get_or_init(Default::default)
}

fn with_source<R>(src: &str, f: impl FnOnce(&mut SourceStatus) -> R) -> R {
    f(status().lock().unwrap().entry(src.to_string()).or_default())
}

fn with_db<R>(src: &str, label: &str, f: impl FnOnce(&mut DbStatus) -> R) -> R {
    with_source(src, |s| f(s.databases.entry(label.to_string()).or_default()))
}

/// A copy of one source's status (tests, /health).
pub fn source_status(name: &str) -> Option<SourceStatus> {
    status().lock().unwrap().get(name).cloned()
}

/// The collector's part of /health: every source, its databases and jobs.
pub fn health_json() -> serde_json::Value {
    let st = status().lock().unwrap();
    let up = st.values().flat_map(|s| s.databases.values()).filter(|d| d.up).count();
    let dbs = st.values().map(|s| s.databases.len()).sum::<usize>();
    serde_json::json!({
        "sources": st.len(),
        "databases": dbs,
        "databases_up": up,
        "by_source": &*st,
    })
}

/// SQLSTATEs counted as timeouts: statement_timeout / cancel, lock_timeout.
fn is_timeout(e: &anyhow::Error) -> bool {
    pg_code(e).is_some_and(|c| c == "57014" || c == "55P03")
}

fn pg_code(e: &anyhow::Error) -> Option<String> {
    e.downcast_ref::<tokio_postgres::Error>()
        .and_then(|pe| pe.as_db_error())
        .map(|d| d.code().code().to_string())
}

fn err_text(e: &anyhow::Error) -> String {
    e.downcast_ref::<tokio_postgres::Error>()
        .and_then(|pe| pe.as_db_error())
        .map(|d| format!("{}: {}", d.code().code(), d.message()))
        .unwrap_or_else(|| format!("{e:#}"))
}

// ---------------------------------------------------------------------------------------
// Connections: one pool per node, shared by its databases.

/// A collector session: its client and the statement_timeout it was last set to.
pub struct PgConn {
    pub client: Client,
    timeout_s: u64,
}

impl PgConn {
    pub async fn open(conn_str: &str, cc: &CollectorCfg) -> anyhow::Result<Self> {
        let client = tokio::time::timeout(Duration::from_secs(30), connect(conn_str, cc))
            .await
            .map_err(|_| anyhow::anyhow!("connect: timed out"))??;
        Ok(Self {
            client,
            timeout_s: cc.default_statement_timeout_s,
        })
    }

    pub async fn set_timeout(&mut self, secs: u64) -> anyhow::Result<()> {
        set_timeout(&self.client, &mut self.timeout_s, secs).await
    }
}

struct Idle {
    conn_str: String,
    conn: PgConn,
    permit: OwnedSemaphorePermit,
    since: tokio::time::Instant,
}

/// Sessions to one node: at most `max_sessions_per_node` across its databases. An idle
/// session is reused by its database; when the cap is reached, the session idle longest (of
/// another database) is closed for the one that needs it. Sessions idle for 10 minutes are
/// closed. A database whose connect failed is not retried before its backoff passes
/// (exponential, with jitter).
pub struct NodePool {
    cfg: CollectorCfg,
    sem: Arc<Semaphore>,
    /// Jobs with a long statement_timeout (or the object snapshot) take one of these first:
    /// at most cap - 1 at a time, so a short job always finds a session.
    heavy: Arc<Semaphore>,
    idle: Mutex<Vec<Idle>>,
    notify: tokio::sync::Notify,
    backoff: Mutex<HashMap<String, (u32, tokio::time::Instant)>>,
}

const IDLE_CLOSE: Duration = Duration::from_secs(600);

/// A session lent to a job. Back to the pool when dropped, unless the connection broke.
pub struct Lease {
    pool: Arc<NodePool>,
    conn_str: String,
    conn: Option<PgConn>,
    permit: Option<OwnedSemaphorePermit>,
}

impl std::ops::Deref for Lease {
    type Target = PgConn;
    fn deref(&self) -> &PgConn {
        self.conn.as_ref().unwrap()
    }
}
impl std::ops::DerefMut for Lease {
    fn deref_mut(&mut self) -> &mut PgConn {
        self.conn.as_mut().unwrap()
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let (Some(conn), Some(permit)) = (self.conn.take(), self.permit.take()) {
            if !conn.client.is_closed() {
                self.pool.idle.lock().unwrap().push(Idle {
                    conn_str: std::mem::take(&mut self.conn_str),
                    conn,
                    permit,
                    since: tokio::time::Instant::now(),
                });
            }
        }
        self.pool.notify.notify_waiters();
    }
}

/// Backoff before connect attempt n (1-based): min * 2^(n-1), at most max, times a random
/// factor in [0.5, 1).
pub fn backoff_delay(min_s: u64, max_s: u64, attempt: u32) -> Duration {
    let base = (min_s.max(1) as f64) * 2f64.powi(attempt.saturating_sub(1).min(30) as i32);
    let capped = base.min(max_s.max(min_s).max(1) as f64);
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u32(attempt);
    let r = (h.finish() % 1_000_000) as f64 / 1_000_000.0;
    Duration::from_secs_f64(capped * (0.5 + 0.5 * r))
}

enum Take {
    Idle(Idle),
    Permit(OwnedSemaphorePermit),
}

impl NodePool {
    pub fn new(cfg: CollectorCfg) -> Arc<Self> {
        let cap = cfg.max_sessions_per_node.max(1);
        Arc::new(Self {
            sem: Arc::new(Semaphore::new(cap)),
            heavy: Arc::new(Semaphore::new(cap.saturating_sub(1).max(1))),
            cfg,
            idle: Mutex::new(Vec::new()),
            notify: tokio::sync::Notify::new(),
            backoff: Mutex::new(HashMap::new()),
        })
    }

    fn try_take(&self, conn_str: &str) -> Option<Take> {
        let mut idle = self.idle.lock().unwrap();
        idle.retain(|i| !i.conn.client.is_closed() && i.since.elapsed() < IDLE_CLOSE);
        if let Some(pos) = idle.iter().rposition(|i| i.conn_str == conn_str) {
            return Some(Take::Idle(idle.swap_remove(pos)));
        }
        if let Ok(p) = self.sem.clone().try_acquire_owned() {
            return Some(Take::Permit(p));
        }
        // close the session idle longest and use its slot
        let oldest = idle.iter().enumerate().min_by_key(|(_, i)| i.since).map(|(n, _)| n)?;
        let i = idle.swap_remove(oldest);
        Some(Take::Permit(i.permit))
    }

    /// When the database may be tried again (None: now).
    pub fn backing_off(&self, conn_str: &str) -> Option<tokio::time::Instant> {
        let b = self.backoff.lock().unwrap();
        b.get(conn_str)
            .map(|(_, t)| *t)
            .filter(|t| *t > tokio::time::Instant::now())
    }

    pub async fn acquire(self: &Arc<Self>, conn_str: &str) -> anyhow::Result<Lease> {
        let take = loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(t) = self.try_take(conn_str) {
                break t;
            }
            notified.await;
        };
        let permit = match take {
            Take::Idle(i) => {
                return Ok(Lease {
                    pool: self.clone(),
                    conn_str: i.conn_str,
                    conn: Some(i.conn),
                    permit: Some(i.permit),
                })
            }
            Take::Permit(p) => p,
        };
        if let Some(t) = self.backing_off(conn_str) {
            drop(permit);
            self.notify.notify_waiters();
            anyhow::bail!(
                "not reconnecting for another {:.0} s (backoff)",
                t.saturating_duration_since(tokio::time::Instant::now()).as_secs_f64()
            );
        }
        match PgConn::open(conn_str, &self.cfg).await {
            Ok(conn) => {
                self.backoff.lock().unwrap().remove(conn_str);
                Ok(Lease {
                    pool: self.clone(),
                    conn_str: conn_str.to_string(),
                    conn: Some(conn),
                    permit: Some(permit),
                })
            }
            Err(e) => {
                {
                    let mut b = self.backoff.lock().unwrap();
                    let n = b.get(conn_str).map_or(0, |(n, _)| *n) + 1;
                    let d = backoff_delay(self.cfg.reconnect_min_s, self.cfg.reconnect_max_s, n);
                    b.insert(conn_str.to_string(), (n, tokio::time::Instant::now() + d));
                }
                drop(permit);
                self.notify.notify_waiters();
                Err(e)
            }
        }
    }

    /// Sessions open now (idle + lent): for tests.
    pub fn open_sessions(&self) -> usize {
        self.cfg.max_sessions_per_node.max(1) - self.sem.available_permits()
    }
}

// ---------------------------------------------------------------------------------------
// Sources and their databases.

/// Node facts every database worker of a source reads; workers wake when they change.
pub struct NodeState {
    tx: watch::Sender<Option<NodeInfo>>,
}

impl Default for NodeState {
    fn default() -> Self {
        Self {
            tx: watch::channel(None).0,
        }
    }
}

impl NodeState {
    pub fn get(&self) -> Option<NodeInfo> {
        self.tx.borrow().clone()
    }
    fn set(&self, i: NodeInfo) {
        self.tx.send_if_modified(|cur| {
            let changed = cur.as_ref().map_or(true, |c| {
                c.in_recovery != i.in_recovery || c.major != i.major || c.sys_id != i.sys_id
            });
            *cur = Some(i);
            changed
        });
    }
}

/// A source's identity as its series label it: `dbname` (the source name), and the
/// `cluster` / `node_name` custom tags.
pub fn source_id(src: &Source) -> SourceId {
    SourceId {
        source: Some(src.name.clone()),
        cluster: src.custom_tags.get("cluster").cloned(),
        node_name: src.custom_tags.get("node_name").cloned(),
    }
}

/// What the collector shares across sources.
#[derive(Clone)]
pub struct CollectorCtx {
    pub em: Emitter,
    /// structure::objects_key(source, datname) -> object definition hashes (snapshots diff
    /// against them).
    pub objects: Arc<Mutex<HashMap<String, HashMap<crate::structure::ObjKey, u64>>>>,
}

/// One database of a source.
pub struct DbCtx {
    pub src: Source,
    /// pgwatch `dbname` label: the source name, or `<source>_<datname>` for a discovered one.
    pub label: String,
    pub conn_str: String,
    pub is_main: bool,
    pub datname: std::sync::RwLock<Option<String>>,
    pub node: Arc<NodeState>,
    pub pool: Arc<NodePool>,
    pub ctx: CollectorCtx,
    pub primary: Arc<Vec<Scheduled>>,
    pub standby: Arc<Vec<Scheduled>>,
}

impl DbCtx {
    /// Labels of every series of this database; None until the node has answered once.
    fn base_labels(&self) -> Option<Vec<(String, String)>> {
        let info = self.node.get()?;
        let datname = self.datname.read().unwrap().clone()?;
        let mut base: BTreeMap<String, String> = BTreeMap::new();
        base.insert("dbname".into(), self.label.clone());
        base.extend(self.src.custom_tags.clone());
        base.insert("real_dbname".into(), datname);
        base.insert("sys_id".into(), info.sys_id);
        base.extend(self.src.scrape_labels.clone());
        Some(base.into_iter().collect())
    }

    /// The node identity structure rows (query_texts, objects) carry.
    fn source_id(&self) -> SourceId {
        source_id(&self.src)
    }

    fn status<R>(&self, f: impl FnOnce(&mut DbStatus) -> R) -> R {
        with_db(&self.src.name, &self.label, f)
    }
}

#[derive(Clone)]
enum JobKind {
    Metric(Scheduled),
    QueryTexts,
    Objects,
    StatsResets,
    /// `up` = 1, as the VictoriaMetrics scrape of pgwatch wrote it (up_series).
    Up(Vec<(String, String)>),
    /// The collector's own series for this database.
    SelfMetrics,
}

impl JobKind {
    fn name(&self) -> String {
        match self {
            JobKind::Metric(m) => m.name.clone(),
            JobKind::QueryTexts => "query_texts".into(),
            JobKind::Objects => "objects".into(),
            JobKind::StatsResets => "stats_resets".into(),
            JobKind::Up(_) => "up".into(),
            JobKind::SelfMetrics => "self_metrics".into(),
        }
    }
}

enum JobState {
    None,
    Metric(MetricState),
    Texts(crate::structure::TextState),
}

struct Slot {
    kind: JobKind,
    interval_s: u64,
    next: tokio::time::Instant,
    state: Arc<tokio::sync::Mutex<JobState>>,
}

/// The statement_timeout a job runs under, and whether it is heavy (takes a heavy permit).
fn job_timeout(kind: &JobKind, cfg: &CollectorCfg) -> (u64, bool) {
    match kind {
        JobKind::Metric(m) => {
            let t = m
                .def
                .statement_timeout_seconds
                .unwrap_or(cfg.default_statement_timeout_s);
            (t, t > cfg.default_statement_timeout_s)
        }
        JobKind::Objects => (cfg.objects_statement_timeout_s, true),
        _ => (cfg.default_statement_timeout_s, false),
    }
}

/// The metric list for a role: a standby runs the standby list when it has one.
fn metrics_for_role<'a>(db: &'a DbCtx, in_recovery: bool) -> &'a [Scheduled] {
    if in_recovery && !db.standby.is_empty() {
        &db.standby
    } else {
        &db.primary
    }
}

fn build_slots(
    db: &DbCtx,
    in_recovery: bool,
    states: &mut HashMap<String, Arc<tokio::sync::Mutex<JobState>>>,
) -> Vec<Slot> {
    let cfg = &db.ctx.em.cfg;
    let start = tokio::time::Instant::now();
    let metrics = metrics_for_role(db, in_recovery);
    let mut kinds: Vec<(JobKind, u64)> = metrics
        .iter()
        .filter(|m| db.is_main || !m.def.is_instance_level)
        .map(|m| (JobKind::Metric(m.clone()), m.interval_s))
        .collect();
    if db.is_main && cfg.query_texts_interval_s > 0 {
        kinds.push((JobKind::QueryTexts, cfg.query_texts_interval_s));
    }
    if cfg.objects_interval_s > 0 {
        kinds.push((JobKind::Objects, cfg.objects_interval_s));
    }
    if db.is_main && cfg.stats_reset_interval_s > 0 {
        kinds.push((JobKind::StatsResets, cfg.stats_reset_interval_s));
    }
    if db.is_main {
        if let (Some((_, _, l)), Some(i)) = (up_series(&db.src.scrape_labels), up_interval_s(metrics)) {
            kinds.push((JobKind::Up(l), i));
        }
    }
    if cfg.self_metrics_interval_s > 0 {
        kinds.push((JobKind::SelfMetrics, cfg.self_metrics_interval_s));
    }
    kinds
        .into_iter()
        .map(|(kind, interval_s)| {
            let state = states
                .entry(kind.name())
                .or_insert_with(|| {
                    Arc::new(tokio::sync::Mutex::new(match &kind {
                        JobKind::Metric(_) => JobState::Metric(MetricState::default()),
                        JobKind::QueryTexts => JobState::Texts(HashMap::new()),
                        _ => JobState::None,
                    }))
                })
                .clone();
            Slot {
                kind,
                interval_s,
                next: start,
                state,
            }
        })
        .collect()
}

/// One database's scheduler. Every job runs as its own task, so a slow metric never delays
/// a fast one; a job whose previous run is still going skips its slot (counted).
async fn run_db(db: Arc<DbCtx>, mut stop: watch::Receiver<bool>) {
    let mut states: HashMap<String, Arc<tokio::sync::Mutex<JobState>>> = HashMap::new();
    let mut node_rx = db.node.tx.subscribe();
    let mut known = db.node.get().is_some();
    let mut role = db.node.get().map(|i| i.in_recovery).unwrap_or(false);
    let mut slots = build_slots(&db, role, &mut states);
    loop {
        let next = slots
            .iter()
            .map(|s| s.next)
            .min()
            .unwrap_or_else(|| tokio::time::Instant::now() + Duration::from_secs(60));
        tokio::select! {
            _ = stop.changed() => {}
            _ = node_rx.changed() => {}
            _ = tokio::time::sleep_until(next) => {}
        }
        // a dropped sender stops the worker too
        if *stop.borrow() || stop.has_changed().is_err() {
            return;
        }
        if let Some(i) = db.node.get() {
            if !known {
                // the node answered for the first time: run everything now
                known = true;
                role = i.in_recovery;
                slots = build_slots(&db, role, &mut states);
            } else if i.in_recovery != role {
                role = i.in_recovery;
                if !db.standby.is_empty() {
                    eprintln!(
                        "collector {}: now a {}: running the {} metric list",
                        db.label,
                        if role { "standby" } else { "primary" },
                        if role { "standby" } else { "primary" }
                    );
                    slots = build_slots(&db, role, &mut states);
                }
            }
        }
        let now = tokio::time::Instant::now();
        for s in slots.iter_mut().filter(|s| s.next <= now) {
            let mut missed = 0u64;
            while s.next <= now {
                s.next += Duration::from_secs(s.interval_s.max(1));
                missed += 1;
            }
            let name = s.kind.name();
            let mut skipped = missed - 1;
            match s.state.clone().try_lock_owned() {
                Ok(guard) => {
                    tokio::spawn(run_job(db.clone(), s.kind.clone(), guard));
                }
                Err(_) => skipped += 1, // the previous run is still going
            }
            if skipped > 0 && matches!(s.kind, JobKind::Metric(_)) {
                db.status(|d| d.jobs.entry(name).or_default().skipped += skipped);
            }
        }
    }
}

async fn run_job(db: Arc<DbCtx>, kind: JobKind, mut state: tokio::sync::OwnedMutexGuard<JobState>) {
    let em = &db.ctx.em;
    match &kind {
        JobKind::Up(labels) => {
            let id = series_id("up", labels);
            em.emit(vec![(id, "up".into(), labels.clone())], now_ms(), vec![(id, 1.0)])
                .await;
            return;
        }
        JobKind::SelfMetrics => {
            emit_self_metrics(&db).await;
            return;
        }
        _ => {}
    }
    let Some(info) = db.node.get() else { return };
    let Some(base) = db.base_labels() else { return };
    if db.src.only_if_master && info.in_recovery {
        return;
    }
    let sql = match &kind {
        JobKind::Metric(m) => {
            match m.def.node_status.as_deref() {
                Some("primary") if info.in_recovery => return,
                Some("standby") if !info.in_recovery => return,
                _ => {}
            }
            match m.def.sql_for(info.major) {
                Some(s) => Some(s.to_string()),
                None => return,
            }
        }
        _ => None,
    };
    let name = kind.name();
    let (timeout_s, heavy) = job_timeout(&kind, &em.cfg);
    let _heavy = if heavy {
        Some(db.pool.heavy.clone().acquire_owned().await.unwrap())
    } else {
        None
    };
    let t0 = std::time::Instant::now();
    let mut lease = match db.pool.acquire(&db.conn_str).await {
        Ok(l) => l,
        Err(e) => {
            let msg = format!("{e:#}");
            let first = db.status(|d| {
                let backing_off = msg.contains("(backoff)");
                if !backing_off {
                    d.connect_failures += 1;
                    d.last_error = Some(format!("connect: {msg}"));
                    d.last_error_ms = Some(now_ms());
                }
                let was_up = d.up;
                d.up = false;
                !backing_off && (was_up || d.connect_failures <= 3 || d.connect_failures % 100 == 0)
            });
            if first {
                eprintln!("collector {}: connect: {msg}", db.label);
            }
            return;
        }
    };
    let res: anyhow::Result<()> = async {
        lease.set_timeout(timeout_s).await?;
        let c = &mut lease.client;
        match (&kind, &mut *state) {
            (JobKind::Metric(m), JobState::Metric(st)) => {
                collect_metric(c, m, sql.as_deref().unwrap(), &base, em, st).await?;
            }
            (JobKind::QueryTexts, JobState::Texts(texts)) => {
                let v =
                    crate::structure::snapshot_query_texts(c, &db.source_id(), texts, &em.shared.fp_to_queryid).await?;
                if !v.is_empty() {
                    let _ = em.tx.send(Msg::Rec(Record::QueryTexts(v))).await;
                }
            }
            (JobKind::Objects, _) => {
                let datname = db.datname.read().unwrap().clone().unwrap_or_default();
                let sid = db.source_id();
                let key = crate::structure::objects_key(sid.source.as_deref(), &datname);
                let mut objs = db.ctx.objects.lock().unwrap().remove(&key).unwrap_or_default();
                let r = crate::structure::snapshot_objects(c, &sid, &datname, &mut objs, em.cfg.objects_max_rows).await;
                db.ctx.objects.lock().unwrap().insert(key, objs);
                let v = r?;
                if !v.is_empty() {
                    let _ = em.tx.send(Msg::Rec(Record::Objects(v))).await;
                }
            }
            (JobKind::StatsResets, _) => collect_stats_resets(c, info.major, &base, em).await?,
            _ => {}
        }
        Ok(())
    }
    .await;
    let closed = lease.client.is_closed();
    drop(lease);
    let now = now_ms();
    let log = db.status(|d| {
        let j = d.jobs.entry(name.clone()).or_default();
        j.runs += 1;
        j.last_duration_ms = Some(t0.elapsed().as_millis() as u64);
        match &res {
            Ok(()) => {
                j.last_success_ms = Some(now);
                j.last_error = None;
                d.up = true;
                d.last_success_ms = Some(now);
                None
            }
            Err(e) => {
                j.errors += 1;
                if is_timeout(e) {
                    j.timeouts += 1;
                }
                let t = err_text(e);
                j.last_error = Some(t.clone());
                d.last_error = Some(format!("{name}: {t}"));
                d.last_error_ms = Some(now);
                if closed {
                    d.up = false;
                }
                let n = j.errors;
                (n <= 3 || n % 100 == 0).then(|| format!("{t} (x{n})"))
            }
        }
    });
    if let Some(l) = log {
        eprintln!("collector {} {name}: {l}", db.label);
    }
}

/// The collector's own series for one database, from its status:
/// pgai_mon_source_up, pgai_mon_source_last_success_epoch_s, pgai_mon_connect_failures_total
/// {dbname}; pgai_mon_errors_total, pgai_mon_timeouts_total, pgai_mon_skipped_slots_total
/// {dbname, metric}. Written also while the database is down (up = 0).
async fn emit_self_metrics(db: &DbCtx) {
    let Some(d) = source_status(&db.src.name).and_then(|s| s.databases.get(&db.label).cloned()) else {
        return;
    };
    let dl = vec![("dbname".to_string(), db.label.clone())];
    let mut defs = Vec::new();
    let mut vals = Vec::new();
    let mut put = |metric: &str, labels: Vec<(String, String)>, v: f64| {
        let id = series_id(metric, &labels);
        defs.push((id, metric.to_string(), labels));
        vals.push((id, v));
    };
    put("pgai_mon_source_up", dl.clone(), if d.up { 1.0 } else { 0.0 });
    if let Some(t) = d.last_success_ms {
        put("pgai_mon_source_last_success_epoch_s", dl.clone(), t as f64 / 1000.0);
    }
    put("pgai_mon_connect_failures_total", dl.clone(), d.connect_failures as f64);
    for (job, j) in &d.jobs {
        let mut l = dl.clone();
        l.push(("metric".into(), job.clone()));
        if j.errors > 0 {
            put("pgai_mon_errors_total", l.clone(), j.errors as f64);
        }
        if j.timeouts > 0 {
            put("pgai_mon_timeouts_total", l.clone(), j.timeouts as f64);
        }
        if j.skipped > 0 {
            put("pgai_mon_skipped_slots_total", l, j.skipped as f64);
        }
    }
    db.ctx.em.emit(defs, now_ms(), vals).await;
}

/// The source's main database: version, role, system id and database name.
async fn probe(pool: &Arc<NodePool>, conn_str: &str, cc: &CollectorCfg) -> anyhow::Result<(NodeInfo, String)> {
    let mut l = pool.acquire(conn_str).await?;
    l.set_timeout(cc.default_statement_timeout_s).await?;
    let info = node_info(&l.client).await?;
    let db: String = l
        .client
        .query_one("select pg_catalog.current_database()::text", &[])
        .await?
        .get(0);
    Ok((info, db))
}

async fn discover(pool: &Arc<NodePool>, src: &Source, cc: &CollectorCfg) -> anyhow::Result<Vec<String>> {
    let mut l = pool.acquire(&src.conn_str).await?;
    l.set_timeout(cc.default_statement_timeout_s).await?;
    let rows = l
        .client
        .query(
            "select datname::text from pg_catalog.pg_database where datallowconn and not datistemplate \
             and datname <> pg_catalog.current_database() order by datname",
            &[],
        )
        .await?;
    Ok(rows
        .iter()
        .map(|r| r.get::<_, String>(0))
        .filter(|d| src.wants_db(d))
        .collect())
}

/// Run one source until `stop`: its main database right away (its jobs start once the node
/// has answered), the node probe (version, role) every role_check_interval_s, and database
/// discovery. Errors never end a source: every failed connect is retried after a backoff.
pub async fn run_source(
    src: Source,
    primary: Vec<Scheduled>,
    standby: Vec<Scheduled>,
    ctx: CollectorCtx,
    mut stop: watch::Receiver<bool>,
) {
    let cc = ctx.em.cfg.clone();
    let pool = NodePool::new(cc.clone());
    let node = Arc::new(NodeState::default());
    let primary = Arc::new(primary);
    let standby = Arc::new(standby);
    with_source(&src.name, |s| {
        *s = SourceStatus {
            state: "starting".into(),
            metrics: primary.len(),
            standby_metrics: standby.len(),
            ..Default::default()
        }
    });
    let mk = |label: String, conn_str: String, is_main: bool, datname: Option<String>| {
        Arc::new(DbCtx {
            src: src.clone(),
            label,
            conn_str,
            is_main,
            datname: std::sync::RwLock::new(datname),
            node: node.clone(),
            pool: pool.clone(),
            ctx: ctx.clone(),
            primary: primary.clone(),
            standby: standby.clone(),
        })
    };
    let main = mk(src.name.clone(), src.conn_str.clone(), true, None);
    let mut workers: BTreeMap<String, watch::Sender<bool>> = BTreeMap::new();
    let (main_stop, rx) = watch::channel(false);
    tokio::spawn(run_db(main.clone(), rx));
    let mut last_discovery: Option<tokio::time::Instant> = None;
    let mut logged_down = false;
    loop {
        let wait = match probe(&pool, &src.conn_str, &cc).await {
            Ok((info, datname)) => {
                let first = node.get().is_none();
                let changed = node
                    .get()
                    .is_some_and(|o| o.in_recovery != info.in_recovery || o.major != info.major);
                if first || changed {
                    eprintln!(
                        "collector {}: PG{} {} sys_id={} database={datname}",
                        src.name,
                        info.major,
                        if info.in_recovery { "standby" } else { "primary" },
                        info.sys_id
                    );
                }
                logged_down = false;
                with_source(&src.name, |s| {
                    s.state = "running".into();
                    s.major = Some(info.major);
                    s.in_recovery = Some(info.in_recovery);
                    s.sys_id = Some(info.sys_id.clone());
                });
                *main.datname.write().unwrap() = Some(datname);
                node.set(info);
                if src.discovers()
                    && last_discovery.map_or(true, |t| {
                        t.elapsed() >= Duration::from_secs(cc.discovery_interval_s.max(1))
                    })
                {
                    match discover(&pool, &src, &cc).await {
                        Ok(dbs) => {
                            last_discovery = Some(tokio::time::Instant::now());
                            workers.retain(|d, tx| {
                                let keep = dbs.contains(d);
                                if !keep {
                                    eprintln!("collector {}: database {d} gone, stopped", src.name);
                                    let _ = tx.send(true);
                                    with_source(&src.name, |s| {
                                        s.databases.remove(&format!("{}_{d}", src.name));
                                    });
                                }
                                keep
                            });
                            for d in dbs {
                                if workers.contains_key(&d) {
                                    continue;
                                }
                                let cs = match conn_str_for_db(&src.conn_str, &d) {
                                    Ok(c) => c,
                                    Err(e) => {
                                        eprintln!("collector {}: {d}: {e}", src.name);
                                        continue;
                                    }
                                };
                                let (tx, rx) = watch::channel(false);
                                tokio::spawn(run_db(mk(format!("{}_{d}", src.name), cs, false, Some(d.clone())), rx));
                                workers.insert(d, tx);
                            }
                        }
                        Err(e) => eprintln!("collector {}: discovery: {}", src.name, err_text(&e)),
                    }
                }
                Duration::from_secs(cc.role_check_interval_s.max(1))
            }
            Err(e) => {
                if !logged_down {
                    eprintln!("collector {}: {}; retrying", src.name, err_text(&e));
                    logged_down = true;
                }
                with_db(&src.name, &src.name, |d| {
                    d.up = false;
                    if !format!("{e:#}").contains("(backoff)") {
                        d.last_error = Some(format!("probe: {}", err_text(&e)));
                        d.last_error_ms = Some(now_ms());
                    }
                });
                // the pool's backoff decides when a connect is tried again
                match pool.backing_off(&src.conn_str) {
                    Some(t) => t.saturating_duration_since(tokio::time::Instant::now()) + Duration::from_millis(10),
                    None => Duration::from_secs(cc.reconnect_min_s.max(1)),
                }
            }
        };
        tokio::select! {
            _ = stop.changed() => {}
            _ = tokio::time::sleep(wait) => {}
        }
        if *stop.borrow() || stop.has_changed().is_err() {
            break;
        }
    }
    let _ = main_stop.send(true);
    for tx in workers.values() {
        let _ = tx.send(true);
    }
}

// ---------------------------------------------------------------------------------------
// The set of sources: started, restarted on change, stopped when removed.

struct Running {
    fingerprint: u64,
    stop: watch::Sender<bool>,
}

/// What a source runs with; a change restarts it.
fn fingerprint(src: &Source, primary: &[Scheduled], standby: &[Scheduled], cc: &CollectorCfg) -> u64 {
    xxhash_rust::xxh3::xxh3_64(format!("{src:?}|{primary:?}|{standby:?}|{cc:?}").as_bytes())
}

/// Start, restart and stop sources to match the config: at start, on SIGHUP and every
/// collector.reload_interval_s (the config file, its sources_file and metrics.yml are read
/// again; a file that fails to load keeps the running set as it is).
pub async fn run_manager(initial: Config, ctx: CollectorCtx) {
    let mut running: BTreeMap<String, Running> = BTreeMap::new();
    let mut hup = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup()).ok();
    let mut cfg = initial;
    let mut first = true;
    loop {
        if !first {
            if let Some(p) = &cfg.config_path {
                match Config::load(p) {
                    Ok(c) => cfg = c,
                    Err(e) => eprintln!("collector: reload {}: {e}; keeping the running sources", p.display()),
                }
            }
        }
        first = false;
        apply_sources(&cfg, &ctx, &mut running);
        let every = cfg.collector.reload_interval_s;
        tokio::select! {
            _ = async {
                match hup.as_mut() {
                    Some(h) => { h.recv().await; }
                    None => std::future::pending::<()>().await,
                }
            } => eprintln!("collector: SIGHUP: reloading sources"),
            _ = async {
                if every == 0 { std::future::pending::<()>().await } else { tokio::time::sleep(Duration::from_secs(every)).await }
            } => {}
        }
    }
}

fn apply_sources(cfg: &Config, ctx: &CollectorCtx, running: &mut BTreeMap<String, Running>) {
    let mf = match crate::metricdefs::load(&cfg.metrics_yml) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "collector: metrics.yml {}: {e}; keeping the running sources",
                cfg.metrics_yml.display()
            );
            return;
        }
    };
    let sources = match crate::config::collect_sources(cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("collector: {e}; keeping the running sources");
            return;
        }
    };
    let mut ctx = ctx.clone();
    ctx.em.cfg = cfg.collector.clone();
    let mut want: BTreeMap<String, (Source, Vec<Scheduled>, Vec<Scheduled>, u64)> = BTreeMap::new();
    for s in sources {
        match mf.for_source(&s, &cfg.preset) {
            Ok((p, sb)) => {
                let fp = fingerprint(&s, &p, &sb, &cfg.collector);
                want.insert(s.name.clone(), (s, p, sb, fp));
            }
            Err(e) => eprintln!("collector: source {}: {e}; skipped", s.name),
        }
    }
    running.retain(|name, r| {
        let keep = want.get(name).is_some_and(|w| w.3 == r.fingerprint);
        if !keep {
            let _ = r.stop.send(true);
            if want.contains_key(name) {
                eprintln!("collector: source {name} changed, restarting");
            } else {
                eprintln!("collector: source {name} removed, stopped");
                status().lock().unwrap().remove(name);
            }
        }
        keep
    });
    for (name, (s, p, sb, fp)) in want {
        if running.contains_key(&name) {
            continue;
        }
        let (tx, rx) = watch::channel(false);
        tokio::spawn(run_source(s, p, sb, ctx.clone(), rx));
        running.insert(
            name,
            Running {
                fingerprint: fp,
                stop: tx,
            },
        );
    }
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

    #[test]
    fn row_fields_follow_the_pgwatch_sink() {
        let mut ts = 7;
        let cols = vec![
            ("epoch_ns", PgVal::Int(5_000_000_000)),
            ("tag_datname", PgVal::Text("app".into())),
            ("tag_queryid", PgVal::Int(-42)),
            ("tag_empty", PgVal::Text(String::new())),
            ("tag_null", PgVal::Null),
            ("calls", PgVal::Int(3)),
            ("ratio", PgVal::Float(0.5)),
            ("on", PgVal::Bool(true)),
            ("bytes", PgVal::Numeric(1e20)),
            ("unit", PgVal::Text("kB".into())),
            ("query", PgVal::Text("select 1".into())),
            ("blank", PgVal::Text(String::new())),
            ("relid", PgVal::Other("16384".into())),
            ("nothing", PgVal::Null),
        ];
        let mut classes = BTreeMap::new();
        classes.insert(
            "m".to_string(),
            BTreeMap::from([("query".to_string(), "drop".to_string())]),
        );
        let (labels, fields) = row_fields("m", &cols, &classes, true, &mut ts);
        assert_eq!(ts, 5000, "epoch_ns is the sample time, in ms");
        assert_eq!(
            labels,
            BTreeMap::from([
                ("datname".to_string(), "app".to_string()),
                ("queryid".to_string(), "-42".to_string()),
                ("unit".to_string(), "kB".to_string()),
            ])
        );
        assert_eq!(
            fields,
            vec![
                ("calls".to_string(), 3.0),
                ("ratio".to_string(), 0.5),
                ("on".to_string(), 1.0)
            ]
        );
        let (_, fields) = row_fields("m", &cols, &classes, false, &mut ts);
        assert!(
            fields.contains(&("bytes".to_string(), 1e20)),
            "NUMERIC kept when drop_numeric is off"
        );
    }

    #[test]
    fn base_labels_override_row_tags_except_dbname() {
        let base = vec![
            ("dbname".to_string(), "src".to_string()),
            ("env".to_string(), "prod".to_string()),
            ("sys_id".to_string(), "1".to_string()),
        ];
        let row = BTreeMap::from([
            ("dbname".to_string(), "from_row".to_string()),
            ("env".to_string(), "row".to_string()),
            ("relname".to_string(), "t".to_string()),
        ]);
        assert_eq!(
            series_labels(&base, row),
            vec![
                ("dbname".to_string(), "from_row".to_string()),
                ("env".to_string(), "prod".to_string()),
                ("relname".to_string(), "t".to_string()),
                ("sys_id".to_string(), "1".to_string()),
            ]
        );
    }

    #[test]
    fn backoff_grows_with_jitter_and_a_cap() {
        for n in 1..40 {
            let d = backoff_delay(2, 120, n).as_secs_f64();
            let full = (2f64 * 2f64.powi(n as i32 - 1)).min(120.0);
            assert!(d >= full * 0.5 && d < full, "attempt {n}: {d}");
        }
        let ds: HashSet<u64> = (0..20).map(|_| backoff_delay(2, 120, 5).as_micros() as u64).collect();
        assert!(ds.len() > 1, "jitter");
    }

    #[test]
    fn job_timeouts_and_heavy_jobs() {
        let cfg = CollectorCfg::default();
        let mut m = sched("db_size", 300);
        assert_eq!(job_timeout(&JobKind::Metric(m.clone()), &cfg), (15, false));
        m.def.statement_timeout_seconds = Some(300);
        assert_eq!(job_timeout(&JobKind::Metric(m), &cfg), (300, true));
        assert_eq!(job_timeout(&JobKind::Objects, &cfg), (60, true));
        assert_eq!(job_timeout(&JobKind::QueryTexts, &cfg), (15, false));
    }

    #[test]
    fn go_formats_for_tag_values() {
        assert_eq!(go_float(0.5), "0.5");
        assert_eq!(go_float(3.0), "3");
        assert_eq!(go_float(123456.0), "123456");
        assert_eq!(go_float(1234567.0), "1.234567e+06");
        assert_eq!(go_float(1e20), "1e+20");
        assert_eq!(go_float(0.0001), "0.0001");
        assert_eq!(go_float(0.00001), "1e-05");
        assert_eq!(go_float(-2.5e-7), "-2.5e-07");
        assert_eq!(go_float(f64::NAN), "NaN");
        assert_eq!(go_float(f64::NEG_INFINITY), "-Inf");
        assert_eq!(go_time_utc(0), "1970-01-01 00:00:00 +0000 UTC");
        assert_eq!(go_time_utc(1_704_164_645_123_400), "2024-01-02 03:04:05.1234 +0000 UTC");
        assert_eq!(go_time_utc(-1), "1969-12-31 23:59:59.999999 +0000 UTC");
        assert_eq!(go_time_utc(951_782_400_000_000), "2000-02-29 00:00:00 +0000 UTC");
    }

    #[test]
    fn structure_rows_carry_the_series_identity() {
        let src: Source =
            serde_yaml::from_str("{name: prod1, conn_str: x, custom_tags: {cluster: main, node_name: n1, env: p}}")
                .unwrap();
        assert_eq!(
            source_id(&src),
            SourceId {
                source: Some("prod1".into()),
                cluster: Some("main".into()),
                node_name: Some("n1".into()),
            }
        );
        let bare: Source = serde_yaml::from_str("{name: x, conn_str: y}").unwrap();
        assert_eq!(source_id(&bare).cluster, None);
    }
}
