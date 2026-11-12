//! Records that flow from producers (collector, log tailer, structure, loader) through the
//! WAL into Parquet.
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

/// VictoriaMetrics' StaleNaN bit pattern: a staleness marker, distinct from a real NaN.
pub const STALE_NAN_BITS: u64 = 0x7ff0_0000_0000_0002;

pub fn stale_nan() -> f64 {
    f64::from_bits(STALE_NAN_BITS)
}

pub fn is_stale(v: f64) -> bool {
    v.to_bits() == STALE_NAN_BITS
}

/// One sample. `ts` is Unix milliseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub series_id: u64,
    pub ts: i64,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SeriesDef {
    pub series_id: u64,
    pub metric: String,
    /// Sorted by key.
    pub labels: Vec<(String, String)>,
    pub first_seen: i64,
}

/// series_id = 16 bits of hash(metric) || 48 bits of hash(metric, labels).
/// The metric prefix makes "all series of metric X" a contiguous id range, so files
/// sorted by series_id prune row groups by metric; the 48-bit part identifies the series.
pub fn series_id(metric: &str, labels: &[(String, String)]) -> u64 {
    let mut buf = Vec::with_capacity(256);
    buf.extend_from_slice(metric.as_bytes());
    for (k, v) in labels {
        buf.push(0);
        buf.extend_from_slice(k.as_bytes());
        buf.push(b'=');
        buf.extend_from_slice(v.as_bytes());
    }
    let full = xxh3_64(&buf);
    let m = xxh3_64(metric.as_bytes());
    ((m >> 48) << 48) | (full & 0x0000_FFFF_FFFF_FFFF)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogRow {
    pub ts: i64, // ms
    pub level: Option<String>,
    pub sqlstate: Option<String>,
    pub pid: Option<i64>,
    pub dbname: Option<String>,
    pub username: Option<String>,
    pub application: Option<String>,
    pub message: String,
    pub query_fingerprint: Option<String>,
    pub plan_hash: Option<String>,
    pub queryid: Option<i64>,
    pub raw: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanEvent {
    pub ts: i64,
    pub plan_hash: String,
    pub queryid: Option<i64>,
    pub dbname: Option<String>,
    pub total_ms: f64,
    pub top_node: String,
    pub shared_hit: i64,
    pub shared_read: i64,
    pub shared_dirtied: i64,
    pub shared_written: i64,
    pub temp_read: i64,
    pub temp_written: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanDef {
    pub plan_hash: String,
    pub first_seen: i64,
    pub plan: String, // JSON, literals redacted
}

/// Which monitored node a structure row came from, as the series label it: the source name
/// (the `dbname` label of its main database) and its `cluster` / `node_name` custom tags.
/// None in rows written before these columns existed (Parquet files without them read as NULL).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SourceId {
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub cluster: Option<String>,
    #[serde(default)]
    pub node_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryText {
    pub queryid: i64,
    pub dbname: String,
    pub query_hash: u64,
    pub query: Option<String>, // only on the first row for a (queryid, dbname, hash)
    pub first_seen: i64,
    pub last_seen: i64,
    #[serde(flatten, default)]
    pub src: SourceId,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObjectEvent {
    pub ts: i64,
    pub dbname: String,
    pub kind: String,
    pub schema: String,
    pub name: String,
    /// None = dropped.
    pub definition: Option<String>,
    #[serde(flatten, default)]
    pub src: SourceId,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Samples(Vec<Sample>),
    Series(Vec<SeriesDef>),
    /// Log rows (and the plans parsed from them) plus the tail position they bring the file
    /// to: one WAL record, so tailing is exactly-once.
    Logs {
        rows: Vec<LogRow>,
        plan_events: Vec<PlanEvent>,
        plan_defs: Vec<PlanDef>,
        file: String,
        offset: u64,
    },
    PlanEvents(Vec<PlanEvent>),
    PlanDefs(Vec<PlanDef>),
    QueryTexts(Vec<QueryText>),
    Objects(Vec<ObjectEvent>),
}

#[derive(Serialize, Deserialize)]
struct LogsPayload {
    rows: Vec<LogRow>,
    #[serde(default)]
    plan_events: Vec<PlanEvent>,
    #[serde(default)]
    plan_defs: Vec<PlanDef>,
    file: String,
    offset: u64,
}

impl Record {
    pub fn tag(&self) -> u8 {
        match self {
            Record::Samples(_) => 1,
            Record::Series(_) => 2,
            Record::Logs { .. } => 3,
            Record::PlanEvents(_) => 4,
            Record::PlanDefs(_) => 5,
            Record::QueryTexts(_) => 6,
            Record::Objects(_) => 7,
        }
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Record::Samples(v) => {
                out.reserve(4 + v.len() * 24);
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for s in v {
                    out.extend_from_slice(&s.series_id.to_le_bytes());
                    out.extend_from_slice(&s.ts.to_le_bytes());
                    out.extend_from_slice(&s.value.to_le_bytes());
                }
            }
            Record::Series(v) => serde_json::to_writer(out, v).unwrap(),
            Record::Logs {
                rows,
                plan_events,
                plan_defs,
                file,
                offset,
            } => serde_json::to_writer(
                out,
                &LogsPayload {
                    rows: rows.clone(),
                    plan_events: plan_events.clone(),
                    plan_defs: plan_defs.clone(),
                    file: file.clone(),
                    offset: *offset,
                },
            )
            .unwrap(),
            Record::PlanEvents(v) => serde_json::to_writer(out, v).unwrap(),
            Record::PlanDefs(v) => serde_json::to_writer(out, v).unwrap(),
            Record::QueryTexts(v) => serde_json::to_writer(out, v).unwrap(),
            Record::Objects(v) => serde_json::to_writer(out, v).unwrap(),
        }
    }

    pub fn decode(tag: u8, b: &[u8]) -> anyhow::Result<Record> {
        Ok(match tag {
            1 => {
                anyhow::ensure!(b.len() >= 4, "short samples record");
                let n = u32::from_le_bytes(b[0..4].try_into()?) as usize;
                anyhow::ensure!(b.len() == 4 + n * 24, "bad samples record length");
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    let o = 4 + i * 24;
                    v.push(Sample {
                        series_id: u64::from_le_bytes(b[o..o + 8].try_into()?),
                        ts: i64::from_le_bytes(b[o + 8..o + 16].try_into()?),
                        value: f64::from_le_bytes(b[o + 16..o + 24].try_into()?),
                    });
                }
                Record::Samples(v)
            }
            2 => Record::Series(serde_json::from_slice(b)?),
            3 => {
                let p: LogsPayload = serde_json::from_slice(b)?;
                Record::Logs {
                    rows: p.rows,
                    plan_events: p.plan_events,
                    plan_defs: p.plan_defs,
                    file: p.file,
                    offset: p.offset,
                }
            }
            4 => Record::PlanEvents(serde_json::from_slice(b)?),
            5 => Record::PlanDefs(serde_json::from_slice(b)?),
            6 => Record::QueryTexts(serde_json::from_slice(b)?),
            7 => Record::Objects(serde_json::from_slice(b)?),
            t => anyhow::bail!("unknown record tag {t}"),
        })
    }
}

pub const DAY_MS: i64 = 86_400_000;

pub fn day_of(ts_ms: i64) -> i64 {
    ts_ms.div_euclid(DAY_MS)
}

/// Civil date (YYYY-MM-DD) of a day number since 1970-01-01 (Howard Hinnant's algorithm).
pub fn day_str(day: i64) -> String {
    let z = day + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{:02}-{:02}", if m <= 2 { y + 1 } else { y }, m, d)
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}
