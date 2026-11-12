//! jsonlog tailer and auto_explain plan parser.
use crate::model::*;
use regex::Regex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use xxhash_rust::xxh3::xxh3_64;

fn re_str() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"'(?:[^']|'')*'").unwrap())
}
fn re_num() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?:^|[^\w$.])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\b").unwrap())
}
fn re_param() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\$\d+").unwrap())
}
fn re_ws() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\s+").unwrap())
}
fn re_inlist() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\(\s*\?(?:\s*,\s*\?)*\s*\)").unwrap())
}

/// Replace literals (strings, numbers) and bind placeholders with `?`.
pub fn redact_literals(s: &str) -> String {
    let s = re_str().replace_all(s, "?");
    let s = re_param().replace_all(&s, "?");
    re_num()
        .replace_all(&s, |c: &regex::Captures| {
            let whole = c.get(0).unwrap().as_str();
            let num = c.get(1).unwrap().as_str();
            format!("{}?", &whole[..whole.len() - num.len()])
        })
        .into_owned()
}

/// Fingerprint of a statement: literals and $n -> ?, IN lists collapsed, case and
/// whitespace folded. A raw statement and its pg_stat_statements text map to the same value.
pub fn fingerprint_text(sql: &str) -> String {
    let r = redact_literals(sql).to_lowercase();
    let r = re_inlist().replace_all(&r, "(?)");
    let r = re_ws().replace_all(r.trim(), " ");
    let r = r.trim_end_matches(';').trim();
    r.to_string()
}

pub fn fingerprint(sql: &str) -> String {
    format!("{:016x}", xxh3_64(fingerprint_text(sql).as_bytes()))
}

/// Keys that define a plan's shape (hashed); everything else (costs, rows, buffers, time)
/// varies run to run.
const SHAPE_KEYS: &[&str] = &[
    "Node Type",
    "Parent Relationship",
    "Join Type",
    "Strategy",
    "Partial Mode",
    "Operation",
    "Relation Name",
    "Schema",
    "Index Name",
    "Scan Direction",
    "CTE Name",
    "Function Name",
    "Subplan Name",
    "Sort Key",
    "Group Key",
    "Hash Cond",
    "Index Cond",
    "Recheck Cond",
    "Filter",
    "Join Filter",
    "Merge Cond",
    "Command",
    "Parallel Aware",
];
/// String fields that are names, not expressions: kept verbatim in the stored plan.
const NAME_KEYS: &[&str] = &[
    "Node Type",
    "Parent Relationship",
    "Join Type",
    "Strategy",
    "Partial Mode",
    "Operation",
    "Relation Name",
    "Schema",
    "Index Name",
    "Scan Direction",
    "CTE Name",
    "Function Name",
    "Subplan Name",
    "Alias",
    "Command",
    "Sort Method",
    "Sort Space Type",
];

fn shape(v: &Value, out: &mut String) {
    if let Value::Object(m) = v {
        out.push('{');
        for k in SHAPE_KEYS {
            if let Some(x) = m.get(*k) {
                out.push_str(k);
                out.push('=');
                match x {
                    Value::String(s) => out.push_str(&redact_literals(s)),
                    Value::Array(a) => {
                        for e in a {
                            if let Value::String(s) = e {
                                out.push_str(&redact_literals(s));
                                out.push(',');
                            }
                        }
                    }
                    other => out.push_str(&other.to_string()),
                }
                out.push(';');
            }
        }
        if let Some(Value::Array(children)) = m.get("Plans") {
            out.push('[');
            for c in children {
                shape(c, out);
            }
            out.push(']');
        }
        out.push('}');
    }
}

/// Copy of a plan with literal values redacted from expression strings.
fn redact_plan(v: &Value, key: &str) -> Value {
    match v {
        Value::Object(m) => {
            let mut o = serde_json::Map::new();
            for (k, x) in m {
                if k == "Query Text" || k == "Query Parameters" {
                    continue;
                }
                o.insert(k.clone(), redact_plan(x, k));
            }
            Value::Object(o)
        }
        Value::Array(a) => Value::Array(a.iter().map(|x| redact_plan(x, key)).collect()),
        Value::String(s) if !NAME_KEYS.contains(&key) => Value::String(redact_literals(s)),
        other => other.clone(),
    }
}

pub struct ParsedPlan {
    pub plan_hash: String,
    pub queryid: Option<i64>,
    pub total_ms: f64,
    pub top_node: String,
    pub buffers: [i64; 6],
    pub plan_redacted: String,
    pub query_fp: Option<String>,
}

/// Parse an auto_explain (log_format=json) message: "duration: X ms  plan:\n{...}".
pub fn parse_auto_explain(message: &str) -> Option<ParsedPlan> {
    let rest = message.strip_prefix("duration: ")?;
    let (ms, after) = rest.split_once(" ms")?;
    let total_ms: f64 = ms.trim().parse().ok()?;
    let after = after.trim_start();
    let json = after.strip_prefix("plan:")?.trim_start();
    if !json.starts_with('{') {
        return None;
    }
    let v: Value = serde_json::from_str(json).ok()?;
    let plan = v.get("Plan")?;
    let mut sh = String::new();
    shape(plan, &mut sh);
    let plan_hash = format!("{:016x}", xxh3_64(sh.as_bytes()));
    let queryid = v.get("Query Identifier").and_then(|x| x.as_i64());
    let g = |k: &str| plan.get(k).and_then(|x| x.as_i64()).unwrap_or(0);
    let buffers = [
        g("Shared Hit Blocks"),
        g("Shared Read Blocks"),
        g("Shared Dirtied Blocks"),
        g("Shared Written Blocks"),
        g("Temp Read Blocks"),
        g("Temp Written Blocks"),
    ];
    let top_node = plan.get("Node Type").and_then(|x| x.as_str()).unwrap_or("").to_string();
    let query_fp = v.get("Query Text").and_then(|x| x.as_str()).map(fingerprint_text);
    Some(ParsedPlan {
        plan_hash,
        queryid,
        total_ms,
        top_node,
        buffers,
        plan_redacted: redact_plan(&v, "").to_string(),
        query_fp,
    })
}

/// "2026-11-09 00:22:46.173 UTC" -> Unix ms. Non-UTC zone names are taken as UTC.
pub fn parse_pg_ts(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 19 {
        return None;
    }
    let num = |a: usize, n: usize| -> Option<i64> { std::str::from_utf8(&b[a..a + n]).ok()?.parse().ok() };
    let (y, mo, d, h, mi, se) = (
        num(0, 4)?,
        num(5, 2)?,
        num(8, 2)?,
        num(11, 2)?,
        num(14, 2)?,
        num(17, 2)?,
    );
    let mut ms = 0;
    if b.len() > 19 && b[19] == b'.' {
        let mut i = 20;
        let mut scale = 100;
        while i < b.len() && b[i].is_ascii_digit() {
            if scale >= 1 {
                ms += (b[i] - b'0') as i64 * scale;
            }
            scale /= 10;
            i += 1;
        }
    }
    // days from civil
    let yy = if mo <= 2 { y - 1 } else { y };
    let era = yy.div_euclid(400);
    let yoe = yy - era * 400;
    let mp = (mo + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(((days * 24 + h) * 60 + mi) * 60_000 + se * 1000 + ms)
}

pub struct ParsedLine {
    pub log: LogRow,
    pub plan: Option<(PlanEvent, PlanDef)>,
}

/// Parse one jsonlog line. `fp_map` links PG14/15 plans (no Query Identifier) to queryid via
/// the normalized-text fingerprint of pg_stat_statements texts.
pub fn parse_line(line: &str, fp_map: Option<&HashMap<String, i64>>) -> Option<ParsedLine> {
    let mut v: Value = serde_json::from_str(line).ok()?;
    let o = v.as_object_mut()?;
    let ts = parse_pg_ts(o.get("timestamp")?.as_str()?)?;
    let s = |o: &serde_json::Map<String, Value>, k: &str| o.get(k).and_then(|x| x.as_str()).map(|x| x.to_string());
    let message = s(o, "message").unwrap_or_default();
    let mut log = LogRow {
        ts,
        level: s(o, "error_severity"),
        sqlstate: s(o, "state_code"),
        pid: o.get("pid").and_then(|x| x.as_i64()),
        dbname: s(o, "dbname"),
        username: s(o, "user"),
        application: s(o, "application_name"),
        message: message.clone(),
        query_fingerprint: None,
        plan_hash: None,
        queryid: o.get("query_id").and_then(|x| x.as_i64()).filter(|q| *q != 0),
        raw: String::new(),
    };
    // Bind parameter values never stored.
    if let Some(d) = s(o, "detail") {
        if d.starts_with("Parameters:") || d.starts_with("parameters:") || d.contains("Query Parameters") {
            o.remove("detail");
        }
    }
    let mut plan = None;
    if let Some(p) = parse_auto_explain(&message) {
        let qid = p.queryid.or_else(|| {
            p.query_fp
                .as_ref()
                .and_then(|fp| fp_map.and_then(|m| m.get(fp).copied()))
        });
        log.message = format!("duration: {:.3} ms  plan: {}", p.total_ms, p.plan_hash);
        log.plan_hash = Some(p.plan_hash.clone());
        log.queryid = qid.or(log.queryid);
        log.query_fingerprint = p.query_fp.as_ref().map(|t| format!("{:016x}", xxh3_64(t.as_bytes())));
        o.insert("message".into(), Value::String(log.message.clone()));
        let ev = PlanEvent {
            ts,
            plan_hash: p.plan_hash.clone(),
            queryid: qid,
            dbname: log.dbname.clone(),
            total_ms: p.total_ms,
            top_node: p.top_node,
            shared_hit: p.buffers[0],
            shared_read: p.buffers[1],
            shared_dirtied: p.buffers[2],
            shared_written: p.buffers[3],
            temp_read: p.buffers[4],
            temp_written: p.buffers[5],
        };
        let def = PlanDef {
            plan_hash: p.plan_hash,
            first_seen: ts,
            plan: p.plan_redacted,
        };
        plan = Some((ev, def));
    } else if let Some(st) = s(o, "statement") {
        log.query_fingerprint = Some(fingerprint(&st));
    } else if let Some(rest) = message.strip_prefix("duration: ") {
        // "duration: 1.2 ms  statement: ..." / "execute <name>: ..."
        if let Some((_, q)) = rest.split_once(": ") {
            log.query_fingerprint = Some(fingerprint(q));
        }
    }
    log.raw = Value::Object(o.clone()).to_string();
    Some(ParsedLine { log, plan })
}

/// Tails every `*suffix` file in `dir`, sending batches that carry the new file offset.
pub struct Tailer {
    pub dir: PathBuf,
    pub suffix: String,
    pub offsets: HashMap<String, u64>,
    pub start_at_end: bool,
    pub known_plans: HashSet<String>,
    pub fp_map: Arc<Mutex<HashMap<String, i64>>>,
    pub max_read: usize,
}

impl Tailer {
    /// One pass over the directory. Returns records to append (each Logs record carries the
    /// file offset it reaches, so offsets advance exactly with the data in the WAL).
    pub fn poll(&mut self) -> anyhow::Result<Vec<Record>> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&self.dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().ends_with(&self.suffix))
            .collect();
        files.sort();
        let mut out = Vec::new();
        // One poll reads at most max_read bytes in total (across files): bounded memory, and
        // the caller's per-record WAL ack gives backpressure while catching up.
        let mut budget = self.max_read;
        for p in files {
            if budget < 4096 {
                break;
            }
            let key = p.to_string_lossy().to_string();
            let len = std::fs::metadata(&p)?.len();
            let off = match self.offsets.get(&key) {
                Some(o) if *o <= len => *o,
                Some(_) => 0, // truncated/replaced
                None if self.start_at_end => {
                    self.offsets.insert(key.clone(), len);
                    continue;
                }
                None => 0,
            };
            if off >= len {
                continue;
            }
            let mut f = std::fs::File::open(&p)?;
            f.seek(SeekFrom::Start(off))?;
            let want = ((len - off) as usize).min(budget);
            let mut buf = vec![0u8; want];
            f.read_exact(&mut buf)?;
            let Some(last_nl) = buf.iter().rposition(|b| *b == b'\n') else {
                if want == self.max_read && budget == self.max_read {
                    // a single line larger than max_read: skip it
                    self.offsets.insert(key, off + want as u64);
                }
                continue;
            };
            let chunk = &buf[..=last_nl];
            budget -= chunk.len();
            let fp = self.fp_map.lock().unwrap().clone();
            let mut rows = Vec::new();
            let mut evs = Vec::new();
            let mut defs = Vec::new();
            for line in chunk.split(|b| *b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                let Ok(line) = std::str::from_utf8(line) else { continue };
                if let Some(pl) = parse_line(line, Some(&fp)) {
                    if let Some((ev, def)) = pl.plan {
                        if self.known_plans.insert(def.plan_hash.clone()) {
                            defs.push(def);
                        }
                        evs.push(ev);
                    }
                    rows.push(pl.log);
                }
            }
            let new_off = off + chunk.len() as u64;
            out.push(Record::Logs {
                rows,
                plan_events: evs,
                plan_defs: defs,
                file: key.clone(),
                offset: new_off,
            });
            self.offsets.insert(key, new_off);
        }
        Ok(out)
    }
}
