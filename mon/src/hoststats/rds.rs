//! One poll of an RDS or Aurora PostgreSQL instance: the poll() of old/rds-host-stats/lib/poll.ts,
//! with the same calls, series names, units, labels, timing rules, dedupe and failure
//! behaviour. The comments of poll.ts that explain the rules are kept here.
use super::aws::{err, parse_iso, AwsApi, AwsError};
use serde_json::{json, Value};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct Target {
    pub instance_id: String,
    pub cluster: String,
    pub node_name: String,
}

/// One sample to write: a host_* series of the target, its value and time (Unix ms).
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub name: &'static str,
    /// `provider="Amazon RDS"` is added (host_provider_info only).
    pub provider: bool,
    pub value: f64,
    pub time: i64,
}

/// The samples not yet written, and the sources that failed this poll (Performance Insights,
/// Enhanced Monitoring): their samples are retried on the next poll, the rest are kept. A
/// malformed Enhanced Monitoring event is reported once and never retried: it stays malformed.
#[derive(Debug, Default)]
pub struct Poll {
    pub points: Vec<Point>,
    pub errors: Vec<AwsError>,
}

/// Written samples (and reported malformed events) by key, with their time, for the dedupe.
pub type State = HashMap<String, i64>;

type Metric = (&'static str, &'static str, f64);
const COMMON: [Metric; 10] = [
    ("host_cpu_utilization_percent", "CPUUtilization", 1.0),
    ("host_memory_available_bytes", "FreeableMemory", 1.0),
    ("host_disk_read_iops", "ReadIOPS", 1.0),
    ("host_disk_write_iops", "WriteIOPS", 1.0),
    ("host_network_receive_bytes_per_second", "NetworkReceiveThroughput", 1.0),
    (
        "host_network_transmit_bytes_per_second",
        "NetworkTransmitThroughput",
        1.0,
    ),
    ("host_disk_read_bytes_per_second", "ReadThroughput", 1.0),
    ("host_disk_write_bytes_per_second", "WriteThroughput", 1.0),
    ("host_disk_read_latency_seconds", "ReadLatency", 1.0),
    ("host_disk_write_latency_seconds", "WriteLatency", 1.0),
];
const RDS: [Metric; 1] = [("host_disk_free_bytes", "FreeStorageSpace", 1.0)];
const AURORA: [Metric; 1] = [("host_local_storage_free_bytes", "FreeLocalStorage", 1.0)];
const VOLUME: [Metric; 3] = [
    ("host_volume_used_bytes", "VolumeBytesUsed", 1.0),
    ("host_volume_read_iops", "VolumeReadIOPs", 1.0 / 300.0),
    ("host_volume_write_iops", "VolumeWriteIOPs", 1.0 / 300.0),
];

/// Enhanced Monitoring fields: (series, path, multiplier). With a multiplier the value is
/// converted as JavaScript's `*` converts (null is 0, a numeric string its number); without one
/// it must be a JSON number.
type OsField = (&'static str, &'static [&'static str], Option<f64>);
const OS: [OsField; 10] = [
    ("host_os_cpu_percent", &["cpuUtilization", "total"], None),
    ("host_os_cpu_iowait_percent", &["cpuUtilization", "wait"], None),
    ("host_os_cpu_user_percent", &["cpuUtilization", "user"], None),
    ("host_os_cpu_system_percent", &["cpuUtilization", "system"], None),
    ("host_os_cpu_steal_percent", &["cpuUtilization", "steal"], None),
    ("host_os_load1", &["loadAverageMinute", "one"], None),
    ("host_os_memory_total_bytes", &["memory", "total"], Some(1024.0)),
    ("host_os_memory_free_bytes", &["memory", "free"], Some(1024.0)),
    ("host_os_memory_cached_bytes", &["memory", "cached"], Some(1024.0)),
    ("host_os_memory_buffers_bytes", &["memory", "buffers"], Some(1024.0)),
];
const SWAP: &str = "host_os_swap_used_bytes";
const PROVIDER: &str = "host_provider_info";

/// JavaScript's Number#toString: shortest round-trip digits, exponent form from 1e21 and below
/// 1e-6 (the text rds-host-stats wrote; the dedupe keys and the golden files use it).
pub fn js_num(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let a = v.abs();
    if (1e-6..1e21).contains(&a) {
        return format!("{v}");
    }
    let e = format!("{v:e}");
    match e.split_once('e') {
        Some((m, x)) if !x.starts_with('-') => format!("{m}e+{x}"),
        _ => e,
    }
}

/// JavaScript ToNumber of a JSON value (for the `* 1024` fields).
fn to_number(v: Option<&Value>) -> f64 {
    match v {
        None => f64::NAN,
        Some(Value::Null) => 0.0,
        Some(Value::Bool(b)) => *b as u8 as f64,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty() {
                0.0
            } else if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                u64::from_str_radix(h, 16).map_or(f64::NAN, |x| x as f64)
            } else if matches!(t, "Infinity" | "+Infinity") {
                f64::INFINITY
            } else if t == "-Infinity" {
                f64::NEG_INFINITY
            } else if t
                .bytes()
                .all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
            {
                t.parse().unwrap_or(f64::NAN)
            } else {
                f64::NAN
            }
        }
        // arrays and objects: [] is 0 and [x] is x in JavaScript; AWS sends neither
        Some(_) => f64::NAN,
    }
}

/// `obj.a.b`: None where JavaScript would throw (an intermediate null or undefined); a missing
/// leaf or a non-object intermediate is undefined.
fn get<'a>(v: &'a Value, path: &[&str]) -> Result<Option<&'a Value>, ()> {
    let mut cur = Some(v);
    for k in path {
        match cur {
            None | Some(Value::Null) => return Err(()),
            Some(Value::Object(m)) => cur = m.get(*k),
            Some(_) => cur = None,
        }
    }
    Ok(cur)
}

/// The `{cluster=...,node_name=...}` text of the dedupe keys.
fn label_text(t: &Target, provider: bool) -> String {
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
    let p = if provider { ",provider=\"Amazon RDS\"" } else { "" };
    format!(
        "{{cluster=\"{}\",node_name=\"{}\"{p}}}",
        esc(&t.cluster),
        esc(&t.node_name)
    )
}

/// The samples as rds-host-stats's import body (Prometheus text, value and time as JavaScript
/// prints them).
pub fn render(t: &Target, points: &[Point]) -> String {
    let (l, lp) = (label_text(t, false), label_text(t, true));
    points
        .iter()
        .map(|p| {
            format!(
                "{}{} {} {}\n",
                p.name,
                if p.provider { &lp } else { &l },
                js_num(p.value),
                p.time
            )
        })
        .collect()
}

/// Labels of the series a point writes.
pub fn labels(t: &Target, p: &Point) -> Vec<(String, String)> {
    let mut l = vec![
        ("cluster".to_string(), t.cluster.clone()),
        ("node_name".to_string(), t.node_name.clone()),
    ];
    if p.provider {
        l.push(("provider".into(), "Amazon RDS".into()));
    }
    l
}

fn num(v: &Value) -> Option<f64> {
    v.as_f64()
}

/// One poll at `now` (Unix ms). Fails when DescribeDBInstances or GetMetricData fails or the
/// instance is gone; `state` then may hold keys of samples not returned, so callers pass a copy
/// and keep it only once the samples are written.
pub async fn poll(api: &impl AwsApi, target: &Target, now: i64, state: &mut State) -> Result<Poll, AwsError> {
    let description = api
        .call(
            "rds",
            "DescribeDBInstancesCommand",
            json!({ "DBInstanceIdentifier": target.instance_id }),
        )
        .await?;
    let db = description["DBInstances"]
        .get(0)
        .cloned()
        .ok_or_else(|| err("Error", "RDS instance not found"))?;
    let aurora = db["Engine"].as_str() == Some("aurora-postgresql");
    // CloudWatch and PI buckets are queried over the 15 minutes up to the last minute boundary,
    // so no bucket is cut by the wall clock. A bucket is written only once a further period has
    // passed after it closed: a partial Average written early would be frozen by the
    // per-timestamp dedupe, and the complete one dropped. CloudWatch stamps a bucket at its
    // start, PI at its end, so the gate is 2 periods for CloudWatch and 1 for PI. The OS log
    // window follows the clock; its events are samples.
    let end = now.div_euclid(60_000) * 60_000;
    let start = end - 15 * 60_000;
    let closed = |time: i64, period: i64| time + 2 * period * 1000 <= end;
    let closed_pi = |time: i64| time + 60_000 <= end;
    let mut metrics: Vec<(Metric, i64)> = COMMON
        .iter()
        .chain(if aurora { AURORA.iter() } else { RDS.iter() })
        .map(|m| (*m, 60))
        .collect();
    let query = |m: &Metric, dim: &str, value: &Value, period: i64| {
        let mut d = json!({ "Name": dim });
        // an undefined value is left out of the request, as the SDK leaves it out
        if !value.is_null() {
            d["Value"] = value.clone();
        }
        json!({
            "Id": m.0,
            "MetricStat": { "Metric": { "Namespace": "AWS/RDS", "MetricName": m.1, "Dimensions": [d] }, "Period": period, "Stat": "Average" },
        })
    };
    let mut queries: Vec<Value> = metrics
        .iter()
        .map(|(m, p)| query(m, "DBInstanceIdentifier", &json!(target.instance_id), *p))
        .collect();
    if aurora {
        for m in &VOLUME {
            queries.push(query(m, "DBClusterIdentifier", &db["DBClusterIdentifier"], 300));
            metrics.push((*m, 300));
        }
    }
    let mut out = Poll::default();
    let labels = label_text(target, false);
    state.retain(|_, t| *t >= start - 15 * 60_000);
    // Only finite numbers are written: a value from a log event is untrusted.
    let mut emit = |out: &mut Poll, state: &mut State, name: &'static str, value: Option<f64>, time: i64| {
        let Some(value) = value.filter(|v| v.is_finite()) else {
            return;
        };
        let key = format!("{name}{labels} {time}");
        if let std::collections::hash_map::Entry::Vacant(e) = state.entry(key) {
            out.points.push(Point {
                name,
                provider: false,
                value,
                time,
            });
            e.insert(time);
        }
    };
    // A fixed, non-sensitive identity series lets the dashboard name the provider without
    // displaying account, region, instance or host IDs.
    let pkey = format!("{PROVIDER}{} {end}", label_text(target, true));
    if let std::collections::hash_map::Entry::Vacant(e) = state.entry(pkey) {
        out.points.push(Point {
            name: PROVIDER,
            provider: true,
            value: 1.0,
            time: end,
        });
        e.insert(end);
    }
    let result = api
        .call(
            "cloudwatch",
            "GetMetricDataCommand",
            json!({ "StartTime": super::aws::iso_ms(start), "EndTime": super::aws::iso_ms(end), "MetricDataQueries": queries }),
        )
        .await?;
    let empty = vec![];
    let results = result["MetricDataResults"].as_array().unwrap_or(&empty);
    for ((name, _, scale), period) in &metrics {
        let data = results.iter().find(|r| r["Id"].as_str() == Some(name));
        let mut points: Vec<(i64, Option<f64>)> = data
            .and_then(|d| d["Timestamps"].as_array())
            .map(|ts| {
                ts.iter()
                    .enumerate()
                    .filter_map(|(i, t)| {
                        let time = t.as_str().and_then(parse_iso)?;
                        Some((time, data.and_then(|d| d["Values"].get(i)).and_then(num)))
                    })
                    .collect()
            })
            .unwrap_or_default();
        points.sort_by_key(|p| p.0);
        for (time, value) in points {
            if let Some(v) = value {
                if closed(time, *period) {
                    emit(&mut out, state, name, Some(v * scale), time);
                }
            }
        }
    }
    // Storage limits from the instance description, stamped at `end` on every poll. Only RDS
    // (not Aurora) reports them: Aurora storage grows on demand and has no provisioned IOPS or
    // throughput. gp2 and magnetic storage report no Iops or StorageThroughput, so their limit
    // series stay empty.
    if !aurora {
        let gib = 1024f64.powi(3);
        let pos = |k: &str| num(&db[k]).filter(|v| *v > 0.0);
        if let Some(a) = pos("AllocatedStorage") {
            emit(&mut out, state, "host_disk_size_bytes", Some(a * gib), end);
        }
        if let Some(i) = pos("Iops") {
            emit(&mut out, state, "host_disk_provisioned_iops", Some(i), end);
        }
        if let Some(t) = pos("StorageThroughput") {
            emit(
                &mut out,
                state,
                "host_disk_provisioned_throughput_bytes_per_second",
                Some(t * 1024f64.powi(2)),
                end,
            );
        }
    }
    if db["PerformanceInsightsEnabled"].as_bool() == Some(true) {
        let mut input = json!({
            "ServiceType": "RDS", "StartTime": super::aws::iso_ms(start), "EndTime": super::aws::iso_ms(end),
            "PeriodInSeconds": 60, "MetricQueries": [{ "Metric": "db.load.avg" }],
        });
        if !db["DbiResourceId"].is_null() {
            input["Identifier"] = db["DbiResourceId"].clone();
        }
        match api.call("pi", "GetResourceMetricsCommand", input).await {
            // PI stamps a point at the end of its minute and rounds EndTime up, so a point after
            // `end` covers a minute that is still open, and the point at `end` a minute that
            // closed moments ago and may still be ingesting.
            Ok(r) => {
                for metric in r["MetricList"].as_array().unwrap_or(&empty) {
                    for p in metric["DataPoints"].as_array().unwrap_or(&empty) {
                        if let Some(t) = p["Timestamp"].as_str().and_then(parse_iso) {
                            if closed_pi(t) {
                                emit(&mut out, state, "host_db_load", num(&p["Value"]), t);
                            }
                        }
                    }
                }
            }
            Err(e) => out.errors.push(e),
        }
    }
    if num(&db["MonitoringInterval"]).unwrap_or(0.0) > 0.0 {
        let mut input = json!({ "logGroupName": "RDSOSMetrics", "startTime": now - 15 * 60_000, "endTime": now });
        if !db["DbiResourceId"].is_null() {
            input["logStreamName"] = db["DbiResourceId"].clone();
        }
        match api.call("logs", "GetLogEventsCommand", input).await {
            Ok(r) => os_samples(&r, now, state, &mut out, &mut emit),
            Err(e) => out.errors.push(e),
        }
    }
    Ok(out)
}

type Emit<'a> = dyn FnMut(&mut Poll, &mut State, &'static str, Option<f64>, i64) + 'a;

fn os_samples(r: &Value, now: i64, state: &mut State, out: &mut Poll, emit: &mut Emit) {
    let mut events: Vec<(Value, i64)> = vec![];
    let none = vec![];
    for ev in r["events"].as_array().unwrap_or(&none) {
        let parsed: Result<(Value, i64), String> = match ev["message"].as_str().map(serde_json::from_str::<Value>) {
            Some(Ok(os)) => match os.get("timestamp").and_then(|t| t.as_str()).and_then(parse_iso) {
                Some(at) => Ok((os, at)),
                None => Err("no timestamp".into()),
            },
            Some(Err(e)) => Err(format!("JSON Parse error: {e}")),
            None => Err("JSON Parse error: Unexpected EOF".into()),
        };
        match parsed {
            Ok(e) => events.push(e),
            Err(why) => {
                // The 15-minute log window returns the same event on every poll; report it
                // once. Remembered in state like a written sample, so it is pruned with them
                // and re-reported only if the write failed.
                let ts = match &ev["timestamp"] {
                    Value::Number(n) => js_num(n.as_f64().unwrap_or(f64::NAN)),
                    Value::Null if ev.get("timestamp").is_some() => "null".into(),
                    Value::Null => "undefined".into(),
                    other => other.to_string(),
                };
                let msg = match &ev["message"] {
                    Value::String(s) => s.clone(),
                    Value::Null => "undefined".into(),
                    other => other.to_string(),
                };
                let key = format!("skipped {ts} {msg}");
                if let std::collections::hash_map::Entry::Vacant(e) = state.entry(key) {
                    e.insert(ev["timestamp"].as_f64().map_or(now, |t| t as i64));
                    out.errors.push(err(
                        "Error",
                        format!("Enhanced Monitoring event at {ts} skipped: {why}"),
                    ));
                }
            }
        }
    }
    for (name, path, mult) in &OS {
        for (os, at) in &events {
            // a field missing from this event: the series stays empty for it
            let Ok(v) = get(os, path) else { continue };
            let value = match mult {
                None => v.and_then(Value::as_f64),
                Some(m) => Some(to_number(v) * m),
            };
            emit(out, state, name, value, *at);
        }
    }
    for (os, at) in &events {
        let (Ok(t), Ok(f)) = (get(os, &["swap", "total"]), get(os, &["swap", "free"])) else {
            continue;
        };
        emit(out, state, SWAP, Some((to_number(t) - to_number(f)) * 1024.0), *at);
    }
    for (os, at) in &events {
        // id 0 rows are the "OS processes" and "RDS processes" aggregates.
        let rss: Vec<f64> = os["processList"]
            .as_array()
            .map(|l| {
                l.iter()
                    .filter(|p| p.is_object() && p["id"].as_f64() != Some(0.0))
                    .filter_map(|p| p["rss"].as_f64())
                    .collect()
            })
            .unwrap_or_default();
        if !rss.is_empty() {
            let max = rss.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            emit(out, state, "host_os_process_max_rss_bytes", Some(max * 1024.0), *at);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn javascript_number_text() {
        for (v, s) in [
            (0.0, "0"),
            (-0.0, "0"),
            (1.0, "1"),
            (48.074999999999996, "48.074999999999996"),
            (239919104.0, "239919104"),
            (107374182400.0, "107374182400"),
            (0.002, "0.002"),
            (1e-6, "0.000001"),
            (1e-7, "1e-7"),
            (1.5e-7, "1.5e-7"),
            (1e21, "1e+21"),
            (1.2345e22, "1.2345e+22"),
            (-2.5, "-2.5"),
            (1.0 / 3.0, "0.3333333333333333"),
        ] {
            assert_eq!(js_num(v), s, "{v}");
        }
    }

    #[test]
    fn javascript_to_number() {
        assert_eq!(to_number(Some(&json!(null))), 0.0);
        assert_eq!(to_number(Some(&json!("12"))), 12.0);
        assert_eq!(to_number(Some(&json!(" 1.5 "))), 1.5);
        assert_eq!(to_number(Some(&json!(""))), 0.0);
        assert_eq!(to_number(Some(&json!(true))), 1.0);
        assert!(to_number(Some(&json!("NaN"))).is_nan());
        assert!(to_number(Some(&json!("1 2"))).is_nan());
        assert!(to_number(None).is_nan());
    }
}
