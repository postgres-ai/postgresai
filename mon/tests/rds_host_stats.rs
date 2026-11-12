//! RDS host stats against the recordings of rds-host-stats (tests/fixtures/rds-host-stats, a copy
//! of old/rds-host-stats/test/fixtures): every case of its poll.test.ts, replayed here with the same
//! inputs and the same expected import text, plus the write path and the AWS wire layer against a
//! local fake AWS endpoint (AssumeRole with External ID, STS failure, IMDSv2). No AWS calls.
mod ingest_common;
use axum::body::Bytes;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use pgai_mon::auth::Auth;
use pgai_mon::hoststats::aws::{self, err, iso_ms, parse_iso, AwsApi, AwsError};
use pgai_mon::hoststats::rds::{self, poll, render, State, Target};
use pgai_mon::hoststats::{tick, RdsTarget};
use pgai_mon::ingest::{Ingest, IngestCfg};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

const CASES: [&str; 3] = ["rds-postgres", "rds-postgres-no-em", "aurora-postgresql"];

#[derive(Clone)]
struct Recorded {
    client: String,
    command: String,
    input: Value,
    output: Value,
}

struct Case {
    recorded: Vec<Recorded>,
    now: i64,
    instance_id: String,
    expected: String,
}

fn dir(name: &str) -> String {
    format!("{}/tests/fixtures/rds-host-stats/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn load(name: &'static str) -> Case {
    let rec: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(format!("{}/recorded.json", dir(name))).unwrap()).unwrap();
    let case: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{}/case.json", dir(name))).unwrap()).unwrap();
    Case {
        recorded: rec
            .into_iter()
            .map(|r| Recorded {
                client: r["client"].as_str().unwrap().into(),
                command: r["command"].as_str().unwrap().into(),
                input: r["input"].clone(),
                output: r["output"].clone(),
            })
            .collect(),
        now: parse_iso(case["now"].as_str().unwrap()).unwrap(),
        instance_id: case["instanceId"].as_str().unwrap().into(),
        expected: std::fs::read_to_string(format!("{}/expected.prom", dir(name))).unwrap(),
    }
}

fn target(c: &Case) -> Target {
    Target {
        instance_id: c.instance_id.clone(),
        cluster: "ci".into(),
        node_name: "node-01".into(),
    }
}

/// Replays recorded AWS responses and fails on any request that differs from the recorded one.
/// `fail` makes a client throw instead.
struct Replay {
    pending: Mutex<Vec<Recorded>>,
    fail: HashMap<&'static str, AwsError>,
}

impl Replay {
    fn new(r: &[Recorded]) -> Self {
        Self {
            pending: Mutex::new(r.to_vec()),
            fail: HashMap::new(),
        }
    }
    fn failing(r: &[Recorded], client: &'static str, e: AwsError) -> Self {
        let mut x = Self::new(r);
        x.fail.insert(client, e);
        x
    }
    fn pending(&self) -> Vec<String> {
        self.pending.lock().unwrap().iter().map(|r| r.command.clone()).collect()
    }
}

impl AwsApi for Replay {
    fn call(
        &self,
        client: &'static str,
        command: &'static str,
        input: Value,
    ) -> impl Future<Output = Result<Value, AwsError>> + Send {
        let r = if let Some(e) = self.fail.get(client) {
            Err(e.clone())
        } else {
            let mut p = self.pending.lock().unwrap();
            let i = p
                .iter()
                .position(|r| r.client == client && r.command == command)
                .unwrap_or_else(|| panic!("unexpected {client} {command}"));
            let r = p.remove(i);
            assert_eq!(input, r.input, "{client} {command} input");
            Ok(r.output)
        };
        async move { r }
    }
}

async fn run(api: &Replay, c: &Case, now: i64, state: &mut State) -> (String, Vec<AwsError>) {
    let p = poll(api, &target(c), now, state).await.unwrap();
    (render(&target(c), &p.points), p.errors)
}

fn find<'a>(r: &'a mut [Recorded], command: &str) -> &'a mut Value {
    &mut r.iter_mut().find(|x| x.command == command).unwrap().output
}

fn has(r: &[Recorded], command: &str) -> bool {
    r.iter().any(|x| x.command == command)
}

fn cpu_line(value: &str, time: i64) -> String {
    format!("host_cpu_utilization_percent{{cluster=\"ci\",node_name=\"node-01\"}} {value} {time}\n")
}

fn metric_results(r: &mut [Recorded]) -> &mut Vec<Value> {
    find(r, "GetMetricDataCommand")["MetricDataResults"]
        .as_array_mut()
        .unwrap()
}

fn series<'a>(r: &'a mut [Recorded], id: &str) -> &'a mut Value {
    metric_results(r).iter_mut().find(|s| s["Id"] == id).unwrap()
}

#[tokio::test]
async fn writes_the_golden_host_samples() {
    for name in CASES {
        let c = load(name);
        let api = Replay::new(&c.recorded);
        let (text, errors) = run(&api, &c, c.now, &mut State::new()).await;
        assert_eq!(text, c.expected, "{name}");
        assert!(errors.is_empty(), "{name}: {errors:?}");
        assert!(api.pending().is_empty(), "{name}");
    }
}

// CloudWatch returns the bucket that `now` falls into as a partial Average, and the
// per-timestamp dedupe would freeze it. Buckets closed for less than a period are left for the
// next poll.
#[tokio::test]
async fn the_newest_cloudwatch_buckets_wait_until_closed_for_a_period() {
    for name in CASES {
        let c = load(name);
        let mut early = c.recorded.clone();
        let s = series(&mut early, "host_cpu_utilization_percent");
        s["Timestamps"]
            .as_array_mut()
            .unwrap()
            .insert(0, json!(iso_ms(c.now - 60_000)));
        s["Values"].as_array_mut().unwrap().insert(0, json!(99));
        let (text, _) = run(&Replay::new(&early), &c, c.now, &mut State::new()).await;
        assert_eq!(text, c.expected, "{name}");
        assert!(!text.contains(&cpu_line("99", c.now - 60_000)));
    }
}

// The recorded requests carry an EndTime on the minute. A poll 17 s later sends the same
// CloudWatch and PI requests and the same import text; only the OS log window follows the clock.
#[tokio::test]
async fn an_unaligned_clock_queries_up_to_the_last_minute_boundary() {
    for name in CASES {
        let c = load(name);
        let later = c.now + 17_000;
        let mut shifted = c.recorded.clone();
        if let Some(l) = shifted.iter_mut().find(|r| r.command == "GetLogEventsCommand") {
            l.input["startTime"] = json!(later - 15 * 60_000);
            l.input["endTime"] = json!(later);
        }
        let api = Replay::new(&shifted);
        let (text, _) = run(&api, &c, later, &mut State::new()).await;
        assert_eq!(text, c.expected, "{name}");
        assert!(api.pending().is_empty(), "{name}");
    }
}

// PI stamps a point at the end of its minute. The recorded replies carry a point at `end`
// itself, a minute that closed as the poll started; it is written by the next poll.
#[tokio::test]
async fn performance_insights_points_at_or_after_the_boundary_wait() {
    for name in CASES {
        let c = load(name);
        if !has(&c.recorded, "GetResourceMetricsCommand") {
            continue;
        }
        let mut future = c.recorded.clone();
        let dp = find(&mut future, "GetResourceMetricsCommand")["MetricList"][0]["DataPoints"]
            .as_array_mut()
            .unwrap();
        assert_eq!(dp.last().unwrap()["Timestamp"], json!(iso_ms(c.now)), "{name}");
        let before = dp[dp.len() - 2]["Value"].as_f64().unwrap();
        dp.last_mut().unwrap()["Value"] = json!(66);
        dp.push(json!({ "Timestamp": iso_ms(c.now + 60_000), "Value": 77 }));
        let (text, _) = run(&Replay::new(&future), &c, c.now, &mut State::new()).await;
        assert_eq!(text, c.expected, "{name}");
        assert!(!text.contains(" 66 ") && !text.contains(" 77 "));
        assert!(text.contains(&format!(
            "host_db_load{{cluster=\"ci\",node_name=\"node-01\"}} {} {}\n",
            rds::js_num(before),
            c.now - 60_000
        )));
    }
}

#[tokio::test]
async fn without_performance_insights_no_pi_call_and_no_db_load() {
    for name in CASES {
        let c = load(name);
        let mut off = c.recorded.clone();
        find(&mut off, "DescribeDBInstancesCommand")["DBInstances"][0]["PerformanceInsightsEnabled"] = json!(false);
        let api = Replay::new(&off);
        let (text, errors) = run(&api, &c, c.now, &mut State::new()).await;
        assert!(errors.is_empty());
        let want: String = c
            .expected
            .lines()
            .filter(|l| !l.starts_with("host_db_load"))
            .map(|l| format!("{l}\n"))
            .collect();
        assert_eq!(text, want, "{name}");
        assert_eq!(api.pending(), vec!["GetResourceMetricsCommand"], "{name}");
    }
}

#[tokio::test]
async fn a_pi_or_em_failure_keeps_the_cloudwatch_samples_and_reports_the_error() {
    for name in CASES {
        let c = load(name);
        for (client, command, e, prefix) in [
            (
                "pi",
                "GetResourceMetricsCommand",
                err("ResourceNotFoundException", "stream not found"),
                "host_db_load",
            ),
            (
                "logs",
                "GetLogEventsCommand",
                err("ThrottlingException", "rate exceeded"),
                "host_os_",
            ),
        ] {
            let api = Replay::failing(&c.recorded, client, e.clone());
            let (text, errors) = run(&api, &c, c.now, &mut State::new()).await;
            assert!(text.contains(c.expected.lines().next().unwrap()), "{name}");
            assert!(!text.contains(prefix), "{name} {client}");
            let want: Vec<String> = if has(&c.recorded, command) {
                vec![e.name.clone()]
            } else {
                vec![]
            };
            assert_eq!(
                errors.iter().map(|e| e.name.clone()).collect::<Vec<_>>(),
                want,
                "{name} {client}"
            );
        }
    }
}

fn log_events(r: &mut [Recorded]) -> &mut Vec<Value> {
    find(r, "GetLogEventsCommand")["events"].as_array_mut().unwrap()
}

// The same 15-minute log window comes back on every poll, so a malformed event is reported once
// per state, not once per tick.
#[tokio::test]
async fn a_malformed_em_event_is_skipped_reported_once_and_not_interpolated() {
    for name in CASES {
        let c = load(name);
        if !has(&c.recorded, "GetLogEventsCommand") {
            continue;
        }
        let mut bad = c.recorded.clone();
        let time = c.now - 30_000;
        log_events(&mut bad).extend([
            json!({ "timestamp": time, "message": "not json" }),
            json!({ "timestamp": time, "message": json!({ "timestamp": "yesterday", "loadAverageMinute": { "one": 9.125 } }).to_string() }),
            json!({ "timestamp": time, "message": json!({
                "timestamp": iso_ms(time),
                "cpuUtilization": { "total": format!("1 {time}\nevil{{cluster=\"x\"}} 1"), "wait": "NaN" },
                "memory": {}, "swap": {}, "loadAverageMinute": { "one": 2.5 }, "processList": "x",
            }).to_string() }),
        ]);
        let mut state = State::new();
        let (text, errors) = run(&Replay::new(&bad), &c, c.now, &mut state).await;
        assert!(
            !text.contains("evil") && !text.contains("NaN") && !text.contains(" 9.125 "),
            "{name}"
        );
        assert!(
            text.contains(&format!(
                "host_os_load1{{cluster=\"ci\",node_name=\"node-01\"}} 2.5 {time}\n"
            )),
            "{name}"
        );
        let msgs: Vec<String> = errors.iter().map(|e| e.message.clone()).collect();
        assert_eq!(msgs.len(), 2, "{msgs:?}");
        // the parser's own wording follows the prefix (Bun's in the TypeScript service)
        assert!(
            msgs[0].starts_with(&format!(
                "Enhanced Monitoring event at {time} skipped: JSON Parse error: "
            )),
            "{msgs:?}"
        );
        assert_eq!(
            msgs[1],
            format!("Enhanced Monitoring event at {time} skipped: no timestamp")
        );
        let (again, errors) = run(&Replay::new(&bad), &c, c.now, &mut state).await;
        assert_eq!(again, "");
        assert!(errors.is_empty());
    }
}

// The two id 0 rows are AWS's "OS processes" and "RDS processes" aggregates. A null or rss-less
// row is an AWS quirk, not a reason to drop the series.
#[tokio::test]
async fn the_largest_process_rss_skips_the_id_0_aggregates_and_malformed_rows() {
    for name in CASES {
        let c = load(name);
        if !has(&c.recorded, "GetLogEventsCommand") {
            continue;
        }
        let mut odd = c.recorded.clone();
        let time = c.now - 30_000;
        log_events(&mut odd).push(json!({ "timestamp": time, "message": json!({
            "timestamp": iso_ms(time),
            "processList": [null, { "id": 0, "rss": 500_000 }, { "id": 7 }, { "id": 5, "rss": 10 }, { "id": 6, "rss": 20 }],
        }).to_string() }));
        let (text, errors) = run(&Replay::new(&odd), &c, c.now, &mut State::new()).await;
        assert!(errors.is_empty());
        let line = format!(
            "host_os_process_max_rss_bytes{{cluster=\"ci\",node_name=\"node-01\"}} {} {time}\n",
            20 * 1024
        );
        assert_eq!(text, format!("{}{line}", c.expected), "{name}");
    }
}

#[tokio::test]
async fn a_sample_that_shows_up_late_behind_a_newer_one_is_still_written() {
    for name in CASES {
        let c = load(name);
        let mut late = c.recorded.clone();
        let s = series(&mut late, "host_cpu_utilization_percent");
        let time = s["Timestamps"].as_array_mut().unwrap().remove(1);
        let value = s["Values"].as_array_mut().unwrap().remove(1);
        let mut state = State::new();
        run(&Replay::new(&late), &c, c.now, &mut state).await;
        let (text, _) = run(&Replay::new(&c.recorded), &c, c.now, &mut state).await;
        assert_eq!(
            text,
            cpu_line(
                &rds::js_num(value.as_f64().unwrap()),
                parse_iso(time.as_str().unwrap()).unwrap()
            ),
            "{name}"
        );
    }
}

#[tokio::test]
async fn a_repeated_poll_writes_nothing_already_written() {
    for name in CASES {
        let c = load(name);
        let mut state = State::new();
        run(&Replay::new(&c.recorded), &c, c.now, &mut state).await;
        assert_eq!(
            run(&Replay::new(&c.recorded), &c, c.now, &mut state).await.0,
            "",
            "{name}"
        );
    }
}

// The recorded Aurora cluster was too young to report its cluster-level and Aurora-only
// metrics, so their scales are asserted on a synthetic reply.
#[tokio::test]
async fn aurora_only_series_are_scaled_to_per_second_rates() {
    let c = load("aurora-postgresql");
    let mut syn = c.recorded.clone();
    let time = c.now - 10 * 60_000;
    for (id, v) in [
        ("host_local_storage_free_bytes", 3e10),
        ("host_volume_used_bytes", 5e9),
        ("host_volume_read_iops", 30_000.0),
        ("host_volume_write_iops", 6_000.0),
    ] {
        let s = series(&mut syn, id);
        s["Timestamps"].as_array_mut().unwrap().push(json!(iso_ms(time)));
        s["Values"].as_array_mut().unwrap().push(json!(v));
    }
    // A 300 s bucket 3 minutes old is closed for a 60 s metric, not for a volume one.
    let recent = c.now - 3 * 60_000;
    for id in ["host_volume_used_bytes", "host_local_storage_free_bytes"] {
        let s = series(&mut syn, id);
        s["Timestamps"].as_array_mut().unwrap().push(json!(iso_ms(recent)));
        s["Values"].as_array_mut().unwrap().push(json!(1));
    }
    let (text, _) = run(&Replay::new(&syn), &c, c.now, &mut State::new()).await;
    let l = "{cluster=\"ci\",node_name=\"node-01\"}";
    for want in [
        format!("host_local_storage_free_bytes{l} 30000000000 {time}\n"),
        format!("host_volume_used_bytes{l} 5000000000 {time}\n"),
        format!("host_volume_read_iops{l} 100 {time}\n"),
        format!("host_volume_write_iops{l} 20 {time}\n"),
        format!("host_local_storage_free_bytes{l} 1 {recent}\n"),
    ] {
        assert!(text.contains(&want), "{want}");
    }
    assert!(!text.contains(&format!("host_volume_used_bytes{l} 1 {recent}\n")));
}

// Recorded instances had no throughput/latency samples. Exercise actual values and
// DescribeDBInstances units.
#[tokio::test]
async fn disk_rates_latency_and_provisioned_limits_retain_their_units() {
    let c = load("rds-postgres");
    let mut rec = c.recorded.clone();
    let db = &mut find(&mut rec, "DescribeDBInstancesCommand")["DBInstances"][0];
    db["AllocatedStorage"] = json!(100);
    db["Iops"] = json!(3000);
    db["StorageThroughput"] = json!(125);
    let time = c.now - 3 * 60_000;
    let raw = [
        ("host_disk_read_bytes_per_second", 1048576.0),
        ("host_disk_write_bytes_per_second", 2097152.0),
        ("host_disk_read_latency_seconds", 0.002),
        ("host_disk_write_latency_seconds", 0.004),
    ];
    for (id, v) in raw {
        metric_results(&mut rec).push(json!({ "Id": id, "Timestamps": [iso_ms(time)], "Values": [v] }));
    }
    let (text, errors) = run(&Replay::new(&rec), &c, c.now, &mut State::new()).await;
    assert!(errors.is_empty());
    let l = "{cluster=\"ci\",node_name=\"node-01\"}";
    for (id, v) in raw {
        assert!(text.contains(&format!("{id}{l} {} {time}\n", rds::js_num(v))), "{id}");
    }
    let n = c.now;
    assert!(text.contains(&format!(
        "host_disk_size_bytes{l} {} {n}\n",
        100u64 * 1024 * 1024 * 1024
    )));
    assert!(text.contains(&format!("host_disk_provisioned_iops{l} 3000 {n}\n")));
    assert!(text.contains(&format!(
        "host_disk_provisioned_throughput_bytes_per_second{l} {} {n}\n",
        125 * 1024 * 1024
    )));
    let db = &mut find(&mut rec, "DescribeDBInstancesCommand")["DBInstances"][0];
    db.as_object_mut().unwrap().remove("Iops");
    db.as_object_mut().unwrap().remove("StorageThroughput");
    let (text, _) = run(&Replay::new(&rec), &c, c.now, &mut State::new()).await;
    assert!(!text.contains("host_disk_provisioned_"));
    assert!(text.contains("host_disk_size_bytes"));
}

#[tokio::test]
async fn an_instance_that_is_gone_fails_the_poll() {
    let c = load("rds-postgres");
    let mut rec = c.recorded.clone();
    find(&mut rec, "DescribeDBInstancesCommand")["DBInstances"] = json!([]);
    let e = poll(&Replay::new(&rec), &target(&c), c.now, &mut State::new())
        .await
        .unwrap_err();
    assert_eq!(e.message, "RDS instance not found");
}

#[test]
fn configuration_problems_are_reported_as_rds_host_stats_reports_them() {
    let env = |m: &'static [(&'static str, &'static str)]| {
        move |k: &str| m.iter().find(|(a, _)| *a == k).map(|(_, v)| v.to_string())
    };
    let base: &[(&str, &str)] = &[
        ("RDS_DB_INSTANCE_IDENTIFIER", "db"),
        ("AWS_REGION", "us-east-1"),
        ("PGAI_CLUSTER", "c"),
        ("PGAI_NODE_NAME", "n"),
    ];
    let ok = RdsTarget::from_env(&env(base)).unwrap().unwrap();
    assert_eq!(ok.problem(), None);
    assert_eq!(ok.interval_s(), 60);
    assert!(RdsTarget::from_env(&env(&[])).is_none(), "no instance: off");
    // compose defaults the variables to empty
    let empty = RdsTarget::from_env(&env(&[("RDS_DB_INSTANCE_IDENTIFIER", "db"), ("PGAI_CLUSTER", "")]))
        .unwrap()
        .unwrap();
    assert!(empty.problem().unwrap().contains("PGAI_CLUSTER"));
    let mut role_only = ok.clone();
    role_only.role_arn = Some("arn:aws:iam::123456789012:role/x".into());
    assert!(role_only.problem().unwrap().contains("both be set"));
    let mut ext_only = ok.clone();
    ext_only.external_id = Some("x".into());
    assert!(ext_only.problem().unwrap().contains("both be set"));
    for bad in [0, 301] {
        let mut t = ok.clone();
        t.poll_interval_s = Some(bad);
        assert!(t.problem().unwrap().contains("from 1 to 300"), "{bad}");
    }
    let frac: &[(&str, &str)] = &[
        ("RDS_DB_INSTANCE_IDENTIFIER", "db"),
        ("RDS_POLL_INTERVAL_SECONDS", "1.5"),
    ];
    assert!(RdsTarget::from_env(&env(frac)).unwrap().is_err());
    // configured targets win over the environment
    let cfg = vec![RdsTarget {
        instance_id: "x".into(),
        ..Default::default()
    }];
    assert_eq!(pgai_mon::hoststats::targets(&cfg, &env(base)).len(), 1);
    assert_eq!(
        pgai_mon::hoststats::targets(&cfg, &env(base))[0]
            .as_ref()
            .unwrap()
            .instance_id,
        "x"
    );
}

/// A store with its store thread, and an Ingest into it.
async fn ingest_server(d: &std::path::Path) -> (ingest_common::Server, Arc<Ingest>) {
    let slot = Arc::new(Mutex::new(None));
    let s2 = slot.clone();
    let s = ingest_common::serve(d, Auth::disabled(), move |tx, shared| {
        let cfg = IngestCfg {
            max_age_s: i64::MAX / 4000,
            ..Default::default()
        };
        *s2.lock().unwrap() = Some(Ingest::new(shared, tx, cfg));
        axum::Router::new()
    })
    .await;
    let ing = slot.lock().unwrap().take().unwrap();
    (s, ing)
}

// state is committed only after the write succeeds, so the samples of a failed write are sent
// again on the next tick; the samples land in the store with their series names and labels.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_tick_writes_the_samples_and_resends_them_after_a_failed_write() {
    let c = load("rds-postgres");
    let t = target(&c);
    let d = tempfile::tempdir().unwrap();
    // an ingest whose store is gone: the write fails
    let dead = {
        let (s, _) = ingest_server(d.path()).await;
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        Ingest::new(
            s.shared.clone(),
            tx,
            IngestCfg {
                max_age_s: i64::MAX / 4000,
                ..Default::default()
            },
        )
    };
    let mut state = State::new();
    let logged = Mutex::new(vec![]);
    let log = |m: String| logged.lock().unwrap().push(m);
    let e = tick(&Replay::new(&c.recorded), &dead, &t, c.now, &mut state, &log)
        .await
        .unwrap_err();
    assert!(e.starts_with("write:"), "{e}");
    assert!(state.is_empty(), "nothing committed");
    let d2 = tempfile::tempdir().unwrap();
    let (s, ing) = ingest_server(d2.path()).await;
    let n = tick(&Replay::new(&c.recorded), &ing, &t, c.now, &mut state, &log)
        .await
        .unwrap();
    assert_eq!(n, c.expected.lines().count());
    assert_eq!(state.len(), n);
    assert_eq!(
        tick(&Replay::new(&c.recorded), &ing, &t, c.now, &mut state, &log)
            .await
            .unwrap(),
        0
    );
    assert!(logged.lock().unwrap().is_empty());
    s.flush().await;
    // every expected line is a stored sample, value exact
    let rows = s.sql(
        "SELECT r.metric, r.labels['cluster'], r.labels['node_name'], coalesce(r.labels['provider'], ''), epoch_ms(x.ts), CAST(x.value AS VARCHAR)
         FROM samples x JOIN series r USING (series_id) ORDER BY 1, 5",
    );
    let mut got: Vec<String> = rows
        .iter()
        .map(|r| {
            let v: f64 = r[5].as_str().unwrap().parse().unwrap();
            let p = r[3].as_str().unwrap();
            let p = if p.is_empty() {
                String::new()
            } else {
                format!(",provider=\"{p}\"")
            };
            format!(
                "{}{{cluster={},node_name={}{p}}} {} {}",
                r[0].as_str().unwrap(),
                r[1],
                r[2],
                rds::js_num(v),
                r[4]
            )
        })
        .collect();
    let mut want: Vec<String> = c.expected.lines().map(String::from).collect();
    got.sort();
    want.sort();
    assert_eq!(got, want);
}

// ------------------------------------------------------------------------------------------
// The wire layer against a fake AWS on 127.0.0.1
// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Req {
    method: String,
    path: String,
    action: String,
    body: String,
    authorization: String,
    token: Option<String>,
}

const ROLE_ARN: &str = "arn:aws:iam::123456789012:role/customer";
const EXTERNAL_ID: &str = "postgresai-test-external-id";

async fn fake_aws(sts_fails: bool) -> (String, Arc<Mutex<Vec<Req>>>) {
    let log: Arc<Mutex<Vec<Req>>> = Arc::default();
    let l2 = log.clone();
    let app = axum::Router::new().fallback(move |m: Method, uri: Uri, h: HeaderMap, body: Bytes| {
        let log = l2.clone();
        async move {
            let body = String::from_utf8_lossy(&body).to_string();
            let hv = |k: &str| h.get(k).map(|v| v.to_str().unwrap().to_string());
            let action = body
                .split('&')
                .find_map(|p| p.strip_prefix("Action="))
                .map(String::from)
                .or_else(|| hv("x-amz-target").map(|t| t.rsplit('.').next().unwrap().to_string()))
                .unwrap_or_default();
            log.lock().unwrap().push(Req {
                method: m.to_string(),
                path: uri.path().to_string(),
                action: action.clone(),
                body: body.clone(),
                authorization: hv("authorization").unwrap_or_default(),
                token: hv("x-amz-security-token").or_else(|| hv("x-aws-ec2-metadata-token")),
            });
            let xml = |s: StatusCode, b: &str| (s, [("content-type", "text/xml")], b.to_string());
            match (uri.path(), action.as_str()) {
                ("/latest/api/token", _) if m == Method::PUT => xml(StatusCode::OK, "imds-token"),
                ("/latest/meta-data/iam/security-credentials/", _) => {
                    if hv("x-aws-ec2-metadata-token").as_deref() == Some("imds-token") {
                        xml(StatusCode::OK, "collector-profile")
                    } else {
                        xml(StatusCode::UNAUTHORIZED, "")
                    }
                }
                ("/latest/meta-data/iam/security-credentials/collector-profile", _) => xml(
                    StatusCode::OK,
                    r#"{"Code":"Success","AccessKeyId":"instance","SecretAccessKey":"instance-secret","Token":"instance-token","Expiration":"2099-01-01T00:00:00Z"}"#,
                ),
                (_, "AssumeRole") if sts_fails => xml(
                    StatusCode::FORBIDDEN,
                    "<ErrorResponse><Error><Type>Sender</Type><Code>AccessDenied</Code><Message>offline STS denied</Message></Error><RequestId>r</RequestId></ErrorResponse>",
                ),
                (_, "AssumeRole") => xml(
                    StatusCode::OK,
                    r#"<AssumeRoleResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/"><AssumeRoleResult><Credentials><AccessKeyId>assumed</AccessKeyId><SecretAccessKey>assumed-secret</SecretAccessKey><SessionToken>assumed-token</SessionToken><Expiration>2099-01-01T00:00:00Z</Expiration></Credentials></AssumeRoleResult><ResponseMetadata><RequestId>r</RequestId></ResponseMetadata></AssumeRoleResponse>"#,
                ),
                (_, "DescribeDBInstances") => xml(
                    StatusCode::OK,
                    r#"<DescribeDBInstancesResponse xmlns="http://rds.amazonaws.com/doc/2014-10-31/"><DescribeDBInstancesResult><DBInstances><DBInstance><DBInstanceIdentifier>db</DBInstanceIdentifier><Engine>postgres</Engine><DbiResourceId>db-X</DbiResourceId><PerformanceInsightsEnabled>false</PerformanceInsightsEnabled><MonitoringInterval>0</MonitoringInterval></DBInstance></DBInstances></DescribeDBInstancesResult><ResponseMetadata><RequestId>r</RequestId></ResponseMetadata></DescribeDBInstancesResponse>"#,
                ),
                (_, "GetMetricData") => (
                    StatusCode::OK,
                    [("content-type", "application/x-amz-json-1.0")],
                    // one closed bucket 5 minutes before the end of the queried window
                    json!({ "MetricDataResults": [{ "Id": "host_cpu_utilization_percent",
                        "Timestamps": [serde_json::from_str::<Value>(&body).unwrap()["EndTime"].as_f64().unwrap() as i64 - 300],
                        "Values": [12.5], "StatusCode": "Complete" }] })
                    .to_string(),
                ),
                _ => xml(StatusCode::BAD_REQUEST, "<ErrorResponse><Error><Code>Unexpected</Code><Message>unexpected</Message></Error></ErrorResponse>"),
            }
        }
    });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    (base, log)
}

fn client(base: &str, role: bool, env: &'static [(&'static str, &'static str)]) -> aws::Client {
    let ep = aws::Endpoints {
        region: "us-east-1".into(),
        endpoint_url: Some(base.into()),
        imds: base.into(),
        imds_disabled: false,
    };
    let role = role.then(|| aws::Role {
        arn: ROLE_ARN.into(),
        external_id: EXTERNAL_ID.into(),
    });
    aws::Client::new(
        ep,
        role,
        Box::new(move |k: &str| env.iter().find(|(a, _)| *a == k).map(|(_, v)| v.to_string())),
    )
}

const AMBIENT: &[(&str, &str)] = &[
    ("AWS_ACCESS_KEY_ID", "ambient"),
    ("AWS_SECRET_ACCESS_KEY", "ambient-secret"),
];
const NOW: &str = "2026-10-01T12:00:00Z";
fn db_target() -> Target {
    Target {
        instance_id: "db".into(),
        cluster: "c".into(),
        node_name: "n".into(),
    }
}

#[tokio::test]
async fn a_customer_role_is_assumed_with_its_external_id_and_its_credentials_sign_the_calls() {
    let (base, log) = fake_aws(false).await;
    let c = client(&base, true, AMBIENT);
    let now = parse_iso(NOW).unwrap();
    let p = poll(&c, &db_target(), now, &mut State::new()).await.unwrap();
    assert_eq!(
        render(&db_target(), &p.points),
        "host_provider_info{cluster=\"c\",node_name=\"n\",provider=\"Amazon RDS\"} 1 1790856000000\n\
         host_cpu_utilization_percent{cluster=\"c\",node_name=\"n\"} 12.5 1790855700000\n"
    );
    let reqs = log.lock().unwrap().clone();
    // one AssumeRole: the assumed credentials are cached until shortly before they expire
    assert_eq!(
        reqs.iter().map(|r| r.action.as_str()).collect::<Vec<_>>(),
        ["AssumeRole", "DescribeDBInstances", "GetMetricData"]
    );
    let sts = &reqs[0];
    assert!(
        sts.body
            .contains(&format!("RoleArn={}", aws::uri_encode(ROLE_ARN, false))),
        "{}",
        sts.body
    );
    assert!(sts.body.contains(&format!("ExternalId={EXTERNAL_ID}")));
    assert!(
        sts.body.contains("RoleSessionName=postgresai-rds-host-stats") && sts.body.contains("DurationSeconds=3600")
    );
    assert!(
        sts.authorization.contains("Credential=ambient/") && sts.authorization.contains("/us-east-1/sts/aws4_request")
    );
    for r in &reqs[1..] {
        assert!(r.authorization.contains("Credential=assumed/"), "{r:?}");
        assert_eq!(r.token.as_deref(), Some("assumed-token"));
    }
    assert!(reqs[1].authorization.contains("/rds/aws4_request"));
    assert!(reqs[2].authorization.contains("/monitoring/aws4_request"));
    // the JSON protocol carries epoch seconds
    let body: Value = serde_json::from_str(&reqs[2].body).unwrap();
    assert_eq!(body["EndTime"], json!(1790856000.0));
    assert_eq!(body["MetricDataQueries"].as_array().unwrap().len(), 11);
}

#[tokio::test]
async fn an_sts_failure_writes_nothing_and_never_queries_rds_or_cloudwatch() {
    let (base, log) = fake_aws(true).await;
    let c = client(&base, true, AMBIENT);
    let e = poll(&c, &db_target(), parse_iso(NOW).unwrap(), &mut State::new())
        .await
        .unwrap_err();
    assert_eq!(e.to_string(), "AccessDenied: offline STS denied");
    assert_eq!(
        log.lock().unwrap().iter().map(|r| r.action.clone()).collect::<Vec<_>>(),
        ["AssumeRole"]
    );
}

#[tokio::test]
async fn without_keys_in_the_environment_the_instance_profile_comes_from_imdsv2() {
    let (base, log) = fake_aws(false).await;
    let c = client(&base, false, &[]);
    let p = poll(&c, &db_target(), parse_iso(NOW).unwrap(), &mut State::new())
        .await
        .unwrap();
    assert_eq!(p.points.len(), 2);
    let reqs = log.lock().unwrap().clone();
    assert_eq!(
        reqs.iter()
            .map(|r| format!("{} {}", r.method, r.path))
            .collect::<Vec<_>>()[..3],
        [
            "PUT /latest/api/token",
            "GET /latest/meta-data/iam/security-credentials/",
            "GET /latest/meta-data/iam/security-credentials/collector-profile"
        ]
    );
    for r in &reqs[3..] {
        assert!(r.authorization.contains("Credential=instance/"), "{r:?}");
        assert_eq!(r.token.as_deref(), Some("instance-token"));
    }
}

/// The product command (`run`, whose serve loop the image starts) polls RDS from the rds-host-stats environment (fake AWS on 127.0.0.1) and its
/// samples are queryable through /sql with the labels rds-host-stats gave them.
#[test]
fn run_polls_rds_from_the_environment_and_stores_the_host_series() {
    use std::io::{Read, Write};
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (base, log) = rt.block_on(fake_aws(false));
    let d = tempfile::tempdir().unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_pgai-mon"));
    let cfg = d.path().join("pgai-mon.yml");
    std::fs::write(
        &cfg,
        format!(
            "store_dir: {}\nstore: {{flush_interval_s: 1}}\nhttp: {{listen: \"127.0.0.1:{port}\", auth: {{disabled: true}}}}\n",
            d.path().join("s").display()
        ),
    )
    .unwrap();
    cmd.args(["run", "--config", cfg.to_str().unwrap()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    for k in [
        "AWS_PROFILE",
        "AWS_SESSION_TOKEN",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "RDS_ROLE_ARN",
        "RDS_EXTERNAL_ID",
    ] {
        cmd.env_remove(k);
    }
    for (k, v) in [
        ("RDS_DB_INSTANCE_IDENTIFIER", "db"),
        ("AWS_REGION", "us-east-1"),
        ("PGAI_CLUSTER", "c"),
        ("PGAI_NODE_NAME", "n"),
        ("RDS_POLL_INTERVAL_SECONDS", "60"),
        ("AWS_ENDPOINT_URL", base.as_str()),
        ("AWS_EC2_METADATA_DISABLED", "true"),
        ("AWS_ACCESS_KEY_ID", "ambient"),
        ("AWS_SECRET_ACCESS_KEY", "ambient-secret"),
    ] {
        cmd.env(k, v);
    }
    struct Kill(std::process::Child);
    impl Drop for Kill {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let _child = Kill(cmd.spawn().unwrap());
    let sql = |q: &str| -> Option<Value> {
        let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
        write!(c, "POST /sql HTTP/1.1\r\nHost: x\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{q}", q.len())
            .ok()?;
        let mut raw = String::new();
        c.read_to_string(&mut raw).ok()?;
        let (head, body) = raw.split_once("\r\n\r\n")?;
        if !head.starts_with("HTTP/1.1 200") || head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
            return None;
        }
        serde_json::from_str(body).ok()
    };
    let q = "SELECT r.metric, array_to_string(list_transform(list_sort(map_entries(r.labels)), e -> e.key || '=' || e.value), ','), s.value
             FROM samples s JOIN series r USING (series_id) WHERE r.metric LIKE 'host_%' AND NOT s.stale ORDER BY 1";
    let mut rows = vec![];
    for _ in 0..300 {
        if let Some(r) = sql(q) {
            rows = r["rows"].as_array().cloned().unwrap_or_default();
            if rows.len() >= 2 {
                break;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert_eq!(
        rows,
        vec![
            json!(["host_cpu_utilization_percent", "cluster=c,node_name=n", 12.5]),
            json!(["host_provider_info", "cluster=c,node_name=n,provider=Amazon RDS", 1.0]),
        ],
        "AWS calls: {:?}",
        log.lock().unwrap().iter().map(|r| r.action.clone()).collect::<Vec<_>>()
    );
    let reqs = log.lock().unwrap().clone();
    assert_eq!(reqs[0].action, "DescribeDBInstances");
    assert!(
        reqs[0].authorization.contains("Credential=ambient/")
            && reqs[0].authorization.contains("/us-east-1/rds/aws4_request")
    );
}
