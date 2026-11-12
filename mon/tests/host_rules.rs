//! The golden cases of the old vmalert host rules (tests/fixtures/host_rules/host_rules.test.yml,
//! a copy of old/tests/host_rules/host_rules.test.yml that vmalert-tool ran) against pgai-mon: the
//! input series are ingested, sql/host_rules.sql is evaluated every minute as `run` does, and
//! each expression is checked on the host_* series written back.
mod ingest_common;
use pgai_mon::auth::Auth;
use pgai_mon::ingest::{parse_line, ExtSample, Ingest, IngestCfg};
use pgai_mon::query::Backend;
use pgai_mon::rules::eval_host_once;
use serde::Deserialize;
use std::sync::{Arc, Mutex};

#[derive(Deserialize)]
struct File {
    tests: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    interval: String,
    input_series: Vec<Input>,
    metricsql_expr_test: Vec<ExprTest>,
}
#[derive(Deserialize)]
struct Input {
    series: String,
    values: String,
}
#[derive(Deserialize)]
struct ExprTest {
    expr: String,
    eval_time: String,
    exp_samples: Vec<Expected>,
}
#[derive(Deserialize)]
struct Expected {
    labels: String,
    value: f64,
}

/// A fixed instant on a minute boundary: test time 0.
const BASE: i64 = 1_790_726_400_000;

fn dur_ms(s: &str) -> i64 {
    pgai_mon::scrape::parse_duration(s).unwrap().as_millis() as i64
}

/// vmalert-tool / promtool series notation: `a+bxN`, `a-bxN`, `axN`, `_`, `_xN`, numbers.
fn expand(values: &str) -> Vec<Option<f64>> {
    let mut out = vec![];
    for tok in values.split_whitespace() {
        if let Some((head, n)) = tok.rsplit_once('x') {
            let n: usize = n.parse().unwrap();
            if head == "_" {
                out.extend(std::iter::repeat(None).take(n + 1));
                continue;
            }
            // a+b / a-b (the sign after the first character), or a alone
            let split = head[1..].find(['+', '-']).map(|i| i + 1);
            let (a, b) = match split {
                Some(i) => (head[..i].parse::<f64>().unwrap(), head[i..].parse::<f64>().unwrap()),
                None => (head.parse::<f64>().unwrap(), 0.0),
            };
            for k in 0..=n {
                out.push(Some(a + b * k as f64));
            }
        } else if tok == "_" {
            out.push(None);
        } else {
            out.push(Some(tok.parse().unwrap()));
        }
    }
    out
}

enum Sel {
    Name(String),
    Re(regex::Regex),
}

/// The expression shapes the golden file uses: `name`, `round(name)`, `round(name, 1e-9)`,
/// `{__name__=~"re"}`.
fn parse_expr(e: &str) -> (Sel, Option<f64>) {
    let e = e.trim();
    if let Some(r) = e.strip_prefix("{__name__=~\"").and_then(|r| r.strip_suffix("\"}")) {
        return (Sel::Re(regex::Regex::new(&format!("^(?:{r})$")).unwrap()), None);
    }
    if let Some(inner) = e.strip_prefix("round(").and_then(|r| r.strip_suffix(')')) {
        let (name, to) = match inner.split_once(',') {
            Some((n, t)) => (n.trim(), t.trim().parse().unwrap()),
            None => (inner.trim(), 1.0),
        };
        return (Sel::Name(name.into()), Some(to));
    }
    assert!(
        e.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_'),
        "unsupported expr {e}"
    );
    (Sel::Name(e.into()), None)
}

fn canon(metric: &str, labels: &[(String, String)]) -> String {
    let mut l = labels.to_vec();
    l.sort();
    format!(
        "{metric}{{{}}}",
        l.iter()
            .map(|(k, v)| format!("{k}={v:?}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

async fn run_case(case: &Case) {
    let d = tempfile::tempdir().unwrap();
    let slot = Arc::new(Mutex::new(None));
    let s2 = slot.clone();
    let s = ingest_common::serve(d.path(), Auth::disabled(), move |tx, shared| {
        let cfg = IngestCfg {
            max_age_s: i64::MAX / 4000,
            max_future_s: i64::MAX / 4000,
            ..Default::default()
        };
        *s2.lock().unwrap() = Some(Ingest::new(shared, tx, cfg));
        axum::Router::new()
    })
    .await;
    let ing: Arc<Ingest> = slot.lock().unwrap().take().unwrap();
    let step = dur_ms(&case.interval);
    let mut samples = vec![];
    for i in &case.input_series {
        let l = parse_line(&format!("{} 0", i.series)).unwrap();
        for (k, v) in expand(&i.values).into_iter().enumerate() {
            if let Some(v) = v {
                samples.push(ExtSample {
                    metric: l.metric.clone(),
                    labels: l.labels.clone(),
                    ts: BASE + k as i64 * step,
                    value: v,
                });
            }
        }
    }
    ing.write(ing.validate(samples).unwrap(), None).await.unwrap();
    s.flush().await;
    let last = case
        .metricsql_expr_test
        .iter()
        .map(|t| dur_ms(&t.eval_time))
        .max()
        .unwrap();
    let backend = Backend::InProc(s.reader.clone());
    // one evaluation per minute from test time 0, as vmalert-tool evaluates the 60 s groups
    let mut t = 0;
    let t0 = std::time::Instant::now();
    while t <= last {
        eval_host_once(&backend, &ing, BASE + t).await.unwrap();
        t += 60_000;
    }
    eprintln!(
        "{}: {:.1} ms per evaluation",
        case.name,
        t0.elapsed().as_secs_f64() * 1000.0 / (last / 60_000 + 1) as f64
    );
    s.flush().await;
    for et in &case.metricsql_expr_test {
        let at = BASE + dur_ms(&et.eval_time);
        // instant query: the last sample of each host_* series in (t - 5m, t], unless it is a
        // staleness marker
        let rows = s.sql(&format!(
            "SELECT r.metric, map_keys(r.labels), map_values(r.labels), CAST(x.value AS VARCHAR) FROM (
               SELECT series_id, value, stale, row_number() OVER (PARTITION BY series_id ORDER BY ts DESC) AS rn
               FROM samples WHERE epoch_ms(ts) > {} AND epoch_ms(ts) <= {at}) x
             JOIN series r USING (series_id) WHERE x.rn = 1 AND NOT x.stale AND r.metric LIKE 'host\\_%' ESCAPE '\\'",
            at - 300_000
        ));
        let (sel, round) = parse_expr(&et.expr);
        let mut got: Vec<(String, f64)> = rows
            .iter()
            .filter(|r| {
                let m = r[0].as_str().unwrap();
                match &sel {
                    Sel::Name(n) => m == n,
                    Sel::Re(re) => re.is_match(m),
                }
            })
            .map(|r| {
                let keys = r[1].as_array().unwrap();
                let vals = r[2].as_array().unwrap();
                let labels: Vec<(String, String)> = keys
                    .iter()
                    .zip(vals)
                    .map(|(k, v)| (k.as_str().unwrap().to_string(), v.as_str().unwrap().to_string()))
                    .collect();
                let mut v: f64 = r[3].as_str().unwrap().parse().unwrap();
                if let Some(to) = round {
                    v = (v / to).round() * to;
                }
                (canon(r[0].as_str().unwrap(), &labels), v)
            })
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        let mut want: Vec<(String, f64)> = et
            .exp_samples
            .iter()
            .map(|e| {
                let l = parse_line(&format!("{} 0", e.labels)).unwrap();
                (canon(&l.metric, &l.labels), e.value)
            })
            .collect();
        want.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            got.iter().map(|g| &g.0).collect::<Vec<_>>(),
            want.iter().map(|w| &w.0).collect::<Vec<_>>(),
            "{}: {} at {}",
            case.name,
            et.expr,
            et.eval_time
        );
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g.1 - w.1).abs() <= 1e-9 * w.1.abs().max(1.0),
                "{}: {} at {}: got {} want {}",
                case.name,
                et.expr,
                et.eval_time,
                g.1,
                w.1
            );
        }
    }
}

fn load(name: &str) -> File {
    serde_yaml::from_str(
        &std::fs::read_to_string(format!(
            "{}/tests/fixtures/host_rules/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap(),
    )
    .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn extra_cases() {
    let f = load("host_rules_extra.test.yml");
    assert_eq!(f.tests.len(), 2);
    for case in &f.tests {
        run_case(case).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn golden_host_rules_cases() {
    let f: File = serde_yaml::from_str(
        &std::fs::read_to_string(format!(
            "{}/tests/fixtures/host_rules/host_rules.test.yml",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(f.tests.len(), 9, "every golden case runs");
    for case in &f.tests {
        run_case(case).await;
    }
}

#[test]
fn series_notation() {
    assert_eq!(expand("1+2x2"), vec![Some(1.0), Some(3.0), Some(5.0)]);
    assert_eq!(expand("5-1x1 _ 7"), vec![Some(5.0), Some(4.0), None, Some(7.0)]);
    assert_eq!(expand("37.5x1"), vec![Some(37.5), Some(37.5)]);
    assert_eq!(expand("1790731200+60x1"), vec![Some(1790731200.0), Some(1790731260.0)]);
}

#[test]
fn the_rules_sql_has_all_37_rules() {
    // each provider block names its outputs; 22 Supabase + 15 ClickHouse = the 37 vmalert rules
    let sql = pgai_mon::rules::HOST_RULES_SQL;
    let block = |from: &str, to: &str| sql[sql.find(from).unwrap()..sql.find(to).unwrap()].to_string();
    let count = |b: &str| {
        let re = regex::Regex::new(r"'(host_[a-z0-9_]+)'").unwrap();
        let mut v: Vec<String> = re.captures_iter(b).map(|c| c[1].to_string()).collect();
        v.sort();
        v.dedup();
        v
    };
    let sb = count(&block("supabase AS (", "ch_cpu AS ("));
    let ch = block("clickhouse AS (", "SELECT metric, cluster");
    let mut ch_names = count(&ch);
    // the four ClickHouse counters are named in ch_ctr
    ch_names.extend(count(&block("ch_ctr AS (", "ch_svc AS (")));
    ch_names.sort();
    ch_names.dedup();
    assert_eq!(sb.len(), 22, "{sb:?}");
    assert_eq!(ch_names.len(), 15, "{ch_names:?}");
}
