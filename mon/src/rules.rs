//! Recording rules materialised at ingest: `sql/host_rules.sql` (the old vmalert host_rules.yml
//! as SQL) is evaluated every 60 s at the minute and its rows are written back as `host_*`
//! series, with staleness markers for host_* series a later evaluation no longer produces.
//!
//! Why materialise rather than define views evaluated at query time:
//! - consumers keep reading `host_*` by name through the same prom_* functions and rollups as
//!   every other series (Dashboard 1's Host row, Dashboard 99, reports, sizing rules), and RDS
//!   writes the same names directly; a view would need its own code path in every consumer;
//! - the values are the ones vmalert wrote (one evaluation per minute over a 3 m window), so the
//!   golden cases of old/tests/host_rules carry over unchanged, and a 30-day panel reads one
//!   precomputed series per node (and its 1 h / 1 d rollups) instead of re-deriving rates of
//!   every per-CPU and per-device counter at each refresh. A ratio of rollups is not the rollup of
//!   a ratio, so query-time views over long ranges would also give different numbers.
use crate::ingest::{ExtSample, Ingest, WriteReport};
use crate::query::Backend;
use crate::Msg;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

pub const HOST_RULES_SQL: &str = include_str!("../sql/host_rules.sql");

/// Staleness group of the rule outputs.
pub const HOST_GROUP: &str = "rules:host";

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct RulesCfg {
    /// Evaluate the host_* rules: None = when scraping is configured (the only producer of their
    /// inputs), Some(b) = always / never.
    pub host: Option<bool>,
    /// Seconds between evaluations (the old rule groups: 60).
    pub interval_s: u64,
    /// Seconds after the evaluation instant before it runs (lets scrapes at the boundary land).
    pub delay_s: u64,
}
impl Default for RulesCfg {
    fn default() -> Self {
        Self {
            host: None,
            interval_s: 60,
            delay_s: 5,
        }
    }
}

/// The rules query for instant `t_ms`.
pub fn host_rules_sql(t_ms: i64) -> String {
    HOST_RULES_SQL.replace("getvariable('t_ms')", &t_ms.to_string())
}

/// Rows (metric, cluster, node_name, provider, value as text) of a rows-format reply.
pub fn samples_from_reply(body: &str, t_ms: i64) -> anyhow::Result<Vec<ExtSample>> {
    #[derive(Deserialize)]
    struct Body {
        rows: Vec<(String, String, String, Option<String>, String)>,
        #[serde(default)]
        truncated: bool,
    }
    let b: Body = serde_json::from_str(body)?;
    anyhow::ensure!(!b.truncated, "host rules result truncated");
    b.rows
        .into_iter()
        .map(|(metric, cluster, node_name, provider, v)| {
            let mut labels = vec![("cluster".to_string(), cluster), ("node_name".to_string(), node_name)];
            if let Some(p) = provider {
                labels.push(("provider".to_string(), p));
            }
            Ok(ExtSample {
                metric,
                labels,
                ts: t_ms,
                value: v.parse().map_err(|_| anyhow::anyhow!("bad value {v:?}"))?,
            })
        })
        .collect()
}

/// Evaluate the host rules at `t_ms` over what the store holds and write the results.
pub async fn eval_host_once(backend: &Backend, ing: &Ingest, t_ms: i64) -> anyhow::Result<WriteReport> {
    let reply = backend
        .call("sql", host_rules_sql(t_ms))
        .await
        .map_err(|e| anyhow::anyhow!("host rules: {}: {}", e.kind, e.message))?;
    let samples = ing
        .validate(samples_from_reply(&reply.body, t_ms)?)
        .map_err(|e| anyhow::anyhow!(e))?;
    ing.write(samples, Some((HOST_GROUP, t_ms))).await
}

/// Every `interval_s`, at the interval boundary plus `delay_s`: flush the store (so the latest
/// scrapes are visible to SQL), then evaluate at the boundary. Missed boundaries are skipped.
pub async fn run_host(cfg: RulesCfg, backend: Backend, ing: Arc<Ingest>, tx: tokio::sync::mpsc::Sender<Msg>) {
    let iv = cfg.interval_s.max(1) as i64 * 1000;
    let delay = cfg.delay_s as i64 * 1000;
    loop {
        let now = crate::model::now_ms();
        let mut t = (now - delay).div_euclid(iv) * iv + iv;
        if t + delay <= now {
            t += iv;
        }
        tokio::time::sleep(Duration::from_millis((t + delay - now).max(0) as u64)).await;
        let (a, b) = tokio::sync::oneshot::channel();
        if tx.send(Msg::Flush(Some(a))).await.is_err() || b.await.is_err() {
            return;
        }
        if let Err(e) = eval_host_once(&backend, &ing, t).await {
            eprintln!("rules: {e}");
        }
    }
}
