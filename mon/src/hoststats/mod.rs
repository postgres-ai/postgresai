//! Host stats of managed instances that have no host to scrape. RDS and Aurora PostgreSQL: the
//! rds-host-stats service inside the binary (rds.rs: the poll; aws.rs: the AWS calls), writing
//! the same host_* series through the ingest path instead of a remote-write to VictoriaMetrics.
pub mod aws;
pub mod rds;

use crate::ingest::{ExtSample, Ingest};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// One RDS/Aurora instance to poll.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct RdsTarget {
    /// DBInstanceIdentifier.
    pub instance_id: String,
    /// AWS region of the instance.
    pub region: String,
    /// `cluster` and `node_name` labels: the target's pgwatch labels.
    pub cluster: String,
    pub node_name: String,
    /// A read-only role in the instance's account to assume (with `external_id`), or neither.
    pub role_arn: Option<String>,
    pub external_id: Option<String>,
    /// Seconds between polls, 1..=300 (default 60).
    pub poll_interval_s: Option<u64>,
}

impl RdsTarget {
    /// The rds-host-stats environment: RDS_DB_INSTANCE_IDENTIFIER, AWS_REGION, PGAI_CLUSTER,
    /// PGAI_NODE_NAME, RDS_ROLE_ARN, RDS_EXTERNAL_ID, RDS_POLL_INTERVAL_SECONDS. None when
    /// RDS_DB_INSTANCE_IDENTIFIER is unset or empty.
    pub fn from_env(env: &dyn Fn(&str) -> Option<String>) -> Option<Result<Self, String>> {
        let get = |k: &str| env(k).filter(|v| !v.is_empty());
        let instance_id = get("RDS_DB_INSTANCE_IDENTIFIER")?;
        let poll_interval_s = match get("RDS_POLL_INTERVAL_SECONDS") {
            None => None,
            Some(v) => match v.trim().parse::<u64>() {
                Ok(n) => Some(n),
                Err(_) => return Some(Err("RDS_POLL_INTERVAL_SECONDS must be an integer from 1 to 300".into())),
            },
        };
        Some(Ok(Self {
            instance_id,
            region: get("AWS_REGION").unwrap_or_default(),
            cluster: get("PGAI_CLUSTER").unwrap_or_default(),
            node_name: get("PGAI_NODE_NAME").unwrap_or_default(),
            role_arn: get("RDS_ROLE_ARN"),
            external_id: get("RDS_EXTERNAL_ID"),
            poll_interval_s,
        }))
    }

    /// The first configuration problem, as rds-host-stats reports it.
    pub fn problem(&self) -> Option<String> {
        let set = |o: &Option<String>| o.as_deref().is_some_and(|v| !v.is_empty());
        if set(&self.role_arn) != set(&self.external_id) {
            return Some(
                "role_arn and external_id (RDS_ROLE_ARN, RDS_EXTERNAL_ID) must both be set or both be unset".into(),
            );
        }
        for (k, v) in [
            ("instance_id (RDS_DB_INSTANCE_IDENTIFIER)", &self.instance_id),
            ("cluster (PGAI_CLUSTER)", &self.cluster),
            ("node_name (PGAI_NODE_NAME)", &self.node_name),
            ("region (AWS_REGION)", &self.region),
        ] {
            if v.is_empty() {
                return Some(format!("{k} is required"));
            }
        }
        // A 300 s bucket is written by a poll that ends 10 to 15 minutes after it starts, so a
        // longer interval would skip Aurora volume buckets.
        if !(1..=300).contains(&self.interval_s()) {
            return Some("poll_interval_s (RDS_POLL_INTERVAL_SECONDS) must be an integer from 1 to 300".into());
        }
        None
    }

    pub fn interval_s(&self) -> u64 {
        self.poll_interval_s.unwrap_or(60)
    }

    pub fn target(&self) -> rds::Target {
        rds::Target {
            instance_id: self.instance_id.clone(),
            cluster: self.cluster.clone(),
            node_name: self.node_name.clone(),
            region: self.region.clone(),
        }
    }

    pub fn role(&self) -> Option<aws::Role> {
        match (&self.role_arn, &self.external_id) {
            (Some(a), Some(e)) if !a.is_empty() && !e.is_empty() => Some(aws::Role {
                arn: a.clone(),
                external_id: e.clone(),
            }),
            _ => None,
        }
    }
}

/// Configured targets, or the rds-host-stats environment when none are.
pub fn targets(cfg: &[RdsTarget], env: &dyn Fn(&str) -> Option<String>) -> Vec<Result<RdsTarget, String>> {
    if !cfg.is_empty() {
        return cfg.iter().cloned().map(Ok).collect();
    }
    RdsTarget::from_env(env).into_iter().collect()
}

/// The samples of a poll for the ingest path.
pub fn samples(t: &rds::Target, points: &[rds::Point]) -> Vec<ExtSample> {
    points
        .iter()
        .map(|p| ExtSample {
            metric: p.name.to_string(),
            labels: rds::labels(t, p),
            ts: p.time,
            value: p.value,
        })
        .collect()
}

/// A poll that takes longer than this is abandoned and the AWS client (its connections) rebuilt.
pub const POLL_DEADLINE: Duration = Duration::from_secs(30);

/// One tick: poll on a copy of `state`, write, and keep the copy only once the write succeeded,
/// so the samples of a failed write are sent again on the next tick. Returns samples written.
pub async fn tick(
    api: &impl aws::AwsApi,
    ing: &Ingest,
    target: &rds::Target,
    now: i64,
    state: &mut rds::State,
    log: &(dyn Fn(String) + Sync),
) -> Result<usize, String> {
    let mut pending = state.clone();
    let p = match tokio::time::timeout(POLL_DEADLINE, rds::poll(api, target, now, &mut pending)).await {
        Err(_) => return Err("PollTimeout: poll timed out after 30 s".into()),
        Ok(Err(e)) => return Err(e.to_string()),
        Ok(Ok(p)) => p,
    };
    for e in &p.errors {
        log(e.to_string());
    }
    let n = p.points.len();
    if n > 0 {
        let s = ing.validate(samples(target, &p.points))?;
        ing.write(s, None).await.map_err(|e| format!("write: {e}"))?;
    }
    *state = pending;
    Ok(n)
}

/// Poll one target forever. An invalid configuration is logged once and the poller idles.
pub async fn run(cfg: Result<RdsTarget, String>, ing: Arc<Ingest>) {
    let name = cfg.as_ref().map(|c| c.instance_id.clone()).unwrap_or_default();
    let log = move |m: String| eprintln!("rds-host-stats[{name}]: {m}");
    let cfg = match cfg.and_then(|c| c.problem().map_or(Ok(c), Err)) {
        Ok(c) => c,
        Err(e) => {
            log(format!(
                "{e}; RDS host stats for this target stay off until the configuration is fixed"
            ));
            return;
        }
    };
    let env = |k: &str| std::env::var(k).ok();
    let make = || {
        aws::Client::new(
            aws::Endpoints::from_env(&cfg.region, &env),
            cfg.role(),
            Box::new(|k: &str| std::env::var(k).ok()),
        )
    };
    let mut client = make();
    let target = cfg.target();
    let iv = cfg.interval_s() as i64 * 1000;
    let mut state = rds::State::new();
    let mut next = crate::model::now_ms();
    loop {
        match tick(&client, &ing, &target, crate::model::now_ms(), &mut state, &log).await {
            Ok(n) => log(format!("samples written: {n}")),
            Err(e) => {
                log(e.clone());
                // a stalled connection would outlive the poll; a new client drops it
                if e.starts_with("PollTimeout") {
                    client = make();
                }
            }
        }
        let now = crate::model::now_ms();
        loop {
            next += iv;
            if next > now {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis((next - now) as u64)).await;
    }
}
