//! pgwatch metrics.yml, interpreted the way pgwatch v3 does:
//! - `sqls` keyed by Postgres major; the highest key <= server major wins.
//! - `gauges`: '*' or a column list; everything else numeric is a counter (TYPE only,
//!   the stored value is the same).
//! - `is_instance_level`: run once per node (on the configured database), not per database.
//! - `node_status`: primary | standby: run only on that role.
//! - `statement_timeout_seconds` per metric.
//! - Columns named `tag_*` become labels (prefix stripped); `epoch_ns` is the timestamp;
//!   other numeric/bool columns become `pgwatch_<metric>_<column>` series; text is never a label.
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Deserialize)]
pub struct MetricsFile {
    pub metrics: BTreeMap<String, MetricDef>,
    #[serde(default)]
    pub presets: BTreeMap<String, Preset>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Preset {
    #[serde(default)]
    pub description: Option<String>,
    pub metrics: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MetricDef {
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub sqls: serde_yaml::Mapping,
    #[serde(default)]
    pub gauges: Option<Vec<String>>,
    #[serde(default)]
    pub is_instance_level: bool,
    #[serde(default)]
    pub node_status: Option<String>,
    #[serde(default)]
    pub statement_timeout_seconds: Option<u64>,
    #[serde(default)]
    pub metric_storage_name: Option<String>,
    #[serde(default)]
    pub init_sql: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Scheduled {
    pub name: String,
    pub interval_s: u64,
    pub def: MetricDef,
}

impl MetricDef {
    /// SQL for a server major version (pgwatch: max key <= version).
    pub fn sql_for(&self, major: u32) -> Option<&str> {
        let mut best: Option<(f64, &str)> = None;
        for (k, v) in &self.sqls {
            let Some(v) = v.as_str() else { continue };
            let kv = match k {
                serde_yaml::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                serde_yaml::Value::String(s) => s.parse().unwrap_or(0.0),
                _ => continue,
            };
            if kv <= major as f64 && best.map_or(true, |(b, _)| kv > b) {
                best = Some((kv, v));
            }
        }
        let sql = best?.1.trim();
        // pgwatch convention: a SQL that is just ";" (plus a comment) means "not on this version".
        if sql.is_empty() || sql.starts_with(';') {
            return None;
        }
        Some(sql)
    }

    pub fn is_gauge(&self, column: &str) -> bool {
        match &self.gauges {
            Some(g) => g.iter().any(|c| c == "*" || c == column),
            None => false,
        }
    }

    pub fn storage_name<'a>(&'a self, name: &'a str) -> &'a str {
        self.metric_storage_name.as_deref().unwrap_or(name)
    }
}

pub fn load(path: &std::path::Path) -> anyhow::Result<MetricsFile> {
    let s = std::fs::read_to_string(path)?;
    Ok(serde_yaml::from_str(&s)?)
}

impl MetricsFile {
    pub fn scheduled(&self, preset: &str) -> anyhow::Result<Vec<Scheduled>> {
        let p = self
            .presets
            .get(preset)
            .ok_or_else(|| anyhow::anyhow!("preset {preset} not in metrics.yml"))?;
        let mut out = Vec::new();
        for (name, interval) in &p.metrics {
            match self.metrics.get(name) {
                Some(def) => out.push(Scheduled {
                    name: name.clone(),
                    interval_s: *interval,
                    def: def.clone(),
                }),
                None => eprintln!("warn: preset {preset} lists unknown metric {name}"),
            }
        }
        Ok(out)
    }
}

/// Prometheus name exactly as pgwatch's sink builds it: `pgwatch_<metric>_<field>`, or
/// `pgwatch_<metric>` when the field equals the metric name.
pub fn prom_name(metric: &str, field: &str) -> String {
    if metric == field {
        format!("pgwatch_{metric}")
    } else {
        format!("pgwatch_{metric}_{field}")
    }
}
