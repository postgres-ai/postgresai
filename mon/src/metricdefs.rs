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
    /// pgwatch v3 spells it `storage_name`; `metric_storage_name` is accepted too.
    #[serde(default, alias = "storage_name")]
    pub metric_storage_name: Option<String>,
    /// Parsed, not run: pgwatch runs it only to create helper functions (with --create-helpers),
    /// and the collector never writes to a monitored database.
    #[serde(default)]
    pub init_sql: Option<String>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_yaml::Value>,
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
    parse(&s)
}

/// Parse metrics.yml; unknown metric fields are reported, not fatal.
pub fn parse(s: &str) -> anyhow::Result<MetricsFile> {
    let mf: MetricsFile = serde_yaml::from_str(s)?;
    for (name, d) in &mf.metrics {
        for k in d.unknown.keys() {
            eprintln!("metrics.yml: {name}: unknown field {k} ignored");
        }
    }
    Ok(mf)
}

impl MetricsFile {
    pub fn scheduled(&self, preset: &str) -> anyhow::Result<Vec<Scheduled>> {
        let p = self
            .presets
            .get(preset)
            .ok_or_else(|| anyhow::anyhow!("preset {preset} not in metrics.yml"))?;
        let list: Vec<(String, f64)> = p.metrics.iter().map(|(k, v)| (k.clone(), *v as f64)).collect();
        Ok(self.schedule(&list, &format!("preset {preset}")))
    }

    /// Metric list with intervals; unknown metrics are reported and left out, and so is an
    /// interval <= 0 (pgwatch: the metric is off).
    fn schedule(&self, list: &[(String, f64)], what: &str) -> Vec<Scheduled> {
        let mut out = Vec::new();
        for (name, interval) in list {
            if *interval <= 0.0 {
                continue;
            }
            match self.metrics.get(name) {
                Some(def) => out.push(Scheduled {
                    name: name.clone(),
                    interval_s: (interval.round() as u64).max(1),
                    def: def.clone(),
                }),
                None => eprintln!("warn: {what} lists unknown metric {name}"),
            }
        }
        out
    }

    /// pgwatch's choice for a source: preset_metrics wins over custom_metrics; neither: the
    /// global preset. The standby list (preset_metrics_standby, else custom_metrics_standby)
    /// may be empty: then a standby runs the primary list.
    pub fn for_source(
        &self,
        src: &crate::config::Source,
        default_preset: &str,
    ) -> anyhow::Result<(Vec<Scheduled>, Vec<Scheduled>)> {
        let pick = |preset: &Option<String>,
                    custom: &Option<BTreeMap<String, f64>>|
         -> anyhow::Result<Option<Vec<Scheduled>>> {
            if let Some(p) = preset.as_deref().filter(|p| !p.is_empty()) {
                return self.scheduled(p).map(Some);
            }
            match custom {
                Some(c) if !c.is_empty() => {
                    let list: Vec<(String, f64)> = c.iter().map(|(k, v)| (k.clone(), *v)).collect();
                    Ok(Some(
                        self.schedule(&list, &format!("source {} custom_metrics", src.name)),
                    ))
                }
                _ => Ok(None),
            }
        };
        let primary = match pick(&src.preset_metrics, &src.custom_metrics)? {
            Some(v) => v,
            None => self.scheduled(default_preset)?,
        };
        let standby = pick(&src.preset_metrics_standby, &src.custom_metrics_standby)?.unwrap_or_default();
        Ok((primary, standby))
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

#[cfg(test)]
mod tests {
    use super::*;

    const MF: &str = r#"
metrics:
  versioned:
    sqls:
      11: select 11
      14: select 14
      "16": "  select 16  "
      18: "; -- not on 18"
    statement_timeout_seconds: 300
  stored:
    sqls: {11: select 1}
    storage_name: other_name
    some_future_key: x
  pilot:
    sqls: {11: select 1}
  pilot_config:
    sqls: {12: select 1}
presets:
  full:
    metrics: {versioned: 60, stored: 30, missing: 10}
  pg_index_pilot:
    metrics: {pilot: 30, pilot_config: 7200}
"#;

    fn mf() -> MetricsFile {
        parse(MF).unwrap()
    }

    #[test]
    fn sql_by_server_version() {
        let m = mf();
        let d = &m.metrics["versioned"];
        assert_eq!(d.sql_for(10), None);
        assert_eq!(d.sql_for(11), Some("select 11"));
        assert_eq!(d.sql_for(13), Some("select 11"));
        assert_eq!(d.sql_for(15), Some("select 14"));
        assert_eq!(d.sql_for(17), Some("select 16"), "string keys count, SQL trimmed");
        assert_eq!(d.sql_for(18), None, "';' means not on this version");
        assert_eq!(m.metrics["pilot_config"].sql_for(11), None);
    }

    #[test]
    fn names_and_storage_name() {
        assert_eq!(prom_name("db_size", "size_b"), "pgwatch_db_size_size_b");
        assert_eq!(prom_name("wal", "wal"), "pgwatch_wal");
        let m = mf();
        assert_eq!(m.metrics["stored"].storage_name("stored"), "other_name");
        assert_eq!(m.metrics["versioned"].storage_name("versioned"), "versioned");
        assert!(m.metrics["stored"].unknown.contains_key("some_future_key"));
    }

    #[test]
    fn presets_and_source_metric_lists() {
        let m = mf();
        let names = |v: &[Scheduled]| {
            v.iter()
                .map(|s| format!("{}:{}", s.name, s.interval_s))
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&m.scheduled("full").unwrap()), vec!["stored:30", "versioned:60"]);
        assert!(m.scheduled("nope").is_err());
        let src = |y: &str| -> crate::config::Source { serde_yaml::from_str(y).unwrap() };
        // neither: the global preset
        let (p, s) = m.for_source(&src("name: a"), "full").unwrap();
        assert_eq!(names(&p), vec!["stored:30", "versioned:60"]);
        assert!(s.is_empty());
        // preset wins over custom_metrics (pgwatch)
        let (p, _) = m
            .for_source(
                &src("{name: a, preset_metrics: pg_index_pilot, custom_metrics: {stored: 5}}"),
                "full",
            )
            .unwrap();
        assert_eq!(names(&p), vec!["pilot:30", "pilot_config:7200"]);
        // custom_metrics: float intervals, 0 = off, unknown metrics left out
        let (p, _) = m
            .for_source(
                &src("{name: a, custom_metrics: {stored: 14.6, versioned: 0, nope: 5}}"),
                "full",
            )
            .unwrap();
        assert_eq!(names(&p), vec!["stored:15"]);
        // standby lists
        let (_, s) = m
            .for_source(&src("{name: a, preset_metrics_standby: pg_index_pilot}"), "full")
            .unwrap();
        assert_eq!(names(&s), vec!["pilot:30", "pilot_config:7200"]);
        let (_, s) = m
            .for_source(&src("{name: a, custom_metrics_standby: {stored: 10}}"), "full")
            .unwrap();
        assert_eq!(names(&s), vec!["stored:10"]);
        assert!(m.for_source(&src("{name: a, preset_metrics: nope}"), "full").is_err());
    }

    #[test]
    fn the_shipped_metrics_yml_parses() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../config/pgwatch-prometheus/metrics.yml");
        let m = load(&p).unwrap();
        let full = m.scheduled("full").unwrap();
        assert!(full.len() >= 50, "{}", full.len());
        assert!(full.iter().any(|s| s.name == "wait_events" && s.interval_s == 15));
        let pilot = m.scheduled("pg_index_pilot").unwrap();
        assert_eq!(pilot.len(), 2);
        for (name, d) in &m.metrics {
            assert!(
                d.unknown.is_empty(),
                "{name}: unknown keys {:?}",
                d.unknown.keys().collect::<Vec<_>>()
            );
            assert!(!d.sqls.is_empty(), "{name}: no sqls");
        }
    }
}
