//! Fetch windows captured while DuckDB binds evaluator calls on the query thread.
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadRange {
    pub start: i64,
    pub end: i64,
    /// High 16 bits of the metric's series IDs. None means unknown dependencies.
    #[serde(default)]
    pub prefixes: Option<Vec<u16>>,
}

thread_local! {
    static RANGE: RefCell<Option<ReadRange>> = const { RefCell::new(None) };
}

pub fn reset() {
    RANGE.with(|r| *r.borrow_mut() = None);
}

pub fn record(metrics: &[&str], start: i64, end: i64) {
    RANGE.with(|r| {
        let mut r = r.borrow_mut();
        let r = r.get_or_insert_with(|| ReadRange {
            start,
            end,
            prefixes: Some(Vec::new()),
        });
        r.start = r.start.min(start);
        r.end = r.end.max(end);
        if let Some(prefixes) = &mut r.prefixes {
            prefixes.extend(
                metrics
                    .iter()
                    .map(|m| (xxhash_rust::xxh3::xxh3_64(m.as_bytes()) >> 48) as u16),
            );
            prefixes.sort_unstable();
            prefixes.dedup();
        }
    });
}

pub fn take() -> Option<ReadRange> {
    RANGE.with(|r| r.borrow_mut().take())
}

/// Prove every relation is an evaluator call, a constant table, a verified catalog view,
/// or a CTE proven in this scope.
/// Unknown macros and SQL views keep generation-only caching.
pub fn bounded_sql(
    v: &serde_json::Value,
    macros: &std::collections::HashSet<String>,
    safe: &std::collections::HashSet<String>,
    ctes: &std::collections::HashSet<String>,
) -> bool {
    use serde_json::Value;
    match v {
        Value::Object(m) => {
            let mut scope = ctes.clone();
            if let Some(defs) = m.get("cte_map").and_then(|v| v["map"].as_array()) {
                for def in defs {
                    if !bounded_sql(&def["value"], macros, safe, &scope) {
                        return false;
                    }
                    scope.insert(def["key"].as_str().unwrap_or("").to_ascii_lowercase());
                }
            }
            match m.get("type").and_then(Value::as_str) {
                Some("BASE_TABLE") => {
                    if !m.get("schema_name").and_then(Value::as_str).unwrap_or("").is_empty()
                        || !m.get("catalog_name").and_then(Value::as_str).unwrap_or("").is_empty()
                        || !scope.contains(&m["table_name"].as_str().unwrap_or("").to_ascii_lowercase())
                    {
                        return false;
                    }
                }
                Some("TABLE_FUNCTION") => {
                    let name = m["function"]["function_name"]
                        .as_str()
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    if !safe.contains(&name)
                        && !matches!(
                            name.as_str(),
                            "prom_eval_rs"
                                | "prom_series_rs"
                                | "prom_topk_rs"
                                | "prom_agg_rs"
                                | "pgai_samples_rs"
                                | "pgai_series_rs"
                                | "range"
                                | "generate_series"
                                | "unnest"
                        )
                    {
                        return false;
                    }
                }
                _ => {}
            }
            if let Some(name) = m.get("function_name").and_then(Value::as_str) {
                let name = name.to_ascii_lowercase();
                if macros.contains(&name) && !safe.contains(&name) {
                    return false;
                }
            }
            m.iter()
                .filter(|(k, _)| k.as_str() != "cte_map")
                .all(|(_, v)| bounded_sql(v, macros, safe, &scope))
        }
        Value::Array(a) => a.iter().all(|v| bounded_sql(v, macros, safe, ctes)),
        _ => true,
    }
}
