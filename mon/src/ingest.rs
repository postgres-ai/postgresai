//! Externally produced samples (name, labels, timestamp, value) into the store: the push endpoint
//! `POST /api/v1/import/prometheus` (Prometheus text exposition, as VictoriaMetrics accepts it, so
//! existing pushers such as rds-host-stats only change the URL), and the in-process producers
//! that are not SQL collectors (scraper, host rules, RDS host stats).
//!
//! Series are registered exactly as the collector registers its own (one SeriesDef per series per
//! day, before its first sample), so ingested series are indistinguishable from collected ones in
//! `series`, `samples` and the prom_* functions.
//!
//! Staleness: a producer that knows when a series ended passes a *group* (one scrape target, one
//! rule group); a series present in the group's previous write and absent from this one gets one
//! staleness marker (StaleNaN) at this write's timestamp, as the collector and Prometheus scraping
//! do. Pushed samples carry no group and no markers, as VictoriaMetrics' import has none: readers
//! fall back to their lookback window.
//!
//! Memory is bounded: at most `max_series` distinct series are tracked (a series unseen for an
//! hour is forgotten when the table is full; new series beyond the cap are dropped and counted),
//! request bodies are capped, and a request is parsed whole before anything is written.
use crate::model::{day_of, now_ms, series_id, stale_nan, Record, Sample, SeriesDef};
use crate::store::Shared;
use crate::Msg;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct IngestCfg {
    /// Largest accepted request body (bytes).
    pub max_body_bytes: usize,
    /// Most samples one request may carry.
    pub max_samples_per_request: usize,
    /// Distinct ingested series tracked at once (cardinality cap).
    pub max_series: usize,
    /// Samples older than this (seconds before now) are dropped.
    pub max_age_s: i64,
    /// Samples further than this in the future are dropped.
    pub max_future_s: i64,
    /// Labels per series at most.
    pub max_labels: usize,
    /// Longest label value (bytes).
    pub max_label_value_bytes: usize,
}
impl Default for IngestCfg {
    fn default() -> Self {
        Self {
            max_body_bytes: 16 << 20,
            max_samples_per_request: 500_000,
            max_series: 200_000,
            max_age_s: 86_400,
            max_future_s: 600,
            max_labels: 64,
            max_label_value_bytes: 4096,
        }
    }
}

/// One externally produced sample. `ts` is Unix milliseconds.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtSample {
    pub metric: String,
    /// Any order; empty values are dropped (a Prometheus label with an empty value is absent).
    pub labels: Vec<(String, String)>,
    pub ts: i64,
    pub value: f64,
}

#[derive(Default)]
pub struct IngestStats {
    pub requests: AtomicU64,
    pub samples: AtomicU64,
    pub stale_markers: AtomicU64,
    pub new_series: AtomicU64,
    pub rejected_requests: AtomicU64,
    pub dropped_too_old: AtomicU64,
    pub dropped_too_new: AtomicU64,
    pub dropped_series_limit: AtomicU64,
    pub dropped_invalid: AtomicU64,
}

impl IngestStats {
    pub fn json(&self) -> serde_json::Value {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        serde_json::json!({
            "requests": g(&self.requests), "samples": g(&self.samples),
            "stale_markers": g(&self.stale_markers), "new_series": g(&self.new_series),
            "rejected_requests": g(&self.rejected_requests),
            "dropped": {"too_old": g(&self.dropped_too_old), "too_new": g(&self.dropped_too_new),
                        "series_limit": g(&self.dropped_series_limit), "invalid": g(&self.dropped_invalid)},
        })
    }
}

#[derive(Default)]
struct Tracked {
    /// series id -> wall-clock ms it was last written (the cardinality table).
    series: HashMap<u64, i64>,
    /// Earliest time the next eviction scan may run (it is O(table)).
    next_evict_ms: i64,
    /// group -> series in its previous write (staleness markers).
    groups: HashMap<String, HashSet<u64>>,
}

/// What a write did.
#[derive(Debug, Default, Clone, PartialEq, Serialize)]
pub struct WriteReport {
    pub samples: usize,
    pub stale_markers: usize,
    pub dropped: usize,
}

pub struct Ingest {
    shared: Shared,
    tx: mpsc::Sender<Msg>,
    pub cfg: IngestCfg,
    state: Mutex<Tracked>,
    pub stats: IngestStats,
}

/// Forget a series unseen for this long when the table is full.
const IDLE_EVICT_MS: i64 = 3_600_000;

fn valid_metric(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty()
        && b.len() <= 256
        && (b[0].is_ascii_alphabetic() || b[0] == b'_' || b[0] == b':')
        && b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_' || *c == b':')
}

fn valid_label(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty()
        && b.len() <= 128
        && (b[0].is_ascii_alphabetic() || b[0] == b'_')
        && b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_')
}

impl Ingest {
    pub fn new(shared: Shared, tx: mpsc::Sender<Msg>, cfg: IngestCfg) -> Arc<Self> {
        Arc::new(Self {
            shared,
            tx,
            cfg,
            state: Mutex::new(Tracked::default()),
            stats: IngestStats::default(),
        })
    }

    /// Labels in store form: sorted, `__`-prefixed and empty ones dropped. Err on an invalid name,
    /// a duplicate, or a limit.
    pub fn normalize_labels(&self, mut labels: Vec<(String, String)>) -> Result<Vec<(String, String)>, String> {
        labels.retain(|(k, v)| !v.is_empty() && !k.starts_with("__"));
        labels.sort();
        for w in labels.windows(2) {
            if w[0].0 == w[1].0 {
                return Err(format!("duplicate label {:?}", w[0].0));
            }
        }
        if labels.len() > self.cfg.max_labels {
            return Err(format!("{} labels (at most {})", labels.len(), self.cfg.max_labels));
        }
        for (k, v) in &labels {
            if !valid_label(k) {
                return Err(format!("invalid label name {k:?}"));
            }
            if v.len() > self.cfg.max_label_value_bytes {
                return Err(format!(
                    "label {k:?} value longer than {} bytes",
                    self.cfg.max_label_value_bytes
                ));
            }
        }
        Ok(labels)
    }

    /// Validate a whole batch (names, labels): Err names the first invalid sample and nothing of
    /// the batch is written. Returns the samples in store form.
    pub fn validate(&self, samples: Vec<ExtSample>) -> Result<Vec<ExtSample>, String> {
        if samples.len() > self.cfg.max_samples_per_request {
            return Err(format!(
                "{} samples in one request (at most {})",
                samples.len(),
                self.cfg.max_samples_per_request
            ));
        }
        samples
            .into_iter()
            .map(|mut s| {
                if !valid_metric(&s.metric) {
                    return Err(format!("invalid metric name {:?}", s.metric));
                }
                s.labels = self
                    .normalize_labels(std::mem::take(&mut s.labels))
                    .map_err(|e| format!("{}: {e}", s.metric))?;
                Ok(s)
            })
            .collect()
    }

    /// Write already validated samples (see `validate`); durable when this returns. `group`:
    /// (key, ts) for staleness markers at ts of the group's series missing from this write.
    pub async fn write(&self, samples: Vec<ExtSample>, group: Option<(&str, i64)>) -> anyhow::Result<WriteReport> {
        let now = now_ms();
        let (min_ts, max_ts) = (now - self.cfg.max_age_s * 1000, now + self.cfg.max_future_s * 1000);
        let mut rep = WriteReport::default();
        let mut defs: Vec<SeriesDef> = Vec::new();
        let mut out: Vec<Sample> = Vec::with_capacity(samples.len());
        {
            let mut st = self.state.lock().unwrap();
            let mut cur: HashSet<u64> = HashSet::new();
            let mut known = self.shared.known_series.lock().unwrap();
            for s in samples {
                if s.ts < min_ts {
                    self.stats.dropped_too_old.fetch_add(1, Ordering::Relaxed);
                    rep.dropped += 1;
                    continue;
                }
                if s.ts > max_ts {
                    self.stats.dropped_too_new.fetch_add(1, Ordering::Relaxed);
                    rep.dropped += 1;
                    continue;
                }
                let id = series_id(&s.metric, &s.labels);
                if !st.series.contains_key(&id) && st.series.len() >= self.cfg.max_series {
                    if now >= st.next_evict_ms {
                        st.series.retain(|_, t| *t >= now - IDLE_EVICT_MS);
                        st.next_evict_ms = now + 60_000;
                    }
                    if st.series.len() >= self.cfg.max_series {
                        self.stats.dropped_series_limit.fetch_add(1, Ordering::Relaxed);
                        rep.dropped += 1;
                        continue;
                    }
                }
                st.series.insert(id, now);
                cur.insert(id);
                let day = day_of(s.ts);
                if known.len() > 100_000 && known.iter().any(|(d, _)| *d < day - 1) {
                    known.retain(|(d, _)| *d >= day - 1); // as the collector: older days are closed
                }
                if known.insert((day, id)) {
                    defs.push(SeriesDef {
                        series_id: id,
                        metric: s.metric,
                        labels: s.labels,
                        first_seen: s.ts,
                    });
                }
                out.push(Sample {
                    series_id: id,
                    ts: s.ts,
                    value: s.value,
                });
            }
            rep.samples = out.len();
            if let Some((key, ts)) = group {
                if let Some(prev) = st.groups.get(key) {
                    for id in prev.difference(&cur) {
                        // only series still tracked: an evicted one has nothing left to end
                        if st.series.contains_key(id) {
                            out.push(Sample {
                                series_id: *id,
                                ts,
                                value: stale_nan(),
                            });
                            rep.stale_markers += 1;
                        }
                    }
                }
                st.groups.insert(key.to_string(), cur);
            }
        }
        self.stats.new_series.fetch_add(defs.len() as u64, Ordering::Relaxed);
        self.stats.samples.fetch_add(rep.samples as u64, Ordering::Relaxed);
        self.stats
            .stale_markers
            .fetch_add(rep.stale_markers as u64, Ordering::Relaxed);
        // Series first, so the WAL never holds a sample whose series is unknown.
        if !defs.is_empty() {
            self.tx
                .send(Msg::Rec(Record::Series(defs)))
                .await
                .map_err(|_| anyhow::anyhow!("store is shut down"))?;
        }
        if !out.is_empty() {
            let (a, b) = tokio::sync::oneshot::channel();
            self.tx
                .send(Msg::RecAck(Record::Samples(out), a))
                .await
                .map_err(|_| anyhow::anyhow!("store is shut down"))?;
            b.await.map_err(|_| anyhow::anyhow!("store is shut down"))?;
        }
        Ok(rep)
    }

    /// End a group: every series of its last write gets a staleness marker at `ts` (a scrape
    /// target removed from the config, a failed scrape).
    pub async fn end_group(&self, key: &str, ts: i64) -> anyhow::Result<usize> {
        let ids: Vec<u64> = {
            let mut st = self.state.lock().unwrap();
            let Some(prev) = st.groups.remove(key) else {
                return Ok(0);
            };
            prev.into_iter().filter(|id| st.series.contains_key(id)).collect()
        };
        if ids.is_empty() {
            return Ok(0);
        }
        let n = ids.len();
        let out = ids
            .into_iter()
            .map(|id| Sample {
                series_id: id,
                ts,
                value: stale_nan(),
            })
            .collect();
        let (a, b) = tokio::sync::oneshot::channel();
        self.tx
            .send(Msg::RecAck(Record::Samples(out), a))
            .await
            .map_err(|_| anyhow::anyhow!("store is shut down"))?;
        b.await.map_err(|_| anyhow::anyhow!("store is shut down"))?;
        self.stats.stale_markers.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }

    /// Series tracked now (for tests and /health).
    pub fn tracked_series(&self) -> usize {
        self.state.lock().unwrap().series.len()
    }
}

// ------------------------------------------------------------------------------------------
// Prometheus text exposition
// ------------------------------------------------------------------------------------------

/// One parsed exposition line.
#[derive(Debug, Clone, PartialEq)]
pub struct PromLine {
    pub metric: String,
    pub labels: Vec<(String, String)>,
    pub value: f64,
    /// Milliseconds, if the line has a timestamp.
    pub ts: Option<i64>,
}

fn parse_value(s: &str) -> Option<f64> {
    match s {
        "+Inf" | "Inf" | "+inf" | "inf" => Some(f64::INFINITY),
        "-Inf" | "-inf" => Some(f64::NEG_INFINITY),
        "NaN" | "nan" => Some(f64::NAN),
        _ => {
            // Rust also parses "infinity" etc.; the exposition format does not have them
            if s.bytes().any(|c| c.is_ascii_alphabetic() && c != b'e' && c != b'E') {
                return None;
            }
            s.parse().ok()
        }
    }
}

/// Parse one non-comment line: `name{l="v",...} value [timestamp_ms]`.
pub fn parse_line(line: &str) -> Result<PromLine, String> {
    let b = line.as_bytes();
    let mut i = 0;
    while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b':') {
        i += 1;
    }
    let metric = &line[..i];
    if metric.is_empty() {
        return Err("no metric name".into());
    }
    let mut labels = Vec::new();
    while i < b.len() && (b[i] == b' ' || b[i] == b'\t') {
        i += 1;
    }
    if i < b.len() && b[i] == b'{' {
        i += 1;
        loop {
            while i < b.len() && (b[i] == b' ' || b[i] == b'\t' || b[i] == b',') {
                i += 1;
            }
            if i >= b.len() {
                return Err("unterminated label set".into());
            }
            if b[i] == b'}' {
                i += 1;
                break;
            }
            let k0 = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let key = &line[k0..i];
            if key.is_empty() {
                return Err(format!("bad label name at byte {k0}"));
            }
            while i < b.len() && (b[i] == b' ' || b[i] == b'\t') {
                i += 1;
            }
            if i >= b.len() || b[i] != b'=' {
                return Err(format!("expected '=' after label {key:?}"));
            }
            i += 1;
            while i < b.len() && (b[i] == b' ' || b[i] == b'\t') {
                i += 1;
            }
            if i >= b.len() || b[i] != b'"' {
                return Err(format!("expected '\"' for label {key:?}"));
            }
            i += 1;
            let mut val = Vec::new();
            loop {
                if i >= b.len() {
                    return Err(format!("unterminated value of label {key:?}"));
                }
                match b[i] {
                    b'"' => {
                        i += 1;
                        break;
                    }
                    b'\\' if i + 1 < b.len() => {
                        match b[i + 1] {
                            b'n' => val.push(b'\n'),
                            b'\\' => val.push(b'\\'),
                            b'"' => val.push(b'"'),
                            c => {
                                val.push(b'\\');
                                val.push(c);
                            }
                        }
                        i += 2;
                    }
                    c => {
                        val.push(c);
                        i += 1;
                    }
                }
            }
            let val = String::from_utf8(val).map_err(|_| format!("label {key:?} value is not UTF-8"))?;
            if key == "__name__" {
                return Err("__name__ inside the label set is not supported".into());
            }
            labels.push((key.to_string(), val));
        }
    }
    let rest = line[i..].trim();
    let mut parts = rest.split_ascii_whitespace();
    let v = parts.next().ok_or("no value")?;
    let value = parse_value(v).ok_or_else(|| format!("bad value {v:?}"))?;
    let ts = match parts.next() {
        None => None,
        Some(t) => Some(t.parse::<i64>().map_err(|_| format!("bad timestamp {t:?}"))?),
    };
    if parts.next().is_some() {
        return Err("trailing data after the timestamp".into());
    }
    Ok(PromLine {
        metric: metric.to_string(),
        labels,
        value,
        ts,
    })
}

/// Parse an exposition body: comments, HELP/TYPE and blank lines are skipped. Err: (line
/// number, reason) of the first bad line.
pub fn parse_text(body: &str) -> Result<Vec<PromLine>, (usize, String)> {
    parse_text_capped(body, usize::MAX)
}

/// `parse_text` that stops at the first sample past `max`, before the rest is held in memory.
pub fn parse_text_capped(body: &str, max: usize) -> Result<Vec<PromLine>, (usize, String)> {
    let mut out = Vec::new();
    for (n, raw) in body.lines().enumerate() {
        let l = raw.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if out.len() >= max {
            return Err((n + 1, format!("more than {max} samples in one request")));
        }
        out.push(parse_line(l).map_err(|e| (n + 1, e))?);
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------
// HTTP
// ------------------------------------------------------------------------------------------
use axum::extract::{RawQuery, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};

fn bad(code: StatusCode, msg: String) -> Response {
    (
        code,
        axum::Json(serde_json::json!({"error": {"kind": "invalid_input", "message": msg}})),
    )
        .into_response()
}

/// `extra_label=name=value` query arguments (VictoriaMetrics): added to every sample, replacing
/// a label of the same name.
fn extra_labels(q: Option<&str>) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for kv in q.unwrap_or("").split('&') {
        let Some(v) = kv.strip_prefix("extra_label=") else {
            continue;
        };
        let v = percent_decode(v)?;
        let (k, val) = v
            .split_once('=')
            .ok_or_else(|| format!("extra_label {v:?} is not name=value"))?;
        if !valid_label(k) {
            return Err(format!("invalid extra_label name {k:?}"));
        }
        out.push((k.to_string(), val.to_string()));
    }
    Ok(out)
}

fn percent_decode(s: &str) -> Result<String, String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                let h = std::str::from_utf8(&b[i + 1..i + 3]).map_err(|_| "bad escape".to_string())?;
                out.push(u8::from_str_radix(h, 16).map_err(|_| format!("bad escape %{h}"))?);
                i += 3;
            }
            b'%' => return Err("truncated escape".into()),
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|_| "not UTF-8".into())
}

/// `POST /api/v1/import/prometheus`: Prometheus text, timestamps in milliseconds (a line
/// without one gets the receive time). The whole body is parsed and validated first: a bad line
/// rejects the request (400) and nothing is written, so a pusher's retry never duplicates.
/// Samples outside the age window or over the series cap are dropped and counted. 204 on success.
async fn import_prometheus(
    State(ing): State<Arc<Ingest>>,
    RawQuery(q): RawQuery,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    ing.stats.requests.fetch_add(1, Ordering::Relaxed);
    let reject = |code, msg| {
        ing.stats.rejected_requests.fetch_add(1, Ordering::Relaxed);
        bad(code, msg)
    };
    if let Some(enc) = headers.get("content-encoding").and_then(|v| v.to_str().ok()) {
        if !enc.eq_ignore_ascii_case("identity") {
            return reject(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                format!("content-encoding {enc} is not supported"),
            );
        }
    }
    let extra = match extra_labels(q.as_deref()) {
        Ok(e) => e,
        Err(e) => return reject(StatusCode::BAD_REQUEST, e),
    };
    // checked before they are copied onto every sample
    if extra.len() > ing.cfg.max_labels || extra.iter().any(|(_, v)| v.len() > ing.cfg.max_label_value_bytes) {
        return reject(
            StatusCode::BAD_REQUEST,
            format!(
                "extra_label: at most {} labels of at most {} bytes",
                ing.cfg.max_labels, ing.cfg.max_label_value_bytes
            ),
        );
    }
    let Ok(text) = std::str::from_utf8(&body) else {
        return reject(StatusCode::BAD_REQUEST, "body is not UTF-8".into());
    };
    let lines = match parse_text_capped(text, ing.cfg.max_samples_per_request) {
        Ok(l) => l,
        Err((n, e)) => return reject(StatusCode::BAD_REQUEST, format!("line {n}: {e}")),
    };
    let now = now_ms();
    let samples: Vec<ExtSample> = lines
        .into_iter()
        .map(|l| {
            let mut labels = l.labels;
            if !extra.is_empty() {
                labels.retain(|(k, _)| !extra.iter().any(|(e, _)| e == k));
                labels.extend(extra.iter().cloned());
            }
            ExtSample {
                metric: l.metric,
                labels,
                ts: l.ts.unwrap_or(now),
                value: l.value,
            }
        })
        .collect();
    let samples = match ing.validate(samples) {
        Ok(s) => s,
        Err(e) => {
            ing.stats.dropped_invalid.fetch_add(1, Ordering::Relaxed);
            return reject(StatusCode::BAD_REQUEST, e);
        }
    };
    match ing.write(samples, None).await {
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => bad(StatusCode::SERVICE_UNAVAILABLE, e.to_string()),
    }
}

async fn ingest_stats(State(ing): State<Arc<Ingest>>) -> Response {
    let mut v = ing.stats.json();
    v["tracked_series"] = ing.tracked_series().into();
    axum::Json(v).into_response()
}

/// The ingest routes (mounted behind authentication by query::app).
pub fn router(ing: Arc<Ingest>) -> axum::Router {
    let limit = ing.cfg.max_body_bytes;
    axum::Router::new()
        .route("/api/v1/import/prometheus", axum::routing::post(import_prometheus))
        .layer(axum::extract::DefaultBodyLimit::max(limit))
        .route("/ingest/stats", axum::routing::get(ingest_stats))
        .with_state(ing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_exposition_format() {
        let l = parse_line(r#"host_cpu_utilization_percent{cluster="prod",node_name="n\"1\\x\n"} 12.5 1790000000000"#)
            .unwrap();
        assert_eq!(l.metric, "host_cpu_utilization_percent");
        assert_eq!(
            l.labels,
            vec![
                ("cluster".into(), "prod".into()),
                ("node_name".into(), "n\"1\\x\n".into())
            ]
        );
        assert_eq!(l.value, 12.5);
        assert_eq!(l.ts, Some(1_790_000_000_000));
        let l = parse_line("up 1").unwrap();
        assert_eq!((l.metric.as_str(), l.labels.len(), l.value, l.ts), ("up", 0, 1.0, None));
        let l = parse_line(r#"x{a="1",} -Inf"#).unwrap();
        assert_eq!(l.value, f64::NEG_INFINITY);
        assert!(parse_line("x NaN").unwrap().value.is_nan());
        assert_eq!(parse_line("x{} +Inf 5").unwrap().value, f64::INFINITY);
        assert_eq!(parse_line("x 1e3").unwrap().value, 1000.0);
        for bad in [
            "x",
            "{a=\"b\"} 1",
            "x{a=\"b} 1",
            "x{a=b} 1",
            "x 1 2 3",
            "x 1 1.5",
            "x infinity",
            "x{__name__=\"y\"} 1",
            "x abc",
        ] {
            assert!(parse_line(bad).is_err(), "{bad}");
        }
        let all = parse_text("# HELP x y\n# TYPE x gauge\n\nx 1\n  y{a=\"b\"} 2 3\n# EOF\n").unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(parse_text("x 1\ny{\n").unwrap_err().0, 2);
        assert_eq!(parse_text_capped("x 1\n# c\ny 2\n", 2).unwrap().len(), 2);
        assert_eq!(parse_text_capped("x 1\ny 2\nz 3\n", 2).unwrap_err().0, 3);
    }

    #[test]
    fn extra_label_arguments() {
        assert_eq!(
            extra_labels(Some("extra_label=job%3Drds&foo=1&extra_label=env=a+b")).unwrap(),
            vec![("job".into(), "rds".into()), ("env".into(), "a b".into())]
        );
        assert!(extra_labels(Some("extra_label=novalue")).is_err());
        assert!(extra_labels(Some("extra_label=1x=a")).is_err());
        assert!(extra_labels(Some("extra_label=a%2")).is_err());
    }
}
