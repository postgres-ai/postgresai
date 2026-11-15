//! Scraper for Prometheus text endpoints, driven by Prometheus scrape-config files (the
//! `host-metrics/*.yml` the CLI writes for Supabase and ClickHouse Cloud) and by inline jobs.
//!
//! Supported, as those files use them: `job_name`, `scheme`, `metrics_path`, `params`,
//! `scrape_interval`, `scrape_timeout`, `sample_limit`, `honor_labels`, `honor_timestamps`,
//! `static_configs` (targets, labels), `basic_auth` (username, password / password_file),
//! `bearer_token(_file)`, `authorization` (type, credentials / credentials_file), and
//! `relabel_configs` / `metric_relabel_configs` with actions keep, drop, replace, labeldrop,
//! labelkeep. Anything else in a file is an error for that file (it is skipped and reported),
//! never silently ignored.
//!
//! Per target, as Prometheus: labels `job` and `instance` (the target address) plus the static
//! labels (those starting with `__` are kept off the series, e.g. the CLI's `__pgai_rev`);
//! exposition labels that collide with them become `exported_<name>` unless honor_labels; the
//! synthetic series `up`, `scrape_duration_seconds`, `scrape_samples_scraped`,
//! `scrape_samples_post_metric_relabeling`. A series missing from a scrape, every series of a
//! failed scrape, and every series of a removed target get a staleness marker.
//!
//! Files are re-read every `reload_interval_s` (no SIGHUP needed); secrets files at every scrape.
use crate::ingest::{parse_text, ExtSample, Ingest};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct ScrapeCfg {
    /// Scrape-config files: paths, `*` allowed in the file name (e.g. /etc/pgai/host-metrics/*.yml).
    /// Each file is a YAML list of scrape configs.
    pub files: Vec<String>,
    /// Scrape configs given inline.
    pub jobs: Vec<ScrapeJob>,
    /// Seconds between re-reads of `files`.
    pub reload_interval_s: u64,
    /// Largest accepted response body (bytes); larger is a failed scrape.
    pub max_body_bytes: usize,
    /// Default sample_limit for jobs that set none (0 = none).
    pub default_sample_limit: usize,
}
impl Default for ScrapeCfg {
    fn default() -> Self {
        Self {
            files: vec![],
            jobs: vec![],
            reload_interval_s: 10,
            max_body_bytes: 32 << 20,
            default_sample_limit: 50_000,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScrapeJob {
    pub job_name: String,
    #[serde(default = "d_scheme")]
    pub scheme: String,
    #[serde(default = "d_path")]
    pub metrics_path: String,
    #[serde(default)]
    pub params: BTreeMap<String, Vec<String>>,
    #[serde(default = "d_interval")]
    pub scrape_interval: String,
    #[serde(default)]
    pub scrape_timeout: Option<String>,
    #[serde(default)]
    pub sample_limit: Option<usize>,
    #[serde(default)]
    pub honor_labels: bool,
    #[serde(default = "d_true")]
    pub honor_timestamps: bool,
    #[serde(default)]
    pub static_configs: Vec<StaticConfig>,
    #[serde(default)]
    pub basic_auth: Option<BasicAuth>,
    #[serde(default)]
    pub bearer_token: Option<String>,
    #[serde(default)]
    pub bearer_token_file: Option<PathBuf>,
    #[serde(default)]
    pub authorization: Option<Authorization>,
    #[serde(default)]
    pub relabel_configs: Vec<Relabel>,
    #[serde(default)]
    pub metric_relabel_configs: Vec<Relabel>,
}
fn d_scheme() -> String {
    "http".into()
}
fn d_path() -> String {
    "/metrics".into()
}
fn d_interval() -> String {
    "60s".into()
}
fn d_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StaticConfig {
    pub targets: Vec<String>,
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BasicAuth {
    pub username: String,
    #[serde(default)]
    pub password: Option<String>,
    #[serde(default)]
    pub password_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Authorization {
    #[serde(default = "d_bearer", rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub credentials: Option<String>,
    #[serde(default)]
    pub credentials_file: Option<PathBuf>,
}
fn d_bearer() -> String {
    "Bearer".into()
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Relabel {
    #[serde(default)]
    pub source_labels: Vec<String>,
    #[serde(default = "d_sep")]
    pub separator: String,
    #[serde(default = "d_regex")]
    pub regex: String,
    #[serde(default)]
    pub target_label: Option<String>,
    #[serde(default = "d_repl")]
    pub replacement: String,
    #[serde(default = "d_action")]
    pub action: String,
}
fn d_sep() -> String {
    ";".into()
}
fn d_regex() -> String {
    "(.*)".into()
}
fn d_repl() -> String {
    "$1".into()
}
fn d_action() -> String {
    "replace".into()
}

/// Prometheus duration: `1h30m`, `60s`, `500ms`, `1d`.
pub fn parse_duration(s: &str) -> anyhow::Result<Duration> {
    let s = s.trim();
    anyhow::ensure!(!s.is_empty(), "empty duration");
    let mut total_ms: u64 = 0;
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let n0 = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        anyhow::ensure!(i > n0, "bad duration {s:?}");
        let n: u64 = s[n0..i].parse()?;
        let u0 = i;
        while i < b.len() && b[i].is_ascii_alphabetic() {
            i += 1;
        }
        let ms = match &s[u0..i] {
            "ms" => 1,
            "s" => 1000,
            "m" => 60_000,
            "h" => 3_600_000,
            "d" => 86_400_000,
            "w" => 7 * 86_400_000,
            "y" => 365 * 86_400_000,
            u => anyhow::bail!("bad duration unit {u:?} in {s:?}"),
        };
        total_ms += n * ms;
    }
    Ok(Duration::from_millis(total_ms))
}

/// A compiled relabel rule.
#[derive(Debug, Clone)]
pub struct Rule {
    source: Vec<String>,
    sep: String,
    re: Regex,
    target: Option<String>,
    repl: String,
    action: Action,
}
#[derive(Debug, Clone, Copy, PartialEq)]
enum Action {
    Keep,
    Drop,
    Replace,
    LabelDrop,
    LabelKeep,
}

impl Rule {
    pub fn compile(r: &Relabel) -> anyhow::Result<Self> {
        let action = match r.action.to_ascii_lowercase().as_str() {
            "keep" => Action::Keep,
            "drop" => Action::Drop,
            "replace" => Action::Replace,
            "labeldrop" => Action::LabelDrop,
            "labelkeep" => Action::LabelKeep,
            a => anyhow::bail!("relabel action {a:?} is not supported (keep, drop, replace, labeldrop, labelkeep)"),
        };
        if action == Action::Replace {
            anyhow::ensure!(r.target_label.is_some(), "relabel replace needs target_label");
        }
        // Prometheus anchors relabel regexes
        let re = Regex::new(&format!("^(?:{})$", r.regex))?;
        Ok(Self {
            source: r.source_labels.clone(),
            sep: r.separator.clone(),
            re,
            target: r.target_label.clone(),
            repl: r.replacement.clone(),
            action,
        })
    }

    /// Apply to a label set (including `__name__`); false = the series is dropped.
    fn apply(&self, l: &mut BTreeMap<String, String>) -> bool {
        let val = || {
            self.source
                .iter()
                .map(|k| l.get(k).map(String::as_str).unwrap_or(""))
                .collect::<Vec<_>>()
                .join(&self.sep)
        };
        match self.action {
            Action::Keep => self.re.is_match(&val()),
            Action::Drop => !self.re.is_match(&val()),
            Action::Replace => {
                let v = val();
                if let Some(c) = self.re.captures(&v) {
                    let mut out = String::new();
                    c.expand(&self.repl, &mut out);
                    let t = self.target.clone().unwrap();
                    if out.is_empty() {
                        l.remove(&t);
                    } else {
                        l.insert(t, out);
                    }
                }
                true
            }
            Action::LabelDrop => {
                l.retain(|k, _| k == "__name__" || !self.re.is_match(k));
                true
            }
            Action::LabelKeep => {
                l.retain(|k, _| k == "__name__" || self.re.is_match(k));
                true
            }
        }
    }
}

fn relabel(rules: &[Rule], l: &mut BTreeMap<String, String>) -> bool {
    rules.iter().all(|r| r.apply(l))
}

/// One scrape target, resolved from a job and a static target.
#[derive(Debug, Clone)]
pub struct Target {
    pub job: String,
    pub url: String,
    pub interval: Duration,
    pub timeout: Duration,
    pub sample_limit: usize,
    pub honor_labels: bool,
    pub honor_timestamps: bool,
    /// Labels stored on every series (job, instance, static labels without `__`).
    pub labels: BTreeMap<String, String>,
    /// All target labels including `__` ones (shown in /scrape/targets).
    pub all_labels: BTreeMap<String, String>,
    pub auth: TargetAuth,
    pub metric_relabel: Vec<Rule>,
    /// Where the target came from (file path or "inline").
    pub source: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TargetAuth {
    None,
    Basic {
        username: String,
        password: Option<String>,
        password_file: Option<PathBuf>,
    },
    Header {
        kind: String,
        credentials: Option<String>,
        file: Option<PathBuf>,
    },
}

impl Target {
    /// Identity: two targets with the same key are the same staleness group.
    pub fn key(&self) -> String {
        format!("scrape\u{0}{}\u{0}{}", self.job, self.url)
    }
    /// What must restart the target's task when it changes.
    fn fingerprint(&self) -> String {
        // all_labels too: a new `__pgai_rev` must show in /scrape/targets (the CLI matches it)
        let rules: Vec<_> = self
            .metric_relabel
            .iter()
            .map(|r| (r.re.as_str(), &r.source, &r.sep, &r.repl, &r.target, r.action as u8))
            .collect();
        format!(
            "{:?}",
            (
                &self.url,
                self.interval,
                self.timeout,
                self.sample_limit,
                self.honor_labels,
                self.honor_timestamps,
                &self.all_labels,
                &self.auth,
                rules
            )
        )
    }
}

/// Resolve a job into its targets (relabel_configs applied to the target labels).
pub fn targets_of(job: &ScrapeJob, default_limit: usize, source: &str) -> anyhow::Result<Vec<Target>> {
    anyhow::ensure!(!job.job_name.is_empty(), "job_name is empty");
    anyhow::ensure!(
        job.scheme == "http" || job.scheme == "https",
        "scheme must be http or https"
    );
    anyhow::ensure!(job.metrics_path.starts_with('/'), "metrics_path must start with '/'");
    let interval = parse_duration(&job.scrape_interval)?;
    anyhow::ensure!(interval >= Duration::from_secs(1), "scrape_interval under 1s");
    let timeout = match &job.scrape_timeout {
        Some(t) => parse_duration(t)?,
        None => interval.min(Duration::from_secs(10)),
    };
    anyhow::ensure!(timeout <= interval, "scrape_timeout is longer than scrape_interval");
    let auth = match (
        &job.basic_auth,
        &job.bearer_token,
        &job.bearer_token_file,
        &job.authorization,
    ) {
        (None, None, None, None) => TargetAuth::None,
        (Some(b), None, None, None) => {
            anyhow::ensure!(
                b.password.is_some() != b.password_file.is_some(),
                "basic_auth needs exactly one of password and password_file"
            );
            TargetAuth::Basic {
                username: b.username.clone(),
                password: b.password.clone(),
                password_file: b.password_file.clone(),
            }
        }
        (None, t, f, None) if t.is_some() != f.is_some() => TargetAuth::Header {
            kind: "Bearer".into(),
            credentials: t.clone(),
            file: f.clone(),
        },
        (None, None, None, Some(a)) => {
            anyhow::ensure!(
                a.credentials.is_some() != a.credentials_file.is_some(),
                "authorization needs exactly one of credentials and credentials_file"
            );
            anyhow::ensure!(
                !a.kind.eq_ignore_ascii_case("basic"),
                "authorization type Basic: use basic_auth"
            );
            TargetAuth::Header {
                kind: a.kind.clone(),
                credentials: a.credentials.clone(),
                file: a.credentials_file.clone(),
            }
        }
        _ => anyhow::bail!("at most one of basic_auth, bearer_token(_file) and authorization"),
    };
    let target_rules = job
        .relabel_configs
        .iter()
        .map(Rule::compile)
        .collect::<anyhow::Result<Vec<_>>>()?;
    let metric_relabel = job
        .metric_relabel_configs
        .iter()
        .map(Rule::compile)
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut query = String::new();
    for (k, vs) in &job.params {
        for v in vs {
            query.push(if query.is_empty() { '?' } else { '&' });
            query.push_str(&url_escape(k));
            query.push('=');
            query.push_str(&url_escape(v));
        }
    }
    let mut out = Vec::new();
    for sc in &job.static_configs {
        for t in &sc.targets {
            anyhow::ensure!(
                !t.is_empty() && !t.contains('/') && !t.contains('?') && !t.contains('#') && !t.contains('@'),
                "target {t:?} must be host[:port]"
            );
            let mut l: BTreeMap<String, String> = BTreeMap::new();
            l.insert("job".into(), job.job_name.clone());
            l.insert("__address__".into(), t.clone());
            l.insert("__scheme__".into(), job.scheme.clone());
            l.insert("__metrics_path__".into(), job.metrics_path.clone());
            for (k, v) in &sc.labels {
                l.insert(k.clone(), v.clone());
            }
            if !relabel(&target_rules, &mut l) {
                continue;
            }
            let addr = l.get("__address__").cloned().unwrap_or_default();
            let scheme = l.get("__scheme__").cloned().unwrap_or_else(|| job.scheme.clone());
            let path = l
                .get("__metrics_path__")
                .cloned()
                .unwrap_or_else(|| job.metrics_path.clone());
            l.entry("instance".into()).or_insert_with(|| addr.clone());
            let labels: BTreeMap<String, String> = l
                .iter()
                .filter(|(k, v)| !k.starts_with("__") && !v.is_empty())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let all_labels = l
                .iter()
                .filter(|(k, _)| !matches!(k.as_str(), "__address__" | "__scheme__" | "__metrics_path__"))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            out.push(Target {
                job: job.job_name.clone(),
                url: format!("{scheme}://{addr}{path}{query}"),
                interval,
                timeout,
                sample_limit: job.sample_limit.unwrap_or(default_limit),
                honor_labels: job.honor_labels,
                honor_timestamps: job.honor_timestamps,
                labels,
                all_labels,
                auth: auth.clone(),
                metric_relabel: metric_relabel.clone(),
                source: source.to_string(),
            });
        }
    }
    Ok(out)
}

fn url_escape(s: &str) -> String {
    let mut o = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            o.push(b as char);
        } else {
            o.push_str(&format!("%{b:02X}"));
        }
    }
    o
}

/// Files matching a pattern: `*` is allowed in the file name only.
pub fn glob(pattern: &str) -> Vec<PathBuf> {
    let p = Path::new(pattern);
    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if !name.contains('*') {
        return if p.is_file() { vec![p.to_path_buf()] } else { vec![] };
    }
    let dir = p.parent().unwrap_or(Path::new("."));
    let re = Regex::new(&format!("^{}$", regex::escape(name).replace("\\*", ".*"))).unwrap();
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_file() && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| re.is_match(n)))
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// Load every target of the config: inline jobs, then each file (a file that fails to parse
/// or validate is skipped whole and reported). Duplicate (job, url) targets: the first wins.
pub fn load_targets(cfg: &ScrapeCfg) -> (Vec<Target>, Vec<String>) {
    let mut targets = Vec::new();
    let mut errors = Vec::new();
    for j in &cfg.jobs {
        match targets_of(j, cfg.default_sample_limit, "inline") {
            Ok(t) => targets.extend(t),
            Err(e) => errors.push(format!("inline job {:?}: {e}", j.job_name)),
        }
    }
    for pat in &cfg.files {
        for f in glob(pat) {
            let src = f.display().to_string();
            let r = std::fs::read_to_string(&f)
                .map_err(anyhow::Error::from)
                .and_then(|s| Ok(serde_yaml::from_str::<Option<Vec<ScrapeJob>>>(&s)?.unwrap_or_default()))
                .and_then(|jobs| {
                    let mut v = Vec::new();
                    for j in &jobs {
                        v.extend(
                            targets_of(j, cfg.default_sample_limit, &src)
                                .map_err(|e| anyhow::anyhow!("job {:?}: {e}", j.job_name))?,
                        );
                    }
                    Ok(v)
                });
            match r {
                Ok(t) => targets.extend(t),
                Err(e) => errors.push(format!("{src}: {e}")),
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    targets.retain(|t| seen.insert(t.key()));
    (targets, errors)
}

/// Result of one scrape.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ScrapeStatus {
    pub health: String,
    pub last_scrape_ms: i64,
    pub last_duration_ms: u64,
    pub samples: usize,
    pub last_error: Option<String>,
}

fn read_secret(p: &Path) -> anyhow::Result<String> {
    Ok(std::fs::read_to_string(p)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", p.display()))?
        .trim()
        .to_string())
}

/// The samples of one scrape body, after target labels and metric_relabel_configs.
pub fn samples_of(t: &Target, body: &str, ts: i64) -> Result<Vec<ExtSample>, String> {
    let lines = parse_text(body).map_err(|(n, e)| format!("line {n}: {e}"))?;
    let mut out = Vec::with_capacity(lines.len());
    for l in lines {
        let mut m: BTreeMap<String, String> = BTreeMap::new();
        for (k, v) in l.labels {
            if t.labels.contains_key(&k) && !t.honor_labels {
                m.insert(format!("exported_{k}"), v);
            } else {
                m.insert(k, v);
            }
        }
        for (k, v) in &t.labels {
            if t.honor_labels && m.contains_key(k) {
                continue;
            }
            m.insert(k.clone(), v.clone());
        }
        m.insert("__name__".into(), l.metric);
        if !relabel(&t.metric_relabel, &mut m) {
            continue;
        }
        let Some(name) = m.remove("__name__") else { continue };
        out.push(ExtSample {
            metric: name,
            labels: m.into_iter().collect(),
            ts: if t.honor_timestamps { l.ts.unwrap_or(ts) } else { ts },
            value: l.value,
        });
    }
    Ok(out)
}

/// Fetch a target's body (bounded size and time).
async fn fetch(client: &reqwest::Client, t: &Target, max_body: usize) -> anyhow::Result<String> {
    let mut req = client
        .get(&t.url)
        .timeout(t.timeout)
        .header("accept", "text/plain;version=0.0.4;q=1,*/*;q=0.1")
        .header(
            "x-prometheus-scrape-timeout-seconds",
            format!("{}", t.timeout.as_secs_f64()),
        );
    match &t.auth {
        TargetAuth::None => {}
        TargetAuth::Basic {
            username,
            password,
            password_file,
        } => {
            let pw = match (password, password_file) {
                (Some(p), _) => p.clone(),
                (None, Some(f)) => read_secret(f)?,
                (None, None) => String::new(),
            };
            req = req.basic_auth(username, Some(pw));
        }
        TargetAuth::Header {
            kind,
            credentials,
            file,
        } => {
            let c = match (credentials, file) {
                (Some(c), _) => c.clone(),
                (None, Some(f)) => read_secret(f)?,
                (None, None) => String::new(),
            };
            req = req.header("authorization", format!("{kind} {c}"));
        }
    }
    let mut resp = req.send().await?;
    anyhow::ensure!(resp.status().is_success(), "HTTP {}", resp.status());
    if let Some(n) = resp.content_length() {
        anyhow::ensure!(
            n as usize <= max_body,
            "body of {n} bytes is over the {max_body}-byte limit"
        );
    }
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await? {
        body.extend_from_slice(&chunk);
        anyhow::ensure!(body.len() <= max_body, "body is over the {max_body}-byte limit");
    }
    String::from_utf8(body).map_err(|_| anyhow::anyhow!("body is not UTF-8"))
}

/// Scrape a target once and write the result (samples or, on failure, `up` 0 and staleness
/// markers for everything the target had).
pub async fn scrape_once(client: &reqwest::Client, ing: &Ingest, t: &Target, max_body: usize, ts: i64) -> ScrapeStatus {
    let t0 = std::time::Instant::now();
    let res = fetch(client, t, max_body)
        .await
        .map_err(|e| e.to_string())
        .and_then(|body| {
            let n_scraped = parse_text(&body).map(|l| l.len()).unwrap_or(0);
            let s = samples_of(t, &body, ts)?;
            if t.sample_limit > 0 && s.len() > t.sample_limit {
                return Err(format!("{} samples over sample_limit {}", s.len(), t.sample_limit));
            }
            Ok((n_scraped, s))
        });
    let dur = t0.elapsed();
    let (up, mut samples, n_scraped, n_post, err) = match res {
        Ok((n, s)) => {
            let k = s.len();
            (1.0, s, n, k, None)
        }
        Err(e) => (0.0, vec![], 0, 0, Some(e)),
    };
    let lab: Vec<(String, String)> = t.labels.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    for (m, v) in [
        ("up", up),
        ("scrape_duration_seconds", dur.as_secs_f64()),
        ("scrape_samples_scraped", n_scraped as f64),
        ("scrape_samples_post_metric_relabeling", n_post as f64),
    ] {
        samples.push(ExtSample {
            metric: m.into(),
            labels: lab.clone(),
            ts,
            value: v,
        });
    }
    let n = samples.len() - 4;
    // a sample with bad names fails the scrape, as a parse error would
    let (samples, err) = match ing.validate(samples) {
        Ok(s) => (s, err),
        Err(e) => {
            let synth: Vec<ExtSample> = [("up", 0.0), ("scrape_duration_seconds", dur.as_secs_f64())]
                .into_iter()
                .map(|(m, v)| ExtSample {
                    metric: m.into(),
                    labels: lab.clone(),
                    ts,
                    value: v,
                })
                .collect();
            (ing.validate(synth).unwrap_or_default(), Some(e))
        }
    };
    let up = err.is_none();
    let werr = ing
        .write(samples, Some((&t.key(), ts)))
        .await
        .err()
        .map(|e| e.to_string());
    ScrapeStatus {
        health: if up && werr.is_none() { "up" } else { "down" }.into(),
        last_scrape_ms: ts,
        last_duration_ms: dur.as_millis() as u64,
        samples: if up { n } else { 0 },
        last_error: err.or(werr),
    }
}

struct Running {
    target: Target,
    fingerprint: String,
    task: tokio::task::JoinHandle<()>,
}

/// The scrape manager: one task per target, reconciled against the config files.
pub struct Scraper {
    cfg: ScrapeCfg,
    ing: Arc<Ingest>,
    client: reqwest::Client,
    running: Mutex<HashMap<String, Running>>,
    pub status: Arc<Mutex<HashMap<String, ScrapeStatus>>>,
    pub errors: Mutex<Vec<String>>,
}

impl Scraper {
    pub fn new(cfg: ScrapeCfg, ing: Arc<Ingest>) -> anyhow::Result<Arc<Self>> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .user_agent(concat!("pgai-mon/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Arc::new(Self {
            cfg,
            ing,
            client,
            running: Mutex::new(HashMap::new()),
            status: Arc::new(Mutex::new(HashMap::new())),
            errors: Mutex::new(vec![]),
        }))
    }

    pub fn enabled(cfg: &ScrapeCfg) -> bool {
        !cfg.files.is_empty() || !cfg.jobs.is_empty()
    }

    /// Re-read the config and start, restart or stop target tasks. Returns (started, stopped).
    pub async fn reconcile(self: &Arc<Self>) -> (usize, usize) {
        let (targets, mut errors) = load_targets(&self.cfg);
        // a file that fails to parse (perhaps caught half-written) keeps the targets of its
        // last good version running, as vmagent does
        let failed = |src: &str| errors.iter().position(|e| e.starts_with(&format!("{src}: ")));
        let mut kept: HashMap<usize, usize> = HashMap::new();
        let want: HashMap<String, Target> = targets.into_iter().map(|t| (t.key(), t)).collect();
        let mut stop = Vec::new();
        let mut start = Vec::new();
        {
            let mut run = self.running.lock().unwrap();
            let keys: Vec<String> = run.keys().cloned().collect();
            for k in keys {
                let keep = match want.get(&k) {
                    Some(t) => t.fingerprint() == run[&k].fingerprint,
                    None => failed(&run[&k].target.source)
                        .inspect(|i| *kept.entry(*i).or_default() += 1)
                        .is_some(),
                };
                if !keep {
                    let r = run.remove(&k).unwrap();
                    r.task.abort();
                    if !want.contains_key(&k) {
                        stop.push(k);
                    }
                }
            }
            for (k, t) in want {
                if let std::collections::hash_map::Entry::Vacant(e) = run.entry(k) {
                    start.push(e.key().clone());
                    let task = tokio::spawn(target_loop(self.clone(), t.clone()));
                    e.insert(Running {
                        fingerprint: t.fingerprint(),
                        target: t,
                        task,
                    });
                }
            }
        }
        for (i, n) in kept {
            errors[i].push_str(&format!("; its {n} running targets keep the last good version"));
        }
        {
            let mut e = self.errors.lock().unwrap();
            for x in &errors {
                if !e.contains(x) {
                    eprintln!("scrape: {x}");
                }
            }
            *e = errors;
        }
        // a removed target's series end now
        let now = crate::model::now_ms();
        for k in &stop {
            self.status.lock().unwrap().remove(k);
            if let Err(e) = self.ing.end_group(k, now).await {
                eprintln!("scrape: {e}");
            }
        }
        (start.len(), stop.len())
    }

    /// Reload loop (spawn once).
    pub async fn run(self: Arc<Self>) {
        let every = Duration::from_secs(self.cfg.reload_interval_s.max(1));
        loop {
            self.reconcile().await;
            tokio::time::sleep(every).await;
        }
    }

    /// Targets with their labels and last scrape (`GET /scrape/targets`).
    pub fn targets_json(&self) -> serde_json::Value {
        let run = self.running.lock().unwrap();
        let st = self.status.lock().unwrap();
        let mut v: Vec<serde_json::Value> = run
            .iter()
            .map(|(k, r)| {
                serde_json::json!({
                    "job": r.target.job, "url": redact_url(&r.target.url), "labels": r.target.all_labels,
                    "source": r.target.source, "interval_s": r.target.interval.as_secs_f64(),
                    "status": st.get(k),
                })
            })
            .collect();
        v.sort_by(|a, b| {
            a["job"]
                .as_str()
                .cmp(&b["job"].as_str())
                .then(a["url"].as_str().cmp(&b["url"].as_str()))
        });
        serde_json::json!({"targets": v, "errors": *self.errors.lock().unwrap()})
    }
}

fn redact_url(u: &str) -> String {
    // query parameters may carry secrets
    u.split('?').next().unwrap_or(u).to_string()
}

/// Offset of a target inside its interval, stable per target (spreads scrapes out).
fn phase(key: &str, interval: Duration) -> Duration {
    let h = xxhash_rust::xxh3::xxh3_64(key.as_bytes());
    Duration::from_millis(h % (interval.as_millis() as u64).max(1))
}

/// The log line after a scrape, given the failures in a row before it: the first failure,
/// every 100th and the recovery are logged, not every failed scrape.
fn failure_log(fails: &mut u64, err: Option<&str>) -> Option<String> {
    match err {
        Some(e) => {
            *fails += 1;
            match *fails {
                1 => Some(e.to_string()),
                n if n % 100 == 0 => Some(format!("{e} ({n} failures in a row)")),
                _ => None,
            }
        }
        None => match std::mem::take(fails) {
            0 => None,
            n => Some(format!("up again after {n} failed scrapes")),
        },
    }
}

async fn target_loop(s: Arc<Scraper>, t: Target) {
    let key = t.key();
    let iv = t.interval;
    let iv_ms = iv.as_millis() as i64;
    // first scrape at the next aligned slot plus this target's phase
    let ph = phase(&key, iv).as_millis() as i64;
    let now = crate::model::now_ms();
    let mut next = (now - ph).div_euclid(iv_ms) * iv_ms + ph + iv_ms;
    if next - now > iv_ms {
        next -= iv_ms;
    }
    let mut fails = 0;
    loop {
        let wait = (next - crate::model::now_ms()).max(0) as u64;
        tokio::time::sleep(Duration::from_millis(wait)).await;
        let ts = crate::model::now_ms();
        let st = scrape_once(&s.client, &s.ing, &t, s.cfg.max_body_bytes, ts).await;
        if let Some(l) = failure_log(&mut fails, st.last_error.as_deref()) {
            eprintln!("scrape {} {}: {l}", t.job, redact_url(&t.url));
        }
        s.status.lock().unwrap().insert(key.clone(), st);
        next += iv_ms;
        let now = crate::model::now_ms();
        if next <= now {
            // missed slots (a slow scrape or a pause) are skipped, not caught up
            next += ((now - next) / iv_ms + 1) * iv_ms;
        }
    }
}

/// `GET /scrape/targets` (mounted behind authentication).
pub fn router(s: Arc<Scraper>) -> axum::Router {
    axum::Router::new()
        .route(
            "/scrape/targets",
            axum::routing::get(|axum::extract::State(s): axum::extract::State<Arc<Scraper>>| async move {
                axum::Json(s.targets_json())
            }),
        )
        .with_state(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations() {
        assert_eq!(parse_duration("60s").unwrap(), Duration::from_secs(60));
        assert_eq!(parse_duration("1m30s").unwrap(), Duration::from_secs(90));
        assert_eq!(parse_duration("500ms").unwrap(), Duration::from_millis(500));
        assert_eq!(parse_duration("2h").unwrap(), Duration::from_secs(7200));
        assert!(parse_duration("1x").is_err());
        assert!(parse_duration("s").is_err());
        assert!(parse_duration("").is_err());
    }

    #[test]
    fn a_down_target_is_logged_on_change_and_every_100th_failure() {
        let mut fails = 0;
        let logged: Vec<Option<String>> = (0..200).map(|_| failure_log(&mut fails, Some("refused"))).collect();
        assert_eq!(logged[0].as_deref(), Some("refused"));
        assert_eq!(logged[99].as_deref(), Some("refused (100 failures in a row)"));
        assert_eq!(logged.iter().flatten().count(), 3, "1st, 100th, 200th");
        assert_eq!(
            failure_log(&mut fails, None).as_deref(),
            Some("up again after 200 failed scrapes")
        );
        assert_eq!(failure_log(&mut fails, None), None);
    }

    #[test]
    fn relabel_actions() {
        let r = |a: &str, src: &[&str], re: &str, tgt: Option<&str>, rep: &str| {
            Rule::compile(&Relabel {
                source_labels: src.iter().map(|s| s.to_string()).collect(),
                separator: ";".into(),
                regex: re.into(),
                target_label: tgt.map(String::from),
                replacement: rep.into(),
                action: a.into(),
            })
            .unwrap()
        };
        let mut l: BTreeMap<String, String> = [("__name__", "node_cpu_seconds_total"), ("mode", "idle"), ("cpu", "0")]
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        assert!(r(
            "keep",
            &["__name__"],
            "node_cpu_seconds_total|node_time_seconds",
            None,
            ""
        )
        .apply(&mut l.clone()));
        // anchored: a partial match is no match
        assert!(!r("keep", &["__name__"], "node_cpu", None, "").apply(&mut l.clone()));
        assert!(!r("drop", &["mode"], "idle", None, "").apply(&mut l.clone()));
        assert!(r("replace", &["mode", "cpu"], "(.*);(.*)", Some("x"), "$1-$2").apply(&mut l));
        assert_eq!(l["x"], "idle-0");
        assert!(r("labeldrop", &[], "c.*", None, "").apply(&mut l));
        assert!(!l.contains_key("cpu") && l.contains_key("__name__"));
        assert!(r("labelkeep", &[], "mode", None, "").apply(&mut l));
        assert_eq!(l.keys().cloned().collect::<Vec<_>>(), vec!["__name__", "mode"]);
        assert!(Rule::compile(&Relabel {
            source_labels: vec![],
            separator: ";".into(),
            regex: "x".into(),
            target_label: None,
            replacement: "".into(),
            action: "hashmod".into()
        })
        .is_err());
    }
}
