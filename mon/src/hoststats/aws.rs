//! The AWS calls RDS host stats makes, without an AWS SDK: SigV4 signing (ring), the default
//! credential chain (environment, web identity, shared credentials file, container endpoint,
//! IMDSv2), optional sts:AssumeRole with an External ID, and the four read-only APIs:
//! rds:DescribeDBInstances (Query/XML), cloudwatch:GetMetricData (JSON 1.0),
//! pi:GetResourceMetrics and logs:GetLogEvents (JSON 1.1).
//!
//! Calls take and return the AWS SDK's shapes as JSON (timestamps as ISO-8601 strings), the
//! shapes rds-host-stats recorded in its fixtures, so the poll logic is tested by replaying those
//! recordings (`AwsApi`), and the wire layer separately against a local fake endpoint.
use ring::{digest, hmac};
use serde_json::{json, Value};
use std::future::Future;
use std::time::Duration;

/// An AWS error: `name` is the service's error code (AccessDenied, ThrottlingException, ...).
#[derive(Debug, Clone, PartialEq)]
pub struct AwsError {
    pub name: String,
    pub message: String,
}
impl std::fmt::Display for AwsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.name, self.message)
    }
}
impl std::error::Error for AwsError {}
pub fn err(name: &str, message: impl Into<String>) -> AwsError {
    AwsError {
        name: name.into(),
        message: message.into(),
    }
}

/// One AWS call by the SDK's client and command names (as in the recorded fixtures):
/// ("rds", "DescribeDBInstancesCommand"), ("cloudwatch", "GetMetricDataCommand"),
/// ("pi", "GetResourceMetricsCommand"), ("logs", "GetLogEventsCommand").
pub trait AwsApi: Send + Sync {
    fn call(
        &self,
        client: &'static str,
        command: &'static str,
        input: Value,
    ) -> impl Future<Output = Result<Value, AwsError>> + Send;
}

// ------------------------------------------------------------------------------------------
// Time
// ------------------------------------------------------------------------------------------

fn civil(ms: i64) -> (i64, i64, i64, i64, i64, i64, i64) {
    let day = ms.div_euclid(86_400_000);
    let (y, m, d) = crate::store::chrono_free::civil_from_days(day);
    let rem = ms.rem_euclid(86_400_000);
    (y, m, d, rem / 3_600_000, rem / 60_000 % 60, rem / 1000 % 60, rem % 1000)
}

/// `2026-10-01T22:01:00.000Z` (JavaScript's Date#toISOString).
pub fn iso_ms(ms: i64) -> String {
    let (y, mo, d, h, mi, s, f) = civil(ms);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{f:03}Z")
}

/// `20150830T123600Z` (SigV4 X-Amz-Date).
pub fn amz_date(ms: i64) -> String {
    let (y, mo, d, h, mi, s, _) = civil(ms);
    format!("{y:04}{mo:02}{d:02}T{h:02}{mi:02}{s:02}Z")
}

/// ISO-8601 date-time to Unix ms: `YYYY-MM-DD` (UTC midnight) or `YYYY-MM-DDTHH:MM[:SS[.fff]]`
/// with `Z` or `±HH:MM` (no zone: UTC). None for anything else (JavaScript's Date.parse accepts
/// more, but AWS only sends these).
pub fn parse_iso(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    let num = |r: std::ops::Range<usize>| -> Option<i64> {
        let t = s.get(r)?;
        if t.is_empty() || !t.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        t.parse().ok()
    };
    if b.len() < 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    let day = crate::store::chrono_free::days_from_civil(y, mo, d);
    if b.len() == 10 {
        return Some(day * 86_400_000);
    }
    if b[10] != b'T' && b[10] != b't' && b[10] != b' ' {
        return None;
    }
    let (h, mi) = (num(11..13)?, num(14..16)?);
    if b.get(13) != Some(&b':') || h > 24 || mi > 59 {
        return None;
    }
    let mut i = 16;
    let mut sec = 0;
    let mut frac_ms = 0;
    if b.get(i) == Some(&b':') {
        sec = num(i + 1..i + 3)?;
        if sec > 60 {
            return None;
        }
        i += 3;
        if b.get(i) == Some(&b'.') {
            let st = i + 1;
            let mut e = st;
            while e < b.len() && b[e].is_ascii_digit() {
                e += 1;
            }
            if e == st {
                return None;
            }
            let digits = &s[st..e.min(st + 3)];
            frac_ms = digits.parse::<i64>().ok()? * 10i64.pow(3 - digits.len() as u32);
            i = e;
        }
    }
    let off_min = match &s[i..] {
        "" | "Z" | "z" => 0,
        z if (z.starts_with('+') || z.starts_with('-')) && z.len() == 6 && z.as_bytes()[3] == b':' => {
            let oh: i64 = z[1..3].parse().ok()?;
            let om: i64 = z[4..6].parse().ok()?;
            let o = oh * 60 + om;
            if z.starts_with('-') {
                -o
            } else {
                o
            }
        }
        _ => return None,
    };
    Some(day * 86_400_000 + h * 3_600_000 + mi * 60_000 + sec * 1000 + frac_ms - off_min * 60_000)
}

// ------------------------------------------------------------------------------------------
// SigV4
// ------------------------------------------------------------------------------------------

#[derive(Clone, PartialEq)]
pub struct Creds {
    pub access_key: String,
    pub secret_key: String,
    pub token: Option<String>,
    /// Unix ms; None = does not expire.
    pub expires_ms: Option<i64>,
}
impl std::fmt::Debug for Creds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Creds({}, expires {:?})", self.access_key, self.expires_ms)
    }
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn sha256_hex(b: &[u8]) -> String {
    hex(digest::digest(&digest::SHA256, b).as_ref())
}
fn hmac_sha256(key: &[u8], msg: &[u8]) -> Vec<u8> {
    hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, key), msg)
        .as_ref()
        .to_vec()
}

/// RFC 3986 percent-encoding as SigV4 wants it (unreserved characters kept).
pub fn uri_encode(s: &str, keep_slash: bool) -> String {
    let mut o = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') || (keep_slash && b == b'/') {
            o.push(b as char);
        } else {
            o.push_str(&format!("%{b:02X}"));
        }
    }
    o
}

/// Form body `k=v&...` (Query protocol), encoded as SigV4's canonical query encodes.
pub fn form(params: &[(&str, String)]) -> String {
    params
        .iter()
        .map(|(k, v)| format!("{}={}", uri_encode(k, false), uri_encode(v, false)))
        .collect::<Vec<_>>()
        .join("&")
}

/// Sign a request: returns the headers to add (x-amz-date, x-amz-security-token, authorization).
/// `headers` are the other headers to sign (lowercase names), host included.
#[allow(clippy::too_many_arguments)]
pub fn sign(
    method: &str,
    path: &str,
    query: &str,
    headers: &[(String, String)],
    body: &[u8],
    creds: &Creds,
    region: &str,
    service: &str,
    now_ms: i64,
) -> Vec<(String, String)> {
    let date_time = amz_date(now_ms);
    let date = &date_time[..8];
    let mut h: Vec<(String, String)> = headers
        .iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    h.push(("x-amz-date".into(), date_time.clone()));
    if let Some(t) = &creds.token {
        h.push(("x-amz-security-token".into(), t.clone()));
    }
    h.sort();
    let mut q: Vec<(String, String)> = query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (k.to_string(), v.to_string())
        })
        .collect();
    q.sort();
    let canonical_query = q.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&");
    let signed: String = h.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>().join(";");
    let canonical_headers: String = h.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let canonical = format!(
        "{method}\n{}\n{canonical_query}\n{canonical_headers}\n{signed}\n{}",
        uri_encode(path, true),
        sha256_hex(body)
    );
    let scope = format!("{date}/{region}/{service}/aws4_request");
    let to_sign = format!(
        "AWS4-HMAC-SHA256\n{date_time}\n{scope}\n{}",
        sha256_hex(canonical.as_bytes())
    );
    let k = hmac_sha256(format!("AWS4{}", creds.secret_key).as_bytes(), date.as_bytes());
    let k = hmac_sha256(&k, region.as_bytes());
    let k = hmac_sha256(&k, service.as_bytes());
    let k = hmac_sha256(&k, b"aws4_request");
    let sig = hex(&hmac_sha256(&k, to_sign.as_bytes()));
    let mut out = vec![("x-amz-date".to_string(), date_time)];
    if let Some(t) = &creds.token {
        out.push(("x-amz-security-token".into(), t.clone()));
    }
    out.push((
        "authorization".into(),
        format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed}, Signature={sig}",
            creds.access_key
        ),
    ));
    out
}

// ------------------------------------------------------------------------------------------
// Minimal XML (AWS Query responses): elements and text, attributes ignored.
// ------------------------------------------------------------------------------------------

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Xml {
    pub name: String,
    pub text: String,
    pub children: Vec<Xml>,
}

impl Xml {
    pub fn child(&self, name: &str) -> Option<&Xml> {
        self.children.iter().find(|c| c.name == name)
    }
    pub fn path(&self, p: &[&str]) -> Option<&Xml> {
        p.iter().try_fold(self, |x, n| x.child(n))
    }
    pub fn text_of(&self, name: &str) -> Option<&str> {
        self.child(name).map(|c| c.text.as_str())
    }
}

fn unescape(s: &str) -> Result<String, String> {
    let mut o = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        o.push_str(&rest[..i]);
        let j = rest[i..].find(';').ok_or("unterminated entity")? + i;
        let ent = &rest[i + 1..j];
        match ent {
            "amp" => o.push('&'),
            "lt" => o.push('<'),
            "gt" => o.push('>'),
            "quot" => o.push('"'),
            "apos" => o.push('\''),
            _ => {
                let code = if let Some(h) = ent.strip_prefix("#x").or_else(|| ent.strip_prefix("#X")) {
                    u32::from_str_radix(h, 16).map_err(|_| format!("bad entity &{ent};"))?
                } else if let Some(d) = ent.strip_prefix('#') {
                    d.parse().map_err(|_| format!("bad entity &{ent};"))?
                } else {
                    return Err(format!("unknown entity &{ent};"));
                };
                o.push(char::from_u32(code).ok_or("bad character reference")?);
            }
        }
        rest = &rest[j + 1..];
    }
    o.push_str(rest);
    Ok(o)
}

/// Parse a document into its root element.
pub fn parse_xml(s: &str) -> Result<Xml, String> {
    let mut stack: Vec<Xml> = vec![Xml::default()];
    let mut i = 0;
    let b = s.as_bytes();
    while i < b.len() {
        if b[i] == b'<' {
            if s[i..].starts_with("<?") {
                i = s[i..].find("?>").ok_or("unterminated declaration")? + i + 2;
            } else if s[i..].starts_with("<!--") {
                i = s[i..].find("-->").ok_or("unterminated comment")? + i + 3;
            } else if s[i..].starts_with("<![CDATA[") {
                let e = s[i..].find("]]>").ok_or("unterminated CDATA")? + i;
                stack.last_mut().unwrap().text.push_str(&s[i + 9..e]);
                i = e + 3;
            } else if s[i..].starts_with("<!") {
                i = s[i..].find('>').ok_or("unterminated declaration")? + i + 1;
            } else if s[i..].starts_with("</") {
                let e = s[i..].find('>').ok_or("unterminated end tag")? + i;
                let name = s[i + 2..e].trim();
                let el = stack.pop().ok_or("unbalanced")?;
                if el.name != name || stack.is_empty() {
                    return Err(format!("mismatched </{name}>"));
                }
                stack.last_mut().unwrap().children.push(el);
                i = e + 1;
            } else {
                let e = s[i..].find('>').ok_or("unterminated tag")? + i;
                let inner = &s[i + 1..e];
                let self_closing = inner.ends_with('/');
                let inner = inner.trim_end_matches('/');
                let name = inner
                    .split(|c: char| c.is_whitespace())
                    .next()
                    .unwrap_or("")
                    .to_string();
                if name.is_empty() {
                    return Err("empty tag name".into());
                }
                let el = Xml {
                    name,
                    ..Default::default()
                };
                if self_closing {
                    stack.last_mut().unwrap().children.push(el);
                } else {
                    stack.push(el);
                }
                i = e + 1;
            }
        } else {
            let e = s[i..].find('<').map_or(b.len(), |x| x + i);
            stack.last_mut().unwrap().text.push_str(&unescape(&s[i..e])?);
            i = e;
        }
    }
    if stack.len() != 1 {
        return Err("unclosed element".into());
    }
    let doc = stack.pop().unwrap();
    doc.children.into_iter().next().ok_or_else(|| "no root element".into())
}

// ------------------------------------------------------------------------------------------
// HTTP client: endpoints, credentials, calls
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Role {
    pub arn: String,
    pub external_id: String,
}

/// Where the client talks to. All overrides exist for tests (a local fake AWS).
#[derive(Clone, Debug)]
pub struct Endpoints {
    pub region: String,
    /// One endpoint for every service (AWS_ENDPOINT_URL), e.g. http://127.0.0.1:1234.
    pub endpoint_url: Option<String>,
    /// IMDS base URL (AWS_EC2_METADATA_SERVICE_ENDPOINT), default http://169.254.169.254.
    pub imds: String,
    pub imds_disabled: bool,
}

impl Endpoints {
    pub fn from_env(region: &str, env: &dyn Fn(&str) -> Option<String>) -> Self {
        Self {
            region: region.into(),
            endpoint_url: env("AWS_ENDPOINT_URL").filter(|v| !v.is_empty()),
            imds: env("AWS_EC2_METADATA_SERVICE_ENDPOINT")
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| "http://169.254.169.254".into()),
            imds_disabled: env("AWS_EC2_METADATA_DISABLED").is_some_and(|v| v.eq_ignore_ascii_case("true")),
        }
    }
    pub fn url(&self, service: &str) -> String {
        if let Some(u) = &self.endpoint_url {
            return u.trim_end_matches('/').to_string();
        }
        let suffix = if self.region.starts_with("cn-") {
            "amazonaws.com.cn"
        } else {
            "amazonaws.com"
        };
        format!("https://{service}.{}.{suffix}", self.region)
    }
}

/// The client: credentials (cached until 5 minutes before they expire) and signed calls.
pub struct Client {
    pub http: reqwest::Client,
    pub ep: Endpoints,
    pub role: Option<Role>,
    env: Box<dyn Fn(&str) -> Option<String> + Send + Sync>,
    base: tokio::sync::Mutex<Option<Creds>>,
    assumed: tokio::sync::Mutex<Option<Creds>>,
}

/// Connect within 5 s, whole request (body included) within 10 s, as rds-host-stats.
pub fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(concat!("pgai-mon/", env!("CARGO_PKG_VERSION"), " rds-host-stats"))
        .build()
        .expect("http client")
}

const REFRESH_MS: i64 = 300_000;

fn fresh(c: &Option<Creds>, now: i64) -> Option<Creds> {
    c.as_ref()
        .filter(|c| c.expires_ms.is_none_or(|e| e - REFRESH_MS > now))
        .cloned()
}

fn net(e: reqwest::Error) -> AwsError {
    let name = if e.is_timeout() { "TimeoutError" } else { "NetworkError" };
    // without the URL: it may carry a query
    err(name, e.without_url().to_string())
}

impl Client {
    pub fn new(ep: Endpoints, role: Option<Role>, env: Box<dyn Fn(&str) -> Option<String> + Send + Sync>) -> Self {
        Self {
            http: http_client(),
            ep,
            role,
            env,
            base: tokio::sync::Mutex::new(None),
            assumed: tokio::sync::Mutex::new(None),
        }
    }

    /// The credentials calls are signed with: the role's, when one is configured.
    pub async fn creds(&self) -> Result<Creds, AwsError> {
        let now = crate::model::now_ms();
        let Some(role) = &self.role else {
            return self.base_creds(now).await;
        };
        let mut a = self.assumed.lock().await;
        if let Some(c) = fresh(&a, now) {
            return Ok(c);
        }
        let base = self.base_creds(now).await?;
        let body = form(&[
            ("Action", "AssumeRole".into()),
            ("Version", "2011-06-15".into()),
            ("RoleArn", role.arn.clone()),
            ("RoleSessionName", "postgresai-rds-host-stats".into()),
            ("ExternalId", role.external_id.clone()),
            ("DurationSeconds", "3600".into()),
        ]);
        let x = self.query_with("sts", &body, &base).await?;
        let c = x
            .path(&["AssumeRoleResult", "Credentials"])
            .ok_or_else(|| err("InvalidResponse", "AssumeRole: no Credentials"))?;
        let creds = Creds {
            access_key: c.text_of("AccessKeyId").unwrap_or_default().to_string(),
            secret_key: c.text_of("SecretAccessKey").unwrap_or_default().to_string(),
            token: c.text_of("SessionToken").map(String::from),
            expires_ms: c.text_of("Expiration").and_then(parse_iso),
        };
        if creds.access_key.is_empty() || creds.secret_key.is_empty() {
            return Err(err("InvalidResponse", "AssumeRole: empty credentials"));
        }
        *a = Some(creds.clone());
        Ok(creds)
    }

    /// The default credential chain.
    async fn base_creds(&self, now: i64) -> Result<Creds, AwsError> {
        let mut b = self.base.lock().await;
        if let Some(c) = fresh(&b, now) {
            return Ok(c);
        }
        let c = self.resolve_chain().await?;
        *b = Some(c.clone());
        Ok(c)
    }

    async fn resolve_chain(&self) -> Result<Creds, AwsError> {
        let env = |k: &str| (self.env)(k).filter(|v| !v.is_empty());
        // 1. environment
        if let (Some(a), Some(s)) = (env("AWS_ACCESS_KEY_ID"), env("AWS_SECRET_ACCESS_KEY")) {
            return Ok(Creds {
                access_key: a,
                secret_key: s,
                token: env("AWS_SESSION_TOKEN"),
                expires_ms: None,
            });
        }
        // 2. web identity (EKS IRSA and the like)
        if let (Some(f), Some(arn)) = (env("AWS_WEB_IDENTITY_TOKEN_FILE"), env("AWS_ROLE_ARN")) {
            let tok =
                std::fs::read_to_string(&f).map_err(|e| err("CredentialsError", format!("web identity token: {e}")))?;
            let body = form(&[
                ("Action", "AssumeRoleWithWebIdentity".into()),
                ("Version", "2011-06-15".into()),
                ("RoleArn", arn),
                (
                    "RoleSessionName",
                    env("AWS_ROLE_SESSION_NAME").unwrap_or_else(|| "pgai-mon".into()),
                ),
                ("WebIdentityToken", tok.trim().to_string()),
            ]);
            let x = self.post_unsigned("sts", &body).await?;
            let c = x
                .path(&["AssumeRoleWithWebIdentityResult", "Credentials"])
                .ok_or_else(|| err("InvalidResponse", "AssumeRoleWithWebIdentity: no Credentials"))?;
            return Ok(Creds {
                access_key: c.text_of("AccessKeyId").unwrap_or_default().into(),
                secret_key: c.text_of("SecretAccessKey").unwrap_or_default().into(),
                token: c.text_of("SessionToken").map(String::from),
                expires_ms: c.text_of("Expiration").and_then(parse_iso),
            });
        }
        // 3. shared credentials file (static keys of the profile)
        let file = env("AWS_SHARED_CREDENTIALS_FILE").or_else(|| env("HOME").map(|h| format!("{h}/.aws/credentials")));
        if let Some(f) = file {
            if let Ok(text) = std::fs::read_to_string(&f) {
                let profile = env("AWS_PROFILE").unwrap_or_else(|| "default".into());
                if let Some(c) = profile_creds(&text, &profile) {
                    return Ok(c);
                }
            }
        }
        // 4. container credentials (ECS)
        let container = env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            .map(|r| format!("http://169.254.170.2{r}"))
            .or_else(|| env("AWS_CONTAINER_CREDENTIALS_FULL_URI"));
        if let Some(url) = container {
            let mut req = self.http.get(&url);
            let token = match env("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE") {
                Some(f) => std::fs::read_to_string(f).ok().map(|s| s.trim().to_string()),
                None => env("AWS_CONTAINER_AUTHORIZATION_TOKEN"),
            };
            if let Some(t) = token {
                req = req.header("authorization", t);
            }
            let r = req.send().await.map_err(net)?;
            if !r.status().is_success() {
                return Err(err(
                    "CredentialsError",
                    format!("container credentials: HTTP {}", r.status()),
                ));
            }
            return json_creds(&json_body(r).await?);
        }
        // 5. IMDSv2 (the instance profile; a container needs a hop limit of 2)
        if !self.ep.imds_disabled {
            return self.imds_creds().await;
        }
        Err(err("CredentialsError", "no AWS credentials found"))
    }

    async fn imds_creds(&self) -> Result<Creds, AwsError> {
        let base = self.ep.imds.trim_end_matches('/');
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(1))
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(net)?;
        let tok = http
            .put(format!("{base}/latest/api/token"))
            .header("x-aws-ec2-metadata-token-ttl-seconds", "21600")
            .send()
            .await
            .map_err(|e| err("CredentialsError", format!("IMDSv2 token: {}", net(e).message)))?;
        if !tok.status().is_success() {
            return Err(err("CredentialsError", format!("IMDSv2 token: HTTP {}", tok.status())));
        }
        let tok = tok.text().await.map_err(net)?;
        let get = |path: String| {
            http.get(format!("{base}/latest/meta-data/iam/security-credentials/{path}"))
                .header("x-aws-ec2-metadata-token", tok.clone())
                .send()
        };
        let r = get(String::new()).await.map_err(net)?;
        if !r.status().is_success() {
            return Err(err(
                "CredentialsError",
                format!("IMDSv2: no instance profile (HTTP {})", r.status()),
            ));
        }
        let role = r.text().await.map_err(net)?;
        let role = role.lines().next().unwrap_or("").trim().to_string();
        if role.is_empty() {
            return Err(err("CredentialsError", "IMDSv2: no instance profile"));
        }
        let r = get(role).await.map_err(net)?;
        if !r.status().is_success() {
            return Err(err(
                "CredentialsError",
                format!("IMDSv2 credentials: HTTP {}", r.status()),
            ));
        }
        json_creds(&json_body(r).await?)
    }

    async fn post_unsigned(&self, service: &str, body: &str) -> Result<Xml, AwsError> {
        let r = self
            .http
            .post(format!("{}/", self.ep.url(service)))
            .header("content-type", "application/x-www-form-urlencoded; charset=utf-8")
            .body(body.to_string())
            .send()
            .await
            .map_err(net)?;
        let status = r.status();
        let text = r.text().await.map_err(net)?;
        let x = parse_xml(&text).map_err(|e| err("InvalidResponse", format!("{service}: {e}")))?;
        if !status.is_success() {
            return Err(query_error(&x, status));
        }
        Ok(x)
    }

    async fn send_signed(
        &self,
        service: &str,
        headers: Vec<(String, String)>,
        body: String,
        creds: &Creds,
    ) -> Result<(reqwest::StatusCode, String), AwsError> {
        let url = format!("{}/", self.ep.url(service));
        let parsed = reqwest::Url::parse(&url).map_err(|e| err("ConfigError", e.to_string()))?;
        let host = match parsed.port() {
            Some(p) => format!("{}:{p}", parsed.host_str().unwrap_or("")),
            None => parsed.host_str().unwrap_or("").to_string(),
        };
        let mut sh = headers.clone();
        sh.push(("host".into(), host));
        let extra = sign(
            "POST",
            "/",
            "",
            &sh,
            body.as_bytes(),
            creds,
            &self.ep.region,
            service,
            crate::model::now_ms(),
        );
        let mut req = self.http.post(url).body(body);
        for (k, v) in headers.into_iter().chain(extra) {
            req = req.header(k, v);
        }
        let r = req.send().await.map_err(net)?;
        let st = r.status();
        Ok((st, r.text().await.map_err(net)?))
    }

    async fn query_with(&self, service: &str, body: &str, creds: &Creds) -> Result<Xml, AwsError> {
        let h = vec![(
            "content-type".to_string(),
            "application/x-www-form-urlencoded; charset=utf-8".to_string(),
        )];
        let (st, text) = retry(|| self.send_signed(service, h.clone(), body.to_string(), creds)).await?;
        let x = parse_xml(&text).map_err(|e| err("InvalidResponse", format!("{service}: {e}")))?;
        if !st.is_success() {
            return Err(query_error(&x, st));
        }
        Ok(x)
    }

    async fn json_call(&self, service: &str, target: &str, version: &str, body: Value) -> Result<Value, AwsError> {
        let creds = self.creds().await?;
        let h = vec![
            ("content-type".to_string(), format!("application/x-amz-json-{version}")),
            ("x-amz-target".to_string(), target.to_string()),
        ];
        let text = body.to_string();
        let (st, resp) = retry(|| self.send_signed(service, h.clone(), text.clone(), &creds)).await?;
        let v: Value = serde_json::from_str(if resp.trim().is_empty() { "{}" } else { &resp })
            .map_err(|e| err("InvalidResponse", format!("{service}: {e}")))?;
        if !st.is_success() {
            let ty = v["__type"].as_str().or(v["code"].as_str()).unwrap_or("UnknownError");
            let name = ty.rsplit('#').next().unwrap_or(ty).split(':').next().unwrap_or(ty);
            let msg = v["message"]
                .as_str()
                .or(v["Message"].as_str())
                .unwrap_or("")
                .to_string();
            return Err(err(name, if msg.is_empty() { format!("HTTP {st}") } else { msg }));
        }
        Ok(v)
    }
}

/// Up to 3 attempts for throttling, 5xx and network errors (the SDK's standard retry mode).
async fn retry<F, Fut>(f: F) -> Result<(reqwest::StatusCode, String), AwsError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<(reqwest::StatusCode, String), AwsError>>,
{
    let mut attempt = 0;
    loop {
        attempt += 1;
        let r = f().await;
        let retryable = match &r {
            Ok((st, body)) => {
                st.is_server_error()
                    || *st == 429
                    || (st.as_u16() == 400 && (body.contains("Throttl") || body.contains("RequestLimitExceeded")))
            }
            Err(e) => e.name == "NetworkError",
        };
        if !retryable || attempt >= 3 {
            return r;
        }
        tokio::time::sleep(Duration::from_millis(100 << attempt)).await;
    }
}

fn query_error(x: &Xml, st: reqwest::StatusCode) -> AwsError {
    let e = x.child("Error").or_else(|| x.path(&["Errors", "Error"]));
    match e {
        Some(e) => err(
            e.text_of("Code").unwrap_or("UnknownError"),
            e.text_of("Message").unwrap_or("").to_string(),
        ),
        None => err("UnknownError", format!("HTTP {st}")),
    }
}

async fn json_body(r: reqwest::Response) -> Result<Value, AwsError> {
    let t = r.text().await.map_err(net)?;
    serde_json::from_str(&t).map_err(|e| err("CredentialsError", format!("credentials endpoint: {e}")))
}

fn json_creds(v: &Value) -> Result<Creds, AwsError> {
    let s = |k: &str| v[k].as_str().map(String::from);
    match (s("AccessKeyId"), s("SecretAccessKey")) {
        (Some(a), Some(sk)) => Ok(Creds {
            access_key: a,
            secret_key: sk,
            token: s("Token").or(s("SessionToken")),
            expires_ms: s("Expiration").as_deref().and_then(parse_iso),
        }),
        _ => Err(err("CredentialsError", "credentials endpoint returned no keys")),
    }
}

/// Static keys of `[profile]` in an AWS shared credentials file.
pub fn profile_creds(text: &str, profile: &str) -> Option<Creds> {
    let mut cur = String::new();
    let (mut a, mut s, mut t) = (None, None, None);
    for line in text.lines() {
        let l = line.trim();
        if l.starts_with('#') || l.starts_with(';') || l.is_empty() {
            continue;
        }
        if l.starts_with('[') && l.ends_with(']') {
            cur = l[1..l.len() - 1].trim().to_string();
            continue;
        }
        if cur != profile {
            continue;
        }
        if let Some((k, v)) = l.split_once('=') {
            let v = v.trim().to_string();
            match k.trim() {
                "aws_access_key_id" => a = Some(v),
                "aws_secret_access_key" => s = Some(v),
                "aws_session_token" => t = Some(v),
                _ => {}
            }
        }
    }
    Some(Creds {
        access_key: a?,
        secret_key: s?,
        token: t,
        expires_ms: None,
    })
}

// ------------------------------------------------------------------------------------------
// The four APIs, SDK shapes in and out
// ------------------------------------------------------------------------------------------

fn epoch_s(v: &Value) -> Value {
    match v.as_str().and_then(parse_iso) {
        Some(ms) => json!(ms as f64 / 1000.0),
        None => v.clone(),
    }
}
fn iso_of_epoch_s(v: &Value) -> Value {
    match v.as_f64() {
        Some(s) => json!(iso_ms((s * 1000.0).round() as i64)),
        None => v.clone(),
    }
}

/// DescribeDBInstances XML -> the SDK's DBInstances (the fields the poller reads, typed).
pub fn db_instances_from_xml(x: &Xml) -> Value {
    let list = x.path(&["DescribeDBInstancesResult", "DBInstances"]);
    let inst: Vec<Value> = list
        .map(|l| l.children.iter().filter(|c| c.name == "DBInstance").collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .map(|d| {
            let mut m = serde_json::Map::new();
            for k in [
                "DBInstanceIdentifier",
                "Engine",
                "DBClusterIdentifier",
                "DbiResourceId",
                "DBInstanceClass",
                "StorageType",
            ] {
                if let Some(t) = d.text_of(k) {
                    m.insert(k.into(), json!(t));
                }
            }
            for k in ["AllocatedStorage", "Iops", "StorageThroughput", "MonitoringInterval"] {
                if let Some(n) = d.text_of(k).and_then(|t| t.trim().parse::<f64>().ok()) {
                    m.insert(k.into(), json!(n));
                }
            }
            if let Some(b) = d.text_of("PerformanceInsightsEnabled") {
                m.insert("PerformanceInsightsEnabled".into(), json!(b.trim() == "true"));
            }
            Value::Object(m)
        })
        .collect();
    json!({ "DBInstances": inst })
}

impl AwsApi for Client {
    fn call(
        &self,
        client: &'static str,
        command: &'static str,
        input: Value,
    ) -> impl Future<Output = Result<Value, AwsError>> + Send {
        async move {
            match (client, command) {
                ("rds", "DescribeDBInstancesCommand") => {
                    let id = input["DBInstanceIdentifier"].as_str().unwrap_or_default().to_string();
                    let body = form(&[
                        ("Action", "DescribeDBInstances".into()),
                        ("Version", "2014-10-31".into()),
                        ("DBInstanceIdentifier", id),
                    ]);
                    let creds = self.creds().await?;
                    let x = self.query_with("rds", &body, &creds).await?;
                    Ok(db_instances_from_xml(&x))
                }
                ("cloudwatch", "GetMetricDataCommand") => {
                    let mut wire = input.clone();
                    wire["StartTime"] = epoch_s(&input["StartTime"]);
                    wire["EndTime"] = epoch_s(&input["EndTime"]);
                    let mut out = self
                        .json_call("monitoring", "GraniteServiceVersion20100801.GetMetricData", "1.0", wire)
                        .await?;
                    if let Some(rs) = out["MetricDataResults"].as_array_mut() {
                        for r in rs {
                            if let Some(ts) = r["Timestamps"].as_array_mut() {
                                for t in ts.iter_mut() {
                                    *t = iso_of_epoch_s(t);
                                }
                            }
                        }
                    }
                    Ok(out)
                }
                ("pi", "GetResourceMetricsCommand") => {
                    let mut wire = input.clone();
                    wire["StartTime"] = epoch_s(&input["StartTime"]);
                    wire["EndTime"] = epoch_s(&input["EndTime"]);
                    let mut out = self
                        .json_call("pi", "PerformanceInsightsv20180227.GetResourceMetrics", "1.1", wire)
                        .await?;
                    for k in ["AlignedStartTime", "AlignedEndTime"] {
                        if !out[k].is_null() {
                            out[k] = iso_of_epoch_s(&out[k]);
                        }
                    }
                    if let Some(ms) = out["MetricList"].as_array_mut() {
                        for m in ms {
                            if let Some(dp) = m["DataPoints"].as_array_mut() {
                                for p in dp {
                                    p["Timestamp"] = iso_of_epoch_s(&p["Timestamp"]);
                                }
                            }
                        }
                    }
                    Ok(out)
                }
                ("logs", "GetLogEventsCommand") => {
                    self.json_call("logs", "Logs_20140328.GetLogEvents", "1.1", input).await
                }
                (c, m) => Err(err("NotImplemented", format!("{c} {m}"))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigv4_matches_the_aws_documentation_examples() {
        let creds = Creds {
            access_key: "AKIDEXAMPLE".into(),
            secret_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".into(),
            token: None,
            expires_ms: None,
        };
        let t = parse_iso("2015-08-30T12:36:00Z").unwrap();
        // IAM ListUsers (Signature Version 4 signing process example)
        let h = sign(
            "GET",
            "/",
            "Action=ListUsers&Version=2010-05-08",
            &[
                (
                    "Content-Type".into(),
                    "application/x-www-form-urlencoded; charset=utf-8".into(),
                ),
                ("Host".into(), "iam.amazonaws.com".into()),
            ],
            b"",
            &creds,
            "us-east-1",
            "iam",
            t,
        );
        let auth = &h.iter().find(|(k, _)| k == "authorization").unwrap().1;
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request, \
             SignedHeaders=content-type;host;x-amz-date, \
             Signature=5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7"
        );
        // aws-sig-v4-test-suite: get-vanilla
        let h = sign(
            "GET",
            "/",
            "",
            &[("host".into(), "example.amazonaws.com".into())],
            b"",
            &creds,
            "us-east-1",
            "service",
            t,
        );
        assert!(h.iter().any(|(k, v)| k == "authorization"
            && v.ends_with("Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31")));
        assert_eq!(h[0], ("x-amz-date".to_string(), "20150830T123600Z".to_string()));
    }

    #[test]
    fn iso_dates() {
        let t = parse_iso("2026-10-01T22:01:00.000Z").unwrap();
        assert_eq!(t, 1_790_892_060_000);
        assert_eq!(iso_ms(t), "2026-10-01T22:01:00.000Z");
        assert_eq!(parse_iso("2026-10-01T22:07:36Z"), Some(1_790_892_456_000));
        assert_eq!(parse_iso("2026-10-01T23:07:36+01:00"), Some(1_790_892_456_000));
        assert_eq!(parse_iso("2026-10-01T22:07:36.5Z"), Some(1_790_892_456_500));
        assert_eq!(parse_iso("2026-10-01"), Some(1_790_812_800_000));
        for bad in [
            "yesterday",
            "",
            "2026-13-01T00:00:00Z",
            "2026-10-01T22:07:36Q",
            "2026-10-01T22:07:36.Z",
        ] {
            assert_eq!(parse_iso(bad), None, "{bad}");
        }
        assert_eq!(amz_date(t), "20261001T220100Z");
    }

    #[test]
    fn xml_and_errors() {
        let x = parse_xml(
            "<?xml version=\"1.0\"?>\n<ErrorResponse xmlns=\"x\"><Error><Type>Sender</Type><Code>AccessDenied</Code>\
             <Message>a &amp; b &lt;c&gt; &#65;&#x42;</Message></Error><RequestId>r</RequestId></ErrorResponse>",
        )
        .unwrap();
        assert_eq!(
            query_error(&x, reqwest::StatusCode::FORBIDDEN),
            err("AccessDenied", "a & b <c> AB")
        );
        assert!(parse_xml("<a><b></a>").is_err());
        assert!(parse_xml("<a>").is_err());
        let x = parse_xml("<R><DescribeDBInstancesResult><DBInstances><DBInstance><Engine>aurora-postgresql</Engine>\
            <AllocatedStorage>1</AllocatedStorage><PerformanceInsightsEnabled>true</PerformanceInsightsEnabled><MonitoringInterval>15</MonitoringInterval>\
            <DbiResourceId>db-X</DbiResourceId><VpcSecurityGroups/></DBInstance></DBInstances></DescribeDBInstancesResult></R>").unwrap();
        assert_eq!(
            db_instances_from_xml(&x),
            json!({"DBInstances": [{"Engine": "aurora-postgresql", "AllocatedStorage": 1.0, "PerformanceInsightsEnabled": true,
                                    "MonitoringInterval": 15.0, "DbiResourceId": "db-X"}]})
        );
    }

    #[test]
    fn shared_credentials_file() {
        let t = "[default]\naws_access_key_id = A\naws_secret_access_key=S\n\n[other]\naws_access_key_id=B\naws_secret_access_key=T\naws_session_token=X\n";
        assert_eq!(profile_creds(t, "default").unwrap().access_key, "A");
        let o = profile_creds(t, "other").unwrap();
        assert_eq!((o.access_key.as_str(), o.token.as_deref()), ("B", Some("X")));
        assert!(profile_creds(t, "none").is_none());
    }
}
