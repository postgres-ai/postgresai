//! HTTP authentication. Every route needs credentials except the liveness probe (`/livez`):
//! a bearer token, HTTP basic auth, or either when both are configured. Credentials come from
//! files or the environment, never from the YAML itself.
//! A second, read-only token (`PGAI_MON_READ_TOKEN`) may read (any GET, and `POST /sql`) but not
//! write (push ingest): it is the one to hand to Grafana and other readers.
//! Running without authentication must be asked for explicitly (`http.auth.disabled: true` or
//! `PGAI_MON_AUTH_DISABLED=1`), for local development only; with no credentials and no opt-out
//! the server refuses to start.
use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// `http.auth` in the YAML. Environment variables override each field:
/// PGAI_MON_TOKEN, PGAI_MON_TOKEN_FILE, PGAI_MON_READ_TOKEN, PGAI_MON_READ_TOKEN_FILE, PGAI_MON_AUTH_USERNAME,
/// PGAI_MON_AUTH_PASSWORD, PGAI_MON_AUTH_PASSWORD_FILE, PGAI_MON_AUTH_DISABLED.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct AuthCfg {
    /// File holding the bearer token (surrounding whitespace is ignored).
    pub token_file: Option<PathBuf>,
    /// File holding the read-only bearer token.
    pub read_token_file: Option<PathBuf>,
    /// Basic auth user name.
    pub username: Option<String>,
    /// File holding the basic auth password (surrounding whitespace is ignored).
    pub password_file: Option<PathBuf>,
    /// Serve without authentication (local development only).
    pub disabled: bool,
}

/// Resolved credentials. Secrets are kept only as SHA-256 digests, compared in constant time.
pub struct Auth {
    token: Option<[u8; 32]>,
    read_token: Option<[u8; 32]>,
    basic: Option<([u8; 32], [u8; 32])>,
    disabled: bool,
}

/// What a request's credentials allow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Any GET, and `POST /sql` (itself read-only).
    Read,
    Full,
}

const MIN_SECRET_LEN: usize = 16;

fn digest(b: &[u8]) -> [u8; 32] {
    let d = ring::digest::digest(&ring::digest::SHA256, b);
    d.as_ref().try_into().unwrap()
}

/// Constant-time equality of two digests (no early exit; the inputs are fixed-length hashes,
/// so neither the secret's length nor its prefix leaks through timing).
fn ct_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut x = 0u8;
    for i in 0..32 {
        x |= a[i] ^ b[i];
    }
    std::hint::black_box(x) == 0
}

fn read_secret(p: &std::path::Path, what: &str) -> anyhow::Result<String> {
    let s = std::fs::read_to_string(p)
        .map_err(|e| anyhow::anyhow!("http.auth: cannot read {what} file {}: {e}", p.display()))?;
    Ok(s.trim().to_string())
}

fn truthy(v: &str) -> bool {
    matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

impl Auth {
    /// Resolve the config plus the environment (`env` is a lookup so tests need not mutate the
    /// process environment).
    pub fn resolve(cfg: &AuthCfg, env: &dyn Fn(&str) -> Option<String>) -> anyhow::Result<Self> {
        let env = |k: &str| env(k).filter(|v| !v.is_empty());
        let disabled = env("PGAI_MON_AUTH_DISABLED").map_or(cfg.disabled, |v| truthy(&v));
        let token = match (env("PGAI_MON_TOKEN"), env("PGAI_MON_TOKEN_FILE")) {
            (Some(t), _) => Some(t.trim().to_string()),
            (None, Some(f)) => Some(read_secret(std::path::Path::new(&f), "token")?),
            (None, None) => match &cfg.token_file {
                Some(f) => Some(read_secret(f, "token")?),
                None => None,
            },
        };
        let read_token = match (env("PGAI_MON_READ_TOKEN"), env("PGAI_MON_READ_TOKEN_FILE")) {
            (Some(t), _) => Some(t.trim().to_string()),
            (None, Some(f)) => Some(read_secret(std::path::Path::new(&f), "read-only token")?),
            (None, None) => match &cfg.read_token_file {
                Some(f) => Some(read_secret(f, "read-only token")?),
                None => None,
            },
        };
        let username = env("PGAI_MON_AUTH_USERNAME").or_else(|| cfg.username.clone());
        let password = match (env("PGAI_MON_AUTH_PASSWORD"), env("PGAI_MON_AUTH_PASSWORD_FILE")) {
            (Some(p), _) => Some(p),
            (None, Some(f)) => Some(read_secret(std::path::Path::new(&f), "password")?),
            (None, None) => match &cfg.password_file {
                Some(f) => Some(read_secret(f, "password")?),
                None => None,
            },
        };
        if let Some(t) = &token {
            anyhow::ensure!(
                t.len() >= MIN_SECRET_LEN,
                "http.auth: the bearer token must be at least {MIN_SECRET_LEN} characters"
            );
        }
        if let Some(t) = &read_token {
            anyhow::ensure!(
                t.len() >= MIN_SECRET_LEN,
                "http.auth: the read-only token must be at least {MIN_SECRET_LEN} characters"
            );
            anyhow::ensure!(
                token.as_ref() != Some(t),
                "http.auth: the read-only token must differ from the bearer token"
            );
        }
        let basic = match (username, password) {
            (Some(u), Some(p)) => {
                anyhow::ensure!(!u.contains(':'), "http.auth: the user name must not contain ':'");
                anyhow::ensure!(
                    p.len() >= MIN_SECRET_LEN,
                    "http.auth: the password must be at least {MIN_SECRET_LEN} characters"
                );
                Some((digest(u.as_bytes()), digest(p.as_bytes())))
            }
            (None, None) => None,
            _ => anyhow::bail!("http.auth: basic auth needs both a user name and a password"),
        };
        let token = token.map(|t| digest(t.as_bytes()));
        let read_token = read_token.map(|t| digest(t.as_bytes()));
        if !disabled && token.is_none() && basic.is_none() && read_token.is_none() {
            anyhow::bail!(
                "http.auth: no credentials. Set PGAI_MON_TOKEN(_FILE) or PGAI_MON_AUTH_USERNAME + \
                 PGAI_MON_AUTH_PASSWORD(_FILE) (or http.auth in the config); for local development only, \
                 PGAI_MON_AUTH_DISABLED=1 / http.auth.disabled: true serves without authentication"
            );
        }
        Ok(Self {
            token,
            read_token,
            basic,
            disabled,
        })
    }

    pub fn from_env(cfg: &AuthCfg) -> anyhow::Result<Self> {
        Self::resolve(cfg, &|k| std::env::var(k).ok())
    }

    /// No authentication (tests and explicit local development).
    pub fn disabled() -> Self {
        Self {
            token: None,
            read_token: None,
            basic: None,
            disabled: true,
        }
    }

    pub fn is_disabled(&self) -> bool {
        self.disabled
    }

    /// Does this Authorization header value carry valid credentials?
    pub fn check(&self, authz: Option<&[u8]>) -> bool {
        self.access(authz).is_some()
    }

    /// What this Authorization header value allows; None without valid credentials.
    pub fn access(&self, authz: Option<&[u8]>) -> Option<Access> {
        if self.disabled {
            return Some(Access::Full);
        }
        let h = std::str::from_utf8(authz?).ok()?;
        let (scheme, rest) = h.trim().split_once(' ').unwrap_or((h, ""));
        let rest = rest.trim();
        if scheme.eq_ignore_ascii_case("bearer") {
            // both compared, always
            let d = digest(rest.as_bytes());
            let full = self.token.as_ref().is_some_and(|t| ct_eq(t, &d));
            let read = self.read_token.as_ref().is_some_and(|t| ct_eq(t, &d));
            return if full {
                Some(Access::Full)
            } else if read {
                Some(Access::Read)
            } else {
                None
            };
        }
        if scheme.eq_ignore_ascii_case("basic") {
            let (u, p) = self.basic.as_ref()?;
            let raw = base64::engine::general_purpose::STANDARD.decode(rest).ok()?;
            let i = raw.iter().position(|&c| c == b':')?;
            // both compared, always (no short-circuit on the user name)
            let ok_u = ct_eq(u, &digest(&raw[..i]));
            let ok_p = ct_eq(p, &digest(&raw[i + 1..]));
            return (ok_u & ok_p).then_some(Access::Full);
        }
        None
    }
}

/// May read-only credentials make this request? Any GET, and the SQL endpoint.
fn read_only_request(method: &axum::http::Method, path: &str) -> bool {
    matches!(*method, axum::http::Method::GET | axum::http::Method::HEAD)
        || (*method == axum::http::Method::POST && path == "/sql")
}

/// axum middleware: 401 without valid credentials, 403 for a write with the read-only token.
pub async fn require(State(auth): State<Arc<Auth>>, req: Request, next: Next) -> Response {
    match auth.access(req.headers().get(header::AUTHORIZATION).map(|v| v.as_bytes())) {
        Some(Access::Full) => return next.run(req).await,
        Some(Access::Read) if read_only_request(req.method(), req.uri().path()) => return next.run(req).await,
        Some(Access::Read) => {
            return (
                StatusCode::FORBIDDEN,
                axum::Json(serde_json::json!({"error": {"kind": "forbidden", "message": "this token is read-only"}})),
            )
                .into_response()
        }
        None => {}
    }
    let mut r = (
        StatusCode::UNAUTHORIZED,
        axum::Json(serde_json::json!({"error": {"kind": "unauthorized", "message": "missing or invalid credentials"}})),
    )
        .into_response();
    if auth.basic.is_some() {
        r.headers_mut().insert(
            header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Basic realm=\"pgai-mon\""),
        );
    } else {
        r.headers_mut().insert(
            header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Bearer realm=\"pgai-mon\""),
        );
    }
    r
}

/// Liveness probe: no credentials, no state, no details.
pub async fn livez() -> &'static str {
    "ok\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k| m.get(k).cloned()
    }
    fn basic(u: &str, p: &str) -> String {
        format!(
            "Basic {}",
            base64::engine::general_purpose::STANDARD.encode(format!("{u}:{p}"))
        )
    }

    #[test]
    fn refuses_to_start_without_credentials_or_opt_out() {
        let e = Auth::resolve(&AuthCfg::default(), &env_of(&[])).err().unwrap();
        assert!(e.to_string().contains("no credentials"), "{e}");
        assert!(
            Auth::resolve(&AuthCfg::default(), &env_of(&[("PGAI_MON_AUTH_DISABLED", "1")]))
                .unwrap()
                .is_disabled()
        );
        let cfg = AuthCfg {
            disabled: true,
            ..Default::default()
        };
        assert!(Auth::resolve(&cfg, &env_of(&[])).unwrap().is_disabled());
        // the environment can turn an opt-out in the file off again
        assert!(Auth::resolve(&cfg, &env_of(&[("PGAI_MON_AUTH_DISABLED", "0")])).is_err());
    }

    #[test]
    fn bearer_token() {
        let a = Auth::resolve(
            &AuthCfg::default(),
            &env_of(&[("PGAI_MON_TOKEN", "dummy-token-0123456789")]),
        )
        .unwrap();
        assert!(a.check(Some(b"Bearer dummy-token-0123456789")));
        assert!(a.check(Some(b"bearer   dummy-token-0123456789 ")));
        assert!(!a.check(Some(b"Bearer 0123456789abcdef-toK")));
        assert!(!a.check(Some(b"Bearer 0123456789abcdef-to")));
        assert!(!a.check(Some(b"Bearer ")));
        assert!(!a.check(None));
        // no basic configured: basic is refused even with the token as password
        assert!(!a.check(Some(basic("x", "dummy-token-0123456789").as_bytes())));
    }

    #[test]
    fn the_read_only_token_reads_and_does_not_write() {
        use axum::http::Method;
        let env = env_of(&[
            ("PGAI_MON_TOKEN", "dummy-token-0123456789"),
            ("PGAI_MON_READ_TOKEN", "dummy-token-read-only-01"),
        ]);
        let a = Auth::resolve(&AuthCfg::default(), &env).unwrap();
        assert_eq!(a.access(Some(b"Bearer dummy-token-0123456789")), Some(Access::Full));
        assert_eq!(a.access(Some(b"Bearer dummy-token-read-only-01")), Some(Access::Read));
        assert_eq!(a.access(Some(b"Bearer dummy-token-read-only-02")), None);
        assert!(read_only_request(&Method::POST, "/sql"));
        assert!(read_only_request(&Method::GET, "/schema"));
        assert!(!read_only_request(&Method::POST, "/api/v1/import/prometheus"));
        // the same value for both would silently make the read-only token a full one
        let same = env_of(&[
            ("PGAI_MON_TOKEN", "dummy-token-0123456789"),
            ("PGAI_MON_READ_TOKEN", "dummy-token-0123456789"),
        ]);
        assert!(Auth::resolve(&AuthCfg::default(), &same).is_err());
        assert!(Auth::resolve(&AuthCfg::default(), &env_of(&[("PGAI_MON_READ_TOKEN", "short")])).is_err());
    }

    #[test]
    fn short_secrets_are_refused() {
        assert!(Auth::resolve(&AuthCfg::default(), &env_of(&[("PGAI_MON_TOKEN", "short")])).is_err());
        assert!(Auth::resolve(
            &AuthCfg::default(),
            &env_of(&[("PGAI_MON_AUTH_USERNAME", "u"), ("PGAI_MON_AUTH_PASSWORD", "short")])
        )
        .is_err());
        assert!(Auth::resolve(&AuthCfg::default(), &env_of(&[("PGAI_MON_AUTH_USERNAME", "u")])).is_err());
    }

    #[test]
    fn basic_auth_from_files_and_both_schemes() {
        let d = tempfile::tempdir().unwrap();
        let (tf, pf) = (d.path().join("token"), d.path().join("pw"));
        std::fs::write(&tf, "tok-0123456789abcdef\n").unwrap();
        std::fs::write(&pf, "  pw-0123456789abcdef:with-colon\n").unwrap();
        let cfg = AuthCfg {
            token_file: Some(tf),
            username: Some("pgai".into()),
            password_file: Some(pf),
            ..Default::default()
        };
        let a = Auth::resolve(&cfg, &env_of(&[])).unwrap();
        assert!(a.check(Some(b"Bearer tok-0123456789abcdef")));
        assert!(a.check(Some(basic("pgai", "pw-0123456789abcdef:with-colon").as_bytes())));
        assert!(!a.check(Some(basic("pgai", "pw-0123456789abcdef").as_bytes())));
        assert!(!a.check(Some(basic("PGAI", "pw-0123456789abcdef:with-colon").as_bytes())));
        assert!(!a.check(Some(b"Basic !!!notbase64")));
        assert!(!a.check(Some(b"Digest x")));
        // environment wins over the file
        let a = Auth::resolve(&cfg, &env_of(&[("PGAI_MON_TOKEN", "dummy-token-from-env-01")])).unwrap();
        assert!(a.check(Some(b"Bearer dummy-token-from-env-01")));
        assert!(!a.check(Some(b"Bearer tok-0123456789abcdef")));
        // a missing file is an error, not "no auth"
        let cfg = AuthCfg {
            token_file: Some(d.path().join("nope")),
            ..Default::default()
        };
        assert!(Auth::resolve(&cfg, &env_of(&[])).is_err());
    }
}
