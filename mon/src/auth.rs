//! HTTP authentication. Every route needs credentials except the liveness probe (`/healthz`):
//! a bearer token, HTTP basic auth, or either when both are configured. Credentials come from
//! files or the environment, never from the YAML itself. Running without authentication must be
//! asked for explicitly (`http.auth.disabled: true` or `PGAI_MON_AUTH_DISABLED=1`), for local
//! development only; with no credentials and no opt-out the server refuses to start.
use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// `http.auth` in the YAML. Environment variables override each field:
/// PGAI_MON_AUTH_TOKEN, PGAI_MON_AUTH_TOKEN_FILE, PGAI_MON_AUTH_USERNAME,
/// PGAI_MON_AUTH_PASSWORD, PGAI_MON_AUTH_PASSWORD_FILE, PGAI_MON_AUTH_DISABLED.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct AuthCfg {
    /// File holding the bearer token (surrounding whitespace is ignored).
    pub token_file: Option<PathBuf>,
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
    basic: Option<([u8; 32], [u8; 32])>,
    disabled: bool,
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
        let token = match (env("PGAI_MON_AUTH_TOKEN"), env("PGAI_MON_AUTH_TOKEN_FILE")) {
            (Some(t), _) => Some(t.trim().to_string()),
            (None, Some(f)) => Some(read_secret(std::path::Path::new(&f), "token")?),
            (None, None) => match &cfg.token_file {
                Some(f) => Some(read_secret(f, "token")?),
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
        if !disabled && token.is_none() && basic.is_none() {
            anyhow::bail!(
                "http.auth: no credentials. Set PGAI_MON_AUTH_TOKEN(_FILE) or PGAI_MON_AUTH_USERNAME + \
                 PGAI_MON_AUTH_PASSWORD(_FILE) (or http.auth in the config); for local development only, \
                 PGAI_MON_AUTH_DISABLED=1 / http.auth.disabled: true serves without authentication"
            );
        }
        Ok(Self { token, basic, disabled })
    }

    pub fn from_env(cfg: &AuthCfg) -> anyhow::Result<Self> {
        Self::resolve(cfg, &|k| std::env::var(k).ok())
    }

    /// No authentication (tests and explicit local development).
    pub fn disabled() -> Self {
        Self {
            token: None,
            basic: None,
            disabled: true,
        }
    }

    pub fn is_disabled(&self) -> bool {
        self.disabled
    }

    /// Does this Authorization header value carry valid credentials?
    pub fn check(&self, authz: Option<&[u8]>) -> bool {
        if self.disabled {
            return true;
        }
        let Some(h) = authz else { return false };
        let Ok(h) = std::str::from_utf8(h) else { return false };
        let (scheme, rest) = h.trim().split_once(' ').unwrap_or((h, ""));
        let rest = rest.trim();
        if scheme.eq_ignore_ascii_case("bearer") {
            return self.token.as_ref().is_some_and(|t| ct_eq(t, &digest(rest.as_bytes())));
        }
        if scheme.eq_ignore_ascii_case("basic") {
            let Some((u, p)) = &self.basic else { return false };
            let Ok(raw) = base64::engine::general_purpose::STANDARD.decode(rest) else {
                return false;
            };
            let Some(i) = raw.iter().position(|&c| c == b':') else {
                return false;
            };
            // both compared, always (no short-circuit on the user name)
            let ok_u = ct_eq(u, &digest(&raw[..i]));
            let ok_p = ct_eq(p, &digest(&raw[i + 1..]));
            return ok_u & ok_p;
        }
        false
    }
}

/// axum middleware: 401 without valid credentials.
pub async fn require(State(auth): State<Arc<Auth>>, req: Request, next: Next) -> Response {
    if auth.check(req.headers().get(header::AUTHORIZATION).map(|v| v.as_bytes())) {
        return next.run(req).await;
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
pub async fn healthz() -> &'static str {
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
            &env_of(&[("PGAI_MON_AUTH_TOKEN", "0123456789abcdef-tok")]),
        )
        .unwrap();
        assert!(a.check(Some(b"Bearer 0123456789abcdef-tok")));
        assert!(a.check(Some(b"bearer   0123456789abcdef-tok ")));
        assert!(!a.check(Some(b"Bearer 0123456789abcdef-toK")));
        assert!(!a.check(Some(b"Bearer 0123456789abcdef-to")));
        assert!(!a.check(Some(b"Bearer ")));
        assert!(!a.check(None));
        // no basic configured: basic is refused even with the token as password
        assert!(!a.check(Some(basic("x", "0123456789abcdef-tok").as_bytes())));
    }

    #[test]
    fn short_secrets_are_refused() {
        assert!(Auth::resolve(&AuthCfg::default(), &env_of(&[("PGAI_MON_AUTH_TOKEN", "short")])).is_err());
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
            disabled: false,
        };
        let a = Auth::resolve(&cfg, &env_of(&[])).unwrap();
        assert!(a.check(Some(b"Bearer tok-0123456789abcdef")));
        assert!(a.check(Some(basic("pgai", "pw-0123456789abcdef:with-colon").as_bytes())));
        assert!(!a.check(Some(basic("pgai", "pw-0123456789abcdef").as_bytes())));
        assert!(!a.check(Some(basic("PGAI", "pw-0123456789abcdef:with-colon").as_bytes())));
        assert!(!a.check(Some(b"Basic !!!notbase64")));
        assert!(!a.check(Some(b"Digest x")));
        // environment wins over the file
        let a = Auth::resolve(&cfg, &env_of(&[("PGAI_MON_AUTH_TOKEN", "env-0123456789abcdef")])).unwrap();
        assert!(a.check(Some(b"Bearer env-0123456789abcdef")));
        assert!(!a.check(Some(b"Bearer tok-0123456789abcdef")));
        // a missing file is an error, not "no auth"
        let cfg = AuthCfg {
            token_file: Some(d.path().join("nope")),
            ..Default::default()
        };
        assert!(Auth::resolve(&cfg, &env_of(&[])).is_err());
    }
}
