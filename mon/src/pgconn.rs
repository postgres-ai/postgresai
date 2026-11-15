//! libpq-style connection handling for the collector: URL and key=value connection
//! strings, PG* environment defaults, the password file (.pgpass), and TLS through rustls
//! with libpq's sslmode semantics (disable, allow, prefer, require, verify-ca, verify-full),
//! sslrootcert (a PEM file, or `system` for the OS store), sslcert/sslkey client certificates.
//!
//! Differences from libpq, by design or not yet done: the default host is `localhost` (not a
//! Unix socket directory); `allow` behaves as `prefer`; service files, sslcrl,
//! sslpassword and GSS are not supported (their keys are ignored with a warning, once per host
//! and database); SCRAM runs without channel binding, so `channel_binding=require` is refused; a
//! root.crt in ~/.postgresql is used only for verify-ca/verify-full (libpq also makes
//! `require` verify against it when present; here `require` verifies only with an explicit
//! sslrootcert).
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::client::WebPkiServerVerifier;
use rustls::crypto::CryptoProvider;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, UnixTime};
use rustls::{CertificateError, DigitallySignedStruct, RootCertStore, SignatureScheme};
use std::collections::BTreeMap;
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio_postgres::config::SslMode;
use tokio_postgres::tls::{ChannelBinding, MakeTlsConnect, TlsConnect, TlsStream};

/// Keys this module understands (libpq names). Others are ignored with a warning.
const KNOWN: &[&str] = &[
    "host",
    "hostaddr",
    "port",
    "dbname",
    "user",
    "password",
    "passfile",
    "connect_timeout",
    "options",
    "application_name",
    "fallback_application_name",
    "sslmode",
    "sslrootcert",
    "sslcert",
    "sslkey",
    "keepalives",
    "keepalives_idle",
    "target_session_attrs",
    "load_balance_hosts",
    "channel_binding",
];

/// Unsupported keys of `p` not reported before for its host and database: the warning comes
/// once, not at every connect.
fn first_unknown(p: &ConnParams) -> Vec<String> {
    static SEEN: std::sync::Mutex<std::collections::BTreeSet<String>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    let mut seen = SEEN.lock().unwrap();
    p.0.keys()
        .filter(|k| !KNOWN.contains(&k.as_str()) && seen.insert(format!("{} {k}", p.redacted())))
        .cloned()
        .collect()
}

/// libpq environment variables and the key each one defaults.
const ENV_DEFAULTS: &[(&str, &str)] = &[
    ("PGHOST", "host"),
    ("PGHOSTADDR", "hostaddr"),
    ("PGPORT", "port"),
    ("PGDATABASE", "dbname"),
    ("PGUSER", "user"),
    ("PGPASSWORD", "password"),
    ("PGPASSFILE", "passfile"),
    ("PGOPTIONS", "options"),
    ("PGCONNECT_TIMEOUT", "connect_timeout"),
    ("PGSSLMODE", "sslmode"),
    ("PGSSLROOTCERT", "sslrootcert"),
    ("PGSSLCERT", "sslcert"),
    ("PGSSLKEY", "sslkey"),
    ("PGTARGETSESSIONATTRS", "target_session_attrs"),
];

/// A parsed connection string: libpq keys and values, as written (no defaults applied).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConnParams(pub BTreeMap<String, String>);

impl ConnParams {
    /// Parse a `postgresql://` / `postgres://` URL or a `key=value ...` string.
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        let t = s.trim();
        if t.starts_with("postgresql://") || t.starts_with("postgres://") {
            parse_url(t)
        } else {
            parse_kv(t)
        }
    }

    pub fn get(&self, k: &str) -> Option<&str> {
        self.0.get(k).map(|s| s.as_str())
    }

    pub fn set(&mut self, k: &str, v: impl Into<String>) {
        self.0.insert(k.to_string(), v.into());
    }

    /// The same node, another database.
    pub fn with_dbname(&self, db: &str) -> Self {
        let mut p = self.clone();
        p.set("dbname", db);
        p
    }

    /// key=value form, every value quoted (round-trips through `parse`).
    pub fn to_kv_string(&self) -> String {
        self.0
            .iter()
            .map(|(k, v)| format!("{k}='{}'", v.replace('\\', "\\\\").replace('\'', "\\'")))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// For logs: host, port and database only.
    pub fn redacted(&self) -> String {
        format!(
            "{}:{}/{}",
            self.get("host").or(self.get("hostaddr")).unwrap_or(""),
            self.get("port").unwrap_or(""),
            self.get("dbname").unwrap_or("")
        )
    }

    /// libpq's resolution: keys not in the string come from the PG* environment, then
    /// defaults; a missing password comes from the password file.
    pub fn resolve(&self, env: &dyn Fn(&str) -> Option<String>) -> anyhow::Result<Self> {
        let mut p = self.clone();
        for (var, key) in ENV_DEFAULTS {
            if !p.0.contains_key(*key) {
                if let Some(v) = env(var).filter(|v| !v.is_empty()) {
                    p.set(key, v);
                }
            }
        }
        if !p.0.contains_key("host") && !p.0.contains_key("hostaddr") {
            p.set("host", "localhost");
        }
        if !p.0.contains_key("user") {
            p.set("user", env("USER").unwrap_or_else(|| "postgres".into()));
        }
        if !p.0.contains_key("dbname") {
            let u = p.get("user").unwrap().to_string();
            p.set("dbname", u);
        }
        if p.get("password").is_none_or(|s| s.is_empty()) {
            let file = p
                .get("passfile")
                .map(PathBuf::from)
                .or_else(|| env("HOME").map(|h| Path::new(&h).join(".pgpass")));
            if let Some(f) = file {
                if let Some(pw) = pgpass_lookup(&f, &p)? {
                    p.set("password", pw);
                }
            }
        }
        if p.get("sslrootcert") == Some("system") {
            match p.get("sslmode") {
                None => p.set("sslmode", "verify-full"),
                Some("verify-full") => {}
                Some(m) => anyhow::bail!("sslrootcert=system requires sslmode=verify-full, not {m}"),
            }
        }
        match p.get("channel_binding") {
            None | Some("prefer") | Some("disable") => {}
            Some("require") => anyhow::bail!("channel_binding=require is not supported (no SCRAM channel binding)"),
            Some(v) => anyhow::bail!("invalid channel_binding value {v:?}"),
        }
        for k in first_unknown(&p) {
            eprintln!(
                "collector: {}: connection option {k} is not supported, ignored",
                p.redacted()
            );
        }
        Ok(p)
    }

    /// tokio-postgres config and the TLS connector for resolved parameters.
    pub fn build(
        &self,
        application_name: &str,
        home: Option<&Path>,
    ) -> anyhow::Result<(tokio_postgres::Config, MakeRustls)> {
        let mut cfg = tokio_postgres::Config::new();
        let hosts: Vec<&str> = self.get("host").map(|h| h.split(',').collect()).unwrap_or_default();
        for h in &hosts {
            cfg.host(*h);
        }
        if let Some(a) = self.get("hostaddr") {
            for a in a.split(',') {
                cfg.hostaddr(a.trim().parse().map_err(|_| anyhow::anyhow!("invalid hostaddr {a}"))?);
            }
        }
        if let Some(ports) = self.get("port") {
            for p in ports.split(',') {
                let p = p.trim();
                cfg.port(if p.is_empty() {
                    5432
                } else {
                    p.parse().map_err(|_| anyhow::anyhow!("invalid port {p}"))?
                });
            }
        }
        if let Some(u) = self.get("user") {
            cfg.user(u);
        }
        if let Some(pw) = self.get("password") {
            cfg.password(pw);
        }
        if let Some(d) = self.get("dbname") {
            cfg.dbname(d);
        }
        if let Some(o) = self.get("options") {
            cfg.options(o);
        }
        cfg.application_name(application_name);
        let ct = match self.get("connect_timeout") {
            // libpq: values below 2 s are treated as 2 s; 0 means wait forever.
            Some(s) => match s.trim().parse::<u64>() {
                Ok(0) => None,
                Ok(n) => Some(n.max(2)),
                Err(_) => anyhow::bail!("invalid connect_timeout {s}"),
            },
            None => Some(10),
        };
        if let Some(s) = ct {
            cfg.connect_timeout(Duration::from_secs(s));
        }
        if let Some(k) = self.get("keepalives") {
            cfg.keepalives(k != "0");
        }
        if let Some(k) = self.get("keepalives_idle") {
            cfg.keepalives_idle(Duration::from_secs(
                k.parse().map_err(|_| anyhow::anyhow!("invalid keepalives_idle {k}"))?,
            ));
        }
        if let Some(t) = self.get("target_session_attrs") {
            cfg.target_session_attrs(match t {
                "any" => tokio_postgres::config::TargetSessionAttrs::Any,
                "read-write" => tokio_postgres::config::TargetSessionAttrs::ReadWrite,
                "read-only" => tokio_postgres::config::TargetSessionAttrs::ReadOnly,
                _ => anyhow::bail!("target_session_attrs {t} is not supported (any, read-write, read-only)"),
            });
        }
        if let Some(l) = self.get("load_balance_hosts") {
            cfg.load_balance_hosts(match l {
                "disable" => tokio_postgres::config::LoadBalanceHosts::Disable,
                "random" => tokio_postgres::config::LoadBalanceHosts::Random,
                _ => anyhow::bail!("invalid load_balance_hosts {l}"),
            });
        }
        let mode = self.get("sslmode").unwrap_or("prefer");
        cfg.ssl_mode(match mode {
            "disable" => SslMode::Disable,
            "allow" | "prefer" => SslMode::Prefer,
            "require" | "verify-ca" | "verify-full" => SslMode::Require,
            m => anyhow::bail!("invalid sslmode {m}"),
        });
        let tls = MakeRustls::new(self, home)?;
        Ok((cfg, tls))
    }
}

fn pct_decode(s: &str) -> anyhow::Result<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let h = s
                .get(i + 1..i + 3)
                .ok_or_else(|| anyhow::anyhow!("bad percent-encoding in connection URL"))?;
            out.push(u8::from_str_radix(h, 16).map_err(|_| anyhow::anyhow!("bad percent-encoding in connection URL"))?);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    Ok(String::from_utf8(out)?)
}

fn parse_url(s: &str) -> anyhow::Result<ConnParams> {
    let rest = s.split_once("://").unwrap().1;
    let (main, query) = match rest.split_once('?') {
        Some((m, q)) => (m, Some(q)),
        None => (rest, None),
    };
    let (authority, path) = match main.find('/') {
        Some(i) => (&main[..i], Some(&main[i + 1..])),
        None => (main, None),
    };
    let mut p = ConnParams::default();
    let hostspec = match authority.rfind('@') {
        Some(i) => {
            let userinfo = &authority[..i];
            match userinfo.split_once(':') {
                Some((u, pw)) => {
                    if !u.is_empty() {
                        p.set("user", pct_decode(u)?);
                    }
                    p.set("password", pct_decode(pw)?);
                }
                None if !userinfo.is_empty() => p.set("user", pct_decode(userinfo)?),
                None => {}
            }
            &authority[i + 1..]
        }
        None => authority,
    };
    if !hostspec.is_empty() {
        let mut hosts = Vec::new();
        let mut ports = Vec::new();
        for h in hostspec.split(',') {
            let (host, port) = if let Some(r) = h.strip_prefix('[') {
                let (v6, after) = r
                    .split_once(']')
                    .ok_or_else(|| anyhow::anyhow!("unterminated [ in connection URL"))?;
                (v6.to_string(), after.strip_prefix(':').unwrap_or("").to_string())
            } else {
                match h.rsplit_once(':') {
                    Some((a, b)) => (pct_decode(a)?, b.to_string()),
                    None => (pct_decode(h)?, String::new()),
                }
            };
            hosts.push(host);
            ports.push(port);
        }
        if hosts.iter().any(|h| !h.is_empty()) {
            p.set("host", hosts.join(","));
        }
        if ports.iter().any(|x| !x.is_empty()) {
            p.set("port", ports.join(","));
        }
    }
    if let Some(d) = path.filter(|d| !d.is_empty()) {
        p.set("dbname", pct_decode(d)?);
    }
    if let Some(q) = query {
        for kv in q.split('&').filter(|x| !x.is_empty()) {
            let (k, v) = kv
                .split_once('=')
                // the text is not echoed: with an unescaped '?' in a password it is part of the secret
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "connection URL: a parameter after '?' has no '=' (a '?' in the password must be written %3F)"
                    )
                })?;
            p.set(&pct_decode(k)?, pct_decode(v)?);
        }
    }
    Ok(p)
}

fn parse_kv(s: &str) -> anyhow::Result<ConnParams> {
    let mut p = ConnParams::default();
    let mut it = s.chars().peekable();
    loop {
        while it.peek().is_some_and(|c| c.is_whitespace()) {
            it.next();
        }
        if it.peek().is_none() {
            break;
        }
        let mut key = String::new();
        while let Some(&c) = it.peek() {
            if c == '=' || c.is_whitespace() {
                break;
            }
            key.push(c);
            it.next();
        }
        while it.peek().is_some_and(|c| c.is_whitespace()) {
            it.next();
        }
        anyhow::ensure!(
            it.next() == Some('='),
            "missing \"=\" after \"{key}\" in connection string"
        );
        while it.peek().is_some_and(|c| c.is_whitespace()) {
            it.next();
        }
        let mut val = String::new();
        if it.peek() == Some(&'\'') {
            it.next();
            loop {
                match it.next() {
                    Some('\\') => val.push(it.next().ok_or_else(|| anyhow::anyhow!("unterminated quoted string"))?),
                    Some('\'') => break,
                    Some(c) => val.push(c),
                    None => anyhow::bail!("unterminated quoted string in connection string"),
                }
            }
        } else {
            while let Some(&c) = it.peek() {
                if c.is_whitespace() {
                    break;
                }
                it.next();
                if c == '\\' {
                    if let Some(n) = it.next() {
                        val.push(n);
                    }
                } else {
                    val.push(c);
                }
            }
        }
        p.set(&key, val);
    }
    Ok(p)
}

/// libpq's password file: `hostname:port:database:username:password`, `*` matches anything,
/// `\:` and `\\` escape. A Unix-socket host matches `localhost`. Ignored (with a warning)
/// when group or others can read it, as libpq does.
pub fn pgpass_lookup(file: &Path, p: &ConnParams) -> anyhow::Result<Option<String>> {
    let Ok(meta) = std::fs::metadata(file) else {
        return Ok(None);
    };
    if !meta.is_file() {
        return Ok(None);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if meta.permissions().mode() & 0o077 != 0 {
            eprintln!(
                "collector: password file {} has group or world access; permissions should be u=rw (0600) or less; ignored",
                file.display()
            );
            return Ok(None);
        }
    }
    let text = std::fs::read_to_string(file)?;
    let host = p.get("host").and_then(|h| h.split(',').next()).unwrap_or("localhost");
    let host = if host.is_empty() || host.starts_with('/') {
        "localhost"
    } else {
        host
    };
    let port = p
        .get("port")
        .and_then(|h| h.split(',').next())
        .filter(|x| !x.is_empty())
        .unwrap_or("5432");
    let want = [host, port, p.get("dbname").unwrap_or(""), p.get("user").unwrap_or("")];
    for line in text.lines() {
        if line.trim_start().starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut fields: Vec<String> = vec![String::new()];
        let mut chars = line.chars();
        while let Some(c) = chars.next() {
            match c {
                '\\' => {
                    if let Some(n) = chars.next() {
                        fields.last_mut().unwrap().push(n);
                    }
                }
                ':' if fields.len() < 5 => fields.push(String::new()),
                c => fields.last_mut().unwrap().push(c),
            }
        }
        if fields.len() != 5 {
            continue;
        }
        if fields[..4].iter().zip(want).all(|(f, w)| f == "*" || f == w) {
            return Ok(Some(fields[4].clone()));
        }
    }
    Ok(None)
}

/// What the TLS handshake checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verify {
    /// prefer/require without sslrootcert: encryption only.
    None,
    /// verify-ca (and require with sslrootcert): the chain, not the host name.
    Ca,
    /// verify-full: the chain and the host name.
    Full,
}

/// tokio-postgres TLS connector over rustls (no OpenSSL). tokio-postgres decides by sslmode
/// whether it is used at all.
#[derive(Clone)]
pub struct MakeRustls {
    config: Option<Arc<rustls::ClientConfig>>,
    pub verify: Verify,
}

fn provider() -> Arc<CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

fn load_roots(spec: &str) -> anyhow::Result<RootCertStore> {
    let mut roots = RootCertStore::empty();
    if spec == "system" {
        let r = rustls_native_certs::load_native_certs();
        for c in r.certs {
            let _ = roots.add(c);
        }
        anyhow::ensure!(
            !roots.is_empty(),
            "sslrootcert=system: no certificates in the system store"
        );
    } else {
        for c in CertificateDer::pem_file_iter(spec).map_err(|e| anyhow::anyhow!("sslrootcert {spec}: {e}"))? {
            roots.add(c.map_err(|e| anyhow::anyhow!("sslrootcert {spec}: {e}"))?)?;
        }
        anyhow::ensure!(!roots.is_empty(), "sslrootcert {spec}: no certificates");
    }
    Ok(roots)
}

impl MakeRustls {
    pub fn new(p: &ConnParams, home: Option<&Path>) -> anyhow::Result<Self> {
        let mode = p.get("sslmode").unwrap_or("prefer");
        if mode == "disable" {
            return Ok(Self {
                config: None,
                verify: Verify::None,
            });
        }
        let verify = match mode {
            "verify-full" => Verify::Full,
            "verify-ca" => Verify::Ca,
            "require" if p.get("sslrootcert").is_some() => Verify::Ca,
            _ => Verify::None,
        };
        let prov = provider();
        let verifier: Arc<dyn ServerCertVerifier> = match verify {
            Verify::None => Arc::new(NoVerify(prov.clone())),
            _ => {
                let spec = match p.get("sslrootcert") {
                    Some(s) => s.to_string(),
                    None => {
                        let f = home.map(|h| h.join(".postgresql/root.crt"));
                        match f {
                            Some(f) if f.exists() => f.to_string_lossy().into_owned(),
                            // pgwatch (pgx) verified against the system store here, so targets
                            // with a public-CA certificate worked with a bare verify-full. Not
                            // for verify-ca: any public CA's certificate for any name would pass.
                            _ if verify == Verify::Full => "system".to_string(),
                            _ => anyhow::bail!(
                                "sslmode={mode}: no root certificate (set sslrootcert, or put one in ~/.postgresql/root.crt)"
                            ),
                        }
                    }
                };
                let inner =
                    WebPkiServerVerifier::builder_with_provider(Arc::new(load_roots(&spec)?), prov.clone()).build()?;
                if verify == Verify::Full {
                    inner
                } else {
                    Arc::new(CaOnly(inner))
                }
            }
        };
        let b = rustls::ClientConfig::builder_with_provider(prov)
            .with_safe_default_protocol_versions()?
            .dangerous()
            .with_custom_certificate_verifier(verifier);
        let config = match (p.get("sslcert"), p.get("sslkey")) {
            (Some(c), Some(k)) => {
                let certs = CertificateDer::pem_file_iter(c)
                    .map_err(|e| anyhow::anyhow!("sslcert {c}: {e}"))?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| anyhow::anyhow!("sslcert {c}: {e}"))?;
                let key = PrivateKeyDer::from_pem_file(k).map_err(|e| anyhow::anyhow!("sslkey {k}: {e}"))?;
                b.with_client_auth_cert(certs, key)?
            }
            (None, None) => b.with_no_client_auth(),
            _ => anyhow::bail!("sslcert and sslkey must be given together"),
        };
        Ok(Self {
            config: Some(Arc::new(config)),
            verify,
        })
    }
}

/// prefer/require: any certificate, but the handshake signatures are still checked.
#[derive(Debug)]
struct NoVerify(Arc<CryptoProvider>);

impl ServerCertVerifier for NoVerify {
    fn verify_server_cert(
        &self,
        _: &CertificateDer<'_>,
        _: &[CertificateDer<'_>],
        _: &ServerName<'_>,
        _: &[u8],
        _: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(m, c, d, &self.0.signature_verification_algorithms)
    }
    fn verify_tls13_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(m, c, d, &self.0.signature_verification_algorithms)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

/// verify-ca: the WebPKI chain check with name mismatches accepted.
#[derive(Debug)]
struct CaOnly(Arc<WebPkiServerVerifier>);

impl ServerCertVerifier for CaOnly {
    fn verify_server_cert(
        &self,
        e: &CertificateDer<'_>,
        i: &[CertificateDer<'_>],
        n: &ServerName<'_>,
        o: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        match self.0.verify_server_cert(e, i, n, o, now) {
            Err(rustls::Error::InvalidCertificate(CertificateError::NotValidForName))
            | Err(rustls::Error::InvalidCertificate(CertificateError::NotValidForNameContext { .. })) => {
                Ok(ServerCertVerified::assertion())
            }
            r => r,
        }
    }
    fn verify_tls12_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.0.verify_tls12_signature(m, c, d)
    }
    fn verify_tls13_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.0.verify_tls13_signature(m, c, d)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.supported_verify_schemes()
    }
}

pub struct RustlsConnect {
    config: Option<Arc<rustls::ClientConfig>>,
    name: ServerName<'static>,
}

impl<S> MakeTlsConnect<S> for MakeRustls
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    type Stream = RustlsStream<S>;
    type TlsConnect = RustlsConnect;
    type Error = io::Error;

    fn make_tls_connect(&mut self, domain: &str) -> io::Result<RustlsConnect> {
        let name = match ServerName::try_from(domain.to_string()) {
            Ok(n) => n,
            // Only a full verification needs the name to be valid.
            Err(e) if self.verify == Verify::Full => return Err(io::Error::new(io::ErrorKind::InvalidInput, e)),
            Err(_) => ServerName::try_from("localhost").unwrap(),
        };
        Ok(RustlsConnect {
            config: self.config.clone(),
            name,
        })
    }
}

impl<S> TlsConnect<S> for RustlsConnect
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    type Stream = RustlsStream<S>;
    type Error = io::Error;
    type Future = Pin<Box<dyn Future<Output = io::Result<RustlsStream<S>>> + Send>>;

    fn connect(self, stream: S) -> Self::Future {
        Box::pin(async move {
            let cfg = self
                .config
                .ok_or_else(|| io::Error::new(io::ErrorKind::Unsupported, "TLS disabled (sslmode=disable)"))?;
            let s = tokio_rustls::TlsConnector::from(cfg).connect(self.name, stream).await?;
            Ok(RustlsStream(s))
        })
    }
}

pub struct RustlsStream<S>(tokio_rustls::client::TlsStream<S>);

impl<S: AsyncRead + AsyncWrite + Unpin> TlsStream for RustlsStream<S> {
    fn channel_binding(&self) -> ChannelBinding {
        ChannelBinding::none()
    }
}

impl<S: AsyncRead + AsyncWrite + Unpin> AsyncRead for RustlsStream<S> {
    fn poll_read(mut self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &mut ReadBuf<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.0).poll_read(cx, buf)
    }
}

impl<S: AsyncRead + AsyncWrite + Unpin> AsyncWrite for RustlsStream<S> {
    fn poll_write(mut self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &[u8]) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.0).poll_write(cx, buf)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.0).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.0).poll_shutdown(cx)
    }
}

fn process_env(k: &str) -> Option<String> {
    std::env::var(k).ok()
}

/// Parse, resolve against the process environment, and connect. The connection task is
/// spawned; its end is logged with `label`.
pub async fn connect(conn_str: &str, application_name: &str) -> anyhow::Result<tokio_postgres::Client> {
    let p = ConnParams::parse(conn_str)?.resolve(&process_env)?;
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let (cfg, tls) = p.build(application_name, home.as_deref())?;
    let (client, conn) = cfg.connect(tls).await?;
    let label = p.redacted();
    tokio::spawn(async move {
        if let Err(e) = conn.await {
            eprintln!("collector: connection to {label} closed: {e}");
        }
    });
    Ok(client)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(p: &ConnParams) -> Vec<(&str, &str)> {
        p.0.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
    }

    #[test]
    fn url_form() {
        let p = ConnParams::parse(
            "postgresql://mon%40x:p%3Ass@db1:5433,[::1]:5434/my%20db?sslmode=verify-full&sslrootcert=/ca.pem",
        )
        .unwrap();
        assert_eq!(
            kv(&p),
            vec![
                ("dbname", "my db"),
                ("host", "db1,::1"),
                ("password", "p:ss"),
                ("port", "5433,5434"),
                ("sslmode", "verify-full"),
                ("sslrootcert", "/ca.pem"),
                ("user", "mon@x"),
            ]
        );
        let p = ConnParams::parse("postgres://host").unwrap();
        assert_eq!(kv(&p), vec![("host", "host")]);
        let p = ConnParams::parse("postgresql:///db?host=%2Fvar%2Frun%2Fpostgresql").unwrap();
        assert_eq!(kv(&p), vec![("dbname", "db"), ("host", "/var/run/postgresql")]);
    }

    #[test]
    fn kv_form_and_round_trip() {
        let p = ConnParams::parse(r"host=h port = 5432 user=u password='x y' dbname=d\ b").unwrap();
        assert_eq!(p.get("password"), Some("x y"));
        assert_eq!(p.get("host"), Some("h"));
        assert_eq!(p.get("port"), Some("5432"));
        assert_eq!(p.get("dbname"), Some("d b"));
        let p = ConnParams::parse(r"password='a b\'c\\d'").unwrap();
        assert_eq!(p.get("password"), Some(r"a b'c\d"));
        assert_eq!(ConnParams::parse(&p.to_kv_string()).unwrap(), p);
        assert!(ConnParams::parse("host").is_err());
        assert!(ConnParams::parse("password='open").is_err());
    }

    #[test]
    fn other_database_keeps_everything_else() {
        let p = ConnParams::parse("postgresql://u:pw@h:6432/main?sslmode=require&options=-c%20x%3D1").unwrap();
        let q = ConnParams::parse(&p.with_dbname("other db").to_kv_string()).unwrap();
        assert_eq!(q.get("dbname"), Some("other db"));
        assert_eq!(q.get("sslmode"), Some("require"));
        assert_eq!(q.get("options"), Some("-c x=1"));
        assert_eq!(q.get("password"), Some("pw"));
        assert_eq!(q.get("port"), Some("6432"));
    }

    #[test]
    fn env_defaults_and_pgpass() {
        let dir = tempfile::tempdir().unwrap();
        let pass = dir.path().join("pgpass");
        std::fs::write(
            &pass,
            "# comment\nother:*:*:*:nope\nh1:5433:*:mon:se\\:cret\n*:*:*:*:fallback\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&pass, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let passfile = pass.to_string_lossy().into_owned();
        let env = move |k: &str| match k {
            "PGHOST" => Some("h1".to_string()),
            "PGPORT" => Some("5433".to_string()),
            "PGUSER" => Some("mon".to_string()),
            "PGPASSFILE" => Some(passfile.clone()),
            "PGSSLMODE" => Some("require".to_string()),
            _ => None,
        };
        let p = ConnParams::parse("dbname=app").unwrap().resolve(&env).unwrap();
        assert_eq!(p.get("host"), Some("h1"));
        assert_eq!(p.get("password"), Some("se:cret"));
        assert_eq!(p.get("sslmode"), Some("require"));
        // the string wins over the environment; another user falls through to `*`
        let p = ConnParams::parse("user=x dbname=app sslmode=disable")
            .unwrap()
            .resolve(&env)
            .unwrap();
        assert_eq!(p.get("password"), Some("fallback"));
        assert_eq!(p.get("sslmode"), Some("disable"));
        // PGPASSWORD before the file
        let env2 = |k: &str| (k == "PGPASSWORD").then(|| "envpw".to_string());
        let p = ConnParams::parse("host=h1").unwrap().resolve(&env2).unwrap();
        assert_eq!(p.get("password"), Some("envpw"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&pass, std::fs::Permissions::from_mode(0o644)).unwrap();
            let p = ConnParams::parse("dbname=app").unwrap().resolve(&env).unwrap();
            assert_eq!(p.get("password"), None, "a world-readable password file is ignored");
        }
    }

    #[test]
    fn sslmodes() {
        let b = |s: &str| ConnParams::parse(s).unwrap().build("t", None);
        assert!(b("host=h sslmode=bogus").is_err());
        assert_eq!(b("host=h sslmode=disable").unwrap().1.verify, Verify::None);
        assert_eq!(b("host=h").unwrap().1.verify, Verify::None);
        assert_eq!(b("host=h sslmode=require").unwrap().1.verify, Verify::None);
        // verify-ca needs a root certificate; a bare verify-full checks against the system store
        let e = b("host=h sslmode=verify-ca").err().unwrap().to_string();
        assert!(e.contains("root certificate"), "{e}");
        assert_eq!(b("host=h sslmode=verify-full").unwrap().1.verify, Verify::Full);
        assert!(b("host=h sslmode=verify-ca sslrootcert=/nonexistent").is_err());
        assert!(b("host=h sslcert=/a").is_err(), "sslcert without sslkey");
        let r = ConnParams::parse("host=h sslrootcert=system sslmode=require")
            .unwrap()
            .resolve(&|_| None);
        assert!(r.is_err());
        let r = ConnParams::parse("host=h sslrootcert=system")
            .unwrap()
            .resolve(&|_| None)
            .unwrap();
        assert_eq!(r.get("sslmode"), Some("verify-full"));
    }

    #[test]
    fn unknown_key_is_reported_once_per_database() {
        let p = ConnParams::parse("host=once-h dbname=a pool_max_conns=4 sslcrl=x").unwrap();
        assert_eq!(first_unknown(&p), vec!["pool_max_conns", "sslcrl"]);
        assert!(first_unknown(&p).is_empty(), "the next connect stays quiet");
        assert_eq!(first_unknown(&p.with_dbname("b")), vec!["pool_max_conns", "sslcrl"]);
    }

    #[test]
    fn channel_binding_require_is_refused() {
        let r = |s: &str| ConnParams::parse(s).unwrap().resolve(&|_| None);
        let e = r("host=h channel_binding=require").err().unwrap().to_string();
        assert!(e.contains("channel_binding=require is not supported"), "{e}");
        assert!(r("host=h channel_binding=prefer").is_ok());
        assert!(r("host=h channel_binding=disable").is_ok());
        assert!(
            first_unknown(&ConnParams::parse("host=cb-h channel_binding=prefer").unwrap()).is_empty(),
            "no warning"
        );
    }
}
