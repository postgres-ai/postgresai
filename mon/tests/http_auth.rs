//! HTTP authentication over the real app: every route needs credentials except /healthz.
mod ingest_common;
use base64::Engine;
use pgai_mon::auth::{Auth, AuthCfg};

fn auth(token: Option<&str>, basic: Option<(&str, &str)>) -> Auth {
    let mut env = vec![];
    if let Some(t) = token {
        env.push(("PGAI_MON_AUTH_TOKEN".to_string(), t.to_string()));
    }
    if let Some((u, p)) = basic {
        env.push(("PGAI_MON_AUTH_USERNAME".to_string(), u.to_string()));
        env.push(("PGAI_MON_AUTH_PASSWORD".to_string(), p.to_string()));
    }
    Auth::resolve(&AuthCfg::default(), &move |k| {
        env.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())
    })
    .unwrap()
}

const TOKEN: &str = "test-token-0123456789";
const PW: &str = "test-password-0123456789";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_route_but_healthz_needs_credentials() {
    let d = tempfile::tempdir().unwrap();
    let s = ingest_common::serve(d.path(), auth(Some(TOKEN), Some(("pgai", PW))), |_, _| {
        axum::Router::new()
    })
    .await;
    let c = reqwest::Client::new();
    let basic = format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode(format!("pgai:{PW}"))
    );
    let wrong = format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode("pgai:nope-nope-nope-nope")
    );

    let r = c.get(format!("{}/healthz", s.base)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.text().await.unwrap(), "ok\n");

    for (m, path) in [
        ("GET", "/health"),
        ("GET", "/schema"),
        ("POST", "/sql"),
        ("GET", "/nope"),
    ] {
        let req = |h: Option<&str>| {
            let mut b = c
                .request(m.parse().unwrap(), format!("{}{path}", s.base))
                .body("select 1");
            if let Some(h) = h {
                b = b.header("authorization", h);
            }
            b.send()
        };
        let r = req(None).await.unwrap();
        assert_eq!(r.status(), 401, "{m} {path} without credentials");
        assert_eq!(r.headers()["www-authenticate"], "Basic realm=\"pgai-mon\"");
        assert!(r.text().await.unwrap().contains("unauthorized"));
        assert_eq!(
            req(Some("Bearer wrong-token-0123456789")).await.unwrap().status(),
            401,
            "{path}"
        );
        assert_eq!(req(Some(&wrong)).await.unwrap().status(), 401, "{path}");
        let want = if path == "/nope" { 404 } else { 200 };
        assert_eq!(
            req(Some(&format!("Bearer {TOKEN}"))).await.unwrap().status(),
            want,
            "{m} {path} bearer"
        );
        assert_eq!(req(Some(&basic)).await.unwrap().status(), want, "{m} {path} basic");
    }
    let r = c
        .post(format!("{}/sql", s.base))
        .bearer_auth(TOKEN)
        .body("select 42 as x")
        .send()
        .await
        .unwrap();
    assert!(r.text().await.unwrap().contains("42"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn token_only_advertises_bearer_and_disabled_serves_everything() {
    let d = tempfile::tempdir().unwrap();
    let s = ingest_common::serve(d.path(), auth(Some(TOKEN), None), |_, _| axum::Router::new()).await;
    let c = reqwest::Client::new();
    let r = c.get(format!("{}/health", s.base)).send().await.unwrap();
    assert_eq!(r.status(), 401);
    assert_eq!(r.headers()["www-authenticate"], "Bearer realm=\"pgai-mon\"");
    let basic = format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode(format!("x:{TOKEN}"))
    );
    let r = c
        .get(format!("{}/health", s.base))
        .header("authorization", basic)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401, "basic is not accepted when only a token is configured");

    let d2 = tempfile::tempdir().unwrap();
    let s2 = ingest_common::serve(d2.path(), Auth::disabled(), |_, _| axum::Router::new()).await;
    assert_eq!(c.get(format!("{}/health", s2.base)).send().await.unwrap().status(), 200);
}

#[test]
fn binary_refuses_to_serve_without_credentials() {
    let d = tempfile::tempdir().unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_pgai-mon"))
        .args([
            "serve",
            "--store",
            d.path().to_str().unwrap(),
            "--listen",
            "127.0.0.1:0",
        ])
        .env_remove("PGAI_MON_AUTH_TOKEN")
        .env_remove("PGAI_MON_AUTH_TOKEN_FILE")
        .env_remove("PGAI_MON_AUTH_USERNAME")
        .env_remove("PGAI_MON_AUTH_PASSWORD")
        .env_remove("PGAI_MON_AUTH_PASSWORD_FILE")
        .env_remove("PGAI_MON_AUTH_DISABLED")
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("no credentials"), "{err}");
}
