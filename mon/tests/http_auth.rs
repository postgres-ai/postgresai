//! HTTP authentication over the real app: every route needs credentials except /livez.
mod ingest_common;
use base64::Engine;
use pgai_mon::auth::{Auth, AuthCfg};

fn auth(token: Option<&str>, basic: Option<(&str, &str)>) -> Auth {
    let mut env = vec![];
    if let Some(t) = token {
        env.push(("PGAI_MON_TOKEN".to_string(), t.to_string()));
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
async fn every_route_but_livez_needs_credentials() {
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

    let r = c.get(format!("{}/livez", s.base)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.text().await.unwrap(), "ok\n");
    // the old name is not a second unauthenticated route
    let r = c.get(format!("{}/healthz", s.base)).send().await.unwrap();
    assert_ne!(r.status(), 200);

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
    let r = c.get(format!("{}/health", s2.base)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let h: serde_json::Value = serde_json::from_str(&r.text().await.unwrap()).unwrap();
    assert_eq!(h["version"], env!("CARGO_PKG_VERSION"));
}

/// The read-only token (the one Grafana holds) queries and cannot write.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_read_only_token_cannot_write() {
    const READ: &str = "test-read-token-0123456789";
    let env = [("PGAI_MON_TOKEN", TOKEN), ("PGAI_MON_READ_TOKEN", READ)];
    let a = Auth::resolve(&AuthCfg::default(), &move |k| {
        env.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
    })
    .unwrap();
    let d = tempfile::tempdir().unwrap();
    let s = ingest_common::serve(d.path(), a, |_, _| {
        axum::Router::new().route("/api/v1/import/prometheus", axum::routing::post(|| async { "written" }))
    })
    .await;
    let c = reqwest::Client::new();
    let post = |path: &'static str, token: &'static str| {
        c.post(format!("{}{path}", s.base))
            .bearer_auth(token)
            .body("select 42 as x")
            .send()
    };
    assert_eq!(post("/sql", READ).await.unwrap().status(), 200);
    let r = post("/api/v1/import/prometheus", READ).await.unwrap();
    assert_eq!(r.status(), 403);
    assert!(r.text().await.unwrap().contains("read-only"));
    assert_eq!(post("/api/v1/import/prometheus", TOKEN).await.unwrap().status(), 200);
    let r = c
        .get(format!("{}/schema", s.base))
        .bearer_auth(READ)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
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
        .env_remove("PGAI_MON_TOKEN")
        .env_remove("PGAI_MON_TOKEN_FILE")
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
