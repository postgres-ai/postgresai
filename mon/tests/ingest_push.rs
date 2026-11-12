//! Push ingest (`POST /api/v1/import/prometheus`) and the in-process ingest API, end to end:
//! HTTP -> WAL -> Parquet -> SQL.
mod ingest_common;
use pgai_mon::auth::{Auth, AuthCfg};
use pgai_mon::ingest::{parse_text, ExtSample, Ingest, IngestCfg};
use std::sync::Arc;

const PW: &str = "push-password-0123456789";

fn basic_auth() -> Auth {
    Auth::resolve(&AuthCfg::default(), &|k| match k {
        "PGAI_MON_AUTH_USERNAME" => Some("pgai".into()),
        "PGAI_MON_AUTH_PASSWORD" => Some(PW.into()),
        _ => None,
    })
    .unwrap()
}

/// Recorded samples are from 2026: accept any age in these tests unless a test sets its own.
fn any_age() -> IngestCfg {
    IngestCfg { max_age_s: i64::MAX / 4000, ..Default::default() }
}

async fn server(dir: &std::path::Path, cfg: IngestCfg) -> (ingest_common::Server, Arc<Ingest>) {
    let slot = Arc::new(std::sync::Mutex::new(None));
    let s2 = slot.clone();
    let s = ingest_common::serve(dir, basic_auth(), move |tx, shared| {
        let ing = Ingest::new(shared, tx, cfg);
        *s2.lock().unwrap() = Some(ing.clone());
        pgai_mon::ingest::router(ing)
    })
    .await;
    let ing = slot.lock().unwrap().take().unwrap();
    (s, ing)
}

async fn push(s: &ingest_common::Server, query: &str, body: String) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{}/api/v1/import/prometheus{query}", s.base))
        .basic_auth("pgai", Some(PW))
        .header("content-type", "text/plain")
        .body(body)
        .send()
        .await
        .unwrap()
}

/// (metric, labels as k=v joined by ',', ts ms, value) of every stored non-stale sample.
fn stored(s: &ingest_common::Server) -> Vec<(String, String, i64, f64)> {
    s.sql(
        "SELECT r.metric, array_to_string(list_transform(list_sort(map_entries(r.labels)), e -> e.key || '=' || e.value), ','),
                epoch_ms(s.ts), CAST(s.value AS VARCHAR)
         FROM samples s JOIN series r USING (series_id) WHERE NOT s.stale ORDER BY 1, 2, 3",
    )
    .into_iter()
    .map(|r| {
        (
            r[0].as_str().unwrap().to_string(),
            r[1].as_str().unwrap().to_string(),
            r[2].as_i64().unwrap(),
            // DOUBLE -> VARCHAR is exact (shortest round trip); the JSON number is not always
            r[3].as_str().unwrap().parse::<f64>().unwrap(),
        )
    })
    .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rds_host_stats_import_bodies_round_trip_exactly() {
    for case in ["rds-postgres", "rds-postgres-no-em", "aurora-postgresql"] {
        let d = tempfile::tempdir().unwrap();
        let (s, ing) = server(d.path(), any_age()).await;
        let body =
            std::fs::read_to_string(format!("{}/tests/fixtures/rds-host-stats/{case}/expected.prom", env!("CARGO_MANIFEST_DIR")))
                .unwrap();
        let r = push(&s, "", body.clone()).await;
        assert_eq!(r.status(), 204, "{case}: {}", r.text().await.unwrap());
        s.flush().await;
        let mut want: Vec<(String, String, i64, f64)> = parse_text(&body)
            .unwrap()
            .into_iter()
            .map(|l| {
                let mut lab = l.labels;
                lab.sort();
                let lab = lab.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(",");
                (l.metric, lab, l.ts.unwrap(), l.value)
            })
            .collect();
        want.sort_by(|a, b| (&a.0, &a.1, a.2).cmp(&(&b.0, &b.1, b.2)));
        let got = stored(&s);
        assert_eq!(got.len(), want.len(), "{case}");
        for (g, w) in got.iter().zip(&want) {
            assert_eq!((&g.0, &g.1, g.2), (&w.0, &w.1, w.2), "{case}");
            assert_eq!(g.3.to_bits(), w.3.to_bits(), "{case} {g:?} vs {w:?}");
        }
        // one series per (metric, labels), registered once
        let n_series = s.sql("SELECT count(*), count(DISTINCT series_id) FROM series");
        let distinct: std::collections::HashSet<_> = want.iter().map(|w| (&w.0, &w.1)).collect();
        assert_eq!(n_series[0][0].as_u64().unwrap() as usize, distinct.len(), "{case}");
        assert_eq!(n_series[0][1], n_series[0][0]);
        assert_eq!(ing.stats.samples.load(std::sync::atomic::Ordering::Relaxed) as usize, want.len());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_bad_line_rejects_the_whole_request() {
    let d = tempfile::tempdir().unwrap();
    let (s, ing) = server(d.path(), IngestCfg::default()).await;
    let now = pgai_mon::model::now_ms();
    for body in [
        format!("good{{a=\"1\"}} 1 {now}\nbad{{a=\"1\" 2 {now}\n"),
        format!("good 1 {now}\nbad{{a=\"1\",a=\"2\"}} 2 {now}\n"),
        format!("good 1 {now}\nbad{{1a=\"x\"}} 2 {now}\n"),
        format!("good 1 {now}\nbad 1 {now}.5\n"),
    ] {
        let r = push(&s, "", body).await;
        assert_eq!(r.status(), 400);
        let t = r.text().await.unwrap();
        assert!(t.contains("invalid_input"), "{t}");
    }
    s.flush().await;
    assert!(stored(&s).is_empty());
    assert_eq!(ing.stats.rejected_requests.load(std::sync::atomic::Ordering::Relaxed), 4);
    // gzip is not accepted (VictoriaMetrics pushers in this repo send plain text)
    let r = reqwest::Client::new()
        .post(format!("{}/api/v1/import/prometheus", s.base))
        .basic_auth("pgai", Some(PW))
        .header("content-encoding", "gzip")
        .body("x 1")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 415);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn push_needs_credentials_and_honours_extra_labels_and_receive_time() {
    let d = tempfile::tempdir().unwrap();
    let (s, _) = server(d.path(), IngestCfg::default()).await;
    let r = reqwest::Client::new()
        .post(format!("{}/api/v1/import/prometheus", s.base))
        .body("x 1")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);
    let t0 = pgai_mon::model::now_ms();
    let r = push(&s, "?extra_label=job%3Drds&extra_label=cluster=c2", "x{cluster=\"c1\",a=\"\"} 7\n".into()).await;
    assert_eq!(r.status(), 204);
    s.flush().await;
    let got = stored(&s);
    assert_eq!(got.len(), 1);
    // the extra label replaces the line's own; an empty label value is no label
    assert_eq!((got[0].0.as_str(), got[0].1.as_str(), got[0].3), ("x", "cluster=c2,job=rds", 7.0));
    assert!(got[0].2 >= t0 && got[0].2 <= pgai_mon::model::now_ms(), "no timestamp: receive time");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn limits_bound_age_cardinality_and_body_size() {
    let d = tempfile::tempdir().unwrap();
    let cfg = IngestCfg { max_series: 3, max_body_bytes: 4096, max_age_s: 3600, ..Default::default() };
    let (s, ing) = server(d.path(), cfg).await;
    let now = pgai_mon::model::now_ms();
    let body: String = (0..5).map(|i| format!("m{{i=\"{i}\"}} {i} {now}\n")).collect::<String>()
        + &format!("old 1 {}\nfuture 1 {}\n", now - 7_200_000, now + 3_600_000);
    assert_eq!(push(&s, "", body).await.status(), 204);
    // a known series is still accepted at the cap
    assert_eq!(push(&s, "", format!("m{{i=\"0\"}} 10 {}\n", now + 1000)).await.status(), 204);
    s.flush().await;
    let got = stored(&s);
    assert_eq!(got.len(), 4, "{got:?}");
    assert_eq!(ing.tracked_series(), 3);
    let st = ing.stats.json();
    assert_eq!(st["dropped"]["series_limit"], 2);
    assert_eq!(st["dropped"]["too_old"], 1);
    assert_eq!(st["dropped"]["too_new"], 1);
    let big = "x 1\n".repeat(2000);
    assert_eq!(push(&s, "", big).await.status(), 413);
    // the stats route is behind auth too
    let r = reqwest::Client::new().get(format!("{}/ingest/stats", s.base)).send().await.unwrap();
    assert_eq!(r.status(), 401);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn groups_get_staleness_markers_like_collected_series() {
    let d = tempfile::tempdir().unwrap();
    let (s, ing) = server(d.path(), IngestCfg::default()).await;
    let t = pgai_mon::model::now_ms() - 60_000;
    let smp = |m: &str, ts: i64| ExtSample { metric: m.into(), labels: vec![("job".into(), "j".into())], ts, value: 1.0 };
    let w = ing.validate(vec![smp("a", t), smp("b", t)]).unwrap();
    assert_eq!(ing.write(w, Some(("target-1", t))).await.unwrap().stale_markers, 0);
    let w = ing.validate(vec![smp("a", t + 15_000)]).unwrap();
    let rep = ing.write(w, Some(("target-1", t + 15_000))).await.unwrap();
    assert_eq!((rep.samples, rep.stale_markers), (1, 1));
    // another group's series are not touched by target-1's writes
    let w = ing.validate(vec![smp("c", t)]).unwrap();
    ing.write(w, Some(("target-2", t))).await.unwrap();
    assert_eq!(ing.end_group("target-1", t + 30_000).await.unwrap(), 1);
    assert_eq!(ing.end_group("target-1", t + 45_000).await.unwrap(), 0);
    s.flush().await;
    let rows = s.sql(
        "SELECT r.metric, epoch_ms(s.ts) - (SELECT min(epoch_ms(ts)) FROM samples), s.stale, isnan(s.value)
         FROM samples s JOIN series r USING (series_id) ORDER BY 1, 2",
    );
    let rows: Vec<(String, i64, bool, bool)> = rows
        .iter()
        .map(|r| (r[0].as_str().unwrap().into(), r[1].as_i64().unwrap(), r[2].as_bool().unwrap(), r[3].as_bool().unwrap()))
        .collect();
    assert_eq!(
        rows,
        vec![
            ("a".into(), 0, false, false),
            ("a".into(), 15_000, false, false),
            ("a".into(), 30_000, true, true),
            ("b".into(), 0, false, false),
            ("b".into(), 15_000, true, true),
            ("c".into(), 0, false, false),
        ]
    );
    // a real NaN pushed over HTTP is a value, not a staleness marker
    assert_eq!(push(&s, "", format!("n NaN {t}\n")).await.status(), 204);
    s.flush().await;
    let r = s.sql("SELECT s.stale FROM samples s JOIN series r USING (series_id) WHERE r.metric = 'n'");
    assert_eq!(r[0][0], false);
}
