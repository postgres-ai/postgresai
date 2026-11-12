//! The reader's row cache (eval/rowcache.rs, --row-cache-mb) is invisible in results: two real
//! `serve` processes on one store, one with the cache and one without, answer every top-k shape
//! with byte-identical bodies: first and repeated calls, concurrent calls (single flight),
//! after new data changes the catalog generation, with the L1 response cache off and on.
use pgai_mon::config::StoreCfg;
use pgai_mon::model::*;
use pgai_mon::store::StoreCore;
use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

const H: i64 = 3_600_000;

/// r_total and q_total: 24 counters each on the same label sets (i = 0..23), 60 s scrapes with
/// resets, a NaN and staleness markers, over three closed days (compacted: rollups) and the
/// six hours before `now` (raw).
fn sample(core: &mut StoreCore, from: i64, to: i64, salt: u64) {
    let mut seed = 7 + salt;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut defs = vec![];
    let mut samples = vec![];
    for (m, mul) in [("r_total", 1.0), ("q_total", 3.0)] {
        for s in 0..24u64 {
            let labels = vec![("i".to_string(), s.to_string())];
            let id = series_id(m, &labels);
            defs.push(SeriesDef {
                series_id: id,
                metric: m.into(),
                labels,
                first_seen: from,
            });
            let mut t = from + (rnd() % 60_000) as i64;
            let mut v = (rnd() % 100) as f64;
            let mut k = 0u64;
            while t < to {
                v += mul * (rnd() % 20) as f64;
                if k % 900 == 400 {
                    v = 1.0;
                }
                let value = match k % 997 {
                    50 => stale_nan(),
                    60 => f64::NAN,
                    _ => v,
                };
                samples.push(Sample {
                    series_id: id,
                    ts: t,
                    value,
                });
                t += 60_000 + (rnd() % 200) as i64 - 100;
                k += 1;
            }
        }
    }
    core.append(Record::Series(defs)).unwrap();
    for c in samples.chunks(50_000) {
        core.append(Record::Samples(c.to_vec())).unwrap();
        core.flush().unwrap();
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

struct Serve {
    child: Child,
    port: u16,
}

impl Drop for Serve {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn serve(store: &std::path::Path, row_cache_mb: u64, l1: bool) -> Serve {
    let port = free_port();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_pgai-mon"));
    cmd.args([
        "serve",
        "--store",
        store.to_str().unwrap(),
        "--no-compact",
        "--no-auth",
        "--listen",
    ])
    .arg(format!("127.0.0.1:{port}"))
    .args(["--row-cache-mb", &row_cache_mb.to_string(), "--max-concurrent", "4"])
    .stdout(Stdio::null())
    .stderr(Stdio::null());
    if !l1 {
        cmd.env("PGAI_QCACHE", "0");
    }
    let s = Serve {
        child: cmd.spawn().unwrap(),
        port,
    };
    for _ in 0..200 {
        if http(port, "GET", "/health", "").is_some() {
            return s;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("serve on {port} did not come up");
}

/// Minimal HTTP/1.1 client: the body, or None when the connection fails.
fn http(port: u16, method: &str, path: &str, body: &str) -> Option<String> {
    let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    write!(
        c,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .ok()?;
    let mut raw = vec![];
    c.read_to_end(&mut raw).ok()?;
    let raw = String::from_utf8(raw).ok()?;
    let (head, rest) = raw.split_once("\r\n\r\n")?;
    if !head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        return Some(rest.to_string());
    }
    let (mut out, mut r) = (String::new(), rest);
    loop {
        let (n, tail) = r.split_once("\r\n")?;
        let n = usize::from_str_radix(n.trim(), 16).ok()?;
        if n == 0 {
            return Some(out);
        }
        out.push_str(&tail[..n]);
        r = &tail[n + 2..];
    }
}

/// The rows of a query (series ids as text: exact), panicking on an error body.
fn rows(port: u16, sql: &str) -> String {
    let b = http(port, "POST", "/sql", sql).unwrap_or_else(|| panic!("no reply on {port}"));
    let v: serde_json::Value = serde_json::from_str(&b).unwrap_or_else(|_| panic!("{sql}: {b}"));
    assert!(v.get("error").is_none(), "{sql}: {b}");
    let r = v["rows"].to_string();
    assert!(r.len() > 2, "{sql}: no rows");
    r
}

fn row_cache(port: u16) -> serde_json::Value {
    let h: serde_json::Value = serde_json::from_str(&http(port, "GET", "/health", "").unwrap()).unwrap();
    // the reader's counters, wherever /health nests them
    fn find(v: &serde_json::Value) -> Option<serde_json::Value> {
        if let Some(x) = v.get("row_cache") {
            return Some(x.clone());
        }
        v.as_object()?.values().find_map(find)
    }
    find(&h).unwrap_or(serde_json::Value::Null)
}

/// A store with closed days (rollups) and recent raw hours, and the top-k shapes to compare:
/// plain, bottom-k, matchers, den, rate, last, last_over_time, expressions, sum by; each on a
/// rollup-served hourly grid and on a raw 1 m grid.
fn build(store: &std::path::Path) -> (StoreCore, Vec<String>, i64) {
    let mut core = StoreCore::open(
        store,
        &StoreCfg {
            delete_grace_s: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let now = now_ms();
    let day0 = now / 86_400_000 * 86_400_000 - 4 * 86_400_000;
    sample(&mut core, day0, day0 + 3 * 86_400_000, 0);
    core.compactor().unwrap().run_once(now, true).unwrap();
    sample(&mut core, now - 6 * H, now - 30 * 60_000, 1);
    let (cs, ce) = (day0 + 5 * H, day0 + 70 * H);
    let (rs, re) = ((now - 6 * H) / 60_000 * 60_000, (now - 20 * 60_000) / 60_000 * 60_000);
    let terms = "[{'m': 'r_total', 'n': 1.0, 'd': 0.0}, {'m': 'q_total', 'n': -1.0, 'd': 0.0}]";
    let ratio = "[{'m': 'r_total', 'n': 1.0, 'd': 1.0}, {'m': 'q_total', 'n': 0.0, 'd': 1.0}]";
    let q = |call: String| format!("SELECT CAST(series_id AS VARCHAR) AS s, t, value FROM {call} ORDER BY 1, 2");
    let mut shapes = vec![];
    for (s, e, step, w) in [(cs, ce, H, H + 5_000), (rs, re, 60_000, 65_000)] {
        shapes.extend([
            q(format!("prom_topk(5, 'irate', 'r_total', [], {s}, {e}, {step}, {w})")),
            q(format!("prom_topk(5, 'irate', 'q_total', [], {s}, {e}, {step}, {w})")),
            q(format!(
                "prom_topk(5, 'irate', 'r_total', [], {s}, {e}, {step}, {w}, p_den_metric := 'q_total')"
            )),
            q(format!(
                "prom_topk(3, 'irate', 'r_total', [], {s}, {e}, {step}, {w}, p_bottom := true)"
            )),
            q(format!(
                "prom_topk(5, 'irate', 'r_total', [['i', '=~', '1.*']], {s}, {e}, {step}, {w})"
            )),
            q(format!("prom_topk(4, 'rate', 'r_total', [], {s}, {e}, {step}, {step})")),
            q(format!("prom_topk(3, 'last', 'r_total', [], {s}, {e}, {step}, 0)")),
            q(format!(
                "prom_topk(3, 'last_over_time', 'q_total', [], {s}, {e}, {step}, {})",
                6 * H
            )),
            q(format!(
                "prom_topk_expr(3, 'irate', {terms}, [], {s}, {e}, {step}, {w})"
            )),
            q(format!(
                "prom_topk_expr(3, 'irate', {ratio}, [], {s}, {e}, {step}, {w})"
            )),
        ]);
        shapes.push(format!(
            "SELECT labels['i'] AS g, t, value FROM prom_topk(3, 'irate', 'r_total', [], {s}, {e}, {step}, {w}, p_by := ['i']) ORDER BY 2, 3, 1"
        ));
    }
    (core, shapes, now)
}

/// Two serves, with and without the row cache, L1 off and on: identical bodies on first,
/// repeated and concurrent calls (single flight), and the repeats are row-cache hits.
#[test]
fn row_cache_serve_results_are_identical_to_uncached() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("s");
    let (core, shapes, _) = build(&store);
    drop(core);
    for l1 in [false, true] {
        let cached = serve(&store, 24, l1);
        let plain = serve(&store, 0, l1);
        let mut want = vec![];
        for sql in &shapes {
            let w = rows(plain.port, sql);
            for pass in 0..2 {
                assert_eq!(rows(cached.port, sql), w, "l1={l1} pass {pass}: {sql}");
            }
            want.push(w);
        }
        std::thread::scope(|sc| {
            let hs: Vec<_> = (0..2)
                .flat_map(|_| shapes.iter().enumerate())
                .map(|(i, sql)| sc.spawn(move || (i, sql, rows(cached.port, sql))))
                .collect();
            for h in hs {
                let (i, sql, got) = h.join().unwrap();
                assert_eq!(got, want[i], "l1={l1} concurrent: {sql}");
            }
        });
        let rc = row_cache(cached.port);
        if !l1 {
            // with L1 off the repeats reach the reader: the row cache served them
            assert!(rc["hits"].as_u64().unwrap_or(0) > 0, "no row-cache hits: {rc}");
            assert!(rc["bytes"].as_u64().unwrap_or(0) > 0, "{rc}");
        }
        assert_eq!(
            row_cache(plain.port)["hits"].as_u64().unwrap_or(0),
            0,
            "cache off but hits"
        );
    }
}

/// Writer and reader in one process (as in serve): after new samples (a new catalog
/// generation) the cached rows are not reused, and every shape still equals the uncached run.
#[test]
fn row_cache_is_not_reused_across_catalog_generations() {
    use pgai_mon::config::HttpCfg;
    use pgai_mon::eval::rowcache;
    use pgai_mon::query::{Reader, EMBEDDED_VIEWS_SQL};
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("s");
    let (mut core, shapes, now) = build(&store);
    let r = Reader::new(
        core.shared.clone(),
        HttpCfg {
            memory_limit: "1GB".into(),
            row_cache_mb: 24,
            ..Default::default()
        },
        EMBEDDED_VIEWS_SQL.to_string(),
        vec![],
    )
    .unwrap();
    let run = |sql: &str| -> String {
        let res = r.run(sql).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
        res.rows.iter().map(|x| x.get()).collect::<Vec<_>>().join("\n")
    };
    let uncached = |sql: &str| {
        rowcache::set_cap(0);
        let x = run(sql);
        rowcache::set_cap(24 << 20);
        x
    };
    let pass = |tag: &str| -> Vec<String> {
        shapes
            .iter()
            .map(|sql| {
                let a = run(sql);
                let b = run(sql); // from the cache
                let u = uncached(sql);
                assert_eq!(a, u, "{tag}: {sql}");
                assert_eq!(b, u, "{tag} (cached): {sql}");
                u
            })
            .collect()
    };
    let h0 = rowcache::HITS.load(std::sync::atomic::Ordering::Relaxed);
    let before = pass("first");
    assert!(
        rowcache::HITS.load(std::sync::atomic::Ordering::Relaxed) > h0,
        "no hits"
    );
    let gen0 = core.shared.generation.load(std::sync::atomic::Ordering::Acquire);
    sample(&mut core, now - 6 * H + 17_000, now - 30 * 60_000, 9);
    assert!(core.shared.generation.load(std::sync::atomic::Ordering::Acquire) > gen0);
    let after = pass("after new data");
    let changed = before.iter().zip(&after).filter(|(a, b)| a != b).count();
    assert!(
        changed > 0,
        "the new samples changed no result: the test proves nothing"
    );
}
