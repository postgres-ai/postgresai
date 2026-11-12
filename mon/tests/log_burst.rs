//! A 200k-line jsonlog burst (real lines, ~1/3 auto_explain) ingests within the writer
//! memory limit: chunked reads, size-capped flushes, chunked DuckDB writes.
use pgai_mon::config::StoreCfg;
use pgai_mon::store::{file_list_sql, StoreCore};
use std::io::Write;

const FIXTURE: &str = include_str!("fixtures/pglog_sample.json");

#[test]
fn burst_200k_lines_within_memory_limit() {
    let dir = tempfile::tempdir().unwrap();
    let logdir = dir.path().join("pglog");
    std::fs::create_dir_all(&logdir).unwrap();
    let base: Vec<serde_json::Value> = FIXTURE.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let n = 200_000usize;
    let mut plans_expected = 0;
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(logdir.join("postgresql-burst.json")).unwrap());
        for i in 0..n {
            let mut v = base[i % base.len()].clone();
            if v["message"].as_str().unwrap().contains("plan:") {
                plans_expected += 1;
            }
            // one line every 400 ms from 2026-11-01 00:00:00 (crosses a day boundary)
            let ms = i as i64 * 400;
            let (s, m) = (ms / 1000, ms % 1000);
            v["timestamp"] = serde_json::Value::String(format!(
                "2026-11-{:02} {:02}:{:02}:{:02}.{:03} UTC",
                1 + s / 86400,
                (s / 3600) % 24,
                (s / 60) % 60,
                s % 60,
                m
            ));
            writeln!(f, "{v}").unwrap();
        }
    }
    let bytes = std::fs::metadata(logdir.join("postgresql-burst.json")).unwrap().len();
    assert!(bytes > 400 << 20 || bytes > 100 << 20, "burst is {bytes} bytes");
    // Writer limit well below the burst size; flush cap 16 MB.
    let cfg = StoreCfg {
        writer_memory_limit: "48MB".into(),
        max_buffer_bytes: 16 << 20,
        ..Default::default()
    };
    let mut core = StoreCore::open(dir.path().join("s").as_path(), &cfg).unwrap();
    let st = pgai_mon::tools::load_logs(&mut core, &logdir, ".json", true).unwrap();
    assert_eq!(st.lines, n as u64);
    assert_eq!(st.plans, plans_expected as u64);
    assert!(st.flushes > 3, "several size-capped flushes, got {}", st.flushes);
    let cat = core.shared.catalog.lock().unwrap().clone();
    let files: Vec<_> = cat.files_of("logs").collect();
    let (rows, ev): (i64, i64) = core
        .conn()
        .query_row(
            &format!(
                "SELECT (SELECT count(*) FROM read_parquet({})), (SELECT count(*) FROM read_parquet({}))",
                file_list_sql(&core.shared.root, &files),
                file_list_sql(&core.shared.root, &cat.files_of("plan_events").collect::<Vec<_>>())
            ),
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(rows, n as i64);
    assert_eq!(ev, plans_expected as i64);
    // rerun: resumes from the cataloged offset, imports nothing twice
    drop(core);
    let mut core = StoreCore::open(dir.path().join("s").as_path(), &cfg).unwrap();
    let st2 = pgai_mon::tools::load_logs(&mut core, &logdir, ".json", false).unwrap();
    assert_eq!(st2.lines, 0);
}
