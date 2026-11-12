//! Real crash test: kill -9 the crash-test writer at random points, restart, verify that no
//! acknowledged step is lost and nothing is duplicated. Plus the negative control: losing
//! acknowledged WAL records must fail verification.
use std::process::{Command, Stdio};
use std::time::Duration;

fn verify(bin: &str, store: &str) -> (bool, serde_json::Value) {
    let out = Command::new(bin)
        .args(["crashtest-verify", "--store", store])
        .output()
        .unwrap();
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).unwrap_or_else(|_| panic!("{}", String::from_utf8_lossy(&out.stderr)));
    (out.status.success(), v)
}

#[test]
fn kill9_during_flush_no_gap_no_dup_no_ack_lost() {
    let bin = env!("CARGO_BIN_EXE_pgai-mon");
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("s");
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    for _ in 0..8 {
        let mut child = Command::new(bin)
            .args([
                "crashtest-write",
                "--store",
                store.to_str().unwrap(),
                "--series",
                "200",
                "--flush-every",
                "5",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        std::thread::sleep(Duration::from_millis(150 + seed % 600));
        child.kill().unwrap(); // SIGKILL
        child.wait().unwrap();
    }
    let (ok, v) = verify(bin, store.to_str().unwrap());
    eprintln!("{v}");
    assert!(ok, "{v}");
    assert_eq!(v["acked_lost"].as_bool(), Some(false));
    assert!(v["last_acked_step"].as_f64().unwrap() > 10.0);
}

#[test]
fn losing_acknowledged_wal_records_fails_verification() {
    let bin = env!("CARGO_BIN_EXE_pgai-mon");
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("s");
    let s = store.to_str().unwrap();
    // 100 acknowledged steps, flushed every 40: steps 80..99 exist only in the WAL.
    let st = Command::new(bin)
        .args([
            "crashtest-write",
            "--store",
            s,
            "--series",
            "50",
            "--flush-every",
            "40",
            "--steps",
            "100",
            "--no-final-flush",
        ])
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(st.success());
    let wal = store.join("wal");
    let seg = std::fs::read_dir(&wal)
        .unwrap()
        .map(|e| e.unwrap().path())
        .max()
        .unwrap();
    let len = std::fs::metadata(&seg).unwrap().len();
    assert!(len > 1000, "unflushed records in {}", seg.display());
    // lose the second half of the acknowledged, unflushed records
    std::fs::OpenOptions::new()
        .write(true)
        .open(&seg)
        .unwrap()
        .set_len(len / 2)
        .unwrap();
    let (ok, v) = verify(bin, s);
    eprintln!("{v}");
    assert!(!ok, "verification must fail: {v}");
    assert_eq!(v["acked_lost"].as_bool(), Some(true));
}
