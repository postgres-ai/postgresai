//! Read-only opens leave `<tmp>/pgai-mon-<pid>` scratch dirs (reader spill, reader spec): the
//! next open sweeps those of dead processes, and a clean exit removes its own (store.rs).
use pgai_mon::store;
use std::process::Command;

/// A pid that has just exited (reaped, so `kill(pid, 0)` reports ESRCH).
fn dead_pid() -> u32 {
    let mut c = Command::new("true").spawn().unwrap();
    let pid = c.id();
    c.wait().unwrap();
    pid
}

#[test]
fn sweeps_dead_scratch_dirs_only() {
    let tmp = tempfile::tempdir().unwrap();
    let t = tmp.path();
    let mut live = Command::new("sleep").arg("30").spawn().unwrap();
    let dead = dead_pid();
    let mk = |n: String| {
        std::fs::create_dir_all(t.join(&n).join("reader")).unwrap();
        std::fs::write(t.join(&n).join("reader_spec.json"), "{}").unwrap();
        t.join(n)
    };
    let gone = [mk(format!("pgai-mon-{dead}")), mk("pgai-mon-4294967295".into())];
    let kept = [
        mk(format!("pgai-mon-{}", std::process::id())),
        mk(format!("pgai-mon-{}", live.id())),
        mk(format!("pgai-mon-tile-fmt-{dead}")),
        mk("pgai-mon-topk".into()),
        mk("pgai-mon-s3cache".into()),
        mk("pgai-mon-".into()),
        mk(format!("other-{dead}")),
    ];
    let file = t.join(format!("pgai-mon-{}", dead_pid()));
    std::fs::write(&file, "x").unwrap();

    assert_eq!(store::sweep_dead_scratch(t), gone.len());
    live.kill().unwrap();
    live.wait().unwrap();
    for d in &gone {
        assert!(!d.exists(), "{} not swept", d.display());
    }
    for d in &kept {
        assert!(d.exists(), "{} swept", d.display());
    }
    assert!(file.exists(), "a plain file is not a scratch dir");

    // own_scratch_in sweeps too, creates this process's dir, and remove_own_scratch removes it
    let again = mk(format!("pgai-mon-{}", dead_pid()));
    let own = store::own_scratch_in(t).unwrap();
    assert_eq!(own, t.join(format!("pgai-mon-{}", std::process::id())));
    assert!(own.is_dir() && !again.exists());
    store::remove_own_scratch();
    assert!(!own.exists());
}
