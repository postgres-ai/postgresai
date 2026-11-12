//! The image build copies only what mon/Dockerfile names: every file the crate includes at
//! compile time from outside src/ must be copied there and let through by the ignore file.
use std::path::Path;

fn includes(dir: &Path, out: &mut Vec<String>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            includes(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            for part in std::fs::read_to_string(&p).unwrap().split("include_str!(\"").skip(1) {
                let rel = part.split('"').next().unwrap();
                if let Some(outside) = rel.strip_prefix("../") {
                    out.push(outside.to_string());
                }
            }
        }
    }
}

#[test]
fn the_image_build_copies_every_included_file() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = vec![];
    includes(&root.join("src"), &mut files);
    assert!(!files.is_empty());
    let docker = std::fs::read_to_string(root.join("Dockerfile")).unwrap();
    let ignore = std::fs::read_to_string(root.join("Dockerfile.dockerignore")).unwrap();
    for f in files {
        let dir = f.split('/').next().unwrap();
        let covered = |text: &str, prefix: &str| {
            text.lines().any(|l| {
                l.starts_with(prefix)
                    && (l.contains(&format!("mon/{f}"))
                        || l.trim_end().ends_with(&format!("mon/{dir}"))
                        || l.contains(&format!("mon/{dir} ")))
            })
        };
        assert!(covered(&docker, "COPY "), "mon/Dockerfile does not copy mon/{f}");
        assert!(covered(&ignore, "!"), "mon/Dockerfile.dockerignore excludes mon/{f}");
    }
}

/// The config baked into the image keeps the old stack's retention: 14 days of metrics, 30 days
/// of query texts.
#[test]
fn the_image_config_sets_the_retention() {
    let yml = Path::new(env!("CARGO_MANIFEST_DIR")).join("docker/pgai-mon.yml");
    let cfg: pgai_mon::config::Config = serde_yaml::from_str(&std::fs::read_to_string(yml).unwrap()).unwrap();
    assert_eq!((cfg.store.retention_days, cfg.store.texts_retention_days), (14, 30));
}
