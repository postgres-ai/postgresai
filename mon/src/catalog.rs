//! The file catalog: the only place readers learn which Parquet files exist.
//! Rewritten atomically (tmp + fsync + rename + dir fsync) on every change.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileEntry {
    pub kind: String,
    /// Relative to the store root.
    pub path: String,
    pub day: i64,
    /// 0 = flush/staging file, 1 = hourly merge, 2 = daily compaction.
    pub level: u8,
    pub rows: u64,
    pub bytes: u64,
    pub min_ts: i64,
    pub max_ts: i64,
    pub created_ms: i64,
    /// series files: its series are in the registry (registry.rs); compaction output inherits
    /// it when every input had it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub registered: bool,
    /// rollup files: format version (2 = also the second-to-last sample per bucket, irate)
    #[serde(default, skip_serializing_if = "is_zero")]
    pub ver: u8,
}

fn is_zero(x: &u8) -> bool {
    *x == 0
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Catalog {
    pub generation: u64,
    /// Physical samples layout of every samples file in this store.
    pub samples_layout: String,
    /// Highest WAL segment whose records are fully in cataloged Parquet (-1: none).
    pub flushed_through_seg: i64,
    pub next_file_id: u64,
    pub files: Vec<FileEntry>,
    /// Files replaced by compaction, deleted after the grace time (path, delete_after_ms).
    pub pending_delete: Vec<(String, i64)>,
    /// Log tail positions covered by flushed data.
    pub log_offsets: BTreeMap<String, u64>,
    /// Bumped when sample content changes other than by appending after the closed watermark
    /// (a late sample, a loader commit, a migration): caches below the SQL layer key on it.
    #[serde(default)]
    pub data_epoch: u64,
    /// The closed-watermark lag in force (StoreCfg::closed_lag_ms), written by the writer.
    #[serde(default)]
    pub closed_lag_ms: i64,
    /// On an object store: the writer that owns the store (bucket.rs fencing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub writer: Option<WriterLease>,
    /// On an object store: the leases that a directory other than their own took this store
    /// over from, oldest first (at most `FENCED_KEPT`). A writer that starts again after it was
    /// fenced finds its lease here, and removes it with its takeover.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fenced: Vec<FencedLease>,
    /// The catalog as last written to disk. A failed `save` puts it back, so the catalog in
    /// memory never names files, deletions or a flush position that the file does not have:
    /// otherwise the next successful save would persist a half-applied change (a flush retried
    /// after a failed save would be recorded twice).
    #[serde(skip)]
    pub(crate) saved: Option<Box<Catalog>>,
}

/// The owner of an object-store store: one writer per prefix. A writer takes it over by a
/// conditional PUT of catalog.json with `epoch + 1`; every later commit is conditional on the
/// catalog it wrote last, so an older owner's next commit fails.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct WriterLease {
    pub id: String,
    pub epoch: u64,
    pub host: String,
    /// Time of this writer's last commit, ms since the epoch.
    pub heartbeat_ms: i64,
    /// Set by the writer's last commit of a clean stop: the next writer need not wait the lease out.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub released: bool,
}

/// A lease another directory took the store over from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FencedLease {
    #[serde(flatten)]
    pub writer: WriterLease,
    /// `flushed_through_seg` of the catalog that had the lease: the WAL numbering of that
    /// writer's directory.
    pub flushed_through_seg: i64,
}

/// Largest catalog.json read from an object store (it is re-read every catalog TTL).
pub const MAX_REMOTE_BYTES: u64 = 64 << 20;

/// Entries kept in `Catalog::fenced`.
pub const FENCED_KEPT: usize = 64;

/// A catalog path that stays under the store root whatever the root is: relative, `/`-separated,
/// no empty, `.` or `..` segment, and no `\`, `:`, `'` or NUL (so `root.join` never yields an
/// absolute or local path, and the path quotes safely into SQL).
pub fn plain_rel(rel: &str) -> bool {
    !rel.is_empty()
        && rel
            .split('/')
            .all(|c| !c.is_empty() && c != "." && c != ".." && !c.contains(['\\', ':', '\'', '\0']))
}

/// Parse a catalog read from an object store: every file path must be `plain_rel`, or the
/// whole catalog is refused (a tampered catalog must not point readers at local files).
pub fn parse_remote(bytes: &[u8]) -> anyhow::Result<Catalog> {
    let c: Catalog = serde_json::from_slice(bytes)?;
    if let Some(f) = c.files.iter().find(|f| !plain_rel(&f.path)) {
        anyhow::bail!(
            "catalog.json refused: file path {:?} is not a plain relative path",
            f.path
        );
    }
    Ok(c)
}

/// Read and check a remote store's catalog.json (bounded by `MAX_REMOTE_BYTES`).
pub fn load_remote(root: &Path) -> anyhow::Result<Catalog> {
    parse_remote(&crate::objstore::read_capped(&path(root), MAX_REMOTE_BYTES)?)
}

pub fn path(root: &Path) -> PathBuf {
    root.join("catalog.json")
}

impl Catalog {
    pub fn load_or_new(root: &Path, layout: &str) -> anyhow::Result<Self> {
        if crate::objstore::is_remote(root) {
            // a remote store is read-only: no catalog means no store
            return load_remote(root);
        }
        let p = path(root);
        if p.exists() {
            let mut c: Catalog = serde_json::from_slice(&std::fs::read(&p)?)?;
            c.saved = Some(Box::new(c.clone()));
            Ok(c)
        } else {
            let mut c = Catalog {
                samples_layout: layout.to_string(),
                flushed_through_seg: -1,
                ..Default::default()
            };
            c.saved = Some(Box::new(c.clone()));
            Ok(c)
        }
    }

    /// Write the catalog (temp file, fsync, rename). On failure the catalog in memory goes back
    /// to what the last successful save wrote.
    pub fn save(&mut self, root: &Path) -> anyhow::Result<()> {
        let before = self.saved.take();
        match self.write(root) {
            Ok(()) => {
                self.saved = Some(Box::new(self.clone()));
                Ok(())
            }
            Err(e) => {
                if let Some(b) = before {
                    // file ids taken since the last save may name files being written: never
                    // hand them out again
                    let next = self.next_file_id;
                    *self = (*b).clone();
                    self.next_file_id = self.next_file_id.max(next);
                    self.saved = Some(b);
                }
                Err(e)
            }
        }
    }

    fn write(&mut self, root: &Path) -> anyhow::Result<()> {
        use std::io::Write;
        self.generation += 1;
        let tmp = root.join("catalog.json.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&serde_json::to_vec(self)?)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path(root))?;
        if let Ok(d) = std::fs::File::open(root) {
            let _ = d.sync_all();
        }
        Ok(())
    }

    pub fn files_of(&self, kind: &str) -> impl Iterator<Item = &FileEntry> {
        let k = kind.to_string();
        self.files.iter().filter(move |f| f.kind == k)
    }

    pub fn alloc_id(&mut self) -> u64 {
        self.next_file_id += 1;
        self.next_file_id
    }
}
