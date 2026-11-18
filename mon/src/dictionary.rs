//! The collector publishes one atomic dictionary snapshot for both reader modes.
use crate::query::DictEntry;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dictionary {
    pub version: u64,
    pub entries: Vec<DictEntry>,
}

impl Dictionary {
    pub fn new(entries: Vec<DictEntry>) -> Self {
        // Content versions also survive a reader child restart.
        let version = xxhash_rust::xxh3::xxh3_64(&serde_json::to_vec(&entries).unwrap());
        Self { version, entries }
    }

    pub fn load(scratch: &Path) -> anyhow::Result<Option<Self>> {
        match std::fs::read(scratch.join("metric_dictionary.json")) {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn publish(&self, scratch: &Path) -> anyhow::Result<()> {
        if Self::load(scratch)?.is_some_and(|d| d.version == self.version) {
            return Ok(());
        }
        std::fs::create_dir_all(scratch)?;
        let tmp = scratch.join("metric_dictionary.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(self)?)?;
        std::fs::rename(tmp, scratch.join("metric_dictionary.json"))?;
        Ok(())
    }
}
