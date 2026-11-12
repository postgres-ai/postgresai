//! Append-only write-ahead log in numbered segments.
//! Frame: len:u32 (of tag+payload) | crc32:u32 (of tag+payload) | tag:u8 | payload.
//! Every append is fsync'ed before the record enters the in-memory buffer, so a crash
//! loses nothing that was acknowledged. A torn tail (short frame or bad CRC) ends replay.
use crate::model::Record;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub struct Wal {
    dir: PathBuf,
    seg: u64,
    file: File,
    pub bytes: u64,
    scratch: Vec<u8>,
}

pub fn seg_path(dir: &Path, seg: u64) -> PathBuf {
    dir.join(format!("{seg:020}.wal"))
}

/// Segments present on disk, ascending.
pub fn list_segments(dir: &Path) -> anyhow::Result<Vec<u64>> {
    let mut v = Vec::new();
    if dir.exists() {
        for e in std::fs::read_dir(dir)? {
            let name = e?.file_name().to_string_lossy().to_string();
            if let Some(n) = name.strip_suffix(".wal") {
                if let Ok(s) = n.parse::<u64>() {
                    v.push(s);
                }
            }
        }
    }
    v.sort();
    Ok(v)
}

/// Read every intact record of a segment. Stops at the first torn/corrupt frame and
/// returns the byte length of the valid prefix.
pub fn read_segment(path: &Path) -> anyhow::Result<(Vec<(Record, usize)>, u64)> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    let mut out = Vec::new();
    let mut o = 0usize;
    while o + 8 <= buf.len() {
        let len = u32::from_le_bytes(buf[o..o + 4].try_into()?) as usize;
        let crc = u32::from_le_bytes(buf[o + 4..o + 8].try_into()?);
        if len == 0 || o + 8 + len > buf.len() {
            break;
        }
        let body = &buf[o + 8..o + 8 + len];
        if crc32fast::hash(body) != crc {
            break;
        }
        match Record::decode(body[0], &body[1..]) {
            Ok(r) => out.push((r, 8 + len)),
            Err(_) => break,
        }
        o += 8 + len;
    }
    Ok((out, o as u64))
}

fn sync_dir(dir: &Path) {
    if let Ok(d) = File::open(dir) {
        let _ = d.sync_all();
    }
}

impl Wal {
    pub fn open(dir: &Path, seg: u64) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = seg_path(dir, seg);
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let bytes = file.metadata()?.len();
        sync_dir(dir);
        Ok(Self {
            dir: dir.to_path_buf(),
            seg,
            file,
            bytes,
            scratch: Vec::new(),
        })
    }

    pub fn seg(&self) -> u64 {
        self.seg
    }

    /// Append and fsync. Returns after the record is durable.
    pub fn append(&mut self, rec: &Record) -> anyhow::Result<usize> {
        self.scratch.clear();
        self.scratch.extend_from_slice(&[0u8; 8]);
        self.scratch.push(rec.tag());
        rec.encode(&mut self.scratch);
        let body_len = self.scratch.len() - 8;
        let crc = crc32fast::hash(&self.scratch[8..]);
        self.scratch[0..4].copy_from_slice(&(body_len as u32).to_le_bytes());
        self.scratch[4..8].copy_from_slice(&crc.to_le_bytes());
        self.file.write_all(&self.scratch)?;
        self.file.sync_data()?;
        self.bytes += self.scratch.len() as u64;
        Ok(self.scratch.len())
    }

    /// Start a new segment; returns the number of the one just closed.
    pub fn rotate(&mut self) -> anyhow::Result<u64> {
        let old = self.seg;
        self.file.sync_all()?;
        *self = Wal::open(&self.dir, old + 1)?;
        Ok(old)
    }

    pub fn remove_segment(&self, seg: u64) {
        let _ = std::fs::remove_file(seg_path(&self.dir, seg));
        sync_dir(&self.dir);
    }

    /// Truncate a segment to its valid prefix (after a torn tail).
    pub fn truncate_segment(dir: &Path, seg: u64, len: u64) -> anyhow::Result<()> {
        let f = OpenOptions::new().write(true).open(seg_path(dir, seg))?;
        f.set_len(len)?;
        f.sync_all()?;
        Ok(())
    }
}
