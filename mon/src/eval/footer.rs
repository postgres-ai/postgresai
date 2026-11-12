//! Parquet footer cache for the reader: parsed metadata per file, shared by every query.
//!
//! Without it each evaluator call parsed the footer of every file it touched (samples, series,
//! rollups, the registry), several times per query. Store files are immutable once in the
//! catalog, so an entry stays valid for as long as its file exists; the key includes the file's
//! length and mtime anyway, so a rewritten path can never serve a stale footer. On a catalog
//! generation change the reader drops entries for files that left the catalog (`retain`).
//! Entries are evicted least-recently-used above a byte cap (`set_cap`; 0 disables the cache).
//! A remote object (objstore.rs) is keyed by its path alone: catalog files are never rewritten
//! under the same name, and the entry keeps the object's length, so a hit needs no request.
use crate::objstore::{self, Src};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

struct Entry {
    len: u64,
    mtime: std::time::SystemTime,
    md: ArrowReaderMetadata,
    bytes: usize,
    used: u64,
}

#[derive(Default)]
struct Cache {
    map: HashMap<PathBuf, Entry>,
    bytes: usize,
    tick: u64,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);
static CAP: AtomicUsize = AtomicUsize::new(16 << 20);
pub static HITS: AtomicU64 = AtomicU64::new(0);
pub static MISSES: AtomicU64 = AtomicU64::new(0);

/// Byte cap of the cache (estimated heap size of the parsed metadata).
pub fn set_cap(bytes: usize) {
    CAP.store(bytes, Ordering::Relaxed);
    let mut g = CACHE.lock().unwrap();
    if let Some(c) = g.as_mut() {
        evict(c, bytes);
    }
}

fn evict(c: &mut Cache, cap: usize) {
    while c.bytes > cap && !c.map.is_empty() {
        let k = c
            .map
            .iter()
            .min_by_key(|(_, e)| e.used)
            .map(|(k, _)| k.clone())
            .unwrap();
        if let Some(e) = c.map.remove(&k) {
            c.bytes -= e.bytes;
        }
    }
}

/// Keep only entries whose path satisfies `keep` (called on a catalog generation change).
pub fn retain(keep: impl Fn(&Path) -> bool) {
    let mut g = CACHE.lock().unwrap();
    if let Some(c) = g.as_mut() {
        let mut freed = 0;
        c.map.retain(|p, e| {
            let k = keep(p);
            if !k {
                freed += e.bytes;
            }
            k
        });
        c.bytes -= freed;
    }
}

pub fn stats() -> (usize, usize, u64, u64) {
    let g = CACHE.lock().unwrap();
    let (n, b) = g.as_ref().map_or((0, 0), |c| (c.map.len(), c.bytes));
    (n, b, HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// The file's reader metadata, from the cache or parsed now (local files).
pub fn metadata(path: &Path, file: &File) -> anyhow::Result<ArrowReaderMetadata> {
    let fm = file.metadata()?;
    let (len, mtime) = (fm.len(), fm.modified().unwrap_or(std::time::UNIX_EPOCH));
    if let Some(md) = lookup(path, Some((len, mtime))) {
        return Ok(md);
    }
    let md = ArrowReaderMetadata::load(file, ArrowReaderOptions::new())?;
    insert(path, len, mtime, &md);
    Ok(md)
}

/// A cached entry. `want` = Some((len, mtime)) for a local file (must match); None for a
/// remote object (immutable once in the catalog: the path alone is the key).
fn lookup(path: &Path, want: Option<(u64, std::time::SystemTime)>) -> Option<ArrowReaderMetadata> {
    lookup_len(path, want).map(|x| x.0)
}

fn lookup_len(path: &Path, want: Option<(u64, std::time::SystemTime)>) -> Option<(ArrowReaderMetadata, u64)> {
    if CAP.load(Ordering::Relaxed) == 0 {
        return None;
    }
    let mut g = CACHE.lock().unwrap();
    let c = g.get_or_insert_with(Default::default);
    c.tick += 1;
    let tick = c.tick;
    if let Some(e) = c.map.get_mut(path) {
        if want.is_none_or(|(len, mtime)| e.len == len && e.mtime == mtime) {
            e.used = tick;
            HITS.fetch_add(1, Ordering::Relaxed);
            return Some((e.md.clone(), e.len));
        }
    }
    drop(g);
    MISSES.fetch_add(1, Ordering::Relaxed);
    None
}

fn insert(path: &Path, len: u64, mtime: std::time::SystemTime, md: &ArrowReaderMetadata) {
    if NO_INSERT.with(|c| c.get()) {
        return;
    }
    let cap = CAP.load(Ordering::Relaxed);
    let bytes = md.metadata().memory_size() + path.as_os_str().len() + 64;
    if cap == 0 || bytes > cap / 4 {
        return;
    }
    let mut g = CACHE.lock().unwrap();
    let c = g.get_or_insert_with(Default::default);
    c.tick += 1;
    let used = c.tick;
    if let Some(old) = c.map.insert(
        path.to_path_buf(),
        Entry {
            len,
            mtime,
            md: md.clone(),
            bytes,
            used,
        },
    ) {
        c.bytes -= old.bytes;
    }
    c.bytes += bytes;
    evict(c, cap);
}

/// Parse a freshly opened remote file's metadata from its suffix GET, cache it, and drop the
/// suffix bytes.
fn load_remote(path: &Path, src: &Src) -> anyhow::Result<ArrowReaderMetadata> {
    let md = ArrowReaderMetadata::load(src, ArrowReaderOptions::new())?;
    if let Src::Remote(f) = src {
        f.clear();
        insert(path, f.len(), std::time::UNIX_EPOCH, &md);
    }
    Ok(md)
}

/// Open a file for reading with its (cached) metadata. A remote file on a cache hit costs no
/// request; on a miss, one suffix GET.
pub fn open(path: &Path) -> anyhow::Result<(Src, ArrowReaderMetadata)> {
    if !objstore::is_remote(path) {
        let f = File::open(path)?;
        let md = metadata(path, &f)?;
        return Ok((Src::Local(Arc::new(f)), md));
    }
    if let Some((md, len)) = lookup_len(path, None) {
        return Ok((objstore::open_known(path, len)?, md));
    }
    let src = objstore::open_src(path)?;
    let md = load_remote(path, &src)?;
    Ok((src, md))
}

/// Metadata only (no reader).
pub fn metadata_of(path: &Path) -> anyhow::Result<ArrowReaderMetadata> {
    Ok(open(path)?.1)
}

/// Fetch the footers of these files that are remote and not cached, concurrently (a scan
/// over many day files would otherwise wait one round trip per file).
pub fn warm(paths: &[PathBuf]) {
    if CAP.load(Ordering::Relaxed) == 0 {
        return;
    }
    let cold: Vec<PathBuf> = {
        let g = CACHE.lock().unwrap();
        paths
            .iter()
            .filter(|p| objstore::is_remote(p) && g.as_ref().is_none_or(|c| !c.map.contains_key(*p)))
            .cloned()
            .collect()
    };
    if cold.len() < 2 {
        return;
    }
    for (p, s) in cold.iter().zip(objstore::open_many(&cold)) {
        if let Ok(s) = s {
            let _ = load_remote(p, &s);
        }
    }
}

/// A reader builder over row groups `rgs` of `b` only, holding metadata for just those (rgs
/// become 0..n). A reader keeps its metadata for its whole life, and a scan keeps a reader open
/// per day file: whole footers (1-2 MB parsed for a day file of 200-600 row groups, ~2x that in
/// heap) were most of a 30 d scan's memory (165 MiB of reader RSS for one series).
pub fn builder_rgs(
    b: ParquetRecordBatchReaderBuilder<Src>,
    src: &Src,
    rgs: &[usize],
) -> anyhow::Result<ParquetRecordBatchReaderBuilder<Src>> {
    let full = b.metadata().clone();
    drop(b);
    let sub = parquet::file::metadata::ParquetMetaData::new(
        full.file_metadata().clone(),
        rgs.iter().map(|&i| full.row_group(i).clone()).collect(),
    );
    let md = ArrowReaderMetadata::try_new(Arc::new(sub), ArrowReaderOptions::new())?;
    Ok(ParquetRecordBatchReaderBuilder::new_with_metadata(src.clone(), md).with_row_groups((0..rgs.len()).collect()))
}

/// A reader builder with the footer cached, and a handle to plan its reads (`Src::prefetch`).
pub fn builder_src(path: &Path) -> anyhow::Result<(ParquetRecordBatchReaderBuilder<Src>, Src)> {
    let (src, md) = open(path)?;
    Ok((ParquetRecordBatchReaderBuilder::new_with_metadata(src.clone(), md), src))
}

/// `ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)`, with the footer cached.
pub fn builder(path: &Path) -> anyhow::Result<ParquetRecordBatchReaderBuilder<Src>> {
    Ok(builder_src(path)?.0)
}

/// A scan over more files than this opens each footer once, so caching them only evicts the
/// footers a selective query will reuse (30 day files cycle through a 16 MB cache).
pub const SCAN_CACHE_FILES: usize = 8;

thread_local! {
    static NO_INSERT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While alive (and `on`), footers this thread parses are not added to the cache.
pub struct NoInsert(Option<bool>);

impl NoInsert {
    pub fn new(on: bool) -> Self {
        NoInsert(on.then(|| NO_INSERT.with(|c| c.replace(true))))
    }
}

impl Drop for NoInsert {
    fn drop(&mut self) {
        if let Some(prev) = self.0 {
            NO_INSERT.with(|c| c.set(prev));
        }
    }
}

/// Reads the selected row groups one at a time with a reader built for that row group only.
/// An open `ParquetRecordBatchReader` holds a zstd decompressor and compressor per projected
/// column chunk (~140 KB each; ~3 MB for all 21 rollup columns), so a k-way merge over 30 files
/// that keeps every reader open costs ~90 MB. A small row group is therefore decoded whole and
/// its reader closed at once; a big one (a month file's) would cost more decoded than its reader
/// does open, so it streams from a reader that lives until that row group ends.
pub struct RgReader {
    src: Src,
    md: ArrowReaderMetadata,
    mask: Option<parquet::arrow::ProjectionMask>,
    pos: usize,
    batch: usize,
    /// the current row group: decoded batches not handed out yet, or its open reader
    queued: std::collections::VecDeque<arrow::array::RecordBatch>,
    open: Option<parquet::arrow::arrow_reader::ParquetRecordBatchReader>,
}

/// What an open reader holds per projected column (zstd decompressor + compressor, measured ~3 MB
/// for 21 columns): a row group decoding to at most this per column costs no more decoded whole,
/// so it is (16384-row day groups: 128 KiB a column); a bigger one streams (131072-row month
/// groups: 1 MiB a column).
pub const RG_EAGER_BYTES_PER_COLUMN: i64 = 144 << 10;

impl RgReader {
    pub fn new(
        b: ParquetRecordBatchReaderBuilder<Src>,
        src: Src,
        rgs: &[usize],
        batch: usize,
        mask: Option<parquet::arrow::ProjectionMask>,
    ) -> anyhow::Result<Self> {
        let md = builder_rgs(b, &src, rgs)?.metadata().clone();
        let md = ArrowReaderMetadata::try_new(md, ArrowReaderOptions::new())?;
        Ok(RgReader {
            src,
            md,
            mask,
            pos: 0,
            batch,
            queued: Default::default(),
            open: None,
        })
    }

    /// Row groups whose metadata this reader keeps (the selected ones only).
    pub fn row_groups_held(&self) -> usize {
        self.md.metadata().num_row_groups()
    }

    /// Decoded bytes of row group `i`'s projected columns, estimated: a column's encoded size
    /// (dictionary and RLE pages are far smaller than the arrow arrays) but at least 8 bytes a row.
    /// Returns (bytes, columns).
    fn projected_bytes(&self, i: usize) -> (i64, i64) {
        let rg = self.md.metadata().row_group(i);
        (0..rg.num_columns())
            .filter(|&c| self.mask.as_ref().map_or(true, |m| m.leaf_included(c)))
            .fold((0, 0), |(b, n), c| {
                (b + rg.column(c).uncompressed_size().max(8 * rg.num_rows()), n + 1)
            })
    }

    /// Whether the row group being read streams from an open reader (a big one).
    pub fn streaming(&self) -> bool {
        self.open.is_some()
    }

    /// The next batch, in row-group order; `None` past the last row group.
    pub fn next_batch(&mut self) -> anyhow::Result<Option<arrow::array::RecordBatch>> {
        loop {
            if let Some(b) = self.queued.pop_front() {
                return Ok(Some(b));
            }
            if let Some(r) = &mut self.open {
                match r.next() {
                    Some(b) => return Ok(Some(b?)),
                    None => self.open = None,
                }
            }
            let i = self.pos;
            if i >= self.md.metadata().num_row_groups() {
                return Ok(None);
            }
            self.pos += 1;
            let rows = self.md.metadata().row_group(i).num_rows().max(1) as usize;
            let mut b = ParquetRecordBatchReaderBuilder::new_with_metadata(self.src.clone(), self.md.clone())
                .with_row_groups(vec![i])
                .with_batch_size(self.batch.min(rows));
            if let Some(m) = &self.mask {
                b = b.with_projection(m.clone());
            }
            let r = b.build()?;
            let (bytes, cols) = self.projected_bytes(i);
            if bytes <= cols * RG_EAGER_BYTES_PER_COLUMN {
                self.queued = r.collect::<Result<_, _>>()?;
            } else {
                self.open = Some(r);
            }
        }
    }
}
