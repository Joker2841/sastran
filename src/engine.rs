//! The top-level storage engine.
//!
//! `Engine` ties together the WAL, the memtable, and on-disk SSTables
//! into a complete crash-safe key-value store.
//!
//! ## Durability ordering
//!
//! Every write follows the order **WAL-append → fsync → memtable update**.
//! Reads may only observe values that are already durable.
//!
//! ## Recovery
//!
//! On open the engine:
//! 1. Lists the data directory.
//! 2. Identifies existing SSTable files (named `sst_L<level>_<id>.sst`),
//!    reconciles any orphans left by an interrupted compaction, and
//!    opens the remaining files in level / age order.
//! 3. If a WAL exists, replays it into a fresh memtable.
//!
//! After this, the engine resumes exactly the durable state it had
//! before the most recent crash or close.
//!
//! ## Flush
//!
//! Calling [`Engine::flush`] writes the current memtable to a new
//! SSTable, registers the resulting reader, and replaces the WAL with
//! a fresh empty one.
//!
//! ## Compaction
//!
//! When L0 reaches `Options::l0_compaction_trigger` SSTables, an
//! L0+L1 → L1 compaction runs synchronously at the end of the write
//! that pushed L0 over the threshold. The compaction merges all of
//! L0 and L1 into a single new L1 SSTable, dropping tombstones
//! (since L1 is the bottom level in this two-level configuration).

use crate::io::Io;
use crate::io::fs::StdFs;
use crate::memtable::{Entry, Lookup, Memtable};
use crate::sstable::{MergeSource, MergingIterator, SsTableLookup, SsTableReader, SsTableWriter};
use crate::wal::{RecordKind, WalReader, WalWriter};
use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

/// Filename of the active write-ahead log inside the engine's directory.
const WAL_FILENAME: &str = "wal.log";

/// SSTable filename format: `sst_L<level>_<id>.sst`. Level is one
/// digit (we currently use L0 and L1); id is six-digit zero-padded.
const SST_EXT: &str = "sst";

/// Engine configuration.
#[derive(Clone)]
pub struct Options {
    /// Directory that holds the engine's files. Created if absent.
    pub path: PathBuf,

    /// Filesystem backend. Production code passes `Arc::new(StdFs::new())`;
    /// tests can substitute a fault-injecting backend.
    pub io: Arc<dyn Io>,

    /// Maximum approximate size, in bytes, the memtable may reach
    /// before an automatic flush is triggered.
    ///
    /// "Approximate" because the size is tracked as the sum of key and
    /// value bytes only — it does not account for the underlying
    /// `BTreeMap`'s per-node overhead. As a result, the actual memory
    /// footprint may be 2-3x larger than this value.
    ///
    /// Default: 4 MiB. Smaller values cause more frequent flushes and
    /// more SSTables; larger values cause longer flush stalls.
    pub memtable_max_size_bytes: usize,

    /// Number of L0 SSTables that triggers an L0+L1 → L1 compaction.
    ///
    /// When `levels.l0.len() >= l0_compaction_trigger` after a flush,
    /// the engine merges all of L0 and L1 into a single new L1 SSTable.
    ///
    /// Default: 4 (LevelDB's value). Smaller values run compaction more
    /// often (lower read amplification, higher write amplification);
    /// larger values do the opposite.
    pub l0_compaction_trigger: usize,

    /// Bits per key for bloom filters embedded in each SSTable.
    ///
    /// Set to 0 to disable bloom filters entirely (SSTables will be
    /// written without a filter block; reads on absent keys will do
    /// one data-block read per SSTable per lookup).
    ///
    /// Default: 10 (≈ 1% false-positive rate, 7 hash functions).
    pub bloom_bits_per_key: u32,
}

impl Options {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            io: Arc::new(StdFs::new()),
            memtable_max_size_bytes: 4 * 1024 * 1024,
            l0_compaction_trigger: 4,
            bloom_bits_per_key: 10,
        }
    }
}

/// An open SSTable in the engine's level structure.
struct SsTableEntry {
    /// The SSTable's id (matches the number in its filename). Used by
    /// compaction for freshness ordering and for cleanup of input
    /// files after the new output is durable.
    id: u64,
    reader: Arc<SsTableReader>,
}

/// The set of live SSTables, organized by level.
#[derive(Default)]
struct LevelState {
    /// L0: most recent flushes. May overlap in key range. Newest first.
    l0: Vec<SsTableEntry>,
    /// L1: compacted layer. After our (whole-level) compaction this
    /// contains exactly one SSTable; before the first compaction it
    /// is empty. Two-level structure for now.
    l1: Vec<SsTableEntry>,
}

impl LevelState {
    /// Total SSTable count across all levels.
    fn total(&self) -> usize {
        self.l0.len() + self.l1.len()
    }
}

struct EngineInner {
    writer: WalWriter,
    memtable: Memtable,
    /// Live SSTables, grouped by level.
    levels: LevelState,
    /// Next SSTable id to allocate. Monotonic across all levels.
    next_sst_id: u64,
    closed: bool,
}

pub struct Engine {
    inner: Mutex<EngineInner>,
    options: Options,
}

impl Engine {
    pub fn open(options: Options) -> Result<Self> {
        options.io.create_dir_all(&options.path)?;

        // Step 1: discover existing SSTables, grouped by level.
        let (levels, next_sst_id) = discover_sstables(options.io.as_ref(), &options.path)?;
        info!(
            l0 = levels.l0.len(),
            l1 = levels.l1.len(),
            "discovered SSTables"
        );

        let wal_path = options.path.join(WAL_FILENAME);

        // Step 2: replay existing WAL if present.
        let mut memtable = Memtable::new();
        if wal_file_exists(options.io.as_ref(), &wal_path)? {
            info!(path = %wal_path.display(), "replaying existing WAL");
            replay_wal(options.io.as_ref(), &wal_path, &mut memtable)?;
            info!(
                entries = memtable.len(),
                approximate_size = memtable.approximate_size(),
                "WAL replay complete"
            );
        } else {
            debug!("no existing WAL; starting with empty memtable");
        }

        // Step 3: open the WAL for append.
        let writer = WalWriter::open(options.io.as_ref(), &wal_path, &options.path)?;

        Ok(Self {
            inner: Mutex::new(EngineInner {
                writer,
                memtable,
                levels,
                next_sst_id,
                closed: false,
            }),
            options,
        })
    }

    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let mut inner = self.lock_inner()?;
        inner.writer.append(RecordKind::Put, key, value)?;
        inner.writer.sync()?;
        inner.memtable.put(key.to_vec(), value.to_vec());
        inner.maybe_auto_flush(&self.options)?;
        inner.maybe_compact(&self.options)?;
        Ok(())
    }

    pub fn delete(&self, key: &[u8]) -> Result<()> {
        let mut inner = self.lock_inner()?;
        inner.writer.append(RecordKind::Delete, key, b"")?;
        inner.writer.sync()?;
        inner.memtable.delete(key.to_vec());
        inner.maybe_auto_flush(&self.options)?;
        inner.maybe_compact(&self.options)?;
        Ok(())
    }

    /// Look up `key`. Walks memtable → L0 (newest first) → L1.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let inner = self.lock_inner()?;

        // 1. Memtable.
        match inner.memtable.get(key) {
            Lookup::Found(v) => return Ok(Some(v.to_vec())),
            Lookup::Deleted => return Ok(None),
            Lookup::Missing => {}
        }

        // 2. L0: search newest-first. May overlap, so we can't stop
        //    early just because a key would sort outside one SSTable's
        //    range.
        for entry in inner.levels.l0.iter() {
            match entry.reader.get(key)? {
                SsTableLookup::Found(v) => return Ok(Some(v)),
                SsTableLookup::Deleted => return Ok(None),
                SsTableLookup::Missing => continue,
            }
        }

        // 3. L1: no overlap between L1 SSTables, so at most one can
        //    contain the key. For now we linear-scan; with multiple L1
        //    SSTables this should be a binary search on per-file key
        //    ranges (tracked once we have a manifest).
        for entry in inner.levels.l1.iter() {
            match entry.reader.get(key)? {
                SsTableLookup::Found(v) => return Ok(Some(v)),
                SsTableLookup::Deleted => return Ok(None),
                SsTableLookup::Missing => continue,
            }
        }

        Ok(None)
    }

    /// Force the current memtable to disk as a new SSTable, then
    /// rotate the WAL. After this returns successfully, the memtable
    /// is empty and the new SSTable is registered for reads.
    ///
    /// Also runs compaction if the resulting L0 count crosses the
    /// trigger.
    pub fn flush(&self) -> Result<()> {
        let mut inner = self.lock_inner()?;
        inner.flush_inner(&self.options)?;
        inner.maybe_compact(&self.options)
    }

    pub fn close(&self) -> Result<()> {
        let mut inner = self.inner.lock().expect("engine mutex poisoned");
        if inner.closed {
            return Ok(());
        }
        inner.writer.sync()?;
        inner.closed = true;
        Ok(())
    }

    /// Number of registered SSTables across all levels. Test / debug only.
    #[doc(hidden)]
    pub fn sstable_count(&self) -> usize {
        self.inner
            .lock()
            .expect("engine mutex poisoned")
            .levels
            .total()
    }

    /// L0 / L1 SSTable counts. Test / debug only.
    #[doc(hidden)]
    pub fn level_counts(&self) -> (usize, usize) {
        let inner = self.inner.lock().expect("engine mutex poisoned");
        (inner.levels.l0.len(), inner.levels.l1.len())
    }

    /// Memtable length. Test / debug only.
    #[doc(hidden)]
    pub fn memtable_len(&self) -> usize {
        self.inner
            .lock()
            .expect("engine mutex poisoned")
            .memtable
            .len()
    }

    pub fn path(&self) -> &Path {
        &self.options.path
    }

    fn lock_inner(&self) -> Result<std::sync::MutexGuard<'_, EngineInner>> {
        let inner = self.inner.lock().expect("engine mutex poisoned");
        if inner.closed {
            return Err(Error::Closed);
        }
        Ok(inner)
    }
}

impl EngineInner {
    /// Flush the memtable to a new L0 SSTable and rotate the WAL.
    ///
    /// Caller holds the engine lock; this method takes `&mut self`
    /// rather than re-locking, so it is safe to invoke from `put` /
    /// `delete` after they have done their WAL append + memtable
    /// update.
    fn flush_inner(&mut self, options: &Options) -> Result<()> {
        if self.memtable.is_empty() {
            self.writer.sync()?;
            return Ok(());
        }

        let id = self.next_sst_id;
        let filename = sst_filename(0, id); // flush always lands at L0
        let sst_path = options.path.join(&filename);
        let tmp_path = options.path.join(format!("{filename}.tmp"));

        // Write the SSTable to a temporary file, then atomically rename.
        // If we crash mid-write, the .tmp file is orphan garbage that
        // discover_sstables cleans up on next open; the directory never
        // sees a partially-written .sst.
        {
            let mut w = SsTableWriter::create(
                options.io.as_ref(),
                &tmp_path,
                self.memtable.len(),
                options.bloom_bits_per_key,
            )?;
            for (key, entry) in self.memtable.iter() {
                w.add(key, entry)?;
            }
            w.finish()?;
        }
        options.io.rename(&tmp_path, &sst_path)?;
        options.io.sync_dir(&options.path)?;

        // Register the new SSTable at L0, head of the list.
        let reader = Arc::new(SsTableReader::open(options.io.as_ref(), &sst_path)?);
        self.levels.l0.insert(0, SsTableEntry { id, reader });
        self.next_sst_id = id + 1;

        // Rotate the WAL.
        let wal_path = options.path.join(WAL_FILENAME);
        let new_writer = {
            options.io.remove_file(&wal_path)?;
            WalWriter::open(options.io.as_ref(), &wal_path, &options.path)?
        };
        let _old_writer = std::mem::replace(&mut self.writer, new_writer);

        // Reset the memtable.
        self.memtable = Memtable::new();

        info!(
            id,
            l0 = self.levels.l0.len(),
            l1 = self.levels.l1.len(),
            "flushed memtable to L0 SSTable"
        );
        Ok(())
    }

    /// If the memtable has exceeded the configured threshold, flush it.
    fn maybe_auto_flush(&mut self, options: &Options) -> Result<()> {
        if self.memtable.approximate_size() >= options.memtable_max_size_bytes {
            self.flush_inner(options)?;
        }
        Ok(())
    }

    /// If L0 has reached the compaction trigger, run an L0+L1 → L1
    /// compaction.
    fn maybe_compact(&mut self, options: &Options) -> Result<()> {
        if self.levels.l0.len() < options.l0_compaction_trigger {
            return Ok(());
        }
        self.compact_l0_l1(options)
    }

    /// Merge all of L0 + L1 into a fresh L1 SSTable.
    ///
    /// On success, the in-memory level state is updated to have an empty
    /// L0 and an L1 containing exactly one (new) SSTable. The old files
    /// are deleted after the in-memory swap.
    ///
    /// If the new SSTable would be empty (e.g. all input entries were
    /// tombstones at the bottom level), we don't write any output and
    /// simply drop all input SSTables.
    fn compact_l0_l1(&mut self, options: &Options) -> Result<()> {
        // Snapshot the inputs we'll consume. We collect (id, reader)
        // tuples; the readers go into MergeSources, the ids let us
        // delete the right files at the end.
        let mut input_ids: Vec<u64> = Vec::new();
        let mut sources: Vec<MergeSource> = Vec::new();

        // L1 first (older sequence values), then L0. MergingIterator
        // uses `sequence` for tie-breaking, not vec order, so input
        // order is just bookkeeping.
        for entry in &self.levels.l1 {
            input_ids.push(entry.id);
            sources.push(MergeSource {
                sequence: entry.id,
                iter: Box::new(entry.reader.iter()),
            });
        }
        for entry in &self.levels.l0 {
            input_ids.push(entry.id);
            sources.push(MergeSource {
                sequence: entry.id,
                iter: Box::new(entry.reader.iter()),
            });
        }

        let new_id = self.next_sst_id;
        let filename = sst_filename(1, new_id); // output lives at L1
        let sst_path = options.path.join(&filename);
        let tmp_path = options.path.join(format!("{filename}.tmp"));

        // Build the merge and write through it, skipping tombstones
        // because L1 is our bottom level (nothing below can hold an
        // older value of any tombstoned key).
        let merge = MergingIterator::new(sources)?;
        // expected_keys here is an overestimate: it sums input entry
        // counts without deduping. The bloom filter handles this by
        // being slightly over-sized, which lowers the false-positive
        // rate; correctness is unaffected.
        let expected_keys: usize = self
            .levels
            .l0
            .iter()
            .chain(self.levels.l1.iter())
            .map(|e| {
                // We don't track per-SSTable entry counts. Use block
                // count * a rough average as a proxy. Acceptable for
                // bloom sizing since being off by 2x only changes the
                // FP rate by a small factor.
                e.reader.block_count() * 256
            })
            .sum();
        let mut writer = SsTableWriter::create(
            options.io.as_ref(),
            &tmp_path,
            expected_keys,
            options.bloom_bits_per_key,
        )?;
        let mut wrote_any = false;
        for item in merge {
            let (key, entry) = item?;
            if matches!(entry, Entry::Tombstone) {
                continue; // drop: bottom-level tombstone garbage collection.
            }
            writer.add(&key, &entry)?;
            wrote_any = true;
        }

        if wrote_any {
            writer.finish()?;
            options.io.rename(&tmp_path, &sst_path)?;
            options.io.sync_dir(&options.path)?;

            let reader = Arc::new(SsTableReader::open(options.io.as_ref(), &sst_path)?);
            self.next_sst_id = new_id + 1;
            self.levels.l0.clear();
            self.levels.l1 = vec![SsTableEntry { id: new_id, reader }];
        } else {
            // No output written. Drop the writer to release the .tmp
            // file handle, then best-effort delete it.
            drop(writer);
            let _ = options.io.remove_file(&tmp_path);
            // Bump next_sst_id anyway so we don't reuse this id for a
            // future SSTable that might collide with an orphan.
            self.next_sst_id = new_id + 1;
            self.levels.l0.clear();
            self.levels.l1.clear();
        }

        // Delete the old files. After this point the on-disk state
        // matches the new in-memory levels. Errors here are non-fatal:
        // discovery on the next open will clean up orphans.
        for id in input_ids {
            // We don't know which level each input was at without
            // storing it, so try both names. (Could be smarter but
            // this is straightforward and only runs occasionally.)
            for level in [0u8, 1u8] {
                let path = options.path.join(sst_filename(level, id));
                let _ = options.io.remove_file(&path);
            }
        }
        let _ = options.io.sync_dir(&options.path);

        info!(
            new_id,
            wrote_output = wrote_any,
            l0 = self.levels.l0.len(),
            l1 = self.levels.l1.len(),
            "compacted L0+L1 -> L1"
        );

        Ok(())
    }
}

fn sst_filename(level: u8, id: u64) -> String {
    format!("sst_L{level}_{id:06}.{SST_EXT}")
}

/// Parse `sst_L<level>_<id>.sst` from a path. Returns `None` if the
/// filename doesn't match.
fn parse_sst_filename(path: &Path) -> Option<(u8, u64)> {
    let stem = path.file_stem()?.to_str()?;
    let rest = stem.strip_prefix("sst_L")?;
    let (level_str, id_str) = rest.split_once('_')?;
    let level: u8 = level_str.parse().ok()?;
    let id: u64 = id_str.parse().ok()?;
    Some((level, id))
}

/// Discover existing SSTables in `dir`. Returns level state plus the
/// next id to allocate.
fn discover_sstables(io: &dyn Io, dir: &Path) -> Result<(LevelState, u64)> {
    let mut l0: Vec<(u64, PathBuf)> = Vec::new();
    let mut l1: Vec<(u64, PathBuf)> = Vec::new();
    let mut max_id: Option<u64> = None;

    for path in io.list_dir(dir)? {
        // Clean up stale .tmp files from a crash mid-flush or
        // mid-compaction. Best-effort: ignore errors.
        if path.extension().is_some_and(|ext| ext == "tmp") {
            let _ = io.remove_file(&path);
            continue;
        }
        if path.extension().is_none_or(|ext| ext != SST_EXT) {
            continue;
        }
        let Some((level, id)) = parse_sst_filename(&path) else {
            warn!(file = %path.display(), "ignoring unparseable SSTable filename");
            continue;
        };
        max_id = Some(max_id.map_or(id, |m| m.max(id)));
        match level {
            0 => l0.push((id, path)),
            1 => l1.push((id, path)),
            other => {
                warn!(
                    file = %path.display(),
                    level = other,
                    "ignoring SSTable at unsupported level"
                );
            }
        }
    }

    // Crash-recovery reconciliation for L1.
    //
    // Compaction's safe ordering is: write new L1 -> update in-memory
    // levels -> delete old L1+L0 files. If the process died after the
    // new L1 was durable but before the old files were removed, we'd
    // see multiple L1 files on disk with overlapping key ranges. Our
    // read path assumes L1 is non-overlapping, so we must reconcile.
    //
    // The new L1 always has the *highest* id (compaction's output uses
    // the next-allocated id). So: among the L1 files we found, keep
    // only the one with the largest id; delete the rest as orphans.
    if l1.len() > 1 {
        l1.sort_by_key(|(id, _)| *id);
        let keepers = l1.split_off(l1.len() - 1);
        for (id, path) in &l1 {
            warn!(
                id,
                file = %path.display(),
                "removing orphan L1 SSTable from interrupted compaction"
            );
            let _ = io.remove_file(path);
        }
        l1 = keepers;
    }

    // After L1 reconciliation, any L0 file with id <= the surviving L1's
    // id is an orphan from a pre-crash compaction (their data is already
    // in the new L1). In the sync-compaction model, all L0 ids that
    // post-date a successful compaction are strictly larger than that
    // compaction's L1 id, so this threshold is precise.
    if let Some((surviving_l1_id, _)) = l1.first() {
        let threshold = *surviving_l1_id;
        let (orphan_l0, live_l0): (Vec<_>, Vec<_>) =
            l0.into_iter().partition(|(id, _)| *id <= threshold);
        for (id, path) in &orphan_l0 {
            warn!(
                id,
                file = %path.display(),
                "removing orphan L0 SSTable from interrupted compaction"
            );
            let _ = io.remove_file(path);
        }
        l0 = live_l0;
    }

    // L0: sort by id ascending, then reverse so newest is first.
    l0.sort_by_key(|(id, _)| *id);
    let mut l0_entries: Vec<SsTableEntry> = l0
        .into_iter()
        .map(|(id, path)| {
            let reader = Arc::new(SsTableReader::open(io, &path)?);
            Ok(SsTableEntry { id, reader })
        })
        .collect::<Result<Vec<_>>>()?;
    l0_entries.reverse();

    // L1: keep id-ascending order (after compaction, ids order matches
    // key range since each compaction produces one new L1).
    l1.sort_by_key(|(id, _)| *id);
    let l1_entries: Vec<SsTableEntry> = l1
        .into_iter()
        .map(|(id, path)| {
            let reader = Arc::new(SsTableReader::open(io, &path)?);
            Ok(SsTableEntry { id, reader })
        })
        .collect::<Result<Vec<_>>>()?;

    let next_id = max_id.map_or(0, |m| m + 1);
    Ok((
        LevelState {
            l0: l0_entries,
            l1: l1_entries,
        },
        next_id,
    ))
}

fn wal_file_exists(io: &dyn Io, wal_path: &Path) -> Result<bool> {
    let parent = wal_path
        .parent()
        .expect("wal_path always has a parent (engine directory)");
    let entries = io.list_dir(parent)?;
    Ok(entries.iter().any(|p| p == wal_path))
}

fn replay_wal(io: &dyn Io, wal_path: &Path, memtable: &mut Memtable) -> Result<()> {
    let mut reader = WalReader::open(io, wal_path)?;
    let mut replayed = 0usize;
    while let Some(rec) = reader.next_record()? {
        match rec.kind {
            RecordKind::Put => memtable.put(rec.key, rec.value),
            RecordKind::Delete => memtable.delete(rec.key),
        }
        replayed += 1;
    }
    if replayed == 0 {
        warn!("WAL existed but contained zero records");
    }
    Ok(())
}