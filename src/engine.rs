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
use crate::hnsw::{HnswIndex, HnswParams, NodeId};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use tracing::{debug, info, warn};

/// Filename of the active write-ahead log inside the engine's directory.
const WAL_FILENAME: &str = "wal.log";

/// SSTable filename format: `sst_L<level>_<id>.sst`. Level is one
/// digit (we currently use L0 and L1); id is six-digit zero-padded.
const SST_EXT: &str = "sst";

/// Engine configuration.
#[derive(Clone)]
pub struct Options {
    pub path: PathBuf,
    pub io: Arc<dyn Io>,
    pub memtable_max_size_bytes: usize,
    pub l0_compaction_trigger: usize,
    pub bloom_bits_per_key: u32,
    /// Parameters for the HNSW vector index. See [`HnswParams`] for
    /// individual knobs.
    pub hnsw_params: HnswParams,
}


impl Options {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            io: Arc::new(StdFs::new()),
            memtable_max_size_bytes: 4 * 1024 * 1024,
            l0_compaction_trigger: 4,
            bloom_bits_per_key: 10,
            hnsw_params: HnswParams::default(),
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

/// Result of a [`Engine::nearest`] query.
#[derive(Debug, Clone, PartialEq)]
pub struct NearestResult {
    /// The matching key.
    pub key: Vec<u8>,
    /// Distance from query under the configured metric. Smaller is
    /// closer; for cosine the range is `[0, 2]`; for Euclidean it
    /// is squared distance; for inner product it is negated.
    pub distance: f32,
}

/// State of the vector index, held behind an RwLock so concurrent
/// searches don't serialize.
struct VectorIndex {
    hnsw: HnswIndex,
    /// Maps engine-key → graph node id.
    key_to_node: HashMap<Vec<u8>, NodeId>,
    /// Reverse of `key_to_node`: graph node id → engine-key. Kept in
    /// lockstep with `key_to_node` via `register`/`unregister` so the
    /// two never drift. Lets `nearest` map results back to keys in
    /// O(1) instead of scanning.
    node_to_key: HashMap<NodeId, Vec<u8>>,
    /// Vector dimension established on first insert; later inserts
    /// must match. `None` until the first insert.
    expected_dim: Option<usize>,
}

impl VectorIndex {
    fn new(params: HnswParams) -> Self {
        Self {
            hnsw: HnswIndex::new(params),
            key_to_node: HashMap::new(),
            node_to_key: HashMap::new(),
            expected_dim: None,
        }
    }

    fn live_count(&self) -> usize {
        self.hnsw.live_len()
    }

    /// Register a key↔node mapping in both directions. Overwrites any
    /// previous mapping for `key` (caller is responsible for deleting
    /// the old graph node first).
    fn register(&mut self, key: Vec<u8>, node_id: NodeId) {
        self.key_to_node.insert(key.clone(), node_id);
        self.node_to_key.insert(node_id, key);
    }

    /// Remove `key`'s mapping from both maps, returning its node id if
    /// it was present.
    fn unregister(&mut self, key: &[u8]) -> Option<NodeId> {
        let node_id = self.key_to_node.remove(key)?;
        self.node_to_key.remove(&node_id);
        Some(node_id)
    }

    /// Look up the key associated with `node_id`, if any.
    fn key_for_node(&self, node_id: NodeId) -> Option<&[u8]> {
        self.node_to_key.get(&node_id).map(|v| v.as_slice())
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
    /// Vector index, locked separately from the LSM state. Concurrent
    /// `nearest` queries take only the read side; mutations
    /// (`put_indexed`, vector-bearing `delete`) take the write side
    /// and are always acquired *after* the LSM lock to avoid deadlock.
    vector_index: Arc<RwLock<VectorIndex>>,
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

        let inner = EngineInner {
            writer,
            memtable,
            levels,
            next_sst_id,
            closed: false,
        };

        // Build the vector index: snapshot if present and valid, else
        // full-rebuild from the LSM.
        let vector_index = recover_vector_index(&inner, &options)?;

        Ok(Self {
            inner: Mutex::new(inner),
            vector_index: Arc::new(RwLock::new(vector_index)),
            options,
        })
    }

    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let next_id_after = {
            let mut inner = self.lock_inner()?;
            inner.writer.append(RecordKind::Put, key, value)?;
            inner.writer.sync()?;
            inner.memtable.put(key.to_vec(), value.to_vec());
            let before = inner.next_sst_id;
            inner.maybe_auto_flush(&self.options)?;
            inner.maybe_compact(&self.options)?;
            (before != inner.next_sst_id).then_some(inner.next_sst_id)
        };
        if let Some(next_id) = next_id_after {
            self.write_snapshot(next_id)?;
        }
        Ok(())
    }

    pub fn delete(&self, key: &[u8]) -> Result<()> {
        // Step 1: LSM.
        let next_id_after = {
            let mut inner = self.lock_inner()?;
            inner.writer.append(RecordKind::Delete, key, b"")?;
            inner.writer.sync()?;
            inner.memtable.delete(key.to_vec());
            let before = inner.next_sst_id;
            inner.maybe_auto_flush(&self.options)?;
            inner.maybe_compact(&self.options)?;
            (before != inner.next_sst_id).then_some(inner.next_sst_id)
        };

        // Step 2: HNSW (only if the key was indexed).
        {
            let mut index = self
                .vector_index
                .write()
                .expect("vector_index lock poisoned");
            if let Some(node_id) = index.unregister(key) {
                index.hnsw.delete(node_id)?;
            }
        }

        // Step 3: snapshot if the LSM changed shape.
        if let Some(next_id) = next_id_after {
            self.write_snapshot(next_id)?;
        }

        Ok(())
    }

    /// Insert or overwrite `key` with the vector `embedding`.
    ///
    /// The vector is serialized as little-endian f32 bytes and
    /// written to the WAL + memtable like a regular value. A future
    /// commit wires the HNSW index on top so a subsequent
    /// `nearest()` finds the key.
    ///
    /// Rejects:
    /// - Empty embeddings (zero-dimension)
    /// - Embeddings containing non-finite components (NaN or ±inf)
    pub fn put_indexed(&self, key: &[u8], embedding: &[f32]) -> Result<()> {
        if embedding.is_empty() {
            return Err(Error::InvalidArgument(
                "embedding must have at least one dimension".into(),
            ));
        }
        if !embedding.iter().all(|x| x.is_finite()) {
            return Err(Error::InvalidArgument(
                "embedding contains non-finite values (NaN or infinity)".into(),
            ));
        }

        // Validate dimension against the established expected_dim
        // *before* touching the LSM, so a mismatch failure leaves the
        // engine in a fully consistent state.
        {
            let index = self
                .vector_index
                .read()
                .expect("vector_index lock poisoned");
            if let Some(d) = index.expected_dim
                && embedding.len() != d
            {
                return Err(Error::InvalidArgument(format!(
                    "embedding dimension {} does not match established \
                     dimension {d}",
                    embedding.len()
                )));
            }
        }

        let vector_bytes = encode_vector(embedding);

        // Step 1: persist to LSM. WAL fsync ordering as ever.
        let next_id_after = {
            let mut inner = self.lock_inner()?;
            inner.writer.append(RecordKind::Vector, key, &vector_bytes)?;
            inner.writer.sync()?;
            inner.memtable.put_vector(key.to_vec(), vector_bytes);
            let before = inner.next_sst_id;
            inner.maybe_auto_flush(&self.options)?;
            inner.maybe_compact(&self.options)?;
            (before != inner.next_sst_id).then_some(inner.next_sst_id)
        };

        // Step 2: update HNSW. LSM lock is now released so concurrent
        // reads against the LSM don't block on index maintenance.
        // The dim check at the top already validated the new vector
        // against any established dim; we re-check here under the
        // write lock to handle the race where the first insert wins.
        {
            let mut index = self
                .vector_index
                .write()
                .expect("vector_index lock poisoned");

            // First-insert path: establish dim now.
            match index.expected_dim {
                None => {
                    index.expected_dim = Some(embedding.len());
                }
                Some(d) => {
                    if embedding.len() != d {
                        // Lost the race: another put_indexed established
                        // a different dim between our outer check and here.
                        return Err(Error::InvalidArgument(format!(
                            "embedding dimension {} does not match \
                             established dimension {d}",
                            embedding.len()
                        )));
                    }
                }
            }

            // Overwrite path: unregister + delete the old node first.
            if let Some(old_id) = index.unregister(key) {
                index.hnsw.delete(old_id)?;
            }

            // Insert the new node and register both directions.
            let new_id = index.hnsw.insert(embedding)?;
            index.register(key.to_vec(), new_id);
        }

        // Snapshot if the LSM produced a new SSTable as a side effect.
        if let Some(next_id) = next_id_after {
            self.write_snapshot(next_id)?;
        }

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

    /// Find the `k` approximate nearest neighbors of `query`.
    ///
    /// Returns results in ascending distance order (closest first).
    /// Distance semantics depend on the configured metric: cosine
    /// returns `1 - cosine_similarity`, Euclidean returns squared
    /// distance, inner product returns negated similarity.
    ///
    /// Returns an empty vector if the index has no live entries or
    /// if `k == 0`.
    pub fn nearest(&self, query: &[f32], k: usize) -> Result<Vec<NearestResult>> {
        let index = self
            .vector_index
            .read()
            .expect("vector_index lock poisoned");
        let hits = index.hnsw.search(query, k)?;

        // Map NodeIds back to keys via the reverse map (O(1) each).
        let mut results = Vec::with_capacity(hits.len());
        for (node_id, distance) in hits {
            if let Some(key) = index.key_for_node(node_id) {
                results.push(NearestResult {
                    key: key.to_vec(),
                    distance,
                });
            }
        }
        Ok(results)
    }

    /// Find the `k` nearest neighbors of `query` whose key satisfies
    /// `predicate`.
    ///
    /// The predicate receives the engine key (not the value). This
    /// suits metadata-in-key patterns — e.g. keys like
    /// `user_42:memory_99` filtered by `|k| k.starts_with(b"user_42:")`.
    ///
    /// To keep `k` results despite filtering, the search over-queries
    /// the graph and widens adaptively if too many candidates are
    /// rejected. A highly selective predicate (one that passes only a
    /// tiny fraction of vectors) escalates toward an O(N) search,
    /// since that is the only way to correctly find the `k` nearest
    /// passing vectors. If fewer than `k` vectors pass the predicate
    /// in the entire index, all passing vectors are returned.
    pub fn nearest_filtered<F>(
        &self,
        query: &[f32],
        k: usize,
        predicate: F,
    ) -> Result<Vec<NearestResult>>
    where
        F: Fn(&[u8]) -> bool,
    {
        if k == 0 {
            return Ok(Vec::new());
        }
        let index = self
            .vector_index
            .read()
            .expect("vector_index lock poisoned");

        let live = index.hnsw.live_len();
        if live == 0 {
            return Ok(Vec::new());
        }

        // Start with a generously wide search rather than doubling up
        // from a tight bound: filtered queries usually have moderate
        // selectivity, so one wide pass typically fills k without the
        // repeated re-search a 2x-doubling schedule incurs. We still
        // escalate if even the wide pass underfills (very selective
        // filter), capped at the live node count.
        let mut factor = 10usize;
        loop {
            let search_k = k.saturating_mul(factor).min(live);
            let hits = index.hnsw.search(query, search_k)?;

            let mut results = Vec::with_capacity(k);
            for (node_id, distance) in &hits {
                if let Some(key) = index.key_for_node(*node_id)
                    && predicate(key)
                {
                    results.push(NearestResult {
                        key: key.to_vec(),
                        distance: *distance,
                    });
                    if results.len() == k {
                        break;
                    }
                }
            }

            // Done when we've filled k, or we've already searched the
            // whole live index (so widening can't find more).
            if results.len() == k || search_k >= live {
                return Ok(results);
            }
            factor = factor.saturating_mul(2);
        }
    }

    /// Number of live vector entries in the index. Test/debug only.
    #[doc(hidden)]
    pub fn vector_count(&self) -> usize {
        let index = self
            .vector_index
            .read()
            .expect("vector_index lock poisoned");
        index.live_count()
    }

    /// Force the current memtable to disk as a new SSTable, then
    /// rotate the WAL. After this returns successfully, the memtable
    /// is empty and the new SSTable is registered for reads.
    ///
    /// Also runs compaction if the resulting L0 count crosses the
    /// trigger.
    pub fn flush(&self) -> Result<()> {
        let next_id_after = {
            let mut inner = self.lock_inner()?;
            let before = inner.next_sst_id;
            inner.flush_inner(&self.options)?;
            inner.maybe_compact(&self.options)?;
            (before != inner.next_sst_id).then_some(inner.next_sst_id)
        };
        if let Some(next_id) = next_id_after {
            self.write_snapshot(next_id)?;
        }
        Ok(())
    }

    /// Write a fresh HNSW snapshot to disk reflecting the current
    /// vector-index state. Uses the atomic-rename pattern.
    ///
    /// `next_sstable_id` is recorded in the snapshot so recovery can
    /// determine which on-disk SSTables (if any) post-date the
    /// snapshot. Pass the engine's current `next_sst_id` (the value
    /// that would be allocated for the *next* SSTable).
    ///
    /// After the new snapshot is durable, this method best-effort
    /// deletes older `hnsw_*.idx` files in the engine directory. A
    /// failure to delete is logged but not propagated, since the new
    /// snapshot is already durable and orphans are harmless (the
    /// highest-id snapshot wins on recovery).
    fn write_snapshot(&self, next_sstable_id: u64) -> Result<()> {
        let bytes = {
            let index = self
                .vector_index
                .read()
                .expect("vector_index lock poisoned");
            index
                .hnsw
                .encode_snapshot(next_sstable_id, &index.key_to_node)
        };

        let filename = snapshot_filename(next_sstable_id);
        let final_path = self.options.path.join(&filename);
        let tmp_path = self.options.path.join(format!("{filename}.tmp"));

        // Atomic write: write to .tmp, rename, fsync directory.
        {
            let mut writer = self.options.io.open_append(&tmp_path)?;
            writer.append(&bytes)?;
            writer.sync()?;
        }
        self.options.io.rename(&tmp_path, &final_path)?;
        self.options.io.sync_dir(&self.options.path)?;

        // Best-effort cleanup of older snapshots.
        match self.options.io.list_dir(&self.options.path) {
            Ok(entries) => {
                for path in entries {
                    if path == final_path {
                        continue;
                    }
                    let Some(other_id) = parse_snapshot_filename(&path) else {
                        continue;
                    };
                    if other_id != next_sstable_id
                        && let Err(e) = self.options.io.remove_file(&path)
                    {
                        warn!(
                            file = %path.display(),
                            error = %e,
                            "failed to remove old HNSW snapshot; will be cleaned up on next open"
                        );
                    }
                }
            }
            Err(e) => {
                warn!(error = %e, "failed to list dir for old-snapshot cleanup");
            }
        }

        Ok(())
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

/// Filename for an HNSW snapshot at the given sequence id.
fn snapshot_filename(id: u64) -> String {
    format!("hnsw_{id:06}.idx")
}

/// Parse an HNSW snapshot filename to its id, or `None` if the pattern
/// doesn't match.
fn parse_snapshot_filename(path: &Path) -> Option<u64> {
    let stem = path.file_stem()?.to_str()?;
    let rest = stem.strip_prefix("hnsw_")?;
    rest.parse::<u64>().ok()
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
            RecordKind::Vector => memtable.put_vector(rec.key, rec.value),
            RecordKind::Delete => memtable.delete(rec.key),
        }
        replayed += 1;
    }
    if replayed == 0 {
        warn!("WAL existed but contained zero records");
    }
    Ok(())
}

/// Bring up the vector index on engine open.
///
/// Snapshot path if a valid `hnsw_*.idx` exists: load it and apply
/// only the post-snapshot delta. Otherwise (missing or corrupt
/// snapshot) fall back to a full rebuild from the LSM. The snapshot
/// is a cache; the LSM is the source of truth, so a bad snapshot
/// never prevents the engine from opening.
fn recover_vector_index(inner: &EngineInner, options: &Options) -> Result<VectorIndex> {
    let snapshot_path = discover_latest_snapshot(options.io.as_ref(), &options.path)?;

    match snapshot_path {
        Some(path) => match load_snapshot_and_apply_delta(inner, options, &path) {
            Ok(idx) => {
                info!(path = %path.display(), "loaded HNSW from snapshot");
                Ok(idx)
            }
            Err(e) => {
                warn!(
                    path = %path.display(),
                    error = %e,
                    "HNSW snapshot load failed; falling back to full rebuild"
                );
                full_rebuild_index_from_lsm(inner, options)
            }
        },
        None => {
            debug!("no HNSW snapshot present; rebuilding from LSM");
            full_rebuild_index_from_lsm(inner, options)
        }
    }
}

/// Find the highest-id `hnsw_*.idx` in `dir`, cleaning up any older
/// orphan snapshots from interrupted writes. Returns `None` if none
/// exist. `.tmp` files are already removed by `discover_sstables`.
fn discover_latest_snapshot(io: &dyn Io, dir: &Path) -> Result<Option<PathBuf>> {
    let mut candidates: Vec<(u64, PathBuf)> = Vec::new();
    for path in io.list_dir(dir)? {
        if let Some(id) = parse_snapshot_filename(&path) {
            candidates.push((id, path));
        }
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    candidates.sort_by_key(|(id, _)| *id);
    let (_, latest) = candidates.last().cloned().unwrap();
    for (id, path) in &candidates[..candidates.len() - 1] {
        warn!(
            id,
            file = %path.display(),
            "removing orphan HNSW snapshot from interrupted write"
        );
        let _ = io.remove_file(path);
    }
    Ok(Some(latest))
}

/// Snapshot-based recovery: deserialize the snapshot, then apply the
/// memtable + post-snapshot SSTables on top, newest-first, with
/// overwrite + shadowing semantics matching live `put_indexed`/`delete`.
fn load_snapshot_and_apply_delta(
    inner: &EngineInner,
    options: &Options,
    snapshot_path: &Path,
) -> Result<VectorIndex> {
    let bytes = read_file_fully(options.io.as_ref(), snapshot_path)?;
    let (hnsw, key_to_node, snapshot_next_sstable_id) =
        HnswIndex::decode_snapshot(&bytes)?;

    let expected_dim = if hnsw.dim() == 0 {
        None
    } else {
        Some(hnsw.dim())
    };

    // Build the reverse map from the decoded forward map.
    let node_to_key: HashMap<NodeId, Vec<u8>> = key_to_node
        .iter()
        .map(|(k, &id)| (id, k.clone()))
        .collect();
    let mut index = VectorIndex {
        hnsw,
        key_to_node,
        node_to_key,
        expected_dim,
    };

    let mut seen_keys: HashSet<Vec<u8>> = HashSet::new();

    // 1. Memtable (newest).
    for (key, entry) in inner.memtable.iter() {
        apply_delta_entry(&mut index, &mut seen_keys, key.to_vec(), entry)?;
    }

    // 2. SSTables with id >= the snapshot's next_sstable_id, newest
    //    first so newer writes shadow older ones.
    let mut delta_ssts: Vec<&SsTableEntry> = inner
        .levels
        .l0
        .iter()
        .chain(inner.levels.l1.iter())
        .filter(|e| e.id >= snapshot_next_sstable_id)
        .collect();
    delta_ssts.sort_by_key(|e| std::cmp::Reverse(e.id));

    for sst in delta_ssts {
        for item in sst.reader.iter() {
            let (key, entry) = item?;
            apply_delta_entry(&mut index, &mut seen_keys, key, &entry)?;
        }
    }

    Ok(index)
}

/// Apply one delta entry to a (snapshot-loaded) index with overwrite +
/// shadowing. No-op if the key was already handled by a newer source.
fn apply_delta_entry(
    index: &mut VectorIndex,
    seen_keys: &mut HashSet<Vec<u8>>,
    key: Vec<u8>,
    entry: &crate::memtable::Entry,
) -> Result<()> {
    if seen_keys.contains(&key) {
        return Ok(());
    }
    seen_keys.insert(key.clone());

    match entry {
        crate::memtable::Entry::Tombstone => {
            if let Some(node_id) = index.unregister(&key) {
                index.hnsw.delete(node_id)?;
            }
        }
        crate::memtable::Entry::Put(_) => {
            // Newest value is plain bytes, not a vector: drop any
            // older vector association.
            if let Some(node_id) = index.unregister(&key) {
                index.hnsw.delete(node_id)?;
            }
        }
        crate::memtable::Entry::Vector(bytes) => {
            let embedding = decode_vector(bytes)?;
            match index.expected_dim {
                None => index.expected_dim = Some(embedding.len()),
                Some(d) if d != embedding.len() => {
                    return Err(Error::Corruption(format!(
                        "delta vector dim {} != index dim {d}",
                        embedding.len()
                    )));
                }
                Some(_) => {}
            }
            if let Some(old_id) = index.unregister(&key) {
                index.hnsw.delete(old_id)?;
            }
            let new_id = index.hnsw.insert(&embedding)?;
            index.register(key, new_id);
        }
    }
    Ok(())
}

/// Read an entire file through the Io trait into a byte vector.
fn read_file_fully(io: &dyn Io, path: &Path) -> Result<Vec<u8>> {
    let file = io.open_read(path)?;
    let len = file.len()?;
    let mut buf = vec![0u8; len as usize];
    file.read_at(0, &mut buf)?;
    Ok(buf)
}

/// Full-walk fallback: build the index from scratch using the LSM as
/// the source of truth. Newest-first with shadowing.
fn full_rebuild_index_from_lsm(
    inner: &EngineInner,
    options: &Options,
) -> Result<VectorIndex> {
    let mut index = VectorIndex::new(options.hnsw_params.clone());
    let mut seen_keys: HashSet<Vec<u8>> = HashSet::new();

    // Memtable first (newest source).
    for (key, entry) in inner.memtable.iter() {
        let key_vec = key.to_vec();
        if seen_keys.contains(&key_vec) {
            continue;
        }
        match entry {
            crate::memtable::Entry::Tombstone => {
                seen_keys.insert(key_vec);
            }
            crate::memtable::Entry::Vector(bytes) => {
                seen_keys.insert(key_vec.clone());
                insert_into_index(&mut index, key_vec, bytes)?;
            }
            crate::memtable::Entry::Put(_) => {
                seen_keys.insert(key_vec);
            }
        }
    }
    // L0 newest-first.
    for entry in inner.levels.l0.iter() {
        ingest_sstable_into_fresh_index(entry.reader.as_ref(), &mut index, &mut seen_keys)?;
    }
    // L1.
    for entry in inner.levels.l1.iter() {
        ingest_sstable_into_fresh_index(entry.reader.as_ref(), &mut index, &mut seen_keys)?;
    }
    Ok(index)
}

fn ingest_sstable_into_fresh_index(
    reader: &SsTableReader,
    index: &mut VectorIndex,
    seen_keys: &mut HashSet<Vec<u8>>,
) -> Result<()> {
    for item in reader.iter() {
        let (key, entry) = item?;
        if seen_keys.contains(&key) {
            continue;
        }
        match entry {
            crate::memtable::Entry::Tombstone => {
                seen_keys.insert(key);
            }
            crate::memtable::Entry::Vector(bytes) => {
                seen_keys.insert(key.clone());
                insert_into_index(index, key, &bytes)?;
            }
            crate::memtable::Entry::Put(_) => {
                seen_keys.insert(key);
            }
        }
    }
    Ok(())
}

fn insert_into_index(index: &mut VectorIndex, key: Vec<u8>, bytes: &[u8]) -> Result<()> {
    let embedding = decode_vector(bytes)?;
    if let Some(d) = index.expected_dim {
        if d != embedding.len() {
            return Err(Error::Corruption(format!(
                "vector entry dim mismatch during index rebuild: \
                 stored {} but expected {d}",
                embedding.len()
            )));
        }
    } else {
        index.expected_dim = Some(embedding.len());
    }
    let node_id = index.hnsw.insert(&embedding)?;
    index.register(key, node_id);
    Ok(())
}

/// Decode a little-endian f32 byte vector.
fn decode_vector(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::Corruption(format!(
            "vector byte length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

/// Encode a slice of `f32` into a `Vec<u8>` in little-endian order.
/// Used to store vectors in entries that ultimately go through the
/// WAL and SSTable byte-oriented APIs. The reverse operation lives
/// next to the (future) `nearest` API.
fn encode_vector(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}