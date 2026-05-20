//! In-memory sorted write buffer.
//!
//! A `Memtable` is the head of an LSM tree's write path: every `put` or
//! `delete` lands here first (in addition to being recorded in the WAL).
//! Reads consult the memtable before falling through to on-disk levels.
//!
//! When the memtable grows past a configured threshold, the engine
//! "flushes" it: the current memtable is sealed and written to disk as
//! an immutable sorted file (an SSTable), and a fresh empty memtable
//! takes its place. The flush mechanism lives in the engine, not here;
//! this module just exposes a sorted in-memory map.
//!
//! ## Tombstones
//!
//! `delete` does **not** remove a key from the underlying map. It
//! inserts a `Tombstone` entry that masks any older value living in
//! lower LSM levels. Tombstones only get garbage-collected during a
//! later compaction once it's known that no older version of the key
//! exists below.
//!
//! ## Concurrency
//!
//! This struct is single-threaded. The engine wraps a memtable in a
//! mutex (or read/write lock for the immutable "flushing" snapshot)
//! before sharing it across threads. Keeping concurrency out of this
//! file makes the data-structure logic easy to test in isolation.

use std::collections::BTreeMap;

/// The value associated with a key in the memtable: either an actual
/// value (`Put`) or a deletion marker (`Tombstone`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    /// Key holds a regular (opaque) value.
    Put(Vec<u8>),
    /// Key holds an embedding vector. Bytes are `dim * 4` long,
    /// containing `dim` f32 values in little-endian order. Once the
    /// HNSW index is wired in (next commit), insertion of a `Vector`
    /// entry will trigger an index update too.
    Vector(Vec<u8>),
    /// Key has been deleted; this masks older versions on disk and in
    /// the vector index.
    Tombstone,
}

impl Entry {
    /// Bytes of the value payload (0 for a tombstone). Used by size
    /// accounting.
    fn payload_size(&self) -> usize {
        match self {
            Entry::Put(v) => v.len(),
            Entry::Vector(v) => v.len(),
            Entry::Tombstone => 0,
        }
    }

    /// True if this entry carries a vector embedding (for index
    /// integration in a later commit).
    pub fn is_vector(&self) -> bool {
        matches!(self, Entry::Vector(_))
    }
}

/// Result of a [`Memtable::get`] lookup. Three-way to preserve the
/// distinction between "deleted" (stop searching lower levels) and
/// "not in this memtable" (continue searching).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lookup<'a> {
    /// Key is present with this value.
    Found(&'a [u8]),
    /// Key has a tombstone in this memtable.
    Deleted,
    /// Key is not in this memtable.
    Missing,
}

/// In-memory sorted write buffer.
#[derive(Debug, Default)]
pub struct Memtable {
    map: BTreeMap<Vec<u8>, Entry>,
    /// Running estimate of in-memory size in bytes (keys + values). Used
    /// by the engine to decide when to trigger a flush. Approximate: it
    /// does not account for `BTreeMap`'s per-node overhead.
    approximate_size: usize,
}

impl Memtable {
    /// Create an empty memtable.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or overwrite `key` with `value`.
    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>) {
        let new_payload = value.len();
        let new_entry = Entry::Put(value);
        match self.map.insert(key.clone(), new_entry) {
            // First insertion: count both key and value.
            None => {
                self.approximate_size += key.len() + new_payload;
            }
            // Overwrite: key bytes are already counted; adjust by the
            // delta in payload size.
            Some(old) => {
                let old_size = old.payload_size();
                self.approximate_size = self
                    .approximate_size
                    .saturating_sub(old_size)
                    .saturating_add(new_payload);
            }
        }
    }

    /// Insert or overwrite `key` with a vector value. Same size
    /// accounting as `put`; differs only in the entry variant stored,
    /// which downstream consumers (WAL/SSTable/HNSW) use to identify
    /// vector entries.
    pub fn put_vector(&mut self, key: Vec<u8>, vector_bytes: Vec<u8>) {
        let new_payload = vector_bytes.len();
        let new_entry = Entry::Vector(vector_bytes);
        match self.map.insert(key.clone(), new_entry) {
            None => {
                self.approximate_size += key.len() + new_payload;
            }
            Some(old) => {
                let old_size = old.payload_size();
                self.approximate_size = self
                    .approximate_size
                    .saturating_sub(old_size)
                    .saturating_add(new_payload);
            }
        }
    }

    /// Record a tombstone for `key`. Idempotent: tombstoning an
    /// already-tombstoned key is a no-op on size.
    pub fn delete(&mut self, key: Vec<u8>) {
        match self.map.insert(key.clone(), Entry::Tombstone) {
            None => {
                // Tombstone for a key we hadn't seen: still adds the
                // key bytes to our footprint.
                self.approximate_size += key.len();
            }
            Some(old) => {
                // Replacing an existing entry with a tombstone: subtract
                // the old payload; key bytes already counted.
                self.approximate_size = self.approximate_size.saturating_sub(old.payload_size());
            }
        }
    }

    /// Look up `key` in this memtable.
    pub fn get(&self, key: &[u8]) -> Lookup<'_> {
        match self.map.get(key) {
            Some(Entry::Put(v)) => Lookup::Found(v.as_slice()),
            Some(Entry::Vector(v)) => Lookup::Found(v.as_slice()),
            Some(Entry::Tombstone) => Lookup::Deleted,
            None => Lookup::Missing,
        }
    }

    /// Number of entries currently held (includes tombstones).
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True if no entries are held.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Estimated in-memory footprint in bytes.
    pub fn approximate_size(&self) -> usize {
        self.approximate_size
    }

    /// Iterate entries in sorted key order. The flush path uses this to
    /// produce a sorted SSTable.
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], &Entry)> {
        self.map.iter().map(|(k, v)| (k.as_slice(), v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_get_is_missing() {
        let m = Memtable::new();
        assert_eq!(m.get(b"nope"), Lookup::Missing);
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        assert_eq!(m.approximate_size(), 0);
    }

    #[test]
    fn put_then_get_returns_value() {
        let mut m = Memtable::new();
        m.put(b"k".to_vec(), b"v".to_vec());
        assert_eq!(m.get(b"k"), Lookup::Found(b"v"));
        assert_eq!(m.len(), 1);
        assert_eq!(m.approximate_size(), 2);
    }

    #[test]
    fn overwrite_replaces_value_and_adjusts_size() {
        let mut m = Memtable::new();
        m.put(b"k".to_vec(), b"v1".to_vec());
        assert_eq!(m.approximate_size(), 3); // 1 (key) + 2 (value)
        m.put(b"k".to_vec(), b"v22".to_vec());
        assert_eq!(m.get(b"k"), Lookup::Found(b"v22"));
        assert_eq!(m.approximate_size(), 4); // 1 + 3
        m.put(b"k".to_vec(), b"".to_vec());
        assert_eq!(m.get(b"k"), Lookup::Found(b""));
        assert_eq!(m.approximate_size(), 1); // 1 + 0
    }

    #[test]
    fn delete_returns_deleted_not_missing() {
        let mut m = Memtable::new();
        m.put(b"k".to_vec(), b"v".to_vec());
        m.delete(b"k".to_vec());
        // The whole point of tombstones: deleted != missing.
        assert_eq!(m.get(b"k"), Lookup::Deleted);
        assert_eq!(m.len(), 1); // tombstone still occupies a slot
    }

    #[test]
    fn delete_on_unseen_key_records_tombstone() {
        let mut m = Memtable::new();
        m.delete(b"ghost".to_vec());
        assert_eq!(m.get(b"ghost"), Lookup::Deleted);
        assert_eq!(m.approximate_size(), 5); // key bytes only
    }

    #[test]
    fn put_after_delete_revives_key() {
        let mut m = Memtable::new();
        m.put(b"k".to_vec(), b"v1".to_vec());
        m.delete(b"k".to_vec());
        assert_eq!(m.get(b"k"), Lookup::Deleted);
        m.put(b"k".to_vec(), b"v2".to_vec());
        assert_eq!(m.get(b"k"), Lookup::Found(b"v2"));
    }

    #[test]
    fn iter_yields_keys_in_sorted_order() {
        let mut m = Memtable::new();
        m.put(b"c".to_vec(), b"3".to_vec());
        m.put(b"a".to_vec(), b"1".to_vec());
        m.put(b"b".to_vec(), b"2".to_vec());
        m.delete(b"d".to_vec());

        let keys: Vec<&[u8]> = m.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![&b"a"[..], &b"b"[..], &b"c"[..], &b"d"[..]]);
    }

    #[test]
    fn iter_distinguishes_put_from_tombstone() {
        let mut m = Memtable::new();
        m.put(b"alive".to_vec(), b"yes".to_vec());
        m.delete(b"dead".to_vec());

        let entries: Vec<(&[u8], &Entry)> = m.iter().collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], (&b"alive"[..], &Entry::Put(b"yes".to_vec())));
        assert_eq!(entries[1], (&b"dead"[..], &Entry::Tombstone));
    }

    // Property test: a memtable, treated as a map from keys to
    // "current state" (present-with-value / deleted / missing), should
    // match a reference model built from the same operation sequence.
    proptest::proptest! {
        #[test]
        fn prop_matches_reference_model(
            ops in proptest::collection::vec(memtable_op_strategy(), 0..100),
        ) {
            // Reference model: `Option<Option<Vec<u8>>>` where outer
            // None = missing, Some(None) = tombstone, Some(Some(v)) = put.
            let mut reference: std::collections::BTreeMap<Vec<u8>, Option<Vec<u8>>>
                = Default::default();
            let mut m = Memtable::new();

            for op in &ops {
                match op {
                    Op::Put(k, v) => {
                        m.put(k.clone(), v.clone());
                        reference.insert(k.clone(), Some(v.clone()));
                    }
                    Op::Delete(k) => {
                        m.delete(k.clone());
                        reference.insert(k.clone(), None);
                    }
                    Op::Get(k) => {
                        let actual = m.get(k);
                        let expected = reference.get(k);
                        let matches = match (actual, expected) {
                            (Lookup::Missing, None) => true,
                            (Lookup::Deleted, Some(None)) => true,
                            (Lookup::Found(v), Some(Some(rv))) => v == rv.as_slice(),
                            _ => false,
                        };
                        proptest::prop_assert!(
                            matches,
                            "mismatch on key {:?}: actual {:?}, expected {:?}",
                            k, actual, expected
                        );
                    }
                }
            }

            // Final cross-check: every key the reference knows about
            // should have a matching state in the memtable.
            for (k, state) in &reference {
                let actual = m.get(k);
                let matches = match (actual, state) {
                    (Lookup::Missing, _) => false, // reference saw this key
                    (Lookup::Deleted, None) => true,
                    (Lookup::Found(v), Some(rv)) => v == rv.as_slice(),
                    _ => false,
                };
                proptest::prop_assert!(matches, "final state mismatch on key {:?}", k);
            }
        }
    }

    #[derive(Debug, Clone)]
    enum Op {
        Put(Vec<u8>, Vec<u8>),
        Delete(Vec<u8>),
        Get(Vec<u8>),
    }

    fn memtable_op_strategy() -> impl proptest::strategy::Strategy<Value = Op> {
        use proptest::prelude::*;
        // Small key space (0..8) so put/delete/get on the same keys
        // happen frequently — otherwise random keys rarely collide and
        // the test loses signal.
        let key = proptest::collection::vec(0u8..8, 1..4);
        let value = proptest::collection::vec(0u8..=255, 0..32);
        prop_oneof![
            (key.clone(), value).prop_map(|(k, v)| Op::Put(k, v)),
            key.clone().prop_map(Op::Delete),
            key.prop_map(Op::Get),
        ]
    }

    #[test]
    fn put_vector_then_get_returns_bytes() {
        let mut m = Memtable::new();
        let v_bytes = vec![1u8, 2, 3, 4, 5, 6, 7, 8]; // 2 f32 values worth
        m.put_vector(b"vec_key".to_vec(), v_bytes.clone());
        assert_eq!(m.get(b"vec_key"), Lookup::Found(v_bytes.as_slice()));
        assert_eq!(m.len(), 1);
        assert_eq!(m.approximate_size(), b"vec_key".len() + v_bytes.len());
    }

    #[test]
    fn put_vector_then_delete_yields_deleted() {
        let mut m = Memtable::new();
        m.put_vector(b"k".to_vec(), vec![0xAB; 12]);
        m.delete(b"k".to_vec());
        assert_eq!(m.get(b"k"), Lookup::Deleted);
    }

    #[test]
    fn put_vector_then_put_overwrites() {
        // Writing a regular Put on top of a Vector replaces it.
        let mut m = Memtable::new();
        m.put_vector(b"k".to_vec(), vec![0xAB; 12]);
        m.put(b"k".to_vec(), b"now i am bytes".to_vec());
        assert_eq!(m.get(b"k"), Lookup::Found(b"now i am bytes"));
    }

    #[test]
    fn iter_yields_vector_entries() {
        let mut m = Memtable::new();
        m.put_vector(b"a".to_vec(), vec![1, 2, 3, 4]);
        m.put(b"b".to_vec(), b"plain".to_vec());
        let entries: Vec<(&[u8], &Entry)> = m.iter().collect();
        assert_eq!(entries.len(), 2);
        assert!(matches!(entries[0].1, Entry::Vector(_)));
        assert!(matches!(entries[1].1, Entry::Put(_)));
    }

    #[test]
    fn is_vector_correctly_identifies_variants() {
        assert!(Entry::Vector(vec![0; 8]).is_vector());
        assert!(!Entry::Put(vec![0; 8]).is_vector());
        assert!(!Entry::Tombstone.is_vector());
    }
}