//! K-way merge of sorted streams with newest-wins deduplication.
//!
//! [`MergingIterator`] consumes several sorted iterators of
//! `(key, Entry)` pairs and produces a single sorted stream in which
//! each key appears exactly once: when multiple inputs hold entries
//! for the same key, the entry from the source with the **highest
//! sequence number** is emitted and the others are discarded.
//!
//! The intended consumers are:
//! - **Compaction**, which merges several SSTable iterators into one
//!   output stream. Tombstone filtering (drop-tombstones-on-bottom-level)
//!   is the consumer's responsibility — this iterator yields everything
//!   that won the per-key contest, tombstones included.
//! - **Range scans** (future), which will merge the memtable's iterator
//!   with several SSTable iterators on the read path.
//!
//! ## Complexity
//!
//! K-way merge over N sources via a binary heap: each output entry
//! costs O(log N) heap operations. Plus, for every duplicate key, an
//! extra O(log N) per stale source.

use crate::memtable::Entry;
use crate::{Error, Result};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// An owned iterator over `(key, entry)` pairs in ascending key order.
///
/// Type alias to keep signatures readable. `Send` is included so a
/// merge can later be moved onto a background thread without further
/// type surgery.
pub type SortedEntryIter = Box<dyn Iterator<Item = Result<(Vec<u8>, Entry)>> + Send>;

/// One source feeding the merge.
pub struct MergeSource {
    /// Higher value = newer. When two sources hold the same key, the
    /// one with the higher sequence wins.
    pub sequence: u64,
    /// The sorted entry stream.
    pub iter: SortedEntryIter,
}

/// A node in the merge heap. Each node represents one source's
/// currently-buffered entry; the heap is sorted so that the smallest
/// `key` (and, on ties, the *largest* sequence) lives at the top.
struct HeapEntry {
    key: Vec<u8>,
    entry: Entry,
    /// Index into `MergingIterator::sources` so we know which source
    /// produced this node and whose `next()` to call when we pop it.
    source_idx: usize,
    /// Source sequence number, copied here so the `Ord` impl doesn't
    /// need to reach into the parent state.
    sequence: u64,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap. We want the *smallest* key at the
        // top, so we invert the key comparison. On equal keys we want
        // the *largest* sequence at the top (freshest wins), so the
        // sequence comparison is *not* inverted.
        match other.key.cmp(&self.key) {
            Ordering::Equal => self.sequence.cmp(&other.sequence),
            non_eq => non_eq,
        }
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.sequence == other.sequence
    }
}

impl Eq for HeapEntry {}

/// K-way merge iterator.
pub struct MergingIterator {
    sources: Vec<SortedEntryIter>,
    heap: BinaryHeap<HeapEntry>,
    /// Sequence number for each source, indexed by source_idx. Held
    /// separately so the heap nodes can be small.
    sequences: Vec<u64>,
    /// Error from a refill that happened during a previous emit. It
    /// surfaces on the next `next()` call, after which the iterator
    /// is poisoned. Holding it here lets us emit the already-decoded
    /// winner before reporting the failure.
    pending_error: Option<Error>,
    /// Set after the first error has been surfaced. Subsequent calls
    /// return None cleanly.
    poisoned: bool,
}

impl MergingIterator {
    /// Build a merge over `sources`. Each source must yield its entries
    /// in strictly ascending key order; the iterator does not check
    /// this and will produce incorrect output if it is violated.
    pub fn new(sources: Vec<MergeSource>) -> Result<Self> {
        let mut iters = Vec::with_capacity(sources.len());
        let mut sequences = Vec::with_capacity(sources.len());
        let mut heap = BinaryHeap::with_capacity(sources.len());

        for (idx, src) in sources.into_iter().enumerate() {
            sequences.push(src.sequence);
            let mut iter = src.iter;
            // Prime the heap with each source's first entry.
            if let Some(item) = iter.next() {
                let (key, entry) = item?;
                heap.push(HeapEntry {
                    key,
                    entry,
                    source_idx: idx,
                    sequence: sequences[idx],
                });
            }
            iters.push(iter);
        }

        Ok(Self {
            sources: iters,
            heap,
            sequences,
            pending_error: None,
            poisoned: false,
        })
    }

    /// Pull the next entry from source `idx` and push it onto the heap
    /// (if there is one).
    fn refill(&mut self, idx: usize) -> Result<()> {
        match self.sources[idx].next() {
            None => Ok(()),
            Some(Err(e)) => Err(e),
            Some(Ok((key, entry))) => {
                self.heap.push(HeapEntry {
                    key,
                    entry,
                    source_idx: idx,
                    sequence: self.sequences[idx],
                });
                Ok(())
            }
        }
    }
}

impl Iterator for MergingIterator {
    type Item = Result<(Vec<u8>, Entry)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.poisoned {
            return None;
        }

        // If a previous emit queued an error, surface it now and poison.
        // The winner from that earlier call was already returned; this
        // call exists to report the latent failure.
        if let Some(e) = self.pending_error.take() {
            self.poisoned = true;
            return Some(Err(e));
        }

        // Pop the freshest entry for the smallest key.
        let winner = self.heap.pop()?;

        // Drop every other heap entry whose key matches the winner's:
        // these are older versions of the same key. Each removal
        // requires a refill from its source. Refill errors do *not*
        // immediately fail this call: we have a perfectly good winner
        // to emit. We queue the first error and let it surface on the
        // next next() call.
        while let Some(top) = self.heap.peek() {
            if top.key != winner.key {
                break;
            }
            // pop() is safe because peek() succeeded.
            let stale = self.heap.pop().unwrap();
            if let Err(e) = self.refill(stale.source_idx) {
                if self.pending_error.is_none() {
                    self.pending_error = Some(e);
                }
                // Stop draining: we may leave more stale duplicates of
                // the winner's key in the heap, but the iterator will
                // be poisoned on the next call so nothing will observe
                // them. Continuing the drain risks shadowing the first
                // error with a later one.
                break;
            }
        }

        // Refill the winner's source. Same rules: queue the error,
        // emit the winner now, surface on the next call.
        if let Err(e) = self.refill(winner.source_idx)
            && self.pending_error.is_none()
        {
            self.pending_error = Some(e);
        }

        Some(Ok((winner.key, winner.entry)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a source from a list of (key, entry) pairs.
    fn src(sequence: u64, entries: Vec<(&[u8], Entry)>) -> MergeSource {
        let owned: Vec<Result<(Vec<u8>, Entry)>> = entries
            .into_iter()
            .map(|(k, e)| Ok((k.to_vec(), e)))
            .collect();
        MergeSource {
            sequence,
            iter: Box::new(owned.into_iter()),
        }
    }

    fn put(v: &[u8]) -> Entry {
        Entry::Put(v.to_vec())
    }

    fn drain(it: MergingIterator) -> Vec<(Vec<u8>, Entry)> {
        it.collect::<Result<Vec<_>>>().unwrap()
    }

    #[test]
    fn empty_input_yields_nothing() {
        let it = MergingIterator::new(vec![]).unwrap();
        assert!(drain(it).is_empty());
    }

    #[test]
    fn single_source_passes_through() {
        let it = MergingIterator::new(vec![src(
            0,
            vec![
                (b"a", put(b"1")),
                (b"b", put(b"2")),
                (b"c", put(b"3")),
            ],
        )])
        .unwrap();
        let got = drain(it);
        assert_eq!(
            got,
            vec![
                (b"a".to_vec(), put(b"1")),
                (b"b".to_vec(), put(b"2")),
                (b"c".to_vec(), put(b"3")),
            ]
        );
    }

    #[test]
    fn two_disjoint_sources_interleave_by_key() {
        let s1 = src(0, vec![(b"a", put(b"1")), (b"c", put(b"3"))]);
        let s2 = src(1, vec![(b"b", put(b"2")), (b"d", put(b"4"))]);
        let got = drain(MergingIterator::new(vec![s1, s2]).unwrap());
        assert_eq!(
            got,
            vec![
                (b"a".to_vec(), put(b"1")),
                (b"b".to_vec(), put(b"2")),
                (b"c".to_vec(), put(b"3")),
                (b"d".to_vec(), put(b"4")),
            ]
        );
    }

    #[test]
    fn duplicate_keys_keep_freshest_only() {
        // s1 is older (sequence 0), s2 is newer (sequence 1).
        // For key "k", the newer value should win.
        let s1 = src(0, vec![(b"k", put(b"old"))]);
        let s2 = src(1, vec![(b"k", put(b"new"))]);
        let got = drain(MergingIterator::new(vec![s1, s2]).unwrap());
        assert_eq!(got, vec![(b"k".to_vec(), put(b"new"))]);
    }

    #[test]
    fn freshness_works_regardless_of_input_order() {
        // Same as above but the freshest source is listed *first* in
        // the input vector. Result should be identical.
        let s_new = src(1, vec![(b"k", put(b"new"))]);
        let s_old = src(0, vec![(b"k", put(b"old"))]);
        let got = drain(MergingIterator::new(vec![s_new, s_old]).unwrap());
        assert_eq!(got, vec![(b"k".to_vec(), put(b"new"))]);
    }

    #[test]
    fn tombstone_can_win_over_put() {
        // Tombstone in the newer source masks the put in the older.
        let s_old = src(0, vec![(b"k", put(b"alive"))]);
        let s_new = src(1, vec![(b"k", Entry::Tombstone)]);
        let got = drain(MergingIterator::new(vec![s_old, s_new]).unwrap());
        assert_eq!(got, vec![(b"k".to_vec(), Entry::Tombstone)]);
    }

    #[test]
    fn put_can_win_over_older_tombstone() {
        // The reverse: a fresh put resurrects a previously-tombstoned key.
        let s_old = src(0, vec![(b"k", Entry::Tombstone)]);
        let s_new = src(1, vec![(b"k", put(b"alive"))]);
        let got = drain(MergingIterator::new(vec![s_old, s_new]).unwrap());
        assert_eq!(got, vec![(b"k".to_vec(), put(b"alive"))]);
    }

    #[test]
    fn three_sources_with_overlapping_keys() {
        // Sequence layout: s_a=2 (newest), s_b=1, s_c=0 (oldest)
        // Keys:           s_a: a, c        s_b: a, b, c    s_c: a, b
        // Expected output:
        //   a -> from s_a (sequence 2, newest)
        //   b -> from s_b (sequence 1, newer than s_c)
        //   c -> from s_a (sequence 2, newest)
        let s_a = src(2, vec![(b"a", put(b"a@2")), (b"c", put(b"c@2"))]);
        let s_b = src(
            1,
            vec![
                (b"a", put(b"a@1")),
                (b"b", put(b"b@1")),
                (b"c", put(b"c@1")),
            ],
        );
        let s_c = src(0, vec![(b"a", put(b"a@0")), (b"b", put(b"b@0"))]);

        let got = drain(MergingIterator::new(vec![s_a, s_b, s_c]).unwrap());
        assert_eq!(
            got,
            vec![
                (b"a".to_vec(), put(b"a@2")),
                (b"b".to_vec(), put(b"b@1")),
                (b"c".to_vec(), put(b"c@2")),
            ]
        );
    }

    #[test]
    fn empty_sources_are_skipped_cleanly() {
        let s1 = src(0, vec![]);
        let s2 = src(1, vec![(b"x", put(b"1"))]);
        let s3 = src(2, vec![]);
        let got = drain(MergingIterator::new(vec![s1, s2, s3]).unwrap());
        assert_eq!(got, vec![(b"x".to_vec(), put(b"1"))]);
    }

    #[test]
    fn error_from_source_poisons_iterator() {
        // Build a source whose iterator yields one Ok followed by Err.
        let err_iter: SortedEntryIter = Box::new(
            vec![
                Ok((b"a".to_vec(), put(b"1"))),
                Err(crate::Error::Corruption("synthetic".into())),
            ]
            .into_iter(),
        );
        let good_iter: SortedEntryIter = Box::new(
            vec![Ok((b"b".to_vec(), put(b"2")))].into_iter(),
        );
        let sources = vec![
            MergeSource { sequence: 0, iter: err_iter },
            MergeSource { sequence: 1, iter: good_iter },
        ];
        let mut it = MergingIterator::new(sources).unwrap();

        // First emit succeeds: "a" from the error source's first entry.
        let first = it.next().expect("first emit").expect("not an error yet");
        assert_eq!(first.0, b"a");

        // Second call refills from the error source and surfaces the error.
        let err = it.next().expect("second emit").expect_err("expected error");
        assert!(matches!(err, crate::Error::Corruption(_)));

        // Iterator is now poisoned: subsequent calls yield None.
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    #[test]
    fn many_keys_across_many_sources() {
        // Build 5 sources, each with 10 unique keys, no overlap. Output
        // should be all 50 keys in sorted order with no duplicates.
        let mut sources = Vec::new();
        for src_idx in 0..5u32 {
            let mut entries = Vec::new();
            for i in 0..10u32 {
                let key = format!("k_{:02}", src_idx * 10 + i);
                let value = format!("v_{src_idx}_{i}");
                entries.push((key.into_bytes(), Entry::Put(value.into_bytes())));
            }
            let owned: Vec<Result<(Vec<u8>, Entry)>> =
                entries.into_iter().map(Ok).collect();
            sources.push(MergeSource {
                sequence: src_idx as u64,
                iter: Box::new(owned.into_iter()),
            });
        }
        let got = drain(MergingIterator::new(sources).unwrap());
        assert_eq!(got.len(), 50);
        for (i, (key, _)) in got.iter().enumerate() {
            let expected = format!("k_{i:02}");
            assert_eq!(key, &expected.into_bytes());
        }
    }

    #[test]
    fn all_sources_have_same_key_freshest_wins() {
        // Pathological case: every source holds an entry for the same key.
        // The freshest sequence should win and all others should drop.
        let s_a = src(0, vec![(b"k", put(b"v@0"))]);
        let s_b = src(5, vec![(b"k", put(b"v@5"))]);
        let s_c = src(3, vec![(b"k", put(b"v@3"))]);
        let s_d = src(99, vec![(b"k", put(b"v@99"))]);
        let s_e = src(1, vec![(b"k", put(b"v@1"))]);

        let got = drain(MergingIterator::new(vec![s_a, s_b, s_c, s_d, s_e]).unwrap());
        assert_eq!(got, vec![(b"k".to_vec(), put(b"v@99"))]);
    }
}