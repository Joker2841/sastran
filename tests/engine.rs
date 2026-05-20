//! End-to-end engine tests.

use sastran::{Engine, Options};
use tempfile::tempdir;

#[test]
fn put_get_delete_basic_flow() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();

    engine.put(b"key1", b"value1").unwrap();
    engine.put(b"key2", b"value2").unwrap();

    assert_eq!(engine.get(b"key1").unwrap(), Some(b"value1".to_vec()));
    assert_eq!(engine.get(b"key2").unwrap(), Some(b"value2".to_vec()));
    assert_eq!(engine.get(b"missing").unwrap(), None);

    engine.delete(b"key1").unwrap();
    assert_eq!(engine.get(b"key1").unwrap(), None);
    assert_eq!(engine.get(b"key2").unwrap(), Some(b"value2".to_vec()));

    engine.close().unwrap();
}

#[test]
fn overwrite_replaces_value() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();

    engine.put(b"k", b"v1").unwrap();
    engine.put(b"k", b"v2").unwrap();
    engine.put(b"k", b"v3").unwrap();
    assert_eq!(engine.get(b"k").unwrap(), Some(b"v3".to_vec()));
}

#[test]
fn data_persists_across_reopen() {
    let dir = tempdir().unwrap();

    {
        let engine = Engine::open(Options::new(dir.path())).unwrap();
        engine.put(b"persist", b"yes").unwrap();
        engine.put(b"also", b"this").unwrap();
        engine.delete(b"also").unwrap();
        engine.close().unwrap();
    }

    let engine = Engine::open(Options::new(dir.path())).unwrap();
    assert_eq!(engine.get(b"persist").unwrap(), Some(b"yes".to_vec()));
    // The delete is also recovered — `also` does not come back.
    assert_eq!(engine.get(b"also").unwrap(), None);
}

#[test]
fn many_writes_survive_reopen() {
    let dir = tempdir().unwrap();

    {
        let engine = Engine::open(Options::new(dir.path())).unwrap();
        for i in 0..1000u32 {
            let key = format!("key_{i:04}");
            let value = format!("value_{i}");
            engine.put(key.as_bytes(), value.as_bytes()).unwrap();
        }
        engine.close().unwrap();
    }

    let engine = Engine::open(Options::new(dir.path())).unwrap();
    assert_eq!(engine.memtable_len(), 1000);
    for i in 0..1000u32 {
        let key = format!("key_{i:04}");
        let expected = format!("value_{i}");
        assert_eq!(
            engine.get(key.as_bytes()).unwrap(),
            Some(expected.into_bytes()),
            "mismatch on {key}"
        );
    }
}

#[test]
fn operations_after_close_fail() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put(b"k", b"v").unwrap();
    engine.close().unwrap();

    let err = engine.put(b"k2", b"v2").unwrap_err();
    assert!(matches!(err, sastran::Error::Closed), "got {err:?}");

    let err = engine.get(b"k").unwrap_err();
    assert!(matches!(err, sastran::Error::Closed), "got {err:?}");
}

#[test]
fn close_is_idempotent() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.close().unwrap();
    engine.close().unwrap(); // second close should not error
}

#[test]
fn empty_value_is_valid() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put(b"k", b"").unwrap();
    assert_eq!(engine.get(b"k").unwrap(), Some(Vec::new()));
}

#[test]
fn put_after_delete_is_visible() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put(b"k", b"v1").unwrap();
    engine.delete(b"k").unwrap();
    assert_eq!(engine.get(b"k").unwrap(), None);
    engine.put(b"k", b"v2").unwrap();
    assert_eq!(engine.get(b"k").unwrap(), Some(b"v2".to_vec()));
}

#[test]
fn delete_on_unseen_key_succeeds() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.delete(b"never_existed").unwrap();
    assert_eq!(engine.get(b"never_existed").unwrap(), None);
}

#[test]
fn engine_is_send_and_sync() {
    // Compile-time check: Engine must be safe to share across threads.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Engine>();
}

#[test]
fn flush_moves_data_to_sstable_and_clears_memtable() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();

    for i in 0..100u32 {
        engine
            .put(format!("k_{i:03}").as_bytes(), format!("v_{i}").as_bytes())
            .unwrap();
    }
    assert_eq!(engine.memtable_len(), 100);
    assert_eq!(engine.sstable_count(), 0);

    engine.flush().unwrap();
    assert_eq!(engine.memtable_len(), 0);
    assert_eq!(engine.sstable_count(), 1);

    // All keys still retrievable, now from SSTable.
    for i in 0..100u32 {
        let key = format!("k_{i:03}");
        let expected = format!("v_{i}");
        assert_eq!(
            engine.get(key.as_bytes()).unwrap(),
            Some(expected.into_bytes())
        );
    }
}

#[test]
fn data_persists_across_reopen_after_flush() {
    let dir = tempdir().unwrap();
    {
        let engine = Engine::open(Options::new(dir.path())).unwrap();
        for i in 0..50u32 {
            engine
                .put(format!("k_{i:03}").as_bytes(), format!("v_{i}").as_bytes())
                .unwrap();
        }
        engine.flush().unwrap();
        engine.close().unwrap();
    }

    let engine = Engine::open(Options::new(dir.path())).unwrap();
    assert_eq!(engine.sstable_count(), 1);
    assert_eq!(engine.memtable_len(), 0);
    for i in 0..50u32 {
        let key = format!("k_{i:03}");
        let expected = format!("v_{i}");
        assert_eq!(
            engine.get(key.as_bytes()).unwrap(),
            Some(expected.into_bytes())
        );
    }
}

#[test]
fn newer_memtable_shadows_older_sstable() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();

    engine.put(b"k", b"old").unwrap();
    engine.flush().unwrap();
    assert_eq!(engine.sstable_count(), 1);

    // Now overwrite in the fresh memtable.
    engine.put(b"k", b"new").unwrap();
    assert_eq!(engine.memtable_len(), 1);

    assert_eq!(engine.get(b"k").unwrap(), Some(b"new".to_vec()));
}

#[test]
fn delete_in_memtable_masks_sstable() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();

    engine.put(b"k", b"v").unwrap();
    engine.flush().unwrap();
    engine.delete(b"k").unwrap();

    assert_eq!(engine.get(b"k").unwrap(), None);

    // After a second flush, the tombstone lives in the newer SSTable.
    engine.flush().unwrap();
    assert_eq!(engine.sstable_count(), 2);
    assert_eq!(engine.get(b"k").unwrap(), None);
}

#[test]
fn newer_sstable_shadows_older_sstable() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();

    engine.put(b"k", b"v1").unwrap();
    engine.flush().unwrap();
    engine.put(b"k", b"v2").unwrap();
    engine.flush().unwrap();
    assert_eq!(engine.sstable_count(), 2);

    assert_eq!(engine.get(b"k").unwrap(), Some(b"v2".to_vec()));
}

#[test]
fn flush_on_empty_memtable_is_noop() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.flush().unwrap();
    assert_eq!(engine.sstable_count(), 0);
}

#[test]
fn multiple_flushes_and_reopen_accumulate_sstables() {
    let dir = tempdir().unwrap();
    {
        let engine = Engine::open(Options::new(dir.path())).unwrap();
        engine.put(b"a", b"1").unwrap();
        engine.flush().unwrap();
        engine.put(b"b", b"2").unwrap();
        engine.flush().unwrap();
        engine.put(b"c", b"3").unwrap();
        engine.flush().unwrap();
        engine.close().unwrap();
    }

    let engine = Engine::open(Options::new(dir.path())).unwrap();
    assert_eq!(engine.sstable_count(), 3);
    assert_eq!(engine.get(b"a").unwrap(), Some(b"1".to_vec()));
    assert_eq!(engine.get(b"b").unwrap(), Some(b"2".to_vec()));
    assert_eq!(engine.get(b"c").unwrap(), Some(b"3".to_vec()));
}

#[test]
fn auto_flush_triggers_when_threshold_exceeded() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    // Small threshold so we don't have to write megabytes in a test.
    options.memtable_max_size_bytes = 1024;
    let engine = Engine::open(options).unwrap();

    // Write enough data to exceed 1 KiB. With 32-byte keys + 32-byte
    // values, ~20 puts comfortably crosses the threshold.
    for i in 0..40u32 {
        let key = format!("key_{i:028}"); // 32 bytes
        let value = format!("val_{i:028}"); // 32 bytes
        engine.put(key.as_bytes(), value.as_bytes()).unwrap();
    }

    // At least one auto-flush should have happened.
    assert!(
        engine.sstable_count() >= 1,
        "expected auto-flush, got sstable_count = {}",
        engine.sstable_count()
    );
    // The memtable should be smaller than the threshold right now.
    // (We don't have a direct getter for approximate_size; the public
    // signal is that memtable_len is well below 40.)
    assert!(
        engine.memtable_len() < 40,
        "auto-flush should have moved entries to SSTable"
    );

    // All keys still retrievable.
    for i in 0..40u32 {
        let key = format!("key_{i:028}");
        let expected = format!("val_{i:028}");
        assert_eq!(
            engine.get(key.as_bytes()).unwrap(),
            Some(expected.into_bytes()),
            "lost key {key}"
        );
    }
}

#[test]
fn auto_flush_does_not_run_below_threshold() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1024 * 1024; // 1 MiB
    let engine = Engine::open(options).unwrap();

    // Small writes — well below 1 MiB.
    for i in 0..50u32 {
        let key = format!("k_{i}");
        let value = format!("v_{i}");
        engine.put(key.as_bytes(), value.as_bytes()).unwrap();
    }
    assert_eq!(engine.sstable_count(), 0, "no auto-flush should have run");
    assert_eq!(engine.memtable_len(), 50);
}

#[test]
fn auto_flushed_data_survives_reopen() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 512;
    {
        let engine = Engine::open(options.clone()).unwrap();
        for i in 0..100u32 {
            let key = format!("k_{i:08}");
            let value = format!("v_{i:08}");
            engine.put(key.as_bytes(), value.as_bytes()).unwrap();
        }
        // Multiple auto-flushes expected.
        assert!(engine.sstable_count() >= 2);
        engine.close().unwrap();
    }

    let engine = Engine::open(options).unwrap();
    for i in 0..100u32 {
        let key = format!("k_{i:08}");
        let expected = format!("v_{i:08}");
        assert_eq!(
            engine.get(key.as_bytes()).unwrap(),
            Some(expected.into_bytes())
        );
    }
}

#[test]
fn auto_flush_handles_oversized_single_value() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 100; // tiny
    let engine = Engine::open(options).unwrap();

    let big = vec![0xAAu8; 4096]; // single value exceeds threshold
    engine.put(b"big", &big).unwrap();

    // The single oversized put should have triggered an auto-flush.
    assert_eq!(engine.sstable_count(), 1);
    assert_eq!(engine.memtable_len(), 0);
    assert_eq!(engine.get(b"big").unwrap(), Some(big));
}

#[test]
fn compaction_drops_tombstones() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1;
    options.l0_compaction_trigger = 2;
    let engine = Engine::open(options).unwrap();

    // Two flushes: one put, one tombstone for the same key. Compaction
    // should fire after the second flush, see the tombstone as the
    // newest entry, and drop it (we're the bottom level, so no older
    // value below could be masked).
    engine.put(b"ghost", b"value").unwrap();
    engine.delete(b"ghost").unwrap();

    let (l0, l1) = engine.level_counts();
    assert_eq!(l0, 0, "compaction should drain L0");
    assert_eq!(
        l1, 0,
        "tombstone-only compaction should produce no L1 output"
    );
    assert_eq!(engine.get(b"ghost").unwrap(), None);
}

#[test]
fn compaction_keeps_newest_value_for_overwritten_key() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1;
    options.l0_compaction_trigger = 3;
    let engine = Engine::open(options).unwrap();

    engine.put(b"k", b"first").unwrap();
    engine.put(b"k", b"second").unwrap();
    engine.put(b"k", b"third").unwrap();

    let (l0, l1) = engine.level_counts();
    assert_eq!(l0, 0);
    assert_eq!(l1, 1);
    assert_eq!(engine.get(b"k").unwrap(), Some(b"third".to_vec()));
}

#[test]
fn compaction_preserves_disjoint_keys_across_l0_files() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1;
    options.l0_compaction_trigger = 4;
    let engine = Engine::open(options).unwrap();

    for i in 0..4u32 {
        let key = format!("disjoint_{i}");
        let value = format!("value_{i}");
        engine.put(key.as_bytes(), value.as_bytes()).unwrap();
    }

    let (l0, l1) = engine.level_counts();
    assert_eq!(l0, 0);
    assert_eq!(l1, 1);
    for i in 0..4u32 {
        let key = format!("disjoint_{i}");
        let expected = format!("value_{i}");
        assert_eq!(
            engine.get(key.as_bytes()).unwrap(),
            Some(expected.into_bytes()),
            "missing key {key}"
        );
    }
}

#[test]
fn multiple_compaction_rounds_keep_state_consistent() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1;
    options.l0_compaction_trigger = 3;
    let engine = Engine::open(options).unwrap();

    // Round 1: write disjoint keys 0..3 → 3 flushes → triggers compaction.
    for i in 0..3u32 {
        engine
            .put(format!("k_{i:02}").as_bytes(), b"old")
            .unwrap();
    }
    let (l0, l1) = engine.level_counts();
    assert_eq!((l0, l1), (0, 1));

    // Round 2: write overlapping keys 2..5 → 3 more flushes → triggers
    // a second compaction merging the new L0s with the existing L1.
    for i in 2..5u32 {
        engine
            .put(format!("k_{i:02}").as_bytes(), b"new")
            .unwrap();
    }
    let (l0, l1) = engine.level_counts();
    assert_eq!((l0, l1), (0, 1));

    // Verify final state: 0,1 keep "old" (only round 1 touched them);
    // 2,3,4 get "new" (round 2 overwrote 2 and added 3,4).
    assert_eq!(engine.get(b"k_00").unwrap(), Some(b"old".to_vec()));
    assert_eq!(engine.get(b"k_01").unwrap(), Some(b"old".to_vec()));
    assert_eq!(engine.get(b"k_02").unwrap(), Some(b"new".to_vec()));
    assert_eq!(engine.get(b"k_03").unwrap(), Some(b"new".to_vec()));
    assert_eq!(engine.get(b"k_04").unwrap(), Some(b"new".to_vec()));
}


#[test]
fn compaction_state_survives_reopen() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1;
    options.l0_compaction_trigger = 3;

    // Phase 1: write exactly enough to trigger one compaction and
    // leave the engine in a freshly-compacted state (L0 = 0, L1 = 1).
    {
        let engine = Engine::open(options.clone()).unwrap();
        for i in 0..3u32 {
            engine
                .put(format!("persist_{i}").as_bytes(), b"value")
                .unwrap();
        }
        let (l0, l1) = engine.level_counts();
        assert_eq!(
            (l0, l1),
            (0, 1),
            "expected freshly-compacted state before close"
        );
        engine.close().unwrap();
    }

    // Phase 2: reopen. State should match phase 1's final state.
    let engine = Engine::open(options).unwrap();
    let (l0_after, l1_after) = engine.level_counts();
    assert_eq!(l0_after, 0, "no L0 files should survive reopen");
    assert_eq!(l1_after, 1, "the one L1 SSTable should survive reopen");

    for i in 0..3u32 {
        let key = format!("persist_{i}");
        assert_eq!(
            engine.get(key.as_bytes()).unwrap(),
            Some(b"value".to_vec()),
            "missing key {key} after reopen"
        );
    }
}

#[test]
fn post_compaction_l0_files_survive_reopen() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1;
    options.l0_compaction_trigger = 3;

    // Phase 1: 3 puts trigger compaction (-> L0=0, L1=1), then 2 more
    // puts add fresh L0 files that DON'T trigger another compaction.
    // Final state: L0 = 2, L1 = 1.
    {
        let engine = Engine::open(options.clone()).unwrap();
        for i in 0..5u32 {
            engine
                .put(format!("key_{i}").as_bytes(), format!("v{i}").as_bytes())
                .unwrap();
        }
        let (l0, l1) = engine.level_counts();
        assert_eq!((l0, l1), (2, 1), "expected 2 fresh L0 + 1 compacted L1");
        engine.close().unwrap();
    }

    // Phase 2: reopen. Discovery must keep both the L1 *and* the
    // newer L0 files. The orphan-cleanup heuristic (drop L0 with
    // id <= surviving L1 id) should leave the newer L0 files alone.
    let engine = Engine::open(options).unwrap();
    let (l0_after, l1_after) = engine.level_counts();
    assert_eq!(l0_after, 2, "newer L0 files must survive reopen");
    assert_eq!(l1_after, 1);

    // All five keys must still read correctly: 0..3 from L1, 3..5 from L0.
    for i in 0..5u32 {
        let key = format!("key_{i}");
        let expected = format!("v{i}");
        assert_eq!(
            engine.get(key.as_bytes()).unwrap(),
            Some(expected.into_bytes())
        );
    }
}

#[test]
fn put_indexed_then_get_returns_vector_bytes() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();

    let embedding = vec![0.1f32, 0.2, 0.3, 0.4];
    engine.put_indexed(b"mem_1", &embedding).unwrap();

    let got = engine.get(b"mem_1").unwrap().expect("key should exist");
    // The bytes are the little-endian f32 encoding.
    let mut expected = Vec::new();
    for x in &embedding {
        expected.extend_from_slice(&x.to_le_bytes());
    }
    assert_eq!(got, expected);
}

#[test]
fn put_indexed_rejects_empty_embedding() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    let err = engine.put_indexed(b"k", &[]).unwrap_err();
    assert!(matches!(err, sastran::Error::InvalidArgument(_)));
}

#[test]
fn put_indexed_rejects_non_finite() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    let err = engine.put_indexed(b"k", &[1.0, f32::NAN]).unwrap_err();
    assert!(matches!(err, sastran::Error::InvalidArgument(_)));
    let err = engine.put_indexed(b"k", &[f32::INFINITY, 0.0]).unwrap_err();
    assert!(matches!(err, sastran::Error::InvalidArgument(_)));
}

#[test]
fn put_indexed_then_delete_masks_correctly() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put_indexed(b"k", &[0.1, 0.2]).unwrap();
    engine.delete(b"k").unwrap();
    assert_eq!(engine.get(b"k").unwrap(), None);
}

#[test]
fn put_indexed_can_be_overwritten_by_put() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put_indexed(b"k", &[0.1, 0.2]).unwrap();
    engine.put(b"k", b"plain bytes now").unwrap();
    assert_eq!(engine.get(b"k").unwrap(), Some(b"plain bytes now".to_vec()));
}

#[test]
fn put_indexed_survives_reopen_via_wal_replay() {
    let dir = tempdir().unwrap();
    let embedding = vec![0.5f32, -0.5, 0.25, -0.25];
    {
        let engine = Engine::open(Options::new(dir.path())).unwrap();
        engine.put_indexed(b"persist", &embedding).unwrap();
        engine.close().unwrap();
    }
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    let got = engine.get(b"persist").unwrap().expect("key should persist");
    let mut expected = Vec::new();
    for x in &embedding {
        expected.extend_from_slice(&x.to_le_bytes());
    }
    assert_eq!(got, expected);
}

#[test]
fn put_indexed_survives_flush_via_sstable() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1; // every write triggers flush
    let engine = Engine::open(options).unwrap();

    let embedding = vec![0.1f32, 0.2, 0.3];
    engine.put_indexed(b"sst_vec", &embedding).unwrap();

    // Force a manual flush in case the threshold didn't fire.
    engine.flush().unwrap();
    let (l0, _) = engine.level_counts();
    assert!(l0 >= 1, "expected at least one SSTable after flush");

    let got = engine.get(b"sst_vec").unwrap().expect("key should be in SST");
    let mut expected = Vec::new();
    for x in &embedding {
        expected.extend_from_slice(&x.to_le_bytes());
    }
    assert_eq!(got, expected);
}

#[test]
fn put_indexed_survives_full_flush_compact_reopen_cycle() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1;
    options.l0_compaction_trigger = 2;

    let embeddings: Vec<Vec<f32>> = (1..=5)
        .map(|i| vec![i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3])
        .collect();

    {
        let engine = Engine::open(options.clone()).unwrap();
        for (i, e) in embeddings.iter().enumerate() {
            let key = format!("vec_{i}");
            engine.put_indexed(key.as_bytes(), e).unwrap();
        }
        engine.close().unwrap();
    }

    let engine = Engine::open(options).unwrap();
    for (i, e) in embeddings.iter().enumerate() {
        let key = format!("vec_{i}");
        let got = engine.get(key.as_bytes()).unwrap().unwrap_or_else(|| {
            panic!("missing key {key} after reopen+compact");
        });
        let mut expected = Vec::new();
        for x in e {
            expected.extend_from_slice(&x.to_le_bytes());
        }
        assert_eq!(got, expected, "vector for {key} corrupted");
    }
}

#[test]
fn nearest_on_empty_engine_returns_empty() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    let r = engine.nearest(&[1.0, 0.0, 0.0], 10).unwrap();
    assert!(r.is_empty());
}

#[test]
fn nearest_returns_self_first() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put_indexed(b"target", &[1.0, 0.0, 0.0]).unwrap();
    engine.put_indexed(b"distractor", &[0.0, 1.0, 0.0]).unwrap();

    let r = engine.nearest(&[1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(r.len(), 2);
    assert_eq!(r[0].key, b"target");
    assert_eq!(r[1].key, b"distractor");
}

#[test]
fn nearest_orders_by_distance() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put_indexed(b"near", &[1.0, 0.01, 0.0]).unwrap();
    engine.put_indexed(b"mid", &[0.7, 0.7, 0.0]).unwrap();
    engine.put_indexed(b"far", &[0.0, 1.0, 0.0]).unwrap();

    let r = engine.nearest(&[1.0, 0.0, 0.0], 3).unwrap();
    assert_eq!(r[0].key, b"near");
    assert_eq!(r[1].key, b"mid");
    assert_eq!(r[2].key, b"far");
    assert!(r[0].distance < r[1].distance);
    assert!(r[1].distance < r[2].distance);
}

#[test]
fn put_indexed_overwrite_updates_search_results() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    // First insert: k is at [1, 0, 0].
    engine.put_indexed(b"k", &[1.0, 0.0, 0.0]).unwrap();
    // Distractor far away.
    engine.put_indexed(b"distractor", &[0.0, 1.0, 0.0]).unwrap();

    let before = engine.nearest(&[1.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(before[0].key, b"k");

    // Now move k to a different region.
    engine.put_indexed(b"k", &[0.0, 0.0, 1.0]).unwrap();
    let after = engine.nearest(&[1.0, 0.0, 0.0], 1).unwrap();
    // The distractor at [0, 1, 0] is now closer to query [1, 0, 0]
    // than the relocated k at [0, 0, 1].
    assert_eq!(after[0].key, b"distractor");
}

#[test]
fn delete_removes_key_from_nearest_results() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put_indexed(b"keep", &[1.0, 0.0, 0.0]).unwrap();
    engine.put_indexed(b"delete_me", &[0.99, 0.01, 0.0]).unwrap();

    let before = engine.nearest(&[1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(before.len(), 2);

    engine.delete(b"delete_me").unwrap();
    let after = engine.nearest(&[1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].key, b"keep");
}

#[test]
fn put_indexed_rejects_dimension_mismatch() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put_indexed(b"first", &[1.0, 0.0, 0.0]).unwrap();
    let err = engine.put_indexed(b"second", &[1.0, 0.0]).unwrap_err();
    assert!(matches!(err, sastran::Error::InvalidArgument(_)));
    // The engine state should be unchanged: only "first" is indexed.
    assert_eq!(engine.vector_count(), 1);
    let r = engine.nearest(&[1.0, 0.0, 0.0], 5).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].key, b"first");
}

#[test]
fn vector_index_recovers_after_reopen() {
    let dir = tempdir().unwrap();
    {
        let engine = Engine::open(Options::new(dir.path())).unwrap();
        engine.put_indexed(b"a", &[1.0, 0.0, 0.0]).unwrap();
        engine.put_indexed(b"b", &[0.0, 1.0, 0.0]).unwrap();
        engine.put_indexed(b"c", &[0.0, 0.0, 1.0]).unwrap();
        engine.close().unwrap();
    }

    let engine = Engine::open(Options::new(dir.path())).unwrap();
    assert_eq!(engine.vector_count(), 3);

    let r = engine.nearest(&[1.0, 0.0, 0.0], 3).unwrap();
    assert_eq!(r.len(), 3);
    assert_eq!(r[0].key, b"a");
}

#[test]
fn vector_index_recovers_through_flush_compact_reopen() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1; // force flush per write
    options.l0_compaction_trigger = 2;

    {
        let engine = Engine::open(options.clone()).unwrap();
        // Five vectors in dim 4: four standard basis vectors + a
        // centroid-ish fifth.
        for i in 0..4 {
            let mut v = vec![0.0f32; 4];
            v[i] = 1.0;
            engine
                .put_indexed(format!("vec_{i}").as_bytes(), &v)
                .unwrap();
        }
        // Fifth vector: a different direction.
        engine
            .put_indexed(b"vec_4", &[0.5f32, 0.5, 0.5, 0.5])
            .unwrap();
        engine.close().unwrap();
    }

    let engine = Engine::open(options).unwrap();
    assert_eq!(engine.vector_count(), 5);

    // Querying each basis direction should find the corresponding
    // basis vector as the top result.
    for i in 0..4 {
        let mut q = vec![0.0f32; 4];
        q[i] = 1.0;
        let r = engine.nearest(&q, 1).unwrap();
        assert!(!r.is_empty());
        let expected_key = format!("vec_{i}");
        assert_eq!(r[0].key, expected_key.as_bytes());
    }
}

#[test]
fn vector_index_recovery_respects_tombstones() {
    let dir = tempdir().unwrap();
    {
        let engine = Engine::open(Options::new(dir.path())).unwrap();
        engine.put_indexed(b"alive", &[1.0, 0.0]).unwrap();
        engine.put_indexed(b"dead", &[0.0, 1.0]).unwrap();
        engine.delete(b"dead").unwrap();
        engine.close().unwrap();
    }

    let engine = Engine::open(Options::new(dir.path())).unwrap();
    // Only "alive" should be in the index after recovery.
    assert_eq!(engine.vector_count(), 1);
    let r = engine.nearest(&[0.0, 1.0], 5).unwrap();
    // Should only find "alive" — "dead" was tombstoned and not
    // re-indexed.
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].key, b"alive");
}

#[test]
fn vector_index_recovery_respects_overwrites() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1; // force flushes

    {
        let engine = Engine::open(options.clone()).unwrap();
        // First write: vector v1 for key "k".
        engine.put_indexed(b"k", &[1.0, 0.0]).unwrap();
        // Overwrite with v2 (likely in a new L0 SSTable).
        engine.put_indexed(b"k", &[0.0, 1.0]).unwrap();
        engine.close().unwrap();
    }

    let engine = Engine::open(options).unwrap();
    assert_eq!(engine.vector_count(), 1);

    // Querying with v2 should find "k"; querying with v1 should
    // find "k" too (it's the only one), but the recovered vector
    // should be v2.
    let r = engine.nearest(&[0.0, 1.0], 1).unwrap();
    assert_eq!(r[0].key, b"k");
    // Distance should be ~0 (we're querying with the stored vector).
    assert!(r[0].distance.abs() < 1e-5);
}

#[test]
fn flush_writes_hnsw_snapshot_file() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put_indexed(b"k", &[1.0, 0.0, 0.0]).unwrap();
    engine.flush().unwrap();

    // Should be exactly one hnsw_*.idx file in the directory now.
    let entries: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("hnsw_")
                && e.file_name().to_string_lossy().ends_with(".idx")
        })
        .collect();
    assert_eq!(entries.len(), 1, "expected exactly one hnsw_*.idx file");
}

#[test]
fn snapshot_file_decodes_back_to_same_index() {
    use sastran::hnsw::HnswIndex;

    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put_indexed(b"a", &[1.0, 0.0, 0.0]).unwrap();
    engine.put_indexed(b"b", &[0.0, 1.0, 0.0]).unwrap();
    engine.put_indexed(b"c", &[0.0, 0.0, 1.0]).unwrap();
    engine.flush().unwrap();

    // Find the snapshot file.
    let snapshot_path = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .map(|n| {
                    let s = n.to_string_lossy();
                    s.starts_with("hnsw_") && s.ends_with(".idx")
                })
                .unwrap_or(false)
        })
        .expect("snapshot file should exist");

    let bytes = std::fs::read(&snapshot_path).unwrap();
    let (restored, _keys, _next_id) = HnswIndex::decode_snapshot(&bytes).unwrap();
    assert_eq!(restored.live_len(), 3);
}

#[test]
fn snapshot_carries_correct_next_sstable_id() {
    use sastran::hnsw::HnswIndex;

    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1; // force flush per write
    let engine = Engine::open(options).unwrap();

    // Three put_indexed calls, each triggering a flush -> SSTable.
    // After the third, next_sst_id should be 3, so the snapshot
    // filename should be hnsw_000003.idx.
    engine.put_indexed(b"a", &[1.0, 0.0]).unwrap();
    engine.put_indexed(b"b", &[0.0, 1.0]).unwrap();
    engine.put_indexed(b"c", &[1.0, 1.0]).unwrap();

    let snapshots: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|s| s.starts_with("hnsw_") && s.ends_with(".idx"))
        .collect();
    assert_eq!(
        snapshots.len(),
        1,
        "expected exactly one snapshot file after old-snapshot cleanup, got {snapshots:?}"
    );
    assert_eq!(snapshots[0], "hnsw_000003.idx");

    // The id inside the file should match the filename.
    let path = dir.path().join(&snapshots[0]);
    let bytes = std::fs::read(&path).unwrap();
    let (_, _keys, next_id) = HnswIndex::decode_snapshot(&bytes).unwrap();
    assert_eq!(next_id, 3);
}

#[test]
fn old_snapshots_cleaned_up_on_new_flush() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1;
    let engine = Engine::open(options).unwrap();

    // First flush -> hnsw_000001.idx.
    engine.put_indexed(b"a", &[1.0, 0.0]).unwrap();
    // Second flush -> hnsw_000002.idx, hnsw_000001.idx deleted.
    engine.put_indexed(b"b", &[0.0, 1.0]).unwrap();

    let snapshots: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|s| s.starts_with("hnsw_") && s.ends_with(".idx"))
        .collect();
    assert_eq!(
        snapshots.len(),
        1,
        "only the latest snapshot should remain, got {snapshots:?}"
    );
    assert_eq!(snapshots[0], "hnsw_000002.idx");
}

#[test]
fn no_snapshot_written_for_writes_that_dont_flush() {
    let dir = tempdir().unwrap();
    // Default 4 MiB memtable; one small write won't trigger a flush.
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put(b"k", b"v").unwrap();

    let snapshots: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|s| s.starts_with("hnsw_") && s.ends_with(".idx"))
        .collect();
    assert!(
        snapshots.is_empty(),
        "no snapshot should be written when no flush occurred, got {snapshots:?}"
    );
}

#[test]
fn snapshot_written_even_when_only_lsm_writes_happen() {
    // Even if we only call `put` (no put_indexed), every flush should
    // still update the HNSW snapshot. The snapshot will just contain
    // an empty index, but it should exist with the right next_id.
    use sastran::hnsw::HnswIndex;
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1;
    let engine = Engine::open(options).unwrap();
    engine.put(b"k", b"v").unwrap();

    let snapshots: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|s| s.starts_with("hnsw_") && s.ends_with(".idx"))
        .collect();
    assert_eq!(snapshots.len(), 1);

    let path = dir.path().join(&snapshots[0]);
    let bytes = std::fs::read(&path).unwrap();
    let (restored, _keys, _) = HnswIndex::decode_snapshot(&bytes).unwrap();
    // No vectors were inserted into the index, but the snapshot must
    // still serialize correctly.
    assert_eq!(restored.live_len(), 0);
}

#[test]
fn snapshot_recovery_basic() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1; // flush per write -> snapshot per write

    {
        let engine = Engine::open(options.clone()).unwrap();
        engine.put_indexed(b"a", &[1.0, 0.0, 0.0]).unwrap();
        engine.put_indexed(b"b", &[0.0, 1.0, 0.0]).unwrap();
        engine.put_indexed(b"c", &[0.0, 0.0, 1.0]).unwrap();
        engine.close().unwrap();
    }

    // Reopen. The latest snapshot should restore all three vectors.
    let engine = Engine::open(options).unwrap();
    assert_eq!(engine.vector_count(), 3);
    let r = engine.nearest(&[1.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(r[0].key, b"a");
}

#[test]
fn snapshot_recovery_applies_post_snapshot_writes() {
    // After the last flush+snapshot, write more vectors that live only
    // in the WAL/memtable. Recovery must apply them on top of the
    // snapshot.
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    // Large memtable: the first writes flush, later writes stay in
    // memtable. Actually we want a controlled split, so flush manually.
    options.memtable_max_size_bytes = 4 * 1024 * 1024;

    {
        let engine = Engine::open(options.clone()).unwrap();
        engine.put_indexed(b"snap_a", &[1.0, 0.0, 0.0]).unwrap();
        engine.put_indexed(b"snap_b", &[0.0, 1.0, 0.0]).unwrap();
        engine.flush().unwrap(); // snapshot now reflects a, b

        // These stay in the WAL/memtable (no further flush).
        engine.put_indexed(b"delta_c", &[0.0, 0.0, 1.0]).unwrap();
        engine.put_indexed(b"delta_d", &[1.0, 1.0, 0.0]).unwrap();
        engine.close().unwrap();
    }

    let engine = Engine::open(options).unwrap();
    // All four should be present: a,b from snapshot; c,d from WAL delta.
    assert_eq!(engine.vector_count(), 4);
    for key in [b"snap_a".as_ref(), b"snap_b", b"delta_c", b"delta_d"] {
        let count = engine
            .nearest(&[1.0, 0.0, 0.0], 4)
            .unwrap()
            .iter()
            .filter(|r| r.key == key)
            .count();
        assert_eq!(count, 1, "missing key {key:?} after recovery");
    }
}

#[test]
fn snapshot_recovery_respects_post_snapshot_delete() {
    // The classic crash scenario: snapshot has a key, then it's deleted,
    // then we crash. Recovery must NOT resurrect the deleted key.
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 4 * 1024 * 1024;

    {
        let engine = Engine::open(options.clone()).unwrap();
        engine.put_indexed(b"doomed", &[1.0, 0.0, 0.0]).unwrap();
        engine.put_indexed(b"survivor", &[0.0, 1.0, 0.0]).unwrap();
        engine.flush().unwrap(); // snapshot reflects both

        engine.delete(b"doomed").unwrap(); // delete lives in WAL delta
        engine.close().unwrap();
    }

    let engine = Engine::open(options).unwrap();
    assert_eq!(engine.vector_count(), 1, "doomed should not be resurrected");
    let r = engine.nearest(&[1.0, 0.0, 0.0], 5).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].key, b"survivor");
}

#[test]
fn snapshot_recovery_respects_post_snapshot_overwrite() {
    // Snapshot has key k pointing at v1. Post-snapshot, k is rewritten
    // to v2. Recovery must reflect v2.
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 4 * 1024 * 1024;

    {
        let engine = Engine::open(options.clone()).unwrap();
        engine.put_indexed(b"k", &[1.0, 0.0, 0.0]).unwrap();
        engine.put_indexed(b"distractor", &[0.0, 1.0, 0.0]).unwrap();
        engine.flush().unwrap(); // snapshot: k at [1,0,0]

        engine.put_indexed(b"k", &[0.0, 0.0, 1.0]).unwrap(); // overwrite
        engine.close().unwrap();
    }

    let engine = Engine::open(options).unwrap();
    assert_eq!(engine.vector_count(), 2);

    // Query at the NEW location of k. k should be the nearest.
    let r = engine.nearest(&[0.0, 0.0, 1.0], 1).unwrap();
    assert_eq!(r[0].key, b"k");
    assert!(r[0].distance.abs() < 1e-5, "should match new vector");

    // Query at the OLD location. k should NOT be there; distractor
    // (at [0,1,0]) is closer to [1,0,0] than k's new spot [0,0,1]...
    // actually both are equidistant. Just confirm k is at its new
    // location, which we did above.
}

#[test]
fn snapshot_recovery_falls_back_on_corrupted_snapshot() {
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1;

    {
        let engine = Engine::open(options.clone()).unwrap();
        engine.put_indexed(b"a", &[1.0, 0.0, 0.0]).unwrap();
        engine.put_indexed(b"b", &[0.0, 1.0, 0.0]).unwrap();
        engine.close().unwrap();
    }

    // Corrupt the snapshot file by truncating it.
    let snapshot_path = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .map(|n| {
                    let s = n.to_string_lossy();
                    s.starts_with("hnsw_") && s.ends_with(".idx")
                })
                .unwrap_or(false)
        })
        .expect("snapshot exists");
    // Overwrite with garbage.
    std::fs::write(&snapshot_path, b"not a valid snapshot").unwrap();

    // Reopen should still succeed via full rebuild from SSTables.
    let engine = Engine::open(options).unwrap();
    assert_eq!(engine.vector_count(), 2, "full rebuild should recover both");
    let r = engine.nearest(&[1.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(r[0].key, b"a");
}

#[test]
fn snapshot_recovery_falls_back_when_no_snapshot() {
    // Write vectors but never flush (so no snapshot), then crash. The
    // WAL replay + full rebuild path must recover them.
    let dir = tempdir().unwrap();
    let options = Options::new(dir.path()); // default 4 MiB; no flush

    {
        let engine = Engine::open(options.clone()).unwrap();
        engine.put_indexed(b"a", &[1.0, 0.0, 0.0]).unwrap();
        engine.put_indexed(b"b", &[0.0, 1.0, 0.0]).unwrap();
        engine.close().unwrap();
    }

    // No snapshot file should exist (nothing flushed).
    let has_snapshot = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| {
            let s = e.file_name().to_string_lossy().to_string();
            s.starts_with("hnsw_") && s.ends_with(".idx")
        });
    assert!(!has_snapshot, "no snapshot should have been written");

    let engine = Engine::open(options).unwrap();
    assert_eq!(engine.vector_count(), 2);
}

#[test]
fn snapshot_recovery_through_compaction() {
    // Force flushes and a compaction, then reopen. The snapshot's
    // next_sstable_id must correctly identify which SSTables post-date
    // it, even across a compaction that renumbers things.
    let dir = tempdir().unwrap();
    let mut options = Options::new(dir.path());
    options.memtable_max_size_bytes = 1;
    options.l0_compaction_trigger = 2;

    {
        let engine = Engine::open(options.clone()).unwrap();
        for i in 0..6 {
            // Unique per-i vectors: a distinct fractional offset on
            // one axis guarantees no two vectors collide (which would
            // make "nearest" ambiguous).
            let v = vec![1.0, 0.1 * (i as f32 + 1.0), 0.0, 0.0];
            engine
                .put_indexed(format!("v{i}").as_bytes(), &v)
                .unwrap();
        }
        engine.close().unwrap();
    }

    let engine = Engine::open(options).unwrap();
    assert_eq!(engine.vector_count(), 6);
    // Every key should be findable.
    for i in 0..6 {
        let key = format!("v{i}");
        let v = vec![1.0, 0.1 * (i as f32 + 1.0), 0.0, 0.0];
        let r = engine.nearest(&v, 1).unwrap();
        assert_eq!(r[0].key, key.as_bytes(), "wrong nearest for {key}");
    }
}

#[test]
fn snapshot_recovery_matches_full_rebuild() {
    // Build an engine two ways: once normally (snapshot recovery),
    // once forcing the fallback (delete the snapshot). Both should
    // recover identical vector sets.
    let dir1 = tempdir().unwrap();
    let dir2 = tempdir().unwrap();
    let mut opt1 = Options::new(dir1.path());
    let mut opt2 = Options::new(dir2.path());
    opt1.memtable_max_size_bytes = 1;
    opt2.memtable_max_size_bytes = 1;

    let vectors: Vec<(String, Vec<f32>)> = (0..20)
        .map(|i| {
            // Give each vector a unique direction: a base axis plus a
            // distinct fractional offset keyed to i, so no two vectors
            // are identical and "nearest to my own vector" is
            // unambiguous.
            let mut v = vec![0.0f32; 8];
            v[i % 8] = 1.0;
            v[(i + 1) % 8] += 0.5;
            v[0] += 0.01 * (i as f32 + 1.0);
            (format!("key_{i}"), v)
        })
        .collect();

    // Engine 1: normal lifecycle.
    {
        let e = Engine::open(opt1.clone()).unwrap();
        for (k, v) in &vectors {
            e.put_indexed(k.as_bytes(), v).unwrap();
        }
        e.close().unwrap();
    }
    // Engine 2: same writes.
    {
        let e = Engine::open(opt2.clone()).unwrap();
        for (k, v) in &vectors {
            e.put_indexed(k.as_bytes(), v).unwrap();
        }
        e.close().unwrap();
    }
    // Force engine 2 into the fallback by deleting its snapshot.
    for entry in std::fs::read_dir(dir2.path()).unwrap().filter_map(|e| e.ok()) {
        let s = entry.file_name().to_string_lossy().to_string();
        if s.starts_with("hnsw_") && s.ends_with(".idx") {
            std::fs::remove_file(entry.path()).unwrap();
        }
    }

    let e1 = Engine::open(opt1).unwrap(); // snapshot recovery
    let e2 = Engine::open(opt2).unwrap(); // full rebuild

    assert_eq!(e1.vector_count(), 20);
    assert_eq!(e2.vector_count(), 20);

    // Both engines return the same nearest neighbor for each key's vec.
    for (k, v) in &vectors {
        let r1 = e1.nearest(v, 1).unwrap();
        let r2 = e2.nearest(v, 1).unwrap();
        assert_eq!(r1[0].key, k.as_bytes());
        assert_eq!(r2[0].key, k.as_bytes());
    }
}

#[test]
fn nearest_filtered_by_key_prefix() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put_indexed(b"user_1:a", &[1.0, 0.0, 0.0]).unwrap();
    engine.put_indexed(b"user_1:b", &[0.9, 0.1, 0.0]).unwrap();
    engine.put_indexed(b"user_2:c", &[1.0, 0.0, 0.0]).unwrap();

    // Filter to user_1 only. Query is closest to user_2:c, but that's
    // filtered out, so the top result should be a user_1 key.
    let r = engine
        .nearest_filtered(&[1.0, 0.0, 0.0], 5, |k| k.starts_with(b"user_1:"))
        .unwrap();
    assert!(!r.is_empty());
    for result in &r {
        assert!(
            result.key.starts_with(b"user_1:"),
            "leaked non-matching key {:?}",
            result.key
        );
    }
    assert_eq!(r.len(), 2, "both user_1 keys should be returned");
}

#[test]
fn nearest_filtered_pass_all_matches_nearest() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put_indexed(b"a", &[1.0, 0.0, 0.0]).unwrap();
    engine.put_indexed(b"b", &[0.0, 1.0, 0.0]).unwrap();
    engine.put_indexed(b"c", &[0.0, 0.0, 1.0]).unwrap();

    let unfiltered = engine.nearest(&[1.0, 0.0, 0.0], 3).unwrap();
    let filtered = engine
        .nearest_filtered(&[1.0, 0.0, 0.0], 3, |_| true)
        .unwrap();
    assert_eq!(unfiltered, filtered);
}

#[test]
fn nearest_filtered_pass_none_returns_empty() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put_indexed(b"a", &[1.0, 0.0, 0.0]).unwrap();
    engine.put_indexed(b"b", &[0.0, 1.0, 0.0]).unwrap();

    let r = engine
        .nearest_filtered(&[1.0, 0.0, 0.0], 5, |_| false)
        .unwrap();
    assert!(r.is_empty());
}

#[test]
fn nearest_filtered_selective_still_finds_k() {
    // Many vectors; a filter that passes only a handful. The adaptive
    // over-query must still find them.
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();

    // 100 vectors. Only those with keys ending in "_keep" pass.
    for i in 0..100 {
        let mut v = vec![0.0f32; 8];
        v[i % 8] = 1.0;
        v[0] += 0.001 * (i as f32 + 1.0); // unique direction
        let key = if i % 25 == 0 {
            format!("vec_{i}_keep")
        } else {
            format!("vec_{i}_drop")
        };
        engine.put_indexed(key.as_bytes(), &v).unwrap();
    }

    // 4 keys pass (i = 0, 25, 50, 75). Ask for 4.
    let r = engine
        .nearest_filtered(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4, |k| {
            k.ends_with(b"_keep")
        })
        .unwrap();
    assert_eq!(r.len(), 4, "should find all 4 matching vectors");
    for result in &r {
        assert!(result.key.ends_with(b"_keep"));
    }
}

#[test]
fn nearest_filtered_respects_distance_order() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    // All pass the filter; verify ordering is preserved.
    engine.put_indexed(b"near", &[1.0, 0.01, 0.0]).unwrap();
    engine.put_indexed(b"mid", &[0.7, 0.7, 0.0]).unwrap();
    engine.put_indexed(b"far", &[0.0, 1.0, 0.0]).unwrap();

    let r = engine
        .nearest_filtered(&[1.0, 0.0, 0.0], 3, |_| true)
        .unwrap();
    assert_eq!(r[0].key, b"near");
    assert_eq!(r[1].key, b"mid");
    assert_eq!(r[2].key, b"far");
}

#[test]
fn nearest_filtered_k_zero_returns_empty() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    engine.put_indexed(b"a", &[1.0, 0.0]).unwrap();
    let r = engine.nearest_filtered(&[1.0, 0.0], 0, |_| true).unwrap();
    assert!(r.is_empty());
}

#[test]
fn nearest_filtered_on_empty_engine() {
    let dir = tempdir().unwrap();
    let engine = Engine::open(Options::new(dir.path())).unwrap();
    let r = engine
        .nearest_filtered(&[1.0, 0.0, 0.0], 5, |_| true)
        .unwrap();
    assert!(r.is_empty());
}