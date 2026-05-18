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