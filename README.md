# sastran

A unified key-value and vector storage engine, written from scratch in Rust.
Built for AI agent memory: the workload where every "thing the agent
remembers" needs both an exact lookup (`get("user_123_profile")`) and a
similarity search (`nearest(some_embedding, 10)`) against the same data.

Today that workload typically requires two systems — a KV store *and* a
vector database — kept in sync by hand. `sastran` collapses them into one
crash-safe engine: a log-structured merge tree for keys and values, with an
HNSW vector index layered on top, sharing the same durability and recovery
machinery.

> **Status:** v0.2.0. The LSM substrate (WAL -> memtable -> SSTables ->
> compaction -> bloom filters) and the HNSW vector index (insert, search,
> delete with edge repair, configurable metrics, filtered search,
> crash-safe snapshots) are both implemented and tested. ~190 tests,
> zero `unsafe`.

---

## What's interesting

What distinguishes this from the many toy LSMs and standalone HNSW crates on
GitHub:

- **It's actually unified.** `put`, `put_indexed`, `get`, `delete`, and
  `nearest` all run through one engine, share one durability ordering, and
  recover together. A vector insert is durable in the WAL before it's
  acknowledged, exactly like a key-value write.
- **Crash-safe vector recovery.** The HNSW graph is snapshotted to disk on
  flush (atomic-rename, CRC-checked). On open, the engine loads the latest
  snapshot and replays only the post-snapshot delta - O(memtable + recent
  SSTables) instead of rebuilding the whole graph. A corrupted or missing
  snapshot falls back to a full rebuild from the LSM, which is always the
  source of truth.
- **Real HNSW deletes.** Not soft tombstones - deleted nodes' neighbors are
  re-linked (edge repair) so graph connectivity and recall hold. Recall@10
  stays above 0.80 after deleting 30% of a 2,000-vector index.
- **Bloom filters give a ~28x speedup on absent-key lookups** (94 ns vs
  2.63 us). Measured, not guessed.
- **Property-tested correctness.** The memtable is checked against a
  `BTreeMap` reference model under randomized operations; the WAL record
  format is fuzzed for corruption detection.
- **Honest measurement.** Quantization retains 96.2% of recall at 4x
  compression on realistic data; filtered queries cost ~5x unfiltered for a
  structural reason that's documented, not hidden. See
  [BENCHMARKS.md](./BENCHMARKS.md).
- **Zero unsafe code.**

## Quick start

```rust
use sastran::{Engine, Options};

let engine = Engine::open(Options::new("/path/to/db"))?;

// Key-value.
engine.put(b"user_42:profile", b"{...}")?;
assert_eq!(engine.get(b"user_42:profile")?, Some(b"{...}".to_vec()));

// Vector: store an embedding under a key.
engine.put_indexed(b"user_42:memory_99", &[0.1, 0.2, 0.3, /* ... */])?;

// Similarity search: k nearest embeddings to a query.
let hits = engine.nearest(&query_embedding, 10)?;
for hit in &hits {
    println!("{:?} at distance {}", hit.key, hit.distance);
}

// Filtered search: nearest, restricted to keys matching a predicate.
let user_hits = engine.nearest_filtered(
    &query_embedding,
    10,
    |key| key.starts_with(b"user_42:"),
)?;

engine.close()?;
# Ok::<(), sastran::Error>(())
```

`put_indexed` writes the embedding to the LSM at full precision *and* inserts
it into the HNSW index. `get` returns the stored bytes; `nearest` returns
keys ranked by similarity. A `delete` removes a key from both the LSM and the
index.

## Architecture

```
                 Engine
                 |-- Mutex<EngineInner>          (LSM: serial mutations)
                 +-- Arc<RwLock<VectorIndex>>     (HNSW: concurrent reads)

   put / put_indexed / delete
        |
        v
   +---------+   +-----------+        +--------------+
   |  WAL    |   |  Memtable |        | HNSW index   |
   | (fsync) |   | (BTreeMap)|        | (graph +     |
   +---------+   +-----+-----+        |  key<->node) |
                       | flush         +------+-------+
                       v                      | snapshot on flush
                 +----------+                 v
                 | L0 SSTs  |           hnsw_<id>.idx
                 |   v compact            (atomic rename, CRC)
                 | L1 SST   |
                 +----+-----+
   get ----> memtable -> L0 -> L1   (bloom filter per SSTable)
   nearest --> HNSW graph traversal (read lock; concurrent-safe)
```

Writes are fsync-ordered for durability (WAL append -> fsync -> memtable).
The vector index is locked separately from the LSM so concurrent `nearest`
queries don't block writes; the two locks are always acquired LSM-first to
avoid deadlock.

## Performance

Measured on WSL2 / ext4, single-threaded, release build. Full methodology
and interpretation in [BENCHMARKS.md](./BENCHMARKS.md).

**Key-value (16 B keys, 100 B values):**

| Operation | Latency | Throughput |
|---|---:|---:|
| `put` (fsync per write) | 436 us | 2.3 K/sec |
| `get` (present) | 1.06 us | 946 K/sec |
| `get` (absent, bloom on) | **94 ns** | 10.7 M/sec |
| `get` (absent, bloom off) | 2.63 us | 381 K/sec |

**Vector (128-dim embeddings, 5 K-vector index):**

| Operation | Latency | Throughput |
|---|---:|---:|
| `put_indexed` | 617 us | 1.6 K/sec |
| `nearest` (k=10) | 111 us | 9.0 K/sec |
| `nearest_filtered` (k=10, ~10% selective) | 563 us | 1.8 K/sec |

**Quantization (8-bit scalar, clustered data):** 96.2% of full-precision
recall@10 at 4x compression.

Reproduce with `cargo bench` and
`cargo run --example quantization_recall --release`.

## What's next

- **Native u8 quantized storage in the HNSW.** The quantizer is implemented
  and measured, but the live index still stores f32. Wiring native u8
  storage in would realize the 4x memory reduction in production.
- **Predicate-aware filtered traversal.** Filtering is currently post-hoc
  (search, then discard non-matches). Pushing the predicate into the graph
  walk would make highly selective filters fast.
- **Group commit** for write throughput (batch fsyncs).
- **Background compaction** to smooth write-latency spikes.
- **Multi-level compaction** beyond the current two levels (needs a
  manifest).
- **Cross-engine benchmarks** against RocksDB / Qdrant with matched
  durability settings.

## Limitations

- Two LSM levels only; the mechanism generalizes but isn't extended yet.
- Synchronous compaction and synchronous snapshot writes block the
  triggering operation.
- No range scans (point lookups only; the merge primitive exists).
- HNSW quantization is measured but not wired into the live index.
- Filtered search is post-hoc; very selective filters degrade toward a scan.
- No manifest file; live SSTables are tracked via filenames.

## Module layout

```
src/
|-- engine.rs      top-level Engine: lifecycle, put/get/delete,
|                  put_indexed/nearest/nearest_filtered, recovery
|-- memtable.rs    sorted write buffer (Put / Vector / Tombstone)
|-- wal/           write-ahead log (Put / Vector / Delete records)
|-- sstable/       block SSTables: writer, reader, merge, bloom
|-- hnsw/          vector index: graph (index.rs) + snapshot format
|-- quantize.rs    8-bit scalar quantizer (standalone, measured)
|-- io/            Io trait + StdFs backend (for fault injection)
+-- error.rs       crate-wide error type
```

~3,000 lines of source, ~2,500 of tests (~190 tests including property
tests and recall benchmarks).

## Testing

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo bench
cargo run --example quantization_recall --release
```

## License

Dual-licensed under MIT or Apache-2.0, at your option.

## Acknowledgements

LSM design from the LevelDB and RocksDB literature. HNSW from Malkov &
Yashunin (2018); the bloom filter uses the Kirsch-Mitzenmacher double-hashing
trick. The Io-trait-for-deterministic-testing approach is inspired by
FoundationDB and TigerBeetle.