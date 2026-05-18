# sastran

A unified key-value and vector storage substrate, written from scratch in
Rust. Built for AI agent memory: the workload where every "thing the agent
remembers" needs both an exact lookup (`get("user_123_profile")`) and a
similarity search (`nearest(some_embedding, k=10)`) against the same data.

Today that workload typically requires two systems — a KV store *and* a
vector database — kept in sync by hand. `sastran` collapses them into one
crash-safe storage engine.

> **Status:** v0.1.0 ships the LSM substrate: durable, crash-safe key-value
> storage with the full classical pipeline (WAL → memtable → SSTables →
> compaction → bloom filters). The HNSW-based vector index layer is the
> next milestone (v0.2.0).

---

## What's interesting

A few things that distinguish this from the dozens of toy LSMs on GitHub:

- **Durability discipline.** Every write follows the order *WAL append →
  fsync → memtable update*, in that exact order. Reads may only observe
  values that are already durable. The README's "Architecture" section
  walks through why this ordering is the only correct one and what breaks
  if you change it.
- **Property-tested correctness.** The memtable is verified against a
  `BTreeMap` reference model under randomized operation sequences
  (`proptest`). The WAL record format has a property test that flips
  random bits and asserts every corruption is either detected or
  produces observably different output.
- **Crash-safe compaction recovery.** Compaction writes its output to a
  `.tmp` file, atomically renames, and only then deletes the inputs. If
  the process dies between rename and delete, `discover_sstables` on
  next open reconciles the orphans by precise rules (kept the highest L1
  id; drop L0 files with id ≤ surviving L1 id). The invariants are tight
  enough to prove without simulation.
- **Bloom filters give a ~28× speedup on absent-key lookups.** Measured,
  not guessed. See [BENCHMARKS.md](./BENCHMARKS.md).
- **Send + Sync engine, enforced by a compile-time test.** The engine can
  be shared across threads via `Arc<Engine>`. Concurrent reads on SSTables
  use `pread(2)`-style positioned reads with no shared mutable state.
- **Zero unsafe code.** No `unsafe` blocks anywhere in the crate.

## Quick start

```rust
use sastran::{Engine, Options};

let engine = Engine::open(Options::new("/path/to/db"))?;

engine.put(b"hello", b"world")?;
assert_eq!(engine.get(b"hello")?, Some(b"world".to_vec()));

engine.delete(b"hello")?;
assert_eq!(engine.get(b"hello")?, None);

engine.close()?;
# Ok::<(), sastran::Error>(())
```

The engine handles flushes and compactions automatically; the public API
is just `open` / `put` / `get` / `delete` / `flush` / `close`.

## Architecture

```
                    Engine (Mutex<EngineInner>)
                    │
       writes ──────┤
        │           │
        ▼           ▼
   ┌─────────┐  ┌───────────┐
   │  WAL    │  │  Memtable │       in memory
   │ (fsync) │  │ (BTreeMap)│
   └─────────┘  └─────┬─────┘
                      │ flush at size threshold
                      ▼
                ┌──────────┐
                │ L0 SST 0 │   ←── newest L0 (may overlap)
                │ L0 SST 1 │
                │ L0 SST … │
                └─────┬────┘
                      │ compact when L0 ≥ trigger
                      ▼
                ┌──────────┐
                │  L1 SST  │   ←── non-overlapping, deduped, tombstones
                └──────────┘       garbage-collected

  reads ──► memtable → L0 (newest first) → L1
                       │
                       ▼
              bloom filter per SSTable
              short-circuits absent keys
```

The write path is fsync-ordered for durability. The read path falls through
levels newest-first and stops on the first found-or-deleted answer (so
tombstones in newer SSTables correctly mask older puts).

### Why fsync ordering matters

Every write does three things: (A) append to WAL, (B) fsync, (C) update
the memtable. The only correct order is **A → B → C**. Any other ordering
silently leaks pre-durable values to readers, which is the kind of bug
that surfaces only on power loss and corrupts user data. The `Engine::put`
implementation is annotated with this invariant.

## Performance

Measured on WSL2 / ext4, single-threaded, 16-byte keys, 100-byte values:

| Workload | Latency / op | Throughput |
|---|---:|---:|
| `put` (fsync per write) | 436 µs | 2.3 K/sec |
| `get` (key present) | 1.06 µs | 946 K/sec |
| `get` (absent, bloom **on**) | **94 ns** | 10.7 M/sec |
| `get` (absent, bloom **off**) | 2.63 µs | 381 K/sec |
| YCSB-A (50% read / 50% write) | 217 µs | 4.6 K/sec |
| YCSB-B (95% read / 5% write) | 27 µs | 36.5 K/sec |

The 28× speedup on absent-key reads is the clearest demonstration that the
bloom filter implementation is doing what its design claims. Write
throughput is currently fsync-bound (synchronous-durability mode);
[BENCHMARKS.md](./BENCHMARKS.md) explains the trade-off and how group
commit would unlock the rest.

Reproduce with `cargo bench`. HTML reports land in `target/criterion/`.

## What's next

- **HNSW vector index layer (v0.2.0).** Values flagged as embeddings get
  inserted into a per-engine persistent HNSW graph alongside the LSM. A
  new `nearest(query, k)` API serves approximate-nearest-neighbor queries.
  Crash safety extends to the vector index through the same atomic-rename
  + recovery patterns used for SSTables.
- **Group commit.** Batch concurrent writes into one fsync. Expected ~10×
  write throughput improvement at the cost of a small state machine in
  the WAL writer.
- **Multi-level compaction.** Currently L0 → L1; extending to L1 → L2 →
  … is mechanical but requires a manifest file to track per-SSTable key
  ranges. Today's level structure is encoded in filenames.
- **Background compaction.** Compaction runs synchronously inside the
  triggering write; a background thread with a small in-flight queue
  would smooth write-latency spikes.
- **Cross-engine benchmark comparison.** `sastran` is currently benchmarked
  only against itself. Comparing against RocksDB and `sled` requires
  careful methodology (matched durability settings, matched workload
  generator) and is intentionally deferred to a future release.

## Limitations

- Two LSM levels only. The mechanism is sound but only two levels are
  implemented; this is a portfolio-scale simplification.
- Synchronous compaction. Compactions block the triggering write. A
  background thread would mitigate.
- No range scans yet. Point lookups only. The `MergingIterator` primitive
  is in place; the public range-scan API is a small addition.
- No manifest file. Live SSTables are tracked via filenames. A manifest
  would be needed to track per-SSTable key ranges for finer-grained
  compaction.
- WAL corruption note: a corrupted `key_len` field in a WAL record that
  decodes to a value below `MAX_KEY_LEN` but above the remaining file
  size is indistinguishable from a torn-tail truncation. This is a
  fundamental property of length-prefixed formats with leading CRCs and
  is documented inline. LevelDB-style fixed-size blocks would close this
  gap.

## Module layout

```
src/
├── lib.rs               public API surface
├── engine.rs            top-level Engine, lifecycle, flush, compaction
├── error.rs             crate-wide Error enum
├── memtable.rs          in-memory sorted write buffer with tombstones
├── io/
│   ├── mod.rs           Io trait — abstraction for filesystem access
│   └── fs.rs            production filesystem backend (StdFs)
├── wal/
│   ├── mod.rs           write-ahead log
│   ├── record.rs        on-disk record format (CRC + length-prefixed)
│   ├── writer.rs        append + fsync
│   └── reader.rs        replay with torn-tail tolerance
└── sstable/
    ├── mod.rs           SSTable file format constants
    ├── writer.rs        block-based writer with optional bloom filter
    ├── reader.rs        point lookups + forward iteration
    ├── merge.rs         k-way merge with newest-wins deduplication
    └── bloom.rs         double-hashing bloom filter (xxh3_128 + KM trick)
```

Roughly 1,800 LOC of source plus 1,500 LOC of tests (97 tests total,
including property tests).

## Testing

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo bench
```

CI runs all three on every push (when added).

## License

Dual-licensed under MIT or Apache-2.0, at your option. Standard for the
Rust ecosystem.

## Acknowledgements

The architecture draws on the canonical LSM literature: the original
LevelDB design notes, the RocksDB documentation, and the
"Mini-LSM in Rust" walkthrough by Skyzh. The bloom filter uses the
Kirsch-Mitzenmacher double-hashing trick (Random Structures &
Algorithms, 2008). The deterministic-testing-via-Io-trait approach is
inspired by FoundationDB and TigerBeetle, though `sastran`'s implementation
is much simpler.
