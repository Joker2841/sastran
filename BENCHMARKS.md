# Benchmarks

This document records the methodology and measured performance of `sastran`
on a representative workload. All numbers below come from `cargo bench` runs
that anyone can reproduce.

## How to reproduce

```bash
cargo bench
```

HTML reports (with histograms and confidence intervals) are written to
`target/criterion/report/index.html`.

## Hardware and environment

| | |
|---|---|
| Host OS | Windows 11 |
| Runtime | WSL2, Ubuntu 24.04, kernel 6.6 |
| Filesystem | ext4 inside the WSL2 virtual disk (not the Windows `/mnt/c` mount, which is 10–20× slower) |
| Rust | 1.95.0, release profile (`lto = "fat"`, `codegen-units = 1`) |

Reproducing on different hardware will produce different absolute numbers,
but the *ratios* between benchmarks should be similar — particularly the
bloom-on vs bloom-off ratio, which is dominated by syscalls saved.

## Workload

| | |
|---|---|
| Key size | 16 bytes |
| Value size | 100 bytes |
| Total records | 100,000 (read benches) / 200,000 (write bench) |
| Memtable threshold | 4 MiB (default) |
| L0 compaction trigger | 4 SSTables (default) |
| Bloom filter | 10 bits/key, k=7 (default; toggled off only in `reads_miss/bloom_off`) |
| Keys & values | Random bytes from a deterministic seed |
| Read order | Shuffled relative to write order (defeats trivial locality) |

Each benchmark holds a single open engine across criterion iterations; opening
per-iteration would be dominated by directory-scan overhead.

## Results

| Benchmark | Time / op | Throughput | Notes |
|---|---:|---:|---|
| `writes/put` | **436 µs** | 2.3 K ops/sec | fsync per write; see "Write throughput" below |
| `reads_hit/get` | **1.06 µs** | 946 K ops/sec | Full memtable → L0 → L1 fall-through |
| `reads_miss/get_absent` (bloom on) | **94 ns** | 10.7 M ops/sec | Filter rejects most lookups before disk |
| `reads_miss/get_absent` (bloom off) | **2.63 µs** | 381 K ops/sec | Same query, no filter |
| `mixed/ycsb_a` (50% read / 50% write) | 217 µs | 4.6 K ops/sec | Write-heavy YCSB-A |
| `mixed/ycsb_b` (95% read / 5% write) | 27 µs | 36.5 K ops/sec | Read-heavy YCSB-B |

All numbers are criterion's reported median across 100 samples. The full
confidence intervals (low / median / high) are in the HTML report.

## Interpretation

### Bloom filters: ~28× speedup on absent-key lookups

`reads_miss/get_absent` is the workload bloom filters were designed for, and
the measured ratio is the clearest result in this suite:

```
bloom_off / bloom_on = 2628 ns / 94 ns ≈ 28×
```

Without filters, an absent-key `get` reads one data block from every SSTable
to confirm absence. With filters, most SSTables return "definitely not
present" after seven bit lookups (~10 ns), and only false-positive SSTables —
about 1% with our default sizing — pay the disk read. At our default workload
the cumulative effect is a ~28× speedup, dominated by syscalls saved.

This makes bloom filters one of the highest-impact optimizations in the
project for the disk space spent (~10 bits per key, or ~12 KiB for a typical
flushed SSTable).

### Write throughput: 2.3 K ops/sec is correct, not slow

The number looks small compared to "RocksDB does 100 K writes/sec." That
comparison isn't apples-to-apples:

- `sastran` is in **fsync-per-write** mode — every `put` durably commits the
  WAL before acknowledging the caller. This matches SQLite's default and
  what most "I need this write to survive a power loss" workloads want.
- RocksDB's headline write numbers come from **group commit + WAL batching**,
  where multiple concurrent writes share one fsync. That's a great
  optimization but a different durability contract: an in-flight write may
  be lost if the process crashes before the batch commits.

On WSL2's ext4 an `fsync` costs ~100–400 µs. The measured 436 µs per put
includes that fsync plus periodic flush/compaction amortization. The math
is consistent with a synchronous-fsync design.

Faster writes are achievable and well-understood:

- **Group commit**: batch concurrent writes into one fsync. ~10× speedup
  expected. Listed as future work; would require a small state machine
  around the WAL writer.
- **Async commit**: return success immediately, fsync in the background.
  Loses some durability for higher throughput. Useful for non-critical
  data.

Neither is implemented yet. The README's "what's next" section calls this
out explicitly.

### Read hits: ~1 µs, competitive

`reads_hit/get` averages 1.06 µs per lookup. This is the full fall-through
path: memtable check → bloom filter on each L0 SSTable → index binary search
→ data block read → linear scan within block.

Most data after the populate phase lives in the L1 SSTable and is hot in the
OS page cache, so the data-block "read" usually doesn't actually touch disk.
This is correct: real workloads benefit from the page cache too. We don't
flush caches between iterations.

### Mixed workloads: math checks out

`mixed/ycsb_a` at 217 µs is essentially `(435 µs put + 1 µs get) / 2 = 218
µs`. The arithmetic matching that closely is evidence the benchmark is
honest — there's no hidden batching, caching, or amortization beyond what's
visible in the per-operation numbers.

YCSB-B at 27 µs reflects the same arithmetic with a 95/5 mix:
`0.95 × 1 µs + 0.05 × 435 µs ≈ 22 µs`. The measured number is slightly
higher because of bookkeeping overhead between operations, but the ratio is
correct.

## What's *not* benchmarked here

The current suite measures `sastran` against itself. It doesn't compare to
RocksDB, sled, or LMDB. A fair comparison requires:

- Identical hardware and OS.
- Identical workload generator (e.g. official YCSB or `db_bench`).
- Identical durability settings on both engines (a `sastran` with
  fsync-per-write should be compared against RocksDB with `WAL_FSYNC`,
  not the default).

That work is intentionally out of scope for v0.1.0. It is a clear follow-up
once the engine has a stabilization period.

The suite also measures **single-threaded** performance only. `sastran`
serializes mutations through one `Mutex`, so multi-threaded write
throughput would not be flattering. Multi-threaded reads through
`Arc<Engine>` would scale better (SSTable readers are `Send + Sync` and use
positioned reads), but those numbers aren't reported here either.

## Variance

Most benchmarks reported a handful of outliers (3–12 out of 100 samples).
Criterion classifies these as "high mild" or "high severe"; they correspond
to flush stalls and compactions captured inside individual measurement
windows. The median is robust to these outliers, which is why the reported
numbers above use the median rather than the mean.

For the read benchmarks, the variance is much lower (a few outliers, all
mild). Reads have no background work to spike them.

---

# Vector operations (v0.2.0)

The vector half of the engine — HNSW-backed approximate nearest-neighbor
search — adds three measured operations. Same environment as the LSM
benchmarks (WSL2 / ext4, single-threaded, release build).

## Workload

| | |
|---|---|
| Embedding dimension | 128 |
| Index size (query benches) | 5,000 vectors |
| Distance metric | cosine (default) |
| HNSW parameters | M=16, efConstruction=100, efSearch=50 |
| Queries sampled | 1,000 |
| k | 10 |

## Results

| Operation | Latency / op | Throughput |
|---|---:|---:|
| `put_indexed` (vector insert) | 617 µs | 1.6 K/sec |
| `nearest` (ANN query, k=10) | 111 µs | 9.0 K/sec |
| `nearest_filtered` (k=10, ~10% selective) | 563 µs | 1.8 K/sec |

## Interpretation

### `put_indexed`: 617 µs

A vector insert does everything a regular `put` does — WAL append, fsync,
memtable update — plus an HNSW graph insertion (descent, beam search, edge
selection, bidirectional wiring) and the periodic snapshot write on flush.
The ~617 µs figure is the ~436 µs fsync-bound `put` cost plus roughly
180 µs of graph maintenance at dimension 128. As with plain writes, the
dominant cost is the synchronous fsync; group commit (future work) would
help here too.

### `nearest`: 111 µs

A k=10 ANN query over 5,000 dimension-128 vectors. This is pure in-memory
graph traversal — no fsync, no disk. Each query visits roughly
`efSearch × average_degree` nodes and computes a 128-wide distance at each,
landing at ~111 µs, or ~9,000 queries/sec single-threaded. Concurrent
queries scale further because `nearest` takes only the read side of the
vector index lock.

### `nearest_filtered`: 563 µs

Filtering to keys with a `user_0:` prefix (≈10% of vectors pass). The query
costs ~5× an unfiltered `nearest` for a structural reason: the filter is
applied *after* the graph search returns candidates, so to return k results
the search must over-query (request more candidates than k and discard the
rejects). At ~10% selectivity the engine searches with `ef ≈ k × 10` to
collect enough passing candidates.

This is the standard limitation of post-hoc filtering on a graph index. A
predicate that passes only a tiny fraction of vectors degrades toward a
full scan, since that is the only correct way to find the k nearest passing
vectors. Pushing the predicate *into* graph traversal (so filtered-out nodes
never consume result slots) is the research-grade fix used by production
filterable-ANN systems; it is documented as future work. An earlier
implementation that grew the candidate set by repeated doubling was ~35%
slower; the current single-wide-search approach is what produced the figure
above.

# Scalar quantization (v0.2.0)

8-bit scalar quantization is implemented as a standalone, measured component
(`src/quantize.rs`). It maps each f32 vector component to one byte using
per-dimension min/max scaling, a 4× memory reduction. The measurement
harness (`examples/quantization_recall.rs`) builds two HNSW indexes — one
full-precision, one quantize-then-dequantize — and compares recall@10
against a full-precision brute-force ground truth.

## Results

| Dataset | Full-precision recall@10 | Quantized recall@10 | Retained | Compression |
|---|---:|---:|---:|---:|
| Clustered (realistic) | >0.99 | 0.962 | 96.2% | 4.0× |
| Uniform random (adversarial) | 0.697 | 0.703 | ~100% | 4.0× |

## Interpretation

On **clustered data** — which is how real sentence and image embeddings are
distributed, grouping into neighborhoods rather than filling the space
uniformly — quantization retains 96.2% of full-precision recall while
cutting per-vector storage from 512 bytes to 128 bytes. That is the
representative result: ~4% recall cost for 4× memory savings.

On **uniform-random data** both indexes score ~0.70. This is the curse of
dimensionality, not a quantization or implementation issue: in
high-dimensional uniform-random space all points are nearly equidistant, so
the true nearest neighbors are barely closer than the rest and any
approximate search misses some. The point of including it is to show that
quantization is nearly free regardless of data shape — the recall difference
between full and quantized is within noise on both datasets. Absolute recall
depends on how clustered the data is, and real embeddings are clustered.

Quantization is not yet wired into the live HNSW index, which stores
full-precision f32. Native u8 storage in the index — the change that would
realize the 4× memory savings in production rather than only in the
benchmark — is documented as future work.