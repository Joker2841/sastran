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
