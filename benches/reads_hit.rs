//! Random reads against a pre-populated engine, on keys that exist.
//!
//! Exercises the full read fall-through path: memtable -> L0 -> L1.
//! The bloom filter never returns "definitely absent" here (every
//! lookup is a hit), so the filter just adds a constant per-SSTable
//! cost. The interesting comparison point is reads_miss.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[path = "common/mod.rs"]
mod common;
use common::{fresh_engine, populate, shuffled};

const N_KEYS: usize = 100_000;

fn bench_reads_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("reads_hit");
    group.throughput(Throughput::Elements(1));

    let env = fresh_engine();
    let keys = populate(&env.engine, N_KEYS);
    // Force a flush so the read path actually exercises SSTables.
    env.engine.flush().expect("flush");

    let read_keys = shuffled(keys);
    let mut idx = 0usize;

    group.bench_function(BenchmarkId::new("get", "100k_keys"), |b| {
        b.iter(|| {
            let k = &read_keys[idx % read_keys.len()];
            let v = env.engine.get(k).expect("get");
            // Sanity: every key was inserted, so every read must hit.
            debug_assert!(v.is_some());
            idx += 1;
        });
    });

    group.finish();
}

criterion_group!(benches, bench_reads_hit);
criterion_main!(benches);