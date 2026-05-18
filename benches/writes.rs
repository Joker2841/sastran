//! Sequential write throughput.
//!
//! Measures `put(key, value)` throughput as the engine warms up
//! through several flush/compaction cycles. The first ~80 K records
//! exercise:
//!   - memtable fills and flushes (default 4 MiB)
//!   - L0 accumulates and triggers compactions (default trigger 4)
//!
//! The measurement is a steady-state per-put time. Criterion's
//! built-in warm-up handles the initial empty-engine state.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

#[path = "common/mod.rs"]
mod common;
use common::{fresh_engine, random_keys, random_values};

fn bench_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("writes");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(8));

    group.bench_function(BenchmarkId::new("put", "16B_key_100B_value"), |b| {
        // Fresh engine per criterion iteration would be too expensive
        // (engine open includes directory scans). Instead, we reuse
        // one engine and stream pre-generated keys through it.
        let env = fresh_engine();
        let keys = random_keys(200_000);
        let values = random_values(200_000);
        let mut idx = 0usize;

        b.iter(|| {
            let k = &keys[idx % keys.len()];
            let v = &values[idx % values.len()];
            env.engine.put(k, v).expect("put");
            idx += 1;
        });
    });

    group.finish();
}

criterion_group!(benches, bench_writes);
criterion_main!(benches);