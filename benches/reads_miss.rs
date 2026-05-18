//! Random reads against keys that don't exist.
//!
//! This is the read pattern bloom filters were designed to accelerate.
//! Without filters, every SSTable must read one data block to confirm
//! absence. With filters, most SSTables skip the disk read entirely.
//!
//! Compares two configurations:
//!   - "bloom_on"  (bloom_bits_per_key = 10, default)
//!   - "bloom_off" (bloom_bits_per_key = 0)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[path = "common/mod.rs"]
mod common;
use common::{
    fresh_engine, fresh_engine_no_bloom, populate, BenchEngine, KEY_SIZE,
};

const N_PRESENT: usize = 100_000;
const N_ABSENT: usize = 100_000;

/// Generate keys deliberately *not* in the populated set by using a
/// different seed. With 16-byte random keys, the collision probability
/// against 100K populated keys is astronomically low; we treat
/// collisions as a no-op for measurement purposes.
fn absent_keys() -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(0xAB5E47DEAD_BEEF_u64);
    (0..N_ABSENT)
        .map(|_| {
            let mut k = vec![0u8; KEY_SIZE];
            rng.fill(&mut k[..]);
            k
        })
        .collect()
}

fn populated(env: BenchEngine) -> BenchEngine {
    populate(&env.engine, N_PRESENT);
    env.engine.flush().expect("flush");
    env
}

fn bench_reads_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("reads_miss");
    group.throughput(Throughput::Elements(1));

    let absent = absent_keys();

    // With bloom filters on.
    {
        let env = populated(fresh_engine());
        let mut idx = 0usize;
        group.bench_function(BenchmarkId::new("get_absent", "bloom_on"), |b| {
            b.iter(|| {
                let k = &absent[idx % absent.len()];
                let v = env.engine.get(k).expect("get");
                debug_assert!(v.is_none());
                idx += 1;
            });
        });
    }

    // With bloom filters off.
    {
        let env = populated(fresh_engine_no_bloom());
        let mut idx = 0usize;
        group.bench_function(BenchmarkId::new("get_absent", "bloom_off"), |b| {
            b.iter(|| {
                let k = &absent[idx % absent.len()];
                let v = env.engine.get(k).expect("get");
                debug_assert!(v.is_none());
                idx += 1;
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_reads_miss);
criterion_main!(benches);