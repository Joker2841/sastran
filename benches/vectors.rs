//! Vector-operation benchmarks: put_indexed, nearest, nearest_filtered.
//!
//! - put_indexed: insert throughput (WAL fsync + memtable + HNSW
//!   insert, plus amortized flush/snapshot). The index grows during
//!   measurement, so the reported figure reflects inserts into an
//!   index of increasing size.
//! - nearest: ANN query latency against a pre-populated 5k-vector
//!   index. No fsync; pure read + graph traversal.
//! - nearest_filtered: same, with a 10%-selective key predicate, so
//!   the adaptive over-query path is exercised.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[path = "common/mod.rs"]
mod common;
use common::fresh_engine;

const DIM: usize = 128;
const POPULATE_N: usize = 5_000;
const SEED: u64 = 0x7EC1_107A;

fn random_embedding(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-1.0..1.0_f32)).collect()
}

fn bench_put_indexed(c: &mut Criterion) {
    // Pre-generate a large pool so we never wrap within a measurement
    // window (=> all fresh inserts, never overwrites).
    let mut rng = StdRng::seed_from_u64(SEED);
    let keys: Vec<Vec<u8>> = (0..100_000).map(|i| format!("vec_{i}").into_bytes()).collect();
    let embeddings: Vec<Vec<f32>> =
        (0..100_000).map(|_| random_embedding(&mut rng, DIM)).collect();

    let mut group = c.benchmark_group("vectors_put_indexed");
    group.throughput(Throughput::Elements(1));
    group.sample_size(30); // each insert is fsync-bound; keep runtime sane

    group.bench_function(BenchmarkId::new("put_indexed", "dim128"), |b| {
        let env = fresh_engine();
        let mut idx = 0usize;
        b.iter(|| {
            let k = &keys[idx % keys.len()];
            let e = &embeddings[idx % embeddings.len()];
            env.engine.put_indexed(k, e).expect("put_indexed");
            idx += 1;
        });
    });

    group.finish();
}

fn bench_nearest(c: &mut Criterion) {
    // Populate ONCE, before bench_function, so it runs exactly once.
    let env = fresh_engine();
    let mut rng = StdRng::seed_from_u64(SEED);
    for i in 0..POPULATE_N {
        let e = random_embedding(&mut rng, DIM);
        env.engine
            .put_indexed(format!("vec_{i}").as_bytes(), &e)
            .expect("populate");
    }
    let queries: Vec<Vec<f32>> = (0..1000).map(|_| random_embedding(&mut rng, DIM)).collect();

    let mut group = c.benchmark_group("vectors_nearest");
    group.throughput(Throughput::Elements(1));
    let mut qi = 0usize;
    group.bench_function(BenchmarkId::new("nearest_k10", "5k_dim128"), |b| {
        b.iter(|| {
            let q = &queries[qi % queries.len()];
            let r = env.engine.nearest(q, 10).expect("nearest");
            qi += 1;
            r
        });
    });
    group.finish();
}

fn bench_nearest_filtered(c: &mut Criterion) {
    let env = fresh_engine();
    let mut rng = StdRng::seed_from_u64(SEED);
    for i in 0..POPULATE_N {
        let e = random_embedding(&mut rng, DIM);
        // 10 users => the user_0: filter passes ~10% of vectors.
        let key = format!("user_{}:vec_{i}", i % 10);
        env.engine.put_indexed(key.as_bytes(), &e).expect("populate");
    }
    let queries: Vec<Vec<f32>> = (0..1000).map(|_| random_embedding(&mut rng, DIM)).collect();

    let mut group = c.benchmark_group("vectors_nearest_filtered");
    group.throughput(Throughput::Elements(1));
    let mut qi = 0usize;
    group.bench_function(
        BenchmarkId::new("nearest_filtered_k10", "10pct_selective"),
        |b| {
            b.iter(|| {
                let q = &queries[qi % queries.len()];
                let r = env
                    .engine
                    .nearest_filtered(q, 10, |k| k.starts_with(b"user_0:"))
                    .expect("nearest_filtered");
                qi += 1;
                r
            });
        },
    );
    group.finish();
}

criterion_group!(benches, bench_put_indexed, bench_nearest, bench_nearest_filtered);
criterion_main!(benches);