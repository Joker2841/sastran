//! Mixed read/write workloads modeled after YCSB.
//!
//! - YCSB-A: 50% reads, 50% writes — write-heavy.
//! - YCSB-B: 95% reads, 5% writes — read-heavy (the more realistic mix).
//!
//! Both run against a pre-populated engine so reads have something to
//! find. The mix is randomized per iteration via a precomputed sequence
//! of read/write decisions (keeps the per-iteration overhead minimal).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[path = "common/mod.rs"]
mod common;
use common::{fresh_engine, populate, random_values, shuffled};

const N_KEYS: usize = 100_000;

/// Pre-compute a read/write decision sequence at the given read ratio.
fn op_sequence(n: usize, read_ratio: f64, seed: u64) -> Vec<bool> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_bool(read_ratio)).collect()
}

fn bench_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed");
    group.throughput(Throughput::Elements(1));

    // YCSB-A: 50/50 read/write.
    {
        let env = fresh_engine();
        let keys = populate(&env.engine, N_KEYS);
        env.engine.flush().expect("flush");

        let read_keys = shuffled(keys.clone());
        let values = random_values(N_KEYS);
        let ops = op_sequence(N_KEYS, 0.5, 0xA0_A0);
        let mut idx = 0usize;

        group.bench_function(BenchmarkId::new("ycsb_a", "50r_50w"), |b| {
            b.iter(|| {
                let i = idx % ops.len();
                if ops[i] {
                    let _ = env.engine.get(&read_keys[i]).expect("get");
                } else {
                    env.engine
                        .put(&read_keys[i], &values[i])
                        .expect("put");
                }
                idx += 1;
            });
        });
    }

    // YCSB-B: 95/5 read/write.
    {
        let env = fresh_engine();
        let keys = populate(&env.engine, N_KEYS);
        env.engine.flush().expect("flush");

        let read_keys = shuffled(keys.clone());
        let values = random_values(N_KEYS);
        let ops = op_sequence(N_KEYS, 0.95, 0xB0_B0);
        let mut idx = 0usize;

        group.bench_function(BenchmarkId::new("ycsb_b", "95r_5w"), |b| {
            b.iter(|| {
                let i = idx % ops.len();
                if ops[i] {
                    let _ = env.engine.get(&read_keys[i]).expect("get");
                } else {
                    env.engine
                        .put(&read_keys[i], &values[i])
                        .expect("put");
                }
                idx += 1;
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_mixed);
criterion_main!(benches);