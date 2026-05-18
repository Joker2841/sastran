//! Shared helpers for sastran's benchmarks.
//!
//! Each `benches/*.rs` file is a separate Cargo binary; this module
//! is included into them via `#[path = "common/mod.rs"] mod common;`.

#![allow(dead_code)] // Not every bench uses every helper.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use sastran::{Engine, Options};
use std::path::PathBuf;
use tempfile::TempDir;

/// Fixed seed for reproducible benchmark runs.
const SEED: u64 = 0xCAFEBABE_DEADBEEF;

/// Standard benchmark key size, in bytes.
pub const KEY_SIZE: usize = 16;

/// Standard benchmark value size, in bytes.
pub const VALUE_SIZE: usize = 100;

/// An engine plus the tempdir backing it. Drop order matters: the
/// engine must close before the tempdir is cleaned up.
pub struct BenchEngine {
    pub engine: Engine,
    _dir: TempDir,
    pub path: PathBuf,
}

/// Open a fresh engine in a tempdir with default options.
pub fn fresh_engine() -> BenchEngine {
    let dir = tempfile::tempdir().expect("create tempdir");
    let path = dir.path().to_path_buf();
    let engine = Engine::open(Options::new(&path)).expect("open engine");
    BenchEngine {
        engine,
        _dir: dir,
        path,
    }
}

/// Open a fresh engine with bloom filters explicitly disabled (for
/// the read-miss benchmark's bloom-vs-no-bloom comparison).
pub fn fresh_engine_no_bloom() -> BenchEngine {
    let dir = tempfile::tempdir().expect("create tempdir");
    let path = dir.path().to_path_buf();
    let mut opts = Options::new(&path);
    opts.bloom_bits_per_key = 0;
    let engine = Engine::open(opts).expect("open engine");
    BenchEngine {
        engine,
        _dir: dir,
        path,
    }
}

/// Generate `n` deterministic keys of `KEY_SIZE` bytes. The keys are
/// in random byte-value space (not sequential strings) to defeat any
/// accidental locality that a sorted/sequential generator would create.
pub fn random_keys(n: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..n)
        .map(|_| {
            let mut k = vec![0u8; KEY_SIZE];
            rng.fill(&mut k[..]);
            k
        })
        .collect()
}

/// Generate `n` deterministic values of `VALUE_SIZE` bytes.
pub fn random_values(n: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(1));
    (0..n)
        .map(|_| {
            let mut v = vec![0u8; VALUE_SIZE];
            rng.fill(&mut v[..]);
            v
        })
        .collect()
}

/// Populate `engine` with `n` random put operations. The keys are
/// returned for the caller to use during reads.
pub fn populate(engine: &Engine, n: usize) -> Vec<Vec<u8>> {
    let keys = random_keys(n);
    let values = random_values(n);
    for (k, v) in keys.iter().zip(values.iter()) {
        engine.put(k, v).expect("put");
    }
    keys
}

/// Shuffle a slice of keys in place. Useful for ensuring read order
/// has no relationship to write order (which would benefit from
/// sequential bloom-filter / index-cache locality).
pub fn shuffled(mut keys: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(2));
    keys.shuffle(&mut rng);
    keys
}