//! Recall and memory tradeoff for 8-bit scalar quantization.
//!
//! Builds two HNSW indexes on the same synthetic dataset:
//!   1. full precision (baseline)
//!   2. quantize-then-dequantize (simulating what a quantized index
//!      would store and compare against)
//!
//! Measures recall@10 for each against a brute-force ground truth,
//! and reports the memory compression ratio. This is a measurement
//! harness, not a microbenchmark, so it prints results rather than
//! using criterion's timing machinery.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sastran::hnsw::{HnswIndex, HnswParams, NodeId};
use sastran::ScalarQuantizer;

const DIM: usize = 128;
const N: usize = 5000;
const QUERIES: usize = 200;
const K: usize = 10;

fn random_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-1.0..1.0_f32)).collect()
}

/// Generate `n` vectors grouped into `clusters` Gaussian-ish blobs.
/// This mimics real embeddings, which form neighborhoods rather than
/// filling the space uniformly — and on which ANN recall is high.
fn clustered_dataset(rng: &mut StdRng, n: usize, dim: usize, clusters: usize) -> Vec<Vec<f32>> {
    // Pick cluster centers.
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0_f32)).collect())
        .collect();
    (0..n)
        .map(|_| {
            let c = &centers[rng.gen_range(0..clusters)];
            // Add small jitter around the center.
            c.iter()
                .map(|&x| x + rng.gen_range(-0.1..0.1_f32))
                .collect()
        })
        .collect()
}

/// Brute-force top-K by cosine distance, returning dataset indices.
fn brute_force(query: &[f32], dataset: &[Vec<f32>], k: usize) -> Vec<usize> {
    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 2.0;
        }
        1.0 - dot / (na * nb)
    }
    let mut scored: Vec<(usize, f32)> = dataset
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_distance(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn build_index(vectors: &[Vec<f32>]) -> (HnswIndex, Vec<NodeId>) {
    let mut idx = HnswIndex::new(HnswParams::default());
    let ids = vectors.iter().map(|v| idx.insert(v).unwrap()).collect();
    (idx, ids)
}

fn measure_recall(
    idx: &HnswIndex,
    ids: &[NodeId],
    ground_truth_dataset: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> f64 {
    let mut total = 0.0;
    for q in queries {
        let truth: std::collections::HashSet<NodeId> = brute_force(q, ground_truth_dataset, K)
            .into_iter()
            .map(|i| ids[i])
            .collect();
        let got = idx.search(q, K).unwrap();
        let hits = got.iter().filter(|(id, _)| truth.contains(id)).count();
        total += hits as f64 / K as f64;
    }
    total / queries.len() as f64
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0x5CA1AB1E);

    // Full-precision dataset.
    let dataset: Vec<Vec<f32>> = (0..N).map(|_| random_vector(&mut rng, DIM)).collect();
    let queries: Vec<Vec<f32>> = (0..QUERIES).map(|_| random_vector(&mut rng, DIM)).collect();

    // Train the quantizer on the dataset, then produce the
    // quantize→dequantize version of every vector.
    let quantizer = ScalarQuantizer::train(&dataset).unwrap();
    let quantized_dataset: Vec<Vec<f32>> = dataset
        .iter()
        .map(|v| quantizer.round_trip(v).unwrap())
        .collect();

    // Build both indexes.
    let (full_idx, full_ids) = build_index(&dataset);
    let (quant_idx, quant_ids) = build_index(&quantized_dataset);

    // Ground truth is always the full-precision dataset (the "true"
    // neighbors a user cares about). We measure how well each index
    // recovers those true neighbors.
    let full_recall = measure_recall(&full_idx, &full_ids, &dataset, &queries);
    let quant_recall = measure_recall(&quant_idx, &quant_ids, &dataset, &queries);

    let full_bytes = DIM * 4;
    let quant_bytes = DIM; // one byte per dim
    let ratio = full_bytes as f64 / quant_bytes as f64;

    println!("=== Scalar Quantization: recall / memory tradeoff ===");
    println!("dataset: {N} vectors, dim {DIM}, {QUERIES} queries, k={K}");
    println!();
    println!("per-vector storage:");
    println!("  full precision:  {full_bytes} bytes");
    println!("  quantized (u8):  {quant_bytes} bytes");
    println!("  compression:     {ratio:.1}x");
    println!();
    println!("recall@{K} (vs full-precision brute-force ground truth):");
    println!("  full-precision index:  {full_recall:.4}");
    println!("  quantized index:       {quant_recall:.4}");
    println!("  recall retained:       {:.1}%", quant_recall / full_recall * 100.0);

    // --- Clustered dataset (representative of real embeddings) ---
    println!();
    println!("--- clustered dataset (mimics real embeddings) ---");
    let clustered: Vec<Vec<f32>> = clustered_dataset(&mut rng, N, DIM, 50);
    let cl_queries: Vec<Vec<f32>> = {
        // Queries near random cluster centers, with jitter.
        let centers: Vec<&Vec<f32>> = clustered.iter().take(50).collect();
        (0..QUERIES)
            .map(|i| {
                let c = centers[i % centers.len()];
                c.iter().map(|&x| x + rng.gen_range(-0.1..0.1_f32)).collect()
            })
            .collect()
    };
    let cl_quantizer = ScalarQuantizer::train(&clustered).unwrap();
    let cl_quantized: Vec<Vec<f32>> = clustered
        .iter()
        .map(|v| cl_quantizer.round_trip(v).unwrap())
        .collect();
    let (cl_full_idx, cl_full_ids) = build_index(&clustered);
    let (cl_quant_idx, cl_quant_ids) = build_index(&cl_quantized);
    let cl_full_recall = measure_recall(&cl_full_idx, &cl_full_ids, &clustered, &cl_queries);
    let cl_quant_recall = measure_recall(&cl_quant_idx, &cl_quant_ids, &clustered, &cl_queries);
    println!("recall@{K}:");
    println!("  full-precision index:  {cl_full_recall:.4}");
    println!("  quantized index:       {cl_quant_recall:.4}");
    println!(
        "  recall retained:       {:.1}%",
        cl_quant_recall / cl_full_recall * 100.0
    );
}