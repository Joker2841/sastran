//! Hierarchical Navigable Small World (HNSW) index for approximate
//! nearest-neighbor search.
//!
//! ## Algorithm summary
//!
//! HNSW builds a multi-layer graph. Each node lives in some prefix of
//! layers `[0, level]` where `level` is drawn from an exponential
//! distribution. Layer 0 contains every node; each higher layer holds
//! roughly `1/M` of the previous. Higher layers act as long-range
//! "highways" that get the search algorithm into the right region
//! quickly; lower layers refine to actual nearest neighbors.
//!
//! Search complexity is O(log N) expected. Recall is probabilistic but
//! typically 95–99% with default parameters (M=16, efSearch=50) on
//! standard ANN benchmarks.
//!
//! ## Status
//!
//! v0.2.0 / message 1 of 10: insert and search only. Real deletes,
//! configurable distance metrics, snapshots, quantization, and engine
//! integration come in subsequent messages. The current index is
//! standalone and uses cosine distance (with vectors normalized at
//! insert time).

pub mod index;
pub mod snapshot;

pub use index::{DistanceMetric, HnswIndex, HnswParams, NodeId};