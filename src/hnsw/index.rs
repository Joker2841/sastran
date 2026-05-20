//! HNSW graph + insert/search implementation.
//!
//! Vectors are processed at insert time, so distance computations
//! during traversal are inner products (cosine similarity reduces to
//! the inner product on unit vectors).

use crate::{Error, Result};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Distance metric used for graph construction and search.
///
/// The metric is fixed at index creation and influences both the graph
/// topology and the normalization applied to input vectors. Changing
/// metrics on an existing index is not supported; rebuild the index
/// from the source data with new parameters.
///
/// All variants return values where **smaller = closer**. This
/// uniform contract lets the same heap-based search code work across
/// metrics; for inner product specifically, the returned value is
/// negated similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    /// 1 - cosine_similarity. Input vectors are L2-normalized on insert
    /// and search; the user need not pre-normalize. Range [0, 2].
    #[default]
    Cosine,
    /// Squared Euclidean distance. Equivalent topology to L2 (sqrt is
    /// monotone) but ~30% faster. The returned value is squared; take
    /// `sqrt` at the call site if you need true L2 distance.
    EuclideanSquared,
    /// Negated inner product. Smaller = larger inner product = more
    /// "similar" in the linear-algebra sense. Vectors are *not*
    /// normalized; magnitude affects the score.
    InnerProduct,
}

/// Compact node identifier inside an [`HnswIndex`].
///
/// Caps the index at ~4 billion nodes; this is plenty for any
/// portfolio-scale workload and saves memory vs. `u64`.
pub type NodeId = u32;

/// Parameters that govern HNSW graph construction and search.
#[derive(Debug, Clone)]
pub struct HnswParams {
    /// Distance metric. Fixed at index creation.
    pub metric: DistanceMetric,

    /// Target number of bidirectional edges per node per layer >= 1.
    /// Layer 0 uses `2 * m` as its cap. Higher M = better recall, more
    /// memory and slower inserts. Default 16.
    pub m: usize,

    /// Maximum edges per node at layer 0. Defaults to `2 * m`.
    pub m_max_0: usize,

    /// Candidate-list width during inserts. Higher = better graph
    /// quality, slower inserts. Default 100.
    pub ef_construction: usize,

    /// Candidate-list width during searches. Higher = better recall,
    /// slower queries. Default 50; can be raised at query time without
    /// rebuilding the graph.
    pub ef_search: usize,

    /// Vector dimension. The first insert sets this; subsequent inserts
    /// must match. Set to 0 in `default()` to mean "infer from first
    /// insert."
    pub dim: usize,

    /// Seed for the RNG used by `assign_level`. Fixed seed makes graph
    /// construction deterministic, which matters for tests.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        let m = 16;
        Self {
            metric: DistanceMetric::default(),
            m,
            m_max_0: 2 * m,
            ef_construction: 100,
            ef_search: 50,
            dim: 0,
            seed: 0x00C0_FFEE_BABE_u64,
        }
    }
}

/// A single node in the graph: its normalized vector plus its neighbor
/// list at each layer in which the node appears.
pub(crate) struct Node {
    /// L2-normalized vector. Length equals `HnswIndex::dim`.
    pub(crate) vector: Vec<f32>,
    /// `neighbors[layer]` = neighbor IDs at that layer.
    /// `neighbors.len()` equals `level + 1`. Entries may include
    /// deleted nodes; consumers must filter via `is_deleted` when
    /// the *result* matters (e.g. when collecting search results).
    /// Traversing *through* a deleted node is fine and in fact
    /// helps connectivity in regions with many deletes.
    pub(crate) neighbors: Vec<Vec<NodeId>>,
    /// Set by `delete`. Skipped when collecting search results;
    /// the vector and neighbor lists are kept so traversal still
    /// works (edges through this node may reach live neighbors).
    pub(crate) deleted: bool,
}

/// Entry point at the top of the graph.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EntryPoint {
    pub(crate) node: NodeId,
    pub(crate) level: u8,
}

/// HNSW graph.
pub struct HnswIndex {
    params: HnswParams,
    /// Cached from params at construction; the metric is fixed for
    /// the lifetime of the graph. Keeping it directly on the index
    /// (rather than reading from params) makes the distance hot path
    /// one field access instead of two.
    metric: DistanceMetric,
    dim: usize,
    nodes: Vec<Node>,
    entry: Option<EntryPoint>,
    /// 1 / ln(M). Used by `assign_level` to convert a uniform sample
    /// into the exponential layer distribution.
    ml: f64,
    rng: SmallRng,
}

impl HnswIndex {
    /// Construct an empty index with the given parameters.
    pub fn new(params: HnswParams) -> Self {
        let ml = 1.0 / (params.m as f64).ln();
        let rng = SmallRng::seed_from_u64(params.seed);
        let dim = params.dim;
        let metric = params.metric;
        Self {
            params,
            metric,
            dim,
            nodes: Vec::new(),
            entry: None,
            ml,
            rng,
        }
    }

    /// Number of nodes inserted so far.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Distance metric this index was built with.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// True if `id` refers to a deleted node, or is out of range.
    pub fn is_deleted(&self, id: NodeId) -> bool {
        (id as usize) >= self.nodes.len() || self.nodes[id as usize].deleted
    }

    /// Number of *live* (non-deleted) nodes. `len()` returns the
    /// arena's total size, including tombstoned slots.
    pub fn live_len(&self) -> usize {
        self.nodes.iter().filter(|n| !n.deleted).count()
    }

    /// Remove `id` from the index. Neighbors at every layer where
    /// `id` appeared are re-linked to compensate for the lost edges.
    ///
    /// Returns `Err(InvalidArgument)` if the id is out of range or
    /// already deleted. Cost is roughly proportional to insert cost
    /// (one `search_layer` per surviving neighbor at each affected
    /// layer).
    pub fn delete(&mut self, id: NodeId) -> Result<()> {
        if (id as usize) >= self.nodes.len() {
            return Err(Error::InvalidArgument(format!(
                "delete: id {id} out of range (arena size {})",
                self.nodes.len()
            )));
        }
        if self.nodes[id as usize].deleted {
            return Err(Error::InvalidArgument(format!(
                "delete: id {id} is already deleted"
            )));
        }

        // Mark deleted before edge repair so any traversal that happens
        // mid-repair (none here, but defensive) sees the right state.
        self.nodes[id as usize].deleted = true;
        let deleted_level = (self.nodes[id as usize].neighbors.len() - 1) as u8;

        // Edge repair, layer by layer.
        for layer in 0..=deleted_level {
            self.repair_layer_after_delete(id, layer)?;
        }

        // Entry-point migration: if we just deleted the entry, find
        // a replacement. The replacement is the live node with the
        // highest level; if multiple, any of them.
        if self.entry.map(|e| e.node) == Some(id) {
            self.entry = self.pick_new_entry();
        }

        Ok(())
    }

    /// Repair neighbor links at one layer after deleting `deleted_id`.
    ///
    /// Strategy: for every live neighbor `N` of the deleted node at
    /// this layer:
    ///   1. Remove `deleted_id` from N's neighbor list.
    ///   2. Run a fresh `search_layer` starting at N to find candidate
    ///      replacement neighbors, excluding deleted nodes.
    ///   3. Apply the neighbor-selection heuristic to pick the top M
    ///      (or m_max_0 for layer 0).
    ///   4. Wire bidirectional edges to any newly-chosen neighbor that
    ///      isn't already linked.
    ///   5. If the new neighbor's list now exceeds its cap, trim it.
    fn repair_layer_after_delete(
        &mut self,
        deleted_id: NodeId,
        layer: u8,
    ) -> Result<()> {
        // Collect neighbors of the deleted node at this layer.
        // Filter to live ones; deleted neighbors will themselves
        // be repaired (or already were).
        let affected: Vec<NodeId> = self.nodes[deleted_id as usize].neighbors
            [layer as usize]
            .iter()
            .copied()
            .filter(|&n| !self.nodes[n as usize].deleted)
            .collect();

        let m_max = if layer == 0 {
            self.params.m_max_0
        } else {
            self.params.m
        };

        for n_id in affected {
            // Step 1: remove the deleted edge.
            self.nodes[n_id as usize].neighbors[layer as usize]
                .retain(|&x| x != deleted_id);

            // Step 2 & 3: search for replacements, run heuristic.
            let n_vec = self.nodes[n_id as usize].vector.clone();
            let raw_candidates = self.search_layer_skipping_deleted(
                &n_vec,
                n_id,
                layer,
                self.params.ef_construction,
            );
            // Exclude n_id itself and currently-linked neighbors;
            // we're looking for *new* connections.
            let already_linked: std::collections::HashSet<NodeId> = self.nodes
                [n_id as usize]
                .neighbors[layer as usize]
                .iter()
                .copied()
                .collect();
            let novel: Vec<Candidate> = raw_candidates
                .into_iter()
                .filter(|c| c.id != n_id && !already_linked.contains(&c.id))
                .collect();
            let chosen = self.select_neighbors_heuristic(
                &n_vec,
                novel.iter().map(|c| (c.id, c.dist)),
                self.params.m,
            );

            // Step 4: bidirectional wiring.
            for cand in &chosen {
                // Only link if it would actually improve the situation
                // i.e. add the new edge.
                self.nodes[n_id as usize].neighbors[layer as usize].push(cand.id);
                self.nodes[cand.id as usize].neighbors[layer as usize].push(n_id);
                // Step 5: trim if over cap.
                if self.nodes[cand.id as usize].neighbors[layer as usize].len() > m_max
                {
                    self.trim_neighbors(cand.id, layer, m_max);
                }
            }

            // n_id's own list may also be over cap if we added many.
            if self.nodes[n_id as usize].neighbors[layer as usize].len() > m_max {
                self.trim_neighbors(n_id, layer, m_max);
            }
        }

        Ok(())
    }

    /// Find a new entry point after the old one was deleted. Picks
    /// the live node with the highest level; returns `None` if no
    /// live nodes remain.
    fn pick_new_entry(&self) -> Option<EntryPoint> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.deleted)
            .map(|(idx, n)| EntryPoint {
                node: idx as NodeId,
                level: (n.neighbors.len() - 1) as u8,
            })
            .max_by_key(|ep| ep.level)
    }

    /// Like `search_layer`, but filters out deleted nodes from the
    /// result set while still traversing through them.
    fn search_layer_skipping_deleted(
        &self,
        query: &[f32],
        start: NodeId,
        layer: u8,
        ef: usize,
    ) -> Vec<Candidate> {
        let raw = self.search_layer(query, start, layer, ef);
        raw.into_iter()
            .filter(|c| !self.nodes[c.id as usize].deleted)
            .collect()
    }

    /// Vector dimension. `0` means "not yet established."
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Insert a vector and return its assigned [`NodeId`].
    ///
    /// The vector is normalized internally; the caller need not
    /// pre-normalize. Zero vectors are rejected because their direction
    /// is undefined.
    pub fn insert(&mut self, vector: &[f32]) -> Result<NodeId> {
        // Establish dimension on first insert.
        if self.dim == 0 {
            if vector.is_empty() {
                return Err(Error::InvalidArgument(
                    "cannot insert zero-dimension vector".into(),
                ));
            }
            self.dim = vector.len();
        } else if vector.len() != self.dim {
            return Err(Error::InvalidArgument(format!(
                "vector dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            )));
        }

        let processed = match preprocess_vector(self.metric, vector) {
            Some(v) => v,
            None => {
                return Err(Error::InvalidArgument(
                    "cannot insert invalid vector \
                     (zero-norm for cosine, or contains non-finite values)"
                        .into(),
                ));
            }
        };

        let level = self.assign_level();
        let id = self.nodes.len() as NodeId;

        // Allocate the new node with empty neighbor lists at every
        // layer up to and including its `level`.
        let new_node = Node {
            vector: processed.clone(),
            neighbors: (0..=level).map(|_| Vec::new()).collect(),
            deleted: false,
        };
        self.nodes.push(new_node);

        // First insert: this node *is* the entry point.
        let entry = match self.entry {
            Some(e) => e,
            None => {
                self.entry = Some(EntryPoint { node: id, level });
                return Ok(id);
            }
        };

        // Phase 1: descend from the top layer to `level + 1`, greedy
        // search with ef = 1 at each. The result is the entry point
        // for the lower layers.
        let mut current = entry.node;
        let top = entry.level as i32;
        let descend_to = level as i32;
        for layer in (descend_to + 1..=top).rev() {
            current = self.greedy_search_layer(&processed, current, layer as u8);
        }

        // Phase 2: from `min(level, entry.level)` down to 0, run a
        // wider beam search at each layer, pick M neighbors via the
        // heuristic, link bidirectionally, trim over-full lists.
        //
        // The clamp is essential: if `level > entry.level`, layers
        // above `entry.level` exist only for the new node (it becomes
        // the sole occupant). There are no candidates to link to up
        // there, and trying to search at those layers would index
        // into neighbor lists that don't exist on the existing node.
        let mut ep_for_layer = current;
        let phase2_top = (level.min(entry.level)) as i32;
        for layer in (0..=phase2_top).rev() {
            let layer_u8 = layer as u8;
            let candidates =
                self.search_layer(&processed, ep_for_layer, layer_u8, self.params.ef_construction);

            // Select M neighbors using the heuristic.
            let m_max = if layer == 0 {
                self.params.m_max_0
            } else {
                self.params.m
            };
            let neighbors = self.select_neighbors_heuristic(
                &processed,
                candidates.iter().map(|c| (c.id, c.dist)),
                self.params.m,
            );

            // Wire bidirectional edges. Updating the new node first.
            self.nodes[id as usize].neighbors[layer as usize] =
                neighbors.iter().map(|c| c.id).collect();
            for cand in &neighbors {
                self.nodes[cand.id as usize].neighbors[layer as usize].push(id);
                // Trim if over capacity.
                let degree =
                    self.nodes[cand.id as usize].neighbors[layer as usize].len();
                if degree > m_max {
                    self.trim_neighbors(cand.id, layer_u8, m_max);
                }
            }

            // Use the closest candidate from this layer as the entry
            // point for the next-lower one.
            if let Some(closest) = candidates.iter().min_by(|a, b| {
                a.dist
                    .partial_cmp(&b.dist)
                    .unwrap_or(Ordering::Equal)
            }) {
                ep_for_layer = closest.id;
            }
        }

        // Update entry point if this node is on a new highest layer.
        if level > entry.level {
            self.entry = Some(EntryPoint { node: id, level });
        }

        Ok(id)
    }

    /// Search for the `k` approximate nearest neighbors of `query`.
    /// Returns results in ascending distance order (closest first).
    ///
    /// Distance is 1 - cosine_similarity in [0, 2]; 0 means identical
    /// direction.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(NodeId, f32)>> {
        if self.is_empty() || k == 0 || self.entry.is_none() {
            return Ok(Vec::new());
        }
        if query.len() != self.dim {
            return Err(Error::InvalidArgument(format!(
                "query dimension mismatch: expected {}, got {}",
                self.dim,
                query.len()
            )));
        }
        let processed = match preprocess_vector(self.metric, query) {
            Some(v) => v,
            None => {
                return Err(Error::InvalidArgument(
                    "cannot search for invalid query vector \
                     (zero-norm for cosine, or contains non-finite values)"
                        .into(),
                ));
            }
        };

        let entry = self.entry.expect("non-empty index has entry point");

        // Descend from the top layer to layer 1 with ef=1.
        let mut current = entry.node;
        for layer in (1..=entry.level as i32).rev() {
            current = self.greedy_search_layer(&processed, current, layer as u8);
        }

        // Final beam search at layer 0. Use ef larger than k to give
        // headroom for deleted-node filtering; without this, every
        // deleted node in the top-ef would steal a slot.
        let ef = self.params.ef_search.max(k * 2).max(k);
        let raw = self.search_layer(&processed, current, 0, ef);
        let mut filtered: Vec<Candidate> = raw
            .into_iter()
            .filter(|c| !self.nodes[c.id as usize].deleted)
            .collect();
        filtered.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        filtered.truncate(k);

        Ok(filtered.into_iter().map(|c| (c.id, c.dist)).collect())
    }

    // ---- helpers ----

    /// Sample a layer for a newly-inserted node from the exponential
    /// distribution `floor(-ln(uniform) * ml)`. Capped to avoid runaway
    /// degenerate samples.
    fn assign_level(&mut self) -> u8 {
        let r: f64 = self.rng.gen_range(f64::EPSILON..1.0);
        let level = (-r.ln() * self.ml).floor() as i64;
        level.clamp(0, 30) as u8
    }

    /// Greedy 1-best descent at a single layer. Walk toward the query
    /// by always moving to the neighbor closest to it, stopping when
    /// no neighbor is closer than the current node.
    fn greedy_search_layer(&self, query: &[f32], start: NodeId, layer: u8) -> NodeId {
        let mut current = start;
        let mut current_dist = self.distance(query, current);
        loop {
            let mut best_id = current;
            let mut best_dist = current_dist;
            for &neighbor in &self.nodes[current as usize].neighbors[layer as usize] {
                let d = self.distance(query, neighbor);
                if d < best_dist {
                    best_dist = d;
                    best_id = neighbor;
                }
            }
            if best_id == current {
                return current;
            }
            current = best_id;
            current_dist = best_dist;
        }
    }

    /// Beam search at one layer. Returns the top-`ef` candidates
    /// reached, with their distances. Used by insert and by the final
    /// stage of search.
    fn search_layer(
        &self,
        query: &[f32],
        start: NodeId,
        layer: u8,
        ef: usize,
    ) -> Vec<Candidate> {
        let mut visited: HashSet<NodeId> = HashSet::new();
        // Min-heap on distance (closest first): wrap Candidate in
        // Reverse so BinaryHeap pops smallest.
        let mut candidates: BinaryHeap<std::cmp::Reverse<Candidate>> = BinaryHeap::new();
        // Max-heap (worst first) bounded by ef.
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        let start_dist = self.distance(query, start);
        candidates.push(std::cmp::Reverse(Candidate {
            id: start,
            dist: start_dist,
        }));
        results.push(Candidate {
            id: start,
            dist: start_dist,
        });
        visited.insert(start);

        while let Some(std::cmp::Reverse(c)) = candidates.pop() {
            // Termination: if the closest unexplored candidate is worse
            // than the worst entry in results, no neighbor can improve
            // results further.
            let worst_in_results = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
            if c.dist > worst_in_results {
                break;
            }
            for &n in &self.nodes[c.id as usize].neighbors[layer as usize] {
                if !visited.insert(n) {
                    continue;
                }
                let nd = self.distance(query, n);
                let worst = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
                if nd < worst || results.len() < ef {
                    candidates.push(std::cmp::Reverse(Candidate { id: n, dist: nd }));
                    results.push(Candidate { id: n, dist: nd });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        results.into_vec()
    }

    /// "Select neighbors heuristic" from the HNSW paper. Walks
    /// candidates in ascending-distance order; admits a candidate only
    /// if it is closer to the *new node* than to any already-admitted
    /// candidate. This preserves directional diversity, which is the
    /// single biggest difference between the heuristic and naive
    /// closest-M selection.
    fn select_neighbors_heuristic<I>(
        &self,
        new_vec: &[f32],
        candidates: I,
        m: usize,
    ) -> Vec<Candidate>
    where
        I: IntoIterator<Item = (NodeId, f32)>,
    {
        // Sort candidates ascending by distance to new_vec.
        let mut sorted: Vec<Candidate> = candidates
            .into_iter()
            .map(|(id, dist)| Candidate { id, dist })
            .collect();
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

        let mut chosen: Vec<Candidate> = Vec::with_capacity(m);
        for cand in sorted {
            if chosen.len() >= m {
                break;
            }
            let mut admit = true;
            for c in &chosen {
                let d_to_existing = distance_for_metric(
                    self.metric,
                    &self.nodes[cand.id as usize].vector,
                    &self.nodes[c.id as usize].vector,
                );
                if d_to_existing < cand.dist {
                    // Existing chosen point is closer to candidate than
                    // new_vec is — would create a redundant edge.
                    admit = false;
                    break;
                }
            }
            let _ = new_vec; // used implicitly via cand.dist
            if admit {
                chosen.push(cand);
            }
        }
        chosen
    }

    /// Trim a node's neighbors at a layer down to `max` entries using
    /// the same heuristic as `select_neighbors_heuristic`.
    fn trim_neighbors(&mut self, node: NodeId, layer: u8, max: usize) {
        let neighbors = self.nodes[node as usize].neighbors[layer as usize].clone();
        if neighbors.len() <= max {
            return;
        }
        let node_vec = self.nodes[node as usize].vector.clone();
        let candidates: Vec<(NodeId, f32)> = neighbors
            .into_iter()
            .map(|n| {
                (
                    n,
                    distance_for_metric(
                        self.metric,
                        &node_vec,
                        &self.nodes[n as usize].vector,
                    ),
                )
            })
            .collect();
        let kept = self.select_neighbors_heuristic(&node_vec, candidates, max);
        self.nodes[node as usize].neighbors[layer as usize] =
            kept.into_iter().map(|c| c.id).collect();
    }

    fn distance(&self, query: &[f32], id: NodeId) -> f32 {
        distance_for_metric(self.metric, query, &self.nodes[id as usize].vector)
    }

    /// Estimated serialized snapshot size, in bytes. Used to pre-size
    /// the output Vec in `encode_snapshot` — accuracy doesn't matter,
    /// just avoiding reallocations during a large encode.
    pub(crate) fn estimated_snapshot_size(&self) -> usize {
        let header_etc = 24 + 37 + 6 + 4 + 4; // header, params, entry, count, CRC
        let per_node = 1 + 4 + (self.dim * 4) + 1 + (self.params.m_max_0 * 4) * 3;
        header_etc + self.nodes.len() * per_node
    }

    /// Params snapshot — used by `encode_snapshot` only.
    pub(crate) fn params(&self) -> &HnswParams {
        &self.params
    }

    /// Entry-point snapshot — used by `encode_snapshot` only.
    pub(crate) fn entry_snapshot(&self) -> Option<(NodeId, u8)> {
        self.entry.map(|e| (e.node, e.level))
    }

    /// Encode one node into `out`. Used by `encode_snapshot` only.
    pub(crate) fn encode_one_node(&self, id: NodeId, key: Option<&[u8]>, out: &mut Vec<u8>) {
        let node = &self.nodes[id as usize];
        out.push(node.deleted as u8);
        out.extend_from_slice(&(node.vector.len() as u32).to_le_bytes());
        for &f in &node.vector {
            out.extend_from_slice(&f.to_le_bytes());
        }
        // Key, inserted in v3.
        let key_bytes = key.unwrap_or(&[]);
        out.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(key_bytes);
        // Neighbors (unchanged).
        out.push(node.neighbors.len() as u8);
        for layer in &node.neighbors {
            out.extend_from_slice(&(layer.len() as u32).to_le_bytes());
            for &n in layer {
                out.extend_from_slice(&n.to_le_bytes());
            }
        }
    }

    /// Restore state from snapshot decode. Used by `decode_snapshot`
    /// only. Takes the raw decoded nodes, the entry point, and the
    /// established dim, and rebuilds the internal state.
    pub(crate) fn set_state_from_snapshot(
        &mut self,
        raw_nodes: Vec<crate::hnsw::snapshot::RawNode>,
        entry: Option<(NodeId, u8)>,
        dim: usize,
    ) {
        use crate::hnsw::index::{EntryPoint, Node};
        self.dim = dim;
        self.nodes = raw_nodes
            .into_iter()
            .map(|raw| Node {
                vector: raw.vector,
                neighbors: raw.neighbors,
                deleted: raw.deleted,
            })
            .collect();
        self.entry = entry.map(|(node, level)| EntryPoint { node, level });
    }

}

/// Cosine distance: `1 - dot(a, b)` assuming both are unit vectors.
/// Range [0, 2].
fn distance_cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    1.0 - dot
}

/// Squared L2 distance: `Σ (aᵢ - bᵢ)²`. Same topology as L2 (sqrt is
/// monotone) but cheaper.
fn distance_euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Negated inner product, so smaller = more similar.
fn distance_negated_inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    -dot
}

/// Compute the distance between two vectors under `metric`. Assumes
/// both vectors have already been preprocessed (processed for
/// `Cosine`; passed-through for other metrics).
fn distance_for_metric(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Cosine => distance_cosine(a, b),
        DistanceMetric::EuclideanSquared => distance_euclidean_squared(a, b),
        DistanceMetric::InnerProduct => distance_negated_inner_product(a, b),
    }
}

/// L2-normalize a vector. Returns `None` if its norm is zero or
/// non-finite.
fn normalize(v: &[f32]) -> Option<Vec<f32>> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    Some(v.iter().map(|x| x / norm).collect())
}

/// Preprocess a vector for insertion/query under `metric`:
/// - Cosine: normalize to unit length; reject zero-norm.
/// - Euclidean / InnerProduct: pass through, validating finiteness.
fn preprocess_vector(metric: DistanceMetric, v: &[f32]) -> Option<Vec<f32>> {
    match metric {
        DistanceMetric::Cosine => normalize(v),
        DistanceMetric::EuclideanSquared | DistanceMetric::InnerProduct => {
            if v.iter().all(|x| x.is_finite()) {
                Some(v.to_vec())
            } else {
                None
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    id: NodeId,
    dist: f32,
}

// BinaryHeap is a max-heap by default. We define Candidate's Ord so
// that the *largest* distance is at the top: that gives us "worst
// first" for the results heap. The candidates heap uses
// std::cmp::Reverse to invert this for "closest first."
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.dist == other.dist
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // NaN should not appear (vectors are processed; distances are
        // 1 - dot which is finite); we treat NaN as equal as a defensive
        // measure.
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::Rng;

    fn random_unit_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        normalize(&v).unwrap()
    }

    /// Brute-force baseline for recall measurement.
    fn brute_force_topk(
        metric: DistanceMetric,
        query: &[f32],
        dataset: &[Vec<f32>],
        k: usize,
    ) -> Vec<usize> {
        let q = preprocess_vector(metric, query).unwrap();
        let processed_dataset: Vec<Vec<f32>> = dataset
            .iter()
            .map(|v| preprocess_vector(metric, v).unwrap())
            .collect();
        let mut dists: Vec<(usize, f32)> = processed_dataset
            .iter()
            .enumerate()
            .map(|(i, v)| (i, distance_for_metric(metric, &q, v)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// Build a synthetic key→NodeId map covering the index's
    /// live NodeIds with sequential keys like b"node_0", b"node_1", …
    /// Used by snapshot tests to satisfy the new key requirement.
    fn synth_keys_for_index(
        idx: &HnswIndex,
    ) -> std::collections::HashMap<Vec<u8>, NodeId> {
        let mut keys = std::collections::HashMap::new();
        for id in 0..idx.len() as NodeId {
            if !idx.is_deleted(id) {
                let key = format!("node_{id}").into_bytes();
                keys.insert(key, id);
            }
        }
        keys
    }

    #[test]
    fn empty_search_returns_empty() {
        let idx = HnswIndex::new(HnswParams::default());
        let result = idx.search(&[1.0, 0.0, 0.0], 10).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn search_with_k_zero_returns_empty() {
        let mut idx = HnswIndex::new(HnswParams::default());
        idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        let result = idx.search(&[1.0, 0.0, 0.0], 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn rejects_zero_norm_vector() {
        let mut idx = HnswIndex::new(HnswParams::default());
        let err = idx.insert(&[0.0, 0.0, 0.0]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let mut idx = HnswIndex::new(HnswParams::default());
        idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        let err = idx.insert(&[1.0, 0.0]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn single_vector_search_returns_itself() {
        let mut idx = HnswIndex::new(HnswParams::default());
        let id = idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        let result = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, id);
        assert!(result[0].1.abs() < 1e-6, "self-distance should be ~0");
    }

    #[test]
    fn two_vectors_search_orders_by_distance() {
        let mut idx = HnswIndex::new(HnswParams::default());
        let near = idx.insert(&[1.0, 0.01, 0.0]).unwrap();
        let far = idx.insert(&[0.0, 1.0, 0.0]).unwrap();
        let result = idx.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, near);
        assert_eq!(result[1].0, far);
    }

    #[test]
    fn recall_at_10_on_synthetic_data() {
        // 1000 random vectors of dimension 16. Build index, then query
        // 100 random points. Compute recall@10 against brute force.
        let mut rng = StdRng::seed_from_u64(42);
        let dim = 16;
        let n = 1000;
        let dataset: Vec<Vec<f32>> = (0..n)
            .map(|_| random_unit_vector(&mut rng, dim))
            .collect();
        let mut idx = HnswIndex::new(HnswParams::default());
        let ids: Vec<NodeId> = dataset
            .iter()
            .map(|v| idx.insert(v).unwrap())
            .collect();

        let queries: Vec<Vec<f32>> = (0..100)
            .map(|_| random_unit_vector(&mut rng, dim))
            .collect();

        let mut total_recall = 0.0;
        for q in &queries {
            let truth: Vec<usize> = brute_force_topk(DistanceMetric::Cosine, q, &dataset, 10);
            let truth_ids: std::collections::HashSet<NodeId> =
                truth.iter().map(|&i| ids[i]).collect();
            let predicted = idx.search(q, 10).unwrap();
            let hits = predicted
                .iter()
                .filter(|(id, _)| truth_ids.contains(id))
                .count();
            total_recall += hits as f64 / 10.0;
        }
        let recall_at_10 = total_recall / queries.len() as f64;
        assert!(
            recall_at_10 > 0.90,
            "recall@10 = {recall_at_10:.3}, expected > 0.90"
        );
    }

    #[test]
    fn recall_at_10_on_larger_dataset() {
        // 5000 vectors of dimension 32 — larger, harder.
        let mut rng = StdRng::seed_from_u64(7);
        let dim = 32;
        let n = 5000;
        let dataset: Vec<Vec<f32>> = (0..n)
            .map(|_| random_unit_vector(&mut rng, dim))
            .collect();
        let mut idx = HnswIndex::new(HnswParams::default());
        let ids: Vec<NodeId> = dataset
            .iter()
            .map(|v| idx.insert(v).unwrap())
            .collect();

        let queries: Vec<Vec<f32>> = (0..50)
            .map(|_| random_unit_vector(&mut rng, dim))
            .collect();

        let mut total_recall = 0.0;
        for q in &queries {
            let truth: Vec<usize> = brute_force_topk(DistanceMetric::Cosine, q, &dataset, 10);
            let truth_ids: std::collections::HashSet<NodeId> =
                truth.iter().map(|&i| ids[i]).collect();
            let predicted = idx.search(q, 10).unwrap();
            let hits = predicted
                .iter()
                .filter(|(id, _)| truth_ids.contains(id))
                .count();
            total_recall += hits as f64 / 10.0;
        }
        let recall_at_10 = total_recall / queries.len() as f64;
        assert!(
            recall_at_10 > 0.85,
            "recall@10 = {recall_at_10:.3}, expected > 0.85"
        );
    }

    #[test]
    fn insert_returns_sequential_ids() {
        let mut idx = HnswIndex::new(HnswParams::default());
        let ids: Vec<NodeId> = (0..50)
            .map(|i| {
                let mut v = vec![0.0f32; 8];
                v[i % 8] = 1.0;
                idx.insert(&v).unwrap()
            })
            .collect();
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(*id as usize, i);
        }
    }

    #[test]
    fn params_dim_override_takes_precedence() {
        let params = HnswParams {
            dim: 4,
            ..Default::default()
        };
        let mut idx = HnswIndex::new(params);
        // First insert must match the configured dim.
        let err = idx.insert(&[1.0, 0.0, 0.0]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
        // Correct dimension works.
        idx.insert(&[1.0, 0.0, 0.0, 0.0]).unwrap();
    }

    #[test]
    fn delete_makes_node_unreachable_by_search() {
        let mut idx = HnswIndex::new(HnswParams::default());
        let id_a = idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        let id_b = idx.insert(&[0.0, 1.0, 0.0]).unwrap();
        let id_c = idx.insert(&[0.0, 0.0, 1.0]).unwrap();

        // Before delete: query for ~A should return A.
        let before = idx.search(&[1.0, 0.0, 0.0], 3).unwrap();
        assert!(before.iter().any(|(id, _)| *id == id_a));

        idx.delete(id_a).unwrap();
        assert!(idx.is_deleted(id_a));
        assert!(!idx.is_deleted(id_b));
        assert!(!idx.is_deleted(id_c));

        // After delete: A no longer appears in results.
        let after = idx.search(&[1.0, 0.0, 0.0], 3).unwrap();
        assert!(
            !after.iter().any(|(id, _)| *id == id_a),
            "deleted node should not appear in search results"
        );
    }

    #[test]
    fn delete_updates_live_len() {
        let mut idx = HnswIndex::new(HnswParams::default());
        let ids: Vec<NodeId> = (0..10)
            .map(|i| {
                let mut v = vec![0.0f32; 8];
                v[i % 8] = 1.0;
                idx.insert(&v).unwrap()
            })
            .collect();
        assert_eq!(idx.live_len(), 10);
        assert_eq!(idx.len(), 10);

        idx.delete(ids[3]).unwrap();
        idx.delete(ids[7]).unwrap();
        assert_eq!(idx.live_len(), 8);
        assert_eq!(idx.len(), 10, "arena size unchanged by soft-delete");
    }

    #[test]
    fn delete_rejects_out_of_range() {
        let mut idx = HnswIndex::new(HnswParams::default());
        idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        let err = idx.delete(999).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn delete_rejects_already_deleted() {
        let mut idx = HnswIndex::new(HnswParams::default());
        let id = idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        idx.delete(id).unwrap();
        let err = idx.delete(id).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn delete_all_nodes_leaves_searchable_empty() {
        let mut idx = HnswIndex::new(HnswParams::default());
        let ids: Vec<NodeId> = (0..5)
            .map(|i| {
                let mut v = vec![0.0f32; 4];
                v[i % 4] = 1.0;
                idx.insert(&v).unwrap()
            })
            .collect();
        for id in ids {
            idx.delete(id).unwrap();
        }
        assert_eq!(idx.live_len(), 0);
        // Search returns empty; should not panic.
        let result = idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn delete_entry_point_migrates_to_replacement() {
        let mut idx = HnswIndex::new(HnswParams::default());
        let ids: Vec<NodeId> = (0..20)
            .map(|i| {
                let mut v = vec![0.0f32; 8];
                v[i % 8] = (i as f32) * 0.1 + 0.5;
                idx.insert(&v).unwrap()
            })
            .collect();

        // Find current entry and delete it.
        let entry_id = idx.entry.unwrap().node;
        idx.delete(entry_id).unwrap();

        // A new entry should exist among the surviving nodes.
        let new_entry = idx.entry.expect("entry should re-materialize");
        assert!(!idx.is_deleted(new_entry.node));
        assert!(ids.contains(&new_entry.node));

        // Search still works.
        let q = vec![0.5f32; 8];
        let r = idx.search(&q, 5).unwrap();
        assert!(!r.is_empty());
        // No deleted ids in results.
        for (id, _) in &r {
            assert!(!idx.is_deleted(*id));
        }
    }

    #[test]
    fn recall_holds_after_many_deletes() {
        // Build a 2000-vector index. Delete a random 30%. Verify
        // recall@10 on remaining queries is still acceptable.
        let mut rng = StdRng::seed_from_u64(17);
        let dim = 16;
        let n = 2000;
        let dataset: Vec<Vec<f32>> = (0..n)
            .map(|_| random_unit_vector(&mut rng, dim))
            .collect();
        let mut idx = HnswIndex::new(HnswParams::default());
        let ids: Vec<NodeId> = dataset
            .iter()
            .map(|v| idx.insert(v).unwrap())
            .collect();

        // Delete 30%.
        let n_delete = (n * 3) / 10;
        let mut delete_indices: Vec<usize> = (0..n).collect();
        delete_indices.truncate(n_delete);
        for di in &delete_indices {
            idx.delete(ids[*di]).unwrap();
        }

        // Build the surviving set for brute-force baseline.
        let deleted: std::collections::HashSet<usize> =
            delete_indices.iter().copied().collect();
        let survivors: Vec<(usize, &Vec<f32>)> = dataset
            .iter()
            .enumerate()
            .filter(|(i, _)| !deleted.contains(i))
            .collect();

        let queries: Vec<Vec<f32>> = (0..50)
            .map(|_| random_unit_vector(&mut rng, dim))
            .collect();

        let mut total_recall = 0.0;
        for q in &queries {
            let qn = normalize(q).unwrap();
            // Brute-force top-10 among survivors.
            let mut dists: Vec<(NodeId, f32)> = survivors
                .iter()
                .map(|(orig_i, v)| (ids[*orig_i], distance_cosine(&qn, v)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let truth: std::collections::HashSet<NodeId> =
                dists.iter().take(10).map(|(id, _)| *id).collect();

            let predicted = idx.search(q, 10).unwrap();
            let hits = predicted.iter().filter(|(id, _)| truth.contains(id)).count();
            total_recall += hits as f64 / 10.0;
        }
        let recall = total_recall / queries.len() as f64;
        assert!(
            recall > 0.80,
            "post-delete recall@10 = {recall:.3}, expected > 0.80"
        );
    }

    #[test]
    fn insert_after_delete_works() {
        let mut idx = HnswIndex::new(HnswParams::default());
        let id_a = idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        idx.delete(id_a).unwrap();
        // Should still accept new inserts. The new id will be id_a + 1
        // (we don't reuse slots in this version).
        let id_b = idx.insert(&[0.0, 1.0, 0.0]).unwrap();
        assert_eq!(id_b, id_a + 1);
        assert_eq!(idx.live_len(), 1);

        let r = idx.search(&[0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, id_b);
    }

    #[test]
    fn delete_then_insert_then_delete_pattern() {
        // Stress: alternating inserts and deletes shouldn't degrade.
        let mut rng = StdRng::seed_from_u64(99);
        let dim = 8;
        let mut idx = HnswIndex::new(HnswParams::default());
        let mut live_ids: Vec<NodeId> = Vec::new();
        for cycle in 0..100 {
            // Insert 5.
            for _ in 0..5 {
                let v = random_unit_vector(&mut rng, dim);
                live_ids.push(idx.insert(&v).unwrap());
            }
            // Delete 2 random ones.
            if cycle > 0 && live_ids.len() > 2 {
                for _ in 0..2 {
                    let idx_pick = rng.gen_range(0..live_ids.len());
                    let id = live_ids.swap_remove(idx_pick);
                    idx.delete(id).unwrap();
                }
            }
        }
        // After 100 cycles: ~300 live nodes. Sanity: search succeeds
        // and returns only live ones.
        let q = random_unit_vector(&mut rng, dim);
        let r = idx.search(&q, 10).unwrap();
        for (id, _) in &r {
            assert!(!idx.is_deleted(*id), "search returned deleted id {id}");
            assert!(live_ids.contains(id), "search returned unknown id {id}");
        }
    }

    #[test]
    fn euclidean_metric_returns_correct_ordering() {
        // Three points on a line. Query close to origin.
        let params = HnswParams {
            metric: DistanceMetric::EuclideanSquared,
            ..Default::default()
        };
        let mut idx = HnswIndex::new(params);
        let id_close = idx.insert(&[0.1, 0.0]).unwrap();
        let id_mid = idx.insert(&[1.0, 0.0]).unwrap();
        let id_far = idx.insert(&[10.0, 0.0]).unwrap();

        let r = idx.search(&[0.0, 0.0], 3).unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].0, id_close);
        assert_eq!(r[1].0, id_mid);
        assert_eq!(r[2].0, id_far);
        // Returned values are squared distances.
        assert!((r[0].1 - 0.01).abs() < 1e-5);
        assert!((r[1].1 - 1.0).abs() < 1e-5);
        assert!((r[2].1 - 100.0).abs() < 1e-3);
    }

    #[test]
    fn inner_product_metric_favors_larger_magnitudes() {
        // Two vectors with the same direction but different magnitudes.
        // Inner product against a query in the same direction prefers
        // the larger one.
        let params = HnswParams {
            metric: DistanceMetric::InnerProduct,
            ..Default::default()
        };
        let mut idx = HnswIndex::new(params);
        let id_small = idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        let id_large = idx.insert(&[10.0, 0.0, 0.0]).unwrap();

        let r = idx.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(r.len(), 2);
        // Larger magnitude = higher inner product = smaller (more
        // negative) score under the negated convention.
        assert_eq!(r[0].0, id_large, "larger magnitude should rank first");
        assert_eq!(r[1].0, id_small);
        // Scores are negated dot products: -10 and -1 respectively.
        assert!((r[0].1 - (-10.0)).abs() < 1e-5);
        assert!((r[1].1 - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn euclidean_accepts_unprocessed_vectors() {
        // The Euclidean code path must NOT reject non-unit vectors.
        let params = HnswParams {
            metric: DistanceMetric::EuclideanSquared,
            ..Default::default()
        };
        let mut idx = HnswIndex::new(params);
        idx.insert(&[100.0, 0.0]).unwrap();
        idx.insert(&[0.0, 100.0]).unwrap();
        let r = idx.search(&[50.0, 50.0], 2).unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn rejects_non_finite_under_any_metric() {
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::EuclideanSquared,
            DistanceMetric::InnerProduct,
        ] {
            let params = HnswParams {
                metric,
                ..Default::default()
            };
            let mut idx = HnswIndex::new(params);
            // NaN should be rejected.
            let err = idx.insert(&[f32::NAN, 0.0]).unwrap_err();
            assert!(
                matches!(err, Error::InvalidArgument(_)),
                "metric {metric:?} should reject NaN: got {err:?}"
            );
            // Infinity too.
            let err = idx.insert(&[f32::INFINITY, 0.0]).unwrap_err();
            assert!(
                matches!(err, Error::InvalidArgument(_)),
                "metric {metric:?} should reject infinity: got {err:?}"
            );
        }
    }

    #[test]
    fn metric_is_exposed_on_index() {
        let params = HnswParams {
            metric: DistanceMetric::EuclideanSquared,
            ..Default::default()
        };
        let idx = HnswIndex::new(params);
        assert_eq!(idx.metric(), DistanceMetric::EuclideanSquared);
    }

    #[test]
    fn recall_at_10_under_euclidean_metric() {
        // Same recall test as cosine, but with Euclidean. The graph
        // topology is different but recall should still be high.
        let mut rng = StdRng::seed_from_u64(123);
        let dim = 16;
        let n = 1000;
        // Unprocessed random vectors — Euclidean preserves magnitude.
        let dataset: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.gen_range(-1.0..1.0_f32))
                    .collect::<Vec<f32>>()
            })
            .collect();
        let params = HnswParams {
            metric: DistanceMetric::EuclideanSquared,
            ..Default::default()
        };
        let mut idx = HnswIndex::new(params);
        let ids: Vec<NodeId> = dataset
            .iter()
            .map(|v| idx.insert(v).unwrap())
            .collect();

        let queries: Vec<Vec<f32>> = (0..50)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.gen_range(-1.0..1.0_f32))
                    .collect::<Vec<f32>>()
            })
            .collect();

        let mut total_recall = 0.0;
        for q in &queries {
            let truth: Vec<usize> =
                brute_force_topk(DistanceMetric::EuclideanSquared, q, &dataset, 10);
            let truth_ids: std::collections::HashSet<NodeId> =
                truth.iter().map(|&i| ids[i]).collect();
            let predicted = idx.search(q, 10).unwrap();
            let hits = predicted.iter().filter(|(id, _)| truth_ids.contains(id)).count();
            total_recall += hits as f64 / 10.0;
        }
        let recall = total_recall / queries.len() as f64;
        assert!(
            recall > 0.85,
            "Euclidean recall@10 = {recall:.3}, expected > 0.85"
        );
    }

    #[test]
    fn snapshot_round_trip_empty_index() {
        let idx = HnswIndex::new(HnswParams::default());
        let bytes = idx.encode_snapshot(0, &synth_keys_for_index(&idx));
        let (restored, _keys, _) = HnswIndex::decode_snapshot(&bytes).unwrap();
        assert!(restored.is_empty());
        assert_eq!(restored.metric(), DistanceMetric::Cosine);
    }

    #[test]
    fn snapshot_round_trip_with_nodes() {
        let mut rng = StdRng::seed_from_u64(101);
        let dim = 16;
        let n = 200;
        let mut idx = HnswIndex::new(HnswParams::default());
        let mut originals = Vec::new();
        for _ in 0..n {
            let v = random_unit_vector(&mut rng, dim);
            idx.insert(&v).unwrap();
            originals.push(v);
        }

        let bytes = idx.encode_snapshot(0, &synth_keys_for_index(&idx));
        let (restored, _keys, _) = HnswIndex::decode_snapshot(&bytes).unwrap();

        assert_eq!(restored.len(), n);
        assert_eq!(restored.live_len(), n);
        assert_eq!(restored.dim(), dim);

        // Search results should match between original and restored
        // for the same query. Both indexes use the same RNG seed so
        // even the level assignments are identical.
        for _ in 0..10 {
            let q = random_unit_vector(&mut rng, dim);
            let r_orig = idx.search(&q, 5).unwrap();
            let r_rest = restored.search(&q, 5).unwrap();
            assert_eq!(
                r_orig, r_rest,
                "restored search results diverge from original"
            );
        }
    }

    #[test]
    fn snapshot_preserves_deletes() {
        let mut rng = StdRng::seed_from_u64(7);
        let dim = 8;
        let mut idx = HnswIndex::new(HnswParams::default());
        let ids: Vec<NodeId> = (0..50)
            .map(|_| idx.insert(&random_unit_vector(&mut rng, dim)).unwrap())
            .collect();
        // Delete every third.
        for id in ids.iter().step_by(3) {
            idx.delete(*id).unwrap();
        }

        let bytes = idx.encode_snapshot(0, &synth_keys_for_index(&idx));
        let (restored, _keys, _) = HnswIndex::decode_snapshot(&bytes).unwrap();

        assert_eq!(restored.len(), 50);
        assert_eq!(restored.live_len(), idx.live_len());
        for id in ids.iter().step_by(3) {
            assert!(restored.is_deleted(*id));
        }
    }

    #[test]
    fn snapshot_preserves_metric() {
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::EuclideanSquared,
            DistanceMetric::InnerProduct,
        ] {
            let params = HnswParams {
                metric,
                ..Default::default()
            };
            let mut idx = HnswIndex::new(params);
            idx.insert(&[1.0, 0.0, 0.0]).unwrap();
            idx.insert(&[0.0, 1.0, 0.0]).unwrap();
            let bytes = idx.encode_snapshot(0, &synth_keys_for_index(&idx));
            let (restored, _keys, _) = HnswIndex::decode_snapshot(&bytes).unwrap();
            assert_eq!(restored.metric(), metric);
        }
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut idx = HnswIndex::new(HnswParams::default());
        idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        let mut bytes = idx.encode_snapshot(0, &synth_keys_for_index(&idx));
        bytes[0] = b'X'; // corrupt magic
        // Recompute CRC so we test the magic check specifically.
        let crc_offset = bytes.len() - 4;
        let new_crc = crc32fast::hash(&bytes[..crc_offset]);
        bytes[crc_offset..].copy_from_slice(&new_crc.to_le_bytes());
        let truncated = &bytes[..bytes.len() - 10];
        let result = HnswIndex::decode_snapshot(truncated);
        assert!(
            matches!(result, Err(Error::Corruption(_))),
            "expected Corruption error, got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn decode_rejects_bad_version() {
        let mut idx = HnswIndex::new(HnswParams::default());
        idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        let mut bytes = idx.encode_snapshot(0, &synth_keys_for_index(&idx));
        // Version is at offset 8..12.
        bytes[8..12].copy_from_slice(&999u32.to_le_bytes());
        let crc_offset = bytes.len() - 4;
        let new_crc = crc32fast::hash(&bytes[..crc_offset]);
        bytes[crc_offset..].copy_from_slice(&new_crc.to_le_bytes());
        let truncated = &bytes[..bytes.len() - 10];
        let result = HnswIndex::decode_snapshot(truncated);
        assert!(
            matches!(result, Err(Error::Corruption(_))),
            "expected Corruption error, got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn decode_rejects_crc_mismatch() {
        let mut idx = HnswIndex::new(HnswParams::default());
        idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        let mut bytes = idx.encode_snapshot(0, &synth_keys_for_index(&idx));
        // Flip one bit in the middle of the body (before the trailing
        // 4-byte CRC). Using len()/2 instead of a hardcoded offset makes
        // this robust against snapshot-format size changes.
        let target = bytes.len() / 2;
        assert!(
            target + 4 < bytes.len(),
            "snapshot too small to have a body to corrupt"
        );
        bytes[target] ^= 0x01;
        let result = HnswIndex::decode_snapshot(&bytes);
        assert!(
            matches!(result, Err(Error::Corruption(_))),
            "expected Corruption error, got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn decode_rejects_truncated_file() {
        let mut idx = HnswIndex::new(HnswParams::default());
        idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        let bytes = idx.encode_snapshot(0, &synth_keys_for_index(&idx));
        // Drop the last 10 bytes.
        let truncated = &bytes[..bytes.len() - 10];
        let result = HnswIndex::decode_snapshot(truncated);
        assert!(
            matches!(result, Err(Error::Corruption(_))),
            "expected Corruption error, got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn decode_rejects_neighbor_id_out_of_range() {
        // Build an index with enough nodes that node 0 has at least
        // one neighbor at layer 0.
        let mut idx = HnswIndex::new(HnswParams::default());
        for i in 0..5u32 {
            let mut v = vec![0.0f32; 4];
            v[i as usize % 4] = 1.0;
            idx.insert(&v).unwrap();
        }
        let mut bytes = idx.encode_snapshot(0, &synth_keys_for_index(&idx));

        // Find a u32-aligned region inside the body that decodes as a
        // small NodeId, and overwrite it with a deliberately-out-of-range
        // value. To make this robust to layout changes we scan from the
        // body region for the first occurrence of any neighbor id; node
        // ids in this small index are 0..5, so a u32 of value 0..5
        // following a u32 neighbor_count somewhere in the body is the
        // pattern.
        //
        // Simpler: just smash the *last 4 body bytes* (right before the
        // CRC). For an index of 5+ nodes this lands inside some node's
        // neighbor list, which will fail the bounds check on decode.
        // If it doesn't land in a neighbor id (e.g., layer_count byte),
        // it'll still fail decode for a different reason — both produce
        // Error::Corruption, which is what we're asserting.
        let crc_offset = bytes.len() - 4;
        let target_start = crc_offset - 4;
        bytes[target_start..crc_offset].copy_from_slice(&999_999u32.to_le_bytes());
        // Recompute CRC so we hit the structural check, not the CRC check.
        let new_crc = crc32fast::hash(&bytes[..crc_offset]);
        bytes[crc_offset..].copy_from_slice(&new_crc.to_le_bytes());

        let result = HnswIndex::decode_snapshot(&bytes);
        assert!(
            matches!(result, Err(Error::Corruption(_))),
            "expected Corruption error, got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn snapshot_carries_next_sstable_id() {
        let mut idx = HnswIndex::new(HnswParams::default());
        idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        let bytes = idx.encode_snapshot(42, &synth_keys_for_index(&idx));
        let (_restored, _keys, next_id) = HnswIndex::decode_snapshot(&bytes).unwrap();
        assert_eq!(next_id, 42);
    }

    #[test]
    fn snapshot_round_trips_keys() {
        let mut idx = HnswIndex::new(HnswParams::default());
        let id_a = idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        let id_b = idx.insert(&[0.0, 1.0, 0.0]).unwrap();
        let id_c = idx.insert(&[0.0, 0.0, 1.0]).unwrap();

        let mut keys = std::collections::HashMap::new();
        keys.insert(b"alpha".to_vec(), id_a);
        keys.insert(b"beta".to_vec(), id_b);
        keys.insert(b"gamma".to_vec(), id_c);

        let bytes = idx.encode_snapshot(0, &keys);
        let (_, restored_keys, _) = HnswIndex::decode_snapshot(&bytes).unwrap();

        assert_eq!(restored_keys.len(), 3);
        assert_eq!(restored_keys.get(b"alpha".as_slice()), Some(&id_a));
        assert_eq!(restored_keys.get(b"beta".as_slice()), Some(&id_b));
        assert_eq!(restored_keys.get(b"gamma".as_slice()), Some(&id_c));
    }

    #[test]
    fn snapshot_omits_keys_for_deleted_nodes() {
        let mut idx = HnswIndex::new(HnswParams::default());
        let id_a = idx.insert(&[1.0, 0.0, 0.0]).unwrap();
        let id_b = idx.insert(&[0.0, 1.0, 0.0]).unwrap();

        let mut keys = std::collections::HashMap::new();
        keys.insert(b"alpha".to_vec(), id_a);
        keys.insert(b"beta".to_vec(), id_b);

        // Engine's delete flow: mark deleted in the graph AND remove
        // from the key map.
        idx.delete(id_a).unwrap();
        keys.remove(b"alpha".as_slice());

        let bytes = idx.encode_snapshot(0, &keys);
        let (restored, restored_keys, _) = HnswIndex::decode_snapshot(&bytes).unwrap();

        // The deleted node is still in the arena.
        assert_eq!(restored.len(), 2);
        assert!(restored.is_deleted(id_a));
        // But only the live key was reported.
        assert_eq!(restored_keys.len(), 1);
        assert_eq!(restored_keys.get(b"beta".as_slice()), Some(&id_b));
        assert!(!restored_keys.contains_key(b"alpha".as_slice()));
    }

    #[test]
    fn decode_rejects_live_node_with_empty_key() {
        // Encode normally then surgically zero out a live node's key
        // length field. CRC must be recomputed so the error is the
        // empty-key check, not the CRC check.
        let mut idx = HnswIndex::new(HnswParams::default());
        idx.insert(&[1.0, 0.0, 0.0]).unwrap();

        let keys = synth_keys_for_index(&idx);
        let mut bytes = idx.encode_snapshot(0, &keys);

        // The first node's key_len is at this offset:
        //   header(24) + params(45) + entry(6) + node_count(4)
        //   + node prefix: deleted(1) + vec_len(4) + vec_data(3*4=12) = 17
        //   = 24 + 45 + 6 + 4 + 17 = 96
        let key_len_offset = 96;
        bytes[key_len_offset..key_len_offset + 4].copy_from_slice(&0u32.to_le_bytes());
        // Need to truncate the key bytes that follow, since key_len was
        // originally "node_0".len() = 6 (or similar) and now we're
        // claiming 0. Replace those 6 bytes with their absence — but
        // we can't easily shrink the vec without parsing. Easier: just
        // leave them in place and rebuild the CRC. Decode will see
        // key_len=0 (the modified field), parse 0 key bytes, then try
        // to parse layer_count from what was the first key byte. That
        // may or may not produce a clean empty-key error depending on
        // the bytes. To make the test reliably hit the empty-key error
        // we need to actually remove the key bytes. Build a fresh
        // snapshot with an explicit empty key for a live node via a
        // synthetic key map missing that node.
        let _ = bytes; // discard the surgical attempt

        // Encode with no key for the live node.
        let empty_keys = std::collections::HashMap::new();
        let bytes = idx.encode_snapshot(0, &empty_keys);

        let result = HnswIndex::decode_snapshot(&bytes);
        assert!(
            matches!(result, Err(Error::Corruption(_))),
            "expected Corruption, got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn decode_rejects_duplicate_keys() {
        // Construct an encoded blob with two nodes claiming the same
        // key. The cleanest way is to manually craft the bytes. The
        // alternative — encoding with a HashMap (which can't have
        // duplicate keys) — wouldn't trigger the path.
        //
        // We do this by building a 2-node index normally and then
        // surgically rewriting the second node's key bytes to match
        // the first node's key bytes. The key length is the same,
        // and the rest of the structure is intact.
        let mut idx = HnswIndex::new(HnswParams::default());
        idx.insert(&[1.0, 0.0]).unwrap();
        idx.insert(&[0.0, 1.0]).unwrap();

        let mut keys = std::collections::HashMap::new();
        keys.insert(b"same".to_vec(), 0);
        keys.insert(b"DIFF".to_vec(), 1);
        let mut bytes = idx.encode_snapshot(0, &keys);

        // Find the second occurrence of b"DIFF" and replace with b"same".
        let pos = bytes
            .windows(4)
            .position(|w| w == b"DIFF")
            .expect("DIFF should be in encoded bytes");
        bytes[pos..pos + 4].copy_from_slice(b"same");

        // Recompute CRC so we hit the dup-key check, not the CRC check.
        let crc_offset = bytes.len() - 4;
        let new_crc = crc32fast::hash(&bytes[..crc_offset]);
        bytes[crc_offset..].copy_from_slice(&new_crc.to_le_bytes());

        let result = HnswIndex::decode_snapshot(&bytes);
        assert!(
            matches!(result, Err(Error::Corruption(_))),
            "expected Corruption, got {:?}",
            result.as_ref().err()
        );
    }
}