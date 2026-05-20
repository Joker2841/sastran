//! On-disk snapshot format for an [`HnswIndex`].
//!
//! Format (all multi-byte integers little-endian):
//!
//! ```text
//! Header (24 bytes):
//!   magic         8B   b"SASTHNS1"
//!   version       u32  = 1
//!   flags         u32  reserved, currently 0
//!   reserved      8B   zero
//!
//! Params (37 bytes):
//!   metric_tag    u8
//!   m             u32
//!   m_max_0       u32
//!   ef_constr     u32
//!   ef_search     u32
//!   dim           u32
//!   seed          u64
//!
//! Entry (6 bytes):
//!   present       u8   0 or 1
//!   node          u32  (meaningful if present == 1)
//!   level         u8
//!
//! Body:
//!   node_count    u32
//!   for each node (in NodeId order, including deleted slots):
//!       deleted     u8
//!       vec_len     u32  (should equal Params.dim)
//!       vec_data    f32 * dim
//!       layer_count u8
//!       for each layer:
//!           neighbor_count u32
//!           neighbor_ids   u32 * neighbor_count
//!
//! Trailer:
//!   crc32         u32  over all preceding bytes
//! ```
//!
//! ## What this format preserves
//!
//! - The full graph topology, including deleted-slot tombstones.
//!   Re-loading a snapshot produces an `HnswIndex` whose `len()` and
//!   `live_len()` match the original; deleted nodes stay deleted.
//! - The exact `HnswParams`, including the RNG seed (so a re-loaded
//!   index produces the same level assignments as the original on
//!   subsequent inserts).
//!
//! ## What this format does *not* preserve
//!
//! - The internal `SmallRng` state. After decode, the RNG is
//!   re-seeded from `params.seed`. Continued inserts will produce
//!   the same sequence as a fresh index would — fine for
//!   determinism, slightly different from "exactly resume."

use crate::hnsw::index::{DistanceMetric, HnswIndex, HnswParams, NodeId};
use crate::{Error, Result};

/// Magic bytes at the start of every snapshot file.
const MAGIC: &[u8; 8] = b"SASTHNS1";

/// Snapshot format version. Bumped on every breaking change.
///
/// v1 → v2: added `next_sstable_id: u64` to the params section so
/// recovery knows which SSTables are already reflected in the
/// snapshot.
/// v2 → v3: added engine-level key bytes to each node, after the
/// vector and before the neighbor layers. The HNSW's NodeId space is
/// internal; the engine needs key↔NodeId mapping for delete and
/// overwrite paths, and embedding the keys here lets us reconstruct
/// that map at recovery time without re-walking pre-snapshot
/// SSTables.
///
/// v1 and v2 files are no longer readable.
const VERSION: u32 = 3;

/// Header section size, in bytes.
const HEADER_LEN: usize = 8 + 4 + 4 + 8;

/// Params section size, in bytes. Includes the v2 addition.
const PARAMS_LEN: usize = 1 + 4 + 4 + 4 + 4 + 4 + 8 + 8;

/// Entry section size, in bytes.
const ENTRY_LEN: usize = 1 + 4 + 1;

impl DistanceMetric {
    fn to_tag(self) -> u8 {
        match self {
            DistanceMetric::Cosine => 0x01,
            DistanceMetric::EuclideanSquared => 0x02,
            DistanceMetric::InnerProduct => 0x03,
        }
    }

    fn from_tag(b: u8) -> Result<Self> {
        match b {
            0x01 => Ok(DistanceMetric::Cosine),
            0x02 => Ok(DistanceMetric::EuclideanSquared),
            0x03 => Ok(DistanceMetric::InnerProduct),
            other => Err(Error::Corruption(format!(
                "unknown distance metric tag {other:#04x}"
            ))),
        }
    }
}

impl HnswIndex {
    /// Serialize the entire index to a byte vector.
    ///
    /// `next_sstable_id` is the engine-level marker that this snapshot
    /// reflects all SSTables with `id < next_sstable_id`. Recovery
    /// uses it to decide which SSTables still need to be walked
    /// post-load. For an index not yet associated with any SSTables,
    /// pass `0`.
    ///
    /// `keys_by_node` maps engine keys to NodeIds. The encoder
    /// inverts this to look up each node's key during serialization.
    /// A NodeId without a mapping is encoded with an empty key — fine
    /// for deleted nodes (which the engine never looks up by key)
    /// but a corruption signal for live ones on decode.
    pub fn encode_snapshot(
        &self,
        next_sstable_id: u64,
        keys_by_node: &std::collections::HashMap<Vec<u8>, NodeId>,
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.estimated_snapshot_size());

        // Build NodeId → key reverse map for fast per-node lookup.
        // Most engines have far fewer than 2^32 keys, so we use a
        // Vec<Option<&[u8]>> indexed by NodeId.
        let mut keys_by_id: Vec<Option<&[u8]>> = vec![None; self.len()];
        for (key, &id) in keys_by_node {
            if (id as usize) < keys_by_id.len() {
                keys_by_id[id as usize] = Some(key.as_slice());
            }
        }

        // Header.
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // flags
        out.extend_from_slice(&[0u8; 8]); // reserved

        // Params.
        let params = self.params();
        out.push(self.metric().to_tag());
        out.extend_from_slice(&(params.m as u32).to_le_bytes());
        out.extend_from_slice(&(params.m_max_0 as u32).to_le_bytes());
        out.extend_from_slice(&(params.ef_construction as u32).to_le_bytes());
        out.extend_from_slice(&(params.ef_search as u32).to_le_bytes());
        out.extend_from_slice(&(self.dim() as u32).to_le_bytes());
        out.extend_from_slice(&params.seed.to_le_bytes());
        out.extend_from_slice(&next_sstable_id.to_le_bytes());

        // Entry.
        match self.entry_snapshot() {
            Some((node, level)) => {
                out.push(1);
                out.extend_from_slice(&node.to_le_bytes());
                out.push(level);
            }
            None => {
                out.push(0);
                out.extend_from_slice(&0u32.to_le_bytes());
                out.push(0);
            }
        }

        // Node count.
        out.extend_from_slice(&(self.len() as u32).to_le_bytes());

        // Nodes, in NodeId order.
        for id in 0..self.len() as NodeId {
            let key = keys_by_id[id as usize];
            self.encode_one_node(id, key, &mut out);
        }

        // Final CRC.
        let crc = crc32fast::hash(&out);
        out.extend_from_slice(&crc.to_le_bytes());

        out
    }

    /// Deserialize an index from a snapshot byte slice.
    ///
    /// Returns `(index, keys_by_node, next_sstable_id)`:
    /// - `index` is the reconstructed graph.
    /// - `keys_by_node` is the engine-level key→NodeId map needed by
    ///   the engine to route `delete(key)` and overwrite-on-`put_indexed`
    ///   operations to the right graph node.
    /// - `next_sstable_id` is the engine marker the snapshot was tagged
    ///   with at write time (see `encode_snapshot`).
    pub fn decode_snapshot(
        bytes: &[u8],
    ) -> Result<(Self, std::collections::HashMap<Vec<u8>, NodeId>, u64)> {
        // Sanity-check overall length.
        let min_len = HEADER_LEN + PARAMS_LEN + ENTRY_LEN + 4 /* node_count */ + 4 /* crc */;
        if bytes.len() < min_len {
            return Err(Error::Corruption(format!(
                "snapshot shorter than minimum: {} bytes, need at least {min_len}",
                bytes.len()
            )));
        }

        // Verify trailing CRC over everything before it.
        let crc_offset = bytes.len() - 4;
        let stored_crc = u32::from_le_bytes(bytes[crc_offset..].try_into().unwrap());
        let computed_crc = crc32fast::hash(&bytes[..crc_offset]);
        if stored_crc != computed_crc {
            return Err(Error::Corruption(format!(
                "snapshot CRC mismatch: stored {stored_crc:#010x}, \
                 computed {computed_crc:#010x}"
            )));
        }

        let body = &bytes[..crc_offset]; // CRC stripped; parse against this
        let mut cursor = 0usize;

        // Header: magic.
        if &body[cursor..cursor + 8] != MAGIC {
            return Err(Error::Corruption(format!(
                "snapshot magic mismatch: got {:?}",
                &body[cursor..cursor + 8]
            )));
        }
        cursor += 8;

        let version = read_u32(body, &mut cursor)?;
        if version != VERSION {
            return Err(Error::Corruption(format!(
                "unsupported snapshot version {version}, this build supports {VERSION}"
            )));
        }
        let _flags = read_u32(body, &mut cursor)?;
        // Skip reserved bytes.
        cursor += 8;

        // Params.
        let metric = DistanceMetric::from_tag(read_u8(body, &mut cursor)?)?;
        let m = read_u32(body, &mut cursor)? as usize;
        let m_max_0 = read_u32(body, &mut cursor)? as usize;
        let ef_construction = read_u32(body, &mut cursor)? as usize;
        let ef_search = read_u32(body, &mut cursor)? as usize;
        let dim = read_u32(body, &mut cursor)? as usize;
        let seed = read_u64(body, &mut cursor)?;
        let next_sstable_id = read_u64(body, &mut cursor)?;

        // Entry.
        let entry_present = read_u8(body, &mut cursor)?;
        let entry_node = read_u32(body, &mut cursor)?;
        let entry_level = read_u8(body, &mut cursor)?;
        let entry = if entry_present == 1 {
            Some((entry_node, entry_level))
        } else {
            None
        };

        // Build the index shell with reconstructed params.
        let params = HnswParams {
            metric,
            m,
            m_max_0,
            ef_construction,
            ef_search,
            dim,
            seed,
        };
        let mut index = HnswIndex::new(params);
        // `new()` doesn't know we have nodes — we populate them next.

        // Node count.
        let node_count = read_u32(body, &mut cursor)? as usize;

        // Per-node parsing. v3 inserts the key between vector data and
        // layer list.
        let mut decoded_nodes: Vec<RawNode> = Vec::with_capacity(node_count);
        let mut keys_by_node: std::collections::HashMap<Vec<u8>, NodeId> =
            std::collections::HashMap::with_capacity(node_count);
        for node_id in 0..node_count {
            let deleted = read_u8(body, &mut cursor)? != 0;
            let vec_len = read_u32(body, &mut cursor)? as usize;
            if vec_len != dim {
                return Err(Error::Corruption(format!(
                    "node {node_id}: vector length {vec_len} != params dim {dim}"
                )));
            }
            let mut vector = Vec::with_capacity(vec_len);
            for _ in 0..vec_len {
                vector.push(read_f32(body, &mut cursor)?);
            }
            // Key (v3): u32 length, then bytes.
            let key_len = read_u32(body, &mut cursor)? as usize;
            if cursor + key_len > body.len() {
                return Err(Error::Corruption(format!(
                    "node {node_id}: key bytes ({key_len}) extend past end of body"
                )));
            }
            let key_bytes = body[cursor..cursor + key_len].to_vec();
            cursor += key_len;
            // Empty keys are legal for deleted nodes only.
            if key_bytes.is_empty() && !deleted {
                return Err(Error::Corruption(format!(
                    "node {node_id}: live node has empty key"
                )));
            }
            // Live nodes populate the key map.
            if !deleted {
                if keys_by_node.contains_key(&key_bytes) {
                    return Err(Error::Corruption(format!(
                        "node {node_id}: duplicate key in snapshot"
                    )));
                }
                keys_by_node.insert(key_bytes, node_id as NodeId);
            }
            // Neighbors (unchanged).
            let layer_count = read_u8(body, &mut cursor)? as usize;
            let mut neighbors: Vec<Vec<NodeId>> = Vec::with_capacity(layer_count);
            for layer in 0..layer_count {
                let nc = read_u32(body, &mut cursor)? as usize;
                let mut ids = Vec::with_capacity(nc);
                for _ in 0..nc {
                    let nid = read_u32(body, &mut cursor)?;
                    if (nid as usize) >= node_count {
                        return Err(Error::Corruption(format!(
                            "node {node_id} layer {layer}: neighbor id {nid} \
                             out of range (node_count {node_count})"
                        )));
                    }
                    ids.push(nid);
                }
                neighbors.push(ids);
            }
            decoded_nodes.push(RawNode {
                vector,
                neighbors,
                deleted,
            });
        }

        // Final consistency: cursor should be at end of body.
        if cursor != body.len() {
            return Err(Error::Corruption(format!(
                "snapshot has trailing garbage: parsed {cursor} of {} body bytes",
                body.len()
            )));
        }

        // Cross-check entry against decoded nodes.
        if let Some((node, level)) = entry {
            if (node as usize) >= node_count {
                return Err(Error::Corruption(format!(
                    "entry node {node} out of range (node_count {node_count})"
                )));
            }
            let layers_present = decoded_nodes[node as usize].neighbors.len();
            if layers_present == 0 || (level as usize) >= layers_present {
                return Err(Error::Corruption(format!(
                    "entry level {level} inconsistent with node {node}'s \
                     layer count {layers_present}"
                )));
            }
        }

        // Hand the decoded nodes + entry to the index.
        index.set_state_from_snapshot(decoded_nodes, entry, dim);

        Ok((index, keys_by_node, next_sstable_id))
    }
}

/// Bytes-only representation of a node returned from snapshot decode.
/// Owned by the snapshot module; converted into the index's internal
/// `Node` type by [`HnswIndex::set_state_from_snapshot`].
pub(crate) struct RawNode {
    pub(crate) vector: Vec<f32>,
    pub(crate) neighbors: Vec<Vec<NodeId>>,
    pub(crate) deleted: bool,
}

fn read_u8(buf: &[u8], cursor: &mut usize) -> Result<u8> {
    if *cursor + 1 > buf.len() {
        return Err(Error::Corruption("unexpected end of snapshot (u8)".into()));
    }
    let v = buf[*cursor];
    *cursor += 1;
    Ok(v)
}

fn read_u32(buf: &[u8], cursor: &mut usize) -> Result<u32> {
    if *cursor + 4 > buf.len() {
        return Err(Error::Corruption("unexpected end of snapshot (u32)".into()));
    }
    let v = u32::from_le_bytes(buf[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

fn read_u64(buf: &[u8], cursor: &mut usize) -> Result<u64> {
    if *cursor + 8 > buf.len() {
        return Err(Error::Corruption("unexpected end of snapshot (u64)".into()));
    }
    let v = u64::from_le_bytes(buf[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(v)
}

fn read_f32(buf: &[u8], cursor: &mut usize) -> Result<f32> {
    if *cursor + 4 > buf.len() {
        return Err(Error::Corruption("unexpected end of snapshot (f32)".into()));
    }
    let v = f32::from_le_bytes(buf[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}