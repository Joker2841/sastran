//! Bloom filter for SSTable point lookups.
//!
//! A [`BloomFilter`] answers "is this key *definitely not* in the
//! SSTable?" with one bit of certainty per stored key, at the cost of
//! a small false-positive rate (default ~1%). Used by the SSTable
//! reader to skip the data-block read when the answer is "definitely
//! not present" — the dominant cost of a `get` that doesn't exist.
//!
//! ## Algorithm
//!
//! Standard bloom filter with the double-hashing trick of
//! Kirsch-Mitzenmacher (2008): hash each key once into a 128-bit
//! digest, split into two 64-bit halves `h1` and `h2`, and synthesize
//! the `k` per-bit positions as `(h1 + i * h2) mod m`. The result is
//! statistically equivalent to `k` truly independent hashes while
//! costing one hash per insert or query.
//!
//! ## Sizing
//!
//! For `n` keys and target false-positive rate `p`:
//!   bits per key  m/n = -log2(p) / ln(2)  ≈ 1.44 * log2(1/p)
//!   hash count    k   = (m/n) * ln(2)
//!
//! At 10 bits per key, k = 7, false-positive rate ≈ 0.82%.

use crate::{Error, Result};

/// In-memory bloom filter built from a stream of keys.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// Number of bits in the array. Stored as u64 so the format can
    /// represent very large filters; in practice m fits in usize.
    m: u64,
    /// Number of hash functions per key.
    k: u8,
    /// The bit array, packed little-endian within each byte
    /// (bit 0 = least significant).
    bits: Vec<u8>,
}

impl BloomFilter {
    /// Create a new empty filter sized for `expected_keys` at the given
    /// `bits_per_key`. A `bits_per_key` of 10 yields roughly 1%
    /// false-positive rate; smaller values mean a smaller filter at a
    /// higher false-positive rate. Caller-visible at the Options level.
    pub fn new(expected_keys: usize, bits_per_key: u32) -> Self {
        assert!(bits_per_key > 0, "bits_per_key must be > 0");
        let m = (expected_keys as u64) * (bits_per_key as u64);
        // Minimum size: m must be at least 64 so the synthetic-hash
        // arithmetic has something to do. Empty filters round up.
        let m = m.max(64);
        // Optimal k = (m/n) * ln(2). We approximate ln(2) ≈ 0.69 with
        // integer math: k ≈ bits_per_key * 69 / 100, clamped to [1, 30].
        let k = ((bits_per_key as u64 * 69) / 100).clamp(1, 30) as u8;
        let bytes = (m as usize).div_ceil(8);
        Self {
            m,
            k,
            bits: vec![0u8; bytes],
        }
    }

    /// Record `key` in the filter.
    pub fn insert(&mut self, key: &[u8]) {
        let (h1, h2) = double_hash(key);
        for i in 0..self.k {
            let bit = synth_index(h1, h2, i, self.m);
            self.set_bit(bit);
        }
    }

    /// Test whether `key` *might* be present. False positives possible;
    /// false negatives never.
    pub fn contains(&self, key: &[u8]) -> bool {
        let (h1, h2) = double_hash(key);
        for i in 0..self.k {
            let bit = synth_index(h1, h2, i, self.m);
            if !self.get_bit(bit) {
                return false;
            }
        }
        true
    }

    /// Number of bits in the array.
    pub fn bit_count(&self) -> u64 {
        self.m
    }

    /// Number of hash functions per key.
    pub fn hash_count(&self) -> u8 {
        self.k
    }

    /// Serialized size, including the trailing CRC.
    pub fn encoded_len(&self) -> usize {
        1 + 8 + self.bits.len() + 4
    }

    /// Serialize: `k(u8) || m(u64 LE) || bits || crc32(LE)`.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        out.push(self.k);
        out.extend_from_slice(&self.m.to_le_bytes());
        out.extend_from_slice(&self.bits);
        let crc = crc32fast::hash(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }

    /// Parse a serialized filter. Validates the trailing CRC.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 1 + 8 + 4 {
            return Err(Error::Corruption(
                "bloom block too short for header + CRC".into(),
            ));
        }
        let split = bytes.len() - 4;
        let stored_crc = u32::from_le_bytes(bytes[split..].try_into().unwrap());
        let computed_crc = crc32fast::hash(&bytes[..split]);
        if stored_crc != computed_crc {
            return Err(Error::Corruption(format!(
                "bloom block CRC mismatch: stored {stored_crc:#010x}, computed {computed_crc:#010x}"
            )));
        }

        let k = bytes[0];
        let m = u64::from_le_bytes(bytes[1..9].try_into().unwrap());
        let bit_bytes = &bytes[9..split];

        // Sanity-check that the bit-array length matches the declared m.
        let expected_bytes = (m as usize).div_ceil(8);
        if bit_bytes.len() != expected_bytes {
            return Err(Error::Corruption(format!(
                "bloom bit-array size mismatch: declared m = {m} ({expected_bytes} bytes), got {} bytes",
                bit_bytes.len()
            )));
        }
        if k == 0 {
            return Err(Error::Corruption("bloom k = 0".into()));
        }

        Ok(Self {
            m,
            k,
            bits: bit_bytes.to_vec(),
        })
    }

    fn set_bit(&mut self, bit: u64) {
        let byte = (bit / 8) as usize;
        let mask = 1u8 << (bit % 8);
        self.bits[byte] |= mask;
    }

    fn get_bit(&self, bit: u64) -> bool {
        let byte = (bit / 8) as usize;
        let mask = 1u8 << (bit % 8);
        self.bits[byte] & mask != 0
    }
}

/// Hash `key` into two 64-bit halves of a 128-bit xxh3 digest.
fn double_hash(key: &[u8]) -> (u64, u64) {
    let digest = xxhash_rust::xxh3::xxh3_128(key);
    let h1 = digest as u64;
    let h2 = (digest >> 64) as u64;
    (h1, h2)
}

/// Compute the i-th bit position via the Kirsch-Mitzenmacher trick.
/// Uses wrapping arithmetic to avoid overflow on very large m * i.
fn synth_index(h1: u64, h2: u64, i: u8, m: u64) -> u64 {
    h1.wrapping_add((i as u64).wrapping_mul(h2)) % m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inserted_keys_test_present() {
        let mut bf = BloomFilter::new(100, 10);
        let keys: Vec<Vec<u8>> = (0..100u32)
            .map(|i| format!("key_{i:03}").into_bytes())
            .collect();
        for k in &keys {
            bf.insert(k);
        }
        for k in &keys {
            assert!(bf.contains(k), "missing key: {k:?}");
        }
    }

    #[test]
    fn absent_keys_mostly_test_absent() {
        // Sanity check on false-positive rate. With 10 bits/key and 1000
        // inserted keys, querying 10000 unrelated keys should produce far
        // fewer than 1000 false positives (expected ~100, i.e. 1%).
        // We assert a loose bound to keep the test robust to RNG variance.
        let mut bf = BloomFilter::new(1000, 10);
        for i in 0..1000u32 {
            bf.insert(format!("inserted_{i:05}").as_bytes());
        }
        let mut false_positives = 0;
        for i in 0..10_000u32 {
            // "absent_" keys are not in the filter.
            if bf.contains(format!("absent_{i:05}").as_bytes()) {
                false_positives += 1;
            }
        }
        // Expected ~100, give a wide margin (factor of 5) for variance.
        assert!(
            false_positives < 500,
            "too many false positives: {false_positives}/10000"
        );
    }

    #[test]
    fn empty_filter_says_no_to_everything() {
        let bf = BloomFilter::new(100, 10);
        // No keys inserted. Every query should return false.
        for i in 0..1000u32 {
            assert!(!bf.contains(format!("k_{i}").as_bytes()));
        }
    }

    #[test]
    fn encode_decode_round_trips() {
        let mut bf = BloomFilter::new(500, 10);
        for i in 0..500u32 {
            bf.insert(format!("k_{i:04}").as_bytes());
        }
        let bytes = bf.encode();
        assert_eq!(bytes.len(), bf.encoded_len());

        let decoded = BloomFilter::decode(&bytes).unwrap();
        assert_eq!(decoded.bit_count(), bf.bit_count());
        assert_eq!(decoded.hash_count(), bf.hash_count());

        // Every previously-inserted key must still test present.
        for i in 0..500u32 {
            assert!(decoded.contains(format!("k_{i:04}").as_bytes()));
        }
    }

    #[test]
    fn decode_detects_crc_corruption() {
        let mut bf = BloomFilter::new(100, 10);
        bf.insert(b"x");
        let mut bytes = bf.encode();
        // Flip a bit in the bit-array.
        bytes[12] ^= 0x01;
        let err = BloomFilter::decode(&bytes).unwrap_err();
        assert!(matches!(err, Error::Corruption(_)));
    }

    #[test]
    fn decode_detects_size_mismatch() {
        let mut bf = BloomFilter::new(100, 10);
        bf.insert(b"x");
        let mut bytes = bf.encode();
        // Bump the declared m to a value that won't match the bit-array size.
        // m is at bytes [1..9]. Set m to u64::MAX so size check fails.
        bytes[1..9].copy_from_slice(&u64::MAX.to_le_bytes());
        // Recompute CRC so the CRC check passes and we hit the size check.
        let split = bytes.len() - 4;
        let new_crc = crc32fast::hash(&bytes[..split]);
        bytes[split..].copy_from_slice(&new_crc.to_le_bytes());

        let err = BloomFilter::decode(&bytes).unwrap_err();
        assert!(matches!(err, Error::Corruption(_)));
    }

    #[test]
    fn tiny_inputs_get_minimum_size() {
        // Even a 1-key filter rounds up to at least 64 bits, so the
        // synthetic-hash arithmetic has a non-degenerate modulus.
        let bf = BloomFilter::new(1, 10);
        assert!(bf.bit_count() >= 64);
    }

    #[test]
    fn different_bits_per_key_give_different_hash_counts() {
        let lo = BloomFilter::new(100, 4);
        let hi = BloomFilter::new(100, 20);
        assert!(hi.hash_count() > lo.hash_count());
    }
}