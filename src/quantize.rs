//! 8-bit scalar quantization for embedding vectors.
//!
//! Maps each f32 component of a vector to a single byte using
//! per-dimension min/max scaling, cutting storage 4×. The quantizer
//! is trained on a sample of vectors to learn each dimension's range,
//! then frozen.
//!
//! ## Accuracy
//!
//! Quantization is lossy. The maximum error per component is half a
//! quantization step: `(max[d] - min[d]) / 510`. For well-distributed
//! embeddings this is small relative to the values, so approximate
//! nearest-neighbor recall typically drops only slightly (a few
//! percent) in exchange for the 4× memory reduction.
//!
//! ## Per-dimension scaling
//!
//! We learn min/max *per dimension* rather than globally. Embedding
//! dimensions often have very different ranges; a single global scale
//! would waste resolution on wide dimensions and crush narrow ones. A
//! per-dimension scale gives every dimension the full 8-bit range.
//!
//! ## Out-of-range values
//!
//! A vector quantized after training may have components outside the
//! learned `[min, max]`. These saturate to 0 or 255 rather than
//! wrapping — accuracy suffers for that component, but the result is
//! always well-defined.
//!
//! ## Status
//!
//! This is a standalone, measured component. It is not yet wired into
//! the live HNSW index (which still stores full-precision f32). The
//! `benches/quantization.rs` benchmark measures the recall/memory
//! tradeoff on quantized-then-dequantized data. Native u8 storage in
//! the index is documented future work.

use crate::{Error, Result};

/// A trained per-dimension 8-bit scalar quantizer.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarQuantizer {
    /// Per-dimension minimum learned at training time.
    min: Vec<f32>,
    /// Per-dimension maximum learned at training time.
    max: Vec<f32>,
}

impl ScalarQuantizer {
    /// Train a quantizer from a sample of vectors. All vectors must
    /// share the same non-zero dimension.
    ///
    /// Returns `InvalidArgument` if the sample is empty, if any vector
    /// has a mismatched dimension, or if any component is non-finite.
    pub fn train(sample: &[Vec<f32>]) -> Result<Self> {
        let first = sample.first().ok_or_else(|| {
            Error::InvalidArgument("cannot train quantizer on empty sample".into())
        })?;
        let dim = first.len();
        if dim == 0 {
            return Err(Error::InvalidArgument(
                "cannot train quantizer on zero-dimension vectors".into(),
            ));
        }

        let mut min = vec![f32::INFINITY; dim];
        let mut max = vec![f32::NEG_INFINITY; dim];

        for (i, v) in sample.iter().enumerate() {
            if v.len() != dim {
                return Err(Error::InvalidArgument(format!(
                    "training vector {i} has dim {} but expected {dim}",
                    v.len()
                )));
            }
            for (d, &x) in v.iter().enumerate() {
                if !x.is_finite() {
                    return Err(Error::InvalidArgument(format!(
                        "training vector {i} component {d} is non-finite"
                    )));
                }
                if x < min[d] {
                    min[d] = x;
                }
                if x > max[d] {
                    max[d] = x;
                }
            }
        }

        Ok(Self { min, max })
    }

    /// The dimension this quantizer was trained for.
    pub fn dim(&self) -> usize {
        self.min.len()
    }

    /// Quantize a vector to one byte per dimension.
    ///
    /// Components outside the trained range saturate to 0 or 255.
    /// Returns `InvalidArgument` on dimension mismatch.
    pub fn quantize(&self, v: &[f32]) -> Result<Vec<u8>> {
        if v.len() != self.dim() {
            return Err(Error::InvalidArgument(format!(
                "quantize: vector dim {} != quantizer dim {}",
                v.len(),
                self.dim()
            )));
        }
        let mut out = Vec::with_capacity(v.len());
        for (d, &x) in v.iter().enumerate() {
            let range = self.max[d] - self.min[d];
            let q = if range == 0.0 {
                // Degenerate dimension: all training values equal. Map
                // everything to 0; dequantize will return the constant.
                0u8
            } else {
                let scaled = (x - self.min[d]) / range * 255.0;
                // Clamp out-of-range / NaN-safe rounding.
                scaled.round().clamp(0.0, 255.0) as u8
            };
            out.push(q);
        }
        Ok(out)
    }

    /// Dequantize a byte vector back to approximate f32.
    ///
    /// Returns `InvalidArgument` on dimension mismatch.
    pub fn dequantize(&self, q: &[u8]) -> Result<Vec<f32>> {
        if q.len() != self.dim() {
            return Err(Error::InvalidArgument(format!(
                "dequantize: byte vector dim {} != quantizer dim {}",
                q.len(),
                self.dim()
            )));
        }
        let mut out = Vec::with_capacity(q.len());
        for (d, &b) in q.iter().enumerate() {
            let range = self.max[d] - self.min[d];
            let x = if range == 0.0 {
                self.min[d]
            } else {
                self.min[d] + (b as f32 / 255.0) * range
            };
            out.push(x);
        }
        Ok(out)
    }

    /// Convenience: quantize then immediately dequantize, yielding the
    /// vector as the index would "see" it after lossy compression.
    /// Used by benchmarks and accuracy tests.
    pub fn round_trip(&self, v: &[f32]) -> Result<Vec<f32>> {
        let q = self.quantize(v)?;
        self.dequantize(&q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_rejects_empty_sample() {
        let err = ScalarQuantizer::train(&[]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn train_rejects_zero_dim() {
        let err = ScalarQuantizer::train(&[vec![]]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn train_rejects_dim_mismatch() {
        let sample = vec![vec![1.0, 2.0], vec![1.0, 2.0, 3.0]];
        let err = ScalarQuantizer::train(&sample).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn train_rejects_non_finite() {
        let sample = vec![vec![1.0, f32::NAN]];
        let err = ScalarQuantizer::train(&sample).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn round_trip_is_close_for_in_range_values() {
        // Train on a spread of values, then check round-trip error is
        // bounded by half a quantization step.
        let sample = vec![
            vec![0.0, -10.0],
            vec![1.0, 10.0],
            vec![0.5, 0.0],
        ];
        let q = ScalarQuantizer::train(&sample).unwrap();

        let test = vec![0.3, 4.2];
        let recovered = q.round_trip(&test).unwrap();

        // Dimension 0: range = 1.0, step = 1/255, max error = 1/510 ≈ 0.00196.
        assert!((recovered[0] - 0.3).abs() <= 1.0 / 510.0 + 1e-6);
        // Dimension 1: range = 20.0, step = 20/255, max error = 20/510 ≈ 0.0392.
        assert!((recovered[1] - 4.2).abs() <= 20.0 / 510.0 + 1e-6);
    }

    #[test]
    fn endpoints_round_trip_nearly_exactly() {
        let sample = vec![vec![2.0], vec![8.0]];
        let q = ScalarQuantizer::train(&sample).unwrap();
        // min endpoint → byte 0 → back to min.
        let lo = q.round_trip(&[2.0]).unwrap();
        assert!((lo[0] - 2.0).abs() < 1e-4);
        // max endpoint → byte 255 → back to max.
        let hi = q.round_trip(&[8.0]).unwrap();
        assert!((hi[0] - 8.0).abs() < 1e-4);
    }

    #[test]
    fn out_of_range_values_saturate() {
        let sample = vec![vec![0.0], vec![1.0]];
        let q = ScalarQuantizer::train(&sample).unwrap();

        // Below min saturates to byte 0.
        assert_eq!(q.quantize(&[-5.0]).unwrap(), vec![0u8]);
        // Above max saturates to byte 255.
        assert_eq!(q.quantize(&[5.0]).unwrap(), vec![255u8]);
    }

    #[test]
    fn degenerate_dimension_is_handled() {
        // A dimension where every training value is identical (range 0).
        let sample = vec![vec![3.0, 1.0], vec![3.0, 9.0]];
        let q = ScalarQuantizer::train(&sample).unwrap();
        // Dimension 0 is constant at 3.0.
        let recovered = q.round_trip(&[3.0, 5.0]).unwrap();
        assert!((recovered[0] - 3.0).abs() < 1e-6, "constant dim should recover exactly");
        // Even an out-of-training value on the constant dim recovers to
        // the constant (no NaN, no panic).
        let recovered2 = q.round_trip(&[999.0, 5.0]).unwrap();
        assert!((recovered2[0] - 3.0).abs() < 1e-6);
        assert!(recovered2[1].is_finite());
    }

    #[test]
    fn quantize_rejects_dim_mismatch() {
        let q = ScalarQuantizer::train(&[vec![0.0, 1.0]]).unwrap();
        let err = q.quantize(&[1.0]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn dequantize_rejects_dim_mismatch() {
        let q = ScalarQuantizer::train(&[vec![0.0, 1.0]]).unwrap();
        let err = q.dequantize(&[128u8]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn quantized_output_is_one_byte_per_dim() {
        let q = ScalarQuantizer::train(&[vec![0.0; 128], vec![1.0; 128]]).unwrap();
        let v = vec![0.5f32; 128];
        let bytes = q.quantize(&v).unwrap();
        assert_eq!(bytes.len(), 128, "one byte per dimension");
        // vs. 128 * 4 = 512 bytes at full precision: 4x compression.
    }
}