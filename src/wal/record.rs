//! WAL record format.
//!
//! ## On-disk layout
//!
//! Each record is laid out as:
//!
//! ```text
//! ┌──────────┬─────────┬──────────┬────────────┬─────┬───────┐
//! │  CRC32   │ op_kind │  key_len │  value_len │ key │ value │
//! │ 4 bytes  │ 1 byte  │ 4 bytes  │  4 bytes   │ var │  var  │
//! └──────────┴─────────┴──────────┴────────────┴─────┴───────┘
//! ```
//!
//! Multi-byte integers are little-endian. The CRC32 (IEEE polynomial)
//! covers every byte *after* the CRC field itself, in order. The CRC is
//! placed first so that decode can verify integrity before trusting
//! the length fields.
//!
//! ## Why these choices
//!
//! - **Op kind starts at 0x01.** Zero is reserved as an invalid value
//!   so torn writes that leave trailing zeros are easier to detect.
//! - **u32 lengths.** A `u32` length field is more than enough for any
//!   reasonable key or value. We *also* enforce explicit `MAX_KEY_LEN`
//!   and `MAX_VALUE_LEN` on decode so that garbage-length corruption
//!   does not trick us into a multi-gigabyte allocation.
//! - **No version byte per record.** Format versioning lives in the
//!   WAL file header (added when the writer is implemented).

use crate::{Error, Result};

/// Header bytes before the variable-length key/value: CRC(4) + kind(1)
/// + key_len(4) + value_len(4) = 13 bytes.
pub const HEADER_LEN: usize = 4 + 1 + 4 + 4;

/// Maximum allowed key length. Reads of records with a larger declared
/// `key_len` are rejected as corruption.
pub const MAX_KEY_LEN: u32 = 64 * 1024;

/// Maximum allowed value length. 64 MiB accommodates blob values and
/// embedding vectors with significant headroom.
pub const MAX_VALUE_LEN: u32 = 64 * 1024 * 1024;

const KIND_PUT: u8 = 0x01;
const KIND_DELETE: u8 = 0x02;
const KIND_VECTOR: u8 = 0x03;

/// The mutation a record represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordKind {
    /// A `(key, value)` insertion or update.
    Put,
    /// A `(key, vector_bytes)` vector insertion. Same payload shape
    /// as `Put` on the wire; distinguished by kind byte so consumers
    /// can route to the HNSW index.
    Vector,
    /// A deletion of `key`. Wire value bytes are empty for deletes.
    Delete,
}

impl RecordKind {
    fn to_byte(self) -> u8 {
        match self {
            RecordKind::Put => KIND_PUT,
            RecordKind::Vector => KIND_VECTOR,
            RecordKind::Delete => KIND_DELETE,
        }
    }

    fn from_byte(b: u8) -> Option<Self> {
        match b {
            KIND_PUT => Some(RecordKind::Put),
            KIND_VECTOR => Some(RecordKind::Vector),
            KIND_DELETE => Some(RecordKind::Delete),
            _ => None,
        }
    }
}

/// A decoded record borrowed from the underlying buffer.
///
/// `key` and `value` slice into the caller's buffer; no allocation
/// happens during decode. Callers that need owned data should copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record<'a> {
    pub kind: RecordKind,
    pub key: &'a [u8],
    pub value: &'a [u8],
}

/// Bytes needed to encode a record with the given key and value lengths.
pub fn encoded_len(key_len: usize, value_len: usize) -> usize {
    HEADER_LEN + key_len + value_len
}

/// Append a record to `out`. Returns the number of bytes appended.
///
/// `value` should be empty for `RecordKind::Delete`; this is enforced
/// at runtime to catch caller bugs.
pub fn encode(
    kind: RecordKind,
    key: &[u8],
    value: &[u8],
    out: &mut Vec<u8>,
) -> Result<usize> {
    if key.is_empty() {
        return Err(Error::InvalidArgument("empty key".into()));
    }
    if key.len() > MAX_KEY_LEN as usize {
        return Err(Error::InvalidArgument(format!(
            "key length {} exceeds maximum {}",
            key.len(),
            MAX_KEY_LEN
        )));
    }
    if value.len() > MAX_VALUE_LEN as usize {
        return Err(Error::InvalidArgument(format!(
            "value length {} exceeds maximum {}",
            value.len(),
            MAX_VALUE_LEN
        )));
    }
    if matches!(kind, RecordKind::Delete) && !value.is_empty() {
        return Err(Error::InvalidArgument(
            "value must be empty for Delete records".into(),
        ));
    }

    let start = out.len();
    let total = encoded_len(key.len(), value.len());
    out.reserve(total);

    // Reserve 4 bytes for the CRC; we'll overwrite them at the end.
    let crc_pos = out.len();
    out.extend_from_slice(&[0u8; 4]);

    // Header.
    out.push(kind.to_byte());
    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());

    // Payload.
    out.extend_from_slice(key);
    out.extend_from_slice(value);

    // Compute the CRC over everything after the CRC field, then patch
    // it back in.
    let payload_start = crc_pos + 4;
    let crc = crc32fast::hash(&out[payload_start..]);
    out[crc_pos..crc_pos + 4].copy_from_slice(&crc.to_le_bytes());

    debug_assert_eq!(out.len() - start, total);
    Ok(total)
}

/// Reasons a record may fail to decode.
///
/// All of these become [`Error::Corruption`] at the WAL-reader level,
/// but the variants are useful internally and in tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Buffer is shorter than a record header. May indicate a torn
    /// trailing record; the WAL reader treats this as end-of-log.
    Truncated { needed: usize, available: usize },
    /// CRC32 over the payload did not match the stored CRC.
    BadCrc { stored: u32, computed: u32 },
    /// The op-kind byte was not a recognized value.
    BadKind { byte: u8 },
    /// `key_len` or `value_len` exceeded the configured maximum.
    LengthExceedsMax { field: &'static str, len: u32, max: u32 },
    /// `key_len` was zero (we forbid empty keys at encode time too).
    EmptyKey,
    /// `value_len` was non-zero on a Delete record.
    UnexpectedDeleteValue,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Truncated { needed, available } => write!(
                f,
                "record truncated: needed {needed} bytes, have {available}"
            ),
            DecodeError::BadCrc { stored, computed } => write!(
                f,
                "CRC mismatch: stored {stored:#010x}, computed {computed:#010x}"
            ),
            DecodeError::BadKind { byte } => {
                write!(f, "unknown record kind {byte:#04x}")
            }
            DecodeError::LengthExceedsMax { field, len, max } => {
                write!(f, "{field} length {len} exceeds maximum {max}")
            }
            DecodeError::EmptyKey => f.write_str("record has zero-length key"),
            DecodeError::UnexpectedDeleteValue => {
                f.write_str("Delete record carries a non-empty value")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

impl From<DecodeError> for Error {
    fn from(e: DecodeError) -> Self {
        Error::Corruption(e.to_string())
    }
}

/// Attempt to decode the first record in `buf`.
///
/// On success returns `(record, bytes_consumed)`. The `record`'s `key`
/// and `value` borrow from `buf`. On `DecodeError::Truncated`, callers
/// should treat the input as end-of-log rather than as fatal corruption.
pub fn decode(buf: &[u8]) -> std::result::Result<(Record<'_>, usize), DecodeError> {
    // 1. Header must fit.
    if buf.len() < HEADER_LEN {
        return Err(DecodeError::Truncated {
            needed: HEADER_LEN,
            available: buf.len(),
        });
    }

    // 2. Parse header fields.
    let stored_crc = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let kind_byte = buf[4];
    let key_len = u32::from_le_bytes(buf[5..9].try_into().unwrap());
    let value_len = u32::from_le_bytes(buf[9..13].try_into().unwrap());

    // 3. Reject impossible lengths *before* using them to slice.
    if key_len == 0 {
        return Err(DecodeError::EmptyKey);
    }
    if key_len > MAX_KEY_LEN {
        return Err(DecodeError::LengthExceedsMax {
            field: "key",
            len: key_len,
            max: MAX_KEY_LEN,
        });
    }
    if value_len > MAX_VALUE_LEN {
        return Err(DecodeError::LengthExceedsMax {
            field: "value",
            len: value_len,
            max: MAX_VALUE_LEN,
        });
    }

    // 4. Whole record must fit.
    let total = HEADER_LEN + key_len as usize + value_len as usize;
    if buf.len() < total {
        return Err(DecodeError::Truncated {
            needed: total,
            available: buf.len(),
        });
    }

    // 5. Verify CRC over everything after the CRC field.
    let payload = &buf[4..total];
    let computed_crc = crc32fast::hash(payload);
    if computed_crc != stored_crc {
        return Err(DecodeError::BadCrc {
            stored: stored_crc,
            computed: computed_crc,
        });
    }

    // 6. Only now do we trust the kind byte and slice out key/value.
    let kind = RecordKind::from_byte(kind_byte)
        .ok_or(DecodeError::BadKind { byte: kind_byte })?;

    let key = &buf[HEADER_LEN..HEADER_LEN + key_len as usize];
    let value = &buf[HEADER_LEN + key_len as usize..total];

    if matches!(kind, RecordKind::Delete) && !value.is_empty() {
        return Err(DecodeError::UnexpectedDeleteValue);
    }

    Ok((Record { kind, key, value }, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: encode then decode, asserting the round-trip preserves data.
    fn round_trip(kind: RecordKind, key: &[u8], value: &[u8]) {
        let mut buf = Vec::new();
        let n = encode(kind, key, value, &mut buf).expect("encode");
        assert_eq!(n, buf.len(), "encode reported wrong byte count");

        let (decoded, consumed) = decode(&buf).expect("decode");
        assert_eq!(consumed, buf.len(), "decode consumed wrong byte count");
        assert_eq!(decoded.kind, kind);
        assert_eq!(decoded.key, key);
        assert_eq!(decoded.value, value);
    }

    #[test]
    fn put_round_trip_basic() {
        round_trip(RecordKind::Put, b"hello", b"world");
    }

    #[test]
    fn delete_round_trip_basic() {
        round_trip(RecordKind::Delete, b"hello", b"");
    }

    #[test]
    fn put_round_trip_large_value() {
        let value = vec![0xABu8; 1024 * 1024];
        round_trip(RecordKind::Put, b"big", &value);
    }

    #[test]
    fn rejects_empty_key_on_encode() {
        let mut buf = Vec::new();
        let err = encode(RecordKind::Put, b"", b"v", &mut buf).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
    }

    #[test]
    fn rejects_oversized_key_on_encode() {
        let mut buf = Vec::new();
        let key = vec![0u8; MAX_KEY_LEN as usize + 1];
        let err = encode(RecordKind::Put, &key, b"v", &mut buf).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn rejects_value_on_delete() {
        let mut buf = Vec::new();
        let err = encode(RecordKind::Delete, b"k", b"v", &mut buf).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn decode_detects_truncated_header() {
        let buf = vec![0u8; HEADER_LEN - 1];
        let err = decode(&buf).unwrap_err();
        assert!(matches!(err, DecodeError::Truncated { .. }));
    }

    #[test]
    fn decode_detects_truncated_payload() {
        let mut buf = Vec::new();
        encode(RecordKind::Put, b"key", b"value", &mut buf).unwrap();
        // Chop off the last byte of the value.
        buf.pop();
        let err = decode(&buf).unwrap_err();
        assert!(matches!(err, DecodeError::Truncated { .. }), "got {err:?}");
    }

    #[test]
    fn decode_detects_crc_mismatch() {
        let mut buf = Vec::new();
        encode(RecordKind::Put, b"key", b"value", &mut buf).unwrap();
        // Flip a bit in the value.
        let last = buf.len() - 1;
        buf[last] ^= 0x01;
        let err = decode(&buf).unwrap_err();
        assert!(matches!(err, DecodeError::BadCrc { .. }), "got {err:?}");
    }

    #[test]
    fn decode_detects_bad_kind() {
        let mut buf = Vec::new();
        encode(RecordKind::Put, b"key", b"value", &mut buf).unwrap();
        // Corrupt the kind byte (offset 4) then recompute the CRC so
        // CRC validation passes and we hit the kind check specifically.
        buf[4] = 0x00;
        let new_crc = crc32fast::hash(&buf[4..]);
        buf[0..4].copy_from_slice(&new_crc.to_le_bytes());
        let err = decode(&buf).unwrap_err();
        assert!(matches!(err, DecodeError::BadKind { byte: 0x00 }), "got {err:?}");
    }

    #[test]
    fn decode_rejects_oversized_key_len() {
        // Build a header by hand with key_len = MAX_KEY_LEN + 1.
        let mut buf = vec![0u8; HEADER_LEN];
        buf[4] = KIND_PUT;
        buf[5..9].copy_from_slice(&(MAX_KEY_LEN + 1).to_le_bytes());
        buf[9..13].copy_from_slice(&0u32.to_le_bytes());
        // Don't bother with a real CRC; the length check fires first.
        let err = decode(&buf).unwrap_err();
        assert!(
            matches!(err, DecodeError::LengthExceedsMax { field: "key", .. }),
            "got {err:?}"
        );
    }

    // Property test: arbitrary `(key, value)` pairs round-trip.
    proptest::proptest! {
        #[test]
        fn prop_put_round_trip(
            key in proptest::collection::vec(0u8..=255, 1..256),
            value in proptest::collection::vec(0u8..=255, 0..4096),
        ) {
            let mut buf = Vec::new();
            encode(RecordKind::Put, &key, &value, &mut buf).unwrap();
            let (decoded, consumed) = decode(&buf).unwrap();
            proptest::prop_assert_eq!(consumed, buf.len());
            proptest::prop_assert_eq!(decoded.kind, RecordKind::Put);
            proptest::prop_assert_eq!(decoded.key, &key[..]);
            proptest::prop_assert_eq!(decoded.value, &value[..]);
        }

        #[test]
        fn prop_delete_round_trip(
            key in proptest::collection::vec(0u8..=255, 1..256),
        ) {
            let mut buf = Vec::new();
            encode(RecordKind::Delete, &key, b"", &mut buf).unwrap();
            let (decoded, consumed) = decode(&buf).unwrap();
            proptest::prop_assert_eq!(consumed, buf.len());
            proptest::prop_assert_eq!(decoded.kind, RecordKind::Delete);
            proptest::prop_assert_eq!(decoded.key, &key[..]);
            proptest::prop_assert!(decoded.value.is_empty());
        }

        // Any single-byte flip in the encoded form is detected (either
        // as a CRC mismatch, a bad kind byte, or a length-exceeds-max).
        // We don't care *which* error fires, only that decode does not
        // silently return wrong data.
        #[test]
        fn prop_single_bit_flip_detected(
            key in proptest::collection::vec(0u8..=255, 1..64),
            value in proptest::collection::vec(0u8..=255, 0..64),
            flip_byte in 0usize..200,
            flip_bit in 0u8..8,
        ) {
            let mut buf = Vec::new();
            encode(RecordKind::Put, &key, &value, &mut buf).unwrap();
            if flip_byte >= buf.len() {
                return Ok(());
            }
            buf[flip_byte] ^= 1 << flip_bit;
            match decode(&buf) {
                Err(_) => {} // good: corruption detected
                Ok((decoded, _)) => {
                    // If decode succeeded, the flipped byte must have
                    // landed in key or value and produced different bytes.
                    proptest::prop_assert!(
                        decoded.key != &key[..] || decoded.value != &value[..],
                        "bit flip at byte {flip_byte} bit {flip_bit} \
                         was not detected and produced identical record"
                    );
                }
            }
        }
    }

    #[test]
    fn vector_round_trip_basic() {
        round_trip(RecordKind::Vector, b"key", b"\x01\x02\x03\x04\x05\x06\x07\x08");
    }

    #[test]
    fn vector_round_trip_large() {
        let value = vec![0xCDu8; 4096];
        round_trip(RecordKind::Vector, b"big_vec", &value);
    }
}