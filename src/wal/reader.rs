//! WAL reader: streaming replay with torn-tail tolerance.
//!
//! A [`WalReader`] opens a WAL file, validates its header, and yields
//! records one at a time. The reader distinguishes two end-of-log
//! conditions:
//!
//! - **Clean end:** the file ends exactly on a record boundary. `next`
//!   returns `Ok(None)`.
//! - **Torn tail:** the file ends in the middle of a record (e.g. a
//!   crash mid-write). `next` also returns `Ok(None)` — the truncated
//!   record was, by definition, not durably written, so losing it is
//!   the correct behavior.
//!
//! Mid-file corruption (bad CRC, bad kind, oversized lengths) is
//! always fatal: returned as `Err(Error::Corruption)`. The reader
//! does *not* skip past corrupted records, because once integrity is
//! in doubt, every byte after is also in doubt.

use crate::io::{FileRead, Io};
use crate::wal::record::{self, DecodeError, RecordKind};
use crate::wal::writer::{FILE_HEADER_LEN, FORMAT_VERSION, MAGIC};
use crate::{Error, Result};
use std::path::Path;

/// An owned record, suitable for handing to callers who hold it across
/// reader-state changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedRecord {
    pub kind: RecordKind,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

/// Streaming reader over a single WAL file.
pub struct WalReader {
    file: Box<dyn FileRead>,
    /// File length, cached at open time. The WAL is append-only and we
    /// hold the only handle, so this is safe to cache.
    file_len: u64,
    /// Next byte offset to read from.
    offset: u64,
    /// Once we hit truncation or clean EOF, further calls to `next`
    /// return `Ok(None)` without re-reading. Prevents reading garbage
    /// past a torn record on repeated polling.
    done: bool,
    /// Reusable buffer for reading record bytes from disk.
    scratch: Vec<u8>,
}

impl WalReader {
    /// Open `path` for replay. Validates the file header before
    /// returning; a bad magic or unknown version is a corruption error.
    pub fn open(io: &dyn Io, path: &Path) -> Result<Self> {
        let file = io.open_read(path)?;
        let file_len = file.len()?;

        if file_len < FILE_HEADER_LEN as u64 {
            return Err(Error::Corruption(format!(
                "WAL file shorter than header: {file_len} bytes, need {FILE_HEADER_LEN}"
            )));
        }

        // Read and validate the header.
        let mut header = [0u8; FILE_HEADER_LEN];
        file.read_at(0, &mut header)?;

        if &header[0..MAGIC.len()] != MAGIC {
            return Err(Error::Corruption(format!(
                "WAL magic mismatch: expected {:?}, got {:?}",
                MAGIC,
                &header[0..MAGIC.len()]
            )));
        }

        let version =
            u32::from_le_bytes(header[MAGIC.len()..FILE_HEADER_LEN].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(Error::Corruption(format!(
                "unsupported WAL format version {version}, this build supports {FORMAT_VERSION}"
            )));
        }

        Ok(Self {
            file,
            file_len,
            offset: FILE_HEADER_LEN as u64,
            done: false,
            scratch: Vec::with_capacity(1024),
        })
    }

    /// Yield the next record, or `Ok(None)` at end-of-log.
    pub fn next_record(&mut self) -> Result<Option<OwnedRecord>> {
        if self.done {
            return Ok(None);
        }
        if self.offset >= self.file_len {
            self.done = true;
            return Ok(None);
        }

        // Read the header so we know the full record length. If we
        // can't even get a full header, treat as torn tail.
        let remaining = self.file_len - self.offset;
        if remaining < record::HEADER_LEN as u64 {
            self.done = true;
            return Ok(None);
        }

        // Read header into scratch.
        self.scratch.resize(record::HEADER_LEN, 0);
        self.file.read_at(self.offset, &mut self.scratch)?;

        // Try decoding to learn the total record length. We will pass
        // the same bytes plus the payload to decode again; cheaper than
        // re-implementing the header parse here.
        //
        // First pass: we only need key_len and value_len to know total
        // size. decode itself returns Truncated when the buffer is
        // short, which is the cue to do a second read for the payload.
        match record::decode(&self.scratch) {
            Err(DecodeError::Truncated { needed, .. }) => {
                // `needed` is the full record length including payload.
                if (remaining as usize) < needed {
                    // Real torn tail: we don't have enough bytes on disk.
                    self.done = true;
                    return Ok(None);
                }
                // We have enough bytes; do a second read covering the
                // whole record.
                self.scratch.resize(needed, 0);
                self.file.read_at(self.offset, &mut self.scratch)?;
                let (rec, consumed) = record::decode(&self.scratch)?;
                debug_assert_eq!(consumed, needed);
                let owned = OwnedRecord {
                    kind: rec.kind,
                    key: rec.key.to_vec(),
                    value: rec.value.to_vec(),
                };
                self.offset += consumed as u64;
                Ok(Some(owned))
            }
            Err(e) => {
                // Real corruption.
                Err(Error::from(e))
            }
            Ok((rec, consumed)) => {
                // Edge case: a record with zero-length key+value would
                // fit entirely in HEADER_LEN bytes. Our encoder forbids
                // empty keys, so reaching here on legitimate data
                // shouldn't happen — but the code handles it correctly
                // anyway.
                let owned = OwnedRecord {
                    kind: rec.kind,
                    key: rec.key.to_vec(),
                    value: rec.value.to_vec(),
                };
                self.offset += consumed as u64;
                Ok(Some(owned))
            }
        }
    }

    /// Current read offset within the file. Useful for tests that want
    /// to assert how far into the file replay got.
    pub fn position(&self) -> u64 {
        self.offset
    }
}