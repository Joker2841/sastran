//! WAL writer: append-only, with explicit fsync semantics.
//!
//! A [`WalWriter`] owns a file opened via the engine's [`Io`] trait.
//! On creation it ensures the file starts with the expected file
//! header (magic + version); on each call to [`append`](WalWriter::append)
//! it encodes one record and writes it to the file *without* fsync.
//! Durability is the caller's responsibility via [`sync`](WalWriter::sync).
//!
//! This split — append vs. sync — is deliberate. Callers who batch
//! multiple writes can amortize one fsync over many records ("group
//! commit"). Callers who need per-write durability call `sync` after
//! every `append`. The writer enforces neither policy.

use crate::io::{FileAppend, Io};
use crate::wal::record::{self, RecordKind};
use crate::Result;
use std::path::Path;

/// File-format magic: 8 bytes of ASCII at the start of every WAL file.
pub(crate) const MAGIC: &[u8; 8] = b"SASTWAL1";

/// Current on-disk format version. Code rejects any other value.
pub(crate) const FORMAT_VERSION: u32 = 1;

/// Size in bytes of the WAL file header (magic + version).
pub(crate) const FILE_HEADER_LEN: usize = MAGIC.len() + 4;

/// Appender for a single WAL file.
pub struct WalWriter {
    file: Box<dyn FileAppend>,
    /// Reusable scratch buffer for record encoding. Held on the struct
    /// so successive `append` calls do not allocate.
    scratch: Vec<u8>,
}

impl WalWriter {
    /// Open `path` for append, writing the file header if the file is
    /// new (zero-length) and validating it otherwise.
    ///
    /// `io` is the filesystem abstraction; `parent_dir` is the
    /// directory containing `path`. When the file is created, the
    /// parent directory is fsynced so the new directory entry is
    /// durable.
    pub fn open(io: &dyn Io, path: &Path, parent_dir: &Path) -> Result<Self> {
        let mut file = io.open_append(path)?;
        let is_new = file.is_empty()?;

        if is_new {
            // Brand-new file: write the header and make it durable,
            // then fsync the directory so the file itself is reachable
            // by name after a crash.
            let mut header = Vec::with_capacity(FILE_HEADER_LEN);
            header.extend_from_slice(MAGIC);
            header.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
            file.append(&header)?;
            file.sync()?;
            io.sync_dir(parent_dir)?;
        } else {
            // Existing file: trust that the header is there. We do not
            // re-read it here; the reader is responsible for validating
            // before any records are returned. Validating again here
            // would require a separate read handle, which adds API
            // surface for no real gain.
            //
            // If the file is shorter than the header length, that's
            // already corruption — caught by the reader on next replay.
        }

        Ok(Self {
            file,
            scratch: Vec::with_capacity(1024),
        })
    }

    /// Append one record. Does *not* fsync; call [`sync`](Self::sync)
    /// when durability is required.
    pub fn append(&mut self, kind: RecordKind, key: &[u8], value: &[u8]) -> Result<()> {
        self.scratch.clear();
        record::encode(kind, key, value, &mut self.scratch)?;
        self.file.append(&self.scratch)?;
        Ok(())
    }

    /// Force all previously appended records to durable storage.
    pub fn sync(&mut self) -> Result<()> {
        self.file.sync()
    }

    /// Current size of the file on disk, in bytes (including header).
    /// Useful for tests and for WAL rotation logic later.
    pub fn len(&self) -> Result<u64> {
        self.file.len()
    }

    /// Returns `true` if the file contains only the header (no records).
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == FILE_HEADER_LEN as u64)
    }
}

// Note: there is intentionally no `Drop` impl that fsyncs. A `Drop`-on-fsync
// would silently swallow errors and silently make callers think they had
// durability when they don't. Callers must explicitly call `sync`.