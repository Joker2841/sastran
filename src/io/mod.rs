//! I/O abstraction layer.
//!
//! All filesystem operations in the engine go through the [`Io`] trait
//! rather than calling `std::fs` directly. This indirection exists so
//! that tests can substitute a fault-injecting implementation that
//! simulates crashes, partial writes, and fsync failures
//! deterministically.
//!
//! Production code uses [`fs::StdFs`], which is a thin wrapper around
//! `std::fs`. A `FaultyFs` implementation for tests will be added in a
//! later commit.
//!
//! ### Design notes
//!
//! The trait is intentionally narrow: it exposes only the operations an
//! LSM-tree storage engine actually performs. Adding new methods should
//! be a deliberate decision, since every method must be implementable
//! by every backend.
//!
//! Handles returned by the trait (`FileRead`, `FileAppend`) are boxed
//! trait objects so the `Io` trait itself remains object-safe. This
//! costs one indirection per file-handle call, which is negligible
//! compared to syscall overhead.

use crate::Result;
use std::path::{Path, PathBuf};

pub mod fs;

/// A handle to a file opened for sequential append.
pub trait FileAppend: Send {
    /// Append `bytes` to the end of the file. The write is buffered;
    /// nothing is guaranteed durable until [`sync`](Self::sync) returns.
    fn append(&mut self, bytes: &[u8]) -> Result<()>;

    /// Force all previously appended bytes to durable storage. Returns
    /// only after the data has reached stable storage (modulo whatever
    /// the underlying device claims about its write cache).
    fn sync(&mut self) -> Result<()>;

    /// Current logical length of the file, in bytes.
    fn len(&self) -> Result<u64>;

    /// Returns `true` if the file has zero bytes. Default implementation
    /// calls [`len`](Self::len); override if a cheaper check is available.
    fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

/// A handle to a file opened for random reads.
pub trait FileRead: Send + Sync {
    /// Read exactly `buf.len()` bytes starting at byte `offset`. Returns
    /// `Err` if the file is shorter than `offset + buf.len()`.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()>;

    /// Total length of the file, in bytes.
    fn len(&self) -> Result<u64>;

    /// Returns `true` if the file has zero bytes. Default implementation
    /// calls [`len`](Self::len); override if a cheaper check is available.
    fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

/// Filesystem abstraction. All engine I/O routes through implementations
/// of this trait.
pub trait Io: Send + Sync {
    /// Open a file for append. Creates the file if it does not exist.
    fn open_append(&self, path: &Path) -> Result<Box<dyn FileAppend>>;

    /// Open an existing file for random reads.
    fn open_read(&self, path: &Path) -> Result<Box<dyn FileRead>>;

    /// Force a directory's metadata to durable storage. Must be called
    /// after creating or renaming files within `dir` if those changes
    /// need to survive a crash.
    fn sync_dir(&self, dir: &Path) -> Result<()>;

    /// Create a directory and all parents. Idempotent.
    fn create_dir_all(&self, path: &Path) -> Result<()>;

    /// Atomically rename a file. On POSIX systems this maps to
    /// `rename(2)`, which is atomic *within* a filesystem.
    fn rename(&self, from: &Path, to: &Path) -> Result<()>;

    /// Delete a file. Errors if the file does not exist.
    fn remove_file(&self, path: &Path) -> Result<()>;

    /// List entries in a directory. Returns absolute paths.
    fn list_dir(&self, path: &Path) -> Result<Vec<PathBuf>>;
}