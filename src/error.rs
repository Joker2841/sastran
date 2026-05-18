//! Crate-wide error type.
//!
//! Every fallible operation in `sastran` returns `Result<T>`, which is
//! shorthand for `std::result::Result<T, Error>`. The variants are
//! intentionally coarse-grained: callers should match on the variant to
//! decide retry vs. abort vs. surface-to-user, not on the inner message.

use std::io;
use thiserror::Error;

/// The crate's error type.
#[derive(Debug, Error)]
pub enum Error {
    /// An underlying I/O operation failed (disk full, permission denied,
    /// file not found, etc.). The wrapped `io::Error` preserves the OS
    /// error code for callers that need to distinguish e.g. ENOSPC.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Data read from disk did not pass an integrity check. This is
    /// surfaced separately from `Io` because the read itself succeeded;
    /// the *contents* are wrong. Typically caused by a torn write at
    /// the tail of a WAL, hardware corruption, or a bug in our writer.
    #[error("data corruption: {0}")]
    Corruption(String),

    /// The caller passed an argument we refuse to accept (e.g. an empty key, a value larger than the configured maximum).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// The engine has been closed; no further operations are allowed.
    #[error("engine is closed")]
    Closed,
}

/// Crate-wide `Result` alias. Saves typing and makes signatures clearer.
pub type Result<T> = std::result::Result<T, Error>;