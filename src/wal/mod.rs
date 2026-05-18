//! Write-Ahead Log.
//!
//! The WAL is an append-only sequence of records that captures every
//! mutation to the engine before it is acknowledged to the caller. On
//! crash recovery, the WAL is replayed to reconstruct the in-memory
//! state that existed at the moment of the last successful write.
//!
//! Submodules:
//! - [`record`]: on-disk record format (encode/decode/checksum).
//! - [`writer`]: append + fsync logic.
//! - [`reader`]: replay with torn-write detection.

pub mod reader;
pub mod record;
pub mod writer;

pub use reader::{OwnedRecord, WalReader};
pub use record::RecordKind;
pub use writer::WalWriter;