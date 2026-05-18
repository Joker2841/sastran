//! On-disk sorted string tables (SSTables).
//!
//! An SSTable is an immutable file containing key-value pairs in
//! sorted key order, written when a memtable is flushed. Each SSTable
//! file has a block-based layout:
//!
//! ```text
//! ┌────────────┬────────────┬─────┬────────────┬────────────┬────────┐
//! │ Data block │ Data block │ ... │ Data block │ Index block│ Footer │
//! └────────────┴────────────┴─────┴────────────┴────────────┴────────┘
//! ```
//!
//! - **Data blocks** are ~4 KiB each (target size). Entries are
//!   length-prefixed and sorted; the block ends with a CRC32 trailer.
//! - **Index block** lists `(first_key, offset, length)` for every
//!   data block, also with a CRC32 trailer.
//! - **Footer** is a fixed 24 bytes: index offset (u64 LE), index
//!   length (u64 LE), magic `b"SASTSST1"`.
//!
//! The reader keeps the index block in memory for the SSTable's
//! lifetime, so a point lookup costs at most one positioned read of
//! the relevant data block.
//!
//! ## What this format does *not* include yet
//!
//! - Bloom filters
//! - Varint encoding
//! - Block compression
//! - Restart points / prefix compression

pub mod bloom;
pub mod merge;
pub mod reader;
pub mod writer;

pub use bloom::BloomFilter;
pub use merge::{MergeSource, MergingIterator, SortedEntryIter};
pub use reader::{SsTableIterator, SsTableLookup, SsTableReader};
pub use writer::SsTableWriter;

/// Magic bytes at the end of every SSTable file footer.
///
/// Version 2 adds bloom-filter offset/length to the footer. There is
/// no backward compatibility with version 1 files; the magic check
/// rejects them with a clear corruption error.
pub(crate) const FOOTER_MAGIC: &[u8; 8] = b"SASTSST2";

/// Fixed footer size:
///   index_offset(8) + index_length(8) + bloom_offset(8) + bloom_length(8) + magic(8).
pub(crate) const FOOTER_LEN: usize = 8 * 4 + 8;

/// Target size for data blocks. Entries larger than this end up in
/// their own block; the writer never splits a single entry across
/// blocks.
pub(crate) const DATA_BLOCK_TARGET_SIZE: usize = 4 * 1024;

/// Marker byte inside an SSTable entry indicating a real value (`Put`).
pub(crate) const KIND_PUT: u8 = 0x01;

/// Marker byte inside an SSTable entry indicating a tombstone.
pub(crate) const KIND_DELETE: u8 = 0x02;