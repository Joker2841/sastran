//! SSTable writer: builds an immutable sorted file from a sorted stream.
//!
//! Lifecycle:
//! 1. [`SsTableWriter::create`] — start a new file. The caller passes
//!    an `expected_keys` count so the bloom filter (if enabled) is
//!    sized correctly up front.
//! 2. [`SsTableWriter::add`] — append entries in strictly ascending
//!    key order.
//! 3. [`SsTableWriter::finish`] — write the bloom block (if enabled),
//!    the index block, and the 40-byte footer; then fsync.

use crate::io::{FileAppend, Io};
use crate::memtable::Entry;
use crate::{Error, Result};
use std::path::Path;

use super::{
    BloomFilter, DATA_BLOCK_TARGET_SIZE, FOOTER_LEN, FOOTER_MAGIC, KIND_DELETE, KIND_PUT,
};

/// Builds an SSTable from a sorted stream of entries.
pub struct SsTableWriter {
    file: Box<dyn FileAppend>,
    current_block: Vec<u8>,
    current_block_first_key: Option<Vec<u8>>,
    index_entries: Vec<IndexEntry>,
    next_block_offset: u64,
    last_key: Option<Vec<u8>>,
    /// Bloom filter (if enabled). Sized at construction; populated
    /// from each key during `add`.
    bloom: Option<BloomFilter>,
}

struct IndexEntry {
    first_key: Vec<u8>,
    offset: u64,
    length: u32,
}

impl SsTableWriter {
    /// Create a new SSTable at `path`.
    ///
    /// `expected_keys` is used only to size the bloom filter; passing
    /// a value that turns out to be wrong won't break correctness, but
    /// significantly under-sized estimates inflate the false-positive
    /// rate. Pass the memtable's `len()` (for flush) or the sum of
    /// input SSTable sizes (for compaction).
    ///
    /// `bloom_bits_per_key = 0` disables the bloom filter; the footer
    /// will record offset = length = 0 for the bloom block.
    pub fn create(
        io: &dyn Io,
        path: &Path,
        expected_keys: usize,
        bloom_bits_per_key: u32,
    ) -> Result<Self> {
        let file = io.open_append(path)?;
        if !file.is_empty()? {
            return Err(Error::InvalidArgument(format!(
                "SSTable file already exists and is non-empty: {}",
                path.display()
            )));
        }
        let bloom = if bloom_bits_per_key > 0 {
            Some(BloomFilter::new(expected_keys, bloom_bits_per_key))
        } else {
            None
        };
        Ok(Self {
            file,
            current_block: Vec::with_capacity(DATA_BLOCK_TARGET_SIZE + 256),
            current_block_first_key: None,
            index_entries: Vec::new(),
            next_block_offset: 0,
            last_key: None,
            bloom,
        })
    }

    /// Append one entry. Keys must be strictly greater than every
    /// previously-added key.
    pub fn add(&mut self, key: &[u8], entry: &Entry) -> Result<()> {
        if key.is_empty() {
            return Err(Error::InvalidArgument("empty key".into()));
        }
        if let Some(prev) = &self.last_key
            && key <= prev.as_slice()
        {
            return Err(Error::InvalidArgument(format!(
                "keys must be strictly increasing; got {:?} after {:?}",
                key, prev
            )));
        }

        let entry_size = encoded_entry_size(key, entry);
        if !self.current_block.is_empty()
            && self.current_block.len() + entry_size > DATA_BLOCK_TARGET_SIZE
        {
            self.flush_current_block()?;
        }

        if self.current_block_first_key.is_none() {
            self.current_block_first_key = Some(key.to_vec());
        }

        encode_entry(key, entry, &mut self.current_block);
        self.last_key = Some(key.to_vec());

        if let Some(bf) = self.bloom.as_mut() {
            bf.insert(key);
        }

        Ok(())
    }

    /// Finish the SSTable: trailing data block, bloom block, index
    /// block, footer, fsync.
    pub fn finish(mut self) -> Result<()> {
        if !self.current_block.is_empty() {
            self.flush_current_block()?;
        }

        // Write the bloom block, if enabled.
        let (bloom_offset, bloom_length) = if let Some(bf) = self.bloom.as_ref() {
            let offset = self.next_block_offset;
            let bytes = bf.encode();
            let length = bytes.len() as u64;
            self.file.append(&bytes)?;
            self.next_block_offset += length;
            (offset, length)
        } else {
            (0u64, 0u64)
        };

        // Write the index block.
        let index_offset = self.next_block_offset;
        let mut index_block = Vec::new();
        for entry in &self.index_entries {
            encode_index_entry(entry, &mut index_block);
        }
        let index_crc = crc32fast::hash(&index_block);
        index_block.extend_from_slice(&index_crc.to_le_bytes());
        let index_length = index_block.len() as u64;
        self.file.append(&index_block)?;

        // Write the footer.
        let mut footer = Vec::with_capacity(FOOTER_LEN);
        footer.extend_from_slice(&index_offset.to_le_bytes());
        footer.extend_from_slice(&index_length.to_le_bytes());
        footer.extend_from_slice(&bloom_offset.to_le_bytes());
        footer.extend_from_slice(&bloom_length.to_le_bytes());
        footer.extend_from_slice(FOOTER_MAGIC);
        debug_assert_eq!(footer.len(), FOOTER_LEN);
        self.file.append(&footer)?;

        self.file.sync()?;
        Ok(())
    }

    fn flush_current_block(&mut self) -> Result<()> {
        debug_assert!(!self.current_block.is_empty());
        let crc = crc32fast::hash(&self.current_block);
        self.current_block.extend_from_slice(&crc.to_le_bytes());

        let block_offset = self.next_block_offset;
        let block_length = self.current_block.len() as u32;
        self.file.append(&self.current_block)?;
        self.next_block_offset += block_length as u64;

        let first_key = self
            .current_block_first_key
            .take()
            .expect("flushing a block must have a first key");
        self.index_entries.push(IndexEntry {
            first_key,
            offset: block_offset,
            length: block_length,
        });

        self.current_block.clear();
        Ok(())
    }
}

fn encoded_entry_size(key: &[u8], entry: &Entry) -> usize {
    let value_len = match entry {
        Entry::Put(v) => v.len(),
        Entry::Tombstone => 0,
    };
    4 + 4 + 1 + key.len() + value_len
}

fn encode_entry(key: &[u8], entry: &Entry, out: &mut Vec<u8>) {
    let (kind_byte, value): (u8, &[u8]) = match entry {
        Entry::Put(v) => (KIND_PUT, v.as_slice()),
        Entry::Tombstone => (KIND_DELETE, &[]),
    };
    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.push(kind_byte);
    out.extend_from_slice(key);
    out.extend_from_slice(value);
}

fn encode_index_entry(entry: &IndexEntry, out: &mut Vec<u8>) {
    out.extend_from_slice(&(entry.first_key.len() as u32).to_le_bytes());
    out.extend_from_slice(entry.first_key.as_slice());
    out.extend_from_slice(&entry.offset.to_le_bytes());
    out.extend_from_slice(&entry.length.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Io;
    use crate::io::fs::StdFs;
    use tempfile::tempdir;

    #[test]
    fn rejects_keys_out_of_order() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        let fs = StdFs::new();
        let mut w = SsTableWriter::create(&fs, &path, 0, 0).unwrap();
        w.add(b"b", &Entry::Put(b"1".to_vec())).unwrap();
        let err = w
            .add(b"a", &Entry::Put(b"2".to_vec()))
            .expect_err("should reject out-of-order key");
        assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
    }

    #[test]
    fn rejects_duplicate_keys() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        let fs = StdFs::new();
        let mut w = SsTableWriter::create(&fs, &path, 0, 0).unwrap();
        w.add(b"k", &Entry::Put(b"1".to_vec())).unwrap();
        let err = w
            .add(b"k", &Entry::Put(b"2".to_vec()))
            .expect_err("should reject duplicate key");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn rejects_empty_key() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        let fs = StdFs::new();
        let mut w = SsTableWriter::create(&fs, &path, 0, 0).unwrap();
        let err = w
            .add(b"", &Entry::Put(b"1".to_vec()))
            .expect_err("should reject empty key");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn empty_sstable_finishes_cleanly() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        let fs = StdFs::new();
        let w = SsTableWriter::create(&fs, &path, 0, 0).unwrap();
        w.finish().unwrap();
        let reader = fs.open_read(&path).unwrap();
        let len = reader.len().unwrap();
        assert!(
            len >= FOOTER_LEN as u64,
            "empty SSTable should still contain the footer"
        );
        let mut magic = [0u8; 8];
        reader.read_at(len - 8, &mut magic).unwrap();
        assert_eq!(&magic, FOOTER_MAGIC);
    }

    #[test]
    fn populated_sstable_has_magic_at_end() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        let fs = StdFs::new();
        let mut w = SsTableWriter::create(&fs, &path, 100, 10).unwrap();
        for i in 0..100u32 {
            let key = format!("k_{i:04}");
            let value = format!("v_{i}");
            w.add(key.as_bytes(), &Entry::Put(value.into_bytes())).unwrap();
        }
        w.finish().unwrap();

        let reader = fs.open_read(&path).unwrap();
        let len = reader.len().unwrap();
        let mut magic = [0u8; 8];
        reader.read_at(len - 8, &mut magic).unwrap();
        assert_eq!(&magic, FOOTER_MAGIC);
    }

    #[test]
    fn many_entries_produce_multiple_blocks() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        let fs = StdFs::new();
        let mut w = SsTableWriter::create(&fs, &path, 100, 10).unwrap();
        let big_value = vec![0xCDu8; 200];
        for i in 0..100u32 {
            let key = format!("k_{i:04}");
            w.add(key.as_bytes(), &Entry::Put(big_value.clone())).unwrap();
        }
        w.finish().unwrap();

        let reader = fs.open_read(&path).unwrap();
        let len = reader.len().unwrap();
        assert!(len > 20_000, "expected multi-block file, got {len} bytes");
    }
}