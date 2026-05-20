//! SSTable reader: point lookups against an immutable on-disk file.
//!
//! On open, the reader validates the footer and loads the index block
//! into memory. Each call to [`SsTableReader::get`] then performs at
//! most one positioned read (to fetch the candidate data block) plus
//! a linear scan of that block.

use crate::io::{FileRead, Io};
use crate::memtable::Entry;
use crate::sstable::BloomFilter;
use crate::{Error, Result};
use std::path::Path;
use std::sync::Arc;

use super::{FOOTER_LEN, FOOTER_MAGIC, KIND_DELETE, KIND_PUT, KIND_VECTOR};

/// Result of a point lookup. Mirrors [`crate::memtable::Lookup`] but
/// returns owned bytes, since the data was just read from disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsTableLookup {
    /// Key is present in the SSTable with this value.
    Found(Vec<u8>),
    /// Key has a tombstone in the SSTable. The caller should treat
    /// this as deleted and stop falling through to older sources.
    Deleted,
    /// Key is not in this SSTable. Caller should continue searching.
    Missing,
}

/// Parsed index entry: where each data block lives in the file, and
/// the smallest key it contains.
#[derive(Debug, Clone)]
struct IndexEntry {
    first_key: Vec<u8>,
    offset: u64,
    length: u32,
}

/// Reader for a single SSTable file.
pub struct SsTableReader {
    file: Arc<dyn FileRead>,
    /// In-memory copy of the index, sorted by `first_key`.
    /// Held in an `Arc` so iterators can share it cheaply.
    index: Arc<Vec<IndexEntry>>,
    /// Bloom filter, if the SSTable file includes one. `None` means
    /// either the file was written with bloom disabled, or no filter
    /// block is recorded in the footer.
    bloom: Option<BloomFilter>,
}

impl SsTableReader {
    /// Open `path` for reading. Validates the footer and loads the
    /// index block into memory before returning.
    pub fn open(io: &dyn Io, path: &Path) -> Result<Self> {
        let file: Arc<dyn FileRead> = Arc::from(io.open_read(path)?);
        let file_len = file.len()?;

        if file_len < FOOTER_LEN as u64 {
            return Err(Error::Corruption(format!(
                "SSTable file shorter than footer: {file_len} bytes"
            )));
        }

        // Read the 40-byte footer.
        let mut footer = [0u8; FOOTER_LEN];
        file.read_at(file_len - FOOTER_LEN as u64, &mut footer)?;

        let index_offset = u64::from_le_bytes(footer[0..8].try_into().unwrap());
        let index_length = u64::from_le_bytes(footer[8..16].try_into().unwrap());
        let bloom_offset = u64::from_le_bytes(footer[16..24].try_into().unwrap());
        let bloom_length = u64::from_le_bytes(footer[24..32].try_into().unwrap());
        let magic = &footer[32..40];

        if magic != FOOTER_MAGIC {
            return Err(Error::Corruption(format!(
                "SSTable footer magic mismatch: got {magic:?}, expected {FOOTER_MAGIC:?}"
            )));
        }

        // Sanity-check the index bounds against the file size.
        if index_offset + index_length + FOOTER_LEN as u64 > file_len {
            return Err(Error::Corruption(format!(
                "SSTable index bounds out of range: offset={index_offset}, length={index_length}, file_len={file_len}"
            )));
        }
        // If a bloom block is recorded, sanity-check its bounds too.
        if bloom_length > 0
            && bloom_offset + bloom_length + index_length + FOOTER_LEN as u64 > file_len
        {
            return Err(Error::Corruption(format!(
                "SSTable bloom bounds out of range: offset={bloom_offset}, length={bloom_length}, file_len={file_len}"
            )));
        }

        // Read and validate the index block.
        let mut index_bytes = vec![0u8; index_length as usize];
        file.read_at(index_offset, &mut index_bytes)?;
        let index = parse_index_block(&index_bytes)?;

        // Read the bloom block, if present.
        let bloom = if bloom_length > 0 {
            let mut bloom_bytes = vec![0u8; bloom_length as usize];
            file.read_at(bloom_offset, &mut bloom_bytes)?;
            Some(BloomFilter::decode(&bloom_bytes)?)
        } else {
            None
        };

        Ok(Self {
            file,
            index: Arc::new(index),
            bloom,
        })
    }


    /// Look up `key`. At most one data-block read is performed.
    ///
    /// If a bloom filter is loaded and reports the key as definitely
    /// absent, returns `Missing` without touching disk at all.
    pub fn get(&self, key: &[u8]) -> Result<SsTableLookup> {
        // Bloom-filter pre-check: definitely-absent keys skip disk
        // entirely. False positives fall through to the normal path.
        if let Some(bf) = self.bloom.as_ref()
            && !bf.contains(key)
        {
            return Ok(SsTableLookup::Missing);
        }

        // Binary-search the index for the candidate block.
        let candidate = match candidate_block(&self.index, key) {
            Some(i) => &self.index[i],
            None => return Ok(SsTableLookup::Missing),
        };

        // Read the candidate data block.
        let mut block_bytes = vec![0u8; candidate.length as usize];
        self.file.read_at(candidate.offset, &mut block_bytes)?;

        // Verify CRC.
        if block_bytes.len() < 4 {
            return Err(Error::Corruption(
                "SSTable data block shorter than CRC trailer".into(),
            ));
        }
        let split = block_bytes.len() - 4;
        let stored_crc = u32::from_le_bytes(block_bytes[split..].try_into().unwrap());
        let computed_crc = crc32fast::hash(&block_bytes[..split]);
        if stored_crc != computed_crc {
            return Err(Error::Corruption(format!(
                "SSTable data block CRC mismatch: stored {stored_crc:#010x}, computed {computed_crc:#010x}"
            )));
        }

        // Linear-scan the block for the target key.
        scan_block_for_key(&block_bytes[..split], key)
    }

    /// Number of data blocks in the SSTable. Diagnostic / test only.
    #[doc(hidden)]
    pub fn block_count(&self) -> usize {
        self.index.len()
    }

    /// Iterate over every entry in this SSTable in ascending key order.
    ///
    /// The returned iterator shares the file handle and index with the
    /// reader (both held in `Arc`), so creating an iterator is cheap
    /// and does not block other operations on the reader.
    pub fn iter(&self) -> SsTableIterator {
        SsTableIterator {
            file: Arc::clone(&self.file),
            index: Arc::clone(&self.index),
            block_idx: 0,
            current_block: None,
            pos: 0,
        }
    }
}

/// Forward iterator over an SSTable's entries.
pub struct SsTableIterator {
    file: Arc<dyn FileRead>,
    index: Arc<Vec<IndexEntry>>,
    /// Index of the data block we're currently reading from.
    block_idx: usize,
    /// CRC-stripped bytes of the current block, or `None` if we need
    /// to load the next block on the next `next` call.
    current_block: Option<Vec<u8>>,
    /// Position within `current_block`.
    pos: usize,
}

impl SsTableIterator {
    /// Load the data block at `self.block_idx` and reset `pos`.
    /// Returns an error if the block fails CRC validation.
    fn load_current_block(&mut self) -> Result<()> {
        let entry = &self.index[self.block_idx];
        let mut block_bytes = vec![0u8; entry.length as usize];
        self.file.read_at(entry.offset, &mut block_bytes)?;

        if block_bytes.len() < 4 {
            return Err(Error::Corruption(
                "SSTable data block shorter than CRC trailer".into(),
            ));
        }
        let split = block_bytes.len() - 4;
        let stored_crc = u32::from_le_bytes(block_bytes[split..].try_into().unwrap());
        let computed_crc = crc32fast::hash(&block_bytes[..split]);
        if stored_crc != computed_crc {
            return Err(Error::Corruption(format!(
                "SSTable data block CRC mismatch in iterator: stored {stored_crc:#010x}, computed {computed_crc:#010x}"
            )));
        }

        block_bytes.truncate(split);
        self.current_block = Some(block_bytes);
        self.pos = 0;
        Ok(())
    }

    /// Decode the entry at `pos` in `current_block`, advancing `pos`.
    /// Returns the parsed `(key, entry)`, or an error on corruption.
    fn decode_entry_at_pos(&mut self) -> Result<(Vec<u8>, Entry)> {
        let block = self
            .current_block
            .as_ref()
            .expect("decode_entry_at_pos called with no current_block");
        let p = self.pos;
        if p + 9 > block.len() {
            return Err(Error::Corruption(
                "truncated entry header in data block".into(),
            ));
        }
        let key_len = u32::from_le_bytes(block[p..p + 4].try_into().unwrap()) as usize;
        let value_len = u32::from_le_bytes(block[p + 4..p + 8].try_into().unwrap()) as usize;
        let kind = block[p + 8];
        let body_start = p + 9;
        let body_end = body_start + key_len + value_len;
        if body_end > block.len() {
            return Err(Error::Corruption(
                "truncated entry payload in data block".into(),
            ));
        }
        let key = block[body_start..body_start + key_len].to_vec();
        let value_bytes = &block[body_start + key_len..body_end];
        let entry = match kind {
            KIND_PUT => Entry::Put(value_bytes.to_vec()),
            KIND_VECTOR => Entry::Vector(value_bytes.to_vec()),
            KIND_DELETE => Entry::Tombstone,
            other => {
                return Err(Error::Corruption(format!(
                    "unknown SSTable entry kind {other:#04x}"
                )));
            }
        };
        self.pos = body_end;
        Ok((key, entry))
    }
}

impl Iterator for SsTableIterator {
    type Item = Result<(Vec<u8>, Entry)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we don't have a block loaded, try to load the next one.
            if self.current_block.is_none() {
                if self.block_idx >= self.index.len() {
                    return None;
                }
                if let Err(e) = self.load_current_block() {
                    // Surface the error and stop iteration. We poison
                    // the iterator by leaving current_block as None
                    // and bumping block_idx past the end so a future
                    // call also returns None rather than re-erroring.
                    self.block_idx = self.index.len();
                    return Some(Err(e));
                }
            }

            // We have a block. Are we at the end of it?
            let at_end = self
                .current_block
                .as_ref()
                .is_some_and(|b| self.pos >= b.len());
            if at_end {
                self.current_block = None;
                self.block_idx += 1;
                continue;
            }

            return Some(self.decode_entry_at_pos());
        }
    }
}

/// Parse an index block. The trailing 4 bytes are a CRC32 over the rest.
fn parse_index_block(bytes: &[u8]) -> Result<Vec<IndexEntry>> {
    if bytes.len() < 4 {
        return Err(Error::Corruption(
            "index block shorter than CRC trailer".into(),
        ));
    }
    let split = bytes.len() - 4;
    let stored_crc = u32::from_le_bytes(bytes[split..].try_into().unwrap());
    let computed_crc = crc32fast::hash(&bytes[..split]);
    if stored_crc != computed_crc {
        return Err(Error::Corruption(format!(
            "SSTable index CRC mismatch: stored {stored_crc:#010x}, computed {computed_crc:#010x}"
        )));
    }

    let mut entries = Vec::new();
    let mut p = 0usize;
    let payload = &bytes[..split];

    while p < payload.len() {
        // Each entry: key_len(u32) || key || offset(u64) || length(u32)
        if p + 4 > payload.len() {
            return Err(Error::Corruption(
                "truncated index entry: key_len header".into(),
            ));
        }
        let key_len =
            u32::from_le_bytes(payload[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        if p + key_len + 8 + 4 > payload.len() {
            return Err(Error::Corruption(
                "truncated index entry: key or offsets".into(),
            ));
        }
        let key = payload[p..p + key_len].to_vec();
        p += key_len;
        let offset = u64::from_le_bytes(payload[p..p + 8].try_into().unwrap());
        p += 8;
        let length = u32::from_le_bytes(payload[p..p + 4].try_into().unwrap());
        p += 4;
        entries.push(IndexEntry {
            first_key: key,
            offset,
            length,
        });
    }

    Ok(entries)
}

/// Locate the data block that could contain `key`.
///
/// Returns the index of the largest entry whose `first_key <= key`,
/// or `None` if `key` is smaller than every block's `first_key`.
fn candidate_block(index: &[IndexEntry], key: &[u8]) -> Option<usize> {
    // partition_point returns the first index where the predicate is false.
    // We want: largest i such that index[i].first_key <= key.
    // Equivalently: partition_point(|e| e.first_key <= key) - 1.
    let p = index.partition_point(|e| e.first_key.as_slice() <= key);
    if p == 0 { None } else { Some(p - 1) }
}

/// Scan a data block (CRC stripped) for `key`.
fn scan_block_for_key(block: &[u8], target: &[u8]) -> Result<SsTableLookup> {
    let mut p = 0usize;
    while p < block.len() {
        // Entry: key_len(u32) || value_len(u32) || kind(u8) || key || value
        if p + 9 > block.len() {
            return Err(Error::Corruption(
                "truncated entry header in data block".into(),
            ));
        }
        let key_len = u32::from_le_bytes(block[p..p + 4].try_into().unwrap()) as usize;
        let value_len =
            u32::from_le_bytes(block[p + 4..p + 8].try_into().unwrap()) as usize;
        let kind = block[p + 8];
        p += 9;

        if p + key_len + value_len > block.len() {
            return Err(Error::Corruption(
                "truncated entry payload in data block".into(),
            ));
        }
        let key_bytes = &block[p..p + key_len];

        // Since entries are sorted, we can stop early once we pass the target.
        if key_bytes > target {
            return Ok(SsTableLookup::Missing);
        }

        if key_bytes == target {
            let value_bytes = &block[p + key_len..p + key_len + value_len];
            return match kind {
                KIND_PUT | KIND_VECTOR => Ok(SsTableLookup::Found(value_bytes.to_vec())),
                KIND_DELETE => Ok(SsTableLookup::Deleted),
                other => Err(Error::Corruption(format!(
                    "unknown SSTable entry kind {other:#04x}"
                ))),
            };
        }

        p += key_len + value_len;
    }
    Ok(SsTableLookup::Missing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::fs::StdFs;
    use crate::sstable::SsTableWriter;
    use tempfile::tempdir;

    fn build_sst(path: &Path, entries: &[(Vec<u8>, Entry)]) {
        let fs = StdFs::new();
        // Tests construct SSTables with bloom filters enabled (10
        // bits/key) so the reader paths involving bloom are exercised.
        // expected_keys is the entries length, which is exact.
        let mut w = SsTableWriter::create(&fs, path, entries.len(), 10).unwrap();
        for (k, e) in entries {
            w.add(k, e).unwrap();
        }
        w.finish().unwrap();
    }

    #[test]
    fn round_trip_small_put() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        build_sst(
            &path,
            &[
                (b"a".to_vec(), Entry::Put(b"1".to_vec())),
                (b"b".to_vec(), Entry::Put(b"2".to_vec())),
                (b"c".to_vec(), Entry::Put(b"3".to_vec())),
            ],
        );

        let fs = StdFs::new();
        let r = SsTableReader::open(&fs, &path).unwrap();
        assert_eq!(r.get(b"a").unwrap(), SsTableLookup::Found(b"1".to_vec()));
        assert_eq!(r.get(b"b").unwrap(), SsTableLookup::Found(b"2".to_vec()));
        assert_eq!(r.get(b"c").unwrap(), SsTableLookup::Found(b"3".to_vec()));
        assert_eq!(r.get(b"missing").unwrap(), SsTableLookup::Missing);
        assert_eq!(r.get(b"").unwrap(), SsTableLookup::Missing);
    }

    #[test]
    fn tombstones_return_deleted_not_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        build_sst(
            &path,
            &[
                (b"alive".to_vec(), Entry::Put(b"yes".to_vec())),
                (b"dead".to_vec(), Entry::Tombstone),
            ],
        );

        let fs = StdFs::new();
        let r = SsTableReader::open(&fs, &path).unwrap();
        assert_eq!(
            r.get(b"alive").unwrap(),
            SsTableLookup::Found(b"yes".to_vec())
        );
        assert_eq!(r.get(b"dead").unwrap(), SsTableLookup::Deleted);
    }

    #[test]
    fn key_before_first_block_is_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        build_sst(
            &path,
            &[
                (b"m".to_vec(), Entry::Put(b"v".to_vec())),
                (b"n".to_vec(), Entry::Put(b"v".to_vec())),
            ],
        );

        let fs = StdFs::new();
        let r = SsTableReader::open(&fs, &path).unwrap();
        assert_eq!(r.get(b"a").unwrap(), SsTableLookup::Missing);
    }

    #[test]
    fn key_after_last_block_is_missing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        build_sst(
            &path,
            &[
                (b"m".to_vec(), Entry::Put(b"v".to_vec())),
                (b"n".to_vec(), Entry::Put(b"v".to_vec())),
            ],
        );

        let fs = StdFs::new();
        let r = SsTableReader::open(&fs, &path).unwrap();
        assert_eq!(r.get(b"z").unwrap(), SsTableLookup::Missing);
    }

    #[test]
    fn many_keys_across_multiple_blocks() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        let big = vec![0xABu8; 256];
        let entries: Vec<_> = (0..500u32)
            .map(|i| (format!("k_{i:04}").into_bytes(), Entry::Put(big.clone())))
            .collect();
        build_sst(&path, &entries);

        let fs = StdFs::new();
        let r = SsTableReader::open(&fs, &path).unwrap();
        assert!(r.block_count() > 1, "expected multiple data blocks");

        for i in 0..500u32 {
            let key = format!("k_{i:04}");
            let got = r.get(key.as_bytes()).unwrap();
            assert_eq!(got, SsTableLookup::Found(big.clone()), "mismatch on {key}");
        }

        assert_eq!(
            r.get(b"k_0500").unwrap(),
            SsTableLookup::Missing,
            "k_0500 should be just past the last key"
        );
        assert_eq!(r.get(b"zzz").unwrap(), SsTableLookup::Missing);
    }

    #[test]
    fn rejects_truncated_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        build_sst(
            &path,
            &[(b"k".to_vec(), Entry::Put(b"v".to_vec()))],
        );
        let len = std::fs::metadata(&path).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(len - 1).unwrap();
        drop(f);

        let fs = StdFs::new();
        let result = SsTableReader::open(&fs, &path);
        assert!(
            matches!(result, Err(Error::Corruption(_))),
            "expected Corruption error, got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn rejects_bad_magic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        build_sst(
            &path,
            &[(b"k".to_vec(), Entry::Put(b"v".to_vec()))],
        );
        use std::io::{Seek, SeekFrom, Write};
        let len = std::fs::metadata(&path).unwrap().len();
        let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.seek(SeekFrom::Start(len - 4)).unwrap();
        f.write_all(b"XXXX").unwrap();
        drop(f);

        let fs = StdFs::new();
        let result = SsTableReader::open(&fs, &path);
        assert!(
            matches!(result, Err(Error::Corruption(_))),
            "expected Corruption error, got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn rejects_corrupted_data_block() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        build_sst(
            &path,
            &[
                (b"a".to_vec(), Entry::Put(b"v1".to_vec())),
                (b"b".to_vec(), Entry::Put(b"v2".to_vec())),
            ],
        );

        use std::io::{Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.seek(SeekFrom::Start(5)).unwrap();
        f.write_all(&[0xFF]).unwrap();
        drop(f);

        let fs = StdFs::new();
        let r = SsTableReader::open(&fs, &path).unwrap(); // header intact
        let err = r.get(b"a");
        assert!(
            matches!(err, Err(Error::Corruption(_))),
            "expected corruption, got {err:?}"
        );
    }

    #[test]
    fn candidate_block_picks_correct_index() {
        let index = vec![
            IndexEntry { first_key: b"c".to_vec(), offset: 0, length: 10 },
            IndexEntry { first_key: b"m".to_vec(), offset: 10, length: 10 },
            IndexEntry { first_key: b"t".to_vec(), offset: 20, length: 10 },
        ];
        assert_eq!(candidate_block(&index, b"a"), None);
        assert_eq!(candidate_block(&index, b"c"), Some(0));
        assert_eq!(candidate_block(&index, b"f"), Some(0));
        assert_eq!(candidate_block(&index, b"m"), Some(1));
        assert_eq!(candidate_block(&index, b"zzz"), Some(2));
    }

    #[test]
    fn iter_empty_sstable_yields_nothing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        build_sst(&path, &[]);

        let fs = StdFs::new();
        let r = SsTableReader::open(&fs, &path).unwrap();
        let items: Vec<_> = r.iter().collect();
        assert!(items.is_empty());
    }

    #[test]
    fn iter_yields_all_entries_in_order() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        let entries: Vec<(Vec<u8>, Entry)> = vec![
            (b"a".to_vec(), Entry::Put(b"1".to_vec())),
            (b"b".to_vec(), Entry::Put(b"2".to_vec())),
            (b"c".to_vec(), Entry::Tombstone),
            (b"d".to_vec(), Entry::Put(b"4".to_vec())),
        ];
        build_sst(&path, &entries);

        let fs = StdFs::new();
        let r = SsTableReader::open(&fs, &path).unwrap();
        let got: Vec<(Vec<u8>, Entry)> = r
            .iter()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(got, entries);
    }

    #[test]
    fn iter_crosses_multiple_blocks() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        let big = vec![0xCDu8; 256];
        let entries: Vec<(Vec<u8>, Entry)> = (0..300u32)
            .map(|i| (format!("k_{i:04}").into_bytes(), Entry::Put(big.clone())))
            .collect();
        build_sst(&path, &entries);

        let fs = StdFs::new();
        let r = SsTableReader::open(&fs, &path).unwrap();
        assert!(r.block_count() > 1, "expected multiple blocks");

        let got: Vec<(Vec<u8>, Entry)> = r
            .iter()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(got.len(), entries.len());
        for (got, expected) in got.iter().zip(entries.iter()) {
            assert_eq!(got.0, expected.0, "key mismatch");
            assert_eq!(got.1, expected.1, "value mismatch on key {:?}", got.0);
        }
    }

    #[test]
    fn iter_stops_after_corruption() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        build_sst(
            &path,
            &[
                (b"a".to_vec(), Entry::Put(b"v1".to_vec())),
                (b"b".to_vec(), Entry::Put(b"v2".to_vec())),
            ],
        );

        use std::io::{Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.seek(SeekFrom::Start(5)).unwrap();
        f.write_all(&[0xFF]).unwrap();
        drop(f);

        let fs = StdFs::new();
        let r = SsTableReader::open(&fs, &path).unwrap();
        let mut it = r.iter();
        let first = it.next().expect("iterator should yield error");
        assert!(matches!(first, Err(Error::Corruption(_))));
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    #[test]
    fn iter_yields_vector_entries() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        let entries: Vec<(Vec<u8>, Entry)> = vec![
            (b"a".to_vec(), Entry::Put(b"plain".to_vec())),
            (b"b".to_vec(), Entry::Vector(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            (b"c".to_vec(), Entry::Tombstone),
        ];
        build_sst(&path, &entries);

        let fs = StdFs::new();
        let r = SsTableReader::open(&fs, &path).unwrap();
        let got: Vec<(Vec<u8>, Entry)> = r
            .iter()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(got, entries);
    }

    #[test]
    fn get_returns_bytes_for_vector_entry() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sst.bin");
        let v = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        build_sst(&path, &[(b"k".to_vec(), Entry::Vector(v.clone()))]);

        let fs = StdFs::new();
        let r = SsTableReader::open(&fs, &path).unwrap();
        assert_eq!(r.get(b"k").unwrap(), SsTableLookup::Found(v));
    }
}