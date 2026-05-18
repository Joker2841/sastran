//! End-to-end WAL tests using only the public API.
//!
//! These tests exercise the full write → sync → reopen → replay loop
//! against the real filesystem (via `tempfile`). They are slower than
//! unit tests but catch integration bugs the per-module tests miss.

use sastran::io::fs::StdFs;
use sastran::wal::{OwnedRecord, RecordKind, WalReader, WalWriter};
use tempfile::tempdir;

#[test]
fn write_then_replay_round_trips_all_records() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let fs = StdFs::new();

    // Write a few records.
    let mut writer = WalWriter::open(&fs, &path, dir.path()).unwrap();
    writer.append(RecordKind::Put, b"k1", b"v1").unwrap();
    writer.append(RecordKind::Put, b"k2", b"v2").unwrap();
    writer.append(RecordKind::Delete, b"k1", b"").unwrap();
    writer.sync().unwrap();
    drop(writer);

    // Reopen for replay.
    let mut reader = WalReader::open(&fs, &path).unwrap();
    let records: Vec<OwnedRecord> = std::iter::from_fn(|| reader.next_record().transpose())
        .collect::<Result<_, _>>()
        .unwrap();

    assert_eq!(records.len(), 3);
    assert_eq!(records[0].kind, RecordKind::Put);
    assert_eq!(records[0].key, b"k1");
    assert_eq!(records[0].value, b"v1");
    assert_eq!(records[1].kind, RecordKind::Put);
    assert_eq!(records[1].key, b"k2");
    assert_eq!(records[1].value, b"v2");
    assert_eq!(records[2].kind, RecordKind::Delete);
    assert_eq!(records[2].key, b"k1");
    assert!(records[2].value.is_empty());
}

#[test]
fn empty_wal_replays_zero_records() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let fs = StdFs::new();

    // Open and immediately drop: file gets the header but no records.
    let writer = WalWriter::open(&fs, &path, dir.path()).unwrap();
    assert!(writer.is_empty().unwrap());
    drop(writer);

    let mut reader = WalReader::open(&fs, &path).unwrap();
    assert!(reader.next_record().unwrap().is_none());
    // Polling after EOF stays at EOF cheaply.
    assert!(reader.next_record().unwrap().is_none());
}

#[test]
fn reopening_writer_appends_to_existing_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let fs = StdFs::new();

    {
        let mut writer = WalWriter::open(&fs, &path, dir.path()).unwrap();
        writer.append(RecordKind::Put, b"a", b"1").unwrap();
        writer.sync().unwrap();
    }
    {
        let mut writer = WalWriter::open(&fs, &path, dir.path()).unwrap();
        writer.append(RecordKind::Put, b"b", b"2").unwrap();
        writer.sync().unwrap();
    }

    let mut reader = WalReader::open(&fs, &path).unwrap();
    let r1 = reader.next_record().unwrap().unwrap();
    let r2 = reader.next_record().unwrap().unwrap();
    assert_eq!(r1.key, b"a");
    assert_eq!(r2.key, b"b");
    assert!(reader.next_record().unwrap().is_none());
}

#[test]
fn torn_tail_is_silently_truncated_on_replay() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let fs = StdFs::new();

    // Write 3 records, sync.
    {
        let mut writer = WalWriter::open(&fs, &path, dir.path()).unwrap();
        writer.append(RecordKind::Put, b"k1", b"v1").unwrap();
        writer.append(RecordKind::Put, b"k2", b"v2").unwrap();
        writer.append(RecordKind::Put, b"k3", b"v3").unwrap();
        writer.sync().unwrap();
    }

    // Simulate a torn write by truncating the last 5 bytes of the file.
    let len_before = std::fs::metadata(&path).unwrap().len();
    let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.set_len(len_before - 5).unwrap();
    drop(f);

    // Replay: should yield k1 and k2 cleanly and stop at the torn k3.
    let mut reader = WalReader::open(&fs, &path).unwrap();
    let r1 = reader.next_record().unwrap().unwrap();
    let r2 = reader.next_record().unwrap().unwrap();
    assert_eq!(r1.key, b"k1");
    assert_eq!(r2.key, b"k2");
    assert!(
        reader.next_record().unwrap().is_none(),
        "torn tail should replay as clean EOF"
    );
}

#[test]
fn mid_file_corruption_is_fatal() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let fs = StdFs::new();

    // Write 3 records, sync.
    {
        let mut writer = WalWriter::open(&fs, &path, dir.path()).unwrap();
        writer.append(RecordKind::Put, b"k1", b"v1").unwrap();
        writer.append(RecordKind::Put, b"k2", b"v2").unwrap();
        writer.append(RecordKind::Put, b"k3", b"v3").unwrap();
        writer.sync().unwrap();
    }

    // Corrupt a byte inside record 2's *value* payload. Choosing the
    // value (not a length field) is deliberate: a corrupted length
    // field can be indistinguishable from a torn tail (the implied
    // record size exceeds the remaining file → looks truncated), which
    // is a fundamental limitation of length-prefixed formats. By
    // corrupting a payload byte, the lengths are unchanged, the record
    // still fits, and the CRC check fires as intended.
    //
    // File layout:
    //   bytes 0..12   file header
    //   bytes 12..29  record 1 (CRC + 1B kind + 4B key_len + 4B value_len + "k1" + "v1")
    //   bytes 29..46  record 2  (same shape, with "k2" + "v2")
    //     -> offset 44 is the first byte of record 2's value, "v"
    use std::io::{Seek, SeekFrom, Write};
    let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.seek(SeekFrom::Start(44)).unwrap();
    f.write_all(&[b'v' ^ 0x01]).unwrap(); // flip one bit -> 'w'
    drop(f);

    let mut reader = WalReader::open(&fs, &path).unwrap();
    let r1 = reader.next_record().unwrap().unwrap();
    assert_eq!(r1.key, b"k1");
    // Second read should surface a corruption error, not a silent stop.
    let err = reader.next_record().unwrap_err();
    assert!(
        matches!(err, sastran::Error::Corruption(_)),
        "expected Corruption, got {err:?}"
    );
}

#[test]
fn rejects_file_with_bad_magic() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("not-a-wal.log");
    std::fs::write(&path, b"NOTAWAL!\x01\x00\x00\x00rest of file").unwrap();

    let fs = StdFs::new();
    let result = WalReader::open(&fs, &path);
    assert!(
        matches!(result, Err(sastran::Error::Corruption(_))),
        "expected Corruption error, got {:?}",
        result.as_ref().err()
    );
}

#[test]
fn rejects_file_with_unknown_version() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("future-wal.log");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"SASTWAL1");
    bytes.extend_from_slice(&999u32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    let fs = StdFs::new();
    let result = WalReader::open(&fs, &path);
    assert!(
        matches!(result, Err(sastran::Error::Corruption(_))),
        "expected Corruption error, got {:?}",
        result.as_ref().err()
    );    
}