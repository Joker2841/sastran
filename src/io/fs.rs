//! Production filesystem backend.
//!
//! Each method here is a direct passthrough to `std::fs` plus error
//! conversion. No buffering happens in this layer; if the caller wants
//! buffered writes, they wrap the returned handle.

use crate::Result;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use super::{FileAppend, FileRead, Io};

/// A handle wrapping a `std::fs::File` opened in append mode.
pub struct StdFileAppend {
    file: File,
}

impl FileAppend for StdFileAppend {
    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        self.file.write_all(bytes)?;
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        // `sync_all` flushes both data and metadata. `sync_data` would
        // skip metadata, which is slightly faster but unsafe when the
        // file has been extended (its length is metadata).
        self.file.sync_all()?;
        Ok(())
    }

    fn len(&self) -> Result<u64> {
        Ok(self.file.metadata()?.len())
    }
}

/// A handle wrapping a `std::fs::File` opened for random reads.
pub struct StdFileRead {
    file: File,
}

impl FileRead for StdFileRead {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        // `read_exact_at` is positioned reads without mutating a file
        // cursor; safe to call from multiple threads on the same handle.
        self.file.read_exact_at(buf, offset)?;
        Ok(())
    }

    fn len(&self) -> Result<u64> {
        Ok(self.file.metadata()?.len())
    }
}

/// Production [`Io`] implementation backed by the real OS filesystem.
#[derive(Debug, Default, Clone)]
pub struct StdFs;

impl StdFs {
    pub fn new() -> Self {
        Self
    }
}

impl Io for StdFs {
    fn open_append(&self, path: &Path) -> Result<Box<dyn FileAppend>> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true) // so we can query length without reopening
            .open(path)?;
        Ok(Box::new(StdFileAppend { file }))
    }

    fn open_read(&self, path: &Path) -> Result<Box<dyn FileRead>> {
        let file = OpenOptions::new().read(true).open(path)?;
        Ok(Box::new(StdFileRead { file }))
    }

    fn sync_dir(&self, dir: &Path) -> Result<()> {
        // Open the directory and fsync it. On Linux this is how you
        // make directory entries (newly created files, renames) durable.
        // On macOS, fsync on a directory is a no-op but doesn't error.
        let dir_file = File::open(dir)?;
        dir_file.sync_all()?;
        Ok(())
    }

    fn create_dir_all(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)?;
        Ok(())
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<()> {
        std::fs::rename(from, to)?;
        Ok(())
    }

    fn remove_file(&self, path: &Path) -> Result<()> {
        std::fs::remove_file(path)?;
        Ok(())
    }

    fn list_dir(&self, path: &Path) -> Result<Vec<PathBuf>> {
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(path)? {
            entries.push(entry?.path());
        }
        // Sort for determinism — tests will appreciate this.
        entries.sort();
        Ok(entries)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn round_trip_append_and_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("hello.bin");

        let fs = StdFs::new();
        let mut writer = fs.open_append(&path).unwrap();
        writer.append(b"hello, world").unwrap();
        writer.sync().unwrap();
        drop(writer);

        let reader = fs.open_read(&path).unwrap();
        let mut buf = vec![0u8; 12];
        reader.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, b"hello, world");
    }

    #[test]
    fn sync_dir_does_not_error_on_existing_dir() {
        let dir = tempdir().unwrap();
        let fs = StdFs::new();
        fs.sync_dir(dir.path()).unwrap();
    }
}