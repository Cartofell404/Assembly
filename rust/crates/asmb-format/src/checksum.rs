//! SHA-256 over blob regions.
//!
//! An archive is gigabytes, so nothing here ever materialises a whole blob.
//! Both the one-shot and streaming paths read in bounded chunks, which is
//! what keeps memory-mapping an archive worth doing in the first place.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::Result;

/// How much is read per pass when hashing a blob.
pub const HASH_CHUNK: usize = 4 * 1024 * 1024;

/// SHA-256 of an in-memory slice, hex.
///
/// Used at build time for headers and small payloads.
pub fn hash_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex(&hasher.finalize())
}

/// SHA-256 of `data[offset..offset + size]`, read in bounded chunks.
///
/// The slice is never copied. Hashing `data[offset..offset + size]` directly
/// would duplicate the whole model in RAM and undo the point of memory-mapping
/// it.
///
/// # Errors
///
/// Returns [`Error::Io`](crate::Error::Io) if the region runs past the end of
/// `data`.
pub fn hash_region(data: &[u8], offset: u64, size: u64) -> Result<String> {
    let start = offset as usize;
    let end = start
        .checked_add(size as usize)
        .filter(|end| *end <= data.len())
        .ok_or_else(|| {
            crate::Error::truncated(format!(
                "region {offset:#x}..{} runs past the end of a {} byte buffer",
                offset.saturating_add(size),
                data.len()
            ))
        })?;

    let mut hasher = Sha256::new();
    for chunk in data[start..end].chunks(HASH_CHUNK) {
        hasher.update(chunk);
    }
    Ok(hex(&hasher.finalize()))
}

/// SHA-256 of a region of a file, read in bounded chunks.
///
/// # Errors
///
/// Returns [`Error::Io`](crate::Error::Io) if the file cannot be opened or
/// read, and [`Error::Truncated`](crate::Error::Truncated) if the region runs
/// past the end of it.
pub fn hash_file_region(path: &Path, offset: u64, size: u64) -> Result<String> {
    let mut file = File::open(path)?;
    let total = file.metadata()?.len();
    if offset.saturating_add(size) > total {
        return Err(crate::Error::truncated(format!(
            "region {offset:#x}..{} runs past the end of {} ({} bytes)",
            offset.saturating_add(size),
            path.display(),
            total
        )));
    }

    file.seek(SeekFrom::Start(offset))?;
    let mut reader = BufReader::with_capacity(HASH_CHUNK, file);

    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_CHUNK];
    let mut remaining = size;
    while remaining > 0 {
        let want = remaining.min(HASH_CHUNK as u64) as usize;
        let read = reader.read(&mut buf[..want])?;
        if read == 0 {
            return Err(crate::Error::truncated(format!(
                "{} ended {remaining} bytes early while hashing",
                path.display()
            )));
        }
        hasher.update(&buf[..read]);
        remaining -= read as u64;
    }
    Ok(hex(&hasher.finalize()))
}

/// SHA-256 of a whole file, read in bounded chunks.
///
/// This is the build-time counterpart to [`hash_file_region`]: a builder wants
/// the digest of an entire source before copying it, so the bytes can be named
/// in the header ahead of the copy that verifies them.
///
/// # Errors
///
/// Returns [`Error::Io`](crate::Error::Io) if the file cannot be opened or read.
pub fn hash_file(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(HASH_CHUNK, file);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_CHUNK];

    loop {
        let read = reader.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

/// Finish a streaming hasher and return lowercase hex.
///
/// Exposed for callers that hash as they copy, where they already hold a
/// [`Sha256`] rather than the whole blob.
pub fn digest_hex(hasher: Sha256) -> String {
    hex(&hasher.finalize())
}

/// Lowercase hex, which is what the header stores and what `--verify`
/// compares against.
fn hex(digest: &[u8]) -> String {
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    #[test]
    fn hashes_empty_input() {
        assert_eq!(hash_bytes(b""), EMPTY);
    }

    #[test]
    fn hashes_known_vector() {
        // "abc" -> ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        assert_eq!(
            hash_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hashes_a_region_of_a_buffer() {
        let data = b"xxxxhello worldyyyy";
        let got = hash_region(data, 4, 11).unwrap();
        assert_eq!(
            got,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn region_past_end_is_an_error() {
        let err = hash_region(b"short", 3, 100).unwrap_err();
        assert!(err.to_string().contains("past the end"), "got {err}");
    }

    #[test]
    fn chunked_hash_matches_one_shot() {
        // Push past HASH_CHUNK so the chunk loop runs more than once.
        let data = vec![7u8; HASH_CHUNK * 2 + 12345];
        assert_eq!(
            hash_region(&data, 0, data.len() as u64).unwrap(),
            hash_bytes(&data)
        );
    }

    #[test]
    fn hashes_a_region_of_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(b"0123456789abcdef").unwrap();
        f.flush().unwrap();
        drop(f);

        // The region is the second half of the payload.
        assert_eq!(
            hash_file_region(&path, 8, 8).unwrap(),
            hash_bytes(b"89abcdef")
        );
    }

    #[test]
    fn file_region_past_end_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.bin");
        std::fs::write(&path, b"tiny").unwrap();
        let err = hash_file_region(&path, 0, 100).unwrap_err();
        assert!(err.to_string().contains("past the end"), "got {err}");
    }
}
