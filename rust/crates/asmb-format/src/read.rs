//! Reading `.asmb` archives.
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │ MAGIC       4 B   b"ASMB"                   │
//! │ VERSION     4 B   uint32-LE                  │
//! │ HEADER_LEN  8 B   uint64-LE (JSON payload)   │
//! │ JSON        variable                          │
//! │ PADDING     zero bytes → next page boundary   │
//! ├──────────────────────────────────────────────┤
//! │ GGUF #0     (page-aligned)                   │
//! │ PADDING     → next page boundary              │
//! │ GGUF #1     (page-aligned)                   │
//! │ …                                            │
//! └──────────────────────────────────────────────┘
//! ```
//!
//! The JSON payload carries **offsets** and **sizes**, so a reader can jump
//! straight to one GGUF blob and memory-map only that range instead of
//! walking the file.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::checksum;
use crate::error::{Error, Result};
use crate::header::{AsmbHeader, EmbeddingEntry, ModelEntry, FORMAT_VERSION, MAGIC, PREAMBLE_SIZE};

/// Return the next page-aligned offset at or after `offset`.
///
/// A `page` of zero disables alignment, which keeps a corrupt header from
/// causing a division by zero before it is rejected.
pub fn align_to_page(offset: u64, page: u64) -> u64 {
    if page == 0 {
        return offset;
    }
    let remainder = offset % page;
    if remainder == 0 {
        offset
    } else {
        offset + (page - remainder)
    }
}

/// Read `version` and the JSON length from the 16-byte preamble.
pub fn read_preamble(data: &[u8]) -> Result<(u32, u64)> {
    if data.len() < PREAMBLE_SIZE {
        return Err(Error::truncated("preamble is shorter than 16 bytes"));
    }
    if data[..4] != MAGIC {
        return Err(Error::BadMagic {
            expected: MAGIC,
            found: [data[0], data[1], data[2], data[3]],
        });
    }
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version != FORMAT_VERSION {
        return Err(Error::UnsupportedVersion {
            found: version,
            supported: FORMAT_VERSION,
        });
    }
    let json_length = u64::from_le_bytes([
        data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
    ]);
    Ok((version, json_length))
}

/// Read the header JSON from a byte slice.
pub fn read_header(data: &[u8]) -> Result<AsmbHeader> {
    let (_, json_length) = read_preamble(data)?;
    let json_start = PREAMBLE_SIZE;
    let json_end = json_start + json_length as usize;
    if json_end > data.len() {
        return Err(Error::truncated("header JSON is truncated"));
    }
    let json_bytes = &data[json_start..json_end];
    AsmbHeader::from_json_bytes(json_bytes)
}

/// Validate that the header's layout matches the file size.
pub fn validate_layout(header: &AsmbHeader, total: u64) -> Result<()> {
    let mut seen = HashSet::new();

    if header.embedding.is_present() {
        let e = &header.embedding;
        if seen.contains(&e.offset) {
            return Err(Error::layout("embedding offset collides with another blob"));
        }
        seen.insert(e.offset);
        if e.offset + e.size > total {
            return Err(Error::layout("embedding extends past end of file"));
        }
    }

    for m in &header.models {
        if seen.contains(&m.offset) {
            return Err(Error::layout(format!(
                "model '{}' offset collides with another blob",
                m.name
            )));
        }
        seen.insert(m.offset);
        if m.offset + m.size > total {
            return Err(Error::layout(format!(
                "model '{}' extends past end of file",
                m.name
            )));
        }
    }
    Ok(())
}

/// An open `.asmb` archive.
///
/// Holds the header and a buffered reader positioned after the header.
/// Blobs are read on demand via [`extract_to`] or [`verify`].
#[derive(Debug)]
pub struct Archive {
    header: AsmbHeader,
    reader: BufReader<File>,
    total_len: u64,
}

impl Archive {
    /// Open an archive from disk and validate its header.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened, plus anything
    /// [`read_header`] raises.
    pub fn open(path: impl AsRef<Path>) -> Result<Archive> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let total_len = file.metadata()?.len();

        let mut reader = BufReader::with_capacity(checksum::HASH_CHUNK, file);

        let mut prefix = vec![0u8; PREAMBLE_SIZE];
        reader.read_exact(&mut prefix)?;
        let json_length = read_preamble(&prefix)?.1;

        let mut json_blob = vec![0u8; json_length as usize];
        reader
            .read_exact(&mut json_blob)
            .map_err(|e| Error::truncated(format!("header is truncated: {e}")))?;

        let header = AsmbHeader::from_json_bytes(&json_blob)?;
        validate_layout(&header, total_len)?;

        Ok(Archive {
            header,
            reader,
            total_len,
        })
    }

    /// The parsed header.
    pub fn header(&self) -> &AsmbHeader {
        &self.header
    }

    /// Total archive length in bytes.
    pub fn len(&self) -> u64 {
        self.total_len
    }

    /// Whether the archive is empty. Always false for a valid archive.
    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    /// The router model entry, if the archive has one.
    pub fn embedding(&self) -> Option<&EmbeddingEntry> {
        if self.header.embedding.is_present() {
            Some(&self.header.embedding)
        } else {
            None
        }
    }

    /// The execution model entries, in header order.
    pub fn models(&self) -> &[ModelEntry] {
        &self.header.models
    }

    /// The blob byte range for one named model.
    ///
    /// The router counts: it is a blob in this archive like any other, and a
    /// caller staging "the router" should not have to know it is stored in a
    /// different header field.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Layout`] when no blob has that name.
    pub fn blob_range(&self, name: &str) -> Result<(u64, u64)> {
        let entry = self
            .find_entry(name)
            .ok_or_else(|| Error::layout(format!("no model named '{name}' in this archive")))?;
        Ok((entry.0, entry.1))
    }

    /// `(offset, size, sha256, label)` for any blob in the archive.
    fn find_entry(&self, name: &str) -> Option<(u64, u64, String, String)> {
        if self.header.embedding.is_present() && self.header.embedding.name == name {
            let e = &self.header.embedding;
            return Some((e.offset, e.size, e.sha256.clone(), e.name.clone()));
        }
        self.header
            .models
            .iter()
            .find(|m| m.name == name)
            .map(|m| (m.offset, m.size, m.sha256.clone(), m.name.clone()))
    }

    /// Copy one named blob's bytes out to a fresh file.
    ///
    /// This is how a blob gets staged for llama.cpp, which wants a real path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on read or write failure, and
    /// [`Error::ChecksumMismatch`] when the header recorded a digest and the
    /// copied bytes disagree with it.
    pub fn extract_to(&mut self, name: &str, dest: impl AsRef<Path>) -> Result<()> {
        let dest = dest.as_ref();
        let (offset, size, sha256, label) = self
            .find_entry(name)
            .ok_or_else(|| Error::layout(format!("no model named '{name}' in this archive")))?;

        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();

        self.reader
            .seek(SeekFrom::Start(offset))
            .map_err(|e| Error::layout(format!("cannot seek to '{label}': {e}")))?;

        let mut out = File::create(dest)?;
        let mut buf = vec![0u8; checksum::HASH_CHUNK];
        let mut remaining = size;
        use std::io::Write as _;
        while remaining > 0 {
            let want = remaining.min(checksum::HASH_CHUNK as u64) as usize;
            let read = self.reader.read(&mut buf[..want])?;
            if read == 0 {
                return Err(Error::truncated(format!(
                    "'{label}' ended {remaining} bytes early"
                )));
            }
            hasher.update(&buf[..read]);
            out.write_all(&buf[..read])?;
            remaining -= read as u64;
        }
        out.flush()?;

        if !sha256.is_empty() {
            let actual = crate::checksum::digest_hex(hasher);
            if actual != sha256 {
                return Err(Error::ChecksumMismatch {
                    name: label,
                    expected: sha256,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// Verify all blobs that have a recorded SHA-256.
    ///
    /// For each model (including the router) that has a recorded SHA-256,
    /// recomputes the hash and compares it. Blobs with empty SHA-256 are
    /// included as "ok" (no checksum to verify).
    ///
    /// Returns a [`VerificationReport`] listing every checked blob and
    /// whether it matched.
    pub fn verify(&mut self) -> Result<VerificationReport> {
        use sha2::{Digest, Sha256};

        let mut blobs = Vec::new();

        // Check router if present - always include it
        if self.header.embedding.is_present() {
            let e = &self.header.embedding;
            if !e.sha256.is_empty() {
                let mut hasher = Sha256::new();
                self.reader.seek(SeekFrom::Start(e.offset)).map_err(|err| {
                    Error::layout(format!("cannot seek to '{}': {}", e.name, err))
                })?;

                let mut buf = vec![0u8; checksum::HASH_CHUNK];
                let mut remaining = e.size;
                while remaining > 0 {
                    let want = remaining.min(checksum::HASH_CHUNK as u64) as usize;
                    let read = self.reader.read(&mut buf[..want])?;
                    if read == 0 {
                        return Err(Error::truncated(format!(
                            "'{}' ended {remaining} bytes early",
                            e.name
                        )));
                    }
                    hasher.update(&buf[..read]);
                    remaining -= read as u64;
                }
                let actual = crate::checksum::digest_hex(hasher);
                let ok = actual == e.sha256;
                blobs.push(BlobVerification {
                    name: e.name.clone(),
                    ok,
                    expected: e.sha256.clone(),
                    actual,
                });
            } else {
                // Router present but no checksum - still include in report as ok
                blobs.push(BlobVerification {
                    name: e.name.clone(),
                    ok: true,
                    expected: String::new(),
                    actual: String::new(),
                });
            }
        }

        // Check all execution models
        for m in &self.header.models {
            if !m.sha256.is_empty() {
                let mut hasher = Sha256::new();
                self.reader
                    .seek(SeekFrom::Start(m.offset))
                    .map_err(|e| Error::layout(format!("cannot seek to '{}': {e}", m.name)))?;

                let mut buf = vec![0u8; checksum::HASH_CHUNK];
                let mut remaining = m.size;
                while remaining > 0 {
                    let want = remaining.min(checksum::HASH_CHUNK as u64) as usize;
                    let read = self.reader.read(&mut buf[..want])?;
                    if read == 0 {
                        return Err(Error::truncated(format!(
                            "'{}' ended {remaining} bytes early",
                            m.name
                        )));
                    }
                    hasher.update(&buf[..read]);
                    remaining -= read as u64;
                }
                let actual = crate::checksum::digest_hex(hasher);
                let ok = actual == m.sha256;
                blobs.push(BlobVerification {
                    name: m.name.clone(),
                    ok,
                    expected: m.sha256.clone(),
                    actual,
                });
            }
        }

        Ok(VerificationReport { blobs })
    }
}

/// Verification result for a single blob.
#[derive(Debug, Clone)]
pub struct BlobVerification {
    /// The name of the blob (model or router).
    pub name: String,
    /// Whether the blob's hash matched the recorded digest.
    pub ok: bool,
    /// The expected SHA-256 from the header.
    pub expected: String,
    /// The actual SHA-256 computed from the blob.
    pub actual: String,
}

/// Result of verifying an entire archive.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Verification results for each blob that had a recorded digest.
    pub blobs: Vec<BlobVerification>,
}

impl VerificationReport {
    /// Returns true if all blobs matched.
    pub fn is_valid(&self) -> bool {
        self.blobs.iter().all(|b| b.ok)
    }

    /// Number of blobs that matched.
    pub fn ok_count(&self) -> usize {
        self.blobs.iter().filter(|b| b.ok).count()
    }

    /// Total bytes verified.
    pub fn ok_bytes(&self) -> u64 {
        self.blobs.iter().filter(|b| b.ok).count() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{AsmbHeader, PAGE_SIZE};
    use std::io::Write as _;

    fn header() -> AsmbHeader {
        AsmbHeader {
            assembly_name: "T".into(),
            version: FORMAT_VERSION,
            page_size: PAGE_SIZE,
            embedding: EmbeddingEntry {
                name: "router".into(),
                offset: PAGE_SIZE as u64,
                size: 4,
                confidence_threshold: 0.7,
                fallback_model: None,
                sha256: String::new(),
            },
            models: vec![ModelEntry::with_vectors(
                "general",
                &[vec![1.0, 0.0]],
                (PAGE_SIZE * 2) as u64,
                4,
                vec!["hi".into()],
                "",
            )
            .unwrap()],
            persona: None,
        }
    }

    /// Build a minimal but valid archive image in memory.
    fn build_image(header: &AsmbHeader, total: usize) -> Vec<u8> {
        let json = header.to_json_bytes().unwrap();
        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(&json);
        buf.resize(total, 0);
        buf
    }

    #[test]
    fn aligns_to_page_boundaries() {
        assert_eq!(align_to_page(0, 4096), 0);
        assert_eq!(align_to_page(1, 4096), 4096);
        assert_eq!(align_to_page(4096, 4096), 4096);
        assert_eq!(align_to_page(4097, 4096), 8192);
    }

    #[test]
    fn reads_preamble() {
        let mut data = MAGIC.to_vec();
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&42u64.to_le_bytes());
        let (version, len) = read_preamble(&data).unwrap();
        assert_eq!((version, len), (1, 42));
    }

    #[test]
    fn rejects_short_preamble() {
        let err = read_preamble(b"ASM").unwrap_err();
        assert!(matches!(err, Error::Truncated(_)), "got {err:?}");
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = b"NOPE".to_vec();
        data.extend_from_slice(&[0u8; 12]);
        let err = read_preamble(&data).unwrap_err();
        assert!(matches!(err, Error::BadMagic { .. }), "got {err:?}");
    }

    #[test]
    fn rejects_future_version() {
        let mut data = MAGIC.to_vec();
        data.extend_from_slice(&99u32.to_le_bytes());
        data.extend_from_slice(&[0u8; 8]);
        let err = read_preamble(&data).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedVersion { found: 99, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn round_trips_a_header_through_bytes() {
        let h = header();
        let image = build_image(&h, PAGE_SIZE * 4);
        assert_eq!(read_header(&image).unwrap(), h);
    }

    #[test]
    fn rejects_header_longer_than_file() {
        let mut data = MAGIC.to_vec();
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&9999u64.to_le_bytes());
        let err = read_header(&data).unwrap_err();
        assert!(matches!(err, Error::Truncated(_)), "got {err:?}");
    }

    #[test]
    fn opens_and_reads_a_real_file() {
        let h = header();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.asmb");
        let mut image = build_image(&h, PAGE_SIZE * 4);
        // Write GGUF at the model offset so extract_to can read it
        image[(PAGE_SIZE * 2)..(PAGE_SIZE * 2 + 4)].copy_from_slice(b"GGUF");
        std::fs::write(&path, image).unwrap();

        let mut archive = Archive::open(&path).unwrap();
        assert_eq!(archive.len(), (PAGE_SIZE * 4) as u64);
        assert!(archive.embedding().is_some());

        let (offset, size) = archive.blob_range("general").unwrap();
        assert_eq!((offset, size), ((PAGE_SIZE * 2) as u64, 4));

        let dest = dir.path().join("general.gguf");
        archive.extract_to("general", &dest).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"GGUF");
    }

    #[test]
    fn extract_detects_a_flipped_byte() {
        let h = header();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.asmb");

        let mut image = build_image(&h, PAGE_SIZE * 4);
        image[(PAGE_SIZE * 2)..(PAGE_SIZE * 2 + 4)].copy_from_slice(b"GGUF");

        // Record the digest of *different* bytes, then write the wrong ones.
        let wrong = checksum::hash_bytes(b"NOPE");
        let mut h2 = h.clone();
        h2.models[0].sha256 = wrong.clone();
        let json = h2.to_json_bytes().unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(&json);
        buf.resize(PAGE_SIZE * 4, 0);
        buf[(PAGE_SIZE * 2)..(PAGE_SIZE * 2 + 4)].copy_from_slice(b"GGUF");

        let mut f = File::create(&path).unwrap();
        f.write_all(&buf).unwrap();
        f.flush().unwrap();
        drop(f);

        let mut archive = Archive::open(&path).unwrap();
        let err = archive
            .extract_to("general", dir.path().join("out.gguf"))
            .unwrap_err();
        assert!(matches!(err, Error::ChecksumMismatch { .. }), "got {err:?}");
        assert_eq!(
            err.to_string(),
            format!(
                "checksum mismatch for 'general': header says {wrong}, bytes hash to {}",
                checksum::hash_bytes(b"GGUF")
            )
        );
    }

    #[test]
    fn unknown_model_is_an_error() {
        let h = header();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.asmb");
        std::fs::write(&path, build_image(&h, PAGE_SIZE * 4)).unwrap();
        let archive = Archive::open(&path).unwrap();
        let err = archive.blob_range("nope").unwrap_err();
        assert!(err.to_string().contains("no model named"), "got {err}");
    }
}
