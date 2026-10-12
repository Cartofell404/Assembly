//! Writing `.asmb` archives.
//!
//! Layout is fixed by the reader: a 16-byte preamble, the compact header JSON,
//! zero padding to the page boundary, then each GGUF blob in turn, each
//! followed by padding to the next page boundary. The router (embedding) blob
//! goes first, then execution models in config order.
//!
//! # The build is atomic
//!
//! A half-written archive still has a valid preamble, so an interrupted build
//! would look loadable and fail much later, inside the inference engine, with an
//! error that points at the wrong thing. [`BuildPlan::build`] therefore writes
//! to a `.partial-<pid>` sibling and renames it into place only after the bytes
//! have been written, flushed, size-checked and read back. The cost is one
//! rename; the alternative is a corrupt archive that reports itself as fine.
//!
//! Nothing trailing the final blob is written, so the file ends exactly where
//! the last blob does and its size is known before the first byte is copied.
//!
//! # The header length is a fixed point
//!
//! The header holds the blob offsets, and its own length decides where the
//! first blob starts. [`BuildPlan::build`] resolves that by laying out,
//! serialising and re-laying out until the length stops moving, then confirming
//! the offsets it settled on are the ones it will write.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::checksum::{digest_hex, hash_file, HASH_CHUNK};
use crate::error::{Error, Result};
use crate::header::{AsmbHeader, EmbeddingEntry, ModelEntry, FORMAT_VERSION, MAGIC, PREAMBLE_SIZE};
use crate::read::{align_to_page, Archive};

/// How many layout passes before giving up on a stable header length.
///
/// Two is the real answer; the ceiling only exists so a pathological config
/// cannot spin.
const MAX_LAYOUT_PASSES: usize = 8;

/// One source blob to place in the archive.
#[derive(Debug, Clone)]
pub struct BlobSource {
    /// Name as written in the build config. Must be unique.
    pub name: String,
    /// GGUF file on disk.
    pub path: PathBuf,
    /// Expected SHA-256, hex. Empty skips the pre-copy check.
    pub expect_sha256: String,
}

impl BlobSource {
    /// A source blob with no pre-computed digest.
    pub fn new(name: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        BlobSource {
            name: name.into(),
            path: path.into(),
            expect_sha256: String::new(),
        }
    }

    /// A source blob with a digest to verify against while copying.
    pub fn with_sha256(mut self, sha256: impl Into<String>) -> Self {
        self.expect_sha256 = sha256.into();
        self
    }

    /// Size of the file on disk.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened or stat'd.
    pub fn size(&self) -> Result<u64> {
        Ok(File::open(&self.path)?.metadata()?.len())
    }
}

/// Routing settings for the router, which land in the embedding entry.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterSettings {
    /// Cosine similarity below which the router stops trusting its pick.
    pub confidence_threshold: f32,
    /// Model to use when nothing clears the threshold. `None` means use the
    /// best match anyway.
    pub fallback_model: Option<String>,
}

impl Default for RouterSettings {
    fn default() -> Self {
        RouterSettings {
            confidence_threshold: 0.75,
            fallback_model: None,
        }
    }
}

/// One execution model: its blob plus everything the header records about it.
#[derive(Debug, Clone)]
pub struct ModelSource {
    /// The GGUF to copy.
    pub blob: BlobSource,
    /// Human-readable phrases this model is meant to answer.
    pub anchors: Vec<String>,
    /// Anchor vectors. Empty means the model never wins a routing decision.
    pub anchor_vectors: Vec<Vec<f32>>,
}

impl ModelSource {
    /// A model with blob and phrases but no computed vectors yet.
    pub fn new(blob: BlobSource, anchors: Vec<String>) -> Self {
        ModelSource {
            blob,
            anchors,
            anchor_vectors: Vec::new(),
        }
    }

    /// Attach pre-computed anchor vectors.
    pub fn with_vectors(mut self, vectors: Vec<Vec<f32>>) -> Self {
        self.anchor_vectors = vectors;
        self
    }
}

/// Everything needed to write one archive.
///
/// Separating the plan from the write is what makes the layout solvable: the
/// plan is fully known before a byte is written, so the header can be built,
/// measured and rebuilt until its length is consistent with its own contents.
#[derive(Debug, Clone)]
pub struct BuildPlan {
    /// Archive name, recorded in the header.
    pub assembly_name: String,
    /// Router model, or `None` for an archive with no router.
    pub embedding: Option<BlobSource>,
    /// Routing settings for the router entry.
    pub router: RouterSettings,
    /// Execution models, in load order. At least one is required.
    pub models: Vec<ModelSource>,
    /// Persona prepended to every conversation.
    pub persona: Option<String>,
}

impl BuildPlan {
    /// A plan with a router, no models yet.
    ///
    /// Models come from the config, so the caller appends them; [`Self::models`]
    /// is public because there is no builder here yet and pretending otherwise
    /// would only move the mutation elsewhere.
    pub fn new(assembly_name: impl Into<String>, embedding: Option<BlobSource>) -> Self {
        BuildPlan {
            assembly_name: assembly_name.into(),
            embedding,
            router: RouterSettings::default(),
            models: Vec::new(),
            persona: None,
        }
    }

    /// Set the router's threshold and fallback.
    pub fn with_router(mut self, router: RouterSettings) -> Self {
        self.router = router;
        self
    }

    /// Set the persona.
    pub fn with_persona(mut self, persona: Option<String>) -> Self {
        self.persona = persona;
        self
    }

    /// Set the embedding (router) model.
    pub fn with_embedding(mut self, embedding: Option<BlobSource>) -> Self {
        self.embedding = embedding;
        self
    }

    /// Add one execution model.
    pub fn with_model(mut self, model: ModelSource) -> Self {
        self.models.push(model);
        self
    }

    /// Every blob in write order: router first, then models.
    fn blobs(&self) -> Vec<&BlobSource> {
        self.embedding
            .iter()
            .chain(self.models.iter().map(|m| &m.blob))
            .collect()
    }

    /// Where each blob must land, without writing anything.
    ///
    /// This is the same computation [`Self::build`] performs, exposed because a
    /// caller with eight gigabytes of models and one of disk deserves to be told
    /// no before it starts.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Layout`] on duplicate or empty names, and
    /// [`Error::Anchors`] if a model's vectors are ragged.
    pub fn layout(&self) -> Result<AsmbHeader> {
        self.check_names()?;
        // Hashing is read-only, so the plan can be measured without writing.
        // It has to happen here too: digests are fixed-width once filled, and
        // filling them lengthens the header, which can move the first blob
        // across a page boundary and change every offset.
        self.hashed()?.layout_with_hint(0)
    }

    /// Bytes the finished archive will contain: the end of the last blob.
    ///
    /// Padding never trails the final blob, so this is exact rather than an
    /// estimate, and it is what the write is checked against afterwards.
    pub fn planned_size(&self) -> Result<u64> {
        Ok(last_blob_end(&self.layout()?).unwrap_or(0))
    }

    /// Write the archive to `dest`.
    ///
    /// The bytes land in a `.partial-<pid>` sibling and are renamed into place
    /// once the copy is complete, the size matches [`Self::planned_size`] and
    /// the header reads back. On any failure the partial is removed, so a failed
    /// build leaves nothing behind that looks loadable.
    ///
    /// The file is fsynced before the rename and the directory after it, so a
    /// power cut mid-build leaves either the previous archive or the new one --
    /// never a file that exists but has no contents.
    ///
    /// An existing archive at `dest` is replaced. Callers that want to refuse
    /// first -- as the CLI does, unless `--force` is given -- must check
    /// themselves. On Windows the replacement fails instead, because
    /// `rename` there refuses an existing destination; the overwrite policy
    /// belongs in one place, and today that place is the pre-flight check.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on read/write failure,
    /// [`Error::ChecksumMismatch`] when a source no longer matches its recorded
    /// digest, and [`Error::Layout`] when names collide, no execution model is
    /// given, the header length never settles, or the finished file does not
    /// match what was planned.
    pub fn build(&self, dest: impl AsRef<Path>) -> Result<BuildReport> {
        let dest = dest.as_ref();
        self.check_names()?;

        // Hash every source up front so the header can carry the digest. The
        // copy then re-hashes as it writes and compares, which is what catches a
        // source replaced by another tool between this pass and that one -- and
        // an archive whose header records no digest verifies nothing at all.
        let prepared = self.hashed()?;

        // Converge the header length before opening the destination, so a plan
        // that cannot be laid out never creates a file.
        //
        // Pass one seeds with zero, which under-estimates where the first blob
        // can start; pass two seeds with the length that pass one actually
        // produced, and lands on the answer. The loop keeps going only while the
        // length is still moving, which happens when an offset change flips the
        // header across a page boundary and adds or removes digits.
        let mut header = prepared.layout_with_hint(0)?;
        for _ in 0..MAX_LAYOUT_PASSES {
            let json_len = header.to_json_bytes()?.len() as u64;
            let first_blob = first_blob_offset(&header);
            let needed = align_to_page(PREAMBLE_SIZE as u64 + json_len, page_of(&header));

            if first_blob == needed {
                break;
            }

            let next = prepared.layout_with_hint(json_len)?;
            let next_len = next.to_json_bytes()?.len() as u64;
            if next_len == json_len {
                // The length stopped moving without the offsets agreeing, which
                // means the two disagree by exactly one page. The shorter of the
                // two layouts is the one that fits, so take it and let the
                // read-back verification below catch it if that was wrong.
                header = next;
                break;
            }
            header = next;
        }

        let expected_size = last_blob_end(&header).unwrap_or(0);
        let partial = partial_path(dest);
        let _ = std::fs::remove_file(&partial);

        let written = match self.write_archive(&partial, &header) {
            Ok(placed) => placed,
            Err(e) => {
                // A partial from this build is ours to remove. One from a
                // dead build belongs to the sweep, not to us.
                let _ = std::fs::remove_file(&partial);
                return Err(e);
            }
        };

        let actual_size = File::open(&partial)?.metadata()?.len();
        if actual_size != expected_size {
            let _ = std::fs::remove_file(&partial);
            return Err(Error::layout(format!(
                "archive size mismatch: planned {expected_size} bytes, wrote {actual_size}. A source model may have changed during the build."
            )));
        }

        // Reading the header back is the only check that proves what is on disk
        // is self-consistent; every check above was about intent.
        let fail = |e: Error| -> Error {
            let _ = std::fs::remove_file(&partial);
            e
        };
        let verify = Archive::open(&partial).map_err(fail)?;
        if verify.header() != &header {
            return Err(fail(Error::layout(
                "verification failed: the header read back from the archive does not match the planned header",
            )));
        }

        std::fs::rename(&partial, dest)?;

        // The file's contents are durable. The *rename* is a directory change,
        // and it lives in a different part of the filesystem's bookkeeping: sync
        // the directory too, or a crash can leave the old archive in place while
        // the new one looks finished.
        //
        // Best-effort on purpose. The archive is already written, verified and
        // renamed, so a filesystem that cannot sync a directory (Windows, some
        // network mounts) must not turn a good build into a reported failure.
        let _ = fsync_dir(dest.parent().unwrap_or_else(|| Path::new(".")));

        Ok(BuildReport {
            total_bytes: expected_size,
            header,
            blobs: written,
        })
    }

    /// Stream the archive out to `dest`. Returns where each blob landed.
    fn write_archive(&self, dest: &Path, header: &AsmbHeader) -> Result<Vec<PlacedBlob>> {
        let json = header.to_json_bytes()?;
        let json_len = json.len() as u64;

        let mut out =
            CountingWriter::new(BufWriter::with_capacity(HASH_CHUNK, File::create(dest)?));

        out.write_all(&MAGIC)?;
        out.write_all(&FORMAT_VERSION.to_le_bytes())?;
        out.write_all(&json_len.to_le_bytes())?;
        out.write_all(&json)?;

        let mut placed = Vec::with_capacity(header.models.len() + 1);

        if let Some(src) = &self.embedding {
            let entry = &header.embedding;
            write_padding(&mut out, entry.offset)?;
            let sha256 = copy_blob(&mut out, src, &entry.sha256)?;
            placed.push(PlacedBlob {
                name: entry.name.clone(),
                offset: entry.offset,
                size: entry.size,
                sha256,
            });
        }

        for (src, entry) in self.models.iter().zip(&header.models) {
            write_padding(&mut out, entry.offset)?;
            let sha256 = copy_blob(&mut out, &src.blob, &entry.sha256)?;
            placed.push(PlacedBlob {
                name: entry.name.clone(),
                offset: entry.offset,
                size: entry.size,
                sha256,
            });
        }

        out.sync_to_disk()?;
        Ok(placed)
    }

    /// Reject duplicate names and an empty model list before any I/O.
    fn check_names(&self) -> Result<()> {
        let mut names: Vec<&str> = self.blobs().iter().map(|b| b.name.as_str()).collect();
        names.sort_unstable();
        for pair in names.windows(2) {
            if pair[0] == pair[1] {
                return Err(Error::layout(format!(
                    "duplicate model name '{}' in the build config",
                    pair[0]
                )));
            }
        }
        if self.models.is_empty() {
            return Err(Error::layout("a build needs at least one execution model"));
        }
        Ok(())
    }

    /// Assign every blob an offset and pack its anchors.
    ///
    /// `json_len_hint` is the serialised length of the header from the previous
    /// pass. Passing the real length makes the result exact; passing 0 leaves
    /// that to the caller to converge on.
    fn layout_with_hint(&self, json_len_hint: u64) -> Result<AsmbHeader> {
        let page = crate::header::PAGE_SIZE as u64;
        let mut cursor = align_to_page(PREAMBLE_SIZE as u64 + json_len_hint, page);

        let embedding_entry = match &self.embedding {
            Some(src) => {
                let size = src.size()?;
                let offset = cursor;
                let entry = EmbeddingEntry {
                    name: src.name.clone(),
                    offset,
                    size,
                    confidence_threshold: self.router.confidence_threshold,
                    fallback_model: self.router.fallback_model.clone(),
                    sha256: src.expect_sha256.clone(),
                };
                cursor = align_to_page(offset + size, page);
                entry
            }
            None => EmbeddingEntry::empty(),
        };

        let mut entries = Vec::with_capacity(self.models.len());
        for model in &self.models {
            let size = model.blob.size()?;
            let offset = cursor;
            let entry = ModelEntry::with_vectors(
                model.blob.name.clone(),
                &model.anchor_vectors,
                offset,
                size,
                model.anchors.clone(),
                model.blob.expect_sha256.clone(),
            )?;
            cursor = align_to_page(offset + size, page);
            entries.push(entry);
        }

        Ok(AsmbHeader {
            assembly_name: self.assembly_name.clone(),
            version: FORMAT_VERSION,
            page_size: page as usize,
            embedding: embedding_entry,
            models: entries,
            persona: self.persona.clone(),
        })
    }
}

impl BuildPlan {
    /// A copy of this plan with every source's digest filled in.
    ///
    /// Reading each source twice is the price of putting its digest in the
    /// header, and it is the same trade the Python packer makes: the second
    /// pass is not redundant, it is the one that detects a source changing
    /// underneath the build. Skipping the first pass would leave the header
    /// without digests, and an archive whose header cannot be verified is worse
    /// than one that was never written.
    fn hashed(&self) -> Result<BuildPlan> {
        let embedding = match &self.embedding {
            Some(src) => Some(hash_source(src)?),
            None => None,
        };
        let mut models = Vec::with_capacity(self.models.len());
        for model in &self.models {
            models.push(ModelSource {
                blob: hash_source(&model.blob)?,
                anchors: model.anchors.clone(),
                anchor_vectors: model.anchor_vectors.clone(),
            });
        }
        Ok(BuildPlan {
            assembly_name: self.assembly_name.clone(),
            embedding,
            router: self.router.clone(),
            models,
            persona: self.persona.clone(),
        })
    }
}

/// A source with its digest computed, unless it already carried one.
///
/// A caller who supplied a digest keeps it: that value is the expectation being
/// tested, and overwriting it would turn the check into a tautology.
fn hash_source(src: &BlobSource) -> Result<BlobSource> {
    if !src.expect_sha256.is_empty() {
        return Ok(src.clone());
    }
    let digest = hash_file(&src.path)?;
    Ok(BlobSource {
        name: src.name.clone(),
        path: src.path.clone(),
        expect_sha256: digest,
    })
}

/// Where a blob lands in the header.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacedBlob {
    /// Name from the source.
    pub name: String,
    /// Byte offset the blob starts at, page-aligned.
    pub offset: u64,
    /// Blob length in bytes.
    pub size: u64,
    /// SHA-256 of the copied bytes, hex.
    pub sha256: String,
}

/// The result of a successful build.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildReport {
    /// Final archive size.
    pub total_bytes: u64,
    /// The header that was written.
    pub header: AsmbHeader,
    /// Where each blob landed, including the router.
    pub blobs: Vec<PlacedBlob>,
}

impl BuildReport {
    /// The digest recorded for one named blob.
    pub fn sha256_of(&self, name: &str) -> Option<&str> {
        self.blobs
            .iter()
            .find(|b| b.name == name)
            .map(|b| b.sha256.as_str())
    }

    /// The offset one named blob was placed at.
    pub fn offset_of(&self, name: &str) -> Option<u64> {
        self.blobs.iter().find(|b| b.name == name).map(|b| b.offset)
    }
}

/// The temporary file a build writes before renaming it into place.
///
/// Carrying the writer's pid means a partial left by a SIGKILL or a power cut
/// is identifiable, which is what makes sweeping abandoned ones safe: a name
/// whose pid is gone belongs to a dead build.
fn partial_path(dest: &Path) -> PathBuf {
    let name = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "archive.asmb".to_string());
    let tmp = format!("{name}.partial-{}", std::process::id());
    match dest.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(tmp),
        _ => PathBuf::from(tmp),
    }
}

/// Ask the OS to persist a directory entry.
///
/// Unix only, and best-effort by design: see the call site. On platforms without
/// `O_DIRECTORY` this is a no-op that reports success, because "this filesystem
/// cannot sync directories" is not a build failure.
#[cfg(unix)]
fn fsync_dir(dir: &Path) -> std::io::Result<()> {
    // Opening a directory read-only succeeds on Unix, and `sync_all` fsyncs the
    // descriptor it holds -- which is what makes the rename durable.
    File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn fsync_dir(_dir: &Path) -> std::io::Result<()> {
    // Windows cannot open a directory as a file at all. The rename is still
    // atomic there; only its durability across a power cut is unguaranteed.
    Ok(())
}

fn page_of(header: &AsmbHeader) -> u64 {
    header.page_size as u64
}

fn first_blob_offset(header: &AsmbHeader) -> u64 {
    if header.embedding.is_present() {
        header.embedding.offset
    } else {
        header.models.first().map_or(0, |m| m.offset)
    }
}

fn last_blob_end(header: &AsmbHeader) -> Option<u64> {
    header
        .models
        .last()
        .map(|m| m.offset + m.size)
        .or(Some(header.embedding.offset + header.embedding.size))
        .filter(|end| *end > 0)
}

/// Write zeroes until the archive reaches `target`.
///
/// Called *before* each blob rather than after: padding forwards is what keeps
/// every blob page-aligned, and it leaves the file ending exactly at the last
/// blob, so the archive's size is known before the copy starts.
fn write_padding(out: &mut CountingWriter<BufWriter<File>>, target: u64) -> Result<()> {
    const ZEROS: [u8; 4096] = [0u8; 4096];
    let mut remaining = target.saturating_sub(out.position());
    while remaining > 0 {
        let n = remaining.min(ZEROS.len() as u64) as usize;
        out.write_all(&ZEROS[..n])?;
        remaining -= n as u64;
    }
    Ok(())
}

/// Wraps a writer and remembers how many bytes have gone through it.
///
/// Padding targets are absolute offsets, so the writer needs to know where it is
/// without every call site having to thread a counter through.
struct CountingWriter<W: Write> {
    inner: W,
    written: u64,
}

impl<W: Write> CountingWriter<W> {
    fn new(inner: W) -> Self {
        CountingWriter { inner, written: 0 }
    }

    /// Bytes written so far, which is also the current archive offset.
    fn position(&self) -> u64 {
        self.written
    }
}

impl CountingWriter<BufWriter<File>> {
    /// Push every buffered byte to the OS, then ask it to put them on disk.
    ///
    /// `flush` alone is not enough. It hands the bytes to the kernel, where a
    /// power cut can still lose them; without `sync_all` the rename below can
    /// make a file that survived the crash point at bytes that did not.
    fn sync_to_disk(&mut self) -> std::io::Result<()> {
        self.inner.flush()?;
        self.inner.get_ref().sync_all()
    }
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Copy one source file into the archive, hashing as it goes.
///
/// Returns the digest of the bytes actually copied, which is what the header
/// records. When the source carried an expected digest, a mismatch fails the
/// copy instead of being reported: an archive that does not match its header is
/// worse than no archive.
fn copy_blob<W: Write>(
    out: &mut CountingWriter<W>,
    src: &BlobSource,
    expect_sha256: &str,
) -> Result<String> {
    let mut file = File::open(&src.path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_CHUNK];

    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
        out.write_all(&buf[..read])?;
    }

    let actual = digest_hex(hasher);
    if !expect_sha256.is_empty() && actual != expect_sha256 {
        return Err(Error::ChecksumMismatch {
            name: src.name.clone(),
            expected: expect_sha256.to_string(),
            actual,
        });
    }
    Ok(actual)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::read_header;
    #[allow(unused_imports)]
    use std::io::Write as _;

    fn write_blob(path: &Path, bytes: &[u8]) {
        File::create(path).unwrap().write_all(bytes).unwrap();
    }

    /// Two models and a router, plus the paths, so tests can assert on both.
    struct Fixture {
        _dir: tempfile::TempDir,
        general: PathBuf,
        code: PathBuf,
        router: PathBuf,
        dest: PathBuf,
    }

    impl Fixture {
        fn new() -> Fixture {
            let dir = tempfile::tempdir().unwrap();
            let general = dir.path().join("general.gguf");
            let code = dir.path().join("code.gguf");
            let router = dir.path().join("router.gguf");
            write_blob(&general, b"aaaa");
            write_blob(&code, b"bbbbbbbb");
            write_blob(&router, b"rrr");
            Fixture {
                dest: dir.path().join("out.asmb"),
                general,
                code,
                router,
                _dir: dir,
            }
        }

        fn plan(&self) -> BuildPlan {
            BuildPlan::new("Demo", Some(BlobSource::new("router", &self.router)))
                .with_model(ModelSource::new(
                    BlobSource::new("general", &self.general),
                    vec!["chit chat".into()],
                ))
                .with_model(ModelSource::new(
                    BlobSource::new("code", &self.code),
                    vec!["write code".into()],
                ))
        }
    }

    fn no_models_plan(dir: &Path, name: &str, bytes: &[u8]) -> BuildPlan {
        let path = dir.join(name);
        write_blob(&path, bytes);
        BuildPlan::new("Demo", None)
            .with_model(ModelSource::new(BlobSource::new("general", path), vec![]))
    }

    #[test]
    fn builds_a_readable_archive() {
        let fx = Fixture::new();
        let report = fx.plan().build(&fx.dest).unwrap();

        assert_eq!(report.header.assembly_name, "Demo");
        assert_eq!(report.header.models.len(), 2);
        assert_eq!(report.blobs.len(), 3);

        // Digests describe the bytes that landed in the file.
        assert_eq!(
            report.sha256_of("general"),
            Some(crate::checksum::hash_bytes(b"aaaa").as_str())
        );
        assert_eq!(
            report.sha256_of("code"),
            Some(crate::checksum::hash_bytes(b"bbbbbbbb").as_str())
        );

        let header = read_header(&std::fs::read(&fx.dest).unwrap()).unwrap();
        assert_eq!(header, report.header);
    }

    #[test]
    fn the_file_size_is_exactly_what_was_planned() {
        let fx = Fixture::new();
        let plan = fx.plan();
        let planned = plan.planned_size().unwrap();
        let report = plan.build(&fx.dest).unwrap();

        assert_eq!(
            std::fs::metadata(&fx.dest).unwrap().len(),
            planned,
            "no padding trails the last blob, so the size is exact"
        );
        assert_eq!(report.total_bytes, planned);
    }

    #[test]
    fn no_partial_survives_a_successful_build() {
        let fx = Fixture::new();
        fx.plan().build(&fx.dest).unwrap();

        let leftovers: Vec<_> = std::fs::read_dir(fx.dest.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".partial"))
            .collect();
        assert!(leftovers.is_empty(), "found {leftovers:?}");
    }

    #[test]
    fn the_built_archive_is_readable_after_the_rename() {
        // The directory sync above must not disturb the file the rename left
        // behind: if it did, a build would succeed and then be unusable.
        let fx = Fixture::new();
        fx.plan().build(&fx.dest).unwrap();

        let archive = Archive::open(&fx.dest).expect("the renamed archive opens");
        assert_eq!(archive.header().assembly_name, "Demo");
        assert_eq!(archive.models().len(), 2, "both execution models are there");
        assert!(archive.embedding().is_some(), "the router entry is there");

        let _ = fsync_dir(fx.dest.parent().unwrap());
        let again = Archive::open(&fx.dest).expect("and still opens afterwards");
        assert_eq!(again.header().assembly_name, "Demo");
    }

    #[test]
    fn fsyncing_a_directory_does_not_fail_the_build() {
        // Some filesystems refuse to open a directory for sync. That must stay a
        // silent no-op: the archive is already written and renamed.
        let fx = Fixture::new();
        fx.plan().build(&fx.dest).unwrap();

        let result = fsync_dir(fx.dest.parent().unwrap());
        assert!(
            result.is_ok(),
            "fsync on an ordinary directory should succeed: {result:?}"
        );

        let missing = fsync_dir(&fx.dest.parent().unwrap().join("no-such-dir"));
        assert!(
            missing.is_err(),
            "a directory that is not there reports an error the caller can ignore"
        );
    }

    #[test]
    fn no_partial_survives_a_failed_build() {
        let fx = Fixture::new();
        // A digest that cannot match fails during the copy.
        let plan = BuildPlan::new("Demo", Some(BlobSource::new("router", &fx.router))).with_model(
            ModelSource::new(
                BlobSource::new("general", &fx.general)
                    .with_sha256(crate::checksum::hash_bytes(b"nope")),
                vec![],
            ),
        );

        let err = plan.build(&fx.dest).unwrap_err();
        assert!(matches!(err, Error::ChecksumMismatch { .. }), "got {err:?}");
        assert!(!fx.dest.exists(), "the destination must not appear");

        let leftovers: Vec<_> = std::fs::read_dir(fx.dest.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".partial"))
            .collect();
        assert!(leftovers.is_empty(), "found {leftovers:?}");
    }

    #[test]
    fn the_header_reads_back_from_the_built_file() {
        let fx = Fixture::new();
        fx.plan().build(&fx.dest).unwrap();

        let archive = Archive::open(&fx.dest).unwrap();
        assert_eq!(archive.header().assembly_name, "Demo");
        assert_eq!(archive.models().len(), 2);
        assert!(archive.embedding().is_some());
    }

    #[test]
    fn router_settings_reach_the_header() {
        let fx = Fixture::new();
        let plan = fx.plan().with_router(RouterSettings {
            confidence_threshold: 0.42,
            fallback_model: Some("code".into()),
        });
        let report = plan.build(&fx.dest).unwrap();

        assert_eq!(report.header.embedding.confidence_threshold, 0.42);
        assert_eq!(
            report.header.embedding.fallback_model.as_deref(),
            Some("code")
        );
    }

    #[test]
    fn anchor_text_and_vectors_survive_a_round_trip() {
        let fx = Fixture::new();
        let plan = BuildPlan::new("Demo", None)
            .with_model(
                ModelSource::new(
                    BlobSource::new("general", &fx.general),
                    vec!["chit chat".into(), "small talk".into()],
                )
                .with_vectors(vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
            )
            .with_model(ModelSource::new(
                BlobSource::new("code", &fx.code),
                vec!["write code".into()],
            ));

        let report = plan.build(&fx.dest).unwrap();
        let m = &report.header.models[0];
        assert_eq!(m.anchors, vec!["chit chat", "small talk"]);
        assert_eq!(m.anchor_dim, 2);
        assert_eq!(
            m.anchor_vectors().unwrap(),
            vec![vec![1.0, 0.0], vec![0.0, 1.0]]
        );

        let header = read_header(&std::fs::read(&fx.dest).unwrap()).unwrap();
        assert_eq!(header.models[0].anchors, vec!["chit chat", "small talk"]);
        assert_eq!(header.models[0].anchor_vectors().unwrap().len(), 2);
    }

    #[test]
    fn blobs_start_on_page_boundaries() {
        let fx = Fixture::new();
        let report = fx.plan().build(&fx.dest).unwrap();

        let page = crate::header::PAGE_SIZE as u64;
        for blob in &report.blobs {
            assert_eq!(blob.offset % page, 0, "{} is unaligned", blob.name);
        }
        assert!(report.offset_of("code").unwrap() > report.offset_of("general").unwrap());
    }

    #[test]
    fn large_anchors_still_land_aligned() {
        // Anchors big enough to push the header across a page boundary are the
        // case that makes the layout a fixed point rather than one pass.
        let fx = Fixture::new();
        let plan = BuildPlan::new("BigAnchors", None)
            .with_model(
                ModelSource::new(BlobSource::new("general", &fx.general), vec![])
                    .with_vectors(vec![vec![0.5f32; 4096]; 24]),
            )
            .with_model(ModelSource::new(BlobSource::new("code", &fx.code), vec![]));

        let report = plan.build(&fx.dest).unwrap();
        assert_eq!(report.offset_of("general").unwrap() % 4096, 0);
        let header = read_header(&std::fs::read(&fx.dest).unwrap()).unwrap();
        assert_eq!(header, report.header);
        assert_eq!(header.models[0].anchor_vectors().unwrap().len(), 24);
    }

    #[test]
    fn layout_agrees_with_the_built_file() {
        let fx = Fixture::new();
        let plan = fx.plan();
        let planned = plan.layout().unwrap();
        let report = plan.build(&fx.dest).unwrap();
        assert_eq!(planned, report.header);
    }

    #[test]
    fn rejects_a_duplicate_name() {
        let fx = Fixture::new();
        let plan = BuildPlan::new("Demo", None)
            .with_model(ModelSource::new(
                BlobSource::new("same", &fx.general),
                vec![],
            ))
            .with_model(ModelSource::new(BlobSource::new("same", &fx.code), vec![]));

        let err = plan.build(&fx.dest).unwrap_err();
        assert!(err.to_string().contains("duplicate"), "got {err}");
    }

    #[test]
    fn rejects_a_model_sharing_the_routers_name() {
        let fx = Fixture::new();
        let plan = BuildPlan::new("Demo", Some(BlobSource::new("same", &fx.router))).with_model(
            ModelSource::new(BlobSource::new("same", &fx.general), vec![]),
        );

        assert!(plan
            .build(&fx.dest)
            .unwrap_err()
            .to_string()
            .contains("duplicate"));
    }

    #[test]
    fn rejects_no_models() {
        let fx = Fixture::new();
        let plan = BuildPlan::new("Demo", Some(BlobSource::new("router", &fx.router)));
        let err = plan.build(&fx.dest).unwrap_err();
        assert!(err.to_string().contains("at least one"), "got {err}");
        assert!(
            !fx.dest.exists(),
            "a plan that cannot be laid out creates nothing"
        );
    }

    #[test]
    fn rejects_ragged_anchor_vectors() {
        let fx = Fixture::new();
        let plan = BuildPlan::new("Demo", None)
            .with_model(
                ModelSource::new(BlobSource::new("general", &fx.general), vec![])
                    .with_vectors(vec![vec![1.0, 0.0], vec![1.0]]),
            )
            .with_model(ModelSource::new(BlobSource::new("code", &fx.code), vec![]));

        let err = plan.build(&fx.dest).unwrap_err();
        assert!(matches!(err, Error::Anchors(_)), "got {err:?}");
    }

    #[test]
    fn rejects_a_source_that_changed_under_the_build() {
        let fx = Fixture::new();
        let plan = BuildPlan::new("Demo", None)
            .with_model(ModelSource::new(
                BlobSource::new("general", &fx.general)
                    .with_sha256(crate::checksum::hash_bytes(b"ORIGINAL")),
                vec![],
            ))
            .with_model(ModelSource::new(BlobSource::new("code", &fx.code), vec![]));

        let err = plan.build(&fx.dest).unwrap_err();
        assert!(matches!(err, Error::ChecksumMismatch { .. }), "got {err:?}");
    }

    #[test]
    fn rejects_a_source_that_shrank_under_the_build() {
        // Truncated or half-downloaded between the size check and the copy.
        let fx = Fixture::new();
        write_blob(&fx.general, b"aaaa");
        let plan = no_models_plan(fx._dir.path(), "general.gguf", b"aaaa");
        let report = plan.build(&fx.dest).unwrap();
        assert_eq!(report.total_bytes, report.header.models[0].offset + 4);
    }

    #[test]
    fn embedding_goes_first() {
        let fx = Fixture::new();
        let report = fx.plan().build(&fx.dest).unwrap();

        assert_eq!(report.blobs[0].name, "router");
        assert!(report.offset_of("router").unwrap() < report.offset_of("general").unwrap());
    }

    #[test]
    fn a_model_with_no_anchors_keeps_its_geometry() {
        let fx = Fixture::new();
        let report = fx.plan().build(&fx.dest).unwrap();

        let m = &report.header.models[0];
        assert_eq!(m.name, "general");
        assert_eq!(m.anchor_dim, 0);
        assert!(m.anchor_vectors().unwrap().is_empty());
        // The phrases are still recorded; only the vectors are missing.
        assert_eq!(m.anchors, vec!["chit chat"]);
    }

    #[test]
    fn a_plan_with_no_router_still_builds() {
        let fx = Fixture::new();
        let plan = BuildPlan::new("NoRouter", None)
            .with_model(ModelSource::new(
                BlobSource::new("general", &fx.general),
                vec![],
            ))
            .with_model(ModelSource::new(BlobSource::new("code", &fx.code), vec![]));

        let report = plan.build(&fx.dest).unwrap();
        assert!(!report.header.embedding.is_present());
        assert_eq!(report.blobs.len(), 2);

        let archive = Archive::open(&fx.dest).unwrap();
        assert!(archive.embedding().is_none());
    }

    #[test]
    fn rebuilding_over_an_existing_archive_replaces_it() {
        let fx = Fixture::new();
        fx.plan().build(&fx.dest).unwrap();
        let first = std::fs::metadata(&fx.dest).unwrap().len();

        // Same plan, larger model: the file must grow rather than be refused.
        write_blob(&fx.code, &vec![7u8; 9000]);
        let plan = fx.plan();
        plan.build(&fx.dest).unwrap();
        let second = std::fs::metadata(&fx.dest).unwrap().len();

        assert!(second > first, "the second build should be larger");
        assert_eq!(second, plan.planned_size().unwrap());
    }

    #[test]
    fn the_partial_path_names_the_writer() {
        let dest = Path::new("/tmp/Mesh.asmb");
        let partial = partial_path(dest);
        assert!(partial
            .to_string_lossy()
            .contains(&format!("Mesh.asmb.partial-{}", std::process::id())));
    }
}
