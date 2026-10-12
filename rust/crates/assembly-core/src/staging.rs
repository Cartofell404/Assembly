//! Staged blob cache.
//!
//! llama.cpp wants a real path for a model, not a byte range inside an
//! archive, so getting a blob ready means copying it out. Doing that on every
//! load would re-copy gigabytes per request, so extracted blobs are kept and
//! reused, keyed by which archive and which region of it they came from.
//!
//! The cache is bounded by *bytes*, not by entry count: three 500 MB models
//! and three 4 GB models need very different treatment, and a count limit
//! cannot tell the difference.
//!
//! # Ordering
//!
//! Least-recently-*used* is the wrong order here. A blob that was just
//! extracted is the one most likely to be asked for again -- that is why it
//! was extracted. Eviction therefore goes oldest-extracted first, which keeps
//! the working set and throws away whatever sat unused longest.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use asmb_format::Result;

/// Identifies one archive on disk.
///
/// Two archives with the same path are the same archive unless their contents
/// changed underneath us, which the build's checksum verification is there to
/// catch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArchiveId {
    /// Absolute path to the `.asmb` file.
    path: PathBuf,
    /// Archive length in bytes when it was registered.
    len: u64,
    /// Modification time, when the platform reports one cheaply.
    mtime: Option<std::time::SystemTime>,
}

impl ArchiveId {
    /// Identify an archive by stat'ing it.
    ///
    /// # Errors
    ///
    /// Returns [`asmb_format::Error::Io`] if the file cannot be stat'd.
    pub fn from_path(path: impl AsRef<Path>) -> Result<ArchiveId> {
        let path = path.as_ref();
        let meta = std::fs::metadata(path)?;
        Ok(ArchiveId {
            path: path.to_path_buf(),
            len: meta.len(),
            mtime: meta.modified().ok(),
        })
    }

    /// Identify an archive without touching the filesystem. Used by tests and
    /// by callers that have already validated the archive.
    pub fn new(path: impl Into<PathBuf>, len: u64) -> ArchiveId {
        ArchiveId {
            path: path.into(),
            len,
            mtime: None,
        }
    }

    /// Path of the archive this identity refers to.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// What a staged blob was cut from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StagedKey {
    /// Which archive.
    pub archive: ArchiveId,
    /// Byte offset of the blob inside that archive.
    pub offset: u64,
    /// Byte length of the blob.
    pub size: u64,
}

impl StagedKey {
    /// The key for one blob of one archive.
    pub fn new(archive: ArchiveId, offset: u64, size: u64) -> StagedKey {
        StagedKey {
            archive,
            offset,
            size,
        }
    }
}

#[derive(Debug, Default)]
struct Inner {
    /// Entries in extraction order, oldest first. `HashMap` gives lookup,
    /// the `Vec` gives eviction order without a separate ordering structure.
    order: Vec<StagedKey>,
    paths: HashMap<StagedKey, PathBuf>,
    bytes: u64,
}

/// Cache of extracted blobs, bounded by total bytes.
/// Where staged blobs go when the caller does not say.
///
/// `ASSEMBLY_TEMP_DIR` wins when set, so a user with a fast scratch volume can
/// put multi-gigabyte extractions there instead of on the system disk. Falls
/// back to the OS temp directory, whose name differs across platforms.
pub fn default_staging_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("ASSEMBLY_TEMP_DIR") {
        if !custom.is_empty() {
            return PathBuf::from(custom);
        }
    }
    std::env::temp_dir().join("assembly-staging")
}

/// Extracted model files, keyed by name, with a total-size ceiling.
///
/// Staging is separate from the resident cache on purpose: the resident cache
/// decides what stays in RAM and may drop a model, while this holds the extracted
/// `.gguf` on disk. A model evicted from memory is then a file rename away from
/// being usable again, not a multi-gigabyte re-extraction.
#[derive(Debug)]
pub struct StagingCache {
    inner: Mutex<Inner>,
    max_bytes: Option<u64>,
    dir: PathBuf,
}

impl StagingCache {
    /// A cache that evicts down to `max_bytes` once exceeded.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::ZeroBudget`] for a limit of zero, which would
    /// evict every blob the instant it was staged and silently turn the cache
    /// into a no-op.
    pub fn with_limit(max_bytes: u64) -> std::result::Result<StagingCache, ConfigError> {
        if max_bytes == 0 {
            return Err(ConfigError::ZeroBudget);
        }
        let dir = default_staging_dir();
        // Created here, not lazily: these constructors cannot report an error,
        // and a missing directory would otherwise surface as a confusing
        // "no such file" from the first extraction instead of here.
        let _ = std::fs::create_dir_all(&dir);
        Ok(StagingCache {
            inner: Mutex::new(Inner::default()),
            max_bytes: Some(max_bytes),
            dir,
        })
    }

    /// A cache that keeps everything until it is dropped.
    ///
    /// The right default when the staging volume is known to be large enough,
    /// since it makes repeat loads free.
    pub fn unbounded() -> StagingCache {
        let dir = default_staging_dir();
        let _ = std::fs::create_dir_all(&dir);
        StagingCache {
            inner: Mutex::new(Inner::default()),
            max_bytes: None,
            dir,
        }
    }

    /// The byte limit, if there is one.
    pub fn limit(&self) -> Option<u64> {
        self.max_bytes
    }

    /// The directory staged blobs are written into.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// A cache rooted at `dir` instead of the system temp directory.
    ///
    /// The directory is created if missing. A caller that cannot create it
    /// finds out here rather than on the first request, which is the point.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::Error`] if the directory cannot be created.
    pub fn in_dir(dir: impl Into<PathBuf>) -> std::io::Result<StagingCache> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(StagingCache {
            inner: Mutex::new(Inner::default()),
            max_bytes: None,
            dir,
        })
    }

    /// A free path for a blob of `size` bytes, named for `label`.
    ///
    /// The file is not created. The caller extracts into it and then records it
    /// with [`StagingCache::remember`], so a half-written file is never visible
    /// to the cache as reusable.
    ///
    /// The name mixes the label with a counter and the current time: models are
    /// staged by name, and two archives in one process must not collide on
    /// `general.gguf`.
    pub fn allocate(&self, label: &str, size: u64) -> std::io::Result<PathBuf> {
        // The directory can vanish between runs (a cleaner, a tmpfs reset), and
        // a request should not fail because of it.
        std::fs::create_dir_all(&self.dir)?;

        let safe: String = label
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        let mut path = self.dir.join(format!("{safe}-{nanos:x}-{size}"));
        // Belt and braces: if the name somehow exists, walk it forward rather
        // than overwriting whatever is there.
        let mut bump = 0u32;
        while path.exists() {
            bump += 1;
            path = self.dir.join(format!("{safe}-{nanos:x}-{size}-{bump}"));
        }
        Ok(path)
    }

    /// Bytes currently held by staged blobs.
    pub fn bytes(&self) -> u64 {
        self.inner.lock().expect("staging cache poisoned").bytes
    }

    /// Number of staged blobs.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("staging cache poisoned")
            .order
            .len()
    }

    /// Whether nothing is staged.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Path of an already-staged blob, if there is one.
    pub fn get(&self, key: &StagedKey) -> Option<PathBuf> {
        let inner = self.inner.lock().expect("staging cache poisoned");
        let path = inner.paths.get(key)?;
        // A blob someone deleted behind our back is not reusable.
        if !path.exists() {
            return None;
        }
        Some(path.clone())
    }

    /// Record `path` as the staged file for `key`, evicting if needed.
    ///
    /// Re-staging an existing key replaces its path and moves it to the newest
    /// slot, so a blob rewritten from a changed archive does not keep the old
    /// file alive under a stale key.
    ///
    /// Returns the paths evicted to make room, in eviction order. The caller
    /// deletes them: this type never unlinks files it did not create, which
    /// keeps it safe to point at a directory someone else owns.
    pub fn remember(&self, key: StagedKey, path: PathBuf) -> Vec<PathBuf> {
        let size = std::fs::metadata(&path)
            .map(|m| m.len())
            .unwrap_or(key.size);

        let mut evicted = Vec::new();
        {
            let mut inner = self.inner.lock().expect("staging cache poisoned");

            // Replacing an entry: drop the old accounting first.
            if let Some(old) = inner.paths.insert(key.clone(), path.clone()) {
                if old != path {
                    let old_size = file_len(&old);
                    inner.bytes = inner.bytes.saturating_sub(old_size);
                    inner.order.retain(|k| *k != key);
                }
            }
            inner.order.push(key.clone());
            inner.bytes += size;

            if let Some(limit) = self.max_bytes {
                // Never evict the blob just staged: the caller needs it right
                // now, and dropping it would make the cache a no-op. A blob
                // larger than the whole budget therefore stays until something
                // else displaces it, which is the only behaviour that leaves
                // the caller with a usable path.
                while inner.bytes > limit && inner.order.len() > 1 {
                    match evict_oldest(&mut inner) {
                        Some(victim) => evicted.push(victim),
                        None => break,
                    }
                }
            }
        }
        evicted
    }

    /// Drop `key` from the cache, returning the path it held.
    pub fn forget(&self, key: &StagedKey) -> Option<PathBuf> {
        let mut inner = self.inner.lock().expect("staging cache poisoned");
        let path = inner.paths.remove(key)?;
        inner.order.retain(|k| k != key);
        inner.bytes = inner.bytes.saturating_sub(file_len(&path));
        Some(path)
    }

    /// Drop every blob belonging to one archive, returning their paths.
    ///
    /// Called when an archive closes: its staged copies are only useful while
    /// it is open.
    pub fn forget_archive(&self, archive: &ArchiveId) -> Vec<PathBuf> {
        let mut inner = self.inner.lock().expect("staging cache poisoned");
        let victims: Vec<StagedKey> = inner
            .order
            .iter()
            .filter(|k| k.archive == *archive)
            .cloned()
            .collect();

        let mut out = Vec::with_capacity(victims.len());
        for key in victims {
            if let Some(path) = inner.paths.remove(&key) {
                inner.bytes = inner.bytes.saturating_sub(file_len(&path));
                out.push(path);
            }
        }
        inner.order.retain(|k| k.archive != *archive);
        out
    }

    /// Drop everything, returning the paths that were held.
    pub fn clear(&self) -> Vec<PathBuf> {
        let mut inner = self.inner.lock().expect("staging cache poisoned");
        let paths: Vec<PathBuf> = inner.paths.values().cloned().collect();
        inner.order.clear();
        inner.paths.clear();
        inner.bytes = 0;
        paths
    }
}

/// Remove the oldest entry, returning its path. `None` when the cache is
/// logically empty, which can only happen if accounting drifted.
fn evict_oldest(inner: &mut Inner) -> Option<PathBuf> {
    let key = inner.order.first()?.clone();
    let path = inner.paths.remove(&key)?;
    let size = file_len(&path);
    inner.order.remove(0);
    inner.bytes = inner.bytes.saturating_sub(size);
    Some(path)
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Configuration problems for [`StagingCache`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// A byte limit of zero was requested.
    #[error("staging cache limit must be greater than zero bytes")]
    ZeroBudget,
    /// A memory headroom fraction outside `(0, 1]` was requested.
    #[error("memory headroom must be greater than 0 and at most 1, got {requested}")]
    BadHeadroom {
        /// The value that was asked for.
        requested: String,
    },
    /// The platform did not report physical memory, or something else about the
    /// machine could not be read.
    #[error("{0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write as _;

    fn blob(dir: &Path, name: &str, size: usize) -> PathBuf {
        let path = dir.join(name);
        File::create(&path)
            .unwrap()
            .write_all(&vec![0u8; size])
            .unwrap();
        path
    }

    /// A key that is unique per `name`, so one test can stage several blobs
    /// whose sizes and offsets would otherwise collide.
    fn key(name: &str, size: u64) -> StagedKey {
        let offset = name.bytes().fold(1u64, |acc, b| acc * 31 + b as u64);
        StagedKey::new(ArchiveId::new("/archives/a.asmb", 1000), offset, size)
    }

    #[test]
    fn starts_empty() {
        let cache = StagingCache::unbounded();
        assert!(cache.is_empty());
        assert_eq!(cache.bytes(), 0);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn remembers_and_returns_a_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = blob(dir.path(), "a.gguf", 100);
        let cache = StagingCache::unbounded();
        let k = key("a", 100);

        assert!(cache.remember(k.clone(), path.clone()).is_empty());
        assert_eq!(cache.get(&k), Some(path));
        assert_eq!(cache.bytes(), 100);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn misses_on_an_unknown_key() {
        let cache = StagingCache::unbounded();
        assert_eq!(cache.get(&key("nope", 10)), None);
    }

    #[test]
    fn misses_when_the_file_vanished() {
        let dir = tempfile::tempdir().unwrap();
        let path = blob(dir.path(), "a.gguf", 100);
        let cache = StagingCache::unbounded();
        let k = key("a", 100);
        cache.remember(k.clone(), path.clone());
        std::fs::remove_file(&path).unwrap();

        assert_eq!(cache.get(&k), None, "a deleted blob is not reusable");
    }

    #[test]
    fn unbounded_cache_never_evicts() {
        let dir = tempfile::tempdir().unwrap();
        let cache = StagingCache::unbounded();
        for i in 0..5 {
            let path = blob(dir.path(), &format!("m{i}.gguf"), 1000);
            assert!(cache.remember(key(&format!("m{i}"), 1000), path).is_empty());
        }
        assert_eq!(cache.len(), 5);
        assert_eq!(cache.bytes(), 5000);
    }

    #[test]
    fn evicts_oldest_first_to_fit() {
        let dir = tempfile::tempdir().unwrap();
        let cache = StagingCache::with_limit(2500).unwrap();

        let a = blob(dir.path(), "a.gguf", 1000);
        let b = blob(dir.path(), "b.gguf", 1000);
        let c = blob(dir.path(), "c.gguf", 1000);

        cache.remember(key("a", 1000), a.clone());
        cache.remember(key("b", 1000), b.clone());
        // This one pushes past the limit, so the oldest (a) goes.
        let evicted = cache.remember(key("c", 1000), c.clone());

        assert_eq!(evicted, vec![a]);
        assert_eq!(cache.get(&key("a", 1000)), None);
        assert_eq!(cache.get(&key("b", 1000)), Some(b));
        assert_eq!(cache.get(&key("c", 1000)), Some(c));
        assert_eq!(cache.bytes(), 2000);
    }

    #[test]
    fn evicts_as_many_as_needed() {
        let dir = tempfile::tempdir().unwrap();
        let cache = StagingCache::with_limit(1200).unwrap();

        let mut paths = Vec::new();
        for i in 0..4 {
            let p = blob(dir.path(), &format!("m{i}.gguf"), 1000);
            paths.push(p.clone());
            cache.remember(key(&format!("m{i}"), 1000), p);
        }

        // 4000 bytes staged under a 1200 limit: three victims.
        assert_eq!(cache.bytes(), 1000);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&key("m0", 1000)), None);
        assert_eq!(cache.get(&key("m3", 1000)), Some(paths[3].clone()));
    }

    #[test]
    fn a_blob_larger_than_the_limit_still_lands() {
        let dir = tempfile::tempdir().unwrap();
        let cache = StagingCache::with_limit(100).unwrap();
        let big = blob(dir.path(), "big.gguf", 500);

        // Nothing else to evict, so the oversized entry stays: dropping it
        // would mean the caller can never use this cache at all.
        let evicted = cache.remember(key("big", 500), big.clone());
        assert!(evicted.is_empty());
        assert_eq!(cache.get(&key("big", 500)), Some(big));
        assert_eq!(cache.bytes(), 500);
    }

    #[test]
    fn rejects_a_zero_limit() {
        let err = StagingCache::with_limit(0).unwrap_err();
        assert!(matches!(err, ConfigError::ZeroBudget));
        assert!(err.to_string().contains("greater than zero"));
    }

    #[test]
    fn re_staging_the_same_key_does_not_double_count() {
        let dir = tempfile::tempdir().unwrap();
        let cache = StagingCache::unbounded();
        let k = key("a", 100);

        let p1 = blob(dir.path(), "a1.gguf", 100);
        let p2 = blob(dir.path(), "a2.gguf", 250);
        cache.remember(k.clone(), p1);
        let evicted: Vec<PathBuf> = cache.remember(k.clone(), p2.clone());

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes(), 250);
        assert_eq!(cache.get(&k), Some(p2));
        let nothing: Vec<PathBuf> = vec![];
        assert_eq!(
            evicted, nothing,
            "same key, so the old path is replaced not evicted"
        );
    }

    #[test]
    fn forget_releases_the_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let cache = StagingCache::unbounded();
        let p = blob(dir.path(), "a.gguf", 700);
        let k = key("a", 700);

        cache.remember(k.clone(), p.clone());
        assert_eq!(cache.bytes(), 700);
        assert_eq!(cache.forget(&k), Some(p));
        assert_eq!(cache.bytes(), 0);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.forget(&k), None);
    }

    #[test]
    fn forgets_a_whole_archive() {
        let dir = tempfile::tempdir().unwrap();
        let cache = StagingCache::unbounded();

        let a = ArchiveId::new("/archives/a.asmb", 1000);
        let b = ArchiveId::new("/archives/b.asmb", 2000);

        let pa = blob(dir.path(), "a1.gguf", 100);
        let pb = blob(dir.path(), "b1.gguf", 200);
        let ka = StagedKey::new(a.clone(), 0, 100);
        let kb = StagedKey::new(b.clone(), 0, 200);

        cache.remember(ka.clone(), pa.clone());
        cache.remember(kb.clone(), pb.clone());
        assert_eq!(cache.bytes(), 300);

        let dropped = cache.forget_archive(&a);
        assert_eq!(dropped, vec![pa]);
        assert_eq!(cache.bytes(), 200);
        assert_eq!(cache.get(&ka), None);
        assert_eq!(cache.get(&kb), Some(pb), "the other archive is untouched");
    }

    #[test]
    fn clear_releases_everything() {
        let dir = tempfile::tempdir().unwrap();
        let cache = StagingCache::unbounded();
        for i in 0..3 {
            let p = blob(dir.path(), &format!("m{i}.gguf"), 100);
            cache.remember(key(&format!("m{i}"), 100), p);
        }
        assert_eq!(cache.clear().len(), 3);
        assert_eq!(cache.bytes(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn archive_identity_covers_length_and_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let path = blob(dir.path(), "a.asmb", 1234);

        let first = ArchiveId::from_path(&path).unwrap();
        assert_eq!(first.path(), path.as_path());

        // Same path, different length: a different archive.
        std::fs::write(&path, vec![0u8; 4321]).unwrap();
        let second = ArchiveId::from_path(&path).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn different_blobs_get_different_keys() {
        let archive = ArchiveId::new("/archives/a.asmb", 1000);
        assert_ne!(
            StagedKey::new(archive.clone(), 0, 100),
            StagedKey::new(archive.clone(), 4096, 100)
        );
        assert_ne!(
            StagedKey::new(archive.clone(), 0, 100),
            StagedKey::new(archive.clone(), 0, 200)
        );
        assert_ne!(
            StagedKey::new(archive, 0, 100),
            StagedKey::new(ArchiveId::new("/archives/b.asmb", 1000), 0, 100)
        );
    }
}
