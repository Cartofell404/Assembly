//! Which models stay resident in memory.
//!
//! A `.asmb` archive of three multi-gigabyte models does not fit in RAM on a
//! laptop, so not every model can be loaded at once. This tracks what is
//! loaded, what is using it, and what to give up when the budget is exceeded.
//!
//! # The two rules
//!
//! **Never evict a model in use.** A model that is mid-generation is holding
//! borrowed state inside llama.cpp. Unloading it would leave a dangling
//! pointer in a token stream another thread is still reading, so in-use models
//! are skipped entirely and eviction moves to the next candidate.
//!
//! **Staged blobs outlive their models.** Unloading a model frees its RAM but
//! not its staged copy on disk, so reloading later is a file open rather than
//! a multi-gigabyte copy. That is the whole reason staging exists, and this
//! type is careful never to delete a staged file as part of eviction.
//!
//! # Budget, not count
//!
//! The budget is in bytes, not model count. Three 500 MB models and three
//! 4 GB models need very different treatment, and a count limit cannot tell
//! them apart. A count-based cap of eight also fails the common case outright:
//! an archive with fewer than eight models never evicts anything, however much
//! RAM it needs.
//!
//! # Threading
//!
//! Interior mutability behind a mutex, so a [`ResidentGuard`] can hold a
//! shared handle while a caller keeps working. Guards are reference-counted per
//! model, so several requests can hold one model concurrently and it is
//! evicted only once every one of them has let go.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::staging::ConfigError;

/// How many models a machine-sized cache will hold regardless of bytes.
///
/// A floor against pathological archives, not a tuning knob: thirty loaded
/// models means thirty sets of weights, and at that point eviction churns
/// rather than helps.
pub const DEFAULT_MODEL_CAP: usize = 8;

/// Where a model sits in the eviction order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// Loaded and nobody is using it. `uses` counts acquisitions since load,
    /// which is what makes "least useful" a stable answer.
    Idle { uses: u64 },
    /// At least one caller holds it. `refs` is that count.
    Busy { refs: u32 },
}

/// One loaded model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resident {
    /// Model name.
    pub name: String,
    /// Bytes the model is expected to occupy. Callers pass the blob size; the
    /// true figure also includes the KV cache, so this is a lower bound.
    pub bytes: u64,
    /// Path of the staged blob backing it, when one exists.
    pub staged_path: Option<std::path::PathBuf>,
}

#[derive(Debug, Default)]
struct Inner {
    entries: HashMap<String, Slot>,
    resident: HashMap<String, Resident>,
    budget: Budget,
}

#[derive(Debug, Default, Clone, Copy)]
struct Budget {
    max_bytes: Option<u64>,
    max_models: Option<usize>,
}

/// Tracks loaded models and decides what to evict.
///
/// Cloneable and shareable: every clone refers to the same state, so a guard
/// taken from one clone is visible through the others.
#[derive(Debug, Clone)]
pub struct ResidentCache {
    inner: Arc<Mutex<Inner>>,
}

impl ResidentCache {
    /// A cache with no limit. Everything stays loaded.
    pub fn unbounded() -> ResidentCache {
        Self::with(Budget::default())
    }

    /// A cache bounded by resident bytes.
    pub fn with_byte_budget(max_bytes: u64) -> ResidentCache {
        Self::with(Budget {
            max_bytes: Some(max_bytes),
            max_models: None,
        })
    }

    /// A cache bounded by model count.
    ///
    /// Kept for parity with the Python default. Prefer
    /// [`ResidentCache::with_byte_budget`]: a count limit cannot express "16 GB
    /// of RAM" and will happily hold three 4 GB models on an 8 GB machine.
    pub fn with_model_limit(max_models: usize) -> ResidentCache {
        Self::with(Budget {
            max_bytes: None,
            max_models: Some(max_models),
        })
    }

    /// A cache bounded by both bytes and model count.
    ///
    /// Both limits apply at once, because they catch different mistakes. Bytes
    /// stop the machine running out of RAM, which a model count cannot: sizes
    /// span 0.5B to 70B and machines span 8 GB to 128 GB, so "three models"
    /// means 1.5 GB on one machine and 60 GB on another. The count cap is still
    /// worth having, because it stops an archive of thirty small models from
    /// thrashing -- each one cheap to hold, none of them enough to serve a
    /// request well.
    pub fn with_budget(max_bytes: u64, max_models: usize) -> ResidentCache {
        Self::with(Budget {
            max_bytes: Some(max_bytes),
            max_models: Some(max_models),
        })
    }

    /// A cache sized to the machine it is running on.
    ///
    /// `headroom` is the fraction of physical RAM to leave free, as a fraction:
    /// `0.8` keeps 80% for models and 20% for the OS, the page cache and the
    /// caller. Below about half the machine's RAM a model count is the better
    /// bound, because a single model larger than the budget would otherwise be
    /// loaded and immediately found over the limit.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] when `headroom` is not in `(0, 1]`, or when
    /// physical memory cannot be determined.
    pub fn sized_for_this_machine(headroom: f64) -> Result<ResidentCache, ConfigError> {
        if !(headroom > 0.0 && headroom <= 1.0) {
            return Err(ConfigError::BadHeadroom {
                requested: format!("{headroom}"),
            });
        }
        let total = physical_memory_bytes()?;
        let usable = (total as f64 * headroom) as u64;
        Ok(Self::with_budget(usable, DEFAULT_MODEL_CAP))
    }

    fn with(budget: Budget) -> ResidentCache {
        ResidentCache {
            inner: Arc::new(Mutex::new(Inner {
                budget,
                ..Inner::default()
            })),
        }
    }

    /// The byte limit, if one is set.
    pub fn byte_budget(&self) -> Option<u64> {
        self.lock().budget.max_bytes
    }

    /// The model-count limit, if one is set.
    pub fn model_limit(&self) -> Option<usize> {
        self.lock().budget.max_models
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Bytes currently held by loaded models.
    pub fn bytes(&self) -> u64 {
        self.lock().resident.values().map(|r| r.bytes).sum()
    }

    /// Number of loaded models.
    pub fn len(&self) -> usize {
        self.lock().resident.len()
    }

    /// Whether nothing is loaded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Models currently loaded, by name.
    pub fn loaded(&self) -> Vec<String> {
        let mut names: Vec<String> = self.lock().resident.keys().cloned().collect();
        names.sort();
        names
    }

    /// Names of loaded models that no caller is currently using.
    pub fn idle(&self) -> Vec<String> {
        let inner = self.lock();
        let mut names: Vec<String> = inner
            .entries
            .iter()
            .filter(|(_, slot)| matches!(slot, Slot::Idle { .. }))
            .filter(|(name, _)| inner.resident.contains_key(*name))
            .map(|(name, _)| name.clone())
            .collect();
        names.sort();
        names
    }

    /// Whether `name` is loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.lock().resident.contains_key(name)
    }

    /// Whether any caller is using `name`.
    pub fn is_busy(&self, name: &str) -> bool {
        matches!(self.lock().entries.get(name), Some(Slot::Busy { .. }))
    }

    /// How many callers are using `name`.
    pub fn users(&self, name: &str) -> u32 {
        match self.lock().entries.get(name) {
            Some(Slot::Busy { refs }) => *refs,
            _ => 0,
        }
    }

    /// The record for one loaded model.
    pub fn get(&self, name: &str) -> Option<Resident> {
        self.lock().resident.get(name).cloned()
    }

    /// Record that `name` is loaded.
    ///
    /// An already-loaded model is left alone: use
    /// [`ResidentCache::acquire`] to take a reference before use. Re-loading an
    /// existing name would reset its use count and could let a caller holding
    /// it be picked as an eviction victim.
    pub fn insert(&self, resident: Resident) {
        let mut inner = self.lock();
        if inner.resident.contains_key(&resident.name) {
            return;
        }
        inner
            .entries
            .insert(resident.name.clone(), Slot::Idle { uses: 0 });
        inner.resident.insert(resident.name.clone(), resident);
    }

    /// Record where a loaded model's blob is staged on disk.
    ///
    /// Separate from [`ResidentCache::insert`] because a model is recorded twice
    /// on its way in: once with a size only, so eviction can see it coming, and
    /// again after the blob is extracted. `insert` deliberately ignores an
    /// existing entry -- resetting it would clear the use count -- so without
    /// this the staged path would stay unset for the life of the entry, and the
    /// manager would re-extract a multi-gigabyte blob on every load.
    pub fn set_staged_path(&self, name: &str, path: PathBuf) {
        let mut inner = self.lock();
        if let Some(resident) = inner.resident.get_mut(name) {
            resident.staged_path = Some(path);
        }
    }

    /// Take a reference to a loaded model, marking it busy.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::NotLoaded`] when `name` is not resident, which
    /// would otherwise let a caller use a model that is about to be unloaded.
    pub fn acquire(&self, name: &str) -> Result<ResidentGuard, CacheError> {
        let mut inner = self.lock();
        if !inner.resident.contains_key(name) {
            return Err(CacheError::NotLoaded(name.to_string()));
        }

        let refs = match inner.entries.get(name).copied() {
            Some(Slot::Busy { refs }) => refs + 1,
            Some(Slot::Idle { uses }) => {
                inner
                    .entries
                    .insert(name.to_string(), Slot::Idle { uses: uses + 1 });
                1
            }
            None => unreachable!("checked resident above"),
        };
        inner.entries.insert(name.to_string(), Slot::Busy { refs });

        Ok(ResidentGuard {
            name: name.to_string(),
            cache: self.clone(),
        })
    }

    /// Release a reference taken by [`ResidentCache::acquire`].
    ///
    /// Prefer dropping the guard, which does this for you.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::NotBusy`] when the model was not busy, which
    /// means two references were released for one acquire.
    pub fn release(&self, name: &str) -> Result<(), CacheError> {
        let mut inner = self.lock();
        let refs = match inner.entries.get(name).copied() {
            Some(Slot::Busy { refs }) => refs,
            _ => return Err(CacheError::NotBusy(name.to_string())),
        };
        if refs > 1 {
            inner
                .entries
                .insert(name.to_string(), Slot::Busy { refs: refs - 1 });
        } else {
            inner
                .entries
                .insert(name.to_string(), Slot::Idle { uses: 0 });
        }
        Ok(())
    }

    /// Evict idle models until the cache is within its limit.
    ///
    /// Returns the models that should now be unloaded, least-useful first. The
    /// caller does the unloading, because only it knows how to release a model
    /// handle; this type just decides.
    ///
    /// In-use models are never returned. If the budget cannot be met because
    /// everything loaded is busy, the list comes back short of what was asked
    /// for, which is the correct outcome: the alternative is dropping a model
    /// out from under a running request.
    pub fn evict_to_fit(&self) -> Vec<Resident> {
        let mut victims = Vec::new();

        loop {
            let mut inner = self.lock();

            let over_bytes = inner
                .budget
                .max_bytes
                .is_some_and(|limit| inner.resident.values().map(|r| r.bytes).sum::<u64>() > limit);
            let over_count = inner
                .budget
                .max_models
                .is_some_and(|limit| inner.resident.len() > limit);
            if !over_bytes && !over_count {
                break;
            }

            match least_useful_idle(&inner) {
                Some(name) => {
                    inner.entries.remove(&name);
                    if let Some(r) = inner.resident.remove(&name) {
                        victims.push(r);
                    }
                }
                // Nothing idle left: everything still loaded is in use.
                None => break,
            }
        }

        victims
    }

    /// Forget a model entirely, whether loaded or not.
    ///
    /// # Panics
    ///
    /// Does not panic on a busy model, but doing this while a caller holds a
    /// guard leaves that guard pointing at a model the cache no longer tracks.
    /// Let the guards drop first.
    pub fn remove(&self, name: &str) -> Option<Resident> {
        let mut inner = self.lock();
        inner.entries.remove(name);
        inner.resident.remove(name)
    }

    /// Drop every record. For use when an archive closes.
    pub fn clear(&self) -> Vec<Resident> {
        let mut inner = self.lock();
        inner.entries.clear();
        inner.resident.drain().map(|(_, r)| r).collect()
    }
}

/// Physical RAM, in bytes.
///
/// # Errors
///
/// Returns [`ConfigError`] when the platform does not say. A server sized
/// by RAM is worth more than one sized by nothing, so callers that can fall
/// back to a count limit should.
#[cfg(target_os = "macos")]
pub fn physical_memory_bytes() -> Result<u64, ConfigError> {
    // sysctl hw.memsize, read directly rather than shelling out to
    // sysctl: this runs on the request path when a cache is first created.
    let mut size: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = b"hw.memsize\0";
    // SAFETY: `name` is a NUL-terminated C string, and `size`/`len` are a
    // correctly typed, aligned out-buffer and its length. `sysctlbyname` writes
    // at most `len` bytes and reports failure through its return value.
    let rc = unsafe {
        libc_sysctlbyname(
            name.as_ptr() as *const std::os::raw::c_char,
            &mut size as *mut u64 as *mut std::os::raw::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && size > 0 {
        return Ok(size);
    }

    Err(no_memory_error())
}

#[cfg(target_os = "macos")]
extern "C" {
    #[link_name = "sysctlbyname"]
    fn libc_sysctlbyname(
        name: *const std::os::raw::c_char,
        oldp: *mut std::os::raw::c_void,
        oldlenp: *mut usize,
        newp: *mut std::os::raw::c_void,
        newlen: usize,
    ) -> std::os::raw::c_int;
}

/// Physical RAM, in bytes.
///
/// # Errors
///
/// Returns [`ConfigError`] when the platform does not say.
#[cfg(target_os = "windows")]
pub fn physical_memory_bytes() -> Result<u64, ConfigError> {
    // `MEMORYSTATUSEX` at the time of writing. The length field must be set to
    // the size of this struct before the call; the API refuses to write if it
    // disagrees. Only `total_physical` is read, but every field stays so the
    // layout matches the C definition the API writes into.
    #[repr(C)]
    #[allow(dead_code)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_physical: u64,
        available_physical: u64,
        total_page_file: u64,
        available_page_file: u64,
        total_virtual: u64,
        available_virtual: u64,
        available_extended_virtual: u64,
    }

    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    // SAFETY: `status` is a correctly sized and aligned `MemoryStatusEx`, and its
    // `length` field is set to that size before the call, which is the contract
    // the API checks before writing. All-zero is a valid value for a struct of
    // integers. The single out-pointer is valid and writable.
    let mut status: MemoryStatusEx = unsafe { std::mem::zeroed() };
    status.length = std::mem::size_of::<MemoryStatusEx>() as u32;
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok != 0 && status.total_physical > 0 {
        return Ok(status.total_physical);
    }

    Err(no_memory_error())
}

/// The error returned when the platform will not report physical memory.
///
/// A shared constructor keeps the wording identical across platforms, which
/// matters because callers fall back to a model-count budget on this error.
fn no_memory_error() -> ConfigError {
    ConfigError::Other(
        "cannot determine physical memory; pass an explicit \
     resident budget instead"
            .to_string(),
    )
}

/// The idle model a caller would use least, if any.
///
/// Least-acquired first, then alphabetically. The second key matters because
/// `HashMap` iteration order is unspecified, so without it the victim would
/// vary between runs and a flaky eviction would be near impossible to
/// reproduce.
fn least_useful_idle(inner: &Inner) -> Option<String> {
    let mut best: Option<(u64, &str)> = None;

    for (name, slot) in &inner.entries {
        let Slot::Idle { uses } = slot else { continue };
        if !inner.resident.contains_key(name) {
            continue;
        }
        match best {
            Some((best_uses, best_name)) => {
                if *uses < best_uses || (*uses == best_uses && name.as_str() < best_name) {
                    best = Some((*uses, name.as_str()));
                }
            }
            None => best = Some((*uses, name.as_str())),
        }
    }

    best.map(|(_, name)| name.to_string())
}

/// Keeps a model busy for as long as it is alive.
///
/// Dropping the guard releases the reference, including on an early return or
/// a panic, so a caller cannot leak a busy flag by forgetting to release.
#[derive(Debug)]
pub struct ResidentGuard {
    name: String,
    cache: ResidentCache,
}

impl ResidentGuard {
    /// The model being held.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// How many callers, this one included, are using the model.
    pub fn shared(&self) -> u32 {
        self.cache.users(&self.name)
    }
}

impl Drop for ResidentGuard {
    fn drop(&mut self) {
        // Swallow the error: a drop must not panic, and the only way this fails
        // is a double release, which is already a caller bug.
        let _ = self.cache.release(&self.name);
    }
}

/// Mistakes callers can make against a [`ResidentCache`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CacheError {
    /// A model was used that is not loaded.
    #[error("model '{0}' is not loaded")]
    NotLoaded(String),
    /// More references were released than taken.
    #[error("model '{0}' was not in use")]
    NotBusy(String),
}

/// Blob sizes from a parsed header, sorted by name, for seeding a
/// [`ResidentCache`].
pub fn sizes_from_header(header: &asmb_format::AsmbHeader) -> Vec<(String, u64)> {
    let mut out: Vec<(String, u64)> = header
        .models
        .iter()
        .map(|m| (m.name.clone(), m.size))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn resident(name: &str, bytes: u64) -> Resident {
        Resident {
            name: name.to_string(),
            bytes,
            staged_path: Some(PathBuf::from(format!("/tmp/{name}.gguf"))),
        }
    }

    #[test]
    fn starts_empty() {
        let cache = ResidentCache::unbounded();
        assert!(cache.is_empty());
        assert_eq!(cache.bytes(), 0);
        assert!(cache.loaded().is_empty());
        assert!(cache.idle().is_empty());
        assert_eq!(cache.byte_budget(), None);
        assert_eq!(cache.model_limit(), None);
    }

    #[test]
    fn tracks_loaded_models() {
        let cache = ResidentCache::unbounded();
        cache.insert(resident("a", 100));
        cache.insert(resident("b", 200));

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.bytes(), 300);
        assert_eq!(cache.loaded(), vec!["a".to_string(), "b".to_string()]);
        assert!(cache.is_loaded("a"));
        assert!(!cache.is_loaded("ghost"));
        assert_eq!(cache.get("b").map(|r| r.bytes), Some(200));
    }

    #[test]
    fn re_inserting_does_not_duplicate_or_recount() {
        let cache = ResidentCache::unbounded();
        cache.insert(resident("a", 100));

        let guard = cache.acquire("a").unwrap();
        cache.insert(resident("a", 100));

        assert!(
            cache.is_busy("a"),
            "a second load must not clear the busy flag"
        );
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes(), 100, "and must not double-count");
        drop(guard);
    }

    #[test]
    fn acquires_and_releases() {
        let cache = ResidentCache::unbounded();
        cache.insert(resident("a", 100));

        assert!(!cache.is_busy("a"));
        {
            let guard = cache.acquire("a").unwrap();
            assert!(cache.is_busy("a"));
            assert_eq!(guard.name(), "a");
            assert_eq!(guard.shared(), 1);
        }
        assert!(!cache.is_busy("a"), "the guard releases on drop");
    }

    #[test]
    fn acquiring_an_unloaded_model_is_an_error() {
        let cache = ResidentCache::unbounded();
        let err = cache.acquire("ghost").unwrap_err();
        assert_eq!(err, CacheError::NotLoaded("ghost".into()));
        assert_eq!(err.to_string(), "model 'ghost' is not loaded");
    }

    #[test]
    fn releasing_without_acquiring_is_an_error() {
        let cache = ResidentCache::unbounded();
        cache.insert(resident("a", 100));
        let err = cache.release("a").unwrap_err();
        assert_eq!(err, CacheError::NotBusy("a".into()));
        assert_eq!(err.to_string(), "model 'a' was not in use");
    }

    #[test]
    fn two_references_need_two_releases() {
        let cache = ResidentCache::unbounded();
        cache.insert(resident("a", 100));

        let first = cache.acquire("a").unwrap();
        let second = cache.acquire("a").unwrap();
        assert!(cache.is_busy("a"));
        assert_eq!(cache.users("a"), 2);

        drop(second);
        assert!(cache.is_busy("a"), "one reference remains");
        drop(first);
        assert!(!cache.is_busy("a"));
        assert_eq!(cache.users("a"), 0);
    }

    #[test]
    fn a_byte_budget_evicts_the_least_used() {
        let cache = ResidentCache::with_byte_budget(300);
        cache.insert(resident("a", 200));
        cache.insert(resident("b", 200));

        let victims = cache.evict_to_fit();
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0].name, "a");
        assert!(cache.bytes() <= 300, "left {}", cache.bytes());
    }

    #[test]
    fn the_more_used_model_survives() {
        let cache = ResidentCache::with_byte_budget(300);
        cache.insert(resident("a", 200));
        cache.insert(resident("b", 200));

        drop(cache.acquire("b").unwrap());
        drop(cache.acquire("b").unwrap());

        let victims = cache.evict_to_fit();
        assert_eq!(victims[0].name, "a", "the unused model is the one to drop");
        assert!(cache.is_loaded("b"));
    }

    #[test]
    fn an_in_use_model_is_never_evicted() {
        let cache = ResidentCache::with_byte_budget(100);
        cache.insert(resident("a", 200));
        cache.insert(resident("b", 200));

        let guard = cache.acquire("a").unwrap();
        let victims = cache.evict_to_fit();

        assert!(cache.is_loaded("a"), "a is mid-request");
        assert_eq!(
            victims.iter().map(|v| &v.name).collect::<Vec<_>>(),
            vec!["b"]
        );

        drop(guard);
        let victims = cache.evict_to_fit();
        assert_eq!(victims[0].name, "a");
    }

    #[test]
    fn an_all_busy_cache_evicts_nothing() {
        let cache = ResidentCache::with_byte_budget(10);
        cache.insert(resident("a", 200));
        cache.insert(resident("b", 200));

        let guard_a = cache.acquire("a").unwrap();
        let guard_b = cache.acquire("b").unwrap();

        assert!(
            cache.evict_to_fit().is_empty(),
            "over budget beats dropping a model under a running request"
        );
        assert_eq!(cache.bytes(), 400);
        assert_eq!(cache.len(), 2);

        drop(guard_a);
        drop(guard_b);
        assert_eq!(cache.evict_to_fit().len(), 2);
    }

    #[test]
    fn a_count_limit_evicts_down_to_it() {
        let cache = ResidentCache::with_model_limit(2);
        for name in ["a", "b", "c", "d"] {
            cache.insert(resident(name, 1));
        }
        assert_eq!(cache.len(), 4);

        let victims = cache.evict_to_fit();
        assert_eq!(victims.len(), 2);
        assert_eq!(cache.len(), 2);
        assert!(cache.is_loaded("c") && cache.is_loaded("d"));
        assert_eq!(cache.model_limit(), Some(2));
    }

    #[test]
    fn a_count_limit_still_never_touches_a_busy_model() {
        let cache = ResidentCache::with_model_limit(1);
        cache.insert(resident("a", 1));
        let guard = cache.acquire("a").unwrap();
        cache.insert(resident("b", 1));

        let victims = cache.evict_to_fit();
        assert_eq!(victims[0].name, "b", "the idle one goes, not the busy one");
        assert!(cache.is_loaded("a"));
        drop(guard);
    }

    #[test]
    fn eviction_is_a_no_op_when_within_budget() {
        let cache = ResidentCache::with_byte_budget(1000);
        cache.insert(resident("a", 100));
        cache.insert(resident("b", 100));
        assert!(cache.evict_to_fit().is_empty());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn eviction_ties_break_deterministically() {
        // Both used once. The alphabetically first must always lose, or the
        // same config evicts differently on different runs.
        for _ in 0..20 {
            let cache = ResidentCache::with_byte_budget(100);
            cache.insert(resident("b", 200));
            cache.insert(resident("a", 200));
            drop(cache.acquire("a").unwrap());
            drop(cache.acquire("b").unwrap());

            let victims = cache.evict_to_fit();
            assert_eq!(victims[0].name, "a");
        }
    }

    #[test]
    fn staged_paths_survive_eviction() {
        let cache = ResidentCache::with_byte_budget(100);
        cache.insert(resident("a", 200));

        let victims = cache.evict_to_fit();
        assert_eq!(
            victims[0].staged_path,
            Some(PathBuf::from("/tmp/a.gguf")),
            "the caller keeps the staged copy; only the resident handle goes"
        );
    }

    #[test]
    fn idle_lists_only_loaded_models() {
        let cache = ResidentCache::unbounded();
        cache.insert(resident("a", 1));
        cache.insert(resident("b", 1));
        let guard = cache.acquire("b").unwrap();

        assert_eq!(cache.idle(), vec!["a".to_string()]);
        drop(guard);
        assert_eq!(cache.idle(), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn remove_and_clear() {
        let cache = ResidentCache::unbounded();
        cache.insert(resident("a", 100));
        cache.insert(resident("b", 100));

        assert_eq!(cache.remove("a").map(|r| r.name), Some("a".to_string()));
        assert!(!cache.is_loaded("a"));
        assert_eq!(cache.bytes(), 100);

        let dropped = cache.clear();
        assert_eq!(dropped.len(), 1);
        assert!(cache.is_empty());
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn a_guard_releases_even_when_early_returned() {
        let cache = ResidentCache::unbounded();
        cache.insert(resident("a", 100));

        fn early(cache: &ResidentCache) {
            let _guard = cache.acquire("a").unwrap();
        }
        early(&cache);
        assert!(!cache.is_busy("a"), "a dropped guard must release");
    }

    #[test]
    fn a_guarded_model_stays_busy_until_every_reference_is_gone() {
        let cache = ResidentCache::unbounded();
        cache.insert(resident("a", 100));

        let outer = cache.acquire("a").unwrap();
        {
            let inner = cache.acquire("a").unwrap();
            assert_eq!(inner.name(), "a");
            assert_eq!(inner.shared(), 2);
        }
        assert!(cache.is_busy("a"), "the outer reference is still held");
        drop(outer);
        assert!(!cache.is_busy("a"));
    }

    #[test]
    fn a_clone_shares_state() {
        let cache = ResidentCache::unbounded();
        let other = cache.clone();
        cache.insert(resident("a", 100));

        assert!(other.is_loaded("a"));
        let guard = other.acquire("a").unwrap();
        assert!(cache.is_busy("a"), "visible through the original");
        drop(guard);
    }

    #[test]
    fn sizes_from_the_header_come_back_sorted() {
        let header = asmb_format::AsmbHeader {
            assembly_name: "T".into(),
            version: 1,
            page_size: 4096,
            embedding: asmb_format::EmbeddingEntry::empty(),
            models: vec![
                asmb_format::ModelEntry::with_vectors("z", &[vec![1.0]], 0, 300, vec![], "")
                    .unwrap(),
                asmb_format::ModelEntry::with_vectors("a", &[vec![1.0]], 0, 100, vec![], "")
                    .unwrap(),
            ],
            persona: None,
        };
        assert_eq!(
            sizes_from_header(&header),
            vec![("a".to_string(), 100), ("z".to_string(), 300)]
        );
    }

    // ── machine-sized budgets ───────────────────────────────────────────

    #[test]
    fn a_machine_sized_cache_keeps_both_limits() {
        // Both apply. Bytes alone cannot stop thirty small models thrashing,
        // and a count alone cannot stop three 4 GB models on an 8 GB machine.
        let cache = ResidentCache::with_budget(1000, 2);
        assert_eq!(cache.byte_budget(), Some(1000));
        assert_eq!(cache.model_limit(), Some(2));
    }

    #[test]
    fn both_limits_evict() {
        let cache = ResidentCache::with_budget(1000, 2);
        for (name, bytes) in [("a", 400), ("b", 400), ("c", 400)] {
            cache.insert(Resident {
                name: name.to_string(),
                bytes,
                staged_path: None,
            });
            // The guard releases on drop, so the model is idle again by the
            // time eviction runs. Calling release() as well would be a second
            // release of the same reference.
            drop(cache.acquire(name).unwrap());
        }
        cache.evict_to_fit();

        // Three models at 1200 bytes breaks both limits, so one goes and the
        // remaining two fit either way.
        assert_eq!(cache.len(), 2, "one model should have been evicted");
    }

    #[test]
    fn headroom_outside_zero_to_one_is_rejected() {
        for bad in [0.0, -0.5, 1.5, f64::NAN] {
            let err = ResidentCache::sized_for_this_machine(bad).unwrap_err();
            assert!(
                matches!(err, ConfigError::BadHeadroom { .. }),
                "headroom {bad} was accepted"
            );
        }
    }

    #[test]
    fn physical_memory_is_reported_when_the_platform_knows() {
        // Every supported platform knows the answer -- sysctl on macOS,
        // GlobalMemoryStatusEx on Windows -- so this is a real check rather than
        // an assumption: it must come back plausible, not an error.
        let bytes = physical_memory_bytes().expect("the platform must report physical memory");
        // Below 64 MB is not a machine; above 1 TiB is not either.
        assert!(bytes > 64 * 1024 * 1024, "implausible: {bytes} bytes");
        assert!(bytes < 1024 * 1024 * 1024 * 1024, "implausible: {bytes}");
    }

    #[test]
    fn a_machine_sized_cache_uses_the_requested_fraction() {
        if let Ok(total) = physical_memory_bytes() {
            let cache = ResidentCache::sized_for_this_machine(0.5).unwrap();
            let expected = (total as f64 * 0.5) as u64;
            assert_eq!(cache.byte_budget(), Some(expected));
            assert_eq!(cache.model_limit(), Some(DEFAULT_MODEL_CAP));
        }
    }
}
