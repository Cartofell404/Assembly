//! The assembly-level manager: one archive, many models, one route per prompt.
//!
//! This is the Rust counterpart of Python's `AssemblyLlama`. It owns the pieces
//! the other crates provide separately and composes them into the flow a user
//! actually wants:
//!
//! 1. open an archive and build a [`Router`] from its anchors,
//! 2. stage a blob out of the archive, once, and remember it,
//! 3. route a prompt embedding to a model,
//! 4. make sure that model is resident in RAM, evicting idle ones if needed,
//! 5. generate.
//!
//! What it deliberately does *not* do is inference itself. Inference goes
//! through an [`Engine`], which is what makes this type testable without a GGUF
//! file: the tests use an engine that returns canned vectors and text.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use asmb_format::{AsmbHeader, EmbeddingEntry, ModelEntry};
use thiserror::Error;

use crate::engine::{
    Chunk, Engine, EngineOptions, LoadOptions, Message, Messages, ModelHandle, Role,
};
use crate::resident::{Resident, ResidentCache, ResidentGuard, DEFAULT_MODEL_CAP};
use crate::router::{RouteResult, Router};
use crate::staging::{ArchiveId, StagedKey, StagingCache};

/// Default context window when the archive does not imply one.
pub const FALLBACK_CONTEXT: u32 = 4096;

/// The persona used when the archive sets none.
///
/// Mirrors the Python default: an assembly is several models behind one voice,
/// and without this they answer as themselves and contradict each other.
pub fn default_persona(assembly_name: &str) -> String {
    format!(
        "You are {assembly_name}, a unified AI assistant. \
         Always identify yourself as {assembly_name}. \
         Never state or imply that you are a different model (such as Qwen, \
         Llama, Mistral, SmolLM, etc.), and never mention that you are composed \
         of multiple separate models. Always respond consistently under this \
         identity."
    )
}

/// Why a manager operation failed.
#[derive(Debug, Error)]
pub enum ManagerError {
    /// The archive could not be opened, parsed or verified.
    #[error("cannot open archive {path}: {reason}")]
    Archive {
        /// The archive path.
        path: PathBuf,
        /// What was wrong with it.
        reason: String,
    },

    /// A blob could not be staged out of the archive.
    #[error("cannot stage model '{name}' from the archive: {reason}")]
    Stage {
        /// The model whose blob could not be staged.
        name: String,
        /// What went wrong.
        reason: String,
    },

    /// The archive has no router model, so nothing can be routed.
    #[error(
        "this archive has no router model, so no prompt can be routed; \
             rebuild it with an embedding model"
    )]
    NoRouter,

    /// The archive stores anchor phrases but no vectors, which is what
    /// `--no-embed` produces.
    #[error(
        "this archive stores anchor phrases but no anchor vectors, so nothing \
         can be routed against it; rebuild without --no-embed"
    )]
    NoAnchorVectors,

    /// The router's own model could not be loaded.
    #[error("cannot load the router model: {0}")]
    Router(String),

    /// An execution model could not be loaded.
    #[error("cannot load model '{name}': {reason}")]
    Model {
        /// The model that would not load.
        name: String,
        /// What the runtime said.
        reason: String,
    },

    /// Embedding a prompt failed.
    #[error("cannot embed the prompt: {0}")]
    Embed(String),

    /// Generation failed.
    #[error("generation failed: {0}")]
    Generate(String),
}

impl From<crate::engine::EngineError> for ManagerError {
    fn from(err: crate::engine::EngineError) -> Self {
        ManagerError::Generate(err.to_string())
    }
}

/// One archive, opened and ready to route.
///
/// Cloning is cheap and shares the caches, so a server can hand the same
/// manager to every request thread.
#[derive(Clone)]
pub struct Assembly {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Assembly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Assembly")
            .field("name", &self.name())
            .field("path", &self.path())
            .field("routable", &self.inner.routable)
            .field("loaded", &self.loaded_models())
            .finish_non_exhaustive()
    }
}

struct Inner {
    path: PathBuf,
    archive_id: ArchiveId,
    header: AsmbHeader,
    router: Router,
    persona: String,
    engine: Arc<dyn Engine>,
    staging: StagingCache,
    resident: ResidentCache,
    /// Runtime objects for loaded models, by name.
    ///
    /// Behind a mutex because a server shares one manager across threads, and
    /// two requests can route to different models at the same moment. Handles
    /// are `Arc` so a caller can clone one out and drop the lock before doing
    /// real work: generation takes seconds, and holding the table locked for
    /// that long would serialise every other request.
    ///
    /// The resident cache tracks *bytes*; this tracks the actual handles.
    loaded: Mutex<HashMap<String, Arc<dyn ModelHandle>>>,
    /// The router model, loaded on first use.
    router_handle: Mutex<Option<Arc<dyn ModelHandle>>>,
    /// Held while a model is being staged and loaded.
    ///
    /// Loading takes seconds and touches gigabytes. Without this, two requests
    /// routing to the same cold model would each load their own copy, briefly
    /// doubling the memory the budget exists to cap. Generation runs *after*
    /// this lock is released, so requests still overlap.
    load_lock: Mutex<()>,
    /// Execution models that have anchors and can therefore be chosen.
    routable: Vec<String>,
}

impl Assembly {
    /// Open `path` and prepare to route.
    ///
    /// `headroom` is the fraction of physical RAM to leave to the models, as a
    /// fraction: `0.8` leaves 20% for the OS and the page cache.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::Archive`] if the archive is missing, truncated,
    /// or fails layout validation.
    pub fn open(
        path: impl AsRef<Path>,
        engine: Arc<dyn Engine>,
        headroom: f64,
    ) -> Result<Assembly, ManagerError> {
        let path = path.as_ref().to_path_buf();
        let archive =
            asmb_format::read::Archive::open(&path).map_err(|e| ManagerError::Archive {
                path: path.clone(),
                reason: e.to_string(),
            })?;

        let total = archive.len();
        let header = archive.header().clone();
        let archive_id = ArchiveId::new(&path, total);

        if header.embedding.name.is_empty() {
            return Err(ManagerError::NoRouter);
        }
        // An archive with phrases but no vectors is valid and readable; it just
        // cannot route. Saying so here is better than returning the first model
        // for every prompt, which looks like it works and is silently wrong.
        if !header.models.iter().any(|m| m.anchor_dim > 0) {
            return Err(ManagerError::NoAnchorVectors);
        }

        let router = Router::new(&header).map_err(|e| ManagerError::Archive {
            path: path.clone(),
            reason: e.to_string(),
        })?;
        let routable = router
            .routable_models()
            .into_iter()
            .map(str::to_string)
            .collect();
        let persona = header
            .persona
            .clone()
            .unwrap_or_else(|| default_persona(&header.assembly_name));

        Ok(Assembly {
            inner: Arc::new(Inner {
                staging: StagingCache::with_limit(8 * 1024 * 1024 * 1024)
                    .unwrap_or_else(|_| StagingCache::unbounded()),
                resident: ResidentCache::sized_for_this_machine(headroom)
                    .unwrap_or_else(|_| ResidentCache::with_model_limit(DEFAULT_MODEL_CAP)),
                path,
                archive_id,
                header,
                router,
                persona,
                engine,
                loaded: Mutex::new(HashMap::new()),
                router_handle: Mutex::new(None),
                load_lock: Mutex::new(()),
                routable,
            }),
        })
    }

    /// The archive name.
    pub fn name(&self) -> &str {
        &self.inner.header.assembly_name
    }

    /// The system prompt prepended to every conversation.
    pub fn persona(&self) -> &str {
        &self.inner.persona
    }

    /// The archive path.
    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    /// Models that have anchors and can therefore be chosen.
    pub fn routable_models(&self) -> &[String] {
        &self.inner.routable
    }

    /// Models currently held in RAM.
    pub fn loaded_models(&self) -> Vec<String> {
        self.inner.resident.loaded()
    }

    /// Bytes held by models currently in RAM.
    pub fn loaded_bytes(&self) -> u64 {
        self.inner.resident.bytes()
    }

    /// Staged blob bytes on disk, for diagnostics.
    pub fn staged_bytes(&self) -> u64 {
        self.inner.staging.bytes()
    }

    /// Where a model's blob is staged, if it has been.
    pub fn staged_path(&self, name: &str) -> Option<PathBuf> {
        self.inner
            .resident
            .get(name)
            .and_then(|r| r.staged_path.clone())
    }

    /// The header, for `info`-style responses.
    pub fn header(&self) -> &AsmbHeader {
        &self.inner.header
    }

    /// Load the router model, if it is not already open.
    ///
    /// Separate from `open` so a caller that only wants to inspect the archive
    /// never pays for a 500 MB embedding model it will not use.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::Stage`] if the router blob cannot be staged out
    /// of the archive, and [`ManagerError::Router`] if it then fails to load.
    pub fn ensure_router(&self) -> Result<(), ManagerError> {
        let _loading = self.inner.load_lock.lock().expect("load lock poisoned");
        if self
            .router_slot()
            .lock()
            .expect("router handle poisoned")
            .is_some()
        {
            return Ok(());
        }

        let entry = &self.inner.header.embedding;

        // Record the size first so eviction can see this model coming, then pin
        // the router for the duration so a load in another thread cannot evict
        // the handle we are about to be using.
        self.inner.resident.insert(Resident {
            name: entry.name.clone(),
            bytes: entry.size,
            staged_path: None,
        });
        let pin = self.inner.resident.acquire(&entry.name).ok();
        for victim in self.inner.resident.evict_to_fit() {
            if victim.name != entry.name {
                self.unload(&victim.name);
            }
        }

        let staged = self.stage(&entry.name, entry.offset, entry.size)?;
        // 512 is what the build used to compute anchors, and what the Python
        // manager uses. A prompt is one short string.
        //
        // As with execution models, a failure clears the placeholder entry, so a
        // later attempt retries the load instead of trusting a phantom.
        let handle = match self.inner.engine.load(
            &staged,
            &LoadOptions::embed()
                .n_ctx(512)
                .n_gpu_layers(0)
                .named(&entry.name),
        ) {
            Ok(handle) => handle,
            Err(e) => {
                self.inner.resident.remove(&entry.name);
                return Err(ManagerError::Router(e.to_string()));
            }
        };

        self.inner.resident.set_staged_path(&entry.name, staged);
        *self.router_slot().lock().expect("router handle poisoned") = Some(Arc::from(handle));
        drop(pin);
        Ok(())
    }

    /// The router handle, if it is loaded.
    ///
    /// Returns a guard rather than a `&dyn ModelHandle` because the handle lives
    /// behind a mutex and the borrow must not outlive it.
    fn with_router<R>(&self, f: impl FnOnce(&dyn ModelHandle) -> R) -> Result<R, ManagerError> {
        self.ensure_router()?;
        let slot = self.router_slot();
        let guard = slot.lock().expect("router handle poisoned");
        let handle = guard.as_deref().expect("ensure_router loaded one");
        Ok(f(handle))
    }

    fn router_slot(&self) -> &Mutex<Option<Arc<dyn ModelHandle>>> {
        &self.inner.router_handle
    }

    /// Stage one blob out of the archive, reusing an existing copy if possible.
    fn stage(&self, name: &str, offset: u64, size: u64) -> Result<PathBuf, ManagerError> {
        let key = StagedKey::new(self.inner.archive_id.clone(), offset, size);
        if let Some(existing) = self.inner.staging.get(&key) {
            if existing.exists() {
                return Ok(existing);
            }
            // Recorded but gone: something cleaned the temp dir underneath us.
            // Forget it so the next stage actually writes a file.
            self.inner.staging.forget(&key);
        }

        let mut archive = asmb_format::read::Archive::open(&self.inner.path).map_err(|e| {
            ManagerError::Stage {
                name: name.to_string(),
                reason: e.to_string(),
            }
        })?;

        let dest = self
            .inner
            .staging
            .allocate(name, size)
            .map_err(|e| ManagerError::Stage {
                name: name.to_string(),
                reason: e.to_string(),
            })?;

        archive
            .extract_to(&self.header_lookup(name)?, &dest)
            .map_err(|e| ManagerError::Stage {
                name: name.to_string(),
                reason: e.to_string(),
            })?;

        // Remember it before returning: a later eviction of the *model* must
        // leave this file alone, which is the whole point of staging.
        self.inner.staging.remember(key, dest.clone());
        Ok(dest)
    }

    fn header_lookup(&self, name: &str) -> Result<String, ManagerError> {
        if self.inner.header.embedding.name == name {
            return Ok(self.inner.header.embedding.name.clone());
        }
        self.inner
            .header
            .models
            .iter()
            .find(|m| m.name == name)
            .map(|m| m.name.clone())
            .ok_or_else(|| ManagerError::Stage {
                name: name.to_string(),
                reason: "no such model in this archive".to_string(),
            })
    }

    /// Embed `text` with the router model.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::Router`] if the router model cannot be loaded,
    /// and [`ManagerError::Embed`] if the text cannot be embedded.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, ManagerError> {
        self.with_router(|handle| {
            self.inner
                .engine
                .embed(handle, text)
                .map_err(|e| ManagerError::Embed(e.to_string()))
        })?
    }

    /// Pick a model for `text` without loading anything.
    ///
    /// Does need the router model, since routing is a comparison against
    /// embeddings of the prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if the router model cannot be loaded or the prompt
    /// cannot be embedded.
    pub fn route(&self, text: &str) -> Result<RouteResult, ManagerError> {
        let vector = self.embed(text)?;
        Ok(self.inner.router.route(&vector))
    }

    /// Make sure `name` is loaded, evicting idle models if it does not fit.
    ///
    /// Returns a guard that must stay alive for as long as the model is in use.
    /// Dropping it makes the model evictable again.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::Model`] if the blob cannot be staged or loaded.
    pub fn ensure_model(&self, name: &str) -> Result<ResidentGuard, ManagerError> {
        // Fast path, keyed on the handle table rather than the resident cache.
        //
        // A load records the model's size in the cache *before* it loads, so
        // eviction can make room. That placeholder makes the model look resident
        // to another thread for the whole of the load, and a guard taken from it
        // would be a ticket for a model with no handle behind it. The handle
        // table is the only thing that says a model is genuinely usable.
        if self.is_loaded(name) {
            if let Ok(guard) = self.inner.resident.acquire(name) {
                return Ok(guard);
            }
        }

        // Serialise the load. Re-checked inside, so two threads racing for a
        // cold model produce one copy rather than two -- briefly doubling the
        // memory the resident budget exists to cap.
        let _loading = self.inner.load_lock.lock().expect("load lock poisoned");
        if self.is_loaded(name) {
            if let Ok(guard) = self.inner.resident.acquire(name) {
                return Ok(guard);
            }
        }

        let entry = self
            .inner
            .header
            .models
            .iter()
            .find(|m| m.name == name)
            .ok_or_else(|| ManagerError::Model {
                name: name.to_string(),
                reason: "no such model in this archive".to_string(),
            })?;

        // Record the size *before* loading, so eviction can already see this
        // model coming and free room for it. A model larger than the whole
        // budget is still loaded: refusing would mean the user cannot use the
        // machine they bought, and one oversize model is better than none.
        self.inner.resident.insert(Resident {
            name: entry.name.clone(),
            bytes: entry.size,
            staged_path: None,
        });

        // Evict before loading, not after: loading first would peak at the old
        // set plus the new model, which is exactly the OOM the budget prevents.
        //
        // The router is pinned across this: it is not in use by *this* request
        // (routing already finished), so eviction would otherwise be free to
        // pick it and leave a live handle nothing is tracking.
        let router_pin = self
            .inner
            .resident
            .acquire(&self.inner.header.embedding.name)
            .ok();
        for victim in self.inner.resident.evict_to_fit() {
            if victim.name != entry.name {
                self.unload(&victim.name);
            }
        }
        drop(router_pin);

        let staged = self.stage(&entry.name, entry.offset, entry.size)?;
        // A load that fails must leave nothing behind. The entry recorded above
        // exists so eviction could make room; if it survived a failure, the next
        // request would find the model "already resident" and hand a name to a
        // runtime that never loaded it.
        let handle = match self
            .inner
            .engine
            .load(&staged, &LoadOptions::generate().named(&entry.name))
        {
            Ok(handle) => handle,
            Err(e) => {
                self.inner.resident.remove(&entry.name);
                return Err(ManagerError::Model {
                    name: entry.name.clone(),
                    reason: e.to_string(),
                });
            }
        };

        self.loaded()
            .lock()
            .expect("loaded handles poisoned")
            .insert(entry.name.clone(), Arc::from(handle));
        self.inner.resident.set_staged_path(&entry.name, staged);

        self.inner
            .resident
            .acquire(&entry.name)
            .map_err(|e| ManagerError::Model {
                name: entry.name.clone(),
                reason: e.to_string(),
            })
    }

    /// The loaded-handle map.
    fn loaded(&self) -> &Mutex<HashMap<String, Arc<dyn ModelHandle>>> {
        &self.inner.loaded
    }

    /// Whether a model has a real handle behind it.
    fn is_loaded(&self, name: &str) -> bool {
        self.loaded()
            .lock()
            .expect("loaded handles poisoned")
            .contains_key(name)
    }

    /// A clone of a loaded handle, or an error naming the model.
    ///
    /// Cloning an `Arc` is a refcount bump, so the caller can hold the handle
    /// for as long as it likes -- even across an eviction -- without keeping the
    /// table locked.
    fn handle_for(&self, name: &str) -> Result<Arc<dyn ModelHandle>, ManagerError> {
        self.loaded()
            .lock()
            .expect("loaded handles poisoned")
            .get(name)
            .cloned()
            .ok_or_else(|| ManagerError::Model {
                name: name.to_string(),
                reason: "loaded but not tracked".to_string(),
            })
    }

    /// Drop a model from RAM, leaving its staged blob on disk.
    fn unload(&self, name: &str) {
        let handle = self
            .loaded()
            .lock()
            .expect("loaded handles poisoned")
            .remove(name);
        if let Some(handle) = handle {
            // Only the last owner releases it, so an in-flight request that
            // cloned the handle keeps a valid model until it finishes.
            self.inner.engine.release(handle);
        }
        self.inner.resident.remove(name);
    }

    /// Answer one prompt, routing first.
    ///
    /// This is the whole point of the type: embed, route, load, generate.
    ///
    /// # Errors
    ///
    /// Returns an error if routing fails, the chosen model cannot be loaded, or
    /// generation fails.
    pub fn ask(&self, text: &str, options: &EngineOptions) -> Result<ChatResponse, ManagerError> {
        let messages = Messages::from_prompt(text).with_persona(Some(&self.inner.persona));
        self.chat(&messages, options)
    }

    /// Answer a full conversation, routing on its last user turn.
    ///
    /// # Errors
    ///
    /// As [`Assembly::ask`].
    pub fn chat(
        &self,
        messages: &Messages,
        options: &EngineOptions,
    ) -> Result<ChatResponse, ManagerError> {
        let route = self.route_for(messages)?;
        let _guard = self.ensure_model(&route.model_name)?;

        let handle = self.handle_for(&route.model_name)?;

        let text = self
            .inner
            .engine
            .chat(handle.as_ref(), messages, options)
            .map_err(|e| ManagerError::Generate(e.to_string()))?;

        Ok(ChatResponse { text, route })
    }

    /// Route on a conversation's last user message.
    fn route_for(&self, messages: &Messages) -> Result<RouteResult, ManagerError> {
        let last = messages
            .items
            .iter()
            .rev()
            .find(|m| matches!(m.role, Role::User))
            .map(|m| m.content.clone())
            .unwrap_or_default();

        let vector = self.embed(&last)?;
        Ok(self.inner.router.route(&vector))
    }

    /// Answer a prompt, handing each piece of text to `on_chunk` as it arrives.
    ///
    /// Callback rather than an iterator on purpose. An iterator would have to
    /// borrow the model handle for as long as the stream lives, which means
    /// either a self-referential struct or holding the handle lock for the whole
    /// stream -- and the lock blocks every other request that touches the same
    /// manager. Here the borrow ends when the call does, and the resident guard
    /// still keeps the model loaded until the last chunk.
    ///
    /// Returning `Err` from `on_chunk` stops generation early. That is how the
    /// server aborts a stream when the client disconnects, without waiting for a
    /// model to finish a sentence nobody is listening to.
    ///
    /// # Errors
    ///
    /// As [`Assembly::ask`]. An error from `on_chunk` is not reported as an
    /// error: the answer was simply cut short, which is a normal outcome.
    pub fn ask_stream(
        &self,
        text: &str,
        options: &EngineOptions,
        on_chunk: &mut dyn FnMut(Chunk) -> Result<(), ()>,
    ) -> Result<RouteResult, ManagerError> {
        let messages = Messages::from_prompt(text);
        self.chat_stream(&messages, options, on_chunk)
    }

    /// Answer a full conversation, handing each piece of text to `on_chunk` as
    /// it arrives.
    ///
    /// Same contract as [`Assembly::ask_stream`] for the callback and its
    /// errors; the difference is that the whole conversation reaches the model,
    /// so an interactive session can remember what was said before. Routing
    /// still uses the last user turn alone, because that is what the router was
    /// trained on -- history steers the answer, not the choice of model.
    ///
    /// # Errors
    ///
    /// As [`Assembly::ask`].
    pub fn chat_stream(
        &self,
        messages: &Messages,
        options: &EngineOptions,
        on_chunk: &mut dyn FnMut(Chunk) -> Result<(), ()>,
    ) -> Result<RouteResult, ManagerError> {
        let messages = messages.with_persona(Some(&self.inner.persona));
        let route = self.route_for(&messages)?;
        let guard = self.ensure_model(&route.model_name)?;

        // The handle is a local clone, so the table lock is released before
        // generation starts and other requests are not blocked behind it. The
        // resident guard still holds the model, so eviction cannot choose it.
        let handle = self.handle_for(&route.model_name)?;

        let chunks = self
            .inner
            .engine
            .chat_stream(handle.as_ref(), &messages, options)
            .map_err(|e| ManagerError::Generate(e.to_string()))?;

        for chunk in chunks {
            if on_chunk(chunk).is_err() {
                break;
            }
        }
        drop(guard);
        Ok(route)
    }

    /// Tokenize `text` with `name`'s model, loading it if needed.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::Model`] if the model cannot be loaded.
    pub fn count_tokens(&self, name: &str, text: &str) -> Result<Vec<u32>, ManagerError> {
        let _guard = self.ensure_model(name)?;
        let handle = self.handle_for(name)?;
        Ok(self.inner.engine.tokenize(handle.as_ref(), text, true)?)
    }

    /// Drop every loaded model and staged blob.
    ///
    /// Safe to call more than once. A server calls this on shutdown so weights
    /// are freed deterministically rather than at process exit.
    pub fn close(&self) {
        let names: Vec<String> = self
            .loaded()
            .lock()
            .expect("loaded handles poisoned")
            .keys()
            .cloned()
            .collect();
        for name in names {
            self.unload(&name);
        }
        if let Some(handle) = self.router_slot().lock().expect("router poisoned").take() {
            self.inner.engine.release(handle);
        }
        self.inner.resident.clear();
        self.inner.staging.clear();
    }
}

/// One message in an OpenAI-style request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatRequest {
    /// `system`, `user` or `assistant`. Anything else is an error at the server
    /// boundary rather than silently coerced.
    pub role: Role,
    /// The message text.
    pub content: String,
}

impl ChatRequest {
    /// A message with an explicit role.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        ChatRequest {
            role,
            content: content.into(),
        }
    }
}

/// One answer, with the routing decision that produced it.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// The generated text.
    pub text: String,
    /// Which model answered, and how confident the router was.
    pub route: RouteResult,
}

/// Turn a conversation into the shape the engine takes.
///
/// Kept here rather than in the server so the same mapping is available to
/// anything else that needs to render a conversation.
pub fn messages_from_pairs(pairs: &[ChatRequest]) -> Messages {
    let mut messages = Messages::new();
    for pair in pairs {
        messages.push(Message::new(pair.role, pair.content.clone()));
    }
    messages
}

/// The embedding entry, for callers that want to report on the router.
pub fn router_entry(header: &AsmbHeader) -> &EmbeddingEntry {
    &header.embedding
}

/// The execution model entries, in archive order.
pub fn model_entries(header: &AsmbHeader) -> &[ModelEntry] {
    &header.models
}
#[cfg(test)]
mod tests {
    use super::*;
    use asmb_format::{BlobSource, BuildPlan, ModelSource, RouterSettings};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::engine::{Chunk, EngineError};

    /// An engine that answers without a GGUF file.
    ///
    /// The whole reason `Assembly` takes an `Arc<dyn Engine>`: everything below
    /// -- staging, the resident budget, routing, streaming -- is exercised here
    /// with no runtime compiled and no model on disk.
    struct FakeEngine {
        loaded: Mutex<Vec<String>>,
        released: AtomicUsize,
        streams: AtomicUsize,
        /// Embeddings to hand back, so routing has something to compare.
        vector: Vec<f32>,
        fail_load: FailMode,
    }

    /// Which loads an engine refuses.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum FailMode {
        /// Nothing fails. The default.
        None,
        /// Only generation models; the router is fine.
        Generation,
    }

    impl FakeEngine {
        fn new(vector: Vec<f32>) -> Arc<FakeEngine> {
            Arc::new(FakeEngine {
                loaded: Mutex::new(Vec::new()),
                released: AtomicUsize::new(0),
                streams: AtomicUsize::new(0),
                vector,
                fail_load: FailMode::None,
            })
        }

        /// How many times a model was loaded.
        ///
        /// The number the resident cache exists to keep at one: a second load of
        /// a model that is already resident means two copies in RAM.
        fn load_count(&self, name: &str) -> usize {
            self.loaded
                .lock()
                .unwrap()
                .iter()
                .filter(|n| *n == name)
                .count()
        }

        /// How many handles were handed back for release.
        fn released(&self) -> usize {
            self.released.load(Ordering::SeqCst)
        }

        /// An engine whose router loads but whose generation models do not.
        fn failing_generation(&self) -> Arc<FakeEngine> {
            Arc::new(FakeEngine {
                loaded: Mutex::new(Vec::new()),
                released: AtomicUsize::new(0),
                streams: AtomicUsize::new(0),
                vector: self.vector.clone(),
                fail_load: FailMode::Generation,
            })
        }
    }

    struct FakeModel {
        name: String,
        path: PathBuf,
    }

    impl ModelHandle for FakeModel {
        fn name(&self) -> &str {
            &self.name
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Engine for FakeEngine {
        fn load(
            &self,
            path: &Path,
            options: &LoadOptions,
        ) -> Result<Box<dyn ModelHandle>, EngineError> {
            let refuse = match self.fail_load {
                FailMode::None => false,
                FailMode::Generation => !options.embedding,
            };
            if refuse {
                return Err(EngineError::Io {
                    path: path.to_path_buf(),
                    source: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "no"),
                });
            }
            let name = options.name.clone().unwrap_or_else(|| {
                path.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "model".to_string())
            });
            self.loaded.lock().unwrap().push(name.clone());
            Ok(Box::new(FakeModel {
                name,
                path: path.to_path_buf(),
            }))
        }

        fn embed(&self, _model: &dyn ModelHandle, _text: &str) -> Result<Vec<f32>, EngineError> {
            Ok(self.vector.clone())
        }

        fn chat(
            &self,
            model: &dyn ModelHandle,
            _messages: &Messages,
            _options: &EngineOptions,
        ) -> Result<String, EngineError> {
            Ok(format!("reply from {}", model.name()))
        }

        fn chat_stream<'a>(
            &'a self,
            _model: &'a dyn ModelHandle,
            _messages: &Messages,
            _options: &EngineOptions,
        ) -> Result<Box<dyn Iterator<Item = Chunk> + 'a>, EngineError> {
            self.streams.fetch_add(1, Ordering::SeqCst);
            let chunks = vec![
                Chunk {
                    text: "hello ".into(),
                    done: false,
                },
                Chunk {
                    text: "world".into(),
                    done: false,
                },
                Chunk {
                    text: String::new(),
                    done: true,
                },
            ];
            Ok(Box::new(chunks.into_iter()))
        }

        fn tokenize(
            &self,
            _model: &dyn ModelHandle,
            text: &str,
            add_special: bool,
        ) -> Result<Vec<u32>, EngineError> {
            let mut ids: Vec<u32> = text.split_whitespace().map(|w| w.len() as u32).collect();
            if add_special {
                ids.insert(0, 1);
                ids.push(2);
            }
            Ok(ids)
        }

        fn release(&self, _model: Arc<dyn ModelHandle>) {
            self.released.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Write a GGUF-shaped file with a valid header, so pre-flight accepts it.
    fn gguf(dir: &Path, name: &str, size: usize) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        use std::io::Write as _;
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&vec![0u8; size.saturating_sub(24)]).unwrap();
        path
    }

    /// An archive with one router and two models whose anchors point in
    /// different directions, so routing has a real choice to make.
    fn archive(dir: &Path) -> PathBuf {
        let router = gguf(dir, "router.gguf", 128);
        let code = gguf(dir, "code.gguf", 256);
        let prose = gguf(dir, "prose.gguf", 512);

        let plan = BuildPlan::new("Test", Some(BlobSource::new("router", router)))
            .with_router(RouterSettings {
                confidence_threshold: 0.5,
                fallback_model: Some("prose".to_string()),
            })
            .with_model(
                ModelSource::new(BlobSource::new("code", code), vec!["code".into()])
                    .with_vectors(vec![vec![1.0, 0.0, 0.0]]),
            )
            .with_model(
                ModelSource::new(BlobSource::new("prose", prose), vec!["prose".into()])
                    .with_vectors(vec![vec![0.0, 1.0, 0.0]]),
            );

        let dest = dir.join("test.asmb");
        plan.build(&dest).unwrap();
        dest
    }

    /// Open a fresh archive. The tempdir is returned alongside so it outlives
    /// the manager, which reads blobs from it.
    fn open(engine: Arc<FakeEngine>) -> (tempfile::TempDir, Assembly) {
        let tmp = tempfile::tempdir().unwrap();
        let path = archive(tmp.path());
        let assembly = Assembly::open(&path, engine, 0.8).unwrap();
        (tmp, assembly)
    }

    #[test]
    fn opens_an_archive_and_reports_its_shape() {
        let (dir, asm) = open(FakeEngine::new(vec![1.0, 0.0, 0.0]));
        assert_eq!(asm.name(), "Test");
        assert_eq!(
            asm.routable_models(),
            ["code", "prose"],
            "both models have anchors, so both are routable"
        );
        assert!(
            asm.loaded_models().is_empty(),
            "open must not load anything"
        );
        drop(asm);
        drop(dir);
    }

    #[test]
    fn routes_to_the_model_whose_anchors_match() {
        let (_dir, asm) = open(FakeEngine::new(vec![1.0, 0.0, 0.0]));
        let route = asm.route("write me a function").unwrap();
        assert_eq!(route.model_name, "code");
        assert!(!route.is_fallback);
    }

    #[test]
    fn asks_through_the_routed_model() {
        let (_dir, asm) = open(FakeEngine::new(vec![1.0, 0.0, 0.0]));
        let reply = asm.ask("code please", &EngineOptions::default()).unwrap();
        assert_eq!(reply.text, "reply from code");
        assert_eq!(reply.route.model_name, "code");
    }

    #[test]
    fn loads_a_model_once_however_many_times_it_is_asked() {
        let engine = FakeEngine::new(vec![1.0, 0.0, 0.0]);
        let (dir, asm) = open(Arc::clone(&engine));

        for _ in 0..5 {
            asm.ask("code please", &EngineOptions::default()).unwrap();
        }

        assert_eq!(
            engine.load_count("code"),
            1,
            "five answers, one copy in RAM"
        );
        drop(asm);
        drop(dir);
    }

    #[test]
    fn a_failing_load_names_the_model_and_keeps_the_cache_honest() {
        let engine = FakeEngine::new(vec![1.0, 0.0, 0.0]);
        let broken = engine.failing_generation();
        let (dir, asm) = open(broken);

        let err = asm
            .ask("code please", &EngineOptions::default())
            .unwrap_err();
        assert!(
            matches!(err, ManagerError::Model { ref name, .. } if name == "code"),
            "the error must name the model that would not load: {err:?}"
        );

        // The size was recorded before the load so eviction could make room.
        // A failed load must not leave a phantom resident entry that blocks a
        // later attempt. The router is legitimately resident here: it loaded
        // fine, and routing needs it.
        assert!(
            !asm.loaded_models().contains(&"code".to_string()),
            "the model that failed to load must not look resident, found {:?}",
            asm.loaded_models()
        );
        drop(asm);
        drop(dir);
    }

    #[test]
    fn releases_every_handle_on_close() {
        let engine = FakeEngine::new(vec![1.0, 0.0, 0.0]);
        let (dir, asm) = open(Arc::clone(&engine));

        asm.ask("code please", &EngineOptions::default()).unwrap();
        asm.route("anything").unwrap();
        assert!(engine.released() == 0, "nothing is released while running");

        asm.close();
        assert_eq!(
            engine.released(),
            2,
            "the router and the execution model are both released"
        );
        assert!(asm.loaded_models().is_empty());
        drop(asm);
        drop(dir);
    }

    #[test]
    fn concurrent_asks_load_each_model_once() {
        let engine = FakeEngine::new(vec![1.0, 0.0, 0.0]);
        let (dir, asm) = open(Arc::clone(&engine));

        // Without the load lock this races two threads into two copies of the
        // same 4 GB model, which is the failure the resident budget exists to
        // prevent and cannot see coming.
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let asm = asm.clone();
                std::thread::spawn(move || {
                    asm.ask("code please", &EngineOptions::default()).unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(
            engine.load_count("code"),
            1,
            "eight threads, one model in RAM"
        );
        drop(asm);
        drop(dir);
    }

    #[test]
    fn persona_is_applied_to_every_conversation() {
        let (_dir, asm) = open(FakeEngine::new(vec![1.0, 0.0, 0.0]));
        assert!(
            asm.persona().contains("Test"),
            "the default persona names the assembly: {}",
            asm.persona()
        );
    }

    #[test]
    fn streams_chunks_in_order_and_stops_when_the_callback_says_so() {
        let (dir, asm) = open(FakeEngine::new(vec![1.0, 0.0, 0.0]));
        let options = EngineOptions::default();

        let mut seen = Vec::new();
        asm.ask_stream("code please", &options, &mut |c| {
            seen.push(c.text);
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, ["hello ", "world", ""], "done chunk included");

        let mut count = 0;
        asm.ask_stream("code please", &options, &mut |_c| {
            count += 1;
            // A client that hung up: the model must not keep talking.
            Err(())
        })
        .unwrap();
        assert_eq!(count, 1, "an early return stops generation");
        drop(asm);
        drop(dir);
    }

    #[test]
    fn refuses_an_archive_built_without_anchor_vectors() {
        let tmp = tempfile::tempdir().unwrap();
        let router = gguf(tmp.path(), "router.gguf", 128);
        let only = gguf(tmp.path(), "only.gguf", 256);
        let plan = BuildPlan::new("NoVectors", Some(BlobSource::new("router", router)))
            .with_router(RouterSettings {
                confidence_threshold: 0.5,
                fallback_model: None,
            })
            .with_model(ModelSource::new(
                BlobSource::new("only", only),
                vec!["anything".into()],
            ));
        let dest = tmp.path().join("nov.asmb");
        plan.build(&dest).unwrap();

        let engine = FakeEngine::new(vec![1.0, 0.0, 0.0]);
        let err = Assembly::open(&dest, engine, 0.8).unwrap_err();
        assert!(
            matches!(err, ManagerError::NoAnchorVectors),
            "expected NoAnchorVectors, got {err:?}"
        );
    }

    #[test]
    fn reports_a_missing_archive_as_such() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = FakeEngine::new(vec![1.0, 0.0, 0.0]);
        let err = Assembly::open(tmp.path().join("nope.asmb"), engine, 0.8).unwrap_err();
        assert!(matches!(err, ManagerError::Archive { .. }), "got {err:?}");
    }

    #[test]
    fn asks_an_unknown_model_by_name_and_says_so() {
        let (dir, asm) = open(FakeEngine::new(vec![1.0, 0.0, 0.0]));
        let err = asm.ensure_model("absent").unwrap_err();
        assert!(
            matches!(err, ManagerError::Model { ref name, .. } if name == "absent"),
            "got {err:?}"
        );
        drop(asm);
        drop(dir);
    }

    #[test]
    fn staging_keeps_the_blob_on_disk_and_accounts_for_it() {
        let (dir, asm) = open(FakeEngine::new(vec![1.0, 0.0, 0.0]));

        asm.ensure_model("code").unwrap();
        assert!(
            asm.staged_bytes() > 0,
            "a staged blob is on disk and counted against the staging budget"
        );

        let path = asm.staged_path("code").expect("code was staged");
        assert!(path.exists(), "the staged file must survive the guard drop");
        drop(asm);
        drop(dir);
    }
}
