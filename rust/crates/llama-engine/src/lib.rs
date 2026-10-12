//! The production [`Engine`]: llama.cpp, behind Assembly's seam.
//!
//! This crate is the only place in the workspace that depends on `llama-cpp-2`.
//! Everything above it -- the router, the caches, the archive layout -- depends
//! on [`assembly_core::engine::Engine`] instead, which is why those can be tested
//! against a fake with no GGUF file in sight.
//!
//! # Model and context
//!
//! llama.cpp splits a model from its context, and that split is the only awkward
//! part of binding it. [`Engine::embed`] and friends receive a
//! `&dyn ModelHandle`, so the context has to be reachable from the handle: one
//! [`LoadedModel`] holds both and never lets them be separated.
//!
//! llama.cpp ties the context's borrow to the model (`LlamaContext<'model>` holds
//! `&'model LlamaModel`), so storing the pair needs the lifetime extended. That
//! is the one `transmute` in [`LlamaEngine::load`], and it is sound because the
//! model sits behind an `Arc` -- whose allocation does not move when the struct
//! does -- and is declared before the context, so it is dropped after it.
//!
//! # Embeddings
//!
//! llama.cpp computes embeddings only when the context was built with
//! `embeddings(true)`, and returns a vector only when it also has a pooling type.
//! Models such as `nomic-embed-text` leave pooling unspecified in their metadata,
//! so [`LlamaEngine::load`] reads it from `LoadOptions::extra["pooling"]`,
//! defaulting to mean pooling as llama.cpp's own server does. Guessing wrong
//! produces vectors that look valid and route wrongly, which is why it is an
//! option rather than a constant.
//!
//! # Streaming
//!
//! [`Engine::chat_stream`] returns an iterator that holds the context lock and
//! generates one token per `next()`, so a caller sees text as it is produced.
//! Two consequences worth knowing before building a server on it:
//!
//! - The lock is held for the response's duration, so one model serves one request
//!   at a time. That is llama.cpp's constraint rather than Assembly's.
//! - Each token is detokenized on its own, because that is the only way to yield
//!   text before the next token exists. A token whose text depends on its
//!   predecessor can therefore render imperfectly at a chunk boundary; the bytes
//!   are still correct.
//! - A failure part-way through ends the stream without saying why, because the
//!   trait's chunk carries text rather than errors. Callers that need the reason
//!   should use [`Engine::chat`].

use std::any::Any;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use assembly_core::engine::{
    Chunk, Engine, EngineError, EngineOptions, LoadOptions, Message, Messages, ModelHandle, Role,
};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::vocab::LlamaVocab;

/// A model and its context, owned together.
pub struct LoadedModel {
    /// Declared first so it is dropped last: the context inside `inner` borrows
    /// it, and dropping the model first would leave the context pointing at freed
    /// weights.
    /// Never read. It exists to keep the model alive for as long as the context
    /// inside `inner` borrows it, and its position in this struct is what makes
    /// the lifetime extension in `open` sound. Removing it would be a
    /// use-after-free, so the dead-code warning is silenced rather than obeyed.
    #[allow(dead_code)]
    model: Arc<LlamaModel>,
    inner: Mutex<Inner>,
    name: String,
    path: PathBuf,
}

// SAFETY: `LlamaContext` holds a raw pointer to llama.cpp's state and is neither
// `Send` nor `Sync`. Moving one to another thread is fine -- llama.cpp's contexts
// are not thread-affine -- and the `Mutex` is what prevents two threads using it
// at once. `LlamaModel` upstream is already `Send + Sync`.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

/// The parts of a loaded model that need exclusive access.
struct Inner {
    context: LlamaContext<'static>,
    /// Resolved once: looking a template up walks the model's metadata.
    chat_template: Option<LlamaChatTemplate>,
    /// Cached because `LlamaContext::model` is a field, not a method.
    vocab: LlamaVocab<'static>,
    /// How many tokens fit, cached because it cannot change.
    n_ctx: u32,
    /// Whether this context produces embeddings, remembered so `embed` can refuse
    /// rather than return a zero-width vector.
    embedding: bool,
}

impl Inner {
    fn model(&self) -> &'static LlamaModel {
        self.context.model
    }
}

impl std::fmt::Debug for LoadedModel {
    /// Deliberately not derived: `LlamaModel` is a raw pointer, and printing one
    /// would say "0x7f..." where the name and path are what a reader wants.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let locked = self.inner.lock();
        f.debug_struct("LoadedModel")
            .field("name", &self.name)
            .field("path", &self.path)
            .field("embedding", &locked.as_ref().map(|i| i.embedding).ok())
            .field("n_ctx", &locked.as_ref().map(|i| i.n_ctx).ok())
            .finish()
    }
}

impl ModelHandle for LoadedModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
}

/// The llama.cpp backend.
///
/// The backend is process-wide global state: `LlamaBackend::init` starts it and
/// its `Drop` shuts it down, and llama.cpp refuses to do either twice. A
/// `OnceLock` never drops its value, so this pairs exactly one init with no
/// shutdown -- the only safe combination when models can be dropped in any order.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| {
        // llama.cpp greets every startup with a wall of ggml/metal chatter on
        // stderr ("GPU family", "compile pipeline", ...) that says nothing the
        // user asked for. Swallow it unless they opt in with ASSEMBLY_LLAMA_LOG=1,
        // which leaves llama.cpp's own stderr logging in place.
        if std::env::var_os("ASSEMBLY_LLAMA_LOG").is_none() {
            llama_cpp_2::send_logs_to_tracing(
                llama_cpp_2::LogOptions::default().with_logs_enabled(false),
            );
        }
        LlamaBackend::init().expect(
            "llama.cpp could not start, which usually means a GPU driver is present \
             but unusable; load with n_gpu_layers 0 to force CPU inference",
        )
    })
}

/// llama.cpp as an [`Engine`].
pub struct LlamaEngine;

impl Default for LlamaEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl LlamaEngine {
    /// An engine using the process-wide backend.
    pub fn new() -> LlamaEngine {
        // Touch the backend here rather than at first load, so a machine that
        // cannot run llama.cpp says so at startup instead of mid-request.
        let _ = backend();
        LlamaEngine
    }

    /// Load `path` and return a handle with its context already created.
    fn open(path: &Path, options: &LoadOptions) -> Result<LoadedModel, EngineError> {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".to_string());

        // `load_from_file` asserts the path exists in debug builds, so a missing
        // file has to be caught before it reaches llama.cpp.
        if !path.is_file() {
            return Err(EngineError::Io {
                path: path.to_path_buf(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
            });
        }

        let mut model_params = LlamaModelParams::default();
        if let Some(layers) = options.n_gpu_layers {
            model_params = model_params.with_n_gpu_layers(layers);
        }

        // Resolved before loading, not after: a mistyped pooling type should be
        // reported immediately rather than after reading gigabytes of weights.
        let pooling = if options.embedding {
            Some(pooling_from(options)?)
        } else {
            None
        };

        let model = LlamaModel::load_from_file(backend(), path, &model_params).map_err(|e| {
            EngineError::NotAModel {
                path: path.to_path_buf(),
                reason: e.to_string(),
            }
        })?;

        let n_ctx = options.n_ctx.unwrap_or(2048).clamp(1, 1 << 20);
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_embeddings(options.embedding)
            // A prompt longer than llama.cpp's 2048-token default batch would fail
            // to decode, and a chat model with a 4096-token window should be able
            // to use it.
            .with_n_batch(n_ctx.min(8192));

        if let Some(pooling) = pooling {
            ctx_params = ctx_params.with_pooling_type(pooling);
        }

        let model = Arc::new(model);

        // SAFETY: `new_context` borrows `model`. The borrow is widened to `'static`
        // because the pair cannot be expressed as a safe self-reference, which is
        // sound while the model is stored in the `Arc` above -- an `Arc` allocation
        // does not move when `LoadedModel` does -- and is dropped after `inner`,
        // because it is declared first.
        let context: LlamaContext<'static> = unsafe {
            let context =
                model
                    .new_context(backend(), ctx_params)
                    .map_err(|e| EngineError::NotAModel {
                        path: path.to_path_buf(),
                        reason: format!("a context could not be created for it: {e}"),
                    })?;
            std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(context)
        };

        // Both read through the context's `'static` reference to the model rather
        // than the local binding: `LlamaVocab` borrows the model it came from, and
        // the local `Arc` is about to be moved into the struct below.
        let vocab = context.model.vocab();
        let n_ctx = context.n_ctx();
        let chat_template = context.model.chat_template(None).ok();

        Ok(LoadedModel {
            model,
            inner: Mutex::new(Inner {
                context,
                chat_template,
                vocab,
                n_ctx,
                embedding: options.embedding,
            }),
            name,
            path: path.to_path_buf(),
        })
    }
}

/// Read the pooling type from `extra["pooling"]`.
fn pooling_from(options: &LoadOptions) -> Result<LlamaPoolingType, EngineError> {
    match options.extra.get("pooling").map(String::as_str) {
        None | Some("mean") => Ok(LlamaPoolingType::Mean),
        Some("cls") => Ok(LlamaPoolingType::Cls),
        Some("last") => Ok(LlamaPoolingType::Last),
        Some("none") => Ok(LlamaPoolingType::None),
        Some(other) => Err(EngineError::Unsupported {
            model: "this model".to_string(),
            operation: format!("be loaded with pooling {other:?}; use mean, cls, last or none"),
        }),
    }
}

/// Recover a [`LoadedModel`] from an opaque handle.
fn as_loaded(model: &dyn ModelHandle) -> Result<&LoadedModel, EngineError> {
    model
        .as_any()
        .and_then(|any| any.downcast_ref::<LoadedModel>())
        .ok_or_else(|| EngineError::Unsupported {
            model: model.name().to_string(),
            operation: "be used with an engine that did not load it".to_string(),
        })
}

/// Take the context lock, turning a poisoned one into an error.
///
/// A panic inside llama.cpp leaves the context in an unknown state, so this
/// refuses to hand it out again rather than risking a second failure.
fn lock(model: &LoadedModel) -> Result<MutexGuard<'_, Inner>, EngineError> {
    model.inner.lock().map_err(|_| EngineError::Generation {
        model: model.name.clone(),
        reason: "it panicked during an earlier call and cannot be used again".to_string(),
    })
}

/// An error with a generated reason.
fn fail(model: &str, reason: impl Into<String>) -> EngineError {
    EngineError::Generation {
        model: model.to_string(),
        reason: reason.into(),
    }
}

impl Engine for LlamaEngine {
    fn load(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn ModelHandle>, EngineError> {
        Ok(Box::new(LlamaEngine::open(path, options)?))
    }

    fn embed(&self, model: &dyn ModelHandle, text: &str) -> Result<Vec<f32>, EngineError> {
        let loaded = as_loaded(model)?;
        let mut guard = lock(loaded)?;

        if !guard.embedding {
            return Err(EngineError::Unsupported {
                model: loaded.name.clone(),
                operation: "produce embeddings (it was loaded for generation)".to_string(),
            });
        }

        let tokens = guard.vocab.tokenize(text.as_bytes(), true, true);
        if tokens.is_empty() {
            return Err(fail(
                &loaded.name,
                format!("the text produced no tokens: {text:?}"),
            ));
        }

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            batch
                .add(*token, i as i32, &[0], true)
                .map_err(|e| fail(&loaded.name, format!("could not queue tokens: {e}")))?;
        }

        guard
            .context
            .encode(&mut batch)
            .map_err(|e| fail(&loaded.name, format!("embedding failed: {e}")))?;

        // Pooling makes this one vector per sequence. Per-token vectors are
        // `embeddings_ith`, which is not what a router wants.
        guard
            .context
            .embeddings_seq_ith(0)
            .map(<[f32]>::to_vec)
            .map_err(|e| {
                fail(
                    &loaded.name,
                    format!(
                        "no pooled vector came back ({e}); this architecture may need a \
                         pooling type set in extra[\"pooling\"]"
                    ),
                )
            })
    }

    fn chat(
        &self,
        model: &dyn ModelHandle,
        messages: &Messages,
        options: &EngineOptions,
    ) -> Result<String, EngineError> {
        let mut text = String::new();
        for chunk in self.chat_stream(model, messages, options)? {
            text.push_str(&chunk.text);
        }
        Ok(text)
    }

    fn chat_stream<'a>(
        &'a self,
        model: &'a dyn ModelHandle,
        messages: &Messages,
        options: &EngineOptions,
    ) -> Result<Box<dyn Iterator<Item = Chunk> + 'a>, EngineError> {
        let loaded = as_loaded(model)?;
        let guard = lock(loaded)?;
        let mut generator = Generator::start(loaded, guard, messages, options)?;
        Ok(Box::new(std::iter::from_fn(move || generator.next_chunk())))
    }

    fn tokenize(
        &self,
        model: &dyn ModelHandle,
        text: &str,
        add_special: bool,
    ) -> Result<Vec<u32>, EngineError> {
        let loaded = as_loaded(model)?;
        let guard = lock(loaded)?;
        Ok(guard
            .vocab
            .tokenize(text.as_bytes(), add_special, true)
            .iter()
            // llama.cpp token ids are non-negative, so the fallback is
            // unreachable in practice; reporting 0 beats a panic on a value we do
            // not control.
            .map(|t| u32::try_from(t.0).unwrap_or_default())
            .collect())
    }

    fn release(&self, model: Arc<dyn ModelHandle>) {
        // Dropping the last `Arc` drops the context, then the model, then the
        // inner `Arc`. llama.cpp frees both, and nothing here can fail usfully.
        drop(model);
    }
}

/// A generation in progress: one `next_chunk` per token.
struct Generator<'m> {
    name: String,
    /// Held for the whole response, which is how one model serves one request.
    guard: MutexGuard<'m, Inner>,
    /// Copied out of `guard` so the generation loop can hold it while also
    /// holding a mutable borrow of the context.
    vocab: LlamaVocab<'static>,
    batch: LlamaBatch<'m>,
    sampler: LlamaSampler,
    /// Decoded bytes not yet emitted, because a multi-byte character can straddle
    /// two tokens.
    pending: Vec<u8>,
    /// Text emitted so far, for matching a stop sequence that straddles a chunk.
    emitted: String,
    /// Sequences that end generation.
    stop: Vec<String>,
    /// Position the next token is written at.
    pos: i32,
    /// Tokens still to generate.
    budget: u32,
    /// Generation has ended; one final `done` chunk is still owed.
    finishing: bool,
    /// That `done` chunk has been yielded.
    finished: bool,
}

impl<'m> Generator<'m> {
    /// Render the conversation, run the prompt through the context, and work out
    /// how many tokens may be generated.
    fn start(
        model: &'m LoadedModel,
        mut guard: MutexGuard<'m, Inner>,
        messages: &Messages,
        options: &EngineOptions,
    ) -> Result<Generator<'m>, EngineError> {
        let name = model.name.clone();
        let prompt = render_prompt(&guard, &name, messages)?;

        let prompt_tokens = guard.vocab.tokenize(prompt.as_bytes(), true, true);
        if prompt_tokens.is_empty() {
            return Err(fail(&name, "the prompt produced no tokens"));
        }

        // The context outlives one conversation, so its KV cache still holds the
        // previous one. Decoding a new prompt at position 0 over stale entries is
        // an invalid batch, which llama.cpp reports as a bare `-1` that
        // llama-cpp-2 misreads as "n_tokens == 0". Clearing first is also the
        // correct semantics: a second `chat` is a new conversation, not a
        // continuation of the last one.
        guard.context.clear_kv_cache();

        // Only the last prompt token needs logits; asking for them on every token
        // costs llama.cpp's output buffer for nothing.
        let mut batch = LlamaBatch::new(prompt_tokens.len(), 1);
        for (i, token) in prompt_tokens.iter().enumerate() {
            let last = i + 1 == prompt_tokens.len();
            batch
                .add(*token, i as i32, &[0], last)
                .map_err(|e| fail(&name, format!("could not queue the prompt: {e}")))?;
        }

        let pos = prompt_tokens.len() as i32;
        guard
            .context
            .decode(&mut batch)
            .map_err(|e| fail(&name, format!("decoding the prompt failed: {e}")))?;

        // A prompt filling the whole window leaves no room to answer, so the
        // budget is what the context has left, not what was asked for.
        let room = guard.n_ctx.saturating_sub(pos as u32);
        let budget = options.max_tokens.unwrap_or(512).min(room);

        // Re-obtained rather than moved out of `guard`: `LlamaVocab` borrows the
        // model, and it cannot be moved out through a `MutexGuard`. It is a
        // transparent pointer wrapper, so this is a pointer read, not a copy of
        // the vocabulary.
        let vocab = guard.context.model.vocab();
        let sampler = build_sampler(guard.model(), options);

        Ok(Generator {
            name,
            guard,
            vocab,
            batch,
            sampler,
            pending: Vec::new(),
            emitted: String::new(),
            stop: options
                .stop
                .iter()
                .filter(|s| !s.is_empty())
                .cloned()
                .collect(),
            pos,
            budget,
            finishing: false,
            finished: false,
        })
    }

    /// The next chunk, or `None` once the stream has ended.
    fn next_chunk(&mut self) -> Option<Chunk> {
        loop {
            if self.finished {
                return None;
            }

            if self.finishing {
                return self.end();
            }

            if self.budget == 0 {
                self.finishing = true;
                continue;
            }

            // -1 is llama.cpp's "last output row", which is the token the previous
            // decode produced logits for.
            let token = self.sampler.sample(&self.guard.context, -1);
            self.budget -= 1;

            if self.vocab.is_eog(token) {
                self.finishing = true;
                continue;
            }

            let bytes = self.vocab.detokenize(&[token], false, true);
            self.pending.extend_from_slice(&bytes);
            let text = self.take_complete();

            let cut = stop_cut(&self.stop, &self.emitted, &text);
            let text = match cut {
                Some(cut) => {
                    self.finishing = true;
                    text[..cut].to_string()
                }
                None => text,
            };
            self.emitted.push_str(&text);

            // Decode this token so the next sample has logits. Skipped when
            // stopping, because nothing will be sampled again.
            if !self.finishing {
                self.batch.clear();
                if self.push_and_decode(token).is_err() {
                    self.finishing = true;
                }
            }

            // A token that renders as nothing still consumed budget; keep going
            // rather than yielding an empty chunk.
            if text.is_empty() && !self.finishing {
                continue;
            }

            return Some(Chunk { text, done: false });
        }
    }

    /// Write the token at the current position and decode it.
    fn push_and_decode(&mut self, token: LlamaToken) -> Result<(), EngineError> {
        self.batch
            .add(token, self.pos, &[0], true)
            .map_err(|e| fail(&self.name, format!("could not queue a token: {e}")))?;
        self.pos += 1;

        self.guard
            .context
            .decode(&mut self.batch)
            .map_err(|e| fail(&self.name, format!("decoding a token failed: {e}")))
    }

    /// The final chunk, then the end of the stream.
    ///
    /// Whatever is left in `pending` is flushed, because stopping on a token
    /// boundary can still leave a partial character behind.
    fn end(&mut self) -> Option<Chunk> {
        self.finished = true;

        if self.pending.is_empty() {
            return Some(Chunk {
                text: String::new(),
                done: true,
            });
        }

        let rest = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        self.emitted.push_str(&rest);
        Some(Chunk {
            text: rest,
            done: false,
        })
    }

    /// Move whole characters out of `pending`, leaving any partial one behind.
    ///
    /// A token boundary can fall inside a multi-byte character, and decoding one
    /// token at a time is the only way to yield text before the next token
    /// exists. Working in bytes and splitting on a character boundary is what
    /// keeps the two halves from becoming two replacement characters.
    fn take_complete(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }

        let split = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) => e.valid_up_to(),
        };

        if split == 0 {
            return String::new();
        }

        let complete = String::from_utf8_lossy(&self.pending[..split]).into_owned();
        self.pending.drain(..split);
        complete
    }
}

/// Where a stop sequence first appears in the text a caller is about to see.
///
/// Returns a byte offset into `new`, or `None`. Only a stop that has actually
/// started being emitted counts: a sequence that merely appears in the prompt is
/// not this generation's business.
fn stop_cut(stop: &[String], before: &str, new: &str) -> Option<usize> {
    let mut best: Option<usize> = None;

    for sequence in stop {
        // An empty sequence is found at offset 0 by `str::find`, which would end
        // generation before the first token. Callers filter these out too, but
        // the check belongs next to the match that would misfire on it.
        if sequence.is_empty() {
            continue;
        }

        let mut consider = |at: usize| {
            if best.is_none_or(|current| at < current) {
                best = Some(at);
            }
        };

        // Wholly inside this chunk.
        if let Some(at) = new.find(sequence.as_str()) {
            consider(at);
        }

        // Or split across the boundary: a proper prefix of the sequence ends the
        // text emitted so far, and the rest starts this chunk. Splitting on
        // `char_indices` keeps the split on a character boundary, since slicing a
        // `str` anywhere else panics.
        for (at, _) in sequence.char_indices().skip(1) {
            let (head, tail) = sequence.split_at(at);
            if before.ends_with(head) && new.starts_with(tail) {
                consider(0);
                break;
            }
        }
    }

    best
}

/// Render a conversation with the model's own chat template.
fn render_prompt(guard: &Inner, name: &str, messages: &Messages) -> Result<String, EngineError> {
    let template = guard.chat_template.as_ref().ok_or_else(|| {
        // A base model has no template, so there is no honest way to render a
        // conversation for it. Saying so beats guessing ChatML.
        EngineError::Unsupported {
            model: name.to_string(),
            operation: "answer a chat completion (it has no chat template, so it is a \
                        base model rather than an instruct one)"
                .to_string(),
        }
    })?;

    let mut chat = Vec::with_capacity(messages.items.len());
    for Message { role, content } in &messages.items {
        chat.push(
            LlamaChatMessage::new(role_name(*role).to_string(), content.clone())
                .map_err(|e| fail(name, format!("a message could not be rendered: {e}")))?,
        );
    }

    guard
        .model()
        .apply_chat_template(template, &chat, true)
        .map_err(|e| fail(name, format!("the chat template could not be applied: {e}")))
}

/// The wire role name for a template.
///
/// `Tool` and `Other` become `user`: llama.cpp templates have no tool role, and
/// tool output is most like something the person said. The Python side maps the
/// same roles the same way.
fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool | Role::Other => "user",
    }
}

/// Assemble a sampler chain for one call.
///
/// Only knobs the caller set are added, so an unset option leaves llama.cpp's own
/// default alone rather than being overwritten with a value that merely matches it
/// today.
fn build_sampler(model: &LlamaModel, options: &EngineOptions) -> LlamaSampler {
    let mut chain: Vec<LlamaSampler> = Vec::new();

    let repeat = options.repeat_penalty.unwrap_or(1.0);
    let frequency = options.frequency_penalty.unwrap_or(0.0);
    let presence = options.presence_penalty.unwrap_or(0.0);

    // llama.cpp takes all three penalties or none. When any is set, the others go
    // in at their neutral values rather than being silently dropped.
    if repeat != 1.0 || frequency != 0.0 || presence != 0.0 {
        chain.push(LlamaSampler::penalties(
            model.n_vocab(),
            0,
            repeat,
            frequency,
            presence,
        ));
    }

    if let Some(k) = options.top_k {
        chain.push(LlamaSampler::top_k(k as i32));
    }
    if let Some(p) = options.top_p {
        chain.push(LlamaSampler::top_p(p, 1));
    }
    if let Some(t) = options.temperature {
        chain.push(LlamaSampler::temp(t));
    }

    // Last, and always present: this is what actually picks a token. A caller who
    // wants reproducibility sets the seed; otherwise the clock supplies one.
    chain.push(LlamaSampler::dist(
        options.seed.map(|s| s as u32).unwrap_or_else(random_seed),
    ));

    if chain.len() == 1 {
        return chain.pop().expect("the length was checked");
    }
    LlamaSampler::chain(chain, /* no_perf */ true)
}

/// A seed for a request that did not ask for one.
fn random_seed() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(sequences: &[&str]) -> Vec<String> {
        sequences.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn a_stop_sequence_inside_one_chunk_is_found() {
        assert_eq!(stop_cut(&stops(&["###"]), "", "one ### two"), Some(4));
    }

    #[test]
    fn a_stop_sequence_spanning_two_chunks_is_found() {
        // "###" arrived as "#" then "##": a per-chunk search alone would miss it.
        assert_eq!(stop_cut(&stops(&["###"]), "one #", "## two"), Some(0));
    }

    #[test]
    fn the_earliest_stop_wins() {
        // Whichever comes first in the text decides, not whichever is listed first.
        assert_eq!(
            stop_cut(&stops(&["late", "early"]), "", "early late"),
            Some(0)
        );
        assert_eq!(
            stop_cut(&stops(&["early", "late"]), "", "early late"),
            Some(0)
        );
    }

    #[test]
    fn text_already_emitted_does_not_retrigger_a_stop() {
        // The stop was already acted on last chunk; matching it again here would
        // keep truncating a stream that is supposed to have moved on.
        assert_eq!(stop_cut(&stops(&["###"]), "###", " more"), None);
    }

    #[test]
    fn an_absent_stop_is_not_found() {
        assert_eq!(stop_cut(&stops(&["###"]), "one", " two"), None);
        assert_eq!(stop_cut(&[], "one", " two"), None);
    }

    #[test]
    fn an_empty_stop_sequence_is_ignored_rather_than_matching_everywhere() {
        // An empty string is found at offset 0 by `str::find`, which would end
        // generation immediately.
        assert_eq!(stop_cut(&stops(&[""]), "", "anything"), None);
    }

    #[test]
    fn a_multi_byte_stop_sequence_is_matched_whole() {
        assert_eq!(
            stop_cut(
                &stops(&["\u{4f60}\u{597d}"]),
                "",
                "say \u{4f60}\u{597d} now"
            ),
            Some(4)
        );
        // Half the sequence is not a match.
        assert_eq!(
            stop_cut(&stops(&["\u{4f60}\u{597d}"]), "", "say \u{4f60} now"),
            None
        );
    }

    #[test]
    fn a_multi_byte_stop_sequence_split_across_chunks_is_matched() {
        assert_eq!(
            stop_cut(
                &stops(&["\u{4f60}\u{597d}"]),
                "say \u{4f60}",
                "\u{597d} now"
            ),
            Some(0)
        );
    }

    #[test]
    fn pooling_defaults_to_mean_and_can_be_overridden() {
        assert_eq!(
            pooling_from(&LoadOptions::embed()).unwrap(),
            LlamaPoolingType::Mean
        );
        assert_eq!(
            pooling_from(&LoadOptions::embed().extra("pooling", "cls")).unwrap(),
            LlamaPoolingType::Cls
        );
        assert_eq!(
            pooling_from(&LoadOptions::embed().extra("pooling", "none")).unwrap(),
            LlamaPoolingType::None
        );
    }

    #[test]
    fn an_unknown_pooling_type_is_rejected_with_the_alternatives() {
        let err = pooling_from(&LoadOptions::embed().extra("pooling", "average")).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("average"), "got {text}");
        assert!(
            text.contains("mean"),
            "the message should list what works: {text}"
        );
    }

    #[test]
    fn a_handle_from_another_engine_is_refused() {
        // `NamedModel` is a handle that no llama.cpp engine produced, so the
        // downcast must fail rather than reinterpret the pointer.
        let other = assembly_core::engine::NamedModel::new("general", "/models/general.gguf");
        let err = as_loaded(&other).unwrap_err();
        assert!(err.to_string().contains("did not load it"), "got {err}");
    }

    #[test]
    fn a_missing_model_file_reports_the_path_rather_than_panicking() {
        // `load_from_file` asserts the file exists in debug builds, so this guard
        // is the only thing standing between a typo and a panic.
        let err = LlamaEngine::open(
            Path::new("/nonexistent/model.gguf"),
            &LoadOptions::generate(),
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::Io { .. }), "got {err}");
        assert!(err.to_string().contains("model.gguf"), "got {err}");
    }

    #[test]
    fn roles_render_as_the_names_templates_expect() {
        assert_eq!(role_name(Role::System), "system");
        assert_eq!(role_name(Role::User), "user");
        assert_eq!(role_name(Role::Assistant), "assistant");
        // Templates have no tool role; the Python side sends these as user too.
        assert_eq!(role_name(Role::Tool), "user");
        assert_eq!(role_name(Role::Other), "user");
    }
}
