//! The inference runtime seam.
//!
//! This is the Rust counterpart of Python's `Backend`: the six operations
//! Assembly needs from a model runtime, and nothing else. Everything above it --
//! the router, the archive layout, the server -- is written against [`Engine`],
//! so llama.cpp stays a detail rather than a dependency of the code that decides
//! *which* model should answer.
//!
//! Two reasons the seam is worth having before the binding exists:
//!
//! - The server, the router and the resident cache can be written and tested now,
//!   against the [`FakeEngine`] below, with no GGUF file anywhere in sight. That
//!   is the difference between "the HTTP layer works" and "the HTTP layer works
//!   and the only untested part is llama.cpp".
//! - A second runtime can be dropped in later without touching the code above it.
//!
//! # Threading
//!
//! Implementations must be safe to call from more than one thread. A model
//! returned by [`Engine::load`] is owned by the caller and dropped with
//! [`Engine::release`], which must tolerate being called twice.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Who a message is from.
///
/// Mirrors the OpenAI wire format, which is what `/v1/chat/completions` accepts
/// and what the persona injection in [`Messages::with_persona`] operates on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    /// Instructions that frame the conversation.
    System,
    /// The person asking.
    User,
    /// The model answering.
    Assistant,
    /// Output of a tool the model invoked.
    Tool,
    /// Anything else the wire format allows.
    ///
    /// Kept as a variant rather than rejected: OpenAI clients send roles this
    /// server has no opinion about, and dropping the request over an unknown
    /// role would be worse than passing it through.
    Other,
}

impl Role {
    /// Parse a role from its wire name.
    pub fn parse(text: &str) -> Role {
        match text {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::Other,
        }
    }

    /// The role's wire name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::Other => "other",
        }
    }
}

/// One message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// Who is speaking.
    pub role: Role,
    /// What they said.
    ///
    /// Empty is legal: OpenAI clients send `content: null` on tool-call
    /// messages, and persona injection prepends to whatever is here.
    pub content: String,
}

impl Message {
    /// A message from the named role.
    pub fn new(role: Role, content: impl Into<String>) -> Message {
        Message {
            role,
            content: content.into(),
        }
    }

    /// A user message.
    pub fn user(content: impl Into<String>) -> Message {
        Message::new(Role::User, content)
    }

    /// A system message.
    pub fn system(content: impl Into<String>) -> Message {
        Message::new(Role::System, content)
    }

    /// An assistant message.
    pub fn assistant(content: impl Into<String>) -> Message {
        Message::new(Role::Assistant, content)
    }
}

/// A conversation, in order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Messages {
    /// The messages, oldest first.
    pub items: Vec<Message>,
}

impl Messages {
    /// An empty conversation.
    pub fn new() -> Messages {
        Messages::default()
    }

    /// One user turn.
    pub fn from_prompt(prompt: impl Into<String>) -> Messages {
        Messages {
            items: vec![Message::user(prompt)],
        }
    }

    /// Append a message.
    pub fn push(&mut self, message: Message) {
        self.items.push(message);
    }

    /// How many messages there are.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the conversation has no messages.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The same conversation with `persona` installed as the system message.
    ///
    /// An existing system message is *prepended to* rather than replaced,
    /// because a caller's own system message carries request-specific
    /// instructions that the archive's persona has no business overwriting. If
    /// the persona is already the whole content, nothing is prepended, so
    /// repeated calls do not stack.
    ///
    /// The input is not mutated. Doing it in place would make
    /// `with_persona(p).with_persona(q)` produce a conversation nobody asked
    /// for, and the caller cannot see what was actually sent.
    pub fn with_persona(&self, persona: Option<&str>) -> Messages {
        let Some(persona) = persona.filter(|p| !p.trim().is_empty()) else {
            return self.clone();
        };

        let mut items = self.items.clone();

        match items.iter_mut().find(|m| m.role == Role::System) {
            Some(existing) => {
                if !existing.content.contains(persona) {
                    existing.content = format!("{persona}\n\n{}", existing.content.trim());
                }
            }
            // No system message yet: the persona becomes one, at the front.
            None => items.insert(0, Message::system(persona)),
        }

        Messages { items }
    }
}

/// Knobs for one inference call.
///
/// llama.cpp accepts dozens of options. This holds the ones Assembly has an
/// opinion about, all optional: `None` means "leave it to the runtime", so the
/// two implementations can keep their own defaults without either side having to
/// restate them here. Anything else belongs in [`EngineOptions::extra`], which
/// passes through untouched, so a new llama.cpp parameter does not need a change
/// here first.
///
/// # Sampling defaults
///
/// [`EngineOptions::defaults`] carries llama.cpp's own defaults, which are what
/// Python's `SamplingParams` uses. They are only applied where a request does
/// not say otherwise. A client asking for nothing gets those; a client asking
/// for one knob keeps llama.cpp's value for the rest.
///
/// The three penalties stay separate because llama.cpp implements them
/// independently. Collapsing them into one "penalty" would change what every
/// existing request means: OpenAI clients send presence and frequency penalties
/// in `-2.0..=2.0`, while `repeat_penalty` is a multiplier that only means
/// anything above `1.0`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EngineOptions {
    /// Sampling temperature. `None` leaves the runtime's default.
    pub temperature: Option<f32>,
    /// Nucleus sampling cut-off.
    pub top_p: Option<f32>,
    /// Top-k sampling.
    pub top_k: Option<u32>,
    /// Penalty for tokens already generated. A multiplier, not an offset.
    pub repeat_penalty: Option<f32>,
    /// Added to the logits of tokens present in the output so far.
    pub presence_penalty: Option<f32>,
    /// Scaled by each token's count in the output so far.
    pub frequency_penalty: Option<f32>,
    /// Fixes the sampler, so the same request gives the same answer.
    pub seed: Option<u64>,
    /// Sequences that stop generation.
    pub stop: Vec<String>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Pass-through options for the runtime, by name.
    pub extra: BTreeMap<String, String>,
}

impl EngineOptions {
    /// Options with everything at the runtime's defaults.
    pub fn new() -> EngineOptions {
        EngineOptions::default()
    }

    /// llama.cpp's defaults, which Python's `SamplingParams` also uses.
    ///
    /// Not the same as [`Self::new`]: this fills in the knobs, `new` defers to
    /// the runtime. Both are valid, and a server should pick `new` and let each
    /// model use the defaults it was built for.
    pub fn defaults() -> EngineOptions {
        EngineOptions {
            temperature: Some(0.8),
            top_p: Some(0.95),
            top_k: Some(40),
            repeat_penalty: Some(1.1),
            presence_penalty: Some(0.0),
            frequency_penalty: Some(0.0),
            seed: None,
            stop: Vec::new(),
            max_tokens: Some(512),
            extra: BTreeMap::new(),
        }
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set the nucleus cut-off.
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Set the top-k cut-off.
    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Set the repeat penalty.
    pub fn repeat_penalty(mut self, p: f32) -> Self {
        self.repeat_penalty = Some(p);
        self
    }

    /// Set the presence penalty.
    pub fn presence_penalty(mut self, p: f32) -> Self {
        self.presence_penalty = Some(p);
        self
    }

    /// Set the frequency penalty.
    pub fn frequency_penalty(mut self, p: f32) -> Self {
        self.frequency_penalty = Some(p);
        self
    }

    /// Fix the sampler seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Add a stop sequence.
    pub fn stop(mut self, sequence: impl Into<String>) -> Self {
        self.stop.push(sequence.into());
        self
    }

    /// Cap the generated length.
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Set a pass-through option.
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// These options layered over `base`, keeping `base` where they say nothing.
    ///
    /// Used for `/v1/chat/completions`, where a client sets one or two knobs and
    /// expects the rest of the assembly's settings to survive. Anything not set
    /// on `self` falls back rather than reverting to a default, so applying an
    /// empty overlay is a no-op.
    pub fn or(self, base: EngineOptions) -> EngineOptions {
        EngineOptions {
            temperature: self.temperature.or(base.temperature),
            top_p: self.top_p.or(base.top_p),
            top_k: self.top_k.or(base.top_k),
            repeat_penalty: self.repeat_penalty.or(base.repeat_penalty),
            presence_penalty: self.presence_penalty.or(base.presence_penalty),
            frequency_penalty: self.frequency_penalty.or(base.frequency_penalty),
            seed: self.seed.or(base.seed),
            // Stop sequences accumulate: a request that adds one means "also
            // stop on this", not "stop on this instead of what I had".
            stop: if self.stop.is_empty() {
                base.stop
            } else {
                let mut all = base.stop;
                all.extend(self.stop);
                all
            },
            max_tokens: self.max_tokens.or(base.max_tokens),
            extra: base.extra.into_iter().chain(self.extra).collect(),
        }
    }

    /// Clamp every knob into the range llama.cpp accepts.
    ///
    /// Clamping rather than rejecting, matching the Python server: a client
    /// asking for something out of range should still get an answer, and the
    /// nearest legal value is closer to what it meant than a 400. A value that
    /// is not a number at all is dropped, which leaves the previous value in
    /// place.
    pub fn clamped(mut self) -> EngineOptions {
        if let Some(t) = self.temperature {
            self.temperature = Some(t.clamp(0.0, 2.0));
        }
        if let Some(p) = self.top_p {
            self.top_p = Some(p.clamp(0.0, 1.0));
        }
        if let Some(k) = self.top_k {
            self.top_k = Some(k.clamp(1, 100));
        }
        if let Some(p) = self.repeat_penalty {
            self.repeat_penalty = Some(p.clamp(1.0, 2.0));
        }
        // OpenAI's range for both of these.
        if let Some(p) = self.presence_penalty {
            self.presence_penalty = Some(p.clamp(-2.0, 2.0));
        }
        if let Some(p) = self.frequency_penalty {
            self.frequency_penalty = Some(p.clamp(-2.0, 2.0));
        }
        // Below 16 a model answers with a fragment; above 8192 it can run for
        // minutes on a request that probably did not need it.
        if let Some(n) = self.max_tokens {
            self.max_tokens = Some(n.clamp(16, 8192));
        }
        self
    }
}

/// Options for loading a model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoadOptions {
    /// Load for embedding rather than generation.
    pub embedding: bool,
    /// Context window in tokens. `None` leaves the model's own default.
    pub n_ctx: Option<u32>,
    /// Layers to offload to the GPU. `None` leaves the runtime's default.
    pub n_gpu_layers: Option<u32>,
    /// The logical model name, as the archive spells it.
    ///
    /// A caller loading out of a staged blob has only a temporary path like
    /// `code-18dafff1-256` to go on, and a name derived from that leaks into
    /// `ModelHandle::name`, into `/v1/models`, and into every log line. The name
    /// the user chose is passed instead.
    pub name: Option<String>,
    /// Pass-through options for the runtime.
    pub extra: BTreeMap<String, String>,
}

impl LoadOptions {
    /// Options for a generation model.
    pub fn generate() -> LoadOptions {
        LoadOptions::default()
    }

    /// Options for an embedding model.
    ///
    /// The small context is deliberate: embedding a phrase needs room for a
    /// sentence, not a conversation, and a larger window costs memory that the
    /// execution models want.
    pub fn embed() -> LoadOptions {
        LoadOptions {
            embedding: true,
            n_ctx: Some(512),
            n_gpu_layers: Some(0),
            name: None,
            extra: BTreeMap::new(),
        }
    }

    /// Set the context window.
    pub fn n_ctx(mut self, n: u32) -> Self {
        self.n_ctx = Some(n);
        self
    }

    /// Record the logical model name.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the GPU offload count.
    pub fn n_gpu_layers(mut self, n: u32) -> Self {
        self.n_gpu_layers = Some(n);
        self
    }

    /// Set a pass-through option.
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// One token's worth of generated text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// The text to append to what came before.
    ///
    /// Empty is meaningful: it is how a multi-byte character or a BPE token that
    /// renders as nothing arrives, and dropping empty chunks would lose the
    /// difference between "the model is thinking" and "the model stopped".
    pub text: String,
    /// Whether generation has finished.
    pub done: bool,
}

/// A loaded model, owned by its caller.
///
/// Opaque on purpose: what a model *is* is the runtime's business, and letting
/// the type name leak an assumption here would put the llama.cpp model type in
/// every signature that touches one.
pub trait ModelHandle: Send + Sync {
    /// The name this model was loaded under.
    fn name(&self) -> &str;

    /// The file it was loaded from.
    fn path(&self) -> &Path;

    /// This handle as an opaque reference, so a runtime can get its own handle
    /// type back.
    ///
    /// `Engine` receives handles as `&dyn ModelHandle`, so a runtime that needs
    /// its own fields back has no other way to reach them: the trait deliberately
    /// exposes no llama.cpp type. `Any` is the standard escape hatch, and it is
    /// safe -- a downcast only succeeds for the type that produced the handle.
    ///
    /// Defaults to `None`, which is what a handle that holds nothing needs.
    /// Returns `Some(self)` from a runtime that can name its own type.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

/// The operations Assembly needs from an inference runtime.
///
/// The same six methods Python's `Backend` exposes, so the port has one place
/// where runtime behaviour differs rather than six.
pub trait Engine: Send + Sync {
    /// Load a model from `path`.
    ///
    /// # Errors
    ///
    /// Returns an error describing why the file could not be loaded: a missing
    /// file, a GGUF this runtime cannot read, a context window the model
    /// rejects. Callers surface the message, so it should read as a sentence
    /// about the file rather than as a runtime error code.
    fn load(&self, path: &Path, options: &LoadOptions)
        -> Result<Box<dyn ModelHandle>, EngineError>;

    /// Embed `text`, returning a vector of `f32`.
    ///
    /// The width is whatever the model produces and is not negotiable;
    /// [`crate::router::Router`] skips models whose anchors disagree with it.
    ///
    /// # Errors
    ///
    /// Returns an error when the model is not an embedding model, or the text
    /// could not be embedded.
    fn embed(&self, model: &dyn ModelHandle, text: &str) -> Result<Vec<f32>, EngineError>;

    /// Run a chat completion, returning the whole response.
    ///
    /// # Errors
    ///
    /// Returns an error when generation fails.
    fn chat(
        &self,
        model: &dyn ModelHandle,
        messages: &Messages,
        options: &EngineOptions,
    ) -> Result<String, EngineError>;

    /// Run a chat completion, yielding text as it is produced.
    ///
    /// `self` and `model` share one lifetime on purpose. With them borrowed
    /// separately, an implementation could not return an iterator that holds
    /// state derived from the model -- which is exactly what streaming needs,
    /// because a real runtime produces tokens while still holding the context
    /// the tokens came from. Sharing the lifetime says so in the type rather than
    /// in a comment, at the cost that the iterator cannot outlive either borrow.
    /// That is not a restriction worth worrying about: a response is only
    /// meaningful while its model is loaded.
    ///
    /// The last chunk always has `done` set and empty text.
    ///
    /// A runtime that fails part-way through yields nothing further and stops,
    /// rather than inventing an error chunk. Callers that need the reason should
    /// use [`Engine::chat`] when they cannot tolerate a truncated answer.
    ///
    /// # Errors
    ///
    /// Returns an error when generation cannot be started.
    fn chat_stream<'a>(
        &'a self,
        model: &'a dyn ModelHandle,
        messages: &Messages,
        options: &EngineOptions,
    ) -> Result<Box<dyn Iterator<Item = Chunk> + 'a>, EngineError>;

    /// Tokenize `text`.
    ///
    /// # Errors
    ///
    /// Returns an error when the text cannot be tokenized.
    fn tokenize(
        &self,
        model: &dyn ModelHandle,
        text: &str,
        add_special: bool,
    ) -> Result<Vec<u32>, EngineError>;

    /// Drop a model and everything it holds.
    ///
    /// Takes an `Arc` because that is what a shared manager holds. Dropping is
    /// refcounted, so an in-flight request that cloned the handle keeps a valid
    /// model while the cache forgets about it.
    ///
    /// Must tolerate being called twice on the same handle, because a caller
    /// unwinding after an error should not have to know whether cleanup already
    /// ran. Failures are swallowed rather than returned: a model that will not
    /// release cleanly is a leak, not a reason to fail a request that already
    /// produced its answer.
    fn release(&self, model: Arc<dyn ModelHandle>);
}

/// Why an engine operation failed.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The model file could not be opened or read.
    #[error("cannot open model {path}: {source}")]
    Io {
        /// The file that failed.
        path: PathBuf,
        /// The underlying reason.
        source: std::io::Error,
    },

    /// The file is not a model this runtime can load.
    #[error("{path} is not a loadable model: {reason}")]
    NotAModel {
        /// The file that failed.
        path: PathBuf,
        /// What was wrong with it.
        reason: String,
    },

    /// The operation was asked for on a model that cannot do it.
    #[error("'{model}' cannot {operation}")]
    Unsupported {
        /// The model that was asked.
        model: String,
        /// What was asked of it.
        operation: String,
    },

    /// Generation failed part-way through.
    #[error("generation failed on '{model}': {reason}")]
    Generation {
        /// The model that failed.
        model: String,
        /// What went wrong.
        reason: String,
    },
}

/// A [`ModelHandle`] for a runtime that does not own anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedModel {
    name: String,
    path: PathBuf,
}

impl NamedModel {
    /// A handle for `name` loaded from `path`.
    pub fn new(name: impl Into<String>, path: impl Into<PathBuf>) -> NamedModel {
        NamedModel {
            name: name.into(),
            path: path.into(),
        }
    }
}

impl ModelHandle for NamedModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// An engine that records what it was asked and returns canned answers.
    ///
    /// This is the reason the seam exists: the router, the server and the caches
    /// can all be tested against it without a GGUF file or a compiled runtime.
    #[derive(Default)]
    struct FakeEngine {
        loaded: Mutex<Vec<(String, PathBuf, LoadOptions)>>,
        released: AtomicUsize,
        embeds: Mutex<Vec<String>>,
        chats: Mutex<Vec<(String, Vec<String>)>>,
        /// Chunk texts returned by `chat_stream`.
        stream: Vec<String>,
        /// Set to make every operation fail, to exercise error paths.
        fail: bool,
    }

    impl FakeEngine {
        fn chat_text(&self, name: &str) -> Option<String> {
            self.chats
                .lock()
                .unwrap()
                .iter()
                .rev()
                .find(|(model, _)| model == name)
                .map(|(_, messages)| messages.join("|"))
        }
    }

    /// A handle carrying the text it should answer with.
    struct FakeModel(NamedModel);

    impl ModelHandle for FakeModel {
        fn name(&self) -> &str {
            self.0.name()
        }

        fn path(&self) -> &Path {
            self.0.path()
        }
    }

    impl Engine for FakeEngine {
        fn load(
            &self,
            path: &Path,
            options: &LoadOptions,
        ) -> Result<Box<dyn ModelHandle>, EngineError> {
            if self.fail {
                return Err(EngineError::Io {
                    path: path.to_path_buf(),
                    source: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "nope"),
                });
            }
            let name = options.name.clone().unwrap_or_else(|| {
                path.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "model".to_string())
            });
            self.loaded
                .lock()
                .unwrap()
                .push((name.clone(), path.to_path_buf(), options.clone()));
            Ok(Box::new(FakeModel(NamedModel::new(name, path))))
        }

        fn embed(&self, _model: &dyn ModelHandle, text: &str) -> Result<Vec<f32>, EngineError> {
            self.embeds.lock().unwrap().push(text.to_string());
            Ok(vec![1.0, 0.0, 0.0])
        }

        fn chat(
            &self,
            model: &dyn ModelHandle,
            messages: &Messages,
            _options: &EngineOptions,
        ) -> Result<String, EngineError> {
            let contents: Vec<String> = messages.items.iter().map(|m| m.content.clone()).collect();
            self.chats
                .lock()
                .unwrap()
                .push((model.name().to_string(), contents));
            Ok(format!("reply from {}", model.name()))
        }

        fn chat_stream<'a>(
            &'a self,
            _model: &'a dyn ModelHandle,
            _messages: &Messages,
            _options: &EngineOptions,
        ) -> Result<Box<dyn Iterator<Item = Chunk> + 'a>, EngineError> {
            let chunks: Vec<Chunk> = self
                .stream
                .iter()
                .map(|t| Chunk {
                    text: t.clone(),
                    done: false,
                })
                .chain(std::iter::once(Chunk {
                    text: String::new(),
                    done: true,
                }))
                .collect();
            Ok(Box::new(chunks.into_iter()))
        }

        fn tokenize(
            &self,
            _model: &dyn ModelHandle,
            text: &str,
            add_special: bool,
        ) -> Result<Vec<u32>, EngineError> {
            // Deterministic and good enough: the point is the width, not the
            // vocabulary.
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

    #[test]
    fn roles_round_trip_through_their_wire_names() {
        for role in [Role::System, Role::User, Role::Assistant, Role::Tool] {
            assert_eq!(Role::parse(role.as_str()), role);
        }
        assert_eq!(Role::parse("function"), Role::Other);
        assert_eq!(Role::Other.as_str(), "other");
    }

    #[test]
    fn a_persona_becomes_the_system_message() {
        let msgs = Messages::from_prompt("hello");
        let with = msgs.with_persona(Some("be brief"));

        assert_eq!(with.len(), 2);
        assert_eq!(with.items[0].role, Role::System);
        assert_eq!(with.items[0].content, "be brief");
        assert_eq!(with.items[1].role, Role::User);
        assert_eq!(with.items[1].content, "hello");
    }

    #[test]
    fn a_persona_is_prepended_to_an_existing_system_message() {
        let mut msgs = Messages::new();
        msgs.push(Message::system("answer in French"));
        msgs.push(Message::user("bonjour"));

        let with = msgs.with_persona(Some("be brief"));
        assert_eq!(with.items[0].content, "be brief\n\nanswer in French");
        assert_eq!(with.items[0].role, Role::System);
        assert_eq!(with.items.len(), 2, "no extra message was added");
    }

    #[test]
    fn a_persona_does_not_stack() {
        let msgs = Messages::from_prompt("hi");
        let once = msgs.with_persona(Some("be brief"));
        let twice = once.with_persona(Some("be brief"));
        assert_eq!(once, twice);
        assert_eq!(twice.items[0].content, "be brief");
    }

    #[test]
    fn persona_injection_does_not_mutate_the_input() {
        let msgs = Messages::from_prompt("hello");
        let before = msgs.clone();
        let _ = msgs.with_persona(Some("be brief"));
        assert_eq!(msgs, before, "with_persona must leave its input alone");
    }

    #[test]
    fn an_absent_or_blank_persona_changes_nothing() {
        let msgs = Messages::from_prompt("hi");
        assert_eq!(msgs.with_persona(None), msgs);
        assert_eq!(msgs.with_persona(Some("")), msgs);
        assert_eq!(msgs.with_persona(Some("   ")), msgs);
    }

    #[test]
    fn a_system_message_after_other_roles_still_receives_the_persona() {
        // Persona injection must find the system message wherever it sits, not
        // only at the front.
        let mut msgs = Messages::new();
        msgs.push(Message::user("first"));
        msgs.push(Message::system("mid-conversation instruction"));
        msgs.push(Message::user("second"));

        let with = msgs.with_persona(Some("be brief"));
        assert_eq!(with.len(), 3, "no message was inserted");
        assert_eq!(with.items[1].role, Role::System);
        assert!(with.items[1].content.starts_with("be brief"));
    }

    #[test]
    fn options_build_the_way_they_read() {
        let opts = EngineOptions::new()
            .temperature(0.2)
            .top_p(0.9)
            .top_k(40)
            .repeat_penalty(1.15)
            .presence_penalty(0.5)
            .frequency_penalty(-0.5)
            .seed(1234)
            .stop("END")
            .max_tokens(256)
            .extra("mirostat", "2");

        assert_eq!(opts.temperature, Some(0.2));
        assert_eq!(opts.top_p, Some(0.9));
        assert_eq!(opts.top_k, Some(40));
        assert_eq!(opts.repeat_penalty, Some(1.15));
        assert_eq!(opts.presence_penalty, Some(0.5));
        assert_eq!(opts.frequency_penalty, Some(-0.5));
        assert_eq!(opts.seed, Some(1234));
        assert_eq!(opts.stop, vec!["END".to_string()]);
        assert_eq!(opts.max_tokens, Some(256));
        assert_eq!(opts.extra.get("mirostat").map(String::as_str), Some("2"));
    }

    #[test]
    fn new_options_defer_to_the_runtime_and_defaults_do_not() {
        let deferred = EngineOptions::new();
        assert_eq!(deferred.temperature, None);
        assert_eq!(deferred.max_tokens, None);

        // These are llama.cpp's own defaults, which Python's SamplingParams also
        // uses. A model that disagrees with them needs the request to say so.
        let defaults = EngineOptions::defaults();
        assert_eq!(defaults.temperature, Some(0.8));
        assert_eq!(defaults.top_p, Some(0.95));
        assert_eq!(defaults.top_k, Some(40));
        assert_eq!(defaults.repeat_penalty, Some(1.1));
        assert_eq!(defaults.presence_penalty, Some(0.0));
        assert_eq!(defaults.frequency_penalty, Some(0.0));
        assert_eq!(defaults.max_tokens, Some(512));
        assert_eq!(
            defaults.seed, None,
            "a fixed seed would make every answer reproducible, which is not a default"
        );
    }

    #[test]
    fn an_overlay_keeps_the_base_where_it_says_nothing() {
        // A chat request that sets one knob must not reset the rest.
        let base = EngineOptions::defaults().temperature(0.3).stop("END");
        let merged = EngineOptions::new().max_tokens(64).or(base.clone());

        assert_eq!(merged.max_tokens, Some(64), "the overlay wins");
        assert_eq!(merged.temperature, Some(0.3), "the base survives");
        assert_eq!(merged.top_k, Some(40), "and so do untouched defaults");
        assert_eq!(merged.stop, vec!["END".to_string()]);
        assert_eq!(
            merged.extra.len(),
            0,
            "neither side had pass-through options"
        );
    }

    #[test]
    fn stop_sequences_accumulate_across_an_overlay() {
        // "also stop on this" rather than "instead of what I had".
        let base = EngineOptions::new().stop("USER:");
        let merged = EngineOptions::new().stop("###").or(base);
        assert_eq!(merged.stop, vec!["USER:".to_string(), "###".to_string()]);
    }

    #[test]
    fn an_empty_overlay_changes_nothing() {
        let base = EngineOptions::defaults().temperature(0.1).stop("END");
        assert_eq!(EngineOptions::new().or(base.clone()), base);
    }

    #[test]
    fn out_of_range_knobs_are_clamped_not_rejected() {
        // A client asking for something out of range should still get an answer,
        // and the nearest legal value is closer to what it meant than a 400.
        let opts = EngineOptions::new()
            .temperature(9.0)
            .top_p(4.0)
            .top_k(5000)
            .repeat_penalty(0.1)
            .presence_penalty(-9.0)
            .frequency_penalty(9.0)
            .max_tokens(1)
            .clamped();

        assert_eq!(opts.temperature, Some(2.0));
        assert_eq!(opts.top_p, Some(1.0));
        assert_eq!(opts.top_k, Some(100));
        assert_eq!(opts.repeat_penalty, Some(1.0));
        assert_eq!(opts.presence_penalty, Some(-2.0));
        assert_eq!(opts.frequency_penalty, Some(2.0));
        assert_eq!(
            opts.max_tokens,
            Some(16),
            "below 16 a model answers with a fragment"
        );
    }

    #[test]
    fn clamping_leaves_in_range_values_alone() {
        let opts = EngineOptions::defaults().clamped();
        assert_eq!(
            opts,
            EngineOptions::defaults(),
            "llama.cpp's own defaults are already legal"
        );
    }

    #[test]
    fn clamping_leaves_unset_knobs_unset() {
        // Clamping must not manufacture a default the caller did not ask for.
        let opts = EngineOptions::new().clamped();
        assert_eq!(opts, EngineOptions::new());
        assert_eq!(opts.temperature, None);
    }

    #[test]
    fn embedding_load_options_are_distinct_from_generation_ones() {
        let embed = LoadOptions::embed();
        assert!(embed.embedding);
        assert_eq!(embed.n_ctx, Some(512));

        let generate = LoadOptions::generate();
        assert!(!generate.embedding);
        assert_eq!(
            generate.n_ctx, None,
            "a generation model keeps its own context"
        );
    }

    #[test]
    fn the_engine_receives_what_it_was_told() {
        let engine = FakeEngine::default();
        let path = Path::new("/models/general.gguf");
        let model = engine
            .load(path, &LoadOptions::generate().n_ctx(4096).n_gpu_layers(99))
            .unwrap();

        let loaded = engine.loaded.lock().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "general", "the stem names the handle");
        assert_eq!(loaded[0].1, path);
        assert_eq!(loaded[0].2.n_ctx, Some(4096));
        assert_eq!(loaded[0].2.n_gpu_layers, Some(99));

        assert_eq!(model.name(), "general");
        assert_eq!(model.path(), path);
    }

    #[test]
    fn chat_sees_the_messages_it_was_given() {
        let engine = FakeEngine::default();
        let model = engine
            .load(Path::new("/models/general.gguf"), &LoadOptions::generate())
            .unwrap();

        let mut msgs = Messages::new();
        msgs.push(Message::user("what is 2+2"));
        let answer = engine
            .chat(model.as_ref(), &msgs, &EngineOptions::new())
            .unwrap();

        assert_eq!(answer, "reply from general");
        assert_eq!(
            engine.chat_text("general").as_deref(),
            Some("what is 2+2"),
            "the engine received the conversation"
        );
    }

    #[test]
    fn streaming_ends_with_a_done_chunk_and_keeps_empty_text() {
        let engine = FakeEngine {
            stream: vec!["Hello".into(), ", ".into(), "world".into()],
            ..Default::default()
        };
        let model = engine
            .load(Path::new("/models/general.gguf"), &LoadOptions::generate())
            .unwrap();

        let chunks: Vec<Chunk> = engine
            .chat_stream(
                model.as_ref(),
                &Messages::from_prompt("hi"),
                &EngineOptions::new(),
            )
            .unwrap()
            .collect();

        assert_eq!(
            chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(),
            vec!["Hello", ", ", "world", ""]
        );
        assert!(chunks.last().unwrap().done, "the stream must end with done");
        assert!(
            chunks[..3].iter().all(|c| !c.done),
            "only the final chunk is done"
        );
    }

    #[test]
    fn tokenize_can_add_the_special_tokens() {
        let engine = FakeEngine::default();
        let model = engine
            .load(Path::new("/models/general.gguf"), &LoadOptions::generate())
            .unwrap();

        let plain = engine.tokenize(model.as_ref(), "one two", false).unwrap();
        let special = engine.tokenize(model.as_ref(), "one two", true).unwrap();

        assert_eq!(plain, vec![3, 3]);
        assert_eq!(special, vec![1, 3, 3, 2]);
    }

    #[test]
    fn a_failing_engine_reports_which_file_it_could_not_open() {
        let engine = FakeEngine {
            fail: true,
            ..Default::default()
        };
        // `unwrap_err` is unavailable here: `Box<dyn ModelHandle>` is not
        // `Debug`, and requiring it would put a formatting burden on every
        // runtime's model type for no gain.
        let Err(err) = engine.load(Path::new("/models/missing.gguf"), &LoadOptions::generate())
        else {
            panic!("a failing engine must not load");
        };

        assert!(err.to_string().contains("missing.gguf"), "got {err}");
        assert!(matches!(err, EngineError::Io { .. }));
    }

    #[test]
    fn release_is_callable_more_than_once() {
        // A caller unwinding after an error cannot know whether cleanup already
        // ran, so releasing twice must not panic.
        let engine = FakeEngine::default();
        let first = engine
            .load(Path::new("/models/general.gguf"), &LoadOptions::generate())
            .unwrap();
        engine.release(Arc::from(first));
        assert_eq!(engine.released.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_handle_that_names_its_type_can_be_given_back_to_its_runtime() {
        // The seam hands implementations `&dyn ModelHandle`, so this is the only
        // way one gets its own handle type back. A runtime that stores nothing
        // leaves it alone.
        let plain = NamedModel::new("general", "/models/general.gguf");
        assert!(plain.as_any().is_none(), "nothing to give back");

        #[derive(Debug)]
        struct Owned(String);

        impl ModelHandle for Owned {
            fn name(&self) -> &str {
                "owned"
            }

            fn path(&self) -> &Path {
                Path::new("/models/owned.gguf")
            }

            fn as_any(&self) -> Option<&dyn std::any::Any> {
                Some(self)
            }
        }

        let owned = Owned(String::new());
        let handle: &dyn ModelHandle = &owned;
        let recovered = handle.as_any().unwrap().downcast_ref::<Owned>();
        assert_eq!(recovered.map(|o| o.0.as_str()), Some(""));

        let wrong: &dyn std::any::Any = handle.as_any().unwrap();
        assert!(
            wrong.downcast_ref::<NamedModel>().is_none(),
            "a downcast must not succeed for the wrong type"
        );
    }

    #[test]
    fn errors_read_as_sentences_about_the_model() {
        let err = EngineError::NotAModel {
            path: PathBuf::from("/models/broken.gguf"),
            reason: "no GGUF header".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "/models/broken.gguf is not a loadable model: no GGUF header"
        );

        let err = EngineError::Unsupported {
            model: "route".to_string(),
            operation: "generate tokens".to_string(),
        };
        assert_eq!(err.to_string(), "'route' cannot generate tokens");

        let err = EngineError::Generation {
            model: "general".to_string(),
            reason: "context overflow".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "generation failed on 'general': context overflow"
        );
    }
}
