//! A local OpenAI-compatible HTTP server for Assembly archives.
//!
//! The crate is deliberately small: five endpoints, one archive, loopback by
//! default. See [`serve`] for the endpoint table and the reasoning behind what
//! is *not* exposed.

pub mod serve;
pub mod settings;
pub mod types;

pub use serve::{serve, RunningServer, ServeError, ENDPOINTS};
pub use settings::{
    Settings, SettingsError, DEFAULT_HEADROOM, DEFAULT_HOST, DEFAULT_MODEL_CAP, DEFAULT_PORT,
    DEFAULT_WORKERS, MAX_BODY_BYTES,
};
pub use types::{ChatRequest, ChatResponse, ErrorBody, ErrorDetail, Message, Role};
