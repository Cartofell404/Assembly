//! Core Assembly logic: build configs, staging, resident-model accounting,
//! and routing.
//!
//! This crate has no llama.cpp dependency. It knows about archives, what to
//! keep on disk, what to keep in memory, and which model should answer a
//! prompt. Binding an inference engine is a later concern, behind the
//! [`ResidentCache`] that decides which models are allowed to be loaded at
//! all.
//!
//! The pieces fit together as:
//!
//! ```text
//!   Archive (asmb-format)  ──▶  StagingCache  ──▶  path llama.cpp can open
//!          │
//!          ├──▶  Router  ──▶  which model answers
//!          │
//!          └──▶  ResidentCache  ──▶  which models may stay in RAM
//! ```
//!
//! ```
//! use assembly_core::resident::{Resident, ResidentCache};
//! use assembly_core::staging::{ArchiveId, StagedKey, StagingCache};
//!
//! // Staging: extract once, reuse many times.
//! let staging = StagingCache::with_limit(2 * 1024 * 1024)?;
//! let archive = ArchiveId::new("/archives/Mesh.asmb", 1024);
//! let key = StagedKey::new(archive, 4096, 1024);
//!
//! // Residency: keep at most 1 GiB of models warm.
//! let mut resident = ResidentCache::with_byte_budget(1024 * 1024 * 1024);
//! resident.insert(Resident { name: "general".into(), bytes: 512 << 20, staged_path: None });
//!
//! // In-use models are never evicted.
//! let guard = resident.acquire("general")?;
//! assert!(resident.is_busy(guard.name()));
//! drop(guard);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]

pub mod config;
pub mod engine;
pub mod manager;
pub mod resident;
pub mod router;
pub mod staging;

pub use config::{
    validate as validate_config, AssemblyConfig, EmbeddingConfig, Issue, ModelConfig, ModelInfo,
    Severity, ValidationReport,
};
pub use engine::{Chunk, Engine, EngineError, EngineOptions, LoadOptions, Message, Messages, Role};
pub use manager::{Assembly, ChatRequest, ChatResponse, ManagerError};
pub use resident::{CacheError, Resident, ResidentCache, ResidentGuard, DEFAULT_MODEL_CAP};
pub use router::{RouteResult, Router};
pub use staging::{ArchiveId, ConfigError, StagedKey, StagingCache};

/// Version string for the core crate, matching the workspace version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
