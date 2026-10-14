//! Reader and writer for `.asmb`, the Assembly Mesh Binary container.
//!
//! A `.asmb` file bundles several GGUF models, the routing vectors that pick
//! between them, and the SHA-256 of every blob, behind one page-aligned
//! header. Offsets live in that header, so a reader can memory-map exactly one
//! model instead of the whole archive.
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
//! # Example
//!
//! ```no_run
//! use asmb_format::Archive;
//!
//! let mut archive = Archive::open("Mesh-1.2.asmb")?;
//! for model in archive.models() {
//!     println!("{} at {:#x} ({} bytes)", model.name, model.offset, model.size);
//! }
//! # Ok::<(), asmb_format::Error>(())
//! ```

#![deny(missing_docs)]

// Assembly is built for macOS on Apple Silicon only. Every crate in the
// workspace depends on this one, so a single guard covers them all.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("Assembly supports only macOS on Apple Silicon (aarch64-apple-darwin)");

pub mod anchors;
pub mod checksum;
pub mod error;
pub mod gguf;
pub mod header;
pub mod read;
pub mod write;

pub use anchors::{
    decode as decode_anchors, encode as encode_anchors, encode_with_dim, AnchorBlock,
};
pub use checksum::{hash_bytes, hash_file, hash_file_region, hash_region, HASH_CHUNK};
pub use error::{Error, Result};
pub use gguf::{Metadata as GgufMetadata, Value as GgufValue, GGUF_MAGIC};
pub use header::{
    AsmbHeader, EmbeddingEntry, ModelEntry, FORMAT_VERSION, MAGIC, PAGE_SIZE, PREAMBLE_SIZE,
};
pub use read::{
    align_to_page, read_header, read_preamble, Archive, BlobVerification, VerificationReport,
};
pub use write::{BlobSource, BuildPlan, BuildReport, ModelSource, PlacedBlob, RouterSettings};

/// Version string for the format crate, matching the workspace version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
