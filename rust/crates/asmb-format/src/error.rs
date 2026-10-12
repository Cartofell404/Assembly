//! Errors returned by the `.asmb` reader and writer.

use std::fmt;

/// Everything that can go wrong reading or writing a `.asmb` archive.
///
/// The variants carry messages meant for a human reading a terminal. A
/// truncated or corrupt archive should fail here, once, with one readable
/// line -- not later as a `serde` error, a slicing panic, or an out-of-range
/// index.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The file is not a `.asmb` archive, or its magic bytes are wrong.
    #[error("not an .asmb archive: expected magic {expected:?}, found {found:?}")]
    BadMagic {
        /// The magic every archive starts with.
        expected: [u8; 4],
        /// What the first four bytes actually were.
        found: [u8; 4],
    },

    /// The archive declares a format version this build does not implement.
    #[error(
        "unsupported .asmb format version {found}; this build understands version {supported}"
    )]
    UnsupportedVersion {
        /// The version stamped in the preamble.
        found: u32,
        /// The version this build knows how to read.
        supported: u32,
    },

    /// The file ended before a structure the header promised was complete.
    #[error("archive is truncated: {0}")]
    Truncated(String),

    /// The header payload is not valid JSON, or is missing a required field.
    #[error("header is not valid JSON: {0}")]
    BadJson(String),

    /// A declared offset or size does not fit inside the archive.
    #[error("{0}")]
    Layout(String),

    /// Anchor vectors could not be decoded.
    #[error("{0}")]
    Anchors(String),

    /// A recorded SHA-256 did not match the bytes on disk.
    #[error("checksum mismatch for '{name}': header says {expected}, bytes hash to {actual}")]
    ChecksumMismatch {
        /// Model or embedding name the checksum belongs to.
        name: String,
        /// The hex digest recorded in the header.
        expected: String,
        /// The hex digest computed from the blob.
        actual: String,
    },

    /// Reading or writing the underlying file failed.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialisation failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

impl Error {
    /// Shorthand for a [`Error::Layout`].
    pub(crate) fn layout(msg: impl fmt::Display) -> Self {
        Error::Layout(msg.to_string())
    }

    /// Shorthand for a [`Error::Anchors`].
    pub(crate) fn anchors(msg: impl fmt::Display) -> Self {
        Error::Anchors(msg.to_string())
    }

    /// Shorthand for a [`Error::Truncated`].
    pub(crate) fn truncated(msg: impl fmt::Display) -> Self {
        Error::Truncated(msg.to_string())
    }
}

/// Convenience alias for results from this crate.
pub type Result<T> = std::result::Result<T, Error>;
