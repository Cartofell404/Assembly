//! The JSON header that lives inside a `.asmb` archive.

use serde::{Deserialize, Serialize};

use crate::anchors;
use crate::error::{Error, Result};

/// Magic bytes at offset zero.
pub const MAGIC: [u8; 4] = *b"ASMB";
/// The container format version this build reads and writes.
pub const FORMAT_VERSION: u32 = 1;
/// GGUF blobs start on multiples of this.
pub const PAGE_SIZE: usize = 4096;
/// `MAGIC(4) + VERSION(4) + HEADER_LEN(8)`.
pub const PREAMBLE_SIZE: usize = 16;

/// Metadata for the router (embedding) model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingEntry {
    /// Model name, as written in the build config.
    pub name: String,
    /// Byte offset of the GGUF blob inside the archive.
    #[serde(default)]
    pub offset: u64,
    /// Blob length in bytes.
    #[serde(default)]
    pub size: u64,
    /// Cosine similarity below which the router stops trusting its pick.
    #[serde(default = "default_threshold")]
    pub confidence_threshold: f32,
    /// Model to use when nothing clears the threshold. `None` means use the
    /// best match anyway.
    #[serde(default)]
    pub fallback_model: Option<String>,
    /// SHA-256 of this blob, hex. Empty means "not recorded", which is what
    /// archives written before checksums were added contain.
    #[serde(default)]
    pub sha256: String,
}

fn default_threshold() -> f32 {
    0.75
}

impl EmbeddingEntry {
    /// A placeholder entry for archives with no embedding model.
    pub fn empty() -> Self {
        EmbeddingEntry {
            name: String::new(),
            offset: 0,
            size: 0,
            confidence_threshold: default_threshold(),
            fallback_model: None,
            sha256: String::new(),
        }
    }

    /// Whether this entry points at a real blob.
    pub fn is_present(&self) -> bool {
        !self.name.is_empty() && self.size > 0
    }
}

/// Metadata for one execution model inside the archive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Model name, as written in the build config.
    pub name: String,
    /// Byte offset of the GGUF blob inside the archive.
    #[serde(default)]
    pub offset: u64,
    /// Blob length in bytes.
    #[serde(default)]
    pub size: u64,
    /// Human-readable phrases this model is meant to answer.
    #[serde(default)]
    pub anchors: Vec<String>,
    /// Anchor vectors: big-endian float32, base64. See [`crate::anchors`].
    #[serde(default)]
    pub anchor_embeddings_b64: String,
    /// Width of each anchor vector.
    #[serde(default)]
    pub anchor_dim: usize,
    /// SHA-256 of this blob, hex. Empty means "not recorded".
    #[serde(default)]
    pub sha256: String,
}

impl ModelEntry {
    /// Decode [`Self::anchor_embeddings_b64`] back into vectors.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Anchors`] if the payload does not match
    /// [`Self::anchor_dim`].
    pub fn anchor_vectors(&self) -> Result<Vec<Vec<f32>>> {
        anchors::decode(&self.anchor_embeddings_b64, self.anchor_dim)
    }

    /// Build an entry from raw vectors, packing them on the way in.
    ///
    /// `vectors` is narrowed to `f32` here, which is what the header stores.
    /// Reading them back yields the narrowed values, so a test comparing
    /// against the original `f32` list is exact while one comparing `f64`
    /// input is not.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Anchors`] if `vectors` is ragged or zero-width.
    pub fn with_vectors(
        name: impl Into<String>,
        vectors: &[Vec<f32>],
        offset: u64,
        size: u64,
        anchor_text: Vec<String>,
        sha256: impl Into<String>,
    ) -> Result<ModelEntry> {
        let block = anchors::encode_with_dim(vectors)?;
        Ok(ModelEntry {
            name: name.into(),
            offset,
            size,
            anchors: anchor_text,
            anchor_embeddings_b64: block.blob,
            anchor_dim: block.dim,
            sha256: sha256.into(),
        })
    }
}

/// The full header stored at the start of a `.asmb` file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsmbHeader {
    /// Archive name, from the build config.
    pub assembly_name: String,
    /// Format version stamped in the preamble.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Alignment every blob starts on.
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    /// The router (embedding) model.
    #[serde(default = "EmbeddingEntry::empty")]
    pub embedding: EmbeddingEntry,
    /// Execution models, in load order.
    #[serde(default)]
    pub models: Vec<ModelEntry>,
    /// Persona prepended to every conversation, if the build set one.
    #[serde(default)]
    pub persona: Option<String>,
}

fn default_version() -> u32 {
    FORMAT_VERSION
}

fn default_page_size() -> usize {
    PAGE_SIZE
}

impl AsmbHeader {
    /// Serialise to compact JSON, UTF-8.
    ///
    /// Compact matters: this payload sits in every archive, so whitespace is
    /// pure overhead.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Json`] if serialisation fails.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(Error::Json)
    }

    /// Parse from JSON bytes.
    ///
    /// Accepts the pre-v1 layout that stored anchor vectors as nested decimal
    /// floats, so an archive built before the binary change still routes. The
    /// legacy vectors are converted to the current form in memory and are
    /// rewritten as base64 when the archive is next built.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Json`] if the payload is not valid JSON or a required
    /// field is missing.
    pub fn from_json_bytes(data: &[u8]) -> Result<AsmbHeader> {
        let mut raw: RawHeader =
            serde_json::from_slice(data).map_err(|e| Error::BadJson(e.to_string()))?;

        let version = raw.version.unwrap_or_else(default_version);
        let page_size = match raw.page_size {
            Some(0) | None => default_page_size(),
            Some(p) => p,
        };
        let embedding = raw.embedding.unwrap_or_else(EmbeddingEntry::empty);

        let mut models = Vec::with_capacity(raw.models.len());
        for m in raw.models.drain(..) {
            // Pre-v1 archives stored decimal floats. Narrow to f32, which is what the
            // binary form stores anyway, so the conversion is lossless with
            // respect to the current format.
            let legacy: Vec<Vec<f32>> = m
                .anchor_embeddings
                .unwrap_or_default()
                .iter()
                .map(|row| row.iter().map(|&v| v as f32).collect())
                .collect();
            let legacy_dim = legacy.first().map_or(0, Vec::len);

            let (blob, dim) = match m.anchor_embeddings_b64 {
                Some(blob) => (blob, m.anchor_dim.unwrap_or(legacy_dim)),
                None => {
                    let block = anchors::encode_with_dim(&legacy)?;
                    (block.blob, block.dim)
                }
            };

            models.push(ModelEntry {
                name: m.name,
                offset: m.offset,
                size: m.size,
                anchors: m.anchors,
                anchor_embeddings_b64: blob,
                anchor_dim: dim,
                sha256: m.sha256,
            });
        }

        Ok(AsmbHeader {
            assembly_name: raw.assembly_name,
            version,
            page_size,
            embedding,
            models,
            persona: raw.persona,
        })
    }
}

/// Wire shape before legacy conversion.
///
/// Optional where a value may be absent from an older archive; `page_size`
/// and `version` fall back to the current constants.
#[derive(Debug, Deserialize)]
struct RawHeader {
    assembly_name: String,
    #[serde(default)]
    version: Option<u32>,
    #[serde(default)]
    page_size: Option<usize>,
    #[serde(default)]
    embedding: Option<EmbeddingEntry>,
    #[serde(default)]
    models: Vec<RawModel>,
    #[serde(default)]
    persona: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawModel {
    name: String,
    #[serde(default)]
    offset: u64,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    anchors: Vec<String>,
    #[serde(default)]
    anchor_embeddings_b64: Option<String>,
    #[serde(default)]
    anchor_dim: Option<usize>,
    #[serde(default)]
    sha256: String,
    /// Pre-v1 nested-float vectors.
    #[serde(default)]
    anchor_embeddings: Option<Vec<Vec<f64>>>,
}

impl RawHeader {
    #[allow(dead_code)]
    fn version_or_default(&self) -> u32 {
        self.version.unwrap_or_else(default_version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> AsmbHeader {
        AsmbHeader {
            assembly_name: "Mesh-1.0".into(),
            version: FORMAT_VERSION,
            page_size: PAGE_SIZE,
            embedding: EmbeddingEntry {
                name: "router".into(),
                offset: 4096,
                size: 500,
                confidence_threshold: 0.7,
                fallback_model: Some("general".into()),
                sha256: "ab".into(),
            },
            models: vec![ModelEntry::with_vectors(
                "general",
                &[vec![1.0, 0.0]],
                8192,
                900,
                vec!["chit chat".into()],
                "cd",
            )
            .unwrap()],
            persona: Some("be brief".into()),
        }
    }

    #[test]
    fn json_round_trips() {
        let header = sample();
        let bytes = header.to_json_bytes().unwrap();
        assert_eq!(AsmbHeader::from_json_bytes(&bytes).unwrap(), header);
    }

    #[test]
    fn json_is_compact() {
        let bytes = sample().to_json_bytes().unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(!text.contains('\n'), "header JSON should be one line");
        assert!(
            !text.contains(": "),
            "header JSON should have no pretty spacing"
        );
    }

    #[test]
    fn json_carries_every_field() {
        let text = String::from_utf8(sample().to_json_bytes().unwrap()).unwrap();
        for key in [
            "assembly_name",
            "version",
            "page_size",
            "persona",
            "embedding",
            "models",
            "anchor_embeddings_b64",
            "anchor_dim",
            "sha256",
            "confidence_threshold",
            "fallback_model",
        ] {
            assert!(text.contains(key), "missing {key} in {text}");
        }
    }

    #[test]
    fn reads_legacy_nested_float_anchors() {
        // The pre-v1 layout: vectors as decimal lists, no anchor_dim.
        let legacy = br#"{
            "assembly_name": "Old",
            "version": 1,
            "page_size": 4096,
            "embedding": {"name": "router", "offset": 0, "size": 10},
            "models": [
                {"name": "general", "offset": 4096, "size": 20,
                 "anchors": ["hi"],
                 "anchor_embeddings": [[1.0, 0.0], [0.0, 1.0]]}
            ],
            "persona": null
        }"#;

        let header = AsmbHeader::from_json_bytes(legacy).unwrap();
        let m = &header.models[0];
        assert_eq!(m.anchor_dim, 2);
        assert!(!m.anchor_embeddings_b64.is_empty());
        assert_eq!(
            m.anchor_vectors().unwrap(),
            vec![vec![1.0, 0.0], vec![0.0, 1.0]]
        );
    }

    #[test]
    fn legacy_read_then_write_uses_base64() {
        let legacy = br#"{
            "assembly_name": "Old",
            "embedding": {"name": "router"},
            "models": [{"name": "g", "anchor_embeddings": [[1.0, 2.0]]}]
        }"#;
        let header = AsmbHeader::from_json_bytes(legacy).unwrap();
        let text = String::from_utf8(header.to_json_bytes().unwrap()).unwrap();
        assert!(text.contains("anchor_embeddings_b64"));
        assert!(!text.contains("anchor_embeddings\":"), "got {text}");
    }

    #[test]
    fn absent_optional_fields_take_defaults() {
        let minimal = br#"{"assembly_name": "Bare", "embedding": {"name": ""}, "models": []}"#;
        let header = AsmbHeader::from_json_bytes(minimal).unwrap();
        assert_eq!(header.page_size, PAGE_SIZE);
        assert_eq!(header.version, FORMAT_VERSION);
        assert_eq!(header.embedding.confidence_threshold, 0.75);
        assert!(header.persona.is_none());
    }

    #[test]
    fn rejects_json_that_is_not_json() {
        let err = AsmbHeader::from_json_bytes(b"{nope").unwrap_err();
        assert!(matches!(err, Error::BadJson(_)), "got {err:?}");
    }

    #[test]
    fn empty_embedding_is_not_present() {
        assert!(!EmbeddingEntry::empty().is_present());
    }
}
