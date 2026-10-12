//! Anchor vector packing.
//!
//! Anchor vectors are stored as **big-endian float32**, row-major, with all
//! vectors concatenated, then base64-encoded into a JSON string.
//!
//! Why this encoding:
//!
//! * The first attempt stored vectors as nested JSON decimal floats. A
//!   4096-dim vector costs roughly 40 KB that way, which is about twice the
//!   bytes of the model it routes to. Big-endian float32 is 16 KB.
//! * float32 is deliberate. The vectors come straight out of the embedder,
//!   so they are already float32 values; narrowing from float64 would lose
//!   routing accuracy and save nothing meaningful.
//! * Big-endian, not little-endian, so a hex dump of the payload reads in the
//!   same order on every machine regardless of the host CPU.

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;

use crate::error::{Error, Result};

/// Bytes per stored anchor component.
pub const ANCHOR_DTYPE_SIZE: usize = 4;

/// A block of anchor vectors plus the row width they were packed with.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AnchorBlock {
    /// Base64 payload, ready to drop into a header field.
    pub blob: String,
    /// Width of each vector, i.e. the embedder's dimension.
    pub dim: usize,
}

/// Pack vectors into the base64 form, recording the row width.
///
/// The width travels with the payload instead of being guessed from a
/// global constant because it is a property of whichever embedder produced
/// the vectors -- 384 for one model, 1024 for the next.
///
/// # Errors
///
/// Returns [`Error::Anchors`] when `vectors` is empty with a zero-width row,
/// or when the rows are ragged. Ragged input is a caller bug and is rejected
/// rather than silently padded, because a mis-shaped matrix would route
/// against the wrong columns.
pub fn encode_with_dim(vectors: &[Vec<f32>]) -> Result<AnchorBlock> {
    let Some(first) = vectors.first() else {
        return Ok(AnchorBlock::default());
    };
    let dim = first.len();
    if dim == 0 {
        return Err(Error::anchors("anchor vectors have zero dimensions"));
    }

    let mut packed = Vec::with_capacity(vectors.len() * dim * ANCHOR_DTYPE_SIZE);
    for vec in vectors {
        if vec.len() != dim {
            return Err(Error::anchors(format!(
                "anchor vectors are ragged: expected {dim} dimensions, got {}",
                vec.len()
            )));
        }
        for &v in vec {
            packed.extend_from_slice(&v.to_be_bytes());
        }
    }

    Ok(AnchorBlock {
        blob: STANDARD.encode(packed),
        dim,
    })
}

/// Pack vectors into base64, discarding the row width.
///
/// The row width is written to the header separately via
/// [`encode_with_dim`]; this is for tests and one-off callers that already
/// know it.
///
/// # Errors
///
/// Same as [`encode_with_dim`].
pub fn encode(vectors: &[Vec<f32>]) -> Result<String> {
    Ok(encode_with_dim(vectors)?.blob)
}

/// Unpack the base64 form back into vectors of width `dim`.
///
/// `dim` is the width recorded in the header at build time. It is trusted
/// here rather than re-derived, because the payload alone is ambiguous: any
/// multiple of four divides evenly into some row width.
///
/// # Errors
///
/// Returns [`Error::Anchors`] when the payload is not valid base64, when
/// `dim` is not positive, or when the byte length does not divide evenly
/// into `dim` float32 values.
pub fn decode(blob: &str, dim: usize) -> Result<Vec<Vec<f32>>> {
    if blob.is_empty() {
        return Ok(Vec::new());
    }

    let packed = STANDARD
        .decode(blob)
        .map_err(|e| Error::anchors(format!("anchor embeddings are not valid base64: {e}")))?;

    if dim == 0 {
        return Err(Error::anchors(
            "anchor embeddings declare an embedding dimension of 0",
        ));
    }
    if packed.len() % ANCHOR_DTYPE_SIZE != 0 {
        return Err(Error::anchors(format!(
            "anchor embeddings are {} bytes, which is not a whole number of {ANCHOR_DTYPE_SIZE}-byte float32 values",
            packed.len()
        )));
    }

    let count = packed.len() / ANCHOR_DTYPE_SIZE;
    if count % dim != 0 {
        return Err(Error::anchors(format!(
            "anchor embeddings hold {count} values, which is not divisible by the recorded embedding dimension {dim}"
        )));
    }

    let mut out = Vec::with_capacity(count / dim);
    for chunk in packed.chunks_exact(dim * ANCHOR_DTYPE_SIZE) {
        let row = chunk
            .chunks_exact(ANCHOR_DTYPE_SIZE)
            .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.push(row);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_vectors() {
        let vectors = vec![vec![1.0, 0.0, -1.0], vec![0.5, 0.25, 0.125]];
        let block = encode_with_dim(&vectors).unwrap();
        assert_eq!(block.dim, 3);
        assert_eq!(decode(&block.blob, block.dim).unwrap(), vectors);
    }

    #[test]
    fn empty_input_is_empty_output() {
        let block = encode_with_dim(&[]).unwrap();
        assert_eq!(block.blob, "");
        assert_eq!(block.dim, 0);
        assert!(decode("", 0).unwrap().is_empty());
    }

    #[test]
    fn rejects_ragged_rows() {
        let err = encode_with_dim(&[vec![1.0, 2.0], vec![1.0]]).unwrap_err();
        assert!(matches!(err, Error::Anchors(_)), "got {err:?}");
        assert!(err.to_string().contains("ragged"));
    }

    #[test]
    fn rejects_zero_width_rows() {
        let err = encode_with_dim(&[vec![]]).unwrap_err();
        assert!(err.to_string().contains("zero dimensions"));
    }

    #[test]
    fn rejects_payload_that_does_not_divide_by_dim() {
        // Two float32 values, claimed width of three.
        let block = encode(&[vec![1.0, 2.0]]).unwrap();
        let err = decode(&block, 3).unwrap_err();
        assert!(err.to_string().contains("not divisible"), "got {err}");
    }

    #[test]
    fn rejects_truncated_float_run() {
        // 6 bytes is not a whole number of float32 values.
        let blob = STANDARD.encode([0u8; 6]);
        let err = decode(&blob, 1).unwrap_err();
        assert!(err.to_string().contains("not a whole number"), "got {err}");
    }

    #[test]
    fn rejects_non_base64() {
        let err = decode("not base64 !!", 1).unwrap_err();
        assert!(err.to_string().contains("not valid base64"), "got {err}");
    }

    #[test]
    fn byte_order_is_big_endian() {
        // 1.0f32 == 0x3F800000, and big-endian puts the high byte first.
        let blob = encode(&[vec![1.0]]).unwrap();
        let raw = STANDARD.decode(&blob).unwrap();
        assert_eq!(raw, vec![0x3F, 0x80, 0x00, 0x00]);
    }
}
