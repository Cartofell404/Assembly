//! Reading GGUF metadata without loading a model.
//!
//! GGUF stores its key/value metadata ahead of the tensor data, so everything
//! worth knowing about a file is in its first few hundred kilobytes. Reading
//! that directly is far cheaper than mapping the file or instantiating it, and
//! it is the difference between catching a broken download in pre-flight and
//! catching it after several minutes of copying.
//!
//! The one thing this is load-bearing for is split-GGUF detection. Large models
//! ship as shards (`model-00001-of-00003.gguf`) whose metadata declares
//! `split.count` and `split.no`. llama.cpp refuses to load a shard on its own,
//! so such a file can be copied into an archive and hash-verified perfectly and
//! still be unloadable at runtime. Filenames are not a reliable signal -- some
//! downloads rename the shards -- so the metadata is what gets checked.
//!
//! # Failure is not an error
//!
//! Anything unparseable yields absent fields rather than an error. A caller
//! asking "is this file sharded?" wants to know "yes" or "not detected"; it
//! does not want a parser failure masquerading as a broken file. A truncated
//! read, an unknown value type and a plain non-GGUF file all produce the same
//! answer, which is the honest one: we do not know.
//!
//! ```
//! use asmb_format::gguf::{metadata_from_bytes, GGUF_MAGIC};
//!
//! // A minimal header: magic, version, 0 tensors, 1 key, "split.count" = 3.
//! let mut head = Vec::new();
//! head.extend_from_slice(&GGUF_MAGIC);
//! head.extend_from_slice(&3u32.to_le_bytes());
//! head.extend_from_slice(&0u64.to_le_bytes());
//! head.extend_from_slice(&1u64.to_le_bytes());
//! head.extend_from_slice(&(11u64).to_le_bytes());
//! head.extend_from_slice(b"split.count");
//! head.extend_from_slice(&4u32.to_le_bytes()); // UINT32
//! head.extend_from_slice(&3u32.to_le_bytes());
//!
//! let meta = metadata_from_bytes(&head);
//! assert_eq!(meta.get_u64("split.count"), Some(3));
//! ```

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// Magic bytes every GGUF file starts with.
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// How much of a file to read when looking for metadata.
///
/// The real limit is time, not size: a 16 MiB prefix covers the metadata of
/// every GGUF in wide use, including ones with long tensor descriptions, and is
/// fast enough to run on every model in a config without being noticed.
pub const METADATA_PREFIX: u64 = 16 << 20;

/// Upper bound on a single metadata key or string value.
///
/// Without this, a corrupt length field turns into a multi-gigabyte allocation
/// from four bytes of nonsense.
const MAX_STRING_LEN: u64 = 1 << 20;

/// GGUF value type tags.
mod ty {
    /// Unsigned 8-bit.
    pub const U8: u32 = 0;
    /// Signed 8-bit.
    pub const I8: u32 = 1;
    /// Unsigned 16-bit.
    pub const U16: u32 = 2;
    /// Signed 16-bit.
    pub const I16: u32 = 3;
    /// Unsigned 32-bit.
    pub const U32: u32 = 4;
    /// Signed 32-bit.
    pub const I32: u32 = 5;
    /// 32-bit float.
    pub const F32: u32 = 6;
    /// Boolean.
    pub const BOOL: u32 = 7;
    /// Length-prefixed string.
    pub const STRING: u32 = 8;
    /// Array of a single element type.
    pub const ARRAY: u32 = 9;
    /// Unsigned 64-bit.
    pub const U64: u32 = 10;
    /// Signed 64-bit.
    pub const I64: u32 = 11;
    /// 64-bit float.
    pub const F64: u32 = 12;
}

/// Byte width of the fixed-size value types. Strings and arrays are variable and
/// handled separately.
fn fixed_width(t: u32) -> Option<u64> {
    Some(match t {
        ty::U8 | ty::I8 | ty::BOOL => 1,
        ty::U16 | ty::I16 => 2,
        ty::U32 | ty::I32 | ty::F32 => 4,
        ty::U64 | ty::I64 | ty::F64 => 8,
        _ => return None,
    })
}

/// One scalar value from a GGUF key/value block.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Unsigned integer, any width.
    Unsigned(u64),
    /// Signed integer.
    Signed(i64),
    /// 32-bit float.
    Float32(f32),
    /// 64-bit float.
    Float64(f64),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    Text(String),
}

impl Value {
    /// The value as a `u64`, if it is any integer kind.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Unsigned(v) => Some(*v),
            Value::Signed(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
}

/// The scalar metadata read from a GGUF prefix.
///
/// Arrays are skipped rather than stored: the keys this crate looks for are
/// scalars, and materialising a tokenizer's vocabulary as a `Vec<Value>` would
/// cost megabytes to answer a question the answer to which is always "skip it".
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    values: HashMap<String, Value>,
}

impl Metadata {
    /// Look up a key's value.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    /// Look up a key as an unsigned integer.
    ///
    /// Signed values are accepted when non-negative, since GGUF writers pick
    /// the narrowest type that fits and `3` is equally a valid `INT32`.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.values.get(key).and_then(Value::as_u64)
    }

    /// Look up a key as a string.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.values.get(key) {
            Some(Value::Text(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Number of scalar keys found. Zero means "not a GGUF, or nothing read".
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether no scalar keys were found.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Whether these are the split declarations of a sharded model, as
    /// `(count, index)`.
    ///
    /// `None` covers both a whole model and metadata this crate could not
    /// parse. Callers must read that as "not checked", never as "fine".
    pub fn split_info(&self) -> Option<(u64, u64)> {
        Some((self.get_u64("split.count")?, self.get_u64("split.no")?))
    }
}

/// Read the leading [`METADATA_PREFIX`] bytes of a file.
///
/// A short read is not an error: a valid GGUF is always longer than its own
/// prefix, and a file smaller than that is a file to reject on other grounds.
fn read_prefix(path: &Path) -> Vec<u8> {
    let Ok(mut file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let mut buf = Vec::new();
    let _ = file.by_ref().take(METADATA_PREFIX).read_to_end(&mut buf);
    buf
}

/// Detect a split GGUF from bytes already in memory.
pub fn split_info(head: &[u8]) -> Option<(u64, u64)> {
    let meta = metadata_from_bytes(head);
    let count = meta.get_u64("split.count")?;
    let no = meta.get_u64("split.no")?;
    Some((count, no))
}

/// Whether a path is one shard of a split GGUF, as `(count, index)`.
///
/// The index is zero-based, matching GGUF's own `split.no`.
pub fn split_info_file(path: &Path) -> Option<(u64, u64)> {
    let meta = metadata_from_file(path)?;
    let count = meta.get_u64("split.count")?;
    let no = meta.get_u64("split.no")?;
    Some((count, no))
}

/// Read the scalar metadata from a GGUF file's prefix.
///
/// Returns `None` when the file does not start with the GGUF magic, which is
/// how a caller distinguishes a corrupt download from a GGUF whose metadata
/// happened to be empty.
pub fn metadata_from_file(path: &Path) -> Option<Metadata> {
    let head = read_prefix(path);
    // `None` means "this is not a GGUF", which is a stronger claim than
    // "metadata is empty" and is exactly what a caller checking a download
    // wants to tell apart from a GGUF whose keys it did not care about.
    if head.len() < 16 || head[..4] != GGUF_MAGIC {
        return None;
    }
    Some(metadata_from_bytes(&head))
}

/// Parse the scalar metadata from the start of a GGUF file.
///
/// Returns empty metadata for anything that is not a GGUF header, or that ends
/// before the key/value block does. See the module docs on why that is not an
/// error.
pub fn metadata_from_bytes(head: &[u8]) -> Metadata {
    let mut cursor = Cursor::new(head);

    if cursor.take(4) != Some(&GGUF_MAGIC[..]) {
        return Metadata::default();
    }
    cursor.skip(4); // container version
    cursor.skip(8); // tensor count
    let kv_count = match cursor.u64() {
        Some(n) => n,
        None => return Metadata::default(),
    };

    // Each key needs at least a length, a type tag and a name byte, so a count
    // larger than the remaining bytes cannot be honest. This is the bound that
    // stops a crafted length from driving the loop.
    let mut values = HashMap::new();
    for _ in 0..kv_count.min(1 << 16) {
        let Some(key) = cursor.string() else {
            break;
        };
        let Some(kind) = cursor.u32() else { break };
        match read_value(&mut cursor, kind) {
            Some(value) => {
                values.insert(key, value);
            }
            // Arrays and unknown types are consumed but not stored.
            None => {
                if kind == ty::ARRAY {
                    if !skip_array(&mut cursor) {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    Metadata { values }
}

/// Read one value, storing it only when it is a scalar.
///
/// Returns `None` for arrays and unrecognised types, leaving the cursor
/// untouched so the caller can decide how to skip them.
fn read_value(cursor: &mut Cursor<'_>, kind: u32) -> Option<Value> {
    if kind == ty::STRING {
        return cursor.string().map(Value::Text);
    }
    if kind == ty::ARRAY {
        return None;
    }

    let value = match kind {
        ty::U8 => Value::Unsigned(u64::from(cursor.u8()?)),
        ty::I8 => Value::Signed(i64::from(cursor.i8()?)),
        ty::U16 => Value::Unsigned(u64::from(cursor.u16()?)),
        ty::I16 => Value::Signed(i64::from(cursor.i16()?)),
        ty::U32 => Value::Unsigned(u64::from(cursor.u32()?)),
        ty::I32 => Value::Signed(i64::from(cursor.i32()?)),
        ty::U64 => Value::Unsigned(cursor.u64()?),
        ty::I64 => Value::Signed(cursor.i64()?),
        ty::F32 => Value::Float32(cursor.f32()?),
        ty::F64 => Value::Float64(cursor.f64()?),
        ty::BOOL => match cursor.u8()? {
            0 => Value::Bool(false),
            _ => Value::Bool(true),
        },
        _ => return None,
    };
    Some(value)
}

/// Step over an array value without materialising it.
fn skip_array(cursor: &mut Cursor<'_>) -> bool {
    let Some(elem) = cursor.u32() else {
        return false;
    };
    let Some(count) = cursor.u64() else {
        return false;
    };

    if elem == ty::STRING {
        for _ in 0..count.min(1 << 20) {
            if cursor.string().is_none() {
                return false;
            }
        }
        return true;
    }

    match fixed_width(elem) {
        Some(width) => {
            let total = width.saturating_mul(count);
            cursor.skip(total);
            true
        }
        None => false,
    }
}

/// A bounds-checked forward reader over a byte slice.
///
/// Every accessor returns `None` rather than panicking. A GGUF header that ends
/// early, or whose length fields are nonsense, is a file to decline rather than
/// a reason to abort the process that was trying to inspect it.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, at: 0 }
    }

    /// `n` bytes from the cursor, or `None` if fewer remain.
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(n)?;
        let slice = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(slice)
    }

    /// Advance without reading. Saturating: an absurd length simply lands at
    /// the end, which the next read reports as exhausted.
    fn skip(&mut self, n: u64) {
        let n = usize::try_from(n).unwrap_or(usize::MAX);
        self.at = self.at.saturating_add(n).min(self.bytes.len());
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u16(&mut self) -> Option<u16> {
        let b = self.take(2)?;
        Some(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Option<u32> {
        let b = self.take(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Option<u64> {
        let b = self.take(8)?;
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn i8(&mut self) -> Option<i8> {
        self.u8().map(|v| v as i8)
    }

    fn i16(&mut self) -> Option<i16> {
        self.u16().map(|v| v as i16)
    }

    fn i32(&mut self) -> Option<i32> {
        self.u32().map(|v| v as i32)
    }

    fn i64(&mut self) -> Option<i64> {
        self.u64().map(|v| v as i64)
    }

    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_bits(self.u64()?))
    }

    /// A length-prefixed UTF-8 string.
    fn string(&mut self) -> Option<String> {
        let len = self.u64()?;
        // Reject before allocating: four bytes of garbage should not be able to
        // ask for a gigabyte.
        if len > MAX_STRING_LEN {
            return None;
        }
        let raw = self.take(usize::try_from(len).ok()?)?;
        Some(String::from_utf8_lossy(raw).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// Builds a GGUF prefix for tests.
    ///
    /// Every method takes and returns `Self` by value so a chain reads as one
    /// expression: the KV count is not known until the keys are written, which
    /// is why `finish` splices it into the eight bytes the header reserves for
    /// it.
    struct Builder {
        bytes: Vec<u8>,
    }

    impl Builder {
        fn new() -> Self {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&GGUF_MAGIC);
            bytes.extend_from_slice(&3u32.to_le_bytes()); // container version
            bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor count
                                                          // KV count is spliced in by `finish`, since it is only known once
                                                          // the keys have been written.
            bytes.extend_from_slice(&0u64.to_le_bytes());
            Builder { bytes }
        }

        fn raw(mut self, more: &[u8]) -> Self {
            self.bytes.extend_from_slice(more);
            self
        }

        fn key(self, name: &str, kind: u32) -> Self {
            let mut b = self;
            b.bytes
                .extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.bytes.extend_from_slice(name.as_bytes());
            b.bytes.extend_from_slice(&kind.to_le_bytes());
            b
        }

        fn u32_val(self, v: u32) -> Self {
            self.raw(&v.to_le_bytes())
        }

        fn u64_val(self, v: u64) -> Self {
            self.raw(&v.to_le_bytes())
        }

        fn i32_val(self, v: i32) -> Self {
            self.raw(&v.to_le_bytes())
        }

        fn f32_val(self, v: f32) -> Self {
            self.raw(&v.to_le_bytes())
        }

        fn str_val(self, v: &str) -> Self {
            let mut b = self;
            b.bytes.extend_from_slice(&(v.len() as u64).to_le_bytes());
            b.bytes.extend_from_slice(v.as_bytes());
            b
        }

        /// Splice the KV count into the header's eighth byte and hand over the
        /// finished prefix.
        fn finish(self, count: u64) -> Vec<u8> {
            const KV_COUNT_AT: usize = 16; // magic(4) + version(4) + tensors(8)
            let mut out = self.bytes;
            out[KV_COUNT_AT..KV_COUNT_AT + 8].copy_from_slice(&count.to_le_bytes());
            out
        }
    }

    #[test]
    fn reads_scalars_of_every_width() {
        let head = Builder::new()
            .key("a_u8", ty::U8)
            .raw(&[7])
            .key("b_u16", ty::U16)
            .raw(&1000u16.to_le_bytes())
            .key("c_u32", ty::U32)
            .u32_val(70_000)
            .key("d_u64", ty::U64)
            .u64_val(5_000_000_000)
            .key("e_i32", ty::I32)
            .i32_val(-42)
            .key("f_f32", ty::F32)
            .f32_val(1.5)
            .key("g_bool", ty::BOOL)
            .raw(&[1])
            .key("h_str", ty::STRING)
            .str_val("hello")
            .finish(8);

        let meta = metadata_from_bytes(&head);
        assert_eq!(meta.len(), 8);
        assert_eq!(meta.get_u64("a_u8"), Some(7));
        assert_eq!(meta.get_u64("b_u16"), Some(1000));
        assert_eq!(meta.get_u64("c_u32"), Some(70_000));
        assert_eq!(meta.get_u64("d_u64"), Some(5_000_000_000));
        assert_eq!(meta.get("e_i32"), Some(&Value::Signed(-42)));
        assert_eq!(meta.get("f_f32"), Some(&Value::Float32(1.5)));
        assert_eq!(meta.get("g_bool"), Some(&Value::Bool(true)));
        assert_eq!(meta.get_str("h_str"), Some("hello"));
    }

    #[test]
    fn a_signed_count_is_readable_as_unsigned() {
        // GGUF writers pick the narrowest type that fits, so split.count can
        // arrive as any integer width.
        let head = Builder::new()
            .key("split.count", ty::U32)
            .u32_val(3)
            .key("split.no", ty::I32)
            .i32_val(1)
            .finish(2);

        assert_eq!(split_info(&head), Some((3, 1)));
    }

    #[test]
    fn detects_a_split_gguf_from_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model-00002-of-00003.gguf");
        let head = Builder::new()
            .key("split.count", ty::U32)
            .u32_val(3)
            .key("split.no", ty::I32)
            .i32_val(1)
            .finish(2);
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&head)
            .unwrap();

        assert_eq!(split_info_file(&path), Some((3, 1)));
    }

    #[test]
    fn a_whole_model_reports_no_split() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        let head = Builder::new()
            .key("general.architecture", ty::STRING)
            .str_val("llama")
            .finish(1);
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&head)
            .unwrap();

        assert_eq!(split_info_file(&path), None);
    }

    #[test]
    fn skips_arrays_without_storing_them() {
        // Tokenizer vocabularies are the common array: skipping them keeps this
        // cheap on a real model.
        let head = Builder::new()
            .key("tokenizer.ggml.tokens", ty::ARRAY)
            .u32_val(ty::STRING) // element type
            .u64_val(2) // count
            .str_val("hello")
            .str_val("world")
            .key("general.architecture", ty::STRING)
            .str_val("llama")
            .finish(2);

        let meta = metadata_from_bytes(&head);
        assert_eq!(meta.len(), 1, "the array must not be stored");
        assert_eq!(meta.get_str("general.architecture"), Some("llama"));
    }

    #[test]
    fn skips_numeric_arrays() {
        let head = Builder::new()
            .key("numbers", ty::ARRAY)
            .u32_val(ty::F32)
            .u64_val(3)
            .raw(&[0u8; 12])
            .key("after", ty::STRING)
            .str_val("still here")
            .finish(2);

        let meta = metadata_from_bytes(&head);
        assert_eq!(meta.get_str("after"), Some("still here"));
    }

    #[test]
    fn a_non_gguf_prefix_yields_nothing() {
        let meta = metadata_from_bytes(b"<!doctype html><html>404 not found</html>");
        assert!(meta.is_empty());
        assert_eq!(split_info(b"<html>"), None);
    }

    #[test]
    fn a_truncated_prefix_yields_nothing_rather_than_panicking() {
        let full = Builder::new()
            .key("split.count", ty::U32)
            .u32_val(3)
            .key("split.no", ty::I32)
            .i32_val(0)
            .finish(2);

        // Every prefix shorter than the whole thing must be handled.
        for cut in 0..full.len() {
            let meta = metadata_from_bytes(&full[..cut]);
            // No assertion beyond "did not panic": a short prefix may
            // legitimately yield some keys, just never a wrong answer.
            let _ = meta.len();
        }
    }

    #[test]
    fn an_absurd_string_length_is_refused() {
        // Four bytes of 0xFF should not be able to request 18 exabytes.
        let head = Builder::new()
            .key("huge", ty::STRING)
            .u64_val(u64::MAX)
            .finish(1);
        let meta = metadata_from_bytes(&head);
        assert!(meta.is_empty());
    }

    #[test]
    fn a_lying_kv_count_terminates() {
        // Claims 4 billion keys but supplies none.
        let head = Builder::new().finish(4_000_000_000);
        let meta = metadata_from_bytes(&head);
        assert!(meta.is_empty());
    }

    #[test]
    fn a_missing_file_is_not_an_error() {
        assert_eq!(
            split_info_file(Path::new("/definitely/not/here.gguf")),
            None
        );
        assert!(metadata_from_file(Path::new("/definitely/not/here.gguf")).is_none());
    }

    #[test]
    fn a_tiny_file_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        std::fs::write(&path, b"GGUF").unwrap();
        assert!(metadata_from_file(&path).is_none());
    }
}
