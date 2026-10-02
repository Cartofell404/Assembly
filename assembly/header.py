"""
Binary header format for *.asmb files.

Layout
======
┌──────────────────────────────────────────────┐
│ MAGIC       4 B   b"ASMB"                   │
│ VERSION     4 B   uint32-LE                  │
│ HEADER_LEN  8 B   uint64-LE (JSON payload)   │
│ JSON        variable                          │
│ PADDING     zero bytes → next page boundary   │
├──────────────────────────────────────────────┤
│ GGUF #0     (page-aligned)                   │
│ PADDING     → next page boundary              │
│ GGUF #1     (page-aligned)                   │
│ …                                            │
└──────────────────────────────────────────────┘

The JSON payload stored inside the header carries **offsets** and **sizes**
so that any reader can jump directly to a specific GGUF blob via mmap.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import mmap
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

# ── constants ────────────────────────────────────────────────────────
MAGIC = b"ASMB"
FORMAT_VERSION = 1
PAGE_SIZE = 4096
# MAGIC(4) + VERSION(4) + HEADER_LEN(8) = 16 bytes preamble
PREAMBLE_SIZE = 16
#: Anchor vectors are stored as big-endian float32.
ANCHOR_DTYPE_SIZE = 4
#: How much is read per pass when hashing a blob.
HASH_CHUNK = 4 * 1024 * 1024


# ── data carriers ────────────────────────────────────────────────────
@dataclass(slots=True)
class ModelEntry:
    """Metadata for one GGUF model inside the .asmb archive."""

    name: str
    offset: int = 0
    size: int = 0
    anchors: list[str] = field(default_factory=list)
    #: Anchor vectors, float32, packed big-endian and base64-encoded. JSON
    #: floats were tried first and cost roughly 2x the bytes of the models
    #: they route: a 4096-dim vector is ~40 KB as decimal text and 16 KB here.
    #: Float32 is deliberate -- the vectors come straight out of the embedder
    #: and narrowing them loses routing accuracy for no useful saving.
    anchor_embeddings_b64: str = ""
    #: Width of each stored anchor vector.
    anchor_dim: int = 0
    #: SHA-256 of this model's blob, hex. Empty means "not recorded".
    sha256: str = ""

    def anchor_vectors(self) -> list[list[float]]:
        """Decode :attr:`anchor_embeddings_b64` back into vectors."""
        if not self.anchor_embeddings_b64:
            return []
        return decode_anchors(self.anchor_embeddings_b64, self.anchor_dim)

    @classmethod
    def with_vectors(
        cls,
        name: str,
        vectors: list[list[float]],
        *,
        offset: int = 0,
        size: int = 0,
        anchors: Optional[list[str]] = None,
        sha256: str = "",
    ) -> "ModelEntry":
        """Build an entry from raw vectors, encoding them on the way in.

        Vectors are float32 in storage, so this is where a float64 list gets
        narrowed. Reading them back returns the float32 values, so a test that
        compares against the original list must allow for that.
        """
        blob, dim = encode_anchors_with_dim(vectors)
        return cls(
            name=name,
            offset=offset,
            size=size,
            anchors=list(anchors or []),
            anchor_embeddings_b64=blob,
            anchor_dim=dim,
            sha256=sha256,
        )

    @property
    def anchor_embeddings(self) -> list[list[float]]:
        """The vectors as a list of lists.

        Convenience for callers that only want to read them; construction
        takes raw vectors through :meth:`with_vectors`.
        """
        return self.anchor_vectors()


@dataclass(slots=True)
class EmbeddingEntry:
    """Metadata for the router (embedding) model."""

    name: str
    offset: int = 0
    size: int = 0
    confidence_threshold: float = 0.75
    fallback_model: Optional[str] = None
    sha256: str = ""


@dataclass(slots=True)
class AsmbHeader:
    """Full header that lives inside the .asmb file."""

    assembly_name: str
    version: int = FORMAT_VERSION
    page_size: int = PAGE_SIZE
    embedding: EmbeddingEntry = field(default_factory=lambda: EmbeddingEntry(name=""))
    models: list[ModelEntry] = field(default_factory=list)
    persona: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json_bytes(self) -> bytes:
        """Serialize header to compact JSON (UTF-8)."""
        payload: dict[str, Any] = {
            "assembly_name": self.assembly_name,
            "version": self.version,
            "page_size": self.page_size,
            "persona": self.persona,
            "embedding": {
                "name": self.embedding.name,
                "offset": self.embedding.offset,
                "size": self.embedding.size,
                "confidence_threshold": self.embedding.confidence_threshold,
                "fallback_model": self.embedding.fallback_model,
                "sha256": self.embedding.sha256,
            },
            "models": [
                {
                    "name": m.name,
                    "offset": m.offset,
                    "size": m.size,
                    "anchors": m.anchors,
                    "anchor_embeddings_b64": m.anchor_embeddings_b64,
                    "anchor_dim": m.anchor_dim,
                    "sha256": m.sha256,
                }
                for m in self.models
            ],
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> AsmbHeader:
        """Deserialize header from JSON bytes."""
        raw = json.loads(data.decode("utf-8"))

        emb_raw = raw["embedding"]
        embedding = EmbeddingEntry(
            name=emb_raw["name"],
            offset=emb_raw["offset"],
            size=emb_raw["size"],
            confidence_threshold=emb_raw.get("confidence_threshold", 0.75),
            fallback_model=emb_raw.get("fallback_model"),
            sha256=emb_raw.get("sha256", ""),
        )

        models: list[ModelEntry] = []
        for m in raw["models"]:
            models.append(
                ModelEntry(
                    name=m["name"],
                    offset=m["offset"],
                    size=m["size"],
                    anchors=m.get("anchors", []),
                    anchor_embeddings_b64=m.get(
                        "anchor_embeddings_b64",
                        # Written by the pre-v1 JSON-float layout, which stored
                        # vectors as nested decimal lists. Read it so an
                        # archive built before the binary change still routes,
                        # then it is rewritten in the current form on rebuild.
                        _encode_legacy_anchors(m.get("anchor_embeddings")),
                    ),
                    anchor_dim=m.get("anchor_dim", _legacy_dim(m)),
                    sha256=m.get("sha256", ""),
                )
            )

        return cls(
            assembly_name=raw["assembly_name"],
            version=raw.get("version", FORMAT_VERSION),
            page_size=raw.get("page_size", PAGE_SIZE),
            embedding=embedding,
            models=models,
            persona=raw.get("persona"),
        )


# ── blob checksums ───────────────────────────────────────────────────

def hash_blob(data: bytes | memoryview) -> str:
    """SHA-256 of one blob, hex. Used at build time for the header."""
    return hashlib.sha256(data).hexdigest()


def hash_region(
    mm: memoryview | mmap.mmap,
    offset: int,
    size: int,
) -> str:
    """SHA-256 of ``mm[offset:offset+size]``, read in bounded chunks.

    An archive is gigabytes, so the slice is never materialised: hashing
    ``mm[offset:offset+size]`` directly would copy the whole model into RAM
    and undo the point of memory-mapping it.
    """
    digest = hashlib.sha256()
    remaining = size
    pos = offset
    while remaining > 0:
        chunk = min(HASH_CHUNK, remaining)
        digest.update(mm[pos : pos + chunk])
        pos += chunk
        remaining -= chunk
    return digest.hexdigest()


# ── low-level read / write helpers ───────────────────────────────────

# ── anchor vectors ────────────────────────────────────────────────────

def encode_anchors(vectors: list[list[float]]) -> str:
    """Pack a list of float vectors into a base64 string.

    Stored as big-endian float32, row-major, vectors concatenated. The row
    width is returned alongside the payload rather than guessed from a global
    constant, because the width is a property of whichever embedder produced
    the vectors -- 384 for one model, 1024 for the next.
    """
    return encode_anchors_with_dim(vectors)[0]


def encode_anchors_with_dim(vectors: list[list[float]]) -> tuple[str, int]:
    """Like :func:`encode_anchors`, but also return the row width."""
    if not vectors:
        return "", 0
    dim = len(vectors[0])
    if dim == 0:
        raise ValueError("anchor vectors have zero dimensions")
    flat: list[float] = []
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError(
                f"anchor vectors are ragged: expected {dim} dimensions, got "
                f"{len(vec)}"
            )
        flat.extend(vec)
    packed = struct.pack(f">{len(flat)}f", *flat)
    return base64.b64encode(packed).decode("ascii"), dim


def _legacy_dim(entry: dict[str, Any]) -> int:
    """Row width of a pre-binary anchor list, 0 if there was none."""
    vecs = entry.get("anchor_embeddings") or []
    return len(vecs[0]) if vecs else 0


def _encode_legacy_anchors(vecs: Any) -> str:
    """Convert an old nested-float list into the base64 form."""
    if not vecs:
        return ""
    return encode_anchors([[float(v) for v in vec] for vec in vecs])


def decode_anchors(blob: str, dim: int) -> list[list[float]]:
    """Inverse of :func:`encode_anchors_with_dim`.

    *dim* is the row width recorded at build time.
    """
    if not blob:
        return []
    try:
        packed = base64.b64decode(blob, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidArchiveError(
            f"anchor embeddings are not valid base64: {exc}"
        ) from exc
    if dim <= 0:
        raise InvalidArchiveError(
            f"anchor embeddings declare an embedding dimension of {dim}"
        )
    if len(packed) % ANCHOR_DTYPE_SIZE:
        raise InvalidArchiveError(
            f"anchor embeddings are {len(packed)} bytes, which is not a whole "
            f"number of {ANCHOR_DTYPE_SIZE}-byte float32 values"
        )
    count = len(packed) // ANCHOR_DTYPE_SIZE
    if count % dim:
        raise InvalidArchiveError(
            f"anchor embeddings hold {count} values, which is not divisible by "
            f"the recorded embedding dimension {dim}"
        )
    values = struct.unpack(f">{count}f", packed)
    return [list(values[i : i + dim]) for i in range(0, count, dim)]


def align_to_page(offset: int, page: int = PAGE_SIZE) -> int:
    """Return the next page-aligned offset (>= *offset*)."""
    remainder = offset % page
    return offset if remainder == 0 else offset + (page - remainder)


def write_preamble(buf: bytearray, json_len: int) -> None:
    """Write the 16-byte preamble (magic + version + header_len)."""
    buf.extend(MAGIC)
    buf.extend(struct.pack("<I", FORMAT_VERSION))
    buf.extend(struct.pack("<Q", json_len))


class InvalidArchiveError(ValueError):
    """Raised when a ``.asmb`` blob is truncated, corrupt or unreadable.

    Subclasses :class:`ValueError` so existing callers that catch a bad
    archive keep working, but carries a message meant for a human.
    """


def read_preamble(data: bytes | memoryview) -> tuple[int, int]:
    """Read magic, version and json_len from the first 16 bytes.

    Returns
    -------
    (version, json_length)

    Raises
    ------
    InvalidArchiveError
        If the blob is too short or the magic bytes are wrong.
    """
    if len(data) < PREAMBLE_SIZE:
        raise InvalidArchiveError(
            f"archive is truncated: expected at least {PREAMBLE_SIZE} header "
            f"bytes, found {len(data)}"
        )
    if bytes(data[:4]) != MAGIC:
        raise InvalidArchiveError(
            f"not an .asmb archive: expected magic {MAGIC!r}, found {bytes(data[:4])!r}"
        )
    version = struct.unpack("<I", data[4:8])[0]
    if version != FORMAT_VERSION:
        raise InvalidArchiveError(
            f"unsupported .asmb format version {version}; "
            f"this build understands version {FORMAT_VERSION}"
        )
    json_length = struct.unpack("<Q", data[8:16])[0]
    return version, json_length


def read_header(data: bytes | memoryview) -> AsmbHeader:
    """Read and validate the AsmbHeader at the start of a ``.asmb`` blob.

    Every offset and size is checked against the length of *data*, so a
    corrupt or truncated archive fails here with one readable message instead
    of surfacing later as a struct, JSON or llama.cpp error.
    """
    _version, json_length = read_preamble(data)

    if PREAMBLE_SIZE + json_length > len(data):
        raise InvalidArchiveError(
            f"header is truncated: it declares {json_length} bytes of JSON but "
            f"only {len(data) - PREAMBLE_SIZE} are present"
        )

    json_blob = bytes(data[PREAMBLE_SIZE : PREAMBLE_SIZE + json_length])
    try:
        header = AsmbHeader.from_json_bytes(json_blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidArchiveError(f"header is not valid JSON: {exc}") from exc

    _validate_layout(header, len(data))
    return header


def _validate_layout(header: AsmbHeader, total: int) -> None:
    """Check that every model blob lies inside the archive and is aligned."""
    page = header.page_size or PAGE_SIZE

    for entry in [header.embedding, *header.models]:
        if entry.offset % page:
            raise InvalidArchiveError(
                f"'{entry.name}' is at unaligned offset {entry.offset:#x} "
                f"(expected a multiple of {page})"
            )
        if entry.offset + entry.size > total:
            raise InvalidArchiveError(
                f"'{entry.name}' runs past the end of the archive: needs bytes "
                f"{entry.offset:#x}..{entry.offset + entry.size:#x} but the file "
                f"is only {total:#x} bytes"
            )

    names = [m.name for m in header.models]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise InvalidArchiveError(f"duplicate model names in header: {dupes}")
    if not names:
        raise InvalidArchiveError("archive contains no execution models")
