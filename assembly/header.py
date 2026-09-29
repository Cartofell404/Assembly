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

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

# ── constants ────────────────────────────────────────────────────────
MAGIC = b"ASMB"
FORMAT_VERSION = 1
PAGE_SIZE = 4096
# MAGIC(4) + VERSION(4) + HEADER_LEN(8) = 16 bytes preamble
PREAMBLE_SIZE = 16


# ── data carriers ────────────────────────────────────────────────────
@dataclass(slots=True)
class ModelEntry:
    """Metadata for one GGUF model inside the .asmb archive."""

    name: str
    offset: int = 0
    size: int = 0
    anchors: list[str] = field(default_factory=list)
    anchor_embeddings: list[list[float]] = field(default_factory=list)


@dataclass(slots=True)
class EmbeddingEntry:
    """Metadata for the router (embedding) model."""

    name: str
    offset: int = 0
    size: int = 0
    confidence_threshold: float = 0.75
    fallback_model: Optional[str] = None


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
            },
            "models": [
                {
                    "name": m.name,
                    "offset": m.offset,
                    "size": m.size,
                    "anchors": m.anchors,
                    "anchor_embeddings": m.anchor_embeddings,
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
        )

        models: list[ModelEntry] = []
        for m in raw["models"]:
            models.append(
                ModelEntry(
                    name=m["name"],
                    offset=m["offset"],
                    size=m["size"],
                    anchors=m.get("anchors", []),
                    anchor_embeddings=m.get("anchor_embeddings", []),
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


# ── low-level read / write helpers ───────────────────────────────────

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
