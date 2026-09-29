"""
Packer — build an .asmb archive from an assembly JSON config.

Usage (programmatic)::

    from assembly.packer import build
    build("config.json", "output.asmb")

Steps
-----
1. Parse & validate the JSON config.
2. Load the embedding model and pre-compute anchor embeddings.
3. Plan the binary layout (offsets for each GGUF blob).
4. Write the preamble + JSON header + padding + GGUF blobs.

The archive is written to a ``.partial`` sibling and renamed into place only
after the size and the embedded header have been verified, so an interrupted
build never leaves a file that looks loadable.
"""

from __future__ import annotations

import mmap
import os
import pathlib
import sys
from typing import Optional

from assembly.config import AssemblyConfig, PreflightError, validate_assembly_preflight
from assembly.header import (
    PREAMBLE_SIZE,
    AsmbHeader,
    EmbeddingEntry,
    ModelEntry,
    align_to_page,
    read_header,
    write_preamble,
)


class BuildError(RuntimeError):
    """Raised when a build fails after validation passed.

    Distinct from :class:`PreflightError`, which means "the config is wrong".
    This one means "the machine let us down" — a source model changed
    mid-copy, the disk filled, or the layout would not converge.
    """


def _file_size(path: pathlib.Path) -> int:
    try:
        return path.stat().st_size
    except OSError as exc:
        raise BuildError(f"cannot stat model file {path}: {exc}") from exc


def _compute_anchor_embeddings(
    embedding_model_path: pathlib.Path,
    models_cfg: list,
) -> dict[str, list[list[float]]]:
    """Load the embedding model and compute anchor vectors.

    Returns a dict mapping model_name → list-of-embedding-vectors.
    """
    # Import here to allow the rest of the package to be imported without
    # llama-cpp-python installed (useful for testing/CI).
    from llama_cpp import Llama

    print(f"  ↳ Loading embedding model: {embedding_model_path}")
    llm = Llama(
        model_path=str(embedding_model_path),
        embedding=True,
        n_ctx=512,
        n_batch=512,
        verbose=False,
    )

    result: dict[str, list[list[float]]] = {}
    for mcfg in models_cfg:
        vecs: list[list[float]] = []
        for anchor_text in mcfg.anchors:
            emb = llm.embed(anchor_text)
            # llm.embed returns list[float] for a single string
            if isinstance(emb[0], list):
                emb = emb[0]
            vecs.append([float(v) for v in emb])
        result[mcfg.name] = vecs
        print(f"    • {mcfg.name}: {len(vecs)} anchors embedded")

    # Explicitly free the model to release memory before packing.
    del llm
    return result


def build(
    config_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    *,
    skip_embeddings: bool = False,
    check_gguf_magic: bool = True,
) -> pathlib.Path:
    """Build an .asmb archive.

    Parameters
    ----------
    config_path:
        Path to the assembly JSON config.
    output_path:
        Destination path for the .asmb file.
    skip_embeddings:
        If *True*, skip pre-computing anchor embeddings (useful for
        tests where no real GGUF models are available).
    check_gguf_magic:
        If *True*, verifies that input files start with the GGUF header magic.

    Returns
    -------
    The resolved *output_path*.
    """
    config_path = pathlib.Path(config_path)
    output_path = pathlib.Path(output_path)

    print(f"[assembly build] config  = {config_path}")
    print(f"[assembly build] output  = {output_path}")

    # ── 0. Pre-flight verification (FAST) ────────────────────────────
    # Runs in <10ms to catch syntax errors, missing models, corrupt files,
    # and insufficient disk space BEFORE loading any heavy models!
    report = validate_assembly_preflight(
        config_path,
        output_path=output_path,
        check_gguf_magic=check_gguf_magic and not skip_embeddings,
    )
    if not report.is_valid:
        raise PreflightError(report)

    for warning in report.warnings:
        print(f"  ⚠ [{warning.field}] {warning.message}", file=sys.stderr)

    # ── 1. Parse config ──────────────────────────────────────────────
    cfg = report.config or AssemblyConfig.from_json(config_path)
    print(f"  Assembly: {cfg.assembly_name}")

    # ── 2. Pre-compute anchor embeddings ─────────────────────────────
    anchor_embs: dict[str, list[list[float]]] = {}
    if not skip_embeddings:
        anchor_embs = _compute_anchor_embeddings(cfg.embedding.path, cfg.models)

    # ── 3. Gather GGUF file sizes ────────────────────────────────────
    emb_size = _file_size(cfg.embedding.path)
    model_sizes = [(m.name, _file_size(m.path)) for m in cfg.models]

    # ── 4. Plan layout ───────────────────────────────────────────────
    # We don't know the exact JSON header size yet because it contains
    # offsets, creating a chicken-and-egg problem.  Solve it iteratively:
    # assume a generous header size, compute offsets, serialize, adjust.

    def _compute_layout(
        estimated_header: int,
    ) -> tuple[AsmbHeader, int]:
        cursor = align_to_page(PREAMBLE_SIZE + estimated_header)

        emb_entry = EmbeddingEntry(
            name=cfg.embedding.name,
            offset=cursor,
            size=emb_size,
            confidence_threshold=cfg.embedding.confidence_threshold,
            fallback_model=cfg.embedding.fallback_model,
        )
        cursor = align_to_page(cursor + emb_size)

        model_entries: list[ModelEntry] = []
        for mcfg, (_, msize) in zip(cfg.models, model_sizes):
            entry = ModelEntry(
                name=mcfg.name,
                offset=cursor,
                size=msize,
                anchors=mcfg.anchors,
                anchor_embeddings=anchor_embs.get(mcfg.name, []),
            )
            model_entries.append(entry)
            cursor = align_to_page(cursor + msize)

        header = AsmbHeader(
            assembly_name=cfg.assembly_name,
            embedding=emb_entry,
            models=model_entries,
            persona=cfg.persona,
        )
        return header, cursor

    def _expected_size(h: AsmbHeader) -> int:
        """Bytes the finished archive must contain.

        This is the end of the last blob, *not* the page-aligned cursor: the
        trailing pad is never written, so it is not part of the file.
        """
        if h.models:
            last = h.models[-1]
            return last.offset + last.size
        return h.embedding.offset + h.embedding.size

    # First pass with generous estimate.
    header, total_size = _compute_layout(estimated_header=8192)
    json_blob = header.to_json_bytes()

    # Re-run with the real header length to fix offsets.
    header, total_size = _compute_layout(estimated_header=len(json_blob) + 256)
    json_blob = header.to_json_bytes()

    # Iterate to a fixed point. The offsets depend on the header length and
    # the header length depends on the offsets (it stores them), so a single
    # "final" pass is only correct if it happens to converge.
    for _ in range(8):
        header, total_size = _compute_layout(estimated_header=len(json_blob))
        blob = header.to_json_bytes()
        if blob == json_blob:
            break
        json_blob = blob
    else:
        raise BuildError(
            "header layout failed to converge; the anchor embeddings are too "
            "large to fit a stable header"
        )

    # The recorded offset must actually clear the JSON we are about to write.
    if PREAMBLE_SIZE + len(json_blob) > header.embedding.offset:
        raise BuildError(
            f"internal layout error: header JSON ({len(json_blob):,} B) "
            f"overlaps the embedding model at {header.embedding.offset:#x}"
        )

    expected_size = _expected_size(header)
    print(f"  Header JSON size : {len(json_blob):,} B")
    print(f"  Total archive    : ~{expected_size / (1024**3):.2f} GiB")

    # ── 5. Write the archive atomically ──────────────────────────────
    # A half-written .asmb still has a valid 16-byte preamble, so an
    # interrupted build would otherwise look loadable and fail much later
    # inside llama.cpp. Write to a sibling temp file and rename on success.
    tmp_path = output_path.with_name(output_path.name + ".partial")
    try:
        with open(tmp_path, "wb") as out:
            # Preamble
            preamble = bytearray()
            write_preamble(preamble, len(json_blob))
            out.write(preamble)

            # JSON header
            out.write(json_blob)

            # Pad to the embedding model's offset
            _pad_to(out, header.embedding.offset)

            # Embedding GGUF
            print(f"  Writing {cfg.embedding.name} @ offset {header.embedding.offset:#x}")
            _copy_file(cfg.embedding.path, out, emb_size, cfg.embedding.name)

            # Execution models
            for mcfg, entry in zip(cfg.models, header.models):
                _pad_to(out, entry.offset)
                print(f"  Writing {entry.name} @ offset {entry.offset:#x}")
                _copy_file(mcfg.path, out, entry.size, entry.name)

            out.flush()
            os.fsync(out.fileno())

        final_size = tmp_path.stat().st_size
        if final_size != expected_size:
            raise BuildError(
                f"archive size mismatch: planned {expected_size:,} bytes, "
                f"wrote {final_size:,}. A source model may have changed "
                f"during the build."
            )

        # Read the header back from the file we just wrote — the only
        # verification that proves the layout on disk is self-consistent.
        # mmap rather than read_bytes: the archive is typically gigabytes.
        with open(tmp_path, "rb") as fh:
            with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                verify = read_header(mm)
        if verify.assembly_name != cfg.assembly_name:
            raise BuildError("verification failed: assembly name mismatch")
        if len(verify.models) != len(cfg.models):
            raise BuildError(
                f"verification failed: header lists {len(verify.models)} models, "
                f"config has {len(cfg.models)}"
            )

        os.replace(tmp_path, output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    final_size = output_path.stat().st_size
    print(f"\nBuilt {output_path}  ({final_size / (1024**3):.2f} GiB)")
    return output_path


# ── helpers ──────────────────────────────────────────────────────────

_COPY_BUF = 8 * 1024 * 1024  # 8 MiB copy buffer


def _pad_to(fh, target_offset: int) -> None:
    """Write zero bytes until the file position equals *target_offset*."""
    current = fh.tell()
    if current > target_offset:
        raise BuildError(
            f"Cannot pad backwards: current={current}, target={target_offset}"
        )
    gap = target_offset - current
    if gap:
        fh.write(b"\x00" * gap)


def _copy_file(src: pathlib.Path, dst, expected: int, name: str) -> None:
    """Stream *src* into *dst*, verifying the byte count matches *expected*.

    A model that is truncated, replaced, or still downloading between the
    ``stat()`` and the copy would otherwise produce a quietly corrupt
    archive.
    """
    written = 0
    with open(src, "rb") as fh:
        while True:
            chunk = fh.read(_COPY_BUF)
            if not chunk:
                break
            dst.write(chunk)
            written += len(chunk)

    if written != expected:
        raise BuildError(
            f"'{name}' changed while building: expected {expected:,} bytes, "
            f"copied {written:,}. Re-run the build."
        )
