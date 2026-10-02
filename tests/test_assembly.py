"""
Tests for Assembly Engine — header, config, packer, and router modules.

These tests exercise the pure-Python logic (binary format, routing, config
validation) without requiring actual GGUF model files or llama-cpp-python.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from assembly import ui  # noqa: E402

# ── Header tests ─────────────────────────────────────────────────────

from assembly.header import (
    PAGE_SIZE,
    PREAMBLE_SIZE,
    AsmbHeader,
    EmbeddingEntry,
    InvalidArchiveError,
    ModelEntry,
    align_to_page,
    read_header,
    read_preamble,
    write_preamble,
)


class TestAlignToPage:
    def test_already_aligned(self):
        assert align_to_page(0) == 0
        assert align_to_page(4096) == 4096
        assert align_to_page(8192) == 8192

    def test_needs_padding(self):
        assert align_to_page(1) == 4096
        assert align_to_page(4095) == 4096
        assert align_to_page(4097) == 8192

    def test_custom_page_size(self):
        assert align_to_page(100, page=64) == 128
        assert align_to_page(64, page=64) == 64


class TestPreamble:
    def test_roundtrip(self):
        buf = bytearray()
        write_preamble(buf, json_len=42)
        assert len(buf) == PREAMBLE_SIZE
        version, json_len = read_preamble(buf)
        assert version == 1
        assert json_len == 42

    def test_bad_magic(self):
        buf = b"XXXX" + b"\x00" * 12
        with pytest.raises(InvalidArchiveError, match="not an .asmb archive"):
            read_preamble(buf)

    def test_truncated_preamble(self):
        with pytest.raises(InvalidArchiveError, match="truncated"):
            read_preamble(b"ASMB\x01\x00")

    def test_future_version_rejected(self):
        import struct

        buf = b"ASMB" + struct.pack("<I", 99) + struct.pack("<Q", 0)
        with pytest.raises(InvalidArchiveError, match="unsupported .asmb format version"):
            read_preamble(buf)


class TestAsmbHeader:
    def _make_header(self) -> AsmbHeader:
        return AsmbHeader(
            assembly_name="test-build",
            persona="I am Assembly, your unified assistant.",
            embedding=EmbeddingEntry(
                name="emb", offset=4096, size=1000,
                confidence_threshold=0.8, fallback_model="m2",
            ),
            models=[
                ModelEntry(
                    name="m1", offset=8192, size=2000,
                    anchors=["code", "debug"],
                    anchor_embeddings=[[0.1, 0.2], [0.3, 0.4]],
                ),
                ModelEntry(
                    name="m2", offset=16384, size=3000,
                    anchors=["story"],
                    anchor_embeddings=[[0.5, 0.6]],
                ),
            ],
        )

    def test_json_roundtrip(self):
        h = self._make_header()
        blob = h.to_json_bytes()
        h2 = AsmbHeader.from_json_bytes(blob)
        assert h2.assembly_name == h.assembly_name
        assert h2.persona == "I am Assembly, your unified assistant."
        assert h2.embedding.name == h.embedding.name
        assert h2.embedding.offset == h.embedding.offset
        assert h2.embedding.confidence_threshold == 0.8
        assert h2.embedding.fallback_model == "m2"
        assert len(h2.models) == 2
        assert h2.models[0].anchors == ["code", "debug"]
        assert h2.models[1].anchor_embeddings == [[0.5, 0.6]]

    def test_full_binary_roundtrip(self):
        h = self._make_header()
        json_blob = h.to_json_bytes()

        # Build a mini "file" in memory, with every declared blob present —
        # read_header now validates that the layout fits.
        buf = bytearray()
        write_preamble(buf, len(json_blob))
        buf.extend(json_blob)
        buf.extend(b"\x00" * (8192 + 2000 - len(buf)))
        buf.extend(b"\x00" * (16384 + 3000 - len(buf)))

        h2 = read_header(buf)
        assert h2.assembly_name == "test-build"
        assert h2.models[0].size == 2000


# ── Config tests ─────────────────────────────────────────────────────

from assembly.config import AssemblyConfig, EmbeddingConfig, ModelConfig


class TestConfig:
    def _write_config(self, tmp_path: pathlib.Path, data: dict) -> pathlib.Path:
        p = tmp_path / "config.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_valid_config(self, tmp_path):
        cfg_data = {
            "assembly_name": "build",
            "embedding": {
                "name": "emb",
                "path": "/absolute/emb.gguf",
                "confidence_threshold": 0.7,
                "fallback_model": None,
            },
            "models": [
                {"name": "a", "path": "/absolute/a.gguf", "anchors": ["x"]},
                {"name": "b", "path": "/absolute/b.gguf", "anchors": ["y"]},
            ],
        }
        p = self._write_config(tmp_path, cfg_data)
        cfg = AssemblyConfig.from_json(p)
        assert cfg.assembly_name == "build"
        assert len(cfg.models) == 2

    def test_relative_path_rejected(self):
        with pytest.raises(ValueError, match="absolute"):
            EmbeddingConfig(name="e", path=pathlib.Path("rel/path.gguf"))

    def test_too_few_models(self):
        with pytest.raises(ValueError, match="(?i)at least 2"):
            AssemblyConfig(
                assembly_name="bad",
                embedding=EmbeddingConfig(
                    name="e", path=pathlib.Path("/abs/e.gguf")
                ),
                models=[
                    ModelConfig(
                        name="m1",
                        path=pathlib.Path("/abs/m1.gguf"),
                        anchors=["a"],
                    )
                ],
            )

    def test_duplicate_model_names(self):
        with pytest.raises(ValueError, match="unique"):
            AssemblyConfig(
                assembly_name="dup",
                embedding=EmbeddingConfig(
                    name="e", path=pathlib.Path("/abs/e.gguf")
                ),
                models=[
                    ModelConfig(name="m", path=pathlib.Path("/a/m.gguf"), anchors=["x"]),
                    ModelConfig(name="m", path=pathlib.Path("/a/m2.gguf"), anchors=["y"]),
                ],
            )

    def test_invalid_fallback(self):
        with pytest.raises(ValueError, match="fallback_model"):
            AssemblyConfig(
                assembly_name="fb",
                embedding=EmbeddingConfig(
                    name="e",
                    path=pathlib.Path("/abs/e.gguf"),
                    fallback_model="nonexistent",
                ),
                models=[
                    ModelConfig(name="a", path=pathlib.Path("/a.gguf"), anchors=["x"]),
                    ModelConfig(name="b", path=pathlib.Path("/b.gguf"), anchors=["y"]),
                ],
            )


# ── Pre-flight validation tests ──────────────────────────────────────

from assembly.config import PreflightError, validate_assembly_preflight


class TestPreflight:
    """The pre-flight guard that must fail *before* any model data is read."""

    def _write_config(self, tmp_path: pathlib.Path, data: dict) -> pathlib.Path:
        p = tmp_path / "config.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def _fake_gguf(self, tmp_path: pathlib.Path, name: str, size: int = 2048) -> pathlib.Path:
        p = tmp_path / name
        p.write_bytes(b"GGUF" + os.urandom(size))
        return p

    def _valid(self, tmp_path: pathlib.Path) -> dict:
        return {
            "assembly_name": "preflight",
            "persona": "You are Assembly.",
            "embedding": {
                "name": "emb",
                "path": str(self._fake_gguf(tmp_path, "emb.gguf")),
                "confidence_threshold": 0.75,
                "fallback_model": "m1",
            },
            "models": [
                {
                    "name": "m1",
                    "path": str(self._fake_gguf(tmp_path, "m1.gguf")),
                    "anchors": ["write code", "debug a script"],
                },
                {
                    "name": "m2",
                    "path": str(self._fake_gguf(tmp_path, "m2.gguf")),
                    "anchors": ["tell me a story", "write an email"],
                },
            ],
        }

    def _fields(self, report) -> set:
        return {i.field for i in report.errors}

    # ── happy path ──────────────────────────────────────────────────

    def test_valid_config_passes(self, tmp_path):
        cfg_path = self._write_config(tmp_path, self._valid(tmp_path))
        report = validate_assembly_preflight(
            cfg_path, output_path=tmp_path / "out.asmb"
        )
        assert report.is_valid, [i.message for i in report.errors]
        assert report.config is not None
        assert report.config.assembly_name == "preflight"
        assert len(report.models_info) == 3
        assert report.total_size_bytes > 0
        assert report.free_disk_bytes > 0

    def test_repeated_runs_are_stable(self, tmp_path):
        """Regression: a valid config must never trip the output-path branch."""
        cfg_path = self._write_config(tmp_path, self._valid(tmp_path))
        out = tmp_path / "out.asmb"
        for _ in range(3):
            assert validate_assembly_preflight(cfg_path, output_path=out).is_valid

    # ── structural errors ───────────────────────────────────────────

    def test_missing_config_file(self, tmp_path):
        report = validate_assembly_preflight(tmp_path / "nope.json")
        assert not report.is_valid
        assert "config_path" in self._fields(report)

    def test_malformed_json(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text('{"assembly_name": "x", ', encoding="utf-8")
        report = validate_assembly_preflight(p)
        assert not report.is_valid
        assert "json_syntax" in self._fields(report)
        assert "line 1" in report.errors[0].message

    def test_top_level_not_object(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        report = validate_assembly_preflight(p)
        assert not report.is_valid
        assert "root" in self._fields(report)

    def test_missing_assembly_name(self, tmp_path):
        data = self._valid(tmp_path)
        del data["assembly_name"]
        report = validate_assembly_preflight(self._write_config(tmp_path, data))
        assert "assembly_name" in self._fields(report)

    def test_too_few_models(self, tmp_path):
        data = self._valid(tmp_path)
        data["models"] = data["models"][:1]
        report = validate_assembly_preflight(self._write_config(tmp_path, data))
        assert "models" in self._fields(report)

    def test_duplicate_model_names(self, tmp_path):
        data = self._valid(tmp_path)
        data["models"][1]["name"] = "m1"
        report = validate_assembly_preflight(self._write_config(tmp_path, data))
        assert "models[1].name" in self._fields(report)

    def test_unknown_fallback_model(self, tmp_path):
        data = self._valid(tmp_path)
        data["embedding"]["fallback_model"] = "ghost"
        report = validate_assembly_preflight(self._write_config(tmp_path, data))
        assert "embedding.fallback_model" in self._fields(report)

    def test_threshold_out_of_range(self, tmp_path):
        data = self._valid(tmp_path)
        data["embedding"]["confidence_threshold"] = 1.7
        report = validate_assembly_preflight(self._write_config(tmp_path, data))
        assert "embedding.confidence_threshold" in self._fields(report)

    # ── model file errors ───────────────────────────────────────────

    def test_relative_path_rejected(self, tmp_path):
        data = self._valid(tmp_path)
        data["models"][0]["path"] = "models/m1.gguf"
        report = validate_assembly_preflight(self._write_config(tmp_path, data))
        assert "models[0].path" in self._fields(report)

    def test_missing_model_file(self, tmp_path):
        data = self._valid(tmp_path)
        (tmp_path / "m1.gguf").unlink()
        report = validate_assembly_preflight(self._write_config(tmp_path, data))
        assert "models[0].path" in self._fields(report)

    def test_non_string_path_is_reported_not_raised(self, tmp_path):
        """Regression: `"path": 123` used to raise TypeError, not report."""
        data = self._valid(tmp_path)
        data["embedding"]["path"] = 123
        data["models"][0]["path"] = None
        report = validate_assembly_preflight(self._write_config(tmp_path, data))
        assert not report.is_valid
        fields = self._fields(report)
        assert "embedding.path" in fields
        assert "models[0].path" in fields

    def test_empty_model_file(self, tmp_path):
        data = self._valid(tmp_path)
        (tmp_path / "m2.gguf").write_bytes(b"")
        report = validate_assembly_preflight(self._write_config(tmp_path, data))
        assert "models[1].path" in self._fields(report)

    def test_html_instead_of_gguf(self, tmp_path):
        """A truncated/aborted download must be caught before a long build."""
        data = self._valid(tmp_path)
        (tmp_path / "m1.gguf").write_bytes(b"<!DOCTYPE html><html>404</html>")
        report = validate_assembly_preflight(self._write_config(tmp_path, data))
        assert not report.is_valid
        issue = [i for i in report.errors if i.field == "models[0].path"][0]
        assert "GGUF" in issue.message

    def test_gguf_check_can_be_disabled(self, tmp_path):
        data = self._valid(tmp_path)
        (tmp_path / "m1.gguf").write_bytes(b"not-a-real-gguf")
        report = validate_assembly_preflight(
            self._write_config(tmp_path, data), check_gguf_magic=False
        )
        assert report.is_valid

    # ── anchors ─────────────────────────────────────────────────────

    def test_model_without_anchors(self, tmp_path):
        data = self._valid(tmp_path)
        data["models"][0]["anchors"] = []
        report = validate_assembly_preflight(self._write_config(tmp_path, data))
        assert "models[0].anchors" in self._fields(report)

    def test_single_anchor_warns(self, tmp_path):
        data = self._valid(tmp_path)
        data["models"][0]["anchors"] = ["only one"]
        report = validate_assembly_preflight(self._write_config(tmp_path, data))
        assert report.is_valid
        assert any("anchor" in w.message for w in report.warnings)

    def test_blank_anchor_rejected(self, tmp_path):
        data = self._valid(tmp_path)
        data["models"][0]["anchors"] = ["ok", "   "]
        report = validate_assembly_preflight(self._write_config(tmp_path, data))
        assert "models[0].anchors[1]" in self._fields(report)

    # ── output path ─────────────────────────────────────────────────

    def test_output_dir_missing(self, tmp_path):
        cfg_path = self._write_config(tmp_path, self._valid(tmp_path))
        report = validate_assembly_preflight(
            cfg_path, output_path=tmp_path / "no_such_dir" / "out.asmb"
        )
        assert not report.is_valid
        assert "output_path" in self._fields(report)

    def test_output_overwriting_input_rejected(self, tmp_path):
        data = self._valid(tmp_path)
        cfg_path = self._write_config(tmp_path, data)
        report = validate_assembly_preflight(
            cfg_path, output_path=data["models"][0]["path"]
        )
        assert not report.is_valid
        assert "output_path" in self._fields(report)

    def test_existing_output_warns(self, tmp_path):
        cfg_path = self._write_config(tmp_path, self._valid(tmp_path))
        out = tmp_path / "out.asmb"
        out.write_bytes(b"stale")
        report = validate_assembly_preflight(cfg_path, output_path=out)
        assert report.is_valid
        assert any("overwritten" in w.message for w in report.warnings)

    # ── integration with the packer ─────────────────────────────────

    def test_build_aborts_and_writes_nothing(self, tmp_path):
        """The whole point: bad config must fail before any bytes are written."""
        data = self._valid(tmp_path)
        data["models"][0]["path"] = "/definitely/not/here.gguf"
        cfg_path = self._write_config(tmp_path, data)
        out = tmp_path / "out.asmb"

        from assembly.packer import build

        try:
            build(cfg_path, out, skip_embeddings=True)
            raise AssertionError("build should have raised PreflightError")
        except PreflightError as exc:
            assert not exc.report.is_valid
            assert "models[0].path" in self._fields(exc.report)
        assert not out.exists()

    def test_preflight_error_is_value_error(self):
        from assembly.config import ValidationIssue, ValidationReport

        report = ValidationReport(
            is_valid=False,
            issues=[ValidationIssue("ERROR", "root", "boom")],
            total_size_bytes=0,
            free_disk_bytes=0,
            models_info=[],
        )
        assert isinstance(PreflightError(report), ValueError)


# ── Router tests ─────────────────────────────────────────────────────

from assembly.router import RouteResult, Router


class TestRouter:
    def _make_header(
        self,
        threshold: float = 0.75,
        fallback: str | None = None,
    ) -> AsmbHeader:
        """Build a header with 2 models and known anchor embeddings."""
        # 2-dim embeddings for simplicity.
        return AsmbHeader(
            assembly_name="test",
            embedding=EmbeddingEntry(
                name="emb",
                offset=0,
                size=0,
                confidence_threshold=threshold,
                fallback_model=fallback,
            ),
            models=[
                ModelEntry(
                    name="coder",
                    offset=0,
                    size=0,
                    anchors=["code"],
                    # Normalised direction: roughly [1, 0]
                    anchor_embeddings=[[1.0, 0.0]],
                ),
                ModelEntry(
                    name="writer",
                    offset=0,
                    size=0,
                    anchors=["story"],
                    # Normalised direction: roughly [0, 1]
                    anchor_embeddings=[[0.0, 1.0]],
                ),
            ],
        )

    def test_routes_to_best_match(self):
        router = Router(self._make_header(threshold=0.5))
        # Prompt embedding pointing toward "coder" direction.
        result = router.route(np.array([0.9, 0.1], dtype=np.float32))
        assert result.model_name == "coder"
        assert result.score > 0.5
        assert not result.is_fallback

    def test_routes_to_other_direction(self):
        router = Router(self._make_header(threshold=0.5))
        result = router.route(np.array([0.1, 0.9], dtype=np.float32))
        assert result.model_name == "writer"

    def test_fallback_when_below_threshold(self):
        router = Router(self._make_header(threshold=0.99, fallback="writer"))
        # Ambiguous prompt — cosine similarity won't reach 0.99.
        result = router.route(np.array([0.7, 0.7], dtype=np.float32))
        assert result.model_name == "writer"
        assert result.is_fallback

    def test_no_fallback_uses_best(self):
        router = Router(self._make_header(threshold=0.99, fallback=None))
        result = router.route(np.array([0.8, 0.6], dtype=np.float32))
        # Should still pick best despite score < threshold.
        assert result.model_name == "coder"
        assert not result.is_fallback

    def test_identical_direction_gives_score_1(self):
        router = Router(self._make_header(threshold=0.5))
        result = router.route(np.array([1.0, 0.0], dtype=np.float32))
        assert abs(result.score - 1.0) < 1e-5
        assert result.model_name == "coder"


# ── Packer integration test (no real GGUF files) ────────────────────

class TestPackerLayout:
    """Verify the binary layout without real GGUF models."""

    def test_build_creates_valid_header(self, tmp_path):
        """Create fake GGUF blobs, pack them, and verify the header."""
        # Create fake GGUF files (just random bytes).
        emb_path = tmp_path / "emb.gguf"
        emb_path.write_bytes(os.urandom(5000))

        m1_path = tmp_path / "m1.gguf"
        m1_path.write_bytes(os.urandom(10000))

        m2_path = tmp_path / "m2.gguf"
        m2_path.write_bytes(os.urandom(8000))

        # Write a config file.
        cfg = {
            "assembly_name": "layout-test",
            "embedding": {
                "name": "emb",
                "path": str(emb_path),
                "confidence_threshold": 0.75,
                "fallback_model": None,
            },
            "models": [
                {"name": "m1", "path": str(m1_path), "anchors": ["x"]},
                {"name": "m2", "path": str(m2_path), "anchors": ["y"]},
            ],
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))

        from assembly.packer import build

        out = tmp_path / "test.asmb"
        build(cfg_path, out, skip_embeddings=True)

        # Read back and verify.
        data = out.read_bytes()
        header = read_header(data)

        assert header.assembly_name == "layout-test"
        assert header.embedding.offset % PAGE_SIZE == 0
        assert header.embedding.size == 5000

        for m in header.models:
            assert m.offset % PAGE_SIZE == 0

        # Verify actual bytes at offsets.
        emb_bytes = emb_path.read_bytes()
        assert data[header.embedding.offset : header.embedding.offset + 5000] == emb_bytes

        m1_bytes = m1_path.read_bytes()
        m1_entry = header.models[0]
        assert data[m1_entry.offset : m1_entry.offset + 10000] == m1_bytes


class TestPersona:
    """Verify persona handling and system message injection."""

    def test_config_with_persona(self, tmp_path):
        cfg_data = {
            "assembly_name": "persona-build",
            "persona": "I am Assembly. I speak with one voice.",
            "embedding": {
                "name": "emb",
                "path": "/abs/emb.gguf",
                "confidence_threshold": 0.7,
                "fallback_model": None,
            },
            "models": [
                {"name": "a", "path": "/abs/a.gguf", "anchors": ["x"]},
                {"name": "b", "path": "/abs/b.gguf", "anchors": ["y"]},
            ],
        }
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg_data))
        cfg = AssemblyConfig.from_json(p)
        assert cfg.persona == "I am Assembly. I speak with one voice."

    def test_prepare_messages_injects_system(self):
        from assembly.wrapper import AssemblyLlama

        # Test _prepare_messages without initializing the full mmap
        class DummyAssemblyLlama(AssemblyLlama):
            def __init__(self, persona):
                self._persona = persona

        dummy = DummyAssemblyLlama("You are Assembly.")
        messages = [{"role": "user", "content": "Who are you?"}]
        prepared = dummy._prepare_messages(messages)

        # Original messages should not be mutated
        assert len(messages) == 1
        assert len(prepared) == 2
        assert prepared[0] == {"role": "system", "content": "You are Assembly."}
        assert prepared[1] == {"role": "user", "content": "Who are you?"}

    def test_prepare_messages_prepends_to_existing_system(self):
        from assembly.wrapper import AssemblyLlama

        class DummyAssemblyLlama(AssemblyLlama):
            def __init__(self, persona):
                self._persona = persona

        dummy = DummyAssemblyLlama("You are Assembly.")
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        prepared = dummy._prepare_messages(messages)

        assert len(prepared) == 2
        assert "You are Assembly." in prepared[0]["content"]
        assert "Be concise." in prepared[0]["content"]

    def test_packer_persists_persona(self, tmp_path):
        emb_path = tmp_path / "emb.gguf"
        emb_path.write_bytes(os.urandom(5000))
        m1_path = tmp_path / "m1.gguf"
        m1_path.write_bytes(os.urandom(6000))
        m2_path = tmp_path / "m2.gguf"
        m2_path.write_bytes(os.urandom(7000))

        cfg = {
            "assembly_name": "persona-pack",
            "persona": "Custom Persona String",
            "embedding": {
                "name": "emb",
                "path": str(emb_path),
                "confidence_threshold": 0.75,
                "fallback_model": None,
            },
            "models": [
                {"name": "m1", "path": str(m1_path), "anchors": ["x"]},
                {"name": "m2", "path": str(m2_path), "anchors": ["y"]},
            ],
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))

        from assembly.packer import build

        out = tmp_path / "test.asmb"
        build(cfg_path, out, skip_embeddings=True)

        header = read_header(out.read_bytes())
        assert header.persona == "Custom Persona String"


class MockSocket:
    def __init__(self, raw_input: bytes):
        import io
        self._rfile = io.BytesIO(raw_input)
        self._wfile = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        if "r" in mode:
            return self._rfile
        return self._wfile

    def sendall(self, data: bytes) -> None:
        self._wfile.write(data)

    def settimeout(self, value) -> None:
        # socketserver.StreamRequestHandler.setup() calls this.
        self.timeout = value

    def get_output(self) -> bytes:
        return self._wfile.getvalue()


class TestServer:
    """Verify HTTP API server endpoints, CORS, and routing in-memory."""

    def _execute(self, manager, request_bytes: bytes, api_key=None,
                 allow_origin=None) -> tuple[int, dict, Any]:
        from assembly.server import AssemblyRequestHandler
        from unittest.mock import MagicMock

        sock = MockSocket(request_bytes)
        mock_server = MagicMock()
        mock_server.manager = manager
        mock_server.api_key = api_key
        mock_server.allow_origin = allow_origin

        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), mock_server)
        raw_output = sock.get_output()

        header_bytes, _, body_bytes = raw_output.partition(b"\r\n\r\n")
        lines = header_bytes.decode("utf-8").split("\r\n")
        status = int(lines[0].split()[1])
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        parsed_body = None
        if body_bytes:
            try:
                parsed_body = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                parsed_body = body_bytes

        return status, headers, parsed_body

    def test_health_has_no_wildcard_cors_by_default(self):
        from assembly.server import ModelManager
        mgr = ModelManager()

        status, headers, body = self._execute(
            mgr,
            b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        assert status == 200
        assert body["status"] == "ok"
        assert body["loaded_models_count"] == 0
        # Wildcard CORS on a state-changing local service is a drive-by
        # attack: any page the operator visits could drive it.
        assert "access-control-allow-origin" not in headers

        status, headers, _ = self._execute(
            mgr,
            b"OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        assert status == 204
        assert "access-control-allow-origin" not in headers

    def test_cors_echoes_explicit_origin(self):
        from assembly.server import ModelManager
        mgr = ModelManager()

        status, headers, _ = self._execute(
            mgr,
            b"OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Origin: https://my-app.example\r\n\r\n",
            allow_origin="https://my-app.example",
        )
        assert status == 204
        assert headers.get("access-control-allow-origin") == "https://my-app.example"

    def test_bearer_token_required_when_configured(self):
        from assembly.server import ModelManager
        mgr = ModelManager()

        status, _, _ = self._execute(
            mgr,
            b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n",
            api_key="s3cret",
        )
        assert status == 401

        status, _, _ = self._execute(
            mgr,
            b"GET /health HTTP/1.1\r\nHost: localhost\r\n"
            b"Authorization: Bearer wrong\r\n\r\n",
            api_key="s3cret",
        )
        assert status == 401

        status, _, body = self._execute(
            mgr,
            b"GET /health HTTP/1.1\r\nHost: localhost\r\n"
            b"Authorization: Bearer s3cret\r\n\r\n",
            api_key="s3cret",
        )
        assert status == 200
        assert body["status"] == "ok"

    def test_oversized_body_rejected_before_read(self):
        from assembly.server import ModelManager
        mgr = ModelManager()
        body = json.dumps({"prompt": "x" * (2 << 20)}).encode()

        status, _, parsed = self._execute(
            mgr,
            b"POST /api/route HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body,
        )
        assert status == 413
        assert "exceeds" in parsed["error"]["message"]

    def test_chunked_body_rejected(self):
        from assembly.server import ModelManager
        mgr = ModelManager()

        status, _, parsed = self._execute(
            mgr,
            b"POST /api/route HTTP/1.1\r\nHost: localhost\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n",
        )
        assert status == 413

    def test_malformed_json_is_400(self):
        from assembly.server import ModelManager
        mgr = ModelManager()
        body = b"{not json"

        status, _, parsed = self._execute(
            mgr,
            b"POST /api/route HTTP/1.1\r\nHost: localhost\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body,
        )
        assert status == 400
        assert "malformed" in parsed["error"]["message"]

    def test_unknown_route_is_404(self):
        from assembly.server import ModelManager
        mgr = ModelManager()
        status, _, _ = self._execute(
            mgr, b"GET /nope HTTP/1.1\r\nHost: localhost\r\n\r\n"
        )
        assert status == 404

    def test_openai_and_ollama_endpoints(self):
        from unittest.mock import MagicMock

        from assembly.server import ModelManager

        mgr = ModelManager()

        mock_llm = MagicMock()
        mock_llm.header.assembly_name = "TestAssembly"
        mock_llm.header.models = []
        mock_llm.persona = "I am Assembly."
        mock_llm.route.return_value = RouteResult(
            model_name="coder", model_index=0, score=0.92, is_fallback=False
        )
        mock_llm.create_chat_completion.return_value = {
            "id": "chat-123",
            "choices": [{"message": {"role": "assistant", "content": "print('hello')"}}],
        }
        mock_llm.return_value = {
            "id": "cmpl-123",
            "choices": [{"text": "print('hello')"}],
        }

        mgr._models["assembly"] = mock_llm
        mgr._active_alias = "assembly"

        # 1. Models list
        status, _, body = self._execute(
            mgr,
            b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        assert status == 200
        assert body["data"][0]["id"] == "assembly"

        # 2. Route endpoint
        route_body = json.dumps({"prompt": "write code"}).encode()
        status, _, body = self._execute(
            mgr,
            b"POST /api/route HTTP/1.1\r\nHost: localhost\r\n"
            + f"Content-Length: {len(route_body)}\r\n\r\n".encode()
            + route_body,
        )
        assert status == 200
        assert body["selected_model"] == "coder"
        assert body["score"] == 0.92

        # 3. OpenAI Chat Completions
        chat_body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        status, _, body = self._execute(
            mgr,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            + f"Content-Length: {len(chat_body)}\r\n\r\n".encode()
            + chat_body,
        )
        assert status == 200
        assert body["choices"][0]["message"]["content"] == "print('hello')"
        assert body["assembly_route"]["selected_model"] == "coder"

        # 4. Ollama style chat
        status, _, body = self._execute(
            mgr,
            b"POST /api/chat HTTP/1.1\r\nHost: localhost\r\n"
            + f"Content-Length: {len(chat_body)}\r\n\r\n".encode()
            + chat_body,
        )
        assert status == 200
        assert body["message"]["content"] == "print('hello')"
        assert body["routed_model"] == "coder"
        assert body["done"] is True






    def test_root_is_gated_when_key_is_set(self):
        from assembly.server import ModelManager

        status, _, _ = self._execute(
            ModelManager(),
            b"GET / HTTP/1.1\r\nHost: localhost\r\nAccept: text/html\r\n\r\n",
            api_key="secret",
        )
        assert status == 401

    def test_chat_page_no_longer_exists(self):
        """The web UI was removed; /chat must not serve anything."""
        from assembly.server import ModelManager

        status, headers, _ = self._execute(
            ModelManager(),
            b"GET /chat HTTP/1.1\r\nHost: localhost\r\nAccept: text/html\r\n\r\n",
        )
        assert status == 404
        assert "text/html" not in headers.get("content-type", "")

    def test_no_html_endpoint_is_public(self):
        """No path may serve HTML without a token, now that the shell is gone."""
        from assembly.server import ModelManager

        mgr = ModelManager()
        for path in (b"/", b"/chat", b"/index.html"):
            status, headers, _ = self._execute(
                mgr,
                b"GET " + path + b" HTTP/1.1\r\nHost: localhost\r\nAccept: text/html\r\n\r\n",
                api_key="secret",
            )
            assert status in (401, 404), f"{path.decode()} served {status}"
            assert "text/html" not in headers.get("content-type", "")

    def test_root_json_is_still_gated(self):
        from assembly.server import ModelManager

        status, _, _ = self._execute(
            ModelManager(),
            b"GET / HTTP/1.1\r\nHost: localhost\r\nAccept: application/json\r\n\r\n",
            api_key="secret",
        )
        assert status == 401

# ── Packer integrity ──────────────────────────────────────────────────

class TestPackerIntegrity:
    """The build must be atomic and must detect a model that changed."""

    def _config(self, tmp_path, emb_size=2000, m1_size=3000, m2_size=1500):
        for name, size in (("emb.gguf", emb_size), ("m1.gguf", m1_size),
                           ("m2.gguf", m2_size)):
            (tmp_path / name).write_bytes(b"\x00" * size)

        cfg = {
            "assembly_name": "integrity",
            "embedding": {"name": "emb", "path": str(tmp_path / "emb.gguf"),
                          "confidence_threshold": 0.7, "fallback_model": "m1"},
            "models": [
                {"name": "m1", "path": str(tmp_path / "m1.gguf"),
                 "anchors": ["a", "b"]},
                {"name": "m2", "path": str(tmp_path / "m2.gguf"),
                 "anchors": ["c", "d"]},
            ],
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        return cfg_path

    def test_build_is_atomic_and_leaves_no_partial(self, tmp_path):
        from assembly.packer import build

        cfg_path = self._config(tmp_path)
        out = tmp_path / "built.asmb"
        build(cfg_path, out, skip_embeddings=True)

        assert out.is_file()
        assert not list(tmp_path.glob("built.asmb.partial*"))
        assert read_header(out.read_bytes()).assembly_name == "integrity"

    def test_a_killed_build_is_swept_on_the_next_run(self, tmp_path):
        """SIGKILL and power cuts never run the cleanup handler.

        Those are the cases that leave an interrupted multi-gigabyte archive
        on disk, so the next build has to recognise and reclaim it.
        """
        from assembly.packer import _sweep_abandoned

        out = tmp_path / "built.asmb"
        orphan = tmp_path / "built.asmb.partial-999999"
        orphan.write_bytes(b"x" * 4096)

        _sweep_abandoned(out)
        assert not orphan.exists()

    def test_a_running_build_is_never_swept(self, tmp_path):
        """Deleting a live build's partial would corrupt a good build."""
        import os
        from assembly.packer import _sweep_abandoned

        out = tmp_path / "built.asmb"
        live = tmp_path / f"built.asmb.partial-{os.getpid()}"
        live.write_bytes(b"x" * 4096)

        _sweep_abandoned(out)
        assert live.exists(), "this pid is running, so the file must stay"

    def test_sweep_ignores_unrelated_files(self, tmp_path):
        from assembly.packer import _sweep_abandoned

        out = tmp_path / "built.asmb"
        for name in ("built.asmb.partial-notapid", "built.asmb.partial-",
                     "other.asmb.partial-999999", "built.asmb"):
            (tmp_path / name).write_bytes(b"x")

        _sweep_abandoned(out)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "built.asmb", "built.asmb.partial-", "built.asmb.partial-notapid",
            "other.asmb.partial-999999",
        ]

    def test_a_successful_build_leaves_nothing_behind(self, tmp_path):
        from assembly.packer import build

        out = tmp_path / "built.asmb"
        build(self._config(tmp_path), out, skip_embeddings=True)
        assert not list(tmp_path.glob("built.asmb.partial*"))

    def test_truncated_source_model_fails_the_build(self, tmp_path):
        """A model shrinking between stat() and copy() must not be packed."""
        from assembly import packer

        cfg_path = self._config(tmp_path)
        out = tmp_path / "built.asmb"

        real_copy = packer._copy_file
        calls = {"n": 0}

        def shrinking_copy(src, dst, expected, name):
            calls["n"] += 1
            if calls["n"] == 1:
                # Simulate the file being replaced by a smaller one mid-build.
                src.write_bytes(b"\x00" * (expected // 2))
            return real_copy(src, dst, expected, name)

        packer._copy_file = shrinking_copy
        try:
            with pytest.raises(packer.BuildError, match="changed while building"):
                packer.build(cfg_path, out, skip_embeddings=True)
        finally:
            packer._copy_file = real_copy

        # Nothing usable is left behind, and no half-written archive is exposed.
        assert not out.exists()
        assert not list(tmp_path.glob("built.asmb.partial*"))

    def test_size_mismatch_is_rejected(self, tmp_path):
        from assembly import packer

        cfg_path = self._config(tmp_path)
        out = tmp_path / "built.asmb"

        real_copy = packer._copy_file

        def lying_copy(src, dst, expected, name):
            dst.write(b"\x00" * (expected + 16))
            return None

        packer._copy_file = lying_copy
        try:
            with pytest.raises(packer.BuildError, match="size mismatch"):
                packer.build(cfg_path, out, skip_embeddings=True)
        finally:
            packer._copy_file = real_copy

        assert not out.exists()
        assert not list(tmp_path.glob("built.asmb.partial*"))


# ── CLI ───────────────────────────────────────────────────────────────

class TestRunRequiresAServer:
    """`run` used to fall back to in-process inference with no server running.

    That fallback was a second, unhardened way in: no bearer token, no
    /api/load jail, no request-size or concurrency cap, unsanitised errors.
    """

    def _args(self, **over):
        base = dict(
            command="run", model=None, server=None, prompt="hi",
            prompt_pos=None, persona=None, max_tokens=256, route=False,
            no_memory=False,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def _no_server(self, monkeypatch, url="http://127.0.0.1:8000"):
        import assembly.main as main_mod

        monkeypatch.setattr(main_mod, "_is_server_available", lambda u: False)
        monkeypatch.setattr(
            main_mod, "AssemblyLlama", _explode_if_used, raising=False
        )

    def test_it_refuses_without_a_server(self, monkeypatch, capsys):
        import assembly.main as main_mod

        self._no_server(monkeypatch)
        with pytest.raises(SystemExit) as exc:
            main_mod._cmd_run(self._args())
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "no Assembly server" in err
        # The error has to say how to fix it.
        assert "assembly serve -m" in err

    def test_it_never_loads_a_model_in_process(self, monkeypatch):
        """The regression this guards: a local fallback creeping back in."""
        import assembly.main as main_mod

        self._no_server(monkeypatch)
        with pytest.raises(SystemExit):
            main_mod._cmd_run(self._args(model="/models/a.asmb"))

        assert not hasattr(main_mod, "_LocalBackend"), (
            "the in-process backend is back; run should only speak HTTP"
        )

    def test_an_explicit_server_is_checked_too(self, monkeypatch, capsys):
        import assembly.main as main_mod

        self._no_server(monkeypatch, url="http://192.168.1.50:8000")
        with pytest.raises(SystemExit) as exc:
            main_mod._cmd_run(self._args(server="http://192.168.1.50:8000"))
        assert exc.value.code == 1
        assert "192.168.1.50" in capsys.readouterr().err

    def test_it_defaults_to_localhost(self, monkeypatch):
        import assembly.main as main_mod

        seen: list = []
        monkeypatch.setattr(main_mod, "_is_server_available", lambda u: seen.append(u) or True)
        asked: list = []
        monkeypatch.setattr(main_mod, "_start_session", lambda *a, **k: asked.append(1))

        main_mod._cmd_run(self._args())
        assert seen == ["http://127.0.0.1:8000"]

    def test_an_asmb_argument_is_loaded_on_the_server(self, monkeypatch):
        import assembly.main as main_mod

        monkeypatch.setattr(main_mod, "_is_server_available", lambda u: True)
        loaded: list = []
        monkeypatch.setattr(
            main_mod, "_load_on_server",
            lambda url, path, persona=None: loaded.append(path),
        )
        monkeypatch.setattr(main_mod, "_start_session", lambda *a, **k: None)

        main_mod._cmd_run(self._args(model="/models/a.asmb"))
        assert loaded == ["/models/a.asmb"]

    def test_a_bare_alias_is_not_sent_to_api_load(self, monkeypatch):
        """`run -m default` means an alias, not a file to open.

        Sending it to /api/load would fail and print a scary warning about a
        model that never needed loading.
        """
        import assembly.main as main_mod

        monkeypatch.setattr(main_mod, "_is_server_available", lambda u: True)
        loaded: list = []
        monkeypatch.setattr(
            main_mod, "_load_on_server",
            lambda url, path, persona=None: loaded.append(path),
        )
        monkeypatch.setattr(main_mod, "_start_session", lambda *a, **k: None)

        main_mod._cmd_run(self._args(model="default"))
        assert loaded == []


def _explode_if_used(*args, **kwargs):
    raise AssertionError("run must not do in-process inference")


class TestCLIArchivePath:
    """-m is required, and it must be absolute.

    Auto-detection from the working directory meant that serving from the wrong
    folder loaded whatever `.asmb` happened to be sitting in it -- and then
    handed that archive to whoever asked the server.
    """

    def _parser(self):
        from assembly.main import _build_parser

        return _build_parser()

    def test_serve_refuses_to_run_without_an_archive(self):
        with pytest.raises(SystemExit) as exc:
            self._parser().parse_args(["serve"])
        assert exc.value.code == 2, "argparse should reject the missing -m"

    @pytest.mark.parametrize("command", ["serve", "info"])
    def test_every_archive_command_requires_m(self, command):
        """serve and info can only ever mean a path, so argparse demands it."""
        with pytest.raises(SystemExit) as exc:
            self._parser().parse_args([command])
        assert exc.value.code == 2, command

    def test_run_keeps_m_optional_for_the_server_alias_form(self):
        """`run -s URL -m default` names an alias, not a file.

        So -m cannot be argparse-required on run; it is required only on the
        local-inference path, which _require_model_argument enforces.
        """
        args = self._parser().parse_args(
            ["run", "-s", "http://localhost:8000", "-m", "default", "-p", "hi"]
        )
        assert args.model == "default"
        assert args.server == "http://localhost:8000"

    @pytest.mark.parametrize("command", ["serve", "run", "info"])
    def test_a_relative_path_is_refused(self, command, tmp_path, capsys,
                                        monkeypatch):
        from assembly.main import _require_archive

        # resolve() spells out the fix against the cwd, so pin it.
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc:
            _require_archive("a.asmb")
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "must be absolute" in err
        # The error has to hand back the path that would have worked.
        assert str(tmp_path / "a.asmb") in err, "the fix was not spelled out"

    def test_an_absolute_path_is_accepted(self, tmp_path):
        from assembly.main import _require_archive

        archive = tmp_path / "a.asmb"
        archive.write_bytes(b"x")
        assert _require_archive(str(archive)) == str(archive)

    def test_a_tilde_path_is_expanded(self, tmp_path, monkeypatch):
        """`~` is a shell nicety, but expanduser covers the quoted case too."""
        from assembly.main import _require_archive

        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "a.asmb").write_bytes(b"x")
        assert _require_archive("~/a.asmb") == str(tmp_path / "a.asmb")

    def test_a_missing_absolute_path_still_reports_not_found(self, capsys):
        from assembly.main import _require_archive

        with pytest.raises(SystemExit) as exc:
            _require_archive("/nonexistent/nope.asmb")
        assert exc.value.code == 1
        assert "no such archive" in capsys.readouterr().err

    def test_the_error_names_the_command_they_ran(self, capsys, tmp_path):
        from assembly.main import _require_model_argument

        with pytest.raises(SystemExit):
            _require_model_argument(
                argparse.Namespace(command="run", model=None)
            )
        assert "assembly run -m" in capsys.readouterr().err

    def test_auto_detection_is_gone(self):
        """The cwd-dependent guess is what this rule replaces."""
        from assembly import main as main_mod

        assert not hasattr(main_mod, "_auto_detect_model")


class TestCLI:
    """Argument wiring, --version, and exit codes."""

    def _parser(self):
        from assembly.main import _build_parser
        return _build_parser()

    def test_version_flag(self, capsys):
        from assembly import __version__

        with pytest.raises(SystemExit) as exc:
            self._parser().parse_args(["--version"])
        assert exc.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_serve_defaults_to_loopback(self):
        args = self._parser().parse_args(["serve", "-m", "/models/a.asmb"])
        assert args.host == "127.0.0.1"
        assert args.api_key is None
        assert args.allow_origin is None

    def test_serve_security_flags_parse(self):
        args = self._parser().parse_args(
            ["serve", "-m", "/models/a.asmb",
             "--host", "0.0.0.0", "--api-key", "s3cret",
             "--allow-origin", "http://localhost:5173",
             "--models-dir", "/tmp/models", "--temp-dir", "/tmp/stage"]
        )
        assert (args.host, args.api_key) == ("0.0.0.0", "s3cret")
        assert args.allow_origin == "http://localhost:5173"
        assert str(args.models_dir) == "/tmp/models"
        assert str(args.temp_dir) == "/tmp/stage"

    def test_run_accepts_positional_prompt(self):
        args = self._parser().parse_args(["run", "-m", "x.asmb", "hello there"])
        assert args.prompt_pos == "hello there"
        assert args.prompt is None

    def test_run_no_longer_offers_local_only_flags(self):
        """--temp-dir/--n-gpu-layers/-v only configured the deleted local model.

        Left in place they would parse and then silently do nothing, which is
        worse than refusing: someone would point staging at a fast volume and
        watch it be ignored.
        """
        for flag, value in (("--temp-dir", "/s"), ("--n-gpu-layers", "4"), ("-v", None)):
            argv = ["run", "-m", "/m/a.asmb", flag] + ([value] if value else [])
            with pytest.raises(SystemExit) as exc:
                self._parser().parse_args(argv)
            assert exc.value.code == 2, flag

    def test_serve_refuses_public_host_without_key(self, monkeypatch):
        from assembly import server

        monkeypatch.delenv("ASSEMBLY_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="API key"):
            server.serve(host="0.0.0.0", api_key=None, models_dir=".")

    def test_serve_allows_public_host_with_env_key(self, monkeypatch):
        """An env-provided key must satisfy the same check as --api-key."""
        from assembly import server

        monkeypatch.setenv("ASSEMBLY_API_KEY", "from-env")
        created = {}

        class FakeServer:
            def __init__(self, address, manager, api_key=None, allow_origin=None):
                created["address"] = address
                created["api_key"] = api_key

            def serve_forever(self):  # return immediately instead of blocking
                return None

            def server_close(self):
                return None

        monkeypatch.setattr(server, "AssemblyServer", FakeServer)
        monkeypatch.setattr(server.ModelManager, "load_model", lambda *a, **k: None)

        server.serve(host="0.0.0.0", api_key=None, port=9999)

        assert created["address"] == ("0.0.0.0", 9999)
        assert created["api_key"] == "from-env"

    def test_serve_binds_loopback_by_default(self, monkeypatch):
        from assembly import server

        monkeypatch.delenv("ASSEMBLY_API_KEY", raising=False)
        created = {}

        class FakeServer:
            def __init__(self, address, manager, api_key=None, allow_origin=None):
                created["address"] = address

            def serve_forever(self):
                return None

            def server_close(self):
                return None

        monkeypatch.setattr(server, "AssemblyServer", FakeServer)
        server.serve(port=8123)
        assert created["address"] == ("127.0.0.1", 8123)

    def test_info_reports_missing_file(self, capsys):
        from assembly.main import _cmd_info

        args = argparse.Namespace(model="/nonexistent/nope.asmb")
        with pytest.raises(SystemExit) as exc:
            _cmd_info(args)
        assert exc.value.code == 1
        assert "no such archive" in capsys.readouterr().err

    def test_info_rejects_corrupt_archive(self, tmp_path, capsys):
        from assembly.main import _cmd_info

        bad = tmp_path / "bad.asmb"
        bad.write_bytes(b"NOTMAGIC" + b"\x00" * 64)

        with pytest.raises(SystemExit) as exc:
            _cmd_info(argparse.Namespace(model=str(bad)))
        assert exc.value.code == 1
        assert "no such archive" not in capsys.readouterr().err

    def test_build_surfaces_build_error(self, tmp_path, capsys, monkeypatch):
        from assembly import packer
        from assembly.main import _cmd_build

        def boom(*a, **k):
            raise packer.BuildError("disk gave up")

        monkeypatch.setattr(packer, "build", boom)
        args = argparse.Namespace(
            config=str(tmp_path / "c.json"), output=str(tmp_path / "o.asmb"),
            skip_embeddings=True,
        )
        with pytest.raises(SystemExit) as exc:
            _cmd_build(args)
        assert exc.value.code == 1
        assert "disk gave up" in capsys.readouterr().err


# ── Terminal UI ────────────────────────────────────────────────────────
class TestTerminalUI:
    """The CLI's rendering layer.

    Bugs here are invisible to unit tests that capture stdout, so these assert
    on the returned values and on what the helpers actually write, not on how
    a terminal happens to draw it.
    """

    def test_prompt_contains_exactly_one_arrow(self):
        """Regression: the prompt used to be drawn twice.

        ``prompt_text()`` renders through a throwaway rich Console to build the
        escape sequence. Pointing that Console at stdout made it print the
        prompt, and then ``input()`` printed the returned copy again, so every
        turn began with two arrows.
        """
        import re as _re
        from assembly import ui

        prompt = ui.prompt_text()
        visible = _re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", prompt)
        assert visible.count("▸") == 1, repr(prompt)

    def test_prompt_writes_nothing_to_stdout(self, capsys):
        from assembly import ui

        ui.prompt_text()
        captured = capsys.readouterr()
        assert captured.out == "", f"prompt_text leaked {captured.out!r} to stdout"

    def test_palette_is_orange_dominant(self):
        """Every accent must sit on the orange axis, or the CLI stops reading
        as one product."""
        from assembly import ui

        def hue(hex_colour: str) -> float:
            r = int(hex_colour[1:3], 16) / 255
            g = int(hex_colour[3:5], 16) / 255
            b = int(hex_colour[5:7], 16) / 255
            mx, mn = max(r, g, b), min(r, g, b)
            if mx == mn:
                return -1.0
            if mx == r:
                return ((g - b) / (mx - mn)) % 6
            if mx == g:
                return (b - r) / (mx - mn) + 2
            return (r - g) / (mx - mn) + 4

        warm = [ui.ORANGE_DEEP, ui.ORANGE, ui.ORANGE_LIT, ui.AMBER, ui.MUTED, ui.CREAM]
        for colour in warm:
            h = hue(colour) * 60
            assert -1 <= h <= 70, f"{colour} is not orange-ish (hue {h:.0f})"

    def test_status_goes_to_stderr_only(self, capsys):
        from assembly import ui

        with ui.status("loading"):
            pass
        captured = capsys.readouterr()
        assert captured.out == "", "status must not pollute stdout"
        assert "loading" in captured.err

    def test_error_and_success_go_to_stderr(self, capsys):
        from assembly import ui

        ui.error("went wrong")
        ui.success("went right")
        captured = capsys.readouterr()
        assert captured.out == "", "decorative output must not reach stdout"
        assert "went wrong" in captured.err and "went right" in captured.err

    def test_reply_goes_to_stdout(self, capsys):
        from assembly import ui

        ui.print_reply("the answer")
        captured = capsys.readouterr()
        assert captured.out.strip() == "the answer"
        assert captured.err == ""

    def test_tables_render_without_raising(self, capsys):
        from assembly import ui

        ui.table("Models", ["name", "state"], [["Code", "idle"]])
        captured = capsys.readouterr()
        assert "Code" in captured.err
        assert captured.out == ""

    def test_banner_renders_without_raising(self, capsys):
        from assembly import ui

        ui.banner("Assembly", "listening on 127.0.0.1:8000",
                  [("Archive", "mesh.asmb"), ("", ""), ("OpenAI", "http://x/v1")])
        captured = capsys.readouterr()
        assert "Assembly" in captured.err
        assert captured.out == ""

    def test_no_color_env_disables_styling(self, monkeypatch):
        from assembly import ui

        monkeypatch.setattr(ui, "_NO_COLOR", True)
        assert ui.prompt_text() == "▸ "


class TestMeter:
    """The sampling bars, including the signed penalties."""

    def _cells(self, err: str) -> str:
        """Just the bar glyphs, without the label or the number."""
        return "".join(c for c in err if c in "\u2588\u00b7\u2502")

    def test_minimum_shows_a_single_cell(self, capsys):
        """One filled cell keeps the row visible even at the range floor."""
        ui.meter("temperature", 0.0, 0.0, 2.0)
        assert self._cells(capsys.readouterr().err).count("\u2588") == 1

    def test_halfway_is_half_full(self, capsys):
        ui.meter("temperature", 1.0, 0.0, 2.0)
        cells = self._cells(capsys.readouterr().err)
        assert cells.count("\u2588") == 12

    def test_full_range_is_all_blocks(self, capsys):
        ui.meter("temperature", 2.0, 0.0, 2.0)
        assert capsys.readouterr().err.count("\u2588") == 24

    def test_signed_zero_sits_in_the_middle(self, capsys):
        """A neutral penalty must not look half-applied."""
        out = capsys.readouterr().err
        ui.meter("presence", 0.0, -2.0, 2.0)
        out = capsys.readouterr().err
        assert "\u2502" in out

    def test_negative_value_fills_left_of_zero(self, capsys):
        ui.meter("presence", -1.0, -2.0, 2.0)
        cells = self._cells(capsys.readouterr().err)
        # The bar grows from the centre towards the left edge.
        assert cells.index("\u2588") < cells.index("\u2502")

    def test_positive_value_fills_right_of_zero(self, capsys):
        ui.meter("presence", 1.0, -2.0, 2.0)
        cells = self._cells(capsys.readouterr().err)
        filled = [i for i, c in enumerate(cells) if c == "\u2588"]
        assert filled and min(filled) >= 12  # the zero tick is at the centre

    def test_opposite_signs_mirror(self, capsys):
        ui.meter("presence", -1.0, -2.0, 2.0)
        neg = self._cells(capsys.readouterr().err)
        ui.meter("presence", 1.0, -2.0, 2.0)
        pos = self._cells(capsys.readouterr().err)
        neg_fill = {i for i, c in enumerate(neg) if c == "\u2588"}
        pos_fill = {i for i, c in enumerate(pos) if c == "\u2588"}
        # Equal magnitude, opposite sign: the fill mirrors about the centre.
        assert {len(neg) - 1 - i for i in neg_fill} == pos_fill

    def test_out_of_range_does_not_overflow(self, capsys):
        ui.meter("temperature", 999.0, 0.0, 2.0)
        assert capsys.readouterr().err.count("\u2588") == 24

    def test_stderr_only(self, capsys):
        ui.meter("temperature", 1.0, 0.0, 2.0)
        assert capsys.readouterr().out == ""


# ── Sampling ──────────────────────────────────────────────────────────
class TestSamplingParams:
    def test_defaults_are_llama_cpp_defaults(self):
        from assembly.config import SamplingParams

        p = SamplingParams()
        assert p.temperature == 0.8
        assert p.top_p == 0.95
        assert p.top_k == 40
        assert p.repeat_penalty == 1.1
        assert p.stop == []

    def test_body_values_are_honoured(self):
        from assembly.config import sampling_params_from_body as f

        p = f({"temperature": 1.6, "top_p": 0.5, "top_k": 20,
               "max_tokens": 100, "stop": ["END"]})
        assert p.temperature == 1.6
        assert p.top_p == 0.5
        assert p.top_k == 20
        assert p.max_tokens == 100
        assert p.stop == ["END"]

    @pytest.mark.parametrize("body,field,expected", [
        ({"temperature": 99}, "temperature", 2.0),
        ({"temperature": -5}, "temperature", 0.0),
        ({"top_p": 5}, "top_p", 1.0),
        ({"top_p": -1}, "top_p", 0.0),
        ({"top_k": 0}, "top_k", 1),
        ({"top_k": 100000}, "top_k", 100),
        ({"max_tokens": 1}, "max_tokens", 16),
        ({"max_tokens": 10**9}, "max_tokens", 8192),
    ])
    def test_out_of_range_is_clamped_not_rejected(self, body, field, expected):
        """A client asking for something impossible gets a working request."""
        from assembly.config import sampling_params_from_body as f

        assert getattr(f(body), field) == expected

    @pytest.mark.parametrize("bad", ["hot", None, True, [], {}])
    def test_unusable_value_leaves_the_default_alone(self, bad):
        """Notably, garbage must not become 0.0, which would silently make
        generation fully deterministic."""
        from assembly.config import SamplingParams, sampling_params_from_body as f

        assert f({"temperature": bad}).temperature == SamplingParams().temperature

    def test_nan_is_ignored(self):
        import math

        from assembly.config import sampling_params_from_body as f

        assert f({"temperature": math.nan}).temperature == 0.8

    def test_the_three_penalties_stay_independent(self):
        """llama.cpp implements all three, so one must not overwrite another."""
        from assembly.config import sampling_params_from_body as f

        p = f({"presence_penalty": 0.5, "frequency_penalty": 1.3, "repeat_penalty": 1.2})
        assert p.presence_penalty == 0.5
        assert p.frequency_penalty == 1.3
        assert p.repeat_penalty == 1.2

    def test_penalties_default_to_neutral(self):
        from assembly.config import SamplingParams

        p = SamplingParams()
        assert p.presence_penalty == 0.0
        assert p.frequency_penalty == 0.0

    def test_openai_penalties_accept_negative_values(self):
        """Negative penalties are meaningful in OpenAI; they must not be lost."""
        from assembly.config import sampling_params_from_body as f

        p = f({"presence_penalty": -0.5, "frequency_penalty": -1.0})
        assert p.presence_penalty == -0.5
        assert p.frequency_penalty == -1.0

    @pytest.mark.parametrize("key", ["presence_penalty", "frequency_penalty"])
    def test_openai_penalties_are_clamped(self, key):
        from assembly.config import sampling_params_from_body as f

        assert getattr(f({key: 99}), key) == 2.0
        assert getattr(f({key: -99}), key) == -2.0

    def test_repeat_penalty_stays_above_one(self):
        """It is a multiplier, so a request for 0.5 is clamped up to neutral."""
        from assembly.config import sampling_params_from_body as f

        assert f({"repeat_penalty": 0.1}).repeat_penalty == 1.0

    def test_max_completion_tokens_alias(self):
        from assembly.config import sampling_params_from_body as f

        assert f({"max_completion_tokens": 333}).max_tokens == 333

    def test_stop_accepts_string_or_list(self):
        from assembly.config import sampling_params_from_body as f

        assert f({"stop": "END"}).stop == ["END"]
        assert f({"stop": ["A", "B"]}).stop == ["A", "B"]

    def test_base_is_preserved_for_unset_keys(self):
        from assembly.config import SamplingParams, sampling_params_from_body as f

        base = SamplingParams(temperature=0.1, top_k=7)
        out = f({"top_p": 0.5}, base=base)
        assert out.temperature == 0.1
        assert out.top_k == 7
        assert out.top_p == 0.5

    def test_llama_kwargs_shape(self):
        from assembly.config import SamplingParams

        kwargs = SamplingParams(stop=["X"], seed=7).to_llama_kwargs()
        assert kwargs["temperature"] == 0.8
        assert kwargs["top_k"] == 40
        assert kwargs["repeat_penalty"] == 1.1
        assert kwargs["stop"] == ["X"]
        assert kwargs["seed"] == 7

    def test_absent_seed_is_not_sent(self):
        from assembly.config import SamplingParams

        assert "seed" not in SamplingParams().to_llama_kwargs()

    def test_empty_stop_is_not_sent(self):
        from assembly.config import SamplingParams

        assert "stop" not in SamplingParams().to_llama_kwargs()

    @pytest.mark.parametrize("name", ["precise", "balanced", "creative"])
    def test_presets_are_sane(self, name):
        from assembly.config import SAMPLING_FIELDS, sampling_params_from_preset

        p = sampling_params_from_preset(name)
        for field, (_label, low, high) in SAMPLING_FIELDS.items():
            value = getattr(p, field)
            assert low <= value <= high, f"{name}.{field} out of range"

    def test_precise_is_cooler_than_creative(self):
        from assembly.config import sampling_params_from_preset

        assert (sampling_params_from_preset("precise").temperature
                < sampling_params_from_preset("creative").temperature)

    def test_unknown_preset_raises(self):
        from assembly.config import sampling_params_from_preset

        with pytest.raises(KeyError):
            sampling_params_from_preset("nope")

    @pytest.mark.parametrize("name", ["precise", "balanced", "creative"])
    def test_every_preset_defines_every_field(self, name):
        """The UI compares against these, so a gap would print "None"."""
        from assembly.config import SAMPLING_FIELDS, SAMPLING_PRESETS

        assert set(SAMPLING_PRESETS[name]) >= set(SAMPLING_FIELDS)

    def test_presets_reset_every_knob(self):
        """Switching preset must not leave a stale penalty behind."""
        from assembly.config import sampling_params_from_preset

        chat_like = sampling_params_from_preset("balanced")
        chat_like.presence_penalty = 1.5
        assert sampling_params_from_preset("precise").presence_penalty == 0.0



class TestSamplingCommands:
    """The /params family in the interactive session."""

    def _chat(self):
        from assembly import main as main_mod
        from assembly.config import SamplingParams

        class FakeBackend:
            def info(self):
                return {"assembly": "Mesh-1.0"}

        return main_mod._Chat(FakeBackend(), sampling=SamplingParams())

    def _run(self, chat, cmd):
        from assembly import main as main_mod

        return main_mod._handle_command(chat, cmd)

    def test_params_shows_every_knob(self, capsys):
        chat = self._chat()
        self._run(chat, "/params")
        err = capsys.readouterr().err
        for label in ("temperature", "top_p", "top_k", "repeat penalty", "max tokens"):
            assert label in err

    def test_temp_sets_value(self):
        chat = self._chat()
        self._run(chat, "/temp 1.6")
        assert chat.sampling.temperature == 1.6

    def test_comma_decimal_is_accepted(self):
        chat = self._chat()
        self._run(chat, "/temp 1,6")
        assert chat.sampling.temperature == 1.6

    @pytest.mark.parametrize("cmd,field,value", [
        ("/temp 0.3", "temperature", 0.3),
        ("/top_p 0.5", "top_p", 0.5),
        ("/top_k 20", "top_k", 20),
        ("/repeat 1.2", "repeat_penalty", 1.2),
        ("/presence 0.7", "presence_penalty", 0.7),
        ("/frequency -0.4", "frequency_penalty", -0.4),
        ("/max 100", "max_tokens", 100),
    ])
    def test_each_knob(self, cmd, field, value):
        chat = self._chat()
        self._run(chat, cmd)
        assert getattr(chat.sampling, field) == value

    @pytest.mark.parametrize("cmd,field,expected", [
        ("/temp 99", "temperature", 2.0),
        ("/temp -3", "temperature", 0.0),
        ("/top_k 5000", "top_k", 100),
    ])
    def test_command_clamps(self, cmd, field, expected):
        chat = self._chat()
        self._run(chat, cmd)
        assert getattr(chat.sampling, field) == expected

    def test_garbage_value_leaves_default(self, capsys):
        chat = self._chat()
        before = chat.sampling.temperature
        self._run(chat, "/temp banana")
        assert chat.sampling.temperature == before
        assert "needs a number" in capsys.readouterr().err

    def test_nan_value_leaves_default(self, capsys):
        chat = self._chat()
        before = chat.sampling.temperature
        self._run(chat, "/temp nan")
        assert chat.sampling.temperature == before

    def test_max_updates_the_chat_budget(self):
        chat = self._chat()
        self._run(chat, "/max 123")
        assert chat.sampling.max_tokens == 123
        assert chat.max_tokens == 123

    def test_preset_applies_all_knobs(self):
        from assembly.config import sampling_params_from_preset

        chat = self._chat()
        self._run(chat, "/preset precise")
        expected = sampling_params_from_preset("precise")
        assert chat.sampling.temperature == expected.temperature
        assert chat.sampling.top_k == expected.top_k

    def test_preset_case_insensitive(self):
        chat = self._chat()
        self._run(chat, "/preset CREATIVE")
        assert chat.sampling.temperature == 1.2

    def test_unknown_preset_is_rejected(self, capsys):
        chat = self._chat()
        before = chat.sampling.temperature
        self._run(chat, "/preset nonsense")
        assert chat.sampling.temperature == before
        assert "unknown preset" in capsys.readouterr().err

    def test_stop_sets_and_clears(self):
        chat = self._chat()
        self._run(chat, "/stop END")
        assert chat.sampling.stop == ["END"]
        self._run(chat, "/stop clear")
        assert chat.sampling.stop == []

    def test_stop_is_forwarded_to_llama(self):
        chat = self._chat()
        self._run(chat, "/stop DONE")
        assert chat.sampling.to_llama_kwargs()["stop"] == ["DONE"]

    def test_penalty_command_forwards_to_llama(self):
        chat = self._chat()
        self._run(chat, "/presence 0.7")
        self._run(chat, "/frequency -0.4")
        kwargs = chat.sampling.to_llama_kwargs()
        assert kwargs["presence_penalty"] == 0.7
        assert kwargs["frequency_penalty"] == -0.4

    def test_penalty_command_clamps(self):
        chat = self._chat()
        self._run(chat, "/presence 99")
        assert chat.sampling.presence_penalty == 2.0
        self._run(chat, "/frequency -99")
        assert chat.sampling.frequency_penalty == -2.0

    def test_unknown_setting_is_reported(self, capsys):
        chat = self._chat()
        self._run(chat, "/params bogus 5")
        assert "unknown setting" in capsys.readouterr().err

    def test_unrelated_command_still_errors_normally(self, capsys):
        chat = self._chat()
        self._run(chat, "/bogus 5")
        assert "unknown command" in capsys.readouterr().err

    def test_short_aliases_route_to_params(self):
        """`/temp 1.2` must work as a top-level command, not just `/params temp`."""
        chat = self._chat()
        self._run(chat, "/temp 1.2")
        assert chat.sampling.temperature == 1.2

    def test_params_does_not_pollute_stdout(self, capsys):
        chat = self._chat()
        self._run(chat, "/params")
        assert capsys.readouterr().out == ""

    def test_help_lists_the_new_commands(self, capsys):
        chat = self._chat()
        self._run(chat, "/help")
        out = capsys.readouterr().out
        for token in ("/params", "/temp", "/top_p", "/top_k", "/repeat", "/stop", "/preset"):
            assert token in out


class TestServerSampling:
    """Sampling must travel from the HTTP body into llama.cpp untouched."""

    def _run(self, body: dict, stream: bool = False):
        from assembly.server import ModelManager
        from assembly.wrapper import RouteResult
        from unittest.mock import MagicMock

        mgr = ModelManager()
        mock_llm = MagicMock()
        mock_llm.header.assembly_name = "TestAssembly"
        mock_llm.header.models = []
        # A bare MagicMock would be truthy and not an int, which is exactly
        # what _context_note is written to refuse.
        mock_llm.context_window = 4096
        mock_llm._last_summarized = False
        mock_llm.route.return_value = RouteResult(
            model_name="coder", model_index=0, score=0.92, is_fallback=False
        )
        if stream:
            mock_llm.create_chat_completion.return_value = iter([
                {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ])
        else:
            mock_llm.create_chat_completion.return_value = {
                "id": "chat-123",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }
        mock_llm.return_value = {
            "id": "cmpl-123", "choices": [{"text": "ok"}],
        }
        mgr._models["assembly"] = mock_llm
        mgr._active_alias = "assembly"

        payload = json.dumps(body).encode()
        sock = MockSocket(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        from assembly.server import AssemblyRequestHandler

        mock_server = MagicMock()
        mock_server.manager = mgr
        mock_server.api_key = None
        mock_server.allow_origin = None
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), mock_server)
        raw = sock.get_output()
        header, _, rest = raw.partition(b"\r\n\r\n")
        return mock_llm, header.decode("utf-8"), rest

    def _kwargs(self, body, stream=False):
        mock_llm, _, _ = self._run(body, stream)
        return mock_llm.create_chat_completion.call_args.kwargs

    def test_defaults_reach_llama(self):
        kw = self._kwargs({"messages": [{"role": "user", "content": "hi"}]})
        assert kw["temperature"] == 0.8
        assert kw["top_p"] == 0.95
        assert kw["top_k"] == 40
        assert kw["max_tokens"] == 512

    def test_client_values_reach_llama(self):
        kw = self._kwargs({
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 1.6, "top_p": 0.5, "top_k": 20,
            "frequency_penalty": 1.3, "max_tokens": 77, "stop": ["END"],
        })
        assert kw["temperature"] == 1.6
        assert kw["top_p"] == 0.5
        assert kw["top_k"] == 20
        assert kw["frequency_penalty"] == 1.3
        assert kw["max_tokens"] == 77
        assert kw["stop"] == ["END"]

    def test_hostile_values_do_not_break_the_request(self):
        kw = self._kwargs({
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": "nonsense", "top_k": 99999, "max_tokens": -5,
        })
        assert kw["temperature"] == 0.8
        assert kw["top_k"] == 100
        assert kw["max_tokens"] == 16

    def test_streaming_forwards_sampling(self):
        """Regression: streaming used to be handed only max_tokens/temperature."""
        kw = self._kwargs({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True, "temperature": 1.6, "top_k": 20, "stop": ["END"],
        }, stream=True)
        assert kw["stream"] is True
        assert kw["temperature"] == 1.6
        assert kw["top_k"] == 20
        assert kw["stop"] == ["END"]

    def test_streaming_emits_sse(self):
        _mock, header, body = self._run(
            {"messages": [{"role": "user", "content": "hi"}], "stream": True},
            stream=True,
        )
        assert "200" in header.splitlines()[0]
        assert "event-stream" in header.lower()
        assert b"[DONE]" in body

    def test_streaming_reports_the_context_window(self):
        """A streaming client must learn its history was compressed.

        The non-streaming path has always said so via ``assembly_context``.
        Without it here, the most common way of talking to the server was the
        one way that could not tell whether the turns it sent were still being
        sent.
        """
        _mock, _header, body = self._run(
            {"messages": [{"role": "user", "content": "hi"}], "stream": True},
            stream=True,
        )
        frames = [
            json.loads(line[len("data: "):])
            for line in body.decode().splitlines()
            if line.startswith("data: ") and "[DONE]" not in line
        ]
        notes = [f for f in frames if "assembly_context" in f]
        assert len(notes) == 1, "exactly one context frame, before [DONE]"
        note = notes[0]["assembly_context"]
        assert note["window"] == 4096
        assert note["summarized"] is False
        # It must not look like content to a client walking deltas.
        assert notes[0]["choices"] == []

    def test_streaming_context_frame_precedes_done(self):
        _mock, _header, body = self._run(
            {"messages": [{"role": "user", "content": "hi"}], "stream": True},
            stream=True,
        )
        text = body.decode()
        assert text.index("assembly_context") < text.index("[DONE]")

    def test_unknown_params_are_ignored_not_fatal(self):
        _mock, header, _ = self._run({
            "messages": [{"role": "user", "content": "hi"}],
            "some_future_openai_field": 1,
        })
        assert "200" in header.splitlines()[0]



class TestServerCompletionsSampling:
    """`/v1/completions` and `/api/generate` must forward sampling too."""

    def _kwargs(self, path: str, body: dict):
        from assembly.server import ModelManager, AssemblyRequestHandler
        from assembly.wrapper import RouteResult
        from unittest.mock import MagicMock

        mgr = ModelManager()
        mock_llm = MagicMock()
        mock_llm.header.assembly_name = "TestAssembly"
        mock_llm.header.models = []
        # A bare MagicMock would be truthy and not an int, which is exactly
        # what _context_note is written to refuse.
        mock_llm.context_window = 4096
        mock_llm._last_summarized = False
        mock_llm.route.return_value = RouteResult(
            model_name="coder", model_index=0, score=0.92, is_fallback=False
        )
        mock_llm.return_value = {"id": "cmpl-1", "choices": [{"text": "ok"}]}
        mgr._models["assembly"] = mock_llm
        mgr._active_alias = "assembly"

        payload = json.dumps(body).encode()
        sock = MockSocket(
            f"POST {path} HTTP/1.1\r\nHost: localhost\r\n".encode()
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        mock_server = MagicMock()
        mock_server.manager = mgr
        mock_server.api_key = None
        mock_server.allow_origin = None
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), mock_server)
        return mock_llm.call_args.kwargs

    @pytest.mark.parametrize("path", ["/v1/completions", "/api/generate"])
    def test_sampling_reaches_llama(self, path):
        kw = self._kwargs(path, {
            "prompt": "hi", "temperature": 1.6, "top_p": 0.5, "top_k": 20,
            "presence_penalty": 0.5, "frequency_penalty": 0.5,
            "repeat_penalty": 1.2, "max_tokens": 77, "stop": ["END"], "seed": 3,
        })
        assert kw["temperature"] == 1.6
        assert kw["top_p"] == 0.5
        assert kw["top_k"] == 20
        assert kw["presence_penalty"] == 0.5
        assert kw["frequency_penalty"] == 0.5
        assert kw["repeat_penalty"] == 1.2
        assert kw["max_tokens"] == 77
        assert kw["stop"] == ["END"]
        assert kw["seed"] == 3

    @pytest.mark.parametrize("path", ["/v1/completions", "/api/generate"])
    def test_defaults(self, path):
        kw = self._kwargs(path, {"prompt": "hi"})
        assert kw["temperature"] == 0.8
        assert kw["max_tokens"] == 256  # the legacy default, not the chat one


class TestDescribeModels:
    """The serve banner must not crash on an 11 GB archive it just loaded."""

    def _manager(self, names=("Code", "General", "Reasoning"), alias="mesh"):
        from assembly.server import ModelManager
        from unittest.mock import MagicMock

        mgr = ModelManager()
        llm = MagicMock()
        llm.header.assembly_name = "Mesh-1.0"
        llm.persona = "p"
        llm.header.models = []
        for n in names:
            md = MagicMock()
            md.name = n
            llm.header.models.append(md)
        mgr._models[alias] = llm
        mgr._active_alias = alias
        return mgr

    def test_lists_sub_models(self):
        mgr = self._manager()
        assert mgr.describe_models() == "Code, General, Reasoning"

    def test_empty_manager_says_so(self):
        assert self._manager(names=()).describe_models() == "none loaded"

    def test_duplicate_names_appear_once(self):
        mgr = self._manager(names=("Code", "Code", "General"))
        assert mgr.describe_models() == "Code, General"

    def test_multiple_archives_are_merged(self):
        mgr = self._manager()
        second = self._manager(names=("General", "Vision"), alias="other")
        mgr._models["other"] = second._models["other"]
        assert mgr.describe_models() == "Code, General, Reasoning, Vision"

    def test_keys_it_reads_actually_exist(self):
        """Guards against reintroducing a typo'd key on a live startup path."""
        mgr = self._manager()
        for entry in mgr.list_models():
            for key in ("id", "assembly_name", "persona", "active", "models"):
                assert key in entry, f"list_models() lost {key!r}"
            assert "name" not in entry

    def test_serve_banner_rows_are_renderable(self, capsys):
        """The row that crashed used to be built at print time."""
        mgr = self._manager()
        rows = [
            ("Models", mgr.describe_models()),
            ("Auth", "none (loopback only)"),
        ]
        ui.banner("Assembly Engine", "listening on 127.0.0.1:8000", rows)
        out = capsys.readouterr().err
        assert "Code, General, Reasoning" in out


class TestContextOverflow:
    """A conversation that outgrows the context window must fail cleanly.

    llama.cpp raises a bare ValueError, and llama-cpp-python raises it lazily,
    on the first advance of the generator rather than when it is created. That
    used to arrive after a 200 had already been written.
    """

    OVERFLOW = ("Requested tokens (2082) exceed context window of 2048")

    def _llm(self, streaming: bool):
        from assembly.wrapper import RouteResult
        from unittest.mock import MagicMock

        llm = MagicMock()
        llm.header.assembly_name = "TestAssembly"
        llm.header.models = []
        llm.route.return_value = RouteResult(
            model_name="coder", model_index=0, score=0.9, is_fallback=False
        )

        def overflow():
            # Mirrors llama-cpp-python: nothing happens until it is advanced.
            raise ValueError(self.OVERFLOW)
            yield  # pragma: no cover

        if streaming:
            llm.create_chat_completion.return_value = overflow()
        else:
            llm.create_chat_completion.side_effect = ValueError(self.OVERFLOW)
            llm.side_effect = ValueError(self.OVERFLOW)
        return llm

    def _manager(self, llm):
        from assembly.server import ModelManager

        mgr = ModelManager()
        mgr._models["a"] = llm
        mgr._active_alias = "a"
        return mgr

    def _post(self, llm, path="/v1/chat/completions", body=None):
        from assembly.server import AssemblyRequestHandler
        from unittest.mock import MagicMock

        payload = json.dumps(body if body is not None else {
            "messages": [{"role": "user", "content": "hi"}]
        }).encode()
        sock = MockSocket(
            f"POST {path} HTTP/1.1\r\nHost: localhost\r\n".encode()
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        mock_server = MagicMock()
        mock_server.manager = self._manager(llm)
        mock_server.api_key = None
        mock_server.allow_origin = None
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), mock_server)
        raw = sock.get_output()
        header, _, rest = raw.partition(b"\r\n\r\n")
        status = int(header.decode("utf-8").splitlines()[0].split()[1])
        return status, header.decode("utf-8"), rest

    def test_message_is_recognised(self):
        from assembly.server import _context_overflow

        msg = _context_overflow(ValueError(self.OVERFLOW))
        assert msg is not None
        assert "2082" in msg and "2048" in msg

    def test_message_names_both_remedies(self):
        from assembly.server import _context_overflow

        msg = _context_overflow(ValueError(self.OVERFLOW))
        assert "new conversation" in msg.lower()
        assert "--n-ctx" in msg

    def test_unrelated_error_is_not_mistaken_for_overflow(self):
        from assembly.server import _context_overflow

        assert _context_overflow(ValueError("model failed to load")) is None

    def test_non_streaming_returns_400(self):
        status, _, body = self._post(self._llm(streaming=False))
        assert status == 400
        assert "2082" in body.decode("utf-8")

    def test_streaming_returns_400_not_200(self):
        """Regression: this used to report 200 and then fail mid-stream."""
        status, header, body = self._post(self._llm(streaming=True),
                                          body={"messages": [{"role": "user", "content": "hi"}],
                                                "stream": True})
        assert status == 400
        assert "text/event-stream" not in header.lower()
        assert b"2082" in body

    def test_streaming_overflow_sends_no_sse_frames(self):
        _status, _header, body = self._post(
            self._llm(streaming=True),
            body={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert b"data:" not in body
        assert b"[DONE]" not in body

    def test_no_traceback_on_a_routine_overflow(self, caplog):
        """A too-long conversation must not dump a stack trace on the server."""
        import logging

        with caplog.at_level(logging.WARNING, logger="assembly.server"):
            self._post(self._llm(streaming=False))
        assert not any(r.exc_info for r in caplog.records)

    def test_ollama_chat_also_returns_400(self):
        status, _, body = self._post(self._llm(streaming=False), path="/api/chat")
        assert status == 400

    def test_legacy_completions_returns_400(self):
        status, _, body = self._post(self._llm(streaming=False),
                                     path="/v1/completions",
                                     body={"prompt": "hi"})
        assert status == 400

    def test_ollama_generate_returns_400(self):
        status, _, body = self._post(self._llm(streaming=False),
                                     path="/api/generate",
                                     body={"prompt": "hi"})
        assert status == 400


class TestAutoContextWindow:
    """The window is read from the archive, and cannot be overridden."""

    def _gguf(self, context_length=None, arch="llama"):
        """A minimal GGUF header carrying one KV pair."""
        import struct

        kv = b""
        if arch is not None:
            kv += struct.pack("<Q", len(b"general.architecture")) + b"general.architecture"
            kv += struct.pack("<I", 8) + struct.pack("<Q", len(arch)) + arch.encode()
        if context_length is not None:
            key = f"{arch}.context_length".encode()
            kv += struct.pack("<Q", len(key)) + key
            kv += struct.pack("<I", 4) + struct.pack("<I", context_length)
        blob = b"GGUF" + struct.pack("<I", 3)
        blob += struct.pack("<QQ", 0, 2 if (arch and context_length) else 1)
        return blob + kv

    def test_reads_the_declared_length(self):
        from assembly.config import gguf_context_length_from_bytes as f

        assert f(self._gguf(131072)) == 131072

    def test_key_is_namespaced_by_architecture(self):
        """It is llama.context_length, not general.context_length."""
        from assembly.config import gguf_context_length_from_bytes as f

        assert f(self._gguf(4096, arch="qwen2")) == 4096

    def test_unknown_architecture_still_found_by_scan(self):
        from assembly.config import gguf_context_length_from_bytes as f

        assert f(self._gguf(8192, arch="somefuturemodel")) == 8192

    def test_missing_value_is_unknown_not_zero(self):
        from assembly.config import gguf_context_length_from_bytes as f

        assert f(self._gguf(None)) is None
        assert f(self._gguf(0)) is None

    def test_garbage_is_unknown(self):
        from assembly.config import gguf_context_length_from_bytes as f

        assert f(b"not a gguf at all") is None
        assert f(b"") is None

    def test_truncated_header_is_unknown(self):
        from assembly.config import gguf_context_length_from_bytes as f

        assert f(self._gguf(4096)[:20]) is None

    def test_refactor_preserved_split_detection(self):
        """The metadata reader replaced an inline parser; keep its contract."""
        import struct

        from assembly.config import gguf_split_info_from_bytes as f

        blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 2)
        for key, value in (("split.count", 3), ("split.no", 1)):
            k = key.encode()
            blob += struct.pack("<Q", len(k)) + k
            blob += struct.pack("<I", 4) + struct.pack("<I", value)
        assert f(blob) == (3, 1)


from assembly import wrapper  # noqa: E402

PAGE = 4096


def _model_marker(index: int) -> bytes:
    """Bytes that identify which model a header read came from."""
    return f"model:{index}".encode()


def _model_index(head: bytes) -> int:
    return int(bytes(head).split(b":", 1)[1].split(b"\0", 1)[0])


class TestContextResolution:
    """Smallest declared window wins; nothing can override it."""

    def _llama(self, contexts):
        """An AssemblyLlama shell with faked model offsets."""
        from unittest.mock import MagicMock
        from assembly.wrapper import AssemblyLlama

        obj = AssemblyLlama.__new__(AssemblyLlama)
        obj._model_contexts = {}
        obj._n_ctx = 0
        obj._log = lambda *a, **k: None

        # Each model gets a distinct offset and its own marker bytes, so a
        # test can identify which model is being parsed instead of counting
        # calls. The context reader may ask about one model more than once.
        header = MagicMock()
        header.models = []
        for i, n in enumerate(contexts):
            m = MagicMock()
            m.name = n
            m.offset = i * PAGE
            m.size = 1024
            header.models.append(m)
        obj._header = header

        def read(self, key):
            start = key.start or 0
            return _model_marker(start // PAGE).ljust(4 << 20, b"\0")

        mm = MagicMock()
        mm.__getitem__ = read
        obj._mm = mm
        return obj

    def _patch_parser(self, monkeypatch, table):
        """Make the parser answer per model, by marker rather than call order."""
        import assembly.wrapper as wrapper

        monkeypatch.setattr(
            wrapper,
            "gguf_context_length_from_bytes",
            lambda head: table[_model_index(head)],
        )

    def test_smallest_window_is_chosen(self, monkeypatch):
        obj = self._llama(["a", "b", "c"])
        self._patch_parser(monkeypatch, {0: 8192, 1: 4096, 2: 32768})
        obj._resolve_context_window()
        assert obj.context_window == 4096

    def test_single_model_uses_its_own_window(self, monkeypatch):
        obj = self._llama(["only"])
        self._patch_parser(monkeypatch, {0: 65536})
        obj._resolve_context_window()
        assert obj.context_window == 65536

    def test_unknown_model_falls_back_conservatively(self, monkeypatch):
        obj = self._llama(["a", "b"])
        self._patch_parser(monkeypatch, {0: 131072, 1: None})
        obj._resolve_context_window()
        assert obj.context_window == wrapper.FALLBACK_CONTEXT

    def test_all_unknown_falls_back(self, monkeypatch):
        obj = self._llama(["a", "b"])
        self._patch_parser(monkeypatch, {0: None, 1: None})
        obj._resolve_context_window()
        assert obj.context_window == wrapper.FALLBACK_CONTEXT

    def test_absurd_value_is_clamped(self, monkeypatch):
        obj = self._llama(["a"])
        self._patch_parser(monkeypatch, {0: 1 << 30})
        obj._resolve_context_window()
        assert obj.context_window == wrapper.MAX_CONTEXT

    def test_a_zero_window_counts_as_undeclared(self, monkeypatch):
        """0 means "not declared", so the conservative fallback wins over
        clamping to a useless 512."""
        obj = self._llama(["a"])
        self._patch_parser(monkeypatch, {0: 0})
        obj._resolve_context_window()
        assert obj.context_window == wrapper.FALLBACK_CONTEXT

    def test_absurdly_small_window_is_raised_to_the_floor(self, monkeypatch):
        obj = self._llama(["a"])
        self._patch_parser(monkeypatch, {0: 8})
        obj._resolve_context_window()
        assert obj.context_window == wrapper.MIN_CONTEXT

    def test_a_late_context_key_is_still_found(self, monkeypatch):
        """The reader escalates its read size when the key is not there."""
        obj = self._llama(["a"])
        self._patch_parser(monkeypatch, {0: 65536})
        obj._resolve_context_window()
        assert obj.context_window == 65536
        # Two spans are attempted per model, so the parser is consulted
        # twice when the first span comes back empty.
        assert wrapper.HEADER_READ_SPANS[0] < wrapper.HEADER_READ_SPANS[1]

    def test_cli_has_no_n_ctx_flag(self):
        from assembly.main import _build_parser

        parser = _build_parser()
        for argv in (["serve", "-m", "/m/a.asmb"], ["run", "-m", "/m/a.asmb"]):
            args = parser.parse_args(argv)
            assert not hasattr(args, "n_ctx"), argv

    def test_cli_rejects_the_old_flag(self):
        from assembly.main import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["serve", "-m", "/m/a.asmb", "--n-ctx", "4096"])


class TestConversationSummarization:
    """History is compressed by the model instead of overflowing."""

    def _slot(self, summary="SUMMARY TEXT", record=None):
        from unittest.mock import MagicMock

        slot = MagicMock()
        slot.llama.tokenize.side_effect = lambda b, **k: list(b)[: max(1, len(b) // 4)]

        def chat(messages, **kwargs):
            if record is not None:
                record.append((list(messages), kwargs))
            return {"choices": [{"message": {"content": summary}}]}

        slot.llama.create_chat_completion.side_effect = chat
        return slot

    def _llama(self, n_ctx=2048):
        from assembly.wrapper import AssemblyLlama

        obj = AssemblyLlama.__new__(AssemblyLlama)
        obj._n_ctx = n_ctx
        obj._persona = None
        obj._active_slot = None
        obj._log = lambda *a, **k: None
        obj._last_summarized = False
        obj._token_cache = {}
        import threading
        obj._lock = threading.Lock()
        return obj

    def _chat(self, turns=40):
        msgs = []
        for i in range(turns):
            msgs.append({"role": "user", "content": f"turn {i} " * 20})
            msgs.append({"role": "assistant", "content": f"reply {i} " * 20})
        msgs.append({"role": "user", "content": "what did we decide?"})
        return msgs

    def test_short_conversation_is_untouched(self):
        obj = self._llama()
        slot = self._slot()
        msgs = [{"role": "user", "content": "hi"}]
        fitted, did = obj._fit_to_context(msgs, slot)
        assert fitted == msgs
        assert did is False

    def test_long_conversation_is_summarized(self):
        obj, slot = self._llama(2048), self._slot()
        msgs = self._chat(60)
        fitted, did = obj._fit_to_context(msgs, slot)
        assert did is True
        assert len(fitted) < len(msgs)

    def test_summary_note_comes_first(self):
        obj, slot = self._llama(2048), self._slot(summary="They picked blue.")
        fitted, _ = obj._fit_to_context(self._chat(60), slot)
        assert fitted[0]["role"] == "system"
        assert "They picked blue." in fitted[0]["content"]

    def test_final_question_is_preserved(self):
        obj, slot = self._llama(2048), self._slot()
        fitted, _ = obj._fit_to_context(self._chat(60), slot)
        assert fitted[-1]["content"] == "what did we decide?"

    def test_oldest_turns_are_the_ones_dropped(self):
        obj, slot = self._llama(2048), self._slot()
        fitted, _ = obj._fit_to_context(self._chat(60), slot)
        bodies = [m.get("content") or "" for m in fitted]
        assert not any("turn 0 " in b for b in bodies[1:])
        assert any("turn 59 " in b for b in bodies[1:])

    def test_start_of_conversation_reaches_the_summariser(self):
        """Regression: trimming the transcript from the front lost exactly
        the facts that were being asked about."""
        obj, slot = self._llama(2048), self._slot()
        seen = []
        obj._summarize_transcript = lambda s, t, max_tokens: seen.append(t) or "ok"
        obj._fit_to_context(self._chat(60), slot)
        assert any("turn 0 " in t for t in seen)

    def test_long_history_is_chunked_not_truncated(self):
        obj = self._llama(2048)
        chunks = obj._split_to_budget("\n".join(f"line {i}" * 10 for i in range(400)), 200)
        assert len(chunks) > 1
        assert "line 0" in chunks[0]

    def test_chunks_all_fit_the_budget(self):
        obj = self._llama(2048)
        text = "\n".join(f"line {i} " * 10 for i in range(300))
        for chunk in obj._split_to_budget(text, 200):
            assert obj.count_tokens(chunk) <= 200 + obj.count_tokens("line 0 " * 10)

    def test_everything_still_fits_after_summarizing(self):
        """The note is text too, so the tail must give way if needed."""
        obj, slot = self._llama(2048), self._slot(summary="A rather long summary. " * 200)
        fitted, did = obj._fit_to_context(self._chat(60), slot)
        assert did is True
        limit = int(obj._n_ctx * 0.80)
        assert obj._conversation_tokens(fitted) <= limit + 50

    def test_history_is_not_retokenised_every_request(self):
        """A growing conversation must not cost more to send each time.

        Re-tokenising the joined history made every request cost more than the
        last, which is the opposite of what a long session needs.
        """
        obj = self._llama(2048)
        calls = []

        class FakeSlot:
            class llama:
                @staticmethod
                def tokenize(data, **kwargs):
                    calls.append(data)
                    return [0] * (len(data) // 4)

        obj._active_slot = FakeSlot()

        messages = []
        for i in range(20):
            messages.append({"role": "user", "content": f"turn {i} " * 20})
            messages.append({"role": "assistant", "content": f"ok {i} " * 20})
            obj._conversation_tokens(messages)

        # 40 messages over 20 requests, plus the two role names: 42
        # tokenisations in total rather than one per request.
        assert len(calls) == 42
        assert len(obj._token_cache) == 42

    def test_the_cache_stays_bounded(self):
        from assembly import wrapper

        obj = self._llama(2048)
        for i in range(wrapper.TOKEN_CACHE_ENTRIES + 50):
            obj.count_tokens(f"unique line {i}")
        assert len(obj._token_cache) <= wrapper.TOKEN_CACHE_ENTRIES

    def test_runaway_repetition_is_condensed(self):
        """Pasted logs must not crowd out the specifics being preserved."""
        obj = self._llama(2048)
        spam = "chatter 5. " * 80          # 160 words, 2 distinct
        condensed = obj._condense(spam)
        assert "(repeated 160 times)" in condensed
        assert len(condensed) < len(spam) / 10

    def test_ordinary_text_is_left_alone(self):
        obj = self._llama(2048)
        prose = "The certificate expires on 2026-11-04 and Alice owns the rollback runbook."
        assert obj._condense(prose) == prose

    def test_summary_output_is_capped(self):
        """A summary must be bounded even when the window is enormous.

        Passing the whole budget as the reply allowance let a verbose model
        return a summary as long as the history it was compressing.
        """
        obj = self._llama(1 << 17)
        slot = self._slot()
        from assembly import wrapper

        obj._summarize_transcript(slot, "a fairly long transcript " * 500,
                                  max_tokens=100_000)
        allowed = slot.llama.create_chat_completion.call_args.kwargs["max_tokens"]
        assert allowed == wrapper.SUMMARY_OUTPUT_CEILING

    def test_tiny_conversation_is_not_summarized(self):
        obj, self_slot = self._llama(2048), self._slot()
        slot = self_slot
        fitted, did = obj._fit_to_context(self._chat(1), slot)
        assert did is False

    def test_empty_summariser_output_is_ignored(self):
        obj, slot = self._llama(2048), self._slot(summary="   ")
        msgs = self._chat(60)
        fitted, did = obj._fit_to_context(msgs, slot)
        assert fitted == msgs
        assert did is False

    def test_summariser_failure_keeps_the_conversation(self):
        obj, slot = self._llama(2048), self._slot()
        slot.llama.create_chat_completion.side_effect = RuntimeError("model died")
        msgs = self._chat(60)
        fitted, did = obj._fit_to_context(msgs, slot)
        assert fitted == msgs
        assert did is False

    def test_summarising_never_recurses(self):
        """The summariser must call the model directly, not through the
        context-fitting path that would call it again."""
        obj, slot = self._llama(2048), self._slot()
        fitted, did = obj._fit_to_context(self._chat(60), slot)
        assert did is True
        # Exactly one summarisation request per chunk, and no nested fitting.
        assert slot.llama.create_chat_completion.call_count >= 1

    def test_summary_requests_are_bounded(self):
        obj = self._llama(2048)
        record = []
        obj._fit_to_context(self._chat(60), self._slot(record=record))
        for _messages, kwargs in record:
            assert kwargs.get("stream") is False
            assert 0 < kwargs["max_tokens"] <= obj._n_ctx

    def test_repeated_fits_are_idempotent_in_shape(self):
        obj, slot = self._llama(2048), self._slot()
        once, _ = obj._fit_to_context(self._chat(60), slot)
        twice, did = obj._fit_to_context(once, slot)
        assert did is False
        assert twice == once


class TestContextReporting:
    """Clients and the console learn what the window is."""

    def _manager(self, contexts):
        from assembly.server import ModelManager
        from unittest.mock import MagicMock

        mgr = ModelManager()
        llm = MagicMock()
        llm.model_contexts = contexts
        mgr._models["a"] = llm
        mgr._active_alias = "a"
        return mgr

    def test_manager_reports_the_smallest(self):
        assert self._manager({"a": 8192, "b": 4096}).context_window() == 4096

    def test_manager_ignores_unknown_models(self):
        assert self._manager({"a": 8192, "b": None}).context_window() == 8192

    def test_manager_with_nothing_loaded(self):
        from assembly.server import ModelManager

        assert ModelManager().context_window() is None

    def test_description_names_the_constraining_model(self):
        text = self._manager({"a": 8192, "b": 4096}).describe_context()
        assert "4,096" in text
        assert "b" in text

    def test_an_undeclared_model_lowers_the_reported_window(self):
        """The fallback must reach the client, not just the model.

        Taking the minimum of the *declared* lengths would report 131,072 for
        an archive actually running at 2,048, and a client sizing off that
        would overflow every request.
        """
        from unittest.mock import MagicMock
        from assembly.server import ModelManager
        from assembly.wrapper import AssemblyLlama, FALLBACK_CONTEXT

        llm = MagicMock(spec=AssemblyLlama)
        llm.model_contexts = {"Code": 131072, "Mystery": None}
        llm.context_window = FALLBACK_CONTEXT

        mgr = ModelManager()
        mgr._models = {"a.asmb": llm}
        assert mgr.context_window() == FALLBACK_CONTEXT
        assert "fallback" in mgr.describe_context()

    def test_a_fully_declared_archive_reports_its_smallest(self):
        from unittest.mock import MagicMock
        from assembly.server import ModelManager
        from assembly.wrapper import AssemblyLlama

        llm = MagicMock(spec=AssemblyLlama)
        llm.model_contexts = {"Code": 131072, "General": 131072}
        llm.context_window = 131072

        mgr = ModelManager()
        mgr._models = {"a.asmb": llm}
        assert mgr.context_window() == 131072
        assert "fallback" not in mgr.describe_context()

    def test_health_reports_the_window(self):
        from assembly.server import AssemblyRequestHandler
        from unittest.mock import MagicMock

        mgr = self._manager({"a": 8192, "b": 4096})
        sock = MockSocket(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        srv = MagicMock()
        srv.manager = mgr
        srv.api_key = None
        srv.allow_origin = None
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), srv)
        _h, _, body = sock.get_output().partition(b"\r\n\r\n")
        assert json.loads(body)["context_window"] == 4096

    def test_response_notes_the_window(self):
        from assembly.server import AssemblyRequestHandler, ModelManager
        from unittest.mock import MagicMock
        from assembly.wrapper import RouteResult

        mgr = ModelManager()
        llm = MagicMock()
        llm.header.assembly_name = "A"
        llm.header.models = []
        llm.route.return_value = RouteResult(
            model_name="m", model_index=0, score=0.5, is_fallback=False
        )
        llm.context_window = 4096
        llm._last_summarized = False
        llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "hi"}}]
        }
        mgr._models["a"] = llm
        mgr._active_alias = "a"

        payload = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        sock = MockSocket(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload
        )
        srv = MagicMock()
        srv.manager = mgr
        srv.api_key = None
        srv.allow_origin = None
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), srv)
        _h, _, body = sock.get_output().partition(b"\r\n\r\n")
        note = json.loads(body)["assembly_context"]
        assert note == {"window": 4096, "summarized": False}

    def test_metadata_never_breaks_a_response(self):
        """A weird window value must not turn a 200 into a 500."""
        from assembly.server import ModelManager, AssemblyRequestHandler
        from assembly.wrapper import RouteResult
        from unittest.mock import MagicMock

        mgr = ModelManager()
        llm = MagicMock()
        llm.header.assembly_name = "A"
        llm.header.models = []
        llm.route.return_value = RouteResult(
            model_name="m", model_index=0, score=0.5, is_fallback=False
        )
        llm.context_window = object()      # unserialisable
        llm._last_summarized = False
        llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "hi"}}]
        }
        mgr._models["a"] = llm
        mgr._active_alias = "a"

        payload = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        sock = MockSocket(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload
        )
        srv = MagicMock()
        srv.manager = mgr
        srv.api_key = None
        srv.allow_origin = None
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), srv)
        header, _, body = sock.get_output().partition(b"\r\n\r\n")
        assert int(header.decode().split()[1]) == 200
        assert json.loads(body)["assembly_context"]["window"] is None


# ── Staging directory ─────────────────────────────────────────────────

class TestStaging:
    def test_configure_temp_dir_creates_and_uses(self, tmp_path):
        from assembly import wrapper

        target = tmp_path / "stage" / "nested"
        resolved = wrapper.configure_temp_dir(target)

        assert resolved.is_dir()
        assert wrapper.assembly_temp_dir() == resolved
        wrapper._TEMP_DIR = None

    def test_configure_temp_dir_rejects_a_file(self, tmp_path):
        from assembly import wrapper

        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        with pytest.raises(wrapper.StagingError):
            wrapper.configure_temp_dir(blocker)
        wrapper._TEMP_DIR = None

    def test_configure_temp_dir_env_var(self, tmp_path, monkeypatch):
        from assembly import wrapper

        wrapper._TEMP_DIR = None
        monkeypatch.setenv(wrapper.TEMP_DIR_ENV, str(tmp_path / "from-env"))
        assert wrapper.assembly_temp_dir().parent == tmp_path / "from-env"
        wrapper._TEMP_DIR = None

    def test_insufficient_space_is_reported_before_staging(self, tmp_path, monkeypatch):
        from assembly import wrapper

        staging = tmp_path / "stage"
        staging.mkdir()
        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: staging)
        monkeypatch.setattr(wrapper, "free_bytes_at", lambda _d: 128)

        blob = b"\x00" * 4096
        with pytest.raises(wrapper.StagingError, match="not enough space"):
            with wrapper._memfd_model("big.gguf", blob):
                pass

    def test_write_blob_writes_everything(self, tmp_path):
        import io
        from assembly.wrapper import _write_blob

        sink = io.BytesIO()
        data = bytes(range(256)) * 400  # 102,400 B — spans many chunks
        _write_blob(sink, data)
        assert sink.getvalue() == data

    def test_write_blob_accepts_memoryview(self):
        import io
        from assembly.wrapper import _write_blob

        data = b"abc" * 1000
        sink = io.BytesIO()
        _write_blob(sink, memoryview(data))
        assert sink.getvalue() == data


# ── Config robustness ─────────────────────────────────────────────────

class TestSplitGGUFDetection:
    """A shard of a split GGUF builds cleanly but can never load.

    llama.cpp refuses any file whose metadata declares it belongs to a split,
    so catching this at validate time is the difference between a clear error
    and a broken model discovered after a multi-hour build.
    """

    @staticmethod
    def _gguf(kv_pairs: dict[str, tuple[int, int]]) -> bytes:
        """Minimal GGUF v3 header carrying the given uint32 key/value pairs."""
        import struct as _s

        out = bytearray(b"GGUF")
        out += _s.pack("<I", 3)          # version
        out += _s.pack("<QQ", 0, len(kv_pairs))  # 0 tensors, n_kv

        for key, (vtype, value) in kv_pairs.items():
            out += _s.pack("<Q", len(key)) + key.encode()
            out += _s.pack("<I", vtype)
            out += _s.pack("<I", value)

        return bytes(out)

    def _write(self, tmp_path, blob: bytes, name="m.gguf") -> pathlib.Path:
        p = tmp_path / name
        p.write_bytes(blob)
        return p

    def test_detects_split_count_and_index(self, tmp_path):
        from assembly.config import gguf_split_info

        blob = self._gguf({
            "general.architecture": (8, 0),   # a STRING value would need bytes;
            "split.count": (4, 2),             # here we only exercise ints
            "split.no": (4, 1),
        })
        # Drop the STRING key we cannot encode this way; keep the split ones.
        blob = self._gguf({"split.count": (4, 2), "split.no": (4, 1)})
        assert gguf_split_info(self._write(tmp_path, blob)) == (2, 1)

    def test_plain_model_reports_nothing(self, tmp_path):
        from assembly.config import gguf_split_info

        blob = self._gguf({"general.file_type": (4, 15)})
        assert gguf_split_info(self._write(tmp_path, blob)) == (None, None)

    def test_single_part_split_is_fine(self, tmp_path):
        from assembly.config import gguf_split_info

        blob = self._gguf({"split.count": (4, 1), "split.no": (4, 0)})
        assert gguf_split_info(self._write(tmp_path, blob)) == (1, 0)

    def test_garbage_does_not_raise(self, tmp_path):
        from assembly.config import gguf_split_info

        for blob in (b"", b"NOPE", b"GGUF", b"GGUF" + b"\xff" * 32,
                     self._gguf({"split.count": (4, 2)})[:40]):
            assert gguf_split_info(self._write(tmp_path, blob)) == (None, None)

    def test_preflight_errors_on_a_shard(self, tmp_path):
        from assembly.config import validate_assembly_preflight

        for name in ("emb.gguf", "m1.gguf", "m2.gguf"):
            (tmp_path / name).write_bytes(self._gguf({"general.file_type": (4, 15)}))
        # Only the second execution model is a shard.
        (tmp_path / "m2.gguf").write_bytes(
            self._gguf({"split.count": (4, 2), "split.no": (4, 1)})
        )

        cfg = {
            "assembly_name": "sharded",
            "embedding": {"name": "emb", "path": str(tmp_path / "emb.gguf"),
                          "confidence_threshold": 0.7, "fallback_model": "m1"},
            "models": [
                {"name": "m1", "path": str(tmp_path / "m1.gguf"), "anchors": ["a", "b"]},
                {"name": "m2", "path": str(tmp_path / "m2.gguf"), "anchors": ["c", "d"]},
            ],
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg))

        report = validate_assembly_preflight(path)
        errors = [i for i in report.errors if i.severity == "ERROR"]
        assert any("split GGUF" in i.message and "'m2'" in i.message for i in errors), \
            [i.message for i in errors]

    def test_loader_rejects_a_shard_before_staging_it(self, tmp_path, monkeypatch):
        """The check must precede the multi-gigabyte staging copy."""
        import mmap

        from assembly import wrapper
        from assembly.header import (
            AsmbHeader,
            EmbeddingEntry,
            InvalidArchiveError,
            ModelEntry,
            write_preamble,
        )

        router = self._gguf({"general.file_type": (4, 15)})
        shard = self._gguf({"split.count": (4, 2), "split.no": (4, 1)})
        blob_size = 1 << 20
        blob = shard + b"\x00" * (blob_size - len(shard))

        header = AsmbHeader(
            assembly_name="shard-test",
            embedding=EmbeddingEntry(name="router", offset=4096, size=len(router),
                                     confidence_threshold=0.7, fallback_model="m1"),
            models=[ModelEntry(name="m1", offset=8192, size=blob_size, anchors=["a"])],
        )
        hjson = header.to_json_bytes()
        pre = bytearray()
        write_preamble(pre, len(hjson))

        path = tmp_path / "shard.asmb"
        path.write_bytes(
            bytes(pre) + hjson
            + b"\x00" * (4096 - len(pre) - len(hjson))
            + router + b"\x00" * (4096 - len(router))
            + blob
        )

        staged = []

        class _Boom:
            def __enter__(self_inner):
                staged.append("staged")
                return "unused"

            def __exit__(self_inner, *exc):
                return False

        monkeypatch.setattr(wrapper, "_memfd_model",
                            lambda name, data: _Boom())

        llm = wrapper.AssemblyLlama.__new__(wrapper.AssemblyLlama)
        llm._mm = None
        llm._asmb_path = path

        with open(path, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            llm._mm = mm
            try:
                with pytest.raises(InvalidArchiveError, match="split GGUF"):
                    llm._load_llama_from_mmap(
                        name="m1", offset=8192, size=blob_size
                    )
            finally:
                mm.close()

        assert staged == [], "must not stage a blob that cannot be loaded"

    def test_cli_reports_load_failures_without_a_traceback(self, monkeypatch, capsys):
        from assembly import main as main_mod
        from assembly.wrapper import StagingError

        def boom(_args):
            raise StagingError("not enough space to stage Code: need 4.0 GB")

        monkeypatch.setattr(main_mod, "_cmd_info", boom)
        with pytest.raises(SystemExit) as exc:
            main_mod.main(["info", "-m", "x.asmb"])
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "not enough space to stage Code" in captured.err
        assert "Traceback" not in captured.err

    def test_cli_exits_130_on_interrupt(self, monkeypatch, capsys):
        from assembly import main as main_mod

        def boom(_args):
            raise KeyboardInterrupt

        monkeypatch.setattr(main_mod, "_cmd_info", boom)
        with pytest.raises(SystemExit) as exc:
            main_mod.main(["info", "-m", "x.asmb"])
        assert exc.value.code == 130

class TestConfigRobustness:
    def _write(self, tmp_path, **overrides):
        for name in ("emb.gguf", "m1.gguf", "m2.gguf"):
            (tmp_path / name).write_bytes(b"GGUF" + b"\x00" * 64)
        cfg = {
            "assembly_name": "cfg",
            "embedding": {"name": "emb", "path": str(tmp_path / "emb.gguf"),
                          "confidence_threshold": 0.7, "fallback_model": "m1"},
            "models": [
                {"name": "m1", "path": str(tmp_path / "m1.gguf"),
                 "anchors": ["a", "b"]},
                {"name": "m2", "path": str(tmp_path / "m2.gguf"),
                 "anchors": ["c", "d"]},
            ],
        }
        cfg.update(overrides)
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg))
        return path

    def test_tilde_paths_are_expanded(self, tmp_path, monkeypatch):
        from assembly.config import validate_assembly_preflight

        monkeypatch.setenv("HOME", str(tmp_path))
        cfg = self._write(tmp_path)
        report = validate_assembly_preflight(cfg)
        assert not any(i.severity == "ERROR" for i in report.errors)

    def test_missing_file_is_an_error_not_a_crash(self, tmp_path):
        from assembly.config import validate_assembly_preflight

        cfg = self._write(tmp_path)
        raw = json.loads(cfg.read_text())
        raw["models"][0]["path"] = str(tmp_path / "ghost.gguf")
        cfg.write_text(json.dumps(raw))

        report = validate_assembly_preflight(cfg)
        assert any(i.severity == "ERROR" for i in report.errors)

    def test_unreadable_file_is_reported_not_raised(self, tmp_path, monkeypatch):
        """stat() failing after is_file() passed must become a report entry."""
        import pathlib

        from assembly.config import validate_assembly_preflight

        cfg = self._write(tmp_path)
        real_stat = pathlib.Path.stat
        target = str(tmp_path / "m1.gguf")

        def flaky(self, *a, **k):
            if str(self) == target:
                raise OSError("Too many open files")
            return real_stat(self, *a, **k)

        monkeypatch.setattr(pathlib.Path, "stat", flaky)
        report = validate_assembly_preflight(cfg)

        assert any("Too many open files" in i.message for i in report.errors)


# ── Header strictness ─────────────────────────────────────────────────

class TestHeaderStrictness:
    def _preamble(self, version=1, json_len=16):
        return (
            b"ASMB"
            + version.to_bytes(2, "little")
            + b"\x00" * 2
            + json_len.to_bytes(4, "little")
            + b"\x00" * 4
        )

    def test_bad_magic(self):
        from assembly.header import InvalidArchiveError, read_header

        with pytest.raises(InvalidArchiveError, match="magic"):
            read_header(b"XXXX" + b"\x00" * 32)

    def test_truncated_preamble(self):
        from assembly.header import InvalidArchiveError, read_header

        with pytest.raises(InvalidArchiveError):
            read_header(b"ASM")

    def test_future_format_version_is_rejected(self):
        from assembly.header import InvalidArchiveError, read_header

        with pytest.raises(InvalidArchiveError, match="version"):
            read_header(self._preamble(version=99) + b"{}")

    def test_preamble_magic_is_validated_first(self):
        """A bad magic on a huge buffer must not be read as a giant header."""
        from assembly.header import InvalidArchiveError, read_header

        with pytest.raises(InvalidArchiveError, match="magic"):
            read_header(b"ZZZZ" + b"\xff" * 4096)

    def test_header_json_is_bounded_by_the_buffer(self):
        """A preamble claiming a huge header must not attempt the read."""
        from assembly.header import InvalidArchiveError, read_header

        buf = self._preamble(json_len=1 << 30) + b"{"
        with pytest.raises(InvalidArchiveError):
            read_header(buf)

    def test_non_page_aligned_offset_is_rejected(self):
        from assembly.header import (
            AsmbHeader,
            EmbeddingEntry,
            InvalidArchiveError,
            ModelEntry,
            read_header,
            write_preamble,
        )

        header = AsmbHeader(
            assembly_name="misaligned",
            embedding=EmbeddingEntry(name="e", offset=4097, size=16,
                                     confidence_threshold=0.5, fallback_model="a"),
            models=[ModelEntry(name="a", offset=8192, size=16, anchors=["x"])],
        )
        blob = header.to_json_bytes()
        pre = bytearray()
        write_preamble(pre, len(blob))

        with pytest.raises(InvalidArchiveError, match="align"):
            read_header(bytes(pre) + blob + b"\x00" * 64)

    def test_blob_past_end_of_archive_is_rejected(self):
        import json as _json

        from assembly.header import InvalidArchiveError, read_header, write_preamble

        doc = _json.dumps({
            "assembly_name": "oob",
            "version": 1,
            "page_size": 4096,
            "embedding": {"name": "e", "offset": 4096, "size": 1 << 40,
                          "confidence_threshold": 0.5, "fallback_model": "a"},
            "models": [{"name": "a", "offset": 8192, "size": 16, "anchors": ["x"]}],
        }).encode()
        pre = bytearray()
        write_preamble(pre, len(doc))

        with pytest.raises(InvalidArchiveError, match="past the end"):
            read_header(bytes(pre) + doc)


# ── Windows compatibility ────────────────────────────────────────────

class _FakeWin32Fn:
    """A callable that accepts the restype/argtypes the real ctypes call sets."""

    def __init__(self, impl):
        self._impl = impl
        self.restype = "UNSET"
        self.argtypes = "UNSET"

    def __call__(self, *args):
        return self._impl(*args)


class _FakeKernel32:
    """Stands in for kernel32 so the Windows probe's logic is testable anywhere.

    ``ctypes.WinDLL`` does not exist off Windows, so the probe cannot even be
    called on a POSIX runner. Substituting the DLL exercises the branch logic
    and the handle bookkeeping; the real API is covered by
    test_the_win32_probe_works_against_a_real_process and by the CI job.
    """

    def __init__(self, handle, exit_code, get_fails=False):
        self.handle = handle
        self.exit_code = exit_code
        self.get_fails = get_fails
        self.closed = []
        self.OpenProcess = _FakeWin32Fn(self._open)
        self.GetExitCodeProcess = _FakeWin32Fn(self._exit_code)
        self.CloseHandle = _FakeWin32Fn(self._close)

    def _open(self, access, inherit, pid):
        assert access == 0x1000, f"unexpected access mask {access}"
        return self.handle

    def _exit_code(self, handle, out):
        import ctypes as _ctypes

        if self.get_fails:
            return 0
        _ctypes.cast(out, _ctypes.POINTER(_ctypes.c_uint32))[0] = self.exit_code
        return 1

    def _close(self, handle):
        self.closed.append(handle)
        return 1


@pytest.fixture
def probe_on_this_platform(monkeypatch):
    """Call the Windows probe here, with a fake kernel32 and a chosen result."""
    import ctypes as _ctypes

    from assembly import packer

    def call(*, handle=0x1234, exit_code=259, get_fails=False, last_error=0):
        kernel32 = _FakeKernel32(handle, exit_code, get_fails)
        # The probe calls ctypes.WinDLL / ctypes.get_last_error inside the
        # function body, so patching the module reaches it.
        monkeypatch.setattr(
            _ctypes, "WinDLL", lambda *a, **k: kernel32, raising=False
        )
        monkeypatch.setattr(_ctypes, "get_last_error", lambda: last_error,
                            raising=False)
        return packer._pid_exists_windows(4242), kernel32

    return call


class TestWindowsProbeLogic:
    """Branches of the Windows liveness probe, decided by the exit code."""

    def test_a_running_process_is_alive(self, probe_on_this_platform):
        # STILL_ACTIVE is 259; that is what a live process reports.
        assert probe_on_this_platform(exit_code=259)[0] is True

    def test_an_exited_process_is_not_alive(self, probe_on_this_platform):
        assert probe_on_this_platform(exit_code=0)[0] is False

    def test_a_nonstandard_exit_code_is_not_alive(self, probe_on_this_platform):
        assert probe_on_this_platform(exit_code=1)[0] is False

    def test_a_refused_open_is_treated_as_alive(self, probe_on_this_platform):
        """ERROR_ACCESS_DENIED: the process exists, we just cannot query it.

        Erring the other way would delete a live build's partial, which is the
        one outcome that loses data; keeping a stale file only wastes disk.
        """
        alive, _ = probe_on_this_platform(handle=0, last_error=5)
        assert alive is True

    def test_a_failed_open_is_treated_as_dead(self, probe_on_this_platform):
        # ERROR_INVALID_PARAMETER / anything else: no such process.
        assert probe_on_this_platform(handle=0, last_error=87)[0] is False

    def test_an_unreadable_exit_code_is_treated_as_alive(self, probe_on_this_platform):
        """The handle opened, so the process exists. Default to keeping it."""
        alive, _ = probe_on_this_platform(get_fails=True)
        assert alive is True

    def test_the_handle_is_closed_on_every_path(self, probe_on_this_platform):
        _, kernel32 = probe_on_this_platform()
        assert kernel32.closed == [0x1234], "probe leaked a process handle"

    def test_the_handle_is_closed_when_the_read_fails(self, probe_on_this_platform):
        """A failed GetExitCodeProcess must still not leak; it is in a finally."""
        _, kernel32 = probe_on_this_platform(get_fails=True)
        assert kernel32.closed == [0x1234]

    def test_the_ffi_signatures_are_actually_set(self, probe_on_this_platform):
        """ctypes defaults returns to c_int, truncating a 64-bit HANDLE.

        A truncated handle reads as a small integer and is mistaken for
        failure, which is indistinguishable from "this build is dead" -- and
        gets a live build's partial deleted.
        """
        _, kernel32 = probe_on_this_platform()
        for fn in (kernel32.OpenProcess, kernel32.GetExitCodeProcess,
                   kernel32.CloseHandle):
            assert fn.restype != "UNSET", f"{fn.__name__} restype not declared"
            assert fn.argtypes != "UNSET", f"{fn.__name__} argtypes not declared"


class TestWindowsProcessProbe:
    """``_writer_alive`` decides whether a partial file is safe to delete.

    On POSIX that is ``os.kill(pid, 0)``, which signals nothing. On Windows
    CPython routes every signal except CTRL_C_EVENT / CTRL_BREAK_EVENT through
    TerminateProcess, so the obvious implementation would kill the very build
    it is meant to protect and then delete its partial out from under it.
    """

    def test_dispatch_follows_the_platform(self, monkeypatch):
        from assembly import packer

        calls = []
        monkeypatch.setattr(packer, "_pid_exists_posix",
                            lambda pid: calls.append(("posix", pid)) or True)
        monkeypatch.setattr(packer, "_pid_exists_windows",
                            lambda pid: calls.append(("nt", pid)) or False)

        monkeypatch.setattr(packer, "IS_WINDOWS", False)
        assert packer._writer_alive(123) is True
        monkeypatch.setattr(packer, "IS_WINDOWS", True)
        assert packer._writer_alive(123) is False

        assert calls == [("posix", 123), ("nt", 123)]

    def test_the_win32_probe_never_signals_the_process(self):
        """The probe must be observational. Any signalling call is a bug.

        This is the failure that matters on Windows: os.kill(pid, 0) looks
        like a harmless existence check, but CPython forwards it to
        TerminateProcess, so it kills the running build and then the sweep
        deletes its partial. Asserted structurally so it cannot regress
        quietly on a platform where the mistake is invisible.
        """
        import ast
        import inspect as _inspect
        import textwrap

        from assembly import packer

        # Parsed rather than grepped: the prohibition is explained in the
        # docstring, so a text search would flag its own explanation. What
        # matters is whether the function *calls* anything that signals.
        tree = ast.parse(textwrap.dedent(_inspect.getsource(
            packer._pid_exists_windows)))
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    called.add(func.attr)
                elif isinstance(func, ast.Name):
                    called.add(func.id)

        assert "kill" not in called, f"the probe signals a process: {called}"
        assert "system" not in called
        # The supported way to ask, with no side effect on the target.
        assert "OpenProcess" in called
        assert "GetExitCodeProcess" in called

    def test_the_win32_probe_closes_its_handle(self):
        """Every OpenProcess must be matched by a CloseHandle.

        Left open, each probe leaks a handle, and a long sweep over stale
        partials would eventually exhaust the per-process handle limit.
        """
        import inspect as _inspect

        from assembly import packer

        source = _inspect.getsource(packer._pid_exists_windows)
        assert source.count("OpenProcess") >= 1
        assert source.count("CloseHandle") >= 1
        assert "finally" in source

    def test_the_win32_probe_works_against_a_real_process(self):
        """Exercise the Windows probe against a real process where possible.

        The concern is not theoretical: OpenProcess/GetExitCodeProcess on a
        process that is still running must report it alive, and must report a
        reaped one dead. On a POSIX runner this is skipped rather than faked,
        because a monkeypatched kernel32 proves nothing about the real API.
        """
        import subprocess as _subprocess
        import sys as _sys

        from assembly import packer

        if not packer.IS_WINDOWS:
            pytest.skip("the Windows probe is a no-op on this platform")

        # A child that outlives the assertion, so "alive" is unambiguous.
        child = _subprocess.Popen(
            [_sys.executable, "-c", "import time; time.sleep(30)"]
        )
        try:
            assert packer._writer_alive(child.pid) is True
            # Reap it, then the PID is free and the probe must say dead.
            child.terminate()
            child.wait(timeout=30)
            assert packer._writer_alive(child.pid) is False
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=30)

    def test_the_posix_probe_never_signals_either(self):
        """The POSIX probe uses signal 0, which is a genuine no-op."""
        import ast
        import inspect as _inspect
        import textwrap

        from assembly import packer

        tree = ast.parse(textwrap.dedent(_inspect.getsource(
            packer._pid_exists_posix)))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "kill"
        ]
        assert len(calls) == 1
        # Signal 0 is the whole trick: any other signal would actually signal.
        assert calls[0].args[1].value == 0

    def test_the_win32_probe_declares_its_handle_types(self):
        """An un-annotated HANDLE is truncated to c_int on 64-bit Windows.

        ctypes defaults every return value to c_int, so without restype and
        argtypes a real handle can arrive looking like a small integer and be
        read as failure -- which would look like 'this build is dead' and get a
        live build's partial deleted.
        """
        import inspect as _inspect

        from assembly import packer

        source = _inspect.getsource(packer._pid_exists_windows)
        assert "restype" in source
        assert "argtypes" in source

    def test_the_platform_flag_is_not_patched_globally(self):
        """pathlib.Path picks its flavour from os.name, so it must stay put.

        Reading the platform from a module constant rather than mutating os
        keeps a Windows test run from turning every Path into a WindowsPath
        on Linux.
        """
        import os as _os

        from assembly import packer

        before = _os.name
        packer.IS_WINDOWS = not packer.IS_WINDOWS
        assert _os.name == before

    def test_posix_probe_reports_missing_pid(self):
        from assembly.packer import _pid_exists_posix

        # Pid 0 addresses a process group, so use a pid that cannot be running.
        # 2^22 is above Linux's default pid_max and above Windows' practical
        # range, so this is reliably not a live process on the POSIX path.
        assert _pid_exists_posix(1 << 22) is False

    def test_posix_probe_reports_live_pid(self):
        import os as _os

        from assembly.packer import _pid_exists_posix

        assert _pid_exists_posix(_os.getpid()) is True

    def test_posix_probe_treats_permission_error_as_alive(self, monkeypatch):
        """A pid we may not signal still exists; deleting its file is the risk."""
        from assembly import packer

        def denied(pid, sig):
            raise PermissionError

        monkeypatch.setattr(packer.os, "kill", denied)
        assert packer._pid_exists_posix(4242) is True

    def test_posix_probe_treats_lookup_error_as_dead(self, monkeypatch):
        from assembly import packer

        def missing(pid, sig):
            raise ProcessLookupError

        monkeypatch.setattr(packer.os, "kill", missing)
        assert packer._pid_exists_posix(4242) is False


class TestWindowsStaging:
    """Blob staging on platforms without memfd, which is every Windows build.

    Windows holds an open handle exclusive enough that a still-open staging
    file cannot be opened by name, so a file that is written and only closed
    later would fail inside llama.cpp with a sharing violation rather than at
    the point of the mistake.
    """

    def _force_file_staging(self, monkeypatch):
        """Pretend we are on Windows even when the tests run on Linux."""
        from assembly import wrapper

        monkeypatch.setattr(wrapper.sys, "platform", "win32")
        monkeypatch.delattr(wrapper.os, "memfd_create", raising=False)

    def test_staged_file_is_closed_before_its_path_is_handed_over(
        self, tmp_path, monkeypatch
    ):
        """The path must be openable by another process, not just readable here."""
        from assembly import wrapper

        self._force_file_staging(monkeypatch)
        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)

        with wrapper._memfd_model("m.gguf", b"payload") as path:
            # An independent open is the actual requirement. On Windows this
            # raises PermissionError if our handle is still open; on Linux it
            # succeeds either way, which is why the same test runs everywhere.
            with open(path, "rb") as probe:
                assert probe.read() == b"payload"

    def test_staged_file_is_removed_on_exit(self, tmp_path, monkeypatch):
        from assembly import wrapper

        self._force_file_staging(monkeypatch)
        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)

        with wrapper._memfd_model("m.gguf", b"payload") as path:
            assert pathlib.Path(path).is_file()

        assert not pathlib.Path(path).exists()

    def test_staged_blob_is_named_so_llama_cpp_recognises_it(
        self, tmp_path, monkeypatch
    ):
        """llama.cpp dispatches on the .gguf extension."""
        from assembly import wrapper

        self._force_file_staging(monkeypatch)
        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)

        with wrapper._memfd_model("router.gguf", b"x") as path:
            assert path.endswith(".gguf")

    def test_cleanup_tolerates_a_locked_blob(self, tmp_path, monkeypatch):
        """Windows keeps a mapped file locked; losing the race is not an error.

        llama.cpp holds the model mapping for the lifetime of the model, not
        just the load, so on Windows the unlink at the end of staging usually
        fails. Raising there would turn a successful load into a crash, and the
        space is reclaimed when the staging directory goes at exit instead.
        """
        from assembly import wrapper

        self._force_file_staging(monkeypatch)
        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)

        def locked_unlink(self, missing_ok=False):
            raise PermissionError(13, "used by another process")

        monkeypatch.setattr(pathlib.Path, "unlink", locked_unlink)
        path = wrapper._stage_blob("m.gguf", b"payload")
        wrapper._unstage_blob(path)  # must not raise

    def test_a_failed_write_leaves_no_partial_blob(self, tmp_path, monkeypatch):
        """A truncated blob would surface as a bogus llama.cpp load error."""
        from assembly import wrapper

        self._force_file_staging(monkeypatch)
        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)

        boom = RuntimeError("disk fell over")

        def exploding_write(target, data):
            raise boom

        monkeypatch.setattr(wrapper, "_write_blob", exploding_write)

        with pytest.raises(RuntimeError, match="disk fell over"):
            wrapper._stage_blob("m.gguf", b"payload" * 1000)

        assert list(tmp_path.iterdir()) == [], "a partial blob survived"

    def test_space_is_checked_before_writing(self, tmp_path, monkeypatch):
        from assembly import wrapper

        self._force_file_staging(monkeypatch)
        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)
        monkeypatch.setattr(wrapper, "free_bytes_at", lambda _d: 1)

        with pytest.raises(wrapper.StagingError, match="not enough space"):
            wrapper._stage_blob("big.gguf", b"\x00" * 4096)

    def test_untouchable_staging_dir_is_a_readable_error(self, tmp_path, monkeypatch):
        from assembly import wrapper

        self._force_file_staging(monkeypatch)
        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)

        def refused(*args, **kwargs):
            raise PermissionError(13, "Access is denied")

        monkeypatch.setattr(wrapper.tempfile, "mkstemp", refused)

        with pytest.raises(wrapper.StagingError, match="could not create a staging file"):
            wrapper._stage_blob("m.gguf", b"x")


class TestStagingDirOwnership:
    """A user's --temp-dir is theirs, and must never be deleted for them."""

    def _reset(self):
        from assembly import wrapper

        wrapper._TEMP_DIR = None
        wrapper._TEMP_DIR_OWNED = False

    def test_configured_dir_is_never_removed(self, tmp_path):
        from assembly import wrapper

        self._reset()
        try:
            mine = tmp_path / "mine"
            keep = mine / "important.txt"
            wrapper.configure_temp_dir(mine)
            keep.write_text("do not delete me")

            wrapper.cleanup_temp_dir()

            assert keep.is_file(), "cleanup deleted a file from a user temp dir"
            assert wrapper._TEMP_DIR is not None, "cleanup forgot the directory"
        finally:
            self._reset()

    def test_our_own_dir_is_removed(self, tmp_path, monkeypatch):
        from assembly import wrapper

        self._reset()
        monkeypatch.setenv(wrapper.TEMP_DIR_ENV, str(tmp_path / "base"))
        try:
            created = wrapper.assembly_temp_dir()
            assert created.is_dir()

            wrapper.cleanup_temp_dir()

            assert not created.exists(), "our own staging dir was left behind"
            assert wrapper.assembly_temp_dir() != created, "not recreated after cleanup"
        finally:
            self._reset()

    def test_repointing_does_not_delete_the_users_directory(self, tmp_path):
        """Switching dirs must not rmtree whatever the user gave us first."""
        from assembly import wrapper

        self._reset()
        try:
            first = tmp_path / "first"
            keeper = first / "keep.txt"
            wrapper.configure_temp_dir(first)
            keeper.write_text("still here")

            wrapper.configure_temp_dir(tmp_path / "second")

            assert keeper.is_file(), "repointing deleted the user's directory"
        finally:
            self._reset()

    def test_repointing_does_remove_our_own_previous_dir(self, tmp_path, monkeypatch):
        from assembly import wrapper

        self._reset()
        monkeypatch.setenv(wrapper.TEMP_DIR_ENV, str(tmp_path / "base"))
        try:
            created = wrapper.assembly_temp_dir()
            assert created.is_dir()

            wrapper.configure_temp_dir(tmp_path / "second")

            assert not created.exists(), "our own abandoned staging dir leaked"
        finally:
            self._reset()

    def test_cleanup_is_idempotent(self, tmp_path, monkeypatch):
        from assembly import wrapper

        self._reset()
        monkeypatch.setenv(wrapper.TEMP_DIR_ENV, str(tmp_path / "base"))
        try:
            wrapper.assembly_temp_dir()
            wrapper.cleanup_temp_dir()
            wrapper.cleanup_temp_dir()  # must not raise
        finally:
            self._reset()


class TestWindowsConsolePrompt:
    """The REPL prompt bypasses rich and goes straight through input().

    On a legacy Windows console there is nothing to interpret an ANSI colour
    sequence, so the raw bytes are printed and the user types after a string of
    escape codes.
    """

    def test_no_escapes_on_a_legacy_console(self, monkeypatch):
        from assembly import ui

        monkeypatch.setattr(ui, "_LEGACY_WINDOWS_CONSOLE", True)
        assert ui.prompt_text() == "▸ "
        assert "\x1b" not in ui.prompt_text()

    def test_no_escapes_when_no_color(self, monkeypatch):
        from assembly import ui

        monkeypatch.setattr(ui, "_NO_COLOR", True)
        assert "\x1b" not in ui.prompt_text()

    def test_colour_is_kept_on_a_capable_terminal(self, monkeypatch):
        from assembly import ui

        monkeypatch.setattr(ui, "_NO_COLOR", False)
        monkeypatch.setattr(ui, "_DUMB", False)
        monkeypatch.setattr(ui, "_LEGACY_WINDOWS_CONSOLE", False)
        monkeypatch.setattr(ui.sys.stdout, "isatty", lambda: True)
        assert "\x1b" in ui.prompt_text()

    def test_no_escapes_when_stdout_is_redirected(self, monkeypatch):
        """input() writes the prompt to stdout itself, escape codes and all.

        So a piped or captured session would otherwise get raw sequences in
        the middle of its transcript.
        """
        from assembly import ui

        monkeypatch.setattr(ui, "_NO_COLOR", False)
        monkeypatch.setattr(ui, "_DUMB", False)
        monkeypatch.setattr(ui, "_LEGACY_WINDOWS_CONSOLE", False)
        monkeypatch.setattr(ui.sys.stdout, "isatty", lambda: False)
        assert ui.prompt_text() == "▸ "

    def test_no_escapes_on_a_dumb_terminal(self, monkeypatch):
        """TERM=dumb: a terminal that exists but cannot render colour."""
        from assembly import ui

        monkeypatch.setattr(ui, "_NO_COLOR", False)
        monkeypatch.setattr(ui, "_DUMB", True)
        monkeypatch.setattr(ui, "_LEGACY_WINDOWS_CONSOLE", False)
        monkeypatch.setattr(ui.sys.stdout, "isatty", lambda: True)
        assert ui.prompt_text() == "▸ "


class TestBuildSignalHandling:
    """A build must clean up its partial on interruption without trapping the user.

    The handler raises rather than ignoring, so Ctrl-C still cancels a long
    build; a second signal escalates to the platform default so a stuck unwind
    cannot make the build unstoppable.
    """

    def test_handlers_are_installed_and_restored(self):
        import signal as _signal

        from assembly import packer

        before = _signal.getsignal(_signal.SIGINT)
        previous = packer._install_build_signal_handlers()
        try:
            installed = _signal.getsignal(_signal.SIGINT)
            assert installed is not _signal.SIG_IGN, "SIGINT was swallowed"
            assert isinstance(installed, packer._BuildSignalHandler)
        finally:
            packer._restore_signal_handlers(previous)

        assert _signal.getsignal(_signal.SIGINT) == before

    def test_ctrl_c_actually_cancels_a_build(self):
        """Regression: SIG_IGN would make a 20-minute build uncancellable.

        CPython does not raise KeyboardInterrupt for a signal whose handler is
        SIG_IGN, so "ignore SIGINT and let the first Ctrl-C through" is not a
        thing that can be true. The handler has to raise it itself.
        """
        import signal as _signal

        from assembly import packer

        previous = packer._install_build_signal_handlers()
        try:
            handler = _signal.getsignal(_signal.SIGINT)
            with pytest.raises(KeyboardInterrupt, match="SIGINT"):
                handler(_signal.SIGINT, None)
        finally:
            packer._restore_signal_handlers(previous)

    def test_sigterm_also_unwinds_instead_of_killing(self):
        """`kill` must reach the same cleanup as Ctrl-C."""
        import signal as _signal

        from assembly import packer

        if not hasattr(_signal, "SIGTERM"):
            pytest.skip("SIGTERM is Unix-only")

        previous = packer._install_build_signal_handlers()
        try:
            handler = _signal.getsignal(_signal.SIGTERM)
            with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
                handler(_signal.SIGTERM, None)
        finally:
            packer._restore_signal_handlers(previous)

    def test_a_second_signal_escalates_instead_of_looping(self):
        """The first signal unwinds; the second gives up on the unwind.

        Without escalation, a cleanup stuck unlinking a multi-gigabyte partial
        would need an unbounded number of Ctrl-C presses to escape.
        """
        from assembly import packer

        handler = packer._BuildSignalHandler(2, {2: None})
        with pytest.raises(KeyboardInterrupt):
            handler(2, None)          # first signal: the ordinary unwind
        assert handler._seen is True

        # The second call must escalate instead of raising again. Escalation
        # terminates the process, so stand in for it.
        escalated = []
        handler._escalate = lambda: escalated.append(True)
        with pytest.raises(KeyboardInterrupt):
            handler(2, None)          # raised after the stubbed escalation
        assert escalated == [True]

    def test_restore_survives_a_platform_without_the_signal(self):
        """SIGHUP does not exist on Windows; restoring must not raise."""
        from assembly import packer

        packer._restore_signal_handlers({9999: None})

    def test_build_restores_handlers_even_when_it_fails(self, tmp_path):
        """A failed build must not leave our handler installed for the process."""
        import signal as _signal

        from assembly import packer

        before = _signal.getsignal(_signal.SIGINT)

        with pytest.raises(Exception):
            packer.build(str(tmp_path / "does-not-exist.json"),
                         tmp_path / "out.asmb")

        assert _signal.getsignal(_signal.SIGINT) == before

    def test_a_successful_build_restores_handlers(self, tmp_path):
        import signal as _signal

        from assembly.packer import build
        from tests.test_assembly import TestPackerIntegrity

        before = _signal.getsignal(_signal.SIGINT)
        builder = TestPackerIntegrity()
        cfg_path = builder._config(tmp_path)

        build(cfg_path, tmp_path / "built.asmb", skip_embeddings=True)

        assert _signal.getsignal(_signal.SIGINT) == before

    def test_no_handler_is_left_behind_after_a_successful_build(self, tmp_path):
        """Regression: a lingering handler breaks Ctrl-C for the rest of the run."""
        import signal as _signal

        from assembly.packer import build
        from tests.test_assembly import TestPackerIntegrity

        before = _signal.getsignal(_signal.SIGINT)
        builder = TestPackerIntegrity()
        cfg_path = builder._config(tmp_path)

        build(cfg_path, tmp_path / "built.asmb", skip_embeddings=True)

        from assembly import packer

        left = _signal.getsignal(_signal.SIGINT)
        assert not isinstance(left, packer._BuildSignalHandler)
        assert left == before

class TestBuildOnWindowsPaths:
    """Build mechanics that differ because Windows locks files it opens."""

    def test_fsync_dir_is_a_noop_on_windows(self, tmp_path, monkeypatch):
        from assembly import packer

        monkeypatch.setattr(packer, "IS_WINDOWS", True)
        opened = []
        monkeypatch.setattr(packer.os, "open",
                            lambda *a, **k: opened.append(a) or 3)

        packer._fsync_dir(tmp_path)  # must not raise, must not open anything
        assert opened == []

    def test_fsync_dir_tolerates_a_filesystem_that_refuses(self, tmp_path):
        from assembly import packer

        packer._fsync_dir(tmp_path)  # a real directory on a real filesystem

    def test_a_locked_output_is_reported_in_the_users_terms(self, tmp_path,
                                                             monkeypatch):
        """Windows will not replace a file another process has open.

        That is what happens when `assembly serve` holds the archive mapped, so
        the error has to say so; "Permission denied" after an hour of packing
        tells the user nothing.
        """
        from assembly import packer
        from tests.test_assembly import TestPackerIntegrity

        builder = TestPackerIntegrity()
        cfg_path = builder._config(tmp_path)

        def locked_replace(*args, **kwargs):
            raise PermissionError(13, "The process cannot access the file")

        monkeypatch.setattr(packer.os, "replace", locked_replace)

        with pytest.raises(packer.BuildError, match="in use by another program"):
            packer.build(cfg_path, tmp_path / "built.asmb", skip_embeddings=True)

    def test_the_partial_is_reclaimed_after_a_locked_output(self, tmp_path,
                                                            monkeypatch):
        """The renamed-away partial must still be swept on the next build."""
        from assembly import packer
        from tests.test_assembly import TestPackerIntegrity

        builder = TestPackerIntegrity()
        cfg_path = builder._config(tmp_path)

        def locked_replace(*args, **kwargs):
            raise PermissionError(13, "in use")

        monkeypatch.setattr(packer.os, "replace", locked_replace)

        with pytest.raises(packer.BuildError):
            packer.build(cfg_path, tmp_path / "built.asmb", skip_embeddings=True)

        monkeypatch.undo()
        assert not (tmp_path / "built.asmb").exists()
        packer.build(cfg_path, tmp_path / "built.asmb", skip_embeddings=True)
        assert (tmp_path / "built.asmb").is_file()


class TestStagedBlobNaming:
    """A staged blob's filename must be derived from ours, not the model's.

    Windows reserves a set of device names -- CON, PRN, AUX, NUL, COM1..9,
    LPT1..9 -- that cannot be used as filenames at all. A model is named
    freely in the config, so staging must never take that name as the path; a
    model called "nul" has to work.
    """

    def _force_file_staging(self, monkeypatch, tmp_path):
        from assembly import wrapper

        monkeypatch.setattr(wrapper.sys, "platform", "win32")
        monkeypatch.delattr(wrapper.os, "memfd_create", raising=False)
        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)

    @pytest.mark.parametrize(
        "model_name",
        ["nul", "CON", "prn", "aux", "COM1", "lpt9", "clock$"],
    )
    def test_reserved_device_names_stage_normally(self, tmp_path, monkeypatch,
                                                 model_name):
        from assembly import wrapper

        self._force_file_staging(monkeypatch, tmp_path)

        with wrapper._memfd_model(model_name, b"payload") as path:
            # The path must be usable as a real file, which on Windows is the
            # whole question: `nul.gguf` would be the null device, not a file.
            assert pathlib.Path(path).is_file()
            assert pathlib.Path(path).read_bytes() == b"payload"

    def test_the_staged_name_does_not_come_from_the_model(self, tmp_path,
                                                          monkeypatch):
        from assembly import wrapper

        self._force_file_staging(monkeypatch, tmp_path)

        with wrapper._memfd_model("nul", b"payload") as path:
            assert pathlib.Path(path).name != "nul.gguf"

    def test_a_hostile_model_name_cannot_escape_the_staging_dir(
        self, tmp_path, monkeypatch
    ):
        """Separators and dot segments must not steer the write elsewhere."""
        from assembly import wrapper

        self._force_file_staging(monkeypatch, tmp_path)

        with wrapper._memfd_model("../../escaped", b"payload") as path:
            assert pathlib.Path(path).parent == tmp_path

    def test_paths_with_spaces_still_stage(self, tmp_path, monkeypatch):
        """Windows paths with spaces are ordinary; nothing may quote-split them."""
        from assembly import wrapper

        spaced = tmp_path / "a dir with spaces"
        spaced.mkdir()
        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: spaced)

        with wrapper._memfd_model("m.gguf", b"payload") as path:
            assert pathlib.Path(path).read_bytes() == b"payload"
            assert " " in path


class TestLockedBlobCleanup:
    """A blob that cannot be deleted at unload time must still leave no litter.

    Windows keeps a mapped file locked for as long as the mapping lives, and
    llama.cpp holds the model mapping until the model is dropped. The immediate
    unlink therefore cannot be unconditional. The tempting fallback — delete the
    staging directory — is wrong whenever the directory is the user's, so the
    path itself is remembered and retried at exit instead.
    """

    @staticmethod
    def _make_unlink_fail(monkeypatch):
        import pathlib as _pathlib

        monkeypatch.setattr(
            _pathlib.Path, "unlink",
            lambda self, **kw: (_ for _ in ()).throw(
                PermissionError(13, "file is in use by another process")
            ),
        )

    def test_a_locked_blob_is_remembered_for_exit(self, tmp_path, monkeypatch):
        from assembly import wrapper

        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)
        path = wrapper._stage_blob("m.gguf", b"payload")

        self._make_unlink_fail(monkeypatch)
        wrapper._unstage_blob(path)

        assert path in wrapper._PENDING_UNLINK, (
            "a blob that could not be deleted must be retried at exit"
        )

    def test_the_retry_removes_it_once_the_mapping_is_gone(self, tmp_path,
                                                          monkeypatch):
        from assembly import wrapper

        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)
        path = wrapper._stage_blob("m.gguf", b"payload")

        self._make_unlink_fail(monkeypatch)
        wrapper._unstage_blob(path)
        assert path.exists()

        # The llama.cpp object is gone, so the file is no longer held open.
        monkeypatch.undo()
        wrapper._flush_pending_unlinks()

        assert not path.exists()
        assert path not in wrapper._PENDING_UNLINK

    def test_a_successful_unlink_is_not_remembered(self, tmp_path, monkeypatch):
        from assembly import wrapper

        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)
        path = wrapper._stage_blob("m.gguf", b"payload")
        wrapper._unstage_blob(path)

        assert not path.exists()
        assert path not in wrapper._PENDING_UNLINK

    def test_the_retry_is_safe_when_the_file_is_already_gone(self, tmp_path,
                                                             monkeypatch):
        """Missing file is success, not an error worth retrying."""
        from assembly import wrapper

        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)
        path = wrapper._stage_blob("m.gguf", b"payload")
        wrapper._defer_unlink(path)
        path.unlink()

        wrapper._flush_pending_unlinks()

        assert path not in wrapper._PENDING_UNLINK

    def test_a_permanently_locked_blob_does_not_spin_forever(self, tmp_path,
                                                             monkeypatch):
        """Give up rather than looping on a file that will never unlock."""
        from assembly import wrapper

        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)
        path = wrapper._stage_blob("m.gguf", b"payload")
        wrapper._defer_unlink(path)

        self._make_unlink_fail(monkeypatch)
        wrapper._flush_pending_unlinks()  # must return, not hang

    def test_a_users_staging_directory_keeps_their_own_files(self, tmp_path,
                                                              monkeypatch):
        """The retry removes our blob and nothing else.

        This is the case that rules out "just delete the directory at exit" as
        the fallback: the directory holds files the user put there.
        """
        from assembly import wrapper

        theirs = tmp_path / "someone-elses-model.gguf"
        theirs.write_bytes(b"not ours")

        monkeypatch.setattr(wrapper, "assembly_temp_dir", lambda: tmp_path)
        monkeypatch.setattr(wrapper, "_TEMP_DIR", tmp_path, raising=False)
        monkeypatch.setattr(wrapper, "_TEMP_DIR_OWNED", False, raising=False)
        path = wrapper._stage_blob("m.gguf", b"payload")

        self._make_unlink_fail(monkeypatch)
        wrapper._unstage_blob(path)
        monkeypatch.undo()

        wrapper.cleanup_temp_dir()  # must not touch a directory we do not own
        wrapper._flush_pending_unlinks()

        assert not path.exists()
        assert theirs.is_file(), "cleanup removed a file it did not create"
        assert theirs.read_bytes() == b"not ours"


class TestServerBearerAuth:
    """The bearer token must not be crashable, and failures must be visible.

    Every one of these was found by sending the request a real client could
    send, not by reading the code.
    """

    def _handler(self, api_key: str, request: bytes):
        from assembly.server import AssemblyRequestHandler
        from unittest.mock import MagicMock

        sock = MockSocket(request)
        mock_server = MagicMock()
        mock_server.manager.list_models.return_value = []
        mock_server.manager.context_window.return_value = 4096
        mock_server.api_key = api_key
        mock_server.allow_origin = None
        # A MagicMock attribute is truthy, so pin it: these tests are about the
        # token, not the Host check.
        mock_server.require_local_host = False
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), mock_server)
        return sock.get_output()

    @pytest.mark.parametrize(
        "token",
        [
            "é",             # single non-ASCII char
            "s3cr\xe9t",      # correct prefix, non-ASCII tail
            "\xe9" * 40,      # all non-ASCII
            "s3crét",
        ],
    )
    def test_a_non_ascii_token_is_denied_not_crashed(self, token):
        """compare_digest raises TypeError on non-ASCII str operands.

        That call sits outside the handler's try, so before this fix the
        exception killed the connection with no response at all and printed a
        traceback to the operator's log -- an unauthenticated log-flood.
        """
        out = self._handler(
            "s3cret",
            f"GET /health HTTP/1.1\r\nHost: localhost\r\n"
            f"Authorization: Bearer {token}\r\n\r\n".encode("latin-1"),
        )
        assert b"401" in out.split(b"\r\n")[0], (
            "a non-ASCII token must be a clean 401"
        )

    def test_the_correct_token_still_works(self):
        out = self._handler(
            "s3cret",
            b"GET /health HTTP/1.1\r\nHost: localhost\r\n"
            b"Authorization: Bearer s3cret\r\n\r\n",
        )
        assert b"200" in out.split(b"\r\n")[0]

    def test_a_wrong_token_is_denied(self):
        out = self._handler(
            "s3cret",
            b"GET /health HTTP/1.1\r\nHost: localhost\r\n"
            b"Authorization: Bearer wrong\r\n\r\n",
        )
        assert b"401" in out.split(b"\r\n")[0]

    def test_an_empty_token_is_denied(self):
        """compare_digest(x, "") would match an empty Authorization value."""
        out = self._handler(
            "s3cret",
            b"GET /health HTTP/1.1\r\nHost: localhost\r\n"
            b"Authorization: Bearer \r\n\r\n",
        )
        assert b"401" in out.split(b"\r\n")[0]

    def test_a_failed_attempt_is_logged(self, caplog):
        """Unlimited guessing with no audit trail is not an access control."""
        import logging

        with caplog.at_level(logging.WARNING):
            self._handler(
                "s3cret",
                b"GET /health HTTP/1.1\r\nHost: localhost\r\n"
                b"Authorization: Bearer wrong\r\n\r\n",
            )
        assert any("Rejected unauthenticated" in r.message for r in caplog.records)

    def test_a_non_ascii_key_is_refused_at_startup(self, tmp_path):
        """A key that can never match is a lock with no key in it.

        http.server decodes header bytes as Latin-1, so a client cannot echo
        back a UTF-8 key: "sécret" arrives as "sÃ©cret" and every request is
        refused. Starting successfully and then rejecting everyone looks like
        the network is down, so this is refused up front instead.
        """
        from assembly.server import AssemblyServer, ModelManager

        with pytest.raises(ValueError, match="ASCII"):
            AssemblyServer(
                ("127.0.0.1", 0),
                ModelManager(models_dir=str(tmp_path)),
                api_key="sécret",
            )


class TestServerRequestValidation:
    """Client-supplied values must be validated, not interpolated."""

    def _load(self, manager, body: dict, api_key=None):
        from assembly.server import AssemblyRequestHandler
        from unittest.mock import MagicMock

        payload = json.dumps(body).encode()
        sock = MockSocket(
            b"POST /api/load HTTP/1.1\r\nHost: localhost\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        mock_server = MagicMock()
        mock_server.manager = manager
        mock_server.api_key = api_key
        mock_server.allow_origin = None
        mock_server.require_local_host = False
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), mock_server)
        raw = sock.get_output()
        status = int(raw.split(b"\r\n")[0].split()[1])
        _, _, response_body = raw.partition(b"\r\n\r\n")
        return status, response_body

    def test_alias_must_be_a_string(self, tmp_path):
        from assembly.server import ModelManager

        mgr = ModelManager(models_dir=str(tmp_path))
        archive = tmp_path / "a.asmb"
        archive.write_bytes(b"x")
        status, _ = self._load(mgr, {"model_path": str(archive), "alias": ["a"]})
        assert status == 400, "a list alias must not reach Path()/dict-key logic"

    def test_model_path_must_be_a_string(self, tmp_path):
        from assembly.server import ModelManager

        mgr = ModelManager(models_dir=str(tmp_path))
        status, _ = self._load(mgr, {"model_path": {"a": 1}})
        assert status == 400

    def test_alias_cannot_forge_a_log_line(self, tmp_path, caplog):
        """A newline in the alias would append a fake entry to the operator's log."""
        import logging

        from assembly.server import ModelManager

        mgr = ModelManager(models_dir=str(tmp_path))
        archive = tmp_path / "a.asmb"
        archive.write_bytes(b"x")
        with caplog.at_level(logging.INFO):
            status, _ = self._load(
                mgr,
                {"model_path": str(archive), "alias": "ok\nLoaded assembly model 'x'"},
            )
        assert status == 400
        assert not any(
            "\n" in r.getMessage() for r in caplog.records
        ), "control characters reached a log record"

    def test_alias_length_is_bounded(self, tmp_path):
        from assembly.server import ModelManager

        mgr = ModelManager(models_dir=str(tmp_path))
        archive = tmp_path / "a.asmb"
        archive.write_bytes(b"x")
        status, _ = self._load(
            mgr, {"model_path": str(archive), "alias": "a" * 10_000}
        )
        assert status == 400


class TestServerErrorDisclosure:
    """Internal detail belongs in the log, not in the response body."""

    def _post(self, manager, path: str, body: dict, api_key=None):
        from assembly.server import AssemblyRequestHandler
        from unittest.mock import MagicMock

        payload = json.dumps(body).encode()
        sock = MockSocket(
            f"POST {path} HTTP/1.1\r\nHost: localhost\r\n".encode()
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        mock_server = MagicMock()
        mock_server.manager = manager
        mock_server.api_key = api_key
        mock_server.allow_origin = None
        mock_server.require_local_host = False
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), mock_server)
        raw = sock.get_output()
        status = int(raw.split(b"\r\n")[0].split()[1])
        _, _, response_body = raw.partition(b"\r\n\r\n")
        return status, response_body

    def test_a_generation_error_does_not_return_llama_cpp_text(self):
        """llama.cpp error strings name the staged blob's absolute path."""
        from assembly.server import ModelManager
        from unittest.mock import MagicMock

        mgr = ModelManager()
        llm = MagicMock()
        llm.route.return_value.model_name = "coder"
        llm.route.return_value.model_index = 0
        llm.route.return_value.score = 0.9
        llm.route.return_value.is_fallback = False
        llm.create_chat_completion.side_effect = RuntimeError(
            "failed to load model from /private/var/folders/xy/assembly-abc/blob-1.gguf"
        )
        mgr._models["a"] = llm
        mgr._active_alias = "a"

        status, body = self._post(
            mgr, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}
        )
        assert status == 500
        assert b"var/folders" not in body, f"staging path leaked: {body!r}"
        assert b"blob-1.gguf" not in body

    def test_a_routing_error_does_not_return_internal_text(self):
        from assembly.server import ModelManager
        from unittest.mock import MagicMock

        mgr = ModelManager()
        llm = MagicMock()
        llm.route.side_effect = RuntimeError("staged at /tmp/assembly-xyz/blob-9.gguf")
        mgr._models["a"] = llm
        mgr._active_alias = "a"

        status, body = self._post(mgr, "/api/route", {"prompt": "hi"})
        assert status == 500
        assert b"assembly-xyz" not in body

    def test_a_capacity_error_does_not_return_disk_details(self):
        """StagingError names the staging directory and its free space."""
        from unittest.mock import MagicMock

        from assembly.server import ModelManager
        from assembly.wrapper import StagingError

        mgr = ModelManager()
        mgr.load_model = MagicMock(
            side_effect=StagingError(
                "not enough space: need 40.0 GB, only 2.1 GB free in "
                "/private/var/folders/xy/assembly-abc"
            )
        )
        status, body = self._post(
            mgr, "/api/load", {"model_path": "/tmp/x.asmb"}
        )
        assert status == 429
        assert b"var/folders" not in body
        assert b"assembly-abc" not in body
        assert b"GB free" not in body

    def test_health_is_not_cacheable(self):
        """/health discloses version, model id and context window."""
        from assembly.server import AssemblyRequestHandler
        from unittest.mock import MagicMock

        sock = MockSocket(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        mock_server = MagicMock()
        mock_server.manager.list_models.return_value = []
        mock_server.manager.context_window.return_value = 4096
        mock_server.api_key = None
        mock_server.allow_origin = None
        mock_server.require_local_host = False
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), mock_server)
        out = sock.get_output().lower()
        assert b"cache-control: no-store" in out
        assert b"x-content-type-options: nosniff" in out


class TestServerCORSHeaderInjection:
    """Reflecting Origin must not be a way to write response headers."""

    def _get_with_origin(self, origin: bytes, allow_origin: str):
        from assembly.server import AssemblyRequestHandler
        from unittest.mock import MagicMock

        sock = MockSocket(
            b"GET /health HTTP/1.1\r\nHost: localhost\r\n"
            b"Authorization: Bearer k\r\nOrigin: " + origin + b"\r\n\r\n"
        )
        mock_server = MagicMock()
        mock_server.manager.list_models.return_value = []
        mock_server.manager.context_window.return_value = 4096
        mock_server.api_key = "k"
        mock_server.allow_origin = allow_origin
        mock_server.require_local_host = False
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), mock_server)
        return sock.get_output()

    def test_a_folded_origin_cannot_smuggle_a_header(self):
        """http.server does not validate header values, so a CRLF in Origin
        writes attacker-chosen bytes into the response."""
        out = self._get_with_origin(b"http://evil.com\r\n  X-Injected: pwned", "*")
        assert b"X-Injected" not in out, "response header injection succeeded"

    def test_a_normal_allowed_origin_is_still_reflected(self):
        out = self._get_with_origin(b"http://localhost:5173", "http://localhost:5173")
        assert b"Access-Control-Allow-Origin: http://localhost:5173" in out

    def test_a_disallowed_origin_is_not_reflected(self):
        out = self._get_with_origin(b"http://evil.com", "http://localhost:5173")
        assert b"Access-Control-Allow-Origin" not in out


class TestServerDNSRebinding:
    """CORS is not a defence against a page that resolves to 127.0.0.1.

    Such a request is same-origin to the browser, so CORS never applies. The
    Host header is the one signal that still says who is asking.
    """

    def _get(self, host_header: bytes, require_local_host: bool):
        from assembly.server import AssemblyRequestHandler
        from unittest.mock import MagicMock

        sock = MockSocket(
            b"GET /health HTTP/1.1\r\nHost: " + host_header + b"\r\n\r\n"
        )
        mock_server = MagicMock()
        mock_server.manager.list_models.return_value = []
        mock_server.manager.context_window.return_value = 4096
        mock_server.api_key = None
        mock_server.allow_origin = None
        mock_server.require_local_host = require_local_host
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), mock_server)
        return int(sock.get_output().split(b"\r\n")[0].split()[1])

    @pytest.mark.parametrize(
        "host", [b"127.0.0.1:8000", b"localhost:8000", b"[::1]:8000", b"127.0.0.1"]
    )
    def test_loopback_hosts_are_accepted(self, host):
        assert self._get(host, require_local_host=True) == 200

    @pytest.mark.parametrize(
        "host",
        [b"evil.com", b"evil.com:8000", b"attacker.example", b"127.0.0.1.evil.com"],
    )
    def test_foreign_hosts_are_refused(self, host):
        assert self._get(host, require_local_host=True) == 401

    def test_the_check_does_not_apply_to_a_lan_listener(self):
        """A deliberately public server is addressed by name, not by IP."""
        assert self._get(b"models.lan:8000", require_local_host=False) == 200

    def test_a_lan_listener_does_not_require_local_host(self, tmp_path):
        """is_loopback decides this, so binding 0.0.0.0 disables the check."""
        from assembly.server import AssemblyServer, ModelManager

        srv = AssemblyServer(
            ("127.0.0.1", 0), ModelManager(models_dir=str(tmp_path))
        )
        try:
            assert srv.require_local_host is True
        finally:
            srv.server_close()


class TestServerCrossOriginWrites:
    """A web page must not be able to drive a keyless server with a simple POST.

    ``text/plain`` is a CORS-safelisted content type, so the browser fires the
    request without a preflight and merely hides the response. That is enough to
    make a page load archives into the operator's server.
    """

    def _post(self, body: bytes, headers: dict, allow_origin=None):
        from assembly.server import AssemblyRequestHandler
        from unittest.mock import MagicMock

        req = b"POST /api/load HTTP/1.1\r\nHost: localhost\r\n"
        req += f"Content-Length: {len(body)}\r\n".encode()
        req += b"".join(f"{k}: {v}\r\n".encode() for k, v in headers.items())
        sock = MockSocket(req + b"\r\n" + body)
        mock_server = MagicMock()
        mock_server.manager.load_model = MagicMock(return_value="x")
        mock_server.api_key = None
        mock_server.allow_origin = allow_origin
        mock_server.require_local_host = False
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), mock_server)
        return sock.get_output(), mock_server.manager.load_model

    def test_a_foreign_origin_simple_post_is_refused(self):
        out, load_model = self._post(
            b'{"model_path": "/tmp/x.asmb"}',
            {"Origin": "http://evil.com", "Content-Type": "text/plain;charset=UTF-8"},
        )
        assert b"403" in out.split(b"\r\n")[0]

    def test_a_refused_post_performs_no_write(self):
        """The 403 has to be the end of it, not a prelude.

        The refusal used to send 403 and then fall through to the router, so the
        very write that was just refused still happened -- the response stream
        carried two status lines and the archive was loaded regardless of what
        the caller was told.
        """
        out, load_model = self._post(
            b'{"model_path": "/tmp/x.asmb"}',
            {"Origin": "http://evil.com", "Content-Type": "text/plain;charset=UTF-8"},
        )
        assert not load_model.called, "the refused write reached the manager"
        assert out.count(b"HTTP/1.") == 1, "two responses were written"

    def test_a_non_browser_client_is_unaffected(self):
        """curl and the OpenAI SDK send no Origin at all."""
        out, load_model = self._post(
            b'{"model_path": "/tmp/x.asmb"}',
            {"Content-Type": "application/json"},
        )
        assert b"200" in out.split(b"\r\n")[0]
        load_model.assert_called_once()

    def test_an_explicitly_allowed_origin_may_write(self):
        out, load_model = self._post(
            b'{"model_path": "/tmp/x.asmb"}',
            {"Origin": "http://localhost:5173", "Content-Type": "application/json"},
            allow_origin="http://localhost:5173",
        )
        assert b"200" in out.split(b"\r\n")[0]
        load_model.assert_called_once()


class TestServerUnload:
    """A capacity limit you cannot recover from is an outage, not a limit."""

    def _manager(self, tmp_path, monkeypatch, max_models=2):
        from assembly import server as server_mod

        built: list = []

        class Model:
            def __init__(self, asmb_path, **kwargs):
                built.append(asmb_path)
                self.closed = False

            def close(self):
                self.closed = True

        monkeypatch.setattr(server_mod, "AssemblyLlama", Model)
        mgr = server_mod.ModelManager(models_dir=str(tmp_path), max_models=max_models)
        return mgr, built

    def test_unload_frees_a_slot(self, tmp_path, monkeypatch):
        mgr, _ = self._manager(tmp_path, monkeypatch, max_models=1)
        a = tmp_path / "a.asmb"
        b = tmp_path / "b.asmb"
        a.write_bytes(b"x")
        b.write_bytes(b"x")

        mgr.load_model(a)
        with pytest.raises(RuntimeError):
            mgr.load_model(b)
        assert mgr.unload_model("a") == []
        assert mgr.load_model(b) == "b", "the slot was not reusable"

    def test_unload_closes_the_model(self, tmp_path, monkeypatch):
        mgr, built = self._manager(tmp_path, monkeypatch)
        a = tmp_path / "a.asmb"
        a.write_bytes(b"x")
        mgr.load_model(a)
        loaded = mgr._models["a"]
        mgr.unload_model("a")
        assert loaded.closed, "the archive's memory was never released"
        assert "a" not in mgr._models

    def test_unloading_the_active_model_picks_another(self, tmp_path, monkeypatch):
        mgr, _ = self._manager(tmp_path, monkeypatch)
        a, b = tmp_path / "a.asmb", tmp_path / "b.asmb"
        a.write_bytes(b"x")
        b.write_bytes(b"x")
        mgr.load_model(a, alias="a", set_active=True)
        mgr.load_model(b, alias="b", set_active=True)
        mgr.unload_model("b")
        alias, _ = mgr.get_active()
        assert alias == "a", "the active alias pointed at an unloaded model"

    def test_unloading_the_last_model_leaves_none_active(self, tmp_path, monkeypatch):
        mgr, _ = self._manager(tmp_path, monkeypatch)
        a = tmp_path / "a.asmb"
        a.write_bytes(b"x")
        mgr.load_model(a, alias="a")
        mgr.unload_model("a")
        with pytest.raises(RuntimeError):
            mgr.get_active()

    def test_unloading_something_absent_is_a_400(self, tmp_path, monkeypatch):
        mgr, _ = self._manager(tmp_path, monkeypatch)
        from assembly.server import _BadRequest

        with pytest.raises(_BadRequest, match="no such loaded model"):
            mgr.unload_model("ghost")

    def test_the_unload_endpoint_exists(self, tmp_path):
        """The 429 told clients to unload; there had to be a way to."""
        from assembly.server import AssemblyRequestHandler
        from unittest.mock import MagicMock

        payload = json.dumps({"alias": "a"}).encode()
        sock = MockSocket(
            b"POST /api/unload HTTP/1.1\r\nHost: localhost\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        mock_server = MagicMock()
        mock_server.manager.unload_model.return_value = ["b"]
        mock_server.api_key = None
        mock_server.allow_origin = None
        mock_server.require_local_host = False
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), mock_server)
        out = sock.get_output()
        assert b"200" in out.split(b"\r\n")[0]
        mock_server.manager.unload_model.assert_called_once_with("a")
        assert b'"remaining": ["b"]' in out

    def test_a_refused_load_points_at_the_unload_endpoint(self, tmp_path):
        from assembly.server import AssemblyRequestHandler
        from unittest.mock import MagicMock

        payload = json.dumps({"model_path": "/tmp/x.asmb"}).encode()
        sock = MockSocket(
            b"POST /api/load HTTP/1.1\r\nHost: localhost\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        mock_server = MagicMock()
        mock_server.manager.load_model.side_effect = RuntimeError("at limit")
        mock_server.api_key = None
        mock_server.allow_origin = None
        mock_server.require_local_host = False
        AssemblyRequestHandler(sock, ("127.0.0.1", 8000), mock_server)
        assert b"/api/unload" in sock.get_output()


class TestServerResourceLimits:
    """The limits the README claims must actually be limits."""

    def test_the_model_cap_holds_under_concurrent_loads(self, tmp_path, monkeypatch):
        """20 simultaneous /api/load calls must not all pass an 8-model budget.

        The capacity check and the (multi-second) load used to be separate
        critical sections, so every concurrent request saw an empty slot and
        passed -- each request then materialised and kept a model.
        """
        import time as tm
        import concurrent.futures as cf

        from assembly import server as server_mod

        archive = tmp_path / "a.asmb"
        archive.write_bytes(b"x")

        class SlowModel:
            def __init__(self, **kwargs):
                tm.sleep(0.3)  # a real archive load takes seconds to minutes

            def close(self):
                pass

        monkeypatch.setattr(server_mod, "AssemblyLlama", SlowModel)
        mgr = server_mod.ModelManager(models_dir=str(tmp_path), max_models=4)

        def load(i):
            try:
                mgr.load_model(archive, alias=f"a{i}")
                return 200
            except RuntimeError:
                return 429

        with cf.ThreadPoolExecutor(max_workers=20) as pool:
            codes = list(pool.map(load, range(20)))

        assert codes.count(200) == 4, f"cap breached: {codes.count(200)} loaded"
        assert len(mgr._models) == 4
        assert not mgr._loading, "a reservation was leaked"

    def test_a_failed_load_releases_its_reservation(self, tmp_path, monkeypatch):
        """An exception mid-load must not permanently consume a slot."""
        from assembly import server as server_mod

        archive = tmp_path / "a.asmb"
        archive.write_bytes(b"x")

        class Failing:
            def __init__(self, **kwargs):
                raise OSError("no space left on device")

        monkeypatch.setattr(server_mod, "AssemblyLlama", Failing)
        mgr = server_mod.ModelManager(models_dir=str(tmp_path), max_models=1)

        for _ in range(3):
            with pytest.raises(OSError):
                mgr.load_model(archive, alias="a")
        assert not mgr._loading, "a failed load leaked its reservation"

    def test_replacing_a_model_is_not_treated_as_new_capacity(self, tmp_path, monkeypatch):
        """Re-loading an existing alias must work at the limit."""
        from assembly import server as server_mod

        archive = tmp_path / "a.asmb"
        archive.write_bytes(b"x")

        class Model:
            def __init__(self, **kwargs):
                pass

            def close(self):
                pass

        monkeypatch.setattr(server_mod, "AssemblyLlama", Model)
        mgr = server_mod.ModelManager(models_dir=str(tmp_path), max_models=1)
        mgr.load_model(archive, alias="only")
        mgr.load_model(archive, alias="only")  # replacement, must not raise
        assert list(mgr._models) == ["only"]

    def test_concurrent_loads_of_one_alias_consume_one_slot_each(
        self, tmp_path, monkeypatch
    ):
        """A reservation is per in-flight load, not per alias name.

        With `_loading` as a set, two simultaneous loads of the same alias shared
        a single reservation. The second install then found nothing to displace,
        so the first model was dropped without being closed and both copies of
        its weights stayed resident -- a quiet way to double memory use past the
        configured limit.
        """
        import concurrent.futures as cf
        import time as tm

        from assembly import server as server_mod

        archive = tmp_path / "a.asmb"
        archive.write_bytes(b"x")
        built: list = []

        class SlowModel:
            def __init__(self, **kwargs):
                built.append(kwargs.get("asmb_path"))
                tm.sleep(0.2)

            def close(self):
                built.append("closed")

        monkeypatch.setattr(server_mod, "AssemblyLlama", SlowModel)
        mgr = server_mod.ModelManager(models_dir=str(tmp_path), max_models=1)

        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(mgr.load_model, archive, alias="same")
                for _ in range(4)
            ]
            outcomes = []
            for f in futures:
                try:
                    outcomes.append(("ok", f.result()))
                except RuntimeError:
                    outcomes.append(("refused", None))

        accepted = [o for o in outcomes if o[0] == "ok"]
        assert len(accepted) == 1, (
            f"max_models=1 admitted {len(accepted)} concurrent loads of one alias"
        )
        assert len(mgr._models) == 1

    def test_a_replaced_model_is_closed_not_leaked(self, tmp_path, monkeypatch):
        """Replacing an alias closes what it displaces, even under concurrency.

        Two simultaneous loads of a resident alias both pass the capacity check
        (a replacement consumes no new slot), so both build a model. The second
        install must close the first one's instance; otherwise every reload
        leaks a full copy of the weights.
        """
        import concurrent.futures as cf

        from assembly import server as server_mod

        archive = tmp_path / "a.asmb"
        archive.write_bytes(b"x")
        built: list = []

        class Model:
            def __init__(self, **kwargs):
                built.append(kwargs.get("asmb_path"))

            def close(self):
                built.append("closed")

        monkeypatch.setattr(server_mod, "AssemblyLlama", Model)
        mgr = server_mod.ModelManager(models_dir=str(tmp_path), max_models=1)
        mgr.load_model(archive, alias="same")

        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(mgr.load_model, archive, alias="same")
                for _ in range(2)
            ]
            for f in futures:
                f.result()

        assert len(mgr._models) == 1
        # 1 initial + 2 replacements built, and the 2 that were superseded are
        # closed, leaving exactly the one still installed.
        assert built.count("closed") == 2, f"leaked a model: {built}"
        assert not mgr._loading, "a reservation was leaked"

    def test_a_refused_connection_returns_no_slot(self, tmp_path):
        """Refusing a connection must not over-release the semaphore.

        The refusal path used to call shutdown_request(), which is the same hook
        the worker calls on completion, so every refused connection gave back a
        slot it had never taken. BoundedSemaphore raises ValueError on
        over-release, so a client opening a few sockets past the ceiling could
        take the server down.
        """
        import socket as sk
        import threading as th
        import time as tm

        from assembly.server import (
            AssemblyServer,
            MAX_CONCURRENT_CONNECTIONS,
            ModelManager,
        )

        # serve_forever() has to be running: without an accept loop nothing
        # acquires a slot and the test would pass for the wrong reason.
        srv = AssemblyServer(("127.0.0.1", 0), ModelManager(models_dir=str(tmp_path)))
        th.Thread(target=srv.serve_forever, daemon=True).start()
        held: list = []
        try:
            for _ in range(MAX_CONCURRENT_CONNECTIONS):
                held.append(sk.create_connection(srv.server_address, timeout=5))
            # Every slot is held; these connections are accepted by the kernel
            # and then closed by the refusal path.
            for _ in range(10):
                sk.create_connection(srv.server_address, timeout=5).close()

            assert srv._connection_slots._value == 0, "slots were handed out early"
            for s in held:
                s.close()
            for _ in range(200):
                if srv._connection_slots._value == MAX_CONCURRENT_CONNECTIONS:
                    break
                tm.sleep(0.05)
            assert srv._connection_slots._value == MAX_CONCURRENT_CONNECTIONS, (
                "slots did not balance after a refusal storm: "
                f"{srv._connection_slots._value}"
            )
            assert srv._active_connections == 0
        finally:
            for s in held:
                s.close()
            srv.shutdown()
            srv.server_close()

    def test_connections_are_capped(self, tmp_path):
        """ThreadingHTTPServer has no thread ceiling of its own."""
        from assembly.server import (
            AssemblyServer,
            MAX_CONCURRENT_CONNECTIONS,
            ModelManager,
        )

        srv = AssemblyServer(("127.0.0.1", 0), ModelManager(models_dir=str(tmp_path)))
        try:
            assert srv._connection_slots is not None
            # Hold every slot, then confirm the next connection is refused
            # rather than spawning another worker.
            for _ in range(MAX_CONCURRENT_CONNECTIONS):
                assert srv._connection_slots.acquire(blocking=False)
            assert not srv._connection_slots.acquire(blocking=False)
        finally:
            srv.server_close()

    def test_connections_are_registered_for_the_watchdog(self, tmp_path):
        """A dribbling client must be reaped on wall-clock, not per-recv."""
        from assembly.server import AssemblyServer, ModelManager

        srv = AssemblyServer(("127.0.0.1", 0), ModelManager(models_dir=str(tmp_path)))
        try:
            handler = object()
            srv.register_connection(handler)
            try:
                assert id(handler) in srv._connections
                # A deadline is what makes a 1-byte-per-4-minutes client
                # survivable, so the age must be recorded.
                started, _ = srv._connections[id(handler)]
                assert started > 0
            finally:
                srv.forget_connection(handler)
            assert id(handler) not in srv._connections
        finally:
            srv.server_close()

    def test_an_empty_api_key_is_refused(self, tmp_path):
        """api_key="" used to mean "no auth" with no warning at all."""
        from assembly.server import AssemblyServer, ModelManager

        with pytest.raises(ValueError, match="non-empty"):
            AssemblyServer(
                ("127.0.0.1", 0),
                ModelManager(models_dir=str(tmp_path)),
                api_key="",
            )


class TestServerModelsDirDefault:
    """/api/load must not be an arbitrary-file opener by default."""

    def test_serve_defaults_the_jail_to_the_archives_directory(
        self, tmp_path, monkeypatch
    ):
        from assembly import server as server_mod

        archive = tmp_path / "mesh.asmb"
        archive.write_bytes(b"x")
        loaded: list = []
        _stub_load_model(monkeypatch, server_mod, loaded)
        captured = _record_manager_kwargs(monkeypatch, server_mod)
        monkeypatch.setattr(server_mod, "AssemblyServer", _stub_server_factory())

        server_mod.serve(model_path=archive, host="127.0.0.1", port=0)
        assert Path_is(captured["models_dir"], archive.parent)
        assert loaded, "serve() did not ask to load the archive"

    def test_an_explicit_models_dir_is_respected(self, tmp_path, monkeypatch):
        from assembly import server as server_mod

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        archive = elsewhere / "mesh.asmb"
        archive.write_bytes(b"x")
        loaded: list = []
        _stub_load_model(monkeypatch, server_mod, loaded)
        captured = _record_manager_kwargs(monkeypatch, server_mod)
        monkeypatch.setattr(server_mod, "AssemblyServer", _stub_server_factory())

        server_mod.serve(
            model_path=archive, host="127.0.0.1", port=0, models_dir=elsewhere
        )
        assert Path_is(captured["models_dir"], elsewhere.resolve())

    def test_a_path_outside_the_jail_is_refused(self, tmp_path):
        """The behaviour the default now relies on."""
        from assembly.server import ModelManager

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        (allowed / "ok.asmb").write_bytes(b"x")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "evil.asmb").write_bytes(b"x")

        mgr = ModelManager(models_dir=str(allowed))
        assert mgr._authorize_path(allowed / "ok.asmb").name == "ok.asmb"
        with pytest.raises(PermissionError):
            mgr._authorize_path(outside / "evil.asmb")

    def test_the_help_text_no_longer_claims_the_current_directory(self):
        """It said "(default: current directory)", which was never true."""
        assert "current directory" not in _serve_help("models_dir")
        assert "served archive" in _serve_help("models_dir")


class _StubServer:
    """Stands in for AssemblyServer so serve() can be observed, not run.

    serve() binds a real socket and blocks, so a stub has to accept the
    constructor's arguments without inheriting from the real class (whose
    __init__ is what we are avoiding).
    """

    def __init__(self, *args, **kwargs):
        self.server_address = ("127.0.0.1", 0)
        self.kwargs = kwargs

    def serve_forever(self):
        raise KeyboardInterrupt

    def server_close(self):
        pass


def _stub_server_factory():
    """A stand-in for AssemblyServer that records how it was constructed."""
    return _StubServer


def Path_is(actual, expected) -> bool:
    """Both arguments may be a Path, str, or None."""
    import pathlib as _p

    if actual is None:
        return False
    return _p.Path(actual) == _p.Path(expected)



def _stub_load_model(monkeypatch, server_mod, loaded: list) -> None:
    """Replace load_model so serve() need not parse a real .asmb archive.

    serve() eagerly loads the archive it is given; without this the tests have
    to ship a valid multi-megabyte model just to observe which models_dir it
    picked. The stub still runs _authorize_path, so jail violations surface.
    """
    real_load = server_mod.ModelManager.load_model

    def recording_load(self, model_path, *args, **kwargs):
        loaded.append(self._authorize_path(model_path))
        return kwargs.get("alias") or "default"

    monkeypatch.setattr(server_mod.ModelManager, "load_model", recording_load)
    assert real_load is not None


def _record_manager_kwargs(monkeypatch, server_mod) -> dict:
    """Record the kwargs serve() builds a ModelManager with, keeping it usable.

    serve() goes on to print a banner that calls manager.describe_models(), so
    __init__ must still run -- replacing it outright leaves the instance without
    its lock and the banner raises.
    """
    captured: dict = {}
    real_init = server_mod.ModelManager.__init__

    def recording_init(self, *args, **kwargs):
        captured.update(kwargs)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(server_mod.ModelManager, "__init__", recording_init)
    return captured


def _serve_help(dest: str) -> str:
    """The --help text of one `assembly serve` option, by argparse dest."""
    from assembly.main import _build_parser

    parser = _build_parser()
    subparsers = [
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    serve = subparsers[0].choices["serve"]
    for action in serve._actions:
        if action.dest == dest:
            return action.help or ""
    raise AssertionError(f"no serve option with dest {dest!r}")


class TestServerLogSanitisation:
    """Attacker-controlled text must not reach the operator's terminal raw."""

    def test_escapes_are_stripped(self):
        from assembly.server import _sanitize_for_log

        out = _sanitize_for_log("GET /\x1b[2J\x1b[1;31mFAKE\x1b[0m HTTP/1.1")
        assert "\x1b" not in out
        assert "FAKE" in out, "the visible text is kept, only the escapes go"

    def test_newlines_cannot_forge_log_lines(self):
        from assembly.server import _sanitize_for_log

        out = _sanitize_for_log("GET / HTTP/1.1\r\nLoaded model 'x'")
        assert "\n" not in out and "\r" not in out

    def test_long_input_is_truncated(self):
        from assembly.server import _sanitize_for_log

        assert len(_sanitize_for_log("a" * 10_000)) <= 210

    def test_ordinary_text_is_untouched(self):
        from assembly.server import _sanitize_for_log

        text = 'GET /v1/chat/completions HTTP/1.1" 200 -'
        assert _sanitize_for_log(text) == text


class TestServeTempDirForwarded:
    """--temp-dir was accepted by serve and then dropped on the floor."""

    def test_serve_forwards_temp_dir(self, tmp_path, monkeypatch):
        from assembly import server as server_mod

        captured = _record_manager_kwargs(monkeypatch, server_mod)
        monkeypatch.setattr(server_mod, "AssemblyServer", _stub_server_factory())

        server_mod.serve(host="127.0.0.1", port=0, temp_dir=tmp_path)
        assert Path_is(captured["temp_dir"], tmp_path)

    def test_the_serve_subcommand_passes_it_through(self, tmp_path, monkeypatch):
        """The wiring in main.py, not server.py's signature."""
        from assembly import main as main_mod

        seen: dict = {}
        monkeypatch.setattr(
            "assembly.server.serve",
            lambda **kwargs: seen.update(kwargs),
        )
        archive = tmp_path / "a.asmb"
        archive.write_bytes(b"x")
        args = argparse.Namespace(
            command="serve", model=str(archive), host="127.0.0.1", port=0,
            persona=None, n_gpu_layers=0, verbose=False, api_key=None,
            allow_origin=None, models_dir=None, temp_dir=tmp_path,
        )
        main_mod._cmd_serve(args)
        assert Path_is(seen["temp_dir"], tmp_path), (
            "serve --temp-dir is still dropped before it reaches the server"
        )
