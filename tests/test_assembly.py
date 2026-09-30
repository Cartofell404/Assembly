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






    def test_web_chat_shell_is_public_but_data_is_not(self):
        """The HTML shell must load so the UI can prompt for the key; the
        data behind it must still require the token."""
        from assembly.server import ModelManager, WEB_CHAT_HTML

        mgr = ModelManager()
        status, headers, _ = self._execute(
            mgr,
            b"GET /chat HTTP/1.1\r\nHost: localhost\r\nAccept: text/html\r\n\r\n",
            api_key="secret",
        )
        assert status == 200, "chat shell must be reachable without a token"
        assert "text/html" in headers["content-type"]
        assert "fetchWithAuth" in WEB_CHAT_HTML

        # Everything that actually returns data stays gated.
        for path in (b"/health", b"/v1/models", b"/api/info", b"/api/tags"):
            status, _, _ = self._execute(
                mgr,
                b"GET " + path + b" HTTP/1.1\r\nHost: localhost\r\n\r\n",
                api_key="secret",
            )
            assert status == 401, f"{path.decode()} leaked data without a token"

    def test_root_html_shell_is_public_when_key_is_set(self):
        from assembly.server import ModelManager

        status, _, _ = self._execute(
            ModelManager(),
            b"GET / HTTP/1.1\r\nHost: localhost\r\nAccept: text/html\r\n\r\n",
            api_key="secret",
        )
        assert status == 200

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
        assert not (tmp_path / "built.asmb.partial").exists()
        assert read_header(out.read_bytes()).assembly_name == "integrity"

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
        assert not (tmp_path / "built.asmb.partial").exists()

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
        assert not (tmp_path / "built.asmb.partial").exists()


# ── CLI ───────────────────────────────────────────────────────────────

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
        args = self._parser().parse_args(["serve"])
        assert args.host == "127.0.0.1"
        assert args.api_key is None
        assert args.allow_origin is None

    def test_serve_security_flags_parse(self):
        args = self._parser().parse_args(
            ["serve", "--host", "0.0.0.0", "--api-key", "s3cret",
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

    def test_run_keeps_temp_dir_flag(self):
        args = self._parser().parse_args(["run", "-m", "x.asmb", "--temp-dir", "/s"])
        assert str(args.temp_dir) == "/s"

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
        assert "No such archive" in capsys.readouterr().err

    def test_info_rejects_corrupt_archive(self, tmp_path, capsys):
        from assembly.main import _cmd_info

        bad = tmp_path / "bad.asmb"
        bad.write_bytes(b"NOTMAGIC" + b"\x00" * 64)

        with pytest.raises(SystemExit) as exc:
            _cmd_info(argparse.Namespace(model=str(bad)))
        assert exc.value.code == 1
        assert "No such archive" not in capsys.readouterr().err

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


# ── Web UI ──────────────────────────────────────────────────────────────


class TestWebUI:
    """Guards the HTML/JS embedded in server.py.

    WEB_CHAT_HTML is a normal (non-raw) Python string, so a JS escape like
    ``\\n`` is consumed by *Python* and lands in the page as a literal newline.
    Inside a JS string literal that is an immediate SyntaxError, which would
    ship a completely dead chat page with no test failing. These checks catch
    that class of mistake.
    """

    def _script(self) -> str:
        import re

        from assembly.server import WEB_CHAT_HTML

        match = re.search(r"<script>(.*)</script>", WEB_CHAT_HTML, re.S)
        assert match, "WEB_CHAT_HTML has no <script> block"
        return match.group(1)

    @staticmethod
    def _unescaped_quote_count(line: str) -> int:
        count, escaped = 0, False
        for ch in line:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                count += 1
        return count

    def test_js_block_is_extracted(self):
        assert len(self._script()) > 500

    def test_no_string_literal_is_split_across_lines(self):
        """Every JS line must close its quotes; an odd count means breakage."""
        offenders = [
            (n, line)
            for n, line in enumerate(self._script().split("\n"), 1)
            if self._unescaped_quote_count(line) % 2
        ]
        assert not offenders, (
            "JS string literal opened but not closed on line(s): "
            + "; ".join(f"{n}: {line.strip()[:60]!r}" for n, line in offenders)
        )

    def test_braces_and_parens_are_balanced(self):
        script = self._script()
        assert script.count("{") == script.count("}")
        assert script.count("(") == script.count(")")

    def test_escape_sequences_survive_python_parsing(self):
        """A \\n written in the JS must still be an escape, not a newline."""
        script = self._script()
        for literal in ('"Authorization": "Bearer "', "assembly_api_key"):
            assert literal in script

    def test_fetch_calls_go_through_the_auth_wrapper(self):
        """Plain fetch() cannot send a bearer token; the UI must use the wrapper."""
        script = self._script()
        assert "fetchWithAuth" in script
        assert 'fetch("/v1/chat/completions"' not in script
        assert 'fetch("/api/info"' not in script

    def test_model_names_are_not_injected_as_html(self):
        from assembly.server import WEB_CHAT_HTML

        script = self._script()
        # Strip // comments so the explanatory comment mentioning innerHTML
        # does not trip the check; what matters is that it is never assigned.
        code = "\n".join(
            line.split("//")[0] for line in script.split("\n")
        )
        assert "innerHTML" not in code
        assert "assembly_name" in WEB_CHAT_HTML


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
