"""Configuration schema — parsing and validation of assembly JSON configs."""

from __future__ import annotations

import json
import os
import pathlib
import struct
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Settings for the router (embedding) model."""

    name: str
    path: pathlib.Path
    confidence_threshold: float = 0.75
    fallback_model: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValueError(
                f"Embedding path must be absolute, got: {self.path}"
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got: "
                f"{self.confidence_threshold}"
            )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Settings for a single execution model."""

    name: str
    path: pathlib.Path
    anchors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValueError(
                f"Model path must be absolute, got: {self.path}"
            )
        if len(self.anchors) < 1:
            raise ValueError(
                f"Model '{self.name}' must define at least one anchor."
            )


@dataclass(frozen=True, slots=True)
class AssemblyConfig:
    """Top-level assembly configuration."""

    assembly_name: str
    embedding: EmbeddingConfig
    models: list[ModelConfig]
    persona: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.models) < 2:
            raise ValueError("At least 2 execution models are required.")

        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError("Model names must be unique.")

        fb = self.embedding.fallback_model
        if fb is not None and fb not in names:
            raise ValueError(
                f"fallback_model '{fb}' is not among defined models: {names}"
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> AssemblyConfig:
        """Load and validate an assembly config from a JSON file."""
        path = pathlib.Path(path)
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        emb_raw = raw["embedding"]
        embedding = EmbeddingConfig(
            name=emb_raw["name"],
            path=pathlib.Path(emb_raw["path"]),
            confidence_threshold=emb_raw.get("confidence_threshold", 0.75),
            fallback_model=emb_raw.get("fallback_model"),
        )

        models: list[ModelConfig] = []
        for m in raw["models"]:
            models.append(
                ModelConfig(
                    name=m["name"],
                    path=pathlib.Path(m["path"]),
                    anchors=m["anchors"],
                )
            )

        return cls(
            assembly_name=raw["assembly_name"],
            embedding=embedding,
            models=models,
            persona=raw.get("persona"),
        )


# ─────────────────────────────────────────────────────────────────────
# Pre-flight Validation System
# ─────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ValidationIssue:
    severity: str  # "ERROR" or "WARNING"
    field: str
    message: str


@dataclass(slots=True)
class ValidationReport:
    is_valid: bool
    issues: list[ValidationIssue]
    total_size_bytes: int
    free_disk_bytes: int
    models_info: list[dict[str, Any]]
    config: Optional[AssemblyConfig] = None

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]


class PreflightError(ValueError):
    """Raised when a config fails pre-flight validation.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handlers
    keep working, while carrying the full :class:`ValidationReport` so the
    CLI can render a detailed report instead of a bare traceback.
    """

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        details = "\n".join(f"  ✗ [{i.field}] {i.message}" for i in report.errors)
        super().__init__(
            f"Pre-flight configuration validation failed "
            f"({len(report.errors)} error(s)):\n{details}\n\n"
            "Fix the issues above before building "
            "(run 'assembly validate -c <config>' for the full report)."
        )


def _resolve_model_path(
    value: Any,
    field: str,
    issues: list[ValidationIssue],
    label: str = "",
) -> Optional[pathlib.Path]:
    """Validate a config-supplied model path.

    Records an issue and returns *None* when the value is not a usable,
    absolute path to an existing file.  Never raises — a malformed config
    must always surface as a report, never a traceback.
    """
    who = f" for '{label}'" if label else ""
    if not isinstance(value, str) or not value.strip():
        issues.append(
            ValidationIssue(
                "ERROR",
                field,
                f"Path{who} must be a non-empty string, got: "
                f"{type(value).__name__} ({value!r})",
            )
        )
        return None

    # expanduser before the absolute check so "~/models/x.gguf" is usable;
    # a bare "~/..." is not absolute and would otherwise be rejected.
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        issues.append(
            ValidationIssue(
                "ERROR", field, f"Path{who} must be absolute, got: '{value}'"
            )
        )
        return None

    if not path.is_file():
        issues.append(
            ValidationIssue("ERROR", field, f"Model file{who} not found: {path}")
        )
        return None

    return path


def _safe_size(
    path: pathlib.Path,
    field: str,
    issues: list[ValidationIssue],
) -> int:
    """Return the size of *path*, recording an issue instead of raising.

    ``is_file()`` already passed, but the file can be removed, renamed, or
    made unreadable between that check and the ``stat()``. Returning -1 marks
    "unknown" so the caller skips the size-dependent checks rather than
    reporting a bogus 0-byte file.
    """
    try:
        return path.stat().st_size
    except OSError as exc:
        issues.append(
            ValidationIssue(
                "ERROR", field, f"Cannot read {path}: {exc}"
            )
        )
        return -1


def gguf_split_info(path: pathlib.Path) -> tuple[Optional[int], Optional[int]]:
    """Return ``(split_count, split_no)`` from a GGUF header on disk."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(64 << 20)
    except OSError:
        return (None, None)
    return gguf_split_info_from_bytes(head)


def gguf_split_info_from_bytes(head: bytes) -> tuple[Optional[int], Optional[int]]:
    """Return ``(split_count, split_no)`` from the first bytes of a GGUF.

    Large models are often published as shards (``model-00001-of-00003.gguf``).
    llama.cpp refuses to load any shard whose metadata declares it is part of a
    split, so such a file can be copied into an archive and byte-verified
    perfectly and still be unloadable at runtime. Detecting it early turns a
    multi-gigabyte staging attempt followed by a confusing failure into an
    immediate, actionable error.

    Returns ``(None, None)`` when the bytes have no split metadata or cannot be
    parsed; callers must treat that as "not checked", never as "invalid".
    """
    import io

    # gguf value type id -> fixed byte width (STRING/ARRAY handled separately)
    fixed = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}

    fh = io.BytesIO(head)

    def read_str() -> None:
        raw = fh.read(8)
        if len(raw) < 8:
            raise ValueError("truncated")
        (n,) = struct.unpack("<Q", raw)
        if n > (1 << 20):
            raise ValueError("implausible string length")
        raw2 = fh.read(n)
        if len(raw2) < n:
            raise ValueError("truncated")
        return raw2.decode("utf-8", "replace")

    def skip_value(t: int) -> None:
        if t == 8:
            read_str()
        elif t == 9:  # ARRAY
            (elem,) = struct.unpack("<I", fh.read(4))
            (count,) = struct.unpack("<Q", fh.read(8))
            if elem == 8:
                for _ in range(count):
                    read_str()
            elif elem in fixed:
                if fixed[elem] * count > len(head):
                    raise ValueError("truncated")
                fh.seek(fixed[elem] * count, os.SEEK_CUR)
            else:
                raise ValueError("unknown array element type")
        elif t in fixed:
            if fixed[t] > len(head):
                raise ValueError("truncated")
            fh.seek(fixed[t], os.SEEK_CUR)
        else:
            raise ValueError("unknown value type")

    try:
        if fh.read(4) != b"GGUF":
            return (None, None)
        if len(fh.read(4)) < 4:
            return (None, None)
        _, n_kv = struct.unpack("<QQ", fh.read(16))

        split_count: Optional[int] = None
        split_no: Optional[int] = None
        for _ in range(n_kv):
            key_start = fh.tell()
            raw = fh.read(8)
            if len(raw) < 8:
                return (None, None)
            (klen,) = struct.unpack("<Q", raw)
            if klen > 4096:
                return (None, None)
            key_raw = fh.read(klen)
            if len(key_raw) < klen:
                return (None, None)
            key = key_raw.decode("utf-8", "replace")
            (vtype,) = struct.unpack("<I", fh.read(4))

            if key in ("split.count", "split.no"):
                width = fixed.get(vtype)
                if width is None:
                    return (None, None)
                raw_value = fh.read(width)
                if len(raw_value) < width:
                    return (None, None)
                value = int.from_bytes(raw_value, "little", signed=vtype in (1, 3, 5, 11))
                if key == "split.count":
                    split_count = value
                else:
                    split_no = value
            else:
                fh.seek(key_start + 8 + klen + 4, os.SEEK_SET)
                skip_value(vtype)
        return (split_count, split_no)
    except (struct.error, ValueError, UnicodeError):
        return (None, None)


def _check_not_sharded(
    path: pathlib.Path,
    field: str,
    label: str,
    issues: list[ValidationIssue],
) -> None:
    """Record an error when *path* is one shard of a split GGUF."""
    split_count, split_no = gguf_split_info(path)
    if split_count is None or split_count <= 1:
        return

    who = f" '{label}'" if label else ""
    issues.append(
        ValidationIssue(
            "ERROR",
            field,
            f"Model{who} is shard {split_no + 1} of {split_count} from a split "
            f"GGUF. llama.cpp cannot load a shard on its own, so this model "
            f"would fail at inference time even though the archive builds "
            f"cleanly. Merge the shards into one .gguf file (or download the "
            f"single-file version) and point the config at that.",
        )
    )


def validate_assembly_preflight(
    config_path: str | pathlib.Path,
    output_path: Optional[str | pathlib.Path] = None,
    check_gguf_magic: bool = True,
) -> ValidationReport:
    """Pre-flight check of an assembly config file before any model building.

    Performs deep validation:
    - JSON syntax and structural schema
    - Absolute path verification and file existence
    - GGUF magic bytes (b"GGUF") check to catch broken/HTML downloads
    - Available disk space calculation
    - Prevention of accidental file overwriting
    - Anchor counts, duplicates, and model name integrity
    """
    import shutil

    issues: list[ValidationIssue] = []
    models_info: list[dict[str, Any]] = []
    total_size_bytes = 0
    free_disk_bytes = 0
    cfg: Optional[AssemblyConfig] = None

    config_path = pathlib.Path(config_path)

    # ── 1. Check file existence & readable JSON ──────────────────────
    if not config_path.is_file():
        issues.append(
            ValidationIssue(
                "ERROR", "config_path", f"Config file does not exist: {config_path}"
            )
        )
        return ValidationReport(
            is_valid=False,
            issues=issues,
            total_size_bytes=0,
            free_disk_bytes=0,
            models_info=[],
        )

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        issues.append(
            ValidationIssue(
                "ERROR",
                "json_syntax",
                f"Malformed JSON in {config_path.name} (line {exc.lineno}, col {exc.colno}): {exc.msg}",
            )
        )
        return ValidationReport(
            is_valid=False,
            issues=issues,
            total_size_bytes=0,
            free_disk_bytes=0,
            models_info=[],
        )
    except Exception as exc:
        issues.append(
            ValidationIssue("ERROR", "config_file", f"Failed to read config: {exc}")
        )
        return ValidationReport(
            is_valid=False,
            issues=issues,
            total_size_bytes=0,
            free_disk_bytes=0,
            models_info=[],
        )

    if not isinstance(raw, dict):
        issues.append(
            ValidationIssue("ERROR", "root", "Top-level JSON element must be an object { ... }")
        )
        return ValidationReport(
            is_valid=False,
            issues=issues,
            total_size_bytes=0,
            free_disk_bytes=0,
            models_info=[],
        )

    # ── 2. Validate assembly_name & persona ──────────────────────────
    name = raw.get("assembly_name")
    if not name or not isinstance(name, str) or not name.strip():
        issues.append(
            ValidationIssue("ERROR", "assembly_name", "Missing or empty 'assembly_name'")
        )

    persona = raw.get("persona")
    if persona is not None and not isinstance(persona, str):
        issues.append(
            ValidationIssue("ERROR", "persona", "'persona' must be a string if defined")
        )

    # ── 3. Validate router embedding model ───────────────────────────
    emb_raw = raw.get("embedding")
    input_paths: list[pathlib.Path] = []
    if not emb_raw or not isinstance(emb_raw, dict):
        issues.append(
            ValidationIssue("ERROR", "embedding", "Missing 'embedding' object in config")
        )
    else:
        emb_name = emb_raw.get("name")
        if not emb_name or not isinstance(emb_name, str):
            issues.append(
                ValidationIssue("ERROR", "embedding.name", "Missing 'embedding.name'")
            )

        emb_path = _resolve_model_path(emb_raw.get("path"), "embedding.path", issues)
        if emb_path is not None:
            input_paths.append(emb_path)
            sz = _safe_size(emb_path, "embedding.path", issues)
            if sz == 0:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "embedding.path",
                        f"Embedding file is empty (0 bytes): {emb_path}",
                    )
                )
            elif sz > 0:
                total_size_bytes += sz
                if check_gguf_magic:
                    try:
                        with open(emb_path, "rb") as f:
                            magic = f.read(4)
                        if magic != b"GGUF":
                            issues.append(
                                ValidationIssue(
                                    "ERROR",
                                    "embedding.path",
                                    f"File is not a valid GGUF model (magic bytes: "
                                    f"{magic!r}, expected b'GGUF'). Check if download "
                                    f"was corrupted.",
                                )
                            )
                    except Exception as e:
                        issues.append(
                            ValidationIssue(
                                "ERROR",
                                "embedding.path",
                                f"Could not read model file: {e}",
                            )
                        )

                    if check_gguf_magic:
                        _check_not_sharded(emb_path, "embedding.path", "", issues)

                models_info.append(
                    {
                        "role": "router_embedding",
                        "name": emb_name or "embedding",
                        "path": str(emb_path),
                        "size_mb": round(sz / 1e6, 1),
                    }
                )

        threshold = emb_raw.get("confidence_threshold", 0.75)
        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            issues.append(
                ValidationIssue(
                    "ERROR",
                    "embedding.confidence_threshold",
                    f"'confidence_threshold' must be between 0.0 and 1.0 (got {threshold})",
                )
            )

    # ── 4. Validate execution models ─────────────────────────────────
    models_raw = raw.get("models")
    model_names: list[str] = []

    if not isinstance(models_raw, list):
        issues.append(
            ValidationIssue("ERROR", "models", "'models' must be a list of at least 2 models")
        )
    elif len(models_raw) < 2:
        issues.append(
            ValidationIssue(
                "ERROR",
                "models",
                f"At least 2 execution models are required (got {len(models_raw)})",
            )
        )
    else:
        for idx, m in enumerate(models_raw):
            if not isinstance(m, dict):
                issues.append(
                    ValidationIssue("ERROR", f"models[{idx}]", "Model entry must be a JSON object")
                )
                continue

            m_name = m.get("name")
            if not m_name or not isinstance(m_name, str) or not m_name.strip():
                issues.append(
                    ValidationIssue("ERROR", f"models[{idx}].name", "Model name cannot be empty")
                )
            else:
                if m_name in model_names:
                    issues.append(
                        ValidationIssue(
                            "ERROR",
                            f"models[{idx}].name",
                            f"Duplicate model name '{m_name}' (names must be unique)",
                        )
                    )
                model_names.append(m_name)

            m_path = _resolve_model_path(
                m.get("path"), f"models[{idx}].path", issues, label=str(m_name)
            )
            if m_path is not None:
                input_paths.append(m_path)
                sz = _safe_size(m_path, f"models[{idx}].path", issues)
                if sz == 0:
                    issues.append(
                        ValidationIssue(
                            "ERROR",
                            f"models[{idx}].path",
                            f"Model file for '{m_name}' is empty (0 bytes): {m_path}",
                        )
                    )
                elif sz > 0:
                    total_size_bytes += sz
                    if check_gguf_magic:
                        try:
                            with open(m_path, "rb") as f:
                                magic = f.read(4)
                            if magic != b"GGUF":
                                issues.append(
                                    ValidationIssue(
                                        "ERROR",
                                        f"models[{idx}].path",
                                        f"File for '{m_name}' is not a valid GGUF "
                                        f"model (magic: {magic!r}). Check if "
                                        f"download was corrupted.",
                                    )
                                )
                        except Exception as e:
                            issues.append(
                                ValidationIssue(
                                    "ERROR",
                                    f"models[{idx}].path",
                                    f"Could not read model file '{m_name}': {e}",
                                )
                            )

                    if check_gguf_magic:
                        _check_not_sharded(
                            m_path, f"models[{idx}].path", str(m_name), issues
                        )

                    anchors = m.get("anchors", [])
                    models_info.append(
                        {
                            "role": "execution_model",
                            "name": m_name,
                            "path": str(m_path),
                            "size_mb": round(sz / 1e6, 1),
                            "anchors_count": (
                                len(anchors) if isinstance(anchors, list) else 0
                            ),
                        }
                    )

            anchors = m.get("anchors")
            if not isinstance(anchors, list) or len(anchors) < 1:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"models[{idx}].anchors",
                        f"Model '{m_name}' must define at least one anchor string",
                    )
                )
            else:
                seen_anchors = set()
                for a_idx, a in enumerate(anchors):
                    if not isinstance(a, str) or not a.strip():
                        issues.append(
                            ValidationIssue(
                                "ERROR",
                                f"models[{idx}].anchors[{a_idx}]",
                                f"Anchor cannot be empty or whitespace in model '{m_name}'",
                            )
                        )
                    elif a.strip().lower() in seen_anchors:
                        issues.append(
                            ValidationIssue(
                                "WARNING",
                                f"models[{idx}].anchors[{a_idx}]",
                                f"Duplicate anchor text '{a}' in model '{m_name}'",
                            )
                        )
                    else:
                        seen_anchors.add(a.strip().lower())

                if len(anchors) < 2:
                    issues.append(
                        ValidationIssue(
                            "WARNING",
                            f"models[{idx}].anchors",
                            f"Model '{m_name}' has only 1 anchor. Adding 3-5 anchors improves routing accuracy.",
                        )
                    )

    # ── 5. Validate fallback_model ───────────────────────────────────
    if isinstance(emb_raw, dict):
        fb = emb_raw.get("fallback_model")
        if fb is not None:
            if not isinstance(fb, str):
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "embedding.fallback_model",
                        f"'fallback_model' must be a string or null, got: {type(fb).__name__}",
                    )
                )
            elif fb not in model_names and model_names:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "embedding.fallback_model",
                        f"fallback_model '{fb}' does not match any defined model: {model_names}",
                    )
                )

    # ── 6. Validate output path & disk space ──────────────────────────
    if output_path:
        out_p = pathlib.Path(output_path).resolve()
        out_dir = out_p.parent

        if out_p.is_dir():
            issues.append(
                ValidationIssue(
                    "ERROR",
                    "output_path",
                    f"Output path is a directory, not a file: {out_p}. "
                    f"Pass a filename, e.g. -o {out_p / (out_p.name + '.asmb')}",
                )
            )

        # Prevent overwriting source models
        for inp in input_paths:
            if out_p == inp.resolve():
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "output_path",
                        f"Output path cannot be identical to input model '{inp.name}'!",
                    )
                )

        if not out_dir.exists():
            issues.append(
                ValidationIssue(
                    "ERROR",
                    "output_path",
                    f"Destination directory does not exist: {out_dir}",
                )
            )
        elif not os.access(out_dir, os.W_OK):
            issues.append(
                ValidationIssue(
                    "ERROR",
                    "output_path",
                    f"Destination directory is not writable: {out_dir}",
                )
            )
        else:
            if out_p.exists() and not out_p.is_dir():
                issues.append(
                    ValidationIssue(
                        "WARNING",
                        "output_path",
                        f"Output file already exists and will be overwritten: {out_p}",
                    )
                )
            try:
                usage = shutil.disk_usage(out_dir)
                free_disk_bytes = usage.free
                # Need models size + header and page alignments (+ 20MB buffer)
                required_bytes = total_size_bytes + (len(models_info) + 2) * 4096 + 20 * 1024 * 1024
                if free_disk_bytes < required_bytes:
                    issues.append(
                        ValidationIssue(
                            "ERROR",
                            "disk_space",
                            f"Insufficient disk space in {out_dir}! "
                            f"Required: {required_bytes / 1e9:.2f} GB, available: {free_disk_bytes / 1e9:.2f} GB.",
                        )
                    )
            except Exception as e:
                issues.append(
                    ValidationIssue(
                        "WARNING",
                        "disk_space",
                        f"Could not verify available disk space: {e}",
                    )
                )

    has_errors = any(i.severity == "ERROR" for i in issues)
    if not has_errors:
        try:
            cfg = AssemblyConfig.from_json(config_path)
        except Exception:
            pass

    return ValidationReport(
        is_valid=not has_errors,
        issues=issues,
        total_size_bytes=total_size_bytes,
        free_disk_bytes=free_disk_bytes,
        models_info=models_info,
        config=cfg,
    )


# ── Sampling ───────────────────────────────────────────────────────────────
# Generation knobs, shared by the HTTP server and the interactive CLI so that
# `/temp 1.2` and a client's `{"temperature": 1.2}` mean exactly the same
# thing.
#
# OpenAI's names are the common denominator, plus `top_k` as a llama.cpp/Ollama
# extension OpenAI does not have. llama.cpp implements repeat, presence and
# frequency penalties as three independent knobs, so all three are kept apart
# rather than folded together. Values arriving from a client are clamped rather
# than rejected — a client sending `frequency_penalty: 3.0` should get a
# working answer, not a 400.


@dataclass
class SamplingParams:
    """Sampler settings for one generation.

    Defaults are llama.cpp's own defaults, which are sane for instruct models.
    """

    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    seed: Optional[int] = None

    def to_llama_kwargs(self) -> dict[str, Any]:
        """The subset llama-cpp-python's ``create_chat_completion`` accepts."""
        out: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_tokens,
        }
        if self.stop:
            out["stop"] = list(self.stop)
        if self.seed is not None:
            out["seed"] = self.seed
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_tokens,
            "stop": list(self.stop),
            "seed": self.seed,
        }


# Named bundles. `precise` is for code and factual work where a stray token
# ruins the answer; `creative` is for prose and brainstorming.
SAMPLING_PRESETS: dict[str, dict[str, Any]] = {
    "precise": {
        "temperature": 0.2, "top_p": 0.90, "top_k": 20,
        "repeat_penalty": 1.05, "max_tokens": 1024,
        "presence_penalty": 0.0, "frequency_penalty": 0.0,
    },
    "balanced": {
        "temperature": 0.8, "top_p": 0.95, "top_k": 40,
        "repeat_penalty": 1.10, "max_tokens": 512,
        "presence_penalty": 0.0, "frequency_penalty": 0.0,
    },
    "creative": {
        "temperature": 1.2, "top_p": 0.98, "top_k": 80,
        "repeat_penalty": 1.15, "max_tokens": 512,
        "presence_penalty": 0.0, "frequency_penalty": 0.0,
    },
}

# Display metadata: label, valid range, and how to format the number.
SAMPLING_FIELDS: dict[str, tuple[str, float, float]] = {
    "temperature": ("temperature", 0.0, 2.0),
    "top_p": ("top_p", 0.0, 1.0),
    "top_k": ("top_k", 1.0, 100.0),
    "repeat_penalty": ("repeat penalty", 1.0, 2.0),
    "presence_penalty": ("presence pen", -2.0, 2.0),
    "frequency_penalty": ("frequency pen", -2.0, 2.0),
    "max_tokens": ("max tokens", 16.0, 8192.0),
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _coerce_float(value: Any, low: float, high: float) -> float | None:
    """Read a float, clamping to the range, or ``None`` if unusable.

    Returning ``None`` matters: a client sending ``temperature: "hot"`` should
    keep the default, not silently become fully deterministic (``0.0``), which
    is a subtle and confusing way to change how a model behaves.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return _clamp(number, low, high)


def sampling_params_from_body(
    body: dict[str, Any], base: SamplingParams | None = None
) -> SamplingParams:
    """Build sampler settings from a request body, over ``base``.

    Accepts OpenAI's names plus llama.cpp's ``top_k``. Each penalty is kept as
    its own knob, because llama.cpp implements all three independently.
    """
    params = base.to_dict() if base else SamplingParams().to_dict()
    stop = list(params["stop"])

    def take(key: str, attr: str, low: float, high: float, cast: Any = float) -> None:
        if key not in body:
            return
        number = _coerce_float(body[key], low, high)
        if number is not None:
            params[attr] = cast(number)

    take("temperature", "temperature", 0.0, 2.0)
    take("top_p", "top_p", 0.0, 1.0)
    take("top_k", "top_k", 1.0, 100.0, int)
    take("max_tokens", "max_tokens", 16.0, 8192.0, int)
    # Newer/alternative spellings of the token budget.
    for key in ("max_completion_tokens", "max_new_tokens"):
        take(key, "max_tokens", 16.0, 8192.0, int)

    if body.get("seed") is not None:
        try:
            params["seed"] = int(body["seed"])
        except (TypeError, ValueError):
            params["seed"] = None

    # llama.cpp exposes all three penalties, so they stay independent. OpenAI's
    # presence/frequency penalties run -2.0..2.0; repeat_penalty is a multiplier
    # that only makes sense above 1.0.
    take("repeat_penalty", "repeat_penalty", 1.0, 2.0)
    take("presence_penalty", "presence_penalty", -2.0, 2.0)
    take("frequency_penalty", "frequency_penalty", -2.0, 2.0)

    raw_stop = body.get("stop")
    if isinstance(raw_stop, str):
        stop = [raw_stop]
    elif isinstance(raw_stop, list):
        stop = [str(s) for s in raw_stop if isinstance(s, (str, int, float))]

    return SamplingParams(
        temperature=float(params["temperature"]),
        top_p=float(params["top_p"]),
        top_k=int(params["top_k"]),
        repeat_penalty=float(params["repeat_penalty"]),
        presence_penalty=float(params["presence_penalty"]),
        frequency_penalty=float(params["frequency_penalty"]),
        max_tokens=int(params["max_tokens"]),
        stop=stop,
        seed=params["seed"],
    )


def sampling_params_from_preset(name: str) -> SamplingParams:
    """Look up a named preset. Raises ``KeyError`` on an unknown name."""
    return SamplingParams(**SAMPLING_PRESETS[name.lower()])
