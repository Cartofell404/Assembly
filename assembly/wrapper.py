"""
AssemblyLlama — a llama-cpp-python wrapper that transparently loads
models from a packed .asmb archive via mmap.

Design constraints
------------------
* **One copy, not an extraction tree.**  GGUF blobs are read through
  memory-mapped pointers into the .asmb file, so a single archive is your
  whole model store.  llama.cpp needs a real filesystem path to mmap a
  model, so each blob is staged into a temporary file that is deleted as
  soon as it is loaded: an anonymous in-memory fd on Linux (``memfd_create``)
  and a temp file on macOS and other platforms.  On macOS that staging copy
  is real disk I/O — see ``assembly_temp_dir`` and ``--temp-dir``.
* **Lazy model loading.**  Only the embedding model is loaded at init
  time.  Execution models are loaded on-demand and cached so that
  switching back to a previously used model is instantaneous.
* **Drop-in replacement.**  ``AssemblyLlama`` exposes ``__call__`` and
  ``create_chat_completion`` with signatures identical to
  ``llama_cpp.Llama``.
* **Thread-safe.**  llama.cpp contexts are not, so loading and generation
  are serialised per archive.
"""

from __future__ import annotations

import mmap
import os
import shutil
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from assembly.config import gguf_split_info_from_bytes
from assembly.header import PREAMBLE_SIZE, AsmbHeader, InvalidArchiveError, read_header
from assembly.router import RouteResult, Router


#: Where the per-process staging directory for model blobs is created.
#: Overridable with the ``ASSEMBLY_TEMP_DIR`` environment variable or the
#: ``--temp-dir`` flag so users can point at a fast, roomy volume.
TEMP_DIR_ENV = "ASSEMBLY_TEMP_DIR"

_TEMP_DIR: Optional[Path] = None
_TEMP_DIR_LOCK = threading.Lock()


def configure_temp_dir(directory: str | Path) -> Path:
    """Point the staging directory at *directory* and create it.

    Raises :class:`StagingError` if it does not exist, cannot be created, or
    is not a directory — a silent fallback to the system temp directory would
    just hand the user a confusing out-of-space error much later.
    """
    global _TEMP_DIR
    target = Path(directory).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StagingError(
            f"cannot use {target} as the staging directory: {exc}"
        ) from exc
    if not target.is_dir():
        raise StagingError(f"staging path is not a directory: {target}")

    with _TEMP_DIR_LOCK:
        # Drop the previously created staging root; the per-model files
        # inside it are unlinked as their context managers exit, so the
        # directory itself is all that leaks.
        if _TEMP_DIR is not None and _TEMP_DIR != target:
            shutil.rmtree(_TEMP_DIR, ignore_errors=True)
        _TEMP_DIR = target
    return target


def assembly_temp_dir() -> Path:
    """Return (creating once) the directory used to stage model blobs."""
    global _TEMP_DIR
    with _TEMP_DIR_LOCK:
        if _TEMP_DIR is None:
            configured = os.environ.get(TEMP_DIR_ENV)
            base = Path(configured).expanduser() if configured else Path(tempfile.gettempdir())
            try:
                base.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise StagingError(
                    f"cannot use {base} as the staging directory: {exc}. "
                    f"Set {TEMP_DIR_ENV} or pass --temp-dir to point at a "
                    f"writable volume."
                ) from exc
            _TEMP_DIR = Path(tempfile.mkdtemp(prefix="assembly-", dir=str(base)))
        return _TEMP_DIR


def free_bytes_at(directory: Path) -> int:
    """Free space on the filesystem backing *directory* (0 if unknown)."""
    try:
        return shutil.disk_usage(str(directory)).free
    except OSError:
        return 0


# ─────────────────────────────────────────────────────────────────────
# Platform-specific in-memory file descriptor helpers
# ─────────────────────────────────────────────────────────────────────

class StagingError(RuntimeError):
    """Raised when a model blob cannot be staged for llama.cpp."""


def _write_blob(target, data: memoryview | bytes) -> None:
    """Copy *data* into the open binary file *target* in bounded chunks."""
    mv = memoryview(data) if not isinstance(data, memoryview) else data
    offset = 0
    total = len(mv)
    while offset < total:
        chunk = bytes(mv[offset : offset + 8 * 1024 * 1024])
        target.write(chunk)
        offset += len(chunk)


@contextmanager
def _memfd_model(name: str, data: memoryview | bytes) -> Iterator[str]:
    """Yield a filesystem path holding *data*, removed on exit.

    On Linux this is an anonymous in-memory file (``memfd_create``) — no
    visible path and no directory entry at all.

    Everywhere else, including macOS, the blob is staged as a real file in a
    private temp directory.  That copy is genuine disk I/O, so we check for
    room first and raise a readable error rather than an ``Errno 28`` from the
    middle of a multi-gigabyte write.
    """
    if sys.platform == "linux":
        fd = os.memfd_create(name)  # type: ignore[attr-defined]
        try:
            # The dup is needed so the memfd offset is reset independently of
            # the file object we write through; it must be closed or the
            # descriptor leaks for the life of the process.
            with os.fdopen(os.dup(fd), "wb", closefd=True) as blob:
                _write_blob(blob, data)
            os.lseek(fd, 0, os.SEEK_SET)
            yield f"/proc/self/fd/{fd}"
        finally:
            os.close(fd)
    else:
        staging = assembly_temp_dir()
        needed = len(data)
        free = free_bytes_at(staging)
        if free and free < needed:
            raise StagingError(
                f"not enough space to stage {name}: need {needed / 1e9:.1f} GB, "
                f"only {free / 1e9:.1f} GB free in {staging}. "
                f"Point ASSEMBLY_TEMP_DIR at a larger volume."
            )
        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".gguf", delete=True, dir=str(staging)
            )
        except OSError as exc:
            raise StagingError(
                f"could not create a staging file in {staging}: {exc}. "
                f"Point ASSEMBLY_TEMP_DIR at a writable volume."
            ) from exc
        with tmp:
            _write_blob(tmp, data)
            tmp.flush()
            yield tmp.name


def _describe_load_failure(exc: Exception) -> str:
    """Turn a llama.cpp load failure into something a user can act on."""
    detail = str(exc).strip() or type(exc).__name__

    if "split file" in detail:
        return (
            "llama.cpp rejected it as a shard of a split GGUF "
            "(only part of the model was present)"
        )
    if "magic" in detail.lower():
        return f"the model data is not valid GGUF ({detail})"
    return f"llama.cpp could not load the model ({detail})"


class _ModelSlot:
    """Holds a loaded ``llama_cpp.Llama`` instance + its metadata."""

    __slots__ = ("name", "llama", "offset", "size")

    def __init__(
        self,
        name: str,
        llama: Any,
        offset: int,
        size: int,
    ) -> None:
        self.name = name
        self.llama = llama
        self.offset = offset
        self.size = size


class AssemblyLlama:
    """Intelligent multi-model wrapper around ``llama_cpp.Llama``.

    Parameters
    ----------
    asmb_path:
        Path to a packed ``.asmb`` archive.
    n_ctx:
        Context length forwarded to every loaded execution model.
    n_gpu_layers:
        GPU layer offload count forwarded to execution models.
    verbose:
        If *True*, print routing decisions and model-load messages.
    temp_dir:
        Directory used to stage model blobs on platforms that cannot pass
        memory to llama.cpp (everywhere except Linux).  Defaults to
        ``$ASSEMBLY_TEMP_DIR`` or the system temp directory.
    **llama_kwargs:
        Any extra keyword arguments forwarded to ``llama_cpp.Llama()``.
    """

    def __init__(
        self,
        asmb_path: str | Path,
        *,
        persona: Optional[str] = None,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        temp_dir: Optional[str | Path] = None,
        **llama_kwargs: Any,
    ) -> None:
        from llama_cpp import Llama

        self._Llama = Llama
        self._asmb_path = Path(asmb_path).expanduser().resolve()
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._verbose = verbose
        self._llama_kwargs = llama_kwargs

        # Must happen before _setup() stages its first blob, and before
        # close() can run, so a bad --temp-dir is a clean startup error.
        if temp_dir is not None:
            configure_temp_dir(temp_dir)

        # Every attribute close() touches is initialised *before* the first
        # fallible call, and the body is wrapped so a failure releases the fd
        # and the mmap instead of leaking them into __del__.
        self._fd: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None
        self._header: Optional[AsmbHeader] = None
        self._router: Optional[Router] = None
        self._embedding_llama: Any = None
        self._model_cache: dict[str, _ModelSlot] = {}
        self._active_slot: Optional[_ModelSlot] = None
        self._persona: Optional[str] = persona or None
        self._lock = threading.RLock()

        try:
            self._setup()
        except BaseException:
            self.close()
            raise

    def _setup(self) -> None:
        """Open the archive, parse it, and load the router embedding model."""
        # ── Open and mmap the archive ────────────────────────────────
        self._fd = os.open(str(self._asmb_path), os.O_RDONLY)
        file_size = os.fstat(self._fd).st_size
        if file_size < PREAMBLE_SIZE:
            raise InvalidArchiveError(
                f"{self._asmb_path.name} is too small to be an .asmb archive "
                f"({file_size} bytes); it may be truncated."
            )
        self._mm = mmap.mmap(self._fd, file_size, access=mmap.ACCESS_READ)

        # ── Parse header (validates version, sizes and offsets) ──────
        self._header = read_header(self._mm)
        header = self._header

        # ── Persona setup ────────────────────────────────────────────
        if self._persona is None:
            if header.persona is not None:
                self._persona = header.persona
            else:
                self._persona = (
                    f"You are {header.assembly_name}, a unified AI assistant. "
                    f"Always identify yourself as {header.assembly_name}. "
                    "Never state or imply that you are a different model (such as Qwen, Llama, Mistral, SmolLM, etc.), "
                    "and never mention that you are composed of multiple separate models. "
                    "Always respond consistently under this identity."
                )

        # ── Router ───────────────────────────────────────────────────
        self._router = Router(header)

        # ── Load the (small) embedding model into RAM ────────────────
        emb = header.embedding
        self._log(f"Loading embedding model '{emb.name}' ({emb.size / 1e6:.1f} MB)")
        self._embedding_llama = self._load_llama_from_mmap(
            name=emb.name,
            offset=emb.offset,
            size=emb.size,
            embedding=True,
            n_ctx=512,
        )

    # ------------------------------------------------------------------
    # Persona & Message Preparation
    # ------------------------------------------------------------------

    def _prepare_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Inject persona into the system message without mutating input."""
        if not self._persona:
            return list(messages)

        msgs = [dict(m) for m in messages]
        has_system = False
        for i, m in enumerate(msgs):
            if m.get("role") == "system":
                has_system = True
                # OpenAI allows content: null on tool-call messages.
                content = m.get("content") or ""
                if self._persona not in content:
                    msgs[i]["content"] = f"{self._persona}\n\n{content}".strip()
                break

        if not has_system:
            msgs.insert(0, {"role": "system", "content": self._persona})

        return msgs

    # ------------------------------------------------------------------
    # Model loading from mmap
    # ------------------------------------------------------------------

    def _load_llama_from_mmap(
        self,
        name: str,
        offset: int,
        size: int,
        embedding: bool = False,
        n_ctx: Optional[int] = None,
    ) -> Any:
        """Load a ``llama_cpp.Llama`` from the mmap'd .asmb region.

        llama.cpp needs a filesystem path, so the blob is staged via
        :func:`_memfd_model` (anonymous in-memory fd on Linux, private temp
        file elsewhere) and removed as soon as llama.cpp has mapped it.
        """
        if self._mm is None:
            raise InvalidArchiveError("archive is not open")

        # memoryview(mm)[a:b] is a zero-copy view of the mapped archive.
        # Slicing the mmap directly — mm[a:b] — returns *bytes*, which would
        # copy the whole multi-gigabyte blob into RAM on every model load.
        model_view = memoryview(self._mm)[offset : offset + size]
        available = len(model_view)
        if available != size:
            model_view.release()
            raise InvalidArchiveError(
                f"model '{name}' is truncated in {self._asmb_path.name} "
                f"(expected {size} bytes at offset {offset:#x}, "
                f"archive holds {available})"
            )

        # Check the metadata before staging. A shard of a split GGUF cannot be
        # loaded by llama.cpp at all, and staging costs a multi-gigabyte copy,
        # so there is no reason to discover that after the write.
        split_count, split_no = gguf_split_info_from_bytes(
            bytes(model_view[: (64 << 20)])
        )
        if split_count is not None and split_count > 1:
            model_view.release()
            raise InvalidArchiveError(
                f"model '{name}' in {self._asmb_path.name} is shard "
                f"{split_no + 1} of {split_count} from a split GGUF, which "
                f"llama.cpp cannot load on its own. Rebuild the archive from a "
                f"single merged .gguf file."
            )

        try:
            with _memfd_model(name, model_view) as model_path:
                return self._Llama(
                    model_path=model_path,
                    embedding=embedding,
                    n_ctx=n_ctx or self._n_ctx,
                    n_gpu_layers=self._n_gpu_layers,
                    verbose=False,
                    **self._llama_kwargs,
                )
        finally:
            # Release the mmap export explicitly. If Llama() raised, the
            # traceback keeps this frame alive, and an outstanding export
            # would make mmap.close() raise BufferError and skip the
            # os.close() below.
            model_view.release()

    def _ensure_model(self, route: RouteResult) -> _ModelSlot:
        """Ensure the execution model for *route* is loaded and cached.

        Serialised: two threads routing to the same cold model would
        otherwise both stage and load a multi-gigabyte copy.
        """
        with self._lock:
            if route.model_name in self._model_cache:
                slot = self._model_cache[route.model_name]
                self._active_slot = slot
                return slot

            entry = self._header.models[route.model_index]
            self._log(
                f"Loading execution model '{entry.name}' "
                f"({entry.size / 1e6:.1f} MB, offset {entry.offset:#x})"
            )
            try:
                llm = self._load_llama_from_mmap(
                    name=entry.name,
                    offset=entry.offset,
                    size=entry.size,
                )
            except InvalidArchiveError:
                raise
            except Exception as exc:
                # llama.cpp reports a sharded GGUF as
                # "invalid split file name", and a damaged blob as a generic
                # load failure. Both arrive as ValueError from a temp path the
                # user has never seen, so restate them in terms they can act on.
                raise RuntimeError(
                    f"could not load execution model '{entry.name}' from "
                    f"{self._asmb_path.name}: {_describe_load_failure(exc)}. "
                    f"If this model came from a sharded download, it must be "
                    f"merged into a single .gguf before it can be packed."
                ) from exc
            slot = _ModelSlot(
                name=entry.name,
                llama=llm,
                offset=entry.offset,
                size=entry.size,
            )
            self._model_cache[entry.name] = slot
            self._active_slot = slot
            return slot

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _embed_prompt(self, text: str) -> np.ndarray:
        """Compute the embedding vector for *text* using the router model."""
        emb = self._embedding_llama.embed(text)
        if not emb:
            raise RuntimeError("router model returned an empty embedding")
        if isinstance(emb[0], list):
            emb = emb[0]
        return np.array(emb, dtype=np.float32)

    def route(self, prompt: str) -> RouteResult:
        """Run the router on *prompt* and return the decision."""
        # The embedding context is shared state, so embedding is serialised.
        with self._lock:
            vec = self._embed_prompt(prompt)
            return self._router.route(vec)

    def warm(self, model_name: str) -> bool:
        """Eagerly load the execution model called *model_name*.

        Returns ``True`` if it had to be loaded, ``False`` if it was already
        resident.  Lets a CLI show progress *before* generation starts rather
        than stalling on the first token.
        """
        if model_name in self._model_cache:
            return False
        for index, entry in enumerate(self._header.models):
            if entry.name == model_name:
                self._ensure_model(RouteResult(
                    model_name=entry.name,
                    model_index=index,
                    score=0.0,
                    is_fallback=False,
                ))
                return True
        raise KeyError(f"Unknown model: {model_name}")

    # ------------------------------------------------------------------
    # Public API  (drop-in for llama_cpp.Llama)
    # ------------------------------------------------------------------

    def __call__(self, prompt: str, **kwargs: Any) -> Any:
        """Generate a completion — routes to the best model first.

        If a persona is configured, it executes with persona context
        and returns a response with both ``choice["text"]`` and
        ``choice["message"]`` for maximum compatibility.
        """
        slot = self._route_and_load(prompt)

        if self._persona:
            try:
                msgs = self._prepare_messages([{"role": "user", "content": prompt}])
                resp = slot.llama.create_chat_completion(messages=msgs, **kwargs)
                # Ensure choices have both 'text' and 'message'
                choices = resp.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    if "text" not in choices[0]:
                        choices[0]["text"] = content
                return resp
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                # Only chat-template shape problems are recoverable here; a
                # swallowed MemoryError or a native failure must surface.
                self._log(
                    f"Chat template unavailable, falling back to raw "
                    f"completion: {type(e).__name__}: {e}"
                )

        with self._lock:
            return slot.llama(prompt, **kwargs)

    def create_completion(self, prompt: str, **kwargs: Any) -> Any:
        """``Llama.create_completion`` equivalent with automatic routing."""
        slot = self._route_and_load(prompt)
        with self._lock:
            return slot.llama.create_completion(prompt, **kwargs)

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        inject_persona: bool = True,
        **kwargs: Any,
    ) -> Any:
        """``Llama.create_chat_completion`` equivalent with routing and persona.

        The last user message is used for routing.
        """
        # Extract the last user message for routing.
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_text = msg.get("content") or ""
                break
        if not user_text and messages:
            user_text = messages[-1].get("content") or ""

        slot = self._route_and_load(user_text)
        effective_messages = (
            self._prepare_messages(messages) if inject_persona else messages
        )
        with self._lock:
            return slot.llama.create_chat_completion(effective_messages, **kwargs)

    def _route_and_load(self, text: str) -> _ModelSlot:
        """Route *text* and return the loaded model slot that will answer."""
        result = self.route(text)
        self._log(
            f"Routed to '{result.model_name}' "
            f"(score={result.score:.4f}, fallback={result.is_fallback})"
        )
        return self._ensure_model(result)

    def chat(self, prompt: str, **kwargs: Any) -> str:
        """Convenience method: sends a prompt and returns the assistant's text."""
        resp = self.create_chat_completion(
            [{"role": "user", "content": prompt}],
            stream=False,
            **kwargs,
        )
        choices = resp.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content", "") or choices[0].get("text", "")
        return ""

    def stream_chat(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream assistant tokens as they are generated in real-time."""
        gen = self.create_chat_completion(
            [{"role": "user", "content": prompt}],
            stream=True,
            **kwargs,
        )
        for chunk in gen:
            if isinstance(chunk, dict):
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "") or choices[0].get("text", "")
                    if token:
                        yield token
            elif isinstance(chunk, str):
                yield chunk

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all resources. Safe to call more than once."""
        # Every step is independent: one failure must not strand the fd.
        for name in list(self._model_cache):
            slot = self._model_cache.pop(name, None)
            if slot is not None:
                try:
                    del slot.llama
                except Exception:
                    pass
        self._active_slot = None

        if self._embedding_llama is not None:
            try:
                del self._embedding_llama
            except Exception:
                pass
            self._embedding_llama = None

        if self._mm is not None:
            mm, self._mm = self._mm, None
            try:
                mm.close()
            except BufferError:
                # Someone still holds a view; drop it rather than leak the fd.
                pass
            except Exception:
                pass

        if self._fd is not None:
            fd, self._fd = self._fd, None
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> AssemblyLlama:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def header(self) -> AsmbHeader:
        if self._header is None:
            raise InvalidArchiveError("archive failed to open")
        return self._header

    @property
    def source_path(self) -> Path:
        """Path of the backing ``.asmb`` archive."""
        return self._asmb_path

    @property
    def active_model(self) -> Optional[str]:
        return self._active_slot.name if self._active_slot else None

    @property
    def persona(self) -> Optional[str]:
        return self._persona

    @property
    def loaded_models(self) -> list[str]:
        return list(self._model_cache.keys())

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"[assembly] {msg}")
