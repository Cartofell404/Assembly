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
  and a temp file everywhere else.  Off Linux that staging copy is real disk
  I/O — see ``assembly_temp_dir`` and ``--temp-dir``.
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

import atexit
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

from assembly.config import (
    gguf_context_length_from_bytes,
    gguf_split_info_from_bytes,
)
from assembly.header import PREAMBLE_SIZE, AsmbHeader, InvalidArchiveError, read_header
from assembly.router import RouteResult, Router


# Used only when an archive declares no usable context length anywhere. It is
# the historical default, so an archive that cannot be introspected behaves
# exactly as it did before the window was derived from the models.
FALLBACK_CONTEXT = 2048

# Bounds on an auto-detected window. The floor keeps a misdeclared value of 0
# from producing an unusable model; the ceiling stops a corrupt header from
# asking for a KV cache large enough to exhaust memory.
MIN_CONTEXT = 512
MAX_CONTEXT = 1 << 20

# How many models may be resident in RAM at once. Past this the least recently
# used is evicted; its staging file is kept, so returning to a model costs no
# disk copy. Deliberately generous: each resident model is a full set of
# weights, so the real ceiling is usually the machine's memory rather than this
# number. The server lowers it from its settings.
DEFAULT_MAX_LOADED_MODELS = 8

# Fraction of the window at which a conversation is summarised, and the share
# of the window the replacement summary plus recent turns may occupy.
# Triggering late would leave no room to fit the summary itself.
SUMMARIZE_AT = 0.80
SUMMARIZED_BUDGET = 0.50
# A summary is running notes, not a retelling. Capping the reply keeps
# summarisation quick and stops a verbose model from returning a "summary"
# as long as the history it was meant to compress.
SUMMARY_OUTPUT_CEILING = 512
# Token counts are memoised so a growing conversation stops costing more and
# more per request. The cap bounds the memory a very long session can hold.
TOKEN_CACHE_ENTRIES = 4096

# Bytes of a model blob read to find its metadata. Matches the split reader
# so both agree on how much is enough; a model with unusually large tokenizer
# metadata ahead of the context key would otherwise read as undeclared.
HEADER_READ_BYTES = 64 << 20
# Tried in order until the context length is found. 4 MiB covers every GGUF
# seen in practice; the larger read is only for a model with unusually large
# tokenizer metadata ahead of the context key.
HEADER_READ_SPANS = (4 << 20, HEADER_READ_BYTES)

# Tokens reserved for the chat template, the persona and the reply itself.
TEMPLATE_OVERHEAD = 96

#: Where the per-process staging directory for model blobs is created.
#: Overridable with the ``ASSEMBLY_TEMP_DIR`` environment variable or the
#: ``--temp-dir`` flag so users can point at a fast, roomy volume.
TEMP_DIR_ENV = "ASSEMBLY_TEMP_DIR"

_TEMP_DIR: Optional[Path] = None
#: Whether _TEMP_DIR is one we created. A directory the user pointed us at with
#: --temp-dir is theirs and must never be deleted on their behalf.
_TEMP_DIR_OWNED = False
_TEMP_DIR_LOCK = threading.Lock()

#: Staged blobs whose removal failed, retried at exit. On Windows a mapped model
#: stays locked for as long as llama.cpp holds it, so the unlink at unload time
#: is expected to fail; by the time the process exits the mapping is gone. These
#: paths are ours alone (mkstemp names), and removing them never touches the
#: directory itself, so this works for a user-supplied --temp-dir too.
_PENDING_UNLINK: set[Path] = set()
_PENDING_UNLINK_LOCK = threading.Lock()


def _defer_unlink(path: Path) -> None:
    """Record *path* for removal at exit."""
    with _PENDING_UNLINK_LOCK:
        _PENDING_UNLINK.add(path)


def _flush_pending_unlinks() -> None:
    """Remove any staged blob that could not be deleted earlier."""
    while True:
        with _PENDING_UNLINK_LOCK:
            if not _PENDING_UNLINK:
                return
            path = _PENDING_UNLINK.pop()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # Still locked, or the volume is gone. Requeue and give up rather
            # than spinning forever on a file that will never open up.
            with _PENDING_UNLINK_LOCK:
                if len(_PENDING_UNLINK) < 64:
                    _PENDING_UNLINK.add(path)
            return


def configure_temp_dir(directory: str | Path) -> Path:
    """Point the staging directory at *directory* and create it.

    Raises :class:`StagingError` if it does not exist, cannot be created, or
    is not a directory — a silent fallback to the system temp directory would
    just hand the user a confusing out-of-space error much later.
    """
    global _TEMP_DIR, _TEMP_DIR_OWNED
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
        # Only ever remove a directory we created ourselves. A user-supplied
        # --temp-dir is theirs: it may already hold other files, and deleting
        # it to "clean up staging" would take those with it.
        if _TEMP_DIR is not None and _TEMP_DIR != target and _TEMP_DIR_OWNED:
            shutil.rmtree(_TEMP_DIR, ignore_errors=True)
        _TEMP_DIR = target
        _TEMP_DIR_OWNED = False
    return target


def assembly_temp_dir() -> Path:
    """Return (creating once) the directory used to stage model blobs."""
    global _TEMP_DIR, _TEMP_DIR_OWNED
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
            # We own this one, so we are allowed to remove it at exit.
            _TEMP_DIR_OWNED = True
        return _TEMP_DIR


def cleanup_temp_dir() -> None:
    """Remove the staging directory, if we created it.

    Registered with :mod:`atexit` so a normal exit does not leave a directory
    behind. Only ever touches a directory this process made for itself; a
    user-supplied ``--temp-dir`` or ``ASSEMBLY_TEMP_DIR`` is left alone, as is
    any staged blob still held open.
    """
    global _TEMP_DIR, _TEMP_DIR_OWNED
    with _TEMP_DIR_LOCK:
        if _TEMP_DIR is None or not _TEMP_DIR_OWNED:
            return
        directory, _TEMP_DIR, _TEMP_DIR_OWNED = _TEMP_DIR, None, False
    shutil.rmtree(directory, ignore_errors=True)


def free_bytes_at(directory: Path) -> int:
    """Free space on the filesystem backing *directory* (0 if unknown)."""
    try:
        return shutil.disk_usage(str(directory)).free
    except OSError:
        return 0


# Registered at import so a normal interpreter exit reclaims the staging
# directory without every caller having to remember to. This only ever removes a
# directory this process created; see cleanup_temp_dir.
atexit.register(cleanup_temp_dir)
# Registered second so it runs *first*: a pending blob must be unlinked while
# its mapping is gone but before anything else touches the directory.
atexit.register(_flush_pending_unlinks)


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

    Everywhere else, including macOS and Windows, the blob is staged as a real
    file in a private temp directory.  That copy is genuine disk I/O, so we
    check for room first and raise a readable error rather than an ``Errno 28``
    from the middle of a multi-gigabyte write.

    The staged file is written, closed and only then handed to llama.cpp. That
    ordering is not cosmetic on Windows: an open handle is exclusive enough to
    stop another process opening the same path, so a file still open when
    llama.cpp goes to map it would fail with a sharing violation. It also means
    cleanup cannot rely on ``delete=True`` — see :func:`_stage_blob`.
    """
    if sys.platform == "linux" and hasattr(os, "memfd_create"):
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
        path = _stage_blob(name, data)
        try:
            yield str(path)
        finally:
            _unstage_blob(path)


def _stage_blob(name: str, data: memoryview | bytes) -> Path:
    """Write *data* to a fresh file in the staging directory and close it.

    Returns the path. The caller owns cleanup via :func:`_unstage_blob`.
    """
    staging = assembly_temp_dir()
    needed = len(data)
    free = free_bytes_at(staging)
    if free and free < needed:
        raise StagingError(
            f"not enough space to stage {name}: need {needed / 1e9:.1f} GB, "
            f"only {free / 1e9:.1f} GB free in {staging}. "
            f"Point ASSEMBLY_TEMP_DIR at a larger volume."
        )
    # mkstemp rather than NamedTemporaryFile: it returns a closed-on-error path
    # we own outright, and nothing keeps the file open behind our back.
    try:
        fd, raw = tempfile.mkstemp(suffix=".gguf", prefix="blob-", dir=str(staging))
    except OSError as exc:
        raise StagingError(
            f"could not create a staging file in {staging}: {exc}. "
            f"Point ASSEMBLY_TEMP_DIR at a writable volume."
        ) from exc
    path = Path(raw)
    try:
        with os.fdopen(fd, "wb") as blob:
            _write_blob(blob, data)
            blob.flush()
            os.fsync(blob.fileno())
    except BaseException:
        # A partial blob is worse than none: llama.cpp would read a truncated
        # model and report a generic load failure.
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path


#: Identifies the archive a staged blob was cut from. Offset and size alone
#: are not unique: two different archives loaded in one server can easily hold
#: a model at the same offset with the same length. The fingerprint includes
#: the mtime so an edited archive never reuses the old bytes.
_ArchiveId = tuple[str, int, int]

#: A staged blob: which archive, and which region of it.
_StagedKey = tuple[_ArchiveId, int, int]

#: Staged blobs kept for reuse, keyed by _StagedKey. Ordered oldest-first, so
#: the entry to give up disk space for is the first. Linux never populates
#: this: there the blob is an anonymous memfd and never reaches the
#: filesystem, so there is nothing to reuse and nothing to spend.
_STAGED_BLOBS: dict[_StagedKey, Path] = {}

#: Bytes currently held by :data:`_STAGED_BLOBS`. Bounds the cache against the
#: one failure mode of keeping files around: filling the volume.
_STAGED_BYTES = 0

#: Evict the least recently used staged blobs past this many bytes. None, the
#: default, means keep them until the archive closes. Set it when the staging
#: volume is too small to hold a second copy of the archive.
STAGED_CACHE_MAX_BYTES: Optional[int] = None

_STAGED_LOCK = threading.Lock()


def _has_memfd() -> bool:
    """Whether blobs can be handed to llama.cpp without touching the disk."""
    return sys.platform == "linux" and hasattr(os, "memfd_create")


def staged_cache_bytes() -> int:
    """Bytes currently held by staged blobs kept for reuse."""
    with _STAGED_LOCK:
        return _STAGED_BYTES


def _remember_staged(key: _StagedKey, path: Path) -> None:
    """Record *path* as the reusable staging file for *key*."""
    global _STAGED_BYTES
    with _STAGED_LOCK:
        stale = _STAGED_BLOBS.pop(key, None)
        if stale is not None and stale != path:
            _evict_paths = [stale]
        else:
            _evict_paths = []
        _STAGED_BLOBS[key] = path
        try:
            _STAGED_BYTES += path.stat().st_size
        except OSError:
            pass
    for victim in _evict_paths:
        _unstage_blob(victim)


def _forget_staged(key: _StagedKey) -> Optional[Path]:
    """Drop *key* from the reuse cache and return the path it held."""
    global _STAGED_BYTES
    with _STAGED_LOCK:
        path = _STAGED_BLOBS.pop(key, None)
        if path is not None:
            try:
                _STAGED_BYTES -= path.stat().st_size
            except OSError:
                pass
        return path


def _evict_staged_to_fit(incoming: int) -> None:
    """Free staged blobs until *incoming* more bytes fit in the cache.

    Oldest first. Never evicts on Linux, where there is nothing cached.
    """
    global _STAGED_BYTES
    limit = STAGED_CACHE_MAX_BYTES
    if limit is None:
        return
    with _STAGED_LOCK:
        victims: list[Path] = []
        while _STAGED_BYTES + incoming > limit and _STAGED_BLOBS:
            key, victim = next(iter(_STAGED_BLOBS.items()))
            del _STAGED_BLOBS[key]
            try:
                _STAGED_BYTES -= victim.stat().st_size
            except OSError:
                pass
            victims.append(victim)
    for victim in victims:
        _unstage_blob(victim)


def clear_staged_cache(archive: Optional[_ArchiveId] = None) -> None:
    """Remove staged blobs kept for reuse.

    With *archive*, only that archive's blobs. Without it, all of them. Called
    when an archive closes; leaving the files behind would keep a second copy
    of every model on the staging volume for as long as the process ran.
    """
    with _STAGED_LOCK:
        keys = [k for k in _STAGED_BLOBS if archive is None or k[0] == archive]
    for key in keys:
        path = _forget_staged(key)
        if path is not None:
            _unstage_blob(path)


def _unstage_blob(path: Path) -> None:
    """Remove a staged blob, tolerating a platform that still holds it open.

    Windows keeps a mapped file locked for as long as the mapping lives, and
    llama.cpp holds the model mapping for the lifetime of the model, not just the
    load. So this cannot be unconditional: the POSIX case (unlinking an open file
    is allowed) frees the space immediately, and a failure is remembered for
    :func:`_flush_pending_unlinks` to retry once the mapping is released. The
    directory itself is never removed to achieve this, so a user-supplied
    ``--temp-dir`` ends up without our files and without losing any of theirs.
    """
    with _PENDING_UNLINK_LOCK:
        _PENDING_UNLINK.discard(path)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        _defer_unlink(path)


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


# ─────────────────────────────────────────────────────────────────────
# Backend seam
# ─────────────────────────────────────────────────────────────────────

class Backend:
    """The six operations Assembly needs from an inference runtime.

    Everything Assembly does to a model goes through this interface, so the
    llama.cpp binding is a detail rather than a dependency of the router, the
    archive layout and the server. Two reasons that matters: a test can supply
    a fake without a multi-gigabyte GGUF, and a second runtime can be dropped
    in without touching the code above it.

    Implementations must be safe to call from more than one thread. A model
    returned by :meth:`load` is owned by the caller and released with
    :meth:`release`.
    """

    def load(self, path: str, *, embedding: bool = False,
             n_ctx: int = 0, n_gpu_layers: int = 0,
             **kwargs: Any) -> Any:
        """Return a loaded model reading from the file at *path*.

        *kwargs* carries pass-through llama.cpp options (n_threads, n_batch,
        flash_attn and the like) so the constructor's ``**llama_kwargs``
        reach llama.cpp instead of being dropped on the floor here.
        """
        raise NotImplementedError

    def embed(self, model: Any, text: str) -> Any:
        """Return the embedding vector for *text*."""
        raise NotImplementedError

    def chat_completion(self, model: Any, messages: list[dict[str, Any]],
                        **kwargs: Any) -> Any:
        """Run a chat completion over *messages*."""
        raise NotImplementedError

    def completion(self, model: Any, prompt: str, **kwargs: Any) -> Any:
        """Run a raw completion over *prompt*."""
        raise NotImplementedError

    def tokenize(self, model: Any, text: str, **kwargs: Any) -> list[int]:
        """Return the token ids for *text*."""
        raise NotImplementedError

    def release(self, model: Any) -> None:
        """Drop *model* and everything it holds. Must tolerate a failure."""
        raise NotImplementedError


class LlamaCppBackend(Backend):
    """The production backend, backed by ``llama_cpp``.

    ``llama_cpp.Llama`` is imported lazily in :meth:`load`, so importing
    Assembly — to read a header, or to answer ``assembly --help`` — does not
    pull in a compiled extension the caller may not have installed.
    """

    def __init__(self, llm_class: Any = None) -> None:
        self._llm_class = llm_class

    def load(self, path: str, *, embedding: bool = False,
             n_ctx: int = 0, n_gpu_layers: int = 0,
             **kwargs: Any) -> Any:
        if self._llm_class is None:
            from llama_cpp import Llama
            self._llm_class = Llama
        return self._llm_class(
            model_path=path,
            embedding=embedding,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            **kwargs,
        )

    def embed(self, model: Any, text: str) -> Any:
        return model.embed(text)

    def chat_completion(self, model: Any, messages: list[dict[str, Any]],
                        **kwargs: Any) -> Any:
        return model.create_chat_completion(messages=messages, **kwargs)

    def completion(self, model: Any, prompt: str, **kwargs: Any) -> Any:
        return model.create_completion(prompt=prompt, **kwargs)

    def tokenize(self, model: Any, text: str, **kwargs: Any) -> list[int]:
        return model.tokenize(text.encode("utf-8"), **kwargs)

    def release(self, model: Any) -> None:
        try:
            del model
        except Exception:
            pass


class _ModelSlot:
    """A model resident in RAM, plus what is needed to load it again."""

    __slots__ = ("name", "llama", "offset", "size", "staged")

    def __init__(
        self,
        name: str,
        llama: Any,
        offset: int,
        size: int,
        staged: Optional[Path] = None,
    ) -> None:
        self.name = name
        self.llama = llama
        self.offset = offset
        self.size = size
        #: The staging file backing this model, kept so a later load of the
        #: same model can reuse it rather than copying the blob again. None on
        #: Linux, where the blob is handed over as an anonymous memfd and
        #: never touches the filesystem.
        self.staged = staged


class AssemblyLlama:
    """Intelligent multi-model wrapper around ``llama_cpp.Llama``.

    Parameters
    ----------
    asmb_path:
        Path to a packed ``.asmb`` archive.
    The context window is not configurable: it is read from the archive as the
    smallest context length declared by any of its execution models, so an
    assembly can never be sized for a model that cannot hold it.
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
        n_gpu_layers: int = 0,
        verbose: bool = False,
        temp_dir: Optional[str | Path] = None,
        max_loaded_models: Optional[int] = None,
        backend: Optional[Backend] = None,
        **llama_kwargs: Any,
    ) -> None:
        self._backend = backend if backend is not None else LlamaCppBackend()
        self._asmb_path = Path(asmb_path).expanduser().resolve()
        # Set from the archive in _setup(); there is deliberately no way to
        # override it, see _resolve_context_window.
        self._n_ctx = FALLBACK_CONTEXT
        self._model_contexts: dict[str, Optional[int]] = {}
        self._n_gpu_layers = n_gpu_layers
        self._verbose = verbose
        self._llama_kwargs = llama_kwargs
        #: How many models may sit in RAM at once. Beyond this the least
        #: recently used is evicted; its staging file survives, so coming back
        #: to it costs no disk copy.
        if max_loaded_models is None:
            max_loaded_models = DEFAULT_MAX_LOADED_MODELS
        elif max_loaded_models < 1:
            # `or` would quietly turn 0 into the default, and a negative cap
            # would make the eviction loop unload every model it sees.
            raise ValueError(
                f"max_loaded_models must be at least 1, got {max_loaded_models}"
            )
        self._max_loaded_models = max_loaded_models
        self._archive_mtime_ns = 0
        self._archive_size = 0

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
        #: In-flight request count per model. Kept apart from _lock so a
        #: long generation does not block the bookkeeping that decides whether
        #: a model may be unloaded.
        self._in_use_count: dict[int, int] = {}
        self._in_use_lock = threading.Lock()
        #: Monotonic clock for LRU order, so two loads of the same size still
        #: have a deterministic order.
        self._use_clock = 0
        self._last_used: dict[str, int] = {}
        self._last_summarized = False
        self._token_cache: dict[str, int] = {}
        self._last_prompt_tokens = 0
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
        stat = os.fstat(self._fd)
        file_size = stat.st_size
        # Fingerprint the archive so a staged blob from an older copy of this
        # file is never mistaken for one cut from the current bytes.
        self._archive_mtime_ns = stat.st_mtime_ns
        self._archive_size = file_size
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

        # ── Derive the context window from the archive ───────────────
        self._resolve_context_window()

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

    def _resolve_context_window(self) -> None:
        """Derive the context window from the archive's execution models.

        The window is whatever the *smallest* execution model in the archive
        declares in its GGUF metadata, so no request can be sized for a model
        that cannot hold it. The router's embedding model is excluded on
        purpose: it embeds single sentences, is typically built with a tiny
        window (2048 is common), and has nothing to do with how long a
        conversation may run.

        There is no override. A user-supplied window could exceed what a
        bundled model was trained for, which fails late and confusingly, or
        undershoot it, which throws away capability for no reason.
        """
        assert self._mm is not None and self._header is not None

        contexts: dict[str, Optional[int]] = {}
        for model in self._header.models:
            # Read little first and escalate only if the key is not there. A
            # too-short read does not fail loudly -- the model just looks like
            # it declared no context length, and the window silently collapses
            # to the fallback. Materialising HEADER_READ_BYTES per model up
            # front would cost hundreds of megabytes of pointless copies.
            found: Optional[int] = None
            for span in HEADER_READ_SPANS:
                head = bytes(self._mm[model.offset : model.offset + span])
                found = gguf_context_length_from_bytes(head)
                if found:
                    break
            contexts[model.name] = found
        self._model_contexts = contexts

        unknown = [name for name, value in contexts.items() if not value]
        known = [value for value in contexts.values() if value]

        if not known:
            self._n_ctx = FALLBACK_CONTEXT
            self._log(
                f"No model declares a context length; using {FALLBACK_CONTEXT}"
            )
            return

        window = min(known)

        if unknown:
            # One unknown model could be smaller than everything measured, and
            # guessing high would fail only once that model is routed to.
            self._n_ctx = FALLBACK_CONTEXT
            self._log(
                f"{', '.join(sorted(unknown))} declare no context length; "
                f"falling back to {FALLBACK_CONTEXT} rather than assume "
                f"{min(known)} is safe"
            )
            return

        if window < MIN_CONTEXT or window > MAX_CONTEXT:
            self._log(
                f"Declared context {window} is out of range; clamping to "
                f"{max(MIN_CONTEXT, min(MAX_CONTEXT, window))}"
            )
        self._n_ctx = max(MIN_CONTEXT, min(MAX_CONTEXT, window))

    @property
    def context_window(self) -> int:
        """The context window every execution model in this archive was
        loaded with."""
        return self._n_ctx

    @property
    def model_contexts(self) -> dict[str, Optional[int]]:
        """Per-model declared context lengths, for diagnostics."""
        return dict(self._model_contexts)

    def _load_llama_from_mmap(
        self,
        name: str,
        offset: int,
        size: int,
        embedding: bool = False,
        n_ctx: Optional[int] = None,
    ) -> Any:
        """Load a ``llama_cpp.Llama`` from the mmap'd .asmb region.

        llama.cpp needs a filesystem path. On Linux the blob is handed over as
        an anonymous in-memory fd, which costs nothing to hold and so is
        created fresh each time. Everywhere else it is staged to a real file,
        and that copy is kept so loading the same model again is instant
        rather than another multi-gigabyte write.
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
            bytes(model_view[:HEADER_READ_BYTES])
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
            if _has_memfd():
                with _memfd_model(name, model_view) as model_path:
                    return self._backend.load(
                        model_path,
                        embedding=embedding,
                        n_ctx=n_ctx or self._n_ctx,
                        n_gpu_layers=self._n_gpu_layers,
                        **self._llama_kwargs,
                    )
            return self._load_llama_from_staging(
                name, model_view, offset, size, embedding, n_ctx
            )
        finally:
            # Release the mmap export explicitly. If the backend raised, the
            # traceback keeps this frame alive, and an outstanding export
            # would make mmap.close() raise BufferError and skip the
            # os.close() below.
            model_view.release()

    def _load_llama_from_staging(
        self,
        name: str,
        model_view: memoryview,
        offset: int,
        size: int,
        embedding: bool = False,
        n_ctx: Optional[int] = None,
    ) -> Any:
        """Load a model from a staging file, reusing an earlier copy if we have one."""
        key = self._staged_key(offset, size)
        staged = self._reuse_staged(key, size)
        if staged is None:
            _evict_staged_to_fit(size)
            staged = _stage_blob(name, model_view)
            try:
                _remember_staged(key, staged)
            except OSError:
                _unstage_blob(staged)
                raise
        try:
            return self._backend.load(
                str(staged),
                embedding=embedding,
                n_ctx=n_ctx or self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                **self._llama_kwargs,
            )
        except Exception:
            # A staging file we cannot load from is worse than none: the next
            # attempt would reuse it and fail the same way. Forget it.
            self._forget_staged_path(key)
            raise

    def _staged_key(self, offset: int, size: int) -> _StagedKey:
        """Identify a blob within the archive currently open."""
        return (
            (str(self._asmb_path), self._archive_mtime_ns, self._archive_size),
            offset,
            size,
        )

    def _reuse_staged(self, key: _StagedKey, size: int) -> Optional[Path]:
        """Return a usable staging file for *key*, or None to stage a new one."""
        with _STAGED_LOCK:
            path = _STAGED_BLOBS.get(key)
            if path is None:
                return None
            try:
                # A truncated or swapped file would load as a corrupt model.
                if path.stat().st_size != size:
                    del _STAGED_BLOBS[key]
                    return None
            except OSError:
                del _STAGED_BLOBS[key]
                return None
            # Re-insert so this is the most recently used entry, not the next
            # one evicted when the cache is over its byte limit.
            del _STAGED_BLOBS[key]
            _STAGED_BLOBS[key] = path
            return path

    def _forget_staged_path(self, key: _StagedKey) -> None:
        path = _forget_staged(key)
        if path is not None:
            _unstage_blob(path)

    def _ensure_model(self, route: RouteResult) -> _ModelSlot:
        """Ensure the execution model for *route* is loaded and cached.

        Serialised: two threads routing to the same cold model would
        otherwise both stage and load a multi-gigabyte copy.
        """
        with self._lock:
            if route.model_name in self._model_cache:
                slot = self._model_cache[route.model_name]
                self._active_slot = slot
                self._last_used[route.model_name] = self._tick()
                return slot

            entry = self._header.models[route.model_index]
            self._evict_if_needed(entry.size)
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
                staged=self._staged_for(entry.offset, entry.size),
            )
            self._model_cache[entry.name] = slot
            self._active_slot = slot
            self._last_used[entry.name] = self._tick()
            return slot

    def _tick(self) -> int:
        self._use_clock += 1
        return self._use_clock

    # ------------------------------------------------------------------
    # Resident-model cache
    # ------------------------------------------------------------------

    def _staged_for(self, offset: int, size: int) -> Optional[Path]:
        """The staging file for this blob, if one is being kept for reuse."""
        key = self._staged_key(offset, size)
        with _STAGED_LOCK:
            return _STAGED_BLOBS.get(key)

    def _in_use(self, slot: _ModelSlot) -> bool:
        """Whether a request is currently running on *slot*.

        Unloading a model out from under a streaming response would truncate
        the answer mid-sentence, so an in-flight slot is never evicted.
        """
        with self._in_use_lock:
            return self._in_use_count.get(id(slot.llama), 0) > 0

    def _acquire(self, slot: _ModelSlot) -> None:
        with self._in_use_lock:
            self._in_use_count[id(slot.llama)] = (
                self._in_use_count.get(id(slot.llama), 0) + 1
            )

    def _release(self, slot: _ModelSlot) -> None:
        with self._in_use_lock:
            key = id(slot.llama)
            remaining = self._in_use_count.get(key, 0) - 1
            if remaining > 0:
                self._in_use_count[key] = remaining
            else:
                self._in_use_count.pop(key, None)

    def _guard_stream(self, slot: _ModelSlot, stream: Any) -> Any:
        """Keep *slot* marked busy for as long as *stream* is being consumed.

        A streaming response is a generator, so the lock around the call only
        covers its creation — the lock is gone by the time the caller pulls the
        first token. Without this the model would look idle for the whole
        stream and eviction could pull it out mid-answer.
        """
        if not hasattr(stream, "__iter__") or isinstance(stream, (dict, list, str, bytes)):
            return stream
        self._acquire(slot)

        def _iterate() -> Iterator[Any]:
            try:
                yield from stream
            finally:
                # Runs on exhaustion, on exception, and on close(), which is
                # what a client disconnect looks like from here.
                self._release(slot)

        return _iterate()

    def _evict_if_needed(self, incoming: int = 0) -> None:
        """Free RAM until there is room for one more model.

        Least recently used first, and never a model with a request running on
        it. The staging file is left alone: that is the whole point of keeping
        it, so a later request for this model is a memory-map rather than
        another multi-gigabyte copy.

        Caller must hold :attr:`_lock`.
        """
        while len(self._model_cache) >= self._max_loaded_models:
            victim = self._least_recently_used()
            if victim is None:
                # Everything resident is mid-request. Refusing here is correct:
                # evicting anyway would truncate somebody's answer, and
                # exceeding the cap beats corrupting a response.
                self._log(
                    f"all {len(self._model_cache)} resident models are busy; "
                    f"loading beyond max_loaded_models={self._max_loaded_models} "
                    f"until one finishes"
                )
                return
            self._unload(victim)

    def _least_recently_used(self) -> Optional[_ModelSlot]:
        """The resident model to give up first, or None if none are free."""
        candidates = [s for s in self._model_cache.values() if not self._in_use(s)]
        if not candidates:
            return None
        victim = min(
            candidates,
            key=lambda s: (self._last_used.get(s.name, 0), s.size),
        )
        return victim

    def _unload(self, slot: _ModelSlot) -> None:
        """Drop *slot* from RAM, keeping its staging file for reuse."""
        self._model_cache.pop(slot.name, None)
        if self._active_slot is slot:
            self._active_slot = None
        self._last_used.pop(slot.name, None)
        try:
            self._backend.release(slot.llama)
        except Exception:
            pass
        slot.llama = None
        self._log(f"Unloaded '{slot.name}' (staging copy kept for reuse)")

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _embed_prompt(self, text: str) -> np.ndarray:
        """Compute the embedding vector for *text* using the router model."""
        emb = self._backend.embed(self._embedding_llama, text)
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
                resp = self._backend.chat_completion(slot.llama, msgs, **kwargs)
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
            return self._backend.completion(slot.llama, prompt, **kwargs)

    def create_completion(self, prompt: str, **kwargs: Any) -> Any:
        """``Llama.create_completion`` equivalent with automatic routing."""
        slot = self._route_and_load(prompt)
        with self._lock:
            return self._backend.completion(slot.llama, prompt, **kwargs)

    # ------------------------------------------------------------------
    # Keeping a conversation inside the context window
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Approximate the token count of *text* with the loaded tokenizer.

        Results are memoised on the exact text. The same turns get counted on
        every request, so without this the history is tokenised over and over:
        slow at first, and slower with every turn added.

        Falls back to a character-based estimate when no model is loaded yet,
        because this is only ever used to decide *whether* to summarise, and a
        rough answer is better than making every call wait on a model load.
        """
        cached = self._token_cache.get(text)
        if cached is not None:
            return cached

        count = None
        if self._active_slot is not None:
            try:
                with self._lock:
                    count = len(
                        self._backend.tokenize(
                            self._active_slot.llama, text,
                            add_bos=False, special=False
                        )
                    )
            except Exception:
                count = None
        if count is None:
            count = max(1, len(text) // 4)

        if len(self._token_cache) >= TOKEN_CACHE_ENTRIES:
            # A conversation only ever grows, so the oldest entries are the
            # least likely to be needed again. Dropping the lot is simpler
            # than ordering them, and costs one re-tokenisation.
            self._token_cache.clear()
        self._token_cache[text] = count
        return count

    def _conversation_tokens(self, messages: "list[dict[str, str]]") -> int:
        """Tokens the templated prompt would need, including chat overhead.

        Counted message by message rather than as one joined blob so the cache
        can reuse each turn's result. Re-joining and re-tokenising the whole
        conversation made every request cost more than the last, which is
        exactly when there is least appetite for it.
        """
        total = TEMPLATE_OVERHEAD
        for message in messages:
            content = message.get("content")
            if not content:
                continue
            total += self.count_tokens(message.get("role", "")) + 1
            total += self.count_tokens(content) + 1
        return total

    _SUMMARY_SYSTEM = (
        "You compress conversations into running notes for a colleague with no "
        "access to the original.\n\n"
        "Copy through verbatim every name, identifier, key, number, date, "
        "path, command, decision and commitment that appears. These are the "
        "whole point: a later question about the start of the conversation can "
        "only be answered if they are still here.\n\n"
        "Discard small talk, repetition and padding. Never describe the "
        "conversation's shape or comment on the transcript -- just record what "
        "was said and decided.\n\n"
        "Answer with the notes only. No preamble, no headings, no questions, "
        "and no closing remarks."
    )

    @staticmethod
    def _condense(text: str) -> str:
        """Collapse runaway repetition within one message.

        Pasted logs, spinner frames and repeated filler can turn a single turn
        into thousands of near-identical words. Left in, they crowd out the
        handful of specifics the summary exists to preserve, and they make the
        summarisation request itself slow. A message built from a tiny
        vocabulary repeated over and over is reported once instead.
        """
        words = text.split()
        if len(words) > 24 and len(set(words)) <= 4:
            return " ".join(words[:8]) + f" (repeated {len(words)} times)"
        return text

    def _summarize_once(
        self, slot: "_ModelSlot", transcript: str, max_output: int
    ) -> str:
        """Ask the model to compress *transcript* into running notes."""
        prompt = [
            {"role": "system", "content": self._SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": f"Summarise this conversation:\n\n{transcript}",
            },
        ]
        result = self._backend.chat_completion(
            slot.llama,
            prompt,
            max_tokens=max_output,
            temperature=0.2,
            top_p=0.9,
            stream=False,
        )
        if not isinstance(result, dict) or not result.get("choices"):
            return ""
        return (result["choices"][0].get("message", {}).get("content") or "").strip()

    def _split_to_budget(self, transcript: str, max_tokens: int) -> "list[str]":
        """Split *transcript* into chunks that each fit *max_tokens*.

        Chunking is what keeps the *beginning* of a conversation intact. The
        naive alternative -- trimming an over-long transcript from the front --
        silently discards the oldest facts, which are usually the ones being
        asked about.
        """
        limit = max(64, max_tokens)
        chunks: list[str] = []
        current: list[str] = []
        used = 0
        for line in transcript.splitlines():
            if not line.strip():
                continue
            cost = self.count_tokens(line) + 1
            if current and used + cost > limit:
                chunks.append("\n".join(current))
                current, used = [], 0
            current.append(line)
            used += cost
        if current:
            chunks.append("\n".join(current))
        return chunks or [transcript]

    def _summarize_transcript(
        self, slot: "_ModelSlot", transcript: str, max_tokens: int
    ) -> str:
        """Compress *transcript* into notes that fit within *max_tokens*.

        Longer conversations are compressed in stages: each chunk that fits is
        summarised on its own, and the resulting notes are themselves
        summarised until they fit in a single note. The number of stages is
        bounded so a pathological conversation cannot loop.
        """
        chunks = self._split_to_budget(transcript, max_tokens)
        if len(chunks) == 1:
            return self._summarize_once(
                slot,
                chunks[0],
                max_output=min(SUMMARY_OUTPUT_CEILING, max_tokens),
            )

        per_chunk = max(
            96, min(SUMMARY_OUTPUT_CEILING, max_tokens // len(chunks))
        )
        notes = [
            self._summarize_once(slot, chunk, max_output=per_chunk)
            for chunk in chunks
        ]
        notes = [n for n in notes if n]
        if not notes:
            return ""

        for _ in range(3):
            combined = "\n".join(
                f"[{i}] {n}" for i, n in enumerate(notes, 1)
            )
            if self.count_tokens(combined) <= max_tokens:
                return combined
            folded = self._summarize_once(
                slot,
                combined,
                max_output=min(SUMMARY_OUTPUT_CEILING, max_tokens),
            )
            if not folded:
                return combined
            notes = [folded]

        return "\n".join(notes)

    def summarize_messages(
        self, messages: "list[dict[str, str]]", slot: "_ModelSlot"
    ) -> "list[dict[str, str]]":
        """Compress *messages* so the conversation fits the context window.

        Called when a prompt approaches the limit. Rather than failing, or
        silently dropping the oldest turns, the model is asked to summarise them
        itself and the summary takes their place, so the thread of the
        conversation stays available to the reply.

        Summarisation is best effort: if any part of it fails the original
        messages are returned unchanged, so generation can still proceed rather
        than failing for a secondary reason.
        """
        if len(messages) < 3:
            return messages

        # The recent turns are what the reply has to react to, so they are kept
        # verbatim; everything older becomes the summary's subject.
        budget = int(self._n_ctx * SUMMARIZED_BUDGET)
        kept: list[dict[str, str]] = []
        for msg in reversed(messages):
            if kept and self._conversation_tokens(kept + [msg]) > budget:
                break
            kept.append(msg)
        kept.reverse()

        if len(kept) >= len(messages):
            return messages
        oldest = messages[: len(messages) - len(kept)]
        if not oldest:
            return messages

        transcript = "\n\n".join(
            f"[{m.get('role', 'user')}] "
            + self._condense((m.get("content") or "").strip())
            for m in oldest
            if (m.get("content") or "").strip()
        )
        if not transcript.strip():
            return messages

        try:
            # Leave room in the request for the system prompt and the reply.
            summary_budget = max(
                128, int(self._n_ctx * SUMMARIZE_AT) - TEMPLATE_OVERHEAD - 512
            )
            summary = self._summarize_transcript(
                slot, transcript, max_tokens=summary_budget
            )
        except Exception as exc:
            self._log(f"Could not summarise the conversation ({exc}); keeping it")
            return messages

        if not summary:
            self._log("Summarisation came back empty; keeping the conversation")
            return messages

        note = {
            "role": "system",
            "content": (
                "Summary of the earlier conversation, which was compressed to "
                f"stay inside the {self._n_ctx}-token context window. Treat it "
                "as what was already established:\n\n" + summary
            ),
        }

        # The summary is itself text, so the tail may now have to give way to
        # it. Trimming from the front of the tail is right here: the newest
        # turns are the ones that must survive.
        fitted = [note] + kept
        limit = int(self._n_ctx * SUMMARIZE_AT)
        while len(fitted) > 2 and self._conversation_tokens(fitted) > limit:
            fitted.pop(1)

        self._log(
            f"Conversation near the {self._n_ctx}-token limit: summarised "
            f"{len(oldest)} earlier message(s) into {len(summary)} characters"
        )
        return fitted

    def _fit_to_context(
        self, messages: "list[dict[str, str]]", slot: "_ModelSlot"
    ) -> "tuple[list[dict[str, str]], bool]":
        """Summarise *messages* if they no longer comfortably fit the window.

        Returns the messages to send and whether anything was compressed, so
        callers can report it without re-deriving it from lengths.
        """
        limit = int(self._n_ctx * SUMMARIZE_AT)
        if self._conversation_tokens(messages) <= limit:
            return messages, False
        fitted = self.summarize_messages(messages, slot)
        return fitted, fitted is not messages

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
        effective_messages, self._last_summarized = self._fit_to_context(
            effective_messages, slot
        )
        # Counted after fitting: this is what the model was actually given,
        # which is the only figure worth reporting to a caller.
        self._last_prompt_tokens = self._conversation_tokens(effective_messages)
        with self._lock:
            self._acquire(slot)
            try:
                result = self._backend.chat_completion(
                    slot.llama, effective_messages, **kwargs
                )
            except BaseException:
                self._release(slot)
                raise
        if kwargs.get("stream"):
            # The lock is released above; the stream keeps the slot pinned
            # instead. See _guard_stream.
            return self._guard_stream(slot, result)
        self._release(slot)
        return result

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
        with self._lock:
            for name in list(self._model_cache):
                slot = self._model_cache.pop(name, None)
                if slot is not None:
                    try:
                        self._backend.release(slot.llama)
                    except Exception:
                        pass
                    slot.llama = None
            self._active_slot = None
            self._last_used.clear()

        self._in_use_count.clear()

        if self._embedding_llama is not None:
            try:
                self._backend.release(self._embedding_llama)
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

        # Staged copies belong to this archive alone; another archive loaded in
        # the same server keeps its own.
        clear_staged_cache(
            (str(self._asmb_path), self._archive_mtime_ns, self._archive_size)
            if self._archive_mtime_ns else None
        )

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
