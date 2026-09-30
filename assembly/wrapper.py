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
        **llama_kwargs: Any,
    ) -> None:
        from llama_cpp import Llama

        self._Llama = Llama
        self._asmb_path = Path(asmb_path).expanduser().resolve()
        # Set from the archive in _setup(); there is deliberately no way to
        # override it, see _resolve_context_window.
        self._n_ctx = FALLBACK_CONTEXT
        self._model_contexts: dict[str, Optional[int]] = {}
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
                        self._active_slot.llama.tokenize(
                            text.encode("utf-8"), add_bos=False, special=False
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
        result = slot.llama.create_chat_completion(
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
