"""
Assembly Engine — HTTP Server Daemon.

Provides a multi-device REST API compatible with:
- OpenAI API (/v1/chat/completions, /v1/completions, /v1/models)
- Ollama-style API (/api/generate, /api/chat)
- Assembly native API (/api/route, /api/info, /api/load, /health)

Allows scripts, external programs, and devices across LAN/WAN
to interact with locally hosted .asmb assemblies.

There is deliberately no web UI: the terminal is the interface, and every
endpoint here is a plain JSON API.

Security posture
----------------
The server holds no secrets but *does* spend CPU/GPU per request and can
read local files by path, so it is hardened by default:

- Binds to loopback unless told otherwise.
- Optional bearer-token auth (``--api-key`` / ``ASSEMBLY_API_KEY``).
- Request bodies are size-capped and sockets are time-limited.
- ``/api/load`` is jailed to ``--models-dir`` and reports errors without
  leaking filesystem detail.
- CORS is same-origin only unless an explicit origin is configured.
"""

from __future__ import annotations

import json
import logging
import re
import os
import secrets
import socket
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from assembly import __version__, ui
from assembly.config import SamplingParams, sampling_params_from_body
from assembly.wrapper import AssemblyLlama

logger = logging.getLogger("assembly.server")

#: Largest request body we will read, in bytes.
MAX_BODY_BYTES = 1 << 20  # 1 MiB

#: Seconds a client may hold a connection open.
SOCKET_TIMEOUT = 300

#: How many archives a single server will hold open at once. Loads in flight
#: count against this, not just the ones already resident.
MAX_LOADED_MODELS = 8

#: Longest accepted client-supplied model alias.
MAX_ALIAS_LENGTH = 128

#: How many connections may be handled at once. ThreadingHTTPServer spawns a
#: thread per connection with no ceiling, so without this a few thousand idle
#: sockets is enough to exhaust the process.
MAX_CONCURRENT_CONNECTIONS = 32

#: Hostnames that mean "this machine only".
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

#: Guards the accept-path counter, which is touched from every worker thread.
_CONNECTION_COUNT_LOCK = threading.Lock()


def is_loopback(host: str) -> bool:
    """True when binding *host* keeps the service on this machine only."""
    return host in LOOPBACK_HOSTS


class _BadRequest(Exception):
    """Malformed request from the client (400)."""


class _BodyTooLarge(Exception):
    """Request body exceeded :data:`MAX_BODY_BYTES` (413)."""


def _sanitize_for_log(text: str, limit: int = 200) -> str:
    """Make an attacker-controlled string safe to write to a terminal.

    The request line and any client-supplied name are echoed into the operator's
    log verbatim by ``http.server``, and terminal escapes in that text are not
    text: ``\\x1b[2J`` clears the screen and ``\\x1b[31m`` forges a colour, so a
    remote client could otherwise rewrite what the operator believes they are
    looking at. Newlines would additionally forge whole log lines.
    """
    cleaned = "".join(
        ch if ch.isprintable() and ch not in "\r\n" else "?" for ch in text
    )
    return cleaned[:limit] + ("..." if len(text) > limit else "")


class ModelManager:
    """Thread-safe manager for loaded .asmb archives."""

    def __init__(
        self,
        default_model_path: Optional[str | Path] = None,
        persona: Optional[str] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        models_dir: Optional[str | Path] = None,
        temp_dir: Optional[str | Path] = None,
        max_models: int = MAX_LOADED_MODELS,
    ) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, AssemblyLlama] = {}
        #: Aliases with a load in flight, counted rather than merely present. A
        #: reservation is taken under the lock before the (multi-second) archive
        #: load and released after, so a burst of concurrent requests cannot each
        #: pass the capacity check and then all materialise a model. Counting per
        #: alias matters: with a set, two simultaneous loads of the *same* alias
        #: share one reservation, so the second one silently drops the first
        #: model on the floor and both copies of its weights stay resident.
        self._loading: dict[str, int] = {}
        self._loading_total = 0
        self._active_alias: Optional[str] = None
        self._persona = persona
        self._n_gpu_layers = n_gpu_layers
        self._verbose = verbose
        self._max_models = max_models
        self._temp_dir = temp_dir
        self._models_dir = (
            Path(models_dir).expanduser().resolve() if models_dir else None
        )

        if default_model_path:
            self.load_model(default_model_path, alias="default", set_active=True)

    @staticmethod
    def _sanitize_alias(alias: Any) -> str:
        """Validate a client-supplied model alias.

        The alias becomes a dict key, appears in the routing decision and in the
        log, so it has to be a bounded, single-line string. Without this a caller
        can pass a list (a TypeError from ``Path``), an unhashable dict (a
        TypeError inside the manager lock), or a multi-kilobyte string.
        """
        if not isinstance(alias, str):
            raise _BadRequest("alias must be a string")
        alias = alias.strip()
        if not alias:
            raise _BadRequest("alias must not be empty")
        if len(alias) > MAX_ALIAS_LENGTH:
            raise _BadRequest(
                f"alias too long (limit {MAX_ALIAS_LENGTH} characters)"
            )
        if any(ch in alias for ch in "\r\n\t"):
            raise _BadRequest("alias must not contain control characters")
        return alias

    def _authorize_path(self, model_path: str | Path) -> Path:
        """Resolve *model_path* and confirm the server is allowed to open it.

        When ``models_dir`` is configured the path must live inside it, so a
        remote caller cannot use this endpoint to probe the filesystem. The
        ``.asmb`` suffix and existence checks are deliberately boring and
        uniform: the error strings must not distinguish "no such file" from
        "not an archive", or this becomes a file-existence oracle.
        """
        # A JSON body can carry a list or dict here, and Path() turns that into a
        # TypeError deep inside the load. str and Path are what every caller
        # legitimately passes; anything else is rejected at the boundary.
        if not isinstance(model_path, (str, Path)):
            raise _BadRequest("model_path must be a string")
        path = Path(model_path).expanduser().resolve()

        if self._models_dir is not None and not path.is_relative_to(self._models_dir):
            raise PermissionError("model is outside the allowed models directory")
        if path.suffix != ".asmb":
            raise ValueError("not an .asmb archive")
        if not path.is_file():
            raise FileNotFoundError("model not found")
        return path

    def load_model(
        self,
        model_path: str | Path,
        alias: Optional[str] = None,
        persona: Optional[str] = None,
        set_active: bool = True,
    ) -> str:
        """Load an .asmb model and return its alias.

        The (slow) archive load happens outside the manager lock so that
        health checks and other requests keep flowing while it runs.
        """
        path = self._authorize_path(model_path)
        name = self._sanitize_alias(alias) if alias else path.stem

        # The reservation has to cover the load, not just the check. Checking
        # capacity and then loading outside the lock lets every concurrent
        # request observe an empty slot and pass, so N simultaneous /api/load
        # calls all succeed and all stay resident -- which is how a client with
        # one request per model turns an 8-model budget into a disk-full.
        with self._lock:
            if name not in self._models:
                if len(self._models) + self._loading_total >= self._max_models:
                    raise RuntimeError(
                        f"too many models loaded or loading "
                        f"(limit {self._max_models})"
                    )
            self._loading[name] = self._loading.get(name, 0) + 1
            self._loading_total += 1

        try:
            # Constructed outside the lock: this takes seconds to minutes.
            llm = AssemblyLlama(
                asmb_path=path,
                persona=persona or self._persona,
                n_gpu_layers=self._n_gpu_layers,
                verbose=self._verbose,
                temp_dir=self._temp_dir,
            )
        except BaseException:
            with self._lock:
                self._release_reservation(name)
            raise

        with self._lock:
            self._release_reservation(name)
            previous = self._models.get(name)
            self._models[name] = llm
            if set_active or self._active_alias is None:
                self._active_alias = name

        if previous is not None:
            previous.close()

        logger.info(
            "Loaded assembly model %s from %s",
            _sanitize_for_log(name, 120),
            _sanitize_for_log(str(path)),
        )
        return name

    def _release_reservation(self, name: str) -> None:
        """Give back one load slot. The manager lock must be held."""
        remaining = self._loading.get(name, 0) - 1
        if remaining > 0:
            self._loading[name] = remaining
        else:
            self._loading.pop(name, None)
        self._loading_total = max(0, self._loading_total - 1)

    def get_active(self, model_name: Optional[str] = None) -> tuple[str, AssemblyLlama]:
        """Return (alias, AssemblyLlama) for requested or active model."""
        with self._lock:
            if not self._models:
                raise RuntimeError(
                    "No models currently loaded. Use /api/load or start server with -m."
                )

            if model_name and model_name in self._models:
                return model_name, self._models[model_name]

            # If default/active
            if self._active_alias and self._active_alias in self._models:
                return self._active_alias, self._models[self._active_alias]

            # Fallback to first available
            first_key = next(iter(self._models.keys()))
            return first_key, self._models[first_key]

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            result = []
            for alias, llm in self._models.items():
                result.append({
                    "id": alias,
                    "assembly_name": llm.header.assembly_name,
                    "persona": llm.persona,
                    "active": alias == self._active_alias,
                    "models": [m.name for m in llm.header.models],
                })
            return result

    def describe_models(self) -> str:
        """Human-readable list of the sub-models that are loaded.

        Kept next to :meth:`list_models` because the shape of that dict is the
        single source of truth; reading it here rather than in the caller keeps
        a wrong key from taking the server down while it prints its banner.
        """
        names: list[str] = []
        for entry in self.list_models():
            for name in entry.get("models") or []:
                if name not in names:
                    names.append(name)
        return ", ".join(names) or "none loaded"

    def unload_model(self, alias: str) -> list[str]:
        """Close and forget one loaded archive, freeing its slot.

        Without this the capacity limit is a one-way door: once `max_models`
        archives are resident the only recovery is a restart. Refusing a load
        with "unload one first" while offering no way to unload is not a limit,
        it is an outage waiting for an operator who has to guess.

        Returns the aliases still loaded.
        """
        name = self._sanitize_alias(alias)
        with self._lock:
            llm = self._models.pop(name, None)
            if llm is None:
                if self._loading.get(name):
                    raise _BadRequest("that model is still loading")
                raise _BadRequest("no such loaded model")
            if self._active_alias == name:
                self._active_alias = next(iter(self._models), None)
            remaining = list(self._models)

        # Closed outside the lock: releasing a multi-gigabyte archive can take
        # a while, and no other request needs to wait for that.
        llm.close()
        logger.info("Unloaded assembly model %s", _sanitize_for_log(name, 120))
        return remaining

    def context_window(self) -> Optional[int]:
        """The context window actually in use, across every loaded archive.

        Read from each archive rather than recomputed from its models. An
        archive whose models do not all declare a length falls back to a
        conservative default, and taking the minimum of the declared lengths
        instead would quietly report the optimistic number: overstating the
        window by 64x invites a client to send prompts that cannot fit.
        """
        with self._lock:
            windows = [
                value
                for llm in self._models.values()
                for value in (self._declared_window(llm),)
                if value
            ]
        return min(windows) if windows else None

    @staticmethod
    def _declared_window(llm: Any) -> Optional[int]:
        """The window one archive resolved to, or None if it has none."""
        window = getattr(llm, "context_window", None)
        if isinstance(window, int) and window > 0:
            return window
        # A stand-in without a resolved window: fall back to its models.
        contexts = getattr(llm, "model_contexts", None) or {}
        known = [v for v in contexts.values() if v]
        return min(known) if known else None

    def describe_context(self) -> str:
        """The context window in use, and which model decided it."""
        windows: dict[str, Optional[int]] = {}
        for alias, llm in self._models.items():
            contexts = getattr(llm, "model_contexts", None) or {}
            for name, value in contexts.items():
                windows[name] = value
        if not windows:
            return "none loaded"

        window = self.context_window()
        if window is None:
            return "unknown (no model declares one)"

        smallest = sorted(n for n, v in windows.items() if v == window)
        suffix = f"  (smallest: {', '.join(smallest)})" if smallest else ""
        undeclared = sorted(n for n, v in windows.items() if not v)
        if undeclared:
            # Be explicit rather than letting a conservative default look like
            # a real reading: the number is a fallback, not a measurement.
            suffix += (
                f"  (fallback: {', '.join(undeclared)} declare none)"
            )
        return f"{window:,} tokens{suffix}"

    def close(self) -> None:
        with self._lock:
            for llm in self._models.values():
                try:
                    llm.close()
                except Exception:
                    pass
            self._models.clear()


def _note_chunk(model_label: str, note: dict[str, Any]) -> dict[str, Any]:
    """A final SSE frame carrying only context metadata.

    Shaped like any other chunk so a client that walks deltas does not trip
    over an unfamiliar frame, and carrying no choices so it cannot be
    mistaken for content.
    """
    return {
        "id": "assembly-context",
        "object": "chat.completion.chunk",
        "model": model_label,
        "assembly_context": note,
        "choices": [],
    }


def _context_overflow(exc: Exception) -> str | None:
    """Turn llama.cpp's context-window error into something actionable.

    llama.cpp raises a bare ``ValueError`` when the prompt does not fit. The
    raw text ("Requested tokens (2082) exceed context window of 2048") is
    accurate but leaves the reader guessing what to do, so it is restated with
    the two remedies we actually support.
    """
    match = re.search(
        r"Requested tokens \((\d+)\) exceed context window of (\d+)",
        str(exc),
    )
    if not match:
        return None
    requested, limit = int(match.group(1)), int(match.group(2))
    return (
        f"Conversation is too long for the context window: {requested} tokens "
        f"against a limit of {limit}. Start a new conversation, shorten the "
        f"history, or restart the server with a larger window "
        f"(e.g. --n-ctx {max(limit * 2, 4096)})."
    )


class AssemblyRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for Assembly Engine API."""

    server: "AssemblyServer"  # Type hint for server instance

    # Per-recv socket timeout. Alone this is not a request deadline: a client
    # that dribbles one byte every few minutes keeps every recv() returning
    # before it expires and holds the connection open indefinitely at a cost of
    # a few bytes. setup() adds a wall-clock watchdog to close the gap.
    timeout = SOCKET_TIMEOUT

    server_version = "AssemblyEngine"
    sys_version = ""

    def finish(self) -> None:
        try:
            super().finish()
        finally:
            self.server.forget_connection(self)

    def setup(self) -> None:
        super().setup()
        # Register with the server's single watchdog so the connection has a
        # wall-clock deadline. One shared sweeper thread, not one Timer per
        # connection: a client holding 32 connections would otherwise double
        # the thread count.
        self.server.register_connection(self)

    def _expire_connection(self) -> None:
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except Exception:
            # The client may already be gone; nothing here is worth a traceback
            # from the sweeper thread.
            pass

    # ── Auth ─────────────────────────────────────────────────────────

    def _host_is_local(self) -> bool:
        """Whether the Host header names this machine.

        CORS is not a defence against DNS rebinding: an attacker's page at
        evil.com resolves to 127.0.0.1, so the browser treats requests to the
        server as *same-origin* and never applies CORS at all. The browser does
        still send the Host header, though, and for a loopback-bound server a
        rebinding request carries the attacker's hostname. Rejecting a Host
        that is not loopback-shaped closes that hole.

        Only enforced when the listener is bound to loopback, where anything
        else is by definition not how a legitimate client would address us.
        """
        if not self.server.require_local_host:
            return True
        host = (self.headers.get("Host") or "").strip()
        if not host:
            return False
        # Strip an optional port, being careful with bracketed IPv6 literals.
        if host.startswith("["):
            authority = host[1:].partition("]")[0]
        else:
            authority = host.rpartition(":")[0] if host.count(":") == 1 else host
        return authority in LOOPBACK_HOSTS or authority == "::1"

    def _authorized(self) -> bool:
        """Check the bearer token when the server was started with one."""
        expected = self.server.api_key
        if not expected:
            return True
        header = self.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return False
        token = token.strip()
        # compare_digest rejects non-ASCII str operands with TypeError, and a
        # token is attacker-controlled: any Latin-1 character in it would
        # otherwise raise before the handler's try block, killing the connection
        # with a traceback on stderr. Both sides go out as bytes so the token is
        # compared regardless of what is in it.
        try:
            supplied = token.encode("utf-8")
            wanted = expected.encode("utf-8")
        except UnicodeError:
            return False
        return secrets.compare_digest(supplied, wanted)

    def _deny(self) -> None:
        logger.warning(
            "Rejected unauthenticated request: %s %s from %s",
            self.command,
            _sanitize_for_log(self.path),
            self.address_string(),
        )
        self._send_json(
            {"error": {"message": "unauthorized", "code": HTTPStatus.UNAUTHORIZED.value}},
            status=HTTPStatus.UNAUTHORIZED,
            extra_headers=[("WWW-Authenticate", "Bearer")],
        )

    def _reject_foreign_host(self) -> None:
        logger.warning(
            "Rejected request with non-local Host %r from %s",
            _sanitize_for_log(self.headers.get("Host") or "", 80),
            self.address_string(),
        )
        self._send_json(
            {"error": {"message": "unauthorized",
                       "code": HTTPStatus.UNAUTHORIZED.value}},
            status=HTTPStatus.UNAUTHORIZED,
        )

    def _send_cors_headers(self) -> None:
        """Emit CORS headers only for an explicitly allowed origin.

        Wildcarding this on a state-changing local service would let any web
        page the operator visits drive it, so it is opt-in.
        """
        origin = self.headers.get("Origin")
        allowed = self.server.allow_origin
        if (
            allowed
            and origin
            and (allowed == "*" or allowed == origin)
            # Reflecting Origin verbatim writes client-controlled bytes into a
            # response header. http.server does not validate header values, so
            # an obs-folded Origin smuggles extra headers past the parser. A
            # real Origin header never contains CR or LF.
            and not any(ch in origin for ch in "\r\n")
        ):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header(
                "Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS"
            )
            self.send_header(
                "Access-Control-Allow-Headers",
                "Content-Type, Authorization, X-Requested-With",
            )

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def _context_note(self, llm: Any) -> dict[str, Any]:
        """Describe the context window and whether history was compressed.

        Clients that maintain conversation history need to know when the server
        summarised it, or they will keep replaying turns that are no longer
        being sent.
        """
        summarized = bool(getattr(llm, "_last_summarized", False))
        if summarized:
            logger.info("Request history was summarised to fit the context window")

        # Only a real integer is reported. This metadata is a convenience, so a
        # surprising value must never be allowed to fail an otherwise good
        # response by breaking JSON serialisation.
        window = getattr(llm, "context_window", None)
        if not isinstance(window, int) or isinstance(window, bool):
            window = None
        return {"window": window, "summarized": summarized}

    def _send_json(
        self,
        data: Any,
        status: HTTPStatus = HTTPStatus.OK,
        extra_headers: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        # /health is a cacheable GET that discloses the version, the active
        # model id and the context window, so a shared cache must not keep it.
        self.send_header("Cache-Control", "no-store")
        # JSON is not HTML and never should be sniffed as such.
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in extra_headers or []:
            self.send_header(key, value)
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"error": {"message": message, "code": status.value}}, status=status)

    def _safe_error(self, message: str, status: HTTPStatus) -> None:
        """Send an error, tolerating a client that already hung up."""
        try:
            self._send_error(message, status)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _reject_cross_origin_write(self) -> bool:
        """Refuse a state-changing request that a web page initiated.

        A browser will not attach an ``Authorization`` header without a
        preflight, so a keyless loopback server is reachable from any page via a
        *simple* POST: no preflight is required for ``text/plain``, and the
        browser fires the request even though it hides the response. That is
        enough to make any page the operator visits load multi-gigabyte archives
        into their server.

        The header check is not the whole story -- DNS rebinding makes the page
        same-origin -- so this is defence in depth on top of the Host check,
        not a substitute for it.

        Returns True when the request may proceed. The caller must honour False:
        a 403 has already been written at that point, and carrying on would both
        corrupt the response stream and perform the very write that was just
        refused.
        """
        origin = self.headers.get("Origin")
        if not origin:
            return True  # not a browser: curl, the OpenAI SDK, our own client
        allowed = self.server.allow_origin
        if allowed and origin == allowed:
            return True
        logger.warning(
            "Refused cross-origin request from %s", _sanitize_for_log(origin, 120)
        )
        self._safe_error("cross-origin requests are not accepted",
                         HTTPStatus.FORBIDDEN)
        return False

    def _read_json_body(self) -> dict[str, Any]:
        """Read and parse a JSON body, refusing anything oversized.

        ``Content-Length`` is entirely client-controlled, so it is bounds- and
        type-checked before a single byte is read; otherwise one request can
        ask the server to allocate an arbitrary amount of memory.
        """
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            raise _BodyTooLarge("chunked transfer encoding is not supported")

        raw_length = self.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_length)
        except (TypeError, ValueError):
            raise _BadRequest("invalid Content-Length header") from None

        if content_length < 0:
            raise _BadRequest("invalid Content-Length header")
        if content_length > MAX_BODY_BYTES:
            raise _BodyTooLarge(
                f"request body exceeds {MAX_BODY_BYTES} bytes"
            )
        if content_length == 0:
            return {}

        raw = self.rfile.read(content_length)
        if len(raw) != content_length:
            raise _BadRequest("truncated request body")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _BadRequest(f"malformed JSON body: {exc}") from None

    # ─────────────────────────────────────────────────────────────────
    # GET Handlers
    # ─────────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        """Authenticate, dispatch, and never leak a traceback to the client."""
        if not self._host_is_local():
            self._reject_foreign_host()
            return
        if not self._authorized():
            self._deny()
            return
        try:
            self._route_get()
        except (BrokenPipeError, ConnectionResetError):
            raise
        except Exception as exc:
            logger.exception(
                "Unhandled error in GET %s", _sanitize_for_log(self.path)
            )
            self._safe_error(f"internal error: {type(exc).__name__}",
                             HTTPStatus.INTERNAL_SERVER_ERROR)

    def _route_get(self) -> None:
        path = self.path.split("?")[0]

        if path in ("/", "/health"):
            models = self.server.manager.list_models()
            self._send_json({
                "status": "ok",
                "service": "Assembly Engine Server",
                "version": __version__,
                "loaded_models_count": len(models),
                "active_model": models[0]["id"] if models else None,
                "context_window": self.server.manager.context_window(),
            })
            return

        if path in ("/v1/models", "/api/tags"):
            # OpenAI / Ollama compatible model listing
            models = self.server.manager.list_models()
            data = [
                {
                    "id": m["id"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "assembly-engine",
                    "assembly_name": m["assembly_name"],
                }
                for m in models
            ]
            self._send_json({"object": "list", "data": data, "models": data})
            return

        if path == "/api/models":
            self._send_json({"models": self.server.manager.list_models()})
            return

        if path == "/api/info":
            try:
                alias, llm = self.server.manager.get_active()
                h = llm.header
                self._send_json({
                    "alias": alias,
                    "assembly_name": h.assembly_name,
                    "persona": llm.persona,
                    "embedding": {
                        "name": h.embedding.name,
                        "confidence_threshold": h.embedding.confidence_threshold,
                        "fallback_model": h.embedding.fallback_model,
                    },
                    "models": [
                        {"name": m.name, "anchors": m.anchors}
                        for m in h.models
                    ],
                })
            except Exception as e:
                # The raw text names the archive path and header offsets.
                logger.warning("Could not describe active model: %s", e)
                self._send_error(
                    "active model could not be described",
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
            return

        self._send_error("Not Found", HTTPStatus.NOT_FOUND)

    # ─────────────────────────────────────────────────────────────────
    # POST Handlers
    # ─────────────────────────────────────────────────────────────────

    def do_POST(self) -> None:
        """Authenticate, dispatch, and never leak a traceback to the client."""
        if not self._host_is_local():
            self._reject_foreign_host()
            return
        if not self._authorized():
            self._deny()
            return
        if not self._reject_cross_origin_write():
            return
        try:
            self._route_post()
        except (BrokenPipeError, ConnectionResetError):
            raise
        except _BodyTooLarge as exc:
            self._safe_error(str(exc), HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        except _BadRequest as exc:
            self._safe_error(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            logger.exception(
                "Unhandled error in POST %s", _sanitize_for_log(self.path)
            )
            self._safe_error(f"internal error: {type(exc).__name__}",
                             HTTPStatus.INTERNAL_SERVER_ERROR)

    def _route_post(self) -> None:
        path = self.path.split("?")[0]
        body = self._read_json_body()

        if path == "/api/load":
            self._handle_load(body)
            return

        if path == "/api/unload":
            self._handle_unload(body)
            return

        if path == "/api/route":
            self._handle_route(body)
            return

        if path in ("/v1/chat/completions", "/api/chat"):
            self._handle_chat_completions(body, is_openai=(path == "/v1/chat/completions"))
            return

        if path in ("/v1/completions", "/api/generate"):
            self._handle_completions(body, is_openai=(path == "/v1/completions"))
            return

        self._send_error("Not Found", HTTPStatus.NOT_FOUND)

    # ─────────────────────────────────────────────────────────────────
    # Endpoint Implementations
    # ─────────────────────────────────────────────────────────────────

    def _handle_load(self, body: dict[str, Any]) -> None:
        model_path = body.get("model_path") or body.get("path")
        if not model_path:
            self._send_error("Missing 'model_path' in request body")
            return

        alias = body.get("alias")
        persona = body.get("persona")
        try:
            loaded_alias = self.server.manager.load_model(
                model_path=model_path,
                alias=alias,
                persona=persona,
                set_active=True,
            )
            self._send_json({"status": "success", "alias": loaded_alias})
        except _BadRequest as exc:
            logger.info("Rejected model load: %s", exc)
            self._send_error(str(exc), HTTPStatus.BAD_REQUEST)
        except (PermissionError, FileNotFoundError, ValueError) as exc:
            # Deliberately terse: echoing the underlying message would turn
            # this endpoint into a filesystem oracle.
            logger.info("Rejected model load: %s", exc)
            self._send_error("model could not be loaded", HTTPStatus.BAD_REQUEST)
        except RuntimeError as exc:
            # str(exc) here can name the staging directory and its free space
            # (StagingError subclasses RuntimeError). The operator gets the
            # detail in the log; the client gets the status.
            logger.warning("Model load refused: %s", exc)
            self._send_error(
                "server is at its model limit; POST /api/unload to free a slot",
                HTTPStatus.TOO_MANY_REQUESTS,
            )
        except Exception as exc:
            logger.exception("Model load failed")
            self._send_error(
                f"failed to load model: {type(exc).__name__}",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_unload(self, body: dict[str, Any]) -> None:
        alias = body.get("alias") or body.get("model")
        if not alias:
            self._send_error("Missing 'alias' in request body")
            return
        try:
            remaining = self.server.manager.unload_model(alias)
        except _BadRequest as exc:
            self._send_error(str(exc))
            return
        self._send_json({
            "status": "success",
            "unloaded": alias,
            "remaining": remaining,
        })

    def _handle_route(self, body: dict[str, Any]) -> None:
        prompt = body.get("prompt") or body.get("text", "")
        if not prompt:
            self._send_error("Missing 'prompt' in request body")
            return

        try:
            alias, llm = self.server.manager.get_active(body.get("model"))
            route = llm.route(prompt)
            self._send_json({
                "assembly": alias,
                "selected_model": route.model_name,
                "score": round(route.score, 4),
                "is_fallback": route.is_fallback,
            })
        except Exception as exc:
            # str(exc) here can carry the staged blob path, so it is logged and
            # not returned.
            logger.warning("Routing failed: %s", exc)
            self._send_error(
                "routing failed", HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def _handle_chat_completions(self, body: dict[str, Any], is_openai: bool = True) -> None:
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            self._send_error("Missing or invalid 'messages' field in body")
            return

        req_model = body.get("model")
        # Read every sampler knob the client sent, not just temperature and
        # max_tokens. Values are clamped rather than rejected, so a client
        # asking for something out of range still gets an answer.
        sampling = sampling_params_from_body(body)
        stream_requested = bool(body.get("stream", False))

        try:
            alias, llm = self.server.manager.get_active(req_model)
            last_user_msg = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content") or ""
                    break
            route = llm.route(last_user_msg)

            # SSE Streaming support for OpenAI clients
            if stream_requested and is_openai:
                self._stream_sse(alias, llm, route, messages, sampling)
                return

            # Non-streaming standard response
            resp = llm.create_chat_completion(
                messages=messages,
                stream=False,
                **sampling.to_llama_kwargs(),
            )

            if is_openai:
                if isinstance(resp, dict):
                    resp["model"] = f"{alias}:{route.model_name}"
                    resp["assembly_route"] = {
                        "selected_model": route.model_name,
                        "score": round(route.score, 4),
                        "is_fallback": route.is_fallback,
                    }
                    resp["assembly_context"] = self._context_note(llm)
                self._send_json(resp)
            else:
                content = ""
                if isinstance(resp, dict) and resp.get("choices"):
                    content = (
                        resp["choices"][0].get("message", {}).get("content", "")
                    )
                note = self._context_note(llm)
                self._send_json({
                    "model": f"{alias}:{route.model_name}",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {"role": "assistant", "content": content},
                    "done": True,
                    "routed_model": route.model_name,
                    "context_window": note["window"],
                    "summarized": note["summarized"],
                })
        except Exception as exc:
            # An over-long conversation is the caller's problem to fix, not a
            # server fault, so it gets a 400 with a usable explanation.
            overflow = _context_overflow(exc)
            if overflow:
                logger.warning("Context window exceeded: %s", overflow)
                self._send_error(overflow, HTTPStatus.BAD_REQUEST)
            else:
                # str(exc) reaches the client through llama.cpp's own message,
                # which names the staged blob path; the operator gets it below.
                logger.warning("Chat completion failed: %s", exc)
                self._safe_error(
                    f"generation failed: {type(exc).__name__}",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    def _stream_sse(
        self,
        alias: str,
        llm: Any,
        route: Any,
        messages: list[dict[str, Any]],
        sampling: SamplingParams,
    ) -> None:
        """Stream an OpenAI-compatible SSE response.

        The generator is primed with one ``next()`` *before* the headers go
        out. llama-cpp-python is lazy: it tokenises on the first advance of the
        generator, so a prompt that cannot fit the context window only fails at
        that point. Priming first means the client gets a real 400 with an
        explanation instead of a 200 followed by an error frame.

        Once the headers are on the wire no HTTP error is possible, so any
        later failure is reported as an SSE ``error`` frame plus ``[DONE]``.
        """
        gen = llm.create_chat_completion(
            messages=messages,
            stream=True,
            **sampling.to_llama_kwargs(),
        )
        first_chunk: dict[str, Any] | None
        try:
            first_chunk = next(gen)
        except StopIteration:
            first_chunk = None
        except Exception as exc:
            # Nothing has been written yet, so a proper HTTP error still fits.
            gen.close()
            overflow = _context_overflow(exc)
            if overflow:
                logger.warning("Context window exceeded: %s", overflow)
                self._send_error(overflow, HTTPStatus.BAD_REQUEST)
            else:
                logger.exception("Streaming setup failed")
                self._safe_error(
                    f"generation failed: {type(exc).__name__}",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._send_cors_headers()
        self.end_headers()
        self._stream_open = True

        def emit(payload: str) -> bool:
            """Write one SSE frame. False if the client has gone away."""
            try:
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                return False

        model_label = f"{alias}:{route.model_name}"
        # Told once, at the end. A client replaying a whole history needs to
        # know that the turns it sent are no longer being sent, and that is
        # only knowable after the generation which compressed them. Extra keys
        # on a chunk are ignored by clients that only read the text.
        note = self._context_note(llm)
        try:
            # Replay the chunk that was pulled while priming.
            chunks = [] if first_chunk is None else [first_chunk]
            chunks.extend(gen)
            for chunk in chunks:
                # llama-cpp-python 0.2.x appends a bare "[DONE]" string;
                # serialising that would emit a second, bogus sentinel.
                if not isinstance(chunk, dict):
                    continue
                chunk["model"] = model_label
                if not emit(f"data: {json.dumps(chunk)}\n\n"):
                    logger.info("SSE client disconnected mid-stream")
                    gen.close()
                    return
        except Exception as exc:
            overflow = _context_overflow(exc)
            if overflow:
                logger.warning("Streaming failed: %s", overflow)
                message = overflow
            else:
                logger.exception("Streaming failed")
                message = f"generation failed: {type(exc).__name__}"
            emit(f"data: {json.dumps({'error': {'message': message}})}\n\n")

        emit(f"data: {json.dumps(_note_chunk(model_label, note))}\n\n")
        emit("data: [DONE]\n\n")

    def _handle_completions(self, body: dict[str, Any], is_openai: bool = True) -> None:
        prompt = body.get("prompt", "")
        if not prompt:
            self._send_error("Missing 'prompt' in body")
            return

        req_model = body.get("model")
        sampling = sampling_params_from_body(body, base=SamplingParams(max_tokens=256))

        try:
            alias, llm = self.server.manager.get_active(req_model)
            route = llm.route(prompt)

            resp = llm(
                prompt=prompt,
                **sampling.to_llama_kwargs(),
            )

            if is_openai:
                if isinstance(resp, dict):
                    resp["model"] = f"{alias}:{route.model_name}"
                    resp["assembly_route"] = {
                        "selected_model": route.model_name,
                        "score": round(route.score, 4),
                        "is_fallback": route.is_fallback,
                    }
                self._send_json(resp)
            else:
                text = ""
                if isinstance(resp, dict) and resp.get("choices"):
                    text = resp["choices"][0].get("text") or (
                        resp["choices"][0].get("message", {}).get("content", "")
                    )
                self._send_json({
                    "model": f"{alias}:{route.model_name}",
                    "response": text,
                    "done": True,
                    "routed_model": route.model_name,
                })
        except Exception as exc:
            overflow = _context_overflow(exc)
            if overflow:
                logger.warning("Context window exceeded: %s", overflow)
                self._send_error(overflow, HTTPStatus.BAD_REQUEST)
            else:
                logger.warning("Completion failed: %s", exc)
                self._safe_error(
                    f"generation failed: {type(exc).__name__}",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    def log_message(self, format: str, *args: Any) -> None:
        """Custom clean logging.

        The formatted line contains the request target, which is
        attacker-controlled and may carry terminal escapes -- raw ``\\x1b``
        sequences in an operator's terminal clear the screen and forge output,
        so control characters are replaced before this reaches stderr.
        """
        line = format % args
        sys.stderr.write(
            f"[assembly-server] {self.address_string()} - "
            f"{_sanitize_for_log(line)}\n"
        )


class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that refuses connections past a fixed ceiling.

    The base class spawns a thread per accepted connection with no limit, so a
    client holding a few thousand idle sockets open grows the thread count to
    match. Counting them here keeps the process's memory bounded regardless of
    what the client does.
    """

    daemon_threads = True
    allow_reuse_address = True
    #: Deep enough that a normal burst is not refused, shallow enough to bound.
    request_queue_size = 64

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._active_connections = 0
        self._connection_slots = threading.BoundedSemaphore(
            MAX_CONCURRENT_CONNECTIONS
        )
        self._connections: dict[int, tuple[float, Any]] = {}
        self._watchdog: Optional[threading.Thread] = None
        super().__init__(*args, **kwargs)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._connection_slots.acquire(blocking=False):
            # Refuse rather than queue: a queued connection still occupies an
            # fd and a kernel buffer, and the client is not doing anything we
            # want to wait for. close_request() rather than shutdown_request():
            # no slot was acquired here, so nothing may be given back -- a
            # BoundedSemaphore raises on over-release, and one client opening a
            # few sockets past the ceiling would take the server down.
            logger.warning(
                "Refused connection from %s: %d already in flight",
                client_address,
                self._active_connections,
            )
            self.close_request(request)
            return
        with _CONNECTION_COUNT_LOCK:
            self._active_connections += 1
        try:
            super().process_request(request, client_address)
        except BaseException:
            # The worker thread never started, so it will not release for us.
            self._release_slot()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release_slot()

    def _release_slot(self) -> None:
        """Return one connection slot. Called exactly once per acquisition."""
        with _CONNECTION_COUNT_LOCK:
            self._active_connections = max(0, self._active_connections - 1)
        self._connection_slots.release()

    # ── Connection deadlines ──────────────────────────────────────────

    def register_connection(self, handler: Any) -> None:
        """Give *handler* a wall-clock deadline and start the sweeper if needed."""
        with _CONNECTION_COUNT_LOCK:
            self._connections[id(handler)] = (time.monotonic(), handler)
            first = self._watchdog is None
            if first:
                self._watchdog = threading.Thread(
                    target=self._expire_stale_connections,
                    name="assembly-connection-watchdog",
                    daemon=True,
                )
        if first:
            self._watchdog.start()

    def forget_connection(self, handler: Any) -> None:
        with _CONNECTION_COUNT_LOCK:
            self._connections.pop(id(handler), None)

    def _expire_stale_connections(self) -> None:
        """Close connections that have outlived SOCKET_TIMEOUT.

        The socket's own timeout is per-recv, so a client sending one byte
        every few minutes keeps every recv() returning and holds the connection
        open indefinitely. This is the wall-clock bound that closes that gap.
        """
        while True:
            time.sleep(1.0)
            now = time.monotonic()
            with _CONNECTION_COUNT_LOCK:
                stale = [
                    handler
                    for started, handler in self._connections.values()
                    if now - started > SOCKET_TIMEOUT
                ]
                for handler in stale:
                    self._connections.pop(id(handler), None)
            for handler in stale:
                logger.warning(
                    "Closing connection from %s after %ds without completing a "
                    "request", handler.address_string(), SOCKET_TIMEOUT
                )
                try:
                    handler._expire_connection()
                except OSError:
                    pass


class AssemblyServer(_BoundedThreadingHTTPServer):
    """Threaded HTTP Server for Assembly Engine, with a bounded worker pool."""

    def __init__(
        self,
        server_address: tuple[str, int],
        manager: ModelManager,
        api_key: Optional[str] = None,
        allow_origin: Optional[str] = None,
    ) -> None:
        # An empty key must not be read as "no authentication required": the
        # loopback guard in serve() keys off exactly this field, so a caller
        # passing "" would otherwise get an open server with no warning.
        api_key = api_key.strip() if isinstance(api_key, str) else api_key
        if api_key is not None and not api_key:
            raise ValueError(
                "api_key must be a non-empty token; pass None for no auth"
            )
        # A bearer token travels as an HTTP header, and RFC 6750's b64token
        # alphabet is ASCII. http.server decodes header bytes as Latin-1, so a
        # client cannot even echo back a non-ASCII key: "sécret" is read back as
        # "sÃ©cret" and every request is refused. Better to say so at startup
        # than to serve a key nobody can ever use.
        if api_key is not None and not api_key.isascii():
            raise ValueError(
                "api_key must be ASCII: HTTP headers are Latin-1 on the wire, "
                "so a non-ASCII token can never match. Use --api-key with "
                "letters, digits, and . _ - only."
            )
        super().__init__(server_address, AssemblyRequestHandler)
        self.manager = manager
        self.api_key = api_key
        self.allow_origin = allow_origin
        # Only meaningful on a loopback listener: there, a Host naming anything
        # else is either a mistake or a DNS rebinding attempt.
        self.require_local_host = is_loopback(server_address[0])


def serve(
    model_path: Optional[str | Path] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    persona: Optional[str] = None,
    n_gpu_layers: int = 0,
    verbose: bool = False,
    api_key: Optional[str] = None,
    allow_origin: Optional[str] = None,
    models_dir: Optional[str | Path] = None,
    temp_dir: Optional[str | Path] = None,
) -> None:
    """Start the Assembly Engine HTTP server.

    Binds to loopback by default. Passing a non-loopback *host* exposes
    unauthenticated inference to the network unless *api_key* is set, so that
    combination is refused rather than silently allowed.
    """
    api_key = api_key or os.environ.get("ASSEMBLY_API_KEY") or None

    if not is_loopback(host) and not api_key:
        raise SystemExit(
            f"refusing to bind {host} without an API key: anyone who can reach "
            f"this machine would be able to run inference. Pass --api-key or "
            f"set ASSEMBLY_API_KEY, or use --host 127.0.0.1."
        )

    # Validated here, not only in AssemblyServer: this is a configuration
    # mistake, and a traceback is a poor way to report one. An empty or
    # non-ASCII key is reported the same way whichever it is.
    if api_key is not None:
        stripped = api_key.strip()
        if not stripped:
            raise SystemExit(
                "the API key is empty: pass None for no auth, or set a real "
                "token via --api-key or ASSEMBLY_API_KEY."
            )
        if not stripped.isascii():
            raise SystemExit(
                "the API key must be ASCII: HTTP headers are Latin-1 on the "
                "wire, so a non-ASCII token can never match and every request "
                "would be refused. Use letters, digits, and . _ - only."
            )
        api_key = stripped

    # Confine /api/load to the served archive's own directory by default. Left
    # unconfined, that endpoint accepts any absolute path to any *.asmb the
    # server user can read, which turns a chat server into an arbitrary-file
    # opener for anything that parses as an archive. A sibling directory is
    # still reachable, which is the case operators actually want; opening up
    # deliberately is one flag away.
    if models_dir is None and model_path:
        models_dir = Path(model_path).expanduser().resolve().parent

    manager = ModelManager(
        default_model_path=model_path,
        persona=persona,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
        models_dir=models_dir,
        temp_dir=temp_dir,
    )

    server_address = (host, port)
    server = AssemblyServer(
        server_address, manager, api_key=api_key, allow_origin=allow_origin
    )

    host_str = "localhost" if is_loopback(host) else host
    base = f"http://{host_str}:{port}"
    rows: list[tuple[str, str]] = [
        ("Archive", Path(model_path).name if model_path else "none — waiting for /api/load"),
        ("Models", manager.describe_models()),
        ("Context", manager.describe_context()),
        ("Auth", "bearer token required" if api_key else "none (loopback only)"),
    ]
    if models_dir:
        rows.append(("Models dir", f"{models_dir}  (/api/load confined here)"))
    else:
        rows.append((
            "Models dir",
            "unconfined — /api/load can open any readable .asmb "
            "(pass --models-dir to restrict it)",
        ))
    rows.extend([
        ("", ""),
        ("OpenAI", f"{base}/v1/chat/completions"),
        ("Ollama", f"{base}/api/chat"),
        ("Route", f"{base}/api/route"),
        ("Health", f"{base}/health"),
    ])
    ui.banner("Assembly Engine", f"listening on {host}:{port}", rows)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        ui.info("stopping server")
    finally:
        server.server_close()
        manager.close()
        ui.dim("server stopped cleanly")
