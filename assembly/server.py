"""
Assembly Engine — HTTP Server Daemon.

Provides a multi-device REST API compatible with:
- OpenAI API (/v1/chat/completions, /v1/completions, /v1/models)
- Ollama-style API (/api/generate, /api/chat)
- Assembly native API (/api/route, /api/info, /api/load, /health)

Allows scripts, external programs, Web UIs, and devices across LAN/WAN
to interact with locally hosted .asmb assemblies.

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
import os
import secrets
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from assembly import __version__
from assembly.wrapper import AssemblyLlama

logger = logging.getLogger("assembly.server")

#: Largest request body we will read, in bytes.
MAX_BODY_BYTES = 1 << 20  # 1 MiB

#: Seconds a client may hold a connection open.
SOCKET_TIMEOUT = 300

#: How many archives a single server will hold open at once.
MAX_LOADED_MODELS = 8

#: Hostnames that mean "this machine only".
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def is_loopback(host: str) -> bool:
    """True when binding *host* keeps the service on this machine only."""
    return host in LOOPBACK_HOSTS


class _BadRequest(Exception):
    """Malformed request from the client (400)."""


class _BodyTooLarge(Exception):
    """Request body exceeded :data:`MAX_BODY_BYTES` (413)."""


class ModelManager:
    """Thread-safe manager for loaded .asmb archives."""

    def __init__(
        self,
        default_model_path: Optional[str | Path] = None,
        persona: Optional[str] = None,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        models_dir: Optional[str | Path] = None,
        temp_dir: Optional[str | Path] = None,
        max_models: int = MAX_LOADED_MODELS,
    ) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, AssemblyLlama] = {}
        self._active_alias: Optional[str] = None
        self._persona = persona
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._verbose = verbose
        self._max_models = max_models
        self._temp_dir = temp_dir
        self._models_dir = (
            Path(models_dir).expanduser().resolve() if models_dir else None
        )

        if default_model_path:
            self.load_model(default_model_path, alias="default", set_active=True)

    def _authorize_path(self, model_path: str | Path) -> Path:
        """Resolve *model_path* and confirm the server is allowed to open it.

        When ``models_dir`` is configured the path must live inside it, so a
        remote caller cannot use this endpoint to probe the filesystem. The
        ``.asmb`` suffix and existence checks are deliberately boring and
        uniform: the error strings must not distinguish "no such file" from
        "not an archive", or this becomes a file-existence oracle.
        """
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
        name = alias or path.stem

        with self._lock:
            replacing = name in self._models
            if not replacing and len(self._models) >= self._max_models:
                raise RuntimeError(
                    f"too many models loaded (limit {self._max_models})"
                )

        # Constructed outside the lock: this takes seconds to minutes.
        llm = AssemblyLlama(
            asmb_path=path,
            persona=persona or self._persona,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=self._verbose,
            temp_dir=self._temp_dir,
        )

        with self._lock:
            previous = self._models.get(name)
            self._models[name] = llm
            if set_active or self._active_alias is None:
                self._active_alias = name

        if previous is not None:
            previous.close()

        logger.info(f"Loaded assembly model '{name}' from {path}")
        return name

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

    def close(self) -> None:
        with self._lock:
            for llm in self._models.values():
                try:
                    llm.close()
                except Exception:
                    pass
            self._models.clear()


class AssemblyRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for Assembly Engine API."""

    server: "AssemblyServer"  # Type hint for server instance

    # Slow clients must not be able to pin a thread forever.
    # socketserver.StreamRequestHandler.setup() applies this for us.
    timeout = SOCKET_TIMEOUT

    server_version = "AssemblyEngine"
    sys_version = ""

    # ── Auth ─────────────────────────────────────────────────────────

    def _authorized(self) -> bool:
        """Check the bearer token when the server was started with one."""
        expected = self.server.api_key
        if not expected:
            return True
        header = self.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return False
        return secrets.compare_digest(token.strip(), expected)

    def _deny(self) -> None:
        self._send_json(
            {"error": {"message": "unauthorized", "code": HTTPStatus.UNAUTHORIZED.value}},
            status=HTTPStatus.UNAUTHORIZED,
            extra_headers=[("WWW-Authenticate", "Bearer")],
        )

    def _send_cors_headers(self) -> None:
        """Emit CORS headers only for an explicitly allowed origin.

        Wildcarding this on a state-changing local service would let any web
        page the operator visits drive it, so it is opt-in.
        """
        origin = self.headers.get("Origin")
        allowed = self.server.allow_origin
        if allowed and origin and (allowed == "*" or allowed == origin):
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

    def _send_html(self, html_content: str) -> None:
        payload = html_content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

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

    def _is_public_shell(self) -> bool:
        """True for the static chat page, which may be served without a token.

        The page is a self-contained HTML shell: no model names, no data, and
        no key are embedded in it — everything it renders comes from /api/info
        and /v1/chat/completions, which stay authenticated. A browser cannot
        attach an Authorization header to a top-level navigation, so gating
        the page itself would make the UI permanently unreachable on an
        authenticated server.
        """
        path = self.path.split("?")[0]
        accept = self.headers.get("Accept", "")
        return path == "/chat" or (path == "/" and "text/html" in accept)

    def do_GET(self) -> None:
        """Authenticate, dispatch, and never leak a traceback to the client."""
        if self._is_public_shell():
            try:
                self._send_html(WEB_CHAT_HTML)
            except (BrokenPipeError, ConnectionResetError):
                raise
            except Exception as exc:
                logger.exception("Unhandled error serving web UI")
                self._safe_error(f"internal error: {type(exc).__name__}",
                                 HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if not self._authorized():
            self._deny()
            return
        try:
            self._route_get()
        except (BrokenPipeError, ConnectionResetError):
            raise
        except Exception as exc:
            logger.exception("Unhandled error in GET %s", self.path)
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
                "web_ui": "/chat",
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
                self._send_error(str(e), HTTPStatus.SERVICE_UNAVAILABLE)
            return

        self._send_error("Not Found", HTTPStatus.NOT_FOUND)

    # ─────────────────────────────────────────────────────────────────
    # POST Handlers
    # ─────────────────────────────────────────────────────────────────

    def do_POST(self) -> None:
        """Authenticate, dispatch, and never leak a traceback to the client."""
        if not self._authorized():
            self._deny()
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
            logger.exception("Unhandled error in POST %s", self.path)
            self._safe_error(f"internal error: {type(exc).__name__}",
                             HTTPStatus.INTERNAL_SERVER_ERROR)

    def _route_post(self) -> None:
        path = self.path.split("?")[0]
        body = self._read_json_body()

        if path == "/api/load":
            self._handle_load(body)
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
        except (PermissionError, FileNotFoundError, ValueError) as exc:
            # Deliberately terse: echoing the underlying message would turn
            # this endpoint into a filesystem oracle.
            logger.info("Rejected model load: %s", exc)
            self._send_error("model could not be loaded", HTTPStatus.BAD_REQUEST)
        except RuntimeError as exc:
            self._send_error(str(exc), HTTPStatus.TOO_MANY_REQUESTS)
        except Exception as exc:
            logger.exception("Model load failed")
            self._send_error(
                f"failed to load model: {type(exc).__name__}",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

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
            logger.warning("Routing failed: %s", exc)
            self._send_error(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_chat_completions(self, body: dict[str, Any], is_openai: bool = True) -> None:
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            self._send_error("Missing or invalid 'messages' field in body")
            return

        req_model = body.get("model")
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)
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
                self._stream_sse(alias, llm, route, messages, max_tokens, temperature)
                return

            # Non-streaming standard response
            resp = llm.create_chat_completion(
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
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
                content = ""
                if isinstance(resp, dict) and resp.get("choices"):
                    content = (
                        resp["choices"][0].get("message", {}).get("content", "")
                    )
                self._send_json({
                    "model": f"{alias}:{route.model_name}",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {"role": "assistant", "content": content},
                    "done": True,
                    "routed_model": route.model_name,
                })
        except Exception as exc:
            logger.warning("Chat completion failed: %s", exc)
            self._send_error(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def _stream_sse(
        self,
        alias: str,
        llm: Any,
        route: Any,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> None:
        """Stream an OpenAI-compatible SSE response.

        Once the headers are on the wire we can no longer send an HTTP error,
        so failures are reported as an SSE ``error`` frame followed by
        ``[DONE]`` — the client sees a failure rather than a silent, empty
        "success".
        """
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
        try:
            gen = llm.create_chat_completion(
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for chunk in gen:
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
            logger.exception("Streaming failed")
            emit(f"data: {json.dumps({'error': {'message': f'generation failed: {type(exc).__name__}'}})}\n\n")

        emit("data: [DONE]\n\n")

    def _handle_completions(self, body: dict[str, Any], is_openai: bool = True) -> None:
        prompt = body.get("prompt", "")
        if not prompt:
            self._send_error("Missing 'prompt' in body")
            return

        req_model = body.get("model")
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.7)

        try:
            alias, llm = self.server.manager.get_active(req_model)
            route = llm.route(prompt)

            resp = llm(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
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
            self._send_error(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        """Custom clean logging."""
        sys.stderr.write(f"[assembly-server] {self.address_string()} - {format % args}\n")


class AssemblyServer(ThreadingHTTPServer):
    """Threaded HTTP Server for Assembly Engine."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        manager: ModelManager,
        api_key: Optional[str] = None,
        allow_origin: Optional[str] = None,
    ) -> None:
        super().__init__(server_address, AssemblyRequestHandler)
        self.manager = manager
        self.api_key = api_key
        self.allow_origin = allow_origin


def serve(
    model_path: Optional[str | Path] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    persona: Optional[str] = None,
    n_ctx: int = 2048,
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

    manager = ModelManager(
        default_model_path=model_path,
        persona=persona,
        n_ctx=n_ctx,
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
    print("=" * 65)
    print("  Assembly Engine Server is running!")
    print(f"  • Listening on: http://{host}:{port}")
    if model_path:
        print(f"  • Active Model: {Path(model_path).name}")
    else:
        print("  • Active Model: (none, waiting for /api/load)")
    if models_dir:
        print(f"  • Models dir:   {models_dir}  (/api/load is jailed here)")
    print(f"  • Auth:         {'bearer token required' if api_key else 'none (loopback only)'}")
    print()
    print("  Endpoints:")
    print(f"  - Web chat:     http://{host_str}:{port}/chat")
    print(f"  - OpenAI:       http://{host_str}:{port}/v1/chat/completions")
    print(f"  - Ollama:       http://{host_str}:{port}/api/chat")
    print(f"  - Route check:  http://{host_str}:{port}/api/route")
    print(f"  - Health:       http://{host_str}:{port}/health")
    print("=" * 65)
    print("Press Ctrl+C to stop the server.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Assembly server...")
    finally:
        server.server_close()
        manager.close()
        print("Server stopped cleanly.")


WEB_CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Assembly Engine • Live Chat</title>
<style>
  :root {
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --accent: #58a6ff;
    --accent-glow: rgba(88, 166, 255, 0.15);
    --text: #c9d1d9;
    --text-bright: #f0f6fc;
    --user-msg: #1f6feb;
    --bot-msg: #21262d;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  header {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; color: var(--text-bright); font-size: 1.1rem; }
  .dot { width: 10px; height: 10px; background: #3fb950; border-radius: 50%; box-shadow: 0 0 8px #3fb950; }
  .badges { display: flex; gap: 8px; font-size: 0.8rem; }
  .badge { background: #30363d; padding: 4px 10px; border-radius: 12px; color: var(--text-bright); }
  .badge-accent { background: var(--accent-glow); color: var(--accent); border: 1px solid var(--accent); }
  #chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    max-width: 900px;
    width: 100%;
    margin: 0 auto;
  }
  .msg-row { display: flex; flex-direction: column; }
  .msg-row.user { align-items: flex-end; }
  .msg-row.assistant { align-items: flex-start; }
  .bubble {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 16px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 0.95rem;
  }
  .msg-row.user .bubble {
    background: var(--user-msg);
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .msg-row.assistant .bubble {
    background: var(--bot-msg);
    color: var(--text-bright);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }
  .route-tag {
    font-size: 0.75rem;
    margin-top: 4px;
    color: #8b949e;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .route-tag span { color: #58a6ff; font-weight: 600; }
  footer {
    background: var(--card);
    border-top: 1px solid var(--border);
    padding: 14px 20px;
  }
  .input-bar {
    max-width: 900px;
    margin: 0 auto;
    display: flex;
    gap: 10px;
  }
  textarea {
    flex: 1;
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 12px;
    color: var(--text-bright);
    padding: 12px 16px;
    font-size: 1rem;
    resize: none;
    height: 48px;
    outline: none;
    font-family: inherit;
  }
  textarea:focus { border-color: var(--accent); }
  button {
    background: var(--accent);
    color: #0d1117;
    border: none;
    border-radius: 12px;
    padding: 0 20px;
    font-weight: 600;
    cursor: pointer;
    font-size: 0.95rem;
    transition: opacity 0.2s;
  }
  button:hover { opacity: 0.9; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="dot"></div>
    <span>Assembly Engine</span>
  </div>
  <div class="badges" id="header-badges">
    <div class="badge badge-accent" id="model-badge">Loading model...</div>
  </div>
</header>
<div id="chat-container">
  <div class="msg-row assistant">
    <div class="bubble">Hello! I am Assembly. How can I help you today?</div>
  </div>
</div>
<footer>
  <div class="input-bar">
    <textarea id="prompt-input" placeholder="Type a message... (Press Enter to send)" rows="1"></textarea>
    <button id="send-btn" onclick="sendMessage()">Send</button>
  </div>
</footer>
<script>
const container = document.getElementById("chat-container");
const input = document.getElementById("prompt-input");
const sendBtn = document.getElementById("send-btn");
const modelBadge = document.getElementById("model-badge");
let conversation = [];

// When the server runs with --api-key, this page has to authenticate too or
// every request comes back 401. The key is kept in sessionStorage only: it is
// per-tab, cleared on close, and never appended to a URL or written to disk.
function authHeaders() {
  const key = sessionStorage.getItem("assembly_api_key");
  return key ? { "Authorization": "Bearer " + key } : {};
}

function promptForKey() {
  const key = window.prompt("This server requires an API key. Enter the key used to start it:");
  if (!key) return false;
  sessionStorage.setItem("assembly_api_key", key.trim());
  return true;
}

async function fetchWithAuth(path, options = {}, retried = false) {
  const res = await fetch(path, {
    ...options,
    headers: { ...authHeaders(), ...(options.headers || {}) }
  });
  if (res.status === 401 && !retried && promptForKey()) {
    return fetchWithAuth(path, options, true);
  }
  return res;
}

async function updateInfo() {
  try {
    const res = await fetchWithAuth("/api/info");
    if (res.ok) {
      const data = await res.json();
      modelBadge.textContent = "⚡ " + data.assembly_name + " (" + data.models.length + " sub-models)";
    } else if (res.status === 401) {
      modelBadge.textContent = "🔒 API key required";
    }
  } catch(e) {}
}
updateInfo();

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  sendBtn.disabled = true;

  appendMsg("user", text);
  conversation.push({ role: "user", content: text });

  const loadingRow = appendMsg("assistant", "Thinking...");

  try {
    const res = await fetchWithAuth("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: conversation, max_tokens: 512 })
    });
    const data = await res.json();
    loadingRow.remove();

    if (res.status === 401) {
      appendMsg("assistant", "Unauthorized: check the API key for this server.");
      conversation.pop();
      return;
    }
    if (data.choices && data.choices[0]) {
      const reply = data.choices[0].message ? data.choices[0].message.content : data.choices[0].text;
      const routeInfo = data.assembly_route || {};
      const subModel = routeInfo.selected_model ? "⚡ routed to: " + routeInfo.selected_model + " (" + Math.round((routeInfo.score || 0) * 100) + "%)" : "";
      appendMsg("assistant", reply, subModel);
      conversation.push({ role: "assistant", content: reply });
    } else {
      appendMsg("assistant", "Error: " + JSON.stringify(data));
    }
  } catch(err) {
    loadingRow.remove();
    appendMsg("assistant", "Network Error: " + err.message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

function appendMsg(role, text, tag) {
  const row = document.createElement("div");
  row.className = "msg-row " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  row.appendChild(bubble);

  if (tag) {
    const routeTag = document.createElement("div");
    routeTag.className = "route-tag";
    // textContent, not innerHTML: the tag embeds a model name from the
    // archive header, which must never be parsed as markup.
    const tagSpan = document.createElement("span");
    tagSpan.textContent = tag;
    routeTag.appendChild(tagSpan);
    row.appendChild(routeTag);
  }

  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
  return row;
}
</script>
</body>
</html>
"""
