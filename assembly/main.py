"""
CLI entry point for Assembly Engine.

Commands
--------
assembly build -c config.json -o out.asmb
    Pack models into a single .asmb archive.

assembly run -m archive.asmb ["prompt"]
    Chat with an archive, or answer a single prompt.

assembly serve -m archive.asmb
    Share the archive over HTTP (OpenAI and Ollama compatible).

assembly info -m archive.asmb
    Print header metadata without loading any models.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

from assembly import __version__, ui
from assembly.config import (
    SAMPLING_FIELDS,
    SAMPLING_PRESETS,
    SamplingParams,
    sampling_params_from_preset,
)
from assembly.header import InvalidArchiveError


def _cmd_build(args: argparse.Namespace) -> None:
    """Handle ``assembly build``."""
    from assembly.config import PreflightError
    from assembly.packer import BuildError, build

    try:
        build(
            config_path=args.config,
            output_path=args.output,
            skip_embeddings=args.skip_embeddings,
        )
    except PreflightError as exc:
        print()
        _render_report(exc.report, args.config)
        sys.exit(1)
    except BuildError as exc:
        ui.error(str(exc))
        sys.exit(1)


def _verify_archive(header, mm, path: Path) -> bool:
    """Re-hash every blob and report mismatches. Returns True if all match.

    Reads through the same memory map the header describes, so this proves
    the recorded digests describe *this* archive rather than whatever was on
    disk when it was built. Slow on purpose: it is opt-in via ``--verify``.
    """
    from assembly.header import hash_region

    ok = True
    entries = [(header.embedding.name, header.embedding.offset,
                header.embedding.size, header.embedding.sha256)]
    entries += [(m.name, m.offset, m.size, m.sha256) for m in header.models]

    for name, offset, size, expected in entries:
        if not expected:
            print(f"  {name}: no digest recorded, skipped")
            continue
        print(f"  {name} …", end="", flush=True)
        actual = hash_region(mm, offset, size)
        if actual == expected:
            print(" ok")
        else:
            ok = False
            print(" MISMATCH")
            print(f"    expected {expected}")
            print(f"    actual   {actual}")
    return ok


def _cmd_info(args: argparse.Namespace) -> None:
    """Handle ``assembly info``."""
    import mmap as _mmap

    from assembly.header import read_header

    path = Path(_require_archive(args.model))

    # try/finally: a corrupt archive must not leak the descriptor.
    with open(path, "rb") as fh:
        try:
            with _mmap.mmap(fh.fileno(), 0, access=_mmap.ACCESS_READ) as mm:
                header = read_header(mm)
                verified = _verify_archive(header, mm, path) \
                    if getattr(args, "verify", False) else None
        except InvalidArchiveError as exc:
            ui.error(str(exc))
            sys.exit(1)
        except OSError as exc:
            ui.error(f"could not read {path}: {exc}")
            sys.exit(1)

    if verified is not None:
        print()
        if not verified:
            ui.error(
                f"{path} is corrupt: blob digests do not match the header. "
                f"Rebuild it with 'assembly build'."
            )
            sys.exit(1)
        ui.success(f"{path} verified: every blob matches its recorded SHA-256.")
        print()

    total = sum(m.size for m in header.models) + header.embedding.size
    print(f"Assembly : {header.assembly_name}")
    print(f"Version  : {header.version}")
    print(f"Page size: {header.page_size}")
    print(f"Size     : {total / (1024 ** 3):.2f} GiB across {len(header.models)} models")
    if header.persona:
        print(f"Persona  : {header.persona}")
    print()
    emb = header.embedding
    print(f"Embedding: {emb.name}")
    print(f"  offset : {emb.offset:#x}  ({emb.offset:,} B)")
    print(f"  size   : {emb.size:,} B  ({emb.size / 1e6:.1f} MB)")
    print(f"  threshold: {emb.confidence_threshold}")
    print(f"  fallback : {emb.fallback_model}")
    if emb.sha256:
        print(f"  sha256   : {emb.sha256}")
    print()
    for i, m in enumerate(header.models):
        print(f"Model [{i}]: {m.name}")
        print(f"  offset : {m.offset:#x}  ({m.offset:,} B)")
        print(f"  size   : {m.size:,} B  ({m.size / 1e6:.1f} MB)")
        print(f"  anchors: {m.anchors}")
        vecs = m.anchor_vectors()
        if vecs:
            print(
                f"  embeddings: {len(vecs)} vectors × {m.anchor_dim} dims "
                f"(float32)"
            )
        if m.sha256:
            print(f"  sha256: {m.sha256}")
        print()


def _cmd_init(args: argparse.Namespace) -> None:
    """Generate a starter assembly configuration template."""
    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        ui.warn(f"File '{out_path}' already exists! Use --force to overwrite.")
        return

    sample = {
        "assembly_name": "Assembly-1.0",
        "persona": "You are Assembly, an advanced unified AI assistant. You speak as a single intelligent entity and maintain this persona across all prompts.",
        "embedding": {
            "name": "bge-small-router",
            "path": "/absolute/path/to/embedding_model.gguf",
            "confidence_threshold": 0.75,
            "fallback_model": "general_model",
        },
        "models": [
            {
                "name": "coder_model",
                "path": "/absolute/path/to/coder.gguf",
                "anchors": [
                    "Write code or a function",
                    "Debug a script or error",
                    "Implement an algorithm or data structure",
                ],
            },
            {
                "name": "general_model",
                "path": "/absolute/path/to/general.gguf",
                "anchors": [
                    "Tell me a story",
                    "Write an email or formal letter",
                    "Explain general topics and concepts",
                ],
            },
        ],
    }
    out_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    ui.success(f"Generated template config at: {out_path.resolve()}")
    print("\nNext steps:")
    print(f"  1. Update model paths in {out_path.name}")
    print(f"  2. Validate configuration:  assembly validate -c {out_path.name}")
    print(f"  3. Pack into single file:   assembly build -c {out_path.name} -o models.asmb\n")


def _render_report(report, config_path: str) -> None:
    """Print a pre-flight validation report to the terminal."""
    name = report.config.assembly_name if report.config else Path(config_path).stem
    ui.blank()
    ui.heading(f"Pre-flight check — {name}")
    ui.rule()

    if report.models_info:
        rows = []
        for m in report.models_info:
            role = "router" if m["role"] == "router_embedding" else "model"
            anchors = f"{m['anchors_count']} anchors" if "anchors_count" in m else "—"
            rows.append([m["name"], role, f"{m['size_mb']:.1f} MB", anchors])
        ui.table("Models", ["name", "role", "size", "anchors"], rows)
        ui.blank()

        total_gb = report.total_size_bytes / 1e9
        ui.dim(f"estimated archive size  {total_gb:.2f} GB ({report.total_size_bytes:,} bytes)")
        if report.free_disk_bytes:
            ui.dim(f"available disk space    {report.free_disk_bytes / 1e9:.2f} GB")

    if report.warnings:
        ui.blank()
        for w in report.warnings:
            ui.warn(f"[{w.field}] {w.message}")

    if report.errors:
        ui.blank()
        for e in report.errors:
            ui.error(f"[{e.field}] {e.message}")
        ui.rule()
        ui.error(
            f"configuration has {len(report.errors)} error(s) — "
            "no model data was written"
        )
        return

    ui.rule()
    ui.success("configuration verified — ready for 'assembly build'")


def _cmd_validate(args: argparse.Namespace) -> None:
    """Pre-flight check of an assembly config file before building."""
    from assembly.config import validate_assembly_preflight

    report = validate_assembly_preflight(
        config_path=args.config,
        output_path=getattr(args, "output", None),
        check_gguf_magic=True,
    )
    _render_report(report, args.config)
    if not report.is_valid:
        sys.exit(1)


def _require_archive(path: str) -> str:
    """Validate the archive argument: absolute, and actually there.

    The path is required to be absolute. Guessing from the working directory
    meant that serving from the wrong folder silently loaded whatever `.asmb`
    happened to be sitting in it -- which for a server that then hands the
    result to a remote client is the wrong failure mode. An explicit absolute
    path is unambiguous about which archive is about to be parsed.
    """
    resolved = Path(path).expanduser()

    if not resolved.is_absolute():
        ui.error(f"the archive path must be absolute: {path}")
        ui.blank()
        ui.dim("a relative path depends on which directory you happen to be in,")
        ui.dim("so the server could load a different archive than you expect.")
        ui.blank()
        ui.dim("fix it with the absolute path:")
        ui.dim(f"  -m {resolved.resolve()}")
        sys.exit(1)

    if not resolved.is_file():
        ui.error(f"no such archive: {resolved}")
        ui.blank()
        ui.dim("build one:")
        ui.dim("  assembly init")
        ui.dim(f"  assembly build -c assembly.config.json -o {resolved.name}")
        ui.blank()
        ui.dim("already have one? point at it with -m /path/to/file.asmb")
        sys.exit(1)

    return str(resolved)


def _require_model_argument(args: argparse.Namespace) -> str:
    """The archive a command must be told about, validated."""
    if not args.model:
        ui.error("no archive given: -m is required.")
        ui.blank()
        ui.dim("pass an absolute path to the archive:")
        ui.dim(f"  assembly {args.command} -m /path/to/models/my_models.asmb")
        sys.exit(1)
    return _require_archive(args.model)


def _cmd_serve(args: argparse.Namespace) -> None:
    """Handle ``assembly serve``."""
    from assembly.server import serve

    model_path = _require_model_argument(args)

    serve(
        model_path=model_path,
        host=args.host,
        port=args.port,
        persona=args.persona,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.verbose,
        api_key=args.api_key,
        allow_origin=args.allow_origin,
        models_dir=args.models_dir,
        temp_dir=args.temp_dir,
    )


def _cmd_run(args: argparse.Namespace) -> None:
    """Handle ``assembly run``.

    Always against a running server. In-process inference used to be the
    fallback when no server answered, which meant a second, unhardened way in:
    no bearer token, no ``/api/load`` jail, no request-size or concurrency
    limit, and unsanitised errors. Routing every request through ``assembly
    serve`` means one code path, and the one with the limits on it.
    """
    prompt = args.prompt or args.prompt_pos
    server_url = args.server or "http://127.0.0.1:8000"

    if not _is_server_available(server_url):
        ui.error(f"no Assembly server at {server_url}")
        ui.dim(f"start one:  assembly serve -m /path/to/models.asmb --host "
                f"{urlparse_host(server_url)}")
        sys.exit(1)

    # -m is either an alias the server already has loaded, or a path to ask it
    # to load. Loading is keyed on the .asmb suffix so that a plain alias does
    # not provoke a pointless /api/load request and the warning that follows.
    if args.model and Path(args.model).expanduser().suffix == ".asmb":
        _load_on_server(server_url, args.model, args.persona)

    backend = _ServerBackend(server_url, args.model)
    _start_session(backend, prompt, args)


def _start_session(backend: "_Backend", prompt: Optional[str], args: argparse.Namespace) -> None:
    """One-shot prompt, or an interactive REPL when no prompt was given."""
    if prompt:
        chat = _Chat(backend, max_tokens=args.max_tokens, show_route=args.route,
                     memory=not args.no_memory)
        chat.ask(prompt)
        return
    _repl(backend, max_tokens=args.max_tokens, show_route=args.route,
          memory=not args.no_memory)


def _is_server_available(server_url: str) -> bool:
    import urllib.request

    try:
        req = urllib.request.Request(f"{server_url.rstrip('/')}/health")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def urlparse_host(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).hostname or "127.0.0.1"


def _load_on_server(
    server_url: str, model_path: str, persona: Optional[str] = None
) -> None:
    import urllib.request

    try:
        payload = {"model_path": str(Path(model_path).resolve())}
        if persona:
            payload["persona"] = persona
        req = urllib.request.Request(
            f"{server_url.rstrip('/')}/api/load",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30):
            pass  # response body is irrelevant; a 2xx is the whole signal
    except Exception as exc:
        ui.warn(f"Could not request server to load model: {exc}")


# ─────────────────────────────────────────────────────────────────────
# Chat backends — one interface, two transports
# ─────────────────────────────────────────────────────────────────────


def _router_line(embedding: Any) -> str:
    """``router · threshold 0.75 · fallback General``.

    Accepts either a header ``EmbeddingEntry`` or the server's JSON dict.
    """
    if not embedding:
        return "unknown"
    if isinstance(embedding, str):
        return embedding

    def get(key: str, default: Any = None) -> Any:
        if isinstance(embedding, dict):
            return embedding.get(key, default)
        return getattr(embedding, key, default)

    bits = [str(get("name", "router"))]
    threshold = get("confidence_threshold")
    if threshold is not None:
        bits.append(f"threshold {threshold}")
    fallback = get("fallback_model")
    if fallback:
        bits.append(f"fallback {fallback}")
    return " · ".join(bits)


class _Backend:
    """What a chat transport must be able to do."""

    def describe(self) -> str:
        """Short human label, e.g. ``Mesh 1.0 · Code, General, Reasoning``."""
        raise NotImplementedError

    def stream(
        self, messages: list[dict[str, str]], sampling: "SamplingParams"
    ) -> Iterator[str]:
        """Yield assistant text incrementally for *messages*."""
        raise NotImplementedError

    def was_summarized(self) -> bool:
        """Whether the last request had its history compressed."""
        return False

    def context_window(self) -> int:
        """The context window in use, for display."""
        return 0

    def route(self, text: str) -> tuple[str, float, bool]:
        """Return ``(model_name, score, is_fallback)`` without generating."""
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        """Structured details for the ``/info`` command."""
        raise NotImplementedError

    def warm(self, text: str) -> None:
        """Pre-load whatever *text* will need. Optional; default is a no-op."""


class _ServerBackend(_Backend):
    """Inference over HTTP against a running ``assembly serve``."""

    def __init__(self, url: str, model: Optional[str] = None) -> None:
        self.url = url.rstrip("/")
        self.alias = Path(model).stem if model else None

    def _post(self, path: str, payload: dict[str, Any], timeout: int = 300) -> Any:
        import urllib.request

        req = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def describe(self) -> str:
        try:
            import urllib.request

            with urllib.request.urlopen(f"{self.url}/api/info", timeout=10) as resp:
                info = json.loads(resp.read().decode("utf-8"))
            names = ", ".join(m["name"] for m in info.get("models", []))
            return f"{info.get('assembly_name', 'assembly')} · {names} · {self.url}"
        except Exception:
            return f"{self.url} ({self.alias})" if self.alias else self.url


    def info(self) -> dict[str, Any]:
        import urllib.request

        with urllib.request.urlopen(f"{self.url}/api/info", timeout=10) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        emb = raw.get("embedding", {}) or {}
        models = []
        for m in raw.get("models", []) or []:
            entry = {"name": m.get("name", "?")}
            if m.get("anchors") is not None:
                entry["anchors"] = len(m.get("anchors") or [])
            models.append(entry)

        return {
            "assembly": raw.get("assembly_name", "assembly"),
            "source": self.url,
            "persona": raw.get("persona"),
            "router": _router_line(emb),
            "models": models,
        }

    def route(self, text: str) -> tuple[str, float, bool]:
        data = self._post("/api/route", {"prompt": text}, timeout=60)
        route = data.get("route", data)
        return (
            route.get("selected_model", route.get("model", "?")),
            float(route.get("score", 0.0)),
            bool(route.get("is_fallback", False)),
        )

    def stream(
        self, messages: list[dict[str, str]], sampling: SamplingParams
    ) -> Iterator[str]:
        """Consume the server's OpenAI-compatible SSE stream."""
        import urllib.request

        payload: dict[str, Any] = {
            "messages": messages,
            "stream": True,
            **sampling.to_dict(),
        }
        if self.alias:
            payload["model"] = self.alias

        req = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    token = delta.get("content") or choice.get("text") or ""
                    if token:
                        yield token


# ─────────────────────────────────────────────────────────────────────
# Conversation
# ─────────────────────────────────────────────────────────────────────


class _Chat:
    """Holds the message history and turns one prompt into one answer."""

    def __init__(
        self,
        backend: _Backend,
        max_tokens: int = 256,
        show_route: bool = False,
        memory: bool = True,
        sampling: SamplingParams | None = None,
    ) -> None:
        self.backend = backend
        self.sampling = sampling or SamplingParams(max_tokens=max_tokens)
        self.max_tokens = self.sampling.max_tokens
        self.show_route = show_route
        self.memory = memory
        self.messages: list[dict[str, str]] = []

    def clear(self) -> None:
        self.messages.clear()

    def ask(self, text: str) -> str:
        """Send *text*, stream the answer to stdout, return the full reply."""
        if self.memory:
            self.messages.append({"role": "user", "content": text})

        if self.show_route:
            name, score, is_fallback = self.backend.route(text)
        else:
            name, score, is_fallback = "", 0.0, False

        self.backend.warm(text)
        if self.show_route:
            ui.dim(ui.badge_router(name, score, is_fallback))

        payload = list(self.messages) if self.memory else [
            {"role": "user", "content": text}
        ]

        chunks: list[str] = []
        announced = False
        try:
            for token in self.backend.stream(payload, self.sampling):
                # The archive compresses history as soon as it approaches the
                # context window. Saying so once, on the first token, keeps the
                # answer on stdout while explaining a memory the model no
                # longer has verbatim.
                if not announced and self.backend.was_summarized():
                    announced = True
                    ui.dim(
                        f"  ↻ earlier turns summarised to fit the "
                        f"{self.backend.context_window():,}-token context window"
                    )
                chunks.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()
        except KeyboardInterrupt:
            # Stop generating but keep the session alive.
            sys.stdout.write("\n")
            sys.stdout.flush()
        print()

        reply = "".join(chunks)
        if self.memory and reply:
            self.messages.append({"role": "assistant", "content": reply})
        return reply


# ─────────────────────────────────────────────────────────────────────
# Interactive session
# ─────────────────────────────────────────────────────────────────────

_HELP = """\
/help              show this
/info              archive, router and model details
/model             models in the assembly and which are loaded
/route <text>      show which model would answer, without generating
/persona           show the active persona
/params            sampling settings, shown as meters
/temp <n>          randomness            (0.0-2.0, default 0.8)
/top_p <n>         nucleus sampling       (0.0-1.0, default 0.95)
/top_k <n>         top-k cutoff           (1-100, default 40)
/repeat <n>        repeat penalty        (1.0-2.0, default 1.1)
/presence <n>      presence penalty      (-2.0 to 2.0, default 0)
/frequency <n>     frequency penalty     (-2.0 to 2.0, default 0)
/max <n>           max tokens per reply   (16-8192, default 512)
/stop <text>       stop sequence         (/stop clear to remove)
/preset <name>     precise | balanced | creative
/clear             forget the conversation
/bye               exit\
"""


# Short forms that jump straight into /params, e.g. `/temp 1.2`.
_PARAM_ALIASES = frozenset({
    "temp", "temperature", "top_p", "topp", "p", "top_k", "topk", "k",
    "repeat", "rep", "presence", "frequency", "max", "tokens", "len",
    "stop", "preset",
})

# Numeric knobs, addressable by short or literal name.
_SAMPLING_ALIASES = {
    "temp": "temperature", "temperature": "temperature",
    "top_p": "top_p", "topp": "top_p", "p": "top_p",
    "top_k": "top_k", "topk": "top_k", "k": "top_k",
    "repeat": "repeat_penalty", "rep": "repeat_penalty",
    "repeat_penalty": "repeat_penalty",
    "presence": "presence_penalty", "presence_penalty": "presence_penalty",
    "frequency": "frequency_penalty", "frequency_penalty": "frequency_penalty",
    "max": "max_tokens", "max_tokens": "max_tokens",
    "tokens": "max_tokens", "len": "max_tokens",
}


def _print_sampling(params: SamplingParams) -> None:
    """Render the sampler settings as meters.

    Values that differ from the balanced preset are flagged inline, so it is
    obvious which knob a session has wandered away from.
    """
    defaults = SAMPLING_PRESETS["balanced"]
    for field, (label, low, high) in SAMPLING_FIELDS.items():
        value = getattr(params, field)
        marker = defaults[field]
        suffix = "" if value == marker else f"    balanced uses {marker}"
        ui.meter(label, float(value), low, high, suffix=suffix)

    ui.dim(
        f"  {'stop':<16}"
        + (", ".join(repr(s) for s in params.stop) if params.stop else "(none)")
    )


def _cmd_params(chat: "_Chat", rest: str) -> None:
    """``/params`` plus its per-knob subcommands."""
    if not rest:
        ui.blank()
        _print_sampling(chat.sampling)
        return

    parts = rest.split(None, 1)
    key = parts[0].lower()
    value = parts[1].strip() if len(parts) > 1 else ""

    if key == "preset":
        if not value:
            ui.dim("usage: /preset precise|balanced|creative")
            return
        try:
            chosen = sampling_params_from_preset(value)
        except KeyError:
            ui.error(f"unknown preset: {value}")
            ui.dim("available: " + ", ".join(SAMPLING_PRESETS))
            return
        chat.sampling = chosen
        chat.max_tokens = chosen.max_tokens
        ui.blank()
        ui.success(f"preset {value.lower()}")
        _print_sampling(chosen)
        return

    if key == "stop":
        if not value or value.lower() in ("clear", "none", "off"):
            chat.sampling.stop = []
            ui.dim("stop sequences cleared")
            return
        chat.sampling.stop = [value]
        ui.dim(f"stopping on {value!r}")
        return

    # Numeric knobs, addressable by short or literal name.
    field = _SAMPLING_ALIASES.get(key)
    if field is None:
        ui.error(f"unknown setting: {key}")
        ui.dim("try: /temp 1.2   /top_p 0.9   /preset creative")
        return
    if not value:
        ui.dim(f"usage: /{key} <value>   current: {getattr(chat.sampling, field)}")
        return

    label, low, high = SAMPLING_FIELDS[field]
    try:
        number = float(value.replace(",", "."))
    except ValueError:
        ui.error(f"{label} needs a number, got {value!r}")
        return
    if number != number:  # NaN
        ui.error(f"{label} needs a number, got {value!r}")
        return

    clamped = max(low, min(high, number))
    if field in ("top_k", "max_tokens"):
        clamped = int(clamped)
    setattr(chat.sampling, field, clamped)

    if clamped != number:
        ui.warn(f"{label} clamped to {clamped} (allowed {low:g} to {high:g})")
    else:
        ui.dim(f"{label} = {clamped}")

    if field == "max_tokens":
        chat.max_tokens = clamped

    ui.blank()
    _print_sampling(chat.sampling)


def _repl(backend: _Backend, max_tokens: int, show_route: bool, memory: bool) -> None:
    """A quiet, ollama-flavoured chat loop: an arrow, a question, an answer."""
    chat = _Chat(backend, max_tokens=max_tokens, show_route=show_route, memory=memory)
    _enable_readline()

    ui.dim(backend.describe())
    if not show_route:
        ui.dim("/help for commands · Ctrl+C or Ctrl+D to exit")
    print()

    while True:
        try:
            line = input(ui.prompt_text())
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            return

        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            if _handle_command(chat, line):
                return
            print()
            continue

        chat.ask(line)
        print()


def _print_info(backend: _Backend) -> None:
    info = backend.info()
    rows = []
    for label, key in (
        ("assembly", "assembly"),
        ("source", "source"),
        ("router", "router"),
    ):
        if info.get(key):
            rows.append([label, str(info[key])])
    persona = info.get("persona")
    if persona:
        short = persona if len(persona) <= 90 else persona[:90] + "…"
        rows.append(["persona", short])

    if rows:
        ui.table("Assembly", ["field", "value"], rows)

    if info.get("models"):
        ui.blank()
        _print_models(backend)


def _print_models(backend: _Backend) -> None:
    rows = []
    for item in backend.info().get("models") or []:
        name = item.get("name", "?")
        bits = []
        if item.get("size"):
            bits.append(f"{item['size'] / 1e9:.2f} GB")
        if item.get("anchors") is not None:
            bits.append(f"{item['anchors']} anchors")
        rows.append([
            name,
            "loaded" if item.get("loaded") else "idle",
            " · ".join(bits) or "—",
        ])
    if rows:
        ui.table("Models", ["name", "state", "detail"], rows)


def _handle_command(chat: _Chat, line: str) -> bool:
    """Run a ``/command``. Returns True when the session should end."""
    parts = line[1:].split(None, 1)
    name = (parts[0] if parts else "").lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if name in ("bye", "exit", "quit", "q"):
        return True

    if name in ("help", "?", "h"):
        print()
        print(_HELP)
        return False

    if name == "clear":
        chat.clear()
        ui.dim("conversation cleared")
        return False

    if name == "info":
        print()
        try:
            _print_info(chat.backend)
        except Exception as exc:
            ui.error(str(exc))
        return False

    if name == "model":
        print()
        try:
            _print_models(chat.backend)
        except Exception as exc:
            ui.error(str(exc))
        return False

    if name == "route":
        if not rest:
            ui.dim("usage: /route <your text>")
            return False
        try:
            model, score, is_fallback = chat.backend.route(rest)
        except Exception as exc:
            ui.error(str(exc))
            return False
        print()
        ui.dim(ui.badge_router(model, score, is_fallback))
        return False

    if name == "persona":
        print()
        ui.dim(chat.backend.info().get("persona") or "(none)")
        return False

    if name == "params" or name in _PARAM_ALIASES:
        rest_for_params = rest if name == "params" else f"{name} {rest}".strip()
        _cmd_params(chat, rest_for_params)
        return False

    ui.error(f"unknown command: /{name}")
    ui.dim("/help to see what works")
    return False


def _enable_readline() -> None:
    """Line editing plus persistent input history when readline is present.

    readline is part of the standard library on macOS and Linux but not on
    Windows, where it ships as a separate distribution. Its absence is not an
    error: the REPL still works, just without arrow-key editing or history. Say
    so once, because the difference is otherwise invisible until someone
    presses Up and nothing happens.
    """
    try:
        import readline
    except ImportError:
        if os.name == "nt":
            ui.dim(
                "readline is unavailable, so this session has no line editing "
                "or saved history. `pip install pyreadline3` enables both."
            )
        return

    hist_path = Path(os.path.expanduser("~")) / ".assembly_history"
    try:
        if hist_path.exists():
            readline.read_history_file(str(hist_path))
    except OSError:
        pass

    try:
        readline.set_history_length(1000)
    except (AttributeError, OSError):
        pass

    def _save() -> None:
        try:
            readline.write_history_file(str(hist_path))
        except OSError:
            pass

    atexit.register(_save)


# ─────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assembly",
        description="Assembly Engine — multi-model GGUF router & server",
    )
    parser.add_argument(
        "--version", action="version", version=f"assembly {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── init ─────────────────────────────────────────────────────────
    p_init = sub.add_parser("init", help="Generate starter assembly config template")
    p_init.add_argument(
        "-o", "--output", default="assembly.config.json", help="Output path (default: assembly.config.json)"
    )
    p_init.add_argument(
        "-f", "--force", action="store_true", default=False, help="Overwrite if file already exists"
    )
    p_init.set_defaults(func=_cmd_init)

    # ── validate ─────────────────────────────────────────────────────
    p_val = sub.add_parser("validate", help="Validate assembly configuration and model paths")
    p_val.add_argument(
        "-c", "--config", required=True, help="Path to assembly JSON config"
    )
    p_val.add_argument(
        "-o", "--output", default=None, help="Target .asmb output path (to check available disk space)"
    )
    p_val.set_defaults(func=_cmd_validate)

    # ── build ────────────────────────────────────────────────────────
    p_build = sub.add_parser("build", help="Pack models into .asmb archive")
    p_build.add_argument(
        "-c", "--config", required=True, help="Path to assembly JSON config"
    )
    p_build.add_argument(
        "-o", "--output", required=True, help="Output .asmb file path"
    )
    p_build.add_argument(
        "--skip-embeddings",
        action="store_true",
        default=False,
        help="Skip pre-computing anchor embeddings (testing only)",
    )
    p_build.set_defaults(func=_cmd_build)

    # ── info ─────────────────────────────────────────────────────────
    p_info = sub.add_parser("info", help="Show .asmb header metadata")
    p_info.add_argument(
        "-m", "--model", default=None, required=True,
        metavar="/path/to/models.asmb",
        help="Absolute path to the .asmb archive (required; not guessed from "
             "the current directory)",
    )
    p_info.add_argument(
        "--verify",
        action="store_true",
        help="Re-hash every blob and compare against the digests recorded in "
             "the header. Reads the whole archive, so on a multi-gigabyte file "
             "this takes a while.",
    )
    p_info.set_defaults(func=lambda args: _cmd_info(args))

    # ── serve ────────────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Start HTTP API server daemon")
    p_serve.add_argument(
        "-m", "--model", default=None, required=True,
        metavar="/path/to/models.asmb",
        help="Absolute path to the .asmb archive to load (required; not "
             "guessed from the current directory)",
    )
    p_serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1, local only). "
             "Binding a public address requires --api-key.",
    )
    p_serve.add_argument(
        "-p", "--port", type=int, default=8000, help="Port to bind (default: 8000)"
    )
    p_serve.add_argument(
        "--api-key",
        default=None,
        help="Require 'Authorization: Bearer <key>' on every request. "
             "Defaults to $ASSEMBLY_API_KEY. Mandatory when not binding loopback. "
             "ASCII only; prefer the environment variable, since a key passed on "
             "the command line is visible to other processes via ps.",
    )
    p_serve.add_argument(
        "--allow-origin",
        default=None,
        help="CORS origin allowed to call this server, e.g. http://localhost:5173. "
             "Omitted by default: no browser origin may call it.",
    )
    p_serve.add_argument(
        "--models-dir",
        default=None,
        help="Directory the /api/load endpoint may read .asmb files from. "
             "Defaults to the directory holding the served archive, so a "
             "client can load its siblings but nothing else on the disk. "
             "Widening this to a shared directory such as the system temp "
             "directory hands every client on the host the ability to open any "
             "readable .asmb; prefer a directory you own and serve from there.",
    )
    p_serve.add_argument(
        "--temp-dir",
        default=None,
        help="Directory used to stage model blobs (default: $ASSEMBLY_TEMP_DIR "
             "or the system temp directory). Point this at a fast volume with "
             "room for the largest model.",
    )
    p_serve.add_argument(
        "--persona", default=None, help="Override or set assistant persona"
    )
    p_serve.add_argument(
        "--n-gpu-layers", type=int, default=0, help="GPU layer offload count"
    )
    p_serve.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Verbose logging"
    )
    p_serve.set_defaults(func=_cmd_serve)

    # ── run ──────────────────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help="Chat with an assembly",
        description="Chat with an .asmb archive on a running server. With no "
                    "prompt it opens an interactive session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
requires a running server; there is no standalone inference mode.

examples:
  assembly serve -m /path/to/mesh.asmb           start the server first
  assembly run                                   open a chat session
  assembly run -m /path/to/other.asmb           ask the server to load it
  assembly run "hello"                           one-shot answer
  assembly run -p "hi" --route                   also show which model answered
  assembly run --no-memory                       forget context each turn
  assembly run -s http://192.168.1.50:8000       use a server elsewhere

session commands: /help /info /model /route /clear /persona /bye
""",
    )
    p_run.add_argument(
        "prompt_pos", nargs="?", metavar="prompt", default=None,
        help="Prompt to answer immediately (same as -p)",
    )
    p_run.add_argument(
        "-m", "--model", default=None, metavar="MODEL",
        help="Absolute path to an .asmb the server should load, or the alias "
             "of one it already has. An argument ending in .asmb is loaded; "
             "anything else is used as an alias.",
    )
    p_run.add_argument(
        "-s", "--server", default=None, metavar="URL",
        help="URL of a running Assembly server (default: http://localhost:8000)",
    )
    p_run.add_argument("-p", "--prompt", default=None, help="Input prompt")
    p_run.add_argument(
        "--persona",
        default=None,
        help="Override or set assistant persona/identity",
    )
    p_run.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens to generate"
    )
    p_run.add_argument(
        "--route", action="store_true",
        help="Show which sub-model answered each turn",
    )
    p_run.add_argument(
        "--no-memory", action="store_true",
        help="Send each prompt on its own, without conversation history",
    )
    p_run.set_defaults(func=_cmd_run)


    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
    except KeyboardInterrupt:
        # Ctrl+C is a normal way to leave a session or stop a server.
        print()
        sys.exit(130)
    except BrokenPipeError:
        # e.g. `assembly run ... | head`: exit quietly.
        try:
            sys.stdout.close()
        except OSError:
            pass
        sys.exit(0)
    except (InvalidArchiveError, RuntimeError, OSError) as exc:
        # InvalidArchiveError is a ValueError; StagingError is a RuntimeError,
        # so this covers both without importing the heavier wrapper module.
        # These are all failures a user can act on. A traceback here would only
        # bury the one useful sentence in a wall of library frames.
        if type(exc) is RuntimeError and not str(exc):
            raise  # not one of ours; keep the traceback
        ui.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
