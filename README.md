# Assembly Engine

**Multi-model GGUF router with llama-cpp-python**

[![CI](https://github.com/assembly-engine/assembly/actions/workflows/ci.yml/badge.svg)](https://github.com/assembly-engine/assembly/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Assembly Engine combines 1 embedding model (router) and N execution models (≥ 2) into a single binary `.asmb` file. When a prompt arrives, the engine computes a fast embedding, measures Cosine Similarity against pre-defined text anchors, selects the best model via `mmap`, and generates the response.

Two commands matter: **`assembly serve`** hosts the archive over HTTP, and
**`assembly run`** talks to it from any device. The rest (`init`, `validate`,
`build`, `info`) exist to get an archive built in the first place.

> **Note on disk usage.** On Linux the archive is memory-mapped and model blobs
> are handed to `llama.cpp` as anonymous in-memory files. On macOS and Windows
> `llama.cpp` requires a real path, so each model is staged as a temporary file
> on first use. The archive is never extracted, but the first run does read each
> model from disk. See [staging](#staging-and-disk-usage).

## Terminal output

Rendering is handled by [`rich`](https://github.com/Textualize/rich) with a
single orange palette; cream is reserved for the model's own text so a reply
is never confused for chrome.

- **stdout is only ever the answer.** Banners, spinners, warnings and routing
  badges all go to stderr, so `assembly run -m /path/to/models.asmb "hi" | pbcopy`
  yields the reply and nothing else.
- Honours `NO_COLOR` and non-TTY output — redirected output contains no escape
  sequences at all.
- Set `ASSEMBLY_NO_SPINNER=1` to replace spinners with static lines.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   .asmb Archive                     │
├─────────────────────────────────────────────────────┤
│  HEADER  (magic + version + JSON with offsets)      │
│  ── page-aligned padding ──                         │
│  [Embedding GGUF]   ← router model (small)         │
│  ── page-aligned padding ──                         │
│  [Model A GGUF]     ← execution model #1           │
│  ── page-aligned padding ──                         │
│  [Model B GGUF]     ← execution model #2           │
│  ...                                                │
└─────────────────────────────────────────────────────┘

Prompt → Embed → Cosine Similarity → Select Model → Generate
```

## Installation

```bash
pip install -e .
```

Requires Python 3.10+ and works on Linux, macOS and Windows. Needs
`llama-cpp-python`, `numpy` and `rich`. `llama-cpp-python` compiles from source
on most platforms; install a prebuilt wheel for yours if a C++ toolchain is
unavailable — on Windows that means Visual Studio Build Tools, and a wheel is
much the better option. See [Windows notes](#windows-notes).

On Windows, add the `history` extra for REPL line editing and saved history:

```bash
pip install -e ".[history]"
```

Check the install with:

```bash
assembly --version
```

## Quick Start

### 1. Generate & Validate Config

You can scaffold a clean config template with one command:

```bash
# Generate starter assembly.config.json
assembly init

# Validate paths, file sizes, and anchors before building
assembly validate -c assembly.config.json
```

`validate` runs a fast pre-flight check (< 100 ms, no model loading) and reports
every problem it finds instead of stopping at the first one:

- JSON syntax (with line/column) and schema errors
- absolute paths, file existence, empty files
- GGUF magic bytes — catches truncated downloads or saved HTML error pages
- duplicate model names, missing/duplicate anchors, invalid `fallback_model`
- free disk space vs. the required archive size
- output path sanity (missing directory, not writable, clobbering an input)

Errors block the build; warnings (e.g. only one anchor on a model) are advisory.

Example `assembly.config.json`:

```json
{
  "assembly_name": "Assembly",
  "persona": "You are Assembly, a unified AI assistant. You always speak as Assembly and never reveal that you are composed of separate sub-models (like Qwen, SmolLM, etc.). Maintain a helpful and consistent identity across all tasks.",
  "embedding": {
    "name": "bge-small-router",
    "path": "/absolute/path/to/embedding_model.gguf",
    "confidence_threshold": 0.75,
    "fallback_model": null
  },
  "models": [
    {
      "name": "coder_model",
      "path": "/absolute/path/to/coder.gguf",
      "anchors": [
        "Napisz funkcję w Pythonie",
        "Zdebuguj kod"
      ]
    },
    {
      "name": "general_model",
      "path": "/absolute/path/to/general.gguf",
      "anchors": [
        "Opowiedz historię",
        "Napisz e-mail"
      ]
    }
  ]
}
```

### 2. Build the archive

```bash
assembly build -c config.json -o my_models.asmb
```

`build` runs the same pre-flight check automatically **before** loading or copying
anything, so a broken config fails in milliseconds instead of after a multi-hour
copy. Nothing is written to the output file unless validation passes.

Validation also catches models that could never load:

- a file that is not a real GGUF (corrupt download),
- **a shard of a split GGUF** (`model-00001-of-00003.gguf`). Large models are
  often published as shards; llama.cpp refuses to load one alone, so packing
  it produces an archive that builds cleanly and then fails at inference.
  Merge the shards into a single `.gguf` first,
- a model file that changed or vanished since validation.

#### An interrupted build does not leave a broken archive

Packing several gigabytes takes a while, and a build can be interrupted — by
Ctrl-C, by a full disk, or by the machine going down. The archive is assembled
under a temporary name and renamed into place only once every byte is written
and the header has been read back off disk:

```
my_models.asmb.partial-48213      ← being written; not a usable archive
my_models.asmb                    ← appears only when the build succeeds
```

So there is never a half-written `.asmb` masquerading as a valid one, and
Ctrl-C leaves nothing behind at all.

A `kill -9` or a power cut cannot run that cleanup, so the abandoned temporary
file is instead reclaimed by the **next** build, which recognises it and
removes it:

```
  ↳ Reclaimed 0.22 GiB from an interrupted build (my_models.asmb.partial-48213)
```

Only files whose owning process is definitely gone are removed, so a build
running in another terminal is never disturbed.

### 3. Start the Server Daemon (`assembly serve`)

Like `ollama serve`, you can start an HTTP server that runs in the background or on a dedicated machine/server:

```bash
# Start server with an active .asmb archive (binds to 127.0.0.1:8000 by default)
assembly serve -m /path/to/my_models.asmb

# Custom host and port
assembly serve -m /path/to/my_models.asmb --host 127.0.0.1 --port 9000
```

`-m` is required and must be an **absolute path**. Assembly will not guess the
archive from your working directory — serving from the wrong folder used to load
whatever `.asmb` happened to be sitting in it, and then hand that archive to
every client that asked.

Other clients on this machine can now reach it:

```bash
curl http://127.0.0.1:8000/v1/models
```

#### Exposing the server to a network

**The server binds to loopback only, by default.** Anything that is not
loopback — `--host 0.0.0.0`, a LAN address, a public interface — is refused
unless you also supply an API key, because an unauthenticated inference
endpoint on a network is open to anyone who can reach the port:

```bash
export ASSEMBLY_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
assembly serve -m /path/to/my_models.asmb --host 0.0.0.0
```

Clients then send it as a bearer token:

```bash
curl http://192.168.1.50:8000/v1/chat/completions \
  -H "Authorization: Bearer $ASSEMBLY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"hi"}]}'
```

The key can also be passed with `--api-key`. Other safety limits apply
regardless of address: request bodies are capped at 1 MiB, socket reads time
out, and the number of concurrently loaded models is bounded.

| Flag | Purpose |
| --- | --- |
| `--api-key KEY` | Require `Authorization: Bearer KEY` (or set `$ASSEMBLY_API_KEY`). ASCII only; the env var is preferable, since argv is visible to other processes |
| `--allow-origin ORIGIN` | Allow one CORS origin, e.g. `http://localhost:5173`. Omitted by default: no browser origin may call the server |
| `--models-dir DIR` | Confine the `/api/load` endpoint to `.asmb` files inside `DIR`. Defaults to the served archive's directory |
| `--temp-dir DIR` | Where model blobs are staged off Linux (see below) |

See [Security model](#security-model) for what these do and do not cover —
in particular, there is no TLS, so anything beyond loopback needs a proxy.

#### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health`, `/` | Status, version, loaded count, active alias, context window |
| `GET` | `/v1/models`, `/api/tags` | OpenAI/Ollama-compatible model listing |
| `GET` | `/api/models`, `/api/info` | Loaded archives; archive header details |
| `POST` | `/api/load` | Load an `.asmb` (`model_path`, optional `alias`, `persona`) |
| `POST` | `/api/unload` | Free a slot (`alias`); returns the aliases still loaded |
| `POST` | `/api/route` | Router decision for a prompt, no generation |
| `POST` | `/v1/chat/completions`, `/api/chat` | OpenAI/Ollama-compatible chat |
| `POST` | `/v1/completions`, `/api/generate` | Raw completion |

#### Context window

The window is not a setting. Assembly reads each model's GGUF metadata at load
time and uses the **smallest** context length declared by any of the archive's
execution models, so a request is never sized for a model that cannot hold it:

```bash
assembly serve -m /path/to/my_models.asmb   # banner shows: Context  131,072 tokens
```

There is deliberately no flag to override it. A caller-chosen window can
exceed what a bundled model was trained for, which fails late and
confusingly, or undershoot it, which discards capability for nothing. The
router's embedding model is excluded, because it embeds single sentences and
is typically built with a tiny window of its own — including it would cap the
whole assembly at the embedding model's window rather than the chat models'.

The current window is also reported on `/health`, as `context_window`.

#### Long conversations are summarised, not refused

When a conversation approaches 80% of the window, Assembly asks the model to
summarise the earlier turns and continues from that summary plus the most
recent exchanges, instead of failing. Summarisation happens in stages when the
history is long enough that one request cannot hold it, so the beginning of the
conversation reaches the summary rather than being trimmed away:

```
▸ What were the first two deploy keys?
  ↻ earlier turns summarised to fit the 131,072-token context window
The first two were secret-000 and secret-001.
```

The note goes to stderr, so the answer on stdout stays pipeable. HTTP clients
learn about it too, so they can stop replaying turns that are no longer sent:

```json
{"assembly_context": {"window": 131072, "summarized": true}}
```

Streaming responses say so too, in a final `assembly_context`-only frame just
before `[DONE]`. It carries no `choices`, so it cannot be mistaken for content:

```
data: {"id":"assembly-context","object":"chat.completion.chunk","model":"default:General","assembly_context":{"window":131072,"summarized":true},"choices":[]}

data: [DONE]
```

If summarisation fails for any reason the original history is used unchanged,
so it never becomes a second way for a request to fail.

### 4. Run Inference (`assembly run`)

`assembly run` automatically connects to your running server, or runs locally if standalone:

```bash
# Connect to a running server on localhost:8000, or to another device on your LAN:
assembly run -m /path/to/my_models.asmb            # interactive session
assembly run -s http://192.168.1.50:8000  # remote server

# One-shot: answer a single prompt and exit
assembly run -m /path/to/my_models.asmb "Write quicksort in Python"
assembly run -m /path/to/my_models.asmb -p "Kim jesteś?"

# Show which sub-model answered each turn
assembly run -m /path/to/my_models.asmb --route

# Forget the conversation between turns
assembly run -m /path/to/my_models.asmb --no-memory
```

The session is deliberately plain — an arrow, your question, the answer:

```
$ assembly run -m /path/to/my_models.asmb
Mesh-1.0 · Code, General, Reasoning
/help for commands · Ctrl+C or Ctrl+D to exit

▸ My name is Cartofell, remember that.
Hello Cartofell! I'll remember that for this conversation.

▸ What is my name?
Cartofell

▸ /clear
conversation cleared
```

Type `/help` for the command list:

| Command | What it does |
|---|---|
| `/help` | show the command list |
| `/info` | archive, router, persona and model details |
| `/model` | models in the assembly and which are loaded |
| `/route <text>` | show which model *would* answer, without generating |
| `/persona` | show the active persona |
| `/params` | sampling settings, drawn as bars |
| `/temp <n>` | randomness, `0.0`–`2.0` |
| `/top_p <n>` | nucleus sampling, `0.0`–`1.0` |
| `/top_k <n>` | top-k cutoff, `1`–`100` |
| `/repeat <n>` | repeat penalty, `1.0`–`2.0` |
| `/presence <n>` | presence penalty, `-2.0`–`2.0` |
| `/frequency <n>` | frequency penalty, `-2.0`–`2.0` |
| `/max <n>` | max tokens per reply, `16`–`8192` |
| `/stop <text>` | stop sequence (`/stop clear` to remove) |
| `/preset <name>` | `precise`, `balanced`, or `creative` |
| `/clear` | forget the conversation |
| `/bye` | exit (also `Ctrl+C` / `Ctrl+D`) |

### Sampling

`/params` shows the current settings as bars, so you can see at a glance how far
a value sits in its range. Anything not matching the `balanced` preset is
labelled:

```
▸ /params
  temperature     ██████████··············  0.8
  top_p           ███████████████████████·  0.95
  top_k           █████████···············  40
  repeat penalty  ██······················  1.1
  presence pen    ············│···········  0
  frequency pen   ············│···········  0
  max tokens      █·······················  512
  stop            (none)
```

Values are clamped to their range rather than rejected, and decimal commas work
(`/temp 1,6`). Presets are just bundles:

| Preset | Use it for |
|---|---|
| `precise` | code and factual work; low temperature, longer answers |
| `balanced` | the default for chat |
| `creative` | prose and brainstorming |

The same settings are accepted by the HTTP API on `/v1/chat/completions` and
`/v1/completions`, using OpenAI's names plus llama.cpp's `top_k`:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"assembly","messages":[{"role":"user","content":"hi"}],
       "temperature":0.2,"top_p":0.9,"top_k":20,"max_tokens":1024,
       "presence_penalty":0.5,"frequency_penalty":0.5,"stop":["\n\n"]}'
```

Unsupported values fall back to the default rather than failing the request, so
a client sending `temperature: "hot"` still gets an answer. `seed` is passed
through when present, which makes runs reproducible.

`Ctrl+C` during a reply stops generating but keeps the session open. Progress
(archive loading, sub-model warm-up) goes to stderr, so piping stdout gives you
the answer and nothing else:

```bash
assembly run -m /path/to/my_models.asmb "hi" 2>/dev/null | pbcopy
```

### 5. Multi-device & External Scripts Access

The server exposes standard, universal APIs. On a loopback server with no API
key configured, no credentials are needed. On a server started with
`--api-key` (which is mandatory for any non-loopback bind), every request must
carry that key as a bearer token — including these examples.

#### A. OpenAI Python SDK (works from any machine/script)
```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.1.50:8000/v1",       # IP of your Assembly server
    api_key=os.environ["ASSEMBLY_API_KEY"],      # the real key; any value
                                                   # only if the server has none
)

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Napisz funkcję w Pythonie"}],
)
print(response.choices[0].message.content)
```

#### B. cURL / Any HTTP Client (Linux, macOS, Windows, Mobile)
```bash
# OpenAI-compatible Chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $ASSEMBLY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages": [{"role": "user", "content": "Tell me a joke"}]}'

# Ollama-compatible API
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer $ASSEMBLY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages": [{"role": "user", "content": "Hello!"}]}'

# Check router decision without full generation
curl -X POST http://localhost:8000/api/route \
  -H "Authorization: Bearer $ASSEMBLY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Debug this memory leak"}'
```

The server speaks JSON only. There is no web UI — the terminal is the
interface, and every endpoint is a plain API for your own scripts.

### 6. Inspect an archive

```bash
assembly info -m /path/to/my_models.asmb
```

## Security model

Read this before exposing the server to anything other than your own machine.
It states what the server does and does not protect against.

### Intended deployment

A **single-user, single-tenant, local-first** service: one operator, on their
own machine, over a socket they control. It is not hardened as a shared or
public service, and the limits below are the honest edges of that.

### What the server does enforce

| Control | Behaviour |
| --- | --- |
| Bind address | Loopback (`127.0.0.1`) by default. A non-loopback `--host` is **refused at startup** unless a key is set. |
| Authentication | One bearer token, compared in constant time. Optional on loopback, mandatory off it. |
| `/api/load` jail | Confined to `--models-dir`, which defaults to the served archive's own directory. Resolved, so `..` and symlinks cannot escape. |
| Browser access | `Host` must name loopback; no CORS origin is allowed unless you pass `--allow-origin`; cross-origin writes are refused. Together these close the DNS-rebinding and CSRF routes into a keyless local server. |
| Capacity | At most 8 loaded archives and 32 concurrent connections. Further loads get `429`; further connections are closed. |
| Request size | 1 MiB body cap, checked against `Content-Length` before a byte is read. Chunked encoding is refused. |
| Stalled clients | A 300-second wall-clock deadline per connection, not just a per-recv socket timeout. |
| Error detail | Errors report a status and a reason. Staging paths, free space, and archive header offsets go to the log, never to the client. |
| Log hygiene | Prompts and tokens are never logged; attacker-supplied text (paths, aliases, origins, hosts) has control characters stripped before it reaches the log. |

### What it does **not** protect against

- **Plaintext transport.** There is no TLS. On anything but loopback, your
  bearer token and every prompt travel in cleartext, readable by anyone on the
  path. Terminate TLS in a reverse proxy, or tunnel over SSH/WireGuard. Do not
  expose the port to an untrusted network directly.
- **Brute force and rate limiting.** There is no per-client rate limit and no
  lockout. Failed attempts are logged for audit, but nothing slows guessing. A
  short, high-entropy key is what protects you.
- **Multiple users.** There are no accounts, roles, quotas, or per-user model
  limits. Anyone holding the key has the full API, including `/api/load`.
- **Concurrent generation.** Generation is serialised per archive, so a second
  request queues behind the first rather than running in parallel. With the
  connection ceiling, several long generations can queue up; there is no
  fairness or cancellation.
- **Malicious archives.** An `.asmb` is parsed by llama.cpp **in this process**.
  Only load archives you built or otherwise trust. `/api/load` restricting paths
  is not a substitute for that.
- **Argv exposure.** `--api-key` is visible to other processes via `ps`. Prefer
  `ASSEMBLY_API_KEY`, and keep it ASCII (non-ASCII keys are rejected at
  startup, because HTTP headers are Latin-1 on the wire).

### Operational notes

```bash
# Recommended: key from the environment, and a directory you own
export ASSEMBLY_API_KEY="$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')"
assembly serve -m /path/to/my_models.asmb --models-dir ~/models
```

- `assembly serve` requires `-m`, and it must be an **absolute** path. The
  archive path is never guessed from the working directory, so starting the
  server in the wrong folder cannot silently load whatever `.asmb` is there.
  Because `serve` always has an archive, `/api/load` is always confined to that
  archive's own directory unless you pass `--models-dir`. (Only a direct
  `assembly.server.serve()` call from Python can leave it unconfined.)
- Freeing a slot at the model limit: `POST /api/unload {"alias": "..."}`.
- Audit trail: failed auth, refused origins, foreign hosts, refused connections,
  and model load failures are logged at `WARNING` and above.

## Python API (In-Process)

```python
from assembly.wrapper import AssemblyLlama

with AssemblyLlama("my_models.asmb", verbose=True) as llm:
    # All underlying models share the same persona identity!
    print(llm.chat("Kim jesteś?"))  # -> "Jestem Assembly..."

    # Automatic routing — the right model is selected per prompt
    result = llm("Napisz funkcję quicksort w Pythonie", max_tokens=512)
    print(result["choices"][0]["text"])

    # Chat completion with custom messages (persona injected into system role)
    response = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": "Opowiedz mi historię o smoku"}
        ],
        max_tokens=256,
    )
```

## Routing Logic

1. The prompt is embedded via the router model.
2. Cosine Similarity is computed against every anchor of every execution model.
3. The model with the highest anchor score (`MAX_SCORE`) is identified.
4. Decision:
   - `MAX_SCORE ≥ confidence_threshold` → use that model.
   - `MAX_SCORE < threshold` **and** `fallback_model` is set → use fallback.
   - `MAX_SCORE < threshold` **and** `fallback_model` is `null` → use best anyway.

## Staging and disk usage

`.asmb` archives are memory-mapped, never extracted. How the model bytes reach
`llama.cpp` differs by platform:

- **Linux** — each blob is copied into an anonymous in-memory file
  (`memfd_create`) and passed as `/proc/self/fd/N`. There is no staging copy on
  disk at all.
- **macOS and Windows** — `llama.cpp` requires a real filesystem path, so each
  model is written to a temporary file before it is loaded. The archive itself
  is still only read; nothing is duplicated wholesale, but the models do pass
  through the temp volume, and the first `run` after a cold start will be
  slower than subsequent ones.

If the system temp volume is small or slow, point staging elsewhere. The
directory is created if needed, and free space is checked before a multi-
gigabyte write begins:

```bash
# macOS
assembly run -m /path/to/my_models.asmb --temp-dir /Volumes/Fast/assembly
# Windows
assembly run -m /path/to/my_models.asmb --temp-dir D:\NVMe\assembly
# or, for both run and serve:
export ASSEMBLY_TEMP_DIR=/Volumes/Fast/assembly   # or: set ASSEMBLY_TEMP_DIR=D:\NVMe\assembly
```

Free space is reported up front, before copying, so a volume that is too small
produces a clear message instead of an out-of-space error partway through a
load.

A directory you pass to `--temp-dir` is yours: Assembly never deletes it. The
private `assembly-*` directory it creates inside is removed at exit, along with
any staged blob still on disk.

## Windows notes

Assembly runs on Windows, macOS and Linux from the same codebase, and CI covers
all three. The Windows specifics worth knowing:

- **`llama-cpp-python` wheels.** Prefer a prebuilt wheel over a source build; it
  needs Visual Studio Build Tools otherwise. Point pip at the wheel index for
  your accelerator, for example
  `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`.
- **Terminal colour.** Modern terminals (Windows Terminal, and Windows 10+
  conhost) render the full palette. On a legacy console, or with
  `NO_COLOR=1`, output is plain.
- **REPL history.** `readline` ships with macOS and Linux but not Windows, so
  without it the REPL has no arrow-key editing or saved history. Install
  `pip install assembly-engine[history]` (or `pyreadline3`) to get both.
- **Rebuilding a served archive.** Windows will not replace a file another
  process has open, so stop `assembly serve` before rebuilding over a live
  archive. The error says so rather than reporting a bare permission failure.
- **Long paths.** Paths over 260 characters need long-path support enabled
  (Windows 10 1607+ with the registry setting, or the `\\?\` prefix). Relevant
  if your models live deep in a directory tree.

## Project Structure

```
assembly/
├── __init__.py     # Package metadata (__version__ is the single source of truth)
├── config.py       # JSON config schema & validation
├── header.py       # Binary .asmb header format
├── packer.py       # Build command — creates .asmb archives
├── router.py       # Cosine Similarity routing engine
├── wrapper.py      # AssemblyLlama — llama-cpp-python wrapper
├── server.py       # Multi-device HTTP API daemon (OpenAI & Ollama compatible)
├── ui.py           # Terminal styling, spinner and prompt
└── main.py         # CLI entry point (build / serve / run / info)
tests/
└── test_assembly.py  # unit tests
```

## Running Tests

```bash
python -m pytest tests/ -q
```

The suite runs in well under a second and loads no real models — the GGUF
blobs, HTTP server, and CLI are all exercised against fakes.

## How this was written

**This project was built with the assistance of AI coding agents.** The
architecture, the format, the router and the release decisions were made with
AI writing much of the code, and AI reviewed and extended it.

Two things follow from that, stated plainly rather than buried:

- **The code has been tested, not verified.** The suite passes and the engine
  has been exercised against real archives, but correctness rests on tests
  written largely by the same system that wrote the code. Please read
  `assembly/header.py` and `assembly/packer.py` before trusting a multi-gigabyte
  archive with your models.
- **Bugs are likely and reports are welcome.** Open an issue with the command
  you ran and what you expected. If the failure is silent — a wrong number
  rather than an error — that is the most valuable kind of report there is.

## License

MIT — see [LICENSE](LICENSE).

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
