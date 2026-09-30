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
> are handed to `llama.cpp` as anonymous in-memory files. On macOS
> `llama.cpp` requires a real path, so each model is staged as a temporary file
> on first use. The archive is never extracted, but the first run on macOS
> does read each model from disk. See [macOS staging](#macos-staging-and-disk-usage).

## Terminal output

Rendering is handled by [`rich`](https://github.com/Textualize/rich) with a
single orange palette; cream is reserved for the model's own text so a reply
is never confused for chrome.

- **stdout is only ever the answer.** Banners, spinners, warnings and routing
  badges all go to stderr, so `assembly run -m models.asmb "hi" | pbcopy`
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

Requires `llama-cpp-python`, `numpy` and `rich`. `llama-cpp-python` compiles from
source on most platforms; install a prebuilt wheel for yours if a C++
toolchain is unavailable.

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

### 3. Start the Server Daemon (`assembly serve`)

Like `ollama serve`, you can start an HTTP server that runs in the background or on a dedicated machine/server:

```bash
# Start server with an active .asmb archive (binds to 127.0.0.1:8000 by default)
assembly serve -m my_models.asmb

# Custom host and port
assembly serve -m my_models.asmb --host 127.0.0.1 --port 9000
```

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
assembly serve -m my_models.asmb --host 0.0.0.0
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
| `--api-key KEY` | Require `Authorization: Bearer KEY` (or set `$ASSEMBLY_API_KEY`) |
| `--allow-origin ORIGIN` | Allow one CORS origin, e.g. `http://localhost:5173`. Omitted by default: no browser origin may call the server |
| `--models-dir DIR` | Confine the `/api/load` endpoint to `.asmb` files inside `DIR` |
| `--temp-dir DIR` | Where model blobs are staged on macOS (see below) |

CORS is off by default on purpose. Enable a single specific origin only if a
browser client needs it; do not use `*` on a network-exposed server.

### 4. Run Inference (`assembly run`)

`assembly run` automatically connects to your running server, or runs locally if standalone:

```bash
# Connect to a running server on localhost:8000, or to another device on your LAN:
assembly run -m my_models.asmb            # interactive session
assembly run -s http://192.168.1.50:8000  # remote server

# One-shot: answer a single prompt and exit
assembly run -m my_models.asmb "Write quicksort in Python"
assembly run -m my_models.asmb -p "Kim jesteś?"

# Show which sub-model answered each turn
assembly run -m my_models.asmb --route

# Forget the conversation between turns
assembly run -m my_models.asmb --no-memory
```

The session is deliberately plain — an arrow, your question, the answer:

```
$ assembly run -m my_models.asmb
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
| `/clear` | forget the conversation |
| `/bye` | exit (also `Ctrl+C` / `Ctrl+D`) |

`Ctrl+C` during a reply stops generating but keeps the session open. Progress
(archive loading, sub-model warm-up) goes to stderr, so piping stdout gives you
the answer and nothing else:

```bash
assembly run -m my_models.asmb "hi" 2>/dev/null | pbcopy
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
assembly info -m my_models.asmb
```

## Python API (In-Process)

```python
from assembly.wrapper import AssemblyLlama

with AssemblyLlama("my_models.asmb", n_ctx=2048, verbose=True) as llm:
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

## macOS staging and disk usage

`.asmb` archives are memory-mapped, never extracted. How the model bytes reach
`llama.cpp` differs by platform:

- **Linux** — each blob is copied into an anonymous in-memory file
  (`memfd_create`) and passed as `/proc/self/fd/N`. There is no staging copy on
  disk at all.
- **macOS** — `llama.cpp` requires a real filesystem path, so each model is
  written to a temporary file before it is loaded. The archive itself is still
  only read; nothing is duplicated wholesale, but the models do pass through
  the temp volume, and the first `run` after a cold start will be slower than
  subsequent ones.

If the system temp volume is small or slow, point staging elsewhere. The
directory is created if needed, and free space is checked before a multi-
gigabyte write begins:

```bash
assembly run -m my_models.asmb --temp-dir /Volumes/Fast/assembly
# or, for both run and serve:
export ASSEMBLY_TEMP_DIR=/Volumes/Fast/assembly
```

Free space is reported up front, before copying, so a volume that is too small
produces a clear message instead of an out-of-space error partway through a
load.

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

## License

MIT — see [LICENSE](LICENSE).

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
