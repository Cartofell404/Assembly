<div align="center">

# Assembly

**Many models. One file. Every prompt goes to the smallest model that can answer it.**

[![CI](https://github.com/Cartofell404/Assembly/actions/workflows/ci.yml/badge.svg)](https://github.com/Cartofell404/Assembly/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)
[![Version](https://img.shields.io/badge/version-1.0.0--beta.1-yellowgreen.svg)](https://github.com/Cartofell404/Assembly/releases)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon%20(arm64)-1f6feb.svg)](https://github.com/Cartofell404/Assembly)

[Quick start](#quick-start) · [Installation](#installation) · [Commands](#commands) · [Architecture](#architecture) · [Contributing](CONTRIBUTING.md)

</div>

---

> [!IMPORTANT]
> **Apple Silicon Macs only.** Assembly targets `aarch64-apple-darwin`. Windows, Linux, and Intel Macs are not supported. Prebuilt binaries, the installer, and CI all assume macOS on an M-series chip.

> [!NOTE]
> **Status: 1.0.0-beta.1.** Feature-complete and usable, but it is a first beta, so expect rough edges. The archive format and CLI are stabilising but not yet frozen.

## What is Assembly?

Assembly packs several [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) models into a single `.asmb` archive and routes each prompt to the **smallest model that can answer it**. Only the chosen model is loaded into RAM, so you get fast, cheap answers on hardware that could never hold a whole stack of models at once.

```
Prompt → Embed → Cosine similarity → Select model → Generate
```

## Features

| | |
| --- | --- |
| **One archive, many models** | A single `.asmb` file bundles an embedding model and any number of execution models, page-aligned, with a SHA-256 digest per blob. |
| **Smart routing** | An embedding-based router picks the smallest capable model for each prompt. A fallback model covers anything it is unsure about. |
| **On-demand loading** | Models are staged and loaded only when picked, then evicted from RAM to make room for the next one. |
| **OpenAI-compatible server** | `assembly serve` exposes `/v1/chat/completions` (with SSE streaming) and `/v1/models`, so existing OpenAI tooling works. |
| **One binary** | The same executable builds, validates, inspects, serves, and chats. No separate runtime. |

## Quick start

```bash
# 1. Write a config.json (full reference below)

# 2. Validate it
assembly validate config.json

# 3. Build the archive (computes anchor vectors)
assembly build config.json -o models.asmb

# 4. Inspect it and verify every blob's digest
assembly info models.asmb --verify

# 5. Chat. Uses a running server if it finds one, otherwise runs locally
assembly run models.asmb "Write a hello world in Rust"
```

## Installation

Assembly supports **macOS on Apple Silicon (arm64) only**. The crates are not published to crates.io (the names are taken), so install from a prebuilt binary or from a checkout.

### Option 1: Prebuilt binary (recommended)

Download the binary from [GitHub Releases](https://github.com/Cartofell404/Assembly/releases) and put it on your `PATH`.

| Platform              | Asset                  |
| --------------------- | ---------------------- |
| macOS (Apple Silicon) | `assembly-macos-arm64` |

### Option 2: Installer script

```bash
git clone https://github.com/Cartofell404/Assembly
cd Assembly
./install.sh    # installs Rust and build dependencies if needed
```

Remove Assembly later with `./uninstall.sh`.

### Option 3: Build from source

```bash
git clone https://github.com/Cartofell404/Assembly
cd Assembly/rust
cargo build --release --locked
# Binary: ./target/release/assembly
# Or install globally:
cargo install --path crates/cli --locked
```

**Prerequisites:** Rust 1.88+ (via [`rustup`](https://rustup.rs)), `cmake`, and `clang`. On macOS, get these from Xcode Command Line Tools (`xcode-select --install`) or Homebrew (`cmake`, `llvm`).

> [!TIP]
> The first build compiles llama.cpp from source and takes several minutes (about 5–15, depending on your machine). Later builds are incremental.

## Commands

| Command | Purpose |
| --- | --- |
| [`assembly build`](#assembly-build) | Build an `.asmb` archive from a JSON config |
| [`assembly validate`](#assembly-validate) | Check a config without building |
| [`assembly info`](#assembly-info) | Inspect an archive, optionally verifying digests |
| [`assembly embed`](#assembly-embed) | Try out an embedding model |
| [`assembly serve`](#assembly-serve) | Start the OpenAI-compatible HTTP server |
| [`assembly run`](#assembly-run) | Interactive chat, via a server or locally |

### `assembly build`

Builds an `.asmb` archive from a JSON configuration.

```bash
assembly build <config.json> [options]
```

| Option | Description |
| --- | --- |
| `-o, --output <path>` | Output archive path (default: `config.asmb`) |
| `--force` | Overwrite an existing archive |
| `--no-check-magic` | Skip the GGUF magic-bytes check |
| `--no-embed` | Skip anchor vector computation (the archive **cannot route**) |

#### Config format

```json
{
  "assembly_name": "my-router",
  "persona": "You are a helpful assistant.",
  "embedding": {
    "name": "router",
    "path": "/models/embedding.gguf",
    "confidence_threshold": 0.6,
    "fallback_model": "general"
  },
  "models": [
    {
      "name": "general",
      "path": "/models/general.gguf",
      "anchors": ["chat", "conversation", "help"]
    },
    {
      "name": "code",
      "path": "/models/code.gguf",
      "anchors": ["code", "function", "debug", "implement"]
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `assembly_name` | Name of the assembly |
| `persona` | System persona applied to the models |
| `embedding.name` / `embedding.path` | The router (embedding) model and where its GGUF lives |
| `embedding.confidence_threshold` | How sure the router must be before it commits to a model |
| `embedding.fallback_model` | Name of the model used when routing confidence is low. Must match an entry in `models` |
| `models[].name` / `models[].path` | An execution model and its GGUF file |
| `models[].anchors` | Example words or phrases describing what this model is good at. Embedded at build time into the anchor vectors used for routing |

### `assembly validate`

Validates a config without building.

```bash
assembly validate config.json
```

### `assembly info`

Inspects an archive.

```bash
assembly info archive.asmb [--verify]
```

`--verify` re-hashes every blob and compares it against the recorded digests.

### `assembly embed`

Tests an embedding model directly.

```bash
assembly embed <model.gguf> <text>
```

### `assembly serve`

Starts the OpenAI-compatible HTTP server.

```bash
assembly serve <archive.asmb> [options]
```

| Option | Description |
| --- | --- |
| `--host <ip>` | Bind address (default: `127.0.0.1`) |
| `--port <n>` | Port (default: `8000`) |
| `--api-key <key>` | Require this bearer token |
| `--settings <file.json>` | Load settings from JSON |
| `--workers <n>` | Worker threads (default: `4`) |
| `--headroom <0-1>` | Fraction of RAM to use for the model cache (default: `0.8`) |
| `--info` | Print archive info and exit |

**Endpoints**

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness check |
| `GET` | `/v1/models` | Lists the archive as a single model |
| `POST` | `/v1/chat/completions` | OpenAI chat completions, with SSE streaming |
| `GET` | `/api/info` | Archive metadata |
| `POST` | `/api/route` | Dry-run a routing decision |

**Example**

```bash
assembly serve models.asmb --port 8000

# In another terminal. Use the id reported by /v1/models as the model name.
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "<id from /v1/models>",
        "messages": [{"role": "user", "content": "Write a hello world in Rust"}],
        "stream": true
      }'
```

If you started the server with `--api-key`, add `-H "Authorization: Bearer <key>"`.

> [!WARNING]
> The server binds to loopback by default. Binding to a non-loopback address requires an API key. See [SECURITY.md](SECURITY.md) for what is and isn't defended against.

### `assembly run`

Interactive chat. Type a prompt, watch the answer stream, type another. The conversation is kept for the whole session, on both paths below.

```bash
assembly run [options] [archive.asmb] [prompt ...]
```

| Option | Description |
| --- | --- |
| `-s, --server <url>` | Chat with this server instead of discovering one |
| `--port <n>` | Same, by port (`http://127.0.0.1:<n>`) |
| `--local` | Never talk to a server; answer in this process |
| `-m, --archive <path>` | Archive for local mode (an alias when using a server) |
| `--settings <file>` | Load settings from JSON |
| `--workers <n>` | Worker threads (local mode) |
| `--headroom <0-1>` | RAM fraction for the cache (local mode) |

**Server mode (default).** `run` scans ports 8000–8005 for a running `assembly serve` and chats with it. If it finds exactly one server, you get a prompt:

```
$ assembly run
chatting with models.asmb — /clear forgets the conversation, /quit exits

> explain what an asmb archive is
An .asmb archive is a page-aligned bundle ...
> /clear
(memory cleared)
> /quit
```

**Local mode.** Pass an archive path (or `--local`) and the same session runs against models loaded in this process:

```bash
assembly run models.asmb                 # chat, no server needed
assembly run models.asmb "one answer"    # answer, then keep chatting
```

Inside a session, `/clear` drops the history and `/quit` (or Ctrl-D) exits. Answers stream token by token as they are generated; nothing is held back until the model finishes.

## Environment variables

| Variable | Effect |
| --- | --- |
| `ASSEMBLY_ARCHIVE` | Archive that `run` uses when no path is given |
| `ASSEMBLY_LLAMA_LOG=1` | Let llama.cpp print its own startup output (`ggml_metal_device_init`, pipeline compiles, and so on). Off by default: it is noise on every start and says nothing about your archive |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   .asmb Archive                     │
├─────────────────────────────────────────────────────┤
│  HEADER  (magic + version + JSON with offsets)      │
│  ── page-aligned padding ──                         │
│  [Embedding GGUF]   ← router model (small)          │
│  ── page-aligned padding ──                         │
│  [Model A GGUF]     ← execution model #1            │
│  ── page-aligned padding ──                         │
│  [Model B GGUF]     ← execution model #2            │
│  ...                                                │
└─────────────────────────────────────────────────────┘
```

**How a prompt is routed**

1. At build time, each model's `anchors` are embedded and stored in the archive as anchor vectors.
2. At request time, the prompt is embedded with the router model.
3. Cosine similarity against the anchor vectors picks the best-matching model.
4. If confidence is below `confidence_threshold`, the `fallback_model` answers instead.
5. Only that model is loaded (evicting others if RAM requires), and the answer streams back.

Workspace crates: `asmb-format` (archive format), `assembly-core` (router), `assembly-server` (HTTP), `llama-engine` (llama.cpp inference), and the `assembly` CLI.

## Requirements

- macOS on Apple Silicon (arm64), the only supported platform
- Rust 1.88+ to build from source
- `cmake` and `clang` to compile llama.cpp
- Models in GGUF format

## Known limitations

- Apple Silicon macOS only
- Beta: the archive format and CLI may still change
- Not published to crates.io; install via binary, installer, or source

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Please do **not** open a public issue for security problems. See [SECURITY.md](SECURITY.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Contact

Questions, ideas, or feedback? Reach the maintainer at [devcartofell@gmail.com](mailto:devcartofell@gmail.com).

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

This project was built with the help of AI tooling. The idea and direction are the author's; AI helped speed up development, iteration, and documentation. Built with curiosity, stubbornness, and a little prompt engineering.