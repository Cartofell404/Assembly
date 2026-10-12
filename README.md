# Assembly

[![CI](https://github.com/Cartofell404/Assembly/actions/workflows/ci.yml/badge.svg)](https://github.com/Cartofell404/Assembly/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)
[![Version](https://img.shields.io/badge/version-1.0.0--beta.1-yellowgreen.svg)](https://github.com/Cartofell404/Assembly/releases)

> **Status: 1.0.0-beta.1** — feature-complete and usable, but it is a first
> beta, so expect rough edges. The format and CLI are stabilising but not yet
> frozen.

**A multi-model LLM router with a single binary distribution.**

Assembly packs several GGUF models into one `.asmb` archive and routes every
prompt to the smallest model that can answer it. Instead of loading everything
in RAM, only the model that gets chosen is loaded — so you get fast, cheap
answers on hardware that could never hold a stack of models at once.

## Features

- **Single archive, many models** — one `.asmb` file bundles an embedding model
  and any number of execution models, page-aligned with per-blob SHA-256
  digests.
- **Smart routing** — an embedding-based router picks the smallest model that
  can answer each prompt; a fallback model covers anything it is unsure about.
- **On-demand loading** — models are staged and loaded only when picked, and
  evicted from RAM to make room for the next one.
- **OpenAI-compatible server** — `assembly serve` exposes `/v1/chat/completions`
  (with SSE streaming), `/v1/models`, and more, so it works with existing
  OpenAI tooling.
- **No runtime outside the binary** — one executable does build, inspect,
  serve, and chat.

## Installation

The crates are not published to crates.io (the names are taken), so install
from a prebuilt binary or from a checkout.

### Option 1: Prebuilt binary (recommended)

Download the binary for your platform from [GitHub Releases](https://github.com/Cartofell404/Assembly/releases) and put it on your `PATH`:

| Platform               | Asset                          |
| ---------------------- | ------------------------------ |
| Linux x86-64           | `assembly-linux-x86_64`        |
| macOS (Apple Silicon)  | `assembly-macos-arm64`         |
| macOS (Intel)          | `assembly-macos-x86_64`        |
| Windows x86-64         | `assembly-windows-x86_64.exe`  |

### Option 2: Installer script

```bash
git clone https://github.com/Cartofell404/Assembly
cd assembly
./install.sh              # Linux / macOS (installs Rust and build deps if needed)
```

On Windows run `install.bat`. Run `./uninstall.sh` to remove Assembly.

### Option 3: Build from source

```bash
git clone https://github.com/Cartofell404/Assembly
cd assembly/rust
cargo build --release --locked
# Binary at ./target/release/assembly
# Or install it globally: cargo install --path crates/cli --locked
```

> **Platform notes:** macOS needs Xcode Command Line Tools
> (`xcode-select --install`). Linux needs `clang`, `cmake`, `pkg-config`.
> Windows needs Visual Studio Build Tools. The first build compiles llama.cpp,
> so it takes 5–10 minutes; later builds are incremental (~30 seconds).

## Quick start

```bash
# 1. Write a config.json (see the format under `assembly build` below)

# 2. Validate it
assembly validate config.json

# 3. Build the archive (computes anchor vectors)
assembly build config.json -o models.asmb

# 4. Inspect the archive
assembly info models.asmb --verify

# 5. Chat (picks up a server if one is running, otherwise runs locally)
assembly run models.asmb "Write a hello world in Rust"
```

## Commands

### `assembly build`

Builds an `.asmb` archive from a JSON configuration.

```bash
assembly build <config.json> [options]

Options:
  -o, --output <path>     Output archive path (default: config.asmb)
  --force                 Overwrite existing archive
  --no-check-magic        Skip GGUF magic bytes check
  --no-embed              Skip anchor vector computation (archive cannot route)
```

**Config format:**

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

Options:
- `--verify` - Re-hash every blob and compare against recorded digests

### `assembly embed`

Test the embedding model.

```bash
assembly embed <model.gguf> <text>
```

### `assembly serve`

Start the OpenAI-compatible HTTP server.

```bash
assembly serve <archive.asmb> [options]

Options:
  --host <ip>           Bind address (default: 127.0.0.1)
  --port <n>            Port (default: 8000)
  --api-key <key>       Require bearer token
  --settings <file.json> Load settings from JSON
  --workers <n>         Worker threads (default: 4)
  --headroom <0-1>      RAM fraction for cache (default: 0.8)
  --info                Print archive info and exit
```

**Endpoints:**
- `GET /health` - Liveness check
- `GET /v1/models` - Lists the archive as a single model
- `POST /v1/chat/completions` - OpenAI chat completions (supports SSE streaming)
- `GET /api/info` - Archive metadata
- `POST /api/route` - Dry-run routing decision

### `assembly run`

**Interactive chat.** Type a prompt, watch the answer stream, type another.
The conversation is kept for the whole session, on both paths below.

```bash
assembly run [options] [archive.asmb] [prompt ...]

Options:
  -s, --server <url>      Chat with this server instead of discovering one
      --port <n>          Same, by port (http://127.0.0.1:<n>)
      --local             Never talk to a server; answer here
  -m, --archive <path>    Archive for local mode (an alias when using a server)
      --settings <file>   Load settings from JSON
      --workers <n>       Worker threads, local mode
      --headroom <0-1>    RAM fraction for cache, local mode
```

**Server mode** (default): scans ports 8000-8005 for a running `assembly serve`
and chats with it. With one server found you get a prompt:

```
$ assembly run
chatting with models.asmb — /clear forgets the conversation, /quit exits

> explain what an asmb archive is
An .asmb archive is a page-aligned bundle ...
> /clear
(memory cleared)
> /quit
```

**Local mode**: pass an archive path (or `--local`), and the same session runs
against models loaded in this process.

```bash
assembly run models.asmb              # chat, no server needed
assembly run models.asmb "one answer" # answer, then keep chatting
```

Inside the session: `/clear` drops the history, `/quit` (or Ctrl-D) exits.
An answer is streamed token by token as it is generated — nothing is held back
until the model is done.

## Environment

- `ASSEMBLY_ARCHIVE` — archive `run` uses when no path is given.
- `ASSEMBLY_LLAMA_LOG=1` — let llama.cpp print its own startup chatter
  (`ggml_metal_device_init`, pipeline compiles, and the rest). By default
  `run` and `serve` keep that to themselves: it is noise on every start and
  says nothing about your archive.

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

## Requirements

- **Rust 1.88+** to build from source (via `rustup`)
- **Build dependencies**: `cmake`, `clang` (for llama.cpp)
- **Models**: GGUF format

## Contact

Questions, ideas, or feedback? Reach the maintainer at
[devcartofell@gmail.com](mailto:devcartofell@gmail.com).

For security issues, do **not** open a public issue — see [SECURITY.md](SECURITY.md).

## License

MIT License — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Acknowledgements

This project was brought to life with the help of AI assistance. The idea was conceived by the author, and AI tooling helped accelerate development, iteration, and documentation. Behind the idea is a human mind — the AI just helped turn caffeine into code faster.

Built with curiosity, stubbornness, and a little bit of prompt engineering.# Assembly
