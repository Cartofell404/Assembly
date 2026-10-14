# Contributing to Assembly

Thanks for helping improve Assembly! This guide covers everything you need to go from a fresh clone to a merged pull request.

By participating, you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md). For security problems, **do not open a public issue**. See [SECURITY.md](SECURITY.md).

## Contents

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Project layout](#project-layout)
- [Running the tests](#running-the-tests)
- [Style](#style)
- [Changing the archive format](#changing-the-archive-format)
- [Before opening a pull request](#before-opening-a-pull-request)
- [Reporting bugs and requesting features](#reporting-bugs-and-requesting-features)

## Ways to contribute

- Report bugs or rough edges (this is a beta, so there are some)
- Improve documentation
- Fix issues or add features
- Test on your own models and share what worked or didn't

Assembly supports **macOS on Apple Silicon only**, so you'll need an M-series Mac to build and run it.

## Development setup

**Prerequisites:** Rust 1.88+ (via [`rustup`](https://rustup.rs)), plus `cmake` and `clang`. Xcode Command Line Tools (`xcode-select --install`) or Homebrew will provide them.

```bash
git clone https://github.com/Cartofell404/Assembly
cd Assembly/rust
cargo build
```

> [!NOTE]
> The first build compiles llama.cpp from source, so expect roughly 5–15 minutes. Later builds are incremental.

## Project layout

The Rust workspace lives in `rust/`:

| Crate | Role |
| --- | --- |
| `asmb-format` | The `.asmb` archive format: header, page alignment, SHA-256 digests, anchor vectors |
| `assembly-core` | The embedding-based router |
| `assembly-server` | The OpenAI-compatible HTTP server |
| `llama-engine` | Local GGUF inference over llama.cpp |
| `cli` (`crates/cli`) | The `assembly` binary |

## Running the tests

```bash
cargo test --workspace
```

The unit and integration tests **never load a real model**. GGUF blobs, the HTTP server, and the CLI are all exercised against fakes, so the suite runs in seconds and needs no network.

> [!WARNING]
> Do not add tests that require a multi-gigabyte `.asmb`.

### Real-model tests (optional)

A separate suite does load real models. It is ignored by default and only runs when you point it at weights explicitly:

```bash
ASSEMBLY_EMBED_MODEL=/path/to/embed.gguf \
ASSEMBLY_CHAT_MODEL=/path/to/chat.gguf \
    cargo test -p llama-engine --test real_models -- --ignored
```

## Style

- Rust, edition 2021, **MSRV 1.88** (set by the `reqwest`/`url` dependency tree).
- Run `cargo fmt --all` and `cargo clippy --workspace -- -D warnings` before pushing. CI enforces both.
- Comments explain *why* something is non-obvious. Don't narrate what the next line does.
- Keep user-facing progress output on **stderr** so piped stdout stays clean.

## Changing the archive format

If your change touches the archive format:

1. Bump the version in `rust/Cargo.toml`. That version is the single source of truth for packaging.
2. Add an entry to [`CHANGELOG.md`](CHANGELOG.md).
3. Note the compatibility impact in your pull request description.

## Before opening a pull request

Run all three and make sure they pass:

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --all --check
```

Checklist:

- [ ] Pull request is focused on one change
- [ ] Tests added or updated (using fakes, not real models)
- [ ] Docs updated if behaviour or CLI flags changed
- [ ] `CHANGELOG.md` updated for user-visible changes

Small, focused pull requests get reviewed faster than large ones.

## Reporting bugs and requesting features

Open a GitHub issue and include:

- Your macOS and chip details (for example, macOS version and M-series model)
- Assembly version (`1.0.0-beta.1`, or the commit hash if built from source)
- The command you ran and the full output
- Whether you were using server mode or local mode
- For routing problems, the output of `assembly info <archive>` and a dry run via `POST /api/route`

Setting `ASSEMBLY_LLAMA_LOG=1` shows llama.cpp's own logs, which can help with model-loading problems.

Questions? Email [devcartofell@gmail.com](mailto:devcartofell@gmail.com).