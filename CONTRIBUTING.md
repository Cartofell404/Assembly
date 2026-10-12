# Contributing

Thanks for helping improve Assembly.

## Setup

```bash
git clone https://github.com/Cartofell404/Assembly
cd assembly/rust
cargo build
```

Building the workspace compiles llama.cpp from source the first time, so expect
5-15 minutes for the initial build and a C++ toolchain (`cmake`, `clang`).
Subsequent builds are incremental.

## Running the tests

```bash
cargo test --workspace
```

The unit and integration tests never load a real model: the GGUF blobs, the HTTP
server, and the CLI are all exercised against fakes, so the suite runs in seconds
and needs no network. Do not add tests that require a multi-gigabyte `.asmb`.

There is a separate suite that does load real models. It is ignored by default
and only runs when you point it at weights explicitly:

```bash
ASSEMBLY_EMBED_MODEL=/path/to/embed.gguf \
ASSEMBLY_CHAT_MODEL=/path/to/chat.gguf \
    cargo test -p llama-engine --test real_models -- --ignored
```

## Style

- Rust, edition 2021, MSRV 1.88 (set by the `reqwest`/`url` dependency tree).
- Run `cargo fmt --all` and `cargo clippy --workspace -- -D warnings` before
  pushing. CI enforces both.
- Comments explain *why* something is non-obvious. Do not narrate what the next
  line does.
- Keep user-facing progress output on stderr so piped stdout stays clean.

## Before opening a pull request

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --all --check
```

Keep pull requests focused. If you change the archive format, bump the version
in `rust/Cargo.toml` and add a `CHANGELOG.md` entry, since that version is the
single source of truth for packaging.
