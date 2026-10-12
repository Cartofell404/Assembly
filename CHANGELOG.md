# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Rust workspace (`rust/`) with the `assembly` CLI, replacing the earlier Python
  implementation.
- `.asmb` archive format with a page-aligned header, per-blob SHA-256 digests,
  and anchor vectors for routing (`asmb-format`).
- Embedding-based router that selects the smallest model able to answer a prompt
  (`assembly-core`).
- OpenAI-compatible HTTP server for a served archive: `/health`, `/v1/models`,
  `/v1/chat/completions` (SSE streaming), `/api/info`, and `/api/route`
  (`assembly-server`).
- Local GGUF inference over llama.cpp (`llama-engine`).
- CLI subcommands: `build`, `validate`, `info`, `embed`, `serve`, `run`.
- Installers for Linux/macOS (`install.sh`), Windows (`install.ps1`,
  `install.bat`), and a `uninstall.sh`.

## [0.1.0]

Initial pre-release of the Rust rewrite.
