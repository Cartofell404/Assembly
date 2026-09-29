# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

First public release.

### Added

- `assembly build` packs an embedding model and two or more execution models
  into a single `.asmb` archive, with anchor embeddings pre-computed at build
  time so routing needs no Python model at inference.
- `assembly run` — Ollama-style REPL: `>>>` prompt, token streaming,
  conversation memory, `/clear`, `/model`, `/route`, `/persona`, persistent
  input history, and one-shot mode via a positional prompt.
- `assembly run -s <url>` talks to a remote server; `--route` shows which
  sub-model answered and `--no-memory` disables history.
- `assembly serve` exposes OpenAI (`/v1/chat/completions`, streaming and
  non-streaming) and Ollama (`/api/chat`, `/api/generate`, `/api/tags`) APIs.
- `assembly info` prints archive metadata without loading any model.
- Archive loading is strict: format version, preamble, bounds, and page
  alignment are all validated, and failures raise `InvalidArchiveError` with a
  readable message instead of corrupting the process.

### Security

- **The server binds to `127.0.0.1` by default.** Binding any other address
  requires an API key; the server refuses to start otherwise.
- `Authorization: Bearer <key>` authentication, with the key taken from
  `--api-key` or `$ASSEMBLY_API_KEY`.
- CORS is opt-in via `--allow-origin`. No browser origin is allowed by default.
- Request bodies are capped at 1 MiB and socket reads are bounded by a timeout.
- `/api/load` can be confined to a directory with `--models-dir`, and reports
  a single generic error for missing files, non-archives, and out-of-jail
  paths so the endpoint cannot be used to probe the filesystem.
- The web UI renders model names as text, not HTML.

### Robustness

- Archives are built to a `.partial` file and renamed only after the written
  size and the on-disk header are verified, so an interrupted build never
  leaves a file that looks loadable.
- Header layout is solved to a fixed point instead of a single estimated pass.
- Each model blob is byte-count verified during the copy, catching models that
  are truncated or still downloading.
- Model loads and generation are guarded by locks; a failed load releases the
  archive descriptor and mmap instead of leaking them.
- The server bounds the number of concurrently loaded models and closes the
  previous model when one is replaced.

### Platform notes

- On macOS each model is staged as a temp file, because `llama.cpp` needs a
  real path. The first `run` after a cold start therefore reads the models
  from disk, which can take a while for large archives. Override the location
  with `--temp-dir` or `$ASSEMBLY_TEMP_DIR` if the system temp volume is small
  or slow; free space is checked before staging and reported up front.
- On Linux, blobs are passed to `llama.cpp` as anonymous in-memory files, so
  there is no staging copy at all.
