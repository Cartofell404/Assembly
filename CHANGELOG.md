# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-09-30

First public release. Treat the archive format as experimental: it is versioned
inside the file header, but the format may still change before 1.0.

First public release.

### Changed

- The context window is now derived from the archive instead of being
  configured. Every model declares its trained context length in its GGUF
  metadata, and Assembly uses the smallest among the execution models, so a
  request can never be sized for a model that cannot hold it. `--n-ctx` and the
  `n_ctx` constructor argument are gone: an override could only exceed what a
  bundled model supports, or waste what it does. Archives that declare nothing
  usable fall back to the previous `2048`. The router's embedding model is
  excluded on purpose -- it embeds single sentences and is usually built with a
  tiny window, so including it would cap the whole assembly at its size.
- Conversations approaching the window are now summarised instead of rejected.
  Past 80% of the window Assembly asks the model to compress the earlier turns
  and continues from that summary plus the recent exchanges. Long histories are
  summarised in stages, so the start of a conversation survives rather than
  being trimmed away -- trimming from the front silently discards exactly the
  facts a question is usually about. Summarisation failing is never fatal: the
  original history is used unchanged. Reported on stderr in the terminal and as
  `assembly_context` in HTTP responses, so clients stop replaying turns that
  are no longer being sent.

### Fixed

- A conversation that outgrew the context window produced a `200` followed by
  an SSE error frame, and printed a full traceback to the server console.
  llama-cpp-python tokenises lazily, on the first advance of the streaming
  generator, so the failure surfaced after the status line had already gone
  out. The stream is now primed with one `next()` before the headers are sent,
  which turns a too-long conversation into a `400` with both token counts and
  the two remedies. The non-streaming endpoints report the same way instead of
  a bare `500`.
- Routine context overflows no longer log a stack trace; only genuinely
  unexpected failures do.
- `assembly serve` crashed with `KeyError: 'name'` while printing its startup
  banner, immediately after finishing a multi-gigabyte archive load. The row
  read a `name` key from `ModelManager.list_models()`, whose entries are keyed
  `id` / `assembly_name` / `persona` / `active` / `models`. The display string
  now comes from `ModelManager.describe_models()`, which lists the archive's
  sub-models, and cannot be broken by a key rename.
- Model loading copied every blob into RAM instead of mapping it. Slicing an
  `mmap` object returns `bytes`, so a 4 GB model was duplicated on each load
  and then crashed on `memoryview.release()`. Blobs are now taken as
  `memoryview(self._mm)[a:b]`, which is a true zero-copy view.
- `run --temp-dir` was parsed but never passed to the loader, so the flag
  silently did nothing.
- A leaked file descriptor on the Linux `memfd_create` staging path: the
  duplicated descriptor was never closed.
- `StagingError` was accidentally decorated with `@contextmanager`, turning
  the exception class into a function, so raising it would have failed.
- `assembly info` leaked a file descriptor on a corrupt archive and did not
  `expanduser()` its argument.
- A model file that disappears between validation and `stat()` produced a
  traceback instead of a validation error.
- `PreflightError` was being raised with a message where a validation report
  is required; build failures now raise a distinct `BuildError`.
- Sharded GGUFs (`model-00001-of-00003.gguf`) are now rejected. llama.cpp
  cannot load a shard on its own, so such a file packed into a clean-looking
  archive and then failed at inference with `invalid split file name`.
  `validate` and `build` report it before copying anything, and the loader
  fails in under a second instead of after staging gigabytes.
- A model that fails to load now reports which model, from which archive, and
  why, instead of a raw `ValueError` naming an internal temp file — and the
  CLI prints that message rather than a traceback.
- A `Ctrl+C` at the CLI now exits quietly with status 130 instead of printing a
  `KeyboardInterrupt` traceback, and a closed pipe (`... | head`) exits 0.

### Changed

- `setup.py` removed in favour of `pyproject.toml` metadata.
- `assembly -i/--interactive` removed; a REPL is already the default when no
  prompt is given.
- Version reported as `0.1.0` and sourced from `assembly/__init__.py`, which
  `pyproject.toml` now reads dynamically.
- `python -m assembly` now works alongside the `assembly` console script.
- The browser chat UI was removed along with the ~270 lines of embedded HTML,
  CSS and JavaScript that carried it. `serve` now exposes JSON endpoints only,
  so `/chat` returns 404 and no HTML is served anywhere. The terminal is the
  interface; `assembly serve` and `assembly run` are the two commands that
  matter day to day.
- Terminal rendering moved to `rich` with a single orange palette. stdout is
  still reserved for the model's reply, so piping behaviour is unchanged.
- The test suite no longer ships a hand-rolled runner that could not support
  the fixtures its own tests use; pytest is the single entry point.


### Added

- Configurable sampling. `/params` shows temperature, top-p, top-k, repeat
  penalty, presence penalty, frequency penalty and the token budget as bars;
  each can be set individually (`/temp 1.2`, `/frequency -0.4`, `/max 1024`,
  `/stop END`) or chosen as a `precise` / `balanced` / `creative` preset.
  Values are clamped rather than rejected and decimal commas are accepted.
- The OpenAI-compatible endpoints now honour `temperature`, `top_p`, `top_k`,
  `presence_penalty`, `frequency_penalty`, `repeat_penalty`, `max_tokens`,
  `stop` and `seed` on both `/v1/chat/completions` and `/v1/completions`, for
  streaming and non-streaming requests alike. Unusable values fall back to the
  default instead of failing the request.
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
