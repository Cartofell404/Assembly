# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **`-m` is now required and must be an absolute path**, for `serve`, `info`,
  and for `run` on its local-inference path. Archive paths were previously
  guessed from the working directory when `-m` was omitted, which meant serving
  from the wrong folder silently loaded whatever `.asmb` happened to be sitting
  in it — and then handed that archive to every client that asked the server. A
  relative path is refused with the absolute form spelled out, and
  `~`/`$VARS` still expand as usual. `run -s URL -m <alias>` is unaffected: there
  `-m` names an already-loaded model on a server, not a file.

  A side effect: because `serve` can no longer start without an archive, there
  is always a directory to confine `/api/load` to, so the unconfined case
  described below is no longer reachable from the command line. Only a direct
  `assembly.server.serve()` call from Python can still leave it unconfined.

### Security

The HTTP server had a handful of issues that a browser, an unauthenticated
local process, or a client holding the key could turn into resource exhaustion
or information disclosure. See the new "Security model" section in the README
for what is and is not defended against.

Fixed:

- `/api/load` was an arbitrary-file opener. Any client could ask for any path
  the server user could read, and Assembly would hand it to the archive parser.
  It is now confined to `--models-dir`, which defaults to the served archive's
  own directory; paths are resolved first, so `..` and symlinks cannot escape.
  Starting `assembly serve` with no `-m` archive still leaves the endpoint
  unconfined — there is no directory to default to — and the startup banner says
  so explicitly. (`-m` is now required, so that case is CLI-unreachable.)
- A cross-origin `POST` was answered with `403` and then **performed anyway**.
  The refusal wrote its response and fell through to the router, so any page the
  operator visited could still load multi-gigabyte archives into their server
  (a `text/plain` POST needs no preflight, so the browser sends it and merely
  hides the reply). The handler now stops at the refusal.
- The model-count limit was not enforced under concurrency. The capacity check
  and the multi-second load were separate critical sections, so N simultaneous
  requests each saw a free slot and all stayed resident. Reservations now span
  the load. They are also counted per in-flight load rather than per alias:
  with one reservation per name, two concurrent loads of the *same* alias
  shared a slot, and the second silently discarded the first model without
  closing it, leaving two copies of its weights resident.
- Concurrent connections were unbounded. `ThreadingHTTPServer` spawns a thread
  per accepted connection, so a client holding a few hundred idle sockets grew
  the thread count to match. Connections are now capped at 32, and each has a
  300-second wall-clock deadline, so a client trickling one byte per minute can
  no longer hold a thread open indefinitely.
- A non-ASCII bearer token crashed the connection handler.
  `secrets.compare_digest` raises `TypeError` on non-ASCII `str`, and the call
  sat outside the handler's `try`, so an unauthenticated request produced a
  traceback on stderr and no response at all — a log-flood available to anyone
  who could reach the port. Tokens are now compared as bytes. Symmetrically, a
  non-ASCII key is refused at startup: HTTP headers are Latin-1 on the wire, so
  such a key could never be echoed back by a client and would have locked
  everyone out.
- Error messages no longer disclose filesystem detail. A staging failure
  reports the archive's staging directory and its free space in its message,
  which was returned verbatim — enough to map the server's disk. Clients now
  get a status and a reason; the detail stays in the log. Model-not-found and
  not-an-archive are reported identically, so `/api/load` is not a file
  existence oracle.
- Attacker-controlled text can no longer forge log entries. Paths, aliases,
  `Origin` and `Host` headers, and failed request lines had their control
  characters stripped before reaching the log, so a request could no longer
  inject newlines or terminal escapes.
- `Host` is now checked on loopback listeners, and responses carry
  `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`. A browser
  treats a rebound `evil.com → 127.0.0.1` request as same-origin, so CORS alone
  never applied; the `Host` check closes that route into a keyless local server.

Added:

- `POST /api/unload` with an `alias`. The model limit used to be a one-way
  door: a `429` told the client to "unload one first" while offering no way to
  unload, so the only recovery was restarting the server.
- A "Security model" section in the README, covering the enforced controls and,
  more importantly, the ones that do not exist: no TLS, no rate limiting, no
  accounts or quotas, serialised generation, and in-process parsing of
  untrusted `.asmb` archives.

### Fixed

- Windows no longer kills a running build when it sweeps abandoned partial
  files. The liveness probe used `os.kill(pid, 0)`, which signals nothing on
  POSIX but, on Windows, CPython forwards every signal except `CTRL_C_EVENT`
  and `CTRL_BREAK_EVENT` straight to `TerminateProcess`. Starting a second
  build would therefore kill the first one and then delete its half-written
  multi-gigabyte partial. The probe now opens the process with
  `OpenProcess`/`GetExitCodeProcess` on Windows, which cannot affect it, and
  declares its ctypes signatures so a 64-bit `HANDLE` is not truncated.
- Staged model blobs no longer break on Windows. The file was handed to
  `llama.cpp` while still open, which Windows refuses to let another process
  re-open; the blob is now written and closed first. Removal is best-effort,
  since Windows keeps a mapped file locked for as long as `llama.cpp` holds
  the model — the space is reclaimed at exit instead of raising.
- A `--temp-dir` the user points Assembly at is no longer deleted on their
  behalf. Only the private `assembly-*` directory Assembly creates is removed at
  exit, and only that.
- A failed or interrupted build no longer leaves `SIGINT` ignored for the rest
  of the process, which made Ctrl-C unusable afterwards. Signal handlers are
  restored in a `finally`, and a second Ctrl-C during a build can no longer
  unwind through a different path than the one that cleans up.
- Rebuilding over an archive that is currently being served now says so.
  Windows will not replace a file another process has open, so `assembly build`
  previously failed with a bare permission error after the whole pack.
- The REPL prompt is no longer uncoloured on every platform: rich's
  `export_text` returns plain characters unless asked for styles, so the arrow
  was silently stripped of its colour. It is also now plain on a legacy
  Windows console, which cannot interpret the escape codes and would print
  them as visible `^[[38;2;...` garbage.

### Added

- Windows is now a tested platform rather than an assumed one: the CI matrix
  covers `windows-latest` alongside Linux and macOS, including a check that
  the process-liveness probe leaves a real child process running.
- `history` extra (`pip install -e ".[history]"`) pulling in `pyreadline3` on
  Windows, where `readline` is not part of the standard library. Without it the
  REPL works but has no arrow-key editing or saved history, and now says so.
- Windows-specific installation and troubleshooting notes in the README.

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

### Added

- An interrupted build no longer risks leaving a broken multi-gigabyte file.
  The archive is assembled under a `.partial-<pid>` temporary name and renamed
  into place only after the bytes are written and the header reads back
  correctly, so a half-written `.asmb` is never presented as a valid archive.
  Ctrl-C already cleaned up after itself; `kill -9` and a power cut could not,
  so an abandoned temporary file is now reclaimed by the next build, which
  removes only those whose owning process is definitely gone. The temporary
  name carries the writer's pid, which also stops two concurrent builds to the
  same output from colliding. The rename is followed by an fsync of the
  directory so it survives a power cut.

### Fixed

- Streaming responses did not report the context window, while the
  non-streaming and Ollama paths did. A client that maintains conversation
  history had no way to learn its turns had been compressed, on the most common
  way of talking to the server. A final `assembly_context`-only frame now
  precedes `[DONE]`.
- Reading a model's metadata for its context length read less than the split
  reader reads, so a model with unusually large tokenizer metadata ahead of the
  context key would look as though it declared none and the window would
  silently collapse to the fallback. The read now escalates to the same 64 MiB
  the split reader uses, but only after a smaller read comes up empty, so the
  common case still costs nothing.
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
