# Contributing

Thanks for helping improve Assembly Engine.

## Setup

```bash
git clone https://github.com/assembly-engine/assembly
cd assembly
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`llama-cpp-python` compiles from source on most platforms. If the build fails
for lack of a compiler, install a prebuilt wheel for your platform first.

## Running the tests

```bash
pytest -q
```

The test suite runs in well under a second and never loads a real model: the
GGUF blobs, the HTTP server, and the CLI are all exercised against fakes. Do
not add tests that require a multi-gigabyte `.asmb` or network access — that
makes the suite unusable for everyone else.

## Style

- Python 3.10+. Type-annotate new public functions.
- Comments explain *why* something is non-obvious. Do not narrate what the
  next line does.
- Keep user-facing progress output on stderr so piped stdout stays clean.

## Before opening a pull request

```bash
pytest -q
python -m compileall -q assembly
```

Keep pull requests focused. If you change the archive format, bump
`__version__` in `assembly/__init__.py` and add a `CHANGELOG.md` entry — the
version there is the single source of truth for packaging.
