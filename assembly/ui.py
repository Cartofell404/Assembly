"""
UI & Terminal Styling utilities for Assembly Engine.

Provides ANSI color helpers, a minimal status spinner, and terminal capability
detection. The CLI deliberately stays quiet by default — no banners, no chrome —
so it reads like a plain text conversation.
"""

from __future__ import annotations

import itertools
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

# Check if colors should be enabled (TTY + not NO_COLOR)
_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ

# Spinner frames — braille, degrades to nothing on a dumb terminal.
_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_ANIM_INTERVAL = 0.08


def is_tty() -> bool:
    """True when we can safely draw spinners / readline editing."""
    return sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


def supports_spinner() -> bool:
    """True when a spinner is visible and worth drawing."""
    return is_tty() and "NO_COLOR" not in os.environ and "ASSEMBLY_NO_SPINNER" not in os.environ


class Color:
    RESET = "\033[0m" if _USE_COLOR else ""
    BOLD = "\033[1m" if _USE_COLOR else ""
    DIM = "\033[2m" if _USE_COLOR else ""
    ITALIC = "\033[3m" if _USE_COLOR else ""
    UNDERLINE = "\033[4m" if _USE_COLOR else ""

    # Colors
    BLACK = "\033[30m" if _USE_COLOR else ""
    RED = "\033[31m" if _USE_COLOR else ""
    GREEN = "\033[32m" if _USE_COLOR else ""
    YELLOW = "\033[33m" if _USE_COLOR else ""
    BLUE = "\033[34m" if _USE_COLOR else ""
    MAGENTA = "\033[35m" if _USE_COLOR else ""
    CYAN = "\033[36m" if _USE_COLOR else ""
    WHITE = "\033[37m" if _USE_COLOR else ""

    # Backgrounds
    BG_CYAN = "\033[46m" if _USE_COLOR else ""
    BG_BLUE = "\033[44m" if _USE_COLOR else ""


def prompt_text() -> str:
    """The input prompt. Deliberately just an arrow — nothing else."""
    return f"{Color.CYAN}{Color.BOLD}>>> {Color.RESET}"


@contextmanager
def status(message: str) -> Iterator[None]:
    """Show an animated ``message …`` on one line, then clear it.

    Always writes to stderr so that piping stdout to a file yields the answer
    and nothing else. Falls back to a single static line when the terminal
    can't animate, and erases the line only if it was actually visible.
    """
    if not is_tty():
        print(f"{Color.DIM}{message}…{Color.RESET}", file=sys.stderr)
        yield
        return

    live = supports_spinner()
    started = time.monotonic()
    if not live:
        print(f"{Color.DIM}{message}…{Color.RESET}", end="\r", file=sys.stderr, flush=True)

    stop = threading.Event()

    def _spin() -> None:
        for frame in itertools.cycle(_FRAMES):
            if stop.wait(_ANIM_INTERVAL):
                return
            sys.stdout.flush()
            sys.stderr.write(f"\r{Color.DIM}{frame} {message}…{Color.RESET}\x1b[K")
            sys.stderr.flush()

    thread = threading.Thread(target=_spin, daemon=True) if live else None
    if thread:
        thread.start()
    try:
        yield
    finally:
        stop.set()
        if thread:
            thread.join(timeout=0.2)
        # Only pay the carriage return if we actually drew something.
        if time.monotonic() - started > 0.15:
            sys.stdout.flush()
            sys.stderr.write("\r\x1b[K")
            sys.stderr.flush()


def dim(msg: str) -> None:
    """Print a subdued helper line (used for /command output)."""
    print(f"{Color.DIM}{msg}{Color.RESET}")


def blank() -> None:
    print()


def badge_router(model_name: str, score: float, is_fallback: bool = False) -> str:
    """A one-line note about which sub-model answered. Opt-in via ``--route``."""
    c = Color
    pct = int(score * 100)
    if is_fallback:
        return f"{c.DIM}↳ {c.YELLOW}{model_name}{c.RESET}{c.DIM} (fallback, {pct}% match){c.RESET}"
    if score >= 0.85:
        color = c.GREEN
    elif score >= 0.70:
        color = c.CYAN
    else:
        color = c.YELLOW
    return f"{c.DIM}↳ {c.RESET}{color}{model_name}{c.RESET}{c.DIM} · {pct}% match{c.RESET}"


def badge_persona(persona_text: str, max_len: int = 60) -> str:
    """Format a styled persona badge."""
    c = Color
    short = persona_text if len(persona_text) <= max_len else persona_text[:max_len] + "..."
    return f"{c.DIM}[persona:{c.RESET} {c.MAGENTA}{short}{c.RESET}{c.DIM}]{c.RESET}"


def success(msg: str) -> None:
    c = Color
    print(f"{c.GREEN}{c.BOLD}✓{c.RESET} {msg}")


def info(msg: str) -> None:
    c = Color
    print(f"{c.CYAN}{c.BOLD}ℹ{c.RESET} {msg}")


def warn(msg: str) -> None:
    c = Color
    print(f"{c.YELLOW}{c.BOLD}⚠{c.RESET} {msg}", file=sys.stderr)


def error(msg: str) -> None:
    c = Color
    print(f"{c.RED}{c.BOLD}✗{c.RESET} {msg}", file=sys.stderr)
