"""
Terminal presentation for Assembly Engine.

Built on ``rich`` with a single warm palette. Orange is the brand colour and
carries all the structural weight — headings, rules, the prompt, routing
decisions — while cream is reserved for the model's own text so a reply is
never confused for chrome.

Two rules keep this from getting in the way of the actual product:

1. **stdout is the answer, stderr is everything else.** Status spinners,
   banners and warnings go to stderr, so ``assembly run ... | pbcopy`` yields
   the reply and nothing else.
2. **Degrade, don't fail.** ``NO_COLOR``, a dumb terminal or a piped stdout
   must all still produce readable output, so every helper checks whether it
   is actually talking to a terminal.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from io import StringIO
from typing import Iterator, Sequence

from rich import box
from rich.color import Color
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ── Palette ─────────────────────────────────────────────────────────────
# One hue family, dark to light. Keeping every accent on the orange axis is
# what makes the CLI read as a single product rather than a bag of helpers.
ORANGE_DEEP = "#B34700"   # borders, rules, lowest-contrast structure
ORANGE = "#E8710A"        # brand: headings, prompt, model names
ORANGE_LIT = "#FF9E3D"    # emphasis, counts, highlights
AMBER = "#FFC078"         # secondary text
CREAM = "#F3E9DE"         # body text — warm white, easier on the eye than #FFF
MUTED = "#9A7B63"         # captions and de-emphasised detail
RED = "#D14343"           # failures only; nothing else in the app uses red

# ── Consoles ────────────────────────────────────────────────────────────
# Two consoles, deliberately: stdout carries only the reply, stderr carries
# everything decorative.
out = Console(highlight=False, soft_wrap=True)
err = Console(stderr=True, highlight=False, soft_wrap=True)

_NO_COLOR = "NO_COLOR" in os.environ
_DUMB = os.environ.get("TERM") == "dumb"

# The legacy Windows console (pre-Windows 10, or VT processing disabled) has no
# ANSI support, and `input()` prints its prompt straight to the console without
# going through rich. An escape sequence there is not stripped, it is printed,
# so the user sees "^[[38;2;255;158;61m" in front of every line they type.
_LEGACY_WINDOWS_CONSOLE = bool(getattr(out, "legacy_windows", False))

# Force plain text when the stream is not a terminal so that redirected
# output contains no escape sequences at all.
if _NO_COLOR or _DUMB:
    out = Console(file=sys.stdout, no_color=True, highlight=False, soft_wrap=True)
    err = Console(file=sys.stderr, no_color=True, highlight=False, soft_wrap=True)


def is_tty() -> bool:
    """True when we can safely draw spinners and readline editing."""
    return sys.stdout.isatty() and not _DUMB


def supports_spinner() -> bool:
    """True when a spinner is visible and worth drawing."""
    return (
        is_tty()
        and not _NO_COLOR
        and os.environ.get("ASSEMBLY_NO_SPINNER") != "1"
    )


# ── Primitives ──────────────────────────────────────────────────────────


def dim(msg: str | Text) -> None:
    """A de-emphasised caption. Accepts plain text or an already-styled Text."""
    err.print(Text(msg, style=MUTED) if isinstance(msg, str) else msg)


def blank() -> None:
    err.print()


def success(msg: str) -> None:
    err.print(Text("✓ ", style=f"bold {ORANGE_LIT}") + Text(msg, style=CREAM))


def info(msg: str) -> None:
    err.print(Text("› ", style=f"bold {ORANGE}") + Text(msg, style=AMBER))


def warn(msg: str) -> None:
    err.print(Text("! ", style=f"bold {ORANGE}") + Text(msg, style=AMBER))


def error(msg: str) -> None:
    """Report a failure. Always stderr, so stdout stays pipeable."""
    err.print(Text("✗ ", style=f"bold {RED}") + Text(msg, style=CREAM))


@contextmanager
def status(message: str) -> Iterator[None]:
    """Animate ``message …`` while slow work happens.

    Uses rich's own status renderer so the spinner shares the palette with the
    rest of the UI. Falls back to a single static line when the terminal
    cannot animate.
    """
    if not supports_spinner():
        err.print(Text(f"{message}…", style=MUTED))
        yield
        return

    with err.status(Text(f"{message}…", style=AMBER), spinner="dots", spinner_style=ORANGE):
        yield


# ── Composites ──────────────────────────────────────────────────────────


def prompt_text() -> str:
    """The input prompt, as a plain string for ``input()``.

    Rendered through rich rather than hand-written escape codes so the colour
    is derived from the palette constant and cannot drift out of sync with it.

    The scratch console must write to a buffer, not to stdout: ``input()``
    prints the returned string itself, so a console pointed at stdout would
    emit the prompt twice.
    """
    # A legacy Windows console cannot interpret these codes, so colour there is
    # a bug, not a downgrade. The same goes for a redirected stdout: input()
    # writes this string straight to the file or pipe, where escape sequences
    # are just noise in someone's captured output.
    if _NO_COLOR or _DUMB or _LEGACY_WINDOWS_CONSOLE or not sys.stdout.isatty():
        return "▸ "
    sink = StringIO()
    scratch = Console(
        file=sink, force_terminal=True, width=64, highlight=False, record=True
    )
    scratch.print(
        Text("▸ ", style=Style(color=Color.parse(ORANGE), bold=True)), end=""
    )
    # styles=True is what puts the escape codes back in. Without it rich
    # returns the plain characters and the prompt is unstyled on every
    # platform -- the colour is silently dropped, not applied.
    return scratch.export_text(styles=True).rstrip()


def rule(width: int = 60) -> None:
    """A thin horizontal divider.

    Printed directly rather than via ``Console.rule()``: rich centres whatever
    you hand it as a title, which turns a run of box-drawing characters into
    three disconnected segments.
    """
    err.print(Text("─" * width, style=ORANGE_DEEP))


def meter(
    label: str,
    value: float,
    low: float,
    high: float,
    *,
    suffix: str = "",
    width: int = 24,
    marker: str | None = None,
) -> None:
    """One labelled bar showing where ``value`` sits in ``[low, high]``.

    The bar is the point: a number like ``0.8`` is hard to judge in isolation,
    whereas a bar makes "half way up the temperature range" obvious at a
    glance. Filled cells use the bright brand colour and the remainder fades,
    so the orange stays dominant without shouting.
    """
    span = (high - low) or 1.0

    # A range that straddles zero gets a centre-anchored bar, because filling
    # from the left would make a neutral 0.0 look half-applied and would hide
    # the sign of the value.
    signed = low < 0.0 < high

    if marker is not None:
        # A preset/target tick, so "where am I vs where I want to be" is visible.
        target = max(0.0, min(1.0, (marker - low) / span))
        target_at = int(round(target * width))
    else:
        target_at = -1

    if signed:
        centre = int(round(((0.0 - low) / span) * width))
        centre = max(0, min(width - 1, centre))
        at = max(0, min(width, int(round(((value - low) / span) * width))))
        bar = Text()
        for i in range(width):
            if i == centre and not (min(at, centre) <= i < max(at, centre)):
                bar.append("│", style=MUTED)
            elif min(at, centre) <= i < max(at, centre):
                bar.append("█", style=ORANGE_LIT)
            else:
                bar.append("·", style=ORANGE_DEEP)
    else:
        fraction = max(0.0, min(1.0, (value - low) / span))
        filled = int(round(fraction * width))
        filled = max(1 if width else 0, filled)  # show at least one cell

        bar = Text()
        for i in range(width):
            if i == target_at and i != filled:
                bar.append("│", style=MUTED)
            elif i < filled:
                bar.append("█", style=ORANGE_LIT)
            else:
                bar.append("·", style=ORANGE_DEEP)

    shown = (
        f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
    )
    err.print(
        Text.assemble(
            (f"  {label:<16}", CREAM),
            bar,
            (f"  {shown}{suffix}", ORANGE_LIT),
        )
    )


def banner(title: str, subtitle: str, rows: Sequence[tuple[str, str]]) -> None:
    """The header shown when the server starts.

    ``rows`` is (label, value) pairs; an empty label renders as a spacer, which
    is how the endpoint list is separated from the status lines.
    """
    body = Table.grid(padding=(0, 2))
    body.add_column(style=ORANGE, no_wrap=True, justify="right")
    body.add_column(style=CREAM)

    for label, value in rows:
        if not label:
            body.add_row("", "")
            continue
        body.add_row(Text(label, style=MUTED), Text(value, style=CREAM))

    err.print(
        Panel(
            body,
            title=Text(f" {title} ", style=f"bold {ORANGE_LIT}"),
            subtitle=Text(f" {subtitle} ", style=ORANGE_DEEP),
            border_style=ORANGE_DEEP,
            padding=(1, 2),
        )
    )


def heading(msg: str) -> None:
    """A section heading inside a command's output."""
    err.print(Text(msg, style=f"bold {ORANGE}"))


def table(title: str, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """A data table, used by /model and /info.

    Uses SIMPLE_HEAD rather than rich's default box: the heavy ``┏━━┳━━┓``
    frame competes with the text, and the point of these tables is to be
    scannable in a glance.
    """
    grid = Table(
        title=Text(title, style=f"bold {ORANGE}"),
        title_justify="left",
        box=box.SIMPLE_HEAD,
        header_style=f"bold {ORANGE_LIT}",
        border_style=ORANGE_DEEP,
        row_styles=[CREAM, AMBER],
        pad_edge=False,
        expand=False,
    )
    for column in columns:
        grid.add_column(column, style=CREAM)
    for row in rows:
        grid.add_row(*row)
    err.print(grid)


def badge_router(model_name: str, score: float, is_fallback: bool = False) -> Text:
    """One line naming the sub-model that answered. Opt-in via ``--route``."""
    pct = int(score * 100)
    if is_fallback:
        return Text.assemble(
            ("↳ ", MUTED),
            (model_name, ORANGE_DEEP),
            (f" (fallback, {pct}% match)", MUTED),
        )
    # Closer matches get brighter, so a confident route is obvious at a glance.
    if score >= 0.85:
        colour = ORANGE_LIT
    elif score >= 0.70:
        colour = ORANGE
    else:
        colour = ORANGE_DEEP
    return Text.assemble(
        ("↳ ", MUTED),
        (model_name, colour),
        (f" · {pct}% match", MUTED),
    )


def badge_persona(persona_text: str, max_len: int = 60) -> Text:
    """A short note about the active persona."""
    short = persona_text if len(persona_text) <= max_len else persona_text[:max_len] + "…"
    return Text.assemble(("persona ", MUTED), (short, ORANGE))


def print_reply(text: str) -> None:
    """Write a model reply to stdout.

    This is the only function that writes to stdout, which is what keeps
    ``assembly run ... | pbcopy`` and ``> answer.txt`` clean.
    """
    out.print(Text(text, style=CREAM))


def markdown(renderable: RenderableType) -> None:
    """Pass a rich renderable through to stdout (e.g. rendered markdown)."""
    out.print(renderable)