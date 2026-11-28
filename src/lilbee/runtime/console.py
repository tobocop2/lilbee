"""The Rich console lilbee prints through, which shows every string as literal text."""

from __future__ import annotations

import os
import sys
from typing import IO, Any, TextIO, TypeAlias

from rich.console import Console, RenderableType
from rich.highlighter import ReprHighlighter
from rich.live import Live
from rich.progress import Progress, ProgressColumn
from rich.status import Status
from rich.style import StyleType
from rich.text import Text, TextType

ProgressDisplay: TypeAlias = Progress
"""The type of a progress display; build one with ``live_progress``."""


def _redirects_stdout(console: Console) -> bool:
    """Whether a live display on *console* moves a stdout print above itself."""
    return not console.stderr


def _redirects_stderr(console: Console) -> bool:
    """Whether a live display on *console* moves a stderr print above itself.

    Only a stdout console on the terminal that ``sys.stderr`` also writes to qualifies. A
    stderr line then gets its own row instead of joining the display row. When stderr goes to
    a file, a pipe or another terminal, the line stays there.
    """
    if console.stderr:
        return False
    try:
        out_fd, err_fd = console.file.fileno(), sys.stderr.fileno()
        return os.isatty(err_fd) and os.path.samestat(os.fstat(out_fd), os.fstat(err_fd))
    except (AttributeError, OSError, ValueError):
        return False


def live_progress(
    *columns: str | ProgressColumn, console: Console, **options: Any
) -> ProgressDisplay:
    """A progress display that moves a print above the bar only on the terminal it shares.

    A stdout print moves above the bar on a stdout console. A stderr print moves above it
    only when stderr writes to the same terminal. Every other print stays on the stream its
    writer chose. *options* go to ``Progress``.
    """
    return Progress(
        *columns,
        console=console,
        redirect_stdout=_redirects_stdout(console),
        redirect_stderr=_redirects_stderr(console),
        **options,
    )


class _LiteralStatus(Status):
    """A spinner whose live display follows the stream rule of ``live_progress``."""

    def __init__(
        self,
        status: RenderableType,
        *,
        console: Console,
        spinner: str,
        spinner_style: StyleType,
        speed: float,
        refresh_per_second: float,
    ) -> None:
        super().__init__(
            status,
            console=console,
            spinner=spinner,
            spinner_style=spinner_style,
            speed=speed,
            refresh_per_second=refresh_per_second,
        )
        self._live = Live(
            self.renderable,
            console=console,
            refresh_per_second=refresh_per_second,
            transient=True,
            redirect_stdout=_redirects_stdout(console),
            redirect_stderr=_redirects_stderr(console),
        )


class PlainConsole(Console):
    """A Rich console that never parses a string as markup; style with ``Text`` or ``style=``."""

    def __init__(
        self,
        *,
        stderr: bool = False,
        file: IO[str] | None = None,
        force_terminal: bool | None = None,
        width: int | None = None,
        quiet: bool = False,
    ) -> None:
        super().__init__(
            stderr=stderr,
            file=file,
            force_terminal=force_terminal,
            width=width,
            quiet=quiet,
            markup=False,
            emoji=False,
        )

    def input(
        self,
        prompt: TextType = "",
        *,
        markup: bool = False,
        emoji: bool = False,
        password: bool = False,
        stream: TextIO | None = None,
    ) -> str:
        """Read a line after showing *prompt* as literal text."""
        return super().input(prompt, markup=markup, emoji=emoji, password=password, stream=stream)

    def status(
        self,
        status: RenderableType,
        *,
        spinner: str = "dots",
        spinner_style: StyleType = "status.spinner",
        speed: float = 1.0,
        refresh_per_second: float = 12.5,
    ) -> Status:
        """A spinner whose *status* string shows as literal text."""
        return _LiteralStatus(
            Text(status) if isinstance(status, str) else status,
            console=self,
            spinner=spinner,
            spinner_style=spinner_style,
            speed=speed,
            refresh_per_second=refresh_per_second,
        )


def styled(*parts: str | tuple[str, StyleType]) -> Text:
    """Literal text from ``(text, style)`` parts, highlighted the way a printed string is."""
    assembled = Text.assemble(*parts)
    highlighted = ReprHighlighter()(Text(assembled.plain))
    highlighted.copy_styles(assembled)
    return highlighted
