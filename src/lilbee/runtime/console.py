"""The Rich console lilbee prints through, which shows every string as literal text."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any, TextIO, TypeAlias

from rich.console import Console, RenderableType
from rich.highlighter import ReprHighlighter
from rich.progress import Progress, ProgressColumn
from rich.style import StyleType
from rich.text import Text, TextType

if TYPE_CHECKING:
    from rich.status import Status


ProgressDisplay: TypeAlias = Progress
"""The type of a progress display; build one with ``live_progress`` or ``stdout_progress``."""


def live_progress(
    *columns: str | ProgressColumn, console: Console, **options: Any
) -> ProgressDisplay:
    """A progress display that leaves ``sys.stdout`` and ``sys.stderr`` alone.

    Rich redirects both streams into the display's console by default. A stderr display would
    then take the text of a stdout print and show it on stderr. *options* go to ``Progress``.
    """
    return Progress(
        *columns, console=console, redirect_stdout=False, redirect_stderr=False, **options
    )


def stdout_progress(
    *columns: str | ProgressColumn, console: Console, **options: Any
) -> ProgressDisplay:
    """A progress display on a stdout console, with Rich's default stream redirects.

    The display shows text another part of the program writes to ``sys.stderr`` above the bar.
    *options* go to ``Progress``.
    """
    return Progress(*columns, console=console, **options)


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
        )

    def input(
        self,
        prompt: TextType = "",
        *,
        markup: bool = False,
        emoji: bool = True,
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
        return super().status(
            Text(status) if isinstance(status, str) else status,
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
