"""The CLI consoles show every string as literal text, never as Rich markup."""

from __future__ import annotations

import io
from collections.abc import Callable

import pytest
from rich.console import Console
from rich.table import Table
from rich.text import Span, Text

from lilbee.app.models import ShowModelResult
from lilbee.cli.app import console as shared_console
from lilbee.cli.model import _render_show
from lilbee.runtime.console import PlainConsole, styled

_VALUES = ("note[red]x[/red].md", "C:\\docs\\[/]draft.md", "q[/x]y")


def _captured(con: Console, emit: Callable[[Console], object]) -> str:
    with con.capture() as capture:
        emit(con)
    return capture.get()


@pytest.mark.parametrize("value", _VALUES)
@pytest.mark.parametrize(
    "emit",
    [
        pytest.param(lambda c, v: c.print(v), id="variable"),
        pytest.param(lambda c, v: c.print("Saved %s" % v), id="percent_format"),  # noqa: UP031
        pytest.param(lambda c, v: c.print("Saved " + v), id="concatenation"),
        pytest.param(lambda c, v: c.print("Saved", v), id="second_positional"),
        pytest.param(lambda c, v: c.log(v), id="log"),
        pytest.param(lambda c, v: c.rule(v), id="rule"),
        pytest.param(lambda c, v: c.input(f"Name {v}: ", stream=io.StringIO("y\n")), id="input"),
    ],
)
def test_the_shared_console_prints_a_bracketed_value_literally(
    emit: Callable[[Console, str], object], value: str
) -> None:
    out = _captured(shared_console, lambda c: emit(c, value))
    assert value in out


@pytest.mark.parametrize("value", _VALUES)
def test_a_table_shows_a_bracketed_cell_and_title_literally(value: str) -> None:
    table = Table(title=value)
    table.add_column("File")
    table.add_row(value)
    out = _captured(PlainConsole(width=200), lambda c: c.print(table))
    assert out.count(value) == 2


@pytest.mark.parametrize("value", _VALUES)
def test_a_status_line_shows_a_bracketed_value_literally(value: str) -> None:
    con = PlainConsole(file=io.StringIO())
    with con.status(value) as status:
        spinner_text = status.renderable.text
    assert isinstance(spinner_text, Text)
    assert spinner_text.plain == value


def test_the_shared_console_is_a_plain_console() -> None:
    assert isinstance(shared_console, PlainConsole)


def test_styled_keeps_the_style_and_the_value_literal() -> None:
    line = styled(("Deleted ", "bold"), "a[red]b[/red]")
    assert line.plain == "Deleted a[red]b[/red]"
    assert Span(0, len("Deleted "), "bold") in line.spans


def test_styled_renders_like_the_markup_string_it_replaces() -> None:
    """Markup styles win over the highlighter, as they do in a printed markup string."""
    markup_console = Console(force_terminal=True, width=100)
    plain_console = PlainConsole(force_terminal=True, width=100)
    as_markup = _captured(markup_console, lambda c: c.print("Lint: [red]1 error(s)[/red], 'x' 2"))
    as_styled = _captured(
        plain_console, lambda c: c.print(styled("Lint: ", ("1 error(s)", "red"), ", 'x' 2"))
    )
    assert as_styled == as_markup


def test_model_show_prints_a_bracketed_name_and_path_literally() -> None:
    data = ShowModelResult(model="org/m[q4]", installed=True, source="native", path="/m/a[/b].gguf")
    out = _captured(PlainConsole(width=200), lambda c: c.print(_render_show(data)))
    assert "org/m[q4]" in out
    assert "/m/a[/b].gguf" in out
