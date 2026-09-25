"""Progress bars show task descriptions and details as literal text."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from rich.console import Console
from rich.progress import Progress

from lilbee.runtime.progress.columns import literal_text_column

_SRC = Path(__file__).resolve().parent.parent / "src" / "lilbee"
_TASK_TEXT_FIELDS = ("task.description", "task.fields")


@pytest.mark.parametrize(
    "description",
    ["Ingested C:\\notes\\[/x].md", "Ingested note[red].md", "Crawled 1/?: https://h/a[b"],
)
def test_a_bracketed_description_renders_as_written(description: str) -> None:
    console = Console(force_terminal=False, width=200)
    progress = Progress(literal_text_column("{task.description}"), console=console)
    progress.add_task(description, total=None)
    with console.capture() as capture:
        console.print(progress.make_tasks_table(progress.tasks))
    assert description in capture.get()


def _task_text_columns_parsing_markup() -> list[str]:
    """Every direct ``TextColumn`` over task text, and every column-less ``Progress``."""
    hits: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else ""
            name = node.func.id if isinstance(node.func, ast.Name) else name
            fmt = node.args[0] if node.args else None
            reads_task_text = (
                isinstance(fmt, ast.Constant)
                and isinstance(fmt.value, str)
                and any(field in fmt.value for field in _TASK_TEXT_FIELDS)
            )
            markup_off = any(
                kw.arg == "markup"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in node.keywords
            )
            if name == "TextColumn" and reads_task_text and not markup_off:
                hits.append(f"{path.relative_to(_SRC)}:{node.lineno}")
            if name == "Progress" and not node.args:
                hits.append(f"{path.relative_to(_SRC)}:{node.lineno} (default columns)")
    return hits


def test_no_progress_bar_parses_task_text_as_markup() -> None:
    """Descriptions carry file names and URLs, so every bar uses the literal column."""
    assert _task_text_columns_parsing_markup() == []
