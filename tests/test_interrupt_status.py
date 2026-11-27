"""The Ctrl+C exit status has one definition, in the cancellation module."""

from __future__ import annotations

import ast
import signal
from pathlib import Path

import lilbee
from lilbee.runtime.cancellation import INTERRUPTED_EXIT_STATUS

_SOURCE_ROOT = Path(lilbee.__file__).parent
_SIGINT_SUM = ("128 + signal.SIGINT", "128 + int(signal.SIGINT)")
_ENDING_CALLS = ("SystemExit", "typer.Exit", "sys.exit", "os._exit")


def _defines_the_status(tree: ast.AST) -> bool:
    """True when the tree assigns 130 or 128 + SIGINT, or ends a command with the literal 130."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AnnAssign):
            if isinstance(node.value, ast.Constant) and node.value.value == 130:
                return True
            if isinstance(node.value, ast.BinOp) and ast.unparse(node.value) in _SIGINT_SUM:
                return True
        elif isinstance(node, ast.Call) and ast.unparse(node.func) in _ENDING_CALLS:
            if any(isinstance(arg, ast.Constant) and arg.value == 130 for arg in node.args):
                return True
    return False


def test_one_module_defines_the_interrupt_status() -> None:
    sources = sorted(_SOURCE_ROOT.rglob("*.py"))
    assert len(sources) > 100
    defining = [
        path.relative_to(_SOURCE_ROOT).as_posix()
        for path in sources
        if _defines_the_status(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert defining == ["runtime/cancellation.py"]


def test_the_status_is_the_shells_for_sigint() -> None:
    assert INTERRUPTED_EXIT_STATUS == 128 + signal.SIGINT == 130


def test_the_census_sees_each_way_to_define_the_status() -> None:
    for text in (
        "X = 130",
        "import signal\nX = 128 + signal.SIGINT",
        "raise SystemExit(130)",
        "raise typer.Exit(130)",
        "os._exit(130)",
        "X: int = 130",
        "import signal\nX: int = 128 + signal.SIGINT",
    ):
        assert _defines_the_status(ast.parse(text)), text
    for text in (
        "X = 7",
        "raise SystemExit(1)",
        "X = 128 + signal.SIGTERM",
        "X: int",
        "os._exit(1)",
    ):
        assert not _defines_the_status(ast.parse(text)), text
