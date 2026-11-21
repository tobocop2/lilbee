"""No test reserves a large file: Windows writes real zeros for a truncate or a seek."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from pathlib import Path

import pytest

TESTS = Path(__file__).parent
LIMIT_BYTES = 64 * 1024**2
# Methods of an open file: the size is the first argument, or this keyword.
_FILE_METHODS = {"truncate": "size", "seek": "offset"}
# Functions of the os module: (position, keyword) of the size argument.
_OS_FUNCTIONS = {"truncate": (1, "length"), "ftruncate": (1, "length"), "lseek": (1, "position")}
# (file, receiver, argument) calls whose argument is not a file size.
_NOT_A_FILE_SIZE = frozenset(
    {
        ("test_embedder.py", "embedder", "text"),  # truncates a string
        ("test_embed_truncation_surfacing.py", "embedder", "text"),  # truncates a string
        ("_hang_watchdog.py", "handle", "size"),  # a cursor in the small hang-dump file
    }
)
_OPERATORS: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.LShift: operator.lshift,
}


def _fold(node: ast.expr, names: dict[str, int]) -> int | None:
    """The integer an expression of literals and module constants states, or None."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _fold(node.left, names), _fold(node.right, names)
        if left is None or right is None:
            return None
        return _OPERATORS[type(node.op)](left, right)
    return None


def _module_constants(tree: ast.Module) -> dict[str, int]:
    """Module-level ``NAME = <integer expression>`` assignments."""
    names: dict[str, int] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            value = _fold(stmt.value, names)
            if isinstance(target, ast.Name) and value is not None:
                names[target.id] = value
    return names


def _os_names(tree: ast.Module) -> tuple[set[str], dict[str, str]]:
    """Names bound to the os module, and names bound to one of its functions."""
    modules = {"os"}
    functions: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.asname for a in node.names if a.name == "os" and a.asname)
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            functions.update({a.asname or a.name: a.name for a in node.names})
    return modules, functions


def _argument(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    """The argument given by position or by keyword, else None."""
    if len(call.args) > position:
        return call.args[position]
    return next((kw.value for kw in call.keywords if kw.arg == keyword), None)


def _size_argument(
    call: ast.Call, file_name: str, modules: set[str], functions: dict[str, str]
) -> ast.expr | None:
    """The size argument of a call that cuts or moves within a file, else None."""
    func = call.func
    if isinstance(func, ast.Name) and functions.get(func.id) in _OS_FUNCTIONS:
        return _argument(call, *_OS_FUNCTIONS[functions[func.id]])
    if not isinstance(func, ast.Attribute):
        return None
    receiver = ast.unparse(func.value)
    if receiver in modules and func.attr in _OS_FUNCTIONS:
        return _argument(call, *_OS_FUNCTIONS[func.attr])
    if func.attr not in _FILE_METHODS:
        return None
    argument = _argument(call, 0, _FILE_METHODS[func.attr])
    if argument is not None and (file_name, receiver, ast.unparse(argument)) in _NOT_A_FILE_SIZE:
        return None
    return argument


def large_file_sites(source: str, file_name: str) -> list[tuple[int, str]]:
    """``(line, reason)`` for each size argument above the limit or not stated as a number."""
    tree = ast.parse(source)
    names = _module_constants(tree)
    modules, functions = _os_names(tree)
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        argument = _size_argument(node, file_name, modules, functions)
        if argument is None:
            continue
        size = _fold(argument, names)
        if size is None:
            sites.append((node.lineno, f"size {ast.unparse(argument)!r} is not a literal"))
        elif size > LIMIT_BYTES:
            sites.append((node.lineno, f"{size} bytes is above {LIMIT_BYTES}"))
    return sites


def test_no_test_truncates_or_seeks_past_the_limit() -> None:
    files = sorted(TESTS.rglob("*.py"))
    assert len(files) > 100
    found = [
        f"{path.relative_to(TESTS.parent).as_posix()}:{line}: {reason}"
        for path in files
        for line, reason in large_file_sites(path.read_text(encoding="utf-8"), path.name)
    ]
    assert found == [], (
        "State the size by patch instead (see tests/_stated_sizes.py):\n" + "\n".join(found)
    )


@pytest.mark.parametrize(
    "source",
    [
        "handle.truncate(20 * 1024**3)",
        "os.truncate(path, 2 * 1024**3)",
        "handle.seek(1 << 40)",
        "GIB = 1024**3\nhandle.truncate(GIB)",
        "handle.truncate(size)",
        "handle.truncate(size=20 * 1024**3)",
        "os.truncate(path, length=20 * 1024**3)",
        "handle.seek(offset=20 * 1024**3)",
        "from os import truncate\ntruncate(path, 20 * 1024**3)",
        "from os import truncate as cut\ncut(path, 20 * 1024**3)",
        "import os as system\nsystem.truncate(path, 20 * 1024**3)",
        "os.lseek(fd, 20 * 1024**3, 0)",
        "os.ftruncate(fd, 20 * 1024**3)",
    ],
    ids=[
        "truncate",
        "os-truncate",
        "seek",
        "named-constant",
        "unstated-size",
        "truncate-keyword",
        "os-truncate-keyword",
        "seek-keyword",
        "from-os-import",
        "from-os-import-alias",
        "module-alias",
        "lseek",
        "ftruncate",
    ],
)
def test_the_scan_flags_a_large_or_unstated_size(source: str) -> None:
    assert len(large_file_sites(source, "test_x.py")) == 1


@pytest.mark.parametrize(
    ("file_name", "source"),
    [
        ("test_embedder.py", "embedder.truncate(20 << 30)"),
        ("test_embed_truncation_surfacing.py", "embedder.truncate(size=20 << 30)"),
        ("_hang_watchdog.py", "handle.seek(10 << 30)"),
        ("_hang_watchdog.py", "handle.seek(offset=10 << 30)"),
    ],
    ids=["embedder", "embedder-keyword", "watchdog", "watchdog-keyword"],
)
def test_an_exempt_call_site_still_fails_on_a_large_number(file_name: str, source: str) -> None:
    assert len(large_file_sites(source, file_name)) == 1


@pytest.mark.parametrize(
    "source",
    [
        "handle.truncate()",
        "handle.seek(0)",
        "handle.truncate(1024)",
        "os.truncate(path, 64 * 1024**2)",
    ],
    ids=["whole-file", "start", "small", "at-the-limit"],
)
def test_the_scan_passes_a_small_call(source: str) -> None:
    assert large_file_sites(source, "test_x.py") == []


@pytest.mark.parametrize(
    ("file_name", "source"),
    [
        ("test_embedder.py", "embedder.truncate(text)"),
        ("test_embed_truncation_surfacing.py", "embedder.truncate(text)"),
        ("_hang_watchdog.py", "handle.seek(size)"),
    ],
    ids=["embedder", "embedder-surfacing", "watchdog"],
)
def test_the_scan_passes_the_exempt_call(file_name: str, source: str) -> None:
    assert large_file_sites(source, file_name) == []
