"""A test that sets PYTHONPATH for a child process builds the value with ``child_pythonpath``."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_NAME = "PYTHONPATH"
_HELPER = "child_pythonpath"
# Each setter that takes the variable name as an argument, with that argument's position.
_SETTERS = {"setenv": 0, "putenv": 0, "setdefault": 0, "setitem": 1}
_ALLOWED = {
    "conftest.py": "the sealing fixture builds the sealed value",
    "test_sealed_engine_children.py": "plants a host engine wheel on PYTHONPATH on purpose",
}


def _is_name(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value == _NAME


def _callee_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _written_values(tree: ast.AST) -> list[ast.expr]:
    """The value of every write to PYTHONPATH: dict key, subscript store, setter call, keyword."""
    values: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            pairs = zip(node.keys, node.values, strict=True)
            values += [value for key, value in pairs if _is_name(key)]
        elif isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is not None and any(
                isinstance(target, ast.Subscript) and _is_name(target.slice) for target in targets
            ):
                values.append(node.value)
        elif isinstance(node, ast.Call):
            values += [keyword.value for keyword in node.keywords if keyword.arg == _NAME]
            callee = _callee_name(node)
            at = _SETTERS.get(callee) if callee else None
            if at is not None and len(node.args) > at + 1 and _is_name(node.args[at]):
                values.append(node.args[at + 1])
    return values


def _is_helper_call(value: ast.expr) -> bool:
    """True only when the value IS a call of the helper, not an expression that holds one."""
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    return (isinstance(func, ast.Name) and func.id == _HELPER) or (
        isinstance(func, ast.Attribute) and func.attr == _HELPER
    )


def pythonpath_writes(source: str) -> list[tuple[int, bool]]:
    """Each write to PYTHONPATH in *source* as ``(line, sealed)``."""
    return [(value.lineno, _is_helper_call(value)) for value in _written_values(ast.parse(source))]


_UNSEALED = {
    "setenv": 'monkeypatch.setenv("PYTHONPATH", "x")',
    "subscript": 'env["PYTHONPATH"] = "x"',
    "keyword": 'dict(PYTHONPATH="x")',
    "setitem": 'monkeypatch.setitem(os.environ, "PYTHONPATH", "x")',
    "dict key": '{"PYTHONPATH": "x"}',
    "dead branch": '{"PYTHONPATH": child_pythonpath("x") if False else "y"}',
    "wrapped helper": '{"PYTHONPATH": os.pathsep.join([child_pythonpath("x")])}',
    "augmented": 'env["PYTHONPATH"] += "x"',
    "wrapped onto the next line": '{"PYTHONPATH":\n    "x"}',
}

_SEALED = {
    "dict key": '{"PYTHONPATH": child_pythonpath("x")}',
    "setenv": 'monkeypatch.setenv("PYTHONPATH", child_pythonpath("x"))',
    "subscript": 'env["PYTHONPATH"] = child_pythonpath("x")',
    "keyword": 'dict(PYTHONPATH=child_pythonpath("x"))',
    "setitem": 'monkeypatch.setitem(os.environ, "PYTHONPATH", child_pythonpath("x"))',
    "module attribute": '{"PYTHONPATH": conftest.child_pythonpath("x")}',
    "wrapped onto the next line": '{"PYTHONPATH":\n    child_pythonpath(\n        "x")}',
}


@pytest.mark.parametrize("source", _UNSEALED.values(), ids=_UNSEALED.keys())
def test_a_write_without_the_helper_is_found(source: str) -> None:
    assert [sealed for _, sealed in pythonpath_writes(source)] == [False]


@pytest.mark.parametrize("source", _SEALED.values(), ids=_SEALED.keys())
def test_a_write_with_the_helper_passes(source: str) -> None:
    assert [sealed for _, sealed in pythonpath_writes(source)] == [True]


@pytest.mark.parametrize(
    "source",
    [
        'os.environ.get("PYTHONPATH", "")',
        'env.pop("PYTHONPATH", None)',
        'x = "PYTHONPATH"',
        'handlers[0]("PYTHONPATH", "x")',
    ],
    ids=["get", "pop", "name", "call of an expression"],
)
def test_a_read_of_pythonpath_is_no_write(source: str) -> None:
    assert pythonpath_writes(source) == []


def test_no_test_sets_pythonpath_without_the_helper() -> None:
    found: dict[str, list[tuple[int, bool]]] = {}
    for path in sorted(_TESTS_DIR.rglob("*.py")):
        name = path.relative_to(_TESTS_DIR).as_posix()
        if name not in _ALLOWED:
            found[name] = pythonpath_writes(path.read_text(encoding="utf-8"))
    sealed = [name for name, writes in found.items() for _, ok in writes if ok]
    unsealed = [f"{name}:{line}" for name, writes in found.items() for line, ok in writes if not ok]
    assert len(sealed) >= 7
    assert unsealed == []
