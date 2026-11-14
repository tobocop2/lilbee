"""Dataclass field defaults must be values Python 3.11 accepts (it rejects unhashable defaults)."""

from __future__ import annotations

import ast
from pathlib import Path

import lilbee

_SRC = Path(lilbee.__file__).parent
_UNHASHABLE_CALLS = {"MappingProxyType", "dict", "list", "set", "defaultdict", "OrderedDict"}


def _is_dataclass(node: ast.ClassDef) -> bool:
    for deco in node.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name == "dataclass":
            return True
    return False


def _unhashable_default(value: ast.expr) -> bool:
    if isinstance(value, ast.Dict | ast.List | ast.Set):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        return name in _UNHASHABLE_CALLS
    return False


def _bad_defaults(source: str) -> tuple[int, list[str]]:
    """Return the dataclass count and the ``Class.field`` names with an unhashable default."""
    classes = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)]
    dataclasses_ = [c for c in classes if _is_dataclass(c)]
    bad = [
        f"{c.name}.{stmt.target.id}"
        for c in dataclasses_
        for stmt in c.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and stmt.value is not None
        and _unhashable_default(stmt.value)
    ]
    return len(dataclasses_), bad


def test_scanner_flags_a_mapping_proxy_default() -> None:
    source = (
        "from dataclasses import dataclass, field\n"
        "@dataclass(frozen=True)\n"
        "class P:\n"
        "    a: Mapping[str, int] = MappingProxyType({})\n"
        "    b: dict[str, int] = field(default_factory=dict)\n"
        "    c: int = 0\n"
        "    d: list[int] = []\n"
        "    e: int = make_dict_default()\n"
        "@dataclass\n"
        "class Q:\n"
        "    f: set[int] = set()\n"
        "class NotADataclass:\n"
        "    g: dict[str, int] = {}\n"
    )
    assert _bad_defaults(source) == (2, ["P.a", "P.d", "Q.f"])


def test_no_dataclass_field_default_is_unhashable() -> None:
    count = 0
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        found, bad = _bad_defaults(path.read_text(encoding="utf-8"))
        count += found
        offenders += [f"{path.relative_to(_SRC).as_posix()}: {name}" for name in bad]
    assert count > 50
    assert offenders == []
