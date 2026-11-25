"""The ``typer>=0.16`` floor holds only if lilbee imports Typer's public API.

Typer 0.16 depends on the click package. Later Typer releases vendor click as
the private ``typer._click`` module, which older releases do not have. A private
import therefore makes the declared floor false.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).parents[1] / "src"


def _private_typer_imports(source: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            parts = node.module.split(".")
            if parts[0] != "typer":
                continue
            if any(p.startswith("_") for p in parts[1:]):
                found.append(node.module)
            else:
                found.extend(
                    f"{node.module}.{alias.name}"
                    for alias in node.names
                    if alias.name.startswith("_")
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "typer" and any(p.startswith("_") for p in parts[1:]):
                    found.append(alias.name)
    return found


def test_source_imports_no_private_typer_module() -> None:
    offenders = {
        str(path.relative_to(_SRC)): hits
        for path in _SRC.rglob("*.py")
        if (hits := _private_typer_imports(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}


def test_detector_flags_a_private_typer_import() -> None:
    assert _private_typer_imports("from typer._click import Context") == ["typer._click"]
    assert _private_typer_imports("import typer._click") == ["typer._click"]
    assert _private_typer_imports("from typer.core import TyperCommand\nimport typer") == []


def test_detector_flags_a_private_name_imported_from_typer() -> None:
    assert _private_typer_imports("from typer import _click") == ["typer._click"]
    assert _private_typer_imports("from typer import _click as c") == ["typer._click"]
    assert _private_typer_imports("from typer import (\n    Option,\n    _click,\n)") == [
        "typer._click"
    ]
    assert _private_typer_imports("from typer import Option, Typer") == []
