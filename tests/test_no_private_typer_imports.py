"""The typer floor in pyproject must hold: lilbee may import only public Typer modules."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src" / "lilbee"


def _private_typer_imports(source: str) -> list[str]:
    """Return every import in *source* that reaches a private module of typer."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
            if node.module == "typer":
                names += [f"typer.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        found += [n for n in names if n.split(".")[0] == "typer" and "._" in n]
    return found


@pytest.mark.parametrize(
    "source",
    [
        "from typer._click import Context",
        "import typer._click",
        "from typer import _click",
    ],
)
def test_detector_flags_a_private_typer_import(source: str) -> None:
    assert _private_typer_imports(source)


@pytest.mark.parametrize(
    "source",
    ["import typer", "from typer.core import TyperCommand", "from typer import Context"],
)
def test_detector_passes_a_public_typer_import(source: str) -> None:
    assert _private_typer_imports(source) == []


def test_source_imports_no_private_typer_module() -> None:
    offenders = {
        str(path.relative_to(_SRC)): hits
        for path in _SRC.rglob("*.py")
        if (hits := _private_typer_imports(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}
