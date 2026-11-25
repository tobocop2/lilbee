"""A fresh interpreter imports each provider-layer entry point first without a cycle."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import lilbee.catalog
import lilbee.core

_CATALOG_DIR = Path(lilbee.catalog.__file__).parent
_CORE_DIR = Path(lilbee.core.__file__).parent


def test_config_import_leaves_the_app_services_unloaded(tmp_path: Path) -> None:
    """A model ref in config.toml runs the ref validator, which reaches the catalog, not app."""
    (tmp_path / "config.toml").write_text(
        'chat_model = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"\n', encoding="utf-8"
    )
    probe = (
        "import sys, lilbee.core.config as c; "
        "print(c.cfg.chat_model); print('lilbee.app.services' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "LILBEE_DATA": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf", "False"]


def _reaches_app(node: ast.AST) -> bool:
    """True for ``import lilbee.app``, ``from lilbee.app import x`` or ``from lilbee import app``."""
    if isinstance(node, ast.Import):
        return any(a.name.startswith("lilbee.app") for a in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return module.startswith("lilbee.app") or (
            module == "lilbee" and any(a.name == "app" for a in node.names)
        )
    return False


def _upward_import_lines(source: str) -> list[int]:
    """Lines of every app import in *source*; every node counts, so a function body too."""
    return [node.lineno for node in ast.walk(ast.parse(source)) if _reaches_app(node)]


@pytest.mark.parametrize(
    "source",
    [
        "import lilbee.app.search",
        "from lilbee.app.search import x",
        "from lilbee import app",
        "from lilbee import config, app",
        "def f():\n    from lilbee import app\n",
        "def f():\n    from lilbee.app.search import x\n",
    ],
)
def test_the_rule_catches_each_way_to_reach_app(source: str) -> None:
    assert _upward_import_lines(source) != []


@pytest.mark.parametrize(
    "source",
    ["from lilbee import config", "from lilbee.core import app", "import lilbee.catalog"],
)
def test_the_rule_passes_imports_that_do_not_reach_app(source: str) -> None:
    assert _upward_import_lines(source) == []


@pytest.mark.parametrize("path", sorted(_CATALOG_DIR.glob("*.py")), ids=lambda p: p.name)
def test_catalog_module_imports_nothing_from_app(path: Path) -> None:
    """catalog is layer 1: app is layer 4, so a caller passes app's objects in.

    Every node counts, so an import inside a function is an upward import too.
    """
    assert _upward_import_lines(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", sorted(_CORE_DIR.rglob("*.py")), ids=lambda p: p.name)
def test_core_module_imports_nothing_from_app(path: Path) -> None:
    """core sits below app, so a caller passes app's objects in, also inside a function."""
    assert _upward_import_lines(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "module",
    [
        "lilbee.providers.fleet.engine_reaper",
        "lilbee.providers.fleet.swap_manager",
        "lilbee.providers.roles",
        "lilbee.providers.reload_keys",
    ],
)
def test_module_imports_first_in_fresh_interpreter(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stderr
