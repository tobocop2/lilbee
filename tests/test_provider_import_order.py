"""A fresh interpreter imports each provider-layer entry point first without a cycle."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import lilbee.catalog

_CATALOG_DIR = Path(lilbee.catalog.__file__).parent


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


@pytest.mark.parametrize("path", sorted(_CATALOG_DIR.glob("*.py")), ids=lambda p: p.name)
def test_catalog_module_imports_nothing_from_app_at_module_level(path: Path) -> None:
    """catalog is layer 1: app is layer 4, so a caller passes app's objects in."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    upward = [
        node.lineno
        for node in tree.body
        if (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("lilbee.app"))
        or (
            isinstance(node, ast.Import)
            and any(a.name.startswith("lilbee.app") for a in node.names)
        )
    ]
    assert upward == []


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
