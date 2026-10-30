"""Options and fixtures of the conformance suite."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

_FAKE_BINARY = Path(__file__).with_name("fake_binary.py")
DEFAULT_BINARY_CFG: dict[str, Any] = {
    "flags": ["--port", "--ctx-size", "--list-devices"],
    "devices": "llama",
    "estimate": True,
    "term": "exit",
    "help_rc": 0,
}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("conformance")
    group.addoption("--engine-url", default=None, help="Base URL of a running engine to check.")
    group.addoption(
        "--claims",
        default="chat",
        help="Comma-separated roles the engine claims: chat, embed, rerank, vision, reasoning.",
    )
    group.addoption("--models", default="", help="Model ids by role, for example chat=ID,embed=ID.")
    group.addoption("--record", default=None, help="Path of the JSON-lines result file to write.")
    group.addoption(
        "--engine-pin",
        default=None,
        help="Path to a pin record (JSON with a binary path): launch it and check tier 2.",
    )


@pytest.fixture
def make_binary(tmp_path: Path) -> Callable[..., Path]:
    """Write an executable stand-in engine whose behaviour is the default plus the overrides."""

    def make(name: str, **overrides: Any) -> Path:
        cfg = {**DEFAULT_BINARY_CFG, **overrides}
        script = (
            f"#!{sys.executable}\n"
            "import os, runpy\n"
            f"os.environ['FAKE_ENGINE_CFG'] = {json.dumps(json.dumps(cfg))}\n"
            f"runpy.run_path({str(_FAKE_BINARY)!r}, run_name='__main__')\n"
        )
        binary = tmp_path / name
        binary.write_text(script, encoding="utf-8")
        binary.chmod(0o755)
        return binary

    return make
