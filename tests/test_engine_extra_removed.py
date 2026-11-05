"""The ``[engine]`` extra is gone: lilbee downloads an engine pack instead."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("README.md", "docs/usage.md", "packaging/engine-wheel/README.md", "site/index.html")


def extras_of(pyproject_text: str) -> set[str]:
    return set(tomllib.loads(pyproject_text)["project"]["optional-dependencies"])


def test_pyproject_declares_no_engine_extra() -> None:
    assert "engine" not in extras_of((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_lock_declares_no_engine_extra() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    (lilbee,) = (p for p in lock["package"] if p["name"] == "lilbee")
    assert "engine" not in lilbee["metadata"]["provides-extras"]
    assert "engine" not in lilbee["optional-dependencies"]


def test_no_install_text_names_the_engine_extra() -> None:
    for name in DOCS:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "[engine" not in text, name
        assert ",engine" not in text, name
