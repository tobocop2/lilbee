"""The ``[engine]`` extra is gone: lilbee downloads an engine pack instead."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# An extras list that names engine, such as [engine] or [cuda,engine]; a link text is not one.
_EXTRA_LIST = re.compile(r"\[(?:[\w-]+\s*,\s*)*engine(?:\s*,\s*[\w-]+)*\]")
# The pattern before spaces were allowed around the commas.
_NO_SPACE_EXTRA_LIST = re.compile(r"\[(?:[\w-]+,)*engine(?:,[\w-]+)*\]")
DOCS = (
    "CONTRIBUTING.md",
    "README.md",
    "docs/install.md",
    "docs/usage.md",
    "docs/engines/README.md",
    "docs/engines/auto.md",
    "docs/engines/choose.md",
    "docs/engines/install.md",
    "docs/architecture.md",
    "docs/internals/release.md",
    "packaging/engine-wheel/README.md",
    "site/index.html",
)


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
        assert not _EXTRA_LIST.search(text), name
        assert ",engine" not in text, name


def test_a_link_named_engine_is_not_an_extra() -> None:
    assert not _EXTRA_LIST.search("Details: [engine internals](internals/engine.md).")
    for spelling in (
        "lilbee[engine]",
        "lilbee[engine,crawler]",
        "lilbee[cuda,engine]",
        "lilbee[engine, crawler]",
        "lilbee[cuda, engine]",
        "lilbee[cuda , engine , crawler]",
    ):
        assert _EXTRA_LIST.search(spelling), spelling


def test_the_pattern_before_spaces_missed_a_spaced_extras_list() -> None:
    for spelling in ("lilbee[engine, crawler]", "lilbee[cuda, engine]"):
        assert not _NO_SPACE_EXTRA_LIST.search(spelling), spelling
        assert _EXTRA_LIST.search(spelling), spelling


def test_no_command_installs_the_wheel_from_an_index_that_lacks_it() -> None:
    for name in ("README.md", "docs/install.md", "docs/usage.md", "site/index.html"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "lilbee-engine-llama-cpp --extra-index-url" not in text, name
        assert "lilbee-engine-llama-cpp lilbee" not in text, name


def test_the_compat_rows_do_not_say_the_install_has_no_engine() -> None:
    for name in ("README.md", "docs/install.md", "site/index.html"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "without an engine" not in text, name
        assert "no engine:" not in text, name
