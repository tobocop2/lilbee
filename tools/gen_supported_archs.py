#!/usr/bin/env python3
"""Regenerate ``lilbee/_generated/engine_archs.py`` from the pinned engine.

The set of architectures lilbee will pull has to be the set the bundled engine can
actually load, so it is read from the engine's own arch table rather than from a
third-party package that happens to enumerate GGUF architecture names.

``engine-versions.env`` pins the engine as a llama-cpp-python release tag. That is a
build-time source coordinate only, not a dependency: ``build_llama_server.sh`` clones
that tag with ``--recurse-submodules`` and compiles llama-server from the llama.cpp
commit it vendors, and lilbee ships neither package. This resolves the same submodule
commit over the GitHub API, so the architectures below are the ones the shipped binary
was actually built with, then reads them from that commit's ``src/llama-arch.cpp``,
which maps every ``LLM_ARCH_*`` to the ``general.architecture`` string a GGUF carries.
Nothing here is imported at runtime, and CI needs no llama.cpp checkout.

Run after bumping ``ENGINE_LLAMA_CPP_VERSION`` (``make engine-archs``); the check in
``tests/test_engine_archs.py`` fails when the generated file is left behind.
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

import httpx
from jinja2 import Template

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENGINE_ENV = _REPO_ROOT / "engine-versions.env"
_OUT = _REPO_ROOT / "src" / "lilbee" / "_generated" / "engine_archs.py"
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_TEMPLATE = "engine_archs.py.jinja"

_LLAMA_CPP_PYTHON = "abetlen/llama-cpp-python"
_LLAMA_CPP = "ggml-org/llama.cpp"
_ARCH_TABLE_PATH = "src/llama-arch.cpp"

# { LLM_ARCH_LLAMA, "llama" } entries of llama.cpp's LLM_ARCH_NAMES table.
_ARCH_ENTRY_RE = re.compile(r'\{\s*LLM_ARCH_[A-Z0-9_]+\s*,\s*"([^"]+)"\s*\}')

# LLM_ARCH_UNKNOWN's name. A sentinel for "not recognised", never a loadable model.
_UNKNOWN_ARCH = "(unknown)"

def _get_json(url: str) -> dict:
    resp = httpx.get(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "lilbee"},
        follow_redirects=True,
        timeout=30.0,
    )
    resp.raise_for_status()
    return dict(resp.json())


def engine_version(env_path: Path) -> str:
    """``ENGINE_LLAMA_CPP_VERSION`` from engine-versions.env."""
    for line in env_path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "ENGINE_LLAMA_CPP_VERSION":
            return value.strip()
    raise SystemExit(f"ENGINE_LLAMA_CPP_VERSION not found in {env_path}")


def vendored_commit(version: str) -> str:
    """The llama.cpp commit the given llama-cpp-python release vendors."""
    url = (
        f"https://api.github.com/repos/{_LLAMA_CPP_PYTHON}/contents/vendor/llama.cpp?ref=v{version}"
    )
    entry = _get_json(url)
    if entry.get("type") != "submodule":
        raise SystemExit(f"vendor/llama.cpp is not a submodule at v{version}")
    return str(entry["sha"])


def arch_names(commit: str) -> frozenset[str]:
    """Every ``general.architecture`` string llama.cpp maps to an arch at *commit*."""
    url = f"https://api.github.com/repos/{_LLAMA_CPP}/contents/{_ARCH_TABLE_PATH}?ref={commit}"
    source = base64.b64decode(_get_json(url)["content"]).decode("utf-8", "replace")
    names = set(_ARCH_ENTRY_RE.findall(source))
    if not names:
        raise SystemExit(f"no LLM_ARCH entries parsed from {_ARCH_TABLE_PATH} at {commit}")
    return frozenset(names - {_UNKNOWN_ARCH})


def render(version: str, commit: str, archs: frozenset[str]) -> str:
    """The generated module, laid out as ``ruff format`` leaves it.

    A Python module rather than a data file on purpose: lilbee's standalone builds
    register package data one ``--include-data-files`` line at a time, so a JSON or
    TOML list would have to be added to every packaging path and would fail at
    runtime in whichever one got missed. A module needs no registration anywhere.
    """
    template = Template(
        (_TEMPLATE_DIR / _TEMPLATE).read_text(),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return template.render(version=version, commit=commit, archs=sorted(archs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the generated file is out of date, writing nothing",
    )
    args = ap.parse_args()

    version = engine_version(_ENGINE_ENV)
    commit = vendored_commit(version)
    archs = arch_names(commit)
    rendered = render(version, commit, archs)

    if args.check:
        current = _OUT.read_text() if _OUT.exists() else ""
        if current != rendered:
            print(f"{_OUT} is out of date; run: make engine-archs", file=sys.stderr)
            return 1
        print(f"{_OUT} is up to date ({len(archs)} architectures)")
        return 0

    _OUT.write_text(rendered)
    print(f"wrote {_OUT}: {len(archs)} architectures from llama.cpp {commit[:12]} (v{version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
