"""The install page says which llama.cpp pack cells read memory from the startup log."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._pack_workflow import pack_cells, pack_pins

_ROOT = Path(__file__).resolve().parents[1]
_PAGE = _ROOT / "docs/engines/install.md"
_BUILD_DIR = _ROOT / "tools/wheel-build"
# Every script of the pack recipe that clones llama.cpp source to build or to collect licences.
_CLONING_SCRIPTS = ("pack_llama.sh", "llama_gpu_manylinux.sh", "llama_cuda_manylinux.sh")
_PINNED_CLONE_URL = '"https://github.com/${UPSTREAM_REPO}.git"'
_UPSTREAM_REPO = "ggml-org/llama.cpp"
_LOG = "startup log"
_ENDPOINT = "`GET /memory`"
_ROW = re.compile(r"^\| `(?P<cell>[^`]+)` \| (?P<mode>[^|]+) \|$", re.MULTILINE)
_BLOCK = re.compile(
    r"<!-- readback-cells:start -->\n(?P<body>.*?)<!-- readback-cells:end -->", re.DOTALL
)
_FORK_PIN = re.compile(r"^ENGINE_LLAMA_CPP_REPO=https://github\.com/(?P<repo>\S+?)(?:\.git)?$", re.M)


def _fork_repo() -> str:
    """The one repository that serves GET /memory: the fork of the lilbee source pin."""
    pin = _FORK_PIN.search((_ROOT / "engine-versions.env").read_text(encoding="utf-8"))
    assert pin, "engine-versions.env lost ENGINE_LLAMA_CPP_REPO"
    return pin["repo"]


def _mode_of_repo(repo: str) -> str:
    """The readback mode of a build from *repo*; an unknown repo is an error, not a guess."""
    if repo == _UPSTREAM_REPO:
        return _LOG
    if repo == _fork_repo():
        return _ENDPOINT
    raise AssertionError(f"no readback mode is known for {repo}")


def _llama_clones(script: str) -> list[str]:
    """The `git clone` lines of *script* that fetch llama.cpp at UPSTREAM_TAG."""
    lines = (line.strip() for line in script.splitlines())
    return [line for line in lines if "git clone" in line and "${UPSTREAM_TAG}" in line]


def _clones_outside_the_pin(script: str) -> list[str]:
    """The llama.cpp clones of *script* that do not clone the repository in UPSTREAM_REPO."""
    return [line for line in _llama_clones(script) if _PINNED_CLONE_URL not in line]


def _expected() -> dict[str, str]:
    mode = _mode_of_repo(pack_pins("llama")["UPSTREAM_REPO"])
    return {cell["cell"]: mode for cell in pack_cells("llama")}


def _noted() -> dict[str, str]:
    block = _BLOCK.search(_PAGE.read_text(encoding="utf-8"))
    assert block, "the install page lost the readback-cells block"
    return {m["cell"]: m["mode"].strip() for m in _ROW.finditer(block["body"])}


@pytest.mark.parametrize("name", _CLONING_SCRIPTS)
def test_every_llama_clone_of_the_pack_recipe_uses_the_pinned_repo(name: str) -> None:
    script = (_BUILD_DIR / name).read_text(encoding="utf-8")
    assert _llama_clones(script)
    assert _clones_outside_the_pin(script) == []


@pytest.mark.parametrize("name", _CLONING_SCRIPTS)
def test_a_clone_of_a_fork_in_a_recipe_script_is_found(name: str) -> None:
    script = (_BUILD_DIR / name).read_text(encoding="utf-8")
    forked = script.replace(
        "https://github.com/${UPSTREAM_REPO}.git", "https://github.com/tobocop2/llama.cpp.git"
    )
    assert forked != script
    assert len(_clones_outside_the_pin(forked)) == len(_llama_clones(script))


def test_the_install_page_lists_the_readback_mode_of_every_cell() -> None:
    assert _noted() == _expected()


def test_the_mode_rule_knows_upstream_and_the_source_pin_fork_only() -> None:
    assert _mode_of_repo(_UPSTREAM_REPO) == _LOG
    assert _mode_of_repo(_fork_repo()) == _ENDPOINT
    with pytest.raises(AssertionError, match="no readback mode is known for someone/llama.cpp"):
        _mode_of_repo("someone/llama.cpp")
