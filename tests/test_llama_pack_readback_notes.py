"""The install page says which llama.cpp pack cells read memory from the startup log."""

from __future__ import annotations

import re
from pathlib import Path

from tests._pack_workflow import pack_cells, pack_pins

_ROOT = Path(__file__).resolve().parents[1]
_PAGE = _ROOT / "docs/engines/install.md"
_RECIPE = _ROOT / "tools/wheel-build/pack_llama.sh"
_UPSTREAM_REPO = "ggml-org/llama.cpp"
_LOG = "startup log"
_ENDPOINT = "`GET /memory`"
_ROW = re.compile(r"^\| `(?P<cell>[^`]+)` \| (?P<mode>[^|]+) \|$", re.MULTILINE)
_BLOCK = re.compile(
    r"<!-- readback-cells:start -->\n(?P<body>.*?)<!-- readback-cells:end -->", re.DOTALL
)


def _mode_of_pins(pins: dict[str, str]) -> str:
    """Upstream llama.cpp has no GET /memory, so a build from it reads the log."""
    return _LOG if pins["UPSTREAM_REPO"] == _UPSTREAM_REPO else _ENDPOINT


def _expected() -> dict[str, str]:
    mode = _mode_of_pins(pack_pins("llama"))
    return {cell["cell"]: mode for cell in pack_cells("llama")}


def _noted() -> dict[str, str]:
    block = _BLOCK.search(_PAGE.read_text(encoding="utf-8"))
    assert block, "the install page lost the readback-cells block"
    return {m["cell"]: m["mode"].strip() for m in _ROW.finditer(block["body"])}


def test_every_source_cell_builds_from_the_pinned_repo() -> None:
    # A source cell clones UPSTREAM_REPO, so the repo pin decides the mode of all cells.
    recipe = _RECIPE.read_text(encoding="utf-8")
    assert 'https://github.com/${UPSTREAM_REPO}.git" "${WORK}/llama.cpp"' in recipe


def test_the_install_page_lists_the_readback_mode_of_every_cell() -> None:
    assert _noted() == _expected()


def test_the_mode_rule_names_the_fork_as_the_memory_endpoint() -> None:
    assert _mode_of_pins({"UPSTREAM_REPO": _UPSTREAM_REPO}) == _LOG
    assert _mode_of_pins({"UPSTREAM_REPO": "tobocop2/llama.cpp"}) == _ENDPOINT
