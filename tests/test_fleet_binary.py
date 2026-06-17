"""Tests for bundled engine binary resolution (llama-server, llama-swap, gguf-parser)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import binary as binary_mod
from lilbee.providers.fleet.binary import (
    EngineTool,
    llama_server_runtime_env,
    resolve_engine_tool,
    resolve_gguf_parser,
    resolve_llama_server,
    resolve_llama_swap,
)

_WHICH = "lilbee.providers.fleet.binary.shutil.which"

_ALL_TOOLS = [EngineTool.LLAMA_SERVER, EngineTool.LLAMA_SWAP, EngineTool.GGUF_PARSER]


def _fake_engine(tmp_path: Path, *, make_files: bool) -> SimpleNamespace:
    """A stand-in ``lilbee_engine`` module exposing the three path accessors."""
    paths = {}
    for tool in _ALL_TOOLS:
        p = tmp_path / tool.value
        if make_files:
            p.write_text("#!/bin/sh\n")
        paths[tool] = p
    return SimpleNamespace(
        get_llama_server_path=lambda: paths[EngineTool.LLAMA_SERVER],
        get_llama_swap_path=lambda: paths[EngineTool.LLAMA_SWAP],
        get_gguf_parser_path=lambda: paths[EngineTool.GGUF_PARSER],
    )


@pytest.mark.parametrize("tool", _ALL_TOOLS)
def test_resolves_bundled_tool_when_present(
    tool: EngineTool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "lilbee_engine", _fake_engine(tmp_path, make_files=True))
    assert resolve_engine_tool(tool) == tmp_path / tool.value


def test_thin_wrappers_resolve_their_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "lilbee_engine", _fake_engine(tmp_path, make_files=True))
    assert resolve_llama_server() == tmp_path / "llama-server"
    assert resolve_llama_swap() == tmp_path / "llama-swap"
    assert resolve_gguf_parser() == tmp_path / "gguf-parser"


def test_bundled_tool_none_when_package_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Importing a module set to None raises ImportError -> resolution falls through.
    monkeypatch.setitem(sys.modules, "lilbee_engine", None)
    assert binary_mod._bundled_tool(EngineTool.LLAMA_SWAP) is None


def test_bundled_tool_none_when_binary_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Installed wheel whose bin/ has no binary yet (CI fills it at build time).
    monkeypatch.setitem(sys.modules, "lilbee_engine", _fake_engine(tmp_path, make_files=False))
    assert binary_mod._bundled_tool(EngineTool.GGUF_PARSER) is None


def test_llama_server_uses_configured_path_when_file_exists(tmp_path: Path) -> None:
    binpath = tmp_path / "llama-server"
    binpath.write_text("#!/bin/sh\n")
    cfg.llama_server_path = str(binpath)
    assert resolve_llama_server() == binpath


def test_llama_server_raises_when_configured_path_missing(tmp_path: Path) -> None:
    cfg.llama_server_path = str(tmp_path / "absent")
    with pytest.raises(ProviderError, match="is not a file"):
        resolve_llama_server()


def test_llama_server_configured_path_wins_over_bundled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "lilbee_engine", _fake_engine(tmp_path, make_files=True))
    custom = tmp_path / "custom-llama-server"
    custom.write_text("#!/bin/sh\n")
    cfg.llama_server_path = str(custom)
    assert resolve_llama_server() == custom


def test_llama_server_unset_falls_back_to_bundled_then_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg.llama_server_path = ""
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    empty_wheel_dir = tmp_path / "empty-wheel"
    empty_wheel_dir.mkdir()
    monkeypatch.setitem(sys.modules, "lilbee_engine", _fake_engine(wheel_dir, make_files=True))
    monkeypatch.setattr(_WHICH, lambda _name: "/usr/local/bin/llama-server")
    assert resolve_llama_server() == wheel_dir / "llama-server"
    monkeypatch.setitem(
        sys.modules, "lilbee_engine", _fake_engine(empty_wheel_dir, make_files=False)
    )
    assert resolve_llama_server() == Path("/usr/local/bin/llama-server")


def test_llama_server_falls_back_to_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg.llama_server_path = ""
    monkeypatch.setattr(_WHICH, lambda _name: "/usr/local/bin/llama-server")
    assert resolve_llama_server() == Path("/usr/local/bin/llama-server")


def test_llama_server_raises_with_install_hint_when_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg.llama_server_path = ""
    monkeypatch.setattr(_WHICH, lambda _name: None)
    with pytest.raises(ProviderError, match="not found"):
        resolve_llama_server()


def test_aux_tools_ignore_llama_server_path_and_use_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # llama-swap / gguf-parser do not honor LILBEE_LLAMA_SERVER_PATH; only PATH.
    cfg.llama_server_path = str(tmp_path / "llama-server")  # would mislead a shared path
    monkeypatch.setattr(_WHICH, lambda name: f"/usr/local/bin/{name}")
    assert resolve_llama_swap() == Path("/usr/local/bin/llama-swap")
    assert resolve_gguf_parser() == Path("/usr/local/bin/gguf-parser")


def test_aux_tool_raises_when_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_WHICH, lambda _name: None)
    with pytest.raises(ProviderError, match="gguf-parser binary not found"):
        resolve_gguf_parser()


def test_runtime_env_delegates_to_cuda_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    # The self-contained wheel and a BYO binary carry their own libs; the only
    # per-spawn env is the CUDA-runtime wheel path that driver-only images need.
    monkeypatch.setattr(
        "lilbee.providers.fleet.cuda_runtime.cuda_runtime_env",
        lambda: {"LD_LIBRARY_PATH": "/wheel/lib"},
    )
    assert llama_server_runtime_env() == {"LD_LIBRARY_PATH": "/wheel/lib"}
