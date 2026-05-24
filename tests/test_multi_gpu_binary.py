"""Tests for multi-GPU llama-server binary resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.multi_gpu.binary import resolve_llama_server_binary

_WHICH = "lilbee.providers.multi_gpu.binary.shutil.which"


def test_uses_configured_path_when_file_exists(tmp_path: Path) -> None:
    binpath = tmp_path / "llama-server"
    binpath.write_text("#!/bin/sh\n")
    cfg.llama_server_path = str(binpath)
    assert resolve_llama_server_binary() == binpath


def test_raises_when_configured_path_missing(tmp_path: Path) -> None:
    cfg.llama_server_path = str(tmp_path / "absent")
    with pytest.raises(ProviderError, match="is not a file"):
        resolve_llama_server_binary()


def test_falls_back_to_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg.llama_server_path = ""
    monkeypatch.setattr(_WHICH, lambda _name: "/usr/local/bin/llama-server")
    assert resolve_llama_server_binary() == Path("/usr/local/bin/llama-server")


def test_raises_with_install_hint_when_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg.llama_server_path = ""
    monkeypatch.setattr(_WHICH, lambda _name: None)
    with pytest.raises(ProviderError, match="not found"):
        resolve_llama_server_binary()
