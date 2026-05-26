"""Tests for multi-GPU llama-server binary resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.multi_gpu import binary as binary_mod
from lilbee.providers.multi_gpu.binary import llama_server_runtime_env, resolve_llama_server_binary

_WHICH = "lilbee.providers.multi_gpu.binary.shutil.which"


def test_uses_bundled_binary_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binpath = tmp_path / "llama-server"
    binpath.write_text("#!/bin/sh\n")
    monkeypatch.setattr("lilbee.providers.multi_gpu.binary._bundled_binary", lambda: binpath)
    assert resolve_llama_server_binary() == binpath


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


def test_runtime_env_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # The self-contained wheel (rpath-baked) and a BYO binary both carry their own
    # libs, so the fleet injects no library search path.
    monkeypatch.setattr(binary_mod, "_bundled_binary", lambda: None)
    assert llama_server_runtime_env() == {}


def test_bundled_binary_none_when_package_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    # Simulate the engine wheel not being installed (BYO / dev without it):
    # the import fails and resolution falls through to the configured path / PATH.
    monkeypatch.setitem(sys.modules, "lilbee_llama_server", None)
    assert binary_mod._bundled_binary() is None


def test_bundled_binary_none_when_binary_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    # Installed wheel whose bin/ has no binary yet (CI fills it at build time):
    # get_binary_path points at a non-existent file, so resolution returns None.
    fake = SimpleNamespace(get_binary_path=lambda: tmp_path / "llama-server")
    monkeypatch.setitem(sys.modules, "lilbee_llama_server", fake)
    assert binary_mod._bundled_binary() is None
