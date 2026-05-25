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


def test_runtime_env_empty_for_byo_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    # No bundled wheel -> a BYO binary carries its own libs; inject nothing.
    monkeypatch.setattr(binary_mod, "_bundled_binary", lambda: None)
    assert llama_server_runtime_env() == {}


def test_runtime_env_empty_when_lib_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(binary_mod, "_bundled_binary", lambda: tmp_path / "llama-server")
    monkeypatch.setattr(binary_mod, "_llama_cpp_lib_dir", lambda: None)
    assert llama_server_runtime_env() == {}


def test_runtime_env_sets_lib_path_for_bundled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = tmp_path / "lib"
    monkeypatch.setattr(binary_mod, "_bundled_binary", lambda: tmp_path / "llama-server")
    monkeypatch.setattr(binary_mod, "_llama_cpp_lib_dir", lambda: lib)
    monkeypatch.setattr(binary_mod, "_lib_path_var", lambda: "LD_LIBRARY_PATH")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    assert llama_server_runtime_env() == {"LD_LIBRARY_PATH": str(lib)}


def test_runtime_env_appends_to_existing_lib_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    lib = tmp_path / "lib"
    monkeypatch.setattr(binary_mod, "_bundled_binary", lambda: tmp_path / "llama-server")
    monkeypatch.setattr(binary_mod, "_llama_cpp_lib_dir", lambda: lib)
    monkeypatch.setattr(binary_mod, "_lib_path_var", lambda: "LD_LIBRARY_PATH")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    assert llama_server_runtime_env() == {"LD_LIBRARY_PATH": f"/existing{os.pathsep}{lib}"}


def test_llama_cpp_lib_dir_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    # llama_cpp is a core dep; its lib dir exists in the venv.
    result = binary_mod._llama_cpp_lib_dir()
    assert result is not None and result.is_dir() and result.name == "lib"


def test_lib_path_var_per_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(binary_mod.sys, "platform", "darwin")
    assert binary_mod._lib_path_var() == "DYLD_LIBRARY_PATH"
    monkeypatch.setattr(binary_mod.sys, "platform", "win32")
    assert binary_mod._lib_path_var() == "PATH"
    monkeypatch.setattr(binary_mod.sys, "platform", "linux")
    assert binary_mod._lib_path_var() == "LD_LIBRARY_PATH"
