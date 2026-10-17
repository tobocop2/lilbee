"""The engine install advice names the command for the way lilbee was installed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import binary as binary_mod
from lilbee.providers.fleet import engine_install as mod
from lilbee.providers.fleet.engine_install import (
    InstallMethod,
    detect_install_method,
    engine_install_hint,
)


def _venv(monkeypatch: pytest.MonkeyPatch, prefix: Path, cfg_text: str | None) -> None:
    prefix.mkdir(parents=True, exist_ok=True)
    if cfg_text is not None:
        (prefix / "pyvenv.cfg").write_text(cfg_text, encoding="utf-8")
    monkeypatch.setattr(mod, "is_frozen", lambda: False)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", str(prefix.parent / "base"))


def test_a_uv_tool_environment_is_detected(monkeypatch, tmp_path) -> None:
    _venv(monkeypatch, tmp_path / "share" / "uv" / "tools" / "lilbee", "uv = 0.9.0\n")
    assert detect_install_method() is InstallMethod.UV_TOOL


def test_a_pipx_environment_is_detected(monkeypatch, tmp_path) -> None:
    _venv(monkeypatch, tmp_path / "pipx" / "venvs" / "lilbee", "home = /usr/bin\n")
    assert detect_install_method() is InstallMethod.PIPX


def test_a_venv_made_by_uv_is_detected(monkeypatch, tmp_path) -> None:
    _venv(monkeypatch, tmp_path / ".venv", "home = /usr/bin\nuv = 0.6.14\n")
    assert detect_install_method() is InstallMethod.UV_ENV


def test_a_venv_not_made_by_uv_is_pip(monkeypatch, tmp_path) -> None:
    _venv(monkeypatch, tmp_path / ".venv", "home = /usr/bin\nversion = 3.13\n")
    assert detect_install_method() is InstallMethod.PIP


def test_a_venv_with_no_config_file_is_pip(monkeypatch, tmp_path) -> None:
    _venv(monkeypatch, tmp_path / ".venv", None)
    assert detect_install_method() is InstallMethod.PIP


def test_the_system_interpreter_is_unknown(monkeypatch) -> None:
    monkeypatch.setattr(mod, "is_frozen", lambda: False)
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    assert detect_install_method() is InstallMethod.UNKNOWN


@pytest.mark.parametrize(
    ("executable", "expected"),
    [
        ("/opt/homebrew/Cellar/lilbee/1.0/bin/lilbee", InstallMethod.HOMEBREW),
        ("/home/me/.cache/lilbee/lilbee", InstallMethod.STANDALONE),
    ],
)
def test_a_frozen_build_is_homebrew_or_standalone(monkeypatch, executable, expected) -> None:
    monkeypatch.setattr(mod, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "executable", executable)
    assert detect_install_method() is expected


@pytest.mark.parametrize(
    ("method", "must_have", "must_not_have"),
    [
        (InstallMethod.PIP, "pip install --pre", "uv "),
        (InstallMethod.UV_ENV, "uv pip install --prerelease=allow", "uv tool"),
        (InstallMethod.UV_TOOL, "uv tool install --force --prerelease=allow", "uv pip"),
        (InstallMethod.PIPX, "pipx install --force --pip-args=", "uv "),
    ],
)
def test_each_install_gets_its_own_command_for_every_index(
    method, must_have, must_not_have
) -> None:
    hint = engine_install_hint(method)
    assert hint.count(must_have) == 5
    for index in ("cu125", "rocm", "metal", "vulkan", "cpu"):
        assert f"https://lilbee.sh/{index}/" in hint
    assert must_not_have not in hint


def test_an_unknown_install_lists_the_pip_and_the_uv_forms() -> None:
    hint = engine_install_hint(InstallMethod.UNKNOWN)
    assert hint.count("pip install --pre") == 5
    assert hint.count("uv tool install") == 5


@pytest.mark.parametrize("method", [InstallMethod.HOMEBREW, InstallMethod.STANDALONE])
def test_a_bundled_build_is_told_to_reinstall_not_to_use_pip(method) -> None:
    hint = engine_install_hint(method)
    assert "brew reinstall lilbee" in hint
    assert "npm install -g lilbee" in hint
    assert "pip" not in hint


def test_the_missing_engine_error_carries_the_advice_for_this_install(monkeypatch) -> None:
    monkeypatch.setattr(binary_mod, "_bundled_tool", lambda _tool: None)
    monkeypatch.setattr(binary_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(mod, "detect_install_method", lambda: InstallMethod.UV_TOOL)
    with pytest.raises(ProviderError, match="llama-server binary not found") as excinfo:
        binary_mod.resolve_llama_server()
    assert "uv tool install --force" in str(excinfo.value)
    assert "pip install --pre" not in str(excinfo.value)
