"""The engine install advice fits a self-contained build or a package install."""

from __future__ import annotations

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import binary as binary_mod
from lilbee.providers.fleet import engine_install as mod
from lilbee.providers.fleet.engine_install import engine_install_hint


def test_a_package_install_is_told_to_run_the_engine_install_command() -> None:
    hint = engine_install_hint(bundled=False)
    assert "lilbee engine install" in hint
    assert "[engine]" not in hint
    assert "extra-index-url" not in hint


def test_a_bundled_build_is_told_to_reinstall_not_to_use_pip() -> None:
    hint = engine_install_hint(bundled=True)
    assert "brew reinstall lilbee" in hint
    assert "npm install -g lilbee" in hint
    assert "pip" not in hint
    assert "lilbee engine install" not in hint


@pytest.mark.parametrize("frozen", [True, False])
def test_the_hint_follows_whether_this_build_is_self_contained(monkeypatch, frozen) -> None:
    monkeypatch.setattr(mod, "is_frozen", lambda: frozen)
    assert engine_install_hint() == engine_install_hint(bundled=frozen)


def test_the_missing_engine_error_carries_the_advice_for_this_install(monkeypatch) -> None:
    monkeypatch.setattr(binary_mod, "_bundled_tool", lambda _tool: None)
    monkeypatch.setattr(binary_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(mod, "is_frozen", lambda: False)
    with pytest.raises(ProviderError, match="llama-server binary not found") as excinfo:
        binary_mod.resolve_llama_server()
    assert "lilbee engine install" in str(excinfo.value)
    assert "[engine]" not in str(excinfo.value)
