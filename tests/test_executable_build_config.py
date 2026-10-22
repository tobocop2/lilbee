"""Config of the executable build: Playwright bytecode plugin and one Nuitka requirement."""

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_BUILD_DIR = _ROOT / "tools" / "wheel-build"
_PLUGIN = _BUILD_DIR / "playwright_generated_bytecode.py"
_NUITKA_INSTALL = re.compile(r"'(nuitka\[onefile\][^']*)'")
_INSTALL_FILES = ("release.yml", "build-gpu-executables.yml", "flatpak-validate.yml")


class _FakePluginBase:
    """Stands in for Nuitka's plugin base class, which the test env does not install."""


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch):
    base = types.ModuleType("nuitka.plugins.PluginBase")
    base.NuitkaPluginBase = _FakePluginBase  # type: ignore[attr-defined]
    for name, module in {
        "nuitka": types.ModuleType("nuitka"),
        "nuitka.plugins": types.ModuleType("nuitka.plugins"),
        "nuitka.plugins.PluginBase": base,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("playwright_generated_bytecode", _PLUGIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NuitkaPluginPlaywrightGeneratedBytecode()


@pytest.mark.parametrize(
    "module_name",
    ["playwright.async_api._generated", "playwright.sync_api._generated"],
)
def test_generated_api_modules_are_included_as_bytecode(plugin, module_name: str) -> None:
    assert plugin.decideCompilation(module_name) == "bytecode"


@pytest.mark.parametrize(
    "module_name",
    ["playwright.async_api", "playwright._impl._page", "lilbee.cli"],
)
def test_other_modules_keep_the_default_decision(plugin, module_name: str) -> None:
    assert plugin.decideCompilation(module_name) is None


def test_the_plugin_is_always_enabled(plugin) -> None:
    assert plugin.isAlwaysEnabled() is True


def test_the_build_loads_the_plugin() -> None:
    script = (_BUILD_DIR / "build_lilbee_binary.sh").read_text(encoding="utf-8")

    assert "--user-plugin=tools/wheel-build/playwright_generated_bytecode.py" in script


def test_every_nuitka_install_names_the_same_requirement() -> None:
    requirements = {
        match
        for name in _INSTALL_FILES
        for match in _NUITKA_INSTALL.findall((_WORKFLOWS / name).read_text(encoding="utf-8"))
    }
    script = (_BUILD_DIR / "build_lilbee_binary.sh").read_text(encoding="utf-8")
    requirements.update(_NUITKA_INSTALL.findall(script))

    assert len(requirements) == 1, f"the install sites disagree: {sorted(requirements)}"
