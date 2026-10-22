"""Executable build config: Playwright plugin, exact Nuitka pin, no publishing from a check."""

import importlib.util
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_BUILD_DIR = _ROOT / "tools" / "wheel-build"
_PLUGIN = _BUILD_DIR / "playwright_generated_bytecode.py"
_NUITKA_INSTALL = re.compile(r"'(nuitka\[onefile\][^']*)'")
_ACTIONS = _ROOT / ".github" / "actions"
_NUITKA_PIN = "nuitka[onefile]==4.3.1"
_PUBLISH_MARKERS = ("gh release", "ci_mirror", "publish-engine-asset", "phase: publish")
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

    assert requirements == {_NUITKA_PIN}, f"the install sites disagree: {sorted(requirements)}"


def test_every_nuitka_install_site_is_counted() -> None:
    """Four sites install Nuitka; a fifth added without the pin must fail the test above."""
    sites = sum(
        len(_NUITKA_INSTALL.findall((_WORKFLOWS / name).read_text(encoding="utf-8")))
        for name in _INSTALL_FILES
    )

    assert sites == 4


def _load(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def _publishing_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        step
        for step in steps
        if any(marker in yaml.safe_dump(step) for marker in _PUBLISH_MARKERS)
        or (step.get("with") or {}).get("phase") == "publish"
        or "asset-release-tag" in (step.get("with") or {})
    ]


def _step_guard(step: dict[str, Any]) -> str:
    """The text a step uses to decide whether it runs or what it may publish to."""
    return str(step.get("if", "")) + str((step.get("with") or {}).get("asset-release-tag", ""))


def test_a_narrowed_release_run_publishes_nothing() -> None:
    steps = _load(_WORKFLOWS / "release.yml")["jobs"]["build"]["steps"]
    publishing = _publishing_steps(steps)

    assert len(publishing) == 3, [step.get("name") for step in publishing]
    unguarded = [step.get("name") for step in publishing if "only_asset" not in _step_guard(step)]
    assert unguarded == [], f"these steps can publish when only_asset is set: {unguarded}"


def test_the_engine_mirror_is_skipped_when_its_release_tag_is_empty() -> None:
    steps = _load(_ACTIONS / "bundle-llama-server" / "action.yml")["runs"]["steps"]
    mirror = [step for step in steps if "publish-engine-asset" in str(step.get("uses", ""))]

    assert len(mirror) == 1
    assert "inputs.asset-release-tag != ''" in mirror[0]["if"]


def test_the_executable_check_passes_only_the_asset_to_the_release_workflow() -> None:
    jobs = _load(_WORKFLOWS / "executable-check.yml")["jobs"]
    called = {name: job for name, job in jobs.items() if "uses" in job}

    assert called
    for name, job in called.items():
        assert list(job["with"]) == ["only_asset"], name
