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
_INSTALL_FILES = (
    "release.yml",
    "build-gpu-executables.yml",
    "build-oldmac-executables.yml",
    "flatpak-validate.yml",
)


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
    """Five sites install Nuitka; a sixth added without the pin must fail the test above."""
    sites = sum(
        len(_NUITKA_INSTALL.findall((_WORKFLOWS / name).read_text(encoding="utf-8")))
        for name in _INSTALL_FILES
    )

    assert sites == 5


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
    unguarded = [
        step.get("name") for step in publishing if "only_asset == ''" not in _step_guard(step)
    ]
    assert unguarded == [], f"these steps can publish when only_asset is set: {unguarded}"
    reversed_ = [step.get("name") for step in publishing if "only_asset != ''" in _step_guard(step)]
    assert reversed_ == [], f"these guards run when only_asset is set: {reversed_}"


def test_the_smoke_test_reads_the_skill_from_the_frozen_binary() -> None:
    steps = _load(_WORKFLOWS / "release.yml")["jobs"]["build"]["steps"]
    smoke = [step for step in steps if step.get("name") == "Smoke test"]

    assert len(smoke) == 1
    assert "lilbee.cli.launchers.skill_install import bundled_skill_dir" in smoke[0]["run"]


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


_BUILD_TOOLING = (
    "tools/wheel-build/**",
    ".github/release-cells.yml",
    ".github/gpu-cells.yml",
    ".github/workflows/release.yml",
    ".github/workflows/build-gpu-executables.yml",
    ".github/workflows/flatpak-validate.yml",
    ".github/workflows/executable-check.yml",
)


def _windows_filter() -> list[str]:
    jobs = _load(_WORKFLOWS / "executable-check.yml")["jobs"]
    steps = jobs["changes"]["steps"]
    filters = [step["with"]["filters"] for step in steps if "paths-filter" in str(step.get("uses"))]
    assert len(filters) == 1
    windows: list[str] = yaml.safe_load(filters[0])["windows"]
    return windows


def test_the_trigger_paths_list_the_build_tooling() -> None:
    workflow = _load(_WORKFLOWS / "executable-check.yml")
    # PyYAML reads the bare key `on` as the boolean True.
    paths = workflow.get("on", workflow.get(True))["pull_request"]["paths"]

    assert set(_BUILD_TOOLING) <= set(paths), sorted(set(_BUILD_TOOLING) - set(paths))


def test_the_windows_filter_lists_the_build_tooling() -> None:
    windows = _windows_filter()

    assert set(_BUILD_TOOLING) <= set(windows), sorted(set(_BUILD_TOOLING) - set(windows))


def test_every_windows_trigger_also_triggers_the_check() -> None:
    workflow = _load(_WORKFLOWS / "executable-check.yml")
    paths = workflow.get("on", workflow.get(True))["pull_request"]["paths"]

    assert set(_windows_filter()) <= set(paths)


_OLDMAC = _WORKFLOWS / "build-oldmac-executables.yml"
_OLDMAC_CELLS = _ROOT / ".github" / "oldmac-cells.yml"
_OLDMAC_PINS = _BUILD_DIR / "oldmac-wheels.in"


def _oldmac_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = yaml.safe_load(_OLDMAC_CELLS.read_text(encoding="utf-8"))
    return cells


def _oldmac_steps() -> dict[str, dict[str, Any]]:
    steps = _load(_OLDMAC)["jobs"]["build"]["steps"]
    return {step["name"]: step for step in steps if "name" in step}


def test_both_old_mac_cells_are_droppable_and_checked_at_their_floor() -> None:
    cells = _oldmac_cells()
    floor_step = _oldmac_steps()["Verify macOS floor"]

    assert {c["asset_name"] for c in cells} == {
        "lilbee-compat-macos-x86_64",
        "lilbee-compat-macos-arm64",
    }
    assert all(c["soft"] is True and c["macos_floor"] == "12.0" for c in cells)
    assert floor_step["env"]["MACOSX_DEPLOYMENT_TARGET"] == "${{ matrix.macos_floor }}"
    assert floor_step["run"] == "bash tools/wheel-build/check_macos_floor.sh"


def test_only_the_intel_cell_swaps_lancedb_and_sets_a_cpu_baseline() -> None:
    by_asset = {c["asset_name"]: c for c in _oldmac_cells()}

    assert by_asset["lilbee-compat-macos-x86_64"]["lancedb_spec"] == "lancedb==0.40.0+compat"
    assert by_asset["lilbee-compat-macos-x86_64"]["march"] == "x86-64-v2"
    assert "lancedb_spec" not in by_asset["lilbee-compat-macos-arm64"]
    assert "march" not in by_asset["lilbee-compat-macos-arm64"]


def test_the_fork_wheels_are_exact_compat_pins_installed_binary_only() -> None:
    pins = [
        line
        for line in _OLDMAC_PINS.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    run = _oldmac_steps()["Install dependencies"]["run"]

    assert pins
    for pin in pins:
        assert re.fullmatch(r"[a-z-]+==\d+\.\d+\.\d+\+compat", pin), pin
    assert "--only-binary" in run
    assert "--extra-index-url https://lilbee.sh/compat/" in run
    assert '--python-platform "${PYTHON_PLATFORM}"' in run
    assert "--no-emit-package" in run


def test_nothing_in_the_release_waits_on_the_old_mac_build() -> None:
    jobs = _load(_WORKFLOWS / "release-candidate.yml")["jobs"]

    assert jobs["build-oldmac"]["uses"] == "./.github/workflows/build-oldmac-executables.yml"
    for name, job in jobs.items():
        needs = job.get("needs", [])
        needs = [needs] if isinstance(needs, str) else needs
        assert "build-oldmac" not in needs, name


def test_pages_serves_every_fork_the_old_mac_pins_name() -> None:
    pages = (_WORKFLOWS / "pages.yml").read_text(encoding="utf-8")
    pins = [
        line.split("==")[0]
        for line in _OLDMAC_PINS.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]

    assert pins
    assert "--repo tobocop2/lancedb" in pages
    for project in pins:
        assert f"--optional-repo tobocop2/{project}" in pages


def test_the_old_mac_cells_are_not_in_the_release_matrix() -> None:
    release = yaml.safe_load((_ROOT / ".github" / "release-cells.yml").read_text(encoding="utf-8"))
    names = {cell["asset_name"] for cell in release}

    assert names
    assert not names & {c["asset_name"] for c in _oldmac_cells()}
