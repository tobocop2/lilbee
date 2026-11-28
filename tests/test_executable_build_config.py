"""Executable build config: Playwright plugin, exact Nuitka pin, no publishing from a check."""

import importlib.util
import re
import subprocess
import sys
import types
from pathlib import Path, PurePosixPath
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


_PACKAGE = "src/lilbee"
_CODE_SUFFIXES = (".py", ".pyc")
_WHOLE_PACKAGE_DATA = "--include-package-data=lilbee"
_DATA_DIR = "--include-data-dir="
# Nuitka's package data rules skip these, so a runtime data file must not use them.
_TYPE_MARKER = "py.typed"
_NUITKA_SKIPPED_SUFFIXES = (
    ".pyi",
    ".pyw",
    ".pyo",
    ".pyx",
    ".so",
    ".pyd",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
)


def _wheel_package_data() -> set[str]:
    """Every tracked non-code file of the package, as the wheel ships it, except the type marker."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", _PACKAGE],
        cwd=_ROOT,
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
    ).stdout
    paths = [PurePosixPath(name) for name in listed.split("\0") if name]
    return {
        path.relative_to("src").as_posix()
        for path in paths
        if path.suffix not in _CODE_SUFFIXES and path.name != _TYPE_MARKER
    }


def _nuitka_flags(script: str) -> list[str]:
    """The option lines of the script's Nuitka call: each line that starts with ``--``."""
    lines = (line.strip().removesuffix("\\").strip() for line in script.splitlines())
    return [line for line in lines if line.startswith("--")]


def _executable_package_data(script: str, files: set[str]) -> set[str]:
    """The files of *files* that a data rule of the build script puts at their package path."""
    flags = _nuitka_flags(script)
    if _WHOLE_PACKAGE_DATA in flags:
        return set(files)
    rules = (flag.removeprefix(_DATA_DIR) for flag in flags if flag.startswith(_DATA_DIR))
    folders = [
        shipped
        for source, _, shipped in (r.partition("=") for r in rules)
        if source == f"src/{shipped}"
    ]
    return {name for name in files if any(name.startswith(f"{folder}/") for folder in folders)}


def test_the_executable_ships_every_data_file_the_wheel_ships() -> None:
    script = (_BUILD_DIR / "build_lilbee_binary.sh").read_text(encoding="utf-8")
    shipped = _wheel_package_data()

    assert "lilbee/providers/fleet/engine_packs.json" in shipped
    missing = shipped - _executable_package_data(script, shipped)
    assert missing == set(), f"the executable leaves out: {sorted(missing)}"
    skipped = sorted(name for name in shipped if name.endswith(_NUITKA_SKIPPED_SUFFIXES))
    assert skipped == [], f"Nuitka does not ship these as package data: {skipped}"


_PROFILE = "lilbee/profiles/builtin/default.toml"
_RECORD = "lilbee/providers/fleet/engine_packs.json"


@pytest.mark.parametrize(
    ("script", "covered"),
    [
        pytest.param(
            "    --include-package-data=lilbee \\\n", {_PROFILE, _RECORD}, id="package-rule"
        ),
        pytest.param(
            "    --include-data-dir=src/lilbee/profiles=lilbee/profiles \\\n",
            {_PROFILE},
            id="folder-rule-covers-its-folder-only",
        ),
        pytest.param("#    --include-package-data=lilbee \\\n", set(), id="commented-out-rule"),
        pytest.param(
            "# --include-package-data=lilbee ships every data file\n",
            set(),
            id="rule-named-in-a-comment",
        ),
        pytest.param(
            "    --include-package-data=lilbee.cli \\\n", set(), id="rule-narrowed-to-a-subpackage"
        ),
    ],
)
def test_only_an_option_line_of_the_nuitka_call_counts_as_a_data_rule(
    script: str, covered: set[str]
) -> None:
    assert _executable_package_data(script, {_PROFILE, _RECORD}) == covered
