"""Executable build config: Playwright plugin, exact Nuitka pin, no publishing from a check."""

import importlib.util
import os
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
_INSTALL_FILES = (
    "release.yml",
    "build-gpu-executables.yml",
    "build-oldmac-executables.yml",
    "flatpak-validate.yml",
)
# The tests that run a workflow step's script drive it through bash with a
# colon-separated PATH, as the runner does.
_POSIX_STEP = pytest.mark.skipif(
    sys.platform == "win32", reason="the step runs only on Linux and macOS runners"
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


_OLDMAC = _WORKFLOWS / "build-oldmac-executables.yml"
_OLDMAC_CELLS = _ROOT / ".github" / "oldmac-cells.yml"
_OLDMAC_PINS = _BUILD_DIR / "oldmac-wheels.in"


def _oldmac_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = yaml.safe_load(_OLDMAC_CELLS.read_text(encoding="utf-8"))
    return cells


def _oldmac_pins() -> list[str]:
    lines = _OLDMAC_PINS.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line and not line.startswith("#")]


def _oldmac_pin_projects() -> list[str]:
    """Each project the pins name, once."""
    return list(dict.fromkeys(pin.split(" @ ")[0] for pin in _oldmac_pins()))


# A pin: the fork's wheel for one arch, a release asset named by URL and sha256.
_ASSET_PIN = re.compile(
    r"(?P<name>[a-z-]+) @ https://github\.com/tobocop2/(?P=name)/releases/download/"
    r"v(?P<version>\d+\.\d+\.\d+)%2Bcompat/[a-z_]+-(?P=version)%2Bcompat-cp311-abi3-"
    r"macosx_11_0_(?P<arch>arm64|x86_64)\.whl#sha256=[0-9a-f]{64} ; platform_machine == '(?P=arch)'"
)


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
    assert all(c["soft"] is True for c in cells)
    assert floor_step["env"]["MACOSX_DEPLOYMENT_TARGET"] == "${{ matrix.macos_floor }}"
    assert floor_step["run"] == "bash tools/wheel-build/check_macos_floor.sh"


def test_each_old_mac_cell_claims_the_floor_its_wheels_reach() -> None:
    """The scipy arm64 wheels bundle libraries built for macOS 12.3; the Intel ones load on 12.0."""
    by_asset = {c["asset_name"]: c["macos_floor"] for c in _oldmac_cells()}

    assert by_asset == {"lilbee-compat-macos-x86_64": "12.0", "lilbee-compat-macos-arm64": "12.3"}


def _macos_floors() -> dict[str, dict[str, str]]:
    """The macOS floor of each Mac build, by npm arch, read from the two cells files."""
    release = yaml.safe_load((_ROOT / ".github" / "release-cells.yml").read_text(encoding="utf-8"))
    floors = {c["asset_name"]: c.get("macos_floor") for c in release + _oldmac_cells()}
    return {
        "arm64": {
            "main": floors["lilbee-macos-arm64"],
            "compat": floors["lilbee-compat-macos-arm64"],
        },
        "x64": {
            "main": floors["lilbee-macos-x86_64"],
            "compat": floors["lilbee-compat-macos-x86_64"],
        },
    }


_MACOS_NAMES = {12: "monterey", 13: "ventura", 14: "sonoma", 15: "sequoia", 26: "tahoe"}


def test_the_npm_launcher_agrees_with_the_cells_on_the_macos_floors() -> None:
    detect = (_ROOT / "packages" / "lilbee" / "lib" / "detect.mjs").read_text(encoding="utf-8")
    table = re.search(r"const MACOS_FLOORS = (\{.*?\}\s*\});", detect, re.DOTALL)

    assert table, "detect.mjs has no MACOS_FLOORS table"
    launcher = {
        arch: dict(re.findall(r'(main|compat): "([\d.]+)"', body))
        for arch, body in re.findall(r"(arm64|x64): \{([^}]*)\}", table.group(1))
    }
    assert launcher == _macos_floors()


def test_the_homebrew_formula_picks_the_compat_build_below_the_main_floor() -> None:
    formula = (_ROOT / "packaging" / "homebrew" / "lilbee.rb").read_text(encoding="utf-8")

    for arch, asset in (
        ("arm64", "lilbee-compat-macos-arm64"),
        ("x64", "lilbee-compat-macos-x86_64"),
    ):
        main = _macos_floors()[arch]["main"]
        last_before_main = _MACOS_NAMES[int(main.split(".")[0]) - 1]
        block = re.escape(f"on_{last_before_main} :or_older do")
        assert re.search(rf'{block}\n\s*url "[^"]*/{asset}"', formula), arch


def _install_calls(tmp_path: Path) -> list[list[str]]:
    """Run the old-Mac install step against a fake uv and return each uv argv."""
    fake = tmp_path / "bin"
    fake.mkdir()
    log = tmp_path / "uv.log"
    (fake / "uv").write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\0" "$@" >> {log}\nprintf "\\n" >> {log}\n',
        encoding="utf-8",
    )
    (fake / "uv").chmod(0o755)
    work = tmp_path / "work"
    (work / "tools" / "wheel-build").mkdir(parents=True)
    (work / "tools" / "wheel-build" / "oldmac-wheels.in").write_text(
        _OLDMAC_PINS.read_text(encoding="utf-8"), encoding="utf-8"
    )
    env = {
        "PATH": f"{fake}:{os.environ['PATH']}",
        "PYTHON_PLATFORM": "aarch64-apple-darwin",
        "LANCEDB_SPEC": "",
    }
    run = _oldmac_steps()["Install dependencies"]["run"]
    subprocess.run(["bash", "-c", run], cwd=work, env=env, check=True, capture_output=True)
    lines = log.read_text(encoding="utf-8").split("\n")
    return [line.rstrip("\0").split("\0") for line in lines if line]


@_POSIX_STEP
def test_the_old_mac_install_takes_every_fork_pin_as_a_wheel_only(tmp_path: Path) -> None:
    install = next(c for c in _install_calls(tmp_path) if c[:2] == ["pip", "install"])
    binary_only = {install[i + 1] for i, arg in enumerate(install) if arg == "--only-binary"}

    assert _oldmac_pin_projects()
    assert binary_only == set(_oldmac_pin_projects())


def test_only_the_intel_cell_swaps_lancedb_and_sets_a_cpu_baseline() -> None:
    by_asset = {c["asset_name"]: c for c in _oldmac_cells()}

    assert by_asset["lilbee-compat-macos-x86_64"]["lancedb_spec"] == "lancedb==0.40.0+compat"
    assert by_asset["lilbee-compat-macos-x86_64"]["march"] == "x86-64-v2"
    assert "lancedb_spec" not in by_asset["lilbee-compat-macos-arm64"]
    assert "march" not in by_asset["lilbee-compat-macos-arm64"]


def test_the_fork_wheels_are_exact_compat_pins_installed_binary_only() -> None:
    pins = _oldmac_pins()
    run = _oldmac_steps()["Install dependencies"]["run"]

    assert pins
    for pin in pins:
        assert _ASSET_PIN.fullmatch(pin), pin
    for project in _oldmac_pin_projects():
        matches = [m for pin in pins if (m := _ASSET_PIN.fullmatch(pin))]
        arches = [m["arch"] for m in matches if m["name"] == project]
        assert sorted(arches) == ["arm64", "x86_64"], project
    assert "https://lilbee.sh/compat/" not in run
    assert '--python-platform "${PYTHON_PLATFORM}"' in run
    assert "--no-emit-package" in run


def _homebrew_render_args(tmp_path: Path, compat_sha: str) -> str:
    """Run the Homebrew render-args step's script and return the args it writes."""
    steps = _load(_WORKFLOWS / "publish-packages.yml")["jobs"]["homebrew"]["steps"]
    step = next(s for s in steps if s.get("name") == "Build render-args")
    out = tmp_path / ("output-with" if compat_sha else "output-without")
    env = {
        "PATH": os.environ["PATH"],
        "VERSION": "1.2.3",
        "SHA_MACOS_ARM64": "a" * 64,
        "SHA_MACOS_X86_64": "b" * 64,
        "SHA_LINUX": "c" * 64,
        "SHA_MACOS_ARM64_COMPAT": compat_sha,
        "SHA_MACOS_X86_64_COMPAT": compat_sha and "e" * 64,
        "GITHUB_OUTPUT": str(out),
    }
    subprocess.run(["bash", "-c", step["run"]], env=env, check=True, capture_output=True)
    return out.read_text(encoding="utf-8")


@_POSIX_STEP
def test_the_homebrew_args_carry_the_old_mac_digest_only_when_the_asset_exists(
    tmp_path: Path,
) -> None:
    with_compat = _homebrew_render_args(tmp_path, "d" * 64)
    without = _homebrew_render_args(tmp_path, "")

    assert f"--sha-macos-arm64-compat {'d' * 64}" in with_compat
    assert f"--sha-macos-x86_64-compat {'e' * 64}" in with_compat
    assert "--sha-macos-arm64-compat" not in without
    assert "--sha-macos-x86_64-compat" not in without
    assert f"--sha-linux-x86_64 {'c' * 64}" in without


def test_the_release_resolves_the_old_mac_digest_without_requiring_it() -> None:
    steps = _load(_WORKFLOWS / "publish-packages.yml")["jobs"]["resolve"]["steps"]
    run = next(s for s in steps if s.get("id") == "info")["run"]

    assert "sha_macos_arm64_compat=$(get_sha lilbee-compat-macos-arm64)" in run
    assert "sha_macos_x86_64_compat=$(get_sha lilbee-compat-macos-x86_64)" in run
    assert "set -euo pipefail" in run


def test_nothing_in_the_release_waits_on_the_old_mac_build() -> None:
    jobs = _load(_WORKFLOWS / "release-candidate.yml")["jobs"]

    assert jobs["build-oldmac"]["uses"] == "./.github/workflows/build-oldmac-executables.yml"
    for name, job in jobs.items():
        needs = job.get("needs", [])
        needs = [needs] if isinstance(needs, str) else needs
        assert "build-oldmac" not in needs, name


def _docs_old_mac_section() -> str:
    docs = (_ROOT / "docs" / "install.md").read_text(encoding="utf-8")
    start = docs.index("## On older macOS")
    return docs[start : docs.index("\n## ", start + 1)]


def test_the_old_mac_pip_commands_take_the_wheels_the_build_pins() -> None:
    docs = _docs_old_mac_section()
    pins = [m for pin in _oldmac_pins() if (m := _ASSET_PIN.fullmatch(pin))]
    urls = re.findall(r"'([a-z-]+ @ [^']+)'", docs)
    from_index = set(re.findall(r"'([a-z-]+)==[\d.]+\+compat'", docs))

    # Both commands take every fork wheel the build pins, for their arch, by URL and sha256.
    assert sorted(urls) == sorted(m.group(0).split(" ; ")[0] for m in pins)
    # Only lancedb comes from the compat index.
    assert from_index == {"lancedb"}


def test_the_compat_index_serves_lancedb_only() -> None:
    """A fork on the compat index outranks PyPI for every pip user of the index, so it
    serves lancedb only. The install guide says the same about the index."""
    jobs = _load(_WORKFLOWS / "pages.yml")["jobs"]
    runs = [s.get("run", "") for job in jobs.values() for s in job.get("steps", [])]
    pages = next(run for run in runs if "build_wheel_indexes.py compat" in run)

    assert re.findall(r"--(\S*repo) (\S+)", pages) == [("repo", "tobocop2/lancedb")]
    docs = (_ROOT / "docs/install.md").read_text(encoding="utf-8")
    carries = re.findall(r"compat package index carries ([^,.]+)", docs)
    assert carries == ["only the patched lancedb"]


def _locked_version(project: str) -> str:
    lock = (_ROOT / "uv.lock").read_text(encoding="utf-8")
    match = re.search(rf'\[\[package\]\]\nname = "{project}"\nversion = "([^"]+)"', lock)
    assert match, project
    return match.group(1)


def test_xberg_comes_from_its_compat_release_at_the_locked_version_on_each_arch() -> None:
    """The PyPI xberg wheels need macOS 15; the fork's need 11.0, and pip must not downgrade."""
    pins = [m for pin in _oldmac_pins() if (m := _ASSET_PIN.fullmatch(pin))]
    xberg = [m for m in pins if m["name"] == "xberg"]

    assert sorted(m["arch"] for m in xberg) == ["arm64", "x86_64"]
    assert {m["version"] for m in xberg} == {_locked_version("xberg")}


def test_the_old_mac_cells_are_not_in_the_release_matrix() -> None:
    release = yaml.safe_load((_ROOT / ".github" / "release-cells.yml").read_text(encoding="utf-8"))
    names = {cell["asset_name"] for cell in release}

    assert names
    assert not names & {c["asset_name"] for c in _oldmac_cells()}
