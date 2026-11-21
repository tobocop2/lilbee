"""The Windows bundler ships the Visual C++ runtime and fails on an absent file."""

from __future__ import annotations

import os
import pathlib
import stat
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the helper is driven here with a POSIX fake of cygpath",
)

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1] / "tools/wheel-build/bundle_windows_runtime.sh"
)
_CRT = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")


def _executable(path: pathlib.Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


# Folder names inside VC/Redist/MSVC/<version>/x64. VC145 is the toolset number expected for
# Visual Studio 2026 (toolset 14.5x); it is not confirmed from a runner or a Microsoft page.
_TOOLSETS = ("VC143", "VC145")


def _redist_tree(root: pathlib.Path, toolset: str, marker: bytes) -> None:
    """One x64 redist folder holding the CRT and OpenMP DLLs, every file set to `marker`."""
    crt = root / f"Microsoft.{toolset}.CRT"
    omp = root / f"Microsoft.{toolset}.OpenMP"
    crt.mkdir(parents=True)
    omp.mkdir(parents=True)
    for name in _CRT:
        (crt / name).write_bytes(marker)
    (omp / "vcomp140.dll").write_bytes(marker)


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    _redist_tree(tmp_path / "redist/x64", "VC143", b"x")
    (tmp_path / "bin").mkdir()
    (tmp_path / "fake").mkdir()
    _executable(tmp_path / "fake/cygpath", 'printf "%s\\n" "$2"\n')
    return tmp_path


def _launch(tree: pathlib.Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env["PATH"] = f"{tree / 'fake'}:{env['PATH']}"
    return subprocess.run(
        ["bash", str(_SCRIPT), str(tree / "bin")],
        env=env,
        capture_output=True,
        encoding="utf-8",
        check=False,
        stdin=subprocess.DEVNULL,
    )


def _run(tree: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return _launch(tree, {**os.environ, "VCToolsRedistDir": str(tree / "redist")})


def _run_without_vc_env(
    tree: pathlib.Path, vswhere: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "VCToolsRedistDir"}
    env["VSWHERE"] = str(vswhere)
    return _launch(tree, env)


def _vs_install(
    tree: pathlib.Path, versions: tuple[str, ...], toolset: str = "VC143"
) -> pathlib.Path:
    """A Visual Studio install with one redist tree per version, and a fake vswhere."""
    install = tree / "vs"
    redist = install / "VC/Redist/MSVC"
    for version in versions:
        _redist_tree(redist / version / "x64", toolset, version.encode())
    _executable(
        tree / "fake/vswhere.exe",
        f'printf "%s\\n" "$*" > "{tree}/vswhere.args"\nprintf "%s\\n" "{install}"\n',
    )
    return install


def test_ships_the_crt_and_the_openmp_runtime(tree) -> None:
    result = _run(tree)
    assert result.returncode == 0, result.stderr
    shipped = {p.name for p in (tree / "bin").iterdir()}
    assert {*_CRT, "vcomp140.dll"} <= shipped


@pytest.mark.parametrize("toolset", _TOOLSETS)
def test_ships_the_runtime_whatever_the_toolset_number(tree, toolset: str) -> None:
    _vs_install(tree, ("14.44.35112",), toolset)
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode == 0, result.stderr
    assert (tree / "bin/vcomp140.dll").read_bytes() == b"14.44.35112"


def test_asks_vswhere_for_the_latest_install_with_the_cpp_tools(tree) -> None:
    _vs_install(tree, ("14.44.35112",))
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode == 0, result.stderr
    asked = (tree / "vswhere.args").read_text(encoding="utf-8")
    assert "-latest" in asked
    assert "-products *" in asked
    assert "-requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64" in asked
    assert "-property installationPath" in asked


def test_picks_the_highest_version_not_the_last_in_text_order(tree) -> None:
    # Text order puts 14.9 after 14.10; version order puts 14.10 last.
    _vs_install(tree, ("14.9.30000", "14.10.40000"), "VC145")
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode == 0, result.stderr
    assert (tree / "bin/msvcp140.dll").read_bytes() == b"14.10.40000"


def test_skips_the_onecore_and_non_redist_folders_and_the_v_folder(tree) -> None:
    install = _vs_install(tree, ("14.44.35112",))
    redist = install / "VC/Redist/MSVC"
    decoys = (
        "14.44.35112/onecore/x64",
        "14.44.35112/debug_nonredist/x64",
        "14.45.1/onecore/x64",
        "14.45.1/debug_nonredist/x64",
        "v143/x64",
    )
    for decoy in decoys:
        _redist_tree(redist / decoy, "VC143", b"decoy")
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode == 0, result.stderr
    assert (tree / "bin/msvcp140.dll").read_bytes() == b"14.44.35112"


def test_vswhere_naming_no_install_refuses(tree) -> None:
    _executable(tree / "fake/vswhere.exe", "exit 0\n")
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode != 0
    assert "no Visual Studio with the C++ tools found" in result.stderr


def test_an_absent_vswhere_refuses(tree) -> None:
    result = _run_without_vc_env(tree, tree / "fake/missing-vswhere.exe")
    assert result.returncode != 0
    assert "no Visual Studio with the C++ tools found" in result.stderr


def test_an_install_with_no_redist_folder_says_so(tree) -> None:
    install = _vs_install(tree, ())
    (install / "VC/Redist/MSVC").mkdir(parents=True)
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode != 0
    assert "no MSVC redistributable folder under" in result.stderr


@pytest.mark.parametrize("missing", ["CRT", "OpenMP"])
def test_an_install_with_no_runtime_folder_names_it(tree, missing: str) -> None:
    install = _vs_install(tree, ("14.44.35112",), "VC145")
    folder = install / f"VC/Redist/MSVC/14.44.35112/x64/Microsoft.VC145.{missing}"
    for entry in folder.iterdir():
        entry.unlink()
    folder.rmdir()
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode != 0
    assert f"no Microsoft.VC*.{missing} folder under" in result.stderr


def test_a_missing_redist_file_fails(tree) -> None:
    (tree / "redist/x64/Microsoft.VC143.OpenMP/vcomp140.dll").unlink()
    result = _run(tree)
    assert result.returncode != 0
    assert "vcomp140.dll" in result.stderr


@pytest.mark.parametrize(
    "workflow",
    [
        ".github/actions/bundle-llama-server/action.yml",
        ".github/workflows/build-multigpu.yml",
        ".github/workflows/warm-engine-cache.yml",
    ],
)
def test_every_engine_cache_key_hashes_the_helper(workflow: str) -> None:
    text = (_SCRIPT.parents[2] / workflow).read_text(encoding="utf-8")
    assert "tools/wheel-build/bundle_windows_runtime.sh" in text
