"""The Windows bundler ships the Visual C++ runtime and refuses an import it did not ship."""

from __future__ import annotations

import os
import pathlib
import stat
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the helper is driven here with POSIX fakes of cygpath and dumpbin",
)

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1] / "tools/wheel-build/bundle_windows_runtime.sh"
)
_CRT = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")


def _executable(path: pathlib.Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    crt = tmp_path / "redist/x64/Microsoft.VC143.CRT"
    omp = tmp_path / "redist/x64/Microsoft.VC143.OpenMP"
    crt.mkdir(parents=True)
    omp.mkdir(parents=True)
    for name in _CRT:
        (crt / name).write_bytes(b"x")
    (omp / "vcomp140.dll").write_bytes(b"x")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/ggml-base.dll").write_bytes(b"x")
    (tmp_path / "fake").mkdir()
    _executable(tmp_path / "fake/cygpath", 'printf "%s\\n" "$2"\n')
    return tmp_path


def _run(tree: pathlib.Path, imports: str) -> subprocess.CompletedProcess[str]:
    _executable(tree / "fake/dumpbin", f"printf '    {imports}\\n'\n")
    env = {
        **os.environ,
        "PATH": f"{tree / 'fake'}:{os.environ['PATH']}",
        "VCToolsRedistDir": str(tree / "redist"),
    }
    return subprocess.run(
        ["bash", str(_SCRIPT), str(tree / "bin")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def test_ships_the_crt_and_the_openmp_runtime(tree) -> None:
    result = _run(tree, "VCOMP140.dll")
    assert result.returncode == 0, result.stderr
    shipped = {p.name for p in (tree / "bin").iterdir()}
    assert {*_CRT, "vcomp140.dll"} <= shipped


def test_an_import_it_did_not_ship_fails_and_names_the_dll(tree) -> None:
    result = _run(tree, "msvcp140_2.dll")
    assert result.returncode != 0
    assert "msvcp140_2.dll" in result.stderr


def test_a_missing_redist_file_fails(tree) -> None:
    (tree / "redist/x64/Microsoft.VC143.OpenMP/vcomp140.dll").unlink()
    result = _run(tree, "vcomp140.dll")
    assert result.returncode != 0
    assert "vcomp140.dll" in result.stderr
