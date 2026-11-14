"""The floor check fails a pack whose files import a glibc or libstdc++ symbol above the wheel tag.

``objdump`` is a stub that prints the undefined dynamic symbols recorded for each file name,
so the tests need no ELF files.
"""

from __future__ import annotations

import pathlib
import shutil
import stat
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the script under test is POSIX shell, Linux-only in CI"
)

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools/wheel-build/check_glibc_floor.sh"
_TOOLS = ("bash", "env", "basename", "cat", "find", "grep", "sed", "sort", "tail")


def _undefined(*versions: str) -> str:
    """The objdump -T lines of a file that imports one symbol per version tag."""
    return "".join(
        f"0000000000000000      DF *UND*  0000000000000000  {tag}  sym{i}\n"
        for i, tag in enumerate(versions)
    )


def _run(
    tmp_path: pathlib.Path, files: dict[str, str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    pack = tmp_path / "pack"
    pack.mkdir()
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    cases = []
    for name, dump in files.items():
        (pack / name).write_text("elf", encoding="utf-8")
        cases.append(f"  {name}) cat <<'DUMP'\n{dump}DUMP\n  ;;")
    objdump = stubs / "objdump"
    header = '#!/usr/bin/env bash\ncase "$(basename "${!#}")" in\n'
    objdump.write_text(header + "\n".join(cases) + "\n  *) ;;\nesac\n", encoding="utf-8")
    objdump.chmod(objdump.stat().st_mode | stat.S_IEXEC)
    for tool in _TOOLS:
        found = shutil.which(tool)
        assert found, f"{tool} not on PATH"
        if not (stubs / tool).exists():
            (stubs / tool).symlink_to(found)
    bash = shutil.which("bash")
    assert bash
    return subprocess.run(
        [bash, str(_SCRIPT), str(pack)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={"PATH": str(stubs), **(env or {})},
        check=False,
    )


def test_passes_files_at_the_floor(tmp_path):
    """glibc 2.34 and libstdc++ 3.4.29 are the limits themselves, so they pass."""
    result = _run(
        tmp_path,
        {"libggml.so": _undefined("GLIBC_2.17", "GLIBC_2.34", "GLIBCXX_3.4.29", "GLIBCXX_3.4.9")},
    )
    assert result.returncode == 0, result.stderr
    assert "libggml.so: glibc 2.34, libstdc++ 3.4.29" in result.stdout
    assert "floor: 1 files checked" in result.stdout


def test_fails_on_a_glibc_symbol_above_the_floor(tmp_path):
    """The comparison is numeric: 2.4 is below 2.34 and 2.38 is above it."""
    packs = {
        "libok.so": _undefined("GLIBC_2.4"),
        "libhipblaslt.so": _undefined("GLIBC_2.34", "GLIBC_2.38"),
    }
    result = _run(tmp_path, packs)
    assert result.returncode == 1
    assert "FLOOR libhipblaslt.so needs glibc 2.38, above 2.34" in result.stderr
    assert "libok.so needs" not in result.stderr


def test_fails_on_a_libstdcxx_symbol_above_the_floor(tmp_path):
    """3.4.30 is one release above what RHEL 9 ships."""
    result = _run(tmp_path, {"libggml-hip.so": _undefined("GLIBCXX_3.4.30", "GLIBC_2.17")})
    assert result.returncode == 1
    assert "FLOOR libggml-hip.so needs libstdc++ 3.4.30, above 3.4.29" in result.stderr
    assert "glibc" not in result.stderr


def test_reports_every_file_that_fails(tmp_path):
    """One run names all offenders, so one CI run is enough to see them."""
    result = _run(
        tmp_path,
        {"liba.so": _undefined("GLIBC_2.35"), "libb.so": _undefined("GLIBCXX_3.4.31")},
    )
    assert result.returncode == 1
    assert "FLOOR liba.so" in result.stderr
    assert "FLOOR libb.so" in result.stderr


def test_limits_come_from_the_environment(tmp_path):
    """MAX_GLIBC moves the glibc limit for a pack built for another tag."""
    result = _run(tmp_path, {"liba.so": _undefined("GLIBC_2.38")}, env={"MAX_GLIBC": "2.38"})
    assert result.returncode == 0, result.stderr


def test_fails_when_nothing_has_dynamic_imports(tmp_path):
    """A directory that holds no dynamic file would otherwise pass without checking anything."""
    result = _run(tmp_path, {"data.bin": ""})
    assert result.returncode == 1
    assert "no dynamic file found" in result.stderr
