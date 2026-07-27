"""The rocm cell builds for the cards this ROCm can still build for, and says so.

AMD drops GPU targets between ROCm releases, and one unsupported entry in
``AMDGPU_TARGETS`` fails the whole cmake configure rather than being skipped: ROCm 7
removed gfx940, which turned the rocm cell red as soon as the corrected flag let it
reach the compiler. The filter reads the toolchain's own device bitcode, so these
tests need no ROCm and no GPU, only a directory shaped like one.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools/wheel-build/cmake_args.sh"

# The ISAs a current ROCm ships bitcode for. gfx940 is deliberately absent: it is the
# target AMD removed, and it must never come back into the emitted list.
_CURRENT_ISAS = ("906", "908", "90a", "942", "1030", "1100", "1101", "1102")


def _rocm_tree(root: pathlib.Path, isas: tuple[str, ...]) -> pathlib.Path:
    """A ROCm install whose device bitcode covers *isas*."""
    bitcode = root / "amdgcn/bitcode"
    bitcode.mkdir(parents=True)
    (root / "lib/llvm/bin").mkdir(parents=True)
    for isa in isas:
        (bitcode / f"oclc_isa_version_{isa}.bc").write_text("")
    return root


def _targets(rocm_root: pathlib.Path) -> tuple[str, str]:
    """The AMDGPU_TARGETS the script emits for *rocm_root*, and its stderr."""
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--print"],
        capture_output=True,
        text=True,
        env={
            "BACKEND": "rocm",
            "RUNNER_OS": "Linux",
            "ROCM_PATH": str(rocm_root),
            "PATH": "/usr/bin:/bin",
        },
        check=True,
    )
    flag = next(a for a in result.stdout.split() if a.startswith("-DAMDGPU_TARGETS="))
    return flag.removeprefix("-DAMDGPU_TARGETS="), result.stderr


def test_builds_every_target_a_current_rocm_supports(tmp_path):
    """Nothing is dropped when the toolchain knows every card we ship for."""
    targets, stderr = _targets(_rocm_tree(tmp_path / "rocm", _CURRENT_ISAS))
    assert targets.split(";") == [
        "gfx906",
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
    ]
    assert "cannot build" not in stderr


def test_never_asks_for_the_target_rocm_7_removed(tmp_path):
    """gfx940 fails the configure outright; it is what broke the first green build."""
    targets, _ = _targets(_rocm_tree(tmp_path / "rocm", _CURRENT_ISAS))
    assert "gfx940" not in targets


def test_drops_targets_this_rocm_cannot_build_and_reports_them(tmp_path):
    """A ROCm bump that removes a target must not fail the build, but must be visible."""
    isas = tuple(i for i in _CURRENT_ISAS if i not in {"942", "1102"})
    targets, stderr = _targets(_rocm_tree(tmp_path / "rocm", isas))
    assert "gfx942" not in targets
    assert "gfx1102" not in targets
    assert "gfx906" in targets
    assert "gfx942" in stderr and "gfx1102" in stderr


def test_falls_back_loudly_when_there_is_no_bitcode_to_read(tmp_path):
    """An install we do not understand must not silently emit a wheel for no card."""
    root = tmp_path / "rocm"
    (root / "lib/llvm/bin").mkdir(parents=True)
    targets, stderr = _targets(root)
    assert "gfx942" in targets
    assert "no device bitcode" in stderr


@pytest.mark.parametrize("isas", [_CURRENT_ISAS, ("942", "1030")])
def test_the_emitted_args_survive_eval(tmp_path, isas):
    """build_llama_server.sh evals this; an unquoted ';' would split it into commands."""
    root = _rocm_tree(tmp_path / f"rocm{len(isas)}", isas)
    emitted = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env={
            "BACKEND": "rocm",
            "RUNNER_OS": "Linux",
            "ROCM_PATH": str(root),
            "PATH": "/usr/bin:/bin",
        },
        check=True,
    ).stdout
    roundtrip = subprocess.run(
        ["bash", "-c", f'{emitted}\nprintf "%s" "$CMAKE_ARGS"'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "-DGGML_HIP=ON" in roundtrip
    assert f"-DAMDGPU_TARGETS=gfx{isas[0]}" in roundtrip
