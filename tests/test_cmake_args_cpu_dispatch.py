"""The macOS x86_64 CPU cell builds runtime CPU dispatch; every other x86 cell
keeps the single-variant AVX cap, and no cell downloads server-UI assets.

cmake caches an unknown ``-D`` instead of failing, so a dropped or renamed flag
here builds successfully and ships the wrong engine. These tests pin the emitted
flag set; ``build_llama_server.sh`` separately asserts the variant modules exist
in the built bundle.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the script under test is POSIX shell, Linux-only in CI"
)

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools/wheel-build/cmake_args.sh"


def _flags(backend: str, runner_os: str, target_arch: str) -> list[str]:
    """The CMAKE_ARGS the script emits for one build cell."""
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--print"],
        capture_output=True,
        text=True,
        env={
            "BACKEND": backend,
            "RUNNER_OS": runner_os,
            "TARGET_ARCH": target_arch,
            "PATH": "/usr/bin:/bin",
        },
        check=True,
    )
    return result.stdout.split()


def test_macos_x86_64_cpu_builds_runtime_dispatch() -> None:
    flags = _flags("cpu", "macOS", "x86_64")
    assert "-DGGML_BACKEND_DL=ON" in flags
    assert "-DGGML_CPU_ALL_VARIANTS=ON" in flags
    # The AVX cap and variant dispatch are mutually exclusive: capping the
    # variants would rebuild the single-baseline engine under a new name.
    assert "-DGGML_AVX2=OFF" not in flags


def test_macos_arm64_cpu_stays_single_variant() -> None:
    flags = _flags("cpu", "macOS", "arm64")
    assert "-DGGML_BACKEND_DL=ON" not in flags
    assert "-DGGML_CPU_ALL_VARIANTS=ON" not in flags


@pytest.mark.parametrize(
    ("backend", "runner_os"),
    [("cpu", "Linux"), ("vulkan", "Linux"), ("vulkan", "Windows"), ("cu121", "Linux")],
)
def test_gpu_and_linux_cells_keep_the_avx_cap(backend: str, runner_os: str) -> None:
    flags = _flags(backend, runner_os, "x86_64")
    assert "-DGGML_AVX=ON" in flags
    assert "-DGGML_AVX2=OFF" in flags
    assert "-DGGML_CPU_ALL_VARIANTS=ON" not in flags


@pytest.mark.parametrize(
    ("backend", "runner_os", "target_arch"),
    [
        ("cpu", "macOS", "x86_64"),
        ("cpu", "macOS", "arm64"),
        ("metal", "macOS", "arm64"),
        ("cpu", "Linux", "x86_64"),
        ("vulkan", "Windows", "x86_64"),
    ],
)
def test_no_cell_downloads_ui_assets(backend: str, runner_os: str, target_arch: str) -> None:
    flags = _flags(backend, runner_os, target_arch)
    assert "-DLLAMA_BUILD_UI=OFF" in flags
    assert "-DLLAMA_USE_PREBUILT_UI=OFF" in flags
