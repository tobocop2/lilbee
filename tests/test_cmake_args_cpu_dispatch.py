"""Pin cmake_args.sh flag emission for every build cell."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the script under test is POSIX shell"
)

# Every x86_64 cell, split by CPU strategy. The union plus _ARM_CELLS is the
# script's whole case table; a new cell must join one of these lists.
_DISPATCH_CELLS = [("cpu", "macOS"), ("cpu", "Linux"), ("cpu", "Windows"), ("vulkan", "Linux")]
_CAPPED_CELLS = [
    ("vulkan", "Windows"),
    *[(f"cu12{n}", os) for n in range(1, 6) for os in ("Linux", "Windows")],
    ("rocm", "Linux"),
    ("sycl", "Linux"),
    ("sycl", "Windows"),
]
_ARM_CELLS = [("cpu", "macOS", "arm64"), ("metal", "macOS", "arm64")]

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools/wheel-build/cmake_args.sh"


def _flags(backend: str, runner_os: str, target_arch: str) -> list[str]:
    """The CMAKE_ARGS the script emits for one build cell."""
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--print"],
        capture_output=True,
        encoding="utf-8",
        env={
            "BACKEND": backend,
            "RUNNER_OS": runner_os,
            "TARGET_ARCH": target_arch,
            "PATH": "/usr/bin:/bin",
        },
        check=True,
    )
    return result.stdout.split()


@pytest.mark.parametrize(("backend", "runner_os"), _DISPATCH_CELLS)
def test_converted_x86_cells_build_runtime_dispatch(backend: str, runner_os: str) -> None:
    flags = _flags(backend, runner_os, "x86_64")
    assert "-DGGML_BACKEND_DL=ON" in flags
    assert "-DGGML_CPU_ALL_VARIANTS=ON" in flags
    # An AVX cap on a dispatch build would cap every variant.
    assert "-DGGML_AVX2=OFF" not in flags


def test_vulkan_linux_dispatch_keeps_the_vulkan_backend() -> None:
    flags = _flags("vulkan", "Linux", "x86_64")
    assert "-DGGML_VULKAN=ON" in flags


def test_macos_arm64_cpu_stays_single_variant() -> None:
    flags = _flags("cpu", "macOS", "arm64")
    assert "-DGGML_BACKEND_DL=ON" not in flags
    assert "-DGGML_CPU_ALL_VARIANTS=ON" not in flags


@pytest.mark.parametrize(("backend", "runner_os"), _CAPPED_CELLS)
def test_deferred_gpu_cells_keep_the_avx_cap(backend: str, runner_os: str) -> None:
    flags = _flags(backend, runner_os, "x86_64")
    assert "-DGGML_AVX=ON" in flags
    assert "-DGGML_AVX2=OFF" in flags
    assert "-DGGML_CPU_ALL_VARIANTS=ON" not in flags


@pytest.mark.parametrize(
    ("backend", "runner_os", "target_arch"),
    [(b, o, "x86_64") for b, o in _DISPATCH_CELLS + _CAPPED_CELLS] + _ARM_CELLS,
)
def test_no_cell_downloads_ui_assets(backend: str, runner_os: str, target_arch: str) -> None:
    flags = _flags(backend, runner_os, target_arch)
    assert "-DLLAMA_BUILD_UI=OFF" in flags
    assert "-DLLAMA_USE_PREBUILT_UI=OFF" in flags
