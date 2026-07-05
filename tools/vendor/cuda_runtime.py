#!/usr/bin/env python3
"""Bundle the CUDA runtime libraries into a CUDA llama-cpp-python wheel.

The CUDA-built ggml library eagerly links cudart and cublas (which needs
cublasLt). Those ship with the CUDA Toolkit, not the NVIDIA driver, so a
wheel without them only loads on machines that have the toolkit installed.
Copying them into llama_cpp/lib makes the wheel self-contained: on Windows
the llama_cpp loader puts that directory on the DLL search path, on Linux
the ggml libraries carry an $ORIGIN RUNPATH, and Nuitka carries every
library in the directory into the frozen executables.

Usage:
    python -m tools.vendor.cuda_runtime <llama-wheel> --cuda-lib <dir>

On Windows pass "%CUDA_PATH%/bin"; on Linux pass "$CUDA_HOME/lib64".
Symlinked libraries are dereferenced, so the wheel holds real files under
the SONAME names the ggml library links against.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from tools.vendor.llama_cpp import (
    BYTES_PER_MB,
    PEP_427_TAG_FIELD_COUNT,
    _find_dist_info,
    _regenerate_record,
    _write_wheel_zip,
)

#: Libraries ggml-cuda links against that only a CUDA Toolkit install
#: provides. The driver-supplied library (nvcuda.dll / libcuda.so.1) is
#: deliberately absent. Keyed by wheel platform.
WINDOWS_LIB_PATTERNS = ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")
LINUX_LIB_PATTERNS = ("libcudart.so.*", "libcublas.so.*", "libcublasLt.so.*")


def runtime_lib_patterns(wheel_name: str) -> tuple[str, ...]:
    """Return the runtime-library glob patterns for *wheel_name*'s platform."""
    plat = Path(wheel_name).stem.rsplit("-", PEP_427_TAG_FIELD_COUNT)[-1]
    if plat.startswith("win"):
        return WINDOWS_LIB_PATTERNS
    if "linux" in plat:
        return LINUX_LIB_PATTERNS
    raise RuntimeError(
        f"CUDA runtime bundling supports Windows and Linux wheels only: {wheel_name}"
    )


def _select_runtime_libs(cuda_lib_dir: Path, patterns: tuple[str, ...]) -> list[Path]:
    """Pick one library per pattern, failing when any pattern has no match.

    On Linux a pattern matches both the SONAME symlink (libcudart.so.12) and
    the real file (libcudart.so.12.5.82); the shortest name is the SONAME the
    ggml library links against, so that one wins and is copied dereferenced.
    """
    if not cuda_lib_dir.is_dir():
        raise RuntimeError(f"CUDA library directory not found: {cuda_lib_dir}")

    chosen: list[Path] = []
    missing: list[str] = []
    for pattern in patterns:
        matches = sorted(cuda_lib_dir.glob(pattern), key=lambda p: (len(p.name), p.name))
        if matches:
            chosen.append(matches[0])
        else:
            missing.append(pattern)
    if missing:
        raise RuntimeError(
            f"CUDA runtime libraries not found in {cuda_lib_dir}: {', '.join(missing)}"
        )
    return chosen


def bundle_cuda_runtime(wheel: Path, cuda_lib_dir: Path) -> None:
    """Copy the CUDA runtime libraries into *wheel*'s llama_cpp/lib in place."""
    libs = _select_runtime_libs(cuda_lib_dir, runtime_lib_patterns(wheel.name))

    work_dir = wheel.parent / "_cuda_runtime_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    try:
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(work_dir)

        lib_dir = work_dir / "llama_cpp" / "lib"
        if not lib_dir.is_dir():
            raise RuntimeError(f"Wheel has no llama_cpp/lib directory: {wheel.name}")

        for lib in libs:
            shutil.copy2(lib, lib_dir / lib.name)
            print(f"Bundled {lib.name} ({lib.stat().st_size / BYTES_PER_MB:.1f} MB)")

        _regenerate_record(work_dir, _find_dist_info(work_dir))
        wheel.unlink()
        _write_wheel_zip(work_dir, wheel)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"Wrote {wheel.name} ({wheel.stat().st_size / BYTES_PER_MB:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle the CUDA runtime into a wheel")
    parser.add_argument("wheel", type=Path, help="Path to the llama-cpp-python .whl to modify")
    parser.add_argument(
        "--cuda-lib",
        type=Path,
        required=True,
        help="CUDA Toolkit directory holding the runtime libraries",
    )
    args = parser.parse_args()

    if not args.wheel.exists():
        parser.error(f"Wheel not found: {args.wheel}")

    bundle_cuda_runtime(args.wheel, args.cuda_lib)


if __name__ == "__main__":  # pragma: no cover
    main()
