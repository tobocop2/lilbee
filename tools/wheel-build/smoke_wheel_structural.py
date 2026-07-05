"""Structural wheel smoke: verify llama_cpp/lib/libllama.* presence + size.

Use --expect-arch ARCH to also assert the bundled binary targets that arch
(needed for cross-compiled wheels where the build host can't import the
foreign-arch library).

Usage: python tools/wheel-build/smoke_wheel_structural.py [--expect-arch ARCH]
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import pathlib
import subprocess
import sys
import tempfile
import zipfile

# Script-mode sys.path[0] is this directory; tools.vendor lives at the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tools.vendor.cuda_runtime import runtime_lib_patterns

_MIN_LIBLLAMA_BYTES = 50_000


def _assert_cuda_runtime_bundled(wheel_name: str, libs: list[str]) -> None:
    """CUDA wheels must carry the CUDA runtime libraries in llama_cpp/lib."""
    names = [pathlib.Path(n).name for n in libs]
    missing = [
        pattern
        for pattern in runtime_lib_patterns(wheel_name)
        if not fnmatch.filter(names, pattern)
    ]
    if missing:
        raise AssertionError(f"CUDA wheel is missing runtime libraries: {missing}")


def _assert_lib_arch(zf: zipfile.ZipFile, lib: str, expect_arch: str) -> None:
    """Extract *lib* and assert `file` reports the expected architecture."""
    with tempfile.TemporaryDirectory() as td:
        zf.extract(lib, td)
        extracted = pathlib.Path(td) / lib
        out = subprocess.check_output(["file", str(extracted)], text=True)  # noqa: S603, S607
        print(out.strip())
        if expect_arch not in out:
            raise AssertionError(f"{lib} arch mismatch: expected {expect_arch}, got: {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-arch", default=None)
    args = parser.parse_args()
    expect_arch = args.expect_arch

    whls = sorted(pathlib.Path("dist").glob("lilbee-*.whl"))
    if len(whls) != 1:
        raise AssertionError(f"expected exactly one lilbee wheel: {whls}")
    whl = whls[0]
    print(f"inspecting {whl.name} ({whl.stat().st_size:,} bytes)")

    if expect_arch and expect_arch not in whl.name:
        raise AssertionError(f"wheel platform tag missing {expect_arch!r}: {whl.name}")

    with zipfile.ZipFile(whl) as zf:
        libs = [n for n in zf.namelist() if n.startswith("llama_cpp/lib/")]
        print(f"llama_cpp/lib entries: {len(libs)}")
        for lib in libs:
            print(f"  {lib} ({zf.getinfo(lib).file_size:,} bytes)")

        llama_libs = [n for n in libs if "llama" in pathlib.Path(n).name.lower()]
        if not llama_libs:
            raise AssertionError(f"wheel ships no libllama.* in llama_cpp/lib/: {libs}")

        if os.environ.get("BACKEND", "").startswith("cu"):
            _assert_cuda_runtime_bundled(whl.name, libs)

        for lib in llama_libs:
            size = zf.getinfo(lib).file_size
            if size < _MIN_LIBLLAMA_BYTES:
                raise AssertionError(f"{lib} is suspiciously small ({size} bytes)")

        if expect_arch:
            _assert_lib_arch(zf, llama_libs[0], expect_arch)

    summary = f"structural smoke OK: {len(llama_libs)} libllama variant(s)"
    if expect_arch:
        summary += f" (arch={expect_arch})"
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
