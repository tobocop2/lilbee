"""Structural wheel smoke: verify llama_cpp/lib/libllama.* presence + size.

Use --expect-arch ARCH to also assert the bundled binary targets that arch
(needed for cross-compiled wheels where the build host can't import the
foreign-arch library).

Usage: python tools/wheel-build/smoke_wheel_structural.py [--expect-arch ARCH]
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile
import zipfile

_MIN_LIBLLAMA_BYTES = 50_000


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

        for lib in llama_libs:
            size = zf.getinfo(lib).file_size
            if size < _MIN_LIBLLAMA_BYTES:
                raise AssertionError(f"{lib} is suspiciously small ({size} bytes)")

        if expect_arch:
            with tempfile.TemporaryDirectory() as td:
                zf.extract(llama_libs[0], td)
                extracted = pathlib.Path(td) / llama_libs[0]
                out = subprocess.check_output(["file", str(extracted)], text=True)
                print(out.strip())
                if expect_arch not in out:
                    raise AssertionError(
                        f"{llama_libs[0]} arch mismatch: expected {expect_arch}, got: {out}"
                    )

    summary = f"structural smoke OK: {len(llama_libs)} libllama variant(s)"
    if expect_arch:
        summary += f" (arch={expect_arch})"
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
