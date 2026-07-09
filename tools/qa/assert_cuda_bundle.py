#!/usr/bin/env python3
"""Assert a lilbee-engine wheel carries the CUDA runtime its engine links.

A CUDA build of llama-server links cudart, cublas and cublasLt dynamically. Only
libcuda / nvcuda comes from the NVIDIA driver, so the rest have to ship inside the
wheel beside the binary. When they are missing the failure is invisible on a
driverless CI runner: Windows dies at process start on a user's machine, and Linux
silently falls back to CPU. Neither shows up in a build log, so check the wheel.

Non-CUDA wheels are checked the other way, that they carry no CUDA runtime at all,
so a copy step can't quietly bloat every backend.

Usage:
    assert_cuda_bundle.py <wheel> [<wheel> ...]
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

BIN_DIR = "lilbee_engine/bin/"

# The SONAME / import names the engine loads by, per wheel platform tag. Named so a
# failure names the file a user would look for, not the pattern that matched it.
REQUIRED = {
    "win_amd64": (
        ("cudart64_<major>.dll", r"cudart64_\d+\.dll"),
        ("cublas64_<major>.dll", r"cublas64_\d+\.dll"),
        ("cublasLt64_<major>.dll", r"cublasLt64_\d+\.dll"),
    ),
    "linux": (
        ("libcudart.so.<major>", r"libcudart\.so\.\d+"),
        ("libcublas.so.<major>", r"libcublas\.so\.\d+"),
        ("libcublasLt.so.<major>", r"libcublasLt\.so\.\d+"),
    ),
}
# Any CUDA runtime library, used to assert non-CUDA wheels stay clean.
ANY_CUDA_RUNTIME = re.compile(r"(cudart|cublas|cublasLt|nvrtc)", re.IGNORECASE)
# The ggml CUDA backend. Its presence is what makes a wheel a CUDA build. Read the
# payload rather than the filename: the ``-1.cu125-`` build tag is added when the
# wheel is attached to the release, so a freshly built artifact does not carry it.
GGML_CUDA = re.compile(r"(lib)?ggml-cuda\.(so|dll)")


def _platform_of(wheel_name: str) -> str:
    if "win_amd64" in wheel_name:
        return "win_amd64"
    if "manylinux" in wheel_name or "linux_x86_64" in wheel_name:
        return "linux"
    return "other"


def _bin_entries(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    return [n[len(BIN_DIR) :] for n in names if n.startswith(BIN_DIR) and not n.endswith("/")]


def check(wheel: Path) -> list[str]:
    """Return a list of problems with *wheel*; empty means it is correct."""
    name = wheel.name
    entries = _bin_entries(wheel)
    if not entries:
        return [f"{name}: no {BIN_DIR} payload at all"]

    is_cuda = any(GGML_CUDA.fullmatch(e) for e in entries)
    platform = _platform_of(name)

    if not is_cuda:
        stowaways = sorted(e for e in entries if ANY_CUDA_RUNTIME.search(e))
        if stowaways:
            return [f"{name}: non-CUDA wheel ships CUDA runtime libraries: {', '.join(stowaways)}"]
        return []

    if platform == "other":
        return []  # A CUDA engine on a platform we ship no CUDA runtime for.

    return [
        f"{name}: CUDA wheel is missing {display} from {BIN_DIR} "
        f"(ggml-cuda links it; only libcuda/nvcuda comes from the driver)"
        for display, pattern in REQUIRED[platform]
        if not any(re.fullmatch(pattern, e) for e in entries)
    ]


def main(argv: list[str]) -> int:
    wheels = [Path(a) for a in argv]
    if not wheels:
        print("usage: assert_cuda_bundle.py <wheel> [<wheel> ...]", file=sys.stderr)
        return 2

    problems: list[str] = []
    for wheel in wheels:
        if not wheel.is_file():
            problems.append(f"{wheel}: not a file")
            continue
        problems.extend(check(wheel))
        print(f"checked {wheel.name}")

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"all {len(wheels)} wheel(s) carry the CUDA runtime they link")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
