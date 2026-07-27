#!/usr/bin/env python3
"""Assert a lilbee-engine wheel carries the backend it claims, and the runtime it links.

Two checks, both on the wheel's payload, because neither failure is visible in a
build log or on a driverless CI runner.

THE BACKEND IT CLAIMS. Every flavor is built to produce one ggml backend library:
rocm builds ``libggml-hip.so``, vulkan ``libggml-vulkan.so``, and so on. A wheel
without its own is a build whose backend flag never took effect, and cmake makes
that silent: an unknown ``-D`` is cached unused rather than failing, so the build
succeeds and emits a CPU-only engine. That is not hypothetical. The published rocm
wheel carried nothing but ``libggml-cpu.so`` for as long as the flag was spelled
``GGML_HIPBLAS``, which upstream had renamed to ``GGML_HIP``, and no gate looked.

THE RUNTIME IT LINKS. A CUDA build of llama-server links cudart, cublas and
cublasLt dynamically. Only libcuda / nvcuda comes from the NVIDIA driver, so the
rest have to ship inside the wheel beside the binary. When they are missing,
Windows dies at process start on a user's machine and Linux silently falls back to
CPU. Non-CUDA wheels are checked the other way, that they carry no CUDA runtime at
all, so a copy step can't quietly bloat every backend.

Pass ``--backend`` at build time, where the flavor is known but the wheel has no
build tag yet. Without it the flavor is read from the ``-1.<backend>-`` tag, which
is what a released wheel carries.

Usage:
    assert_engine_bundle.py [--backend <flavor>] <wheel> [<wheel> ...]
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
# What a ROCm wheel must carry beside its backend, since libggml-hip links all of
# it and only the kernel driver comes from the host. Matched on the library rather
# than its SONAME version: those move every ROCm release, and pinning them would
# fail the gate on an upgrade that is in fact correct.
#
# The kernels are the entry that is easy to miss. rocBLAS loads them as data at
# runtime instead of linking them, so omitting them yields a wheel that resolves
# every library and then dies on the first matrix multiply.
ROCM_RUNTIME = (
    ("libamdhip64.so.<major>", r"libamdhip64\.so(\.[\d.]+)?"),
    ("librocblas.so.<major>", r"librocblas\.so(\.[\d.]+)?"),
    ("libhipblas.so.<major>", r"libhipblas\.so(\.[\d.]+)?"),
    ("rocblas/library/ (rocBLAS Tensile kernels)", r"rocblas/library/.+"),
)
# Any CUDA runtime library, used to assert non-CUDA wheels stay clean.
ANY_CUDA_RUNTIME = re.compile(r"(cudart|cublas|cublasLt|nvrtc)", re.IGNORECASE)
# The ggml CUDA backend. Its presence is what makes a wheel a CUDA build. Read the
# payload rather than the filename: the ``-1.cu125-`` build tag is added when the
# wheel is attached to the release, so a freshly built artifact does not carry it.
GGML_CUDA = re.compile(r"(lib)?ggml-cuda\.(so|dll)")

# The ggml backend library each flavor is built to produce, keyed by the build tag
# the index writes. Every cu* flavor maps to the same one, so they are resolved by
# prefix rather than listed. A flavor absent here asserts nothing: a new backend
# should not fail the gate before anyone has taught it the library's name.
BACKEND_LIBRARY = {
    "cpu": "ggml-cpu",
    "metal": "ggml-metal",
    "rocm": "ggml-hip",
    "sycl": "ggml-sycl",
    "vulkan": "ggml-vulkan",
}
# How each platform spells a shared library, and how a version lands in the name:
# ``libggml-hip.so.0.15.1`` on Linux, ``libggml-metal.0.15.1.dylib`` on macOS.
# Matching the versioned form matters because the engine links by SONAME, so that
# file is the library itself and the bare name beside it is only a symlink.
_LIBRARY_FORMS = {
    "win_amd64": ("{name}.dll", r"{name}\.dll"),
    "linux": ("lib{name}.so", r"lib{name}\.so(\.[\d.]+)?"),
    "macos": ("lib{name}.dylib", r"lib{name}(\.[\d.]+)?\.dylib"),
}
# "lilbee_engine-0.6.90-1.rocm-py3-none-manylinux_2_17_x86_64.whl". PEP 427 puts
# the build tag third, and build_pep503_indexes.py writes it as 1.<backend>.
_BUILD_TAG = re.compile(r"^[^-]+-[^-]+-\d+\.(?P<backend>[a-z0-9]+)-")


def _platform_of(wheel_name: str) -> str:
    if "win_amd64" in wheel_name:
        return "win_amd64"
    if "manylinux" in wheel_name or "linux_x86_64" in wheel_name:
        return "linux"
    if "macosx" in wheel_name:
        return "macos"
    return "other"


def _backend_of(wheel_name: str) -> str | None:
    """The flavor *wheel_name*'s build tag declares, or None if it carries no tag."""
    match = _BUILD_TAG.match(wheel_name)
    return match.group("backend") if match else None


def _missing_backend_library(
    name: str, backend: str, platform: str, entries: list[str]
) -> str | None:
    """The problem with *entries* not holding *backend*'s ggml library, if it doesn't."""
    library = "ggml-cuda" if backend.startswith("cu") else BACKEND_LIBRARY.get(backend)
    form = _LIBRARY_FORMS.get(platform)
    if library is None or form is None:
        return None
    display, pattern = (part.format(name=library) for part in form)
    if any(re.fullmatch(pattern, entry) for entry in entries):
        return None
    return (
        f"{name}: {backend} wheel is missing {display} from {BIN_DIR} "
        f"(the build produced no {backend} backend, so this engine runs on CPU)"
    )


def _bin_entries(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    return [n[len(BIN_DIR) :] for n in names if n.startswith(BIN_DIR) and not n.endswith("/")]


def check(wheel: Path, backend: str | None = None) -> list[str]:
    """Return a list of problems with *wheel*; empty means it is correct.

    *backend* names the flavor the wheel was built as. Pass it at build time, when
    the wheel has no build tag to read it from yet.
    """
    name = wheel.name
    entries = _bin_entries(wheel)
    if not entries:
        return [f"{name}: no {BIN_DIR} payload at all"]

    platform = _platform_of(name)
    backend = backend or _backend_of(name)
    problems = []
    missing = _missing_backend_library(name, backend, platform, entries) if backend else None
    if missing:
        problems.append(missing)

    if backend == "rocm" and platform == "linux" and not missing:
        # Only worth asking once the backend is actually there: a wheel with no
        # libggml-hip.so has a bigger problem than a missing runtime beside it.
        problems.extend(
            f"{name}: rocm wheel is missing {display} from {BIN_DIR} "
            f"(libggml-hip links it, and only the kernel driver comes from the host)"
            for display, pattern in ROCM_RUNTIME
            if not any(re.fullmatch(pattern, entry) for entry in entries)
        )

    # A declared flavor is the better answer, since the whole point of the check
    # above is that the payload can disagree with what the build asked for.
    is_cuda = backend.startswith("cu") if backend else any(GGML_CUDA.fullmatch(e) for e in entries)

    if not is_cuda:
        stowaways = sorted(e for e in entries if ANY_CUDA_RUNTIME.search(e))
        if stowaways:
            problems.append(
                f"{name}: non-CUDA wheel ships CUDA runtime libraries: {', '.join(stowaways)}"
            )
        return problems

    if platform not in REQUIRED:
        return problems  # A CUDA engine on a platform we ship no CUDA runtime for.

    problems.extend(
        f"{name}: CUDA wheel is missing {display} from {BIN_DIR} "
        f"(ggml-cuda links it; only libcuda/nvcuda comes from the driver)"
        for display, pattern in REQUIRED[platform]
        if not any(re.fullmatch(pattern, e) for e in entries)
    )
    return problems


def main(argv: list[str]) -> int:
    backend = None
    if len(argv) >= 2 and argv[0] == "--backend":
        backend, argv = argv[1], argv[2:]

    wheels = [Path(a) for a in argv]
    if not wheels:
        print(
            "usage: assert_engine_bundle.py [--backend <flavor>] <wheel> [<wheel> ...]",
            file=sys.stderr,
        )
        return 2

    problems: list[str] = []
    for wheel in wheels:
        if not wheel.is_file():
            problems.append(f"{wheel}: not a file")
            continue
        problems.extend(check(wheel, backend))
        print(f"checked {wheel.name}")

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"all {len(wheels)} wheel(s) carry the backend they claim")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
