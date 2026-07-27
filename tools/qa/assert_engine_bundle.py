#!/usr/bin/env python3
"""Assert a lilbee-engine wheel carries the backend it claims, and the runtime it links.

Both checks read the payload, because neither failure shows in a build log or on a
driverless runner. A flavor missing its ggml backend library is a build whose flag
never took effect: cmake caches an unknown ``-D`` instead of failing, which is how
the published rocm wheel shipped as a CPU build. A CUDA wheel missing cudart/cublas
dies at process start on Windows and drops to CPU on Linux; non-CUDA wheels are
checked the other way, that no copy step quietly bloated them.

``--backend`` at build time, where the wheel has no ``-1.<backend>-`` tag yet.

Usage:
    assert_engine_bundle.py [--backend <flavor>] <wheel> [<wheel> ...]
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

BIN_DIR = "lilbee_engine/bin/"

# GitHub caps a single artifact at 2 GiB, and every wheel travels as one. A bundled
# ROCm engine is the only flavor near it: its runtime and per-gfx kernels are most of
# the payload, so a target added here can push the wheel past what CI can upload.
WHEEL_SIZE_LIMIT = 2 * 1024**3

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
# What a ROCm wheel carries beside its backend. Matched on the library, not its
# SONAME version, which moves every release. The kernels are the easy one to miss:
# rocBLAS loads them as data, so without them the engine dies on the first matmul.
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


def _oversized(wheel: Path) -> str | None:
    """The problem with *wheel* being too big to upload, if it is."""
    size = wheel.stat().st_size
    if size <= WHEEL_SIZE_LIMIT:
        return None
    return (
        f"{wheel.name}: {size / 1024**3:.2f} GiB exceeds the "
        f"{WHEEL_SIZE_LIMIT / 1024**3:.0f} GiB artifact limit"
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
    oversized = _oversized(wheel)
    problems = [oversized] if oversized else []
    entries = _bin_entries(wheel)
    if not entries:
        return [*problems, f"{name}: no {BIN_DIR} payload at all"]

    platform = _platform_of(name)
    backend = backend or _backend_of(name)
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
