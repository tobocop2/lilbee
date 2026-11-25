#!/usr/bin/env python3
"""Assert a lilbee-engine-llama-cpp bundle carries the backend it claims, and the runtime it links.

A bundle is a wheel, or the ``lilbee_engine_llama_cpp/bin`` directory of an executable's
standalone tree. Both go through the same checks; only the wheel has a size limit.

Both checks read the payload, because neither failure shows in a build log or on a
driverless runner. A flavor missing its ggml backend library is a build whose flag
never took effect: cmake caches an unknown ``-D`` instead of failing, which is how
the published rocm wheel shipped as a CPU build. A CUDA wheel missing cudart/cublas
dies at process start on Windows and drops to CPU on Linux; non-CUDA wheels are
checked the other way, that no copy step quietly bloated them.

``--backend`` at build time, where the wheel has no ``-1.<backend>-`` tag yet.
A directory has no file name to read the platform from: ``--platform`` takes the
runner OS (``Linux``, ``macOS`` or ``Windows``).

``--tool`` names the engine server of the pin. With it, the wheel must also hold that
server with the execute bit. The rule is the one the engine store applies when it
installs a wheel.

``--library`` names a library the server needs, and may be given more than once. Each
one must be a shared library in the package of the server, by the store's own rule.
It needs ``--tool``.

Usage:
    assert_engine_bundle.py [--backend <flavor>] [--tool <server> [--library <name> ...]]
        <wheel> [<wheel> ...]
    assert_engine_bundle.py [--backend <flavor>] --platform <runner OS> <bin dir>
    assert_engine_bundle.py --imports-only <folder> [<folder> ...]
    assert_engine_bundle.py --prune-runtime <folder> [<folder> ...]

``--imports-only`` runs the import check of a Windows bundle alone, on any folder of
Windows files. The mistral.rs wheel check uses it on the folder its wheel installs.

``--prune-runtime`` deletes each Visual C++ runtime DLL of a folder that no file of the
folder imports, read from the same import tables. The Windows packs run it before the build.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import zipfile
from pathlib import Path

_RULE = Path(__file__).resolve().parents[2] / "src/lilbee/providers/fleet/wheel_server.py"

# The full module name keeps coverage tracing the file when a test loads this script first.
_spec = importlib.util.spec_from_file_location("lilbee.providers.fleet.wheel_server", _RULE)
assert _spec and _spec.loader
wheel_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wheel_server)

BIN_DIR = "lilbee_engine_llama_cpp/bin/"

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
# The MSVC runtime every Windows build links, with the OpenMP runtime ggml imports.
# A fresh Windows image ships an older msvcp140 and no vcomp140, so the engine
# faults or fails to load before listing a device.
WINDOWS_CRT = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcomp140.dll")
# The list above is what was found so far. The import check finds the next one: each
# DLL a bundled PE file imports is in the bundle, or is a file Windows or the GPU
# driver provides. Names are lower case, as Windows compares them.
WINDOWS_SYSTEM_DLLS = frozenset(
    {
        "advapi32.dll",
        "bcrypt.dll",
        "bcryptprimitives.dll",
        "cfgmgr32.dll",
        "combase.dll",
        "comctl32.dll",
        "comdlg32.dll",
        "crypt32.dll",
        "dbghelp.dll",
        "dxgi.dll",
        "gdi32.dll",
        "imm32.dll",
        "iphlpapi.dll",
        "kernel32.dll",
        "kernelbase.dll",
        "msvcrt.dll",
        "ncrypt.dll",
        "normaliz.dll",
        "ntdll.dll",
        "nvcuda.dll",
        "ole32.dll",
        "oleaut32.dll",
        "pdh.dll",
        "powrprof.dll",
        "psapi.dll",
        "rpcrt4.dll",
        "sechost.dll",
        "secur32.dll",
        "setupapi.dll",
        "shell32.dll",
        "shlwapi.dll",
        "ucrtbase.dll",
        "user32.dll",
        "userenv.dll",
        "version.dll",
        "vulkan-1.dll",
        "winmm.dll",
        "wldap32.dll",
        "ws2_32.dll",
    }
)
# Windows maps these virtual names onto the system's own files.
_WINDOWS_SYSTEM_PREFIXES = ("api-ms-win-", "ext-ms-win-")
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
    "vulkan": "ggml-vulkan",
}
# Cells built with GGML_CPU_ALL_VARIANTS ship the CPU backend as one module per
# x86 feature level, picked at runtime. A single module is the flag not taking
# effect: the fallback lib carries only the build host's instruction set.
DISPATCH_CELLS = {("cpu", "linux"), ("cpu", "win_amd64"), ("vulkan", "linux")}
_VARIANT_FORMS = {
    "linux": ("libggml-cpu-<variant>.so", r"libggml-cpu-.+\.so(\.[\d.]+)?"),
    "win_amd64": ("ggml-cpu-<variant>.dll", r"ggml-cpu-.+\.dll"),
}
# The loader the vulkan module dlopens under GGML_BACKEND_DL. Nothing links it,
# so a missing loader is a silent CPU fallback, visible only in the payload.
_VULKAN_LOADER = ("libvulkan.so.1", r"libvulkan\.so\.[\d.]+")
# How each platform spells a shared library, and how a version lands in the name:
# ``libggml-hip.so.0.15.1`` on Linux, ``libggml-metal.0.15.1.dylib`` on macOS.
# Matching the versioned form matters because the engine links by SONAME, so that
# file is the library itself and the bare name beside it is only a symlink.
_LIBRARY_FORMS = {
    "win_amd64": ("{name}.dll", r"{name}\.dll"),
    "linux": ("lib{name}.so", r"lib{name}\.so(\.[\d.]+)?"),
    "macos": ("lib{name}.dylib", r"lib{name}(\.[\d.]+)?\.dylib"),
}
# The runner OS names a directory bundle's platform, as a wheel's file name does.
_RUNNER_PLATFORMS = {"Linux": "linux", "macOS": "macos", "Windows": "win_amd64"}
# "lilbee_engine_llama_cpp-0.6.90-1.rocm-py3-none-manylinux_2_17_x86_64.whl". PEP 427 puts
# the build tag third, and build_wheel_indexes.py writes it as 1.<backend>.
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
    name: str, backend: str, platform: str, entries: list[str], kind: str
) -> str | None:
    """The problem with *entries* not holding *backend*'s ggml library, if it doesn't."""
    if backend == "cpu" and (backend, platform) in DISPATCH_CELLS:
        # A dispatch cell has no single libggml-cpu; the variant check owns it.
        return None
    library = "ggml-cuda" if backend.startswith("cu") else BACKEND_LIBRARY.get(backend)
    form = _LIBRARY_FORMS.get(platform)
    if library is None or form is None:
        return None
    display, pattern = (part.format(name=library) for part in form)
    if any(re.fullmatch(pattern, entry) for entry in entries):
        return None
    return (
        f"{name}: {backend} {kind} is missing {display} from {BIN_DIR} "
        f"(the build produced no {backend} backend, so this engine runs on CPU)"
    )


def _missing_dispatch_modules(
    name: str, backend: str, platform: str, entries: list[str], kind: str
) -> list[str]:
    """The problems with *entries* not holding a runtime-dispatch CPU backend."""
    if (backend, platform) not in DISPATCH_CELLS:
        return []
    problems = []
    display, pattern = _VARIANT_FORMS[platform]
    variants = [e for e in entries if re.fullmatch(pattern, e)]
    if len(variants) < 2:
        problems.append(
            f"{name}: {backend} {kind} has {len(variants)} {display} modules in {BIN_DIR} "
            f"(runtime CPU dispatch needs at least two; a single module carries only "
            f"the build host's instruction set)"
        )
    if backend == "vulkan" and platform == "linux":
        loader_display, loader_pattern = _VULKAN_LOADER
        if not any(re.fullmatch(loader_pattern, e) for e in entries):
            problems.append(
                f"{name}: vulkan {kind} is missing {loader_display} from {BIN_DIR} "
                f"(the vulkan module dlopens the loader; without it the engine "
                f"silently runs on CPU)"
            )
    return problems


def _oversized(wheel: Path) -> str | None:
    """The problem with *wheel* being too big to upload, if it is."""
    size = wheel.stat().st_size
    if size <= WHEEL_SIZE_LIMIT:
        return None
    return (
        f"{wheel.name}: {size / 1024**3:.2f} GiB exceeds the "
        f"{WHEEL_SIZE_LIMIT / 1024**3:.0f} GiB artifact limit"
    )


def _bin_entries(bundle: Path) -> list[str]:
    """The files under ``lilbee_engine_llama_cpp/bin`` of a wheel, or under a bin directory."""
    if bundle.is_dir():
        return sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if not p.is_dir())
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
    return [n[len(BIN_DIR) :] for n in names if n.startswith(BIN_DIR) and not n.endswith("/")]


def _pe_imports(data: bytes) -> list[str]:
    """The DLL names the PE file *data* imports, lower case; empty for a file that is not a PE.

    Both the load-time and the delay-load tables count: a delay-loaded DLL that is
    absent fails on the first call into it, which is as late and as bad as it sounds.

    Raises ValueError when a table has an address and pefile parsed no entry from it. pefile
    only warns about a truncated or corrupt table, and an empty answer would read as
    "imports nothing".
    """
    if data[:2] != b"MZ":
        return []
    # A QA requirement (tools/qa/requirements.txt), needed only for a Windows bundle.
    import pefile

    pe = pefile.PE(data=data, fast_load=True)
    try:
        tables = (
            ("IMAGE_DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_IMPORT"),
            ("IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"),
        )
        indexes = [pefile.DIRECTORY_ENTRY[directory] for directory, _ in tables]
        pe.parse_data_directories(directories=indexes)
        names: set[str] = set()
        for index, (directory, attribute) in zip(indexes, tables, strict=True):
            entries = getattr(pe, attribute, ())
            if not entries and pe.OPTIONAL_HEADER.DATA_DIRECTORY[index].VirtualAddress:
                raise ValueError(f"the {directory} table of a file is truncated or corrupt")
            names.update(t.dll.decode("ascii", "replace").lower() for t in entries)
        return sorted(names)
    finally:
        pe.close()


def _unbundled_imports(imports: dict[str, list[str]], entries: list[str]) -> dict[str, list[str]]:
    """Map each file of *imports* to the imported DLLs that neither the bundle nor Windows holds."""
    bundled = {e.rsplit("/", 1)[-1].lower() for e in entries}
    return {
        file: missing
        for file, names in imports.items()
        if (
            missing := [
                n
                for n in names
                if n not in bundled
                and n not in WINDOWS_SYSTEM_DLLS
                and not n.startswith(_WINDOWS_SYSTEM_PREFIXES)
            ]
        )
    }


def _import_problems(bundle: Path, name: str, entries: list[str], kind: str) -> list[str]:
    """The problems with a Windows *bundle* importing a DLL it does not carry."""
    imports: dict[str, list[str]] = {}
    archive = None if bundle.is_dir() else zipfile.ZipFile(bundle)
    try:
        for entry in entries:
            if not entry.lower().endswith((".dll", ".exe")):
                continue
            data = (
                (bundle / entry).read_bytes()
                if archive is None
                else archive.read(f"{BIN_DIR}{entry}")
            )
            imports[entry] = _pe_imports(data)
    finally:
        if archive is not None:
            archive.close()
    return [
        f"{name}: windows {kind} {file} imports {', '.join(missing)}, which the bundle does not "
        "carry and Windows does not provide (a machine without it cannot start the engine)"
        for file, missing in _unbundled_imports(imports, entries).items()
    ]


def _server_problems(
    bundle: Path, name: str, tool: str | None, libraries: tuple[str, ...]
) -> list[str]:
    """The store's server check on a wheel; a directory or no *tool* has none."""
    if not tool or bundle.is_dir():
        return []
    with zipfile.ZipFile(bundle) as archive:
        found = wheel_server.server_problems(archive, tool, libraries)
    return [f"{name}: {p}" for p in found]


def check(
    bundle: Path,
    backend: str | None = None,
    runner_os: str | None = None,
    tool: str | None = None,
    libraries: tuple[str, ...] = (),
) -> list[str]:
    """Return a list of problems with *bundle*; empty means it is correct.

    *bundle* is a wheel, or the ``lilbee_engine_llama_cpp/bin`` directory of an executable.
    *backend* names the flavor it was built as. Pass it at build time, when a wheel
    has no build tag to read it from yet. A directory has no file name to read
    the platform from, so it needs *runner_os*. *tool* names the engine server of
    the pin; a wheel must then hold a runnable server and each of *libraries*.
    """
    if bundle.is_dir():
        name, kind, problems = str(bundle), "executable", []
        platform = _RUNNER_PLATFORMS.get(runner_os or "", "other")
    else:
        name, kind = bundle.name, "wheel"
        oversized = _oversized(bundle)
        problems = [oversized] if oversized else []
        platform = _platform_of(name)
    entries = _bin_entries(bundle)
    if not entries:
        return [*problems, f"{name}: no {BIN_DIR} payload at all"]

    problems.extend(_server_problems(bundle, name, tool, libraries))
    backend = backend or _backend_of(name)
    missing = _missing_backend_library(name, backend, platform, entries, kind) if backend else None
    if missing:
        problems.append(missing)
    if backend:
        problems.extend(_missing_dispatch_modules(name, backend, platform, entries, kind))

    if backend == "rocm" and platform == "linux" and not missing:
        # Only worth asking once the backend is actually there: a wheel with no
        # libggml-hip.so has a bigger problem than a missing runtime beside it.
        problems.extend(
            f"{name}: rocm {kind} is missing {display} from {BIN_DIR} "
            f"(libggml-hip links it, and only the kernel driver comes from the host)"
            for display, pattern in ROCM_RUNTIME
            if not any(re.fullmatch(pattern, entry) for entry in entries)
        )

    if platform == "win_amd64":
        # A wheel ships only the runtime DLLs its files import (--prune-runtime), so the
        # import check below finds a missing one. An executable folder ships all of them.
        problems.extend(
            f"{name}: windows {kind} is missing {crt} from {BIN_DIR} "
            "(the engine links the MSVC runtime; a fresh Windows image ships an older copy)"
            for crt in WINDOWS_CRT
            if crt not in entries and kind == "executable"
        )
        problems.extend(_import_problems(bundle, name, entries, kind))

    # A declared flavor is the better answer, since the whole point of the check
    # above is that the payload can disagree with what the build asked for.
    is_cuda = backend.startswith("cu") if backend else any(GGML_CUDA.fullmatch(e) for e in entries)

    if not is_cuda:
        stowaways = sorted(e for e in entries if ANY_CUDA_RUNTIME.search(e))
        if stowaways:
            problems.append(
                f"{name}: non-CUDA {kind} ships CUDA runtime libraries: {', '.join(stowaways)}"
            )
        return problems

    if platform not in REQUIRED:
        return problems  # A CUDA engine on a platform we ship no CUDA runtime for.

    problems.extend(
        f"{name}: CUDA {kind} is missing {display} from {BIN_DIR} "
        f"(ggml-cuda links it; only libcuda/nvcuda comes from the driver)"
        for display, pattern in REQUIRED[platform]
        if not any(re.fullmatch(pattern, e) for e in entries)
    )
    return problems


def _parse(argv: list[str]) -> tuple[dict[str, str], list[str], list[str]]:
    """Split *argv* into the single-value flags, the ``--library`` names and the paths."""
    values: dict[str, str] = {}
    libraries: list[str] = []
    while len(argv) >= 2 and argv[0] in ("--backend", "--platform", "--tool", "--library"):
        if argv[0] == "--library":
            libraries.append(argv[1])
        else:
            values[argv[0]] = argv[1]
        argv = argv[2:]
    return values, libraries, argv


def _refusal(bundle: Path, runner_os: str | None, backend: str | None, tool: str | None) -> str:
    """Why *bundle* cannot be checked with these flags; empty when it can."""
    if not bundle.is_dir():
        return "" if bundle.is_file() else f"{bundle}: not a file"
    if runner_os not in _RUNNER_PLATFORMS:
        return f"{bundle}: a directory needs --platform Linux, macOS or Windows"
    if backend is None:
        return f"{bundle}: a directory needs --backend <flavor>"
    if tool:
        return f"{bundle}: --tool reads a wheel, not a directory"
    return ""


def _unused_runtime(folder: Path) -> list[Path]:
    """The bundled Visual C++ runtime DLLs of *folder* that no file of the bundle reaches.

    Every PE file (a file that starts with MZ, whatever its suffix) that is not a runtime DLL
    counts as a root: the loader may open any of them. A runtime DLL is reached when a root
    imports it, or when a reached runtime DLL does. A file whose import table cannot be read
    stops the prune with an error before it deletes anything.
    """
    paths = [folder / entry for entry in _bin_entries(folder)]
    imports = {path: _pe_imports(path.read_bytes()) for path in paths}
    runtime = {p.name.lower(): p for p in paths if p.name.lower() in WINDOWS_CRT}
    pending = [n for p, names in imports.items() if p.name.lower() not in runtime for n in names]
    reached: set[str] = set()
    while pending:
        name = pending.pop()
        if name in runtime and name not in reached:
            reached.add(name)
            pending.extend(imports.get(runtime[name], ()))
    return [path for name, path in sorted(runtime.items()) if name not in reached]


def _prune_runtime(folders: list[str]) -> int:
    """Delete each runtime DLL that nothing in a folder imports."""
    paths = [Path(f) for f in folders]
    if not paths or not all(p.is_dir() for p in paths):
        print(
            "usage: assert_engine_bundle.py --prune-runtime <folder> [<folder> ...]",
            file=sys.stderr,
        )
        return 2
    for path in paths:
        for unused in _unused_runtime(path):
            print(f"pruned {unused.name}: nothing in {path} imports it")
            unused.unlink()
    return 0


def _imports_only(folders: list[str]) -> int:
    """Check each folder of Windows files for an import that neither it nor Windows holds."""
    paths = [Path(f) for f in folders]
    if not paths or not all(p.is_dir() for p in paths):
        print(
            "usage: assert_engine_bundle.py --imports-only <folder> [<folder> ...]", file=sys.stderr
        )
        return 2
    problems = [
        problem
        for path in paths
        for problem in _import_problems(path, str(path), _bin_entries(path), "executable")
    ]
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--imports-only"]:
        return _imports_only(argv[1:])
    if argv[:1] == ["--prune-runtime"]:
        return _prune_runtime(argv[1:])
    values, libraries, argv = _parse(argv)
    backend = values.get("--backend")
    runner_os = values.get("--platform")
    tool = values.get("--tool")

    bundles = [Path(a) for a in argv]
    if not bundles or (libraries and not tool):
        print(
            "usage: assert_engine_bundle.py [--backend <flavor>] "
            "[--tool <server> [--library <name> ...]] <wheel> [<wheel> ...]\n"
            "       assert_engine_bundle.py [--backend <flavor>] --platform <runner OS> <bin dir>",
            file=sys.stderr,
        )
        return 2

    problems: list[str] = []
    for bundle in bundles:
        refusal = _refusal(bundle, runner_os, backend, tool)
        if refusal:
            problems.append(refusal)
            continue
        problems.extend(check(bundle, backend, runner_os, tool, tuple(libraries)))
        print(f"checked {bundle if bundle.is_dir() else bundle.name}")

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"all {len(bundles)} bundle(s) carry the backend they claim")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
