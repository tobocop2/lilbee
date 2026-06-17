"""Make a CUDA-linked llama-server start on driver-only GPU images.

Driver-only GPU images (common on RunPod) ship ``libcuda.so`` from the kernel
driver but not the CUDA 12 runtime the engine links: ``libcudart.so.12``,
``libcublas.so.12``, ``libnvrtc.so.12``. Without them llama-server exits before
binding its port and llama-swap reports only an opaque "exited prematurely".

The pip wheels ``nvidia-cuda-runtime-cu12`` / ``nvidia-cublas-cu12`` /
``nvidia-cuda-nvrtc-cu12`` carry those libraries under ``site-packages/nvidia``.
When they are installed, :func:`cuda_runtime_env` puts their ``lib`` directories
on the spawned server's ``LD_LIBRARY_PATH`` so GPU ingest works out of the box;
when the libraries are still unresolved, :func:`preflight_cuda_runtime` turns the
opaque failure into an install hint that names the exact packages.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from lilbee.providers.base import ProviderError

log = logging.getLogger(__name__)

# (import name, pip package, the soname the wheel provides). The import names are
# the subpackages the nvidia-*-cu12 wheels install under the ``nvidia`` namespace.
_CUDA_WHEEL_PACKAGES: tuple[tuple[str, str, str], ...] = (
    ("nvidia.cuda_runtime", "nvidia-cuda-runtime-cu12", "libcudart.so.12"),
    ("nvidia.cublas", "nvidia-cublas-cu12", "libcublas.so.12"),
    ("nvidia.cuda_nvrtc", "nvidia-cuda-nvrtc-cu12", "libnvrtc.so.12"),
)
_SONAME_TO_PACKAGE = {soname: pkg for _imp, pkg, soname in _CUDA_WHEEL_PACKAGES}

_LDD_TIMEOUT_S = 10


def _wheel_lib_dir(import_name: str) -> Path | None:
    """The ``lib/`` directory of an installed nvidia CUDA wheel subpackage, or None."""
    try:
        spec = importlib.util.find_spec(import_name)
    except ModuleNotFoundError:
        # The ``nvidia`` namespace parent is not installed at all.
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    lib = Path(next(iter(spec.submodule_search_locations))) / "lib"
    return lib if lib.is_dir() else None


def _cuda_wheel_lib_dirs() -> list[Path]:
    """Lib directories of every installed CUDA-runtime wheel, in link order."""
    dirs = []
    for import_name, _pkg, _soname in _CUDA_WHEEL_PACKAGES:
        lib = _wheel_lib_dir(import_name)
        if lib is not None:
            dirs.append(lib)
    return dirs


def cuda_runtime_env() -> dict[str, str]:
    """``LD_LIBRARY_PATH`` carrying the CUDA-runtime wheel libs, or empty.

    Empty off Linux (where the wheels and ``LD_LIBRARY_PATH`` do not apply) and
    when no wheel is installed. Wheel directories are prepended so they win over
    any stale system copy, with the caller's existing path preserved behind them.
    """
    if not sys.platform.startswith("linux"):
        return {}
    dirs = _cuda_wheel_lib_dirs()
    if not dirs:
        return {}
    parts = [str(d) for d in dirs]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    return {"LD_LIBRARY_PATH": os.pathsep.join(parts)}


def _missing_cuda_libs(binary: Path, env: dict[str, str]) -> list[str]:
    """CUDA runtime sonames *binary* links but ``ldd`` cannot resolve under *env*."""
    ldd = shutil.which("ldd")
    if ldd is None:
        return []
    try:
        proc = subprocess.run(  # noqa: S603 - ldd path and the resolved binary
            [ldd, str(binary)],
            capture_output=True,
            text=True,
            timeout=_LDD_TIMEOUT_S,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # Not an ELF, a static binary, or a timeout: nothing to preflight.
        return []
    return [soname for soname in _SONAME_TO_PACKAGE if f"{soname} => not found" in proc.stdout]


def preflight_cuda_runtime(binary: Path) -> None:
    """Raise an actionable error if *binary* links CUDA libs that won't load.

    Runs after the wheel directories are on ``LD_LIBRARY_PATH`` so it only fires
    when the runtime is genuinely absent. A CPU/Vulkan-only build does not link
    the CUDA runtime, so ``ldd`` reports nothing missing and this is a no-op.
    """
    if not sys.platform.startswith("linux"):
        return
    env = {**os.environ, **cuda_runtime_env()}
    missing = _missing_cuda_libs(binary, env)
    if not missing:
        return
    packages = " ".join(_SONAME_TO_PACKAGE[soname] for soname in missing)
    raise ProviderError(
        f"llama-server needs the CUDA 12 runtime but {', '.join(missing)} could not be "
        "found on this host (common on driver-only GPU images). Install the runtime with: "
        f"pip install {packages} -- lilbee adds their libraries to the engine's search path "
        "automatically. Or set LD_LIBRARY_PATH to a directory that contains them."
    )
