"""Put the CUDA 12 runtime wheels on the engine's library search path.

Driver-only GPU images (common on RunPod) ship ``libcuda.so`` from the kernel
driver but not the CUDA 12 runtime that llama-server links (``libcudart.so.12``,
``libcublas.so.12``, ``libnvrtc.so.12``); without them llama-server exits before
binding its port. Installing lilbee with the ``cuda12`` extra pulls the
``nvidia-cuda-runtime-cu12`` / ``nvidia-cublas-cu12`` / ``nvidia-cuda-nvrtc-cu12``
wheels, which carry those libraries under ``site-packages/nvidia``.
:func:`cuda_runtime_env` adds their ``lib`` directories to the spawned server's
``LD_LIBRARY_PATH`` so GPU ingest works without a system CUDA toolkit -- the path
can't be a baked rpath because the wheels' location is only known at install time.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Subpackages the nvidia-*-cu12 wheels install under the ``nvidia`` namespace
# (the ``cuda12`` extra: nvidia-cuda-runtime-cu12, nvidia-cublas-cu12,
# nvidia-cuda-nvrtc-cu12).
_CUDA_WHEEL_IMPORTS: tuple[str, ...] = (
    "nvidia.cuda_runtime",
    "nvidia.cublas",
    "nvidia.cuda_nvrtc",
)


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
    return [lib for name in _CUDA_WHEEL_IMPORTS if (lib := _wheel_lib_dir(name)) is not None]


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
