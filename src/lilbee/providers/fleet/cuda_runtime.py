"""Put the CUDA 12 runtime wheels on the engine's library search path.

Driver-only GPU images (common on RunPod) ship ``libcuda.so`` from the kernel driver
but not the CUDA 12 runtime that llama-server links (``libcudart.so.12``,
``libcublas.so.12``, ``libnvrtc.so.12``). The bundled engine now carries those beside
the binary and resolves them through its baked ``$ORIGIN`` rpath, so this module is
the fallback for an engine built elsewhere: installing lilbee with the ``cuda12``
extra pulls the ``nvidia-cuda-runtime-cu12`` / ``nvidia-cublas-cu12`` /
``nvidia-cuda-nvrtc-cu12`` wheels, which carry those libraries under
``site-packages/nvidia``. :func:`cuda_runtime_env` adds their ``lib`` directories to
the spawned server's ``LD_LIBRARY_PATH`` -- the path can't be a baked rpath because
the wheels' location is only known at install time.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from lilbee.providers import model_cache
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.engine_diagnostics import device_probe_diagnostic, links_any

if TYPE_CHECKING:
    from lilbee.providers.fleet.devices import FleetDevice

# Subpackages the nvidia-*-cu12 wheels install under the ``nvidia`` namespace
# (the ``cuda12`` extra: nvidia-cuda-runtime-cu12, nvidia-cublas-cu12,
# nvidia-cuda-nvrtc-cu12).
_CUDA_WHEEL_IMPORTS: tuple[str, ...] = (
    "nvidia.cuda_runtime",
    "nvidia.cublas",
    "nvidia.cuda_nvrtc",
)
# The sonames those wheels provide, used to tell from ``ldd`` whether a binary is a
# CUDA build (it lists the soname whether or not the runtime resolves).
_CUDA_SONAMES: tuple[str, ...] = ("libcudart.so.12", "libcublas.so.12", "libnvrtc.so.12")

log = logging.getLogger(__name__)


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
    for part in existing.split(os.pathsep):
        if part and part not in parts:
            parts.append(part)
    return {"LD_LIBRARY_PATH": os.pathsep.join(parts)}


def apply_cuda_runtime_env() -> None:
    """Put the CUDA-runtime wheel libs on this process's ``LD_LIBRARY_PATH``.

    The device probe and the child servers then resolve the same runtime, so a
    zero-device probe reflects a genuinely unusable GPU rather than a probe that
    merely ran without the wheel libraries on its search path.
    """
    os.environ.update(cuda_runtime_env())


def _links_cuda_runtime(binary: Path, env: dict[str, str]) -> bool:
    """True when *binary* lists a CUDA runtime soname (a CUDA build), resolved or not."""
    return links_any(binary, env, _CUDA_SONAMES)


def assert_cuda_devices_usable(binary: Path, devices: list[FleetDevice], probe_output: str) -> None:
    """Fail loud when a CUDA build links a runtime it cannot initialize a GPU with.

    *devices* and *probe_output* are the engine's own ``--list-devices`` result.
    When the list is empty yet *binary* is a CUDA build and the host has an NVIDIA
    GPU, the runtime loaded but enumerated no device. The probe's own diagnostic is
    surfaced and the likely causes are listed (rather than asserting one), so
    placement does not silently fall to CPU.
    """
    if not sys.platform.startswith("linux"):
        return
    if devices:
        return
    env = {**os.environ, **cuda_runtime_env()}
    if not _links_cuda_runtime(binary, env):
        return
    if not model_cache.has_nvidia_gpu():
        return
    diagnostic = device_probe_diagnostic(probe_output)
    raise ProviderError(
        "The engine links the CUDA runtime and this host has an NVIDIA GPU, but it "
        "enumerated no CUDA-capable device, so GPU work would silently fall back to CPU.\n"
        f"The engine reported: {diagnostic}\n"
        "Likely causes: the installed CUDA runtime is newer than the GPU driver supports "
        "(check the driver's CUDA version with 'nvidia-smi' and match the "
        "nvidia-cuda-runtime-cu12 / nvidia-cublas-cu12 / nvidia-cuda-nvrtc-cu12 wheels to "
        "the engine's CUDA build, e.g. 12.4.x for a cu124 build, or update the driver); a "
        "restrictive CUDA_VISIBLE_DEVICES; or the runtime libraries missing from the path."
    )
