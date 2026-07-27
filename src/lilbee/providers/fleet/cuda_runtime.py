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
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from lilbee.providers import model_cache
from lilbee.providers.base import ProviderError

if TYPE_CHECKING:
    from lilbee.providers.fleet.devices import FleetDevice

# Subpackages the NVIDIA runtime wheels install under the ``nvidia`` namespace.
# The distribution name carries the CUDA major (nvidia-cuda-runtime-cu12,
# -cu13) but the import path does not, so this resolves whichever major is
# installed and needs no version of its own. Only the packaging extra is
# major-specific; see the ``cuda12`` extra in pyproject.toml.
_CUDA_WHEEL_IMPORTS: tuple[str, ...] = (
    "nvidia.cuda_runtime",
    "nvidia.cublas",
    "nvidia.cuda_nvrtc",
)
# The CUDA runtime sonames, matched by library name with the major read out of
# the version suffix rather than pinned into the string. A build linking
# libcudart.so.13 is as much a CUDA build as one linking .so.12, and pinning the
# major meant the whole guard returned early on the newer one: a cu13 engine that
# could not initialize a device fell to CPU in exactly the silence this exists to
# break. Mirrors ollama's cudaRuntimeSORegex (discover/llama_server.go).
_CUDA_SONAME_RE = re.compile(r"\blib(?:cudart|cublas|nvrtc)\.so\.(\d+)")
# The HIP equivalents. A ROCm build links these and none of the CUDA sonames, so
# the CUDA guard above never fired for it.
_HIP_SONAMES: tuple[str, ...] = ("libamdhip64.so", "librocblas.so", "libhipblas.so")
# Substrings that mark a CUDA init failure in the engine's --list-devices output.
_CUDA_ERROR_MARKERS: tuple[str, ...] = ("error", "fail", "no cuda")
_LDD_TIMEOUT_S = 10
# PCI vendor id for AMD, as sysfs reports it.
log = logging.getLogger(__name__)

_AMD_PCI_VENDOR_ID = "0x1002"
# How much of the probe output to quote when no specific error line is found.
_DIAGNOSTIC_TAIL_CHARS = 300


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


def _ships_its_own_cuda_runtime(binary: Path) -> bool:
    """Whether the CUDA runtime sits in the same directory as *binary*.

    The bundled engine ships its libraries beside itself, and a wheel directory
    on the search path can only shadow them with a different build.
    """
    parent = binary.parent
    return any(_CUDA_SONAME_RE.search(entry.name) for entry in _dir_entries(parent))


def _dir_entries(directory: Path) -> list[Path]:
    """Entries of *directory*, empty when it cannot be read."""
    try:
        return list(directory.iterdir())
    except OSError:
        return []


def cuda_runtime_env(binary: Path | None = None) -> dict[str, str]:
    """``LD_LIBRARY_PATH`` for running *binary*, or empty when there is nothing to add.

    Ordering, which matters more than it looks: the binary's own directory, then
    any CUDA-runtime wheel directories, then whatever the caller already had.
    ``$ORIGIN`` lands in ``DT_RUNPATH``, which the loader searches *after*
    ``LD_LIBRARY_PATH``, so a wheel directory in front silently replaces the
    libraries the engine ships beside itself. On a host that merely has torch
    installed, that swapped the bundled engine's CUDA runtime for torch's.

    Wheel directories are added only for a binary that actually links CUDA and
    does not already carry its own runtime. A Vulkan or CPU build has no use for
    them, and putting them on its path only gives an unrelated install a way to
    interfere. Without a *binary* to reason about, the wheel directories are
    returned as before.

    Empty off Linux, where neither the wheels nor ``LD_LIBRARY_PATH`` apply.
    """
    if not sys.platform.startswith("linux"):
        return {}
    parts: list[str] = []
    if binary is not None:
        parts.append(str(binary.parent))
        wants_wheels = _links_cuda_runtime(binary, dict(os.environ)) and not (
            _ships_its_own_cuda_runtime(binary)
        )
    else:
        wants_wheels = True
    dirs = _cuda_wheel_lib_dirs() if wants_wheels else []
    if not parts and not dirs:
        return {}
    parts += [str(d) for d in dirs]
    # Drop existing entries that are already wheel dirs so calling this on every
    # reload pass (apply_cuda_runtime_env) is idempotent instead of accumulating
    # duplicate copies that get baked into each spawned server's environment.
    wheel_dirs = set(parts)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts.extend(entry for entry in existing.split(os.pathsep) if entry and entry not in wheel_dirs)
    return {"LD_LIBRARY_PATH": os.pathsep.join(parts)}


def apply_cuda_runtime_env(binary: Path | None = None) -> None:
    """Put the CUDA-runtime wheel libs on this process's ``LD_LIBRARY_PATH``.

    The device probe and the child servers then resolve the same runtime, so a
    zero-device probe reflects a genuinely unusable GPU rather than a probe that
    merely ran without the wheel libraries on its search path.
    """
    os.environ.update(cuda_runtime_env(binary))


def _ldd_output(binary: Path, env: dict[str, str]) -> str | None:
    """``ldd`` stdout for *binary* under *env*; None when ldd can't run on it."""
    ldd = shutil.which("ldd")
    if ldd is None:
        return None
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
        # Not an ELF, a static binary, or a timeout: nothing to inspect.
        return None
    return proc.stdout


def _linked_cuda_major(ldd_output: str) -> int | None:
    """The CUDA runtime major *ldd_output* links, or ``None`` when it links none."""
    match = _CUDA_SONAME_RE.search(ldd_output)
    return int(match.group(1)) if match else None


def _links_cuda_runtime(binary: Path, env: dict[str, str]) -> bool:
    """True when *binary* lists a CUDA runtime soname (a CUDA build), resolved or not."""
    out = _ldd_output(binary, env)
    return out is not None and _linked_cuda_major(out) is not None


def _device_probe_diagnostic(probe_output: str) -> str:
    """The probe's CUDA error line, or a short tail of its output."""
    out = probe_output.strip()
    for line in out.splitlines():
        lowered = line.lower()
        if "cuda" in lowered and any(marker in lowered for marker in _CUDA_ERROR_MARKERS):
            return line.strip()
    return out[-_DIAGNOSTIC_TAIL_CHARS:] if out else "(the engine's device probe printed nothing)"


def _links_hip_runtime(binary: Path, env: dict[str, str]) -> bool:
    """True when *binary* lists a HIP runtime soname (a ROCm build), resolved or not."""
    out = _ldd_output(binary, env)
    if out is None:
        return False
    return any(soname in out for soname in _HIP_SONAMES)


def _amd_gpu_present() -> bool:
    """Whether the kernel exposes an AMD GPU, without needing ROCm to work.

    Read from sysfs rather than from amd-smi or rocm-smi: those ship with ROCm,
    and the failure this guards is precisely ROCm being installed wrong, so a
    tool-based check would report "no GPU" for the case it exists to catch.
    """
    if not Path("/dev/kfd").exists():
        return False
    for vendor in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            if vendor.read_text().strip().lower() == _AMD_PCI_VENDOR_ID:
                return True
        except OSError:
            continue
    return False


def _amd_discrete_gpu_proven() -> bool:
    """Whether a discrete AMD card is positively known to be present.

    Positive evidence only. The sysfs checks that find an AMD GPU cannot tell a
    discrete card from an APU, and the difference decides between refusing to
    start and merely running slower, so the question is put to the Vulkan loader,
    which reports the device type. An unreachable loader proves nothing and
    answers no, which keeps the softer path.
    """
    from lilbee.providers.fleet.gpu_select import PCIVendorID, discrete_gpu_from_vendor

    return discrete_gpu_from_vendor(PCIVendorID.AMD) is True


def assert_gpu_devices_usable(binary: Path, devices: list[FleetDevice], probe_output: str) -> None:
    """Fail loud when a GPU build cannot initialize any device on a host that has one.

    Dispatches on what the binary actually links, so a ROCm build gets the same
    treatment a CUDA build has always had. ROCm has a wide silent-failure class
    (kernel and user-space version mismatch, an unsupported gfx target, no
    permission on /dev/kfd) and every one of them previously ended as a quiet
    fall back to CPU.
    """
    assert_cuda_devices_usable(binary, devices, probe_output)
    if not sys.platform.startswith("linux") or devices:
        return
    env = {**os.environ, **cuda_runtime_env(binary)}
    if not (_links_hip_runtime(binary, env) and _amd_gpu_present()):
        return
    if not _amd_discrete_gpu_proven():
        # An APU is an AMD GPU by every check above: amdgpu exposes /dev/kfd for
        # integrated parts too, and the iGPU carries vendor 0x1002. But AMD's
        # population of GPUs ROCm legitimately does not support is large, and an
        # unsupported gfx target is the normal case for an APU rather than a
        # misconfiguration. Failing here would stop the engine on a laptop where
        # CPU serving worked, which is worse than the slow fallback this guard
        # exists to catch, so say so and let it start.
        log.warning(
            "The engine links the ROCm/HIP runtime and this host has an AMD GPU, but it "
            "enumerated no device, so GPU work will fall back to CPU. No discrete AMD card "
            "was found, so this is most likely an APU whose gfx target this ROCm build does "
            "not support (check with 'rocminfo'). The engine reported: %s",
            _device_probe_diagnostic(probe_output),
        )
        return
    raise ProviderError(
        "The engine links the ROCm/HIP runtime and this host has an AMD GPU, but it "
        "enumerated no device, so GPU work would silently fall back to CPU.\n"
        f"The engine reported: {_device_probe_diagnostic(probe_output)}\n"
        "Likely causes: the ROCm user-space version does not match the amdgpu kernel "
        "driver; the GPU's gfx target is not supported by this ROCm build (check with "
        "'rocminfo'); no read/write permission on /dev/kfd (the user is usually added "
        "to the 'render' and 'video' groups); or a restrictive ROCR_VISIBLE_DEVICES or "
        "HIP_VISIBLE_DEVICES."
    )


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
    env = {**os.environ, **cuda_runtime_env(binary)}
    if not _links_cuda_runtime(binary, env):
        return
    if not model_cache.has_nvidia_gpu():
        return
    diagnostic = _device_probe_diagnostic(probe_output)
    raise ProviderError(
        "The engine links the CUDA runtime and this host has an NVIDIA GPU, but it "
        "enumerated no CUDA-capable device, so GPU work would silently fall back to CPU.\n"
        f"The engine reported: {diagnostic}\n"
        "Likely causes: MIG is enabled on the card, whose parent device answers as an "
        "NVIDIA GPU while CUDA enumerates only its instances (list them with "
        "'nvidia-smi -L' and name one in CUDA_VISIBLE_DEVICES by its MIG- UUID); the "
        "installed CUDA runtime is newer than the GPU driver supports (check the driver's "
        "CUDA version with 'nvidia-smi' and match the nvidia-cuda-runtime / nvidia-cublas / "
        "nvidia-cuda-nvrtc wheels to the engine's CUDA build, or update the driver); a "
        "restrictive CUDA_VISIBLE_DEVICES; or the runtime libraries missing from the path."
    )
