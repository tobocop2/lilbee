"""Guard a ROCm engine build against AMD hosts it cannot actually serve.

ROCm has a wide silent-failure class (kernel and user-space version mismatch,
an unsupported gfx target, no permission on /dev/kfd) that ends either as a
quiet CPU fallback or as a rocBLAS abort mid-inference. The checks here read
the kernel driver and the bundled artifacts rather than ROCm tooling, because
ROCm being broken is the case they exist to catch.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.engine_diagnostics import device_probe_diagnostic, links_any
from lilbee.providers.fleet.gpu_select import PCIVendorID, discrete_gpu_from_vendor

if TYPE_CHECKING:
    from lilbee.providers.fleet.devices import FleetDevice

log = logging.getLogger(__name__)

# A ROCm build links these and none of the CUDA sonames.
_HIP_SONAMES: tuple[str, ...] = ("libamdhip64.so", "librocblas.so", "libhipblas.so")
# PCI vendor id for AMD, as sysfs reports it.
_AMD_PCI_VENDOR_ID = "0x1002"
# Backends whose devices run rocBLAS; ggml prints either name depending on version.
_AMD_BACKENDS = ("ROCm", "HIP")
# AMD's escape hatch: the runtime treats every device as the given gfx version.
_HSA_OVERRIDE_VAR = "HSA_OVERRIDE_GFX_VERSION"


def _links_hip_runtime(binary: Path, env: dict[str, str]) -> bool:
    """True when *binary* lists a HIP runtime soname (a ROCm build), resolved or not."""
    return links_any(binary, env, _HIP_SONAMES)


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
    return discrete_gpu_from_vendor(PCIVendorID.AMD) is True


def _gfx_name(target_version: int) -> str:
    """The gfx name for a KFD ``gfx_target_version``; minor and step print as hex."""
    major, rest = divmod(target_version, 10000)
    minor, step = divmod(rest, 100)
    return f"gfx{major}{minor:x}{step:x}"


def _host_amd_gfx_targets() -> set[str]:
    """The gfx targets of this host's AMD GPUs, from the driver's KFD topology.

    CPU nodes report a target version of 0. Empty when the topology is absent
    or unreadable, which is "no claim".
    """
    targets: set[str] = set()
    for props in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties"):
        try:
            text = props.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            name, _, value = line.partition(" ")
            if name == "gfx_target_version" and value.strip().isdigit() and int(value.strip()):
                targets.add(_gfx_name(int(value.strip())))
    return targets


def _bundled_rocblas_gfx_targets(binary: Path) -> set[str] | None:
    """The gfx targets covered by the rocBLAS Tensile masters bundled beside *binary*.

    rocBLAS aborts the process on a card it has no masters for, so the shipped
    files are the ground truth for which cards this build can run a GEMM on.
    None when no bundle sits beside the binary (a system-ROCm engine): that is
    "no claim", where an empty set would mean "supports nothing".
    """
    library = binary.parent / "rocblas" / "library"
    if not library.is_dir():
        return None
    return {
        f.name.removeprefix("TensileLibrary_lazy_").removesuffix(".dat")
        for f in library.glob("TensileLibrary_lazy_gfx*.dat")
    }


def _hsa_override_gfx() -> str | None:
    """The gfx target a user-set ``HSA_OVERRIDE_GFX_VERSION`` maps every device to."""
    raw = os.environ.get(_HSA_OVERRIDE_VAR, "")
    try:
        major, minor, step = (int(part) for part in raw.split("."))
    except ValueError:
        return None
    return _gfx_name(major * 10000 + minor * 100 + step)


def _rocm_support_facts(binary: Path) -> str:
    """What is known about shipped kernels and host gfx targets, as message text."""
    shipped = _bundled_rocblas_gfx_targets(binary)
    parts = []
    if shipped:
        parts.append(f"This build ships GPU kernels for: {', '.join(sorted(shipped))}.")
    if host := _host_amd_gfx_targets():
        parts.append(f"This host's AMD GPU targets: {', '.join(sorted(host))}.")
    return " " + " ".join(parts) if parts else ""


def _warn_if_override_uncovered(override: str, shipped: set[str]) -> None:
    """The user overrode explicitly; respect it, but say what will happen."""
    if override in shipped:
        return
    log.warning(
        "%s maps every AMD device to %s, but this build ships GPU kernels only "
        "for: %s. The engine will abort at the first matrix multiplication if a "
        "model runs on the GPU.",
        _HSA_OVERRIDE_VAR,
        override,
        ", ".join(sorted(shipped)),
    )


def _assert_rocblas_covers_enumerated_devices(binary: Path, devices: list[FleetDevice]) -> None:
    """Refuse a card the bundle ships no rocBLAS kernels for, before rocBLAS aborts.

    An enumerated device is no proof of support: the engine's device code and
    rocBLAS's kernels are built from different lists, and a card covered by the
    first but not the second initializes fine and then aborts the whole engine at
    the first batched matrix multiply. Support is read from the bundled Tensile
    masters rather than kept as a constant, so it cannot drift from what shipped.
    """
    if not any(d.backend in _AMD_BACKENDS for d in devices):
        return
    shipped = _bundled_rocblas_gfx_targets(binary)
    if shipped is None:
        return
    if (override := _hsa_override_gfx()) is not None:
        _warn_if_override_uncovered(override, shipped)
        return
    host = _host_amd_gfx_targets()
    if not host or host <= shipped:
        return
    if host & shipped:
        log.warning(
            "This host has AMD GPU(s) with target %s, which this build ships no GPU "
            "kernels for; the engine will abort if a model is placed on one. Restrict "
            "HIP_VISIBLE_DEVICES to the supported cards, or set %s if the card is a "
            "near miss of a shipped target (gfx1031 runs gfx1030 kernels with "
            "%s=10.3.0).",
            ", ".join(sorted(host - shipped)),
            _HSA_OVERRIDE_VAR,
            _HSA_OVERRIDE_VAR,
        )
        return
    raise ProviderError(
        f"This host's AMD GPU is {', '.join(sorted(host))}, but this engine build ships "
        f"GPU kernels only for: {', '.join(sorted(shipped))}. The engine would start and "
        "then abort at the first matrix multiplication, so it is refused up front.\n"
        f"If the card is a near miss of a shipped target, set {_HSA_OVERRIDE_VAR} to that "
        f"target's version (a gfx1031 card runs the gfx1030 kernels with "
        f"{_HSA_OVERRIDE_VAR}=10.3.0). Otherwise install lilbee's Vulkan build, which "
        "supports AMD cards ROCm does not."
    )


def assert_rocm_devices_usable(binary: Path, devices: list[FleetDevice], probe_output: str) -> None:
    """Fail loud when a ROCm build cannot serve the AMD hardware in front of it.

    *devices* and *probe_output* are the engine's own ``--list-devices`` result.
    An enumerated card is checked against the bundled rocBLAS kernels; an empty
    list on an AMD host means the runtime loaded and enumerated no device, which
    would otherwise silently fall back to CPU.
    """
    if not sys.platform.startswith("linux"):
        return
    _assert_rocblas_covers_enumerated_devices(binary, devices)
    if devices:
        return
    if not (_links_hip_runtime(binary, dict(os.environ)) and _amd_gpu_present()):
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
            "not support (check with 'rocminfo').%s The engine reported: %s",
            _rocm_support_facts(binary),
            device_probe_diagnostic(probe_output),
        )
        return
    raise ProviderError(
        "The engine links the ROCm/HIP runtime and this host has an AMD GPU, but it "
        "enumerated no device, so GPU work would silently fall back to CPU.\n"
        f"The engine reported: {device_probe_diagnostic(probe_output)}\n"
        "Likely causes: the ROCm user-space version does not match the amdgpu kernel "
        "driver; the GPU's gfx target is not supported by this ROCm build (check with "
        "'rocminfo'); no read/write permission on /dev/kfd (the user is usually added "
        "to the 'render' and 'video' groups); or a restrictive ROCR_VISIBLE_DEVICES or "
        f"HIP_VISIBLE_DEVICES.{_rocm_support_facts(binary)}"
    )
