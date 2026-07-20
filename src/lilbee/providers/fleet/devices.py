"""Enumerate and pin GPUs using the llama-server binary's own device view.

The hazard this avoids: a device index from one API (Vulkan) is meaningless to
another (CUDA); the same ordinal can be a different physical card. So both
enumeration and pinning go through the binary's native backend index space,
obtained from ``llama-server --list-devices``. The Vulkan VRAM probe is only a
fallback when the binary can't enumerate. See docs/architecture.md.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.gpu_select import USABLE_VULKAN_TYPES, VkDeviceType

log = logging.getLogger(__name__)

_PROVIDER = "llama-server"
_LIST_DEVICES_TIMEOUT_S = 60.0
# How long to wait for a killed probe to be reaped before abandoning it: a child
# wedged in uninterruptible GPU-driver I/O ignores even SIGKILL.
_PROBE_KILL_WAIT_S = 5.0
_TOPO_TIMEOUT_S = 15.0
_GPU_LABEL_RE = re.compile(r"^GPU(\d+)$")
# nvidia-smi emits SGR escapes (e.g. an underlined header) even when stdout is
# not a tty; strip them or the header's GPU labels never match.
_ANSI_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")
# A topo-matrix header is 2+ leading GPU labels; a data row has exactly one. And
# a link needs at least two GPUs to exist between.
_TOPO_MIN_GPUS = 2
MIB = 1024 * 1024
# Per-backend visible-devices env vars (the probe inherits them; the children
# re-emit them, composed through any parent restriction).
_CUDA_VISIBLE_VAR = "CUDA_VISIBLE_DEVICES"
_CUDA_ORDER_VAR = "CUDA_DEVICE_ORDER"
_PCI_BUS_ID_ORDER = "PCI_BUS_ID"
_ROCR_VISIBLE_VAR = "ROCR_VISIBLE_DEVICES"
_HIP_VISIBLE_VAR = "HIP_VISIBLE_DEVICES"
_VK_VISIBLE_VAR = "GGML_VK_VISIBLE_DEVICES"
_ONEAPI_SELECTOR_VAR = "ONEAPI_DEVICE_SELECTOR"
_LEVEL_ZERO_PREFIX = "level_zero:"
# "  CUDA0: NVIDIA GeForce RTX 3090 (24268 MiB, 23500 MiB free)"
_DEVICE_RE = re.compile(
    r"^\s*([A-Za-z]+)(\d+):\s*(.+?)\s*\((\d+)\s*MiB(?:,\s*(\d+)\s*MiB\s*free)?\)\s*$"
)
# Pin priority when a build reports more than one GPU backend: a real GPU
# backend always wins over Vulkan, which wins over CPU.
_BACKEND_RANK = {"CUDA": 3, "ROCm": 3, "HIP": 3, "MTL": 3, "Metal": 3, "SYCL": 2, "Vulkan": 1}
# Backends whose memory is always the host's: Apple Silicon reports a working-set
# slice of system RAM, never a dedicated pool.
_UNIFIED_BACKENDS = frozenset({"MTL", "Metal"})


@dataclass(frozen=True)
class FleetDevice:
    """One GPU as the binary's backend enumerates it (native index space)."""

    backend: str
    index: int
    name: str
    total_bytes: int
    free_bytes: int
    # Whether this device's memory is the host's memory. An integrated GPU or an
    # Apple Silicon Mac has no dedicated VRAM, so its reported total is a slice
    # of the same RAM the OS and every other process is using, and placement
    # must stay inside the system budget rather than treating it as headroom.
    unified: bool = False


@dataclass(frozen=True)
class DeviceProbe:
    """The device probe's parsed devices plus its raw output for diagnostics."""

    devices: list[FleetDevice]
    output: str


def _parse_topo_matrix(topo_text: str) -> tuple[set[int], set[frozenset[int]]]:
    """GPU row indices and NVLink-joined pairs from ``nvidia-smi topo -m`` output.

    The matrix header row labels the GPU columns; each ``GPU<r>`` row lists the
    link type to each column (``NV#`` is NVLink; ``PIX``/``PHB``/``SYS`` are PCIe).
    """
    header_cols: list[int] = []
    gpu_rows: set[int] = set()
    pairs: set[frozenset[int]] = set()
    for line in _ANSI_SGR_RE.sub("", topo_text).splitlines():
        tokens = line.split()
        # Leading run of GPU-label tokens: the header is all labels (>=2), a data
        # row is one label ("GPU3") followed by link-type cells. split() strips the
        # header's leading whitespace, so this run length is what tells them apart.
        leading_labels: list[int] = []
        for token in tokens:
            match = _GPU_LABEL_RE.match(token)
            if match is None:
                break
            leading_labels.append(int(match.group(1)))
        if len(leading_labels) >= _TOPO_MIN_GPUS:
            header_cols = leading_labels
        elif len(leading_labels) == 1:
            row_idx = leading_labels[0]
            gpu_rows.add(row_idx)
            for col_idx, cell in zip(header_cols, tokens[1:], strict=False):
                if row_idx != col_idx and cell.startswith("NV"):
                    pairs.add(frozenset({row_idx, col_idx}))
    return gpu_rows, pairs


def host_lacks_nvlink() -> bool:
    """Whether this host's GPUs are joined only by PCIe (no NVLink anywhere).

    Tensor-splitting a large model across PCIe-only cards is all-reduce bound and
    much slower than over NVLink. Deliberately a host-level claim: the fleet's
    device indices live in the serving binary's backend index space, which does
    not map onto ``nvidia-smi``'s physical numbering under a visible-devices
    restriction (the very hazard this module exists to avoid), so per-pair
    verdicts against plan indices would be unreliable. Returns False (no claim)
    when the probe fails or reports fewer than two GPUs, so a non-NVIDIA or
    single-card host stays silent rather than warning wrongly.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "topo", "-m"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=_TOPO_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    gpu_rows, pairs = _parse_topo_matrix(proc.stdout)
    return len(gpu_rows) >= _TOPO_MIN_GPUS and not pairs


def _probe_env() -> dict[str, str]:
    """Env for the probe: stable PCI ordering so CUDA indices match what we pin.

    A preset ``CUDA_DEVICE_ORDER`` is respected; ``visible_env`` re-emits the same
    order var, so the probe and the spawned servers see one device ordering.
    """
    env = dict(os.environ)
    env.setdefault(_CUDA_ORDER_VAR, _PCI_BUS_ID_ORDER)
    return env


def probe_devices(binary: Path, *, timeout_s: float = _LIST_DEVICES_TIMEOUT_S) -> DeviceProbe:
    """Parse ``<binary> --list-devices``; empty devices when unavailable/unparseable.

    Filtered to a single GPU backend (the highest-ranked one present) so device
    indices are unambiguous when a build exposes several backends. A probe that
    does not respond within *timeout_s* raises a ``ProviderError`` naming the
    stuck probe: that is a wedged GPU driver, not a GPU-less host, and treating
    it as "no devices" would silently plan a CPU fleet on a GPU box.
    """
    try:
        output = _run_list_devices(binary, timeout_s)
    except (OSError, subprocess.SubprocessError):
        return DeviceProbe([], "")
    return DeviceProbe(_select_backend(_parse_devices(output)), output)


def _run_list_devices(binary: Path, timeout_s: float) -> str:
    """Run the probe in its own process group; kill the group and raise on timeout.

    ``subprocess.run``'s timeout path waits indefinitely for the killed child to
    be reaped, so a probe wedged in uninterruptible GPU-driver I/O would hang the
    caller forever; this bounds the reap and abandons an unkillable child.
    """
    proc = subprocess.Popen(  # noqa: S603 - binary is the resolved llama-server
        [str(binary), "--list-devices"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_probe_env(),
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_probe(proc)
        raise ProviderError(
            f"The GPU device probe ({binary.name} --list-devices) did not respond "
            f"within {timeout_s:.0f}s, so the engine cannot start. The GPU driver "
            "may be in a bad state: check that 'nvidia-smi' responds, and reboot "
            "the host if it hangs.",
            provider=_PROVIDER,
            kind=ProviderErrorKind.SERVER,
        ) from None
    return output or ""


def _kill_probe(proc: subprocess.Popen[str]) -> None:
    """SIGKILL the timed-out probe's process group; abandon it if it cannot be reaped."""
    if os.name == "posix":
        # start_new_session made the probe its own group leader.
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    else:  # pragma: no cover - Windows has no process groups to kill
        proc.kill()
    try:
        proc.communicate(timeout=_PROBE_KILL_WAIT_S)
    except subprocess.TimeoutExpired:
        log.warning(
            "The GPU device probe (pid %d) ignored SIGKILL and was abandoned; "
            "it is likely stuck in the GPU driver.",
            proc.pid,
        )


def _parse_devices(text: str) -> list[FleetDevice]:
    devices: list[FleetDevice] = []
    for line in text.splitlines():
        match = _DEVICE_RE.match(line)
        if match is None:
            continue
        backend, index, name, total_mib, free_mib = match.groups()
        total = int(total_mib) * MIB
        free = int(free_mib) * MIB if free_mib else total
        devices.append(
            FleetDevice(
                backend,
                int(index),
                name.strip(),
                total,
                free,
                unified=_is_unified(backend, name.strip()),
            )
        )
    return devices


# Mesa and friends expose CPU rasterizers through the Vulkan loader, and
# llama.cpp's Vulkan backend enumerates them exactly like a GPU: same
# "VulkanN: <name> (<total> MiB, <free> MiB free)" shape, with system RAM
# reported as VRAM. Planning against one is worse than having no GPU at all,
# because the "VRAM" looks enormous: a host with a real iGPU beside lavapipe
# can be planned as a two-GPU machine and tensor-split across a real adapter
# and a software renderer, which runs orders of magnitude slower than either
# CPU inference or the iGPU alone.
_SOFTWARE_RENDERER_MARKERS = ("llvmpipe", "lavapipe", "softpipe", "swiftshader")


def _is_software_renderer(device: FleetDevice) -> bool:
    """Whether *device* is a CPU rasterizer masquerading as a GPU.

    A name test, so it only recognizes the rasterizers it already knows, and a
    renamed or newly written one walks past it. It stays as the answer for hosts
    where the Vulkan loader can't be opened from this process and the device
    type is therefore unavailable; where the type is available,
    ``_is_unusable_vulkan`` decides and this never gets the chance to be wrong.
    """
    name = device.name.casefold()
    return any(marker in name for marker in _SOFTWARE_RENDERER_MARKERS)


def _vulkan_device_type(name: str) -> VkDeviceType | None:
    """The loader's type for the Vulkan adapter the engine printed as *name*.

    ``None`` when the loader can't be reached or reports no adapter by that
    name, which reads as "no opinion": the device is kept and assumed dedicated,
    preserving the behaviour of hosts that never had a type to consult.
    """
    from lilbee.providers.fleet.gpu_select import vulkan_device_types_by_name

    return vulkan_device_types_by_name().get(name)


def _is_unusable_vulkan(device: FleetDevice) -> bool:
    """Whether *device* is a Vulkan adapter ggml would not choose to run on.

    ggml's Vulkan backend builds its device pool from discrete and integrated
    adapters only, and falls back to the first non-CPU adapter when it finds
    neither. In a VM that fallback is a paravirtual adapter (VMware SVGA,
    VirtIO-GPU Venus, QXL), which reports guest RAM as VRAM and is typically
    compute-incomplete or fails at allocation. Planning a fleet onto one costs
    more than planning no GPU at all, since a non-empty device list also turns
    off the shared-RAM budget.
    """
    if device.backend != "Vulkan":
        return False
    device_type = _vulkan_device_type(device.name)
    return device_type is not None and device_type not in USABLE_VULKAN_TYPES


def _is_unified(backend: str, name: str) -> bool:
    """Whether the device *backend* printed as *name* shares its memory with the host.

    Metal is unified by construction on Apple Silicon: the figure it reports is
    ``recommendedMaxWorkingSetSize``, a slice of system RAM rather than a
    separate pool. For Vulkan the loader knows the device type, so the type is
    asked for rather than guessed; a size heuristic cannot work here, since a
    24 GB discrete card in a 32 GB host and an Apple GPU reporting two thirds of
    RAM are indistinguishable by proportion.

    CUDA, ROCm and SYCL print no type at all, and an AMD APU or a Jetson looks
    exactly like a discrete card there while reporting system RAM as its memory.
    Those fall back to a question about the machine rather than the device: a
    host whose Vulkan loader sees adapters but no discrete one has no discrete
    GPU for another backend to be enumerating.
    """
    if backend in _UNIFIED_BACKENDS:
        return True
    if backend == "Vulkan":
        return _vulkan_device_type(name) is VkDeviceType.INTEGRATED_GPU
    from lilbee.providers.fleet.gpu_select import host_has_no_discrete_gpu

    return host_has_no_discrete_gpu()


def _select_backend(devices: list[FleetDevice]) -> list[FleetDevice]:
    """Keep one GPU backend's devices: highest rank, then most memory.

    Returns a single backend so pinning is unambiguous: ``visible_env`` keys off
    one backend, and mixing index spaces is the very hazard this module avoids.

    CUDA, ROCm, HIP and Metal all rank alike, and a build that loads several
    backends (``ggml_backend_load_all`` does) makes the tie real. Breaking it on
    the backend's name meant a host with a 4090 beside an RX 6600 planned onto
    the AMD card because "ROCm" sorts after "CUDA", and the NVIDIA card idled
    with nothing said. Total memory decides instead; the name is only the last
    resort that keeps the choice deterministic.
    """
    ranked = [
        d
        for d in devices
        if d.backend in _BACKEND_RANK
        and not _is_software_renderer(d)
        and not _is_unusable_vulkan(d)
    ]
    if not ranked:
        return []
    by_backend: dict[str, list[FleetDevice]] = {}
    for device in ranked:
        by_backend.setdefault(device.backend, []).append(device)
    backend, chosen = max(by_backend.items(), key=_backend_preference)
    for other, group in by_backend.items():
        if other != backend:
            log.info(
                "Engine reports %d %s device(s) beside %d %s device(s); planning onto %s, "
                "which has more memory. Backends cannot be mixed: their device indexes "
                "name different cards.",
                len(group),
                other,
                len(chosen),
                backend,
                backend,
            )
    return chosen


def _backend_preference(item: tuple[str, list[FleetDevice]]) -> tuple[int, int, str]:
    backend, group = item
    return _BACKEND_RANK[backend], sum(d.total_bytes for d in group), backend


def _compose_visible(indices: list[int], parent_value: str | None) -> str:
    """Visible-devices value naming the same physical devices the probe saw.

    When the parent env already restricts the var, the probe's indices are
    relative to that comma-separated list (integer or UUID entries), so each
    index maps through it; the child's value then names the same physical
    devices instead of being re-interpreted as absolute.
    """
    if parent_value is None:
        return ",".join(str(i) for i in indices)
    entries = [entry.strip() for entry in parent_value.split(",") if entry.strip()]
    out: list[str] = []
    for i in indices:
        if i >= len(entries):
            # The probe enumerates devices under the parent restriction, so every
            # index must map into it. An out-of-range index is an invariant
            # violation; emitting a bare ``str(i)`` would pin an absolute integer
            # into a possibly UUID-namespaced list, silently selecting the wrong
            # GPU. Fail loudly instead.
            raise ValueError(
                f"device index {i} is outside the parent visible-devices list "
                f"{parent_value!r}; cannot compose a child pin without selecting the wrong GPU"
            )
        out.append(entries[i])
    return ",".join(out)


def visible_env(devices: tuple[FleetDevice, ...]) -> dict[str, str]:
    """Env that pins a child to *devices* via the right var for their backend.

    Indices are the backend-native ones from ``probe_devices``, composed through
    any parent visible-devices restriction so the child names the same physical
    devices the probe enumerated; no cross-API index translation occurs.
    """
    if not devices:
        return {}
    backend = devices[0].backend
    indices = [d.index for d in devices]
    if backend == "CUDA":
        return {
            _CUDA_VISIBLE_VAR: _compose_visible(indices, os.environ.get(_CUDA_VISIBLE_VAR)),
            _CUDA_ORDER_VAR: os.environ.get(_CUDA_ORDER_VAR, _PCI_BUS_ID_ORDER),
        }
    if backend in ("ROCm", "HIP"):
        return _amd_visible_env(indices)
    if backend == "Vulkan":
        # Deliberately not GGML_VK_VISIBLE_DEVICES. That variable indexes the raw
        # loader enumeration, while these indices come from the engine's own
        # filtered list, so the two disagree wherever ggml drops or merges a
        # device -- two ICDs for one card being the clear case. Setting it also
        # disables ggml's type filter, support check and dedup. Vulkan is pinned
        # with --device instead, in the same space the names were parsed from.
        return {}
    if backend == "SYCL":
        return {_ONEAPI_SELECTOR_VAR: _compose_sycl(indices, os.environ.get(_ONEAPI_SELECTOR_VAR))}
    return {}


def amd_visible_var() -> str:
    """The one AMD visibility var an index list may be written to.

    ``ROCR_VISIBLE_DEVICES`` and ``HIP_VISIBLE_DEVICES`` are applied sequentially
    by the runtime: ROCr filters first, then HIP re-indexes within the survivors.
    Writing the same indices to both therefore double-filters and selects the
    wrong cards, or none at all: ``1`` on a two-GPU box exposes physical GPU 1 as
    index 0 through ROCr, and HIP then asks for index 1 of a one-device list.

    So exactly one is ever written: the one the environment already restricts,
    or HIP when it restricts neither. Every caller writing an AMD pin asks here,
    since two callers each picking their own would put the pair back.
    """
    if _ROCR_VISIBLE_VAR in os.environ and _HIP_VISIBLE_VAR not in os.environ:
        return _ROCR_VISIBLE_VAR
    return _HIP_VISIBLE_VAR


def _amd_visible_env(indices: list[int]) -> dict[str, str]:
    """Pin an AMD ROCm/HIP child to the probe's *indices* with one visibility var.

    The probe enumerated a single index space already filtered by whichever var
    the parent set, so the chosen var is composed against that parent value and
    the other is left inherited untouched. The child inherits the parent env, so
    an unset override keeps any inherited sibling var in force.
    """
    var = amd_visible_var()
    return {var: _compose_visible(indices, os.environ.get(var))}


def _compose_sycl(indices: list[int], parent_value: str | None) -> str:
    """``ONEAPI_DEVICE_SELECTOR`` value naming the same devices the probe saw.

    A parent selector shaped ``level_zero:i,j`` makes the probe's indices
    relative to its post-colon list, so each index maps through that list like
    :func:`_compose_visible`; any other shape (or none) emits absolute indices.
    """
    if parent_value is not None and parent_value.startswith(_LEVEL_ZERO_PREFIX):
        parent_list = parent_value[len(_LEVEL_ZERO_PREFIX) :]
        return _LEVEL_ZERO_PREFIX + _compose_visible(indices, parent_list)
    return _LEVEL_ZERO_PREFIX + ",".join(str(i) for i in indices)
