"""Enumerate and pin GPUs using the llama-server binary's own device view.

The hazard this avoids: a device index from one API (Vulkan) is meaningless to
another (CUDA); the same ordinal can be a different physical card. So both
enumeration and pinning go through the binary's native backend index space,
obtained from ``llama-server --list-devices``. The Vulkan VRAM probe is only a
fallback when the binary can't enumerate. See docs/architecture.md.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.gpu_select import USABLE_VULKAN_TYPES, VkDeviceType
from lilbee.providers.fleet.proc import run_bounded

log = logging.getLogger(__name__)

_PROVIDER = "llama-server"
_LIST_DEVICES_TIMEOUT_S = 60.0
# How long to wait for a killed probe to be reaped before abandoning it: a child
# wedged in uninterruptible GPU-driver I/O ignores even SIGKILL.
_PROBE_KILL_WAIT_S = 5.0
# How much of the probe's own output to quote in a diagnostic. Enough to carry
# the driver's error line, short enough to stay a readable message.
_PROBE_TAIL_CHARS = 400
_TOPO_TIMEOUT_S = 15.0
_GPU_LABEL_RE = re.compile(r"^GPU(\d+)$")
# llama-server prints this before the device loop, so a run that lists no GPUs
# still prints it. Its absence means the binary never got as far as enumerating.
_DEVICE_LIST_HEADER = "Available devices:"
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
# ROCm's third numeric visibility variable, filtering exactly as the other two do.
_GPU_DEVICE_ORDINAL_VAR = "GPU_DEVICE_ORDINAL"
_VK_VISIBLE_VAR = "GGML_VK_VISIBLE_DEVICES"
# "  CUDA0: NVIDIA GeForce RTX 3090 (24268 MiB, 23500 MiB free)"
_DEVICE_RE = re.compile(
    r"^\s*([A-Za-z]+)(\d+):\s*(.+?)\s*\((\d+)\s*MiB(?:,\s*(\d+)\s*MiB\s*free)?\)\s*$"
)
# Pin priority when a build reports more than one GPU backend: a real GPU
# backend always wins over Vulkan, which wins over CPU.
# The engine's own name for the backend. Vendor-agnostic, so several rules key
# on it: a Vulkan device's type has to be asked of the loader, and its util
# source is chosen by the vendor in its device name rather than by the backend.
VULKAN_BACKEND = "Vulkan"
_BACKEND_RANK = {"CUDA": 3, "ROCm": 3, "HIP": 3, "MTL": 3, "Metal": 3, "SYCL": 2, VULKAN_BACKEND: 1}
# Backends whose memory is always the host's: Apple Silicon reports a working-set
# slice of system RAM, never a dedicated pool.
_UNIFIED_BACKENDS = frozenset({"MTL", "Metal"})
# Below this, a reported total is a BIOS carveout rather than a card's own pool.
# An APU hands out a fixed slice of system RAM as "VRAM", often a few hundred
# MiB, and planned as a dedicated device that size it refuses every role while
# the machine has the whole system's memory to share. No real discrete GPU worth
# serving from ships with less.
_DEDICATED_VRAM_FLOOR = 2 * 1024 * MIB


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
    # Whether this device came from the host's Vulkan loader rather than from the
    # engine's own listing. Its index is then a raw loader ordinal, which is a
    # different space from the one the engine names its devices in, so it can be
    # sized against but never pinned by.
    from_loader: bool = False


@dataclass(frozen=True)
class DeviceProbe:
    """The device probe's parsed devices plus its raw output for diagnostics."""

    devices: list[FleetDevice]
    output: str
    # Whether the engine answered --list-devices at all: exited cleanly and
    # printed the header it always prints. False means the binary does not speak
    # this protocol (a build predating the flag prints usage text and exits
    # non-zero), so its silence about devices is not a statement that there are
    # none. Defaults False so a probe that never ran is never mistaken for one
    # that ran and found nothing.
    spoke_protocol: bool = False
    # Whether the engine listed GPU devices and every one was rejected. Distinct
    # from a host that simply has none: the engine will still pick one of those
    # devices at launch unless it is told not to.
    refused_all: bool = False


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
        stdout, _ = run_bounded(
            ["nvidia-smi", "topo", "-m"],
            timeout_s=_TOPO_TIMEOUT_S,
            kill_wait_s=_PROBE_KILL_WAIT_S,
            label="nvidia-smi topo",
        )
    except (OSError, subprocess.SubprocessError):
        return False
    gpu_rows, pairs = _parse_topo_matrix(stdout)
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
        output, returncode = _run_list_devices(binary, timeout_s)
    except (OSError, subprocess.SubprocessError) as exc:
        # Silently returning an empty probe here made an unrunnable binary look
        # exactly like a host with no GPU, and the fleet planned for CPU with
        # nothing said. The reason is the whole diagnosis: a wrong architecture,
        # a missing loader, a permission denial.
        log.warning(
            "Could not run the GPU device probe (%s --list-devices): %s. Continuing "
            "as though this host has no GPU; check that the engine binary is "
            "executable and built for this machine.",
            binary.name,
            exc,
        )
        return DeviceProbe([], "")
    parsed = _parse_devices(output)
    selected = _select_backend(parsed)
    offered = [d for d in parsed if d.backend in _BACKEND_RANK]
    answered = _DEVICE_LIST_HEADER in output
    spoke = returncode == 0 and answered
    if not spoke and answered:
        # It knew the flag and started answering, then died. Blaming the flag
        # here sent the reader looking for the wrong engine build, when what
        # they have is a crash partway through enumeration.
        log.warning(
            "%s --list-devices printed its device header then crashed (exit %d), so the "
            "device list may be incomplete. This is usually a GPU driver or ICD fault "
            "during enumeration. The probe reported: %s",
            binary.name,
            returncode,
            _probe_tail(output),
        )
    elif not spoke:
        log.warning(
            "%s --list-devices exited %d without printing its device header, so it "
            "does not appear to support the flag. Falling back to the host's Vulkan "
            "loader to find GPUs; set %s if this is not the engine you meant to use.",
            binary.name,
            returncode,
            "LILBEE_ENGINE_DIR",
        )
    return DeviceProbe(
        selected, output, spoke_protocol=spoke, refused_all=bool(offered) and not selected
    )


def _run_list_devices(binary: Path, timeout_s: float) -> tuple[str, int]:
    """Run the probe with a bounded reap; raise on timeout.

    A probe wedged in uninterruptible GPU-driver I/O would otherwise hang the
    caller forever, since ``subprocess.run``'s timeout waits unbounded for the
    reap; ``run_bounded`` abandons an unkillable child after a short wait.

    The probe holds a device context and writes no state file, so nothing can reap
    it later by record. It is the one caller that opts into the lifetime binding,
    where the kernel offers one, and it is killed on the way out of every abort,
    not just the timeout.
    """
    try:
        return run_bounded(
            [str(binary), "--list-devices"],
            timeout_s=timeout_s,
            kill_wait_s=_PROBE_KILL_WAIT_S,
            env=_probe_env(),
            merge_stderr=True,
            label=f"{binary.name} --list-devices",
            bind_lifetime=True,
        )
    except subprocess.TimeoutExpired as exc:
        # Whatever the probe managed to print before it wedged says more than any
        # fixed advice can, and the fixed advice named one vendor's tool at a host
        # that may have neither that vendor nor that tool.
        raise ProviderError(
            f"The GPU device probe ({binary.name} --list-devices) did not respond "
            f"within {timeout_s:.0f}s, so the engine cannot start. The GPU driver is "
            "most likely wedged; check that your vendor's tool responds (nvidia-smi, "
            "rocm-smi, xpu-smi) and reboot the host if it hangs.\n"
            f"The probe reported: {_probe_tail(_decoded_output(exc.output))}",
            provider=_PROVIDER,
            kind=ProviderErrorKind.SERVER,
        ) from None


def _decoded_output(output: object) -> str:
    """Partial child output from a timeout, which arrives as bytes even under text mode."""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output if isinstance(output, str) else ""


def _probe_tail(output: str) -> str:
    """The tail of what the probe printed, for a message that has to stay readable."""
    text = output.strip()
    return text[-_PROBE_TAIL_CHARS:] if text else "(nothing)"


def _parse_devices(text: str) -> list[FleetDevice]:
    devices: list[FleetDevice] = []
    # Sampled at most once per parse, and only when a line actually needs it:
    # free memory is live, so it is read fresh here rather than cached, and the
    # loader must not be opened once per device line to answer the same question.
    loader_free: dict[str, int] | None = None
    for line in text.splitlines():
        match = _DEVICE_RE.match(line)
        if match is None:
            continue
        backend, index, name, total_mib, free_mib = match.groups()
        total = int(total_mib) * MIB
        if total == 0:
            # No memory is not a small GPU, it is one that cannot hold a model:
            # a driver listing an adapter before its memory is queryable. Kept, it
            # is the smallest card in the fleet and collapses every budget sized
            # against the smallest, while the non-empty list switches off the
            # shared-memory budget a host with no usable GPU depends on.
            log.warning(
                "Ignoring GPU %s%s (%s): it reports no memory, so nothing can be "
                "placed on it. Check the GPU driver if this device should be usable.",
                backend,
                index,
                name.strip(),
            )
            continue
        if free_mib:
            free = int(free_mib) * MIB
        else:
            if loader_free is None:
                loader_free = _loader_free_bytes(backend)
            free = loader_free.get(name.strip(), total)
        devices.append(
            FleetDevice(
                backend,
                int(index),
                name.strip(),
                total,
                free,
                unified=_is_unified(backend, name.strip()) or total < _DEDICATED_VRAM_FLOOR,
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


def _loader_free_bytes(backend: str) -> dict[str, int]:
    """Live free memory per device name, for a listing that printed no free figure.

    ggml omits the figure when the driver has no ``VK_EXT_memory_budget``, and
    treating the omission as "all of it" is how a desktop holding gigabytes of
    compositor and browser VRAM was planned as an empty card. The loader exposes
    that extension to this process even when the engine build cannot use it, so
    it is asked directly; a name it cannot speak for keeps the heap size.

    Empty for any other backend: the Vulkan loader knows nothing about the
    devices a CUDA or ROCm listing names.
    """
    if backend != VULKAN_BACKEND:
        return {}
    from lilbee.providers.fleet.gpu_select import vulkan_free_bytes_by_name

    return vulkan_free_bytes_by_name()


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

    Only a positive claim counts. VIRTUAL_GPU and CPU are the loader naming what
    the adapter is; OTHER is it declining to, and refusing on a shrug took the
    GPU away from real hardware whose driver simply does not classify itself.
    """
    if device.backend != VULKAN_BACKEND:
        return False
    device_type = _vulkan_device_type(device.name)
    if device_type is None or device_type in USABLE_VULKAN_TYPES:
        return False
    # OTHER is the loader shrugging, not an accusation. The spec's own wording is
    # "does not match any other available types", which a driver reaches for when
    # it cannot classify itself, and some real adapters do. Refusing on it took a
    # working GPU away from a machine the engine had already listed one for.
    # VIRTUAL_GPU and CPU are positive claims and keep their veto.
    return device_type is not VkDeviceType.OTHER


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
    if backend == VULKAN_BACKEND:
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


def _backend_preference(item: tuple[str, list[FleetDevice]]) -> tuple[int, int, int, str]:
    """Sort key for choosing one backend's devices: rank, dedicated bytes, size.

    Dedicated bytes come before raw size because the discrete backends all tie at
    the same rank, and a shared-heap carveout reports a total that is host RAM
    the host budget already counts. Left on raw size, an APU advertising a large
    carveout beat a discrete card, which was then discarded and left idle while
    the plan double-promised memory it did not have.
    """
    backend, group = item
    dedicated = sum(d.total_bytes for d in group if not d.unified)
    return _BACKEND_RANK[backend], dedicated, sum(d.total_bytes for d in group), backend


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
    if backend == VULKAN_BACKEND:
        # Deliberately not GGML_VK_VISIBLE_DEVICES. That variable indexes the raw
        # loader enumeration, while these indices come from the engine's own
        # filtered list, so the two disagree wherever ggml drops or merges a
        # device -- two ICDs for one card being the clear case. Setting it also
        # disables ggml's type filter, support check and dedup. Vulkan is pinned
        # with --device instead, in the same space the names were parsed from.
        return {}
    if backend == "SYCL":
        # Deliberately no ONEAPI_DEVICE_SELECTOR. It is a selector grammar over a
        # backend runtime, not the index space --list-devices numbers, so a
        # composed level_zero ordinal can name a different physical card than the
        # one the probe enumerated. SYCL pins by --device instead, in the space
        # the indices were read from. An inherited parent selector still applies:
        # the engine enumerated behind it, so its names are already relative to it.
        return {}
    return {}


def amd_visible_var() -> str:
    """The one AMD visibility var an index list may be written to.

    ``ROCR_VISIBLE_DEVICES`` and ``HIP_VISIBLE_DEVICES`` are applied sequentially:
    ROCr filters first, then HIP re-indexes within the survivors. Writing the same
    indices to both double-filters and selects the wrong cards, or none at all.
    ``GPU_DEVICE_ORDINAL`` is the third and filters the same way, so writing HIP
    over an ordinal mask both overrides it and re-exposes cards it had hidden.

    So exactly one is ever written: whichever the environment already restricts,
    in the runtime's precedence (HIP, then the ordinal, then ROCr), or HIP when
    nothing restricts. An empty value means "no devices" rather than "this is the
    variable in use", so it does not claim precedence. Every caller writing an AMD
    pin asks here; two callers each picking their own would put the pair back.
    """
    for name in (_HIP_VISIBLE_VAR, _GPU_DEVICE_ORDINAL_VAR, _ROCR_VISIBLE_VAR):
        if os.environ.get(name, "").strip():
            return name
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
