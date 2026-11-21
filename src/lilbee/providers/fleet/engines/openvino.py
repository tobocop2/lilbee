"""The OpenVINO adapter: upstream llama-server on OpenVINO, serving chat on CPU, GPU or NPU."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from lilbee.catalog.hf_client import HfToken
from lilbee.catalog.types import ModelFormat
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.adapters import ROLE_SPECS, RoleServerSpec, build_server_argv
from lilbee.providers.fleet.devices import DeviceProbe, require_clean_probe_exit
from lilbee.providers.fleet.engine_manifest import pack_digest, pack_platforms
from lilbee.providers.fleet.engines.base import (
    EngineName,
    LaunchRequest,
    RequestProfile,
    ServerProcess,
)
from lilbee.providers.fleet.engines.host_gate import PASS, PLATFORM_UNREAD, HostGate, npu_gate
from lilbee.providers.fleet.engines.llama import _WHEEL_VERSION as _LLAMA_PACK_VERSION
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS, NpuVendor
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import estimate_memory
from lilbee.providers.fleet.proc import run_bounded
from lilbee.providers.gguf_meta import read_gguf_metadata
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

# The pack is the openvino cell of the llama.cpp pack build: the llama.cpp wheel at the version
# of the other llama.cpp packs, built from the same llama.cpp tag, with the build tag "openvino".
_VERSION = _LLAMA_PACK_VERSION
_PACK_ENGINE = EngineName.LLAMA.value
_BACKEND = "openvino"
_TOOL = "llama-server"
# The License-Expression of the pack: the llama.cpp terms of the Linux packs, the OpenVINO runtime
# under its Intel distribution licence, oneTBB and oneDNN (Apache-2.0), hwloc (BSD-3-Clause) and
# the OpenCL loader (BSD-2-Clause). The Intel term is not on the permissive list, so an install
# refuses the pack until that licence is ruled on.
_RUNTIME_TERM = "LicenseRef-Intel-OpenVINO-Distribution-License"
_LICENCES = [
    f"MIT AND {_RUNTIME_TERM} AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause "
    "AND ISC AND MIT-0 AND Unlicense"
]
_DEVICE_VAR = "GGML_OPENVINO_DEVICE"
_CPU_DEVICE = "CPU"
_GPU_DEVICE = "GPU"
_INTEL = "intel"
_ENV = "/usr/bin/env"
_LIST_TIMEOUT_S = 60.0
_LIST_KILL_WAIT_S = 5.0
_DEVICE_LINE = re.compile(
    r"^\s*OPENVINO\d+:\s*" + _DEVICE_VAR + r"=([\w.]+)(?:\s*\(selected\))?\s*-\s*([^\n]*)",
    re.MULTILINE,
)
_PIN_LAUNCH = ["{binary}", "--model", "{model}", "--host", "127.0.0.1", "--port", "{port}"]
_NPU_DEVICE = "NPU"
_NPU_CTX = 512
"""The context of an NPU process; upstream keeps it small ("-c 512") to avoid load failures."""
_INTEL_PCI_VENDOR = 0x8086
_NO_INTEL_DEVICE = (
    "OpenVINO is offered on computers with an Intel GPU or an Intel NPU, "
    "and this computer has neither."
)
_GRAPHICS_UNREAD = "The graphics cards of this computer could not be read."
NPU_MISSING = (
    "OpenVINO runs on the Intel NPU only when the computer has one and its driver is installed, "
    "and this computer has no NPU that OpenVINO can use. "
    "OpenVINO does not switch to another device by itself. "
    "Install the Intel NPU driver, or place this role on the CPU or the GPU."
)
GPU_MISSING = (
    "OpenVINO runs on Intel graphics only, and this computer has no Intel graphics device "
    "that OpenVINO can use. A missing Intel compute runtime also causes this. "
    "OpenVINO does not switch to the CPU by itself. "
    "Install the Intel compute runtime, or place this role on the CPU."
)

_PROFILE = RequestProfile(
    embed_fields={},
    stream_fields={"return_progress": True},
    tokenize_path="/tokenize",
    detokenize_path="/detokenize",
    apply_template_path="/apply-template",
    label="llama-server (OpenVINO)",
    context_overflow=re.compile("exceed_context_size|context size", re.IGNORECASE),
    batch_overflow=re.compile("too large to process"),
    unreadable_embedding_hint="OpenVINO serves chat only.",
    reasoning_field="reasoning_content",
)


@dataclass(frozen=True)
class ListedDevice:
    """One line of ``llama-server --list-devices``: the full name and its description."""

    name: str
    description: str


def parse_devices(output: str) -> tuple[ListedDevice, ...]:
    """The devices that ``llama-server --list-devices`` printed, in list order."""
    return tuple(ListedDevice(name, rest) for name, rest in _DEVICE_LINE.findall(output))


def _gpu_ordinal(name: str) -> int:
    """The number in ``GPU.1``; plain ``GPU`` is the alias of ``GPU.0``."""
    _, _, tail = name.partition(".")
    return int(tail) if tail.isdigit() else 0


def has_npu(devices: tuple[ListedDevice, ...]) -> bool:
    """Whether the list names the NPU device, which OpenVINO lists as plain ``NPU``."""
    return any(d.name == _NPU_DEVICE for d in devices)


def pick_gpu(devices: tuple[ListedDevice, ...]) -> str | None:
    """The one GPU name a launch passes, or None when no listed GPU is an Intel GPU.

    The OpenVINO GPU plugin runs on Intel GPUs only, integrated or discrete, although
    the list also shows other cards. Intel GPUs are numbered ``GPU.X``, and an
    integrated one is always ``GPU.0``, so the lowest number wins.
    """
    gpus = [
        d
        for d in devices
        if (d.name == _GPU_DEVICE or d.name.startswith(_GPU_DEVICE + "."))
        and d.description.lower().startswith(_INTEL)
    ]
    if not gpus:
        return None
    return min(gpus, key=lambda d: _gpu_ordinal(d.name)).name


def _intel_device_gate(host: HostCaps) -> HostGate:
    """Offer the engine only on a host with an Intel GPU or an Intel NPU.

    An empty list of display controllers means "cannot tell", so it never refuses.
    """
    if _INTEL_PCI_VENDOR in host.display_vendors:
        return PASS
    npu = npu_gate(host, NpuVendor.INTEL)
    if npu.ok and npu.known:
        return PASS
    if not host.display_vendors:
        return HostGate(True, _GRAPHICS_UNREAD, known=False)
    if not npu.known:
        return npu
    return HostGate(False, _NO_INTEL_DEVICE)


class OpenvinoAdapter:
    """llama-server on OpenVINO behind the engine contract; it serves chat only.

    Requantized weights change vectors, so embed and rerank stay with llama.cpp.
    The NPU runs one slot and a small context. No recorded run exists, so
    ``HARDWARE_PROOF`` lists it as not run and ``auto`` never picks the engine.
    """

    name = EngineName.OPENVINO
    formats = frozenset({ModelFormat.GGUF})
    architectures: frozenset[str] = frozenset()
    served_roles = frozenset({WorkerRole.CHAT})
    memory_kind = MemoryKind.SIZED
    takes_device_names = False
    concatenates_pooling = False
    bundled = False
    server_process = ServerProcess(name=_TOOL)

    def supports_architecture(self, architecture: str) -> bool:
        return True

    def host_gate(self, host: HostCaps) -> HostGate:
        """Linux on x86_64 only. A host with an Intel NPU passes and says the NPU stays unused."""
        if host.os is None or host.arch is None:
            return PLATFORM_UNREAD
        if (host.os, host.arch) != (HostOS.LINUX, HostArch.X86_64):
            return HostGate(False, f"OpenVINO is not offered on {host.os.value} {host.arch.value}.")
        return _intel_device_gate(host)

    def binary(self) -> Path:
        from lilbee.catalog.runnable import session_host
        from lilbee.providers.fleet.engines.choice import install_advice, store_tool

        tool = store_tool(self.pin_record(), _TOOL, session_host())
        if tool is None:
            raise ProviderError(f"The OpenVINO engine is not installed. {install_advice(self)}")
        return tool

    def identity(self, binary: Path) -> str:
        return f"{self.name.value} {_VERSION} {binary}"

    def pin_record(self) -> Mapping[str, Any]:
        from lilbee.catalog.runnable import session_host

        host = session_host()
        return {
            "engine": self.name.value,
            "version": _VERSION,
            "backend": _BACKEND,
            "sha256": pack_digest(_PACK_ENGINE, _VERSION, host, _BACKEND),
            "tool": _TOOL,
            "launch": _PIN_LAUNCH,
            "platforms": pack_platforms(_PACK_ENGINE, _VERSION, host, _BACKEND),
            "memory": {"sizing": None, "rule": None},
            "licenses": _LICENCES,
        }

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec:
        if role not in self.served_roles:
            raise ProviderError(f"Engine {self.name.value} does not serve {role}.")
        return ROLE_SPECS[role]

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]:
        """The llama command behind ``env``, which sets the one device of the process.

        OpenVINO aborts when the named device is missing and never falls back to the
        CPU, so a GPU target is checked against the device list before it starts.
        """
        device, request = self._device_for(request)
        argv = build_server_argv(
            binary=request.binary,
            spec=spec,
            model_path=request.model_path,
            devices=(),
            n_gpu_layers=request.n_gpu_layers,
            slots=request.slots,
            ctx_per_slot=request.ctx_per_slot,
            mmproj=request.mmproj,
            flash_attn=request.flash_attn,
            cache_type=request.cache_type,
            batch_size=request.batch_size,
            no_mmap=request.no_mmap,
        )
        return [_ENV, f"{_DEVICE_VAR}={device}", *argv]

    def _device_for(self, request: LaunchRequest) -> tuple[str, LaunchRequest]:
        """The device name of the process, and the request with the limits of that device."""
        if request.target is RoleTarget.CPU:
            return _CPU_DEVICE, replace(request, n_gpu_layers=0, fit_target_mib=())
        listed = self._listed(request.binary)
        if request.target is RoleTarget.NPU:
            if not has_npu(listed):
                raise ProviderError(NPU_MISSING)
            capped = min(request.ctx_per_slot, _NPU_CTX)
            return _NPU_DEVICE, replace(request, slots=1, ctx_per_slot=capped, fit_target_mib=())
        gpu = pick_gpu(listed)
        if gpu is None:
            raise ProviderError(GPU_MISSING)
        return gpu, request

    @staticmethod
    def _listed(binary: Path) -> tuple[ListedDevice, ...]:
        output, code = run_bounded(
            [str(binary), "--list-devices"],
            timeout_s=_LIST_TIMEOUT_S,
            kill_wait_s=_LIST_KILL_WAIT_S,
            merge_stderr=True,
        )
        require_clean_probe_exit(binary, code, output)
        return parse_devices(output)

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        listed = self._listed(binary)
        targets = {RoleTarget.CPU}
        if pick_gpu(listed) is not None:
            targets.add(RoleTarget.GPU)
        if has_npu(listed):
            targets.add(RoleTarget.NPU)
        return frozenset(targets)

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(devices=[], output="", spoke_protocol=False, has_listing=False)

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        return estimate_memory(ModelFormat.GGUF, request)

    def check_readback(self, binary: Path) -> bool:
        return False

    def supports_fit_target(self, binary: Path) -> bool:
        return False

    def supports_shared_kv(self, binary: Path) -> bool:
        return False

    def model_meta(self, path: Path) -> dict[str, str] | None:
        return read_gguf_metadata(path)

    def assert_loadable(self, repo: str, filename: str, token: HfToken) -> None:
        return None

    def runtime_env(self) -> Mapping[str, str]:
        return {}

    def target_env(self, target: RoleTarget, cards: int) -> Mapping[str, str]:
        return {}

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


OPENVINO = OpenvinoAdapter()
ADAPTER = OPENVINO
