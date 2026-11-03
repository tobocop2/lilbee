"""The OpenVINO adapter: upstream llama-server built on OpenVINO, serving chat on CPU or iGPU."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from lilbee.catalog.types import ModelFormat
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.adapters import ROLE_SPECS, RoleServerSpec, build_server_argv
from lilbee.providers.fleet.devices import DeviceProbe
from lilbee.providers.fleet.engine_manifest import pack_digest, pack_platforms
from lilbee.providers.fleet.engines.base import EngineName, LaunchRequest, RequestProfile
from lilbee.providers.fleet.engines.host_gate import PASS, PLATFORM_UNREAD, HostGate, npu_gate
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS, NpuVendor
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import estimate_memory
from lilbee.providers.fleet.proc import run_bounded
from lilbee.providers.gguf_meta import read_gguf_metadata
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

# The wheel version follows the llama.cpp rule: 1.<upstream build>.<wheel build>.
_VERSION = "1.12020.1"
_BACKEND = "openvino"
_TOOL = "llama-server"
_LICENCES = ["MIT", "Apache-2.0"]
_DEVICE_VAR = "GGML_OPENVINO_DEVICE"
_CPU_DEVICE = "CPU"
_GPU_DEVICE = "GPU"
_ENV = "/usr/bin/env"
_LIST_TIMEOUT_S = 60.0
_LIST_KILL_WAIT_S = 5.0
_DEVICE_LINE = re.compile(r"^\s*OPENVINO\d+:\s*" + _DEVICE_VAR + r"=(\w+)", re.MULTILINE)
_PIN_LAUNCH = ["{binary}", "--model", "{model}", "--host", "127.0.0.1", "--port", "{port}"]
_NPU_UNUSED = (
    "This computer has an Intel NPU. OpenVINO runs on the CPU or the integrated graphics, "
    "because no run on the NPU is recorded."
)
GPU_MISSING = (
    "OpenVINO cannot use the Intel graphics device: the Intel compute runtime is not "
    "installed on this computer. OpenVINO does not switch to the CPU by itself. "
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


def listed_devices(output: str) -> frozenset[str]:
    """The OpenVINO device names that ``llama-server --list-devices`` printed."""
    return frozenset(_DEVICE_LINE.findall(output))


class OpenvinoAdapter:
    """llama-server on OpenVINO behind the engine contract; it serves chat only.

    Requantized weights change vectors, so embed and rerank stay with llama.cpp.
    The NPU is held: no recorded run exists, so ``HARDWARE_PROOF`` lists it as not run.
    """

    name = EngineName.OPENVINO
    formats = frozenset({ModelFormat.GGUF})
    architectures: frozenset[str] = frozenset()
    served_roles = frozenset({WorkerRole.CHAT})
    memory_kind = MemoryKind.SIZED
    takes_device_names = False
    bundled = False

    def supports_architecture(self, architecture: str) -> bool:
        return True

    def host_gate(self, host: HostCaps) -> HostGate:
        """Linux on x86_64 only. A host with an Intel NPU passes and says the NPU stays unused."""
        if host.os is None or host.arch is None:
            return PLATFORM_UNREAD
        if (host.os, host.arch) != (HostOS.LINUX, HostArch.X86_64):
            return HostGate(False, f"OpenVINO is not offered on {host.os.value} {host.arch.value}.")
        npu = npu_gate(host, NpuVendor.INTEL)
        return HostGate(True, _NPU_UNUSED) if npu.ok and npu.known else PASS

    def binary(self) -> Path:
        from lilbee.catalog.runnable import session_host
        from lilbee.providers.fleet.engines.choice import store_tool

        tool = store_tool(self.pin_record(), _TOOL, session_host())
        if tool is None:
            raise ProviderError(
                "The OpenVINO engine is not installed. Run lilbee engine install openvino."
            )
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
            "sha256": pack_digest(self.name.value, _VERSION, host, _BACKEND),
            "tool": _TOOL,
            "launch": _PIN_LAUNCH,
            "platforms": pack_platforms(self.name.value, _VERSION, host, _BACKEND),
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
        on_cpu = request.target is RoleTarget.CPU
        if not on_cpu and _GPU_DEVICE not in self._listed(request.binary):
            raise ProviderError(GPU_MISSING)
        if on_cpu:
            request = replace(request, n_gpu_layers=0, fit_target_mib=())
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
        device = _CPU_DEVICE if on_cpu else _GPU_DEVICE
        return [_ENV, f"{_DEVICE_VAR}={device}", *argv]

    @staticmethod
    def _listed(binary: Path) -> frozenset[str]:
        output, _code = run_bounded(
            [str(binary), "--list-devices"],
            timeout_s=_LIST_TIMEOUT_S,
            kill_wait_s=_LIST_KILL_WAIT_S,
            merge_stderr=True,
        )
        return listed_devices(output)

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        if _GPU_DEVICE in self._listed(binary):
            return frozenset({RoleTarget.CPU, RoleTarget.GPU})
        return frozenset({RoleTarget.CPU})

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(devices=[], output="", spoke_protocol=False)

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

    def assert_loadable(self, repo: str, filename: str, token: str | Literal[False] | None) -> None:
        return None

    def runtime_env(self) -> Mapping[str, str]:
        return {}

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


OPENVINO = OpenvinoAdapter()
ADAPTER = OPENVINO
